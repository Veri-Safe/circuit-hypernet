"""Native Query-only HyperNet generation and addressed residual SAE interventions."""
from __future__ import annotations

from math import ceil, prod
import sys

from shared import ROOT, HERE, LOG, OUT, BLOCKS, CHECKPOINT, WIDTH, FEATURES, sha
sys.path[:0] = [str(ROOT / "s5_lean_circuit/src"), str(ROOT / "s2_hypernet/src"),
               str(ROOT / "s5_lean_circuit/scripts")]
import torch
from s5_lean_circuit.circuit.model_io import load_tokenizer
from s5_lean_circuit.circuit.transcoder import RectangularReLUTranscoder
from s5_lean_circuit.composition_loco import load_eval_bank
from s5_lean_circuit.composition_loco_config import load_config
from s5_lean_circuit.composition_loco_train import _load_hypernet
from s5_lean_circuit.protocol import extract_first_tactic


def load(protocol, dictionaries=False):
    assert torch.cuda.is_available()
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    config = load_config(ROOT / protocol["config_path"])
    assert config.config_sha256 == protocol["config_semantic_sha256"]
    assert config.route == config.model["condition_source"] == "query_only" and config.heldout_task == "T_nested_or"
    core = _load_hypernet(config, torch.device("cuda"))
    payload = torch.load(ROOT / protocol["checkpoint_path"], map_location="cpu", mmap=True, weights_only=False)
    for key, expected in {"format":"s5_composition_loco_checkpoint_v1", "route":"query_only",
        "protocol_id":config.protocol_id, "fold_id":"holdout_T_nested_or", "heldout_task":"T_nested_or",
        "global_step":2000,"kind":"model_only"}.items():
        assert payload[key] == expected, key
    core.load_trainable_state_dict(payload["model_state_dict"])
    del payload
    core.to("cuda").eval().requires_grad_(False)
    pg = core.parameter_generator.model
    assert (core.condition_source == "query_only" and core.fusion_mode == "append" and core.circuit_token_count == 0)
    assert (len(pg.blocks), pg.seq_n, pg.seq_h, pg.d_model) == (12, 8, 34, WIDTH)
    assert core.controller.scale_multiplier == 1.0 and core.controller.target_scope == "all"
    tokenizer = load_tokenizer(ROOT / protocol["tokenizer_path"], expected_sha256=protocol["tokenizer_sha256"])
    banks = {label:load_eval_bank(ROOT / b["path"]) for label,b in protocol["banks"].items()}
    models, scales = {}, {}
    if dictionaries:
        import json
        locked = json.loads((LOG / "dictionary_manifest.json").read_text())
        assert locked["complete"] and locked["checkpoint_sha256"] == CHECKPOINT
        for b in BLOCKS:
            item = locked["blocks"][str(b)]
            assert sha(ROOT / item["path"]) == item["sha256"]
            x = torch.load(ROOT / item["path"], map_location="cpu", weights_only=False)
            assert x["identity"]["checkpoint_sha256"] == CHECKPOINT and x["block"] == b
            assert x["config"] == protocol["dictionary"] and x["site_rms"] == item["site_rms"]
            models[b] = RectangularReLUTranscoder.from_state_dict(x["state_dict"]).cuda().eval().requires_grad_(False)
            scales[b] = float(item["site_rms"])
    return core, tokenizer, banks, models, scales


def slot_map(core):
    gen = core.parameter_generator; model = gen.model; tokenizer = gen.tokenizer
    parts = tokenizer.rank // model.dim_accumulation
    rows, group = [], 0
    for key, shape in tokenizer.shape_state_dict.items():
        groups = ceil(prod(shape) / (tokenizer.rank * tokenizer.token_size[1]))
        for token in range(groups):
            for subrank in range(parts):
                for seq, layer in enumerate(gen.target_layer_indices):
                    rows.append({"flat_slot":seq*model.seq_h+group*parts+subrank,"pg_layer_slot":seq,
                        "target_base_layer":layer,"module_factor":key,"token_group_within_factor":token,"rank_group":subrank})
            group += 1
    rows.sort(key=lambda x:x["flat_slot"])
    assert len(rows) == 272 and [r["flat_slot"] for r in rows] == list(range(272))
    return rows


def question_input(q, banks):
    bank, index = banks[q["bank"]], q["index"]
    assert bank.records[index].task_type == q["task"]
    assert bytes(bank.payload["event_key"][index]).hex() == q["event_key"]
    assert int(bank.payload["query_lengths"][index]) == q["query_length"]
    assert extract_first_tactic(bank.records[index].output_text) == q["gold"]
    return bank.payload["query_ids"][index:index+1,:q["query_length"]].cuda().long()


def generate(core, ids):
    with torch.autocast("cuda", dtype=torch.bfloat16):
        return core.generate_lora(ids, torch.ones_like(ids, dtype=torch.bool), None, circuit_present=None, gate_multiplier=0.5)


def canonical(value):
    assert value.shape == (1,8,34,WIDTH)
    return value.reshape(272, WIDTH)


def native_capture(core, ids):
    values, handles = {}, []
    for b in BLOCKS:
        def hook(_m,_a,out,b=b):
            assert b not in values
            canonical(out); values[b] = out.detach().clone()
        handles.append(core.parameter_generator.model.blocks[b].register_forward_hook(hook))
    try:
        state = generate(core, ids)
    finally:
        for h in handles:h.remove()
    assert set(values) == set(BLOCKS)
    return state, values


def predict(core, tokenizer, ids, state, q):
    values = ids.clone(); start = values.shape[1]; stop = "max_new_tokens"
    with core.controller.activate(state):
        for _ in range(50):
            if values.shape[1] >= 128:
                stop = "context_limit";break
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logits = core.base_logits(values)[:,-1,:]
            assert bool(torch.isfinite(logits).all())
            values = torch.cat((values,logits.argmax(-1,keepdim=True)),dim=1)
            full = tokenizer.decode(values[0].tolist())
            marker = next((m for m in ("proof complete","no goals") if m in full),None)
            if marker:
                stop = marker.replace(" ","_");break
    tokens = values[0,start:].tolist(); text = tokenizer.decode(tokens); first = extract_first_tactic(text)
    return {"generated_token_ids":tokens,"generated_text":text,"first_tactic":first,
            "correct":first==q["gold"],"stop_reason":stop}


def coefficients(value, block, models, scales, features):
    with torch.autocast("cuda", enabled=False):
        m = models[block]
        return torch.relu(torch.nn.functional.linear(canonical(value).float()/scales[block],
            m.W_enc[features].float(), m.b_enc[features].float()))


def edit(core, ids, group, native, models, scales, noop=False):
    handles, calls = [], []
    for s in group["sites"]:
        b, features = s["block"], s["feature_ids"]
        z = coefficients(native[b], b, models, scales, features)
        alpha = 1.0 if noop else group["alpha"]
        with torch.autocast("cuda", enabled=False):
            delta = -(1.0-alpha)*scales[b]*(z @ models[b].W_dec[features].float())
        def hook(_m,_a,out,b=b,delta=delta):
            calls.append(b)
            value = (canonical(out).float()+delta).reshape_as(out).to(out.dtype)
            assert bool(torch.isfinite(value).all())
            if noop:assert torch.equal(value,out)
            return value
        handles.append(core.parameter_generator.model.blocks[b].register_forward_hook(hook))
    try:state = generate(core,ids)
    finally:
        for h in handles:h.remove()
    assert calls == [s["block"] for s in group["sites"]]
    return state


def tactic_tokens(tokenizer, reference, q):
    assert reference["correct"] and reference["first_tactic"] == q["gold"]
    for n in range(1,len(reference["generated_token_ids"])+1):
        tokens = reference["generated_token_ids"][:n]
        if extract_first_tactic(tokenizer.decode(tokens)) == q["gold"]:return tokens
    raise ValueError("Native correct tactic has no matching token prefix")


def gradient_effects(core, ids, tokens, models, scales, check=False):
    captures, perturbations, handles = {}, {}, []
    betas = {b:torch.zeros((),device="cuda",dtype=torch.float32,requires_grad=True) for b in BLOCKS}
    for b in BLOCKS:
        def hook(_m,_a,out,b=b):
            value = out.detach().clone().requires_grad_(True) if b == BLOCKS[0] else out
            captures[b] = value
            if not check:return value
            with torch.no_grad(),torch.autocast("cuda",enabled=False):
                z = coefficients(value.detach(),b,models,scales,list(range(FEATURES)))
                f = int(z.sum(0).argmax()); delta = -scales[b]*(z[:,f:f+1]@models[b].W_dec[f:f+1].float())
                perturbations[b] = delta
            return (value.float()+betas[b]*delta.reshape_as(value)).to(value.dtype)
        handles.append(core.parameter_generator.model.blocks[b].register_forward_hook(hook))
    try:
        state = generate(core,ids)
        target = torch.tensor(tokens,device="cuda",dtype=torch.long)
        full = torch.cat((ids,target[:-1][None,:]),dim=1)
        assert full.shape[1] <= 128
        with core.controller.activate(state):
            with torch.autocast("cuda",dtype=torch.bfloat16):logits = core.base_logits(full)
            selected = logits[0,ids.shape[1]-1:ids.shape[1]-1+len(tokens)].float()
            loss = torch.nn.functional.cross_entropy(selected,target)
            assert torch.isfinite(loss)
            requested = [captures[b] for b in BLOCKS]+([betas[b] for b in BLOCKS] if check else [])
            gradients = torch.autograd.grad(loss, requested)
        effects, traces, norm = [], [], 0.0
        with torch.no_grad(),torch.autocast("cuda",enabled=False):
            for i,b in enumerate(BLOCKS):
                g = canonical(gradients[i]).float()
                z = coefficients(captures[b].detach(),b,models,scales,list(range(FEATURES)))
                effect = -scales[b]*(z*(g@models[b].W_dec.float().T)).sum(0)
                assert effect.shape == (FEATURES,) and bool(torch.isfinite(effect).all())
                effects.append(effect.cpu()); norm += float(g.square().sum())
                if check:
                    predicted = (g*perturbations[b]).sum(); actual = gradients[len(BLOCKS)+i].float()
                    assert torch.allclose(predicted,actual,rtol=1e-4,atol=1e-7),(b,float(predicted),float(actual))
                    traces.append({"block":b,"predicted":float(predicted),"direct":float(actual),
                                   "absolute_error":float((predicted-actual).abs())})
        assert not any(p.grad is not None for p in core.parameters())
        detached = {k:v.detach() for k,v in state.items()} if check else None
        return torch.stack(effects).numpy(),float(loss.detach()),norm,detached,traces
    finally:
        for h in handles:h.remove()
