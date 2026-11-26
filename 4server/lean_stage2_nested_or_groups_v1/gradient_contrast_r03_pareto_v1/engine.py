"""The unchanged R03 inference route and explicit native-contribution edits."""
from __future__ import annotations

from contextlib import contextmanager
import json
import sys

from shared import ROOT, LOG, SOURCE, BLOCKS, CHECKPOINT

sys.path[:0] = [str(ROOT / "s5_lean_circuit/src"), str(ROOT / "s2_hypernet/src"),
               str(ROOT / "s5_lean_circuit/scripts"), str(ROOT / "temp_s")]
import torch
from s5_lean_circuit.composition_loco import load_eval_bank
from s5_lean_circuit.pg_transcoder_adapter import load_adapter
from s5_lean_circuit.protocol import extract_first_tactic
from lc026_nested_or_r03.common import generate, replay_with_wx, module_at, canonical, slot_map
from lc026_nested_or_r03.conditions import load_sae
from lc026_nested_or_r03.functional import coefficients, decoded_change


def load(protocol):
    assert torch.cuda.is_available()
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    source = json.loads((SOURCE / "manifest.json").read_text())
    target = json.loads((SOURCE / "target.json").read_text())
    adapter = load_adapter(target, root=ROOT, device=torch.device("cuda"))
    assert adapter.identity["checkpoint_sha256"] == CHECKPOINT and adapter.gate_multiplier == 0.5
    assert adapter.hypernet.condition_source == "circuit" and adapter.hypernet.fusion_mode == "append"
    assert adapter.hypernet.circuit_token_count == 1 and adapter.hypernet.controller.scale_multiplier == 1.0
    assert adapter.hypernet.controller.target_scope == "all"
    assert not any(p.requires_grad for p in adapter.hypernet.parameters())
    models, scales = {}, {}
    for block in BLOCKS:
        model, rms, sites, digest, _, _ = load_sae(block, source, SOURCE / "manifest.json", target)
        assert digest == protocol["dictionary"][str(block)]["sha256"]
        models[block] = model.requires_grad_(False)
        scales.update(dict(zip(sites, rms)))
    banks = {name: load_eval_bank(ROOT / item["path"]) for name, item in protocol["banks"].items()}
    assert [s["flat_slot"] for s in slot_map(adapter.hypernet.parameter_generator)] == list(range(272))
    return adapter, models, scales, banks, target


def question_input(q, banks):
    bank, index = banks[q["bank"]], q["index"]
    assert bank.records[index].task_type == q["task"]
    assert bytes(bank.payload["event_key"][index]).hex() == q["event_key"]
    assert int(bank.payload["query_lengths"][index]) == q["query_length"]
    ids = bank.payload["query_ids"][index:index+1, :q["query_length"]].cuda().long()
    vector = bank.payload["vectors"][index:index+1].cuda()
    assert vector.shape == (1, 397) and bool(torch.isfinite(vector).all())
    return ids, vector


def native_capture(adapter, ids, vector):
    captures, handles = {}, []
    for block in BLOCKS:
        site = f"block{block}.residual"
        def hook(_module, _args, output, site=site):
            assert site not in captures
            captures[site] = output.detach().clone()
        handles.append(module_at(adapter.hypernet.parameter_generator.model, site).register_forward_hook(hook))
    try:
        state = generate(adapter.hypernet, ids, vector, True, 0.5)
    finally:
        for handle in handles:
            handle.remove()
    assert len(captures) == len(BLOCKS)
    return state, captures


def edit(adapter, ids, vector, group, native, models, scales, *, noop=False):
    handles, calls = [], []
    for item in group["sites"]:
        site, features = item["site"], item["feature_ids"]
        z = coefficients(native[site], site, models, scales, features)
        alpha = 1.0 if noop else group["alpha"]
        def hook(_module, _args, output, site=site, features=features, z=z, alpha=alpha):
            calls.append(site)
            result = decoded_change(output, site, models, scales, features, -(1.0-alpha)*z, None)
            assert bool(torch.isfinite(result).all())
            if noop:
                assert torch.equal(output, result)
            return result
        handles.append(module_at(adapter.hypernet.parameter_generator.model, site).register_forward_hook(hook))
    try:
        state = generate(adapter.hypernet, ids, vector, True, 0.5)
    finally:
        for handle in handles:
            handle.remove()
    assert calls == [x["site"] for x in group["sites"]]
    return state


def predict(adapter, ids, state, target, q):
    prediction, _ = replay_with_wx(adapter.hypernet, adapter.tokenizer, ids, state, target)
    prediction["correct"] = prediction["first_tactic"] == q["gold"]
    return prediction


def tactic_tokens(adapter, reference, q):
    assert reference["correct"] and reference["first_tactic"] == q["gold"]
    tokens = reference["generated_token_ids"]
    for length in range(1, len(tokens)+1):
        if extract_first_tactic(adapter.tokenizer.decode(tokens[:length])) == q["gold"]:
            return tokens[:length]
    raise ValueError("Native-correct first tactic has no matching token prefix")


def gradient_effects(adapter, ids, vector, tokens, models, scales, *, check=False):
    """True autograd derivative, with a zero-strength scalar chain-rule check."""
    captures, handles = {}, []
    beta = torch.zeros((), device="cuda", dtype=torch.float32, requires_grad=True)
    perturbation = {}
    for block in BLOCKS:
        site = f"block{block}.residual"
        def hook(_module, _args, output, site=site, block=block):
            value = output.detach().clone().requires_grad_(True) if block == BLOCKS[0] else output
            captures[site] = value
            if block == 5 and check:
                with torch.no_grad(), torch.autocast("cuda", enabled=False):
                    z = coefficients(value.detach(), site, models, scales, list(range(8192)))
                    feature = int(z.sum(0).argmax())
                    delta = -(z[:, feature:feature+1] @ models[block].W_dec[feature:feature+1].float()) * scales[site]
                    perturbation["delta"] = delta
                return (value.float() + beta*delta.reshape_as(value)).to(value.dtype)
            return value
        handles.append(module_at(adapter.hypernet.parameter_generator.model, site).register_forward_hook(hook))
    try:
        with torch.enable_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            state = adapter.hypernet.generate_lora(ids, torch.ones_like(ids, dtype=torch.bool), vector,
                circuit_present=torch.ones(1, device="cuda", dtype=torch.bool), gate_multiplier=0.5)
        target_tokens = torch.tensor(tokens, device="cuda", dtype=torch.long)
        full = torch.cat((ids, target_tokens[:-1][None, :]), dim=1)
        assert full.shape[1] <= 128
        with adapter.hypernet.controller.activate(state):
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logits = adapter.hypernet.base_logits(full)
            selected = logits[0, ids.shape[1]-1:ids.shape[1]-1+len(tokens)].float()
            assert selected.shape[0] == len(tokens)
            loss = torch.nn.functional.cross_entropy(selected, target_tokens)
            assert bool(torch.isfinite(loss))
            requested = [captures[f"block{b}.residual"] for b in BLOCKS] + ([beta] if check else [])
            gradients = torch.autograd.grad(loss, requested)
        effects, norm, check_trace = [], 0.0, None
        with torch.no_grad(), torch.autocast("cuda", enabled=False):
            for block, gradient in zip(BLOCKS, gradients):
                site = f"block{block}.residual"
                v = captures[site].detach()
                g = canonical(site, gradient).float()
                z = coefficients(v, site, models, scales, list(range(8192)))
                projection = g @ models[block].W_dec.float().T
                effect = -(z*projection).sum(0) * scales[site]
                assert effect.shape == (8192,) and bool(torch.isfinite(effect).all())
                effects.append(effect.cpu())
                norm += float(g.square().sum())
            if check:
                predicted = (canonical("block5.residual", gradients[1]).float()*perturbation["delta"]).sum()
                actual = gradients[-1].float()
                assert torch.allclose(predicted, actual, rtol=1e-4, atol=1e-7), (predicted.item(), actual.item())
                check_trace = {"predicted_scalar_derivative": float(predicted),
                               "direct_autograd_derivative": float(actual),
                               "absolute_error": float((predicted-actual).abs())}
        assert norm >= 0 and not any(p.grad is not None for p in adapter.hypernet.parameters())
        detached_state = {k: v.detach() for k, v in state.items()} if check else None
        return torch.stack(effects).numpy(), float(loss.detach()), norm, detached_state, check_trace
    finally:
        for handle in handles:
            handle.remove()
