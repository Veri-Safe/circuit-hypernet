"""Geometry helpers reused verbatim from M12; task identity is explicit."""
from contextlib import contextmanager
from math import ceil, prod, sqrt
import hashlib
import json
import torch
from .context import ROOT, SOURCE, target_identity
from s5_lean_circuit.protocol import extract_first_tactic
KINDS = ("self_attn_n", "self_attn_hw", "cross_attn", "feed_forward", "residual")
SLOT_COUNT = 272


def slot_map(generator) -> list[dict]:
    model = generator.model
    tokenizer = generator.tokenizer
    parts = tokenizer.rank // model.dim_accumulation
    group = 0
    rows = []
    for key, shape in tokenizer.shape_state_dict.items():
        groups = ceil(prod(shape) / (tokenizer.rank * tokenizer.token_size[1]))
        for token in range(groups):
            for subrank in range(parts):
                for sequence, base_layer in enumerate(generator.target_layer_indices):
                    rows.append({"flat_slot": sequence * model.seq_h + group * parts + subrank,
                                 "pg_layer_slot": sequence, "target_base_layer": base_layer,
                                 "module_factor": key, "token_group_within_factor": token,
                                 "rank_group": subrank})
            group += 1
    rows.sort(key=lambda row: row["flat_slot"])
    if (model.seq_n != 8 or model.seq_h != 34 or len(rows) != 272
            or [row["flat_slot"] for row in rows] != list(range(272))
            or group * parts != model.seq_h):
        raise ValueError("R03 all-272 slot-to-LoRA mapping differs")
    return rows

@contextmanager
def noop_hooks(core):
    pg = core.parameter_generator.model
    modules = []
    for block in pg.blocks:
        modules.extend((block.self_attn_n, block.self_attn_hw,
                        block.cross_attn, block.feed_forward, block))
    modules.append(pg.norm_final)
    calls = [0] * len(modules)
    handles = []
    for index, module in enumerate(modules):
        def hook(_module, _inputs, output, index=index):
            if not isinstance(output, torch.Tensor) or not torch.isfinite(output).all():
                raise ValueError("nonfinite or malformed PG no-op hook output")
            calls[index] += 1
            return output.clone()
        handles.append(module.register_forward_hook(hook))
    try:
        yield calls
    finally:
        for handle in handles:
            handle.remove()

@torch.inference_mode()
def generate(core, ids, vector, present: bool, gate: float):
    with torch.autocast("cuda", dtype=torch.bfloat16):
        return core.generate_lora(ids, torch.ones_like(ids, dtype=torch.bool), vector,
                                  circuit_present=torch.tensor([present], device="cuda", dtype=torch.bool),
                                  gate_multiplier=gate)

def module_at(pg, site: str):
    """The same five PG output sites as lc026_pg_whole_sae_mediation."""
    block_name, kind = site.split(".")
    index = int(block_name.removeprefix("block"))
    if block_name != f"block{index}" or not 0 <= index < 12 or kind not in KINDS:
        raise ValueError(f"unknown R03 PG site: {site}")
    return pg.blocks[index] if kind == "residual" else getattr(pg.blocks[index], kind)

def canonical(site: str, output: torch.Tensor) -> torch.Tensor:
    kind = site.split(".", 1)[1]
    if kind == "self_attn_n":
        if output.shape != (34, 8, 1024):
            raise ValueError(f"PG site shape differs: {site}: {tuple(output.shape)}")
        return output.permute(1, 0, 2).reshape(SLOT_COUNT, 1024)
    if kind == "self_attn_hw":
        expected = (8, 34, 1024)
    elif kind == "cross_attn":
        expected = (1, SLOT_COUNT, 1024)
    else:
        expected = (1, 8, 34, 1024)
    if output.shape != expected:
        raise ValueError(f"PG site shape differs: {site}: {tuple(output.shape)}")
    return output.reshape(SLOT_COUNT, 1024)

def module_names(state: dict[str, torch.Tensor]) -> list[str]:
    names = sorted(key.removesuffix(".lora_A") for key in state if key.endswith(".lora_A"))
    if len(names) != 32 or any(name + ".lora_B" not in state for name in names):
        raise ValueError("R03 effective LoRA target module inventory differs")
    return names

def effective_inner(a: dict[str, torch.Tensor], b: dict[str, torch.Tensor], module: str) -> float:
    # <scale_a B_a A_a, scale_b B_b A_b> without materializing a large Delta-W.
    aa, ba = a[module + ".lora_A"][0].double(), a[module + ".lora_B"][0].double()
    ab, bb = b[module + ".lora_A"][0].double(), b[module + ".lora_B"][0].double()
    gram_b = ba.T @ bb
    gram_a = ab @ aa.T
    return float(torch.trace(gram_b @ gram_a).item()
                 * float(a["scale"].item()) * float(b["scale"].item()))

def effective_norms(state: dict[str, torch.Tensor]) -> dict[str, float]:
    return {name: sqrt(max(0.0, effective_inner(state, state, name)))
            for name in module_names(state)}

def delta_lora(a: dict[str, torch.Tensor], b: dict[str, torch.Tensor]) -> dict[str, float]:
    if module_names(a) != module_names(b):
        raise ValueError("paired effective LoRA module names differ")
    result = {}
    for name in module_names(a):
        squared = (effective_inner(a, a, name) + effective_inner(b, b, name)
                   - 2 * effective_inner(a, b, name))
        result[name] = sqrt(max(0.0, squared))
    return result


@torch.inference_mode()
def fixed_reference_inputs(core, ids, names):
    """One frozen no-LoRA Base prefill per question; identical x for all states."""
    key = tuple(ids[0].tolist())
    old = getattr(core, "_mechanism_fixed_x", None)
    if old is not None and old[0] == key: return old[1], old[2]
    if core.controller._state is not None:
        raise ValueError("reference Base prefill must not have an active LoRA")
    inputs, handles = {}, []
    for name in names:
        def hook(_module, args, name=name):
            x = args[0]
            if name in inputs or x.ndim != 3 or x.shape[0] != 1:
                raise ValueError("fixed-input LoRA probe layout differs")
            inputs[name] = x[0,-1].detach().float().clone()
        handles.append(core.base_model.get_submodule(name).register_forward_pre_hook(hook))
    try:
        with torch.autocast("cuda", dtype=torch.bfloat16): core.base_logits(ids)
    finally:
        for handle in handles: handle.remove()
    if set(inputs) != set(names): raise ValueError("fixed-input module coverage differs")
    h = hashlib.sha256()
    for name in names:
        h.update(name.encode());h.update(inputs[name].cpu().numpy().tobytes())
    digest = h.hexdigest();core._mechanism_fixed_x = (key, inputs, digest)
    return inputs, digest


@torch.inference_mode()
def replay_with_wx(core, tokenizer, ids, state, target):
    names = module_names(state)
    inputs, reference_sha = fixed_reference_inputs(core, ids, names)
    wx = {}
    with torch.autocast("cuda", enabled=False):
        for name in names:
            x = inputs[name];a = state[name + ".lora_A"][0].float();b = state[name + ".lora_B"][0].float()
            if x.numel() != a.shape[1] or b.shape[1] != a.shape[0]: raise ValueError("fixed DeltaWx layout differs")
            wx[name] = ((x @ a.T) @ b.T * float(state["scale"].item())).detach().cpu()
    values = ids.clone();start = values.shape[1];stop_reason = "max_new_tokens"
    with core.controller.activate(state):
        for _ in range(target["generation"]["max_new_tokens"]):
            if values.shape[1] >= target["generation"]["n_ctx"]:
                stop_reason = "context_limit";break
            with torch.autocast("cuda", dtype=torch.bfloat16): logits = core.base_logits(values)[:,-1,:]
            if not bool(torch.isfinite(logits).all()): raise ValueError("nonfinite greedy logits")
            values = torch.cat((values,logits.argmax(-1,keepdim=True)),dim=1)
            full = tokenizer.decode(values[0].tolist())
            marker = next((m for m in ("proof complete","no goals") if m in full),None)
            if marker: stop_reason = marker.replace(" ","_");break
    tokens = values[0,start:].tolist();text = tokenizer.decode(tokens)
    return {"generated_token_ids":tokens,"generated_text":text,"first_tactic":extract_first_tactic(text),
            "stop_reason":stop_reason,"fixed_wx_reference_sha256":reference_sha},wx
