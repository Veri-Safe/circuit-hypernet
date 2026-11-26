#!/usr/bin/env python3
"""Trace local feature-to-feature PG graphs ending at effective LoRA updates.

This is a per-question local-Jacobian graph, not an exact transcoder replacement.
Every operation output is an independent source for the next operation; this
cuts indirect paths when estimating direct edges. SAE errors, affine offsets,
pruned incoming mass and unexpanded frontier nodes are explicit. All outputs
and decoding remain the frozen model's actual outputs. No training occurs.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import gzip
import heapq
import json
import os
from pathlib import Path
import sys
from time import perf_counter

import numpy as np
import torch
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[2]
for directory in (ROOT / "s5_lean_circuit/src", ROOT / "s2_hypernet/src",
                  ROOT / "s5_lean_circuit/scripts", ROOT / "temp_s"):
    sys.path.insert(0, str(directory))

from .context import ATOMIC_BANK, COMPOSITION_BANK
from .sae import (
    TASKS, KINDS, SLOTS, check_question, source_questions, sha, sha_bytes,
    canonical_json, write_json,
)
from .conditions import load_sae, condition_input
from .common import canonical, module_at
from .common import target_identity, slot_map
from .common import generate
from .common import (
    module_names, effective_norms, delta_lora, replay_with_wx,
)
from s5_lean_circuit.composition_loco import load_eval_bank
from s5_lean_circuit.pg_transcoder_adapter import load_adapter

from .context import OUT
from .context import SOURCE
SITES = [f"block{block}.{kind}" for block in range(12) for kind in KINDS]
CONDITIONS = ("on", "off", "zero12")
# Fixed before the first complete-dataset run. Frontier/omitted mass remain
# measurable, so a small pruned graph cannot be mistaken for a full circuit.
MAX_EXPANDED_PER_SITE = 8
TOP_FEATURE_EDGES = 8


def parents(site):
    block_s, kind = site.split(".")
    block = int(block_s[5:])
    previous = f"block{block-1}.residual" if block else "input.parameter_tokens"
    prior = [previous] + [f"block{block}.{k}" for k in KINDS[:KINDS.index(kind)]]
    if kind == "cross_attn":
        prior.append("input.condition_memory")
    return prior


@contextmanager
def direct_graph_hooks(core):
    pg = core.parameter_generator.model
    if pg.condition_injection_mode != "none" or pg.gradient_checkpointing:
        raise ValueError("unaccounted condition injection or checkpointed duplicate hooks")
    captured, handles = {}, []
    modules = [("input.parameter_tokens", pg.norm_tokens),
               ("input.condition_memory", pg.norm_input)]
    modules += [(site, module_at(pg, site)) for site in SITES]
    for site, module in modules:
        def hook(_module, _args, output, site=site):
            if site in captured or not isinstance(output, torch.Tensor):
                raise ValueError(f"duplicate/non-tensor graph site {site}")
            leaf = output.detach().requires_grad_(True)
            captured[site] = {"raw": output, "leaf": leaf}
            return leaf
        handles.append(module.register_forward_hook(hook))
    try:
        yield captured
    finally:
        for handle in handles:
            handle.remove()


def feature_name(site, slot, feature):
    return f"{site}.slot{slot}.sae{feature}"


def save_gzip(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    temp.write_bytes(gzip.compress(canonical_json(value), compresslevel=6, mtime=0))
    os.replace(temp, path)


def projection(state, reference, name, denominator):
    a, b = state[name + ".lora_A"][0].float(), state[name + ".lora_B"][0].float()
    ar, br = reference[name + ".lora_A"][0].float(), reference[name + ".lora_B"][0].float()
    return ((b.T @ br) * (a @ ar.T)).sum() * state["scale"].float() * reference["scale"].float() / denominator


class Graph:
    def __init__(self, captured, models, scales, query_length, condition):
        self.captured, self.models, self.scales = captured, models, scales
        self.query_length, self.condition = query_length, condition
        self.nodes, self.edges, self.accounting = {}, [], []
        self.pending, self.expanded, self.counts = [], set(), {}
        self.input_connected = set()
        self.z, self.error, self.site_quality = {}, {}, {}
        with torch.no_grad():
            for site in SITES:
                model, scale = models[int(site.split(".")[0][5:])], scales[site]
                x = canonical(site, captured[site]["leaf"]).float()
                z = F.linear(x / scale, model.W_enc.float(), model.b_enc.float()).relu()
                reconstruction = (z @ model.W_dec.float() + model.b_dec.float()) * scale
                error = x - reconstruction
                self.z[site], self.error[site] = z, error
                self.site_quality[site] = {"reconstruction_nmse": float(error.square().sum() / x.square().sum().clamp_min(1e-20)),
                                           "mean_l0": float(z.gt(0).sum(-1).float().mean())}

    def add_feature(self, site, slot, feature, priority=0.):
        name = feature_name(site, slot, feature)
        activation = float(self.z[site][slot, feature])
        row = self.nodes.setdefault(name, {"id": name, "kind": "sae_feature", "site": site,
                   "block": int(site.split(".")[0][5:]), "slot": slot,
                   "feature_id": feature, "activation": activation, "expanded": False,
                   "max_root_priority": 0.})
        if priority > row["max_root_priority"]:
            row["max_root_priority"] = priority
            heapq.heappush(self.pending, (-priority, name, site, slot, feature))
        return name

    def edge(self, source, target, value, kind):
        self.edges.append({"source": source, "target": target, "value": value,
                           "kind": kind, "causally_validated": False})

    def incoming(self, target, scalar, source_sites, downstream_scale):
        leaves = [self.captured[site]["leaf"] for site in source_sites]
        grads = torch.autograd.grad(scalar, leaves, retain_graph=True, allow_unused=True)
        feature_candidates, input_candidates = [], []
        signed_all, feature_abs, error_abs, input_abs = 0., 0., 0., 0.
        for site, grad in zip(source_sites, grads):
            if grad is None:
                continue
            if site.startswith("input."):
                leaf = self.captured[site]["leaf"].detach().float()
                values = (leaf * grad.float()).reshape(-1, 1024).sum(-1)
                signed_all += float(values.sum())
                input_abs += float(values.abs().sum())
                if site == "input.condition_memory":
                    expected = self.query_length + int(self.condition != "off")
                    if len(values) != expected:
                        raise ValueError("true OFF memory length differs")
                    keep = torch.topk(values.abs(), min(4, len(values))).indices.tolist()
                    if self.condition != "off" and self.query_length not in keep:
                        keep.append(self.query_length)
                    for index in keep:
                        name = "input.circuit_token" if index == self.query_length else f"input.query_token{index}"
                        input_candidates.append((name, float(values[index])))
                else:
                    for index in torch.topk(values.abs(), min(4, len(values))).indices.tolist():
                        input_candidates.append((f"input.parameter_slot{index}", float(values[index])))
                continue
            model, scale = self.models[int(site.split(".")[0][5:])], self.scales[site]
            g = canonical(site, grad).float()
            with torch.no_grad():
                contributions = F.linear(g, model.W_dec.float()) * self.z[site] * scale
                bias_effect = float((g * (model.b_dec.float() * scale)).sum())
                error_effect = float((g * self.error[site]).sum())
                signed_all += float(contributions.sum()) + bias_effect + error_effect
                feature_abs += float(contributions.abs().sum())
                error_abs += abs(error_effect)
                flat = contributions.flatten()
                vals, indices = torch.topk(flat.abs(), min(TOP_FEATURE_EDGES, flat.numel()))
                for absolute, index in zip(vals.tolist(), indices.tolist()):
                    if absolute > 1e-14:
                        feature_candidates.append((absolute, site, index // 8192, index % 8192, float(flat[index])))
                for suffix, value, kind in (("sae_error", error_effect, "sae_error"),
                                             ("decoder_bias", bias_effect, "decoder_bias")):
                    name = f"{site}.{suffix}"
                    self.nodes.setdefault(name, {"id": name, "site": site, "kind": kind})
                    self.edge(name, target, value, kind)
        selected = sorted(feature_candidates, reverse=True)[:TOP_FEATURE_EDGES]
        kept_signed, kept_abs = 0., 0.
        for absolute, site, slot, feature, value in selected:
            name = self.add_feature(site, slot, feature, absolute * downstream_scale)
            self.edge(name, target, value, "local_jacobian_feature_edge")
            kept_signed += value
            kept_abs += absolute
        for name, value in input_candidates:
            self.nodes.setdefault(name, {"id": name, "kind": "model_input"})
            self.edge(name, target, value, "local_jacobian_input_edge")
        # The intercept makes a tangent approximation exact at this point only.
        # It is NOT counted as explained mechanism or an empirical fidelity test.
        scalar_value = float(scalar.detach())
        intercept = scalar_value - signed_all
        name = target + ".local_affine_offset"
        self.nodes[name] = {"id": name, "kind": "local_affine_offset"}
        self.edge(name, target, intercept, "local_affine_offset")
        kept_total = sum(row["value"] for row in self.edges if row["target"] == target)
        omitted = scalar_value - kept_total
        name = target + ".omitted_incoming"
        self.nodes[name] = {"id": name, "kind": "pruned_incoming"}
        self.edge(name, target, omitted, "pruned_incoming")
        self.accounting.append({"target": target, "target_value": scalar_value,
            "feature_absolute_mass": feature_abs, "retained_feature_absolute_mass": kept_abs,
            "retained_feature_fraction": kept_abs / feature_abs if feature_abs > 1e-20 else None,
            "sae_error_absolute_mass": error_abs, "input_absolute_mass": input_abs,
            "local_affine_offset": intercept, "pruned_signed_sum": omitted,
            "bookkeeping_is_not_fidelity": True})

    def trace(self, state, reference, reference_norms):
        roots = []
        for module in module_names(state):
            name = "effective_lora." + module
            roots.append(name)
            denominator = max(reference_norms[module], 1e-12)
            scalar = projection(state, reference, module, denominator)
            self.nodes[name] = {"id": name, "kind": "effective_lora_output", "module": module,
                "projection_on_fixed_ON_direction": float(scalar.detach()), "on_frobenius_norm": denominator}
            self.incoming(name, scalar, ["block11.residual"], 1. / denominator)
        while self.pending:
            negative_priority, name, site, slot, feature = heapq.heappop(self.pending)
            if name in self.expanded or self.counts.get(site, 0) >= MAX_EXPANDED_PER_SITE:
                continue
            self.expand_feature(name, site, slot, feature, -negative_priority, backbone=False)
        # Budget pruning must not systematically erase entire LoRA modules.
        # For each endpoint, expand a strongest retained ancestry until an
        # actual nonzero input edge is reached, even above the per-site budget.
        # This never invents an edge; failures remain explicit.
        connectivity = {name: self.connect_input(name) for name in roots}
        return {"nodes": list(self.nodes.values()), "edges": self.edges,
                "incoming_accounting": self.accounting, "site_quality": self.site_quality,
                "expanded_per_site": self.counts,
                "input_path_by_endpoint": connectivity,
                "unexpanded_feature_count": sum(n["kind"] == "sae_feature" and not n["expanded"] for n in self.nodes.values()),
                "graph_type": "pruned prompt-specific direct local-Jacobian attribution graph",
                "complete_causal_circuit": False}

    def expand_feature(self, name, site, slot, feature, priority, backbone):
        self.expanded.add(name)
        self.counts[site] = self.counts.get(site, 0) + 1
        self.nodes[name]["expanded"] = True
        self.nodes[name]["backbone_expansion"] = backbone
        model, scale = self.models[int(site.split(".")[0][5:])], self.scales[site]
        x = canonical(site, self.captured[site]["raw"])[slot].float()
        scalar = (x / scale * model.W_enc[feature].float()).sum() + model.b_enc[feature].float()
        self.incoming(name, scalar, parents(site), priority / max(float(scalar.detach()), 1e-12))

    def connect_input(self, name):
        if name in self.input_connected:
            return True
        node = self.nodes[name]
        if node["kind"] == "model_input":
            self.input_connected.add(name)
            return True
        if node["kind"] == "sae_feature" and not node["expanded"]:
            self.expand_feature(name, node["site"], node["slot"], node["feature_id"],
                                node["max_root_priority"], backbone=True)
        incoming = [row for row in self.edges if row["target"] == name
                    and row["kind"] in ("local_jacobian_feature_edge", "local_jacobian_input_edge")
                    and abs(row["value"]) > 1e-14]
        for row in sorted(incoming, key=lambda row: (-abs(row["value"]), row["source"])):
            if self.connect_input(row["source"]):
                self.input_connected.add(name)
                return True
        return False


def one_question(adapter, bank, question, target, models, scales, identity):
    started = perf_counter()
    ids_cpu, vector_cpu = check_question(bank, question)
    ids, vector = ids_cpu.to("cuda", torch.long), vector_cpu.to("cuda")
    native, wxs, predictions = {}, {}, {}
    for condition in CONDITIONS:
        controlled, present = condition_input(vector, condition)
        native[condition] = generate(adapter.hypernet, ids, controlled, present, float(adapter.gate_multiplier))
        predictions[condition], wxs[condition] = replay_with_wx(adapter.hypernet, adapter.tokenizer, ids, native[condition], target)
        predictions[condition]["correct"] = predictions[condition]["first_tactic"] == question["gold_first_tactic"]
    baseline_path = SOURCE / "baseline" / f"shard{question['shard']:02d}" / f"q{question['ordinal']:04d}.json"
    baseline = json.loads(baseline_path.read_text())
    if (not baseline["complete"] or baseline["question"] != question
            or set(baseline['predictions']) != {'on','off','zero12','zero397','shuffled'}
            or {c:baseline['predictions'][c] for c in CONDITIONS} != predictions):
        raise ValueError("graph predictions differ from the frozen same-question full baseline")
    # Inference tensors cannot be saved by a backward operation; clone outside
    # inference mode into ordinary constant tensors for the fixed ON direction.
    reference = {key: value.clone().detach() for key, value in native["on"].items()}
    norms = effective_norms(reference)
    graphs = {}
    for condition in CONDITIONS:
        controlled, present = condition_input(vector, condition)
        with direct_graph_hooks(adapter.hypernet) as captured:
            with torch.autocast("cuda", dtype=torch.bfloat16):
                state = adapter.hypernet.generate_lora(ids, torch.ones_like(ids, dtype=torch.bool), controlled,
                    circuit_present=torch.tensor([present], device="cuda"), gate_multiplier=float(adapter.gate_multiplier))
        if set(state) != set(native[condition]) or any(not torch.equal(state[k], native[condition][k]) for k in state):
            raise ValueError("direct graph hooks changed effective LoRA factors")
        graph = Graph(captured, models, scales, ids.shape[1], condition)
        graphs[condition] = graph.trace(state, reference, norms)
        graphs[condition]["no_op_lora_bitwise_equal"] = True
        graphs[condition]["effective_lora_norms"] = effective_norms(native[condition])
        del state, graph, captured
    changes = {}
    for condition in ("off", "zero12"):
        changes[condition] = {"effective_delta_w_l2": delta_lora(native["on"], native[condition]),
             "effective_delta_wx_l2": {name: float((wxs["on"][name] - wxs[condition][name]).norm()) for name in norms},
             "right_to_wrong": predictions["on"]["correct"] and not predictions[condition]["correct"],
             "wrong_to_right": not predictions["on"]["correct"] and predictions[condition]["correct"]}
    return {"complete": True, "identity_sha256": sha_bytes(canonical_json(identity)), "question": question,
            "graphs": graphs, "predictions": predictions, "condition_changes": changes,
            "elapsed_seconds": perf_counter() - started}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shard", type=int, required=True, choices=range(12))
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise ValueError("CUDA is required")
    if args.limit is not None and args.limit < 1:
        raise ValueError("smoke limit must be positive")
    torch.set_num_threads(4)
    source, grouped, source_path = source_questions()
    target = target_identity()
    models, scales, hashes = {}, {}, {}
    for block in range(12):
        model, rms, sites, model_sha, _, _ = load_sae(block, source, source_path, target)
        models[block] = model.requires_grad_(False)
        scales.update(zip(sites, rms))
        hashes[str(block)] = model_sha
    atom = load_eval_bank(ATOMIC_BANK)
    banks = {"T_nested_or": load_eval_bank(COMPOSITION_BANK), "S_or_l": atom, "S_or_r": atom}
    selected = [row for task in TASKS for row in grouped[task] if row["shard"] == args.shard]
    if args.limit is not None:
        selected = [next(row for row in selected if row["task_type"] == task) for task in TASKS][:args.limit]
    adapter = load_adapter(target, root=ROOT, device=torch.device("cuda"))
    identity = {"format": "r03_sae_circuit_graph_v1", "tasks": list(TASKS), "question_creation": False,
        "checkpoint_sha256": target["checkpoint_sha256"], "sae_sha256": hashes,
        "runtime_config_sha256": sha(ROOT / target["config_path"]), "source_sha256": sha(Path(__file__)),
        "question_manifest_sha256": sha(source_path), "slot_map": slot_map(adapter.hypernet.parameter_generator),
        "conditions": {"on": {"circuit_present": True, "zero_dimensions": []},
                       "off": {"circuit_present": False, "zero_dimensions": []},
                       "zero12": {"circuit_present": True, "zero_dimensions": list(range(9)) + [131,132,133]}},
        "max_expanded_per_site": MAX_EXPANDED_PER_SITE, "top_feature_edges": TOP_FEATURE_EDGES,
        "backbone_rule": "expand strongest retained ancestors per LoRA endpoint until a measured input edge is reached; no invented edges",
        "precision": "native PG BF16; SAE encoding, decoding, and local Jacobian contractions FP32",
        "endpoints": "all32 effective DeltaW modules projected onto this question's fixed ON DeltaW direction; all272 slots retained",
        "claim": "candidate attribution graph; edges require intervention, sparse graph is not a complete causal explanation"}
    directory = OUT / ("smoke" if args.limit is not None else "graphs") / f"shard{args.shard:02d}"
    directory.mkdir(parents=True, exist_ok=True)
    write_json(directory / "identity.json", identity)
    try:
        for pos, question in enumerate(selected):
            path = directory / f"q{question['ordinal']:04d}.json.gz"
            if path.exists():
                old = json.loads(gzip.decompress(path.read_bytes()))
                if not old["complete"] or old["identity_sha256"] != sha_bytes(canonical_json(identity)) or old["question"] != question:
                    raise ValueError("saved graph identity differs")
            else:
                save_gzip(path, one_question(adapter, banks[question["task_type"]], question, target, models, scales, identity))
            write_json(directory / "status.json", {"complete": pos + 1 == len(selected), "completed": pos + 1,
                       "assigned": len(selected), "smoke_limit": args.limit, "identity_sha256": sha_bytes(canonical_json(identity))})
            print("R03_SAE_CIRCUIT_GRAPH_PROGRESS", args.shard, pos + 1, len(selected), flush=True)
    finally:
        adapter.close()
    print("R03_SAE_CIRCUIT_GRAPH_COMPLETE", args.shard, flush=True)


if __name__ == "__main__":
    main()
