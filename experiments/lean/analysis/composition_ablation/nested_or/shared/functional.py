#!/usr/bin/env python3
"""Real-PG feature and direct-edge patching for the per-block SAE circuit graph."""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import gzip
import json
from pathlib import Path
import sys

import torch
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[2]
for directory in (ROOT / "s5_lean_circuit/src", ROOT / "s2_hypernet/src",
                  ROOT / "s5_lean_circuit/scripts", ROOT / "temp_s"):
    sys.path.insert(0, str(directory))

from .context import COMPOSITION_BANK, ATOMIC_BANK
from .graph import OUT, SITES, parents
from .sae import (
    TASKS, check_question, source_questions, sha, canonical_json, sha_bytes, write_json,
)
from .conditions import load_sae
from .common import canonical, module_at
from .common import target_identity
from .common import generate
from .common import delta_lora, module_names, replay_with_wx
from s5_lean_circuit.composition_loco import load_eval_bank
from s5_lean_circuit.pg_transcoder_adapter import load_adapter


def raw_layout(site, flat, template):
    if site.endswith(".self_attn_n"):
        return flat.reshape(8, 34, 1024).permute(1, 0, 2).reshape_as(template)
    return flat.reshape_as(template)


def block_model(models, site):
    return models[int(site.split(".")[0][5:])]


def coefficients(value, site, models, scales, features):
    model = block_model(models, site)
    with torch.autocast("cuda", enabled=False):
        return F.linear(canonical(site, value).float() / scales[site],
                        model.W_enc[features].float(), model.b_enc[features].float()).relu()


def decoded_change(value, site, models, scales, features, delta_z, slots=None):
    model = block_model(models, site)
    if slots is not None:
        mask = torch.zeros((272, 1), device=value.device)
        mask[slots] = 1
        delta_z = delta_z * mask
    with torch.autocast("cuda", enabled=False):
        contribution = (delta_z.float() @ model.W_dec[features].float()) * scales[site]
    flat = canonical(site, value).float() + contribution
    return raw_layout(site, flat, value).to(value.dtype)


@contextmanager
def hooks(core, callback):
    handles = []
    for site in SITES:
        def hook(_module, _args, output, site=site):
            return callback(site, output)
        handles.append(module_at(core.parameter_generator.model, site).register_forward_hook(hook))
    try:
        yield
    finally:
        for handle in handles:
            handle.remove()


@torch.inference_mode()
def capture_native(adapter, ids, vector):
    sink = {}
    def capture(site, output):
        if site in sink:
            raise ValueError("duplicate real PG capture")
        sink[site] = output.detach().clone()
    with hooks(adapter.hypernet, capture):
        state = generate(adapter.hypernet, ids, vector, True, float(adapter.gate_multiplier))
    if set(sink) != set(SITES):
        raise ValueError("incomplete real PG capture")
    return sink, state


@torch.inference_mode()
def remove_node(adapter, ids, vector, source, models, scales, amount=1.):
    calls = []
    def edit(site, output):
        if site != source["site"]:
            return None
        calls.append(site)
        z = coefficients(output, site, models, scales, source["feature_ids"])
        return decoded_change(output, site, models, scales, source["feature_ids"], -amount * z, source.get("slots"))
    with hooks(adapter.hypernet, edit):
        state = generate(adapter.hypernet, ids, vector, True, float(adapter.gate_multiplier))
    if len(calls) != 1:
        raise ValueError("node patch coverage differs")
    return state


@torch.inference_mode()
def direct_edge(adapter, ids, vector, native, source, target, models, scales, amount=1.):
    if source["site"] not in parents(target["site"]):
        raise ValueError("requested source is not a direct structural parent")
    cutoff = SITES.index(target["site"])
    donor = {}
    def isolate(site, output):
        if site == source["site"]:
            z = coefficients(native[site], site, models, scales, source["feature_ids"])
            return decoded_change(native[site], site, models, scales, source["feature_ids"], -amount * z, source.get("slots"))
        if site == target["site"]:
            donor["value"] = output.detach().clone()
        if SITES.index(site) < cutoff:
            # Clamp other operations before the receiver to their exact native
            # outputs, so only the specified direct route reaches the receiver.
            return native[site]
        return None
    with hooks(adapter.hypernet, isolate):
        generate(adapter.hypernet, ids, vector, True, float(adapter.gate_multiplier))
    if "value" not in donor:
        raise ValueError("direct edge receiver was not captured")
    old = coefficients(native[target["site"]], target["site"], models, scales, target["feature_ids"])
    new = coefficients(donor["value"], target["site"], models, scales, target["feature_ids"])
    delta = new - old
    def receiver_patch(site, output):
        if site == target["site"]:
            return decoded_change(output, site, models, scales, target["feature_ids"], delta, target.get("slots"))
        return None
    with hooks(adapter.hypernet, receiver_patch):
        state = generate(adapter.hypernet, ids, vector, True, float(adapter.gate_multiplier))
    chosen = target.get("slots") or list(range(272))
    return state, {"native_target_z": old[chosen].cpu().tolist(),
                   "source_removed_target_z": new[chosen].cpu().tolist(),
                   "actual_target_delta_sum": float(delta[chosen].sum()),
                   "full_target_raw_delta_l2": float((donor["value"].float() - native[target["site"]].float()).norm())}


def prediction(adapter, ids, state, target, question):
    result, wx = replay_with_wx(adapter.hypernet, adapter.tokenizer, ids, state, target)
    result["correct"] = result["first_tactic"] == question["gold_first_tactic"]
    return result, wx


def smoke_edges(graph):
    nodes = {node["id"]: node for node in graph["nodes"]}
    selected = []
    for lo, hi, kind in ((lo, hi, kind) for lo, hi in ((0, 3), (4, 7), (8, 11))
                         for kind in ("cross_attn", "feed_forward", "residual")):
        choices = [edge for edge in graph["edges"] if edge["kind"] == "local_jacobian_feature_edge"
                   and nodes[edge["target"]]["kind"] == "sae_feature"
                   and nodes[edge["target"]]["site"].endswith("." + kind)
                   and lo <= nodes[edge["target"]]["block"] <= hi]
        if not choices:
            raise ValueError("graph has no feature edge in one depth range")
        edge = max(choices, key=lambda edge: abs(edge["value"]))
        src, dst = nodes[edge["source"]], nodes[edge["target"]]
        selected.append({"source": {"site": src["site"], "feature_ids": [src["feature_id"]], "slots": [src["slot"]]},
                         "target": {"site": dst["site"], "feature_ids": [dst["feature_id"]], "slots": [dst["slot"]]},
                         "predicted_target_delta_for_full_removal": -edge["value"]})
    return selected


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("edge-smoke", "functional"), required=True)
    parser.add_argument("--shard", type=int, default=0, choices=range(12))
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise ValueError("CUDA is required")
    torch.set_num_threads(4)
    source_manifest, grouped, source_path = source_questions()
    runtime = target_identity()
    models, scales, hashes = {}, {}, {}
    for block in range(12):
        model, rms, sites, model_sha, _, _ = load_sae(block, source_manifest, source_path, runtime)
        models[block] = model.requires_grad_(False); scales.update(zip(sites, rms)); hashes[str(block)] = model_sha
    atom = load_eval_bank(ATOMIC_BANK)
    banks = {"T_nested_or": load_eval_bank(COMPOSITION_BANK), "S_or_l": atom, "S_or_r": atom}
    questions = [row for task in TASKS for row in grouped[task] if row["shard"] == args.shard]
    if args.phase == "edge-smoke":
        questions = [next(row for row in questions if row["task_type"] == task) for task in TASKS]
        lock_sha, lock = None, None
    else:
        lock_path = OUT / "on_graph_candidate_lock.json"
        lock = json.loads(lock_path.read_text()); lock_sha = sha(lock_path)
        if not lock["complete"] or lock["sae_sha256"] != hashes or not lock["candidates"]:
            raise ValueError("ON graph candidate lock differs")
    adapter = load_adapter(runtime, root=ROOT, device=torch.device("cuda"))
    root = OUT / ("edge_smoke" if args.phase == "edge-smoke" else "functional") / f"shard{args.shard:02d}"
    try:
        for pos, question in enumerate(questions):
            path = root / f"q{question['ordinal']:04d}.json"
            identity = {"question": question, "sae_sha256": hashes, "candidate_lock_sha256": lock_sha,
                        "source_sha256": sha(Path(__file__)), "checkpoint_sha256": runtime["checkpoint_sha256"],
                        "runtime_config_sha256": sha(ROOT / runtime["config_path"])}
            if path.exists():
                prior = json.loads(path.read_text())
                if prior["identity"] != identity or not prior["complete"]:
                    raise ValueError("saved edge intervention identity differs")
                write_json(root / 'status.json', {'complete':pos+1==len(questions),
                    'completed':pos+1,'assigned':len(questions),'phase':args.phase,
                    'candidate_lock_sha256':lock_sha})
                continue
            graph_path = OUT / ("smoke" if args.phase == "edge-smoke" else "graphs") / f"shard{args.shard:02d}" / f"q{question['ordinal']:04d}.json.gz"
            graph_record = json.loads(gzip.decompress(graph_path.read_bytes()))
            candidates = smoke_edges(graph_record["graphs"]["on"]) if lock is None else lock["candidates"]
            ids_cpu, vec_cpu = check_question(banks[question["task_type"]], question)
            ids, vector = ids_cpu.to("cuda", torch.long), vec_cpu.to("cuda")
            native, baseline_state = capture_native(adapter, ids, vector)
            baseline, wx = prediction(adapter, ids, baseline_state, runtime, question)
            if baseline != graph_record["predictions"]["on"]:
                raise ValueError("graph and real-patch native predictions differ")
            results = []
            for index, candidate in enumerate(candidates):
                src, dst = candidate["source"], candidate.get("target")
                for amount in ((0., .1, 1.) if args.phase == "edge-smoke" else (1.,)):
                    if dst is not None:
                        edited, activation = direct_edge(adapter, ids, vector, native, src, dst, models, scales, amount)
                    else:
                        edited = remove_node(adapter, ids, vector, src, models, scales, amount)
                        activation = {}
                    if amount == 0 and any(not torch.equal(edited[k], baseline_state[k]) for k in edited):
                        raise ValueError("zero-strength direct edge patch changed LoRA")
                    answer, edited_wx = prediction(adapter, ids, edited, runtime, question)
                    results.append({"candidate": index, "definition": candidate, "amount": amount,
                        "activation": activation, "prediction": answer,
                        "right_to_wrong": baseline["correct"] and not answer["correct"],
                        "wrong_to_right": not baseline["correct"] and answer["correct"],
                        "effective_delta_w_l2": delta_lora(baseline_state, edited),
                        "effective_delta_wx_l2": {name: float((wx[name] - edited_wx[name]).norm()) for name in module_names(edited)}})
            write_json(path, {"complete": True, "identity": identity, "baseline": baseline, "results": results,
                             "program_check_only": args.phase == "edge-smoke"})
            write_json(root / "status.json", {"complete": pos + 1 == len(questions), "completed": pos + 1,
                       "assigned": len(questions), "phase": args.phase, "candidate_lock_sha256": lock_sha})
            print("R03_SAE_CIRCUIT_PATH_PROGRESS", args.phase, args.shard, pos + 1, len(questions), flush=True)
    finally:
        adapter.close()
    print("R03_SAE_CIRCUIT_PATH_COMPLETE", args.phase, args.shard, flush=True)


if __name__ == "__main__":
    main()
