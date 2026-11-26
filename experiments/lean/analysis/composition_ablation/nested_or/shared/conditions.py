#!/usr/bin/env python3
"""Measure ON/OFF/zero12 response of a full-272-slot R03 block SAE.

Each block has its own 8192-feature SAE shared across its five PG sites. This
reads only the existing T_nested_or, S_or_l and S_or_r dev500 questions. The same
question, Base-derived 397D vector, and 272 LoRA parameter slots are used in
all three conditions. OFF masks the Circuit token; zero12 keeps that token.

Raw resumable shards live under outputs/pg_transcoder/.../condition/blockXX;
small identity receipts, status and final statistics live under
logs/eval/.../condition/blockXX. No question is constructed by this script.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
for directory in (ROOT / "s5_lean_circuit/src", ROOT / "s2_hypernet/src",
                  ROOT / "s5_lean_circuit/scripts", ROOT / "temp_s"):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

from .context import ATOMIC_BANK, COMPOSITION_BANK
from .sae import (
    DATA, FEATURES, KINDS, LOG, SHARD_QUESTIONS, SHARDS_PER_TASK, SLOTS,
    SOURCE, SPLITS, TASKS, WIDTH, canonical_json, capture_block,
    check_question, sha, sha_bytes, sites_for, source_questions, write_json,
)
from .common import slot_map, target_identity
from .context import BRANCHES, COMPOSITION
from .common import generate
from s5_lean_circuit.circuit.transcoder import RectangularReLUTranscoder
from s5_lean_circuit.composition_loco import load_eval_bank
from s5_lean_circuit.pg_transcoder_adapter import load_adapter

CONDITIONS = ("on", "off", "zero12")
CONTROLS = ("off", "zero12")
ZERO12 = tuple(range(9)) + (131, 132, 133)
STAT_SHAPES = {
    "feature_sum": (3, 5, 3, FEATURES),
    "feature_question_support": (3, 5, 3, FEATURES),
    "response_abs_sum": (3, 5, 2, FEATURES),
    "response_question_support": (3, 5, 2, FEATURES),
    "reconstruction_sse": (3, 5, 3),
    "reconstruction_energy": (3, 5, 3),
    "delta_sse": (3, 5, 2),
    "delta_energy": (3, 5, 2),
    "question_count": (3,),
}
COHORTS = TASKS + BRANCHES
STAT_SHAPES = {key: (len(COHORTS), *shape[1:]) for key, shape in STAT_SHAPES.items()}
for label in ("positive_slot", "negative_slot", "positive_question", "negative_question"):
    STAT_SHAPES[f"response_{label}_support"] = (len(COHORTS), 5, 2, FEATURES)


def write_npz(path: Path, **arrays: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}.npz")
    try:
        np.savez_compressed(temporary, **arrays)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def load_sae(block: int, source: dict, source_path: Path, target: dict):
    capture_path = LOG / f"block{block:02d}_capture_manifest.json"
    capture = json.loads(capture_path.read_text())
    sites = sites_for(block)
    expected_split = {key: list(value) for key, value in SPLITS.items()}
    expected_starts = {(task, shard * SHARD_QUESTIONS)
                       for task in TASKS for shard in range(SHARDS_PER_TASK)}
    receipts = capture.get("capture_shards", [])
    receipt_starts = {(row["identity"]["task"], row["identity"]["task_index_start"])
                      for row in receipts}
    if (capture.get("format") != "lean_lc026_r03_block_fullslot_capture_manifest_v2"
            or capture.get("complete") is not True
            or capture.get("question_creation") is not False
            or capture.get("tasks") != list(TASKS)
            or capture.get("questions_per_task") != 500
            or capture.get("split") != expected_split
            or capture.get("site_names") != sites
            or capture.get("slot_indices") != list(SLOTS)
            or capture.get("source_manifest_sha256") != sha(source_path)
            or capture.get("source_questions_sha256") != source["questions_sha256"]
            or capture.get("r03_checkpoint_sha256") != target["checkpoint_sha256"]
            or capture.get("no_op_generated_lora_parity") is not True
            or len(receipts) != len(expected_starts)
            or receipt_starts != expected_starts):
        raise ValueError(f"block{block} fullslot ON capture manifest differs")
    for row in receipts:
        item = row["identity"]
        if (item.get("format") != "lean_lc026_r03_block_fullslot_capture_v2"
                or item.get("block") != block
                or item.get("condition") != "on"
                or item.get("circuit_present") is not True
                or item.get("question_count") != SHARD_QUESTIONS
                or item.get("shape") != [SHARD_QUESTIONS, len(KINDS), len(SLOTS), WIDTH]
                or item.get("slot_indices") != list(SLOTS)
                or item.get("site_names") != sites
                or item.get("source_questions_sha256") != source["questions_sha256"]
                or item.get("slot_map_sha256") != capture["slot_map_sha256"]):
            raise ValueError(f"block{block} fullslot ON capture shard identity differs")

    fit_path = LOG / f"block{block:02d}_fit_summary.json"
    fit = json.loads(fit_path.read_text())
    checkpoint_path = DATA / "models" / f"block{block:02d}_shared5site_8192" / "best.pt"
    checkpoint_sha = sha(checkpoint_path)
    capture_sha = sha(capture_path)
    if (fit.get("format") != "lean_lc026_r03_block_sae_fullslot_fit_v2"
            or fit.get("complete") is not True
            or fit.get("question_creation") is not False
            or fit.get("block") != block
            or fit.get("capture_manifest_sha256") != capture_sha
            or fit.get("checkpoint_path") != str(checkpoint_path.relative_to(ROOT))
            or fit.get("checkpoint_sha256") != checkpoint_sha):
        raise ValueError(f"block{block} fullslot SAE fit receipt differs")
    saved = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    config = saved["config"]
    scales = config.get("site_rms", [])
    if (saved.get("format") != "lean_lc026_r03_block_sae_fullslot_checkpoint_v2"
            or saved.get("capture_manifest_sha256") != capture_sha
            or fit.get("config") != config
            or config.get("format") != "lean_lc026_r03_block_sae_fullslot_fit_config_v2"
            or config.get("block") != block
            or config.get("site_names") != sites
            or config.get("slot_indices") != list(SLOTS)
            or config.get("training_tasks") != list(TASKS)
            or config.get("split") != expected_split
            or config.get("dictionary_size") != FEATURES
            or config.get("d_in") != WIDTH or config.get("d_out") != WIDTH
            or config.get("r03_checkpoint_sha256") != target["checkpoint_sha256"]
            or config.get("capture_manifest_sha256") != capture_sha
            or config.get("site_rms_source") != "all train300 questions and all 272 slots for each of three tasks"
            or len(scales) != len(KINDS)
            or any(not np.isfinite(float(x)) or float(x) <= 0 for x in scales)):
        raise ValueError(f"block{block} fullslot SAE checkpoint or normalization differs")
    model = RectangularReLUTranscoder.from_state_dict(saved["state_dict"]).eval().cuda()
    if (model.d_in, model.d_out, model.d_feature) != (WIDTH, WIDTH, FEATURES):
        raise ValueError("fullslot SAE model dimensions differ")
    return model, [float(x) for x in scales], sites, checkpoint_sha, capture_sha, capture


def condition_input(vector: torch.Tensor, condition: str) -> tuple[torch.Tensor, bool]:
    if condition not in CONDITIONS or vector.shape != (1, 397):
        raise ValueError("condition or original Circuit397 shape differs")
    controlled = vector.clone()
    if condition == "zero12":
        controlled[:, list(ZERO12)] = 0
    return controlled, condition != "off"


@torch.inference_mode()
def capture_conditions(adapter, ids: torch.Tensor, vector: torch.Tensor,
                       block: int) -> dict[str, dict[str, torch.Tensor]]:
    captured = {}
    for condition in CONDITIONS:
        controlled, present = condition_input(vector, condition)
        sink: dict[str, torch.Tensor] = {}
        with capture_block(adapter.hypernet, block, sink):
            generate(adapter.hypernet, ids, controlled, present,
                     float(adapter.gate_multiplier))
        captured[condition] = sink
    return captured


@torch.inference_mode()
def no_op_checks(adapter, banks: dict, rows: dict, block: int,
                 capture_manifest: dict) -> list[dict]:
    checks = []
    on_receipts = {receipt["identity"]["task"]: receipt
                   for receipt in capture_manifest["capture_shards"]
                   if receipt["identity"]["task_index_start"] == 0}
    if set(on_receipts) != set(TASKS):
        raise ValueError("fullslot ON first-shard capture coverage differs")
    for task in TASKS:
        ids_cpu, vector_cpu = check_question(banks[task], rows[task][0])
        ids = ids_cpu.to("cuda", torch.long)
        vector = vector_cpu.to("cuda")
        for condition in CONDITIONS:
            controlled, present = condition_input(vector, condition)
            native = generate(adapter.hypernet, ids, controlled, present,
                              float(adapter.gate_multiplier))
            sink: dict[str, torch.Tensor] = {}
            with capture_block(adapter.hypernet, block, sink):
                hooked = generate(adapter.hypernet, ids, controlled, present,
                                  float(adapter.gate_multiplier))
            if (set(native) != set(hooked)
                    or any(not torch.equal(native[key], hooked[key]) for key in native)):
                raise ValueError(f"block{block} {task}/{condition} no-op hook changed LoRA")
            on_capture_exact = None
            if condition == "on":
                receipt = on_receipts[task]
                data_path = ROOT / receipt["path"]
                if sha(data_path) != receipt["sha256"]:
                    raise ValueError(f"block{block} {task} ON training capture hash differs")
                trained = np.load(data_path, mmap_mode="r", allow_pickle=False)[0]
                on_capture_exact = all(np.array_equal(sink[site].numpy(), trained[index])
                                       for index, site in enumerate(sites_for(block)))
                if not on_capture_exact:
                    raise ValueError(f"block{block} {task} ON recapture differs from SAE training input")
            checks.append({"task": task, "task_index": 0, "condition": condition,
                           "event_key": rows[task][0]["event_key"],
                           "site_count": len(sink), "lora_key_count": len(native),
                           "bitwise_equal": True,
                           "on_training_capture_exact": on_capture_exact})
    return checks


def empty_stats() -> dict[str, torch.Tensor]:
    output = {}
    for key, shape in STAT_SHAPES.items():
        dtype = torch.int32 if "support" in key or key == "question_count" else torch.float64
        output[key] = torch.zeros(shape, dtype=dtype, device="cuda")
    return output


@torch.inference_mode()
def measure_one(stats: dict[str, torch.Tensor], task: str, captured: dict,
                model, scales: list[float], sites: list[str],
                decoder_norms: torch.Tensor, branch: str) -> None:
    task_index = [TASKS.index(task)]
    if task == COMPOSITION:
        task_index.append(COHORTS.index(branch))
    for site_index, site in enumerate(sites):
        scale = scales[site_index]
        x = {condition: captured[condition][site].to("cuda", dtype=torch.float32) / scale
             for condition in CONDITIONS}
        z, reconstructed = {}, {}
        for condition_index, condition in enumerate(CONDITIONS):
            with torch.autocast("cuda", dtype=torch.bfloat16):
                reconstructed[condition], z[condition] = model(x[condition])
            if (z[condition].shape != (len(SLOTS), FEATURES)
                    or reconstructed[condition].shape != (len(SLOTS), WIDTH)):
                raise ValueError("fullslot SAE output shape differs")
            physical = z[condition].float() * (decoder_norms * scale)[None, :]
            stats["feature_sum"][task_index, site_index, condition_index] += physical.sum(dim=0).double()
            stats["feature_question_support"][task_index, site_index, condition_index] += z[condition].gt(0).any(dim=0).int()
            target = x[condition].float()
            predicted = reconstructed[condition].float()
            stats["reconstruction_sse"][task_index, site_index, condition_index] += (predicted - target).double().square().sum()
            stats["reconstruction_energy"][task_index, site_index, condition_index] += target.double().square().sum()
        for control_index, control in enumerate(CONTROLS):
            response = (z["on"].float() - z[control].float()) * (decoder_norms * scale)[None, :]
            stats["response_abs_sum"][task_index, site_index, control_index] += response.abs().sum(dim=0).double()
            stats["response_question_support"][task_index, site_index, control_index] += response.ne(0).any(dim=0).int()
            stats["response_positive_slot_support"][task_index, site_index, control_index] += response.gt(0).sum(dim=0).int()
            stats["response_negative_slot_support"][task_index, site_index, control_index] += response.lt(0).sum(dim=0).int()
            mean_response = response.mean(dim=0)
            stats["response_positive_question_support"][task_index, site_index, control_index] += mean_response.gt(0).int()
            stats["response_negative_question_support"][task_index, site_index, control_index] += mean_response.lt(0).int()
            target_delta = x["on"].float() - x[control].float()
            decoded_delta = reconstructed["on"].float() - reconstructed[control].float()
            stats["delta_sse"][task_index, site_index, control_index] += (decoded_delta - target_delta).double().square().sum()
            stats["delta_energy"][task_index, site_index, control_index] += target_delta.double().square().sum()
    stats["question_count"][task_index] += 1


def part_identity(identity: dict, task: str, shard: int, questions: list[dict]) -> dict:
    question_ids = [{key: row[key] for key in (
        "ordinal", "task_index", "canonical_global_index", "event_key",
        "prompt_sha256", "gold_output_sha256", "raw397_model_input_sha256", "branch",
    )} for row in questions]
    return {"format": "lean_lc026_r03_block_sae_fullslot_condition_part_v2",
            "run_identity_sha256": sha_bytes(canonical_json(identity)),
            "task": task, "task_index_start": shard * SHARD_QUESTIONS,
            "question_count": len(questions),
            "question_identities": question_ids,
            "question_identities_sha256": sha_bytes(canonical_json(question_ids)),
            "slot_indices": list(SLOTS)}


def part_paths(root: Path, data_root: Path, task: str, shard: int) -> tuple[Path, Path]:
    name = f"part{shard:02d}"
    return data_root / task / f"{name}.npz", root / "parts" / task / f"{name}.json"


def load_part(data_path: Path, receipt_path: Path, expected: dict) -> dict[str, np.ndarray] | None:
    if not receipt_path.is_file():
        return None
    receipt = json.loads(receipt_path.read_text())
    if (receipt.get("identity") != expected
            or receipt.get("path") != str(data_path.relative_to(ROOT))
            or not data_path.is_file()
            or receipt.get("sha256") != sha(data_path)):
        raise ValueError(f"resumed condition shard identity/checksum differs: {data_path}")
    with np.load(data_path, allow_pickle=False) as saved:
        if set(saved.files) != set(STAT_SHAPES):
            raise ValueError(f"resumed condition shard fields differ: {data_path}")
        arrays = {key: saved[key].copy() for key in STAT_SHAPES}
    for key, shape in STAT_SHAPES.items():
        value = arrays[key]
        if (value.shape != shape or not np.isfinite(value).all()
                or ("support" in key or key == "question_count")
                != np.issubdtype(value.dtype, np.integer)):
            raise ValueError(f"resumed condition shard value differs: {data_path}:{key}")
    task = expected["task"]
    count = np.zeros(len(COHORTS), dtype=np.int32)
    count[TASKS.index(task)] = expected["question_count"]
    if task == COMPOSITION:
        for question in expected["question_identities"]:
            count[COHORTS.index(question["branch"])] += 1
    if (not np.array_equal(arrays["question_count"], count)
            or np.any(arrays["feature_question_support"] > expected["question_count"])
            or np.any(arrays["response_question_support"] > expected["question_count"])
            or np.any(arrays["reconstruction_sse"] < 0)
            or np.any(arrays["reconstruction_energy"] < 0)
            or np.any(arrays["delta_sse"] < 0)
            or np.any(arrays["delta_energy"] < 0)):
        raise ValueError(f"resumed condition shard statistics differ: {data_path}")
    return arrays


def accumulate(parts: list[dict[str, np.ndarray]]) -> dict[str, np.ndarray]:
    return {key: np.stack([part[key] for part in parts], axis=0).sum(axis=0)
            for key in STAT_SHAPES}


def ratio(numerator: np.ndarray, denominator: np.ndarray) -> np.ndarray:
    return np.divide(numerator, denominator,
                     out=np.full(numerator.shape, np.nan, dtype=np.float64),
                     where=denominator > 1e-12)


def summary_arrays(total: dict[str, np.ndarray], sites: list[str],
                   identity: dict, population: str, expected_per_task: int) -> dict[str, np.ndarray]:
    counts = total["question_count"]
    if (not np.array_equal(counts[:3], [expected_per_task] * len(TASKS))
            or int(counts[3:].sum()) != expected_per_task):
        raise ValueError(f"{population} three-task condition coverage incomplete")
    denominator = counts[:, None, None, None].astype(np.float64) * len(SLOTS)
    mean = total["feature_sum"] / denominator
    abs_mean = total["response_abs_sum"] / denominator
    output = {
        "tasks": np.asarray(COHORTS), "sites": np.asarray(sites),
        "cohort_kind": np.asarray(["task"] * 3 + ["gold_branch"] * 3),
        "conditions": np.asarray(CONDITIONS), "controls": np.asarray(CONTROLS),
        "slot_indices": np.asarray(SLOTS, dtype=np.int16),
        "feature_ids": np.arange(FEATURES, dtype=np.int16),
        "question_count": counts.astype(np.int32),
        "population": np.asarray(population),
        "identity_json": np.asarray(json.dumps(identity, sort_keys=True, separators=(",", ":"))),
    }
    for index, condition in enumerate(CONDITIONS):
        output[f"{condition}_mean"] = mean[:, :, index].astype(np.float32)
        support = total["feature_question_support"][:, :, index]
        output[f"{condition}_question_support"] = support.astype(np.int32)
        output[f"{condition}_question_support_rate"] = (support / counts[:, None, None]).astype(np.float32)
        output[f"{condition}_reconstruction_sse"] = total["reconstruction_sse"][:, :, index].astype(np.float64)
        output[f"{condition}_reconstruction_energy"] = total["reconstruction_energy"][:, :, index].astype(np.float64)
        output[f"{condition}_reconstruction_nmse"] = ratio(
            total["reconstruction_sse"][:, :, index], total["reconstruction_energy"][:, :, index])
    for index, control in enumerate(CONTROLS):
        name = f"on_minus_{control}"
        output[f"{name}_signed"] = (mean[:, :, 0] - mean[:, :, index + 1]).astype(np.float32)
        output[f"{name}_absolute"] = abs_mean[:, :, index].astype(np.float32)
        support = total["response_question_support"][:, :, index]
        output[f"{name}_question_support"] = support.astype(np.int32)
        output[f"{name}_question_support_rate"] = (support / counts[:, None, None]).astype(np.float32)
        output[f"{name}_sse"] = total["delta_sse"][:, :, index].astype(np.float64)
        output[f"{name}_energy"] = total["delta_energy"][:, :, index].astype(np.float64)
        output[f"{name}_nmse"] = ratio(total["delta_sse"][:, :, index],
                                       total["delta_energy"][:, :, index])
        for granularity, denominator in (("slot", counts[:, None, None] * len(SLOTS)),
                                          ("question", counts[:, None, None])):
            positive = total[f"response_positive_{granularity}_support"][:, :, index]
            negative = total[f"response_negative_{granularity}_support"][:, :, index]
            output[f"{name}_positive_{granularity}_fraction"] = (positive / denominator).astype(np.float32)
            output[f"{name}_negative_{granularity}_fraction"] = (negative / denominator).astype(np.float32)
            signed = output[f"{name}_signed"]
            matching = np.where(signed > 0, positive, np.where(signed < 0, negative, 0))
            output[f"{name}_mean_direction_{granularity}_agreement"] = (matching / denominator).astype(np.float32)
            output[f"{name}_mean_direction_{granularity}_agreement_nonzero"] = ratio(matching, positive + negative).astype(np.float32)
    return output


def completed_parts(root: Path, data_root: Path) -> list[str]:
    return [f"{task}/part{shard:02d}" for task in TASKS for shard in range(SHARDS_PER_TASK)
            if (root / "parts" / task / f"part{shard:02d}.json").is_file()
            and (data_root / task / f"part{shard:02d}.npz").is_file()]


@torch.inference_mode()
def run(block: int, limit: int | None) -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    torch.set_num_threads(4)
    source, rows, source_path = source_questions()
    target = target_identity()
    runtime_config_path = ROOT / target["config_path"]
    if not runtime_config_path.is_file():
        raise ValueError("R03 runtime config file is missing")
    runtime_config_sha = sha(runtime_config_path)
    if source.get("conditions") != {
            "on": {"circuit_present": True, "zeroed_raw397_dimensions": []},
            "off": {"circuit_present": False, "zeroed_raw397_dimensions": []},
            "zero12": {"circuit_present": True,
                       "zeroed_raw397_dimensions": list(ZERO12)},
            "zero397": {"circuit_present":True,"zeroed_raw397_dimensions":list(range(397))},
            "shuffled": {"circuit_present":True,"zeroed_raw397_dimensions":[],
                "protocol":"independent permutation within each existing task, random.Random(42+task_index), frozen before inference"}}:
        raise ValueError("locked five-baseline condition protocol differs")
    model, scales, sites, checkpoint_sha, capture_sha, capture_manifest = load_sae(
        block, source, source_path, target)
    g01 = json.loads((SOURCE / "g01_smoke.json").read_text())
    if (g01.get("complete") is not True or g01.get("all_noop_checks_passed") is not True
            or g01.get("checkpoint_sha256") != target["checkpoint_sha256"]
            or g01.get("slot_count") != len(SLOTS)):
        raise ValueError("existing R03 no-op and 272-slot audit differs")
    banks = {"T_nested_or": load_eval_bank(COMPOSITION_BANK),
             "S_or_l": load_eval_bank(ATOMIC_BANK),
             "S_or_r": load_eval_bank(ATOMIC_BANK)}
    identity = {
        "format": "lean_lc026_r03_block_sae_fullslot_condition_fidelity_v2",
        "block": block, "tasks": list(TASKS), "sites": sites,
        "conditions": {"on": {"circuit_present": True, "zeroed_dimensions": []},
                       "off": {"circuit_present": False, "zeroed_dimensions": []},
                       "zero12": {"circuit_present": True,
                                  "zeroed_dimensions": list(ZERO12)}},
        "slot_indices": list(SLOTS), "feature_count": FEATURES,
        "feature_address": "(block, site, feature_id); IDs are unrelated across blocks",
        "r03_checkpoint_sha256": target["checkpoint_sha256"],
        "runtime_config_path": str(runtime_config_path.relative_to(ROOT)),
        "runtime_config_sha256": runtime_config_sha,
        "sae_checkpoint_sha256": checkpoint_sha,
        "sae_capture_manifest_sha256": capture_sha,
        "question_manifest_sha256": sha(source_path),
        "question_list_sha256": source["questions_sha256"],
        "bank_sha256": {"composition": source["composition_bank_sha256"],
                        "atomic": source["atomic_bank_sha256"]},
        "slot_map_sha256": capture_manifest["slot_map_sha256"],
        "activation_units": "mean over questions and all 272 slots of ReLU coefficient times decoder L2 norm times site RMS",
        "fidelity_units": "site-RMS-normalized squared error and target energy, summed across questions, slots and 1024 coordinates",
        "precision": "PG outputs captured in float16 as in v2 training; SAE forward uses CUDA bfloat16 autocast; fidelity accumulated in float64",
        "cohorts": list(COHORTS),
        "direction_agreement": "sign of per-slot or per-question mean physical response agrees with the aggregate signed mean; zero responses remain in all-item denominator; nonzero denominator saved separately",
        "atomic_pairing": "independent existing atomic questions; no same-mother pairing",
        "question_creation": False,
    }
    root = LOG / ("condition_smoke" if limit is not None else "condition") / f"block{block:02d}"
    data_root = DATA / ("condition_smoke" if limit is not None else "condition") / f"block{block:02d}"
    adapter = load_adapter(target, root=ROOT, device=torch.device("cuda"))
    try:
        mapping = slot_map(adapter.hypernet.parameter_generator)
        if (mapping != g01["slot_map"]
                or sha_bytes(canonical_json(mapping)) != capture_manifest["slot_map_sha256"]):
            raise ValueError("current 272-slot LoRA endpoint mapping differs")
        noop = no_op_checks(adapter, banks, rows, block, capture_manifest)
        if len(noop) != len(TASKS) * len(CONDITIONS):
            raise ValueError("condition no-op coverage incomplete")
        decoder_norms = model.W_dec.float().norm(dim=1)
        planned_shards = range(1) if limit is not None else range(SHARDS_PER_TASK)
        expected_names = [f"{task}/part{shard:02d}" for task in TASKS
                          for shard in planned_shards]
        for task in TASKS:
            bank = banks[task]
            for shard in planned_shards:
                questions = rows[task][shard * SHARD_QUESTIONS:(shard + 1) * SHARD_QUESTIONS]
                if limit is not None:
                    questions = questions[:limit]
                if not questions:
                    continue
                expected = part_identity(identity, task, shard, questions)
                data_path, receipt_path = part_paths(root, data_root, task, shard)
                prior = load_part(data_path, receipt_path, expected)
                if prior is None:
                    stats = empty_stats()
                    for index, row in enumerate(questions, 1):
                        ids_cpu, vector_cpu = check_question(bank, row)
                        ids = ids_cpu.to("cuda", torch.long)
                        vector = vector_cpu.to("cuda")
                        captured = capture_conditions(adapter, ids, vector, block)
                        measure_one(stats, task, captured, model, scales, sites, decoder_norms, row["branch"])
                        if index % 10 == 0 or index == len(questions):
                            print("R03_FULLSLOT_CONDITION_PROGRESS", block, task, shard,
                                  index, len(questions), flush=True)
                    arrays = {key: value.cpu().numpy() for key, value in stats.items()}
                    write_npz(data_path, **arrays)
                    write_json(receipt_path, {"identity": expected,
                                             "path": str(data_path.relative_to(ROOT)),
                                             "sha256": sha(data_path),
                                             "bytes": data_path.stat().st_size})
                    load_part(data_path, receipt_path, expected)
                done = completed_parts(root, data_root)
                write_json(root / "status.json", {"format": identity["format"],
                           "complete": False, "block": block,
                           "completed_parts": done,
                           "pending_parts": [name for name in expected_names if name not in set(done)],
                           "smoke_limit_per_task_shard": limit,
                           "no_op_checks": noop, "identity": identity})
    finally:
        adapter.close()
    if limit is not None:
        print("R03_FULLSLOT_CONDITION_SMOKE_COMPLETE", block, flush=True)
        return

    parts = {}
    for task in TASKS:
        for shard in range(SHARDS_PER_TASK):
            questions = rows[task][shard * SHARD_QUESTIONS:(shard + 1) * SHARD_QUESTIONS]
            expected = part_identity(identity, task, shard, questions)
            data_path, receipt_path = part_paths(root, data_root, task, shard)
            arrays = load_part(data_path, receipt_path, expected)
            if arrays is None:
                raise ValueError(f"fullslot condition part is missing: {task}/{shard}")
            parts[(task, shard)] = arrays
    populations = {
        "all500": (list(range(SHARDS_PER_TASK)), 500),
        "train300": (list(range(SPLITS["train"][1] // SHARD_QUESTIONS)), 300),
        "last200": (list(range(SPLITS["train"][1] // SHARD_QUESTIONS, SHARDS_PER_TASK)), 200),
    }
    summary_paths = {}
    for name, (shards, count) in populations.items():
        total = accumulate([parts[(task, shard)] for task in TASKS for shard in shards])
        output = summary_arrays(total, sites, identity, name, count)
        path = root / f"{name}.npz"
        write_npz(path, **output)
        summary_paths[name] = {"path": str(path.relative_to(ROOT)), "sha256": sha(path)}
    with np.load(root / "all500.npz", allow_pickle=False) as summary:
        fidelity = {site: {"on_nmse_by_task": summary["on_reconstruction_nmse"][:, i].tolist(),
                           "on_minus_off_nmse_by_task": summary["on_minus_off_nmse"][:, i].tolist(),
                           "on_minus_zero12_nmse_by_task": summary["on_minus_zero12_nmse"][:, i].tolist()}
                    for i, site in enumerate(sites)}
    # Replace undefined zero-energy ratios with JSON null; their raw SSE and
    # energy remain in the NPZ rather than being silently treated as zero.
    fidelity = json.loads(json.dumps(fidelity, allow_nan=True),
                          parse_constant=lambda _value: None)
    write_json(root / "status.json", {"format": identity["format"],
               "complete": True, "block": block,
               "completed_parts": expected_names, "pending_parts": [],
               "questions_per_task": {task: 500 for task in TASKS},
               "no_op_checks": noop,
               "summaries": summary_paths, "site_fidelity": fidelity,
               "identity": identity})
    print("R03_FULLSLOT_CONDITION_COMPLETE", block, fidelity, flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--block", type=int, required=True, choices=range(12))
    parser.add_argument("--limit", type=int, default=None,
                        help="program smoke on the first N questions of each task; writes isolated outputs")
    args = parser.parse_args()
    if args.limit is not None and not 1 <= args.limit <= SHARD_QUESTIONS:
        parser.error(f"--limit must be in 1..{SHARD_QUESTIONS}")
    run(args.block, args.limit)


if __name__ == "__main__":
    main()
