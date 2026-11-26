#!/usr/bin/env python3
"""Lock C25 dev500-only inputs without reading the final-test bank."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
OUT = ROOT / "outputs/pg_transcoder/lean_stage2_and_or_groups_c25_dev500_9task_v1"
for source in (ROOT / "s5_lean_circuit/src", ROOT / "s2_hypernet/src"):
    sys.path.insert(0, str(source))
from s5_lean_circuit.composition_loco import load_eval_bank

RUN = ROOT / "outputs/s5_lean_circuit/composition_loco_circuit_retention20_v2/runs/T_and_or/C25/circuit_a"
CONFIG = ROOT / "s5_lean_circuit/configs/composition_loco_circuit_retention20_v2/T_and_or/C25/circuit_a.json"
CHECKPOINT = RUN / "checkpoints/model_step012000.pt"
CHECKPOINT_SHA = "36244f86e609b4aacea9ccd1e4a24c5a96a332aefd12ec54143a1ddbc64a8b72"
BASE = ROOT / "outputs/s5_lean_circuit/atomic_only_dense_v1/dense-bgoff_attempt_20260820_200505/best_model.pt"
TOKENIZER = ROOT / "datasets/s5_lean_circuit/tokenizer_v1/tokenizer.model"
VISIBLE = ROOT / "outputs/s5_lean_circuit/composition_loco_sft_v1/eval/visible_validation.pt"
ATOMIC = ROOT / "outputs/s5_lean_circuit/composition_loco_sft_v1/eval/atomic_retention.pt"
SPARSE = ROOT / "outputs/pg_transcoder/lean-c25-sparse-features-v1/models/sae/best.pt"
CAPTURE_MANIFEST = ROOT / "logs/eval/2026-11-02/lean-c25-sparse-features-v1/capture-manifest.json"
OLD_CANDIDATES = ROOT / "4server/lean_stage2_and_or_groups_v1/candidate_groups.json"
OLD_TARGET = ROOT / "4server/lean_stage2_and_or_groups_v1/target.json"
TASKS = {"composition": ("T_and_or", "T_or_and", "T_and_or_and", "T_nested_and", "T_nested_or"),
         "atomic": ("S_and", "S_or_l", "S_or_r", "S_id")}


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def bank_plan(name: str, bank, expected_identity: dict) -> dict:
    path = VISIBLE if name == "composition" else ATOMIC
    if (bank.metadata["expected_source_base_identity"] != expected_identity["expected_source_base_identity"]
            or bank.metadata["input_treatment"] != expected_identity["input_treatment"]
            or bank.metadata["tokenizer_sha256"] != sha(TOKENIZER)):
        raise ValueError(f"{name} dev500 Base Circuit/tokenizer identity differs")
    selected = {task: [] for task in TASKS[name]}
    keys = set()
    by_length = {}
    for index, record in enumerate(bank.records):
        if record.task_type not in selected:
            continue
        selected[record.task_type].append(index)
        key = bytes(bank.payload["event_key"][index].tolist())
        if key in keys:
            raise ValueError(f"duplicate {name} event key")
        keys.add(key)
        vector = bank.payload["vectors"][index]
        if vector.numel() != 397 or not bool(torch.isfinite(vector).all()):
            raise ValueError(f"invalid {name} 397D Circuit vector")
        length = int(bank.payload["query_lengths"][index])
        by_length.setdefault(length, []).append(index)
    counts = {task: len(indices) for task, indices in selected.items()}
    if counts != {task: 500 for task in TASKS[name]}:
        raise ValueError(f"incomplete dev500 bank: {counts}")
    batches = [indices[start:start + 16] for _, indices in sorted(by_length.items())
               for start in range(0, len(indices), 16)]
    if sorted(i for batch in batches for i in batch) != sorted(i for indices in selected.values() for i in indices):
        raise ValueError("dev500 batch coverage differs")
    return {"path": str(path.relative_to(ROOT)), "sha256": sha(path),
            "metadata_identity_sha256": bank.metadata["identity_sha256"],
            "tasks": counts, "event_keys_sha256": hashlib.sha256(b"".join(sorted(keys))).hexdigest(),
            "batch_size": 16, "batches": batches}


def main() -> None:
    final = json.loads((RUN / "final.json").read_text())
    config = json.loads(CONFIG.read_text())
    if sha(CHECKPOINT) != CHECKPOINT_SHA:
        raise ValueError("C25 checkpoint identity differs")
    archive = torch.load(CHECKPOINT, map_location="cpu", weights_only=False, mmap=True)
    if (final["checkpoint"]["sha256"] != CHECKPOINT_SHA
            or final["checkpoint"]["global_step"] != 12000
            or final["config_sha256"] != archive["config_sha256"]
            or sha(CONFIG) != "566b8bbdb2142574e009bba7d93e1b594d2978fadc035ca8a2828ac97cf9827b"
            or sha(CHECKPOINT) != CHECKPOINT_SHA
            or sha(BASE) != config["base"]["checkpoint_sha256"]
            or sha(TOKENIZER) != final["final_bank_identity"]["tokenizer_sha256"]
            or config["model"]["condition_source"] != "circuit"
            or config["model"]["gate_max"] != .5
            or config["evaluation"]["max_new_tokens"] != 50
            or config["evaluation"]["metric"] != "strict_first_tactic_exact_match"):
        raise ValueError("C25 model/config/Base/tokenizer/score identity differs")
    composition = load_eval_bank(VISIBLE)
    atomic = load_eval_bank(ATOMIC)
    banks = {"composition": bank_plan("composition", composition, final["final_bank_identity"]),
             "atomic": bank_plan("atomic", atomic, final["atomic_bank_identity"])}
    if sum(sum(info["tasks"].values()) for info in banks.values()) != 4500:
        raise ValueError("dev500 protocol requires exactly 4500 selected questions")
    candidates = json.loads(OLD_CANDIDATES.read_text())
    capture = json.loads(CAPTURE_MANIFEST.read_text())
    sae = torch.load(SPARSE, map_location="cpu", weights_only=False, mmap=True)
    if (candidates["checkpoint_sha256"] != CHECKPOINT_SHA
            or candidates["dictionary_sha256"] != sha(SPARSE)
            or candidates["capture_manifest_sha256"] != sha(CAPTURE_MANIFEST)
            or capture["checkpoint_sha256"] != CHECKPOINT_SHA or not capture["complete"]
            or capture["limit_per_task"] != 500
            or sae["capture_manifest_sha256"] != sha(CAPTURE_MANIFEST)
            or sae["kind"] != "sae"):
        raise ValueError("C25-only SAE dictionary/capture identity differs")
    if any(g["identity"]["checkpoint_sha256"] != CHECKPOINT_SHA
           or g["identity"]["task"] not in {"T_and_or", "S_and", "S_or_l", "S_or_r", "S_id"}
           or "full500/" not in g["path"] or "final_test" in g["path"]
           for g in capture["groups"]):
        raise ValueError("C25 SAE capture provenance includes excluded data")
    if any("final_test" in str(group) or "2000" in str(group) for group in candidates["groups"]):
        raise ValueError("candidate inventory refers to excluded test bank")
    target = json.loads(OLD_TARGET.read_text())
    if target["checkpoint_sha256"] != CHECKPOINT_SHA or target["gate_multiplier"] != .5:
        raise ValueError("C25 target identity differs")
    write(HERE / "source_manifest.json", {
        "format": "lean_stage2_c25_dev500_source_v1", "protocol": "existing_dev500_nine_tasks",
        "excluded_final_test": True, "checkpoint_sha256": CHECKPOINT_SHA,
        "config_sha256": sha(CONFIG), "base_sha256": sha(BASE), "tokenizer_sha256": sha(TOKENIZER),
        "score": "greedy_max50_strict_first_tactic_exact_match", "gate": .5,
        "external_lora_multiplier": 1.0, "slot_count": 272, "banks": banks})
    write(HERE / "candidate_groups.json", candidates)
    write(HERE / "target.json", target)
    print("C25_DEV500_PREPARED", {name: info["tasks"] for name, info in banks.items()}, flush=True)


if __name__ == "__main__":
    main()
