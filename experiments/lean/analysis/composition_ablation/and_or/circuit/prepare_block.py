#!/usr/bin/env python3
"""Lock C25 visible-validation and atomic questions for independent full-slot PG SAEs."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
for p in (ROOT / "s5_lean_circuit/src", ROOT / "s2_hypernet/src", HERE):
    sys.path.insert(0, str(p))
from s5_lean_circuit.composition_loco import load_eval_bank
from prepare import CHECKPOINT_SHA, VISIBLE as COMPOSITION, ATOMIC, RUN, sha, write

VISIBLE = ROOT / "outputs/s5_lean_circuit/composition_loco_sft_v1/eval/visible_validation.pt"
TASKS = ("T_and_or", "T_or_and", "T_and_or_and", "T_nested_and", "T_nested_or",
         "S_and", "S_or_l", "S_or_r", "S_id")
SOURCE = HERE / "block_source"


def digest(blob: bytes) -> str:
    return hashlib.sha256(blob).hexdigest()


def main():
    final = json.loads((RUN / "final.json").read_text())
    if final["checkpoint"]["sha256"] != CHECKPOINT_SHA:
        raise ValueError("original C25 checkpoint differs")
    banks = {"composition": load_eval_bank(VISIBLE), "atomic": load_eval_bank(ATOMIC)}
    if (banks["atomic"].metadata["identity_sha256"] != final["atomic_bank_identity"]["identity_sha256"]
            or banks["composition"].metadata["expected_source_base_identity"]
               != final["final_bank_identity"]["expected_source_base_identity"]
            or banks["atomic"].metadata["expected_source_base_identity"]
               != final["atomic_bank_identity"]["expected_source_base_identity"]
            or banks["composition"].metadata["input_treatment"] != final["final_bank_identity"]["input_treatment"]
            or banks["composition"].metadata["tokenizer_sha256"] != final["final_bank_identity"]["tokenizer_sha256"]):
        raise ValueError("C25 SAE training bank treatment differs")
    rows = []
    seen = set()
    for task in TASKS:
        bank_name = "atomic" if task.startswith("S_") else "composition"
        bank = banks[bank_name]
        indices = [i for i, record in enumerate(bank.records) if record.task_type == task]
        if len(indices) != 500:
            raise ValueError(f"C25 SAE train {task} denominator differs")
        for ordinal, index in enumerate(indices):
            record = bank.records[index]
            key = bytes(bank.payload["event_key"][index].tolist()).hex()
            if key in seen:
                raise ValueError("C25 SAE source event key duplicated")
            seen.add(key)
            length = int(bank.payload["query_lengths"][index])
            vector = bank.payload["vectors"][index:index+1].contiguous()
            if vector.shape != (1,397) or not bool(torch.isfinite(vector).all()):
                raise ValueError("C25 SAE source Base Circuit397 malformed")
            rows.append({"task_type": task, "task_index": ordinal,
                         "canonical_global_index": index, "bank": bank_name,
                         "event_key": key, "prompt_sha256": digest(record.input_text.encode()),
                         "gold_output_sha256": digest(record.output_text.encode()),
                         "gold_first_tactic": getattr(record, "first_tactic", None),
                         "query_length": length,
                         "raw397_model_input_sha256": digest(vector.numpy().tobytes())})
    if len(rows) != 4500:
        raise ValueError("C25 SAE source needs 4500 visible/atomic questions")
    SOURCE.mkdir(parents=True, exist_ok=True)
    questions = SOURCE / "questions.jsonl"
    tmp = questions.with_name(f".{questions.name}.{os.getpid()}.tmp")
    with tmp.open("w") as stream:
        for row in rows:
            stream.write(json.dumps(row, sort_keys=True) + "\n")
    os.replace(tmp, questions)
    write(SOURCE / "manifest.json", {"format": "lean_stage2_c25_block_source_v1",
          "complete": True, "question_creation": False,
          "checkpoint_sha256": CHECKPOINT_SHA,
          "questions_path": str(questions.relative_to(ROOT)),
          "questions_sha256": sha(questions), "questions_per_task": 500,
          "question_count": len(rows), "tasks": list(TASKS),
          "composition_bank_path": str(VISIBLE.relative_to(ROOT)),
          "composition_bank_sha256": sha(VISIBLE),
          "composition_bank_identity_sha256": banks["composition"].metadata["identity_sha256"],
          "atomic_bank_path": str(ATOMIC.relative_to(ROOT)),
          "atomic_bank_sha256": sha(ATOMIC),
          "atomic_bank_identity_sha256": banks["atomic"].metadata["identity_sha256"]})
    print("C25_BLOCK_SOURCE_PREPARED", len(rows), flush=True)


if __name__ == "__main__":
    main()
