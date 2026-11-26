#!/usr/bin/env python3
"""Lock the nine 500-question banks for Query-only native PG activations."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

from prepare import CHECKPOINT_SHA, HERE, ROOT, TASKS as BANK_TASKS, sha, write

TASKS = (*BANK_TASKS["composition"], *BANK_TASKS["atomic"])
SOURCE = HERE / "block_source"
VISIBLE = ROOT / "outputs/s5_lean_circuit/composition_loco_sft_v1/eval/visible_validation.pt"
ATOMIC = ROOT / "outputs/s5_lean_circuit/composition_loco_sft_v1/eval/atomic_retention.pt"
PRIOR = ROOT / "4server/lean_stage2_and_or_groups_c25_dev500_9task_v1/block_source"


def main() -> None:
    target = json.loads((HERE / "target.json").read_text())
    approved = json.loads((HERE / "source_manifest.json").read_text())
    previous = json.loads((PRIOR / "manifest.json").read_text())
    questions = PRIOR / "questions.jsonl"
    if (target["checkpoint_sha256"] != CHECKPOINT_SHA
            or approved["checkpoint_sha256"] != CHECKPOINT_SHA
            or approved["excluded_final_test"] is not True
            or previous["format"] != "lean_stage2_c25_block_source_v1"
            or previous["complete"] is not True
            or previous["questions_sha256"] != sha(questions)
            or previous["tasks"] != list(TASKS)
            or previous["questions_per_task"] != 500
            or previous["composition_bank_sha256"] != approved["banks"]["composition"]["sha256"]
            or previous["atomic_bank_sha256"] != approved["banks"]["atomic"]["sha256"]
            or sha(VISIBLE) != previous["composition_bank_sha256"]
            or sha(ATOMIC) != previous["atomic_bank_sha256"]):
        raise ValueError("Query-only source identity differs from locked nine-task banks")
    SOURCE.mkdir(parents=True, exist_ok=True)
    local = SOURCE / "questions.jsonl"
    temporary = local.with_name(f".{local.name}.{os.getpid()}.tmp")
    temporary.write_bytes(questions.read_bytes())
    os.replace(temporary, local)
    if sha(local) != previous["questions_sha256"]:
        raise ValueError("Query-only copied question identity differs")
    rows = [json.loads(line) for line in local.read_text().splitlines()]
    if (len(rows) != 4500 or len({r["event_key"] for r in rows}) != 4500
            or {t: sum(r["task_type"] == t for r in rows) for t in TASKS}
               != {t: 500 for t in TASKS}):
        raise ValueError("Query-only question membership differs")
    write(SOURCE / "manifest.json", {
        "format": "lean_stage2_query_m12_block_source_v1", "complete": True,
        "question_creation": False, "checkpoint_sha256": CHECKPOINT_SHA,
        "source_manifest_sha256": sha(HERE / "source_manifest.json"),
        "prior_questions_sha256": previous["questions_sha256"],
        "questions_path": str(local.relative_to(ROOT)), "questions_sha256": sha(local),
        "questions_per_task": 500, "question_count": 4500, "tasks": list(TASKS),
        "composition_bank_sha256": sha(VISIBLE), "atomic_bank_sha256": sha(ATOMIC),
        "composition_bank_identity_sha256": approved["banks"]["composition"]["metadata_identity_sha256"],
        "atomic_bank_identity_sha256": approved["banks"]["atomic"]["metadata_identity_sha256"],
        "prior_source_manifest_sha256": sha(PRIOR / "manifest.json"),
    })
    print("QUERY_M12_BLOCK_SOURCE_PREPARED", len(rows), flush=True)


if __name__ == "__main__":
    main()
