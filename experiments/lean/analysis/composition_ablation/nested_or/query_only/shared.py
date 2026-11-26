"""Frozen Query-only Q12 identities, approved banks, and isolated artifacts."""
from __future__ import annotations

from collections import Counter
import gzip
import hashlib
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
LOG = ROOT / "logs/eval/2026-11-14/lean-stage2-nested-or-query-q12-v1"
OUT = ROOT / "outputs/pg_transcoder/lean_stage2_nested_or_query_q12_v1"
TASKS = ("T_nested_or", "T_and_or", "T_or_and", "T_and_or_and", "T_nested_and",
         "S_and", "S_or_l", "S_or_r", "S_id")
BLOCKS, WIDTH, FEATURES = (0, 5, 10), 1024, 8192
CHECKPOINT = "205ed7d489cf392aed4df42d962f1a639bccbc0bff949792c26ea941b68fd5d7"


def sha(path):
    with Path(path).open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False)+"\n")
    os.replace(tmp, path)


def freeze(path, value):
    if path.exists():
        assert json.loads(path.read_text()) == value, f"Frozen artifact differs: {path}"
    else:
        write(path, value)


def inputs():
    p = json.loads((HERE / "protocol.json").read_text())
    a = json.loads((LOG / "data_authorization.json").read_text())
    assert p["checkpoint_sha256"] == CHECKPOINT and p["condition_source"] == "query_only"
    assert p["circuit_token_count"] == 0 and p["generation"]["max_new_tokens"] == 50
    assert p["gate_multiplier"] == 0.5 and p["external_lora_multiplier"] == 1.0
    assert p["blocks"] == list(BLOCKS)
    assert a["approved"] and a["banks"] == p["banks"] and a["totals"] == p["totals"]
    assert p["data_authorization_sha256"] == sha(LOG / "data_authorization.json")
    assert p["totals"] == {t: 500 for t in TASKS} and p["scope"] == "visible500_discovery_only"
    for path_key, hash_key in (("checkpoint_path", "checkpoint_sha256"), ("config_path", "config_file_sha256"),
                                ("base_path", "base_sha256"), ("tokenizer_path", "tokenizer_sha256")):
        assert sha(ROOT / p[path_key]) == p[hash_key], path_key
    for label, b in p["banks"].items():
        assert b["path"].endswith("/visible_validation.pt" if label == "visible" else "/atomic_retention.pt")
        assert sha(ROOT / b["path"]) == b["sha256"]
    for name, digest in p["native_code_sha256"].items():
        assert sha(ROOT / name) == digest, f"Native runtime changed: {name}"
    assert p["questions_sha256"] == sha(HERE / "questions.json")
    rows = json.loads((HERE / "questions.json").read_text())["rows"]
    assert len(rows) == len({(q["task"], q["index"]) for q in rows}) == len({q["event_key"] for q in rows}) == 4500
    assert Counter(q["task"] for q in rows) == p["totals"]
    for task in TASKS:
        assert [q["task_index"] for q in rows if q["task"] == task] == list(range(500))
    return p, rows


def identity(phase, unit, parts):
    return {"phase": phase, "unit": unit, "parts": parts, "checkpoint_sha256": CHECKPOINT,
            "protocol_sha256": sha(HERE / "protocol.json"),
            "code_sha256": {p.name: sha(p) for p in HERE.glob("*.py")}}


def check_receipt(receipt, phase, unit, parts):
    assert receipt["complete"] and receipt["identity"] == identity(phase, unit, parts)


def references():
    summary = json.loads((LOG / "baseline_summary.json").read_text())
    assert summary["complete"] and sha(LOG / "baseline_summary.json") == json.loads((HERE / "protocol.json").read_text())["baseline_record_sha256"]
    result = {}
    for r in summary["receipts"]:
        assert sha(ROOT / r["path"]) == r["sha256"]
        with gzip.open(ROOT / r["path"], "rt") as f:
            for line in f:
                row = json.loads(line); q = row["question"]; key = q["task"], q["index"]
                assert key not in result
                result[key] = row
    assert len(result) == 4500
    return result


def screen_questions(rows):
    return [q for t in TASKS for q in rows if q["task"] == t and q["task_index"] < (200 if t == TASKS[0] else 50)]
