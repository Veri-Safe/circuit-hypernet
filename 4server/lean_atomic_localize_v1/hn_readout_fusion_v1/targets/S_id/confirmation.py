#!/usr/bin/env python3
"""S_id immutable freeze, genuinely fresh atomic bank, and formal result audit.

Importing this module never generates data or opens a model.  ``generate`` is
available only after a complete final_lock exists; a failed attempt never frees
its reserved seed interval.  Model execution belongs to the target launcher.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import fcntl
from fractions import Fraction
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import random
import re
import subprocess
import sys
import tempfile

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[4]
PROTOCOL_DIR = HERE.parents[1]
TASKS = ("S_and", "S_or_l", "S_or_r", "S_id")
TARGET = "S_id"
PROTOCOL_ID = "lean_atomic_hn_readout_fusion_v1"
DATA_ROOT = ROOT / "logs/data/2026-11-14/lean-atomic-hn-readout-fusion-v1/S_id"
ANALYSIS_ROOT = ROOT / "logs/analysis/2026-11-14/lean-atomic-hn-readout-fusion-v1/S_id"
SEED_START, SEED_END, RESERVATION_SIZE = 60419000000, 60420000000, 10000
LABELS = ("base_original", "base_direct", "base_joint", "hn_original", "hn_suppressed")
BASE_SHA = "ee2a7efdf77262ee1fbdf7d2080f930c2bedc27e45b7712537fba964ee05931c"
HN_SHA = "d948792b7206d36f1a06697428fcb928aefaf7db120558529fae3c38ab957cde"


def utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024**2), b""):
            h.update(block)
    return h.hexdigest()


def canonical(value: object) -> bytes:
    return (json.dumps(value, sort_keys=True, ensure_ascii=False, indent=2,
                       allow_nan=False) + "\n").encode()


def immutable_bytes(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() != content:
            raise FileExistsError(f"immutable artifact differs: {path}")
        return
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="wb", dir=path.parent,
                                         prefix=f".{path.name}.", delete=False) as stream:
            temporary = Path(stream.name)
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError:
            if path.read_bytes() != content:
                raise FileExistsError(f"immutable artifact differs: {path}")
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def write(path: Path, value: object) -> None:
    immutable_bytes(path, canonical(value))


def _path(path: str | Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else ROOT / path


def _relative(path: Path) -> str:
    return str(path.resolve().relative_to(ROOT))


def _base_ids(group: list) -> set[tuple[int, int]]:
    result = set()
    for item in group:
        if isinstance(item, str):
            layer, feature = item.split(":F")
            pair = int(layer.removeprefix("L")), int(feature)
        elif isinstance(item, dict):
            pair = int(item["layer"]), int(item["feature_id"])
        else:
            pair = int(item[0]), int(item[1])
        if not 0 <= pair[0] < 8 or not 0 <= pair[1] < 16384:
            raise ValueError("invalid Base dictionary feature identity")
        result.add(pair)
    return result


def validate_lock_spec(spec: dict) -> None:
    required = ("models", "dictionaries", "hn_group", "readout", "candidate_pool",
                "K", "base_direct", "base_joint", "base_position_rule", "alpha",
                "combination_rule", "search_budget", "data_generation")
    missing = [name for name in required if name not in spec or spec[name] is None]
    if missing:
        raise ValueError(f"incomplete final lock: {missing}")
    if spec.get("target") != TARGET or spec.get("protocol_id") != PROTOCOL_ID:
        raise ValueError("wrong target/protocol")
    if spec["alpha"] != 0 or not spec["base_position_rule"]:
        raise ValueError("final Base intervention must fix alpha=0 and positions")
    k = int(spec["K"])
    pool = _base_ids(spec["candidate_pool"])
    if k <= 0 or len(pool) < k:
        raise ValueError("invalid K/candidate pool")
    for name in ("base_direct", "base_joint"):
        ids = _base_ids(spec[name])
        if len(spec[name]) != k or len(ids) != k or not ids <= pool:
            raise ValueError(f"{name} must have K unique identities from common pool")
    if not spec["hn_group"] or not spec["dictionaries"] or not spec["readout"]:
        raise ValueError("HN identities, dictionaries and readout must be frozen")
    plan = spec["data_generation"]
    if (plan.get("count_per_task") != 100 or plan.get("n_distractors") != 2
            or plan.get("seed_start") != SEED_START or plan.get("seed_end_exclusive") != SEED_END):
        raise ValueError("confirmation generation settings differ from protocol")


def freeze_pipeline(path: Path, spec: dict, code_paths: list[Path]) -> dict:
    """Seal a fully specified pipeline. No final bank is opened by this call."""
    path = Path(path)
    validate_lock_spec(spec)
    if path.exists():
        lock = verify_lock(path)
        if any(lock.get(k) != v for k, v in spec.items()):
            raise ValueError("existing lock differs; use a new development attempt")
        return lock
    if re.fullmatch(r"attempt_\d{3}", path.parent.name):
        bank_dir = DATA_ROOT / path.parent.name
        if any((bank_dir / name).exists() for name in ("questions.jsonl", "generation_transaction.json", "identity.json")):
            raise ValueError("attempt already contains confirmation material; cannot freeze retrospectively")
    if not code_paths:
        raise ValueError("freeze requires actual implementation code files")
    paths = list(dict.fromkeys([Path(p).resolve() for p in code_paths] + [Path(__file__).resolve()]))
    protocol_files = [PROTOCOL_DIR / x for x in ("PROTOCOL.md", "SEARCH.md", "ASSETS.md", "protocol.json", "prompts/S_id.md")]
    lock = dict(spec)
    lock.update(format="hn_readout_fusion_s_id_final_lock_v1", frozen_at_utc=utc(),
                git_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
                code_sha256={_relative(p): sha(p) for p in paths},
                protocol_sha256={_relative(p): sha(p) for p in protocol_files},
                confirmation_generated_after_lock=True)
    write(path, lock)
    immutable_bytes(path.with_suffix(".sha256"), (sha(path) + "\n").encode())
    path.chmod(0o444)
    return lock


def verify_lock(path: Path, *, verify_code: bool = True) -> dict:
    path = Path(path)
    value = json.loads(path.read_text())
    validate_lock_spec(value)
    if path.with_suffix(".sha256").read_text().strip() != sha(path):
        raise ValueError("final_lock seal changed")
    if not value.get("frozen_at_utc") or not value.get("confirmation_generated_after_lock"):
        raise ValueError("lock lacks temporal provenance")
    if verify_code:
        for key in ("code_sha256", "protocol_sha256"):
            for name, expected in value[key].items():
                if sha(_path(name)) != expected:
                    raise ValueError(f"frozen source changed: {name}")
    return value


def verify_development_evidence(lock: dict) -> None:
    """Verify potentially large evidence once at formal phase boundaries."""
    evidence = lock.get("development_evidence")
    if not isinstance(evidence, dict) or not evidence:
        raise ValueError("final lock has no frozen development evidence")
    for path, expected in evidence.items():
        if sha(_path(path)) != expected:
            raise ValueError(f"frozen development evidence changed: {path}")


def append_history(event: dict) -> None:
    """Durable append-only audit; repeated completion events remain visible."""
    DATA_ROOT.mkdir(parents=True, exist_ok=True)
    with (DATA_ROOT / "attempt_history.jsonl").open("a+") as stream:
        fcntl.flock(stream, fcntl.LOCK_EX)
        stream.write(json.dumps({"recorded_at_utc": utc(), **event}, sort_keys=True, allow_nan=False) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def reserve_interval(attempt: str, lock_sha: str) -> dict:
    if not re.fullmatch(r"attempt_\d{3}", attempt):
        raise ValueError("attempt must use attempt_NNN namespace")
    DATA_ROOT.mkdir(parents=True, exist_ok=True)
    with (DATA_ROOT / "attempt_history.jsonl").open("a+") as stream:
        fcntl.flock(stream, fcntl.LOCK_EX)
        stream.seek(0)
        rows = [json.loads(line) for line in stream if line.strip()]
        reservations = [x for x in rows if x["event"] == "seed_interval_reserved"]
        existing = [x for x in reservations if x["attempt"] == attempt]
        if existing:
            if len(existing) != 1 or existing[0]["final_lock_sha256"] != lock_sha:
                raise ValueError("attempt is already bound to another final lock")
            return existing[0]
        start = max([x["seed_end_exclusive"] for x in reservations] + [SEED_START])
        end = start + RESERVATION_SIZE
        if end > SEED_END:
            raise RuntimeError("S_id seed range exhausted")
        record = {"event": "seed_interval_reserved", "attempt": attempt,
                  "final_lock_sha256": lock_sha, "seed_start": start,
                  "seed_end_exclusive": end, "recorded_at_utc": utc()}
        stream.seek(0, os.SEEK_END)
        stream.write(json.dumps(record, sort_keys=True) + "\n")
        stream.flush()
        os.fsync(stream.fileno())
        return record


def _load_shared(name: str):
    for location in (ROOT / "s5_lean_circuit/src", ROOT / "s2_hypernet/src", ROOT / "4server/lean_atomic_localize_v1/shared"):
        if str(location) not in sys.path:
            sys.path.insert(0, str(location))
    path = ROOT / "4server/lean_atomic_localize_v1/shared" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(f"s_id_fusion_reuse_{name}", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _forbidden(path: Path) -> bool:
    name = str(path).lower()
    return any(x in name for x in ("final_test", "2000_per_family", "2000-per-family", "composition"))


def _known_banks(exclusion_paths: list[Path] | None, own_bank: Path) -> list[Path]:
    # All JSONL names are considered, including historical samples/entries files.
    # Revoked composition banks are never opened, including for deduplication.
    paths = set()
    for root in (ROOT / "logs/data", ROOT / "logs/analysis"):
        for path in root.rglob("*.jsonl"):
            paths.add(path.resolve())
        registry_names = ("identity", "manifest", "preregister", "protocol", "frozen", "ready")
        for registry in root.rglob("*.json"):
            if _forbidden(registry) or not any(name in registry.stem.lower() for name in registry_names):
                continue
            def find_references(value, key=""):
                if isinstance(value, dict):
                    for name, item in value.items():
                        yield from find_references(item, name)
                elif isinstance(value, list):
                    for item in value:
                        yield from find_references(item, key)
                elif isinstance(value, str) and value.endswith(".jsonl"):
                    # Registries also name predictions; including those is safe
                    # and preserves prompt hashes from banks since moved away.
                    path = _path(value)
                    if not path.exists() and "/HyperCombine/" in value:
                        path = ROOT / value.split("/HyperCombine/", 1)[1]
                    if path.exists():
                        yield path.resolve()
                    elif any(token in key.lower() for token in ("question", "bank", "validation", "holdout")) and not _forbidden(path):
                        raise FileNotFoundError(f"registered old bank unavailable for exact exclusion: {registry}: {value}")
            paths.update(find_references(json.loads(registry.read_text())))
    paths.update(_path(p).resolve() for p in (exclusion_paths or []))
    return sorted(p for p in paths if own_bank.parent.resolve() not in p.parents and not _forbidden(p))


def _scan_hashes(path: Path, known: set[str]) -> dict:
    count, hash_count = 0, 0
    with path.open() as stream:
        for line in stream:
            if not line.strip():
                continue
            row = json.loads(line)
            count += 1
            if not isinstance(row, dict):
                continue
            prompt_hash = row.get("prompt_sha256", row.get("input_text_sha256"))
            prompt = row.get("prompt_text", row.get("input_text"))
            if prompt is not None:
                computed = hashlib.sha256(prompt.encode()).hexdigest()
                if prompt_hash is not None and computed != prompt_hash:
                    raise ValueError(f"stored prompt hash mismatch: {path}:{count}")
                prompt_hash = computed
            if prompt_hash:
                known.add(prompt_hash)
                hash_count += 1
    return {"path": _relative(path), "sha256": sha(path), "rows": count, "prompt_hash_rows": hash_count}


def generate_confirmation(lock_path: Path, attempt_dir: Path,
                          exclusion_paths: list[Path] | None = None) -> list[dict]:
    """Use exact released atomic generator; never uses model outputs to select."""
    lock_path, attempt_dir = Path(lock_path), Path(attempt_dir)
    lock = verify_lock(lock_path)
    if attempt_dir.parent.resolve() != DATA_ROOT.resolve():
        raise ValueError("fresh bank must use this target's logs/data namespace")
    if os.environ.get("PYTHONHASHSEED") != "42":
        raise ValueError("released atomic renderer requires PYTHONHASHSEED=42")
    bank = attempt_dir / "questions.jsonl"
    if bank.exists() and (attempt_dir / "identity.json").exists():
        return load_confirmation(lock_path, attempt_dir)
    transaction = attempt_dir / "generation_transaction.json"
    if transaction.exists():
        pending = json.loads(transaction.read_text())
        if pending["identity"]["final_lock_sha256"] != sha(lock_path):
            raise ValueError("pending bank transaction belongs to another freeze")
        content = "".join(json.dumps(x, sort_keys=True, ensure_ascii=False) + "\n" for x in pending["rows"]).encode()
        if hashlib.sha256(content).hexdigest() != pending["identity"]["questions_sha256"]:
            raise ValueError("pending bank transaction hash differs")
        immutable_bytes(bank, content)
        write(attempt_dir / "identity.json", pending["identity"])
        append_history({"event": "fresh_bank_recovered_after_interruption", "attempt": attempt_dir.name,
                        "questions_sha256": sha(bank), "final_lock_sha256": sha(lock_path)})
        return load_confirmation(lock_path, attempt_dir)
    reservation = reserve_interval(attempt_dir.name, sha(lock_path))
    prepared = _load_shared("prepare_cpu")
    from s5_lean_circuit.circuit.model_io import load_tokenizer
    manifest = json.loads((prepared.CORPUS / "manifest.json").read_text())
    if (manifest.get("status") != "complete" or manifest["event_count"] != 1_200_000
            or manifest["samples_sha256"] != sha(prepared.CORPUS / "samples.jsonl")
            or manifest["task_counts"] != {t: 300000 for t in TASKS}):
        raise ValueError("atomic training corpus identity differs")
    plan = lock["data_generation"]
    if plan.get("renderer_version") != manifest["renderer_version"]:
        raise ValueError("frozen renderer differs from original atomic corpus")
    target = json.loads(prepared.SOURCE_TARGET.read_text())
    tokenizer = load_tokenizer(prepared.TOKENIZER, expected_sha256=target["tokenizer_sha256"])
    known, sources = set(), []
    sources.append(_scan_hashes(prepared.CORPUS / "samples.jsonl", known))
    for path in _known_banks(exclusion_paths, bank):
        sources.append(_scan_hashes(path, known))
    known_count = len(known)
    rows, rejected = [], []
    stride = RESERVATION_SIZE // len(TASKS)
    for index, task in enumerate(TASKS):
        start = reservation["seed_start"] + index * stride
        accepted = 0
        for seed in range(start, start + stride):
            if accepted == 100:
                break
            row = prepared.make_row(task, seed, "fresh_confirmation", tokenizer,
                                    source_index=None, pilot=False)
            if row["prompt_sha256"] in known:
                rejected.append({"task_type": task, "seed": seed,
                                 "prompt_sha256": row["prompt_sha256"], "reason": "known_exact_prompt"})
                continue
            row.update(attempt=attempt_dir.name, final_lock_sha256=sha(lock_path))
            rows.append(row)
            known.add(row["prompt_sha256"])
            accepted += 1
        if accepted != 100:
            append_history({"event": "bank_generation_failed", "attempt": attempt_dir.name,
                            "reason": "seed_subinterval_exhausted", "task": task})
            raise RuntimeError("reserved subinterval exhausted; preserve it and start next attempt")
    content = "".join(json.dumps(x, sort_keys=True, ensure_ascii=False) + "\n" for x in rows).encode()
    manifest = {"format": "hn_readout_fusion_s_id_fresh_bank_v1", "generated_at_utc": utc(),
                "final_lock_sha256": sha(lock_path), "final_lock_frozen_at_utc": lock["frozen_at_utc"],
                "questions_sha256": hashlib.sha256(content).hexdigest(), "attempt": attempt_dir.name,
                "seed_reservation": reservation, "count_by_task": dict(Counter(r["task_type"] for r in rows)),
                "data_generation": plan, "known_prompt_hash_count_before_generation": known_count,
                "exclusion_sources": sources, "rejected_duplicates": rejected,
                "exact_prompt_overlap": 0, "revoked_banks_opened": False,
                "semantic_template_overlap_excluded": False,
                "source_generator_sha256": sha(ROOT / "4server/lean_atomic_localize_v1/shared/prepare_cpu.py")}
    # A single durable transaction permits exact recovery between bank/manifest
    # publication, even if other targets generate new banks meanwhile.
    write(transaction, {"rows": rows, "identity": manifest})
    immutable_bytes(bank, content)
    write(attempt_dir / "identity.json", manifest)
    append_history({"event": "fresh_bank_generated", "attempt": attempt_dir.name,
                    "final_lock_sha256": sha(lock_path), "questions_sha256": sha(bank),
                    "count": len(rows), "identity_sha256": sha(attempt_dir / "identity.json")})
    immutable_bytes(attempt_dir / "index.md", b"# S_id fresh confirmation bank\n\n- questions.jsonl: 100 new questions per task.\n- identity.json: freeze, seed reservation, and global exact-prompt exclusion.\n- native_graphs/: original Base graphs extracted once, after freeze.\n")
    immutable_bytes(attempt_dir / "summary.md", b"# S_id fresh bank preparation\n\n400 fresh questions generated after the immutable pipeline lock; no model-performance filtering. All known atomic bank exact prompt hashes and the complete atomic training corpus are excluded. Semantic-template novelty is not asserted.\n")
    return load_confirmation(lock_path, attempt_dir)


def load_confirmation(lock_path: Path, attempt_dir: Path) -> list[dict]:
    lock = verify_lock(lock_path)
    identity = json.loads((attempt_dir / "identity.json").read_text())
    bank = attempt_dir / "questions.jsonl"
    if identity["final_lock_sha256"] != sha(lock_path) or identity["questions_sha256"] != sha(bank):
        raise ValueError("confirmation bank identity changed")
    if datetime.fromisoformat(identity["generated_at_utc"]) < datetime.fromisoformat(lock["frozen_at_utc"]):
        raise ValueError("bank predates freeze")
    rows = [json.loads(line) for line in bank.open() if line.strip()]
    if len(rows) != 400 or Counter(r["task_type"] for r in rows) != {t: 100 for t in TASKS}:
        raise ValueError("formal bank must contain exactly 100/task")
    if len({r["question_id"] for r in rows}) != 400 or len({r["prompt_sha256"] for r in rows}) != 400:
        raise ValueError("formal bank has duplicate identity/prompt")
    return rows


def _without_extraction_timings(value):
    """Remove wall-clock measurements only; retain every scientific/cache field."""
    if isinstance(value, dict):
        return {key: _without_extraction_timings(item) for key, item in value.items()
                if key != "seconds" and not key.endswith("_seconds")}
    if isinstance(value, list):
        return [_without_extraction_timings(item) for item in value]
    return value


def preserve_extraction_receipt(path: Path, observed: dict, retry_directory: Path) -> dict:
    """Keep the first immutable receipt when an unfinished graph is re-extracted."""
    if not observed.get("payload_sha256"):
        raise ValueError("fresh extraction needs a complete scientific payload fingerprint")
    if not path.exists():
        write(path, observed)
        return observed
    first = json.loads(path.read_text())
    if not first.get("payload_sha256"):
        raise ValueError("unfinished extraction predates payload fingerprint; safe graph equivalence cannot be established")
    if _without_extraction_timings(first) != _without_extraction_timings(observed):
        raise ValueError("retried original extraction changed graph, vector, source, or generated Base outputs")
    retry_directory.mkdir(parents=True, exist_ok=True)
    number = len(list(retry_directory.glob("retry_*.json")))
    write(retry_directory / f"retry_{number:04d}.json", {
        "format": "hn_readout_fusion_s_id_original_extraction_retry_v1",
        "recorded_at_utc": utc(), "first_extraction_path": str(path),
        "first_extraction_sha256": sha(path), "payload_sha256": first["payload_sha256"],
        "scientific_content_exact_match": True, "first_receipt_preserved": True,
        "retry_extraction_receipt": observed["receipt"],
        "reason": "interruption before native graph completion; original extraction recomputed without selecting or generating questions"})
    return first


def native_graph_record(source, row: dict, data_dir: Path, lock_path: Path) -> dict:
    """Exactly one original Base graph extraction; cached graph edits are later."""
    verify_lock(lock_path)
    if row.get("final_lock_sha256") != sha(lock_path) or row.get("split") != "fresh_confirmation":
        raise ValueError("formal graph extraction requires the fresh locked bank")
    if source.identity["actual_base_identity"]["checkpoint_sha256"] != BASE_SHA:
        raise ValueError("original Base graph checkpoint differs")
    old = _load_shared("prepare_gpu")
    from base_engine import graph_sha
    import torch
    from s5_lean_circuit.circuit.descriptor_v2_397 import fixed_log1p_397, vectorize_payload_v2_397
    from s5_lean_circuit.circuit.payload_v2 import verify_payload_v2_shard_record
    key = hashlib.sha256(row["question_id"].encode()).hexdigest()
    path = data_dir / "native_records" / f"{key}.json"
    if path.exists():
        value = json.loads(path.read_text())
        if (value["final_lock_sha256"] != sha(lock_path) or value["prompt_sha256"] != row["prompt_sha256"]
                or value["source_identity_sha256"] != source.identity["identity_sha256"]):
            raise ValueError("resumed original graph identity differs")
        payload, = verify_payload_v2_shard_record(value["graph"]["shard"],
            expected_archive_identity_sha256=source.identity["identity_sha256"])
        if value.get("original_payload_sha256") and graph_sha(payload) != value["original_payload_sha256"]:
            raise ValueError("resumed original graph scientific payload differs")
        raw = vectorize_payload_v2_397(payload, n_layers=8, d_feature=16384)
        if (not torch.equal(raw, torch.tensor(value["raw_vector"], dtype=torch.float32))
                or not torch.equal(fixed_log1p_397(raw), torch.tensor(value["model_input_vector"], dtype=torch.float32))):
            raise ValueError("resumed original graph/vector mismatch")
        return value
    extraction_path = data_dir / "native_extractions" / f"{key}.json"
    graph_path = data_dir / "native_graphs" / f"{key}.complete.json"
    if graph_path.exists():
        # Survive interruption after graph write without extracting that graph again.
        graph = json.loads(graph_path.read_text())
        extraction = json.loads(extraction_path.read_text())
        payload, = verify_payload_v2_shard_record(graph["shard"],
            expected_archive_identity_sha256=source.identity["identity_sha256"])
        raw = vectorize_payload_v2_397(payload, n_layers=8, d_feature=16384)
        if (graph["question_id"] != row["question_id"]
                or graph["trace_event_key"] != extraction["trace_event_key"]
                or (extraction.get("payload_sha256") and graph_sha(payload) != extraction["payload_sha256"])
                or not torch.equal(raw, torch.tensor(extraction["raw_vector"], dtype=torch.float32))
                or not torch.equal(fixed_log1p_397(raw), torch.tensor(extraction["model_input_vector"], dtype=torch.float32))):
            raise ValueError("completed graph/extraction provenance mismatch")
    else:
        extraction = old._extract_with_payload(source, row)
        observed = {k: v for k, v in extraction.items() if k != "payload"}
        observed["payload_sha256"] = graph_sha(extraction["payload"])
        retained = preserve_extraction_receipt(extraction_path, observed,
                                               data_dir / "native_extraction_retries" / key)
        extraction = {**retained, "payload": extraction["payload"]}
        original = old.GRAPHS
        try:
            old.GRAPHS = data_dir / "native_graphs"
            graph = old._graph(source, row, extraction)
        finally:
            old.GRAPHS = original
    tokens = extraction["generated_token_ids"]
    value = {"question_id": row["question_id"], "task_type": row["task_type"],
             "prompt_sha256": row["prompt_sha256"], "final_lock_sha256": sha(lock_path),
             "source_identity_sha256": source.identity["identity_sha256"],
             "original_payload_sha256": extraction.get("payload_sha256"),
             "graph": graph, "graph_receipt": extraction["receipt"],
             "raw_vector": extraction["raw_vector"], "model_input_vector": extraction["model_input_vector"],
             "base": old._text_answer(source.tokenizer, row, tokens,
                 old._base_stop(source.tokenizer, row["input_ids"], tokens, 50, 128))}
    write(path, value)
    return value


def metrics_from_counts(counts: dict[str, dict]) -> dict:
    """Exact rational decisions; C is sum of absolute per-task net changes."""
    per_task = {}
    for task in TASKS:
        n, n0, n1 = (int(counts[task][key]) for key in ("N", "n0", "n1"))
        if not 0 <= n0 <= n or not 0 <= n1 <= n:
            raise ValueError("incorrect count outside task denominator")
        drop = Fraction(n0 - n1, n0) if n0 else None
        per_task[task] = {"N": n, "n0": n0, "n1": n1,
                          "accuracy0": n0 / n if n else None, "accuracy1": n1 / n if n else None,
                          "net_correct_change": n1 - n0,
                          "signed_relative_drop": float(drop) if drop is not None else None}
    d = counts[TARGET]["n0"] - counts[TARGET]["n1"]
    n0t = counts[TARGET]["n0"]
    b = sum(counts[t]["n0"] for t in TASKS if t != TARGET)
    a = sum(abs(counts[t]["n0"] - counts[t]["n1"]) for t in TASKS if t != TARGET)
    if n0t == 0 or b == 0:
        return {"per_task": per_task, "L": None, "C": None, "R": None, "L_minus_C": None,
                "valid": False, "status": "undefined_original_denominator", "hn_gate": False,
                "joint_quality_goal_met": False, "exact": {}}
    loss, control = Fraction(d, n0t), Fraction(a, b)
    ratio = loss / control if control else None
    status = "finite_ratio" if control else ("zero_control_net_change" if loss > 0 else "no_positive_target_effect")
    return {"per_task": per_task, "L": float(loss), "C": float(control),
            "R": float(ratio) if ratio is not None else ("Infinity" if loss > 0 else None),
            "L_minus_C": float(loss - control), "valid": True, "status": status,
            "hn_gate": 5*d >= 2*n0t and 10*(d*b-a*n0t) > n0t*b,
            "joint_quality_goal_met": 5*d >= 2*n0t,
            "exact": {"L": str(loss), "C": str(control), "R": str(ratio) if ratio is not None else None,
                      "L_minus_C": str(loss-control)}}


def _correct(row: dict, label: str) -> bool:
    value = row.get("results", row)[label]
    correct = value["correct"] if isinstance(value, dict) else value
    if not isinstance(correct, bool):
        raise ValueError(f"correctness must be an explicit boolean: {label}")
    return correct


def count_metrics(rows: list[dict], baseline_key: str, intervention_key: str) -> dict:
    return metrics_from_counts({t: {"N": sum(r["task_type"] == t for r in rows),
        "n0": sum(_correct(r, baseline_key) for r in rows if r["task_type"] == t),
        "n1": sum(_correct(r, intervention_key) for r in rows if r["task_type"] == t)} for t in TASKS})


def _comparison(direct: dict, joint: dict) -> dict:
    if not direct["valid"] or not joint["valid"]:
        return {"G": None, "ratio_advantage": False, "status": "undefined_original_denominator",
                "delta_L": None, "control_reduction": None, "delta_R": None, "ratio_multiple": None}
    ld, cd = (Fraction(direct["exact"][x]) for x in ("L", "C"))
    lj, cj = (Fraction(joint["exact"][x]) for x in ("L", "C"))
    g = lj*cd-ld*cj
    rd, rj = direct["R"], joint["R"]
    finite = isinstance(rd, (float, int)) and isinstance(rj, (float, int))
    if ld <= 0 or lj <= 0:
        advantage, status = False, "no_positive_target_effect"
    elif rd is None or rj is None:
        advantage, status = False, "ratio_unavailable"
    elif rd == rj == "Infinity":
        advantage, status = False, "both_zero_control_no_ratio_advantage"
    elif rj == "Infinity":
        advantage, status = True, "joint_positive_infinity_direct_finite"
    elif rd == "Infinity":
        advantage, status = False, "direct_positive_infinity_joint_finite"
    else:
        advantage, status = Fraction(joint["exact"]["R"]) > Fraction(direct["exact"]["R"]), "finite_ratios"
    return {"G": float(g), "G_exact": str(g), "ratio_advantage": advantage, "status": status,
            "delta_L": float(lj-ld), "control_reduction": float(cd-cj),
            "delta_R": rj-rd if finite else None,
            "ratio_multiple": rj/rd if finite and rd > 0 else None}


def confirmation_summary(rows: list[dict], lock: dict, bootstrap: int = 5000) -> dict:
    if bootstrap != 5000:
        raise ValueError("formal protocol fixes 5000 paired stratified bootstrap samples")
    if len(rows) != 400 or Counter(r["task_type"] for r in rows) != {t: 100 for t in TASKS}:
        raise ValueError("formal result requires all 100 questions per task")
    for row in rows:
        for label in LABELS:
            _correct(row, label)
    hn = count_metrics(rows, "hn_original", "hn_suppressed")
    direct = count_metrics(rows, "base_original", "base_direct")
    joint = count_metrics(rows, "base_original", "base_joint")
    comparison = _comparison(direct, joint)
    by_task = {t: [r for r in rows if r["task_type"] == t] for t in TASKS}
    # A single sampled index selects every condition, preserving all pairing.
    bits = {t: [tuple(int(_correct(r, label)) for label in LABELS) for r in by_task[t]] for t in TASKS}
    rng = random.Random(42)
    samples = {x: [] for x in ("hn_L", "hn_C", "direct_L", "direct_C", "joint_L", "joint_C", "G")}
    for _ in range(bootstrap):
        totals = {t: [0]*len(LABELS) for t in TASKS}
        for task in TASKS:
            for _ in range(100):
                item = bits[task][rng.randrange(100)]
                for col, value in enumerate(item):
                    totals[task][col] += value
        values = {}
        for name, first, second in (("hn", 3, 4), ("direct", 0, 1), ("joint", 0, 2)):
            values[name] = metrics_from_counts({t: {"N": 100, "n0": totals[t][first], "n1": totals[t][second]} for t in TASKS})
            for metric in ("L", "C"):
                if values[name][metric] is not None:
                    samples[f"{name}_{metric}"].append(values[name][metric])
        g = _comparison(values["direct"], values["joint"])["G"]
        if g is not None:
            samples["G"].append(g)
    def interval(values):
        ordered = sorted(values)
        if not ordered:
            return None
        def quantile(q):
            pos = (len(ordered)-1)*q
            low, high = int(pos), min(int(pos)+1, len(ordered)-1)
            return ordered[low] + (ordered[high]-ordered[low])*(pos-low)
        return [quantile(.025), quantile(.975)]
    intervals = {name: {"ci95": interval(value), "valid_resamples": len(value),
                        "undefined_resamples": bootstrap-len(value)} for name, value in samples.items()}
    g_interval = intervals["G"]["ci95"]
    status = {"hn_group_pass": hn["hn_gate"], "ratio_advantage": comparison["ratio_advantage"],
              "joint_quality_goal_met": joint["joint_quality_goal_met"],
              "chain_positive": hn["hn_gate"] and comparison["ratio_advantage"] and joint["joint_quality_goal_met"],
              "interval_support": bool(g_interval is not None and g_interval[0] > 0)}
    return {"format": "hn_readout_fusion_s_id_formal_result_v1", "target": TARGET, "K": lock["K"],
            "formal_source": "fresh_confirmation_only", "hypernet": hn, "direct": direct, "joint": joint,
            "comparison": comparison, "stage_status": status,
            "bootstrap": {"resamples": 5000, "seed": 42, "paired": True, "stratified_by": "task",
                          "percentile_method": "linear_interpolation_at_(n-1)*q", "intervals": intervals}}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("verify-lock", "generate", "report"))
    parser.add_argument("--lock", required=True, type=Path)
    parser.add_argument("--data-dir", type=Path)
    parser.add_argument("--results", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.phase == "verify-lock":
        value = verify_lock(args.lock)
        print(json.dumps({"valid": True, "K": value["K"], "sha256": sha(args.lock)}))
    elif args.phase == "generate":
        print(json.dumps({"count": len(generate_confirmation(args.lock, args.data_dir))}))
    else:
        from report import write_report
        write_report(args.lock, args.data_dir, args.results, args.output)


if __name__ == "__main__":
    main()
