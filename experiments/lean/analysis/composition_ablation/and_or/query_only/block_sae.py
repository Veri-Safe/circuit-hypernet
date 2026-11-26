#!/usr/bin/env python3
"""Capture and fit one independent Query M12 block SAE using every PG parameter slot.

Only the Query M12 visible-validation and atomic banks are used. Capture
files live outside Git under outputs/pg_transcoder; small provenance and fit
receipts live under logs/eval. Run --phase capture before --phase fit for each
block. Both phases can be restarted without silently accepting stale inputs.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
for directory in (ROOT / "s5_lean_circuit/src", ROOT / "s2_hypernet/src",
                  ROOT / "s5_lean_circuit/scripts", ROOT / "temp_s"):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

from prepare import CHECKPOINT_SHA, sha as source_sha
from prepare_block import SOURCE, VISIBLE as COMPOSITION_BANK, ATOMIC as ATOMIC_BANK, TASKS
from lc026_nested_or_r03.common import generate, slot_map
from s5_lean_circuit.circuit.transcoder import RectangularReLUTranscoder, transcoder_objective
from s5_lean_circuit.composition_loco import load_eval_bank
from query_adapter import load_query_adapter

LOG_DATE = os.environ.get("QUERY_M12_LOG_DATE", datetime.now(timezone.utc).strftime("%Y-%m-%d"))
LOG = ROOT / "logs/eval" / LOG_DATE / "lean-stage2-and-or-groups-query-m12-dev500-9task-v1/block-sae"
DATA = ROOT / "outputs/pg_transcoder/lean_stage2_and_or_groups_query_m12_dev500_9task_v1/block-sae"
KINDS = ("self_attn_n", "self_attn_hw", "cross_attn", "feed_forward", "residual")
SLOTS = tuple(range(272))
SPLITS = {"train": (0, 300), "validation": (300, 400), "test": (400, 500)}
SHARD_QUESTIONS = 25
SHARDS_PER_TASK = 500 // SHARD_QUESTIONS
WIDTH = 1024
FEATURES = 8192


def target_identity() -> dict:
    target = json.loads((ROOT / "4server/lean_stage2_and_or_groups_query_m12_dev500_9task_v1/target.json").read_text())
    if (target["checkpoint_sha256"] != CHECKPOINT_SHA
            or source_sha(ROOT / target["checkpoint_path"]) != CHECKPOINT_SHA):
        raise ValueError("original Query M12 target identity differs")
    return target


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def canonical_json(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, ensure_ascii=False,
                      allow_nan=False, separators=(",", ":")).encode()


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    temporary.write_bytes(json.dumps(value, sort_keys=True, indent=2,
                                     ensure_ascii=False, allow_nan=False).encode() + b"\n")
    os.replace(temporary, path)


def torch_save_atomic(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    torch.save(value, temporary)
    os.replace(temporary, path)


def sites_for(block: int) -> list[str]:
    if not 0 <= block < 12:
        raise ValueError("block must be in 0..11")
    return [f"block{block}.{kind}" for kind in KINDS]


def source_questions() -> tuple[dict, dict[str, list[dict]], Path]:
    manifest_path = SOURCE / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    question_path = ROOT / manifest["questions_path"]
    if (manifest.get("format") != "lean_stage2_query_m12_block_source_v1"
            or manifest.get("complete") is not True
            or manifest.get("question_creation") is not False
            or manifest.get("questions_per_task") != 500
            or manifest.get("tasks") != list(TASKS)
            or sha(question_path) != manifest["questions_sha256"]
            or sha(COMPOSITION_BANK) != manifest["composition_bank_sha256"]
            or sha(ATOMIC_BANK) != manifest["atomic_bank_sha256"]
            or target_identity()["checkpoint_sha256"] != manifest["checkpoint_sha256"]):
        raise ValueError("locked Query M12 visible/atomic source identity differs")
    rows = [json.loads(line) for line in question_path.read_text().splitlines()]
    selected: dict[str, list[dict]] = {}
    for task in TASKS:
        task_rows = sorted((row for row in rows if row["task_type"] == task),
                           key=lambda row: row["task_index"])
        if (len(task_rows) != 500
                or [row["task_index"] for row in task_rows] != list(range(500))
                or len({row["event_key"] for row in task_rows}) != 500
                or any(row["bank"] != ("atomic" if task.startswith("S_") else "composition")
                       for row in task_rows)):
            raise ValueError(f"existing Query M12 {task} source selection differs")
        selected[task] = task_rows
    if len({row["event_key"] for group in selected.values() for row in group}) != 4500:
        raise ValueError("nine-task event keys are not unique")
    return manifest, selected, manifest_path


def check_question(bank, row: dict) -> tuple[torch.Tensor, torch.Tensor]:
    index = int(row["canonical_global_index"])
    if not 0 <= index < len(bank.records):
        raise ValueError("canonical question index outside bank")
    record = bank.records[index]
    length = int(bank.payload["query_lengths"][index])
    vector = bank.payload["vectors"][index:index + 1].contiguous()
    if (record.task_type != row["task_type"]
            or bytes(bank.payload["event_key"][index]).hex() != row["event_key"]
            or sha_bytes(record.input_text.encode()) != row["prompt_sha256"]
            or sha_bytes(record.output_text.encode()) != row["gold_output_sha256"]
            or length != row["query_length"]
            or vector.shape != (1, 397)
            or not bool(torch.isfinite(vector).all())
            or sha_bytes(vector.numpy().tobytes()) != row["raw397_model_input_sha256"]):
        raise ValueError(f"Query M12 {row['task_type']} question or Base Circuit397 differs")
    ids = bank.payload["query_ids"][index:index + 1, :length]
    if ids.shape != (1, length):
        raise ValueError("Query M12 tokenized prompt shape differs")
    return ids, vector


@contextmanager
def capture_block(core, block_index: int, sink: dict[str, torch.Tensor]):
    pg = core.parameter_generator.model
    if (core.condition_source != "query_only" or core.fusion_mode != "append"
            or core.circuit_token_count != 0 or len(pg.blocks) != 12
            or pg.seq_n != 8 or pg.seq_h != 34 or pg.d_model != WIDTH):
        raise ValueError("Query M12 PG architecture or Query-only condition differs")
    block = pg.blocks[block_index]
    handles = []
    for kind in KINDS:
        site = f"block{block_index}.{kind}"
        module = block if kind == "residual" else getattr(block, kind)

        def hook(_module, _inputs, output, *, kind=kind, site=site):
            if site in sink or not isinstance(output, torch.Tensor):
                raise ValueError(f"duplicate or malformed Query M12 PG site: {site}")
            if kind == "self_attn_n":
                if output.shape != (34, 8, WIDTH):
                    raise ValueError(f"Query M12 PG site layout differs: {site}")
                canonical = output.reshape(34, 8, WIDTH).permute(1, 0, 2)
            elif kind == "self_attn_hw":
                if output.shape != (8, 34, WIDTH):
                    raise ValueError(f"Query M12 PG site layout differs: {site}")
                canonical = output
            elif kind == "cross_attn":
                if output.shape != (1, 272, WIDTH):
                    raise ValueError(f"Query M12 PG site layout differs: {site}")
                canonical = output.reshape(8, 34, WIDTH)
            else:
                if output.shape != (1, 8, 34, WIDTH):
                    raise ValueError(f"Query M12 PG site layout differs: {site}")
                canonical = output[0]
            sink[site] = canonical.reshape(272, WIDTH).detach().to(
                device="cpu", dtype=torch.float16).contiguous()

        handles.append(module.register_forward_hook(hook))
    try:
        yield
        if set(sink) != set(sites_for(block_index)):
            raise ValueError("Query M12 PG block site capture incomplete")
    finally:
        for handle in handles:
            handle.remove()


def capture_identity(block: int, task: str, shard: int,
                     source_manifest: dict, rows: list[dict], slot_map_sha: str) -> dict:
    selected = rows[shard * SHARD_QUESTIONS:(shard + 1) * SHARD_QUESTIONS]
    if len(selected) != SHARD_QUESTIONS:
        raise ValueError("capture shard question count differs")
    return {
        "format": "lean_stage2_query_m12_block_fullslot_capture_v2",
        "block": block, "task": task, "condition": "matched",
        "circuit_present_argument": True, "circuit_token_count": 0, "question_creation": False,
        "task_index_start": shard * SHARD_QUESTIONS,
        "question_count": SHARD_QUESTIONS,
        "canonical_global_indices": [row["canonical_global_index"] for row in selected],
        "event_keys_sha256": sha_bytes(b"".join(bytes.fromhex(row["event_key"])
                                             for row in selected)),
        "source_question_rows_sha256": sha_bytes(canonical_json(selected)),
        "raw397_sha256": sha_bytes(b"".join(bytes.fromhex(row["raw397_model_input_sha256"])
                                          for row in selected)),
        "query_m12_checkpoint_sha256": source_manifest["checkpoint_sha256"],
        "composition_bank_sha256": source_manifest["composition_bank_sha256"],
        "atomic_bank_sha256": source_manifest["atomic_bank_sha256"],
        "source_questions_sha256": source_manifest["questions_sha256"],
        "slot_map_sha256": slot_map_sha,
        "site_names": sites_for(block), "slot_indices": list(SLOTS),
        "shape": [SHARD_QUESTIONS, len(KINDS), len(SLOTS), WIDTH],
        "dtype": "float16",
    }


def capture_paths(block: int, task: str, shard: int) -> tuple[Path, Path]:
    stem = f"part{shard:02d}"
    return (DATA / "captures" / f"block{block:02d}" / task / f"{stem}.npy",
            LOG / "captures" / f"block{block:02d}" / task / f"{stem}.json")


def load_capture_receipt(block: int, task: str, shard: int,
                         identity: dict) -> dict | None:
    data_path, receipt_path = capture_paths(block, task, shard)
    if not data_path.exists() and not receipt_path.exists():
        return None
    # A crash between the data rename and receipt write leaves an orphaned
    # shard. Recompute it; a complete receipt is the only resumable unit.
    if not receipt_path.is_file():
        return None
    receipt = json.loads(receipt_path.read_text())
    if (receipt.get("identity") != identity
            or receipt.get("path") != str(data_path.relative_to(ROOT))):
        raise ValueError(f"existing capture shard identity or checksum differs: {data_path}")
    if not data_path.is_file():
        return None
    if sha(data_path) != receipt.get("sha256"):
        raise ValueError(f"existing capture shard checksum differs: {data_path}")
    saved = np.load(data_path, mmap_mode="r", allow_pickle=False)
    if saved.shape != tuple(identity["shape"]) or saved.dtype != np.float16:
        raise ValueError(f"existing capture shard shape or dtype differs: {data_path}")
    return receipt


@torch.inference_mode()
def run_capture(block: int) -> None:
    source, rows, source_path = source_questions()
    target = target_identity()
    if source["checkpoint_sha256"] != target["checkpoint_sha256"]:
        raise ValueError("Query M12 source checkpoint differs")
    adapter = load_query_adapter(torch.device("cuda"))
    try:
        mapping = slot_map(adapter.hypernet.parameter_generator)
        if len(mapping) != len(SLOTS) or [row["flat_slot"] for row in mapping] != list(SLOTS):
            raise ValueError("Query M12 all-272 LoRA endpoint map differs")
        mapping_sha = sha_bytes(canonical_json(mapping))
        composition = load_eval_bank(COMPOSITION_BANK)
        atomic = load_eval_bank(ATOMIC_BANK)
        banks = {task: atomic if task.startswith("S_") else composition for task in TASKS}
        first_rows = [rows[task][0] for task in TASKS]
        for row in first_rows:
            ids_cpu, vector_cpu = check_question(banks[row["task_type"]], row)
            ids, vector = ids_cpu.to("cuda", torch.long), vector_cpu.to("cuda")
            native = generate(adapter.hypernet, ids, vector, True,
                              float(adapter.gate_multiplier))
            sink: dict[str, torch.Tensor] = {}
            with capture_block(adapter.hypernet, block, sink):
                replay = generate(adapter.hypernet, ids, vector, True,
                                  float(adapter.gate_multiplier))
            if (native.keys() != replay.keys()
                    or any(not torch.equal(native[key], replay[key]) for key in native)):
                raise ValueError(f"block{block} no-op hook changed generated LoRA")

        receipts = []
        for task in TASKS:
            bank = banks[task]
            for shard in range(SHARDS_PER_TASK):
                identity = capture_identity(block, task, shard, source, rows[task], mapping_sha)
                receipt = load_capture_receipt(block, task, shard, identity)
                if receipt is None:
                    data_path, receipt_path = capture_paths(block, task, shard)
                    data_path.parent.mkdir(parents=True, exist_ok=True)
                    temporary = data_path.with_name(f".{data_path.stem}.tmp.{os.getpid()}.npy")
                    try:
                        array = np.lib.format.open_memmap(temporary, mode="w+",
                                                          dtype=np.float16,
                                                          shape=tuple(identity["shape"]))
                        for local, row in enumerate(rows[task][shard * SHARD_QUESTIONS:
                                                               (shard + 1) * SHARD_QUESTIONS]):
                            ids_cpu, vector_cpu = check_question(bank, row)
                            sink = {}
                            with capture_block(adapter.hypernet, block, sink):
                                generate(adapter.hypernet, ids_cpu.to("cuda", torch.long),
                                         vector_cpu.to("cuda"), True,
                                         float(adapter.gate_multiplier))
                            tensor = torch.stack([sink[site] for site in sites_for(block)])
                            if tensor.shape != (5, 272, WIDTH) or not bool(torch.isfinite(tensor).all()):
                                raise ValueError(f"block{block} nonfinite PG capture: {task}/{row['task_index']}")
                            array[local] = tensor.numpy()
                        array.flush()
                        del array
                        os.replace(temporary, data_path)
                    finally:
                        temporary.unlink(missing_ok=True)
                    receipt = {"identity": identity,
                               "path": str(data_path.relative_to(ROOT)),
                               "sha256": sha(data_path),
                               "bytes": data_path.stat().st_size}
                    write_json(receipt_path, receipt)
                receipts.append(receipt)
                write_json(LOG / f"block{block:02d}_capture_manifest.json", {
                    "format": "lean_stage2_query_m12_block_fullslot_capture_manifest_v2",
                    "complete": False,
                    "question_creation": False,
                    "tasks": list(TASKS), "questions_per_task": 500,
                    "split": {key: list(value) for key, value in SPLITS.items()},
                    "site_names": sites_for(block), "slot_indices": list(SLOTS),
                    "source_manifest_sha256": sha(source_path),
                    "source_questions_sha256": source["questions_sha256"],
                    "query_m12_checkpoint_sha256": source["checkpoint_sha256"],
                    "slot_map_sha256": mapping_sha,
                    "no_op_generated_lora_parity": True,
                    "capture_shards": receipts,
                })
                print("QUERY_M12_FULLSLOT_CAPTURE", block, task, shard + 1,
                      "/", SHARDS_PER_TASK, flush=True)
        capture_manifest_path = LOG / f"block{block:02d}_capture_manifest.json"
        complete_manifest = json.loads(capture_manifest_path.read_text())
        complete_manifest["complete"] = True
        write_json(capture_manifest_path, complete_manifest)
        print("QUERY_M12_FULLSLOT_CAPTURE_COMPLETE", block, flush=True)
    finally:
        adapter.close()


def load_capture_mmaps(block: int) -> tuple[dict, dict[tuple[str, int], np.memmap], str]:
    source, rows, source_path = source_questions()
    manifest_path = LOG / f"block{block:02d}_capture_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    if (manifest.get("format") != "lean_stage2_query_m12_block_fullslot_capture_manifest_v2"
            or manifest.get("complete") is not True
            or manifest.get("question_creation") is not False
            or manifest.get("tasks") != list(TASKS)
            or manifest.get("questions_per_task") != 500
            or manifest.get("split") != {key: list(value) for key, value in SPLITS.items()}
            or manifest.get("site_names") != sites_for(block)
            or manifest.get("slot_indices") != list(SLOTS)
            or manifest.get("source_manifest_sha256") != sha(source_path)
            or manifest.get("source_questions_sha256") != source["questions_sha256"]
            or manifest.get("query_m12_checkpoint_sha256") != source["checkpoint_sha256"]
            or manifest.get("no_op_generated_lora_parity") is not True
            or len(manifest.get("capture_shards", [])) != len(TASKS) * SHARDS_PER_TASK):
        raise ValueError(f"block{block} fullslot capture manifest incomplete or changed")
    mmaps: dict[tuple[str, int], np.memmap] = {}
    for index, (task, shard) in enumerate((task, shard) for task in TASKS
                                         for shard in range(SHARDS_PER_TASK)):
        identity = capture_identity(block, task, shard, source, rows[task],
                                    manifest["slot_map_sha256"])
        receipt = load_capture_receipt(block, task, shard, identity)
        if receipt is None or receipt != manifest["capture_shards"][index]:
            raise ValueError(f"block{block} capture shard inventory differs: {task}/{shard}")
        mmaps[(task, shard)] = np.load(ROOT / receipt["path"], mmap_mode="r",
                                      allow_pickle=False)
    return manifest, mmaps, sha(manifest_path)


def site_rms(mmaps: dict[tuple[str, int], np.memmap]) -> list[float]:
    square = np.zeros(len(KINDS), dtype=np.float64)
    count = 0
    for task in TASKS:
        for shard in range(SPLITS["train"][1] // SHARD_QUESTIONS):
            values = mmaps[(task, shard)]
            for start in range(0, SHARD_QUESTIONS, 5):
                chunk = np.asarray(values[start:start + 5], dtype=np.float32)
                square += np.sum(np.square(chunk), axis=(0, 2, 3), dtype=np.float64)
                count += len(chunk) * len(SLOTS) * WIDTH
    scales = np.maximum(np.sqrt(square / count), 1e-4)
    if count != len(TASKS) * SPLITS["train"][1] * len(SLOTS) * WIDTH or not np.isfinite(scales).all():
        raise ValueError("fullslot nine-task train300 site RMS differs")
    return scales.tolist()


def sample_rows(mmaps: dict[tuple[str, int], np.memmap], task: str, site: int,
                low: int, high: int, count: int, generator: torch.Generator,
                scale: float) -> torch.Tensor:
    questions = torch.randint(low, high, (count,), generator=generator).numpy()
    slots = torch.randint(0, len(SLOTS), (count,), generator=generator).numpy()
    output = np.empty((count, WIDTH), dtype=np.float32)
    shard_ids = questions // SHARD_QUESTIONS
    for shard in np.unique(shard_ids):
        positions = np.flatnonzero(shard_ids == shard)
        values = mmaps[(task, int(shard))]
        output[positions] = values[questions[positions] % SHARD_QUESTIONS,
                                   site, slots[positions]]
    output /= scale
    if not np.isfinite(output).all():
        raise ValueError("sampled fullslot SAE rows are nonfinite")
    return torch.from_numpy(output)


def validation_rows(mmaps: dict[tuple[str, int], np.memmap], scales: list[float],
                    seed: int, rows_per_task_site: int = 2048) -> torch.Tensor:
    generator = torch.Generator(device="cpu").manual_seed(seed + 1_000_000)
    pieces = [sample_rows(mmaps, task, site, *SPLITS["validation"],
                          rows_per_task_site, generator, scales[site])
              for task in TASKS for site in range(len(KINDS))]
    return torch.cat(pieces).to("cuda", dtype=torch.float32)


@torch.inference_mode()
def validation_quality(model, x: torch.Tensor, sparsity: float) -> dict:
    model.eval()
    sse = energy = sparse = active = 0.0
    mean = x.mean(dim=0)
    centered = 0.0
    decoder_norms = model.W_dec.float().norm(dim=1)
    for start in range(0, len(x), 2048):
        chunk = x[start:start + 2048]
        with torch.autocast("cuda", dtype=torch.bfloat16):
            prediction, features = model(chunk)
        target = chunk.float()
        z = features.float()
        sse += float((prediction.float() - target).square().sum())
        energy += float(target.square().sum())
        centered += float((target - mean).square().sum())
        sparse += float(torch.tanh(4.0 * z * decoder_norms).sum())
        active += float(z.gt(0).sum())
    return {"slot_count": len(x), "energy_nmse": sse / energy,
            "centered_nmse": sse / centered if centered > 0 else None,
            "l0_mean": active / len(x),
            "sse_per_slot": sse / len(x),
            "sparsity_penalty_per_slot": sparsity * sparse / len(x)}


@torch.inference_mode()
def test_quality_by_site(model, mmaps: dict[tuple[str, int], np.memmap],
                         scales: list[float], block: int) -> dict:
    model.eval()
    output = {}
    for site, name in enumerate(KINDS):
        task_rows = {}
        for task in TASKS:
            n = 0
            sse = energy = active = 0.0
            target_sum = torch.zeros(WIDTH, dtype=torch.float64)
            support = torch.zeros(FEATURES, dtype=torch.bool, device="cuda")
            for shard in range(SPLITS["test"][0] // SHARD_QUESTIONS, SHARDS_PER_TASK):
                mmap = mmaps[(task, shard)]
                for start in range(0, SHARD_QUESTIONS, 4):
                    end = min(start + 4, SHARD_QUESTIONS)
                    x = np.asarray(mmap[start:end, site], dtype=np.float32).reshape(-1, WIDTH)
                    x /= scales[site]
                    batch = torch.from_numpy(x).to("cuda", dtype=torch.float32)
                    with torch.autocast("cuda", dtype=torch.bfloat16):
                        prediction, features = model(batch)
                    target_sum += batch.double().sum(dim=0).cpu()
                    sse += float((prediction.float() - batch).square().sum())
                    energy += float(batch.square().sum())
                    active += float(features.gt(0).sum())
                    support |= features.gt(0).any(dim=0)
                    n += len(batch)
            if n != (SPLITS["test"][1] - SPLITS["test"][0]) * len(SLOTS):
                raise ValueError(f"test coverage differs for {task}/{name}")
            centered = energy - float(target_sum.square().sum() / n)
            task_rows[task] = {
                "slot_count": n, "energy_nmse": sse / energy,
                "centered_nmse": sse / centered if centered > 0 else None,
                "l0_mean": active / n,
                "dead_feature_fraction": float((~support).float().mean()),
                "sse": sse, "energy": energy,
                "target_sum": target_sum.tolist(),
            }
        total_n = sum(row["slot_count"] for row in task_rows.values())
        total_sse = sum(row["sse"] for row in task_rows.values())
        total_energy = sum(row["energy"] for row in task_rows.values())
        total_mean = np.sum(np.asarray([row["target_sum"] for row in task_rows.values()]), axis=0)
        total_centered = total_energy - float(np.square(total_mean).sum() / total_n)
        output[sites_for(block)[site]] = {
            "all_tasks": {"slot_count": total_n,
                          "energy_nmse": total_sse / total_energy,
                          "centered_nmse": total_sse / total_centered if total_centered > 0 else None,
                          "l0_mean": sum(row["l0_mean"] * row["slot_count"]
                                         for row in task_rows.values()) / total_n},
            "by_task": {task: {key: value for key, value in row.items()
                               if key not in ("sse", "energy", "target_sum")}
                        for task, row in task_rows.items()},
        }
    return output


def training_config(block: int, scales: list[float], capture_sha: str,
                    source: dict, steps: int) -> dict:
    return {
        "format": "lean_stage2_query_m12_block_sae_fullslot_fit_config_v2",
        "kind": "SAE", "training": "query_only_matched_activation_reconstruction",
        "block": block, "site_names": sites_for(block),
        "feature_address": "(block, site, feature_id); IDs are unrelated across blocks",
        "slot_indices": list(SLOTS), "training_tasks": list(TASKS),
        "split": {key: list(value) for key, value in SPLITS.items()},
        "site_rms": scales, "site_rms_source": "all train300 questions and all 272 slots for each of nine tasks",
        "d_in": WIDTH, "d_out": WIDTH, "dictionary_size": FEATURES,
        "steps": steps, "batch_size": 512, "lr": 0.001,
        "sparsity": 0.5, "seed": 177 + block,
        "validation_rows_per_task_site": 2048,
        "query_m12_checkpoint_sha256": source["query_m12_checkpoint_sha256"],
        "capture_manifest_sha256": capture_sha,
    }


def run_fit(block: int, steps: int) -> None:
    manifest, mmaps, capture_sha = load_capture_mmaps(block)
    scales = site_rms(mmaps)
    config = training_config(block, scales, capture_sha, manifest, steps)
    model_dir = DATA / "models" / f"block{block:02d}_shared5site_8192"
    best_path, last_path = model_dir / "best.pt", model_dir / "last.pt"
    report_path = LOG / f"block{block:02d}_fit_summary.json"
    if report_path.exists():
        report = json.loads(report_path.read_text())
        if (report.get("config") != config
                or report.get("capture_manifest_sha256") != capture_sha
                or not report.get("complete")
                or not best_path.is_file()
                or report.get("checkpoint_sha256") != sha(best_path)):
            raise ValueError(f"existing block{block} fit report differs")
        print("QUERY_M12_FULLSLOT_FIT_ALREADY_COMPLETE", block, flush=True)
        return

    torch.manual_seed(config["seed"])
    model = RectangularReLUTranscoder(WIDTH, WIDTH, FEATURES)
    model.reset_parameters(seed=config["seed"])
    model = model.cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"])
    generator = torch.Generator(device="cpu").manual_seed(config["seed"] + 1000)
    validation = validation_rows(mmaps, scales, config["seed"],
                                 config["validation_rows_per_task_site"])
    start_step, best_step, best_score = 0, -1, float("inf")
    history: list[dict] = []
    if last_path.exists():
        saved = torch.load(last_path, map_location="cpu", weights_only=False)
        if (saved.get("config") != config
                or saved.get("capture_manifest_sha256") != capture_sha
                or saved.get("step", 0) < 100
                or saved.get("step", 0) > steps):
            raise ValueError(f"existing block{block} training state differs")
        model.load_state_dict(saved["state_dict"])
        optimizer.load_state_dict(saved["optimizer_state_dict"])
        generator.set_state(saved["generator_state"])
        start_step, best_step, best_score = saved["step"], saved["best_step"], saved["best_score"]
        history = saved["history"]
        if best_step >= 0 and not best_path.is_file():
            raise ValueError("resumed training lacks its best checkpoint")
    model_dir.mkdir(parents=True, exist_ok=True)
    for step in range(start_step + 1, steps + 1):
        model.train()
        task = TASKS[((step - 1) // len(KINDS)) % len(TASKS)]
        site = (step - 1) % len(KINDS)
        x = sample_rows(mmaps, task, site, *SPLITS["train"],
                        config["batch_size"], generator, scales[site]).to("cuda")
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            prediction, features = model(x)
        objective = transcoder_objective(prediction, x, features, model.W_dec,
                                         sparsity_coeff=config["sparsity"], c_coeff=4.0)
        if not bool(torch.isfinite(objective.loss)):
            raise ValueError(f"block{block} nonfinite SAE loss at step {step}")
        objective.loss.backward()
        optimizer.step()
        if step % 100 == 0 or step == steps:
            metrics = validation_quality(model, validation, config["sparsity"])
            criterion = metrics["sse_per_slot"] + metrics["sparsity_penalty_per_slot"]
            history.append({"step": step, "validation": metrics, "objective": criterion})
            if criterion < best_score:
                best_score, best_step = criterion, step
                torch_save_atomic(best_path, {
                    "format": "lean_stage2_query_m12_block_sae_fullslot_checkpoint_v2",
                    "config": config, "capture_manifest_sha256": capture_sha,
                    "step": step, "validation": metrics,
                    "state_dict": {key: value.detach().cpu().clone()
                                   for key, value in model.state_dict().items()},
                })
            torch_save_atomic(last_path, {
                "format": "lean_stage2_query_m12_block_sae_fullslot_train_state_v2",
                "config": config, "capture_manifest_sha256": capture_sha,
                "step": step, "best_step": best_step,
                "best_score": best_score, "history": history,
                "state_dict": {key: value.detach().cpu().clone()
                               for key, value in model.state_dict().items()},
                "optimizer_state_dict": optimizer.state_dict(),
                "generator_state": generator.get_state(),
            })
            print("QUERY_M12_FULLSLOT_FIT", block, step, "validation_nmse",
                  round(metrics["energy_nmse"], 5), "l0", round(metrics["l0_mean"], 2),
                  flush=True)
    best = torch.load(best_path, map_location="cpu", weights_only=False)
    if best["config"] != config or best["capture_manifest_sha256"] != capture_sha:
        raise ValueError("best fullslot checkpoint identity differs")
    model.load_state_dict(best["state_dict"])
    test = test_quality_by_site(model, mmaps, scales, block)
    report = {
        "format": "lean_stage2_query_m12_block_sae_fullslot_fit_v2",
        "complete": True, "question_creation": False,
        "config": config, "block": block,
        "capture_manifest_sha256": capture_sha,
        "checkpoint_path": str(best_path.relative_to(ROOT)),
        "checkpoint_sha256": sha(best_path),
        "best_step": best_step, "best_validation_objective": best_score,
        "test_quality_by_site": test,
        "history": history,
        "test_population": "each task's visible/atomic questions 400-499, all 272 slots",
        "condition_delta_fidelity_evaluated": False,
        "functional_evidence_evaluated": False,
    }
    write_json(report_path, report)
    print("QUERY_M12_FULLSLOT_FIT_COMPLETE", block,
          {site: round(value["all_tasks"]["energy_nmse"], 5)
           for site, value in test.items()}, flush=True)


@torch.inference_mode()
def run_smoke(block: int) -> None:
    source, rows, _ = source_questions()
    target = target_identity()
    composition = load_eval_bank(COMPOSITION_BANK)
    atomic = load_eval_bank(ATOMIC_BANK)
    adapter = load_query_adapter(torch.device("cuda"))
    try:
        mapping = slot_map(adapter.hypernet.parameter_generator)
        if len(mapping) != 272 or [row["flat_slot"] for row in mapping] != list(SLOTS):
            raise ValueError("Query M12 smoke all-272 slot mapping differs")
        checks = []
        for task in ("T_and_or", "T_or_and", "S_and"):
            row = rows[task][0]
            bank = atomic if task.startswith("S_") else composition
            ids_cpu, vector_cpu = check_question(bank, row)
            ids, vector = ids_cpu.to("cuda", torch.long), vector_cpu.to("cuda")
            original = generate(adapter.hypernet, ids, vector, True, float(adapter.gate_multiplier))
            sink = {}
            with capture_block(adapter.hypernet, block, sink):
                replay = generate(adapter.hypernet, ids, vector, True, float(adapter.gate_multiplier))
            if (set(original) != set(replay)
                    or any(not torch.equal(original[key], replay[key]) for key in original)
                    or set(sink) != set(sites_for(block))
                    or any(value.shape != (272, WIDTH) or not bool(torch.isfinite(value).all())
                           for value in sink.values())):
                raise ValueError(f"Query M12 block{block} smoke no-op or site coverage differs")
            checks.append(task)
        result = {"format": "lean_stage2_query_m12_block_sae_smoke_v1", "complete": True,
                  "checkpoint_sha256": source["checkpoint_sha256"],
                  "block": block, "sites": sites_for(block), "slot_count": len(mapping),
                  "tasks_checked": checks, "no_op_generated_lora_parity": True}
        write_json(LOG / f"block{block:02d}_smoke.json", result)
        print("QUERY_M12_FULLSLOT_SMOKE_COMPLETE", result, flush=True)
    finally:
        adapter.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", required=True, choices=("smoke", "capture", "fit"))
    parser.add_argument("--block", type=int, required=True)
    parser.add_argument("--steps", type=int, default=2000,
                        help="SAE optimizer steps; used only by --phase fit")
    args = parser.parse_args()
    if not 0 <= args.block < 12 or args.steps < 100:
        parser.error("block must be 0..11 and steps must be at least 100")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    torch.set_num_threads(4)
    if args.phase == "smoke":
        run_smoke(args.block)
    elif args.phase == "capture":
        run_capture(args.block)
    else:
        run_fit(args.block, args.steps)


if __name__ == "__main__":
    main()
