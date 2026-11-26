"""A separate, identity-bound stage-2 permuted-Circuit training control.

The matched atomic endpoint, recipients, losses and optimization stay fixed.
Only the student's Circuit rows change; the frozen KD teacher stays matched.
"""

from __future__ import annotations

import argparse
from dataclasses import replace
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

import torch
import torch.distributed as dist
from torch import Tensor

FORMAT = "s5_stage2_permuted_control_v1"
KEY = "stage2_permuted_control"
PACKAGE = Path("4server/lean_stage2_permuted_control_v1")
OUTPUT = Path("outputs/s5_lean_circuit/stage2_permuted_control_v1")
LOG = Path("logs/train/2026-11-21/lean-stage2-permuted-control-v1")
ANCHORS = {
    "T_and_or": "C25",
    "T_or_and": "M25",
    "T_and_or_and": "M12",
    "T_nested_and": "C25",
    "T_nested_or": "M25",
}
BANKS = [
    {"path": "outputs/pg_transcoder/lean_stage1_composition_groups_test500_v1/banks/suite1.pt",
     "sha256": "0a8829e9bfa2732eda17a1a1038c15fb211795b664633e23967967bb8240eb24",
     "samples_per_task": 500},
    {"path": "outputs/pg_transcoder/lean_stage1_composition_groups_test500_v1/banks/suite2.pt",
     "sha256": "1c9146975df2df4e84efa436481c47d28c64b944a6d8db6fcd3e467fa3b3d8e5",
     "samples_per_task": 500},
]


def anchor_path(task: str) -> Path:
    return (Path("s5_lean_circuit/configs/composition_loco_circuit_retention20_v2")
            / task / ANCHORS[task] / "circuit_a.json")


def settings(task: str, baseline_sha: str) -> dict[str, Any]:
    return {
        "format": FORMAT,
        "baseline_config_path": anchor_path(task).as_posix(),
        "baseline_config_sha256": baseline_sha,
        "permutation_seed": 741000042,
        "permutation": "global_batch_n_cycle_query_disjoint_v1",
        "scope": "all_student_rows_including_atomic_replay",
        "teacher_condition": "matched",
        "log_dir": (LOG / task).as_posix(),
    }


def validate_config(config: Any) -> None:
    """Validate the complete legacy anchor before allowing the narrow delta."""
    from .composition_loco_config import load_config

    task = config.heldout_task
    if task not in ANCHORS:
        raise ValueError("stage-2 permuted control has an unknown fold")
    baseline = load_config(anchor_path(task))
    expected = settings(task, baseline.computed_sha256)
    if config.training.get(KEY) != expected:
        raise ValueError("stage-2 permutation/provenance settings changed")
    if config.output_dir != (OUTPUT / task).as_posix():
        raise ValueError("stage-2 control must use its independent output directory")
    if (config.data.get("final_bank") != BANKS[0]["path"]
            or config.evaluation.get("final_samples_per_task") != 500
            or config.evaluation.get("reporting_banks") != BANKS):
        raise ValueError("stage-2 reporting must use the approved 500-per-task banks")
    # The parent validator is applied to the parent JSON, never to a weakened
    # approximation. Its historical final-bank reference is not opened.
    training = {k: v for k, v in config.training.items() if k != KEY}
    data = {**config.data, "final_bank": baseline.data["final_bank"]}
    evaluation = {k: v for k, v in config.evaluation.items() if k != "reporting_banks"}
    evaluation["final_samples_per_task"] = baseline.evaluation["final_samples_per_task"]
    restored = replace(config, training=training, data=data, evaluation=evaluation,
                       output_dir=baseline.output_dir, config_sha256=None)
    if restored.to_dict() != baseline.to_dict():
        raise ValueError("stage-2 control changed a matched-anchor invariant")


def make_config(task: str) -> Any:
    from .composition_loco_config import load_config

    baseline = load_config(anchor_path(task))
    return replace(
        baseline,
        training={**baseline.training, KEY: settings(task, baseline.computed_sha256)},
        data={**baseline.data, "final_bank": BANKS[0]["path"]},
        evaluation={**baseline.evaluation, "final_samples_per_task": 500,
                    "reporting_banks": BANKS},
        output_dir=(OUTPUT / task).as_posix(), config_sha256=None,
    )


def _digest(value: Tensor) -> str:
    return hashlib.sha256(value.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def query_fingerprints(query_ids: Tensor, query_mask: Tensor) -> Tensor:
    """Hash actual prompt tokens; padding, task labels and gold are excluded."""
    ids, mask = query_ids.detach().cpu().long(), query_mask.detach().cpu().bool()
    if ids.ndim != 2 or ids.shape != mask.shape or not bool(mask.any(1).all()):
        raise ValueError("invalid prompt tokens/mask for condition shuffle")
    hashes = [list(hashlib.sha256(row[valid].numpy().astype("<i8").tobytes()).digest())
              for row, valid in zip(ids, mask, strict=True)]
    return torch.tensor(hashes, dtype=torch.uint8)


def donor_permutation(fingerprints: Tensor, *, seed: int, step: int) -> tuple[Tensor, dict]:
    from .hypernet.content_origin_v2 import derangement, repair_same_query_donors

    if step < 0 or fingerprints.ndim != 2 or fingerprints.shape[1] != 32:
        raise ValueError("invalid step or prompt fingerprints")
    fingerprints = fingerprints.cpu()
    sources = derangement(len(fingerprints), seed=seed, epoch=step)
    sources, repair = repair_same_query_donors(
        sources, fingerprints, torch.full((len(fingerprints),), 32, dtype=torch.long)
    )
    if bool((fingerprints == fingerprints.index_select(0, sources)).all(1).any()):
        raise ValueError("condition shuffle retained an identical prompt donor")
    return sources, repair


def _all_gather_rows(value: Tensor) -> Tensor:
    if not dist.is_available() or not dist.is_initialized():
        return value
    rows = [torch.empty_like(value) for _ in range(dist.get_world_size())]
    dist.all_gather(rows, value.contiguous())
    return torch.cat(rows, dim=0)


def permuted_student_batch(
    batch: Mapping[str, Tensor], *, seed: int, step: int,
) -> tuple[dict[str, Tensor], dict[str, Any]]:
    """Derange the full DDP batch, returning a new student-only batch.

    Every rank participates once per update. This uses no model/global RNG,
    adds no model forward, and preserves the global vector multiset exactly.
    """
    vectors = batch["vector397"]
    if vectors.ndim != 2 or vectors.shape[1] != 397 or vectors.dtype != torch.float32:
        raise ValueError("stage-2 shuffle requires FP32 [batch,397] Circuit vectors")
    fingerprints = query_fingerprints(batch["query_ids"], batch["query_mask"]).to(vectors.device)
    global_fingerprints = _all_gather_rows(fingerprints).cpu()
    global_vectors = _all_gather_rows(vectors)
    sources, repair = donor_permutation(global_fingerprints, seed=seed, step=step)
    rank = dist.get_rank() if dist.is_available() and dist.is_initialized() else 0
    start, end = rank * len(vectors), (rank + 1) * len(vectors)
    donated = global_vectors.index_select(0, sources.to(vectors.device))
    student_batch = {**batch, "vector397": donated[start:end]}
    # Both index namespaces are recorded. Source means atomic/composition;
    # donor positions refer to the rank-major global batch, not a dataset.
    identities = torch.stack((batch["is_atomic"].long(), batch["source_indices"].long()), dim=1)
    global_identities = _all_gather_rows(identities).cpu()
    trace = {
        "format": FORMAT, "step": step + 1, "permutation_seed": seed,
        "global_batch_size": len(sources), "donor_positions": sources.tolist(),
        "recipient_source_and_index": global_identities.tolist(),
        "prompt_fingerprints_sha256": _digest(global_fingerprints),
        "donor_positions_sha256": _digest(sources),
        "matched_vectors_sha256": _digest(global_vectors),
        "permuted_vectors_sha256": _digest(donated),
        **repair,
    }
    return student_batch, trace


def append_trace(log_dir: str | Path, trace: Mapping[str, Any]) -> None:
    """Bound each full-mapping JSONL chunk to 250 optimizer updates."""
    path = Path(log_dir) / f"condition-pairing-{(int(trace['step']) - 1) // 250:04d}.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(dict(trace), separators=(",", ":"), sort_keys=True) + "\n")


def prepare() -> dict[str, Any]:
    from .composition_loco_config import write_config

    rows = []
    for task, group in ANCHORS.items():
        config = make_config(task)
        path = PACKAGE / "configs" / f"{task}.json"
        write_config(path, config)
        rows.append({"heldout_task": task, "recipe": group, "config_path": str(path),
                     "config_sha256": config.computed_sha256, "steps": 12000,
                     "training_seed": 42})
    matrix = {"format": FORMAT, "training_treatment": "permuted",
              "job_count": 5, "total_updates": 60000, "rows": rows,
              "reporting_banks": BANKS}
    (PACKAGE / "run_matrix.json").write_text(json.dumps(matrix, indent=2) + "\n")
    return matrix


def preflight() -> dict[str, Any]:
    """Check only permitted artifacts; never open the revoked final bank."""
    from .composition_loco import validate_eval_bank, sha256_file
    from .composition_loco_config import load_config

    checked: dict[str, dict] = {}
    for task in ANCHORS:
        config = load_config(PACKAGE / "configs" / f"{task}.json")
        for value in (config.initialization["checkpoint_path"], config.base["checkpoint_path"],
                      config.base["architecture_config_path"], config.data["cache_path"],
                      config.data["fold_index"], config.data["base_ce_path"],
                      config.data["validation_bank"], config.data["atomic_bank"],
                      config.data["tokenizer_path"], config.data["atomic_replay"]["cache_path"]):
            path = Path(value)
            if not path.is_file():
                raise FileNotFoundError(f"missing required training artifact: {path}")
            checked.setdefault(str(path), {"bytes": path.stat().st_size})
        path = str(config.initialization["checkpoint_path"])
        if "sha256" not in checked[path]:
            checked[path]["sha256"] = sha256_file(Path(path))
        if checked[path]["sha256"] != config.initialization["checkpoint_sha256"]:
            raise ValueError("atomic initialization checksum changed")
    for bank in BANKS:
        path = Path(bank["path"])
        if sha256_file(path) != bank["sha256"]:
            raise ValueError(f"approved reporting bank checksum changed: {path}")
        metadata = validate_eval_bank(path)
        if metadata["event_count"] != 4500 or metadata["samples_per_task"] != 500:
            raise ValueError("approved reporting bank denominator changed")
        checked[str(path)] = {"sha256": bank["sha256"], "event_count": 4500}
    return {"format": FORMAT, "preflight_passed": True, "model_forward_calls": 0,
            "training_started": False, "checked_artifacts": checked}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "preflight"))
    args = parser.parse_args()
    print(json.dumps(prepare() if args.action == "prepare" else preflight(), indent=2))


if __name__ == "__main__":
    main()
