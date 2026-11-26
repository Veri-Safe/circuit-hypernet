#!/usr/bin/env python3
"""Frozen data, augmentation, ranking, and scoring rules for NVARC ARC evals."""

from __future__ import annotations

import hashlib
import json
import os
import platform
import tempfile
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

import numpy as np


INTERNAL_PROTOCOL_VERSION = "nvarc_internal_eval_v1"
OFFICIAL_PROTOCOL_VERSION = "nvarc_official_fidelity_eval_v1"
ARC_AGI1_INTERNAL_PROTOCOL_VERSION = "arc_agi1_nvarc_internal_eval_v1"
ARC_AGI1_PUBLIC_VALIDATION_PROTOCOL_VERSION = (
    "arc_agi1_nvarc_public_validation_v1"
)
INTERNAL_CONFIG_SHA256 = "5d5231cd9aac3d8964c725fec879d8ea9c81e069064d642be1b0ea033e128dbb"
OFFICIAL_CONFIG_SHA256 = "ee021fb9c00cbe4b0bd47d79dba40c218103c3eefd48783bd8e8c5719036e77a"
ARC_AGI1_INTERNAL_CONFIG_SHA256 = "afffeddcfc5bf8164f5873a0f5be331a64fbec2ea7da99eec616650ee7173498"
ARC_AGI1_PUBLIC_VALIDATION_CONFIG_SHA256 = "54037b1baac229754a7f372b1de887025482733f39941184ada0afa6877d235e"
ARC_AGI2_SOURCE_COMMIT = "f3283f727488ad98fe575ea6a5ac981e4a188e49"
ARC_AGI2_CHALLENGE_SHA256 = "93d3e1b76403937ce5424ae507a37d62a94c0d0e74c8fa01314c0e5572061d98"
ARC_AGI2_SOLUTION_SHA256 = "a0be86d2e8b77b8dec9396a10014f68c6659d049eed130ea8c51ca029f30f29b"
ARC_AGI2_BUNDLE_SHA256 = "d95ae0da92faa3ad9433cc5df209a40c87e19046834be9abbd5004e87dc327ff"
ARC_AGI2_TASK_COUNT = 120
ARC_AGI2_TEST_COUNT = 167
ARC_AGI1_SOURCE_COMMIT = "399030444e0ab0cc8b4e199870fb20b863846f34"
ARC_AGI1_CHALLENGE_SHA256 = "46b27a6330e9b2dde16e33c81f80f96a3d7553d1c44b14eb712a256349fd55f5"
ARC_AGI1_SOLUTION_SHA256 = "e97693b0899f8f3fcc87d58c3891329cad63ed054489f9b0d440329de0bb9945"
ARC_AGI1_BUNDLE_SHA256 = "173c6096d70d4893c4ef7562db45483a0be1bb9b77f2ab12cbe950b7605078eb"
ARC_AGI1_TASK_COUNT = 400
ARC_AGI1_TEST_COUNT = 419

Grid = list[list[int]]


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def semantic_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_frozen_config(
    path: str | Path,
    expected_protocol: str,
    expected_sha256: str | None = None,
) -> tuple[dict[str, Any], str]:
    config_path = Path(path)
    actual_sha256 = sha256_file(config_path)
    if expected_sha256 is not None and actual_sha256 != expected_sha256:
        raise ValueError(
            f"Frozen config SHA256 mismatch: {actual_sha256} != {expected_sha256} for {config_path}"
        )
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if config.get("protocol_version") != expected_protocol:
        raise ValueError(
            f"Expected protocol_version={expected_protocol!r}, "
            f"found {config.get('protocol_version')!r} in {config_path}"
        )
    if config.get("frozen") is not True:
        raise ValueError(f"Eval config is not frozen: {config_path}")
    return config, actual_sha256


def normalize_grid(grid: Any, *, label: str = "grid") -> Grid:
    if not isinstance(grid, list) or not 1 <= len(grid) <= 30:
        raise ValueError(f"{label} must contain 1..30 rows")
    width: int | None = None
    result: Grid = []
    for row_index, row in enumerate(grid):
        if not isinstance(row, list) or not 1 <= len(row) <= 30:
            raise ValueError(f"{label}[{row_index}] must contain 1..30 cells")
        if width is None:
            width = len(row)
        elif len(row) != width:
            raise ValueError(f"{label} must be rectangular")
        normalized_row: list[int] = []
        for column_index, cell in enumerate(row):
            if isinstance(cell, bool) or not isinstance(cell, (int, np.integer)):
                raise ValueError(f"{label}[{row_index}][{column_index}] is not an integer")
            value = int(cell)
            if not 0 <= value <= 9:
                raise ValueError(f"{label}[{row_index}][{column_index}] is outside 0..9")
            normalized_row.append(value)
        result.append(normalized_row)
    return result


def grid_to_string(grid: Sequence[Sequence[int]]) -> str:
    return "\n".join("".join(str(int(cell)) for cell in row) for row in grid)


def serialized_grid(grid: Sequence[Sequence[int]]) -> str:
    return grid_to_string(grid)


def _validate_example(example: Any, *, require_output: bool, label: str) -> dict[str, Grid]:
    if not isinstance(example, dict):
        raise ValueError(f"{label} must be an object")
    expected = {"input", "output"} if require_output else {"input"}
    if set(example) != expected:
        raise ValueError(f"{label} fields must be {sorted(expected)}, found {sorted(example)}")
    normalized = {"input": normalize_grid(example["input"], label=f"{label}.input")}
    if require_output:
        normalized["output"] = normalize_grid(example["output"], label=f"{label}.output")
    return normalized


def validate_challenges(challenges: Any) -> dict[str, dict[str, list[dict[str, Grid]]]]:
    if not isinstance(challenges, dict) or not challenges:
        raise ValueError("ARC challenges must be a non-empty object")
    result: dict[str, dict[str, list[dict[str, Grid]]]] = {}
    for task_id in sorted(challenges):
        task = challenges[task_id]
        if not isinstance(task_id, str) or not task_id:
            raise ValueError("ARC task ids must be non-empty strings")
        if not isinstance(task, dict) or set(task) != {"train", "test"}:
            raise ValueError(f"Task {task_id} must have exactly train and test fields")
        if not isinstance(task["train"], list) or not task["train"]:
            raise ValueError(f"Task {task_id} has no training examples")
        if not isinstance(task["test"], list) or not task["test"]:
            raise ValueError(f"Task {task_id} has no test inputs")
        result[task_id] = {
            "train": [
                _validate_example(example, require_output=True, label=f"{task_id}.train[{index}]")
                for index, example in enumerate(task["train"])
            ],
            "test": [
                _validate_example(example, require_output=False, label=f"{task_id}.test[{index}]")
                for index, example in enumerate(task["test"])
            ],
        }
    return result


def validate_solutions(
    solutions: Any,
    challenges: Mapping[str, Mapping[str, Sequence[Any]]],
) -> dict[str, list[Grid]]:
    if not isinstance(solutions, dict) or set(solutions) != set(challenges):
        raise ValueError("Solution task ids do not exactly match challenge task ids")
    result: dict[str, list[Grid]] = {}
    for task_id in sorted(challenges):
        task_solutions = solutions[task_id]
        if not isinstance(task_solutions, list):
            raise ValueError(f"Solutions for {task_id} must be a list")
        if len(task_solutions) != len(challenges[task_id]["test"]):
            raise ValueError(f"Solution count does not match test count for {task_id}")
        result[task_id] = [
            normalize_grid(grid, label=f"{task_id}.solutions[{index}]")
            for index, grid in enumerate(task_solutions)
        ]
    return result


def load_challenges(path: str | Path) -> dict[str, dict[str, list[dict[str, Grid]]]]:
    return validate_challenges(json.loads(Path(path).read_text(encoding="utf-8")))


def load_solutions(
    path: str | Path,
    challenges: Mapping[str, Mapping[str, Sequence[Any]]],
) -> dict[str, list[Grid]]:
    return validate_solutions(json.loads(Path(path).read_text(encoding="utf-8")), challenges)


def load_arc_task_directory(task_dir: str | Path) -> tuple[dict[str, Any], dict[str, Any]]:
    source = Path(task_dir)
    files = sorted(source.glob("*.json"))
    if not files:
        raise FileNotFoundError(f"No ARC task JSON files found under {source}")
    challenges: dict[str, Any] = {}
    solutions: dict[str, Any] = {}
    for path in files:
        raw = json.loads(path.read_text(encoding="utf-8"))
        task_id = path.stem
        if task_id in challenges:
            raise ValueError(f"Duplicate ARC task id: {task_id}")
        if not isinstance(raw, dict) or set(raw) != {"train", "test"}:
            raise ValueError(f"Source task {path} must have exactly train and test fields")
        challenge_test = []
        task_solutions = []
        for index, example in enumerate(raw["test"]):
            normalized = _validate_example(
                example,
                require_output=True,
                label=f"{path.name}.test[{index}]",
            )
            challenge_test.append({"input": normalized["input"]})
            task_solutions.append(normalized["output"])
        challenges[task_id] = {"train": raw["train"], "test": challenge_test}
        solutions[task_id] = task_solutions
    challenges = validate_challenges(challenges)
    solutions = validate_solutions(solutions, challenges)
    return challenges, solutions


def validate_arc_agi2_identity(
    challenges: Mapping[str, Any],
    solutions: Mapping[str, Any] | None = None,
    *,
    expected_task_count: int = ARC_AGI2_TASK_COUNT,
    expected_test_count: int = ARC_AGI2_TEST_COUNT,
    expected_challenge_sha256: str = ARC_AGI2_CHALLENGE_SHA256,
    expected_solution_sha256: str = ARC_AGI2_SOLUTION_SHA256,
    expected_bundle_sha256: str = ARC_AGI2_BUNDLE_SHA256,
) -> dict[str, Any]:
    task_count = len(challenges)
    test_count = sum(len(task["test"]) for task in challenges.values())
    challenge_sha256 = semantic_sha256(challenges)
    if task_count != expected_task_count:
        raise ValueError(f"Expected {expected_task_count} ARC tasks, found {task_count}")
    if test_count != expected_test_count:
        raise ValueError(f"Expected {expected_test_count} ARC test cases, found {test_count}")
    if challenge_sha256 != expected_challenge_sha256:
        raise ValueError(
            f"Challenge semantic SHA256 mismatch: {challenge_sha256} != {expected_challenge_sha256}"
        )
    result = {
        "task_count": task_count,
        "test_count": test_count,
        "challenge_semantic_sha256": challenge_sha256,
    }
    if solutions is not None:
        solution_sha256 = semantic_sha256(solutions)
        if solution_sha256 != expected_solution_sha256:
            raise ValueError(
                f"Solution semantic SHA256 mismatch: {solution_sha256} != {expected_solution_sha256}"
            )
        result["solution_semantic_sha256"] = solution_sha256
        bundle_sha256 = semantic_sha256(
            {"challenges": challenges, "solutions": solutions}
        )
        if bundle_sha256 != expected_bundle_sha256:
            raise ValueError(
                f"Bundle semantic SHA256 mismatch: {bundle_sha256} != {expected_bundle_sha256}"
            )
        result["bundle_semantic_sha256"] = bundle_sha256
    return result


def validate_arc_agi1_identity(
    challenges: Mapping[str, Any],
    solutions: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    task_count = len(challenges)
    test_count = sum(len(task["test"]) for task in challenges.values())
    challenge_sha256 = semantic_sha256(challenges)
    if task_count != ARC_AGI1_TASK_COUNT:
        raise ValueError(
            f"Expected {ARC_AGI1_TASK_COUNT} ARC-AGI-1 tasks, found {task_count}"
        )
    if test_count != ARC_AGI1_TEST_COUNT:
        raise ValueError(
            f"Expected {ARC_AGI1_TEST_COUNT} ARC-AGI-1 test cases, found {test_count}"
        )
    if challenge_sha256 != ARC_AGI1_CHALLENGE_SHA256:
        raise ValueError(
            "ARC-AGI-1 challenge semantic SHA256 mismatch: "
            f"{challenge_sha256} != {ARC_AGI1_CHALLENGE_SHA256}"
        )
    result = {
        "task_count": task_count,
        "test_count": test_count,
        "challenge_semantic_sha256": challenge_sha256,
    }
    if solutions is not None:
        solution_sha256 = semantic_sha256(solutions)
        if solution_sha256 != ARC_AGI1_SOLUTION_SHA256:
            raise ValueError(
                "ARC-AGI-1 solution semantic SHA256 mismatch: "
                f"{solution_sha256} != {ARC_AGI1_SOLUTION_SHA256}"
            )
        result["solution_semantic_sha256"] = solution_sha256
        bundle_sha256 = semantic_sha256(
            {"challenges": challenges, "solutions": solutions}
        )
        if bundle_sha256 != ARC_AGI1_BUNDLE_SHA256:
            raise ValueError(
                "ARC-AGI-1 bundle semantic SHA256 mismatch: "
                f"{bundle_sha256} != {ARC_AGI1_BUNDLE_SHA256}"
            )
        result["bundle_semantic_sha256"] = bundle_sha256
    return result


def prepare_arc_eval_bundle(
    task_dir: str | Path,
    output_dir: str | Path,
    *,
    enforce_arc_agi2_identity: bool = True,
) -> dict[str, Any]:
    source = Path(task_dir).resolve()
    destination = Path(output_dir).resolve()
    challenges, solutions = load_arc_task_directory(source)
    identity = (
        validate_arc_agi2_identity(challenges, solutions)
        if enforce_arc_agi2_identity
        else {
            "task_count": len(challenges),
            "test_count": sum(len(task["test"]) for task in challenges.values()),
            "challenge_semantic_sha256": semantic_sha256(challenges),
            "solution_semantic_sha256": semantic_sha256(solutions),
            "bundle_semantic_sha256": semantic_sha256(
                {"challenges": challenges, "solutions": solutions}
            ),
        }
    )
    manifest = {
        "format_version": "nvarc_arc_eval_bundle_v1",
        "source_directory": str(source),
        "source_repository": "https://github.com/arcprize/ARC-AGI-2",
        "source_commit": ARC_AGI2_SOURCE_COMMIT if enforce_arc_agi2_identity else None,
        **identity,
        "python_version": platform.python_version(),
        "numpy_version": np.__version__,
    }

    if destination.exists():
        manifest_path = destination / "manifest.json"
        if not manifest_path.is_file():
            raise FileExistsError(f"Existing bundle has no manifest: {destination}")
        existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        comparable_keys = (
            "format_version",
            "source_commit",
            "task_count",
            "test_count",
            "challenge_semantic_sha256",
            "solution_semantic_sha256",
            "bundle_semantic_sha256",
        )
        if any(existing.get(key) != manifest.get(key) for key in comparable_keys):
            raise FileExistsError(f"Existing ARC eval bundle manifest does not match: {destination}")
        stored_challenges = load_challenges(destination / "challenges.json")
        stored_solutions = load_solutions(destination / "solutions.json", stored_challenges)
        stored_identity = (
            validate_arc_agi2_identity(stored_challenges, stored_solutions)
            if enforce_arc_agi2_identity
            else {
                "challenge_semantic_sha256": semantic_sha256(stored_challenges),
                "solution_semantic_sha256": semantic_sha256(stored_solutions),
                "bundle_semantic_sha256": semantic_sha256(
                    {"challenges": stored_challenges, "solutions": stored_solutions}
                ),
            }
        )
        for key in (
            "challenge_semantic_sha256",
            "solution_semantic_sha256",
            "bundle_semantic_sha256",
        ):
            if stored_identity[key] != existing.get(key):
                raise ValueError(f"Existing ARC eval bundle content changed for {key}")
        if sha256_file(destination / "challenges.json") != existing.get("challenge_file_sha256"):
            raise ValueError("Existing ARC challenges file SHA256 changed")
        if sha256_file(destination / "solutions.json") != existing.get("solution_file_sha256"):
            raise ValueError("Existing ARC solutions file SHA256 changed")
        return existing

    destination.parent.mkdir(parents=True, exist_ok=True)
    temp_dir = Path(tempfile.mkdtemp(prefix=f".{destination.name}.tmp-", dir=destination.parent))
    try:
        (temp_dir / "challenges.json").write_bytes(canonical_json_bytes(challenges) + b"\n")
        (temp_dir / "solutions.json").write_bytes(canonical_json_bytes(solutions) + b"\n")
        manifest["challenge_file_sha256"] = sha256_file(temp_dir / "challenges.json")
        manifest["solution_file_sha256"] = sha256_file(temp_dir / "solutions.json")
        (temp_dir / "manifest.json").write_bytes(canonical_json_bytes(manifest) + b"\n")
        os.replace(temp_dir, destination)
    except BaseException:
        for child in temp_dir.iterdir() if temp_dir.exists() else []:
            child.unlink()
        if temp_dir.exists():
            temp_dir.rmdir()
        raise
    return manifest


@dataclass(frozen=True)
class ViewSpec:
    view_index: int
    geometry_ops: tuple[str, ...]
    color_permutation: tuple[int, ...]
    train_order: tuple[int, ...]

    @property
    def descriptor(self) -> str:
        geometry = ".".join(self.geometry_ops) if self.geometry_ops else "identity"
        colors = "".join(str(value) for value in self.color_permutation)
        order = ",".join(str(value) for value in self.train_order)
        return f"{geometry}.permute{colors}.train[{order}]"


AUTHOR_GEOMETRIES: tuple[tuple[str, ...], ...] = (
    (),
    ("transpose",),
    ("rot90",),
    ("transpose", "rot90"),
    ("rot90", "rot90"),
    ("transpose", "rot90", "rot90"),
    ("rot90", "rot90", "rot90"),
    ("transpose", "rot90", "rot90", "rot90"),
)


def build_author_view_specs(train_count: int, *, augment_n: int, seed: int) -> list[ViewSpec]:
    if train_count <= 0:
        raise ValueError("train_count must be positive")
    if augment_n <= 0:
        raise ValueError("augment_n must be positive")
    rng = np.random.RandomState(seed)
    partial: list[tuple[tuple[str, ...], tuple[int, ...]]] = []
    for _ in range(augment_n):
        for geometry in AUTHOR_GEOMETRIES:
            colors = tuple(int(value) for value in rng.permutation(10).tolist())
            partial.append((geometry, colors))
    specs = []
    for view_index, (geometry, colors) in enumerate(partial):
        train_order = tuple(int(value) for value in rng.permutation(train_count).tolist())
        specs.append(ViewSpec(view_index, geometry, colors, train_order))
    return specs


def canonical_view_spec(train_count: int) -> ViewSpec:
    return ViewSpec(0, (), tuple(range(10)), tuple(range(train_count)))


def _apply_geometry(array: np.ndarray, operations: Sequence[str], *, inverse: bool) -> np.ndarray:
    operations_iterable = reversed(operations) if inverse else operations
    result = array
    for operation in operations_iterable:
        if operation == "transpose":
            result = np.swapaxes(result, 0, 1)
        elif operation == "rot90":
            result = np.rot90(result, k=3 if inverse else 1)
        else:
            raise ValueError(f"Unknown geometry operation: {operation}")
    return result


def transform_grid(grid: Sequence[Sequence[int]], spec: ViewSpec) -> Grid:
    array = np.asarray(normalize_grid([list(row) for row in grid]), dtype=np.int64)
    array = _apply_geometry(array, spec.geometry_ops, inverse=False)
    array = np.asarray(spec.color_permutation, dtype=np.int64)[array]
    return array.tolist()


def invert_grid(grid: Sequence[Sequence[int]], spec: ViewSpec) -> Grid:
    array = np.asarray(normalize_grid([list(row) for row in grid]), dtype=np.int64)
    inverse_permutation = np.argsort(np.asarray(spec.color_permutation, dtype=np.int64))
    array = inverse_permutation[array]
    array = _apply_geometry(array, spec.geometry_ops, inverse=True)
    return array.tolist()


def transform_test_view(task: Mapping[str, Any], test_index: int, spec: ViewSpec) -> dict[str, Any]:
    train = task["train"]
    if sorted(spec.train_order) != list(range(len(train))):
        raise ValueError("View train_order is not a permutation of all training examples")
    return {
        "train": [
            {
                "input": transform_grid(train[index]["input"], spec),
                "output": transform_grid(train[index]["output"], spec),
            }
            for index in spec.train_order
        ],
        "test_input": transform_grid(task["test"][test_index]["input"], spec),
    }


def format_qwen_prompt(train: Sequence[Mapping[str, Grid]], test_input: Grid) -> str:
    text = ""
    for example in train:
        text += (
            f"<|im_start|>user\n{grid_to_string(example['input'])}"
            f"<|im_end|><|im_start|>assistant\n{grid_to_string(example['output'])}<|im_end|>"
        )
    return (
        text
        + f"<|im_start|>user\n{grid_to_string(test_input)}"
        + "<|im_end|><|im_start|>assistant\n"
    )


def truncate_prompt_to_tokens(
    transformed: Mapping[str, Any],
    encode: Callable[[str], Sequence[int]],
    max_input_tokens: int,
) -> tuple[str, int, int]:
    train = list(transformed["train"])
    removed = 0
    while True:
        prompt = format_qwen_prompt(train, transformed["test_input"])
        token_count = len(encode(prompt))
        if token_count <= max_input_tokens:
            return prompt, token_count, removed
        if not train:
            raise ValueError(
                f"ARC test input alone requires {token_count} tokens, above limit {max_input_tokens}"
            )
        train = train[1:]
        removed += 1


@dataclass(frozen=True)
class ViewPrediction:
    view_index: int
    grid: Grid | None
    normalized_logprob: float


@dataclass(frozen=True)
class RankedCandidate:
    grid: Grid
    votes: int
    mean_normalized_logprob: float
    serialized: str


def rank_view_predictions(predictions: Iterable[ViewPrediction]) -> list[RankedCandidate]:
    grouped: dict[str, list[ViewPrediction]] = defaultdict(list)
    grids: dict[str, Grid] = {}
    for prediction in predictions:
        if prediction.grid is None:
            continue
        normalized = normalize_grid(prediction.grid, label=f"view[{prediction.view_index}]")
        key = serialized_grid(normalized)
        grouped[key].append(prediction)
        grids[key] = normalized
    ranked = [
        RankedCandidate(
            grid=grids[key],
            votes=len(group),
            mean_normalized_logprob=float(
                sum(item.normalized_logprob for item in group) / len(group)
            ),
            serialized=key,
        )
        for key, group in grouped.items()
    ]
    ranked.sort(key=lambda item: (-item.votes, -item.mean_normalized_logprob, item.serialized))
    return ranked


def build_submission(
    challenges: Mapping[str, Mapping[str, Sequence[Any]]],
    attempts: Mapping[tuple[str, int], Sequence[Grid]],
) -> dict[str, list[dict[str, Grid]]]:
    submission: dict[str, list[dict[str, Grid]]] = {}
    for task_id in sorted(challenges):
        task_attempts = []
        for test_index in range(len(challenges[task_id]["test"])):
            candidates = list(attempts.get((task_id, test_index), []))
            task_attempts.append(
                {
                    "attempt_1": candidates[0] if candidates else [[0]],
                    "attempt_2": candidates[1] if len(candidates) > 1 else [[0]],
                }
            )
        submission[task_id] = task_attempts
    return submission


def score_attempts(
    challenges: Mapping[str, Mapping[str, Sequence[Any]]],
    solutions: Mapping[str, Sequence[Grid]],
    attempts: Mapping[tuple[str, int], Sequence[Grid]],
    *,
    top_k: int,
) -> dict[str, Any]:
    if top_k <= 0:
        raise ValueError("top_k must be positive")
    task_score = 0.0
    solved_test_cases = 0
    per_task: dict[str, float] = {}
    for task_id in sorted(challenges):
        test_count = len(challenges[task_id]["test"])
        score = 0.0
        for test_index, gold in enumerate(solutions[task_id]):
            candidate_grids = attempts.get((task_id, test_index), [])[:top_k]
            if any(candidate == gold for candidate in candidate_grids):
                score += 1.0 / test_count
                solved_test_cases += 1
        per_task[task_id] = score
        task_score += score
    task_count = len(challenges)
    return {
        "task_score": task_score,
        "task_count": task_count,
        "accuracy": task_score / task_count,
        "solved_test_cases": solved_test_cases,
        "test_case_count": sum(len(task["test"]) for task in challenges.values()),
        "per_task_score": per_task,
    }


def summarize_view_counts(predictions: Iterable[ViewPrediction]) -> dict[str, Any]:
    items = list(predictions)
    valid = [item for item in items if item.grid is not None]
    counts = Counter(serialized_grid(item.grid) for item in valid if item.grid is not None)
    return {
        "view_count": len(items),
        "valid_view_count": len(valid),
        "valid_grid_rate": len(valid) / len(items) if items else 0.0,
        "unique_candidates": len(counts),
    }
