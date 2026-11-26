"""Frozen task generation and strict exact-match evaluation for S5.

The implementation intentionally mirrors the public
``circuit-extraction-stability`` SFT/evaluation code.  Atomic-only changes only
the training task pool; validation and test evaluation still cover all nine
task families.
"""

from __future__ import annotations

import hashlib
import json
import random
import re
from dataclasses import dataclass
from typing import Any, Iterable, Sequence

from .logic_tasks import TrulyRandomDatasetGenerator


ATOMIC_TASKS: tuple[str, ...] = ("S_and", "S_or_l", "S_or_r", "S_id")
COMPOSITION_TASKS: tuple[str, ...] = (
    "T_and_or",
    "T_or_and",
    "T_and_or_and",
    "T_nested_and",
    "T_nested_or",
)
ALL_TASKS: tuple[str, ...] = ATOMIC_TASKS + COMPOSITION_TASKS
SURFACE_FAMILIES: tuple[str, ...] = ("narrow", "broad_medium", "broad_full")
SURFACE_VARIANTS: tuple[str, ...] = (
    "base",
    "distractor_heavy",
    "compact_input",
    "relabel_input",
    "goal_first_input",
    "long_names",
)
STAGE1_SCHEDULE_TYPES: tuple[str, ...] = (
    "curriculum",
    "atomic_only",
    "composition_only",
)
DEPTH3_TASKS: tuple[str, ...] = (
    "D3_nested_and",
    "D3_nested_or",
    "D3_alternating",
)
KNOWN_TASKS: tuple[str, ...] = ALL_TASKS + DEPTH3_TASKS
TASK_TO_ID: dict[str, int] = {task: index for index, task in enumerate(ALL_TASKS)}

TRAIN_SEED_MULTIPLIER = 1_000_000_000
PUBLIC_VAL_SEED_BASE = 888_888_000
PUBLIC_TEST_SEED_BASE = 999_999_000
STAGE_B_SUPERVISED_SEED_BASE = 50_000_000_000
STAGE_B_TASK_SEED_STRIDE = 100_000_000
STAGE_B_SAMPLING_CYCLE = 100
STAGE_B_SAMPLER_VERSION = "s5_stage_b_weighted_cycle_v1"
STAGE_B_LEGACY_RENDERER_VERSION = "s5_public_global_rng_renderer_v1"
STAGE_B_RENDERER_VERSION_V1 = "s5_stage_b_seeded_premise_order_v1"
STAGE_B_RENDERER_VERSION = "s5_stage_b_public_props_seeded_premises_v2"


@dataclass(frozen=True)
class EvaluationSample:
    task_type: str
    seed: int
    input_text: str
    output_text: str


@dataclass(frozen=True)
class TrainingSample:
    task_type: str
    seed: int


def validate_surface_family(surface_family: str) -> str:
    normalized = str(surface_family)
    if normalized not in SURFACE_FAMILIES:
        raise ValueError(f"unknown surface family: {normalized}")
    return normalized


def validate_task_types(task_types: Iterable[str]) -> tuple[str, ...]:
    normalized = tuple(task_types)
    if not normalized:
        raise ValueError("task_types must not be empty")
    unknown = sorted(set(normalized) - set(KNOWN_TASKS))
    if unknown:
        raise ValueError(f"unknown task types: {unknown}")
    if len(set(normalized)) != len(normalized):
        raise ValueError("task_types must not contain duplicates")
    return normalized


def generate_sample(
    task_type: str,
    seed: int,
    *,
    n_distractors: int = 2,
    renderer_version: str = STAGE_B_LEGACY_RENDERER_VERSION,
    surface_family: str = "narrow",
) -> tuple[str, str]:
    """Generate one public narrow or expanded-family sample.

    The expanded-family formatting is vendored from
    ``data/family_generator.py`` at upstream commit 4624427. Stage-B keeps its
    separate narrow renderer contract and therefore cannot request an expanded
    family.
    """

    validate_task_types([task_type])
    family = validate_surface_family(surface_family)
    if family != "narrow" and renderer_version != STAGE_B_LEGACY_RENDERER_VERSION:
        raise ValueError("expanded families require the public Stage-1 renderer")
    variant_rng = random.Random(int(seed))
    if family == "narrow":
        variant = "base"
    elif family == "broad_medium":
        variant = variant_rng.choices(
            ["base", "distractor_heavy", "goal_first_input", "long_names"],
            weights=[0.40, 0.20, 0.20, 0.20],
        )[0]
    else:
        variant = variant_rng.choices(
            [
                "base",
                "distractor_heavy",
                "compact_input",
                "relabel_input",
                "goal_first_input",
                "long_names",
            ],
            weights=[0.20, 0.20, 0.15, 0.15, 0.15, 0.15],
        )[0]
    return generate_sample_variant(
        task_type,
        seed,
        surface_variant=variant,
        n_distractors=n_distractors,
        renderer_version=renderer_version,
    )


def generate_sample_variant(
    task_type: str,
    seed: int,
    *,
    surface_variant: str,
    n_distractors: int = 2,
    renderer_version: str = STAGE_B_LEGACY_RENDERER_VERSION,
    deterministic_premise_order: bool = False,
) -> tuple[str, str]:
    """Generate one sample with an explicitly frozen public surface variant.

    This entry point is reserved for boundary-condition evaluation.  Training
    continues to use :func:`generate_sample`, where the expanded-family mixture
    is sampled from the public family recipe.  Keeping the explicit variant out
    of ordinary training calls prevents a paper-aligned OOD column from
    silently changing the training distribution.
    """

    validate_task_types([task_type])
    variant = str(surface_variant)
    if variant not in SURFACE_VARIANTS:
        raise ValueError(f"unknown surface variant: {variant}")
    if renderer_version not in {
        STAGE_B_LEGACY_RENDERER_VERSION,
        STAGE_B_RENDERER_VERSION_V1,
        STAGE_B_RENDERER_VERSION,
    }:
        raise ValueError(f"unknown renderer version: {renderer_version}")
    if variant != "base" and renderer_version != STAGE_B_LEGACY_RENDERER_VERSION:
        raise ValueError("paper boundary variants require the public renderer")

    generator = TrulyRandomDatasetGenerator(
        seed=int(seed),
        n_distractors=6 if variant == "distractor_heavy" else int(n_distractors),
    )
    seeded_premises = renderer_version != STAGE_B_LEGACY_RENDERER_VERSION
    public_props = renderer_version != STAGE_B_RENDERER_VERSION_V1
    sample = generator.generate_sample(task_type)
    if variant in {"base", "distractor_heavy"}:
        return sample.to_input_output(
            premise_order_seed=(
                int(seed)
                if seeded_premises or deterministic_premise_order
                else None
            ),
            public_proposition_order=public_props,
        )

    def output_text() -> str:
        _, value = sample.to_input_output(
            premise_order_seed=(int(seed) if deterministic_premise_order else None),
            public_proposition_order=public_props,
        )
        return value

    all_premises = sample.premises + sample.distractors
    prop_decl = (
        " ".join(sorted(set(sample.props + [p for _, p in sample.distractors])))
        + " : Prop"
    )
    if variant == "compact_input":
        premise_str = " ; ".join(f"{h} : {p}" for h, p in all_premises)
        input_text = f"state_0 | {prop_decl} | {premise_str} | goal {sample.goal}"
        return input_text, output_text()
    if variant == "relabel_input":
        premise_lines = [f"{h} : {p}" for h, p in all_premises]
        input_text = (
            "theorem_0:\n"
            + prop_decl
            + "\nassumptions:\n"
            + "\n".join(premise_lines)
            + f"\nprove {sample.goal}"
        )
        return input_text, output_text()
    if variant == "goal_first_input":
        premise_lines = [f"{h} : {p}" for h, p in all_premises]
        input_text = (
            f"state_0:\n⊢ {sample.goal}\nwhere\n{prop_decl}\n"
            + "\n".join(premise_lines)
        )
        return input_text, output_text()
    if variant == "long_names":
        input_text, output_text = sample.to_input_output()
        mapping: dict[str, str] = {
            proposition: f"prop_{index}_{proposition}_long"
            for index, proposition in enumerate(sample.props)
        }
        for index, (hypothesis, _) in enumerate(all_premises):
            mapping[hypothesis] = f"hyp_{index}_{hypothesis}_long"
        for source, target in sorted(
            mapping.items(), key=lambda item: len(item[0]), reverse=True
        ):
            pattern = rf"(?<![A-Za-z0-9_]){re.escape(source)}(?![A-Za-z0-9_])"
            input_text = re.sub(pattern, target, input_text)
            output_text = re.sub(pattern, target, output_text)
        return input_text, output_text
    raise AssertionError(f"unhandled surface variant: {variant}")


def stage1_task_for_sample(
    *,
    sample_seed: int,
    global_sample_index: int,
    total_samples: int,
    schedule_type: str,
) -> str:
    """Choose one Stage-1 task under the frozen 30/30/40 curriculum.

    The gradual phase linearly increases compositional probability from zero
    to 5/9, which is the compositional mass of a uniform draw over all nine
    tasks. This is the explicit local interpretation of the paper's otherwise
    underspecified ``gradual mix`` phrase.
    """

    schedule = str(schedule_type)
    if schedule not in STAGE1_SCHEDULE_TYPES:
        raise ValueError(f"unknown Stage-1 schedule type: {schedule}")
    if total_samples < 1 or not 0 <= global_sample_index < total_samples:
        raise ValueError("invalid Stage-1 sample position")
    rng = random.Random(int(sample_seed))
    if schedule == "atomic_only":
        return rng.choice(ATOMIC_TASKS)
    if schedule == "composition_only":
        return rng.choice(COMPOSITION_TASKS)

    scaled_position = int(global_sample_index) * 10
    if scaled_position < int(total_samples) * 3:
        return rng.choice(ATOMIC_TASKS)
    if scaled_position >= int(total_samples) * 6:
        return rng.choice(ALL_TASKS)

    phase_progress = (
        (int(global_sample_index) / int(total_samples) - 0.3) / 0.3
    )
    composition_probability = phase_progress * len(COMPOSITION_TASKS) / len(ALL_TASKS)
    task_pool = (
        COMPOSITION_TASKS
        if rng.random() < composition_probability
        else ATOMIC_TASKS
    )
    return rng.choice(task_pool)


def stage1_training_samples(
    *,
    model_seed: int,
    global_step: int,
    global_batch_size: int,
    rank: int,
    world_size: int,
    total_steps: int,
    schedule_type: str,
) -> list[TrainingSample]:
    seeds = training_sample_seeds(
        model_seed=model_seed,
        global_step=global_step,
        global_batch_size=global_batch_size,
        rank=rank,
        world_size=world_size,
    )
    local_batch_size = int(global_batch_size) // int(world_size)
    global_start = (
        (int(global_step) - 1) * int(global_batch_size)
        + int(rank) * local_batch_size
    )
    total_samples = int(total_steps) * int(global_batch_size)
    return [
        TrainingSample(
            task_type=stage1_task_for_sample(
                sample_seed=sample_seed,
                global_sample_index=global_start + local_index,
                total_samples=total_samples,
                schedule_type=schedule_type,
            ),
            seed=sample_seed,
        )
        for local_index, sample_seed in enumerate(seeds)
    ]


def create_stage1_training_batch(
    tokenizer: Any,
    *,
    samples: Sequence[TrainingSample],
    surface_family: str,
    n_ctx: int = 128,
    n_distractors: int = 2,
) -> torch.Tensor:
    """Create an upstream-style Stage-1 batch, including public truncation."""

    import torch

    family = validate_surface_family(surface_family)
    rows: list[list[int]] = []
    for sample in samples:
        input_text, output_text = generate_sample(
            sample.task_type,
            sample.seed,
            n_distractors=n_distractors,
            surface_family=family,
        )
        token_ids = list(tokenizer.encode(input_text + output_text))[:n_ctx]
        if len(token_ids) < n_ctx:
            token_ids.extend([0] * (n_ctx - len(token_ids)))
        rows.append(token_ids)
    return torch.tensor(rows, dtype=torch.long)


def training_sample_seeds(
    *,
    model_seed: int,
    global_step: int,
    global_batch_size: int,
    rank: int = 0,
    world_size: int = 1,
) -> list[int]:
    """Return one rank's deterministic slice of the global sample sequence."""

    if global_step < 1:
        raise ValueError("global_step is one-indexed and must be positive")
    if world_size < 1 or not 0 <= rank < world_size:
        raise ValueError("invalid distributed rank/world_size")
    if global_batch_size % world_size:
        raise ValueError("global_batch_size must be divisible by world_size")

    local_batch_size = global_batch_size // world_size
    global_start = (
        int(model_seed) * TRAIN_SEED_MULTIPLIER
        + (int(global_step) - 1) * int(global_batch_size)
    )
    rank_start = global_start + rank * local_batch_size
    return list(range(rank_start, rank_start + local_batch_size))


def validate_sampling_counts(
    counts_per_cycle: dict[str, int],
    *,
    train_tasks: Sequence[str],
    cycle_size: int = STAGE_B_SAMPLING_CYCLE,
) -> dict[str, int]:
    tasks = validate_task_types(train_tasks)
    if set(counts_per_cycle) != set(tasks):
        raise ValueError(
            "sampling-count tasks must exactly equal train_tasks"
        )
    normalized = {task: int(counts_per_cycle[task]) for task in tasks}
    if any(count <= 0 for count in normalized.values()):
        raise ValueError("sampling counts must be positive")
    if sum(normalized.values()) != int(cycle_size):
        raise ValueError(f"sampling counts must sum to {cycle_size}")
    return normalized


def _stable_stage_b_cycle(
    counts_per_cycle: dict[str, int],
    *,
    cycle_index: int,
    sampling_seed: int,
) -> list[str]:
    entries: list[tuple[str, int]] = []
    for task in ALL_TASKS:
        for occurrence in range(int(counts_per_cycle.get(task, 0))):
            entries.append((task, occurrence))

    def stable_key(entry: tuple[str, int]) -> bytes:
        task, occurrence = entry
        payload = (
            f"s5-stage-b:{int(sampling_seed)}:{int(cycle_index)}:"
            f"{TASK_TO_ID[task]}:{int(occurrence)}"
        ).encode("ascii")
        return hashlib.sha256(payload).digest()

    entries.sort(key=stable_key)
    return [task for task, _ in entries]


def stage_b_sampler_provenance(
    *,
    total_samples: int,
    train_tasks: Sequence[str],
    counts_per_cycle: dict[str, int],
    sampling_seed: int,
    seed_base: int,
    task_seed_stride: int,
    renderer_version: str = STAGE_B_LEGACY_RENDERER_VERSION,
) -> dict[str, Any]:
    counts = validate_sampling_counts(
        counts_per_cycle,
        train_tasks=train_tasks,
    )
    full_cycles, remainder = divmod(int(total_samples), STAGE_B_SAMPLING_CYCLE)
    sample_counts = {
        task: full_cycles * counts[task]
        for task in validate_task_types(train_tasks)
    }
    if remainder:
        tail_cycle = _stable_stage_b_cycle(
            counts,
            cycle_index=full_cycles,
            sampling_seed=sampling_seed,
        )
        for task in tail_cycle[:remainder]:
            sample_counts[task] += 1
    provenance = {
        "version": STAGE_B_SAMPLER_VERSION,
        "renderer_version": str(renderer_version),
        "task_to_id": {task: TASK_TO_ID[task] for task in ALL_TASKS},
        "train_tasks": list(validate_task_types(train_tasks)),
        "counts_per_100": counts,
        "sampling_seed": int(sampling_seed),
        "seed_base": int(seed_base),
        "task_seed_stride": int(task_seed_stride),
        "total_samples": int(total_samples),
        "sample_counts": sample_counts,
    }
    canonical = json.dumps(
        provenance,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    provenance["sha256"] = hashlib.sha256(canonical).hexdigest()
    return provenance


def stage_b_training_samples(
    *,
    global_step: int,
    global_batch_size: int,
    rank: int,
    world_size: int,
    train_tasks: Sequence[str],
    counts_per_cycle: dict[str, int],
    sampling_seed: int = 42,
    seed_base: int = STAGE_B_SUPERVISED_SEED_BASE,
    task_seed_stride: int = STAGE_B_TASK_SEED_STRIDE,
) -> list[TrainingSample]:
    """Return one rank's reconstructible Stage-B task/seed assignments."""

    if global_step < 1:
        raise ValueError("global_step is one-indexed and must be positive")
    if world_size < 1 or not 0 <= rank < world_size:
        raise ValueError("invalid distributed rank/world_size")
    if global_batch_size % world_size:
        raise ValueError("global_batch_size must be divisible by world_size")
    if int(seed_base) == 42 * TRAIN_SEED_MULTIPLIER:
        raise ValueError("Stage-B seed namespace must be disjoint from Stage A")
    if int(task_seed_stride) < STAGE_B_SAMPLING_CYCLE:
        raise ValueError("task_seed_stride is too small")

    counts = validate_sampling_counts(
        counts_per_cycle,
        train_tasks=train_tasks,
    )
    local_batch_size = int(global_batch_size) // int(world_size)
    global_start = (int(global_step) - 1) * int(global_batch_size)
    rank_start = global_start + int(rank) * local_batch_size
    cycle_cache: dict[int, list[str]] = {}
    samples: list[TrainingSample] = []
    for global_index in range(rank_start, rank_start + local_batch_size):
        cycle_index, cycle_position = divmod(
            global_index,
            STAGE_B_SAMPLING_CYCLE,
        )
        cycle = cycle_cache.get(cycle_index)
        if cycle is None:
            cycle = _stable_stage_b_cycle(
                counts,
                cycle_index=cycle_index,
                sampling_seed=sampling_seed,
            )
            cycle_cache[cycle_index] = cycle
        task_type = cycle[cycle_position]
        prefix_count = cycle[:cycle_position].count(task_type)
        task_counter = cycle_index * counts[task_type] + prefix_count
        if task_counter >= int(task_seed_stride):
            raise RuntimeError(
                f"Stage-B seed block exhausted for {task_type}: {task_counter}"
            )
        sample_seed = (
            int(seed_base)
            + TASK_TO_ID[task_type] * int(task_seed_stride)
            + task_counter
        )
        samples.append(TrainingSample(task_type=task_type, seed=sample_seed))
    return samples


def create_training_batch(
    tokenizer: Any,
    *,
    seeds: Sequence[int],
    task_types: Sequence[str] = ATOMIC_TASKS,
    n_ctx: int = 128,
    n_distractors: int = 2,
) -> torch.Tensor:
    """Create one padded next-token batch with upstream-compatible ordering."""

    import torch

    tasks = validate_task_types(task_types)
    if n_ctx < 2:
        raise ValueError("n_ctx must be at least two")
    rows: list[list[int]] = []
    for sample_seed in seeds:
        rng = random.Random(int(sample_seed))
        task_type = rng.choice(tasks)
        input_text, output_text = generate_sample(
            task_type,
            int(sample_seed),
            n_distractors=n_distractors,
        )
        token_ids = list(tokenizer.encode(input_text + output_text))
        token_ids = token_ids[:n_ctx]
        if len(token_ids) < n_ctx:
            token_ids.extend([0] * (n_ctx - len(token_ids)))
        rows.append(token_ids)
    return torch.tensor(rows, dtype=torch.long)


def create_training_batch_from_samples(
    tokenizer: Any,
    *,
    samples: Sequence[TrainingSample],
    n_ctx: int = 128,
    n_distractors: int = 2,
    renderer_version: str = STAGE_B_LEGACY_RENDERER_VERSION,
) -> torch.Tensor:
    """Create a padded batch from explicit task/seed assignments."""

    import torch

    if n_ctx < 2:
        raise ValueError("n_ctx must be at least two")
    if not samples:
        raise ValueError("samples must not be empty")
    rows: list[list[int]] = []
    for sample in samples:
        validate_task_types([sample.task_type])
        input_text, output_text = generate_sample(
            sample.task_type,
            sample.seed,
            n_distractors=n_distractors,
            renderer_version=renderer_version,
        )
        token_ids = list(tokenizer.encode(input_text + output_text))
        if len(token_ids) > n_ctx:
            raise RuntimeError(
                "Stage-B sample exceeds context without truncation permission: "
                f"task={sample.task_type}, seed={sample.seed}, "
                f"tokens={len(token_ids)}, context={n_ctx}"
            )
        if len(token_ids) < n_ctx:
            token_ids.extend([0] * (n_ctx - len(token_ids)))
        rows.append(token_ids)
    return torch.tensor(rows, dtype=torch.long)


def create_evaluation_dataset(
    *,
    task_types: Sequence[str] = ALL_TASKS,
    samples_per_task: int,
    seed_base: int,
    n_distractors: int = 2,
    surface_family: str = "narrow",
) -> list[EvaluationSample]:
    """Generate a fixed suite with sequential, disjoint public-style seeds."""

    tasks = validate_task_types(task_types)
    if samples_per_task < 1:
        raise ValueError("samples_per_task must be positive")

    records: list[EvaluationSample] = []
    current_seed = int(seed_base)
    for task_type in tasks:
        collected = 0
        attempts = 0
        max_attempts = samples_per_task * 3
        while collected < samples_per_task and attempts < max_attempts:
            sample_seed = current_seed
            current_seed += 1
            attempts += 1
            try:
                input_text, output_text = generate_sample(
                    task_type,
                    sample_seed,
                    n_distractors=n_distractors,
                    surface_family=surface_family,
                )
            except Exception:
                continue
            records.append(
                EvaluationSample(
                    task_type=task_type,
                    seed=sample_seed,
                    input_text=input_text,
                    output_text=output_text,
                )
            )
            collected += 1
        if collected != samples_per_task:
            raise RuntimeError(
                f"could generate only {collected}/{samples_per_task} "
                f"samples for {task_type}"
            )
    return records


def extract_first_tactic(text: str) -> str:
    """Extract the first generated tactic exactly as the public evaluator does."""

    for marker in ("state_1", "proof complete", "no goals"):
        if marker in text:
            text = text.split(marker, 1)[0]
            break
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if not lines:
        return ""
    if lines[0] == "state_0_tactic_0:" and len(lines) > 1:
        return lines[1]
    first_line = lines[0]
    if ":" in first_line and first_line.startswith("state"):
        first_line = first_line.split(":", 1)[1].strip()
    return first_line


def expected_first_tactic(output_text: str) -> str:
    lines = output_text.strip().splitlines()
    if len(lines) < 2:
        return output_text.strip()
    return lines[1].strip()


def empty_task_counts(task_types: Sequence[str] = ALL_TASKS) -> dict[str, dict[str, int]]:
    return {task: {"correct": 0, "total": 0} for task in validate_task_types(task_types)}


def accuracy_from_counts(
    task_counts: dict[str, dict[str, int]],
) -> dict[str, Any]:
    """Compute only strict-exact-match accuracy aggregates."""

    task_accuracy: dict[str, float] = {}
    for task, counts in task_counts.items():
        total = int(counts["total"])
        correct = int(counts["correct"])
        task_accuracy[task] = 100.0 * correct / total if total else 0.0

    atomic_values = [task_accuracy[t] for t in ATOMIC_TASKS if t in task_accuracy]
    composition_values = [
        task_accuracy[t] for t in COMPOSITION_TASKS if t in task_accuracy
    ]
    overall_correct = sum(int(v["correct"]) for v in task_counts.values())
    overall_total = sum(int(v["total"]) for v in task_counts.values())
    atomic_macro = sum(atomic_values) / len(atomic_values) if atomic_values else None
    composition_macro = (
        sum(composition_values) / len(composition_values)
        if composition_values
        else None
    )
    gap = (
        atomic_macro - composition_macro
        if atomic_macro is not None and composition_macro is not None
        else None
    )
    return {
        "metric": "strict_first_tactic_exact_match",
        "task_counts": task_counts,
        "task_accuracy": task_accuracy,
        "atomic_macro_accuracy": atomic_macro,
        "composition_macro_accuracy": composition_macro,
        "atomic_to_composition_gap": gap,
        "overall_correct": overall_correct,
        "overall_total": overall_total,
        "overall_accuracy": (
            100.0 * overall_correct / overall_total if overall_total else 0.0
        ),
    }
