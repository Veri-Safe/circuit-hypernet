"""Condition-safe self-generated S5 trace records and chunk storage."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, replace
import hashlib
import json
import os
from pathlib import Path
import random
from typing import Any, Iterable, Iterator, Mapping, Sequence

import torch
from safetensors.torch import load_file, save_file

from ..protocol import ALL_TASKS, DEPTH3_TASKS, ATOMIC_TASKS, generate_sample


TRACE_FORMAT = "s5_self_generated_full_output_trace_v1"
TRACE_PARITY_FALLBACK_POLICY = (
    "canonical_batch_one_prefix_replay_after_batch_parity_failure_v2"
)
# Circuit extraction can score the dedicated Depth3 OOD prompts as well as the
# nine RL task families.  Keep the canonical nine-task order unchanged and
# append stable IDs for the three Depth3 labels; training corpus validators
# continue to use ``ALL_TASKS`` and therefore remain unaffected.
TRACE_TASKS = ALL_TASKS + DEPTH3_TASKS
TASK_TO_ID = {task: index for index, task in enumerate(TRACE_TASKS)}
ID_TO_TASK = {index: task for task, index in TASK_TO_ID.items()}


def _semantic_sha256(value: Any) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def trace_generation_code_sha256() -> str:
    """Hash code that determines prompt serialization and greedy Trace bytes."""

    package_root = Path(__file__).resolve().parents[1]
    project_root = Path(__file__).resolve().parents[3]
    paths = [
        package_root / "logic_tasks.py",
        package_root / "protocol.py",
        package_root / "modeling.py",
        package_root / "config.py",
        package_root / "circuit" / "model_io.py",
        package_root / "circuit" / "traces.py",
        project_root / "scripts" / "generate_circuit_traces.py",
    ]
    paths.extend(sorted((package_root / "upstream_runtime").rglob("*.py")))
    digest = hashlib.sha256()
    for path in sorted(paths):
        digest.update(path.relative_to(project_root).as_posix().encode("utf-8"))
        digest.update(path.read_bytes())
    return digest.hexdigest()


def rl_trace_generation_code_sha256() -> str:
    """Hash the HyperNet/RL Trace producer and its prompt/runtime inputs.

    This identity is deliberately separate from ``trace_generation_code_sha256``:
    the latter is the legacy train/validation Trace backend used by the width
    bank, while this one identifies the rendered HyperNet RL Trace stream.
    """

    package_root = Path(__file__).resolve().parents[1]
    project_root = Path(__file__).resolve().parents[3]
    paths = [
        package_root / "logic_tasks.py",
        package_root / "protocol.py",
        package_root / "grpo_config.py",
        package_root / "modeling.py",
        package_root / "config.py",
        package_root / "circuit" / "rl_union.py",
        package_root / "circuit" / "hypernet_sft_corpus.py",
        package_root / "circuit" / "hypernet_validation_corpus.py",
        package_root / "circuit" / "hypernet_final_corpus.py",
        package_root / "circuit" / "composition_circuit_corpus.py",
        package_root / "circuit" / "model_io.py",
        package_root / "circuit" / "settings.py",
        package_root / "circuit" / "traces.py",
        project_root / "scripts" / "generate_rl_circuit_traces.py",
    ]
    runtime_compat = (
        project_root.parent / "temp_s" / "s5_trace_runtime_compat" / "sitecustomize.py"
    )
    if runtime_compat.is_file():
        paths.append(runtime_compat)
    hash_root = project_root.parent
    digest = hashlib.sha256()
    for path in sorted(paths):
        digest.update(path.relative_to(hash_root).as_posix().encode("utf-8"))
        digest.update(path.read_bytes())
    return digest.hexdigest()


@dataclass(frozen=True)
class CircuitTrace:
    event_key: str
    task_type: str
    seed: int
    prompt_token_ids: tuple[int, ...]
    generated_token_ids: tuple[int, ...]
    eligible_token_count: int | None = None

    @property
    def condition_length(self) -> int:
        return len(self.prompt_token_ids)

    @property
    def target_count(self) -> int:
        return len(self.generated_token_ids)

    @property
    def replay_token_ids(self) -> tuple[int, ...]:
        if not self.generated_token_ids:
            raise ValueError("trace has no generated target token")
        return self.prompt_token_ids + self.generated_token_ids[:-1]

    @property
    def full_token_ids(self) -> tuple[int, ...]:
        return self.prompt_token_ids + self.generated_token_ids

    @property
    def eligible_tokens(self) -> int:
        return (
            len(self.full_token_ids)
            if self.eligible_token_count is None
            else int(self.eligible_token_count)
        )

    def validate(self, *, context_length: int = 128) -> None:
        if len(self.event_key) != 64:
            raise ValueError("trace event_key must be SHA256")
        if self.task_type not in TASK_TO_ID:
            raise ValueError(f"unknown trace task: {self.task_type}")
        if not self.prompt_token_ids or not self.generated_token_ids:
            raise ValueError("prompt and generated token IDs must be nonempty")
        if len(self.full_token_ids) > context_length:
            raise ValueError("trace exceeds the S5 context length")
        if not 0 < self.eligible_tokens <= len(self.full_token_ids):
            raise ValueError("trace eligible-token count is invalid")
        if min(self.full_token_ids) < 0 or max(self.full_token_ids) >= 2048:
            raise ValueError("trace token ID is outside the S5 vocabulary")
        expected = trace_event_key(
            task_type=self.task_type,
            seed=self.seed,
            prompt_token_ids=self.prompt_token_ids,
            generated_token_ids=self.generated_token_ids,
        )
        if expected != self.event_key:
            raise ValueError("trace event identity changed")


def trace_event_key(
    *,
    task_type: str,
    seed: int,
    prompt_token_ids: Sequence[int],
    generated_token_ids: Sequence[int],
) -> str:
    value = {
        "format": TRACE_FORMAT,
        "task_type": str(task_type),
        "seed": int(seed),
        "prompt_token_ids": [int(item) for item in prompt_token_ids],
        "generated_token_ids": [int(item) for item in generated_token_ids],
    }
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def deterministic_logic_sample(
    task_type: str, seed: int, *, n_distractors: int
) -> tuple[str, str]:
    """Isolate the generator's legacy global premise shuffle from process RNG."""

    state = random.getstate()
    random.seed(int(seed) ^ 0x5A17_C1C7)
    try:
        return generate_sample(
            task_type, int(seed), n_distractors=int(n_distractors)
        )
    finally:
        random.setstate(state)


@torch.inference_mode()
def greedy_generated_token_ids(
    model: torch.nn.Module,
    tokenizer: Any,
    *,
    prompt_token_ids: Sequence[int],
    device: torch.device,
    context_length: int = 128,
    max_new_tokens: int = 50,
    stop_markers: Sequence[str] = ("proof complete", "no goals"),
) -> tuple[int, ...]:
    """Canonical evaluator-aligned batch-one greedy decoding."""

    prompt = tuple(int(item) for item in prompt_token_ids)
    if not prompt or len(prompt) >= context_length:
        raise ValueError("prompt does not fit the circuit generation context")
    tokens = torch.tensor([prompt], dtype=torch.long, device=device)
    generated: list[int] = []
    for _ in range(int(max_new_tokens)):
        if tokens.shape[1] >= context_length:
            break
        logits = model(tokens)[0]
        next_id = int(logits[0, -1].argmax().item())
        generated.append(next_id)
        tokens = torch.cat(
            (tokens, torch.tensor([[next_id]], dtype=torch.long, device=device)), dim=1
        )
        decoded = tokenizer.decode(tokens[0].tolist())
        if any(marker in decoded for marker in stop_markers):
            break
    if not generated:
        raise RuntimeError("greedy generation returned no output token")
    return tuple(generated)


@torch.inference_mode()
def greedy_generated_token_ids_batched(
    model: torch.nn.Module,
    tokenizer: Any,
    prompt_token_ids: Sequence[Sequence[int]],
    *,
    device: torch.device,
    batch_size: int,
    context_length: int = 128,
    max_new_tokens: int = 50,
    stop_markers: Sequence[str] = ("proof complete", "no goals"),
) -> tuple[tuple[int, ...], ...]:
    """Decode independent prompts in same-length batches without padding."""

    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    prompts = [tuple(int(item) for item in prompt) for prompt in prompt_token_ids]
    if not prompts:
        return ()
    for prompt in prompts:
        if not prompt or len(prompt) >= context_length:
            raise ValueError("prompt does not fit the circuit generation context")

    buckets: dict[int, list[int]] = defaultdict(list)
    for index, prompt in enumerate(prompts):
        buckets[len(prompt)].append(index)
    results: list[tuple[int, ...] | None] = [None] * len(prompts)

    for prompt_length in sorted(buckets):
        indices = buckets[prompt_length]
        for chunk_start in range(0, len(indices), int(batch_size)):
            active = indices[chunk_start : chunk_start + int(batch_size)]
            tokens = torch.tensor(
                [prompts[index] for index in active],
                dtype=torch.long,
                device=device,
            )
            generated = {index: [] for index in active}
            for _ in range(int(max_new_tokens)):
                if tokens.shape[1] >= context_length:
                    break
                logits = model(tokens)[0]
                next_tensor = logits[:, -1].argmax(dim=-1)
                tokens = torch.cat((tokens, next_tensor[:, None]), dim=1)
                token_rows = tokens.detach().cpu().tolist()
                next_ids = [int(row[-1]) for row in token_rows]
                decoded_rows = [tokenizer.decode(row) for row in token_rows]
                survivors: list[int] = []
                survivor_rows: list[int] = []
                for row, (index, next_id, decoded) in enumerate(
                    zip(active, next_ids, decoded_rows, strict=True)
                ):
                    generated[index].append(next_id)
                    if any(marker in decoded for marker in stop_markers):
                        results[index] = tuple(generated[index])
                    else:
                        survivors.append(index)
                        survivor_rows.append(row)
                if not survivors:
                    active = []
                    break
                active = survivors
                tokens = tokens.index_select(
                    0,
                    torch.tensor(survivor_rows, dtype=torch.long, device=device),
                )
            for index in active:
                results[index] = tuple(generated[index])

    if any(result is None or not result for result in results):
        raise RuntimeError("batched greedy generation returned an empty result")
    return tuple(result for result in results if result is not None)


def trace_generation_contract(config: Mapping[str, Any]) -> dict[str, Any]:
    """Return the manifest identity for the frozen formal Trace backend."""

    return {
        "backend": str(config["generation_backend"]),
        "batch_size": int(config["generation_batch_size"]),
        "pool_size": int(config["generation_pool_size"]),
        "teacher_forced_verify_all": bool(config["teacher_forced_verify_all"]),
        "canonical_regeneration_interval": int(
            config["canonical_regeneration_interval"]
        ),
    }


def greedy_full_output_traces_batched(
    model: torch.nn.Module,
    tokenizer: Any,
    *,
    requests: Sequence[tuple[str, int]],
    device: torch.device,
    batch_size: int,
    n_distractors: int = 2,
    context_length: int = 128,
    max_new_tokens: int = 50,
    stop_markers: Sequence[str] = ("proof complete", "no goals"),
    teacher_forced_verify_all: bool = True,
    canonical_regeneration_mask: Sequence[bool] | None = None,
) -> tuple[CircuitTrace, ...]:
    """Generate a deterministic request batch and fail closed on parity."""

    if not requests:
        return ()
    if canonical_regeneration_mask is None:
        canonical_regeneration_mask = (False,) * len(requests)
    if len(canonical_regeneration_mask) != len(requests):
        raise ValueError("canonical regeneration mask does not match requests")
    prompts = []
    for task_type, seed in requests:
        input_text, _gold_output = deterministic_logic_sample(
            task_type, int(seed), n_distractors=n_distractors
        )
        prompts.append(tuple(int(item) for item in tokenizer.encode(input_text)))
    generated_rows = greedy_generated_token_ids_batched(
        model,
        tokenizer,
        prompts,
        device=device,
        batch_size=batch_size,
        context_length=context_length,
        max_new_tokens=max_new_tokens,
        stop_markers=stop_markers,
    )
    results = []
    for index, ((task_type, seed), prompt, generated) in enumerate(
        zip(requests, prompts, generated_rows, strict=True)
    ):
        if teacher_forced_verify_all:
            audit = verify_greedy_trace_teacher_forced(
                model,
                tokenizer,
                prompt_token_ids=prompt,
                generated_token_ids=generated,
                device=device,
                context_length=context_length,
                max_new_tokens=max_new_tokens,
                stop_markers=stop_markers,
            )
            if not audit["passes"]:
                raise RuntimeError(
                    f"batched Trace failed teacher-forced parity for {task_type}/{seed}: {audit}"
                )
        if canonical_regeneration_mask[index]:
            canonical = greedy_generated_token_ids(
                model,
                tokenizer,
                prompt_token_ids=prompt,
                device=device,
                context_length=context_length,
                max_new_tokens=max_new_tokens,
                stop_markers=stop_markers,
            )
            if canonical != generated:
                raise RuntimeError(
                    f"batched Trace failed canonical regeneration for {task_type}/{seed}"
                )
        trace = CircuitTrace(
            event_key=trace_event_key(
                task_type=task_type,
                seed=seed,
                prompt_token_ids=prompt,
                generated_token_ids=generated,
            ),
            task_type=task_type,
            seed=int(seed),
            prompt_token_ids=prompt,
            generated_token_ids=generated,
        )
        trace.validate(context_length=context_length)
        results.append(trace)
    return tuple(results)


@torch.inference_mode()
def greedy_full_output_traces_from_prompts_batched(
    model: torch.nn.Module,
    tokenizer: Any,
    *,
    requests: Sequence[tuple[str, int, Sequence[int]]],
    device: torch.device,
    batch_size: int,
    context_length: int = 128,
    max_new_tokens: int = 50,
    stop_markers: Sequence[str] = ("proof complete", "no goals"),
    teacher_forced_verify_all: bool = True,
    canonical_regeneration_mask: Sequence[bool] | None = None,
    parity_fallbacks: list[dict[str, Any]] | None = None,
) -> tuple[CircuitTrace, ...]:
    """Generate verified traces from already rendered, identity-checked prompts."""

    if not requests:
        return ()
    if canonical_regeneration_mask is None:
        canonical_regeneration_mask = (False,) * len(requests)
    if len(canonical_regeneration_mask) != len(requests):
        raise ValueError("canonical regeneration mask does not match prompt requests")
    prompts = [tuple(int(item) for item in request[2]) for request in requests]
    generated_rows = greedy_generated_token_ids_batched(
        model,
        tokenizer,
        prompts,
        device=device,
        batch_size=batch_size,
        context_length=context_length,
        max_new_tokens=max_new_tokens,
        stop_markers=stop_markers,
    )
    results = []
    for index, ((task_type, seed, _prompt), prompt, generated) in enumerate(
        zip(requests, prompts, generated_rows, strict=True)
    ):
        generated = _resolve_verified_generated_token_ids(
            model,
            tokenizer,
            task_type=str(task_type),
            seed=int(seed),
            request_index=index,
            prompt_token_ids=prompt,
            generated_token_ids=generated,
            device=device,
            context_length=context_length,
            max_new_tokens=max_new_tokens,
            stop_markers=stop_markers,
            teacher_forced_verify_all=teacher_forced_verify_all,
            canonical_regeneration_required=bool(canonical_regeneration_mask[index]),
            parity_fallbacks=parity_fallbacks,
            error_prefix="rendered prompt",
        )
        trace = CircuitTrace(
            event_key=trace_event_key(
                task_type=task_type,
                seed=seed,
                prompt_token_ids=prompt,
                generated_token_ids=generated,
            ),
            task_type=str(task_type),
            seed=int(seed),
            prompt_token_ids=prompt,
            generated_token_ids=generated,
        )
        trace.validate(context_length=context_length)
        results.append(trace)
    return tuple(results)


@torch.inference_mode()
def greedy_generated_token_ids_right_padded(
    model: torch.nn.Module,
    tokenizer: Any,
    prompt_token_ids: Sequence[Sequence[int]],
    *,
    device: torch.device,
    batch_size: int,
    context_length: int = 128,
    max_new_tokens: int = 50,
    stop_markers: Sequence[str] = ("proof complete", "no goals"),
    pad_token_id: int = 0,
) -> tuple[tuple[int, ...], ...]:
    """Decode mixed-length prompts with causally invisible right padding.

    Every row keeps its real tokens at absolute positions ``0..length-1``.
    Logits are gathered at ``length-1``, before any padding, so causal
    attention prevents padded positions from affecting the next-token result.
    """

    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if pad_token_id < 0:
        raise ValueError("pad_token_id must be nonnegative")
    prompts = [tuple(int(item) for item in prompt) for prompt in prompt_token_ids]
    if not prompts:
        return ()
    for prompt in prompts:
        if not prompt or len(prompt) >= context_length:
            raise ValueError("prompt does not fit the circuit generation context")
    results: list[tuple[int, ...] | None] = [None] * len(prompts)

    for chunk_start in range(0, len(prompts), int(batch_size)):
        active = list(
            range(chunk_start, min(len(prompts), chunk_start + int(batch_size)))
        )
        storage_cpu = torch.full(
            (len(active), int(context_length)),
            int(pad_token_id),
            dtype=torch.long,
        )
        for row, index in enumerate(active):
            storage_cpu[row, : len(prompts[index])] = torch.tensor(prompts[index])
        storage = storage_cpu.to(device=device, non_blocking=True)
        lengths = torch.tensor(
            [len(prompts[index]) for index in active],
            dtype=torch.long,
            device=device,
        )
        generated = {index: [] for index in active}

        for _ in range(int(max_new_tokens)):
            length_values = [int(item) for item in lengths.detach().cpu().tolist()]
            within_context_rows = [
                row for row, length in enumerate(length_values) if length < context_length
            ]
            if len(within_context_rows) != len(active):
                keep = set(within_context_rows)
                for row, index in enumerate(active):
                    if row not in keep:
                        results[index] = tuple(generated[index])
                if not within_context_rows:
                    active = []
                    break
                row_index = torch.tensor(
                    within_context_rows, dtype=torch.long, device=device
                )
                active = [active[row] for row in within_context_rows]
                storage = storage.index_select(0, row_index)
                lengths = lengths.index_select(0, row_index)

            max_length = int(lengths.max().item())
            logits = model(storage[:, :max_length])[0]
            rows = torch.arange(len(active), dtype=torch.long, device=device)
            next_tensor = logits[rows, lengths - 1].argmax(dim=-1)
            storage[rows, lengths] = next_tensor
            lengths = lengths + 1

            length_values = [int(item) for item in lengths.detach().cpu().tolist()]
            token_rows = storage[:, : max(length_values)].detach().cpu().tolist()
            survivors: list[int] = []
            survivor_rows: list[int] = []
            for row, (index, length) in enumerate(
                zip(active, length_values, strict=True)
            ):
                next_id = int(token_rows[row][length - 1])
                generated[index].append(next_id)
                decoded = tokenizer.decode(token_rows[row][:length])
                if any(marker in decoded for marker in stop_markers):
                    results[index] = tuple(generated[index])
                else:
                    survivors.append(index)
                    survivor_rows.append(row)
            if not survivors:
                active = []
                break
            active = survivors
            row_index = torch.tensor(survivor_rows, dtype=torch.long, device=device)
            storage = storage.index_select(0, row_index)
            lengths = lengths.index_select(0, row_index)

        for index in active:
            results[index] = tuple(generated[index])

    if any(result is None or not result for result in results):
        raise RuntimeError("right-padded greedy generation returned an empty result")
    return tuple(result for result in results if result is not None)


@torch.inference_mode()
def verify_greedy_trace_teacher_forced(
    model: torch.nn.Module,
    tokenizer: Any,
    *,
    prompt_token_ids: Sequence[int],
    generated_token_ids: Sequence[int],
    device: torch.device,
    context_length: int = 128,
    max_new_tokens: int = 50,
    stop_markers: Sequence[str] = ("proof complete", "no goals"),
) -> dict[str, Any]:
    """Verify all greedy decisions with one canonical batch-one forward pass."""

    prompt = tuple(int(item) for item in prompt_token_ids)
    generated = tuple(int(item) for item in generated_token_ids)
    if not prompt or not generated:
        raise ValueError("teacher-forced verification requires prompt and output tokens")
    if len(prompt) + len(generated) > context_length:
        raise ValueError("teacher-forced verification sequence exceeds context")
    replay = prompt + generated[:-1]
    logits = model(torch.tensor([replay], dtype=torch.long, device=device))[0]
    positions = torch.arange(
        len(prompt) - 1,
        len(prompt) + len(generated) - 1,
        dtype=torch.long,
        device=device,
    )
    predicted = tuple(
        int(item)
        for item in logits[0].index_select(0, positions).argmax(dim=-1).cpu().tolist()
    )
    mismatches = tuple(
        index
        for index, (actual, expected) in enumerate(
            zip(generated, predicted, strict=True)
        )
        if actual != expected
    )
    marker_steps = tuple(
        index
        for index in range(len(generated))
        if any(
            marker in tokenizer.decode(list(prompt + generated[: index + 1]))
            for marker in stop_markers
        )
    )
    stop_valid = (
        (bool(marker_steps) and marker_steps[0] == len(generated) - 1)
        or (
            not marker_steps
            and (
                len(generated) == int(max_new_tokens)
                or len(prompt) + len(generated) == int(context_length)
            )
        )
    )
    return {
        "passes": not mismatches and stop_valid,
        "token_mismatch_count": len(mismatches),
        "first_token_mismatch": mismatches[0] if mismatches else None,
        "stop_valid": bool(stop_valid),
        "first_marker_step": marker_steps[0] if marker_steps else None,
    }


@torch.inference_mode()
def verify_greedy_trace_prefix_replay(
    model: torch.nn.Module,
    tokenizer: Any,
    *,
    prompt_token_ids: Sequence[int],
    generated_token_ids: Sequence[int],
    device: torch.device,
    context_length: int = 128,
    max_new_tokens: int = 50,
    stop_markers: Sequence[str] = ("proof complete", "no goals"),
) -> dict[str, Any]:
    """Verify tokens with independent batch-one prefix forwards.

    The teacher-forced audit uses one forward over the complete replay
    sequence.  A GPU kernel can make an earlier argmax depend on that forward
    shape at a floating-point boundary.  This replay mirrors the canonical
    autoregressive decoder: every token is checked with a fresh batch-one
    forward containing only the prompt and accepted generated prefix.
    """

    prompt = tuple(int(item) for item in prompt_token_ids)
    generated = tuple(int(item) for item in generated_token_ids)
    if not prompt or not generated:
        raise ValueError("prefix replay requires prompt and output tokens")
    if len(prompt) + len(generated) > context_length:
        raise ValueError("prefix replay sequence exceeds context")

    predicted: list[int] = []
    for index in range(len(generated)):
        prefix = prompt + generated[:index]
        logits = model(torch.tensor([prefix], dtype=torch.long, device=device))[0]
        predicted.append(int(logits[0, -1].argmax().item()))

    mismatches = tuple(
        index
        for index, (actual, expected) in enumerate(
            zip(generated, predicted, strict=True)
        )
        if actual != expected
    )
    marker_steps = tuple(
        index
        for index in range(len(generated))
        if any(
            marker in tokenizer.decode(list(prompt + generated[: index + 1]))
            for marker in stop_markers
        )
    )
    stop_valid = (
        (bool(marker_steps) and marker_steps[0] == len(generated) - 1)
        or (
            not marker_steps
            and (
                len(generated) == int(max_new_tokens)
                or len(prompt) + len(generated) == int(context_length)
            )
        )
    )
    return {
        "passes": not mismatches and stop_valid,
        "token_mismatch_count": len(mismatches),
        "first_token_mismatch": mismatches[0] if mismatches else None,
        "stop_valid": bool(stop_valid),
        "first_marker_step": marker_steps[0] if marker_steps else None,
        "replay_mode": "batch_one_prefix",
        "replayed_token_count": len(predicted),
        "predicted_token_ids_sha256": _semantic_sha256(predicted),
        "generated_token_ids_sha256": _semantic_sha256(list(generated)),
    }


@torch.inference_mode()
def _resolve_verified_generated_token_ids(
    model: torch.nn.Module,
    tokenizer: Any,
    *,
    task_type: str,
    seed: int,
    request_index: int,
    prompt_token_ids: Sequence[int],
    generated_token_ids: Sequence[int],
    device: torch.device,
    context_length: int,
    max_new_tokens: int,
    stop_markers: Sequence[str],
    teacher_forced_verify_all: bool,
    canonical_regeneration_required: bool,
    parity_fallbacks: list[dict[str, Any]] | None,
    error_prefix: str,
) -> tuple[int, ...]:
    """Resolve one batch result while retaining the strict parity gate.

    Exact-length batching is the throughput path, but CUDA kernels can make a
    greedy argmax at a narrow numerical boundary depend on the batch shape.  A
    failed batch result is therefore retried with the evaluator's canonical
    batch-one autoregressive loop. The replacement is accepted only when its
    prefix replay passes; the full-sequence audit is retained as a diagnostic.
    A canonical sample that fails prefix replay remains a hard failure.
    """

    prompt = tuple(int(item) for item in prompt_token_ids)
    batched = tuple(int(item) for item in generated_token_ids)
    current = batched
    batched_audit: dict[str, Any] | None = None
    canonical: tuple[int, ...] | None = None
    canonical_audit: dict[str, Any] | None = None
    prefix_replay_audit: dict[str, Any] | None = None
    fallback_reason: str | None = None

    if teacher_forced_verify_all:
        batched_audit = verify_greedy_trace_teacher_forced(
            model,
            tokenizer,
            prompt_token_ids=prompt,
            generated_token_ids=batched,
            device=device,
            context_length=context_length,
            max_new_tokens=max_new_tokens,
            stop_markers=stop_markers,
        )
        if not batched_audit["passes"]:
            fallback_reason = "teacher_forced_parity"
            canonical = greedy_generated_token_ids(
                model,
                tokenizer,
                prompt_token_ids=prompt,
                device=device,
                context_length=context_length,
                max_new_tokens=max_new_tokens,
                stop_markers=stop_markers,
            )
            canonical_audit = verify_greedy_trace_teacher_forced(
                model,
                tokenizer,
                prompt_token_ids=prompt,
                generated_token_ids=canonical,
                device=device,
                context_length=context_length,
                max_new_tokens=max_new_tokens,
                stop_markers=stop_markers,
            )
            prefix_replay_audit = verify_greedy_trace_prefix_replay(
                model,
                tokenizer,
                prompt_token_ids=prompt,
                generated_token_ids=canonical,
                device=device,
                context_length=context_length,
                max_new_tokens=max_new_tokens,
                stop_markers=stop_markers,
            )
            if not prefix_replay_audit["passes"]:
                raise RuntimeError(
                    f"{error_prefix} Trace failed canonical fallback prefix-replay "
                    f"parity for {task_type}/{seed}: "
                    f"batched_audit={batched_audit}, canonical_audit={canonical_audit}, "
                    f"prefix_replay_audit={prefix_replay_audit}"
                )
            current = canonical

    if canonical_regeneration_required:
        if canonical is None:
            canonical = greedy_generated_token_ids(
                model,
                tokenizer,
                prompt_token_ids=prompt,
                device=device,
                context_length=context_length,
                max_new_tokens=max_new_tokens,
                stop_markers=stop_markers,
            )
        if canonical != current:
            if not teacher_forced_verify_all:
                raise RuntimeError(
                    f"{error_prefix} Trace failed canonical regeneration for "
                    f"{task_type}/{seed}: batch-one output differs from the "
                    "batched output while teacher-forced verification is disabled"
                )
            if fallback_reason is None:
                fallback_reason = "canonical_regeneration"
            current = canonical
            if teacher_forced_verify_all:
                if canonical_audit is None:
                    canonical_audit = verify_greedy_trace_teacher_forced(
                        model,
                        tokenizer,
                        prompt_token_ids=prompt,
                        generated_token_ids=canonical,
                        device=device,
                        context_length=context_length,
                        max_new_tokens=max_new_tokens,
                        stop_markers=stop_markers,
                    )
                if prefix_replay_audit is None:
                    prefix_replay_audit = verify_greedy_trace_prefix_replay(
                        model,
                        tokenizer,
                        prompt_token_ids=prompt,
                        generated_token_ids=canonical,
                        device=device,
                        context_length=context_length,
                        max_new_tokens=max_new_tokens,
                        stop_markers=stop_markers,
                    )
                if not prefix_replay_audit["passes"]:
                    raise RuntimeError(
                        f"{error_prefix} Trace failed canonical regeneration "
                        f"prefix-replay parity for {task_type}/{seed}: "
                        f"batched_audit={batched_audit}, "
                        f"canonical_audit={canonical_audit}, "
                        f"prefix_replay_audit={prefix_replay_audit}"
                    )

    if fallback_reason is not None and parity_fallbacks is not None:
        parity_fallbacks.append(
            {
                "request_index": int(request_index),
                "task_type": str(task_type),
                "seed": int(seed),
                "reason": fallback_reason,
                "batched_token_count": len(batched),
                "canonical_token_count": len(current),
                "batched_token_ids_sha256": _semantic_sha256(list(batched)),
                "canonical_token_ids_sha256": _semantic_sha256(list(current)),
                "batched_audit": batched_audit,
                "canonical_audit": canonical_audit,
                "prefix_replay_audit": prefix_replay_audit,
            }
        )
    return current


@torch.inference_mode()
def greedy_full_output_trace(
    model: torch.nn.Module,
    tokenizer: Any,
    *,
    task_type: str,
    seed: int,
    device: torch.device,
    n_distractors: int = 2,
    context_length: int = 128,
    max_new_tokens: int = 50,
    stop_markers: Sequence[str] = ("proof complete", "no goals"),
) -> CircuitTrace:
    input_text, _gold_output = deterministic_logic_sample(
        task_type, int(seed), n_distractors=n_distractors
    )
    prompt = tuple(int(item) for item in tokenizer.encode(input_text))
    generated = greedy_generated_token_ids(
        model,
        tokenizer,
        prompt_token_ids=prompt,
        device=device,
        context_length=context_length,
        max_new_tokens=max_new_tokens,
        stop_markers=stop_markers,
    )
    result = CircuitTrace(
        event_key=trace_event_key(
            task_type=task_type,
            seed=seed,
            prompt_token_ids=prompt,
            generated_token_ids=generated,
        ),
        task_type=task_type,
        seed=int(seed),
        prompt_token_ids=prompt,
        generated_token_ids=generated,
    )
    result.validate(context_length=context_length)
    return result


def balanced_trace_stream(
    model: torch.nn.Module,
    tokenizer: Any,
    *,
    tasks: Sequence[str],
    seed_base: int,
    tokens_per_task: int,
    device: torch.device,
    n_distractors: int = 2,
    context_length: int = 128,
    max_new_tokens: int = 50,
    start_offsets: Mapping[str, int] | None = None,
    initial_totals: Mapping[str, int] | None = None,
) -> Iterator[CircuitTrace]:
    if tokens_per_task <= 0:
        raise ValueError("tokens_per_task must be positive")
    unknown = set(tasks) - set(ALL_TASKS)
    if unknown:
        raise ValueError(f"unknown tasks: {sorted(unknown)}")
    totals = {task: int((initial_totals or {}).get(task, 0)) for task in tasks}
    offsets = {task: int((start_offsets or {}).get(task, 0)) for task in tasks}
    if any(not 0 <= totals[task] <= tokens_per_task for task in tasks):
        raise ValueError("initial token totals are outside the declared quota")
    while any(total < tokens_per_task for total in totals.values()):
        for task_index, task in enumerate(tasks):
            if totals[task] >= tokens_per_task:
                continue
            seed = int(seed_base) + task_index * 100_000_000 + offsets[task]
            offsets[task] += 1
            trace = greedy_full_output_trace(
                model,
                tokenizer,
                task_type=task,
                seed=seed,
                device=device,
                n_distractors=n_distractors,
                context_length=context_length,
                max_new_tokens=max_new_tokens,
            )
            remaining = tokens_per_task - totals[task]
            eligible = min(trace.eligible_tokens, remaining)
            exact_trace = replace(trace, eligible_token_count=eligible)
            exact_trace.validate(context_length=context_length)
            totals[task] += exact_trace.eligible_tokens
            yield exact_trace


def balanced_trace_stream_batched(
    model: torch.nn.Module,
    tokenizer: Any,
    *,
    tasks: Sequence[str],
    seed_base: int,
    tokens_per_task: int,
    device: torch.device,
    generation_batch_size: int,
    generation_pool_size: int,
    teacher_forced_verify_all: bool,
    canonical_regeneration_interval: int,
    n_distractors: int = 2,
    context_length: int = 128,
    max_new_tokens: int = 50,
    stop_markers: Sequence[str] = ("proof complete", "no goals"),
    start_offsets: Mapping[str, int] | None = None,
    initial_totals: Mapping[str, int] | None = None,
) -> Iterator[CircuitTrace]:
    """Yield exact per-task token quotas from verified same-length batches."""

    if min(tokens_per_task, generation_batch_size, generation_pool_size) <= 0:
        raise ValueError("batched Trace sizes and token quota must be positive")
    if generation_pool_size < generation_batch_size:
        raise ValueError("generation pool must be at least one generation batch")
    if canonical_regeneration_interval <= 0:
        raise ValueError("canonical regeneration interval must be positive")
    unknown = set(tasks) - set(ALL_TASKS)
    if unknown:
        raise ValueError(f"unknown tasks: {sorted(unknown)}")
    totals = {task: int((initial_totals or {}).get(task, 0)) for task in tasks}
    offsets = {task: int((start_offsets or {}).get(task, 0)) for task in tasks}
    if any(not 0 <= totals[task] <= tokens_per_task for task in tasks):
        raise ValueError("initial token totals are outside the declared quota")

    while any(total < tokens_per_task for total in totals.values()):
        for task_index, task in enumerate(tasks):
            if totals[task] >= tokens_per_task:
                continue
            start = offsets[task]
            requests = tuple(
                (
                    task,
                    int(seed_base)
                    + task_index * 100_000_000
                    + start
                    + local_index,
                )
                for local_index in range(int(generation_pool_size))
            )
            regeneration_mask = tuple(
                (start + local_index) % canonical_regeneration_interval == 0
                for local_index in range(len(requests))
            )
            traces = greedy_full_output_traces_batched(
                model,
                tokenizer,
                requests=requests,
                device=device,
                batch_size=generation_batch_size,
                n_distractors=n_distractors,
                context_length=context_length,
                max_new_tokens=max_new_tokens,
                stop_markers=stop_markers,
                teacher_forced_verify_all=teacher_forced_verify_all,
                canonical_regeneration_mask=regeneration_mask,
            )
            for trace in traces:
                remaining = tokens_per_task - totals[task]
                eligible = min(trace.eligible_tokens, remaining)
                exact_trace = replace(trace, eligible_token_count=eligible)
                exact_trace.validate(context_length=context_length)
                totals[task] += exact_trace.eligible_tokens
                offsets[task] += 1
                yield exact_trace
                if totals[task] >= tokens_per_task:
                    break


def _metadata_path(path: Path) -> Path:
    return path.with_suffix(path.suffix + ".json")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_trace_chunk(
    path: str | Path,
    traces: Sequence[CircuitTrace],
    *,
    config_sha256: str,
) -> dict[str, Any]:
    destination = Path(path).resolve()
    if not traces:
        raise ValueError("cannot write an empty trace chunk")
    for trace in traces:
        trace.validate()
    offsets = [0]
    token_ids: list[int] = []
    prompt_lengths: list[int] = []
    target_lengths: list[int] = []
    eligible_lengths: list[int] = []
    task_ids: list[int] = []
    seeds: list[int] = []
    records: list[dict[str, Any]] = []
    for trace in traces:
        token_ids.extend(trace.full_token_ids)
        offsets.append(len(token_ids))
        prompt_lengths.append(trace.condition_length)
        target_lengths.append(trace.target_count)
        eligible_lengths.append(trace.eligible_tokens)
        task_ids.append(TASK_TO_ID[trace.task_type])
        seeds.append(trace.seed)
        records.append({"event_key": trace.event_key})
    tensors = {
        "offsets": torch.tensor(offsets, dtype=torch.int64),
        "token_ids": torch.tensor(token_ids, dtype=torch.int16),
        "prompt_lengths": torch.tensor(prompt_lengths, dtype=torch.int16),
        "target_lengths": torch.tensor(target_lengths, dtype=torch.int16),
        "eligible_lengths": torch.tensor(eligible_lengths, dtype=torch.int16),
        "task_ids": torch.tensor(task_ids, dtype=torch.uint8),
        "seeds": torch.tensor(seeds, dtype=torch.int64),
    }
    metadata = {
        "format": TRACE_FORMAT,
        "config_sha256": config_sha256,
        "records": records,
    }
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name("." + destination.name + ".tmp")
    sidecar = _metadata_path(destination)
    sidecar_tmp = _metadata_path(temporary)
    save_file(tensors, str(temporary))
    metadata["tensor_sha256"] = _sha256_file(temporary)
    sidecar_tmp.write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    if len(read_trace_chunk(temporary)) != len(traces):
        raise RuntimeError("trace chunk full reopen validation failed")
    sidecar_sha256 = _sha256_file(sidecar_tmp)
    os.replace(sidecar_tmp, sidecar)
    os.replace(temporary, destination)
    if _sha256_file(destination) != metadata["tensor_sha256"]:
        raise RuntimeError("trace tensor hash changed during publication")
    if _sha256_file(sidecar) != sidecar_sha256:
        raise RuntimeError("trace sidecar hash changed during publication")
    read_trace_chunk(destination)
    return {
        "path": str(destination),
        "sidecar": str(sidecar),
        "records": len(traces),
        "eligible_tokens": sum(eligible_lengths),
        "bytes": destination.stat().st_size + sidecar.stat().st_size,
        "tensor_sha256": metadata["tensor_sha256"],
        "sidecar_sha256": sidecar_sha256,
    }


def read_trace_chunk(path: str | Path) -> tuple[CircuitTrace, ...]:
    source = Path(path).resolve()
    tensors = load_file(str(source), device="cpu")
    metadata = json.loads(_metadata_path(source).read_text(encoding="utf-8"))
    if metadata.get("format") != TRACE_FORMAT:
        raise ValueError("unsupported trace chunk format")
    if metadata.get("tensor_sha256") != _sha256_file(source):
        raise ValueError("trace chunk tensor checksum mismatch")
    offsets = tensors["offsets"].tolist()
    result = []
    for index, record in enumerate(metadata["records"]):
        start, end = int(offsets[index]), int(offsets[index + 1])
        values = tuple(int(item) for item in tensors["token_ids"][start:end].tolist())
        prompt_length = int(tensors["prompt_lengths"][index])
        target_length = int(tensors["target_lengths"][index])
        eligible_length = int(tensors["eligible_lengths"][index])
        if prompt_length + target_length != len(values):
            raise ValueError("trace chunk token boundaries changed")
        trace = CircuitTrace(
            event_key=str(record["event_key"]),
            task_type=ID_TO_TASK[int(tensors["task_ids"][index])],
            seed=int(tensors["seeds"][index]),
            prompt_token_ids=values[:prompt_length],
            generated_token_ids=values[prompt_length:],
            eligible_token_count=(
                None if eligible_length == len(values) else eligible_length
            ),
        )
        trace.validate()
        result.append(trace)
    return tuple(result)


def iter_trace_chunks(paths: Iterable[str | Path]) -> Iterator[CircuitTrace]:
    for path in paths:
        yield from read_trace_chunk(path)


def iter_trace_chunks_round_robin(
    paths: Iterable[str | Path], *, task_order: Sequence[str] = ALL_TASKS
) -> Iterator[CircuitTrace]:
    grouped: dict[str, list[Path]] = {task: [] for task in task_order}
    for value in paths:
        path = Path(value).resolve()
        task = path.parent.name
        if task not in grouped:
            raise ValueError(f"trace path is outside the frozen task order: {path}")
        grouped[task].append(path)
    if any(not grouped[task] for task in task_order):
        raise ValueError("round-robin trace stream is missing a task shard family")

    def task_stream(task: str) -> Iterator[CircuitTrace]:
        for path in sorted(grouped[task]):
            for trace in read_trace_chunk(path):
                if trace.task_type != task:
                    raise ValueError("trace chunk task disagrees with its task directory")
                yield trace

    iterators = {task: iter(task_stream(task)) for task in task_order}
    active = list(task_order)
    while active:
        next_active = []
        for task in active:
            try:
                yield next(iterators[task])
                next_active.append(task)
            except StopIteration:
                continue
        active = next_active


def iter_trace_prefix(
    traces: Iterable[CircuitTrace], *, max_tokens: int
) -> Iterator[CircuitTrace]:
    """Yield an exact eligible-token prefix without changing record order."""

    if int(max_tokens) <= 0:
        raise ValueError("Trace prefix token quota must be positive")
    total = 0
    for trace in traces:
        remaining = int(max_tokens) - total
        if remaining <= 0:
            break
        value = trace
        if trace.eligible_tokens > remaining:
            value = replace(trace, eligible_token_count=remaining)
            value.validate()
        yield value
        total += value.eligible_tokens
        if total == int(max_tokens):
            break
    if total != int(max_tokens):
        raise ValueError(
            f"Trace stream contains {total} eligible prefix tokens; expected {max_tokens}"
        )


def trace_chunk_token_count(path: str | Path) -> int:
    tensors = load_file(str(Path(path).resolve()), device="cpu")
    return int(tensors["eligible_lengths"].long().sum())


def validate_trace_inventory(
    paths: Sequence[str | Path],
    *,
    expected_split: str,
    expected_config_sha256: str,
    expected_tasks: Sequence[str],
    expected_tokens_per_task: int,
    expected_checkpoint_config_sha256: str,
    expected_architecture_config_sha256: str,
    expected_checkpoint_sha256: str,
    expected_model_dtype: str,
    expected_tokenizer_sha256: str,
    expected_generation_code_sha256: str,
    expected_seed_base: int,
    expected_generation_contract: Mapping[str, Any],
) -> dict[str, Any]:
    resolved = tuple(Path(path).resolve() for path in paths)
    if not resolved:
        raise ValueError("trace inventory is empty")
    parents = sorted({path.parent for path in resolved})
    manifests = []
    seen_chunks: set[Path] = set()
    seen_events: set[str] = set()
    task_tokens: dict[str, int] = {}
    for parent in parents:
        manifest_path = parent / "manifest.json"
        if not manifest_path.is_file():
            raise FileNotFoundError(f"trace manifest is missing: {manifest_path}")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("format") != "s5_circuit_trace_manifest_v1":
            raise ValueError("trace manifest format changed")
        if manifest.get("status") != "complete" or manifest.get("split") != expected_split:
            raise ValueError("trace manifest split/status mismatch")
        if manifest.get("config_sha256") != expected_config_sha256:
            raise ValueError("trace manifest config identity mismatch")
        if manifest.get("generation_contract") != dict(expected_generation_contract):
            raise ValueError("trace manifest generation contract mismatch")
        model_identity = manifest.get("model_identity", {})
        if model_identity.get("checkpoint_config_sha256") != expected_checkpoint_config_sha256:
            raise ValueError("trace manifest source-checkpoint identity mismatch")
        if (
            model_identity.get("current_architecture_config_sha256") != expected_architecture_config_sha256
        ):
            raise ValueError("trace manifest architecture-config identity mismatch")
        if model_identity.get("checkpoint_sha256") != expected_checkpoint_sha256:
            raise ValueError("trace manifest source-checkpoint hash mismatch")
        if model_identity.get("dtype") != expected_model_dtype:
            raise ValueError("trace manifest source-model dtype mismatch")
        if manifest.get("tokenizer_sha256") != expected_tokenizer_sha256:
            raise ValueError("trace manifest tokenizer identity mismatch")
        if manifest.get("generation_code_sha256") != expected_generation_code_sha256:
            raise ValueError("trace manifest generation-code identity mismatch")
        task = manifest.get("task_filter")
        if task not in expected_tasks or task in task_tokens:
            raise ValueError("trace manifest task coverage is duplicated or unexpected")
        records = int(manifest.get("records", -1))
        verification = manifest.get("verification", {})
        if int(verification.get("teacher_forced_verified_records", -1)) != records:
            raise ValueError("trace manifest does not verify every record")
        interval = int(expected_generation_contract["canonical_regeneration_interval"])
        expected_regenerated = 0 if records == 0 else (records - 1) // interval + 1
        if int(verification.get("canonical_regenerated_records", -1)) != expected_regenerated:
            raise ValueError("trace manifest canonical-regeneration coverage changed")
        declared_chunks = {
            Path(item["path"]).resolve(): item for item in manifest.get("chunks", [])
        }
        actual_chunks = {path for path in resolved if path.parent == parent}
        if set(declared_chunks) != actual_chunks:
            raise ValueError("trace glob does not exactly match the frozen manifest chunks")
        eligible = 0
        task_record_index = 0
        for path in sorted(actual_chunks):
            if path in seen_chunks:
                raise ValueError("trace chunk is duplicated")
            seen_chunks.add(path)
            record = declared_chunks[path]
            if _sha256_file(path) != record.get("tensor_sha256"):
                raise ValueError("trace manifest tensor checksum mismatch")
            if _sha256_file(_metadata_path(path)) != record.get("sidecar_sha256"):
                raise ValueError("trace manifest sidecar checksum mismatch")
            count = trace_chunk_token_count(path)
            if int(record["eligible_tokens"]) != count:
                raise ValueError("trace chunk eligible-token count changed")
            if int(record.get("bytes", -1)) != (
                path.stat().st_size + _metadata_path(path).stat().st_size
            ):
                raise ValueError("trace chunk byte count changed")
            eligible += count
            sidecar = json.loads(_metadata_path(path).read_text(encoding="utf-8"))
            if sidecar.get("config_sha256") != expected_config_sha256:
                raise ValueError("trace chunk config identity mismatch")
            decoded_records = read_trace_chunk(path)
            if len(decoded_records) != len(sidecar["records"]):
                raise ValueError("trace sidecar record count changed")
            if int(record.get("records", -1)) != len(decoded_records):
                raise ValueError("trace manifest chunk record count changed")
            for record, trace in zip(
                sidecar["records"], decoded_records, strict=True
            ):
                event_key = str(record["event_key"])
                if event_key in seen_events:
                    raise ValueError("trace event is duplicated across chunks")
                seen_events.add(event_key)
                expected_seed = (
                    int(expected_seed_base)
                    + ALL_TASKS.index(str(task)) * 100_000_000
                    + task_record_index
                )
                if trace.task_type != task or trace.seed != expected_seed:
                    raise ValueError("trace task/seed stream is not the frozen sequence")
                task_record_index += 1
        if int(manifest.get("records", -1)) != task_record_index:
            raise ValueError("trace manifest total record count changed")
        if int(manifest.get("eligible_tokens", -1)) != eligible:
            raise ValueError("trace manifest total eligible-token count changed")
        if eligible != int(expected_tokens_per_task):
            raise ValueError(
                f"trace task {task} has {eligible} eligible tokens, "
                f"expected {expected_tokens_per_task}"
            )
        task_tokens[str(task)] = eligible
        manifests.append(str(manifest_path))
    if set(task_tokens) != set(expected_tasks):
        raise ValueError("trace inventory does not cover the exact expected task set")
    return {
        "split": expected_split,
        "tasks": task_tokens,
        "chunks": len(seen_chunks),
        "events": len(seen_events),
        "manifests": manifests,
    }


def validate_audit_trace_inventory(
    paths: Sequence[str | Path],
    *,
    expected_split: str,
    expected_examples: int,
    expected_config_sha256: str,
    expected_checkpoint_config_sha256: str,
    expected_architecture_config_sha256: str,
    expected_checkpoint_sha256: str,
    expected_model_dtype: str,
    expected_tokenizer_sha256: str,
    expected_generation_code_sha256: str,
    expected_generation_contract: Mapping[str, Any],
) -> tuple[CircuitTrace, ...]:
    resolved = tuple(Path(path).resolve() for path in paths)
    parents = {path.parent for path in resolved}
    if len(parents) != 1:
        raise ValueError("audit trace inventory must have one manifest root")
    manifest_path = next(iter(parents)) / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if (
        manifest.get("format") != "s5_circuit_trace_manifest_v1"
        or manifest.get("status") != "complete"
        or manifest.get("split") != expected_split
        or manifest.get("config_sha256") != expected_config_sha256
        or manifest.get("task_filter") is not None
        or manifest.get("generation_contract") != dict(expected_generation_contract)
    ):
        raise ValueError("audit trace manifest identity changed")
    if (
        manifest.get("model_identity", {}).get("checkpoint_config_sha256")
        != expected_checkpoint_config_sha256
    ):
        raise ValueError("audit trace source-checkpoint identity changed")
    if (
        manifest.get("model_identity", {}).get("current_architecture_config_sha256") != expected_architecture_config_sha256
    ):
        raise ValueError("audit trace architecture-config identity changed")
    if (
        manifest.get("model_identity", {}).get("checkpoint_sha256")
        != expected_checkpoint_sha256
    ):
        raise ValueError("audit trace source-checkpoint hash changed")
    if manifest.get("model_identity", {}).get("dtype") != expected_model_dtype:
        raise ValueError("audit trace source-model dtype changed")
    if manifest.get("tokenizer_sha256") != expected_tokenizer_sha256:
        raise ValueError("audit trace tokenizer identity changed")
    if manifest.get("generation_code_sha256") != expected_generation_code_sha256:
        raise ValueError("audit trace generation-code identity changed")
    verification = manifest.get("verification", {})
    if (
        int(verification.get("teacher_forced_verified_records", -1))
        != int(expected_examples)
        or int(verification.get("canonical_regenerated_records", -1))
        != int(expected_examples)
    ):
        raise ValueError("audit trace verification coverage changed")
    declared = {
        Path(item["path"]).resolve(): item for item in manifest.get("chunks", [])
    }
    if set(declared) != set(resolved):
        raise ValueError("audit trace glob does not exactly match its manifest")
    audit_records = 0
    audit_eligible = 0
    for path in resolved:
        record = declared[path]
        sidecar_path = _metadata_path(path)
        if (
            _sha256_file(path) != record.get("tensor_sha256")
            or _sha256_file(sidecar_path) != record.get("sidecar_sha256")
        ):
            raise ValueError("audit trace manifest checksum mismatch")
        sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
        if sidecar.get("config_sha256") != expected_config_sha256:
            raise ValueError("audit trace chunk config identity mismatch")
        decoded = read_trace_chunk(path)
        eligible = sum(trace.eligible_tokens for trace in decoded)
        if (
            int(record.get("records", -1)) != len(decoded)
            or int(record.get("eligible_tokens", -1)) != eligible
            or int(record.get("bytes", -1))
            != path.stat().st_size + sidecar_path.stat().st_size
        ):
            raise ValueError("audit trace manifest chunk totals changed")
        audit_records += len(decoded)
        audit_eligible += eligible
    if (
        int(manifest.get("records", -1)) != audit_records
        or int(manifest.get("eligible_tokens", -1)) != audit_eligible
    ):
        raise ValueError("audit trace manifest totals changed")
    traces = tuple(iter_trace_chunks(resolved))
    if len(traces) != int(expected_examples):
        raise ValueError("audit trace count changed")
    per_task = expected_examples // len(ATOMIC_TASKS)
    expected = [
        (task, 777_777_000 + task_index * 100_000_000 + offset)
        for offset in range(per_task)
        for task_index, task in enumerate(ATOMIC_TASKS)
    ]
    actual = [(trace.task_type, trace.seed) for trace in traces]
    if actual != expected or len({trace.event_key for trace in traces}) != len(traces):
        raise ValueError("audit trace task/seed/event ordering changed")
    return traces


def verified_trace_corpus_identity(
    paths: Sequence[str | Path],
    traces: Sequence[CircuitTrace],
    *,
    split: str,
    seed_base: int,
) -> dict[str, Any]:
    """Bind a validated trace inventory and its ordered events into one identity."""

    resolved = tuple(Path(path).resolve() for path in paths)
    if not resolved or not traces:
        raise ValueError("verified trace corpus identity input is empty")
    manifests = sorted({path.parent / "manifest.json" for path in resolved})
    if any(not path.is_file() for path in manifests):
        raise FileNotFoundError("verified trace corpus manifest is missing")
    manifest_inventory = [
        {"name": path.name, "sha256": _sha256_file(path)} for path in manifests
    ]
    chunk_inventory = [
        {
            "name": path.name,
            "tensor_sha256": _sha256_file(path),
            "sidecar_sha256": _sha256_file(_metadata_path(path)),
        }
        for path in resolved
    ]
    event_keys = [trace.event_key for trace in traces]
    if len(set(event_keys)) != len(event_keys):
        raise ValueError("verified trace corpus contains duplicate events")
    return {
        "format": "s5_verified_trace_corpus_v1",
        "split": str(split),
        "seed_base": int(seed_base),
        "events": len(event_keys),
        "event_keys_sha256": _semantic_sha256(event_keys),
        "manifest_sha256": _semantic_sha256(manifest_inventory),
        "inventory_sha256": _semantic_sha256(chunk_inventory),
    }


def analysis_join_record(trace: CircuitTrace, *, gold_tactic: str, correct: bool) -> dict[str, Any]:
    """Create a record for the separately rooted, non-condition-safe join."""

    return {
        "event_key": trace.event_key,
        "task_type": trace.task_type,
        "seed": trace.seed,
        "gold_tactic": str(gold_tactic),
        "correct": bool(correct),
    }
