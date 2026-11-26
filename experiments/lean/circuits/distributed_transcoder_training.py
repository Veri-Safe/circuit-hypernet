"""Exact-order multi-GPU activation replay for S5 Transcoder training.

The frozen source model is replicated across the available ranks.  Rank zero
builds each canonical 8,192-token optimizer batch and deterministically assigns
equal contiguous slices to the participating ranks.  Every rank performs the
unchanged batch-one source-model replay for only its slice.  Activations are
gathered back in rank order, after which rank zero applies the original
Transcoder update independently to every learning-rate candidate.

This keeps source activations, token order, microbatch order, optimizer state,
and checkpoint semantics unchanged while parallelizing the dominant frozen
model replay and sharing it across learning-rate candidates.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass
import json
import math
from pathlib import Path
import random
import time
from typing import Any

import torch
import torch.distributed as dist
from torch import Tensor

from .traces import CircuitTrace
from .transcoder import RectangularReLUTranscoder, transcoder_objective
from .transcoder_training import (
    TRAINING_FORMAT,
    _append_jsonl,
    _candidate_targets,
    _save_training_checkpoint,
    _schedule_multiplier,
    _write_json_atomic,
    capture_layer_activations,
)


SHARDED_TRAINING_BACKEND = "exact_sharded_activation_replay_v1"


@dataclass(frozen=True)
class TraceActivationSlice:
    """One contiguous eligible-token slice of one frozen Trace record."""

    token_ids: tuple[int, ...]
    start: int
    end: int
    event_key: str

    @property
    def tokens(self) -> int:
        return self.end - self.start


@dataclass(frozen=True)
class ShardedTraceStep:
    """One canonical optimizer batch split into contiguous rank slices."""

    assignments: tuple[tuple[TraceActivationSlice, ...], ...]
    tokens: int


def iter_sharded_trace_steps(
    traces: Iterable[CircuitTrace],
    *,
    effective_tokens: int,
    world_size: int,
    total_tokens: int,
    start_processed_tokens: int = 0,
) -> Iterator[ShardedTraceStep]:
    """Partition the canonical eligible-token stream without changing its order."""

    if min(effective_tokens, world_size, total_tokens) <= 0:
        raise ValueError("sharded replay sizes must be positive")
    if effective_tokens % world_size:
        raise ValueError("effective tokens must divide evenly across replay ranks")
    if not 0 <= start_processed_tokens <= total_tokens:
        raise ValueError("sharded replay resume cursor is invalid")

    local_capacity = effective_tokens // world_size
    source = iter(traces)
    current: CircuitTrace | None = None
    cursor = 0

    def ensure_trace() -> CircuitTrace:
        nonlocal current, cursor
        if current is None:
            current = next(source)
            current.validate()
            cursor = 0
        return current

    remaining_skip = int(start_processed_tokens)
    while remaining_skip:
        trace = ensure_trace()
        available = trace.eligible_tokens - cursor
        take = min(available, remaining_skip)
        cursor += take
        remaining_skip -= take
        if cursor == trace.eligible_tokens:
            current = None

    processed = int(start_processed_tokens)
    while processed < total_tokens:
        step_tokens = min(effective_tokens, total_tokens - processed)
        assignments: list[list[TraceActivationSlice]] = [
            [] for _ in range(world_size)
        ]
        step_cursor = 0
        while step_cursor < step_tokens:
            trace = ensure_trace()
            available = trace.eligible_tokens - cursor
            remaining_step = step_tokens - step_cursor
            trace_take = min(available, remaining_step)
            remaining_trace_take = trace_take
            while remaining_trace_take:
                rank = min(step_cursor // local_capacity, world_size - 1)
                rank_end = min((rank + 1) * local_capacity, step_tokens)
                piece = min(remaining_trace_take, rank_end - step_cursor)
                if piece <= 0:
                    raise RuntimeError("sharded replay failed to advance a rank slice")
                assignments[rank].append(
                    TraceActivationSlice(
                        token_ids=trace.full_token_ids,
                        start=cursor,
                        end=cursor + piece,
                        event_key=trace.event_key,
                    )
                )
                cursor += piece
                step_cursor += piece
                remaining_trace_take -= piece
            if cursor == trace.eligible_tokens:
                current = None
        yield ShardedTraceStep(
            assignments=tuple(tuple(value) for value in assignments),
            tokens=step_tokens,
        )
        processed += step_tokens


def _capture_rank_slice(
    model: torch.nn.Module,
    assignments: Sequence[TraceActivationSlice],
    *,
    layer: int,
    local_capacity: int,
    d_in: int,
    d_out: int,
    device: torch.device,
) -> tuple[Tensor, Tensor, int]:
    xs: list[Tensor] = []
    ys: list[Tensor] = []
    for assignment in assignments:
        x, y = capture_layer_activations(
            model, assignment.token_ids, layer=layer, device=device
        )
        xs.append(x[assignment.start : assignment.end])
        ys.append(y[assignment.start : assignment.end])
    count = sum(value.tokens for value in assignments)
    if count > local_capacity:
        raise RuntimeError("rank activation slice exceeds its fixed capacity")
    local_x = (
        torch.cat(xs, dim=0)
        if xs
        else torch.empty((0, d_in), device=device, dtype=torch.float32)
    )
    local_y = (
        torch.cat(ys, dim=0)
        if ys
        else torch.empty((0, d_out), device=device, dtype=torch.float32)
    )
    if local_x.shape != (count, d_in) or local_y.shape != (count, d_out):
        raise RuntimeError("captured rank activations have an invalid shape")
    if count < local_capacity:
        local_x = torch.cat(
            (
                local_x,
                torch.zeros(
                    (local_capacity - count, d_in),
                    device=device,
                    dtype=local_x.dtype,
                ),
            ),
            dim=0,
        )
        local_y = torch.cat(
            (
                local_y,
                torch.zeros(
                    (local_capacity - count, d_out),
                    device=device,
                    dtype=local_y.dtype,
                ),
            ),
            dim=0,
        )
    return local_x.float().contiguous(), local_y.float().contiguous(), count


@dataclass
class _CandidateState:
    label: str
    learning_rate: float
    transcoder: RectangularReLUTranscoder
    optimizer: torch.optim.Optimizer
    scheduler: torch.optim.lr_scheduler.LRScheduler
    output_root: Path
    manifest: dict[str, Any]
    target_to_percent: dict[int, int]
    update: int = 0
    processed_tokens: int = 0


def _load_candidate_resume(
    candidate: _CandidateState, resume_path: Path | None
) -> tuple[Tensor, object] | None:
    if resume_path is None:
        return None
    resume = torch.load(resume_path, map_location="cpu", weights_only=False)
    if resume.get("format") != TRAINING_FORMAT:
        raise ValueError("unsupported sharded Transcoder resume checkpoint")
    expected_manifest = resume.get("manifest", {})
    mismatches = {
        key: (expected_manifest.get(key), candidate.manifest.get(key))
        for key in candidate.manifest
        if expected_manifest.get(key) != candidate.manifest.get(key)
    }
    if mismatches:
        raise ValueError(f"sharded Transcoder resume identity mismatch: {mismatches}")
    candidate.transcoder.load_state_dict(resume["transcoder_state_dict"], strict=True)
    candidate.optimizer.load_state_dict(resume["optimizer_state_dict"])
    candidate.scheduler.load_state_dict(resume["scheduler_state_dict"])
    candidate.update = int(resume["update"])
    candidate.processed_tokens = int(resume["processed_tokens"])
    return resume["torch_rng_state"], resume["python_rng_state"]


def _assert_candidate_checkpoint_finite(candidate: _CandidateState) -> None:
    """Fail closed before publishing a checkpoint containing nonfinite state."""

    for name, parameter in candidate.transcoder.named_parameters():
        if not bool(torch.isfinite(parameter.detach()).all()):
            raise FloatingPointError(
                f"candidate {candidate.label} parameter {name} is nonfinite"
            )
    for parameter, state in candidate.optimizer.state.items():
        for key, value in state.items():
            if isinstance(value, Tensor) and not bool(torch.isfinite(value).all()):
                raise FloatingPointError(
                    f"candidate {candidate.label} optimizer state {key} is nonfinite "
                    f"for parameter shape {tuple(parameter.shape)}"
                )


def _initialize_candidates(
    model: torch.nn.Module,
    *,
    layer: int,
    d_feature: int,
    seed: int,
    learning_rates: Sequence[tuple[str, float]],
    output_dirs: Mapping[str, Path],
    resume_paths: Mapping[str, Path | None],
    device: torch.device,
    config_sha256: str,
    total_tokens: int,
    effective_tokens: int,
    beta1: float,
    beta2: float,
    eps: float,
    weight_decay: float,
    warmup_fraction: float,
    cooldown_fraction: float,
    checkpoint_percent: int,
    world_size: int,
    manifest_metadata: Mapping[str, Any] | None,
) -> list[_CandidateState]:
    d_in = int(model.config.d_model + (model.config.d_pos_emb or 0))
    d_out = int(model.config.d_model)
    total_steps = math.ceil(total_tokens / effective_tokens)
    candidates: list[_CandidateState] = []
    resume_rng_states: list[tuple[Tensor, object]] = []
    for label, learning_rate in learning_rates:
        random.seed(int(seed))
        torch.manual_seed(int(seed))
        transcoder = RectangularReLUTranscoder(d_in, d_out, int(d_feature)).to(
            device=device, dtype=torch.float32
        )
        transcoder.reset_parameters(seed=int(seed))
        optimizer = torch.optim.Adam(
            transcoder.parameters(),
            lr=float(learning_rate),
            betas=(float(beta1), float(beta2)),
            eps=float(eps),
            weight_decay=float(weight_decay),
            fused=bool(device.type == "cuda"),
        )
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer,
            lr_lambda=lambda step: _schedule_multiplier(
                step,
                total_steps=total_steps,
                warmup_fraction=warmup_fraction,
                cooldown_fraction=cooldown_fraction,
            ),
        )
        output_root = output_dirs[label].resolve()
        manifest = {
            "config_sha256": config_sha256,
            "layer": int(layer),
            "d_feature": int(d_feature),
            "seed": int(seed),
            "learning_rate": float(learning_rate),
            "total_tokens": int(total_tokens),
            "total_steps": int(total_steps),
            "training_backend": SHARDED_TRAINING_BACKEND,
            "activation_replay_world_size": int(world_size),
        }
        if manifest_metadata:
            overlap = sorted(set(manifest).intersection(manifest_metadata))
            if overlap:
                raise ValueError(
                    f"candidate manifest metadata overrides frozen fields: {overlap}"
                )
            manifest.update(dict(manifest_metadata))
        _write_json_atomic(
            output_root / "manifest.json", {**manifest, "status": "running"}
        )
        targets = _candidate_targets(total_steps, int(checkpoint_percent))
        candidate = _CandidateState(
            label=label,
            learning_rate=float(learning_rate),
            transcoder=transcoder,
            optimizer=optimizer,
            scheduler=scheduler,
            output_root=output_root,
            manifest=manifest,
            target_to_percent={step: percent for percent, step in targets.items()},
        )
        resume_rng = _load_candidate_resume(candidate, resume_paths.get(label))
        if resume_rng is not None:
            resume_rng_states.append(resume_rng)
        candidates.append(candidate)
    cursors = {(value.update, value.processed_tokens) for value in candidates}
    if len(cursors) != 1:
        raise ValueError("all shared-replay candidates must resume at one common cursor")
    if resume_rng_states:
        reference_torch_rng, reference_python_rng = resume_rng_states[0]
        if any(
            not torch.equal(torch_rng, reference_torch_rng)
            or python_rng != reference_python_rng
            for torch_rng, python_rng in resume_rng_states[1:]
        ):
            raise ValueError("shared-replay resume checkpoints have mixed RNG state")
        torch.set_rng_state(reference_torch_rng)
        random.setstate(reference_python_rng)  # type: ignore[arg-type]
    return candidates


def _train_candidate_batch(
    candidate: _CandidateState,
    batch_x: Tensor,
    batch_y: Tensor,
    *,
    take: int,
    microbatch_tokens: int,
    max_grad_norm: float,
    sparsity_coeff: float,
    c_coeff: float,
) -> dict[str, float | int]:
    candidate.optimizer.zero_grad(set_to_none=True)
    sums = {"loss": 0.0, "reconstruction": 0.0, "sparsity": 0.0, "l0_mean": 0.0}
    for start in range(0, take, microbatch_tokens):
        end = min(start + microbatch_tokens, take)
        y_hat, features = candidate.transcoder(batch_x[start:end])
        objective = transcoder_objective(
            y_hat,
            batch_y[start:end],
            features,
            candidate.transcoder.W_dec,
            sparsity_coeff=sparsity_coeff,
            c_coeff=c_coeff,
        )
        objective_values = {
            "loss": objective.loss,
            "reconstruction": objective.reconstruction,
            "sparsity": objective.sparsity,
            "l0_mean": objective.l0_mean,
        }
        if any(
            not bool(torch.isfinite(value.detach()))
            for value in objective_values.values()
        ):
            raise FloatingPointError(
                f"candidate {candidate.label} produced a nonfinite objective"
            )
        weight = objective.token_count / take
        (objective.loss * weight).backward()
        for key in sums:
            sums[key] += float(getattr(objective, key).detach().cpu()) * weight
    grad_norm = float(
        torch.nn.utils.clip_grad_norm_(
            candidate.transcoder.parameters(),
            float(max_grad_norm),
            error_if_nonfinite=True,
        )
        .detach()
        .cpu()
    )
    candidate.optimizer.step()
    candidate.scheduler.step()
    candidate.update += 1
    candidate.processed_tokens += take
    metrics: dict[str, float | int] = {
        "update": candidate.update,
        "processed_tokens": candidate.processed_tokens,
        **sums,
        "grad_norm": grad_norm,
        "lr": float(candidate.optimizer.param_groups[0]["lr"]),
    }
    if any(
        isinstance(value, (int, float)) and not math.isfinite(float(value))
        for value in metrics.values()
    ):
        raise FloatingPointError(
            f"candidate {candidate.label} produced nonfinite training metrics"
        )
    _append_jsonl(candidate.output_root / "train_metrics.jsonl", metrics)
    return metrics


def train_transcoder_layer_group_distributed(
    model: torch.nn.Module,
    traces: Iterable[CircuitTrace] | None,
    *,
    layer: int,
    d_feature: int,
    seed: int,
    learning_rates: Sequence[tuple[str, float]],
    output_dirs: Mapping[str, Path],
    resume_paths: Mapping[str, Path | None],
    device: torch.device,
    config_sha256: str,
    effective_tokens: int,
    microbatch_tokens: int,
    beta1: float,
    beta2: float,
    eps: float,
    weight_decay: float,
    max_grad_norm: float,
    warmup_fraction: float,
    cooldown_fraction: float,
    checkpoint_percent: int,
    sparsity_coeff: float,
    c_coeff: float,
    total_tokens: int,
    progress_interval: int = 100,
    manifest_metadata: Mapping[str, Any] | None = None,
) -> dict[str, Any] | None:
    """Train one layer's LR candidates with exact-order sharded source replay."""

    if not dist.is_initialized():
        raise RuntimeError("sharded Transcoder training requires torch.distributed")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    if world_size < 1 or effective_tokens % world_size:
        raise ValueError("sharded Transcoder training requires a divisor world size")
    if microbatch_tokens != effective_tokens // world_size:
        raise ValueError(
            "exact sharded replay requires one canonical microbatch per replay rank"
        )
    d_in = int(model.config.d_model + (model.config.d_pos_emb or 0))
    d_out = int(model.config.d_model)
    local_capacity = effective_tokens // world_size

    candidates: list[_CandidateState] = []
    if rank == 0:
        if traces is None:
            raise ValueError("rank zero requires the canonical Trace iterator")
        candidates = _initialize_candidates(
            model,
            layer=layer,
            d_feature=d_feature,
            seed=seed,
            learning_rates=learning_rates,
            output_dirs=output_dirs,
            resume_paths=resume_paths,
            device=device,
            config_sha256=config_sha256,
            total_tokens=total_tokens,
            effective_tokens=effective_tokens,
            beta1=beta1,
            beta2=beta2,
            eps=eps,
            weight_decay=weight_decay,
            warmup_fraction=warmup_fraction,
            cooldown_fraction=cooldown_fraction,
            checkpoint_percent=checkpoint_percent,
            world_size=world_size,
            manifest_metadata=manifest_metadata,
        )
        start_update = candidates[0].update
        start_processed = candidates[0].processed_tokens
        step_iterator = iter_sharded_trace_steps(
            traces,
            effective_tokens=effective_tokens,
            world_size=world_size,
            total_tokens=total_tokens,
            start_processed_tokens=start_processed,
        )
    else:
        start_update = 0
        start_processed = 0
        step_iterator = None

    cursor_payload: list[Any] = [(start_update, start_processed) if rank == 0 else None]
    dist.broadcast_object_list(cursor_payload, src=0)
    start_update, start_processed = map(int, cursor_payload[0])
    total_steps = math.ceil(total_tokens / effective_tokens)
    model.eval()
    started = time.perf_counter()
    last_metrics: dict[str, Mapping[str, float | int]] = {}

    for expected_update in range(start_update + 1, total_steps + 1):
        step_payload: list[Any] = [next(step_iterator) if rank == 0 else None]
        dist.broadcast_object_list(step_payload, src=0)
        step: ShardedTraceStep = step_payload[0]
        local_x, local_y, local_count = _capture_rank_slice(
            model,
            step.assignments[rank],
            layer=layer,
            local_capacity=local_capacity,
            d_in=d_in,
            d_out=d_out,
            device=device,
        )
        expected_local_count = max(
            0, min(local_capacity, step.tokens - rank * local_capacity)
        )
        if local_count != expected_local_count:
            raise RuntimeError("rank activation token count changed from partition contract")

        gathered_x_tensor = torch.empty(
            (world_size * local_capacity, d_in),
            device=device,
            dtype=local_x.dtype,
        )
        gathered_y_tensor = torch.empty(
            (world_size * local_capacity, d_out),
            device=device,
            dtype=local_y.dtype,
        )
        dist.all_gather_into_tensor(gathered_x_tensor, local_x)
        dist.all_gather_into_tensor(gathered_y_tensor, local_y)

        if rank == 0:
            gathered_x = gathered_x_tensor.view(
                world_size, local_capacity, d_in
            )
            gathered_y = gathered_y_tensor.view(
                world_size, local_capacity, d_out
            )
            counts = [
                max(0, min(local_capacity, step.tokens - value * local_capacity))
                for value in range(world_size)
            ]
            batch_x = torch.cat(
                [gathered_x[index, :count] for index, count in enumerate(counts)],
                dim=0,
            )
            batch_y = torch.cat(
                [gathered_y[index, :count] for index, count in enumerate(counts)],
                dim=0,
            )
            if batch_x.shape != (step.tokens, d_in) or batch_y.shape != (
                step.tokens,
                d_out,
            ):
                raise RuntimeError("gathered activation batch changed canonical shape")
            for candidate in candidates:
                metrics = _train_candidate_batch(
                    candidate,
                    batch_x,
                    batch_y,
                    take=step.tokens,
                    microbatch_tokens=microbatch_tokens,
                    max_grad_norm=max_grad_norm,
                    sparsity_coeff=sparsity_coeff,
                    c_coeff=c_coeff,
                )
                if candidate.update != expected_update:
                    raise RuntimeError("candidate update cursor diverged")
                last_metrics[candidate.label] = metrics
                if candidate.update in candidate.target_to_percent:
                    percent = candidate.target_to_percent[candidate.update]
                    _assert_candidate_checkpoint_finite(candidate)
                    _save_training_checkpoint(
                        candidate.output_root / "candidates" / f"p{percent:03d}.pt",
                        transcoder=candidate.transcoder,
                        optimizer=candidate.optimizer,
                        scheduler=candidate.scheduler,
                        manifest=candidate.manifest,
                        update=candidate.update,
                        processed_tokens=candidate.processed_tokens,
                    )
            if (
                expected_update == start_update + 1
                or expected_update % int(progress_interval) == 0
                or expected_update == total_steps
            ):
                elapsed = max(time.perf_counter() - started, 1e-9)
                processed_delta = candidates[0].processed_tokens - start_processed
                tokens_per_second = processed_delta / elapsed
                remaining = total_tokens - candidates[0].processed_tokens
                print(
                    json.dumps(
                        {
                            "sharded_transcoder_progress": {
                                "backend": SHARDED_TRAINING_BACKEND,
                                "layer": int(layer),
                                "world_size": world_size,
                                "update": expected_update,
                                "total_steps": total_steps,
                                "processed_tokens": candidates[0].processed_tokens,
                                "total_tokens": total_tokens,
                                "tokens_per_second": tokens_per_second,
                                "eta_seconds": remaining / max(tokens_per_second, 1e-9),
                                "candidate_metrics": last_metrics,
                            }
                        },
                        sort_keys=True,
                    ),
                    flush=True,
                )

    result: dict[str, Any] | None = None
    if rank == 0:
        results: dict[str, Any] = {}
        for candidate in candidates:
            _assert_candidate_checkpoint_finite(candidate)
            _save_training_checkpoint(
                candidate.output_root / "latest.pt",
                transcoder=candidate.transcoder,
                optimizer=candidate.optimizer,
                scheduler=candidate.scheduler,
                manifest=candidate.manifest,
                update=candidate.update,
                processed_tokens=candidate.processed_tokens,
            )
            value = {
                **candidate.manifest,
                "processed_tokens": candidate.processed_tokens,
                "updates": candidate.update,
                "status": "complete",
            }
            _write_json_atomic(candidate.output_root / "manifest.json", value)
            results[candidate.label] = value
        result = {
            "format": "s5_sharded_transcoder_layer_group_result_v1",
            "status": "complete",
            "backend": SHARDED_TRAINING_BACKEND,
            "layer": int(layer),
            "world_size": world_size,
            "candidates": results,
            "elapsed_seconds": time.perf_counter() - started,
        }
    dist.barrier()
    return result
