from __future__ import annotations

import hashlib
import json
import time
from contextlib import contextmanager, nullcontext
from typing import Any, Callable, ContextManager, Mapping, Sequence

import torch

from s2_hypernet.nvarc_pg.events import TokenizedRow
from s2_hypernet.nvarc_pg.objective import (
    ddp_scaled_token_loss,
    distributed_token_count,
    generated_lora_factor_l2_per_query,
)

from .artifacts import CircuitVectorBank
from .batching import build_microbatch_plan
from .join import JoinRecord
from .lora_scale import (
    generated_lora_effective_delta_l2_per_query,
    scale_generated_lora_state,
)
from .model import FrozenQwenCircuitPerQueryLoRA
from .optimization import partition_parameter_generator_parameters


def _sync_context(module: torch.nn.Module, *, synchronize: bool) -> ContextManager[Any]:
    if synchronize or not hasattr(module, "no_sync"):
        return nullcontext()
    return module.no_sync()  # type: ignore[no-any-return,attr-defined]


def _gradient_l2(parameters: Sequence[torch.nn.Parameter]) -> float:
    values = [parameter.grad for parameter in parameters if parameter.grad is not None]
    if not values:
        return 0.0
    total = torch.zeros((), device=values[0].device, dtype=torch.float32)
    for value in values:
        norm = torch.linalg.vector_norm(value.detach(), dtype=torch.float32)
        total = total + norm.square()
    return float(torch.sqrt(total).detach().cpu())


@contextmanager
def _phase_timer(
    name: str,
    *,
    device: torch.device,
    cpu_seconds: dict[str, float],
    cuda_events: dict[str, list[tuple[torch.cuda.Event, torch.cuda.Event]]],
):
    if device.type == "cuda":
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        try:
            yield
        finally:
            end.record()
            cuda_events.setdefault(name, []).append((start, end))
        return
    started = time.perf_counter()
    try:
        yield
    finally:
        cpu_seconds[name] = cpu_seconds.get(name, 0.0) + time.perf_counter() - started


def _finalize_phase_seconds(
    *,
    device: torch.device,
    cpu_seconds: dict[str, float],
    cuda_events: dict[str, list[tuple[torch.cuda.Event, torch.cuda.Event]]],
) -> dict[str, float]:
    if device.type != "cuda":
        return cpu_seconds
    torch.cuda.synchronize(device)
    return {
        name: sum(start.elapsed_time(end) for start, end in pairs) / 1_000.0
        for name, pairs in cuda_events.items()
    }


class TrainingCircuitIndex:
    def __init__(
        self,
        records: Mapping[str, JoinRecord],
        vector_bank: CircuitVectorBank,
    ) -> None:
        by_coordinate: dict[tuple[int, int], JoinRecord] = {}
        for record in records.values():
            key = (record.dataset_index, record.pair_index)
            if key in by_coordinate:
                raise ValueError(f"Training circuit join repeats coordinates: {key}")
            by_coordinate[key] = record
        if len(by_coordinate) != len(records):
            raise ValueError("Training circuit join coordinate coverage changed")
        self.records = by_coordinate
        self.vector_bank = vector_bank
        ordered = sorted(record.event_key for record in records.values())
        if len(ordered) < 2:
            raise ValueError("Training circuit negatives require at least two events")
        offset = len(ordered) // 2
        self.negative_event_keys = {
            event_key: ordered[(index + offset) % len(ordered)]
            for index, event_key in enumerate(ordered)
        }
        if any(key == value for key, value in self.negative_event_keys.items()):
            raise ValueError("Training circuit negative assignment contains fixed points")
        self.negative_assignment_sha256 = hashlib.sha256(
            json.dumps(
                self.negative_event_keys,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest()

    def event_keys(
        self,
        rows: Sequence[TokenizedRow],
        event_index: int,
    ) -> tuple[str, ...]:
        values = []
        for row in rows:
            key = (int(row.dataset_index), int(event_index))
            if key not in self.records:
                raise KeyError(f"Circuit join has no row/event coordinate: {key}")
            values.append(self.records[key].event_key)
        return tuple(values)

    def vectors(
        self,
        rows: Sequence[TokenizedRow],
        event_index: int,
        *,
        device: torch.device,
    ) -> torch.Tensor:
        return self.vector_bank.vectors(self.event_keys(rows, event_index)).to(
            device=device,
            dtype=torch.float32,
            non_blocking=True,
        )

    def negative_vectors(
        self,
        rows: Sequence[TokenizedRow],
        event_index: int,
        *,
        device: torch.device,
    ) -> torch.Tensor:
        matched = self.event_keys(rows, event_index)
        negatives = tuple(self.negative_event_keys[event_key] for event_key in matched)
        return self.vector_bank.vectors(negatives).to(
            device=device,
            dtype=torch.float32,
            non_blocking=True,
        )


def _negative_selected(
    dataset_index: int,
    event_index: int,
    *,
    fraction: float,
    seed: int,
) -> bool:
    if fraction <= 0.0:
        return False
    digest = hashlib.blake2b(
        f"{seed}:{dataset_index}:{event_index}".encode("ascii"),
        digest_size=8,
    ).digest()
    value = int.from_bytes(digest, "big") / float(2**64)
    return value < fraction


def selected_event_indices(
    event_count: int,
    *,
    supervised_assistant_start_index: int,
    objective_weighting_mode: str,
) -> tuple[int, ...]:
    if event_count <= supervised_assistant_start_index:
        return ()
    available = tuple(range(supervised_assistant_start_index, event_count))
    if objective_weighting_mode == "last_two_event_balanced_v1":
        return available[-2:]
    if objective_weighting_mode in {
        "token_balanced_v1",
        "event_balanced_v1",
        "depth_weighted_event_v1",
    }:
        return available
    raise ValueError(f"Unsupported S4 objective weighting: {objective_weighting_mode}")


def event_objective_weight(
    event_count: int,
    event_index: int,
    *,
    supervised_assistant_start_index: int,
    objective_weighting_mode: str,
    depth_min_weight: float,
    depth_max_weight: float,
) -> float:
    selected = selected_event_indices(
        event_count,
        supervised_assistant_start_index=supervised_assistant_start_index,
        objective_weighting_mode=objective_weighting_mode,
    )
    if event_index not in selected:
        raise ValueError("Requested objective weight for an unselected S4 event")
    if objective_weighting_mode != "depth_weighted_event_v1":
        return 1.0
    if len(selected) == 1:
        return float(depth_max_weight)
    position = selected.index(event_index)
    fraction = float(position) / float(len(selected) - 1)
    return float(depth_min_weight) + fraction * float(
        depth_max_weight - depth_min_weight
    )


def _distributed_float_sum(value: float, device: torch.device) -> torch.Tensor:
    result = torch.tensor(value, device=device, dtype=torch.float64)
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        torch.distributed.all_reduce(result, op=torch.distributed.ReduceOp.SUM)
    return result


def main_objective_numerator(
    event_loss_sums: torch.Tensor,
    event_token_counts: torch.Tensor,
    event_weights: torch.Tensor,
    *,
    objective_weighting_mode: str,
) -> torch.Tensor:
    if event_loss_sums.ndim != 1 or event_token_counts.shape != event_loss_sums.shape:
        raise ValueError("S4 event losses and token counts must be aligned vectors")
    if event_weights.shape != event_loss_sums.shape:
        raise ValueError("S4 event weights do not align with event losses")
    if objective_weighting_mode == "token_balanced_v1":
        return event_loss_sums.sum()
    if objective_weighting_mode in {
        "event_balanced_v1",
        "depth_weighted_event_v1",
        "last_two_event_balanced_v1",
    }:
        event_means = event_loss_sums / event_token_counts.to(
            event_loss_sums.dtype
        )
        return (event_means * event_weights).sum()
    raise ValueError(f"Unsupported S4 objective weighting: {objective_weighting_mode}")


def train_optimizer_window(
    system: FrozenQwenCircuitPerQueryLoRA,
    *,
    rows: Sequence[TokenizedRow],
    circuits: TrainingCircuitIndex,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    device: torch.device,
    world_size: int,
    max_grad_norm: float,
    factor_l2: float,
    micro_batch_size: int,
    effective_delta_l2: float = 0.0,
    lora_gate_multiplier: float = 1.0,
    batching_mode: str = "fixed_rows_v1",
    max_micro_rows: int = 1,
    padded_token_budget: int = 0,
    supervised_assistant_start_index: int = 0,
    objective_weighting_mode: str = "token_balanced_v1",
    objective_depth_min_weight: float = 1.0,
    objective_depth_max_weight: float = 1.0,
    base_preservation_coefficient: float = 0.0,
    base_loss_cache: dict[tuple[int, int], float] | None = None,
    circuit_negative_mode: str = "none",
    circuit_negative_fraction: float = 0.0,
    circuit_negative_coefficient: float = 0.0,
    circuit_negative_margin: float = 0.0,
    circuit_negative_seed: int = 42,
    progress_callback: Callable[[int, int], None] | None = None,
) -> dict[str, float | int | str]:
    if not rows or micro_batch_size <= 0:
        raise ValueError("S4 optimizer window requires rows and a positive micro batch")
    if factor_l2 < 0:
        raise ValueError("S4 factor_l2 must be non-negative")
    if effective_delta_l2 < 0:
        raise ValueError("S4 effective_delta_l2 must be non-negative")
    if factor_l2 and effective_delta_l2:
        raise ValueError("S4 factor and effective-delta penalties must be isolated")
    if not 0.0 <= lora_gate_multiplier <= 1.0:
        raise ValueError("S4 LoRA gate multiplier must be in [0,1]")
    if supervised_assistant_start_index < 0:
        raise ValueError("S4 supervised assistant start index must be non-negative")
    if base_preservation_coefficient < 0:
        raise ValueError("S4 base-preservation coefficient must be non-negative")
    if circuit_negative_mode not in {
        "none",
        "deterministic_shuffled_one_sided_margin_v1",
    }:
        raise ValueError("Unsupported S4 circuit negative mode")
    negative_enabled = circuit_negative_mode != "none"
    selected_rows = tuple(
        (
            row,
            selected_event_indices(
                len(row.events),
                supervised_assistant_start_index=supervised_assistant_start_index,
                objective_weighting_mode=objective_weighting_mode,
            ),
        )
        for row in rows
    )
    if any(not selected for _, selected in selected_rows):
        raise ValueError("S4 loss protocol leaves a row with no supervised event")
    total_event_count = sum(len(row.events) for row in rows)
    event_count = sum(len(selected) for _, selected in selected_rows)
    local_token_count = sum(
        row.events[event_index].response_token_count
        for row, selected in selected_rows
        for event_index in selected
    )
    global_token_count = distributed_token_count(local_token_count, device)
    global_event_count = distributed_token_count(event_count, device)
    local_event_weight_sum = sum(
        event_objective_weight(
            len(row.events),
            event_index,
            supervised_assistant_start_index=supervised_assistant_start_index,
            objective_weighting_mode=objective_weighting_mode,
            depth_min_weight=objective_depth_min_weight,
            depth_max_weight=objective_depth_max_weight,
        )
        for row, selected in selected_rows
        for event_index in selected
    )
    global_event_weight_sum = _distributed_float_sum(
        local_event_weight_sum,
        device,
    )
    global_objective_weight_sum = (
        global_token_count.to(dtype=torch.float64)
        if objective_weighting_mode == "token_balanced_v1"
        else global_event_weight_sum
    )
    local_negative_event_count = sum(
        _negative_selected(
            int(row.dataset_index),
            event_index,
            fraction=circuit_negative_fraction,
            seed=circuit_negative_seed,
        )
        for row, selected in selected_rows
        for event_index in selected
    )
    global_negative_event_count = distributed_token_count(
        local_negative_event_count, device
    )
    if negative_enabled and int(global_negative_event_count.item()) <= 0:
        raise ValueError("S4 circuit negative schedule selected no events")
    parameter_generator = system.parameter_generator
    parameter_generator.train()
    optimizer.zero_grad(set_to_none=True)
    local_loss_sum = 0.0
    local_main_objective_sum = 0.0
    local_factor_l2_sum = 0.0
    local_effective_delta_l2_sum = 0.0
    local_base_preservation_hinge_sum = 0.0
    local_base_preservation_active_events = 0
    local_base_loss_cache_hits = 0
    local_base_loss_cache_misses = 0
    local_negative_margin_sum = 0.0
    local_negative_active_events = 0
    event_forward_batches = 0
    phase_cpu_seconds: dict[str, float] = {}
    phase_cuda_events: dict[str, list[tuple[torch.cuda.Event, torch.cuda.Event]]] = {}
    batch_plan = build_microbatch_plan(
        rows,
        mode=batching_mode,
        fixed_micro_rows=micro_batch_size,
        max_micro_rows=max_micro_rows,
        padded_token_budget=padded_token_budget,
        supervised_assistant_start_index=supervised_assistant_start_index,
    )
    microbatches = batch_plan.microbatches
    completed_rows = 0

    for microbatch_index, microbatch in enumerate(microbatches):
        with _phase_timer(
            "condition_encode",
            device=device,
            cpu_seconds=phase_cpu_seconds,
            cuda_events=phase_cuda_events,
        ):
            encoded_rows = system.encode_rows(microbatch, device=device)
        event_rounds = max(len(row.events) for row in microbatch)
        for event_index in range(
            supervised_assistant_start_index,
            event_rounds,
        ):
            active_pairs = tuple(
                (encoded, row)
                for encoded, row in zip(encoded_rows, microbatch, strict=True)
                if event_index
                in selected_event_indices(
                    len(row.events),
                    supervised_assistant_start_index=(
                        supervised_assistant_start_index
                    ),
                    objective_weighting_mode=objective_weighting_mode,
                )
            )
            if not active_pairs:
                continue
            active_encoded = tuple(value[0] for value in active_pairs)
            active_rows = tuple(value[1] for value in active_pairs)
            event_indices = (event_index,) * len(active_rows)
            event_weights = torch.tensor(
                [
                    event_objective_weight(
                        len(row.events),
                        event_index,
                        supervised_assistant_start_index=(
                            supervised_assistant_start_index
                        ),
                        objective_weighting_mode=objective_weighting_mode,
                        depth_min_weight=objective_depth_min_weight,
                        depth_max_weight=objective_depth_max_weight,
                    )
                    for row in active_rows
                ],
                device=device,
                dtype=torch.float32,
            )
            negative_positions = tuple(
                index
                for index, row in enumerate(active_rows)
                if negative_enabled
                and _negative_selected(
                    int(row.dataset_index),
                    event_index,
                    fraction=circuit_negative_fraction,
                    seed=circuit_negative_seed,
                )
            )
            expected = sum(
                row.events[event_index].response_token_count
                for row in active_rows
            )
            synchronize = (
                microbatch_index == len(microbatches) - 1
                and event_index == event_rounds - 1
            )
            with _phase_timer(
                "circuit_fetch",
                device=device,
                cpu_seconds=phase_cpu_seconds,
                cuda_events=phase_cuda_events,
            ):
                circuit_vectors = circuits.vectors(
                    active_rows,
                    event_index,
                    device=device,
                )
            with _sync_context(parameter_generator, synchronize=synchronize):
                defer_matched_sync = bool(
                    synchronize
                    and negative_positions
                    and hasattr(parameter_generator, "require_backward_grad_sync")
                )
                previous_backward_sync = None
                previous_forward_sync = None
                if defer_matched_sync:
                    previous_backward_sync = parameter_generator.require_backward_grad_sync
                    previous_forward_sync = parameter_generator.require_forward_param_sync
                    parameter_generator.require_backward_grad_sync = False
                if base_preservation_coefficient:
                    cache_keys = tuple(
                        (int(row.dataset_index), int(event_index))
                        for row in active_rows
                    )
                    cached_means = tuple(
                        None if base_loss_cache is None else base_loss_cache.get(key)
                        for key in cache_keys
                    )
                    if any(value is None for value in cached_means):
                        base_loss_sums, base_token_counts = (
                            system.frozen_base_loss_sums_from_encoded(
                                active_encoded,
                                event_indices=event_indices,
                            )
                        )
                        computed_means = (
                            base_loss_sums
                            / base_token_counts.to(base_loss_sums.dtype)
                        )
                        resolved_means = []
                        for index, (key, cached) in enumerate(
                            zip(cache_keys, cached_means, strict=True)
                        ):
                            if cached is None:
                                value = float(computed_means[index].detach().cpu())
                                if base_loss_cache is not None:
                                    base_loss_cache[key] = value
                                local_base_loss_cache_misses += 1
                            else:
                                value = float(cached)
                                local_base_loss_cache_hits += 1
                            resolved_means.append(value)
                    else:
                        resolved_means = [float(value) for value in cached_means]
                        local_base_loss_cache_hits += len(resolved_means)
                    base_loss_means = torch.tensor(
                        resolved_means,
                        device=device,
                        dtype=torch.float32,
                    )
                else:
                    base_loss_means = torch.zeros(
                        len(active_rows), device=device, dtype=torch.float32
                    )
                with _phase_timer(
                    "parameter_generation",
                    device=device,
                    cpu_seconds=phase_cpu_seconds,
                    cuda_events=phase_cuda_events,
                ):
                    raw_state = system.generate_lora_batch_with_circuits(
                        active_encoded,
                        event_indices=event_indices,
                        circuit_vectors=circuit_vectors,
                    )
                    state = scale_generated_lora_state(
                        raw_state,
                        lora_gate_multiplier,
                    )
                    factor_l2_sum = (
                        generated_lora_factor_l2_per_query(state).sum()
                        if factor_l2
                        else torch.zeros((), device=device, dtype=torch.float32)
                    )
                    effective_delta_l2_sum = (
                        generated_lora_effective_delta_l2_per_query(state).sum()
                        if effective_delta_l2
                        else torch.zeros((), device=device, dtype=torch.float32)
                    )
                with _phase_timer(
                    "adapted_forward_backward",
                    device=device,
                    cpu_seconds=phase_cpu_seconds,
                    cuda_events=phase_cuda_events,
                ):
                    with system.activate(state):
                        if (
                            base_preservation_coefficient
                            or negative_enabled
                            or objective_weighting_mode != "token_balanced_v1"
                        ):
                            event_loss_sums, event_token_counts = (
                                system.event_batch_loss_sums(
                                    active_rows,
                                    event_indices=event_indices,
                                )
                            )
                            loss_sum = event_loss_sums.sum()
                            token_count = event_token_counts.sum()
                        else:
                            loss_sum, token_count = system.event_batch_loss_sum(
                                active_rows,
                                event_indices=event_indices,
                            )
                        if int(token_count.item()) != expected:
                            raise ValueError(
                                f"S4 response mask changed at event {event_index}: "
                                f"{int(token_count.item())} != {expected}"
                            )
                        if base_preservation_coefficient:
                            preservation_per_event = torch.relu(
                                event_loss_sums
                                / event_token_counts.to(event_loss_sums.dtype)
                                - base_loss_means
                            )
                            preservation_hinge = (
                                preservation_per_event * event_weights
                            ).sum()
                        else:
                            preservation_hinge = torch.zeros(
                                (), device=device, dtype=torch.float32
                            )
                        if objective_weighting_mode == "token_balanced_v1":
                            main_objective_sum = loss_sum
                            main_denominator = global_token_count
                        else:
                            main_objective_sum = main_objective_numerator(
                                event_loss_sums,
                                event_token_counts,
                                event_weights,
                                objective_weighting_mode=objective_weighting_mode,
                            )
                            main_denominator = global_objective_weight_sum
                        scaled_loss = ddp_scaled_token_loss(
                            main_objective_sum,
                            main_denominator,
                            world_size=world_size,
                        )
                        if factor_l2:
                            scaled_loss = (
                                scaled_loss
                                + factor_l2
                                * ddp_scaled_token_loss(
                                    factor_l2_sum,
                                    global_event_count,
                                    world_size=world_size,
                                )
                            )
                        if effective_delta_l2:
                            scaled_loss = (
                                scaled_loss
                                + effective_delta_l2
                                * ddp_scaled_token_loss(
                                    effective_delta_l2_sum,
                                    global_event_count,
                                    world_size=world_size,
                                )
                            )
                        if base_preservation_coefficient:
                            scaled_loss = (
                                scaled_loss
                                + base_preservation_coefficient
                                * ddp_scaled_token_loss(
                                    preservation_hinge,
                                    global_event_weight_sum,
                                    world_size=world_size,
                                )
                            )
                        scaled_loss.backward()
                        activation = system.activation_summary()
                        if activation["active"] != activation["total"]:
                            raise RuntimeError(
                                f"Generated LoRA activation incomplete: {activation}"
                            )
                if defer_matched_sync:
                    parameter_generator.require_backward_grad_sync = (
                        previous_backward_sync
                    )
                    parameter_generator.require_forward_param_sync = previous_forward_sync
                if negative_positions:
                    negative_encoded = tuple(
                        active_encoded[index] for index in negative_positions
                    )
                    negative_rows = tuple(active_rows[index] for index in negative_positions)
                    negative_event_indices = (event_index,) * len(negative_rows)
                    matched_means = (
                        event_loss_sums
                        / event_token_counts.to(event_loss_sums.dtype)
                    ).detach()[list(negative_positions)]
                    with _phase_timer(
                        "negative_parameter_generation",
                        device=device,
                        cpu_seconds=phase_cpu_seconds,
                        cuda_events=phase_cuda_events,
                    ):
                        negative_vectors = circuits.negative_vectors(
                            negative_rows,
                            event_index,
                            device=device,
                        )
                        negative_raw_state = system.generate_lora_batch_with_circuits(
                            negative_encoded,
                            event_indices=negative_event_indices,
                            circuit_vectors=negative_vectors,
                        )
                        negative_state = scale_generated_lora_state(
                            negative_raw_state,
                            lora_gate_multiplier,
                        )
                    with _phase_timer(
                        "negative_forward_backward",
                        device=device,
                        cpu_seconds=phase_cpu_seconds,
                        cuda_events=phase_cuda_events,
                    ):
                        with system.activate(negative_state):
                            negative_loss_sums, negative_token_counts = (
                                system.event_batch_loss_sums(
                                    negative_rows,
                                    event_indices=negative_event_indices,
                                )
                            )
                            negative_means = negative_loss_sums / negative_token_counts.to(
                                negative_loss_sums.dtype
                            )
                            negative_margin = torch.relu(
                                circuit_negative_margin
                                + matched_means
                                - negative_means
                            )
                            negative_margin_sum = negative_margin.sum()
                            negative_objective = (
                                circuit_negative_coefficient
                                * ddp_scaled_token_loss(
                                    negative_margin_sum,
                                    global_negative_event_count,
                                    world_size=world_size,
                                )
                            )
                            negative_objective.backward()
                            activation = system.activation_summary()
                            if activation["active"] != activation["total"]:
                                raise RuntimeError(
                                    "Negative Generated LoRA activation incomplete: "
                                    f"{activation}"
                                )
                    local_negative_margin_sum += float(
                        negative_margin_sum.detach().cpu()
                    )
                    local_negative_active_events += int(
                        negative_margin.gt(0).sum().detach().cpu()
                    )
                    event_forward_batches += 1
                local_loss_sum += float(loss_sum.detach().cpu())
                local_main_objective_sum += float(
                    main_objective_sum.detach().cpu()
                )
                local_factor_l2_sum += float(factor_l2_sum.detach().cpu())
                local_effective_delta_l2_sum += float(
                    effective_delta_l2_sum.detach().cpu()
                )
                local_base_preservation_hinge_sum += float(
                    preservation_hinge.detach().cpu()
                )
                local_base_preservation_active_events += int(
                    (
                        torch.relu(
                            event_loss_sums
                            / event_token_counts.to(event_loss_sums.dtype)
                            - base_loss_means
                        )
                        > 0
                    )
                    .sum()
                    .detach()
                    .cpu()
                    if base_preservation_coefficient
                    else 0
                )
                event_forward_batches += 1
        completed_rows += len(microbatch)
        if progress_callback is not None:
            progress_callback(completed_rows, len(rows))

    raw_model = getattr(parameter_generator, "module", parameter_generator)
    partition = partition_parameter_generator_parameters(
        raw_model,
        require_circuit=not bool(getattr(raw_model, "prompt_only", False)),
    )
    fusion_parameters = tuple(raw_model.fusion.parameters())
    circuit_grad_l2 = _gradient_l2(partition.circuit)
    fusion_grad_l2 = _gradient_l2(fusion_parameters)
    trunk_grad_l2 = _gradient_l2(partition.trunk)
    with _phase_timer(
        "optimizer",
        device=device,
        cpu_seconds=phase_cpu_seconds,
        cuda_events=phase_cuda_events,
    ):
        grad_norm = torch.nn.utils.clip_grad_norm_(
            parameter_generator.parameters(),
            max_norm=max_grad_norm,
        )
        if not torch.isfinite(torch.as_tensor(grad_norm)):
            raise FloatingPointError(f"Non-finite S4 PG grad norm: {grad_norm}")
        optimizer.step()
        scheduler.step()
    optimizer.zero_grad(set_to_none=True)
    phase_seconds = _finalize_phase_seconds(
        device=device,
        cpu_seconds=phase_cpu_seconds,
        cuda_events=phase_cuda_events,
    )
    raw_grad_norm = float(torch.as_tensor(grad_norm).detach().cpu())
    clip_coefficient = min(1.0, max_grad_norm / (raw_grad_norm + 1.0e-6))
    circuit_grad_l2_postclip = circuit_grad_l2 * clip_coefficient
    fusion_grad_l2_postclip = fusion_grad_l2 * clip_coefficient
    trunk_grad_l2_postclip = trunk_grad_l2 * clip_coefficient
    return {
        "row_count": len(rows),
        "total_event_count": total_event_count,
        "skipped_event_count": total_event_count - event_count,
        "supervised_assistant_start_index": supervised_assistant_start_index,
        "event_count": event_count,
        "event_forward_batches": event_forward_batches,
        "micro_batch_size": micro_batch_size,
        "batching_mode": batch_plan.mode,
        "configured_max_micro_rows": batch_plan.max_micro_rows,
        "actual_max_micro_rows": max(len(batch) for batch in microbatches),
        "padded_token_budget": batch_plan.padded_token_budget,
        "peak_padded_event_tokens": batch_plan.peak_padded_event_tokens,
        "peak_padded_condition_tokens": batch_plan.peak_padded_condition_tokens,
        "padded_event_tokens": batch_plan.padded_event_tokens,
        "unpadded_event_tokens": batch_plan.unpadded_event_tokens,
        "event_padding_ratio": batch_plan.event_padding_ratio,
        "padded_response_tokens": batch_plan.padded_response_tokens,
        "unpadded_response_tokens": batch_plan.unpadded_response_tokens,
        "lm_head_token_reduction_ratio": (
            batch_plan.padded_event_tokens / max(batch_plan.padded_response_tokens, 1)
        ),
        "padded_condition_tokens": batch_plan.padded_condition_tokens,
        "unpadded_condition_tokens": batch_plan.unpadded_condition_tokens,
        "condition_padding_ratio": batch_plan.condition_padding_ratio,
        "gradient_accumulation_steps": len(microbatches),
        "local_token_count": local_token_count,
        "global_token_count": int(global_token_count.item()),
        "global_event_count": int(global_event_count.item()),
        "objective_weighting_mode": objective_weighting_mode,
        "objective_depth_min_weight": objective_depth_min_weight,
        "objective_depth_max_weight": objective_depth_max_weight,
        "local_event_weight_sum": local_event_weight_sum,
        "global_event_weight_sum": float(global_event_weight_sum.item()),
        "global_objective_weight_sum": float(
            global_objective_weight_sum.item()
        ),
        "local_loss_sum": local_loss_sum,
        "local_main_objective_sum": local_main_objective_sum,
        "local_factor_l2_sum": local_factor_l2_sum,
        "factor_l2_coefficient": factor_l2,
        "local_effective_delta_l2_sum": local_effective_delta_l2_sum,
        "effective_delta_l2_coefficient": effective_delta_l2,
        "local_base_preservation_hinge_sum": (
            local_base_preservation_hinge_sum
        ),
        "local_base_preservation_active_events": (
            local_base_preservation_active_events
        ),
        "base_preservation_coefficient": base_preservation_coefficient,
        "circuit_negative_mode": circuit_negative_mode,
        "circuit_negative_fraction": circuit_negative_fraction,
        "circuit_negative_coefficient": circuit_negative_coefficient,
        "circuit_negative_margin": circuit_negative_margin,
        "circuit_negative_seed": circuit_negative_seed,
        "local_negative_event_count": local_negative_event_count,
        "global_negative_event_count": int(global_negative_event_count.item()),
        "local_negative_margin_sum": local_negative_margin_sum,
        "local_negative_active_events": local_negative_active_events,
        "local_base_loss_cache_hits": local_base_loss_cache_hits,
        "local_base_loss_cache_misses": local_base_loss_cache_misses,
        "local_base_loss_cache_size": (
            0 if base_loss_cache is None else len(base_loss_cache)
        ),
        "lora_gate_multiplier": lora_gate_multiplier,
        "learned_global_lora_scale": float(
            raw_model.global_lora_scale_value().detach().cpu()
        ),
        "grad_norm": raw_grad_norm,
        "grad_clip_coefficient": clip_coefficient,
        "grad_was_clipped": int(raw_grad_norm > max_grad_norm),
        "circuit_grad_l2_preclip": circuit_grad_l2,
        "fusion_grad_l2_preclip": fusion_grad_l2,
        "trunk_grad_l2_preclip": trunk_grad_l2,
        "circuit_grad_l2_postclip": circuit_grad_l2_postclip,
        "fusion_grad_l2_postclip": fusion_grad_l2_postclip,
        "trunk_grad_l2_postclip": trunk_grad_l2_postclip,
        **{f"phase_{name}_seconds": seconds for name, seconds in phase_seconds.items()},
        "gpu_active_seconds": sum(phase_seconds.values()),
        "missing_circuits": 0,
        "corrupt_circuits": 0,
    }
