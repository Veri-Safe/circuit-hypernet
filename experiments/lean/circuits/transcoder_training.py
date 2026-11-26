"""Train and validate one S5 Transcoder layer on frozen generated traces."""

from __future__ import annotations

from collections import defaultdict
import json
import math
import os
from pathlib import Path
import random
import time
from typing import Any, Iterable, Sequence

import numpy as np
import torch
from torch import Tensor

from ..upstream_runtime.hook_utils import hook_recorder
from .traces import CircuitTrace
from .transcoder import RectangularReLUTranscoder, transcoder_objective


TRAINING_FORMAT = "s5_rectangular_transcoder_training_v1"
VALIDATION_FORMAT = "s5_rectangular_transcoder_validation_v1"


def _write_json_atomic(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


def _append_jsonl(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(value, sort_keys=True) + "\n")
        handle.flush()


def _schedule_multiplier(
    step: int,
    *,
    total_steps: int,
    warmup_fraction: float,
    cooldown_fraction: float,
) -> float:
    warmup_steps = max(1, math.ceil(total_steps * warmup_fraction))
    cooldown_start = max(warmup_steps, math.floor(total_steps * (1.0 - cooldown_fraction)))
    if step < warmup_steps:
        return min(1.0, float(step + 1) / warmup_steps)
    if step < cooldown_start:
        return 1.0
    if step >= total_steps:
        return 0.0
    return float(total_steps - step) / max(1, total_steps - cooldown_start)


@torch.inference_mode()
def capture_layer_activations(
    model: torch.nn.Module,
    token_ids: Sequence[int],
    *,
    layer: int,
    device: torch.device,
) -> tuple[Tensor, Tensor]:
    key_in = f"{int(layer)}.mlp.act_in"
    key_out = f"{int(layer)}.mlp.resid_delta"
    tokens = torch.tensor([tuple(int(item) for item in token_ids)], device=device)
    with hook_recorder(
        regex=rf"^{int(layer)}\.mlp\.(act_in|resid_delta)$"
    ) as captured:
        model(tokens)
    if key_in not in captured or key_out not in captured:
        raise RuntimeError(f"failed to capture S5 MLP layer {layer}")
    x = captured[key_in][0].detach()
    y = captured[key_out][0].detach()
    if x.ndim != 2 or y.ndim != 2 or x.shape[0] != y.shape[0]:
        raise ValueError("captured S5 MLP activations do not align by token")
    return x, y


def _replacement_logits(
    model: torch.nn.Module,
    transcoder: RectangularReLUTranscoder,
    token_ids: Sequence[int],
    *,
    layer: int,
    device: torch.device,
) -> Tensor:
    key_in = f"{int(layer)}.mlp.act_in"
    key_out = f"{int(layer)}.mlp.resid_delta"
    state: dict[str, Tensor] = {}

    def remember(value: Tensor) -> Tensor:
        state["x"] = value
        return value

    def replace(original: Tensor) -> Tensor:
        if "x" not in state:
            raise RuntimeError("MLP output replacement ran before input capture")
        reconstruction, _ = transcoder(state["x"].float())
        return reconstruction.to(dtype=original.dtype)

    tokens = torch.tensor([tuple(int(item) for item in token_ids)], device=device)
    with torch.inference_mode(), hook_recorder(
        regex=r"a^",
        interventions={key_in: remember, key_out: replace},
    ):
        return model(tokens)[0].detach()


def _candidate_targets(total_steps: int, percent: int) -> dict[int, int]:
    return {
        value: max(1, round(total_steps * value / 100))
        for value in range(percent, 101, percent)
    }


def _save_training_checkpoint(
    path: Path,
    *,
    transcoder: RectangularReLUTranscoder,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    manifest: dict[str, Any],
    update: int,
    processed_tokens: int,
) -> None:
    value = {
        "format": TRAINING_FORMAT,
        "manifest": manifest,
        "update": int(update),
        "processed_tokens": int(processed_tokens),
        "transcoder_state_dict": transcoder.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "torch_rng_state": torch.get_rng_state(),
        "python_rng_state": random.getstate(),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    torch.save(value, temporary)
    os.replace(temporary, path)


def train_transcoder_layer(
    model: torch.nn.Module,
    traces: Iterable[CircuitTrace],
    *,
    layer: int,
    d_feature: int,
    seed: int,
    learning_rate: float,
    output_dir: str | Path,
    device: torch.device,
    config_sha256: str,
    effective_tokens: int = 8192,
    microbatch_tokens: int = 1024,
    beta1: float = 0.9,
    beta2: float = 0.999,
    eps: float = 1e-8,
    weight_decay: float = 0.0,
    max_grad_norm: float = 1.0,
    warmup_fraction: float = 0.02,
    cooldown_fraction: float = 0.20,
    checkpoint_percent: int = 10,
    sparsity_coeff: float = 8.0,
    c_coeff: float = 4.0,
    total_tokens: int,
    resume_path: str | Path | None = None,
) -> dict[str, Any]:
    if not 0 <= layer < int(model.config.n_layer):
        raise ValueError("Transcoder layer is outside the source model")
    if min(effective_tokens, microbatch_tokens) <= 0:
        raise ValueError("training token batches must be positive")
    if effective_tokens % microbatch_tokens:
        raise ValueError("effective_tokens must be divisible by microbatch_tokens")
    if total_tokens <= 0:
        raise ValueError("trace stream contains no training token")

    random.seed(int(seed))
    torch.manual_seed(int(seed))
    d_in = int(model.config.d_model + (model.config.d_pos_emb or 0))
    d_out = int(model.config.d_model)
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
    total_steps = math.ceil(total_tokens / int(effective_tokens))
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=lambda step: _schedule_multiplier(
            step,
            total_steps=total_steps,
            warmup_fraction=warmup_fraction,
            cooldown_fraction=cooldown_fraction,
        ),
    )
    output_root = Path(output_dir).resolve()
    metrics_path = output_root / "train_metrics.jsonl"
    manifest = {
        "config_sha256": config_sha256,
        "layer": int(layer),
        "d_feature": int(d_feature),
        "seed": int(seed),
        "learning_rate": float(learning_rate),
        "total_tokens": int(total_tokens),
        "total_steps": int(total_steps),
    }
    _write_json_atomic(output_root / "manifest.json", {**manifest, "status": "running"})
    targets = _candidate_targets(total_steps, int(checkpoint_percent))
    target_to_percent = {step: percent for percent, step in targets.items()}

    pending_x: list[Tensor] = []
    pending_y: list[Tensor] = []
    pending_count = 0
    update = 0
    processed = 0
    if resume_path is not None:
        resume = torch.load(Path(resume_path), map_location="cpu", weights_only=False)
        if resume.get("format") != TRAINING_FORMAT:
            raise ValueError("unsupported Transcoder resume checkpoint")
        expected_manifest = resume.get("manifest", {})
        mismatches = {
            key: (expected_manifest.get(key), manifest.get(key))
            for key in manifest
            if expected_manifest.get(key) != manifest.get(key)
        }
        if mismatches:
            raise ValueError(f"Transcoder resume identity mismatch: {mismatches}")
        transcoder.load_state_dict(resume["transcoder_state_dict"], strict=True)
        optimizer.load_state_dict(resume["optimizer_state_dict"])
        scheduler.load_state_dict(resume["scheduler_state_dict"])
        torch.set_rng_state(resume["torch_rng_state"])
        random.setstate(resume["python_rng_state"])
        update = int(resume["update"])
        processed = int(resume["processed_tokens"])
        if not 0 <= processed <= total_tokens:
            raise ValueError("Transcoder resume token cursor is invalid")
    skip_tokens = processed

    def process_ready(*, final: bool = False) -> None:
        nonlocal pending_x, pending_y, pending_count, update, processed
        while pending_count >= effective_tokens or (final and pending_count > 0):
            take = min(effective_tokens, pending_count)
            x_all = torch.cat(pending_x, dim=0)
            y_all = torch.cat(pending_y, dim=0)
            batch_x, remainder_x = x_all[:take], x_all[take:]
            batch_y, remainder_y = y_all[:take], y_all[take:]
            pending_x = [remainder_x] if remainder_x.numel() else []
            pending_y = [remainder_y] if remainder_y.numel() else []
            pending_count -= take

            optimizer.zero_grad(set_to_none=True)
            sums = {"loss": 0.0, "reconstruction": 0.0, "sparsity": 0.0, "l0_mean": 0.0}
            for start in range(0, take, microbatch_tokens):
                end = min(start + microbatch_tokens, take)
                y_hat, features = transcoder(batch_x[start:end].float())
                objective = transcoder_objective(
                    y_hat,
                    batch_y[start:end].float(),
                    features,
                    transcoder.W_dec,
                    sparsity_coeff=sparsity_coeff,
                    c_coeff=c_coeff,
                )
                weight = objective.token_count / take
                (objective.loss * weight).backward()
                for key in sums:
                    sums[key] += float(getattr(objective, key).detach().cpu()) * weight
            grad_norm = float(
                torch.nn.utils.clip_grad_norm_(
                    transcoder.parameters(), float(max_grad_norm)
                ).detach().cpu()
            )
            optimizer.step()
            scheduler.step()
            update += 1
            processed += take
            _append_jsonl(
                metrics_path,
                {
                    "update": update,
                    "processed_tokens": processed,
                    **sums,
                    "grad_norm": grad_norm,
                    "lr": float(optimizer.param_groups[0]["lr"]),
                },
            )
            if update in target_to_percent:
                percent = target_to_percent[update]
                _save_training_checkpoint(
                    output_root / "candidates" / f"p{percent:03d}.pt",
                    transcoder=transcoder,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    manifest=manifest,
                    update=update,
                    processed_tokens=processed,
                )

    model.eval()
    for trace in traces:
        trace.validate()
        if skip_tokens >= trace.eligible_tokens:
            skip_tokens -= trace.eligible_tokens
            continue
        x, y = capture_layer_activations(
            model, trace.full_token_ids, layer=layer, device=device
        )
        x = x[: trace.eligible_tokens]
        y = y[: trace.eligible_tokens]
        if skip_tokens:
            x = x[skip_tokens:]
            y = y[skip_tokens:]
            skip_tokens = 0
        pending_x.append(x)
        pending_y.append(y)
        pending_count += int(x.shape[0])
        process_ready()
    process_ready(final=True)
    if skip_tokens:
        raise RuntimeError("Transcoder resume cursor exceeds the trace stream")
    if processed != int(total_tokens):
        raise RuntimeError(
            f"streamed {processed} Transcoder tokens, expected {total_tokens}"
        )
    _save_training_checkpoint(
        output_root / "latest.pt",
        transcoder=transcoder,
        optimizer=optimizer,
        scheduler=scheduler,
        manifest=manifest,
        update=update,
        processed_tokens=processed,
    )
    result = {**manifest, "processed_tokens": processed, "updates": update, "status": "complete"}
    _write_json_atomic(output_root / "manifest.json", result)
    return result


class _MomentAccumulator:
    def __init__(self, d_out: int) -> None:
        self.tokens = 0
        self.error_ss = 0.0
        self.target_ss = 0.0
        self.target_sum = torch.zeros(d_out, dtype=torch.float64)

    def update(self, target: Tensor, reconstruction: Tensor) -> None:
        y = target.detach().float().cpu().double()
        y_hat = reconstruction.detach().float().cpu().double()
        self.update_cpu(y, y_hat)

    def update_cpu(self, target: Tensor, reconstruction: Tensor) -> None:
        y = target.detach().double().cpu()
        y_hat = reconstruction.detach().double().cpu()
        self.tokens += int(y.shape[0])
        self.error_ss += float((y_hat - y).square().sum())
        self.target_ss += float(y.square().sum())
        self.target_sum += y.sum(dim=0)

    def summary(self) -> dict[str, float | int]:
        if self.tokens <= 0:
            return {"tokens": 0, "mse": 0.0, "fvu": 0.0, "explained_variance": 0.0}
        centered = self.target_ss - float(self.target_sum.square().sum()) / self.tokens
        fvu = self.error_ss / max(centered, 1e-12)
        return {
            "tokens": self.tokens,
            "mse": self.error_ss / (self.tokens * self.target_sum.numel()),
            "fvu": fvu,
            "explained_variance": 1.0 - fvu,
        }


def summarize_precomputed_validation(
    transcoder: RectangularReLUTranscoder,
    traces: Sequence[CircuitTrace],
    xs: Sequence[Tensor],
    ys: Sequence[Tensor],
    replacement_kls: Sequence[float],
    *,
    device: torch.device,
    metric_batch_tokens: int = 1024,
) -> dict[str, Any]:
    """Compute frozen validation metrics with token-batched Transcoder replay."""

    if metric_batch_tokens <= 0:
        raise ValueError("metric_batch_tokens must be positive")
    if not (
        len(traces) == len(xs) == len(ys) == len(replacement_kls)
    ):
        raise ValueError("precomputed validation rows do not align")
    transcoder.to(device=device, dtype=torch.float32).eval()
    moments: dict[str, _MomentAccumulator] = defaultdict(
        lambda: _MomentAccumulator(transcoder.d_out)
    )
    l0_values: dict[str, list[np.ndarray]] = defaultdict(list)
    kl_values: dict[str, list[float]] = defaultdict(list)
    active = torch.zeros(transcoder.d_feature, dtype=torch.bool)
    rows: list[dict[str, Any]] = []

    pending: list[tuple[int, CircuitTrace, Tensor, Tensor, float]] = []
    pending_tokens = 0

    def update_group(label: str, y: Tensor, y_hat: Tensor, l0: Tensor) -> None:
        if y.shape[0] <= 0:
            return
        moments[label].update_cpu(y, y_hat)
        l0_values[label].append(l0.numpy())

    def flush() -> None:
        nonlocal pending, pending_tokens, active
        if not pending:
            return
        flat_x = torch.cat([item[2] for item in pending], dim=0).contiguous()
        flat_y = torch.cat([item[3] for item in pending], dim=0).contiguous()
        gpu_x = flat_x.to(device=device, non_blocking=device.type == "cuda")
        with torch.inference_mode():
            y_hat, features = transcoder(gpu_x.float())
        y_cpu = flat_y.float().double().cpu()
        y_hat_cpu = y_hat.float().double().cpu()
        l0_cpu = features.gt(0).sum(dim=-1).float().cpu()
        active |= features.gt(0).any(dim=0).cpu()
        cursor = 0
        for row_index, trace, _x, y, kl in pending:
            full_count = int(y.shape[0])
            eligible_count = int(trace.eligible_tokens)
            row_y = y_cpu[cursor : cursor + full_count][:eligible_count]
            row_y_hat = y_hat_cpu[cursor : cursor + full_count][:eligible_count]
            row_l0 = l0_cpu[cursor : cursor + full_count][:eligible_count]
            cursor += full_count
            prompt_count = min(int(trace.condition_length), eligible_count)
            output_count = eligible_count - prompt_count
            task = trace.task_type
            update_group("overall", row_y, row_y_hat, row_l0)
            update_group(f"task:{task}", row_y, row_y_hat, row_l0)
            update_group(
                "region:prompt",
                row_y[:prompt_count],
                row_y_hat[:prompt_count],
                row_l0[:prompt_count],
            )
            update_group(
                f"task_region:{task}:prompt",
                row_y[:prompt_count],
                row_y_hat[:prompt_count],
                row_l0[:prompt_count],
            )
            if output_count:
                update_group(
                    "region:output",
                    row_y[prompt_count:],
                    row_y_hat[prompt_count:],
                    row_l0[prompt_count:],
                )
                update_group(
                    f"task_region:{task}:output",
                    row_y[prompt_count:],
                    row_y_hat[prompt_count:],
                    row_l0[prompt_count:],
                )
            kl_values["overall"].append(float(kl))
            kl_values[f"task:{task}"].append(float(kl))
            centered = row_y - row_y.mean(dim=0, keepdim=True)
            row_fvu = float(
                (row_y_hat - row_y).square().sum().div(
                    centered.square().sum().clamp_min(1e-12)
                )
            )
            rows.append(
                {
                    "event_key": trace.event_key,
                    "task_type": task,
                    "seed": int(trace.seed),
                    "eligible_tokens": int(trace.eligible_tokens),
                    "replacement_kl": float(kl),
                    "fvu": row_fvu,
                    "l0_median": float(row_l0.median()) if row_l0.numel() else 0.0,
                    "row_index": row_index,
                }
            )
        if cursor != int(flat_y.shape[0]):
            raise RuntimeError("token-batched validation cursor changed")
        pending = []
        pending_tokens = 0

    for row_index, (trace, x, y, kl) in enumerate(
        zip(traces, xs, ys, replacement_kls, strict=True)
    ):
        row_x = x.float().cpu().contiguous()
        row_y = y.float().cpu().contiguous()
        full_tokens = int(row_x.shape[0])
        if pending and pending_tokens + full_tokens > metric_batch_tokens:
            flush()
        pending.append((row_index, trace, row_x, row_y, float(kl)))
        pending_tokens += full_tokens
        if pending_tokens >= metric_batch_tokens:
            flush()
    flush()
    rows.sort(key=lambda row: int(row["row_index"]))

    summaries = {}
    for label, accumulator in moments.items():
        values = np.concatenate(l0_values[label])
        summaries[label] = {
            **accumulator.summary(),
            "l0_mean": float(values.mean()),
            "l0_median": float(np.median(values)),
            "l0_p95": float(np.percentile(values, 95)),
            "replacement_kl": (
                float(np.mean(kl_values[label])) if label in kl_values else None
            ),
        }
    overall = summaries.get("overall", {})
    return {
        "format": "s5_rectangular_transcoder_validation_token_batched_v1",
        "summaries": summaries,
        "active_features": int(active.sum()),
        "dead_features": int((~active).sum()),
        "dead_feature_fraction": float((~active).float().mean()),
        "rows": rows,
        "overall_fvu": overall.get("fvu"),
        "overall_explained_variance": overall.get("explained_variance"),
        "metric_batch_tokens": int(metric_batch_tokens),
        "status": "complete",
    }


def summarize_precomputed_validation_vectorized(
    transcoder: RectangularReLUTranscoder,
    traces: Sequence[CircuitTrace],
    xs: Sequence[Tensor],
    ys: Sequence[Tensor],
    replacement_kls: Sequence[float],
    *,
    device: torch.device,
    metric_batch_tokens: int = 1024,
) -> dict[str, Any]:
    """GPU-vectorized equivalent of ``summarize_precomputed_validation``.

    Source activations and target tensors remain the canonical batch-one CPU
    artifacts.  Only the sufficient statistics (error sum, target moments,
    row-FVU reductions, and task/region reductions) move to GPU.  L0 samples
    and row identities retain the legacy order and CPU quantile definition.
    """

    if device.type != "cuda":
        return summarize_precomputed_validation(
            transcoder,
            traces,
            xs,
            ys,
            replacement_kls,
            device=device,
            metric_batch_tokens=metric_batch_tokens,
        )
    if metric_batch_tokens <= 0:
        raise ValueError("metric_batch_tokens must be positive")
    if not (len(traces) == len(xs) == len(ys) == len(replacement_kls)):
        raise ValueError("precomputed validation rows do not align")
    transcoder.to(device=device, dtype=torch.float32).eval()
    task_names = list(dict.fromkeys(trace.task_type for trace in traces))
    labels = [
        "overall",
        *[f"task:{task}" for task in task_names],
        "region:prompt",
        "region:output",
        *[
            label
            for task in task_names
            for label in (
                f"task_region:{task}:prompt",
                f"task_region:{task}:output",
            )
        ],
    ]
    label_to_id = {label: index for index, label in enumerate(labels)}
    accumulators: dict[str, dict[str, Any]] = {}
    for label in labels:
        accumulators[label] = {
            "tokens": 0,
            "error_ss": 0.0,
            "target_ss": 0.0,
            "target_sum": torch.zeros(transcoder.d_out, dtype=torch.float64),
            "l0_values": [],
            "kl_values": [],
        }
    active = torch.zeros(transcoder.d_feature, dtype=torch.bool)
    rows: list[dict[str, Any]] = []
    pending: list[tuple[int, CircuitTrace, Tensor, Tensor, float]] = []
    pending_tokens = 0

    def update_groups(
        group_ids: Tensor,
        values_y: Tensor,
        values_error: Tensor,
        values_target_ss: Tensor,
        values_l0_cpu: Tensor,
    ) -> None:
        group_count = len(labels)
        counts = torch.bincount(group_ids, minlength=group_count)
        error_sum = torch.zeros(group_count, device=device, dtype=torch.float64)
        error_sum.index_add_(0, group_ids, values_error)
        target_ss = torch.zeros(group_count, device=device, dtype=torch.float64)
        target_ss.index_add_(0, group_ids, values_target_ss)
        target_sum = torch.zeros(
            (group_count, transcoder.d_out), device=device, dtype=torch.float64
        )
        target_sum.index_add_(0, group_ids, values_y)
        group_ids_cpu = group_ids.cpu()
        counts_cpu = counts.cpu()
        error_cpu = error_sum.cpu()
        target_ss_cpu = target_ss.cpu()
        target_sum_cpu = target_sum.cpu()
        for group_index, label in enumerate(labels):
            count = int(counts_cpu[group_index])
            if count <= 0:
                continue
            accumulator = accumulators[label]
            accumulator["tokens"] += count
            accumulator["error_ss"] += float(error_cpu[group_index])
            accumulator["target_ss"] += float(target_ss_cpu[group_index])
            accumulator["target_sum"] += target_sum_cpu[group_index]
            accumulator["l0_values"].append(
                values_l0_cpu[group_ids_cpu == group_index].numpy()
            )

    def flush() -> None:
        nonlocal pending, pending_tokens, active
        if not pending:
            return
        flat_x = torch.cat([item[2] for item in pending], dim=0).contiguous()
        flat_y = torch.cat([item[3] for item in pending], dim=0).contiguous()
        full_lengths = [int(item[3].shape[0]) for item in pending]
        eligible_lengths = [int(item[1].eligible_tokens) for item in pending]
        gpu_x = flat_x.to(device=device, non_blocking=True)
        gpu_y = flat_y.to(device=device, non_blocking=True).float()
        with torch.inference_mode():
            y_hat, features = transcoder(gpu_x.float())
        active |= features.gt(0).any(dim=0).cpu()
        row_count = len(pending)
        full_lengths_gpu = torch.tensor(full_lengths, device=device, dtype=torch.long)
        eligible_lengths_gpu = torch.tensor(
            eligible_lengths, device=device, dtype=torch.long
        )
        row_ids = torch.repeat_interleave(
            torch.arange(row_count, device=device, dtype=torch.long), full_lengths_gpu
        )
        starts = torch.cumsum(full_lengths_gpu, dim=0) - full_lengths_gpu
        positions = torch.arange(flat_y.shape[0], device=device) - starts[row_ids]
        valid = positions < eligible_lengths_gpu[row_ids]
        valid_rows = row_ids[valid]
        valid_positions = positions[valid]
        y_valid = gpu_y[valid].double()
        y_hat_valid = y_hat[valid].double()
        error_valid = (y_hat_valid - y_valid).square().sum(dim=-1)
        target_ss_valid = y_valid.square().sum(dim=-1)
        l0_valid_cpu = features[valid].gt(0).sum(dim=-1).float().cpu()
        row_error = torch.zeros(row_count, device=device, dtype=torch.float64)
        row_error.index_add_(0, valid_rows, error_valid)
        row_target_ss = torch.zeros(row_count, device=device, dtype=torch.float64)
        row_target_ss.index_add_(0, valid_rows, target_ss_valid)
        row_target_sum = torch.zeros(
            (row_count, transcoder.d_out), device=device, dtype=torch.float64
        )
        row_target_sum.index_add_(0, valid_rows, y_valid)
        row_counts = torch.bincount(valid_rows, minlength=row_count).double()
        centered = row_target_ss - row_target_sum.square().sum(dim=-1).div(
            row_counts.clamp_min(1.0)
        )
        row_fvu = row_error.div(centered.clamp_min(1e-12)).cpu()
        task_ids = torch.tensor(
            [task_names.index(item[1].task_type) for item in pending],
            device=device,
            dtype=torch.long,
        )[valid_rows]
        prompt_mask = valid_positions < torch.tensor(
            [int(item[1].condition_length) for item in pending],
            device=device,
            dtype=torch.long,
        )[valid_rows]
        update_groups(
            torch.zeros_like(valid_rows),
            y_valid,
            error_valid,
            target_ss_valid,
            l0_valid_cpu,
        )
        update_groups(1 + task_ids, y_valid, error_valid, target_ss_valid, l0_valid_cpu)
        update_groups(
            label_to_id["region:prompt"]
            + (~prompt_mask).long(),
            y_valid,
            error_valid,
            target_ss_valid,
            l0_valid_cpu,
        )
        task_region_ids = torch.tensor(
            [
                label_to_id[f"task_region:{task}:prompt"]
                for task in task_names
            ],
            device=device,
            dtype=torch.long,
        )[task_ids] + (~prompt_mask).long()
        update_groups(
            task_region_ids, y_valid, error_valid, target_ss_valid, l0_valid_cpu
        )
        valid_cursor = 0
        row_fvu_cpu = row_fvu.tolist()
        for row_index, (trace, _x, _y, kl) in enumerate(
            [(item[1], item[2], item[3], item[4]) for item in pending]
        ):
            eligible_count = int(trace.eligible_tokens)
            row_l0 = l0_valid_cpu[valid_cursor : valid_cursor + eligible_count]
            valid_cursor += eligible_count
            task = trace.task_type
            accumulators["overall"]["kl_values"].append(float(kl))
            accumulators[f"task:{task}"]["kl_values"].append(float(kl))
            rows.append(
                {
                    "event_key": trace.event_key,
                    "task_type": task,
                    "seed": int(trace.seed),
                    "eligible_tokens": eligible_count,
                    "replacement_kl": float(kl),
                    "fvu": float(row_fvu_cpu[row_index]),
                    "l0_median": float(row_l0.median()) if row_l0.numel() else 0.0,
                    "row_index": pending[row_index][0],
                }
            )
        if valid_cursor != int(l0_valid_cpu.shape[0]):
            raise RuntimeError("vectorized validation row cursor changed")
        pending = []
        pending_tokens = 0

    for row_index, (trace, x, y, kl) in enumerate(
        zip(traces, xs, ys, replacement_kls, strict=True)
    ):
        row_x = x.float().cpu().contiguous()
        row_y = y.float().cpu().contiguous()
        full_tokens = int(row_x.shape[0])
        if pending and pending_tokens + full_tokens > metric_batch_tokens:
            flush()
        pending.append((row_index, trace, row_x, row_y, float(kl)))
        pending_tokens += full_tokens
        if pending_tokens >= metric_batch_tokens:
            flush()
    flush()
    rows.sort(key=lambda row: int(row["row_index"]))
    summaries: dict[str, Any] = {}
    for label, accumulator in accumulators.items():
        if accumulator["tokens"] <= 0:
            continue
        values = np.concatenate(accumulator["l0_values"])
        tokens = int(accumulator["tokens"])
        centered = accumulator["target_ss"] - float(
            accumulator["target_sum"].square().sum()
        ) / tokens
        summaries[label] = {
            "tokens": tokens,
            "mse": accumulator["error_ss"] / (tokens * transcoder.d_out),
            "fvu": accumulator["error_ss"] / max(centered, 1e-12),
            "explained_variance": 1.0
            - accumulator["error_ss"] / max(centered, 1e-12),
            "l0_mean": float(values.mean()),
            "l0_median": float(np.median(values)),
            "l0_p95": float(np.percentile(values, 95)),
            "replacement_kl": (
                float(np.mean(accumulator["kl_values"]))
                if accumulator["kl_values"]
                else None
            ),
        }
    overall = summaries.get("overall", {})
    return {
        "format": "s5_rectangular_transcoder_validation_token_batched_v2",
        "summaries": summaries,
        "active_features": int(active.sum()),
        "dead_features": int((~active).sum()),
        "dead_feature_fraction": float((~active).float().mean()),
        "rows": rows,
        "overall_fvu": overall.get("fvu"),
        "overall_explained_variance": overall.get("explained_variance"),
        "metric_batch_tokens": int(metric_batch_tokens),
        "status": "complete",
    }


def validate_transcoder_layer_token_batched(
    model: torch.nn.Module,
    transcoder: RectangularReLUTranscoder,
    traces: Iterable[CircuitTrace],
    *,
    layer: int,
    device: torch.device,
    metric_batch_tokens: int,
) -> dict[str, Any]:
    """Exact batch-one source replay with token-batched Transcoder metrics."""

    trace_rows = tuple(traces)
    xs: list[Tensor] = []
    ys: list[Tensor] = []
    replacement_kls: list[float] = []
    capture_seconds = 0.0
    replacement_seconds = 0.0
    for trace in trace_rows:
        trace.validate()
        started = time.perf_counter()
        x, y = capture_layer_activations(
            model, trace.full_token_ids, layer=layer, device=device
        )
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        capture_seconds += time.perf_counter() - started
        xs.append(x.detach().float().cpu())
        ys.append(y.detach().float().cpu())

        started = time.perf_counter()
        with torch.inference_mode():
            tokens = torch.tensor(
                [trace.full_token_ids], device=device, dtype=torch.long
            )
            baseline_logits = model(tokens)[0][0].detach()
            replaced_logits = _replacement_logits(
                model,
                transcoder,
                trace.full_token_ids,
                layer=layer,
                device=device,
            )[0]
            positions = torch.arange(
                trace.condition_length - 1,
                trace.condition_length + trace.target_count - 1,
                device=device,
            )
            base_logp = torch.log_softmax(
                baseline_logits.index_select(0, positions).float(), dim=-1
            )
            repl_logp = torch.log_softmax(
                replaced_logits.index_select(0, positions).float(), dim=-1
            )
            kl = (base_logp.exp() * (base_logp - repl_logp)).sum(dim=-1).mean()
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        replacement_seconds += time.perf_counter() - started
        replacement_kls.append(float(kl.cpu()))

    metric_started = time.perf_counter()
    result = summarize_precomputed_validation_vectorized(
        transcoder,
        trace_rows,
        xs,
        ys,
        replacement_kls,
        device=device,
        metric_batch_tokens=metric_batch_tokens,
    )
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    result["timing"] = {
        "capture_seconds": capture_seconds,
        "replacement_seconds": replacement_seconds,
        "metric_seconds": time.perf_counter() - metric_started,
    }
    return result


def validate_transcoder_layer(
    model: torch.nn.Module,
    transcoder: RectangularReLUTranscoder,
    traces: Iterable[CircuitTrace],
    *,
    layer: int,
    device: torch.device,
) -> dict[str, Any]:
    transcoder.to(device=device, dtype=torch.float32).eval()
    moments: dict[str, _MomentAccumulator] = defaultdict(
        lambda: _MomentAccumulator(transcoder.d_out)
    )
    l0_values: dict[str, list[np.ndarray]] = defaultdict(list)
    kl_values: dict[str, list[float]] = defaultdict(list)
    active = torch.zeros(transcoder.d_feature, dtype=torch.bool)
    rows = []
    for trace in traces:
        trace.validate()
        x, y = capture_layer_activations(
            model, trace.full_token_ids, layer=layer, device=device
        )
        with torch.inference_mode():
            y_hat, features = transcoder(x.float())
            baseline_tokens = torch.tensor(
                [trace.full_token_ids], device=device, dtype=torch.long
            )
            baseline_logits = model(baseline_tokens)[0].detach()
            replaced_logits = _replacement_logits(
                model,
                transcoder,
                trace.full_token_ids,
                layer=layer,
                device=device,
            )
        token_positions = torch.arange(x.shape[0], device=device)
        eligible_mask = token_positions < trace.eligible_tokens
        prompt_mask = (token_positions < trace.condition_length) & eligible_mask
        output_mask = (token_positions >= trace.condition_length) & eligible_mask
        task = trace.task_type
        for label, mask in (
            ("overall", eligible_mask),
            (f"task:{task}", eligible_mask),
            ("region:prompt", prompt_mask),
            ("region:output", output_mask),
            (f"task_region:{task}:prompt", prompt_mask),
            (f"task_region:{task}:output", output_mask),
        ):
            if bool(mask.any()):
                moments[label].update(y[mask], y_hat[mask])
                l0_values[label].append(
                    features[mask].gt(0).sum(dim=-1).float().cpu().numpy()
                )
        active |= features.gt(0).any(dim=0).cpu()

        positions = torch.arange(
            trace.condition_length - 1,
            trace.condition_length + trace.target_count - 1,
            device=device,
        )
        base_logp = torch.log_softmax(
            baseline_logits[0].index_select(0, positions).float(), dim=-1
        )
        repl_logp = torch.log_softmax(
            replaced_logits[0].index_select(0, positions).float(), dim=-1
        )
        probabilities = base_logp.exp()
        kl = (probabilities * (base_logp - repl_logp)).sum(dim=-1)
        kl_mean = float(kl.mean().cpu())
        kl_values["overall"].append(kl_mean)
        kl_values[f"task:{task}"].append(kl_mean)
        eligible_y = y[eligible_mask].float()
        eligible_y_hat = y_hat[eligible_mask].float()
        centered = eligible_y - eligible_y.mean(dim=0, keepdim=True)
        row_fvu = float(
            (eligible_y_hat - eligible_y).square().sum().div(
                centered.square().sum().clamp_min(1e-12)
            ).cpu()
        )
        row_l0 = features[eligible_mask].gt(0).sum(dim=-1).float()
        rows.append(
            {
                "event_key": trace.event_key,
                "task_type": task,
                "replacement_kl": kl_mean,
                "fvu": row_fvu,
                "l0_median": float(row_l0.median().cpu()),
            }
        )

    summaries = {}
    for label, accumulator in moments.items():
        l0 = np.concatenate(l0_values[label])
        summaries[label] = {
            **accumulator.summary(),
            "l0_mean": float(l0.mean()),
            "l0_median": float(np.median(l0)),
            "l0_p95": float(np.percentile(l0, 95)),
            "replacement_kl": (
                float(np.mean(kl_values[label])) if label in kl_values else None
            ),
        }
    return {
        "format": VALIDATION_FORMAT,
        "layer": int(layer),
        "d_feature": transcoder.d_feature,
        "summaries": summaries,
        "active_features": int(active.sum()),
        "dead_features": int((~active).sum()),
        "dead_feature_fraction": float((~active).float().mean()),
        "rows": rows,
        "status": "complete",
    }


def load_transcoder_checkpoint(
    path: str | Path, *, device: torch.device
) -> tuple[RectangularReLUTranscoder, dict[str, Any]]:
    # These checkpoints are loaded once per layer by each worker.
    # Memory-mapping lets the OS share the serialized pages instead of
    # forcing eight independent reads/copies from the shared volume.
    value = torch.load(
        Path(path), map_location="cpu", weights_only=False, mmap=True
    )
    if value.get("format") != TRAINING_FORMAT:
        raise ValueError("unsupported Transcoder checkpoint format")
    model = RectangularReLUTranscoder.from_state_dict(
        value["transcoder_state_dict"]
    ).to(device=device, dtype=torch.float32)
    return model, value
