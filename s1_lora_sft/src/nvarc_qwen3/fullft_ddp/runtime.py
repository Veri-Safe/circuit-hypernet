"""Distributed runtime helpers shared by the HF/DDP trainer and its tests."""

from __future__ import annotations

import hashlib
import inspect
import json
import os
import random
from dataclasses import dataclass
from datetime import timedelta
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel

from nvarc_qwen3.nvarc_sft_data import TokenizedNvarcDataset, causal_loss_sum


class TrainingError(RuntimeError):
    """Raised when a fail-closed distributed training invariant is violated."""


@dataclass(frozen=True)
class DistributedContext:
    rank: int
    local_rank: int
    world_size: int
    device: torch.device

    @property
    def is_rank_zero(self) -> bool:
        return self.rank == 0


def setup_distributed(*, expected_world_size: int, expected_gpus_per_node: int) -> DistributedContext:
    missing = [name for name in ("RANK", "LOCAL_RANK", "WORLD_SIZE") if name not in os.environ]
    if missing:
        raise TrainingError(f"Missing torchrun environment variables: {missing}")
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    if world_size != expected_world_size:
        raise TrainingError(f"Expected world_size={expected_world_size}, got {world_size}")
    visible = torch.cuda.device_count()
    if visible != expected_gpus_per_node:
        raise TrainingError(
            f"Expected {expected_gpus_per_node} visible GPUs per node, got {visible}"
        )
    if not 0 <= local_rank < visible:
        raise TrainingError(f"LOCAL_RANK={local_rank} is outside {visible} visible GPUs")
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    dist.init_process_group(
        backend="nccl", timeout=timedelta(minutes=30), device_id=device
    )
    return DistributedContext(rank, local_rank, world_size, device)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class CachedTokenizedNvarcDataset(TokenizedNvarcDataset):
    """Cache the small delta-4 dataset after its first tokenization per rank."""

    @lru_cache(maxsize=None)
    def _cached(self, index: int) -> dict[str, Any]:
        return super().__getitem__(index)

    def __getitem__(self, index: int) -> dict[str, Any]:
        return self._cached(int(index))


def append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, sort_keys=True) + "\n")
        handle.flush()


def atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def distributed_totals(
    loss_sum: torch.Tensor, token_count: torch.Tensor
) -> tuple[float, int]:
    global_loss = loss_sum.detach().to(dtype=torch.float64).clone()
    global_tokens = token_count.detach().to(dtype=torch.int64).clone()
    dist.all_reduce(global_loss, op=dist.ReduceOp.SUM)
    dist.all_reduce(global_tokens, op=dist.ReduceOp.SUM)
    return float(global_loss.item()), int(global_tokens.item())


def normalize_ddp_gradients(
    parameters: Iterable[torch.nn.Parameter], *, world_size: int, global_tokens: int
) -> float:
    if world_size <= 0 or global_tokens <= 0:
        raise ValueError("world_size and global_tokens must be positive")
    scale = world_size / global_tokens
    for parameter in parameters:
        if parameter.grad is not None:
            parameter.grad.mul_(scale)
    return scale


def optimizer_parameter_groups(
    model: torch.nn.Module, *, weight_decay: float
) -> list[dict[str, Any]]:
    decay: list[torch.nn.Parameter] = []
    no_decay: list[torch.nn.Parameter] = []
    seen: set[int] = set()
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad or id(parameter) in seen:
            continue
        seen.add(id(parameter))
        if parameter.ndim < 2 or name.endswith(".bias"):
            no_decay.append(parameter)
        else:
            decay.append(parameter)
    if not decay or not no_decay:
        raise TrainingError("Expected both decayed and non-decayed parameter groups")
    return [
        {"params": decay, "weight_decay": weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]


def sampled_tensor_positions(
    numel: int, *, sample_count: int = 8
) -> tuple[int, ...]:
    """Return exact, evenly spread integer offsets including both endpoints."""

    if numel <= 0 or sample_count <= 0:
        raise ValueError("numel and sample_count must be positive")
    count = min(numel, sample_count)
    if count == 1:
        return (0,)
    final = numel - 1
    denominator = count - 1
    return tuple(index * final // denominator for index in range(count))


def model_fingerprint(
    model: torch.nn.Module, *, floating_sample_dtype: torch.dtype | None = None
) -> str:
    """Hash deterministic samples from every tensor.

    ``floating_sample_dtype`` models the checkpoint serialization dtype.  It is
    used by smoke tests to compare the in-memory FP32 model with a BF16 HF
    save/reload without treating the expected BF16 rounding as corruption.
    """

    digest = hashlib.sha256()
    with torch.no_grad():
        for name, tensor in model.state_dict().items():
            digest.update(name.encode("utf-8"))
            digest.update(str(tuple(tensor.shape)).encode("ascii"))
            flat = tensor.detach().reshape(-1)
            if flat.numel():
                positions = torch.tensor(
                    sampled_tensor_positions(flat.numel()),
                    device=flat.device,
                    dtype=torch.long,
                )
                sample = flat.index_select(0, positions)
                if floating_sample_dtype is not None and sample.is_floating_point():
                    sample = sample.to(floating_sample_dtype)
                sample = sample.float().cpu().numpy()
                digest.update(sample.tobytes())
    return digest.hexdigest()


def assert_same_fingerprint(fingerprint: str, world_size: int) -> None:
    gathered: list[str | None] = [None] * world_size
    dist.all_gather_object(gathered, fingerprint)
    if len(set(gathered)) != 1:
        raise TrainingError(f"Model parameters differ across ranks: {gathered}")


def wrap_ddp(model: torch.nn.Module, context: DistributedContext) -> DistributedDataParallel:
    kwargs: dict[str, Any] = {
        "device_ids": [context.local_rank],
        "output_device": context.local_rank,
        "broadcast_buffers": False,
        "find_unused_parameters": False,
        "gradient_as_bucket_view": True,
        "bucket_cap_mb": 200,
    }
    if "init_sync" in inspect.signature(DistributedDataParallel).parameters:
        kwargs["init_sync"] = False
    return DistributedDataParallel(model, **kwargs)


@torch.inference_mode()
def evaluate(
    model: torch.nn.Module,
    dataloader: Any,
    *,
    device: torch.device,
) -> tuple[float, int]:
    model.eval()
    local_loss = torch.zeros((), device=device, dtype=torch.float64)
    local_tokens = torch.zeros((), device=device, dtype=torch.long)
    for batch in dataloader:
        input_ids = batch["input_ids"].to(device, non_blocking=True)
        attention_mask = batch["attention_mask"].to(device, non_blocking=True)
        labels = batch["labels"].to(device, non_blocking=True)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            logits = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                use_cache=False,
            ).logits
            loss_sum, valid_tokens = causal_loss_sum(logits, labels)
        local_loss += loss_sum.double()
        local_tokens += valid_tokens
        del logits, loss_sum, valid_tokens
        del input_ids, attention_mask, labels
    global_loss, global_tokens = distributed_totals(local_loss, local_tokens)
    model.train()
    # Validation can contain much longer sequences than training. Release its
    # cached activation blocks before the next optimizer step so a single rank
    # can release cached device allocations.
    torch.cuda.empty_cache()
    if global_tokens <= 0:
        raise TrainingError("Validation produced no assistant tokens")
    return global_loss / global_tokens, global_tokens
