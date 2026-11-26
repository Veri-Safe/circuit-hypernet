from __future__ import annotations

import torch
from torch import Tensor


def deterministic_block_permutation(count: int, *, seed: int, block_index: int) -> Tensor:
    if count <= 0 or block_index < 0:
        raise ValueError("Invalid deterministic block permutation request")
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed) + 1_000_003 * int(block_index))
    return torch.randperm(count, generator=generator, dtype=torch.int64)


class ActivationBlockBuffer:
    """Fixed-size BF16 x/y buffer with explicit exact-resume cursor support."""

    def __init__(
        self,
        *,
        capacity: int,
        d_model: int,
        device: torch.device,
        dtype: torch.dtype = torch.bfloat16,
    ) -> None:
        if capacity <= 0 or d_model <= 0:
            raise ValueError("Activation buffer dimensions must be positive")
        self.capacity = int(capacity)
        self.d_model = int(d_model)
        self.x = torch.empty((capacity, d_model), device=device, dtype=dtype)
        self.y = torch.empty((capacity, d_model), device=device, dtype=dtype)
        self.count = 0

    @property
    def remaining(self) -> int:
        return self.capacity - self.count

    @property
    def full(self) -> bool:
        return self.count == self.capacity

    def append(self, x: Tensor, y: Tensor, *, start: int = 0) -> int:
        if x.shape != y.shape or x.ndim != 2 or x.shape[1] != self.d_model:
            raise ValueError("Activation chunk shape does not match the block buffer")
        if not 0 <= start <= x.shape[0]:
            raise ValueError("Activation chunk start is out of range")
        consumed = min(self.remaining, x.shape[0] - start)
        if consumed:
            end = self.count + consumed
            self.x[self.count : end].copy_(x[start : start + consumed])
            self.y[self.count : end].copy_(y[start : start + consumed])
            self.count = end
        return consumed

    def values(self) -> tuple[Tensor, Tensor]:
        if self.count <= 0:
            raise ValueError("Activation buffer is empty")
        return self.x[: self.count], self.y[: self.count]

    def reset(self) -> None:
        self.count = 0
