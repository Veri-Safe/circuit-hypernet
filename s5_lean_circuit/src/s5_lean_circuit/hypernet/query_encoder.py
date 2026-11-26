"""The protocol-frozen S5 query encoder and query-side fusion helpers."""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn


VOCAB_SIZE = 2_048
QUERY_WIDTH = 1_024
QUERY_BLOCKS = 2
QUERY_HEADS = 8
QUERY_HEAD_DIM = 128
QUERY_FFN_WIDTH = 4_096


class RMSNorm(nn.Module):
    """RMS normalization without a bias term."""

    def __init__(self, width: int, *, eps: float = 1.0e-6) -> None:
        super().__init__()
        if width <= 0:
            raise ValueError("RMSNorm width must be positive")
        self.eps = float(eps)
        self.weight = nn.Parameter(torch.ones(width))

    def forward(self, values: Tensor) -> Tensor:
        if values.shape[-1] != self.weight.numel():
            raise ValueError("RMSNorm input width does not match its parameters")
        variance = values.float().pow(2).mean(dim=-1, keepdim=True)
        normalized = values * torch.rsqrt(variance + self.eps).to(dtype=values.dtype)
        return normalized * self.weight.to(dtype=values.dtype)


class ConditionTypeEmbeddings(nn.Module):
    """Shared query/circuit type embeddings for A/B common initialization."""

    def __init__(self, *, width: int = QUERY_WIDTH, block_count: int = 5) -> None:
        super().__init__()
        if width <= 0 or block_count <= 0:
            raise ValueError("Type-embedding dimensions must be positive")
        self.query = nn.Parameter(torch.empty(width))
        self.circuit = nn.Parameter(torch.empty(width))
        self.block = nn.Parameter(torch.empty(block_count, width))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.normal_(self.query, mean=0.0, std=0.02)
        nn.init.normal_(self.circuit, mean=0.0, std=0.02)
        nn.init.normal_(self.block, mean=0.0, std=0.02)


class RotaryEmbedding(nn.Module):
    """One-dimensional, non-learned RoPE for an even attention head width."""

    def __init__(self, head_dim: int = QUERY_HEAD_DIM, *, theta: float = 10_000.0) -> None:
        super().__init__()
        if head_dim <= 0 or head_dim % 2:
            raise ValueError("RoPE head_dim must be a positive even number")
        self.head_dim = int(head_dim)
        inverse_frequency = 1.0 / (
            theta ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim)
        )
        self.register_buffer("inverse_frequency", inverse_frequency, persistent=False)

    def forward(self, values: Tensor, positions: Tensor) -> Tensor:
        if values.ndim != 4 or values.shape[-1] != self.head_dim:
            raise ValueError("RoPE values must have shape [B, heads, T, head_dim]")
        if positions.ndim != 1 or positions.numel() != values.shape[-2]:
            raise ValueError("RoPE positions must have shape [T]")
        angles = torch.outer(
            positions.to(device=values.device, dtype=torch.float32),
            self.inverse_frequency.to(device=values.device),
        )
        cos = angles.cos().to(dtype=values.dtype)[None, None, :, :]
        sin = angles.sin().to(dtype=values.dtype)[None, None, :, :]
        even = values[..., 0::2]
        odd = values[..., 1::2]
        return torch.stack((even * cos - odd * sin, even * sin + odd * cos), dim=-1).flatten(-2)


class _SelfAttention(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        width = QUERY_WIDTH
        self.qkv = nn.Linear(width, 3 * width)
        self.output = nn.Linear(width, width)
        self.rope = RotaryEmbedding()

    def forward(self, values: Tensor, valid_mask: Tensor) -> Tensor:
        batch, length, width = values.shape
        qkv = self.qkv(values).view(batch, length, 3, QUERY_HEADS, QUERY_HEAD_DIM)
        q, k, v = qkv.unbind(dim=2)
        q = self.rope(q.transpose(1, 2), torch.arange(length, device=values.device))
        k = self.rope(k.transpose(1, 2), torch.arange(length, device=values.device))
        v = v.transpose(1, 2)
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(QUERY_HEAD_DIM)
        scores = scores.masked_fill(~valid_mask[:, None, None, :], torch.finfo(scores.dtype).min)
        attention = torch.softmax(scores, dim=-1).nan_to_num(0.0)
        attended = torch.matmul(attention, v).transpose(1, 2).reshape(batch, length, width)
        return self.output(attended) * valid_mask[:, :, None].to(dtype=attended.dtype)


class _TransformerBlock(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.attention_norm = RMSNorm(QUERY_WIDTH)
        self.attention = _SelfAttention()
        self.ffn_norm = RMSNorm(QUERY_WIDTH)
        self.ffn = nn.Sequential(
            nn.Linear(QUERY_WIDTH, QUERY_FFN_WIDTH),
            nn.SiLU(),
            nn.Linear(QUERY_FFN_WIDTH, QUERY_WIDTH),
        )

    def forward(self, values: Tensor, valid_mask: Tensor) -> Tensor:
        values = values * valid_mask[:, :, None].to(dtype=values.dtype)
        values = values + self.attention(self.attention_norm(values), valid_mask)
        values = values + self.ffn(self.ffn_norm(values)) * valid_mask[:, :, None]
        return values * valid_mask[:, :, None].to(dtype=values.dtype)


class QueryEncoder(nn.Module):
    """Encode token IDs as masked ``[B, T, 1024]`` query memory tokens."""

    def __init__(
        self,
        *,
        vocab_size: int = VOCAB_SIZE,
        width: int = QUERY_WIDTH,
        num_blocks: int = QUERY_BLOCKS,
        num_heads: int = QUERY_HEADS,
        head_dim: int = QUERY_HEAD_DIM,
        ffn_width: int = QUERY_FFN_WIDTH,
        type_embeddings: ConditionTypeEmbeddings | None = None,
    ) -> None:
        super().__init__()
        frozen = (vocab_size, width, num_blocks, num_heads, head_dim, ffn_width)
        if frozen != (VOCAB_SIZE, QUERY_WIDTH, QUERY_BLOCKS, QUERY_HEADS, QUERY_HEAD_DIM, QUERY_FFN_WIDTH):
            raise ValueError("S5 Query Encoder dimensions are protocol-frozen")
        self.embedding = nn.Embedding(vocab_size, width)
        self.blocks = nn.ModuleList(_TransformerBlock() for _ in range(num_blocks))
        self.output_norm = RMSNorm(width)
        self.type_embeddings = type_embeddings or ConditionTypeEmbeddings(width=width)

    @property
    def query_type_embedding(self) -> Tensor:
        return self.type_embeddings.query

    def forward(
        self,
        token_ids: Tensor,
        attention_mask: Tensor | None = None,
        *,
        add_type_embedding: bool = True,
    ) -> Tensor:
        if token_ids.ndim != 2:
            raise ValueError("Query token IDs must have shape [B, T]")
        if token_ids.dtype not in (torch.int8, torch.int16, torch.int32, torch.int64):
            raise ValueError("Query token IDs must be an integer tensor")
        if token_ids.numel() and (token_ids.min() < 0 or token_ids.max() >= VOCAB_SIZE):
            raise ValueError("Query token ID is outside the S5 vocabulary")
        if attention_mask is None:
            valid_mask = torch.ones(token_ids.shape, device=token_ids.device, dtype=torch.bool)
        else:
            if attention_mask.shape != token_ids.shape:
                raise ValueError("Query attention mask must have shape [B, T]")
            valid_mask = attention_mask.to(device=token_ids.device, dtype=torch.bool)
        values = self.embedding(token_ids.long())
        if add_type_embedding:
            values = values + self.query_type_embedding.to(device=values.device, dtype=values.dtype)
        for block in self.blocks:
            values = block(values, valid_mask)
        values = self.output_norm(values)
        return values * valid_mask[:, :, None].to(dtype=values.dtype)


__all__ = [
    "ConditionTypeEmbeddings",
    "QUERY_BLOCKS",
    "QUERY_FFN_WIDTH",
    "QUERY_HEAD_DIM",
    "QUERY_HEADS",
    "QUERY_WIDTH",
    "QueryEncoder",
    "RMSNorm",
    "RotaryEmbedding",
    "VOCAB_SIZE",
]
