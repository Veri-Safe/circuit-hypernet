"""Vector397 Candidate A/B encoders and append-after-valid-query fusion."""

from __future__ import annotations

from typing import Sequence

import torch
from torch import Tensor, nn

from .circuit_projection import (
    VECTOR381_DIM,
    VECTOR397_DIM,
    validate_retained_indices,
)
from .query_encoder import ConditionTypeEmbeddings, QUERY_WIDTH


VECTOR397_BLOCK_DIMS = (30, 112, 162, 51, 42)
VECTOR397_BLOCK_OFFSETS = (0, 30, 142, 304, 355, 397)


def _check_vector(values: Tensor, *, input_dim: int, allow_raw: bool) -> Tensor:
    if values.ndim != 2 or values.shape[1] not in ({input_dim, VECTOR397_DIM} if allow_raw else {input_dim}):
        expected = f"[{input_dim}]" if not allow_raw else f"[{input_dim}] or [{VECTOR397_DIM}]"
        raise ValueError(f"Circuit input must have shape [B, {expected[1:-1]}]")
    if values.dtype != torch.float32:
        raise ValueError("Circuit input must be FP32")
    if not torch.isfinite(values).all():
        raise ValueError("Circuit input contains non-finite values")
    return values


def _prepare_input(
    values: Tensor,
    *,
    retained_indices: tuple[int, ...] | None,
    input_dim: int,
) -> Tensor:
    values = _check_vector(
        values,
        input_dim=input_dim,
        allow_raw=retained_indices is not None,
    )
    if retained_indices is not None and values.shape[1] == VECTOR397_DIM:
        index = torch.as_tensor(retained_indices, dtype=torch.long, device=values.device)
        values = values.index_select(1, index)
    if values.shape[1] != input_dim:
        raise ValueError("Circuit input projection dimension changed")
    return values


def _compact_block_positions(
    retained_indices: tuple[int, ...] | None,
) -> tuple[tuple[int, ...], ...]:
    """Map each frozen 397-D block to positions in the compact input."""

    if retained_indices is None:
        return tuple(
            tuple(range(start, end))
            for start, end in zip(VECTOR397_BLOCK_OFFSETS[:-1], VECTOR397_BLOCK_OFFSETS[1:])
        )
    positions = {raw: compact for compact, raw in enumerate(retained_indices)}
    blocks: list[tuple[int, ...]] = []
    for start, end in zip(VECTOR397_BLOCK_OFFSETS[:-1], VECTOR397_BLOCK_OFFSETS[1:]):
        block = tuple(positions[raw] for raw in range(start, end) if raw in positions)
        if not block:
            raise ValueError("compact Circuit projection removed an entire Vector397 block")
        blocks.append(block)
    return tuple(blocks)


def _mlp(input_dim: int, hidden_dim: int, output_dim: int) -> nn.Sequential:
    """Build a normally initialized MLP encoder.

    Encoder outputs must not be zero-initialized.  The protocol's zero
    initialization applies only to the PG B-side output projection, which is
    owned by the parameter-generator implementation rather than this module.
    """

    return nn.Sequential(
        nn.Linear(input_dim, hidden_dim),
        nn.SiLU(),
        nn.Linear(hidden_dim, output_dim),
    )


class CandidateAEncoder(nn.Module):
    """Encode the live Circuit coordinates as one ``[B, 1, 1024]`` token.

    ``retained_indices`` is optional for backward compatibility.  When set,
    the MLP is 381-dimensional and accepts either the compact input or a raw
    397-D vector that it projects using the exact recorded indices.
    """

    def __init__(
        self,
        *,
        type_embeddings: ConditionTypeEmbeddings | None = None,
        add_type_embedding: bool = True,
        retained_indices: Sequence[int] | Tensor | None = None,
    ) -> None:
        super().__init__()
        self.retained_indices = (
            None
            if retained_indices is None
            else validate_retained_indices(retained_indices)
        )
        self.input_dim = (
            VECTOR397_DIM if self.retained_indices is None else len(self.retained_indices)
        )
        self.network = _mlp(self.input_dim, 2048, QUERY_WIDTH)
        self.type_embeddings = type_embeddings or ConditionTypeEmbeddings()
        self.add_type_embedding = bool(add_type_embedding)

    @property
    def circuit_type_embedding(self) -> Tensor:
        return self.type_embeddings.circuit

    def forward(self, vector397: Tensor, *, block_permutation: Tensor | None = None) -> Tensor:
        values = _prepare_input(
            vector397,
            retained_indices=self.retained_indices,
            input_dim=self.input_dim,
        )
        token = self.network(values).unsqueeze(1)
        if self.add_type_embedding:
            token = token + self.circuit_type_embedding.to(device=token.device, dtype=token.dtype)
        return token


class CandidateBEncoder(nn.Module):
    """Encode the five frozen Vector397 blocks as ``[B, 5, 1024]`` tokens."""

    def __init__(
        self,
        *,
        type_embeddings: ConditionTypeEmbeddings | None = None,
        add_type_embedding: bool = True,
        retained_indices: Sequence[int] | Tensor | None = None,
    ) -> None:
        super().__init__()
        self.retained_indices = (
            None
            if retained_indices is None
            else validate_retained_indices(retained_indices)
        )
        self.input_dim = (
            VECTOR397_DIM if self.retained_indices is None else len(self.retained_indices)
        )
        self.block_input_positions = _compact_block_positions(self.retained_indices)
        self.block_encoders = nn.ModuleList(
            _mlp(len(positions), 512, QUERY_WIDTH)
            for positions in self.block_input_positions
        )
        self.type_embeddings = type_embeddings or ConditionTypeEmbeddings()
        self.add_type_embedding = bool(add_type_embedding)

    @property
    def circuit_type_embedding(self) -> Tensor:
        return self.type_embeddings.circuit

    @property
    def block_type_embeddings(self) -> Tensor:
        return self.type_embeddings.block

    def forward(
        self,
        vector397: Tensor,
        *,
        block_permutation: Tensor | None = None,
    ) -> Tensor:
        values = _prepare_input(
            vector397,
            retained_indices=self.retained_indices,
            input_dim=self.input_dim,
        )
        blocks = [
            values.index_select(
                1,
                torch.as_tensor(positions, dtype=torch.long, device=values.device),
            )
            for positions in self.block_input_positions
        ]
        tokens = torch.stack(
            [encoder(block) for encoder, block in zip(self.block_encoders, blocks)], dim=1
        )
        if block_permutation is not None:
            permutation = torch.as_tensor(
                block_permutation, device=tokens.device, dtype=torch.long
            )
            if permutation.ndim != 1 or permutation.numel() != len(blocks):
                raise ValueError("block_permutation must have shape [5]")
            expected = torch.arange(len(blocks), device=tokens.device)
            if not bool(torch.equal(torch.sort(permutation).values, expected)):
                raise ValueError("block_permutation must be a permutation of [0,1,2,3,4]")
            tokens = tokens.index_select(1, permutation)
        if self.add_type_embedding:
            type_values = self.block_type_embeddings.to(device=tokens.device, dtype=tokens.dtype)
            tokens = tokens + type_values.unsqueeze(0)
            tokens = tokens + self.circuit_type_embedding.to(device=tokens.device, dtype=tokens.dtype)
        return tokens


class CandidateABEncoder(nn.Module):
    """Encode one global A token followed by five addressable B block tokens."""

    def __init__(
        self,
        *,
        type_embeddings: ConditionTypeEmbeddings | None = None,
        add_type_embedding: bool = True,
        retained_indices: Sequence[int] | Tensor | None = None,
    ) -> None:
        super().__init__()
        self.retained_indices = (
            None
            if retained_indices is None
            else validate_retained_indices(retained_indices)
        )
        self.input_dim = (
            VECTOR397_DIM if self.retained_indices is None else len(self.retained_indices)
        )
        self.block_input_positions = _compact_block_positions(self.retained_indices)
        self.global_network = _mlp(self.input_dim, 2048, QUERY_WIDTH)
        self.block_encoders = nn.ModuleList(
            _mlp(len(positions), 512, QUERY_WIDTH)
            for positions in self.block_input_positions
        )
        self.type_embeddings = type_embeddings or ConditionTypeEmbeddings()
        self.add_type_embedding = bool(add_type_embedding)

    @property
    def circuit_type_embedding(self) -> Tensor:
        return self.type_embeddings.circuit

    @property
    def block_type_embeddings(self) -> Tensor:
        return self.type_embeddings.block

    def global_parameters(self) -> list[nn.Parameter]:
        """Return the global encoder and shared Circuit-type parameters."""

        return list(self.global_network.parameters()) + [self.type_embeddings.circuit]

    def block_parameters(self) -> list[nn.Parameter]:
        """Return block encoders and the address/type embedding parameters."""

        return list(self.block_encoders.parameters()) + [self.type_embeddings.block]

    def forward(
        self,
        vector397: Tensor,
        *,
        block_permutation: Tensor | None = None,
    ) -> Tensor:
        values = _prepare_input(
            vector397,
            retained_indices=self.retained_indices,
            input_dim=self.input_dim,
        )
        global_token = self.global_network(values).unsqueeze(1)
        blocks = [
            values.index_select(
                1,
                torch.as_tensor(positions, dtype=torch.long, device=values.device),
            )
            for positions in self.block_input_positions
        ]
        block_tokens = torch.stack(
            [encoder(block) for encoder, block in zip(self.block_encoders, blocks)],
            dim=1,
        )
        if block_permutation is not None:
            permutation = torch.as_tensor(
                block_permutation, device=block_tokens.device, dtype=torch.long
            )
            if permutation.ndim != 1 or permutation.numel() != len(blocks):
                raise ValueError("block_permutation must have shape [5]")
            expected = torch.arange(len(blocks), device=block_tokens.device)
            if not bool(torch.equal(torch.sort(permutation).values, expected)):
                raise ValueError("block_permutation must be a permutation of [0,1,2,3,4]")
            block_tokens = block_tokens.index_select(1, permutation)
        if self.add_type_embedding:
            global_token = global_token + self.circuit_type_embedding.to(
                device=global_token.device, dtype=global_token.dtype
            )
            block_tokens = block_tokens + self.block_type_embeddings.to(
                device=block_tokens.device, dtype=block_tokens.dtype
            ).unsqueeze(0)
            block_tokens = block_tokens + self.circuit_type_embedding.to(
                device=block_tokens.device, dtype=block_tokens.dtype
            )
        return torch.cat((global_token, block_tokens), dim=1)


# Descriptive aliases used by callers that refer to candidates as circuit encoders.
CircuitEncoderA = CandidateAEncoder
CircuitEncoderB = CandidateBEncoder
CircuitEncoderAB = CandidateABEncoder
FlatCircuitEncoder = CandidateAEncoder
FiveBlockCircuitEncoder = CandidateBEncoder
GlobalAndBlockCircuitEncoder = CandidateABEncoder


def build_circuit_present_mask(
    circuit_present: Tensor,
    *,
    token_count: int,
) -> Tensor:
    """Expand ``[B]`` presence into a bool ``[B, token_count]`` mask."""

    if token_count <= 0:
        raise ValueError("token_count must be positive")
    if circuit_present.ndim != 1:
        raise ValueError("circuit_present must have shape [B]")
    return circuit_present.to(dtype=torch.bool)[:, None].expand(-1, token_count).clone()


def build_memory_mask(
    query_mask: Tensor,
    circuit_present: Tensor,
    *,
    circuit_token_count: int,
) -> Tensor:
    """Build the compact ``[valid-query, present-circuit]`` memory mask."""

    if query_mask.ndim != 2:
        raise ValueError("query_mask must have shape [B, T]")
    if circuit_present.shape != (query_mask.shape[0],):
        raise ValueError("circuit_present must have shape [B]")
    circuit_mask = build_circuit_present_mask(
        circuit_present.to(device=query_mask.device),
        token_count=circuit_token_count,
    )
    # This helper mirrors ``append_after_valid_query``.  A raw concatenation
    # would place circuit tokens after padding, which is a different memory
    # layout for variable-length rows.  The compact result has one contiguous
    # valid prefix per row and is right-padded to the longest row.
    query_lengths = query_mask.to(dtype=torch.bool).sum(dim=1)
    circuit_lengths = circuit_mask.sum(dim=1)
    lengths = query_lengths + circuit_lengths
    max_length = int(lengths.max().item()) if lengths.numel() else 0
    positions = torch.arange(max_length, device=query_mask.device)[None, :]
    return positions < lengths[:, None]


# Short spelling retained for callers that treat presence as a first-class mask.
circuit_present_mask = build_circuit_present_mask


def append_after_valid_query(
    query_tokens: Tensor,
    query_mask: Tensor,
    circuit_tokens: Tensor,
    circuit_mask: Tensor,
) -> tuple[Tensor, Tensor]:
    """Append circuit tokens after each valid query, then right-pad the batch."""

    if query_tokens.ndim != 3:
        raise ValueError("query_tokens must have shape [B, T, H]")
    batch, _query_length, width = query_tokens.shape
    if query_mask.shape != query_tokens.shape[:2]:
        raise ValueError("query_mask must have shape [B, T]")
    if circuit_tokens.ndim != 3 or circuit_tokens.shape[0] != batch or circuit_tokens.shape[2] != width:
        raise ValueError("circuit_tokens must have shape [B, C, H]")
    if circuit_mask.shape != circuit_tokens.shape[:2]:
        raise ValueError("circuit_mask must have shape [B, C]")
    valid_query = query_mask.to(device=query_tokens.device, dtype=torch.bool)
    valid_circuit = circuit_mask.to(device=query_tokens.device, dtype=torch.bool)
    rows: list[Tensor] = []
    lengths: list[int] = []
    for row in range(batch):
        current = torch.cat(
            (query_tokens[row][valid_query[row]], circuit_tokens[row][valid_circuit[row]]),
            dim=0,
        )
        rows.append(current)
        lengths.append(current.shape[0])
    max_length = max(lengths, default=0)
    memory = query_tokens.new_zeros((batch, max_length, width))
    memory_mask = torch.zeros((batch, max_length), device=query_tokens.device, dtype=torch.bool)
    for row, current in enumerate(rows):
        if current.numel():
            memory[row, : current.shape[0]] = current
            memory_mask[row, : current.shape[0]] = True
    return memory, memory_mask


def append_masked_memory(
    query_tokens: Tensor,
    query_mask: Tensor,
    circuit_tokens: Tensor,
    circuit_mask: Tensor,
) -> tuple[Tensor, Tensor]:
    """Compatibility spelling for append-after-valid-query fusion."""

    return append_after_valid_query(query_tokens, query_mask, circuit_tokens, circuit_mask)


__all__ = [
    "CandidateAEncoder",
    "CandidateBEncoder",
    "CandidateABEncoder",
    "CircuitEncoderA",
    "CircuitEncoderB",
    "CircuitEncoderAB",
    "ConditionTypeEmbeddings",
    "FlatCircuitEncoder",
    "FiveBlockCircuitEncoder",
    "GlobalAndBlockCircuitEncoder",
    "VECTOR397_BLOCK_DIMS",
    "VECTOR397_BLOCK_OFFSETS",
    "VECTOR397_DIM",
    "VECTOR381_DIM",
    "append_after_valid_query",
    "append_masked_memory",
    "build_circuit_present_mask",
    "build_memory_mask",
    "circuit_present_mask",
]
