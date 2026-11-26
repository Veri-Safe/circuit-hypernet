"""Versioned projection from the raw Vector397 schema to the live 381-D input.

The raw descriptor remains the canonical 397-D artifact.  This module only
defines the condition-side projection used by the compact Circuit encoder.  A
projection is always derived from the persisted train-only normalizer mask;
callers must never infer it by dropping the first 16 coordinates.
"""

from __future__ import annotations

from collections import OrderedDict
from typing import Mapping, Sequence
import re

import numpy as np
import torch
from torch import Tensor


VECTOR397_DIM = 397
VECTOR381_DIM = 381
DEAD_COORDINATE_COUNT = VECTOR397_DIM - VECTOR381_DIM
COMPACT_PROJECTION_FORMAT = "s5_hypernet_vector397_dead_coordinate_projection_v1"
# Frozen Vector397 block schema used by Candidate-B/AB first layers.
VECTOR397_BLOCK_DIMS = (30, 112, 162, 51, 42)
VECTOR397_BLOCK_OFFSETS = (0, 30, 142, 304, 355, 397)


def validate_dead_mask(
    dead_mask: Sequence[bool] | np.ndarray | Tensor,
    *,
    expected_dead_count: int = DEAD_COORDINATE_COUNT,
) -> np.ndarray:
    """Validate and return a contiguous boolean ``[397]`` dead mask."""

    if isinstance(dead_mask, Tensor):
        value = dead_mask.detach().cpu().numpy()
    else:
        value = np.asarray(dead_mask)
    if value.shape != (VECTOR397_DIM,) or value.dtype.kind != "b":
        raise ValueError("dead_mask must be bool[397]")
    value = np.ascontiguousarray(value, dtype=np.bool_)
    if int(value.sum()) != int(expected_dead_count):
        raise ValueError(
            f"dead_mask must contain exactly {expected_dead_count} dead coordinates"
        )
    return value


def retained_indices_from_dead_mask(
    dead_mask: Sequence[bool] | np.ndarray | Tensor,
    *,
    expected_dead_count: int = DEAD_COORDINATE_COUNT,
) -> tuple[int, ...]:
    """Return the canonical raw-coordinate indices retained in the 381-D view."""

    mask = validate_dead_mask(dead_mask, expected_dead_count=expected_dead_count)
    indices = tuple(int(index) for index in np.flatnonzero(~mask))
    if len(indices) != VECTOR381_DIM:
        raise AssertionError("validated dead mask did not produce Vector381")
    return indices


def validate_retained_indices(
    indices: Sequence[int] | np.ndarray | Tensor,
    *,
    expected_dim: int = VECTOR381_DIM,
) -> tuple[int, ...]:
    """Validate a sorted, unique raw-coordinate index list."""

    if isinstance(indices, Tensor):
        value = indices.detach().cpu().numpy()
    else:
        value = np.asarray(indices)
    if value.ndim != 1 or value.dtype.kind not in "iu":
        raise ValueError("retained indices must be an integer vector")
    result = tuple(int(index) for index in value.tolist())
    if len(result) != int(expected_dim):
        raise ValueError(f"retained indices must contain exactly {expected_dim} entries")
    if result != tuple(sorted(result)) or len(set(result)) != len(result):
        raise ValueError("retained indices must be sorted and unique")
    if any(index < 0 or index >= VECTOR397_DIM for index in result):
        raise ValueError("retained indices are outside the Vector397 schema")
    return result


def compact_vector397_numpy(
    values: np.ndarray,
    retained_indices: Sequence[int] | np.ndarray | Tensor,
) -> np.ndarray:
    """Project a raw or batched Vector397 array into its compact view."""

    indices = validate_retained_indices(retained_indices)
    array = np.asarray(values)
    if array.ndim == 1:
        if array.shape != (VECTOR397_DIM,):
            raise ValueError("Vector397 input must have shape [397]")
        return np.ascontiguousarray(array[list(indices)])
    if array.ndim == 2 and array.shape[1] == VECTOR397_DIM:
        return np.ascontiguousarray(array[:, list(indices)])
    raise ValueError("Vector397 input must have shape [397] or [N,397]")


def compact_vector397_tensor(
    values: Tensor,
    retained_indices: Sequence[int] | np.ndarray | Tensor,
) -> Tensor:
    """Project a raw or batched Vector397 tensor into its compact view."""

    indices = validate_retained_indices(retained_indices)
    if values.ndim not in (1, 2) or values.shape[-1] != VECTOR397_DIM:
        raise ValueError("Vector397 tensor must have shape [397] or [N,397]")
    index = torch.as_tensor(indices, dtype=torch.long, device=values.device)
    return values.index_select(-1, index)


def compact_circuit_state_dict(
    state: Mapping[str, Tensor],
    retained_indices: Sequence[int] | np.ndarray | Tensor,
) -> OrderedDict[str, Tensor]:
    """Adapt a legacy 397-D Circuit-encoder init to the compact 381-D view.

    Candidate encoder initializations are normally generated from the same
    seed.  When a pre-existing 397-D candidate artifact is reused, its first
    MLP layer can be sliced exactly because the 16 dead input coordinates are
    zero after train-only z-score.  All other tensors must already match; a
    silent reshape would invalidate the initialization contract.
    """

    indices = validate_retained_indices(retained_indices)
    raw_index = torch.as_tensor(indices, dtype=torch.long)
    result: OrderedDict[str, Tensor] = OrderedDict()
    block_ranges = tuple(
        zip(VECTOR397_BLOCK_OFFSETS[:-1], VECTOR397_BLOCK_OFFSETS[1:])
    )
    block_local_indices = tuple(
        torch.as_tensor(
            [raw - start for raw in indices if start <= raw < end],
            dtype=torch.long,
        )
        for start, end in block_ranges
    )
    block_pattern = re.compile(r"^block_encoders\.(\d+)\.0\.weight$")
    for key, value in state.items():
        if not isinstance(value, Tensor):
            raise ValueError(f"Circuit init tensor is not a Tensor: {key}")
        adapted = value
        if key in {"network.0.weight", "global_network.0.weight"}:
            if value.ndim != 2 or value.shape[1] not in {
                VECTOR397_DIM,
                VECTOR381_DIM,
            }:
                raise ValueError(f"legacy global Circuit layer shape changed: {key}")
            if value.shape[1] == VECTOR397_DIM:
                adapted = value.index_select(1, raw_index.to(device=value.device))
        else:
            match = block_pattern.match(key)
            if match is not None:
                block = int(match.group(1))
                if block >= len(block_local_indices):
                    raise ValueError(f"legacy Circuit block index changed: {key}")
                local = block_local_indices[block].to(device=value.device)
                expected_width = VECTOR397_BLOCK_DIMS[block]
                if value.ndim != 2 or value.shape[1] not in {
                    expected_width,
                    len(local),
                }:
                    raise ValueError(f"legacy Circuit block layer shape changed: {key}")
                if value.shape[1] == expected_width:
                    adapted = value.index_select(1, local)
        result[key] = adapted
    return result


__all__ = [
    "COMPACT_PROJECTION_FORMAT",
    "DEAD_COORDINATE_COUNT",
    "VECTOR381_DIM",
    "VECTOR397_BLOCK_DIMS",
    "VECTOR397_BLOCK_OFFSETS",
    "VECTOR397_DIM",
    "compact_vector397_numpy",
    "compact_vector397_tensor",
    "compact_circuit_state_dict",
    "retained_indices_from_dead_mask",
    "validate_dead_mask",
    "validate_retained_indices",
]
