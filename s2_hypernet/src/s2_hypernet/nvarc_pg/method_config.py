from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Mapping

from .variants import ExperimentVariant


@dataclass(frozen=True)
class MethodConfig:
    input_dim: int
    d_pg: int
    blocks: int
    head_dim: int
    token_dim: int
    dim_accumulation: int
    rank: int
    alpha: int
    dropout: float

    @classmethod
    def from_mapping(
        cls,
        value: Mapping[str, Any],
        *,
        variant: ExperimentVariant,
    ) -> "MethodConfig":
        missing = [field for field in cls.__annotations__ if field not in value]
        if missing:
            raise ValueError(f"Missing required method fields: {missing}")
        integer_fields = (
            "input_dim",
            "d_pg",
            "blocks",
            "head_dim",
            "token_dim",
            "dim_accumulation",
            "rank",
            "alpha",
        )
        result = cls(
            **{field: int(value[field]) for field in integer_fields},
            dropout=float(value["dropout"]),
        )
        expected = {
            "input_dim": 2560,
            "d_pg": variant.d_pg,
            "blocks": variant.blocks,
            "head_dim": 128,
            "token_dim": 1024,
            "dim_accumulation": 8,
            "rank": variant.rank,
            "alpha": variant.alpha,
            "dropout": 0.0,
        }
        actual = asdict(result)
        if actual != expected:
            raise ValueError(
                f"NVARC PG {variant.variant_id} method contract changed: "
                f"{actual} != {expected}"
            )
        return result
