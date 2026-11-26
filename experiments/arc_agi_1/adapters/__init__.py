"""Circuit-conditioned NVARC Parameter Generator experiments."""

from .config import (
    LossProtocolConfig,
    OptimizationConfig,
    RuntimeOptimizationConfig,
    S4R1Config,
    load_config,
)
from .contracts import FUSION_VARIANTS, VECTOR_DIM, VECTOR_SCHEMA

__all__ = [
    "FUSION_VARIANTS",
    "LossProtocolConfig",
    "OptimizationConfig",
    "RuntimeOptimizationConfig",
    "S4R1Config",
    "VECTOR_DIM",
    "VECTOR_SCHEMA",
    "load_config",
]
