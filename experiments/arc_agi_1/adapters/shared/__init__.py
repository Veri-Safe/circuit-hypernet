"""S2 Hypernet: HY-WU-style generated residual LoRA for Gemma 2."""

from .config import ParameterGeneratorConfig, gemma2_s2_pg_mapping
from .parameter_generator import ParameterGenerator

__all__ = [
    "ParameterGenerator",
    "ParameterGeneratorConfig",
    "gemma2_s2_pg_mapping",
]
