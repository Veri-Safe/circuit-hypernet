from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ExperimentVariant:
    variant_id: str
    d_pg: int
    blocks: int
    rank: int
    alpha: int
    peak_lr: float
    min_lr: float
    factor_l2: float
    hypothesis: str


BASE_VARIANT = ExperimentVariant(
    variant_id="base",
    d_pg=1280,
    blocks=12,
    rank=16,
    alpha=32,
    peak_lr=5e-5,
    min_lr=5e-6,
    factor_l2=0.0,
    hypothesis="Anchor configuration selected in the original design round.",
)

PG_LR4E5_VARIANT = ExperimentVariant(
    variant_id="pg_lr4e5",
    d_pg=1280,
    blocks=12,
    rank=16,
    alpha=32,
    peak_lr=4e-5,
    min_lr=4e-6,
    factor_l2=0.0,
    hypothesis=(
        "A 0.8x PG learning-rate schedule may retain the early Safe-P2E gain "
        "without the later validation regression."
    ),
)

PG_LR6P5E5_VARIANT = ExperimentVariant(
    variant_id="pg_lr6p5e5",
    d_pg=1280,
    blocks=12,
    rank=16,
    alpha=32,
    peak_lr=6.5e-5,
    min_lr=6.5e-6,
    factor_l2=0.0,
    hypothesis=(
        "A 1.3x PG learning-rate schedule may reach a stronger early checkpoint "
        "while Safe-P2E remains RMS bounded."
    ),
)

LR2X_VARIANT = ExperimentVariant(
    variant_id="lr2x",
    d_pg=1280,
    blocks=12,
    rank=16,
    alpha=32,
    peak_lr=1e-4,
    min_lr=1e-5,
    factor_l2=0.0,
    hypothesis="One epoch may under-optimize the PG; scale the full LR schedule by 2x.",
)


def _generated_lora_l2_variant(
    *,
    variant_id: str,
    peak_lr: float,
    min_lr: float,
    factor_l2: float,
) -> ExperimentVariant:
    return ExperimentVariant(
        variant_id=variant_id,
        d_pg=1280,
        blocks=12,
        rank=16,
        alpha=32,
        peak_lr=peak_lr,
        min_lr=min_lr,
        factor_l2=factor_l2,
        hypothesis=(
            "Penalize the per-query mean squared magnitude of every generated "
            f"LoRA A/B factor with coefficient {factor_l2:g}."
        ),
    )


GENERATED_LORA_L2_VARIANTS = (
    _generated_lora_l2_variant(
        variant_id="pg4_l2e4", peak_lr=4e-5, min_lr=4e-6, factor_l2=1e-4
    ),
    _generated_lora_l2_variant(
        variant_id="pg4_l2_1e3", peak_lr=4e-5, min_lr=4e-6, factor_l2=1e-3
    ),
    _generated_lora_l2_variant(
        variant_id="pg4_l2_1e2", peak_lr=4e-5, min_lr=4e-6, factor_l2=1e-2
    ),
    _generated_lora_l2_variant(
        variant_id="pg4_l2_3e2", peak_lr=4e-5, min_lr=4e-6, factor_l2=3e-2
    ),
    _generated_lora_l2_variant(
        variant_id="pg4_l2_1e1", peak_lr=4e-5, min_lr=4e-6, factor_l2=1e-1
    ),
    _generated_lora_l2_variant(
        variant_id="pg5_l2_1e3", peak_lr=5e-5, min_lr=5e-6, factor_l2=1e-3
    ),
    _generated_lora_l2_variant(
        variant_id="pg5_l2_1e2", peak_lr=5e-5, min_lr=5e-6, factor_l2=1e-2
    ),
    _generated_lora_l2_variant(
        variant_id="pg5_l2_3e2", peak_lr=5e-5, min_lr=5e-6, factor_l2=3e-2
    ),
    _generated_lora_l2_variant(
        variant_id="pg5_l2_1e1", peak_lr=5e-5, min_lr=5e-6, factor_l2=1e-1
    ),
)

PGWIDE1536_VARIANT = ExperimentVariant(
    variant_id="pgwide1536",
    d_pg=1536,
    blocks=12,
    rank=16,
    alpha=32,
    peak_lr=5e-5,
    min_lr=5e-6,
    factor_l2=0.0,
    hypothesis="A wider generator may model query-to-adapter mappings more accurately.",
)

R32A64_VARIANT = ExperimentVariant(
    variant_id="r32a64",
    d_pg=1280,
    blocks=12,
    rank=32,
    alpha=64,
    peak_lr=5e-5,
    min_lr=5e-6,
    factor_l2=0.0,
    hypothesis="Higher adapter rank may provide stronger per-query actuation at scale 2.",
)

L2E4_VARIANT = ExperimentVariant(
    variant_id="l2e4",
    d_pg=1280,
    blocks=12,
    rank=16,
    alpha=32,
    peak_lr=5e-5,
    min_lr=5e-6,
    factor_l2=1e-4,
    hypothesis=(
        "HY-WU-style factor regularization may preserve the saturated base while "
        "retaining query-specific corrections."
    ),
)

LR2X_L2E4_VARIANT = ExperimentVariant(
    variant_id="lr2x_l2e4",
    d_pg=1280,
    blocks=12,
    rank=16,
    alpha=32,
    peak_lr=1e-4,
    min_lr=1e-5,
    factor_l2=1e-4,
    hypothesis=(
        "Test whether the faster LR schedule benefits from the same factor-norm "
        "constraint on a saturated base."
    ),
)

DELTA4_LR1E5_L2E4_VARIANT = ExperimentVariant(
    variant_id="delta4_lr1e5_l2e4",
    d_pg=1280,
    blocks=12,
    rank=16,
    alpha=32,
    peak_lr=1e-5,
    min_lr=1e-6,
    factor_l2=1e-4,
    hypothesis=(
        "A conservative PG learning rate plus light generated-factor regularization "
        "may improve the saturated RE-ARC64 base without erasing its solution prior."
    ),
)

DELTA4_LR2E5_L2E4_VARIANT = ExperimentVariant(
    variant_id="delta4_lr2e5_l2e4",
    d_pg=1280,
    blocks=12,
    rank=16,
    alpha=32,
    peak_lr=2e-5,
    min_lr=2e-6,
    factor_l2=1e-4,
    hypothesis=(
        "A 2x PG learning rate may fit the small disjoint delta block faster while "
        "light factor regularization limits query-adapter magnitude."
    ),
)

DELTA4_LR1E5_L2E3_VARIANT = ExperimentVariant(
    variant_id="delta4_lr1e5_l2e3",
    d_pg=1280,
    blocks=12,
    rank=16,
    alpha=32,
    peak_lr=1e-5,
    min_lr=1e-6,
    factor_l2=1e-3,
    hypothesis=(
        "Stronger factor regularization may better preserve a saturated base when the "
        "delta corpus is repeatedly traversed."
    ),
)

DELTA4_LR2E5_L2E3_VARIANT = ExperimentVariant(
    variant_id="delta4_lr2e5_l2e3",
    d_pg=1280,
    blocks=12,
    rank=16,
    alpha=32,
    peak_lr=2e-5,
    min_lr=2e-6,
    factor_l2=1e-3,
    hypothesis=(
        "The high-LR/high-regularization corner tests whether faster fitting can retain "
        "a directional matched-over-shuffled gain."
    ),
)


def _raw_weight_scale_v10_variant(
    *,
    variant_id: str,
    d_pg: int,
    rank: int,
    alpha: int,
    peak_lr: float = 4e-5,
    min_lr: float = 4e-6,
    hypothesis: str,
) -> ExperimentVariant:
    return ExperimentVariant(
        variant_id=variant_id,
        d_pg=d_pg,
        blocks=12,
        rank=rank,
        alpha=alpha,
        peak_lr=peak_lr,
        min_lr=min_lr,
        factor_l2=1e-4,
        hypothesis=hypothesis,
    )


# Raw-weight scale-v10 keeps the Safe-P2E conditioner fixed and lowers the
# generated LoRA actuation.  These variants intentionally encode the complete
# PG/LoRA contract so a matrix row cannot silently change only the YAML.
RAW_WEIGHT_SCALE_V10_VARIANTS = (
    _raw_weight_scale_v10_variant(
        variant_id="rw10_r16a4_d1280",
        d_pg=1280,
        rank=16,
        alpha=4,
        hypothesis="Low-scale rank-16 anchor for the raw-weight comparison.",
    ),
    _raw_weight_scale_v10_variant(
        variant_id="rw10_r8a2_d1280",
        d_pg=1280,
        rank=8,
        alpha=2,
        hypothesis="Lower rank may make the generated correction easier to optimize.",
    ),
    _raw_weight_scale_v10_variant(
        variant_id="rw10_r32a8_d1280",
        d_pg=1280,
        rank=32,
        alpha=8,
        hypothesis="Higher rank tests whether the conditioner is capacity limited.",
    ),
    _raw_weight_scale_v10_variant(
        variant_id="rw10_r16a4_d1024",
        d_pg=1024,
        rank=16,
        alpha=4,
        hypothesis="A smaller PG may regularize the per-query adapter mapping.",
    ),
    _raw_weight_scale_v10_variant(
        variant_id="rw10_r16a4_d1536",
        d_pg=1536,
        rank=16,
        alpha=4,
        hypothesis="A wider PG tests whether generator capacity is the bottleneck.",
    ),
    _raw_weight_scale_v10_variant(
        variant_id="rw10_r16a2_d1280",
        d_pg=1280,
        rank=16,
        alpha=2,
        hypothesis="Half-anchor LoRA scale reduces destructive base perturbation.",
    ),
    _raw_weight_scale_v10_variant(
        variant_id="rw10_r16a1_d1280",
        d_pg=1280,
        rank=16,
        alpha=1,
        hypothesis="Very small LoRA scale tests the safest useful correction regime.",
    ),
    _raw_weight_scale_v10_variant(
        variant_id="rw10_r8a1_d1024_lr2",
        d_pg=1024,
        rank=8,
        alpha=1,
        peak_lr=2e-5,
        min_lr=2e-6,
        hypothesis="Joint conservative rank, scale, capacity, and LR configuration.",
    ),
)


def _raw_weight_e11_tune_v11_variant(
    *,
    variant_id: str,
    peak_lr: float,
    factor_l2: float,
    hypothesis: str,
) -> ExperimentVariant:
    return ExperimentVariant(
        variant_id=variant_id,
        d_pg=1280,
        blocks=12,
        rank=16,
        alpha=4,
        peak_lr=peak_lr,
        # Keep E11's absolute floor fixed; only the approved peak LR changes.
        min_lr=4e-6,
        factor_l2=factor_l2,
        hypothesis=hypothesis,
    )


RAW_WEIGHT_E11_TUNE_V11_VARIANTS = (
    _raw_weight_e11_tune_v11_variant(
        variant_id="rw11_pg3_l2e4",
        peak_lr=3e-5,
        factor_l2=1e-4,
        hypothesis="E11 architecture with a slower PG peak and the original factor L2.",
    ),
    _raw_weight_e11_tune_v11_variant(
        variant_id="rw11_pg2_l2e4",
        peak_lr=2e-5,
        factor_l2=1e-4,
        hypothesis="Conservative PG and raw-conditioner optimization from step zero.",
    ),
    _raw_weight_e11_tune_v11_variant(
        variant_id="rw11_pg5_l2e4",
        peak_lr=5e-5,
        factor_l2=1e-4,
        hypothesis="Faster PG optimization while retaining the E11 LoRA scale and cap.",
    ),
    _raw_weight_e11_tune_v11_variant(
        variant_id="rw11_pg4_l2_3e4",
        peak_lr=4e-5,
        factor_l2=3e-4,
        hypothesis="E11 PG rate with moderately stronger generated-factor regularization.",
    ),
    _raw_weight_e11_tune_v11_variant(
        variant_id="rw11_pg3_l2e3",
        peak_lr=3e-5,
        factor_l2=1e-3,
        hypothesis="Slow PG optimization with a stronger generated-factor norm penalty.",
    ),
    _raw_weight_e11_tune_v11_variant(
        variant_id="rw11_pg3_l2_3e4",
        peak_lr=3e-5,
        factor_l2=3e-4,
        hypothesis="Slow PG optimization plus moderate factor regularization.",
    ),
)


VARIANTS = {
    variant.variant_id: variant
    for variant in (
        BASE_VARIANT,
        PG_LR4E5_VARIANT,
        PG_LR6P5E5_VARIANT,
        LR2X_VARIANT,
        *GENERATED_LORA_L2_VARIANTS,
        PGWIDE1536_VARIANT,
        R32A64_VARIANT,
        L2E4_VARIANT,
        LR2X_L2E4_VARIANT,
        DELTA4_LR1E5_L2E4_VARIANT,
        DELTA4_LR2E5_L2E4_VARIANT,
        DELTA4_LR1E5_L2E3_VARIANT,
        DELTA4_LR2E5_L2E3_VARIANT,
        *RAW_WEIGHT_SCALE_V10_VARIANTS,
        *RAW_WEIGHT_E11_TUNE_V11_VARIANTS,
    )
}


def variant_for_id(variant_id: str) -> ExperimentVariant:
    try:
        return VARIANTS[variant_id]
    except KeyError as error:
        raise ValueError(f"Unknown NVARC PG variant_id: {variant_id!r}") from error
