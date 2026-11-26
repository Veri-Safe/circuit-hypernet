from __future__ import annotations

from dataclasses import dataclass

ADAPTIVE_REPLAY_POLICY = "replay_length_buckets_v1"
ADAPTIVE_REPLAY_DOWNSHIFT_POLICY = "replay_length_buckets_v1_downshift"
HIGH_UTIL_REPLAY_POLICY = "replay_length_buckets_high_util_v2"
FIXED_CAP_POLICY = "fixed_cap_v1"
SUPPORTED_COMPUTE_MICROBATCH_POLICIES = frozenset(
    {
        ADAPTIVE_REPLAY_POLICY,
        ADAPTIVE_REPLAY_DOWNSHIFT_POLICY,
        HIGH_UTIL_REPLAY_POLICY,
        FIXED_CAP_POLICY,
    }
)

# The buckets control the size of retained autograd graphs for each replay length.
ADAPTIVE_REPLAY_BATCH_BUCKETS: tuple[tuple[int, int], ...] = (
    (512, 160),
    (768, 64),
    (1024, 32),
    (1280, 20),
    (1536, 16),
    (2048, 8),
    (3072, 4),
    (4608, 2),
)

ADAPTIVE_REPLAY_DOWNSHIFT_BATCH_BUCKETS: tuple[tuple[int, int], ...] = tuple(
    (max_replay_length, max(1, batch_size // 2))
    for max_replay_length, batch_size in ADAPTIVE_REPLAY_BATCH_BUCKETS
)

# This alternative controls physical VJP microbatching for longer replays.
HIGH_UTIL_REPLAY_BATCH_BUCKETS: tuple[tuple[int, int], ...] = (
    (512, 160),
    (768, 96),
    (1024, 48),
    (1280, 32),
    (1536, 24),
    (2048, 12),
    (3072, 6),
    (4608, 2),
)


@dataclass(frozen=True, slots=True)
class AttributionBatchingPlan:
    replay_length: int
    logical_row_batch_size: int
    update_interval: int
    logical_queue_size: int
    compute_microbatch_policy: str
    compute_microbatch_cap: int
    compute_microbatch_size: int


def resolve_attribution_batching(
    *,
    replay_length: int,
    logical_row_batch_size: int,
    update_interval: int,
    compute_microbatch_policy: str,
    compute_microbatch_cap: int,
) -> AttributionBatchingPlan:
    """Resolve GPU execution size without changing CodeCircuit queue semantics."""

    if (
        min(
            replay_length,
            logical_row_batch_size,
            update_interval,
            compute_microbatch_cap,
        )
        <= 0
    ):
        raise ValueError("Attribution batching inputs must be positive")
    if compute_microbatch_policy not in SUPPORTED_COMPUTE_MICROBATCH_POLICIES:
        raise ValueError(
            f"Unsupported compute microbatch policy: {compute_microbatch_policy}"
        )
    if compute_microbatch_cap > logical_row_batch_size:
        raise ValueError("Compute microbatch cap cannot exceed the logical row batch")

    if compute_microbatch_policy == FIXED_CAP_POLICY:
        planned = compute_microbatch_cap
    else:
        if compute_microbatch_policy == ADAPTIVE_REPLAY_DOWNSHIFT_POLICY:
            buckets = ADAPTIVE_REPLAY_DOWNSHIFT_BATCH_BUCKETS
        elif compute_microbatch_policy == HIGH_UTIL_REPLAY_POLICY:
            buckets = HIGH_UTIL_REPLAY_BATCH_BUCKETS
        else:
            buckets = ADAPTIVE_REPLAY_BATCH_BUCKETS
        planned = 1
        for max_replay_length, bucket_batch_size in buckets:
            if replay_length <= max_replay_length:
                planned = bucket_batch_size
                break
    compute_microbatch_size = min(
        logical_row_batch_size,
        compute_microbatch_cap,
        planned,
    )
    return AttributionBatchingPlan(
        replay_length=int(replay_length),
        logical_row_batch_size=int(logical_row_batch_size),
        update_interval=int(update_interval),
        logical_queue_size=int(logical_row_batch_size * update_interval),
        compute_microbatch_policy=compute_microbatch_policy,
        compute_microbatch_cap=int(compute_microbatch_cap),
        compute_microbatch_size=int(compute_microbatch_size),
    )
