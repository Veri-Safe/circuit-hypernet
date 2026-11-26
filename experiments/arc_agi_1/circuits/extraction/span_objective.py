from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch
from torch import Tensor


@dataclass(frozen=True)
class SpanObjective:
    target_token_ids: Tensor
    target_positions: Tensor
    target_probabilities: Tensor
    target_entropies: Tensor
    target_log_probabilities: Tensor
    residual_directions: Tensor
    replay_seed: Tensor

    @property
    def target_count(self) -> int:
        return int(self.target_token_ids.numel())


def target_prediction_positions(
    condition_length: int, target_count: int
) -> tuple[int, ...]:
    if condition_length <= 0 or target_count <= 0:
        raise ValueError("Condition length and target count must be positive")
    return tuple(range(condition_length - 1, condition_length + target_count - 1))


def causal_replay_token_ids(
    condition_token_ids: Sequence[int], target_token_ids: Sequence[int]
) -> tuple[int, ...]:
    if not condition_token_ids or not target_token_ids:
        raise ValueError("Condition and target token IDs must be nonempty")
    return tuple(int(item) for item in condition_token_ids) + tuple(
        int(item) for item in target_token_ids[:-1]
    )


def build_span_objective(
    logits: Tensor,
    unembedding: Tensor,
    *,
    condition_length: int,
    target_token_ids: Sequence[int] | Tensor,
) -> SpanObjective:
    """Build the full-softmax joint log-probability seed at all target positions.

    ``logits`` is `(replay_length, vocab)` and ``unembedding`` is
    `(d_model, vocab)`. Directions live at the post-final-normalization,
    pre-unembedding interface.
    """

    if logits.ndim != 2 or unembedding.ndim != 2:
        raise ValueError("logits and unembedding must both be rank two")
    if logits.shape[1] != unembedding.shape[1]:
        raise ValueError("Vocabulary dimensions do not match")
    targets = torch.as_tensor(target_token_ids, device=logits.device, dtype=torch.long)
    if targets.ndim != 1 or targets.numel() == 0:
        raise ValueError("target_token_ids must be a nonempty vector")
    positions = torch.tensor(
        target_prediction_positions(condition_length, int(targets.numel())),
        device=logits.device,
        dtype=torch.long,
    )
    if int(positions[-1]) >= logits.shape[0]:
        raise ValueError(
            f"Replay logits have length {logits.shape[0]} but target position "
            f"{int(positions[-1])} is required"
        )
    selected_logits = logits.index_select(0, positions).float()
    log_probs = torch.log_softmax(selected_logits, dim=-1)
    probs = torch.exp(log_probs)
    target_log_probs = log_probs.gather(1, targets[:, None]).squeeze(1)
    target_probs = torch.exp(target_log_probs)
    entropies = -(probs * log_probs).sum(dim=-1)
    unembedding_float = unembedding.to(device=logits.device, dtype=torch.float32)
    expected_unembedding = probs @ unembedding_float.T
    target_unembedding = unembedding_float.T.index_select(0, targets)
    directions = target_unembedding - expected_unembedding
    replay_seed = torch.zeros(
        (logits.shape[0], unembedding.shape[0]),
        device=logits.device,
        dtype=torch.float32,
    )
    replay_seed.index_copy_(0, positions, directions)
    return SpanObjective(
        target_token_ids=targets,
        target_positions=positions,
        target_probabilities=target_probs,
        target_entropies=entropies,
        target_log_probabilities=target_log_probs,
        residual_directions=directions,
        replay_seed=replay_seed,
    )


def validate_generation_diagnostics(
    objective: SpanObjective,
    *,
    target_probabilities: Sequence[float],
    target_entropies: Sequence[float],
    target_log_probabilities: Sequence[float],
    atol: float,
    rtol: float,
) -> dict[str, float]:
    expected = (
        torch.tensor(
            target_probabilities,
            dtype=torch.float32,
            device=objective.target_probabilities.device,
        ),
        torch.tensor(
            target_entropies,
            dtype=torch.float32,
            device=objective.target_entropies.device,
        ),
        torch.tensor(
            target_log_probabilities,
            dtype=torch.float32,
            device=objective.target_log_probabilities.device,
        ),
    )
    actual = (
        objective.target_probabilities,
        objective.target_entropies,
        objective.target_log_probabilities,
    )
    labels = ("probability", "entropy", "log_probability")
    result: dict[str, float] = {}
    for label, actual_value, expected_value in zip(
        labels, actual, expected, strict=True
    ):
        if actual_value.shape != expected_value.shape:
            raise ValueError(f"Saved {label} diagnostics changed shape")
        error = float((actual_value.float() - expected_value).abs().max().item())
        result[f"{label}_max_abs"] = error
        if not torch.allclose(
            actual_value.float(), expected_value, atol=atol, rtol=rtol
        ):
            raise ValueError(
                f"Saved generation {label} diagnostics do not match replay"
            )
    return result
