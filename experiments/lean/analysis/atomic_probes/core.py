"""Frozen-representation token readout and the approved atomic relevance metric.

The probe observes an encoder's final output before query/circuit fusion. It
does not receive answers, proof labels, task IDs, or a second encoder's tokens.
Representations are detached here as well as frozen by the caller.

Zero-padding the feature axis preserves every native coordinate and gives all
four routes the same registered head parameters. This does not make their
effective input capacity identical: Base has 2048 native coordinates, while the
other routes have 1024, and Circuit has a single token. Attention over that
single token is always one; it is not attention over circuit graph nodes.
"""
from __future__ import annotations

import math
from numbers import Integral
from typing import Iterable, Sequence

import torch
from torch import Tensor, nn
from torch.nn import functional as F


class TokenAttentionProbe(nn.Module):
    """Four learned atomic queries read the complete, masked token sequence.

    Initialization follows the caller's ``torch.manual_seed``. There are no
    positional embeddings in this head: token positions have already been
    encoded by the frozen model. The head is invariant to reordering complete
    representation tokens together with their mask.
    """

    def __init__(
        self,
        native_width: int,
        common_width: int = 2048,
        probe_width: int = 128,
        hidden_width: int = 128,
        num_classes: int = 4,
    ) -> None:
        super().__init__()
        for name, value in (
            ("native_width", native_width), ("common_width", common_width),
            ("probe_width", probe_width), ("hidden_width", hidden_width),
            ("num_classes", num_classes),
        ):
            if not isinstance(value, Integral) or isinstance(value, bool) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if native_width > common_width:
            raise ValueError("native_width must not exceed common_width")
        if num_classes != 4:
            raise ValueError("The frozen label contract has exactly four atomic classes")
        self.native_width = int(native_width)
        self.common_width = int(common_width)
        self.probe_width = int(probe_width)
        self.num_classes = int(num_classes)
        self.key = nn.Linear(common_width, probe_width, bias=False)
        self.value = nn.Linear(common_width, probe_width, bias=True)
        self.class_queries = nn.Parameter(torch.empty(num_classes, probe_width))
        nn.init.normal_(self.class_queries, mean=0.0, std=0.02)
        self.output_norm = nn.LayerNorm(probe_width)
        self.readout = nn.Sequential(
            nn.Linear(probe_width, hidden_width),
            nn.GELU(),
            nn.Linear(hidden_width, 1),
        )

    def forward(
        self,
        hidden: Tensor,
        mask: Tensor,
        *,
        return_attention: bool = False,
    ) -> Tensor | tuple[Tensor, Tensor]:
        """Return raw ``[B,4]`` scores, optionally with ``[B,4,L]`` attention.

        Padding values are removed *before* projection, so even nonfinite
        values in excluded positions cannot contaminate valid scores. Invalid
        values in an included token fail rather than being silently repaired.
        """
        if hidden.ndim != 3 or hidden.shape[-1] != self.native_width:
            raise ValueError("hidden must have shape [B,L,native_width]")
        if hidden.shape[0] == 0 or hidden.shape[1] == 0:
            raise ValueError("The batch and token sequence must be nonempty")
        if mask.shape != hidden.shape[:2] or mask.dtype != torch.bool:
            raise ValueError("mask must be boolean with shape [B,L]")
        if hidden.device != self.key.weight.device or mask.device != hidden.device:
            raise ValueError("hidden, mask, and probe must be on the same device")
        if not hidden.is_floating_point() or hidden.dtype != self.key.weight.dtype:
            raise ValueError("hidden must use the probe's floating-point dtype")
        if not bool(mask.any(dim=1).all()):
            raise ValueError("Every example must have at least one valid token")
        safe_hidden = torch.where(mask.unsqueeze(-1), hidden.detach(), 0.0)
        if not bool(torch.isfinite(safe_hidden).all()):
            raise ValueError("Valid token representations must be finite")
        if self.native_width < self.common_width:
            safe_hidden = F.pad(safe_hidden, (0, self.common_width - self.native_width))
        keys = self.key(safe_hidden)
        values = self.value(safe_hidden)
        logits = torch.einsum("cd,bld->bcl", self.class_queries, keys)
        logits = logits / math.sqrt(self.probe_width)
        logits = logits.masked_fill(~mask.unsqueeze(1), float("-inf"))
        attention = logits.softmax(dim=-1)
        pooled = torch.einsum("bcl,bld->bcd", attention, values)
        states = self.output_norm(pooled + self.class_queries.unsqueeze(0))
        scores = self.readout(states).squeeze(-1)
        if not bool(torch.isfinite(scores).all()):
            raise ValueError("Probe scores became nonfinite")
        return (scores, attention) if return_attention else scores


def _binary_labels(labels: Sequence[int]) -> list[int]:
    if len(labels) != 4 or any(
        not isinstance(value, Integral) or value not in (0, 1) for value in labels
    ):
        raise ValueError("Labels must contain four binary values: And, Or-left, Or-right, Identity")
    return [int(value) for value in labels]


def corrected_labels(query: str, proof_labels: Sequence[int]) -> list[int]:
    """Apply the user-confirmed 2026-11-21 And relevance correction.

    A conjunction anywhere in the target is relevant even if the reference
    proof bypasses it. Or-left/right remain actual reference-proof operations;
    the caller's Identity convention is preserved. The input list is unchanged.
    """
    labels = _binary_labels(proof_labels)
    if not isinstance(query, str):
        raise ValueError("query must be text")
    goals = [line.lstrip() for line in query.splitlines() if line.lstrip().startswith("\u22a2")]
    if len(goals) != 1:
        raise ValueError("Expected exactly one target line beginning with the turnstile")
    labels[0] = int(bool(labels[0]) or "\u2227" in goals[0])
    return labels


def ranking_metrics(
    scores: Sequence[float],
    labels: Sequence[int],
    candidates: Iterable[int],
) -> dict[str, bool | float | str | None]:
    """Strict relevant-set ranking and within-question pairwise AUROC.

    Candidates are explicitly supplied, typically range(3) when Identity is
    excluded or range(4) when included. Ties fail strict success and count 0.5
    in AUROC. Questions without both positive and negative candidates are N/A.
    """
    binary = _binary_labels(labels)
    if len(scores) != 4:
        raise ValueError("Expected four atomic scores")
    values = [float(value) for value in scores]
    if not all(math.isfinite(value) for value in values):
        raise ValueError("Atomic scores must be finite")
    indices = list(candidates)
    if any(
        not isinstance(index, Integral) or isinstance(index, bool) or not 0 <= index < 4
        for index in indices
    ) or len(set(indices)) != len(indices):
        raise ValueError("Candidates must be unique atomic indices between 0 and 3")
    positives = [values[index] for index in indices if binary[index]]
    negatives = [values[index] for index in indices if not binary[index]]
    if not positives or not negatives:
        return {"success": None, "auc": None, "margin": None, "status": "not_applicable"}
    margin = min(positives) - max(negatives)
    auc = sum(
        float(a > b) + 0.5 * float(a == b) for a in positives for b in negatives
    ) / (len(positives) * len(negatives))
    return {"success": margin > 0, "auc": auc, "margin": margin, "status": "scored"}
