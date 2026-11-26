from __future__ import annotations

import pytest
import torch
from torch import nn

from s4_circuit.lora_scale import (
    GeneratedLoRANormAccumulator,
    exact_batched_delta_frobenius,
    generated_lora_effective_delta_l2_per_query,
    scale_generated_lora_state,
    validate_lora_scale_multiplier,
)


def test_norm_accumulator_uses_gloo_compatible_cpu_collectives(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[str, str]] = []
    monkeypatch.setattr("s4_circuit.lora_scale.dist.is_available", lambda: True)
    monkeypatch.setattr("s4_circuit.lora_scale.dist.is_initialized", lambda: True)
    monkeypatch.setattr("s4_circuit.lora_scale.dist.get_backend", lambda: "gloo")

    def record_all_reduce(value: torch.Tensor, *, op: object) -> None:
        calls.append((value.device.type, str(op)))

    monkeypatch.setattr("s4_circuit.lora_scale.dist.all_reduce", record_all_reduce)

    value = torch.ones(3)
    GeneratedLoRANormAccumulator._all_reduce(value, op=torch.distributed.ReduceOp.SUM)

    assert calls == [("cpu", "RedOpType.SUM")]


def test_scale_generated_lora_state_changes_only_scale() -> None:
    a = torch.randn(2, 3, 5)
    b = torch.randn(2, 7, 3)
    state = {
        "model.layers.0.self_attn.q_proj.lora_A": a,
        "model.layers.0.self_attn.q_proj.lora_B": b,
        "scale": torch.tensor(2.0),
    }
    scaled = scale_generated_lora_state(state, 0.25)
    assert scaled["scale"].item() == pytest.approx(0.5)
    assert scaled[next(key for key in state if key.endswith("lora_A"))] is a
    assert scaled[next(key for key in state if key.endswith("lora_B"))] is b
    assert state["scale"].item() == pytest.approx(2.0)


def test_scale_zero_preserves_factors_and_zeros_effective_update() -> None:
    state = {
        "target.lora_A": torch.randn(1, 2, 3),
        "target.lora_B": torch.randn(1, 4, 2),
        "scale": torch.tensor(2.0),
    }
    scaled = scale_generated_lora_state(state, 0.0)
    assert scaled["scale"].item() == 0.0
    assert torch.equal(scaled["target.lora_A"], state["target.lora_A"])
    assert torch.equal(scaled["target.lora_B"], state["target.lora_B"])


def test_exact_low_rank_delta_norm_matches_materialized_matrix() -> None:
    generator = torch.Generator().manual_seed(24)
    a = torch.randn(4, 3, 5, generator=generator)
    b = torch.randn(4, 7, 3, generator=generator)
    expected = torch.bmm(b, a).flatten(1).norm(dim=1) * 1.75
    actual = exact_batched_delta_frobenius(a, b, scale=1.75)
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-5)


def test_effective_delta_l2_matches_materialized_normalized_update() -> None:
    generator = torch.Generator().manual_seed(25)
    a = torch.randn(3, 2, 5, generator=generator)
    b = torch.randn(3, 7, 2, generator=generator)
    state = {
        "first.lora_A": a,
        "first.lora_B": b,
        "second.lora_A": a * 0.5,
        "second.lora_B": b * 2.0,
        "scale": torch.tensor(0.5),
    }
    materialized = torch.bmm(b, a) * 0.5
    expected = materialized.square().mean(dim=(1, 2))
    actual = generated_lora_effective_delta_l2_per_query(state)
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)


def test_effective_delta_l2_is_factor_rescaling_invariant() -> None:
    a = torch.randn(2, 3, 4)
    b = torch.randn(2, 6, 3)
    base = {
        "target.lora_A": a,
        "target.lora_B": b,
        "scale": torch.tensor(2.0),
    }
    rescaled = {
        "target.lora_A": a * 8.0,
        "target.lora_B": b / 8.0,
        "scale": torch.tensor(2.0),
    }
    torch.testing.assert_close(
        generated_lora_effective_delta_l2_per_query(base),
        generated_lora_effective_delta_l2_per_query(rescaled),
    )


class _TinyTargetModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.model = nn.Module()
        self.model.layers = nn.ModuleList(
            [nn.ModuleDict({"proj": nn.Linear(3, 2, bias=False)}) for _ in range(36)]
        )
        self.lm_head = nn.Linear(3, 2, bias=False)


def test_norm_accumulator_caps_queries_and_reports_all_slots() -> None:
    model = _TinyTargetModel()
    target_keys = tuple(f"model.layers.{layer}.proj" for layer in range(36)) + (
        "lm_head",
    )
    generator = torch.Generator().manual_seed(42)
    state: dict[str, torch.Tensor] = {"scale": torch.tensor(2.0)}
    for target in target_keys:
        state[f"{target}.lora_A"] = torch.randn(4, 1, 3, generator=generator)
        state[f"{target}.lora_B"] = torch.randn(4, 2, 1, generator=generator)
    accumulator = GeneratedLoRANormAccumulator(
        model,
        target_keys=target_keys,
        device=torch.device("cpu"),
        max_queries=2,
    )
    accumulator.update(state)
    accumulator.update(state)
    summary = accumulator.finalize(lora_scale_multiplier=0.25)
    assert summary["query_count"] == 2
    assert summary["max_queries_per_rank"] == 2
    assert len(summary["slots"]) == 37
    first = summary["slots"][0]
    assert first["raw_delta_frobenius"]["mean"] > 0
    assert first["effective_delta_frobenius"]["mean"] == pytest.approx(
        first["raw_delta_frobenius"]["mean"] * 0.25
    )


@pytest.mark.parametrize("value", (-1.0, float("inf"), float("nan")))
def test_invalid_lora_scale_multiplier_is_rejected(value: float) -> None:
    with pytest.raises(ValueError):
        validate_lora_scale_multiplier(value)
