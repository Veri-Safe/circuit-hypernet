from __future__ import annotations

import pytest
import torch

from s2_hypernet.config import ParameterGeneratorConfig

from s4_circuit.pg_core import S4CircuitParameterGenerator


def _tiny_config() -> ParameterGeneratorConfig:
    return ParameterGeneratorConfig(
        token_dim=8,
        rank=2,
        alpha=4,
        input_dim=12,
        dim_accumulation=1,
        d_model=16,
        head_dim=4,
        num_base_model_layers=37,
        output_dim=8,
        num_pg_layers=2,
        prefix="model.layers.",
        pg_mapping={
            "self_attn.q_proj": {"lora_A_dim": 4, "lora_B_dim": 4},
            "lm_head": {"lora_A_dim": 4, "lora_B_dim": 4},
        },
        target_layer_indices=list(range(37)),
        near_zero_b_init_std=0.0,
        condition_injection_mode="none",
    )


def _inputs() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    torch.manual_seed(8)
    condition = torch.randn(2, 5, 12)
    mask = torch.tensor([[1, 1, 1, 0, 0], [1, 1, 1, 1, 1]])
    circuit = torch.randn(2, 59, dtype=torch.float32)
    return condition, mask, circuit


def _assert_state_equal(left: dict[str, torch.Tensor], right: dict[str, torch.Tensor]) -> None:
    assert left.keys() == right.keys()
    for key in left:
        assert torch.equal(left[key], right[key]), key


def test_f2_zero_init_matches_query_only_exactly() -> None:
    torch.manual_seed(42)
    model = S4CircuitParameterGenerator(_tiny_config(), variant_id="f2_fully_direct", encoder_hidden_dim=4)
    condition, mask, circuit = _inputs()
    query_only = model.base(condition, condition_attention_mask=mask)
    fused = model(
        condition,
        condition_attention_mask=mask,
        circuit_vector=circuit,
    )
    _assert_state_equal(query_only, fused)


def test_f3_zero_init_matches_query_only_exactly() -> None:
    condition, mask, circuit = _inputs()
    for variant in ("f3_shared_attention", "f3_independent_attention"):
        torch.manual_seed(42)
        model = S4CircuitParameterGenerator(_tiny_config(), variant_id=variant, encoder_hidden_dim=4)
        query_only = model.base(condition, condition_attention_mask=mask)
        fused = model(
            condition,
            condition_attention_mask=mask,
            circuit_vector=circuit,
        )
        _assert_state_equal(query_only, fused)


def test_f0_f1_absent_remove_circuit_memory() -> None:
    condition, mask, circuit = _inputs()
    absent = torch.zeros(condition.shape[0], dtype=torch.bool)
    for variant in ("f0_single_token", "f1_layer_shared_memory"):
        torch.manual_seed(42)
        model = S4CircuitParameterGenerator(_tiny_config(), variant_id=variant, encoder_hidden_dim=4)
        query_only = model.base(condition, condition_attention_mask=mask)
        fused = model(
            condition,
            condition_attention_mask=mask,
            circuit_vector=circuit,
            circuit_present=absent,
        )
        _assert_state_equal(query_only, fused)


def test_f3_attention_ownership_and_zero_output_projections() -> None:
    shared = S4CircuitParameterGenerator(_tiny_config(), variant_id="f3_shared_attention", encoder_hidden_dim=4)
    independent = S4CircuitParameterGenerator(_tiny_config(), variant_id="f3_independent_attention", encoder_hidden_dim=4)
    assert len(shared.circuit_attentions) == 0
    assert len(independent.circuit_attentions) == 2
    assert len(shared.circuit_output_projections) == 2
    assert len(independent.circuit_output_projections) == 2
    for model in (shared, independent):
        for projection in model.circuit_output_projections:
            assert torch.count_nonzero(projection.weight) == 0
            assert torch.count_nonzero(projection.bias) == 0


def test_f3_zero_branch_projection_receives_a_first_step_gradient() -> None:
    condition, mask, circuit = _inputs()
    torch.manual_seed(42)
    model = S4CircuitParameterGenerator(
        _tiny_config(),
        variant_id="f3_shared_attention",
        encoder_hidden_dim=4,
    )
    state = model(
        condition,
        condition_attention_mask=mask,
        circuit_vector=circuit,
    )
    loss = sum(value.float().sum() for key, value in state.items() if key != "scale")
    loss.backward()
    gradients = [projection.weight.grad for projection in model.circuit_output_projections]
    assert all(value is not None for value in gradients)
    assert any(torch.count_nonzero(value) > 0 for value in gradients if value is not None)


def test_f3_global_topology_encoder_is_not_a_dead_two_zero_path() -> None:
    condition, mask, circuit = _inputs()
    torch.manual_seed(42)
    model = S4CircuitParameterGenerator(
        _tiny_config(),
        variant_id="f3_shared_attention",
        encoder_hidden_dim=4,
    )

    def backward_once() -> None:
        state = model(
            condition,
            condition_attention_mask=mask,
            circuit_vector=circuit,
        )
        sum(value.float().sum() for key, value in state.items() if key != "scale").backward()

    backward_once()
    with torch.no_grad():
        for projection in model.circuit_output_projections:
            assert projection.weight.grad is not None
            projection.weight.add_(projection.weight.grad, alpha=-1.0e-3)
    model.zero_grad(set_to_none=True)
    backward_once()
    assert model.fusion.structured_encoder is not None
    global_final = model.fusion.structured_encoder.global_encoder[-1]
    topology_final = model.fusion.structured_encoder.topology_encoder[-1]
    assert global_final.weight.grad is not None
    assert topology_final.weight.grad is not None
    assert torch.count_nonzero(global_final.weight.grad) > 0
    assert torch.count_nonzero(topology_final.weight.grad) > 0


def test_only_fusion_changes_when_common_trunk_is_reloaded() -> None:
    torch.manual_seed(42)
    source = S4CircuitParameterGenerator(_tiny_config(), variant_id="f0_single_token", encoder_hidden_dim=4)
    state = source.common_trunk_state_dict()
    models = []
    for variant in ("f2_fully_direct", "f3_independent_attention"):
        torch.manual_seed(100 + len(models))
        model = S4CircuitParameterGenerator(_tiny_config(), variant_id=variant, encoder_hidden_dim=4)
        model.load_common_trunk_state_dict(state)
        models.append(model)
    for key in state:
        assert torch.equal(models[0].base.state_dict()[key], models[1].base.state_dict()[key])


def test_trainable_global_lora_scale_is_checkpointed_and_applied() -> None:
    torch.manual_seed(42)
    fixed = S4CircuitParameterGenerator(
        _tiny_config(),
        variant_id="f3_independent_attention",
        encoder_hidden_dim=4,
    )
    learned = S4CircuitParameterGenerator(
        _tiny_config(),
        variant_id="f3_independent_attention",
        encoder_hidden_dim=4,
        trainable_global_lora_scale=True,
        global_lora_scale_initial=1.0,
        global_lora_scale_max=1.5,
    )
    condition, mask, circuit = _inputs()
    fixed_state = fixed(
        condition,
        condition_attention_mask=mask,
        circuit_vector=circuit,
    )
    learned_state = learned(
        condition,
        condition_attention_mask=mask,
        circuit_vector=circuit,
    )

    assert learned.global_lora_scale_control is not None
    assert learned.global_lora_scale_value().item() == pytest.approx(1.0)
    assert learned_state["scale"].item() == pytest.approx(
        fixed_state["scale"].item()
    )
    assert "global_lora_scale_control.logit" in learned.state_dict()
    assert "global_lora_scale_control.logit" not in fixed.state_dict()
    learned_state["scale"].float().backward()
    assert learned.global_lora_scale_control.logit.grad is not None
    assert torch.count_nonzero(learned.global_lora_scale_control.logit.grad) == 1
