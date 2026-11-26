from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
from torch import nn

from s5_lean_circuit.hypernet.query_encoder import ConditionTypeEmbeddings
from s5_lean_circuit.hypernet.runtime import S5HyperNet


class _TinyQuery(nn.Module):
    def __init__(self, type_embeddings: ConditionTypeEmbeddings) -> None:
        super().__init__()
        self.type_embeddings = type_embeddings
        self.projection = nn.Linear(4, 4)


class _TinyParameterGenerator(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.projection = nn.Linear(4, 4)


def _dummy_runtime(seed: int) -> SimpleNamespace:
    torch.manual_seed(seed)
    embeddings = ConditionTypeEmbeddings(width=4, block_count=5)
    runtime = SimpleNamespace(
        type_embeddings=embeddings,
        query_encoder=_TinyQuery(embeddings),
        parameter_generator=_TinyParameterGenerator(),
    )
    runtime.common_state_dict = lambda: _common_state(runtime)
    return runtime


def _common_state(runtime: SimpleNamespace) -> dict[str, torch.Tensor]:
    state = {
        "type_embeddings.query": runtime.type_embeddings.query.detach().clone(),
        "type_embeddings.circuit": runtime.type_embeddings.circuit.detach().clone(),
        "type_embeddings.block": runtime.type_embeddings.block.detach().clone(),
    }
    state.update(
        {
            f"query_encoder.{key}": value.detach().clone()
            for key, value in runtime.query_encoder.state_dict().items()
            if not key.startswith("type_embeddings.")
        }
    )
    state.update(
        {
            f"parameter_generator.{key}": value.detach().clone()
            for key, value in runtime.parameter_generator.state_dict().items()
        }
    )
    return state


def test_common_state_round_trip_restores_shared_type_embeddings() -> None:
    source = _dummy_runtime(seed=42)
    target = _dummy_runtime(seed=43)
    state = _common_state(source)

    S5HyperNet.load_common_state_dict(target, state)

    restored = _common_state(target)
    assert set(restored) == set(state)
    for key in state:
        torch.testing.assert_close(restored[key], state[key])


def test_common_state_loader_rejects_missing_shared_embedding() -> None:
    source = _dummy_runtime(seed=42)
    target = _dummy_runtime(seed=43)
    state = _common_state(source)
    state.pop("type_embeddings.block")

    with pytest.raises(ValueError, match="common S5 init keys changed"):
        S5HyperNet.load_common_state_dict(target, state)


def test_query_only_condition_memory_ignores_circuit_vector() -> None:
    class TinyQueryEncoder(nn.Module):
        def forward(self, token_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
            return token_ids.float().unsqueeze(-1).expand(-1, -1, 4)

    runtime = S5HyperNet.__new__(S5HyperNet)
    nn.Module.__init__(runtime)
    runtime.condition_source = "query_only"
    runtime.query_encoder = TinyQueryEncoder()
    ids = torch.tensor([[1, 2, 0]])
    mask = torch.tensor([[True, True, False]])
    first, first_mask = runtime.condition_memory(ids, mask, torch.zeros(1, 397))
    second, second_mask = runtime.condition_memory(ids, mask, torch.ones(1, 397))
    assert first.shape == (1, 3, 4)
    assert torch.equal(first, second)
    assert torch.equal(first_mask, mask)
    assert torch.equal(second_mask, mask)
