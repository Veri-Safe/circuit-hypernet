"""CPU contract tests; no checkpoint or evaluation bank is loaded."""
from pathlib import Path
import sys

import pytest
import torch
from torch import nn

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))
from four_way_token_probe import encoders


class ToyQuery(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(16, 1024)
        self.type_embedding = nn.Parameter(torch.full((1024,), 0.25))
        self.called_with_type = None

    def forward(self, ids, mask, *, add_type_embedding=True):
        self.called_with_type = add_type_embedding
        result = self.embedding(ids)
        if add_type_embedding:
            result = result + self.type_embedding
        return result.masked_fill(~mask.unsqueeze(-1), 0)


class ToyBase(nn.Module):
    def __init__(self, fail=False):
        super().__init__()
        self.embedding = nn.Embedding(16, 2048)
        self.transformer = nn.Module()
        self.transformer.ln_f = nn.LayerNorm(2048)
        self.lm_head = nn.Linear(2048, 16)
        self.fail = fail

    def forward(self, ids):
        hidden = self.transformer.ln_f(self.embedding(ids))
        if self.fail:
            raise RuntimeError("native failure after capture")
        return self.lm_head(hidden)


class ToyCircuit(nn.Module):
    def __init__(self):
        super().__init__()
        self.project = nn.Linear(397, 1024)
        self.type_embedding = nn.Parameter(torch.full((1024,), 0.5))

    def forward(self, vectors):
        return (self.project(vectors) + self.type_embedding).unsqueeze(1)


def make_adapter(monkeypatch, route, model):
    monkeypatch.setattr(encoders, "_load_route", lambda _route, _device: (model, {"origin": "test"}))
    return encoders.FrozenEncoder(route, "cpu")


def query_batch():
    return {"query_ids": torch.tensor([[1, 2, 3, 0], [4, 5, 0, 0]]),
            "query_lengths": torch.tensor([3, 2]),
            "gold": "must not reach encoder", "task_ids": torch.tensor([0, 1])}


@pytest.mark.parametrize("route", ["hn_query1024", "query_only_hn1024"])
def test_query_keeps_full_sequence_type_and_mask(monkeypatch, route):
    model = ToyQuery()
    adapter = make_adapter(monkeypatch, route, model)
    batch = query_batch()
    tokens, mask = adapter.encode(batch)
    assert tokens.shape == (2, 3, 1024)
    assert mask.tolist() == [[True, True, True], [True, True, False]]
    assert model.called_with_type is True
    assert torch.equal(tokens[0, 1], model.embedding.weight[2] + model.type_embedding)
    assert torch.equal(tokens[1, 2], torch.zeros(1024))
    assert not tokens.requires_grad and not tokens.is_inference()
    assert not model.training and not any(p.requires_grad for p in model.parameters())


def test_native_base_hook_is_before_lm_head_and_is_removed(monkeypatch):
    model = ToyBase()
    adapter = make_adapter(monkeypatch, "base_query2048", model)
    before = len(model.transformer.ln_f._forward_hooks)
    tokens, mask = adapter.encode(query_batch())
    with torch.no_grad():
        expected = model.transformer.ln_f(model.embedding(torch.tensor([[1, 2, 3], [4, 5, 0]])))
    assert tokens.shape == (2, 3, 2048)
    assert torch.equal(tokens[mask], expected[mask])
    assert not tokens[~mask].any()
    assert len(model.transformer.ln_f._forward_hooks) == before


def test_native_base_hook_is_removed_on_forward_error(monkeypatch):
    model = ToyBase(fail=True)
    adapter = make_adapter(monkeypatch, "base_query2048", model)
    with pytest.raises(RuntimeError, match="native failure"):
        adapter.encode(query_batch())
    assert not model.transformer.ln_f._forward_hooks


def test_circuit_reads_vector_only_and_returns_one_token(monkeypatch):
    model = ToyCircuit()
    adapter = make_adapter(monkeypatch, "circuit1024", model)
    vectors = torch.randn(3, 397)
    tokens, mask = adapter.encode({"vectors": vectors, "query_ids": "not used", "gold": "not used"})
    assert tokens.shape == (3, 1, 1024)
    assert mask.shape == (3, 1) and mask.all()
    with torch.no_grad():
        assert torch.equal(tokens, model(vectors))


def test_frozen_tokens_can_train_head_without_touching_encoder(monkeypatch):
    adapter = make_adapter(monkeypatch, "hn_query1024", ToyQuery())
    before = adapter.state_digest()
    with torch.inference_mode():
        tokens, mask = adapter.encode(query_batch())
    assert not tokens.is_inference() and not mask.is_inference()
    head = nn.Linear(1024, 4)
    optimizer = torch.optim.SGD(head.parameters(), lr=0.1)
    old = head.weight.detach().clone()
    optimizer.zero_grad()
    loss = head(tokens[mask]).square().mean()
    loss.backward()
    optimizer.step()
    assert not torch.equal(head.weight, old)
    assert adapter.state_digest() == before
    assert all(parameter.grad is None for parameter in adapter.model.parameters())


def test_state_digest_detects_mutation(monkeypatch):
    adapter = make_adapter(monkeypatch, "hn_query1024", ToyQuery())
    before = adapter.state_digest()
    with torch.no_grad():
        adapter.model.embedding.weight[0, 0].add_(1)
    assert adapter.state_digest() != before


@pytest.mark.parametrize("lengths", [torch.tensor([0, 2]), torch.tensor([5, 2]), torch.tensor([3., 2.])])
def test_invalid_lengths_fail(monkeypatch, lengths):
    adapter = make_adapter(monkeypatch, "hn_query1024", ToyQuery())
    batch = query_batch()
    batch["query_lengths"] = lengths
    with pytest.raises(ValueError):
        adapter.encode(batch)


def test_nonfinite_valid_output_fails(monkeypatch):
    adapter = make_adapter(monkeypatch, "hn_query1024", ToyQuery())
    with torch.no_grad():
        adapter.model.embedding.weight[1, 0] = float("nan")
    with pytest.raises(ValueError, match="nonfinite"):
        adapter.encode(query_batch())


def test_accidental_encoder_train_mode_fails(monkeypatch):
    adapter = make_adapter(monkeypatch, "hn_query1024", ToyQuery())
    adapter.model.train()
    with pytest.raises(ValueError, match="frozen"):
        adapter.encode(query_batch())


def test_unknown_route_fails_before_loading():
    with pytest.raises(ValueError, match="Unknown"):
        encoders.FrozenEncoder("composition_checkpoint", "cpu")
