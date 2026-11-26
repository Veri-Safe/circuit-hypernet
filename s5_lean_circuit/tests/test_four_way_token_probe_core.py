"""Scientific-contract checks for the four-way frozen token probe."""
from __future__ import annotations

from pathlib import Path
import sys

import pytest
import torch

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))
from four_way_token_probe.core import TokenAttentionProbe, corrected_labels, ranking_metrics


def make_probe(native_width=6, common_width=8):
    torch.manual_seed(42)
    return TokenAttentionProbe(native_width, common_width, probe_width=8, hidden_width=12)


def test_padding_values_do_not_change_scores_or_receive_attention():
    probe = make_probe().eval()
    hidden = torch.randn(2, 5, 6)
    mask = torch.tensor([[True, True, False, False, False], [True, True, True, True, False]])
    scores, weights = probe(hidden, mask, return_attention=True)
    changed = hidden.clone()
    changed[~mask] = 100_000 * torch.randn_like(changed[~mask])
    other, other_weights = probe(changed, mask, return_attention=True)
    torch.testing.assert_close(other, scores, rtol=0, atol=0)
    torch.testing.assert_close(other_weights, weights, rtol=0, atol=0)
    assert torch.count_nonzero(weights.masked_select(~mask[:, None, :])) == 0
    torch.testing.assert_close(weights.sum(-1), torch.ones(2, 4))


def test_padding_nan_and_inf_are_masked_before_projection_and_backward():
    probe = make_probe()
    hidden = torch.randn(2, 3, 6)
    mask = torch.tensor([[True, False, False], [True, True, False]])
    reference = probe(hidden, mask)
    hidden[0, 1] = float("nan")
    hidden[0, 2] = float("inf")
    hidden[1, 2] = -float("inf")
    scores = probe(hidden, mask)
    torch.testing.assert_close(scores, reference, rtol=0, atol=0)
    scores.square().sum().backward()
    assert all(parameter.grad is not None and torch.isfinite(parameter.grad).all()
               for parameter in probe.parameters())


def test_nonfinite_valid_token_is_rejected():
    probe = make_probe()
    hidden = torch.randn(1, 2, 6)
    hidden[0, 0, 0] = float("nan")
    with pytest.raises(ValueError, match="must be finite"):
        probe(hidden, torch.ones(1, 2, dtype=torch.bool))


def test_permuting_complete_tokens_and_mask_preserves_scores():
    probe = make_probe().eval()
    hidden = torch.randn(2, 5, 6)
    mask = torch.tensor([[True, True, False, True, False], [True, True, True, True, True]])
    perm = torch.tensor([3, 1, 4, 0, 2])
    scores, weights = probe(hidden, mask, return_attention=True)
    other, other_weights = probe(hidden[:, perm], mask[:, perm], return_attention=True)
    torch.testing.assert_close(scores, other, rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(weights[:, :, perm], other_weights, rtol=1e-5, atol=1e-6)


def test_head_training_does_not_backpropagate_to_encoder():
    torch.manual_seed(7)
    encoder = torch.nn.Linear(3, 6)
    probe = make_probe()
    encoder_before = {name: value.detach().clone() for name, value in encoder.named_parameters()}
    head_before = {name: value.detach().clone() for name, value in probe.named_parameters()}
    hidden = encoder(torch.randn(3, 4, 3))
    # A detached frozen representation is the ordinary use case. The head also
    # detaches defensively if a caller accidentally leaves encoder autograd on.
    assert hidden.requires_grad
    scores = probe(hidden, torch.ones(3, 4, dtype=torch.bool))
    optimizer = torch.optim.SGD(probe.parameters(), lr=0.01)
    scores.square().sum().backward()
    optimizer.step()
    assert all(parameter.grad is None for parameter in encoder.parameters())
    assert all(torch.equal(parameter, encoder_before[name])
               for name, parameter in encoder.named_parameters())
    assert any(not torch.equal(parameter, head_before[name])
               for name, parameter in probe.named_parameters())


def test_inference_mode_encoder_outputs_can_train_the_head():
    probe = make_probe()
    with torch.inference_mode():
        hidden = torch.randn(2, 3, 6)
        mask = torch.ones(2, 3, dtype=torch.bool)
    assert hidden.is_inference()
    probe(hidden, mask).square().sum().backward()
    assert torch.isfinite(probe.value.weight.grad).all()


def test_single_circuit_token_attention_is_one():
    probe = make_probe()
    hidden = torch.randn(3, 1, 6)
    scores, weights = probe(hidden, torch.ones(3, 1, dtype=torch.bool), return_attention=True)
    assert scores.shape == (3, 4)
    assert torch.equal(weights, torch.ones(3, 4, 1))
    scores.square().sum().backward()
    # No token-choice information can be learned from a sequence of length 1.
    assert torch.count_nonzero(probe.key.weight.grad) == 0
    assert torch.count_nonzero(probe.value.weight.grad) > 0


def test_native_width_does_not_change_head_parameter_count_or_initialization():
    narrow = make_probe(native_width=6, common_width=8)
    wide = make_probe(native_width=8, common_width=8)
    assert sum(p.numel() for p in narrow.parameters()) == sum(p.numel() for p in wide.parameters())
    assert narrow.state_dict().keys() == wide.state_dict().keys()
    for name in narrow.state_dict():
        assert torch.equal(narrow.state_dict()[name], wide.state_dict()[name])
    values = torch.randn(2, 3, 6)
    padded = torch.nn.functional.pad(values, (0, 2))
    mask = torch.ones(2, 3, dtype=torch.bool)
    torch.testing.assert_close(narrow(values, mask), wide(padded, mask), rtol=0, atol=0)


def test_production_widths_have_equal_registered_head_capacity():
    query_probe = TokenAttentionProbe(1024)
    base_probe = TokenAttentionProbe(2048)
    assert sum(p.numel() for p in query_probe.parameters()) == 541825
    assert sum(p.numel() for p in base_probe.parameters()) == 541825


@pytest.mark.parametrize("mask", [torch.zeros(2, 3, dtype=torch.bool),
                                  torch.ones(2, 3), torch.ones(2, 4, dtype=torch.bool)])
def test_invalid_or_empty_masks_are_rejected(mask):
    with pytest.raises(ValueError):
        make_probe()(torch.randn(2, 3, 6), mask)


def test_width_mismatch_is_rejected():
    with pytest.raises(ValueError, match="native_width"):
        make_probe()(torch.randn(2, 3, 7), torch.ones(2, 3, dtype=torch.bool))
    with pytest.raises(ValueError):
        TokenAttentionProbe(2049)


def test_relevance_adds_bypassed_and_and_preserves_or_direction_and_identity():
    old = [0, 0, 1, 1]
    new = corrected_labels("state_0:\nf : C\n\u22a2 (A \u2227 B) \u2228 C", old)
    assert new == [1, 0, 1, 1]
    assert old == [0, 0, 1, 1]


def test_relevance_reads_target_only_and_never_removes_proof_operations():
    assert corrected_labels("h : A \u2227 B\n\u22a2 C \u2228 D", [0, 0, 1, 0]) == [0, 0, 1, 0]
    assert corrected_labels("  \u22a2 A", [1, 0, 0, 0]) == [1, 0, 0, 0]


@pytest.mark.parametrize("query", ["no goal", "\u22a2 A\n\u22a2 B"])
def test_ambiguous_or_missing_goal_is_rejected(query):
    with pytest.raises(ValueError):
        corrected_labels(query, [0, 1, 0, 0])


def test_multilabel_success_requires_every_relevant_score_above_every_negative():
    failed = ranking_metrics([.35108, .15298, .34567, .15026], [1, 1, 0, 0], range(3))
    assert failed["success"] is False
    assert failed["auc"] == .5
    assert failed["margin"] == pytest.approx(-.19269)
    passed = ranking_metrics([.6, -.1, .8, 9.0], [1, 0, 1, 0], range(3))
    assert passed["success"] is True and passed["auc"] == 1.0
    assert ranking_metrics([.6, -.1, .8, 9.0], [1, 0, 1, 0], range(4))["success"] is False


def test_ties_fail_success_and_count_half_for_auc():
    result = ranking_metrics([.5, .5, .2, .0], [1, 0, 0, 0], range(3))
    assert result == {"success": False, "auc": .75, "margin": 0.0, "status": "scored"}


@pytest.mark.parametrize("labels,candidates", [([1, 1, 1, 0], range(3)),
                                                ([0, 0, 0, 1], range(3)),
                                                ([1, 0, 0, 0], [])])
def test_no_positive_or_no_negative_candidates_are_not_applicable(labels, candidates):
    result = ranking_metrics([.1, .2, .3, .4], labels, candidates)
    assert result == {"success": None, "auc": None, "margin": None, "status": "not_applicable"}


@pytest.mark.parametrize("scores,labels,candidates", [
    ([1., float("nan"), 0., 0.], [1, 0, 0, 0], range(3)),
    ([1., 0., 0., 0.], [1, 0, 0, 0], [0, 0, 1]),
    ([1., 0., 0., 0.], [1, 0, 0, 0], [4]),
    ([1., 0., 0., 0.], [1, 2, 0, 0], range(4)),
])
def test_invalid_scoring_inputs_fail_fast(scores, labels, candidates):
    with pytest.raises(ValueError):
        ranking_metrics(scores, labels, candidates)


def test_experiment_microbatch_weighting_matches_full_batch_updates(tmp_path):
    from four_way_token_probe.experiment import train_head

    class ToyFrozenEncoder:
        route = "hn_query1024"

        def __init__(self):
            torch.manual_seed(9)
            self.model = torch.nn.Embedding(11, 6).eval().requires_grad_(False)

        def encode(self, batch):
            tokens = self.model(batch["query_ids"]).detach()
            mask = torch.arange(tokens.shape[1])[None, :] < batch["query_lengths"][:, None]
            return tokens, mask

    # Both the global final batch (2 examples) and the first microbatch group
    # (2+2+1) are uneven, exercising denominators on the actual fitting path.
    bank = {"query_ids": torch.tensor([[0, 1, 2], [1, 3, 5], [2, 4, 6], [3, 5, 7],
                                      [4, 6, 8], [5, 7, 9], [6, 8, 10]]),
            "query_lengths": torch.tensor([3, 2, 1, 3, 2, 3, 2]),
            "task_ids": torch.tensor([0, 1, 2, 3, 0, 1, 2])}
    encoder = ToyFrozenEncoder()
    before = encoder.model.weight.detach().clone()
    common = {"rows": 7, "epochs": 1, "batch_size": 5, "seed": 42,
              "learning_rate": 0.0003, "weight_decay": 0.001}
    small, full = make_probe(), make_probe()
    micro_log, full_log = tmp_path / "micro", tmp_path / "full"
    micro_log.mkdir(); full_log.mkdir()
    micro_steps = train_head(encoder, small, bank,
                             {"training": common | {"encoder_batch_size": 2}}, micro_log)
    full_steps = train_head(encoder, full, bank,
                            {"training": common | {"encoder_batch_size": 5}}, full_log)
    assert micro_steps == full_steps == 2
    for name, value in small.state_dict().items():
        torch.testing.assert_close(value, full.state_dict()[name], rtol=1e-5, atol=2e-6)
    assert torch.equal(before, encoder.model.weight)
    assert encoder.model.weight.grad is None
