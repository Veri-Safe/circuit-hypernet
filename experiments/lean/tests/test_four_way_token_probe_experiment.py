"""CPU integration checks for frozen fitting and corrected historical controls."""
from pathlib import Path
import json
import sys

import pytest
import torch

SCRIPTS = Path(__file__).resolve().parents[1] / 'scripts'
sys.path.insert(0, str(SCRIPTS))
from four_way_token_probe import experiment as exp


class ToyEncoder:
    route = 'hn_query1024'

    def __init__(self):
        self.model = torch.nn.Embedding(16, 4).eval().requires_grad_(False)

    def encode(self, batch):
        ids = batch['query_ids']
        mask = torch.arange(ids.shape[1])[None] < batch['query_lengths'][:, None]
        with torch.no_grad():
            return self.model(ids).detach(), mask


def test_microbatch_fit_matches_logical_batch_and_keeps_encoder_frozen(tmp_path):
    torch.manual_seed(19)
    encoder = ToyEncoder()
    before = encoder.model.weight.detach().clone()
    bank = {'query_ids': torch.arange(27).reshape(9, 3) % 16,
            'query_lengths': torch.full((9,), 3), 'task_ids': torch.arange(9) % 4}
    cfg = {'training': {'rows': 9, 'epochs': 1, 'batch_size': 4, 'encoder_batch_size': 2,
                        'seed': 42, 'learning_rate': .001, 'weight_decay': .001}}
    torch.manual_seed(42)
    first = exp.TokenAttentionProbe(4, common_width=4, probe_width=8, hidden_width=8)
    torch.manual_seed(42)
    second = exp.TokenAttentionProbe(4, common_width=4, probe_width=8, hidden_width=8)
    (tmp_path / 'micro').mkdir(); (tmp_path / 'full').mkdir()
    assert exp.train_head(encoder, first, bank, cfg, tmp_path / 'micro') == 3
    cfg['training']['encoder_batch_size'] = 4
    assert exp.train_head(encoder, second, bank, cfg, tmp_path / 'full') == 3
    for a, b in zip(first.parameters(), second.parameters()):
        assert torch.allclose(a, b, atol=2e-6, rtol=1e-5)
    assert torch.equal(before, encoder.model.weight)
    assert encoder.model.weight.grad is None


def test_select_batch_never_forwards_gold_or_task_id():
    bank = {'query_ids': torch.zeros(2, 3, dtype=torch.long), 'query_lengths': torch.tensor([3, 3]),
            'vectors': torch.zeros(2, 397), 'gold': ['private', 'answer'], 'task_ids': torch.tensor([0, 1])}
    assert set(exp.select_batch(bank, torch.tensor([0]), 'hn_query1024')) == {'query_ids', 'query_lengths'}
    assert set(exp.select_batch(bank, torch.tensor([0]), 'circuit1024')) == {'vectors'}


def test_historical_control_relabelled_without_modifying_scores_or_file(tmp_path, monkeypatch):
    monkeypatch.setattr(exp, 'ROOT', tmp_path)
    event = 'ab' * 32
    old = {'domain': 'composition', 'condition': 'clean_no_identity', 'probe': 'hn_query1024',
           'task_type': 'T_and_or', 'event_key': event, 'bank_index': 2000,
           'labels': [0, 0, 1, 0], 'scores': [-.33866, .34267, .60117, .39482],
           'branch': 'Or.inr_single', 'success': True, 'auc': 1.0, 'status': 'scored'}
    source = tmp_path / 'old.jsonl'; source.write_text(json.dumps(old) + '\n')
    prior = source.read_bytes()
    manifest = {'per_question_sources': [{'domain': 'composition', 'condition': 'clean_no_identity',
                  'probe': 'hn_query1024', 'path': 'old.jsonl', 'sha256': exp.sha(source)}]}
    (tmp_path / 'manifest.json').write_text(json.dumps(manifest))
    cfg = {'control': {'source_manifest': {'path': 'manifest.json'}},
           'evaluation': {'atomic_conditions': [], 'composition_conditions': ['clean_no_identity']}}
    q = {'event_key': event, 'bank_index': 2000, 'task_type': 'T_and_or',
         'parent_query': 'state_0:\nA B C : Prop\nh : C\n\u22a2 (A \u2227 B) \u2228 C'}
    data = {'composition_original': {'rows': [q]}, 'atomic_original': {'rows': []}}
    predictions = {'composition_clean': torch.tensor([[.9, .1, .8, 0.]])}
    log = tmp_path / 'log'; log.mkdir()
    totals = exp.score_route(cfg, 'hn_query1024', predictions, data, log)
    selected = {r['method']: r for r in totals if r['task_type'] == 'ALL'}
    assert selected['mean_linear_ridge']['correct'] == 0
    assert selected['token_attention']['correct'] == 1
    assert source.read_bytes() == prior
    for path in (log / 'per_question').glob('*.jsonl'):
        row = exp.rows(path)[0]
        assert row['labels'] == [1, 0, 1, 0]


def test_score_rejects_wrong_event_mapping(tmp_path, monkeypatch):
    monkeypatch.setattr(exp, 'ROOT', tmp_path)
    assert exp.corrected_labels('\u22a2 (A \u2227 B) \u2228 C', [0, 0, 1, 0]) == [1, 0, 1, 0]
    with pytest.raises(ValueError, match='escapes'):
        exp.root_path('../outside.pt')
    with pytest.raises(ValueError, match='Revoked'):
        exp.root_path('any/final_test.pt')


def test_config_locks_training_and_approved_bank_denominator(tmp_path):
    path = SCRIPTS.parent / 'configs/four_way_token_probe_v2.json'
    config = json.loads(path.read_text(encoding='utf-8'))
    config['evaluation']['rows_per_family'] = 2000
    bad = tmp_path / 'bad.json'; bad.write_text(json.dumps(config))
    with pytest.raises(ValueError, match='approved'):
        exp.load_config(bad)
    config['evaluation']['rows_per_family'] = 500
    config['training']['evaluation_before_freeze'] = True
    bad.write_text(json.dumps(config))
    with pytest.raises(ValueError, match='evaluation_before_freeze'):
        exp.load_config(bad)
