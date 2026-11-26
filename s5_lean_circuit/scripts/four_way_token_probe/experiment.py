"""Data identity, fitting and evaluation for the four-way token probe.

No graph extraction, model training, test-set selection, or bank generation.
Large input artifacts are verified once by preflight and stat-checked by workers.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import subprocess
import time

import torch
from torch.nn import functional as F

from .core import TokenAttentionProbe, corrected_labels, ranking_metrics
from .encoders import FrozenEncoder

ROOT = Path(__file__).resolve().parents[3]
TASKS = ('S_and', 'S_or_l', 'S_or_r', 'S_id')
FAMILIES = ('T_and_or', 'T_or_and', 'T_and_or_and', 'T_nested_and', 'T_nested_or')
ROUTES = ('base_query2048', 'circuit1024', 'hn_query1024', 'query_only_hn1024')
LEGACY_LOCK = 'logs/analysis/2026-11-21/lean-query-only-hn-probe-id-matrix-v1/IMPLEMENTATION_LOCK.json'


def now():
    return datetime.now(timezone.utc).isoformat()


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def rows(path):
    return [json.loads(line) for line in Path(path).read_text(encoding='utf-8').splitlines() if line]


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf-8') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')


def load_tensor(path):
    return torch.load(path, map_location='cpu', weights_only=True, mmap=True)


def root_path(value):
    path = (ROOT / value).resolve()
    if not path.is_relative_to(ROOT.resolve()):
        raise ValueError('Artifact path escapes repository: ' + str(value))
    if path.name == 'final_test.pt':
        raise ValueError('Revoked evaluation bank is forbidden')
    return path


def load_config(path):
    cfg = read(path)
    if cfg['format_version'] != 1 or tuple(cfg['routes']) != ROUTES:
        raise ValueError('Unexpected four-way configuration')
    train = cfg['training']
    required = {'rows': 1200000, 'rows_per_atomic_task': 300000, 'epochs': 1,
                'selection': 'fixed_final_step', 'query_treatment': 'original',
                'loss': 'mse_one_hot', 'optimizer': 'AdamW', 'encoder_updates': 0,
                'evaluation_before_freeze': False, 'shuffle': True}
    for key, value in required.items():
        if train.get(key) != value:
            raise ValueError('This frozen protocol requires training.' + key + '=' + str(value))
    if not 0 < train['encoder_batch_size'] <= train['batch_size']:
        raise ValueError('Invalid encoder microbatch size')
    if cfg['evaluation']['rows_per_family'] != 500 or cfg['evaluation']['composition_rows'] != 2500:
        raise ValueError('Only the approved complete five-by-500 composition bank is allowed')
    if cfg['labels']['rule'] != 'target_and_plus_proof_or_v2':
        raise ValueError('Corrected And labels required')
    for group in [cfg['data'], {'control': cfg['control']['source_manifest']}]:
        for record in group.values():
            root_path(record['path'])
    for key in ['output_root', 'log_root']:
        root_path(cfg[key])
    return cfg


def source_records(cfg):
    manifest = read(root_path(cfg['control']['source_manifest']['path']))
    result = []
    for item in manifest['per_question_sources']:
        allowed = cfg['evaluation'][item['domain'] + '_conditions']
        if item['condition'] in allowed:
            result.append(item)
    return result


def own_sources(config_path):
    directory = Path(__file__).parent
    return [Path(config_path).resolve(), *sorted(directory.glob('*.py')),
            directory.parent / 'run_four_way_token_probe.py']


def preflight(config_path, cfg):
    log = root_path(cfg['log_root'])
    if (log / 'preflight.json').exists():
        verify_preflight(config_path, cfg)
        return read(log / 'preflight.json')
    lock = read(ROOT / LEGACY_LOCK)
    artifacts = {}
    # Frozen old helper/checkpoint/cache identities, including all real model weights.
    for group in ('sources', 'inputs'):
        for path, digest in lock[group].items():
            # Old clean atomic inputs are not needed for this experiment.
            if 'lean_atomic_probe_id_matrix_v1_20261121/clean_query_bank.pt' in path:
                continue
            artifacts[path] = digest
    for record in [*cfg['data'].values(), cfg['control']['source_manifest']]:
        artifacts[record['path']] = record['sha256']
    manifest = read(root_path(cfg['control']['source_manifest']['path']))
    base_identity_path = 'logs/analysis/2026-11-18/lean-base-query-atomic-only-probe-v1/identity.json'
    base_record = next(r for r in manifest['source_artifacts'] if r['path'] == base_identity_path)
    if sha(root_path(base_identity_path)) != base_record['sha256']:
        raise ValueError('Frozen Base representation identity changed')
    artifacts[base_identity_path] = base_record['sha256']
    base_identity = read(root_path(base_identity_path))
    artifacts.update(base_identity['source_sha256'])
    artifacts[base_identity['base_config']['path']] = base_identity['base_config']['sha256']
    # Circuit HN checkpoint and native Base weights/config are recorded by legacy loaders.
    hn = 'outputs/s5_lean_circuit/hypernet_ab_formal_1p2m_seed42/formal_lr_gate_r4a8/a_p03/'
    base = 'outputs/s5_lean_circuit/atomic_only_dense_v1/dense-bgoff_attempt_20260820_200505/'
    artifacts.update({hn + 'checkpoints/model_step0015258.pt': 'd948792b7206d36f1a06697428fcb928aefaf7db120558529fae3c38ab957cde',
                      hn + 'resolved_config.json': '4ea0754f6fdf63612b1878180513f94c7b777cc9bc20a4aefa043c030a451be1',
                      base + 'best_model.pt': 'ee2a7efdf77262ee1fbdf7d2080f930c2bedc27e45b7712537fba964ee05931c'})
    verified = []
    for path, digest in artifacts.items():
        target = root_path(path)
        if not target.is_file() or sha(target) != digest:
            raise ValueError('Missing or changed frozen artifact: ' + path)
        stat = target.stat()
        verified.append({'path': path, 'sha256': digest, 'bytes': stat.st_size, 'mtime_ns': stat.st_mtime_ns})
    for item in source_records(cfg):
        if sha(root_path(item['path'])) != item['sha256']:
            raise ValueError('Legacy prediction mismatch: ' + item['path'])
    cache = load_tensor(root_path(cfg['data']['train_cache']['path']))
    if tuple(cache['query_ids'].shape) != (1200000, 128):
        raise ValueError('Original complete atomic training cache required')
    if torch.bincount(cache['task_ids'].long(), minlength=4).tolist() != [300000] * 4:
        raise ValueError('Training class counts changed')
    if tuple(cache['vectors'].shape) != (1200000, 397):
        raise ValueError('Missing original circuit vectors')
    del cache
    data = evaluation_data(cfg)
    own = {p.relative_to(ROOT).as_posix(): sha(p) for p in own_sources(config_path)}
    # Base representation helper is not in the Query-only implementation lock.
    for name in ['base_query_probe_representation.py', 'query_atomic_only_probe.py',
                 'analyze_atomic_encoder_task_readout.py']:
        p = ROOT / 's5_lean_circuit/scripts' / name
        own[p.relative_to(ROOT).as_posix()] = sha(p)
    record = {'status': 'passed', 'created_utc': now(), 'config_sha256': sha(config_path),
              'source_commit': cfg['source_commit'], 'own_sources': own, 'artifacts': verified,
              'training_rows': 1200000, 'evaluation_rows': {k: len(v['rows']) for k, v in data.items()},
              'evaluation_metadata_read_for_integrity_only': True,
              'model_updates': 0, 'new_graph_extraction': False,
              'control_sources': source_records(cfg)}
    save(log / 'preflight.json', record)
    return record


def verify_preflight(config_path, cfg):
    record = read(root_path(cfg['log_root']) / 'preflight.json')
    if record['status'] != 'passed' or record['config_sha256'] != sha(config_path):
        raise ValueError('Preflight/config mismatch')
    for path, digest in record['own_sources'].items():
        if sha(root_path(path)) != digest:
            raise ValueError('Code changed after preflight: ' + path)
    for item in record['artifacts']:
        stat = root_path(item['path']).stat()
        if stat.st_size != item['bytes'] or stat.st_mtime_ns != item['mtime_ns']:
            raise ValueError('Input changed after preflight; do not reuse this run: ' + item['path'])
        if item['path'].endswith(('.py', '.sh', '.yml', '.yaml')) and sha(root_path(item['path'])) != item['sha256']:
            raise ValueError('Frozen implementation changed: ' + item['path'])
    return record


def event_list(tensor):
    if tensor.ndim != 2 or tensor.shape[1] != 32 or tensor.dtype != torch.uint8:
        raise ValueError('Invalid event key tensor')
    return [bytes(x).hex() for x in tensor.tolist()]


def evaluation_data(cfg):
    result = {}
    for domain in ('atomic', 'composition'):
        records = rows(root_path(cfg['data'][domain + '_rows']['path']))
        families = TASKS if domain == 'atomic' else FAMILIES
        if Counter(r['task_type'] for r in records) != Counter({t: 500 for t in families}):
            raise ValueError('Incomplete evaluation families')
        bank = load_tensor(root_path(cfg['data'][domain + '_bank']['path']))
        start = 0 if domain == 'atomic' else 2000
        indices = torch.arange(start, start + len(records))
        events = event_list(bank['event_key'][indices])
        if events != [r['event_key'] for r in records] or indices.tolist() != [r['bank_index'] for r in records]:
            raise ValueError('Original query row alignment mismatch')
        inputs = {k: bank[k][indices] for k in ('query_ids', 'query_lengths', 'vectors')}
        result[domain + '_original'] = {'inputs': inputs, 'rows': records, 'events': events}
        if domain == 'composition':
            clean = load_tensor(root_path(cfg['data']['clean_composition_bank']['path']))
            circuits = load_tensor(root_path(cfg['data']['clean_composition_circuits']['path']))
            if event_list(clean['parent_event_key']) != events or not torch.equal(clean['bank_indices'], indices):
                raise ValueError('Clean query parent alignment mismatch')
            if not torch.equal(clean['clean_query_sha256'], circuits['clean_query_sha256']) or not torch.equal(clean['bank_indices'], circuits['bank_indices']):
                raise ValueError('Clean circuit/query pairing mismatch')
            if tuple(circuits['vectors'].shape) != (2500, 397):
                raise ValueError('Clean circuit cache is incomplete')
            if circuits['source_query_bank_sha256'] != cfg['data']['clean_composition_bank']['sha256']:
                raise ValueError('Clean circuit provenance differs')
            inputs = {k: clean[k] for k in ('query_ids', 'query_lengths')}
            inputs['vectors'] = circuits['vectors']
            result['composition_clean'] = {'inputs': inputs, 'rows': records, 'events': events}
    return result


def select_batch(bank, indices, route):
    keys = ('vectors',) if route == 'circuit1024' else ('query_ids', 'query_lengths')
    return {key: bank[key][indices] for key in keys}


def make_head(cfg, width, device):
    torch.manual_seed(cfg['training']['seed'])
    spec = cfg['probe']
    return TokenAttentionProbe(native_width=width, common_width=spec['common_width'],
                               probe_width=spec['probe_width'], hidden_width=spec['hidden_width'],
                               num_classes=spec['num_classes']).to(device)


def train_head(encoder, head, bank, cfg, log):
    opt = cfg['training']
    optimizer = torch.optim.AdamW(head.parameters(), lr=opt['learning_rate'], weight_decay=opt['weight_decay'])
    generator = torch.Generator(device='cpu').manual_seed(opt['seed'])
    head.train()
    n, steps = opt['rows'], 0
    start = time.monotonic()
    with (log / 'training.jsonl').open('x', encoding='utf-8') as stream:
        for epoch in range(opt['epochs']):
            order = torch.randperm(n, generator=generator)
            for begin in range(0, n, opt['batch_size']):
                indices = order[begin:begin + opt['batch_size']]
                optimizer.zero_grad(set_to_none=True)
                loss_value = 0.0
                for offset in range(0, len(indices), opt['encoder_batch_size']):
                    micro = indices[offset:offset + opt['encoder_batch_size']]
                    hidden, mask = encoder.encode(select_batch(bank, micro, encoder.route))
                    scores = head(hidden, mask)
                    labels = F.one_hot(bank['task_ids'][micro].long(), 4).to(device=scores.device, dtype=scores.dtype)
                    loss = F.mse_loss(scores, labels) * len(micro) / len(indices)
                    if not torch.isfinite(loss):
                        raise ValueError('Nonfinite training loss')
                    loss.backward()
                    loss_value += float(loss.detach())
                if any(p.grad is not None for p in encoder.model.parameters()):
                    raise ValueError('Frozen encoder received gradients')
                optimizer.step()
                steps += 1
                if steps % 100 == 0 or begin + len(indices) == n:
                    record = {'step': steps, 'epoch': epoch + 1, 'rows': begin + len(indices),
                              'loss': loss_value, 'seconds': time.monotonic() - start}
                    stream.write(json.dumps(record) + '\n'); stream.flush()
                    print(json.dumps(record), flush=True)
    if steps != opt['epochs'] * math.ceil(n / opt['batch_size']):
        raise ValueError('Training budget was not completed')
    return steps


def predict(encoder, head, bank, batch_size):
    n = len(next(iter(bank.values())))
    values = []
    with torch.no_grad():
        for begin in range(0, n, batch_size):
            indices = torch.arange(begin, min(begin + batch_size, n))
            hidden, mask = encoder.encode(select_batch(bank, indices, encoder.route))
            values.append(head(hidden, mask).double().cpu())
    scores = torch.cat(values)
    if tuple(scores.shape) != (n, 4) or not torch.isfinite(scores).all():
        raise ValueError('Invalid predictions')
    return scores


def aggregate(records):
    grouped = defaultdict(list)
    for row in records:
        for task in (row['task_type'], 'ALL'):
            grouped[row['domain'], row['condition'], row['method'], row['probe'], task].append(row)
    result = []
    for key, group in grouped.items():
        scored = [r for r in group if r['status'] == 'scored']
        result.append(dict(zip(('domain', 'condition', 'method', 'probe', 'task_type'), key)) |
                      {'total': len(group), 'n': len(scored), 'correct': sum(r['success'] for r in scored),
                       'pct': 100 * sum(r['success'] for r in scored) / len(scored) if scored else None,
                       'auc': sum(r['auc'] for r in scored) / len(scored) if scored else None})
    return result


def score_route(cfg, route, predictions, data, log):
    lookup = {domain: {r['event_key']: (i, r) for i, r in enumerate(data[domain + '_original']['rows'])}
              for domain in ('atomic', 'composition')}
    result = []
    for source in source_records(cfg):
        if source['probe'] != route:
            continue
        path = root_path(source['path'])
        if sha(path) != source['sha256']:
            raise ValueError('Historical control changed')
        condition, domain = source['condition'], source['domain']
        candidate = range(3) if condition in ('original_no_identity', 'clean_no_identity') else range(4)
        view = domain + ('_clean' if condition.startswith('clean_') else '_original')
        for old in rows(path):
            index, question = lookup[domain][old['event_key']]
            if old['bank_index'] != question['bank_index'] or old['task_type'] != question['task_type']:
                raise ValueError('Historical control row mismatch')
            query = question['query'] if domain == 'atomic' else question['parent_query']
            labels = corrected_labels(query, old['labels'])
            for method, scores in [('mean_linear_ridge', old['scores']), ('token_attention', predictions[view][index].tolist())]:
                result.append({'domain': domain, 'condition': condition, 'method': method, 'probe': route,
                               'task_type': old['task_type'], 'event_key': old['event_key'],
                               'bank_index': old['bank_index'], 'branch': old['branch'], 'labels': labels,
                               'scores': scores, **ranking_metrics(scores, labels, candidate)})
    groups = defaultdict(list)
    for row in result:
        groups[row['domain'], row['condition'], row['method'], row['task_type']].append(row)
    detail = log / 'per_question'; detail.mkdir()
    for key, group in groups.items():
        path = detail / ('__'.join(key) + '.jsonl')
        with path.open('x', encoding='utf-8') as stream:
            for row in group:
                stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + '\n')
        if path.stat().st_size >= 5000000:
            raise ValueError('Split log size limit exceeded')
    return aggregate(result)


def run(config_path, cfg, route):
    if route not in ROUTES:
        raise ValueError('Unknown route')
    preflight_record = verify_preflight(config_path, cfg)
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA is required for probe training')
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    log, out = root_path(cfg['log_root']) / route, root_path(cfg['output_root']) / route
    log.mkdir(parents=True, exist_ok=True); out.mkdir(parents=True, exist_ok=True)
    save(log / 'started.json', {'started_utc': now(), 'route': route, 'config_sha256': sha(config_path)})
    if (out / 'probe.pt').exists():
        raise FileExistsError('Existing probe must be preserved; this protocol has no automatic retries')
    encoder = FrozenEncoder(route, 'cuda')
    before = encoder.state_digest()
    head = make_head(cfg, encoder.width, 'cuda')
    bank = load_tensor(root_path(cfg['data']['train_cache']['path']))
    smoke_indices = torch.arange(cfg['training']['encoder_batch_size'])
    smoke_hidden, smoke_mask = encoder.encode(select_batch(bank, smoke_indices, route))
    smoke_scores = head(smoke_hidden, smoke_mask)
    smoke_labels = F.one_hot(bank['task_ids'][smoke_indices].long(), 4).to(smoke_scores)
    smoke_loss = F.mse_loss(smoke_scores, smoke_labels)
    smoke_loss.backward()
    if not torch.isfinite(smoke_loss) or any(p.grad is not None and not torch.isfinite(p.grad).all() for p in head.parameters()):
        raise ValueError('Actual-checkpoint smoke produced nonfinite loss/gradients')
    if any(p.grad is not None for p in encoder.model.parameters()):
        raise ValueError('Actual-checkpoint smoke leaked encoder gradients')
    head.zero_grad(set_to_none=True)
    save(log / 'smoke.json', {'status': 'passed', 'rows': len(smoke_indices),
                             'optimizer_updates': 0, 'encoder_updates': 0,
                             'loss': float(smoke_loss.detach()),
                             'tokens_shape': list(smoke_hidden.shape),
                             'peak_allocated_bytes': torch.cuda.max_memory_allocated()})
    del smoke_hidden, smoke_mask, smoke_scores, smoke_labels, smoke_loss
    steps = train_head(encoder, head, bank, cfg, log)
    del bank
    after_fit = encoder.state_digest()
    if after_fit != before:
        raise ValueError('Encoder was changed during probe fit')
    head.eval().requires_grad_(False)
    torch.save({'state_dict': head.state_dict(), 'config': cfg, 'route': route, 'encoder_identity': encoder.identity,
                'encoder_state_sha256': before, 'training_rows': 1200000, 'steps': steps}, out / 'probe.pt')
    frozen = {'frozen_utc': now(), 'probe_sha256': sha(out / 'probe.pt'), 'encoder_state_sha256': before,
              'evaluation_outcomes_read': False, 'encoder_updates': 0, 'steps': steps,
              'evaluation_outcomes_read_scope': 'Training worker fitting; preflight may read evaluation metadata for integrity checks only.',
              'parameter_count': sum(p.numel() for p in head.parameters()), 'native_width': encoder.width,
              'config_sha256': sha(config_path), 'encoder_identity': encoder.identity}
    save(log / 'fit_frozen.json', frozen)
    # Evaluation banks are opened for this worker only after final weights are frozen.
    data = evaluation_data(cfg)
    predictions = {name: predict(encoder, head, value['inputs'], cfg['training']['encoder_batch_size'])
                   for name, value in data.items()}
    if encoder.state_digest() != before or sha(out / 'probe.pt') != frozen['probe_sha256']:
        raise ValueError('Frozen weights changed during evaluation')
    torch.save({'scores': predictions, 'events': {k: v['events'] for k, v in data.items()},
                'probe_sha256': frozen['probe_sha256'], 'config_sha256': sha(config_path)}, out / 'predictions.pt')
    result = score_route(cfg, route, predictions, data, log)
    save(log / 'summary.json', {'route': route, 'results': result, 'config_sha256': sha(config_path)})
    save(log / 'complete.json', {'status': 'complete', 'completed_utc': now(), 'route': route,
         'config_sha256': sha(config_path), 'frozen': frozen, 'encoder_state_sha256_after': before,
         'forward_rows': sum(len(v['rows']) for v in data.values()), 'summary_sha256': sha(log / 'summary.json'),
         'predictions_sha256': sha(out / 'predictions.pt'), 'preflight_config': preflight_record['config_sha256']})


def report(config_path, cfg):
    verify_preflight(config_path, cfg)
    log = root_path(cfg['log_root'])
    result, receipts = [], []
    for route in ROUTES:
        complete = read(log / route / 'complete.json')
        if complete['status'] != 'complete' or complete['config_sha256'] != sha(config_path):
            raise ValueError('Incomplete/mismatched worker: ' + route)
        if sha(log / route / 'summary.json') != complete['summary_sha256']:
            raise ValueError('Worker summary changed')
        result.extend(read(log / route / 'summary.json')['results']); receipts.append(complete)
    if len({r['frozen']['parameter_count'] for r in receipts}) != 1:
        raise ValueError('Head parameter counts differ')
    expected_control = dict(zip(ROUTES, (399, 1048, 972, 1097)))
    for route in ROUTES:
        control = next(r for r in result if r['domain'] == 'composition'
                       and r['condition'] == 'clean_no_identity' and r['method'] == 'mean_linear_ridge'
                       and r['probe'] == route and r['task_type'] == 'ALL')
        if control['n'] != 2500 or control['correct'] != expected_control[route]:
            raise ValueError('Corrected historical baseline did not reproduce: ' + route)
    for row in result:
        expected_n = (2500 if row['domain'] == 'composition' else 2000) if row['task_type'] == 'ALL' else 500
        if row['n'] != expected_n or row['total'] != expected_n:
            raise ValueError('Evaluation denominator changed: ' + str(row))
    paired = []
    for route in ROUTES:
        by_key = {}
        for path in sorted((log / route / 'per_question').glob('*.jsonl')):
            for row in rows(path):
                key = row['domain'], row['condition'], row['task_type'], row['event_key']
                by_key.setdefault(key, {})[row['method']] = row
        groups = defaultdict(list)
        for key, pair in by_key.items():
            old, new = pair['mean_linear_ridge'], pair['token_attention']
            if old['labels'] != new['labels'] or old['status'] != new['status']:
                raise ValueError('Paired evaluation differs in labels or denominator')
            for task in (key[2], 'ALL'):
                groups[key[0], key[1], task].append((old, new))
        for key, pairs in groups.items():
            gained = sum(not a['success'] and b['success'] for a, b in pairs)
            lost = sum(a['success'] and not b['success'] for a, b in pairs)
            paired.append(dict(zip(('domain', 'condition', 'task_type'), key)) |
                          {'probe': route, 'n': len(pairs), 'new_only_success': gained,
                           'control_only_success': lost, 'delta_pp': 100 * (gained - lost) / len(pairs),
                           'auc_delta': sum(b['auc'] - a['auc'] for a, b in pairs) / len(pairs)})
    payload = {'status': 'complete', 'created_utc': now(), 'results': result, 'workers': receipts,
               'paired_comparisons': paired,
               'labels': cfg['labels'], 'config_sha256': sha(config_path),
               'limits': ['Exploratory existing banks; not fresh confirmation.',
                          'Attention head changes readout capacity and optimization as well as pooling.',
                          'Identical registered head count, but native widths/token counts differ.',
                          'Encoder training budgets differ; no causal training-transfer conclusion.']}
    save(log / 'summary.json', payload)
    names = {'base_query2048': 'Base', 'circuit1024': 'Circuit', 'hn_query1024': 'Circuit HN query', 'query_only_hn1024': 'Query-only query'}
    lines = ['# Four-way frozen token probes', '', 'Corrected target-And labels; Or directions follow reference proof. Existing mean+ridge controls are preserved.', '']
    for condition in cfg['evaluation']['composition_conditions']:
        for method in ('mean_linear_ridge', 'token_attention'):
            lines += ['## ' + condition + ' / ' + method, '', '| Task | ' + ' | '.join(names.values()) + ' |', '|---|---:|---:|---:|---:|']
            for task in (*FAMILIES, 'ALL'):
                cells = []
                for route in ROUTES:
                    row = next(r for r in result if r['domain'] == 'composition' and r['condition'] == condition and r['method'] == method and r['probe'] == route and r['task_type'] == task)
                    cells.append(f"{row['correct']}/{row['n']} ({row['pct']:.2f}%); AUC {row['auc']:.4f}" if row['n'] else 'N/A')
                lines.append('| ' + ' | '.join((task, *cells)) + ' |')
            lines.append('')
    with (log / 'summary.md').open('x', encoding='utf-8') as stream:
        stream.write('\n'.join(lines) + '\n')
    save(log / 'COMPLETE.json', {'status': 'complete', 'summary_sha256': sha(log / 'summary.json'), 'workers': 4})
