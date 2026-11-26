#!/usr/bin/env python3
"""Extract canonical generated-trajectory circuits for input-only atomic queries.

A worker loads the frozen base and MLP transcoders.
internal attribution gradients are enabled, but no model parameter is trained.
Smoke and full shards share identity-keyed event artifacts and restart checks.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import datetime as dt
import fcntl
import gc
import hashlib
import json
import os
from pathlib import Path
import resource
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_LOG = ROOT / 'logs/analysis/2026-11-17/lean-circuit-atomic-similarity-v1'
DEFAULT_OUTPUT = ROOT / 'outputs/s5_lean_circuit/lean_circuit_atomic_similarity_v1_20261117'


def stable_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                     separators=(',', ':')).encode()).hexdigest()


def file_sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text())


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + f'.tmp.{os.getpid()}')
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    os.replace(temp, path)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read_protocol(protocol_path, identity_path, *, verify_assets=False):
    protocol, identity = read_json(protocol_path), read_json(identity_path)
    declared = protocol['identity_sha256']
    require(stable_hash({k: v for k, v in protocol.items() if k != 'identity_sha256'}) == declared,
            'Extraction protocol hash changed')
    require(identity['extraction']['identity_sha256'] == declared, 'Candidate extraction identity mismatch')
    source = protocol['source_model']
    require(identity['base'] == {k: v for k, v in source.items() if k not in ('tokenizer', 'tokenizer_sha256')},
            'Candidate Base identity differs from extraction source')
    require(identity['tokenizer'] == {'path': source['tokenizer'], 'sha256': source['tokenizer_sha256']},
            'Candidate tokenizer identity differs from extraction source')
    require(identity['dictionary'] == protocol['dictionary'], 'Candidate dictionary identity differs')
    require(protocol['source_model']['checkpoint_sha256'] ==
            'ee2a7efdf77262ee1fbdf7d2080f930c2bedc27e45b7712537fba964ee05931c', 'Wrong native Base')
    require(protocol['source_model']['checkpoint_step'] == 38000, 'Wrong native Base step')
    require(protocol['feature_retention'] is None, 'No feature intervention is allowed')
    require(protocol['graph_policy'] == {
        'operator': 'joint_signed_full_output', 'backend': 'streaming_cached_exact',
        'feature_budget': 512, 'node_threshold': 0.9, 'edge_threshold': 0.99,
        'queue_size': 640, 'row_batch_size': 64}, 'Extraction policy differs from formal Suite 1')
    require(protocol['generation'] == {
        'decoding': 'greedy_argmax', 'target': 'self_generated_full_output',
        'max_new_tokens': 50, 'context_length': 128,
        'stop_markers': ['proof complete', 'no goals']}, 'Generation protocol changed')
    for path, expected in protocol['source_file_sha256'].items():
        require(file_sha(ROOT / path) == expected, 'Frozen extraction source changed: ' + path)
    require(file_sha(ROOT / protocol['circuit_config']) == protocol['circuit_config_sha256'],
            'Frozen circuit config changed')
    if verify_assets:
        source = protocol['source_model']
        for path_key, sha_key in [('checkpoint', 'checkpoint_sha256'), ('tokenizer', 'tokenizer_sha256'),
                                  ('architecture_config', 'architecture_config_sha256')]:
            require(file_sha(ROOT / source[path_key]) == source[sha_key], 'Source asset changed: ' + path_key)
        for layer in protocol['dictionary']['layers']:
            path = ROOT / layer['path']
            require(path.stat().st_size == layer['bytes'] and file_sha(path) == layer['sha256'],
                    'MLP Transcoder asset changed: ' + str(layer['layer']))
    return protocol, identity


def load_candidates(path, identity, shard_index=0, shard_count=1):
    require(0 <= shard_index < shard_count, 'Invalid shard index/count')
    rows = [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]
    seen_ids, seen_keys = set(), set()
    for row in rows:
        index, key = row['unique_index'], row['cache_key']
        require(type(index) is int and index >= 0 and index not in seen_ids, 'Duplicate/invalid unique index')
        require(key not in seen_keys, 'Duplicate candidate cache key')
        require(row['query_sha256'] == hashlib.sha256(row['query'].encode()).hexdigest(), 'Query SHA changed')
        require(row['candidate_query_id'] == row['query_sha256'], 'Candidate query ID changed')
        require(row['extraction_identity_sha256'] == stable_hash(identity), 'Candidate identity hash changed')
        require(key == stable_hash({'query': row['query'], 'identity': identity}), 'Candidate cache key changed')
        seen_ids.add(index)
        seen_keys.add(key)
    return [row for row in rows if row['unique_index'] % shard_count == shard_index]


def verify_complete(event_dir, row, protocol):
    marker = Path(event_dir) / 'complete.json'
    if not marker.exists():
        return None
    result = read_json(marker)
    for key in ('candidate_query_id', 'query_sha256', 'cache_key', 'extraction_identity_sha256'):
        require(result[key] == row[key], 'Completed candidate identity changed: ' + key)
    require(result['protocol_identity_sha256'] == protocol['identity_sha256'], 'Completed protocol changed')
    require(result['status'] == 'complete', 'Invalid completion status')
    require(set(result['files']) >= {'trace.json', 'vectors.pt', 'graph_v1.safetensors', 'graph_v2.safetensors'},
            'Completion omits required trace/vector/full retained-graph artifacts')
    for name, record in result['files'].items():
        require(Path(name).name == name, 'Unsafe completion artifact path')
        path = Path(event_dir) / name
        require(path.is_file() and path.stat().st_size == record['bytes'] and file_sha(path) == record['sha256'],
                'Completed candidate artifact changed: ' + name)
    return result


def validate_vectors(raw, processed):
    import torch
    for name, vector in (('raw_vector397', raw), ('vector397', processed)):
        require(tuple(vector.shape) == (397,), name + ' must have shape [397]')
        require(bool(torch.isfinite(vector).all()), name + ' contains nonfinite values')
        require(float(torch.linalg.vector_norm(vector.double())) > 0, name + ' has zero L2 norm')


def run(args):
    started = time.monotonic()
    log_dir, output_dir = Path(args.log_dir).resolve(), Path(args.output_dir).resolve()
    require('logs' in log_dir.parts and 'outputs' in output_dir.parts, 'Logs and large artifacts need canonical roots')
    protocol, identity = read_protocol(args.protocol, args.identity, verify_assets=True)
    rows = load_candidates(args.candidates, identity, args.shard_index, args.shard_count)
    attempt = dt.datetime.now(dt.timezone.utc).strftime('%Y%m%dT%H%M%S') + f'-{os.getpid()}'
    receipt_path = log_dir / 'extraction_workers' / f'shard-{args.shard_index:03d}-{attempt}.json'
    progress_path = receipt_path.with_suffix('.jsonl')
    receipt = {'status': 'initializing', 'started_utc': dt.datetime.now(dt.timezone.utc).isoformat(),
               'protocol_identity_sha256': protocol['identity_sha256'], 'identity_sha256': stable_hash(identity),
               'candidate_file': str(Path(args.candidates).resolve()), 'candidate_file_sha256': file_sha(args.candidates),
               'shard_index': args.shard_index, 'shard_count': args.shard_count, 'assigned_count': len(rows),
               'worker_sha256': file_sha(__file__), 'pid': os.getpid(), 'hostname': os.uname().nodename,
               'generated': 0, 'extracted': 0, 'reused': 0, 'failed': 0, 'busy': 0, 'pending': len(rows),
               'events': [], 'failures': [], 'training_steps': 0, 'base_weight_updates': 0,
               'generation_uses_gold': False, 'teacher_forced_gold': False,
               'internal_attribution_gradients': True, 'model_weights_require_grad': False}
    atomic_json(receipt_path, receipt)
    pending = []
    for row in rows:
        event_dir = output_dir / 'events' / row['cache_key']
        if verify_complete(event_dir, row, protocol):
            receipt['reused'] += 1
        else:
            pending.append(row)
    receipt['pending'] = len(pending)
    if args.preflight_only or not pending:
        receipt['status'] = 'preflight_passed' if args.preflight_only else 'complete'
        receipt['wall_seconds'] = time.monotonic() - started
        atomic_json(receipt_path, receipt)
        print(json.dumps(receipt), flush=True)
        return 0
    sys.path[:0] = [str(ROOT / 's5_lean_circuit/src'), str(ROOT / 's3_circuit/src'),
                    str(ROOT / 's2_hypernet/src'), str(ROOT / 'temp_s')]
    import torch
    from s5_lean_circuit.config import load_config
    from s5_lean_circuit.circuit.model_io import load_frozen_source_model, load_tokenizer
    from s5_lean_circuit.circuit.bank import load_canonical_transcoder_bank
    from s5_lean_circuit.circuit.traces import CircuitTrace, greedy_generated_token_ids, trace_event_key
    from s5_lean_circuit.circuit.pilot import prepare_cached_graph, materialize_prepared_graph
    from s5_lean_circuit.circuit.payload import circuit_artifact_id, write_payload_chunk
    from s5_lean_circuit.circuit.payload_v2 import payload_v2_from_v1, write_payload_v2_shard, semantic_sha256
    from s5_lean_circuit.circuit.descriptor_v2_397 import vectorize_payload_v2_397, fixed_log1p_397
    from s5_lean_circuit.circuit.identity import model_artifact_identity
    torch.set_num_threads(args.cpu_threads)
    torch.set_num_interop_threads(1)
    torch.manual_seed(42)
    device = torch.device('cuda:0')
    require(torch.cuda.is_available(), 'This worker requires a CUDA device')
    source = protocol['source_model']
    model, model_identity = load_frozen_source_model(
        load_config(ROOT / source['architecture_config']), ROOT / source['checkpoint'], device=device,
        expected_checkpoint_config_sha256=source['checkpoint_config_sha256'],
        expected_checkpoint_sha256=source['checkpoint_sha256'],
        expected_architecture_config_sha256=source['architecture_config_sha256'],
        verified_checkpoint_sha256=source['checkpoint_sha256'])
    tokenizer = load_tokenizer(ROOT / source['tokenizer'], expected_sha256=source['tokenizer_sha256'])
    bank, bank_sha = load_canonical_transcoder_bank(
        [ROOT / item['path'] for item in protocol['dictionary']['layers']], device=device,
        expected_config_sha256=protocol['dictionary']['selection_config_sha256'],
        verified_sha256={item['layer']: item['sha256'] for item in protocol['dictionary']['layers']})
    for module in (model, *bank):
        module.requires_grad_(False).eval()
    require(all(not p.requires_grad for m in (model, *bank) for p in m.parameters()), 'Model weights are not frozen')
    policy = protocol['graph_policy']
    artifact_model = model_artifact_identity(model_identity)
    archive_identity = {**policy, 'format': 'atomic_similarity_generated_trajectory_v1',
                        'model_identity': artifact_model, 'tokenizer_sha256': source['tokenizer_sha256'],
                        'n_layers': 8, 'd_feature': 16384, 'transcoder_bank_sha256': bank_sha,
                        'protocol_identity_sha256': protocol['identity_sha256'],
                        'extraction_identity_sha256': stable_hash(identity), 'payload_schema_version': 2}
    archive_sha = semantic_sha256(archive_identity)
    receipt.update(status='extracting', model_identity=artifact_model, transcoder_bank_sha256=bank_sha,
                   torch=torch.__version__,
                   tf32_matmul=torch.backends.cuda.matmul.allow_tf32, archive_identity_sha256=archive_sha,
                   initialization_seconds=time.monotonic()-started)
    atomic_json(receipt_path, receipt)
    for row in pending:
        if args.max_seconds and time.monotonic() - started >= args.max_seconds:
            break
        event_dir = output_dir / 'events' / row['cache_key']
        event_dir.mkdir(parents=True, exist_ok=True)
        lock = (event_dir / '.lock').open('a')
        try:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            lock.close()
            receipt['busy'] += 1
            continue
        event_started = time.monotonic()
        prepared = extracted = graph = None
        try:
            if verify_complete(event_dir, row, protocol):
                receipt['reused'] += 1
                receipt['pending'] -= 1
                continue
            prompt = tuple(int(x) for x in tokenizer.encode(row['query']))
            # A neutral legal Trace metadata tag is required by the existing
            # serializer. Neither this tag nor the derived seed enters model
            # forward, feature selection, vectorization, or generation.
            trace_task, trace_seed = 'S_id', int(row['query_sha256'][:15], 16)
            generated = greedy_generated_token_ids(model, tokenizer, prompt_token_ids=prompt,
                device=device, context_length=128, max_new_tokens=50,
                stop_markers=('proof complete', 'no goals'))
            trace = CircuitTrace(event_key=trace_event_key(task_type=trace_task, seed=trace_seed,
                prompt_token_ids=prompt, generated_token_ids=generated), task_type=trace_task,
                seed=trace_seed, prompt_token_ids=prompt, generated_token_ids=generated)
            trace.validate(context_length=128)
            atomic_json(event_dir / 'trace.json', {**asdict(trace), 'query_sha256': row['query_sha256'],
                'generated_text': tokenizer.decode(list(generated)), 'generation_seconds': time.monotonic()-event_started,
                'trace_task_metadata_only': True, 'correctness_filter': False})
            receipt['generated'] += 1
            prepared = prepare_cached_graph(model, bank, trace, device=device,
                feature_budget=512, queue_size=640, row_batch_size=64,
                max_cache_bytes=protocol['graph_limits']['max_cache_bytes'])
            require(prepared.context.feature_retention is None, 'Unexpected feature retention in native extraction')
            artifact_id = circuit_artifact_id(event_key=trace.event_key, model_identity=artifact_model,
                tokenizer_sha256=source['tokenizer_sha256'], transcoder_bank_sha256=bank_sha,
                graph_policy=policy, code_identity=stable_hash(protocol['source_file_sha256']))
            extracted = materialize_prepared_graph(prepared, event_key=trace.event_key,
                artifact_id=artifact_id, condition_length=trace.condition_length, d_feature=16384,
                node_threshold=0.9, edge_threshold=0.99,
                max_selected_dense_bytes=protocol['graph_limits']['max_selected_dense_bytes'])
            graph = payload_v2_from_v1(extracted.payload, archive_identity_sha256=archive_sha,
                                       n_layers=8, d_feature=16384)
            write_payload_chunk(event_dir / 'graph_v1.safetensors', [extracted.payload],
                                archive_identity=archive_identity)
            write_payload_v2_shard(event_dir / 'graph_v2.safetensors', [graph], archive_identity=archive_identity)
            raw = vectorize_payload_v2_397(graph, n_layers=8, d_feature=16384,
                                           archive_identity_sha256=archive_sha)
            processed = fixed_log1p_397(raw)
            validate_vectors(raw, processed)
            torch.save({'raw_vector397': raw, 'vector397': processed,
                'input_treatment': 'fixed_log1p_v1', 'query_ids': torch.tensor(prompt, dtype=torch.long),
                'query_length': len(prompt), 'generated_ids': torch.tensor(generated, dtype=torch.long),
                'selected_features': extracted.selected_features, 'retained_features': extracted.retained_features,
                'selected_feature_influence': extracted.selected_feature_influence,
                'event_key': trace.event_key, 'cache_key': row['cache_key'],
                'extraction_identity_sha256': stable_hash(identity)}, event_dir / 'vectors.pt')
            torch.cuda.synchronize(device)
            result = {key: row[key] for key in ('unique_index', 'candidate_query_id', 'query_sha256',
                                               'cache_key', 'extraction_identity_sha256')}
            result.update(status='complete', protocol_identity_sha256=protocol['identity_sha256'],
                event_key=trace.event_key, total_seconds=time.monotonic()-event_started,
                feature_count=graph.feature_count, edge_count=graph.edge_count,
                generated_token_count=len(generated), prompt_token_count=len(prompt),
                cache_stats=dict(extracted.cache_stats), files={
                    p.name: {'bytes': p.stat().st_size, 'sha256': file_sha(p)}
                    for p in sorted(event_dir.iterdir()) if p.is_file() and p.name != '.lock'
                    and not p.name.startswith('complete') and '.tmp.' not in p.name})
            atomic_json(event_dir / 'complete.json', result)
            verify_complete(event_dir, row, protocol)
            receipt['extracted'] += 1
            receipt['pending'] -= 1
            compact = {k: result[k] for k in ('unique_index', 'cache_key', 'total_seconds',
                                              'feature_count', 'edge_count')}
            receipt['events'].append(compact)
            with progress_path.open('a') as stream:
                stream.write(json.dumps({'status': 'complete', **compact})+'\n')
            print(json.dumps({'status': 'complete', **compact}), flush=True)
        except Exception as exc:
            error = {'unique_index': row['unique_index'], 'cache_key': row['cache_key'],
                     'error': repr(exc), 'traceback': traceback.format_exc(),
                     'seconds': time.monotonic()-event_started, 'correctness_filter': False}
            receipt['failed'] += 1
            receipt['failures'].append(error)
            atomic_json(log_dir / 'extraction_failures' / f'{row["cache_key"]}-{attempt}.json', error)
            print(json.dumps({'status': 'failed', **error}), flush=True)
        finally:
            prepared = extracted = graph = None
            gc.collect()
            torch.cuda.empty_cache()
            fcntl.flock(lock.fileno(), fcntl.LOCK_UN)
            lock.close()
            receipt['wall_seconds'] = time.monotonic()-started
            atomic_json(receipt_path, receipt)
    receipt.update(status='complete' if receipt['pending'] == 0 else 'partial',
        ended_utc=dt.datetime.now(dt.timezone.utc).isoformat(), wall_seconds=time.monotonic()-started,
        peak_rss_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024)
    atomic_json(receipt_path, receipt)
    print(json.dumps(receipt), flush=True)
    return 0 if receipt['status'] == 'complete' else 2


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--candidates', type=Path, required=True)
    parser.add_argument('--protocol', type=Path, default=DEFAULT_LOG/'extraction_protocol.json')
    parser.add_argument('--identity', type=Path, default=DEFAULT_LOG/'candidate_extraction_identity.json')
    parser.add_argument('--output-dir', type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument('--log-dir', type=Path, default=DEFAULT_LOG)
    parser.add_argument('--shard-index', type=int, default=0)
    parser.add_argument('--shard-count', type=int, default=1)
    parser.add_argument('--cpu-threads', type=int, default=1)
    parser.add_argument('--max-seconds', type=int, default=0)
    parser.add_argument('--preflight-only', action='store_true')
    return run(parser.parse_args())


if __name__ == '__main__':
    raise SystemExit(main())
