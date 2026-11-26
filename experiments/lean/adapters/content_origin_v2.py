"""New matched/permuted/constant training from an identical untrained origin.

This module deliberately leaves historical formal runners and caches untouched.
It reuses their batch stream, losses, optimizer schedule and checkpoint format.
Only the descriptor presented to the existing trainable Circuit branch changes.
"""
from __future__ import annotations

from collections import Counter
from contextlib import nullcontext
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import time

import torch

FORMAT = 's5_content_origin_plan_v2'
VARIANTS = ('matched', 'permuted', 'mean')
STOP = 15258
VAL_STEPS = (100, 7629, STOP)
SAVE_STEPS = (0, 100, 1000, 7629, STOP)
ATOMIC = ('S_and', 'S_or_l', 'S_or_r', 'S_id')


def optimizer_learning_rates(optimizer):
    """Read the actual optimizer API: param_groups is a list, not a method."""
    return [float(group['lr']) for group in optimizer.param_groups]


def now():
    return datetime.now(timezone.utc).isoformat()


def digest(t):
    t = t.detach().cpu().contiguous()
    return hashlib.sha256(t.reshape(-1).view(torch.uint8).numpy().tobytes()).hexdigest()


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b''): h.update(chunk)
    return h.hexdigest()


def file(path):
    p = Path(path).resolve()
    return dict(path=str(p), bytes=p.stat().st_size, sha256=sha(p))


def write(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x') as f:
        json.dump(value, f, indent=2, sort_keys=True, allow_nan=False); f.write('\n')


def verify(records):
    for key, item in records.items():
        p = Path(item['path'])
        if p.stat().st_size != item['bytes'] or sha(p) != item['sha256']:
            raise ValueError('Frozen input changed: ' + key)


def training_mean(vectors, chunk=50000):
    if vectors.ndim != 2 or vectors.shape[1] != 397 or vectors.dtype != torch.float32:
        raise ValueError('Expected unique training already-treated FP32 Circuit397')
    total = torch.zeros(397, dtype=torch.float64)
    for part in vectors.split(chunk):
        if not bool(torch.isfinite(part).all()): raise ValueError('Nonfinite training descriptor')
        total += part.double().sum(0)
    return (total / len(vectors)).float()


def derangement(size, *, seed, epoch=0):
    """One deterministic random N-cycle; local generator never changes model RNG."""
    if size < 2: raise ValueError('Need at least two examples')
    g = torch.Generator(device='cpu').manual_seed(int(seed) + int(epoch) * 1000003)
    order = torch.randperm(size, generator=g)
    sources = torch.empty(size, dtype=torch.long)
    sources[order] = order.roll(-1)
    if bool((sources == torch.arange(size)).any()) or torch.unique(sources).numel() != size:
        raise ValueError('Not a one-to-one derangement')
    return sources


def treated_vectors(vectors, indices, mode, mean, sources=None):
    index = torch.as_tensor(indices, dtype=torch.long, device='cpu')
    if mode == 'matched': return vectors.index_select(0, index)
    if mode == 'mean': return mean.unsqueeze(0).expand(len(index), -1).clone()
    if mode == 'permuted' and sources is not None:
        return vectors.index_select(0, sources.index_select(0, index))
    raise ValueError('Unknown treatment or missing permutation')


def repair_same_query_donors(sources, query_ids, query_lengths):
    """Deterministically enforce different prompt donors using token IDs only.

    Preserve bijection; never inspect task labels, correctness or model output.
    No random redraws: scan recipients and swap with the first legal recipient.
    """
    result = sources.clone(); size = len(result)
    def same(a, b):
        return bool(query_lengths[a] == query_lengths[b]) and torch.equal(query_ids[a], query_ids[b])
    duplicate = []
    for first in range(0, size, 50000):
        ix = torch.arange(first, min(first+50000, size)); donor = result[ix]
        equal = (query_ids[ix] == query_ids[donor]).all(1) & (query_lengths[ix] == query_lengths[donor])
        duplicate.extend(ix[equal].tolist())
    swaps = 0
    for i in duplicate:
        if not same(i, int(result[i])): continue
        for offset in range(1, size):
            j = (i + offset) % size; di, dj = int(result[i]), int(result[j])
            if di != j and dj != i and not same(i, dj) and not same(j, di):
                result[i], result[j] = dj, di; swaps += 1; break
        else: raise ValueError('No legal different-query donor swap exists')
    if bool((result == torch.arange(size)).any()) or torch.unique(result).numel() != size:
        raise ValueError('Donor repair broke bijection or introduced a self pair')
    return result, {'initial_same_query_donor_count': len(duplicate), 'deterministic_donor_swaps': swaps,
        'final_same_query_donor_count': 0, 'repair_uses': 'recipient_and_donor_prompt_token_ids_only'}


def atomic_bank(bank):
    from .formal_evaluation import MatchedValidationBank
    indices = [i for i, r in enumerate(bank.records) if r.task_type in ATOMIC]
    counts = Counter(bank.records[i].task_type for i in indices)
    if counts != Counter({k: 50 for k in ATOMIC}): raise ValueError('Need exactly 200 atomic validation records')
    return MatchedValidationBank(tuple(bank.records[i] for i in indices), bank.vectors[indices].clone(),
        {**dict(bank.identity), 'derived_scope': 'atomic_only_no_composition_inference',
         'derived_source_indices': indices, 'derived_event_count': 200, 'derived_task_counts': dict(counts)})


def prepare(*, root, output):
    from .config import load_run_config, write_run_config
    from .packed_cache import PackedTokenizedVector397Dataset
    from . import formal_training  # Import actual transitive runtime dependencies for receipts.
    root = Path(root).resolve(); output = (root / output).resolve()
    if (output / 'plan.json').exists(): raise FileExistsError('Preserve previous plan')
    torch.set_num_threads(4)
    anchor_path = root / 's5_lean_circuit/configs/hypernet_atomic_1p2m_seed42/a_p03.json'
    anchor = load_run_config(anchor_path)
    dataset = PackedTokenizedVector397Dataset(root / anchor.data.cache_path, verify_payload_checksum=True)
    if len(dataset) != 1200000 or set(dataset.payload['task_ids'].tolist()) != {0, 1, 2, 3}:
        raise ValueError('Source is not complete 1.2M four-atomic corpus')
    mean = training_mean(dataset.payload['vectors'])
    if digest(mean) != 'a07e1199d0a24eda924b4f28b1eabd0df660cd802335f573544d03450151823e':
        raise ValueError('Mean differs from independent CPU audit')
    shared = root / 'outputs/s5_lean_circuit/content_origin_v2/shared'
    shared.mkdir(parents=True, exist_ok=False)
    mean_path = shared / 'train_mean.pt'
    with mean_path.open('xb') as f: torch.save({'mean': mean, 'count': len(dataset), 'already_fixed_log1p': True}, f)
    files = {'previous_failed_attempt': file(root / 'logs/server/2026-10-19/s5-content-origin-ydao-deployment-v1/matched/attempt-failure.json'),
        'previous_frozen_plan': file(root / 'logs/train/2026-10-19/s5_content_origin_v1/plan.json'),
        'anchor_config': file(anchor_path), 'training_mean': file(mean_path),
        'packed_cache': file(dataset.path), 'packed_cache_metadata': file(str(dataset.path) + '.json'),
        'base_ce': file(root / anchor.data.base_ce_path), 'base_ce_metadata': file(str(root / anchor.data.base_ce_path) + '.json'),
        'base_checkpoint': file(root / anchor.base.checkpoint_path), 'base_config': file(root / anchor.base.architecture_config_path),
        'tokenizer': file(root / anchor.data.tokenizer_path), 'archive_manifest': file(root / anchor.data.archive_manifest)}
    for role in ('common_init_manifest', 'candidate_init_manifest'):
        p = root / getattr(anchor, role); body = json.loads((p / 'manifest.json').read_text())
        files[role] = file(p / 'manifest.json'); files[role + '_state'] = file(p / body['state_path'])
    for role in ('corpus_manifest', 'vector_manifest', 'binding_path'):
        files['validation:' + role] = file(root / anchor.extra['matched_validation'][role])
    # Bind the actual reused Python implementation, never edit these sources.
    dependencies = ['content_origin_v2.py', 'formal_training.py', 'formal_evaluation.py', 'packed_cache.py',
                    'runtime.py', 'config.py', 'losses.py', 'circuit_normalizer.py']
    for name in dependencies: files['code:' + name] = file(Path(__file__).parent / name)
    import sys
    for module_name, module in sorted(sys.modules.items()):
        origin = getattr(module, '__file__', None)
        if not origin: continue
        p = Path(origin).resolve()
        if p.suffix == '.py' and p.is_relative_to(root) and any(x in p.parts for x in ('s5_lean_circuit', 's2_hypernet', 's1_lora_sft')):
            files['imported-code:' + module_name] = file(p)
    for rel in ('s5_lean_circuit/scripts/train_content_origin_v2.py', 's5_lean_circuit/tests/test_content_origin_v2.py'):
        files['code:' + rel] = file(root / rel)
    for rel in ('4server/s5_content_origin_v2/run.sh',
                '4server/s5_content_origin_v2/matched.yml',
                '4server/s5_content_origin_v2/permuted.yml',
                '4server/s5_content_origin_v2/mean.yml',
                'temp_s/s5_project_gpu_budget_20261019.py',
                'temp_s/s5_content_origin_ydao_siflow_20261019_v2.py',
                's5_lean_circuit/scripts/chunk_log_stream.py',
                '4server/0713/nvarc_lora/activate_nvarc_conda.sh'):
        files['launcher:' + rel] = file(root / rel)
    from dataclasses import replace
    variants = {}
    for mode in VARIANTS:
        cfg_path = root / f's5_lean_circuit/configs/content_origin_v2/{mode}.json'
        if cfg_path.exists(): raise FileExistsError('Preserve new controlled config')
        out = root / f'outputs/s5_lean_circuit/content_origin_v2/{mode}'
        extra = {'matrix': FORMAT, 'condition_source': 'circuit', 'fusion_mode': 'append',
            'mode': 'fixed_origin', 'content_treatment': mode, 'train_mean_path': str(mean_path),
            'permutation_seed': 731000042, 'atomic_validation': dict(anchor.extra['matched_validation']),
            'atomic_validation_only': True, 'atomic_validation_steps': list(VAL_STEPS),
            'stop_after_step': STOP, 'scheduler_total_steps': anchor.model.total_steps,
            'pg_gradient_checkpointing': False, 'checkpoint_selection': False,
            'mismatch_training': {'enabled': False}, 'alignment_training': {'enabled': False}}
        cfg = replace(anchor, output_dir=str(out), run_name=f's5-content-origin-{mode}-s42', extra=extra)
        cfg_path.parent.mkdir(parents=True, exist_ok=True); write_run_config(cfg_path, cfg)
        files['config:' + mode] = file(cfg_path)
        variants[mode] = {'config': str(cfg_path), 'output_dir': str(out), 'config_semantic_sha256': cfg.config_sha256}
    plan = dict(format=FORMAT, previous_failed_uuid='0b42001d-216e-4f5a-ad6c-c5c3724aa2e8',
        previous_optimizer_updates_before_failure=1, wrapper_revision=2, prepared_at=now(), status='frozen_before_new_training', files=files, variants=variants,
        stop_after_step=STOP, seed=42, atomic_validation_steps=list(VAL_STEPS),
        atomic_validation_only=True, sealed_read=False, mean_tensor_sha256=digest(mean),
        unique_train_events=1200000, scheduler_total_steps=anchor.model.total_steps,
        global_batch_size=512, mean_fit='unique_training_already_fixed_log1p_FP64_then_FP32',
        permutation='new_deterministic_global_N_cycle_each_epoch_then_query_only_deterministic_donor_swaps',
        train_data_order='unchanged_balanced_epoch_1.2M_plus_128_rotating_repeats',
        repeated_events_use_same_epoch_donor=True, additional_mismatch_or_alignment_objective=False,
        checkpoint_selection=False, initialization='same_untrained_common_and_candidate_A_manifest',
        validation_derangement_seed=732000042,
        validation_decoding_protocol='legacy_single_event_per_rank_BF16_greedy_diagnostic_not_official_batch64',
        exact_training_presentations=STOP*512, exact_padded_training_tokens=STOP*512*128,
        validation_scope='200 atomic only; native-policy diagnostics at fixed milestones; 3-input cross matrix at final fixed step',
        primary_final_comparison={'matched': 'matched', 'permuted': 'matched', 'mean': 'mean'},
        training_seed_replicates=1, generalization_or_significance_claimed=False)
    write(output / 'plan.json', plan)
    return {'plan': str(output / 'plan.json'), 'plan_sha256': sha(output / 'plan.json'), 'variants': variants}


def run(*, config, plan, plan_sha256, stop_after_step=STOP):
    from . import formal_training as ft
    from .config import load_run_config, write_run_config
    from .packed_cache import PackedTokenizedVector397Dataset, packed_cache_identity, load_base_ce_artifact
    from .runtime import S5HyperNet, load_s5_base_model, resolve_common_initialization_root, validate_common_initialization, validate_candidate_initialization
    from .formal_evaluation import load_matched_validation_bank, evaluate_matched_allnine, MatchedValidationBank
    from ..circuit.model_io import load_tokenizer
    from .losses import lora_gate_at_step, scale_microbatch_objective_for_ddp
    cfg = load_run_config(config); frozen = json.loads(Path(plan).read_text())
    mode = cfg.extra['content_treatment']
    if sha(plan) != plan_sha256 or frozen['format'] != FORMAT or mode not in VARIANTS:
        raise ValueError('Content protocol identity changed')
    if int(stop_after_step) != STOP or frozen['stop_after_step'] != STOP or cfg.model.total_steps != 38000:
        raise ValueError('Fixed stop or preserved 38000-step schedule changed')
    if cfg.config_sha256 != frozen['variants'][mode]['config_semantic_sha256']:
        raise ValueError('Config changed')
    rank, world, _, device = ft._init_distributed()
    if world != cfg.model.expected_world_size:
        raise ValueError('Execution topology differs from the loaded runtime config')
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    # Full physical input verification only on rank0; peers wait before work.
    if rank == 0: verify(frozen['files'])
    ft._barrier()
    output = Path(cfg.output_dir); logroot = Path(plan).resolve().parent / mode
    if rank == 0:
        output.mkdir(parents=True, exist_ok=False); logroot.mkdir(parents=True, exist_ok=True)
        write_run_config(output / 'resolved_config.json', cfg)
    ft._barrier()
    dataset = PackedTokenizedVector397Dataset(cfg.data.cache_path,
        expected_identity=packed_cache_identity(archive_manifest=cfg.data.archive_manifest,
            tokenizer_path=cfg.data.tokenizer_path, prefix=cfg.data.prefix, n_ctx=cfg.data.n_ctx,
            metadata_root=cfg.data.metadata_root), verify_payload_checksum=False)
    mean_payload = torch.load(cfg.extra['train_mean_path'], map_location='cpu', weights_only=True)
    mean = mean_payload['mean']
    if digest(mean) != frozen['mean_tensor_sha256'] or len(dataset) != 1200000:
        raise ValueError('Training descriptor statistics changed')
    base_ce, base_ce_meta = load_base_ce_artifact(cfg.data.base_ce_path, expected_count=len(dataset),
        expected_cache_identity_sha256=dataset.metadata['identity_sha256'],
        expected_cache_payload_sha256=dataset.metadata['payload_sha256'], expected_base_identity=asdict(cfg.base))
    ft._seed_everything(cfg.model.seed + rank)
    base, base_identity = load_s5_base_model(cfg.base.architecture_config_path, cfg.base.checkpoint_path,
        device=device, dtype=torch.bfloat16, expected_identity=cfg.base)
    core = S5HyperNet(base, candidate='A', seed=42, condition_source='circuit', fusion_mode='append')
    common = resolve_common_initialization_root(cfg.common_init_manifest)
    common_manifest = validate_common_initialization(common, expected_base_identity=asdict(base_identity),
        expected_seed=42, expected_layout_sha256=core.layout.layout_sha256)
    core.load_common_state_dict(torch.load(common / common_manifest['state_path'], map_location='cpu', weights_only=True))
    candidate = resolve_common_initialization_root(cfg.candidate_init_manifest)
    candidate_manifest = validate_candidate_initialization(candidate, expected_base_identity=asdict(base_identity),
        expected_seed=42, expected_candidate='A', expected_layout_sha256=core.layout.layout_sha256)
    core.load_circuit_state_dict(torch.load(candidate / candidate_manifest['state_path'], map_location='cpu', weights_only=True))
    core.to(device=device); core.validate_step0_initialization(); core.disable_gradient_checkpointing()
    ft._validate_step0_parity(core, dataset, base_ce, device=device)
    vc = cfg.extra['atomic_validation']
    tokenizer = load_tokenizer(cfg.data.tokenizer_path, expected_sha256=vc['tokenizer_sha256'])
    bank = atomic_bank(load_matched_validation_bank(corpus_manifest=vc['corpus_manifest'], vector_manifest=vc['vector_manifest'],
        binding_path=vc['binding_path'], expected_suite_sha256=vc['suite_sha256']))
    eval_sources = derangement(200, seed=732000042)
    prompt_tokens = [tokenizer.encode(r.input_text) for r in bank.records]
    query_lengths = torch.tensor([len(v) for v in prompt_tokens])
    query_ids = torch.zeros((200, int(query_lengths.max())), dtype=torch.long)
    for i, values in enumerate(prompt_tokens): query_ids[i, :len(values)] = torch.tensor(values)
    eval_sources, eval_repair = repair_same_query_donors(eval_sources, query_ids, query_lengths)
    core.eval()
    initial_hash = ft._file_sha256(common / common_manifest['state_path']) + ':' + ft._file_sha256(candidate / candidate_manifest['state_path'])
    initial_tensors = core.trainable_state_dict()
    initial_aggregate = hashlib.sha256()
    for name in sorted(initial_tensors):
        initial_aggregate.update(name.encode()); initial_aggregate.update(digest(initial_tensors[name]).encode())
    initial_aggregate = initial_aggregate.hexdigest()
    del initial_tensors
    model = torch.nn.parallel.DistributedDataParallel(core, device_ids=[device.index], output_device=device.index,
        broadcast_buffers=False, find_unused_parameters=False)
    optimizer = torch.optim.AdamW(core.parameter_groups(pg_lr=cfg.model.pg_lr,
        query_lr_multiplier=cfg.model.query_lr_multiplier, circuit_lr_multiplier=cfg.model.circuit_lr_multiplier,
        weight_decay=cfg.model.weight_decay), betas=(cfg.model.adam_beta1, cfg.model.adam_beta2), weight_decay=0.)
    scheduler = ft._scheduler(optimizer, cfg)
    identity = {**ft._checkpoint_identity(config=cfg, cache_metadata=dataset.metadata, base_ce_metadata=base_ce_meta,
        common_manifest=common_manifest, candidate_manifest=candidate_manifest, world_size=world),
        'content_origin_plan_sha256': plan_sha256, 'content_treatment': mode, 'from_first_exposure': True,
        'mean_tensor_sha256': digest(mean), 'permutation_seed': cfg.extra['permutation_seed']}
    if rank == 0:
        write(output / 'run_identity.json', {**identity, 'initialization_files_sha256': initial_hash,
            'actual_initial_trainable_tensor_aggregate_sha256': initial_aggregate,
            'validation_bank': dict(bank.identity), 'validation_permutation_sha256': digest(eval_sources),
            'validation_permutation_query_repair': eval_repair,
            'new_training_updates': 0, 'native_policy_validation_only_not_checkpoint_selection': True,
            'base_frozen': not any(p.requires_grad for p in base.parameters()),
            'trainable_parameters': sum(p.numel() for p in core.parameters() if p.requires_grad),
            'tf32': False, 'deterministic_algorithms_enabled': torch.are_deterministic_algorithms_enabled(),
            'fixed_stop': STOP, 'scheduler_horizon': cfg.model.total_steps})
    def checkpoint(step):
        if rank == 0:
            ft._save_model_only(output / f'checkpoints/model_step{step:07d}.pt', core=core,
                global_step=step, identity=identity, write_integrity=True)
        ft._barrier()
    def evaluate(step, eval_mode):
        vectors = treated_vectors(bank.vectors, torch.arange(200), eval_mode, mean, eval_sources)
        viewed = MatchedValidationBank(bank.records, vectors, {**dict(bank.identity), 'descriptor_policy': eval_mode})
        result = evaluate_matched_allnine(core, tokenizer, viewed, device=device, rank=rank, world_size=world,
            gate_multiplier=lora_gate_at_step(step, total_steps=cfg.model.total_steps,
                gate_max=cfg.model.gate_max, warmup_ratio=cfg.model.gate_warmup_ratio),
            task_types=ATOMIC, collect_predictions=True)
        if rank == 0:
            result.update(condition=eval_mode, training_treatment=mode, step=step, fixed_stop=STOP,
                checkpoint_selection=False, scope='atomic_only_development_fixed_stop_single_training_seed',
                decoding_protocol='legacy_single_event_per_rank_BF16_greedy_diagnostic_not_official_batch64_comparison')
            write(logroot / f'evaluations/step{step:07d}-{eval_mode}.json', result)
        ft._barrier()
    checkpoint(0)
    started = time.perf_counter(); started_at = now()
    epoch_cache = None; perm_epoch = None; sources = None; step_times = []; metrics = logroot / 'train_metrics.jsonl'
    data_trace = hashlib.sha256(); descriptor_trace = hashlib.sha256()
    model.train()
    for step in range(STOP):
        indices, epoch_cache = ft.rank_indices_for_step(step, dataset_size=len(dataset), seed=cfg.assignment_seed,
            rank=rank, world_size=world, cached_epoch=epoch_cache)
        epoch = step // ft.STEPS_PER_LOGICAL_EPOCH
        if mode == 'permuted' and epoch != perm_epoch:
            sources = derangement(len(dataset), seed=cfg.extra['permutation_seed'], epoch=epoch)
            sources, repair = repair_same_query_donors(sources, dataset.payload['query_ids'], dataset.payload['query_lengths'])
            perm_epoch = epoch
            if rank == 0:
                write(logroot / f'permutations/epoch{epoch:03d}.json', dict(epoch=epoch, seed=cfg.extra['permutation_seed'],
                    source_mapping_sha256=digest(sources), size=len(sources), fixed_points=0, bijective=True,
                    assignment='global_atomic_pool_without_task_labels', repeated_events_same_epoch_donor=True,
                    **repair, duplicate_query_donors_resampled=False))
        data_trace.update(indices.numpy().tobytes())
        descriptor_indices = sources[indices] if mode == 'permuted' else indices
        descriptor_trace.update(descriptor_indices.numpy().tobytes() if mode != 'mean' else b'train_mean')
        count = (dataset.payload['input_lengths'][indices].long() - 1).sum().to(device)
        total_tokens = ft._all_reduce(count)
        gate = lora_gate_at_step(step + 1, total_steps=cfg.model.total_steps, gate_max=cfg.model.gate_max,
            warmup_ratio=cfg.model.gate_warmup_ratio)
        optimizer.zero_grad(set_to_none=True)
        tick = time.perf_counter()
        batch = dataset.collate_indices(indices, device=device, frozen_base_ce=base_ce)
        # Keep recipient query/labels/baseCE unchanged; substitute only descriptor.
        batch['vector397'] = treated_vectors(dataset.payload['vectors'], indices, mode, mean, sources).to(device)
        with torch.autocast('cuda', dtype=torch.bfloat16):
            result = model(query_ids=batch['query_ids'], query_mask=batch['query_mask'], input_ids=batch['input_ids'],
                labels=batch['labels'], vector397=batch['vector397'], frozen_base_ce=batch['frozen_base_ce'],
                gate_multiplier=gate, task_ids=batch['task_ids'])
            hinge = result['base_preservation_hinge'].sum(); delta = result['effective_delta_l2'].sum()
            objective = scale_microbatch_objective_for_ddp(loss_sum=result['loss_sum'], hinge_sum=hinge,
                effective_delta_sum=delta, world_size=world, global_token_count=total_tokens,
                global_event_count=512, base_preservation_coefficient=cfg.model.base_preservation_coefficient,
                effective_delta_coefficient=cfg.model.effective_delta_l2)
        if not bool(torch.isfinite(objective)): raise FloatingPointError(f'Nonfinite objective at {step+1}')
        ce_sum = result['loss_sum'].detach().double(); hinge_metric=hinge.detach().double(); delta_metric=delta.detach().double()
        objective.backward()
        del result, objective, hinge, delta, batch
        groups = core.group_gradient_norms()
        required = ft._gradient_reachability_groups('circuit', step+1)
        if required and any(not math.isfinite(float(groups.get(k,0))) or groups.get(k,0)<=0 for k in required):
            raise RuntimeError('Unreachable gradient route: ' + str(groups))
        norm = torch.nn.utils.clip_grad_norm_([p for p in core.parameters() if p.requires_grad], cfg.model.max_grad_norm)
        if not bool(torch.isfinite(norm)): raise FloatingPointError('Nonfinite gradient')
        optimizer.step(); scheduler.step(); torch.cuda.synchronize(device)
        elapsed = time.perf_counter() - tick; step_times.append(elapsed)
        if (step+1) % 50 == 0 or step == 0:
            total_ce = ft._all_reduce(ce_sum); total_hinge = ft._all_reduce(hinge_metric); total_delta = ft._all_reduce(delta_metric)
            if rank == 0:
                record = dict(step=step+1, epoch=epoch, full_sequence_ce=float(total_ce/total_tokens),
                    base_preservation_hinge=float(total_hinge/512), effective_delta_l2=float(total_delta/512),
                    gate=gate, learning_rates=optimizer_learning_rates(optimizer), seconds=elapsed,
                    gradient_norm=float(norm), gradient_groups=groups, data_trace_sha256_rank0=data_trace.hexdigest(),
                    descriptor_trace_sha256_rank0=descriptor_trace.hexdigest(),
                    peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated(device),
                    peak_cuda_reserved_bytes=torch.cuda.max_memory_reserved(device))
                with metrics.open('a') as f: f.write(json.dumps(record,allow_nan=False)+'\n')
                print(json.dumps({'stage':'training','variant':mode,**record},allow_nan=False),flush=True)
        if step+1 in SAVE_STEPS: checkpoint(step+1)
        if step+1 in VAL_STEPS:
            evaluate(step+1, mode)
            if step+1 == STOP:
                for other in VARIANTS:
                    if other != mode: evaluate(step+1, other)
    runtimes=ft._gather_objects(dict(rank=rank, data_trace_sha256=data_trace.hexdigest(),
        descriptor_trace_sha256=descriptor_trace.hexdigest(), mean_step_seconds=sum(step_times)/len(step_times),
        steps=len(step_times),peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated(device),
        peak_cuda_reserved_bytes=torch.cuda.max_memory_reserved(device)),world)
    rng=ft._gather_objects(ft._rng_state(),world)
    if rank == 0:
        resume_path=output/f'checkpoints/resume_step{STOP:07d}.pt'
        ft._save_resume(resume_path,core=core,optimizer=optimizer,scheduler=scheduler,global_step=STOP,
            identity=identity,rng_by_rank=rng,runtime_by_rank=runtimes)
        result=dict(status='complete_fixed_origin',started_at=started_at,finished_at=now(),
            wall_seconds=time.perf_counter()-started,global_step=STOP,training_treatment=mode,
            plan_sha256=plan_sha256,identity=identity,runtime_by_rank=runtimes,
            model_checkpoint=file(output/f'checkpoints/model_step{STOP:07d}.pt'),resume_checkpoint=file(resume_path),
            checkpoint_selection=False,atomic_validation_count=200,composition_validation_inference=0,
            exact_training_presentations=STOP*512,exact_padded_training_tokens=STOP*512*128,
            additional_objectives=False,training_seed_replicates=1,sealed_read=False)
        write(output/'target_reached_summary.json',result);write(output/'target_reached.ready',result)
        write(logroot/'summary.json',result)
        print(json.dumps({'stage':'complete','variant':mode,'step':STOP}),flush=True)
    ft._barrier();torch.distributed.destroy_process_group()
    return {'status':'complete_fixed_origin','training_treatment':mode,'global_step':STOP}


def main():
    import argparse
    p=argparse.ArgumentParser(description=__doc__); sub=p.add_subparsers(dest='action',required=True)
    q=sub.add_parser('prepare');q.add_argument('--root',default='.');q.add_argument('--output',required=True)
    q=sub.add_parser('run')
    for name in ('config','plan','plan-sha256'):q.add_argument('--'+name,required=True)
    q.add_argument('--stop-after-step',type=int,default=STOP)
    args=vars(p.parse_args());action=args.pop('action')
    print(json.dumps((prepare if action=='prepare' else run)(**args),indent=2),flush=True)


if __name__=='__main__':main()
