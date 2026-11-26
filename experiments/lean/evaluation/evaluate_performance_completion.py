#!/usr/bin/env python3
"""Evaluate frozen 9-task suites with the existing greedy decoder and scorer."""
from __future__ import annotations
import argparse
from dataclasses import replace
import hashlib
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT/'s5_lean_circuit/src'),str(ROOT/'s2_hypernet/src')]
import torch
import torch.distributed as dist
from s5_lean_circuit.performance_completion import OUTPUT, PACKAGE, read, evaluation_cases
from s5_lean_circuit.composition_loco import load_eval_bank, sha256_file
from s5_lean_circuit.composition_loco_config import load_config, matches_checkpoint_config
from s5_lean_circuit.composition_loco_train import _rank_world, _load_direct, _load_hypernet, _base_identity
from s5_lean_circuit.composition_loco_eval import evaluate_bank, _shuffle_permutation
from s5_lean_circuit.circuit.model_io import load_tokenizer
from s5_lean_circuit.hypernet.runtime import load_s5_base_model

def write(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_suffix(path.suffix+f'.tmp.{os.getpid()}')
    temp.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n');temp.replace(path)

def main():
    p=argparse.ArgumentParser();p.add_argument('--spec',required=True);a=p.parse_args()
    spec=read(a.spec); signature=hashlib.sha256(Path(a.spec).read_bytes()).hexdigest()
    rank,world,_,device=_rank_world()
    config=load_config(ROOT/spec['config']); checkpoint=ROOT/spec['checkpoint']
    checkpoint_sha = sha256_file(checkpoint)
    if checkpoint_sha!=spec['checkpoint_sha256']:raise ValueError('Evaluation checkpoint SHA changed')
    suite=spec['suite']; bank=load_eval_bank(ROOT/OUTPUT/'banks'/f'{suite}.pt')
    expected=read(PACKAGE/'expected_corpus_identities.json')[suite]
    if bank.metadata['corpus_manifest_identity_sha256']!=expected['manifest_identity_sha256']:
        raise ValueError('Bank is not the frozen new suite')
    corpus=ROOT/OUTPUT/'corpora'/suite
    samples=[json.loads(x) for x in (corpus/'samples.jsonl').read_text().splitlines()]
    tasks=tuple(read(corpus/'manifest.json')['tasks'])
    if len(samples)!=4500 or any(sum(r.task_type==t for r in bank.records)!=500 for t in tasks):
        raise ValueError('Full nine-task 500-per-task coverage required')
    cases=evaluation_cases(spec,tasks)
    permutation=_shuffle_permutation(bank.records)
    mapping=[]
    for target,source in enumerate(permutation):
        if source==target or bank.records[source].task_type!=bank.records[target].task_type:
            raise ValueError('Shuffle donor is self or crosses task')
        mapping.append({'suite':suite,'task':bank.records[target].task_type,'target_index':target,'source_index':source,
                        'target_event_key':bytes(bank.payload['event_key'][target].tolist()).hex(),
                        'source_event_key':bytes(bank.payload['event_key'][source].tolist()).hex()})
    if sorted(permutation)!=list(range(4500)):raise ValueError('Shuffle is not a full permutation')
    mapping_sha=hashlib.sha256(json.dumps(mapping,sort_keys=True,separators=(',',':')).encode()).hexdigest()
    mapping_record={'shuffle_seed':42,'mapping_sha256':mapping_sha,'bank_identity_sha256':bank.metadata['identity_sha256'],
                    'scope':'full_nine_task_bank_per_suite','mapping':mapping}
    mapping_path=ROOT/'logs/data/2026-11-03/lean-performance-completion-v1'/f'shuffle_{suite}.json'
    if rank==0:
        if mapping_path.exists() and read(mapping_path)!=mapping_record:raise ValueError('Frozen shared shuffle mapping changed')
        if not mapping_path.exists():write(mapping_path,mapping_record)
    tokenizer=load_tokenizer(ROOT/config.data['tokenizer_path'])
    route=spec['route']; kind=spec['kind']
    if kind=='base':
        model,_=load_s5_base_model(ROOT/config.base['architecture_config_path'],checkpoint,
                                   device=device,dtype=torch.bfloat16,expected_identity=_base_identity(config))
        eval_route='direct_sft'
    else:
        model=_load_direct(config,device) if route=='direct_sft' else _load_hypernet(config,device)
        eval_route=route
        payload=torch.load(checkpoint,map_location='cpu',weights_only=False,mmap=True)
        if int(payload.get('global_step',payload.get('step',-1)))!=spec['step']:
            raise ValueError('Checkpoint step differs')
        if kind=='composition':
            if (not matches_checkpoint_config(config.computed_sha256, payload.get('config_sha256'), checkpoint_sha) or payload.get('route')!=route
                    or payload.get('fold_id')!=config.fold_id):raise ValueError('Checkpoint recipe/fold differs')
        elif checkpoint.resolve()!=(ROOT/config.initialization['checkpoint_path']).resolve():
            raise ValueError('Atomic evaluation must load the exact route initialization')
        state=payload['model_state_dict']
        if route=='direct_sft':model.load_state_dict(state,strict=True)
        else:model.load_trainable_state_dict(state)
        del payload,state
    out=ROOT/spec['result_dir']; started=time.monotonic()
    identity={k:spec[k] for k in ('kind','route','heldout','group','role','training_seed','checkpoint','checkpoint_sha256','step','suite','selection')}
    identity.update(spec_sha256=signature,config_sha256=config.computed_sha256,
                    bank_identity_sha256=bank.metadata['identity_sha256'],batch_size=16,world_size=world,
                    precision='bf16_autocast',n_ctx=128,max_new_tokens=50)
    identity.update(shuffle_seed=42,shuffle_mapping_sha256=mapping_sha)
    conditions=spec['conditions']
    if 'cross_task' in conditions:
        identity.pop('shuffle_seed',None);identity.pop('shuffle_mapping_sha256',None)
        identity.update(pairing_seed=spec['pairing_seed'],reference_eval_id=spec['reference_eval_id'])
    preferred=[spec['heldout']] if spec['heldout'] in tasks else ['T_and_or','T_nested_or']
    task_order=preferred+[t for t in tasks if t not in preferred]
    cases.sort(key=lambda case: task_order.index(case['task']))
    for case in cases:
        condition,task,donor=case['label'],case['task'],case['donor_task']
        eval_bank=bank
        if condition=='zero12_token_present':
            payload=dict(bank.payload);payload['vectors']=payload['vectors'].clone()
            payload['vectors'][:,list(range(9))+[131,132,133]]=0
            eval_bank=replace(bank,payload=payload)
        target=out/f'{condition}.{task}.json'
        predpath=out/f'predictions.{condition}.{task}.jsonl'
        if target.exists():
            old=read(target)
            if old['identity']!=identity or old['predictions_sha256']!=sha256_file(predpath):
                raise ValueError('Existing evaluation identity/records changed')
            continue
        result=evaluate_bank(model,tokenizer,eval_bank,device=device,task_types=(task,),route=eval_route,
                 condition=case['condition'], donor_task=donor, pairing_seed=int(spec.get('pairing_seed',42)),
                 gate_multiplier=float(config.model.get('gate_max',1.0)),
                 collect_predictions=True,collect_generated_text=True,batch_size=16)
        if rank==0:
            predictions=result.pop('predictions')
            if len(predictions)!=500 or result['overall_total']!=500:raise ValueError('Incomplete task result')
            records=[]
            for row in predictions:
                sample=samples[row['canonical_global_index']]
                row.update(identity)
                row.update(condition=condition,task=task,prompt_sha256=sample['input_text_sha256'],
                           sample_id=sample['prompt_key'],circuit_present=condition!='absent',
                           vector_fill='matched_values_masked' if condition=='absent' else 'cross_task_donor' if donor is not None else 'same_task_donor' if condition=='shuffled' else 'matched',
                           vector_edit_stage='post_fixed_log1p_v1' if condition=='zero12_token_present' else None,
                           zero_dimensions=list(range(9))+[131,132,133] if condition=='zero12_token_present' else [])
                if donor is not None:
                    row.update(donor_task=donor,cross_task_mapping_sha256=result['condition_mapping']['sha256'])
                row['contains_gold_tactic']=row['expected_tactic'] in row['generated_text']
                row['strict_fail_with_gold_present']=not row['correct'] and row['contains_gold_tactic']
                records.append(row)
            out.mkdir(parents=True,exist_ok=True)
            raw=''.join(json.dumps(r,ensure_ascii=False,separators=(',',':'))+'\n' for r in records)
            tmp=predpath.with_suffix('.jsonl.tmp');tmp.write_text(raw);tmp.replace(predpath)
            result.update(identity=identity,condition=condition,task=task,predictions_path=str(predpath.relative_to(ROOT)),
                          predictions_sha256=sha256_file(predpath),status='complete',donor_task=donor)
            write(target,result)
            print(json.dumps({'suite':suite,'route':route,'group':spec['group'],'task':task,'condition':condition,
                              'correct':result['overall_correct'],'total':500}),flush=True)
        if dist.is_initialized():
            dist.barrier()
    if rank==0:write(out/'complete.json',{'status':'complete','identity':identity,'conditions':conditions,'tasks':sorted({c['task'] for c in cases}),'cases':cases,'wall_seconds':time.monotonic()-started})
    if dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()

if __name__=='__main__':main()
