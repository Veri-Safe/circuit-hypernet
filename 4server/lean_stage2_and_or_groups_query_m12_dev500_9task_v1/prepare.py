#!/usr/bin/env python3
"""Lock the Query-only M12 step4000 checkpoint and nine existing dev500 tasks."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import sys

import torch

ROOT=Path(__file__).resolve().parents[2]
HERE=Path(__file__).resolve().parent
OUT=ROOT/"outputs/pg_transcoder/lean_stage2_and_or_groups_query_m12_dev500_9task_v1"
LOG=ROOT/"logs/eval/2026-11-12/lean-stage2-and-or-groups-query-m12-dev500-9task-v1"
M12=ROOT/"4server/lean_stage2_and_or_groups_m12_dev500_9task_v1"
RUN=ROOT/"outputs/s5_lean_circuit/composition_loco_query_retention20_v1/runs/T_and_or/M12/query_only"
CONFIG=ROOT/"s5_lean_circuit/configs/composition_loco_query_retention20_v1/T_and_or/M12/query_only.json"
CHECKPOINT=RUN/"checkpoints/model_step004000.pt"
CHECKPOINT_SHA="9d66b8acb215c64502294be421463ef3d5cba9db73b0d835f51b899a0d3e73bb"
BASE=ROOT/"outputs/s5_lean_circuit/atomic_only_dense_v1/dense-bgoff_attempt_20260820_200505/best_model.pt"
TOKENIZER=ROOT/"datasets/s5_lean_circuit/tokenizer_v1/tokenizer.model"
TASKS={"composition":("T_and_or","T_or_and","T_and_or_and","T_nested_and","T_nested_or"),
       "atomic":("S_and","S_or_l","S_or_r","S_id")}
for p in (ROOT/"s5_lean_circuit/src",ROOT/"s2_hypernet/src"):
    sys.path.insert(0,str(p))
from s5_lean_circuit.composition_loco import load_eval_bank
from s5_lean_circuit.composition_loco_config import load_config


def sha(path:Path)->str:
    h=hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda:stream.read(8*1024*1024),b""):h.update(chunk)
    return h.hexdigest()


def write(path:Path,value:object)->None:
    path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_name(f".{path.name}.{os.getpid()}.tmp")
    tmp.write_text(json.dumps(value,ensure_ascii=False,indent=2,sort_keys=True)+"\n")
    os.replace(tmp,path)


def main()->None:
    config=load_config(CONFIG)
    raw=json.loads(CONFIG.read_text())
    selection=json.loads((RUN/"selection.json").read_text())
    ood=json.loads((RUN/"ood_validation.json").read_text())
    if sha(CHECKPOINT) != CHECKPOINT_SHA:
        raise ValueError("Query M12 checkpoint identity differs")
    ckpt=torch.load(CHECKPOINT,map_location="cpu",weights_only=False,mmap=True)
    if (sha(CHECKPOINT)!=CHECKPOINT_SHA
            or sha(CONFIG)!="fa04cceb2cffea9802128ebc60ffea7ff1e6b0192cf869c9d2281e7d1c57fe90"
            or config.config_sha256!="449826f4f09e7bc43336a72b6f910f2f4e5e80539921fe682b0261c6b0e24e05"
            or sha(BASE)!=raw["base"]["checkpoint_sha256"]
            or sha(TOKENIZER)!="1053e327f96ee13ceca0c4ab4d9e48eefa9eb2c2eddf65359715642ec701199f"
            or config.route!="query_only" or config.heldout_task!="T_and_or"
            or config.model["condition_source"]!="query_only"
            or config.model["group"]!="QO02" or config.model["gate_max"]!=.5
            or config.evaluation["max_new_tokens"]!=50
            or config.evaluation["metric"]!="strict_first_tactic_exact_match"
            or config.initialization["checkpoint_sha256"]!="80b51d000c9e48e1d5a8abe3bcb2a8e9ce9c86481db1977021dd6a0876c366d4"
            or selection["selected_step"]!=4000 or ood["selected_step"]!=4000
            or ood["conditions"]["matched"]["overall_correct"]!=106
            or ood["conditions"]["matched"]["overall_total"]!=500
            or ckpt.get("format")!="s5_composition_loco_checkpoint_v1"
            or ckpt.get("protocol_id")!=config.protocol_id
            or ckpt.get("route")!="query_only"
            or ckpt.get("fold_id")!=config.fold_id
            or ckpt.get("heldout_task")!="T_and_or"
            or ckpt.get("global_step")!=4000
            or ckpt["config_sha256"] != selection["config_sha256"]
            or ckpt.get("kind")!="model_only"
            or not isinstance(ckpt.get("model_state_dict"),dict)):
        raise ValueError("Query M12 checkpoint/config/base/score identity differs")
    prior=json.loads((M12/"source_manifest.json").read_text())
    if (prior["protocol"]!="existing_dev500_nine_tasks" or prior["excluded_final_test"] is not True
            or prior["base_sha256"]!=sha(BASE) or prior["tokenizer_sha256"]!=sha(TOKENIZER)):
        raise ValueError("approved nine-task bank source differs")
    seen=set()
    for bank_name,info in prior["banks"].items():
        path=ROOT/info["path"]
        if ("final_test" in info["path"] or sha(path)!=info["sha256"]
                or info["tasks"]!={t:500 for t in TASKS[bank_name]}):
            raise ValueError("Query M12 bank hash/denominator differs")
        bank=load_eval_bank(path)
        if (bank.metadata["identity_sha256"]!=info["metadata_identity_sha256"]
                or bank.metadata["expected_source_base_identity"]["checkpoint_sha256"]!=sha(BASE)
                or bank.metadata["tokenizer_sha256"]!=sha(TOKENIZER)
                or sorted(i for batch in info["batches"] for i in batch)
                   != sorted(i for i,r in enumerate(bank.records) if r.task_type in TASKS[bank_name])):
            raise ValueError("Query M12 bank metadata/coverage differs")
        for i,r in enumerate(bank.records):
            if r.task_type in TASKS[bank_name]:
                key=bytes(bank.payload["event_key"][i].tolist())
                if key in seen:raise ValueError("Query M12 duplicate event key")
                seen.add(key)
    if len(seen)!=4500:raise ValueError("Query M12 requires 4500 unique task events")
    selected=json.loads((M12/"main_arm.json").read_text())
    if (not selected["selection_locked"] or selected["feature_count"]!=64
            or selected["checkpoint_sha256"]!="71ec472481877655b50a4534e5410b912c7bcac720abd16772957406a6fff03a"):
        raise ValueError("Circuit M12 selected direction source differs")
    for group in selected["groups"]:
        p=ROOT/group["dictionary_path"]
        if sha(p)!=group["dictionary_sha256"] or group["slot_range"]!=[0,272]:
            raise ValueError("Circuit M12 transfer SAE identity differs")
    write(HERE/"source_manifest.json",{**prior,
          "format":"lean_stage2_query_m12_dev500_source_v1",
          "checkpoint_sha256":CHECKPOINT_SHA,"config_sha256":sha(CONFIG),
          "bank_source_manifest_sha256":sha(M12/"source_manifest.json")})
    write(HERE/"target.json",{"format":"lean_stage2_query_m12_target_v1",
          "checkpoint_path":str(CHECKPOINT.relative_to(ROOT)),"checkpoint_sha256":CHECKPOINT_SHA,
          "config_path":str(CONFIG.relative_to(ROOT)),"config_file_sha256":sha(CONFIG),
          "config_semantic_sha256":config.config_sha256,
          "base_sha256":sha(BASE),"tokenizer_path":str(TOKENIZER.relative_to(ROOT)),
          "tokenizer_sha256":sha(TOKENIZER),"route":"query_only",
          "condition_source":"query_only","condition":"matched","gate_multiplier":.5,
          "external_lora_multiplier":1.0,"slot_count":272,"max_new_tokens":50,
          "score":"strict_first_tactic_exact_match"})
    write(HERE/"transfer_group.json",{"format":"lean_stage2_query_m12_circuit_direction_transfer_v1",
          "query_checkpoint_sha256":CHECKPOINT_SHA,
          "source_circuit_m12_main_sha256":sha(M12/"main_arm.json"),
          "source_circuit_m12_checkpoint_sha256":selected["checkpoint_sha256"],
          "interpretation":"external fixed M12 Circuit SAE directions in Query-only PG coordinates; not native Query-only feature IDs",
          "groups":selected["groups"]})
    print("QUERY_M12_PREPARED",{"checkpoint":CHECKPOINT_SHA,"questions":len(seen),
          "historical_ood_correct":106},flush=True)


if __name__=="__main__":main()
