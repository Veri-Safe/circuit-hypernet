#!/usr/bin/env python3
"""Apply a C25 block/site SAE group in the real PG and score full generations."""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import json
from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
for p in (ROOT / "s5_lean_circuit/src", ROOT / "s2_hypernet/src", HERE):
    sys.path.insert(0, str(p))
from s5_lean_circuit.circuit.transcoder import RectangularReLUTranscoder
from s5_lean_circuit.composition_loco import load_eval_bank
from s5_lean_circuit.composition_loco_eval import evaluate_bank
from s5_lean_circuit.pg_transcoder_adapter import load_adapter
from prepare import CHECKPOINT_SHA, OUT, TASKS, sha, write
from prepare_block import VISIBLE, ATOMIC

MODES = ("native_on", "native_off", "noop", "ablate", "withdraw", "restore", "whole_swap")


def select_bank(scope, bank_name):
    if scope == "full500":
        manifest = json.loads((HERE / "source_manifest.json").read_text())
        if (manifest.get("protocol") != "existing_dev500_nine_tasks"
                or manifest.get("excluded_final_test") is not True):
            raise ValueError("C25 dev500-only protocol differs")
        info = manifest["banks"][bank_name]
        path = ROOT / info["path"]
        if "final_test" in info["path"] or info["tasks"] != {t: 500 for t in TASKS[bank_name]}:
            raise ValueError("C25 dev500 full500 task identity differs")
        if sha(path) != info["sha256"]:
            raise ValueError("C25 full500 bank SHA differs")
        bank = load_eval_bank(path)
        if bank.metadata["identity_sha256"] != info["metadata_identity_sha256"]:
            raise ValueError("C25 full500 bank identity differs")
        return bank, info["batches"], info["sha256"]
    source = json.loads((HERE / "block_source/manifest.json").read_text())
    path = VISIBLE if bank_name == "composition" else ATOMIC
    expected = source[f"{bank_name}_bank_sha256"]
    if sha(path) != expected:
        raise ValueError("C25 visible/atomic bank SHA differs")
    bank = load_eval_bank(path)
    selected = []
    for task in TASKS[bank_name]:
        indices = [i for i,r in enumerate(bank.records) if r.task_type == task]
        if len(indices) != 500:
            raise ValueError("C25 visible task denominator differs")
        selected.extend(indices[400:500])
    by_length = {}
    for index in sorted(selected):
        length = int(bank.payload["query_lengths"][index])
        by_length.setdefault(length, []).append(index)
    batches = [indices[start:start+16] for _,indices in sorted(by_length.items())
               for start in range(0,len(indices),16)]
    if len({i for batch in batches for i in batch}) != 100 * len(TASKS[bank_name]):
        raise ValueError("C25 visible100 batch coverage differs")
    return bank, batches, expected


def load_group(block, name):
    if name is None:
        return None, None
    path = HERE / "block_groups" / f"block{block:02d}.json"
    document = json.loads(path.read_text())
    groups = [g for g in document["groups"] if g["name"] == name]
    if (not document["complete"] or document["checkpoint_sha256"] != CHECKPOINT_SHA
            or len(groups) != 1 or groups[0]["pg_block"] != block):
        raise ValueError("C25 block group identity differs")
    group = groups[0]
    ckpt_path = ROOT / group["dictionary_path"]
    if sha(ckpt_path) != group["dictionary_sha256"]:
        raise ValueError("C25 block SAE hash differs")
    checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False, mmap=True)
    config = checkpoint["config"]
    site_index = config["site_names"].index(group["site"])
    if (checkpoint["capture_manifest_sha256"] != group["capture_manifest_sha256"]
            or config["c25_checkpoint_sha256"] != CHECKPOINT_SHA
            or config["block"] != block or config["dictionary_size"] != 8192
            or config["slot_indices"] != list(range(272))
            or float(config["site_rms"][site_index]) != float(group["site_rms"])
            or group["slot_range"] != [0,272]
            or len(group["feature_ids"]) != len(set(group["feature_ids"]))
            or any(not 0 <= fid < 8192 for fid in group["feature_ids"])):
        raise ValueError("C25 block SAE feature/normalization identity differs")
    model = RectangularReLUTranscoder.from_state_dict(checkpoint["state_dict"]).eval().cuda()
    return group, model


def canonical(site, output, batch):
    kind = site.split(".")[1]
    if kind == "self_attn_n":
        if output.shape != (batch*34,8,1024):
            raise ValueError("C25 self_attn_n shape differs")
        return output.reshape(batch,34,8,1024).permute(0,2,1,3).reshape(batch,272,1024)
    if kind == "self_attn_hw":
        if output.shape != (batch*8,34,1024):
            raise ValueError("C25 self_attn_hw shape differs")
        return output.reshape(batch,272,1024)
    if kind == "cross_attn":
        if output.shape != (batch,272,1024):
            raise ValueError("C25 cross_attn shape differs")
        return output
    if output.shape != (batch,8,34,1024):
        raise ValueError(f"C25 {kind} shape differs")
    return output.reshape(batch,272,1024)


def raw_layout(site, flat, template, batch):
    if site.endswith(".self_attn_n"):
        return flat.reshape(batch,8,34,1024).permute(0,2,1,3).reshape_as(template)
    return flat.reshape_as(template)


@contextmanager
def site_hook(core, group, batch, callback):
    block = core.parameter_generator.model.blocks[group["pg_block"]]
    kind = group["site"].split(".")[1]
    module = block if kind == "residual" else getattr(block,kind)
    calls = {"count":0}
    def hook(_module,_inputs,output):
        calls["count"] += 1
        native = canonical(group["site"],output,batch)
        changed = callback(native)
        if changed.shape != native.shape or not bool(torch.isfinite(changed).all()):
            raise ValueError("C25 block SAE site edit malformed")
        return raw_layout(group["site"],changed,output,batch).to(output.dtype)
    handle=module.register_forward_hook(hook)
    try: yield calls
    finally: handle.remove()


def decode(model,value,ids,scale):
    with torch.autocast("cuda",dtype=torch.bfloat16):
        z=model.encode(value.float()/scale)
    return (z[...,ids].float() @ model.W_dec[ids].float())*scale


@torch.inference_mode()
def pair_at_site(core,bank,indices,group):
    batch=len(indices)
    length=int(bank.payload["query_lengths"][indices[0]])
    ids=bank.payload["query_ids"][indices,:length].to("cuda",torch.long)
    mask=torch.ones_like(ids,dtype=torch.bool)
    vectors=bank.payload["vectors"][indices].to("cuda")
    values={}
    for name,present in (("on",True),("off",False)):
        captured=[]
        with site_hook(core,group,batch,lambda out:(captured.append(out.detach().clone()) or out)) as calls:
            with torch.autocast("cuda",dtype=torch.bfloat16):
                core.generate_lora(ids,mask,vectors,
                    circuit_present=torch.full((batch,),present,device="cuda",dtype=torch.bool),
                    gate_multiplier=.5)
        if calls["count"]!=1 or len(captured)!=1:
            raise ValueError("C25 paired site capture differs")
        values[name]=captured[0]
    return values


@torch.inference_mode()
def evaluate_one(adapter,bank,indices,bank_name,group,model,mode,alpha):
    core=adapter.hypernet
    pair=pair_at_site(core,bank,indices,group) if mode in {"withdraw","restore","whole_swap"} else None
    delta=None
    if pair is not None:
        if mode=="whole_swap":
            delta=pair["off"].float()-pair["on"].float()
        else:
            delta=decode(model,pair["on"],group["feature_ids"],group["site_rms"])-decode(
                model,pair["off"],group["feature_ids"],group["site_rms"])
    def edit(native):
        if mode=="noop" or alpha==0:
            return native.clone()
        if mode=="whole_swap":
            return (native.float()+alpha*delta).to(native.dtype)
        change=decode(model,native,group["feature_ids"],group["site_rms"]) if mode=="ablate" else delta
        sign=1 if mode=="restore" else -1
        return (native.float()+sign*alpha*change).to(native.dtype)
    if group:
        context=site_hook(core,group,len(indices),edit)
    else:
        from contextlib import nullcontext
        context=nullcontext({"count":0})
    with context as calls:
        result=evaluate_bank(core,adapter.tokenizer,bank,device=torch.device("cuda"),
            task_types=TASKS[bank_name],n_ctx=128,max_new_tokens=50,route="circuit_a",
            condition="absent" if mode in {"native_off","restore"} else "matched",
            gate_multiplier=.5,indices=indices,collect_predictions=True,collect_generated_text=True,batch_size=16)
    if (result is None or len(result["predictions"])!=len(indices)
            or calls["count"]!=(1 if group else 0)):
        raise ValueError("C25 block group evaluation incomplete")
    return result["predictions"]


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--scope",choices=("visible100","full500"),required=True)
    parser.add_argument("--bank",choices=TASKS,required=True)
    parser.add_argument("--block",type=int,required=True)
    parser.add_argument("--candidate")
    parser.add_argument("--mode",choices=MODES,required=True)
    parser.add_argument("--alpha",type=float,default=1.0)
    parser.add_argument("--parts",type=int,default=1)
    parser.add_argument("--part",type=int,default=0)
    args=parser.parse_args()
    if (not torch.cuda.is_available()
            or not 0<=args.block<12 or not 0<=args.alpha<=1
            or args.parts<1 or not 0<=args.part<args.parts
            or (args.mode not in {"native_on","native_off"} and args.candidate is None)):
        raise ValueError("C25 block evaluation requires CUDA and valid candidate/partition")
    torch.set_num_threads(4)
    bank,batches,bank_sha=select_bank(args.scope,args.bank)
    group,model=load_group(args.block,args.candidate)
    target=json.loads((HERE/"target.json").read_text())
    if target["checkpoint_sha256"]!=CHECKPOINT_SHA:
        raise ValueError("C25 target changed")
    adapter=load_adapter(target,root=ROOT,device=torch.device("cuda"))
    label=(args.candidate or "native")+f"_{args.mode}_a{args.alpha:g}"
    folder=OUT/("runs" if args.scope=="full500" else "visible100")/label/args.bank
    try:
        for batch_id,indices in enumerate(batches):
            if batch_id%args.parts!=args.part: continue
            path=folder/f"batch{batch_id:04d}.json"
            identity={"checkpoint_sha256":CHECKPOINT_SHA,"bank_sha256":bank_sha,
                      "scope":args.scope,"block":args.block,"indices":indices,"batch_id":batch_id,
                      "candidate":group,"dictionary_sha256":group["dictionary_sha256"] if group else None,
                      "mode":args.mode,"alpha":args.alpha,"batch_size":16,
                      "source_sha256":sha(Path(__file__))}
            if path.exists():
                old=json.loads(path.read_text())
                if old.get("identity")!=identity or not old.get("complete"):
                    raise ValueError(f"C25 block evaluation cache identity differs: {path}")
                continue
            rows=evaluate_one(adapter,bank,indices,args.bank,group,model,args.mode,args.alpha)
            write(path,{"identity":identity,"complete":True,
                        "predictions":rows})
            if batch_id%20==0:
                print("C25_BLOCK_PROGRESS",args.scope,label,args.bank,args.part,batch_id,len(batches),flush=True)
        print("C25_BLOCK_PART_COMPLETE",args.scope,label,args.bank,args.part,flush=True)
    finally:
        adapter.close()


if __name__=="__main__":main()
