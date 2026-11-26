#!/usr/bin/env python3
"""Downstream teacher-forced derivatives rank features; greedy accuracy stays separate."""
import argparse
import json
import os

import numpy as np
from shared import ROOT,HERE,LOG,OUT,TASKS,BLOCKS,FEATURES,inputs,references,identity,sha,write
import engine
from engine import torch


def main():
    p=argparse.ArgumentParser();p.add_argument("--phase",choices=("gradient_smoke","rank"),required=True)
    p.add_argument("--parts", type=int, default=1)
    p.add_argument("--unit", type=int, default=0)
    a = p.parse_args()
    if not 0 <= a.unit < a.parts:
        p.error("unit must be within the requested partitions")
    protocol,rows=inputs();refs=references()
    smoke=a.phase=="gradient_smoke";assert a.unit==0 if smoke else 0<=a.unit<a.parts
    selected=[next(q for q in rows if q["task"]==t and refs[q["task"],q["index"]]["predictions"]["native"]["correct"]) for t in TASKS] if smoke else rows[a.unit::a.parts]
    core,tok,banks,models,scales=engine.load(protocol,dictionaries=True)
    write(LOG/f"slot_maps/{a.phase}/parts{a.parts}/unit{a.unit}.json",engine.slot_map(core))
    effect=np.zeros((9,3,FEATURES),dtype=np.float64);correct=np.zeros(9,dtype=np.int64);totals=np.zeros_like(correct)
    checks=[];losses=np.zeros(9);norms=np.zeros(9)
    try:
        for i,q in enumerate(selected,1):
            ti=TASKS.index(q["task"]);totals[ti]+=1;ref=refs[q["task"],q["index"]]["predictions"]["native"]
            if ref["correct"]:
                ids=engine.question_input(q,banks)
                if smoke:
                    with torch.inference_mode():
                        native,sites=engine.native_capture(core,ids)
                        assert engine.predict(core,tok,ids,native,q)==ref
                        group={"alpha":0.,"sites":[{"block":b,"feature_ids":[0,1,2,3]} for b in BLOCKS]}
                        noop=engine.edit(core,ids,group,sites,models,scales,noop=True)
                        assert all(torch.equal(noop[k],native[k]) for k in native)
                        assert engine.predict(core,tok,ids,noop,q)==ref
                tokens=engine.tactic_tokens(tok,ref,q)
                effects,loss,norm,state,traces=engine.gradient_effects(core,ids,tokens,models,scales,check=smoke)
                if smoke:
                    assert all(torch.equal(state[k],native[k]) for k in native)
                    checks.append({"task":q["task"],"index":q["index"],"native_state_equal":True,
                                   "joint_noop_state_and_prediction_equal":True,"chain_rule_checks":traces})
                effect[ti]+=effects.astype(np.float64);correct[ti]+=1;losses[ti]+=loss;norms[ti]+=norm
            if i==1 or i%25==0 or i==len(selected):
                print("QUERY_NOR_PROGRESS",a.phase,a.unit,i,len(selected),flush=True)
        assert np.isfinite(effect).all() and norms.sum()>0
        dest=OUT/f"gradient/{a.phase}/parts{a.parts}/unit{a.unit}.npz";assert not dest.exists();dest.parent.mkdir(parents=True,exist_ok=True)
        temp=dest.with_suffix(".tmp.npz")
        np.savez_compressed(temp,effect_sum=effect,correct=correct,totals=totals,loss_sum=losses,gradient_norm_squared_sum=norms)
        os.replace(temp,dest)
        write(LOG/f"receipts/{a.phase}/parts{a.parts}/unit{a.unit}.json",{"complete":True,"identity":identity(a.phase,a.unit,a.parts),
            "path":str(dest.relative_to(ROOT)),"sha256":sha(dest),"correct":dict(zip(TASKS,correct.tolist())),
            "totals":dict(zip(TASKS,totals.tolist())),"question_keys":[[q["task"],q["index"]] for q in selected],
            "checks":checks,"site_rms":scales,"no_parameter_updates":True,"loss_is_ranking_only":True,
            "dictionary_manifest_sha256":sha(LOG/"dictionary_manifest.json"),
        })
        print("QUERY_NOR_COMPLETE",a.phase,a.unit,flush=True)
    finally:core.close()


if __name__=="__main__":main()
