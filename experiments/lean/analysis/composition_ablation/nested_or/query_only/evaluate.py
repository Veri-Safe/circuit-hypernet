#!/usr/bin/env python3
"""Real greedy candidate comparison with per-question native replay checks."""
import argparse
import gzip
import json
import os

from shared import ROOT,HERE,LOG,OUT,TASKS,inputs,references,screen_questions,identity,sha,write
import engine
from engine import torch


def main():
    p=argparse.ArgumentParser();p.add_argument("--phase",choices=("screen","full"),required=True)
    p.add_argument("--parts", type=int, default=1)
    p.add_argument("--unit", type=int, default=0)
    a = p.parse_args()
    if not 0 <= a.unit < a.parts:
        p.error("unit must be within the requested partitions")
    protocol,rows=inputs();assert 0<=a.unit<a.parts
    refs=references();gd=json.loads((HERE/"groups.json").read_text())
    assert gd["complete"] and gd["protocol_sha256"]==sha(HERE/"protocol.json")
    groups=gd["groups"]
    if a.phase=="screen":population=screen_questions(rows)
    else:
        s=json.loads((LOG/"selection.json").read_text());assert s["complete"] and sha(LOG/"selection.json") == protocol["selection_record_sha256"]
        groups=[g for g in groups if g["id"] in s["groups"]];assert len(groups)==7
        population=rows
    selected=population[a.unit::a.parts];names=[g["id"] for g in groups]
    core,tok,banks,models,scales=engine.load(protocol,dictionaries=True)
    write(LOG/f"slot_maps/{a.phase}/parts{a.parts}/unit{a.unit}.json",engine.slot_map(core))
    counts={n:{t:0 for t in TASKS} for n in ("native",*names)};totals={t:0 for t in TASKS}
    target=OUT/f"predictions/{a.phase}/parts{a.parts}/unit{a.unit}.jsonl.gz";assert not target.exists()
    target.parent.mkdir(parents=True,exist_ok=True);tmp=target.with_name(f".{target.name}.{os.getpid()}.tmp")
    try:
        with gzip.open(tmp,"wt") as f,torch.inference_mode():
            for i,q in enumerate(selected,1):
                ids=engine.question_input(q,banks);native,sites=engine.native_capture(core,ids)
                predictions={"native":engine.predict(core,tok,ids,native,q)}
                assert predictions["native"]==refs[q["task"],q["index"]]["predictions"]["native"],"Native replay differs"
                for g in groups:
                    if i==1:
                        noop=engine.edit(core,ids,g,sites,models,scales,noop=True)
                        assert all(torch.equal(noop[k],native[k]) for k in native)
                    changed=engine.edit(core,ids,g,sites,models,scales)
                    predictions[g["id"]]=engine.predict(core,tok,ids,changed,q)
                for n,pred in predictions.items():counts[n][q["task"]]+=int(pred["correct"])
                totals[q["task"]]+=1
                f.write(json.dumps({"question":q,"predictions":predictions},ensure_ascii=False,separators=(",",":"))+"\n")
                if i==1 or i%25==0 or i==len(selected):
                    print("QUERY_NOR_PROGRESS",a.phase,a.unit,i,len(selected),flush=True)
        os.replace(tmp,target)
        write(LOG/f"receipts/{a.phase}/parts{a.parts}/unit{a.unit}.json",{"complete":True,"identity":identity(a.phase,a.unit,a.parts),
            "path":str(target.relative_to(ROOT)),"sha256":sha(target),"counts":counts,"totals":totals,"candidates":names,
            "groups_sha256":sha(HERE/"groups.json"),"dictionary_manifest_sha256":sha(LOG/"dictionary_manifest.json"),
            "first_question_all_group_noop_states_equal":True,"native_reproduced_every_question":True})
        print("QUERY_NOR_COMPLETE",a.phase,a.unit,flush=True)
    finally:core.close()


if __name__=="__main__":main()
