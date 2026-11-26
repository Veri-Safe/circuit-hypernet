#!/usr/bin/env python3
"""Real greedy screening/full evaluation of fixed gradient-selected feature sets."""
from __future__ import annotations

import argparse
import gzip
import json
import os

from shared import (ROOT, HERE, LOG, OUT, TASKS, inputs, references,
                    screen_questions, identity, sha, write)
import engine
from engine import torch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=("screen", "full"), required=True)
    parser.add_argument("--parts", type=int, default=1)
    parser.add_argument("--lane", type=int, default=0)
    args = parser.parse_args()
    if not 0 <= args.lane < args.parts:
        parser.error("lane must be within the requested partitions")
    protocol, questions = inputs()
    refs = references(questions)
    groups_data = json.loads((HERE / "groups.json").read_text())
    assert groups_data["complete"] and groups_data["protocol_sha256"] == sha(HERE / "protocol.json")
    groups = groups_data["groups"]
    if args.phase == "screen":
        selected = screen_questions(questions)[args.lane::args.parts]
    else:
        selection = json.loads((LOG / "selection.json").read_text())
        assert selection["complete"] and sha(LOG / "selection.json") == protocol["selection_record_sha256"]
        groups = [g for g in groups if g["id"] in selection["groups"]]
        assert len(groups) == 6 and groups[0]["id"] == "anchor8" and groups[1]["id"] == "anchor11"
        selected = questions[args.lane::args.parts]
    adapter, models, scales, banks, target = engine.load(protocol)
    write(LOG / "slot_maps" / args.phase / f"parts{args.parts}" / f"lane{args.lane}.json", engine.slot_map(adapter.hypernet.parameter_generator))
    names = [g["id"] for g in groups]
    counts = {name: {task: 0 for task in TASKS} for name in ("native", *names)}
    totals = {task: 0 for task in TASKS}
    output = OUT / "predictions" / args.phase / f"parts{args.parts}" / f"lane{args.lane}.jsonl.gz"
    assert not output.exists(), "Do not overwrite predictions"
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.{os.getpid()}.tmp")
    try:
        with gzip.open(temporary, "wt") as stream, torch.inference_mode():
            for ordinal, q in enumerate(selected, 1):
                ids, vector = engine.question_input(q, banks)
                native_state, native_sites = engine.native_capture(adapter, ids, vector)
                predictions = {"native": engine.predict(adapter, ids, native_state, target, q)}
                reference = refs[q["task"], q["index"]]
                assert predictions["native"] == reference["native"], "Native per-question replay differs"
                for group in groups:
                    if ordinal == 1:
                        noop = engine.edit(adapter, ids, vector, group, native_sites, models, scales, noop=True)
                        assert all(torch.equal(noop[k], native_state[k]) for k in noop)
                    edited = engine.edit(adapter, ids, vector, group, native_sites, models, scales)
                    predictions[group["id"]] = engine.predict(adapter, ids, edited, target, q)
                for anchor in ("anchor8", "anchor11"):
                    assert predictions[anchor] == reference[anchor], f"{anchor} per-question replay differs"
                for name, prediction in predictions.items():
                    counts[name][q["task"]] += int(prediction["correct"])
                totals[q["task"]] += 1
                stream.write(json.dumps({"question": q, "predictions": predictions}, ensure_ascii=False, separators=(",", ":")) + "\n")
                if ordinal == 1 or ordinal % 25 == 0 or ordinal == len(selected):
                    print("LEAN_NOR_R03_GRADIENT_PROGRESS", args.phase, args.lane, ordinal,
                          len(selected), flush=True)
        os.replace(temporary, output)
        write(LOG / "receipts" / args.phase / f"parts{args.parts}" / f"lane{args.lane}.json", {
            "complete": True, "identity": identity(args.phase, args.lane, args.parts),
            "groups_sha256": sha(HERE / "groups.json"), "candidates": names,
            "path": str(output.relative_to(ROOT)), "sha256": sha(output),
            "totals": totals, "counts": counts,
            "batch_size": 1})
        print("LEAN_NOR_R03_GRADIENT_COMPLETE", args.phase, args.lane, flush=True)
    finally:
        adapter.close()


if __name__ == "__main__":
    main()
