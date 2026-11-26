#!/usr/bin/env python3
"""Rank Query M12 full-slot SAE features by held-out validation activation specificity."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
for p in (ROOT / "s5_lean_circuit/src", ROOT / "s2_hypernet/src", HERE):
    sys.path.insert(0, str(p))
from s5_lean_circuit.circuit.transcoder import RectangularReLUTranscoder
from block_sae import (DATA, FEATURES, KINDS, LOG, SHARD_QUESTIONS, SLOTS, SPLITS,
                       TASKS, WIDTH, load_capture_mmaps, sha, sites_for, write_json)
from prepare import CHECKPOINT_SHA


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--block", type=int, required=True)
    args = parser.parse_args()
    block = args.block
    if not 0 <= block < 12 or not torch.cuda.is_available():
        raise ValueError("Query M12 rank needs a valid block and CUDA")
    _, mmaps, capture_sha = load_capture_mmaps(block)
    report_path = LOG / f"block{block:02d}_fit_summary.json"
    report = json.loads(report_path.read_text())
    ckpt_path = DATA / "models" / f"block{block:02d}_shared5site_8192/best.pt"
    saved = torch.load(ckpt_path, map_location="cpu", weights_only=False, mmap=True)
    config = saved["config"]
    if (not report["complete"] or report["checkpoint_sha256"] != sha(ckpt_path)
            or report["capture_manifest_sha256"] != capture_sha
            or saved["capture_manifest_sha256"] != capture_sha
            or config["query_m12_checkpoint_sha256"] != CHECKPOINT_SHA
            or config["site_names"] != sites_for(block)
            or config["training_tasks"] != list(TASKS)
            or config["dictionary_size"] != FEATURES):
        raise ValueError("Query M12 block SAE checkpoint or training identity differs")
    model = RectangularReLUTranscoder.from_state_dict(saved["state_dict"]).eval().cuda()
    scales = config["site_rms"]
    dec_norm = model.W_dec.float().norm(dim=1).cpu().numpy()
    result = {"format": "lean_stage2_query_m12_block_sae_rank_v1", "complete": False,
              "checkpoint_sha256": CHECKPOINT_SHA, "block": block,
              "dictionary_path": str(ckpt_path.relative_to(ROOT)),
              "dictionary_sha256": sha(ckpt_path), "capture_manifest_sha256": capture_sha,
              "source_split": [300,400], "all_272_slots": True, "sites": {}}
    for site_index, site in enumerate(sites_for(block)):
        means = {}
        for task in TASKS:
            totals = torch.zeros(FEATURES, device="cuda", dtype=torch.float64)
            n = 0
            for shard in range(SPLITS["validation"][0] // SHARD_QUESTIONS,
                               SPLITS["validation"][1] // SHARD_QUESTIONS):
                mmap = mmaps[(task, shard)]
                for q in range(0, SHARD_QUESTIONS, 4):
                    chunk = np.asarray(mmap[q:min(q+4,SHARD_QUESTIONS),site_index], dtype=np.float32)
                    values = torch.from_numpy(chunk.reshape(-1,WIDTH)).to("cuda") / float(scales[site_index])
                    with torch.autocast("cuda", dtype=torch.bfloat16):
                        z = model.encode(values)
                    totals += z.double().sum(dim=0)
                    n += len(values)
            if n != 100 * len(SLOTS):
                raise ValueError(f"Query M12 rank {task} slot count differs")
            means[task] = (totals / n).cpu().numpy()
        ood = means["T_and_or"]
        id_max = np.max(np.stack([means[t] for t in TASKS[1:5]]),axis=0)
        atom_max = np.max(np.stack([means[t] for t in TASKS[5:]]),axis=0)
        selective = (ood - np.maximum(id_max, atom_max)) * dec_norm
        general = ood * dec_norm
        site_rows = {}
        for rule, score in (("selective", selective), ("ood_activation", general)):
            order = np.argsort(-score, kind="stable")
            site_rows[rule] = [{"feature_id": int(fid), "score": float(score[fid]),
                                "ood_mean": float(ood[fid]), "id_max_mean": float(id_max[fid]),
                                "atomic_max_mean": float(atom_max[fid]),
                                "decoder_norm": float(dec_norm[fid])}
                               for fid in order[:256]]
        result["sites"][site] = site_rows
        write_json(LOG / f"block{block:02d}_rank.json", result)
        print("QUERY_M12_BLOCK_RANK", site, float(selective.max()), flush=True)
    result["complete"] = True
    write_json(LOG / f"block{block:02d}_rank.json", result)
    groups = []
    for site_index, site in enumerate(sites_for(block)):
        for rule, rows in result["sites"][site].items():
            for size in (32, 64, 128, 256):
                groups.append({"name": f"qm12-b{block:02d}-{KINDS[site_index]}-{rule}-{size}",
                               "dictionary": "query_m12_block_fullslot_sae",
                               "dictionary_path": result["dictionary_path"],
                               "dictionary_sha256": result["dictionary_sha256"],
                               "capture_manifest_sha256": capture_sha,
                               "pg_block": block, "site": site,
                               "slot_range": [0,272],
                               "feature_ids": [row["feature_id"] for row in rows[:size]],
                               "site_rms": float(scales[site_index]),
                               "ranking": rule, "ranking_split": "Query_M12_visible_validation_300_399"})
    write_json(HERE / "block_groups" / f"block{block:02d}.json",
               {"format": "lean_stage2_query_m12_block_sae_groups_v1", "complete": True,
                "checkpoint_sha256": CHECKPOINT_SHA, "rank_sha256": sha(LOG / f"block{block:02d}_rank.json"),
                "groups": groups})


if __name__ == "__main__":
    main()
