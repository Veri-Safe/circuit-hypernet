#!/usr/bin/env python3
"""Load the native Query-only LOCO checkpoint with strict source identity."""
from __future__ import annotations

import json
from pathlib import Path

import torch

from prepare import BASE,CHECKPOINT,CHECKPOINT_SHA,CONFIG,HERE,ROOT,TOKENIZER,sha
from s5_lean_circuit.circuit.model_io import load_tokenizer
from s5_lean_circuit.composition_loco_config import load_config
from s5_lean_circuit.composition_loco_train import _load_hypernet


class QueryAdapter:
    def __init__(self,core,tokenizer):
        self.hypernet=core
        self.tokenizer=tokenizer
        self.gate_multiplier=.5

    def close(self):
        self.hypernet.close()


def load_query_adapter(device:torch.device)->QueryAdapter:
    target=json.loads((HERE/"target.json").read_text())
    config=load_config(CONFIG)
    if (target["checkpoint_sha256"]!=CHECKPOINT_SHA or sha(CHECKPOINT)!=CHECKPOINT_SHA
            or (sha(CONFIG), config.config_sha256) != ("fa04cceb2cffea9802128ebc60ffea7ff1e6b0192cf869c9d2281e7d1c57fe90", "449826f4f09e7bc43336a72b6f910f2f4e5e80539921fe682b0261c6b0e24e05")
            or (target["config_file_sha256"], target["config_semantic_sha256"]) != (sha(CONFIG), config.config_sha256)
            or target["base_sha256"]!=sha(BASE)
            or target["tokenizer_sha256"]!=sha(TOKENIZER)
            or not (target["route"]==config.route=="query_only")
            or not (target["condition_source"]==config.model["condition_source"]=="query_only")
            or not (target["gate_multiplier"]==float(config.model["gate_max"])==.5)
            or target["external_lora_multiplier"]!=1.0
            or target["slot_count"]!=272 or target["max_new_tokens"]!=50):
        raise ValueError("Query M12 adapter target identity differs")
    core=_load_hypernet(config,device)
    payload=torch.load(CHECKPOINT,map_location="cpu",weights_only=False,mmap=True)
    if (payload["format"]!="s5_composition_loco_checkpoint_v1"
            or payload["protocol_id"]!=config.protocol_id
            or payload["route"]!="query_only"
            or payload["fold_id"]!=config.fold_id
            or payload["heldout_task"]!="T_and_or"
            or payload["global_step"]!=4000
            or payload["kind"]!="model_only"):
        raise ValueError("Query M12 checkpoint envelope differs")
    core.load_trainable_state_dict(payload["model_state_dict"])
    core.to(device).eval()
    pg=core.parameter_generator.model
    if (core.condition_source!="query_only" or core.fusion_mode!="append"
            or len(pg.blocks)!=12 or pg.seq_n!=8 or pg.seq_h!=34
            or pg.d_model!=1024 or core.controller.scale_multiplier!=1.0
            or core.controller.target_scope!="all"):
        raise ValueError("Query M12 native PG/LoRA layout differs")
    tokenizer=load_tokenizer(TOKENIZER,expected_sha256=target["tokenizer_sha256"])
    return QueryAdapter(core,tokenizer)
