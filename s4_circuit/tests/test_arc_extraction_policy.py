import argparse
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
import json
from pathlib import Path
import runpy
from types import SimpleNamespace

import pytest

from s3circuit.nvarc_circuit.config import load_config
from s3circuit.nvarc_circuit.policy import (
    REARC8_POLICY_FORMAT,
    REARC8_V4_NUMERICS_BACKEND,
    Rearc8ProductionPolicy,
    add_compute_microbatch_argument,
    load_production_policy,
)


def test_extraction_policy_preserves_source_and_finalization_identity(tmp_path, monkeypatch):
    root = Path(__file__).resolve().parents[2]
    monkeypatch.setenv("HC_ROOT", str(root))
    config = load_config(
        root / "s3_circuit/configs/nvarc_circuit/rearc8_s42_step130_seed24_exclusive_stage_b_v1.yaml"
    )
    source = Rearc8ProductionPolicy(
        config_sha256=config.semantic_sha256, model_identity_sha256="a" * 64,
        tokenizer_identity_sha256="b" * 64, transcoder_identity_sha256="c" * 64,
        backend_identity=config.attribution.backend, operator_identity=config.attribution.operator,
        generation_batch_size=4, row_batch_size=config.attribution.row_batch_size,
        compute_microbatch_policy="fixed_cap_v1", compute_microbatch_cap=8,
        feature_budget=8192, update_interval=config.attribution.attribute_update_interval,
        node_threshold=0.8, edge_threshold=0.98, path_epsilon=config.attribution.path_epsilon,
        diagnostic_atol=0.0, diagnostic_rtol=0.0, max_node_selected_dense_bytes=1_000_000,
        gate_sha256={"stage_b_preflight": "d" * 64, "v4_closure": "e" * 64},
        numerics_backend=REARC8_V4_NUMERICS_BACKEND,
    )
    source_path = tmp_path / "input.json"
    source_bytes = json.dumps({
        "format": REARC8_POLICY_FORMAT, **asdict(source), "policy_sha256": source.semantic_sha256,
    }).encode()
    source_path.write_bytes(source_bytes)
    verified = load_production_policy(source_path, config=config)
    parser = argparse.ArgumentParser()
    add_compute_microbatch_argument(parser)
    effective = verified.with_compute_microbatch_cap(parser.parse_args([]).compute_microbatch_cap)
    assert asdict(effective) == {**asdict(source), "compute_microbatch_cap": 1}
    assert effective.semantic_sha256 != source.semantic_sha256
    with ThreadPoolExecutor(max_workers=4) as workers:
        paths = list(workers.map(effective.save_for_extraction, [tmp_path] * 8))
    assert set(paths) == {tmp_path / "policies" / f"{effective.semantic_sha256}.json"}
    assert list(paths[0].parent.iterdir()) == [paths[0]]
    assert load_production_policy(paths[0], config=config) == effective
    assert source_path.read_bytes() == source_bytes
    requested = parser.parse_args(["--compute-microbatch-cap", "16"]).compute_microbatch_cap
    assert verified.with_compute_microbatch_cap(requested).semantic_sha256 == source.semantic_sha256
    with pytest.raises(SystemExit):
        parser.parse_args(["--compute-microbatch-cap", "0"])
    source_path.write_bytes(source_bytes.replace(b'"compute_microbatch_cap": 8', b'"compute_microbatch_cap": 3'))
    with pytest.raises(ValueError, match="checksum"):
        load_production_policy(source_path, config=config)


def test_worker_schedule_accepts_only_the_source_or_effective_policy():
    root = Path(__file__).resolve().parents[2]
    worker = runpy.run_path(str(root / "s3_circuit/scripts/extract_nvarc_circuit_payload_worker.py"))
    resolve = worker["_schedule_policy_sha256"]
    source = SimpleNamespace(semantic_sha256="a" * 64)
    effective = SimpleNamespace(semantic_sha256="b" * 64)
    for known in (source.semantic_sha256, effective.semantic_sha256):
        assert resolve({"production_policy_sha256": known}, source, effective) == known
    with pytest.raises(ValueError, match="different production policy"):
        resolve({"production_policy_sha256": "c" * 64}, source, effective)
