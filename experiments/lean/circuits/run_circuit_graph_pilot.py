#!/usr/bin/env python3
"""Run one S5 Graph parity or audit work unit; production is intentionally blocked."""

from __future__ import annotations

import argparse
import glob
import hashlib
import json
import sys
from pathlib import Path


S5_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = S5_ROOT.parent
sys.path.insert(0, str(S5_ROOT / "src"))

import torch  # noqa: E402

from s5_lean_circuit.circuit.model_io import (  # noqa: E402
    load_frozen_source_model,
    sha256_file,
)
from s5_lean_circuit.circuit.bank import load_canonical_transcoder_bank  # noqa: E402
from s5_lean_circuit.circuit.payload import (  # noqa: E402
    circuit_artifact_id,
    semantic_sha256,
    write_dense_reference_artifact,
    write_payload_chunk,
)
from s5_lean_circuit.circuit.pilot import (  # noqa: E402
    compare_parity,
    extract_graph,
    intervention_score,
    operator_causal_comparison,
)
from s5_lean_circuit.circuit.settings import (  # noqa: E402
    load_circuit_config,
    trace_source_config_sha256,
)
from s5_lean_circuit.circuit.traces import (  # noqa: E402
    trace_generation_code_sha256,
    trace_generation_contract,
    verified_trace_corpus_identity,
    validate_audit_trace_inventory,
)
from s5_lean_circuit.config import load_config  # noqa: E402


def repo_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else REPO_ROOT / path


def parse_bank(
    values: list[str], *, device: torch.device, config_sha256: str
):
    rows = {}
    for value in values:
        layer_text, path_text = value.split("=", 1)
        layer = int(layer_text)
        rows[layer] = repo_path(path_text)
    if set(rows) != set(range(8)):
        raise ValueError("Graph pilot requires one Transcoder checkpoint for layers 0..7")
    return load_canonical_transcoder_bank(
        [rows[layer] for layer in range(8)],
        device=device,
        expected_config_sha256=config_sha256,
    )


def code_identity() -> str:
    digest = hashlib.sha256()
    source_root = S5_ROOT / "src" / "s5_lean_circuit" / "circuit"
    for path in sorted(source_root.glob("*.py")):
        digest.update(path.name.encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--trace-glob", required=True)
    parser.add_argument("--trace-index", type=int, required=True)
    parser.add_argument("--transcoder", action="append", default=[], metavar="LAYER=PATH")
    parser.add_argument("--scope", choices=("parity", "audit64", "audit256"), required=True)
    parser.add_argument("--feature-budget", type=int, choices=(2048, 4096, 8192, 16384), required=True)
    parser.add_argument("--node-threshold", type=float, choices=(0.8, 0.9, 0.95), required=True)
    parser.add_argument("--edge-threshold", type=float, choices=(0.98, 0.99, 0.995), required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--analysis-output", required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    config = load_circuit_config(repo_path(args.config))
    if bool(config["deferred_gates"]["production_900_authorized"]):
        raise ValueError("pilot config must not authorize production_900")
    device = torch.device(args.device)
    source = config["source_model"]
    source_config = load_config(repo_path(source["architecture_config"]))
    model, model_identity = load_frozen_source_model(
        source_config,
        repo_path(source["checkpoint"]),
        device=device,
        expected_checkpoint_config_sha256=source["checkpoint_config_sha256"],
        expected_checkpoint_sha256=source["checkpoint_sha256"],
        expected_architecture_config_sha256=source["architecture_config_sha256"],
    )
    bank, bank_sha = parse_bank(
        args.transcoder, device=device, config_sha256=config["_config_sha256"]
    )
    paths = sorted(glob.glob(str(repo_path(args.trace_glob))))
    if not paths:
        raise FileNotFoundError("Graph trace glob matched no files")
    tokenizer_sha = sha256_file(repo_path(source["tokenizer"]))
    if tokenizer_sha != source["tokenizer_sha256"]:
        raise ValueError("circuit tokenizer file SHA256 mismatch")
    corpus_split = "audit64" if args.scope in {"parity", "audit64"} else "audit256"
    corpus_examples = 64 if corpus_split == "audit64" else 256
    traces = validate_audit_trace_inventory(
        paths,
        expected_split=corpus_split,
        expected_examples=corpus_examples,
        expected_config_sha256=trace_source_config_sha256(config),
        expected_checkpoint_config_sha256=source["checkpoint_config_sha256"],
        expected_architecture_config_sha256=model_identity[
            "current_architecture_config_sha256"
        ],
        expected_checkpoint_sha256=model_identity["checkpoint_sha256"],
        expected_model_dtype=model_identity["dtype"],
        expected_tokenizer_sha256=tokenizer_sha,
        expected_generation_code_sha256=trace_generation_code_sha256(),
        expected_generation_contract=trace_generation_contract(config["traces"]),
    )
    if not 0 <= args.trace_index < len(traces):
        raise IndexError("trace-index is outside the trace corpus")
    trace = traces[args.trace_index]
    policy_common = {
        "operator": "joint_signed_full_output",
        "feature_budget": args.feature_budget,
        "node_threshold": args.node_threshold,
        "edge_threshold": args.edge_threshold,
        "queue_size": int(config["graph"]["queue_size"]),
        "row_batch_size": int(config["graph"]["row_batch_size"]),
    }
    code_sha = code_identity()
    output = repo_path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    analysis_output = repo_path(args.analysis_output)
    try:
        analysis_output.relative_to(output)
    except ValueError:
        pass
    else:
        raise ValueError("Graph analysis output must be outside payload archive")
    try:
        output.relative_to(analysis_output)
    except ValueError:
        pass
    else:
        raise ValueError("Graph payload archive must be outside analysis root")
    analysis_output.mkdir(parents=True, exist_ok=True)

    def run(backend: str):
        policy = {**policy_common, "backend": backend}
        artifact_id = circuit_artifact_id(
            event_key=trace.event_key,
            model_identity=model_identity,
            tokenizer_sha256=tokenizer_sha,
            transcoder_bank_sha256=bank_sha,
            graph_policy=policy,
            code_identity=code_sha,
        )
        return extract_graph(
            model,
            bank,
            trace,
            device=device,
            event_key=trace.event_key,
            artifact_id=artifact_id,
            feature_budget=args.feature_budget,
            node_threshold=args.node_threshold,
            edge_threshold=args.edge_threshold,
            queue_size=int(config["graph"]["queue_size"]),
            row_batch_size=int(config["graph"]["row_batch_size"]),
            backend=backend,
            max_cache_bytes=int(config["graph"]["max_cache_bytes"]),
            max_dense_reference_bytes=int(config["graph"]["max_dense_reference_bytes"]),
            max_selected_dense_bytes=int(config["graph"]["max_selected_dense_bytes"]),
        )

    archive_identity = {
        "config_sha256": config["_config_sha256"],
        "model_identity": model_identity,
        "tokenizer_sha256": tokenizer_sha,
        "transcoder_bank_sha256": bank_sha,
        "code_identity": code_sha,
        "scope": args.scope,
        "trace_corpus_identity": verified_trace_corpus_identity(
            paths,
            [trace],
            split=corpus_split,
            seed_base=777_777_000,
        ),
    }
    archive_identity["trace_corpus_identity_sha256"] = semantic_sha256(
        archive_identity["trace_corpus_identity"]
    )
    candidate = run("streaming_cached_exact")
    candidate_chunk = write_payload_chunk(
        output / "candidate_payload.safetensors",
        [candidate.payload],
        archive_identity=archive_identity,
    )
    result = {
        "format": "s5_circuit_graph_pilot_result_v1",
        "scope": args.scope,
        "trace_index": args.trace_index,
        "event_key": trace.event_key,
        "task_type": trace.task_type,
        "seed": trace.seed,
        "policy": policy_common,
        "candidate": {
            "seconds": candidate.seconds,
            "selected_features": int(candidate.selected_features.numel()),
            "retained_features": candidate.payload.feature_count,
            "retained_errors": candidate.payload.error_count,
            "edges": int(candidate.payload.edge_source.numel()),
            "bytes": candidate.payload.approximate_bytes(),
            "feature_influence_sum": float(candidate.payload.feature_influence.sum()),
            "error_influence_sum": float(candidate.payload.error_influence.sum()),
            "token_influence_sum": float(candidate.payload.token_influence.sum()),
            "edge_influence_sum": float(candidate.payload.edge_nonnegative_influence.sum()),
            "cache": dict(candidate.cache_stats),
            "chunk": candidate_chunk,
            "artifact_id": candidate.payload.artifact_id,
        },
        "status": "complete",
    }
    if args.scope == "parity":
        reference = run("dense_autograd_reference")
        reference_chunk = write_payload_chunk(
            output / "reference_payload.safetensors",
            [reference.payload],
            archive_identity={**archive_identity, "backend": "dense_autograd_reference"},
        )
        result["parity"] = compare_parity(
            reference, candidate, atol=1e-5, rtol=1e-4
        )
        result["reference_chunk"] = reference_chunk
        if (
            reference.dense_reference_signed_direct_effect is None
            or reference.dense_reference_selected_records is None
        ):
            raise RuntimeError("dense reference backend did not retain its matrix")
        result["dense_reference"] = write_dense_reference_artifact(
            output / "dense_reference.safetensors",
            signed_direct_effect=reference.dense_reference_signed_direct_effect,
            selected_features=reference.selected_features,
            selected_feature_records=reference.dense_reference_selected_records,
            replay_token_ids=reference.payload.replay_token_ids,
            event_key=trace.event_key,
            artifact_id=reference.payload.artifact_id,
            archive_identity={
                **archive_identity,
                "backend": "dense_autograd_reference",
            },
        )
        if not result["parity"]["passes"]:
            result["status"] = "failed"
    else:
        result["intervention"] = intervention_score(
            model,
            bank,
            trace,
            candidate,
            device=device,
            intervention_fraction=float(config["graph"]["intervention_fraction"]),
            random_controls=int(config["graph"]["random_controls"]),
        )
        result["operator_causal_comparison"] = operator_causal_comparison(
            model,
            bank,
            trace,
            candidate,
            device=device,
            feature_budget=args.feature_budget,
            queue_size=int(config["graph"]["queue_size"]),
            row_batch_size=int(config["graph"]["row_batch_size"]),
            max_cache_bytes=int(config["graph"]["max_cache_bytes"]),
            intervention_fraction=float(config["graph"]["intervention_fraction"]),
            random_controls=int(config["graph"]["random_controls"]),
        )
    result_path = analysis_output / "result.json"
    result_path.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({key: value for key, value in result.items() if key != "candidate"}, sort_keys=True))
    if result["status"] != "complete":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
