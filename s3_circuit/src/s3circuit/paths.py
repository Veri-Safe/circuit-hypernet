from __future__ import annotations

from pathlib import Path


S3_ROOT = Path(__file__).resolve().parents[2]
REPO_ROOT = S3_ROOT.parent
OUTPUT_ROOT = S3_ROOT / "outputs"


def resolve_repo_path(path_text: str | Path) -> Path:
    path = Path(path_text)
    if path.is_absolute():
        return path
    return (REPO_ROOT / path).resolve()


def resolve_s3_path(path_text: str | Path) -> Path:
    path = Path(path_text)
    if path.is_absolute():
        return path
    return (S3_ROOT / path).resolve()


def shard_name(shard_id: int) -> str:
    return f"shard_{shard_id:03d}"


def default_manifest_path(run_name: str) -> Path:
    return OUTPUT_ROOT / "manifests" / run_name / "manifest.jsonl"


def default_registry_path(registry_name: str) -> Path:
    return OUTPUT_ROOT / "registries" / registry_name / "registry.jsonl"


def default_registry_summary_path(registry_name: str) -> Path:
    return OUTPUT_ROOT / "registries" / registry_name / "registry_summary.json"


def default_delta_manifest_path(delta_name: str) -> Path:
    return OUTPUT_ROOT / "manifests" / delta_name / "manifest.jsonl"


def default_delta_summary_path(delta_name: str) -> Path:
    return OUTPUT_ROOT / "manifests" / delta_name / "manifest_summary.json"


def default_generation_path(run_name: str, shard_id: int) -> Path:
    return OUTPUT_ROOT / "generations" / run_name / f"{shard_name(shard_id)}.jsonl"


def default_graph_dir(run_name: str, shard_id: int) -> Path:
    return OUTPUT_ROOT / "graphs" / run_name / shard_name(shard_id)


def default_payload_dir(run_name: str, shard_id: int) -> Path:
    return OUTPUT_ROOT / "payloads" / run_name / shard_name(shard_id)


def default_graph_metadata_path(run_name: str, shard_id: int) -> Path:
    return OUTPUT_ROOT / "metadata" / run_name / f"{shard_name(shard_id)}.jsonl"


def default_vector_path(run_name: str, shard_id: int) -> Path:
    return OUTPUT_ROOT / "vectors" / run_name / f"{shard_name(shard_id)}.jsonl"


def default_log_path(run_name: str, shard_id: int) -> Path:
    return OUTPUT_ROOT / "logs" / run_name / f"{shard_name(shard_id)}.log"
