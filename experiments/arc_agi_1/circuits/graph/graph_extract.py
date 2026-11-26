from __future__ import annotations

import sys
from pathlib import Path
from collections.abc import Iterable
from typing import Any

import torch
import yaml
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from s3circuit.codecircuit_compat import patch_huggingface_hub_for_codecircuit
from s3circuit.io_jsonl import JsonRecord, append_jsonl, existing_example_ids, iter_jsonl
from s3circuit.paths import REPO_ROOT, S3_ROOT
from s3circuit.prompts import build_trace_text


def ensure_codecircuit_on_path() -> Path:
    codecircuit_root = REPO_ROOT / "ref_repos" / "CodeCircuit"
    if not codecircuit_root.exists():
        raise FileNotFoundError(f"Cannot locate CodeCircuit repo at {codecircuit_root}")
    path_text = str(codecircuit_root)
    if path_text not in sys.path:
        sys.path.insert(0, path_text)
    return codecircuit_root


def dtype_from_name(dtype_name: str) -> torch.dtype:
    normalized = dtype_name.lower()
    aliases = {
        "fp32": "float32",
        "bf16": "bfloat16",
        "fp16": "float16",
    }
    normalized = aliases.get(normalized, normalized)
    return getattr(torch, normalized)


def load_replacement_model(
    model_name_or_path: str,
    transcoder_set: str,
    dtype_name: str,
    device: str,
    lazy_encoder: bool = False,
    lazy_decoder: bool = True,
) -> Any:
    ensure_codecircuit_on_path()
    patch_huggingface_hub_for_codecircuit()
    from circuit_tracer import ReplacementModel

    dtype = dtype_from_name(dtype_name)
    resolved_model, tl_model_name, hf_model, tokenizer = resolve_graph_model_inputs(
        model_name_or_path,
        dtype=dtype,
        device=device,
    )
    torch_device = torch.device(device)
    model_kwargs: dict[str, Any] = {}
    if hf_model is not None:
        model_kwargs["hf_model"] = hf_model
    if tokenizer is not None:
        model_kwargs["tokenizer"] = tokenizer

    local_config_path = resolve_local_transcoder_config_path(transcoder_set)
    if local_config_path is not None:
        transcoders = load_local_transcoders(
            local_config_path,
            dtype=dtype,
            device=torch_device,
            lazy_encoder=lazy_encoder,
            lazy_decoder=lazy_decoder,
        )
        model = ReplacementModel.from_pretrained_and_transcoders(
            tl_model_name,
            transcoders,
            dtype=dtype,
            device=torch_device,
            **model_kwargs,
        )
    else:
        model = ReplacementModel.from_pretrained(
            tl_model_name,
            transcoder_set,
            dtype=dtype,
            device=torch_device,
            lazy_encoder=lazy_encoder,
            lazy_decoder=lazy_decoder,
            **model_kwargs,
        )
    return model.to(device)


def resolve_model_name_or_path(model_name_or_path: str) -> str:
    path = Path(model_name_or_path)
    candidates = [path] if path.is_absolute() else [REPO_ROOT / path, path]
    for candidate in candidates:
        if candidate.exists():
            return str(candidate.resolve())
    return model_name_or_path


def resolve_graph_model_inputs(
    model_name_or_path: str,
    dtype: torch.dtype,
    device: str,
) -> tuple[str, str, Any | None, Any | None]:
    resolved_model = resolve_model_name_or_path(model_name_or_path)
    model_path = Path(resolved_model)
    if not model_path.exists():
        return resolved_model, resolved_model, None, None

    tl_model_name = infer_transformer_lens_model_name(model_path)
    tokenizer = AutoTokenizer.from_pretrained(str(model_path), use_fast=True, local_files_only=True)
    hf_model = AutoModelForCausalLM.from_pretrained(
        str(model_path),
        torch_dtype=dtype,
        local_files_only=True,
    )
    hf_model.to(device)
    hf_model.eval()
    return resolved_model, tl_model_name, hf_model, tokenizer


def infer_transformer_lens_model_name(model_path: Path) -> str:
    config_path = model_path / "config.json"
    if config_path.exists():
        try:
            import json

            with config_path.open(encoding="utf-8") as handle:
                config = json.load(handle)
            name_or_path = str(config.get("_name_or_path") or "")
            if name_or_path == "google/gemma-2-2b-it":
                return "google/gemma-2-2b-it"
        except Exception:
            pass

    if model_path.name == "gemma-2-2b-it":
        return "google/gemma-2-2b-it"
    return str(model_path)


def resolve_local_transcoder_config_path(transcoder_set: str) -> Path | None:
    path = Path(transcoder_set)
    candidates = [path] if path.is_absolute() else [REPO_ROOT / path, S3_ROOT / path, path]
    for candidate in candidates:
        if candidate.is_file() and candidate.suffix.lower() in {".yaml", ".yml"}:
            return candidate.resolve()
    return None


def load_local_transcoders(
    config_path: Path,
    dtype: torch.dtype,
    device: torch.device,
    lazy_encoder: bool,
    lazy_decoder: bool,
) -> Any:
    ensure_codecircuit_on_path()
    from circuit_tracer.utils.hf_utils import load_transcoders

    with config_path.open(encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError(f"Local transcoder config must be a mapping: {config_path}")

    config = dict(config)
    config["scan"] = config.get("scan") or f"local:{config_path}"
    config["repo_id"] = config.get("repo_id") or "local/gemma-scope-transcoders"
    config["revision"] = config.get("revision")
    if "transcoders" in config:
        config["transcoders"] = [
            _resolve_local_transcoder_path(config_path.parent, item)
            for item in config["transcoders"]
        ]

    return load_transcoders(
        config,
        device=device,
        dtype=dtype,
        lazy_encoder=lazy_encoder,
        lazy_decoder=lazy_decoder,
    )


def _resolve_local_transcoder_path(config_dir: Path, item: Any) -> Any:
    if not isinstance(item, str) or item.startswith("hf://"):
        return item
    item_path = Path(item)
    if item_path.is_absolute():
        return str(item_path)
    return str((config_dir / item_path).resolve())


def load_generation_records(path: str | Path) -> dict[str, JsonRecord]:
    records: dict[str, JsonRecord] = {}
    for record in iter_jsonl(path):
        if record.get("status") == "ok":
            records[str(record["example_id"])] = record
    return records


def _graph_output_path(graph_dir: str | Path, example_id: str) -> Path:
    return Path(graph_dir) / f"{example_id}.pt"


def _payload_output_path(payload_dir: str | Path, example_id: str) -> Path:
    return Path(payload_dir) / f"{example_id}.payload.pt"


def _atomic_graph_save(graph: Any, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(path.name + ".tmp")
    graph.to_pt(str(tmp_path))
    tmp_path.replace(path)
    Path(str(path) + ".done").write_text("ok\n", encoding="utf-8")


def normalize_oom_retry_batch_sizes(
    primary_batch_size: int,
    retry_batch_sizes: Iterable[int] | None,
) -> list[int]:
    """Return unique positive retry batch sizes, starting with the primary size."""

    if primary_batch_size <= 0:
        raise ValueError("primary_batch_size must be positive")

    normalized = [int(primary_batch_size)]
    seen = {int(primary_batch_size)}
    for value in retry_batch_sizes or []:
        value = int(value)
        if value <= 0 or value in seen:
            continue
        seen.add(value)
        normalized.append(value)
    return normalized


def _normalize_offload(value: str | None) -> str | None:
    if value is None or value == "none":
        return None
    return value


def _build_oom_retry_attempts(
    *,
    batch_size: int,
    retry_batch_sizes: Iterable[int] | None,
    offload: str | None,
    oom_retry_offload: str | None,
) -> list[dict[str, Any]]:
    batch_sizes = normalize_oom_retry_batch_sizes(batch_size, retry_batch_sizes)
    primary_offload = _normalize_offload(offload)
    retry_offload = _normalize_offload(oom_retry_offload)

    attempts: list[dict[str, Any]] = []
    seen: set[tuple[int, str | None]] = set()
    for candidate_offload in [primary_offload, retry_offload]:
        if candidate_offload is None and attempts:
            continue
        for candidate_batch_size in batch_sizes:
            key = (candidate_batch_size, candidate_offload)
            if key in seen:
                continue
            seen.add(key)
            attempts.append(
                {
                    "batch_size": candidate_batch_size,
                    "offload": candidate_offload,
                }
            )
    return attempts


def _is_cuda_oom(error: Exception) -> bool:
    if isinstance(error, torch.cuda.OutOfMemoryError):
        return True
    return "out of memory" in str(error).lower() and "cuda" in str(error).lower()


def _clear_cuda_cache(device: str | None) -> None:
    if torch.cuda.is_available() and (device is None or str(device).startswith("cuda")):
        torch.cuda.empty_cache()


def _cleanup_partial_outputs(*paths: Path | None) -> None:
    for path in paths:
        if path is None:
            continue
        for candidate in [
            path.with_name(path.name + ".tmp"),
            Path(str(path) + ".tmp"),
        ]:
            try:
                if candidate.exists():
                    candidate.unlink()
            except OSError:
                pass


def _metadata_record(
    generation: JsonRecord,
    graph_path: Path | None,
    status: str,
    payload_path: Path | None = None,
    trace_tokens: int | None = None,
    error: Exception | None = None,
    params: dict[str, Any] | None = None,
) -> JsonRecord:
    record: JsonRecord = {
        "row_index": generation.get("row_index"),
        "example_id": generation.get("example_id"),
        "dataset": generation.get("dataset"),
        "split": generation.get("split"),
        "trace_type": generation.get("trace_type"),
        "graph_path": str(graph_path.resolve()) if graph_path is not None else None,
        "payload_path": str(payload_path.resolve()) if payload_path is not None else None,
        "prompt_tokens": generation.get("prompt_tokens"),
        "completion_tokens": generation.get("completion_tokens"),
        "trace_tokens": trace_tokens,
        "status": status,
    }
    for key in [
        "source_corpus",
        "source",
        "source_row_id",
        "source_row_index",
        "source_index",
        "source_path",
        "prompt_protocol",
        "condition_key",
        "condition_text_hash",
        "generation_prompt_hash",
        "choice_count",
        "choice_labels",
        "audit_full_graph",
    ]:
        if key in generation:
            record[key] = generation[key]
    if params:
        record.update(params)
    if error is not None:
        record["error_type"] = type(error).__name__
        record["error_message"] = str(error)
    return record


def extract_graphs(
    generation_path: str | Path,
    graph_dir: str | Path,
    metadata_path: str | Path,
    model_name_or_path: str,
    payload_dir: str | Path | None = None,
    transcoder_set: str = "gemma",
    dtype_name: str = "bfloat16",
    device: str | None = None,
    max_n_logits: int = 10,
    desired_logit_prob: float = 0.95,
    max_feature_nodes: int = 8192,
    batch_size: int = 64,
    node_threshold: float = 0.8,
    save_payload: bool = False,
    save_full_graph: bool = True,
    full_graph_example_ids: set[str] | None = None,
    offload: str | None = None,
    oom_retry_batch_sizes: Iterable[int] | None = None,
    oom_retry_offload: str | None = None,
    verbose: bool = False,
    lazy_encoder: bool = False,
    lazy_decoder: bool = True,
    resume: bool = True,
) -> int:
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    graph_dir = Path(graph_dir)
    graph_dir.mkdir(parents=True, exist_ok=True)
    payload_dir_path = Path(payload_dir) if save_payload and payload_dir is not None else None
    if save_payload:
        if payload_dir_path is None:
            raise ValueError("payload_dir is required when save_payload=True")
        payload_dir_path.mkdir(parents=True, exist_ok=True)
    if not save_payload and not save_full_graph:
        raise ValueError("At least one of save_payload or save_full_graph must be enabled")
    completed = existing_example_ids(metadata_path) if resume else set()

    model = load_replacement_model(
        model_name_or_path=model_name_or_path,
        transcoder_set=transcoder_set,
        dtype_name=dtype_name,
        device=device,
        lazy_encoder=lazy_encoder,
        lazy_decoder=lazy_decoder,
    )
    ensure_codecircuit_on_path()
    from circuit_tracer import attribute

    params = {
        "transcoder_set": transcoder_set,
        "transcoder_scan": model.scan,
        "feature_input_hook": model.feature_input_hook,
        "feature_output_hook": model.feature_output_hook,
        "dtype": dtype_name,
        "max_n_logits": max_n_logits,
        "desired_logit_prob": desired_logit_prob,
        "max_feature_nodes": max_feature_nodes,
        "batch_size": batch_size,
        "node_threshold": node_threshold,
        "save_payload": save_payload,
        "save_full_graph": save_full_graph,
        "selective_full_graph": full_graph_example_ids is not None,
        "full_graph_example_ids_count": len(full_graph_example_ids) if full_graph_example_ids is not None else None,
        "offload": offload,
        "oom_retry_batch_sizes": normalize_oom_retry_batch_sizes(batch_size, oom_retry_batch_sizes),
        "oom_retry_offload": _normalize_offload(oom_retry_offload),
    }
    written = 0
    for generation in tqdm(iter_jsonl(generation_path), desc="Extracting graphs"):
        example_id = str(generation["example_id"])
        if example_id in completed:
            continue
        should_save_full_graph = save_full_graph and (
            full_graph_example_ids is None or example_id in full_graph_example_ids
        )
        graph_path = _graph_output_path(graph_dir, example_id) if should_save_full_graph else None
        payload_path = (
            _payload_output_path(payload_dir_path, example_id)
            if payload_dir_path is not None
            else None
        )
        row_params = dict(params)
        row_params["save_full_graph"] = should_save_full_graph
        row_params["full_graph_saved"] = False
        row_params["payload_saved"] = False
        if generation.get("status") != "ok":
            append_jsonl(
                metadata_path,
                _metadata_record(
                    generation,
                    graph_path,
                    "skipped_generation_failed",
                    payload_path=payload_path,
                    params=row_params,
                ),
            )
            written += 1
            continue
        full_graph_done = (not should_save_full_graph) or (
            graph_path is not None and graph_path.exists()
        )
        payload_done = (not save_payload) or (
            payload_path is not None and payload_path.exists()
        )
        if full_graph_done and payload_done:
            row_params["full_graph_saved"] = bool(should_save_full_graph)
            row_params["payload_saved"] = bool(save_payload)
            append_jsonl(
                metadata_path,
                _metadata_record(
                    generation,
                    graph_path,
                    "skipped_existing",
                    payload_path=payload_path,
                    params=row_params,
                ),
            )
            written += 1
            continue

        trace_text = build_trace_text(
            str(generation["prompt_text"]),
            str(generation["generated_answer"]),
        )
        attempts = _build_oom_retry_attempts(
            batch_size=batch_size,
            retry_batch_sizes=oom_retry_batch_sizes,
            offload=offload,
            oom_retry_offload=oom_retry_offload,
        )
        failed_attempts: list[dict[str, Any]] = []
        final_error: Exception | None = None
        final_params = dict(row_params)
        for attempt_index, attempt in enumerate(attempts):
            graph: Any | None = None
            attempt_params = dict(row_params)
            attempt_params["effective_batch_size"] = int(attempt["batch_size"])
            attempt_params["effective_offload"] = attempt["offload"]
            attempt_params["oom_retry_attempt_index"] = attempt_index
            attempt_params["oom_retry_attempts"] = attempts
            attempt_params["oom_failed_attempts"] = failed_attempts
            final_params = attempt_params
            try:
                graph = attribute(
                    prompt=trace_text,
                    model=model,
                    max_n_logits=max_n_logits,
                    desired_logit_prob=desired_logit_prob,
                    batch_size=int(attempt["batch_size"]),
                    max_feature_nodes=max_feature_nodes,
                    offload=attempt["offload"],
                    verbose=verbose,
                )
                if should_save_full_graph and graph_path is not None:
                    _atomic_graph_save(graph, graph_path)
                    attempt_params["full_graph_saved"] = True
                if save_payload and payload_path is not None:
                    from s3circuit.compact_payload import save_compact_payload

                    save_compact_payload(graph, payload_path, node_threshold=node_threshold)
                    attempt_params["payload_saved"] = True
                append_jsonl(
                    metadata_path,
                    _metadata_record(
                        generation,
                        graph_path,
                        "ok",
                        payload_path=payload_path,
                        trace_tokens=graph.n_pos,
                        params=attempt_params,
                    ),
                )
                written += 1
                final_error = None
                break
            except Exception as error:
                final_error = error
                failed_attempts.append(
                    {
                        "batch_size": int(attempt["batch_size"]),
                        "offload": attempt["offload"],
                        "error_type": type(error).__name__,
                        "error_message": str(error),
                    }
                )
                if _is_cuda_oom(error) and attempt_index + 1 < len(attempts):
                    _cleanup_partial_outputs(graph_path, payload_path)
                    _clear_cuda_cache(device)
                    continue
                final_params["oom_failed_attempts"] = failed_attempts
                append_jsonl(
                    metadata_path,
                    _metadata_record(
                        generation,
                        graph_path,
                        "failed",
                        payload_path=payload_path,
                        error=error,
                        params=final_params,
                    ),
                )
                written += 1
                break
            finally:
                if graph is not None:
                    del graph
                _clear_cuda_cache(device)
        if final_error is not None:
            _cleanup_partial_outputs(graph_path, payload_path)
    return written
