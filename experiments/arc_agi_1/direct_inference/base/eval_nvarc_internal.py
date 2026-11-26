#!/usr/bin/env python3
"""Run a frozen, no-TTFT NVARC internal ARC evaluator."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import sys
import time
from pathlib import Path, PureWindowsPath
from typing import Any, Callable, Sequence

from arc_eval_protocol import (
    ARC_AGI1_INTERNAL_CONFIG_SHA256,
    ARC_AGI1_INTERNAL_PROTOCOL_VERSION,
    ARC_AGI1_PUBLIC_VALIDATION_CONFIG_SHA256,
    ARC_AGI1_PUBLIC_VALIDATION_PROTOCOL_VERSION,
    INTERNAL_PROTOCOL_VERSION,
    INTERNAL_CONFIG_SHA256,
    ViewPrediction,
    build_author_view_specs,
    build_submission,
    canonical_json_bytes,
    canonical_view_spec,
    invert_grid,
    load_challenges,
    load_frozen_config,
    load_solutions,
    rank_view_predictions,
    score_attempts,
    sha256_file,
    summarize_view_counts,
    transform_test_view,
    truncate_prompt_to_tokens,
    validate_arc_agi1_identity,
    validate_arc_agi2_identity,
)
from nvarc_grid_grammar import GridGrammarState, decode_grid_tokens


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument(
        "--adapter",
        type=Path,
        help="Optional native HyperCombine LoRA checkpoint directory.",
    )
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--challenges", type=Path, required=True)
    parser.add_argument("--solutions", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model-label", required=True)
    parser.add_argument("--dtype", choices=("bfloat16", "float16"), default="bfloat16")
    parser.add_argument("--attn-implementation", default="flash_attention_2")
    return parser.parse_args()


def _load_eval_protocol(
    config_path: Path,
) -> tuple[dict[str, Any], str, str, Callable[..., dict[str, Any]]]:
    raw = _read_json_object(config_path, "internal eval config")
    protocol_version = raw.get("protocol_version")
    if protocol_version == INTERNAL_PROTOCOL_VERSION:
        expected_sha256 = INTERNAL_CONFIG_SHA256
        identity_validator = validate_arc_agi2_identity
    elif protocol_version == ARC_AGI1_INTERNAL_PROTOCOL_VERSION:
        expected_sha256 = ARC_AGI1_INTERNAL_CONFIG_SHA256
        identity_validator = validate_arc_agi1_identity
    elif protocol_version == ARC_AGI1_PUBLIC_VALIDATION_PROTOCOL_VERSION:
        expected_sha256 = ARC_AGI1_PUBLIC_VALIDATION_CONFIG_SHA256
        identity_validator = validate_arc_agi1_identity
    else:
        raise ValueError(
            f"Unsupported internal eval protocol_version={protocol_version!r}"
        )
    config, config_sha256 = load_frozen_config(
        config_path,
        protocol_version,
        expected_sha256,
    )
    return config, config_sha256, protocol_version, identity_validator


def _read_json_object(path: Path, label: str) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Missing {label}: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected {label} to contain a JSON object: {path}")
    return payload


def _manifest_path(path: Path) -> str:
    """Record the path from the launch workspace, preserving local link names."""
    return Path(os.path.relpath(path, Path.cwd())).as_posix()


def _hash_artifact_set(root: Path, paths: Sequence[Path]) -> dict[str, Any]:
    root = root.resolve()
    unique_paths = {path.resolve() for path in paths}
    files: dict[str, dict[str, Any]] = {}
    for path in sorted(unique_paths, key=lambda item: item.name):
        if not path.is_file():
            raise FileNotFoundError(f"Missing identity artifact: {path}")
        files[path.name] = {
            "sha256": sha256_file(path),
            "size_bytes": path.stat().st_size,
        }
    aggregate_sha256 = hashlib.sha256(
        canonical_json_bytes({"files": files})
    ).hexdigest()
    return {
        "root": str(root),
        "files": files,
        "aggregate_sha256": aggregate_sha256,
    }


def hash_hf_model_artifacts(model_dir: Path) -> dict[str, Any]:
    """Hash the model config, weight index, and every local model shard."""

    model_dir = model_dir.resolve()
    config_path = model_dir / "config.json"
    if not config_path.is_file():
        raise FileNotFoundError(f"Missing HF model config: {config_path}")

    artifacts = [config_path]
    generation_config = model_dir / "generation_config.json"
    if generation_config.is_file():
        artifacts.append(generation_config)

    weight_paths: set[Path] = set()
    for index_name in (
        "model.safetensors.index.json",
        "pytorch_model.bin.index.json",
    ):
        index_path = model_dir / index_name
        if not index_path.is_file():
            continue
        artifacts.append(index_path)
        index_payload = _read_json_object(index_path, "HF weight index")
        weight_map = index_payload.get("weight_map")
        if not isinstance(weight_map, dict) or not weight_map:
            raise ValueError(f"HF weight index has no weight_map: {index_path}")
        for filename in set(weight_map.values()):
            if not isinstance(filename, str) or not filename:
                raise ValueError(f"Invalid shard name in {index_path}: {filename!r}")
            shard = model_dir / filename
            if not shard.is_file():
                raise FileNotFoundError(f"HF weight index references missing shard: {shard}")
            weight_paths.add(shard)

    weight_paths.update(model_dir.glob("*.safetensors"))
    weight_paths.update(model_dir.glob("pytorch_model*.bin"))
    if not weight_paths:
        raise FileNotFoundError(f"No HF model weight files found in {model_dir}")
    artifacts.extend(sorted(weight_paths))
    identity = _hash_artifact_set(model_dir, artifacts)
    identity["weight_file_count"] = len(weight_paths)
    return identity


def load_lora_adapter_metadata(adapter_dir: Path) -> dict[str, Any]:
    """Validate and normalize the metadata needed to reconstruct training LoRA."""

    adapter_dir = adapter_dir.resolve()
    adapter_file = adapter_dir / "adapter_model.safetensors"
    if not adapter_file.is_file():
        raise FileNotFoundError(f"Missing LoRA adapter weights: {adapter_file}")
    config = _read_json_object(adapter_dir / "adapter_config.json", "LoRA adapter config")
    manifest = _read_json_object(
        adapter_dir / "hypercombine_adapter_manifest.json",
        "HyperCombine adapter manifest",
    )

    required_config = {
        "peft_type": "LORA",
        "task_type": "CAUSAL_LM",
        "bias": "none",
        "inference_mode": True,
        "use_dora": False,
        "use_rslora": False,
    }
    for key, expected in required_config.items():
        if config.get(key) != expected:
            raise ValueError(
                f"Unsupported adapter config {key}={config.get(key)!r}; expected {expected!r}"
            )

    rank = config.get("r")
    alpha = config.get("lora_alpha")
    dropout = config.get("lora_dropout")
    if not isinstance(rank, int) or isinstance(rank, bool) or rank <= 0:
        raise ValueError(f"Invalid LoRA rank: {rank!r}")
    if not isinstance(alpha, int) or isinstance(alpha, bool) or alpha <= 0:
        raise ValueError(f"Invalid LoRA alpha: {alpha!r}")
    if not isinstance(dropout, (int, float)) or isinstance(dropout, bool) or dropout < 0:
        raise ValueError(f"Invalid LoRA dropout: {dropout!r}")

    config_targets = config.get("target_modules")
    manifest_targets = manifest.get("target_modules")
    if not isinstance(config_targets, list) or not config_targets:
        raise ValueError("adapter_config.json target_modules must be a non-empty list")
    if not isinstance(manifest_targets, list) or not manifest_targets:
        raise ValueError("HyperCombine manifest target_modules must be a non-empty list")
    if any(not isinstance(target, str) or not target for target in config_targets):
        raise ValueError("adapter_config.json contains an invalid target module")
    if any(not isinstance(target, str) or not target for target in manifest_targets):
        raise ValueError("HyperCombine manifest contains an invalid target module")
    if len(set(config_targets)) != len(config_targets):
        raise ValueError("adapter_config.json contains duplicate target modules")
    if len(set(manifest_targets)) != len(manifest_targets):
        raise ValueError("HyperCombine manifest contains duplicate target modules")
    if set(config_targets) != set(manifest_targets):
        raise ValueError("Adapter config and HyperCombine manifest target modules differ")

    raw_groups = manifest.get("shared_A_groups")
    if not isinstance(raw_groups, list):
        raise ValueError("HyperCombine manifest shared_A_groups must be a list")
    shared_groups: list[tuple[str, ...]] = []
    grouped_targets: set[str] = set()
    for raw_group in raw_groups:
        if not isinstance(raw_group, list) or not raw_group:
            raise ValueError("Each shared_A group must be a non-empty list")
        group = tuple(raw_group)
        if any(not isinstance(target, str) or not target for target in group):
            raise ValueError("A shared_A group contains an invalid target module")
        if len(set(group)) != len(group):
            raise ValueError("A shared_A group contains duplicate target modules")
        if grouped_targets.intersection(group):
            raise ValueError("A target module appears in more than one shared_A group")
        if not set(group).issubset(manifest_targets):
            raise ValueError("A shared_A group references a non-target module")
        grouped_targets.update(group)
        shared_groups.append(group)

    expected_parameterization = (
        "fused_projection_equivalent_shared_A"
        if shared_groups
        else "independent_per_projection"
    )
    if manifest.get("training_parametrization") != expected_parameterization:
        raise ValueError(
            "HyperCombine manifest training_parametrization does not match shared_A_groups"
        )
    if manifest.get("format_version") != 1:
        raise ValueError(f"Unsupported adapter manifest format: {manifest.get('format_version')!r}")
    if manifest.get("peft_inference_compatible") is not True:
        raise ValueError("Adapter is not marked PEFT-inference-compatible")
    wrapped_module_count = manifest.get("wrapped_module_count")
    trainable_parameters = manifest.get("trainable_parameters")
    if not isinstance(wrapped_module_count, int) or wrapped_module_count <= 0:
        raise ValueError(f"Invalid wrapped_module_count: {wrapped_module_count!r}")
    if not isinstance(trainable_parameters, int) or trainable_parameters <= 0:
        raise ValueError(f"Invalid trainable_parameters: {trainable_parameters!r}")

    base_model_name = config.get("base_model_name_or_path")
    if not isinstance(base_model_name, str) or not base_model_name:
        raise ValueError("Adapter config has no base_model_name_or_path")
    trainer_state_path = adapter_dir / "trainer_state.json"
    completed_steps = None
    if trainer_state_path.is_file():
        trainer_state = _read_json_object(trainer_state_path, "LoRA trainer state")
        completed_steps = trainer_state.get("completed_steps")
        if not isinstance(completed_steps, int) or completed_steps < 0:
            raise ValueError(f"Invalid trainer_state completed_steps: {completed_steps!r}")

    return {
        "path": str(adapter_dir),
        "adapter_file": str(adapter_file),
        "base_model_name_or_path": base_model_name,
        "rank": rank,
        "alpha": alpha,
        "dropout": float(dropout),
        "target_modules": tuple(sorted(config_targets)),
        "shared_A_groups": tuple(shared_groups),
        "training_parametrization": expected_parameterization,
        "wrapped_module_count": wrapped_module_count,
        "trainable_parameters": trainable_parameters,
        "completed_steps": completed_steps,
    }


def _validate_lora_tensor_keys(adapter_file: Path, module_names: Sequence[str]) -> int:
    from safetensors import safe_open

    expected = {
        f"base_model.model.{name}.{suffix}"
        for name in module_names
        for suffix in ("lora_A.weight", "lora_B.weight")
    }
    with safe_open(str(adapter_file), framework="pt", device="cpu") as handle:
        actual = set(handle.keys())
    if actual != expected:
        missing = sorted(expected - actual)
        unexpected = sorted(actual - expected)
        raise ValueError(
            "LoRA adapter tensor keys do not match the injected model: "
            f"missing={missing[:1]} unexpected={unexpected[:1]} "
            f"expected_count={len(expected)} actual_count={len(actual)}"
        )
    return len(actual)


def attach_native_lora_adapter(
    model: Any,
    adapter_dir: Path,
    torch_module: Any,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Recreate the native training parameterization and load its adapter."""

    from ddp_lora import (
        inject_lora,
        iter_lora_modules,
        load_adapter,
        trainable_parameter_summary,
    )

    metadata = metadata or load_lora_adapter_metadata(adapter_dir)
    replaced = inject_lora(
        model,
        rank=metadata["rank"],
        alpha=metadata["alpha"],
        dropout=metadata["dropout"],
        target_modules=metadata["target_modules"],
        shared_A_groups=metadata["shared_A_groups"],
        adapter_dtype=torch_module.float32,
    )
    if len(replaced) != metadata["wrapped_module_count"]:
        raise ValueError(
            "Injected LoRA module count does not match checkpoint manifest: "
            f"{len(replaced)} != {metadata['wrapped_module_count']}"
        )
    tensor_count = _validate_lora_tensor_keys(
        Path(metadata["adapter_file"]),
        replaced,
    )
    load_adapter(model, Path(metadata["adapter_file"]))
    parameter_summary = trainable_parameter_summary(model)
    if parameter_summary["trainable_parameters"] != metadata["trainable_parameters"]:
        raise ValueError(
            "Injected LoRA trainable parameter count does not match checkpoint manifest: "
            f"{parameter_summary['trainable_parameters']} != {metadata['trainable_parameters']}"
        )
    adapter_dtypes = sorted(
        {
            str(parameter.dtype)
            for _, module in iter_lora_modules(model)
            for parameter in (module.lora_A, module.lora_B)
        }
    )
    if adapter_dtypes != [str(torch_module.float32)]:
        raise ValueError(f"Expected FP32 native LoRA parameters, found {adapter_dtypes}")
    model.requires_grad_(False)
    return {
        "wrapped_module_count": len(replaced),
        "adapter_tensor_count": tensor_count,
        "trainable_parameters_before_eval_freeze": parameter_summary[
            "trainable_parameters"
        ],
        "adapter_parameter_dtypes": adapter_dtypes,
    }


def hash_lora_adapter_artifacts(adapter_dir: Path) -> dict[str, Any]:
    metadata = load_lora_adapter_metadata(adapter_dir)
    adapter_dir = adapter_dir.resolve()
    paths = [
        adapter_dir / "adapter_model.safetensors",
        adapter_dir / "adapter_config.json",
        adapter_dir / "hypercombine_adapter_manifest.json",
    ]
    trainer_state = adapter_dir / "trainer_state.json"
    if trainer_state.is_file():
        paths.append(trainer_state)
    identity = _hash_artifact_set(adapter_dir, paths)
    identity["metadata"] = {
        key: value
        for key, value in metadata.items()
        if key not in {"path", "adapter_file"}
    }
    identity["optimizer_excluded_from_inference_identity"] = True
    return identity


def _distributed_context(torch_module: Any) -> tuple[int, int, int]:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if world_size > 1:
        torch_module.distributed.init_process_group(backend="nccl")
    torch_module.cuda.set_device(local_rank)
    return rank, local_rank, world_size


def _barrier(torch_module: Any, world_size: int) -> None:
    if world_size > 1:
        torch_module.distributed.barrier()


def _verify_token_contract(tokenizer: Any, model: Any) -> dict[str, Any]:
    expected_literals = {str(value): value for value in range(10)}
    expected_literals["\n"] = 10
    expected_literals["user"] = 11
    expected_literals["assistant"] = 12
    literal_ids: dict[str, int] = {}
    for literal, expected_id in expected_literals.items():
        encoded = tokenizer.encode(literal, add_special_tokens=False)
        if encoded != [expected_id]:
            raise ValueError(f"Tokenizer contract mismatch for {literal!r}: {encoded} != {[expected_id]}")
        literal_ids[literal] = encoded[0]

    expected_specials = {
        "<|im_start|>": 14,
        "<|im_end|>": 15,
    }
    special_ids = {}
    for token, expected_id in expected_specials.items():
        token_id = tokenizer.convert_tokens_to_ids(token)
        if token_id != expected_id:
            raise ValueError(f"Tokenizer contract mismatch for {token}: {token_id} != {expected_id}")
        special_ids[token] = token_id
    if tokenizer.pad_token_id != 13:
        raise ValueError(f"Expected pad_token_id=13, found {tokenizer.pad_token_id}")
    if tokenizer.eos_token_id != 15:
        raise ValueError(f"Expected eos_token_id=15, found {tokenizer.eos_token_id}")
    if int(model.config.vocab_size) != 16:
        raise ValueError(f"Expected 16-token model vocab, found {model.config.vocab_size}")
    return {
        "digit_token_ids": [literal_ids[str(value)] for value in range(10)],
        "newline_token_id": literal_ids["\n"],
        "user_token_id": literal_ids["user"],
        "assistant_token_id": literal_ids["assistant"],
        "pad_token_id": tokenizer.pad_token_id,
        "im_start_token_id": special_ids["<|im_start|>"],
        "eos_token_id": special_ids["<|im_end|>"],
        "vocab_size": int(model.config.vocab_size),
    }


def _constrained_greedy_batch(
    model: Any,
    tokenizer: Any,
    prompts: Sequence[str],
    token_contract: dict[str, Any],
    max_new_tokens: int,
    device: Any,
    torch_module: Any,
) -> list[dict[str, Any]]:
    tokenizer.padding_side = "left"
    encoded = tokenizer(
        list(prompts),
        add_special_tokens=False,
        padding=True,
        return_tensors="pt",
    )
    input_ids = encoded["input_ids"].to(device)
    attention_mask = encoded["attention_mask"].to(device)
    states = [GridGrammarState() for _ in prompts]
    generated: list[list[int]] = [[] for _ in prompts]
    logprob_sum = [0.0 for _ in prompts]
    token_count = [0 for _ in prompts]
    digit_ids = tuple(token_contract["digit_token_ids"])
    newline_id = int(token_contract["newline_token_id"])
    eos_id = int(token_contract["eos_token_id"])
    pad_id = int(token_contract["pad_token_id"])

    with torch_module.inference_mode():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, use_cache=True)
        logits = outputs.logits[:, -1, :]
        past_key_values = outputs.past_key_values
        for _ in range(max_new_tokens):
            log_probs = torch_module.nn.functional.log_softmax(logits.float(), dim=-1)
            active_before = [not state.finished for state in states]
            next_tokens: list[int] = []
            for batch_index, state in enumerate(states):
                if state.finished:
                    next_tokens.append(pad_id)
                    continue
                allowed = state.allowed_token_ids(digit_ids, newline_id, eos_id)
                if not allowed:
                    raise RuntimeError(f"Grid grammar reached a dead end: {state}")
                allowed_tensor = torch_module.tensor(allowed, device=device, dtype=torch_module.long)
                allowed_scores = log_probs[batch_index].index_select(0, allowed_tensor)
                selected_offset = int(torch_module.argmax(allowed_scores).item())
                selected_token = int(allowed[selected_offset])
                logprob_sum[batch_index] += float(log_probs[batch_index, selected_token].item())
                token_count[batch_index] += 1
                generated[batch_index].append(selected_token)
                state.consume(selected_token, digit_ids, newline_id, eos_id)
                next_tokens.append(selected_token)
            if all(state.finished for state in states):
                break
            next_tensor = torch_module.tensor(next_tokens, device=device, dtype=torch_module.long).unsqueeze(1)
            new_mask = torch_module.tensor(
                active_before,
                device=device,
                dtype=attention_mask.dtype,
            ).unsqueeze(1)
            attention_mask = torch_module.cat((attention_mask, new_mask), dim=1)
            outputs = model(
                input_ids=next_tensor,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
                use_cache=True,
            )
            logits = outputs.logits[:, -1, :]
            past_key_values = outputs.past_key_values

    digit_by_id = {token_id: value for value, token_id in enumerate(digit_ids)}
    results = []
    for index, tokens in enumerate(generated):
        grid = decode_grid_tokens(tokens, digit_by_id, newline_id, eos_id)
        results.append(
            {
                "token_ids": tokens,
                "grid": grid,
                "normalized_logprob": (
                    logprob_sum[index] / token_count[index] if token_count[index] else float("-inf")
                ),
                "new_tokens": token_count[index],
                "finished": states[index].finished,
            }
        )
    return results


def _write_json(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes(canonical_json_bytes(value) + b"\n")
    os.replace(temporary, path)


def _write_jsonl(path: Path, records: Sequence[dict[str, Any]]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=True, sort_keys=True, separators=(",", ":")))
            handle.write("\n")
    os.replace(temporary, path)


def _merge_and_score(
    args: argparse.Namespace,
    config: dict[str, Any],
    config_sha256: str,
    challenge_identity: dict[str, Any],
    token_contract: dict[str, Any],
    world_size: int,
    started_at: float,
    torch_module: Any,
    transformers_module: Any,
    adapter_runtime: dict[str, Any] | None,
    protocol_version: str,
    identity_validator: Callable[..., dict[str, Any]],
) -> None:
    records = []
    for rank in range(world_size):
        shard = args.output_dir / f"predictions.rank{rank}.jsonl"
        if not shard.is_file():
            raise FileNotFoundError(f"Missing prediction shard: {shard}")
        with shard.open("r", encoding="utf-8") as handle:
            records.extend(json.loads(line) for line in handle if line.strip())
    records.sort(key=lambda record: record["global_test_index"])
    if [record["global_test_index"] for record in records] != list(range(config["dataset"]["test_case_count"])):
        raise ValueError("Prediction shards do not contain each frozen test case exactly once")

    challenges = load_challenges(args.challenges)
    solutions = load_solutions(args.solutions, challenges)
    full_identity = identity_validator(challenges, solutions)
    if full_identity["challenge_semantic_sha256"] != challenge_identity["challenge_semantic_sha256"]:
        raise ValueError("Challenge data changed between generation and scoring")

    canonical_attempts = {}
    selected_attempts = {}
    oracle_attempts = {}
    valid_views = 0
    total_views = 0
    unique_candidates = []
    for record in records:
        key = (record["task_id"], record["test_index"])
        canonical = record["canonical_prediction"]
        canonical_attempts[key] = [canonical] if canonical is not None else []
        selected_attempts[key] = [candidate["grid"] for candidate in record["selected_candidates"]]
        oracle_attempts[key] = [view["grid"] for view in record["selection_views"] if view["grid"] is not None]
        valid_views += record["view_summary"]["valid_view_count"]
        total_views += record["view_summary"]["view_count"]
        unique_candidates.append(record["view_summary"]["unique_candidates"])

    canonical_score = score_attempts(challenges, solutions, canonical_attempts, top_k=1)
    selected_score_1 = score_attempts(challenges, solutions, selected_attempts, top_k=1)
    selected_score_2 = score_attempts(challenges, solutions, selected_attempts, top_k=2)
    oracle_score = score_attempts(challenges, solutions, oracle_attempts, top_k=16)
    metrics = {
        "protocol_version": protocol_version,
        "primary_metric": "selected_exact@2",
        "canonical_exact@1": canonical_score["accuracy"],
        "canonical_task_score@1": canonical_score["task_score"],
        "selected_exact@1": selected_score_1["accuracy"],
        "selected_task_score@1": selected_score_1["task_score"],
        "selected_exact@2": selected_score_2["accuracy"],
        "selected_task_score@2": selected_score_2["task_score"],
        "oracle_view@16": oracle_score["accuracy"],
        "oracle_task_score@16": oracle_score["task_score"],
        "valid_grid_rate": valid_views / total_views,
        "valid_view_count": valid_views,
        "selection_view_count": total_views,
        "mean_unique_candidates": sum(unique_candidates) / len(unique_candidates),
        "elapsed_seconds": time.time() - started_at,
        "score_details": {
            "canonical_exact@1": canonical_score,
            "selected_exact@1": selected_score_1,
            "selected_exact@2": selected_score_2,
            "oracle_view@16": oracle_score,
        },
    }
    submission = build_submission(challenges, selected_attempts)
    _write_json(args.output_dir / "metrics.json", metrics)
    _write_json(args.output_dir / "submission.json", submission)
    _write_jsonl(args.output_dir / "predictions.jsonl", records)

    model_config_path = args.model / "config.json"
    tokenizer_config_path = args.tokenizer / "tokenizer_config.json"
    print("rank=0 phase=artifact_hashing_start", flush=True)
    model_artifacts = hash_hf_model_artifacts(args.model)
    model_artifacts["root"] = _manifest_path(args.model)
    adapter_artifacts = None
    if args.adapter is not None:
        adapter_artifacts = hash_lora_adapter_artifacts(args.adapter)
        adapter_artifacts["root"] = _manifest_path(args.adapter)
        metadata = adapter_artifacts["metadata"]
        # A checkpoint can carry an absolute base path from another machine.
        # The evaluator has already checked its model name against --model.
        if PureWindowsPath(metadata["base_model_name_or_path"]).anchor:
            metadata["base_model_name_or_path"] = model_artifacts["root"]
    evaluator_artifacts = {
        "eval_nvarc_internal.py": sha256_file(Path(__file__)),
        "arc_eval_protocol.py": sha256_file(Path(__file__).with_name("arc_eval_protocol.py")),
        "nvarc_grid_grammar.py": sha256_file(Path(__file__).with_name("nvarc_grid_grammar.py")),
    }
    if args.adapter is not None:
        evaluator_artifacts["ddp_lora.py"] = sha256_file(
            Path(__file__).with_name("ddp_lora.py")
        )
    print("rank=0 phase=artifact_hashing_complete", flush=True)
    manifest = {
        "protocol_version": protocol_version,
        "config_path": _manifest_path(args.config),
        "config_sha256": config_sha256,
        "model_source": "hf_plus_native_lora" if args.adapter is not None else "hf",
        "model_path": model_artifacts["root"],
        "model_label": args.model_label,
        "tokenizer_path": _manifest_path(args.tokenizer),
        "model_config_sha256": sha256_file(model_config_path),
        "model_artifacts": model_artifacts,
        "tokenizer_config_sha256": sha256_file(tokenizer_config_path),
        "adapter_path": adapter_artifacts["root"] if adapter_artifacts is not None else None,
        "adapter_artifacts": adapter_artifacts,
        "adapter_runtime": adapter_runtime,
        "evaluator_artifacts": evaluator_artifacts,
        "challenge_path": _manifest_path(args.challenges),
        "solution_path": _manifest_path(args.solutions),
        "dataset_identity": full_identity,
        "token_contract": token_contract,
        "world_size": world_size,
        "python_version": platform.python_version(),
        "torch_version": torch_module.__version__,
        "transformers_version": transformers_module.__version__,
        "cuda_version": torch_module.version.cuda,
        "primary_metric": metrics["primary_metric"],
        "metrics_sha256": sha256_file(args.output_dir / "metrics.json"),
        "submission_sha256": sha256_file(args.output_dir / "submission.json"),
        "predictions_sha256": sha256_file(args.output_dir / "predictions.jsonl"),
    }
    _write_json(args.output_dir / "manifest.json", manifest)


def main() -> None:
    args = parse_args()
    started_at = time.time()
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    import torch
    import transformers
    from transformers import AutoModelForCausalLM, AutoTokenizer

    rank, local_rank, world_size = _distributed_context(torch)
    device = torch.device("cuda", local_rank)
    config, config_sha256, protocol_version, identity_validator = (
        _load_eval_protocol(args.config)
    )
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError(f"Internal eval output directory is not empty: {args.output_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    challenges = load_challenges(args.challenges)
    challenge_identity = identity_validator(challenges)
    if challenge_identity["challenge_semantic_sha256"] != config["dataset"]["challenge_semantic_sha256"]:
        raise ValueError("Challenge data does not match the frozen internal config")

    dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float16
    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer,
        local_files_only=True,
        trust_remote_code=False,
    )
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        local_files_only=True,
        trust_remote_code=False,
        torch_dtype=dtype,
        attn_implementation=args.attn_implementation,
        low_cpu_mem_usage=True,
    ).to(device)
    adapter_runtime = None
    if args.adapter is not None:
        adapter_metadata = load_lora_adapter_metadata(args.adapter)
        declared_base_name = Path(adapter_metadata["base_model_name_or_path"]).name
        selected_base_name = args.model.resolve().name
        if declared_base_name != selected_base_name:
            raise ValueError(
                "LoRA adapter base model does not match --model: "
                f"{declared_base_name!r} != {selected_base_name!r}"
            )
        adapter_runtime = attach_native_lora_adapter(
            model,
            args.adapter,
            torch,
            adapter_metadata,
        )
    model.eval()
    token_contract = _verify_token_contract(tokenizer, model)

    test_units = [
        (task_id, test_index)
        for task_id in sorted(challenges)
        for test_index in range(len(challenges[task_id]["test"]))
    ]
    max_input_tokens = (
        int(config["prompt"]["max_sequence_length"])
        - int(config["prompt"]["max_new_tokens"])
    )
    local_records = []
    for global_test_index, (task_id, test_index) in enumerate(test_units):
        if global_test_index % world_size != rank:
            continue
        task = challenges[task_id]
        canonical_spec = canonical_view_spec(len(task["train"]))
        selection_specs = build_author_view_specs(
            len(task["train"]),
            augment_n=int(config["selection_views"]["author_augment_n"]),
            seed=int(config["selection_views"]["seed"]),
        )
        if len(selection_specs) != int(config["selection_views"]["count"]):
            raise ValueError("Frozen selection view count does not match generated views")
        all_specs = [canonical_spec, *selection_specs]
        prompts = []
        prompt_metadata = []
        for spec in all_specs:
            transformed = transform_test_view(task, test_index, spec)
            prompt, prompt_tokens, removed_examples = truncate_prompt_to_tokens(
                transformed,
                lambda text: tokenizer.encode(text, add_special_tokens=False),
                max_input_tokens,
            )
            prompts.append(prompt)
            prompt_metadata.append(
                {
                    "prompt_tokens": prompt_tokens,
                    "removed_train_examples": removed_examples,
                    "view_descriptor": spec.descriptor,
                }
            )
        decoded = _constrained_greedy_batch(
            model,
            tokenizer,
            prompts,
            token_contract,
            int(config["prompt"]["max_new_tokens"]),
            device,
            torch,
        )
        canonical_prediction = decoded[0]["grid"]
        view_predictions = []
        selection_view_records = []
        for spec, prediction, metadata in zip(selection_specs, decoded[1:], prompt_metadata[1:]):
            inverted = invert_grid(prediction["grid"], spec) if prediction["grid"] is not None else None
            view_predictions.append(
                ViewPrediction(
                    view_index=spec.view_index,
                    grid=inverted,
                    normalized_logprob=prediction["normalized_logprob"],
                )
            )
            selection_view_records.append(
                {
                    "view_index": spec.view_index,
                    "view_descriptor": spec.descriptor,
                    "grid": inverted,
                    "normalized_logprob": prediction["normalized_logprob"],
                    "new_tokens": prediction["new_tokens"],
                    "prompt_tokens": metadata["prompt_tokens"],
                    "removed_train_examples": metadata["removed_train_examples"],
                }
            )
        ranked = rank_view_predictions(view_predictions)
        local_records.append(
            {
                "global_test_index": global_test_index,
                "task_id": task_id,
                "test_index": test_index,
                "canonical_prediction": canonical_prediction,
                "canonical_normalized_logprob": decoded[0]["normalized_logprob"],
                "canonical_prompt_tokens": prompt_metadata[0]["prompt_tokens"],
                "canonical_removed_train_examples": prompt_metadata[0]["removed_train_examples"],
                "selection_views": selection_view_records,
                "selected_candidates": [
                    {
                        "grid": candidate.grid,
                        "votes": candidate.votes,
                        "mean_normalized_logprob": candidate.mean_normalized_logprob,
                        "serialized": candidate.serialized,
                    }
                    for candidate in ranked[:2]
                ],
                "view_summary": summarize_view_counts(view_predictions),
            }
        )
        print(
            f"rank={rank} test={global_test_index + 1}/{len(test_units)} "
            f"task={task_id} test_index={test_index} valid={local_records[-1]['view_summary']['valid_view_count']}/16",
            flush=True,
        )

    _write_jsonl(args.output_dir / f"predictions.rank{rank}.jsonl", local_records)
    _barrier(torch, world_size)
    if rank == 0:
        _merge_and_score(
            args,
            config,
            config_sha256,
            challenge_identity,
            token_contract,
            world_size,
            started_at,
            torch,
            transformers,
            adapter_runtime,
            protocol_version,
            identity_validator,
        )
    _barrier(torch, world_size)
    if world_size > 1:
        torch.distributed.destroy_process_group()


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr, flush=True)
        raise
