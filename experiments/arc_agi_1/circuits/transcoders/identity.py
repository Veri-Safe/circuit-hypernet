from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any, Mapping

from .config import LOCKED_RUN_NAME, NvarcTranscoderConfig, PinnedHfBaseConfig


TRACE_GENERATION_CONTRACT = "nvarc_generated_trace_v1"
LEGACY_TRACE_CODE_CONTRACTS = {
    # Validation chunks started from fb37797b. Subsequent changes added
    # snapshot/training utilities and a read-only trace digest; generation
    # semantics and serialized trace bytes remained unchanged.
    "dc60b7657f2a6bce74aabb1766067151195dd374e5f9cc3e8bb597cd0b53893d": (
        TRACE_GENERATION_CONTRACT
    ),
    "728865870d9d067fd2cb24fea598fbd1c7c9e64c9059e5fa424280b38bdcc159": (
        TRACE_GENERATION_CONTRACT
    ),
    "8111a5a667259ec42bd5f22d54aff2dcf2a9eaa81eb030adca1092aae5efa7a0": (
        TRACE_GENERATION_CONTRACT
    ),
    # The outside-RE-ARC128 format dispatch was audited in 3f6e2be9; it
    # changed resume metadata plumbing without changing generated trace bytes.
    "26695aeaa07b273d976144a126663192c0e2e121e58b35f046f83f544ec83b9c": (
        TRACE_GENERATION_CONTRACT
    ),
    # ceee7851's trace code identity preceded a finalizer-only metadata fix;
    # generated trace bytes and the resume contract remain unchanged.
    "949a3d909fea2d48baa54f18a05103fbe6955ba26d011626f4bc6252cb4c8b26": (
        TRACE_GENERATION_CONTRACT
    ),
    # 654f075a's resumable-trace refresh preceded the finalizer identity
    # migration; generated trace bytes and the resume contract remain intact.
    "9abdcee0b79142284d8eaaf367ff89af4d54152bed69253679995dbe8c2a664e": (
        TRACE_GENERATION_CONTRACT
    ),
    # 925c089f was the current identity when the existing outside-RE-ARC128
    # traces were refreshed.  Snapshot identity normalization is a
    # non-generation change, so those persisted compatibility records remain
    # valid after the snapshot reader itself changes.
    "925c089fe37c3e33af57d7f5509980ce4786f9bdb5f95202c3f1ce71354588f0": (
        TRACE_GENERATION_CONTRACT
    ),
}


MODEL_CONTRACT = {
    "model_type": "qwen3",
    "hidden_size": 2560,
    "intermediate_size": 9728,
    "num_hidden_layers": 36,
    "vocab_size": 16,
    "tie_word_embeddings": True,
}


def semantic_sha256(value: Any) -> str:
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def package_code_identity() -> dict[str, Any]:
    package_root = Path(__file__).resolve().parent
    s3_root = package_root.parents[2]
    paths = sorted(package_root.glob("*.py")) + sorted(
        (s3_root / "scripts").glob("*nvarc*.py")
    )
    artifacts = {
        str(path.relative_to(s3_root)).replace("\\", "/"): sha256_file(path)
        for path in paths
    }
    return {
        "format": "nvarc_transcoder_code_identity_v1",
        "artifacts": artifacts,
        "sha256": semantic_sha256(artifacts),
    }


def trace_resume_code_compatibility(
    existing: Mapping[str, Any],
    current: Mapping[str, Any],
) -> dict[str, Any] | None:
    """Allow only audited code-only resume migrations for generated traces."""

    if dict(existing) == dict(current):
        return None
    mismatches = {
        key: {"existing": existing.get(key), "current": current.get(key)}
        for key in sorted(set(existing) | set(current))
        if existing.get(key) != current.get(key)
    }
    non_code = {
        key: value
        for key, value in mismatches.items()
        if key != "code_identity_sha256"
    }
    legacy_sha = existing.get("code_identity_sha256")
    current_sha = current.get("code_identity_sha256")
    legacy_contract = LEGACY_TRACE_CODE_CONTRACTS.get(str(legacy_sha))
    if non_code or legacy_contract != TRACE_GENERATION_CONTRACT:
        raise ValueError(
            "Trace resume identity changed outside an audited compatibility "
            f"migration: {mismatches}"
        )
    if not isinstance(current_sha, str) or len(current_sha) != 64:
        raise ValueError("Current trace code identity is not a SHA256")
    return {
        "format": "nvarc_trace_resume_code_compatibility_v1",
        "trace_generation_contract": TRACE_GENERATION_CONTRACT,
        "legacy_code_identity_sha256": legacy_sha,
        "current_code_identity_sha256": current_sha,
        "reason": "audited_non_generation_code_change",
    }


def trace_resume_compatibility_is_audited(
    persisted: Mapping[str, Any],
    expected: Mapping[str, Any],
) -> bool:
    """Accept a persisted compatibility record from an audited prior refresh.

    A shard can pass through more than one audited code-only refresh.  In that
    case its persisted record may point at an intermediate code identity even
    though the current run has already advanced to a later identity.  The
    intermediate identity must itself be registered as an audited trace
    contract, and all non-identity metadata must still match exactly.
    """

    if dict(persisted) == dict(expected):
        return True
    stable_fields = (
        "format",
        "trace_generation_contract",
        "legacy_code_identity_sha256",
        "reason",
    )
    if any(persisted.get(key) != expected.get(key) for key in stable_fields):
        return False
    prior_code_identity = persisted.get("current_code_identity_sha256")
    return (
        isinstance(prior_code_identity, str)
        and LEGACY_TRACE_CODE_CONTRACTS.get(prior_code_identity)
        == TRACE_GENERATION_CONTRACT
    )


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json_object(path: str | Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return value


def config_semantic_sha256_for_root(
    config: NvarcTranscoderConfig,
    *,
    current_root: str | Path,
    identity_root: str | Path,
) -> str:
    current = Path(current_root).resolve()
    identity = Path(identity_root)

    def rebase(value: Any) -> Any:
        if isinstance(value, dict):
            return {key: rebase(item) for key, item in value.items()}
        if isinstance(value, list):
            return [rebase(item) for item in value]
        if isinstance(value, tuple):
            return tuple(rebase(item) for item in value)
        if not isinstance(value, str):
            return value
        path = Path(value)
        if not path.is_absolute():
            return value
        try:
            relative = path.relative_to(current)
        except ValueError:
            try:
                relative = path.resolve().relative_to(current)
            except ValueError:
                return value
        return str(identity).rstrip("/\\") + "/" + relative.as_posix()

    return semantic_sha256(rebase(config.to_dict()))


def _cached_base_identity_matches_mount_alias(
    config: NvarcTranscoderConfig,
    value: dict[str, Any],
) -> bool:
    root_value = os.environ.get("HC_ROOT")
    if not root_value:
        return False
    current_root = Path(root_value).resolve()
    current_hf = Path(config.base.hf_checkpoint).resolve()
    try:
        hf_relative = current_hf.relative_to(current_root)
    except ValueError:
        return False
    cached_hf = Path(str(value.get("hf_checkpoint", "")))
    if not cached_hf.is_absolute() or tuple(cached_hf.parts[-len(hf_relative.parts) :]) != tuple(
        hf_relative.parts
    ):
        return False
    cached_root = cached_hf
    for _ in hf_relative.parts:
        cached_root = cached_root.parent
    if value.get("config_sha256") != config_semantic_sha256_for_root(
        config,
        current_root=current_root,
        identity_root=cached_root,
    ):
        return False
    expected_identity = {
        "format": (
            "nvarc_rearc_pinned_hf_base_identity_v2"
            if isinstance(config.base, PinnedHfBaseConfig)
            else "nvarc_rearc_step399_base_identity_v1"
        ),
        "run_name": config.base.run_name,
        "source_step": config.base.source_step,
    }
    if isinstance(config.base, PinnedHfBaseConfig):
        expected_identity["hf_aggregate_sha256"] = config.base.aggregate_sha256
    if any(value.get(key) != expected for key, expected in expected_identity.items()):
        return False
    path_fields = {
        "hf_checkpoint": config.base.hf_checkpoint,
        "tokenizer": config.base.tokenizer,
    }
    if not isinstance(config.base, PinnedHfBaseConfig):
        path_fields.update(
            {
                "run_manifest": config.base.run_manifest,
                "megatron_checkpoint": config.base.megatron_checkpoint,
            }
        )
    for key, current_value in path_fields.items():
        try:
            relative = Path(current_value).resolve().relative_to(current_root)
        except ValueError:
            return False
        if str(cached_root / relative) != value.get(key):
            return False
    return True


def _inspect_step399_base(config: NvarcTranscoderConfig) -> dict[str, Any]:
    base = config.base
    run_manifest_path = Path(base.run_manifest).resolve()
    megatron_path = Path(base.megatron_checkpoint).resolve()
    hf_path = Path(base.hf_checkpoint).resolve()
    tokenizer_path = Path(base.tokenizer).resolve()
    for path, label in (
        (run_manifest_path, "source run manifest"),
        (megatron_path, "Megatron step_399 checkpoint"),
        (hf_path, "HF step_399 checkpoint"),
        (tokenizer_path, "16-token tokenizer"),
    ):
        if not path.exists():
            raise FileNotFoundError(f"Missing {label}: {path}")

    run_manifest = read_json_object(run_manifest_path)
    if hashlib.sha256(str(run_manifest.get("run_name", "")).encode()).hexdigest() != "88d0ca794cb2daf4fdfcf6e339adf413a1910049c2c861a1db6f507898ee141c":
        raise ValueError("Unexpected source run identity")
    expected_manifest = {
        "run_kind": "arc1_rearc",
        "status": "complete",
        "exit_status": 0,
        "max_steps": 399,
        "max_epochs": 1,
        "lr_decay_iters": 399,
        "lr_warmup_iters": 6,
        "train_splits": ["rearc"],
        "val_split": "arc1_safe_validation",
        "expected_train_rows": 102_392,
        "expected_data_batches_sha256": (
            "57a918c44191a315e4c3182d1cce9c785c516c92a6e5afc68fb2bb062e4c1d73"
        ),
        "expected_data_permutation_sha256": (
            "e34454cf4740b3488b575bc5f9a93effad77ffc8db6a60c95ca4f31ba04096dd"
        ),
    }
    mismatches = {
        key: {"expected": expected, "actual": run_manifest.get(key)}
        for key, expected in expected_manifest.items()
        if run_manifest.get(key) != expected
    }
    if mismatches:
        raise ValueError(f"Locked RE-ARC source manifest mismatch: {mismatches}")
    if megatron_path.name != "step_399" or hf_path.name != "step_399-hf":
        raise ValueError("Resolved checkpoint paths are not step_399")

    model_config_path = hf_path / "config.json"
    tokenizer_config_path = tokenizer_path / "tokenizer_config.json"
    token_mapping_path = tokenizer_path / "nvarc_16t_token_mapping.json"
    for path in (model_config_path, tokenizer_config_path, token_mapping_path):
        if not path.is_file():
            raise FileNotFoundError(f"Missing base identity artifact: {path}")
    model_config = read_json_object(model_config_path)
    model_mismatches = {
        key: {"expected": expected, "actual": model_config.get(key)}
        for key, expected in MODEL_CONTRACT.items()
        if model_config.get(key) != expected
    }
    if model_mismatches:
        raise ValueError(f"Locked HF model contract mismatch: {model_mismatches}")

    weight_paths = sorted(hf_path.glob("*.safetensors")) + sorted(
        hf_path.glob("pytorch_model*.bin")
    )
    if not weight_paths:
        raise FileNotFoundError(f"HF checkpoint has no weight shards: {hf_path}")
    artifacts = {
        path.name: {"bytes": path.stat().st_size, "sha256": sha256_file(path)}
        for path in weight_paths
    }
    return {
        "format": "nvarc_rearc_step399_base_identity_v1",
        "run_name": LOCKED_RUN_NAME,
        "source_step": 399,
        "run_manifest": str(run_manifest_path),
        "run_manifest_sha256": sha256_file(run_manifest_path),
        "megatron_checkpoint": str(megatron_path),
        "hf_checkpoint": str(hf_path),
        "tokenizer": str(tokenizer_path),
        "model_config_sha256": sha256_file(model_config_path),
        "tokenizer_config_sha256": sha256_file(tokenizer_config_path),
        "token_mapping_sha256": sha256_file(token_mapping_path),
        "hf_artifacts": artifacts,
        "config_sha256": config.semantic_sha256,
    }


def _inspect_pinned_hf_base(config: NvarcTranscoderConfig) -> dict[str, Any]:
    base = config.base
    if not isinstance(base, PinnedHfBaseConfig):
        raise TypeError("Expected a pinned HF base configuration")
    hf_path = Path(base.hf_checkpoint).resolve()
    tokenizer_path = Path(base.tokenizer).resolve()
    if not hf_path.is_dir():
        raise FileNotFoundError(f"Missing pinned HF checkpoint: {hf_path}")
    if not tokenizer_path.is_dir():
        raise FileNotFoundError(f"Missing 16-token tokenizer: {tokenizer_path}")

    artifacts: dict[str, dict[str, Any]] = {}
    for name, expected_sha256 in sorted(base.expected_files.items()):
        path = hf_path / name
        if not path.is_file():
            raise FileNotFoundError(f"Missing pinned HF artifact: {path}")
        actual_sha256 = sha256_file(path)
        if actual_sha256 != expected_sha256:
            raise ValueError(
                f"Pinned HF artifact hash mismatch for {name}: "
                f"{actual_sha256} != {expected_sha256}"
            )
        artifacts[name] = {
            "sha256": actual_sha256,
            "size_bytes": path.stat().st_size,
        }
    aggregate_sha256 = semantic_sha256({"files": artifacts})
    if aggregate_sha256 != base.aggregate_sha256:
        raise ValueError(
            "Pinned HF aggregate hash mismatch: "
            f"{aggregate_sha256} != {base.aggregate_sha256}"
        )

    model_config_path = hf_path / "config.json"
    model_config = read_json_object(model_config_path)
    model_mismatches = {
        key: {"expected": expected, "actual": model_config.get(key)}
        for key, expected in MODEL_CONTRACT.items()
        if model_config.get(key) != expected
    }
    if model_mismatches:
        raise ValueError(f"Pinned HF model contract mismatch: {model_mismatches}")

    tokenizer_config_path = tokenizer_path / "tokenizer_config.json"
    token_mapping_path = tokenizer_path / "nvarc_16t_token_mapping.json"
    tokenizer_hashes = {
        tokenizer_config_path: base.tokenizer_config_sha256,
        token_mapping_path: base.token_mapping_sha256,
    }
    for path, expected_sha256 in tokenizer_hashes.items():
        if not path.is_file():
            raise FileNotFoundError(f"Missing tokenizer identity artifact: {path}")
        actual_sha256 = sha256_file(path)
        if actual_sha256 != expected_sha256:
            raise ValueError(
                f"Tokenizer artifact hash mismatch for {path.name}: "
                f"{actual_sha256} != {expected_sha256}"
            )

    return {
        "format": "nvarc_rearc_pinned_hf_base_identity_v2",
        "label": base.label,
        "run_name": base.run_name,
        "source_step": base.source_step,
        "hf_checkpoint": str(hf_path),
        "hf_aggregate_sha256": aggregate_sha256,
        "hf_artifacts": artifacts,
        "tokenizer": str(tokenizer_path),
        "model_config_sha256": sha256_file(model_config_path),
        "tokenizer_config_sha256": sha256_file(tokenizer_config_path),
        "token_mapping_sha256": sha256_file(token_mapping_path),
        "config_sha256": config.semantic_sha256,
    }


def inspect_locked_base(config: NvarcTranscoderConfig) -> dict[str, Any]:
    if isinstance(config.base, PinnedHfBaseConfig):
        return _inspect_pinned_hf_base(config)
    return _inspect_step399_base(config)


def load_or_create_base_identity(
    config: NvarcTranscoderConfig,
    *,
    destination: str | Path,
    allow_hc_root_mount_alias: bool = False,
) -> dict[str, Any]:
    from filelock import FileLock

    path = Path(destination).resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    with FileLock(str(path) + ".lock"):
        if path.is_file():
            value = read_json_object(path)
            expected = {
                "format": (
                    "nvarc_rearc_pinned_hf_base_identity_v2"
                    if isinstance(config.base, PinnedHfBaseConfig)
                    else "nvarc_rearc_step399_base_identity_v1"
                ),
                "run_name": config.base.run_name,
                "source_step": config.base.source_step,
                "config_sha256": config.semantic_sha256,
                "hf_checkpoint": str(Path(config.base.hf_checkpoint).resolve()),
            }
            if isinstance(config.base, PinnedHfBaseConfig):
                expected["hf_aggregate_sha256"] = config.base.aggregate_sha256
            mismatches = {
                key: {"expected": expected_value, "actual": value.get(key)}
                for key, expected_value in expected.items()
                if value.get(key) != expected_value
            }
            if mismatches:
                if allow_hc_root_mount_alias and _cached_base_identity_matches_mount_alias(
                    config, value
                ):
                    return value
                raise ValueError(f"Cached base identity mismatch: {mismatches}")
            return value
        value = inspect_locked_base(config)
        temporary = path.with_name(path.name + ".tmp")
        temporary.write_text(
            json.dumps(value, indent=2, sort_keys=True, ensure_ascii=True) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, path)
        return value


def load_or_create_native_identity(
    config: NvarcTranscoderConfig,
    *,
    destination: str | Path,
) -> dict[str, Any]:
    from filelock import FileLock

    path = Path(destination).resolve()
    model_root = Path(config.model.native_model).resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    with FileLock(str(path) + ".lock"):
        if path.is_file():
            value = read_json_object(path)
            expected = {
                "format": "nvarc_native_qwen3_4b_identity_v1",
                "model": str(model_root),
                "config_sha256": config.semantic_sha256,
            }
            mismatches = {
                key: {"expected": expected_value, "actual": value.get(key)}
                for key, expected_value in expected.items()
                if value.get(key) != expected_value
            }
            if mismatches:
                raise ValueError(f"Cached native Qwen identity mismatch: {mismatches}")
            return value
        config_path = model_root / "config.json"
        if not config_path.is_file():
            raise FileNotFoundError(f"Native Qwen config is missing: {config_path}")
        model_config = read_json_object(config_path)
        expected_model = {
            "model_type": "qwen3",
            "hidden_size": 2560,
            "num_hidden_layers": 36,
        }
        mismatches = {
            key: {"expected": expected, "actual": model_config.get(key)}
            for key, expected in expected_model.items()
            if model_config.get(key) != expected
        }
        if mismatches:
            raise ValueError(f"Native Qwen model contract mismatch: {mismatches}")
        weight_paths = sorted(model_root.glob("*.safetensors")) + sorted(
            model_root.glob("pytorch_model*.bin")
        )
        if not weight_paths:
            raise FileNotFoundError(f"Native Qwen has no local weight shards: {model_root}")
        value = {
            "format": "nvarc_native_qwen3_4b_identity_v1",
            "model": str(model_root),
            "config_sha256": config.semantic_sha256,
            "model_config_sha256": sha256_file(config_path),
            "artifacts": {
                item.name: {"bytes": item.stat().st_size, "sha256": sha256_file(item)}
                for item in weight_paths
            },
            "tokenizer_artifacts": {
                item.name: sha256_file(item)
                for item in sorted(model_root.glob("tokenizer*"))
                if item.is_file()
            },
        }
        temporary = path.with_name(path.name + ".tmp")
        temporary.write_text(
            json.dumps(value, indent=2, sort_keys=True, ensure_ascii=True) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, path)
        return value


def inspect_dataset(
    path: str | Path,
    *,
    expected_rows: int,
    expected_puzzles: int,
    expected_events: int | None = None,
    expected_generation_modes: Mapping[str, int] | None = None,
) -> tuple[Any, dict[str, Any]]:
    from datasets import load_from_disk

    resolved = Path(path).resolve()
    if not resolved.is_dir():
        raise FileNotFoundError(f"Dataset is missing: {resolved}")
    dataset = load_from_disk(str(resolved))
    puzzle_names = {
        str(value)
        for value in dataset["puzzle_name"]
        if value is not None and str(value)
    }
    if len(dataset) != expected_rows or len(puzzle_names) != expected_puzzles:
        raise ValueError(
            f"Dataset identity mismatch at {resolved}: "
            f"rows={len(dataset)}, puzzles={len(puzzle_names)}"
        )
    has_generation_modes = "trace_generation_mode" in dataset.column_names
    generation_modes = (
        [str(value) for value in dataset["trace_generation_mode"]]
        if has_generation_modes
        else ["cumulative_generated"] * len(dataset)
    )
    mode_counts: dict[str, int] = {}
    event_count = 0
    for index, messages in enumerate(dataset["messages"]):
        mode = generation_modes[index]
        mode_counts[mode] = mode_counts.get(mode, 0) + 1
        assistant_events = sum(
            1
            for message in messages
            if str(message["role"]).lower() == "assistant"
        )
        if mode == "cumulative_generated":
            event_count += assistant_events
        elif mode == "final_only_gold_prefix":
            if assistant_events < 1:
                raise ValueError(
                    f"Final-only dataset row {index} has no terminal assistant slot"
                )
            event_count += 1
        else:
            raise ValueError(
                f"Unsupported trace_generation_mode at row {index}: {mode}"
            )
    if expected_events is not None and event_count != expected_events:
        raise ValueError(
            f"Dataset event mismatch at {resolved}: "
            f"events={event_count}, expected={expected_events}"
        )
    if expected_generation_modes is not None:
        expected_modes = {
            str(key): int(value) for key, value in expected_generation_modes.items()
        }
        if mode_counts != expected_modes:
            raise ValueError(
                f"Dataset generation-mode mismatch at {resolved}: "
                f"modes={mode_counts}, expected={expected_modes}"
            )
    identity = {
        "path": str(resolved),
        "rows": len(dataset),
        "puzzles": len(puzzle_names),
        "events": event_count,
        "fingerprint": getattr(dataset, "_fingerprint", None),
    }
    if has_generation_modes:
        identity["generation_mode_rows"] = mode_counts
    return dataset, identity
