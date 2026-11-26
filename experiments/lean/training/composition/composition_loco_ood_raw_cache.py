"""Strict verification and initialization of the frozen raw-weight condition.

A changed serialization is not evidence of equal tensors.  The only alternate
load accepted here requires an archived original manifest that names the frozen
file SHA and exactly matches all current per-target tensors and semantic fields.
This module neither edits caches/configs nor uses the training-resume exception.
The explicitly approved LR-half fresh-training route reuses this same strict
initialization; the historical v1 training loader is not changed.
"""

from __future__ import annotations

from dataclasses import asdict
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, TYPE_CHECKING

if TYPE_CHECKING:
    import torch
    from .composition_loco_config import CompositionLOCOConfig
    from .hypernet.runtime import S5HyperNet


RECEIPT_FORMAT = "s5_composition_loco_ood_raw_cache_receipt_v1"
RECREATED_RECEIPT_FORMAT = "s5_composition_loco_recreated_raw_cache_receipt_v1"
_BASE_FIELDS = ("checkpoint_sha256", "model_state_sha256", "config_sha256", "step")
_SEMANTIC_FIELDS = (
    "format", "kind", "matrix_count", "target_keys", "target_slots",
    "target_modules", "row_token_count", "column_token_count", "projection_seed",
    "storage_dtype", "normalization", "rms_epsilon", "log_rms_population_statistics",
    "log_rms_mean", "log_rms_std", "layout_sha256", "target_tensor_sha256",
)


class RawCacheOODBlocked(ValueError):
    """Only the raw-weight evaluation is blocked; other routes may proceed."""


def _json_sha(value: Any) -> str:
    return hashlib.sha256(json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True,
    ).encode("utf-8")).hexdigest()


def _file_sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _resolve(root: Path, value: str | Path) -> Path:
    path = Path(value).expanduser()
    return (path if path.is_absolute() else root / path).resolve()


def _relative(root: Path, path: Path) -> str:
    try:
        return path.relative_to(root).as_posix()
    except ValueError as exc:
        raise RawCacheOODBlocked(f"raw cache evidence must be inside repository: {path}") from exc


def _read_manifest(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RawCacheOODBlocked(f"raw cache manifest is not an object: {path}")
    body = dict(value)
    if body.pop("manifest_sha256", None) != _json_sha(body):
        raise RawCacheOODBlocked(f"raw cache manifest self-hash changed: {path}")
    return body


def _semantic_manifest(value: Mapping[str, Any]) -> dict[str, Any]:
    missing = [name for name in (*_SEMANTIC_FIELDS, "base_identity") if name not in value]
    if missing:
        raise RawCacheOODBlocked(f"raw cache semantic manifest is incomplete: {missing}")
    base = value["base_identity"]
    if not isinstance(base, Mapping) or any(name not in base for name in _BASE_FIELDS):
        raise RawCacheOODBlocked("raw cache manifest has incomplete Base identity")
    hashes = value["target_tensor_sha256"]
    if not isinstance(hashes, list) or len(hashes) != 32 or any(
        not isinstance(item, str) or len(item) != 64
        or any(char not in "0123456789abcdef" for char in item) for item in hashes
    ):
        raise RawCacheOODBlocked("raw cache manifest must contain 32 target tensor SHA256s")
    return {
        **{name: value[name] for name in _SEMANTIC_FIELDS},
        "base_identity": {name: base[name] for name in _BASE_FIELDS},
    }


def _candidate_original_manifests(config: CompositionLOCOConfig, root: Path, cache: Path):
    """Inspect bounded provenance directories, never recursively scan outputs."""
    initial = _resolve(root, str(config.initialization["checkpoint_path"]))
    folders = {cache.parent, initial.parent, initial.parent.parent}
    for key in ("config_path", "source_config_path"):
        if config.initialization.get(key):
            folders.add(_resolve(root, str(config.initialization[key])).parent)
    folders |= {
        folder / child for folder in tuple(folders)
        for child in ("provenance", "artifacts", "metadata", "source_copies")
    }
    seen: set[Path] = set()
    for folder in sorted(folders):
        if not folder.is_dir():
            continue
        _relative(root, folder.resolve())
        for path in sorted(folder.iterdir()):
            # Include .json.bak/.json.original archive suffixes without reading
            # model payloads, logs or arbitrary large run artifacts.
            if ".json" not in path.name or not path.is_file():
                continue
            path = path.resolve()
            _relative(root, path)
            if path not in seen and path.stat().st_size <= 1024 * 1024:
                seen.add(path)
                yield path


def check_raw_cache_for_ood(
    config: CompositionLOCOConfig, root: str | Path,
) -> dict[str, Any]:
    """Return a sealed cache-identity receipt or fail only this raw route.

    Exact original bytes are accepted normally. Changed bytes additionally need
    an actual archived manifest binding the frozen original SHA to equal tensor
    hashes. A repair permission, matching Base labels, or a fresh reconstruction
    alone is insufficient. This verifies cache identity, not training history.
    """
    from .hypernet.raw_weight import S5RawWeightCache

    if config.route != "raw_weight" or config.model.get("condition_source") != "raw_weight":
        raise RawCacheOODBlocked("raw OOD cache verifier is restricted to raw_weight")
    repository = Path(root).expanduser().resolve()
    cache_path = _resolve(repository, str(config.model["raw_weight_cache_path"]))
    relative_cache = _relative(repository, cache_path)
    expected = str(config.model["raw_weight_cache_sha256"])
    if len(expected) != 64 or any(char not in "0123456789abcdef" for char in expected):
        raise RawCacheOODBlocked("raw OOD configured cache SHA256 is malformed")
    if not cache_path.is_file():
        raise RawCacheOODBlocked(f"raw OOD cache is missing: {relative_cache}")
    actual = _file_sha(cache_path)
    sidecar = cache_path.with_suffix(cache_path.suffix + ".json")
    try:
        cache = S5RawWeightCache.load(
            cache_path, expected_sha256=actual, expected_base_identity=dict(config.base),
        )
        current_manifest = _read_manifest(sidecar)
        semantic = _semantic_manifest(cache.manifest())
        if _semantic_manifest(current_manifest) != semantic:
            raise RawCacheOODBlocked("current raw manifest semantic fields do not match loaded tensors")
    except (OSError, ValueError, KeyError, RuntimeError) as exc:
        raise RawCacheOODBlocked(f"raw OOD current cache validation failed: {exc}") from exc

    reference = sidecar
    kind = "original_bytes"
    if actual != expected:
        reference = None
        for candidate in _candidate_original_manifests(config, repository, cache_path):
            try:
                candidate_body = json.loads(candidate.read_text(encoding="utf-8"))
            except (OSError, ValueError, UnicodeError):
                continue
            if not isinstance(candidate_body, dict) or candidate_body.get("tensor_sha256") != expected:
                continue
            original = _read_manifest(candidate)
            original_semantic = _semantic_manifest(original)
            if original_semantic != semantic:
                changed = [name for name in semantic if original_semantic[name] != semantic[name]]
                raise RawCacheOODBlocked(
                    "raw OOD original/current cache semantic mismatch: " + ", ".join(changed)
                )
            reference = candidate
            kind = "original_manifest_exact_tensor_equivalence"
            break
        if reference is None:
            wanted = cache_path.with_name(cache_path.name + ".original.json")
            raise RawCacheOODBlocked(
                f"raw OOD cache bytes differ: expected {expected}, actual {actual}; "
                "no archived original manifest proves tensor equivalence. Restore the actual "
                f"pre-repair sidecar to {_relative(repository, wanted)} (tensor_sha256={expected}, "
                "all 32 target_tensor_sha256 values and original manifest_sha256), or restore "
                "the original cache bytes with their sidecar. Do not manufacture a new reference "
                "from the rebuilt cache. Other routes may proceed."
            )

    receipt = {
        "format": RECEIPT_FORMAT,
        "status": "verified",
        "verification_kind": kind,
        "scope": "frozen_raw_condition_identity_only_not_training_history",
        "config_sha256": config.computed_sha256,
        "cache_path": relative_cache,
        "expected_cache_sha256": expected,
        "actual_cache_sha256": actual,
        "current_manifest_sha256": _file_sha(sidecar),
        "reference_manifest_path": _relative(repository, reference),
        "reference_manifest_file_sha256": _file_sha(reference),
        "semantic_sha256": _json_sha(semantic),
        "target_tensor_sha256": semantic["target_tensor_sha256"],
        "base_identity": semantic["base_identity"],
    }
    receipt["receipt_sha256"] = _json_sha(receipt)
    return receipt


def load_raw_hypernet_for_ood(
    config: CompositionLOCOConfig,
    device: torch.device,
    receipt: Mapping[str, Any],
    *,
    root: str | Path | None = None,
) -> S5HyperNet:
    """Load the raw route initialization with a freshly reverified receipt.

    The evaluator must subsequently load its already selected LOCO checkpoint.
    The training loader and its resume authorization remain unchanged.
    """
    import torch
    from .composition_loco_train import _load_state, _verify_initial_checkpoint
    from .hypernet.config import S5BaseIdentity
    from .hypernet.raw_weight import S5RawWeightCache
    from .hypernet.runtime import S5HyperNet, load_s5_base_model

    repository = Path.cwd().resolve() if root is None else Path(root).expanduser().resolve()
    body = dict(receipt)
    if body.pop("receipt_sha256", None) != _json_sha(body):
        raise RawCacheOODBlocked("raw OOD cache receipt self-hash changed")
    verified = check_raw_cache_for_ood(config, repository)
    if dict(receipt) != verified:
        raise RawCacheOODBlocked("raw OOD cache identity changed after preflight")
    cache = S5RawWeightCache.load(
        _resolve(repository, verified["cache_path"]),
        expected_sha256=verified["actual_cache_sha256"],
        expected_base_identity=dict(config.base),
    )
    if _json_sha(_semantic_manifest(cache.manifest())) != verified["semantic_sha256"]:
        raise RawCacheOODBlocked("raw OOD cache tensors changed during load")
    base_identity = S5BaseIdentity(
        checkpoint_path=str(_resolve(repository, config.base["checkpoint_path"])),
        architecture_config_path=str(_resolve(repository, config.base["architecture_config_path"])),
        **{name: config.base[name] for name in _BASE_FIELDS},
    )
    base, identity = load_s5_base_model(
        base_identity.architecture_config_path, base_identity.checkpoint_path,
        device=device, dtype=torch.bfloat16, expected_identity=base_identity,
    )
    if {name: asdict(identity)[name] for name in _BASE_FIELDS} != verified["base_identity"]:
        raise RawCacheOODBlocked("raw OOD loaded Base identity changed")
    core = S5HyperNet(
        base, candidate="A", seed=int(config.training["seed"]), condition_source="raw_weight",
        fusion_mode="append", raw_weight_cache=cache,
        raw_fusion_mode=str(config.model.get("raw_fusion_mode", "query_cross")),
    )
    core.validate_step0_initialization()
    initial = _resolve(repository, str(config.initialization["checkpoint_path"]))
    _verify_initial_checkpoint(initial, config)
    state, payload = _load_state(initial)
    if (
        payload.get("format") != "s5_hypernet_formal_checkpoint_v1"
        or payload.get("candidate") != "A"
        or payload.get("condition_source") != "raw_weight"
        or payload.get("group") != config.model.get("group")
        or int(payload.get("global_step", -1)) != int(config.initialization["step"])
        or payload.get("raw_weight_cache_sha256") != verified["expected_cache_sha256"]
    ):
        raise RawCacheOODBlocked("raw OOD initial checkpoint route/cache identity changed")
    core.load_trainable_state_dict(state)
    core.to(device)
    if core.raw_weight_conditioner is not None:
        core.raw_weight_conditioner.restore_fp32_output_path()
    core.__dict__["_s5_ood_raw_cache_receipt"] = dict(verified)
    return core


def check_recreated_raw_cache(config: CompositionLOCOConfig, root: str | Path) -> dict[str, Any]:
    """Verify the deliberately rebuilt cache without calling it historical."""
    from .hypernet.raw_weight import S5RawWeightCache

    repository = Path(root).expanduser().resolve()
    cache_path = _resolve(repository, str(config.model["raw_weight_cache_path"]))
    expected = str(config.model["raw_weight_cache_sha256"])
    if not cache_path.is_file() or _file_sha(cache_path) != expected:
        raise RawCacheOODBlocked("recreated raw cache bytes are missing or changed")
    sidecar = cache_path.with_suffix(cache_path.suffix + ".json")
    cache = S5RawWeightCache.load(
        cache_path, expected_sha256=expected, expected_base_identity=dict(config.base),
    )
    current_manifest = _read_manifest(sidecar)
    semantic = _semantic_manifest(cache.manifest())
    if _semantic_manifest(current_manifest) != semantic:
        raise RawCacheOODBlocked("recreated raw cache manifest does not match loaded tensors")
    historical = str(config.initialization.get("historical_raw_weight_cache_sha256", ""))
    if historical != "a70eb76788bcef9c2ca301ad4a4d3febc94eb247ce1f02cf41ebdccda8bb87e1":
        raise RawCacheOODBlocked("recreated raw config has no approved historical cache reference")
    receipt = {
        "format": RECREATED_RECEIPT_FORMAT,
        "status": "verified_recreated_bytes",
        "scope": "rebuilt_raw_condition_identity_not_historical_cache_equivalence",
        "config_sha256": config.computed_sha256,
        "cache_path": _relative(repository, cache_path),
        "historical_cache_sha256": historical,
        "recreated_cache_sha256": expected,
        "current_manifest_sha256": _file_sha(sidecar),
        "semantic_sha256": _json_sha(semantic),
        "target_tensor_sha256": semantic["target_tensor_sha256"],
        "base_identity": semantic["base_identity"],
    }
    receipt["receipt_sha256"] = _json_sha(receipt)
    return receipt


def load_raw_hypernet_for_recreated_inputs(
    config: CompositionLOCOConfig,
    device: torch.device,
    receipt: Mapping[str, Any],
    *,
    root: str | Path | None = None,
) -> S5HyperNet:
    """Load the Raw route with an explicit rebuilt-cache provenance record."""
    import torch
    from .composition_loco_train import _load_state, _verify_initial_checkpoint
    from .hypernet.config import S5BaseIdentity
    from .hypernet.raw_weight import S5RawWeightCache
    from .hypernet.runtime import S5HyperNet, load_s5_base_model

    repository = Path.cwd().resolve() if root is None else Path(root).expanduser().resolve()
    body = dict(receipt)
    if body.pop("receipt_sha256", None) != _json_sha(body):
        raise RawCacheOODBlocked("recreated raw cache receipt self-hash changed")
    verified = check_recreated_raw_cache(config, repository)
    if dict(receipt) != verified:
        raise RawCacheOODBlocked("recreated raw cache identity changed after preflight")
    cache = S5RawWeightCache.load(
        _resolve(repository, verified["cache_path"]),
        expected_sha256=verified["recreated_cache_sha256"],
        expected_base_identity=dict(config.base),
    )
    base_identity = S5BaseIdentity(
        checkpoint_path=str(_resolve(repository, config.base["checkpoint_path"])),
        architecture_config_path=str(_resolve(repository, config.base["architecture_config_path"])),
        **{name: config.base[name] for name in _BASE_FIELDS},
    )
    base, identity = load_s5_base_model(
        base_identity.architecture_config_path, base_identity.checkpoint_path,
        device=device, dtype=torch.bfloat16, expected_identity=base_identity,
    )
    if {name: asdict(identity)[name] for name in _BASE_FIELDS} != verified["base_identity"]:
        raise RawCacheOODBlocked("recreated Raw loaded Base identity changed")
    core = S5HyperNet(
        base, candidate="A", seed=int(config.training["seed"]), condition_source="raw_weight",
        fusion_mode="append", raw_weight_cache=cache,
        raw_fusion_mode=str(config.model.get("raw_fusion_mode", "query_cross")),
    )
    core.validate_step0_initialization()
    initial = _resolve(repository, str(config.initialization["checkpoint_path"]))
    _verify_initial_checkpoint(initial, config)
    state, payload = _load_state(initial)
    historical = str(config.initialization["historical_raw_weight_cache_sha256"])
    if (
        payload.get("format") != "s5_hypernet_formal_checkpoint_v1"
        or payload.get("candidate") != "A"
        or payload.get("condition_source") != "raw_weight"
        or payload.get("group") != config.model.get("group")
        or int(payload.get("global_step", -1)) != int(config.initialization["step"])
        or payload.get("raw_weight_cache_sha256") != historical
    ):
        raise RawCacheOODBlocked("recreated Raw initial checkpoint identity changed")
    core.load_trainable_state_dict(state)
    core.to(device)
    if core.raw_weight_conditioner is not None:
        core.raw_weight_conditioner.restore_fp32_output_path()
    core.__dict__["_s5_ood_raw_cache_receipt"] = dict(verified)
    return core


__all__ = [
    "RawCacheOODBlocked", "check_raw_cache_for_ood", "load_raw_hypernet_for_ood",
    "check_recreated_raw_cache", "load_raw_hypernet_for_recreated_inputs",
]
