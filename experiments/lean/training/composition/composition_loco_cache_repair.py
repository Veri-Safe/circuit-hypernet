"""Fail-closed handling for a rebuilt raw-weight cache during LOCO recovery."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping


RAW_CACHE_REPAIR_FORMAT = "s5_composition_loco_raw_cache_repair_v1"
BASE_CONTENT_IDENTITY_FIELDS = (
    "checkpoint_sha256",
    "model_state_sha256",
    "config_sha256",
    "step",
)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json_sha256(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode(
            "utf-8"
        )
    ).hexdigest()


def _resolve(root: Path, value: str | Path) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = root / path
    return path.resolve()


def _relative(root: Path, path: Path, *, label: str) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError as exc:
        raise ValueError(f"raw cache repair {label} must be inside the repository: {path}") from exc


def _content_identity(value: Mapping[str, Any]) -> dict[str, Any]:
    result = {field: value.get(field) for field in BASE_CONTENT_IDENTITY_FIELDS}
    if result["step"] is not None:
        result["step"] = int(result["step"])
    return result


def inspect_raw_weight_cache(
    *,
    root: str | Path,
    cache_path: str | Path,
    expected_sha256: str,
    route: str,
    resume_path: str | Path | None,
    base_identity: Mapping[str, Any],
    repair_manifest_path: str | Path | None = None,
) -> dict[str, Any]:
    """Validate the cache and, when explicitly recovering, its repair record.

    The normal path returns the configured SHA as ``load_sha256``.  A rebuilt
    cache is accepted only when a resume path and a sidecar repair manifest
    authorize that exact path.  The returned record is JSON-safe and is meant
    to be included in run provenance.
    """

    repository = Path(root).expanduser().resolve()
    cache = _resolve(repository, cache_path)
    if not cache.is_file():
        raise FileNotFoundError(f"S5 raw cache is missing: {cache}")
    expected = str(expected_sha256)
    if len(expected) != 64 or any(char not in "0123456789abcdef" for char in expected):
        raise ValueError("S5 raw cache expected SHA256 is malformed")
    actual = _sha256_file(cache)
    result: dict[str, Any] = {
        "cache_repaired": False,
        "cache_path": _relative(repository, cache, label="cache path"),
        "expected_sha256": expected,
        "actual_sha256": actual,
        "load_sha256": expected,
        "repair": None,
    }
    if actual == expected:
        return result

    if route != "raw_weight":
        raise ValueError("raw cache SHA differs outside the raw_weight route")
    if resume_path is None or not str(resume_path):
        raise ValueError(
            "S5 raw cache SHA differs; a rebuilt cache is allowed only for an explicit LOCO resume"
        )
    resume = _resolve(repository, resume_path)
    resume_relative = _relative(repository, resume, label="resume path")
    default_manifest = cache.with_name(cache.name + ".repair.json")
    manifest = default_manifest if repair_manifest_path is None else _resolve(repository, repair_manifest_path)
    if manifest != default_manifest:
        raise ValueError(f"raw cache repair manifest must sit beside the cache: {manifest}")
    if not manifest.is_file():
        raise FileNotFoundError(f"raw cache repair manifest is missing: {manifest}")
    try:
        value = json.loads(manifest.read_text(encoding="utf-8"))
    except Exception as exc:
        raise ValueError(f"raw cache repair manifest is not valid JSON: {manifest}") from exc
    if not isinstance(value, Mapping):
        raise ValueError("raw cache repair manifest must be an object")
    body = dict(value)
    recorded_manifest_sha = body.pop("manifest_sha256", None)
    if recorded_manifest_sha != _json_sha256(body):
        raise ValueError("raw cache repair manifest identity changed")
    if body.get("format") != RAW_CACHE_REPAIR_FORMAT or body.get("status") != "approved":
        raise ValueError("raw cache repair manifest is not approved")
    if body.get("cache_path") != result["cache_path"]:
        raise ValueError("raw cache repair manifest cache path changed")
    if body.get("expected_cache_sha256") != expected:
        raise ValueError("raw cache repair expected SHA differs from the run config")
    if body.get("actual_cache_sha256") != actual:
        raise ValueError("raw cache repair actual SHA differs from the current cache")
    allowed = body.get("allowed_resume_paths")
    if (
        not isinstance(allowed, list)
        or not allowed
        or any(not isinstance(item, str) or not item for item in allowed)
        or len(set(allowed)) != len(allowed)
        or resume_relative not in allowed
    ):
        raise ValueError("raw cache repair resume path is not authorized by the manifest")
    reason = body.get("reason")
    created_at = body.get("created_at")
    if not isinstance(reason, str) or not reason.strip() or not isinstance(created_at, str) or not created_at.strip():
        raise ValueError("raw cache repair manifest must record reason and created_at")
    manifest_identity = body.get("base_identity")
    if not isinstance(manifest_identity, Mapping) or _content_identity(manifest_identity) != _content_identity(base_identity):
        raise ValueError("raw cache repair Base identity differs from the run config")
    repair = {
        "cache_repaired": True,
        "repair_manifest": _relative(repository, manifest, label="manifest path"),
        "cache_path": result["cache_path"],
        "expected_cache_sha256": expected,
        "actual_cache_sha256": actual,
        "base_identity": _content_identity(base_identity),
        "resume_path": resume_relative,
        "reason": reason,
        "created_at": created_at,
    }
    return {
        **result,
        "cache_repaired": True,
        "load_sha256": actual,
        "repair": repair,
    }


__all__ = [
    "BASE_CONTENT_IDENTITY_FIELDS",
    "RAW_CACHE_REPAIR_FORMAT",
    "inspect_raw_weight_cache",
]
