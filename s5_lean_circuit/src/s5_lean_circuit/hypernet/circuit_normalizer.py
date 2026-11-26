"""Train-only fixed-log1p and FP64 streaming z-score for Vector397."""

from __future__ import annotations

from dataclasses import dataclass, replace
import hashlib
import json
import os
from pathlib import Path
import tempfile
from typing import Any, Iterable, Mapping

import numpy as np
import torch

from ..circuit.descriptor_v2_397 import SCHEMA_397, fixed_log1p_397
from .data import (
    VECTOR397_DIM,
    VECTOR397_SCHEMA_SHA256,
    StrictDataError,
    Vector397Row,
    iter_vector397_rows,
    semantic_sha256,
)


DEAD_STD_THRESHOLD = 1e-6
NORMALIZER_FORMAT = "s5_hypernet_vector397_normalizer_v2"
FIXED_LOG1P_FORMAT = "s5_hypernet_vector397_fixed_log1p_v1"


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _sha256_hex(value: Any, field: str) -> str:
    text = str(value)
    if len(text) != 64 or any(character not in "0123456789abcdef" for character in text):
        raise StrictDataError(f"{field} is not a lowercase SHA256")
    return text


def _fixed_log1p_mask() -> np.ndarray:
    return np.ascontiguousarray(np.asarray(SCHEMA_397.log1p_mask.numpy(), dtype=np.bool_))


def _array_sha256(value: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


FIXED_LOG1P_MASK_SHA256 = _array_sha256(_fixed_log1p_mask())


def _sidecar_path(path: Path) -> Path:
    return path.with_suffix(path.suffix + ".json")


def _artifact_metadata(normalizer: "Vector397Normalizer") -> dict[str, Any]:
    metadata = normalizer.metadata()
    metadata.pop("artifact_sha256", None)
    return metadata


def _artifact_digest_payload(
    metadata: Mapping[str, Any],
    *,
    mean: np.ndarray,
    std: np.ndarray,
    dead_mask: np.ndarray,
    fixed_log1p_mask: np.ndarray,
) -> bytes:
    """Return the canonical bytes used for the semantic artifact checksum."""

    digest = hashlib.sha256()
    digest.update(_canonical_json(dict(metadata)).encode("utf-8"))
    for name, value in (
        ("mean", mean),
        ("std", std),
        ("dead_mask", dead_mask),
        ("fixed_log1p_mask", fixed_log1p_mask),
    ):
        encoded_name = name.encode("ascii")
        digest.update(len(encoded_name).to_bytes(4, "big"))
        digest.update(encoded_name)
        digest.update(np.ascontiguousarray(value).tobytes())
    return digest.digest()


def _artifact_semantic_sha256(
    metadata: Mapping[str, Any],
    *,
    mean: np.ndarray,
    std: np.ndarray,
    dead_mask: np.ndarray,
    fixed_log1p_mask: np.ndarray,
) -> str:
    return hashlib.sha256(
        _artifact_digest_payload(
            metadata,
            mean=mean,
            std=std,
            dead_mask=dead_mask,
            fixed_log1p_mask=fixed_log1p_mask,
        )
    ).hexdigest()


def fixed_log1p_vector397(vector: Any, *, descriptor_schema_sha256: str | None = None) -> np.ndarray:
    """Apply the frozen Vector397 mask, without clipping any coordinate."""

    value = np.asarray(vector, dtype=np.float32)
    if value.shape != (VECTOR397_DIM,):
        raise StrictDataError("Vector397 input must have shape [397]")
    if not np.isfinite(value).all():
        raise StrictDataError("Vector397 input contains non-finite values")
    expected_schema = VECTOR397_SCHEMA_SHA256
    if descriptor_schema_sha256 is not None and str(descriptor_schema_sha256) != expected_schema:
        raise StrictDataError("descriptor_schema_sha256 does not match frozen Vector397 schema")
    try:
        result = fixed_log1p_397(torch.from_numpy(np.ascontiguousarray(value)))
    except ValueError as error:
        raise StrictDataError(str(error)) from error
    return np.ascontiguousarray(result.numpy(), dtype=np.float32)


@dataclass(frozen=True)
class FixedLog1pVector397:
    """Data-independent initial A/B input treatment; no fitted statistics."""

    descriptor_schema_sha256: str = VECTOR397_SCHEMA_SHA256
    fixed_log1p_mask_sha256: str = FIXED_LOG1P_MASK_SHA256

    def __post_init__(self) -> None:
        if self.descriptor_schema_sha256 != VECTOR397_SCHEMA_SHA256:
            raise StrictDataError("fixed-log1p schema identity mismatch")
        if self.fixed_log1p_mask_sha256 != FIXED_LOG1P_MASK_SHA256:
            raise StrictDataError("fixed-log1p mask identity mismatch")

    @property
    def artifact_sha256(self) -> str:
        return semantic_sha256(self.metadata())

    def transform(self, vector: Any, *, descriptor_schema_sha256: str | None = None) -> np.ndarray:
        result = fixed_log1p_vector397(
            vector,
            descriptor_schema_sha256=(
                descriptor_schema_sha256 or self.descriptor_schema_sha256
            ),
        )
        return np.ascontiguousarray(result, dtype=np.float32)

    def transform_row(self, row: Vector397Row) -> np.ndarray:
        if row.identity["descriptor_schema_sha256"] != self.descriptor_schema_sha256:
            raise StrictDataError("row descriptor schema identity mismatch")
        return self.transform(row.vector)

    def metadata(self) -> dict[str, Any]:
        return {
            "format": FIXED_LOG1P_FORMAT,
            "descriptor_schema_sha256": self.descriptor_schema_sha256,
            "fixed_log1p_mask_sha256": self.fixed_log1p_mask_sha256,
            "transform": "fixed_schema_log1p_no_zscore_no_clip",
            "fitted_statistics": False,
        }


@dataclass(frozen=True)
class Vector397Normalizer:
    """Frozen train-only statistics; arrays remain FP64 for reproducibility."""

    descriptor_schema_sha256: str
    training_event_count: int
    training_identity_sha256: str
    mean: np.ndarray
    std: np.ndarray
    dead_mask: np.ndarray
    # These are optional only to preserve construction compatibility with the
    # original in-memory API.  A persisted protocol artifact requires both.
    archive_identity_sha256: str | None = None
    training_prefix_global_index_sha256: str | None = None
    fixed_log1p_mask_sha256: str = FIXED_LOG1P_MASK_SHA256
    artifact_sha256: str | None = None

    def __post_init__(self) -> None:
        for name, value in (("mean", self.mean), ("std", self.std), ("dead_mask", self.dead_mask)):
            array = np.asarray(value)
            expected_dtype = np.bool_ if name == "dead_mask" else np.float64
            if array.shape != (VECTOR397_DIM,) or array.dtype != expected_dtype:
                raise StrictDataError(f"normalizer {name} must be {expected_dtype}[397]")
        if self.descriptor_schema_sha256 != VECTOR397_SCHEMA_SHA256:
            raise StrictDataError("normalizer schema identity mismatch")
        if self.training_event_count <= 0:
            raise StrictDataError("normalizer training identity is invalid")
        _sha256_hex(self.training_identity_sha256, "training_identity_sha256")
        if self.archive_identity_sha256 is not None:
            _sha256_hex(self.archive_identity_sha256, "archive_identity_sha256")
        if self.training_prefix_global_index_sha256 is not None:
            _sha256_hex(
                self.training_prefix_global_index_sha256,
                "training_prefix_global_index_sha256",
            )
        if self.fixed_log1p_mask_sha256 != FIXED_LOG1P_MASK_SHA256:
            raise StrictDataError("normalizer fixed-log1p mask identity mismatch")
        if self.artifact_sha256 is not None:
            _sha256_hex(self.artifact_sha256, "artifact_sha256")
        if not np.isfinite(self.mean).all() or not np.isfinite(self.std).all():
            raise StrictDataError("normalizer statistics are non-finite")
        if np.any(self.std < 0.0) or np.any(self.dead_mask != (self.std < DEAD_STD_THRESHOLD)):
            raise StrictDataError("normalizer dead-coordinate statistics are invalid")

    @property
    def normalizer_sha256(self) -> str:
        digest = hashlib.sha256()
        digest.update(self.descriptor_schema_sha256.encode("ascii"))
        digest.update(str(self.training_event_count).encode("ascii"))
        digest.update(self.training_identity_sha256.encode("ascii"))
        digest.update((self.archive_identity_sha256 or "").encode("ascii"))
        digest.update((self.training_prefix_global_index_sha256 or "").encode("ascii"))
        digest.update(self.fixed_log1p_mask_sha256.encode("ascii"))
        for value in (self.mean, self.std, self.dead_mask):
            digest.update(np.ascontiguousarray(value).tobytes())
        return digest.hexdigest()

    @property
    def training_prefix_global_index_digest(self) -> str | None:
        """Compatibility spelling for the exact canonical-index digest."""

        return self.training_prefix_global_index_sha256

    def transform(self, vector: Any, *, descriptor_schema_sha256: str | None = None) -> np.ndarray:
        value = fixed_log1p_vector397(
            vector,
            descriptor_schema_sha256=descriptor_schema_sha256 or self.descriptor_schema_sha256,
        )
        return self.transform_fixed_log1p(value)

    def transform_fixed_log1p(self, vector: Any) -> np.ndarray:
        """Apply this frozen normalizer to an already fixed-log1p vector.

        The composition cache stores the data-independent fixed-log1p view.
        This method is deliberately separate from :meth:`transform` so a
        caller cannot accidentally apply log1p twice when consuming that
        cache.  The returned value is FP64 for the in-memory compatibility
        API; production artifact writers explicitly cast to FP32 at the
        storage boundary.
        """

        value = np.asarray(vector, dtype=np.float32)
        if value.shape != (VECTOR397_DIM,) or not np.isfinite(value).all():
            raise StrictDataError("fixed-log1p Vector397 input must be finite FP32[397]")
        result = np.zeros(VECTOR397_DIM, dtype=np.float64)
        live = ~self.dead_mask
        result[live] = (value[live] - self.mean[live]) / self.std[live]
        if not np.isfinite(result).all():
            raise StrictDataError("normalized Vector397 is non-finite")
        return result

    def transform_fixed_log1p_batch(self, vectors: Any) -> np.ndarray:
        """Transform a finite FP32 ``[N,397]`` fixed-log1p batch.

        Statistics stay FP64 while the caller may cast the result to FP32 for
        the persisted fingerprint.  Keeping this vectorized path here avoids
        Python-level loops during the 1.5M-row materialization.
        """

        value = np.asarray(vectors, dtype=np.float32)
        if value.ndim != 2 or value.shape[1] != VECTOR397_DIM:
            raise StrictDataError("fixed-log1p Vector397 batch must have shape [N,397]")
        if not np.isfinite(value).all():
            raise StrictDataError("fixed-log1p Vector397 batch is non-finite")
        result = np.zeros(value.shape, dtype=np.float64)
        live = ~self.dead_mask
        result[:, live] = (
            value[:, live].astype(np.float64, copy=False) - self.mean[live]
        ) / self.std[live]
        if not np.isfinite(result).all():
            raise StrictDataError("normalized Vector397 batch is non-finite")
        return result

    def transform_row(self, row: Vector397Row) -> np.ndarray:
        if row.identity["descriptor_schema_sha256"] != self.descriptor_schema_sha256:
            raise StrictDataError("row descriptor schema identity mismatch")
        return self.transform(row.vector)

    def metadata(self) -> dict[str, Any]:
        return {
            "format": NORMALIZER_FORMAT,
            "descriptor_schema_sha256": self.descriptor_schema_sha256,
            "archive_identity_sha256": self.archive_identity_sha256,
            "training_event_count": int(self.training_event_count),
            "training_identity_sha256": self.training_identity_sha256,
            "training_prefix_global_index_sha256": self.training_prefix_global_index_sha256,
            "fixed_log1p_mask_sha256": self.fixed_log1p_mask_sha256,
            "dead_mask_sha256": _array_sha256(np.asarray(self.dead_mask, dtype=np.bool_)),
            "mean_dtype": str(np.asarray(self.mean).dtype),
            "std_dtype": str(np.asarray(self.std).dtype),
            "normalizer_sha256": self.normalizer_sha256,
            "artifact_sha256": self.artifact_sha256,
            "dead_std_threshold": DEAD_STD_THRESHOLD,
            "transform": "fixed_schema_log1p_then_train_only_population_zscore_no_clip",
        }

    def require_persistence_provenance(self) -> None:
        """Reject an in-memory-only normalizer at the artifact boundary."""

        if self.archive_identity_sha256 is None:
            raise StrictDataError("normalizer is missing source archive identity")
        if self.training_prefix_global_index_sha256 is None:
            raise StrictDataError(
                "normalizer is missing exact training-prefix/global-index digest"
            )

    def with_provenance(
        self,
        *,
        archive_identity_sha256: str,
        training_prefix_global_index_sha256: str,
    ) -> "Vector397Normalizer":
        """Return this normalizer bound to the source archive and exact prefix."""

        _sha256_hex(archive_identity_sha256, "archive_identity_sha256")
        _sha256_hex(
            training_prefix_global_index_sha256,
            "training_prefix_global_index_sha256",
        )
        return replace(
            self,
            archive_identity_sha256=archive_identity_sha256,
            training_prefix_global_index_sha256=training_prefix_global_index_sha256,
            artifact_sha256=None,
        )


def write_vector397_normalizer(
    path: str | Path,
    normalizer: Vector397Normalizer,
    *,
    archive_identity_sha256: str | None = None,
    training_prefix_global_index_sha256: str | None = None,
) -> dict[str, Any]:
    """Atomically write a self-checking Vector397 normalizer artifact.

    The canonical artifact consists of ``path`` (a compressed NumPy container)
    and ``path + '.json'``.  Both are installed together; an orphan is never
    accepted by the reader.  Provenance may be supplied here for compatibility
    with callers that fit first and bind the source at publication time.
    """

    candidate = normalizer
    if archive_identity_sha256 is not None or training_prefix_global_index_sha256 is not None:
        if archive_identity_sha256 is None or training_prefix_global_index_sha256 is None:
            raise StrictDataError("archive and prefix provenance must be supplied together")
        candidate = normalizer.with_provenance(
            archive_identity_sha256=archive_identity_sha256,
            training_prefix_global_index_sha256=training_prefix_global_index_sha256,
        )
    candidate.require_persistence_provenance()
    candidate = replace(candidate, artifact_sha256=None)
    metadata = _artifact_metadata(candidate)
    fixed_mask = _fixed_log1p_mask()
    artifact_sha256 = _artifact_semantic_sha256(
        metadata,
        mean=candidate.mean,
        std=candidate.std,
        dead_mask=candidate.dead_mask,
        fixed_log1p_mask=fixed_mask,
    )
    metadata = {**metadata, "artifact_sha256": artifact_sha256}
    destination = Path(path).resolve()
    sidecar = _sidecar_path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() or sidecar.exists():
        raise FileExistsError(f"normalizer artifact already exists: {destination}")
    temporary_base = Path(tempfile.mkdtemp(prefix=f".{destination.name}.", dir=str(destination.parent)))
    temporary = temporary_base / destination.name
    temporary_sidecar = temporary_base / sidecar.name
    try:
        # Use a file handle so NumPy cannot silently append ``.npz`` and alter
        # the caller's requested artifact path.
        with temporary.open("wb") as handle:
            np.savez_compressed(
                handle,
                mean=np.asarray(candidate.mean, dtype=np.float64),
                std=np.asarray(candidate.std, dtype=np.float64),
                dead_mask=np.asarray(candidate.dead_mask, dtype=np.bool_),
                fixed_log1p_mask=fixed_mask,
            )
        os.replace(temporary, destination)
        temporary_sidecar.write_text(
            _canonical_json({**metadata, "tensor_sha256": _file_sha256(destination)}) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary_sidecar, sidecar)
    except Exception:
        destination.unlink(missing_ok=True)
        sidecar.unlink(missing_ok=True)
        raise
    finally:
        temporary.unlink(missing_ok=True)
        temporary_sidecar.unlink(missing_ok=True)
        temporary_base.rmdir()
    reopened, _ = read_vector397_normalizer(destination)
    if reopened.artifact_sha256 != artifact_sha256:
        raise RuntimeError("normalizer artifact changed during reopen")
    return {
        "path": str(destination),
        "sidecar": str(sidecar),
        "artifact_sha256": artifact_sha256,
        "normalizer_sha256": reopened.normalizer_sha256,
    }


def read_vector397_normalizer(
    path: str | Path,
    *,
    expected_archive_identity_sha256: str | None = None,
    expected_training_prefix_global_index_sha256: str | None = None,
    expected_descriptor_schema_sha256: str = VECTOR397_SCHEMA_SHA256,
) -> tuple[Vector397Normalizer, dict[str, Any]]:
    """Reopen and fail closed on any provenance, tensor, or checksum drift."""

    source = Path(path).resolve()
    sidecar = _sidecar_path(source)
    if source.is_file() != sidecar.is_file() or not source.is_file():
        raise StrictDataError("normalizer artifact tensor and sidecar must be present together")
    try:
        metadata = json.loads(sidecar.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise StrictDataError("normalizer artifact sidecar is unreadable") from error
    if not isinstance(metadata, dict) or metadata.get("format") != NORMALIZER_FORMAT:
        raise StrictDataError("unsupported Vector397 normalizer artifact")
    if metadata.get("descriptor_schema_sha256") != expected_descriptor_schema_sha256:
        raise StrictDataError("normalizer artifact schema identity mismatch")
    if expected_descriptor_schema_sha256 != VECTOR397_SCHEMA_SHA256:
        raise StrictDataError("expected descriptor schema is not the frozen Vector397 schema")
    archive_sha = metadata.get("archive_identity_sha256")
    prefix_sha = metadata.get("training_prefix_global_index_sha256")
    if archive_sha is None or prefix_sha is None:
        raise StrictDataError("normalizer artifact omits required source provenance")
    _sha256_hex(archive_sha, "archive_identity_sha256")
    _sha256_hex(prefix_sha, "training_prefix_global_index_sha256")
    if expected_archive_identity_sha256 is not None and archive_sha != expected_archive_identity_sha256:
        raise StrictDataError("normalizer source archive identity changed")
    if expected_training_prefix_global_index_sha256 is not None and prefix_sha != expected_training_prefix_global_index_sha256:
        raise StrictDataError("normalizer training-prefix/global-index digest changed")
    if metadata.get("tensor_sha256") != _file_sha256(source):
        raise StrictDataError("normalizer artifact tensor checksum mismatch")
    try:
        with np.load(source, allow_pickle=False) as tensors:
            expected_names = {"mean", "std", "dead_mask", "fixed_log1p_mask"}
            if set(tensors.files) != expected_names:
                raise StrictDataError("normalizer artifact tensor schema changed")
            mean = np.array(tensors["mean"], dtype=np.float64, copy=True)
            std = np.array(tensors["std"], dtype=np.float64, copy=True)
            dead_mask = np.array(tensors["dead_mask"], dtype=np.bool_, copy=True)
            fixed_mask = np.array(tensors["fixed_log1p_mask"], dtype=np.bool_, copy=True)
    except (OSError, ValueError, KeyError) as error:
        raise StrictDataError("normalizer artifact tensors are unreadable") from error
    if not np.array_equal(fixed_mask, _fixed_log1p_mask()):
        raise StrictDataError("normalizer fixed-log1p mask changed")
    if metadata.get("fixed_log1p_mask_sha256") != _array_sha256(fixed_mask):
        raise StrictDataError("normalizer fixed-log1p mask checksum mismatch")
    if metadata.get("dead_mask_sha256") != _array_sha256(dead_mask):
        raise StrictDataError("normalizer dead-coordinate mask checksum mismatch")
    candidate = Vector397Normalizer(
        descriptor_schema_sha256=str(metadata["descriptor_schema_sha256"]),
        training_event_count=int(metadata["training_event_count"]),
        training_identity_sha256=str(metadata["training_identity_sha256"]),
        mean=mean,
        std=std,
        dead_mask=dead_mask,
        archive_identity_sha256=str(archive_sha),
        training_prefix_global_index_sha256=str(prefix_sha),
        fixed_log1p_mask_sha256=str(metadata["fixed_log1p_mask_sha256"]),
        artifact_sha256=None,
    )
    expected_metadata = _artifact_metadata(candidate)
    if metadata.get("normalizer_sha256") != candidate.normalizer_sha256:
        raise StrictDataError("normalizer semantic checksum mismatch")
    expected_artifact = _artifact_semantic_sha256(
        expected_metadata,
        mean=mean,
        std=std,
        dead_mask=dead_mask,
        fixed_log1p_mask=fixed_mask,
    )
    if metadata.get("artifact_sha256") != expected_artifact:
        raise StrictDataError("normalizer artifact checksum mismatch")
    candidate = replace(candidate, artifact_sha256=expected_artifact)
    return candidate, metadata


# Short aliases are intentionally provided for callers that use generic
# persistence names; the explicit names above remain the canonical API.
save_vector397_normalizer = write_vector397_normalizer
load_vector397_normalizer = read_vector397_normalizer


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _row_values(row: Vector397Row | Mapping[str, Any], vector: Any | None = None) -> tuple[np.ndarray, Mapping[str, Any]]:
    if isinstance(row, Vector397Row):
        return np.asarray(row.vector), row.identity
    if vector is None:
        vector = row.get("vector", row.get("descriptor_raw"))
    if vector is None:
        raise StrictDataError("normalizer row is missing Vector397 values")
    return np.asarray(vector), row


def fit_vector397_normalizer(
    rows: Iterable[Vector397Row | Mapping[str, Any]],
    *,
    descriptor_schema_sha256: str | None = None,
    dead_std_threshold: float = DEAD_STD_THRESHOLD,
    archive_identity_sha256: str | None = None,
    training_prefix_global_index_sha256: str | None = None,
) -> Vector397Normalizer:
    """Fit population z-score statistics using one FP64 Welford pass.

    Every input row must explicitly carry ``split=train``.  A validation/test
    row, a duplicate identity, a missing protocol field, or a schema mismatch
    aborts fitting before a normalizer is returned.
    """

    if dead_std_threshold != DEAD_STD_THRESHOLD:
        raise StrictDataError("the Vector397 dead-coordinate threshold is frozen at 1e-6")
    schema_sha = descriptor_schema_sha256 or VECTOR397_SCHEMA_SHA256
    if schema_sha != VECTOR397_SCHEMA_SHA256:
        raise StrictDataError("descriptor_schema_sha256 does not match frozen Vector397 schema")
    mean = np.zeros(VECTOR397_DIM, dtype=np.float64)
    m2 = np.zeros(VECTOR397_DIM, dtype=np.float64)
    seen: set[tuple[Any, ...]] = set()
    identity_rows: list[dict[str, Any]] = []
    count = 0
    for row in rows:
        values, identity_source = _row_values(row)
        row_split = row.split if isinstance(row, Vector397Row) else identity_source.get("split")
        if row_split != "train":
            raise StrictDataError("normalizer fitting accepts train rows only")
        if isinstance(row, Vector397Row):
            identity = dict(row.identity)
        else:
            from .data import validate_identity_row
            identity = validate_identity_row(row, expected_index=None)
        if identity["descriptor_schema_sha256"] != schema_sha:
            raise StrictDataError("training row descriptor schema identity mismatch")
        key = tuple(identity[field] for field in ("canonical_global_index", "prompt_key", "event_key", "payload_artifact_id"))
        if key in seen:
            raise StrictDataError("duplicate training identity")
        seen.add(key)
        transformed = fixed_log1p_vector397(values, descriptor_schema_sha256=schema_sha)
        count += 1
        delta = transformed - mean
        mean += delta / count
        m2 += delta * (transformed - mean)
        identity_rows.append(identity)
    if count == 0:
        raise StrictDataError("cannot fit Vector397 normalizer on empty training data")
    variance = np.maximum(m2 / count, 0.0)
    std = np.sqrt(variance, dtype=np.float64)
    dead = std < DEAD_STD_THRESHOLD
    training_identity = semantic_sha256(
        {
            "descriptor_schema_sha256": schema_sha,
            "split": "train",
            "rows": identity_rows,
        }
    )
    calculated_prefix_digest = semantic_sha256(
        [identity["canonical_global_index"] for identity in identity_rows]
    )
    if training_prefix_global_index_sha256 is not None and (
        training_prefix_global_index_sha256 != calculated_prefix_digest
    ):
        raise StrictDataError("training-prefix/global-index digest does not match rows")
    if archive_identity_sha256 is not None:
        _sha256_hex(archive_identity_sha256, "archive_identity_sha256")
    return Vector397Normalizer(
        descriptor_schema_sha256=schema_sha,
        training_event_count=count,
        training_identity_sha256=training_identity,
        mean=np.ascontiguousarray(mean),
        std=np.ascontiguousarray(std),
        dead_mask=np.ascontiguousarray(dead),
        archive_identity_sha256=archive_identity_sha256,
        training_prefix_global_index_sha256=(
            training_prefix_global_index_sha256 or calculated_prefix_digest
        ),
    )


def fit_vector397_normalizer_from_archive(
    manifest_path: str | Path,
    *,
    prefix: int,
    metadata_root: str | Path | None = None,
    samples_path: str | Path | None = None,
    supervision_path: str | Path | None = None,
    event_join_path: str | Path | None = None,
) -> Vector397Normalizer:
    """Fit a train-only normalizer from the exact canonical archive prefix.

    This is the only convenience path used by S5 launchers.  It supplies the
    explicit ``split=train`` marker (the published archive intentionally does
    not mutate its raw rows) and binds both the archive identity and the
    canonical ``0..prefix-1`` digest before returning.
    """

    source = Path(manifest_path).resolve()
    manifest = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(manifest, dict) or manifest.get("status") != "complete":
        raise StrictDataError("cannot fit a normalizer from an incomplete archive")
    archive_identity = manifest.get("archive_identity_sha256")
    if not isinstance(archive_identity, str):
        raise StrictDataError("Vector397 archive identity is missing")
    if int(prefix) <= 0:
        raise StrictDataError("normalizer prefix must be positive")
    prefix_digest = semantic_sha256(list(range(int(prefix))))

    def train_rows() -> Iterable[dict[str, Any]]:
        expected = 0
        for row in iter_vector397_rows(
            source,
            prefix=int(prefix),
            metadata_root=metadata_root,
            samples_path=samples_path,
            supervision_path=supervision_path,
            event_join_path=event_join_path,
        ):
            if row.canonical_global_index != expected:
                raise StrictDataError("normalizer source prefix is not canonical")
            expected += 1
            value = row.as_dict()
            value.update({"split": "train", "descriptor_raw": row.vector})
            yield value
        if expected != int(prefix):
            raise StrictDataError("normalizer source prefix ended early")

    return fit_vector397_normalizer(
        train_rows(),
        descriptor_schema_sha256=str(manifest.get("descriptor_schema_sha256")),
        archive_identity_sha256=archive_identity,
        training_prefix_global_index_sha256=prefix_digest,
    )
