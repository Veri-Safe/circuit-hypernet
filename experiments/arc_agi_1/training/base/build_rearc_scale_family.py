#!/usr/bin/env python3
"""Build the strict nested RE-ARC-{8,16,32,64} dataset family."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import shutil
import time
import uuid
from pathlib import Path
from typing import Any

import numpy as np
from datasets import load_from_disk

from nvarc_qwen3.rearc_subset_common import (
    SourceContract,
    SubsetSpec,
    inspect_source,
    select_subset_indices,
    validate_derived,
)


SOURCE = SourceContract(
    fingerprint="332b5af0e60bee1d", rows=102_392, puzzles=400
)
ALGORITHM = "rearc128_per_puzzle_sha256_v1"
SELECTION_SEED = 42
SCALES = (8, 16, 32, 64)
EXPECTED = {
    8: {
        "indices_sha256": "17669730a429768a3a3a97e626086f132c0c82b7c12fc7daa1e9ba5e76037f7b",
        "assistant_events": 20_743,
    },
    16: {
        "indices_sha256": "a92809034c866cea50540af0f98e8f2bdd23d5b9febe6b97febfb9fcf65f5e1e",
        "assistant_events": 41_568,
    },
    32: {
        "indices_sha256": "f91ca136790e5c52cee7f446ed2b6866565c6c7c09905046907221bc6ce3b12e",
        "assistant_events": 83_222,
    },
    64: {
        "indices_sha256": "31d8071a799d79827d7a729527b381f12d4a4a71170deead386947a424fad1d4",
        "assistant_events": 166_337,
    },
}


def _indices_hash(indices: np.ndarray) -> str:
    """Match the existing RE-ARC-128 manifest: hash the complete .npy file."""

    buffer = io.BytesIO()
    np.save(buffer, np.asarray(indices, dtype=np.int64), allow_pickle=False)
    return hashlib.sha256(buffer.getvalue()).hexdigest()


def _load_manifest(root: Path) -> dict[str, Any]:
    return json.loads((root / "manifest.json").read_text(encoding="utf-8"))


def validate_existing(root: Path) -> dict[str, Any]:
    manifest = _load_manifest(root)
    if manifest.get("format_version") != "rearc_nested_scale_family_v1":
        raise ValueError(f"Unexpected scale-family manifest: {root}")
    for scale in SCALES:
        entry = manifest["members"][str(scale)]
        dataset = load_from_disk(root / "hf" / f"rearc{scale}")
        if len(dataset) != 400 * scale:
            raise ValueError(f"RE-ARC-{scale} row count changed")
        if getattr(dataset, "_fingerprint", None) != entry["output_fingerprint"]:
            raise ValueError(f"RE-ARC-{scale} fingerprint differs from its manifest")
        indices = np.load(root / "indices" / f"rearc{scale}.npy", allow_pickle=False)
        if _indices_hash(indices) != EXPECTED[scale]["indices_sha256"]:
            raise ValueError(f"RE-ARC-{scale} index hash changed")
        if entry["indices_sha256"] != EXPECTED[scale]["indices_sha256"]:
            raise ValueError(f"RE-ARC-{scale} manifest hash changed")
    return manifest


def build_family(source_path: Path, output_root: Path) -> dict[str, Any]:
    if (output_root / "manifest.json").is_file():
        return validate_existing(output_root)
    output_root.parent.mkdir(parents=True, exist_ok=True)
    lock = output_root.parent / f".{output_root.name}.build.lock"
    try:
        lock.mkdir()
        owns_lock = True
    except FileExistsError:
        owns_lock = False
    if not owns_lock:
        deadline = time.monotonic() + 3600
        while time.monotonic() < deadline:
            if (output_root / "manifest.json").is_file():
                return validate_existing(output_root)
            time.sleep(5)
        raise TimeoutError(f"Timed out waiting for dataset builder lock: {lock}")

    temporary = output_root.parent / f".{output_root.name}.tmp-{os.getpid()}-{uuid.uuid4().hex}"
    try:
        source = load_from_disk(source_path)
        puzzle_names = inspect_source(source, SOURCE)
        temporary.mkdir()
        (temporary / "hf").mkdir()
        (temporary / "indices").mkdir()
        members: dict[str, Any] = {}
        selected_sets: dict[int, set[int]] = {}
        for scale in SCALES:
            spec = SubsetSpec(
                algorithm_version=ALGORITHM,
                seed=SELECTION_SEED,
                rows_per_puzzle=scale,
                output_split=f"rearc{scale}",
                expected_assistant_events=int(EXPECTED[scale]["assistant_events"]),
            )
            indices = select_subset_indices(
                puzzle_names, source_fingerprint=SOURCE.fingerprint, spec=spec
            )
            observed_hash = _indices_hash(indices)
            if observed_hash != EXPECTED[scale]["indices_sha256"]:
                raise ValueError(f"RE-ARC-{scale} selected index hash mismatch")
            derived = source.select(indices.tolist())
            stats = validate_derived(
                derived, source, source_contract=SOURCE, spec=spec
            )
            np.save(
                temporary / "indices" / f"rearc{scale}.npy",
                indices,
                allow_pickle=False,
            )
            derived.save_to_disk(temporary / "hf" / f"rearc{scale}")
            reloaded = load_from_disk(temporary / "hf" / f"rearc{scale}")
            stats = validate_derived(
                reloaded, source, source_contract=SOURCE, spec=spec
            )
            selected_sets[scale] = set(map(int, indices))
            members[str(scale)] = {
                "split": f"rearc{scale}",
                "rows_per_puzzle": scale,
                "indices_sha256": observed_hash,
                "output_fingerprint": getattr(reloaded, "_fingerprint", None),
                "stats": stats,
            }
        nesting = {
            f"rearc{small}_subset_of_rearc{large}": selected_sets[small] < selected_sets[large]
            for small, large in zip(SCALES, SCALES[1:])
        }
        if not all(nesting.values()):
            raise AssertionError(f"Strict nesting failed: {nesting}")
        manifest = {
            "format_version": "rearc_nested_scale_family_v1",
            "source_path": str(source_path),
            "source_fingerprint": SOURCE.fingerprint,
            "source_rows": SOURCE.rows,
            "algorithm_version": ALGORITHM,
            "selection_seed": SELECTION_SEED,
            "strict_nesting": nesting,
            "members": members,
        }
        (temporary / "manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        temporary.replace(output_root)
        return validate_existing(output_root)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)
        shutil.rmtree(lock, ignore_errors=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    manifest = build_family(args.source, args.output_root)
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
