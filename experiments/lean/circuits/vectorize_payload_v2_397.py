#!/usr/bin/env python3
"""Index S5 Payload v2 archives and materialize canonical raw 397D vectors."""

from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path
import sys


S5_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = S5_ROOT.parent
sys.path.insert(0, str(S5_ROOT / "src"))

from s5_lean_circuit.circuit.descriptor_v2_397 import SCHEMA_397  # noqa: E402
from s5_lean_circuit.circuit.descriptor_v2_397_artifact import (  # noqa: E402
    finalize_descriptor397_archive,
    write_worker_descriptor397,
)
from s5_lean_circuit.circuit.payload_v2_collection import (  # noqa: E402
    build_payload_v2_collection_index,
)


def repo_path(value: str | Path) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (REPO_ROOT / path).resolve()


def matched_paths(patterns: list[str]) -> list[Path]:
    result = {
        Path(item).resolve()
        for pattern in patterns
        for item in glob.glob(str(repo_path(pattern)), recursive=True)
    }
    if not result:
        raise FileNotFoundError(f"no files matched patterns: {patterns}")
    return sorted(result)


def main() -> None:
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest="command", required=True)

    schema = commands.add_parser("schema")
    schema.add_argument("--output", default=None)

    index = commands.add_parser("index")
    index.add_argument("--worker-manifest-glob", action="append", required=True)
    index.add_argument("--expected-workers", type=int, required=True)
    index.add_argument("--expected-events", type=int, required=True)
    index.add_argument("--output", required=True)

    worker = commands.add_parser("vectorize-worker")
    worker.add_argument("--collection-index", required=True)
    worker.add_argument("--worker-id", type=int, required=True)
    worker.add_argument("--output", required=True)
    worker.add_argument(
        "--trusted-fast-read",
        action="store_true",
        help="skip redundant outer shard-record hashes; shard readers still validate sidecars/tensors",
    )

    finalize = commands.add_parser("finalize")
    finalize.add_argument("--collection-index", required=True)
    finalize.add_argument("--worker-artifact-glob", action="append", required=True)
    finalize.add_argument("--output-dir", required=True)
    finalize.add_argument("--max-shard-bytes", type=int, default=512 * 1024**2)

    args = parser.parse_args()
    if args.command == "schema":
        value = {**SCHEMA_397.identity(), "schema_sha256": SCHEMA_397.sha256}
        text = json.dumps(value, indent=2, sort_keys=True) + "\n"
        if args.output is None:
            print(text, end="")
        else:
            output = repo_path(args.output)
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_text(text, encoding="utf-8")
            print(json.dumps({"output": str(output), "schema_sha256": SCHEMA_397.sha256}))
        return
    if args.command == "index":
        result = build_payload_v2_collection_index(
            matched_paths(args.worker_manifest_glob),
            output_path=repo_path(args.output),
            expected_workers=int(args.expected_workers),
            expected_events=int(args.expected_events),
        )
        print(
            json.dumps(
                {
                    "output": str(repo_path(args.output)),
                    "events": result["events"],
                    "workers": result["worker_count"],
                    "index_identity_sha256": result["index_identity_sha256"],
                },
                sort_keys=True,
            )
        )
        return
    if args.command == "vectorize-worker":
        result = write_worker_descriptor397(
            collection_index_path=repo_path(args.collection_index),
            worker_id=int(args.worker_id),
            output_path=repo_path(args.output),
            full_payload_verify=not bool(args.trusted_fast_read),
        )
        print(json.dumps(result, sort_keys=True))
        return
    result = finalize_descriptor397_archive(
        matched_paths(args.worker_artifact_glob),
        collection_index_path=repo_path(args.collection_index),
        output_dir=repo_path(args.output_dir),
        max_shard_bytes=int(args.max_shard_bytes),
    )
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
