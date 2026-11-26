"""Run the published experiments with their original import and artifact paths."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from tempfile import NamedTemporaryFile

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_WORKSPACE = ROOT / ".circuit-workspace"
TEST_TARGETS = (
    "s4_circuit/tests",
    "s5_lean_circuit/tests",
    "4server/lean_stage2_permuted_formal_eval_v1/test_formal_eval.py",
    "4server/lean_atomic_localize_v1/hn_readout_fusion_v1/targets/S_or_l/test_metrics.py",
)


def read_layout() -> dict[str, str]:
    with (ROOT / "experiments" / "paths.tsv").open(encoding="utf-8", newline="") as f:
        rows = csv.DictReader(f, delimiter="\t")
        return {row["release_path"]: row["original_path"] for row in rows}


def _destination(workspace: Path, name: str) -> Path:
    relative = Path(name)
    if relative.anchor or ".." in relative.parts or not relative.parts:
        raise ValueError(f"Invalid workspace path: {name}")
    target = workspace / relative
    if target.resolve() != target:
        raise ValueError(f"A mapped workspace path follows a link: {target}")
    return target


def _write(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = NamedTemporaryFile(dir=path.parent, delete=False)
    try:
        with tmp:
            tmp.write(data)
        os.replace(tmp.name, path)
    finally:
        Path(tmp.name).unlink(missing_ok=True)


def _circuit_profile(data: bytes) -> bytes:
    """Keep the original loaders and their checkpoint configuration hashes compatible."""
    import yaml

    profile = yaml.safe_load(data)
    path_fields = {
        "base": ("hf_checkpoint", "tokenizer"),
        "stage_a": ("profile", "production_lr_policy", "canonical_export_manifest"),
        "data": ("manifest_path", "train_path", "validation_path"),
        "traces": ("train_root", "validation_root"),
        "model": ("public_transcoder_root", "native_model", "output_root"),
        "artifacts": ("output_root",),
    }
    for section, keys in path_fields.items():
        fields = profile.get(section, {})
        for key in keys:
            path = fields.get(key)
            if path and not Path(path).is_absolute() and not path.startswith("$"):
                fields[key] = "${HC_ROOT}/" + path
    return yaml.safe_dump(profile, sort_keys=False).encode("utf-8")


def prepare_workspace(workspace: Path | None = None) -> Path:
    """Prepare only mapped release files. Leave user-supplied banks and runs in place."""
    from filelock import FileLock

    workspace = (workspace or DEFAULT_WORKSPACE).expanduser().resolve()
    if ROOT.is_relative_to(workspace) or (workspace.is_relative_to(ROOT) and workspace != DEFAULT_WORKSPACE):
        raise ValueError(f"Use {DEFAULT_WORKSPACE} or a workspace outside the checkout.")
    workspace.mkdir(parents=True, exist_ok=True)
    receipt = _destination(workspace, ".circuit-layout.json")
    lock = _destination(workspace, ".circuit-layout.lock")
    with FileLock(str(lock)):
        owned = json.loads(receipt.read_bytes()) if receipt.exists() else {}
        current: dict[str, str] = {}
        pending = {}
        for released, original in read_layout().items():
            src = ROOT / released
            if not src.resolve().is_relative_to(ROOT):
                raise ValueError(f"Release path outside checkout: {released}")
            dst = _destination(workspace, original)
            data = src.read_bytes()
            if src.parent == ROOT / "experiments/arc_agi_1/circuits/configs" and src.suffix == ".yaml":
                data = _circuit_profile(data)
            current[original] = digest = hashlib.sha256(data).hexdigest()
            if dst.exists():
                if original not in owned:
                    raise FileExistsError(f"Existing user file at {dst}. Use a separate workspace.")
                old = hashlib.sha256(dst.read_bytes()).hexdigest()
                if old == digest:
                    continue
                if old != owned[original]:
                    raise ValueError(f"The workspace copy was edited: {dst}. Move it aside to refresh.")
            pending[original] = data

        removed = []
        for original in owned.keys() - current.keys():
            dst = _destination(workspace, original)
            if dst.exists():
                if hashlib.sha256(dst.read_bytes()).hexdigest() != owned[original]:
                    raise ValueError(f"Retired workspace file was edited: {dst}. Move it aside to continue.")
                removed.append(dst)

        # Claim new names before copying. An interrupted refresh can resume safely.
        before_copy = {**current, **owned}
        if before_copy != owned:
            _write(receipt, json.dumps(before_copy, sort_keys=True).encode())
        for original, data in pending.items():
            _write(workspace / original, data)
        for dst in removed:
            dst.unlink()
        if before_copy != current:
            _write(receipt, json.dumps(current, sort_keys=True).encode())
    return workspace


def _original_arg(arg: str, layout: dict[str, str]) -> str:
    option, sep, value = arg.partition("=") if arg.startswith("--") and "=" in arg else ("", "", arg)
    path = Path(value)
    if path.is_relative_to(ROOT):
        path = path.relative_to(ROOT)
    return option + sep + layout.get(path.as_posix(), value)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--workspace", type=Path, help="work directory containing separately restored assets")
    parser.add_argument("--test", nargs=argparse.REMAINDER, metavar="PYTEST_ARG", help="run the original CPU test suites")
    parser.add_argument("script", nargs="?", help="path to a released experiment script, relative to the checkout")
    parser.add_argument("arguments", nargs=argparse.REMAINDER, help="arguments to the script")
    args = parser.parse_args(argv)
    if not args.script and args.test is None:
        parser.error("supply an experiment script or --test")
    layout = read_layout()
    if args.script:
        script = _original_arg(args.script, layout)
        if script not in layout.values() or not script.endswith(".py"):
            parser.error(f"unknown experiment script: {args.script}")
        command = [script, *(_original_arg(arg, layout) for arg in args.arguments)]
    try:
        workspace = prepare_workspace(args.workspace)
    except (OSError, ValueError) as error:
        parser.exit(1, f"{parser.prog}: {error}\n")
    if args.test is not None:
        command = ["-m", "pytest", "-c", str(ROOT / "pytest.ini"), f"--rootdir={workspace}",
                   f"--confcutdir={workspace}", *TEST_TARGETS, *args.test]
    env = {**os.environ, "HC_ROOT": str(workspace)}
    return subprocess.call([sys.executable, *command], cwd=workspace, env=env)


if __name__ == "__main__":
    raise SystemExit(main())
