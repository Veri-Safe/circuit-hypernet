from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import sys

import pytest

from s4_circuit.arc_targeted_repair_v1.identity_compat import (
    ARCHIVE_PATH, ARCHIVE_SHA256, CHECKPOINT_RELATIVE, archived_mount,
    read_archive_bytes, rebind_mount_fields, semantic_sha256,
)


def make_mount_alias(alias: Path, target: Path) -> None:
    try:
        alias.symlink_to(target, target_is_directory=True)
    except OSError as exc:
        if sys.platform != "win32" or exc.winerror != 1314:
            raise
        # An NTFS directory junction exercises the same directory-identity checks
        # when the account lacks the Windows symbolic-link privilege.
        from _winapi import CreateJunction
        CreateJunction(str(target), str(alias))


@pytest.mark.parametrize("name", ARCHIVE_SHA256)
def test_packaged_archive_matches_release_digest_and_rejects_edits(tmp_path, name):
    baseline = read_archive_bytes(Path(__file__).resolve().parents[2], name)
    copy = tmp_path / ARCHIVE_PATH / name
    copy.parent.mkdir(parents=True)
    copy.write_bytes(baseline + b" ")
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        read_archive_bytes(tmp_path, name)


def test_unknown_archive_path_is_rejected(tmp_path):
    with pytest.raises(ValueError, match="Unknown S0 archive file"):
        read_archive_bytes(tmp_path, "../checkpoint_identity.json")


def test_mount_is_discovered_from_restored_native_manifest(tmp_path, monkeypatch):
    monkeypatch.delenv("ARC_ARCHIVE_ROOT", raising=False)
    root = tmp_path / "workspace"
    root.mkdir()
    alias = tmp_path / "original-mount"
    make_mount_alias(alias, root)
    data = read_archive_bytes(Path(__file__).resolve().parents[2], "base_checkpoint_manifest.json")
    published = root / ARCHIVE_PATH / "base_checkpoint_manifest.json"
    published.parent.mkdir(parents=True)
    published.write_bytes(data)
    native = json.loads(data)
    assert not Path(native["hf_checkpoint"]).anchor
    native["hf_checkpoint"] = str(alias / native["hf_checkpoint"])
    restored = (root / CHECKPOINT_RELATIVE).parents[1] / published.name
    restored.parent.mkdir(parents=True)
    restored.write_text(json.dumps(native))
    assert archived_mount(root) == alias
    monkeypatch.setenv("ARC_ARCHIVE_ROOT", str(tmp_path))
    with pytest.raises(ValueError, match="mount alias"):
        archived_mount(root)


def test_mount_rebinding_preserves_semantics_and_requires_same_resource(tmp_path):
    actual = tmp_path / "actual"
    actual.mkdir()
    (actual / "config.yaml").write_text("native: true\n")
    old = tmp_path / "legacy"
    make_mount_alias(old, actual)
    original = {"config": str(actual / "config.yaml"), "rank": 16,
                "do_not_rewrite": str(actual / "unrelated")}
    result, proof = rebind_mount_fields(original, fields=(("config",),),
                                        current_root=actual, historical_root=old)
    assert result == {**original, "config": str(old / "config.yaml")}
    assert proof[0]["samefile"] and proof[0]["changed"]
    assert original["config"] == str(actual / "config.yaml")
    assert semantic_sha256(result) != semantic_sha256(original)
    changed_rank = deepcopy(result)
    changed_rank["rank"] = 8
    assert semantic_sha256(changed_rank) != semantic_sha256(result)


def test_different_mount_and_missing_file_are_rejected(tmp_path):
    actual, other = tmp_path / "actual", tmp_path / "other"
    actual.mkdir()
    other.mkdir()
    with pytest.raises(ValueError, match="same directory"):
        rebind_mount_fields({"config": str(actual / "file")}, fields=(("config",),),
                            current_root=actual, historical_root=other)
    old = tmp_path / "legacy"
    make_mount_alias(old, actual)
    with pytest.raises(ValueError, match="missing"):
        rebind_mount_fields({"config": str(actual / "file")}, fields=(("config",),),
                            current_root=actual, historical_root=old)


def test_paths_outside_roots_and_parent_traversal_are_rejected(tmp_path):
    actual = tmp_path / "actual"
    actual.mkdir()
    old = tmp_path / "legacy"
    make_mount_alias(old, actual)
    for path in (tmp_path / "outside", actual / ".." / "outside"):
        with pytest.raises(ValueError):
            rebind_mount_fields({"config": str(path)}, fields=(("config",),),
                                current_root=actual, historical_root=old)
