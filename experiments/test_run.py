"""Regression tests for the public layout and the original scripts' working directory."""

import os

import pytest

from experiments import run


@pytest.fixture
def release(tmp_path, monkeypatch):
    root = tmp_path / "checkout"
    scripts = root / "experiments" / "lean"
    scripts.mkdir(parents=True)
    (root / "experiments" / "paths.tsv").write_text(
        "original_path\trelease_path\n"
        "original/scripts/entry.py\texperiments/lean/entry.py\n"
        "original/config.txt\texperiments/lean/settings.txt\n",
        encoding="utf-8",
    )
    (scripts / "entry.py").write_text(
        "from pathlib import Path\n"
        "import sys\n"
        "assert Path(__file__).resolve() == Path.cwd() / 'original/scripts/entry.py'\n"
        "assert Path(sys.argv[1]).read_text() == 'setting'\n"
        "sys.exit(17)\n",
        encoding="utf-8",
    )
    (scripts / "settings.txt").write_text("setting", encoding="utf-8")
    monkeypatch.setattr(run, "ROOT", root)
    monkeypatch.setattr(run, "DEFAULT_WORKSPACE", root / ".circuit-workspace")
    return scripts, tmp_path / "workspace"


def test_script_uses_original_paths_and_propagates_exit_code(release):
    _, workspace = release
    result = run.main(["--workspace", os.fspath(workspace), "experiments/lean/entry.py",
                       "experiments/lean/settings.txt"])
    assert result == 17


def test_refresh_keeps_local_edits_and_unmapped_assets(release):
    scripts, workspace = release
    run.prepare_workspace(workspace)
    user_asset = workspace / "model.txt"
    user_asset.write_text("restored model", encoding="utf-8")
    (scripts / "settings.txt").write_text("new release", encoding="utf-8")
    copy = workspace / "original/config.txt"
    run.prepare_workspace(workspace)
    assert copy.read_text() == "new release"
    copy.write_text("local edit", encoding="utf-8")
    with pytest.raises(ValueError, match="edited"):
        run.prepare_workspace(workspace)
    assert copy.read_text() == "local edit"
    assert user_asset.read_text() == "restored model"
