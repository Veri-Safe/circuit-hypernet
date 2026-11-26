"""Collect the original experiment tests using their recorded directory layout."""

from pathlib import Path

from experiments.run import ROOT, TEST_TARGETS, prepare_workspace, read_layout


def pytest_sessionstart(session):
    config = session.config
    workspace = prepare_workspace()
    layout = read_layout()
    targets = []
    for arg in config.args:
        name, separator, node = str(arg).partition("::")
        path = Path(name).resolve()
        if path == ROOT:
            targets.extend(str(workspace / target) for target in TEST_TARGETS)
            targets.append(str(ROOT / "experiments" / "test_run.py"))
            continue
        if path.is_relative_to(ROOT / "experiments"):
            key = path.relative_to(ROOT).as_posix()
            if key in layout:
                targets.append(str(workspace / layout[key]) + separator + node)
                continue
            if path.is_dir():
                targets.extend(str(workspace / original) for release, original in layout.items()
                               if release.startswith(key + "/") and Path(original).match("test_*.py"))
                if (ROOT / "experiments" / "test_run.py").is_relative_to(path):
                    targets.append(str(ROOT / "experiments" / "test_run.py"))
                continue
        targets.append(arg)
    config.args[:] = targets
