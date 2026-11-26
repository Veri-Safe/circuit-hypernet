import json
from pathlib import Path

import pytest

from s5_lean_circuit.composition_loco_config import load_config, matches_checkpoint_config
from s5_lean_circuit.hypernet.config import load_run_config


CONFIGS = Path(__file__).resolve().parents[1] / "configs"


def test_atomic_recipe_is_independent_of_execution_layout(monkeypatch):
    path = CONFIGS / "hypernet_atomic_1p2m_seed42/a_p03.json"
    recipe_digest = json.loads(path.read_text(encoding="utf-8"))["config_sha256"]
    for processes in (1, 4):
        monkeypatch.setenv("WORLD_SIZE", str(processes))
        monkeypatch.setenv("HYPERNET_MICROBATCH", "16")
        recipe = load_run_config(path)
        model = recipe.model
        assert model.expected_world_size == processes
        assert model.micro_batch_size * processes * model.gradient_accumulation_steps == model.global_batch_size
        assert recipe.config_sha256 == recipe_digest
    monkeypatch.setenv("HYPERNET_MICROBATCH", "3")
    with pytest.raises(ValueError, match="divide"):
        load_run_config(path)


def test_archived_checkpoint_requires_the_selected_file_for_its_recipe():
    path = CONFIGS / "composition_loco_circuit_retention20_v2/T_and_or/C25/circuit_a.json"
    recipe_sha = load_config(path).computed_sha256
    pinned = json.loads((CONFIGS / "selected_checkpoint_hashes.json").read_text(encoding="utf-8"))
    assert matches_checkpoint_config(recipe_sha, "archived", pinned[recipe_sha])
    assert not matches_checkpoint_config(recipe_sha, "archived", "0" * 64)
    assert not matches_checkpoint_config("0" * 64, "archived", pinned[recipe_sha])
