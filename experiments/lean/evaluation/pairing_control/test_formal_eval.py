"""CPU contract tests: selection leakage, immutable identities and coverage."""
from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest

from common import (TASKS, ATOMS, load_manifest, validate_selection, validate_counts,
                    verify_predictions, write_prediction_chunks, write_new, sha, merge_family_results)


def fixture():
    manifest = load_manifest()
    task = TASKS[0]
    spec = deepcopy(manifest["tasks"][task])
    spec["selected_step"] = 1000
    seen = list(TASKS[1:])
    rows, history = [], {}
    for step in [0] + manifest["validation"]["eligible_steps"]:
        # Step zero is strongest but ineligible. Positive steps tie: earliest wins.
        counts = {t: {"correct": 499 if step == 0 else 100, "total": 500} for t in seen}
        correct = sum(r["correct"] for r in counts.values())
        row = {"step": step, "task_counts": counts, "overall_correct": correct,
               "overall_total": 2000, "score": correct / 20}
        if step:
            row["checkpoint"] = {"global_step": step, "path": f"model_step{step:06d}.pt",
                                 "payload_sha256": spec["checkpoint_sha256"]}
        rows.append(row)
        history[step] = {**deepcopy(row), "format": "s5_composition_loco_visible_validation_v1",
                         "heldout_task": task, "visible_tasks": seen,
                         "validation_bank_identity": {"identity_sha256": manifest["validation"]["identity_sha256"],
                                                      "payload_sha256": manifest["validation"]["sha256"]}}
    selection = {"status": "complete", "config_sha256": spec["config_sha256"],
                 "heldout_task": task, "visible_tasks": seen,
                 "selection_metric": "visible_composition_macro_accuracy", "selection_denominator": 2000,
                 "step_zero_reference_only": True, "validation_steps": [0] + manifest["validation"]["eligible_steps"],
                 "atomic_validation": {"selection_uses_atomic": False}, "evaluations": rows,
                 "selected_step": 1000, "selected_correct": 400, "selected_score": 20}
    return manifest, task, spec, selection, history


class FormalContracts(unittest.TestCase):
    def check_selection(self, manifest, task, spec, selection, history):
        return validate_selection(selection, task, spec, manifest["validation"], history)

    def test_dev_earliest_tie_and_step_zero_exclusion(self):
        m, t, spec, selection, history = fixture()
        self.assertEqual(self.check_selection(m, t, spec, selection, history)["selected_step"], 1000)
        spec["selected_step"] = selection["selected_step"] = 2000
        with self.assertRaisesRegex(ValueError, "earliest dev winner"):
            self.check_selection(m, t, spec, selection, history)

    def test_heldout_cannot_enter_selection(self):
        m, t, spec, selection, history = fixture()
        selection["visible_tasks"][-1] = t
        with self.assertRaises(ValueError):
            self.check_selection(m, t, spec, selection, history)

    def test_formal_bank_cannot_be_used_as_development(self):
        m, t, spec, selection, history = fixture()
        history[1000]["validation_bank_identity"]["identity_sha256"] = m["suites"]["suite1"]["identity_sha256"]
        with self.assertRaisesRegex(ValueError, "bank mismatch"):
            self.check_selection(m, t, spec, selection, history)

    def test_missing_validation_candidate_rejected(self):
        m, t, spec, selection, history = fixture()
        del history[8000]
        with self.assertRaisesRegex(ValueError, "incomplete"):
            self.check_selection(m, t, spec, selection, history)

    def test_frozen_checkpoint_binding(self):
        m, t, spec, selection, history = fixture()
        selection["evaluations"][1]["checkpoint"]["payload_sha256"] = "0" * 64
        with self.assertRaisesRegex(ValueError, "checkpoint binding"):
            self.check_selection(m, t, spec, selection, history)

    def test_full_counts_and_prediction_coverage(self):
        tasks = ATOMS + TASKS
        counts = {t: {"total": 500, "correct": 0} for t in tasks}
        rows = [{"canonical_global_index": i, "target_event_key": f"k{i}",
                 "task_type": tasks[i // 500], "correct": False} for i in range(4500)]
        result = {"task_counts": counts, "predictions": rows, "event_count": 4500}
        verify_predictions(result, tasks)
        rows[1]["target_event_key"] = rows[0]["target_event_key"]
        with self.assertRaisesRegex(ValueError, "duplicate"):
            verify_predictions(result, tasks)
        counts[TASKS[0]]["total"] = 499
        with self.assertRaisesRegex(ValueError, "exactly 500"):
            validate_counts(counts, tasks)

    def test_chunks_and_history_never_overwritten(self):
        with tempfile.TemporaryDirectory() as temp:
            directory = Path(temp)
            rows = [{"index": i, "text": "x" * 30} for i in range(15)]
            parts = write_prediction_chunks(directory, rows, limit=250)
            restored = []
            for part in parts:
                path = directory / part["path"]
                self.assertLessEqual(path.stat().st_size, 250)
                self.assertEqual(sha(path), part["sha256"])
                restored.extend(json.loads(line) for line in path.read_text().splitlines())
            self.assertEqual(rows, restored)
            target = directory / "summary.json"
            write_new(target, {"old": True})
            with self.assertRaises(FileExistsError):
                write_new(target, {"new": True})
            self.assertEqual(json.loads(target.read_text()), {"old": True})

    def test_per_family_merge_preserves_predictions_and_rejects_mixed_family(self):
        tasks = ATOMS + TASKS
        results = []
        for index, task in enumerate(tasks):
            rows = [{"canonical_global_index": index * 500 + i, "target_event_key": f"{task}-{i}",
                     "task_type": task, "correct": i < index} for i in range(500)]
            results.append({"task_counts": {task: {"correct": index, "total": 500}},
                            "proof_groups": {task: {}}, "condition": "matched", "route": "circuit_a",
                            "condition_mapping": {"kind": "identity"}, "event_count": 500,
                            "predictions": list(reversed(rows))})
        merged = merge_family_results(results, tasks)
        self.assertEqual(len(merged["predictions"]), 4500)
        self.assertEqual(merged["predictions"][501]["target_event_key"], f"{tasks[1]}-1")
        self.assertEqual(sum(row["correct"] for row in merged["task_counts"].values()), 36)
        results[1]["predictions"][0]["task_type"] = tasks[0]
        with self.assertRaisesRegex(ValueError, "identity/coverage"):
            merge_family_results(results, tasks)

    def test_manifest_formal_mapping_and_frozen_steps(self):
        manifest = load_manifest()
        self.assertFalse(manifest["training_allowed"])
        self.assertEqual(manifest["inference"]["validation_batch_size"], 16)
        self.assertEqual(manifest["inference"]["batch_size"], 16)
        self.assertTrue(manifest["runtime_sources"])
        self.assertEqual(manifest["suites"]["suite1"]["seed_base"], 730000000000)
        self.assertTrue(manifest["suites"]["suite1"]["path"].endswith("/suite2.pt"))
        self.assertEqual(manifest["suites"]["suite2"]["seed_base"], 720000000000)
        self.assertTrue(manifest["suites"]["suite2"]["path"].endswith("/suite1.pt"))
        self.assertEqual([manifest["tasks"][t]["selected_step"] for t in TASKS],
                         [10000, 10000, 8000, 10000, 12000])


if __name__ == "__main__":
    unittest.main()
