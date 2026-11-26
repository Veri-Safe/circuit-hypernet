"""CPU checks for protocol boundaries, paired uncertainty and fair selection."""
import unittest

from metrics import TASKS, choose_best, compare, from_counts, paired_bootstrap, select_pair


def metric(target_drop, control_drops, n0=100):
    drops = dict(zip(("S_and", "S_or_r", "S_id"), control_drops))
    drops["S_or_l"] = target_drop
    return from_counts({t: {"N": n0 + 30, "n0": n0, "n1": n0 - drops[t]} for t in TASKS})


class MetricsTests(unittest.TestCase):
    def test_strict_ten_percentage_point_hn_boundary(self):
        # L=.4,C=.3: the exact 10-point gap must fail, not pass by float error.
        self.assertFalse(metric(40, [30, 30, 30])["hn_group_pass"])
        self.assertTrue(metric(40, [29, 30, 30])["hn_group_pass"])
        self.assertFalse(metric(39, [0, 0, 0])["hn_group_pass"])

    def test_controls_abs_before_task_aggregation(self):
        m = metric(50, [20, -10, 0])
        self.assertEqual(m["control_abs_net_correct_change"], 30)
        self.assertAlmostEqual(m["C"], .1)
        self.assertAlmostEqual(m["per_task"]["S_or_r"]["signed_relative_drop"], -.1)

    def test_zero_denominators_and_no_positive_effect(self):
        d, j = metric(20, [0, 0, 0]), metric(50, [0, 0, 0])
        self.assertEqual(j["R"], "+inf")
        c = compare(d, j)
        self.assertFalse(c["ratio_advantage"])
        self.assertIsNone(c["delta_R"])
        self.assertTrue(compare(metric(20, [10, 0, 0]), j)["ratio_advantage"])
        self.assertFalse(compare(metric(0, [10, 0, 0]), j)["ratio_advantage"])
        counts = {t: {"N": 10, "n0": 0, "n1": 0} for t in TASKS}
        self.assertFalse(from_counts(counts)["hn_group_pass"])
        self.assertIsNone(from_counts(counts)["L"])

    def test_direct_normal_best_not_weakened(self):
        weak = {"id": "weak", "K": 8, "metrics": metric(10, [20, 0, 0])}
        strong = {"id": "strong", "K": 8, "metrics": metric(30, [10, 0, 0])}
        self.assertEqual(choose_best([weak, strong], "direct")["id"], "strong")
        tiny = {"id": "tiny", "K": 8, "metrics": metric(1, [0, 0, 0])}
        self.assertEqual(choose_best([tiny, strong], "joint")["id"], "strong")

    def test_cross_k_uses_ratio_gap_not_G(self):
        candidates = []
        # K8 G=.006, ratio gap 1.8; K16 G=.06, ratio gap 1.5.
        for k, d, j in ((8, metric(40, [20, 0, 0]), metric(52, [20, 0, 0])),
                        (16, metric(40, [60, 0, 0]), metric(70, [60, 0, 0]))):
            for route, metrics in (("direct", d), ("joint", j)):
                candidates.append({"id": f"{k}-{route}", "K": k, "route": route, "metrics": metrics})
        selected = select_pair(candidates)
        self.assertEqual(selected["K"], 8)

    def test_paired_bootstrap_identical_groups_zero_G(self):
        rows = []
        for t in TASKS:
            for i in range(12):
                original = int(i % 3 != 0)
                edited = int(i % 4 != 0)
                rows.append({"task_type": t, "question_id": f"{t}:{i}", "outputs": {
                    "base_baseline": {"correct": original}, "base_direct": {"correct": edited},
                    "base_joint": {"correct": edited}}})
        result = paired_bootstrap(rows)
        self.assertEqual(result["intervals"]["G"]["lower"], 0)
        self.assertEqual(result["intervals"]["G"]["upper"], 0)
        self.assertEqual(result, paired_bootstrap(list(reversed(rows))))


if __name__ == "__main__":
    unittest.main()
