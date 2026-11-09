"""Statistical and benchmark checks for the revised grid experiment."""
import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
from certified_dispatch.interval_study import mean_intervals
from certified_dispatch.run_rl import experiment, aggregate, budget_grid

CONFIG = json.loads((Path(__file__).resolve().parents[1] / 'configs/rl_grid10_smoke.json').read_text())


class IntervalTests(unittest.TestCase):
    def test_eq23_and_empirical_extrema_are_different_claims(self):
        # All-zero draws can arise from a nondegenerate Bernoulli distribution.
        samples = np.zeros((2, 2, 4))
        lo = np.zeros((2, 2))
        hi = np.ones((2, 2))
        L, U = mean_intervals(samples, lo, hi, 4, 0.05)
        np.testing.assert_allclose(L, 0)
        np.testing.assert_allclose(U, np.sqrt(np.log(160) / 8))
        empirical = mean_intervals(samples, samples.min(-1), samples.max(-1), 4, 0.05)
        self.assertTrue(np.all(empirical[1] < 0.5))  # True Bernoulli mean missed.

    def test_split_support_clipping_stays_ordered(self):
        samples = np.full((2, 2, 10), 4.0)
        L, U = mean_intervals(samples, np.zeros((2, 2)), np.ones((2, 2)), 4, 0.05)
        np.testing.assert_array_equal(L, U)
        np.testing.assert_array_equal(U, 1)

    def test_paired_all_pair_experiment_and_aggregation(self):
        c = {
            **CONFIG,
            'side': 2,
            'fleet': 2,
            'seeds': 1,
            'batches': 1,
            'checkpoints': [0, 10],
            'evaluation_counts': [1, 8],
            'support_test_count': 16,
        }
        with tempfile.TemporaryDirectory() as directory:
            rows, training, coverage = experiment(c, Path(directory))
        self.assertEqual(len(rows), 2 * 2 * 2 * 2 * 4 * 5)
        for r in rows:
            self.assertAlmostEqual(r['bound_excess'], r['lower_benchmark_gap'] + r['selected_upper_gap'])
            self.assertAlmostEqual(r['assignment_loss'], r['expected_cost_per_driver'] - r['opt_D_per_driver'])
            if r['bound_method'] == 'known_support':
                self.assertLessEqual(r['control_bound'], r['optimal_routing_bound'] + 1e-8)
                self.assertEqual(r['heldout_support_exceedance'], 0)
            if r['bound_method'].startswith('empirical'):
                self.assertEqual(r['certified_control_and_priority'], 0)
            if r['required_bounds_covered']:
                self.assertEqual(r['assignment_bound_violated'], 0)
                self.assertEqual(r['menu_bound_violated'], 0)
            if r['mechanism'] == 'dual_certified' and r['required_bounds_covered']:
                self.assertGreaterEqual(r['minimum_surplus'], c['eta'] - 1e-8)
        self.assertTrue(any(r['heldout_support_exceedance'] > 0 for r in rows if r['bound_method'] == 'empirical_same'))
        seeds, summary = aggregate(rows)
        self.assertEqual(len(seeds), len(rows))
        self.assertEqual({r['bound_method'] for r in summary}, set(r['bound_method'] for r in rows))
        self.assertEqual(len([r for r in coverage if r['checkpoint'] == 'all']), 4)
        for r in budget_grid(rows, c):
            if not r['theoretical_guarantee']:
                self.assertEqual(r['certified_fraction'], 0)


if __name__ == '__main__':
    unittest.main()
