"""Scientific correctness and reproducibility tests for the RL experiment."""
import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
from certified_dispatch.model import build_roads, random_stream
from certified_dispatch.rl import GoalRoads, Learner, locations, pair_costs, frozen_trips, save_checkpoint
from certified_dispatch.run_rl import experiment, validate

CONFIG = json.loads((Path(__file__).resolve().parents[1] / 'configs/rl_smoke.json').read_text())


class RLTests(unittest.TestCase):
    def setUp(self):
        self.roads = build_roads(3)
        self.env = GoalRoads(self.roads)

    def test_admissible_routes_and_public_bounds_by_enumeration(self):
        def costs(s, g, weights):
            if s == g:
                return [0.0]
            return [
                weights[self.env.edge[s, a]] + v
                for a in self.env.actions[g][s]
                for v in costs(int(self.env.next[s, a]), g, weights)
            ]

        for weights, maximize in (
            (self.roads.free, False),
            (self.roads.mean, False),
            (self.roads.free + self.roads.delay, True),
        ):
            values = self.env.route_values(weights, maximize)
            for s in range(self.env.nodes):
                for g in range(self.env.nodes):
                    possibilities = costs(s, g, weights)
                    self.assertAlmostEqual(values[s, g], max(possibilities) if maximize else min(possibilities))

    def test_exact_q_recovers_cost_objective_and_optimal_routes(self):
        learner = Learner.create(self.env, 42)
        optimal = self.env.route_values(self.roads.mean)
        for g in range(self.env.nodes):
            for s in range(self.env.nodes):
                for a in self.env.actions[g][s]:
                    learner.q[g, s, a] = self.roads.mean[self.env.edge[s, a]] + optimal[self.env.next[s, a], g]
        np.testing.assert_allclose(learner.prediction(), optimal)
        for (s, g), route in learner.frozen_paths().items():
            self.assertAlmostEqual(self.roads.mean[route].sum(), optimal[s, g])
            self.assertEqual(len(route), self.env.distance[s, g])

    def test_training_changes_values_and_visits(self):
        learner = Learner.create(self.env, 31)
        before = learner.q.copy()
        learner.train_to(200, CONFIG)
        self.assertFalse(np.array_equal(before, learner.q))
        self.assertEqual(int(learner.visits.sum()), learner.transitions)
        self.assertEqual(learner.episodes, 200)
        self.assertGreater(learner.transitions, 200)
        self.assertTrue(np.isfinite(learner.prediction()).all())
        self.assertTrue((learner.prediction() >= 0).all())

    def test_convergence_audit_distinguishes_policy_from_values(self):
        from certified_dispatch.diagnostics import policy_metrics

        learner = Learner.create(self.env, 42)
        optimal = self.env.route_values(self.roads.mean)
        mask = np.zeros_like(learner.q, dtype=bool)
        for g in range(self.env.nodes):
            for s in range(self.env.nodes):
                for a in self.env.actions[g][s]:
                    mask[g, s, a] = True
                    learner.q[g, s, a] = self.roads.mean[self.env.edge[s, a]] + optimal[self.env.next[s, a], g]
        chosen, exact = policy_metrics(self.env, learner.q, learner.tie_order, optimal, mask)
        for metric, value in exact.items():
            self.assertAlmostEqual(value, 0.0, msg=metric)
        # A common shift preserves routing choices but corrupts predicted cost.
        learner.q[mask] += 0.5
        shifted, errors = policy_metrics(self.env, learner.q, learner.tie_order, optimal, mask)
        np.testing.assert_array_equal(chosen, shifted)
        self.assertAlmostEqual(errors['route_gap_mean'], 0.0)
        self.assertAlmostEqual(errors['prediction_error_mean'], 0.5)
        self.assertAlmostEqual(errors['q_star_error_mean'], 0.5)
        self.assertAlmostEqual(errors['bellman_residual_max'], 0.5)

    def test_evaluation_does_not_change_training(self):
        first = Learner.create(self.env, 88)
        second = Learner.create(self.env, 88)
        first.train_to(100, CONFIG)
        second.train_to(20, CONFIG)
        old_q = second.q.copy()
        old_visits = second.visits.copy()
        paths = second.frozen_paths()
        second.prediction()
        trips = frozen_trips(self.roads, [paths] * 3, *locations(self.env, 3, 8, 0))
        trips.sample(0, 0, 20, random_stream(66))
        np.testing.assert_array_equal(old_q, second.q)
        np.testing.assert_array_equal(old_visits, second.visits)
        second.train_to(100, CONFIG)
        np.testing.assert_array_equal(first.q, second.q)
        np.testing.assert_array_equal(first.visits, second.visits)
        self.assertEqual(first.cumulative_cost, second.cumulative_cost)

    def test_private_learners_are_independent(self):
        a = Learner.create(self.env, 44)
        b = Learner.create(self.env, 45)
        a.train_to(40, CONFIG)
        b.train_to(40, CONFIG)
        self.assertFalse(np.shares_memory(a.q, b.q))
        self.assertFalse(np.array_equal(a.q, b.q))
        for learner in (a, b):
            for (s, g), path in learner.frozen_paths().items():
                self.assertEqual(len(path), self.env.distance[s, g])

    def test_selected_trip_oracle_matches_sampling_and_support(self):
        learners = [Learner.create(self.env, 10 + i) for i in range(3)]
        for learner in learners:
            learner.train_to(50, CONFIG)
        loc = locations(self.env, 3, 22, 0)
        trips = frozen_trips(self.roads, [l.frozen_paths() for l in learners], *loc)
        lower = pair_costs(self.env.route_values(self.roads.free), *loc)
        upper = pair_costs(self.env.route_values(self.roads.free + self.roads.delay, True), *loc)
        oracle = trips.expected_costs()
        for i in range(3):
            for k in range(3):
                samples = trips.sample(i, k, 10000, random_stream(99, i, k))
                self.assertTrue(np.all(samples >= lower[i, k] - 1e-8))
                self.assertTrue(np.all(samples <= upper[i, k] + 1e-8))
                self.assertAlmostEqual(samples.mean(), oracle[i, k], delta=0.04)

    def test_checkpoint_round_trip(self):
        learner = Learner.create(self.env, 12)
        learner.train_to(30, CONFIG)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'checkpoint.npz'
            save_checkpoint(path, learner, [learner])
            saved = np.load(path)
            restored = Learner.create(self.env, int(saved['reference_seed']))
            restored.q = saved['reference_q']
            restored.tie_order = saved['reference_tie_order']
            np.testing.assert_array_equal(learner.prediction(), restored.prediction())
            for key, p in learner.frozen_paths().items():
                np.testing.assert_array_equal(p, restored.frozen_paths()[key])

    def test_paired_end_to_end_cases(self):
        c = {
            **CONFIG,
            'side': 2,
            'fleet': 2,
            'seeds': 1,
            'batches': 1,
            'checkpoints': [0, 10],
            'evaluation_counts': [8],
        }
        validate(c)
        with tempfile.TemporaryDirectory() as directory:
            rows, training, coverage = experiment(c, Path(directory))
        self.assertEqual(len(rows), 2 * 2 * 5)
        for row in rows:
            if row['scenario'] == 'shared_current':
                self.assertEqual(row['max_policy_mismatch'], 0)
            if row['mechanism'] == 'dual_certified' and row['selected_upper_covered']:
                self.assertGreaterEqual(row['minimum_surplus'], c['eta'] - 1e-8)
            if row['checkpoint'] == 10:
                self.assertGreater(row['reference_training_transitions'], 0)
                if row['scenario'] == 'independent_current':
                    self.assertGreater(row['driver_training_transitions'], 0)
        self.assertEqual(len(training), 6)
        self.assertEqual(coverage[-1]['scenario'], 'both')


if __name__ == '__main__':
    unittest.main()
