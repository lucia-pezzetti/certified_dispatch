"""Correctness checks for the scientific claims and the experiment boundary."""
import itertools
import unittest
import numpy as np
from certified_dispatch.model import build_roads, make_batch, random_stream
from certified_dispatch.platform import (
    assign,
    optimum,
    mechanisms,
    evaluate_payments,
    certificate,
    selected_upper,
    slack_certificate,
    training_estimate,
)


class ExperimentTests(unittest.TestCase):
    def test_assignment_and_duals_against_enumeration(self):
        rng = random_stream(902)
        for n in (2, 3, 4):
            for _ in range(8):
                C = rng.integers(0, 10, (n, n)).astype(float)
                result = assign(C)
                brute = min(sum(C[i, p[i]] for i in range(n)) for p in itertools.permutations(range(n)))
                self.assertAlmostEqual(optimum(C), brute)
                self.assertAlmostEqual(result.phi.sum() + result.psi.sum(), brute)
                self.assertGreaterEqual(result.slack.min(), -1e-8)
                self.assertAlmostEqual(evaluate_payments(C, result.sigma, result.payments)['menu_regret'], 0)

    def test_paper_example_and_allocation_probabilities(self):
        eps = 0.1
        D = np.array([[10 + eps, 11 - eps], [11, 10]])
        result = evaluate_payments(D, np.arange(2), np.array([11, 12]))
        self.assertAlmostEqual(D.trace(), optimum(D))
        self.assertAlmostEqual(result['menu_regret'], 2 * eps)
        self.assertEqual(result['priority_regret'], 0)
        # Alternative with utility 3, recommendation utility 1: random regret .5.
        result = evaluate_payments(np.array([[2.0, 0.0], [0.0, 2.0]]), np.arange(2), np.array([3.0, 3.0]))
        self.assertAlmostEqual(result['random_regret'], 0.5)
        self.assertAlmostEqual(result['menu_regret'], 2.0)
        self.assertEqual(result['priority_regret'], 0)
        result = evaluate_payments(np.ones((2, 2)) * 5, np.arange(2), np.zeros(2))
        self.assertEqual(result['priority_regret'], 5)
        self.assertEqual(result['random_regret'], 5)

    def test_interval_certificate_at_all_box_corners(self):
        L = np.array([[1.0, 2.0], [0.8, 1.5]])
        high = L + np.array([[0.4, 0.3], [0.2, 0.6]])
        assignment = assign((L + high) / 2)
        ids, sig = np.arange(2), assignment.sigma
        menus, _ = mechanisms(assignment, high[ids, sig], 0.01)
        for bits in itertools.product((0, 1), repeat=4):
            D = np.where(np.array(bits).reshape(2, 2), high, L)
            for name, pay in menus.items():
                cert = certificate(sig, pay, L, high[ids, sig])
                actual = evaluate_payments(D, sig, pay)
                self.assertLessEqual(actual['menu_regret'], cert['menu_bound'] + 1e-8)
                self.assertLessEqual(D[ids, sig].mean() - optimum(L) / 2, cert['control_bound'] + 1e-8)
                if name in ('dual_certified', 'direct_upper', 'direct_upper_matched'):
                    self.assertGreaterEqual(actual['minimum_surplus'], 0.01 - 1e-8)
        self.assertAlmostEqual(menus['dual_certified'].sum(), menus['uniform_matched'].sum())
        self.assertAlmostEqual(menus['dual_certified'].sum(), menus['direct_upper_matched'].sum())

    def test_slack_certificate_and_common_shift(self):
        C = np.array([[10.0, 11.0], [11.0, 10.0]])
        assignment = assign(C)
        radius = np.full_like(C, 0.2)
        bound = slack_certificate(assignment, radius)
        for bits in itertools.product((-1, 1), repeat=4):
            D = C + radius * np.array(bits).reshape(2, 2)
            actual = evaluate_payments(D, assignment.sigma, assignment.payments)
            self.assertLessEqual(actual['menu_regret'], bound + 1e-8)
        before = assignment.payments[None, :] - C
        after = (assignment.payments + 2)[None, :] - C
        np.testing.assert_allclose(before[:, 0] - before[:, 1], after[:, 0] - after[:, 1])

    def test_hidden_policy_costs_and_observation_support(self):
        roads = build_roads(3)
        for scenario in ('optimal', 'shared', 'heterogeneous'):
            trips, Cstar, L, B = make_batch(roads, 3, 34, 0, scenario)
            D = trips.expected_costs()
            self.assertTrue(np.all(L <= Cstar + 1e-8))
            self.assertTrue(np.all(Cstar <= D + 1e-8))
            self.assertTrue(np.all(D <= B + 1e-8))
            if scenario == 'optimal':
                np.testing.assert_allclose(D, Cstar)
            for i, k in itertools.product(range(3), repeat=2):
                obs = trips.sample(i, k, 20000, random_stream(44, i, k))
                self.assertTrue(np.all(obs >= L[i, k] - 1e-8))
                self.assertTrue(np.all(obs <= B[i, k] + 1e-8))
                self.assertAlmostEqual(obs.mean(), D[i, k], delta=0.04)
            changed = trips.adverse_policy_change()
            self.assertTrue(np.all(changed.expected_costs() >= D - 1e-8))

    def test_sample_prefix_and_stream_separation(self):
        trips, *_ = make_batch(build_roads(3), 3, 55, 0, 'heterogeneous')
        small = trips.sample(0, 0, 10, random_stream(10, 30))
        large = trips.sample(0, 0, 50, random_stream(10, 30))
        np.testing.assert_array_equal(small, large[:10])
        independent = trips.sample(0, 0, 10, random_stream(10, 40))
        self.assertFalse(np.array_equal(small, independent))

    def test_confidence_multiplicity_and_support_clipping(self):
        obs = np.zeros((2, 100))
        L = np.zeros(2)
        B = np.ones(2)
        actual = selected_upper(obs, L, B, 24, 0.025)
        expected = np.sqrt(np.log(24 / 0.025) / 200)
        np.testing.assert_allclose(actual, expected)
        self.assertTrue(np.all(selected_upper(obs, L, B, 48, 0.025) > actual))
        np.testing.assert_allclose(selected_upper(np.ones((2, 1)), L, B, 24, 0.025), B)
        _, e = training_estimate(np.zeros((2, 2, 100)), np.zeros((2, 2)), np.ones((2, 2)), 3, 0.025)
        np.testing.assert_allclose(e, np.sqrt(np.log(2 * 4 * 3 / 0.025) / 200))

    def test_invalid_assignment(self):
        for matrix in (np.array([[np.inf, 0], [0, 1]]), np.ones((2, 3)), np.ones((1, 1))):
            with self.assertRaises(ValueError):
                assign(matrix)


if __name__ == '__main__':
    unittest.main()
