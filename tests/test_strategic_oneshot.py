"""Checks for the estimated-cheapest allocation rule, the vectorized one-shot reply and perception noise."""
import itertools
import unittest

import numpy as np
from certified_dispatch.platform import assign, mechanisms
from certified_dispatch.strategic_game import (
    ALL_RULES,
    action_values,
    allocate,
    compliance_values,
    no_contest_compliance,
    regrets,
    respond,
    surplus,
)


class CheapestRuleTests(unittest.TestCase):
    def test_action_values_against_explicit_winner_enumeration(self):
        sigma = np.array([1, 2, 0])
        pay = np.array([4.0, 5.0, 3.0])
        cost = np.array([[1.0, 4.0, 5.0], [3.0, 1.0, 4.0], [2.0, 6.0, 2.0]])
        estimate = np.array([[1.2, 3.9, 5.1], [3.1, 1.2, 3.9], [1.2, 6.2, 2.1]])  # ties on job 0 and job 1
        owner = np.argsort(sigma)
        for profile in itertools.product(range(-1, 3), repeat=3):
            for d in range(3):
                values = action_values(d, cost[d], pay, profile, sigma, 'cheapest', estimate=estimate)
                self.assertEqual(values[0], 0.0)
                for job in range(3):
                    changed = list(profile)
                    changed[d] = job
                    applicants = [i for i in range(3) if changed[i] == job]
                    winner = min(applicants, key=lambda i: (estimate[i, job], i != owner[job], i))
                    self.assertAlmostEqual(values[job + 1], int(winner == d) * (pay[job] - cost[d, job]))

    def test_allocation_lowest_estimate_wins_ties_to_owner_then_index(self):
        sigma = np.arange(3)
        estimate = np.array([[5.0, 2.0, 9.0], [5.0, 1.0, 9.0], [5.0, 1.0, 9.0]])
        served = allocate(np.array([1, 1, 1]), sigma, 'cheapest', np.random.default_rng(0), estimate=estimate)
        np.testing.assert_array_equal(served, [-1, 1, -1])  # driver 1 owns job 1 and ties with driver 2
        served = allocate(np.array([0, 0, 0]), sigma, 'cheapest', np.random.default_rng(0), estimate=estimate)
        np.testing.assert_array_equal(served, [0, -1, -1])  # three-way tie, owner wins
        served = allocate(np.array([2, 2, -1]), sigma, 'cheapest', np.random.default_rng(0), estimate=estimate)
        np.testing.assert_array_equal(served, [2, -1, -1])  # tie without the owner, lowest index wins
        with self.assertRaises(ValueError):
            allocate(np.array([0, 1, 2]), sigma, 'cheapest', np.random.default_rng(0))

    def test_dual_prices_have_no_regret_under_cheapest_with_exact_estimates_but_direct_prices_do(self):
        rng = np.random.default_rng(31)
        for n in (2, 4, 6):
            for _ in range(10):
                estimate = rng.uniform(1, 9, (n, n))
                assignment = assign(estimate)
                menus, _ = mechanisms(assignment, estimate[np.arange(n), assignment.sigma], 0.01)
                for rule in ALL_RULES:
                    regret = regrets(
                        estimate, assignment.sigma, menus['dual_base'], assignment.sigma, rule, estimate=estimate
                    )[0]
                    self.assertLessEqual(regret.max(), 1e-9)
        cost = np.array([[1.0, 2.0], [5.0, 2.5]])
        assignment = assign(cost)
        np.testing.assert_array_equal(assignment.sigma, [0, 1])
        menus, _ = mechanisms(assignment, cost[np.arange(2), assignment.sigma], 0.01)
        regret = regrets(cost, assignment.sigma, menus['direct_upper'], assignment.sigma, 'cheapest', estimate=cost)[0]
        self.assertAlmostEqual(regret[0], (2.51 - 2.0) - 0.01)  # driver 0 is estimated cheaper on job 1 and takes it
        self.assertAlmostEqual(regret[1], 0.0)
        self.assertAlmostEqual(
            regrets(cost, assignment.sigma, menus['direct_upper'], assignment.sigma, 'priority')[0].max(), 0.0
        )


class OneShotTests(unittest.TestCase):
    def test_compliance_values_match_profile_values_for_every_rule(self):
        rng = np.random.default_rng(7)
        for n in (2, 4, 6):
            for _ in range(8):
                cost, estimate = rng.uniform(0, 8, (n, n)), rng.uniform(0, 8, (n, n))
                sigma, pay = rng.permutation(n), rng.uniform(0, 8, n)
                for rule in ALL_RULES:
                    table = compliance_values(cost, sigma, pay, rule, estimate=estimate)
                    for d in range(n):
                        np.testing.assert_allclose(
                            table[d], action_values(d, cost[d], pay, sigma, sigma, rule, estimate=estimate)
                        )

    def test_one_step_from_compliance_equals_loop_version_and_zero_noise_is_exact(self):
        rng = np.random.default_rng(11)
        for _ in range(20):
            n = 5
            cost, estimate = rng.uniform(0, 8, (n, n)), rng.uniform(0, 8, (n, n))
            sigma, pay = rng.permutation(n), rng.uniform(0, 8, n)
            for rule in ALL_RULES:
                fast = respond(cost, sigma, pay, rule, 'one_step', np.random.default_rng(3), estimate=estimate)
                slow = respond(
                    cost,
                    sigma,
                    pay,
                    rule,
                    'one_step',
                    np.random.default_rng(3),
                    estimate=estimate,
                    initial_requests=sigma.copy(),
                )  # explicit profile takes the generic path
                np.testing.assert_array_equal(fast.requests, slow.requests)
                noiseless = respond(
                    cost, sigma, pay, rule, 'one_step', np.random.default_rng(3), estimate=estimate, tau=0.0
                )
                np.testing.assert_array_equal(fast.requests, noiseless.requests)

    def test_noise_under_priority_matches_probit_and_regret_is_at_true_costs(self):
        cost = np.array([[3.0, 9.0], [9.0, 3.0]])
        sigma = np.arange(2)
        pay = cost[np.arange(2), sigma] + 0.3  # surplus .3 for both drivers
        tau = 0.5
        rng = np.random.default_rng(5)
        complied, regret_ok = 0, True
        draws = 4000
        for _ in range(draws):
            r = respond(cost, sigma, pay, 'priority', 'one_step', rng, tau=tau)
            complied += int(np.sum(r.requests == sigma))
            idle = r.requests == -1
            expected = surplus(cost, sigma, pay)[idle].max() if idle.any() else 0.0
            regret_ok &= abs(r.max_regret - expected) < 1e-9  # an idle driver forgoes its true surplus
            self.assertTrue(np.all((r.requests == -1) | (r.requests == sigma)))  # other jobs are worth zero
        self.assertTrue(regret_ok)
        expected = no_contest_compliance(np.array([0.3]), tau)[0]
        self.assertAlmostEqual(expected, 0.7257, places=3)
        self.assertLess(abs(complied / (2 * draws) - expected), 0.03)
        np.testing.assert_array_equal(no_contest_compliance(np.array([-0.1, 0.0, 0.2]), 0.0), [0.0, 1.0, 1.0])

    def test_switching_threshold_keeps_small_gains_and_releases_large_ones(self):
        cost = np.array([[1.0, 2.0], [5.0, 2.5]])
        sigma = np.arange(2)
        pay = np.array([1.01, 2.51])  # direct prices: driver 0 gains .5 on job 1
        for threshold, expected in ((0.0, [1, 1]), (0.3, [1, 1]), (0.6, [0, 1])):
            r = respond(
                cost, sigma, pay, 'cheapest', 'one_step', np.random.default_rng(0), estimate=cost, threshold=threshold
            )
            np.testing.assert_array_equal(r.requests, expected)
        close = np.array([[1.0, 2.49], [5.0, 2.5]])  # driver 0 is estimated cheaper on job 1 by .01
        pay = np.array([1.01, 2.51])  # direct prices: gain .01 for driver 0
        tiny = respond(close, sigma, pay, 'cheapest', 'one_step', np.random.default_rng(0), estimate=close)
        np.testing.assert_array_equal(tiny.requests, [1, 1])  # an exact optimizer moves for .01 minutes
        held = respond(
            close, sigma, pay, 'cheapest', 'one_step', np.random.default_rng(0), estimate=close, threshold=0.05
        )
        np.testing.assert_array_equal(held.requests, [0, 1])
        with self.assertRaises(ValueError):
            respond(cost, sigma, pay, 'priority', 'one_step', np.random.default_rng(0), threshold=-0.1)

    def test_noisy_uniform_reply_can_contest_and_is_reproducible(self):
        cost = np.array([[9.0, 20.0, 20.0], [0.0, 9.0, 20.0], [0.0, 20.0, 9.0]])
        sigma, pay = np.arange(3), np.full(3, 10.0)
        a = respond(cost, sigma, pay, 'uniform', 'one_step', np.random.default_rng(2), tau=1.0)
        b = respond(cost, sigma, pay, 'uniform', 'one_step', np.random.default_rng(2), tau=1.0)
        np.testing.assert_array_equal(a.requests, b.requests)
        exact = respond(cost, sigma, pay, 'uniform', 'one_step', np.random.default_rng(2))
        np.testing.assert_array_equal(exact.requests, [0, 0, 0])
        with self.assertRaises(ValueError):
            respond(cost, sigma, pay, 'uniform', 'one_step', np.random.default_rng(2), tau=-1.0)


if __name__ == '__main__':
    unittest.main()
