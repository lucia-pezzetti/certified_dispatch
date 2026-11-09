"""Independent checks of the compiled kernel: the Hungarian solver, the confidence radii, service sampling,
restart exactness, and the information available to every decision."""
import itertools
import unittest
import numpy as np
from scipy.optimize import linear_sum_assignment
from certified_dispatch.optimistic_kernel import hungarian, route_sample, confidence_radius, METRICS
from certified_dispatch.platform import certificate, evaluate_payments
from certified_dispatch.run_manhattan_explore import fresh_state, chunk


def fixture():
    """Six drivers, two hubs, one synthetic edge per nonempty leg with traversal time uniform on [0.5, 1.5]."""
    h = 2
    paths = np.zeros((7, h, h, 1), dtype=np.int64)
    lengths = np.ones((7, h, h), dtype=np.int64)
    lengths[:, 0, 0] = 0
    lengths[:, 1, 1] = 0
    lo, hi, truth = np.zeros(h**3), np.zeros(h**3), np.zeros((6, h**3))
    for s in range(h):
        for p in range(h):
            for d in range(h):
                q = (s * h + p) * h + d
                legs = int(s != p) + int(p != d)
                lo[q], hi[q], truth[:, q] = 0.5 * legs, 1.5 * legs, float(legs)
    inputs = dict(
        support_lo=lo,
        support_hi=hi,
        true_contexts=truth,
        paths=paths,
        lengths=lengths,
        free=np.array([0.5]),
        delay=np.array([1.0]),
    )
    config = dict(
        block_size=10, alpha=0.05, eta=0.01, payment_budget=10.0, assignment_tolerance=0.01, exploration_budget=10.0
    )
    cell = dict(seed=7, rho=0.02, fleet=6, hubs=h, epochs=200)
    return config, cell, inputs


class KernelTests(unittest.TestCase):
    def test_hungarian_primal_dual_and_enumeration(self):
        rng = np.random.default_rng(19)
        for n in (2, 3, 6):
            for j in range(40):
                cost = rng.normal(size=(n, n)) if j % 2 else rng.integers(-3, 8, (n, n)).astype(float)
                sigma, phi, psi = hungarian(cost)
                value = cost[np.arange(n), sigma].sum()
                self.assertAlmostEqual(value, cost[linear_sum_assignment(cost)].sum())
                self.assertAlmostEqual(value, phi.sum() + psi.sum())
                self.assertGreaterEqual((cost - phi[:, None] - psi).min(), -1e-9)
                np.testing.assert_allclose(cost[np.arange(n), sigma], phi + psi[sigma], atol=1e-9)
                if n < 4:
                    self.assertAlmostEqual(
                        value, min(sum(cost[i, p[i]] for i in range(n)) for p in itertools.permutations(range(n)))
                    )

    def test_compiled_restart_exact_duplicate_contexts_and_reference_trace(self):
        config, cell, inputs = fixture()
        one, split = fresh_state(inputs, 6, 2, 7), fresh_state(inputs, 6, 2, 7)
        self.assertIn('squares', one)
        a, trace = chunk(config, cell, 0, 200, one, inputs)
        x, _ = chunk(config, cell, 0, 70, split, inputs)
        y, trace2 = chunk(config, cell, 70, 200, split, inputs)
        for key in one:
            np.testing.assert_array_equal(one[key], split[key])
        np.testing.assert_array_equal(a, np.concatenate((x, y)))
        np.testing.assert_array_equal(trace, trace2)
        self.assertTrue(np.all(one['counts'].sum(axis=(1, 2)) == 1200))
        # Sum of squares is at least count times the squared mean (Cauchy-Schwarz).
        self.assertTrue(np.all(one['squares'] * np.maximum(one['counts'], 1) >= one['totals'] ** 2 - 1e-9))
        self.assertTrue(np.all((one['squares'] > 0) == (one['counts'] > 0)))
        for arm, t in enumerate(trace):
            C, L, U, D = t[:4]
            sigma, executed, payments = t[4, 0].astype(int), t[4, 1].astype(int), t[4, 2]
            self.assertAlmostEqual(C[np.arange(6), sigma].sum(), C[linear_sum_assignment(C)].sum())
            cert = certificate(executed, payments, L, U[np.arange(6), executed])
            ev = evaluate_payments(D, executed, payments)
            self.assertLessEqual(ev['menu_regret'], cert['menu_bound'] + 1e-9)
            self.assertGreaterEqual(ev['minimum_surplus'], 0.01 - 1e-9)
        self.assertTrue((one['coverage_min'] >= -1e-9).all())

    def test_hoeffding_radius_at_one_observation_and_without_variance(self):
        for n in (1, 2, 10, 1000):
            for bernstein in (0, 1):
                if n == 1 or not bernstein:
                    self.assertAlmostEqual(
                        confidence_radius(bernstein, n, 0.3, 0.3 * n, 0.1 * n, 2.0, 24, 0.05),
                        2.0 * np.sqrt((np.log(2 * 24 / 0.05) + np.log(n) + np.log(n + 1)) / (2 * n)),
                    )

    @staticmethod
    def synthetic_sums(rng, reps, n, legs, delay):
        # Route costs: sums of independent uniform edge delays on [0, delay]; support [0, legs*delay].
        x = rng.uniform(0, delay, size=(reps, n, legs)).sum(axis=2)
        counts = np.arange(1, n + 1)
        totals, squares = np.cumsum(x, axis=1), np.cumsum(x * x, axis=1)
        return x, counts, totals, squares, totals / counts

    def test_bernstein_never_exceeds_hoeffding_plus_second_order_term(self):
        rng = np.random.default_rng(21)
        legs, delay, family, alpha = 24, 2.0, 448, 0.05
        width = legs * delay
        _, counts, totals, squares, means = self.synthetic_sums(rng, 4, 20000, legs, delay)
        nlog = np.log(2 * family / alpha) + np.log(counts) + np.log(counts + 1)
        blog = nlog + np.log(2)
        hoeffding = width * np.sqrt(nlog / (2 * counts))
        second = np.where(counts > 1, 7 * width * blog / (3 * np.maximum(counts - 1, 1)), 0.0)
        for r in range(4):
            bernstein = np.array(
                [
                    confidence_radius(
                        1, int(n), means[r, n - 1], totals[r, n - 1], squares[r, n - 1], width, family, alpha
                    )
                    for n in counts
                ]
            )
            self.assertTrue(np.all(bernstein <= hoeffding + second + 1e-9))
            self.assertEqual(bernstein[0], hoeffding[0])
            # At the production median count the variance-adaptive radius is far below Hoeffding.
            self.assertLess(bernstein[-1], hoeffding[-1] / 3)
            self.assertLess(bernstein[-1], 0.4)

    def test_bernstein_coverage_on_uniform_sums(self):
        # 2000 independent contexts observed 300 times each; a context fails when any count n <= 300
        # excludes the true mean. The budget is alpha over all counts (family 1).
        rng = np.random.default_rng(22)
        reps, n, legs, delay, alpha = 2000, 300, 6, 3.0, 0.05
        width, truth = legs * delay, legs * delay / 2
        _, counts, totals, squares, means = self.synthetic_sums(rng, reps, n, legs, delay)
        variance = np.maximum(0.0, (squares - totals * means) / np.maximum(counts - 1, 1))
        nlog = np.log(2 / alpha) + np.log(counts) + np.log(counts + 1)
        blog = nlog + np.log(2)
        radius = np.where(
            counts > 1,
            np.sqrt(2 * variance * blog / counts) + 7 * width * blog / (3 * np.maximum(counts - 1, 1)),
            width * np.sqrt(nlog / (2 * counts)),
        )
        for r, m in ((0, 1), (0, 2), (5, 17), (1999, 300)):
            self.assertAlmostEqual(
                radius[r, m - 1],
                confidence_radius(1, m, means[r, m - 1], totals[r, m - 1], squares[r, m - 1], width, 1, alpha),
            )
        clipped = np.minimum(radius, width)
        failures = np.any(np.abs(means - truth) > clipped, axis=1).mean()
        self.assertLessEqual(failures, alpha)
        # The interval is informative: at n = 300 it is much narrower than the support.
        self.assertLess(np.median(radius[:, -1]), width / 4)

    def test_exact_service_distribution(self):
        _, _, i = fixture()
        samples = np.array(
            [
                route_sample(30, e, 0, 0, 1, 0, 0.3, i['paths'], i['lengths'], i['free'], i['delay'])
                for e in range(30000)
            ]
        )
        self.assertGreaterEqual(samples.min(), 1.0)
        self.assertLessEqual(samples.max(), 3.0)
        self.assertAlmostEqual(samples.mean(), 2.0, delta=0.012)
        self.assertAlmostEqual(samples.var(), 1 / 6, delta=0.008)

    def test_each_decision_uses_only_previous_observations(self):
        config, cell, inputs = fixture()
        config['block_size'] = 1
        cell = dict(cell, seed=11)
        state = fresh_state(inputs, 6, 2, 11)
        for epoch in range(8):
            before = {key: value.copy() for key, value in state.items()}
            rows, traces = chunk(config, cell, epoch, epoch + 1, state, inputs)
            for a, tr in enumerate(traces):
                C, L, U, D = tr[:4]
                origins = tr[4, 5].astype(int)
                pickups, dropoffs = tr[5, 0].astype(int), tr[5, 1].astype(int)
                for i in range(6):
                    owner = i if a in (3, 4) else 0
                    for k in range(6):
                        q = (origins[i] * 2 + pickups[k]) * 2 + dropoffs[k]
                        count = before['counts'][a, owner, q]
                        lo, hi = inputs['support_lo'][q], inputs['support_hi'][q]
                        total, square = before['totals'][a, owner, q], before['squares'][a, owner, q]
                        mean = (lo + hi) / 2 if not count else total / count
                        family = 24 if a in (3, 4) else 4
                        allowance = 0.0 if a in (3, 4, 5) else 0.02 * (hi - lo)
                        if not count:
                            radius = np.inf
                        else:
                            nlog = np.log(2 * family * count * (count + 1) / 0.05)
                            if count > 1:
                                variance = max(0.0, (square - total * total / count) / (count - 1))
                                radius = np.sqrt(2 * variance * (nlog + np.log(2)) / count) + 7 * (hi - lo) * (
                                    nlog + np.log(2)
                                ) / (3 * (count - 1))
                            else:
                                radius = (hi - lo) * np.sqrt(nlog / (2 * count))
                        self.assertAlmostEqual(C[i, k], mean)
                        self.assertAlmostEqual(L[i, k], max(lo, mean - radius - allowance))
                        self.assertAlmostEqual(U[i, k], min(hi, mean + radius + allowance))
                executed, payments = tr[4, 1].astype(int), tr[4, 2]
                # Independent re-simulation of the six completed services checks the sufficient statistics.
                counts, totals = before['counts'][a].copy(), before['totals'][a].copy()
                squares = before['squares'][a].copy()
                for i in range(6):
                    k = executed[i]
                    s, p, d = origins[i], pickups[k], dropoffs[k]
                    v = route_sample(
                        11, epoch, i, s, p, d, 0.02, inputs['paths'], inputs['lengths'], inputs['free'], inputs['delay']
                    )
                    owner = i if a in (3, 4) else 0
                    q = (s * 2 + p) * 2 + d
                    counts[owner, q] += 1
                    totals[owner, q] += v
                    squares[owner, q] += v * v
                np.testing.assert_array_equal(state['counts'][a], counts)
                np.testing.assert_allclose(state['totals'][a], totals, rtol=0, atol=1e-9)
                np.testing.assert_allclose(state['squares'][a], squares, rtol=0, atol=1e-9)
                cert = certificate(executed, payments, L, U[np.arange(6), executed])
                ev = evaluate_payments(D, executed, payments)
                for metric, value in [
                    ('executed_bound', cert['control_bound']),
                    ('menu_bound', cert['menu_bound']),
                    ('menu_regret', ev['menu_regret']),
                    ('minimum_surplus', ev['minimum_surplus']),
                ]:
                    self.assertAlmostEqual(rows[0, a, METRICS.index(metric)], value)


if __name__ == '__main__':
    unittest.main()
