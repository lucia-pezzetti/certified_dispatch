"""Tests of the closed-loop strategic Manhattan runner: menus, replies and allocations against the Python
reference modules, bit-identity of the certified-dual priority arm with the kernel's pooled-variance arm,
restart exactness, the bookkeeping of lost jobs and kept positions, and a saved run of the pooled loop."""
import json
import os
from pathlib import Path
import unittest
import numpy as np
from certified_dispatch.manhattan_kernel import ARMS as KERNEL_ARMS, METRICS as KERNEL_METRICS
from certified_dispatch.optimistic_kernel import hungarian, menu, stream as _stream
from certified_dispatch.platform import Assignment, mechanisms
from certified_dispatch.run_manhattan_scale import chunk as kernel_chunk, fresh_state as kernel_fresh_state
from certified_dispatch.run_manhattan_strategic import (
    PRICES,
    RULES,
    METRICS,
    ROWS,
    COMMON,
    arm_name,
    posted_menu,
    one_step_reply,
    allocate_rule,
    fresh_state,
    chunk,
)
from certified_dispatch.strategic_game import allocate, regrets, respond
from .test_manhattan_scale import synthetic_inputs

ROOT = Path(__file__).resolve().parents[1]
# A completed run of run_manhattan_scale with the paper's defaults; the archived-run test is skipped without it.
REFERENCE = Path(os.environ.get('CERTIFIED_DISPATCH_POOLING_RUN', ROOT / 'results/pooling'))
CONFIG = dict(block_size=10, alpha=1e-4, eta=0.01, payment_budget=1.0, assignment_tolerance=1.0)


def stream(seed, epoch, purpose):
    """Compiled functions return the uint64 state as a Python int; pass it back as uint64."""
    return np.uint64(_stream(seed, epoch, purpose))


def random_instance(rng, n):
    C = rng.uniform(5, 40, (n, n))
    D = C + rng.normal(0, 0.5, (n, n))
    sigma, phi, psi = hungarian(C)
    owner = np.argsort(sigma)
    return C, D, sigma, phi, psi, owner


def cell_for(arms, epochs, seed=11, rho=0.0, fleet=6, hubs=3):
    return dict(
        name='test', seed=seed, rho=rho, fleet=fleet, hubs=hubs, epochs=epochs, checkpoint_every=epochs, arms=arms
    )


class ReplyAndAllocationTests(unittest.TestCase):
    def test_posted_menus_match_platform_mechanisms(self):
        rng = np.random.default_rng(1)
        for n in (2, 4, 6):
            for _ in range(10):
                C, D, sigma, phi, psi, owner = random_instance(rng, n)
                U = C + rng.uniform(0, 2, (n, n))
                ids = np.arange(n)
                base = phi.max() + psi
                certified, shift = menu(base, U, sigma, 0.01)
                menus, shift2 = mechanisms(
                    Assignment(sigma, phi, psi, C - phi[:, None] - psi, base), U[ids, sigma], 0.01
                )
                self.assertAlmostEqual(shift, shift2)
                for code, price in enumerate(PRICES):
                    np.testing.assert_allclose(
                        posted_menu(code, base, certified, U, sigma, 0.01), menus[price], atol=1e-12
                    )
                self.assertAlmostEqual(menus['direct_upper_matched'].sum(), certified.sum())
                self.assertAlmostEqual(menus['uniform_matched'].sum(), certified.sum())

    def test_one_step_reply_matches_strategic_game(self):
        rng = np.random.default_rng(2)
        checked = {rule: 0 for rule in RULES}
        for n in (2, 4, 6):
            for _ in range(40):
                C, D, sigma, phi, psi, owner = random_instance(rng, n)
                U = C + rng.uniform(0, 2, (n, n))
                base = phi.max() + psi
                certified, _ = menu(base, U, sigma, 0.01)
                for code, price in enumerate(PRICES):
                    pay = posted_menu(code, base, certified, U, sigma, 0.01)
                    for rc, rule in enumerate(RULES):
                        requests, max_regret, _ = one_step_reply(D, C, sigma, owner, pay, rc, stream(3, 0, 6))
                        reference = respond(D, sigma, pay, rule, 'one_step', np.random.default_rng(0), estimate=C)
                        np.testing.assert_array_equal(requests, reference.requests)
                        self.assertAlmostEqual(max_regret, regrets(D, sigma, pay, sigma, rule, estimate=C)[0].max())
                        checked[rule] += int(np.any(requests != sigma))
        for rule in RULES:
            self.assertGreater(checked[rule], 20)  # the instances include deviations under every rule

    def test_priority_reply_is_the_sign_of_the_surplus(self):
        rng = np.random.default_rng(4)
        C, D, sigma, phi, psi, owner = random_instance(rng, 6)
        pay = np.empty(6)
        pay[sigma] = D[np.arange(6), sigma] + np.array([0.5, -0.5, 1e-12, -1e-6, 3.0, -3.0])
        requests, max_regret, _ = one_step_reply(D, C, sigma, owner, pay, 0, stream(3, 0, 6))
        np.testing.assert_array_equal(requests, [sigma[0], -1, sigma[2], -1, sigma[4], -1])
        self.assertAlmostEqual(max_regret, 3.0)

    def test_allocation_matches_strategic_game(self):
        rng = np.random.default_rng(5)
        for n in (2, 4, 6):
            for _ in range(60):
                C, D, sigma, phi, psi, owner = random_instance(rng, n)
                requests = rng.integers(-1, n, size=n)
                for rc, rule in enumerate(RULES):
                    served, _ = allocate_rule(requests, sigma, owner, C, rc, stream(3, 0, 5))
                    if rule != 'uniform':
                        np.testing.assert_array_equal(
                            served, allocate(requests, sigma, rule, np.random.default_rng(0), estimate=C)
                        )
                    for j in range(n):
                        applicants = np.flatnonzero(requests == j)
                        winners = np.flatnonzero(served == j)
                        self.assertEqual(len(winners), int(len(applicants) > 0))
                        if len(applicants):
                            self.assertIn(winners[0], applicants)
                    for d in range(n):
                        self.assertTrue(served[d] == -1 or served[d] == requests[d])
        # Ties under the cheapest rule go to the recommended driver, then to the lowest index.
        sigma = np.arange(3)
        owner = np.argsort(sigma)
        estimate = np.array([[5.0, 2.0, 9.0], [5.0, 1.0, 9.0], [5.0, 1.0, 9.0]])
        np.testing.assert_array_equal(
            allocate_rule(np.array([1, 1, 1]), sigma, owner, estimate, 2, stream(3, 0, 5))[0], [-1, 1, -1]
        )
        np.testing.assert_array_equal(
            allocate_rule(np.array([0, 0, 0]), sigma, owner, estimate, 2, stream(3, 0, 5))[0], [0, -1, -1]
        )
        np.testing.assert_array_equal(
            allocate_rule(np.array([2, 2, -1]), sigma, owner, estimate, 2, stream(3, 0, 5))[0], [2, -1, -1]
        )

    def test_uniform_allocation_is_uniform_and_shared_across_identical_requests(self):
        sigma = np.arange(4)
        owner = np.argsort(sigma)
        C = np.zeros((4, 4))
        requests = np.array([0, 0, 0, 2])
        wins = np.zeros(4)
        for epoch in range(6000):
            served, _ = allocate_rule(requests, sigma, owner, C, 1, stream(9, epoch, 5))
            again, _ = allocate_rule(requests, sigma, owner, C, 1, stream(9, epoch, 5))
            np.testing.assert_array_equal(served, again)
            wins[np.flatnonzero(served == 0)[0]] += 1
            self.assertEqual(served[3], 2)
        np.testing.assert_allclose(wins[:3] / 6000, 1 / 3, atol=0.03)
        self.assertEqual(wins[3], 0)


class ClosedLoopKernelTests(unittest.TestCase):
    def test_certified_priority_arm_is_the_kernel_pooled_variance_arm(self):
        """With covered intervals every driver's surplus is at least eta, so the certified dual under
        recommendation priority executes every recommendation and the two loops coincide bit for bit."""
        fleet, hubs, epochs = 6, 3, 300
        inputs = synthetic_inputs(fleet, hubs)
        arms = [arm_name('dual_certified', 'priority'), arm_name('uniform_matched', 'uniform')]
        cell = cell_for(arms, epochs, fleet=fleet, hubs=hubs)
        state = fresh_state(inputs, len(arms), fleet, hubs, cell['seed'])
        stats, trace, rows = chunk(CONFIG, cell, 0, epochs, state, inputs)
        ref_state = kernel_fresh_state(inputs, fleet, hubs, cell['seed'])
        ref_stats, ref_trace = kernel_chunk(CONFIG, cell, 0, epochs, ref_state, inputs)
        pv = KERNEL_ARMS.index('pooled_variance')
        kix = {m: i for i, m in enumerate(KERNEL_METRICS)}
        ix = {m: i for i, m in enumerate(METRICS)}
        self.assertEqual(ref_stats[:, pv, kix['covered']].sum(), epochs)
        for km, m in COMMON.items():
            np.testing.assert_array_equal(stats[:, 0, ix[m]], ref_stats[:, pv, kix[km]], err_msg=m)
        self.assertEqual(stats[:, 0, ix['completion']].sum(), epochs)
        self.assertEqual(stats[:, 0, ix['request_compliance']].sum(), epochs)
        self.assertEqual(stats[:, 0, ix['false_certificate']].sum(), 0.0)
        self.assertEqual(state['first_deviation'][0], -1)
        for key in ('counts', 'totals', 'squares', 'means', 'lows', 'highs'):
            np.testing.assert_array_equal(state[key][0, 0], ref_state[key][pv, 0], err_msg=key)
        np.testing.assert_array_equal(state['origins'][0], ref_state['origins'][pv])
        self.assertEqual(state['coverage_min'][0], ref_state['coverage_min'][pv])
        np.testing.assert_array_equal(trace[0], ref_trace[pv, :4])
        np.testing.assert_array_equal(rows[0, ROWS.index('sigma')], ref_trace[pv, 4, 0])
        np.testing.assert_array_equal(rows[0, ROWS.index('payments')], ref_trace[pv, 4, 1])
        np.testing.assert_array_equal(rows[0, ROWS.index('phi')], ref_trace[pv, 4, 2])
        np.testing.assert_array_equal(rows[0, ROWS.index('psi')], ref_trace[pv, 4, 3])
        np.testing.assert_array_equal(rows[0, ROWS.index('origins')], ref_trace[pv, 4, 4])
        np.testing.assert_array_equal(rows[0, ROWS.index('pickups')], ref_trace[pv, 5, 0])
        np.testing.assert_array_equal(rows[0, ROWS.index('dropoffs')], ref_trace[pv, 5, 1])
        np.testing.assert_array_equal(rows[0, ROWS.index('served')], rows[0, ROWS.index('sigma')])
        # The second arm shares demand and positions at the start but not afterwards.
        self.assertLess(stats[:, 1, ix['completion']].sum(), epochs)
        self.assertGreater(state['first_deviation'][1], -1)

    def test_restart_exact(self):
        fleet, hubs, epochs = 6, 3, 200
        inputs = synthetic_inputs(fleet, hubs)
        arms = [arm_name(p, r) for p in PRICES for r in RULES]
        cell = cell_for(arms, epochs, fleet=fleet, hubs=hubs)
        one = fresh_state(inputs, len(arms), fleet, hubs, cell['seed'])
        split = fresh_state(inputs, len(arms), fleet, hubs, cell['seed'])
        a, trace, rows = chunk(CONFIG, cell, 0, epochs, one, inputs)
        x, _, _ = chunk(CONFIG, cell, 0, 70, split, inputs)
        y, trace2, rows2 = chunk(CONFIG, cell, 70, epochs, split, inputs)
        for key in one:
            np.testing.assert_array_equal(one[key], split[key], err_msg=key)
        np.testing.assert_array_equal(a, np.concatenate((x, y)))
        np.testing.assert_array_equal(trace, trace2)
        np.testing.assert_array_equal(rows, rows2)
        ix = {m: i for i, m in enumerate(METRICS)}
        # Executed services are the only entries of the histories.
        np.testing.assert_array_equal(
            one['counts'].sum(axis=(1, 2)), np.rint(a[:, :, ix['completion']].sum(axis=0) * fleet)
        )
        self.assertTrue(np.all(one['squares'] * np.maximum(one['counts'], 1) >= one['totals'] ** 2 - 1e-9))
        self.assertTrue((one['coverage_min'] >= -1e-9).all())

    def test_lost_jobs_kept_positions_and_payments(self):
        """Epoch by epoch: unrequested jobs are not executed, idle and rejected drivers keep their position,
        served drivers move to the dropoff, payments are those of the executed jobs, and a false certificate
        is a certified epoch with a deviation."""
        fleet, hubs = 6, 3
        inputs = synthetic_inputs(fleet, hubs)
        arms = [arm_name(p, r) for p in PRICES for r in RULES]
        cell = cell_for(arms, 1, fleet=fleet, hubs=hubs)
        # Every epoch passes the two tests here, so a false certificate is exactly an epoch with a deviation.
        config = dict(CONFIG, block_size=1, payment_budget=1e6, assignment_tolerance=1e6)
        state = fresh_state(inputs, len(arms), fleet, hubs, cell['seed'])
        ix = {m: i for i, m in enumerate(METRICS)}
        lost, deviations, false_certificates = 0, 0, 0
        for epoch in range(120):
            before = {k: v.copy() for k, v in state.items()}
            stats, trace, rows = chunk(config, cell, epoch, epoch + 1, state, inputs)
            for a in range(len(arms)):
                sigma = rows[a, ROWS.index('sigma')].astype(int)
                requests = rows[a, ROWS.index('requests')].astype(int)
                served = rows[a, ROWS.index('served')].astype(int)
                pay = rows[a, ROWS.index('payments')]
                origins = rows[a, ROWS.index('origins')].astype(int)
                pickups, dropoffs = rows[a, ROWS.index('pickups')].astype(int), rows[a, ROWS.index('dropoffs')].astype(
                    int
                )
                np.testing.assert_array_equal(origins, before['origins'][a])
                self.assertEqual(sorted(sigma.tolist()), list(range(fleet)))
                executed = np.zeros(hubs**3, np.int64)
                for i in range(fleet):
                    if served[i] >= 0:
                        self.assertEqual(served[i], requests[i])
                        self.assertEqual(state['origins'][a, i], dropoffs[served[i]])
                        executed[(origins[i] * hubs + pickups[served[i]]) * hubs + dropoffs[served[i]]] += 1
                    else:
                        self.assertEqual(state['origins'][a, i], origins[i])
                for j in range(fleet):
                    self.assertEqual(int(np.sum(served == j)), int(np.any(requests == j)))
                np.testing.assert_array_equal(state['counts'][a, 0] - before['counts'][a, 0], executed)
                self.assertAlmostEqual(stats[0, a, ix['total_payment']], pay[served[served >= 0]].sum())
                self.assertAlmostEqual(stats[0, a, ix['offered_payment']], pay.sum())
                self.assertEqual(stats[0, a, ix['completion']], np.mean(served >= 0))
                self.assertEqual(
                    stats[0, a, ix['false_certificate']],
                    float(rows[a, ROWS.index('flags'), 0] == 1.0 and np.any(requests != sigma)),
                )
                lost += int(np.sum(served < 0))
                deviations += int(np.any(requests != sigma))
                false_certificates += int(stats[0, a, ix['false_certificate']])
                self.assertEqual(rows[a, ROWS.index('flags'), 0], 1.0)
                if arms[a] == arm_name('dual_certified', 'priority'):
                    self.assertTrue(np.array_equal(served, sigma))
        self.assertGreater(lost, 0)
        self.assertGreater(deviations, 0)
        self.assertEqual(false_certificates, deviations)
        self.assertGreater(false_certificates, 0)

    def test_archived_scale_run_first_chunk(self):
        """The certified-dual priority arm reproduces the first 200,000 epochs of the saved pooled-variance
        trajectory of the N = 6, rho = 0 cell with seed 20264800: block statistics and the saved decision trace."""
        cell_dir = REFERENCE / 'fleet6_hubs8_rho0_seed20264800'
        if not (cell_dir / 'blocks_0000200000.npz').exists():
            self.skipTest('No saved run of run_manhattan_scale at ' + str(REFERENCE))
        rc = json.loads((REFERENCE / 'config.json').read_text())
        inputs = dict(np.load(REFERENCE / 'inputs' / 'fleet6_hubs8_rho0_seed20264800.npz'))
        arms = [arm_name('dual_certified', 'priority')]
        cell = dict(
            name='fleet6_hubs8_rho0_seed20264800',
            seed=20264800,
            rho=0.0,
            fleet=6,
            hubs=8,
            epochs=200_000,
            checkpoint_every=200_000,
            arms=arms,
        )
        config = dict(
            block_size=rc['block_size'],
            alpha=rc['alpha'],
            eta=rc['eta'],
            payment_budget=rc['payment_budget'],
            assignment_tolerance=rc['assignment_tolerance'],
        )
        state = fresh_state(inputs, 1, 6, 8, 20264800)
        stats, trace, rows = chunk(config, cell, 0, 200_000, state, inputs)
        archived = dict(np.load(cell_dir / 'blocks_0000200000.npz'))
        pv = KERNEL_ARMS.index('pooled_variance')
        kix = {m: i for i, m in enumerate(KERNEL_METRICS)}
        ix = {m: i for i, m in enumerate(METRICS)}
        for km, m in COMMON.items():
            np.testing.assert_array_equal(stats[:, 0, ix[m]], archived['stats'][:, pv, kix[km]], err_msg=m)
        old = archived['trace'][pv]
        np.testing.assert_array_equal(trace[0], old[:4])
        np.testing.assert_array_equal(rows[0, ROWS.index('sigma')], old[4, 0])
        np.testing.assert_array_equal(rows[0, ROWS.index('served')], old[4, 0])
        np.testing.assert_array_equal(rows[0, ROWS.index('payments')], old[4, 1])
        np.testing.assert_array_equal(rows[0, ROWS.index('phi')], old[4, 2])
        np.testing.assert_array_equal(rows[0, ROWS.index('psi')], old[4, 3])
        np.testing.assert_array_equal(rows[0, ROWS.index('origins')], old[4, 4])
        np.testing.assert_array_equal(rows[0, ROWS.index('pickups')], old[5, 0])
        np.testing.assert_array_equal(rows[0, ROWS.index('dropoffs')], old[5, 1])
        self.assertEqual(state['first_deviation'][0], -1)
        self.assertEqual(stats[:, 0, ix['completion']].sum(), 200_000)


if __name__ == '__main__':
    unittest.main()
