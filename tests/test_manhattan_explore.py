"""Tests of the exploration hub sweep: the optimistic kernel at a larger fleet and the audit checks."""
import unittest
import numpy as np
from certified_dispatch.optimistic_kernel import ARMS, METRICS
from certified_dispatch.run_manhattan_explore import fresh_state, chunk, guaranteed
from .test_manhattan_scale import synthetic_inputs


def run_synthetic(fleet, hubs, epochs, exploration_cap, rho=0.0, alpha=1e-4, seed=11, block=10):
    inputs = synthetic_inputs(fleet, hubs, rho)
    config = dict(
        block_size=block,
        alpha=alpha,
        eta=0.01,
        payment_budget=1.0,
        assignment_tolerance=1.0,
        exploration_budget=exploration_cap,
    )
    cell = dict(seed=seed, rho=rho, fleet=fleet, hubs=hubs, epochs=epochs)
    state = fresh_state(inputs, fleet, hubs, seed)
    stats, trace = chunk(config, cell, 0, epochs, state, inputs)
    return config, cell, inputs, state, stats, trace


class ExploreTests(unittest.TestCase):
    def test_independent_random_uses_own_certificate(self):
        fleet, hubs = 6, 3
        inputs = synthetic_inputs(fleet, hubs)
        config = dict(
            block_size=1, alpha=0.05, eta=0.01, payment_budget=1.0, assignment_tolerance=1.0, exploration_budget=1e9
        )
        cell = dict(seed=11, rho=0.0, fleet=fleet, hubs=hubs, epochs=1)
        state = fresh_state(inputs, fleet, hubs, cell['seed'])
        # The optimistic partner is already certified, while random has no data.
        for key in ('means', 'lows', 'highs'):
            state[key][6] = inputs['true_contexts'][0]
        old = {k: v.copy() for k, v in state.items()}
        coupled, _ = chunk(config, cell, 0, 1, old, inputs)
        independent, _ = chunk(config, cell, 0, 1, state, inputs, only_arm=7, independent_random=True)
        ix = {m: j for j, m in enumerate(METRICS)}
        self.assertEqual(coupled[0, 6, ix['certified']], 1)
        self.assertEqual(coupled[0, 7, ix['explored']], 0)
        self.assertEqual(independent[0, 7, ix['explored']], 1)
        self.assertEqual(independent[0, 7, ix['proposed_pass']], 0)
        self.assertEqual(state['counts'][:7].sum(), 0)
        self.assertEqual(state['counts'][7].sum(), fleet)
        # Once its own recommendation is certified, random must stop exploring.
        for key in ('means', 'lows', 'highs'):
            state[key][7] = inputs['true_contexts'][0]
        passed, _ = chunk(config, cell, 1, 2, state, inputs, only_arm=7, independent_random=True)
        self.assertEqual(passed[0, 7, ix['certified']], 1)
        self.assertEqual(passed[0, 7, ix['explored']], 0)

    def test_random_only_matches_joint_and_resumed_execution(self):
        fleet, hubs, epochs = 6, 3, 300
        inputs = synthetic_inputs(fleet, hubs)
        config = dict(
            block_size=10, alpha=0.05, eta=0.01, payment_budget=1.0, assignment_tolerance=1.0, exploration_budget=1e9
        )
        cell = dict(seed=11, rho=0.0, fleet=fleet, hubs=hubs, epochs=epochs)
        joint = fresh_state(inputs, fleet, hubs, cell['seed'])
        single = fresh_state(inputs, fleet, hubs, cell['seed'])
        resumed = fresh_state(inputs, fleet, hubs, cell['seed'])
        joint_stats, joint_trace = chunk(config, cell, 0, epochs, joint, inputs, independent_random=True)
        single_stats, single_trace = chunk(config, cell, 0, epochs, single, inputs, only_arm=7, independent_random=True)
        parts = []
        for start, stop in ((0, 100), (100, 300)):
            stats, trace = chunk(config, cell, start, stop, resumed, inputs, only_arm=7, independent_random=True)
            parts.append(stats)
        np.testing.assert_array_equal(joint_stats[:, 7], single_stats[:, 7])
        np.testing.assert_array_equal(joint_trace[7], single_trace[7])
        np.testing.assert_array_equal(np.concatenate(parts), single_stats)
        np.testing.assert_array_equal(trace, single_trace)
        for key in single:
            np.testing.assert_array_equal(joint[key][7], single[key][7], err_msg=key)
            np.testing.assert_array_equal(resumed[key], single[key], err_msg=key)
        ix = {m: j for j, m in enumerate(METRICS)}
        np.testing.assert_array_equal(
            single_stats[:, 7, ix['explored']], config['block_size'] - single_stats[:, 7, ix['proposed_pass']]
        )

    def test_guarantee_table(self):
        for a in range(len(ARMS)):
            self.assertTrue(guaranteed(a, 0.0))
        self.assertTrue(guaranteed(3, 0.1) and guaranteed(4, 0.1))
        self.assertFalse(guaranteed(5, 0.1) or guaranteed(0, 0.1))

    def test_kernel_with_larger_fleet_and_exploration(self):
        fleet, hubs, epochs = 20, 3, 300
        config, cell, inputs, state, stats, trace = run_synthetic(fleet, hubs, epochs, 1e9)
        ix = {m: j for j, m in enumerate(METRICS)}
        self.assertEqual(stats.shape, (epochs // config['block_size'], len(ARMS), len(METRICS)))
        self.assertTrue(np.all(state['counts'].sum(axis=(1, 2)) == fleet * epochs))
        series = stats / config['block_size']
        np.testing.assert_allclose(
            series[..., ix['bound']],
            series[..., ix['assignment_loss']]
            + series[..., ix['selected_upper_gap']]
            + series[..., ix['lower_benchmark_gap']],
            atol=1e-7,
        )
        self.assertTrue(np.all(state['coverage_min'] >= -1e-8))
        self.assertEqual(stats[..., ix['false_certificate']].sum(), 0.0)
        # The ungated arm explores on uncertified epochs; the gated arm is blocked while widths exceed the budget.
        explored = stats[..., ix['explored']].sum(axis=0)
        self.assertGreater(explored[6], 0.0)
        self.assertEqual(explored[0], 0.0)
        self.assertEqual(explored[3], 0.0)
        cumulative = stats.cumsum(axis=0)
        for a in (1, 4, 6):
            self.assertTrue(
                np.all(
                    cumulative[:, a, ix['explored_loss_total']] <= cumulative[:, a, ix['explored_width_total']] + 1e-5
                )
            )
        self.assertEqual(trace.shape, (len(ARMS), 6, fleet, fleet))
        for a in range(len(ARMS)):
            self.assertEqual(sorted(trace[a, 4, 1].astype(int).tolist()), list(range(fleet)))


if __name__ == '__main__':
    unittest.main()
