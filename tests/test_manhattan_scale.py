"""Tests of the pooled Manhattan loop: horizons, input preparation for any fleet size and the kernel."""
from pathlib import Path
import tempfile
import unittest
import numpy as np
from certified_dispatch.manhattan_kernel import ARMS, METRICS
from certified_dispatch.run_manhattan_scale import epochs_for, checkpoint_for, prepare_inputs_scale, fresh_state, chunk

ROOT = Path(__file__).resolve().parents[1]
SMALL = ROOT / 'data/manhattan_small.json'


def synthetic_inputs(fleet, hubs, rho=0.0, seed=0):
    """Two-leg contexts with three synthetic edges per leg; private routes slightly slower."""
    qs = hubs**3
    legs = np.zeros(qs)
    for s in range(hubs):
        for p in range(hubs):
            for d in range(hubs):
                legs[(s * hubs + p) * hubs + d] = int(s != p) + int(p != d)
    free = np.array([3.0, 3.0, 3.0, 4.0, 4.0, 4.0])
    delay = np.array([10.0, 10.0, 10.0, 10.0, 10.0, 10.0])
    paths = np.zeros((fleet + 1, hubs, hubs, 3), np.int64)
    paths[1:] = np.array([3, 4, 5])
    lengths = np.full((fleet + 1, hubs, hubs), 3, np.int64)
    for s in range(hubs):
        lengths[:, s, s] = 0
    common, private = 3 * (3 + 5), 3 * (4 + 5)
    truth = np.array([((1 - rho) * common + rho * private) * legs for _ in range(fleet)])
    lo, hi = 9.0 * legs, 42.0 * legs
    return dict(support_lo=lo, support_hi=hi, true_contexts=truth, paths=paths, lengths=lengths, free=free, delay=delay)


def run_synthetic(fleet, hubs, epochs, rho=0.0, alpha=1e-4, seed=11, block=10):
    inputs = synthetic_inputs(fleet, hubs, rho)
    config = dict(block_size=block, alpha=alpha, eta=0.01, payment_budget=1.0, assignment_tolerance=1.0)
    cell = dict(seed=seed, rho=rho, fleet=fleet, hubs=hubs, epochs=epochs)
    state = fresh_state(inputs, fleet, hubs, seed)
    stats, trace = chunk(config, cell, 0, epochs, state, inputs)
    return config, cell, inputs, state, stats, trace


class ScaleTests(unittest.TestCase):
    def test_horizons(self):
        self.assertEqual(epochs_for(60_000_000, 6, 10_000), 10_000_000)
        self.assertEqual(epochs_for(60_000_000, 48, 10_000), 1_250_000)
        for fleet in (6, 12, 24, 48):
            epochs = epochs_for(60_000_000, fleet, 10_000)
            every = checkpoint_for(epochs, 50, 10_000)
            self.assertEqual(epochs % 10_000, 0)
            self.assertEqual(every % 10_000, 0)
            self.assertGreaterEqual(every, 10_000)
            self.assertLessEqual(abs(epochs / every - 50), 15)

    def test_prepare_larger_fleet(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'c.npz'
            prepare_inputs_scale(SMALL, 3, 0.1, 9, 5, path)
            data = dict(np.load(path))
            self.assertEqual(data['true_contexts'].shape, (9, 27))
            self.assertEqual(data['paths'].shape[:3], (10, 3, 3))
            self.assertEqual(data['lengths'].shape, (10, 3, 3))
            spread = data['true_contexts'].max(axis=0) - data['true_contexts'].min(axis=0)
            self.assertTrue(np.all(spread <= 0.1 * (data['support_hi'] - data['support_lo']) + 1e-8))
            self.assertGreater(spread.max(), 0.0)
            self.assertEqual(int(data['fleet']), 9)

    def test_kernel_with_larger_fleet(self):
        fleet, hubs, epochs = 10, 3, 300
        config, cell, inputs, state, stats, trace = run_synthetic(fleet, hubs, epochs)
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
        # Every arm is guaranteed at rho = 0; with alpha = 1e-4 a coverage failure is a defect.
        self.assertTrue(np.all(state['coverage_min'] >= -1e-8))
        self.assertEqual(stats[..., ix['false_certificate']].sum(), 0.0)
        self.assertEqual(trace.shape, (len(ARMS), 6, fleet, fleet))
        for a in range(len(ARMS)):
            self.assertEqual(sorted(trace[a, 4, 0].astype(int).tolist()), list(range(fleet)))
        # Private histories are per driver; pooled ones share one row.
        self.assertTrue(np.all(state['counts'][2].sum(axis=1) == epochs))
        self.assertTrue(np.all(state['counts'][0, 1:] == 0))


if __name__ == '__main__':
    unittest.main()
