"""Logging must preserve learning, statistical families, and saved results."""
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import MagicMock, patch
import numpy as np
from certified_dispatch.model import build_roads
from certified_dispatch.rl import GoalRoads, Learner
from certified_dispatch.tracking import RunLogger, DiagnosticTracker
from certified_dispatch.run_rl import experiment

CONFIG = json.loads((Path(__file__).resolve().parents[1] / 'configs/rl_smoke.json').read_text())


class TrackingTests(unittest.TestCase):
    def test_dense_logging_preserves_learning_and_interval_totals(self):
        env = GoalRoads(build_roads(3))
        a = Learner.create(env, 29)
        b = Learner.create(env, 29)
        baseline = a.train_to(100, CONFIG)
        with tempfile.TemporaryDirectory() as directory:
            logger = RunLogger(Path(directory) / 'logs', CONFIG)
            logger.start_seed(0)
            diagnostics = DiagnosticTracker(env, logger)
            diagnostics.record('reference', b)
            windows = []

            def callback(metrics):
                windows.append(metrics)
                logger.emit('train', 'reference', metrics['episodes'], metrics)
                diagnostics.record('reference', b)

            logged = b.train_to(100, CONFIG, on_log=callback, log_every=13)
            logger.close()
            self.assertEqual(sum(w['interval_episodes'] for w in windows), 100)
            self.assertEqual(sum(w['interval_transitions'] for w in windows), b.transitions)
            self.assertTrue(all(w['td_rmse'] >= w['mean_abs_td_error'] - 1e-12 for w in windows))
            events = [json.loads(x) for x in (Path(directory) / 'logs/seed_000.jsonl').read_text().splitlines()]
            self.assertTrue(any(e['kind'] == 'oracle' for e in events))
        self.assertEqual(baseline, logged)
        np.testing.assert_array_equal(a.q, b.q)
        np.testing.assert_array_equal(a.visits, b.visits)
        self.assertEqual(a.cumulative_cost, b.cumulative_cost)

    def test_end_to_end_logging_preserves_outputs(self):
        c = {
            **CONFIG,
            'side': 2,
            'fleet': 2,
            'seeds': 1,
            'batches': 1,
            'checkpoints': [0, 10, 30],
            'evaluation_counts': [8],
        }
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            a = root / 'a'
            b = root / 'b'
            a.mkdir()
            b.mkdir()
            plain = experiment(c, a)
            logger = RunLogger(b / 'telemetry', c)
            logged = experiment(c, b, logger, log_every=5, diagnostic_every=10)
            logger.close()
            self.assertEqual(plain, logged)
            for path in (a / 'checkpoints').rglob('*.npz'):
                with np.load(path) as x, np.load(b / path.relative_to(a)) as y:
                    for key in x.files:
                        np.testing.assert_array_equal(x[key], y[key])

    def test_wandb_axes_seed_lifecycle_and_local_fallback(self):
        runs = [MagicMock(id='first', url='url1'), MagicMock(id='second', url='url2')]
        sdk = SimpleNamespace(init=MagicMock(side_effect=runs), Settings=MagicMock())
        with tempfile.TemporaryDirectory() as directory, patch.dict('sys.modules', {'wandb': sdk}), patch(
            'certified_dispatch.tracking.importlib.metadata.version', return_value='test'
        ):
            path = Path(directory) / 'logs'
            logger = RunLogger(path, CONFIG, mode='offline', group='test')
            logger.start_seed(0)
            logger.emit('train', 'reference', 100, {'td_rmse': 0.5})
            logger.emit('train', 'driver_0', 10, {'td_rmse': 0.7})
            with self.assertRaises(ValueError):
                logger.emit('train', 'reference', 99, {'td_rmse': 0.4})
            runs[0].define_metric.assert_any_call('train/driver_0/*', step_metric='train/driver_0/episodes')
            runs[0].log.side_effect = RuntimeError('SDK outage')
            with self.assertWarns(RuntimeWarning):
                logger.emit('train', 'reference', 101, {'td_rmse': 0.4})
            logger.emit('train', 'reference', 102, {'td_rmse': 0.3})
            logger.finish_seed()
            logger.start_seed(1)
            logger.emit('train', 'reference', 1, {'td_rmse': 0.8})
            logger.close()
            self.assertEqual(sdk.init.call_count, 2)
            self.assertEqual(sdk.init.call_args.kwargs['mode'], 'offline')
            runs[0].finish.assert_called_once_with(exit_code=1)
            runs[1].finish.assert_called_once_with(exit_code=0)
            self.assertEqual(len((path / 'seed_000.jsonl').read_text().splitlines()), 4)
            self.assertEqual(
                json.loads((path / 'tracking.json').read_text())['errors'][0]['error_type'], 'RuntimeError'
            )


if __name__ == '__main__':
    unittest.main()
