"""Directionality, proper routes, unrestricted benchmarks, and integration."""
import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
from certified_dispatch.networks import DirectedRoads, DirectedGoalRoads, load_network, network_diagnostics
from certified_dispatch.rl import Learner

ROOT = Path(__file__).resolve().parents[1]


class NetworkTests(unittest.TestCase):
    def test_real_topology_and_proper_paths(self):
        roads = load_network(ROOT / 'data/manhattan_small.json')
        env = DirectedGoalRoads(roads)
        report = network_diagnostics(env)
        self.assertEqual((report['nodes'], report['directed_edges']), (72, 123))
        self.assertEqual(report['edges_without_reverse'], 99)
        self.assertGreater(report['branching_state_goal_fraction'], 0.1)
        paths = Learner.create(env, 21).frozen_paths()
        shortest, _ = roads.paths(roads.free)
        np.testing.assert_allclose(shortest, env.route_values(roads.free))
        lower = env.route_values(roads.free)
        upper = env.route_values(roads.free + roads.delay, True)
        for (s, g), path in paths.items():
            at = s
            seen = {s}
            for e in path:
                u, v = roads.edges[e]
                self.assertEqual(u, at)
                self.assertLess(env.distance[v, g], env.distance[u, g])
                self.assertNotIn(v, seen)
                seen.add(v)
                at = v
            self.assertEqual(at, g)
            self.assertLess(len(path), roads.nodes)
            self.assertLessEqual(lower[s, g], roads.free[path].sum() + 1e-9)
            self.assertGreaterEqual(upper[s, g] + 1e-9, (roads.free + roads.delay)[path].sum())

    def test_action_restriction_can_exclude_expected_time_optimum(self):
        roads = DirectedRoads(
            3,
            np.array([[0, 1], [0, 2], [2, 1], [1, 0]]),
            np.array([1.0, 2.0, 1.0, 1.0]),
            np.array([100.0, 0.0, 0.0, 0.0]),
            np.zeros((3, 2)),
            {},
        )
        env = DirectedGoalRoads(roads)
        unrestricted, _ = roads.paths(roads.mean)
        self.assertEqual(unrestricted[0, 1], 3)
        self.assertEqual(env.route_values(roads.mean)[0, 1], 51)
        self.assertGreater(network_diagnostics(env)['restriction_gap_max_minutes'], 0)

        # Exhaustively verify DP extrema on the entire declared action class.
        def costs(s, g, weights):
            if s == g:
                return [0.0]
            return [
                weights[env.edge[s, a]] + c for a in env.actions[g][s] for c in costs(int(env.next[s, a]), g, weights)
            ]

        for weights, maximize in ((roads.mean, False), (roads.free + roads.delay, True)):
            result = env.route_values(weights, maximize)
            for s in range(3):
                for g in range(3):
                    values = costs(s, g, weights)
                    self.assertAlmostEqual(result[s, g], max(values) if maximize else min(values))

    def test_disconnected_and_invalid_data_rejected(self):
        roads = DirectedRoads(3, np.array([[0, 1], [1, 0]]), np.ones(2), np.ones(2), np.zeros((3, 2)), {})
        with self.assertRaisesRegex(ValueError, 'strongly connected'):
            DirectedGoalRoads(roads)
        data = json.loads((ROOT / 'data/manhattan_small.json').read_text())
        data['edges'].append(data['edges'][0])
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'invalid.json'
            path.write_text(json.dumps(data))
            with self.assertRaisesRegex(ValueError, 'duplicates'):
                load_network(path)


if __name__ == '__main__':
    unittest.main()
