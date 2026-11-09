"""Simulator only: hidden driver policies, sampling, and evaluation oracles.

Each road traversal is independent Uniform(free, free + delay), in minutes.
A driver chooses one of two fixed simple paths per leg at the start of a trip.
No simulator object is passed to the platform module.
"""
from dataclasses import dataclass
import numpy as np
from scipy.sparse.csgraph import shortest_path


def random_stream(seed, *keys):
    return np.random.default_rng(np.random.SeedSequence([seed, *keys]))


@dataclass
class Roads:
    side: int
    edges: np.ndarray
    free: np.ndarray
    delay: np.ndarray

    @property
    def nodes(self):
        return self.side**2

    @property
    def mean(self):
        return self.free + self.delay / 2

    def paths(self, weights):
        graph = np.full((self.nodes, self.nodes), np.inf)
        np.fill_diagonal(graph, 0)
        graph[self.edges[:, 0], self.edges[:, 1]] = weights
        distances, previous = shortest_path(graph, directed=True, return_predecessors=True)
        index = {tuple(edge): j for j, edge in enumerate(self.edges)}
        paths = {}
        for source in range(self.nodes):
            for target in range(self.nodes):
                path, at = [], target
                while at != source:
                    parent = int(previous[source, at])
                    if parent < 0:
                        raise ValueError('Road graph is disconnected')
                    path.append(index[parent, at])
                    at = parent
                paths[source, target] = np.asarray(path[::-1], dtype=int)
        return distances, paths


def build_roads(side=4, seed=913):
    edges = []
    for row in range(side):
        for col in range(side):
            u = row * side + col
            for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                r, c = row + dr, col + dc
                if 0 <= r < side and 0 <= c < side:
                    edges.append((u, r * side + c))
    rng = random_stream(seed)
    count = len(edges)
    return Roads(side, np.asarray(edges), rng.uniform(0.15, 0.4, count), rng.uniform(0.1, 0.65, count))


@dataclass
class DriverTrips:
    roads: Roads
    paths_a: list
    paths_b: list
    probability_a: np.ndarray

    @property
    def fleet(self):
        return len(self.paths_a)

    def expected_costs(self):
        a = np.array([[self.roads.mean[p].sum() for p in row] for row in self.paths_a])
        b = np.array([[self.roads.mean[p].sum() for p in row] for row in self.paths_b])
        return self.probability_a * a + (1 - self.probability_a) * b

    def sample(self, driver, job, count, rng):
        width = max(len(self.paths_a[driver][job]), len(self.paths_b[driver][job]))
        draws = rng.random((count, width + 1))
        use_a = draws[:, 0] < self.probability_a[driver, job]
        values = np.zeros(count)
        for mask, path in ((use_a, self.paths_a[driver][job]), (~use_a, self.paths_b[driver][job])):
            m = int(mask.sum())
            if m and len(path):
                # Repeated edges use fresh traversal noise, not a cached edge draw.
                noise = draws[mask, 1 : len(path) + 1]
                values[mask] = (self.roads.free[path] + noise * self.roads.delay[path]).sum(axis=1)
        return values

    def adverse_policy_change(self):
        """Choose the higher-mean of the same two admissible routes for each pair."""
        a = np.array([[self.roads.mean[p].sum() for p in row] for row in self.paths_a])
        b = np.array([[self.roads.mean[p].sum() for p in row] for row in self.paths_b])
        return DriverTrips(self.roads, self.paths_a, self.paths_b, (a >= b).astype(float))


def make_batch(roads, fleet, seed, batch, scenario):
    rng = random_stream(seed, 10, batch)
    origins = rng.integers(roads.nodes, size=fleet)
    pickups = rng.integers(roads.nodes, size=fleet)
    # Each delivery has a nonzero service leg.
    dropoffs = (pickups + rng.integers(1, roads.nodes, size=fleet)) % roads.nodes
    true_dist, true_paths = roads.paths(roads.mean)
    free_dist, shared_paths = roads.paths(roads.free)
    pair_cost = lambda d: d[origins[:, None], pickups[None, :]] + d[pickups, dropoffs][None, :]
    optimal = pair_cost(true_dist)
    lower = pair_cost(free_dist)
    # Public operating assumption: each leg follows a simple path. This cap
    # uses only the graph and traversal support, never a hidden route's length.
    leg_cap = (roads.nodes - 1) * float(np.max(roads.free + roads.delay))
    upper_support = leg_cap * (1 + (origins[:, None] != pickups[None, :]).astype(int))
    paths_a, paths_b, probs = [], [], []
    for i in range(fleet):
        personal_rng = random_stream(seed, 20, batch, i)
        if scenario == 'optimal':
            first = second = true_paths
            q = 1.0
        elif scenario == 'shared':
            first = second = shared_paths
            q = 1.0
        elif scenario == 'heterogeneous':
            _, first = roads.paths(roads.free * personal_rng.lognormal(0, 0.65, len(roads.edges)))
            _, second = roads.paths(roads.free * personal_rng.lognormal(0, 0.65, len(roads.edges)))
            q = personal_rng.uniform(0.25, 0.75)
        else:
            raise ValueError('Unknown scenario: ' + scenario)
        join = lambda paths, k: np.concatenate((paths[origins[i], pickups[k]], paths[pickups[k], dropoffs[k]]))
        paths_a.append([join(first, k) for k in range(fleet)])
        paths_b.append([join(second, k) for k in range(fleet)])
        probs.append(np.full(fleet, q))
    trips = DriverTrips(roads, paths_a, paths_b, np.array(probs))
    return trips, optimal, lower, upper_support
