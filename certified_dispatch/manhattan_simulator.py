"""Hidden fixed routing policies and oracle costs on the Manhattan road graph.

For a set of service hubs, the simulator builds the public support of every
origin-pickup-dropoff context from the road bounds and the class of routes that
strictly approach the goal, the common free-flow routing policy, one private
policy per driver (free-flow weights perturbed by independent lognormal factors),
and the hidden expected service costs of the mixture policy of every driver.
Dynamic programs run over all road nodes for each hub goal; sampled routes are
reconstructed on demand. Only `run_manhattan_scale.prepare_inputs_scale` reads
these objects; the platform code never does.
"""
from functools import lru_cache
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components, dijkstra

from .model import random_stream
from .networks import load_network, network_path


class Paths:
    """Edge paths of one fixed policy toward the hub goals, built lazily from the saved action choices."""

    def __init__(self, roads, goals, selected_edges):
        self.roads = roads
        self.columns = {int(g): j for j, g in enumerate(goals)}
        self.selected_edges = selected_edges
        self.cached = lru_cache(maxsize=4096)(self._path)

    def _path(self, source, goal):
        at, edges = source, []
        column = self.columns[goal]
        while at != goal:
            e = int(self.selected_edges[at, column])
            if e < 0 or len(edges) >= self.roads.nodes:
                raise AssertionError('Invalid proper policy path')
            edges.append(e)
            at = int(self.roads.edges[e, 1])
        return np.asarray(edges, dtype=int)

    def __getitem__(self, pair):
        return self.cached(int(pair[0]), int(pair[1]))


class Simulator:
    """Hubs, public supports, fixed policies and exact expected costs for one seed."""

    def __init__(self, config, seed):
        roads = self.roads = load_network(network_path(config))
        n = roads.nodes
        u, v = roads.edges.T
        graph = csr_matrix((roads.free, (u, v)), shape=(n, n))
        if connected_components(graph, directed=True, connection='strong')[0] != 1:
            raise ValueError('The road network must be strongly connected')
        count = config['hubs']
        if count == 'all':
            hubs = np.arange(n)
        else:
            count = int(count)
            if not 2 <= count <= n:
                raise ValueError('Need between 2 and V exact service hubs')
            # Farthest-point selection in symmetric free-flow distance, starting from node 0.
            selected, nearest = [0], np.full(n, np.inf)
            while len(selected) < count:
                last = selected[-1]
                distance = (dijkstra(graph, indices=last) + dijkstra(graph.T.tocsr(), indices=last)) / 2
                nearest = np.minimum(nearest, distance)
                nearest[selected] = -np.inf
                selected.append(int(nearest.argmax()))
            hubs = np.asarray(selected)
        self.hubs = hubs
        self.rho, self.fleet = config['rho'], config['fleet']
        distance = dijkstra(graph.T.tocsr(), indices=hubs).T
        outgoing = [np.flatnonzero(u == s) for s in range(n)]
        # The common policy follows free-flow times; private policies perturb them per driver.
        policies = [roads.free]
        self.policy_weights = [
            roads.free * random_stream(seed, 501, driver).lognormal(0.0, 0.65, len(u)) for driver in range(self.fleet)
        ]
        if self.rho:
            policies.extend(self.policy_weights)
        choices = [np.full((n, len(hubs)), -1, dtype=np.int32) for _ in policies]
        means = [np.full((n, n), np.nan) for _ in policies]
        self.lower = np.full((n, n), np.nan)
        self.upper = np.full((n, n), np.nan)
        self.restricted_optimal = np.full((n, n), np.nan)
        for j, g in enumerate(hubs):
            values = np.zeros((len(policies), n))
            expected = np.zeros_like(values)
            upper = np.zeros(n)
            restricted = np.zeros(n)
            for s in np.argsort(distance[:, j]):
                if s == g:
                    continue
                # Admissible actions strictly reduce the free-flow distance to the goal.
                edges = outgoing[s][distance[v[outgoing[s]], j] < distance[s, j]]
                if not len(edges):
                    raise ValueError('A nonterminal node has no progress action')
                to = v[edges]
                upper[s] = (roads.free[edges] + roads.delay[edges] + upper[to]).max()
                restricted[s] = (roads.mean[edges] + restricted[to]).min()
                for p, weights in enumerate(policies):
                    scores = weights[edges] + values[p, to]
                    a = int(scores.argmin())
                    e = edges[a]
                    choices[p][s, j] = e
                    values[p, s] = scores[a]
                    expected[p, s] = roads.mean[e] + expected[p, v[e]]
            self.lower[:, g] = values[0]
            self.upper[:, g] = upper
            self.restricted_optimal[:, g] = restricted
            for p in range(len(policies)):
                means[p][:, g] = expected[p]
        self.common = Paths(roads, hubs, choices[0])
        if self.rho:
            self.private = [Paths(roads, hubs, c) for c in choices[1:]]
            self.means = np.array([(1 - self.rho) * means[0] + self.rho * m for m in means[1:]])
        else:
            self.private = [self.common] * self.fleet
            self.means = np.broadcast_to(means[0], (self.fleet, n, n))
        mean_graph = csr_matrix((roads.mean, (u, v)), shape=(n, n))
        self.optimal = np.full((n, n), np.nan)
        self.optimal[:, hubs] = dijkstra(mean_graph.T.tocsr(), indices=hubs).T
        self.context_count = len(hubs) ** 2 * (len(hubs) - 1)

    def true_context(self, context):
        """Exact expected service cost of every driver at an (origin, pickup, dropoff) context."""
        s, p, d = context
        return self.means[:, s, p] + self.means[:, p, d]
