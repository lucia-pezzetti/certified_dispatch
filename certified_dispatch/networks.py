"""Directed road data and a public, proper routing action class.

The free-flow distance to the goal must strictly decrease at each move. This
rule guarantees completion, but may exclude a minimum expected-time route.
The experiment therefore also evaluates unrestricted shortest-path costs.
"""
from dataclasses import dataclass
import json
from pathlib import Path
import numpy as np
from scipy.sparse.csgraph import shortest_path
from .common import ROOT
from .model import Roads, build_roads
from .rl import GoalRoads


@dataclass
class DirectedRoads:
    nodes: int
    edges: np.ndarray
    free: np.ndarray
    delay: np.ndarray
    coordinates: np.ndarray
    metadata: dict

    mean = Roads.mean
    paths = Roads.paths


class DirectedGoalRoads(GoalRoads):
    def __init__(self, roads):
        self.roads, self.nodes = roads, roads.nodes
        degree = np.bincount(roads.edges[:, 0], minlength=self.nodes)
        self.next = np.full((self.nodes, int(degree.max())), -1, dtype=int)
        self.edge = np.full_like(self.next, -1)
        counts = np.zeros(self.nodes, dtype=int)
        for e, (u, v) in enumerate(roads.edges):
            a = counts[u]
            self.next[u, a], self.edge[u, a] = v, e
            counts[u] += 1
        graph = np.full((self.nodes, self.nodes), np.inf)
        np.fill_diagonal(graph, 0)
        graph[roads.edges[:, 0], roads.edges[:, 1]] = roads.free
        self.distance = shortest_path(graph, directed=True)
        if not np.isfinite(self.distance).all():
            raise ValueError('The road network must be strongly connected')
        self.actions = [[None for _ in range(self.nodes)] for _ in range(self.nodes)]
        for g in range(self.nodes):
            for s in range(self.nodes):
                valid = [a for a in range(int(degree[s])) if self.distance[self.next[s, a], g] < self.distance[s, g]]
                if s != g and not valid:
                    raise ValueError('A nonterminal node has no progress action')
                self.actions[g][s] = np.asarray(valid, dtype=int)


def network_path(config):
    path = Path(config['network_file'])
    return path if path.is_absolute() else ROOT / path


def load_network(path):
    data = json.loads(Path(path).read_text())
    nodes, edges = data['nodes'], data['edges']
    if [n['id'] for n in nodes] != list(range(len(nodes))) or len(nodes) < 2:
        raise ValueError('Node ids must be contiguous and ordered from zero')
    pairs = np.asarray([(e['u'], e['v']) for e in edges], dtype=int)
    free = np.asarray([e['free_flow_minutes'] for e in edges])
    delay = np.asarray([e['delay_width_minutes'] for e in edges])
    if (
        not len(edges)
        or np.any(pairs < 0)
        or np.any(pairs >= len(nodes))
        or np.any(pairs[:, 0] == pairs[:, 1])
        or len(set(map(tuple, pairs))) != len(edges)
    ):
        raise ValueError('Need valid directed edges without duplicates or self loops')
    if not np.isfinite(free).all() or not np.isfinite(delay).all() or np.any(free <= 0) or np.any(delay < 0):
        raise ValueError('Need positive finite free times and nonnegative finite delay widths')
    coordinates = np.asarray([(n['x'], n['y']) for n in nodes])
    return DirectedRoads(
        len(nodes), pairs, free, delay, coordinates, {k: v for k, v in data.items() if k not in ('nodes', 'edges')}
    )


def environment(config):
    if config.get('network_file'):
        return DirectedGoalRoads(load_network(network_path(config)))
    return GoalRoads(build_roads(config['side'], config['graph_seed']))


def network_diagnostics(env):
    restricted = env.route_values(env.roads.mean)
    unrestricted, _ = env.roads.paths(env.roads.mean)
    gap = restricted - unrestricted
    nonterminal = ~np.eye(env.nodes, dtype=bool)
    branching = np.array([[len(env.actions[g][s]) for g in range(env.nodes)] for s in range(env.nodes)])
    edge_set = set(map(tuple, env.roads.edges))
    return dict(
        nodes=env.nodes,
        directed_edges=len(edge_set),
        edges_without_reverse=sum((v, u) not in edge_set for u, v in edge_set),
        nonterminal_state_goal_pairs=env.nodes * (env.nodes - 1),
        branching_state_goal_fraction=float(np.mean(branching[nonterminal] > 1)),
        restriction_gap_mean_minutes=float(gap[nonterminal].mean()),
        restriction_gap_max_minutes=float(gap.max()),
        restriction_gap_positive_fraction=float(np.mean(gap[nonterminal] > 1e-9)),
    )
