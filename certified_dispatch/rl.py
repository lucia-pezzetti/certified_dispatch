"""Goal-conditioned tabular Q-learning for bounded, undiscounted service costs.

This controlled grid environment permits moves that reduce the grid distance to
the goal by one. Every route completes, including at checkpoint zero. The restriction is
part of the model, not an unreported timeout or an oracle routing repair.
"""
from dataclasses import dataclass
import time
import numpy as np
from .model import Roads, DriverTrips, random_stream


class GoalRoads:
    def __init__(self, roads: Roads):
        self.roads = roads
        self.nodes = roads.nodes
        self.next = np.full((self.nodes, 4), -1, dtype=int)
        self.edge = np.full((self.nodes, 4), -1, dtype=int)
        # Up, down, left, right; reverse directions are distinct roads.
        directions = {(-1, 0): 0, (1, 0): 1, (0, -1): 2, (0, 1): 3}
        for e, (u, v) in enumerate(roads.edges):
            ur, uc = divmod(int(u), roads.side)
            vr, vc = divmod(int(v), roads.side)
            a = directions[vr - ur, vc - uc]
            self.next[u, a], self.edge[u, a] = v, e
        xy = np.array([divmod(s, roads.side) for s in range(self.nodes)])
        self.distance = np.abs(xy[:, None, :] - xy[None, :, :]).sum(axis=2)
        self.actions = [[None for _ in range(self.nodes)] for _ in range(self.nodes)]
        for g in range(self.nodes):
            for s in range(self.nodes):
                valid = [
                    a
                    for a in range(4)
                    if self.next[s, a] >= 0 and self.distance[self.next[s, a], g] < self.distance[s, g]
                ]
                self.actions[g][s] = np.asarray(valid, dtype=int)

    def route_values(self, weights, maximize=False):
        """Exact route DP for the declared action class, independent of policies."""
        result = np.zeros((self.nodes, self.nodes))  # [origin, goal]
        for g in range(self.nodes):
            for s in np.argsort(self.distance[:, g]):
                acts = self.actions[g][s]
                if len(acts):
                    choices = weights[self.edge[s, acts]] + result[self.next[s, acts], g]
                    result[s, g] = choices.max() if maximize else choices.min()
        return result


@dataclass
class Learner:
    env: GoalRoads
    q: np.ndarray
    visits: np.ndarray
    tie_order: np.ndarray
    seed: int
    episodes: int = 0
    transitions: int = 0
    cumulative_cost: float = 0.0

    @classmethod
    def create(cls, env, seed):
        q = np.zeros((env.nodes, env.nodes, env.next.shape[1]))  # [goal, state, action]
        tie = random_stream(seed, 60).random(q.shape)
        return cls(env, q, np.zeros(q.shape, dtype=np.int64), tie, seed)

    def value(self, state, goal):
        acts = self.env.actions[goal][state]
        return float(self.q[goal, state, acts].min()) if len(acts) else 0.0

    def greedy(self, state, goal):
        acts = self.env.actions[goal][state]
        if not len(acts):
            raise ValueError('Terminal states have no routing action')
        values = self.q[goal, state, acts]
        tied = acts[values == values.min()]
        return int(tied[np.argmin(self.tie_order[goal, state, tied])])

    def prediction(self):
        return np.array([[self.value(s, g) for g in range(self.env.nodes)] for s in range(self.env.nodes)])

    def frozen_paths(self):
        paths = {}
        for s in range(self.env.nodes):
            for g in range(self.env.nodes):
                current, path = s, []
                while current != g:
                    a = self.greedy(current, g)
                    path.append(self.env.edge[current, a])
                    current = int(self.env.next[current, a])
                paths[s, g] = np.array(path, dtype=int)
        return paths

    def train_to(self, target, config, on_log=None, log_every=5000):
        """Off-policy Q-learning from epsilon-greedy single-agent trajectories.

        Separate per-episode streams make saved checkpoints independent of how
        many intermediate evaluations are requested. No evaluation draw is reused.
        """
        if target < self.episodes:
            raise ValueError('Training cannot go backwards')
        if on_log is not None and log_every < 1:
            raise ValueError('Logging interval must be positive')
        before_steps, before_cost = self.transitions, self.cumulative_cost
        before_episodes = self.episodes
        window_episode, window_steps, window_cost = self.episodes, self.transitions, self.cumulative_cost
        td_abs = td_squared = update_abs = 0.0
        started = time.perf_counter()
        while self.episodes < target:
            rng = random_stream(self.seed, 70, self.episodes)
            s = int(rng.integers(self.env.nodes))
            g = int((s + rng.integers(1, self.env.nodes)) % self.env.nodes)
            fraction = min(1.0, self.episodes / config['epsilon_decay_episodes'])
            eps = config['epsilon_start'] + (config['epsilon_end'] - config['epsilon_start']) * fraction
            while s != g:
                acts = self.env.actions[g][s]
                a = int(rng.choice(acts)) if rng.random() < eps else self.greedy(s, g)
                edge = self.env.edge[s, a]
                cost = float(self.env.roads.free[edge] + rng.random() * self.env.roads.delay[edge])
                nxt = int(self.env.next[s, a])
                self.visits[g, s, a] += 1
                rate = float(self.visits[g, s, a] ** (-config['learning_rate_power']))
                # gamma=1, terminal value=0: the same total-cost objective as the paper.
                target_value = cost + self.value(nxt, g)
                error = target_value - self.q[g, s, a]
                update = rate * error
                self.q[g, s, a] += update
                if on_log is not None:
                    td_abs += abs(float(error))
                    td_squared += float(error) ** 2
                    update_abs += abs(float(update))
                self.transitions += 1
                self.cumulative_cost += cost
                s = nxt
            self.episodes += 1
            if on_log is not None and (self.episodes % log_every == 0 or self.episodes == target):
                steps = self.transitions - window_steps
                episodes = self.episodes - window_episode
                elapsed = max(time.perf_counter() - started, 1e-12)
                on_log(
                    dict(
                        episodes=self.episodes,
                        transitions=self.transitions,
                        interval_start_episode=window_episode,
                        interval_episodes=episodes,
                        interval_transitions=steps,
                        mean_cost_per_trip=(self.cumulative_cost - window_cost) / episodes,
                        mean_cost_per_transition=(self.cumulative_cost - window_cost) / steps,
                        mean_trip_steps=steps / episodes,
                        epsilon=eps,
                        mean_abs_td_error=td_abs / steps,
                        td_rmse=float(np.sqrt(td_squared / steps)),
                        mean_abs_q_update=update_abs / steps,
                        transitions_per_second=steps / elapsed,
                    )
                )
                window_episode, window_steps, window_cost = self.episodes, self.transitions, self.cumulative_cost
                td_abs = td_squared = update_abs = 0.0
                started = time.perf_counter()
        return dict(
            episodes=self.episodes,
            transitions=self.transitions,
            interval_episodes=self.episodes - before_episodes,
            interval_transitions=self.transitions - before_steps,
            interval_mean_training_cost=(self.cumulative_cost - before_cost) / max(1, self.episodes - before_episodes),
        )


def locations(env, fleet, seed, batch):
    rng = random_stream(seed, 80, batch)
    origins = rng.integers(env.nodes, size=fleet)
    pickups = rng.integers(env.nodes, size=fleet)
    dropoffs = (pickups + rng.integers(1, env.nodes, size=fleet)) % env.nodes
    return origins, pickups, dropoffs


def pair_costs(values, origins, pickups, dropoffs):
    return values[origins[:, None], pickups[None, :]] + values[pickups, dropoffs][None, :]


def frozen_trips(roads, paths_by_driver, origins, pickups, dropoffs):
    routes = [
        [np.concatenate((paths[origins[i], p], paths[p, d])) for p, d in zip(pickups, dropoffs)]
        for i, paths in enumerate(paths_by_driver)
    ]
    # DriverTrips exposes black-box sampling to the evaluation procedure.
    return DriverTrips(roads, routes, routes, np.ones((len(origins), len(pickups))))


def save_checkpoint(path, reference, drivers):
    values = {
        'reference_q': reference.q,
        'reference_visits': reference.visits,
        'reference_tie_order': reference.tie_order,
        'episodes': np.array(reference.episodes),
        'reference_transitions': np.array(reference.transitions),
        'reference_seed': np.array(reference.seed),
    }
    for i, learner in enumerate(drivers):
        values.update(
            {
                f'driver_{i}_q': learner.q,
                f'driver_{i}_visits': learner.visits,
                f'driver_{i}_tie_order': learner.tie_order,
                f'driver_{i}_transitions': np.array(learner.transitions),
                f'driver_{i}_seed': np.array(learner.seed),
            }
        )
    np.savez_compressed(path, **values)
