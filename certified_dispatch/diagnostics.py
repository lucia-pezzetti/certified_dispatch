"""Read-only exact simulator diagnostics for the declared routing class."""
import numpy as np


def check(condition, description):
    if not condition:
        raise AssertionError(description)


def policy_metrics(env, q, tie, optimal, mask):
    # Match Learner.greedy exactly, including its fixed tie order.
    masked = np.where(mask, q, np.inf)
    minimum = masked.min(axis=2)
    chosen = np.argmin(np.where(mask & (q == minimum[:, :, None]), tie, np.inf), axis=2)
    value = np.where(mask.any(axis=2), minimum, 0.0)  # [goal,state]
    execution = np.zeros_like(value)
    for g in range(env.nodes):
        for s in np.argsort(env.distance[:, g]):
            if s != g:
                a = chosen[g, s]
                execution[g, s] = env.roads.mean[env.edge[s, a]] + execution[g, env.next[s, a]]
    goals, states, acts = np.where(mask)
    nxt = env.next[states, acts]
    edge = env.edge[states, acts]
    residual = np.abs(q[goals, states, acts] - env.roads.mean[edge] - value[goals, nxt])
    qerror = np.abs(q[goals, states, acts] - env.roads.mean[edge] - optimal[nxt, goals])
    nonterminal = ~np.eye(env.nodes, dtype=bool)
    calibration = np.abs(value - execution)[nonterminal]
    gap = (execution - optimal.T)[nonterminal]
    check(np.min(gap) > -1e-8, 'Greedy policy appears cheaper than optimal DP')
    return chosen, dict(
        route_gap_mean=float(gap.mean()),
        route_gap_max=float(gap.max()),
        prediction_error_mean=float(calibration.mean()),
        prediction_error_max=float(calibration.max()),
        q_star_error_mean=float(qerror.mean()),
        q_star_error_max=float(qerror.max()),
        bellman_residual_mean=float(residual.mean()),
        bellman_residual_max=float(residual.max()),
    )
