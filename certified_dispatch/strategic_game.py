"""Myopic job requests with explicit contention and ex ante expected utilities.

Drivers know their own fixed-policy expected costs, exactly or up to a perception
error of scale tau (perceived cost = true cost + tau * standard normal, independent
over jobs). Requests are either one simultaneous best reply to the posted menu, each
driver assuming the others comply, or public tentative requests revised in random
order; prices stay fixed and no service is sampled until allocation. This models a
stage game, not forward-looking strategic learning. Regrets are always evaluated at
the true expected costs.

Allocation rules. 'priority': the recommended driver wins its own job, otherwise the
lowest index. 'uniform': uniform at random among requesters. 'cheapest': the
requester with the lowest platform estimate wins, ties to the recommended driver and
then to the lowest index; this rule needs the platform's estimate matrix.
"""
from dataclasses import dataclass
import numpy as np
from scipy.special import erf

TOL = 1e-9
RULES = ('priority', 'uniform')  # rules that need no platform information
ESTIMATE_RULES = ('cheapest',)  # rules that use the platform's estimate matrix
ALL_RULES = RULES + ESTIMATE_RULES
BEHAVIORS = ('imposed', 'one_step', 'best_response')


def validate_game(cost, sigma, payments):
    cost = np.asarray(cost, dtype=float)
    sigma = np.asarray(sigma)
    payments = np.asarray(payments, dtype=float)
    n = len(cost)
    if (
        cost.shape != (n, n)
        or n < 2
        or sigma.shape != (n,)
        or not np.issubdtype(sigma.dtype, np.integer)
        or sorted(sigma.tolist()) != list(range(n))
        or payments.shape != (n,)
        or not np.isfinite(cost).all()
        or not np.isfinite(payments).all()
    ):
        raise ValueError('Need finite square costs, a permutation, and one price per job')
    return cost, sigma, payments


def validate_requests(requests, n, rule):
    requests = np.asarray(requests)
    if (
        rule not in ALL_RULES
        or requests.shape != (n,)
        or not np.issubdtype(requests.dtype, np.integer)
        or np.any(requests < -1)
        or np.any(requests >= n)
    ):
        raise ValueError('Invalid allocation rule or requests; -1 denotes idling')
    return requests


def validate_estimate(rule, estimate, n):
    if rule not in ESTIMATE_RULES:
        return None
    if estimate is None:
        raise ValueError(f'Allocation rule {rule} needs the platform estimate matrix')
    estimate = np.asarray(estimate, dtype=float)
    if estimate.shape != (n, n) or not np.isfinite(estimate).all():
        raise ValueError('Estimate must be a finite square matrix')
    return estimate


def _cheapest_winner(candidates, job, owner, estimate):
    # Lowest platform estimate wins; ties go to the recommended driver, then to the lowest index.
    return int(min(candidates, key=lambda d: (estimate[d, job], d != owner[job], d)))


def _probability(driver, job, requests, owner, rule, estimate=None):
    # Remove the driver's old request before evaluating its possible new request.
    others = np.flatnonzero(requests == job)
    others = others[others != driver]
    if rule == 'uniform':
        return 1.0 / (len(others) + 1)
    if rule == 'cheapest':
        return float(_cheapest_winner(np.append(others, driver), job, owner, estimate) == driver)
    if driver == owner[job]:
        return 1.0
    if owner[job] in others:
        return 0.0
    # The paper's fixed fallback rule is increasing driver index.
    return float(len(others) == 0 or driver < others.min())


def _values(driver, own_cost, payments, requests, owner, rule, estimate=None):
    # Action array index zero is idle; index j+1 is a request for job j.
    return np.array(
        [0.0]
        + [
            _probability(driver, j, requests, owner, rule, estimate) * (payments[j] - own_cost[j])
            for j in range(len(payments))
        ]
    )


def action_values(driver, own_cost, payments, requests, sigma, rule, estimate=None):
    """Payoffs of idle and every job; requires only this driver's cost row."""
    n = len(payments)
    requests = validate_requests(requests, n, rule)
    sigma = np.asarray(sigma)
    own_cost, payments = np.asarray(own_cost), np.asarray(payments)
    if (
        not 0 <= driver < n
        or own_cost.shape != (n,)
        or sorted(sigma.tolist()) != list(range(n))
        or not np.isfinite(own_cost).all()
        or not np.isfinite(payments).all()
    ):
        raise ValueError('Invalid driver costs, prices, or recommendation')
    estimate = validate_estimate(rule, estimate, n)
    return _values(driver, own_cost, payments, requests, np.argsort(sigma), rule, estimate)


def compliance_values(cost, sigma, payments, rule, estimate=None):
    """Action values of every driver when all others comply, as an (n, n+1) array.

    Equals stacking `action_values` at the compliance profile; vectorized for the
    one-shot reply. Column zero is idle, column j+1 a request for job j.
    """
    cost, sigma, payments = validate_game(cost, sigma, payments)
    n = len(sigma)
    if rule not in ALL_RULES:
        raise ValueError('Invalid allocation rule')
    estimate = validate_estimate(rule, estimate, n)
    ids = np.arange(n)
    owner = np.argsort(sigma)
    if rule == 'priority':
        win = np.zeros((n, n))
    elif rule == 'uniform':
        win = np.full((n, n), 0.5)
    else:
        # A deviator beats the recommended driver of job j only if its estimate is strictly lower.
        win = (estimate < estimate[owner, ids][None, :]).astype(float)
    win[ids, sigma] = 1.0
    values = win * (payments[None, :] - cost)
    return np.concatenate([np.zeros((n, 1)), values], axis=1)


def _regrets(cost, sigma, payments, requests, rule, estimate=None):
    owner = np.argsort(sigma)
    values = np.array([_values(d, cost[d], payments, requests, owner, rule, estimate) for d in range(len(sigma))])
    payoff = values[np.arange(len(sigma)), requests + 1]
    return values.max(axis=1) - payoff, payoff


def regrets(cost, sigma, payments, requests, rule, estimate=None):
    """Exact unilateral regrets and expected payoffs at any request profile."""
    cost, sigma, payments = validate_game(cost, sigma, payments)
    requests = validate_requests(requests, len(sigma), rule)
    estimate = validate_estimate(rule, estimate, len(sigma))
    return _regrets(cost, sigma, payments, requests, rule, estimate)


def surplus(cost, sigma, payments):
    """Recommended payoff of each driver at the true costs."""
    cost, sigma, payments = validate_game(cost, sigma, payments)
    return payments[sigma] - cost[np.arange(len(sigma)), sigma]


def no_contest_compliance(surplus_values, tau):
    """Probability that a driver with perception noise tau takes a job it is sure to get.

    With perceived cost = true cost + tau * standard normal, the driver complies when its
    perceived surplus is nonnegative, with probability Phi(s / tau); the indicator of
    s >= 0 at tau = 0. Under recommendation priority every other request is worth zero,
    so this is the exact compliance probability of each driver.
    """
    s = np.asarray(surplus_values, dtype=float)
    if tau <= 0:
        return (s >= 0).astype(float)
    return 0.5 * (1.0 + erf(s / (tau * np.sqrt(2.0))))


def _best_reply(values, current, recommended, rng, threshold=0.0):
    best = values.max()
    if values[current + 1] >= best - TOL - threshold:
        return current  # Inertia prevents costless cycling on ties; a switching cost keeps small gains idle.
    candidates = np.flatnonzero(values >= best - TOL) - 1
    if recommended in candidates:
        return int(recommended)
    if -1 in candidates:
        return -1
    return int(rng.choice(candidates))


@dataclass
class Response:
    requests: np.ndarray
    max_regret: float
    equilibrium: bool
    limit_reached: bool
    sweeps: int
    revisions: int


def respond(
    cost,
    sigma,
    payments,
    rule,
    behavior,
    rng,
    initial='compliance',
    max_sweeps=50,
    initial_requests=None,
    estimate=None,
    tau=0.0,
    threshold=0.0,
):
    """Generate requests without assuming convergence or selecting a Nash outcome.

    one_step: simultaneous best replies to the same initial profile.
    best_response: random-order asynchronous strict improvements, with a cap.
    imposed: the existing compliance comparator, not a strategic model.
    tau > 0 adds an independent normal perception error of scale tau to every entry of
    the driver's own cost row before it decides; regrets are reported at true costs.
    threshold >= 0 is a switching cost in minutes: a driver leaves its current request
    only for a perceived gain above it.
    """
    cost, sigma, payments = validate_game(cost, sigma, payments)
    n = len(sigma)
    if rule not in ALL_RULES or behavior not in BEHAVIORS or max_sweeps < 1 or tau < 0 or threshold < 0:
        raise ValueError('Invalid behavior, rule, revision limit, noise scale or switching cost')
    if initial not in ('compliance', 'random'):
        raise ValueError('Initial profile must be compliance or random')
    estimate = validate_estimate(rule, estimate, n)
    perceived = cost + tau * rng.standard_normal(cost.shape) if tau > 0 else cost
    if initial_requests is not None:
        requests = validate_requests(initial_requests, n, rule).copy()
    elif initial == 'compliance':
        requests = sigma.copy()
    else:
        requests = rng.integers(-1, n, size=n)
    owner = np.argsort(sigma)
    revisions, sweeps = 0, 0
    if behavior == 'imposed':
        requests = sigma.copy()
    elif behavior == 'one_step':
        previous = requests.copy()
        table = (
            compliance_values(perceived, sigma, payments, rule, estimate) if np.array_equal(previous, sigma) else None
        )
        for d in range(n):
            values = (
                table[d] if table is not None else _values(d, perceived[d], payments, previous, owner, rule, estimate)
            )
            requests[d] = _best_reply(values, previous[d], sigma[d], rng, threshold)
        revisions = int(np.sum(requests != previous))
        sweeps = 1
    else:
        for sweep in range(max_sweeps):
            changed = 0
            for d in rng.permutation(n):
                values = _values(d, perceived[d], payments, requests, owner, rule, estimate)
                best = _best_reply(values, requests[d], sigma[d], rng, threshold)
                changed += int(best != requests[d])
                requests[d] = best
            revisions += changed
            sweeps = sweep + 1
            if not changed:
                break
    residual = float(_regrets(cost, sigma, payments, requests, rule, estimate)[0].max())
    equilibrium = residual <= TOL
    return Response(requests, residual, equilibrium, behavior == 'best_response' and not equilibrium, sweeps, revisions)


def allocate(requests, sigma, rule, rng, estimate=None):
    """One winner per requested job; return job per driver, -1 if unserved."""
    sigma = np.asarray(sigma)
    if sorted(sigma.tolist()) != list(range(len(sigma))):
        raise ValueError('Recommendation must be a permutation')
    requests = validate_requests(requests, len(sigma), rule)
    estimate = validate_estimate(rule, estimate, len(sigma))
    owner = np.argsort(sigma)
    served = np.full(len(sigma), -1, dtype=int)
    for job in range(len(sigma)):
        candidates = np.flatnonzero(requests == job)
        if not len(candidates):
            continue
        if rule == 'uniform':
            winner = int(rng.choice(candidates))
        elif rule == 'cheapest':
            winner = _cheapest_winner(candidates, job, owner, estimate)
        else:
            winner = int(owner[job] if owner[job] in candidates else candidates[0])
        served[winner] = job
    return served
