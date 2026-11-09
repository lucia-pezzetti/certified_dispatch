"""Platform calculations: observations and public bounds only, no policy oracle."""
from dataclasses import dataclass
import numpy as np
from scipy.optimize import linear_sum_assignment, linprog

TOL = 1e-8


def optimum(cost):
    return float(cost[linear_sum_assignment(cost)].sum())


@dataclass
class Assignment:
    sigma: np.ndarray
    phi: np.ndarray
    psi: np.ndarray
    slack: np.ndarray
    payments: np.ndarray


def assign(cost):
    cost = np.asarray(cost, dtype=float)
    if cost.ndim != 2 or cost.shape[0] != cost.shape[1] or len(cost) < 2 or not np.isfinite(cost).all():
        raise ValueError('Costs must be a finite square matrix with at least two drivers')
    n = len(cost)
    constraints = np.zeros((2 * n, n * n))
    for i in range(n):
        constraints[i, i * n : (i + 1) * n] = 1
        constraints[n + i, i::n] = 1
    result = linprog(cost.ravel(), A_eq=constraints, b_eq=np.ones(2 * n), bounds=(0, None), method='highs')
    if not result.success:
        raise RuntimeError(result.message)
    sigma = linear_sum_assignment(cost)[1]
    phi, psi = result.eqlin.marginals[:n], result.eqlin.marginals[n:]
    slack = cost - phi[:, None] - psi
    if slack.min() < -1e-6 or not np.isclose(phi.sum() + psi.sum(), cost[np.arange(n), sigma].sum()):
        raise RuntimeError('Invalid assignment dual')
    if np.max(np.abs(slack[np.arange(n), sigma])) > 1e-6:
        raise RuntimeError('Complementary slackness failed')
    return Assignment(sigma, phi, psi, slack, phi.max() + psi)


def training_estimate(samples, lower, support, checkpoints, alpha):
    """Two-sided Hoeffding event over N^2 pairs and all training checkpoints."""
    n, _, count = samples.shape
    mean = samples.mean(axis=2)
    radius = (support - lower) * np.sqrt(np.log(2 * n * n * checkpoints / alpha) / (2 * count))
    return mean, radius


def selected_upper(samples, lower, support, comparisons, alpha):
    """One-sided Hoeffding event over all declared selected-pair comparisons."""
    count = samples.shape[1]
    radius = (support - lower) * np.sqrt(np.log(comparisons / alpha) / (2 * count))
    return np.minimum(support, samples.mean(axis=1) + radius)


def certificate(sigma, payments, lower, upper):
    ids = np.arange(len(sigma))
    comparison = payments[None, :] - payments[sigma, None] + upper[:, None] - lower
    comparison[ids, sigma] = -np.inf
    return dict(
        control_bound=float((upper.sum() - optimum(lower)) / len(sigma)),
        menu_bound=max(0.0, float(comparison.max()), float((upper - payments[sigma]).max())),
        minimum_surplus_bound=float((payments[sigma] - upper).min()),
    )


def mechanisms(assignment, upper, eta):
    sigma, base = assignment.sigma, assignment.payments
    shift = max(0.0, float((upper - base[sigma]).max())) + eta
    certified = base + shift
    direct = np.empty(len(sigma))
    direct[sigma] = upper + eta
    # Equal-expenditure comparison supplements the native direct-cost baseline.
    matched = direct + (certified.sum() - direct.sum()) / len(sigma)
    return (
        dict(
            dual_base=base,
            dual_certified=certified,
            uniform_matched=np.full(len(sigma), certified.mean()),
            direct_upper=direct,
            direct_upper_matched=matched,
        ),
        shift,
    )


def slack_certificate(assignment, radius):
    ids, sigma = np.arange(len(assignment.sigma)), assignment.sigma
    rec_e = radius[ids, sigma]
    pairs = radius + rec_e[:, None] - assignment.slack
    pairs[ids, sigma] = -np.inf
    surplus = assignment.phi.max() - assignment.phi
    return max(0.0, float((rec_e - surplus).max()), float(pairs.max()))


def evaluate_payments(cost, sigma, payments):
    """Evaluation only. Exact unilateral regrets at the compliance profile."""
    ids = np.arange(len(sigma))
    utility = payments[None, :] - cost
    rec = utility[ids, sigma]
    gains = np.maximum(0.0, utility.max(axis=1)) - rec
    alternatives = utility.copy()
    alternatives[ids, sigma] = -np.inf
    # A unilateral request contests a job with exactly its recommended driver.
    # Uniform random allocation grants the deviator that job with probability 1/2.
    random_gain = np.maximum(0.0, 0.5 * alternatives.max(axis=1)) - rec
    return dict(
        menu_regret=max(0.0, float(gains.max())),
        mean_gross_gain=float(gains.mean()),
        priority_regret=max(0.0, float(-rec.min())),
        random_regret=max(0.0, float(random_gain.max())),
        negative_surplus_fraction=float(np.mean(rec < -TOL)),
        minimum_surplus=float(rec.min()),
        total_payment=float(payments.sum()),
    )
