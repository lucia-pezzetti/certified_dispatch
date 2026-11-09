"""Paired all-pair interval study; simulator truth is used only for scoring."""
import numpy as np
from .model import random_stream
from .platform import assign, optimum, certificate, mechanisms, evaluate_payments

BOUND_METHODS = ('known_support', 'empirical_same', 'empirical_split', 'structural_lower')
ESTIMATE_METHODS = ('sample_mean', 'reference_q')


def mean_intervals(samples, lower, upper, family_size, alpha):
    """Eq. (23); empirical endpoints are a heuristic, not a confidence bound.

    Clipping both endpoints keeps split-support intervals ordered even when a
    fresh sample mean lies outside the observed calibration range.
    """
    count = samples.shape[-1]
    radius = (upper - lower) * np.sqrt(np.log(2 * family_size / alpha) / (2 * count))
    mean = samples.mean(axis=-1)
    return np.clip(mean - radius, lower, upper), np.clip(mean + radius, lower, upper)


def evaluate_batch(
    c,
    trips,
    estimate,
    lower,
    support,
    Cstar,
    admissible_Cstar,
    reference_cost,
    master,
    checkpoint,
    batch,
    case,
    scenario,
    seed,
    counters,
):
    n = len(estimate)
    ids = np.arange(n)
    max_count = max(c['evaluation_counts'])

    # Distinct streams for mean estimation, support calibration and held-out tails.
    def sample(stream, count):
        return np.array(
            [
                [trips.sample(i, k, count, random_stream(master, stream, checkpoint, batch, case, i, k)) for k in ids]
                for i in ids
            ]
        )

    observations = sample(110, max_count)
    calibration = sample(111, max_count)
    heldout = sample(112, c['support_test_count'])
    D = trips.expected_costs()
    family = n * n * 2 * c['batches'] * len(c['checkpoints']) * len(c['evaluation_counts'])
    rows = []
    for count in c['evaluation_counts']:
        samples = observations[:, :, :count]
        known_L, known_U = mean_intervals(samples, lower, support, family, c['alpha'])
        intervals = {}
        intervals['known_support'] = (known_L, known_U, lower, support)
        for method, pool in [('empirical_same', samples), ('empirical_split', calibration[:, :, :count])]:
            lo, hi = pool.min(axis=-1), pool.max(axis=-1)
            L, U = mean_intervals(samples, lo, hi, family, c['alpha'])
            intervals[method] = (L, U, lo, hi)
        intervals['structural_lower'] = (lower, known_U, lower, support)
        for estimate_method in ESTIMATE_METHODS:
            predicted = samples.mean(axis=-1) if estimate_method == 'sample_mean' else estimate
            assignment = assign(predicted)
            sigma = assignment.sigma
            own = ids, sigma
            execution = float(D[own].mean())
            optD = optimum(D) / n
            loss = execution - optD
            for method, (L, U, lo, hi) in intervals.items():
                upper = U[own]
                joint = bool(np.all((L <= D + 1e-8) & (D <= U + 1e-8)))
                required = bool(np.all(L <= D + 1e-8) and np.all(D[own] <= upper + 1e-8))
                valid = method in ('known_support', 'structural_lower')
                menus, shift = mechanisms(assignment, upper, c['eta'])
                for name, payments in menus.items():
                    cert = certificate(sigma, payments, L, upper)
                    measured = evaluate_payments(D, sigma, payments)
                    passed = cert['control_bound'] <= c['control_tolerance'] and shift <= c['payment_budget']
                    bad = loss > c['control_tolerance'] + 1e-8 or measured['minimum_surplus'] < c['eta'] - 1e-8
                    if required:
                        assert loss <= cert['control_bound'] + 1e-7
                        assert measured['menu_regret'] <= cert['menu_bound'] + 1e-7
                        if name == 'dual_certified':
                            assert measured['minimum_surplus'] >= c['eta'] - 1e-7
                    row = dict(
                        scenario=scenario,
                        seed=seed,
                        batch=batch,
                        checkpoint=checkpoint,
                        evaluation_count=count,
                        estimate_method=estimate_method,
                        bound_method=method,
                        mechanism=name,
                        theoretical_guarantee=int(valid),
                        assignment_loss=loss,
                        assignment_optimal=int(abs(loss) < 1e-8),
                        expected_cost_per_driver=execution,
                        opt_D_per_driver=optD,
                        lower_opt_D_per_driver=optimum(L) / n,
                        lower_benchmark_gap=optD - optimum(L) / n,
                        selected_upper_gap=float((upper - D[own]).mean()),
                        bound_excess=cert['control_bound'] - loss,
                        optimal_routing_bound=float((upper.sum() - optimum(lower)) / n),
                        opt_Cstar_per_driver=optimum(Cstar) / n,
                        control_loss=execution - optimum(Cstar) / n,
                        routing_loss=float((D[own] - Cstar[own]).mean()),
                        routing_restriction_loss=float((admissible_Cstar[own] - Cstar[own]).mean()),
                        routing_learning_loss=float((D[own] - admissible_Cstar[own]).mean()),
                        max_prediction_error=float(np.abs(D - predicted).max()),
                        max_policy_mismatch=float(np.abs(D - reference_cost).max()),
                        dual_certified_shift=shift,
                        selected_upper_covered=int(np.all(D[own] <= upper + 1e-8)),
                        all_pair_covered=int(joint),
                        required_bounds_covered=int(required),
                        mean_interval_width=float((U - L).mean()),
                        heldout_support_exceedance=float(
                            np.mean((heldout < lo[:, :, None]) | (heldout > hi[:, :, None]))
                        ),
                        assignment_bound_violated=int(loss > cert['control_bound'] + 1e-8),
                        menu_bound_violated=int(measured['menu_regret'] > cert['menu_bound'] + 1e-8),
                        test_pass=int(passed) if name == 'dual_certified' else 0,
                        certified_control_and_priority=int(valid and passed) if name == 'dual_certified' else 0,
                        false_pass=int(passed and bad) if name == 'dual_certified' else 0,
                        evaluation_trips=n * n * count,
                        support_calibration_trips=n * n * count if method == 'empirical_split' else 0,
                        support_test_trips=n * n * c['support_test_count'],
                        **counters,
                        **cert,
                        **measured
                    )
                    rows.append(row)
    return rows
