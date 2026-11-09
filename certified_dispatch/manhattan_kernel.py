"""Final Manhattan dispatch loop: eight paired arms, no exploration.

Every arm runs the certified dispatch loop of the paper with imposed
compliance. Arms differ only in the service history and in the confidence
radius; all arms share the demand and the service random numbers of an epoch
and keep their own histories and driver positions, so comparisons are paired.

    history   contexts    allowance Delta_q        radius     guarantee
    pooled    |Q|         rho (v_q - ell_q)        support    any rho
    pooled    |Q|         rho (v_q - ell_q)        variance   rho = 0 only
    private   N |Q|       0                        support    any rho
    private   N |Q|       0                        variance   any rho (i.i.d. per driver)
    naive     |Q|         0                        either     rho = 0 only
    oracle    |Q|         true spread max_a D_a(q) - min_a D_a(q)
                                                   support    any rho (Delta_q known and tight)
                                                   variance   rho = 0 only

The oracle arms are a diagnostic: they take the heterogeneity allowance from
the simulator's exact costs, which the platform does not have. Every other
use of the exact costs only scores decisions. The compiled helpers (Hungarian
solver, counter-based uniforms, route sampling, confidence radius) are the
ones of `optimistic_kernel`; the certified-dual priority arm of
`run_manhattan_strategic` reproduces the pooled-variance arm bit for bit
(see tests/test_manhattan_strategic.py).
"""
import numpy as np
from numba import njit
from .optimistic_kernel import hungarian, uniform, stream, route_sample, matrix, menu, selected_sum, confidence_radius

ARMS = (
    'pooled_support',
    'pooled_variance',
    'private_support',
    'private_variance',
    'naive_support',
    'naive_variance',
    'oracle_support',
    'oracle_variance',
)
HISTORY = tuple(a.split('_')[0] for a in ARMS)
RADIUS = tuple(a.split('_')[1] for a in ARMS)
PRIVATE = np.array([h == 'private' for h in HISTORY])
BERNSTEIN = np.array([r == 'variance' for r in RADIUS], np.int64)
METRICS = (
    'certified',
    'bound',
    'assignment_loss',
    'selected_upper_gap',
    'lower_benchmark_gap',
    'shift',
    'menu_regret',
    'menu_bound',
    'minimum_surplus',
    'mean_width',
    'max_width',
    'mean_allowance',
    'true_spread',
    'covered',
    'false_certificate',
    'assignment_pass',
    'payment_pass',
    'selected_count',
    'lower_count',
    'unobserved_fraction',
    'total_payment',
    'realized_cost',
    'loss_bound_violation',
    'menu_bound_violation',
    'lower_width',
)


def guaranteed(arm, rho):
    """Whether the coverage guarantee applies to this arm at this rho."""
    history, radius = arm.split('_')
    if history == 'private':
        return True
    if history == 'naive':
        return rho == 0.0
    return radius == 'support' or rho == 0.0


def allowances(rho, support_lo, support_hi, true_contexts):
    """Per-arm, per-context allowance Delta_q added on both sides of the interval."""
    public = rho * (support_hi - support_lo)
    spread = true_contexts.max(axis=0) - true_contexts.min(axis=0)
    out = np.zeros((len(ARMS), len(support_lo)))
    for a, history in enumerate(HISTORY):
        if history == 'pooled':
            out[a] = public
        elif history == 'oracle':
            out[a] = spread
    return out


@njit(cache=True)
def run_chunk(
    seed,
    start,
    stop,
    block_size,
    rho,
    alpha,
    eta,
    cap,
    tolerance,
    hubs,
    private_flags,
    bernstein_flags,
    allowance,
    counts,
    totals,
    squares,
    means,
    lows,
    highs,
    origins,
    cumulative,
    coverage_min,
    support_lo,
    support_hi,
    true_contexts,
    paths,
    lengths,
    free,
    delay,
):
    arms, n = origins.shape
    qfamily = hubs * hubs * (hubs - 1)
    blocks = (stop - start + block_size - 1) // block_size
    stats = np.zeros((blocks, arms, len(METRICS)))
    # Decision trace at the final epoch of the chunk, for independent rechecks.
    trace = np.zeros((arms, 6, n, n))
    for epoch in range(start, stop):
        b = (epoch - start) // block_size
        pickups, dropoffs = np.empty(n, np.int64), np.empty(n, np.int64)
        state = stream(seed, epoch, 1)
        for i in range(n):
            state, r = uniform(state)
            pickups[i] = int(r * hubs)
            state, r = uniform(state)
            dropoffs[i] = (pickups[i] + 1 + int(r * (hubs - 1))) % hubs
        for a in range(arms):
            private = private_flags[a]
            C = matrix(means[a], origins[a], pickups, dropoffs, private, hubs)
            L = matrix(lows[a], origins[a], pickups, dropoffs, private, hubs)
            U = matrix(highs[a], origins[a], pickups, dropoffs, private, hubs)
            counts_matrix = matrix(counts[a], origins[a], pickups, dropoffs, private, hubs)
            sigma, phi, psi = hungarian(C)
            sigL, phiL, psiL = hungarian(L)
            optL = selected_sum(L, sigL)
            payments, shift = menu(phi.max() + psi, U, sigma, eta)
            bound = (selected_sum(U, sigma) - optL) / n
            passed = bound <= tolerance and shift <= cap
            # Oracle evaluation begins here. Never used to select an action.
            D = matrix(true_contexts, origins[a], pickups, dropoffs, True, hubs)
            sigD, _, _ = hungarian(D)
            optD = selected_sum(D, sigD) / n
            loss = selected_sum(D, sigma) / n - optD
            width = np.empty(n)
            surplus, regret, regret_bound = np.inf, 0.0, 0.0
            selected_count, lower_count = 0.0, 0.0
            allowance_mean, spread_mean = 0.0, 0.0
            for i in range(n):
                k = sigma[i]
                q = (origins[a, i] * hubs + pickups[k]) * hubs + dropoffs[k]
                width[i] = U[i, k] - L[i, k]
                surplus = min(surplus, payments[k] - D[i, k])
                regret = max(regret, D[i, k] - payments[k])
                regret_bound = max(regret_bound, U[i, k] - payments[k])
                selected_count += counts_matrix[i, k] / n
                lower_count += counts_matrix[i, sigL[i]] / n
                allowance_mean += allowance[a, q] / n
                spread = -np.inf
                for x in range(n):
                    for y in range(n):
                        spread = max(spread, true_contexts[x, q] - true_contexts[y, q])
                spread_mean += spread / n
                for j in range(n):
                    if j != k:
                        regret = max(regret, payments[j] - payments[k] + D[i, k] - D[i, j])
                        regret_bound = max(regret_bound, payments[j] - payments[k] + U[i, k] - L[i, j])
            covered = np.all(L <= D + 1e-8) and np.all(D <= U + 1e-8)
            false_cert = passed and (loss > tolerance + 1e-7 or surplus < eta - 1e-7)
            if covered:
                assert loss <= bound + 1e-7
                assert regret <= regret_bound + 1e-7
                assert surplus >= eta - 1e-7
            # Complete services only after all assignments and prices are chosen.
            realized = 0.0
            for i in range(n):
                k = sigma[i]
                s, p, d = origins[a, i], pickups[k], dropoffs[k]
                q = (s * hubs + p) * hubs + d
                value = route_sample(seed, epoch, i, s, p, d, rho, paths, lengths, free, delay)
                assert support_lo[q] - 1e-8 <= value <= support_hi[q] + 1e-8
                realized += value / n
                owner = i if private else 0
                counts[a, owner, q] += 1
                totals[a, owner, q] += value
                squares[a, owner, q] += value * value
                count = counts[a, owner, q]
                mean = totals[a, owner, q] / count
                family = qfamily * n if private else qfamily
                radius = confidence_radius(
                    bernstein_flags[a],
                    count,
                    mean,
                    totals[a, owner, q],
                    squares[a, owner, q],
                    support_hi[q] - support_lo[q],
                    family,
                    alpha,
                )
                lo = max(support_lo[q], mean - radius - allowance[a, q])
                hi = min(support_hi[q], mean + radius + allowance[a, q])
                means[a, owner, q], lows[a, owner, q], highs[a, owner, q] = mean, lo, hi
                margin = min(true_contexts[i, q] - lo, hi - true_contexts[i, q])
                if not private:
                    for driver in range(n):
                        margin = min(margin, true_contexts[driver, q] - lo, hi - true_contexts[driver, q])
                coverage_min[a] = min(coverage_min[a], margin)
            vals = np.array(
                [
                    (1.0 if passed else 0.0),
                    bound,
                    loss,
                    (selected_sum(U, sigma) - selected_sum(D, sigma)) / n,
                    optD - optL / n,
                    shift,
                    regret,
                    regret_bound,
                    surplus,
                    width.mean(),
                    width.max(),
                    allowance_mean,
                    spread_mean,
                    (1.0 if covered else 0.0),
                    (1.0 if false_cert else 0.0),
                    (1.0 if bound <= tolerance else 0.0),
                    (1.0 if shift <= cap else 0.0),
                    selected_count,
                    lower_count,
                    float(np.sum(counts_matrix == 0)) / (n * n),
                    payments.sum(),
                    realized,
                    max(0.0, loss - bound),
                    max(0.0, regret - regret_bound),
                    (selected_sum(U, sigL) - optL) / n,
                ]
            )
            stats[b, a] += vals
            cumulative[a] += vals
            if epoch == stop - 1:
                trace[a, 0], trace[a, 1], trace[a, 2], trace[a, 3] = C, L, U, D
                for i in range(n):
                    trace[a, 4, 0, i], trace[a, 4, 1, i] = sigma[i], payments[i]
                    trace[a, 4, 2, i], trace[a, 4, 3, i] = phi[i] + 0.0, psi[i]
                    trace[a, 4, 4, i] = origins[a, i]
                    trace[a, 5, 0, i], trace[a, 5, 1, i] = pickups[i], dropoffs[i]
            for i in range(n):
                origins[a, i] = dropoffs[sigma[i]]
    return stats, trace
