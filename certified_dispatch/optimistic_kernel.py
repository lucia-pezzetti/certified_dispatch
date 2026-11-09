"""Compiled, deterministic eight-arm experiment. Oracle arrays only score decisions.

Histories update after all dispatch decisions of an epoch. Routes sample every edge with
fresh counter-based uniforms; repeated edges also receive independent draws.
The confidence radius is Hoeffding (default) or empirical Bernstein; see
`confidence_radius`. Both keep a running sum of squares per arm, owner and context.
"""
import numpy as np
from numba import njit

ARMS = (
    'pooled_none',
    'pooled_optimistic',
    'pooled_random',
    'private_none',
    'private_optimistic',
    'naive_optimistic',
    'pooled_optimistic_relaxed',
    'pooled_random_relaxed',
)
METRICS = (
    'certified',
    'proposed_pass',
    'explored',
    'attempted',
    'blocked',
    'bound',
    'assignment_loss',
    'selected_upper_gap',
    'lower_benchmark_gap',
    'shift',
    'executed_shift',
    'executed_loss',
    'executed_bound',
    'menu_regret',
    'menu_bound',
    'minimum_surplus',
    'mean_width',
    'max_width',
    'estimated_price',
    'covered',
    'false_certificate',
    'assignment_pass',
    'payment_pass',
    'selected_count',
    'lower_count',
    'unobserved_fraction',
    'total_payment',
    'realized_cost',
    'explored_loss_total',
    'explored_width_total',
    'explored_shift',
    'random_requested',
    'exceeds_payment_cap',
    'width_identity_error',
    'loss_bound_violation',
    'menu_bound_violation',
    'shift_width_violation',
    'lower_only_fraction',
    'lower_low_count_fraction',
    'lower_only_low_count_fraction',
    'changed_context_fraction',
    'lower_width',
)


@njit(cache=True)
def hungarian(cost):
    """O(n^3) shortest augmenting paths, feasible dual phi+psi<=cost."""
    n = len(cost)
    u, v = np.zeros(n + 1), np.zeros(n + 1)
    p, way = np.zeros(n + 1, np.int64), np.zeros(n + 1, np.int64)
    for i in range(1, n + 1):
        p[0] = i
        j0 = 0
        dist = np.full(n + 1, np.inf)
        used = np.zeros(n + 1, np.bool_)
        while True:
            used[j0] = True
            i0, delta, j1 = p[j0], np.inf, 0
            for j in range(1, n + 1):
                if not used[j]:
                    cur = cost[i0 - 1, j - 1] - u[i0] - v[j]
                    if cur < dist[j]:
                        dist[j], way[j] = cur, j0
                    if dist[j] < delta:
                        delta, j1 = dist[j], j
            for j in range(n + 1):
                if used[j]:
                    u[p[j]] += delta
                    v[j] -= delta
                else:
                    dist[j] -= delta
            j0 = j1
            if p[j0] == 0:
                break
        while True:
            j1 = way[j0]
            p[j0] = p[j1]
            j0 = j1
            if j0 == 0:
                break
    sigma = np.empty(n, np.int64)
    for j in range(1, n + 1):
        sigma[p[j] - 1] = j - 1
    return sigma, u[1:], v[1:]


@njit(cache=True)
def uniform(state):
    # SplitMix64: explicit uint64 wraparound, top 53 bits to [0,1).
    state = np.uint64(state + np.uint64(0x9E3779B97F4A7C15))
    z = state
    z = (z ^ (z >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
    z = (z ^ (z >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
    z = z ^ (z >> np.uint64(31))
    return state, float(z >> np.uint64(11)) / 9007199254740992.0


@njit(cache=True)
def stream(seed, epoch, purpose):
    return (
        np.uint64(seed)
        ^ (np.uint64(epoch + 1) * np.uint64(0xD2B74407B1CE6E93))
        ^ (np.uint64(purpose + 1) * np.uint64(0xCA5A826395121157))
    )


@njit(cache=True)
def route_sample(seed, epoch, driver, origin, pickup, dropoff, rho, paths, lengths, free, delay):
    state = stream(seed, epoch, 100 + driver)
    state, r = uniform(state)
    policy = driver + 1 if r < rho else 0
    value = 0.0
    for leg in range(2):
        s, g = (origin, pickup) if leg == 0 else (pickup, dropoff)
        for j in range(lengths[policy, s, g]):
            state, r = uniform(state)
            edge = paths[policy, s, g, j]
            value += free[edge] + delay[edge] * r
    return value


@njit(cache=True)
def matrix(history, origins, pickups, dropoffs, private, hubs):
    n = len(origins)
    out = np.empty((n, n))
    for i in range(n):
        for k in range(n):
            q = (origins[i] * hubs + pickups[k]) * hubs + dropoffs[k]
            out[i, k] = history[i if private else 0, q]
    return out


@njit(cache=True)
def menu(base, upper, sigma, eta):
    shift = 0.0
    for i in range(len(sigma)):
        shift = max(shift, upper[i, sigma[i]] - base[sigma[i]])
    shift += eta
    return base + shift, shift


@njit(cache=True)
def selected_sum(cost, sigma):
    value = 0.0
    for i in range(len(sigma)):
        value += cost[i, sigma[i]]
    return value


@njit(cache=True)
def confidence_radius(bernstein, count, mean, total, square, width, family, alpha):
    """Two-sided radius for one context after `count` observations in an interval of length `width`.

    Validity, uniformly over the `family` contexts (|Q| pooled, N|Q| private) and
    over all counts n >= 1, with probability at least 1-alpha: each (context, count)
    pair receives the failure budget delta = alpha/(family*n*(n+1)); the union over
    counts closes because sum_n 1/(n(n+1)) = 1, and the union over contexts
    because there are `family` of them. Observations of one context are i.i.d.
    and bounded by the public support, as both inequalities require.
    Hoeffding: two-sided inequality at budget delta, radius width*sqrt(ln(2/delta)/(2n)).
    Bernstein: Maurer and Pontil (2009), Theorem 4, the one-sided empirical
    Bernstein bound  E[X]-mean <= sqrt(2 V ln(2/d)/n) + 7 width ln(2/d)/(3(n-1)),
    V the sample variance with denominator n-1, applied once to X and once to
    width-X, each at d = delta/2 (so ln(2/d) = ln(4 family n(n+1)/alpha) = nlog+ln 2).
    n = 1 has no sample variance and uses the Hoeffding radius. Clipping to the
    support and the heterogeneity allowance are applied by the caller, unchanged.
    """
    nlog = np.log(2 * family / alpha) + np.log(count) + np.log(count + 1)
    if bernstein and count > 1:
        variance = max(0.0, (square - total * mean) / (count - 1))
        blog = nlog + np.log(2.0)
        return np.sqrt(2 * variance * blog / count) + 7 * width * blog / (3 * (count - 1))
    return width * np.sqrt(nlog / (2 * count))


@njit(cache=True)
def run_chunk(
    seed,
    start,
    stop,
    horizon,
    block_size,
    rho,
    alpha,
    eta,
    cap,
    tolerance,
    exploration_cap,
    hubs,
    bernstein,
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
    opportunities=None,
    only_arm=-1,
    independent_random=False,
):
    arms, n = origins.shape
    qfamily = hubs * hubs * (hubs - 1)
    blocks = (stop - start + block_size - 1) // block_size
    stats = np.zeros((blocks, arms, len(METRICS)))
    # Full independent-recalculation trace at the final epoch of this chunk.
    trace = np.zeros((arms, 6, n, n))
    opt_schedule = False
    relaxed_schedule = False
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
            if only_arm >= 0 and a != only_arm:
                continue
            private = a == 3 or a == 4
            C = matrix(means[a], origins[a], pickups, dropoffs, private, hubs)
            L = matrix(lows[a], origins[a], pickups, dropoffs, private, hubs)
            U = matrix(highs[a], origins[a], pickups, dropoffs, private, hubs)
            counts_matrix = matrix(counts[a], origins[a], pickups, dropoffs, private, hubs)
            sigma, phi, psi = hungarian(C)
            sigL, phiL, psiL = hungarian(L)
            # Exposure counts include every offered driver-job pair, with multiplicity.
            proposed_q = np.empty(n, np.int64)
            lower_q = np.empty(n, np.int64)
            for i in range(n):
                proposed_q[i] = (origins[a, i] * hubs + pickups[sigma[i]]) * hubs + dropoffs[sigma[i]]
                lower_q[i] = (origins[a, i] * hubs + pickups[sigL[i]]) * hubs + dropoffs[sigL[i]]
                if opportunities is not None:
                    for k in range(n):
                        q = (origins[a, i] * hubs + pickups[k]) * hubs + dropoffs[k]
                        opportunities[a, i if private else 0, q] += 1
            lower_only, lower_low, lower_only_low = 0.0, 0.0, 0.0
            for i in range(n):
                included = False
                for j in range(n):
                    if lower_q[i] == proposed_q[j] and (not private or i == j):
                        included = True
                low_count = counts_matrix[i, sigL[i]] < 100
                lower_only += (0.0 if included else 1.0) / n
                lower_low += (1.0 if low_count else 0.0) / n
                lower_only_low += (1.0 if low_count and not included else 0.0) / n
            optL = selected_sum(L, sigL)
            payments, shift = menu(phi.max() + psi, U, sigma, eta)
            bound = (selected_sum(U, sigma) - optL) / n
            passed = bound <= tolerance and shift <= cap
            executed = sigma.copy()
            exec_shift = shift
            explored, attempted, blocked = False, False, False
            random_requested = False
            gate = exploration_cap if a >= 6 else cap
            if a == 1 or a == 4 or a == 5 or a == 6:
                attempted = not passed
                if attempted:
                    payL, hL = menu(phiL.max() + psiL, U, sigL, eta)
                    blocked = hL > gate
                    if not blocked:
                        explored, executed, payments, exec_shift = True, sigL.copy(), payL, hL
                if a == 1:
                    opt_schedule = explored
                if a == 6:
                    relaxed_schedule = explored
            if a == 2 or a == 7:
                random_requested = not passed if independent_random else (opt_schedule if a == 2 else relaxed_schedule)
                attempted = random_requested and not passed
                if attempted:
                    perm = np.arange(n)
                    state = stream(seed, epoch, 2)
                    for j in range(n - 1, 0, -1):
                        state, r = uniform(state)
                        k = int(r * (j + 1))
                        perm[j], perm[k] = perm[k], perm[j]
                    payR, hR = menu(phi.max() + psi, U, perm, eta)
                    blocked = hR > gate
                    if not blocked:
                        explored, executed, payments, exec_shift = True, perm, payR, hR
            # Oracle evaluation begins here. Never used to select an action.
            D = matrix(true_contexts, origins[a], pickups, dropoffs, True, hubs)
            sigD, _, _ = hungarian(D)
            optD = selected_sum(D, sigD) / n
            loss = selected_sum(D, sigma) / n - optD
            exec_loss = selected_sum(D, executed) / n - optD
            exec_bound = (selected_sum(U, executed) - optL) / n
            width = np.empty(n)
            surplus, regret, regret_bound = np.inf, 0.0, 0.0
            selected_count, lower_count, estimated = 0.0, 0.0, 0.0
            for i in range(n):
                k = executed[i]
                width[i] = U[i, k] - L[i, k]
                surplus = min(surplus, payments[k] - D[i, k])
                regret = max(regret, D[i, k] - payments[k])
                regret_bound = max(regret_bound, U[i, k] - payments[k])
                selected_count += counts_matrix[i, k] / n
                lower_count += counts_matrix[i, sigL[i]] / n
                estimated += C[i, k] - phi[i] - psi[k]
                for j in range(n):
                    if j != k:
                        regret = max(regret, payments[j] - payments[k] + D[i, k] - D[i, j])
                        regret_bound = max(regret_bound, payments[j] - payments[k] + U[i, k] - L[i, j])
            covered = np.all(L <= D + 1e-8) and np.all(D <= U + 1e-8)
            certified = passed and not explored
            false_cert = certified and (exec_loss > tolerance + 1e-7 or surplus < eta - 1e-7)
            identity_error, shift_violation = 0.0, 0.0
            if explored and a != 2 and a != 7:
                identity_error = abs(exec_bound - width.mean())
                shift_violation = max(0.0, exec_shift - width.max() - eta)
                assert identity_error < 1e-7 and shift_violation < 1e-7
                assert regret_bound <= width.max() + 1e-7
            if covered:
                assert exec_loss <= exec_bound + 1e-7
                assert regret <= regret_bound + 1e-7
                assert surplus >= eta - 1e-7
            # Multiset difference, not merely permutation changes among duplicate jobs.
            consumed = np.zeros(n, np.bool_)
            changed_contexts = 0.0
            for i in range(n):
                q = (origins[a, i] * hubs + pickups[executed[i]]) * hubs + dropoffs[executed[i]]
                found = False
                for j in range(n):
                    if not consumed[j] and q == proposed_q[j] and (not private or i == j):
                        consumed[j] = True
                        found = True
                        break
                changed_contexts += (0.0 if found else 1.0) / n
            # Complete services only after all assignments and prices are chosen.
            realized = 0.0
            for i in range(n):
                k = executed[i]
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
                    bernstein,
                    count,
                    mean,
                    totals[a, owner, q],
                    squares[a, owner, q],
                    support_hi[q] - support_lo[q],
                    family,
                    alpha,
                )
                allowance = 0.0 if private or a == 5 else rho * (support_hi[q] - support_lo[q])
                lo, hi = max(support_lo[q], mean - radius - allowance), min(support_hi[q], mean + radius + allowance)
                means[a, owner, q], lows[a, owner, q], highs[a, owner, q] = mean, lo, hi
                margin = min(true_contexts[i, q] - lo, hi - true_contexts[i, q])
                if not private:
                    for driver in range(n):
                        margin = min(margin, true_contexts[driver, q] - lo, hi - true_contexts[driver, q])
                coverage_min[a] = min(coverage_min[a], margin)
            vals = np.array(
                [
                    (1.0 if certified else 0.0),
                    (1.0 if passed else 0.0),
                    (1.0 if explored else 0.0),
                    (1.0 if attempted else 0.0),
                    (1.0 if blocked else 0.0),
                    bound,
                    loss,
                    (selected_sum(U, sigma) - selected_sum(D, sigma)) / n,
                    optD - optL / n,
                    shift,
                    exec_shift,
                    exec_loss,
                    exec_bound,
                    regret,
                    regret_bound,
                    surplus,
                    width.mean(),
                    width.max(),
                    estimated,
                    (1.0 if covered else 0.0),
                    (1.0 if false_cert else 0.0),
                    (1.0 if bound <= tolerance else 0.0),
                    (1.0 if shift <= cap else 0.0),
                    selected_count,
                    lower_count,
                    float(np.sum(counts_matrix == 0)) / (n * n),
                    payments.sum(),
                    realized,
                    n * exec_loss if explored else 0.0,
                    width.sum() if explored else 0.0,
                    exec_shift if explored else 0.0,
                    (1.0 if random_requested else 0.0),
                    (1.0 if exec_shift > cap else 0.0),
                    identity_error,
                    max(0.0, exec_loss - exec_bound),
                    max(0.0, regret - regret_bound),
                    shift_violation,
                    lower_only,
                    lower_low,
                    lower_only_low,
                    changed_contexts,
                    (selected_sum(U, sigL) - optL) / n,
                ]
            )
            stats[b, a] += vals
            cumulative[a] += vals
            if epoch == stop - 1:
                trace[a, 0], trace[a, 1], trace[a, 2], trace[a, 3] = C, L, U, D
                for i in range(n):
                    trace[a, 4, 0, i], trace[a, 4, 1, i] = sigma[i], executed[i]
                    trace[a, 4, 2, i], trace[a, 4, 3, i] = payments[i], phi[i] + 0.0
                    trace[a, 4, 4, i], trace[a, 4, 5, i] = psi[i], origins[a, i]
                    trace[a, 5, 0, i], trace[a, 5, 1, i] = pickups[i], dropoffs[i]
                trace[a, 5, 2, 0], trace[a, 5, 2, 1] = (1.0 if explored else 0.0), exec_shift
            for i in range(n):
                origins[a, i] = dropoffs[executed[i]]
    return stats, trace
