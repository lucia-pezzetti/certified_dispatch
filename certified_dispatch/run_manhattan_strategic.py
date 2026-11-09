"""Closed-loop strategic Manhattan dispatch: drivers reply to every offer, the platform learns from executed services.

python -m certified_dispatch.run_manhattan_strategic --output results/strategic
Resume with the same command plus --resume; --report-only rebuilds tables, figures and the reproduction check.

The dispatch loop of `manhattan_kernel` with the pooled-variance history, in which the recommendation is
an offer rather than an order. At every epoch the platform solves the estimated assignment, posts a price
menu and each driver replies once, as in `strategic_game.respond` with behavior one_step, exact knowledge of
its own expected costs (tau = 0) and no switching cost: it requests its recommended job, another job, or
idles, assuming the others comply. The allocation rule (priority, uniform, cheapest; `strategic_game.allocate`)
gives every requested job to one requester. Only the services actually executed enter the pooled history and
move the drivers. Three modelling choices are fixed here: an unrequested job is lost for the epoch; a driver
that idles or loses a contest keeps its position; drivers do not learn across epochs, every reply is a fresh
one-shot reply to the current offer.

Arms are price rule x allocation rule, each with its own history and positions. All arms share the demand and
the service noise of an epoch through the kernel's counter-based uniforms, and the uniform rule's allocation
draws, so comparisons stay paired after the histories diverge. The dual_certified x priority arm executes every
recommendation while the intervals cover the truth and therefore reproduces the pooled-variance trajectory of
`run_manhattan_scale`; with --reference pointing at that run, the report checks this block by block.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
import importlib.metadata
import json
import os
from pathlib import Path
import shutil
import sys
import time
import numpy as np
from numba import njit
from .manhattan_kernel import ARMS as KERNEL_ARMS, METRICS as KERNEL_METRICS
from .optimistic_kernel import hungarian, uniform, stream, route_sample, matrix, menu, selected_sum, confidence_radius
from .common import (
    NETWORK,
    INPUT_KEYS,
    write_csv,
    interval,
    atomic_json,
    atomic_npz,
    sha,
    snapshot_source,
    check_source_unchanged,
    verify_source,
)
from .run_manhattan_scale import prepare_inputs_scale, cell_name

PRICES = ('dual_certified', 'dual_base', 'direct_upper', 'direct_upper_matched', 'uniform_matched')
RULES = ('priority', 'uniform', 'cheapest')
LABELS = {
    'dual_certified': 'Certified dual',
    'dual_base': 'Base dual',
    'direct_upper': 'Direct upper',
    'direct_upper_matched': 'Direct, matched',
    'uniform_matched': 'Uniform, matched',
}
COLORS = {
    'dual_base': '#0072B2',
    'dual_certified': '#009E73',
    'direct_upper': '#CC79A7',
    'direct_upper_matched': '#8B5A2B',
    'uniform_matched': '#E69F00',
}
METRICS = (
    'completion',
    'request_compliance',
    'idle',
    'deviating',
    'rejected',
    'contested_jobs',
    'unrequested_jobs',
    'full_execution',
    'certified',
    'false_certificate',
    'kernel_false_certificate',
    'bound',
    'shift',
    'assignment_pass',
    'payment_pass',
    'assignment_loss',
    'compliance_regret',
    'menu_regret',
    'minimum_surplus',
    'mean_width',
    'max_width',
    'covered',
    'selected_count',
    'unobserved_fraction',
    'offered_payment',
    'total_payment',
    'executed_expected_cost',
    'recommended_expected_cost',
    'realized_cost',
    'lower_benchmark_gap',
)
ROWS = (
    'sigma',
    'payments',
    'phi',
    'psi',
    'origins',
    'requests',
    'served',
    'pickups',
    'dropoffs',
    'certified_menu',
    'base_menu',
    'flags',
)
STATE_KEYS = (
    'counts',
    'totals',
    'squares',
    'means',
    'lows',
    'highs',
    'origins',
    'cumulative',
    'coverage_min',
    'first_deviation',
)
# Kernel metric -> closed-loop metric with the same definition, for the reproduction check.
COMMON = {
    'certified': 'certified',
    'bound': 'bound',
    'assignment_loss': 'assignment_loss',
    'shift': 'shift',
    'menu_regret': 'menu_regret',
    'minimum_surplus': 'minimum_surplus',
    'mean_width': 'mean_width',
    'max_width': 'max_width',
    'covered': 'covered',
    'assignment_pass': 'assignment_pass',
    'payment_pass': 'payment_pass',
    'selected_count': 'selected_count',
    'unobserved_fraction': 'unobserved_fraction',
    'total_payment': 'offered_payment',
    'realized_cost': 'realized_cost',
    'lower_benchmark_gap': 'lower_benchmark_gap',
}
TOL = 1e-9  # tie tolerance of `strategic_game`


def arm_name(price, rule):
    return f'{price}__{rule}'


def split_arm(name):
    price, rule = name.split('__')
    return price, rule


@njit(cache=True)
def posted_menu(price, base, certified, U, sigma, eta):
    """The menus of `platform.mechanisms` from the kernel's dual and certified menu."""
    n = len(sigma)
    pay = np.empty(n)
    if price == 0:
        pay[:] = certified
    elif price == 1:
        pay[:] = base
    elif price == 2 or price == 3:
        for i in range(n):
            pay[sigma[i]] = U[i, sigma[i]] + eta
        if price == 3:
            pay += (certified.sum() - pay.sum()) / n
    else:
        pay[:] = certified.mean()
    return pay


@njit(cache=True)
def one_step_reply(D, C, sigma, owner, pay, rule, state):
    """One simultaneous best reply per driver at its true costs, the others assumed to comply.

    Same decision as `strategic_game.respond(..., 'one_step', tau=0)`: the win probability of another
    driver's job is 0 under priority, 1/2 under uniform allocation, and 1 under the cheapest rule when the
    platform's estimate ranks the deviator strictly cheaper. Ties keep the recommendation, then idle, then
    a uniform draw among the tied jobs. Returns the requests (-1 idles) and the largest regret at compliance.
    """
    n = len(sigma)
    requests = np.empty(n, np.int64)
    values = np.empty(n + 1)
    max_regret = 0.0
    for d in range(n):
        values[0] = 0.0
        for j in range(n):
            if j == sigma[d]:
                w = 1.0
            elif rule == 0:
                w = 0.0
            elif rule == 1:
                w = 0.5
            else:
                w = 1.0 if C[d, j] < C[owner[j], j] else 0.0
            values[j + 1] = w * (pay[j] - D[d, j])
        best = values.max()
        current = sigma[d]
        max_regret = max(max_regret, best - values[current + 1])
        if values[current + 1] >= best - TOL:
            requests[d] = current
        elif values[0] >= best - TOL:
            requests[d] = -1
        else:
            m = 0
            for j in range(n):
                if values[j + 1] >= best - TOL:
                    m += 1
            state, r = uniform(state)
            pick = int(r * m)
            m = 0
            requests[d] = -1
            for j in range(n):
                if values[j + 1] >= best - TOL:
                    if m == pick:
                        requests[d] = j
                    m += 1
    return requests, max_regret, state


@njit(cache=True)
def allocate_rule(requests, sigma, owner, C, rule, state):
    """One winner per requested job, as `strategic_game.allocate`; -1 for drivers left without a job.

    The uniform rule draws one uniform per job whether or not the job is contested, so arms that share
    the draw stream receive the same winner whenever they receive the same requests.
    """
    n = len(sigma)
    served = np.full(n, -1, np.int64)
    for j in range(n):
        m = 0
        for d in range(n):
            if requests[d] == j:
                m += 1
        r = 0.0
        if rule == 1:
            state, r = uniform(state)
        if m == 0:
            continue
        winner = -1
        if rule == 0:
            if requests[owner[j]] == j:
                winner = owner[j]
            else:
                for d in range(n):
                    if requests[d] == j:
                        winner = d
                        break
        elif rule == 1:
            pick = int(r * m)
            c = 0
            for d in range(n):
                if requests[d] == j:
                    if c == pick:
                        winner = d
                    c += 1
        else:
            for d in range(n):
                if requests[d] == j:
                    if winner < 0 or C[d, j] < C[winner, j] or (C[d, j] == C[winner, j] and d == owner[j]):
                        winner = d
        served[winner] = j
    return served, state


@njit(cache=True)
def run_chunk_strategic(
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
    prices,
    rules,
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
    first_deviation,
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
    # Decision trace at the final epoch of the chunk: matrices and per-driver or per-job rows.
    trace = np.zeros((arms, 4, n, n))
    rows = np.zeros((arms, len(ROWS), n))
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
            price, rule = prices[a], rules[a]
            C = matrix(means[a], origins[a], pickups, dropoffs, False, hubs)
            L = matrix(lows[a], origins[a], pickups, dropoffs, False, hubs)
            U = matrix(highs[a], origins[a], pickups, dropoffs, False, hubs)
            counts_matrix = matrix(counts[a], origins[a], pickups, dropoffs, False, hubs)
            sigma, phi, psi = hungarian(C)
            sigL, phiL, psiL = hungarian(L)
            optL = selected_sum(L, sigL)
            base = phi.max() + psi
            certified, shift = menu(base, U, sigma, eta)
            bound = (selected_sum(U, sigma) - optL) / n
            passed = bound <= tolerance and shift <= cap
            payments = posted_menu(price, base, certified, U, sigma, eta)
            owner = np.empty(n, np.int64)
            for i in range(n):
                owner[sigma[i]] = i
            # The true expected costs are the drivers' own knowledge; the platform never reads D.
            D = matrix(true_contexts, origins[a], pickups, dropoffs, True, hubs)
            requests, compliance_regret, _ = one_step_reply(D, C, sigma, owner, payments, rule, stream(seed, epoch, 6))
            served, _ = allocate_rule(requests, sigma, owner, C, rule, stream(seed, epoch, 5))
            # Oracle evaluation of the recommendation, as in the kernel.
            sigD, _, _ = hungarian(D)
            optD = selected_sum(D, sigD) / n
            loss = selected_sum(D, sigma) / n - optD
            width = np.empty(n)
            surplus, regret = np.inf, 0.0
            selected_count = 0.0
            for i in range(n):
                k = sigma[i]
                width[i] = U[i, k] - L[i, k]
                surplus = min(surplus, payments[k] - D[i, k])
                regret = max(regret, D[i, k] - payments[k])
                selected_count += counts_matrix[i, k] / n
                for j in range(n):
                    if j != k:
                        regret = max(regret, payments[j] - payments[k] + D[i, k] - D[i, j])
            covered = np.all(L <= D + 1e-8) and np.all(D <= U + 1e-8)
            # Replies and allocation.
            n_served, n_comply, n_idle, n_deviate, n_rejected = 0, 0, 0, 0, 0
            requested = np.zeros(n, np.int64)
            full = True
            for d in range(n):
                if requests[d] == -1:
                    n_idle += 1
                else:
                    requested[requests[d]] += 1
                    if requests[d] == sigma[d]:
                        n_comply += 1
                    else:
                        n_deviate += 1
                    if served[d] < 0:
                        n_rejected += 1
                if served[d] >= 0:
                    n_served += 1
                if served[d] != sigma[d]:
                    full = False
            n_contested, n_unrequested = 0, 0
            for j in range(n):
                if requested[j] > 1:
                    n_contested += 1
                if requested[j] == 0:
                    n_unrequested += 1
            deviation = n_comply < n
            if deviation and first_deviation[a] < 0:
                first_deviation[a] = epoch
            false_cert = passed and deviation
            kernel_false = passed and (loss > tolerance + 1e-7 or surplus < eta - 1e-7)
            if covered and price == 0:
                assert surplus >= eta - 1e-7
                assert loss <= bound + 1e-7
            paid, executed_cost, recommended_cost = 0.0, 0.0, 0.0
            for i in range(n):
                recommended_cost += D[i, sigma[i]]
                if served[i] >= 0:
                    paid += payments[served[i]]
                    executed_cost += D[i, served[i]]
            # Complete the allocated services only; idle and rejected drivers keep their position.
            realized = 0.0
            for i in range(n):
                k = served[i]
                if k < 0:
                    continue
                s, p, d = origins[a, i], pickups[k], dropoffs[k]
                q = (s * hubs + p) * hubs + d
                value = route_sample(seed, epoch, i, s, p, d, rho, paths, lengths, free, delay)
                assert support_lo[q] - 1e-8 <= value <= support_hi[q] + 1e-8
                realized += value / n
                counts[a, 0, q] += 1
                totals[a, 0, q] += value
                squares[a, 0, q] += value * value
                count = counts[a, 0, q]
                mean = totals[a, 0, q] / count
                radius = confidence_radius(
                    1, count, mean, totals[a, 0, q], squares[a, 0, q], support_hi[q] - support_lo[q], qfamily, alpha
                )
                lo = max(support_lo[q], mean - radius - allowance[q])
                hi = min(support_hi[q], mean + radius + allowance[q])
                means[a, 0, q], lows[a, 0, q], highs[a, 0, q] = mean, lo, hi
                margin = min(true_contexts[i, q] - lo, hi - true_contexts[i, q])
                for driver in range(n):
                    margin = min(margin, true_contexts[driver, q] - lo, hi - true_contexts[driver, q])
                coverage_min[a] = min(coverage_min[a], margin)
            vals = np.array(
                [
                    n_served / n,
                    n_comply / n,
                    n_idle / n,
                    n_deviate / n,
                    n_rejected / n,
                    n_contested / n,
                    n_unrequested / n,
                    (1.0 if full else 0.0),
                    (1.0 if passed else 0.0),
                    (1.0 if false_cert else 0.0),
                    (1.0 if kernel_false else 0.0),
                    bound,
                    shift,
                    (1.0 if bound <= tolerance else 0.0),
                    (1.0 if shift <= cap else 0.0),
                    loss,
                    compliance_regret,
                    regret,
                    surplus,
                    width.mean(),
                    width.max(),
                    (1.0 if covered else 0.0),
                    selected_count,
                    float(np.sum(counts_matrix == 0)) / (n * n),
                    payments.sum(),
                    paid,
                    executed_cost,
                    recommended_cost,
                    realized,
                    optD - optL / n,
                ]
            )
            stats[b, a] += vals
            cumulative[a] += vals
            if epoch == stop - 1:
                trace[a, 0], trace[a, 1], trace[a, 2], trace[a, 3] = C, L, U, D
                for i in range(n):
                    rows[a, 0, i], rows[a, 1, i], rows[a, 2, i], rows[a, 3, i] = (
                        sigma[i],
                        payments[i],
                        phi[i] + 0.0,
                        psi[i],
                    )
                    rows[a, 4, i], rows[a, 5, i], rows[a, 6, i] = origins[a, i], requests[i], served[i]
                    rows[a, 7, i], rows[a, 8, i], rows[a, 9, i], rows[a, 10, i] = (
                        pickups[i],
                        dropoffs[i],
                        certified[i],
                        base[i],
                    )
                rows[a, 11, 0], rows[a, 11, 1], rows[a, 11, 2] = (1.0 if passed else 0.0), bound, shift
            for i in range(n):
                if served[i] >= 0:
                    origins[a, i] = dropoffs[served[i]]
    return stats, trace, rows


def fresh_state(inputs, arms, fleet, hubs, seed):
    """Prior intervals at the public support, same initial positions as `run_manhattan_scale.fresh_state`."""
    shape = (arms, 1, hubs**3)
    lo = np.broadcast_to(inputs['support_lo'], shape).copy()
    hi = np.broadcast_to(inputs['support_hi'], shape).copy()
    origins = np.tile(np.random.default_rng(seed).integers(hubs, size=fleet), (arms, 1))
    return dict(
        counts=np.zeros(shape, np.int64),
        totals=np.zeros(shape),
        squares=np.zeros(shape),
        means=(lo + hi) / 2,
        lows=lo,
        highs=hi,
        origins=origins,
        cumulative=np.zeros((arms, len(METRICS))),
        coverage_min=np.full(arms, np.inf),
        first_deviation=np.full(arms, -1, np.int64),
    )


def chunk(config, cell, start, stop, state, inputs):
    prices = np.array([PRICES.index(split_arm(a)[0]) for a in cell['arms']], np.int64)
    rules = np.array([RULES.index(split_arm(a)[1]) for a in cell['arms']], np.int64)
    allowance = cell['rho'] * (inputs['support_hi'] - inputs['support_lo'])
    return run_chunk_strategic(
        cell['seed'],
        start,
        stop,
        config['block_size'],
        cell['rho'],
        config['alpha'],
        config['eta'],
        config['payment_budget'],
        config['assignment_tolerance'],
        cell['hubs'],
        prices,
        rules,
        allowance,
        *[state[k] for k in STATE_KEYS],
        *[inputs[k] for k in INPUT_KEYS],
    )


def prepare_cell(output, config, cell):
    path = Path(output) / 'inputs' / f"{cell['name']}.npz"
    if not path.exists():
        prepare_inputs_scale(config['network_file'], cell['hubs'], cell['rho'], cell['fleet'], cell['seed'], path)
    reference = Path(config['reference']) / 'inputs' / f"{cell['name']}.npz" if config.get('reference') else None
    if reference is not None and reference.exists():
        ours, theirs = dict(np.load(path)), dict(np.load(reference))
        for key in INPUT_KEYS:
            if not np.array_equal(ours[key], theirs[key]):
                raise ValueError(f"Prepared inputs differ from the reference run: {cell['name']} {key}")
    return cell['name']


def run_cell(output, cell):
    output = Path(output)
    config = json.loads((output / 'config.json').read_text())
    directory = output / cell['name']
    directory.mkdir(exist_ok=True)
    status_path = directory / 'status.json'
    if status_path.exists() and json.loads(status_path.read_text())['status'] == 'complete':
        return cell['name']
    inputs = dict(np.load(output / 'inputs' / f"{cell['name']}.npz"))
    checkpoint = directory / 'checkpoint.npz'
    if checkpoint.exists():
        saved = dict(np.load(checkpoint))
        start = int(saved['epoch'])
        state = {k: saved[k] for k in STATE_KEYS}
    else:
        start = 0
        state = fresh_state(inputs, len(cell['arms']), cell['fleet'], cell['hubs'], cell['seed'])
    started = time.perf_counter()
    resumed_at = start
    status = dict(status='running', pid=os.getpid(), epoch=start, **cell)
    atomic_json(status_path, status)
    try:
        while start < cell['epochs']:
            stop = min(start + cell['checkpoint_every'], cell['epochs'])
            stats, trace, rows = chunk(config, cell, start, stop, state, inputs)
            atomic_npz(
                directory / f'blocks_{stop:010d}.npz', start=start, stop=stop, stats=stats, trace=trace, rows=rows
            )
            atomic_npz(checkpoint, epoch=stop, **state)
            elapsed = time.perf_counter() - started
            status.update(
                epoch=stop,
                elapsed_session_seconds=elapsed,
                epochs_per_second=(stop - resumed_at) / elapsed,
                completion={a: float(state['cumulative'][j, 0] / stop) for j, a in enumerate(cell['arms'])},
                certified_fraction={
                    a: float(state['cumulative'][j, METRICS.index('certified')] / stop)
                    for j, a in enumerate(cell['arms'])
                },
                first_deviation_epoch={a: int(state['first_deviation'][j]) for j, a in enumerate(cell['arms'])},
                minimum_coverage_margin=state['coverage_min'].tolist(),
            )
            atomic_json(status_path, status)
            print(
                f"{cell['name']}: {stop:,}/{cell['epochs']:,} epochs; {status['epochs_per_second']:.0f} "
                f"{len(cell['arms'])}-arm epochs/s",
                flush=True,
            )
            start = stop
        status['status'] = 'complete'
        atomic_json(status_path, status)
        return cell['name']
    except BaseException as exc:
        status.update(status='interrupted' if isinstance(exc, KeyboardInterrupt) else 'failed', error=repr(exc))
        atomic_json(status_path, status)
        raise


def build_cells(args, arms):
    cells = []
    for seed in range(args.base_seed, args.base_seed + args.seeds):
        for rho in args.rhos:
            cells.append(
                dict(
                    name=cell_name(args.fleet, args.hubs, rho, seed),
                    seed=seed,
                    rho=rho,
                    fleet=args.fleet,
                    hubs=args.hubs,
                    epochs=args.epochs,
                    checkpoint_every=args.checkpoint_every,
                    arms=arms,
                )
            )
    return cells


# ----------------------------------------------------------------------------------------------------------------
# Report: tables, curves, executed services per context, reproduction check, comparison with the snapshot study.


def load_cell(output, cell):
    directory = Path(output) / cell['name']
    status = json.loads((directory / 'status.json').read_text())
    if status['status'] != 'complete':
        raise ValueError('Not complete: ' + cell['name'])
    blocks, traces, rows, end = [], [], [], 0
    for file in sorted(directory.glob('blocks_*.npz')):
        data = dict(np.load(file))
        if int(data['stop']) > cell['epochs']:
            continue
        assert int(data['start']) == end
        end = int(data['stop'])
        blocks.append(data['stats'])
        traces.append((end, data['trace']))
        rows.append((end, data['rows']))
    assert end == cell['epochs']
    stats = np.concatenate(blocks)
    checkpoint = dict(np.load(directory / 'checkpoint.npz'))
    assert int(checkpoint['epoch']) == cell['epochs']
    np.testing.assert_allclose(stats.sum(axis=0), checkpoint['cumulative'], rtol=1e-9, atol=1e-6)
    return dict(stats=stats, traces=traces, rows=rows, checkpoint=checkpoint, status=status)


def window_means(stats, block_size, blocks):
    """Per-epoch means of every metric over the final `blocks` blocks: (arms, metrics)."""
    return stats[-blocks:].sum(axis=0) / (blocks * block_size)


def summarize_arms(c, loaded):
    """Means across seeds (within seed first) of every metric per arm, rho and window."""
    block_size = c['block_size']
    ix = {m: j for j, m in enumerate(METRICS)}
    out, flat = [], []
    for rho in c['rhos']:
        cells = [cell for cell in c['cells'] if cell['rho'] == rho]
        blocks = cells[0]['epochs'] // block_size
        windows = (
            ('whole_run', blocks),
            ('terminal_1pct', max(1, blocks // 100)),
            ('terminal_100k', min(blocks, max(1, 100_000 // block_size))),
        )
        for name, count in windows:
            per_seed = np.array([window_means(loaded[cell['name']]['stats'], block_size, count) for cell in cells])
            for j, arm in enumerate(c['arms']):
                price, rule = split_arm(arm)
                row = dict(rho=rho, window=name, price=price, allocation=rule, seeds=len(cells))
                for m in METRICS:
                    values = per_seed[:, j, ix[m]]
                    row[m] = float(values.mean())
                    row[m + '_ci95'] = interval(values)
                row['false_certificates_total'] = float(
                    np.sum([loaded[cell['name']]['stats'][-count:, j, ix['false_certificate']].sum() for cell in cells])
                )
                row['services_total'] = float(
                    np.sum(
                        [
                            loaded[cell['name']]['stats'][-count:, j, ix['completion']].sum() * cells[0]['fleet']
                            for cell in cells
                        ]
                    )
                )
                out.append(row)
                for m in (
                    'completion',
                    'certified',
                    'false_certificate',
                    'offered_payment',
                    'total_payment',
                    'compliance_regret',
                    'idle',
                    'deviating',
                    'rejected',
                    'mean_width',
                    'bound',
                ):
                    flat.append(
                        dict(
                            rho=rho,
                            window=name,
                            price=price,
                            allocation=rule,
                            metric=m,
                            mean=float(per_seed[:, j, ix[m]].mean()),
                            ci95_halfwidth=interval(per_seed[:, j, ix[m]]),
                            seeds=len(cells),
                        )
                    )
        # Means at the saved checkpoint epochs, the sample of the snapshot study (one epoch per checkpoint).
        per_seed = []
        for cell in cells:
            rows = loaded[cell['name']]['rows']
            served = np.array([(r[:, ROWS.index('served')] >= 0).mean(axis=1) for _, r in rows])  # (checkpoints, arms)
            passed = np.array([r[:, ROWS.index('flags'), 0] for _, r in rows])
            per_seed.append(np.stack([served.mean(axis=0), passed.mean(axis=0)]))
        per_seed = np.array(per_seed)  # (seeds, 2, arms)
        for j, arm in enumerate(c['arms']):
            price, rule = split_arm(arm)
            for m, k in (('completion', 0), ('certified', 1)):
                flat.append(
                    dict(
                        rho=rho,
                        window='checkpoint_epochs',
                        price=price,
                        allocation=rule,
                        metric=m,
                        mean=float(per_seed[:, k, j].mean()),
                        ci95_halfwidth=interval(per_seed[:, k, j]),
                        seeds=len(cells),
                    )
                )
    return out, flat


def curves(c, loaded):
    """Seed means of block-level per-epoch metrics against the epoch (end of block) per arm and rho."""
    block_size = c['block_size']
    ix = {m: j for j, m in enumerate(METRICS)}
    out = []
    for rho in c['rhos']:
        cells = [cell for cell in c['cells'] if cell['rho'] == rho]
        stats = (
            np.array([loaded[cell['name']]['stats'] for cell in cells]) / block_size
        )  # (seeds, blocks, arms, metrics)
        cumulative_false = np.cumsum(
            np.array([loaded[cell['name']]['stats'][:, :, ix['false_certificate']] for cell in cells]), axis=1
        )
        for bi in range(stats.shape[1]):
            for j, arm in enumerate(c['arms']):
                price, rule = split_arm(arm)
                row = dict(
                    rho=rho,
                    price=price,
                    allocation=rule,
                    epoch=(bi + 1) * block_size,
                    services=float(
                        stats[:, : bi + 1, j, ix['completion']].sum(axis=1).mean() * block_size * cells[0]['fleet']
                    ),
                )
                for m in (
                    'completion',
                    'certified',
                    'false_certificate',
                    'offered_payment',
                    'total_payment',
                    'mean_width',
                    'bound',
                    'idle',
                    'deviating',
                    'rejected',
                ):
                    row[m] = float(stats[:, bi, j, ix[m]].mean())
                    row[m + '_ci95'] = interval(stats[:, bi, j, ix[m]])
                row['false_certificates_cumulative'] = float(cumulative_false[:, bi, j].mean())
                out.append(row)
    return out


def services_per_context(c, loaded):
    """Executed services per context at the end of the run, per arm and seed, with a per-arm summary."""
    rows, summary = [], []
    hubs = c['hubs']
    for cell in c['cells']:
        counts = loaded[cell['name']]['checkpoint']['counts'][:, 0, :]
        valid = np.array([(q // hubs) % hubs != q % hubs for q in range(hubs**3)])  # pickup != dropoff
        for j, arm in enumerate(c['arms']):
            price, rule = split_arm(arm)
            for q in range(hubs**3):
                if valid[q]:
                    rows.append(
                        dict(
                            rho=cell['rho'],
                            seed=cell['seed'],
                            price=price,
                            allocation=rule,
                            context=q,
                            origin=q // (hubs * hubs),
                            pickup=(q // hubs) % hubs,
                            dropoff=q % hubs,
                            services=int(counts[j, q]),
                        )
                    )
            cq = counts[j, valid]
            summary.append(
                dict(
                    rho=cell['rho'],
                    seed=cell['seed'],
                    price=price,
                    allocation=rule,
                    services_total=int(cq.sum()),
                    contexts=int(valid.sum()),
                    contexts_unobserved=int((cq == 0).sum()),
                    contexts_below_20=int((cq < 20).sum()),
                    services_min=int(cq.min()),
                    services_median=float(np.median(cq)),
                    services_max=int(cq.max()),
                    share_of_compliant=float(cq.sum() / (cell['epochs'] * cell['fleet'])),
                )
            )
    return rows, summary


def reproduction_check(c, loaded, reference):
    """Compare the dual_certified x priority arm with the pooled-variance arm of the reference scale run.

    Block statistics of every common metric, the decision trace at every common checkpoint, and the final
    history when the horizons coincide. Reports the first epoch at which the arm deviated, if any, and the
    first block or checkpoint where the trajectories differ.
    """
    arm = arm_name('dual_certified', 'priority')
    if not reference or arm not in c['arms'] or not (Path(reference) / 'config.json').exists():
        return [dict(status='skipped', reason='arm or reference run missing')]
    reference = Path(reference)
    rc = json.loads((reference / 'config.json').read_text())
    assert tuple(rc['arms']) == KERNEL_ARMS and tuple(rc['metrics']) == KERNEL_METRICS
    assert rc['block_size'] == c['block_size']
    ref_arm = KERNEL_ARMS.index('pooled_variance')
    j = c['arms'].index(arm)
    kix = {m: i for i, m in enumerate(KERNEL_METRICS)}
    ix = {m: i for i, m in enumerate(METRICS)}
    out = []
    for cell in c['cells']:
        ref_cell = next((x for x in rc['cells'] if x['name'] == cell['name']), None)
        row = dict(
            cell=cell['name'],
            rho=cell['rho'],
            seed=cell['seed'],
            arm=arm,
            reference_arm='pooled_variance',
            first_deviation_epoch=int(loaded[cell['name']]['checkpoint']['first_deviation'][j]),
        )
        if ref_cell is None:
            row.update(status='skipped', reason='no reference cell')
            out.append(row)
            continue
        data = loaded[cell['name']]
        blocks = min(data['stats'].shape[0], ref_cell['epochs'] // rc['block_size'])
        ref_stats, ref_traces, end = [], {}, 0
        for file in sorted((reference / cell['name']).glob('blocks_*.npz')):
            d = dict(np.load(file))
            if int(d['stop']) > ref_cell['epochs']:
                continue
            ref_stats.append(d['stats'])
            ref_traces[int(d['stop'])] = d['trace'][ref_arm]
            end = int(d['stop'])
            if end >= cell['epochs']:
                break
        ref_stats = np.concatenate(ref_stats)[:blocks]
        mismatches = []
        first_bad_block = None
        for km, m in COMMON.items():
            ours, theirs = data['stats'][:blocks, j, ix[m]], ref_stats[:, ref_arm, kix[km]]
            bad = np.flatnonzero(ours != theirs)
            if len(bad):
                worst = float(np.max(np.abs(ours[bad] - theirs[bad])))
                exact_fail = bool(np.any(~np.isclose(ours[bad], theirs[bad], rtol=1e-9, atol=1e-7)))
                mismatches.append(
                    dict(
                        metric=m,
                        reference_metric=km,
                        first_block=int(bad[0]),
                        first_epoch_range=[int(bad[0]) * c['block_size'], int(bad[0] + 1) * c['block_size']],
                        blocks=int(len(bad)),
                        max_abs_difference=worst,
                        beyond_rounding=exact_fail,
                    )
                )
                if exact_fail:
                    first_bad_block = int(bad[0]) if first_bad_block is None else min(first_bad_block, int(bad[0]))
        trace_checked, trace_bad = 0, []
        for (stop, tr), (stop2, rw) in zip(data['traces'], data['rows']):
            assert stop == stop2
            if stop not in ref_traces:
                continue
            ref = ref_traces[stop]
            same = (
                np.array_equal(tr[j], ref[:4])
                and np.array_equal(rw[j, ROWS.index('sigma')], ref[4, 0])
                and np.array_equal(rw[j, ROWS.index('payments')], ref[4, 1])
                and np.array_equal(rw[j, ROWS.index('phi')], ref[4, 2])
                and np.array_equal(rw[j, ROWS.index('psi')], ref[4, 3])
                and np.array_equal(rw[j, ROWS.index('origins')], ref[4, 4])
                and np.array_equal(rw[j, ROWS.index('pickups')], ref[5, 0])
                and np.array_equal(rw[j, ROWS.index('dropoffs')], ref[5, 1])
                and np.array_equal(rw[j, ROWS.index('served')], ref[4, 0])
            )
            trace_checked += 1
            if not same:
                trace_bad.append(int(stop))
        final_state = None
        if cell['epochs'] == ref_cell['epochs']:
            ref_ckpt = dict(np.load(reference / cell['name'] / 'checkpoint.npz'))
            final_state = dict(
                counts=bool(np.array_equal(data['checkpoint']['counts'][j, 0], ref_ckpt['counts'][ref_arm, 0])),
                totals=bool(np.array_equal(data['checkpoint']['totals'][j, 0], ref_ckpt['totals'][ref_arm, 0])),
                squares=bool(np.array_equal(data['checkpoint']['squares'][j, 0], ref_ckpt['squares'][ref_arm, 0])),
                means=bool(np.array_equal(data['checkpoint']['means'][j, 0], ref_ckpt['means'][ref_arm, 0])),
                lows=bool(np.array_equal(data['checkpoint']['lows'][j, 0], ref_ckpt['lows'][ref_arm, 0])),
                highs=bool(np.array_equal(data['checkpoint']['highs'][j, 0], ref_ckpt['highs'][ref_arm, 0])),
                origins=bool(np.array_equal(data['checkpoint']['origins'][j], ref_ckpt['origins'][ref_arm])),
                coverage_min=bool(data['checkpoint']['coverage_min'][j] == ref_ckpt['coverage_min'][ref_arm]),
            )
        executed = float(data['stats'][:blocks, j, ix['completion']].sum() * cell['fleet'])
        row.update(
            status='checked',
            blocks_compared=int(blocks),
            epochs_compared=int(blocks * c['block_size']),
            services_executed=executed,
            services_expected=float(blocks * c['block_size'] * cell['fleet']),
            metrics_compared=len(COMMON),
            metrics_bit_identical=len(COMMON) - len(mismatches),
            mismatches=mismatches,
            first_mismatch_block_beyond_rounding=first_bad_block,
            traces_compared=trace_checked,
            traces_different_at=trace_bad,
            final_state_identical=final_state,
            reproduced=(
                first_bad_block is None
                and not trace_bad
                and row['first_deviation_epoch'] < 0
                and (final_state is None or all(final_state.values()))
            ),
        )
        out.append(row)
    return out


def compare_with_snapshots(c, flat, snapshots):
    """Closed-loop completion against the one-shot snapshot study on the same cells."""
    import csv

    if not snapshots or not (Path(snapshots) / 'summary_whole.csv').exists():
        return []
    path = Path(snapshots) / 'summary_whole.csv'
    snap = {}
    with path.open() as f:
        for r in csv.DictReader(f):
            if (
                int(r['fleet']) == c['fleet']
                and int(r['hubs']) == c['hubs']
                and r['behavior'] == 'one_step'
                and float(r['tau']) == 0
                and float(r['threshold']) == 0
                and r['metric'] == 'completion_rate'
            ):
                snap[(float(r['rho']), r['mechanism'], r['allocation'])] = (float(r['mean']), r['ci95_halfwidth'])
    closed = {(r['rho'], r['price'], r['allocation'], r['window']): r for r in flat if r['metric'] == 'completion'}
    out = []
    for rho in c['rhos']:
        for arm in c['arms']:
            price, rule = split_arm(arm)
            if (rho, price, rule) not in snap:
                continue
            s_mean, s_ci = snap[(rho, price, rule)]
            row = dict(rho=rho, price=price, allocation=rule, snapshot_completion=s_mean, snapshot_ci95=s_ci)
            for window in ('checkpoint_epochs', 'whole_run', 'terminal_100k'):
                r = closed[(rho, price, rule, window)]
                row[f'closed_loop_{window}'] = r['mean']
                row[f'closed_loop_{window}_ci95'] = r['ci95_halfwidth']
                row[f'gap_{window}'] = r['mean'] - s_mean
            out.append(row)
    return out


def plot(output, c, curve_rows):
    import matplotlib

    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    plt.rcParams.update({'font.size': 8, 'axes.spines.top': False, 'axes.spines.right': False, 'pdf.fonttype': 42})
    prices = list(dict.fromkeys(split_arm(a)[0] for a in c['arms']))
    rules = list(dict.fromkeys(split_arm(a)[1] for a in c['arms']))
    (output / 'figures').mkdir(exist_ok=True)
    for metric, ylabel, name in (
        ('completion', 'Jobs completed (%)', 'completion_over_time'),
        ('certified', 'Certified epochs (%)', 'certified_over_time'),
        ('false_certificates_cumulative', 'Certified epochs with a deviation', 'false_certificates'),
    ):
        for rho in c['rhos']:
            fig, axes = plt.subplots(
                1, len(rules), figsize=(3.1 * len(rules), 2.5), constrained_layout=True, squeeze=False
            )
            for ax, rule in zip(axes[0], rules):
                for price in prices:
                    picked = [
                        r for r in curve_rows if r['rho'] == rho and r['price'] == price and r['allocation'] == rule
                    ]
                    x = np.array([r['epoch'] for r in picked]) / 1e6
                    y = np.array([r[metric] for r in picked])
                    if metric != 'false_certificates_cumulative':
                        y = 100 * y
                    # Smooth block means over ten blocks for legibility; the CSV keeps the raw blocks.
                    k = max(1, min(10, len(y) // 10))
                    ys = (
                        np.convolve(y, np.ones(k) / k, mode='valid') if metric != 'false_certificates_cumulative' else y
                    )
                    xs = x[k - 1 :] if metric != 'false_certificates_cumulative' else x
                    ax.plot(xs, ys, color=COLORS[price], lw=1.1, label=LABELS[price])
                ax.set(title=f'{rule} allocation', xlabel='Epochs (millions)')
                if metric != 'false_certificates_cumulative':
                    ax.set_ylim(-2, 103)
            axes[0, 0].set_ylabel(ylabel)
            axes[0, -1].legend(fontsize=6.5, loc='lower right' if metric == 'completion' else 'best')
            fig.suptitle(f'Closed loop, N = {c["fleet"]}, rho = {rho:g}, means over {c["seeds"]} seeds', fontsize=8)
            for ext in ('pdf', 'png'):
                fig.savefig(output / 'figures' / f'{name}_rho{rho:g}.{ext}', dpi=200)
            plt.close(fig)


def report(output):
    output = Path(output)
    c = json.loads((output / 'config.json').read_text())
    manifest = json.loads((output / 'metadata.json').read_text())
    verify_source(output, manifest)
    loaded = {cell['name']: load_cell(output, cell) for cell in c['cells']}
    arms_rows, flat = summarize_arms(c, loaded)
    write_csv(output / 'summary_arms.csv', arms_rows)
    write_csv(output / 'summary_metrics.csv', flat)
    curve_rows = curves(c, loaded)
    write_csv(output / 'curves.csv', curve_rows)
    per_context, context_summary = services_per_context(c, loaded)
    write_csv(output / 'services_per_context.csv', per_context)
    write_csv(output / 'services_per_context_summary.csv', context_summary)
    repro = reproduction_check(c, loaded, c.get('reference'))
    (output / 'reproduction.json').write_text(json.dumps(repro, indent=2) + '\n')
    comparison = compare_with_snapshots(c, flat, c.get('snapshots'))
    write_csv(output / 'comparison_with_snapshots.csv', comparison)
    plot(output, c, curve_rows)
    timing = {
        cell['name']: dict(
            elapsed_session_seconds=loaded[cell['name']]['status'].get('elapsed_session_seconds'),
            epochs_per_second=loaded[cell['name']]['status'].get('epochs_per_second'),
        )
        for cell in c['cells']
    }
    (output / 'timing.json').write_text(json.dumps(timing, indent=2) + '\n')
    checked = [r for r in repro if r.get('status') == 'checked']
    print(
        f"Report written to {output}: {len(c['cells'])} cells, {len(c['arms'])} arms; reproduction checked on "
        f"{len(checked)} cells, reproduced on {sum(r['reproduced'] for r in checked)}",
        flush=True,
    )


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--epochs', type=int, default=10_000_000)
    p.add_argument('--seeds', type=int, default=5)
    p.add_argument('--base-seed', type=int, default=20264800)
    p.add_argument('--rhos', type=float, nargs='+', default=[0.0])
    p.add_argument('--fleet', type=int, default=6)
    p.add_argument('--hubs', type=int, default=8)
    p.add_argument('--prices', nargs='+', default=list(PRICES), choices=PRICES)
    p.add_argument('--rules', nargs='+', default=list(RULES), choices=RULES)
    p.add_argument('--block-size', type=int, default=10_000)
    p.add_argument(
        '--checkpoint-every',
        type=int,
        default=200_000,
        help='epochs per saved chunk; 200,000 matches the checkpoints of the reference scale run',
    )
    p.add_argument(
        '--reference',
        type=Path,
        default=None,
        help='completed run of run_manhattan_scale with the same cells: its inputs are checked against ours '
        'and the report verifies that the certified-dual priority arm reproduces its pooled-variance arm',
    )
    p.add_argument('--snapshots', type=Path, default=None, help=argparse.SUPPRESS)
    p.add_argument('--workers', type=int, default=5)
    p.add_argument('--resume', action='store_true')
    p.add_argument('--report-only', action='store_true')
    args = p.parse_args()
    output = args.output.resolve()
    if args.report_only:
        report(output)
        return
    arms = [arm_name(price, rule) for price in args.prices for rule in args.rules]
    if args.resume:
        config = json.loads((output / 'config.json').read_text())
        manifest = json.loads((output / 'metadata.json').read_text())
        check_source_unchanged(manifest)
        for filename, digest in manifest.get('input_sha256', {}).items():
            if sha(output / 'inputs' / filename) != digest:
                raise ValueError('Saved simulation inputs changed: ' + filename)
    else:
        if (
            args.epochs < 1
            or args.seeds < 1
            or args.workers < 1
            or args.block_size < 1
            or args.checkpoint_every % args.block_size
            or args.epochs % args.checkpoint_every
            or not 5 <= args.fleet <= 64
            or not 2 <= args.hubs <= 32
            or len(set(args.rhos)) != len(args.rhos)
            or not all(0 <= r <= 1 for r in args.rhos)
        ):
            p.error(
                'Invalid sizes: epochs a multiple of the checkpoint, checkpoint a multiple of the block, '
                'fleet in [5, 64], hubs in [2, 32], distinct rhos in [0, 1]'
            )
        output.mkdir(parents=True, exist_ok=False)
        (output / 'inputs').mkdir()
        hashes = snapshot_source(output)
        shutil.copy2(NETWORK, output / 'network.json')
        config = dict(
            epochs=args.epochs,
            seeds=args.seeds,
            base_seed=args.base_seed,
            rhos=args.rhos,
            fleet=args.fleet,
            hubs=args.hubs,
            block_size=args.block_size,
            checkpoint_every=args.checkpoint_every,
            alpha=0.05,
            eta=0.01,
            payment_budget=1.0,
            assignment_tolerance=1.0,
            network_file=str(output / 'network.json'),
            reference=str(args.reference.resolve()) if args.reference else None,
            snapshots=str(args.snapshots.resolve()) if args.snapshots else None,
            history='pooled_variance',
            prices=args.prices,
            rules=args.rules,
            arms=arms,
            metrics=METRICS,
            rows=ROWS,
            cells=None,
        )
        config['cells'] = build_cells(args, arms)
        atomic_json(output / 'config.json', config)
        manifest = dict(
            status='preparing',
            created_utc=datetime.now(timezone.utc).isoformat(),
            source_sha256=hashes,
            network_sha256=sha(output / 'network.json'),
            python=sys.version,
            versions={k: importlib.metadata.version(k) for k in ('numpy', 'scipy', 'numba', 'matplotlib')},
            protocol=(
                'Closed loop: at every epoch the platform posts the recommendation and a price menu; '
                'each driver replies once at its exact expected costs assuming the others comply '
                '(one_step, tau = 0, delta = 0); the allocation rule gives each requested job to one '
                'requester; only executed services update the pooled variance history and move drivers.'
            ),
            modelling_choices=(
                '(1) an unrequested job is lost for the epoch; (2) a driver that idles or loses '
                'a contest keeps its position; (3) drivers do not learn across epochs.'
            ),
            prices=(
                'dual_certified: adjusted dual prices; dual_base: kappa + psi without margin; direct_upper: '
                'U + eta on the recommended job; direct_upper_matched and uniform_matched: same offered '
                'total as the certified dual at that epoch.'
            ),
            allocation=(
                'priority: recommended driver wins its own job, else lowest index; uniform: uniform at '
                'random among requesters (one shared draw per job); cheapest: lowest platform estimate '
                'among requesters, ties to the recommended driver, then lowest index.'
            ),
            false_certificate=(
                'false_certificate: epoch passing both certificate tests with at least one driver '
                'not requesting its recommendation; kernel_false_certificate: the kernel\'s '
                'definition (recommended loss above the tolerance or surplus below eta).'
            ),
            rng=(
                'Counter-based SplitMix64 of the kernel; purposes 1 (demand) and 100+driver (routes) as in '
                'the kernel, 5 (uniform allocation) and 6 (reply ties) shared by all arms.'
            ),
        )
    atomic_json(output / 'metadata.json', manifest)
    pending = [cell for cell in config['cells'] if not (output / 'inputs' / f"{cell['name']}.npz").exists()]
    if pending:
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            for future in as_completed([pool.submit(prepare_cell, str(output), config, cell) for cell in pending]):
                print('Prepared ' + future.result(), flush=True)
    manifest['input_sha256'] = {p.name: sha(p) for p in sorted((output / 'inputs').glob('*.npz'))}
    manifest['status'] = 'running'
    manifest['started_utc'] = datetime.now(timezone.utc).isoformat()
    atomic_json(output / 'metadata.json', manifest)
    started = time.perf_counter()
    try:
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            futures = [pool.submit(run_cell, str(output), cell) for cell in config['cells']]
            for future in as_completed(futures):
                print('Completed ' + future.result(), flush=True)
        manifest['wall_clock_seconds'] = time.perf_counter() - started
        report(output)
        manifest.update(status='complete', completed_utc=datetime.now(timezone.utc).isoformat())
    except BaseException as exc:
        manifest.update(status='interrupted' if isinstance(exc, KeyboardInterrupt) else 'failed', error=repr(exc))
        raise
    finally:
        manifest['elapsed_seconds'] = time.perf_counter() - started
        atomic_json(output / 'metadata.json', manifest)


if __name__ == '__main__':
    main()
