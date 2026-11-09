"""Audit a completed exploration hub sweep, export seed-level data and figures.

No simulation happens here. Saved decision traces are rechecked with the independent
platform code and SciPy's assignment solver, including the width identity of explored
epochs (the certificate of sigma_L equals the mean width of its pairs and its shift is
within the gate and the maximum width plus eta); block statistics are reconciled with
the checkpoints; the decomposition of B is verified; guaranteed arms are checked for
coverage failures and false certificates; explored loss is checked against the width
bound on the coverage event.
"""
import hashlib
import json
from pathlib import Path
import numpy as np
from scipy.optimize import linear_sum_assignment
from .optimistic_kernel import ARMS, METRICS
from .platform import certificate, evaluate_payments
from .common import write_csv, interval, verify_source
from .report_manhattan_scale import summarize, windows
from .run_manhattan_explore import guaranteed

CURVE_METRICS = (
    'certified',
    'explored',
    'blocked',
    'bound',
    'assignment_loss',
    'executed_loss',
    'selected_upper_gap',
    'lower_benchmark_gap',
    'shift',
    'executed_shift',
    'mean_width',
    'unobserved_fraction',
    'false_certificate',
    'covered',
    'total_payment',
)
GROUP = ('fleet', 'hubs', 'rho')
OPTIMISTIC_ARMS = (1, 4, 6)


def audit_cell(output, c, cell):
    ix = {name: j for j, name in enumerate(METRICS)}
    n = cell['fleet']
    ids = np.arange(n)
    directory = output / cell['name']
    status = json.loads((directory / 'status.json').read_text())
    if status['status'] != 'complete':
        raise ValueError('Not complete: ' + cell['name'])
    checkpoint = dict(np.load(directory / 'checkpoint.npz'))
    assert int(checkpoint['epoch']) == cell['epochs']
    inputs = dict(np.load(output / 'inputs' / f"{cell['name']}.npz"))
    assert inputs['true_contexts'].shape == (n, cell['hubs'] ** 3)
    blocks, end, traces_checked = [], 0, 0
    for file in sorted(directory.glob('blocks_*.npz')):
        data = dict(np.load(file))
        if int(data['stop']) > cell['epochs']:
            continue
        assert int(data['start']) == end
        end = int(data['stop'])
        assert data['stats'].shape[0] * c['block_size'] == end - int(data['start'])
        blocks.append(data['stats'])
        for arm, tr in enumerate(data['trace']):
            C, L, U, D = tr[:4]
            sigma = tr[4, 0].astype(int)
            executed = tr[4, 1].astype(int)
            payments, phi, psi = tr[4, 2], tr[4, 3], tr[4, 4]
            assert sorted(executed.tolist()) == list(range(n))
            assert np.isclose(C[ids, sigma].sum(), C[linear_sum_assignment(C)].sum(), atol=1e-8)
            assert (C - phi[:, None] - psi).min() >= -1e-7
            assert np.isclose(phi.sum() + psi.sum(), C[ids, sigma].sum(), atol=1e-7)
            cert = certificate(executed, payments, L, U[ids, executed])
            ev = evaluate_payments(D, executed, payments)
            loss = D[ids, executed].mean() - D[linear_sum_assignment(D)].mean()
            if np.all(L <= D + 1e-8) and np.all(D <= U + 1e-8):
                assert loss <= cert['control_bound'] + 1e-7
                assert ev['menu_regret'] <= cert['menu_bound'] + 1e-7
                assert ev['minimum_surplus'] >= c['eta'] - 1e-7
            if bool(tr[5, 2, 0]) and arm not in (2, 7):
                width = (U - L)[ids, executed]
                assert np.isclose(cert['control_bound'], width.mean(), atol=1e-7)
                cap = c['exploration_budget'] if arm >= 6 else c['payment_budget']
                assert tr[5, 2, 1] <= min(cap, width.max() + c['eta']) + 1e-7
            traces_checked += 1
    assert end == cell['epochs']
    block = np.concatenate(blocks)
    np.testing.assert_allclose(block.sum(axis=0), checkpoint['cumulative'], rtol=2e-8, atol=1e-5)
    assert np.all(checkpoint['counts'].sum(axis=(1, 2)) == n * cell['epochs'])
    base = {k: cell[k] for k in GROUP}
    base['seed'] = cell['seed']
    curves, terminal, thresholds, audits = [], [], [], []
    for a, arm in enumerate(ARMS):
        series = block[:, a] / c['block_size']
        np.testing.assert_allclose(
            series[:, ix['bound']],
            series[:, ix['assignment_loss']]
            + series[:, ix['selected_upper_gap']]
            + series[:, ix['lower_benchmark_gap']],
            atol=1e-7,
        )
        cumulative = block[:, a].cumsum(axis=0)
        coverage = bool(checkpoint['coverage_min'][a] >= -1e-8)
        if a in OPTIMISTIC_ARMS and coverage:
            assert np.all(cumulative[:, ix['explored_loss_total']] <= cumulative[:, ix['explored_width_total']] + 1e-5)
        for b, values in enumerate(series):
            epoch = (b + 1) * c['block_size']
            curves.append(
                dict(
                    **base,
                    arm=arm,
                    epoch=epoch,
                    services=epoch * n,
                    **{m: float(values[ix[m]]) for m in CURVE_METRICS},
                    cumulative_explored_fraction=cumulative[b, ix['explored']] / epoch,
                    cumulative_explored_loss=cumulative[b, ix['explored_loss_total']],
                    cumulative_explored_width=cumulative[b, ix['explored_width_total']],
                    cumulative_certified_fraction=cumulative[b, ix['certified']] / epoch,
                )
            )
        for window, size in windows(cell, c['block_size']):
            values = block[-size:, a].sum(axis=0) / (size * c['block_size'])
            terminal.append(
                dict(
                    **base,
                    arm=arm,
                    window=window,
                    window_epochs=size * c['block_size'],
                    **dict(zip(METRICS, values.tolist())),
                    conditional_explored_shift=(
                        values[ix['explored_shift']] / values[ix['explored']] if values[ix['explored']] > 0 else ''
                    ),
                    conditional_explored_loss=(
                        values[ix['explored_loss_total']] / n / values[ix['explored']]
                        if values[ix['explored']] > 0
                        else ''
                    ),
                    all_history_updates_covered=int(coverage),
                )
            )
        pass_blocks = series[:, ix['certified']] >= 0.9
        first = np.flatnonzero(pass_blocks)
        bad = np.flatnonzero(~pass_blocks)
        sustained = int(bad[-1]) + 1 if len(bad) else 0
        first_epoch = (int(first[0]) + 1) * c['block_size'] if len(first) else None
        sustained_epoch = (sustained + 1) * c['block_size'] if sustained < len(pass_blocks) else None
        thresholds.append(
            dict(
                **base,
                arm=arm,
                epochs=cell['epochs'],
                first_90_epoch='' if first_epoch is None else first_epoch,
                sustained_90_epoch='' if sustained_epoch is None else sustained_epoch,
                censored=int(sustained_epoch is None),
            )
        )
        audits.append(
            dict(
                **base,
                arm=arm,
                theoretical_guarantee=int(guaranteed(a, cell['rho'])),
                all_history_updates_covered=int(coverage),
                minimum_coverage_margin=float(checkpoint['coverage_min'][a]),
                false_certificates=int(round(checkpoint['cumulative'][a, ix['false_certificate']])),
                uncovered_decisions=int(round(cell['epochs'] - checkpoint['cumulative'][a, ix['covered']])),
                explored_epochs=int(round(checkpoint['cumulative'][a, ix['explored']])),
                blocked_epochs=int(round(checkpoint['cumulative'][a, ix['blocked']])),
                observed_services=n * cell['epochs'],
                epochs=cell['epochs'],
                independently_checked_traces=traces_checked // len(ARMS),
            )
        )
    return curves, terminal, thresholds, audits


def report(output):
    output = Path(output)
    c = json.loads((output / 'config.json').read_text())
    assert tuple(c['arms']) == ARMS and tuple(c['metrics']) == METRICS
    metadata = json.loads((output / 'metadata.json').read_text())
    verify_source(output, metadata)
    assert hashlib.sha256((output / 'network.json').read_bytes()).hexdigest() == metadata['network_sha256']
    for filename, digest in metadata['input_sha256'].items():
        assert hashlib.sha256((output / 'inputs' / filename).read_bytes()).hexdigest() == digest
    curves, terminal, thresholds, audits = [], [], [], []
    for cell in c['cells']:
        cu, te, th, au = audit_cell(output, c, cell)
        curves += cu
        terminal += te
        thresholds += th
        audits += au
    for row in audits:
        if row['theoretical_guarantee']:
            assert (
                row['all_history_updates_covered'] == 1
                and row['false_certificates'] == 0
                and row['uncovered_decisions'] == 0
            ), row
    write_csv(output / 'figure_data.csv', curves)
    write_csv(output / 'terminal_seed_summary.csv', terminal)
    write_csv(output / 'thresholds.csv', thresholds)
    write_csv(output / 'audit.csv', audits)
    write_csv(
        output / 'summary.csv',
        summarize(
            terminal, (*GROUP, 'window', 'arm'), (*METRICS, 'conditional_explored_shift', 'conditional_explored_loss')
        ),
    )
    write_csv(
        output / 'threshold_summary.csv',
        summarize(thresholds, (*GROUP, 'arm'), ('first_90_epoch', 'sustained_90_epoch', 'censored')),
    )
    (output / 'figures').mkdir(exist_ok=True)
    plot(output, c, curves)
    guaranteed_rows = [r for r in audits if r['theoretical_guarantee']]
    heuristic_rows = [r for r in audits if not r['theoretical_guarantee']]
    (output / 'audit.json').write_text(
        json.dumps(
            dict(
                status='passed',
                cells=len(c['cells']),
                independent_seeds=c['seeds'],
                arms=list(ARMS),
                guaranteed_rows=len(guaranteed_rows),
                guaranteed_coverage_failures=sum(1 - r['all_history_updates_covered'] for r in guaranteed_rows),
                guaranteed_false_certificates=sum(r['false_certificates'] for r in guaranteed_rows),
                heuristic_rows=len(heuristic_rows),
                heuristic_coverage_failures=sum(1 - r['all_history_updates_covered'] for r in heuristic_rows),
                heuristic_false_certificates=sum(r['false_certificates'] for r in heuristic_rows),
                note='Coverage observations do not establish the nominal confidence level.',
            ),
            indent=2,
        )
        + '\n'
    )
    print('Audit, summaries and figures saved to ' + str(output), flush=True)


COLORS = {
    'pooled_none': '#0072B2',
    'pooled_optimistic': '#009E73',
    'pooled_optimistic_relaxed': '#D55E00',
    'pooled_random_relaxed': '#CC79A7',
    'pooled_random': '#E69F00',
    'private_none': '#56B4E9',
    'private_optimistic': '#8B5A2B',
    'naive_optimistic': '#999999',
}
LABELS = {
    'pooled_none': 'Pooled, no exploration',
    'pooled_optimistic': 'Pooled, optimistic (gate 1)',
    'pooled_optimistic_relaxed': 'Pooled, optimistic (ungated)',
    'pooled_random_relaxed': 'Pooled, random matching',
    'pooled_random': 'Pooled, random (gate 1)',
    'private_none': 'Driver-specific, no exploration',
    'private_optimistic': 'Driver-specific, optimistic (gate 1)',
    'naive_optimistic': 'Naive, optimistic',
}


def mean_curve(curves, select, metric):
    rows = [r for r in curves if select(r)]
    seeds = sorted({r['seed'] for r in rows})
    values = np.array([[float(r[metric]) for r in rows if r['seed'] == s] for s in seeds])
    first = [r for r in rows if r['seed'] == seeds[0]]
    epochs = np.array([r['epoch'] for r in first], float)
    mean = values.mean(axis=0)
    half = np.array([interval(values[:, j]) for j in range(values.shape[1])]) if len(seeds) > 1 else np.zeros_like(mean)
    return epochs, mean, half


def plot(output, c, curves):
    import matplotlib

    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    plt.rcParams.update({'font.size': 8, 'axes.spines.top': False, 'axes.spines.right': False, 'pdf.fonttype': 42})
    hubs_all = c['hubs']
    shown = (
        'pooled_none',
        'pooled_optimistic',
        'pooled_optimistic_relaxed',
        'pooled_random_relaxed',
        'private_none',
        'private_optimistic',
    )
    panels = (
        ('certified', 'Certified fraction per block', (-0.02, 1.02)),
        ('cumulative_explored_fraction', 'Explored fraction (cumulative)', (-0.02, 1.02)),
        ('lower_benchmark_gap', 'Lower-benchmark gap (min/driver)', None),
        ('unobserved_fraction', 'Unobserved pairs (fraction)', None),
    )
    fig, axes = plt.subplots(
        len(panels),
        len(hubs_all),
        figsize=(3.4 * len(hubs_all), 2.6 * len(panels)),
        constrained_layout=True,
        squeeze=False,
    )
    for j, hubs in enumerate(hubs_all):
        for i, (metric, ylabel, ylim) in enumerate(panels):
            ax = axes[i, j]
            for arm in shown:
                sel = lambda r, h=hubs, a=arm: r['hubs'] == h and r['arm'] == a
                if not any(sel(r) for r in curves):
                    continue
                ep, mean, half = mean_curve(curves, sel, metric)
                ax.plot(
                    ep / 1e6,
                    mean,
                    color=COLORS[arm],
                    lw=1.1,
                    label=LABELS[arm],
                    ls='--' if arm.startswith('private') else '-',
                )
                ax.fill_between(ep / 1e6, mean - half, mean + half, color=COLORS[arm], alpha=0.10, lw=0)
            if ylim:
                ax.set_ylim(*ylim)
            if metric in ('lower_benchmark_gap',):
                ax.set_yscale('log')
            if i == 0:
                ax.set_title(f"{hubs} hubs, |Q| = {hubs*hubs*(hubs-1):,}, N = {c['fleet']}")
            if j == 0:
                ax.set_ylabel(ylabel)
            if i == len(panels) - 1:
                ax.set_xlabel('Dispatch epochs (millions)')
    handles, labels_ = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels_, loc='outside lower center', ncol=3, fontsize=7)
    for ext in ('pdf', 'png'):
        fig.savefig(output / 'figures' / f'exploration_by_hubs.{ext}', dpi=200)
    plt.close(fig)
    # Explored loss against its width bound, relaxed arm.
    fig, axes = plt.subplots(
        1, len(hubs_all), figsize=(3.4 * len(hubs_all), 2.7), constrained_layout=True, squeeze=False
    )
    for j, hubs in enumerate(hubs_all):
        ax = axes[0, j]
        for arm, ls in (('pooled_optimistic_relaxed', '-'), ('pooled_optimistic', ':')):
            sel = lambda r, h=hubs, a=arm: r['hubs'] == h and r['arm'] == a
            if not any(sel(r) for r in curves):
                continue
            ep, loss, _ = mean_curve(curves, sel, 'cumulative_explored_loss')
            _, width, _ = mean_curve(curves, sel, 'cumulative_explored_width')
            ax.plot(
                ep / 1e6, loss / c['fleet'], color=COLORS[arm], ls=ls, lw=1.1, label=LABELS[arm] + ': explored loss'
            )
            ax.plot(
                ep / 1e6,
                width / c['fleet'],
                color=COLORS[arm],
                ls=ls,
                lw=0.7,
                alpha=0.6,
                label=LABELS[arm] + ': width bound',
            )
        ax.set(title=f'{hubs} hubs', xlabel='Dispatch epochs (millions)', yscale='log')
        if j == 0:
            ax.set_ylabel('Cumulative explored loss and bound (min per driver)')
    axes[0, 0].legend(fontsize=6)
    for ext in ('pdf', 'png'):
        fig.savefig(output / 'figures' / f'explored_loss_bound.{ext}', dpi=200)
    plt.close(fig)
