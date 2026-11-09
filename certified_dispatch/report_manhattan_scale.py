"""Audit a completed Manhattan scale run, export seed-level data and figures.

No simulation happens here. Saved decision traces are rechecked with the independent
platform code (`platform.certificate`, `platform.evaluate_payments`, SciPy's assignment
solver), block statistics are reconciled with the checkpoints, the decomposition of B
is verified block by block, and guaranteed arms are checked for coverage failures and
false certificates. Thresholds are reported in epochs and in completed services.
"""
import hashlib
import json
from pathlib import Path
import numpy as np
from scipy.optimize import linear_sum_assignment
from .manhattan_kernel import ARMS, METRICS, guaranteed
from .platform import certificate, evaluate_payments
from .common import write_csv, interval, verify_source

CURVE_METRICS = ('certified', 'bound', 'shift', 'mean_width', 'assignment_loss', 'false_certificate', 'covered')
GROUP = ('sweep', 'fleet', 'hubs', 'rho')


def windows(cell, block_size):
    """Terminal windows in blocks: the final one percent of epochs, the final 100k epochs, the run."""
    blocks = cell['epochs'] // block_size
    return (
        ('terminal_1pct', max(1, blocks // 100)),
        ('terminal_100k', min(blocks, max(1, 100_000 // block_size))),
        ('whole_run', blocks),
    )


def audit_cell(output, c, cell):
    ix = {name: j for j, name in enumerate(METRICS)}
    n = cell['fleet']
    directory = output / cell['name']
    status = json.loads((directory / 'status.json').read_text())
    if status['status'] != 'complete':
        raise ValueError('Not complete: ' + cell['name'])
    checkpoint = dict(np.load(directory / 'checkpoint.npz'))
    assert int(checkpoint['epoch']) == cell['epochs']
    inputs = dict(np.load(output / 'inputs' / f"{cell['name']}.npz"))
    assert inputs['true_contexts'].shape == (n, cell['hubs'] ** 3)
    spread = inputs['true_contexts'].max(axis=0) - inputs['true_contexts'].min(axis=0)
    public = cell['rho'] * (inputs['support_hi'] - inputs['support_lo'])
    assert np.all(spread <= public + 1e-8)
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
            payments, phi, psi = tr[4, 1], tr[4, 2], tr[4, 3]
            assert sorted(sigma.tolist()) == list(range(n))
            assert np.isclose(C[np.arange(n), sigma].sum(), C[linear_sum_assignment(C)].sum(), atol=1e-8)
            assert (C - phi[:, None] - psi).min() >= -1e-7
            assert np.isclose(phi.sum() + psi.sum(), C[np.arange(n), sigma].sum(), atol=1e-7)
            assert np.all(L <= C + 1e-9) and np.all(C <= U + 1e-9)
            cert = certificate(sigma, payments, L, U[np.arange(n), sigma])
            ev = evaluate_payments(D, sigma, payments)
            loss = D[np.arange(n), sigma].mean() - D[linear_sum_assignment(D)].mean()
            assert cert['minimum_surplus_bound'] >= c['eta'] - 1e-7
            if np.all(L <= D + 1e-8) and np.all(D <= U + 1e-8):
                assert loss <= cert['control_bound'] + 1e-7
                assert ev['menu_regret'] <= cert['menu_bound'] + 1e-7
                assert ev['minimum_surplus'] >= c['eta'] - 1e-7
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
        coverage = bool(checkpoint['coverage_min'][a] >= -1e-8)
        cumulative = block[:, a].cumsum(axis=0)
        for b, values in enumerate(series):
            epoch = (b + 1) * c['block_size']
            curves.append(
                dict(
                    **base,
                    arm=arm,
                    epoch=epoch,
                    services=epoch * n,
                    **{m: float(values[ix[m]]) for m in CURVE_METRICS},
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
                first_90_services='' if first_epoch is None else first_epoch * n,
                sustained_90_epoch='' if sustained_epoch is None else sustained_epoch,
                sustained_90_services='' if sustained_epoch is None else sustained_epoch * n,
                censored=int(sustained_epoch is None),
            )
        )
        audits.append(
            dict(
                **base,
                arm=arm,
                theoretical_guarantee=int(guaranteed(arm, cell['rho'])),
                all_history_updates_covered=int(coverage),
                minimum_coverage_margin=float(checkpoint['coverage_min'][a]),
                false_certificates=int(round(checkpoint['cumulative'][a, ix['false_certificate']])),
                uncovered_decisions=int(round(cell['epochs'] - checkpoint['cumulative'][a, ix['covered']])),
                observed_services=n * cell['epochs'],
                epochs=cell['epochs'],
                independently_checked_traces=traces_checked // len(ARMS),
            )
        )
    return curves, terminal, thresholds, audits


def summarize(rows, keys, metrics):
    groups = {}
    for r in rows:
        groups.setdefault(tuple(r[k] for k in keys), []).append(r)
    out = []
    for key, group in sorted(groups.items(), key=lambda kv: tuple(str(x) for x in kv[0])):
        for metric in metrics:
            values = np.array([float(r[metric]) for r in group if r[metric] != ''])
            out.append(
                dict(
                    zip(keys, key),
                    metric=metric,
                    mean=values.mean() if len(values) else '',
                    ci95_halfwidth=interval(values) if len(values) > 1 else '',
                    seeds=len(group),
                    defined=len(values),
                )
            )
    return out


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
    write_csv(output / 'summary.csv', summarize(terminal, (*GROUP, 'window', 'arm'), METRICS))
    write_csv(
        output / 'threshold_summary.csv',
        summarize(
            thresholds,
            (*GROUP, 'arm'),
            ('first_90_epoch', 'first_90_services', 'sustained_90_epoch', 'sustained_90_services', 'censored'),
        ),
    )
    (output / 'figures').mkdir(exist_ok=True)
    plot(output, c, curves, thresholds)
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


FLEET_COLORS = {6: '#0072B2', 12: '#009E73', 24: '#E69F00', 48: '#D55E00', 96: '#CC79A7'}
HUB_COLORS = {8: '#0072B2', 12: '#009E73', 16: '#E69F00', 24: '#D55E00', 32: '#CC79A7'}


def mean_curve(curves, select, metric):
    """Mean and 95% interval over seeds of a block series; returns x (epochs), x (services), mean, half-width."""
    rows = [r for r in curves if select(r)]
    seeds = sorted({r['seed'] for r in rows})
    values = np.array([[float(r[metric]) for r in rows if r['seed'] == s] for s in seeds])
    first = [r for r in rows if r['seed'] == seeds[0]]
    epochs = np.array([r['epoch'] for r in first], float)
    services = np.array([r['services'] for r in first], float)
    mean = values.mean(axis=0)
    half = np.array([interval(values[:, j]) for j in range(values.shape[1])]) if len(seeds) > 1 else np.zeros_like(mean)
    return epochs, services, mean, half


def plot(output, c, curves, thresholds):
    import matplotlib

    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    plt.rcParams.update({'font.size': 8, 'axes.spines.top': False, 'axes.spines.right': False, 'pdf.fonttype': 42})
    fleets = c['fleets']
    rhos = c['rhos']
    # Figure 1: fleet sweep, pooled and driver-specific variance arms, against epochs and services.
    fig, axes = plt.subplots(2, len(rhos), figsize=(3.4 * len(rhos), 5.0), constrained_layout=True, squeeze=False)
    for j, rho in enumerate(rhos):
        for fleet in fleets:
            color = FLEET_COLORS.get(fleet, '#555555')
            for arm, ls in (('pooled_variance', '-'), ('private_variance', '--')):
                sel = lambda r, f=fleet, a=arm, rr=rho: (
                    r['sweep'] == 'fleet'
                    and r['fleet'] == f
                    and r['rho'] == rr
                    and r['hubs'] == c['hubs']
                    and r['arm'] == a
                )
                if not any(sel(r) for r in curves):
                    continue
                ep, sv, mean, half = mean_curve(curves, sel, 'certified')
                label = f'N = {fleet}' if arm == 'pooled_variance' else None
                axes[0, j].plot(ep / 1e6, mean, color=color, ls=ls, lw=1.1, label=label)
                axes[1, j].plot(sv / 1e6, mean, color=color, ls=ls, lw=1.1, label=label)
                for ax, x in ((axes[0, j], ep / 1e6), (axes[1, j], sv / 1e6)):
                    ax.fill_between(x, mean - half, mean + half, color=color, alpha=0.10, lw=0)
        axes[0, j].set(title=f'rho = {rho:g}', xlabel='Dispatch epochs (millions)', ylim=(-0.02, 1.02))
        axes[1, j].set(xlabel='Completed services (millions)', ylim=(-0.02, 1.02))
    axes[0, 0].set_ylabel('Certified fraction per block')
    axes[1, 0].set_ylabel('Certified fraction per block')
    axes[0, 0].legend(title='pooled solid, driver-specific dashed', fontsize=6.5, title_fontsize=6.5, loc='lower right')
    for ext in ('pdf', 'png'):
        fig.savefig(output / 'figures' / f'certification_by_fleet.{ext}', dpi=200)
    plt.close(fig)
    # Figure 2: hub sweep at the first fleet and rho.
    hubs_all = [c['hubs'], *c['hub_sweep']]
    fig, axes = plt.subplots(1, 2, figsize=(6.8, 2.7), constrained_layout=True)
    for hubs in hubs_all:
        color = HUB_COLORS.get(hubs, '#555555')
        for arm, ls, ax in (
            ('pooled_variance', '-', axes[0]),
            ('pooled_support', ':', axes[0]),
            ('private_variance', '-', axes[1]),
        ):
            sel = lambda r, h=hubs, a=arm: (
                r['fleet'] == fleets[0] and r['rho'] == rhos[0] and r['hubs'] == h and r['arm'] == a
            )
            if not any(sel(r) for r in curves):
                continue
            ep, sv, mean, half = mean_curve(curves, sel, 'certified')
            label = f'{hubs} hubs, |Q| = {hubs*hubs*(hubs-1)}' if arm != 'pooled_support' else None
            ax.plot(ep / 1e6, mean, color=color, ls=ls, lw=1.1, label=label)
            ax.fill_between(ep / 1e6, mean - half, mean + half, color=color, alpha=0.10, lw=0)
    axes[0].set(
        title='Pooled (variance solid, support dotted)',
        xlabel='Dispatch epochs (millions)',
        ylabel='Certified fraction per block',
        ylim=(-0.02, 1.02),
    )
    axes[1].set(title='Driver-specific, variance radius', xlabel='Dispatch epochs (millions)', ylim=(-0.02, 1.02))
    axes[0].legend(fontsize=6.5, loc='lower right')
    for ext in ('pdf', 'png'):
        fig.savefig(output / 'figures' / f'certification_by_hubs.{ext}', dpi=200)
    plt.close(fig)
    # Figure 3: sustained-90% threshold against fleet size, with a 1/N reference.
    fig, ax = plt.subplots(figsize=(3.4, 2.7), constrained_layout=True)
    for rho, marker in zip(rhos, 'os^d'):
        xs, ys, errs = [], [], []
        for fleet in fleets:
            vals = [
                float(r['sustained_90_epoch'])
                for r in thresholds
                if r['sweep'] == 'fleet'
                and r['fleet'] == fleet
                and r['rho'] == rho
                and r['hubs'] == c['hubs']
                and r['arm'] == 'pooled_variance'
                and r['sustained_90_epoch'] != ''
            ]
            if len(vals):
                xs.append(fleet)
                ys.append(np.mean(vals) / 1e6)
                errs.append(interval(vals) / 1e6 if len(vals) > 1 else 0.0)
        if xs:
            ax.errorbar(xs, ys, yerr=errs, marker=marker, ls='none', ms=4, capsize=2, label=f'rho = {rho:g}')
            ref = np.array(xs, float)
            ax.plot(ref, ys[0] * xs[0] / ref, color='grey', ls=':', lw=0.9)
    ax.set(
        xscale='log',
        yscale='log',
        xlabel='Fleet size N',
        ylabel='Sustained 90% certification (million epochs)',
        title='Pooled, variance radius; dotted: 1/N',
    )
    ax.set_xticks(fleets, [str(f) for f in fleets])
    if ax.get_legend_handles_labels()[0]:
        ax.legend(fontsize=6.5)
    for ext in ('pdf', 'png'):
        fig.savefig(output / 'figures' / f'threshold_by_fleet.{ext}', dpi=200)
    plt.close(fig)
