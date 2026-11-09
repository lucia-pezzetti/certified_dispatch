"""Export the data and the PGFPlots sources of the three figures of Section IV from completed runs.

python -m certified_dispatch.build_figures --pooling results/pooling --exploration results/exploration \
    --strategic results/strategic --output figures

Figure 1(a): certification rate of the six-driver run in blocks of 10^5 epochs, pooled variance-adaptive,
driver-specific variance-adaptive and pooled support arms. Figure 1(b): certification rate of the
twenty-driver run in blocks of 10^4 epochs, with and without exploration. Figure 2: assignment loss and
extra payment per exploratory dispatch of the uncapped optimistic and random rules. Figure 3: jobs completed
by allocation rule and certification under uniform allocation in the strategic run. Curves are means over
seeds with two-sided 95% Student-t intervals across seed means; nothing is smoothed or subsampled.

The output directory receives `data/*.csv` (the exact plotted values), `figure{1,2,3}.tikz` (the layouts of the
paper, filled from `templates/`, to \\input in a document that loads `figure_style.tex`), standalone
`figure{1,2,3}.tex` wrappers, and `provenance.json` with the SHA-256 of every input and output file.
"""
import argparse
from collections import defaultdict
import csv
import hashlib
import json
import math
from pathlib import Path
import numpy as np
from scipy.stats import t

from .common import write_csv

PANEL_A_BLOCK = 100_000
PANEL_A_ARMS = ('pooled_variance', 'private_variance', 'pooled_support')
PANEL_B_ARMS = ('pooled_none', 'pooled_optimistic', 'pooled_optimistic_relaxed', 'pooled_random_relaxed')
PRICES = ('dual_certified', 'dual_base', 'direct_upper', 'direct_upper_matched', 'uniform_matched')
RULES = ('priority', 'uniform', 'cheapest')

TEMPLATES = Path(__file__).resolve().parent / 'templates'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path, sources):
    sources[str(path)] = sha(path)
    with Path(path).open() as handle:
        return list(csv.DictReader(handle))


def half_width(values):
    values = np.asarray(values, float)
    return float(t.ppf(0.975, len(values) - 1) * values.std(ddof=1) / math.sqrt(len(values)))


def curve_rows(x, y, h):
    return [dict(x=a, y=b, lo=max(0, b - c), hi=min(100, b + c), ci=c) for a, b, c in zip(x, y, h)]


def panel_a(pooling, sources, fleet=6, hubs=8, rho=0.0):
    """Six-driver certification rate in blocks of 10^5 epochs: blocks of the run are averaged within each seed."""
    config = json.loads((pooling / 'config.json').read_text())
    per_block = PANEL_A_BLOCK // config['block_size']
    assert PANEL_A_BLOCK % config['block_size'] == 0
    rows = read(pooling / 'figure_data.csv', sources)
    out = {}
    for arm in PANEL_A_ARMS:
        series = defaultdict(list)
        for r in rows:
            if r['arm'] == arm and int(r['fleet']) == fleet and int(r['hubs']) == hubs and float(r['rho']) == rho:
                series[int(r['seed'])].append((int(r['epoch']), float(r['certified'])))
        seeds = sorted(series)
        values = np.array([[v for _, v in sorted(series[s])] for s in seeds])
        blocks = values.shape[1] // per_block
        coarse = 100 * values[:, : blocks * per_block].reshape(len(seeds), blocks, per_block).mean(axis=2)
        epochs = np.arange(1, blocks + 1) * PANEL_A_BLOCK
        mean = coarse.mean(axis=0)
        half = np.array([half_width(coarse[:, j]) for j in range(blocks)])
        out[arm] = curve_rows([e / 1e6 for e in epochs], mean.tolist(), half.tolist())
    return out


def panel_b(exploration, sources, fleet=20, hubs=8, rho=0.0):
    """Twenty-driver certification rate per block of the run, every seed."""
    rows = read(exploration / 'figure_data.csv', sources)
    out = {}
    for arm in PANEL_B_ARMS:
        group = defaultdict(list)
        for r in rows:
            if r['arm'] == arm and int(r['fleet']) == fleet and int(r['hubs']) == hubs and float(r['rho']) == rho:
                group[int(r['epoch'])].append(float(r['certified']) * 100)
        epochs = sorted(group)
        seeds = {len(v) for v in group.values()}
        assert len(seeds) == 1, 'every epoch needs the same number of seeds'
        mean = [float(np.mean(group[e])) for e in epochs]
        half = [half_width(group[e]) for e in epochs]
        out[arm] = curve_rows([e / 1e6 for e in epochs], mean, half)
    return out


def exploration_costs(exploration, sources, fleet=20, hubs=8, rho=0.0):
    """Assignment loss and extra payment per exploratory dispatch, whole run, per seed then across seeds."""
    seeds = read(exploration / 'terminal_seed_summary.csv', sources)
    stats = []
    for arm in ('pooled_optimistic_relaxed', 'pooled_random_relaxed'):
        rows = [
            r
            for r in seeds
            if r['arm'] == arm
            and r['window'] == 'whole_run'
            and int(r['fleet']) == fleet
            and int(r['hubs']) == hubs
            and float(r['rho']) == rho
        ]
        for metric in ('conditional_explored_loss', 'conditional_explored_shift'):
            values = np.array([float(r[metric]) for r in rows])
            stats.append(dict(arm=arm, metric=metric, mean=float(values.mean()), ci=half_width(values)))
    return stats


def strategic(strategic_run, sources, rho='0.0'):
    """Whole-run completion by allocation rule with t intervals; certification per block under uniform allocation."""
    arms = read(strategic_run / 'summary_arms.csv', sources)
    curves = read(strategic_run / 'curves.csv', sources)
    completion, certification = {}, {}
    for price in PRICES:
        rows = [
            next(
                r
                for r in arms
                if r['price'] == price and r['allocation'] == rule and r['rho'] == rho and r['window'] == 'whole_run'
            )
            for rule in RULES
        ]
        completion[price] = [
            dict(x=i, y=100 * float(r['completion']), ci=100 * float(r['completion_ci95'])) for i, r in enumerate(rows)
        ]
        rows = sorted(
            [r for r in curves if r['price'] == price and r['allocation'] == 'uniform' and r['rho'] == rho],
            key=lambda r: int(r['epoch']),
        )
        certification[price] = [dict(x=int(r['epoch']) / 1e6, y=100 * float(r['certified'])) for r in rows]
    return completion, certification


def figure2(stats):
    """Fill the bar coordinates, interval ends and labels of the Figure 2 template."""
    values = {}
    for arm, prefix in (('pooled_optimistic_relaxed', 'opt'), ('pooled_random_relaxed', 'rand')):
        for metric, name in (('conditional_explored_loss', 'loss'), ('conditional_explored_shift', 'shift')):
            r = next(r for r in stats if r['arm'] == arm and r['metric'] == metric)
            values[f'{name}_{prefix}'] = r['mean']
            values[f'{name}_{prefix}_lo'] = r['mean'] - r['ci']
            values[f'{name}_{prefix}_hi'] = r['mean'] + r['ci']
    return (TEMPLATES / 'figure2.tikz').read_text().format(**values)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--pooling', type=Path, required=True, help='completed run of run_manhattan_scale')
    p.add_argument('--exploration', type=Path, required=True, help='completed run of run_manhattan_explore')
    p.add_argument('--strategic', type=Path, required=True, help='completed run of run_manhattan_strategic')
    p.add_argument('--output', type=Path, default=Path('figures'))
    p.add_argument(
        '--data-prefix',
        default='data/',
        help='path of the data files as the TikZ sources reference them; the paper, which inputs the '
        'figures from its main file, uses figures/tikz/data/',
    )
    args = p.parse_args()
    data = args.output / 'data'
    data.mkdir(parents=True, exist_ok=True)
    sources = {}
    for arm, rows in panel_a(args.pooling, sources).items():
        write_csv(data / f'fig1a_{arm}.csv', rows)
    for arm, rows in panel_b(args.exploration, sources).items():
        write_csv(data / f'fig1b_{arm}.csv', rows)
    stats = exploration_costs(args.exploration, sources)
    write_csv(data / 'fig2_summary.csv', stats)
    completion, certification = strategic(args.strategic, sources)
    for price in PRICES:
        write_csv(data / f'fig3a_{price}.csv', completion[price])
        write_csv(data / f'fig3b_{price}.csv', certification[price])
    (args.output / 'figure_style.tex').write_text((TEMPLATES / 'figure_style.tex').read_text())
    for number in (1, 2, 3):
        body = figure2(stats) if number == 2 else (TEMPLATES / f'figure{number}.tikz').read_text()
        (args.output / f'figure{number}.tikz').write_text(body.replace('{data}', args.data_prefix))
        (args.output / f'figure{number}.tex').write_text(
            '\\documentclass[border=1pt]{standalone}\n\\usepackage{mathptmx}\n\\input{figure_style.tex}\n'
            f'\\begin{{document}}\n\\input{{figure{number}.tikz}}\n\\end{{document}}\n'
        )
    provenance = dict(
        sources_sha256=sources,
        figure1='Panel (a): blocks of 10^5 epochs of the six-driver run; panel (b): blocks of the twenty-driver run. '
        'Means over seeds, 95% Student-t bands across seed means, clipped to [0, 100] for display.',
        figure2='Per-seed conditional means over exploration epochs of the whole run; 95% Student-t intervals across seeds.',
        figure3='Whole-run completion means and t intervals by allocation rule; all certification blocks under uniform '
        'allocation, no smoothing or subsampling.',
        data_prefix=args.data_prefix,
        outputs_sha256={
            str(p.relative_to(args.output)): sha(p)
            for p in sorted(args.output.rglob('*'))
            if p.is_file() and p.name != 'provenance.json'
        },
    )
    (args.output / 'provenance.json').write_text(json.dumps(provenance, indent=2) + '\n')
    print(f'Wrote {len(provenance["outputs_sha256"])} files (data, TikZ sources, standalone wrappers) to {args.output}')


if __name__ == '__main__':
    main()
