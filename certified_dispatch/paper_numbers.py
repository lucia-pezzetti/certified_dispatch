"""Print every number quoted in Section IV of the paper from the completed runs that produced it.

python -m certified_dispatch.paper_numbers --pooling results/pooling --exploration results/exploration \
    --strategic results/strategic [--grid results/grid]

Each line gives the quantity, its value computed from the saved report tables, and the statement of the
paper it supports. Means are over seeds (within seed first); runs that are not available are skipped.
"""
import argparse
import csv
from pathlib import Path


def rows(path):
    with Path(path).open() as handle:
        return list(csv.DictReader(handle))


def pick(table, **where):
    out = [r for r in table if all(r[k] == str(v) for k, v in where.items())]
    assert len(out) == 1, f'expected one row for {where}, found {len(out)}'
    return out[0]


def line(quantity, value, statement):
    print(f'{quantity:<58} {value:>14}   {statement}')


def grid(run):
    print('\nIV-A  Assignment accuracy and participation (10 x 10 grid, 6 drivers)')
    table = rows(run / 'summary.csv')
    final = dict(
        scenario='independent_current',
        checkpoint=1000000,
        evaluation_count=4096,
        estimate_method='sample_mean',
        bound_method='known_support',
    )

    def value(metric, mechanism='dual_base'):
        return float(pick(table, metric=metric, mechanism=mechanism, **final)['mean'])

    line('Estimated assignment optimal for the true costs', f'{100*value("assignment_optimal"):.1f} %', '95.8 %')
    line('Mean assignment loss (minutes per driver)', f'{value("assignment_loss"):.1e}', '5e-5')
    line(
        'Drivers with negative payoff at unadjusted dual prices',
        f'{100*value("negative_surplus_fraction"):.1f} %',
        '10.1 %',
    )
    line('Mean worst shortfall per dispatch (minutes)', f'{value("priority_regret"):.1e}', '2.9e-3')


def pooling(run):
    print('\nIV-B  Online certification from pooled observations (Manhattan, 6 drivers, 10^7 epochs)')
    table = rows(run / 'summary.csv')
    cell = dict(fleet=6, hubs=8, rho=0.0, window='terminal_100k', metric='certified')
    for arm, statement in (('private_variance', '43.9 %'), ('pooled_variance', '99.2 %')):
        r = pick(table, arm=arm, **cell)
        line(f'Certification rate, final 10^5 epochs, {arm}', f'{100*float(r["mean"]):.1f} %', statement)


def exploration(run):
    print('\nIV-B  Exploration (Manhattan, 20 drivers, 3 x 10^6 epochs)')
    summary = rows(run / 'summary.csv')
    thresholds = rows(run / 'threshold_summary.csv')
    cell = dict(fleet=20, hubs=8, rho=0.0)
    terminal = dict(window='terminal_1pct', **cell)
    none = {
        m: float(pick(summary, arm='pooled_none', metric=m, **terminal)['mean'])
        for m in ('certified', 'assignment_loss', 'bound', 'lower_benchmark_gap')
    }
    line('Certification rate without exploration, final 3 x 10^4 epochs', f'{100*none["certified"]:.1f} %', '1.8 %')
    line('Mean assignment loss without exploration (min/driver)', f'{none["assignment_loss"]:.1e}', 'below 2e-3')
    line('Mean assignment loss bound (min/driver)', f'{none["bound"]:.2f}', '4.86')
    line('Of which the lower-benchmark gap (OPT(C) - OPT(L))/N', f'{none["lower_benchmark_gap"]:.2f}', '4.80')
    for arm, statement in (
        ('pooled_optimistic', '1.9 %'),
        ('pooled_optimistic_relaxed', '98.6 %'),
        ('pooled_random_relaxed', '99.9 %'),
    ):
        r = pick(summary, arm=arm, metric='certified', **terminal)
        line(f'Terminal certification rate, {arm}', f'{100*float(r["mean"]):.1f} %', statement)
    for arm, statement in (('pooled_random_relaxed', '0.27 million'), ('pooled_optimistic_relaxed', '0.93 million')):
        r = pick(thresholds, arm=arm, metric='first_90_epoch', **cell)
        line(f'First block with 90 % certification, {arm}', f'{float(r["mean"])/1e6:.2f} million', statement)
    whole = dict(window='whole_run', **cell)
    cost = {
        (arm, m): float(pick(summary, arm=arm, metric=m, **whole)['mean'])
        for arm in ('pooled_optimistic_relaxed', 'pooled_random_relaxed')
        for m in ('conditional_explored_loss', 'conditional_explored_shift')
    }
    o, r = 'pooled_optimistic_relaxed', 'pooled_random_relaxed'
    line(
        'Assignment loss per exploratory dispatch, optimistic / random',
        f'{cost[o, "conditional_explored_loss"]:.3f} / {cost[r, "conditional_explored_loss"]:.1f}',
        'Fig. 2(a)',
    )
    line(
        'Reduction of assignment loss by optimistic exploration',
        f'{100*(1-cost[o, "conditional_explored_loss"]/cost[r, "conditional_explored_loss"]):.1f} %',
        '99.4 %',
    )
    line(
        'Extra payment per exploratory dispatch, optimistic / random',
        f'{cost[o, "conditional_explored_shift"]:.1f} / {cost[r, "conditional_explored_shift"]:.1f}',
        'Fig. 2(b)',
    )
    line(
        'Reduction of the extra payment by optimistic exploration',
        f'{100*(1-cost[o, "conditional_explored_shift"]/cost[r, "conditional_explored_shift"]):.1f} %',
        '72.4 %',
    )


def strategic(run):
    print('\nIV-C  Strategic job requests (Manhattan, 6 drivers, 10^7 epochs)')
    arms = rows(run / 'summary_arms.csv')
    curves = rows(run / 'curves.csv')
    whole = dict(rho=0.0, window='whole_run')

    def arm(price, rule):
        return pick(arms, price=price, allocation=rule, **whole)

    for price, rule, metric, statement in (
        ('dual_certified', 'priority', 'completion', 'every job'),
        ('direct_upper', 'priority', 'completion', 'every job'),
        ('dual_base', 'priority', 'idle', '30 % idle'),
        ('dual_certified', 'uniform', 'completion', 'almost every job'),
        ('direct_upper', 'uniform', 'completion', '44 %'),
        ('direct_upper_matched', 'uniform', 'completion', '78 %'),
        ('uniform_matched', 'uniform', 'completion', '64 %'),
        ('dual_certified', 'cheapest', 'completion', '88 %'),
    ):
        line(
            f'{metric.capitalize()}, {price} prices, {rule} allocation',
            f'{100*float(arm(price, rule)[metric]):.1f} %',
            statement,
        )
    direct, dual = float(arm('direct_upper', 'priority')['total_payment']), float(
        arm('dual_certified', 'priority')['total_payment']
    )
    line(
        'Payment of direct prices relative to adjusted dual prices, priority',
        f'{100*(1-direct/dual):.1f} % lower',
        '24 % lower',
    )
    line(
        'Mean maximum deviation gain, adjusted dual, cheapest (minutes)',
        f'{float(arm("dual_certified", "cheapest")["compliance_regret"]):.1e}',
        '6e-3',
    )
    for price, statement in (('dual_certified', '2.3 million'), ('direct_upper', '4.3 million')):
        series = sorted(
            [r for r in curves if r['price'] == price and r['allocation'] == 'uniform' and r['rho'] == '0.0'],
            key=lambda r: int(r['epoch']),
        )
        first = next((int(r['epoch']) for r in series if float(r['certified']) >= 0.9), None)
        line(
            f'First block with 90 % certification under uniform allocation, {price}',
            'not reached' if first is None else f'{first/1e6:.2f} million',
            statement,
        )


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--grid', type=Path, help='completed run of run_rl with configs/rl_grid10_paper.json')
    p.add_argument('--pooling', type=Path, help='completed run of run_manhattan_scale')
    p.add_argument('--exploration', type=Path, help='completed run of run_manhattan_explore')
    p.add_argument('--strategic', type=Path, help='completed run of run_manhattan_strategic')
    args = p.parse_args()
    print(f'{"Quantity":<58} {"Computed":>14}   Paper')
    for run, report in (
        (args.grid, grid),
        (args.pooling, pooling),
        (args.exploration, exploration),
        (args.strategic, strategic),
    ):
        if run is not None:
            report(run)


if __name__ == '__main__':
    main()
