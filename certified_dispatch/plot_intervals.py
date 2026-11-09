"""Primary OPT(D) figures; never label empirical-extrema passes as certified."""
import json
from .plot_rl import read, CASES, LABELS
from .interval_study import BOUND_METHODS

NAMES = dict(
    known_support='Known support: Eq. (23)',
    empirical_same='Observed min/max: same data',
    empirical_split='Observed min/max: separate data',
    structural_lower='Structural lower bound',
)
COLORS = dict(zip(BOUND_METHODS, ('#277da1', '#d97706', '#933d86', '#6b7280')))


def plot_results(directory):
    import matplotlib

    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.ticker import NullFormatter

    plt.rcParams.update({'font.size': 9, 'axes.spines.top': False, 'axes.spines.right': False})
    c = json.loads((directory / 'config.json').read_text())
    rows = read(directory / 'summary.csv')
    figures = directory / 'figures'
    figures.mkdir(exist_ok=True)

    def data(
        case, metric, bound='known_support', estimate='sample_mean', mechanism='dual_certified', axis='evaluation_count'
    ):
        fixed = 'checkpoint' if axis == 'evaluation_count' else 'evaluation_count'
        maximum = max(c['checkpoints'] if fixed == 'checkpoint' else c['evaluation_counts'])
        return sorted(
            [
                r
                for r in rows
                if r['scenario'] == case
                and r['metric'] == metric
                and r['bound_method'] == bound
                and r['estimate_method'] == estimate
                and r['mechanism'] == mechanism
                and int(r[fixed]) == maximum
            ],
            key=lambda r: int(r[axis]),
        )

    def line(ax, values, label, color, axis='evaluation_count', style='-'):
        xs = [int(r[axis]) for r in values]
        ys = [float(r['mean']) for r in values]
        halves = [float(r['ci95_halfwidth'] or 0) for r in values]
        fractional = values[0]['metric'] in ('all_pair_covered', 'test_pass', 'false_pass')
        error = (
            ([min(h, y) for h, y in zip(halves, ys)], [min(h, 1 - y) for h, y in zip(halves, ys)])
            if fractional
            else halves
        )
        ax.errorbar(xs, ys, yerr=error, label=label, color=color, linestyle=style, marker='.', capsize=2)
        ax.set_xscale(
            'log' if axis == 'evaluation_count' else 'symlog',
            **({} if axis == 'evaluation_count' else {'linthresh': 100})
        )
        if axis == 'evaluation_count':
            ax.set_xlim(min(xs) * 0.85, max(xs) * 1.15)
            ax.set_xticks(xs, [str(x) for x in xs])
            ax.xaxis.set_minor_formatter(NullFormatter())
        if fractional:
            ax.set_ylim(-0.03, 1.03)
        ax.grid(alpha=0.2)
        ax.set_xlabel(
            'Mean-estimation trips per pair K' if axis == 'evaluation_count' else 'Training trips per learner'
        )

    def save(fig, name):
        for ext in ('png', 'svg'):
            fig.savefig(figures / (name + '.' + ext), dpi=180)
        plt.close(fig)

    panels = [
        ('control_bound', 'Assignment loss bound B_D (min/driver)'),
        ('lower_benchmark_gap', 'OPT(D)/N − OPT(L^D)/N (min/driver)'),
        ('dual_certified_shift', 'Additional payment per driver'),
        ('all_pair_covered', 'All-pair mean coverage (fraction)'),
        ('test_pass', 'Dispatch test pass fraction'),
        ('false_pass', 'False passes / all batches'),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(12, 7), constrained_layout=True)
    # Separate cases to avoid conflating policy heterogeneity with interval choice.
    for case in CASES:
        for ax, (metric, title) in zip(axes.flat, panels):
            for bound in BOUND_METHODS:
                line(ax, data(case, metric, bound), NAMES[bound], COLORS[bound])
            if metric == 'control_bound':
                line(ax, data(case, 'assignment_loss'), 'Measured assignment loss', '#111111', style='--')
            ax.set_title(title)
        axes[0, 0].legend(fontsize=6)
        fig.suptitle(
            LABELS[case]
            + ': sample-mean assignment; empirical extrema are not certificates\nMeans and 95%% t intervals across %d seeds; fraction intervals intersected with [0,1]'
            % c['seeds'],
            fontsize=11,
        )
        save(fig, 'intervals_' + case)
        if case != CASES[-1]:
            fig, axes = plt.subplots(2, 3, figsize=(12, 7), constrained_layout=True)

    fig, axes = plt.subplots(2, 3, figsize=(12, 6.5), constrained_layout=True)
    for axs, case in zip(axes, CASES):
        for ax, metric, title in zip(
            axs,
            ['assignment_loss', 'max_prediction_error', 'routing_loss'],
            ['Loss relative to OPT(D)', 'Cost prediction error', 'Routing loss (secondary)'],
        ):
            for estimate, color in [('sample_mean', '#277da1'), ('reference_q', '#933d86')]:
                line(ax, data(case, metric, estimate=estimate, axis='checkpoint'), estimate, color, axis='checkpoint')
            ax.set(title=title, ylabel=LABELS[case] + '\nMinutes')
    axes[0, 0].legend()
    fig.suptitle('Training and estimation are separate budgets; maximum K, known-support intervals')
    save(fig, 'interval_learning')

    fig, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
    for ax, case in zip(axes, CASES):
        for bound in BOUND_METHODS:
            line(ax, data(case, 'lower_opt_D_per_driver', bound), NAMES[bound], COLORS[bound])
        line(ax, data(case, 'opt_D_per_driver'), 'Exact OPT(D)/N', '#111111', style='--')
        line(ax, data(case, 'expected_cost_per_driver'), 'Executed expected cost', '#238636', style=':')
        ax.set(title=LABELS[case], ylabel='Minutes per driver')
    axes[0].legend(fontsize=7)
    fig.suptitle('Data-driven lower benchmark versus OPT(D): final checkpoint')
    save(fig, 'interval_benchmarks')

    methods = ['dual_base', 'dual_certified', 'uniform_matched', 'direct_upper', 'direct_upper_matched']
    fig, axes = plt.subplots(2, 3, figsize=(12, 6.5), constrained_layout=True)
    for axs, case in zip(axes, CASES):
        for ax, metric in zip(axs, ['menu_regret', 'priority_regret', 'total_payment']):
            values = [data(case, metric, mechanism=m)[-1] for m in methods]
            ax.bar(
                range(len(methods)),
                [float(r['mean']) for r in values],
                yerr=[float(r['ci95_halfwidth'] or 0) for r in values],
                capsize=2,
            )
            ax.set_xticks(range(len(methods)), [m.replace('_', '\n') for m in methods], fontsize=7)
            ax.set(title=metric.replace('_', ' '), ylabel=LABELS[case])
    fig.suptitle('Final checkpoint and K: sample-mean assignments, known-support intervals')
    save(fig, 'interval_payments')
