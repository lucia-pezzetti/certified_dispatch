"""Plots of complete single-agent training runs and checkpoint certification."""
import csv
import json
import numpy as np

CASES = ('shared_current', 'independent_current')
LABELS = {'shared_current': 'Shared current policy', 'independent_current': 'Independent current policies'}
COLORS = {'shared_current': '#277da1', 'independent_current': '#933d86'}


def read(path):
    with path.open() as f:
        return list(csv.DictReader(f))


def plot_results(directory):
    import matplotlib

    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    plt.rcParams.update({'font.size': 9, 'axes.spines.top': False, 'axes.spines.right': False})
    c = json.loads((directory / 'config.json').read_text())
    rows = read(directory / 'summary.csv')
    figures = directory / 'figures'
    figures.mkdir(exist_ok=True)

    def data(case, metric, method='dual_base', checkpoint=None, evaluation=None):
        return [
            r
            for r in rows
            if r['scenario'] == case
            and r['metric'] == metric
            and r['mechanism'] == method
            and (checkpoint is None or int(r['checkpoint']) == checkpoint)
            and (evaluation is None or int(r['evaluation_count']) == evaluation)
        ]

    def line(ax, values, xkey, label, color, style='-'):
        values = sorted(values, key=lambda r: int(r[xkey]))
        ax.errorbar(
            [int(r[xkey]) for r in values],
            [float(r['mean']) for r in values],
            yerr=[float(r['ci95_halfwidth'] or 0) for r in values],
            color=color,
            linestyle=style,
            marker='o',
            capsize=2,
            label=label,
        )

    def save(fig, name):
        fig.savefig(figures / (name + '.png'), dpi=180)
        fig.savefig(figures / (name + '.svg'))
        plt.close(fig)

    fig, axes = plt.subplots(2, 3, figsize=(10.5, 6.6), constrained_layout=True)
    metrics = [
        'assignment_loss',
        'routing_loss',
        'max_prediction_error',
        'menu_regret',
        'negative_surplus_fraction',
        'max_policy_mismatch',
    ]
    titles = [
        'Assignment loss under driver policies',
        'Routing loss on selected pairs',
        'Maximum cost prediction error',
        'Maximum gross deviation gain',
        'Negative recommended surplus',
        'Driver vs. reference policy costs',
    ]
    for ax, metric, title in zip(axes.flat, metrics, titles):
        for case in CASES:
            line(
                ax, data(case, metric, evaluation=max(c['evaluation_counts'])), 'checkpoint', LABELS[case], COLORS[case]
            )
        ax.set(
            title=title,
            xlabel='Training trips per learner',
            ylabel='Fraction' if metric.endswith('fraction') else 'Minutes',
        )
        ax.set_xscale('symlog', linthresh=100)
        ax.grid(alpha=0.2)
    axes[0, 0].legend(fontsize=7)
    fig.suptitle(c['name'].replace('_', ' ').title() + ': Q-learning from scratch; base dual payments')
    save(fig, 'rl_learning')

    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.2), constrained_layout=True)
    for case in CASES:
        for metric, style, label in [('control_loss', '-', 'measured'), ('control_bound', '--', 'bound')]:
            line(
                axes[0],
                data(case, metric, 'dual_certified', max(c['checkpoints'])),
                'evaluation_count',
                LABELS[case] + ' ' + label,
                COLORS[case],
                style,
            )
        line(
            axes[1],
            data(case, 'dual_certified_shift', 'dual_certified', max(c['checkpoints'])),
            'evaluation_count',
            LABELS[case],
            COLORS[case],
        )
        line(
            axes[2],
            data(case, 'certified_control_and_priority', 'dual_certified', evaluation=max(c['evaluation_counts'])),
            'checkpoint',
            LABELS[case],
            COLORS[case],
        )
    for ax, title in zip(
        axes[:2], ['Execution loss and bound: final checkpoint', 'Additional payment: final checkpoint']
    ):
        ax.set(
            title=title, xlabel='Evaluation trips per selected driver', ylabel='Minutes / payment units', xscale='log'
        )
        ax.grid(alpha=0.2)
    axes[0].legend(fontsize=6)
    axes[2].set(
        title='Fraction passing the certificate', xlabel='Training trips per learner', ylabel='Certified fraction'
    )
    axes[2].set_xscale('symlog', linthresh=100)
    axes[2].grid(alpha=0.2)
    fig.suptitle(
        'Certification: control tolerance %.2f, additional payment budget %.2f'
        % (c['control_tolerance'], c['payment_budget'])
    )
    save(fig, 'rl_certificates')

    methods = ['dual_base', 'dual_certified', 'uniform_matched', 'direct_upper', 'direct_upper_matched']
    names = ['Dual\nbase', 'Dual\ncertified', 'Uniform\nmatched', 'Direct\nupper', 'Direct\nmatched']
    fig, axes = plt.subplots(2, 3, figsize=(10.5, 5.8), constrained_layout=True)
    for row, case in enumerate(CASES):
        for ax, metric, title in zip(
            axes[row],
            ['menu_regret', 'random_regret', 'total_payment'],
            ['Gross deviation gain', 'Unilateral regret: random allocation', 'Total payment at compliance'],
        ):
            values = [data(case, metric, m, max(c['checkpoints']), max(c['evaluation_counts']))[0] for m in methods]
            ax.bar(
                range(5),
                [float(r['mean']) for r in values],
                yerr=[float(r['ci95_halfwidth'] or 0) for r in values],
                color=COLORS[case],
                capsize=2,
            )
            ax.set_xticks(range(5), names, fontsize=7)
            ax.set(title=title, ylabel='Payment units' if metric == 'total_payment' else 'Minutes')
        axes[row, 0].set_ylabel(LABELS[case] + '\nMinutes')
    fig.suptitle('Final trained checkpoints: matched assignments and driver policies')
    save(fig, 'rl_mechanisms')

    budgets = read(directory / 'budgets.csv')
    fig, axes = plt.subplots(1, 2, figsize=(8, 3.8), constrained_layout=True)
    for ax, case in zip(axes, CASES):
        matrix = np.array(
            [
                [
                    next(
                        float(r['certified_fraction'])
                        for r in budgets
                        if r['scenario'] == case
                        and float(r['control_tolerance']) == tol
                        and float(r['payment_budget']) == budget
                    )
                    for budget in c['payment_budgets']
                ]
                for tol in c['control_tolerances']
            ]
        )
        im = ax.imshow(matrix, origin='lower', aspect='auto', vmin=0, vmax=1, cmap='Blues')
        ax.set_xticks(range(len(c['payment_budgets'])), c['payment_budgets'])
        ax.set_yticks(range(len(c['control_tolerances'])), c['control_tolerances'])
        ax.set(
            title=LABELS[case], xlabel='Additional payment per driver', ylabel='Fleet-cost tolerance (minutes/driver)'
        )
        for i in range(len(matrix)):
            for j in range(len(matrix[0])):
                ax.text(
                    j,
                    i,
                    '%.0f%%' % (100 * matrix[i, j]),
                    ha='center',
                    va='center',
                    fontsize=7,
                    color='white' if matrix[i, j] > 0.55 else 'black',
                )
    fig.colorbar(im, ax=axes, label='Certified fraction', shrink=0.85)
    save(fig, 'rl_budgets')
