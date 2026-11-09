"""Plot flushed local training histories, including partially completed runs."""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import numpy as np


def plot_tracking(directory, learner='reference'):
    import matplotlib

    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.ticker import ScalarFormatter

    events = []
    for path in sorted(directory.glob('seed_*.jsonl')):
        # A live writer may have an incomplete final line; complete lines stay usable.
        with path.open() as handle:
            for line in handle:
                try:
                    event = json.loads(line)
                except json.JSONDecodeError:
                    if not line.endswith('\n'):
                        continue
                    raise
                if event['stream'] == learner:
                    events.append(event)
    figures = directory / 'figures'
    figures.mkdir(exist_ok=True)
    last_episode = max((event['episodes'] for event in events), default=1)
    groups = [
        (
            'training_curves',
            'train',
            [
                ('mean_cost_per_trip', 'Exploratory training cost', 'Minutes / trip'),
                ('mean_cost_per_transition', 'Traversal cost', 'Minutes / edge'),
                ('epsilon', 'Exploration probability', 'Probability'),
                ('td_rmse', 'Sampled TD error: root mean square', 'Minutes'),
                ('mean_abs_q_update', 'Mean absolute Q update', 'Minutes'),
                ('transitions_per_second', 'Training throughput', 'Transitions / second'),
            ],
        ),
        (
            'policy_diagnostics',
            'oracle',
            [
                ('route_gap_mean', 'Greedy route loss: admissible optimum', 'Minutes / leg'),
                ('prediction_error_mean', 'Prediction vs. own greedy policy', 'Mean absolute error (minutes)'),
                ('q_star_error_mean', 'Q values vs. exact optimal Q', 'Mean absolute error (minutes)'),
                ('bellman_residual_mean', 'Expected Bellman residual', 'Minutes'),
                (
                    'action_change_fraction',
                    'Greedy action changes: choice states',
                    'Fraction since previous diagnostic',
                ),
                ('valid_actions_unvisited_fraction', 'Admissible actions never visited', 'Fraction'),
            ],
        ),
    ]
    for filename, kind, panels in groups:
        fig, axes = plt.subplots(2, 3, figsize=(11, 6.5), constrained_layout=True)
        for ax, (metric, title, ylabel) in zip(axes.flat, panels):
            by_seed = defaultdict(list)
            by_step = defaultdict(list)
            for event in events:
                if event['kind'] == kind and metric in event['metrics']:
                    x, y = event['episodes'], event['metrics'][metric]
                    by_seed[event['seed']].append((x, y))
                    by_step[x].append(y)
            for values in by_seed.values():
                values = sorted(values)
                ax.plot(*np.array(values).T, color='#277da1', alpha=0.18, linewidth=0.9)
            if by_step:
                xs = sorted(by_step)
                ax.plot(
                    xs, [np.mean(by_step[x]) for x in xs], color='#174f75', marker='.', label='Mean of available seeds'
                )
            else:
                ax.text(0.5, 0.5, 'Not recorded in this run', ha='center', va='center', transform=ax.transAxes)
            ax.set(title=title, xlabel='Completed training trips per learner', ylabel=ylabel)
            if last_episode >= 1000:
                ax.set_xscale('symlog', linthresh=1000)
                ax.set_xticks([0] + [10**power for power in range(3, int(np.log10(last_episode)) + 1)])
            else:
                ax.xaxis.set_major_formatter(ScalarFormatter())
            ax.set_xlim(0, max(1, last_episode) * 1.1)
            ax.grid(alpha=0.2)
        fig.suptitle(
            '%s — %s\nThin lines: individual seeds; thick line: descriptive mean, not a confidence interval'
            % (
                learner,
                'Training observations (exploration enabled)'
                if kind == 'train'
                else 'Read-only simulator diagnostics (greedy policy)',
            ),
            fontsize=11,
        )
        for ext in ('png', 'svg'):
            fig.savefig(figures / ('%s_%s.%s' % (filename, learner, ext)), dpi=180)
        plt.close(fig)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--directory', type=Path, required=True)
    parser.add_argument('--learner', default='reference')
    args = parser.parse_args()
    plot_tracking(args.directory, args.learner)
