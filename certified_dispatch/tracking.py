"""Local scalar histories with optional W&B mirroring, one run per seed.

No simulator state or tracking observation is returned to the learner/platform.
W&B is imported only when explicitly enabled. Local histories are authoritative.
"""
from collections import defaultdict
import importlib.metadata
import json
from pathlib import Path
import warnings
import numpy as np
from .diagnostics import policy_metrics

DISPATCH_METRICS = (
    'assignment_loss',
    'routing_loss',
    'control_loss',
    'max_prediction_error',
    'max_policy_mismatch',
    'menu_regret',
    'random_regret',
    'priority_regret',
    'negative_surplus_fraction',
    'minimum_surplus',
    'total_payment',
    'dual_certified_shift',
    'control_bound',
    'menu_bound',
    'selected_upper_covered',
    'certified_control_and_priority',
)


def add_tracking_arguments(parser):
    parser.add_argument(
        '--wandb',
        choices=('disabled', 'offline', 'online'),
        default='disabled',
        help='Local JSONL logging is always enabled; W&B is optional',
    )
    parser.add_argument('--wandb-project', default='certified-dispatch')
    parser.add_argument('--wandb-entity', default=None)
    parser.add_argument('--wandb-group', default=None)


class RunLogger:
    def __init__(
        self,
        directory,
        config,
        mode='disabled',
        project='certified-dispatch',
        entity=None,
        group=None,
        origin='live',
        provenance=None,
    ):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=False)
        self.config = config
        self.mode = mode
        self.project = project
        self.entity = entity
        self.group = group or self.directory.parent.name
        self.origin = origin
        self.provenance = provenance or {}
        self.handle = None
        self.remote = None
        self.seed = None
        self.defined = set()
        self.last_steps = {}
        self.wandb = None
        self.remote_error = False
        if mode != 'disabled':
            try:
                import wandb
            except ImportError as exc:
                raise RuntimeError('Install wandb to enable W&B logging (pip install wandb).') from exc
            self.wandb = wandb
        self.status = dict(
            mode=mode,
            project=project,
            entity=entity,
            group=self.group,
            origin=origin,
            status='running',
            runs=[],
            errors=[],
            wandb_version=None if self.wandb is None else importlib.metadata.version('wandb'),
        )
        self._save_status()

    def _save_status(self):
        (self.directory / 'tracking.json').write_text(json.dumps(self.status, indent=2) + '\n')

    def start_seed(self, seed):
        if self.handle is not None:
            raise RuntimeError('Finish the current seed before opening another')
        self.seed = seed
        self.defined = set()
        self.last_steps = {}
        self.remote_error = False
        if self.wandb is not None:
            self.remote = self.wandb.init(
                project=self.project,
                entity=self.entity,
                group=self.group,
                name='%s-seed-%03d' % (self.group, seed),
                job_type=self.origin,
                mode=self.mode,
                dir=str(self.directory.resolve()),
                config={
                    **self.config,
                    'training_seed_index': seed,
                    'logging_origin': self.origin,
                    'provenance': self.provenance,
                },
                settings=self.wandb.Settings(disable_git=True, disable_code=True, console='off', init_timeout=30),
            )
        self.handle = (self.directory / ('seed_%03d.jsonl' % seed)).open('x')
        self.status['runs'].append(
            dict(
                seed=seed,
                id=None if self.remote is None else self.remote.id,
                url=None if self.remote is None or self.mode != 'online' else self.remote.url,
                status='running',
            )
        )
        self._save_status()

    def emit(self, kind, stream, episodes, metrics):
        if self.handle is None:
            raise RuntimeError('No active seed')
        # Each stream has its own episode axis; learner order never merges curves.
        prefix = kind + '/' + stream
        if episodes < self.last_steps.get(prefix, -1):
            raise ValueError('Episode axis decreased for ' + prefix)
        clean = {k: float(v) for k, v in metrics.items() if v is not None}
        if not all(np.isfinite(v) for v in clean.values()):
            raise ValueError('Nonfinite logging metric')
        self.last_steps[prefix] = episodes
        event = dict(
            seed=self.seed, origin=self.origin, kind=kind, stream=stream, episodes=int(episodes), metrics=clean
        )
        self.handle.write(json.dumps(event, allow_nan=False) + '\n')
        self.handle.flush()
        if self.remote is not None and not self.remote_error:
            try:
                axis = prefix + '/episodes'
                if prefix not in self.defined:
                    self.remote.define_metric(axis)
                    self.remote.define_metric(prefix + '/*', step_metric=axis)
                    self.defined.add(prefix)
                self.remote.log({axis: episodes, **{prefix + '/' + k: v for k, v in clean.items()}})
            except Exception as exc:
                self._remote_failure(exc)

    def _remote_failure(self, exc):
        self.remote_error = True
        # Exception type suffices here; avoid persisting credential-bearing URLs.
        self.status['errors'].append(dict(seed=self.seed, error_type=type(exc).__name__))
        self._save_status()
        warnings.warn('W&B logging failed for this seed; local JSONL logging continues.', RuntimeWarning)

    def log_dispatch(self, rows):
        groups = defaultdict(list)
        for row in rows:
            stream = row['scenario'] + '/' + row['mechanism']
            if 'bound_method' in row:
                stream += '/' + row['estimate_method'] + '/' + row['bound_method']
            groups[stream, int(row['evaluation_count']), int(row['checkpoint'])].append(row)
        for (stream, count, cp), group in sorted(groups.items()):
            metrics = DISPATCH_METRICS
            if 'bound_method' in group[0]:
                metrics += (
                    'lower_opt_D_per_driver',
                    'opt_D_per_driver',
                    'lower_benchmark_gap',
                    'all_pair_covered',
                    'test_pass',
                    'false_pass',
                    'heldout_support_exceedance',
                )
            self.emit(
                'dispatch',
                stream + '/eval_%d' % count,
                cp,
                {k: float(np.mean([float(r[k]) for r in group])) for k in metrics},
            )

    def finish_seed(self, success=True):
        if self.remote is not None:
            try:
                self.remote.finish(exit_code=0 if success and not self.remote_error else 1)
            except Exception as exc:
                self._remote_failure(exc)
            self.remote = None
        if self.handle is not None:
            self.handle.close()
            self.handle = None
            self.status['runs'][-1]['status'] = 'complete' if success else 'failed'
            self.status['runs'][-1]['wandb_logging_error'] = self.remote_error
            self._save_status()

    def close(self, success=True):
        self.finish_seed(success)
        self.status['status'] = 'complete' if success else 'failed'
        self._save_status()


class DiagnosticTracker:
    def __init__(self, env, logger):
        self.env = env
        self.logger = logger
        self.previous = {}
        self.mask = np.zeros((env.nodes, env.nodes, env.next.shape[1]), dtype=bool)
        for g in range(env.nodes):
            for s in range(env.nodes):
                self.mask[g, s, env.actions[g][s]] = True
        self.branching = self.mask.sum(axis=2) > 1
        self.optimal = env.route_values(env.roads.mean)

    def record(self, name, learner):
        prior = self.previous.get(name)
        if prior is not None and prior['episodes'] == learner.episodes:
            return
        chosen, metrics = policy_metrics(self.env, learner.q, learner.tie_order, self.optimal, self.mask)
        metrics.update(
            valid_actions_unvisited_fraction=float(np.mean(learner.visits[self.mask] == 0)),
            valid_actions_median_visits=float(np.median(learner.visits[self.mask])),
        )
        if prior is not None:
            metrics.update(
                action_change_fraction=float(np.mean(chosen[self.branching] != prior['chosen'][self.branching]))
                if self.branching.any()
                else 0.0,
                q_change_mean=float(np.mean(np.abs(learner.q[self.mask] - prior['q']))),
                comparison_interval_episodes=learner.episodes - prior['episodes'],
            )
        self.logger.emit('oracle', name, learner.episodes, metrics)
        self.previous[name] = dict(episodes=learner.episodes, chosen=chosen, q=learner.q[self.mask].copy())
