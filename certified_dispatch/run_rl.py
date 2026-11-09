"""Assignment accuracy and participation on the stochastic grid (Section IV-A).

python -m certified_dispatch.run_rl --output results/grid

Each seed trains one reference and `fleet` private goal-conditioned tabular Q learners
on the 10 x 10 grid and freezes their greedy routing policies at the configured
checkpoints. At every checkpoint and for every batch of driver and job locations, the
platform estimates the service costs of each driver-job pair from K independent
simulated services, computes the estimated-optimal assignment and its dual prices, and
the simulator scores the result at the exact expected costs: assignment loss, the share
of drivers with a negative payoff under the unadjusted dual prices, the worst shortfall,
and the certificates of the paper. With `interval_study` set, as in the paper's
configuration, all-pair observations and four interval constructions are evaluated.
"""
import argparse
from collections import defaultdict
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time
import numpy as np
from scipy.stats import t

from .model import random_stream
from .rl import Learner, locations, pair_costs, frozen_trips, save_checkpoint
from .platform import assign, optimum, selected_upper, certificate, mechanisms, evaluate_payments
from .common import CONFIGS, write_csv, snapshot_source
from .networks import environment, network_path, network_diagnostics
from .tracking import RunLogger, DiagnosticTracker, add_tracking_arguments

SCENARIOS = ('shared_current', 'independent_current')
GROUPS = ('scenario', 'checkpoint', 'evaluation_count', 'mechanism')


def validate(c):
    keys = ('fleet', 'seeds', 'batches') if c.get('network_file') else ('side', 'fleet', 'seeds', 'batches')
    for key in keys:
        if not isinstance(c[key], int) or c[key] < (2 if key in ('side', 'fleet') else 1):
            raise ValueError('Invalid ' + key)
    for key, minimum in (('checkpoints', 0), ('evaluation_counts', 1)):
        if not c[key] or c[key] != sorted(set(c[key])) or any(not isinstance(k, int) or k < minimum for k in c[key]):
            raise ValueError('Invalid ' + key)
    if c['checkpoints'][-1] <= 0 or not 0 < c['alpha'] < 1 or c['eta'] <= 0:
        raise ValueError('Need training, valid alpha, and positive eta')
    if not 0 <= c['epsilon_end'] <= c['epsilon_start'] <= 1 or c['epsilon_decay_episodes'] <= 0:
        raise ValueError('Invalid exploration schedule')
    if not 0.5 < c['learning_rate_power'] <= 1:
        raise ValueError('Learning-rate exponent must be in (0.5, 1]')
    for key in ('control_tolerances', 'payment_budgets'):
        if not c[key] or c[key] != sorted(set(c[key])) or any(x < 0 for x in c[key]):
            raise ValueError('Invalid ' + key)
    if c.get('interval_study') and (not isinstance(c.get('support_test_count'), int) or c['support_test_count'] < 1):
        raise ValueError('Need positive support_test_count for interval study')
    if c['control_tolerance'] < 0 or c['payment_budget'] < 0:
        raise ValueError('Negative certification threshold')


def aggregate(rows):
    groups = GROUPS + (('estimate_method', 'bound_method') if 'bound_method' in rows[0] else ())
    metrics = [k for k in rows[0] if k not in (*groups, 'seed', 'batch')]
    grouped = defaultdict(list)
    for r in rows:
        grouped[tuple(r[k] for k in groups) + (r['seed'],)].append(r)
    seeds = [
        {**dict(zip((*groups, 'seed'), key)), **{m: float(np.mean([r[m] for r in group])) for m in metrics}}
        for key, group in sorted(grouped.items())
    ]
    grouped = defaultdict(list)
    for r in seeds:
        grouped[tuple(r[k] for k in groups)].append(r)
    summary = []
    for key, group in sorted(grouped.items()):
        for metric in metrics:
            values = np.array([r[metric] for r in group])
            half = (
                float(t.ppf(0.975, len(values) - 1) * values.std(ddof=1) / np.sqrt(len(values)))
                if len(values) > 1
                else None
            )
            summary.append(
                {
                    **dict(zip(groups, key)),
                    'metric': metric,
                    'mean': float(values.mean()),
                    'ci95_halfwidth': half,
                    'independent_seeds': len(values),
                }
            )
    return seeds, summary


def budget_grid(rows, c):
    result = []
    extra = ('estimate_method', 'bound_method') if 'bound_method' in rows[0] else ()
    groups = ('scenario',) + extra
    final = [
        r
        for r in rows
        if r['checkpoint'] == max(c['checkpoints'])
        and r['evaluation_count'] == max(c['evaluation_counts'])
        and r['mechanism'] == 'dual_certified'
    ]
    for key in sorted({tuple(r[k] for k in groups) for r in final}):
        data = [r for r in final if tuple(r[k] for k in groups) == key]
        for control in c['control_tolerances']:
            for payment in c['payment_budgets']:
                rates = [
                    np.mean(
                        [
                            r['control_bound'] <= control and r['dual_certified_shift'] <= payment
                            for r in data
                            if r['seed'] == seed
                        ]
                    )
                    for seed in range(c['seeds'])
                ]
                entry = dict(zip(groups, key)) | dict(
                    control_tolerance=control, payment_budget=payment, certified_fraction=float(np.mean(rates))
                )
                if extra:
                    passes = [r for r in data if r['control_bound'] <= control and r['dual_certified_shift'] <= payment]
                    false_count = sum(
                        r['assignment_loss'] > control + 1e-8 or r['minimum_surplus'] < c['eta'] - 1e-8 for r in passes
                    )
                    valid = int(data[0]['theoretical_guarantee'])
                    entry.update(
                        test_pass_fraction=float(np.mean(rates)),
                        theoretical_guarantee=valid,
                        certified_fraction=float(np.mean(rates)) * valid,
                        false_pass_fraction=false_count / len(data),
                        false_pass_among_passes=false_count / len(passes) if passes else None,
                    )
                result.append(entry)
    return result


def experiment(c, output, logger=None, log_every=5000, diagnostic_every=50000):
    if log_every < 1 or diagnostic_every < log_every or diagnostic_every % log_every:
        raise ValueError('Diagnostic interval must be a positive multiple of the logging interval')
    interval_study = c.get('interval_study', False)
    env = environment(c)
    roads = env.roads
    lower_values = env.route_values(roads.free)
    support_values = env.route_values(roads.free + roads.delay, maximize=True)
    admissible_optimal_values = env.route_values(roads.mean)
    # On a road network, compare with the unrestricted directed-road optimum.
    # The action restriction's cost is reported separately from learning error.
    optimal_values = roads.paths(roads.mean)[0] if c.get('network_file') else admissible_optimal_values
    if c.get('network_file'):
        report = network_diagnostics(env)
        (output / 'network_diagnostics.json').write_text(json.dumps(report, indent=2) + '\n')
    rows, training, coverage = [], [], []
    n = c['fleet']
    ids = np.arange(n)
    # Joint coverage across BOTH cases, every batch, checkpoint, and evaluation
    # budget within one independent training seed. Q values are not certified.
    comparisons = n * len(SCENARIOS) * c['batches'] * len(c['checkpoints']) * len(c['evaluation_counts'])
    for seed_index in range(c['seeds']):
        if logger is not None:
            logger.start_seed(seed_index)
            diagnostics = DiagnosticTracker(env, logger)
        master = c['base_seed'] + seed_index
        learner_seeds = [int(random_stream(master, 90, i).integers(0, 2**31)) for i in range(n + 1)]
        reference = Learner.create(env, learner_seeds[0])
        drivers = [Learner.create(env, s) for s in learner_seeds[1:]]
        all_covered = True
        seed_dir = output / 'checkpoints' / ('seed_%03d' % seed_index)
        seed_dir.mkdir(parents=True)
        for checkpoint in c['checkpoints']:
            checkpoint_start = len(rows)
            for i, learner in enumerate([reference] + drivers):
                name = 'reference' if i == 0 else 'driver_%d' % (i - 1)

                def log_window(metrics):
                    logger.emit(
                        'train', name, metrics['episodes'], {k: v for k, v in metrics.items() if k != 'episodes'}
                    )
                    if metrics['episodes'] % diagnostic_every == 0:
                        diagnostics.record(name, learner)

                trace = learner.train_to(
                    checkpoint, c, on_log=None if logger is None else log_window, log_every=log_every
                )
                if logger is not None:
                    diagnostics.record(name, learner)
                training.append(
                    dict(
                        seed=seed_index,
                        learner='reference' if i == 0 else 'driver_%d' % (i - 1),
                        learner_seed=learner.seed,
                        **trace
                    )
                )
            save_checkpoint(seed_dir / ('episode_%06d.npz' % checkpoint), reference, drivers)
            predictions = reference.prediction()
            reference_paths = reference.frozen_paths()
            private_paths = [d.frozen_paths() for d in drivers]
            epoch_covered = {name: True for name in SCENARIOS}
            for batch in range(c['batches']):
                loc = locations(env, n, master, batch)
                estimate = pair_costs(predictions, *loc)
                lower = pair_costs(lower_values, *loc)
                support = pair_costs(support_values, *loc)
                Cstar = pair_costs(optimal_values, *loc)
                admissible_Cstar = pair_costs(admissible_optimal_values, *loc)
                # This is the only assignment computation. It has no private Qs,
                # driver paths, exact expected costs, or evaluation observations.
                assignment = assign(estimate)
                sigma = assignment.sigma
                own = ids, sigma
                shared = frozen_trips(roads, [reference_paths] * n, *loc)
                reference_cost = shared.expected_costs()
                for case, scenario in enumerate(SCENARIOS):
                    trips = shared if case == 0 else frozen_trips(roads, private_paths, *loc)
                    D = trips.expected_costs()  # Used only for evaluation below.
                    if not np.all((lower <= Cstar + 1e-8) & (Cstar <= D + 1e-8) & (D <= support + 1e-8)):
                        raise AssertionError('Route benchmark or public bound inconsistent')
                    if interval_study:
                        from .interval_study import evaluate_batch

                        new_rows = evaluate_batch(
                            c,
                            trips,
                            estimate,
                            lower,
                            support,
                            Cstar,
                            admissible_Cstar,
                            reference_cost,
                            master,
                            checkpoint,
                            batch,
                            case,
                            scenario,
                            seed_index,
                            dict(
                                reference_training_trips=reference.episodes,
                                reference_training_transitions=reference.transitions,
                                driver_training_trips=0 if case == 0 else sum(d.episodes for d in drivers),
                                driver_training_transitions=0 if case == 0 else sum(d.transitions for d in drivers),
                            ),
                        )
                        rows.extend(new_rows)
                        epoch_covered[scenario] &= all(
                            r['all_pair_covered'] for r in new_rows if r['bound_method'] == 'known_support'
                        )
                        continue
                    evaluations = np.array(
                        [
                            trips.sample(
                                i,
                                int(sigma[i]),
                                max(c['evaluation_counts']),
                                random_stream(master, 100, checkpoint, batch, case, i),
                            )
                            for i in ids
                        ]
                    )
                    if not np.all(
                        (evaluations >= lower[own][:, None] - 1e-8) & (evaluations <= support[own][:, None] + 1e-8)
                    ):
                        raise AssertionError('Evaluation trip outside public support')
                    for count in c['evaluation_counts']:
                        upper = selected_upper(
                            evaluations[:, :count], lower[own], support[own], comparisons, c['alpha']
                        )
                        event = bool(np.all(D[own] <= upper + 1e-8))
                        epoch_covered[scenario] &= event
                        menus, shift = mechanisms(assignment, upper, c['eta'])
                        for name, payments in menus.items():
                            cert = certificate(sigma, payments, lower, upper)
                            measured = evaluate_payments(D, sigma, payments)
                            control_loss = float(D[own].mean() - optimum(Cstar) / n)
                            if event:
                                if (
                                    control_loss > cert['control_bound'] + 1e-7
                                    or measured['menu_regret'] > cert['menu_bound'] + 1e-7
                                ):
                                    raise AssertionError('Certificate violated on its coverage event')
                                if name == 'dual_certified' and measured['minimum_surplus'] < c['eta'] - 1e-7:
                                    raise AssertionError('Certified participation failed')
                            rows.append(
                                dict(
                                    scenario=scenario,
                                    seed=seed_index,
                                    batch=batch,
                                    checkpoint=checkpoint,
                                    evaluation_count=count,
                                    mechanism=name,
                                    assignment_loss=float(D[own].mean() - optimum(D) / n),
                                    routing_loss=float((D[own] - Cstar[own]).mean()),
                                    routing_restriction_loss=float((admissible_Cstar[own] - Cstar[own]).mean()),
                                    routing_learning_loss=float((D[own] - admissible_Cstar[own]).mean()),
                                    assignment_loss_optimal_routes=float(Cstar[own].mean() - optimum(Cstar) / n),
                                    control_loss=control_loss,
                                    expected_cost_per_driver=float(D[own].mean()),
                                    max_prediction_error=float(np.max(np.abs(D - estimate))),
                                    reference_policy_prediction_error=float(np.max(np.abs(reference_cost - estimate))),
                                    max_policy_mismatch=float(np.max(np.abs(D - reference_cost))),
                                    mean_selected_policy_mismatch=float(np.mean(np.abs(D[own] - reference_cost[own]))),
                                    dual_certified_shift=shift,
                                    selected_upper_covered=int(event),
                                    certified_control_and_priority=int(
                                        cert['control_bound'] <= c['control_tolerance'] and shift <= c['payment_budget']
                                    )
                                    if name == 'dual_certified'
                                    else 0,
                                    reference_training_trips=reference.episodes,
                                    reference_training_transitions=reference.transitions,
                                    driver_training_trips=0 if case == 0 else sum(d.episodes for d in drivers),
                                    driver_training_transitions=0 if case == 0 else sum(d.transitions for d in drivers),
                                    evaluation_trips=n * count,
                                    **cert,
                                    **measured
                                )
                            )
            if interval_study:
                from .interval_study import BOUND_METHODS

                for scenario in SCENARIOS:
                    for method in BOUND_METHODS:
                        selected = [
                            r
                            for r in rows[checkpoint_start:]
                            if r['scenario'] == scenario and r['bound_method'] == method
                        ]
                        coverage.append(
                            dict(
                                seed=seed_index,
                                checkpoint=checkpoint,
                                scenario=scenario,
                                bound_method=method,
                                all_batches_and_budgets_covered=int(all(r['all_pair_covered'] for r in selected)),
                            )
                        )
            else:
                for scenario, event in epoch_covered.items():
                    coverage.append(
                        dict(
                            seed=seed_index,
                            checkpoint=checkpoint,
                            scenario=scenario,
                            all_batches_and_budgets_covered=int(event),
                        )
                    )
                    all_covered &= event
            print(
                'Seed %d/%d: trained all %d learners to %d episodes; evaluated both cases.'
                % (seed_index + 1, c['seeds'], n + 1, checkpoint),
                flush=True,
            )
            if logger is not None:
                logger.log_dispatch(rows[checkpoint_start:])
        if interval_study:
            for method in BOUND_METHODS:
                event = all(
                    r['all_batches_and_budgets_covered']
                    for r in coverage
                    if r['seed'] == seed_index and r['bound_method'] == method
                )
                coverage.append(
                    dict(
                        seed=seed_index,
                        checkpoint='all',
                        scenario='both',
                        bound_method=method,
                        all_batches_and_budgets_covered=int(event),
                    )
                )
        else:
            coverage.append(
                dict(
                    seed=seed_index, checkpoint='all', scenario='both', all_batches_and_budgets_covered=int(all_covered)
                )
            )
        if logger is not None:
            logger.finish_seed()
    return rows, training, coverage


def git_head():
    try:
        return subprocess.check_output(
            ['git', 'rev-parse', 'HEAD'], cwd=Path(__file__).resolve().parents[1], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--config', type=Path, default=CONFIGS / 'rl_grid10_paper.json')
    parser.add_argument('--output', type=Path, required=True, help='New directory; existing results are preserved')
    parser.add_argument('--log-every', type=int, default=5000, help='Training trips per local/W&B observation window')
    parser.add_argument('--diagnostic-every', type=int, default=50000, help='Trips between exact simulator diagnostics')
    add_tracking_arguments(parser)
    args = parser.parse_args()
    c = json.loads(args.config.read_text())
    validate(c)
    if args.log_every < 1 or args.diagnostic_every < args.log_every or args.diagnostic_every % args.log_every:
        parser.error('--diagnostic-every must be a positive multiple of --log-every')
    if args.wandb != 'disabled':
        try:
            importlib.metadata.version('wandb')
        except importlib.metadata.PackageNotFoundError:
            parser.error('Install wandb to enable W&B logging (pip install wandb)')
    if args.output.exists():
        parser.error('Output already exists; use a new directory')
    args.output.mkdir(parents=True)
    start = time.perf_counter()
    (args.output / 'config.json').write_text(json.dumps(c, indent=2) + '\n')
    runtime_config = dict(c)
    network_hash = None
    if c.get('network_file'):
        network_copy = args.output / 'network.json'
        shutil.copy2(network_path(c), network_copy)
        network_hash = hashlib.sha256(network_copy.read_bytes()).hexdigest()
        runtime_config['network_file'] = str(network_copy.resolve())
    hashes = snapshot_source(args.output)
    meta = dict(
        status='running',
        started_utc=datetime.now(timezone.utc).isoformat(),
        python=sys.version,
        python_executable=sys.executable,
        tracking=dict(
            wandb_mode=args.wandb,
            project=args.wandb_project,
            entity=args.wandb_entity,
            log_every=args.log_every,
            diagnostic_every=args.diagnostic_every,
        ),
        versions={k: importlib.metadata.version(k) for k in ('numpy', 'scipy', 'matplotlib')},
        source_sha256=hashes,
        config=c,
        git_head=git_head(),
        method='Goal-conditioned off-policy tabular Q-learning, gamma=1, estimated-cost assignment and checkpoint certification.',
        admissible_routes=(
            'Each action strictly reduces directed free-flow distance to the goal.'
            if c.get('network_file')
            else 'Each action reduces Manhattan distance to the current goal.'
        )
        + ' Approach and delivery are separate legs.',
        routing_benchmark='Unrestricted shortest expected-time directed path'
        if c.get('network_file')
        else 'Shortest expected-time admissible grid path',
        network_sha256=network_hash,
        estimate_source=(
            'All-pair sample mean (primary), reference learner min-action Q (ablation); no private policies or oracle costs.'
            if c.get('interval_study')
            else 'Reference learner min-action Q; no private driver policy, private Q value, or oracle cost.'
        ),
        coverage_scope=(
            'All pairs, two-sided known-support intervals; both cases, all batches, checkpoints and budgets per seed. Empirical extrema have no nominal guarantee.'
            if c.get('interval_study')
            else 'All selected pairs, both driver cases, all batches, checkpoints and evaluation budgets, per training seed.'
        ),
        units='Minutes; one payment unit per minute; training episode is one nontrivial origin-to-goal trip.',
        execution_scope='Expected cost conditional on compliance; no claim of equilibrium selection or convergence.',
    )
    metadata = args.output / 'metadata.json'
    metadata.write_text(json.dumps(meta, indent=2) + '\n')
    logger = None
    try:
        logger = RunLogger(
            args.output / 'telemetry',
            c,
            args.wandb,
            args.wandb_project,
            args.wandb_entity,
            args.wandb_group,
            provenance=dict(
                source_sha256=hashes,
                git_head=meta['git_head'],
                network_sha256=network_hash,
                log_every=args.log_every,
                diagnostic_every=args.diagnostic_every,
            ),
        )
        rows, training, coverage = experiment(
            runtime_config, args.output, logger, args.log_every, args.diagnostic_every
        )
        seeds, summary = aggregate(rows)
        for name, data in [
            ('raw.csv', rows),
            ('training.csv', training),
            ('coverage.csv', coverage),
            ('seed_summary.csv', seeds),
            ('summary.csv', summary),
            ('budgets.csv', budget_grid(rows, c)),
        ]:
            write_csv(args.output / name, data)
        if c.get('interval_study'):
            from .plot_intervals import plot_results
        else:
            from .plot_rl import plot_results
        plot_results(args.output)
        from .plot_tracking import plot_tracking

        plot_tracking(args.output / 'telemetry')
        final = [r for r in training if r['episodes'] == max(c['checkpoints'])]
        meta.update(
            status='complete',
            rows=len(rows),
            elapsed_seconds=time.perf_counter() - start,
            total_training_trips=sum(r['episodes'] for r in final),
            total_training_transitions=sum(r['transitions'] for r in final),
            total_reference_training_trips=c['seeds'] * max(c['checkpoints']),
            total_private_driver_training_trips=c['seeds'] * c['fleet'] * max(c['checkpoints']),
            total_evaluation_trips=c['seeds']
            * len(SCENARIOS)
            * c['batches']
            * len(c['checkpoints'])
            * c['fleet']
            * max(c['evaluation_counts']),
        )
        if c.get('interval_study'):
            pairs = c['seeds'] * len(SCENARIOS) * c['batches'] * len(c['checkpoints']) * c['fleet'] ** 2
            meta.update(
                total_evaluation_trips=pairs * max(c['evaluation_counts']),
                total_support_calibration_trips=pairs * max(c['evaluation_counts']),
                total_support_test_trips=pairs * c['support_test_count'],
                primary_benchmark='OPT(D)',
                primary_estimate='sample_mean',
            )
    except BaseException as exc:
        if logger is not None:
            logger.close(success=False)
        meta.update(status='failed', error=str(exc), elapsed_seconds=time.perf_counter() - start)
        metadata.write_text(json.dumps(meta, indent=2) + '\n')
        raise
    if logger is not None:
        logger.close()
    metadata.write_text(json.dumps(meta, indent=2) + '\n')
    print('Saved %d rows to %s (%.1f seconds)' % (len(rows), args.output, meta['elapsed_seconds']))


if __name__ == '__main__':
    main()
