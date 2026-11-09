"""Exploration on uncertified epochs with a larger fleet (Section IV-B, twenty drivers).

python -m certified_dispatch.run_manhattan_explore --output results/exploration
Resume with the same command plus --resume; --report-only rebuilds the tables and figures.

Every cell runs the eight arms of `optimistic_kernel` (pooled none / optimistic /
random, private none / optimistic, naive optimistic, and pooled optimistic / random
with the separate exploration cap) with the empirical Bernstein radius. The
optimistic rule dispatches the lower-bound assignment sigma_L with the menu of L's own
dual on every epoch where the certificate fails, provided its extra payment h_L does
not exceed the gate: the payment budget for the capped arm, the exploration cap for the
uncapped arm (the default cap of 1e9 never blocks). The random arms draw a uniformly
random assignment whenever their own certificate fails and shift the estimated-cost
dual prices to cover the assigned upper costs plus eta; --coupled-random instead makes
them explore exactly when the optimistic arm with the same gate explores. Cells differ
in the number of hubs and the horizon; the fleet is fixed. The defaults are the
configuration of the paper: twenty drivers, eight hubs, 3 x 10^6 epochs, five seeds.
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
from .common import NETWORK, INPUT_KEYS, atomic_json, atomic_npz, sha, snapshot_source, check_source_unchanged
from .optimistic_kernel import ARMS, METRICS, run_chunk
from .run_manhattan_scale import prepare_inputs_scale, checkpoint_for

STATE_KEYS = ('counts', 'totals', 'squares', 'means', 'lows', 'highs', 'origins', 'cumulative', 'coverage_min')
PRIVATE_ARMS = (3, 4)
GUARANTEED_AT_POSITIVE_RHO = (
    0,
    1,
    2,
    3,
    4,
    6,
    7,
)  # naive (arm 5) has no guarantee at rho > 0; variance arms of pooled histories are guaranteed at rho = 0 only


def guaranteed(arm_index, rho):
    if arm_index in PRIVATE_ARMS:
        return True
    return rho == 0.0


def cell_name(fleet, hubs, rho, seed):
    return f'explore_fleet{fleet}_hubs{hubs}_rho{rho:g}_seed{seed}'


def fresh_state(inputs, fleet, hubs, seed):
    shape = (len(ARMS), fleet, hubs**3)
    lo = np.broadcast_to(inputs['support_lo'], shape).copy()
    hi = np.broadcast_to(inputs['support_hi'], shape).copy()
    origins = np.tile(np.random.default_rng(seed).integers(hubs, size=fleet), (len(ARMS), 1))
    return dict(
        counts=np.zeros(shape, np.int64),
        totals=np.zeros(shape),
        squares=np.zeros(shape),
        means=(lo + hi) / 2,
        lows=lo,
        highs=hi,
        origins=origins,
        cumulative=np.zeros((len(ARMS), len(METRICS))),
        coverage_min=np.full(len(ARMS), np.inf),
    )


def chunk(config, cell, start, stop, state, inputs, *, only_arm=-1, independent_random=None):
    if independent_random is None:
        independent_random = bool(config.get('independent_random', False))
    return run_chunk(
        cell['seed'],
        start,
        stop,
        cell['epochs'],
        config['block_size'],
        cell['rho'],
        config['alpha'],
        config['eta'],
        config['payment_budget'],
        config['assignment_tolerance'],
        config['exploration_budget'],
        cell['hubs'],
        1,
        *[state[k] for k in STATE_KEYS],
        *[inputs[k] for k in INPUT_KEYS],
        only_arm=only_arm,
        independent_random=independent_random,
    )


def prepare_cell(output, config, cell):
    path = Path(output) / 'inputs' / f"{cell['name']}.npz"
    if not path.exists():
        prepare_inputs_scale(config['network_file'], cell['hubs'], cell['rho'], cell['fleet'], cell['seed'], path)
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
        state = fresh_state(inputs, cell['fleet'], cell['hubs'], cell['seed'])
    started = time.perf_counter()
    resumed_at = start
    status = dict(status='running', pid=os.getpid(), epoch=start, **cell)
    atomic_json(status_path, status)
    ix = {m: j for j, m in enumerate(METRICS)}
    try:
        while start < cell['epochs']:
            stop = min(start + cell['checkpoint_every'], cell['epochs'])
            stats, trace = chunk(config, cell, start, stop, state, inputs)
            atomic_npz(directory / f'blocks_{stop:010d}.npz', start=start, stop=stop, stats=stats, trace=trace)
            atomic_npz(checkpoint, epoch=stop, **state)
            elapsed = time.perf_counter() - started
            status.update(
                epoch=stop,
                elapsed_session_seconds=elapsed,
                epochs_per_second=(stop - resumed_at) / elapsed,
                certified_fraction={
                    a: float(state['cumulative'][j, ix['certified']] / stop) for j, a in enumerate(ARMS)
                },
                explored_fraction={a: float(state['cumulative'][j, ix['explored']] / stop) for j, a in enumerate(ARMS)},
                minimum_coverage_margin=state['coverage_min'].tolist(),
            )
            atomic_json(status_path, status)
            print(
                f"{cell['name']}: {stop:,}/{cell['epochs']:,} epochs; "
                f"{status['epochs_per_second']:.0f} eight-arm epochs/s",
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


def build_cells(args):
    cells = []
    for seed in range(args.base_seed, args.base_seed + args.seeds):
        for hubs, epochs in zip(args.hubs, args.epochs):
            cells.append(
                dict(
                    name=cell_name(args.fleet, hubs, args.rho, seed),
                    seed=seed,
                    rho=args.rho,
                    fleet=args.fleet,
                    hubs=hubs,
                    epochs=epochs,
                    checkpoint_every=checkpoint_for(epochs, args.snapshots, args.block_size),
                )
            )
    cells.sort(key=lambda c: -c['epochs'])
    return cells


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--fleet', type=int, default=20)
    p.add_argument('--hubs', type=int, nargs='+', default=[8])
    p.add_argument('--epochs', type=int, nargs='+', default=[3_000_000], help='one horizon per hub count')
    p.add_argument('--rho', type=float, default=0.0)
    p.add_argument('--seeds', type=int, default=5)
    p.add_argument('--base-seed', type=int, default=20264800)
    p.add_argument('--block-size', type=int, default=10_000)
    p.add_argument('--snapshots', type=int, default=50)
    p.add_argument('--payment-budget', type=float, default=1.0)
    p.add_argument(
        '--exploration-cap',
        type=float,
        default=1e9,
        help='gate of the relaxed optimistic arm; 1e9 explores every uncertified epoch',
    )
    p.add_argument(
        '--coupled-random',
        action='store_true',
        help='random arms follow the exploration schedule of the optimistic arm with the same gate '
        'instead of their own certificate',
    )
    p.add_argument('--workers', type=int, default=5)
    p.add_argument('--resume', action='store_true')
    p.add_argument('--prepare-only', action='store_true')
    p.add_argument('--report-only', action='store_true')
    args = p.parse_args()
    output = args.output.resolve()
    if args.report_only:
        from .report_manhattan_explore import report

        report(output)
        return
    if args.resume:
        config = json.loads((output / 'config.json').read_text())
        manifest = json.loads((output / 'metadata.json').read_text())
        check_source_unchanged(manifest)
        for filename, digest in manifest.get('input_sha256', {}).items():
            if sha(output / 'inputs' / filename) != digest:
                raise ValueError('Saved simulation inputs changed: ' + filename)
        if sha(output / 'network.json') != manifest['network_sha256']:
            raise ValueError('Saved road graph changed')
    else:
        if (
            len(args.hubs) != len(args.epochs)
            or len(set(args.hubs)) != len(args.hubs)
            or not all(2 <= h <= 32 for h in args.hubs)
            or not 6 <= args.fleet <= 512
            or any(e < args.block_size or e % args.block_size for e in args.epochs)
            or not 0 <= args.rho <= 1
            or args.seeds < 1
            or args.workers < 1
            or args.snapshots < 1
        ):
            p.error('Invalid sizes: one horizon per hub count, horizons multiples of the block, fleet in [6, 512]')
        output.mkdir(parents=True, exist_ok=False)
        (output / 'inputs').mkdir()
        hashes = snapshot_source(output)
        shutil.copy2(NETWORK, output / 'network.json')
        config = dict(
            fleet=args.fleet,
            hubs=args.hubs,
            epochs_by_hubs=args.epochs,
            rho=args.rho,
            rhos=[args.rho],
            seeds=args.seeds,
            base_seed=args.base_seed,
            block_size=args.block_size,
            snapshots=args.snapshots,
            alpha=0.05,
            eta=0.01,
            payment_budget=args.payment_budget,
            exploration_budget=args.exploration_cap,
            assignment_tolerance=1.0,
            radius='bernstein',
            independent_random=not args.coupled_random,
            network_file=str(output / 'network.json'),
            arms=ARMS,
            metrics=METRICS,
            cells=build_cells(args),
        )
        atomic_json(output / 'config.json', config)
        manifest = dict(
            status='preparing',
            created_utc=datetime.now(timezone.utc).isoformat(),
            source_sha256=hashes,
            network_sha256=sha(output / 'network.json'),
            python=sys.version,
            versions={k: importlib.metadata.version(k) for k in ('numpy', 'scipy', 'numba', 'matplotlib')},
            behavior=(
                'Compliant dispatch of the estimated optimum when certified; on uncertified epochs the '
                'optimistic arms dispatch the lower-bound assignment sigma_L with the menu of L\'s dual '
                'shifted by h_L, when h_L is within the gate (payment budget, or the exploration cap for '
                'the relaxed arms); random arms draw a uniformly random assignment '
                + (
                    'when the optimistic arm with the same gate explores.'
                    if args.coupled_random
                    else 'whenever their own certificate fails.'
                )
            ),
            gate=f"primary arms: h_L <= {args.payment_budget}; relaxed arms: h_L <= {args.exploration_cap:g}",
            radius=(
                'Empirical Bernstein (Maurer and Pontil 2009, Thm 4) on X and width-X at budget '
                'alpha/(2 family n(n+1)) each; n = 1 Hoeffding.'
            ),
            guarantee=(
                'Coverage per seed and arm over all contexts and counts, for every arm at rho = 0; '
                'private arms at every rho.'
            ),
            rng='Counter-based SplitMix64; shared epoch/driver uniforms across arms.',
            demand='Distance-selected hubs on the full southern Manhattan graph; synthetic delays/demand.',
        )
    atomic_json(output / 'metadata.json', manifest)
    pending = [cell for cell in config['cells'] if not (output / 'inputs' / f"{cell['name']}.npz").exists()]
    if pending:
        print(f'Preparing {len(pending)} cells with {args.workers} workers', flush=True)
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            for future in as_completed([pool.submit(prepare_cell, str(output), config, cell) for cell in pending]):
                print('Prepared ' + future.result(), flush=True)
    manifest['input_sha256'] = {p.name: sha(p) for p in (output / 'inputs').glob('*.npz')}
    manifest['status'] = 'prepared' if args.prepare_only else 'running'
    atomic_json(output / 'metadata.json', manifest)
    if args.prepare_only:
        return
    try:
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            futures = [pool.submit(run_cell, str(output), cell) for cell in config['cells']]
            for future in as_completed(futures):
                print('Completed ' + future.result(), flush=True)
        from .report_manhattan_explore import report

        report(output)
        manifest.update(status='complete', completed_utc=datetime.now(timezone.utc).isoformat())
    except BaseException as exc:
        manifest.update(status='interrupted' if isinstance(exc, KeyboardInterrupt) else 'failed', error=repr(exc))
        raise
    finally:
        atomic_json(output / 'metadata.json', manifest)


if __name__ == '__main__':
    main()
