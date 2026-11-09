"""Online certification from pooled observations on the Manhattan road graph (Section IV-B, six drivers).

python -m certified_dispatch.run_manhattan_scale --output results/pooling
Resume with the same command plus --resume; --report-only rebuilds the tables and figures.

Every cell runs the eight paired arms of `manhattan_kernel` with imposed compliance:
pooled, driver-specific, naive and oracle histories, each with the support radius and
with the variance-adaptive radius. Cells differ in the fleet size N (drivers and jobs
per epoch), the number of hubs h (|Q| = h^2 (h-1) contexts), the private-route share
rho and the seed. Horizons are matched in completed services: a fleet of N drivers
runs services/N epochs. The defaults are the configuration of the paper: six drivers,
eight hubs, rho = 0, 10^7 epochs and five seeds. --fleets, --hub-sweep and --rhos
extend the run to fleet and hub sweeps.
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
from .manhattan_kernel import ARMS, HISTORY, RADIUS, PRIVATE, BERNSTEIN, METRICS, allowances, run_chunk
from .manhattan_simulator import Simulator

STATE_KEYS = ('counts', 'totals', 'squares', 'means', 'lows', 'highs', 'origins', 'cumulative', 'coverage_min')


def cell_name(fleet, hubs, rho, seed):
    return f'fleet{fleet}_hubs{hubs}_rho{rho:g}_seed{seed}'


def epochs_for(services, fleet, block_size):
    """Epochs that complete about `services` services with `fleet` drivers, a multiple of the block."""
    return max(block_size, int(round(services / fleet / block_size)) * block_size)


def checkpoint_for(epochs, snapshots, block_size):
    """Checkpoint spacing giving about `snapshots` saved decision traces, a multiple of the block."""
    return max(block_size, int(round(epochs / snapshots / block_size)) * block_size)


def prepare_inputs_scale(network_file, hubs, rho, fleet, seed, path):
    """Public supports, hidden expected costs and fixed routes of one cell, saved as an npz file.

    The simulator draws one private routing policy per driver; the saved arrays have a
    leading driver dimension of `fleet`. The expected costs are recomputed from the saved
    edge paths as an independent check of the simulator.
    """
    c = dict(network_file=str(network_file), hubs=hubs, rho=rho, fleet=fleet)
    sim = Simulator(c, seed)
    h = len(sim.hubs)
    qs = h**3
    lo, hi = np.zeros(qs), np.zeros(qs)
    truth = np.zeros((fleet, qs))
    for s, source in enumerate(sim.hubs):
        for p, pickup in enumerate(sim.hubs):
            for d, dropoff in enumerate(sim.hubs):
                q = (s * h + p) * h + d
                lo[q] = sim.lower[source, pickup] + sim.lower[pickup, dropoff]
                hi[q] = sim.upper[source, pickup] + sim.upper[pickup, dropoff]
                truth[:, q] = sim.true_context((source, pickup, dropoff))
    policies = [sim.common, *sim.private]
    assert len(policies) == fleet + 1
    lengths = np.array([[[len(policy[s, g]) for g in sim.hubs] for s in sim.hubs] for policy in policies])
    paths = np.full((fleet + 1, h, h, max(1, int(lengths.max()))), -1, np.int64)
    for policy in range(fleet + 1):
        for s, source in enumerate(sim.hubs):
            for g, goal in enumerate(sim.hubs):
                edges = policies[policy][source, goal]
                paths[policy, s, g, : len(edges)] = edges
    # Independently recompute expected route costs from the saved edge paths.
    expected = np.zeros((fleet + 1, h, h))
    for policy in range(fleet + 1):
        for s in range(h):
            for g in range(h):
                expected[policy, s, g] = sim.roads.mean[paths[policy, s, g, : lengths[policy, s, g]]].sum()
    for i in range(fleet):
        for s in range(h):
            for p in range(h):
                for d in range(h):
                    oracle = (1 - rho) * (expected[0, s, p] + expected[0, p, d]) + rho * (
                        expected[i + 1, s, p] + expected[i + 1, p, d]
                    )
                    assert abs(oracle - truth[i, (s * h + p) * h + d]) < 1e-9
    assert np.all(truth.max(axis=0) - truth.min(axis=0) <= rho * (hi - lo) + 1e-8)
    atomic_npz(
        path,
        support_lo=lo,
        support_hi=hi,
        true_contexts=truth,
        paths=paths,
        lengths=lengths,
        free=sim.roads.free,
        delay=sim.roads.delay,
        hubs=sim.hubs,
        fleet=fleet,
        rho=rho,
        private_weights=np.array(sim.policy_weights),
    )


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


def chunk(config, cell, start, stop, state, inputs):
    allowance = allowances(cell['rho'], inputs['support_lo'], inputs['support_hi'], inputs['true_contexts'])
    return run_chunk(
        cell['seed'],
        start,
        stop,
        config['block_size'],
        cell['rho'],
        config['alpha'],
        config['eta'],
        config['payment_budget'],
        config['assignment_tolerance'],
        cell['hubs'],
        PRIVATE,
        BERNSTEIN,
        allowance,
        *[state[k] for k in STATE_KEYS],
        *[inputs[k] for k in INPUT_KEYS],
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
                certified_fraction={a: float(state['cumulative'][j, 0] / stop) for j, a in enumerate(ARMS)},
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
        for rho in args.rhos:
            for fleet in args.fleets:
                epochs = epochs_for(args.services, fleet, args.block_size)
                cells.append(
                    dict(
                        name=cell_name(fleet, args.hubs, rho, seed),
                        seed=seed,
                        rho=rho,
                        fleet=fleet,
                        hubs=args.hubs,
                        epochs=epochs,
                        checkpoint_every=checkpoint_for(epochs, args.snapshots, args.block_size),
                        sweep='fleet',
                    )
                )
        for hubs in args.hub_sweep:
            fleet = args.fleets[0]
            epochs = epochs_for(args.services, fleet, args.block_size)
            cells.append(
                dict(
                    name=cell_name(fleet, hubs, args.rhos[0], seed),
                    seed=seed,
                    rho=args.rhos[0],
                    fleet=fleet,
                    hubs=hubs,
                    epochs=epochs,
                    checkpoint_every=checkpoint_for(epochs, args.snapshots, args.block_size),
                    sweep='hubs',
                )
            )
    # Longest cells first: the kernel cost per epoch grows like the cube of the fleet.
    cells.sort(key=lambda c: -(c['fleet'] ** 3 * c['epochs']))
    return cells


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--fleets', type=int, nargs='+', default=[6], help='fleet sizes; the paper uses 6')
    p.add_argument('--hubs', type=int, default=8, help='hubs of the fleet sweep')
    p.add_argument(
        '--hub-sweep',
        type=int,
        nargs='*',
        default=[],
        help='extra hub counts run with the first fleet and the first rho',
    )
    p.add_argument('--rhos', type=float, nargs='+', default=[0.0], help='private-route shares; the paper uses 0')
    p.add_argument(
        '--services', type=float, default=60_000_000, help='completed services per cell; epochs = services/fleet'
    )
    p.add_argument('--seeds', type=int, default=5)
    p.add_argument('--base-seed', type=int, default=20264800)
    p.add_argument('--block-size', type=int, default=10_000)
    p.add_argument('--snapshots', type=int, default=50, help='saved decision traces per cell')
    p.add_argument('--workers', type=int, default=5)
    p.add_argument('--resume', action='store_true')
    p.add_argument('--prepare-only', action='store_true')
    p.add_argument('--report-only', action='store_true')
    args = p.parse_args()
    output = args.output.resolve()
    if args.report_only:
        from .report_manhattan_scale import report

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
            args.services < 1
            or args.seeds < 1
            or args.workers < 1
            or args.block_size < 1
            or args.snapshots < 1
            or not all(5 <= f <= 512 for f in args.fleets)
            or len(set(args.fleets)) != len(args.fleets)
            or not all(2 <= h <= 32 for h in [args.hubs, *args.hub_sweep])
            or args.hubs in args.hub_sweep
            or len(set(args.rhos)) != len(args.rhos)
            or not all(0 <= r <= 1 for r in args.rhos)
        ):
            p.error('Invalid sizes: fleets in [5, 512] and distinct, hubs in [2, 32] and distinct, rhos in [0, 1]')
        output.mkdir(parents=True, exist_ok=False)
        (output / 'inputs').mkdir()
        hashes = snapshot_source(output)
        shutil.copy2(NETWORK, output / 'network.json')
        config = dict(
            services=args.services,
            fleets=args.fleets,
            hubs=args.hubs,
            hub_sweep=args.hub_sweep,
            rhos=args.rhos,
            seeds=args.seeds,
            base_seed=args.base_seed,
            block_size=args.block_size,
            snapshots=args.snapshots,
            alpha=0.05,
            eta=0.01,
            payment_budget=1.0,
            assignment_tolerance=1.0,
            network_file=str(output / 'network.json'),
            arms=ARMS,
            history=HISTORY,
            radius=RADIUS,
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
            behavior='Imposed compliance on every epoch; no exploration; eight paired arms of the final kernel.',
            design=(
                'Fleet sweep at fixed hubs with horizons matched in completed services (epochs = '
                'services/fleet); hub sweep at the first fleet and rho with the same services.'
            ),
            guarantee=(
                'Coverage per seed and arm over all contexts and counts: pooled/support and '
                'oracle/support at every rho, private arms at every rho, variance arms of pooled '
                'histories and naive arms at rho = 0 only.'
            ),
            allowance=(
                'pooled: rho (b_q - a_q) from the public support; private and naive: 0; oracle: '
                'the true across-driver spread of the context (diagnostic).'
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
        from .report_manhattan_scale import report

        report(output)
        manifest.update(status='complete', completed_utc=datetime.now(timezone.utc).isoformat())
    except BaseException as exc:
        manifest.update(status='interrupted' if isinstance(exc, KeyboardInterrupt) else 'failed', error=repr(exc))
        raise
    finally:
        atomic_json(output / 'metadata.json', manifest)


if __name__ == '__main__':
    main()
