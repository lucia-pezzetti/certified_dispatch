# Certifying Dispatch of Self-Interested Drivers via Dual Pricing

Code and data for the numerical results of the paper *Certifying Dispatch of
Self-Interested Drivers via Dual Pricing* (submitted to the European Control
Conference 2027).

A platform recommends one job to each driver and posts a price per job. The
drivers' service costs are not known; the platform estimates them from completed
services and keeps confidence intervals around the estimates. The paper derives
prices from an optimal dual of the estimated assignment problem, adjusted so
that every recommended cost is covered with a margin, and a certificate that
bounds the assignment loss and the extra payment. Section IV of the paper tests
this loop in three simulated settings, all reproduced by the code here:

| Section | Experiment | Entry point |
|---|---|---|
| IV-A | Assignment accuracy and participation on a stochastic 10 x 10 grid with Q-learning routing policies | `run_rl.py` |
| IV-B | Online certification from pooled observations on the southern Manhattan road graph, six drivers | `run_manhattan_scale.py` |
| IV-B | Exploration on uncertified epochs with twenty drivers | `run_manhattan_explore.py` |
| IV-C | Strategic job requests, five price rules and three allocation rules | `run_manhattan_strategic.py` |

Every run saves its configuration, the exact versions of its dependencies, a
snapshot and the SHA-256 digests of the source files, the road network it used,
its random seeds and a decision trace per checkpoint. The report step rechecks
the saved decisions with an independent implementation of the assignment,
pricing and certificate computations before it writes the summary tables.

## Repository layout

```
certified_dispatch/        Python package
  platform.py              assignment, dual prices, price menus, certificates (SciPy)
  optimistic_kernel.py     compiled dispatch loop with exploration (Numba); Hungarian solver,
                           counter-based random streams, confidence radii
  manhattan_kernel.py      compiled dispatch loop without exploration: pooled, driver-specific,
                           naive and oracle histories, support and variance-adaptive radii
  manhattan_simulator.py   hubs, public supports, fixed routing policies and hidden expected
                           costs on a road graph
  model.py, rl.py          grid road model, tabular Q-learning, frozen policies
  networks.py              directed road networks and the admissible routing action class
  interval_study.py        all-pair intervals and payment comparisons of the grid study
  strategic_game.py        one-shot replies, allocation rules and regrets of the strategic drivers
  run_rl.py                Section IV-A runner
  run_manhattan_scale.py   Section IV-B runner, six drivers (report: report_manhattan_scale.py)
  run_manhattan_explore.py Section IV-B runner, twenty drivers (report: report_manhattan_explore.py)
  run_manhattan_strategic.py  Section IV-C runner and report
  build_figures.py         data and PGFPlots sources of Figures 1 to 3 from completed runs
  paper_numbers.py         every number quoted in Section IV, computed from completed runs
  tracking.py, diagnostics.py, plot_*.py   logging and diagnostic plots of the grid study
  templates/               the TikZ layouts of the three figures of the paper
configs/                   grid configurations (paper and smoke sizes)
data/                      road networks (OpenStreetMap derivatives, see data/README.md)
figures/                   data and TikZ sources of the paper's figures, as generated from the runs
tests/                     57 unit tests
```

## Installation

Python 3.9 or later with NumPy, SciPy, Numba and Matplotlib:

```sh
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/python -m unittest discover -s tests -t .
```

The runs in the paper used Python 3.11.7, NumPy 1.26.4, SciPy 1.16.2, Numba
0.59.0 and Matplotlib 3.8.0 on macOS; `metadata.json` of every run records the
versions actually used. The compiled kernels take about a minute to build the
first time a runner starts.

## Reproducing the paper

All commands run from the repository root and write to `results/`, which Git
ignores. Each runner refuses to overwrite an existing output directory and
resumes an interrupted run with `--resume`. Wall-clock times are for five
worker processes on a laptop.

```sh
make pooling       # Section IV-B, six drivers: about 15 minutes
make exploration   # Section IV-B, twenty drivers: about 20 minutes
make strategic     # Section IV-C: about 15 minutes (checks itself against the pooling run)
make figures       # data and TikZ sources of Figures 1 to 3 in figures/
make numbers       # the numbers quoted in Section IV, next to the values stated in the paper
make grid          # Section IV-A: the most expensive run, many hours on one core
```

### Section IV-A, assignment accuracy and participation

```sh
python -m certified_dispatch.run_rl --config configs/rl_grid10_paper.json --output results/grid
```

Twenty training seeds, 50 location batches per seed, six drivers and six jobs,
one reference and six private Q learners frozen at six checkpoints, and up to
4,096 simulated services per driver and job pair. The run is sequential and
simulates about 3.6 billion services in total; `configs/rl_grid10_smoke.json`
is a two-minute version with the same code path. The four numbers of Section
IV-A are means at the final checkpoint and the largest observation count in
`results/grid/summary.csv`, rows with `scenario = independent_current`,
`estimate_method = sample_mean` and `bound_method = known_support`:

| Statement in the paper | Metric | Price rule |
|---|---|---|
| Estimated assignment optimal for the true costs in 95.8 % of instances | `assignment_optimal` | any |
| Mean assignment loss 5e-5 minutes per driver | `assignment_loss` | any |
| Unadjusted prices leave 10.1 % of drivers with a negative payoff | `negative_surplus_fraction` | `dual_base` |
| Mean shortfall of the worst driver 2.9e-3 minutes | `priority_regret` | `dual_base` |

`python -m certified_dispatch.paper_numbers --grid results/grid` prints them.

### Section IV-B, online certification from pooled observations

```sh
python -m certified_dispatch.run_manhattan_scale --output results/pooling --workers 5
```

Six drivers and six jobs per epoch on the southern Manhattan graph (2,087 nodes,
4,250 directed links), eight hubs giving 448 recurring contexts, 10^7 epochs,
five seeds, starting from empty histories. The eight arms share the demand and
the service noise of every epoch: pooled, driver-specific, naive and oracle
histories, each with the support radius and with the variance-adaptive
(empirical Bernstein) radius. Figure 1(a) and the certification rates of 43.9 %
and 99.2 % come from the `private_variance` and `pooled_variance` arms over the
final 10^5 epochs (`summary.csv`, window `terminal_100k`). Options `--fleets`,
`--hub-sweep` and `--rhos` extend the run to the fleet and hub sweeps and to
heterogeneous private routes.

### Section IV-B, exploration with twenty drivers

```sh
python -m certified_dispatch.run_manhattan_explore --output results/exploration --workers 5
```

Twenty drivers, eight hubs, 3 x 10^6 epochs and five seeds, so that the number
of completed services matches the six-driver run. On every uncertified epoch
the optimistic arms dispatch the lower-bound assignment with the prices of its
own dual, either capped at the payment budget or uncapped, and the random arm
dispatches a uniformly random assignment at its estimated-cost dual prices
shifted to cover the assigned upper costs. Figure 1(b), Figure 2 and the
exploration paragraph of Section IV-B use the arms `pooled_none`,
`pooled_optimistic`, `pooled_optimistic_relaxed` and `pooled_random_relaxed`
(`summary.csv`, `threshold_summary.csv`, `terminal_seed_summary.csv`).

The published run was produced in two steps that the current code merges: the
random arms of the original sweep explored whenever the optimistic arm with the
same cap explored, and the uncapped random arm was then rerun from empty
histories with its own certificate as the trigger. `--coupled-random` restores
the original rule; the default reproduces the published arm bit for bit.

### Section IV-C, strategic job requests

```sh
python -m certified_dispatch.run_manhattan_strategic --output results/strategic --workers 5 \
    --reference results/pooling
```

The six-driver loop in which every driver replies once to the posted menu,
assuming the others comply, and contested jobs are resolved by recommendation
priority, uniformly at random, or in favour of the requester with the lowest
estimated cost. Five price rules (adjusted dual, unadjusted dual, direct upper
cost plus margin, shifted direct, flat) times three allocation rules, 10^7
epochs, five seeds; only executed services update the pooled history. Figure 3
and the numbers of Section IV-C come from `summary_arms.csv` (whole-run means
per arm) and `curves.csv` (per-block means). With `--reference`, the report also
verifies that the adjusted-dual arm under recommendation priority reproduces the
pooled variance-adaptive arm of the pooling run block by block, as the theory
predicts while the intervals cover the costs; the test suite repeats this check
on the first 200,000 epochs when a pooling run is present.

### Figures and numbers

```sh
python -m certified_dispatch.build_figures --pooling results/pooling --exploration results/exploration \
    --strategic results/strategic --output figures
python -m certified_dispatch.paper_numbers --grid results/grid --pooling results/pooling \
    --exploration results/exploration --strategic results/strategic
```

`figures/data/*.csv` are the exact plotted values, `figures/figure{1,2,3}.tikz`
the PGFPlots sources with the layouts of the paper, and `figures/figure{1,2,3}.tex`
standalone wrappers (`latexmk -pdf figure1.tex` inside `figures/`). The paper
inputs the TikZ files from its main file, so its copies reference the data as
`figures/tikz/data/`; `--data-prefix figures/tikz/data/` writes that variant.
The committed `figures/` directory was generated from the runs behind the
submitted paper. Panel (a) of Figure 1 in the paper was drawn from an earlier
run of the same configuration: the plotted means coincide with the regenerated
ones, and the 95 % half-widths agree to within 2e-4 percentage points.

## What a run contains

```
results/<run>/
  config.json              every parameter, the arms, the metrics and the cells
  metadata.json            status, timestamps, Python and package versions, SHA-256 of the
                           sources, the network and the prepared inputs, protocol notes
  source/certified_dispatch/   snapshot of the package at the start of the run
  network.json             the road graph used
  inputs/<cell>.npz        public supports, hidden expected costs and fixed routes of each cell
  <cell>/blocks_*.npz      per-block sums of every metric and the decision trace at each checkpoint
  <cell>/checkpoint.npz    sufficient statistics of the histories, for --resume
  figure_data.csv, terminal_seed_summary.csv, summary.csv, thresholds.csv, audit.csv, figures/
```

The compiled kernels use a counter-based random stream (SplitMix64) indexed by
seed, epoch and purpose, so every arm of a cell sees the same demand and the
same service noise, and a resumed run is identical to an uninterrupted one. The
simulator's exact expected costs are read only to score decisions and to check
coverage; the platform code never receives them.

## Licenses

The code is released under the MIT License (see `LICENSE`). The road networks in
`data/` are derivative databases of OpenStreetMap data, © OpenStreetMap
contributors, available under the Open Database License 1.0; see
`data/README.md` for the extraction and the synthetic delay model.
