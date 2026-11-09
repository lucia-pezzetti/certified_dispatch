# Certifying Dispatch of Self-Interested Drivers via Dual Pricing

Code and data for the numerical results of the paper *Certifying Dispatch of
Self-Interested Drivers via Dual Pricing* (submitted to the European Control
Conference 2027).

A platform dispatches self-interested drivers in a network.
It recommends one job to each driver and posts a price per job to make drivers follow the recommendation.
Since the drivers' service costs are not known, the platform estimates them from completed
services and keeps confidence intervals around the estimates. The paper derives
prices from an optimal dual of the estimated assignment problem, adjusted so
that every recommended cost is covered with a margin, and a certificate that
bounds the assignment loss and the extra payment.

## Installation

Install dependencies with

```sh
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/python -m unittest discover -s tests -t .
```

## Reproducing the paper

Run from the repository root. Results go to `results/`, and `--resume` continues
an interrupted run. Times are for five worker processes on a laptop.

### Assignment accuracy and participation (Section IV-A)

Six drivers on a stochastic 10 x 10 grid with Q-learning routing policies, 20
seeds and 50 dispatch instances each: estimated-cost assignments and unadjusted
dual prices are scored at the true costs (sequential, many hours).

```sh
python -m certified_dispatch.run_rl --config configs/rl_grid10_paper.json --output results/grid
```

### Online certification from pooled observations (Section IV-B)

Six drivers on the southern Manhattan graph with eight hubs, 10^7 epochs and
five seeds: pooled and driver-specific histories with the support and the
variance-adaptive radius (about 15 minutes).

```sh
python -m certified_dispatch.run_manhattan_scale --output results/pooling --workers 5
```

### Exploration with twenty drivers (Section IV-B)

Twenty drivers, 3 x 10^6 epochs and five seeds: on uncertified epochs the
optimistic arms dispatch the lower-bound assignment, capped or uncapped, and the
random arm a uniformly random assignment (about 20 minutes).

```sh
python -m certified_dispatch.run_manhattan_explore --output results/exploration --workers 5
```

### Strategic job requests (Section IV-C)

Six drivers who reply to the posted menu, with contested jobs allocated by
priority, uniformly at random, or to the cheapest requester, under five price
rules, 10^7 epochs and five seeds (about 15 minutes).

```sh
python -m certified_dispatch.run_manhattan_strategic --output results/strategic --workers 5 \
    --reference results/pooling
```

### Figures and numbers

Data and TikZ sources of Figures 1 to 3, and every number quoted in Section IV
next to the value stated in the paper.

```sh
python -m certified_dispatch.build_figures --pooling results/pooling --exploration results/exploration \
    --strategic results/strategic --output figures
python -m certified_dispatch.paper_numbers --grid results/grid --pooling results/pooling \
    --exploration results/exploration --strategic results/strategic
```

The same commands are available as `make grid`, `make pooling`, `make exploration`,
`make strategic`, `make figures` and `make numbers`.

## What a run contains

```
results/<run>/
  config.json, metadata.json   parameters, versions, SHA-256 of sources, network and inputs
  source/certified_dispatch/   snapshot of the package at the start of the run
  network.json, inputs/        the road graph and the prepared cells
  <cell>/blocks_*.npz          per-block metrics and decision traces; checkpoint.npz for --resume
  *.csv, audit.json, figures/  audited summary tables and diagnostic plots
```

## Licenses

The code is released under the MIT License (see `LICENSE`). The road networks in
`data/` are derivative databases of OpenStreetMap data, © OpenStreetMap
contributors, available under the Open Database License 1.0; see
`data/README.md` for the extraction and the synthetic delay model.
