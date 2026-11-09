PYTHON ?= python3
RESULTS ?= results

.PHONY: test grid pooling exploration strategic reports figures numbers

test:
	$(PYTHON) -m unittest discover -s tests -t .

# Section IV-A: 20 seeds x 50 batches, Q-learning policies, K up to 4096 services per pair (hours on one core).
grid:
	$(PYTHON) -m certified_dispatch.run_rl --config configs/rl_grid10_paper.json --output $(RESULTS)/grid

# Section IV-B, six drivers: eight paired arms, 10^7 epochs, 5 seeds (about 2.5 h on 5 cores).
pooling:
	$(PYTHON) -m certified_dispatch.run_manhattan_scale --output $(RESULTS)/pooling --workers 5

# Section IV-B, twenty drivers: exploration arms, 3 x 10^6 epochs, 5 seeds (about 25 min on 5 cores).
exploration:
	$(PYTHON) -m certified_dispatch.run_manhattan_explore --output $(RESULTS)/exploration --workers 5

# Section IV-C: five price rules x three allocation rules, 10^7 epochs, 5 seeds (about 20 min on 5 cores).
strategic:
	$(PYTHON) -m certified_dispatch.run_manhattan_strategic --output $(RESULTS)/strategic --workers 5 \
	  --reference $(RESULTS)/pooling

# Rebuild the audited tables and diagnostic figures of completed runs without simulating.
reports:
	$(PYTHON) -m certified_dispatch.run_manhattan_scale --output $(RESULTS)/pooling --report-only
	$(PYTHON) -m certified_dispatch.run_manhattan_explore --output $(RESULTS)/exploration --report-only
	$(PYTHON) -m certified_dispatch.run_manhattan_strategic --output $(RESULTS)/strategic --report-only

figures:
	$(PYTHON) -m certified_dispatch.build_figures --pooling $(RESULTS)/pooling --exploration $(RESULTS)/exploration \
	  --strategic $(RESULTS)/strategic --output figures

numbers:
	$(PYTHON) -m certified_dispatch.paper_numbers --grid $(RESULTS)/grid --pooling $(RESULTS)/pooling \
	  --exploration $(RESULTS)/exploration --strategic $(RESULTS)/strategic
