# SafeSlice: safe dynamic network slicing for SLA-preserving SDN.
#
# Every target here runs on the simulator and needs no root, no Mininet and no Open vSwitch.
# There are deliberately NO testbed targets: the real backends are not implemented (see the
# "Not built" section of README.md). A make target that shelled out to `mn` and failed would
# read as a broken build rather than as work that was never done.
#
# On Windows use the commands directly; `make` is not usually present. Each recipe below is a
# single command line, so copying it into PowerShell works.

PYTHON ?= python
SEEDS  ?=
POLICY ?= linucb
SCEN   ?= burst
SEED   ?= 0

.PHONY: help
help:
	@echo "SafeSlice targets"
	@echo ""
	@echo "  make test         run the test suite (fast, no network stack needed)"
	@echo "  make run          one run:  POLICY=$(POLICY) SCEN=$(SCEN) SEED=$(SEED)"
	@echo "  make suite        the full grid: every policy x scenario x seed (~30 min)"
	@echo "  make suite-resume continue an interrupted suite"
	@echo "  make tune         alpha and epsilon sweeps on the training seeds (~11 min)"
	@echo "  make sensitivity  reward-weight sensitivity study (~40 min, resumable)"
	@echo "  make tables       seed-averaged tables from results/raw"
	@echo "  make figures      the five report figures"
	@echo "  make all          suite, then tables, then figures"
	@echo "  make clean        remove generated results and caches"
	@echo ""
	@echo "IMPORTANT: do not run two experiment targets at once. decision_latency_ms is"
	@echo "measured against a real wall clock, so a second job on the same machine"
	@echo "corrupts it. See experiments/run_suite.py."

.PHONY: setup
setup:
	$(PYTHON) -m pip install -r requirements.txt

.PHONY: test
test:
	$(PYTHON) -m pytest -q

.PHONY: run
run:
	$(PYTHON) experiments/run_experiment.py --policy $(POLICY) --scenario $(SCEN) --seed $(SEED)

.PHONY: suite
suite:
	$(PYTHON) experiments/run_suite.py $(if $(SEEDS),--seeds $(SEEDS),)

.PHONY: tune
tune:
	$(PYTHON) experiments/tune.py

.PHONY: sensitivity
# --resume keeps the rows already in results/summary/sensitivity_runs.csv. Safe to re-run after
# an interruption; it will not redo finished cells and will not append to a stale study.
sensitivity:
	$(PYTHON) experiments/sensitivity.py --resume

.PHONY: suite-resume
suite-resume:
	$(PYTHON) experiments/run_suite.py --resume

.PHONY: tables
tables:
	$(PYTHON) -m analysis.aggregate

.PHONY: figures
figures:
	$(PYTHON) -m analysis.plots

.PHONY: all
all: suite tables figures

.PHONY: clean
clean:
	rm -rf results/summary/figures results/summary/*.csv results/summary/*.json
	rm -rf .pytest_cache
	find . -name __pycache__ -type d -prune -exec rm -rf {} +

# Deliberately NOT cleaned by `make clean`: results/raw, which holds the per-step logs every
# table and figure is recomputed from. Regenerating it costs half an hour, and a clean target
# that silently deletes half an hour of measurements is a footgun. Remove it by hand.
