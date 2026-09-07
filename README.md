# SafeSlice

Safe dynamic network slicing for SLA-preserving SDN. A lightweight contextual bandit chooses how
much bottleneck capacity to give an eMBB slice each second, and a deterministic guardrail stands
between it and the actuator so that URLLC latency risk is bounded by a mechanism that does not
depend on the learned model being right.

Three slices share one 10 Mbps bottleneck: URLLC (latency-critical), eMBB (throughput-hungry) and
Best Effort. Every second the controller sees telemetry, picks one of five absolute eMBB rate
caps, and is scored on a reward that trades throughput against SLA damage.

---

## Read this first: what is and is not measured here

**Everything in this repository runs on a simulator.** `net/sim_backend.py` is a per-queue token
bucket feeding a work-conserving shared link, written to match the semantics of an Open vSwitch
HTB queue. It has **not** been validated against real Open vSwitch, and its load-bearing
assumption — how leftover capacity is divided between backlogged queues — is a modelling choice,
not a measurement. `docs/DESIGN.md` section 2 says exactly what that assumption is and what
happens to the project's conclusions if reality behaves differently.

So: the numbers here are real, reproducible, and produced by code in this repository that
actually ran. They are simulator numbers. They are not testbed numbers and must not be quoted as
such.

### Not built

The original plan (`docs/PLAN.md`) had a Mininet/OVS/Ryu testbed track. None of it exists:

| Component | Status |
|---|---|
| `OvsCliBackend`, Mininet topology, `traffic/generator.py` (iperf3/ping) | not written |
| `experiments/measure_noise_floor.py`, testbed SLO derivation | not written |
| `RyuBackend`, `controller/ryu_app.py`, the Ryu feasibility check | not written |
| Simulator-vs-real-OVS comparison | not run |

The development machine is Windows with no Mininet, no OVS, no root, so none of it could have
been executed here, and unexecuted code that drives real network hardware is not worth much. The
abstraction it would plug into does exist and is unused-but-ready: `net/backend.py` defines the
`NetworkBackend` contract, and the runner, policies, guardrail and metrics are all written
against it rather than against the simulator.

---

## Headline result

Four scenarios, ten seeds, 320 runs. Differences are paired per seed, since a seed fixes the
traffic bit for bit.

**LinUCB learning online loses to a hand-tuned reactive threshold on three of the four
scenarios.** The same algorithm pre-trained on held-out seeds and frozen beats that threshold on
three of four — with *fewer* SLA violations, not more — and closes the gap to the best
phase-static allocation almost entirely on two of them. The whole difference between those two
results is the cost of exploring online.

| vs `threshold`, mean reward | adversarial | burst | ramp | sawtooth |
|---|---|---|---|---|
| `linucb` (online) | −0.073 | **+0.025** | −0.022 | −0.010 |
| `linucb_pretrained` | −0.005 (ties) | **+0.058** | **+0.002** | **+0.023** |

A context-free ε-greedy bandit does clearly worse than LinUCB on the adversarial scenario, so the
context vector is carrying real information rather than decorating the method.

The guardrail is active — up to 51.6 % intervention — but **it does not eliminate violations**. It
acts on the previous interval's telemetry, so it cannot prevent the first interval of a spike it
has not yet seen. What it guarantees is narrower and provable: the applied action is always inside
the allowed mask, whatever the policy asks for.

A 720-run sweep over the reward weights shows the ranking is **not stable**: three different
policies take first place depending on `w_sla` and `w_drop`, so no unqualified "policy X is best"
claim is supportable. What does survive the sweep is more interesting than a ranking — LinUCB is
the *only* policy whose behaviour responds to the weights at all. Raise the SLA penalty sixteen-
fold and it gives up 0.21 Mbps of goodput and cuts violations by 42 % on its own; `threshold` and
the static baselines are invariant to every digit, because they are fixed rules that never read
the reward.

Full numbers, including what each of these is not allowed to be read as, in
`docs/EXPERIMENTS.md` and `docs/REPORT_OUTLINE.md`.

---

## Quick start

```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\Activate.ps1
pip install -r requirements.txt
pytest -q
```

One run:

```bash
python experiments/run_experiment.py --policy linucb --scenario burst --seed 0
```

The whole thing, from nothing to tables and figures (about 90 minutes):

```bash
python experiments/run_suite.py      # every policy x scenario x seed -> results/raw/
python -m analysis.aggregate         # seed-averaged tables -> results/summary/
python -m analysis.plots             # five figures -> results/summary/figures/
```

**Do not run two experiment scripts at once.** `decision_latency_ms` is measured against a real
wall clock, and a second job on the same machine corrupts exactly the metric used to claim the
method is lightweight.

---

## The policies

| Name | What it is |
|---|---|
| `static_safe` | Lowest eMBB cap, always. Never violates, wastes capacity. The floor. |
| `static_equal` | The eMBB level nearest an equal three-way split. The naive reference. |
| `threshold` | Reactive: steps the cap down when smoothed latency is high, up when it is low. The non-learning competitor. |
| `epsilon_greedy` | **Context-free** bandit over arm means. The ablation that isolates whether the context vector is worth anything. |
| `linucb` | Disjoint LinUCB over the 9-feature context. The proposal. Learns online during the evaluated run. |
| `linucb_pretrained`, `epsilon_greedy_pretrained` | Trained on held-out seeds, then frozen and evaluated with exploration off. |
| `oracle` | Best **phase-static** allocation, chosen with knowledge of the trace. A reference line, not a method. Regret against it can legitimately be negative. |

Both an online and a converged number are reported for the learned policies. Reporting only the
converged one would delete the cost of exploration from the headline; reporting only the online
one would understate what the method does once it has been running a while.

---

## Layout

```
config/            default.yaml plus four traffic scenarios
net/               NetworkBackend contract, and the simulator behind it
traffic/           deterministic per-seed offered-load traces
agent/             context features, reward, guardrail, bandits, policies
analysis/          the CSV schema, metric definitions, aggregation, figures
experiments/       one run, the full grid, hyperparameter sweeps, sensitivity study
tests/             the suite; `pytest -q`
docs/              PLAN, DESIGN, EXPERIMENTS, SETUP, REPORT_OUTLINE
results/raw/       per-step logs, one CSV plus an env.json fingerprint per run
results/summary/   seed-averaged tables and figures, all regenerable
```

## Documentation

- `docs/DESIGN.md` — the design decisions, each with the reasoning and the limitation.
- `docs/EXPERIMENTS.md` — the protocol, the metric definitions, and the results tables.
- `docs/REPORT_OUTLINE.md` — what the write-up says, and the claims it is not allowed to make.
- `docs/SETUP.md` — environments. Its VM sections describe a testbed that was never built.
- `docs/PLAN.md` — the original six-week plan, kept as written for the record.

## Reproducibility

A run is fully determined by `(policy, scenario, seed)`. Trace seeds are derived with blake2b, not
Python's `hash()`, which is salted per process; a test pins that. Every run writes an `env.json`
with the interpreter, platform, CPU, git commit and a hash of the fully merged config. The suite
executes in a randomized order from a logged seed, recorded in
`results/summary/suite_order.json`.

Hyperparameters were tuned on `experiment.train_seeds`, which are disjoint from the evaluation
seeds. The pre-trained variants were trained on the same held-out seeds. Nothing was tuned or
trained on a seed it was later scored on.
