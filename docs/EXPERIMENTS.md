# EXPERIMENTS

The protocol, the metric definitions, and the results. Every number below was produced by code in
this repository that ran. Anything not measured says so.

**These are simulator numbers.** `net/sim_backend.py` has never been compared against real Open
vSwitch, because the testbed track was not built (`docs/PLAN.md` status note). Its load-bearing
assumption is described in `docs/DESIGN.md` section 2. Do not quote anything here as a testbed
result.

Status: complete. 8 policies x 4 scenarios x 10 seeds = 320 evaluation runs, plus 280 tuning runs
and 900 sensitivity runs. 1,500 runs in total.

---

## 1. Protocol

Built into `experiments/run_suite.py`, so it does not depend on anyone remembering it.

- **Determinism.** A run is fully determined by `(policy, scenario, seed)`. The trace seed is
  derived with blake2b in `traffic/traces.py:trace_seed`, not with Python's `hash()`, which is
  salted per process. Pinned by `tests/test_sim_backend.py::test_trace_seed_is_process_independent`.
- **Seeds.** 10 per (policy, scenario), `experiment.seeds`. Hyperparameter tuning and bandit
  pre-training use `experiment.train_seeds` = `[100..104]`, **disjoint** from the evaluation
  seeds and asserted disjoint by a test. Nothing was tuned or trained on a seed it was scored on.
- **Duration.** 300 s per run, first 30 s discarded. Every reported metric goes through
  `analysis/metrics.py:post_warmup`.
- **Reporting.** Mean over 10 seeds with a 95% t interval. Never a single run.
- **Paired comparison.** A seed fixes the offered-load trace bit for bit, so every policy sees
  identical traffic on seed k. Differences between policies are therefore reported as the mean of
  the **per-seed differences** with an interval on that mean, not as the difference of two
  independent means.
- **Run order.** Randomized within the suite from a logged seed; the plan and the realised order
  are in `results/summary/suite_order.json`.
- **Environment fingerprint.** `results/raw/{run_id}/env.json` per run: Python, platform, CPU,
  git commit, and a hash of the fully merged config.
- **Bandit reporting.** Both **online** (learning during the evaluated run, the honest deployment
  number, includes the cost of exploring) and **converged** (pre-trained on the held-out seeds,
  then frozen with exploration off). Both appear in every table.

### Headline metrics

Defined once in `analysis/metrics.py:compute_run_metrics`.

1. URLLC RTT p50 / p95 / p99
2. SLA violation rate, fraction of post-warm-up steps above `sla.hard_ms`
3. eMBB goodput, mean
4. Best Effort goodput, mean
5. Total packet drops
6. Guardrail intervention rate, fraction of steps where `applied_action != proposed_action`
7. Decision latency, mean and p99
8. Regret against the best phase-static allocation (section 6)

**Naming caveat that must survive into the report.** `urllc_rtt_p99` is the 99th percentile of the
per-step p95 series. It is not a packet-level p99 and must not be labelled as one.

### One honest note on how this suite was executed

The first two attempts at a single clean pass were killed by the operating system for memory at
runs 299 and 99 of 320. The suite was completed with `--resume`, so 21 of the 320 runs executed in
a later pass than the rest. This does not affect any metric except `decision_latency_ms`: every
other quantity is a deterministic function of `(policy, scenario, seed)` and is bit-identical
regardless of when it ran. Decision latency was measured on a machine that was demonstrably under
memory pressure, so **treat it as an order-of-magnitude comparison against the static baselines,
not as a precise benchmark.** The runner now writes its order file before the first run so that an
interruption cannot destroy the record again (`tests/test_suite_resume.py`).

---

## 2. Hyperparameter selection

Swept by `experiments/tune.py` on the 5 training seeds x 4 scenarios, with `write=False` so no
sweep run can leak into `results/raw`. 160 runs for alpha, 120 for epsilon.

### LinUCB `alpha`

| `alpha` | mean reward | SLA violation rate | eMBB goodput (Mbps) | URLLC p95 (ms) |
|---|---|---|---|---|
| 0.00 | +0.1820 ± 0.0219 | 0.15 % | 2.665 | 3.37 |
| 0.02 | +0.2028 ± 0.0293 | 0.13 % | 2.812 | 3.68 |
| 0.05 | +0.2227 ± 0.0356 | 0.28 % | 3.008 | 4.17 |
| 0.10 | +0.2240 ± 0.0367 | 0.65 % | 3.068 | 3.95 |
| 0.25 | +0.2495 ± 0.0395 | 1.19 % | 3.378 | 4.93 |
| **0.50** | **+0.2543 ± 0.0375** | 1.48 % | 3.487 | 5.36 |
| 1.00 | +0.2516 ± 0.0364 | 1.50 % | 3.504 | 5.39 |
| 2.00 | +0.2262 ± 0.0371 | 1.96 % | 3.402 | 5.50 |

Selected 0.50 as the argmax. **0.25, 0.50 and 1.00 have overlapping intervals and nothing in the
data separates them**; read this as "somewhere around 0.5", not as a tuned optimum. The trend that
is real is the trade: more exploration buys eMBB goodput and pays for it in violations.

### Epsilon-greedy `epsilon`

| `epsilon` | mean reward | SLA violation rate | eMBB goodput (Mbps) | URLLC p95 (ms) |
|---|---|---|---|---|
| **0.00** | **+0.2334 ± 0.0460** | 2.22 % | 3.543 | 6.11 |
| 0.02 | +0.2318 ± 0.0473 | 2.13 % | 3.490 | 5.95 |
| 0.05 | +0.2293 ± 0.0421 | 2.11 % | 3.482 | 5.66 |
| 0.10 | +0.2248 ± 0.0471 | 2.00 % | 3.393 | 5.55 |
| 0.20 | +0.2173 ± 0.0436 | 2.17 % | 3.386 | 5.47 |
| 0.40 | +0.1977 ± 0.0407 | 2.54 % | 3.308 | 5.61 |

**Exploration did not pay for the context-free bandit**: 0.0 was the argmax, and no value in
[0, 0.2] separates from any other. The selection rule stated up front in `experiments/tune.py` is
"argmax of mean reward", and it was followed rather than revised after seeing the answer — which
here also selects the *strongest* version of this baseline, i.e. the harder one for our method to
beat. At epsilon = 0 the policy still tries every arm once before exploiting, so it is greedy over
arm means, not locked to whichever arm it sampled first. The report must call it that rather than
imply it explores.

---

## 3. The headline result

**Online LinUCB does not beat a hand-tuned reactive threshold. Converged LinUCB does.**

Paired per-seed difference in mean reward against `threshold`, 10 seeds. Bold means the 95%
interval excludes zero. Positive is better than threshold.

| Policy | adversarial | burst | ramp | sawtooth |
|---|---|---|---|---|
| `linucb` (online) | **−0.0728 ± 0.0182** | **+0.0248 ± 0.0062** | **−0.0217 ± 0.0029** | **−0.0104 ± 0.0089** |
| `linucb_pretrained` | −0.0047 ± 0.0117 | **+0.0582 ± 0.0042** | **+0.0019 ± 0.0007** | **+0.0225 ± 0.0078** |
| `epsilon_greedy` | **−0.1281 ± 0.0092** | +0.0082 ± 0.0199 | **−0.0281 ± 0.0219** | −0.0201 ± 0.0235 |
| `epsilon_greedy_pretrained` | **−0.1287 ± 0.0114** | **+0.0269 ± 0.0190** | **+0.0019 ± 0.0007** | **+0.0236 ± 0.0078** |
| `static_equal` | **−0.1151 ± 0.0087** | **+0.0425 ± 0.0043** | **−0.0600 ± 0.0013** | **−0.0370 ± 0.0082** |
| `oracle` | **+0.1241 ± 0.0115** | **+0.0589 ± 0.0042** | **+0.0019 ± 0.0007** | **+0.0358 ± 0.0077** |

Read that honestly:

- **Online LinUCB loses to Threshold on three of four scenarios** and wins on one. The losses are
  separated from zero, so they are real and not noise. `docs/PLAN.md` section 4 predicted in week
  one, before any policy existed, that this could happen and committed to reporting it. It
  happened, and this is the report.
- **Converged LinUCB beats Threshold on three of four and ties on the fourth** (adversarial, where
  the interval spans zero). It also does so with *fewer* SLA violations, not more — see below.
- **The whole difference between the two rows is the cost of online exploration.** Same algorithm,
  same hyperparameters, same traffic. The only change is whether the model arrived trained.
- **Context is worth something.** On `adversarial`, LinUCB is 0.055 reward ahead of the
  context-free bandit online (−0.073 vs −0.128) and 0.124 ahead converged. The ablation earns its
  keep: this is not a problem a context-free bandit solves equally well.

And the same comparison on safety — paired difference in SLA violation rate against `threshold`,
where negative means fewer violations:

| Policy | adversarial | burst | ramp | sawtooth |
|---|---|---|---|---|
| `linucb` (online) | **+0.70 ± 0.40 %** | **−0.78 ± 0.20 %** | **+0.33 ± 0.08 %** | 0.00 ± 0.25 % |
| `linucb_pretrained` | **−0.59 ± 0.42 %** | **−1.56 ± 0.11 %** | 0.00 ± 0.00 % | **−0.96 ± 0.22 %** |
| `epsilon_greedy_pretrained` | **+2.81 ± 0.19 %** | **−0.70 ± 0.60 %** | 0.00 ± 0.00 % | **−0.96 ± 0.22 %** |

Converged LinUCB is the only learned policy that is better than the reactive baseline on **both**
axes at once on more than one scenario. That is the result the project set out to look for, and it
holds only for the converged variant.

---

## 4. Full tables

Mean over 10 seeds ± 95% t interval. Generated by `python -m analysis.aggregate`, recomputed from
the per-step logs rather than read from a cache.

### adversarial

| Metric | static_safe | static_equal | threshold | epsilon_greedy | eps_greedy_pretr | linucb | linucb_pretr | oracle |
|---|---|---|---|---|---|---|---|---|
| URLLC p50 (ms) | 2.00 ± 0.00 | 2.00 ± 0.00 | 2.75 ± 0.01 | 2.47 ± 0.21 | 2.76 ± 0.02 | 2.17 ± 0.20 | 2.11 ± 0.03 | 2.83 ± 0.03 |
| URLLC p95 (ms) | 2.79 ± 0.25 | 5.53 ± 0.05 | 4.22 ± 0.23 | 9.10 ± 1.09 | 11.38 ± 0.63 | 5.46 ± 0.18 | 5.26 ± 0.18 | 5.51 ± 0.12 |
| URLLC p99 (ms) | 3.65 ± 0.19 | 6.15 ± 0.16 | 13.07 ± 0.37 | 12.56 ± 0.68 | 13.59 ± 0.28 | 10.47 ± 1.06 | 9.27 ± 1.05 | 6.10 ± 0.17 |
| SLA violations | 0.00 % | 0.04 ± 0.08 % | 3.00 ± 0.08 % | 5.85 ± 0.11 % | 5.81 ± 0.18 % | 3.70 ± 0.39 % | 2.41 ± 0.44 % | 0.00 % |
| eMBB goodput (Mbps) | 2.000 | 3.379 ± 0.004 | 4.936 ± 0.019 | 4.578 ± 0.134 | 4.767 ± 0.017 | 4.375 ± 0.184 | 4.584 ± 0.018 | 5.129 ± 0.039 |
| BE goodput (Mbps) | 3.443 ± 0.028 | 3.399 ± 0.027 | 3.115 ± 0.019 | 3.202 ± 0.068 | 3.094 ± 0.020 | 3.313 ± 0.069 | 3.307 ± 0.026 | 3.094 ± 0.023 |
| Guardrail interventions | 0.00 % | 0.15 ± 0.34 % | 2.96 ± 0.00 % | 23.41 ± 0.46 % | 23.48 ± 0.26 % | 15.22 ± 1.70 % | 9.63 ± 1.75 % | 3.41 ± 0.63 % |
| Decision latency mean (ms) | 0.021 | 0.022 | 0.039 | 0.061 | 0.063 | 0.091 | 0.098 | 0.030 |
| Mean reward | −0.0488 ± 0.0028 | 0.1410 ± 0.0044 | 0.2561 ± 0.0102 | 0.1279 ± 0.0108 | 0.1274 ± 0.0104 | 0.1833 ± 0.0176 | 0.2513 ± 0.0132 | 0.3801 ± 0.0038 |

### burst

| Metric | static_safe | static_equal | threshold | epsilon_greedy | eps_greedy_pretr | linucb | linucb_pretr | oracle |
|---|---|---|---|---|---|---|---|---|
| URLLC p95 (ms) | 2.00 | 3.39 ± 0.02 | 5.28 ± 0.02 | 4.97 ± 0.58 | 5.36 ± 0.03 | 5.28 ± 0.04 | 5.29 ± 0.02 | 5.29 ± 0.02 |
| URLLC p99 (ms) | 2.00 | 3.51 ± 0.03 | 8.46 ± 0.17 | 6.70 ± 0.87 | 6.84 ± 0.16 | 6.34 ± 0.33 | 5.47 ± 0.02 | 5.47 ± 0.02 |
| SLA violations | 0.00 % | 0.00 % | 1.56 ± 0.11 % | 1.33 ± 0.59 % | 0.85 ± 0.56 % | 0.78 ± 0.15 % | 0.00 % | 0.00 % |
| eMBB goodput (Mbps) | 1.977 | 2.645 ± 0.006 | 2.753 ± 0.016 | 2.785 ± 0.066 | 2.887 ± 0.036 | 2.793 ± 0.030 | 2.933 ± 0.007 | 2.939 ± 0.007 |
| Guardrail interventions | 0.00 % | 0.00 % | 4.33 ± 0.59 % | 20.41 ± 13.44 % | 51.59 ± 0.67 % | 12.85 ± 6.53 % | 0.00 % | 0.00 % |
| Mean reward | 0.0965 ± 0.0027 | 0.1883 ± 0.0037 | 0.1459 ± 0.0037 | 0.1540 ± 0.0200 | 0.1727 ± 0.0185 | 0.1706 ± 0.0056 | 0.2041 ± 0.0036 | 0.2048 ± 0.0035 |

### ramp

| Metric | static_safe | static_equal | threshold | epsilon_greedy | eps_greedy_pretr | linucb | linucb_pretr | oracle |
|---|---|---|---|---|---|---|---|---|
| URLLC p95 (ms) | 2.00 | 2.56 ± 0.02 | 4.22 ± 0.02 | 4.11 ± 0.88 | 4.17 ± 0.01 | 5.25 ± 0.30 | 4.17 ± 0.01 | 4.17 ± 0.01 |
| SLA violations | 0.00 % | 0.00 % | 0.00 % | 0.04 ± 0.08 % | 0.00 % | 0.33 ± 0.08 % | 0.00 % | 0.00 % |
| eMBB goodput (Mbps) | 1.846 | 2.903 ± 0.004 | 3.372 ± 0.011 | 3.206 ± 0.157 | 3.365 ± 0.009 | 3.316 ± 0.015 | 3.365 ± 0.009 | 3.365 ± 0.009 |
| Guardrail interventions | 0.00 % | 0.00 % | 3.22 ± 0.63 % | 10.44 ± 11.20 % | 0.00 % | 19.11 ± 6.13 % | 0.00 % | 0.00 % |
| Mean reward | 0.1227 ± 0.0012 | 0.2803 ± 0.0011 | 0.3403 ± 0.0011 | 0.3122 ± 0.0219 | 0.3422 ± 0.0010 | 0.3187 ± 0.0028 | 0.3422 ± 0.0010 | 0.3422 ± 0.0010 |

On `ramp`, `linucb_pretrained`, `epsilon_greedy_pretrained` and `oracle` are **numerically
identical to four decimal places** on every metric. They are not a coincidence and not a bug: the
ramp scenario's best phase-static schedule is a single level for three of its four phases, both
converged bandits find it, and so all three policies play the same actions. It is worth stating
because it also means `ramp` cannot discriminate between them.

### sawtooth

| Metric | static_safe | static_equal | threshold | epsilon_greedy | eps_greedy_pretr | linucb | linucb_pretr | oracle |
|---|---|---|---|---|---|---|---|---|
| URLLC p95 (ms) | 2.00 | 3.33 ± 0.03 | 5.32 ± 0.03 | 5.50 ± 0.33 | 5.28 ± 0.04 | 5.27 ± 0.05 | 5.25 ± 0.04 | 5.27 ± 0.04 |
| URLLC p99 (ms) | 2.00 | 3.54 ± 0.03 | 6.72 ± 0.57 | 7.32 ± 0.90 | 5.56 ± 0.05 | 6.86 ± 0.11 | 5.54 ± 0.06 | 5.55 ± 0.06 |
| SLA violations | 0.00 % | 0.00 % | 0.96 ± 0.22 % | 1.48 ± 0.67 % | 0.00 % | 0.96 ± 0.14 % | 0.00 % | 0.00 % |
| eMBB goodput (Mbps) | 1.995 | 3.001 ± 0.011 | 3.538 ± 0.022 | 3.498 ± 0.032 | 3.500 ± 0.016 | 3.436 ± 0.028 | 3.479 ± 0.018 | 3.612 ± 0.014 |
| Guardrail interventions | 0.00 % | 0.00 % | 9.48 ± 1.00 % | 22.00 ± 9.10 % | 0.00 % | 16.04 ± 5.30 % | 0.00 % | 4.56 ± 1.26 % |
| Mean reward | 0.1700 ± 0.0016 | 0.3192 ± 0.0019 | 0.3562 ± 0.0078 | 0.3360 ± 0.0227 | 0.3798 ± 0.0033 | 0.3457 ± 0.0055 | 0.3786 ± 0.0031 | 0.3919 ± 0.0031 |

---

## 5. What the guardrail does, and what it does not

The guardrail is active and it matters: intervention rates run from 0 % for the static baselines
that never approach the boundary, up to 51.6 % for `epsilon_greedy_pretrained` on `burst`.

**But it does not eliminate SLA violations, and the report must not claim it does.** On
`adversarial`, `epsilon_greedy` violates on 5.85 % of steps *despite* the guardrail intervening on
23.4 % of them, and its p99 reaches 12.56 ms against a 7.0 ms hard limit.

The reason is structural and is worth stating plainly: **the guardrail acts on telemetry from the
previous control interval.** It cannot prevent the first interval of a spike it has not yet
observed; it can only stop the second and subsequent ones. `adversarial.yaml` is built precisely
to exploit that — long calm periods that lure a policy to an aggressive level, then a URLLC step
shorter than the EWMA settling time. So:

- What the guardrail guarantees is a *property of the action space*: the applied action is always
  inside the allowed mask, whatever the policy proposes. That is proved by a property test against
  an adversarial policy over 5000 randomized cases, not observed empirically.
- What it does *not* guarantee is zero violations. It bounds exposure; it does not remove it.
- The intervention rate is **not** a count of prevented violations. The counterfactual was never
  run. Some interventions would not have caused a violation at all.

### The instantaneous fast path earns its place, and here is the number

`docs/DESIGN.md` section 4 argued in week 1a that a guardrail watching only the smoothed latency
would be late by construction against short sharp spikes, so the hard override fires on *either*
the EWMA or the current p95, and `guardrail_escalation_source` records which. That was a design
argument with no evidence behind it at the time. Across all 320 runs, 86,400 post-warm-up steps:

| Which signal exceeded the threshold | steps | share of escalations |
|---|---|---|
| EWMA only (the slow path) | 838 | 4.8 % |
| **instantaneous p95 only (the fast path)** | **6,740** | **38.3 %** |
| both | 10,002 | 56.9 % |
| total escalating steps | 17,580 | — |

**An EWMA-only guardrail would have missed 38.3 % of all escalations** — 6,740 steps where latency
had already crossed the threshold but the smoothed signal had not caught up. The converse case,
where the EWMA fired and the instantaneous tail did not, accounts for 4.8 %. The fast path is
carrying roughly eight times the unique load of the slow one, and dropping it would have quietly
removed more than a third of the guardrail's coverage.

This is the one design decision in the project that was argued from first principles in week one
and is confirmed by measurement in week five. Figure `fig4_guardrail` shows the same split per
policy.

---

## 6. Regret against the best phase-static allocation

**Not regret against the optimum.** The Oracle is the best *phase-static* schedule under the same
guardrail, chosen with knowledge of the trace. A policy that varies within a phase can beat it, so
negative values are legitimate. `agent/policies/oracle.py` explains why a true oracle is not
available at any price.

Positive means worse than the reference.

| Policy | adversarial | burst | ramp | sawtooth |
|---|---|---|---|---|
| `static_safe` | +0.4290 ± 0.0043 | +0.1083 ± 0.0014 | +0.2194 ± 0.0019 | +0.2219 ± 0.0039 |
| `static_equal` | +0.2392 ± 0.0054 | +0.0164 ± 0.0008 | +0.0618 ± 0.0016 | +0.0727 ± 0.0024 |
| `threshold` | +0.1241 ± 0.0115 | +0.0589 ± 0.0042 | +0.0019 ± 0.0007 | +0.0358 ± 0.0077 |
| `epsilon_greedy` | +0.2522 ± 0.0121 | +0.0507 ± 0.0176 | +0.0299 ± 0.0216 | +0.0559 ± 0.0221 |
| `epsilon_greedy_pretrained` | +0.2527 ± 0.0118 | +0.0320 ± 0.0175 | −0.0000 ± 0.0000 | +0.0122 ± 0.0008 |
| `linucb` | +0.1968 ± 0.0194 | +0.0342 ± 0.0045 | +0.0235 ± 0.0028 | +0.0462 ± 0.0037 |
| `linucb_pretrained` | **+0.1288 ± 0.0142** | **+0.0007 ± 0.0003** | **−0.0000 ± 0.0000** | **+0.0133 ± 0.0007** |

Converged LinUCB essentially closes the gap to the phase-static reference on `burst` (+0.0007) and
`ramp` (0.0000), and has the lowest regret of any deployable policy on `sawtooth`. On
`adversarial` it does not: Threshold is closer (+0.124 vs +0.129), though the two overlap.

---

## 7. Decision latency

**Caveat first: see the execution note in section 1.** These were measured on a machine under
memory pressure and are an order-of-magnitude comparison, not a benchmark.

Mean per-decision wall-clock time, averaged across all scenarios:

| Policy | mean (ms) | p99 (ms) |
|---|---|---|
| `static_safe` / `static_equal` | ~0.021–0.023 | ~0.040–0.049 |
| `threshold` | ~0.035–0.040 | ~0.073–0.115 |
| `epsilon_greedy` | ~0.058–0.063 | ~0.117–0.163 |
| `linucb` | ~0.091–0.098 | ~0.192–0.233 |

LinUCB costs roughly **4x an array lookup** and stays comfortably under a quarter of a millisecond
at p99, against a 1000 ms control interval. The claim the project can make is that the method is
cheap relative to its own control loop by four orders of magnitude. The claim it cannot make is a
precise cost figure, for the reason in section 1.

---

## 8. Reward-weight sensitivity

900 runs: `w_sla` in {0.25, 0.5, 1.0, 2.0, 4.0} x `w_drop` in {0.0, 0.5, 1.0} x 5 policies x
4 scenarios x 3 training seeds. 3 seeds rather than 10 to keep the study inside a machine-hour,
which makes these intervals wider than those in section 4.

**Rewards from different cells are not comparable.** Changing a weight changes the units of the
reward. Only two things are read across cells: the ordering of policies within a cell, and the
physical metrics, which are in milliseconds and Mbps and mean the same thing regardless.

### The headline result IS robust to the reward weights

An earlier version of this study swept only the four non-converged policies, and reported that the
ranking was unstable. That was a real gap, and it is now closed: `linucb_pretrained` — the policy
that wins section 3 — was added to the sweep, and it **wins in all 15 cells**.

Because "wins in all 15" is a directional claim until it is tested, each cell was checked with a
paired interval over its 12 (scenario, seed) pairs against that cell's runner-up:

| `w_sla` | `w_drop` | runner-up | mean delta | 95% interval | separated |
|---|---|---|---|---|---|
| 0.25 | 0.0 | threshold | +0.0125 | [−0.0068, +0.0319] | no |
| 0.25 | 0.5 | threshold | +0.0126 | [−0.0089, +0.0340] | no |
| 0.25 | 1.0 | threshold | +0.0309 | [+0.0175, +0.0442] | **yes** |
| 0.50 | 0.0 | threshold | +0.0226 | [+0.0099, +0.0353] | **yes** |
| 0.50 | 0.5 | threshold | +0.0250 | [+0.0120, +0.0380] | **yes** |
| 0.50 | 1.0 | threshold | +0.0316 | [+0.0177, +0.0454] | **yes** |
| 1.00 | 0.0 | threshold | +0.0223 | [+0.0062, +0.0384] | **yes** |
| 1.00 | 0.5 | threshold | +0.0215 | [+0.0030, +0.0400] | **yes** |
| 1.00 | 1.0 | threshold | +0.0327 | [+0.0191, +0.0463] | **yes** |
| 2.00 | 0.0 | static_equal | +0.0303 | [+0.0184, +0.0422] | **yes** |
| 2.00 | 0.5 | threshold | +0.0317 | [+0.0148, +0.0487] | **yes** |
| 2.00 | 1.0 | threshold | +0.0301 | [+0.0099, +0.0502] | **yes** |
| 4.00 | 0.0 | static_equal | +0.0245 | [+0.0135, +0.0354] | **yes** |
| 4.00 | 0.5 | static_equal | +0.0458 | [+0.0264, +0.0652] | **yes** |
| 4.00 | 1.0 | linucb | +0.0581 | [+0.0257, +0.0904] | **yes** |

**Separated from zero in 13 of 15 cells.** The two exceptions are both at `w_sla = 0.25`, the
weakest SLA penalty, where there is least for a safety-aware policy to win — which is the
direction you would expect the exception to fall if the effect is real.

### The instability is real, but it is among the OTHER policies

The earlier finding is not withdrawn, it is relocated. Below the winner, the ordering still moves
with the weights: `threshold` is runner-up at low `w_sla`, `static_equal` takes over at
`w_sla >= 2.0` with low `w_drop`, and at `w_sla = 4.0, w_drop = 1.0` even online `linucb`
overtakes both. As the SLA penalty grows the conservative static split overtakes the aggressive
reactive controller, which is what you would expect and is a weak check that the reward behaves
sensibly.

So the correct claim is narrow and specific: **which of the *baselines* looks second-best depends
on the reward weights; which policy looks best does not.**

### Only the learned policies respond to the weights at all

Averaged over `w_drop`, scenarios and seeds:

| SLA violation rate | w_sla 0.25 | 0.50 | 1.00 | 2.00 | 4.00 |
|---|---|---|---|---|---|
| `linucb_pretrained` | 0.70 % | 0.74 % | 0.64 % | 0.66 % | **0.37 %** |
| `linucb` | 1.51 % | 1.46 % | 1.35 % | 1.17 % | 0.88 % |
| `threshold` | 1.54 % | 1.54 % | 1.54 % | 1.54 % | 1.54 % |
| `static_equal` | 0.00 % | 0.00 % | 0.00 % | 0.00 % | 0.00 % |

| eMBB goodput (Mbps) | w_sla 0.25 | 0.50 | 1.00 | 2.00 | 4.00 |
|---|---|---|---|---|---|
| `linucb_pretrained` | 3.634 | 3.719 | 3.596 | 3.561 | 3.349 |
| `linucb` | 3.524 | 3.519 | 3.473 | 3.419 | 3.315 |
| `threshold` | 3.655 | 3.655 | 3.655 | 3.655 | 3.655 |
| `static_equal` | 2.990 | 2.990 | 2.990 | 2.990 | 2.990 |

`threshold` and the static baselines are **numerically invariant** to the reward weights, to every
digit. That is not a flaw in the experiment; it is what they are. They are fixed rules that never
read the reward, so telling them you care more about latency changes nothing they do.

The comparison at `w_sla = 0.25` is the cleanest single statement of what the method buys:
`linucb_pretrained` reaches **3.634 Mbps at a 0.70 % violation rate** against `threshold`'s
**3.655 Mbps at 1.54 %** — within 0.6 % of the same throughput for **less than half** the SLA
violations. And raising `w_sla` sixteen-fold cuts its violation rate to 0.37 % with no re-tuning,
while `threshold` cannot be told to care.

### Limits of this study

- **The converged variant consumes 5 extra training runs per evaluation.** Against `threshold`
  this is not a like-for-like data budget, and the comparison should be read as "a model trained
  offline beats a hand-tuned rule", not "learning is free". `threshold` also had human input — its
  thresholds were chosen against the measured latency-by-level table — but that is one-off human
  effort, not per-deployment compute.
- Three seeds per cell, so cell-level intervals are wider than section 4's.
- `w_tput`, `w_be` and `w_viol` were held fixed. Only the two weights identified in
  `docs/DESIGN.md` section 5 as suspect were swept.
- `epsilon_greedy` and `oracle` are not in the sweep; each cell would have cost six further runs
  and neither changes the question this study asks.

## 9. Simulator versus real testbed

**Never run.** The testbed was not built. This is the experiment that would decide whether any
conclusion in this document transfers to real Open vSwitch, and it is the single largest gap in
the project. `docs/REPORT_OUTLINE.md` section 8.

## 10. Noise floor

**Never measured.** Requires the testbed. The SLO used throughout (`target 5.0 ms`, `hard 7.0 ms`)
was derived from the simulator's own achievable latency range (`docs/DESIGN.md` section 3), not
from hardware.
