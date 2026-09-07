# EXPERIMENTS

The protocol, the metric definitions, and the results tables. Tables are populated only by
numbers that code in this repository actually produced. Everything else says `TBD`.

Status: Week 1a. Two static baselines on the simulator only. No testbed numbers exist yet.

---

## 1. Protocol

Built into `experiments/run_experiment.py` and (from Week 2) `experiments/run_suite.py`, so it
does not depend on anyone remembering it.

- **Determinism.** A run is fully determined by `(scenario, seed)`. The trace seed is derived with
  blake2b in `traffic/traces.py:trace_seed`, not with Python's `hash()`, which is salted per
  process and would silently break the "same seed means the same traffic bit for bit" guarantee.
  Pinned by `tests/test_sim_backend.py::test_trace_seed_is_process_independent`.
- **Seeds.** 10 per (policy, scenario) pair. `experiment.seeds` in `config/default.yaml`.
- **Duration.** 300 s per run, first 30 s discarded as warm-up. Both configurable at `run.duration_s`
  and `run.warmup_s`. Every reported metric goes through `analysis/metrics.py:post_warmup`.
- **Reporting.** Mean with 95% confidence interval across seeds. Never a single run.
- **Run order.** Randomized within the suite so thermal or memory drift does not correlate with
  policy. `TBD`, lands with `run_suite.py` in Week 2.
- **Environment fingerprint.** `results/raw/{run_id}/env.json` per run: kernel, OVS version,
  Mininet version, iperf3 version, Python version, CPU, load average at start, git commit, and a
  hash of the fully merged config.
- **Bandit reporting.** Both online (learning while running, the honest deployment number) and
  converged (pre-trained on other seeds, then evaluated) are reported. Week 3.

### Headline metrics

Defined once in `analysis/metrics.py:compute_run_metrics`.

1. URLLC RTT p50 / p95 / p99
2. SLA violation rate, fraction of post-warm-up control steps above `sla.hard_ms`
3. eMBB goodput, mean
4. Best Effort goodput, mean
5. Total packet drops
6. Guardrail intervention rate, fraction of steps where `applied_action != proposed_action`
7. Decision latency, mean and p99
8. Regret against the Oracle upper bound. Week 4.

**Naming caveat that must survive into the report.** `urllc_rtt_p99` is the 99th percentile of
the per-step p95 series. It is not a packet-level p99 and must not be labelled as one.

---

## 2. Reward weights

Current values, `config/default.yaml` under `reward`:

| Weight | Value | Role |
|---|---|---|
| `w_tput` | 1.00 | eMBB goodput, normalised by link capacity |
| `w_be` | 0.30 | Best Effort goodput. Nonzero so the action is a tradeoff, not a one-sided maximisation |
| `w_sla` | 1.00 | Hinge penalty above `sla.target_ms` |
| `w_viol` | 2.00 | Step penalty above `sla.hard_ms` |
| `w_drop` | 0.50 | Drops, normalised by packets the link carries in one interval |

### Sensitivity study

`TBD`. Week 4. Sweep planned over `w_sla` and, for the reason below, `w_drop`.

**`w_drop` must be swept, not just `w_sla`.** In the Week 1a runs, `total_drops` is dominated by
q1 tail drops caused by eMBB offering far more than its cap permits, so raising the eMBB cap
*reduces* drops. `w_drop` therefore behaves as a second throughput incentive opposing the SLA
term, not as a congestion penalty. If the conclusion turns out to be sensitive to `w_drop`, that
has to be reported. See `docs/DESIGN.md` section 5.

---

## 3. Week 1a results: static baselines, simulator only

**These are simulator numbers.** The simulator has not been validated against real OVS. Its
load-bearing assumption (`sim.excess_sharing: demand_proportional`) is unverified. Do not quote
these as testbed results.

Single seed (0), 300 s, 30 s warm-up, `SimBackend`. Ten-seed means with confidence intervals
arrive with `run_suite.py` in Week 2.

### StaticEqual (eMBB level 0.35) vs StaticSafe (eMBB level 0.20), `burst.yaml`, seed 0

| Metric | StaticEqual | StaticSafe |
|---|---|---|
| URLLC RTT p50 (ms) | 2.00 | 2.00 |
| URLLC RTT p95 (ms) | 3.34 | 2.00 |
| URLLC RTT p99 (ms) | 3.43 | 2.00 |
| SLA violation rate | 0.00 % | 0.00 % |
| eMBB goodput (Mbps) | 2.642 | 1.974 |
| BE goodput (Mbps) | 3.726 | 3.816 |
| Total drops (packets) | 84785 | 97767 |
| Guardrail intervention rate | 0.00 % | 0.00 % |
| Decision latency mean (ms) | 0.0726 | 0.0427 |
| Decision latency p99 (ms) | 0.2504 | 0.2140 |
| Mean reward | 0.1876 | 0.0947 |
| Link utilisation mean | 93.60 % | 87.83 % |

This is the expected direction: StaticSafe buys lower latency with lower eMBB goodput. It is the
"never violates, wastes capacity" reference the dynamic policies have to beat.

### StaticEqual across all four scenarios, seed 0

| Metric | burst | ramp | sawtooth | adversarial |
|---|---|---|---|---|
| URLLC RTT p50 (ms) | 2.00 | 2.00 | 2.00 | 2.00 |
| URLLC RTT p95 (ms) | 3.34 | 2.60 | 3.35 | 5.54 |
| URLLC RTT p99 (ms) | 3.43 | 2.63 | 3.50 | 6.26 |
| SLA violation rate | 0.00 % | 0.00 % | 0.00 % | 0.00 % |
| eMBB goodput (Mbps) | 2.642 | 2.907 | 2.998 | 3.381 |
| BE goodput (Mbps) | 3.726 | 3.419 | 3.521 | 3.397 |
| Total drops (packets) | 84785 | 50777 | 38281 | 127502 |
| Guardrail intervention rate | 0.00 % | 0.00 % | 0.00 % | 0.00 % |
| Decision latency mean (ms) | 0.0726 | 0.0897 | 0.0379 | 0.0283 |
| Decision latency p99 (ms) | 0.2504 | 0.5077 | 0.1114 | 0.1357 |
| Mean reward | 0.1876 | 0.2805 | 0.3203 | 0.1429 |
| Link utilisation mean | 93.60 % | 88.35 % | 87.41 % | 85.18 % |

### Reading these honestly

- **Zero SLA violations everywhere, and a zero guardrail intervention rate.** That is correct
  behaviour, not a bug. Both static baselines sit at levels 0.20 and 0.35, and the measured p95
  at those levels is 2.00 and 3.39 ms against a 7.0 ms hard limit (see `docs/DESIGN.md` section 3).
  Neither baseline can reach the levels that violate. The guardrail becomes active once a policy
  can select level 0.50 and above, which means Week 2 (`Threshold`) at the earliest.
- **Decision latency is dominated by measurement overhead.** Both policies are an array lookup;
  a mean of 0.03 to 0.09 ms is mostly `time.perf_counter` and Python dispatch. These are the
  floor against which LinUCB's cost gets compared, which is the point of measuring them now.
- **The drop counts are large and are mostly eMBB.** See the `w_drop` caveat in section 2.

### Guardrail plumbing check (not a result)

To confirm the mask reaches the actuator in a live run, `static_equal` was run on `adversarial`
seed 0 with `guardrail.warn_ms` forced to 2.5 ms and `safe_ceiling_index` to 0. Under those
artificial thresholds the guardrail intervened on 18.52 % of steps (50 `warn_mask_reselect`
events), URLLC p95 fell from 5.54 to 3.74 ms, and eMBB goodput fell from 3.381 to 3.192 Mbps.

That is the safety-for-throughput trade actually being paid end to end. **It is a plumbing check
under artificial thresholds, not a result, and must not appear in the report as one.**

---

## 4. Simulator versus real testbed

`TBD`. Week 1b. Identical trace through `SimBackend` and `OvsCliBackend`, comparing all three
`sim.excess_sharing` modes. This is the experiment that decides whether any simulator conclusion
in this document transfers.

## 5. Noise floor

`TBD`. Week 1b, from `experiments/measure_noise_floor.py`. Idle RTT over 60 s, p50/p95/p99. The
testbed SLO is derived from this, and it will differ from the simulator SLO.
