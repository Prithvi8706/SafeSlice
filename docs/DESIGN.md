# DESIGN

Design decisions, with the reasoning and the limitations. Every number in this file that is
presented as a measurement was produced by code in this repository that actually ran; anything
not yet measured says `TBD`.

**UPDATE 2026-09-13: the testbed track was built after all, on branch `feature/mininet-testbed`.**
The status paragraph immediately below was true when written and is now superseded; it is kept so
the history stays visible. The simulator has now been compared against real Open vSwitch at one
operating point. Summary, with every number traceable to `docs/PLAN_TESTBED.md` sections 2.8 to 2.12:

- Section 2's load-bearing assumption was tested. Of the three sharing modes, only
  `demand_proportional` reproduces URLLC latency rising with the eMBB level; `equal` and
  `min_rate_proportional` both predict it stays flat, which the real switch contradicts. The
  control problem exists by measurement.
- Under the rule recorded before the comparison ran, the simulator is classified
  `TRACKS_demand_proportional` at that operating point. It also has systematic, statistically real
  biases: it underestimates eMBB goodput by 5 to 12 percent and overestimates Best Effort goodput by
  7 to 39 percent at every congested level, and its URLLC median is off by up to 50 percent.
- Section 3's latency tail: on the testbed, URLLC p95 and p99 carry host jitter present even with an
  empty queue, so simulator p95 has no testbed counterpart at the 7 ms scale.

Status (as written before the testbed existed): simulator track complete, 320-run suite in
docs/EXPERIMENTS.md. **Nothing here has ever been validated against real Mininet or Open vSwitch,
and now never will be within this project: the testbed track was not built.** Every reference below
to "Week 1b" describes work that did not happen; the sentences are kept rather than deleted so the
unmet dependency stays visible.

---

## 1. Action space: absolute rate caps, not relative adjustments

The action is a choice of one **absolute** eMBB rate cap from a discrete set, expressed as a
fraction of bottleneck capacity. The default set is `[0.20, 0.35, 0.50, 0.65, 0.80]`, configured
at `config/default.yaml` under `action.embb_levels`. The mapping from an action index to the
three queue caps is `net/backend.py:allocation_from_level`.

The alternative, an action set of `{increase, maintain, decrease}`, was rejected. Under relative
actions the effect of an action depends on the current allocation, so the current allocation
becomes part of the state. The problem is then a full Markov decision process and a contextual
bandit is the wrong tool for it: a bandit assumes the reward of an arm depends on the context
and the arm alone, not on the history of arms pulled. Choosing an absolute level makes each
round's mapping from arm to expected reward depend only on the observed context, which is the
assumption LinUCB's regret bound is stated under.

### The residual violation of the bandit assumption, stated plainly

The rounds are still not independent and identically distributed, and pretending otherwise would
be dishonest. Two mechanisms carry state across rounds:

1. **Queue backlog persists.** The URLLC and Best Effort queues do not drain to zero between
   control intervals. An aggressive action at step `t` leaves backlog that raises latency at step
   `t+1` even if step `t+1`'s action is conservative. So the action at `t` influences the reward
   at `t+1`, which a bandit does not model.
2. **The offered load is autocorrelated.** Traffic phases last tens of seconds, so consecutive
   contexts are highly correlated rather than drawn independently.

Why we accept this:

- The context vector includes `q0_backlog_norm` and both a smoothed and an instantaneous latency
  feature (`agent/context.py:FEATURE_NAMES`). Carried-over backlog is therefore partly
  *observable* rather than purely hidden. A bandit conditioning on observed backlog is closer to
  correct than one conditioning on latency alone.
- The control interval is 1 s, and the measured equilibrium queueing delay across all action
  levels is under 10 ms (see section 3). The backlog time constant is therefore roughly two
  orders of magnitude shorter than the decision interval, so most of the state genuinely does
  decay within one round.
- The guardrail bounds the cost of the residual error. The worst outcome a myopic decision can
  produce is capped by a deterministic mechanism that does not depend on the learned model
  being correct.

This remains an approximation. If the results show LinUCB underperforming, inter-round coupling
is one of the candidate explanations and the report should say so rather than attributing the
result entirely to the method.

---

## 2. The simulator, and the one assumption everything rests on

`net/sim_backend.py`. Per-queue token bucket feeding a work-conserving shared link, simulated at
10 ms sub-steps inside each 1 s control interval.

**Why a token bucket.** The real actuator is an OVS HTB queue with `other-config:max-rate`, which
is a token bucket. Using the same primitive means "cap eMBB at 0.5 of capacity" denotes the same
thing in sim and on the testbed. A fluid model would not reproduce burst absorption; an M/M/1
model has no rate cap, so the action space would not exist.

**Why not strict priority.** If q0 were served with strict priority, URLLC would never queue,
latency would never rise, the guardrail would never fire, and we would report zero SLA violations
as a result. That failure mode flatters the method, so the link uses rate-capped work-conserving
sharing.

### The load-bearing assumption: how leftover capacity is split

After every queue takes its min-rate guarantee, something must divide the rest. That choice
decides whether the project has a problem to solve. Three modes are implemented and selectable at
`sim.excess_sharing`:

| Mode | Behaviour | Consequence for this project |
|---|---|---|
| `demand_proportional` (default) | A backlogged class with a high ceiling captures most of the leftover | URLLC gets crowded out, latency rises, the control problem exists. Pessimistic end of the plausible range. |
| `equal` | Strict per-queue fair share, idealised DRR | URLLC's residual demand is largely covered for free. **No controller is needed and the project is vacuous.** |
| `min_rate_proportional` | Leftover split in proportion to the guarantees | Intermediate, and protective of URLLC. |

We chose `demand_proportional`. It is an assumption, not a measurement. It is made visible and
testable by `tests/test_sim_backend.py::test_excess_sharing_mode_decides_whether_the_problem_exists`,
which asserts that `demand_proportional` is strictly worse for URLLC than `equal` at the same
action.

**This was to be settled by running the identical trace through `OvsCliBackend` and comparing all
three modes against it. THAT EXPERIMENT WAS NEVER RUN.** If real OVS behaves like `equal`, this
control problem does not exist at this operating point and every result in
`docs/EXPERIMENTS.md` is a study of an artefact. Nothing in this project rules that out. It is
the single largest threat to validity and `docs/REPORT_OUTLINE.md` section 8 states it as such.

**UPDATE 2026-09-13: the experiment has now been run** as a fixed-level sweep on real OVS rather than
through `OvsCliBackend` (`docs/PLAN_TESTBED.md` sections 2.11 and 2.12). Real OVS does not behave like
`equal` at this operating point: URLLC median RTT rose from 0.10 to 7.67 ms across the five eMBB
levels, while `equal` predicted 0.09 ms at every level. The threat above is resolved for this operating
point under constant load. It is not resolved for the jittered scenarios the policies were evaluated
on, which were never run on the testbed.

### A property of this model worth knowing about: URLLC self-stabilises

Under `demand_proportional`, a queue that falls behind accumulates backlog, which raises its
residual demand, which increases its share of the leftover. URLLC therefore does not starve; it
settles at a bounded equilibrium backlog. This is why the achievable latency range is narrow
(section 3). It is a real consequence of the modelling choice, not a bug, and it is the single
most important thing for Week 1b to check against reality.

---

## 3. Emulation-scaled SLO, derived from measurement

The slide deck proposed a 15 ms target and a 25 ms hard limit. **Those are unreachable in this
simulator.** Because of the self-stabilisation described above, URLLC queueing delay is bounded.

Measured URLLC latency by action level, `burst.yaml`, seed 0, burst phase only, at the configured
operating point (`slices.urllc.min_share: 0.10`, `slices.be_cap_share: 0.45`). Produced by
`net/sim_backend.py` driven at a fixed level:

| eMBB level | eMBB goodput (Mbps) | BE goodput (Mbps) | URLLC p50 (ms) | URLLC p95 (ms) |
|---|---|---|---|---|
| 0.20 | 2.000 | 4.001 | 2.00 | 2.00 |
| 0.35 | 3.193 | 3.737 | 3.14 | 3.39 |
| 0.50 | 3.773 | 3.238 | 5.01 | 5.27 |
| 0.65 | 4.191 | 2.820 | 6.84 | 7.16 |
| 0.80 | 4.507 | 2.504 | 8.65 | 9.04 |

A 25 ms hard limit against a 9.04 ms ceiling would produce zero violations for every policy, an
idle guardrail, and nothing to measure. We could have raised the ceiling by inflating
`sim.substep_s`, since queueing delay scales with the scheduler quantum. We did not, because that
is tuning the physics to fit the threshold.

Instead the SLO is set from the measured range:

- `sla.target_ms: 5.0`
- `sla.hard_ms: 7.0`
- `guardrail.warn_ms: 4.0`, `guardrail.safe_ceiling_index: 2`

This places the decision boundary between levels 0.50 and 0.65, so different policies land on
different sides of it and the SLA constrains something real.

### Limitation, stated up front

**The testbed SLO will be different, and sim and testbed SLA violation rates are not comparable
until both are derived the same way.** On real Mininet, scheduler jitter alone is milliseconds,
so the achievable range there is unknown and the deck's 15 ms may well be correct.
`experiments/measure_noise_floor.py` was to measure the idle RTT distribution over 60 s and derive
the testbed SLO from it. **It was never written and never run.** The testbed threshold is
therefore not `TBD` in the sense of "pending"; it is absent, and the SLO used throughout is the
simulator-derived one above.

The contextual structure the project depends on does exist. Measured on `sawtooth.yaml`, seed 0,
at the most aggressive action level, URLLC p95 is 4.09 ms when URLLC's offered load is light and
9.20 ms when it is heavy. The safe action therefore depends on an observable feature, which is
what makes this a contextual problem rather than a fixed one.

---

## 4. The guardrail

`agent/guardrail.py`. It sits between the policy and the actuator. Per control step:

1. The policy proposes against the unrestricted action set. This is logged as `proposed_action`
   and exists only so we can measure how often the guardrail changed the outcome.
2. The guardrail computes an allowed mask from the context, in milliseconds.
3. If the proposal is outside the mask, the policy is asked to choose again from the allowed set.
   Asked, not overruled, so whatever the policy knows about the relative value of the safe arms
   is preserved.
4. Whatever comes back is clamped to the mask regardless. The invariant "the applied action is
   always inside the mask" is a property of this one file and does not depend on policies being
   well behaved. `tests/test_guardrail.py` proves it against an adversarial policy that always
   demands the most aggressive arm, over a 5000-case randomized sweep.
5. Every step is logged with the proposal, the applied action, and a reason.

States, in priority order: `holddown`, `hard_override`, `warn_mask_reselect` / `warn_mask_pass`,
`ok`.

**Intervention is defined as `applied_action != proposed_action`.** A step where the guardrail
masked arms but the policy had already chosen a safe one is not counted. Counting it would
inflate our headline safety metric by crediting the guardrail for steps where nothing changed,
and that metric is the direct evidence for the word "safe" in the project title.

**Why the instantaneous tail is checked, not only the EWMA.** The EWMA lags by roughly `1/alpha`
steps. A guardrail watching only the EWMA would be late by construction against exactly the short
sharp spikes `adversarial.yaml` is built to produce, and we would be claiming a safety property
we do not have. The hard override fires on either the EWMA or the current p95, and
`escalation_source` records which one tripped so the report can quantify how much of the safety
came from the fast path.

---

## 5. The reward function

`agent/reward.py`. Weights at `config/default.yaml` under `reward`.

```
r = w_tput * embb_goodput/capacity
  + w_be   * be_goodput/capacity
  - w_sla  * max(0, rtt - sla_target)/sla_target
  - w_viol * 1[rtt > sla_hard]
  - w_drop * total_drops/drop_reference
```

- **Which RTT statistic**: config-selected, default p95. A latency SLO stated on the median lets
  half the traffic miss it unnoticed. p50 is still logged so the conclusion's sensitivity to this
  can be checked.
- **Goodput normalised by capacity, not offered load.** Normalising by offered load would turn
  the reward into a fraction-served measure, under which starving eMBB during a burst scores the
  same as serving it fully during a lull.
- **Best Effort carries real weight (0.30).** If BE were worth zero, the optimal policy would be
  to hand the entire non-URLLC link to eMBB and the action space would collapse to "as high as
  the guardrail permits".
- **Hinge plus step, not one or the other.** The hinge means missing by 1 ms scores better than
  missing by 50 ms. The step makes crossing the hard limit discretely expensive. Hinge alone
  would let a policy trade many small misses for throughput; step alone would make everything
  below the hard limit look identical.
- The reward sees telemetry only. It never sees the action, the policy identity, or the guardrail
  state, so a policy cannot be rewarded for its own machinery rather than for the network outcome.

### Known issue with the drop term, to be examined in the sensitivity study

In the runs measured so far, `total_drops` is dominated by q1 tail drops caused by eMBB offering
far more than its cap allows (for example 9.5 Mbps offered into a 3.5 Mbps cap). Raising the eMBB
cap reduces those drops. So `w_drop` is not primarily measuring congestion damage; it acts as a
**second throughput incentive** pushing the cap upward, opposing the SLA term.

That tension is legitimate, but the term does not mean what its name suggests. Flagged in week 1a
rather than discovered in week 5, and acted on: the sensitivity study in `docs/EXPERIMENTS.md`
sweeps `w_drop` as well as `w_sla` for exactly this reason. The report must not describe `w_drop`
as a congestion penalty without this caveat.

---

## 6. Two modelling choices that were changed after they were written down

Recorded because both would have silently corrupted results.

**The round-trip factor.** An early draft of `docs/PLAN.md` modelled URLLC RTT as
`base_rtt + 2 * queueing_delay`. That is wrong. QoS is applied on the `s1 -> s2` egress only, so
the echo reply crosses an unloaded path and exactly one queueing delay applies. The 2x would have
inflated every latency number in the report. `net/sim_backend.py` uses 1x.

**Token bucket depth.** The first implementation used a single global `bucket_burst_bytes` of
12500, which is 10 ms of full link capacity. Any queue whose per-substep token accrual exceeded
that constant pinned its bucket full, and the `max-rate` cap stopped binding. Levels 0.50, 0.65
and 0.80 produced byte-identical telemetry, meaning the action had no effect above 0.35. The
bucket is now sized per queue from that queue's own cap
(`max(cap * substep / 8 * multiplier, min_bytes)`), which is how `tc` sizes HTB burst and for the
same reason.

---

## 7. Deviations from the specified repository layout

- `config_loader.py` sits at the repository root. The specified layout lists `config/` as a data
  directory and gives the loader no home, and burying it in `net/backend.py` would contradict
  that file's stated contents. It is the only file outside the specified tree.
- `net/backend.py` adds `offered_bps_per_queue` to `TelemetrySample` beyond the required minimum.
  Without it the Oracle cannot be computed and the sim/OVS comparison has no independent variable
  to align on. It is observable in the sim and derivable on the testbed from the iperf3 target
  rates.
