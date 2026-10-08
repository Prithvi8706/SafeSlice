# SafeSlice: the whole project, start to end

Safe dynamic network slicing for SLA-preserving SDN.

Prithvi Raghu (24BCE2624) · Sumanta Kumar (24BCT0302) · Guide: Sasikala R · SCOPE, VIT

This document is the single narrative account: what the project set out to do, what was actually
built, what was measured, what was found, what was got wrong along the way and corrected, and what
remains. Every number here was produced by code in this repository that ran. Where something was
not measured, it says so.

---

## 1. The question

Three kinds of traffic share one bottleneck link and want incompatible things:

| Slice | Queue | Wants |
|---|---|---|
| URLLC | q0 | Low latency. It is the protected slice. |
| eMBB | q1 | As much throughput as it can get. |
| Best Effort | q2 | Whatever is left. |

Allocating capacity statically forces a bad choice. Reserve enough for URLLC's peak and the link
sits idle whenever the peak does not arrive. Reserve for its average and an eMBB burst starves it.

Dynamic allocation is the obvious fix, and most of the literature does it with deep reinforcement
learning, which brings training cost, compute cost, and no bound on what the policy does while it
is still learning. This project asks whether something much lighter is enough:

> **Can a contextual bandit plus a deterministic safety mechanism beat a well-tuned reactive rule,
> and what exactly does the safety mechanism guarantee?**

The answer was allowed to be "no" from the start. A clean negative result with good method was
treated as a better outcome than a forced win.

---

## 2. The system

A 10 Mbps bottleneck between two switches, `h1,h2,h3 --- s1 === s2 --- h4,h5,h6`, with all queueing
on s1's egress toward s2. Once per second a controller reads telemetry, picks a cap for the eMBB
slice, and is scored by a reward.

**The action is one of five absolute rate caps**, `[0.20, 0.35, 0.50, 0.65, 0.80]` of link capacity.
Absolute, not "increase / hold / decrease". This is load-bearing, not stylistic: under relative
actions the effect of an action depends on the current allocation, so the allocation becomes part of
the state, the problem becomes a full Markov decision process, and a contextual bandit is the wrong
tool. Choosing an absolute level keeps the bandit's assumption approximately true.

Approximately, not exactly, and the project says so rather than assuming it away. Queue backlog does
not fully drain between intervals and offered load is autocorrelated, so rounds are not independent.
The context vector includes backlog and latency so the carried state is partly observable, and the
measured queueing delay is under 10 ms against a 1 s control interval.

**The reward:**

```
r = w_tput·(embb/C) + w_be·(be/C) − w_sla·max(0, rtt−target)/target
      − w_viol·1[rtt > hard] − w_drop·(drops/drop_ref)
```

Best Effort carries real weight on purpose. If it were worth zero, the best policy would hand the
whole non-URLLC link to eMBB and the action space would collapse to "as high as allowed".

**The guardrail** sits between the policy and the actuator, not after it. Each step the policy
proposes against the full action set, which is logged so interventions can be counted; the guardrail
computes an allowed mask in milliseconds; if the proposal is outside the mask the policy is asked to
choose again from what remains; and whatever comes back is clamped to the mask regardless. Three
tiers: a warning mask above 4 ms, a hard override to the safest level above 7 ms, and a three-step
hold-down afterwards so the controller cannot oscillate straight back.

**The policies compared:**

| Policy | What it is |
|---|---|
| `static_safe` | Always the lowest cap. Never violates, wastes capacity. The floor. |
| `static_equal` | Cap nearest an equal three-way split. |
| `threshold` | Reactive rule: step down when smoothed latency is high, up when low. **The real competitor.** |
| `epsilon_greedy` | Context-free bandit. The ablation that isolates what the context is worth. |
| `linucb` | Contextual bandit, learning during the evaluated run. |
| `*_pretrained` | Same, trained on held-out seeds and frozen. |
| `oracle` | Best phase-static allocation, knowing the trace. A reference line, not a method. |

---

## 3. How the project was built

Two tracks, in two phases.

**Phase one, the simulator.** `net/sim_backend.py` models the bottleneck as a per-queue token bucket
feeding a work-conserving shared link, chosen because the real actuator is an Open vSwitch HTB queue,
which is a token bucket. Everything else in the project is written against a `NetworkBackend`
interface, so the runner, policies, guardrail and metrics never know which backend they are driving.
This phase produced the full policy study: 320 evaluation runs, 280 tuning runs, 900 sensitivity
runs.

**Phase two, the testbed.** Real Mininet and Open vSwitch on WSL2 Ubuntu 24.04. Built in seven
stages, each gated on the previous one producing output, with the rule for reading each result
written down *before* the measurement was taken. That habit caught three real problems, described in
section 5.

The second phase exists because the first one rested on an assumption nobody had checked.

---

## 4. What the simulator study found

Protocol first, because the result depends on it. A run is fully determined by `(policy, scenario,
seed)`, with trace seeds derived from blake2b rather than Python's `hash()`, which is salted per
process. A seed fixes the traffic bit for bit, so policies are compared on identical load and
differences are reported as the mean of **per-seed differences** with a confidence interval on that
mean. Hyperparameters were tuned, and the pre-trained variants trained, on seeds `[100…104]`,
disjoint from the evaluation seeds `[0…9]` and asserted disjoint by a test. 300 s per run, first 30 s
discarded, 10 seeds per cell, run order randomized from a logged seed.

### The headline: exploration is what decides it

Paired difference in mean reward against `threshold`. **Bold** means the 95% interval excludes zero.

| Policy | adversarial | burst | ramp | sawtooth |
|---|---|---|---|---|
| `linucb` (online) | **−0.0728 ± 0.0182** | **+0.0248 ± 0.0062** | **−0.0217 ± 0.0029** | **−0.0104 ± 0.0089** |
| `linucb_pretrained` | −0.0047 ± 0.0117 | **+0.0582 ± 0.0042** | **+0.0019 ± 0.0007** | **+0.0225 ± 0.0078** |
| `epsilon_greedy` | **−0.1281 ± 0.0092** | +0.0082 ± 0.0199 | **−0.0281 ± 0.0219** | −0.0201 ± 0.0235 |
| `oracle` | **+0.1241 ± 0.0115** | **+0.0589 ± 0.0042** | **+0.0019 ± 0.0007** | **+0.0358 ± 0.0077** |

**LinUCB learning online loses to a hand-tuned threshold on three of four scenarios**, and the losses
are separated from zero. Pre-trained and frozen, the same algorithm wins on three of four, with
*fewer* SLA violations rather than by trading safety for throughput. The entire difference between
those two rows is the cost of exploring online.

This outcome was predicted in writing in week one, before any policy existed, with a commitment to
report it rather than tune until it disappeared.

### The context is what the bandit contributes

The `adversarial` scenario lures a policy to an aggressive level with long calm periods, then spikes
URLLC for less than the smoothing time constant. All 562 violations it produces happen inside a
spike. Violation rate within spikes:

| Policy | rate |
|---|---|
| `epsilon_greedy` (context-free) | 30.4 % |
| `linucb` (online) | 19.2 % |
| `threshold` | 15.6 % |
| `linucb_pretrained` | **12.5 %** |

A context-free bandit is caught more than twice as often as a converged contextual one on identical
traffic. The two differ only in whether they see the context vector.

### The guardrail's guarantee is narrower than it sounds

It does **not** eliminate SLA violations, and the project refuses to claim it does. On `adversarial`,
`epsilon_greedy` violates on 5.85 % of steps despite the guardrail intervening on 23.4 % of them. The
reason is structural: the guardrail acts on the previous interval's telemetry, so it cannot prevent
the first interval of a spike it has not yet seen.

What it does guarantee is provable: the applied action is always inside the allowed mask, whatever
the policy proposes. That is established by a property test against an adversarial policy that always
demands the most aggressive arm, over 5,000 randomized cases.

One design decision was argued from first principles in week one with no evidence, and later
confirmed by measurement. The guardrail escalates on either the smoothed latency or the instantaneous
tail. Across all 320 runs, 86,400 post-warm-up steps, 17,580 escalating:

| Signal that escalated | steps | share |
|---|---|---|
| smoothed only | 838 | 4.8 % |
| **instantaneous only** | **6,740** | **38.3 %** |
| both | 10,002 | 56.9 % |

A guardrail watching only the smoothed signal would have missed more than a third of all escalations.

### Cost, and whether "lightweight" is earned

| Policy | decision latency mean | p99 |
|---|---|---|
| static baselines | 0.021 – 0.023 ms | 0.040 – 0.049 ms |
| `threshold` | 0.035 – 0.040 ms | 0.073 – 0.115 ms |
| `linucb` | 0.091 – 0.098 ms | 0.192 – 0.233 ms |

LinUCB costs about four times an array lookup and stays under a quarter of a millisecond at p99
against a 1,000 ms control interval.

### Does the conclusion survive the reward weights?

900 runs over `w_sla` × `w_drop`. `linucb_pretrained` wins in all 15 weight combinations, separated
from zero in 13. Which *baseline* ranks second does move with the weights. Only the learned policies
respond to the weights at all: the reactive and static rules are numerically invariant to every
digit, because they are fixed rules that never read the reward.

---

## 5. What the testbed found

The simulator study rested on one assumption: how leftover capacity is divided among backlogged
queues. Under an equal-share rule URLLC would be protected for free, no controller would be needed,
and the whole project would be a study of an artifact. The project flagged this in week one as its
largest threat to validity and built the testbed to settle it.

### Stage 1: the topology, and a bug in my own check

Mininet topology, OVS HTB queues, port-based classification flows. Connectivity, queue visibility and
tc classes all passed.

The cap check did not, though it claimed to. It read iperf3's `end.sum` as the receiver's rate. On
iperf 3.16 that field is the **sender**. A controlled test across a 1 Mbit cap settled it: `end.sum`
reported 4.002 Mbps while `sum_received` reported 0.976. The check was rewritten to read sender and
receiver separately, and the flawed result kept under a "superseded" filename rather than deleted.

Once corrected, the cap binds exactly. HTB counts whole Ethernet frames, so a 1400-byte payload rides
in a 1442-byte frame and a 3.5 Mbps cap predicts 3.398 Mbps of payload. Measured: **3.399**.

### Stage 1b and 1c: the testbed was not behaving like a network

Asked to send 14 Mbps into a 3.5 Mbps cap, the sender sent 3.53 and **nothing was dropped**. The
simulator assumes the opposite: traffic keeps arriving and the excess is dropped.

The queue never overflowed because the sender stopped first. The deepest backlog seen was 134,106
bytes, which is 93 frames, against a socket send buffer of 212,992 bytes and a switch queue holding
1,000 packets. In a real network the sender is on another machine and a switch queue cannot block an
application; this only happens because Mininet runs both in one kernel. It is an emulation artifact.

Rather than pick a setting that worked, three queue sizes were tested with predictions recorded
first:

| Queue | Predicted | Result |
|---|---|---|
| OVS default, 1000 packets | blocks the sender | blocks, backlog 134,106 B |
| 62,500 B (the simulator's own value, set in week one) | drops instead | **drops**, backlog 62,006 B = 43 frames exactly |
| 500,000 B, the control | blocks again | blocks, backlog 134,106 B again |

All three held. The control is the important one: two very different queue limits both stopped at the
same 134,106 bytes, so the limit is not what caps the backlog. With the finite queue, iperf3's
receiver and the switch's own drop counter agreed to within one packet at every rate.

Actuation was checked at the same time: queue rate changes reached the kernel shaper in 21 to 62 ms,
more than fifteen times inside the control interval.

### Stage 2: the noise floor

60 s of idle round-trip time through the bottleneck and the URLLC queue: **p50 0.093 ms, p95 0.133,
p99 0.166**, 1,182 samples, zero loss. The simulator's assumed base latency of 2.0 ms was a modelling
constant about twenty times larger, which is why the later comparison substitutes the measured value.

### Stage 4a: the latency tail is the laptop, not the network

The first sweep showed URLLC latency with a tail that queueing could not explain: at the lowest eMBB
level URLLC had an empty queue and a median of 0.11 ms, yet a p99 of 4.66 ms and a worst reply of
35.7 ms.

Running ping at real-time priority separates a slow probe from slow packet handling. Real-time
priority did not remove the tail; in fact it produced the two largest single replies. So the tail
comes from load on the host, below the probe, and this test cannot tell the kernel's packet path
apart from Hyper-V pausing the virtual machine.

The consequence was adopted rather than worked around: **median latency is the primary URLLC metric**,
and p95 and p99 are reported only against the loaded uncongested floor, never against zero.

A metric bug was fixed here too. "Ping delivered" read 90 to 98 % while ping's own summary showed
0 % loss, because the metric assumed exact 0.05 s pacing and ping paces closer to 0.055 s. It now
counts gaps in ICMP sequence numbers.

### Stage 4: slicing works on real Open vSwitch

Constant load (URLLC 3, eMBB 10, BE 4 Mbps), three repeats of 120 s per level, shuffled order,
calibration passed, 15 of 15 runs valid. Mean ± 95 % interval:

| eMBB level | eMBB Mbps | BE Mbps | URLLC median RTT | URLLC drops |
|---|---|---|---|---|
| 0.20 | 1.995 ± 0.009 | 3.997 ± 0.015 | 0.10 ± 0.03 ms | 0 |
| 0.35 | 3.270 ± 0.030 | 3.638 ± 0.010 | 2.63 ± 0.21 ms | 0 |
| 0.50 | 4.082 ± 0.083 | 2.812 ± 0.020 | 3.65 ± 0.27 ms | 0 |
| 0.65 | 4.424 ± 0.015 | 2.465 ± 0.022 | 4.44 ± 0.44 ms | 0 |
| 0.80 | 5.087 ± 0.018 | 1.796 ± 0.013 | 7.67 ± 0.24 ms | 0 |

- Raising the eMBB cap moves capacity from Best Effort to eMBB at every step.
- URLLC throughput is fully protected: zero drops and 100 % probe delivery in all 15 runs.
- Raising eMBB costs URLLC **latency**, about 77-fold from the lowest level to the highest, with
  adjacent levels' intervals disjoint.
- Every packet is accounted for: delivered plus dropped is 99.6 to 100.0 % of offered in every run.

The third point settles the project's largest open question. **The control problem exists, by
measurement**, not by assumption.

### Stage 5: the simulator against the switch

The simulator was run under matched conditions (same load, no jitter, the measured 0.093 ms base
latency) in all three sharing modes, scored by a rule recorded beforehand. Normalised error, where at
most 0.25 counts as tracking:

| Mode | eMBB goodput | BE goodput | URLLC median | Mean |
|---|---|---|---|---|
| `demand_proportional` | 0.085 | 0.158 | 0.087 | **0.110** |
| `equal` | 0.167 | 0.256 | 0.476 | 0.300 |
| `min_rate_proportional` | 0.096 | 0.117 | 0.476 | 0.230 |

**Result: `TRACKS_demand_proportional`.** The mode the entire 320-run study used is the one the real
switch supports.

The robust part does not depend on the threshold at all: `demand_proportional` is the only mode that
reproduces URLLC latency rising with the eMBB level. The other two predict 0.09 ms at every level,
which the switch contradicts by about 85-fold at the top level.

Two honest qualifications:

- The label is threshold-sensitive. At 0.15 or stricter it becomes "partial", with Best Effort
  goodput as the metric that stops matching.
- The simulator is systematically biased. At every congested level it **underestimates eMBB goodput
  by 5 to 12 %** and **overestimates Best Effort by 7 to 39 %**, and every one of those errors is
  larger than the testbed's own interval. The real switch favours eMBB over Best Effort more than the
  model does.

---

## 6. What the project establishes

1. **Slicing works on a real programmable switch.** Rate caps bind exactly, the protected slice keeps
   its throughput, and the cost of giving eMBB more is paid in URLLC latency. (Testbed, stage 4.)
2. **The control problem is real, not an artifact of the model.** Two of three plausible models
   predicted URLLC would be protected for free. The switch says otherwise. (Stage 5.)
3. **Exploration, not learning, is the deciding cost.** Online LinUCB loses to a well-tuned reactive
   rule; the same algorithm trained offline and frozen beats it with fewer violations. If a
   deployment can train offline, the bandit is worth it. If it must learn in production, a threshold
   is the better engineering choice at this scale.
4. **The context is what the bandit contributes**, not learning as such. A context-free bandit is
   caught by spikes more than twice as often on identical traffic.
5. **The guardrail bounds the action space provably, and does not bound the violation rate.** It acts
   on the previous interval. Systems claiming safety from such a mechanism should say which of the
   two they mean.
6. **The simulator is right in kind and biased in degree**, checked at one operating point.

---

## 7. Limitations, in the order they matter

1. **One operating point, constant load.** The testbed comparison covers a single offered load with
   no jitter and five fixed caps. It does not show the simulator agrees under the fluctuating traffic
   the policies were actually evaluated on.
2. **No policy has ever run on the real switch.** Every policy result is a simulator result.
3. **The testbed cannot resolve tail latency at the scale the SLA uses.** Host jitter puts the
   uncongested 95th percentile at 5 to 12 ms against a 7 ms limit, so p95 on this hardware measures
   the laptop as much as the queue.
4. **Rounds are not independent**, which is not what a bandit assumes.
5. **Ten seeds of one simulator is not ten samples of reality.**
6. **The scenarios are hand-written**, and `adversarial` was designed against the guardrail's known
   weakness. It is a stress test, not a traffic model.
7. **The reward weights decide which baseline ranks second**, though not which policy wins.
8. **The "SDN" is OVS-native QoS, not an OpenFlow controller.** The data plane is real and
   programmable, but allocation decisions are sent over the OVS management interface rather than
   OpenFlow.

---

## 8. What is not built

| Component | Status |
|---|---|
| `net/ovs_cli_backend.py` | Not written. Would let existing policies run live on the switch. |
| Live policy loop on the testbed | Not run. Blocked on the tail-latency problem in limitation 3. |
| `RyuBackend`, `controller/ryu_app.py` | Deliberately out of scope. Ryu is unmaintained, and it closes no gap. |
| Report rewrite around the testbed results | Not done. `docs/REPORT.md` still describes the simulator study alone. |

The branch `feature/mininet-testbed` holds the entire testbed track and has not been merged.

---

## 9. Reproducing

Simulator, any machine, no privileges:

```bash
pip install -r requirements.txt && pytest -q     # 233 tests
python experiments/run_suite.py                  # 320 runs
python -m analysis.aggregate                     # every simulator table
python -m analysis.plots                         # every simulator figure
```

Testbed, Linux with root. Setup in `docs/TESTBED_SETUP.md`:

```bash
python3 net/topology/slice_topo.py --check              # topology, queues, cap binding
python3 experiments/measure_noise_floor.py              # idle latency floor
python3 experiments/sweep_levels_ovs.py --quick         # 5 runs, about 4 min
python3 experiments/sweep_levels_ovs.py                 # 15 runs, about 32 min
python experiments/compare_sim_vs_ovs.py                # simulator vs switch
```

Every table is recomputed from per-step logs rather than read from a cache, so changing a metric
definition changes the table instead of leaving a stale number behind.

---

## 10. Where things are

| Path | What |
|---|---|
| `net/backend.py` | The interface everything is written against |
| `net/sim_backend.py` | The queue simulator, and the assumption that mattered |
| `net/topology/slice_topo.py` | Mininet topology, OVS QoS, the testbed checks |
| `traffic/generator.py` | iperf3 and ping orchestration, and the measurement definitions |
| `agent/` | Context, reward, guardrail, bandits, policies |
| `experiments/` | One run, the grid, tuning, sensitivity, and the five testbed experiments |
| `analysis/` | The CSV schema, metrics, aggregation, figures |
| `docs/PLAN.md` | The original six-week plan, kept as written |
| `docs/PLAN_TESTBED.md` | The testbed plan, and every result with the rule recorded before it |
| `docs/DESIGN.md` | Design decisions, each with its reasoning and its limitation |
| `docs/EXPERIMENTS.md` | Protocol, metric definitions, simulator result tables |
| `docs/REPORT.md` | The write-up of the simulator study |

---

## 11. A note on method

Three things in this project were got wrong and corrected rather than buried: a cap check that read
the sender's rate as the receiver's, a delivery metric that measured ping's timer instead of packet
loss, and a throughput measurement diluted by connection setup time. Each is recorded in the plan
with the date and the correction, and the superseded result files are kept.

The habit that caught them was writing the rule for reading a measurement *before* taking it. Every
testbed stage has a pre-recorded decision table in `docs/PLAN_TESTBED.md`, including the cases that
would have killed the project. Two of those tables named outcomes that would have invalidated the
simulator study, and both were run rather than avoided.
