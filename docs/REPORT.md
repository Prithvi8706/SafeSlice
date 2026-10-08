# Safe Dynamic Network Slicing for SLA-Preserving SDN

**A simulation study of a contextual-bandit slicing controller with a deterministic safety guardrail,
with its load-bearing assumption checked on real Mininet and Open vSwitch.**

Prithvi Raghu (24BCE2624) · Sumanta Kumar (24BCT0302) · Guide: Sasikala R · SCOPE, VIT

---

## Abstract

Static bandwidth allocation cannot react when traffic demand shifts, and deep reinforcement
learning approaches to dynamic slicing carry training and compute costs that are hard to justify
at small scale. We study a lighter alternative: a contextual bandit that selects one of five
absolute rate caps for an eMBB slice once per second, paired with a deterministic guardrail that
sits between the policy and the actuator so that latency risk is bounded by a mechanism which does
not depend on the learned model being correct.

Across four traffic scenarios and ten seeds on a token-bucket queue simulator (320 evaluation
runs; 1,500 runs in total across evaluation, tuning and sensitivity), we find that **LinUCB learning online loses to a hand-tuned reactive
threshold on three of four scenarios**. The same algorithm pre-trained on held-out seeds and
frozen beats that threshold on three of four, with fewer SLA violations, and closes the gap to the
best phase-static allocation almost entirely on two. The entire difference between those two
results is the cost of exploring online. A context-free bandit ablation performs substantially
worse than either, confirming that the context vector carries real information rather than
decorating the method.

A 900-run sweep over the reward weights finds this conclusion robust: the converged policy wins in
all 15 weight combinations, separated from zero in 13. Which *baseline* ranks second does move
with the weights, and only the learned policies respond to the weights at all — the reactive and
static rules are numerically invariant to them, being fixed rules that never read the reward.

We also report a negative result that constrains the contribution. The guardrail does **not**
eliminate SLA violations — it acts on the previous interval's telemetry and so cannot prevent the
first interval of an unanticipated spike. What it guarantees is narrower and provable: the applied
action is always inside the allowed action mask.

Every one of those results rests on one modelling assumption: how leftover link capacity is
divided among backlogged queues. Under two of three plausible rules URLLC would be protected for
free and no controller would be needed. We built a Mininet / Open vSwitch testbed to settle it.
On the real switch, raising the eMBB cap from the lowest to the highest level raises URLLC median
latency about **77-fold** (0.10 to 7.67 ms) while URLLC throughput stays fully protected, so **the
control problem exists by measurement**. Of the three sharing rules, only the one the simulator
study used reproduces that (normalised error 0.110 against 0.230 and 0.300). The simulator is right
in kind and biased in degree: it underestimates eMBB goodput by 5 to 12 % and overestimates Best
Effort by 7 to 39 % at every congested level. That check covers one operating point at constant
load, and no policy was run live on the switch.

---

## 1. Problem

A shared bottleneck link carries three traffic classes with incompatible requirements: URLLC
(latency-critical), eMBB (throughput-hungry) and Best Effort. Provisioning statically forces a
choice between two bad options. Provision URLLC for its peak and capacity sits idle whenever the
peak does not arrive; provision it for the average and a burst of eMBB traffic starves it and
breaches the latency SLA.

Dynamic allocation is the obvious answer, and the literature has largely pursued it through deep
reinforcement learning [1, 2, 5]. DRL brings training time, compute overhead, and — more
importantly for a safety-relevant system — no bound on what the policy may do while it is still
learning. A contextual bandit is a lighter fit for a problem where each decision's outcome is
observed within one control interval, but a bandit alone offers no safety property either.

**The question this project asks is therefore not "can a bandit allocate bandwidth" but "can a
lightweight learner plus a deterministic safety mechanism beat a well-tuned reactive rule, and
what exactly does the safety mechanism guarantee".**

## 2. System

Three queues share a 10 Mbps bottleneck. Every second the controller observes telemetry, selects
an eMBB rate cap, and receives a reward trading throughput against SLA damage.

**Action space: five absolute rate caps**, `[0.20, 0.35, 0.50, 0.65, 0.80]` of link capacity.
Absolute, not `{increase, maintain, decrease}`. This choice is load-bearing rather than
stylistic. Under relative actions the effect of an action depends on the current allocation, so
the allocation becomes part of the state, the problem becomes a full Markov decision process, and
a contextual bandit is the wrong tool for it — LinUCB's regret bound is stated under the
assumption that reward depends on context and arm alone. Choosing an absolute level keeps that
assumption approximately true.

**Approximately, not exactly.** Two mechanisms carry state across rounds and we state them rather
than assume them away: queue backlog does not fully drain between intervals, and offered load is
autocorrelated because traffic phases last tens of seconds. The context vector includes backlog
and both smoothed and instantaneous latency, so the carried state is partly *observable* rather
than hidden, and the measured equilibrium queueing delay is under 10 ms against a 1 s control
interval — roughly two orders of magnitude shorter. The approximation is defensible, not exact,
and inter-round coupling remains a candidate explanation for any underperformance.

**Reward.**

```
r = w_tput·(embb/C) + w_be·(be/C) − w_sla·max(0, rtt−target)/target
      − w_viol·1[rtt > hard] − w_drop·(drops/drop_ref)
```

with `w_tput = 1.0`, `w_be = 0.30`, `w_sla = 1.0`, `w_viol = 2.0`, `w_drop = 0.50`, evaluated on
per-step p95 latency. Best Effort carries real weight deliberately: were it zero, the optimal
policy would hand the entire non-URLLC link to eMBB and the action space would collapse to "as
high as permitted". The reward sees telemetry only — never the action, the policy identity, or
the guardrail state — so a policy cannot be rewarded for its own machinery.

**A known defect in the drop term, reported rather than hidden.** In these runs `total_drops` is
dominated by eMBB tail drops caused by offering far more than the cap allows, so *raising* the cap
*reduces* drops. `w_drop` therefore acts as a second throughput incentive opposing the SLA term,
not as a congestion penalty. This was identified in week one and is why the sensitivity study
sweeps `w_drop` as well as `w_sla` (§6).

## 3. Method

### 3.1 Context

Nine features: a bias term, smoothed (EWMA) latency, instantaneous p95 latency, per-slice
utilisation for all three classes, link utilisation, URLLC backlog, and a smoothed drop count. All
normalised to roughly [0, 1] so that LinUCB's single `alpha` is meaningful across dimensions.

Two inclusions are deliberate. The vector carries **both** a smoothed and an instantaneous latency
signal, because the gap between them is the only mechanism by which a learned policy can respond
faster than a latency threshold. And it carries **URLLC utilisation**, because the correct eMBB
level is largely determined by whether URLLC's offered load sits above or below its guarantee — a
policy that observes that can act *before* latency rises, whereas one watching only latency cannot.

### 3.2 The guardrail

Positioned between policy and actuator, not after it. Per control step: the policy proposes
against the unrestricted action set (logged, so we can measure how often the guardrail changed
anything); the guardrail computes an allowed mask from the context in physical units; if the
proposal falls outside, the policy is **asked to choose again** from the allowed set rather than
overruled, preserving whatever it knows about the relative value of safe arms; and whatever comes
back is clamped to the mask regardless.

Three tiers, in priority order: **hold-down** (3 steps after a hard override, so the controller
cannot oscillate straight back), **hard override** (> 7 ms: clamp to the safest level), **warn
mask** (> 4 ms: mask aggressive levels). The hard override fires on *either* the EWMA or the
instantaneous p95, and which one tripped is recorded — §5.3 shows this mattered.

**Intervention is defined as `applied_action ≠ proposed_action`.** A step where the guardrail
masked arms but the policy had already chosen safely is not counted. Counting it would inflate the
headline safety metric by crediting the guardrail for steps where nothing changed.

### 3.3 Policies

| Policy | Description |
|---|---|
| `static_safe` | Lowest cap always. Never violates, wastes capacity. The floor. |
| `static_equal` | Cap nearest an equal three-way split. Naive reference. |
| `threshold` | Reactive: steps down when smoothed latency is high, up when low. **The competitor.** |
| `epsilon_greedy` | **Context-free** bandit over arm means. The ablation. |
| `linucb` | Disjoint LinUCB over the 9-feature context. Learns during the evaluated run. |
| `*_pretrained` | Trained on held-out seeds, frozen, exploration off. |
| `oracle` | Best **phase-static** allocation, knowing the trace. A reference line, not a method. |

`epsilon_greedy` ignoring the context is the point, not a weakness: it isolates whether the context
vector is worth anything. `threshold` watches only the smoothed signal, also deliberately —
handing it the fast signal would close the gap the learned policy is supposed to exploit and make
the comparison uninformative.

## 4. Experimental protocol

- **Determinism.** A run is fully determined by `(policy, scenario, seed)`. Trace seeds derive
  from blake2b, not Python's `hash()`, which is salted per process; a test pins this.
- **Paired comparison.** A seed fixes the offered-load trace bit for bit, so every policy sees
  identical traffic on seed *k*. Differences are reported as the mean of **per-seed differences**
  with a t interval on that mean, not as the difference of two independent means.
- **Held-out seeds.** Hyperparameters were tuned, and the pre-trained variants trained, on
  `train_seeds = [100…104]`, disjoint from the evaluation seeds `[0…9]` and asserted disjoint by
  a test. Nothing was tuned or trained on a seed it was later scored on.
- **Scale.** 300 s per run, first 30 s discarded. 10 seeds per (policy, scenario). Run order
  randomized from a logged seed.
- **Both bandit variants reported.** Online (includes the cost of exploring — the honest
  deployment number) and converged (pre-trained, frozen). Reporting only the latter would delete
  exploration cost from the headline.

**Emulation-scaled SLO.** The proposal deck specified 15 ms / 25 ms. Those are *unreachable* here:
because a backlogged queue automatically claims a larger share of leftover capacity, URLLC
self-stabilises at bounded backlog, and the measured p95 ceiling across every action level is
about 9 ms. A 25 ms limit would yield zero violations for every policy and an idle guardrail.
Measured p95 by level (burst, seed 0):

| level | 0.20 | 0.35 | 0.50 | 0.65 | 0.80 |
|---|---|---|---|---|---|
| URLLC p95 (ms) | 2.00 | 3.39 | 5.27 | 7.16 | 9.04 |

The SLO was set from this range — `target 5.0 ms`, `hard 7.0 ms` — placing the decision boundary
between levels 0.50 and 0.65 so that policies land on different sides of it. We could have raised
the ceiling by inflating the scheduler quantum, but that is tuning the physics to fit the
threshold.

## 5. Results

### 5.1 The headline

Paired per-seed difference in mean reward against `threshold`, 10 seeds. **Bold** = 95 % interval
excludes zero. Positive is better than threshold.

| Policy | adversarial | burst | ramp | sawtooth |
|---|---|---|---|---|
| `linucb` (online) | **−0.0728 ± 0.0182** | **+0.0248 ± 0.0062** | **−0.0217 ± 0.0029** | **−0.0104 ± 0.0089** |
| `linucb_pretrained` | −0.0047 ± 0.0117 | **+0.0582 ± 0.0042** | **+0.0019 ± 0.0007** | **+0.0225 ± 0.0078** |
| `epsilon_greedy` | **−0.1281 ± 0.0092** | +0.0082 ± 0.0199 | **−0.0281 ± 0.0219** | −0.0201 ± 0.0235 |
| `epsilon_greedy_pretrained` | **−0.1287 ± 0.0114** | **+0.0269 ± 0.0190** | **+0.0019 ± 0.0007** | **+0.0236 ± 0.0078** |
| `oracle` | **+0.1241 ± 0.0115** | **+0.0589 ± 0.0042** | **+0.0019 ± 0.0007** | **+0.0358 ± 0.0077** |

**Online LinUCB loses on three of four scenarios**, and the losses are separated from zero, so
they are real rather than noise. This outcome was predicted in week one, before any policy
existed, with a written commitment to report it rather than tune until it disappeared.

**Converged LinUCB wins on three of four** and ties on adversarial. Crucially it does so with
*fewer* violations, not by trading safety for throughput — paired difference in SLA violation rate
against `threshold`:

| Policy | adversarial | burst | ramp | sawtooth |
|---|---|---|---|---|
| `linucb` (online) | **+0.70 ± 0.40 %** | **−0.78 ± 0.20 %** | **+0.33 ± 0.08 %** | 0.00 ± 0.25 % |
| `linucb_pretrained` | **−0.59 ± 0.42 %** | **−1.56 ± 0.11 %** | 0.00 ± 0.00 % | **−0.96 ± 0.22 %** |

Converged LinUCB is the only learned policy better than the reactive baseline on **both** axes at
once on more than one scenario.

**The entire difference between the online and converged rows is the cost of exploration.** Same
algorithm, same hyperparameters, same traffic; the only change is whether the model arrived
trained. For a deployment that can be trained offline before it goes live, that cost is avoidable.
For one that cannot, these results say a well-tuned threshold is the better choice.

### 5.2 Does the context earn its keep?

Yes, and the adversarial scenario shows it most sharply. That scenario alternates long calm
periods that lure a policy to an aggressive level with short URLLC spikes shorter than the EWMA
settling time. **All 562 violations it produces occur inside a spike; zero occur in calm phases.**
Violation rate within spike phases, by policy:

| Policy | violation rate in spikes |
|---|---|
| `epsilon_greedy` (context-free) | 30.4 % |
| `linucb` (online) | 19.2 % |
| `threshold` | 15.6 % |
| `linucb_pretrained` | **12.5 %** |
| `static_equal` | 0.2 % |
| `static_safe` / `oracle` | 0.0 % |

A context-free bandit is caught more than **twice as often** as a converged contextual one on
identical traffic. Since the two differ only in whether they observe the context vector, this is
direct evidence that the context carries information about an impending spike. The static
baselines' zeros are not evidence of safety — they simply never climb high enough to be caught.

### 5.3 What the guardrail does, and what it does not

The guardrail is active: intervention rates run from 0 % for baselines that never approach the
boundary up to 51.6 % for `epsilon_greedy_pretrained` on burst.

**It does not eliminate violations, and the report must not claim it does.** On adversarial,
`epsilon_greedy` violates on 5.85 % of steps *despite* the guardrail intervening on 23.4 %,
reaching a p99 of 12.56 ms against a 7.0 ms limit.

The reason is structural: **the guardrail acts on telemetry from the previous control interval.**
It cannot prevent the first interval of a spike it has not yet observed; it can only stop the
second and subsequent ones. What it does guarantee is narrower and stronger for being provable —
the applied action is always inside the allowed mask, whatever the policy proposes. That is
established by a property test against an adversarial policy that always demands the most
aggressive arm, over 5,000 randomized cases. It bounds the action space, not the violation rate.

**The fast path is load-bearing.** Design argued in week one from first principles, with no
evidence at the time, that a guardrail watching only smoothed latency would be late by
construction. Across all 320 runs (86,400 post-warm-up steps, 17,580 escalating):

| Signal that escalated | steps | share |
|---|---|---|
| EWMA only (slow path) | 838 | 4.8 % |
| **instantaneous p95 only (fast path)** | **6,740** | **38.3 %** |
| both | 10,002 | 56.9 % |

**An EWMA-only guardrail would have missed more than a third of all escalations.** This is the one
design decision argued a priori and confirmed by measurement.

### 5.4 Regret against the best phase-static allocation

Not regret against the optimum. The Oracle plays the best *phase-static* schedule under the same
guardrail, knowing the trace; a policy varying within a phase can beat it, so negative values are
legitimate. A true oracle over the joint action sequence is exponential (5^270) and unavailable at
any price. Positive means worse than the reference.

| Policy | adversarial | burst | ramp | sawtooth |
|---|---|---|---|---|
| `static_safe` | +0.4290 | +0.1083 | +0.2194 | +0.2219 |
| `static_equal` | +0.2392 | +0.0164 | +0.0618 | +0.0727 |
| `threshold` | +0.1241 | +0.0589 | +0.0019 | +0.0358 |
| `epsilon_greedy` | +0.2522 | +0.0507 | +0.0299 | +0.0559 |
| `linucb` | +0.1968 | +0.0342 | +0.0235 | +0.0462 |
| `linucb_pretrained` | **+0.1288** | **+0.0007** | **−0.0000** | **+0.0133** |

Converged LinUCB essentially closes the gap on burst and ramp and has the lowest regret of any
deployable policy on sawtooth. On adversarial it does not; `threshold` is marginally closer,
though the intervals overlap.

On `ramp`, `linucb_pretrained`, `epsilon_greedy_pretrained` and `oracle` are numerically identical
to four decimal places. This is not a coincidence: ramp's best phase-static schedule is a single
level for three of its four phases, both converged bandits find it, and all three play identical
actions. It also means **ramp cannot discriminate between these policies** and should not be read
as three independent confirmations.

### 5.5 Computational cost

| Policy | decision latency mean (ms) | p99 (ms) |
|---|---|---|
| static baselines | 0.021 – 0.023 | 0.040 – 0.049 |
| `threshold` | 0.035 – 0.040 | 0.073 – 0.115 |
| `epsilon_greedy` (both variants) | 0.058 – 0.066 | 0.117 – 0.163 |
| `linucb` (both variants) | 0.091 – 0.098 | 0.192 – 0.233 |

LinUCB costs roughly **4× an array lookup** and stays under a quarter of a millisecond at p99
against a 1,000 ms control interval — cheap relative to its own control loop by four orders of
magnitude. This supports the "lightweight" claim in the sense that matters.

**Caveat.** The operating system killed three long experiment jobs for memory during this work, so
these were measured on a machine demonstrably under memory pressure. Treat them as an
order-of-magnitude comparison, not a precise benchmark. Every other metric in this report is a
deterministic function of `(policy, scenario, seed)` and is unaffected.

## 6. Reward-weight sensitivity

The reward weights are a human choice, so the question is whether the §5 conclusion is a property
of the system or of that choice. 900 runs: `w_sla` in {0.25, 0.5, 1.0, 2.0, 4.0} x `w_drop` in
{0.0, 0.5, 1.0} x 5 policies x 4 scenarios x 3 held-out seeds.

Rewards from different cells are not comparable — changing a weight changes the units. Only the
*ordering* within a cell and the physical metrics are read across cells.

**The headline survives.** `linucb_pretrained` wins in all 15 cells. Tested per cell with a paired
interval over its 12 (scenario, seed) pairs against that cell's runner-up, the win is **separated
from zero in 13 of 15**. Both exceptions are at `w_sla = 0.25`, the weakest SLA penalty, where
there is least for a safety-aware policy to win — the direction the exception should fall if the
effect is real.

**The instability is real but sits below the winner.** Which baseline is *second* does move with
the weights: `threshold` at low `w_sla`, `static_equal` from `w_sla ≥ 2.0` with low `w_drop`, and
online `linucb` at the extreme corner. As the SLA penalty grows, the conservative static split
overtakes the aggressive reactive controller — the expected direction, and a weak check that the
reward behaves sensibly. The precise claim is therefore: **which baseline looks second-best
depends on the reward weights; which policy looks best does not.**

**Only the learned policies respond to the weights at all.** Averaged over `w_drop`, scenarios and
seeds:

| SLA violation rate | w_sla 0.25 | 0.50 | 1.00 | 2.00 | 4.00 |
|---|---|---|---|---|---|
| `linucb_pretrained` | 0.70 % | 0.74 % | 0.64 % | 0.66 % | **0.37 %** |
| `linucb` | 1.51 % | 1.46 % | 1.35 % | 1.17 % | 0.88 % |
| `threshold` | 1.54 % | 1.54 % | 1.54 % | 1.54 % | 1.54 % |
| `static_equal` | 0.00 % | 0.00 % | 0.00 % | 0.00 % | 0.00 % |

`threshold` and the static baselines are numerically invariant to the weights, to every digit —
they are fixed rules that never read the reward, so telling them to care more about latency
changes nothing they do.

The single cleanest statement of what the method buys is the `w_sla = 0.25` cell:
`linucb_pretrained` reaches **3.634 Mbps at 0.70 % violations** against `threshold`'s **3.655 Mbps
at 1.54 %** — within 0.6 % of the same throughput for less than half the SLA violations. Raising
`w_sla` sixteen-fold then cuts its violation rate to 0.37 % with no re-tuning, an adjustment
`threshold` cannot be asked to make.

**Caveat on the comparison.** The converged variant consumes five extra training runs per
evaluation, so against `threshold` this is not a like-for-like data budget. The defensible reading
is "a model trained offline beats a hand-tuned rule", not "learning is free". `threshold` had
human input too — its thresholds were set against the measured latency-by-level table — but that
is one-off human effort rather than per-deployment compute.

## 7. The simulator against a real switch

Every result in §5 and §6 was produced under `sim.excess_sharing: demand_proportional`: leftover
capacity goes to backlogged queues in proportion to their demand. That choice decides whether the
control problem exists. Under `equal` sharing, or sharing weighted by guaranteed rate, a backlogged
URLLC queue claims enough leftover capacity that its latency barely moves whatever cap eMBB is given,
and the whole study would be a study of an artifact. We flagged this in week one as the largest
threat to validity and built a testbed to settle it. Full record, with the reading rule for each
stage written before the stage ran: `docs/PLAN_TESTBED.md`.

### 7.1 Setup

Mininet and Open vSwitch 3.3.9 on WSL2 Ubuntu 24.04, kernel 6.6.87.2. Topology
`h1,h2,h3 — s1 ═ s2 — h4,h5,h6` with the 10 Mbps bottleneck on `s1 → s2`. Three HTB queues on
s1's bottleneck port, min-rates from `config/default.yaml`, eMBB ceiling set by the action level.
Static OpenFlow rules classify by transport port into the queues; ICMP shares the URLLC queue,
because ping is how URLLC latency is measured and must experience the same queue. iperf3 3.16 UDP
flows carry the load, `ping -i 0.05` measures URLLC round-trip time.

**The "SDN" here is OVS-native QoS, not an OpenFlow controller.** The data plane is real and
programmable; allocation decisions go over the OVS management interface (`ovs-vsctl set queue`),
not OpenFlow.

### 7.2 Three measurement problems, found and corrected before the comparison

**The cap check read the wrong number.** It took iperf3's `end.sum` as receiver goodput; on iperf
3.16 that field is the sender. Across a 1 Mbit cap `end.sum` read 4.002 Mbps and `sum_received`
0.976. Corrected, the cap binds exactly: HTB counts whole Ethernet frames, so a 3.5 Mbps cap on
1,400-byte payloads (1,442-byte frames) predicts 3.398 Mbps of payload, and we measured 3.399. The
void result is kept under a `superseded` filename.

**The testbed was closed-loop where the simulator is open-loop.** Offered 4× a 3.5 Mbps cap, the
sender sent 3.53 Mbps and nothing was dropped. With sender and switch in one kernel, packets queued
in the switch stay charged to the sending socket, which blocks at about 93 frames (134,106 B) before
the default 1,000-packet queue can fill. A real switch cannot block an application on another
machine, so this is an emulation artifact. The fix was to give each queue the finite buffer the
simulator already declared in week one, 62,500 B, not a value fitted to the testbed. Three queue
sizes were tested with predictions recorded first, and all three held:

| Leaf queue | Predicted | Measured at 4× the cap |
|---|---|---|
| OVS default, 1,000 packets | sender blocks | blocks; backlog max 134,106 B |
| 62,500 B | queue drops | **drops**; backlog max 62,006 B = 43 frames exactly; iperf3 loss and switch drops agree within one packet |
| 500,000 B (control) | sender blocks | blocks; backlog max 134,106 B again |

The control matters most: two very different limits stopped at the same 134,106 B, so the limit is
not what caps the backlog. The finite buffer is the testbed default from here on, which also means
simulator and testbed share the same tail-drop buffer by construction. The same run measured the
actuator: every `set queue max-rate` change reached the kernel shaper in 21 to 62 ms, at least 16
times inside the 1 s control interval.

**Host load creates a latency tail that queueing does not.** Idle URLLC RTT over 60 s: p50
0.093 ms, p95 0.133, p99 0.166, 1,182 samples, zero loss. That replaces the simulator's base latency
of 2.0 ms, a modelling constant about twenty times larger. Under load with the URLLC queue empty
(zero backlog, zero drops), the median was unchanged at about 0.1 ms but p99 rose to about 5 ms and
single replies reached 20 ms. Running ping at real-time priority did not remove it, so it is delay
below the probe: either the kernel packet path under load or Hyper-V descheduling the virtual CPU;
the test cannot separate them. **Median RTT is therefore the primary URLLC metric on the testbed**,
and p95 and p99 are read only against the loaded uncongested floor. A delivery metric that assumed
exact 0.05 s ping pacing was also fixed here; it now counts gaps in ICMP sequence numbers.

### 7.3 The level sweep: slicing works, and URLLC pays in latency

Constant offered load at the burst operating point (URLLC 3, eMBB 10, Best Effort 4 Mbps, L2), each
of the five eMBB levels held for 120 s with the first 10 s discarded, three repeats, shuffled order.
Calibration passed; 15 of 15 runs valid. Mean ± 95 % t interval, n = 3:

| eMBB level | eMBB Mbps | Best Effort Mbps | URLLC median RTT (ms) | URLLC p95 (ms) | URLLC drops |
|---|---|---|---|---|---|
| 0.20 | 1.995 ± 0.009 | 3.997 ± 0.015 | 0.10 ± 0.03 | 8.30 ± 10.00 | 0 |
| 0.35 | 3.270 ± 0.030 | 3.638 ± 0.010 | 2.63 ± 0.21 | 10.18 ± 2.85 | 0 |
| 0.50 | 4.082 ± 0.083 | 2.812 ± 0.020 | 3.65 ± 0.27 | 15.15 ± 9.31 | 0 |
| 0.65 | 4.424 ± 0.015 | 2.465 ± 0.022 | 4.44 ± 0.44 | 16.97 ± 0.69 | 0 |
| 0.80 | 5.087 ± 0.018 | 1.796 ± 0.013 | 7.67 ± 0.24 | 23.86 ± 5.99 | 0 |

- **The caps move capacity as programmed.** eMBB rises and Best Effort falls at every step, with
  goodput intervals no wider than ± 0.083 Mbps.
- **URLLC throughput is fully protected:** zero drops and 100 % probe delivery in all 15 runs.
- **Raising eMBB costs URLLC latency**, about 77-fold in the median from the lowest level to the
  highest, and the intervals of adjacent levels do not overlap.
- **Every packet is accounted for.** For eMBB and Best Effort, delivered plus switch drops is 99.6
  to 100.0 % of offered in every run.
- **p95 cannot separate the low levels.** At the uncongested 0.20 level it is 8.30 ± 10.00 ms
  (runs 12.33, 4.29 and 8.28), which is host jitter, as §7.2 predicted. The median separates every
  level.

The third point answers the question the testbed was built for. **On the real switch URLLC is not
protected for free; the control problem exists, by measurement rather than by assumption.**

### 7.4 Which sharing rule the switch follows

The simulator was run at the identical constant loads, with jitter off, the same 62,500 B buffers,
the same measurement window, and the measured 0.093 ms base latency, under each of its three sharing
modes. With zero jitter each cell is deterministic. The scoring rule was recorded before any of these
runs: for each metric, normalised mean absolute error (NMAE) is the mean error over the five levels
divided by the testbed's range for that metric, a mode *tracks* a metric at NMAE ≤ 0.25, and the
best mode has the lowest mean. URLLC p95 is excluded, since the simulator's p95 equals its median at
constant load while the testbed's is mostly host jitter.

| Mode | eMBB goodput | Best Effort goodput | URLLC median RTT | Mean NMAE |
|---|---|---|---|---|
| `demand_proportional` | 0.085 | 0.158 | 0.087 | **0.110** |
| `equal` | 0.167 | 0.256 | 0.476 | 0.300 |
| `min_rate_proportional` | 0.096 | 0.117 | 0.476 | 0.230 |

URLLC median RTT in ms, testbed against each mode:

| Level | Testbed | `demand_proportional` | `equal` | `min_rate_proportional` |
|---|---|---|---|---|
| 0.20 | 0.10 ± 0.03 | 0.09 | 0.09 | 0.09 |
| 0.35 | 2.63 ± 0.21 | 1.31 | 0.09 | 0.09 |
| 0.50 | 3.65 ± 0.27 | 3.12 | 0.09 | 0.09 |
| 0.65 | 4.44 ± 0.44 | 4.94 | 0.09 | 0.09 |
| 0.80 | 7.67 ± 0.24 | 6.76 | 0.09 | 0.09 |

**Classification: `TRACKS_demand_proportional`.** The simulator study was run under the one mode of
the three that the real switch supports. Figure: `results/summary/figures/fig6_testbed_sweep.png`.

**What is robust.** Only `demand_proportional` reproduces URLLC latency rising with the eMBB level.
The other two predict the base latency at every level, which the switch contradicts by about 85-fold
at level 0.80. Their URLLC NMAE of 0.476 fails any threshold that could reasonably be called
agreement, so this does not depend on the 0.25 cut-off.

**What is threshold-sensitive.** The label holds at thresholds of 0.20 and above. At 0.15 or
stricter it becomes `PARTIAL`, because Best Effort goodput (0.158) stops counting as tracked.

**What is systematic, not noise.** At every congested level each `demand_proportional` error is
larger than the testbed's own interval, and the goodput errors always point the same way:

| Level | eMBB goodput error | Best Effort goodput error | URLLC median error |
|---|---|---|---|
| 0.35 | −0.154 Mbps (−5 %) | +0.246 (+7 %) | −1.33 ms (−50 %) |
| 0.50 | −0.332 (−8 %) | +0.438 (+16 %) | −0.53 (−14 %) |
| 0.65 | −0.240 (−5 %) | +0.351 (+14 %) | +0.51 (+11 %) |
| 0.80 | −0.587 (−12 %) | +0.704 (+39 %) | −0.91 (−12 %) |

The real switch gives eMBB more and Best Effort less than the model. That `min_rate_proportional`
fits Best Effort best is consistent with HTB borrowing between eMBB and Best Effort being weighted by
their guaranteed rates while URLLC's latency behaves demand-driven. That is a hypothesis, not a
finding.

### 7.5 What this changes in §5 and §6

The largest threat to validity is narrowed rather than removed. The simulator's sharing rule is the
right one in kind, so the control problem the policies were evaluated on is real. It is biased in
degree, so absolute throughput numbers from the simulator should not be read as predictions for a
switch. The check covers one operating point at constant load; it does not show the simulator
agrees under the jittered scenarios the policies were scored on.

## 8. Limitations

Ordered by how much they threaten the conclusions.

1. **The simulator is checked at one operating point, at constant load** (§7). It tracks the real
   switch in kind and is systematically biased in degree. Agreement under the fluctuating traffic
   of §5 is not shown.
2. **No policy has run on the real switch.** Every policy result is a simulator result.
3. **The testbed cannot resolve tail latency at the scale the SLA uses.** Host jitter puts the
   uncongested p95 at 4 to 12 ms against a 7 ms limit, so p95 on this hardware measures the laptop
   as much as the queue (§7.2).
4. **The reward weights decide which baseline ranks second** (§6), though not which policy wins.
   The weights used throughout §5 are a defensible choice, not the only one.
5. **Rounds are not independent** (§2), which is not what a bandit assumes.
6. **Ten seeds of one simulator is not ten samples of reality.** The intervals describe seed
   variation and nothing else.
7. **The scenarios are hand-written**, and `adversarial` was designed against the guardrail's known
   weakness. It is a stress test, not a traffic model.
8. **The "SDN" is OVS-native QoS, not an OpenFlow controller** (§7.1).
9. **Single operating point** in the simulator too: one capacity, one set of guarantees, one control
   interval.

### What was not built

- **`OvsCliBackend` and a live policy loop on the switch.** The runner, policies and guardrail are
  written against a `NetworkBackend` interface and could drive the switch unchanged, but the
  guardrail's p95 input and 7 ms limit cannot be used on this hardware as-is (limitation 3).
- **`RyuBackend` and an OpenFlow controller app**, deliberately. Ryu is unmaintained, and a
  controller closes no validity gap that the OVS management interface leaves open.

**The defensible description of this work is a simulation study of a control policy and a safety
mechanism, whose central modelling assumption was checked against a real programmable switch** —
not an SDN controller evaluated end to end in emulation.

## 9. Conclusion

**The control problem is real.** On a real Open vSwitch, giving eMBB more of the link costs URLLC
about 77-fold in median latency while its throughput stays protected, and of three plausible
models of the scheduler only the one this study used reproduces that. The simulator is right in
kind and biased in degree, checked at one operating point.

Within that problem, a contextual bandit plus a deterministic guardrail is a workable design for
SLA-preserving dynamic slicing, with three qualifications that this study establishes rather than
assumes.

**Exploration is the deciding cost.** Online LinUCB loses to a well-tuned reactive threshold on
most scenarios; pre-trained and frozen, it wins on most, with fewer violations. If a deployment
can train offline, the bandit is worth it; if it must learn in production, a threshold is the
better engineering choice at this scale. Reporting only the converged number would have hidden
exactly this.

**Context is what the bandit contributes**, not learning as such. A context-free bandit is caught
by adversarial spikes more than twice as often as a converged contextual one on identical traffic.

**The guardrail's guarantee is narrower than it appears.** It bounds the action space provably;
it does not bound the violation rate, because it reacts to the previous interval. Systems claiming
safety from such a mechanism should state which of the two they mean.

The immediate future work is not a better algorithm. It is running the same policies live on the
switch, which first needs a guardrail input that host jitter cannot trip, and checking the
simulator under fluctuating rather than constant load.

---

## References

1. Koo et al., *Deep Reinforcement Learning for Network Slicing*, 2019.
2. Wei et al., *Dynamic Network Slice Reconfiguration using DRL*, 2020.
3. Dayot et al., *Deep Contextual Bandit-Based Slice Provisioning*, 2022.
4. *Safe and Fast Reinforcement Learning for Network Slicing Resource Allocation*, 2023.
5. Xie et al., *DRL-Based Resource Allocation for Network Slicing*, 2022.

No DRL baseline was implemented in this project. References [1], [2] and [5] are cited for
motivation and are **not** compared against by measurement.

---

## Reproducing

Simulator, any machine, no privileges:

```bash
pip install -r requirements.txt && pytest -q     # 233 tests
python experiments/run_suite.py                  # 320 runs -> results/raw/
python -m analysis.aggregate                     # every simulator table
python -m analysis.plots                         # every simulator figure
```

Testbed, Linux with root; setup in `docs/TESTBED_SETUP.md`:

```bash
python3 net/topology/slice_topo.py --check              # topology, queues, cap binding
python3 experiments/measure_noise_floor.py              # idle latency floor
python3 experiments/sweep_levels_ovs.py                 # 15 runs, about 32 min
python experiments/compare_sim_vs_ovs.py                # the §7.4 comparison
python -m analysis.plot_testbed                         # figure 6
```

Every table is recomputed from per-step logs rather than read from a cache, so a changed metric
definition changes the table instead of leaving a stale number behind.
