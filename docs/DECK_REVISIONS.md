# DECK REVISIONS

`AI_Driven_Safe_Dynamic_Network_Slicing_Apple_LiquidGlass_Final.pdf` is the **proposal** deck. It
was written before any code existed and it describes a system that was not built. Presenting it
unchanged alongside these results invites an examiner to find the gap before you point it out.

This file is the slide-by-slide reconciliation: what each slide currently claims, whether the
implementation supports it, and the exact replacement wording. Verdicts are one of:

| | meaning |
|---|---|
| **OK** | claim is supported by code that ran |
| **FALSE** | claim describes something that does not exist |
| **WRONG NUMBER** | mechanism exists, stated value is not the implemented one |
| **OVERSTATED** | partly true, but claims more than was done |

The two most damaging are on slide 8, because they misdescribe the method itself rather than the
infrastructure: the action space and the safety rule.

---

## Slide 2 — Problem Domain & Motivation

> URLLC ... **"< 15 ms guardrail"**

**WRONG NUMBER.** The implemented guardrail is `warn 4.0 ms` / `hard 7.0 ms`, on a
simulator-derived latency scale. 15 ms is not merely different, it is *unreachable* in this
simulator — the measured p95 ceiling across every action level is about 9 ms, so a 15 ms rule
would never fire and the guardrail would be decorative.

**Replace with:** `latency-critical · 7 ms SLA limit (simulator-derived, see note)`

Add a footer to this slide: *SLO derived from the simulator's measured latency range, not from
hardware. `docs/DESIGN.md` §3.*

---

## Slide 5 — Research Gap

> **"Real-Time SDN Telemetry (collecting live network state)"**

**FALSE as an achievement, OK as a gap statement.** This slide describes the gap in the
literature, so it can stand — but only if slides 6–9 stop claiming you filled the SDN half of it.

**No change needed here**, provided the changes below are made.

---

## Slide 6 — Proposed Solution

> **"Mininet + Open vSwitch + Ryu Controller + Contextual Bandit decision engine"**

**FALSE for three of the four named components.** Of the seven pipeline stages:

| Stage | Status |
|---|---|
| 1. Traffic Generation (Mininet hosts) | **FALSE** — offered load is a seeded synthetic trace, `traffic/traces.py` |
| 2. Mininet + OVS data plane | **FALSE** — never built |
| 3. Ryu Telemetry Collection | **FALSE** — telemetry comes from the simulator |
| 4. Contextual Bandit | **OK** — `agent/bandit.py`, disjoint LinUCB |
| 5. SLA Safety Guardrail | **OK** — `agent/guardrail.py` |
| 6. Dynamic Queue / **OpenFlow meter** update | **FALSE** — the action reaches a simulated token bucket, not a switch |
| 7. Network Response | **OK** in simulation |

**Replace the header with:** `Discrete-event queue simulator + Contextual Bandit decision engine
+ deterministic SLA guardrail`

**Replace stages 1–3 and 6** with: `1. Seeded traffic trace → 2. Token-bucket queue model
(HTB semantics) → 3. Per-interval telemetry → 4. Contextual Bandit → 5. SLA guardrail →
6. Rate-cap update → 7. Measured response`

Keep the original seven-stage diagram as a **"future work / testbed architecture"** slide if you
want to show the intended design — clearly labelled as not built.

---

## Slide 7 — Objectives & Novelty

| Bullet | Verdict |
|---|---|
| "Implement SDN network slicing using Mininet and OVS" | **FALSE** |
| "Collect real-time throughput, latency and packet-drop telemetry" | **OVERSTATED** — collected per control interval, from the simulator |
| "Develop lightweight Contextual Bandit resource allocation" | **OK** |
| "Protect URLLC using an SLA guardrail" | **OK** |
| "Compare static and dynamic slicing performance" | **OK** — six policies, four scenarios, ten seeds |
| Novelty: "Dynamic meter and queue control in SDN" | **FALSE** |

**Replace bullet 1 with:** `Model SDN slice scheduling with HTB token-bucket semantics`
**Replace bullet 2 with:** `Collect per-interval throughput, latency and drop telemetry`
**Replace the novelty bullet with:** `Deterministic guardrail whose safety property is proved,
not merely observed`

That last one is a stronger novelty claim than the one it replaces, and unlike it, it is true:
the mask invariant is established by a property test against an adversarial policy over 5000
randomized cases, not by the runs happening to come out well.

---

## Slide 8 — Methodology ⚠️ the two that matter

> **"Actions: Increase, Maintain, or Decrease eMBB Rate"**

**FALSE, and it contradicts the method.** The implemented action space is **five absolute rate
caps** — `[0.20, 0.35, 0.50, 0.65, 0.80]` of link capacity. Relative actions were explicitly
*rejected*, and the reason is not cosmetic: under relative actions the effect of an action depends
on the current allocation, so the allocation becomes part of the state, the problem becomes a full
MDP, and **a contextual bandit is the wrong tool for it** — LinUCB's regret bound assumes reward
depends on context and arm alone. Presenting relative actions while claiming a bandit is a
methodological contradiction an examiner may well catch. `docs/DESIGN.md` §1.

**Replace with:** `Actions: one of five absolute eMBB rate caps (0.20 – 0.80 of capacity)`

> **"Safety Rule: If URLLC Latency exceeds 15 ms, restore maximum URLLC isolation immediately"**

**WRONG NUMBER and oversimplified mechanism.** The guardrail has three tiers, not one:

- **warn band** (> 4 ms): aggressive levels are masked out and the policy is asked to re-choose
- **hard override** (> 7 ms): clamp to the safest level and arm a hold-down
- **hold-down** (3 steps): stay clamped so the controller cannot oscillate straight back

**Replace with:** `Safety rule: above 4 ms mask aggressive actions; above 7 ms clamp to the safest
allocation and hold for 3 intervals`

> State: "eMBB Throughput, eMBB Drop Rate, URLLC Latency"

**OVERSTATED** — the actual context is **9 features**, including both a smoothed *and* an
instantaneous latency signal. That distinction is worth a slide of its own (see below), because
it is one of the few design decisions the measurements confirmed.

> Topology H1–H3 → S1 → S2 → H4–H6, queues Q0/Q1/Q2

**FALSE as built, OK as a model.** Relabel the diagram *"modelled topology"*.

---

## Slide 9 — Tools & Work Plan

| Tool | Status |
|---|---|
| Python, CSV, Matplotlib | **OK** |
| Mininet, Ryu, Open vSwitch, OpenFlow 1.3, iperf3 | **FALSE — remove or move to "future work"** |

Add: **NumPy, pandas, SciPy, pytest**.

The 6-week plan is superseded; replace it with what was actually done (see `docs/PLAN.md` status
note) or drop it — a final-presentation deck rarely needs the original schedule.

---

## Slides you do not have and now need

The deck is a proposal and contains **no results**. At minimum add:

**R1 — Headline.** Online LinUCB *loses* to a hand-tuned threshold on 3 of 4 scenarios; the
pre-trained frozen variant wins on 3 of 4 with fewer SLA violations. The whole difference is the
cost of exploring online. Use the paired table from `docs/EXPERIMENTS.md` §3.

Lead with this even though it is not the flattering result. It is the honest one, it was
*predicted in week one before any policy existed* (`docs/PLAN.md` §4), and "we predicted this
failure mode in advance and reported it" is a much stronger position than being asked why the
method underperforms and having no answer ready.

**R2 — The guardrail's fast path.** 38.3 % of escalations were caught by the instantaneous p95
alone, against 4.8 % by the EWMA alone: an EWMA-only guardrail would have missed over a third.
This is the design decision argued from first principles in week 1a and confirmed by measurement,
and it justifies the 9-feature context that slide 8 currently undersells.

**R3 — What the guardrail does not do.** It acts on the *previous* interval's telemetry, so it
cannot stop the first interval of an unanticipated spike. Measured: 5.85 % violations on
adversarial despite intervening on 23.4 % of steps. Stating this yourself is far better than
being asked "so does it guarantee the SLA?" and having to concede it live.

**R4 — Limitations.** The testbed was not built, so the simulator's load-bearing assumption
(`sim.excess_sharing`) is unverified and every result is conditional on it. Use
`docs/REPORT_OUTLINE.md` §7.

---

## The one framing change that matters most

Anywhere the deck implies a working SDN deployment, say **simulation study** instead. The
defensible sentence is:

> *A simulation study of a safe contextual-bandit slicing controller, with testbed evaluation as
> future work.*

Not:

> *An SDN slicing system implemented in Mininet and evaluated in emulation.*

`docs/REPORT_OUTLINE.md` §3 lists the specific sentences the write-up may not contain. The same
list applies to what you say out loud while presenting.
