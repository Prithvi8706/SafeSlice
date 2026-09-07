# PLAN

Project: Safe Dynamic Network Slicing for SLA-Preserving SDN.
Branch: `dev`. `main` is unborn and stays untouched.
Status of this document: awaiting go-ahead. No implementation code written yet.

---

## 0. Ground truth about the starting state

The repository was empty at the start of this session. No commits, no files. Everything below is
new work, not a continuation.

The development sandbox is Windows 11 with Python 3.9.13. It has no Mininet, no Open vSwitch,
no iperf3, no root. Consequences:

- `SimBackend`, all policies, guardrail, reward, context, metrics, aggregation and their tests
  are written **and executed** here. Numbers from these are real.
- `OvsCliBackend`, `RyuBackend`, the Mininet topology, the traffic generator and
  `measure_noise_floor.py` are written here but **cannot be executed** here. Every number they
  would produce is `TBD` until you run them on the VM.
- Any table in any doc that would contain a testbed measurement gets `TBD` until it does not.

---

## 1. Decisions locked in from your answers

| Decision | Choice | Consequence |
|---|---|---|
| Sim clock | Virtual time | Full sim suite runs in minutes, not roughly 20 hours. Decision latency is still measured against a real wall clock per decision. |
| Sim fidelity | Independent idealized model, not fitted to OVS | The Week 1b sim-vs-OVS gap is a finding, not a tuning target. If the gap invalidates sim conclusions I stop and tell you. |
| URLLC load | CBR UDP carries the slice, ping measures its RTT | URLLC actually consumes bandwidth and can actually be starved, so the guardrail protects something real. |
| Week 1a batching | One batch | You review the whole pure-Python core at once. |

---

## 2. Open items I will not guess

1. **VM distro and Python version.** Blocks `docs/SETUP.md` and the Ryu / os-ken feasibility check.
   Not blocking for Week 1a.
2. **Whether `iperf3` and sub-second `ping` are available on the VM.** Sub-second ping intervals
   need root or a widened `net.ipv4.ping_group_range`. Blocks the traffic generator design in
   Week 2. Not blocking for Week 1a.

---

## 3. Deviations from the build order in your brief, with reasons

I am proposing three deviations. Say the word and I will revert any of them to your original ordering.

**D1. Scenario configs move from Week 2 into Week 1a.**
Reason: `SimBackend` cannot run without an offered-load trace, and the trace generator cannot be
tested without at least one scenario definition. Building all four now costs almost nothing extra
because they are YAML phase lists over the same generator.
Tradeoff: the scenarios get defined before we have seen any real testbed behaviour, so the
`adversarial` scenario in particular is a guess about what trips the guardrail. It will likely need
retuning after Week 1b. I will label it as provisional in the file.

**D2. `agent/reward.py`, `agent/context.py` and `agent/guardrail.py` move from Week 3 into Week 1a.**
Reason: your section 4 logging contract requires every CSV row to carry the context vector, the
proposed action, the applied action, the guardrail reason and the reward. A Week 1a pipeline that
is genuinely end to end therefore needs all three. The alternative is placeholder columns, which
violates the no-fabrication rule in spirit.
Tradeoff: Week 1a becomes noticeably larger. The bandits stay in Week 3, so Week 3 becomes
`LinUCB` plus `EpsilonGreedy` plus their tests plus tuning, which is still a full week of work.
This is the deviation most worth pushing back on. If you would rather keep Week 1a small, I will
ship it with a pass-through guardrail and a reward function only, and defer context features.

**D3. `Oracle` stays in Week 4 as your brief has it, but I want to flag now what it can and cannot be.**
A true oracle over the joint action sequence is exponential. Your brief says "best fixed-per-phase
allocation by brute force", which is tractable: for each scenario phase, sweep all levels, pick the
best by total reward, holding it fixed for the phase. That is an upper bound on *phase-static*
policies, not on all policies. A dynamic policy could in principle beat it within a phase.
Tradeoff: "regret vs Oracle" then means "regret vs the best phase-static allocation", and it can go
negative. I will name the metric accordingly and say so in the report rather than let it look like a
true optimum. Flagging this in Week 1 so it does not surprise anyone in Week 5.

---

## 4. SimBackend model, and the one modelling choice that decides whether this project exists

**Model: per-queue token bucket feeding a work-conserving shared link, simulated at a sub-step
granularity (default 10 ms) inside each 1 s control interval.**

Why a token bucket and not a fluid or M/M/1 model: the real actuator is an OVS HTB queue with
`other-config:max-rate`. That primitive is a token bucket. Matching it means the semantics of an
action ("cap eMBB at 0.5 of capacity") are identical in sim and on the testbed, which is the only
property that makes sim-tuned policies meaningful on real hardware. A fluid model would not
reproduce burst absorption, and an M/M/1 model has no rate cap at all, so the action space would
not exist.

**The choice that matters: queue service discipline.**

If the link served q0 with strict priority, URLLC would never queue, latency would never rise, the
guardrail would never fire and the bandit would have nothing to learn. The project would be
vacuous, and it would be vacuous in a way that flatters our method, because we would report zero
SLA violations and call it a result.

So the sim uses **rate-capped work-conserving sharing**, which is what OVS HTB actually does:
each queue has a min-rate guarantee and a max-rate cap, spare capacity is shared, and when the sum
of offered loads exceeds capacity every queue is served below its offered rate in proportion to its
share. URLLC then builds backlog and its latency rises, which is the phenomenon the whole system
is supposed to manage.

URLLC RTT per sub-step is modelled as `base_rtt + 2 * (q0_backlog_bytes / q0_service_bps)`, and the
control step reports p50 and p95 over the sub-steps in that interval. `base_rtt` is a config
parameter representing propagation and stack overhead.

**Risk I am flagging now, before writing a line of it.** This model makes URLLC latency a smooth
monotone function of q0 backlog, and q0 backlog a smooth monotone function of how much capacity
eMBB is allowed to take. A one-dimensional monotone relationship is exactly the situation where a
hand-tuned threshold is close to optimal and a contextual bandit has nothing extra to learn. If
that happens, `LinUCB` will tie or lose to `Threshold` in sim, and the honest conclusion will be
that at this scale learning is not warranted. I think that outcome is more likely than not in the
sim, and possibly less likely on the real testbed where noise, TCP dynamics and measurement jitter
give the context vector more to work with. I would rather say this in Week 1 than discover it in
Week 5. It does not change the plan. It changes what we expect the answer to be.

---

## 5. Week 1a deliverables, concretely

Files created, all executable and tested in this sandbox:

```
.gitignore
README.md                      minimal, expanded in Week 6
requirements.txt
Makefile                       Linux targets, noted as not runnable on the Windows sandbox
config/default.yaml            capacity, levels, SLO, reward weights, seeds, sim params
config/scenarios/burst.yaml
config/scenarios/ramp.yaml
config/scenarios/sawtooth.yaml
config/scenarios/adversarial.yaml   marked provisional, see D1
net/backend.py                 NetworkBackend ABC, TelemetrySample frozen dataclass, Allocation
net/sim_backend.py             the model in section 4
traffic/traces.py              deterministic per-seed offered-load traces from scenario YAML
agent/context.py               EWMA features, normalization, fixed feature order  [D2]
agent/reward.py                the weighted reward from your section 2.5          [D2]
agent/guardrail.py             pre-action mask, hard override, hold-down, logging  [D2]
agent/policies/base.py         Policy interface
agent/policies/static.py       StaticEqual, StaticSafe
analysis/metrics.py            single source of truth for the CSV schema and validator
experiments/run_experiment.py  one run: policy x scenario x seed, writes one CSV plus env.json
tests/test_sim_backend.py
tests/test_reward.py
tests/test_guardrail.py
tests/conftest.py              registers the requires_mininet marker, skipped by default
docs/DESIGN.md                 sections 2.1 (absolute actions) and 2.6 (emulation-scaled SLO)
```

Not in Week 1a: bandits, Mininet, OVS, Ryu, traffic generator, plots, aggregation, Oracle.

### Verification for Week 1a

1. `pytest -q` passes, with the Mininet-marked tests reported as skipped.
   Verify: I show you the actual pytest output, not a summary of it.
2. `python experiments/run_experiment.py --policy static_equal --scenario burst --seed 0` writes
   exactly one CSV whose schema passes the validator in `analysis/metrics.py`.
   Verify: I show you the command output and the head of the CSV.
3. The same command with `--policy static_safe` produces a different CSV, and `StaticSafe` shows
   lower eMBB goodput and lower URLLC p95 than `StaticEqual` on the same seed.
   Verify: I show you both numbers. If the ordering does not come out that way, the sim model is
   wrong and I stop and tell you rather than adjusting the model until it agrees.
4. Guardrail unit test asserts that across a generated sweep of contexts and proposed actions, the
   applied action is never a member of the masked set. Property style, not three hand-picked cases.
   Verify: pytest output.

Success criterion for the whole of Week 1a: three static-policy runs across three scenarios produce
schema-valid CSVs with no NaN in required columns, and check 3 above comes out in the predicted
direction.

---

## 6. Remaining weeks

Unchanged from your brief except for D1 and D2 above.

- **Week 1b.** Mininet topology and OVS queue setup, `OvsCliBackend`, `measure_noise_floor.py`.
  You run these. First real numbers. Then the sim-vs-OVS comparison under an identical trace.
  Gate: if the gap is large enough that sim-tuned policies would not transfer, I stop and report
  before anything is built on top.
- **Week 2.** Telemetry hardening, `traffic/generator.py` with iperf3 and ping, `Threshold` policy,
  `analysis/aggregate.py` and `analysis/plots.py`, first plots. Retune `adversarial.yaml` against
  observed testbed behaviour.
- **Week 3.** `agent/bandit.py` with `LinUCB` and `EpsilonGreedy`, `agent/policies/bandit_policy.py`,
  tests, alpha and epsilon tuning in sim. Explicit check of whether `LinUCB` separates from
  `Threshold` at all.
- **Week 4.** Ryu / os-ken feasibility check reported to you first. If it passes, `RyuBackend` and
  `controller/ryu_app.py`. If it does not, reward sensitivity study over `w_sla` plus `Oracle`.
- **Week 5.** Full suite. 6 policies by 4 scenarios by 10 seeds, randomized run order, environment
  fingerprint per run. Sim suite in full. Testbed suite at whatever scale your VM time allows, which
  I expect to be smaller than 10 seeds and which I will report as the reduced scale it is.
- **Week 6.** `docs/REPORT_OUTLINE.md` populated with real numbers only, README reproduction steps,
  cleanup.

---

## 7. Standing commitments for this project

- No number appears in any document unless code that ran produced it. Otherwise `TBD`.
- Both the online and the pre-trained bandit results get reported, even if online looks bad.
- If `LinUCB` does not separate from `Threshold`, that is the reported finding.
- If a metric definition would flatter our method, I raise it before computing it, not after.
- No commits without your explicit say-so. Staging plus `git diff --stat` only.
