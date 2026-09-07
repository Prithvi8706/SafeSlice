# REPORT OUTLINE

What the write-up says, section by section, and — more usefully — the claims it is **not**
allowed to make. Numbers live in `docs/EXPERIMENTS.md`; this file is the argument and its limits.

Status: populated from the simulator suite. No testbed work exists. See section 8.

---

## 0. The one-paragraph version

A contextual bandit picks one of five absolute eMBB rate caps every second on a shared 10 Mbps
bottleneck, and a deterministic guardrail clamps its choice whenever URLLC latency approaches the
SLA. Across four traffic scenarios and ten seeds on the simulator, **LinUCB learning online loses
to a hand-tuned reactive threshold on three of four scenarios**; the same algorithm pre-trained on
held-out seeds and frozen beats that threshold on three of four, with fewer SLA violations, and
essentially closes the gap to the best phase-static allocation on two. The entire difference
between those two rows is the cost of exploring online. A context-free bandit does substantially
worse than either, so the context vector is carrying real information. Decisions cost roughly four
times an array lookup and stay under a quarter of a millisecond against a one-second control
interval. Whether any of this transfers to real Open vSwitch is **untested** — the testbed was
never built.

---

## 1. Sections

1. **Problem.** Static slicing cannot react to demand; DRL approaches are heavy. A contextual
   bandit is the lightweight middle, but a bandit alone gives no safety guarantee, which is why
   the guardrail is a separate deterministic mechanism rather than a term in the reward.
2. **System.** Three queues on one bottleneck, absolute rate-cap action space, 1 s control
   interval. `docs/DESIGN.md` section 1 for why absolute and not increase/maintain/decrease.
3. **Method.** Context (9 features), reward, guardrail, LinUCB. The guardrail sits *between*
   policy and actuator and re-asks the policy rather than overruling it.
4. **Experimental protocol.** Determinism, seeds, warm-up, paired comparison, randomized run
   order, held-out tuning and training seeds.
5. **Results.** `docs/EXPERIMENTS.md` sections 3 to 6.
6. **Sensitivity.** Whether the ordering of policies survives changes to the reward weights. It
   does not: three policies win somewhere in the sweep. What does survive is that only the
   learned policy responds to the weights at all.
7. **Limitations.** Section 8 below, in full, not compressed to a sentence.

---

## 2. Claims the report is allowed to make

Each of these is supported by a number in `docs/EXPERIMENTS.md` produced by code in this
repository that ran.

- The applied action is always inside the guardrail's allowed mask, **regardless of what the
  policy proposes**. This is a property of `agent/guardrail.py` proved by a property test against
  an adversarial policy that always demands the most aggressive arm, over a 5000-case randomized
  sweep — not an empirical observation that happened to hold on the runs we did. Note carefully
  what this is and is not: it bounds the action space, not the violation rate. See the next
  section.
- Decision latency of the learned policy is sub-millisecond and within an order of magnitude of
  an array lookup, measured against a real wall clock per decision.
- The comparison between policies is on identical traffic: a seed fixes the offered-load trace
  bit for bit, so every difference reported is paired.
- Hyperparameters were selected on seeds disjoint from the evaluation seeds, and the pre-trained
  variants were trained on those same held-out seeds. Nothing was tuned or trained on a seed it
  was scored on.
- **The learned policy is the only one whose behaviour responds to the reward weights.** Raising
  `w_sla` sixteen-fold moves LinUCB's violation rate from 1.51 % to 0.88 % and its eMBB goodput
  from 3.524 to 3.315 Mbps, with no re-tuning; `threshold` and both static baselines are
  numerically invariant to every digit, because they are fixed rules that never read the reward.
  This is the claim about learned control that the sensitivity study actually supports.

## 3. Claims the report is NOT allowed to make

This list exists because each of these is a sentence that would be easy to write and wrong.

- **"Validated on Mininet / Open vSwitch / a real SDN testbed."** Nothing here has touched any of
  them. There is no `OvsCliBackend`, no Ryu app, no topology, no iperf3 traffic. The word
  "emulation" must not be used for what is a simulation.
- **"Regret against the optimal allocation."** The Oracle is the best **phase-static** allocation
  under the same guardrail. A policy that varies within a phase can beat it, so the number can
  be negative. Say "regret against the best phase-static allocation" or do not say regret.
- **"p99 latency."** `urllc_rtt_p99` is the 99th percentile of a series of per-step p95 values.
  It is not a packet-level p99 and must not be labelled as one.
- **"Zero SLA violations proves the method is safe."** The static baselines also show zero, and
  they show it because they cannot reach the levels that violate, not because anything protected
  them. A zero on a policy that never approaches the boundary is not evidence about the guardrail.
- **"The guardrail prevents SLA violations."** It does not. It acts on telemetry from the previous
  control interval, so it cannot prevent the first interval of an unanticipated spike — only the
  ones after it. Measured: on `adversarial`, `epsilon_greedy` violates on 5.85 % of steps while
  the guardrail intervenes on 23.4 % of them, reaching a p99 of 12.56 ms against a 7.0 ms limit.
  The guardrail bounds exposure; it does not remove it. `docs/EXPERIMENTS.md` section 5.
- **"The guardrail intervened on X % of steps, therefore it prevented X % violations."**
  Intervention is `applied_action != proposed_action`. Some of those interventions would not have
  caused a violation. The counterfactual was not run.
- **"w_drop penalises congestion."** In these runs `total_drops` is dominated by eMBB tail drops
  caused by offering far more than the cap allows, so raising the cap *reduces* drops. The term
  behaves as a second throughput incentive. `docs/DESIGN.md` section 5.
- **"The simulator shows URLLC latency reaching X ms, so the SLA should be Y."** The simulator's
  achievable latency range is a consequence of one unverified modelling assumption about how
  leftover capacity is shared. `docs/DESIGN.md` section 2.
- **"LinUCB is better than DRL for this problem."** No DRL baseline was implemented. The
  comparison in the literature survey is a citation, not a measurement.
- **"Policy X is the best."** Unqualified, this is not supportable. The reward-weight sweep puts
  three different policies in first place depending on `w_sla` and `w_drop`
  (`docs/EXPERIMENTS.md` section 8). Any ranking claim must name the weights it holds under.
- **"The sensitivity study shows the headline result is robust."** It does not test the headline
  result. It sweeps *online* LinUCB; the converged variant, which is the one that wins in
  section 3, was not included. The robustness of the actual headline to the reward weights is
  untested.

---

## 4. The result that would have been reported if it had come out that way

Recorded here because a project that only ever reports the outcome it hoped for is not evidence
of anything. `docs/PLAN.md` section 4 predicted, in week 1 and before any policy existed, that
LinUCB might fail to separate from a hand-tuned threshold, and committed to reporting that as the
finding rather than tuning until it went away.

**That is roughly what happened, and it is reported.** Online LinUCB is worse than `threshold` on
`adversarial` (−0.073 reward), `ramp` (−0.022) and `sawtooth` (−0.010), all separated from zero,
and better only on `burst` (+0.025). The prediction was not that the method is worthless — the
converged variant does win — but the headline a reader would most want, "our contextual bandit
beats the baselines", is **not supported for the online case** and the write-up must not imply it.

Two smaller instances of the same commitment:

- The ε sweep selected ε = 0, meaning exploration did not pay for the context-free bandit. The
  selection rule was followed rather than revised after seeing the answer, even though ε = 0 makes
  the "epsilon-greedy" label a misnomer and yields the strongest, hardest-to-beat version of that
  baseline.
- On `ramp`, `linucb_pretrained`, `epsilon_greedy_pretrained` and `oracle` produce numerically
  identical results. Rather than present three matching columns as three confirmations, the tables
  say the scenario cannot discriminate between them.

See `docs/EXPERIMENTS.md` sections 2, 3 and 4.

---

## 5. Figures

| Figure | Question it answers |
|---|---|
| `fig1_timeseries_*` | Does the controller do anything? Latency, chosen level, guardrail events over one run. |
| `fig2_tradeoff` | The headline: eMBB goodput against SLA violation rate, per policy per scenario. |
| `fig3_bars` | The same with 95% intervals, so overlapping bars are visible as non-results. |
| `fig4_guardrail` | How often the guardrail fired and which signal tripped it — the evidence for watching the instantaneous tail as well as the EWMA. |
| `fig5_learning_*` | Reward against step, warm-up included, so the cost of online exploration is visible rather than cropped. |

---

## 6. What a reader should be able to reproduce

```bash
pip install -r requirements.txt
pytest -q
python experiments/run_suite.py     # ~90 min, writes results/raw/
python -m analysis.aggregate        # the tables in docs/EXPERIMENTS.md
python -m analysis.plots            # the figures above
```

Every table in `docs/EXPERIMENTS.md` is regenerated by `analysis/aggregate.py` from the per-step
logs, not copied from a cache, so a changed metric definition changes the table rather than
leaving a stale number behind.

---

## 7. Threats to validity, in the order they matter

1. **The simulator is unvalidated.** One assumption — `sim.excess_sharing` — decides whether the
   control problem exists at all. Under the `equal` setting URLLC is largely protected for free
   and no controller is needed. We chose the pessimistic setting and said so, but "we chose the
   setting under which our method is useful" is a real threat and is stated as one.
2. **Rounds are not independent.** Queue backlog and autocorrelated load both carry state across
   control intervals, which is not what a bandit assumes. `docs/DESIGN.md` section 1 sets out why
   this is tolerable and what it would explain if LinUCB underperformed.
3. **Ten seeds of one simulator is not ten samples of reality.** The confidence intervals describe
   seed variation, nothing more.
4. **The scenarios are hand-written**, and `adversarial.yaml` in particular was designed against
   the guardrail's known weakness. It is a stress test, not a traffic model.
5. **Single operating point.** One capacity, one set of min-rate guarantees, one control interval.
6. **The reward weights decide the ranking.** `docs/EXPERIMENTS.md` section 8. The `w_sla = 1.0`
   column used throughout section 4 is a choice, and a different defensible choice reorders the
   podium.

---

## 8. What was not built, and what that costs the conclusions

The Mininet/OVS/Ryu track from `docs/PLAN.md` weeks 1b, 2 and 4 does not exist. Consequences,
stated plainly:

- There is no measured noise floor, so the SLO used here (`target 5 ms`, `hard 7 ms`) is derived
  from the simulator's own achievable range, not from hardware. The slide deck's 15 ms / 25 ms
  figures are unreachable in this simulator and were not used.
- The sim-vs-OVS comparison that would test the load-bearing assumption was never run, so the
  gate `docs/PLAN.md` set for itself — stop if the gap is large enough that sim-tuned policies
  would not transfer — was never evaluated.
- The OpenFlow meter / queue actuation path in the proposal deck (stage 6) is unimplemented. The
  action reaches a simulated token bucket, not a switch.

The right sentence for the report is: *this is a simulation study of a control policy and a
safety mechanism, with a testbed evaluation left as future work* — not *an SDN implementation
evaluated in emulation*.
