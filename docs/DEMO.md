# SafeSlice demo script

About 12 minutes, plus questions. Every command below runs on Windows in PowerShell, in a few
seconds, with no Mininet and no admin rights. The testbed half is shown from its recorded results,
because a live testbed run takes 23 to 32 minutes. An optional live testbed step is at the end, for
when WSL2 is working.

Numbers in quotes are what the command prints. They are deterministic: the same command prints
the same numbers every time.

---

## Before you start (5 minutes, before the audience arrives)

```powershell
cd C:\Users\prith\SafeSlice
python -m pytest -q
```

Expect `253 passed` in about 40 seconds. Leave that output on screen, since it's a good opening.

Open these in an image viewer, in this order, so you can switch with one click:

1. `results\summary\figures\fig1_timeseries_linucb_pretrained_adversarial_0.png`
2. `results\summary\figures\fig2_tradeoff.png`
3. `results\summary\figures\fig4_guardrail.png`
4. `results\summary\figures\fig6_testbed_sweep.png`

Also open `docs\REPORT.md` at section 7.5 (the live policy table), for the last part.

Make the terminal font large. The comparison table in part 4 is 108 characters wide.

---

## Part 1: the problem (1.5 minutes, no commands)

**Say:**

> Three kinds of traffic share one 10 Mbps link. URLLC needs low latency, eMBB wants as much
> throughput as it can get, and Best Effort takes what's left. A static split wastes capacity or
> starves URLLC during an eMBB burst.
>
> Every second, our controller picks one of five caps for eMBB, from 20 % to 80 % of the link. A
> contextual bandit, LinUCB, makes that choice. A deterministic guardrail sits between the bandit and
> the switch, so safety doesn't depend on the learned model being right.
>
> The question was never "can a bandit allocate bandwidth". It was "can a lightweight learner plus
> a safety mechanism beat a well-tuned hand-written rule, and what exactly does the safety mechanism
> guarantee?" We allowed the answer to be no.

---

## Part 2: one run, three policies (3 minutes)

All three see identical traffic: the burst scenario, seed 0.

```powershell
python experiments/run_experiment.py --policy threshold --scenario burst --seed 0
```

**Point at:** `SLA violation rate 1.48 %`, `eMBB goodput mean 2.746 Mbps`, `mean reward 0.1464`.

**Say:** "This is the competitor: a hand-tuned rule that steps the cap down when latency is high and
up when it's low."

```powershell
python experiments/run_experiment.py --policy linucb --scenario burst --seed 0
```

**Point at:** `0.74 %` violations, `2.747 Mbps`, reward `0.1669`, guardrail intervention `23.70 %`.

**Say:** "This is LinUCB learning from scratch during the run. Same throughput, half the
violations. But it explores, so the guardrail had to step in on almost a quarter of the steps."

```powershell
python experiments/run_experiment.py --policy linucb_pretrained --scenario burst --seed 0
```

(About 5 seconds, because it trains on five held-out seeds first.)

**Point at:** `0.00 %` violations, `2.918 Mbps`, reward `0.2029`, guardrail `0.00 %`.

**Say:** "This is the same algorithm, trained offline on seeds it's never scored on, then frozen.
More throughput, no violations, and the guardrail never had to act. The whole gap between the last
two runs is the cost of exploring online."

**Then show figure 2 (`fig2_tradeoff.png`).** "That was one seed. Here are all 320 runs: four
scenarios, ten seeds. Up is more throughput, left is fewer violations. Pre-trained LinUCB sits in
the top-left on burst, ramp and sawtooth."

**Say the honest version, because it's the headline:**

> Online LinUCB loses to the threshold rule on three of four scenarios. Pre-trained, it wins on
> three of four, with fewer violations. So if you can train offline, use the bandit. If it has to
> learn in production, a well-tuned threshold is the better engineering choice at this scale.

---

## Part 3: what the guardrail does and doesn't do (2 minutes)

**Show figure 1 (`fig1_timeseries_..._adversarial_0.png`).**

**Say:**

> The adversarial scenario was built to attack the guardrail: long calm periods lure the policy to a
> high cap, then a short URLLC spike hits. The bottom panel shows the proposed level (dashed) and
> the applied one (green). Every red dot is the guardrail overriding the policy and clamping to the
> safest level.
>
> But look at the top panel: there are still violations, the red dots above the 7 ms line. The
> guardrail acts on the previous second's telemetry, so it can't stop the first second of a spike it
> hasn't seen yet. We don't claim it eliminates violations.

```powershell
python -m pytest tests/test_guardrail.py -v
```

**Point at:** `test_applied_action_is_always_inside_the_mask`.

**Say:** "Here's what it does guarantee, and we can prove it: whatever the policy asks for, the
applied action is always inside the allowed set. This test runs an adversarial policy that always
demands the most aggressive cap, over thousands of random cases. It bounds the actions, not the
violation rate."

**Show figure 4 (`fig4_guardrail.png`), right panel.**

**Say:** "We decided before having any data to watch the instantaneous latency as well as the smoothed
one. The red bars are escalations only the fast signal caught: 38 % of all of them. A
smoothed-only guardrail would have missed more than a third."

---

## Part 4: is the simulator right? The real switch (3 minutes)

**Say:**

> All of that ran on our simulator, and it rests on one assumption: how a switch divides leftover
> capacity between busy queues. Under two of the three plausible rules, URLLC is protected for free,
> no controller is needed and our whole study is an artefact. So we built a Mininet and Open
> vSwitch testbed to settle it.

**Show figure 6 (`fig6_testbed_sweep.png`).**

**Say:** "These are real measurements on Open vSwitch, three repeats per level. Left: raising the eMBB
cap moves capacity from Best Effort to eMBB, exactly as programmed. Right: it costs URLLC about
77 times its median latency, from 0.10 to 7.67 milliseconds. URLLC never dropped a packet. Its
latency is what pays. So the control problem is real, measured rather than assumed."

```powershell
python experiments/compare_sim_vs_ovs.py
```

(About 5 seconds. It reruns the simulator under all three sharing rules and scores each one against
the recorded switch measurements.)

**Point at:** the `URLLC median RTT` block, then the last lines: `mean NMAE: demand_proportional
0.110 equal 0.300 min_rate_proportional 0.230` and `CLASSIFICATION TRACKS_demand_proportional`.

**Say:** "Two of the rules predict 0.09 ms at every level, which is flat and contradicted by the switch.
Only `demand_proportional`, the one our study used, tracks it. The simulator is right in kind but
biased in degree: it underestimates eMBB by 5 to 12 % and overestimates Best Effort by 7 to 39 %."

---

## Part 5: the policies live on the switch (2 minutes)

**Show `docs\REPORT.md` section 7.5, the table.**

**Say:**

> Last, we ran the same policy code, unmodified, live on the real switch. It was the same runner,
> guardrail and reward, with only the backend swapped. The control loop held on all four runs:
> every one of 1,080 steps measured and confirmed in the kernel shaper.
>
> But the ranking didn't transfer. The simulator predicted pre-trained LinUCB would come first. On
> the switch it tied for last.
>
> Here's why. It holds eMBB bursts at the 50 % cap, which the simulator rates safe: a steady
> 3.12 ms. On the real switch that level's latency jitters from second to second, and it crossed the
> 4 ms hard limit in 24 % of those seconds. Every one of its violations happened there. The
> guardrail contained the damage, but it couldn't make that operating point a good one.
>
> A policy trained in a simulator inherits the simulator's blind spots. That's one run per policy,
> so it doesn't rank the policies on hardware. But it tells us what to do next.

---

## Close (30 seconds)

> The control problem is real, and we measured it on a real switch. Within it, a bandit plus a
> guardrail works, with three qualifications we established rather than assumed: exploration is
> the deciding cost, the guardrail bounds actions but not violations, and a simulator-trained
> policy carries the simulator's blind spots. The next step isn't a better algorithm. It's a
> simulator with realistic latency noise, or training on the switch itself.

---

## Likely questions

| Question | Answer |
|---|---|
| Where's the SDN controller? | There isn't an OpenFlow controller. The "SDN" is Open vSwitch's own QoS, programmed over its management interface (`ovs-vsctl set queue`). A controller would add nothing the management interface leaves open, and Ryu is unmaintained. The report says so in section 7.1. |
| Why not deep RL? | Each decision's outcome is seen within one second, which is a bandit problem. DRL brings training cost and no bound on behaviour while learning. LinUCB decides in about 0.1 ms, against a 1,000 ms control interval. We did not implement a DRL baseline, so we don't claim to beat one. |
| Why five absolute caps, not up/down/hold? | With relative actions the current cap becomes part of the state, the problem becomes a full MDP and a bandit is the wrong tool. Absolute caps keep a bandit's assumptions approximately true. |
| Your SLO is 5 / 7 ms, but the proposal said 15 / 25 ms. Why? | On this link URLLC's p95 never goes above about 9 ms at any cap, so 25 ms would give zero violations for every policy and an idle guardrail. The limits were placed between the measured levels so policies land on different sides. |
| Isn't pre-training unfair to the threshold rule? | Partly, yes. The pre-trained model used five extra training runs. The fair reading is "a model trained offline beats a hand-tuned rule", not "learning is free". |
| Do the reward weights decide the winner? | Not the winner. A 900-run sweep over 15 weight settings: pre-trained LinUCB won all 15, clearly in 13. The weights do change which baseline comes second. |
| Is one run per policy on the switch enough? | No, not to rank policies on hardware, and we don't claim that. It's enough to show the predicted winner's operating point is unsafe on real hardware, and why. |
| Why median latency on the testbed, not p95? | With the URLLC queue empty, host jitter alone puts p95 at 4 to 12 ms. p95 on this laptop measures the laptop as much as the queue. The median separates every level cleanly. |

---

## If something goes wrong

- **A command prints different numbers:** check the scenario and seed. Every number above is for
  `--seed 0`.
- **No terminal access:** every number in this script is also in `docs/REPORT.md` (sections 5 and 7)
  and in `docs/SafeSlice_Project_Report.pdf`.
- **Asked to "show it on the real switch":** explain that the four live runs took 23 minutes, so
  you're showing the recorded results. Each run's per-step log is in
  `results\testbed_policies\ovs_cli\`.

---

## Optional: a live testbed step (needs WSL2 working)

On 2026-10-26, WSL2 would not start on this laptop. It reported that virtualization is not enabled.
Turning on virtualization in the BIOS and the "Virtual Machine Platform" Windows feature should fix
it. Test this the day before, not on the day.

If it works, this builds the real topology, installs the three queues and checks that the eMBB cap
binds. It takes about 10 seconds:

```bash
cd ~/SafeSlice && source .venv/bin/activate
sudo python3 net/topology/slice_topo.py --check --report /tmp/topo_check_demo.json
```

The `--report` path keeps the demo from overwriting the committed `results/summary/topo_check.json`.

**Point at:** the last check, where a UDP flow offered at twice the eMBB cap comes out at the cap.
**Say:** "That's the switch's shaper doing what the controller tells it to."
