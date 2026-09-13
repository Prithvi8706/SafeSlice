# PLAN: Mininet testbed track

Branch: `feature/mininet-testbed`. `main` is protected; this lands by PR.

Status, 2026-09-13:

- **Stage 1: PASS, on the corrected check.** The first run's cap check read iperf3's sender rate as
  receiver goodput and was void (kept at
  `results/summary/topo_check_superseded_sender_rate_bug.json`). The re-run reads the receiver
  explicitly and passes: `results/summary/topo_check.json`.
- **Stage 2: complete, CLEAN.** Idle RTT h1 to h4 through q0: p50 0.093 ms, p95 0.133 ms,
  p99 0.166 ms, max 7.850 ms, 1182 samples, 0 percent loss, 1226 ICMP packets counted in q0.
  `results/summary/noise_floor.json`. Unaffected by the stage 1 bug, since it used ping only.
- **Stage 1b, added: `BACKPRESSURE`.** Section 2.6 rule, result in section 2.7. The sender is held
  at the cap with zero drops even at 4x offered load. The simulator does not model this.
- **Stage 1c, added: all three predictions met.** Section 2.8. The finite 62,500 B per-queue buffer
  is now the default in `slice_topo.apply_qos` for every later stage.
- **Stage 2 remains valid under that change.** It measured idle RTT with empty queues, where
  queue depth does not matter. It does not describe the host under load; see section 2.9.
- **Stages 3 and 4, quick run: calibration passed; slicing works; two problems found.** Section 2.9.
  One metric bug fixed, and a latency tail under load that queueing cannot explain.
- **Stage 4a, added: attribute the tail, pending.** Section 2.9. Gates the full sweep.

---

## 0. What this track is for

Two goals, and one piece of work serves both.

1. **What the faculty asked for.** Demonstrate on real Mininet and Open vSwitch that the slicing
   works: three slices over a shared bottleneck, rate caps that actually bind, a protected slice
   whose latency stays low.
2. **What the project actually needs.** `docs/PLAN.md` set a gate in week one: compare the
   simulator against real OVS under an identical load, and stop if the gap invalidates the
   simulator's conclusions. That gate was never evaluated. Every number in `docs/EXPERIMENTS.md`
   is conditional on `sim.excess_sharing: demand_proportional` being how real OVS behaves, and
   nothing rules out that it is not.

Stage 4 below is the same experiment for both purposes. It is a Mininet demonstration that
slicing works, and it is the measurement that tests the load-bearing assumption.

**Scope decision, stated explicitly.** This is not the full testbed replication of the 320-run
simulator suite. Running the policy comparison on real hardware in real time would cost roughly
27 hours of VM time for one seed per cell, and it is not what closes the gap. The gap is closed
by a five-point fixed-level sweep. The live policy loop (stage 6) is the demonstration, not the
statistics.

---

## 1. What does not exist yet

| Missing | Purpose |
|---|---|
| `net/topology/slice_topo.py` | Mininet topology, OVS QoS, three HTB queues, classification flows |
| `experiments/measure_noise_floor.py` | Idle RTT distribution, from which the testbed SLO is derived |
| `traffic/generator.py` | iperf3 and ping orchestration |
| `experiments/sweep_levels_ovs.py` | The gate experiment: fixed-level sweep on real OVS |
| `net/ovs_cli_backend.py` | `NetworkBackend` implementation so existing policies run live |
| `net/ryu_backend.py`, `controller/ryu_app.py` | Out of scope for this track. See section 7. |

---

## 2. Design decisions that need your eyes

These are the ones where I am choosing between viable options, not filling in detail.

### 2.1 Classification into queues: static flows, no controller

`ovs-ofctl add-flow` rules on s1 matching on transport port, with `set_queue` then `normal`.
URLLC on 5201, eMBB on 5202, BE on 5203. ICMP goes to q0 alongside URLLC, because ping is how
URLLC's RTT is measured and it must experience the same queue.

Tradeoff: this means the "SDN" in the title is OVS-native QoS rather than an OpenFlow controller
making the decisions. The controller version is `RyuBackend`, which is out of scope here. The
honest framing for the report is that the data plane is real and programmable, and the control
decisions are taken by an external agent over the OVS management interface rather than over
OpenFlow. I would rather say that plainly than imply a controller exists.

### 2.2 Where the measurement of backlog comes from, and a divergence I cannot avoid

`TelemetrySample` carries `backlog_bytes_per_queue`, and `agent/context.py` exposes
`q0_backlog_norm` as a context feature. **Open vSwitch does not report queue backlog.**
`ovs-ofctl queue-stats` gives transmitted bytes, packets and errors, not instantaneous depth.

Options:

- **(a) Read it from `tc`.** `ovs-vsctl` configures an HTB qdisc that `tc -s class show dev <if>`
  can read, and that does report backlog per class. Requires mapping OVS queue numbers to tc
  class handles, which is version-dependent and is the fragile part.
- **(b) Report backlog as zero on the testbed** and accept that `q0_backlog_norm` is dead on this
  backend.

I propose (a) with an automatic fallback to (b), and the backend records which one it used in
`env.json` so no result can be read without knowing. If it falls back, **the context vector is not
the same object on the two backends**, and any sim-trained policy evaluated on OVS is running with
one feature pinned to zero. That must be stated wherever such a number appears.

### 2.3 RTT sampling rate

A 1 s control interval with `ping -i 0.2` gives 5 samples per interval, and the p95 of 5 samples
is just the maximum. That is not a percentile, it is a max with a misleading name.

Proposal: `ping -i 0.05`, 20 samples per interval. Needs root or a widened
`net.ipv4.ping_group_range`; you have root. Even 20 samples makes p95 coarse, so
`analysis/metrics.py` gets a recorded `rtt_samples_per_step` field and the report states the
sampling rate next to every testbed percentile. The simulator computes p95 over 100 sub-steps, so
**testbed and simulator percentiles are not computed from comparable sample counts** and the
comparison in stage 5 will be stated on p50 primarily, with p95 as supporting evidence.

### 2.4 The gate experiment uses constant load, not the seeded jittered traces

The simulator's traces apply per-second lognormal jitter. iperf3 cannot change its target rate
mid-flight, so exact per-second replay would mean restarting iperf3 every second, where connection
setup would dominate the measurement.

Proposal: **the gate experiment runs at constant offered rates with jitter disabled on both
sides.** This is not a compromise, it is a better experiment. It removes a confound, makes the
independent variable exactly known, and is the cleanest way to ask "at this offered load and this
cap, what does each system do". A `--no-jitter` path is added to `traffic/traces.py`, and the
simulator side of the comparison runs through the same code path as everything else.

Jittered seeded replay stays possible later for the live policy demo in stage 6, where iperf3 is
restarted at phase boundaries only and the actual achieved rate is recorded from iperf3's own
output rather than assumed.

### 2.5 Real time, and what that costs

The simulator runs in virtual time. Mininet does not. Budget:

| Stage | Runs | Wall clock |
|---|---|---|
| 2, noise floor | 1 x 60 s | ~2 min |
| 4, gate sweep: 5 levels x 3 repeats x 120 s | 15 | ~35 min plus setup |
| 6, live policy demo: 4 policies x 1 scenario x 300 s | 4 | ~25 min |

Under an hour and a half of VM time total. That is the reason for the narrow scope.

### 2.6 Does a full shaper queue push back on the sender? (added 2026-09-13, before measuring)

The simulator assumes traffic is **open-loop**: a source offers its rate whether or not the queue
is full, and the excess is tail-dropped. Every contention result in `docs/EXPERIMENTS.md` rests
on that, and so does the `w_drop` reward term, which `docs/DESIGN.md` section 5 already notes is
dominated by eMBB tail drops.

The first stage 1 run produced a reason to doubt it on this testbed. Asked to send 7 Mbps into a
3.5 Mbps cap, the iperf3 sender reported sending 3.53 Mbps. In a separate test across a
tbf-shaped veth with a small queue, the sender did send its full rate and the excess was dropped.
So the sender slowing itself down is not normal iperf3 behaviour. It is specific to something
about this topology, and the plausible candidate is the depth of the HTB leaf queue relative to
the sending socket's buffer. That is a hypothesis, not a finding.

`python3 net/topology/slice_topo.py --diagnose-backpressure` offers eMBB at 0.5x, 1x, 2x and 4x
the cap and reads each flow three ways: iperf3 sender, iperf3 receiver, and tc class counters.
Classification rule, fixed now, implemented in `slice_topo.py:classify_backpressure`:

| Label | Condition | What it means for the project |
|---|---|---|
| `GENERATOR_LIMITED` | sender below 90 percent of requested at 0.5x cap, the control row | The traffic generator is broken. Fix it; conclude nothing else. |
| `OPEN_LOOP` | sender at 90 percent of requested or more at 2x and 4x | The simulator's arrival model holds. Proceed to stage 3 as planned. |
| `BACKPRESSURE` | sender within 25 percent of the cap at 2x and 4x, whatever was requested | **The simulator's arrival model is wrong for this testbed.** Offered load cannot exceed the cap, tail drops become rare, the `w_drop` term reads near zero, and queues sit at a depth set by socket buffers rather than `sim.queue_limit_bytes`. This must be reported as a sim-vs-testbed divergence and resolved before stage 4, either by making the generator genuinely open-loop or by stating that the testbed is closed-loop and comparing on that basis. |
| `MIXED` | neither | Report the table. Draw no conclusion. |

### 2.7 Stage 1b result, and the mechanism test (recorded 2026-09-13, before stage 1c ran)

**Stage 1b classified the testbed `BACKPRESSURE`.** `results/summary/backpressure_diagnosis.json`.
eMBB cap 3.5 Mbps, UDP payload 1400 bytes:

| Offered | Sender | Receiver | Lost | tc drops | Backlog max |
|---|---|---|---|---|---|
| 0.5x (control) | 1.750 | 1.750 | 0 | 0 | 0 B |
| 1x | 3.500 | 3.383 | 0 | 0 | 112,476 B |
| 2x | 3.480 | 3.399 | 0 | 0 | 132,664 B |
| 4x | 3.530 | 3.399 | 0 | 0 | 134,106 B |

Two things this establishes, and one it suggests.

- **The shaper binds exactly.** HTB counts L2 bytes. A 1400-byte payload rides in a 1442-byte
  frame (plus 8 UDP, 20 IPv4, 14 Ethernet), so a 3.5 Mbps ceil predicts 3.398 Mbps of payload
  goodput. Measured at 2x and at 4x: 3.399 Mbps. The re-run stage 1 cap check also passes
  (`results/summary/topo_check.json`).
- **No packet is ever dropped, even at 4x the cap.** The sender stops instead.
- **Suggested, not proven:** the largest backlog, 134,106 B, is 93.0 frames. The default socket
  send buffer is 212,992 B, which is 2,290 B of socket accounting per frame, a plausible kernel
  per-packet overhead. The default leaf queue holds 1000 packets. If packets queued in the switch
  stay charged to the sending socket, which can happen when sender and switch share one kernel,
  the socket buffer is exhausted at about 93 frames and the queue never fills far enough to drop.

**Why the fix is not tuning the testbed to agree with the simulator.** A switch queue in a real
network cannot block an application on another machine. The coupling above exists only because
Mininet puts sender and switch in one kernel, so it is an emulation artefact. A real switch has a
finite per-queue buffer and tail-drops when it is full. The simulator already declares that buffer,
`sim.queue_limit_bytes: 62500`, set in week 1a and never tuned against anything. Using it is
choosing an independently fixed value, not fitting one.

**Stage 1c: the test, designed to confirm the mechanism rather than to find a setting that works.**
Three leaf-queue conditions in one run, each swept 0.5x to 4x, classified by the section 2.6 rule:

| Condition | Leaf queue | Predicted | Why |
|---|---|---|---|
| A | OVS default, 1000 packets | `BACKPRESSURE` | Reproduces stage 1b. |
| B | bfifo 62,500 B | `OPEN_LOOP` | Queue fills before the ~134 KB socket bound, so it drops first. Also expected: tc drops above zero, sender near requested, backlog max at or under 62,500 B. |
| C | bfifo 500,000 B | `BACKPRESSURE` | **The control.** Deeper than the socket bound, so the sender should block again. |

Interpretation, fixed now:

- A, B and C all as predicted: mechanism confirmed. All later stages use condition B's leaf queue,
  and the report states the artefact and the correction.
- B as predicted but C not: a small buffer fixes it, but the socket-buffer explanation is wrong.
  Adopt B anyway, since a finite switch buffer is correct on realism grounds alone, and report the
  mechanism as unexplained.
- B not `OPEN_LOOP`: the queue depth is not the cause. Stop before stage 3.

The same run also checks that `set_queue_max_rate` reaches the kernel shaper and how long it takes,
and that the bfifo leaves survive a rate change. Stage 6 depends on both.

### 2.8 Stage 1c result (2026-09-13)

`results/summary/backpressure_mechanism.json`. **All three predictions met.**

| Condition | Class | 4x: sender | 4x: iperf3 lost / tc drops | Backlog max, all rates |
|---|---|---|---|---|
| A, OVS default | `BACKPRESSURE` | 3.464 Mbps | 0 / 0 | 134,106 B = 93 frames |
| B, bfifo 62,500 B | `OPEN_LOOP` | 13.997 Mbps | 7,536 / 7,537 | 62,006 B = 43 frames |
| C, bfifo 500,000 B | `BACKPRESSURE` | 3.530 Mbps | 0 / 0 | 134,106 B = 93 frames |

What it establishes:

1. **All loss in condition B happens at the bottleneck queue.** Receiver-side iperf3 loss and the
   switch's tc drop counter are independent measurements, and they agree to within one packet at
   every rate: 30/30, 2,529/2,530, 7,536/7,537.
2. **The buffer behaves exactly as specified.** 62,500 B holds floor(62,500 / 1,442) = 43 frames =
   62,006 B, and the measured maximum was 62,006 B at every rate above the cap.
3. **The ~134 KB ceiling is independent of the queue limit.** Conditions A (1000 packets) and C
   (500,000 B) and the stage 1b run all stopped at exactly 134,106 B. So whether this testbed is
   open-loop is decided by queue depth against a fixed ~93-frame bound, not by the queue limit.
4. **The actuator is fit for a 1 s control loop.** Every `set_queue_max_rate` change reached the tc
   ceil in 21 to 62 ms, at least 16 times inside the control interval, and every bfifo leaf
   survived the change.

What it does not establish: that the 93-frame bound is specifically the socket send buffer. That
remains the best explanation (212,992 B / 93 = 2,290 B of accounting per frame) but the socket
buffer was never varied. The project's decision does not depend on it.

One measurement column is still unreliable: `tc_l2`. Three readings came in about 10 percent low
(3.161, 3.176, 3.154 Mbps) even after the stage 1b correction, most likely because the sampling
window can run past the end of the flow while iperf3 exchanges its final report. No conclusion
above uses it. Stage 4 bounds the counter window to the known flow interval instead.

**Decision, per the rule in section 2.7:** condition B's finite buffer is the testbed default from
here on, and the report states the emulation artefact and its correction. This changes one thing
the simulator comparison has to account for: the testbed and the simulator now share the same
per-queue tail-drop buffer by construction, so any divergence found in stage 5 cannot be blamed
on buffer size.

### 2.9 First quick sweep, and the latency tail (recorded 2026-09-13, before stage 4a ran)

`results/summary/ovs_level_sweep_quick.json`. **One repeat of 30 s per level, so no intervals and no
conclusions beyond direction.** Calibration passed: every flow delivered within 1 percent of
offered with zero drops, and every sender reached its target.

| eMBB level | eMBB L2 Mbps | BE L2 Mbps | URLLC L2 Mbps | URLLC drops | URLLC p50 ms | URLLC pooled p99 ms | URLLC max ms |
|---|---|---|---|---|---|---|---|
| 0.20 | 1.997 | 3.998 | 3.014 | 0 | 0.11 | 4.66 | 35.70 |
| 0.35 | 3.250 | 3.633 | 3.015 | 0 | 2.89 | 17.42 | 32.20 |
| 0.50 | 3.982 | 2.763 | 2.947 | 0 | 3.28 | 13.92 | 22.50 |
| 0.65 | 4.364 | 2.572 | 3.016 | 0 | 4.71 | 15.87 | 28.20 |
| 0.80 | 5.085 | 1.858 | 3.015 | 0 | 7.52 | 20.66 | 27.30 |

What holds even at one repeat:

- **The slicing works on real OVS.** eMBB goodput rises with every level, Best Effort falls with
  every level, URLLC median latency rises with every level, and at 0.20 eMBB delivers 1.997 Mbps
  against its 2.0 Mbps cap. This is the demonstration the faculty asked for.
- **URLLC throughput is fully protected.** 3.0 Mbps delivered and zero drops at every level. What
  the eMBB level takes from URLLC is latency, not throughput.

Two problems it exposed:

1. **A metric bug, now fixed.** `urllc_ping_delivered_pct` read 90.6 to 98.5 percent while ping's
   own summary showed 0.0 percent loss at every level. It divided the reply count by window /
   requested interval, and ping paces slower than requested on this host. It now uses sequence
   number gaps (`traffic/generator.py:ping_delivery`) and records the effective interval.
2. **A latency tail that queueing cannot explain.** At level 0.20 URLLC is uncongested: median
   backlog zero, median RTT 0.11 ms, the same as idle. Yet pooled p99 was 4.66 ms and max 35.7 ms,
   against 0.17 ms and 7.85 ms idle in stage 2. Same path, empty queue; the difference is 17 Mbps of
   other traffic on the same host. It also makes the per-window p95 non-monotone across levels.
   This matters directly: the reward and the guardrail's fast path use p95 against a 7 ms hard
   limit, and an uncongested p99 of 4.66 ms would put the guardrail partly at the mercy of the host.
   Note also that stage 2's `CLEAN` verdict was measured idle; the bands of
   `docs/TESTBED_SETUP.md` section 4 applied to this loaded but uncongested p99 would fall in `STOP`.
   The rule was written for the idle case and is not being retroactively applied, but the gap has
   to be resolved rather than ignored.

**Stage 4a, the test.** `experiments/diagnose_rtt_tail.py`. At eMBB level 0.20 so URLLC stays
uncongested, under the full burst load, measure URLLC RTT with ping at normal priority and at
real-time priority (`chrt -f 99`, confirmed available on this host), alternated idle, normal, RT,
normal, RT, 30 s each. Real-time priority removes delay in the ping process itself and leaves delay
in the kernel packet path. Rule, on mean pooled p99 across the two runs of each priority,
implemented in `diagnose_rtt_tail.py:classify_rtt_tail`:

| Label | Condition | Consequence |
|---|---|---|
| `NO_TAIL` | normal-priority p99 under 1 ms | The quick sweep's tail did not reproduce. Proceed to the full sweep. |
| `TOOL_SCHEDULING` | RT p99 under 1 ms | The tail was the probe. All URLLC latency measurement moves to RT-priority ping. The quick sweep's tail numbers are artefacts. |
| `PATH_JITTER` | RT p99 at least half of normal p99 | The tail is real packet-path delay on this emulator. p95 and p99 cannot be compared to the simulator at the 7 ms scale; the comparison in stage 5 is made on p50, the testbed SLO is re-derived from the loaded uncongested floor, and both are stated as limitations. |
| `PARTIAL` | anything else | More than half the tail was the probe and a real residual remains. Adopt RT ping, report the residual as the loaded floor. |

The run also records URLLC backlog and drops in every loaded condition. If URLLC was not actually
uncongested, part of the tail could be queueing, and the output says so.

---

## 3. Stages, in order, each with a verification you will see

I cannot run any of this. You run each command on the VM and paste the output; I work from that.
Nothing proceeds to stage N+1 until stage N has produced output you have seen.

### Stage 1: topology and queues

Build `net/topology/slice_topo.py`. Topology `h1,h2,h3 --- s1 === s2 --- h4,h5,h6`, bottleneck on
`s1 <-> s2`, QoS with three HTB queues on s1's egress port toward s2, min-rates from
`config/default.yaml` `slices.*.min_share`, max-rates from `allocation_from_level`.

Verify, and these are the numbers I need back from you:
- `pingall` reaches 100 percent.
- `ovs-ofctl -O OpenFlow13 queue-stats s1` lists exactly three queues on the bottleneck port.
- `tc -s class show dev <bottleneck-if>` shows three HTB classes with the configured rates.
- A single iperf3 UDP flow on port 5202 is capped at the programmed eMBB max-rate, within a few
  percent. **If the cap does not bind, stop. Nothing downstream is meaningful.**

### Stage 2: noise floor

Build `experiments/measure_noise_floor.py`. 60 s of idle RTT, p50/p95/p99, histogram, written to
`results/summary/noise_floor.json`.

This produces the first real number in the entire project and it decides the testbed SLO.
`config/default.yaml` currently carries `base_rtt_ms: 2.0`, which is a modelling constant, not a
measurement. Whatever this returns replaces it for the testbed config.

### Stage 3: traffic generator

Build `traffic/generator.py`. iperf3 servers on h4, h5, h6; UDP clients on h1, h2, h3 at
per-slice target rates; `ping -i 0.05` from h1 to h4 for URLLC RTT. Records achieved rate from
iperf3's own report, not the requested rate.

Verify: with all caps wide open, achieved rates match requested within a few percent, and the sum
does not exceed link capacity.

### Stage 4: the gate experiment, and the faculty deliverable

Build `experiments/sweep_levels_ovs.py`. Constant offered load matching the simulator's operating
point. For each of the five eMBB levels, hold it for 120 s and record URLLC p50/p95, eMBB goodput,
BE goodput, drops. Three repeats.

Output: the real-OVS counterpart to the table in `docs/DESIGN.md` section 3.

**This is the deliverable that shows slicing works.** eMBB goodput should rise with the level and
URLLC latency should rise with it too. If URLLC latency does not move at all, that is the `equal`
sharing outcome and it is a finding, not a failure.

### Stage 5: the comparison, and the decision point

Run the simulator at the identical constant loads under all three `sim.excess_sharing` modes.
Put the four columns side by side.

Three possible outcomes, and I am writing down now what each one means so the answer is not
chosen after seeing the data:

| Outcome | What it means | What we do |
|---|---|---|
| OVS tracks `demand_proportional` | The simulator's assumption holds. | `docs/EXPERIMENTS.md` loses its biggest caveat. The 320-run study stands. |
| OVS tracks `equal` | URLLC is protected for free by the scheduler. **The control problem does not exist at this operating point.** | Report it as the headline finding. The simulator study becomes a study of an artifact and must be relabelled as such. This is a legitimate and publishable outcome. |
| OVS tracks neither | The model is wrong in a way we did not anticipate. | Stop and report. Do not tune the simulator to match; that would make the agreement circular. |

### Stage 6: live policy loop on real OVS

Build `net/ovs_cli_backend.py` against the existing `NetworkBackend` ABC, so `static_safe`,
`static_equal`, `threshold` and `linucb_pretrained` run live on Mininet through the same runner,
the same guardrail and the same CSV schema. Four runs, 300 s each.

This is the strongest version of the demonstration: the same policy code, unmodified, driving a
real switch.

### Stage 7: documentation

Update `docs/DESIGN.md` section 2 (the assumption is now tested), `docs/SETUP.md` (remove the
"never built" banner for what now exists), `docs/EXPERIMENTS.md` (new testbed section),
`docs/REPORT_OUTLINE.md` section 8 (the top threat to validity is now resolved or confirmed).

---

## 4. What stays true regardless of outcome

- No number enters any document unless code that ran produced it. Otherwise `TBD`.
- If the testbed contradicts the simulator, the contradiction is the result and gets reported as
  the headline. The simulator does not get tuned until it agrees.
- Testbed and simulator SLA violation rates are not comparable until both SLOs are derived the
  same way. Stage 2 derives the testbed one.
- Tests that need Mininet are marked `@pytest.mark.requires_mininet` and stay skipped by default,
  so `pytest -q` keeps passing on a laptop.

## 5. What I need from you

Three facts I will not guess, none of which block stage 1 planning but all of which block stage 1
execution:

1. VM distro and version.
2. `python3 -V`, `mn --version`, `ovs-vsctl --version`, `iperf3 --version` on the VM.
3. Whether `ping -i 0.05 -c 5 127.0.0.1` works as your user, or needs sudo.

## 6. Risks

- **`ovs-vsctl set queue` reconfiguration latency.** Changing `max-rate` mid-run goes through
  ovs-vswitchd to tc. If that takes longer than a control interval, the 1 s loop is not viable on
  the testbed and the interval has to grow. Measured in stage 6, not assumed.
- **Per-queue drop counters may not be populated** on all OVS versions. Fall back to tc, and
  record which source was used.
- **Host CPU contention.** Six hosts, three iperf3 pairs and a fast ping on one VM. At 10 Mbps
  this should be comfortable, but if the VM is under-provisioned the latency measurements become
  a measurement of the hypervisor. Stage 2's noise floor is the check: if idle p99 is already
  near the SLO, stop and raise the SLO or the VM spec.

## 7. Explicitly out of scope

`RyuBackend` and `controller/ryu_app.py`. Ryu is unmaintained, its documentation points at
`os-ken`, and both depend on `eventlet` where Python-version breakage lives. It was optional in
the original plan and it is not what closes the validity gap. If stages 1 to 6 land with time to
spare, it is the obvious next thing.
