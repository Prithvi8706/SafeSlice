# PLAN: Mininet testbed track

Branch: `feature/mininet-testbed`. `main` is protected; this lands by PR.

Status: awaiting go-ahead. No implementation written yet.

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
