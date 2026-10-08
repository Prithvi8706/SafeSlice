"""OvsCliBackend: the NetworkBackend contract on real Mininet and Open vSwitch.

Stage 6 of docs/PLAN_TESTBED.md. The runner, the policies, the guardrail, the reward and the CSV
schema are the same objects that produced the simulator study; only this file differs. It drives
the testbed over the OVS management interface (`ovs-vsctl set queue`) and reads it through tc and
ping, which is what stages 1 to 4 established as trustworthy.

TIMING
`read_telemetry()` blocks until the next interval boundary, t0 + k * interval, then reads. If the
caller arrives after the boundary (a slow decision or a slow actuator call) it does not sleep, and
how late it was is logged per step in `step_log`, so an overrun is a recorded number rather than a
silent stretch of the control interval.

WHERE EACH TELEMETRY FIELD COMES FROM
- Goodput, URLLC throughput, drops: tc class counters on the bottleneck, differenced between
  consecutive reads. L2 bytes, the same units as the simulator's capacity.
- Backlog per queue: tc's instantaneous class backlog at the read. This is option (a) of
  docs/PLAN_TESTBED.md section 2.2. There is no fallback to zero: if tc cannot be read the run
  stops, because a context feature silently pinned to zero is worse than no run.
- URLLC RTT p50 and p95: the ping replies timestamped inside the interval. If no reply arrived in
  an interval the previous value is carried forward and the step is counted in `step_log`; it is
  never filled with a guess.
- Offered load: the trace's per-step target, which is what iperf3 was asked for. Whether the
  sender achieved it is checked afterwards from iperf3's own report (`segment_reports`).

TRAFFIC
iperf3 cannot change its rate mid-flight, so offered load must be piecewise constant (scenario
jitter off; docs/PLAN_TESTBED.md section 2.4). Each constant stretch of each slice is one iperf3
client, started at the interval boundary where the stretch begins. Consecutive stretches of a slice
alternate between its two server ports (SLICE_PORTS, SLICE_ALT_PORTS), so a new client never finds
its server still closing the previous test. The cost is an overlap of one connection setup, tens of
milliseconds, at each rate change.
"""

from __future__ import annotations

import math
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np

from experiments.measure_noise_floor import parse_ping_output
from net.backend import EMBB, N_QUEUES, Allocation, NetworkBackend, TelemetrySample
from net.topology.slice_topo import (
    SLICE_ALT_PORTS,
    SLICE_HOSTS,
    SLICE_PORTS,
    SLICE_QUEUE,
    UDP_PAYLOAD_BYTES,
    apply_qos,
    parse_iperf3_udp,
    parse_tc_rate,
    read_tc_classes,
    set_queue_max_rate,
    tc_handle_for_queue,
)
from traffic.generator import SLICES, payload_bps_for_l2

#: A programmed eMBB ceiling counts as confirmed if tc reports it within this fraction.
CEIL_TOLERANCE = 0.01


# --------------------------------------------------------------------------- pure functions


def rate_segments(step_offered_bps, interval_s: float) -> List[Dict]:
    """Split one slice's per-step offered load into constant stretches.

    Returns [{start_step, n_steps, bps}] covering every step in order. Refuses a trace that changes
    rate on more than a fifth of its steps: that is a jittered trace, which iperf3 cannot follow
    without restarting every second, and restarting every second would make connection setup part
    of the measurement.
    """
    rates = np.asarray(step_offered_bps, dtype=float)
    if rates.size == 0:
        return []
    segments: List[Dict] = []
    start = 0
    for i in range(1, rates.size + 1):
        if i == rates.size or rates[i] != rates[start]:
            segments.append({"start_step": start, "n_steps": i - start, "bps": float(rates[start])})
            start = i
    if len(segments) > max(1, rates.size // 5):
        raise ValueError(
            f"offered load changes {len(segments)} times in {rates.size} steps; the testbed needs "
            "piecewise-constant load, so set scenario.jitter_cv to zero"
        )
    return segments


def interval_rtt(
    samples: List[Tuple[float, int, float]], t_lo: float, t_hi: float
) -> Tuple[Optional[float], Optional[float], int]:
    """p50 and p95 of the ping replies timestamped in [t_lo, t_hi), and how many there were."""
    rtts = [s[2] for s in samples if t_lo <= s[0] < t_hi]
    if not rtts:
        return None, None, 0
    a = np.asarray(rtts, dtype=float)
    return float(np.percentile(a, 50)), float(np.percentile(a, 95)), len(rtts)


def counter_delta(prev: Optional[int], cur: Optional[int]) -> Optional[int]:
    """Increase of a cumulative counter, or None if either read is missing or it went backwards."""
    if prev is None or cur is None or cur < prev:
        return None
    return int(cur - prev)


# --------------------------------------------------------------------------- the backend


class OvsCliBackend(NetworkBackend):
    name = "ovs_cli"

    def __init__(
        self,
        cfg,
        trace,
        net,
        iface: str,
        ping_interval_s: float = 0.05,
        clock: Callable[[], float] = time.time,
        sleep: Callable[[float], None] = time.sleep,
        settle_s: float = 1.0,
    ):
        self.cfg = cfg
        self.trace = trace
        self.net = net
        self.iface = iface
        self.ping_interval_s = float(ping_interval_s)
        self.clock = clock
        self.sleep = sleep
        self.settle_s = float(settle_s)

        self.interval_s = float(cfg.run.control_interval_s)
        self.capacity_bps = float(cfg.link.capacity_bps)
        self.base_rtt_ms = float(cfg.link.base_rtt_ms)
        self.n_steps = min(int(round(float(cfg.run.duration_s) / self.interval_s)), trace.n_steps)
        offered = np.asarray(trace.step_offered_bps, dtype=float)[:, : self.n_steps]
        self.segments = {s: rate_segments(offered[SLICE_QUEUE[s]], self.interval_s) for s in SLICES}
        self._offered = offered
        self._handles = {s: tc_handle_for_queue(SLICE_QUEUE[s]) for s in SLICES}
        self._started = False

    # ------------------------------------------------------------------ contract

    def reset(self) -> None:
        """Program the initial allocation, start ping and the first iperf3 clients, mark t0."""
        self.close()
        self.workdir = Path(tempfile.mkdtemp(prefix="safeslice_live_"))
        self.ping_path = self.workdir / "ping.txt"
        self._procs: List = []
        self.step_log: List[Dict] = []
        self.segment_reports: List[Dict] = []
        self._segment_files: List[Dict] = []
        self._step = 0
        self._last_rtt = (self.base_rtt_ms, self.base_rtt_ms)

        initial = int(self.cfg.action.initial_index)
        programmed = apply_qos(self.iface, self.cfg, initial)
        self._programmed_embb_bps = int(programmed["embb"])
        self.sleep(self.settle_s)

        h1, h4 = self.net[SLICE_HOSTS["urllc"][0]], self.net[SLICE_HOSTS["urllc"][1]]
        total_s = self.n_steps * self.interval_s
        self._procs.append(h1.popen(
            ["sh", "-c",
             f"ping -D -n -i {self.ping_interval_s} -w {int(math.ceil(total_s + 5))} "
             f"{h4.IP()} > {self.ping_path} 2>&1"],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        ))
        self._t0 = self.clock()
        self._t_prev = self._t0
        self._prev_counters = self._read_counters()
        self._launch_segments_starting_at(0)
        self._started = True

    def apply_allocation(self, alloc: Allocation) -> None:
        """Only eMBB's ceiling depends on the action (net/backend.py:allocation_from_level), so only
        it is reprogrammed, and only when it changes: each change is three ovs-vsctl calls."""
        bps = int(alloc.embb_cap_bps)
        if bps != self._programmed_embb_bps:
            set_queue_max_rate(EMBB, bps)
            self._programmed_embb_bps = bps

    def read_telemetry(self) -> TelemetrySample:
        k = self._step
        boundary = self._t0 + (k + 1) * self.interval_s
        lateness = self.clock() - boundary
        if lateness < 0:
            self.sleep(-lateness)

        # Timestamp the middle of the tc call, so its own duration is not charged to one side.
        t_before = self.clock()
        classes = read_tc_classes(self.iface)
        t_read = (t_before + self.clock()) / 2.0
        if not classes:
            raise RuntimeError(f"tc class counters unreadable on {self.iface}; stopping the run")
        counters = self._counters_from(classes)
        dt = t_read - self._t_prev

        sent_bps, drops, counter_ok = [], [], True
        for s in SLICES:
            d_sent = counter_delta(self._prev_counters[s]["sent_bytes"], counters[s]["sent_bytes"])
            d_drop = counter_delta(self._prev_counters[s]["dropped_pkts"], counters[s]["dropped_pkts"])
            counter_ok = counter_ok and d_sent is not None and d_drop is not None
            sent_bps.append(0.0 if d_sent is None or dt <= 0 else d_sent * 8.0 / dt)
            drops.append(0 if d_drop is None else d_drop)
        backlog = tuple(float(counters[s]["backlog_bytes"] or 0) for s in SLICES)
        # On a saturated link, read-time jitter of a few milliseconds can put the measured total a
        # fraction of a percent over capacity. The CSV schema bounds link_util to [0, 1], so it is
        # capped there and the raw value is kept in step_log rather than lost.
        link_util_raw = float(sum(sent_bps) / self.capacity_bps)

        text = self.ping_path.read_text(encoding="utf-8", errors="replace") \
            if self.ping_path.exists() else ""
        samples, _ = parse_ping_output(text)
        p50, p95, n_rtt = interval_rtt(samples, self._t_prev, t_read)
        rtt_missing = n_rtt == 0
        if rtt_missing:
            p50, p95 = self._last_rtt
        self._last_rtt = (p50, p95)

        ceil_bps = parse_tc_rate(classes.get(self._handles["embb"], {}).get("ceil"))
        ceil_ok = ceil_bps is not None and \
            abs(ceil_bps - self._programmed_embb_bps) <= CEIL_TOLERANCE * self._programmed_embb_bps

        self.step_log.append({
            "step": k,
            "lateness_s": float(lateness),
            "dt_s": float(dt),
            "n_rtt": int(n_rtt),
            "rtt_missing": bool(rtt_missing),
            "counter_ok": bool(counter_ok),
            "programmed_embb_bps": int(self._programmed_embb_bps),
            "tc_embb_ceil_bps": ceil_bps,
            "ceil_ok": bool(ceil_ok),
            "link_util_raw": link_util_raw,
        })

        self._prev_counters = counters
        self._t_prev = t_read
        self._step += 1
        if self._step < self.n_steps:
            self._launch_segments_starting_at(self._step)

        offered = self._offered[:, k]
        return TelemetrySample(
            t_wall=t_read,
            t_step=self._step * self.interval_s,
            urllc_rtt_ms_p50=float(p50),
            urllc_rtt_ms_p95=float(p95),
            urllc_tx_bps=sent_bps[0],
            embb_goodput_bps=sent_bps[1],
            be_goodput_bps=sent_bps[2],
            q0_drops=drops[0],
            q1_drops=drops[1],
            q2_drops=drops[2],
            link_util=min(1.0, link_util_raw),
            backlog_bytes_per_queue=backlog,
            offered_bps_per_queue=tuple(float(x) for x in offered[:N_QUEUES]),
        )

    def idle_sample(self) -> TelemetrySample:
        """Primes the context builder at step 0, as SimBackend.idle_sample does. Not a measurement."""
        return TelemetrySample(
            t_wall=self.clock(), t_step=0.0,
            urllc_rtt_ms_p50=self.base_rtt_ms, urllc_rtt_ms_p95=self.base_rtt_ms,
            urllc_tx_bps=0.0, embb_goodput_bps=0.0, be_goodput_bps=0.0,
            q0_drops=0, q1_drops=0, q2_drops=0, link_util=0.0,
            backlog_bytes_per_queue=(0.0, 0.0, 0.0), offered_bps_per_queue=(0.0, 0.0, 0.0),
        )

    @property
    def done(self) -> bool:
        return getattr(self, "_step", 0) >= self.n_steps

    def close(self) -> None:
        """Stop every process this backend started and parse what the iperf3 clients reported."""
        if not getattr(self, "_started", False):
            return
        for p in self._procs:
            if p.poll() is None:
                p.terminate()
        for p in self._procs:
            try:
                p.wait(timeout=5)
            except subprocess.TimeoutExpired:
                p.kill()
        # The popen'd processes are `sh -c` wrappers, so terminating them can orphan the iperf3 or
        # ping underneath. Mininet hosts share one PID namespace, so one pkill reaches them all.
        self.net[SLICE_HOSTS["urllc"][0]].cmd("pkill -f 'iperf3 -c' ; pkill -f 'ping -D -n' ; true")
        for seg in self._segment_files:
            path = Path(seg["json"])
            text = path.read_text(encoding="utf-8", errors="replace") if path.exists() else ""
            parsed = parse_iperf3_udp(text) if text else {"parse_ok": False}
            self.segment_reports.append({**{k: v for k, v in seg.items() if k != "json"},
                                         **{k: parsed.get(k) for k in
                                            ("parse_ok", "sender_bps", "receiver_bps")}})
        self._started = False

    # ------------------------------------------------------------------ internals

    def _counters_from(self, classes: Dict) -> Dict[str, Dict]:
        out = {}
        for s in SLICES:
            c = classes.get(self._handles[s], {})
            out[s] = {key: c.get(key) for key in ("sent_bytes", "dropped_pkts", "backlog_bytes")}
        return out

    def _read_counters(self) -> Dict[str, Dict]:
        classes = read_tc_classes(self.iface)
        if not classes:
            raise RuntimeError(f"tc class counters unreadable on {self.iface}")
        return self._counters_from(classes)

    def _launch_segments_starting_at(self, step: int) -> None:
        for s in SLICES:
            for idx, seg in enumerate(self.segments[s]):
                if seg["start_step"] != step or seg["bps"] <= 0:
                    continue
                port = SLICE_PORTS[s] if idx % 2 == 0 else SLICE_ALT_PORTS[s]
                sender, receiver = self.net[SLICE_HOSTS[s][0]], self.net[SLICE_HOSTS[s][1]]
                duration = max(1, int(round(seg["n_steps"] * self.interval_s)))
                payload = int(round(payload_bps_for_l2(seg["bps"])))
                out = self.workdir / f"{s}_{idx}.json"
                self._procs.append(sender.popen(
                    ["sh", "-c",
                     f"iperf3 -c {receiver.IP()} -p {port} -u -b {payload} "
                     f"-l {UDP_PAYLOAD_BYTES} -t {duration} -J > {out} 2> {out}.err"],
                    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                ))
                self._segment_files.append({
                    "slice": s, "segment": idx, "start_step": step, "port": port,
                    "requested_l2_bps": seg["bps"], "duration_s": duration, "json": str(out),
                })
