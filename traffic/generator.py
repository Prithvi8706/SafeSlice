"""Traffic generation on the Mininet testbed: one iperf3 UDP flow per slice, plus ping for URLLC.

    URLLC  h1 -> h4  port 5201  q0   plus ICMP echo h1 -> h4, also classified into q0
    eMBB   h2 -> h5  port 5202  q1
    BE     h3 -> h6  port 5203  q2

UNITS: EVERY RATE HERE IS IN L2 BITS UNLESS ITS NAME SAYS PAYLOAD
HTB enforces its caps on whole frames, and stage 1b measured that exactly: a 3.5 Mbps cap delivered
3.399 Mbps of 1400-byte UDP payload, against 3.398 predicted from the 1442-byte frame. The simulator
counts offered load in the same units as its link capacity, so the fair match is frames too.
iperf3's `-b` sets PAYLOAD rate, so a requested L2 rate is converted with `payload_bps_for_l2`
before it reaches iperf3. Asking iperf3 for the L2 number directly would offer about 3 percent more
than the simulator did, in every flow, as a built-in error in the comparison.

WHAT IS MEASURED, AND FROM WHERE
- Goodput and drops per slice come from the bottleneck's tc class counters, sampled once per second
  and differenced over a window bounded to the known flow interval. That is L2, and it is what the
  shaper actually transmitted and discarded. Stage 1c showed tc drops and iperf3 receiver loss
  agree to within one packet, so tc is trustworthy for loss.
- iperf3's own sender and receiver totals are recorded as a cross-check, not as the primary number,
  because they cover the whole run including connection setup and the drain tail.
- URLLC latency comes from ping replies inside the same window.

ONE BIAS THAT MUST NOT BE LOST
Ping only measures packets that come back. If q0 is full and tail-drops ICMP, the dropped probes
carry no RTT, so the latency percentiles describe only the survivors and understate how bad q0 was.
Ping loss is therefore always reported next to the latency numbers, and the report must never quote
one without the other.
"""

from __future__ import annotations

import math
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from experiments.measure_noise_floor import parse_ping_output
from net.topology.slice_topo import (
    SLICE_HOSTS,
    SLICE_PORTS,
    SLICE_QUEUE,
    UDP_PAYLOAD_BYTES,
    parse_iperf3_udp,
    read_tc_classes,
    tc_handle_for_queue,
)

SLICES = ("urllc", "embb", "be")

#: UDP 8 + IPv4 20 + Ethernet 14. The frame HTB counts when a 1400-byte payload is sent.
FRAME_OVERHEAD_BYTES = 8 + 20 + 14
FRAME_BYTES = UDP_PAYLOAD_BYTES + FRAME_OVERHEAD_BYTES


# --------------------------------------------------------------------------- pure functions


def payload_bps_for_l2(l2_bps: float) -> float:
    """The iperf3 `-b` value that puts `l2_bps` of frames on the wire."""
    return float(l2_bps) * UDP_PAYLOAD_BYTES / FRAME_BYTES


def l2_bps_from_payload(payload_bps: float) -> float:
    return float(payload_bps) * FRAME_BYTES / UDP_PAYLOAD_BYTES


def _in_window(series: List[Tuple[float, float]], t_lo: float, t_hi: float):
    return [(t, v) for t, v in series if t_lo <= t <= t_hi]


def counter_window_delta(
    series: List[Tuple[float, float]], t_lo: float, t_hi: float
) -> Optional[Dict]:
    """Increase of a cumulative counter across the samples that fall inside [t_lo, t_hi].

    Only samples inside the window are used, so time outside the flow (connection setup before,
    the final report exchange after) cannot dilute the rate. That dilution is what made the stage
    1b and 1c `tc_l2` column read about 10 percent low.

    Returns None for fewer than two samples in the window, a zero-length window, or a counter that
    went backwards, which means it was reset and the difference means nothing.
    """
    inside = _in_window(series, t_lo, t_hi)
    if len(inside) < 2:
        return None
    (ta, va), (tb, vb) = inside[0], inside[-1]
    if tb <= ta or vb < va:
        return None
    return {"delta": vb - va, "window_s": tb - ta, "n_samples": len(inside)}


def per_window_rtt(
    samples: List[Tuple[float, int, float]],
    t_lo: float,
    t_hi: float,
    window_s: float = 1.0,
    min_samples: int = 5,
) -> Dict:
    """URLLC latency statistics over the measurement window, in two forms.

    Per-window form, matching how the simulator's table in docs/DESIGN.md section 3 was built: take
    the p50 and the p95 inside each one-second window, then report the median of the p50 series and
    the 95th percentile of the p95 series. That is what a controller sampling once per second sees.

    Pooled form: percentiles over every reply in the window. Far better resolved, but a different
    quantity, and it must not be put in the same column as the per-window numbers.

    Windows with fewer than `min_samples` replies are excluded from the per-window form and counted,
    because a p95 of two samples is just their maximum.
    """
    inside = [s for s in samples if t_lo <= s[0] < t_hi]
    out: Dict = {
        "n_samples": len(inside),
        "n_windows_used": 0,
        "n_windows_thin": 0,
        "p50_of_window_p50_ms": None,
        "p95_of_window_p95_ms": None,
        "p99_of_window_p95_ms": None,
        "pooled_p50_ms": None,
        "pooled_p95_ms": None,
        "pooled_p99_ms": None,
        "pooled_max_ms": None,
    }
    if not inside:
        return out

    rtts = np.array([s[2] for s in inside], dtype=float)
    out.update(
        pooled_p50_ms=float(np.percentile(rtts, 50)),
        pooled_p95_ms=float(np.percentile(rtts, 95)),
        pooled_p99_ms=float(np.percentile(rtts, 99)),
        pooled_max_ms=float(rtts.max()),
    )

    buckets: Dict[int, List[float]] = {}
    for ts, _seq, rtt in inside:
        buckets.setdefault(int(math.floor((ts - t_lo) / window_s)), []).append(rtt)
    used = [v for v in buckets.values() if len(v) >= min_samples]
    out["n_windows_thin"] = len(buckets) - len(used)
    out["n_windows_used"] = len(used)
    if used:
        p50s = np.array([np.percentile(v, 50) for v in used])
        p95s = np.array([np.percentile(v, 95) for v in used])
        out.update(
            p50_of_window_p50_ms=float(np.percentile(p50s, 50)),
            p95_of_window_p95_ms=float(np.percentile(p95s, 95)),
            p99_of_window_p95_ms=float(np.percentile(p95s, 99)),
        )
    return out


def ping_delivery(samples: List[Tuple[float, int, float]], t_lo: float, t_hi: float) -> Dict:
    """Probe delivery inside the window, from ICMP sequence-number gaps. Pure, unit tested.

    CORRECTED 2026-09-13. The first version divided the reply count by window / requested
    interval. ping does not pace at exactly the requested interval on this testbed (stage 2: about
    0.055 s per probe when 0.05 s was requested), so that ratio read 90.6 to 98.5 percent in the
    first quick sweep while ping's own summary showed 0.0 percent loss at every level. It was
    measuring ping's timer, not the network. Sequence numbers are assigned per probe sent, so the
    span between the first and last sequence seen in the window is the number sent, whatever the
    pacing.

    Also returns the effective interval, so pacing drift is recorded rather than hidden.
    """
    inside = [s for s in samples if t_lo <= s[0] < t_hi]
    if len(inside) < 2:
        return {"sent_in_window": None, "received_in_window": len(inside),
                "delivered_pct": None, "effective_interval_s": None}
    seqs = sorted({s[1] for s in inside})
    span = seqs[-1] - seqs[0] + 1
    times = [s[0] for s in inside]
    return {
        "sent_in_window": span,
        "received_in_window": len(seqs),
        "delivered_pct": 100.0 * len(seqs) / span,
        "effective_interval_s": (max(times) - min(times)) / (span - 1) if span > 1 else None,
    }


# --------------------------------------------------------------------------- testbed side effects


def sample_tc(iface: str, until_ts: float, period_s: float = 1.0) -> Dict[str, Dict[str, List]]:
    """Poll the bottleneck's tc class counters until `until_ts`. Returns per-slice series of
    (timestamp, value) for sent bytes, dropped packets and backlog bytes."""
    series = {s: {"sent_bytes": [], "dropped_pkts": [], "backlog_bytes": []} for s in SLICES}
    handles = {s: tc_handle_for_queue(SLICE_QUEUE[s]) for s in SLICES}
    while True:
        now = time.time()
        classes = read_tc_classes(iface)
        for s in SLICES:
            c = classes.get(handles[s], {})
            for key in ("sent_bytes", "dropped_pkts", "backlog_bytes"):
                if c.get(key) is not None:
                    series[s][key].append((now, int(c[key])))
        if now >= until_ts:
            return series
        time.sleep(max(0.0, min(period_s, until_ts - time.time())))


def start_servers(net) -> None:
    """One persistent iperf3 server per slice on its receiver. They serve tests sequentially."""
    for s in SLICES:
        receiver = net[SLICE_HOSTS[s][1]]
        receiver.cmd(f"iperf3 -s -p {SLICE_PORTS[s]} -D --logfile /tmp/safeslice_srv_{s}.log")
    for s in SLICES:
        receiver = net[SLICE_HOSTS[s][1]]
        for _ in range(50):
            if f":{SLICE_PORTS[s]}" in receiver.cmd("ss -ltn"):
                break
            time.sleep(0.1)
        else:
            raise RuntimeError(f"iperf3 server for {s} did not start on port {SLICE_PORTS[s]}")


def stop_servers(net) -> None:
    for s in SLICES:
        net[SLICE_HOSTS[s][1]].cmd(f"pkill -f 'iperf3 -s -p {SLICE_PORTS[s]}' || true")


def launch_load(
    net,
    offered_l2_bps: Dict[str, float],
    duration_s: float,
    ping_interval_s: float = 0.05,
    ping_prefix: str = "",
) -> Dict:
    """Start one UDP flow per slice and a URLLC ping, concurrently. Returns handles for
    `collect_load`. Output goes to files, not pipes: a 120 s ping writes far more than a pipe
    buffer holds, and an unread full pipe would stall the process being measured.

    `ping_prefix` is prepended to the ping command, for example "chrt -f 99 " to run the probe at
    real-time priority. Slices offered zero are not started, so an all-zero load is a ping alone."""
    workdir = Path(tempfile.mkdtemp(prefix="safeslice_load_"))
    procs = {}
    for s in SLICES:
        l2 = float(offered_l2_bps.get(s, 0.0))
        if l2 <= 0:
            continue
        sender, receiver = net[SLICE_HOSTS[s][0]], net[SLICE_HOSTS[s][1]]
        payload = int(round(payload_bps_for_l2(l2)))
        procs[s] = sender.popen(
            ["sh", "-c",
             f"iperf3 -c {receiver.IP()} -p {SLICE_PORTS[s]} -u -b {payload} "
             f"-l {UDP_PAYLOAD_BYTES} -t {duration_s:g} -J "
             f"> {workdir}/{s}.json 2> {workdir}/{s}.err"],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
    h1, h4 = net[SLICE_HOSTS["urllc"][0]], net[SLICE_HOSTS["urllc"][1]]
    procs["ping"] = h1.popen(
        ["sh", "-c",
         f"{ping_prefix}ping -D -n -i {ping_interval_s} -w {int(math.ceil(duration_s))} "
         f"{h4.IP()} > {workdir}/ping.txt 2>&1"],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    return {"t_launch": time.time(), "workdir": str(workdir), "procs": procs,
            "duration_s": float(duration_s), "offered_l2_bps": dict(offered_l2_bps)}


def collect_load(handles: Dict, timeout_extra_s: float = 30.0) -> Dict:
    """Wait for every process and parse its output."""
    deadline = handles["t_launch"] + handles["duration_s"] + timeout_extra_s
    for name, p in handles["procs"].items():
        remaining = max(1.0, deadline - time.time())
        try:
            p.wait(timeout=remaining)
        except subprocess.TimeoutExpired:
            p.kill()
    wd = Path(handles["workdir"])
    iperf = {}
    for s in SLICES:
        path = wd / f"{s}.json"
        if path.exists():
            text = path.read_text(encoding="utf-8", errors="replace")
            iperf[s] = parse_iperf3_udp(text)
            if not iperf[s]["parse_ok"]:
                iperf[s]["raw_tail"] = text[-400:]
    ping_text = (wd / "ping.txt").read_text(encoding="utf-8", errors="replace") \
        if (wd / "ping.txt").exists() else ""
    samples, summary = parse_ping_output(ping_text)
    return {"iperf": iperf, "ping_samples": samples, "ping_summary": summary,
            "ping_raw_tail": ping_text[-300:] if not samples else None}
