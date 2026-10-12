"""Measure the idle round-trip-time noise floor of the testbed.

Stage 2 of docs/PLAN_TESTBED.md, and the gate on whether this environment can be used at all.

WHAT IT MEASURES
ICMP round-trip time from h1 to h4 with no other traffic, through s1 -> s2 with the OVS QoS
hierarchy configured and ICMP classified into q0. That is the exact path URLLC's latency is
measured on. Pinging 127.0.0.1 would measure the loopback device, which shares nothing with the
path that matters.

WHY IT MATTERS
The project claims to resolve queueing delay differences of a few milliseconds. If the host's own
idle jitter is comparable to that, a testbed latency number is a measurement of the hypervisor,
not of the slicing. This script says which, using the decision rule pre-registered in
docs/TESTBED_SETUP.md section 4 before any measurement existed.

It also produces the first real value for `link.base_rtt_ms`, which in config/default.yaml is a
modelling constant of 2.0 that has never been checked against anything.

TWO THINGS DONE DELIBERATELY
1. ARP is resolved with a separate short ping before measuring, and the first second of the
   measured series is discarded. The first packet to a new neighbour waits for address
   resolution, which can add milliseconds to one sample and would land squarely in p99.
2. Statistics are reported two ways: over all samples, and per one-second window. The backend
   computes latency percentiles per control interval from roughly 20 samples, so the per-window
   p95 is what the controller will actually see at idle. The whole-run p99 over ~1200 samples is a
   different and much better-resolved quantity, and the two must not be confused in the report.

Run as root inside the WSL checkout:

    python3 experiments/measure_noise_floor.py
"""

from __future__ import annotations

import argparse
import json
import math
import os
import platform
import re
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np  # noqa: E402

from config_loader import config_hash, load_config  # noqa: E402

# --------------------------------------------------------------------------- pure functions
# Everything in this section is free of Mininet so it can be unit tested on a machine that has no
# networking stack. The parser in particular fails silently if it is wrong: a regex that matches
# nothing yields zero samples, and zero samples is easy to mistake for a clean result.

# `ping -D` prefixes each reply with a unix timestamp in brackets.
_REPLY = re.compile(
    r"^\[(?P<ts>\d+\.\d+)\].*?icmp_seq=(?P<seq>\d+).*?time=(?P<rtt>[\d.]+)\s*ms", re.MULTILINE
)
_SUMMARY = re.compile(
    r"(?P<tx>\d+) packets transmitted, (?P<rx>\d+) (?:packets )?received"
    r"(?:, \+\d+ errors)?, (?P<loss>[\d.]+)% packet loss"
)


def parse_ping_output(out: str) -> Tuple[List[Tuple[float, int, float]], Optional[Dict]]:
    """Parse `ping -D` output into (timestamp, seq, rtt_ms) samples plus the summary line."""
    samples = [
        (float(m.group("ts")), int(m.group("seq")), float(m.group("rtt")))
        for m in _REPLY.finditer(out)
    ]
    summary = None
    s = _SUMMARY.search(out)
    if s:
        summary = {
            "transmitted": int(s.group("tx")),
            "received": int(s.group("rx")),
            "loss_pct": float(s.group("loss")),
        }
    return samples, summary


def discard_warmup(
    samples: List[Tuple[float, int, float]], warmup_s: float
) -> List[Tuple[float, int, float]]:
    """Drop samples within `warmup_s` of the first timestamp."""
    if not samples:
        return []
    t0 = samples[0][0]
    return [s for s in samples if s[0] - t0 >= warmup_s]


def summarize(rtts_ms: List[float]) -> Dict:
    """Whole-series statistics. p99 here is over every sample, not over windows."""
    a = np.asarray(rtts_ms, dtype=float)
    if a.size == 0:
        raise ValueError("no RTT samples to summarize")
    return {
        "n": int(a.size),
        "mean_ms": float(a.mean()),
        "std_ms": float(a.std(ddof=1)) if a.size > 1 else 0.0,
        "min_ms": float(a.min()),
        "p50_ms": float(np.percentile(a, 50)),
        "p95_ms": float(np.percentile(a, 95)),
        "p99_ms": float(np.percentile(a, 99)),
        "max_ms": float(a.max()),
    }


def windowed_stats(
    samples: List[Tuple[float, int, float]], window_s: float, min_samples: int = 5
) -> Dict:
    """Per-window p50 and p95, which is what the backend sees per control interval.

    Windows with fewer than `min_samples` replies are dropped rather than reported, because a p95
    of two samples is just their maximum and would be a misleading number to put in a table.
    """
    if not samples:
        raise ValueError("no samples")
    t0 = samples[0][0]
    buckets: Dict[int, List[float]] = {}
    for ts, _seq, rtt in samples:
        buckets.setdefault(int(math.floor((ts - t0) / window_s)), []).append(rtt)

    kept = {k: v for k, v in buckets.items() if len(v) >= min_samples}
    if not kept:
        raise ValueError(f"no window had at least {min_samples} samples")

    p50s = np.array([np.percentile(v, 50) for v in kept.values()])
    p95s = np.array([np.percentile(v, 95) for v in kept.values()])
    counts = np.array([len(v) for v in kept.values()])
    return {
        "window_s": float(window_s),
        "windows_total": len(buckets),
        "windows_kept": len(kept),
        "samples_per_window_median": float(np.median(counts)),
        "samples_per_window_min": int(counts.min()),
        "per_window_p50_median_ms": float(np.median(p50s)),
        "per_window_p95_median_ms": float(np.median(p95s)),
        "per_window_p95_p95_ms": float(np.percentile(p95s, 95)),
        "per_window_p95_max_ms": float(p95s.max()),
    }


def verdict(p99_ms: float, hard_ms: float) -> Tuple[str, str]:
    """The decision rule from docs/TESTBED_SETUP.md section 4, implemented exactly.

    Pre-registered before any measurement. Do not adjust these bands after seeing a result; if a
    band turns out to be wrong, change the document first, say why, and date it.
    """
    if p99_ms < 1.0:
        return "CLEAN", "Proceed."
    if p99_ms <= 3.0:
        return "USABLE", "Proceed. Report this floor next to every testbed latency number."
    if p99_ms < 0.5 * hard_ms:
        return (
            "MARGINAL",
            "Proceed only with the testbed SLO rescaled from this floor, stated as a limitation.",
        )
    return (
        "STOP",
        "Idle jitter is comparable to the effect being measured. Try the VirtualBox fallback "
        "and re-measure before building anything on this environment.",
    )


# --------------------------------------------------------------------------- measurement


def _loadavg() -> Optional[List[float]]:
    try:
        return [float(x) for x in os.getloadavg()]
    except (AttributeError, OSError):
        return None


def measure(duration_s: float, interval_s: float, warmup_s: float, scenario: str) -> Dict:
    """Build the testbed, ping h1 -> h4 idle, tear down. Requires root and Mininet."""
    from mininet.log import setLogLevel

    from net.topology.slice_topo import (
        apply_qos,
        bottleneck_iface,
        build_network,
        clear_qos,
        install_flows,
        read_queue_stats,
    )

    setLogLevel("warning")
    cfg = load_config(scenario=scenario)
    load_start = _loadavg()

    net = build_network(cfg)
    try:
        iface = bottleneck_iface(net)
        apply_qos(iface, cfg, int(cfg.action.initial_index))
        install_flows("s1")

        h1, h4 = net["h1"], net["h4"]
        target = h4.IP()

        # Resolve ARP and let OVS learn the MACs before anything is measured.
        h1.cmd(f"ping -c 3 -i 0.2 {target} > /dev/null 2>&1")
        time.sleep(0.5)

        q_before = read_queue_stats()
        count = int(round(duration_s / interval_s))
        t_start = time.time()
        # -D timestamps each reply. -w bounds the run even if replies stop arriving.
        raw = h1.cmd(
            f"ping -D -n -i {interval_s} -c {count} -w {int(duration_s + 10)} {target}"
        )
        t_end = time.time()
        q_after = read_queue_stats()
    finally:
        try:
            clear_qos(bottleneck_iface(net))
        except Exception:  # noqa: BLE001
            clear_qos()
        net.stop()

    samples, summary = parse_ping_output(raw)
    if not samples:
        raise RuntimeError(
            "ping produced no parseable replies. Last 800 characters of output:\n" + raw[-800:]
        )

    measured = discard_warmup(samples, warmup_s)

    # Did ICMP actually traverse q0? If classification failed, this measured the wrong queue.
    q0_pkts_delta = None
    if 0 in q_before and 0 in q_after:
        q0_pkts_delta = q_after[0]["pkts"] - q_before[0]["pkts"]

    whole = summarize([s[2] for s in measured])
    windows = windowed_stats(measured, window_s=float(cfg.run.control_interval_s))
    hard_ms = float(cfg.sla.hard_ms)
    label, action = verdict(whole["p99_ms"], hard_ms)

    return {
        "what": "idle ICMP RTT h1 -> h4 through s1 -> s2 with OVS QoS configured, ICMP in q0",
        "verdict": label,
        "verdict_action": action,
        "verdict_rule": "docs/TESTBED_SETUP.md section 4",
        "reference_sla_hard_ms": hard_ms,
        "p99_over_half_hard_ms": whole["p99_ms"] / (0.5 * hard_ms),
        "whole_series": whole,
        "per_control_interval": windows,
        "ping_summary": summary,
        "samples_raw": len(samples),
        "samples_after_warmup": len(measured),
        "warmup_discarded_s": warmup_s,
        "requested_interval_s": interval_s,
        "requested_duration_s": duration_s,
        "wall_clock_s": t_end - t_start,
        "q0_pkts_delta": q0_pkts_delta,
        "q0_classification_ok": (
            None if q0_pkts_delta is None else q0_pkts_delta >= 0.9 * len(samples)
        ),
        "suggested_testbed_base_rtt_ms": whole["p50_ms"],
        "environment": {
            "collected_at_iso": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "platform": platform.platform(),
            "kernel_release": platform.release(),
            "python_version": sys.version.split()[0],
            "cpu_count": os.cpu_count(),
            "loadavg_start": load_start,
            "loadavg_end": _loadavg(),
            "config_hash": config_hash(cfg),
            "scenario_config_used": scenario,
        },
    }


# --------------------------------------------------------------------------- cli


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Measure the idle RTT noise floor of the testbed.")
    ap.add_argument("--duration", type=float, default=60.0, help="seconds of measurement")
    ap.add_argument("--interval", type=float, default=0.05, help="seconds between pings")
    ap.add_argument("--warmup", type=float, default=1.0, help="seconds discarded at the start")
    ap.add_argument("--scenario", default="burst", help="config to load (for QoS rates only)")
    ap.add_argument("--out", default="results/summary/noise_floor.json")
    args = ap.parse_args(argv)

    if os.geteuid() != 0:
        print("measure_noise_floor.py must run as root (Mininet needs it).", file=sys.stderr)
        return 2

    print(f"[noise_floor] measuring {args.duration:.0f} s of idle RTT at {args.interval} s "
          f"intervals, h1 -> h4 through the bottleneck. Do not use the machine meanwhile.")
    report = measure(args.duration, args.interval, args.warmup, args.scenario)

    out = REPO_ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")

    w = report["whole_series"]
    pw = report["per_control_interval"]
    print("")
    print("=" * 72)
    print(f"  samples after warm-up        {report['samples_after_warmup']}"
          f"   (ping loss {report['ping_summary']['loss_pct'] if report['ping_summary'] else 'n/a'} %)")
    print(f"  whole series  p50 / p95 / p99 / max   "
          f"{w['p50_ms']:.3f} / {w['p95_ms']:.3f} / {w['p99_ms']:.3f} / {w['max_ms']:.3f} ms")
    print(f"  per 1 s window, median p95            {pw['per_window_p95_median_ms']:.3f} ms"
          f"   (worst window {pw['per_window_p95_max_ms']:.3f} ms,"
          f" ~{pw['samples_per_window_median']:.0f} samples each)")
    print(f"  ICMP counted in q0                    {report['q0_pkts_delta']} packets"
          f"   classification ok: {report['q0_classification_ok']}")
    print(f"  load average start / end              {report['environment']['loadavg_start']}"
          f" / {report['environment']['loadavg_end']}")
    print("-" * 72)
    print(f"  VERDICT  {report['verdict']}   (p99 {w['p99_ms']:.3f} ms against half of the "
          f"{report['reference_sla_hard_ms']:.1f} ms hard SLO)")
    print(f"  {report['verdict_action']}")
    print("=" * 72)
    print(f"[noise_floor] wrote {out}")

    if report["q0_classification_ok"] is False:
        print("[noise_floor] WARNING: ICMP did not go through q0. The flows in "
              "net/topology/slice_topo.py:install_flows are not matching, so this measured the "
              "wrong queue. Treat the numbers above as invalid.")
        return 1
    return 0 if report["verdict"] != "STOP" else 1


if __name__ == "__main__":
    raise SystemExit(main())
