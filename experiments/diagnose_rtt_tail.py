"""Stage 4a: is URLLC's latency tail under load a property of the network or of the probe?

WHY THIS EXISTS
The first quick sweep (results/summary/ovs_level_sweep_quick.json) showed URLLC latency with a heavy
tail that queueing cannot explain. At the lowest eMBB level URLLC is uncongested: its queue backlog
had a median of zero and its median RTT was 0.11 ms, the same as the idle noise floor. Yet its
pooled p99 was 4.66 ms and its worst reply 35.7 ms, against 0.17 ms and 7.85 ms idle. The path
and the empty queue were identical; the only difference was 17 Mbps of other traffic sharing the
host. The simulator's SLO and the guardrail's fast path both act on p95 against a 7 ms hard limit,
so a tail of this size would make them react to the host rather than to the queue.

Two explanations, which call for opposite responses:

  TOOL_SCHEDULING  the tail is the ping process itself being delayed by the scheduler between
                   taking its send timestamp and the packet actually leaving. That is a
                   measurement artefact. Fix the probe; the network is fine.
  PATH_JITTER      the tail is real delay in the kernel packet path (softirq, OVS datapath) when
                   the host is busy. That is genuine latency on this emulator, and it limits what
                   the testbed can resolve. The SLO or the statistic has to change instead.

Running ping at real-time priority (`chrt -f 99`) removes the first cause and leaves the second.
So the test is ping at normal priority against ping at real-time priority, under identical load, with
URLLC uncongested, alternated A B A B so slow drift cannot favour either. An idle run is included as a
reference and should reproduce stage 2.

The classification rule is in docs/PLAN_TESTBED.md section 2.9, recorded before this ran, and is
implemented in `classify_rtt_tail` below.

Run as root inside the WSL checkout (about 3 minutes):

    python3 experiments/diagnose_rtt_tail.py
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np  # noqa: E402

from config_loader import load_config  # noqa: E402
from experiments.sweep_levels_ovs import burst_phase_loads_l2  # noqa: E402
from traffic.generator import SLICES, counter_window_delta, per_window_rtt, ping_delivery  # noqa: E402

RT_PREFIX = "chrt -f 99 "

#: Run order. Loaded conditions alternate so drift over the run cannot line up with priority.
SCHEDULE = (
    ("idle_normal", False, ""),
    ("loaded_normal", True, ""),
    ("loaded_rt", True, RT_PREFIX),
    ("loaded_normal", True, ""),
    ("loaded_rt", True, RT_PREFIX),
)


def classify_rtt_tail(normal_p99_ms: List[float], rt_p99_ms: List[float]) -> Tuple[str, str]:
    """The rule from docs/PLAN_TESTBED.md section 2.9. Pure function, unit tested.

    Uses the mean pooled p99 across the repeated loaded runs of each priority.
    """
    if not normal_p99_ms or not rt_p99_ms:
        return "INCONCLUSIVE", "a loaded condition produced no usable samples"
    normal = float(np.mean(normal_p99_ms))
    rt = float(np.mean(rt_p99_ms))
    if normal < 1.0:
        return "NO_TAIL", (
            f"loaded p99 at normal priority was {normal:.2f} ms; the quick sweep's tail did not "
            "reproduce, so there is nothing to attribute"
        )
    if rt < 1.0:
        return "TOOL_SCHEDULING", (
            f"real-time priority cut p99 from {normal:.2f} to {rt:.2f} ms, back under 1 ms: the tail "
            "was the probe's own scheduling, not the network"
        )
    if rt >= 0.5 * normal:
        return "PATH_JITTER", (
            f"real-time priority left p99 at {rt:.2f} ms against {normal:.2f} ms, at least half: the "
            "tail is in the packet path under load and is real latency on this emulator"
        )
    return "PARTIAL", (
        f"real-time priority cut p99 from {normal:.2f} to {rt:.2f} ms: more than half the tail was the "
        "probe, but a real residual remains in the path"
    )


def summarize_condition(load: Dict, tc_series: Dict, t_launch: float, duration_s: float,
                        warmup_s: float) -> Dict:
    t_lo, t_hi = t_launch + warmup_s, t_launch + duration_s - 1.0
    rtt = per_window_rtt(load["ping_samples"], t_lo, t_hi)
    delivery = ping_delivery(load["ping_samples"], t_lo, t_hi)
    backlog = [v for t, v in tc_series["urllc"]["backlog_bytes"] if t_lo <= t <= t_hi]
    drops = counter_window_delta(tc_series["urllc"]["dropped_pkts"], t_lo, t_hi)
    return {
        "n_samples": rtt["n_samples"],
        "pooled_p50_ms": rtt["pooled_p50_ms"],
        "pooled_p95_ms": rtt["pooled_p95_ms"],
        "pooled_p99_ms": rtt["pooled_p99_ms"],
        "pooled_max_ms": rtt["pooled_max_ms"],
        "p95_of_window_p95_ms": rtt["p95_of_window_p95_ms"],
        "delivered_pct": delivery["delivered_pct"],
        "effective_interval_s": delivery["effective_interval_s"],
        "urllc_backlog_median_bytes": float(np.median(backlog)) if backlog else None,
        "urllc_backlog_max_bytes": max(backlog) if backlog else None,
        "urllc_drops": None if drops is None else int(drops["delta"]),
        "ping_error_tail": load.get("ping_raw_tail"),
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Attribute URLLC's latency tail: probe or path?")
    ap.add_argument("--duration", type=float, default=30.0)
    ap.add_argument("--warmup", type=float, default=5.0)
    ap.add_argument("--out", default="results/summary/rtt_tail_diagnosis.json")
    args = ap.parse_args(argv)

    if os.geteuid() != 0:
        print("diagnose_rtt_tail.py must run as root.", file=sys.stderr)
        return 2

    from mininet.log import setLogLevel

    from net.topology.slice_topo import (
        apply_qos, bottleneck_iface, build_network, clear_qos, install_flows,
    )
    from traffic.generator import collect_load, launch_load, sample_tc, start_servers, stop_servers

    setLogLevel("warning")
    cfg = load_config(scenario="burst")
    offered = burst_phase_loads_l2(cfg)
    idle = {s: 0.0 for s in SLICES}

    net = None
    results = []
    try:
        net = build_network(cfg)
        iface = bottleneck_iface(net)
        install_flows("s1")
        start_servers(net)

        rt_check = net["h1"].cmd("chrt -f 99 true && echo RT_OK || echo RT_FAILED").strip()
        if "RT_OK" not in rt_check:
            print(f"[rtt_tail] cannot run at real-time priority here: {rt_check}", file=sys.stderr)
            return 1

        # Lowest eMBB level: URLLC is uncongested, so any tail cannot be URLLC queueing.
        apply_qos(iface, cfg, int(cfg.action.floor_index))
        for name, loaded, prefix in SCHEDULE:
            print(f"[rtt_tail] {name:<14} {args.duration:g} s ...", flush=True)
            time.sleep(2.0)
            handles = launch_load(net, offered if loaded else idle, args.duration,
                                  ping_prefix=prefix)
            series = sample_tc(iface, until_ts=handles["t_launch"] + args.duration + 0.5)
            load = collect_load(handles)
            row = summarize_condition(load, series, handles["t_launch"], args.duration, args.warmup)
            row.update(condition=name, loaded=loaded, realtime=bool(prefix))
            results.append(row)
    finally:
        if net is not None:
            try:
                stop_servers(net)
            except Exception:  # noqa: BLE001
                pass
            try:
                clear_qos(bottleneck_iface(net))
            except Exception:  # noqa: BLE001
                clear_qos()
            net.stop()

    def _p99(name):
        return [r["pooled_p99_ms"] for r in results
                if r["condition"] == name and r["pooled_p99_ms"] is not None]

    label, reason = classify_rtt_tail(_p99("loaded_normal"), _p99("loaded_rt"))
    uncongested = all(
        (r["urllc_backlog_median_bytes"] or 0) <= 1442 and (r["urllc_drops"] or 0) == 0
        for r in results if r["loaded"]
    )
    report = {"conditions": results, "classification": label, "reason": reason,
              "urllc_was_uncongested_throughout": uncongested,
              "rule": "docs/PLAN_TESTBED.md section 2.9, recorded before this run",
              "offered_l2_bps": offered, "eMBB_level": float(cfg.action.embb_levels[0])}
    out = REPO_ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")

    def _f(v, nd=2):
        return "   n/a" if v is None else f"{v:6.{nd}f}"

    print("\n" + "=" * 104)
    print("  condition        n     p50     p95     p99      max   win-p95  delivered%  "
          "interval_s  URLLC backlog med  drops")
    for r in results:
        print(f"  {r['condition']:<14} {r['n_samples']:>4}  {_f(r['pooled_p50_ms'])}  "
              f"{_f(r['pooled_p95_ms'])}  {_f(r['pooled_p99_ms'])}  {_f(r['pooled_max_ms'])}  "
              f"{_f(r['p95_of_window_p95_ms'])}    {_f(r['delivered_pct'], 1)}     "
              f"{_f(r['effective_interval_s'], 4)}    {r['urllc_backlog_median_bytes']!s:>8}  "
              f"{r['urllc_drops']!s:>5}")
    print("-" * 104)
    print(f"  CLASSIFICATION  {label}")
    print(f"  {reason}")
    if not uncongested:
        print("  WARNING: URLLC was not uncongested in every loaded run, so part of the tail may be "
              "queueing. Read the classification with that in mind.")
    print("=" * 104)
    print(f"[rtt_tail] wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
