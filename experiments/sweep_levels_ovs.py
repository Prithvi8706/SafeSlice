"""Stages 3 and 4 of docs/PLAN_TESTBED.md: calibrate the traffic generator, then sweep the five eMBB
levels on the real Open vSwitch bottleneck under constant load.

This is the faculty deliverable ("show on Mininet that the slicing works") and the testbed half of
the simulator comparison, in one experiment.

LOAD
The burst phase of config/scenarios/burst.yaml, which is the load behind the simulator's
fixed-level table in docs/DESIGN.md section 3: URLLC 3.0, eMBB 10.0, BE 4.0 Mbps, in L2 bits (see
traffic/generator.py for why L2). Held constant, no jitter. docs/PLAN_TESTBED.md section 2.4 explains
why constant load is the better experiment here and not a compromise.

BUFFERS
Every run uses the finite 62,500 B per-queue buffer that stage 1c established as necessary
(docs/PLAN_TESTBED.md section 2.8). Without it this testbed blocks senders instead of dropping, and
none of the numbers below would describe a real switch.

PROTOCOL
- Calibration first. All three flows below their caps and below link capacity. Each must deliver
  within 5 percent of what was offered with zero drops, or the sweep refuses to start: a generator
  that cannot hit its targets unshaped makes every shaped number meaningless.
- Each (level, repeat) is an independent run on a freshly programmed QoS hierarchy.
- Run order is shuffled per repeat from a logged seed and written to disk before the first run.
- Per-run rows are appended to a JSONL file as they finish, so an interrupted sweep keeps its data.
- The first `warmup` seconds and the final second of every run are excluded from every metric.

Run as root inside the WSL checkout:

    python3 experiments/sweep_levels_ovs.py --quick      # 1 repeat, 30 s per level, ~4 min
    python3 experiments/sweep_levels_ovs.py              # 3 repeats, 120 s per level, ~35 min
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import platform
import random
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np  # noqa: E402

from analysis.aggregate import mean_ci  # noqa: E402
from config_loader import config_hash, load_config  # noqa: E402
from traffic.generator import (  # noqa: E402
    SLICES,
    counter_window_delta,
    l2_bps_from_payload,
    per_window_rtt,
)

#: Calibration load: every flow under its cap at the most permissive eMBB level, total under
#: capacity, so nothing should be shaped or dropped and delivered should equal offered.
CALIBRATION_LOAD_L2 = {"urllc": 2.0e6, "embb": 3.0e6, "be": 3.0e6}
CALIBRATION_TOLERANCE = 0.05

#: Metrics carried into the per-level aggregate, in report order.
AGG_KEYS = (
    "embb_goodput_l2_mbps",
    "be_goodput_l2_mbps",
    "urllc_goodput_l2_mbps",
    "link_util",
    "urllc_rtt_p50_ms",
    "urllc_rtt_p95_ms",
    "urllc_rtt_pooled_p99_ms",
    "urllc_ping_delivered_pct",
    "urllc_drops",
    "embb_drops",
    "be_drops",
    "urllc_backlog_median_bytes",
    "embb_backlog_median_bytes",
)


# --------------------------------------------------------------------------- pure functions


def burst_phase_loads_l2(cfg) -> Dict[str, float]:
    """Offered load from the scenario phase with the highest eMBB demand, in L2 bits per second."""
    phases = [dict(p) for p in cfg.scenario.phases]
    if not phases:
        raise ValueError("scenario has no phases")
    burst = max(phases, key=lambda p: float(p["embb_mbps"]))
    return {s: float(burst[f"{s}_mbps"]) * 1e6 for s in SLICES}


def run_order(n_levels: int, repeats: int, seed: int) -> List[Dict]:
    """Every level once per repeat, shuffled within each repeat from one seeded generator.

    Shuffled so that slow drift over a 35-minute sweep (host temperature, background activity) does
    not line up with the level being measured. Deterministic in `seed` so the order is reproducible.
    """
    rng = random.Random(seed)
    order = []
    for r in range(repeats):
        levels = list(range(n_levels))
        rng.shuffle(levels)
        for level in levels:
            order.append({"position": len(order), "repeat": r, "level_index": level})
    return order


def summarize_run(
    tc_series: Dict[str, Dict[str, List]],
    load: Dict,
    t_launch: float,
    duration_s: float,
    warmup_s: float,
    capacity_bps: float,
    offered_l2_bps: Dict[str, float],
    ping_interval_s: float,
) -> Dict:
    """Reduce one run to a flat row of metrics, all taken inside the measurement window."""
    t_lo = t_launch + warmup_s
    t_hi = t_launch + duration_s - 1.0
    row: Dict = {"window_s_nominal": t_hi - t_lo}
    total_goodput = 0.0
    goodput_ok = True

    for s in SLICES:
        sent = counter_window_delta(tc_series[s]["sent_bytes"], t_lo, t_hi)
        drop = counter_window_delta(tc_series[s]["dropped_pkts"], t_lo, t_hi)
        backlog = [v for t, v in tc_series[s]["backlog_bytes"] if t_lo <= t <= t_hi]

        goodput = None if sent is None else sent["delta"] * 8.0 / sent["window_s"]
        if goodput is None:
            goodput_ok = False
        else:
            total_goodput += goodput
        row[f"{s}_goodput_l2_mbps"] = None if goodput is None else goodput / 1e6
        row[f"{s}_offered_l2_mbps"] = offered_l2_bps.get(s, 0.0) / 1e6
        row[f"{s}_drops"] = None if drop is None else int(drop["delta"])
        row[f"{s}_backlog_median_bytes"] = float(np.median(backlog)) if backlog else None
        row[f"{s}_backlog_max_bytes"] = max(backlog) if backlog else None

        ip = load.get("iperf", {}).get(s)
        row[f"{s}_iperf_parse_ok"] = bool(ip and ip["parse_ok"])
        row[f"{s}_iperf_sender_l2_mbps"] = (
            None if not ip or ip["sender_bps"] is None
            else l2_bps_from_payload(ip["sender_bps"]) / 1e6
        )
        row[f"{s}_iperf_receiver_l2_mbps"] = (
            None if not ip or ip["receiver_bps"] is None
            else l2_bps_from_payload(ip["receiver_bps"]) / 1e6
        )

    row["link_util"] = total_goodput / capacity_bps if goodput_ok else None

    rtt = per_window_rtt(load.get("ping_samples", []), t_lo, t_hi)
    row["urllc_rtt_p50_ms"] = rtt["p50_of_window_p50_ms"]
    row["urllc_rtt_p95_ms"] = rtt["p95_of_window_p95_ms"]
    row["urllc_rtt_p99_of_window_p95_ms"] = rtt["p99_of_window_p95_ms"]
    row["urllc_rtt_pooled_p50_ms"] = rtt["pooled_p50_ms"]
    row["urllc_rtt_pooled_p95_ms"] = rtt["pooled_p95_ms"]
    row["urllc_rtt_pooled_p99_ms"] = rtt["pooled_p99_ms"]
    row["urllc_rtt_pooled_max_ms"] = rtt["pooled_max_ms"]
    row["urllc_rtt_windows_used"] = rtt["n_windows_used"]
    row["urllc_rtt_windows_thin"] = rtt["n_windows_thin"]

    # Delivered probes as a share of those sent inside the window. Latency percentiles describe
    # only delivered probes, so this number must always travel with them.
    expected = (t_hi - t_lo) / ping_interval_s
    row["urllc_ping_delivered_pct"] = (
        None if expected <= 0 else min(100.0, 100.0 * rtt["n_samples"] / expected)
    )
    summary = load.get("ping_summary") or {}
    row["urllc_ping_loss_pct_whole_run"] = summary.get("loss_pct")

    row["valid"] = bool(goodput_ok and rtt["n_windows_used"] > 0)
    return row


def calibration_verdict(
    row: Dict, offered_l2_bps: Dict[str, float], tolerance: float = CALIBRATION_TOLERANCE
) -> Tuple[bool, Dict]:
    """Pass only if every flow was delivered within tolerance of offered, with zero drops, and the
    sender itself reached its target. An unshaped run that misses on any of these means the
    generator, not the network, is setting the numbers."""
    details: Dict = {}
    ok = bool(row.get("valid"))
    for s in SLICES:
        offered = offered_l2_bps[s] / 1e6
        good = row.get(f"{s}_goodput_l2_mbps")
        sender = row.get(f"{s}_iperf_sender_l2_mbps")
        drops = row.get(f"{s}_drops")
        d = {
            "offered_l2_mbps": offered,
            "delivered_l2_mbps": good,
            "delivered_over_offered": None if good is None else good / offered,
            "sender_over_offered": None if sender is None else sender / offered,
            "drops": drops,
        }
        d["ok"] = (
            good is not None and abs(good / offered - 1.0) <= tolerance
            and sender is not None and abs(sender / offered - 1.0) <= tolerance
            and drops == 0
        )
        ok = ok and d["ok"]
        details[s] = d
    return ok, details


def aggregate_levels(rows: List[Dict], levels: List[float]) -> List[Dict]:
    """Mean and 95 percent t interval across repeats, per level, over valid runs only."""
    out = []
    for li, share in enumerate(levels):
        runs = [r for r in rows if r["level_index"] == li and r.get("valid")]
        entry: Dict = {"level_index": li, "embb_share": share, "n_valid_runs": len(runs),
                       "n_invalid_runs": sum(1 for r in rows if r["level_index"] == li
                                             and not r.get("valid"))}
        for key in AGG_KEYS:
            vals = [r[key] for r in runs if r.get(key) is not None]
            entry[key] = mean_ci(vals) if vals else None
        out.append(entry)
    return out


# --------------------------------------------------------------------------- execution


def _env(cfg) -> Dict:
    def _cmd(args):
        import subprocess
        try:
            return subprocess.run(args, capture_output=True, text=True, timeout=10).stdout.strip()
        except Exception:  # noqa: BLE001
            return None

    try:
        load = list(os.getloadavg())
    except (AttributeError, OSError):
        load = None
    return {
        "collected_at_iso": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "kernel_release": platform.release(),
        "python_version": sys.version.split()[0],
        "cpu_count": os.cpu_count(),
        "loadavg": load,
        "ovs_version": _cmd(["ovs-vsctl", "--version"]),
        "iperf3_version": _cmd(["iperf3", "--version"]),
        "git_commit": _cmd(["git", "-C", str(REPO_ROOT), "rev-parse", "HEAD"]),
        "config_hash": config_hash(cfg),
        "leaf_limit_bytes": int(cfg.sim.queue_limit_bytes),
    }


def _one_run(net, iface, cfg, level_index, offered, duration_s, warmup_s, ping_interval_s):
    from net.topology.slice_topo import apply_qos
    from traffic.generator import collect_load, launch_load, sample_tc

    programmed = apply_qos(iface, cfg, level_index)
    time.sleep(1.0)
    handles = launch_load(net, offered, duration_s, ping_interval_s)
    series = sample_tc(iface, until_ts=handles["t_launch"] + duration_s + 0.5)
    load = collect_load(handles)
    row = summarize_run(series, load, handles["t_launch"], duration_s, warmup_s,
                        float(cfg.link.capacity_bps), offered, ping_interval_s)
    row["programmed_max_rates_bps"] = programmed
    return row


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Calibrate, then sweep eMBB levels on real OVS.")
    ap.add_argument("--scenario", default="burst")
    ap.add_argument("--duration", type=float, default=120.0)
    ap.add_argument("--warmup", type=float, default=10.0)
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--ping-interval", type=float, default=0.05)
    ap.add_argument("--quick", action="store_true", help="1 repeat, 30 s, 5 s warm-up")
    ap.add_argument("--calibrate-only", action="store_true")
    ap.add_argument("--out-prefix", default=None)
    args = ap.parse_args(argv)

    if args.quick:
        args.duration, args.warmup, args.repeats = 30.0, 5.0, 1
    prefix = args.out_prefix or (
        "results/summary/ovs_level_sweep_quick" if args.quick else "results/summary/ovs_level_sweep"
    )
    if os.geteuid() != 0:
        print("sweep_levels_ovs.py must run as root (Mininet needs it).", file=sys.stderr)
        return 2
    if args.duration - 1.0 - args.warmup < 5.0:
        print("duration must leave at least 5 s after warm-up and the final second.", file=sys.stderr)
        return 2

    from mininet.log import setLogLevel

    from net.topology.slice_topo import (
        bottleneck_iface, build_network, clear_qos, install_flows,
    )
    from traffic.generator import start_servers, stop_servers

    setLogLevel("warning")
    cfg = load_config(scenario=args.scenario)
    levels = [float(x) for x in cfg.action.embb_levels]
    offered = burst_phase_loads_l2(cfg)
    order = run_order(len(levels), args.repeats, args.seed)

    base = REPO_ROOT / prefix
    base.parent.mkdir(parents=True, exist_ok=True)
    meta = {"args": vars(args), "levels": levels, "offered_l2_bps": offered, "order": order,
            "env_start": _env(cfg)}
    # Written before anything runs, so an interrupted sweep still records what it meant to do.
    Path(f"{base}_order.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    jsonl = Path(f"{base}_runs.jsonl")
    jsonl.write_text("", encoding="utf-8")

    est_min = (len(order) * (args.duration + 5) + 30) / 60
    print(f"[sweep] {len(order)} runs x {args.duration:g} s, about {est_min:.0f} min. "
          f"Offered L2 Mbps: " + ", ".join(f"{s} {offered[s] / 1e6:g}" for s in SLICES))
    print("[sweep] keep the machine plugged in and idle until it finishes.\n")

    net = None
    try:
        net = build_network(cfg)
        iface = bottleneck_iface(net)
        install_flows("s1")
        start_servers(net)

        # ---- stage 3: calibration
        print("[sweep] calibration: all flows under their caps, nothing should be shaped ...")
        cal_row = _one_run(net, iface, cfg, len(levels) - 1, CALIBRATION_LOAD_L2,
                           20.0, 5.0, args.ping_interval)
        cal_ok, cal_details = calibration_verdict(cal_row, CALIBRATION_LOAD_L2)
        Path(f"{base}_calibration.json").write_text(
            json.dumps({"passed": cal_ok, "details": cal_details, "row": cal_row}, indent=2),
            encoding="utf-8")
        for s in SLICES:
            d = cal_details[s]
            print(f"   {s:5s} offered {d['offered_l2_mbps']:.3f}  delivered "
                  f"{d['delivered_l2_mbps'] if d['delivered_l2_mbps'] is None else round(d['delivered_l2_mbps'], 3)}"
                  f"  sender/offered {d['sender_over_offered'] if d['sender_over_offered'] is None else round(d['sender_over_offered'], 3)}"
                  f"  drops {d['drops']}  {'ok' if d['ok'] else 'FAIL'}")
        print(f"[sweep] calibration {'PASSED' if cal_ok else 'FAILED'}\n")
        if not cal_ok:
            print("[sweep] refusing to sweep: the generator cannot hit its own targets unshaped.")
            return 1
        if args.calibrate_only:
            return 0

        # ---- stage 4: the sweep
        rows = []
        for item in order:
            li = item["level_index"]
            row = _one_run(net, iface, cfg, li, offered, args.duration, args.warmup,
                           args.ping_interval)
            row.update(item)
            row["embb_share"] = levels[li]
            rows.append(row)
            with open(jsonl, "a", encoding="utf-8") as fh:
                fh.write(json.dumps(row) + "\n")

            def _f(v, nd=3):
                return "  n/a" if v is None else f"{v:.{nd}f}"
            print(f"[sweep] {item['position'] + 1:>2}/{len(order)}  level {levels[li]:.2f}  "
                  f"eMBB {_f(row['embb_goodput_l2_mbps'])}  BE {_f(row['be_goodput_l2_mbps'])}  "
                  f"URLLC p50/p95 {_f(row['urllc_rtt_p50_ms'], 2)}/{_f(row['urllc_rtt_p95_ms'], 2)} ms"
                  f"  ping delivered {_f(row['urllc_ping_delivered_pct'], 1)}%"
                  f"  drops u/e/b {row['urllc_drops']}/{row['embb_drops']}/{row['be_drops']}"
                  f"{'' if row['valid'] else '  INVALID'}", flush=True)
            time.sleep(3.0)

        agg = aggregate_levels(rows, levels)
        result = {**meta, "calibration_passed": cal_ok, "rows": rows, "aggregate": agg,
                  "env_end": _env(cfg),
                  "noise_floor_reference": "results/summary/noise_floor.json"}
        Path(f"{base}.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
        with open(f"{base}.csv", "w", newline="", encoding="utf-8") as fh:
            keys = sorted({k for r in rows for k in r if not isinstance(r[k], dict)})
            w = csv.DictWriter(fh, fieldnames=keys, extrasaction="ignore")
            w.writeheader()
            w.writerows(rows)

        def _ci(e, nd=3):
            if e is None:
                return "     n/a      "
            h = e["ci95_half"]
            return f"{e['mean']:.{nd}f}" + ("" if h != h else f" ± {h:.{nd}f}")

        print("\n" + "=" * 118)
        print(f"  Real OVS, constant load, {args.repeats} repeat(s) x {args.duration:g} s per level."
              f"  Mean ± 95% CI.  L2 Mbps.")
        print("  level  eMBB goodput     BE goodput       URLLC p50 ms    URLLC p95 ms    "
              "ping delivered %   eMBB drops")
        for e in agg:
            invalid = e["n_invalid_runs"]
            note = "" if invalid == 0 else f"   ({invalid} invalid run(s) excluded)"
            print(f"  {e['embb_share']:.2f}   {_ci(e['embb_goodput_l2_mbps']):<16} "
                  f"{_ci(e['be_goodput_l2_mbps']):<16} {_ci(e['urllc_rtt_p50_ms'], 2):<15} "
                  f"{_ci(e['urllc_rtt_p95_ms'], 2):<15} {_ci(e['urllc_ping_delivered_pct'], 1):<16} "
                  f"{_ci(e['embb_drops'], 0)}{note}")
        print("=" * 118)
        print(f"[sweep] wrote {base}.json, {base}.csv, {base}_runs.jsonl")
        return 0
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


if __name__ == "__main__":
    raise SystemExit(main())
