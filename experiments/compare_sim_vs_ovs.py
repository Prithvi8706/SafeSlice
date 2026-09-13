"""Stage 5 of docs/PLAN_TESTBED.md: does the simulator reproduce the real OVS level sweep, and under
which leftover-capacity sharing mode?

Runs entirely in the simulator, on any machine. It reads the testbed result
(results/summary/ovs_level_sweep.json) and the measured idle floor (results/summary/noise_floor.json),
runs the simulator under conditions matched to the testbed, and applies the rule recorded in
docs/PLAN_TESTBED.md section 2.11 before this script existed.

MATCHED CONDITIONS
- One constant phase at the testbed's offered load, with every jitter coefficient set to zero, so
  the simulator sees the same unchanging load iperf3 produced.
- 120 one-second steps, keeping steps 10 to 118, which cover seconds 10 to 119: the same window the
  testbed summarizer keeps.
- `link.base_rtt_ms` set to the measured idle median from stage 2 rather than the modelling constant.
  Without that, every simulated latency would sit about 1.9 ms above the testbed for reasons that
  have nothing to do with queueing.
- The same 62,500 B per-queue buffer, caps and guarantees, since both sides read one config.
With zero jitter each (mode, level) cell is deterministic, so it is run once.

    python experiments/compare_sim_vs_ovs.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np  # noqa: E402

from config_loader import load_config  # noqa: E402
from net.backend import allocation_from_level  # noqa: E402
from net.sim_backend import SimBackend  # noqa: E402
from traffic.traces import build_trace  # noqa: E402

MODES = ("demand_proportional", "equal", "min_rate_proportional")
#: Metrics the rule is applied to. URLLC p95 is excluded per docs/PLAN_TESTBED.md section 2.10.
RULE_METRICS = ("embb_goodput_l2_mbps", "be_goodput_l2_mbps", "urllc_rtt_p50_ms")
TRACK_THRESHOLD = 0.25


# --------------------------------------------------------------------------- simulator side


def constant_load_cfg(mode: str, loads_mbps: Dict[str, float], duration_s: float,
                      warmup_s: float, base_rtt_ms: float):
    """Config for one matched simulator run: constant load, no jitter, measured base RTT."""
    return load_config(
        scenario="burst",
        overrides={
            "scenario.phases": [{
                "duration_s": float(duration_s) + 100.0,
                "urllc_mbps": float(loads_mbps["urllc"]),
                "embb_mbps": float(loads_mbps["embb"]),
                "be_mbps": float(loads_mbps["be"]),
            }],
            "scenario.jitter_cv": {"urllc": 0.0, "embb": 0.0, "be": 0.0},
            "run.duration_s": float(duration_s),
            "run.warmup_s": float(warmup_s),
            "link.base_rtt_ms": float(base_rtt_ms),
            "sim.excess_sharing": mode,
        },
    )


def run_sim_level(cfg, level_index: int, duration_s: float, warmup_s: float) -> Dict[str, float]:
    """Drive the simulator at one fixed level and reduce it with the testbed's definitions."""
    trace = build_trace(cfg, seed=0)
    backend = SimBackend(cfg, trace)
    alloc = allocation_from_level(cfg, level_index)
    first, last = int(warmup_s), int(duration_s) - 1   # keep steps first..last-1
    kept = []
    for step in range(int(duration_s)):
        if backend.done:
            break
        backend.apply_allocation(alloc)
        tel = backend.read_telemetry()
        if first <= step < last:
            kept.append(tel)
    if not kept:
        raise RuntimeError("no simulator steps inside the comparison window")
    return {
        "n_steps": len(kept),
        "embb_goodput_l2_mbps": float(np.mean([t.embb_goodput_bps for t in kept])) / 1e6,
        "be_goodput_l2_mbps": float(np.mean([t.be_goodput_bps for t in kept])) / 1e6,
        "urllc_goodput_l2_mbps": float(np.mean([t.urllc_tx_bps for t in kept])) / 1e6,
        "urllc_rtt_p50_ms": float(np.percentile([t.urllc_rtt_ms_p50 for t in kept], 50)),
        "urllc_rtt_p95_ms": float(np.percentile([t.urllc_rtt_ms_p95 for t in kept], 95)),
    }


# --------------------------------------------------------------------------- comparison (pure)


def load_testbed_rows(sweep: Dict) -> List[Dict]:
    """Per-level testbed means and interval half-widths from the stage 4 aggregate."""
    out = []
    for e in sorted(sweep["aggregate"], key=lambda e: e["level_index"]):
        row = {"level_index": e["level_index"], "embb_share": e["embb_share"]}
        for m in RULE_METRICS + ("urllc_rtt_p95_ms", "urllc_goodput_l2_mbps"):
            agg = e.get(m)
            row[m] = None if agg is None else agg["mean"]
            row[f"{m}_ci95_half"] = None if agg is None else agg["ci95_half"]
        out.append(row)
    return out


def errors_for_mode(sim_rows: List[Dict], tb_rows: List[Dict]) -> Dict[str, Dict]:
    """MAE, testbed range and NMAE per rule metric. Levels must be in the same order."""
    if [r["level_index"] for r in sim_rows] != [r["level_index"] for r in tb_rows]:
        raise ValueError("simulator and testbed levels are not aligned")
    out = {}
    for m in RULE_METRICS:
        tb = np.array([r[m] for r in tb_rows], dtype=float)
        sim = np.array([r[m] for r in sim_rows], dtype=float)
        mae = float(np.mean(np.abs(sim - tb)))
        rng = float(tb.max() - tb.min())
        halves = [r.get(f"{m}_ci95_half") for r in tb_rows]
        halves = [h for h in halves if h is not None and h == h]
        out[m] = {
            "mae": mae,
            "testbed_range": rng,
            "nmae": float("inf") if rng <= 0 else mae / rng,
            "tracks": rng > 0 and mae / rng <= TRACK_THRESHOLD,
            "testbed_mean_ci95_half": float(np.mean(halves)) if halves else None,
            "max_abs_error": float(np.max(np.abs(sim - tb))),
        }
    return out


def classify_match(errors_by_mode: Dict[str, Dict[str, Dict]]) -> Tuple[str, str, Dict]:
    """The rule from docs/PLAN_TESTBED.md section 2.11. Pure function, unit tested."""
    mean_nmae = {k: float(np.mean([v[m]["nmae"] for m in RULE_METRICS]))
                 for k, v in errors_by_mode.items()}
    best = min(mean_nmae, key=mean_nmae.get)
    tracked = [m for m in RULE_METRICS if errors_by_mode[best][m]["tracks"]]
    per_metric_winner = {
        m: min(errors_by_mode, key=lambda k: errors_by_mode[k][m]["nmae"]) for m in RULE_METRICS
    }
    detail = {"best_mode": best, "mean_nmae": mean_nmae, "metrics_tracked_by_best": tracked,
              "per_metric_winner": per_metric_winner}
    if len(tracked) == len(RULE_METRICS):
        return f"TRACKS_{best}", best, detail
    if tracked:
        return f"PARTIAL_{best}", best, detail
    return "NEITHER", best, detail


# --------------------------------------------------------------------------- cli


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Compare the simulator against the real OVS sweep.")
    ap.add_argument("--sweep", default="results/summary/ovs_level_sweep.json")
    ap.add_argument("--noise-floor", default="results/summary/noise_floor.json")
    ap.add_argument("--out", default="results/summary/sim_vs_ovs.json")
    args = ap.parse_args(argv)

    sweep = json.loads((REPO_ROOT / args.sweep).read_text(encoding="utf-8"))
    floor = json.loads((REPO_ROOT / args.noise_floor).read_text(encoding="utf-8"))
    base_rtt = float(floor["whole_series"]["p50_ms"])
    duration = float(sweep["args"]["duration"])
    warmup = float(sweep["args"]["warmup"])
    loads = {s: v / 1e6 for s, v in sweep["offered_l2_bps"].items()}
    levels = [float(x) for x in sweep["levels"]]

    tb = load_testbed_rows(sweep)
    sim_by_mode: Dict[str, List[Dict]] = {}
    errors_by_mode: Dict[str, Dict] = {}
    for mode in MODES:
        cfg = constant_load_cfg(mode, loads, duration, warmup, base_rtt)
        rows = []
        for li, share in enumerate(levels):
            r = run_sim_level(cfg, li, duration, warmup)
            r.update(level_index=li, embb_share=share)
            rows.append(r)
        sim_by_mode[mode] = rows
        errors_by_mode[mode] = errors_for_mode(rows, tb)

    label, best, detail = classify_match(errors_by_mode)
    report = {
        "classification": label,
        "rule": "docs/PLAN_TESTBED.md section 2.11, recorded before this run",
        "track_threshold_nmae": TRACK_THRESHOLD,
        "matched_conditions": {"loads_l2_mbps": loads, "duration_s": duration, "warmup_s": warmup,
                               "base_rtt_ms": base_rtt, "jitter": 0.0,
                               "queue_limit_bytes": int(load_config().sim.queue_limit_bytes)},
        "testbed": tb,
        "simulator": sim_by_mode,
        "errors": errors_by_mode,
        **detail,
    }
    out = REPO_ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")

    def _f(v, nd=3):
        return "  n/a" if v is None else f"{v:.{nd}f}"

    print("=" * 108)
    print(f"  Simulator vs real OVS, matched constant load. base_rtt {base_rtt} ms, no jitter.")
    for m, nd, label_m in (("embb_goodput_l2_mbps", 3, "eMBB goodput, L2 Mbps"),
                           ("be_goodput_l2_mbps", 3, "BE goodput, L2 Mbps"),
                           ("urllc_rtt_p50_ms", 2, "URLLC median RTT, ms"),
                           ("urllc_rtt_p95_ms", 2, "URLLC p95 RTT, ms (not in rule)")):
        print(f"\n  {label_m}")
        print("  level   testbed           " + "  ".join(f"{k:>22}" for k in MODES))
        for i, share in enumerate(levels):
            half = tb[i].get(f"{m}_ci95_half")
            tbs = _f(tb[i][m], nd) + ("" if half is None else f" ± {_f(half, nd)}")
            print(f"  {share:.2f}    {tbs:<17} " +
                  "  ".join(f"{_f(sim_by_mode[k][i][m], nd):>22}" for k in MODES))
        if m in RULE_METRICS:
            print("  NMAE              " + " " * 18 +
                  "  ".join(f"{errors_by_mode[k][m]['nmae']:>17.3f} {'ok' if errors_by_mode[k][m]['tracks'] else '  '}  "
                            for k in MODES))
    print("\n" + "-" * 108)
    print("  mean NMAE:  " + "   ".join(f"{k} {v:.3f}" for k, v in detail["mean_nmae"].items()))
    print(f"  CLASSIFICATION  {label}   (threshold NMAE <= {TRACK_THRESHOLD})")
    print(f"  best mode tracks: {detail['metrics_tracked_by_best'] or 'nothing'}")
    print("=" * 108)
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
