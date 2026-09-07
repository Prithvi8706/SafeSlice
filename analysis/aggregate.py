"""Seed-averaged aggregation over run CSVs.

EVERY NUMBER HERE IS RECOMPUTED FROM THE PER-STEP LOGS.
`results/summary/runs.csv` written by the suite runner is a cache and is deliberately ignored by
default. If a metric definition in `analysis/metrics.py` changes, re-running this produces the
new number; reading the cache would produce the old one, and nothing in the file would say which
definition it came from. The `--from-cache` flag exists only for the case where the raw logs have
been deleted, and it prints a warning saying the numbers may predate the current definitions.

WHY THE COMPARISON IS PAIRED
A seed fixes the offered-load trace bit for bit (`traffic/traces.py:trace_seed`), so every policy
sees the identical traffic on seed k. The right question is therefore "on the same traffic, how
much better was policy A than policy B", and the right statistic is the mean of the per-seed
differences with a CI on that mean, not the difference of two independent means. Unpaired
intervals here would be wider than the truth and would hide real separations behind
scenario-to-scenario variance that both policies experienced equally.

WHY THE INTERVAL IS A t INTERVAL AND WHAT IT IS NOT
n is 10 seeds, so the normal approximation is not appropriate and scipy's t quantile is used.
The interval describes variation across SEEDS of the same simulator. It is not a confidence
interval on real network behaviour, and it says nothing about whether the simulator resembles
the testbed. That question is answered by the sim-vs-OVS comparison, not by narrowing this
interval with more seeds.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from analysis.metrics import (  # noqa: E402
    RunMetrics,
    compute_run_metrics,
    metrics_to_row,
    validate_run_csv,
)
from config_loader import list_scenarios  # noqa: E402

#: The metrics reported in tables, in the order the report shows them. Keys are columns of the
#: frame returned by load_runs(); the value is (display name, unit, "higher is better"?).
REPORTED_METRICS = {
    "urllc_rtt_p50_ms": ("URLLC RTT p50", "ms", False),
    "urllc_rtt_p95_ms": ("URLLC RTT p95", "ms", False),
    "urllc_rtt_p99_ms": ("URLLC RTT p99", "ms", False),
    "sla_violation_rate": ("SLA violation rate", "frac", False),
    "embb_goodput_mbps": ("eMBB goodput", "Mbps", True),
    "be_goodput_mbps": ("BE goodput", "Mbps", True),
    "total_drops": ("Total drops", "pkts", False),
    "guardrail_intervention_rate": ("Guardrail intervention rate", "frac", False),
    "decision_latency_mean_ms": ("Decision latency mean", "ms", False),
    "decision_latency_p99_ms": ("Decision latency p99", "ms", False),
    "mean_reward": ("Mean reward", "", True),
    "link_util_mean": ("Link utilisation", "frac", True),
}


# --------------------------------------------------------------------------- loading


def parse_run_id(run_id: str, scenarios: Optional[Sequence[str]] = None):
    """Split "{policy}_{scenario}_{seed}" back into its parts.

    Policy names contain underscores ("static_equal") and scenario names do not, so the split is
    done from the right. The scenario is then checked against the known set, because a silent
    misparse here would attribute a run to the wrong policy and no downstream check would catch
    it.
    """
    scenarios = list(scenarios) if scenarios is not None else list_scenarios()
    head, _, seed_str = run_id.rpartition("_")
    policy, _, scenario = head.rpartition("_")
    if not policy or scenario not in scenarios:
        raise ValueError(
            f"cannot parse run id {run_id!r}: got policy={policy!r} scenario={scenario!r} "
            f"seed={seed_str!r} (known scenarios: {scenarios})"
        )
    return policy, scenario, int(seed_str)


def load_runs(raw_dir: Optional[Path] = None, strict: bool = True) -> pd.DataFrame:
    """Recompute headline metrics for every run CSV under `raw_dir`.

    Identity comes from `{run_id}/env.json` when it exists, because that file recorded the
    policy, scenario and seed at run time. Falling back to the filename is only for logs whose
    env.json was lost.
    """
    raw_dir = Path(raw_dir) if raw_dir else REPO_ROOT / "results" / "raw"
    if not raw_dir.exists():
        raise FileNotFoundError(f"no raw results directory at {raw_dir}")

    rows: List[Dict] = []
    for csv_path in sorted(raw_dir.glob("*.csv")):
        run_id = csv_path.stem
        env_path = csv_path.parent / run_id / "env.json"
        if env_path.exists():
            with open(env_path, "r", encoding="utf-8") as fh:
                env = json.load(fh)
            policy, scenario, seed = env["policy"], env["scenario"], int(env["seed"])
        else:
            policy, scenario, seed = parse_run_id(run_id)

        try:
            df = validate_run_csv(csv_path)
        except Exception as exc:  # a malformed log must never be silently averaged in
            if strict:
                raise
            print(f"  skipped {run_id}: {exc}", file=sys.stderr)
            continue

        m = compute_run_metrics(df, policy, scenario, seed)
        row = metrics_to_row(m)
        row["run_id"] = run_id
        rows.append(row)

    if not rows:
        raise FileNotFoundError(f"no run CSVs found under {raw_dir}")
    return pd.DataFrame(rows).sort_values(["policy", "scenario", "seed"]).reset_index(drop=True)


# --------------------------------------------------------------------------- statistics


def _t_quantile(n: int, conf: float = 0.95) -> float:
    """Two-sided t quantile with n-1 degrees of freedom. NaN when n < 2."""
    if n < 2:
        return float("nan")
    from scipy import stats  # imported here so metrics-only users need no scipy

    return float(stats.t.ppf(0.5 + conf / 2.0, df=n - 1))


def mean_ci(values: Iterable[float], conf: float = 0.95) -> Dict[str, float]:
    """Mean and a t-based confidence interval. n < 2 gives a mean and a NaN interval.

    The NaN is deliberate. A single-seed run has no measurable spread, and emitting a zero-width
    interval would present one run as if it were a converged estimate.
    """
    arr = np.asarray(list(values), dtype=float)
    n = int(arr.size)
    mean = float(arr.mean()) if n else float("nan")
    if n < 2:
        return {"n": n, "mean": mean, "sd": float("nan"), "ci95_half": float("nan"),
                "ci95_lo": float("nan"), "ci95_hi": float("nan")}
    sd = float(arr.std(ddof=1))
    half = _t_quantile(n, conf) * sd / np.sqrt(n)
    return {
        "n": n,
        "mean": mean,
        "sd": sd,
        "ci95_half": float(half),
        "ci95_lo": float(mean - half),
        "ci95_hi": float(mean + half),
    }


def aggregate(runs: pd.DataFrame, metrics: Optional[Sequence[str]] = None) -> pd.DataFrame:
    """Long-format seed-averaged table: one row per (policy, scenario, metric)."""
    metrics = list(metrics) if metrics else list(REPORTED_METRICS)
    out: List[Dict] = []
    for (policy, scenario), grp in runs.groupby(["policy", "scenario"], sort=True):
        for metric in metrics:
            stats_row = mean_ci(grp[metric].to_numpy())
            out.append({"policy": policy, "scenario": scenario, "metric": metric, **stats_row})
    return pd.DataFrame(out)


def paired_deltas(
    runs: pd.DataFrame, baseline: str, metrics: Optional[Sequence[str]] = None
) -> pd.DataFrame:
    """Per-seed differences against `baseline`, averaged, with a paired CI.

    Only seeds present for BOTH policies are used, and `n_paired` reports how many that was. An
    unbalanced grid silently dropping to two shared seeds would otherwise produce a plausible
    looking interval built from almost nothing.
    """
    metrics = list(metrics) if metrics else list(REPORTED_METRICS)
    if baseline not in set(runs["policy"]):
        raise KeyError(f"baseline policy {baseline!r} not in results (have {sorted(set(runs['policy']))})")

    out: List[Dict] = []
    for scenario, scen_runs in runs.groupby("scenario", sort=True):
        base = scen_runs[scen_runs["policy"] == baseline].set_index("seed")
        if base.empty:
            continue
        for policy, grp in scen_runs.groupby("policy", sort=True):
            if policy == baseline:
                continue
            other = grp.set_index("seed")
            shared = sorted(set(base.index) & set(other.index))
            if not shared:
                continue
            for metric in metrics:
                diffs = other.loc[shared, metric].to_numpy(dtype=float) - base.loc[
                    shared, metric
                ].to_numpy(dtype=float)
                stats_row = mean_ci(diffs)
                # A difference whose interval straddles zero is not evidence of a difference.
                separated = bool(
                    np.isfinite(stats_row["ci95_lo"])
                    and (stats_row["ci95_lo"] > 0 or stats_row["ci95_hi"] < 0)
                )
                out.append(
                    {
                        "policy": policy,
                        "baseline": baseline,
                        "scenario": scenario,
                        "metric": metric,
                        "n_paired": len(shared),
                        "mean_delta": stats_row["mean"],
                        "ci95_half": stats_row["ci95_half"],
                        "ci95_lo": stats_row["ci95_lo"],
                        "ci95_hi": stats_row["ci95_hi"],
                        "separated_from_zero": separated,
                    }
                )
    return pd.DataFrame(out)


# --------------------------------------------------------------------------- presentation


def _fmt(metric: str, value: float) -> str:
    if not np.isfinite(value):
        return "n/a"
    _, unit, _ = REPORTED_METRICS.get(metric, ("", "", True))
    if unit == "frac":
        return f"{value * 100:.2f} %"
    if unit == "pkts":
        return f"{value:,.0f}"
    if unit == "ms":
        return f"{value:.3f}" if abs(value) < 1 else f"{value:.2f}"
    if unit == "Mbps":
        return f"{value:.3f}"
    return f"{value:.4f}"


def _fmt_ci(metric: str, mean: float, half: float) -> str:
    if not np.isfinite(half):
        return _fmt(metric, mean)
    _, unit, _ = REPORTED_METRICS.get(metric, ("", "", True))
    if unit == "frac":
        return f"{mean * 100:.2f} ± {half * 100:.2f} %"
    return f"{_fmt(metric, mean)} ± {_fmt(metric, half)}"


def markdown_table(summary: pd.DataFrame, scenario: str, policies: Optional[Sequence[str]] = None) -> str:
    """One scenario's table: metrics down the side, policies across the top, mean ± CI."""
    sub = summary[summary["scenario"] == scenario]
    if sub.empty:
        return f"(no runs for scenario {scenario})"
    cols = list(policies) if policies else sorted(set(sub["policy"]))
    n_by_policy = {
        p: int(sub[sub["policy"] == p]["n"].max()) if not sub[sub["policy"] == p].empty else 0
        for p in cols
    }

    lines = [
        "| Metric | " + " | ".join(f"{p} (n={n_by_policy[p]})" for p in cols) + " |",
        "|---" * (len(cols) + 1) + "|",
    ]
    for metric, (label, unit, _) in REPORTED_METRICS.items():
        cells = []
        for p in cols:
            row = sub[(sub["policy"] == p) & (sub["metric"] == metric)]
            if row.empty:
                cells.append("—")
            else:
                cells.append(
                    _fmt_ci(metric, float(row["mean"].iloc[0]), float(row["ci95_half"].iloc[0]))
                )
        unit_suffix = f" ({unit})" if unit and unit not in ("frac",) else ""
        lines.append(f"| {label}{unit_suffix} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def markdown_deltas(deltas: pd.DataFrame, metric: str = "mean_reward") -> str:
    """Paired differences for one metric, policies down the side, scenarios across."""
    sub = deltas[deltas["metric"] == metric]
    if sub.empty:
        return f"(no paired deltas for {metric})"
    scenarios = sorted(set(sub["scenario"]))
    policies = sorted(set(sub["policy"]))
    baseline = sub["baseline"].iloc[0]

    lines = [
        f"Paired per-seed difference vs `{baseline}`, metric `{metric}`. "
        "Bold means the 95% interval excludes zero.",
        "",
        "| Policy | " + " | ".join(scenarios) + " |",
        "|---" * (len(scenarios) + 1) + "|",
    ]
    for p in policies:
        cells = []
        for s in scenarios:
            row = sub[(sub["policy"] == p) & (sub["scenario"] == s)]
            if row.empty:
                cells.append("—")
                continue
            mean = float(row["mean_delta"].iloc[0])
            half = float(row["ci95_half"].iloc[0])
            text = _fmt_ci(metric, mean, half)
            if bool(row["separated_from_zero"].iloc[0]):
                text = f"**{text}**"
            cells.append(text)
        lines.append(f"| {p} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


# --------------------------------------------------------------------------- cli


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Seed-averaged tables from the raw run logs.")
    ap.add_argument("--raw-dir", default=None)
    ap.add_argument("--summary-dir", default=None)
    ap.add_argument("--baseline", default="static_equal", help="policy the paired deltas use")
    ap.add_argument("--delta-metric", default="mean_reward")
    ap.add_argument("--scenarios", nargs="+", default=None)
    ap.add_argument(
        "--lenient",
        action="store_true",
        help="skip malformed run CSVs instead of failing (they are reported on stderr)",
    )
    args = ap.parse_args(argv)

    runs = load_runs(Path(args.raw_dir) if args.raw_dir else None, strict=not args.lenient)
    summary = aggregate(runs)

    summary_dir = Path(args.summary_dir) if args.summary_dir else REPO_ROOT / "results" / "summary"
    summary_dir.mkdir(parents=True, exist_ok=True)
    runs.to_csv(summary_dir / "runs_recomputed.csv", index=False)
    summary.to_csv(summary_dir / "summary.csv", index=False)

    try:
        deltas = paired_deltas(runs, baseline=args.baseline)
        deltas.to_csv(summary_dir / "deltas.csv", index=False)
    except KeyError as exc:
        deltas = pd.DataFrame()
        print(f"no paired deltas: {exc}", file=sys.stderr)

    scenarios = args.scenarios or sorted(set(runs["scenario"]))
    for scenario in scenarios:
        print("")
        print(f"### {scenario}")
        print("")
        print(markdown_table(summary, scenario))
    if not deltas.empty:
        print("")
        print(markdown_deltas(deltas, metric=args.delta_metric))

    print("")
    print(f"wrote {summary_dir / 'summary.csv'}, {summary_dir / 'runs_recomputed.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
