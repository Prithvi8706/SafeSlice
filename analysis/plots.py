"""Figures for the report.

Five figures, each answering one question. Nothing here computes a statistic: every number
plotted comes from `analysis/aggregate.py` or straight from a per-step log, so a figure and the
corresponding table cannot disagree.

    fig1_timeseries      Does the controller actually do anything? One run, latency and the
                         applied level over time, with the SLA lines and the guardrail events.
    fig2_tradeoff        THE HEADLINE. eMBB goodput against SLA violation rate, one point per
                         policy per scenario. A method that is better is up and to the left.
    fig3_bars            Per-metric bars with 95% CIs across seeds. The honest version of the
                         headline: where the intervals overlap, there is no separation to claim.
    fig4_guardrail       Guardrail intervention rate and the split between the EWMA path and the
                         fast instantaneous path, which is the evidence for that design choice.
    fig5_learning        Bandit learning curve: mean reward against step, against the flat lines
                         of the non-learning policies. Shows the online cost of exploration.

CHART CONVENTIONS
Error bars are the 95% t interval across seeds from aggregate.mean_ci, and every figure that
shows one says n in the caption. A bar with no error bar means one seed and is labelled as such
rather than drawn as if it were converged.

No figure uses a dual y-axis for two unrelated quantities except fig1, where latency and the
action level share a time axis and the action level is drawn as a step plot on a secondary axis
with its own explicit tick labels. Dual axes invite the reader to see a correlation that the
scaling invented, so it is used once, where the x axis genuinely is shared.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import matplotlib

matplotlib.use("Agg")  # no display on a headless VM, and figures must render identically
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from analysis.aggregate import REPORTED_METRICS, aggregate, load_runs  # noqa: E402
from analysis.metrics import post_warmup, validate_run_csv  # noqa: E402
from config_loader import load_config  # noqa: E402

#: One stable colour per policy so a policy is the same colour in every figure of the report.
POLICY_COLOURS = {
    "static_safe": "#4C72B0",
    "static_equal": "#55A868",
    "threshold": "#C44E52",
    "epsilon_greedy": "#8172B2",
    "epsilon_greedy_pretrained": "#B8A2D8",   # lighter shade of its online counterpart
    "linucb": "#CCB974",
    "linucb_pretrained": "#8C6D1F",           # darker shade of its online counterpart
    "oracle": "#937860",
}
#: Online and converged variants sit next to each other so the cost of exploration reads off the
#: figure as an adjacent pair rather than as two points the reader has to hunt for.
POLICY_ORDER = [
    "static_safe",
    "static_equal",
    "threshold",
    "epsilon_greedy",
    "epsilon_greedy_pretrained",
    "linucb",
    "linucb_pretrained",
    "oracle",
]


def _colour(policy: str) -> str:
    return POLICY_COLOURS.get(policy, "#666666")


def _ordered(policies: Sequence[str]) -> List[str]:
    known = [p for p in POLICY_ORDER if p in policies]
    return known + sorted(p for p in policies if p not in POLICY_ORDER)


def _save(fig, out_dir: Path, name: str) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{name}.png"
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return path


# --------------------------------------------------------------------------- fig 1


def fig1_timeseries(run_csv: Path, cfg=None, out_dir: Optional[Path] = None) -> Path:
    """One run over time: URLLC p95, the applied level, and where the guardrail fired."""
    run_csv = Path(run_csv)
    df = validate_run_csv(run_csv)
    cfg = cfg or load_config()
    d = post_warmup(df)

    fig, (ax_lat, ax_act) = plt.subplots(
        2, 1, figsize=(11, 6), sharex=True, gridspec_kw={"height_ratios": [2, 1]}
    )

    t = d["t_step"].to_numpy()
    ax_lat.plot(t, d["urllc_rtt_ms_p95"], lw=1.1, color="#333333", label="URLLC p95 (per step)")
    ax_lat.plot(t, d["urllc_rtt_ms_p50"], lw=0.9, color="#999999", label="URLLC p50 (per step)")
    ax_lat.axhline(float(cfg.sla.target_ms), ls="--", lw=1.0, color="#E8A33D",
                   label=f"SLA target {cfg.sla.target_ms} ms")
    ax_lat.axhline(float(cfg.sla.hard_ms), ls="--", lw=1.2, color="#C44E52",
                   label=f"SLA hard {cfg.sla.hard_ms} ms")
    ax_lat.axhline(float(cfg.guardrail.warn_ms), ls=":", lw=1.0, color="#4C72B0",
                   label=f"guardrail warn {cfg.guardrail.warn_ms} ms")

    viol = d[d["sla_violated"] == 1]
    if len(viol):
        ax_lat.scatter(viol["t_step"], viol["urllc_rtt_ms_p95"], s=14, color="#C44E52",
                       zorder=5, label=f"SLA violation ({len(viol)} steps)")
    ax_lat.set_ylabel("URLLC RTT (ms)")
    ax_lat.legend(loc="upper left", fontsize=8, ncol=2)
    ax_lat.set_title(run_csv.stem)

    levels = np.asarray(cfg.action.embb_levels, dtype=float)
    ax_act.step(t, d["applied_action"], where="post", lw=1.2, color="#55A868", label="applied")
    ax_act.step(t, d["proposed_action"], where="post", lw=0.8, ls="--", color="#8172B2",
                label="proposed")
    inter = d[d["guardrail_intervened"] == 1]
    if len(inter):
        ax_act.scatter(inter["t_step"], inter["applied_action"], s=16, color="#C44E52",
                       zorder=5, label=f"guardrail intervened ({len(inter)} steps)")
    ax_act.set_yticks(range(len(levels)))
    ax_act.set_yticklabels([f"{i}: {v:.2f}" for i, v in enumerate(levels)], fontsize=8)
    ax_act.set_ylabel("eMBB level")
    ax_act.set_xlabel("time (s, post warm-up)")
    ax_act.legend(loc="upper left", fontsize=8, ncol=3)

    out_dir = Path(out_dir) if out_dir else REPO_ROOT / "results" / "summary" / "figures"
    return _save(fig, out_dir, f"fig1_timeseries_{run_csv.stem}")


# --------------------------------------------------------------------------- fig 2


def fig2_tradeoff(runs: pd.DataFrame, out_dir: Optional[Path] = None) -> Path:
    """eMBB goodput against SLA violation rate. Up and to the left is better.

    This is the figure the whole project argues about, so it plots the seed MEAN with error
    bars in both axes rather than a cloud of per-seed points: a cloud lets the eye pick the
    convenient corner of each policy's spread.
    """
    summary = aggregate(runs, metrics=["embb_goodput_mbps", "sla_violation_rate"])
    scenarios = sorted(set(runs["scenario"]))
    policies = _ordered(sorted(set(runs["policy"])))

    ncols = min(len(scenarios), 2)
    nrows = int(np.ceil(len(scenarios) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 4.4 * nrows), squeeze=False)

    for ax, scenario in zip(axes.flat, scenarios):
        placed = []          # (x, y, policy) already drawn on this axis, for label fanning
        for policy in policies:
            g = summary[(summary["policy"] == policy) & (summary["scenario"] == scenario)]
            if g.empty:
                continue
            x = g[g["metric"] == "sla_violation_rate"]
            y = g[g["metric"] == "embb_goodput_mbps"]
            if x.empty or y.empty:
                continue
            xm, xe = float(x["mean"].iloc[0]) * 100, float(x["ci95_half"].iloc[0]) * 100
            ym, ye = float(y["mean"].iloc[0]), float(y["ci95_half"].iloc[0])
            ax.errorbar(
                xm, ym,
                xerr=None if not np.isfinite(xe) else xe,
                yerr=None if not np.isfinite(ye) else ye,
                fmt="o", ms=8, capsize=3, color=_colour(policy), label=policy,
            )
            # Policies whose means coincide would stack their labels on top of each other and
            # become unreadable, which matters here because coinciding IS the finding on ramp:
            # three policies land on the same point. Fan the offsets out by index so every
            # label survives, rather than silently drawing them over one another.
            placed.append((xm, ym, policy))
            dy = 5 + 11 * sum(
                1 for px, py, _ in placed[:-1]
                if abs(px - xm) <= 0.02 * max(abs(xm), 1e-9) + 1e-6
                and abs(py - ym) <= 0.01 * max(abs(ym), 1e-9) + 1e-9
            )
            ax.annotate(
                policy, (xm, ym), textcoords="offset points", xytext=(8, dy), fontsize=8
            )
        ax.set_title(scenario)
        ax.set_xlabel("SLA violation rate (% of steps)  ← better")
        ax.set_ylabel("eMBB goodput (Mbps)  better →")
        ax.grid(alpha=0.25)

    for ax in axes.flat[len(scenarios):]:
        ax.axis("off")
    n = int(runs.groupby(["policy", "scenario"]).size().max())
    fig.suptitle(f"Safety / throughput trade-off, mean over up to {n} seeds, 95% t intervals")
    fig.tight_layout()

    out_dir = Path(out_dir) if out_dir else REPO_ROOT / "results" / "summary" / "figures"
    return _save(fig, out_dir, "fig2_tradeoff")


# --------------------------------------------------------------------------- fig 3


def fig3_bars(
    runs: pd.DataFrame,
    metrics: Sequence[str] = ("mean_reward", "embb_goodput_mbps", "urllc_rtt_p95_ms",
                             "sla_violation_rate"),
    out_dir: Optional[Path] = None,
) -> Path:
    """Grouped bars per metric, policies within a scenario, with 95% CIs across seeds."""
    summary = aggregate(runs, metrics=list(metrics))
    scenarios = sorted(set(runs["scenario"]))
    policies = _ordered(sorted(set(runs["policy"])))

    fig, axes = plt.subplots(len(metrics), 1, figsize=(1.6 * len(scenarios) * max(len(policies), 3), 3.2 * len(metrics)), squeeze=False)
    width = 0.8 / max(len(policies), 1)

    for ax, metric in zip(axes.flat, metrics):
        label, unit, higher_better = REPORTED_METRICS[metric]
        base = np.arange(len(scenarios))
        for j, policy in enumerate(policies):
            means, errs = [], []
            for scenario in scenarios:
                row = summary[
                    (summary["policy"] == policy)
                    & (summary["scenario"] == scenario)
                    & (summary["metric"] == metric)
                ]
                if row.empty:
                    means.append(np.nan)
                    errs.append(np.nan)
                else:
                    m = float(row["mean"].iloc[0])
                    h = float(row["ci95_half"].iloc[0])
                    if unit == "frac":
                        m, h = m * 100, h * 100
                    means.append(m)
                    errs.append(h)
            errs_arr = np.asarray(errs, dtype=float)
            ax.bar(
                base + j * width, means, width * 0.92,
                yerr=None if np.all(~np.isfinite(errs_arr)) else np.nan_to_num(errs_arr),
                capsize=3, color=_colour(policy), label=policy,
            )
        ax.set_xticks(base + width * (len(policies) - 1) / 2)
        ax.set_xticklabels(scenarios)
        suffix = " (%)" if unit == "frac" else (f" ({unit})" if unit else "")
        arrow = "higher is better" if higher_better else "lower is better"
        ax.set_ylabel(f"{label}{suffix}")
        ax.set_title(f"{label} — {arrow}", fontsize=10)
        ax.grid(axis="y", alpha=0.25)
    axes.flat[0].legend(fontsize=8, ncol=len(policies))
    fig.tight_layout()

    out_dir = Path(out_dir) if out_dir else REPO_ROOT / "results" / "summary" / "figures"
    return _save(fig, out_dir, "fig3_bars")


# --------------------------------------------------------------------------- fig 4


def fig4_guardrail(raw_dir: Optional[Path] = None, out_dir: Optional[Path] = None) -> Path:
    """Guardrail activity: intervention rate per policy, and which signal escalated.

    The right panel is the evidence for the design decision in docs/DESIGN.md section 4. If the
    fast instantaneous path never fires, watching it was pointless and the report should say so.
    """
    raw_dir = Path(raw_dir) if raw_dir else REPO_ROOT / "results" / "raw"
    rows: List[Dict] = []
    for csv_path in sorted(raw_dir.glob("*.csv")):
        df = post_warmup(pd.read_csv(csv_path))
        policy, _, scenario = csv_path.stem.rpartition("_")[0].rpartition("_")
        counts = df["guardrail_escalation_source"].value_counts().to_dict()
        rows.append(
            {
                "policy": policy,
                "scenario": scenario,
                "intervention_rate": float(df["guardrail_intervened"].mean()),
                "n_steps": len(df),
                "src_ewma": int(counts.get("ewma", 0)),
                "src_p95": int(counts.get("p95", 0)),
                "src_both": int(counts.get("both", 0)),
            }
        )
    act = pd.DataFrame(rows)

    fig, (ax_l, ax_r) = plt.subplots(1, 2, figsize=(13, 4.4))

    grouped = act.groupby("policy")["intervention_rate"].agg(["mean", "count"])
    policies = _ordered(list(grouped.index))
    ax_l.bar(
        range(len(policies)),
        [grouped.loc[p, "mean"] * 100 for p in policies],
        color=[_colour(p) for p in policies],
    )
    ax_l.set_xticks(range(len(policies)))
    ax_l.set_xticklabels(policies, rotation=20, ha="right")
    ax_l.set_ylabel("guardrail intervention rate (% of steps)")
    ax_l.set_title("How often the applied action differed from the proposal")
    ax_l.grid(axis="y", alpha=0.25)

    src = act.groupby("policy")[["src_ewma", "src_p95", "src_both"]].sum().reindex(policies)
    bottom = np.zeros(len(policies))
    for col, colour, label in (
        ("src_ewma", "#4C72B0", "EWMA only (slow path)"),
        ("src_p95", "#C44E52", "instantaneous p95 only (fast path)"),
        ("src_both", "#937860", "both"),
    ):
        vals = src[col].to_numpy(dtype=float)
        ax_r.bar(range(len(policies)), vals, bottom=bottom, color=colour, label=label)
        bottom += vals
    ax_r.set_xticks(range(len(policies)))
    ax_r.set_xticklabels(policies, rotation=20, ha="right")
    ax_r.set_ylabel("escalating steps (count, all runs)")
    ax_r.set_title("Which signal tripped the guardrail")
    ax_r.legend(fontsize=8)
    ax_r.grid(axis="y", alpha=0.25)

    fig.tight_layout()
    out_dir = Path(out_dir) if out_dir else REPO_ROOT / "results" / "summary" / "figures"
    return _save(fig, out_dir, "fig4_guardrail")


# --------------------------------------------------------------------------- fig 5


def fig5_learning(
    raw_dir: Optional[Path] = None,
    scenario: str = "burst",
    window: int = 20,
    out_dir: Optional[Path] = None,
) -> Path:
    """Reward against step, averaged over seeds, one line per policy.

    Includes the warm-up steps on purpose. This is the only figure that does, and it is
    labelled: the cost of online exploration is paid at the start of a run, so cropping the
    warm-up would hide precisely the thing the figure exists to show.
    """
    raw_dir = Path(raw_dir) if raw_dir else REPO_ROOT / "results" / "raw"
    by_policy: Dict[str, List[np.ndarray]] = {}
    for csv_path in sorted(raw_dir.glob(f"*_{scenario}_*.csv")):
        policy = csv_path.stem.rpartition("_")[0].rpartition("_")[0]
        df = pd.read_csv(csv_path)
        by_policy.setdefault(policy, []).append(df["reward"].to_numpy(dtype=float))

    fig, ax = plt.subplots(figsize=(11, 4.6))
    for policy in _ordered(list(by_policy)):
        runs = by_policy[policy]
        n = min(len(r) for r in runs)
        stacked = np.vstack([r[:n] for r in runs])
        mean = stacked.mean(axis=0)
        if window > 1:
            kernel = np.ones(window) / window
            mean = np.convolve(mean, kernel, mode="valid")
            x = np.arange(len(mean)) + window - 1
        else:
            x = np.arange(len(mean))
        ax.plot(x, mean, lw=1.4, color=_colour(policy), label=f"{policy} (n={len(runs)})")

    ax.set_xlabel(f"control step (1 s each), warm-up INCLUDED, {window}-step moving average")
    ax.set_ylabel("reward")
    ax.set_title(f"Reward against step, {scenario}, averaged over seeds")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.25)
    fig.tight_layout()

    out_dir = Path(out_dir) if out_dir else REPO_ROOT / "results" / "summary" / "figures"
    return _save(fig, out_dir, f"fig5_learning_{scenario}")


# --------------------------------------------------------------------------- cli


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Render the report figures from the raw run logs.")
    ap.add_argument("--raw-dir", default=None)
    ap.add_argument("--out-dir", default=None)
    ap.add_argument(
        "--timeseries-run",
        default=None,
        help="run id for fig1, e.g. threshold_burst_0. Default: the first run found.",
    )
    ap.add_argument("--learning-scenario", default="burst")
    args = ap.parse_args(argv)

    raw_dir = Path(args.raw_dir) if args.raw_dir else REPO_ROOT / "results" / "raw"
    out_dir = Path(args.out_dir) if args.out_dir else REPO_ROOT / "results" / "summary" / "figures"

    runs = load_runs(raw_dir)
    written = []

    ts_run = args.timeseries_run or runs["run_id"].iloc[0]
    ts_path = raw_dir / f"{ts_run}.csv"
    if not ts_path.exists():
        raise FileNotFoundError(f"no such run log: {ts_path}")
    scenario = ts_run.rpartition("_")[0].rpartition("_")[2]
    written.append(fig1_timeseries(ts_path, cfg=load_config(scenario=scenario), out_dir=out_dir))
    written.append(fig2_tradeoff(runs, out_dir=out_dir))
    written.append(fig3_bars(runs, out_dir=out_dir))
    written.append(fig4_guardrail(raw_dir=raw_dir, out_dir=out_dir))
    written.append(fig5_learning(raw_dir=raw_dir, scenario=args.learning_scenario, out_dir=out_dir))

    for p in written:
        print(f"wrote {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
