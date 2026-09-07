"""Reward-weight sensitivity study.

THE QUESTION THIS ASKS
Not "what are the best reward weights" - that question is circular, since the weights define
what "best" means. The question is: **does the ordering of the policies survive a change in the
weights?** If LinUCB beats Threshold only at w_sla = 1.0 and loses at 2.0, then the headline
result is a property of a number someone chose, and the report has to say so. If the ordering
holds across the sweep, the conclusion is robust to that choice.

WHY w_drop IS SWEPT AND NOT JUST w_sla
docs/DESIGN.md section 5 records a problem with the drop term found in week 1a: `total_drops` is
dominated by q1 tail drops caused by eMBB offering far more than its cap allows, so RAISING the
eMBB cap REDUCES drops. w_drop is therefore not a congestion penalty, it is a second throughput
incentive pulling against the SLA term. A sensitivity study that swept only w_sla would leave
that confound untested while appearing thorough.

IT WRITES EVERY ROW AS IT GOES, AND IT RESUMES
The first attempt at this study was killed by the operating system for memory after roughly
an hour, and because it accumulated everything in memory and wrote the CSV only at the end,
all of it was lost. So each run is appended to `sensitivity_runs.csv` immediately, and
`--resume` reads that file and skips cells already present. A study that cannot survive being
interrupted silently costs an hour every time the machine sneezes.

THE ONE COMPARISON THAT IS NOT VALID HERE
Rewards from different cells of this sweep are not comparable to each other. Changing a weight
changes the units of the reward, so a bigger number in one cell does not mean a better outcome
than a smaller number in another. Only two things are read across cells: the RANKING of policies
within a cell, and the physical metrics - SLA violation rate, eMBB goodput, URLLC p95 - which are
measured in milliseconds and Mbps and mean the same thing regardless of what the weights are.
The output tables are built to make that hard to get wrong: rankings and physical metrics are
printed, raw cross-cell rewards are written to CSV but never tabulated side by side.
"""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from config_loader import list_scenarios, load_config  # noqa: E402
from experiments.run_experiment import run_once  # noqa: E402

DEFAULT_W_SLA = [0.25, 0.5, 1.0, 2.0, 4.0]
DEFAULT_W_DROP = [0.0, 0.25, 0.5, 1.0]

#: The policies whose ordering the study is about. The static baselines bracket the range, the
#: reactive baseline is the thing to beat, and LinUCB is the proposal. Pre-trained variants and
#: the Oracle are left out: each costs six runs per cell and neither changes the question.
DEFAULT_POLICIES = ["static_safe", "static_equal", "threshold", "linucb"]


ROW_FIELDS = [
    "w_sla", "w_drop", "policy", "scenario", "seed", "mean_reward", "sla_violation_rate",
    "embb_goodput_mbps", "urllc_rtt_p95_ms", "guardrail_intervention_rate",
]


def _load_existing(path: Path) -> pd.DataFrame:
    if path.exists():
        try:
            return pd.read_csv(path)
        except Exception:
            return pd.DataFrame(columns=ROW_FIELDS)
    return pd.DataFrame(columns=ROW_FIELDS)


def study(
    policies: Sequence[str] = DEFAULT_POLICIES,
    w_sla_values: Sequence[float] = DEFAULT_W_SLA,
    w_drop_values: Sequence[float] = DEFAULT_W_DROP,
    scenarios: Optional[Sequence[str]] = None,
    seeds: Optional[Sequence[int]] = None,
    quiet: bool = False,
    out_csv: Optional[Path] = None,
    resume: bool = False,
) -> pd.DataFrame:
    """One run per (policy, w_sla, w_drop, scenario, seed). Returns the long table.

    Rows are appended to `out_csv` as they are produced, so an interrupted study keeps whatever it
    had already finished and `resume=True` continues from there.
    """
    scenarios = list(scenarios) if scenarios else list_scenarios()
    if seeds is None:
        seeds = list(load_config().experiment.train_seeds)

    done = set()
    rows: List[Dict] = []
    if out_csv is not None:
        out_csv = Path(out_csv)
        if resume:
            for r in _load_existing(out_csv).to_dict("records"):
                done.add(
                    (float(r["w_sla"]), float(r["w_drop"]), r["policy"], r["scenario"],
                     int(r["seed"]))
                )
                rows.append(r)
        if not resume or not out_csv.exists():
            out_csv.parent.mkdir(parents=True, exist_ok=True)
            with open(out_csv, "w", encoding="utf-8", newline="") as fh:
                csv.DictWriter(fh, fieldnames=ROW_FIELDS).writeheader()
            done.clear()
            rows.clear()

    cells = list(itertools.product(w_sla_values, w_drop_values))
    total = len(cells) * len(policies) * len(scenarios) * len(seeds)
    i = 0
    for w_sla, w_drop in cells:
        for policy in policies:
            for scenario in scenarios:
                for seed in seeds:
                    i += 1
                    if (float(w_sla), float(w_drop), policy, scenario, int(seed)) in done:
                        continue
                    cfg = load_config(
                        scenario=scenario,
                        overrides={"reward.w_sla": w_sla, "reward.w_drop": w_drop},
                    )
                    t0 = time.time()
                    _, m, _ = run_once(
                        policy_name=policy, scenario=scenario, seed=int(seed),
                        cfg=cfg, write=False,
                    )
                    row = {
                        "w_sla": w_sla,
                        "w_drop": w_drop,
                        "policy": policy,
                        "scenario": scenario,
                        "seed": int(seed),
                        "mean_reward": m.mean_reward,
                        "sla_violation_rate": m.sla_violation_rate,
                        "embb_goodput_mbps": m.embb_goodput_mbps,
                        "urllc_rtt_p95_ms": m.urllc_rtt_p95,
                        "guardrail_intervention_rate": m.guardrail_intervention_rate,
                    }
                    rows.append(row)
                    if out_csv is not None:
                        with open(out_csv, "a", encoding="utf-8", newline="") as fh:
                            csv.DictWriter(fh, fieldnames=ROW_FIELDS).writerow(row)
                    if not quiet:
                        print(
                            f"[{i:>4}/{total}] w_sla={w_sla:<5} w_drop={w_drop:<5} "
                            f"{policy:<14} {scenario:<12} seed {seed} "
                            f"{time.time() - t0:5.2f}s",
                            flush=True,
                        )
    return pd.DataFrame(rows)


def rankings(runs: pd.DataFrame) -> pd.DataFrame:
    """Per (w_sla, w_drop) cell: the policies ordered by mean reward within that cell."""
    out: List[Dict] = []
    for (w_sla, w_drop), grp in runs.groupby(["w_sla", "w_drop"], sort=True):
        means = grp.groupby("policy")["mean_reward"].mean().sort_values(ascending=False)
        out.append(
            {
                "w_sla": w_sla,
                "w_drop": w_drop,
                "ranking": " > ".join(means.index),
                "winner": means.index[0],
                # Margin between first and second, in that cell's own reward units. Only
                # meaningful relative to that cell, never across cells.
                "margin_first_to_second": float(means.iloc[0] - means.iloc[1])
                if len(means) > 1
                else float("nan"),
            }
        )
    return pd.DataFrame(out)


def physical_table(runs: pd.DataFrame) -> pd.DataFrame:
    """Physical metrics per policy, averaged over the whole study.

    These do NOT depend on the reward weights - a millisecond is a millisecond whatever w_sla is
    - except through the policies' behaviour, which is the entire point. A learned policy's
    latency changes across the sweep because the weights changed what it learned to do.
    """
    return (
        runs.groupby(["policy", "w_sla", "w_drop"])[
            ["sla_violation_rate", "embb_goodput_mbps", "urllc_rtt_p95_ms"]
        ]
        .mean()
        .reset_index()
    )


def markdown_rankings(rank: pd.DataFrame) -> str:
    lines = [
        "Ordering by mean reward within each cell. Rewards are NOT comparable between cells: "
        "changing a weight changes the units.",
        "",
        "| `w_sla` | `w_drop` | ordering (best first) |",
        "|---|---|---|",
    ]
    for _, r in rank.iterrows():
        lines.append(f"| {r['w_sla']} | {r['w_drop']} | {r['ranking']} |")
    winners = sorted(set(rank["winner"]))
    lines.append("")
    if len(winners) == 1:
        lines.append(
            f"**The ordering's winner is stable across the whole sweep: `{winners[0]}` wins in "
            f"all {len(rank)} cells.**"
        )
    else:
        lines.append(
            f"**The winner is NOT stable: {len(winners)} different policies win somewhere in the "
            f"sweep ({', '.join(winners)}). The headline result depends on the reward weights "
            f"and the report must say so.**"
        )
    return "\n".join(lines)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Reward-weight sensitivity study.")
    ap.add_argument("--policies", nargs="+", default=DEFAULT_POLICIES)
    ap.add_argument("--w-sla", nargs="+", type=float, default=DEFAULT_W_SLA)
    ap.add_argument("--w-drop", nargs="+", type=float, default=DEFAULT_W_DROP)
    ap.add_argument("--scenarios", nargs="+", default=None, choices=list_scenarios())
    ap.add_argument("--seeds", nargs="+", type=int, default=None)
    ap.add_argument("--summary-dir", default=None)
    ap.add_argument(
        "--resume",
        action="store_true",
        help="keep rows already in sensitivity_runs.csv and run only the missing cells",
    )
    args = ap.parse_args(argv)

    summary_dir = Path(args.summary_dir) if args.summary_dir else REPO_ROOT / "results" / "summary"
    summary_dir.mkdir(parents=True, exist_ok=True)
    runs = study(
        policies=args.policies,
        w_sla_values=args.w_sla,
        w_drop_values=args.w_drop,
        scenarios=args.scenarios,
        seeds=args.seeds,
        out_csv=summary_dir / "sensitivity_runs.csv",
        resume=args.resume,
    )
    rank = rankings(runs)
    phys = physical_table(runs)

    rank.to_csv(summary_dir / "sensitivity_rankings.csv", index=False)
    phys.to_csv(summary_dir / "sensitivity_physical.csv", index=False)

    print("")
    print(markdown_rankings(rank))
    print("")
    print(f"wrote {summary_dir / 'sensitivity_rankings.csv'} and two more")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
