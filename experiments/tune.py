"""Hyperparameter sweeps for the learned policies.

THE ONLY METHODOLOGICAL POINT THAT MATTERS HERE
The sweep runs on `experiment.train_seeds`, which are disjoint from `experiment.seeds`. Tuning
alpha on the seeds the method is then scored on is selection on the test set: with seven values
of alpha and ten evaluation seeds you can find a configuration that beats the baselines on those
seeds and learn nothing about whether the method works. The cost of doing it properly is that
the chosen alpha may not be the one that would have won on the evaluation seeds, which is the
entire point.

The selection criterion is mean reward, because that is what the reward function was designed to
express: throughput minus SLA damage minus drops. SLA violation rate and eMBB goodput are
reported alongside so that a configuration winning on reward while behaving badly on latency is
visible rather than hidden behind a single number.

Every value swept is also run to completion and logged, so the sweep table in
docs/EXPERIMENTS.md is a measurement, not a claim about what "worked well".
"""

from __future__ import annotations

import argparse
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

from analysis.aggregate import mean_ci  # noqa: E402
from config_loader import list_scenarios, load_config  # noqa: E402
from experiments.run_experiment import run_once  # noqa: E402

#: What to sweep, per policy: the dotted config key and the values.
SWEEPS = {
    "linucb": (
        "policy.linucb.alpha",
        # 0.0 is included on purpose: it is pure greedy, and if it wins then the confidence term
        # is not buying anything and the "UCB" in LinUCB is decoration at this problem size.
        [0.0, 0.02, 0.05, 0.10, 0.25, 0.50, 1.00, 2.00],
    ),
    "epsilon_greedy": (
        "policy.epsilon_greedy.epsilon",
        [0.0, 0.02, 0.05, 0.10, 0.20, 0.40],
    ),
}


def sweep(
    policy_name: str,
    scenarios: Optional[Sequence[str]] = None,
    seeds: Optional[Sequence[int]] = None,
    values: Optional[Sequence[float]] = None,
    quiet: bool = False,
) -> pd.DataFrame:
    """Run one policy at every value of its hyperparameter. Returns one row per run."""
    if policy_name not in SWEEPS:
        raise KeyError(f"no sweep defined for {policy_name!r}; have {sorted(SWEEPS)}")
    key, default_values = SWEEPS[policy_name]
    values = list(values) if values is not None else list(default_values)
    scenarios = list(scenarios) if scenarios else list_scenarios()
    if seeds is None:
        seeds = list(load_config().experiment.train_seeds)

    rows: List[Dict] = []
    total = len(values) * len(scenarios) * len(seeds)
    i = 0
    for value in values:
        for scenario in scenarios:
            for seed in seeds:
                i += 1
                cfg = load_config(scenario=scenario, overrides={key: value})
                t0 = time.time()
                # write=False: a sweep is not a result. Its logs must never land in results/raw
                # where analysis/aggregate.py would average them into the headline tables.
                _, m, _ = run_once(
                    policy_name=policy_name,
                    scenario=scenario,
                    seed=int(seed),
                    cfg=cfg,
                    write=False,
                )
                rows.append(
                    {
                        "policy": policy_name,
                        "param": key,
                        "value": value,
                        "scenario": scenario,
                        "seed": int(seed),
                        "mean_reward": m.mean_reward,
                        "sla_violation_rate": m.sla_violation_rate,
                        "embb_goodput_mbps": m.embb_goodput_mbps,
                        "urllc_rtt_p95_ms": m.urllc_rtt_p95,
                        "guardrail_intervention_rate": m.guardrail_intervention_rate,
                        "decision_latency_mean_ms": m.decision_latency_mean_ms,
                    }
                )
                if not quiet:
                    print(
                        f"[{i:>3}/{total}] {key}={value:<5} {scenario:<12} seed {seed} "
                        f"{time.time() - t0:5.2f}s  reward {m.mean_reward:+.4f}",
                        flush=True,
                    )
    return pd.DataFrame(rows)


def summarise(runs: pd.DataFrame) -> pd.DataFrame:
    """Collapse the sweep to one row per hyperparameter value, averaged over the sweep grid."""
    out: List[Dict] = []
    for value, grp in runs.groupby("value", sort=True):
        row = {"value": value, "n_runs": len(grp)}
        for metric in (
            "mean_reward",
            "sla_violation_rate",
            "embb_goodput_mbps",
            "urllc_rtt_p95_ms",
            "guardrail_intervention_rate",
            "decision_latency_mean_ms",
        ):
            stats = mean_ci(grp[metric].to_numpy())
            row[metric] = stats["mean"]
            row[f"{metric}_ci95"] = stats["ci95_half"]
        out.append(row)
    return pd.DataFrame(out).sort_values("value").reset_index(drop=True)


def best_value(summary: pd.DataFrame) -> float:
    return float(summary.loc[summary["mean_reward"].idxmax(), "value"])


def markdown(summary: pd.DataFrame, param: str) -> str:
    lines = [
        f"| `{param}` | mean reward | SLA violation rate | eMBB goodput (Mbps) | "
        "URLLC p95 (ms) | decision latency (ms) |",
        "|---|---|---|---|---|---|",
    ]
    best = best_value(summary)
    for _, r in summary.iterrows():
        label = f"**{r['value']}**" if r["value"] == best else f"{r['value']}"
        lines.append(
            f"| {label} | {r['mean_reward']:+.4f} ± {r['mean_reward_ci95']:.4f} | "
            f"{r['sla_violation_rate'] * 100:.2f} % | {r['embb_goodput_mbps']:.3f} | "
            f"{r['urllc_rtt_p95_ms']:.2f} | {r['decision_latency_mean_ms']:.4f} |"
        )
    lines.append("")
    lines.append(f"Selected: `{param}` = {best} (highest mean reward).")
    return "\n".join(lines)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Sweep a learned policy's hyperparameter.")
    ap.add_argument("--policy", default=None, choices=sorted(SWEEPS),
                    help="default: sweep every policy that has a sweep defined")
    ap.add_argument("--scenarios", nargs="+", default=None, choices=list_scenarios())
    ap.add_argument("--seeds", nargs="+", type=int, default=None,
                    help="default: experiment.train_seeds, which are disjoint from the "
                         "evaluation seeds. Overriding this with evaluation seeds is tuning on "
                         "the test set.")
    ap.add_argument("--values", nargs="+", type=float, default=None)
    ap.add_argument("--summary-dir", default=None)
    args = ap.parse_args(argv)

    policies = [args.policy] if args.policy else sorted(SWEEPS)
    summary_dir = Path(args.summary_dir) if args.summary_dir else REPO_ROOT / "results" / "summary"
    summary_dir.mkdir(parents=True, exist_ok=True)

    chosen: Dict[str, float] = {}
    for policy_name in policies:
        param, _ = SWEEPS[policy_name]
        runs = sweep(
            policy_name,
            scenarios=args.scenarios,
            seeds=args.seeds,
            values=args.values,
        )
        summary = summarise(runs)
        runs.to_csv(summary_dir / f"tuning_{policy_name}_runs.csv", index=False)
        summary.to_csv(summary_dir / f"tuning_{policy_name}.csv", index=False)
        chosen[param] = best_value(summary)
        print("")
        print(f"### {policy_name}")
        print("")
        print(markdown(summary, param))
        print("")

    with open(summary_dir / "tuning_selected.json", "w", encoding="utf-8") as fh:
        json.dump(
            {
                "selected": chosen,
                "seeds": args.seeds or list(load_config().experiment.train_seeds),
                "scenarios": args.scenarios or list_scenarios(),
                "note": "tuned on training seeds, disjoint from experiment.seeds",
            },
            fh,
            indent=2,
        )
    print("Update config/default.yaml by hand with the selected values, then re-run the suite.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
