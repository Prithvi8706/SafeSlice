"""The full experiment grid: policies x scenarios x seeds, in randomized order.

WHY THE ORDER IS RANDOMIZED
Runs execute back to back on one machine. If the grid ran in nested-loop order, every
`static_equal` run would happen while the machine was cold and every `linucb` run after an hour
of load, so any thermal throttling, memory pressure or background drift would land entirely on
one policy and appear in the results as a property of that policy. The randomization uses a
fixed, logged seed, so the order is reproducible: `results/summary/suite_order.json` records the
exact sequence that produced a given summary.

This matters most for `decision_latency_ms`, which is the one metric here measured against a
real wall clock rather than the virtual sim clock. It is the reason this runner is SEQUENTIAL
and not parallel. Running four workers at once would cut the suite time by roughly four and
would corrupt exactly the metric the project uses to claim the method is lightweight. A
throughput number obtained by contaminating the latency number is not a saving.

WHAT IT WRITES
    results/raw/{run_id}.csv         one per run, full per-step log, via run_experiment.run_once
    results/raw/{run_id}/env.json    environment fingerprint per run
    results/summary/runs.csv         one row per run, the headline metrics
    results/summary/suite_order.json the randomized order, the seed that produced it, timings

`runs.csv` is a cache, not a source of truth. `analysis/aggregate.py` recomputes every metric
from the per-step CSVs, so a change to a metric definition cannot leave a stale number in a
table.

SURVIVING AN INTERRUPTION
`suite_order.json` is written BEFORE the first run and updated after each one. An earlier version
wrote it only at the end; the OS killed that suite at run 299 of 320 for memory, and the record of
which order the 299 completed runs had executed in died with the process. Since randomized order
is a claim the protocol makes, losing the record invalidates the claim even though the runs were
fine. Writing the plan up front costs one file write and makes `--resume` possible: with it, cells
whose CSV already exists are skipped and marked as skipped in the order file, so a resumed suite
is honestly distinguishable from one that ran start to finish in a single pass.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import pandas as pd  # noqa: E402

from analysis.metrics import metrics_to_row  # noqa: E402
from config_loader import list_scenarios, load_config  # noqa: E402
from experiments.run_experiment import ALL_POLICY_NAMES, run_once  # noqa: E402

DEFAULT_ORDER_SEED = 20250907


def run_suite(
    policies: List[str],
    scenarios: List[str],
    seeds: List[int],
    out_dir: Optional[Path] = None,
    summary_dir: Optional[Path] = None,
    order_seed: int = DEFAULT_ORDER_SEED,
    overrides: Optional[Dict] = None,
    backend_name: str = "sim",
    quiet: bool = False,
    resume: bool = False,
) -> pd.DataFrame:
    """Execute the full grid and return one metrics row per run."""
    for p in policies:
        if p not in ALL_POLICY_NAMES:
            raise KeyError(f"unknown policy {p!r}; known: {ALL_POLICY_NAMES}")
    known_scenarios = list_scenarios()
    for s in scenarios:
        if s not in known_scenarios:
            raise KeyError(f"unknown scenario {s!r}; known: {known_scenarios}")

    grid = [(p, s, int(seed)) for p in policies for s in scenarios for seed in seeds]
    rng = random.Random(order_seed)
    rng.shuffle(grid)

    out_dir = Path(out_dir) if out_dir else REPO_ROOT / "results" / "raw"
    summary_dir = Path(summary_dir) if summary_dir else REPO_ROOT / "results" / "summary"
    summary_dir.mkdir(parents=True, exist_ok=True)

    rows: List[Dict] = []
    timings: List[Dict] = []
    t_suite = time.time()
    order_path = summary_dir / "suite_order.json"

    def _write_order(complete: bool) -> None:
        with open(order_path, "w", encoding="utf-8") as fh:
            json.dump(
                {
                    "order_seed": order_seed,
                    "n_runs": len(grid),
                    "complete": complete,
                    "resumed": bool(resume),
                    "suite_wall_seconds": time.time() - t_suite,
                    "policies": policies,
                    "scenarios": scenarios,
                    "seeds": list(seeds),
                    "overrides": overrides or {},
                    "planned_order": [
                        {"position": n, "policy": p, "scenario": s, "seed": sd}
                        for n, (p, s, sd) in enumerate(grid, start=1)
                    ],
                    "execution_order": timings,
                },
                fh,
                indent=2,
            )

    # The plan goes to disk before anything runs, so an interrupted suite still has a record of
    # the order it intended to execute in.
    _write_order(complete=False)

    for i, (policy, scenario, seed) in enumerate(grid, start=1):
        if resume and (out_dir / f"{policy}_{scenario}_{seed}.csv").exists():
            timings.append(
                {
                    "position": i, "policy": policy, "scenario": scenario, "seed": seed,
                    "wall_seconds": None, "skipped_existing": True,
                }
            )
            continue
        # The config is reloaded per run rather than shared, so a run cannot mutate the
        # configuration seen by a later run. Cheap, and it removes a whole class of
        # order-dependent bug that randomized ordering would otherwise make irreproducible.
        cfg = load_config(scenario=scenario, overrides=overrides)
        t0 = time.time()
        _, metrics, _ = run_once(
            policy_name=policy,
            scenario=scenario,
            seed=seed,
            cfg=cfg,
            out_dir=out_dir,
            backend_name=backend_name,
            write=True,
        )
        elapsed = time.time() - t0
        rows.append(metrics_to_row(metrics))
        timings.append(
            {
                "position": i,
                "policy": policy,
                "scenario": scenario,
                "seed": seed,
                "wall_seconds": elapsed,
                "skipped_existing": False,
            }
        )
        _write_order(complete=False)
        if not quiet:
            print(
                f"[{i:>4}/{len(grid)}] {policy:<14} {scenario:<12} seed {seed:<2} "
                f"{elapsed:6.2f}s  reward {metrics.mean_reward:+.4f}  "
                f"p95 {metrics.urllc_rtt_p95:5.2f} ms  viol {metrics.sla_violation_rate:6.2%}",
                flush=True,
            )

    df = pd.DataFrame(rows)
    # Sorted for readability. The EXECUTION order is the shuffled one and is recorded separately;
    # sorting the output file must not be mistaken for sorting the runs.
    if len(df):
        df = df.sort_values(["policy", "scenario", "seed"]).reset_index(drop=True)
        df.to_csv(summary_dir / "runs.csv", index=False)
    _write_order(complete=True)
    return df


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Run the full policy x scenario x seed grid.")
    ap.add_argument(
        "--policies",
        nargs="+",
        default=ALL_POLICY_NAMES,
        choices=ALL_POLICY_NAMES,
        help="default: every registered policy",
    )
    ap.add_argument(
        "--scenarios",
        nargs="+",
        default=list_scenarios(),
        choices=list_scenarios(),
    )
    ap.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=None,
        help="default: experiment.seeds from config/default.yaml",
    )
    ap.add_argument("--order-seed", type=int, default=DEFAULT_ORDER_SEED)
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--summary-dir", default=None)
    ap.add_argument("--set", action="append", default=[], metavar="KEY=VALUE")
    ap.add_argument(
        "--resume",
        action="store_true",
        help="skip cells whose run CSV already exists. Marks them as skipped in "
             "suite_order.json so a resumed suite is distinguishable from a single clean pass.",
    )
    args = ap.parse_args(argv)

    overrides = {}
    for item in args.set:
        if "=" not in item:
            ap.error(f"--set expects KEY=VALUE, got {item!r}")
        key, _, value = item.partition("=")
        try:
            overrides[key] = json.loads(value)
        except json.JSONDecodeError:
            overrides[key] = value

    seeds = args.seeds if args.seeds is not None else list(load_config().experiment.seeds)

    df = run_suite(
        policies=list(args.policies),
        scenarios=list(args.scenarios),
        seeds=seeds,
        out_dir=Path(args.out_dir) if args.out_dir else None,
        summary_dir=Path(args.summary_dir) if args.summary_dir else None,
        order_seed=args.order_seed,
        overrides=overrides or None,
        resume=args.resume,
    )
    print("")
    print(f"{len(df)} runs written to results/summary/runs.csv")
    print("run `python -m analysis.aggregate` for the seed-averaged tables")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
