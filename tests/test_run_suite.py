"""Suite runner tests.

The interesting one is test_execution_order_is_shuffled_and_reproducible. Randomized run order
is a claim docs/EXPERIMENTS.md makes about the protocol, so it has to be checked rather than
asserted: a shuffle that silently degraded to nested-loop order would leave every table looking
exactly the same while quietly reintroducing the confound it exists to remove.
"""

from __future__ import annotations

import json

import pandas as pd
import pytest

from experiments.run_suite import DEFAULT_ORDER_SEED, run_suite


SHORT = {"run.duration_s": 60, "run.warmup_s": 10}


@pytest.fixture(scope="module")
def suite(tmp_path_factory):
    """One small grid, reused by every test in this module. Two policies, two seeds."""
    root = tmp_path_factory.mktemp("suite")
    raw, summary = root / "raw", root / "summary"
    df = run_suite(
        policies=["static_safe", "threshold"],
        scenarios=["burst"],
        seeds=[0, 1],
        out_dir=raw,
        summary_dir=summary,
        overrides=SHORT,
        quiet=True,
    )
    return df, raw, summary


def test_every_cell_of_the_grid_ran(suite):
    df, raw, _ = suite
    assert len(df) == 4
    assert set(df["policy"]) == {"static_safe", "threshold"}
    assert set(df["seed"]) == {0, 1}
    assert sorted(p.stem for p in raw.glob("*.csv")) == [
        "static_safe_burst_0",
        "static_safe_burst_1",
        "threshold_burst_0",
        "threshold_burst_1",
    ]


def test_env_json_written_per_run(suite):
    _, raw, _ = suite
    for run_id in ("static_safe_burst_0", "threshold_burst_1"):
        with open(raw / run_id / "env.json", "r", encoding="utf-8") as fh:
            env = json.load(fh)
        assert env["run_id"] == run_id
        assert env["backend"] == "sim"
        assert env["config_hash"]


def test_execution_order_is_shuffled_and_reproducible(suite, tmp_path):
    _, _, summary = suite
    with open(summary / "suite_order.json", "r", encoding="utf-8") as fh:
        order = json.load(fh)

    executed = [(r["policy"], r["scenario"], r["seed"]) for r in order["execution_order"]]
    assert len(executed) == 4
    assert len(set(executed)) == 4                     # every cell exactly once
    assert order["order_seed"] == DEFAULT_ORDER_SEED

    nested_loop = [
        (p, "burst", s) for p in ("static_safe", "threshold") for s in (0, 1)
    ]
    assert executed != nested_loop, "shuffle degenerated to nested-loop order"

    # Same order seed, same order. This is what makes a suite reproducible rather than merely
    # randomized.
    repeat = run_suite(
        policies=["static_safe", "threshold"],
        scenarios=["burst"],
        seeds=[0, 1],
        out_dir=tmp_path / "raw",
        summary_dir=tmp_path / "summary",
        overrides=SHORT,
        quiet=True,
    )
    with open(tmp_path / "summary" / "suite_order.json", "r", encoding="utf-8") as fh:
        again = json.load(fh)
    assert [
        (r["policy"], r["scenario"], r["seed"]) for r in again["execution_order"]
    ] == executed
    assert len(repeat) == 4


def test_runs_csv_is_sorted_not_in_execution_order(suite):
    """The output file is sorted for reading; the shuffled order lives in suite_order.json."""
    df, _, summary = suite
    on_disk = pd.read_csv(summary / "runs.csv")
    expected = on_disk.sort_values(["policy", "scenario", "seed"]).reset_index(drop=True)
    pd.testing.assert_frame_equal(on_disk, expected)


def test_unknown_policy_fails_before_any_run(tmp_path):
    with pytest.raises(KeyError, match="unknown policy"):
        run_suite(
            policies=["no_such_policy"],
            scenarios=["burst"],
            seeds=[0],
            out_dir=tmp_path / "raw",
            summary_dir=tmp_path / "summary",
            overrides=SHORT,
            quiet=True,
        )
    assert not (tmp_path / "raw").exists()


def test_unknown_scenario_fails_before_any_run(tmp_path):
    with pytest.raises(KeyError, match="unknown scenario"):
        run_suite(
            policies=["static_safe"],
            scenarios=["no_such_scenario"],
            seeds=[0],
            out_dir=tmp_path / "raw",
            summary_dir=tmp_path / "summary",
            overrides=SHORT,
            quiet=True,
        )
    assert not (tmp_path / "raw").exists()
