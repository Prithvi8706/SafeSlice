"""Resume and crash-survival tests for the suite runner.

These exist because of a real incident, not a hypothetical one: the first full suite was killed
by the OS at run 299 of 320 for memory, and because `suite_order.json` was written only on
successful completion, the record of the randomized execution order died with the process. The
runs themselves were fine; the protocol claim that they executed in a randomized order was no
longer evidenced. Hence: write the plan first, update it as you go, and mark resumed cells so a
resumed suite can never be mistaken for a single clean pass.
"""

from __future__ import annotations

import json

import pytest

from experiments.run_suite import run_suite

SHORT = {"run.duration_s": 60, "run.warmup_s": 10}


def test_order_file_exists_before_the_suite_finishes(tmp_path):
    """The plan must be on disk up front, so an interruption cannot erase it."""
    raw, summary = tmp_path / "raw", tmp_path / "summary"
    run_suite(
        policies=["static_safe"], scenarios=["burst"], seeds=[0],
        out_dir=raw, summary_dir=summary, overrides=SHORT, quiet=True,
    )
    with open(summary / "suite_order.json", "r", encoding="utf-8") as fh:
        order = json.load(fh)
    assert order["complete"] is True
    assert order["resumed"] is False
    # The planned order is recorded independently of what actually executed.
    assert [(r["policy"], r["scenario"], r["seed"]) for r in order["planned_order"]] == [
        ("static_safe", "burst", 0)
    ]


def test_resume_skips_completed_cells_and_says_so(tmp_path):
    raw, summary = tmp_path / "raw", tmp_path / "summary"
    run_suite(
        policies=["static_safe"], scenarios=["burst"], seeds=[0, 1],
        out_dir=raw, summary_dir=summary, overrides=SHORT, quiet=True,
    )
    assert (raw / "static_safe_burst_0.csv").exists()

    # Resuming the same grid should execute nothing new.
    df = run_suite(
        policies=["static_safe"], scenarios=["burst"], seeds=[0, 1],
        out_dir=raw, summary_dir=summary, overrides=SHORT, quiet=True, resume=True,
    )
    assert len(df) == 0

    with open(summary / "suite_order.json", "r", encoding="utf-8") as fh:
        order = json.load(fh)
    assert order["resumed"] is True
    assert all(r["skipped_existing"] for r in order["execution_order"])
    assert len(order["execution_order"]) == 2


def test_resume_runs_only_the_missing_cells(tmp_path):
    raw, summary = tmp_path / "raw", tmp_path / "summary"
    run_suite(
        policies=["static_safe"], scenarios=["burst"], seeds=[0],
        out_dir=raw, summary_dir=summary, overrides=SHORT, quiet=True,
    )
    df = run_suite(
        policies=["static_safe"], scenarios=["burst"], seeds=[0, 1],
        out_dir=raw, summary_dir=summary, overrides=SHORT, quiet=True, resume=True,
    )
    assert len(df) == 1
    assert int(df.iloc[0]["seed"]) == 1

    with open(summary / "suite_order.json", "r", encoding="utf-8") as fh:
        order = json.load(fh)
    executed = [r for r in order["execution_order"] if not r["skipped_existing"]]
    skipped = [r for r in order["execution_order"] if r["skipped_existing"]]
    assert len(executed) == 1 and len(skipped) == 1
    assert executed[0]["wall_seconds"] is not None
    assert skipped[0]["wall_seconds"] is None


def test_a_clean_pass_is_not_marked_resumed(tmp_path):
    """The distinction has to be visible in the artefact, not just in someone's memory."""
    raw, summary = tmp_path / "raw", tmp_path / "summary"
    run_suite(
        policies=["static_safe"], scenarios=["burst"], seeds=[0],
        out_dir=raw, summary_dir=summary, overrides=SHORT, quiet=True,
    )
    with open(summary / "suite_order.json", "r", encoding="utf-8") as fh:
        order = json.load(fh)
    assert order["resumed"] is False
    assert not any(r["skipped_existing"] for r in order["execution_order"])
