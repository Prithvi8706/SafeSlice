"""The one and only definition of the per-run CSV schema, plus the metric definitions.

Everything that writes a run CSV imports RUN_COLUMNS from here, and everything that reads one
calls validate_run_csv on it. A schema defined in two places is a schema that will disagree with
itself somewhere around week 5, at which point half the results are silently incomparable.

The validator is called at the end of every run, not just in tests, because the failure mode it
catches (a column quietly going NaN for one policy) is invisible in aggregate plots.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import numpy as np
import pandas as pd

from agent.context import FEATURE_NAMES
from agent.guardrail import ALL_REASONS

# --------------------------------------------------------------------------- schema

TELEMETRY_COLUMNS: List[str] = [
    "t_wall",
    "t_step",
    "urllc_rtt_ms_p50",
    "urllc_rtt_ms_p95",
    "urllc_tx_bps",
    "embb_goodput_bps",
    "be_goodput_bps",
    "q0_drops",
    "q1_drops",
    "q2_drops",
    "link_util",
    "backlog_q0_bytes",
    "backlog_q1_bytes",
    "backlog_q2_bytes",
    "offered_urllc_bps",
    "offered_embb_bps",
    "offered_be_bps",
]

CONTEXT_COLUMNS: List[str] = [f"ctx_{n}" for n in FEATURE_NAMES]

DECISION_COLUMNS: List[str] = [
    "proposed_action",
    "applied_action",
    "applied_embb_share",
    "guardrail_reason",
    "guardrail_intervened",
    "guardrail_escalation_source",
    "decision_latency_ms",
]

REWARD_COLUMNS: List[str] = [
    "reward",
    "reward_tput",
    "reward_be",
    "reward_sla",
    "reward_viol",
    "reward_drop",
    "reward_rtt_used_ms",
    "sla_violated",
]

META_COLUMNS: List[str] = ["step_idx", "phase_id", "warmup"]

RUN_COLUMNS: List[str] = (
    META_COLUMNS + TELEMETRY_COLUMNS + CONTEXT_COLUMNS + DECISION_COLUMNS + REWARD_COLUMNS
)

#: Columns that must never contain NaN in a valid run.
REQUIRED_NUMERIC: List[str] = [
    c for c in RUN_COLUMNS if c not in ("guardrail_reason", "guardrail_escalation_source")
]

INTEGER_COLUMNS: List[str] = [
    "step_idx",
    "phase_id",
    "q0_drops",
    "q1_drops",
    "q2_drops",
    "proposed_action",
    "applied_action",
]


class SchemaError(ValueError):
    """Raised when a run CSV does not match RUN_COLUMNS or contains impossible values."""


def validate_run_csv(path: Path, n_actions: Optional[int] = None) -> pd.DataFrame:
    """Load a run CSV and assert it is well formed. Returns the DataFrame on success."""
    path = Path(path)
    if not path.exists():
        raise SchemaError(f"no such run CSV: {path}")
    df = pd.read_csv(path)
    validate_run_frame(df, source=str(path), n_actions=n_actions)
    return df


def validate_run_frame(
    df: pd.DataFrame, source: str = "<frame>", n_actions: Optional[int] = None
) -> None:
    actual = list(df.columns)
    if actual != RUN_COLUMNS:
        missing = [c for c in RUN_COLUMNS if c not in actual]
        extra = [c for c in actual if c not in RUN_COLUMNS]
        raise SchemaError(
            f"{source}: column mismatch.\n  missing: {missing}\n  extra: {extra}\n"
            f"  (order matters; expected {len(RUN_COLUMNS)} columns, got {len(actual)})"
        )

    if len(df) == 0:
        raise SchemaError(f"{source}: run has zero rows")

    for col in REQUIRED_NUMERIC:
        series = pd.to_numeric(df[col], errors="coerce")
        n_bad = int(series.isna().sum())
        if n_bad:
            rows = list(df.index[series.isna()][:5])
            raise SchemaError(
                f"{source}: column {col!r} has {n_bad} NaN or non-numeric values "
                f"(first rows: {rows})"
            )

    step = df["step_idx"].to_numpy()
    if not np.array_equal(step, np.arange(len(df))):
        raise SchemaError(f"{source}: step_idx is not 0..{len(df) - 1} contiguous")

    bad_reasons = sorted(set(df["guardrail_reason"]) - set(ALL_REASONS))
    if bad_reasons:
        raise SchemaError(f"{source}: unknown guardrail_reason values {bad_reasons}")

    if n_actions is not None:
        for col in ("proposed_action", "applied_action"):
            vals = df[col].to_numpy()
            if vals.min() < 0 or vals.max() >= n_actions:
                raise SchemaError(
                    f"{source}: {col} outside 0..{n_actions - 1} "
                    f"(saw {vals.min()}..{vals.max()})"
                )

    for col in ("link_util",):
        vals = df[col].to_numpy()
        if vals.min() < -1e-9 or vals.max() > 1.0 + 1e-6:
            raise SchemaError(f"{source}: {col} outside [0, 1] (saw {vals.min()}..{vals.max()})")

    for col in ("urllc_rtt_ms_p50", "urllc_rtt_ms_p95"):
        if (df[col].to_numpy() < 0).any():
            raise SchemaError(f"{source}: {col} has negative values")

    if (df["urllc_rtt_ms_p95"].to_numpy() + 1e-9 < df["urllc_rtt_ms_p50"].to_numpy()).any():
        raise SchemaError(f"{source}: p95 latency below p50 on at least one step")


# --------------------------------------------------------------------------- metrics


@dataclass(frozen=True)
class RunMetrics:
    """The headline metrics for one run, all computed post-warmup."""

    policy: str
    scenario: str
    seed: int
    n_steps: int
    urllc_rtt_p50: float
    urllc_rtt_p95: float
    urllc_rtt_p99: float
    sla_violation_rate: float
    embb_goodput_mbps: float
    be_goodput_mbps: float
    total_drops: int
    guardrail_intervention_rate: float
    decision_latency_mean_ms: float
    decision_latency_p99_ms: float
    mean_reward: float
    link_util_mean: float


def post_warmup(df: pd.DataFrame) -> pd.DataFrame:
    """Drop warm-up rows. Every reported metric goes through here."""
    return df.loc[df["warmup"] == 0]


def compute_run_metrics(df: pd.DataFrame, policy: str, scenario: str, seed: int) -> RunMetrics:
    """Headline metrics for one run.

    Note on urllc_rtt_p50/p95/p99: these are percentiles over the per-step statistics, not over
    individual packets. A p99 of a per-step p95 is not the same thing as a packet-level p99 and
    it must not be labelled as one in the report. Naming it honestly here so the distinction
    survives into the write-up.
    """
    d = post_warmup(df)
    if len(d) == 0:
        raise ValueError("no post-warmup rows; run.warmup_s is >= run.duration_s")

    p50_series = d["urllc_rtt_ms_p50"].to_numpy()
    p95_series = d["urllc_rtt_ms_p95"].to_numpy()
    lat = d["decision_latency_ms"].to_numpy()

    return RunMetrics(
        policy=policy,
        scenario=scenario,
        seed=int(seed),
        n_steps=int(len(d)),
        urllc_rtt_p50=float(np.percentile(p50_series, 50)),
        urllc_rtt_p95=float(np.percentile(p95_series, 95)),
        urllc_rtt_p99=float(np.percentile(p95_series, 99)),
        sla_violation_rate=float(d["sla_violated"].mean()),
        embb_goodput_mbps=float(d["embb_goodput_bps"].mean() / 1e6),
        be_goodput_mbps=float(d["be_goodput_bps"].mean() / 1e6),
        total_drops=int(d[["q0_drops", "q1_drops", "q2_drops"]].to_numpy().sum()),
        guardrail_intervention_rate=float(d["guardrail_intervened"].mean()),
        decision_latency_mean_ms=float(np.mean(lat)),
        decision_latency_p99_ms=float(np.percentile(lat, 99)),
        mean_reward=float(d["reward"].mean()),
        link_util_mean=float(d["link_util"].mean()),
    )


def metrics_to_row(m: RunMetrics) -> Dict[str, object]:
    return {
        "policy": m.policy,
        "scenario": m.scenario,
        "seed": m.seed,
        "n_steps": m.n_steps,
        "urllc_rtt_p50_ms": m.urllc_rtt_p50,
        "urllc_rtt_p95_ms": m.urllc_rtt_p95,
        "urllc_rtt_p99_ms": m.urllc_rtt_p99,
        "sla_violation_rate": m.sla_violation_rate,
        "embb_goodput_mbps": m.embb_goodput_mbps,
        "be_goodput_mbps": m.be_goodput_mbps,
        "total_drops": m.total_drops,
        "guardrail_intervention_rate": m.guardrail_intervention_rate,
        "decision_latency_mean_ms": m.decision_latency_mean_ms,
        "decision_latency_p99_ms": m.decision_latency_p99_ms,
        "mean_reward": m.mean_reward,
        "link_util_mean": m.link_util_mean,
    }
