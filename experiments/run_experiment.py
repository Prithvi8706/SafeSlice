"""One run: a single (policy, scenario, seed) triple.

CAUSAL ORDERING, which is the part that is easy to get silently wrong.

Within one control step the loop does:

    decide using the context built from the PREVIOUS interval's telemetry
    program the allocation
    let the interval elapse and read the telemetry it produced
    compute the reward from that telemetry
    credit the reward to the action that was applied at the start of the interval

This matters twice. First, a controller can only act on what it has already measured, so
conditioning on telemetry from the interval the action is about to influence would be
lookahead and would make every learned policy look better than it could ever be in
deployment. Second, the bandit update has to credit the action that caused the outcome. An
off-by-one here would train the model on the wrong arm and the bug would show up as "the
bandit does not learn", which is a conclusion we would then wrongly report as a finding.

The logged row therefore carries the telemetry measured at the END of interval t, the context
used to DECIDE at the start of interval t (built from interval t-1), and the action applied
during interval t. That is one coherent causal unit.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import platform
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np  # noqa: E402

from agent.context import FEATURE_NAMES, ContextBuilder  # noqa: E402
from agent.guardrail import Guardrail  # noqa: E402
from agent.policies.base import all_allowed  # noqa: E402
from agent.policies.static import StaticEqual, StaticSafe  # noqa: E402
from agent.policies.threshold import Threshold  # noqa: E402
from agent.reward import reward_breakdown  # noqa: E402
from analysis.metrics import (  # noqa: E402
    RUN_COLUMNS,
    compute_run_metrics,
    validate_run_csv,
)
from config_loader import config_hash, list_scenarios, load_config  # noqa: E402
from net.backend import allocation_from_level  # noqa: E402
from net.sim_backend import SimBackend  # noqa: E402
from traffic.traces import build_trace  # noqa: E402

#: The registry. Adding a policy is a one-line change here and the runner never needs to know
#: what kind of policy it is holding: static, reactive, learned or oracle all go through the
#: same guardrail and produce the same CSV schema, which is what makes the rows comparable.
POLICIES = {
    "static_equal": StaticEqual,
    "static_safe": StaticSafe,
    "threshold": Threshold,
}

BACKENDS = ("sim",)  # ovs_cli and ryu land in Week 1b and Week 4


# --------------------------------------------------------------------------- environment


def environment_fingerprint(cfg, extra: Optional[Dict] = None) -> Dict:
    """Everything about the machine that could plausibly move a number.

    Recorded per run so that a result which cannot be reproduced can at least be explained.
    Fields that do not exist on this platform are recorded as null rather than omitted, so a
    missing field is distinguishable from a field that was never collected.
    """

    def _cmd(args) -> Optional[str]:
        exe = shutil.which(args[0])
        if exe is None:
            return None
        try:
            out = subprocess.run(
                args, capture_output=True, text=True, timeout=10, check=False
            )
            return (out.stdout or out.stderr).strip().splitlines()[0]
        except Exception:
            return None

    try:
        loadavg = list(os.getloadavg())
    except (AttributeError, OSError):
        loadavg = None  # not available on Windows

    fp = {
        "collected_at_unix": time.time(),
        "collected_at_iso": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "python_version": sys.version.split()[0],
        "python_implementation": platform.python_implementation(),
        "platform": platform.platform(),
        "kernel_release": platform.release(),
        "machine": platform.machine(),
        "processor": platform.processor() or None,
        "cpu_count": os.cpu_count(),
        "loadavg_1_5_15": loadavg,
        "numpy_version": np.__version__,
        "ovs_version": _cmd(["ovs-vsctl", "--version"]),
        "mininet_version": _cmd(["mn", "--version"]),
        "iperf3_version": _cmd(["iperf3", "--version"]),
        "git_commit": _cmd(["git", "rev-parse", "HEAD"]),
        "git_branch": _cmd(["git", "rev-parse", "--abbrev-ref", "HEAD"]),
        "config_hash": config_hash(cfg),
    }
    if extra:
        fp.update(extra)
    return fp


# --------------------------------------------------------------------------- the run loop


def run_once(
    policy_name: str,
    scenario: str,
    seed: int,
    cfg=None,
    out_dir: Optional[Path] = None,
    backend_name: str = "sim",
    write: bool = True,
    overrides: Optional[Dict] = None,
):
    """Execute one run and return (csv_path, RunMetrics, rows)."""
    if policy_name not in POLICIES:
        raise KeyError(f"unknown policy {policy_name!r}; known: {sorted(POLICIES)}")
    if backend_name not in BACKENDS:
        raise KeyError(f"unknown backend {backend_name!r}; available now: {BACKENDS}")

    if cfg is None:
        cfg = load_config(scenario=scenario, overrides=overrides)

    trace = build_trace(cfg, seed)
    backend = SimBackend(cfg, trace)

    policy = POLICIES[policy_name](cfg)
    policy.reset(seed)
    guardrail = Guardrail(cfg)
    ctx_builder = ContextBuilder(cfg)

    n_actions = len(cfg.action.embb_levels)
    unrestricted = all_allowed(n_actions)
    interval_s = float(cfg.run.control_interval_s)
    warmup_steps = int(round(float(cfg.run.warmup_s) / interval_s))
    n_steps = min(int(round(float(cfg.run.duration_s) / interval_s)), trace.n_steps)
    step_phase = trace.step_phase_id()

    # Prime the context with an idle sample so the first decision is made against a defined
    # state rather than an all-zero vector implying an impossibly good latency.
    ctx = ctx_builder.update(backend.idle_sample())

    rows: List[Dict] = []
    for step in range(n_steps):
        if backend.done:
            break

        t0 = time.perf_counter()
        proposed = int(policy.select(ctx, unrestricted))
        decision = guardrail.apply(ctx, proposed, policy)
        decision_latency_ms = (time.perf_counter() - t0) * 1000.0

        alloc = allocation_from_level(cfg, decision.applied_action)
        backend.apply_allocation(alloc)

        tel = backend.read_telemetry()
        rb = reward_breakdown(tel, cfg)
        policy.update(ctx, decision.applied_action, rb.total)

        rows.append(
            {
                "step_idx": step,
                "phase_id": int(step_phase[step]) if step < len(step_phase) else -1,
                "warmup": 1 if step < warmup_steps else 0,
                "t_wall": tel.t_wall,
                "t_step": tel.t_step,
                "urllc_rtt_ms_p50": tel.urllc_rtt_ms_p50,
                "urllc_rtt_ms_p95": tel.urllc_rtt_ms_p95,
                "urllc_tx_bps": tel.urllc_tx_bps,
                "embb_goodput_bps": tel.embb_goodput_bps,
                "be_goodput_bps": tel.be_goodput_bps,
                "q0_drops": tel.q0_drops,
                "q1_drops": tel.q1_drops,
                "q2_drops": tel.q2_drops,
                "link_util": tel.link_util,
                "backlog_q0_bytes": tel.backlog_bytes_per_queue[0],
                "backlog_q1_bytes": tel.backlog_bytes_per_queue[1],
                "backlog_q2_bytes": tel.backlog_bytes_per_queue[2],
                "offered_urllc_bps": tel.offered_bps_per_queue[0],
                "offered_embb_bps": tel.offered_bps_per_queue[1],
                "offered_be_bps": tel.offered_bps_per_queue[2],
                **{
                    f"ctx_{name}": float(value)
                    for name, value in zip(FEATURE_NAMES, ctx.vector)
                },
                "proposed_action": decision.proposed_action,
                "applied_action": decision.applied_action,
                "applied_embb_share": alloc.embb_share,
                "guardrail_reason": decision.reason,
                "guardrail_intervened": int(decision.intervened),
                "guardrail_escalation_source": decision.escalation_source,
                "decision_latency_ms": decision_latency_ms,
                "reward": rb.total,
                "reward_tput": rb.tput_term,
                "reward_be": rb.be_term,
                "reward_sla": rb.sla_term,
                "reward_viol": rb.viol_term,
                "reward_drop": rb.drop_term,
                "reward_rtt_used_ms": rb.rtt_used_ms,
                "sla_violated": int(rb.violated),
            }
        )

        # Build the context for the NEXT decision from the telemetry we just observed.
        ctx = ctx_builder.update(tel)

    run_id = f"{policy_name}_{scenario}_{seed}"
    out_dir = Path(out_dir) if out_dir else REPO_ROOT / str(cfg.experiment.results_dir)
    csv_path = out_dir / f"{run_id}.csv"

    if write:
        out_dir.mkdir(parents=True, exist_ok=True)
        with open(csv_path, "w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=RUN_COLUMNS)
            writer.writeheader()
            writer.writerows(rows)

        env_dir = out_dir / run_id
        env_dir.mkdir(parents=True, exist_ok=True)
        with open(env_dir / "env.json", "w", encoding="utf-8") as fh:
            json.dump(
                environment_fingerprint(
                    cfg,
                    extra={
                        "run_id": run_id,
                        "policy": policy_name,
                        "scenario": scenario,
                        "seed": seed,
                        "backend": backend_name,
                        "n_steps_logged": len(rows),
                        "guardrail_reason_counts": guardrail.reason_counts,
                    },
                ),
                fh,
                indent=2,
                sort_keys=True,
            )

        # Validate what was actually written to disk, not the in-memory rows. A schema bug in
        # the writer is exactly the kind of thing an in-memory check would miss.
        df = validate_run_csv(csv_path, n_actions=n_actions)
    else:
        import pandas as pd

        df = pd.DataFrame(rows, columns=RUN_COLUMNS)

    metrics = compute_run_metrics(df, policy_name, scenario, seed)
    return csv_path, metrics, df


# --------------------------------------------------------------------------- cli


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Run one (policy, scenario, seed) experiment.")
    ap.add_argument("--policy", required=True, choices=sorted(POLICIES))
    ap.add_argument("--scenario", required=True, choices=list_scenarios())
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--backend", default="sim", choices=BACKENDS)
    ap.add_argument("--out-dir", default=None)
    ap.add_argument(
        "--set",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="dotted config override, e.g. --set run.duration_s=60",
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

    csv_path, m, _ = run_once(
        policy_name=args.policy,
        scenario=args.scenario,
        seed=args.seed,
        out_dir=Path(args.out_dir) if args.out_dir else None,
        backend_name=args.backend,
        overrides=overrides or None,
    )

    print(f"wrote {csv_path}")
    print(f"      {csv_path.parent / csv_path.stem / 'env.json'}")
    print(f"schema validated, {m.n_steps} post-warmup steps")
    print("")
    print(f"  policy                       {m.policy}")
    print(f"  scenario                     {m.scenario}  seed {m.seed}")
    print(f"  URLLC RTT p50 / p95 / p99    {m.urllc_rtt_p50:.2f} / {m.urllc_rtt_p95:.2f} / "
          f"{m.urllc_rtt_p99:.2f} ms")
    print(f"  SLA violation rate           {m.sla_violation_rate * 100:.2f} % of steps")
    print(f"  eMBB goodput mean            {m.embb_goodput_mbps:.3f} Mbps")
    print(f"  BE goodput mean              {m.be_goodput_mbps:.3f} Mbps")
    print(f"  total drops                  {m.total_drops} packets")
    print(f"  guardrail intervention rate  {m.guardrail_intervention_rate * 100:.2f} % of steps")
    print(f"  decision latency mean / p99  {m.decision_latency_mean_ms:.4f} / "
          f"{m.decision_latency_p99_ms:.4f} ms")
    print(f"  mean reward                  {m.mean_reward:.4f}")
    print(f"  link utilisation mean        {m.link_util_mean * 100:.2f} %")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
