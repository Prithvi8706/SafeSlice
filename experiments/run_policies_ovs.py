"""Stage 6 of docs/PLAN_TESTBED.md: the same policies, unmodified, driving the real switch.

Four policies, one run each, through `experiments.run_experiment.run_once`: the same runner, the
same guardrail, the same reward and the same CSV schema as the simulator study. The only things
that differ are the backend (net/ovs_cli_backend.py) and the testbed SLO overlay
(config/testbed.yaml, whose derivation is written in the file).

The same script runs the identical protocol in the simulator with `--backend sim`. That run is made
FIRST and recorded in docs/PLAN_TESTBED.md section 2.13 as the prediction the testbed run is read
against, before the testbed run exists.

LOAD
config/scenarios/burst.yaml with its jitter switched off, so offered load is constant within each
phase and iperf3 can follow it (docs/PLAN_TESTBED.md section 2.4).

THE PRE-TRAINED POLICY
`linucb_pretrained` is trained in the SIMULATOR, as in the simulator study: on the held-out training
seeds, with the scenario's jitter on, under the testbed overlay so it learns the same reward it is
scored on. It is then frozen and evaluated on the jitter-free load, on whichever backend was
chosen. On the testbed that is a sim-to-real transfer, and it is reported as one.

Run as root inside the WSL checkout for the testbed; anywhere for the simulator:

    python experiments/run_policies_ovs.py --backend sim               # the prediction, ~1 min
    python3 experiments/run_policies_ovs.py --backend ovs_cli --quick  # 4 x 90 s smoke test, ~7 min
    python3 experiments/run_policies_ovs.py --backend ovs_cli          # 4 x 300 s, ~22 min
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import random
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List, Optional

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import yaml  # noqa: E402

from config_loader import config_hash, load_config  # noqa: E402
from experiments.run_experiment import POLICIES, PRETRAINED, pretrain, run_once  # noqa: E402
from traffic.traces import build_trace  # noqa: E402

TESTBED_OVERLAY = REPO_ROOT / "config" / "testbed.yaml"
DEFAULT_POLICIES = ("static_safe", "static_equal", "threshold", "linucb_pretrained")

#: Loop-health rule, fixed in docs/PLAN_TESTBED.md section 2.13 before the testbed run.
ON_TIME_TOLERANCE = 0.10       # an interval counts as on time within 10 % of its nominal length
MIN_RTT_SAMPLES = 5            # replies needed for a step's latency to count as measured
SENDER_TOLERANCE = 0.05        # iperf3 sender within 5 % of the requested rate
LOOP_THRESHOLDS = {"on_time": 0.95, "rtt_measured": 0.99, "ceil_confirmed": 0.99,
                   "counters_ok": 0.99}


# --------------------------------------------------------------------------- pure functions


def live_configs(scenario: str, duration_s: Optional[float] = None):
    """(train_cfg, eval_cfg). Both carry the testbed overlay; only eval switches jitter off."""
    with open(TESTBED_OVERLAY, "r", encoding="utf-8") as fh:
        overlay = yaml.safe_load(fh)
    train_cfg = load_config(scenario=scenario, overrides=copy.deepcopy(overlay))
    eval_overrides = copy.deepcopy(overlay)
    eval_overrides["scenario.jitter_cv"] = {"urllc": 0.0, "embb": 0.0, "be": 0.0}
    if duration_s is not None:
        eval_overrides["run.duration_s"] = float(duration_s)
    eval_cfg = load_config(scenario=scenario, overrides=eval_overrides)
    return train_cfg, eval_cfg


def build_policy(name: str, train_cfg, eval_cfg, scenario: str, seed: int):
    """Return (policy, learn). Pre-trained policies are trained in the simulator and frozen."""
    if name in PRETRAINED:
        policy = POLICIES[PRETRAINED[name]](train_cfg)
        policy.reset(seed)
        pretrain(policy, cfg=train_cfg, scenario=scenario,
                 seeds=list(train_cfg.experiment.train_seeds), backend_name="sim")
        return policy, False
    if name not in POLICIES:
        raise KeyError(f"{name!r} cannot run live; choose from {sorted(POLICIES) + sorted(PRETRAINED)}")
    policy = POLICIES[name](eval_cfg)
    policy.reset(seed)
    return policy, True


def classify_loop(step_log: List[Dict], segment_reports: List[Dict], warmup_steps: int,
                  interval_s: float) -> Dict:
    """Apply the section 2.13 loop-health rule to one testbed run. Pure, unit tested."""
    from traffic.generator import l2_bps_from_payload

    post = step_log[warmup_steps:]
    n = len(post)

    def share(pred):
        return sum(1 for s in post if pred(s)) / n if n else 0.0

    shares = {
        "on_time": share(lambda s: abs(s["dt_s"] - interval_s) <= ON_TIME_TOLERANCE * interval_s),
        "rtt_measured": share(lambda s: s["n_rtt"] >= MIN_RTT_SAMPLES),
        "ceil_confirmed": share(lambda s: s["ceil_ok"]),
        "counters_ok": share(lambda s: s["counter_ok"]),
    }
    senders = []
    for seg in segment_reports:
        ok = bool(seg.get("parse_ok")) and seg.get("sender_bps") is not None and abs(
            l2_bps_from_payload(seg["sender_bps"]) / seg["requested_l2_bps"] - 1.0
        ) <= SENDER_TOLERANCE
        senders.append(ok)
    failed = [k for k, v in shares.items() if v < LOOP_THRESHOLDS[k]]
    if not all(senders):
        failed.append("senders_on_target")
    return {
        "label": "LOOP_OK" if not failed else "LOOP_DEGRADED",
        "failed_checks": failed,
        "post_warmup_steps": n,
        **{f"share_{k}": v for k, v in shares.items()},
        "segments": len(senders),
        "segments_sender_on_target": sum(senders),
        "max_lateness_s": max((s["lateness_s"] for s in post), default=None),
        "steps_rtt_missing": sum(1 for s in post if s["rtt_missing"]),
    }


# --------------------------------------------------------------------------- execution


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Run the policies live on OVS, or the same in sim.")
    ap.add_argument("--backend", choices=("sim", "ovs_cli"), required=True)
    ap.add_argument("--policies", nargs="+", default=list(DEFAULT_POLICIES))
    ap.add_argument("--scenario", default="burst")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--quick", action="store_true", help="90 s per policy instead of 300")
    args = ap.parse_args(argv)

    if args.backend == "ovs_cli" and os.geteuid() != 0:
        print("--backend ovs_cli must run as root (Mininet needs it).", file=sys.stderr)
        return 2

    duration = 90.0 if args.quick else None
    train_cfg, eval_cfg = live_configs(args.scenario, duration)
    tag = f"{args.backend}{'_quick' if args.quick else ''}"
    out_dir = REPO_ROOT / "results" / "testbed_policies" / tag
    summary_path = REPO_ROOT / "results" / "summary" / f"policy_live_{tag}.json"
    order = list(args.policies)
    random.Random(args.seed).shuffle(order)
    interval_s = float(eval_cfg.run.control_interval_s)
    warmup_steps = int(round(float(eval_cfg.run.warmup_s) / interval_s))

    summary: Dict = {
        "backend": args.backend, "scenario": args.scenario, "seed": args.seed,
        "duration_s": float(eval_cfg.run.duration_s), "warmup_s": float(eval_cfg.run.warmup_s),
        "order": order, "rule": "docs/PLAN_TESTBED.md section 2.13",
        "testbed_overlay": yaml.safe_load(TESTBED_OVERLAY.read_text(encoding="utf-8")),
        "config_hash_eval": config_hash(eval_cfg), "config_hash_train": config_hash(train_cfg),
        "runs": {},
    }
    print(f"[live] backend {args.backend}, order {order}, "
          f"{eval_cfg.run.duration_s:g} s each, results to {out_dir}")

    net = iface = None
    try:
        if args.backend == "ovs_cli":
            from mininet.log import setLogLevel

            from net.topology.slice_topo import (
                SLICE_ALT_PORTS, bottleneck_iface, build_network, install_flows,
            )
            from traffic.generator import start_servers

            setLogLevel("warning")
            net = build_network(eval_cfg)
            iface = bottleneck_iface(net)
            install_flows("s1")
            start_servers(net)
            start_servers(net, SLICE_ALT_PORTS)

        for i, name in enumerate(order):
            policy, learn = build_policy(name, train_cfg, eval_cfg, args.scenario, args.seed)
            backend = None
            if args.backend == "ovs_cli":
                from net.ovs_cli_backend import OvsCliBackend

                backend = OvsCliBackend(eval_cfg, build_trace(eval_cfg, args.seed), net, iface)
                backend.reset()
            try:
                csv_path, m, _ = run_once(
                    policy_name=name, scenario=args.scenario, seed=args.seed, cfg=eval_cfg,
                    out_dir=out_dir, backend_name=args.backend, policy=policy, learn=learn,
                    backend=backend,
                )
            finally:
                if backend is not None:
                    backend.close()
            run = {"csv": str(csv_path.relative_to(REPO_ROOT)), "metrics": asdict(m)}
            if backend is not None:
                run["loop"] = classify_loop(backend.step_log, backend.segment_reports,
                                            warmup_steps, interval_s)
                run["segment_reports"] = backend.segment_reports
                (out_dir / f"{csv_path.stem}_steps.json").write_text(
                    json.dumps(backend.step_log, indent=1), encoding="utf-8")
            summary["runs"][name] = run
            loop = f"  {run['loop']['label']}" if "loop" in run else ""
            print(f"[live] {i + 1}/{len(order)} {name:18s} reward {m.mean_reward:+.4f}  "
                  f"viol {m.sla_violation_rate * 100:5.2f} %  intervene "
                  f"{m.guardrail_intervention_rate * 100:5.2f} %  eMBB {m.embb_goodput_mbps:.3f}  "
                  f"BE {m.be_goodput_mbps:.3f}  URLLC p50 {m.urllc_rtt_p50:.2f} ms{loop}",
                  flush=True)
            summary_path.parent.mkdir(parents=True, exist_ok=True)
            summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
            if args.backend == "ovs_cli" and i + 1 < len(order):
                time.sleep(5.0)
    finally:
        if net is not None:
            from net.topology.slice_topo import SLICE_ALT_PORTS, clear_qos
            from traffic.generator import stop_servers

            for ports in (None, SLICE_ALT_PORTS):
                try:
                    stop_servers(net, ports)
                except Exception:  # noqa: BLE001
                    pass
            try:
                clear_qos(iface)
            except Exception:  # noqa: BLE001
                clear_qos()
            net.stop()

    print(f"[live] wrote {summary_path.relative_to(REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
