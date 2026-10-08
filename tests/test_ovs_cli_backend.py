"""Tests for net/ovs_cli_backend.py and experiments/run_policies_ovs.py.

Mininet-free. The backend's side effects (ovs-vsctl, tc, the hosts' processes, the clock) are
replaced by fakes with known behaviour, so the tests can check exactly what it computes from them:
that rates are differenced over the real elapsed time, that latency comes only from replies inside
the interval, that a missing reply is carried forward and counted rather than invented, that the
actuator is called only on a change, and that a jittered trace is refused.
"""

from __future__ import annotations

import numpy as np
import pytest

import net.ovs_cli_backend as ob
from experiments.run_experiment import run_once
from experiments.run_policies_ovs import (
    DEFAULT_POLICIES,
    build_policy,
    classify_loop,
    live_configs,
)
from net.backend import allocation_from_level
from traffic.generator import payload_bps_for_l2
from traffic.traces import build_trace

T0 = 1_000_000.0


# --------------------------------------------------------------------------- pure functions


def test_rate_segments_split_a_phase_trace_into_constant_stretches():
    rates = [2e6] * 40 + [10e6] * 35 + [2e6] * 25
    segs = ob.rate_segments(rates, 1.0)
    assert segs == [
        {"start_step": 0, "n_steps": 40, "bps": 2e6},
        {"start_step": 40, "n_steps": 35, "bps": 10e6},
        {"start_step": 75, "n_steps": 25, "bps": 2e6},
    ]


def test_rate_segments_refuse_a_jittered_trace():
    with pytest.raises(ValueError, match="jitter_cv"):
        ob.rate_segments(np.linspace(1e6, 2e6, 50), 1.0)


def test_interval_rtt_uses_only_replies_inside_the_half_open_interval():
    samples = [(9.99, 1, 50.0), (10.0, 2, 1.0), (10.5, 3, 2.0), (10.9, 4, 3.0), (11.0, 5, 99.0)]
    p50, p95, n = ob.interval_rtt(samples, 10.0, 11.0)
    assert n == 3
    assert p50 == pytest.approx(2.0)
    assert p95 == pytest.approx(np.percentile([1.0, 2.0, 3.0], 95))
    assert ob.interval_rtt(samples, 20.0, 21.0) == (None, None, 0)


def test_counter_delta_rejects_missing_and_reset_counters():
    assert ob.counter_delta(100, 250) == 150
    assert ob.counter_delta(None, 250) is None
    assert ob.counter_delta(300, 250) is None


# --------------------------------------------------------------------------- fakes


class FakeClock:
    def __init__(self, t=T0):
        self.t = t

    def __call__(self):
        return self.t

    def sleep(self, d):
        self.t += max(0.0, d)


class FakeProc:
    def __init__(self, args):
        self.args = args
        self.done = False

    def poll(self):
        return 0 if self.done else None

    def terminate(self):
        self.done = True

    def kill(self):
        self.done = True

    def wait(self, timeout=None):
        return 0


class FakeHost:
    def __init__(self, name, ip):
        self.name, self.ip = name, ip
        self.popened, self.cmds = [], []

    def IP(self):
        return self.ip

    def popen(self, args, **_):
        p = FakeProc(args)
        self.popened.append(p)
        return p

    def cmd(self, c):
        self.cmds.append(c)
        return ""


class FakeNet(dict):
    def __init__(self):
        super().__init__({f"h{i}": FakeHost(f"h{i}", f"10.0.0.{i}") for i in range(1, 7)})


class FakeSwitch:
    """tc counters that rise at fixed per-queue rates; eMBB at whatever ceiling is programmed."""

    def __init__(self, clock, urllc_bps=1e6, be_bps=1e6, embb_drop_pps=10):
        self.clock = clock
        self.rates = {"1:1": urllc_bps, "1:2": None, "1:3": be_bps}
        self.embb_ceil = None
        self.embb_drop_pps = embb_drop_pps
        self.sent = {"1:1": 0.0, "1:2": 0.0, "1:3": 0.0}
        self.drops = {"1:1": 0.0, "1:2": 0.0, "1:3": 0.0}
        self.t_last = clock()
        self.set_calls = []

    def apply_qos(self, iface, cfg, level_index, *a, **k):
        alloc = allocation_from_level(cfg, level_index)
        self.embb_ceil = int(alloc.embb_cap_bps)
        return {"urllc": int(alloc.urllc_cap_bps), "embb": self.embb_ceil,
                "be": int(alloc.be_cap_bps)}

    def set_queue_max_rate(self, q, bps):
        assert q == 1
        self.set_calls.append(int(bps))
        self.embb_ceil = int(bps)

    def read_tc_classes(self, iface):
        now = self.clock()
        dt = now - self.t_last
        self.t_last = now
        for h in self.sent:
            rate = self.embb_ceil if h == "1:2" else self.rates[h]
            self.sent[h] += rate * dt / 8.0
        self.drops["1:2"] += self.embb_drop_pps * dt
        return {
            h: {"sent_bytes": int(self.sent[h]), "dropped_pkts": int(self.drops[h]),
                "backlog_bytes": 1442 if h == "1:1" else 62006,
                "ceil": f"{self.embb_ceil / 1000:g}Kbit" if h == "1:2" else "10Mbit",
                "rate": "1Mbit"}
            for h in self.sent
        }


@pytest.fixture
def rig(monkeypatch):
    clock = FakeClock()
    switch = FakeSwitch(clock)
    monkeypatch.setattr(ob, "apply_qos", switch.apply_qos)
    monkeypatch.setattr(ob, "set_queue_max_rate", switch.set_queue_max_rate)
    monkeypatch.setattr(ob, "read_tc_classes", switch.read_tc_classes)
    _, cfg = live_configs("burst", duration_s=80.0)
    net = FakeNet()
    backend = ob.OvsCliBackend(cfg, build_trace(cfg, 0), net, "s1-eth4",
                               clock=clock, sleep=clock.sleep)
    return backend, clock, switch, net, cfg


def _write_pings(backend, t_lo, t_hi, rtt_ms, every=0.05, seq0=1):
    lines, t, seq = [], t_lo, seq0
    while t < t_hi:
        lines.append(f"[{t:.6f}] 64 bytes from 10.0.0.4: icmp_seq={seq} ttl=64 time={rtt_ms} ms")
        t += every
        seq += 1
    with open(backend.ping_path, "a", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")
    return seq


# --------------------------------------------------------------------------- the backend


def test_reset_starts_ping_and_one_client_per_active_slice_at_the_requested_payload_rate(rig):
    backend, clock, switch, net, cfg = rig
    backend.reset()
    ping = [p for p in net["h1"].popened if "ping -D" in p.args[2]]
    assert len(ping) == 1 and "-i 0.05" in ping[0].args[2]
    first_embb = net["h2"].popened[0].args[2]
    # burst.yaml opens with eMBB at 2.0 Mbps L2 for 40 s, on the slice's primary port.
    assert f"-b {int(round(payload_bps_for_l2(2.0e6)))}" in first_embb
    assert "-p 5202" in first_embb and "-t 40" in first_embb
    assert len(net["h3"].popened) == 1  # Best Effort is constant: one client for the whole run


def test_telemetry_is_differenced_over_elapsed_time_and_latency_is_from_the_interval(rig):
    backend, clock, switch, net, cfg = rig
    backend.reset()
    t0 = clock()
    _write_pings(backend, t0, t0 + 1.0, 2.5)
    tel = backend.read_telemetry()
    assert clock() == pytest.approx(t0 + 1.0)
    assert tel.urllc_tx_bps == pytest.approx(1e6, rel=1e-3)
    assert tel.embb_goodput_bps == pytest.approx(switch.embb_ceil, rel=1e-3)
    assert tel.be_goodput_bps == pytest.approx(1e6, rel=1e-3)
    assert tel.link_util == pytest.approx((2e6 + switch.embb_ceil) / 10e6, rel=1e-3)
    assert tel.q1_drops == 10
    assert tel.urllc_rtt_ms_p50 == pytest.approx(2.5)
    assert tel.backlog_bytes_per_queue == (1442.0, 62006.0, 62006.0)
    assert tel.t_step == pytest.approx(1.0)
    assert tel.offered_bps_per_queue == pytest.approx((3e6, 2e6, 4e6))
    log = backend.step_log[0]
    assert log["ceil_ok"] and log["counter_ok"] and log["n_rtt"] == 20


def test_a_late_caller_is_not_made_later_and_the_lateness_is_logged(rig):
    backend, clock, switch, net, cfg = rig
    backend.reset()
    clock.t += 1.3  # the caller overran the boundary by 0.3 s
    backend.read_telemetry()
    assert backend.step_log[0]["lateness_s"] == pytest.approx(0.3)
    assert backend.step_log[0]["dt_s"] == pytest.approx(1.3)


def test_an_interval_without_replies_carries_the_last_rtt_forward_and_is_counted(rig):
    backend, clock, switch, net, cfg = rig
    backend.reset()
    t0 = clock()
    _write_pings(backend, t0, t0 + 1.0, 3.2)
    backend.read_telemetry()
    tel = backend.read_telemetry()  # no replies written for the second interval
    assert tel.urllc_rtt_ms_p50 == pytest.approx(3.2)
    assert backend.step_log[1]["rtt_missing"] is True
    assert backend.step_log[0]["rtt_missing"] is False


def test_actuator_is_called_only_when_the_embb_ceiling_changes(rig):
    backend, clock, switch, net, cfg = rig
    backend.reset()
    initial = int(cfg.action.initial_index)
    backend.apply_allocation(allocation_from_level(cfg, initial))
    assert switch.set_calls == []
    backend.apply_allocation(allocation_from_level(cfg, 3))
    backend.apply_allocation(allocation_from_level(cfg, 3))
    assert switch.set_calls == [int(allocation_from_level(cfg, 3).embb_cap_bps)]
    backend.read_telemetry()
    assert backend.step_log[0]["ceil_ok"]


def test_a_ceiling_tc_does_not_report_is_flagged(rig):
    backend, clock, switch, net, cfg = rig
    backend.reset()
    backend._programmed_embb_bps = 6_500_000  # programmed, but the switch never received it
    backend.read_telemetry()
    assert backend.step_log[0]["ceil_ok"] is False


def test_rate_changes_alternate_server_ports_at_the_phase_boundary(rig):
    backend, clock, switch, net, cfg = rig
    backend.reset()
    for _ in range(40):
        backend.read_telemetry()
    clients = [p.args[2] for p in net["h2"].popened]
    assert len(clients) == 2
    assert "-p 5212" in clients[1] and "-t 35" in clients[1]
    assert f"-b {int(round(payload_bps_for_l2(10.0e6)))}" in clients[1]


def test_link_util_is_capped_at_one_and_the_raw_value_kept(rig):
    backend, clock, switch, net, cfg = rig
    switch.rates["1:1"], switch.rates["1:3"] = 4e6, 4.5e6  # 8.5 + 3.5 Mbps > capacity
    backend.reset()
    tel = backend.read_telemetry()
    assert tel.link_util == 1.0
    assert backend.step_log[0]["link_util_raw"] == pytest.approx(1.2, rel=1e-3)


def test_unreadable_tc_stops_the_run_rather_than_reporting_zeros(rig, monkeypatch):
    backend, clock, switch, net, cfg = rig
    backend.reset()
    monkeypatch.setattr(ob, "read_tc_classes", lambda iface: {})
    with pytest.raises(RuntimeError, match="unreadable"):
        backend.read_telemetry()


def test_full_run_through_the_unmodified_runner(rig, tmp_path):
    """The point of stage 6: run_once drives this backend exactly as it drives the simulator."""
    backend, clock, switch, net, cfg = rig
    backend.reset()
    policy, learn = build_policy("threshold", cfg, cfg, "burst", 0)
    _, m, df = run_once("threshold", "burst", 0, cfg=cfg, out_dir=tmp_path,
                        backend_name="ovs_cli", policy=policy, learn=learn, backend=backend)
    backend.close()
    assert len(df) == backend.n_steps == 80
    assert backend.done
    assert all(p.done for h in net.values() for p in h.popened)
    assert (tmp_path / "threshold_burst_0.csv").exists()
    assert len(backend.segment_reports) == len(backend._segment_files)


# --------------------------------------------------------------------------- the experiment


def test_testbed_overlay_changes_only_the_slo_and_eval_switches_jitter_off():
    train, ev = live_configs("burst")
    for cfg in (train, ev):
        assert cfg.sla.hard_ms == 4.0 and cfg.sla.target_ms == 3.5
        assert cfg.guardrail.warn_ms == 3.0
        assert cfg.reward.rtt_statistic == "urllc_rtt_ms_p50"
        assert cfg.context.fast_rtt_statistic == "urllc_rtt_ms_p50"
        assert cfg.link.base_rtt_ms == 0.093
        assert cfg.sim.excess_sharing == "demand_proportional"
    assert dict(train.scenario.jitter_cv.as_dict())["embb"] > 0
    assert all(v == 0.0 for v in ev.scenario.jitter_cv.as_dict().values())


def test_default_config_still_uses_p95_for_the_fast_path():
    from config_loader import load_config

    assert load_config("burst").context.fast_rtt_statistic == "urllc_rtt_ms_p95"


def test_pretrained_policy_is_frozen_and_the_others_learn():
    train, ev = live_configs("burst", duration_s=60.0)
    train_short = live_configs("burst", duration_s=None)[0]
    policy, learn = build_policy("static_safe", train_short, ev, "burst", 0)
    assert learn is True
    assert set(DEFAULT_POLICIES) == {"static_safe", "static_equal", "threshold", "linucb_pretrained"}
    with pytest.raises(KeyError):
        build_policy("oracle", train, ev, "burst", 0)


def _steps(n, **over):
    base = {"dt_s": 1.0, "n_rtt": 18, "ceil_ok": True, "counter_ok": True, "lateness_s": 0.0,
            "rtt_missing": False}
    return [{**base, **over} for _ in range(n)]


def test_loop_rule_passes_a_clean_run_and_names_each_failed_check():
    seg = [{"parse_ok": True, "sender_bps": payload_bps_for_l2(2e6), "requested_l2_bps": 2e6}]
    ok = classify_loop(_steps(300), seg, 30, 1.0)
    assert ok["label"] == "LOOP_OK" and ok["failed_checks"] == []

    late = _steps(30) + _steps(255) + _steps(15, dt_s=1.2)
    assert classify_loop(late, seg, 30, 1.0)["failed_checks"] == ["on_time"]

    thin = _steps(296) + _steps(4, n_rtt=2)
    assert classify_loop(thin, seg, 0, 1.0)["failed_checks"] == ["rtt_measured"]

    slow_sender = [{"parse_ok": True, "sender_bps": payload_bps_for_l2(1.8e6),
                    "requested_l2_bps": 2e6}]
    assert classify_loop(_steps(300), slow_sender, 30, 1.0)["failed_checks"] == \
        ["senders_on_target"]


def test_loop_rule_ignores_warm_up_steps():
    seg = []
    log = _steps(30, dt_s=3.0) + _steps(270)
    assert classify_loop(log, seg, 30, 1.0)["label"] == "LOOP_OK"
