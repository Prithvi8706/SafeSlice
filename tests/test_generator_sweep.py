"""Tests for traffic/generator.py and experiments/sweep_levels_ovs.py.

Everything here is Mininet-free. The testbed execution cannot be tested on the development machine,
but every number the sweep reports passes through the functions below, so they are where a wrong
metric would come from: a window that includes idle time, a percentile taken over the wrong series,
an unshaped calibration that passes because the sender was quietly throttled.

Synthetic counter series are constructed with known rates so the expected answers are exact.
"""

from __future__ import annotations

import pytest

from config_loader import load_config
from experiments.sweep_levels_ovs import (
    AGG_KEYS,
    aggregate_levels,
    burst_phase_loads_l2,
    calibration_verdict,
    run_order,
    summarize_run,
)
from traffic.generator import (
    FRAME_BYTES,
    SLICES,
    counter_window_delta,
    l2_bps_from_payload,
    payload_bps_for_l2,
    per_window_rtt,
)

T0 = 1_000_000.0


# --------------------------------------------------------------------------- units


def test_frame_is_payload_plus_udp_ip_ethernet_headers():
    assert FRAME_BYTES == 1400 + 8 + 20 + 14


def test_l2_to_payload_conversion_matches_what_the_testbed_measured():
    """Stage 1b measured 3.399 Mbps of payload through a 3.5 Mbps HTB cap. The conversion used to
    request rates must predict that, or every offered load in the sweep is off by the same factor."""
    assert payload_bps_for_l2(3.5e6) / 1e6 == pytest.approx(3.399, abs=0.002)


def test_unit_conversions_round_trip():
    for bps in (1.0e6, 3.0e6, 10.0e6):
        assert l2_bps_from_payload(payload_bps_for_l2(bps)) == pytest.approx(bps)


# --------------------------------------------------------------------------- counter windows


def _counter(rate_per_s, start, end, step=1.0, begin_value=0):
    """Cumulative counter rising at rate_per_s between start and end, flat outside it."""
    out, t = [], start - 3.0
    while t <= end + 3.0:
        active = min(max(t, start), end) - start
        out.append((t, int(begin_value + rate_per_s * active)))
        t += step
    return out


def test_window_delta_uses_only_samples_inside_the_window():
    series = _counter(1000, T0, T0 + 30)
    d = counter_window_delta(series, T0 + 10, T0 + 29)
    assert d["delta"] / d["window_s"] == pytest.approx(1000)


def test_window_delta_is_not_diluted_by_the_idle_tail_after_the_flow():
    """The stage 1b/1c `tc_l2` bug: a window running past the end of the flow reads low."""
    series = _counter(1000, T0, T0 + 30)
    bounded = counter_window_delta(series, T0 + 10, T0 + 29)
    overrun = counter_window_delta(series, T0 + 10, T0 + 33)
    assert bounded["delta"] / bounded["window_s"] == pytest.approx(1000)
    assert overrun["delta"] / overrun["window_s"] < 0.9 * 1000


@pytest.mark.parametrize(
    "series",
    [[], [(T0, 5)], [(T0, 5), (T0, 9)], [(T0, 900), (T0 + 1, 100)]],
)
def test_window_delta_refuses_meaningless_input(series):
    assert counter_window_delta(series, T0 - 10, T0 + 10) is None


# --------------------------------------------------------------------------- latency windows


def _pings(rtt_fn, start, end, interval=0.05):
    out, t, seq = [], start, 0
    while t < end:
        out.append((t, seq, float(rtt_fn(t))))
        t += interval
        seq += 1
    return out


def test_per_window_and_pooled_statistics_are_different_quantities():
    # Mostly 1 ms, with one 50 ms spike in each second. Pooled p95 barely notices it; a per-window
    # p95 over 20 samples is pulled up towards it every second.
    def rtt(t):
        return 50.0 if (t - T0) % 1.0 < 0.05 else 1.0

    stats = per_window_rtt(_pings(rtt, T0, T0 + 20), T0, T0 + 20)
    assert stats["pooled_p50_ms"] == pytest.approx(1.0)
    assert stats["n_windows_used"] == 20
    assert stats["p95_of_window_p95_ms"] > stats["pooled_p95_ms"]


def test_thin_windows_are_excluded_and_counted_not_silently_used():
    samples = _pings(lambda t: 1.0, T0, T0 + 10)
    samples += [(T0 + 10.2, 999, 80.0), (T0 + 10.4, 1000, 80.0)]   # a 2-sample second
    stats = per_window_rtt(samples, T0, T0 + 11, min_samples=5)
    assert stats["n_windows_thin"] == 1
    assert stats["p95_of_window_p95_ms"] == pytest.approx(1.0)
    assert stats["pooled_max_ms"] == pytest.approx(80.0), "pooled still sees every reply"


def test_no_samples_gives_nones_not_zeros():
    stats = per_window_rtt([], T0, T0 + 10)
    assert stats["n_samples"] == 0
    assert stats["p95_of_window_p95_ms"] is None
    assert stats["pooled_p50_ms"] is None


# --------------------------------------------------------------------------- sweep design


def test_offered_load_is_the_burst_phase_the_simulator_table_used():
    """docs/DESIGN.md section 3's fixed-level table was built on burst.yaml's burst phase."""
    loads = burst_phase_loads_l2(load_config(scenario="burst"))
    assert loads == {"urllc": 3.0e6, "embb": 10.0e6, "be": 4.0e6}


def test_run_order_covers_every_level_once_per_repeat():
    order = run_order(n_levels=5, repeats=3, seed=0)
    assert len(order) == 15
    assert [o["position"] for o in order] == list(range(15))
    for r in range(3):
        assert sorted(o["level_index"] for o in order if o["repeat"] == r) == [0, 1, 2, 3, 4]


def test_run_order_is_reproducible_and_actually_shuffled():
    assert run_order(5, 3, seed=7) == run_order(5, 3, seed=7)
    assert run_order(5, 3, seed=7) != run_order(5, 3, seed=8)
    ascending = [0, 1, 2, 3, 4] * 3
    assert [o["level_index"] for o in run_order(5, 3, seed=0)] != ascending


# --------------------------------------------------------------------------- one run's metrics

DURATION, WARMUP, CAPACITY = 30.0, 5.0, 10.0e6


def _tc(rates_bps, drops_per_s=None, backlog=None):
    drops_per_s = drops_per_s or {}
    backlog = backlog or {}
    out = {}
    for s in SLICES:
        out[s] = {
            "sent_bytes": _counter(rates_bps[s] / 8.0, T0, T0 + DURATION),
            "dropped_pkts": _counter(drops_per_s.get(s, 0), T0, T0 + DURATION),
            "backlog_bytes": [(t, backlog.get(s, 0)) for t, _ in _counter(1, T0, T0 + DURATION)],
        }
    return out


def _iperf(payload_bps):
    return {"parse_ok": True, "sender_bps": payload_bps, "receiver_bps": payload_bps}


def _load(offered_l2, sender_scale=1.0):
    return {
        "iperf": {s: _iperf(payload_bps_for_l2(offered_l2[s]) * sender_scale) for s in SLICES},
        "ping_samples": _pings(lambda t: 2.0, T0, T0 + DURATION),
        "ping_summary": {"transmitted": 600, "received": 600, "loss_pct": 0.0},
    }


OFFERED = {"urllc": 2.0e6, "embb": 3.0e6, "be": 3.0e6}


def test_summarize_recovers_known_goodput_exactly_in_l2_units():
    row = summarize_run(_tc(OFFERED), _load(OFFERED), T0, DURATION, WARMUP, CAPACITY, OFFERED, 0.05)
    assert row["valid"]
    for s in SLICES:
        assert row[f"{s}_goodput_l2_mbps"] == pytest.approx(OFFERED[s] / 1e6, rel=1e-3)
        assert row[f"{s}_iperf_sender_l2_mbps"] == pytest.approx(OFFERED[s] / 1e6, rel=1e-6)
    assert row["link_util"] == pytest.approx(0.8, rel=1e-3)
    assert row["urllc_rtt_p50_ms"] == pytest.approx(2.0)


def test_summarize_counts_drops_only_inside_the_window():
    row = summarize_run(_tc(OFFERED, drops_per_s={"embb": 100}), _load(OFFERED),
                        T0, DURATION, WARMUP, CAPACITY, OFFERED, 0.05)
    window = (DURATION - 1.0) - WARMUP
    assert row["embb_drops"] == pytest.approx(100 * window, abs=100)
    assert row["urllc_drops"] == 0


def test_ping_delivery_is_reported_so_latency_is_never_quoted_without_it():
    load = _load(OFFERED)
    load["ping_samples"] = load["ping_samples"][::2]   # half the probes lost in q0
    row = summarize_run(_tc(OFFERED), load, T0, DURATION, WARMUP, CAPACITY, OFFERED, 0.05)
    assert row["urllc_ping_delivered_pct"] == pytest.approx(50.0, abs=3.0)


def test_a_missing_counter_invalidates_the_run_rather_than_reading_zero():
    tc = _tc(OFFERED)
    tc["embb"]["sent_bytes"] = []
    row = summarize_run(tc, _load(OFFERED), T0, DURATION, WARMUP, CAPACITY, OFFERED, 0.05)
    assert row["embb_goodput_l2_mbps"] is None
    assert row["link_util"] is None
    assert not row["valid"]


# --------------------------------------------------------------------------- calibration gate


def _cal_row(tc_rates=None, sender_scale=1.0, drops=None):
    rates = tc_rates or OFFERED
    return summarize_run(_tc(rates, drops_per_s=drops), _load(OFFERED, sender_scale),
                         T0, DURATION, WARMUP, CAPACITY, OFFERED, 0.05)


def test_calibration_passes_when_everything_is_delivered_unshaped():
    ok, details = calibration_verdict(_cal_row(), OFFERED)
    assert ok, details


def test_calibration_fails_on_any_drop():
    ok, details = calibration_verdict(_cal_row(drops={"be": 5}), OFFERED)
    assert not ok
    assert not details["be"]["ok"]


def test_calibration_fails_when_the_sender_is_throttled_even_if_delivery_looks_fine():
    """The stage 1b failure mode: a held-back sender can still produce plausible delivered rates.
    Calibration must check the sender reached its target, not only what arrived."""
    ok, details = calibration_verdict(_cal_row(sender_scale=0.8), OFFERED)
    assert not ok
    assert details["embb"]["sender_over_offered"] == pytest.approx(0.8, rel=1e-3)


def test_calibration_fails_when_delivery_misses_the_offer():
    short = {s: 0.9 * OFFERED[s] for s in SLICES}
    ok, _ = calibration_verdict(_cal_row(tc_rates=short), OFFERED)
    assert not ok


# --------------------------------------------------------------------------- aggregation


def test_aggregate_excludes_invalid_runs_and_says_how_many():
    levels = [0.2, 0.35]
    rows = []
    for rep, good in enumerate((3.0, 3.2, 3.4)):
        r = {k: None for k in AGG_KEYS}
        r.update(level_index=0, valid=True, embb_goodput_l2_mbps=good)
        rows.append(r)
    bad = {k: None for k in AGG_KEYS}
    bad.update(level_index=0, valid=False, embb_goodput_l2_mbps=99.0)
    rows.append(bad)

    agg = aggregate_levels(rows, levels)
    assert agg[0]["n_valid_runs"] == 3
    assert agg[0]["n_invalid_runs"] == 1
    assert agg[0]["embb_goodput_l2_mbps"]["mean"] == pytest.approx(3.2)
    assert agg[0]["embb_goodput_l2_mbps"]["n"] == 3
    assert agg[1]["n_valid_runs"] == 0
    assert agg[1]["embb_goodput_l2_mbps"] is None


# --------------------------------------------------------------------------- ping delivery (bug fix)

from traffic.generator import ping_delivery  # noqa: E402


def test_delivery_is_100_percent_when_ping_paces_slower_than_requested():
    """The quick-sweep bug: 0 % loss, but ping sent every 0.055 s instead of 0.05 s, and the old
    count-based metric reported about 91 % delivered."""
    samples = _pings(lambda t: 1.0, T0, T0 + 24, interval=0.055)
    d = ping_delivery(samples, T0, T0 + 24)
    assert d["delivered_pct"] == pytest.approx(100.0)
    assert d["effective_interval_s"] == pytest.approx(0.055, rel=1e-6)
    old_metric = 100.0 * len(samples) / (24 / 0.05)
    assert old_metric == pytest.approx(90.9, abs=0.5), "reproduces the reading that was wrong"


def test_delivery_counts_real_gaps_in_the_sequence():
    samples = _pings(lambda t: 1.0, T0, T0 + 10)
    lossy = [s for s in samples if s[1] % 10 != 3]   # every tenth probe lost
    assert ping_delivery(lossy, T0, T0 + 10)["delivered_pct"] == pytest.approx(90.0, abs=0.5)


def test_delivery_with_too_few_samples_is_none_not_zero():
    assert ping_delivery([], T0, T0 + 10)["delivered_pct"] is None
    assert ping_delivery([(T0 + 1, 5, 1.0)], T0, T0 + 10)["delivered_pct"] is None


# --------------------------------------------------------------------------- stage 4a rule

from experiments.diagnose_rtt_tail import SCHEDULE, classify_rtt_tail  # noqa: E402


@pytest.mark.parametrize(
    "normal, rt, expected",
    [
        ([0.4, 0.6], [0.3, 0.3], "NO_TAIL"),          # tail did not reproduce
        ([4.7, 5.1], [0.2, 0.3], "TOOL_SCHEDULING"),  # RT removes it entirely
        ([4.7, 5.1], [3.9, 4.2], "PATH_JITTER"),      # RT barely helps
        ([4.0, 4.0], [2.0, 2.0], "PATH_JITTER"),      # exactly half counts as path
        ([4.7, 5.1], [1.5, 1.7], "PARTIAL"),          # RT removes most, not all
        ([], [0.2], "INCONCLUSIVE"),
    ],
)
def test_rtt_tail_rule_implements_the_preregistered_bands(normal, rt, expected):
    assert classify_rtt_tail(normal, rt)[0] == expected


def test_rtt_tail_schedule_alternates_priorities_and_includes_an_idle_reference():
    names = [n for n, _loaded, _prefix in SCHEDULE]
    assert names[0] == "idle_normal"
    loaded = [n for n in names if n.startswith("loaded")]
    assert loaded == ["loaded_normal", "loaded_rt", "loaded_normal", "loaded_rt"]
    for name, loaded_flag, prefix in SCHEDULE:
        assert loaded_flag == name.startswith("loaded")
        assert bool(prefix) == name.endswith("_rt")
