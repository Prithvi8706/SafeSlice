"""Tests for the pure functions in experiments/measure_noise_floor.py.

The measurement itself needs Mininet and root. The parts that can silently lie do not: a ping
parser that matches nothing returns zero samples, and zero samples is easy to mistake for a quiet
machine. The decision rule is also tested band by band, because it was pre-registered in
docs/TESTBED_SETUP.md section 4 and the code must implement that table exactly.

The ping output below is a format fixture in the documented `ping -D` shape, not a measurement.
"""

from __future__ import annotations

import pytest

from experiments.measure_noise_floor import (
    discard_warmup,
    parse_ping_output,
    summarize,
    verdict,
    windowed_stats,
)

PING_D_OUTPUT = """PING 10.0.0.4 (10.0.0.4) 56(84) bytes of data.
[1757666400.000000] 64 bytes from 10.0.0.4: icmp_seq=1 ttl=64 time=4.10 ms
[1757666400.050000] 64 bytes from 10.0.0.4: icmp_seq=2 ttl=64 time=0.210 ms
[1757666400.100000] 64 bytes from 10.0.0.4: icmp_seq=3 ttl=64 time=0.190 ms
[1757666401.000000] 64 bytes from 10.0.0.4: icmp_seq=4 ttl=64 time=0.200 ms
[1757666401.050000] 64 bytes from 10.0.0.4: icmp_seq=5 ttl=64 time=0.230 ms

--- 10.0.0.4 ping statistics ---
6 packets transmitted, 5 received, 16.6667% packet loss, time 1050ms
rtt min/avg/max/mdev = 0.190/0.986/4.100/1.557 ms
"""


def test_parser_extracts_every_reply_with_timestamp_seq_and_rtt():
    samples, summary = parse_ping_output(PING_D_OUTPUT)
    assert len(samples) == 5
    assert samples[0] == (1757666400.0, 1, 4.10)
    assert samples[2] == (1757666400.1, 3, 0.190)
    assert [s[1] for s in samples] == [1, 2, 3, 4, 5]


def test_parser_reads_the_summary_including_fractional_loss():
    _, summary = parse_ping_output(PING_D_OUTPUT)
    assert summary == {"transmitted": 6, "received": 5, "loss_pct": 16.6667}


def test_parser_on_garbage_returns_no_samples_rather_than_raising():
    samples, summary = parse_ping_output("ping: connect: Network is unreachable\n")
    assert samples == []
    assert summary is None


def test_parser_ignores_replies_without_a_timestamp():
    """Without -D there is no timestamp, and windowing would be impossible. Refuse them."""
    no_d = "64 bytes from 10.0.0.4: icmp_seq=1 ttl=64 time=0.2 ms\n"
    samples, _ = parse_ping_output(no_d)
    assert samples == []


def test_warmup_discards_the_arp_affected_first_sample():
    """The 4.10 ms first reply is the ARP-resolution outlier the warm-up exists to remove."""
    samples, _ = parse_ping_output(PING_D_OUTPUT)
    kept = discard_warmup(samples, warmup_s=1.0)
    assert [s[1] for s in kept] == [4, 5]
    assert max(s[2] for s in kept) < 1.0


def test_summarize_reports_percentiles_over_all_samples():
    s = summarize([1.0, 2.0, 3.0, 4.0, 5.0])
    assert s["n"] == 5
    assert s["p50_ms"] == pytest.approx(3.0)
    assert s["min_ms"] == 1.0 and s["max_ms"] == 5.0
    assert s["p50_ms"] <= s["p95_ms"] <= s["p99_ms"] <= s["max_ms"]


def test_summarize_refuses_an_empty_series():
    with pytest.raises(ValueError):
        summarize([])


def test_windows_with_too_few_samples_are_dropped_not_reported():
    """A p95 of two samples is their maximum and would be a misleading number in a table."""
    samples = [(100.0 + i * 0.05, i, 0.2) for i in range(20)]   # window 0: 20 samples
    samples += [(101.5, 99, 9.9), (101.6, 100, 9.9)]            # window 1: 2 samples
    w = windowed_stats(samples, window_s=1.0, min_samples=5)
    assert w["windows_total"] == 2
    assert w["windows_kept"] == 1
    assert w["per_window_p95_max_ms"] == pytest.approx(0.2)


def test_windows_raise_when_nothing_is_usable():
    with pytest.raises(ValueError):
        windowed_stats([(100.0, 1, 0.2)], window_s=1.0, min_samples=5)


# --------------------------------------------------------------------------- the rule


@pytest.mark.parametrize(
    "p99, expected",
    [
        (0.20, "CLEAN"),
        (0.999, "CLEAN"),
        (1.0, "USABLE"),
        (3.0, "USABLE"),
        (3.01, "MARGINAL"),
        (3.49, "MARGINAL"),
        (3.5, "STOP"),
        (12.0, "STOP"),
    ],
)
def test_verdict_implements_the_preregistered_bands_at_hard_ms_7(p99, expected):
    """Bands from docs/TESTBED_SETUP.md section 4, with sla.hard_ms = 7.0 so half is 3.5 ms."""
    assert verdict(p99, hard_ms=7.0)[0] == expected


def test_stop_band_moves_with_the_configured_sla():
    """The STOP boundary is half of sla.hard_ms, not a hard-coded 3.5."""
    assert verdict(4.0, hard_ms=7.0)[0] == "STOP"
    assert verdict(4.0, hard_ms=20.0)[0] == "MARGINAL"


def test_a_tight_sla_never_lets_stop_undercut_the_usable_band():
    """If half of hard_ms fell below 3 ms, the fixed CLEAN and USABLE bands take precedence.

    Stated as a test because it is a real property of the rule as written: with hard_ms = 4.0,
    half is 2.0, but a p99 of 2.5 ms is still USABLE since that band is checked first. Whether
    that is the right rule for a very tight SLO is a judgment call; it is at least now explicit.
    """
    assert verdict(2.5, hard_ms=4.0)[0] == "USABLE"
    assert verdict(3.5, hard_ms=4.0)[0] == "STOP"
