"""Tests for the Open vSwitch and tc output parsers in net/topology/slice_topo.py.

These parsers are the highest-risk code in the testbed track, because the machine this project is
developed on has no Mininet, no OVS and no tc, so nothing else here can exercise them. A parser
bug would not crash; it would silently return an empty dict and every testbed telemetry sample
would read zero. That failure mode is invisible in a plot, so it gets tested against realistic
command output instead.

The fixtures below are the documented output shapes of `ovs-ofctl queue-stats` (OpenFlow 1.0 and
1.3 forms) and `tc -s class show dev <iface>` for an HTB hierarchy. They are format fixtures, not
measurements, and no number in them is used as a result anywhere.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]


def _load_slice_topo():
    """Load by path, not by import, so this works whether or not Mininet is installed."""
    spec = importlib.util.spec_from_file_location(
        "slice_topo", REPO_ROOT / "net" / "topology" / "slice_topo.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


topo = _load_slice_topo()


# --------------------------------------------------------------------------- queue-stats

QUEUE_STATS_OF13 = """OFPST_QUEUE reply (OF1.3) (xid=0x2): 3 queues
  port  4 queue  0: bytes=125000, pkts=100, errors=0, duration=61.239s
  port  4 queue  1: bytes=437500, pkts=350, errors=12, duration=61.239s
  port  4 queue  2: bytes=562500, pkts=450, errors=7, duration=61.239s
"""

QUEUE_STATS_OF10 = """OFPST_QUEUE reply (xid=0x2): 3 queues
  port  4 queue  0: bytes=125000, pkts=100, errors=0
  port  4 queue  1: bytes=437500, pkts=350, errors=12
  port  4 queue  2: bytes=562500, pkts=450, errors=7
"""

QUEUE_STATS_TWO_PORTS = """OFPST_QUEUE reply (OF1.3) (xid=0x2): 4 queues
  port  1 queue  0: bytes=999, pkts=9, errors=0, duration=61.239s
  port  4 queue  0: bytes=125000, pkts=100, errors=0, duration=61.239s
  port  4 queue  1: bytes=437500, pkts=350, errors=12, duration=61.239s
  port  4 queue  2: bytes=562500, pkts=450, errors=7, duration=61.239s
"""


@pytest.mark.parametrize("fixture", [QUEUE_STATS_OF13, QUEUE_STATS_OF10])
def test_queue_stats_parses_both_openflow_versions(fixture):
    stats = topo.parse_queue_stats(fixture)
    assert sorted(stats) == [0, 1, 2]
    assert stats[0]["bytes"] == 125000
    assert stats[1]["pkts"] == 350
    assert stats[2]["errors"] == 7


def test_queue_stats_filters_to_one_port():
    """s1 has queues on the bottleneck port only, but the reply covers every port."""
    everything = topo.parse_queue_stats(QUEUE_STATS_TWO_PORTS)
    assert everything[0]["bytes"] == 125000, "later port should win without a filter"

    filtered = topo.parse_queue_stats(QUEUE_STATS_TWO_PORTS, iface_ofport="4")
    assert sorted(filtered) == [0, 1, 2]
    assert filtered[0]["bytes"] == 125000

    other = topo.parse_queue_stats(QUEUE_STATS_TWO_PORTS, iface_ofport="1")
    assert sorted(other) == [0]
    assert other[0]["bytes"] == 999


def test_queue_stats_on_empty_output_returns_empty_not_garbage():
    assert topo.parse_queue_stats("") == {}
    assert topo.parse_queue_stats("OFPST_QUEUE reply (OF1.3) (xid=0x2): 0 queues\n") == {}


# --------------------------------------------------------------------------- tc classes

TC_CLASSES = """class htb 1:fffe root rate 10Mbit ceil 10Mbit burst 1600b cburst 1600b
 Sent 1125000 bytes 900 pkt (dropped 0, overlimits 0 requeues 0)
 rate 9000Kbit 750pps backlog 0b 0p requeues 0
 lended: 0 borrowed: 0 giants: 0
 tokens: 1875 ctokens: 1875

class htb 1:1 parent 1:fffe prio 0 rate 1Mbit ceil 10Mbit burst 1600b cburst 1600b
 Sent 125000 bytes 100 pkt (dropped 0, overlimits 0 requeues 0)
 rate 1000Kbit 83pps backlog 0b 0p requeues 0
 lended: 100 borrowed: 0 giants: 0
 tokens: 1875 ctokens: 1875

class htb 1:2 parent 1:fffe prio 0 rate 1Mbit ceil 3500Kbit burst 1600b cburst 1600b
 Sent 437500 bytes 350 pkt (dropped 12, overlimits 0 requeues 0)
 rate 3500Kbit 291pps backlog 4500b 3p requeues 0
 lended: 100 borrowed: 250 giants: 0
 tokens: 1875 ctokens: 1875

class htb 1:3 parent 1:fffe prio 0 rate 500Kbit ceil 4500Kbit burst 1600b cburst 1600b
 Sent 562500 bytes 450 pkt (dropped 7, overlimits 0 requeues 0)
 rate 4500Kbit 375pps backlog 7500b 5p requeues 0
 lended: 50 borrowed: 400 giants: 0
 tokens: 1875 ctokens: 1875
"""


def test_tc_parses_every_class():
    classes = topo.parse_tc_classes(TC_CLASSES)
    assert set(classes) == {"1:fffe", "1:1", "1:2", "1:3"}


def test_tc_extracts_backlog_which_is_the_whole_reason_tc_is_used():
    classes = topo.parse_tc_classes(TC_CLASSES)
    assert classes["1:1"]["backlog_bytes"] == 0
    assert classes["1:2"]["backlog_bytes"] == 4500
    assert classes["1:3"]["backlog_bytes"] == 7500


def test_tc_extracts_drops_and_sent_bytes():
    classes = topo.parse_tc_classes(TC_CLASSES)
    assert classes["1:2"]["dropped_pkts"] == 12
    assert classes["1:3"]["dropped_pkts"] == 7
    assert classes["1:2"]["sent_bytes"] == 437500


def test_tc_rate_comes_from_the_class_line_not_the_stats_line():
    """The word 'rate' appears twice per class: the configured rate and the measured rate.

    Taking the wrong one would report the instantaneous throughput as if it were the configured
    guarantee, which would look plausible and be wrong.
    """
    classes = topo.parse_tc_classes(TC_CLASSES)
    assert classes["1:1"]["rate"] == "1Mbit"      # configured, not the 1000Kbit measured
    assert classes["1:3"]["rate"] == "500Kbit"    # configured, not the 4500Kbit measured
    assert classes["1:2"]["ceil"] == "3500Kbit"


def test_queue_index_maps_onto_the_ovs_class_handle():
    assert topo.tc_handle_for_queue(0) == "1:1"
    assert topo.tc_handle_for_queue(1) == "1:2"
    assert topo.tc_handle_for_queue(2) == "1:3"
    classes = topo.parse_tc_classes(TC_CLASSES)
    for q in (0, 1, 2):
        assert topo.tc_handle_for_queue(q) in classes


def test_tc_on_empty_output_returns_empty_not_garbage():
    assert topo.parse_tc_classes("") == {}


def test_tc_missing_backlog_field_yields_none_not_zero():
    """A queue whose backlog could not be read must be distinguishable from one with no backlog.

    Reporting None as 0.0 would silently turn a failed measurement into a measurement of zero.
    """
    truncated = "class htb 1:1 parent 1:fffe prio 0 rate 1Mbit ceil 10Mbit\n"
    classes = topo.parse_tc_classes(truncated)
    assert classes["1:1"]["backlog_bytes"] is None


# --------------------------------------------------------------------------- config coupling


def test_slice_ports_and_queues_agree_with_the_queue_convention():
    """q0 URLLC, q1 eMBB, q2 Best Effort, matching net/backend.py."""
    from net.backend import BE, EMBB, URLLC

    assert topo.SLICE_QUEUE["urllc"] == URLLC
    assert topo.SLICE_QUEUE["embb"] == EMBB
    assert topo.SLICE_QUEUE["be"] == BE
    assert len(set(topo.SLICE_PORTS.values())) == 3, "slice ports must be distinct"


def test_every_slice_has_a_host_pair_across_the_bottleneck():
    senders = {pair[0] for pair in topo.SLICE_HOSTS.values()}
    receivers = {pair[1] for pair in topo.SLICE_HOSTS.values()}
    assert senders == {"h1", "h2", "h3"}, "senders must sit on s1"
    assert receivers == {"h4", "h5", "h6"}, "receivers must sit on s2"
    assert not senders & receivers


# --------------------------------------------------------------------------- iperf3 (real captures)
#
# These two fixtures are REAL iperf 3.16 output captured on the project's WSL2 testbed machine on
# 2026-09-13, not hand-written structures:
#   iperf3_316_udp_lossless.json      loopback, 2 Mbps offered, no loss
#   iperf3_316_udp_capped_lossy.json  two namespaces over a veth capped at 1 Mbit by tbf, 4 Mbps
#                                     offered, so sender and receiver genuinely differ
# They exist to pin down which JSON field is the receiver. The first stage 1 check got that wrong.
# The rates in them are parser fixtures only and are not cited as results anywhere.

FIXTURES = REPO_ROOT / "tests" / "fixtures"


def _fixture(name: str) -> str:
    return (FIXTURES / name).read_text(encoding="utf-8")


def test_end_sum_is_the_sender_on_iperf_316_which_is_the_bug_this_guards():
    """The regression test for the stage 1 error, stated directly against the real capture."""
    import json

    j = json.loads(_fixture("iperf3_316_udp_capped_lossy.json"))
    assert j["end"]["sum"]["sender"] is True
    assert j["end"]["sum"]["bits_per_second"] == j["end"]["sum_sent"]["bits_per_second"]
    assert j["end"]["sum"]["bits_per_second"] > 3 * j["end"]["sum_received"]["bits_per_second"]


def test_parser_takes_receiver_goodput_from_sum_received_not_sum():
    p = topo.parse_iperf3_udp(_fixture("iperf3_316_udp_capped_lossy.json"))
    assert p["parse_ok"]
    assert p["sender_bps"] == pytest.approx(4001634, rel=1e-6)
    assert p["receiver_bps"] == pytest.approx(976370, rel=1e-6)
    assert p["receiver_bps"] < p["sender_bps"]
    assert p["lost_packets"] == 803
    assert p["lost_percent"] == pytest.approx(74.9, abs=0.1)
    assert p["iperf_version"] == "iperf 3.16"


def test_parser_on_a_lossless_run_gives_near_equal_sender_and_receiver():
    p = topo.parse_iperf3_udp(_fixture("iperf3_316_udp_lossless.json"))
    assert p["parse_ok"]
    assert p["lost_packets"] == 0
    assert p["receiver_bps"] == pytest.approx(p["sender_bps"], rel=0.05)


def test_parser_refuses_to_substitute_sum_for_a_missing_receiver():
    """If sum_received is absent, receiver must be None, never quietly taken from end.sum."""
    import json

    j = json.loads(_fixture("iperf3_316_udp_capped_lossy.json"))
    del j["end"]["sum_received"]
    p = topo.parse_iperf3_udp(json.dumps(j))
    assert p["parse_ok"], "sender is still known"
    assert p["receiver_bps"] is None
    assert p["sender_bps"] == pytest.approx(4001634, rel=1e-6)


def test_parser_survives_leading_noise_and_reports_iperf_errors():
    noisy = "warning: something\n" + _fixture("iperf3_316_udp_lossless.json")
    assert topo.parse_iperf3_udp(noisy)["parse_ok"]

    err = topo.parse_iperf3_udp('{"start": {}, "intervals": [], "error": "unable to connect"}')
    assert not err["parse_ok"]
    assert "unable to connect" in err["error"]

    garbage = topo.parse_iperf3_udp("iperf3: error - unable to connect to server")
    assert not garbage["parse_ok"]
    assert garbage["receiver_bps"] is None


# --------------------------------------------------------------------------- tc rate strings


@pytest.mark.parametrize(
    "text, bps",
    [
        ("10Mbit", 10_000_000),     # all four of these appeared in the real stage 1 tc output
        ("3500Kbit", 3_500_000),
        ("500Kbit", 500_000),
        ("1Mbit", 1_000_000),
        ("64bit", 64),
        ("1Gbit", 1_000_000_000),
        ("100Kbps", 800_000),       # tc 'bps' is BYTES per second
    ],
)
def test_tc_rate_parser(text, bps):
    assert topo.parse_tc_rate(text) == pytest.approx(bps)


@pytest.mark.parametrize("text", [None, "", "fast", "10Mb"])
def test_tc_rate_parser_returns_none_on_unreadable_input(text):
    assert topo.parse_tc_rate(text) is None


def test_tc_ceil_from_the_parsed_class_round_trips_to_the_programmed_cap():
    classes = topo.parse_tc_classes(TC_CLASSES)
    assert topo.parse_tc_rate(classes["1:2"]["ceil"]) == pytest.approx(3_500_000)


# --------------------------------------------------------------------------- backpressure rule

CAP = 3_500_000.0


def _row(mult, sender_ratio_of_request=None, sender_bps=None):
    requested = mult * CAP
    if sender_bps is None:
        sender_bps = None if sender_ratio_of_request is None else sender_ratio_of_request * requested
    return {"offered_bps_requested": requested, "iperf3": {"sender_bps": sender_bps}}


def test_open_loop_when_the_sender_tracks_the_request_above_the_cap():
    rows = [_row(0.5, 1.0), _row(1.0, 1.0), _row(2.0, 0.99), _row(4.0, 0.97)]
    assert topo.classify_backpressure(rows, CAP)[0] == "OPEN_LOOP"


def test_backpressure_when_the_sender_is_pinned_near_the_cap_whatever_is_requested():
    """The pattern the first stage 1 run hinted at: 7 Mbps requested, about 3.5 Mbps sent."""
    rows = [
        _row(0.5, 1.0),
        _row(1.0, 1.0),
        _row(2.0, sender_bps=1.01 * CAP),
        _row(4.0, sender_bps=1.02 * CAP),
    ]
    assert topo.classify_backpressure(rows, CAP)[0] == "BACKPRESSURE"


def test_generator_limited_when_the_control_row_below_the_cap_already_falls_short():
    """If the sender cannot reach half the cap, the generator is broken and nothing else counts."""
    rows = [_row(0.5, 0.6), _row(1.0, 0.6), _row(2.0, sender_bps=CAP), _row(4.0, sender_bps=CAP)]
    assert topo.classify_backpressure(rows, CAP)[0] == "GENERATOR_LIMITED"


def test_mixed_when_the_sender_neither_tracks_nor_pins():
    rows = [_row(0.5, 1.0), _row(2.0, sender_bps=1.6 * CAP), _row(4.0, sender_bps=2.4 * CAP)]
    assert topo.classify_backpressure(rows, CAP)[0] == "MIXED"


def test_rule_refuses_to_classify_without_a_control_row():
    rows = [_row(2.0, 1.0), _row(4.0, 1.0)]
    assert topo.classify_backpressure(rows, CAP)[0] == "MIXED"


def test_a_missing_sender_rate_in_the_control_row_is_not_treated_as_healthy():
    rows = [_row(0.5, None), _row(2.0, 1.0), _row(4.0, 1.0)]
    assert topo.classify_backpressure(rows, CAP)[0] == "GENERATOR_LIMITED"
