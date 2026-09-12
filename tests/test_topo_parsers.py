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
