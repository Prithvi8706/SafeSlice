"""Mininet topology and Open vSwitch QoS for the three-slice bottleneck.

    h1,h2,h3 --- s1 === s2 --- h4,h5,h6

All QoS lives on s1's egress port toward s2. The reverse direction is unshaped, which is why
`net/sim_backend.py` adds one queueing delay to the RTT rather than two.

TWO THINGS THAT WILL SILENTLY RUIN THE MEASUREMENT IF GOT WRONG

1. Do NOT set `bw` on the s1-s2 link. Mininet's TCLink installs its own HTB qdisc on the
   interface, and `ovs-vsctl set port ... qos=@newqos` with type=linux-htb installs another one
   on the same interface. Whichever is applied last wins and the other silently does nothing, so
   you end up measuring a rate limiter you did not think you were using. The bottleneck here is
   the OVS QoS `max-rate` on the port, and there is exactly one HTB hierarchy, owned by OVS.

2. OVS QoS and Queue records are NOT garbage collected when a port stops referencing them. They
   accumulate in the database across runs, and a stale record can be picked up by a later run.
   `clear_qos()` destroys them explicitly and is called on setup as well as teardown.

Run as root:

    sudo python3 net/topology/slice_topo.py --check    # build, verify, tear down
    sudo python3 net/topology/slice_topo.py --cli      # build, verify, drop to the Mininet CLI
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np  # noqa: E402

from config_loader import load_config  # noqa: E402
from net.backend import allocation_from_level  # noqa: E402

# Transport ports that identify each slice. Both TCP and UDP are matched on these, because
# iperf3 runs its control channel over TCP on the same port as its UDP data. Leaving the control
# channel unmatched would drop it into the default queue, which is q0, and put iperf3 signalling
# traffic inside the protected URLLC slice.
SLICE_PORTS = {"urllc": 5201, "embb": 5202, "be": 5203}
SLICE_QUEUE = {"urllc": 0, "embb": 1, "be": 2}

# h1 -> h4 carries URLLC, h2 -> h5 eMBB, h3 -> h6 Best Effort.
SLICE_HOSTS = {"urllc": ("h1", "h4"), "embb": ("h2", "h5"), "be": ("h3", "h6")}


# --------------------------------------------------------------------------- shell helpers


def sh(cmd: List[str], check: bool = True, timeout: int = 30) -> str:
    """Run a command and return stdout. Raises with both streams on failure."""
    p = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    if check and p.returncode != 0:
        raise RuntimeError(
            f"command failed ({p.returncode}): {' '.join(cmd)}\n"
            f"stdout: {p.stdout.strip()}\nstderr: {p.stderr.strip()}"
        )
    return p.stdout


# --------------------------------------------------------------------------- QoS programming


def clear_qos(iface: Optional[str] = None) -> None:
    """Remove QoS from a port and destroy every orphaned QoS and Queue record.

    The `--all destroy` calls are the important part. Without them, records leak across runs and
    a later run can bind a queue configured by an earlier one.
    """
    if iface:
        sh(["ovs-vsctl", "--if-exists", "clear", "port", iface, "qos"], check=False)
    sh(["ovs-vsctl", "--all", "destroy", "qos"], check=False)
    sh(["ovs-vsctl", "--all", "destroy", "queue"], check=False)


def apply_qos(iface: str, cfg, level_index: int) -> Dict[str, int]:
    """Create the HTB QoS hierarchy on `iface` with three queues.

    Returns the caps in bits per second that were programmed, so the caller can assert against
    what it asked for rather than trusting that it landed.
    """
    capacity = int(cfg.link.capacity_bps)
    alloc = allocation_from_level(cfg, level_index)

    min_rates = {
        "urllc": int(float(cfg.slices.urllc.min_share) * capacity),
        "embb": int(float(cfg.slices.embb.min_share) * capacity),
        "be": int(float(cfg.slices.be.min_share) * capacity),
    }
    max_rates = {
        "urllc": int(alloc.urllc_cap_bps),
        "embb": int(alloc.embb_cap_bps),
        "be": int(alloc.be_cap_bps),
    }

    clear_qos(iface)

    cmd = [
        "ovs-vsctl",
        "--", "set", "port", iface, "qos=@newqos",
        "--", "--id=@newqos", "create", "qos", "type=linux-htb",
        f"other-config:max-rate={capacity}",
        "queues:0=@q0", "queues:1=@q1", "queues:2=@q2",
    ]
    for name, ref in (("urllc", "@q0"), ("embb", "@q1"), ("be", "@q2")):
        cmd += [
            "--", f"--id={ref}", "create", "queue",
            f"other-config:min-rate={min_rates[name]}",
            f"other-config:max-rate={max_rates[name]}",
        ]
    sh(cmd)
    return max_rates


def set_queue_max_rate(queue_index: int, bps: int) -> None:
    """Change one queue's max-rate in place. This is the actuator the agent drives.

    Looked up by the queue's position in the QoS record's `queues` column rather than by a cached
    UUID, so a re-created QoS record cannot leave this writing to a dead queue.
    """
    qos_uuid = sh(["ovs-vsctl", "--columns=_uuid", "--bare", "list", "qos"]).strip().splitlines()
    if not qos_uuid:
        raise RuntimeError("no QoS record exists; apply_qos() has not run")
    queues = sh(["ovs-vsctl", "get", "qos", qos_uuid[0], "queues"]).strip()
    # Format: {0=<uuid>, 1=<uuid>, 2=<uuid>}
    m = re.search(rf"\b{queue_index}\s*=\s*([0-9a-f-]+)", queues)
    if not m:
        raise RuntimeError(f"queue {queue_index} not present in QoS queues column: {queues}")
    sh(["ovs-vsctl", "set", "queue", m.group(1), f"other-config:max-rate={int(bps)}"])


# --------------------------------------------------------------------------- flows


def install_flows(switch: str = "s1") -> None:
    """Classify each slice into its queue.

    `set_queue` then `normal` so OVS still does its own L2 learning and forwarding. The switch is
    created in standalone fail mode, so NORMAL is populated even with no controller attached.

    ICMP goes to q0 deliberately: ping is how URLLC's RTT is measured, so it has to sit in the
    same queue as the URLLC traffic or it measures a queue nobody is using.
    """
    sh(["ovs-ofctl", "-O", "OpenFlow13", "del-flows", switch])
    flows = []
    for name, port in SLICE_PORTS.items():
        q = SLICE_QUEUE[name]
        flows.append(f"priority=300,udp,tp_dst={port},actions=set_queue:{q},normal")
        flows.append(f"priority=300,tcp,tp_dst={port},actions=set_queue:{q},normal")
        flows.append(f"priority=300,tcp,tp_src={port},actions=set_queue:{q},normal")
    flows.append(f"priority=250,icmp,actions=set_queue:{SLICE_QUEUE['urllc']},normal")
    flows.append("priority=0,actions=normal")
    for f in flows:
        sh(["ovs-ofctl", "-O", "OpenFlow13", "add-flow", switch, f])


# --------------------------------------------------------------------------- topology


def build_network(cfg):
    """Construct and start the Mininet network. Imports Mininet lazily so this module can be
    imported (and unit tested) on a machine that has no Mininet installed."""
    from mininet.link import TCLink
    from mininet.net import Mininet
    from mininet.node import OVSKernelSwitch

    net = Mininet(switch=OVSKernelSwitch, link=TCLink, controller=None, autoSetMacs=True)

    s1 = net.addSwitch("s1", failMode="standalone")
    s2 = net.addSwitch("s2", failMode="standalone")

    for i in range(1, 7):
        net.addHost(f"h{i}", ip=f"10.0.0.{i}/24")

    for i in (1, 2, 3):
        net.addLink(f"h{i}", s1)
    for i in (4, 5, 6):
        net.addLink(f"h{i}", s2)

    # No bw= here. See the module docstring. Delay is the one tc parameter that does not fight
    # with OVS QoS, because netem sits at a different point in the qdisc chain than the HTB root
    # OVS installs; it is left unset so that the measured base RTT is the testbed's own, which is
    # what experiments/measure_noise_floor.py is for.
    net.addLink(s1, s2)

    net.start()
    return net


def bottleneck_iface(net) -> str:
    """s1's interface facing s2, discovered rather than hard-coded."""
    links = net["s1"].connectionsTo(net["s2"])
    if not links:
        raise RuntimeError("no link between s1 and s2")
    return links[0][0].name


# --------------------------------------------------------------------------- verification


def parse_queue_stats(out: str, iface_ofport: Optional[str] = None) -> Dict[int, Dict]:
    """Parse `ovs-ofctl queue-stats` output. Pure function, unit tested.

    Handles both the OpenFlow 1.0 form and the 1.3 form, which appends a duration field. Keyed
    by queue id; when `iface_ofport` is given, only that port's queues are returned, which
    matters because s1 has queues configured on one port and the reply covers all of them.
    """
    stats: Dict[int, Dict] = {}
    for m in re.finditer(
        r"port\s+(\S+)\s+queue\s+(\d+):\s*bytes=(\d+),\s*pkts=(\d+),\s*errors=(\d+)", out
    ):
        port, qid, b, p, e = m.groups()
        if iface_ofport is not None and port != str(iface_ofport):
            continue
        stats[int(qid)] = {"bytes": int(b), "pkts": int(p), "errors": int(e), "port": port}
    return stats


def read_queue_stats(iface_ofport: Optional[str] = None, switch: str = "s1") -> Dict[int, Dict]:
    out = sh(["ovs-ofctl", "-O", "OpenFlow13", "queue-stats", switch])
    return parse_queue_stats(out, iface_ofport)


def tc_handle_for_queue(queue_index: int) -> str:
    """The tc class handle Open vSwitch gives to queue `queue_index`.

    OVS's linux-htb implementation creates each queue as class `1:(queue_id + 1)`, reserving
    `1:fffe` for the default class. This is an OVS implementation detail, not a promise, so
    `parse_tc_classes` is written to survive the handle not being there and `verify()` asserts
    the expected handles exist rather than assuming they do.
    """
    return f"1:{queue_index + 1}"


def parse_tc_classes(out: str) -> Dict[str, Dict]:
    """Parse `tc -s class show dev <iface>` output. Pure function, unit tested.

    tc is the only source of instantaneous queue BACKLOG. Open vSwitch does not report it:
    `queue-stats` gives transmitted bytes, packets and errors, not depth. See
    docs/PLAN_TESTBED.md section 2.2.
    """
    classes: Dict[str, Dict] = {}
    current = None
    for line in out.splitlines():
        stripped = line.strip()
        m = re.match(r"class htb (\S+)", stripped)
        if m:
            current = m.group(1)
            classes[current] = {
                "rate": None,
                "ceil": None,
                "backlog_bytes": None,
                "sent_bytes": None,
                "dropped_pkts": None,
            }
            r = re.search(r"\brate (\S+)", stripped)
            c = re.search(r"\bceil (\S+)", stripped)
            if r:
                classes[current]["rate"] = r.group(1)
            if c:
                classes[current]["ceil"] = c.group(1)
            continue
        if current is None:
            continue
        # "Sent 12345 bytes 100 pkt (dropped 5, overlimits 0 requeues 0)"
        s = re.search(r"Sent (\d+) bytes", stripped)
        if s:
            classes[current]["sent_bytes"] = int(s.group(1))
        d = re.search(r"dropped (\d+)", stripped)
        if d:
            classes[current]["dropped_pkts"] = int(d.group(1))
        # "rate 500Kbit 40pps backlog 3000b 2p requeues 0"
        b = re.search(r"backlog\s+(\d+)b", stripped)
        if b:
            classes[current]["backlog_bytes"] = int(b.group(1))
    return classes


def read_tc_classes(iface: str) -> Dict[str, Dict]:
    """Run tc and parse it. Returns {} if tc is unavailable or the command fails, so the caller
    can record that it fell back rather than reporting zero backlog as if it were measured."""
    try:
        out = sh(["tc", "-s", "class", "show", "dev", iface])
    except Exception:  # noqa: BLE001
        return {}
    return parse_tc_classes(out)


def queue_backlog_bytes(iface: str) -> Dict[int, Optional[int]]:
    """Backlog per slice queue index, or None per queue when it could not be read."""
    classes = read_tc_classes(iface)
    out: Dict[int, Optional[int]] = {}
    for q in (0, 1, 2):
        entry = classes.get(tc_handle_for_queue(q))
        out[q] = entry.get("backlog_bytes") if entry else None
    return out


def verify(net, cfg, programmed: Dict[str, int]) -> Tuple[bool, Dict]:
    """Stage 1 acceptance checks. Returns (all_passed, report)."""
    iface = bottleneck_iface(net)
    report: Dict = {"bottleneck_iface": iface, "programmed_max_rates_bps": programmed}
    ok = True

    # 1. Full connectivity.
    loss = net.pingAll(timeout="1")
    report["pingall_loss_pct"] = loss
    if loss != 0:
        ok = False

    # 2. Three queues visible to OpenFlow.
    qstats = read_queue_stats()
    report["queue_ids_seen"] = sorted(qstats)
    if not {0, 1, 2}.issubset(set(qstats)):
        ok = False

    # 3. Three HTB classes visible to tc, and whether backlog is readable.
    tc_classes = read_tc_classes(iface)
    report["tc_classes"] = tc_classes
    report["backlog_readable"] = any(
        v.get("backlog_bytes") is not None for v in tc_classes.values()
    )
    if len(tc_classes) < 3:
        ok = False

    # 4. The eMBB cap actually binds. This is the check that matters most: if the shaper is not
    #    engaging, every downstream measurement is of an unshaped link and means nothing.
    #
    #    CORRECTED 2026-09-13. The first version read iperf3's `end.sum.bits_per_second` as the
    #    receiver goodput. On iperf 3.16 that field is the SENDER rate (`"sender": true`); the
    #    receiver is `end.sum_received`. Verified against a controlled lossy run, which is now a
    #    test fixture: 4.002 Mbps in `end.sum`, 0.976 Mbps in `end.sum_received`, across a 1 Mbit
    #    cap. The first stage 1 run therefore reported a sender rate as if it were goodput, and its
    #    PASS on this check is void. It is kept at
    #    results/summary/topo_check_superseded_sender_rate_bug.json.
    embb_cap = programmed["embb"]
    probe = embb_probe(net, iface, offered_bps=2.0 * embb_cap, duration_s=8.0)
    report["embb_probe"] = probe
    report["embb_cap_bps"] = embb_cap

    if not probe["iperf3"]["parse_ok"] or probe["iperf3"]["receiver_bps"] is None:
        ok = False
    else:
        receiver_over_cap = probe["iperf3"]["receiver_bps"] / embb_cap
        report["embb_receiver_over_cap"] = receiver_over_cap
        # Allow 15 percent under (UDP header accounting, iperf3 interval edges) and 10 over.
        if not (0.85 <= receiver_over_cap <= 1.10):
            ok = False

    report["transport_facts"] = transport_facts(net, iface)
    report["all_passed"] = ok
    return ok, report


# --------------------------------------------------------------------------- iperf3 and probes

#: Explicit UDP payload. Below the 1500 byte MTU once IP and UDP headers are added, so no datagram
#: is fragmented. A fragmented datagram is lost whole if any fragment is dropped, which inflates
#: loss at the bottleneck and would make drop counts depend on a default we never chose.
UDP_PAYLOAD_BYTES = 1400


def parse_iperf3_udp(text: str) -> Dict:
    """Parse `iperf3 -u -J` client output. Pure function, tested against real iperf 3.16 captures.

    The field that matters and that was once got wrong: on iperf 3.16, `end.sum` is marked
    `"sender": true` and its `bits_per_second` is the SENDER rate. Receiver goodput lives only in
    `end.sum_received`. If `sum_received` is absent (older iperf3), receiver goodput is reported as
    None rather than silently substituted from `end.sum`, because substituting it is precisely the
    bug this function exists to prevent.
    """
    out: Dict = {
        "parse_ok": False,
        "error": None,
        "sender_bps": None,
        "sender_bytes": None,
        "receiver_bps": None,
        "receiver_bytes": None,
        "packets_sent": None,
        "packets_received": None,
        "lost_packets": None,
        "lost_percent": None,
        "jitter_ms": None,
        "iperf_version": None,
    }
    try:
        blob = text[text.index("{"): text.rindex("}") + 1]
        j = json.loads(blob)
    except (ValueError, json.JSONDecodeError) as exc:
        out["error"] = f"unparseable iperf3 output: {exc}"
        return out

    if "error" in j:
        out["error"] = str(j["error"])
        return out

    out["iperf_version"] = j.get("start", {}).get("version")
    end = j.get("end", {})
    sent = end.get("sum_sent")
    recv = end.get("sum_received")

    if sent is None and "sum" in end and end["sum"].get("sender") is True:
        sent = end["sum"]

    if sent is not None:
        out["sender_bps"] = float(sent["bits_per_second"])
        out["sender_bytes"] = int(sent["bytes"])
        out["packets_sent"] = sent.get("packets")
    if recv is not None:
        out["receiver_bps"] = float(recv["bits_per_second"])
        out["receiver_bytes"] = int(recv["bytes"])
        out["packets_received"] = recv.get("packets")
        out["lost_packets"] = recv.get("lost_packets")
        out["lost_percent"] = recv.get("lost_percent")
        out["jitter_ms"] = recv.get("jitter_ms")

    out["parse_ok"] = sent is not None
    return out


def parse_tc_rate(text: Optional[str]) -> Optional[float]:
    """Convert a tc rate string such as '3500Kbit' to bits per second.

    tc prints bit rates with SI prefixes ('Kbit' is 1000 bits). A 'bps' suffix in tc means BYTES
    per second, so it is multiplied by 8. Returns None when the string cannot be read.
    """
    if not text:
        return None
    m = re.fullmatch(r"\s*([\d.]+)\s*([KMG]?)(bit|bps)\s*", text, flags=re.IGNORECASE)
    if not m:
        return None
    value = float(m.group(1))
    value *= {"": 1.0, "K": 1e3, "M": 1e6, "G": 1e9}[m.group(2).upper()]
    if m.group(3).lower() == "bps":
        value *= 8.0
    return value


def transport_facts(net, iface: str) -> Dict:
    """Kernel settings that decide whether a full shaper queue pushes back on the sender.

    Recorded on every check because they are the discriminating evidence for the backpressure
    question in docs/PLAN_TESTBED.md: a UDP sender can only be slowed by the shaper if the queue
    holding its packets is deep enough to exhaust the socket's send buffer before it drops.
    """
    def _read(cmd: List[str]) -> Optional[str]:
        try:
            return sh(cmd, check=False).strip()
        except Exception:  # noqa: BLE001
            return None

    return {
        "bottleneck_qdisc": _read(["tc", "qdisc", "show", "dev", iface]),
        "bottleneck_txqueuelen": _read(["cat", f"/sys/class/net/{iface}/tx_queue_len"]),
        "sender_host_qdisc": net["h2"].cmd("tc qdisc show dev h2-eth0").strip(),
        "net_core_wmem_default": _read(["sysctl", "-n", "net.core.wmem_default"]),
        "net_core_wmem_max": _read(["sysctl", "-n", "net.core.wmem_max"]),
        "udp_payload_bytes": UDP_PAYLOAD_BYTES,
    }


def embb_probe(net, iface: str, offered_bps: float, duration_s: float = 8.0) -> Dict:
    """Send one eMBB UDP flow h2 -> h5 and measure it from both ends and from the switch.

    Three independent views of the same flow, because the first stage 1 run trusted one view and
    it was the wrong one:
      - iperf3 sender:   what the application actually managed to send
      - iperf3 receiver: what crossed the bottleneck
      - tc class 1:2:    bytes the shaper transmitted and packets it dropped
    Queue backlog is sampled while the flow runs, since a backlog that sits full is the signature
    of a shaper pushing back on its sender.
    """
    import subprocess as _sp
    import tempfile

    h2, h5 = net["h2"], net["h5"]
    port = SLICE_PORTS["embb"]
    handle = tc_handle_for_queue(SLICE_QUEUE["embb"])
    workdir = tempfile.mkdtemp(prefix="safeslice_probe_")
    client_json = f"{workdir}/client.json"

    h5.cmd(f"iperf3 -s -p {port} -D --logfile {workdir}/server.log")
    for _ in range(50):
        if f":{port}" in h5.cmd("ss -ltn"):
            break
        time.sleep(0.1)

    before = read_tc_classes(iface).get(handle, {})
    t0 = time.time()
    proc = h2.popen(
        ["sh", "-c",
         f"iperf3 -c {h5.IP()} -p {port} -u -b {int(offered_bps)} -l {UDP_PAYLOAD_BYTES} "
         f"-t {duration_s:g} -J > {client_json} 2> {workdir}/client.err"],
        stdout=_sp.DEVNULL, stderr=_sp.DEVNULL,
    )

    backlog_samples: List[int] = []
    time.sleep(1.5)  # let the queue reach steady state before sampling
    while proc.poll() is None:
        b = queue_backlog_bytes(iface).get(SLICE_QUEUE["embb"])
        if b is not None:
            backlog_samples.append(int(b))
        time.sleep(0.5)
    proc.wait()
    elapsed = time.time() - t0
    after = read_tc_classes(iface).get(handle, {})
    h5.cmd(f"pkill -f 'iperf3 -s -p {port}' || true")

    try:
        text = Path(client_json).read_text(encoding="utf-8", errors="replace")
    except OSError as exc:
        text = f"<no client output: {exc}>"
    parsed = parse_iperf3_udp(text)
    if not parsed["parse_ok"]:
        parsed["raw_tail"] = text[-600:]

    def _delta(key):
        a, b = after.get(key), before.get(key)
        return None if a is None or b is None else a - b

    sent_delta = _delta("sent_bytes")
    return {
        "offered_bps_requested": float(offered_bps),
        "duration_s": float(duration_s),
        "wall_elapsed_s": float(elapsed),
        "iperf3": parsed,
        "sender_over_requested": (
            None if parsed["sender_bps"] is None else parsed["sender_bps"] / offered_bps
        ),
        "tc_class": handle,
        "tc_sent_bytes_delta": sent_delta,
        "tc_sent_bps_l2": None if sent_delta is None else sent_delta * 8.0 / elapsed,
        "tc_dropped_pkts_delta": _delta("dropped_pkts"),
        "tc_ceil": after.get("ceil"),
        "tc_ceil_bps": parse_tc_rate(after.get("ceil")),
        "backlog_bytes_samples": backlog_samples,
        "backlog_bytes_median": (
            float(np.median(backlog_samples)) if backlog_samples else None
        ),
        "backlog_bytes_max": max(backlog_samples) if backlog_samples else None,
    }


def classify_backpressure(rows: List[Dict], cap_bps: float) -> Tuple[str, str]:
    """Decide, from a sweep of offered rates, whether the sender is open-loop or throttled.

    Pre-registered in docs/PLAN_TESTBED.md section 2.6 before the diagnostic had run. Pure
    function, unit tested.

    rows must include one below the cap (the control) and at least one well above it.

      GENERATOR_LIMITED  the sender cannot reach its requested rate even BELOW the cap, so the
                         traffic generator itself is broken and nothing else can be concluded.
      OPEN_LOOP          above the cap, the sender still sends what it was asked. Excess is
                         dropped at the queue. This is what net/sim_backend.py assumes.
      BACKPRESSURE       above the cap, the sender is held near the cap. The queue pushes back on
                         the application instead of dropping. The simulator does NOT model this.
      MIXED              neither pattern cleanly. Report the table, draw no conclusion.
    """
    below = [r for r in rows if r["offered_bps_requested"] <= 0.75 * cap_bps]
    above = [r for r in rows if r["offered_bps_requested"] >= 1.9 * cap_bps]
    if not below or not above:
        return "MIXED", "sweep must include a row below 0.75x cap and one at 1.9x cap or above"

    def _sender_ratio(r):
        s = r["iperf3"]["sender_bps"]
        return None if s is None else s / r["offered_bps_requested"]

    control = [_sender_ratio(r) for r in below]
    if any(c is None for c in control) or min(control) < 0.9:
        return (
            "GENERATOR_LIMITED",
            "the sender missed its requested rate below the cap, so the generator, not the "
            "shaper, is limiting it",
        )

    ratios = [_sender_ratio(r) for r in above]
    sender_over_cap = [
        r["iperf3"]["sender_bps"] / cap_bps for r in above if r["iperf3"]["sender_bps"] is not None
    ]
    if all(x is not None and x >= 0.9 for x in ratios):
        return "OPEN_LOOP", "above the cap the sender still sent what it was asked; excess dropped"
    if sender_over_cap and all(x <= 1.25 for x in sender_over_cap):
        return (
            "BACKPRESSURE",
            "above the cap the sender was held within 25 percent of the cap regardless of the "
            "requested rate; the queue is pushing back on the application",
        )
    return "MIXED", "sender neither tracked the request nor stayed near the cap"


def diagnose_backpressure(net, iface: str, cfg, programmed: Dict[str, int]) -> Dict:
    """Stage 1b. Sweep the offered eMBB rate across the cap and classify the sender's behaviour."""
    cap = float(programmed["embb"])
    multipliers = (0.5, 1.0, 2.0, 4.0)
    rows = []
    for m in multipliers:
        print(f"[slice_topo] probing eMBB at {m:g}x cap = {m * cap / 1e6:.2f} Mbps ...", flush=True)
        rows.append(embb_probe(net, iface, offered_bps=m * cap, duration_s=8.0))
        time.sleep(2.0)  # let the queue drain before the next rate
    label, reason = classify_backpressure(rows, cap)
    return {
        "embb_cap_bps": cap,
        "multipliers": list(multipliers),
        "rows": rows,
        "classification": label,
        "classification_reason": reason,
        "transport_facts": transport_facts(net, iface),
    }


# --------------------------------------------------------------------------- entry point


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Build the slice topology and verify the QoS.")
    ap.add_argument("--scenario", default="burst")
    ap.add_argument("--level", type=int, default=1, help="eMBB action level index to program")
    ap.add_argument("--check", action="store_true", help="run stage 1 checks then tear down")
    ap.add_argument(
        "--diagnose-backpressure",
        action="store_true",
        help="sweep offered eMBB rate across the cap and classify sender behaviour",
    )
    ap.add_argument("--cli", action="store_true", help="drop to the Mininet CLI after setup")
    ap.add_argument("--report", default=None)
    args = ap.parse_args(argv)

    modes = [args.check, args.diagnose_backpressure, args.cli]
    if sum(bool(m) for m in modes) != 1:
        ap.error("pass exactly one of --check, --diagnose-backpressure, --cli")

    from mininet.log import setLogLevel

    setLogLevel("info")
    cfg = load_config(scenario=args.scenario)

    net = None
    try:
        net = build_network(cfg)
        iface = bottleneck_iface(net)
        print(f"\n[slice_topo] bottleneck interface: {iface}")
        programmed = apply_qos(iface, cfg, args.level)
        install_flows("s1")
        print(f"[slice_topo] programmed max-rates (bps): {programmed}\n")

        if args.diagnose_backpressure:
            diag = diagnose_backpressure(net, iface, cfg, programmed)
            out = REPO_ROOT / (args.report or "results/summary/backpressure_diagnosis.json")
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(json.dumps(diag, indent=2, sort_keys=True), encoding="utf-8")

            def _mbps(v):
                return "   n/a" if v is None else f"{v / 1e6:6.3f}"

            print("\n" + "=" * 92)
            print(f"  eMBB cap {diag['embb_cap_bps'] / 1e6:.2f} Mbps.  All rates in Mbps.")
            print("  mult  requested  sender  receiver  tc_l2_sent  iperf_loss%  tc_drops  "
                  "backlog_med_B  backlog_max_B")
            for m, r in zip(diag["multipliers"], diag["rows"]):
                ip = r["iperf3"]
                loss = ip["lost_percent"]
                print(f"  {m:4g}  {_mbps(r['offered_bps_requested'])}     "
                      f"{_mbps(ip['sender_bps'])}  {_mbps(ip['receiver_bps'])}    "
                      f"{_mbps(r['tc_sent_bps_l2'])}      "
                      f"{'  n/a' if loss is None else f'{loss:6.2f}'}     "
                      f"{r['tc_dropped_pkts_delta']!s:>7}  "
                      f"{r['backlog_bytes_median']!s:>13}  {r['backlog_bytes_max']!s:>13}")
            print("-" * 92)
            print(f"  CLASSIFICATION  {diag['classification']}")
            print(f"  {diag['classification_reason']}")
            print("=" * 92)
            tf = diag["transport_facts"]
            print(f"  bottleneck txqueuelen {tf['bottleneck_txqueuelen']}   "
                  f"wmem_default {tf['net_core_wmem_default']}   wmem_max {tf['net_core_wmem_max']}")
            print(f"  bottleneck qdisc:\n    " + (tf["bottleneck_qdisc"] or "n/a").replace("\n", "\n    "))
            print(f"[slice_topo] wrote {out}")
            return 0

        if args.check:
            ok, report = verify(net, cfg, programmed)
            out = REPO_ROOT / (args.report or "results/summary/topo_check.json")
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
            print("\n" + "=" * 70)
            print(json.dumps(report, indent=2, sort_keys=True))
            print("=" * 70)
            print(f"[slice_topo] {'PASS' if ok else 'FAIL'}   report written to {out}")
            probe = report.get("embb_probe", {})
            sor = probe.get("sender_over_requested")
            if sor is not None and sor < 0.9:
                print(
                    f"[slice_topo] NOTE: the sender managed only {sor * 100:.0f} percent of the "
                    "rate it was asked for.\n"
                    "             The cap check above is about the RECEIVER and can still pass,\n"
                    "             but a throttled sender means offered load is not what the\n"
                    "             simulator assumes. Run --diagnose-backpressure next."
                )
            if not report.get("backlog_readable"):
                print(
                    "[slice_topo] NOTE: queue backlog is NOT readable from tc on this system.\n"
                    "             ctx_q0_backlog_norm will be pinned to zero on this backend.\n"
                    "             See docs/PLAN_TESTBED.md section 2.2."
                )
            return 0 if ok else 1

        from mininet.cli import CLI

        CLI(net)
        return 0
    finally:
        if net is not None:
            try:
                clear_qos(bottleneck_iface(net))
            except Exception:  # noqa: BLE001
                clear_qos()
            net.stop()


if __name__ == "__main__":
    raise SystemExit(main())
