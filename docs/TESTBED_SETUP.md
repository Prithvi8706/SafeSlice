# TESTBED SETUP

How to get a machine that can run Mininet, Open vSwitch and the stage 1 to 6 experiments in
`docs/PLAN_TESTBED.md`.

---

## 0. The short answer: you already have one

You do not need to build a virtual machine. Your existing WSL2 Ubuntu 24.04 can do this. That was
not a guess; these are the probe results from your machine on 2026-09-12:

| Requirement | Result | Why it matters |
|---|---|---|
| Kernel | `6.6.87.2-microsoft-standard-WSL2` | Recent enough |
| `openvswitch.ko` present | **Yes**, `CONFIG_OPENVSWITCH=m` | This is the one that usually kills WSL2 for this. It is present. |
| `modprobe openvswitch` | **Loads** | The kernel datapath works, so no userspace-datapath fallback |
| Network namespaces | **Work** | Mininet builds hosts out of these |
| `tc` and HTB qdisc | **Work** | HTB is what OVS QoS is built on |
| systemd as PID 1 | **Yes**, `systemctl is-system-running` = `running` | `openvswitch-switch` is a systemd service |
| CPU / RAM available to WSL | 22 threads / 7 GB | Far more than a 10 Mbps emulation needs |

If `openvswitch.ko` had been absent, the only options would have been a custom WSL kernel or a
real VM. It is present, so neither is needed.

You also have VirtualBox 7.1.0 installed. Keep it as the fallback in section 3, not the first
choice.

---

## 1. Install (WSL2, recommended)

Run these from PowerShell one at a time. Each drops into WSL, does one thing, and comes back.

**1.1 Open a WSL root shell** and stay in it for the rest of section 1:

```powershell
wsl -d Ubuntu-24.04 -u root
```

Everything from here is inside that shell.

**1.2 Update the package index:**

```bash
apt update
```

**1.3 Install the testbed packages:**

```bash
apt install -y mininet openvswitch-switch openvswitch-common iperf3 iproute2 python3-venv python3-pip git
```

**1.4 Load the kernel module and make it stick across restarts:**

```bash
modprobe openvswitch && echo openvswitch >> /etc/modules-load.d/openvswitch.conf
```

**1.5 Start the switch daemon:**

```bash
systemctl enable --now openvswitch-switch && systemctl is-active openvswitch-switch
```

Expect `active`. If this fails, nothing downstream works, so stop and send me the output of
`journalctl -u openvswitch-switch -n 40 --no-pager`.

**1.6 Record the versions.** Paste this output back to me; it goes into `env.json` for every run
and into the report's reproducibility section:

```bash
uname -r; python3 -V; mn --version; ovs-vsctl --version | head -1; iperf3 --version | head -1
```

**1.7 Clear any stale Mininet state.** Do this before every session, not just once:

```bash
mn -c
```

---

## 2. Project environment

Ubuntu 24.04 enforces PEP 668, so `pip install` into the system Python is refused. You need a
virtualenv, and it must be created with `--system-site-packages` or `import mininet` will not
resolve inside it (the Mininet Python bindings come from the apt package, not from pip).

**2.1 Get the repository into WSL.** Two options, and the choice matters:

```bash
# Option A, recommended: clone into the WSL filesystem.
cd ~ && git clone https://github.com/Prithvi8706/SafeSlice.git && cd SafeSlice
```

```bash
# Option B: use the Windows checkout you already have, via /mnt/c.
cd /mnt/c/Users/prith/SafeSlice
```

**Use option A.** Option B works but `/mnt/c` is a 9p filesystem bridge, and file IO across it is
slow enough to show up in a measurement that logs a CSV row every second. It would put an
artefact of the Windows/Linux filesystem bridge into the decision-latency column. Option A also
means an accidental `mn -c` or a crashed run cannot touch your Windows working tree.

Trade-off with option A: you now have two checkouts and must push and pull between them rather
than editing in one place. Since I work in the Windows one and you run in the WSL one, the flow
is: I push a branch, you `git pull` in WSL, you run, you paste the output back.

**2.2 Create the environment:**

```bash
python3 -m venv --system-site-packages .venv
```

**2.3 Install dependencies:**

```bash
source .venv/bin/activate && pip install -r requirements.txt
```

**2.4 Confirm both halves work:**

```bash
python -c "import mininet, numpy, pandas, scipy; print('mininet + scientific stack OK')"
```

**2.5 Run the existing test suite.** This must pass before any testbed work, because it proves
the simulator half is intact on this machine:

```bash
pytest -q
```

Expected: **121 passed on `main`, 134 on `feature/mininet-testbed`** (the branch adds 13 parser
tests). Verified on this machine on 2026-09-12: 121 passed on `main` under Python 3.12.3,
numpy 2.5.3, pandas 3.0.5, scipy 1.18.1.

Note for the report's reproducibility section: those are major-version jumps from the numpy 1.26.4
and pandas 2.3.1 the simulator results in `docs/EXPERIMENTS.md` were produced with. The tests pass
on both, but tests passing is not the same as the published tables being bit-identical. Any
simulator number that gets compared directly against a testbed number in stage 5 is re-run on this
machine rather than read from the old tables.

**2.6 Check sub-second ping.** `experiments/measure_noise_floor.py` needs it, and section 2.3 of
`docs/PLAN_TESTBED.md` explains why 20 samples per interval rather than 5:

```bash
ping -i 0.05 -c 5 127.0.0.1 >/dev/null 2>&1 && echo "fast ping OK as this user" || echo "needs sudo or sysctl"
```

If that fails, either run the experiments under `sudo`, or widen the range once per boot:

```bash
sysctl -w net.ipv4.ping_group_range="0 2147483647"
```

---

## 3. Fallback: a real VirtualBox VM

Only if section 4 says WSL2's timing is not good enough.

Ubuntu 22.04 LTS desktop or server ISO, 4 CPUs, 4 GB RAM, 25 GB disk. Then section 1 from step
1.2 onward, unchanged, except Python will be 3.10 rather than 3.12.

One warning specific to your machine: `VirtualMachinePlatform` is enabled (that is what runs
WSL2), so VirtualBox will run on the Hyper-V backend rather than its own. That is slower and has
its own timing characteristics. It is not obviously better than WSL2 for this, which is why it is
the fallback and not the recommendation. **If you go this route, re-run the section 4 check on the
VM and compare, rather than assuming the VM is quieter.**

---

## 4. The check that decides whether this environment is usable at all

This is the part that matters more than the install.

WSL2 is a lightweight Hyper-V virtual machine. Its scheduler and clock introduce jitter that a
bare-metal Linux box would not. We are trying to measure queueing delay with a hard SLO in the
single-digit milliseconds. **If the idle round-trip jitter of the environment is comparable to the
SLO, the environment is measuring itself and not the slicing.**

So stage 2 of `docs/PLAN_TESTBED.md` is not a formality, it is the gate on this whole setup:

```bash
sudo python3 experiments/measure_noise_floor.py
```

It reports idle RTT p50, p95 and p99 over 60 seconds, measured h1 to h4 **through the configured
bottleneck and queue q0**, which is the path URLLC traffic actually takes. Pinging `127.0.0.1`
would measure the loopback device and tell us nothing about the path that matters.

The decision rule, written down before any measurement exists so it cannot be chosen to suit the
result. The reference scale is `sla.hard_ms` from `config/default.yaml` (currently 7.0 ms),
because that is the latency scale the whole project is built to resolve.

| Idle p99 | Verdict | Action |
|---|---|---|
| under 1 ms | `CLEAN` | Proceed. |
| 1 ms to 3 ms | `USABLE` | Proceed. Report the floor next to every testbed latency number. |
| above 3 ms, below half of `sla.hard_ms` | `MARGINAL` | Proceed only with the testbed SLO rescaled from this floor, and state that as a limitation. |
| at or above half of `sla.hard_ms` | `STOP` | The environment's own jitter is comparable to the effect being measured. Try the VirtualBox fallback and re-measure. If no better, this hardware cannot resolve the latency scale the project is built on. |

**Revision note.** An earlier version of this table defined only the first two bands plus "comparable
to or above the intended SLO", leaving everything between 3 ms and the SLO undefined. It was made
precise, with "comparable" fixed at half of `sla.hard_ms`, on 2026-09-12 before the script had been
written or run. The script implements this table exactly, in
`experiments/measure_noise_floor.py:verdict`.

Note that whatever this returns replaces `link.base_rtt_ms: 2.0` in `config/default.yaml` for the
testbed. That 2.0 is a modelling constant, not a measurement, and it has never been checked
against anything.

---

## 5. Known WSL2 caveats to watch for

- **Mininet leaks state on a crash.** `mn -c` before every session. If WSL is shut down mid-run
  (`wsl --shutdown`), veth pairs and namespaces can survive in the OVS database; `mn -c` plus
  `ovs-vsctl --all destroy qos` and `--all destroy queue` clears it. `net/topology/slice_topo.py`
  calls the latter two on both setup and teardown for exactly this reason.
- **`wsl --shutdown` from Windows kills a running experiment instantly.** Do not run it while a
  sweep is in progress.
- **Windows may throttle or sleep the WSL VM** when the laptop idles or goes on battery. Keep it
  plugged in and awake for the duration of a sweep, or the latency series will contain a gap that
  looks like a network event and is not.
- **Python is 3.12 here.** Fine for everything in this project. It is the wrong version for Ryu,
  but Ryu is explicitly out of scope per `docs/PLAN_TESTBED.md` section 7.
- **7 GB of RAM is allocated to WSL.** `docs/EXPERIMENTS.md` records that the simulator suite was
  twice killed by the OS for memory. If you re-run the simulator suite inside WSL rather than on
  Windows, watch for that, or raise the limit in `C:\Users\prith\.wslconfig`.

---

## 6. What to send me

After section 2:

1. The version block from step 1.6.
2. The `pytest -q` result from step 2.5.
3. The fast-ping result from step 2.6.

Then the stage 1 check, which is the first thing that actually exercises the topology:

```bash
sudo python3 net/topology/slice_topo.py --check
```

The check that matters in its output is the last one: a UDP flow offered at double the eMBB cap
must come out at the cap. **If the cap does not bind, stop there and send me the whole report
JSON.** Nothing downstream means anything if the shaper is not engaging.
