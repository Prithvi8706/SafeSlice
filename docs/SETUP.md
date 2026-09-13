# SETUP

**UPDATE 2026-09-13: a Mininet testbed now exists**, on branch `feature/mininet-testbed`, built and
run on WSL2 Ubuntu 24.04 rather than the VirtualBox VM this file assumes. For that environment follow
`docs/TESTBED_SETUP.md`, not sections 1, 2 and 5 below. The banner below was true when written. It
remains true that `OvsCliBackend`, `RyuBackend` and the live policy loop on the testbed do not exist;
what exists is the topology, the traffic generator and the fixed-level experiments.

## STATUS: sections 1, 2, 4 and 5 describe a testbed that was never built

Read this before following anything below. **There are no real backends.** `OvsCliBackend`,
`RyuBackend`, the Mininet topology and the iperf3/ping traffic generator were never written, so
the VM install, the Ryu notes and the pre-session checklist are instructions for software that
does not exist in this repository. They are kept because they remain the correct starting point
if someone picks the testbed track up later, and deleting them would hide the fact that a planned
half of the project is missing.

**To run everything that does exist, you need section 3 only**: any OS, any Python 3.9+, no root,
no Mininet, no Open vSwitch.

---

Two environments. The pure-Python side runs anywhere. The real backends would need a Linux VM
with root.

Anything below marked `TBD` was never verified, and is now not pending but abandoned.

---

## 1. Which Python version, and why

**Use Python 3.9 or 3.10 on the VM.** Recommended: whatever the distro ships as the system
Python, and create the virtualenv from that.

Reasons, in order of how much they constrain the choice:

1. **Mininet's Python bindings are installed system-wide by the distro package, not by pip.**
   `mn` and `import mininet` come from `python3-mininet` or from the `install.sh` in the Mininet
   source tree, and they are built against the system interpreter. If you create a virtualenv on
   a *different* Python than the system one, `import mininet` will not resolve inside it. So
   create the venv with `--system-site-packages` on the system Python.
2. **Ryu is unmaintained and breaks on newer Pythons.** Its own documentation points at
   OpenStack's `os-ken` fork. The known breakage is around `eventlet` and stdlib modules removed
   in Python 3.12 (`distutils`, and `imp` before it). Python 3.9 and 3.10 are the versions most
   likely to work without patching. **The exact current state on your VM is `TBD` until the
   Week 4 feasibility check.** Do not install Ryu until then; nothing before Week 4 needs it.
3. Everything in this repository is written to run on 3.9 and up. It is developed and tested on
   Python 3.9.13.

Do not use Python 3.12+ on the VM if you want `RyuBackend` to have a chance.

---

## 2. VM install

Assumes Ubuntu 22.04 LTS. If your VM is a different distro, tell me and I will redo this section
rather than have you guess at package names.

Run these one at a time and check each before moving on.

```bash
sudo apt update
```

```bash
sudo apt install -y mininet openvswitch-switch openvswitch-common iperf3 iproute2 \
                    python3-venv python3-pip git
```

```bash
# Confirm the switch daemon is up. Nothing works without this.
sudo systemctl enable --now openvswitch-switch
sudo systemctl status openvswitch-switch --no-pager
```

```bash
# Record the versions. These go into docs/EXPERIMENTS.md and every results/raw/*/env.json.
mn --version; ovs-vsctl --version | head -1; iperf3 --version | head -1; uname -r; python3 -V
```

```bash
# Clean out any Mininet state left over from a previous crash. Do this before every session.
sudo mn -c
```

### Project environment

```bash
git clone <your-repo-url> SafeSlice && cd SafeSlice
```

```bash
# --system-site-packages so that `import mininet` resolves inside the venv. See section 1.
python3 -m venv --system-site-packages .venv
```

```bash
source .venv/bin/activate && pip install -r requirements.txt
```

```bash
# Everything except the real backends must pass here, without root.
pytest -q
```

Expected: all tests pass. There are **no** `requires_mininet` tests, because there is no code that
would need Mininet. The marker and the `--run-mininet` flag remain registered in
`tests/conftest.py` so that testbed tests have somewhere to land if that work is ever done.

### Sub-second ping

`experiments/measure_noise_floor.py` and the Week 2 traffic generator want ping intervals below
1 s. Unprivileged users cannot do that by default. Either run those scripts under `sudo`, or
widen the range once per boot:

```bash
sudo sysctl -w net.ipv4.ping_group_range="0 2147483647"
```

**Whether this is needed on your VM is `TBD`** and depends on your kernel and whether `ping` is
setuid or has `cap_net_raw`. Check with `ping -i 0.2 -c 3 127.0.0.1` as a normal user.

---

## 3. Development sandbox (no root, no Mininet)

This is where `SimBackend`, the policies, the guardrail, the reward and the metrics are developed
and tested. Windows, macOS or Linux, no privileges needed.

```bash
python -m venv .venv
```

```bash
# Windows PowerShell: .venv\Scripts\Activate.ps1
source .venv/bin/activate && pip install -r requirements.txt
```

```bash
pytest -q
```

```bash
python experiments/run_experiment.py --policy linucb --scenario burst --seed 0
```

The full grid, the tuning sweeps, the sensitivity study, the tables and the figures:

```bash
python experiments/run_suite.py       # 320 runs, ~30 min. --resume continues an interrupted one.
python experiments/tune.py            # alpha and epsilon sweeps, ~11 min
python experiments/sensitivity.py     # reward-weight study
python -m analysis.aggregate          # tables
python -m analysis.plots              # figures
```

Run these **one at a time**. `decision_latency_ms` is measured against a real wall clock, so a
second job on the same machine corrupts it.

Verified working on Windows 11, Python 3.9.13, numpy 1.26.4, pandas 2.3.1, scipy 1.13.1,
matplotlib 3.9.4, pytest 8.3.5.

---

## 4. Ryu / os-ken

**Do not install either yet.** The feasibility check is a Week 4 deliverable and its outcome
decides whether `RyuBackend` gets built at all. The plan explicitly allows Week 4 to be spent on
the reward sensitivity study and the Oracle baseline instead. Installing Ryu speculatively on a
VM that also has to run the real experiments risks breaking a working environment for a component
that is not on the critical path.

What is known going in: Ryu is no longer maintained upstream, its documentation directs users to
`os-ken`, and both depend on `eventlet`, which is where Python-version breakage shows up. What
that means on *your* VM at *your* Python version is `TBD` and will be established by running the
check, not by assuming.

---

## 5. Before each experiment session on the VM

```bash
sudo mn -c
```

```bash
uptime          # the load average is recorded per run; start from a quiet machine
```

Close anything else that uses CPU. The decision-latency metric and the RTT measurements are both
sensitive to host scheduling, and a busy machine will show up as noise attributed to the policy.
