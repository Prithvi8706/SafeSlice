# SETUP

Two environments. The pure-Python side runs anywhere. The real backends need a Linux VM with
root.

Anything below marked `TBD` has not been verified on your VM and I will not guess at it.

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

Expected: all tests pass, and tests marked `requires_mininet` are reported as skipped. As of
Week 1a there are no `requires_mininet` tests yet; the marker and the `--run-mininet` flag are
registered in `tests/conftest.py` and the first marked tests arrive with `OvsCliBackend` in
Week 1b.

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
python experiments/run_experiment.py --policy static_equal --scenario burst --seed 0
```

Verified working on Windows 11, Python 3.9.13, numpy 1.26.4, pandas 2.3.1, scipy 1.13.1,
pytest 8.3.5.

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
