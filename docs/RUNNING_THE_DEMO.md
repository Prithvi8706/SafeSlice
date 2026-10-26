# Running the SafeSlice demo on your own machine

This guide takes you from a fresh machine to every demo result in about 15 minutes. It needs no
admin rights, no Linux and no network emulator. Everything runs on a simulator, and the real-switch
results are included in the repository as recorded data.

If you just want to know what the project is, read `docs/OVERVIEW.md` first. The presenter's
version of this demo, with talking points, is `docs/DEMO.md`.

---

## What you need

- **Python 3.9 or newer.** Check with `python --version` (Windows) or `python3 --version` (macOS
  and Linux).
- **Git.**
- About 200 MB of disk space for the Python packages.

The commands below are written for Windows PowerShell. On macOS or Linux, use `python3` where they
say `python`, and the activation line given in step 2.

---

## 1. Get the code

```powershell
git clone https://github.com/Prithvi8706/SafeSlice.git
cd SafeSlice
```

## 2. Create an environment and install the packages

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

On macOS or Linux, activate with `source .venv/bin/activate` instead.

If PowerShell refuses to run the activation script, run this once and try again:
`Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`.

## 3. Check that it works

```powershell
python -m pytest -q
```

**Expected:** `253 passed` in under a minute. If any test fails, stop here, because the
numbers in the steps below will not match.

---

## The demo

Every result is deterministic. A run is fixed by its policy, scenario and seed, so you get exactly
the numbers quoted below.

### Step 1: three controllers on identical traffic

The controller decides each second how much of a 10 Mbps link video traffic (eMBB) may use, while
protecting latency-critical traffic (URLLC). Run three controllers on the same traffic:

```powershell
python experiments/run_experiment.py --policy threshold --scenario burst --seed 0
python experiments/run_experiment.py --policy linucb --scenario burst --seed 0
python experiments/run_experiment.py --policy linucb_pretrained --scenario burst --seed 0
```

Each prints a summary. Compare these lines:

| Policy | What it is | SLA violation rate | eMBB goodput mean | guardrail intervention rate | mean reward |
|---|---|---|---|---|---|
| `threshold` | hand-written rule, the competitor | 1.48 % | 2.746 Mbps | 4.07 % | 0.1464 |
| `linucb` | the learning controller, learning live | 0.74 % | 2.747 Mbps | 23.70 % | 0.1669 |
| `linucb_pretrained` | the same, trained beforehand and frozen | 0.00 % | 2.918 Mbps | 0.00 % | 0.2029 |

**What it shows:** learning live costs exploration, so the safety guardrail has to step in often.
Trained beforehand, the same algorithm gets more throughput with no violations. The pre-trained
run takes a few seconds longer because it trains first.

**Across all 320 runs**, open `results/summary/figures/fig2_tradeoff.png`. Up is more throughput,
left is fewer violations. Online LinUCB loses to `threshold` on three of four scenarios;
pre-trained, it wins on three of four.

Other policies to try: `static_safe`, `static_equal`, `epsilon_greedy`, `oracle`. Other
scenarios: `adversarial`, `ramp`, `sawtooth`. Any seed works; these numbers are for seed 0.

### Step 2: what the safety guardrail guarantees

Open `results/summary/figures/fig1_timeseries_linucb_pretrained_adversarial_0.png`. The bottom
panel shows the level the controller proposed (dashed) and the level actually applied (green).
Red dots mark where the guardrail overrode it. Red dots in the top panel are violations that
still got through: the guardrail reacts to the previous second, so it cannot stop the first
second of a sudden spike.

What it does guarantee is tested directly:

```powershell
python -m pytest tests/test_guardrail.py -v
```

**Expected:** `13 passed`. The key one is `test_applied_action_is_always_inside_the_mask`. It
throws randomized cases at an adversarial policy that always demands the most aggressive level,
and checks that the applied level is always one the guardrail allowed.

Then open `results/summary/figures/fig4_guardrail.png`. The red bars on the right are escalations
caught only by the instantaneous latency signal, 38 % of all of them. A guardrail that watched
only smoothed latency would have missed them.

### Step 3: does the simulator match a real switch?

Every result so far comes from a simulator. Its key assumption was checked on a real Mininet and
Open vSwitch network, and those measurements are committed in `results/summary/`.

Open `results/summary/figures/fig6_testbed_sweep.png`. On the real switch, raising video's share
from 20 % to 80 % raises URLLC median latency from 0.10 to 7.67 ms, about 77-fold. The control
problem is real.

Now score the simulator against those real measurements:

```powershell
python experiments/compare_sim_vs_ovs.py
```

It runs in about 5 seconds and prints tables comparing three models of how a switch shares spare
capacity. **Expected**, near the end:

```
  mean NMAE:  demand_proportional 0.110   equal 0.300   min_rate_proportional 0.230
  CLASSIFICATION  TRACKS_demand_proportional   (threshold NMAE <= 0.25)
```

Only `demand_proportional`, the model the simulator study used, tracks the real switch. The other
two predict flat latency, which the switch contradicts. The command rewrites
`results/summary/sim_vs_ovs.json` with identical content, so `git status` stays clean.

### Step 4: the controllers live on the real switch

Read `docs/REPORT.md` section 7.5. The same code ran live on the real switch. The simulator
predicted that `linucb_pretrained` would come first; on the switch it tied for last. It holds
video at the 50 % level, which the simulator rates safe. On the real switch, latency at that level
jitters from second to second and crossed the hard limit in 24 % of those seconds.

The per-second logs of those runs are in `results/testbed_policies/ovs_cli/`.

**The takeaway:** a controller trained in a simulator inherits the simulator's blind spots.

---

## Optional: reproduce everything from scratch

The figures above are committed. To regenerate the simulator results yourself (about 30 minutes):

```powershell
python experiments/run_suite.py
python -m analysis.aggregate
python -m analysis.plots
```

Don't run two of these at the same time, or anything else heavy. One of the recorded metrics is
decision time, measured against the real clock, and a second job would distort it. If
`run_suite.py` is interrupted, `python experiments/run_suite.py --resume` continues where it
stopped.

## Optional: run the real-switch testbed

This needs Linux with root, Mininet and Open vSwitch. On Windows, that means WSL2 with
virtualization enabled in the BIOS. Setup is in `docs/TESTBED_SETUP.md`. The quickest check,
which builds the topology and confirms that the bandwidth cap is enforced, takes about 10 seconds:

```bash
sudo python3 net/topology/slice_topo.py --check --report /tmp/topo_check.json
```

The full testbed experiments take 23 to 32 minutes each, and are listed in `README.md` under
"The testbed".

---

## Troubleshooting

| Problem | Fix |
|---|---|
| `python` not found on macOS or Linux | Use `python3`. |
| `pip install` fails building pandas or scipy | Your Python is too old. Use 3.9 or newer. |
| Numbers differ from this guide | Check you used `--seed 0` and the scenario named in the step. |
| `±` shows as garbage in Git Bash | Cosmetic only. PowerShell, macOS and Linux terminals display it correctly. |
| Can't open images from the terminal | They are ordinary PNG files in `results/summary/figures/`. Open the folder in your file browser. |
