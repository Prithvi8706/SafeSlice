"""Figure 6: the real Open vSwitch level sweep, against the simulator.

Two panels, never one dual-axis chart, because goodput (Mbps) and latency (ms) are different units.

Encoding, held constant across both panels so a reader learns it once:
- colour is the SLICE: eMBB blue, Best Effort orange, URLLC aqua (reference palette slots 1 to 3,
  validated for colour-vision deficiency with scripts/validate_palette.js; all checks pass, with a
  contrast warning on aqua that is answered by labelling every line directly and by the tables in
  docs/REPORT.md section 7 carrying the same numbers)
- line style is the SOURCE: solid with filled markers is the testbed measurement (mean and 95 % t
  interval over 3 repeats), dashed with hollow markers is the simulator in `demand_proportional`
  mode, dotted grey is the simulator in the other two modes, which coincide exactly for URLLC

    python -m analysis.plot_testbed
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

SURFACE = "#fcfcfb"
TEXT_PRIMARY = "#0b0b0b"
TEXT_SECONDARY = "#52514e"
GRID = "#e6e5e1"
NEUTRAL = "#8a8986"
SLICE_COLOUR = {"embb": "#2a78d6", "be": "#eb6834", "urllc": "#1baf7a"}


def _style_axis(ax):
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(TEXT_SECONDARY)
        ax.spines[side].set_linewidth(0.8)
    ax.tick_params(colors=TEXT_SECONDARY, labelsize=9)
    ax.yaxis.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def _label(ax, x, y, text):
    ax.annotate(text, (x, y), xytext=(8, 0), textcoords="offset points", va="center",
                fontsize=8.5, color=TEXT_PRIMARY)


def make_figure(sweep_path: Path, compare_path: Path, out_path: Path) -> Path:
    sweep = json.loads(sweep_path.read_text(encoding="utf-8"))
    cmp_ = json.loads(compare_path.read_text(encoding="utf-8"))
    agg = sorted(sweep["aggregate"], key=lambda e: e["level_index"])
    levels = [e["embb_share"] for e in agg]
    sim_dp = cmp_["simulator"]["demand_proportional"]
    sim_eq = cmp_["simulator"]["equal"]
    sim_mr = cmp_["simulator"]["min_rate_proportional"]
    n = agg[0]["embb_goodput_l2_mbps"]["n"]

    fig, (ax_g, ax_l) = plt.subplots(1, 2, figsize=(11.5, 4.6), facecolor=SURFACE)
    fig.subplots_adjust(wspace=0.55, right=0.86, bottom=0.2)

    # ---- panel A: goodput
    _style_axis(ax_g)
    for s, name in (("embb", "eMBB"), ("be", "Best Effort")):
        key = f"{s}_goodput_l2_mbps"
        c = SLICE_COLOUR[s]
        mean = [e[key]["mean"] for e in agg]
        half = [e[key]["ci95_half"] for e in agg]
        ax_g.errorbar(levels, mean, yerr=half, color=c, linewidth=2, marker="o", markersize=7,
                      capsize=3, elinewidth=1.2, zorder=3)
        sim = [r[key] for r in sim_dp]
        ax_g.plot(levels, sim, color=c, linewidth=2, linestyle=(0, (5, 3)), marker="o",
                  markersize=7, markerfacecolor=SURFACE, markeredgewidth=1.6, zorder=2)
        _label(ax_g, levels[-1], mean[-1], f"{name}, testbed")
        _label(ax_g, levels[-1], sim[-1], f"{name}, simulator")
    ax_g.set_ylim(0, 6)
    ax_g.set_xlim(0.15, 0.85)
    ax_g.set_xticks(levels)
    ax_g.set_xlabel("eMBB rate cap (share of 10 Mbps link)", color=TEXT_SECONDARY, fontsize=9)
    ax_g.set_ylabel("Goodput (L2 Mbps)", color=TEXT_SECONDARY, fontsize=9)
    ax_g.set_title("Raising the eMBB cap moves capacity from Best Effort to eMBB",
                   color=TEXT_PRIMARY, fontsize=10, loc="left")

    # ---- panel B: URLLC median latency
    _style_axis(ax_l)
    c = SLICE_COLOUR["urllc"]
    key = "urllc_rtt_p50_ms"
    mean = [e[key]["mean"] for e in agg]
    half = [e[key]["ci95_half"] for e in agg]
    ax_l.errorbar(levels, mean, yerr=half, color=c, linewidth=2, marker="o", markersize=7,
                  capsize=3, elinewidth=1.2, zorder=3)
    dp = [r[key] for r in sim_dp]
    ax_l.plot(levels, dp, color=c, linewidth=2, linestyle=(0, (5, 3)), marker="o", markersize=7,
              markerfacecolor=SURFACE, markeredgewidth=1.6, zorder=2)
    other = [r[key] for r in sim_eq]
    if [round(v, 6) for v in other] != [round(r[key], 6) for r in sim_mr]:
        raise ValueError("equal and min_rate_proportional URLLC medians differ; relabel the figure")
    ax_l.plot(levels, other, color=NEUTRAL, linewidth=2, linestyle=(0, (1, 2.5)), zorder=1)
    _label(ax_l, levels[-1], mean[-1], "URLLC, testbed")
    _label(ax_l, levels[-1], dp[-1] - 0.35, "simulator,\ndemand_proportional")
    _label(ax_l, levels[-1], other[-1] + 0.45, "simulator, equal and\nmin_rate_proportional")
    ax_l.set_ylim(0, 9)
    ax_l.set_xlim(0.15, 0.85)
    ax_l.set_xticks(levels)
    ax_l.set_xlabel("eMBB rate cap (share of 10 Mbps link)", color=TEXT_SECONDARY, fontsize=9)
    ax_l.set_ylabel("URLLC median RTT (ms)", color=TEXT_SECONDARY, fontsize=9)
    ax_l.set_title("...and costs URLLC latency, which only one model reproduces",
                   color=TEXT_PRIMARY, fontsize=10, loc="left")

    # ---- shared legend for the source encoding
    from matplotlib.lines import Line2D

    handles = [
        Line2D([], [], color=TEXT_SECONDARY, linewidth=2, marker="o", markersize=7,
               label=f"real OVS testbed, mean ± 95% interval (n = {n})"),
        Line2D([], [], color=TEXT_SECONDARY, linewidth=2, linestyle=(0, (5, 3)), marker="o",
               markersize=7, markerfacecolor=SURFACE, markeredgewidth=1.6,
               label="simulator, demand_proportional (the mode the policy study used)"),
        Line2D([], [], color=NEUTRAL, linewidth=2, linestyle=(0, (1, 2.5)),
               label="simulator, equal and min_rate_proportional"),
    ]
    leg = fig.legend(handles=handles, loc="lower left", bbox_to_anchor=(0.06, 0.0), ncol=3,
                     frameon=False, fontsize=8.5)
    for t in leg.get_texts():
        t.set_color(TEXT_PRIMARY)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=160, facecolor=SURFACE)
    plt.close(fig)
    return out_path


def main() -> int:
    out = make_figure(
        REPO_ROOT / "results" / "summary" / "ovs_level_sweep.json",
        REPO_ROOT / "results" / "summary" / "sim_vs_ovs.json",
        REPO_ROOT / "results" / "summary" / "figures" / "fig6_testbed_sweep.png",
    )
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
