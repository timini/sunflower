"""Figures for the site. Reads results/, writes docs/img/plots/.

Colours follow one fixed order everywhere: low blue, mid orange, tweeter
green, the summed response in ink. FEM curves are dashed in the same hue.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from sunflower import design as D  # noqa: E402

ROOT = D.ROOT
RES = ROOT / "results"
OUT = ROOT / "docs/img/plots"
COL = dict(low="#2a78d6", mid="#eb6834", tweeter="#1baf7a", sum="#0b0b0b")
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"
NAMES = dict(low="Low: 15NMB420, reversed", mid="Mid: 6MCF200Nd", tweeter="Tweeter: T90A", sum="Sum")

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 10, "axes.edgecolor": MUTED, "axes.labelcolor": INK,
    "xtick.color": MUTED, "ytick.color": MUTED, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8, "lines.linewidth": 2.0,
    "legend.frameon": False, "figure.dpi": 150, "savefig.bbox": "tight", "axes.titleweight": "bold",
    "axes.titlesize": 11, "axes.titlelocation": "left",
})


def freq_axis(ax, lo=20, hi=20000):
    ax.set_xscale("log")
    ax.set_xlim(lo, hi)
    ticks = [t for t in (20, 50, 100, 200, 500, 1000, 2000, 5000, 10000, 20000) if lo <= t <= hi]
    ax.set_xticks(ticks)
    ax.set_xticklabels([f"{t/1000:g}k" if t >= 1000 else f"{t}" for t in ticks])
    ax.set_xlabel("Frequency (Hz)")


def excess_phase(f, phase_deg, gd_ms, band):
    """Wrap phase after removing the median group delay in band."""
    b = (f >= band[0]) & (f <= band[1])
    tau = float(np.median(gd_ms[b])) / 1000
    ph = np.unwrap(np.radians(phase_deg)) + 2 * np.pi * f * tau
    return np.degrees(np.angle(np.exp(1j * ph))), tau * 1000


BANDS = dict(low=(40, 2000), mid=(300, 12000), tweeter=(2000, 30000))
PASS = dict(low=(100, 500), mid=(500, 7000), tweeter=(7000, 20000))


def section_figure(name):
    pw = pd.read_csv(RES / f"planewave/{name}.csv")
    fem = pd.read_csv(RES / f"fem/{name}.csv") if (RES / f"fem/{name}.csv").exists() else None
    lo, hi = BANDS[name]
    pw = pw[(pw.frequency_hz >= lo) & (pw.frequency_hz <= hi)]
    f = pw.frequency_hz.values
    c = COL[name]
    fig, axes = plt.subplots(2, 2, figsize=(12, 7.6), layout="constrained")
    ax = axes[0, 0]
    ax.axvspan(*PASS[name], color=c, alpha=0.06, lw=0)
    ax.plot(f, pw.spl_db, color=c, label="Plane-wave")
    if fem is not None:
        ax.plot(fem.frequency_hz, fem.fem_0deg_spl_db, color=c, ls="--", lw=1.6, label="FEM, axisymmetric")
    ax.set(ylabel="dB SPL, 2.83 V, 1 m", title="Magnitude, on axis, unfiltered")
    top = np.ceil(pw.spl_db.max() / 5) * 5 + 3
    ax.set_ylim(top - 45, top)
    freq_axis(ax, lo, hi)
    ax.legend(loc="lower center")
    ax = axes[0, 1]
    ph, tau = excess_phase(f, pw.phase_deg.values, pw.group_delay_ms.values, PASS[name])
    ax.plot(f, ph, color=c, label="Plane-wave")
    if fem is not None:
        ff = fem.frequency_hz.values
        php = np.unwrap(np.radians(fem.fem_0deg_phase_deg.values)) + 2 * np.pi * ff * tau / 1000
        ax.plot(ff, np.degrees(np.angle(np.exp(1j * php))), color=c, ls="--", lw=1.6, label="FEM")
    ax.set(ylabel="Phase (deg)", ylim=(-185, 185), yticks=[-180, -90, 0, 90, 180],
           title=f"Phase, with {tau:.2f} ms bulk delay removed")
    freq_axis(ax, lo, hi)
    ax.legend(loc="lower left")
    ax = axes[1, 0]
    ax.plot(f, pw.group_delay_ms, color=c, label="Plane-wave")
    if fem is not None:
        ax.plot(fem.frequency_hz, fem.fem_group_delay_ms, color=c, ls="--", lw=1.6, label="FEM")
    b = (f >= PASS[name][0]) & (f <= PASS[name][1])
    gmax = float(np.percentile(pw.group_delay_ms.values[b], 95)) * 2.2 + 0.2
    ax.set(ylabel="Group delay (ms)", ylim=(0, gmax), title="Group delay")
    freq_axis(ax, lo, hi)
    ax.legend(loc="upper right")
    ax = axes[1, 1]
    if fem is not None:
        for th, ls in ((0, "-"), (15, "--"), (30, ":"), (45, "-.")):
            ax.plot(fem.frequency_hz, fem[f"fem_{th}deg_spl_db"], color=c, ls=ls, lw=1.6, label=f"{th} deg")
        ax.set_title("Off axis, FEM")
    else:
        from sunflower.simulate import tweeter_section
        for th, ls in ((0, "-"), (15, "--"), (30, ":"), (45, "-.")):
            from sunflower.acoustics import spl
            ax.plot(f, spl(tweeter_section(f, th)[0]), color=c, ls=ls, lw=1.6, label=f"{th} deg")
        ax.set_title("Off axis, 60 mm piston model")
    ax.set(ylabel="dB SPL, 2.83 V, 1 m")
    ax.set_ylim(top - 45, top)
    freq_axis(ax, lo, hi)
    ax.legend(loc="lower center", ncol=4)
    fig.suptitle(f"{NAMES[name]}", x=0.01, ha="left", fontsize=13, fontweight="bold", color=INK)
    fig.savefig(OUT / f"{name}-section.png")
    plt.close(fig)


def impedance_figure():
    fig, axes = plt.subplots(1, 2, figsize=(12, 3.8), layout="constrained")
    for ax, name in zip(axes, ("low", "mid")):
        d = pd.read_csv(RES / f"planewave/{name}.csv")
        d = d[(d.frequency_hz >= BANDS[name][0]) & (d.frequency_hz <= BANDS[name][1])]
        ax.plot(d.frequency_hz, d.impedance_ohm, color=COL[name])
        ax.set(ylabel="|Z| (ohm)", title=f"{NAMES[name]}: electrical impedance", ylim=(0, None))
        freq_axis(ax, *BANDS[name])
    fig.savefig(OUT / "impedance.png")
    plt.close(fig)


def system_figure():
    d = pd.read_csv(RES / "planewave/system.csv")
    dsp = json.loads((RES / "planewave/dsp.json").read_text())
    f = d.frequency_hz.values
    fig, axes = plt.subplots(2, 1, figsize=(12, 8), layout="constrained", sharex=True,
                             gridspec_kw=dict(height_ratios=(2.2, 1)))
    ax = axes[0]
    for k in ("low", "mid", "tweeter"):
        ax.plot(f, d[f"{k}_0deg_spl_db"], color=COL[k], lw=1.8, label=NAMES[k])
    ax.plot(f, d["sum_0deg_spl_db"], color=COL["sum"], lw=2.4, label="Sum, on axis")
    t = dsp["target_db"]
    ax.set(ylim=(t - 30, t + 8), ylabel="dB SPL, 2.83 V, 1 m",
           title=f"Through the crossover: LR4 at {dsp['crossover']['low_mid_hz']:.0f} Hz and "
                 f"{dsp['crossover']['mid_high_hz']/1000:.0f} kHz, gains and delays from dsp.json")
    for fx in (dsp["crossover"]["low_mid_hz"], dsp["crossover"]["mid_high_hz"]):
        ax.axvline(fx, color=MUTED, lw=0.8, ls=":")
    ax.legend(loc="lower center", ncol=4)
    ax = axes[1]
    gd = -np.gradient(np.unwrap(np.radians(d["sum_0deg_phase_deg"].values)), 2 * np.pi * f) * 1000
    ax.plot(f, gd, color=COL["sum"])
    ax.set(ylabel="Group delay (ms)", ylim=(0, 12), title="Group delay of the sum")
    freq_axis(ax, 20, 20000)
    fig.savefig(OUT / "system.png")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(12, 4.6), layout="constrained")
    for th, ls in ((0, "-"), (15, "--"), (30, ":")):
        ax.plot(f, d[f"sum_{th}deg_spl_db"], color=COL["sum"], ls=ls, lw=2.0 if th == 0 else 1.6, label=f"{th} deg")
    ax.set(ylim=(t - 30, t + 8), ylabel="dB SPL, 2.83 V, 1 m", title="Summed response off axis, plane-wave with curved-wavefront mouths")
    freq_axis(ax, 20, 20000)
    ax.legend(loc="lower left", ncol=3)
    fig.savefig(OUT / "system-off-axis.png")
    plt.close(fig)


def area_figure():
    lo, m = D.LOW, D.MID
    fig, axes = plt.subplots(1, 2, figsize=(12, 3.8), layout="constrained")
    z = np.linspace(lo.throat_z, lo.mouth_z, 800)
    ax = axes[0]
    ax.plot((z - lo.throat_z) * 1000, [lo.outer_area(v) * 1e4 for v in z], color=MUTED, lw=1.4, label="Walls only")
    ax.plot((z - lo.throat_z) * 1000, [lo.area(v) * 1e4 for v in z], color=COL["low"], label="Open area, walls minus centre body")
    ax.set(xlabel="Distance from the 15 inch voice-coil plane (mm)", ylabel="Area (cm²)",
           title="Low horn area", ylim=(0, None))
    ax.legend(loc="upper left")
    zm = np.linspace(0, m.length, 400)
    ax = axes[1]
    ax.plot(zm * 1000, [m.area(v) * 1e4 for v in zm], color=COL["mid"], label="Open area around the bullet")
    ax.set(xlabel="Distance from the mid throat (mm)", ylabel="Area (cm²)", title="Mid horn area", ylim=(0, None))
    ax.annotate("tweeter face", xy=(100, m.area(0.1001) * 1e4), xytext=(55, m.area(0.1001) * 1e4 + 60),
                arrowprops=dict(arrowstyle="->", color=MUTED), color=MUTED)
    ax.legend(loc="upper left")
    fig.savefig(OUT / "area.png")
    plt.close(fig)


def section_drawing():
    """2D half-section through the axis, to scale, with stations."""
    lo, m, tw = D.LOW, D.MID, D.TWEETER
    cut = json.loads((ROOT / "cad/cutlist.json").read_text())
    st = cut["stations_mm"]
    fig, ax = plt.subplots(figsize=(13, 6.2), layout="constrained")
    z = np.linspace(lo.baffle, lo.mouth_z, 200) * 1000
    w = np.array([lo.width(v / 1000) / 2 for v in z]) * 1000
    ax.fill_between(z, w, w + 18, color="#c9a77c", lw=0)
    ax.fill_between(z, -w, -w - 18, color="#c9a77c", lw=0)
    ax.fill_between([0, 25], [-350, -350], [-176.5, -176.5], color="#c9a77c", lw=0)
    ax.fill_between([0, 25], [176.5, 176.5], [350, 350], color="#c9a77c", lw=0)
    zi = np.linspace(lo.throat_z, lo.mouth_z, 900)
    ri = np.array([lo.inner_radius(v) for v in zi]) * 1000
    ax.fill_between(zi * 1000, -ri, ri, where=ri > 0, color="#1b1b1b", lw=0)
    # cone and motor of the 15
    a = lo.body.cone_radius * 1000
    ax.plot([0, lo.body.cone_depth * 1000], [a, 40], color=COL["low"], lw=3)
    ax.plot([0, lo.body.cone_depth * 1000], [-a, -40], color=COL["low"], lw=3)
    # mid bore
    zb = np.linspace(st["mid_throat"], st["mouth"], 50)
    rb = np.array([m.radius((v - st["mid_throat"]) / 1000) for v in zb]) * 1000
    ax.fill_between(zb, -rb, rb, color="white", lw=0)
    ax.fill_between([st["mid_throat"] - 25, st["tweeter_face"]], [-32, -32], [32, 32], color="#e3a21a", lw=0)
    ax.plot([st["mid_cone_rim"], st["mid_cone_rim"] - 28], [70, 22], color=COL["mid"], lw=3)
    ax.plot([st["mid_cone_rim"], st["mid_cone_rim"] - 28], [-70, -22], color=COL["mid"], lw=3)
    ax.plot([st["tweeter_face"]] * 2, [-29, 29], color=COL["tweeter"], lw=3)
    for zrod in (st["support_rods"],):
        ax.plot([zrod, zrod], [100, lo.width(zrod / 1000) / 2 * 1000], color="#9aa0a6", lw=2.5)
        ax.plot([zrod, zrod], [-100, -lo.width(zrod / 1000) / 2 * 1000], color="#9aa0a6", lw=2.5)
    ax.fill_between([-lo.body.cone_depth * 0 - 45, 0], [-350, -350], [350, 350], color="#f1efe9", lw=0)
    ax.text(-22, 0, "25 L sealed\nchamber", rotation=90, ha="center", va="center", color=MUTED, fontsize=9)
    labels = [(st["mouth"], "mouth"), (st["tweeter_face"], "tweeter face"), (st["mid_throat"], "mid throat"),
              (st["mid_cone_rim"], "6.5 in cone"), (st["mid_can_back"], "can back"), (st["mid_tail_tip"], "tail tip"),
              (lo.nose_end * 1000, "15 in nose tip"), (lo.body.cone_depth * 1000, "15 in voice coil"), (0, "cone rim")]
    dense = 0
    for i, (zz, lab) in enumerate(sorted(labels)):
        if zz > 550:
            row = dense % 4
            dense += 1
        else:
            row = i % 2
        y = -372 - row * 26
        ax.annotate(f"{lab} {zz:.0f}", xy=(zz, y), ha="center", va="top", fontsize=8, color=MUTED)
        ax.plot([zz, zz], [y + 6, -360], color=MUTED, lw=0.6)
    ax.set(aspect="equal", xlim=(-70, 880), ylim=(-490, 380), xlabel="z, distance from the 15 inch cone rim (mm)",
           ylabel="mm", title="Section through the axis, to scale (low horn walls shown across flats)")
    ax.grid(False)
    fig.savefig(OUT / "section-drawing.png")
    plt.close(fig)


def fem_compare():
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2), layout="constrained")
    for ax, name in zip(axes, ("low", "mid")):
        d = pd.read_csv(RES / f"fem/{name}.csv")
        for th, ls in ((0, "-"), (30, ":")):
            ax.plot(d.frequency_hz, d[f"planewave_{th}deg_spl_db"], color=MUTED, ls=ls, lw=1.6, label=f"Plane-wave {th} deg")
            ax.plot(d.frequency_hz, d[f"fem_{th}deg_spl_db"], color=COL[name], ls=ls, lw=2.0, label=f"FEM {th} deg")
        ax.axvspan(*PASS[name], color=COL[name], alpha=0.06, lw=0)
        top = np.ceil(d.fem_0deg_spl_db.max() / 5) * 5 + 3
        ax.set(ylim=(top - 40, top), ylabel="dB SPL, 2.83 V, 1 m", title=f"{NAMES[name]}: FEM against plane-wave")
        freq_axis(ax, d.frequency_hz.min(), d.frequency_hz.max())
        ax.legend(loc="lower center", ncol=2, fontsize=8)
    fig.savefig(OUT / "fem-vs-planewave.png")
    plt.close(fig)


def screen_figure():
    df = pd.read_csv(RES / "screen-15/search.csv")
    best = df[df.passes].drop_duplicates("driver_id")
    fig, ax = plt.subplots(figsize=(12, 4.6), layout="constrained")
    ax.scatter(best.ripple_db, best.mean_spl_db, s=22, color="#b7b5ae", edgecolor="white", linewidth=0.6,
               label="Best horn for each 15 inch driver")
    sel = best[best.driver_id == "18sound-15nmb420"].iloc[0]
    ax.scatter([sel.ripple_db], [sel.mean_spl_db], s=90, color=COL["low"], edgecolor="white", linewidth=2,
               label="18 Sound 15NMB420 (chosen)", zorder=3)
    for _, r in best.head(6).iterrows():
        ax.annotate(r.label, (r.ripple_db, r.mean_spl_db), xytext=(6, 4), textcoords="offset points", fontsize=8, color=MUTED)
    ax.set(xlabel="Peak-to-peak variation, 100 to 700 Hz (dB)", ylabel="Mean SPL, 2.83 V, 1 m",
           title=f"Low section screen: {df.driver_id.nunique()} drivers, {len(df)} driver, horn and chamber combinations",
           xlim=(2, 10))
    ax.legend(loc="lower right")
    fig.savefig(OUT / "screen-15.png")
    plt.close(fig)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    for n in ("low", "mid", "tweeter"):
        section_figure(n)
    impedance_figure()
    system_figure()
    area_figure()
    section_drawing()
    fem_compare()
    screen_figure()
    print("figures in", OUT)


if __name__ == "__main__":
    main()
