"""Figures for the site. Reads results/ and cad/, writes docs/img/plots/.

Colours follow one fixed order everywhere: low blue, mid orange, tweeter
green, the summed response in ink. FEM curves are dashed in the same hue.
"""
from __future__ import annotations

import json
import math

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from sunflower import design as D
from sunflower.geometry import law_name  # noqa: E402
from sunflower.acoustics import C  # noqa: E402

ROOT = D.ROOT
RES = ROOT / "results"
OUT = ROOT / "docs/img/plots"
COL = dict(low="#2a78d6", mid="#eb6834", tweeter="#1baf7a", sum="#0b0b0b")
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"
WOOD, PRINT, PLUG = "#c9a77c", "#2b2b2b", "#e3a21a"

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 10, "axes.edgecolor": MUTED, "axes.labelcolor": INK,
    "xtick.color": MUTED, "ytick.color": MUTED, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8, "lines.linewidth": 2.0,
    "legend.frameon": False, "figure.dpi": 150, "savefig.bbox": "tight", "axes.titleweight": "bold",
    "axes.titlesize": 11, "axes.titlelocation": "left",
})


def names():
    s = D.summary()
    return dict(low=f"Low: {s['low']['driver']}, reversed", mid=f"Mid: {s['mid']['driver']}",
                tweeter=f"Tweeter: {s['tweeter']['driver']}", sum="Sum")


def bands():
    f1, f2 = D.CROSSOVER_LOW_MID_HZ, D.CROSSOVER_MID_HIGH_HZ
    return dict(view=dict(low=(30, 3000), mid=(150, 16000), tweeter=(1000, 30000)),
                passband=dict(low=(100, f1), mid=(f1, f2), tweeter=(f2, 20000)))


def freq_axis(ax, lo=20, hi=20000):
    ax.set_xscale("log")
    ax.set_xlim(lo, hi)
    ticks = [t for t in (20, 50, 100, 200, 500, 1000, 2000, 5000, 10000, 20000, 30000) if lo <= t <= hi]
    ax.set_xticks(ticks)
    ax.set_xticklabels([f"{t/1000:g}k" if t >= 1000 else f"{t}" for t in ticks])
    ax.minorticks_off()
    ax.set_xlabel("Frequency (Hz)")


def excess_phase(f, phase_deg, gd_ms, band):
    b = (f >= band[0]) & (f <= band[1])
    tau = float(np.median(gd_ms[b])) / 1000
    ph = np.unwrap(np.radians(phase_deg)) + 2 * np.pi * f * tau
    return np.degrees(np.angle(np.exp(1j * ph))), tau * 1000


# ---------------------------------------------------------------- cutoffs
def cutoffs():
    """Characteristic frequencies of each section, with how each was found."""
    out = {}
    for name in ("low", "mid", "tweeter"):
        d = pd.read_csv(RES / f"planewave/{name}.csv")
        f, s = d.frequency_hz.values, d.spl_db.values
        lo, hi = bands()["passband"][name]
        ref = s[(f >= lo) & (f <= hi)].mean()
        below = f[(f < lo) & (s < ref - 3)]
        above = f[(f > hi) & (s < ref - 3)]
        out[name] = dict(f_low_3db=float(below.max()) if len(below) else None,
                         f_high_3db=float(above.min()) if len(above) else None, level_db=float(ref))
    lo = D.LOW
    a_m = math.sqrt(lo.outer_area(lo.mouth_z) / math.pi)
    out["low"]["mouth_cutoff_hz"] = C / (2 * math.pi * a_m)
    out["low"]["front_chamber_corner_hz"] = lo.area(lo.throat_z) * C / (2 * math.pi * lo.front_chamber_m3)
    m = D.MID
    out["mid"]["mouth_cutoff_hz"] = C / (math.pi * m.mouth_diameter)
    out["tweeter"]["datasheet_cutoff_hz"] = D.TW.cutoff_hz
    out["crossover_hz"] = [D.CROSSOVER_LOW_MID_HZ, D.CROSSOVER_MID_HIGH_HZ]
    (RES / "planewave/cutoffs.json").write_text(json.dumps(out, indent=2) + "\n")
    return out


# ---------------------------------------------------------------- per section
def section_figures(name):
    pw = pd.read_csv(RES / f"planewave/{name}.csv")
    polar = pd.read_csv(RES / f"planewave/{name}-polar.csv")
    fem = pd.read_csv(RES / f"fem/{name}.csv") if (RES / f"fem/{name}.csv").exists() else None
    lo, hi = bands()["view"][name]
    pb = bands()["passband"][name]
    pw = pw[(pw.frequency_hz >= lo) & (pw.frequency_hz <= hi)]
    polar = polar[(polar.frequency_hz >= lo) & (polar.frequency_hz <= hi)]
    f = pw.frequency_hz.values
    c = COL[name]
    top = math.ceil(pw.spl_db.max() / 5) * 5 + 3
    co = cutoffs()[name]

    fig, ax = plt.subplots(figsize=(11, 4.4), layout="constrained")
    ax.axvspan(*pb, color=c, alpha=0.07, lw=0)
    ax.plot(f, pw.spl_db, color=c, label="Plane-wave" if name != "tweeter" else "Datasheet")
    if fem is not None:
        ax.plot(fem.frequency_hz, fem.fem_0deg_spl_db, color=c, ls="--", lw=1.6, label="FEM, axisymmetric")
    for i, (key, lab) in enumerate((("f_low_3db", "-3 dB"), ("f_high_3db", "-3 dB"), ("mouth_cutoff_hz", "mouth cutoff"),
                                    ("datasheet_cutoff_hz", "datasheet cutoff"))):
        v = co.get(key)
        if v and lo < v < hi:
            ax.axvline(v, color=MUTED, lw=0.8, ls=":")
            ax.text(v * 1.03, top - 2 - 6 * (i % 2), f"{lab}\n{v:.0f} Hz", fontsize=8, color=MUTED, va="top")
    ax.set(ylabel="dB SPL, 2.83 V, 1 m", ylim=(top - 40, top),
           title="Magnitude on axis, unfiltered. Shaded: the band this section plays")
    freq_axis(ax, lo, hi)
    ax.legend(loc="lower center")
    fig.savefig(OUT / f"{name}-magnitude.png")
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.0), layout="constrained")
    ph, tau = excess_phase(f, pw.phase_deg.values, pw.group_delay_ms.values, pb)
    ax = axes[0]
    ax.plot(f, ph, color=c, label="Plane-wave" if name != "tweeter" else "Minimum phase from the datasheet")
    if fem is not None:
        ff = fem.frequency_hz.values
        php = np.unwrap(np.radians(fem.fem_0deg_phase_deg.values)) + 2 * np.pi * ff * tau / 1000
        ax.plot(ff, np.degrees(np.angle(np.exp(1j * php))), color=c, ls="--", lw=1.6, label="FEM")
    ax.set(ylabel="Phase (deg)", ylim=(-185, 185), yticks=[-180, -90, 0, 90, 180],
           title=f"Phase, {tau:.2f} ms bulk delay removed")
    freq_axis(ax, lo, hi)
    ax.legend(loc="lower left")
    ax = axes[1]
    ax.plot(f, pw.group_delay_ms, color=c)
    if fem is not None:
        ax.plot(fem.frequency_hz, fem.fem_group_delay_ms, color=c, ls="--", lw=1.6)
    b = (f >= pb[0]) & (f <= pb[1])
    gmax = float(np.percentile(pw.group_delay_ms.values[b], 95)) * 2.2 + 0.2
    ax.set(ylabel="Group delay (ms)", ylim=(0, gmax), title="Group delay, including the path to the mouth")
    freq_axis(ax, lo, hi)
    fig.savefig(OUT / f"{name}-phase.png")
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2), layout="constrained")
    ax = axes[0]
    for th, ls in ((0, "-"), (15, "--"), (30, ":"), (45, "-."), (60, (0, (1, 3)))):
        ax.plot(polar.frequency_hz, polar[f"spl_{th}deg_db"], color=c, ls=ls, lw=1.6, label=f"{th} deg")
    ax.set(ylabel="dB SPL, 2.83 V, 1 m", ylim=(top - 40, top), title="Off axis" + (
        ", from the datasheet" if name == "tweeter" else ", curved mouth wavefront"))
    freq_axis(ax, lo, hi)
    ax.legend(loc="lower center", ncol=5, fontsize=8)
    ax = axes[1]
    angles = [0, 15, 30, 45, 60]
    z = np.array([polar[f"spl_{a}deg_db"].values - polar["spl_0deg_db"].values for a in angles])
    z = np.vstack([z[::-1][:-1], z])
    ang = [-a for a in angles[::-1][:-1]] + angles
    cs = ax.contourf(polar.frequency_hz.values, ang, z, levels=np.arange(-24, 1, 3), cmap="Greys_r", extend="min")
    ax.contour(polar.frequency_hz.values, ang, z, levels=[-6], colors=[c], linewidths=1.6)
    freq_axis(ax, lo, hi)
    ax.set(ylabel="Angle (deg)", title="Polar map relative to on axis. Line: -6 dB")
    ax.grid(False)
    fig.colorbar(cs, ax=ax, label="dB")
    fig.savefig(OUT / f"{name}-polar.png")
    plt.close(fig)

    if "impedance_ohm" in pw:
        fig, axes = plt.subplots(1, 2, figsize=(12, 3.8), layout="constrained")
        axes[0].plot(f, pw.impedance_ohm, color=c)
        axes[0].set(ylabel="|Z| (ohm)", title="Electrical impedance", ylim=(0, None))
        freq_axis(axes[0], lo, hi)
        rec = D.LOW_RECORD if name == "low" else D.MID_RECORD
        z_nom = rec["parameters"].get("nominal_impedance_ohm") or 8
        p_w = rec["parameters"].get("power_w") or 100
        scale = math.sqrt(p_w * z_nom) / 2.83
        xmax = (rec["parameters"].get("xmax_m") or float("nan")) * 1000
        axes[1].plot(f, pw.excursion_peak_mm_2p83v * scale, color=c, label=f"{p_w:.0f} W, unfiltered")
        axes[1].axhline(xmax, color=MUTED, ls=":", lw=1)
        axes[1].text(f[0] * 1.1, xmax, f"Xmax {xmax:.1f} mm", va="bottom", fontsize=8, color=MUTED)
        axes[1].set(ylabel="Peak excursion (mm)", title=f"Cone excursion at {p_w:.0f} W", ylim=(0, max(xmax * 1.6, 1)))
        freq_axis(axes[1], lo, hi)
        axes[1].legend(loc="upper right")
        fig.savefig(OUT / f"{name}-electrical.png")
        plt.close(fig)


# ---------------------------------------------------------------- dimensioned drawings
def dim(ax, p0, p1, text, off=(0, 0), fs=8.5):
    ax.annotate("", xy=p1, xytext=p0, arrowprops=dict(arrowstyle="<->", color=INK, lw=0.9, shrinkA=0, shrinkB=0))
    mx, my = (p0[0] + p1[0]) / 2 + off[0], (p0[1] + p1[1]) / 2 + off[1]
    ax.text(mx, my, text, ha="center", va="center", fontsize=fs, color=INK,
            bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.9))


def mid_drawing():
    m, tw = D.MID, D.TW
    fig, ax = plt.subplots(figsize=(12, 6.8), layout="constrained")
    L = m.length * 1000
    ss = np.linspace(0, m.length, 300)
    ro = np.array([m.radius(s) for s in ss]) * 1000
    ri = np.array([m.plug(s) for s in ss]) * 1000
    rc = m.can_diameter / 2 * 1000
    z = ss * 1000
    ax.fill_between(z, ro, rc, color=PRINT, lw=0)
    ax.fill_between(z, -ro, -rc, color=PRINT, lw=0)
    ax.fill_between(z, -ri, ri, where=ri > 0, color=PLUG, lw=0)
    sh = np.linspace(m.s_exit, m.s_tip, 60)
    hr = np.array([m.hf_horn_radius(s) for s in sh]) * 1000
    ax.fill_between(sh * 1000, -hr, hr, color="white", lw=0)
    nose = np.linspace(-m.nose_length, 0, 30)
    nr = np.array([m.plug(s) for s in nose]) * 1000
    ax.fill_between(nose * 1000, -nr, nr, color=PLUG, lw=0)
    rb = tw.body_radius * 1000
    ax.fill_between([(m.s_exit - tw.body_depth) * 1000, m.s_exit * 1000], [-rb, -rb], [rb, rb], color="#7b7f84", lw=0)
    rcone = m.cone_diameter / 2 * 1000
    for sgn in (1, -1):
        ax.plot([0, -0.4 * rcone], [sgn * rcone, sgn * 0.3 * rcone], color=COL["mid"], lw=3)
    y0 = -rc - 22
    dim(ax, (0, y0), (L, y0), f"mid horn {L:.0f}")
    dim(ax, (L + 22, -m.mouth_diameter * 500), (L + 22, m.mouth_diameter * 500), f"mouth\n{m.mouth_diameter*1000:.0f}", off=(24, 0))
    dim(ax, (L + 70, -rc), (L + 70, rc), f"can\n{rc*2:.0f}", off=(22, 0))
    rt = m.outer_throat_radius * 1000
    dim(ax, (-14, -rt), (-14, rt), f"throat\n{2*rt:.0f}", off=(-34, 0))
    dim(ax, (m.s_exit * 1000 - 20, -m.plug_radius * 1000), (m.s_exit * 1000 - 20, m.plug_radius * 1000),
        f"plug {2*m.plug_radius*1000:.0f}", off=(0, -m.plug_radius * 1000 - 14))
    dim(ax, (m.s_exit * 1000, rc + 16), (m.s_tip * 1000, rc + 16), f"printed HF horn {m.hf_len*1000:.0f}")
    dim(ax, (m.s_tip * 1000 + 4, -m.hf_mouth_radius * 1000), (m.s_tip * 1000 + 4, m.hf_mouth_radius * 1000),
        f"{2*m.hf_mouth_radius*1000:.0f}", off=(14, 0))
    ax.text(-60, rc + 44, f"{law_name(m.law)} open-area law, {m.compression:.1f}:1 compression. Grey: {tw.model}, "
            f"{tw.body_diameter*1000:.0f} mm body. Yellow: phase plug, tapering so the open area follows the law.",
            fontsize=9, color=MUTED)
    ax.set(aspect="equal", xlim=(-90, L + 125), ylim=(-rc - 50, rc + 56), xlabel="mm from the mid throat (the cone rim)",
           title="Mid horn and phase plug, section through the axis, mm")
    ax.grid(False)
    fig.savefig(OUT / "mid-drawing.png")
    plt.close(fig)


def low_drawing():
    lo, m = D.LOW, D.MID
    cut = json.loads((ROOT / "cad/cutlist.json").read_text())
    st = cut["stations_mm"]
    fig, ax = plt.subplots(figsize=(13, 7.6), layout="constrained")
    zb, zm = lo.baffle * 1000, lo.mouth_z * 1000
    t = D.PLY * 1000
    zback = st["back"]
    # shell: angled top and bottom, back, baffle
    zz = np.linspace(zback, zm, 50)
    ys = np.array([lo.shell_half_height(z / 1000) for z in zz]) * 1000
    ax.fill_between(zz, ys, ys + t, color=WOOD, lw=0)
    ax.fill_between(zz, -ys, -ys - t, color=WOOD, lw=0)
    ax.fill_between([zback, zback + t], [-ys[0]] * 2, [ys[0]] * 2, color=WOOD, lw=0)
    hb = lo.shell_half_height(0) * 1000
    ax.fill_between([0, zb], [-hb] * 2, [-lo.cutout_radius * 1000] * 2, color=WOOD, lw=0)
    ax.fill_between([0, zb], [lo.cutout_radius * 1000] * 2, [hb] * 2, color=WOOD, lw=0)
    # sealed chamber everywhere inside the shell outside the horn
    ax.fill_between(zz, -ys, ys, color="#ece9e1", lw=0, zorder=0)
    # horn top and bottom walls
    z2 = np.linspace(zb, zm, 50)
    h = np.array([lo.height(z / 1000) / 2 for z in z2]) * 1000
    ax.fill_between(z2, -h, h, color="white", lw=0, zorder=1)
    ax.fill_between(z2, h, h + t, color="#b8925f", lw=0, zorder=2)
    ax.fill_between(z2, -h, -h - t, color="#b8925f", lw=0, zorder=2)
    # one-piece centre body
    zi = np.linspace(lo.throat_z, zm / 1000, 600)
    ri = np.array([lo.inner_radius(z) for z in zi]) * 1000
    ax.fill_between(zi * 1000, -ri, ri, color=PRINT, lw=0, zorder=3)
    b = lo.body
    a, cd = b.cone_radius * 1000, b.cone_depth * 1000
    for sgn in (1, -1):
        ax.plot([0, cd], [sgn * a, sgn * 40], color=COL["low"], lw=3, zorder=4)
    rr = np.linspace(b.motor_radius * 1000 + 3, lo.throat_radius * 1000, 20)
    zf = cd * (1 - rr / a) + (b.xmax + lo.gap) * 1000
    for sgn in (1, -1):
        ax.fill_betweenx(sgn * rr, zf, zb + 2, color=PRINT, lw=0, zorder=3)
    ax.fill_between([cd, lo.motor_end * 1000], [-b.motor_radius * 1000] * 2, [b.motor_radius * 1000] * 2, color="#55595e", zorder=4)
    # mid horn bore and HF plug
    s0 = st["mid_throat"]
    ss = np.linspace(0, m.length, 80)
    ax.fill_between(s0 + ss * 1000, [-m.radius(s) * 1000 for s in ss], [m.radius(s) * 1000 for s in ss], color="white", lw=0, zorder=5)
    pr = np.array([m.plug(s) for s in ss]) * 1000
    ax.fill_between(s0 + ss * 1000, -pr, pr, where=pr > 0, color=PLUG, lw=0, zorder=6)
    zr = st["support_rods"]
    hr = lo.height(zr / 1000) / 2 * 1000
    for sgn in (1, -1):
        ax.plot([zr, zr], [sgn * m.can_diameter * 500, sgn * hr], color="#9aa0a6", lw=2.5, zorder=6)
    H = lo.mouth_height * 1000
    dim(ax, (zm + 34, -H / 2), (zm + 34, H / 2), f"mouth\n{H:.0f}\nhigh", off=(40, 0))
    dim(ax, (zm + 100, -H / 2 - t), (zm + 100, H / 2 + t), f"outside\n{H + 2*t:.0f}", off=(40, 0))
    dim(ax, (zback - 30, -ys[0] - t), (zback - 30, ys[0] + t), f"{2*(ys[0] + t):.0f}", off=(-30, 0))
    dim(ax, (zb, -H / 2 - 40), (zm, -H / 2 - 40), f"horn {lo.length*1000:.0f}")
    dim(ax, (zback, -H / 2 - 80), (zm, -H / 2 - 80), f"cabinet depth {zm - zback:.0f}")
    dim(ax, (zb + 14, -lo.throat_width * 500), (zb + 14, lo.throat_width * 500), f"throat {lo.throat_width*1000:.0f}",
        off=(48, 150))
    dim(ax, (lo.throat_z * 1000, H / 2 + 46), (st["mid_driver_back"], H / 2 + 46), "one-piece centre body")
    ax.text(zback + 10, 0, f"{D.LOW_REAR_LITRES:.0f} L sealed chamber\n(shaded, joined through the baffle ports)", fontsize=8,
            color=MUTED, rotation=90, ha="center", va="center", zorder=7)
    ax.text(zm - 10, H / 2 + 14, f"splay {lo.splay_deg:.1f} deg between stacked boxes", fontsize=8, color=MUTED, ha="right")
    ax.set(aspect="equal", xlim=(zback - 70, zm + 190), ylim=(-H / 2 - 115, H / 2 + 80),
           xlabel="z, mm from the 15 inch cone rim", title="Cabinet section through the axis (vertical), mm")
    ax.grid(False)
    fig.savefig(OUT / "low-drawing.png")
    plt.close(fig)


def tweeter_drawing():
    tw, m = D.TW, D.MID
    fig, ax = plt.subplots(figsize=(10, 4.6), layout="constrained")
    rb, d, dt = tw.body_radius * 1000, tw.body_depth * 1000, tw.depth_with_terminals * 1000
    ax.fill_between([-d, 0], [-rb, -rb], [rb, rb], color="#7b7f84", lw=0)
    ss = np.linspace(m.s_exit, m.s_tip, 60)
    hr = np.array([m.hf_horn_radius(s) for s in ss]) * 1000
    pr = np.array([m.plug(s) for s in ss]) * 1000
    x = (ss - m.s_exit) * 1000
    ax.fill_between(x, hr, pr, color=PLUG, lw=0)
    ax.fill_between(x, -hr, -pr, color=PLUG, lw=0)
    ax.fill_between([-dt, -d], [-rb * 0.5] * 2, [rb * 0.5] * 2, color="#c7a14a", lw=0)
    dim(ax, (-d - 14, -rb), (-d - 14, rb), f"body\n{2*rb:.0f}", off=(-22, 0))
    dim(ax, (-d, -rb - 12), (0, -rb - 12), f"{d:.0f}")
    dim(ax, (0, rb + 12), (x[-1], rb + 12), f"printed HF horn {x[-1]:.0f}")
    dim(ax, (x[-1] + 6, -hr[-1]), (x[-1] + 6, hr[-1]), f"{2*hr[-1]:.0f}", off=(16, 0))
    dim(ax, (2, -tw.exit_diameter * 500), (2, tw.exit_diameter * 500), f"exit {tw.exit_diameter*1000:.0f}", off=(18, -tw.exit_diameter * 500 - 10))
    ax.text(-dt, rb + 28, f"{tw.model}: body cylinder {2*rb:.0f} mm (flange ignored). Yellow: the printed HF horn, whose "
            "outside is the tapering tip of the mid phase plug.", fontsize=8, color=MUTED)
    ax.set(aspect="equal", xlim=(-dt - 40, x[-1] + 50), ylim=(-rb - 34, rb + 40), xlabel="mm from the HF exit",
           title="HF unit and its printed horn, section, mm")
    ax.grid(False)
    fig.savefig(OUT / "tweeter-drawing.png")
    plt.close(fig)


def area_figures():
    lo, m = D.LOW, D.MID
    fig, ax = plt.subplots(figsize=(11, 3.8), layout="constrained")
    z = np.linspace(lo.throat_z, lo.mouth_z, 800)
    x = (z - lo.throat_z) * 1000
    ax.plot(x, [lo.outer_area(v) * 1e4 for v in z], color=MUTED, lw=1.4, label="Inside the walls")
    ax.plot(x, [np.pi * lo.inner_radius(v) ** 2 * 1e4 for v in z], color=PRINT, lw=1.4, label="Centre body")
    ax.plot(x, [lo.area(v) * 1e4 for v in z], color=COL["low"], label=f"Open area ({law_name(lo.law)} law to the can)")
    ax.set(xlabel="mm from the low throat (baffle front)", ylabel="Area (cm²)", title="Low horn area", ylim=(0, None))
    ax.legend(loc="upper left")
    fig.savefig(OUT / "low-area.png")
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(11, 3.8), layout="constrained")
    zm = np.linspace(0, m.length, 400)
    ax.plot(zm * 1000, [np.pi * m.radius(v) ** 2 * 1e4 for v in zm], color=MUTED, lw=1.4, label="Inside the wall")
    ax.plot(zm * 1000, [np.pi * m.plug(v) ** 2 * 1e4 for v in zm], color=PLUG, lw=1.6, label="Phase plug")
    ax.plot(zm * 1000, [m.area(v) * 1e4 for v in zm], color=COL["mid"], label=f"Open area ({law_name(m.law)} law)")
    ax.set(xlabel="mm from the mid throat", ylabel="Area (cm²)", title="Mid horn area: wall minus plug", ylim=(0, None))
    ax.legend(loc="upper left")
    fig.savefig(OUT / "mid-area.png")
    plt.close(fig)


def chain_figure():
    rows = D.chain()
    fig, ax = plt.subplots(figsize=(12, 5.2), layout="constrained")
    labels = [r[0] for r in rows][::-1]
    vals = [r[1] for r in rows][::-1]
    notes = [r[2] for r in rows][::-1]
    cols = [COL["tweeter"]] * 2 + [COL["mid"]] * 5 + [COL["low"]] * 3
    ax.barh(labels, vals, color=cols[::-1], height=0.6)
    for i, (v, n) in enumerate(zip(vals, notes)):
        unit = "cm²" if "area" in labels[i] else "mm"
        ax.text(v + 8, i, f"{v:.0f} {unit}: {n}", va="center", fontsize=8.5, color=INK)
    ax.set(xlim=(0, max(vals) * 1.9), xlabel="mm (cm² for areas)", title="Designed from the inside out: each size sets the next")
    ax.grid(axis="y", visible=False)
    fig.savefig(OUT / "chain.png")
    plt.close(fig)


def hf_fem_figure():
    p = RES / "fem/hf-in-mid.csv"
    if not p.exists():
        return
    d = pd.read_csv(p)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.0), layout="constrained")
    ax = axes[0]
    ax.plot(d.frequency_hz, d.onaxis_vs_piston_db, color=COL["tweeter"])
    ax.axhline(0, color=MUTED, lw=0.8)
    ax.set(ylabel="dB", title="On axis: HF in the mid horn, relative to a free piston", ylim=(-25, 15))
    freq_axis(ax, d.frequency_hz.min(), 16000)
    ax = axes[1]
    for th, ls in ((15, "--"), (30, ":"), (60, (0, (1, 3)))):
        ax.plot(d.frequency_hz, d[f"fem_rel_{th}deg_db"], color=COL["tweeter"], ls=ls, label=f"FEM {th} deg")
        ax.plot(d.frequency_hz, d[f"piston_rel_{th}deg_db"], color=MUTED, ls=ls, lw=1.2, label=f"piston {th} deg")
    ax.set(ylabel="dB relative to on axis", title="Off axis: FEM against the piston model", ylim=(-30, 10))
    freq_axis(ax, d.frequency_hz.min(), 16000)
    ax.legend(loc="lower left", ncol=2, fontsize=8)
    fig.savefig(OUT / "hf-in-mid.png")
    plt.close(fig)


# ---------------------------------------------------------------- system
def full_range():
    co = cutoffs()
    nm = names()
    f1, f2 = co["crossover_hz"]
    sysd = pd.read_csv(RES / "planewave/system.csv")
    dsp = json.loads((RES / "planewave/dsp.json").read_text())
    t = dsp["target_db"]
    fig, axes = plt.subplots(2, 1, figsize=(13, 9.4), layout="constrained", sharex=True,
                             gridspec_kw=dict(height_ratios=(1.3, 1)))
    ax = axes[0]
    for k in ("low", "mid", "tweeter"):
        d = pd.read_csv(RES / f"planewave/{k}.csv")
        ax.plot(d.frequency_hz, d.spl_db + dsp["gain_db"][k], color=COL[k], lw=1.8, label=nm[k] + ", unfiltered")
    marks = [(co["low"]["f_low_3db"], "low -3 dB", COL["low"]),
             (co["low"]["mouth_cutoff_hz"], "low mouth cutoff", COL["low"]),
             (co["low"]["f_high_3db"], "low upper -3 dB", COL["low"]),
             (co["mid"]["f_low_3db"], "mid -3 dB", COL["mid"]),
             (co["mid"]["mouth_cutoff_hz"], "mid mouth cutoff", COL["mid"]),
             (co["mid"]["f_high_3db"], "mid upper -3 dB", COL["mid"]),
             (co["tweeter"]["datasheet_cutoff_hz"], "tweeter horn cutoff", COL["tweeter"]),
             (co["tweeter"]["f_low_3db"], "tweeter -3 dB", COL["tweeter"])]
    marks = sorted([mk for mk in marks if mk[0]], key=lambda x: x[0])
    for i, (v, lab, c) in enumerate(marks):
        ax.axvline(v, color=c, lw=0.9, ls=":")
        ax.text(v * 1.02, t + 9 - (i % 4) * 2.4, f"{lab} {v:.0f}", fontsize=8, color=c)
    for v in (f1, f2):
        ax.axvline(v, color=INK, lw=1.0, ls="--")
    ax.set(ylim=(t - 30, t + 10), ylabel="dB SPL, 2.83 V, 1 m, after gain",
           title="Each section on its own, level-matched, with the cutoffs that limit it. Dashed: crossovers")
    ax.legend(loc="lower center", ncol=3, fontsize=8)
    ax = axes[1]
    f = sysd.frequency_hz
    for k in ("low", "mid", "tweeter"):
        ax.plot(f, sysd[f"{k}_0deg_spl_db"], color=COL[k], lw=1.5)
    ax.plot(f, sysd["sum_0deg_spl_db"], color=INK, lw=2.4, label="Sum, on axis")
    ax.plot(f, sysd["sum_30deg_spl_db"], color=INK, lw=1.4, ls=":", label="Sum, 30 deg")
    if "sum_shelf_0deg_spl_db" in sysd:
        ax.plot(f, sysd["sum_shelf_0deg_spl_db"], color=COL["low"], lw=1.4, ls="--", label="No-sub shelf on the low channel")
    for v in (f1, f2):
        ax.axvline(v, color=INK, lw=1.0, ls="--")
        ax.text(v * 1.03, t - 27, f"crossover\n{v:.0f} Hz", fontsize=8, color=INK)
    ax.set(ylim=(t - 30, t + 8), ylabel="dB SPL, 2.83 V, 1 m",
           title="Through the LR4 crossover with the tuned gains, delays and polarity")
    freq_axis(ax, 20, 30000)
    ax.legend(loc="lower left")
    fig.savefig(OUT / "full-range.png")
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(13, 3.4), layout="constrained")
    gd = -np.gradient(np.unwrap(np.radians(sysd["sum_0deg_phase_deg"].values)), 2 * np.pi * f) * 1000
    ax.plot(f, gd, color=INK)
    ax.set(ylabel="Group delay (ms)", ylim=(0, 14), title="Group delay of the sum, on axis")
    freq_axis(ax, 20, 30000)
    fig.savefig(OUT / "full-range-group-delay.png")
    plt.close(fig)


def fem_compare():
    nm = names()
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2), layout="constrained")
    for ax, name in zip(axes, ("low", "mid")):
        d = pd.read_csv(RES / f"fem/{name}.csv")
        for th, ls in ((0, "-"), (30, ":")):
            ax.plot(d.frequency_hz, d[f"planewave_{th}deg_spl_db"], color=MUTED, ls=ls, lw=1.6, label=f"Plane-wave {th} deg")
            ax.plot(d.frequency_hz, d[f"fem_{th}deg_spl_db"], color=COL[name], ls=ls, lw=2.0, label=f"FEM {th} deg")
        ax.axvspan(*bands()["passband"][name], color=COL[name], alpha=0.06, lw=0)
        top = math.ceil(d.fem_0deg_spl_db.max() / 5) * 5 + 3
        ax.set(ylim=(top - 40, top), ylabel="dB SPL, 2.83 V, 1 m", title=f"{nm[name]}: FEM against plane-wave")
        freq_axis(ax, d.frequency_hz.min(), d.frequency_hz.max())
        ax.legend(loc="lower center", ncol=2, fontsize=8)
    fig.savefig(OUT / "fem-vs-planewave.png")
    plt.close(fig)


def tradeoff():
    from sunflower import tweeters as TWS
    sysd = pd.read_csv(RES / "optimise/system-search.csv")
    sel = D.SELECTED
    palette = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#6250d6"]
    fig, ax = plt.subplots(figsize=(12, 5), layout="constrained")
    for i, (tid, g) in enumerate(sysd.groupby("tweeter")):
        t = TWS.load(tid)
        price = f", ${t.price_usd:.0f}" if t.price_usd else ""
        ax.scatter(g.depth * 1000, g.score, s=34, color=palette[i % len(palette)], edgecolor="white", lw=0.6,
                   label=f"{t.model}{price}")
    ax.scatter([sel["cabinet"]["depth"] * 1000], [sel["scores"]["score"]], s=220, facecolor="none",
               edgecolor=INK, lw=2, label="Chosen", zorder=3)
    ax.set(xlabel="Cabinet depth (mm)", ylabel="System score, lower is better",
           title="Every finalist by HF unit: depth against performance")
    ax.legend(loc="upper right", fontsize=8)
    fig.savefig(OUT / "tradeoff.png")
    plt.close(fig)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    for old in OUT.glob("*.png"):
        old.unlink()
    cutoffs()
    for n in ("low", "mid", "tweeter"):
        section_figures(n)
    mid_drawing()
    low_drawing()
    tweeter_drawing()
    area_figures()
    full_range()
    fem_compare()
    tradeoff()
    chain_figure()
    hf_fem_figure()
    print("figures in", OUT)


if __name__ == "__main__":
    main()
