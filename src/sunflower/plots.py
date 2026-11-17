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

from sunflower import design as D  # noqa: E402
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
    fig, ax = plt.subplots(figsize=(12, 6.6), layout="constrained")
    L = m.length * 1000
    z = np.linspace(0, m.length, 200) * 1000
    r = np.array([m.radius(v / 1000) for v in z]) * 1000
    rc = m.can_diameter / 2 * 1000
    ax.fill_between(z, r, rc, color=PRINT, lw=0)
    ax.fill_between(z, -r, -rc, color=PRINT, lw=0)
    zf = m.tweeter_face * 1000
    rp = m.plug_radius * 1000
    ax.fill_between([-25, zf], [-rp, -rp], [rp, rp], color=PLUG, lw=0)
    rb = tw.body_radius * 1000
    ax.fill_between([zf - tw.body_depth * 1000, zf], [-rb, -rb], [rb, rb], color="#7b7f84", lw=0)
    rmo, hd = tw.mouth_diameter / 2 * 1000, tw.horn_depth * 1000
    ax.fill_between([zf - hd, zf], [-rmo * 0.35, -rmo], [rmo * 0.35, rmo], color="white", lw=0)
    ax.fill([zf - hd, zf - hd * 0.1, zf - hd], [-tw.bullet_base * 500, 0, tw.bullet_base * 500], color="#b0b4b8")
    rcone = math.sqrt(m.sd / math.pi) * 1000
    za = -m.adapter_length * 1000
    for sgn in (1, -1):
        ax.plot([za, za - 0.4 * rcone], [sgn * rcone, sgn * 0.3 * rcone], color=COL["mid"], lw=3)
    y0 = -rc - 22
    dim(ax, (0, y0), (L, y0), f"horn length {L:.0f}")
    dim(ax, (L + 22, -m.mouth_diameter * 500), (L + 22, m.mouth_diameter * 500), f"mouth\n{m.mouth_diameter*1000:.0f}", off=(24, 0))
    dim(ax, (L + 70, -rc), (L + 70, rc), f"can\n{rc*2:.0f}", off=(22, 0))
    rt = m.outer_throat_radius * 1000
    dim(ax, (-10, -rt), (-10, rt), f"throat\n{2*rt:.0f}", off=(-30, 0))
    dim(ax, (0, rc + 18), (zf, rc + 18), f"tweeter face {zf:.0f} from throat")
    dim(ax, (zf - tw.body_depth * 1000, -rb - 12), (zf, -rb - 12), f"tweeter {tw.body_depth*1000:.0f}", off=(0, -9))
    gap = (m.radius(m.tweeter_face) - m.plug_radius) * 1000
    ax.text(zf + 5, rp + 8, f"annulus {gap:.0f}", fontsize=8, color="white")
    ax.text(-60, rc + 42, f"{m.profile} profile, {m.compression_ratio:.1f}:1 compression. Phase plug {2*rp:.0f} mm: the "
            f"{tw.body_diameter*1000:.0f} mm tweeter in a 2 mm sleeve", fontsize=9, color=MUTED)
    ax.set(aspect="equal", xlim=(-80, L + 120), ylim=(-rc - 48, rc + 52), xlabel="mm from the mid throat",
           title="Mid horn, section through the axis, mm")
    ax.grid(False)
    fig.savefig(OUT / "mid-drawing.png")
    plt.close(fig)


def low_drawing():
    lo, m = D.LOW, D.MID
    st = json.loads((ROOT / "cad/cutlist.json").read_text())["stations_mm"]
    flare = json.loads((ROOT / "cad/cutlist.json").read_text())["panels"]["flare_half_angle_deg"]
    fig, ax = plt.subplots(figsize=(13, 7.6), layout="constrained")
    zb, zm = lo.baffle * 1000, lo.mouth_z * 1000
    t = D.PLY * 1000
    z = np.linspace(zb, zm, 100)
    w = np.array([lo.width(v / 1000) / 2 for v in z]) * 1000
    ax.fill_between(z, w, w + t, color=WOOD, lw=0)
    ax.fill_between(z, -w, -w - t, color=WOOD, lw=0)
    W, WT = lo.mouth_width * 1000, lo.throat_width * 1000
    half = W / 2
    ax.fill_between([0, zb], [-half] * 2, [-lo.cutout_radius * 1000] * 2, color=WOOD, lw=0)
    ax.fill_between([0, zb], [lo.cutout_radius * 1000] * 2, [half] * 2, color=WOOD, lw=0)
    zc = -D.REAR_CHAMBER_DEPTH * 1000
    ax.fill_between([zc, 0], [-half] * 2, [half] * 2, color="#ece9e1", lw=0)
    ax.text(zc / 2, 0, f"{D.LOW_REAR_LITRES:.0f} L sealed chamber", rotation=90, ha="center", va="center",
            fontsize=8, color=MUTED, zorder=6)
    ax.fill_between([zc - t, zc], [-half - t] * 2, [half + t] * 2, color=WOOD, lw=0)
    ax.fill_between([zc - t, zm], [half, half], [half + t, half + t], color=WOOD, lw=0, alpha=0.5)
    ax.fill_between([zc - t, zm], [-half - t, -half - t], [-half, -half], color=WOOD, lw=0, alpha=0.5)
    zi = np.linspace(lo.throat_z, lo.mouth_z, 900)
    ri = np.array([lo.inner_radius(v) for v in zi]) * 1000
    ax.fill_between(zi * 1000, -ri, ri, where=ri > 0, color=PRINT, lw=0)
    a = lo.body.cone_radius * 1000
    for sgn in (1, -1):
        ax.plot([0, lo.body.cone_depth * 1000], [sgn * a, sgn * 40], color=COL["low"], lw=3)
    zt = st["mid_throat"]
    zz = np.linspace(zt, zm, 50)
    rr = np.array([m.radius((v - zt) / 1000) for v in zz]) * 1000
    ax.fill_between(zz, -rr, rr, color="white", lw=0)
    ax.fill_between([zt - 20, st["tweeter_face"]], [-m.plug_radius * 1000] * 2, [m.plug_radius * 1000] * 2, color=PLUG, lw=0)
    zr = st["support_rods"]
    wr = lo.width(zr / 1000) / 2 * 1000
    for sgn in (1, -1):
        ax.plot([zr, zr], [sgn * m.can_diameter * 500, sgn * wr], color="#9aa0a6", lw=2.5)
    dim(ax, (zm + 34, -half), (zm + 34, half), f"mouth\n{W:.0f}\nsquare", off=(40, 0))
    dim(ax, (zm + 100, -half - t), (zm + 100, half + t), f"outside\n{W + 2*t:.0f}", off=(40, 0))
    dim(ax, (zb + 14, -WT / 2), (zb + 14, WT / 2), f"throat {WT:.0f}\nacross flats", off=(52, 150))
    dim(ax, (zb, -half - 40), (zm, -half - 40), f"horn {lo.length*1000:.0f}")
    dim(ax, (zc - t, -half - 80), (zm, -half - 80), f"cabinet depth {zm - zc + t:.0f}")
    dim(ax, (zc, -half - 40), (0, -half - 40), f"{-zc:.0f}", off=(0, -14))
    zt0 = st["mid_tail_tip"]
    dim(ax, (zt0, half + 40), (zm, half + 40), f"mid can with tail {zm - zt0:.0f}")
    dim(ax, (zm - 14, -m.can_diameter * 500), (zm - 14, m.can_diameter * 500), f"{m.can_diameter*1000:.0f}", off=(-26, 0))
    dim(ax, (0, half + 80), (lo.nose_end * 1000, half + 80), f"15 in and nose {lo.nose_end*1000:.0f}")
    ax.text(zb + 40, half - 40, f"walls flare {flare:.1f} deg", fontsize=8, color=MUTED)
    ax.set(aspect="equal", xlim=(zc - 50, zm + 190), ylim=(-half - 115, half + 110),
           xlabel="z, mm from the 15 inch cone rim", title="Cabinet section through the axis, mm")
    ax.grid(False)
    fig.savefig(OUT / "low-drawing.png")
    plt.close(fig)


def tweeter_drawing():
    tw = D.TW
    fig, ax = plt.subplots(figsize=(9, 4.8), layout="constrained")
    rb, d, dt = tw.body_radius * 1000, tw.body_depth * 1000, tw.depth_with_terminals * 1000
    rm, hd, bb = tw.mouth_diameter / 2 * 1000, tw.horn_depth * 1000, tw.bullet_base / 2 * 1000
    ax.fill_between([0, d], [-rb, -rb], [rb, rb], color="#7b7f84", lw=0)
    t = np.linspace(0, 1, 30)
    zz = d - hd * t
    rr = rm * np.exp(np.log((bb + 3) / rm) * t)
    ax.fill_between(zz, -rr, rr, color="white", lw=0)
    ax.fill([d - hd, d - 0.1 * hd, d - hd], [-bb, 0, bb], color="#b0b4b8")
    for sgn in (1, -1):
        ax.fill_between([d - dt, 0], [sgn * rb * 0.4] * 2, [sgn * rb * 0.6] * 2, color="#c7a14a", lw=0)
    dim(ax, (d + 14, -rb), (d + 14, rb), f"body\n{2*rb:.0f}", off=(18, 0))
    dim(ax, (d + 4, -rm), (d + 4, rm), f"mouth {2*rm:.0f}", off=(-22, -rm - 7))
    dim(ax, (0, -rb - 10), (d, -rb - 10), f"body {d:.1f}")
    dim(ax, (d - dt, -rb - 26), (d, -rb - 26), f"with terminals {dt:.1f}")
    dim(ax, (d - hd, rb + 8), (d, rb + 8), f"horn {hd:.0f}")
    ax.text(d - dt, rb + 22, f"{tw.model}. Body sizes from the datasheet drawing; horn mouth, bullet and horn depth "
            "scaled from photographs.", fontsize=8, color=MUTED)
    ax.set(aspect="equal", xlim=(d - dt - 10, d + 50), ylim=(-rb - 40, rb + 32), xlabel="mm",
           title="Bullet tweeter, section, mm")
    ax.grid(False)
    fig.savefig(OUT / "tweeter-drawing.png")
    plt.close(fig)


def area_figures():
    lo, m = D.LOW, D.MID
    fig, ax = plt.subplots(figsize=(11, 3.8), layout="constrained")
    z = np.linspace(lo.throat_z, lo.mouth_z, 800)
    ax.plot((z - lo.throat_z) * 1000, [lo.outer_area(v) * 1e4 for v in z], color=MUTED, lw=1.4, label="Walls only")
    ax.plot((z - lo.throat_z) * 1000, [lo.area(v) * 1e4 for v in z], color=COL["low"], label="Open area, walls minus centre body")
    ax.set(xlabel="mm from the 15 inch voice-coil plane", ylabel="Area (cm²)", title="Low horn area", ylim=(0, None))
    ax.legend(loc="upper left")
    fig.savefig(OUT / "low-area.png")
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(11, 3.8), layout="constrained")
    zm = np.linspace(0, m.length, 400)
    ax.plot(zm * 1000, [m.area(v) * 1e4 for v in zm], color=COL["mid"], label="Open area around the tweeter")
    ax.axvline(m.tweeter_face * 1000, color=MUTED, ls=":", lw=0.8)
    ax.text(m.tweeter_face * 1000 + 3, 5, "tweeter face", color=MUTED, fontsize=8)
    ax.set(xlabel="mm from the mid throat", ylabel="Area (cm²)", title="Mid horn area", ylim=(0, None))
    ax.legend(loc="upper left")
    fig.savefig(OUT / "mid-area.png")
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
    mids = pd.read_csv(RES / "optimise/mid-shortlist.csv").set_index("key")
    sel = D.SELECTED
    sysd["tweeter"] = mids.loc[sysd.mid_key, "tweeter"].values
    fig, ax = plt.subplots(figsize=(12, 5), layout="constrained")
    tcol = {"fostex-t90a": "#2a78d6", "fostex-t925a": "#eb6834", "jbl-2402h": "#1baf7a"}
    for tid, g in sysd.groupby("tweeter"):
        ax.scatter(g.volume_m3 * 1000, g.score, s=30, color=tcol[tid], edgecolor="white", lw=0.6,
                   label=f"with the {TWS.load(tid).model}")
    ax.scatter([sel["cabinet"]["volume_m3"] * 1000], [sel["scores"]["score"]], s=200, facecolor="none",
               edgecolor=INK, lw=2, label="Chosen", zorder=3)
    lim = sysd[(sysd.width <= 0.85) & (sysd.depth <= 1.2)].volume_m3.max() * 1000
    ax.axvline(lim, color=MUTED, ls=":", lw=1)
    ax.set(xlabel="External cabinet volume (litres)", ylabel="System score, lower is better",
           title="Every finalist: size against performance")
    ax.legend(loc="upper right")
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
    print("figures in", OUT)


if __name__ == "__main__":
    main()
