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
from sunflower.geometry import SQ2  # noqa: E402
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
    return dict(low=f"Kick: {s['kick']['driver']}, reversed", mid=f"Mid: {s['mid']['driver']}",
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
        ref = s[(f >= max(lo, 150)) & (f <= hi)].mean()
        below = f[(f < lo) & (s < ref - 3)]
        above = f[(f > hi) & (s < ref - 3)]
        out[name] = dict(f_low_3db=float(below.max()) if len(below) else None,
                         f_high_3db=float(above.min()) if len(above) else None, level_db=float(ref))
    k = D.KICK
    a_m = math.sqrt(k.wall_area(k.mouth_z) / math.pi)
    out["low"]["mouth_cutoff_hz"] = C / (2 * math.pi * a_m)
    drv = D.kick_driver()
    vas = 1.225 * C**2 * drv.sd**2 * drv.cms * 1000
    qts = 1 / (1 / (drv.fs * 2 * math.pi * drv.mms / drv.rms) + drv.re * 2 * math.pi * drv.fs * drv.mms / drv.bl**2)
    out["low"]["sealed_resonance_hz"] = drv.fs * math.sqrt(1 + vas / k.chamber_litres)
    out["low"]["sealed_q"] = qts * math.sqrt(1 + vas / k.chamber_litres)
    m = D.MID
    out["mid"]["mouth_cutoff_hz"] = C / (math.pi * m.mouth_diameter)
    out["mid"]["front_cavity_corner_hz"] = C * m.throat_area / (2 * math.pi * m.front_cavity_litres / 1000)
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
                                    ("datasheet_cutoff_hz", "datasheet cutoff"), ("sealed_resonance_hz", "sealed resonance"),
                                    ("front_cavity_corner_hz", "front cavity corner"))):
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
        rec = D.KICK_RECORD if name == "low" else D.MID_RECORD
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


def _body_outline():
    """Centre body silhouette (z, r) in mm: shroud underside, lip, law body, mid horn outside to the mid mouth."""
    k, m = D.KICK, D.MID
    pts = [(float(k.shroud_z(k.channel_start)) * 1000, (k.motor_body_radius - k.body_wall) * 1000)]
    rr = np.linspace(k.channel_start, k.drv.rim_r, 40)
    pts += [(float(k.shroud_z(r)) * 1000, r * 1000) for r in rr]
    r_lip = k.body_radius(k.turn_z)
    pts += [(k.turn_z * 1000, k.drv.rim_r * 1000), (k.turn_z * 1000, r_lip * 1000)]
    zz = np.linspace(k.turn_z, k.mouth_z, 160)
    pts += [(z * 1000, k.body_radius(z) * 1000) for z in zz]
    ss = np.linspace(k.mouth_z - k.mid_throat_z, m.length, 40)
    pts += [((k.mid_throat_z + s) * 1000, m.outer(s) * 1000) for s in ss]
    return np.array(pts)


def _driver_section(ax, zc, sgn, drv, colour, reversed_=True, face=0.0):
    """Cone, motor and flange of a driver; zc maps driver z (from its front face) to the drawing."""
    r = np.linspace(drv.vc_r, drv.rim_r, 60)
    z = drv.cone_z(r)
    ax.plot(zc(z) , sgn * r * 1000, color=colour, lw=2.6, zorder=8)
    zt = zc(np.array([drv.top_plate_z, drv.top_plate_z + drv.motor_h]))
    ax.fill_between(np.sort(zt), 0, sgn * drv.motor_r * 1000, color="#55595e", lw=0, zorder=7)
    zf = np.sort(zc(np.array([0.0, drv.flange_t])))
    ax.fill_between(zf, sgn * (drv.rim_r + 0.012) * 1000, sgn * drv.frame_od / 2 * 1000, color="#55595e", lw=0, zorder=7)


def kick_drawing():
    """Section through the axis. Upper half: vertical, through the flat top and its wedge. Lower half: horizontal,
    through an angled side wall."""
    k, m, drv = D.KICK, D.MID, D.KICK.drv
    fig, ax = plt.subplots(figsize=(13.5, 8.8), layout="constrained")
    zb = -(k.baffle + k.chamber_depth)
    t = k.ply / math.cos(math.radians(k.half_angle_deg))
    zz = np.linspace(zb, k.mouth_z, 80)
    hw = np.array([k.half_width(z) for z in zz])
    H = k.half_height
    Z = zz * 1000
    # upper half: flat top panel, sealed wedge from the baffle to the mouth, box behind the baffle full height
    ax.fill_between(Z, H * 1000, (H + k.ply) * 1000, color=WOOD, lw=0)
    zw = np.linspace(0, k.mouth_z, 60)
    ax.fill_between(zw * 1000, [k.half_width(z) * 1000 for z in zw], H * 1000, color="#b8925f", lw=0)
    # lower half: angled side wall
    ax.fill_between(Z, -hw * 1000, -(hw + t) * 1000, color=WOOD, lw=0)
    ax.fill_between([(zb - k.ply) * 1000, zb * 1000], [-(hw[0] + t) * 1000] * 2, [(H + k.ply) * 1000] * 2, color=WOOD, lw=0)
    zc_ = np.linspace(zb, -k.baffle, 10)
    ax.fill_between(zc_ * 1000, [-k.half_width(z) * 1000 for z in zc_], H * 1000, color="#ece9e1", lw=0, zorder=0)
    rc = drv.frame_od / 2 - 0.017
    ax.fill_between([-k.baffle * 1000, 0], [rc * 1000] * 2, [H * 1000] * 2, color=WOOD, lw=0)
    ax.fill_between([-k.baffle * 1000, 0], [-rc * 1000] * 2, [-k.half_width(0) * 1000] * 2, color=WOOD, lw=0)
    body = _body_outline()
    keep = body[:, 0] <= k.mouth_z * 1000 + 1e-6
    for sgn in (1, -1):
        ax.fill_between(body[keep, 0], 0, sgn * body[keep, 1], color=PRINT, lw=0, zorder=3)
    mz0 = float(k.shroud_z(k.channel_start)) * 1000
    for sgn in (1, -1):
        ax.fill_between([mz0, (k.motor_front - k.body_wall) * 1000], 0, sgn * (k.motor_body_radius - k.body_wall) * 1000,
                        color="#d9d6cf", lw=0, zorder=4)
        ax.fill_between([k.motor_front * 1000, k.mid_flange_z * 1000], 0, sgn * (m.pod_radius - m.wall) * 1000,
                        color="#d9d6cf", lw=0, zorder=4)
    ss = np.linspace(0, m.length, 200)
    zs = (k.mid_throat_z + ss) * 1000
    ax.fill_between(zs, [-m.radius(s) * 1000 for s in ss], [m.radius(s) * 1000 for s in ss], color="white", lw=0, zorder=5)
    pr = np.array([m.plug(s) for s in ss]) * 1000
    ax.fill_between(zs, -pr, pr, where=pr > 0, color="#3d3d3d", lw=0, zorder=6)
    for sgn in (1, -1):
        _driver_section(ax, lambda z: z * 1000, sgn, drv, COL["low"])
        _driver_section(ax, lambda z: (k.mid_flange_z - z) * 1000, sgn, m.drv, COL["mid"])
    zm, W = k.mouth_z * 1000, k.mouth_width * 1000
    top = (H + k.ply) * 1000
    xm = zm + 40
    dim(ax, (xm, 0), (xm, W / 2), f"mouth {W:.0f}\nsquare,\ninside", off=(44, 0))
    dim(ax, (xm, -k.external_mouth * 500), (xm, 0), f"outside\n{k.external_mouth*1000:.0f} wide", off=(44, 0))
    dim(ax, ((zb - k.ply) * 1000, top + 40), (zm, top + 40), f"cabinet depth {k.depth*1000:.0f}")
    dim(ax, (zb * 1000, -(hw[0] + t) * 1000 - 40), (-k.baffle * 1000, -(hw[0] + t) * 1000 - 40),
        f"chamber {k.chamber_depth*1000:.0f}\n{k.chamber_litres:.0f} L sealed", off=(0, -22))
    dim(ax, (-60, -k.throat_width * 500), (-60, k.throat_width * 500), "", fs=8)
    ax.text(-75, 30, f"throat\n{k.throat_width*1000:.0f}\nacross\nflats", ha="right", fontsize=8)
    ax.text(zm - 20, -(hw[-1] + t) * 1000 - 14, f"sides {k.half_angle_deg:.0f} deg: {k.splay_deg:.0f} deg between columns", fontsize=8,
            color=MUTED, ha="right", va="top")
    ax.text((zb - k.ply) * 1000, -(hw[-1] + t) * 1000 - 60,
            "Upper half: vertical section, through the flat top and its sealed wedge (darker). Lower half: horizontal section, "
            "through an angled side.\nBlack: one-piece printed centre body and the bullet holding the HF horn. Blue: the reversed "
            "15 inch cone and motor. Orange: the mid cone. The mid mouth sits in the kick mouth plane.", fontsize=8, color=MUTED, va="top")
    ax.set(aspect="equal", xlim=((zb - k.ply) * 1000 - 110, zm + 200), ylim=(-(hw[-1] + t) * 1000 - 110, top + 80),
           xlabel="z, mm from the baffle front face", title="Kick cabinet, section through the axis, mm")
    ax.grid(False)
    fig.savefig(OUT / "kick-drawing.png")
    plt.close(fig)


def throat_drawing():
    """The reversed 15 inch: cone, basket envelope, shroud gap and the turn into the horn."""
    k, drv = D.KICK, D.KICK.drv
    fig, ax = plt.subplots(figsize=(12, 6.2), layout="constrained")
    r = np.linspace(drv.vc_r, drv.rim_r, 120)
    ax.plot(r * 1000, drv.cone_z(r) * 1000, color=COL["low"], lw=3, label="Cone (estimated profile)")
    rs = np.linspace(k.channel_start, drv.rim_r, 80)
    sz = k.shroud_z(rs)
    body = _body_outline()
    keep = body[:, 0] < 230
    ax.fill_between(body[keep, 1], body[keep, 0], 230, color=PRINT, lw=0, zorder=1)
    ax.fill_betweenx([k.shroud_z(k.channel_start) * 1000, 230], 0, k.motor_body_radius * 1000, color=PRINT, lw=0, zorder=1)
    ax.fill_betweenx([drv.top_plate_z * 1000, (drv.top_plate_z + drv.motor_h) * 1000], 0, drv.motor_r * 1000,
                     color="#55595e", zorder=2, label="Motor")
    arm = drv.cone_z(rs) + k.arm / np.cos(np.arctan(drv.cone_slope(rs)))
    ax.plot(rs * 1000, arm * 1000, color=MUTED, ls="--", lw=1.4, label="Basket arms (estimated)")
    ax.fill_between([drv.rim_r * 1000 + 12, drv.frame_od * 500], 0, drv.flange_t * 1000, color="#55595e", lw=0)
    ax.fill_between([0, k.half_width(0) * 1000], -k.baffle * 1000, 0, color=WOOD, lw=0)
    ax.fill_between([0, (drv.frame_od / 2 - 0.017) * 1000], -k.baffle * 1000, 0, color="white", lw=0)
    for rv in (k.channel_start + 0.008, 0.5 * (k.channel_start + drv.rim_r), drv.rim_r - 0.006):
        z0 = float(drv.cone_z(rv))
        n = np.array([-float(drv.cone_slope(rv)), 1.0])
        n /= np.linalg.norm(n)
        g = float(k.gap(rv))
        p0 = np.array([rv, z0]) * 1000
        p1 = p0 + n * g * 1000
        dim(ax, tuple(p0), tuple(p1), f"{g*1000:.0f}", off=(10, 0), fs=8)
    W2 = k.half_width(0) * 1000
    ax.axvline(W2, color=WOOD, lw=4)
    ax.text(W2 + 4, 140, "wall flat", rotation=90, fontsize=8, color=MUTED)
    ax.text(drv.rim_r * 1000 + 10, k.turn_z * 1000 - 14, "turn under the lip", fontsize=8, color=INK)
    ax.text(10, -k.baffle * 1000 - 18, f"Channel area at each radius = cone area inside it / {k.compression:.1f}, less {k.blockage*100:.0f}% "
            "for the basket arms;\nnever closer to the cone than the arms plus 9 mm or the mechanical travel plus 3 mm. "
            f"Throat {k.throat_area*1e4:.0f} cm² at the rim.", fontsize=8, color=MUTED, va="top")
    ax.set(aspect="equal", xlim=(0, W2 + 30), ylim=(-k.baffle * 1000 - 45, 230), xlabel="radius (mm)",
           ylabel="z (mm from the baffle front)", title="Reversed 15 inch throat: cone, shroud and gap, mm")
    ax.legend(loc="center right", fontsize=8)
    ax.grid(False)
    fig.savefig(OUT / "throat-drawing.png")
    plt.close(fig)


def mid_drawing():
    m, tw = D.MID, D.TW
    fig, ax = plt.subplots(figsize=(12, 6.8), layout="constrained")
    L = m.length * 1000
    ss = np.linspace(0, m.length, 300)
    ro = np.array([m.radius(s) for s in ss]) * 1000
    rw = np.array([m.outer(s) for s in ss]) * 1000
    ri = np.array([m.plug(s) for s in ss]) * 1000
    z = ss * 1000
    for sgn in (1, -1):
        ax.fill_between(z, sgn * ro, sgn * rw, color=PRINT, lw=0)
        th = np.linspace(-math.pi / 2, math.pi / 2, 30)
        lr = m.lip_radius * 1000
        ax.fill(L + lr * np.cos(th) * 0 + lr * np.sin(th + math.pi / 2) * 0 + lr * np.cos(th),
                sgn * (ro[-1] + lr + lr * np.sin(th)), color=PRINT, lw=0)
    ax.fill_between(z, -ri, ri, where=ri > 0, color=PLUG, lw=0)
    rn = np.linspace(0, m.plug_r0, 30)
    sn = m.nose(rn) * 1000
    for sgn in (1, -1):
        ax.fill_betweenx(sgn * rn * 1000, sn, 0, color=PLUG, lw=0)
    rr = np.linspace(m.outer_throat_radius, m.drv.rim_r + 0.006, 30)
    for sgn in (1, -1):
        ax.fill_betweenx(sgn * rr * 1000, m.nose(np.minimum(rr, m.drv.rim_r)) * 1000, 0, color=PRINT, lw=0)
    sh = np.linspace(m.s_exit, m.s_tip, 60)
    hr = np.array([m.hf_horn_radius(s) for s in sh]) * 1000
    ax.fill_between(sh * 1000, -hr, hr, color="white", lw=0)
    rb = tw.body_radius * 1000
    ax.fill_between([(m.s_exit - tw.body_depth) * 1000, m.s_exit * 1000], [-rb, -rb], [rb, rb], color="#7b7f84", lw=0)
    rc = np.linspace(0, m.drv.rim_r, 80)
    for sgn in (1, -1):
        ax.plot(m.s_cone(rc) * 1000, sgn * rc * 1000, color=COL["mid"], lw=3)
    R = rw[-1] + m.lip_radius * 2000
    dim(ax, (0, -R - 10), (L, -R - 10), f"mid horn {L:.0f}")
    dim(ax, (L + 16, -m.mouth_diameter * 500), (L + 16, m.mouth_diameter * 500), f"mouth\n{m.mouth_diameter*1000:.0f}",
        off=(60, 0))
    rt = m.outer_throat_radius * 1000
    dim(ax, (-70, -rt), (-70, rt), f"throat\n{2*rt:.0f}", off=(-26, 0))
    dim(ax, (m.s_exit * 1000 - 18, -m.plug_r0 * 1000), (m.s_exit * 1000 - 18, m.plug_r0 * 1000),
        f"bullet {2*m.plug_r0*1000:.0f}", off=(0, -m.plug_r0 * 1000 - 14))
    dim(ax, (m.s_exit * 1000, R - 20), (m.s_tip * 1000, R - 20), f"printed HF horn {m.hf_len*1000:.0f}")
    ax.text(-110, R + 12, f"{law_name(m.law)} open-area law, {m.compression:.1f}:1 compression, cone-to-plug gap "
            f"{m.gap*1000:.1f} mm. Grey: {tw.model}, {tw.body_diameter*1000:.0f} mm body. Yellow: the bullet, one smooth "
            "taper, its open area always subtracted.", fontsize=8.5, color=MUTED)
    ax.set(aspect="equal", xlim=(-140, L + 150), ylim=(-R - 30, R + 30), xlabel="mm from the mid throat plane",
           title="Mid horn and phase plug, section through the axis, mm")
    ax.grid(False)
    fig.savefig(OUT / "mid-drawing.png")
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
    k, m, drv = D.KICK, D.MID, D.KICK.drv
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.2), layout="constrained", gridspec_kw=dict(width_ratios=(1, 2)))
    ax = axes[0]
    r = np.linspace(k.channel_start, drv.rim_r, 100)
    ax.plot(r * 1000, 2 * np.pi * r * k.gap(r) * (1 - k.blockage) * 1e4, color=COL["low"], label="Channel, open")
    ax.plot(r * 1000, np.pi * r**2 / k.compression * 1e4, color=MUTED, ls=":", label=f"Cone area inside r / {k.compression:.1f}")
    ax.set(xlabel="radius on the cone (mm)", ylabel="Area (cm²)", title="Over the reversed cone", ylim=(0, None))
    ax.legend(loc="upper left", fontsize=8)
    ax = axes[1]
    z = np.linspace(k.turn_z, k.mouth_z, 600)
    x = z * 1000
    ax.plot(x, [k.wall_area(v) * 1e4 for v in z], color=MUTED, lw=1.4, label="Inside the walls, less the fillers")
    ax.plot(x, [2 * k.filler(v) ** 2 * 1e4 for v in z], color="#b8925f", lw=1.4, label="Corner fillers")
    ax.plot(x, [np.pi * k.body_radius(v) ** 2 * 1e4 for v in z], color=PRINT, lw=1.4, label="Centre body")
    ax.plot(x, [k.area(v) * 1e4 for v in z], color=COL["low"], label=f"Open area ({law_name(k.law)} law)")
    p = RES / "cad/area-check.csv"
    if p.exists():
        d = pd.read_csv(p)
        if "kind" in d:
            d = d[d.kind == "kick"]
        zc = [c for c in d.columns if c.startswith("z")][0]
        ac = [c for c in d.columns if "cad" in c][0]
        scale = 1000 if d[zc].max() < 5 else 1
        sa = 1e4 if d[ac].max() < 5 else (1e-2 if d[ac].max() > 1e5 else 1)
        ax.plot(d[zc] * scale, d[ac] * sa, "o", ms=3.5, color=COL["low"], mfc="white", label="Sliced from the CAD air volume")
    ax.set(xlabel="z, mm from the baffle front face", ylabel="Area (cm²)", title="Forward kick horn", ylim=(0, None))
    ax.legend(loc="upper left", fontsize=8)
    fig.savefig(OUT / "low-area.png")
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(11, 3.8), layout="constrained")
    zm = np.linspace(0, m.length, 400)
    ax.plot(zm * 1000, [np.pi * m.radius(v) ** 2 * 1e4 for v in zm], color=MUTED, lw=1.4, label="Inside the wall")
    ax.plot(zm * 1000, [np.pi * m.plug(v) ** 2 * 1e4 for v in zm], color=PLUG, lw=1.6, label="Bullet")
    ax.plot(zm * 1000, [m.area(v) * 1e4 for v in zm], color=COL["mid"], label=f"Open area ({law_name(m.law)} law)")
    ax.set(xlabel="mm from the mid throat", ylabel="Area (cm²)", title="Mid horn area: wall minus the bullet", ylim=(0, None))
    ax.legend(loc="upper left")
    fig.savefig(OUT / "mid-area.png")
    plt.close(fig)


def chain_figure():
    rows = D.chain()
    fig, ax = plt.subplots(figsize=(12, 5.2), layout="constrained")
    labels = [r[0] for r in rows][::-1]
    vals = [r[1] for r in rows][::-1]
    notes = [r[2] for r in rows][::-1]
    cols = [COL["tweeter"]] * 2 + [COL["mid"]] * 4 + [COL["low"]] * 4
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
        if k == "low":
            d = d[d.frequency_hz <= 3000]
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


def kick_low_end():
    d = pd.read_csv(RES / "planewave/array.csv")
    dsp = json.loads((RES / "planewave/dsp.json").read_text())
    fig, ax = plt.subplots(figsize=(12, 4.4), layout="constrained")
    ref = d.single_free_db[(d.frequency_hz >= 150) & (d.frequency_hz <= 250)].mean()
    ax.plot(d.frequency_hz, d.cluster9_free_db - 20 * np.log10(9), color=COL["low"], lw=2.4,
            label="Per box, in a 3 x 3 cluster (array total less 19 dB)")
    ax.plot(d.frequency_hz, d.single_half_space_db, color=COL["low"], lw=1.6, ls="--", label="One box on the ground")
    ax.plot(d.frequency_hz, d.single_free_db, color=MUTED, lw=1.6, label="One box in free space")
    sh = dsp["no_sub_shelf"]
    ax.axvline(sh["f_minus6_hz"], color=MUTED, ls=":", lw=1)
    ax.text(sh["f_minus6_hz"] * 1.03, ref - 21, f"with the no-sub shelf, one box\nreaches -6 dB at {sh['f_minus6_hz']:.0f} Hz",
            fontsize=8, color=MUTED)
    for k, lab in (("single_free_db", "free"), ("single_half_space_db", "ground"), ("cluster9_free_db", "cluster")):
        v = dsp["kick_low_end"][k]["f_minus6_hz"]
        ax.plot([v], [dsp["kick_low_end"][k]["level_150_250_db"] - 6 - (20 * np.log10(9) if k.startswith("cluster") else 0)],
                "o", color=COL["low"] if k != "single_free_db" else MUTED, ms=5)
    ax.set(ylim=(ref - 30, ref + 10), ylabel="dB SPL, 2.83 V per box, 1 m",
           title="Kick low end: free space, on the ground, and coupled in a 3 x 3 cluster. Dots: -6 dB")
    freq_axis(ax, 30, 1000)
    ax.legend(loc="lower right", fontsize=8)
    fig.savefig(OUT / "kick-low-end.png")
    plt.close(fig)


def array_coverage():
    """Incoherent coverage of the 3 x 3 cluster in three bands, from the summed box polar and the splay."""
    sysd = pd.read_csv(RES / "planewave/system.csv")
    pol = {k: pd.read_csv(RES / f"planewave/{k}-polar.csv") for k in ("mid", "tweeter")}
    dsp = json.loads((RES / "planewave/dsp.json").read_text())
    k = D.KICK
    s = math.radians(k.splay_deg)
    f2 = D.CROSSOVER_MID_HIGH_HZ
    bands_ = (1000.0, 4000.0, 10000.0)
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.6), layout="constrained")
    az = np.radians(np.linspace(-80, 80, 161))
    AZ, EL = np.meshgrid(az, az)
    dirs = np.stack([np.sin(AZ) * np.cos(EL), np.sin(EL), np.cos(AZ) * np.cos(EL)], -1)
    for ax, fc in zip(axes, bands_):
        sec = "mid" if fc < f2 else "tweeter"
        p = pol[sec]
        i = int(np.argmin(np.abs(p.frequency_hz.values - fc)))
        ang = np.array([0, 15, 30, 45, 60, 90])
        lv = np.array([p[f"spl_{a}deg_db"].values[i] for a in ang[:-1]])
        lv = np.append(lv, lv[-1] - 6)
        lv -= lv[0]
        tot = np.zeros(AZ.shape)
        for bi in (-1, 0, 1):
            for bj in (-1, 0, 1):
                d = np.array([math.tan(bi * s), math.tan(bj * s), 1.0])
                d /= np.linalg.norm(d)
                th = np.degrees(np.arccos(np.clip(dirs @ d, -1, 1)))
                tot += 10 ** (np.interp(th, ang, lv, right=lv[-1] - 6) / 10)
        db = 10 * np.log10(tot / tot.max())
        cs = ax.contourf(np.degrees(az), np.degrees(az), db, levels=np.arange(-18, 0.1, 1.5), cmap="Greys_r", extend="min")
        ax.contour(np.degrees(az), np.degrees(az), db, levels=[-6], colors=[COL[sec]], linewidths=1.8)
        ax.set(aspect="equal", xlabel="Horizontal (deg)", ylabel="Vertical (deg)" if fc == bands_[0] else None,
               title=f"{fc/1000:g} kHz ({'mid' if sec == 'mid' else 'HF'})")
        ax.grid(False)
    fig.colorbar(cs, ax=axes, label="dB relative to the loudest direction, power sum")
    fig.suptitle(f"3 x 3 cluster at {k.splay_deg:.0f} deg splay: coverage map. Line: -6 dB", x=0.02, ha="left",
                 fontweight="bold", fontsize=11)
    fig.savefig(OUT / "array-coverage.png")
    plt.close(fig)


def fem_compare():
    nm = names()
    files = [(n, RES / f"fem/{n}-vs-network.csv") for n in ("kick", "mid")]
    files = [(n, p) for n, p in files if p.exists()]
    if not files:
        return
    fig, axes = plt.subplots(1, len(files), figsize=(6.2 * len(files), 4.2), layout="constrained", squeeze=False)
    for ax, (name, p) in zip(axes[0], files):
        d = pd.read_csv(p)
        key = "low" if name == "kick" else "mid"
        fcol = [c for c in d.columns if c.startswith("fem") and "spl" in c and ("0deg" in c or "deg" not in c)][0]
        ncol = [c for c in d.columns if c.startswith("network") and "spl" in c and ("0deg" in c or "deg" not in c)][0]
        ax.plot(d.frequency_hz, d[ncol], color=MUTED, lw=1.6, label="Network (plane-wave, sliced areas)")
        ax.plot(d.frequency_hz, d[fcol], color=COL[key], lw=2.0, label="FEM, cone as a vibrating boundary")
        ax.axvspan(*bands()["passband"][key], color=COL[key], alpha=0.06, lw=0)
        top = math.ceil(max(d[fcol].max(), d[ncol].max()) / 5) * 5 + 3
        ax.set(ylim=(top - 40, top), ylabel="dB SPL, 2.83 V, 1 m, half space", title=f"{nm[key]}: FEM against the network")
        freq_axis(ax, d.frequency_hz.min(), d.frequency_hz.max())
        ax.legend(loc="lower center", fontsize=8)
    fig.savefig(OUT / "fem-vs-planewave.png")
    plt.close(fig)


def tradeoff():
    kd = pd.read_csv(RES / "optimise/kick-search.csv")
    sel = D.SELECTED
    palette = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#6250d6", "#7b7f84"]
    fig, ax = plt.subplots(figsize=(12, 5), layout="constrained")
    best = kd.sort_values("score").drop_duplicates(["kick_driver", "half_angle_deg", "protrusion", "compression", "law"])
    for i, (kid, g) in enumerate(best.groupby("kick_driver")):
        rec = D.load_drivers("kick")[kid]
        ax.scatter(g.external_mouth * 1000, g.f6, s=26, color=palette[i % len(palette)], edgecolor="white", lw=0.5,
                   label=f"{rec['manufacturer']} {rec['model_name']}")
    k = D.KICK
    dsp = json.loads((RES / "planewave/dsp.json").read_text())
    ax.scatter([k.external_mouth * 1000], [dsp["kick_low_end"]["single_free_db"]["f_minus6_hz"]], s=220, facecolor="none",
               edgecolor=INK, lw=2, label="Chosen", zorder=3)
    ax.set(xlabel="Box width at the mouth, outside (mm)", ylabel="Kick -6 dB, one box in free space (Hz)",
           title="Every kick variant in a 600 mm deep box: size against low-end extension")
    ax.legend(loc="upper right", fontsize=8)
    fig.savefig(OUT / "tradeoff.png")
    plt.close(fig)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    for old in OUT.glob("*.png"):
        if old.name != "slices.png":          # written by sunflower.slicer from the CAD
            old.unlink()
    cutoffs()
    for n in ("low", "mid", "tweeter"):
        section_figures(n)
    mid_drawing()
    kick_drawing()
    throat_drawing()
    tweeter_drawing()
    area_figures()
    full_range()
    kick_low_end()
    array_coverage()
    fem_compare()
    tradeoff()
    chain_figure()
    hf_fem_figure()
    print("figures in", OUT)


if __name__ == "__main__":
    main()
