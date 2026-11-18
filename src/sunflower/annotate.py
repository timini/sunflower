"""Dimension lines on the orthographic renders.

render.py writes docs/img/renders/cameras.json with the orthographic scale
and centre of each view, so CAD millimetres map exactly to pixels.
"""
from __future__ import annotations

import json
import math

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from sunflower import design as D  # noqa: E402

ROOT = D.ROOT
REN = ROOT / "docs/img/renders"
INK = "#111111"
ACCENT = "#b04a00"


def _setup(name):
    cams = json.loads((REN / "cameras.json").read_text())
    cam = cams[name]
    img = plt.imread(REN / f"{name}.jpg")
    w, h = cam["res"]
    s = cam["ortho"] * 1000 / max(w, h)   # mm per pixel; Blender's ortho scale spans the wider side
    fig = plt.figure(figsize=(w / 150, h / 150), dpi=150)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.imshow(img)
    ax.set_xlim(0, w)
    ax.set_ylim(h, 0)
    ax.set_axis_off()

    def px(hmm, vmm):
        return w / 2 + (hmm - cam["centre_h"]) / s, h / 2 - (vmm - cam["centre_v"]) / s
    return fig, ax, px


def _label(ax, x, y, text, color=INK, size=11):
    ax.text(x, y, text, ha="center", va="center", fontsize=size, color=color,
            bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="#bbbbbb", lw=0.6, alpha=0.95))


def _dim(ax, a, b, text, off=(0, 0), color=INK):
    ax.annotate("", xy=b, xytext=a, arrowprops=dict(arrowstyle="<->", color=color, lw=1.4, shrinkA=0, shrinkB=0))
    _label(ax, (a[0] + b[0]) / 2 + off[0], (a[1] + b[1]) / 2 + off[1], text, color)


def _line(ax, a, b, color=INK, ls="-", lw=0.8):
    ax.plot([a[0], b[0]], [a[1], b[1]], color=color, ls=ls, lw=lw)


def _arc(ax, P, centre, r, t0, t1, color=ACCENT):
    """Arc in CAD (h, v) millimetres about centre, angles in degrees from +h."""
    n = 40
    pts = [P(centre[0] + r * math.cos(math.radians(t0 + (t1 - t0) * i / n)),
             centre[1] + r * math.sin(math.radians(t0 + (t1 - t0) * i / n))) for i in range(n + 1)]
    ax.plot([p[0] for p in pts], [p[1] for p in pts], color=color, lw=1.4)


def front():
    """Square mouth, inside and outside, with the mid and HF mouths."""
    cut = json.loads((ROOT / "cad/cutlist.json").read_text())
    m, k = D.MID, D.KICK
    fig, ax, px = _setup("front")
    ex = cut["external_mm"]
    we, he = ex["width"] / 2, ex["height_front"] / 2
    wi = k.summary()["mouth_width_m"] * 500               # inside of the mouth, ply faces
    _dim(ax, px(-we, he + 45), px(we, he + 45), f"{2*we:.0f} outside")
    _dim(ax, px(we + 45, -he), px(we + 45, he), f"{2*he:.0f}\noutside", off=(48, 0))
    _line(ax, px(-wi, -wi), px(-wi, -he - 55), lw=0.7)
    _line(ax, px(wi, -wi), px(wi, -he - 55), lw=0.7)
    _dim(ax, px(-wi, -he - 45), px(wi, -he - 45), f"mouth {2*wi:.0f} inside")
    _line(ax, px(-wi, -wi), px(-he - 55, -wi), lw=0.7)
    _line(ax, px(-wi, wi), px(-he - 55, wi), lw=0.7)
    _dim(ax, px(-we - 45, -wi), px(-we - 45, wi), f"{2*wi:.0f}\ninside,\nsquare", off=(-44, 0))
    rm = m.mouth_diameter * 500
    _line(ax, px(-rm, 0), px(-rm, 300), lw=0.7)
    _line(ax, px(rm, 0), px(rm, 300), lw=0.7)
    _dim(ax, px(-rm, 290), px(rm, 290), f"mid mouth {2*rm:.0f}")
    rh = m.hf_mouth_radius * 1000
    _dim(ax, px(-rh, -rh - 30), px(rh, -rh - 30), f"HF mouth {2*rh:.0f}", off=(0, 26))
    fig.savefig(REN / "front-dims.jpg", dpi=150, pil_kwargs=dict(quality=88))
    plt.close(fig)


def section_side():
    """Depth, the equal front and back heights, and the splay of the wedge faces."""
    cut = json.loads((ROOT / "cad/cutlist.json").read_text())
    st = cut["stations_mm"]
    ex = cut["external_mm"]
    wd = cut["wedges"]
    s = D.KICK.summary()
    fig, ax, px = _setup("section-side")
    P = lambda z, y: px(-z, y)  # noqa: E731   mouth on the left
    he, hb = ex["height_front"] / 2, ex["height_back"] / 2
    z_back, z_mouth = st["back"], st["mouth"]
    _dim(ax, P(z_back, -he - 50), P(z_mouth, -he - 50), f"depth {ex['depth']:.0f}")
    _line(ax, P(z_mouth, -he - 60), P(z_mouth, -he), lw=0.7, ls=":")
    _line(ax, P(z_back, -he - 60), P(z_back, -hb), lw=0.7, ls=":")
    _dim(ax, P(z_mouth + 40, -he), P(z_mouth + 40, he), f"{2*he:.0f}", off=(-36, 0))
    _dim(ax, P(z_back - 40, -hb), P(z_back - 40, hb), f"{2*hb:.0f}", off=(36, 0))
    # splay: the bottom wedge face against a line parallel to the axis
    half = s["splay_deg"] / 2
    yi = s["mouth_width_m"] * 500                          # inside half height at the mouth
    y0 = yi - wd["depth_at_baffle_mm"]                     # wedge face at the baffle front
    z0 = st["baffle_front"]
    _line(ax, P(z0, -y0), P(z_mouth, -y0), ls="--", color=ACCENT, lw=1.0)
    r = 0.85 * (z_mouth - z0)
    _arc(ax, lambda h, v: px(h, v), (-z0, -y0), r, 180, 180 + half)
    zl = z0 + 0.5 * r                                      # label on the wedge, inside the arc
    lx, ly = P(zl, -y0 - (zl - z0) * math.tan(math.radians(half)) - 45)
    _label(ax, lx, ly, f"wedge face {half:.0f} deg per wall, splay {s['splay_deg']:.0f} deg\n"
           "(the side walls the same)", color=ACCENT, size=10)
    fig.savefig(REN / "section-side-dims.jpg", dpi=150, pil_kwargs=dict(quality=88))
    plt.close(fig)


def stack_side():
    """Centre column of the 3 x 3 array from the side: rows stacked flat."""
    cams = json.loads((REN / "cameras.json").read_text())
    if "stack-side" not in cams or not (REN / "stack-side.jpg").exists():
        return
    arr = json.loads((ROOT / "cad/array.json").read_text())
    ex = json.loads((ROOT / "cad/cutlist.json").read_text())["external_mm"]
    fig, ax, px = _setup("stack-side")
    P = lambda z, y: px(-z, y)  # noqa: E731
    pitch = arr["row_pitch_mm"]
    z_m = json.loads((ROOT / "cad/cutlist.json").read_text())["stations_mm"]["mouth"]
    for j in (-1, 0, 1):
        _dim(ax, P(z_m + 60, j * pitch - ex["height_front"] / 2), P(z_m + 60, j * pitch + ex["height_front"] / 2),
             f"{ex['height_front']:.0f}", off=(-36, 0))
    w, h = cams["stack-side"]["res"]
    ax.text(w * 0.03, h - 16, f"Rows stacked flat at {pitch:.0f} mm pitch: no tilt between rows.\n"
            "Centre column, cut on its axis.", fontsize=11, color=INK, va="bottom",
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#bbbbbb"))
    fig.savefig(REN / "stack-side-dims.jpg", dpi=150, pil_kwargs=dict(quality=88))
    plt.close(fig)


def stack_top():
    """Plan of the 3 x 3 array: the three columns fan about a vertical line behind the centre box."""
    cams = json.loads((REN / "cameras.json").read_text())
    if "stack-top" not in cams or not (REN / "stack-top.jpg").exists():
        return
    arr = json.loads((ROOT / "cad/array.json").read_text())
    st = json.loads((ROOT / "cad/cutlist.json").read_text())["stations_mm"]
    fig, ax, px = _setup("stack-top")
    P = lambda x, z: px(x, -z)  # noqa: E731   plan, mouths at the bottom
    apex_z = -arr["apex_behind_baffle_mm"]
    reach = arr["apex_behind_baffle_mm"] + st["mouth"] + 160
    angles = []
    for b in arr["boxes"]:
        if b["j"] != 1:
            continue
        M = b["matrix"]
        axx, axz = M[0][2], M[2][2]                        # box +z axis in the array frame
        _line(ax, P(0, apex_z), P(reach * axx, apex_z + reach * axz), color=ACCENT, ls="--", lw=1.0)
        angles.append(math.degrees(math.atan2(-axz, axx)))  # in plan (h, v) about the apex
    angles.sort()
    r = reach - 60
    for a0, a1 in zip(angles, angles[1:]):
        _arc(ax, px, (0, -apex_z), r, a0, a1)
        t = math.radians((a0 + a1) / 2)
        _label(ax, *px((r + 80) * math.cos(t), -apex_z + (r + 80) * math.sin(t)), f"{a1 - a0:.0f} deg",
               color=ACCENT)
    ax.plot(*P(0, apex_z), "o", color=ACCENT, ms=5)
    _label(ax, *px(0, -apex_z + 70), f"axes meet {arr['apex_behind_baffle_mm']:.0f} mm\nbehind the centre baffle",
           size=10)
    w = cams["stack-top"]["res"][0]
    ax.text(w * 0.03, 50, f"3 x 3 array from above: columns {arr['splay_deg']:.0f} degrees apart, "
            f"{2*arr['splay_deg']:.0f} degrees side to side.\nNine identical boxes; side faces meet exactly, no spacers.",
            fontsize=11, color=INK, va="top", bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#bbbbbb"))
    fig.savefig(REN / "stack-top-dims.jpg", dpi=150, pil_kwargs=dict(quality=88))
    plt.close(fig)


def main():
    front()
    section_side()
    stack_side()
    stack_top()
    print("dimensioned renders written")


if __name__ == "__main__":
    main()
