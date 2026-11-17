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
    cut = json.loads((ROOT / "cad/cutlist.json").read_text())
    m = D.MID
    fig, ax, px = _setup("front")
    ex = cut["external_mm"]
    we, he = ex["width"] / 2, ex["height_front"] / 2
    wi = cut["walls"]["inner_face"]["mouth_edge"] / 2       # inside of the mouth, ply faces
    _dim(ax, px(-we, he + 45), px(we, he + 45), f"{2*we:.0f} outside")
    _dim(ax, px(we + 45, -he), px(we + 45, he), f"{2*he:.0f}\nsquare", off=(48, 0))
    _line(ax, px(-wi, -wi), px(-wi, -he - 55), lw=0.7)
    _line(ax, px(wi, -wi), px(wi, -he - 55), lw=0.7)
    _dim(ax, px(-wi, -he - 45), px(wi, -he - 45), f"mouth {2*wi:.0f} inside, square")
    rm = m.mouth_diameter * 500
    _line(ax, px(-rm, 0), px(-rm, 300), lw=0.7)
    _line(ax, px(rm, 0), px(rm, 300), lw=0.7)
    _dim(ax, px(-rm, 290), px(rm, 290), f"mid mouth {2*rm:.0f}")
    rh = m.hf_mouth_radius * 1000
    _dim(ax, px(-rh, -rh - 30), px(rh, -rh - 30), f"HF mouth {2*rh:.0f}", off=(0, 26))
    fig.savefig(REN / "front-dims.jpg", dpi=150, pil_kwargs=dict(quality=88))
    plt.close(fig)


def section_side():
    cut = json.loads((ROOT / "cad/cutlist.json").read_text())
    st = cut["stations_mm"]
    ex = cut["external_mm"]
    k = D.KICK
    s = k.summary()
    fig, ax, px = _setup("section-side")
    P = lambda z, y: px(-z, y)  # noqa: E731   mouth on the left
    he, hb = ex["height_front"] / 2, ex["height_back"] / 2
    z_back, z_mouth = st["back"], st["mouth"]
    z_front = z_back + ex["overall_depth"]
    lip = D.MID.mouth_diameter * 500 + 40
    _dim(ax, P(z_back, -he - 50), P(z_mouth, -he - 50), f"cabinet depth {ex['depth']:.0f}")
    _dim(ax, P(z_back, he + 55), P(z_front, he + 55), f"overall {ex['overall_depth']:.0f}")
    _line(ax, P(z_front, he + 65), P(z_front, lip - 40), lw=0.7)
    _line(ax, P(z_mouth, -he - 60), P(z_mouth, he + 10), lw=0.7, ls=":")
    _line(ax, P(st["mid_mouth"], lip + 20), P(st["mid_mouth"], -lip), lw=0.7, ls=":")
    _dim(ax, P(z_mouth, lip + 10), P(st["mid_mouth"], lip + 10), f"protrudes {k.protrusion*1000:.0f}", off=(0, -22))
    _dim(ax, P(z_mouth + 40, -he), P(z_mouth + 40, he), f"{2*he:.0f}", off=(-36, (P(0, -0.8 * he)[1] - P(0, 0)[1])))
    _dim(ax, P(z_back - 40, -hb), P(z_back - 40, hb), f"{2*hb:.0f}", off=(36, 0))
    # splay: the bottom wall against a line parallel to the axis
    half = s["splay_deg"] / 2
    pivot = (-z_back, -hb)                      # (h, v) in mm at the back bottom corner
    _line(ax, P(z_back, -hb), P(z_mouth, -hb), ls="--", color=ACCENT, lw=1.0)
    r = 0.8 * (z_mouth - z_back)
    _arc(ax, lambda h, v: px(h, v), pivot, r, 180, 180 + half)
    mid_t = math.radians(180 + half / 2)
    lx, ly = px(pivot[0] + (r + 70) * math.cos(mid_t), pivot[1] + (r + 70) * math.sin(mid_t))
    _label(ax, lx, ly, f"{half:.0f} deg per wall,\nsplay {s['splay_deg']:.0f} deg", color=ACCENT, size=10)
    fig.savefig(REN / "section-side-dims.jpg", dpi=150, pil_kwargs=dict(quality=88))
    plt.close(fig)


def stack_side():
    """Centre column of the 3 x 3 array from the side: box axes from the common apex and the splay."""
    cams = json.loads((REN / "cameras.json").read_text())
    if "stack-side" not in cams or not (REN / "stack-side.jpg").exists():
        return
    arr = json.loads((ROOT / "cad/array.json").read_text())
    fig, ax, px = _setup("stack-side")
    P = lambda z, y: px(-z, y)  # noqa: E731
    apex = (0.0, -arr["apex_behind_baffle_mm"])           # (y, z) mm in the array frame
    reach = arr["apex_behind_baffle_mm"] + 790
    angles = []
    for b in arr["boxes"]:
        if b["i"] != 0:
            continue
        M = b["matrix"]
        ay, az = M[1][2], M[2][2]                          # box +z axis in the array frame
        oy, oz = M[1][3] * 1000, M[2][3] * 1000
        _line(ax, P(oz - 120 * az, oy - 120 * ay), P(apex[1] + reach * az, apex[0] + reach * ay),
              color=ACCENT, ls="--", lw=1.0)
        angles.append(math.degrees(math.atan2(ay, az)))
    angles.sort()
    r = reach - 40
    Q = P                                                 # arcs in (z, y) about the apex
    for a0, a1 in zip(angles, angles[1:]):
        _arc(ax, Q, (apex[1], apex[0]), r, a0, a1)
        t = math.radians((a0 + a1) / 2)
        _label(ax, *Q(apex[1] + (r + 75) * math.cos(t), apex[0] + (r + 75) * math.sin(t)),
               f"{a1 - a0:.0f} deg", color=ACCENT)
    w = cams["stack-side"]["res"][0]
    ax.text(w * 0.03, 50, f"3 x 3 array, {arr['splay_deg']:.0f} degrees between neighbours, "
            f"{2*arr['splay_deg']:.0f} degrees top to bottom and side to side.\n"
            f"Axes meet {arr['apex_behind_baffle_mm']:.0f} mm behind the centre baffle. Centre column shown.",
            fontsize=11, color=INK, va="top", bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#bbbbbb"))
    fig.savefig(REN / "stack-side-dims.jpg", dpi=150, pil_kwargs=dict(quality=88))
    plt.close(fig)


def main():
    front()
    section_side()
    stack_side()
    print("dimensioned renders written")


if __name__ == "__main__":
    main()
