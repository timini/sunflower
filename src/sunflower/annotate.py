"""Dimension lines on the orthographic renders.

render.py writes docs/img/renders/cameras.json with the orthographic scale
of each view, so CAD millimetres map exactly to pixels.
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


def _setup(name):
    cams = json.loads((REN / "cameras.json").read_text())
    cam = cams[name]
    img = plt.imread(REN / f"{name}.jpg")
    w, h = cam["res"]
    s = cam["ortho"] * 1000 / w           # mm per pixel
    fig = plt.figure(figsize=(w / 150, h / 150), dpi=150)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.imshow(img)
    ax.set_axis_off()

    def px(hmm, vmm):
        return w / 2 + (hmm - cam["centre_h"]) / s, h / 2 - (vmm - cam["centre_v"]) / s
    return fig, ax, px


def _dim(ax, a, b, text, off=(0, 0), color=INK):
    ax.annotate("", xy=b, xytext=a, arrowprops=dict(arrowstyle="<->", color=color, lw=1.4, shrinkA=0, shrinkB=0))
    ax.text((a[0] + b[0]) / 2 + off[0], (a[1] + b[1]) / 2 + off[1], text, ha="center", va="center", fontsize=11,
            color=color, bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="#bbbbbb", lw=0.6, alpha=0.95))


def front():
    cut = json.loads((ROOT / "cad/cutlist.json").read_text())
    lo, m, tw = D.LOW, D.MID, D.TW
    fig, ax, px = _setup("front")
    ex = cut["external_mm"]
    we, he = ex["width"] / 2, ex["height_front"] / 2
    wm, hm = lo.mouth_width * 500, lo.mouth_height * 500
    _dim(ax, px(-we, he + 40), px(we, he + 40), f"{2*we:.0f} outside")
    _dim(ax, px(we + 40, -he), px(we + 40, he), f"{2*he:.0f}\noutside", off=(46, 0))
    _dim(ax, px(-wm, -hm - 30), px(wm, -hm - 30), f"mouth {2*wm:.0f} x {2*hm:.0f}")
    rm = m.mouth_diameter * 500
    _dim(ax, px(-rm, rm + 25), px(rm, rm + 25), f"mid mouth {2*rm:.0f}")
    rc = m.can_diameter * 500
    _dim(ax, px(-rc - 25, -rc), px(-rc - 25, rc), f"can\n{2*rc:.0f}", off=(-42, 0))
    rh = m.hf_mouth_radius * 1000
    _dim(ax, px(-rh, -rh - 22), px(rh, -rh - 22), f"HF mouth {2*rh:.0f}", off=(0, 24))
    fig.savefig(REN / "front-dims.jpg", dpi=150, pil_kwargs=dict(quality=88))
    plt.close(fig)


def section_side():
    cut = json.loads((ROOT / "cad/cutlist.json").read_text())
    st = cut["stations_mm"]
    lo, m = D.LOW, D.MID
    fig, ax, px = _setup("section-side")
    P = lambda z, y: px(-z, y)  # noqa: E731
    he = cut["external_mm"]["height_front"] / 2
    hb = cut["external_mm"]["height_back"] / 2
    _dim(ax, P(st["back"], -he - 50), P(st["mouth"], -he - 50), f"cabinet depth {cut['external_mm']['depth']:.0f}")
    _dim(ax, P(st["baffle_front"], he + 45), P(st["mouth"], he + 45), f"kick horn {lo.length*1000:.0f}")
    _dim(ax, P(st["mid_throat"], m.can_diameter * 500 + 25), P(st["mouth"], m.can_diameter * 500 + 25),
         f"mid horn {m.length*1000:.0f}")
    _dim(ax, P(st["throat"], -m.can_diameter * 500 - 30), P(st["mid_driver_back"], -m.can_diameter * 500 - 30),
         "one-piece centre body")
    _dim(ax, P(st["mouth"] + 30, -he), P(st["mouth"] + 30, he), f"{2*he:.0f}", off=(-36, 0))
    _dim(ax, P(st["back"] - 30, -hb), P(st["back"] - 30, hb), f"{2*hb:.0f}", off=(36, 0))
    fig.savefig(REN / "section-side-dims.jpg", dpi=150, pil_kwargs=dict(quality=88))
    plt.close(fig)


def stack_side():
    p = REN / "stack-side.jpg"
    if not p.exists():
        return
    st = json.loads((ROOT / "cad/stack.json").read_text())
    img = plt.imread(p)
    fig = plt.figure(figsize=(img.shape[1] / 150, img.shape[0] / 150), dpi=150)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.imshow(img)
    ax.set_axis_off()
    ax.text(img.shape[1] * 0.04, img.shape[0] * 0.06,
            f"Three boxes, {st['splay_deg']:.1f} degrees between neighbours, {2*st['splay_deg']:.0f} degrees top to bottom",
            fontsize=12, color=INK, bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#bbbbbb"))
    fig.savefig(REN / "stack-side-dims.jpg", dpi=150, pil_kwargs=dict(quality=88))
    plt.close(fig)


def main():
    front()
    section_side()
    stack_side()
    print("dimensioned renders written")


if __name__ == "__main__":
    main()
