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
    we = cut["external_mm"]["width"] / 2
    wm = lo.mouth_width * 500
    wt = lo.throat_width * 500
    _dim(ax, px(-we, we + 40), px(we, we + 40), f"{2*we:.0f} outside")
    _dim(ax, px(-wm, -we - 40), px(wm, -we - 40), f"mouth {2*wm:.0f}")
    _dim(ax, px(-wt, -wt - 14), px(wt, -wt - 14), f"throat {2*wt:.0f} across flats")
    rm = m.mouth_diameter * 500
    _dim(ax, px(-rm, rm + 30), px(rm, rm + 30), f"mid mouth {2*rm:.0f}")
    rc = m.can_diameter * 500
    _dim(ax, px(rc + 25, -rc), px(rc + 25, rc), f"can\n{2*rc:.0f}", off=(42, 0))
    rp = m.plug_radius * 1000
    _dim(ax, px(-rp, -rp - 18), px(rp, -rp - 18), f"{tw.model} {tw.body_diameter*1000:.0f}", off=(0, 26))
    ax.set_xlim(0, fig.get_figwidth() * 150)
    fig.savefig(REN / "front-dims.jpg", dpi=150, pil_kwargs=dict(quality=88))
    plt.close(fig)


def section_side():
    cut = json.loads((ROOT / "cad/cutlist.json").read_text())
    st = cut["stations_mm"]
    lo, m = D.LOW, D.MID
    fig, ax, px = _setup("section-side")
    # horizontal coordinate on this view is the CAD z axis (mouth on the left)
    P = lambda z, y: px(-z, y)  # noqa: E731
    we = cut["external_mm"]["width"] / 2
    zback = st["mouth"] - cut["external_mm"]["depth"]
    _dim(ax, P(zback, -we - 45), P(st["mouth"], -we - 45), f"cabinet depth {cut['external_mm']['depth']:.0f}")
    _dim(ax, P(st["baffle_front"], we + 45), P(st["mouth"], we + 45), f"low horn {lo.length*1000:.0f}")
    _dim(ax, P(st["mid_tail_tip"], -m.can_diameter * 500 - 40), P(st["mouth"], -m.can_diameter * 500 - 40),
         f"mid can {st['mouth'] - st['mid_tail_tip']:.0f}")
    _dim(ax, P(st["mid_throat"], m.can_diameter * 500 + 30), P(st["mouth"], m.can_diameter * 500 + 30),
         f"mid horn {m.length*1000:.0f}")
    _dim(ax, P(st["mouth"] + 30, -we), P(st["mouth"] + 30, we), f"{2*we:.0f}", off=(-36, 0))
    _dim(ax, P(0, lo.body.cone_radius * 1000 + 30), P(lo.nose_end * 1000, lo.body.cone_radius * 1000 + 30),
         f"15 in with nose {lo.nose_end*1000:.0f}")
    fig.savefig(REN / "section-side-dims.jpg", dpi=150, pil_kwargs=dict(quality=88))
    plt.close(fig)


def main():
    front()
    section_side()
    print("dimensioned renders written")


if __name__ == "__main__":
    main()
