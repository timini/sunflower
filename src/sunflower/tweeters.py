"""Bullet tweeters from their datasheets.

The tweeters are not simulated. Each one is the manufacturer's published
on-axis and off-axis magnitude, read by eye from the datasheet graph (about
plus or minus 2 dB) and joined with a monotone spline in log frequency, with minimum phase reconstructed from the magnitude.
Physical sizes come from the datasheet drawings; the ones marked
``estimated`` in the JSON were scaled from photographs.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np

from sunflower.acoustics import C, P_REF

ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class Tweeter:
    id: str
    model: str
    body_diameter: float
    body_depth: float
    depth_with_terminals: float
    mouth_diameter: float
    bullet_base: float
    horn_depth: float
    sensitivity_db: float
    crossover_min_hz: float
    cutoff_hz: float
    source: str
    off_axis: str = "table"

    @property
    def body_radius(self):
        return self.body_diameter / 2


def load(tid: str) -> Tweeter:
    r = json.loads((ROOT / f"data/tweeters/{tid}.json").read_text())
    return Tweeter(id=r["id"], model=r["model"], body_diameter=r["body_diameter_mm"] / 1000,
                   body_depth=r["body_depth_mm"] / 1000, depth_with_terminals=r["depth_with_terminals_mm"] / 1000,
                   mouth_diameter=r["mouth_diameter_mm"] / 1000, bullet_base=r["bullet_base_mm"] / 1000,
                   horn_depth=r["horn_depth_mm"] / 1000, sensitivity_db=r["sensitivity_db_1w"],
                   crossover_min_hz=r["crossover_min_hz"], cutoff_hz=r["cutoff_hz"], source=r["source"],
                   off_axis=r.get("off_axis", "table"))


def all_ids():
    return sorted(p.stem for p in (ROOT / "data/tweeters").glob("*.json"))


def _table(tid):
    r = json.loads((ROOT / f"data/tweeters/{tid}.json").read_text())
    return r["response_db"]


def _extend(f_pts, db_pts):
    """Log-frequency magnitude with gentle extrapolation outside the table."""
    lf = np.log10(f_pts)
    db = np.asarray(db_pts, float)
    lo_slope = min((db[1] - db[0]) / (lf[1] - lf[0]), 24 / np.log10(2))   # at most 24 dB/oct
    lo_slope = max(lo_slope, 12 / np.log10(2))                             # at least 12 dB/oct
    hi_slope = -12 / np.log10(2)

    from scipy.interpolate import PchipInterpolator
    smooth = PchipInterpolator(lf, db)

    def mag(f):
        x = np.log10(np.maximum(f, 1.0))
        y = smooth(np.clip(x, lf[0], lf[-1]))
        y = np.where(x < lf[0], db[0] + lo_slope * (x - lf[0]), y)
        y = np.where(x > lf[-1], db[-1] + hi_slope * (x - lf[-1]), y)
        return y
    return mag


@lru_cache(maxsize=32)
def _minphase(tid, column):
    t = _table(tid)
    mag = _extend(t["f"], t[column])
    fs, n = 400_000.0, 2**19
    f = np.fft.rfftfreq(n, 1 / fs)
    m = 10 ** ((mag(f) - mag(np.array([5000.0]))[0]) / 20)
    logm = np.log(np.maximum(m, 1e-9))
    full = np.concatenate([logm, logm[-2:0:-1]])
    cep = np.fft.ifft(full).real
    w = np.zeros(n)
    w[0] = 1
    w[1:n // 2] = 2
    w[n // 2] = 1
    h = np.exp(np.fft.fft(cep * w))[: n // 2 + 1]
    return f, np.unwrap(np.angle(h))


def response(tid: str, f, theta=0.0):
    """Complex pressure at 1 m for 2.83 V (1 W into 8 ohm), phase referenced to the diaphragm."""
    f = np.asarray(f, float)
    t = _table(tid)
    tw = load(tid)
    th = abs(theta)
    if tw.off_axis == "piston" or "30" not in t:
        db = _extend(t["f"], t["0"])(f)
        if th > 0:
            from scipy.special import j1
            k = 2 * np.pi * f / C
            x = k * tw.mouth_diameter / 2 * np.sin(np.radians(th))
            with np.errstate(invalid="ignore", divide="ignore"):
                d = np.where(np.abs(x) < 1e-9, 1.0, np.abs(2 * j1(x) / x))
            db = db + 20 * np.log10(np.maximum(d, 1e-4))
    else:
        cols = [(0, "0"), (30, "30"), (60, "60")]
        mags = [_extend(t["f"], t[c])(f) for _, c in cols]
        if th <= 30:
            w = th / 30
            db = (1 - w) * mags[0] + w * mags[1]
        else:
            w = min((th - 30) / 30, 1.5)
            db = (1 - w) * mags[1] + w * mags[2]
    ff, ph = _minphase(tid, "0")
    phase = np.interp(f, ff, ph)
    return P_REF * 10 ** (db / 20) * np.exp(1j * phase)
