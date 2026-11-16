"""Plane-wave simulation of all three sections, the crossover and the sum.

Outputs (results/planewave/):
  low.csv, mid.csv, tweeter.csv   unfiltered section responses at 2.83 V, 1 m
  system.csv                       filtered sections and their sum, 0/15/30 degrees
  dsp.json                         crossover, gains, delays and polarity
"""
from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd

from sunflower import design as D
from sunflower.acoustics import C, far_field, group_delay, horn_transfer, solve, spl, P_REF
from sunflower.screen_15 import low_response

ROOT = D.ROOT
F = np.geomspace(20, 30000, 1200)
ANGLES = (0.0, 15.0, 30.0)


# ---------------------------------------------------------------- sections
def low_section(f=F, theta=0.0):
    p, s = low_response(D.LOW, D.low_driver(), f, D.LOW_REAR_LITRES, theta=theta)
    return p, s


def mid_transfer(f=F):
    return horn_transfer(f, D.MID.area, D.MID.length, 600, breaks=(D.TWEETER.face_from_mid_throat,))


def mid_section(f=F, theta=0.0):
    s = solve(D.mid_driver(), mid_transfer(f), front_m3=D.MID.front_chamber_cc * 1e-6, rear_litres=None)
    return far_field(f, s["u_mouth"], D.MID.mouth_diameter / 2, theta,
                     wavefront_radius=D.MID.wavefront_radius), s


def _hp2(s, w0, q):
    return s**2 / (s**2 + s * w0 / q + w0**2)


def _lp2(s, w0, q):
    return w0**2 / (s**2 + s * w0 / q + w0**2)


def tweeter_section(f=F, theta=0.0):
    """Datasheet model of the T90A: 106 dB/W/m, 2nd-order horn cut-off at
    3.6 kHz, 2nd-order roll-off at 35 kHz, piston directivity of its 60 mm
    mouth, acoustic centre recessed behind the mid horn mouth.
    """
    tw = D.TWEETER
    s = 2j * np.pi * f
    h = _hp2(s, 2 * np.pi * tw.horn_cutoff_hz, 0.6) * _lp2(s, 2 * np.pi * tw.upper_hz, 0.6)
    a = tw.housing_diameter / 2 - 0.002
    k = 2 * np.pi * f / C
    x = k * a * np.sin(np.radians(theta))
    from scipy.special import j1
    with np.errstate(invalid="ignore", divide="ignore"):
        dirc = np.where(np.abs(x) < 1e-9, 1.0, 2 * j1(x) / x)
    p0 = P_REF * 10 ** (tw.sensitivity_db / 20)
    return p0 * h * dirc, None


def source_offset():
    """Axial distance (m) of each acoustic reference behind the common mouth plane."""
    recess = D.MID.length - D.TWEETER.face_from_mid_throat
    return dict(low=0.0, mid=0.0, tweeter=recess + D.TWEETER_ACOUSTIC_CENTRE_BEHIND_FACE)


def place(f, p, offset, theta):
    k = 2 * np.pi * f / C
    return p * np.exp(-1j * k * offset * np.cos(np.radians(theta)))


# ---------------------------------------------------------------- crossover
def lr4(f, fc, kind):
    s = 2j * np.pi * f
    w = 2 * np.pi * fc
    q = 1 / np.sqrt(2)
    h = _lp2(s, w, q) if kind == "lp" else _hp2(s, w, q)
    return h * h


def filters(f):
    f1, f2 = D.CROSSOVER_LOW_MID_HZ, D.CROSSOVER_MID_HIGH_HZ
    return dict(low=lr4(f, f1, "lp"), mid=lr4(f, f1, "hp") * lr4(f, f2, "lp"), tweeter=lr4(f, f2, "hp"))


def band_level(f, p, lo, hi):
    b = (f >= lo) & (f <= hi)
    return float(np.mean(spl(p[b])))


def sections(f, theta):
    off = source_offset()
    return dict(low=place(f, low_section(f, theta)[0], off["low"], theta),
                mid=place(f, mid_section(f, theta)[0], off["mid"], theta),
                tweeter=place(f, tweeter_section(f, theta)[0], off["tweeter"], theta))


def apply(f, raw, dsp):
    filt = filters(f)
    out = {}
    for name, p in raw.items():
        g = 10 ** (dsp["gain_db"][name] / 20) * dsp["polarity"][name]
        out[name] = p * filt[name] * g * np.exp(-2j * np.pi * f * dsp["delay_ms"][name] / 1000)
    out["sum"] = sum(out.values())
    return out


def tune(f, raw):
    """Gains to the quietest section, then delays and polarity for the flattest sum."""
    levels = dict(low=band_level(f, raw["low"], 150, 400), mid=band_level(f, raw["mid"], 800, 5000),
                  tweeter=band_level(f, raw["tweeter"], 9000, 16000))
    target = min(levels.values())
    gains = {k: target - v for k, v in levels.items()}
    band = (f >= 100) & (f <= 20000)

    def cost(dsp):
        s = spl(apply(f, raw, dsp)["sum"][band]) - target
        return float(np.sqrt(np.mean(s**2)))

    best = None
    steps = np.arange(0, 3.0001, 0.02)
    # Mid/tweeter first (low is far below 7 kHz), then the low/mid pair.
    for pol_t in (1, -1):
        for dt in steps:
            for dm in (0.0,):
                dsp = dict(gain_db=gains, polarity=dict(low=1, mid=1, tweeter=pol_t),
                           delay_ms=dict(low=0.0, mid=dm, tweeter=dt))
                c = cost(dsp)
                if best is None or c < best[0]:
                    best = (c, dsp)
    mt = best[1]
    best = None
    for pol_l, rel in itertools.product((1, -1), np.arange(-3, 3.0001, 0.02)):
        dl, dm = max(rel, 0.0), max(-rel, 0.0)
        dsp = dict(gain_db=gains, polarity=dict(low=pol_l, mid=1, tweeter=mt["polarity"]["tweeter"]),
                   delay_ms=dict(low=dl, mid=dm, tweeter=mt["delay_ms"]["tweeter"] + dm))
        c = cost(dsp)
        if best is None or c < best[0]:
            best = (c, dsp)
    cost_db, dsp = best
    base = min(dsp["delay_ms"].values())
    dsp["delay_ms"] = {k: round(v - base, 3) for k, v in dsp["delay_ms"].items()}
    dsp["gain_db"] = {k: round(v, 2) for k, v in gains.items()}
    dsp.update(crossover=dict(type="Linkwitz-Riley 24 dB/oct",
                              low_mid_hz=D.CROSSOVER_LOW_MID_HZ, mid_high_hz=D.CROSSOVER_MID_HIGH_HZ),
               section_levels_db=levels, target_db=target, rms_deviation_100_20k_db=cost_db)
    return dsp


def write_section(path, f, p, s):
    df = pd.DataFrame(dict(frequency_hz=f, spl_db=spl(p), phase_deg=np.degrees(np.angle(p)),
                           group_delay_ms=group_delay(f, p) * 1000))
    if s is not None:
        df["impedance_ohm"] = np.abs(s["z_e"])
        df["excursion_peak_mm_2p83v"] = s["x_peak_m"] * 1000
    df.to_csv(path, index=False, float_format="%.6g")
    return df


def run(out: Path):
    out.mkdir(parents=True, exist_ok=True)
    f = F
    off = source_offset()
    for name, fn in (("low", low_section), ("mid", mid_section), ("tweeter", tweeter_section)):
        p, s = fn(f)
        write_section(out / f"{name}.csv", f, place(f, p, off[name], 0.0), s)
    raw0 = sections(f, 0.0)
    dsp = tune(f, raw0)
    cols = dict(frequency_hz=f)
    for th in ANGLES:
        res = apply(f, sections(f, th), dsp)
        for k, v in res.items():
            cols[f"{k}_{int(th)}deg_spl_db"] = spl(v)
            cols[f"{k}_{int(th)}deg_phase_deg"] = np.degrees(np.angle(v))
    pd.DataFrame(cols).to_csv(out / "system.csv", index=False, float_format="%.6g")
    (out / "dsp.json").write_text(json.dumps(dsp, indent=2) + "\n")
    (out / "design.json").write_text(json.dumps(D.summary(), indent=2) + "\n")
    print(json.dumps(dsp, indent=2))
    return dsp


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--output", type=Path, default=ROOT / "results/planewave")
    run(ap.parse_args().output)
