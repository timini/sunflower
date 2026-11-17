"""Section responses, crossover tuning and the summed system.

``Config`` holds one complete design. ``evaluate`` returns its section
responses, DSP settings and scores; the optimiser and the published design
both go through it.

Outputs of ``run`` (results/planewave/):
  low.csv, mid.csv, tweeter.csv   unfiltered section responses at 2.83 V, 1 m
  system.csv                       filtered sections and their sum, 0/15/30 degrees
  dsp.json                         crossover, gains, delays and polarity
"""
from __future__ import annotations

import argparse
import itertools
import json
import math
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
import pandas as pd

from sunflower import tweeters as TW
from sunflower.acoustics import C, Driver, far_field, group_delay, horn_transfer, solve, spl
from sunflower.geometry import LowDriverBody, LowHorn, MidHorn

ROOT = Path(__file__).resolve().parents[2]
F = np.geomspace(20, 30000, 1200)
ANGLES = (0.0, 15.0, 30.0)


@dataclass
class Config:
    low: LowHorn
    low_driver: Driver
    rear_litres: float
    mid: MidHorn
    mid_driver: Driver
    tweeter: str
    f1: float
    f2: float
    use_hf_fem: bool = False

    @property
    def tw(self):
        return TW.load(self.tweeter)


# ---------------------------------------------------------------- sections
def low_section(cfg: Config, f=F, theta=0.0):
    h = cfg.low
    tr = horn_transfer(f, h.acoustic_area, h.acoustic_length, 300, h.breaks, mouth_area=h.outer_area(h.mouth_z))
    s = solve(cfg.low_driver, tr, front_m3=h.front_chamber_m3, rear_litres=cfg.rear_litres)
    a_out = math.sqrt(h.outer_area(h.mouth_z) / math.pi)
    p = far_field(f, s["u_mouth"], a_out, theta, inner_radius=h.mid.can_diameter / 2,
                  wavefront_radius=h.wavefront_radius)
    return p, s


def mid_transfer(cfg: Config, f=F):
    m = cfg.mid
    return horn_transfer(f, m.area, m.length, 400, breaks=(m.s_exit, m.s_tip))


def mid_section(cfg: Config, f=F, theta=0.0):
    m = cfg.mid
    rear = None if cfg.mid_driver.factory_sealed else m.rear_volume_litres
    s = solve(cfg.mid_driver, mid_transfer(cfg, f), front_m3=m.front_chamber_m3, rear_litres=rear)
    return far_field(f, s["u_mouth"], m.mouth_diameter / 2, theta, wavefront_radius=m.wavefront_radius), s


HF_FEM = ROOT / "results/fem/hf-in-mid.csv"


def _smooth_log(f, y, octave=1 / 6):
    lf = np.log2(f)
    out = np.empty_like(y)
    for i, x in enumerate(lf):
        m = np.abs(lf - x) <= octave / 2
        out[i] = y[m].mean()
    return out


def tweeter_section(cfg: Config, f=F, theta=0.0, use_fem=True):
    """HF unit: datasheet on-axis response, corrected by the FEM of its printed horn mouth inside the mid
    horn (relative to a free piston of the same mouth, smoothed to 1/6 octave) when that FEM exists for this
    design. Without it: a piston of the HF horn mouth, or the datasheet's own angles for bullets."""
    mr = cfg.mid.hf_mouth_radius if cfg.tw.kind == "compression" else None
    if not (use_fem and HF_FEM.exists() and getattr(cfg, "use_hf_fem", False)):
        return TW.response(cfg.tweeter, f, theta, mouth_radius=mr), None
    d = pd.read_csv(HF_FEM)
    col = f"fem_rel_{int(theta)}deg_db"
    if col not in d:
        return TW.response(cfg.tweeter, f, theta, mouth_radius=mr), None
    fd = d.frequency_hz.values
    corr = _smooth_log(fd, d[col].values + d.onaxis_vs_piston_db.values)
    p0 = TW.response(cfg.tweeter, f, 0.0, mouth_radius=mr)
    c = np.interp(np.log10(f), np.log10(fd), corr)
    below = f < fd.min()
    piston = 20 * np.log10(np.abs(TW.response(cfg.tweeter, f, theta, mouth_radius=mr)) / np.abs(p0))
    c0 = np.interp(np.log10(fd.min()), np.log10(fd), corr) - np.interp(np.log10(fd.min()), np.log10(f), piston)
    c = np.where(below, piston + c0 * (f / fd.min()) ** 2, c)
    return p0 * 10 ** (c / 20), None


def source_offset(cfg: Config):
    """Axial distance (m) of each acoustic reference behind the common mouth plane."""
    m, tw = cfg.mid, cfg.tw
    behind = 0.020 if tw.kind == "compression" else tw.horn_depth   # diaphragm behind the exit
    return dict(low=0.0, mid=0.0, tweeter=m.length - m.s_exit + behind)


def place(f, p, offset, theta):
    k = 2 * np.pi * f / C
    return p * np.exp(-1j * k * offset * np.cos(np.radians(theta)))


def sections(cfg, f, theta):
    off = source_offset(cfg)
    return dict(low=place(f, low_section(cfg, f, theta)[0], off["low"], theta),
                mid=place(f, mid_section(cfg, f, theta)[0], off["mid"], theta),
                tweeter=place(f, tweeter_section(cfg, f, theta)[0], off["tweeter"], theta))


# ---------------------------------------------------------------- crossover
def lr4(f, fc, kind):
    s = 2j * np.pi * f / (2 * np.pi * fc)
    q = 1 / math.sqrt(2)
    h = (s * s if kind == "hp" else 1.0) / (s * s + s / q + 1)
    return h * h


def filters(cfg, f):
    return dict(low=lr4(f, cfg.f1, "lp"), mid=lr4(f, cfg.f1, "hp") * lr4(f, cfg.f2, "lp"),
                tweeter=lr4(f, cfg.f2, "hp"))


def band_level(f, p, lo, hi):
    b = (f >= lo) & (f <= hi)
    return float(np.mean(spl(p[b])))


def apply(cfg, f, raw, dsp):
    filt = filters(cfg, f)
    out = {}
    for name, p in raw.items():
        g = 10 ** (dsp["gain_db"][name] / 20) * dsp["polarity"][name]
        out[name] = p * filt[name] * g * np.exp(-2j * np.pi * f * dsp["delay_ms"][name] / 1000)
    out["sum"] = sum(out.values())
    return out


def tune(cfg, f, raw):
    """Gains to the quietest section, then delays and polarity for the flattest sum."""
    levels = dict(low=band_level(f, raw["low"], 150, 0.8 * cfg.f1),
                  mid=band_level(f, raw["mid"], 1.6 * cfg.f1, 0.7 * cfg.f2),
                  tweeter=band_level(f, raw["tweeter"], 1.5 * cfg.f2, 16000))
    target = min(levels.values())
    gains = {k: target - v for k, v in levels.items()}
    band = (f >= 100) & (f <= 16000)

    def cost(dsp):
        s = spl(apply(cfg, f, raw, dsp)["sum"][band]) - target
        return float(np.sqrt(np.mean(s**2)))

    best = None
    for pol_t, dt in itertools.product((1, -1), np.arange(-1.0, 3.0001, 0.02)):
        dsp = dict(gain_db=gains, polarity=dict(low=1, mid=1, tweeter=pol_t),
                   delay_ms=dict(low=0.0, mid=max(-dt, 0.0), tweeter=max(dt, 0.0)))
        c = cost(dsp)
        if best is None or c < best[0]:
            best = (c, dsp)
    mt = best[1]
    best = None
    for pol_l, rel in itertools.product((1, -1), np.arange(-3, 3.0001, 0.02)):
        dl, dm = max(rel, 0.0), max(-rel, 0.0)
        dsp = dict(gain_db=gains, polarity=dict(low=pol_l, mid=1, tweeter=mt["polarity"]["tweeter"]),
                   delay_ms=dict(low=dl, mid=mt["delay_ms"]["mid"] + dm, tweeter=mt["delay_ms"]["tweeter"] + dm))
        c = cost(dsp)
        if best is None or c < best[0]:
            best = (c, dsp)
    cost_db, dsp = best
    base = min(dsp["delay_ms"].values())
    dsp["delay_ms"] = {k: round(v - base, 3) for k, v in dsp["delay_ms"].items()}
    dsp["gain_db"] = {k: round(v, 2) for k, v in gains.items()}
    dsp.update(crossover=dict(type="Linkwitz-Riley 24 dB/oct", low_mid_hz=cfg.f1, mid_high_hz=cfg.f2),
               section_levels_db=levels, target_db=target, rms_deviation_100_16k_db=cost_db)
    return dsp


def evaluate(cfg: Config, f=None):
    f = np.geomspace(20, 30000, 500) if f is None else f
    raw0 = sections(cfg, f, 0.0)
    dsp = tune(cfg, f, raw0)
    on = apply(cfg, f, raw0, dsp)["sum"]
    off = apply(cfg, f, sections(cfg, f, 30.0), dsp)["sum"]
    s0, s30 = spl(on), spl(off)
    b = (f >= 100) & (f <= 16000)
    rms = float(np.sqrt(np.mean((s0[b] - dsp["target_db"]) ** 2)))
    d = (f >= 300) & (f <= 12000)
    rel = s30[d] - s0[d]
    lf = np.log10(f[d])
    spread = float(np.std(rel - np.polyval(np.polyfit(lf, rel, 1), lf)))
    ref = s0[(f >= 200) & (f <= 2000)].mean()
    f3 = float(f[np.argmax(s0 > ref - 3)])
    score = rms + 0.5 * spread + max(0.0, f3 - 70) / 20
    return dict(dsp=dsp, scores=dict(score=score, rms_on_db=rms, spread_30_db=spread, f3_hz=f3,
                                     sensitivity_db=float(dsp["target_db"])))


def write_section(path, f, p, s):
    df = pd.DataFrame(dict(frequency_hz=f, spl_db=spl(p), phase_deg=np.degrees(np.angle(p)),
                           group_delay_ms=group_delay(f, p) * 1000))
    if s is not None:
        df["impedance_ohm"] = np.abs(s["z_e"])
        df["excursion_peak_mm_2p83v"] = s["x_peak_m"] * 1000
    df.to_csv(path, index=False, float_format="%.6g")
    return df


def run(out: Path):
    from sunflower import design as D
    cfg = D.config()
    out.mkdir(parents=True, exist_ok=True)
    f = F
    off = source_offset(cfg)
    for name, fn in (("low", low_section), ("mid", mid_section), ("tweeter", tweeter_section)):
        p, s = fn(cfg, f)
        write_section(out / f"{name}.csv", f, place(f, p, off[name], 0.0), s)
        cols = dict(frequency_hz=f)
        for th in (0, 15, 30, 45, 60):
            cols[f"spl_{th}deg_db"] = spl(fn(cfg, f, th)[0])
        pd.DataFrame(cols).to_csv(out / f"{name}-polar.csv", index=False, float_format="%.6g")
    raw0 = sections(cfg, f, 0.0)
    dsp = tune(cfg, f, raw0)
    cols = dict(frequency_hz=f)
    for th in ANGLES:
        res = apply(cfg, f, sections(cfg, f, th), dsp)
        for k, v in res.items():
            cols[f"{k}_{int(th)}deg_spl_db"] = spl(v)
            cols[f"{k}_{int(th)}deg_phase_deg"] = np.degrees(np.angle(v))
    # No-sub option: +6 dB second-order low shelf on the low channel at its -3 dB point, 30 Hz protection high-pass
    lowp, lows = low_section(cfg, f)
    ref = spl(lowp)[(f >= 150) & (f <= cfg.f1)].mean()
    f3 = float(f[np.argmax(spl(lowp) > ref - 3)])
    s = 2j * np.pi * f
    w = 2 * np.pi * f3
    shelf = (s * s + s * w * math.sqrt(2) * 2 ** 0.25 + 2 * w * w) / (s * s + s * w * math.sqrt(2) + w * w)
    hp = lr4(f, 30.0, "hp")
    res = apply(cfg, f, raw0, dsp)
    shelved = res["sum"] + res["low"] * (shelf * hp - 1)
    cols["sum_shelf_0deg_spl_db"] = spl(shelved)
    pd.DataFrame(cols).to_csv(out / "system.csv", index=False, float_format="%.6g")
    z_nom = 8.0
    x_shelf = np.abs(lows["x_peak_m"] * shelf * hp * lr4(f, cfg.f1, "lp")) * np.sqrt(cfg.low_driver.power_w * z_nom
                                                                                   if cfg.low_driver.power_w else 800) / 2.83
    sref = spl(shelved)[(f >= 200) & (f <= 2000)].mean()
    dsp["no_sub_shelf"] = dict(type="2nd-order low shelf +6 dB at the low section's -3 dB point, LR4 high-pass 30 Hz",
                               shelf_hz=f3, f_minus6_hz=float(f[np.argmax(spl(shelved) > sref - 6)]),
                               max_excursion_mm=float(x_shelf[(f >= 30) & (f <= 200)].max() * 1000),
                               xmax_mm=(cfg.low_driver.xmax or float("nan")) * 1000,
                               rated_power_w=cfg.low_driver.power_w)
    xs = dsp["no_sub_shelf"]
    if cfg.low_driver.power_w and cfg.low_driver.xmax:
        xs["safe_power_w"] = float(cfg.low_driver.power_w * min(1.0, (xs["xmax_mm"] / xs["max_excursion_mm"]) ** 2))
    (out / "dsp.json").write_text(json.dumps(dsp, indent=2) + "\n")
    (out / "design.json").write_text(json.dumps(D.summary(), indent=2) + "\n")
    print(json.dumps(dsp, indent=2))
    return dsp


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--output", type=Path, default=ROOT / "results/planewave")
    run(ap.parse_args().output)
