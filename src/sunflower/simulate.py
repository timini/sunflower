"""Section responses, crossover tuning and the summed system.

``Config`` holds one complete design. ``evaluate`` returns its section
responses, DSP settings and scores; the optimiser and the published design
both go through it. The kick and mid come from the nodal networks in
``sunflower.models``, built from the real geometry.

Single-box results are in free space: a baffle-step shelf for the box
front takes the half-space horn models down to 4 pi below about 115/width
Hz. ``cluster=9`` gives one box of a 3 x 3 array: its mouth load is
raised by the other eight mouths, and the baffle is the whole array.

Outputs of ``run`` (results/planewave/):
  low.csv, mid.csv, tweeter.csv   unfiltered section responses at 2.83 V, 1 m
  system.csv                       filtered sections and their sum, 0/15/30 degrees
  array.csv                        kick of one box and of a 3 x 3 cluster, low frequencies
  dsp.json                         crossover, gains, EQ, delays and polarity
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
from sunflower.acoustics import C, Driver, baffle_step, couple, far_field, group_delay, spl
from sunflower.geometry import Kick, MidHorn
from sunflower.models import kick_network, mid_network

ROOT = Path(__file__).resolve().parents[2]
F = np.geomspace(20, 30000, 1200)
ANGLES = (0.0, 15.0, 30.0)


@dataclass
class Config:
    kick: Kick
    kick_driver: Driver
    mid: MidHorn
    mid_driver: Driver
    tweeter: str
    f1: float
    f2: float
    use_hf_fem: bool = False
    use_mid_fem: bool = False
    cluster: int = 1
    free_space: bool = True

    @property
    def tw(self):
        return TW.load(self.tweeter)

    @property
    def baffle_width(self):
        return self.kick.external_mouth * math.sqrt(self.cluster)


_CACHE = {}


def _solved(kind, cfg, f):
    key = (kind, cfg.kick if kind == "kick" else cfg.mid, cfg.cluster, len(f), float(f[0]), float(f[-1]))
    if key not in _CACHE:
        if len(_CACHE) > 400:
            _CACHE.clear()
        net = kick_network(cfg.kick, cluster=cfg.cluster) if kind == "kick" else mid_network(cfg.mid)
        _CACHE[key] = (net, net.solve(f))
    return _CACHE[key]


def _bs(cfg, f):
    return baffle_step(f, cfg.baffle_width) if cfg.free_space else 1.0


# ---------------------------------------------------------------- sections
def low_section(cfg: Config, f=F, theta=0.0):
    k = cfg.kick
    fk = np.minimum(f, 4000.0)
    net, r = _solved("kick", cfg, fk)
    s = couple(cfg.kick_driver, r, rear_litres=k.chamber_litres)
    u = s["u_mouth"] * cfg.cluster
    p = far_field(f, u, net.mouth_radius * math.sqrt(cfg.cluster), theta, inner_radius=net.mouth_inner)
    p = np.where(f > 4000, p * (4000 / f) ** 2, p)
    return p * _bs(cfg, f), s


MID_FEM = ROOT / "results/fem/mid-vs-network.csv"


def mid_fem_correction(f, theta):
    """FEM minus network for the mid (dB), smoothed to 1/6 octave, faded in from 0.8 to 1.2 kHz.

    The network lumps the air over the cone at the throat annulus, so it misses the radial cavity
    mode that the FEM shows from about 2.7 kHz. Below 1 kHz the two agree, and the FEM's rigid baffle
    at the kick mouth is a worst case at the low crossover, so it is not applied there."""
    d = pd.read_csv(MID_FEM)
    angles = [0, 15, 30, 45, 60]
    th = min(angles, key=lambda a: abs(a - theta))
    fd = d.frequency_hz.values
    corr = _smooth_log(fd, d[f"diff_{th}deg_db"].values)
    c = np.interp(np.log10(f), np.log10(fd), corr, left=0.0, right=corr[-1])
    w = np.clip((np.log10(f) - np.log10(800)) / (np.log10(1200) - np.log10(800)), 0, 1)
    return c * w


def mid_section(cfg: Config, f=F, theta=0.0):
    m = cfg.mid
    net, r = _solved("mid", cfg, f)
    rear = m.rear_volume_litres
    s = couple(cfg.mid_driver, r, rear_litres=rear)
    p = far_field(f, s["u_mouth"], m.mouth_diameter / 2, theta, wavefront_radius=m.wavefront_radius) * _bs(cfg, f)
    if cfg.use_mid_fem and MID_FEM.exists():
        p = p * 10 ** (mid_fem_correction(f, theta) / 20)
    return p, s


HF_FEM = ROOT / "results/fem/hf-in-mid.csv"


def _smooth_log(f, y, octave=1 / 6):  # noqa: D103
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
        return TW.response(cfg.tweeter, f, theta, mouth_radius=mr) * _bs(cfg, f), None
    d = pd.read_csv(HF_FEM)
    col = f"fem_rel_{int(theta)}deg_db"
    if col not in d:
        return TW.response(cfg.tweeter, f, theta, mouth_radius=mr), None
    fd = d.frequency_hz.values
    corr = _smooth_log(fd, d[col].values)   # off-axis shape only: the datasheet on-axis curve already has horn gain
    p0 = TW.response(cfg.tweeter, f, 0.0, mouth_radius=mr)
    c = np.interp(np.log10(f), np.log10(fd), corr)
    below = f < fd.min()
    piston = 20 * np.log10(np.abs(TW.response(cfg.tweeter, f, theta, mouth_radius=mr)) / np.abs(p0))
    c0 = np.interp(np.log10(fd.min()), np.log10(fd), corr) - np.interp(np.log10(fd.min()), np.log10(f), piston)
    c = np.where(below, piston + c0 * (f / fd.min()) ** 2, c)
    return p0 * 10 ** (c / 20) * _bs(cfg, f), None


def source_offset(cfg: Config):
    """Axial distance (m) of each acoustic reference behind the common mouth plane."""
    m, tw = cfg.mid, cfg.tw
    behind = 0.020 if tw.kind == "compression" else tw.horn_depth   # diaphragm behind the exit
    return dict(low=cfg.kick.protrusion, mid=0.0, tweeter=m.length - m.s_exit + behind)


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


def high_shelf(f, fc, gain_db):
    """Second-order high shelf, Q 0.707 (RBJ analogue prototype)."""
    A = 10 ** (gain_db / 40)
    s = 1j * np.asarray(f) / fc
    q = 1 / math.sqrt(2)
    return A * (A * s * s + math.sqrt(A) / q * s + 1) / (s * s + math.sqrt(A) / q * s + A)


def filters(cfg, f, eq=None):
    mid = lr4(f, cfg.f1, "hp") * lr4(f, cfg.f2, "lp")
    if eq and eq.get("mid_high_shelf_db"):
        mid = mid * high_shelf(f, eq["mid_high_shelf_hz"], eq["mid_high_shelf_db"])
    return dict(low=lr4(f, cfg.f1, "lp"), mid=mid, tweeter=lr4(f, cfg.f2, "hp"))


def band_level(f, p, lo, hi):
    b = (f >= lo) & (f <= hi)
    return float(np.mean(spl(p[b])))


def apply(cfg, f, raw, dsp):
    filt = filters(cfg, f, dsp.get("eq"))
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
    # mid high shelf: the gain (0 to 12 dB, corner 0.55 f2) that flattens the mid from f2/3 to f2
    mb = (f >= cfg.f2 / 3) & (f <= cfg.f2)
    fc = 0.55 * cfg.f2
    best_eq = min(np.arange(0, 12.01, 0.5),
                  key=lambda g: np.std(spl(raw["mid"][mb] * high_shelf(f[mb], fc, g))))
    eq = dict(mid_high_shelf_hz=round(fc, 1), mid_high_shelf_db=float(best_eq))
    levels["mid"] = band_level(f, raw["mid"] * high_shelf(f, fc, best_eq), 1.6 * cfg.f1, 0.7 * cfg.f2)
    target = min(levels.values())
    gains = {k: target - v for k, v in levels.items()}
    band = (f >= 100) & (f <= 16000)

    def cost(dsp):
        s = spl(apply(cfg, f, raw, dsp)["sum"][band]) - target
        return float(np.sqrt(np.mean(s**2)))

    best = None
    for pol_t, dt in itertools.product((1, -1), np.arange(-1.0, 3.0001, 0.02)):
        dsp = dict(gain_db=gains, eq=eq, polarity=dict(low=1, mid=1, tweeter=pol_t),
                   delay_ms=dict(low=0.0, mid=max(-dt, 0.0), tweeter=max(dt, 0.0)))
        c = cost(dsp)
        if best is None or c < best[0]:
            best = (c, dsp)
    mt = best[1]
    best = None
    for pol_l, rel in itertools.product((1, -1), np.arange(-3, 3.0001, 0.02)):
        dl, dm = max(rel, 0.0), max(-rel, 0.0)
        dsp = dict(gain_db=gains, eq=eq, polarity=dict(low=pol_l, mid=1, tweeter=mt["polarity"]["tweeter"]),
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
    score = rms + 0.5 * spread + max(0.0, f3 - 90) / 20
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
    x_shelf = np.abs(lows["x_peak_m"] * shelf * hp * lr4(f, cfg.f1, "lp")) * np.sqrt(cfg.kick_driver.power_w * z_nom
                                                                                   if cfg.kick_driver.power_w else 800) / 2.83
    sref = spl(shelved)[(f >= 200) & (f <= 2000)].mean()
    dsp["no_sub_shelf"] = dict(type="2nd-order low shelf +6 dB at the low section's -3 dB point, LR4 high-pass 30 Hz",
                               shelf_hz=f3, f_minus6_hz=float(f[np.argmax(spl(shelved) > sref - 6)]),
                               max_excursion_mm=float(x_shelf[(f >= 30) & (f <= 200)].max() * 1000),
                               xmax_mm=(cfg.kick_driver.xmax or float("nan")) * 1000,
                               rated_power_w=cfg.kick_driver.power_w)
    xs = dsp["no_sub_shelf"]
    if cfg.kick_driver.power_w and cfg.kick_driver.xmax:
        xs["safe_power_w"] = float(cfg.kick_driver.power_w * min(1.0, (xs["xmax_mm"] / xs["max_excursion_mm"]) ** 2))
    # Kick alone at low frequency: one box in free space, one on the ground (half space), and a 3 x 3 cluster
    fl = np.geomspace(30, 1000, 300)
    arr = dict(frequency_hz=fl,
               single_free_db=spl(low_section(cfg, fl)[0]),
               single_half_space_db=spl(low_section(replace(cfg, free_space=False), fl)[0]),
               cluster9_free_db=spl(low_section(replace(cfg, cluster=9), fl)[0]))
    pd.DataFrame(arr).to_csv(out / "array.csv", index=False, float_format="%.6g")
    def fm(y, drop):
        ref = y[(fl >= 150) & (fl <= 250)].mean()
        return float(fl[np.argmax(y > ref - drop)])
    dsp["kick_low_end"] = {k: dict(f_minus3_hz=fm(arr[k], 3), f_minus6_hz=fm(arr[k], 6),
                                   level_150_250_db=float(arr[k][(fl >= 150) & (fl <= 250)].mean()))
                           for k in ("single_free_db", "single_half_space_db", "cluster9_free_db")}
    (out / "dsp.json").write_text(json.dumps(dsp, indent=2) + "\n")
    (out / "design.json").write_text(json.dumps(D.summary(), indent=2) + "\n")
    print(json.dumps(dsp, indent=2))
    return dsp


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--output", type=Path, default=ROOT / "results/planewave")
    run(ap.parse_args().output)
