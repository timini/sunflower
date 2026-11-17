"""Sunflower v0.2 optimiser: inside-out, with size and cost trade-offs.

Stage 1, HF and mid. For every HF unit (bullet or 1 inch compression driver)
the phase plug is its body cylinder in a 2 mm sleeve, tapering to a small
printed HF horn. Only mid drivers whose cone is at least 1.6 times the plug
diameter are allowed. Mid horns are an open-area law around that plug
(throat = Sd / compression), so the plug is always subtracted. Score:
in-band variation, coverage mismatch with the HF unit at the upper crossover,
a loading penalty when the mouth is smaller than c / (pi f1), an output
floor, and a reward for a wider mid band (Funktion-One style).

Stage 2, low. The best mids of each can size become the end of a one-piece
centre body inside a flat-panel kick horn, driven by a pro mid-bass 15 inch. The cabinet is at most 600 mm
deep and splays at least 10 degrees in a stack, the low -3 dB point must be at or below 90 Hz (penalised above 75 Hz;
the no-sub shelf takes it lower) and the mean output at least 96 dB. Score: variation from 80 Hz to 1.4 f1, coverage match with the
mid at f1, an output penalty below 100 dB, and a horn suitability term for
the 15 inch (low Qes).

Stage 3, system. The best combinations for each HF unit go through the
crossover with tuned gains, delays and polarity (simulate.evaluate). Among
designs within 0.25 of the best score, the cheapest HF unit wins, then the
shallowest cabinet.

Outputs in results/optimise/.
"""
from __future__ import annotations

import argparse
import itertools
import json
import math
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

from sunflower import tweeters as TW
from sunflower.acoustics import C, Driver, DriverSet, beamwidth, far_field, horn_transfer, solve_many, spl
from sunflower.geometry import LAWS, SIZE_ENVELOPE, SIZE_MAX_XO, LowDriverBody, LowHorn, MidHorn

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results/optimise"
F_MID = np.geomspace(150, 16000, 200)
F_LOW = np.geomspace(30, 2500, 180)
F1S = (250.0, 315.0, 400.0, 500.0)
MAX_DEPTH = 0.60
PLY, BAFFLE, BACK_BOX = 0.018, 0.025, 0.045
EXCLUDED = json.loads((ROOT / "data/drivers/excluded.json").read_text())["drivers"]
INSTRUMENT_SERIES = ("legend", "wheel_house", "wheel house", "retrobass", "big_ben", "basslite", "bn15", "cannabis",
                     "red_fang", "swamp", "texas", "governor", "private_jack", "man_o_war", "pulse", "truvox", "lab_15",
                     "kappalite_3015lf")


def hf_crossovers(tw):
    if tw.kind == "compression":
        return [x for x in (3500.0, 4500.0, 5500.0) if x >= tw.crossover_min_hz]
    return [tw.crossover_min_hz]


def lr4(f, fc, kind):
    s = 2j * np.pi * f / (2 * np.pi * fc)
    q = 1 / math.sqrt(2)
    h = (s * s if kind == "hp" else 1.0) / (s * s + s / q + 1)
    return h * h


# ------------------------------------------------------------------ catalogues
def mid_catalogue():
    recs = json.loads((ROOT / "data/drivers/mid-catalogue.json").read_text())["drivers"]
    out = []
    for r in recs:
        p = r["parameters"]
        if r["driver_id"] in EXCLUDED:
            continue
        if not all(p.get(k) for k in ("fs_hz", "re_ohm", "bl_tm", "sd_m2", "mms_kg")):
            continue
        if p.get("le_h") is None or p["le_h"] > 0.002:
            continue
        try:
            d = Driver.from_record(r)
        except (KeyError, TypeError, ZeroDivisionError):
            continue
        env = r.get("physical_envelope_mm") or {}
        frame, depth = SIZE_ENVELOPE[r["nominal_diameter"]]
        if env.get("maximum_width") and env.get("depth"):
            frame, depth = env["maximum_width"] / 1000, env["depth"] / 1000
        out.append(dict(rec=r, drv=d, nominal=r["nominal_diameter"], frame=frame, depth=depth,
                        xmax=p.get("xmax_m") or 0.003))
    return out


def low_catalogue(include_excluded=False):
    """15 inch pro mid-bass drivers for a kick horn.

    No instrument speakers, no coaxials, at least 4 mm Xmax, and a pro
    mid-bass motor: moving mass at most 130 g, inductance at most 2 mH and
    resonance at least 30 Hz (hi-fi subwoofer drivers are too heavy and too
    quiet for a kick bin).
    """
    recs = json.loads((ROOT / "data/drivers/fifteen-inch-catalogue.json").read_text())["drivers"]
    out = []
    for r in recs:
        if r["driver_id"] in EXCLUDED and not include_excluded:
            continue
        if "cx" in r["model_name"].lower():
            continue
        name = r["model_name"].lower().replace("-", "_")
        if any(name.startswith(sn.replace("-", "_")) for sn in INSTRUMENT_SERIES):
            continue
        p = r["parameters"]
        if (p.get("xmax_m") or 0) < 0.004:
            continue
        if p.get("mms_kg", 1) > 0.130 or (p.get("le_h") or 1) > 0.002 or p.get("fs_hz", 0) < 30:
            continue
        try:
            out.append(Driver.from_record(r))
        except (KeyError, TypeError, ZeroDivisionError):
            pass
    return out


def make_mid(c, tw, law, cr, mouth, length, hf_horn_length=0.0):
    return MidHorn(driver_id=c["drv"].driver_id, nominal=c["nominal"], sd=c["drv"].sd, xmax=c["xmax"],
                   frame_diameter=c["frame"], driver_depth=c["depth"], factory_sealed=c["drv"].factory_sealed,
                   hf_id=tw.id, hf_kind=tw.kind, hf_body_diameter=tw.body_diameter, hf_depth=tw.depth_with_terminals,
                   hf_exit_diameter=tw.exit_diameter, hf_horn_length=hf_horn_length,
                   law=law, compression=cr, mouth_diameter=mouth, length=length)


def hf_polar(tw, mid, f2):
    mr = mid.hf_mouth_radius if tw.kind == "compression" else None
    return lambda th: TW.response(tw.id, np.array([f2]), th, mouth_radius=mr)


# ------------------------------------------------------------------ stage 1
def mid_stage():
    cat = mid_catalogue()
    rows = []
    for tid in TW.all_ids():
        tw = TW.load(tid)
        plug_d = tw.body_diameter + 0.004
        for f2 in hf_crossovers(tw):
            pool = [c for c in cat if SIZE_MAX_XO[c["nominal"]] >= f2
                    and 2 * math.sqrt(c["drv"].sd / math.pi) >= 1.6 * plug_d]
            if not pool:
                continue
            # keep the most efficient 60 (Bl^2 Sd^2 / Re Mms^2) plus every datasheet-verified driver
            eff = lambda c: c["drv"].bl**2 * c["drv"].sd**2 / (c["drv"].re * c["drv"].mms**2)  # noqa: E731
            ranked = sorted(pool, key=eff, reverse=True)
            pool = ranked[:60] + [c for c in ranked[60:] if c["rec"].get("catalogue_status") == "manufacturer_verified"]
            bw_cache = {}
            # group by Sd so each horn (throat = Sd / compression) is solved once per group
            bins = {}
            for c in pool:
                bins.setdefault(round(c["drv"].sd / 0.002) * 0.002, []).append(c)
            for sd_bin, group in bins.items():
                ds = DriverSet.from_drivers([c["drv"] for c in group])
                sealed = all(c["drv"].factory_sealed for c in group)
                for law, cr, mouth, length in itertools.product(LAWS, (1.6, 2.2, 3.0), (0.22, 0.26, 0.30, 0.34, 0.38),
                                                                (0.12, 0.15, 0.18, 0.22, 0.26)):
                    m = replace(make_mid(group[0], tw, law, cr, mouth, length), sd=sd_bin)
                    if m.check():
                        continue
                    tr = horn_transfer(F_MID, m.area, length, 160, breaks=(m.s_exit, m.s_tip))
                    key = (law, cr, mouth, length)
                    if key not in bw_cache:
                        R = m.wavefront_radius
                        w0 = far_field(F_MID, np.ones_like(F_MID), mouth / 2, 0.0, wavefront_radius=R)
                        bw = {}
                        for fq in (*F1S, f2):
                            pf = lambda th, fq=fq: far_field(np.array([fq]), np.ones(1), mouth / 2, th,  # noqa: E731
                                                             wavefront_radius=R)
                            bw[fq] = float(beamwidth(np.array([fq]), pf)[0])
                        bw_cache[key] = (w0, bw, float(beamwidth(np.array([f2]), hf_polar(tw, m, f2))[0]))
                    w0, bw, bw_hf = bw_cache[key]
                    s = solve_many(ds, tr, front_m3=m.front_chamber_m3, rear_litres=None if sealed else m.rear_volume_litres)
                    level = spl(s["u_mouth"] * w0[None, :])
                    for f1 in F1S:
                        band = (F_MID >= f1) & (F_MID <= f2)
                        lb = level[:, band]
                        ripple, mean = np.ptp(lb, axis=1), lb.mean(axis=1)
                        hp = np.abs(lr4(F_MID, f1, "hp"))[None, :]
                        x50 = np.nanmax(s["x_peak_m"] * hp * (np.sqrt(50 * 8) / 2.83), axis=1)
                        ok = x50 < np.nan_to_num(ds.xmax[:, 0], nan=0.002)
                        load = max(0.0, C / (math.pi * f1) - mouth) / 0.1
                        mis = abs(math.log2(bw[f2] / bw_hf))
                        score = (ripple + 2 * mis + 3 * load + 0.5 * np.maximum(0, 104 - mean)
                                 - 0.75 * math.log2(f2 / f1))
                        cand = np.where(ok)[0]
                        for j in cand[np.argsort(score[cand])][:3]:
                            c = group[j]
                            mm = make_mid(c, tw, law, cr, mouth, length)
                            rows.append(dict(tweeter=tid, hf_kind=tw.kind, hf_price=tw.price_usd, f1=f1, f2=f2,
                                             driver_id=c["drv"].driver_id, label=c["drv"].label, nominal=c["nominal"],
                                             law=law, compression=cr, mouth=mouth, length=length,
                                             hf_horn_length=mm.hf_horn_length, ripple_db=ripple[j], mean_db=mean[j],
                                             x50_mm=x50[j] * 1000, bw_f1=bw[f1], bw_f2=bw[f2], bw_hf=bw_hf,
                                             loading_penalty=load, can_d=mm.can_diameter, can_l=mm.can_length,
                                             plug_d=2 * mm.plug_radius, cone_d=mm.cone_diameter, score=score[j]))
            print(f"mid stage {tid} {f2:.0f} Hz: {len(rows)} rows", flush=True)
    return pd.DataFrame(rows).sort_values("score")


def mid_from_row(r, cat_by_id):
    c = cat_by_id[r.driver_id]
    return make_mid(c, TW.load(r.tweeter), r.law, r.compression, r.mouth, r.length, r.hf_horn_length)


# ------------------------------------------------------------------ stage 2
def low_body(d: Driver):
    return LowDriverBody(cone_radius=math.sqrt(d.sd / math.pi), xmax=d.xmax or 0.005)


MIN_SPLAY_DEG = 10.0


def cabinet(h: LowHorn):
    width = h.mouth_width + 2 * PLY
    height = h.mouth_height + 2 * PLY
    return dict(width=width, height=height, depth=h.depth, volume_m3=width * height * h.depth,
                avail_litres=h.chamber_litres())


def low_stage(mids, cat_by_id):
    lows = low_catalogue()
    ds = DriverSet.from_drivers(lows)
    qes = np.array([d.re * 2 * np.pi * d.fs * d.mms / d.bl**2 for d in lows])
    generic = LowDriverBody(cone_radius=math.sqrt(0.085 / math.pi), xmax=0.005)
    rows = []
    for _, r in mids.iterrows():
        mid = mid_from_row(r, cat_by_id)
        f1 = r.f1
        band = (F_LOW >= 80) & (F_LOW <= 1.4 * f1)
        for law, wm, hm, length in itertools.product(("exponential", "hypex", "conical"), (0.56, 0.60, 0.64),
                                                     (0.50, 0.55, 0.60), (0.40, 0.45, 0.50)):
            h = LowHorn(mouth_width=wm, mouth_height=hm, length=length, law=law, body=generic, mid=mid)
            if h.check():
                continue
            cab = cabinet(h)
            if cab["depth"] > MAX_DEPTH or h.splay_deg < MIN_SPLAY_DEG:
                continue
            tr = horn_transfer(F_LOW, h.acoustic_area, h.acoustic_length, 220, h.breaks,
                               mouth_area=h.outer_area(h.mouth_z))
            a_out = math.sqrt(h.outer_area(h.mouth_z) / math.pi)
            w0 = far_field(F_LOW, np.ones_like(F_LOW), a_out, 0.0, inner_radius=h.can_radius,
                           wavefront_radius=h.wavefront_radius)
            pf = lambda th: far_field(np.array([f1]), np.ones(1), a_out, th, inner_radius=h.can_radius,  # noqa: E731
                                      wavefront_radius=h.wavefront_radius)
            bw_low = float(beamwidth(np.array([f1]), pf)[0])
            for vr in (15.0, 25.0, 35.0):
                if vr > cab["avail_litres"]:
                    continue
                s = solve_many(ds, tr, front_m3=h.front_chamber_m3, rear_litres=vr)
                level = spl(s["u_mouth"] * w0[None, :])
                lb = level[:, band]
                mean, ripple = lb.mean(axis=1), np.ptp(lb, axis=1)
                ref = level[:, (F_LOW >= 150) & (F_LOW <= f1)].mean(axis=1)
                f3 = F_LOW[np.argmax(level > (ref - 3)[:, None], axis=1)]
                x100 = np.max(s["x_peak_m"][:, band], axis=1) * np.sqrt(800) / 2.83
                ok = (mean >= 96) & (f3 <= 90) & (x100 < np.nan_to_num(ds.xmax[:, 0], nan=0))
                mis = abs(math.log2(bw_low / r.bw_f1))
                score = (ripple + 2 * mis + 5 * np.maximum(0, qes - 0.35) + 0.5 * np.maximum(0, 100 - mean)
                         + np.maximum(0, f3 - 75) / 5)
                for j in np.where(ok)[0]:
                    rows.append(dict(mid_key=r.key, low_driver=ds.ids[j], low_label=ds.labels[j], law=law,
                                     mouth_width=wm, mouth_height=hm, length=length, rear_litres=vr,
                                     compression=h.compression, ripple_db=ripple[j], mean_db=mean[j], f3=f3[j],
                                     qes=qes[j], bw_low_f1=bw_low, splay_deg=h.splay_deg, **cab, low_score=score[j]))
        print(f"low stage {r.key}: {len(rows)} rows", flush=True)
    return pd.DataFrame(rows)


# ------------------------------------------------------------------ stage 3
def config_from(m, c, cat_by_id):
    from sunflower.simulate import Config
    mid = mid_from_row(m, cat_by_id)
    lowd = {d.driver_id: d for d in low_catalogue(include_excluded=True)}[c.low_driver]
    low = LowHorn(mouth_width=c.mouth_width, mouth_height=c.mouth_height, length=c.length, law=c.law,
                  body=low_body(lowd), mid=mid)
    return Config(low, lowd, c.rear_litres, mid, cat_by_id[m.driver_id]["drv"], m.tweeter, m.f1, m.f2)


def system_stage(combos, mids, cat_by_id):
    from sunflower import simulate
    rows = []
    for _, c in combos.iterrows():
        m = mids.loc[c.mid_key]
        cfg = config_from(m, c, cat_by_id)
        if cfg.low.check():
            continue
        res = simulate.evaluate(cfg)
        rows.append(dict(mid_key=c.mid_key, tweeter=m.tweeter, hf_price=m.hf_price, low_driver=c.low_driver,
                         law=c.law, mouth_width=c.mouth_width, mouth_height=c.mouth_height, length=c.length,
                         rear_litres=c.rear_litres, width=c.width, height=c.height, depth=c.depth,
                         volume_m3=c.volume_m3, splay_deg=c.splay_deg, **res["scores"]))
    return pd.DataFrame(rows)


def select(best, sysdf):
    feas = sysdf[(sysdf.depth <= MAX_DEPTH) & ~sysdf.low_driver.isin(EXCLUDED)]
    floor = feas.score.min()
    near = feas[feas.score <= floor + 0.25].copy()
    near["price"] = near.hf_price.fillna(1e9)
    choice = near.sort_values(["price", "depth", "score"]).iloc[0]
    m = best.loc[choice.mid_key]
    sel = dict(tweeter=m.tweeter, crossover_hz=[float(m.f1), float(m.f2)],
               mid=dict(driver_id=m.driver_id, label=m.label, nominal=m.nominal, law=m.law,
                        compression=float(m.compression), mouth_diameter=float(m.mouth), length=float(m.length),
                        hf_horn_length=float(m.hf_horn_length)),
               low=dict(driver_id=choice.low_driver, law=choice.law, mouth_width=float(choice.mouth_width),
                        mouth_height=float(choice.mouth_height), length=float(choice.length),
                        rear_litres=float(choice.rear_litres)),
               scores={k: float(choice[k]) for k in ("score", "rms_on_db", "spread_30_db", "f3_hz")},
               cabinet=dict(width=float(choice.width), height=float(choice.height), depth=float(choice.depth),
                            volume_m3=float(choice.volume_m3), splay_deg=float(choice.splay_deg)),
               hf_price_usd=None if pd.isna(choice.hf_price) else float(choice.hf_price),
               rule=f"among designs within 0.25 of the best score ({floor:.2f}) and at most {MAX_DEPTH*1000:.0f} mm deep: "
                    "the cheapest HF unit, then the shallowest cabinet")
    (OUT / "selected.json").write_text(json.dumps(sel, indent=2) + "\n")
    print(json.dumps(sel, indent=2))
    return sel


def combos_from(lows, best):
    lows = lows.copy()
    lows["pre"] = lows.low_score + best.loc[lows.mid_key, "score"].values
    lows["tweeter"] = best.loc[lows.mid_key, "tweeter"].values
    lows["size_class"] = (lows.depth / 0.05).round().astype(int) * 100 + (lows.mouth_height / 0.05).round().astype(int)
    return (lows.sort_values("pre").groupby(["mid_key", "size_class"], as_index=False).head(1)
            .sort_values("pre").groupby("tweeter").head(16))


def run():
    OUT.mkdir(parents=True, exist_ok=True)
    cat_by_id = {c["drv"].driver_id: c for c in mid_catalogue()}
    mids = mid_stage()
    mids.to_csv(OUT / "mid-search.csv.gz", index=False)
    mids["can_class"] = (mids.can_d / 0.03).round().astype(int) * 100 + (mids.can_l / 0.04).round().astype(int)
    best = (mids.groupby(["tweeter", "f1", "f2", "can_class"], as_index=False).head(1)
            .sort_values("score").groupby("tweeter").head(8).reset_index(drop=True))
    best["key"] = [f"m{i:02d}" for i in range(len(best))]
    best.to_csv(OUT / "mid-shortlist.csv", index=False)
    lows = low_stage(best, cat_by_id)
    lows.to_csv(OUT / "low-search.csv.gz", index=False)
    best = best.set_index("key")
    sysdf = system_stage(combos_from(lows, best), best, cat_by_id)
    sysdf.to_csv(OUT / "system-search.csv", index=False)
    select(best, sysdf)


def from_low():
    """Re-run stages 2 and 3 from the saved mid shortlist."""
    cat_by_id = {c["drv"].driver_id: c for c in mid_catalogue()}
    best = pd.read_csv(OUT / "mid-shortlist.csv")
    lows = low_stage(best, cat_by_id)
    lows.to_csv(OUT / "low-search.csv.gz", index=False)
    best = best.set_index("key")
    sysdf = system_stage(combos_from(lows, best), best, cat_by_id)
    sysdf.to_csv(OUT / "system-search.csv", index=False)
    return select(best, sysdf)


def system_only():
    cat_by_id = {c["drv"].driver_id: c for c in mid_catalogue()}
    best = pd.read_csv(OUT / "mid-shortlist.csv").set_index("key")
    lows = pd.read_csv(OUT / "low-search.csv.gz")
    lows = lows[lows.low_driver.isin({d.driver_id for d in low_catalogue()})]
    sysdf = system_stage(combos_from(lows, best), best, cat_by_id)
    sysdf.to_csv(OUT / "system-search.csv", index=False)
    return select(best, sysdf)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--system-only", action="store_true")
    ap.add_argument("--from-low", action="store_true", help="re-run stages 2 and 3 from the saved mid shortlist")
    a = ap.parse_args()
    system_only() if a.system_only else from_low() if a.from_low else run()
