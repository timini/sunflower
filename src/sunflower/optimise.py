"""Joint optimisation of the three nested sections, with size trade-offs.

Stage 1, mid: for each bullet tweeter and crossover pair, every mid horn on
the grid (profile, throat, mouth, length, tweeter depth) is solved once and
coupled to every cone midrange in the catalogue that can reach the upper
crossover. The mid mouth must be at least c / (pi f1) across, so the horn
still loads its driver at the lower crossover. Score: in-band variation,
coverage mismatch with the tweeter at the upper crossover, and an output
penalty below 104 dB.

Stage 2, low: the best mid of each can size is placed in every low horn on
the grid, coupled to every 15 inch driver at four sealed-chamber volumes.
Score: variation from 100 Hz to 1.4 times the lower crossover, coverage
mismatch with the mid at the lower crossover, and bass extension.

Stage 3, system: the best combinations are put through the crossover with
tuned gains, delays and polarity. System score:
    on-axis RMS deviation, 100 Hz to 16 kHz
  + 0.5 x roughness of the 30 degree response relative to on axis, 300 Hz to
    12 kHz (RMS residual after a straight-line fit against log frequency, so
    steady narrowing is allowed and steps at the crossovers are not)
  + bass penalty, (f-3 - 70 Hz) / 20 when above 70 Hz.
Each design also has a cabinet volume, and the selection is the smallest
cabinet within 0.25 of the best score that fits 850 mm wide and 1200 mm deep.

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
from sunflower.acoustics import (C, Driver, DriverSet, beamwidth, far_field, horn_transfer, solve_many, spl)
from sunflower.geometry import (PROFILES, SIZE_ENVELOPE, SIZE_MAX_XO, LowDriverBody, LowHorn, MidHorn)

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results/optimise"
F_MID = np.geomspace(150, 16000, 240)
F_LOW = np.geomspace(30, 2500, 200)
XO = {"fostex-t90a": [7000.0], "fostex-t925a": [6000.0], "jbl-2402h": [3000.0, 3500.0]}
XO_LOW = [400.0, 500.0, 630.0]
MAX_WIDTH, MAX_DEPTH = 0.85, 1.20
PLY, BAFFLE, BACK = 0.018, 0.025, 0.018


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
        if not all(p.get(k) for k in ("fs_hz", "re_ohm", "bl_tm", "sd_m2", "mms_kg")):
            continue
        if p.get("le_h") is None or p["le_h"] > 0.002 or p["sd_m2"] <= 0:
            continue
        try:
            d = Driver.from_record(r)
        except (KeyError, TypeError, ZeroDivisionError):
            continue
        env = r.get("physical_envelope_mm")
        frame, depth = SIZE_ENVELOPE[r["nominal_diameter"]]
        if env and env.get("maximum_width") and env.get("depth"):
            frame, depth = env["maximum_width"] / 1000, env["depth"] / 1000
        out.append(dict(rec=r, drv=d, nominal=r["nominal_diameter"], frame=frame, depth=depth,
                        verified=r.get("catalogue_status") == "manufacturer_verified"))
    return out


INSTRUMENT_SERIES = ("legend", "wheel_house", "wheel house", "retrobass", "big_ben", "basslite", "bn15", "cannabis",
                     "red_fang", "swamp", "texas", "governor", "private_jack", "man_o_war")


EXCLUDED = json.loads((ROOT / "data/drivers/excluded.json").read_text())["drivers"]


def low_catalogue(include_excluded=False):
    """15 inch drivers for horn loading: no instrument speakers, no coaxials, at least 4 mm Xmax."""
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
        if (r["parameters"].get("xmax_m") or 0) < 0.004:
            continue
        try:
            out.append(Driver.from_record(r))
        except (KeyError, TypeError, ZeroDivisionError):
            pass
    return out


# ------------------------------------------------------------------ stage 1
def mid_stage(quick=False):
    cat = mid_catalogue()
    rows = []
    profiles = PROFILES if not quick else ("conical", "os")
    deqs = (0.050, 0.064, 0.076, 0.090, 0.110)
    mouths = (0.19, 0.24, 0.29, 0.34, 0.40, 0.46)
    lengths = (0.12, 0.16, 0.20, 0.26, 0.32, 0.40, 0.50)
    for tid in [t for t in TW.all_ids() if t in XO]:
        tw = TW.load(tid)
        for f2 in XO[tid]:
            pool = [c for c in cat if SIZE_MAX_XO[c["nominal"]] >= f2]
            ds = DriverSet.from_drivers([c["drv"] for c in pool])
            twp = lambda th: TW.response(tid, np.array([f2]), th)  # noqa: E731
            bw_tw = float(beamwidth(np.array([f2]), twp)[0])
            seen = set()
            for prof, deq, mouth, length, depth_frac in itertools.product(profiles, deqs, mouths, lengths, (0.0, 0.5)):
                face = round(max(tw.depth_with_terminals + 0.005, depth_frac * length), 4)
                if face > length or (prof, deq, mouth, length, face) in seen:
                    continue
                seen.add((prof, deq, mouth, length, face))
                m = MidHorn(profile=prof, equivalent_throat_diameter=deq, mouth_diameter=mouth, length=length,
                            tweeter_id=tid, tweeter_face=face, tweeter_body_diameter=tw.body_diameter)
                geo = [p for p in m.check() if "compression" not in p]
                if geo:
                    continue
                tr = horn_transfer(F_MID, m.area, length, 200, breaks=(face,))
                R = m.wavefront_radius
                w0 = far_field(F_MID, np.ones_like(F_MID), mouth / 2, 0.0, wavefront_radius=R)
                bwf = lambda th, ff: far_field(ff, np.ones_like(ff), mouth / 2, th, wavefront_radius=R)  # noqa: E731
                bw = {fq: float(beamwidth(np.array([fq]), lambda th: bwf(th, np.array([fq])))[0])
                      for fq in [*XO_LOW, f2]}
                # can and rear volume depend on the driver size: group by nominal size
                for nominal in sorted({c["nominal"] for c in pool}):
                    idx = np.array([c["nominal"] == nominal for c in pool])
                    sub = ds.subset(idx)
                    frame, depth = SIZE_ENVELOPE[nominal]
                    mm = replace(m, frame_diameter=frame, driver_depth=depth, nominal=nominal)
                    cr = sub.sd[:, 0] / m.throat_area
                    s = solve_many(sub, tr, front_m3=25e-6 * (sub.sd[:, 0] / 0.014)[:, None], rear_litres=mm.rear_volume_litres)
                    level = spl(s["u_mouth"] * w0[None, :])
                    for f1 in XO_LOW:
                        if mouth < C / (math.pi * f1):   # mouth must load the driver down to the crossover
                            continue
                        band = (F_MID >= f1) & (F_MID <= f2)
                        lb = level[:, band]
                        ripple = np.ptp(lb, axis=1)
                        mean = lb.mean(axis=1)
                        hp = np.abs(lr4(F_MID, f1, "hp"))[None, :]
                        x50 = np.nanmax(s["x_peak_m"] * hp * (np.sqrt(50 * 8) / 2.83), axis=1)
                        ok = (x50 < np.nan_to_num(sub.xmax[:, 0], nan=0.002)) & (cr >= 1.4) & (cr <= 7.0)
                        mis = abs(math.log2(bw[f2] / bw_tw))
                        score = ripple + 2 * mis + np.maximum(0, 104 - mean) * 0.5
                        cand = np.where(ok)[0]
                        for j in cand[np.argsort(score[cand])][:4]:
                            rows.append(dict(tweeter=tid, f1=f1, f2=f2, driver_id=sub.ids[j], label=sub.labels[j],
                                             nominal=nominal, profile=prof, deq=deq, mouth=mouth, length=length,
                                             tweeter_face=face, ripple_db=ripple[j], mean_db=mean[j], x50_mm=x50[j] * 1000,
                                             compression=cr[j], bw_f1=bw[f1], bw_f2=bw[f2], bw_tweeter=bw_tw,
                                             can_d=mm.can_diameter, can_l=mm.can_length + mm.tail_length,
                                             score=score[j]))
            print(f"mid stage {tid} {f2:.0f} Hz: {len(rows)} rows", flush=True)
    df = pd.DataFrame(rows).sort_values("score")
    return df


def mid_from_row(r, cat_by_id):
    c = cat_by_id[r.driver_id]
    tw = TW.load(r.tweeter)
    return MidHorn(driver_id=r.driver_id, nominal=r.nominal, sd=c["drv"].sd, frame_diameter=c["frame"],
                   driver_depth=c["depth"], profile=r.profile, equivalent_throat_diameter=r.deq,
                   mouth_diameter=r.mouth, length=r.length, tweeter_id=r.tweeter, tweeter_face=r.tweeter_face,
                   tweeter_body_diameter=tw.body_diameter, front_chamber_cc=25.0 * c["drv"].sd / 0.014)


# ------------------------------------------------------------------ stage 2
def low_grid():
    return itertools.product((0.36, 0.40), (0.60, 0.70, 0.80, 0.90), (0.70, 0.85, 1.00, 1.15), (15.0, 20.0, 25.0, 35.0))


def cabinet(wm, length, rear_litres, body=LowDriverBody()):
    inner = wm
    cone = math.pi * body.cone_radius**2 * body.cone_depth / 3
    box = max(rear_litres / 1000 - cone, 0.004)
    chamber_depth = box / inner**2
    depth = BACK + chamber_depth + BAFFLE + length
    width = wm + 2 * PLY
    return width, depth, width * width * depth


def low_stage(mid_rows, cat_by_id):
    lows = low_catalogue()
    ds = DriverSet.from_drivers(lows)
    rows = []
    for _, r in mid_rows.iterrows():
        mid = mid_from_row(r, cat_by_id)
        f1 = r.f1
        band = (F_LOW >= 100) & (F_LOW <= 1.4 * f1)
        for wt, wm, length, vr in low_grid():
            h = LowHorn(throat_width=wt, mouth_width=wm, length=length, mid=mid)
            if h.check():
                continue
            if vr == 15.0:
                tr = horn_transfer(F_LOW, h.acoustic_area, h.acoustic_length, 250, h.breaks,
                                   mouth_area=h.outer_area(h.mouth_z))
                a_out = math.sqrt(h.outer_area(h.mouth_z) / math.pi)
                w0 = far_field(F_LOW, np.ones_like(F_LOW), a_out, 0.0, inner_radius=mid.can_diameter / 2,
                               wavefront_radius=h.wavefront_radius)
                pf = lambda th: far_field(np.array([f1]), np.ones(1), a_out, th, inner_radius=mid.can_diameter / 2,  # noqa: E731
                                          wavefront_radius=h.wavefront_radius)
                bw_low = float(beamwidth(np.array([f1]), pf)[0])
            s = solve_many(ds, tr, front_m3=h.front_chamber_m3, rear_litres=vr)
            level = spl(s["u_mouth"] * w0[None, :])
            lb = level[:, band]
            mean = lb.mean(axis=1)
            ripple = np.ptp(lb, axis=1)
            f3 = F_LOW[np.argmax(level > (mean - 3)[:, None], axis=1)]
            x100 = np.max(s["x_peak_m"][:, band], axis=1) * np.sqrt(800) / 2.83
            ok = (mean >= 103) & (x100 < np.nan_to_num(ds.xmax[:, 0], nan=0))
            width, depth, vol = cabinet(wm, length, vr)
            mis = abs(math.log2(bw_low / r.bw_f1))
            score = ripple + 3 * mis + np.maximum(0, f3 - 75) / 10
            for j in np.where(ok)[0]:
                rows.append(dict(mid_key=r.key, low_driver=ds.ids[j], low_label=ds.labels[j], throat_width=wt,
                                 mouth_width=wm, length=length, rear_litres=vr, ripple_db=ripple[j], mean_db=mean[j],
                                 f3=f3[j], bw_low_f1=bw_low, width=width, depth=depth, volume_m3=vol,
                                 low_score=score[j]))
        print(f"low stage {r.key}: {len(rows)} rows", flush=True)
    return pd.DataFrame(rows)


# ------------------------------------------------------------------ stage 3
def system_stage(combos, mids, cat_by_id):
    from sunflower import simulate
    rows = []
    for _, c in combos.iterrows():
        m = mids.loc[c.mid_key]
        cfg = simulate.Config.from_rows(m, c, cat_by_id)
        res = simulate.evaluate(cfg)
        rows.append(dict(mid_key=c.mid_key, low_driver=c.low_driver, throat_width=c.throat_width,
                         mouth_width=c.mouth_width, length=c.length, rear_litres=c.rear_litres,
                         width=c.width, depth=c.depth, volume_m3=c.volume_m3, **res["scores"]))
    return pd.DataFrame(rows)


def select(best, sysdf):
    feas = sysdf[(sysdf.width <= MAX_WIDTH) & (sysdf.depth <= MAX_DEPTH) & ~sysdf.low_driver.isin(EXCLUDED)]
    floor = feas.score.min()
    choice = feas[feas.score <= floor + 0.25].sort_values("volume_m3").iloc[0]
    m = best.loc[choice.mid_key]
    sel = dict(tweeter=m.tweeter, crossover_hz=[float(m.f1), float(m.f2)],
               mid=dict(driver_id=m.driver_id, label=m.label, nominal=m.nominal, profile=m.profile,
                        equivalent_throat_diameter=float(m.deq), mouth_diameter=float(m.mouth),
                        length=float(m.length), tweeter_face=float(m.tweeter_face)),
               low=dict(driver_id=choice.low_driver, throat_width=float(choice.throat_width),
                        mouth_width=float(choice.mouth_width), length=float(choice.length),
                        rear_litres=float(choice.rear_litres)),
               scores={k: float(choice[k]) for k in ("score", "rms_on_db", "spread_30_db", "f3_hz")},
               cabinet=dict(width=float(choice.width), depth=float(choice.depth), volume_m3=float(choice.volume_m3)),
               rule=f"smallest cabinet within 0.25 of the best score ({floor:.2f}) that fits {MAX_WIDTH*1000:.0f} mm wide "
                    f"and {MAX_DEPTH*1000:.0f} mm deep, excluding drivers in data/drivers/excluded.json")
    (OUT / "selected.json").write_text(json.dumps(sel, indent=2) + "\n")
    print(json.dumps(sel, indent=2))
    return sel


def system_only():
    """Re-run stage 3 from the saved stage 1 and 2 results (after an exclusion change)."""
    cat_by_id = {c["drv"].driver_id: c for c in mid_catalogue()}
    best = pd.read_csv(OUT / "mid-shortlist.csv").set_index("key")
    lows = pd.read_csv(OUT / "low-search.csv.gz")
    allowed = {d.driver_id for d in low_catalogue()}
    lows = lows[lows.low_driver.isin(allowed)].copy()
    lows["pre"] = lows.low_score + best.loc[lows.mid_key, "score"].values
    lows["size_class"] = (lows.volume_m3 / 0.05).round().astype(int)
    lows["tweeter"] = best.loc[lows.mid_key, "tweeter"].values
    combos = (lows.sort_values("pre").groupby(["mid_key", "size_class"], as_index=False).head(1)
              .sort_values("pre").groupby("tweeter").head(20))
    sysdf = system_stage(combos, best, cat_by_id)
    sysdf.to_csv(OUT / "system-search.csv", index=False)
    return select(best, sysdf)


def run(quick=False):
    OUT.mkdir(parents=True, exist_ok=True)
    cat = mid_catalogue()
    cat_by_id = {c["drv"].driver_id: c for c in cat}
    mids = mid_stage(quick)
    mids.to_csv(OUT / "mid-search.csv.gz", index=False)
    # Best mid for each tweeter, crossover pair and can size class
    mids["can_class"] = (mids.can_d / 0.04).round().astype(int)
    best = (mids.groupby(["tweeter", "f1", "f2", "can_class"], as_index=False).head(1)
            .sort_values("score").groupby("tweeter").head(10).reset_index(drop=True))
    best["key"] = [f"m{i:02d}" for i in range(len(best))]
    best.to_csv(OUT / "mid-shortlist.csv", index=False)
    lows = low_stage(best, cat_by_id)
    lows.to_csv(OUT / "low-search.csv.gz", index=False)
    best = best.set_index("key")
    lows["pre"] = lows.low_score + best.loc[lows.mid_key, "score"].values
    lows["size_class"] = (lows.volume_m3 / 0.05).round().astype(int)
    lows["tweeter"] = best.loc[lows.mid_key, "tweeter"].values
    combos = (lows.sort_values("pre").groupby(["mid_key", "size_class"], as_index=False).head(1)
              .sort_values("pre").groupby("tweeter").head(20))
    sysdf = system_stage(combos, best, cat_by_id)
    sysdf.to_csv(OUT / "system-search.csv", index=False)
    select(best, sysdf)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--system-only", action="store_true", help="re-run stage 3 from saved stage 1 and 2 results")
    a = ap.parse_args()
    system_only() if a.system_only else run(a.quick)
