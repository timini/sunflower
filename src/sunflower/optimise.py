"""Sunflower optimiser: inside out, Funktion-One Res 2 band split.

Stage 1, HF and mid. For each 1 inch compression driver, the mid phase plug
is a smooth bullet round its body cylinder, tapering to a printed HF horn.
Each 8 or 10 inch mid driver is tried in an OS horn round that bullet,
over compression ratio, mouth and length. Each design is solved in its
geometry network (``sunflower.models``) with the cone as a distributed
source. The crossover to the kick is fixed at 250 Hz, as in the Res 2. The
upper crossover f2 is the highest of 3.5 to 5.5 kHz the mid can reach.
Score:
- ripple from 250 Hz to f2, after the best high shelf of up to 12 dB;
- the shelf gain it needs;
- droop at 250 Hz;
- the coverage step to the HF unit at f2;
- output;
- a reward for a higher f2 (the mid should carry the whole vocal range).

Stage 2, kick. The best mids become the core of the centre body. Each kick
driver is tried, reversed, over throat compression, area law, wall angle
and mid protrusion. Each cabinet is 600 mm deep at most; the sealed chamber
takes whatever depth is left. Score:
- the free-space -6 dB point against 95 Hz (HD-15-like);
- ripple and output from there to 1.3 f1;
- excursion at rated power;
- horizontal splay against the mid's coverage (0.8x, the point-source rule);
- width.

Stage 3, system. The best combinations go through the crossover with tuned
gains, EQ, delays and polarity (``simulate.evaluate``). Among designs within
0.25 of the best, the cheapest wins.

Outputs are in results/optimise/.
"""
from __future__ import annotations

import argparse
import itertools
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

from sunflower import tweeters as TW
from sunflower.acoustics import Driver, beamwidth, couple, far_field, baffle_step, spl
from sunflower.design_tools import make_kick, make_mid
from sunflower.geometry import load_drivers
from sunflower.models import kick_network, mid_network

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results/optimise"
F_MID = np.geomspace(150, 12000, 320)
F_LOW = np.geomspace(30, 1200, 110)
F1 = 250.0
F2S = (4000.0, 4500.0, 5000.0)   # Res 2 style: the mid carries the voice up to 4 to 5 kHz
MAX_DEPTH = 0.60
EXCLUDED = json.loads((ROOT / "data/drivers/excluded.json").read_text())["drivers"]


def _shelf(f, fc, g):
    from sunflower.simulate import high_shelf
    return high_shelf(f, fc, g)


# ------------------------------------------------------------------ stage 1
def mid_stage(quick=False):
    mids = {k: r for k, r in load_drivers("mid").items() if k not in EXCLUDED and r["parameters"]["fs_hz"] < F1 / 2
            and "discontinued" not in str(r.get("status", "")).lower()}
    hfs = [t for t in TW.all_ids() if TW.load(t).kind == "compression"]
    grid = dict(compression=(1.5, 1.8, 2.2, 2.7), mouth_diameter=(0.36, 0.40, 0.44), length=(0.16, 0.20, 0.24, 0.28, 0.32, 0.36),
                hf_mouth_ratio=(1.8, 2.4))
    if quick:
        grid = dict(compression=(1.8, 2.2), mouth_diameter=(0.40,), length=(0.28, 0.32), hf_mouth_ratio=(1.8, 2.4))
    rows = []
    for tid in hfs:
        tw = TW.load(tid)
        for mid_id, rec in mids.items():
            drv = Driver.from_record(rec)
            for cr, mo, ln, hr in itertools.product(*grid.values()):
                m = make_mid(rec, tw, law="os", compression=cr, mouth_diameter=mo, length=ln, hf_mouth_ratio=hr)
                hf_bw = {f2: beamwidth(np.array([f2]), lambda th: TW.response(tid, np.array([f2]), th,
                                                                            mouth_radius=m.hf_mouth_radius))[0]
                         for f2 in F2S}
                if m.check():
                    continue
                r = mid_network(m).solve(F_MID)
                s = couple(drv, r, rear_litres=m.rear_volume_litres)
                fp = lambda th: far_field(F_MID, s["u_mouth"], mo / 2, th, wavefront_radius=m.wavefront_radius)  # noqa: E731
                lvl = spl(fp(0.0))
                bw = beamwidth(F_MID, fp)
                for f2 in F2S:
                    if f2 < 1.5 * tw.crossover_min_hz:
                        continue
                    b = (F_MID >= F1) & (F_MID <= f2)
                    hb = (F_MID >= f2 / 3) & (F_MID <= f2)
                    fc = 0.55 * f2
                    g = min(np.arange(0, 12.01, 0.5), key=lambda g: np.std(lvl[hb] + 20 * np.log10(np.abs(_shelf(F_MID[hb], fc, g)))))
                    eq = lvl + 20 * np.log10(np.abs(_shelf(F_MID, fc, g)))
                    ref = np.median(eq[b])
                    ripple = float(np.std(eq[b] - ref))
                    droop = float(max(0.0, ref - eq[F_MID >= F1][0]))
                    dip = float(max(0.0, (ref - eq[b]).max() - 6))
                    bw_f2 = float(np.interp(f2, F_MID, bw))
                    cov = 1.5 * abs(math.log2(bw_f2 / hf_bw[f2]))     # about the directivity-index step at f2
                    out = float(np.median(lvl[b]))
                    score = 2 * ripple + 0.5 * dip + 0.15 * g + 0.3 * max(0.0, droop - 3) + cov - 0.1 * (out - 100) \
                        - 0.6 * (f2 - 4000) / 1000
                    rows.append(dict(tweeter=tid, mid_driver=mid_id, compression=cr, mouth_diameter=mo, length=ln,
                                     hf_mouth_ratio=hr,
                                     f2=f2, shelf_db=float(g), ripple_db=ripple, droop_db=droop, mid_bw_f2=bw_f2,
                                     hf_bw_f2=float(hf_bw[f2]), mid_bw_1_4k=float(np.median(bw[(F_MID >= 1000) & (F_MID <= 4000)])),
                                     level_db=out, half_angle=m.half_angle_deg, score=score,
                                     price=(rec.get("price_usd") or 300) + (tw.price_usd or 200)))
        print(f"mid stage: {tid} done, {len(rows)} rows")
    df = pd.DataFrame(rows).sort_values("score")
    df.to_csv(OUT / "mid-search.csv", index=False, float_format="%.5g")
    return df


# ------------------------------------------------------------------ stage 2
def kick_stage(mid_rows, quick=False):
    kicks = {k: r for k, r in load_drivers("kick").items() if k not in EXCLUDED}
    grid = dict(compression=(1.6, 2.0, 2.5, 3.0), law=("exponential", "hypex", "conical"),
                half_angle_deg=(9.0, 11.0, 13.0, 15.0, 17.0), protrusion=(0.0,),   # owner's rule: mid mouth inside the kick mouth
                wedge_chamber=(False, True))
    if quick:
        grid = dict(compression=(2.0, 2.5), law=("exponential", "hypex"), half_angle_deg=(11.0, 13.0), protrusion=(0.0,),
                    wedge_chamber=(False, True))
    rows = []
    mids = load_drivers("mid")
    for _, mr in mid_rows.iterrows():
        tw = TW.load(mr.tweeter)
        mid = make_mid(mids[mr.mid_driver], tw, law="os", compression=mr.compression,
                       mouth_diameter=mr.mouth_diameter, length=mr.length, hf_mouth_ratio=mr.hf_mouth_ratio,
                       gap_override=float(mr.get("gap_override", 0.0) or 0.0))
        for kid, rec in kicks.items():
            drv = Driver.from_record(rec)
            for cr, law, ha, pr, wc in itertools.product(*grid.values()):
                k0 = make_kick(rec, mid, compression=cr, law=law, half_angle_deg=ha, protrusion=pr, chamber_depth=0.05,
                               wedge_chamber=wc)
                lc = MAX_DEPTH - (k0.depth - k0.chamber_depth)
                if lc < 0.04:
                    continue
                k = make_kick(rec, mid, compression=cr, law=law, half_angle_deg=ha, protrusion=pr,
                              chamber_depth=round(min(lc, 0.20), 3), wedge_chamber=wc)
                if k.check():
                    continue
                net = kick_network(k)
                s = couple(drv, net.solve(F_LOW), rear_litres=k.chamber_litres)
                p = far_field(F_LOW, s["u_mouth"], net.mouth_radius, 0.0, inner_radius=net.mouth_inner)
                lvl = spl(p * baffle_step(F_LOW, k.external_mouth))
                ref = lvl[(F_LOW >= 150) & (F_LOW <= F1)].mean()
                f6 = float(F_LOW[np.argmax(lvl > ref - 6)])
                f3 = float(F_LOW[np.argmax(lvl > ref - 3)])
                b = (F_LOW >= f3) & (F_LOW <= 1.3 * F1)
                lf = np.log2(F_LOW[b])
                fit = np.polyfit(lf, lvl[b], 1)
                ripple = float(np.std(lvl[b] - np.polyval(fit, lf)))
                xr = np.sqrt((rec["parameters"]["power_w"] or 400) * 8) / 2.83
                xb = (F_LOW >= 80) & (F_LOW <= F1)
                xratio = float((s["x_peak_m"][xb] * xr).max() / drv.xmax)
                splay = 2 * ha / mr.mid_bw_1_4k
                score = (max(0.0, f6 - 95) / 10 + ripple + 0.1 * abs(fit[0]) - 0.1 * (ref - 100)
                         + 3 * abs(splay - 0.8) + 4 * max(0.0, k.external_mouth - 0.75) + 2 * max(0.0, xratio - 1))
                rows.append(dict(tweeter=mr.tweeter, mid_driver=mr.mid_driver, mid_compression=mr.compression,
                                 hf_mouth_ratio=mr.hf_mouth_ratio, gap_override=float(mr.get("gap_override", 0.0) or 0.0),
                                 mid_mouth=mr.mouth_diameter, mid_length=mr.length, f2=mr.f2, kick_driver=kid,
                                 compression=cr, law=law, half_angle_deg=ha, protrusion=pr, wedge_chamber=wc,
                                 chamber_depth=k.chamber_depth, chamber_litres=k.chamber_litres, f3=f3, f6=f6,
                                 ripple_db=ripple, slope_db_oct=float(fit[0]), level_db=float(ref), xratio=xratio,
                                 splay_ratio=splay, external_mouth=k.external_mouth, depth=k.depth, score=score,
                                 price=(rec.get("price_usd") or 350)))
        print(f"kick stage: mid {mr.mid_driver} done, {len(rows)} rows")
    df = pd.DataFrame(rows).sort_values("score")
    df.to_csv(OUT / "kick-search.csv", index=False, float_format="%.5g")
    return df


# ------------------------------------------------------------------ stage 3
def selection(row):
    return dict(
        tweeter=row.tweeter,
        mid=dict(driver_id=row.mid_driver, geometry=dict(law="os", compression=float(row.mid_compression),
                                                         mouth_diameter=float(row.mid_mouth), length=float(row.mid_length),
                                                         hf_mouth_ratio=float(row.hf_mouth_ratio),
                                                         gap_override=float(getattr(row, "gap_override", 0.0) or 0.0))),
        kick=dict(driver_id=row.kick_driver, geometry=dict(compression=float(row.compression), law=row.law,
                                                           half_angle_deg=float(row.half_angle_deg),
                                                           protrusion=float(row.protrusion),
                                                           chamber_depth=float(row.chamber_depth),
                                                           wedge_chamber=bool(row.wedge_chamber))),
        crossover_hz=[F1, float(row.f2)])


def system_stage(kick_rows, n=12):
    from sunflower import simulate as S
    from sunflower.design_tools import config_from
    rows = []
    seen = set()
    for _, r in kick_rows.iterrows():
        key = (r.tweeter, r.mid_driver, r.kick_driver)
        if key in seen:
            continue
        seen.add(key)
        sel = selection(r)
        res = S.evaluate(config_from(sel))
        sc = res["scores"]
        total = sc["score"] + 3 * abs(r.splay_ratio - 0.8) + 4 * max(0.0, r.external_mouth - 0.75)
        tw = TW.load(r.tweeter)
        price = r.price + (load_drivers("mid")[r.mid_driver].get("price_usd") or 300) + (tw.price_usd or 200)
        rows.append(dict(**r.to_dict(), system_score=total, price_total=price, **{f"sys_{k}": v for k, v in sc.items()},
                         selection=json.dumps(sel)))
        if len(rows) >= n:
            break
    df = pd.DataFrame(rows).sort_values("system_score")
    df.to_csv(OUT / "system-search.csv", index=False, float_format="%.5g")
    best = df.system_score.min()
    pick = df[df.system_score <= best + 0.25].sort_values("price_total").iloc[0]
    sel = json.loads(pick.selection)
    sel["scores"] = dict(system=float(pick.system_score), **{k[4:]: float(pick[k]) for k in df.columns if k.startswith("sys_")})
    sel["rule"] = "lowest system score; among designs within 0.25 of it, the cheapest"
    (OUT / "selected.json").write_text(json.dumps(sel, indent=2) + "\n")
    print(json.dumps(sel, indent=2))
    return sel


def revise_f2():
    """After the FEM: re-pick the upper crossover with the FEM-corrected mid.

    Take the highest crossover from 2.5 to 4 kHz that scores within 0.15 of the best of them (the FEM, not the
    network, sets how high the mid can go)."""
    from dataclasses import replace
    from sunflower import simulate as S
    from sunflower.design_tools import config_from
    sel = json.loads((OUT / "selected.json").read_text())
    cfg = config_from(sel, use_hf_fem=True, use_mid_fem=True)
    scores = {f2: S.evaluate(replace(cfg, f2=f2))["scores"]["score"] for f2 in (2500.0, 3000.0, 3500.0, 4000.0)}
    best = min(scores.values())
    f2 = max(f for f, v in scores.items() if v <= best + 0.15)
    sel["crossover_hz"] = [F1, f2]
    sel["f2_after_fem"] = {f"{k:.0f}": round(v, 3) for k, v in scores.items()}
    sel["crossover_note"] = ("The network model alone picks a higher crossover. The FEM shows the mid falling 8 to 12 dB from "
                             "3 to 6 kHz (a radial cavity mode over the cone that the network lumps away), so f2 is re-picked "
                             "with that correction: the highest of 2.5 to 4 kHz scoring within 0.15 of the best.")
    (OUT / "selected.json").write_text(json.dumps(sel, indent=2) + "\n")
    print("f2 after FEM:", f2, sel["f2_after_fem"])


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--max-depth", type=float, default=0.60, help="cabinet depth limit, m")
    ap.add_argument("--out", type=Path, default=None, help="output folder (default results/optimise)")
    ap.add_argument("--from-kick", action="store_true", help="reuse mid-search.csv")
    ap.add_argument("--system-only", action="store_true", help="reuse kick-search.csv")
    ap.add_argument("--fem-f2", action="store_true", help="after the FEM: re-pick the upper crossover")
    a = ap.parse_args()
    global MAX_DEPTH, OUT
    MAX_DEPTH = a.max_depth
    if a.out:
        OUT = a.out
    if a.fem_f2:
        return revise_f2()
    OUT.mkdir(parents=True, exist_ok=True)
    if a.system_only:
        kick = pd.read_csv(OUT / "kick-search.csv")
    else:
        mid = pd.read_csv(OUT / "mid-search.csv") if a.from_kick else mid_stage(a.quick)
        # the depth limit decides which mid lengths fit, so carry the best of each driver and length
        best = mid.drop_duplicates(["mid_driver", "length"]).groupby("length").head(3)
        kick = kick_stage(best, a.quick)
    system_stage(kick)


if __name__ == "__main__":
    main()
