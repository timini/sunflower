"""The Sunflower range: the same rules and checks applied to several driver line-ups.

Each variant fixes the HF unit, the mid driver and the kick size class, then the kick stage of the optimiser
picks the kick driver and horn inside a 700 mm box. The mid is checked on FEM (the network overstates its top
end) and the upper crossover is the highest of 2.5, 3 and 3.5 kHz where the FEM mid is within 3 dB of its
mid-band level. Centre-body material is measured two ways: the space the body takes in the kick horn, and the
plastic in a 6 mm printed shell with ribs (the interior then becomes the mid's sealed rear chamber).

Writes results/range/<variant>/selected.json and results/range/summary.json.
"""
from __future__ import annotations

import json
import math
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results/range"

VARIANTS = {
    "15C": dict(title="15 inch kick, 8 inch mid, compression HF", hf="bc-de111", mid="beyma-8mi100", mid_cr=2.0, kick="15in"),
    "15C6": None,
    "15B": dict(title="15 inch kick, 8 inch mid, bullet HF in a big plug at the mid throat", hf="ds18-pro-tw920",
                mid="beyma-8mi100", mid_cr=4.0, kick="15in", hf_mount="throat", plug_cover=0.65),
    "12C": dict(title="12 inch kick, 8 inch mid, compression HF", hf="bc-de111", mid="beyma-8mi100", mid_cr=2.0, kick="12in",
                ),
    "12C6": None,
}
SHELL = 0.006
MAX_DEPTH, MAX_SIZE = 0.60, 0.60     # owner's limit: every box fits a 600 mm cube
RIBS = 1.25          # ribs and bosses add about a quarter to the shell


def body_material(k):
    """Envelope of the centre body in the kick horn (litres) and plastic in a ribbed 6 mm shell (litres)."""
    z = np.linspace(k.turn_z, k.mouth_z, 400)
    rb = np.array([k.body_radius(v) for v in z])
    envelope = float(np.trapezoid(np.pi * rb**2, z)) * 1000
    ds = np.hypot(np.diff(z), np.diff(rb))
    side = float(np.sum(2 * np.pi * 0.5 * (rb[1:] + rb[:-1]) * ds))
    r = np.linspace(k.channel_start, k.drv.rim_r, 100)
    sz = k.shroud_z(r)
    under = float(np.sum(2 * np.pi * 0.5 * (r[1:] + r[:-1]) * np.hypot(np.diff(r), np.diff(sz))))
    lip = math.pi * (k.body_radius(k.turn_z) ** 2 - k.drv.rim_r**2)
    shell = (side + under + lip) * SHELL * RIBS * 1000
    return envelope, shell


def mid_top_fem(m, kick_rec, name):
    from sunflower import design as D, fem as FE
    from sunflower.acoustics import Driver, couple, spl
    from sunflower.design_tools import make_kick
    from sunflower.models import mid_network
    D.MID, D.KICK = m, make_kick(kick_rec, m, protrusion=0.0)
    f = np.geomspace(300, 6000, 26)
    raw, _ = FE.run_case(f"range-{name}", FE.case_mid(freqs=f, flush=True), 10)
    rn = mid_network(m).solve(raw.f.values)
    from sunflower.geometry import load_drivers
    drv = Driver.from_record(load_drivers("mid")[m.drv.driver_id])
    sf = couple(drv, FE.fem_net(raw, rn["load"]), rear_litres=m.rear_volume_litres)
    p = spl(FE.cplx(raw, "h0") * sf["vel"])
    ff = raw.f.values
    ref = float(np.median(p[(ff >= 400) & (ff <= 1500)]))
    e6 = ff[(ff > 1500) & (p < ref - 6)]
    f2 = max([x for x in (2500.0, 3000.0, 3500.0) if np.interp(x, ff, p) >= ref - 3] or [2500.0])
    return float(e6.min()) if len(e6) else 6000.0, f2


def run(names=None):
    import sunflower.optimise as O
    from sunflower import simulate as S, tweeters as TW
    from sunflower.design_tools import config_from, make_mid, parts_from
    from sunflower.geometry import load_drivers
    mids, kicks = load_drivers("mid"), load_drivers("kick")
    ms = pd.read_csv(ROOT / "results/optimise/depth-600/mid-search.csv")
    summary = json.loads((OUT / "summary.json").read_text()) if (OUT / "summary.json").exists() else {}
    for name, v in VARIANTS.items():
        if v is None or (names and name not in names):
            continue
        try:
            TW.load(v["hf"])
        except FileNotFoundError:
            print(name, "skipped: no data for", v["hf"])
            continue
        out = OUT / name
        out.mkdir(parents=True, exist_ok=True)
        rec = mids[v["mid"]]
        ref = ms[ms.mid_driver == v["mid"]].sort_values("score").iloc[0]
        cand = []
        for L in (0.16, 0.20, 0.24, 0.28):
            for mouth in (0.28, 0.32, 0.36):
                for ang in v.get("hf_angles", (15.0, 20.0, 25.0)):
                    b = ref.copy()
                    b["tweeter"], b["compression"], b["length"], b["mouth_diameter"] = v["hf"], v["mid_cr"], L, mouth
                    b["gap_override"] = (rec["parameters"].get("xmax_m") or 0.004) + 0.0015
                    b["f2"] = 3000.0
                    b["hf_law"], b["hf_half_angle_deg"] = "os", ang
                    b["hf_mount"], b["plug_cover"] = v.get("hf_mount", "bullet"), v.get("plug_cover", 0.0)
                    cand.append(b)
        O.OUT, O.MAX_DEPTH, O.MAX_SIZE = out, MAX_DEPTH, MAX_SIZE
        O.KICK_IDS = [k for k, r in kicks.items() if r.get("nominal_diameter") == v["kick"]]
        kick = O.kick_stage(pd.DataFrame(cand))
        if kick.empty:
            print(name, "nothing fits")
            continue
        sel = O.system_stage(kick, n=6)
        krec, mrec, tw, M, K = parts_from(sel)
        top6, f2 = mid_top_fem(M, krec, name)
        sel["crossover_hz"] = [250.0, f2]
        (out / "selected.json").write_text(json.dumps(sel, indent=2) + "\n")
        cfg = config_from(sel)
        ev = S.evaluate(cfg)
        fl = np.geomspace(30, 600, 200)
        half = S.spl(S.low_section(replace(cfg, free_space=False), fl)[0]) if hasattr(S, "spl") else None
        from sunflower.acoustics import spl
        half = spl(S.low_section(replace(cfg, free_space=False), fl)[0])
        ref = half[(fl >= 150) & (fl <= 250)].mean()
        env, shell = body_material(K)
        price = sum(x or 0 for x in (krec.get("price_usd"), mrec.get("price_usd"), tw.price_usd))
        summary[name] = dict(title=v["title"], kick=f"{krec['manufacturer']} {krec['model_name']}",
                             mid=f"{mrec['manufacturer']} {mrec['model_name']}", hf=tw.model, hf_kind=tw.kind,
                             crossover_hz=[250.0, f2], mid_fem_minus6_hz=top6,
                             kick_minus6_ground_hz=float(fl[np.argmax(half > ref - 6)]),
                             sensitivity_db=ev["scores"]["sensitivity_db"], rms_100_16k_db=ev["scores"]["rms_on_db"],
                             external_mm=[K.external_mouth * 1000, K.external_height * 1000, K.depth * 1000],
                             body_envelope_litres=env, body_shell_litres=shell, mid_pod_litres=M.rear_volume_litres,
                             horn_air_litres=float(np.trapezoid([K.area(z) for z in np.linspace(K.turn_z, K.mouth_z, 300)],
                                                                np.linspace(K.turn_z, K.mouth_z, 300))) * 1000,
                             drivers_usd=price, checks=K.check() + M.check())
        print(name, json.dumps(summary[name]), flush=True)
        (OUT / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


if __name__ == "__main__":
    import sys
    run(sys.argv[1:] or None)
