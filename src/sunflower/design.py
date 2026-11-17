"""The chosen Sunflower design, loaded from the optimiser's selection.

results/optimise/selected.json is written by ``sunflower.optimise``. Driver
parameters come from data/drivers/verified/ when a datasheet-checked record
exists, otherwise from the catalogues. Every other script reads from here.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

from sunflower import tweeters as TWS
from sunflower.acoustics import Driver
from sunflower.geometry import SIZE_ENVELOPE, LowDriverBody, LowHorn, MidHorn

ROOT = Path(__file__).resolve().parents[2]
SELECTED = json.loads((ROOT / "results/optimise/selected.json").read_text())

PLY = 0.018
BAFFLE = 0.025
BACK_BOX = 0.045              # shallow box behind the baffle, joined to the side voids
SUPPORT_ROD_DIAMETER = 0.012
SUPPORT_ROD_COUNT = 4


def _record(driver_id, catalogue):
    v = ROOT / f"data/drivers/verified/{driver_id}.json"
    if v.exists():
        return json.loads(v.read_text())
    recs = json.loads((ROOT / f"data/drivers/{catalogue}").read_text())["drivers"]
    return next(r for r in recs if r["driver_id"] == driver_id)


LOW_RECORD = _record(SELECTED["low"]["driver_id"], "fifteen-inch-catalogue.json")
MID_RECORD = _record(SELECTED["mid"]["driver_id"], "mid-catalogue.json")
TW = TWS.load(SELECTED["tweeter"])


def low_driver() -> Driver:
    return Driver.from_record(LOW_RECORD)


def mid_driver() -> Driver:
    return Driver.from_record(MID_RECORD)


def label(rec):
    return f"{rec['manufacturer']} {rec['model_name']}"


def _low_body():
    p = LOW_RECORD["parameters"]
    b = LOW_RECORD.get("body_mm") or {}
    kw = dict(cone_radius=math.sqrt(p["sd_m2"] / math.pi), xmax=p.get("xmax_m") or 0.005)
    if b:
        kw.update(cone_depth=b["cone_depth"] / 1000, motor_radius=b["motor_diameter"] / 2000,
                  motor_length=b["motor_length"] / 1000)
    return LowDriverBody(**kw)


def _mid():
    s = SELECTED["mid"]
    p = MID_RECORD["parameters"]
    env = MID_RECORD.get("physical_envelope_mm") or {}
    frame, depth = SIZE_ENVELOPE[MID_RECORD["nominal_diameter"]]
    if env.get("maximum_width") and env.get("depth"):
        frame, depth = env["maximum_width"] / 1000, env["depth"] / 1000
    return MidHorn(driver_id=s["driver_id"], nominal=MID_RECORD["nominal_diameter"], sd=p["sd_m2"],
                   xmax=p.get("xmax_m") or 0.003, frame_diameter=frame, driver_depth=depth,
                   factory_sealed=bool(MID_RECORD.get("factory_sealed_back")),
                   hf_id=TW.id, hf_kind=TW.kind, hf_body_diameter=TW.body_diameter, hf_depth=TW.depth_with_terminals,
                   hf_exit_diameter=TW.exit_diameter, hf_horn_length=0.0,
                   law=s["law"], compression=s["compression"], mouth_diameter=s["mouth_diameter"], length=s["length"])


MID = _mid()
LOW_BODY = _low_body()
LOW = LowHorn(mouth_width=SELECTED["low"]["mouth_width"], mouth_height=SELECTED["low"]["mouth_height"],
              length=SELECTED["low"]["length"], law=SELECTED["low"]["law"], body=LOW_BODY, mid=MID)
LOW_REAR_LITRES = SELECTED["low"]["rear_litres"]
CROSSOVER_LOW_MID_HZ, CROSSOVER_MID_HIGH_HZ = SELECTED["crossover_hz"]
REAR_CHAMBER_DEPTH = BACK_BOX


def config():
    from sunflower.simulate import Config
    return Config(LOW, low_driver(), LOW_REAR_LITRES, MID, mid_driver(), SELECTED["tweeter"],
                  CROSSOVER_LOW_MID_HZ, CROSSOVER_MID_HIGH_HZ, use_hf_fem=True)


def chain():
    """The inside-out design chain, each size with what set it."""
    m, lo, tw = MID, LOW, TW
    return [
        ("HF unit body cylinder", tw.body_diameter * 1000, f"{tw.model}, flange ignored"),
        ("Mid phase plug", 2 * m.plug_radius * 1000, "body cylinder plus a 2 mm sleeve"),
        ("Smallest mid cone allowed", 1.6 * 2 * m.plug_radius * 1000, "1.6 times the plug"),
        ("Mid cone used", m.cone_diameter * 1000, label(MID_RECORD)),
        ("Mid throat, outer", 2 * m.outer_throat_radius * 1000,
         f"Sd / {m.compression:.1f} as an annulus round the plug"),
        ("Mid mouth", m.mouth_diameter * 1000, f"{m.law} law, {m.length*1000:.0f} mm long"),
        ("Mid can", m.can_diameter * 1000, "mouth or driver frame plus the wall"),
        ("Low centre body at the throat", 2 * lo.throat_radius * 1000, "covers the 15 inch cone in front of it"),
        ("Low throat area", lo.throat_area * 1e4, f"{lo.compression:.1f}:1 compression"),
        ("Low mouth", lo.mouth_width * 1000, f"x {lo.mouth_height*1000:.0f} mm, {lo.law} law to the can"),
    ]


def summary() -> dict:
    return dict(
        low=dict(driver=label(LOW_RECORD), throat_width_mm=LOW.throat_width * 1000,
                 mouth_width_mm=LOW.mouth_width * 1000, mouth_height_mm=LOW.mouth_height * 1000,
                 length_mm=LOW.length * 1000, law=LOW.law, rear_chamber_litres=LOW_REAR_LITRES, **LOW.summary()),
        mid=dict(driver=label(MID_RECORD), nominal=MID.nominal, law=MID.law, compression=MID.compression,
                 plug_diameter_mm=2 * MID.plug_radius * 1000, outer_throat_diameter_mm=MID.outer_throat_radius * 2000,
                 mouth_diameter_mm=MID.mouth_diameter * 1000, length_mm=MID.length * 1000,
                 can_diameter_mm=MID.can_diameter * 1000, can_length_mm=MID.can_length * 1000),
        tweeter=dict(driver=TW.model, kind=TW.kind, body_diameter_mm=TW.body_diameter * 1000,
                     hf_horn_mouth_mm=2 * MID.hf_mouth_radius * 1000, price_usd=TW.price_usd),
        crossover_hz=[CROSSOVER_LOW_MID_HZ, CROSSOVER_MID_HIGH_HZ],
        selection=SELECTED,
    )
