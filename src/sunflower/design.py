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

# Cabinet and panel stock
PLY = 0.018
BAFFLE = 0.025
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


def low_driver() -> Driver:
    return Driver.from_record(LOW_RECORD)


def mid_driver() -> Driver:
    return Driver.from_record(MID_RECORD)


def _low_body():
    b = LOW_RECORD.get("body_mm")
    sd = LOW_RECORD["parameters"]["sd_m2"]
    if b:
        return LowDriverBody(cone_radius=math.sqrt(sd / math.pi), cone_depth=b["cone_depth"] / 1000,
                             motor_radius=b["motor_diameter"] / 2000, motor_length=b["motor_length"] / 1000,
                             nose_length=b["nose_length"] / 1000)
    return LowDriverBody(cone_radius=math.sqrt(sd / math.pi))


def _mid():
    s = SELECTED["mid"]
    tw = TWS.load(SELECTED["tweeter"])
    env = MID_RECORD.get("physical_envelope_mm") or {}
    frame, depth = SIZE_ENVELOPE[MID_RECORD["nominal_diameter"]]
    if env.get("maximum_width") and env.get("depth"):
        frame, depth = env["maximum_width"] / 1000, env["depth"] / 1000
    sd = MID_RECORD["parameters"]["sd_m2"]
    return MidHorn(driver_id=s["driver_id"], nominal=MID_RECORD["nominal_diameter"], sd=sd, frame_diameter=frame,
                   driver_depth=depth, profile=s["profile"], equivalent_throat_diameter=s["equivalent_throat_diameter"],
                   mouth_diameter=s["mouth_diameter"], length=s["length"], tweeter_id=SELECTED["tweeter"],
                   tweeter_face=s["tweeter_face"], tweeter_body_diameter=tw.body_diameter,
                   front_chamber_cc=25.0 * sd / 0.014)


TW = TWS.load(SELECTED["tweeter"])
MID = _mid()
LOW_BODY = _low_body()
LOW = LowHorn(throat_width=SELECTED["low"]["throat_width"], mouth_width=SELECTED["low"]["mouth_width"],
              length=SELECTED["low"]["length"], body=LOW_BODY, mid=MID)
LOW_REAR_LITRES = SELECTED["low"]["rear_litres"]
CROSSOVER_LOW_MID_HZ, CROSSOVER_MID_HIGH_HZ = SELECTED["crossover_hz"]

# Rear chamber depth that gives the selected net volume behind the baffle:
# box plus the concave side of the cone minus the flange.
_cone = math.pi * LOW_BODY.cone_radius**2 * LOW_BODY.cone_depth / 3
REAR_CHAMBER_DEPTH = max(LOW_REAR_LITRES / 1000 - _cone, 0.004) / LOW.mouth_width**2


def config():
    from sunflower.simulate import Config
    return Config(LOW, low_driver(), LOW_REAR_LITRES, MID, mid_driver(), SELECTED["tweeter"],
                  CROSSOVER_LOW_MID_HZ, CROSSOVER_MID_HIGH_HZ)


def label(rec):
    return f"{rec['manufacturer']} {rec['model_name']}"


def summary() -> dict:
    return dict(
        low=dict(driver=label(LOW_RECORD), throat_width_mm=LOW.throat_width * 1000,
                 mouth_width_mm=LOW.mouth_width * 1000, length_mm=LOW.length * 1000,
                 rear_chamber_litres=LOW_REAR_LITRES, **LOW.summary()),
        mid=dict(driver=label(MID_RECORD), nominal=MID.nominal, profile=MID.profile,
                 outer_throat_diameter_mm=MID.outer_throat_radius * 2000,
                 mouth_diameter_mm=MID.mouth_diameter * 1000, length_mm=MID.length * 1000,
                 compression_ratio=MID.compression_ratio,
                 can_diameter_mm=MID.can_diameter * 1000, can_length_mm=MID.can_length * 1000),
        tweeter=dict(driver=TW.model, body_diameter_mm=TW.body_diameter * 1000,
                     face_from_mid_throat_mm=MID.tweeter_face * 1000,
                     face_recess_mm=(MID.length - MID.tweeter_face) * 1000),
        crossover_hz=[CROSSOVER_LOW_MID_HZ, CROSSOVER_MID_HIGH_HZ],
        selection=SELECTED,
    )
