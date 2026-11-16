"""The chosen Sunflower design. Every script reads dimensions from here."""
from __future__ import annotations

import json
import math
from pathlib import Path

from sunflower.acoustics import Driver
from sunflower.geometry import LowDriverBody, LowHorn, MidHorn, Tweeter

ROOT = Path(__file__).resolve().parents[2]

# 15NMB420, values from the 18 Sound datasheet (checked 2026-11-15).
# Motor diameter and the split between cone depth and motor length are
# estimated from the datasheet outline drawing: measure your driver.
LOW_BODY = LowDriverBody(
    cone_radius=math.sqrt(0.090 / math.pi),
    cone_depth=0.105,
    motor_radius=0.075,
    motor_length=0.064,
    nose_length=0.120,
)
MID = MidHorn(front_chamber_cc=25.0)   # cone-following plug, 1.5 mm clearance
TWEETER = Tweeter()
LOW = LowHorn(throat_width=0.36, mouth_width=0.70, length=0.80, body=LOW_BODY, mid=MID)

LOW_REAR_LITRES = 25.0
CROSSOVER_LOW_MID_HZ = 500.0
CROSSOVER_MID_HIGH_HZ = 7000.0
TWEETER_ACOUSTIC_CENTRE_BEHIND_FACE = 0.030   # assumption, T90A internal horn

# Cabinet and panel stock
PLY = 0.018          # birch ply panels
BAFFLE = 0.025       # 15 inch baffle, two layers of 12.5 mm or one 25 mm
REAR_CHAMBER_DEPTH = 0.045
SUPPORT_ROD_DIAMETER = 0.010
SUPPORT_ROD_COUNT = 4


def low_driver() -> Driver:
    path = ROOT / "data/drivers/verified/18sound-15nmb420.json"
    return Driver.from_record(json.loads(path.read_text()))


def mid_driver() -> Driver:
    return Driver.from_record(json.loads((ROOT / "data/drivers/verified/beyma-6mcf200nd.json").read_text()))


def summary() -> dict:
    return dict(
        low=dict(driver="18 Sound 15NMB420", throat_width_mm=LOW.throat_width * 1000,
                 mouth_width_mm=LOW.mouth_width * 1000, length_mm=LOW.length * 1000,
                 rear_chamber_litres=LOW_REAR_LITRES, **LOW.summary()),
        mid=dict(driver="Beyma 6MCF200Nd", profile="conical",
                 outer_throat_diameter_mm=MID.outer_throat_radius * 2000,
                 mouth_diameter_mm=MID.mouth_diameter * 1000, length_mm=MID.length * 1000,
                 can_diameter_mm=MID.can_diameter * 1000, can_length_mm=MID.can_length * 1000),
        tweeter=dict(driver=TWEETER.model, housing_diameter_mm=TWEETER.housing_diameter * 1000,
                     face_recess_mm=(MID.length - TWEETER.face_from_mid_throat) * 1000),
        crossover_hz=[CROSSOVER_LOW_MID_HZ, CROSSOVER_MID_HIGH_HZ],
    )
