"""The chosen Sunflower design, loaded from the optimiser's selection.

results/optimise/selected.json is written by ``sunflower.optimise``. Driver
parameters and dimensions come from data/drivers/kick-candidates.json and
mid-candidates.json (each record cites its datasheet). Every other script
reads the design from here.
"""
from __future__ import annotations

import json
from pathlib import Path

from sunflower import tweeters as TWS
from sunflower.acoustics import Driver
from sunflower.design_tools import make_kick, make_mid
from sunflower.geometry import load_drivers  # noqa: F401  (re-exported for plots)

ROOT = Path(__file__).resolve().parents[2]
SELECTED = json.loads((ROOT / "results/optimise/selected.json").read_text())

SUPPORT_ROD_DIAMETER = 0.020
SUPPORT_ROD_COUNT = 4

KICK_RECORD = load_drivers("kick")[SELECTED["kick"]["driver_id"]]
MID_RECORD = load_drivers("mid")[SELECTED["mid"]["driver_id"]]
TW = TWS.load(SELECTED["tweeter"])


def label(rec):
    return f"{rec['manufacturer']} {rec['model_name']}"


MID = make_mid(MID_RECORD, TW, **SELECTED["mid"]["geometry"])
KICK = make_kick(KICK_RECORD, MID, **SELECTED["kick"]["geometry"])
CROSSOVER_LOW_MID_HZ, CROSSOVER_MID_HIGH_HZ = SELECTED["crossover_hz"]


def kick_driver() -> Driver:
    return Driver.from_record(KICK_RECORD)


def mid_driver() -> Driver:
    return Driver.from_record(MID_RECORD)


def config(**kw):
    from sunflower.simulate import Config
    return Config(KICK, kick_driver(), MID, mid_driver(), SELECTED["tweeter"],
                  CROSSOVER_LOW_MID_HZ, CROSSOVER_MID_HIGH_HZ, use_hf_fem=True, use_mid_fem=True, **kw)


def chain():
    """The inside-out design chain: each size and what set it."""
    m, k, tw = MID, KICK, TW
    return [
        ("HF unit body cylinder", tw.body_diameter * 1000, f"{tw.model}, flange ignored"),
        ("Mid phase plug (bullet) at the throat", 2 * m.plug_r0 * 1000, "body cylinder, 2 mm sleeve, smooth taper"),
        ("Mid cone (rim)", 2 * m.drv.rim_r * 1000, label(MID_RECORD)),
        ("Mid throat annulus, outer", 2 * m.outer_throat_radius * 1000,
         f"Sd / {m.compression:.1f} round the bullet"),
        ("Mid mouth", m.mouth_diameter * 1000, f"{m.law.upper()} law, {m.length*1000:.0f} mm long"),
        ("Mid pod", 2 * m.pod_radius * 1000, "driver frame plus the wall"),
        ("Kick cone (rim)", 2 * k.drv.rim_r * 1000, label(KICK_RECORD) + ", reversed"),
        ("Kick throat area", k.throat_area * 1e4, f"cone area / {k.compression:.1f}, at the shroud lip"),
        ("Kick centre body, largest", 2 * k.summary()["max_body_radius_m"] * 1000, f"{k.law} law, solved"),
        ("Kick mouth (square, inside)", k.mouth_width * 1000, f"walls at {k.half_angle_deg:.1f} deg"),
    ]


def summary() -> dict:
    k, m = KICK, MID
    return dict(
        kick=dict(driver=label(KICK_RECORD), law=k.law, throat_width_mm=k.throat_width * 1000,
                  mouth_width_mm=k.mouth_width * 1000, protrusion_mm=k.protrusion * 1000,
                  estimated_geometry=list(k.drv.estimated), **k.summary()),
        mid=dict(driver=label(MID_RECORD), law=m.law, compression=m.compression,
                 plug_diameter_mm=2 * m.plug_r0 * 1000, outer_throat_diameter_mm=m.outer_throat_radius * 2000,
                 mouth_diameter_mm=m.mouth_diameter * 1000, length_mm=m.length * 1000,
                 half_angle_deg=m.half_angle_deg, gap_mm=m.gap * 1000, front_cavity_litres=m.front_cavity_litres,
                 rear_litres=m.rear_volume_litres, estimated_geometry=list(m.drv.estimated)),
        tweeter=dict(driver=TW.model, kind=TW.kind, body_diameter_mm=TW.body_diameter * 1000,
                     hf_horn_mouth_mm=2 * m.hf_mouth_radius * 1000, price_usd=TW.price_usd),
        crossover_hz=[CROSSOVER_LOW_MID_HZ, CROSSOVER_MID_HIGH_HZ],
        prices_usd=dict(kick=KICK_RECORD.get("price_usd"), mid=MID_RECORD.get("price_usd"), hf=TW.price_usd),
        selection=SELECTED,
    )
