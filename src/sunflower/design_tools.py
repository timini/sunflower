"""Build Sunflower parts from driver records and a selection (no dependence on the chosen design)."""
from __future__ import annotations

from sunflower import tweeters as TWS
from sunflower.acoustics import Driver
from sunflower.geometry import ConeDriver, Kick, MidHorn, load_drivers


def make_mid(rec, tw, **kw):
    return MidHorn(ConeDriver.from_record(rec, "mid"), hf_id=tw.id, hf_kind=tw.kind,
                   hf_body_diameter=tw.body_diameter, hf_depth=tw.depth_with_terminals,
                   hf_exit_diameter=tw.exit_diameter, **kw)


def make_kick(rec, mid, **kw):
    return Kick(ConeDriver.from_record(rec, "kick"), mid, **kw)


def parts_from(sel):
    krec = load_drivers("kick")[sel["kick"]["driver_id"]]
    mrec = load_drivers("mid")[sel["mid"]["driver_id"]]
    tw = TWS.load(sel["tweeter"])
    mid = make_mid(mrec, tw, **sel["mid"]["geometry"])
    kick = make_kick(krec, mid, **sel["kick"]["geometry"])
    return krec, mrec, tw, mid, kick


def config_from(sel, **kw):
    from sunflower.simulate import Config
    krec, mrec, tw, mid, kick = parts_from(sel)
    f1, f2 = sel["crossover_hz"]
    return Config(kick, Driver.from_record(krec), mid, Driver.from_record(mrec), sel["tweeter"], f1, f2, **kw)
