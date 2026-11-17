import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from sunflower import design as D
from sunflower.acoustics import Driver, far_field, horn_transfer, solve, spl
from sunflower.geometry import LowHorn, MidHorn

ROOT = Path(__file__).resolve().parents[1]


def test_mid_port_matches_horn_simulation_branch_study():
    """Same geometry and inputs as horn-simulation codex/6p5-mid-500-6500."""
    ref = pd.read_csv(ROOT / "tests/reference/branch-study-beyma-6mcf200nd.csv")
    f = ref.frequency_hz.values
    mid = MidHorn(front_chamber_cc=20.0)
    tr = horn_transfer(f, mid.area, mid.length, 600, breaks=(0.1,))
    beyma = Driver.from_record(json.loads((ROOT / "data/drivers/verified/beyma-6mcf200nd.json").read_text()))
    s = solve(beyma, tr, front_m3=20e-6, rear_litres=None)
    level = spl(far_field(f, s["u_mouth"], mid.mouth_diameter / 2))
    band = (f >= 500) & (f <= 6500)
    assert np.max(np.abs(level[band] - ref.spl_db.values[band])) < 0.5
    assert abs(level[band].mean() - 104.44) < 0.5
    assert abs(np.ptp(level[band]) - 3.55) < 0.5


def test_straight_tube_matches_analytic_input_impedance():
    f = np.geomspace(50, 2000, 200)
    a, L, S = 0.05, 0.3, np.pi * 0.05**2
    tr = horn_transfer(f, lambda z: S, L, 50)
    k = 2 * np.pi * f / 343.0
    zc = 1.225 * 343.0 / S
    exact = zc * (tr.load + 1j * zc * np.tan(k * L)) / (zc + 1j * tr.load * np.tan(k * L))
    assert np.max(np.abs(tr.z_in - exact) / np.abs(exact)) < 1e-9


def test_area_functions_are_continuous_where_expected():
    lo = D.LOW
    z = np.linspace(lo.throat_z, lo.mouth_z, 4000)
    a = np.array([lo.area(v) for v in z])
    assert np.all(a > 0)
    jumps = np.abs(np.diff(a)) / a[:-1]
    # only the step off the back of the 15 inch motor is a discontinuity
    assert np.sum(jumps > 0.05) <= 1


def test_nested_clearances():
    lo, mid, tw = D.LOW, D.MID, D.TW
    assert lo.check() == []
    assert mid.check() == []
    # tweeter body, terminals included, sits in front of the mid throat and inside the mid mouth
    assert tw.depth_with_terminals < mid.tweeter_face <= mid.length
    # open annulus around the tweeter all the way to its face
    assert mid.radius(mid.tweeter_face) - mid.plug_radius >= 0.006
    # mid driver frame and horn mouth fit inside the can wall
    assert mid.can_diameter / 2 - max(mid.frame_diameter, mid.mouth_diameter) / 2 >= 0.005
    # tail and nose fairings do not touch
    assert lo.tail_start > lo.nose_end
    assert lo.cutout_radius > lo.body.cone_radius


def test_profiles_hit_their_end_radii():
    from sunflower.geometry import PROFILES, profile_radius
    for p in PROFILES:
        assert profile_radius(p, 0.05, 0.2, 0.3, 0.0) == pytest.approx(0.05)
        assert profile_radius(p, 0.05, 0.2, 0.3, 0.3) == pytest.approx(0.2, rel=1e-6)


def test_tweeter_models_match_their_datasheet_points():
    from sunflower import tweeters as TW
    for tid in TW.all_ids():
        t = json.loads((ROOT / f"data/tweeters/{tid}.json").read_text())["response_db"]
        lvl = spl(TW.response(tid, np.array(t["f"])))
        assert np.max(np.abs(lvl - np.array(t["0"]))) < 0.01


def test_selected_design_is_the_best_feasible_within_tolerance():
    sel = json.loads((ROOT / "results/optimise/selected.json").read_text())
    sysdf = pd.read_csv(ROOT / "results/optimise/system-search.csv")
    feas = sysdf[(sysdf.width <= 0.85) & (sysdf.depth <= 1.2)]
    assert sel["scores"]["score"] <= feas.score.min() + 0.25 + 1e-9
    assert sel["cabinet"]["width"] <= 0.85 and sel["cabinet"]["depth"] <= 1.2


def test_rear_chamber_volume_matches_design():
    cut = json.loads((ROOT / "cad/cutlist.json").read_text())
    assert abs(cut["rear_chamber"]["net_litres"] - D.LOW_REAR_LITRES) < 1.0


def test_cad_parts_match_design_dimensions():
    parts = {p["name"]: p for p in json.loads((ROOT / "cad/parts.json").read_text())}
    mouth = D.LOW.mouth_z * 1000
    assert parts["mid_horn"]["bbox_mm"][5] == pytest.approx(mouth, abs=0.5)
    assert parts["mid_horn"]["bbox_mm"][3] == pytest.approx(D.MID.can_diameter * 500, abs=0.5)
    assert parts["drv_tweeter_body"]["bbox_mm"][3] == pytest.approx(D.TW.body_diameter * 500, abs=0.5)
    assert parts["horn_wall_right"]["bbox_mm"][3] == pytest.approx(D.LOW.mouth_width * 500, abs=0.5)
    assert parts["baffle"]["bbox_mm"][5] == pytest.approx(D.BAFFLE * 1000, abs=0.5)
    ext = parts["side_right"]["bbox_mm"]
    assert ext[3] == pytest.approx(D.LOW.mouth_width * 500 + D.PLY * 1000, abs=0.5)


def test_dsp_settings_give_a_flat_sum():
    dsp = json.loads((ROOT / "results/planewave/dsp.json").read_text())
    assert dsp["rms_deviation_100_16k_db"] < 1.5
    assert min(dsp["delay_ms"].values()) == 0


def test_verified_drivers_have_sources():
    for p in (ROOT / "data/drivers/verified").glob("*.json"):
        r = json.loads(p.read_text())
        assert r["catalogue_status"] == "manufacturer_verified"
        assert r["parameter_source"].startswith("https://")
        Driver.from_record(r)


def test_committed_results_are_reproducible(tmp_path):
    from sunflower import simulate
    simulate.run(tmp_path)
    for name in ("low", "mid", "tweeter", "system"):
        new = pd.read_csv(tmp_path / f"{name}.csv")
        old = pd.read_csv(ROOT / f"results/planewave/{name}.csv")
        spl_cols = [c for c in old.columns if "spl" in c]
        assert np.allclose(new[spl_cols], old[spl_cols], atol=1e-3), name
