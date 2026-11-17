import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from sunflower import design as D
from sunflower.acoustics import Driver, far_field, horn_transfer, solve, spl
from sunflower.geometry import LAWS, profile_radius

ROOT = Path(__file__).resolve().parents[1]


def test_plane_wave_model_matches_horn_simulation_branch_study():
    """Same geometry and inputs as horn-simulation codex/6p5-mid-500-6500 (the v0.1 mid geometry)."""
    from sunflower.geometry_v01 import MidHorn as MidHornV01
    ref = pd.read_csv(ROOT / "tests/reference/branch-study-beyma-6mcf200nd.csv")
    f = ref.frequency_hz.values
    mid = MidHornV01(front_chamber_cc=20.0)
    tr = horn_transfer(f, mid.area, mid.length, 600, breaks=(0.1,))
    beyma = Driver.from_record(json.loads((ROOT / "data/drivers/verified/beyma-6mcf200nd.json").read_text()))
    s = solve(beyma, tr, front_m3=20e-6, rear_litres=None)
    level = spl(far_field(f, s["u_mouth"], mid.mouth_diameter / 2))
    band = (f >= 500) & (f <= 6500)
    assert np.max(np.abs(level[band] - ref.spl_db.values[band])) < 0.5


def test_straight_tube_matches_analytic_input_impedance():
    f = np.geomspace(50, 2000, 200)
    L, S = 0.3, np.pi * 0.05**2
    tr = horn_transfer(f, lambda z: S, L, 50)
    k = 2 * np.pi * f / 343.0
    zc = 1.225 * 343.0 / S
    exact = zc * (tr.load + 1j * zc * np.tan(k * L)) / (zc + 1j * tr.load * np.tan(k * L))
    assert np.max(np.abs(tr.z_in - exact) / np.abs(exact)) < 1e-9


def test_profiles_hit_their_end_radii():
    for p in LAWS:
        assert profile_radius(p, 0.05, 0.2, 0.3, 0.0) == pytest.approx(0.05)
        assert profile_radius(p, 0.05, 0.2, 0.3, 0.3) == pytest.approx(0.2, rel=1e-6)


def test_mid_plug_is_always_subtracted_and_open_area_follows_the_law():
    m = D.MID
    for s in np.linspace(0, m.length, 300):
        assert m.radius(s) >= m.plug(s) - 1e-9
        assert m.area(s) == pytest.approx(math.pi * (m.radius(s) ** 2 - m.plug(s) ** 2))
        if s < m.s_tip - 1e-6:
            assert m.area(s) == pytest.approx(math.pi * m.eq_radius(s) ** 2, rel=0.02)


def test_outer_wall_never_narrows_and_mid_area_steps_only_at_the_hf_mouth():
    m = D.MID
    ss = np.linspace(0, m.length, 2000)
    ro = np.array([m.radius(s) for s in ss])
    assert np.all(np.diff(ro) >= -1e-9)
    a = np.array([m.area(s) for s in ss])
    jumps = np.abs(np.diff(a)) / a[:-1]
    big = ss[1:][jumps > 0.05]
    assert all(abs(z - m.s_tip) < 0.002 for z in big)
    assert math.pi * m.tip_radius**2 / (math.pi * m.radius(m.s_tip) ** 2) <= 0.10


def test_low_open_area_is_continuous_and_body_is_one_piece():
    lo = D.LOW
    z = np.linspace(lo.throat_z, lo.mouth_z, 3000)
    a = np.array([lo.area(v) for v in z])
    assert np.all(a > 0)
    assert np.max(np.abs(np.diff(a)) / a[:-1]) < 0.05
    ri = np.array([lo.inner_radius(v) for v in z])
    assert np.all(ri > 0)
    assert lo.inner_radius(lo.can_start) == pytest.approx(lo.can_radius)


def test_inside_out_rules():
    m, lo, tw = D.MID, D.LOW, D.TW
    assert m.plug_radius == pytest.approx(tw.body_diameter / 2 + 0.002)
    assert m.cone_diameter >= 1.6 * 2 * m.plug_radius
    assert m.throat_area == pytest.approx(m.sd / m.compression)
    assert m.can_diameter / 2 - max(m.frame_diameter, 2 * m.radius(m.length)) / 2 >= 0.005
    assert lo.can_start > lo.motor_end + 0.02
    assert lo.check() == [] and m.check() == []
    assert 1.3 <= lo.compression <= 4.0


def test_depth_splay_and_chamber():
    cut = json.loads((ROOT / "cad/cutlist.json").read_text())
    assert 400 <= cut["external_mm"]["depth"] <= 600
    assert D.LOW.splay_deg >= 10
    assert cut["chamber"]["available_litres"] >= D.LOW_REAR_LITRES - 0.5


def test_printable_parts_fit_the_bed():
    import struct
    for p in json.loads((ROOT / "cad/parts.json").read_text()):
        for fn in p.get("print_files", []):
            b = (ROOT / "cad/print" / fn).read_bytes()
            n = struct.unpack("<I", b[80:84])[0]
            v = np.frombuffer(b[84:84 + n * 50], dtype=np.dtype([("n", "<3f4"), ("v", "<9f4"), ("a", "<u2")]))["v"]
            xyz = v.reshape(-1, 3)
            size = np.sort(xyz.max(0) - xyz.min(0))
            assert size[0] <= 252 and size[1] <= 252, (fn, size)


def test_tweeter_models_match_their_datasheet_points():
    from sunflower import tweeters as TW
    for tid in TW.all_ids(include_excluded=True):
        t = json.loads((ROOT / f"data/tweeters/{tid}.json").read_text())["response_db"]
        lvl = spl(TW.response(tid, np.array(t["f"])))
        assert np.max(np.abs(lvl - np.array(t["0"]))) < 0.01


def test_hf_units_have_body_cylinder_and_source():
    from sunflower import tweeters as TW
    for tid in TW.all_ids():
        t = TW.load(tid)
        assert 0.04 < t.body_diameter < 0.13
        assert t.source.startswith("https://")


def test_selected_design_follows_the_rule():
    sel = json.loads((ROOT / "results/optimise/selected.json").read_text())
    sysdf = pd.read_csv(ROOT / "results/optimise/system-search.csv")
    excluded = json.loads((ROOT / "data/drivers/excluded.json").read_text())["drivers"]
    feas = sysdf[(sysdf.depth <= 0.6) & ~sysdf.low_driver.isin(excluded)]
    assert sel["scores"]["score"] <= feas.score.min() + 0.25 + 1e-9
    assert sel["tweeter"] not in excluded


def test_dsp_settings_give_a_flat_sum():
    dsp = json.loads((ROOT / "results/planewave/dsp.json").read_text())
    assert dsp["rms_deviation_100_16k_db"] < 2.0
    assert min(dsp["delay_ms"].values()) == 0
    assert "no_sub_shelf" in dsp


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
        cols = [c for c in old.columns if "spl" in c]
        assert np.allclose(new[cols], old[cols], atol=1e-3), name
