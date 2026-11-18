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
    for s in np.linspace(0, m.s_tip - 1e-4, 300):
        assert m.radius(s) >= m.plug(s) - 1e-9
        assert m.area(s) == pytest.approx(math.pi * (m.radius(s) ** 2 - m.plug(s) ** 2))
        assert m.area(s) == pytest.approx(math.pi * m.eq_radius(s) ** 2, rel=0.02)


def test_bullet_is_one_smooth_taper_that_holds_the_hf_body_and_horn():
    m = D.MID
    ss = np.linspace(0, m.s_tip, 400)
    r = np.array([m.plug(s) for s in ss])
    assert np.all(np.diff(r) <= 1e-12)                      # no bulge, no straight section
    assert np.all(np.abs(np.diff(r, 2)) < 1e-4)             # smooth
    assert m.plug(m.s_exit) >= D.TW.body_diameter / 2
    hf = np.linspace(m.s_exit, m.s_tip, 50)
    assert min(m.plug(s) - m.hf_horn_radius(s) for s in hf) >= m.hf_wall - 1e-4


def test_kick_body_is_solved_from_the_law_and_holds_its_contents():
    k = D.KICK
    z = np.linspace(k.turn_z, k.mouth_z, 400)
    a = np.array([k.area(v) for v in z])
    law = np.array([k.law_area(v) for v in z])
    assert np.allclose(a, law, rtol=1e-6)
    rb = np.array([k.body_radius(v) for v in z])
    assert np.all(np.abs(np.diff(rb, 2)) < 2e-3)            # smooth body, no straight can
    assert k.check() == [] and D.MID.check() == []


def test_square_mouth_single_ply_frustum_and_fillers():
    k = D.KICK
    assert k.filler(k.mouth_z) == 0.0                       # fillers vanish: the mouth is fully square
    assert k.wall_area(k.mouth_z) == pytest.approx(k.mouth_width**2)
    c0 = k.filler(0.0)
    assert c0 == pytest.approx(k.throat_width / (2 + math.sqrt(2)))   # regular octagon at the throat
    assert k.splay_deg == pytest.approx(2 * k.half_angle_deg)
    assert k.outer_half_width(k.mouth_z) - k.half_width(k.mouth_z) == pytest.approx(
        k.ply / math.cos(math.radians(k.half_angle_deg)))


def test_shroud_clears_the_cone_by_its_travel():
    k = D.KICK
    r = np.linspace(k.channel_start, k.drv.rim_r, 100)
    assert np.all(k.gap(r) >= k.drv.xmech + 0.003 - 1e-9)
    assert np.all(k.shroud_z(r) > k.drv.cone_z(r))


def test_inside_out_rules():
    m, k, tw = D.MID, D.KICK, D.TW
    assert m.plug_radius == pytest.approx(tw.body_diameter / 2 + 0.002)
    assert 2 * m.drv.rim_r >= 1.6 * 2 * m.plug_radius
    assert m.throat_area == pytest.approx(m.drv.sd / m.compression)
    assert k.mouth_body_radius == pytest.approx(m.outer(k.mouth_z - k.mid_throat_z))
    assert k.motor_front > k.drv.top_plate_z + k.drv.motor_h


def test_depth_and_chamber():
    k = D.KICK
    assert 0.40 <= k.depth <= 0.75 + 1e-9   # owner chose 700 mm (depth comparison on the validation page)
    assert k.chamber_litres > 5


def test_network_matches_analytic_tube():
    from sunflower.acoustics import Network
    f = np.geomspace(50, 2000, 50)
    L, S = 0.3, np.pi * 0.05**2
    net = Network()
    a = net.node()
    net.source(a, S)
    path = net.path(a, [S] * 30, [L / 30] * 30)
    net.radiate(path[-1], 0.05)
    r = net.solve(f)
    k = 2 * np.pi * f / 343.0
    zc = 1.225 * 343.0 / S
    zl = r["load"]
    zin = zc * (zl + 1j * zc * np.tan(k * L)) / (zc + 1j * zl * np.tan(k * L))
    assert np.allclose(r["z_rad"], S**2 * zin, rtol=1e-9)


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
    assert sel["scores"]["system"] <= sysdf.system_score.min() + 0.25 + 1e-9   # network-model selection; f2 is revised after FEM (crossover_note)
    for k in (sel["tweeter"], sel["mid"]["driver_id"], sel["kick"]["driver_id"]):
        assert k not in excluded


def test_dsp_settings_give_a_flat_sum():
    dsp = json.loads((ROOT / "results/planewave/dsp.json").read_text())
    assert dsp["rms_deviation_100_16k_db"] < 2.5   # includes the sealed kick roll-off from 120 Hz down to 100 Hz
    assert min(dsp["delay_ms"].values()) == 0
    assert "no_sub_shelf" in dsp


def test_candidate_drivers_have_sources():
    for kind in ("kick", "mid"):
        for r in json.loads((ROOT / f"data/drivers/{kind}-candidates.json").read_text())["drivers"]:
            assert r["parameter_source"].startswith("http")
            Driver.from_record(r)


def test_committed_results_are_reproducible(tmp_path):
    from sunflower import simulate
    simulate.run(tmp_path)
    for name in ("low", "mid", "tweeter", "system"):
        new = pd.read_csv(tmp_path / f"{name}.csv")
        old = pd.read_csv(ROOT / f"results/planewave/{name}.csv")
        cols = [c for c in old.columns if "spl" in c]
        assert np.allclose(new[cols], old[cols], atol=1e-3), name
