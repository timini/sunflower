"""Screen 15 inch drivers and low horn sizes for the reversed low section.

For every flat-panel low horn on the grid, the plane-wave transfer is
computed once and then coupled to every 15 inch driver in the catalogue at
three sealed rear-chamber volumes. Ranking: reject mean SPL below 103 dB or
excursion beyond Xmax at 100 W in band, then minimum peak-to-peak variation
over 100 to 700 Hz.

The catalogue records are legacy and unverified. The winners are checked
by hand against manufacturer datasheets (see data/drivers/verified/).
"""
from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd

from sunflower.acoustics import Driver, far_field, horn_transfer, solve, spl
from sunflower.geometry import LowHorn

ROOT = Path(__file__).resolve().parents[2]
BAND = (100.0, 700.0)
GRID = dict(throat_width=(0.36, 0.40, 0.44), mouth_width=(0.56, 0.62, 0.68, 0.75),
            length=(0.75, 0.85, 0.95))
REAR_LITRES = (30.0, 50.0, 80.0)
POWER_W = 100.0


def load_catalogue(path=ROOT / "data/drivers/fifteen-inch-catalogue.json"):
    out = []
    for r in json.loads(Path(path).read_text())["drivers"]:
        try:
            out.append((r, Driver.from_record(r)))
        except (KeyError, TypeError, ZeroDivisionError):
            continue
    return out


def low_response(horn: LowHorn, driver: Driver, f, rear_litres, tr=None, theta=0.0, curved=True):
    if tr is None:
        tr = horn_transfer(f, horn.acoustic_area, horn.acoustic_length, 300, horn.breaks,
                           mouth_area=horn.outer_area(horn.mouth_z))
    s = solve(driver, tr, front_m3=horn.front_chamber_m3, rear_litres=rear_litres)
    a_out = np.sqrt(horn.outer_area(horn.mouth_z) / np.pi)
    p = far_field(f, s["u_mouth"], a_out, theta, inner_radius=horn.mid.can_diameter / 2,
                  wavefront_radius=horn.wavefront_radius if curved else None)
    return p, s


def run(out: Path):
    out.mkdir(parents=True, exist_ok=True)
    f = np.unique(np.r_[np.geomspace(40, 2000, 301), BAND])
    band = (f >= BAND[0]) & (f <= BAND[1])
    cat = load_catalogue()
    rows = []
    for wt, wm, length in itertools.product(*GRID.values()):
        horn = LowHorn(throat_width=wt, mouth_width=wm, length=length)
        if horn.check():
            continue
        tr = horn_transfer(f, horn.acoustic_area, horn.acoustic_length, 300, horn.breaks,
                           mouth_area=horn.outer_area(horn.mouth_z))
        for rec, d in cat:
            for vr in REAR_LITRES:
                p, s = low_response(horn, d, f, vr, tr)
                level = spl(p)[band]
                znom = rec["parameters"].get("nominal_impedance_ohm") or 8.0
                x100 = s["x_peak_m"][band].max() * np.sqrt(POWER_W * znom) / 2.83
                lo6 = f[np.argmax(spl(p) > level.mean() - 6)]
                rows.append(dict(driver_id=d.driver_id, label=d.label, throat_width_m=wt,
                                 mouth_width_m=wm, length_m=length, rear_litres=vr,
                                 ripple_db=float(np.ptp(level)), mean_spl_db=float(level.mean()),
                                 f_minus6_hz=float(lo6), x_peak_100w_mm=float(x100 * 1000),
                                 xmax_mm=float((d.xmax or np.nan) * 1000)))
        print(f"W {wt} -> {wm}, L {length}: {len(rows)} rows", flush=True)
    df = pd.DataFrame(rows)
    df["passes"] = (df.mean_spl_db >= 103) & (df.x_peak_100w_mm <= df.xmax_mm)
    df = df.sort_values(["passes", "ripple_db"], ascending=[False, True])
    df.to_csv(out / "search.csv", index=False)
    best = df[df.passes].drop_duplicates("driver_id").head(15)
    best.to_csv(out / "best-per-driver.csv", index=False)
    print(best.to_string(index=False))
    return df


def refine(out: Path, driver_id="18sound-15nmb420"):
    """Finer grid for the chosen driver, using its own body dimensions."""
    from sunflower import design as D
    d = D.low_driver()
    f = np.unique(np.r_[np.geomspace(40, 2000, 301), BAND])
    band = (f >= BAND[0]) & (f <= BAND[1])
    rows = []
    for wt, wm, length, vr in itertools.product((0.36, 0.38), (0.62, 0.66, 0.70, 0.75, 0.80),
                                                (0.75, 0.80, 0.85, 0.90, 1.0), (15, 20, 25, 30, 40)):
        horn = LowHorn(throat_width=wt, mouth_width=wm, length=length, body=D.LOW_BODY, mid=D.MID)
        if horn.check():
            continue
        p, s = low_response(horn, d, f, vr)
        level = spl(p)[band]
        rows.append(dict(throat_width_m=wt, mouth_width_m=wm, length_m=length, rear_litres=vr,
                         ripple_db=float(np.ptp(level)), mean_spl_db=float(level.mean()),
                         f_minus6_hz=float(f[np.argmax(spl(p) > level.mean() - 6)]),
                         x_peak_100w_mm=float(s["x_peak_m"][band].max() * np.sqrt(800) / 2.83 * 1000)))
    df = pd.DataFrame(rows).sort_values("ripple_db")
    df.to_csv(out / f"refine-{driver_id}.csv", index=False)
    print(df.head(12).to_string(index=False))
    return df


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--output", type=Path, default=ROOT / "results/screen-15")
    ap.add_argument("--refine", action="store_true")
    a = ap.parse_args()
    a.output.mkdir(parents=True, exist_ok=True)
    refine(a.output) if a.refine else run(a.output)
