"""FEM check of the mid and low horns (axisymmetric, dolfinx in Docker).

The low horn is octagonal to square in reality. The FEM model uses the
equal-area circular equivalent, so it checks the plane-wave assumption along
the horn and the mouth radiation, not the corner geometry. The FEM throat
impedance and far-field transfer are coupled to the same lumped driver
models as the plane-wave run, so the two results differ only in the acoustics.

Run:  uv run python -m sunflower.fem [--only tube|mid|low]
"""
from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd

from sunflower import design as D
from sunflower.acoustics import HornTransfer, far_field, group_delay, horn_transfer, solve, spl
from sunflower.screen_15 import low_response

ROOT = D.ROOT
IMAGE = "dolfinx/dolfinx:v0.8.0"
ANGLES = (0, 15, 30, 45, 60)


def case_tube():
    z = np.linspace(0, 0.3, 2)
    return dict(z=z.tolist(), r_outer=[0.05, 0.05], r_inner=[0.0, 0.0],
                freqs=np.geomspace(100, 1500, 25).tolist(), radius=1.0,
                h_horn=0.006, h_far=0.03, angles=list(ANGLES))


def case_mid():
    m, tw = D.MID, D.TWEETER
    zs = np.r_[np.linspace(0, tw.face_from_mid_throat, 30), tw.face_from_mid_throat,
               np.linspace(tw.face_from_mid_throat, m.length, 10)[1:]]
    ro = [m.radius(z) for z in zs]
    ri = [tw.housing_diameter / 2] * 30 + [0.0] + [0.0] * 9
    return dict(z=zs.tolist(), r_outer=ro, r_inner=ri,
                freqs=np.geomspace(300, 12000, 70).tolist(), radius=0.5,
                h_horn=0.003, h_far=0.007, angles=list(ANGLES))


def case_low():
    h = D.LOW
    zs = np.linspace(h.throat_z, h.mouth_z, 240)
    zs = np.unique(np.r_[zs, [h.throat_z + b for b in h.breaks]])
    ro = [float(np.sqrt(h.outer_area(z) / np.pi)) for z in zs]
    ri = [h.inner_radius(z) for z in zs]
    return dict(z=zs.tolist(), r_outer=ro, r_inner=ri,
                freqs=np.geomspace(40, 1500, 60).tolist(), radius=2.5,
                h_horn=0.02, h_far=0.05, angles=list(ANGLES))


def run_case(name, case, out):
    work = out / "work"
    work.mkdir(parents=True, exist_ok=True)
    (work / f"{name}.json").write_text(json.dumps(case))
    cmd = ["docker", "run", "--rm", "-v", f"{ROOT}:/w", "-w", "/w", IMAGE, "bash", "-c",
           f"source /usr/local/bin/dolfinx-complex-mode && python3 fem/axisym.py "
           f"results/fem/work/{name}.json results/fem/work/{name}-raw.csv"]
    subprocess.run(cmd, check=True)
    return pd.read_csv(work / f"{name}-raw.csv")


def fem_transfer(raw, throat_area):
    f = raw.f.values
    z_in = raw.z_in_re.values + 1j * raw.z_in_im.values
    u_ratio = raw.u_ratio_re.values + 1j * raw.u_ratio_im.values
    return f, HornTransfer(f, z_in, u_ratio, np.zeros_like(z_in), throat_area, np.nan)


def h(raw, th):
    return raw[f"h{th}_re"].values + 1j * raw[f"h{th}_im"].values


def compare_tube(raw, out):
    case = case_tube()
    f = raw.f.values
    tr = horn_transfer(f, lambda z: np.pi * 0.05**2, 0.3, 200)
    zf = raw.z_in_re.values + 1j * raw.z_in_im.values
    err = np.abs(zf - tr.z_in) / np.abs(tr.z_in)
    df = pd.DataFrame(dict(frequency_hz=f, fem_abs=np.abs(zf), tmm_abs=np.abs(tr.z_in), rel_error=err))
    df.to_csv(out / "tube-check.csv", index=False)
    return dict(max_relative_impedance_error=float(err.max()), median=float(np.median(err)))


def couple(name, raw):
    if name == "mid":
        m = D.MID
        f, tr = fem_transfer(raw, m.area(0.0))
        s = solve(D.mid_driver(), tr, front_m3=m.front_chamber_cc * 1e-6, rear_litres=None)
        pw_tr = horn_transfer(f, m.area, m.length, 600, breaks=(D.TWEETER.face_from_mid_throat,))
        spw = solve(D.mid_driver(), pw_tr, front_m3=m.front_chamber_cc * 1e-6, rear_litres=None)
        pw = {th: far_field(f, spw["u_mouth"], m.mouth_diameter / 2, th, wavefront_radius=m.wavefront_radius)
              for th in ANGLES}
    else:
        lo = D.LOW
        f, tr = fem_transfer(raw, lo.area(lo.throat_z))
        s = solve(D.low_driver(), tr, front_m3=lo.front_chamber_m3, rear_litres=D.LOW_REAR_LITRES)
        pw = {th: low_response(lo, D.low_driver(), f, D.LOW_REAR_LITRES, theta=th)[0] for th in ANGLES}
    cols = dict(frequency_hz=f)
    for th in ANGLES:
        p = h(raw, th) * s["u_throat"]
        cols[f"fem_{th}deg_spl_db"] = spl(p)
        cols[f"planewave_{th}deg_spl_db"] = spl(pw[th])
        if th == 0:
            cols["fem_0deg_phase_deg"] = np.degrees(np.angle(p))
            cols["planewave_0deg_phase_deg"] = np.degrees(np.angle(pw[0]))
            cols["fem_group_delay_ms"] = group_delay(f, p) * 1000
            cols["planewave_group_delay_ms"] = group_delay(f, pw[0]) * 1000
    return pd.DataFrame(cols)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--only", choices=("tube", "mid", "low"))
    ap.add_argument("--reuse", action="store_true", help="reuse raw FEM CSVs")
    args = ap.parse_args()
    out = ROOT / "results/fem"
    out.mkdir(parents=True, exist_ok=True)
    summary = json.loads((out / "summary.json").read_text()) if (out / "summary.json").exists() else {}
    for name, case in (("tube", case_tube()), ("mid", case_mid()), ("low", case_low())):
        if args.only and name != args.only:
            continue
        raw_path = out / "work" / f"{name}-raw.csv"
        raw = pd.read_csv(raw_path) if args.reuse and raw_path.exists() else run_case(name, case, out)
        if name == "tube":
            summary["tube"] = compare_tube(raw, out)
        else:
            df = couple(name, raw)
            df.to_csv(out / f"{name}.csv", index=False, float_format="%.6g")
            band = dict(mid=(500, 7000), low=(100, 500))[name]
            b = (df.frequency_hz >= band[0]) & (df.frequency_hz <= band[1])
            d = df.fem_0deg_spl_db[b] - df.planewave_0deg_spl_db[b]
            summary[name] = dict(band_hz=band, max_abs_difference_db=float(d.abs().max()),
                                 mean_difference_db=float(d.mean()),
                                 fem_ripple_db=float(np.ptp(df.fem_0deg_spl_db[b])),
                                 planewave_ripple_db=float(np.ptp(df.planewave_0deg_spl_db[b])))
        (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
        print(name, json.dumps(summary.get(name), indent=1))


if __name__ == "__main__":
    main()
