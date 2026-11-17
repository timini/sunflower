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
from sunflower.simulate import low_section, mid_section

ROOT = D.ROOT
IMAGE = "dolfinx/dolfinx:v0.8.0"
ANGLES = (0, 15, 30, 45, 60)


def case_tube():
    z = np.linspace(0, 0.3, 2)
    return dict(z=z.tolist(), r_outer=[0.05, 0.05], r_inner=[0.0, 0.0],
                freqs=np.geomspace(100, 1500, 25).tolist(), radius=1.0,
                h_horn=0.006, h_far=0.03, angles=list(ANGLES))


def case_mid():
    m = D.MID
    fmax = min(1.6 * D.CROSSOVER_MID_HIGH_HZ, 12000)
    zs = np.unique(np.r_[np.linspace(0, m.length, 120), m.s_exit, m.s_tip, m.s_tip + 1e-5])
    ro = [m.radius(z) for z in zs]
    ri = [m.plug(z) if z <= m.s_tip else 0.0 for z in zs]
    h = min(343 / fmax / 10, 0.006)
    return dict(z=zs.tolist(), r_outer=ro, r_inner=ri,
                freqs=np.geomspace(0.6 * D.CROSSOVER_LOW_MID_HZ, fmax, 70).tolist(),
                radius=max(0.6, 1.5 * m.mouth_diameter), h_horn=h, h_far=2 * h, angles=list(ANGLES))


def case_hf():
    """HF unit radiating from its printed horn mouth inside the mid horn (mid annulus behind it, rigid)."""
    m = D.MID
    st = m.s_tip
    back = np.linspace(st, 0, 40)
    poly = [(0.0, st), (m.hf_mouth_radius, st), (m.tip_radius, st)]
    poly += [(m.plug(s), s) for s in back[1:]]
    fwd = np.linspace(0, m.length, 120)
    poly += [(m.radius(s), s) for s in fwd]
    poly += [(0.0, m.length)]
    fmax = 16000
    h = 343 / fmax / 8
    return dict(z=[0.0, m.length], r_outer=[m.radius(0), m.radius(m.length)], r_inner=[0.0, 0.0],
                polygon=poly, inlet=(0.0, m.hf_mouth_radius, st),
                freqs=np.geomspace(0.5 * D.CROSSOVER_MID_HIGH_HZ, fmax, 40).tolist(),
                radius=0.4, h_horn=h, h_far=2.5 * h, angles=list(ANGLES))


def case_low():
    h = D.LOW
    zs = np.unique(np.r_[np.linspace(h.throat_z, h.mouth_z, 240), h.can_start, h.can_start + 1e-5])
    ro = [float(np.sqrt(h.outer_area(z) / np.pi)) for z in zs]
    ri = [h.inner_radius(z) for z in zs]
    return dict(z=zs.tolist(), r_outer=ro, r_inner=ri,
                freqs=np.geomspace(40, min(3 * D.CROSSOVER_LOW_MID_HZ, 2000), 60).tolist(), radius=2.5,
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
        cfg = D.config()
        rear = None if cfg.mid_driver.factory_sealed else m.rear_volume_litres
        f, tr = fem_transfer(raw, m.area(0.0))
        s = solve(cfg.mid_driver, tr, front_m3=m.front_chamber_cc * 1e-6, rear_litres=rear)
        pw = {th: mid_section(cfg, f, th)[0] for th in ANGLES}
    else:
        lo = D.LOW
        cfg = D.config()
        f, tr = fem_transfer(raw, lo.area(lo.throat_z))
        s = solve(cfg.low_driver, tr, front_m3=lo.front_chamber_m3, rear_litres=D.LOW_REAR_LITRES)
        pw = {th: low_section(cfg, f, th)[0] for th in ANGLES}
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


def hf_compare(raw, out):
    """FEM polar of the HF mouth inside the mid horn, relative to on axis, against the piston model."""
    from sunflower.acoustics import far_field
    m = D.MID
    f = raw.f.values
    cols = dict(frequency_hz=f)
    ref = np.abs(h(raw, 0))
    piston0 = np.abs(far_field(f, np.ones_like(f), m.hf_mouth_radius, 0.0))
    cols["onaxis_vs_piston_db"] = 20 * np.log10(ref / piston0)
    for th in ANGLES:
        cols[f"fem_rel_{th}deg_db"] = 20 * np.log10(np.abs(h(raw, th)) / ref)
        cols[f"piston_rel_{th}deg_db"] = 20 * np.log10(np.abs(far_field(f, np.ones_like(f), m.hf_mouth_radius, th))
                                                      / piston0)
    df = pd.DataFrame(cols)
    df.to_csv(out / "hf-in-mid.csv", index=False, float_format="%.6g")
    b = (f >= D.CROSSOVER_MID_HIGH_HZ) & (f <= 16000)
    return dict(band_hz=[D.CROSSOVER_MID_HIGH_HZ, 16000],
                onaxis_change_db_mean=float(df.onaxis_vs_piston_db[b].mean()),
                rel_30deg_fem_mean_db=float(df.fem_rel_30deg_db[b].mean()),
                rel_30deg_piston_mean_db=float(df.piston_rel_30deg_db[b].mean()))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--only", choices=("tube", "mid", "low", "hf"))
    ap.add_argument("--reuse", action="store_true", help="reuse raw FEM CSVs")
    args = ap.parse_args()
    out = ROOT / "results/fem"
    out.mkdir(parents=True, exist_ok=True)
    summary = json.loads((out / "summary.json").read_text()) if (out / "summary.json").exists() else {}
    for name, case in (("tube", case_tube()), ("mid", case_mid()), ("low", case_low()), ("hf", case_hf())):
        if args.only and name != args.only:
            continue
        raw_path = out / "work" / f"{name}-raw.csv"
        raw = pd.read_csv(raw_path) if args.reuse and raw_path.exists() else run_case(name, case, out)
        if name == "tube":
            summary["tube"] = compare_tube(raw, out)
        elif name == "hf":
            summary["hf"] = hf_compare(raw, out)
        else:
            df = couple(name, raw)
            df.to_csv(out / f"{name}.csv", index=False, float_format="%.6g")
            band = dict(mid=(D.CROSSOVER_LOW_MID_HZ, D.CROSSOVER_MID_HIGH_HZ), low=(100, D.CROSSOVER_LOW_MID_HZ))[name]
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
