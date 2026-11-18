"""FEM check of the kick, mid and HF-in-mid acoustics (axisymmetric, dolfinx in Docker).

Each case is the real 2D (r, z) profile from ``sunflower.geometry``. The
cone is a vibrating boundary that moves along the axis (normal velocity
v n_z on the curved surface), not a piston at a plane. The solver is
``fem/axisym.py``: P2 Lagrange elements, a rigid baffle with a half-space
exterior, a first-order Bayliss-Turkel condition on the outer arc, and the
far field from a Kirchhoff-Helmholtz integral over a hemisphere with the
half-space Green's function, so a body may protrude past the baffle.

Cases
-----
tube  Validation: a 50 mm radius, 300 mm long tube driven by a flat piston,
      mouth in a baffle. Compared with a one-duct network and the baffled
      piston far field, which are close to exact here.
mid   Air between the mid cone and the bullet nose (constant gap), the
      throat annulus, the lead-in ring over the outer cone, the horn
      between radius(s) and plug(s), and the HF horn bore as a closed side
      cavity (the HF unit is rigid for this case). The surround is a ring at
      half the cone velocity. The felt under the nose is the same lossy
      layer as in the network (wavenumber k (1 - j felt_loss)). Radiation:
      the kick mouth plane is the baffle and the mid horn protrudes past it
      as a solid with its rolled lip, so the mid mouth is not baffled.
kick  Equal-area axisymmetric model of the square frustum: wall radius
      sqrt(wall_area(z) / pi), which includes the corner fillers. The
      reversed cone drives the air between it and the shroud. Inside
      channel_start the air between the cone and a flat roof at the shroud
      height is a short closed cavity (at least 3 mm deep), and the voice
      coil former (r = vc_r) closes it; the dust cap area vents into the
      motor, not the horn. The surround is a ring at half velocity, then
      the basket flange top and the baffle. The air turns under the shroud
      lip and runs forward between the centre body and the walls to the
      kick mouth, which opens into a half space (baffle at the kick mouth
      plane) with the mid horn protruding as a rigid body closed at its
      mouth.
hf    The HF unit's exit (flat, unit velocity) at s_exit, its horn bore to
      s_tip, inside the mid horn with the mid cone rigid. Same exterior as
      the mid case. A velocity source at the exit overstates the bore
      resonances that a real compression driver's source impedance damps.
kick3d  Optional check of the equal-area assumption (fem/kick3d.py): the real
      square frustum with the corner filler tetrahedra, quarter symmetry,
      the same solid of revolution for the driver, centre body and mid
      horn, P2 tetrahedra sized for 1 kHz. On-axis only. Uses the kick
      frequencies (every other one) so the results line up.

Outputs (results/fem/)
----------------------
kick-vs-network.csv, mid-vs-network.csv  (one row per frequency)
    frequency_hz
    fem_<a>deg_spl_db, network_<a>deg_spl_db, diff_<a>deg_db
        SPL at 1 m, 2.83 V, half space, after coupling each acoustic model to
        the same Thiele-Small driver with acoustics.couple (rear volume as in
        simulate). diff = FEM - network. Kick: 0 deg only. Mid: 0, 15, 30,
        45, 60 deg.
    fem_zrad_re, fem_zrad_im, network_zrad_re, network_zrad_im
        mechanical radiation impedance on the cone, N s/m, unit velocity
    zrad_diff_db       20 log10(|Z_fem| / |Z_network|)
    fem_u_mouth_abs, network_u_mouth_abs
        mouth volume velocity per unit cone velocity, m^3/s per m/s (FEM:
        flux through the mouth plane)
    fem_onaxis_per_velocity_db
        20 log10 |p(1 m, 0 deg)| for unit cone velocity, before coupling
    fem_impedance_ohm, network_impedance_ohm, fem_excursion_mm,
    network_excursion_mm   electrical impedance and peak excursion, 2.83 V
    mid only, FEM variants that isolate each difference from the network
    (fem_<v>_0deg_spl_db, fem_<v>_30deg_spl_db, fem_<v>_zrad_re/im):
        flush               baffle at the mid mouth plane, no lip (the
                            network's radiation assumption)
        flush_nohf          as flush, bullet closed at s_tip (no HF bore):
                            the closest FEM match to the network
        nofelt              as the main case without the felt loss
        flush_nohf_nofelt   flush_nohf without the felt loss
        network_nofelt_0deg_spl_db is the network without the felt loss, the
        reference for the two nofelt variants.
hf-in-mid.csv  (read by simulate.tweeter_section)
    frequency_hz
    fem_rel_<a>deg_db     FEM far field at angle a relative to on axis
    onaxis_vs_piston_db   FEM on-axis level relative to a baffled flat piston
                          of the HF horn mouth radius with the same volume
                          velocity as the HF exit
    piston_rel_<a>deg_db  that piston at angle a relative to its on axis
kick-3d-vs-axisym.csv
    frequency_hz, fem3d_0deg_spl_db, fem2d_0deg_spl_db, network_0deg_spl_db
    (coupled, 2.83 V, half space), diff_3d_2d_db, diff_3d_network_db,
    fem3d_zrad_re/im, fem2d_zrad_re/im, zrad_diff_3d_2d_db
mid-felt-sweep.csv
    frequency_hz (2 to 7 kHz), fem_eta<x>_0deg_spl_db, network_eta<x>_0deg_spl_db:
    coupled on-axis SPL with the felt loss factor x under the bullet nose
    (x = 0 is lossless). summary.json mid.felt_sweep gives the null
    frequency and depth for each.
tube-check.csv  frequency_hz, fem/network |Z_rad|, relative error, far-field error
summary.json   per case: agreement with the network per band (max, RMS and mean
               of FEM - network in dB, SPL and |Z_rad|), mesh sizes, the
               convergence check (h halved at 3 frequencies), the energy
               balance (power through the far-field arc against Re(Z_rad)/2,
               lossless runs only), runtime. "design" records the drivers
               and the geometry clearances the meshes had to resolve. "low"
               repeats the kick headline number for site.py.
work/          case JSON and raw FEM output (per unit velocity), logs

Run:  uv run python -m sunflower.fem [--only tube|mid|kick|hf|kick3d] [--reuse] [--procs N]
          [--no-convergence] [--no-3d]
Everything takes about 11 minutes on 11 cores and 8 GB (the 3D kick is about 9
of them; it runs two processes because each needs about 2 GB).
"""
from __future__ import annotations

import argparse
import json
import math
import subprocess
import time
from dataclasses import replace

import numpy as np
import pandas as pd

from sunflower import design as D
from sunflower.acoustics import C, Network, couple, far_field, spl
from sunflower.models import kick_network, mid_network

ROOT = D.ROOT
IMAGE = "dolfinx/dolfinx:v0.8.0"
ANGLES = (0, 15, 30, 45, 60)
AXIS, WALL, BAFFLE, FAR = 1, 2, 3, 4
CONE, SURROUND, HF_EXIT = 10, 11, 12


# ---------------------------------------------------------------- outlines
class Outline:
    """Closed (r, z) polygon. Each point carries a target element size and the tag of the edge leaving it."""

    def __init__(self):
        self.pts = []

    def seg(self, rz, tag, h):
        """Add a polyline; its last point starts the next segment, so it is not added here."""
        rz = np.asarray(rz, float)
        hs = np.broadcast_to(np.asarray(h, float), (len(rz),))
        for (r, z), hh in zip(rz[:-1], hs[:-1]):
            self.pts.append([float(r), float(z), float(hh), int(tag)])

    def case(self):
        out = []
        for p in self.pts:
            if out and math.hypot(p[0] - out[-1][0], p[1] - out[-1][1]) < 2e-6:
                out[-1][3] = p[3]
                out[-1][2] = min(out[-1][2], p[2])
                continue
            out.append(list(p))
        if math.hypot(out[0][0] - out[-1][0], out[0][1] - out[-1][1]) < 2e-6:
            out.pop()
        return dict(points=[p[:3] for p in out], tags=[p[3] for p in out])


def self_intersections(case):
    """Pairs of non-adjacent outline edges that cross (the arc is checked as its chord)."""
    P = np.array([p[:2] for p in case["points"]])
    A, B = P, np.roll(P, -1, axis=0)
    n = len(P)
    hits = []
    d = B - A
    for i in range(n):
        j = np.arange(i + 2, n)
        if i == 0:
            j = j[j != n - 1]
        if not len(j):
            continue
        p, r = A[i], d[i]
        q, s = A[j], d[j]
        rxs = r[0] * s[:, 1] - r[1] * s[:, 0]
        qp = q - p
        ok = np.abs(rxs) > 1e-14
        t = np.where(ok, (qp[:, 0] * s[:, 1] - qp[:, 1] * s[:, 0]) / np.where(ok, rxs, 1), -1)
        u = np.where(ok, (qp[:, 0] * r[1] - qp[:, 1] * r[0]) / np.where(ok, rxs, 1), -1)
        bad = ok & (t > 1e-9) & (t < 1 - 1e-9) & (u > 1e-9) & (u < 1 - 1e-9)
        hits += [(i, int(k), P[i].tolist()) for k in j[bad]]
    return hits


def surround_width(drv):
    return max(2 * (math.sqrt(drv.sd / math.pi) - drv.rim_r), 0.004)


def lip(m, s0=0.0, h=0.004, n=48):
    """Rolled lip at the mid mouth: an arc of radius lip_radius tangent to the inner wall at the mouth, curling
    outwards and back until it meets the outside of the wall. Returns points from the mouth edge to that junction
    (in z = s + s0) and the junction s."""
    L, Rm, a = m.length, m.radius(m.length), math.radians(m.half_angle_deg)
    lr = m.lip_radius
    cr, cs = Rm + lr * math.cos(a), L - lr * math.sin(a)
    phis = np.linspace(a, 2 * math.pi + a, 2000)
    pr, ps = cr - lr * np.cos(phis), cs + lr * np.sin(phis)
    g = pr - np.array([m.outer(min(s, L)) for s in ps])
    k = np.argmax((phis > math.pi) & (g <= 0))
    # refine the junction
    lo, hi = phis[k - 1], phis[k]
    for _ in range(40):
        mid = 0.5 * (lo + hi)
        s = cs + lr * math.sin(mid)
        if cr - lr * math.cos(mid) - m.outer(min(s, L)) > 0:
            lo = mid
        else:
            hi = mid
    ph = np.linspace(a, hi, n)
    pts = np.column_stack([cr - lr * np.cos(ph), cs + lr * np.sin(ph) + s0])
    return pts, float(cs + lr * math.sin(hi))


def mid_outline(m, *, s_b, R, hf_drive=False, h_wall, h_far, h_gap_min=0.0007, h_throat=0.003, h_hf=0.004,
                flush=False, hf_cavity=True, slots=None):
    """Mid and HF-in-mid air domain in s coordinates (z = s). Baffle plane at s = s_b.

    flush: baffle at the mid mouth plane instead (s_b is ignored, no lip), the network's assumption.
    hf_cavity: False closes the bullet flat at s_tip (no HF horn bore).
    slots: list of (r_in, r_out) annular slots. The phase plug then fills the cone dish at the gap everywhere
    and the air reaches the throat plane (s = 0) only through the slots; rings between slots are islands."""
    d = m.drv
    o = Outline()
    g = m.gap
    sw = surround_width(d)

    def ngap_h(r):
        slope = np.abs(np.gradient(m.s_cone(r), r))
        return np.clip(g / np.sqrt(1 + slope**2) / 3, h_gap_min, h_wall)

    rp, ro = m.plug_r0, m.outer_throat_radius
    rc = np.unique(np.r_[np.linspace(0, d.vc_r, 30), np.linspace(d.vc_r, d.rim_r, 80), rp, ro])
    hc = ngap_h(rc)
    hc = np.where((rc > rp + 0.004) & (rc < ro - 0.004), np.minimum(h_throat, h_wall), hc)
    o.seg(np.column_stack([rc, m.s_cone(rc)]), WALL if hf_drive else CONE, hc)
    s_rim = float(m.s_cone(d.rim_r))
    o.seg([(d.rim_r, s_rim), (d.rim_r + sw, s_rim)], WALL if hf_drive else SURROUND, h_gap_min * 2)
    o.seg([(d.rim_r + sw, s_rim), (d.rim_r + sw, s_rim + g), (d.rim_r, s_rim + g)], WALL, h_gap_min * 2)
    if slots:
        bl = slots[-1][1]
        rc = np.unique(np.r_[rc, [x for sl in slots for x in sl]])
        rr = rc[(rc >= bl) & (rc <= d.rim_r)][::-1]
        o.seg(np.column_stack([rr, m.s_cone(rr) + g]), WALL, ngap_h(rr))
        o.seg([(bl, float(m.s_cone(bl) + g)), (bl, 0.0)], WALL, [ngap_h(np.array([bl, bl + 1e-4]))[0], h_throat])
        o.seg([(bl, 0.0), (ro, 0.0)], WALL, h_throat)
    else:
        rr = rc[(rc >= ro) & (rc <= d.rim_r)][::-1]
        o.seg(np.column_stack([rr, m.s_cone(rr) + g]), WALL, ngap_h(rr))
        s_ring = float(m.s_cone(ro) + g)
        o.seg([(ro, s_ring), (ro, 0.0)], WALL, [ngap_h(np.array([ro, ro + 1e-4]))[0], h_throat])
    ss = np.linspace(0, m.length, 120)
    o.seg(np.column_stack([[m.radius(s) for s in ss], ss]), WALL,
          np.minimum(h_wall, np.linspace(h_throat, h_wall, len(ss))))
    if flush:
        s_b = m.length
        o.seg([(m.radius(s_b), s_b), (R, s_b)], BAFFLE, [h_wall, h_far])
    else:
        lp, s_j = lip(m)
        o.seg(lp, WALL, min(h_wall, 0.004))
        so = np.linspace(s_j, s_b, 40)
        o.seg(np.column_stack([[m.outer(s) for s in so], so]), WALL, h_wall)
        o.seg([(m.outer(s_b), s_b), (R, s_b)], BAFFLE, [h_wall, h_far])
    o.seg([(R, s_b), (0.0, s_b + R)], FAR, h_far)
    if hf_cavity:
        o.seg([(0.0, s_b + R), (0.0, m.s_exit)], AXIS, [h_far, h_hf])
        re = m.hf_exit_diameter / 2
        o.seg([(0.0, m.s_exit), (re, m.s_exit)], HF_EXIT if hf_drive else WALL, h_hf)
        sh = np.linspace(m.s_exit, m.s_tip, 40)
        o.seg(np.column_stack([[m.hf_horn_radius(s) for s in sh], sh]), WALL, h_hf)
        o.seg([(m.hf_mouth_radius, m.s_tip), (m.tip_radius, m.s_tip)], WALL, min(h_hf, 0.0015))
    else:
        o.seg([(0.0, s_b + R), (0.0, m.s_tip)], AXIS, [h_far, h_hf])
        o.seg([(0.0, m.s_tip), (m.tip_radius, m.s_tip)], WALL, min(h_hf, 0.0015))
    sp = np.linspace(m.s_tip, 0, 100)
    o.seg(np.column_stack([[m.plug(s) for s in sp], sp]), WALL, np.linspace(h_hf, h_throat, len(sp)))
    if slots:
        a1 = slots[0][0]
        o.seg([(rp, 0.0), (a1, 0.0)], WALL, h_throat)
        o.seg([(a1, 0.0), (a1, float(m.nose(a1)))], WALL, [h_throat, ngap_h(np.array([a1 - 1e-4, a1]))[-1]])
        rn = rc[rc <= a1][::-1]
    else:
        s_nose = float(m.nose(rp))
        o.seg([(rp, 0.0), (rp, s_nose)], WALL, [h_throat, ngap_h(np.array([rp - 1e-4, rp]))[-1]])
        rn = rc[rc <= rp][::-1]
    o.seg(np.column_stack([rn, m.nose(rn)]), WALL, ngap_h(rn))
    o.seg([(0.0, float(m.nose(0.0))), (0.0, float(m.s_cone(0.0)))], AXIS, h_gap_min * 2)
    out = o.case()
    if slots and len(slots) > 1:
        holes = []
        for (_, b), (a, _) in zip(slots[:-1], slots[1:]):
            rr = np.linspace(b, a, 12)
            ring = [(b, 0.0), (a, 0.0)] + [(x, float(m.nose(x))) for x in rr[::-1]]
            holes.append([[float(r), float(z), float(min(h_throat, g / 3))] for r, z in ring])
        out["holes"] = holes
    return out


def mid_exterior_reach(m, s_b):
    lp, s_j = lip(m)
    return float(np.max(np.hypot(lp[:, 0], lp[:, 1] - s_b)))


def case_mid(scale=1.0, freqs=None, hf=False, flush=False, hf_cavity=True, felt=True, slots=None):
    """felt: True uses MidHorn.felt_loss, False none, a number sets the loss factor."""
    m, k = D.MID, D.KICK
    eta = 0.0 if hf or felt is False else (m.felt_loss if felt is True else float(felt))
    if k.mid_mouth_z - k.mouth_z < 1e-6:
        flush = True                    # mid mouth in the kick mouth plane: that plane is the baffle
    s_b = m.length if flush else k.mouth_z - k.mid_throat_z
    if hf:
        fmax = 16000.0
        h_wall, h_far, R = C / fmax / 8, C / fmax / 6, 0.55
        fr = np.geomspace(0.5 * D.CROSSOVER_MID_HIGH_HZ, fmax, 40)
    else:
        fmax = 8000.0
        h_wall, h_far, R = C / fmax / 8, C / fmax / 6, 0.75
        fr = np.geomspace(150, fmax, 90)
    reach = m.radius(m.length) if flush else mid_exterior_reach(m, s_b)
    Rk = min(0.5 * (reach + R), R - 4 * h_far)
    case = mid_outline(m, s_b=s_b, R=R, hf_drive=hf, h_wall=h_wall, h_far=h_far,
                       h_hf=min(h_wall, 0.004), flush=flush, hf_cavity=hf_cavity, slots=slots)
    drive = {HF_EXIT: 1.0} if hf else {CONE: 1.0, SURROUND: 0.5}
    case.update(drive=drive, far=dict(zb=s_b, R=R), kh=dict(R=Rk, n=1200), h_max=h_far, scale=scale,
                flux=[0.0, m.radius(m.length - 0.005) - 1e-5, m.length - 0.005],
                freqs=list(map(float, fr if freqs is None else freqs)), angles=list(ANGLES),
                felt=dict(eta=eta, r_max=m.plug_r0, z_max=0.0) if eta > 0 else None,
                meta=dict(baffle_s=s_b, baffle="mid mouth plane" if flush else "kick mouth plane",
                          protrusion_m=m.length - s_b, exterior_reach_m=reach, kh_radius_m=Rk, fmax=fmax,
                          hf_cavity=hf_cavity, felt_loss=eta))
    return case


def kick_outline(k, *, R, h_far, h_wall=0.015):
    d, m = k.drv, k.mid
    o = Outline()
    sw = surround_width(d)
    rw = lambda z: math.sqrt(k.wall_area(z) / math.pi)  # noqa: E731
    rs = k.channel_start
    z_roof = float(k.shroud_z(rs))

    def roof(r):
        return np.maximum(z_roof, d.cone_z(r) + 0.003)

    rcav = np.linspace(d.vc_r, rs, 40)
    rch = np.linspace(rs, d.rim_r, 60)[1:]
    rc = np.r_[rcav, rch]
    hcav = np.clip((roof(rcav) - d.cone_z(rcav)) / 3, 0.001, 0.010)
    hch = np.minimum(k.gap(rch) / 4, h_wall)
    o.seg(np.column_stack([rc, d.cone_z(rc)]), CONE, np.r_[hcav, hch])
    under = 0.010
    o.seg([(d.rim_r, d.rim_z), (d.rim_r + sw, d.rim_z)], SURROUND, [hch[-1], under])
    fr = d.frame_od / 2
    o.seg([(d.rim_r + sw, d.rim_z), (d.rim_r + sw, d.flange_t), (fr, d.flange_t), (fr, 0.0), (rw(0.0), 0.0)],
          WALL, under)
    zz = np.linspace(0.0, k.mouth_z, 90)
    gapw = np.array([rw(z) - (k.body_radius(z) if z >= k.turn_z else 0.0) for z in zz])
    hw = np.minimum(np.where(zz >= k.turn_z, gapw / 4, under), h_wall)
    o.seg(np.column_stack([[rw(z) for z in zz], zz]), WALL, hw)
    o.seg([(rw(k.mouth_z), k.mouth_z), (R, k.mouth_z)], BAFFLE, [h_wall, h_far])
    o.seg([(R, k.mouth_z), (0.0, k.mouth_z + R)], FAR, h_far)
    s0 = k.mid_throat_z
    Lz = k.mid_mouth_z
    o.seg([(0.0, k.mouth_z + R), (0.0, Lz)], AXIS, [h_far, h_wall])
    if Lz - k.mouth_z < 1e-6:
        # mid mouth flush in the kick mouth plane: the (rigid, closed) mid horn face lies in the baffle plane
        o.seg([(0.0, Lz), (k.body_radius(k.mouth_z), Lz)], WALL, h_wall)
    else:
        o.seg([(0.0, Lz), (m.radius(m.length), Lz)], WALL, h_wall)
        lp, s_j = lip(m, s0=s0)
        o.seg(lp, WALL, 0.008)
        so = np.linspace(s_j, k.mouth_z - s0, 30)
        o.seg(np.column_stack([[m.outer(s) for s in so], so + s0]), WALL, h_wall)
    zb_ = np.linspace(k.mouth_z, k.turn_z, 90)
    gapb = np.array([rw(z) - k.body_radius(z) for z in zb_])
    o.seg(np.column_stack([[k.body_radius(z) for z in zb_], zb_]), WALL, np.minimum(gapb / 4, h_wall))
    r_lip = k.body_radius(k.turn_z)
    pts = [(r_lip, k.turn_z), (d.rim_r, k.turn_z)]
    if k.turn_z > float(k.shroud_z(d.rim_r)) + 1e-6:
        pts.append((d.rim_r, float(k.shroud_z(d.rim_r))))
    o.seg(pts + [(d.rim_r, float(k.shroud_z(d.rim_r)))], WALL,
          [min((rw(k.turn_z) - r_lip) / 4, h_wall)] + [hch[-1]] * len(pts))
    rsh = rch[::-1]
    o.seg(np.column_stack([np.r_[rsh, rs], k.shroud_z(np.r_[rsh, rs])]), WALL, np.r_[hch[::-1], hcav[-1]])
    rr = rcav[::-1]
    o.seg(np.column_stack([rr, roof(rr)]), WALL, hcav[::-1])
    o.seg([(d.vc_r, float(roof(d.vc_r))), (d.vc_r, float(d.cone_z(d.vc_r)))], WALL, 0.001)
    return o.case()


def case_kick(scale=1.0, freqs=None):
    k = D.KICK
    fmax = 1000.0
    h_far = C / fmax / 6
    R = 1.5
    reach = mid_exterior_reach(k.mid, k.mouth_z - k.mid_throat_z)
    case = kick_outline(k, R=R, h_far=h_far)
    case.update(drive={CONE: 1.0, SURROUND: 0.5}, far=dict(zb=k.mouth_z, R=R), kh=dict(R=1.1, n=600),
                h_max=h_far, scale=scale,
                flux=[k.body_radius(k.mouth_z - 0.005) + 1e-5,
                      math.sqrt(k.wall_area(k.mouth_z - 0.005) / math.pi) - 1e-5,
                      k.mouth_z - 0.005],
                freqs=list(map(float, np.geomspace(40, fmax, 80) if freqs is None else freqs)), angles=list(ANGLES),
                meta=dict(exterior_reach_m=reach, kh_radius_m=1.1, fmax=fmax))
    return case


def case_kick3d(scale=1.0, freqs=None, fmax=600.0):
    """3D quarter-symmetry kick: the real frustum and corner fillers (fem/kick3d.py)."""
    k = D.KICK
    d, m = k.drv, k.mid
    sw = surround_width(d)
    fr = d.frame_od / 2
    rs = k.channel_start
    z_roof = float(k.shroud_z(rs))
    rr = np.linspace(d.vc_r, rs, 400)
    above = rr[d.cone_z(rr) + 0.003 > z_roof]
    solid = [dict(kind="line", pts=[(0.0, 0.0), (fr, 0.0), (fr, d.flange_t), (d.rim_r + sw, d.flange_t),
                                    (d.rim_r + sw, d.rim_z), (d.rim_r, d.rim_z)])]
    rc = np.linspace(d.rim_r, d.vc_r, 40)
    solid.append(dict(kind="spline", pts=[(float(r), float(d.cone_z(r))) for r in rc]))
    if len(above):
        rx = float(above.max())
        ra = np.linspace(d.vc_r, rx, 20)
        zn = float(d.cone_z(d.vc_r))
        solid.append(dict(kind="line", pts=[(d.vc_r, zn), (d.vc_r, zn + 0.003)]))
        solid.append(dict(kind="spline", pts=[(float(r), float(d.cone_z(r)) + 0.003) for r in ra]))
        solid.append(dict(kind="line", pts=[(rx, float(d.cone_z(rx)) + 0.003), (rx, z_roof), (rs, z_roof)]))
    else:
        solid.append(dict(kind="line", pts=[(d.vc_r, float(d.cone_z(d.vc_r))), (d.vc_r, z_roof), (rs, z_roof)]))
    rsh = np.linspace(rs, d.rim_r, 40)
    solid.append(dict(kind="spline", pts=[(float(r), float(k.shroud_z(r))) for r in rsh]))
    r_lip = k.body_radius(k.turn_z)
    lip_pts = [(d.rim_r, float(k.shroud_z(d.rim_r)))]
    if k.turn_z > float(k.shroud_z(d.rim_r)) + 1e-6:
        lip_pts.append((d.rim_r, k.turn_z))
    solid.append(dict(kind="line", pts=lip_pts + [(r_lip, k.turn_z)]))
    zb_ = np.linspace(k.turn_z, k.mouth_z, 60)
    solid.append(dict(kind="spline", pts=[(k.body_radius(z), float(z)) for z in zb_]))
    s0 = k.mid_throat_z
    lp, s_j = lip(m, s0=s0)
    so = np.linspace(k.mouth_z - s0, s_j, 30)
    solid.append(dict(kind="spline", pts=[(m.outer(s), float(s + s0)) for s in so]))
    solid.append(dict(kind="spline", pts=[tuple(map(float, p)) for p in lp[::-1]]))
    solid.append(dict(kind="line", pts=[(m.radius(m.length), k.mid_mouth_z), (0.0, k.mid_mouth_z)]))
    h_far = C / fmax / 6
    hw0, hwm = k.half_width(0.0), k.half_width(k.mouth_z)
    R = 1.0
    boxes = [dict(h=0.0015, box=[0, 0.065, 0, 0.065, z_roof - 0.05, z_roof + 0.02], thickness=0.01),
             dict(h=0.008, box=[0, hw0 + 0.01, 0, hw0 + 0.01, -0.01, k.turn_z + 0.06], thickness=0.03),
             dict(h=min(0.02, h_far), box=[0, hwm + 0.02, 0, hwm + 0.02, -0.01, k.mid_mouth_z + m.lip_radius + 0.05],
                  thickness=0.1)]
    case = dict(mouth_z=k.mouth_z, R=R, kh_R=0.75, hw0=hw0, hwm=hwm, c0=k.filler(0.0), solid=solid,
                cone=[(float(r), float(d.cone_z(r))) for r in np.linspace(d.vc_r, d.rim_r, 400)],
                surround=[(d.rim_r, d.rim_z), (d.rim_r + sw, d.rim_z)], boxes=boxes, h_far=h_far, scale=scale,
                freqs=list(map(float, np.geomspace(40, fmax, 24) if freqs is None else freqs)),
                meta=dict(fmax=fmax, R=R, kh_R=0.75, h_far_m=h_far))
    return case


def case_tube(scale=1.0, freqs=None):
    a, L, R = 0.05, 0.3, 0.75
    o = Outline()
    o.seg([(0, 0), (a, 0)], CONE, 0.004)
    o.seg([(a, 0), (a, L)], WALL, 0.004)
    o.seg([(a, L), (R, L)], BAFFLE, [0.004, 0.03])
    o.seg([(R, L), (0, L + R)], FAR, 0.03)
    o.seg([(0, L + R), (0, 0)], AXIS, [0.03, 0.004])
    case = o.case()
    case.update(drive={CONE: 1.0}, far=dict(zb=L, R=R), kh=dict(R=0.5, n=600), h_max=0.03, scale=scale,
                flux=[0.0, a - 1e-5, L - 0.005],
                freqs=list(map(float, np.geomspace(100, 1500, 25) if freqs is None else freqs)), angles=list(ANGLES),
                meta=dict(a=a, L=L))
    return case


# ---------------------------------------------------------------- running
def run_case(name, case, procs, script="fem/axisym.py"):
    work = D.RESULTS / "fem/work"
    rel = work.relative_to(ROOT).as_posix()
    work.mkdir(parents=True, exist_ok=True)
    (work / f"{name}.json").write_text(json.dumps(case))
    procs = max(1, min(procs, len(case["freqs"])))
    for old in work.glob(f"{name}-raw-*.csv*"):
        old.unlink()
    loop = " ".join(
        f"python3 {script} {rel}/{name}.json {rel}/{name}-raw-{i}.csv {i} {procs} "
        f"> {rel}/{name}-{i}.log 2>&1 &" for i in range(procs))
    cmd = ["docker", "run", "--rm", "-v", f"{ROOT}:/w", "-w", "/w", "-e", "OMP_NUM_THREADS=1", IMAGE, "bash", "-c",
           f"source /usr/local/bin/dolfinx-complex-mode >/dev/null; {loop} wait"]
    t0 = time.time()
    subprocess.run(cmd, check=True)
    wall = time.time() - t0
    parts, metas = [], []
    for i in range(procs):
        p = work / f"{name}-raw-{i}.csv"
        if not p.exists():
            raise RuntimeError(f"{name} chunk {i} failed; see {work / f'{name}-{i}.log'}:\n"
                               + (work / f"{name}-{i}.log").read_text()[-3000:])
        parts.append(pd.read_csv(p))
        metas.append(json.loads((work / f"{name}-raw-{i}.csv.meta.json").read_text()))
        p.unlink()
        (work / f"{name}-raw-{i}.csv.meta.json").unlink()
        (work / f"{name}-{i}.log").unlink()
    raw = pd.concat(parts).sort_values("f").reset_index(drop=True)
    raw.to_csv(work / f"{name}-raw.csv", index=False, float_format="%.10g")
    meta = dict(metas[0], wall_s=wall, procs=procs, n_freqs=len(raw))
    meta.pop("n_freqs", None)
    meta["n_freqs"] = len(raw)
    (work / f"{name}-raw.meta.json").write_text(json.dumps(meta, indent=1))
    return raw, meta


def cplx(raw, a, b=None):
    if b is None:
        return raw[f"{a}_re"].values + 1j * raw[f"{a}_im"].values
    return raw[a].values + 1j * raw[b].values


def fem_net(raw, load):
    return dict(f=raw.f.values, z_rad=cplx(raw, "zrad"), u_mouth=cplx(raw, "umouth"), load=load)


def stats(d, f, band):
    b = (f >= band[0]) & (f <= band[1])
    x = np.asarray(d)[b]
    return dict(band_hz=list(band), max_abs_db=float(np.abs(x).max()), rms_db=float(np.sqrt(np.mean(x**2))),
                mean_db=float(x.mean()))


# ---------------------------------------------------------------- comparisons
def compare_tube(raw):
    case = case_tube()
    a, L = case["meta"]["a"], case["meta"]["L"]
    f = raw.f.values
    S = math.pi * a * a
    net = Network()
    n0, n1 = net.node(), net.node()
    net.source(n0, S)
    net.duct(n0, n1, S, L)
    net.radiate(n1, a)
    r = net.solve(f)
    zf = cplx(raw, "zrad")
    err = np.abs(zf - r["z_rad"]) / np.abs(r["z_rad"])
    hn = far_field(f, r["u_mouth"], a, 0.0)
    hf = cplx(raw, "h0")
    ff = 20 * np.log10(np.abs(hf) / np.abs(hn))
    ph = np.degrees(np.angle(hf / hn))
    df = pd.DataFrame(dict(frequency_hz=f, fem_zrad_abs=np.abs(zf), network_zrad_abs=np.abs(r["z_rad"]),
                           rel_error=err, farfield_0deg_diff_db=ff, farfield_0deg_phase_diff_deg=ph,
                           u_mouth_ratio=np.abs(cplx(raw, "umouth") / r["u_mouth"])))
    df.to_csv(D.RESULTS / "fem/tube-check.csv", index=False, float_format="%.6g")
    return dict(max_relative_impedance_error=float(err.max()), median=float(np.median(err)),
                farfield_max_abs_db=float(np.abs(ff).max()), farfield_max_phase_deg=float(np.abs(ph).max()))


def compare(name, raw, variants=None):
    """variants: {label: (raw, felt)} extra FEM runs of the mid; each gets fem_<label>_0deg_spl_db and
    fem_<label>_zrad_re/im, compared with the network with the same felt setting."""
    cfg = replace(D.config(), free_space=False)
    f = raw.f.values
    if name == "kick":
        k = D.KICK
        drv, rear, angles = cfg.kick_driver, k.chamber_litres, (0,)
        net = kick_network(k)
        rn = net.solve(f)

        def net_p(s, th):
            return far_field(f, s["u_mouth"], net.mouth_radius, th, inner_radius=net.mouth_inner)
    else:
        m = D.MID
        drv, rear, angles = cfg.mid_driver, m.rear_volume_litres, ANGLES
        net = mid_network(m)
        rn = net.solve(f)

        def net_p(s, th):
            return far_field(f, s["u_mouth"], m.mouth_diameter / 2, th, wavefront_radius=m.wavefront_radius)
    sn = couple(drv, rn, rear_litres=rear)
    sf = couple(drv, fem_net(raw, rn["load"]), rear_litres=rear)
    cols = dict(frequency_hz=f)
    for th in angles:
        pf = cplx(raw, f"h{th}") * sf["vel"]
        pn = net_p(sn, th)
        cols[f"fem_{th}deg_spl_db"] = spl(pf)
        cols[f"network_{th}deg_spl_db"] = spl(pn)
        cols[f"diff_{th}deg_db"] = spl(pf) - spl(pn)
    zf = cplx(raw, "zrad")
    cols.update(fem_zrad_re=zf.real, fem_zrad_im=zf.imag, network_zrad_re=rn["z_rad"].real,
                network_zrad_im=rn["z_rad"].imag, zrad_diff_db=20 * np.log10(np.abs(zf) / np.abs(rn["z_rad"])),
                fem_u_mouth_abs=np.abs(cplx(raw, "umouth")), network_u_mouth_abs=np.abs(rn["u_mouth"]),
                fem_onaxis_per_velocity_db=spl(cplx(raw, "h0")),
                fem_impedance_ohm=np.abs(sf["z_e"]), network_impedance_ohm=np.abs(sn["z_e"]),
                fem_excursion_mm=sf["x_peak_m"] * 1000, network_excursion_mm=sn["x_peak_m"] * 1000)
    variants = variants or {}
    if any(not felt for _, felt in variants.values()):
        rn0 = mid_network(D.MID, felt=0.0).solve(f)
        sn0 = couple(drv, rn0, rear_litres=rear)
        cols["network_nofelt_0deg_spl_db"] = spl(net_p(sn0, 0))
    for lab, (rv, felt) in variants.items():
        rr = rn if felt else rn0
        sv = couple(drv, fem_net(rv, rr["load"]), rear_litres=rear)
        cols[f"fem_{lab}_0deg_spl_db"] = spl(cplx(rv, "h0") * sv["vel"])
        cols[f"fem_{lab}_30deg_spl_db"] = spl(cplx(rv, "h30") * sv["vel"])
        z = cplx(rv, "zrad")
        cols[f"fem_{lab}_zrad_re"], cols[f"fem_{lab}_zrad_im"] = z.real, z.imag
    df = pd.DataFrame(cols)
    df.to_csv(D.RESULTS / f"fem/{name}-vs-network.csv", index=False, float_format="%.6g")
    if name == "kick":
        bands = dict(passband=(40.0, D.CROSSOVER_LOW_MID_HZ), full=(40.0, 1000.0))
    else:
        bands = dict(passband=(D.CROSSOVER_LOW_MID_HZ, D.CROSSOVER_MID_HIGH_HZ), full=(150.0, 8000.0))
    out = {}
    for bn, band in bands.items():
        e = dict(spl_0deg=stats(df.diff_0deg_db, f, band), zrad_magnitude=stats(df.zrad_diff_db, f, band))
        for th in angles[1:]:
            e[f"spl_{th}deg"] = stats(df[f"diff_{th}deg_db"], f, band)
        for lab, (rv, felt) in variants.items():
            ref = df.network_0deg_spl_db if felt else df.network_nofelt_0deg_spl_db
            zr = rn["z_rad"] if felt else rn0["z_rad"]
            v = dict(spl_0deg=stats(df[f"fem_{lab}_0deg_spl_db"] - ref, f, band),
                     zrad_magnitude=stats(20 * np.log10(np.abs(cplx(rv, "zrad")) / np.abs(zr)), f, band))
            if felt:
                v["spl_30deg"] = stats(df[f"fem_{lab}_30deg_spl_db"] - df.network_30deg_spl_db, f, band)
            e[f"variant_{lab}"] = v
        out[bn] = e
    p = out["passband"]["spl_0deg"]
    out.update(max_abs_difference_db=p["max_abs_db"], mean_difference_db=p["mean_db"], rms_difference_db=p["rms_db"])
    return out


def compare_3d(raw3, raw2):
    """3D quarter-symmetry kick against the equal-area axisymmetric FEM and the network, coupled to the driver."""
    cfg = replace(D.config(), free_space=False)
    k = D.KICK
    f = raw3.f.values
    r2 = raw2.set_index("f").loc[f].reset_index()
    net = kick_network(k)
    rn = net.solve(f)
    drv, rear = cfg.kick_driver, k.chamber_litres
    s3 = couple(drv, fem_net(raw3.assign(umouth_re=0.0, umouth_im=0.0), rn["load"]), rear_litres=rear)
    s2 = couple(drv, fem_net(r2, rn["load"]), rear_litres=rear)
    sn = couple(drv, rn, rear_litres=rear)
    p3 = spl(cplx(raw3, "h0") * s3["vel"])
    p2 = spl(cplx(r2, "h0") * s2["vel"])
    pn = spl(far_field(f, sn["u_mouth"], net.mouth_radius, 0.0, inner_radius=net.mouth_inner))
    z3, z2 = cplx(raw3, "zrad"), cplx(r2, "zrad")
    df = pd.DataFrame(dict(frequency_hz=f, fem3d_0deg_spl_db=p3, fem2d_0deg_spl_db=p2, network_0deg_spl_db=pn,
                           diff_3d_2d_db=p3 - p2, diff_3d_network_db=p3 - pn,
                           fem3d_zrad_re=z3.real, fem3d_zrad_im=z3.imag, fem2d_zrad_re=z2.real, fem2d_zrad_im=z2.imag,
                           zrad_diff_3d_2d_db=20 * np.log10(np.abs(z3) / np.abs(z2))))
    df.to_csv(D.RESULTS / "fem/kick-3d-vs-axisym.csv", index=False, float_format="%.6g")
    out = {}
    for bn, band in dict(passband=(40.0, D.CROSSOVER_LOW_MID_HZ), full=(40.0, 1000.0)).items():
        out[bn] = dict(spl_3d_vs_2d=stats(df.diff_3d_2d_db, f, band),
                       spl_3d_vs_network=stats(df.diff_3d_network_db, f, band),
                       zrad_3d_vs_2d=stats(df.zrad_diff_3d_2d_db, f, band))
    return out


FELT_ETAS = (0.0, 0.05, 0.1, 0.2, 0.3, 0.5, 0.8)


def null_depth(f, y, search=(3200.0, 4300.0), ref=(2500.0, 6000.0)):
    """Deepest point in the search band and its depth below the median level of the reference band."""
    s = (f >= search[0]) & (f <= search[1])
    i = np.flatnonzero(s)[np.argmin(y[s])]
    r = (f >= ref[0]) & (f <= ref[1])
    return float(f[i]), float(np.median(y[r]) - y[i])


def felt_sweep(get, procs):
    """The centre cavity under the bullet nose, lossless and with the felt layer at several loss factors.

    FEM: the main mid geometry with a lossy fluid region under the nose (k (1 - j eta), rho c kept), on a
    dense grid from 2 to 7 kHz. Network: mid_network(m, felt=eta). Both coupled to the driver, on axis."""
    m = D.MID
    cfg = replace(D.config(), free_space=False)
    f = np.geomspace(2000.0, 7000.0, 120)
    cols, rows = dict(frequency_hz=f), []
    for eta in FELT_ETAS:
        lab = f"felt{eta:g}".replace(".", "p")
        raw, _ = get(f"mid-{lab}", case_mid(freqs=f, felt=eta if eta > 0 else False), procs)
        rn = mid_network(m, felt=eta).solve(f)
        sn = couple(cfg.mid_driver, rn, rear_litres=m.rear_volume_litres)
        sf = couple(cfg.mid_driver, fem_net(raw, rn["load"]), rear_litres=m.rear_volume_litres)
        pf = spl(cplx(raw, "h0") * sf["vel"])
        pn = spl(far_field(f, sn["u_mouth"], m.mouth_diameter / 2, 0.0, wavefront_radius=m.wavefront_radius))
        cols[f"fem_eta{eta:g}_0deg_spl_db"] = pf
        cols[f"network_eta{eta:g}_0deg_spl_db"] = pn
        ff, df_ = null_depth(f, pf)
        fn, dn = null_depth(f, pn)
        rows.append(dict(eta=eta, fem_null_hz=ff, fem_null_depth_db=df_, network_null_hz=fn, network_null_depth_db=dn))
    pd.DataFrame(cols).to_csv(D.RESULTS / "fem/mid-felt-sweep.csv", index=False, float_format="%.6g")
    return dict(note="centre-cavity dip: deepest on-axis point 3.2 to 4.3 kHz, depth below the 2.5 to 6 kHz "
                     "median, coupled SPL",
                rows=rows)


def hf_compare(raw):
    m = D.MID
    f = raw.f.values
    s_exit = math.pi * (m.hf_exit_diameter / 2) ** 2
    cols = dict(frequency_hz=f)
    h0 = np.abs(cplx(raw, "h0"))
    piston0 = np.abs(far_field(f, np.full_like(f, s_exit), m.hf_mouth_radius, 0.0))
    cols["onaxis_vs_piston_db"] = 20 * np.log10(h0 / piston0)
    for th in ANGLES:
        cols[f"fem_rel_{th}deg_db"] = 20 * np.log10(np.abs(cplx(raw, f"h{th}")) / h0)
        cols[f"piston_rel_{th}deg_db"] = 20 * np.log10(
            np.abs(far_field(f, np.ones_like(f), m.hf_mouth_radius, th))
            / np.abs(far_field(f, np.ones_like(f), m.hf_mouth_radius, 0.0)))
    df = pd.DataFrame(cols)
    df.to_csv(D.RESULTS / "fem/hf-in-mid.csv", index=False, float_format="%.6g")
    b = (f >= D.CROSSOVER_MID_HIGH_HZ) & (f <= 16000)
    return dict(band_hz=[D.CROSSOVER_MID_HIGH_HZ, 16000],
                onaxis_change_db_mean=float(df.onaxis_vs_piston_db[b].mean()),
                rel_30deg_fem_mean_db=float(df.fem_rel_30deg_db[b].mean()),
                rel_30deg_piston_mean_db=float(df.piston_rel_30deg_db[b].mean()))


def convergence(name, base, fine, couple_drv=None, angles=(0, 30)):
    """Compare a run at half the element size with the base run at the same frequencies."""
    out = []
    for _, row in fine.iterrows():
        b = base.iloc[int(np.argmin(np.abs(base.f.values - row.f)))]
        zb_, zf_ = complex(b.zrad_re, b.zrad_im), complex(row.zrad_re, row.zrad_im)
        hb, hf_ = complex(b.h0_re, b.h0_im), complex(row.h0_re, row.h0_im)
        e = dict(frequency_hz=float(row.f), zrad_change_pct=100 * abs(zf_ - zb_) / abs(zb_),
                 onaxis_change_db=20 * math.log10(abs(hf_) / abs(hb)))
        if 30 in angles:
            h30b, h30f = complex(b.h30_re, b.h30_im), complex(row.h30_re, row.h30_im)
            e["h30_change_db"] = 20 * math.log10(abs(h30f) / abs(h30b))
        if couple_drv is not None:
            drv, rear = couple_drv
            fa = np.array([row.f])
            one = dict(f=fa, u_mouth=np.array([0j]), load=np.ones(1))
            sb = couple(drv, dict(one, z_rad=np.array([zb_])), rear_litres=rear)
            sf = couple(drv, dict(one, z_rad=np.array([zf_])), rear_litres=rear)
            e["coupled_spl_change_db"] = float(spl(hf_ * sf["vel"])[0] - spl(hb * sb["vel"])[0])
        out.append(e)
    return out


def power_check(raw):
    """Largest |P_arc - Re(Z)/2| / (Re(Z)/2): energy conservation, valid only without the felt loss."""
    pin = raw.zrad_re.values / 2
    return float(np.max(np.abs(raw.power_arc.values - pin) / np.abs(pin)))


def geometry_notes():
    m, k = D.MID, D.KICK
    d, kd = m.drv, k.drv
    r = np.linspace(1e-3, d.rim_r, 600)
    ng = m.gap / np.sqrt(1 + np.gradient(m.s_cone(r), r) ** 2)
    rw = lambda z: math.sqrt(k.wall_area(z) / math.pi)  # noqa: E731
    zz = np.linspace(k.turn_z, k.mouth_z, 300)
    rc = np.linspace(k.channel_start, kd.rim_r, 100)
    rr = np.linspace(kd.vc_r, k.channel_start, 200)
    roof = float(k.shroud_z(k.channel_start))
    above = rr[kd.cone_z(rr) > roof - 0.003]
    return dict(
        mid_gap_vertical_mm=1000 * m.gap, mid_gap_normal_min_mm=1000 * float(ng.min()),
        mid_gap_normal_min_at_r_mm=1000 * float(r[np.argmin(ng)]), mid_xmech_mm=1000 * d.xmech,
        mid_surround_width_mm=1000 * surround_width(d),
        kick_channel_gap_mm=[1000 * float(k.gap(rc).min()), 1000 * float(k.gap(rc).max())],
        kick_passage_min_mm=1000 * float(min(rw(z) - k.body_radius(z) for z in zz)),
        kick_lip_height_over_flange_mm=1000 * (k.turn_z - kd.flange_t),
        kick_cavity_litres_fem=float(np.trapezoid(2 * np.pi * rr * (np.maximum(roof, kd.cone_z(rr) + 0.003)
                                                                     - kd.cone_z(rr)), rr) * 1000),
        kick_trapped_litres_network=k.trapped_litres,
        kick_cone_above_roof_inside_r_mm=1000 * float(above.max()) if len(above) else None,
        kick_dust_cap_area_not_driven_m2=math.pi * kd.vc_r**2, kick_sd_m2=kd.sd)


def pick(freqs, targets):
    fa = np.asarray(freqs)
    return [float(fa[np.argmin(np.abs(fa - t))]) for t in targets]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--only", choices=("tube", "mid", "kick", "hf", "kick3d"))
    ap.add_argument("--no-3d", action="store_true", help="skip the 3D kick (about 3 minutes)")
    ap.add_argument("--reuse", action="store_true", help="reuse raw FEM CSVs in results/fem/work")
    ap.add_argument("--procs", type=int, default=8)
    ap.add_argument("--no-convergence", action="store_true")
    args = ap.parse_args()
    out = D.RESULTS / "fem"
    work = out / "work"
    out.mkdir(parents=True, exist_ok=True)
    sp = out / "summary.json"
    summary = json.loads(sp.read_text()) if sp.exists() else {}
    cfg = D.config()
    summary["design"] = dict(kick_driver=cfg.kick_driver.driver_id, mid_driver=cfg.mid_driver.driver_id,
                             tweeter=cfg.tweeter, crossover_hz=[cfg.f1, cfg.f2], geometry=geometry_notes())
    builders = dict(tube=case_tube, mid=case_mid, kick=case_kick, hf=lambda **kw: case_mid(hf=True, **kw))
    conv_f = dict(mid=(500, 2000, 8000), kick=(60, 250, 1000), hf=(4000, 8000, 16000))
    conv_drv = dict(mid=(cfg.mid_driver, D.MID.rear_volume_litres), kick=(cfg.kick_driver, D.KICK.chamber_litres))
    t_all = time.time()

    def get(name, case, procs, script="fem/axisym.py"):
        rp = work / f"{name}-raw.csv"
        mp = work / f"{name}-raw.meta.json"
        if args.reuse and rp.exists():
            return pd.read_csv(rp), json.loads(mp.read_text()) if mp.exists() else {}
        return run_case(name, case, procs, script)

    for name in ("tube", "mid", "kick", "hf"):
        if args.only and name != args.only:
            continue
        t0 = time.time()
        case = builders[name]()
        hits = self_intersections(case)
        if hits:
            raise RuntimeError(f"{name}: outline crosses itself at {hits[:5]}")
        raw, meta = get(name, case, args.procs)
        mesh = dict(cells=meta.get("cells"), p2_dofs=meta.get("dofs"), h_min_mm=1000 * meta.get("h_min_m", np.nan),
                    h_max_mm=1000 * meta.get("h_max_m", np.nan), outline_points=len(case["points"]))
        entry = dict(power_balance_max_rel_error=power_check(raw) if name != "mid" else None, mesh=mesh,
                     case=case.get("meta", {}), n_freqs=len(raw))
        if name == "tube":
            entry.update(compare_tube(raw))
        elif name == "hf":
            entry.update(hf_compare(raw))
        elif name == "kick":
            entry.update(compare("kick", raw))
        else:
            variants = {}
            for lab, kw, felt in (("flush", dict(flush=True), True),
                                  ("flush_nohf", dict(flush=True, hf_cavity=False), True),
                                  ("nofelt", dict(felt=False), False),
                                  ("flush_nohf_nofelt", dict(flush=True, hf_cavity=False, felt=False), False)):
                rv, mv = get(f"mid-{lab}", case_mid(**kw), args.procs)
                variants[lab] = (rv, felt)
                if not felt:
                    entry[f"power_balance_max_rel_error_{lab}"] = power_check(rv)
            entry.update(compare("mid", raw, variants=variants))
            entry["felt_sweep"] = felt_sweep(get, args.procs)
        if name in conv_f and not args.no_convergence:
            fc = pick(case["freqs"], conv_f[name])
            fine_case = builders[name](scale=0.5, freqs=fc)
            fine, fmeta = get(f"{name}-fine", fine_case, 1 if name == "hf" else 3)
            entry["convergence"] = dict(scale=0.5, fine_cells=fmeta.get("cells"), fine_p2_dofs=fmeta.get("dofs"),
                                        points=convergence(name, raw, fine, conv_drv.get(name)))
        entry["runtime_s"] = round(time.time() - t0, 1)
        summary[name] = entry
        if name == "kick":
            summary["low"] = dict(note="alias of kick for site.py",
                                  max_abs_difference_db=entry["max_abs_difference_db"])
        sp.write_text(json.dumps(summary, indent=2) + "\n")
        print(name, json.dumps({k: v for k, v in entry.items() if k not in ("case",)}, indent=1))
    if (args.only in (None, "kick3d")) and not args.no_3d:
        t0 = time.time()
        kp = work / "kick-raw.csv"
        k2 = pd.read_csv(kp) if kp.exists() else get("kick", case_kick(), args.procs)[0]
        case = case_kick3d(freqs=k2.f.values[::2], fmax=1000.0)
        raw3, meta3 = get("kick3d", case, 2, "fem/kick3d.py")
        entry = compare_3d(raw3, k2)
        fine3, fmeta3 = get("kick3d-fine", case_kick3d(scale=0.75, freqs=pick(case["freqs"], (100, 1000)), fmax=1000.0),
                            1, "fem/kick3d.py")
        entry.update(mesh=dict(cells=meta3.get("cells"), p2_dofs=meta3.get("dofs"),
                               driven_area_m2=meta3.get("driven_area_m2")),
                     power_balance_max_rel_error=power_check(raw3), case=case["meta"],
                     convergence=dict(scale=0.75, fine_cells=fmeta3.get("cells"), fine_p2_dofs=fmeta3.get("dofs"),
                                      points=convergence("kick3d", raw3, fine3, conv_drv["kick"], angles=(0,))),
                     runtime_s=round(time.time() - t0, 1))
        summary["kick3d"] = entry
        sp.write_text(json.dumps(summary, indent=2) + "\n")
        print("kick3d", json.dumps({k: v for k, v in entry.items() if k != "case"}, indent=1))

    if not args.only:
        key = "total_runtime_s_with_reuse" if args.reuse else "total_runtime_s"
        summary.pop("total_runtime_s_with_reuse", None)
        summary[key] = round(time.time() - t_all, 1)
        sp.write_text(json.dumps(summary, indent=2) + "\n")


if __name__ == "__main__":
    main()
