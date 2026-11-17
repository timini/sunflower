"""Build every Sunflower part as STEP and STL (millimetres) from sunflower.design.

Coordinates follow sunflower.geometry, in millimetres: z = 0 is the front face
of the baffle (where the reversed kick flange sits), z is positive towards the
listener, x is to the right when facing the cabinet, y is up.

Outputs:
  cad/step/<part>.step, cad/stl/<part>.stl   one file per part, in place
  cad/print/<part>-<n>[-s<k>].stl             printable pieces (z slab n, sector k), each fits a 250 mm bed
  cad/sunflower-assembly.step                 everything in place
  cad/parts.json                              part list: material, category, explode vector, bbox, print files
  cad/cutlist.json                            panels, compound mitre angles, fillers, chamber, stations
  cad/stack.json                              splay for a vertical stack of three (renderer)
  cad/array.json, cad/array.stl               3 x 3 curved array: placement of each box, joint gaps
  cad/dxf/<panel>.dxf                         flat patterns
  results/cad/area-check.csv                  open area sliced from the CAD air against the model

Driver bodies are dimensioned proxies for renders and clearance checks.
"""
from __future__ import annotations

import json
import math
import os
import struct
import sys
import time

import gmsh
import numpy as np

from sunflower import design as D
from sunflower.geometry import array_layout

ROOT = D.ROOT
OUT = ROOT / "cad"
RES = ROOT / "results" / "cad"
TMP = OUT / ".tmp"
MM = 1000.0
BED = 250.0          # printer bed, mm
SLAB = 240.0         # tallest printed z slab, mm

K, M, TW = D.KICK, D.MID, D.TW
KD, MD = K.drv, M.drv
T = K.ply * MM
A = math.radians(K.half_angle_deg)

# ---------------------------------------------------------------- stations (mm)
Z_BAF = -K.baffle * MM                             # baffle back face
Z_WB = -(K.baffle + K.chamber_depth) * MM          # wall back ends, back panel front face
Z_BACK = Z_WB - T                                  # back panel back face
Z_M = K.mouth_z * MM                               # kick mouth
Z_TURN = K.turn_z * MM
Z_MF = K.mid_flange_z * MM
Z_MT = K.mid_throat_z * MM
Z_MM = K.mid_mouth_z * MM
Z_MOTOR = K.motor_front * MM
WALL_B = K.body_wall * MM


def hw(z):
    return K.half_width(z / MM) * MM


def ohw(z):
    return K.outer_half_width(z / MM) * MM


def zs(s):
    """Mid horn coordinate (m) to absolute z (mm)."""
    return Z_MT + s * MM


# centre body radii (mm)
R_CS = K.channel_start * MM                        # inner edge of the shroud
R_MC = (KD.motor_r + K.motor_clear) * MM           # motor cavity
R_POD = MD.frame_od / 2 * MM + 0.5                 # mid pod cavity (frame plus 0.5 mm)
J1 = max(KD.top_plate_z * MM, float(K.shroud_z(K.channel_start)) * MM + 10)   # shroud / fairing joint
Z_CUT = KD.frame_od / 2 * MM - 17                  # baffle cutout radius (as in Kick.chamber_litres)

# supports
STANDOFF_R = 10.0
STANDOFF_AZ = [math.radians(45 + 90 * k) for k in range(4)]
DUCT = 0                                           # index of the hollow standoff (cable duct)
R_STANDOFF = KD.frame_od / 2 * MM + 12             # just outside the kick flange
SOCKET = 25.0                                      # standoff depth into the shroud
VANE_T, VANE_CHORD = 6.0, 60.0
Z_VANE = Z_M - 100.0
BV_T, BV_S0, BV_S1 = 4.0, 0.008, 0.040             # bullet vanes: thickness mm, s range m
BV_AZ = [math.radians(45 + 120 * k) for k in range(3)]
S_HF_CABLE = 0.5 * (BV_S0 + BV_S1)
R_HF_CH = R_POD + 20                               # HF cable channel radius in the body
Z_CAB = Z_MOTOR + 30                               # duct channel enters the pod here
SPIGOT_H = 6.0

occ = None
parts: list[dict] = []
warnings: list[str] = []


# ---------------------------------------------------------------- gmsh helpers
def start(name):
    global occ
    gmsh.clear()
    gmsh.model.add(name)
    occ = gmsh.model.occ


def _dedupe(pts, tol=1e-3):
    out = []
    for r, z in pts:
        p = (max(float(r), 0.0), float(z))
        if not out or math.hypot(p[0] - out[-1][0], p[1] - out[-1][1]) > tol:
            out.append(p)
    return out


def revolve(segs, angle=2 * math.pi):
    """Revolve a closed (r, z) profile about the z axis.

    ``segs`` is a list of (kind, points), kind "line" or "spline". Segments are
    joined end to start (with a straight line if they do not meet) and the
    profile is closed back to its first point.
    """
    pieces = []
    for kind, pts in segs:
        pts = _dedupe(pts)
        if pieces:
            last = pieces[-1][1][-1]
            if math.hypot(pts[0][0] - last[0], pts[0][1] - last[1]) > 1e-3:
                pieces.append(("line", [last, pts[0]]))
            else:
                pts[0] = last
        pieces.append((kind, pts))
    first, last = pieces[0][1][0], pieces[-1][1][-1]
    if math.hypot(first[0] - last[0], first[1] - last[1]) > 1e-3:
        pieces.append(("line", [last, first]))
    else:
        pieces[-1][1][-1] = first
    tags = {}

    def P(p):
        if p not in tags:
            tags[p] = occ.addPoint(p[0], 0, p[1])
        return tags[p]

    curves = []
    for kind, pts in pieces:
        ids = [P(p) for p in pts]
        if kind == "spline" and len(ids) > 2:
            curves.append(occ.addSpline(ids))
        else:
            curves += [occ.addLine(ids[k], ids[k + 1]) for k in range(len(ids) - 1) if ids[k] != ids[k + 1]]
    surf = occ.addPlaneSurface([occ.addCurveLoop(curves)])
    out = occ.revolve([(2, surf)], 0, 0, 0, 0, 0, 1, angle)
    return [e for e in out if e[0] == 3]


def poly_volume(faces_pts):
    cache, edges = {}, {}

    def P(p):
        key = tuple(round(v, 6) for v in p)
        if key not in cache:
            cache[key] = occ.addPoint(*p)
        return cache[key]

    def L(a, b):
        k = (min(a, b), max(a, b))
        if k not in edges:
            edges[k] = occ.addLine(a, b)
        return edges[k]

    surfs = []
    for face in faces_pts:
        ids = [P(p) for p in face]
        loop = [L(ids[i], ids[(i + 1) % len(ids)]) for i in range(len(ids))]
        surfs.append(occ.addPlaneSurface([occ.addCurveLoop(loop)]))
    return [(3, occ.addVolume([occ.addSurfaceLoop(surfs)]))]


def hexa(a, b):
    """Solid between two matching quads a (one end) and b (other end), each 4 xyz points in order."""
    faces = [a, b[::-1]] + [[a[i], b[i], b[(i + 1) % 4], a[(i + 1) % 4]] for i in range(4)]
    return poly_volume(faces)


def square(h, z):
    return [(-h, -h, z), (h, -h, z), (h, h, z), (-h, h, z)]


def frustum(z0, z1, half):
    """Square frustum between z0 and z1 with half-width half(z)."""
    return hexa(square(half(z0), z0), square(half(z1), z1))


def cyl_at(r, az, z0, z1, rad):
    x, y = r * math.cos(az), r * math.sin(az)
    return (3, occ.addCylinder(x, y, z0, 0, 0, z1 - z0, rad))


def radial_hole(az, r0, r1, z, rad):
    c, s = math.cos(az), math.sin(az)
    return (3, occ.addCylinder(r0 * c, r0 * s, z, (r1 - r0) * c, (r1 - r0) * s, 0, rad))


def ring(z0, z1, r0, r1):
    o = occ.addCylinder(0, 0, z0, 0, 0, z1 - z0, r1)
    if r0 <= 0:
        return [(3, o)]
    i = occ.addCylinder(0, 0, z0 - 1, 0, 0, z1 - z0 + 2, r0)
    res, _ = occ.cut([(3, o)], [(3, i)])
    return res


def slab(z0, z1, big=2000.0):
    return [(3, occ.addBox(-big, -big, z0, 2 * big, 2 * big, z1 - z0))]


def clean():
    occ.synchronize()
    v = gmsh.model.getEntities(3)
    keep = set(gmsh.model.getBoundary(v, combined=False, oriented=False, recursive=True))
    for dim in (2, 1):
        keep |= set(gmsh.model.getBoundary([e for e in keep if e[0] == dim + 1], combined=False,
                                           oriented=False, recursive=False))
    stray = [e for d in (0, 1, 2) for e in gmsh.model.getEntities(d) if e not in keep]
    if stray:
        occ.remove(stray)
        occ.synchronize()


def quiet_write(path):
    """gmsh.write without the OCC STEP writer's banner on stdout."""
    sys.stdout.flush()
    fd = os.dup(1)
    nul = os.open(os.devnull, os.O_WRONLY)
    os.dup2(nul, 1)
    try:
        gmsh.write(str(path))
    finally:
        os.dup2(fd, 1)
        os.close(nul)
        os.close(fd)


def mesh_write(path, size=12.0):
    gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", 36)
    gmsh.option.setNumber("Mesh.MeshSizeMax", size)
    gmsh.option.setNumber("Mesh.MeshSizeMin", 0.5)
    for attempt, alg in enumerate((6, 1, 5, 2)):
        try:
            gmsh.option.setNumber("Mesh.Algorithm", alg)
            gmsh.model.mesh.clear()
            gmsh.model.mesh.generate(2)
            break
        except Exception:
            if attempt == 3:
                raise
    gmsh.option.setNumber("Mesh.Algorithm", 6)
    gmsh.option.setNumber("Mesh.Binary", 1)
    gmsh.write(str(path))
    return gmsh.model.mesh.getNodes()[1].reshape(-1, 3)


def save(name, material, category, explode=(0, 0, 0), group="", note="", printable=False, qty=1):
    clean()
    gmsh.model.removePhysicalGroups()
    quiet_write(OUT / "step" / f"{name}.step")
    vol = sum(occ.getMass(*d) for d in gmsh.model.getEntities(3))
    xyz = mesh_write(OUT / "stl" / f"{name}.stl")
    bb = [*xyz.min(0), *xyz.max(0)]
    rec = dict(name=name, material=material, category=category, qty=qty, explode_mm=[float(v) for v in explode],
               group=group, volume_cm3=round(vol / 1000, 1), bbox_mm=[round(float(v), 1) for v in bb], note=note)
    if printable:
        rec["print_files"] = split_for_print(name, bb)
    parts.append(rec)
    print(f"  {name}: {rec['volume_cm3']} cm3" + (f", {len(rec['print_files'])} print pieces" if printable else ""),
          flush=True)
    return rec


def _fits(xyz):
    s = np.sort(xyz.max(0) - xyz.min(0))
    return s[0] <= BED and s[1] <= BED and s[2] <= 1.3 * BED


def split_for_print(name, bb):
    """Cut a part into z slabs of at most SLAB mm, then into equal sectors about the axis until every piece fits.

    Sectors start at azimuth 0, so with 8 or more each piece lies inside one
    octant. Pieces stay in assembly coordinates.
    """
    z0, z1 = bb[2], bb[5]
    nz = max(1, math.ceil((z1 - z0 - 0.5) / SLAB))
    big = 2 * max(abs(v) for v in bb) + 100
    for ns in (1, 2, 4, 8, 12, 16, 24):
        files, ok = [], True
        tmp = []
        for k in range(nz):
            za, zb = z0 - 0.5 + (z1 - z0 + 1) * k / nz, z0 - 0.5 + (z1 - z0 + 1) * (k + 1) / nz
            for j in range(ns):
                gmsh.model.add(f"{name}-{k}-{j}")
                o = gmsh.model.occ
                src = o.importShapes(str(OUT / "step" / f"{name}.step"))
                if ns == 1:
                    tool = o.addBox(-big, -big, za, 2 * big, 2 * big, zb - za)
                else:
                    tool = o.addCylinder(0, 0, za, 0, 0, zb - za, big, angle=2 * math.pi / ns)
                    o.rotate([(3, tool)], 0, 0, 0, 0, 0, 1, 2 * math.pi * j / ns)
                res, _ = o.intersect(src, [(3, tool)])
                o.synchronize()
                if res and sum(o.getMass(3, t) for d, t in res if d == 3) > 1.0:
                    fn = f"{name}-{k + 1}" + (f"-s{j + 1}" if ns > 1 else "") + ".stl"
                    xyz = mesh_write(OUT / "print" / fn, size=8.0)
                    tmp.append(fn)
                    if not _fits(xyz):
                        ok = False
                gmsh.model.remove()
                if not ok:
                    break
            if not ok:
                break
        if ok:
            files = tmp
            break
        for fn in tmp:
            (OUT / "print" / fn).unlink(missing_ok=True)
    else:
        warnings.append(f"{name}: no sector split fits the bed")
    gmsh.model.setCurrent(name)
    return files


# ---------------------------------------------------------------- cabinet
def wall_quads(side, z0, z1, outer=False):
    f = ohw if outer else hw
    out = []
    for z in (z0, z1):
        h = f(z)
        if side == "right":
            out += [(h, -h, z), (h, h, z)]
        elif side == "left":
            out += [(-h, h, z), (-h, -h, z)]
        elif side == "top":
            out += [(h, h, z), (-h, h, z)]
        else:
            out += [(-h, -h, z), (h, -h, z)]
    a, b, c, d = out
    return [a, b, d, c]


def walls():
    for side, ex in (("right", (350, 0, 0)), ("left", (-350, 0, 0)), ("top", (0, 350, 0)), ("bottom", (0, -350, 0))):
        start(f"wall_{side}")
        inner = wall_quads(side, Z_WB, Z_M)
        outer = wall_quads(side, Z_WB, Z_M, outer=True)
        hexa(inner, outer)
        save(f"wall_{side}", "ply", "cabinet", explode=ex, group="cabinet",
             note=f"18 mm birch, trapezoid, compound mitres both long edges, ends bevelled {K.half_angle_deg:.1f} deg")


def mitre_angles():
    """Blade bevel and edge taper for the four identical frustum panels (each face flat on the saw)."""
    p = np.array(wall_quads("right", Z_WB, Z_M))          # inner face of the right wall: a, b, (z1 +y), (z1 -y)
    e = p[2] - p[1]
    e /= np.linalg.norm(e)                                 # corner line along the +y edge, back to mouth
    n_p = np.array([math.cos(A), 0, -math.sin(A)])         # right wall outward normal
    n_t = np.array([0, math.cos(A), -math.sin(A)])         # top wall outward normal
    m = n_p - n_t
    m /= np.linalg.norm(m)                                 # mitre (bisector) plane normal
    u = np.cross(e, n_p)
    u /= np.linalg.norm(u)
    bevel = math.degrees(math.acos(min(1.0, abs(float(u @ m)))))
    length_dir = np.array([math.sin(A), 0, math.cos(A)])   # up the panel, square to its ends
    taper = math.degrees(math.acos(min(1.0, abs(float(e @ length_dir)))))
    dihedral = math.degrees(math.pi - math.acos(float(n_p @ n_t)))
    return dict(blade_bevel_deg=round(bevel, 2), edge_taper_deg=round(taper, 2),
                end_bevel_deg=round(K.half_angle_deg, 2), corner_dihedral_deg=round(dihedral, 2))


def baffle():
    start("baffle")
    b = frustum(Z_BAF, 0.0, hw)
    holes = [(3, occ.addCylinder(0, 0, Z_BAF - 1, 0, 0, -Z_BAF + 2, Z_CUT))]
    pcd = KD.frame_od / 2 * MM - 9
    for k in range(8):
        az = math.radians(22.5 + 45 * k)
        holes.append(cyl_at(pcd, az, Z_BAF - 1, 1, 3.5))
    for i, az in enumerate(STANDOFF_AZ):
        holes.append(cyl_at(R_STANDOFF, az, Z_BAF - 1, 1, 6.0 if i == DUCT else 4.25))
    occ.cut(b, holes)
    save("baffle", "ply", "cabinet", explode=(0, 0, -260), group="cabinet",
         note=f"25 mm birch, {2*hw(0):.0f} square at the front face, edges bevelled {K.half_angle_deg:.1f} deg, "
              f"{2*Z_CUT:.0f} mm cutout, 8 flange holes on {2*pcd:.0f} PCD, 4 standoff bolts (one 12 mm duct hole)")
    return dict(front_square_mm=round(2 * hw(0), 1), back_square_mm=round(2 * hw(Z_BAF), 1), thickness_mm=K.baffle * MM,
                cutout_diameter_mm=round(2 * Z_CUT, 1), flange_pcd_mm=round(2 * pcd, 1), flange_holes=8,
                standoff_pcd_mm=round(2 * R_STANDOFF, 1), edge_bevel_deg=K.half_angle_deg)


def back_panel():
    start("back")
    b = frustum(Z_BACK, Z_WB, ohw)
    h = hw(Z_WB)
    groove_o = [(3, occ.addBox(-(h + 9), -(h + 9), Z_WB - 2, 2 * (h + 9), 2 * (h + 9), 3))]
    groove_i = [(3, occ.addBox(-(h + 2), -(h + 2), Z_WB - 3, 2 * (h + 2), 2 * (h + 2), 5))]
    groove, _ = occ.cut(groove_o, groove_i)
    tools = list(groove)
    rs = h + 13.5
    for k in range(4):
        for t in (-0.6, 0.0, 0.6):
            x, y = [(rs, t * h), (-rs, t * h), (t * h, rs), (t * h, -rs)][k]
            tools.append((3, occ.addCylinder(x, y, Z_BACK - 1, 0, 0, T + 2, 2.75)))
    tools.append((3, occ.addCylinder(0, 0, Z_BACK - 1, 0, 0, T + 2, 32)))
    occ.cut(b, tools)
    save("back", "ply", "cabinet", explode=(0, 0, -480), group="cabinet",
         note=f"18 mm birch, removable, {2*ohw(Z_WB):.0f} square at the front face, edges bevelled "
              f"{K.half_angle_deg:.1f} deg to continue the walls; 2 mm deep gasket groove; 12 M5 screws into "
              f"inserts in the wall ends; 64 mm terminal cup")
    return dict(front_square_mm=round(2 * ohw(Z_WB), 1), back_square_mm=round(2 * ohw(Z_BACK), 1), thickness_mm=T,
                edge_bevel_deg=K.half_angle_deg, screws="12 x M5 into threaded inserts in the wall end grain",
                gasket="3 mm closed-cell foam in a 7 mm wide, 2 mm deep groove over the wall ends")


def filler_tet(sx, sy):
    c0 = K.filler(0.0) * MM
    h0 = hw(0.0)
    a = (sx * h0, sy * h0, 0.0)
    b = (sx * (h0 - c0), sy * h0, 0.0)
    c = (sx * h0, sy * (h0 - c0), 0.0)
    apex = (sx * hw(Z_M), sy * hw(Z_M), Z_M)
    return poly_volume([[a, b, c], [a, apex, b], [b, apex, c], [c, apex, a]])


CORNERS = (("tr", 1, 1), ("tl", -1, 1), ("bl", -1, -1), ("br", 1, -1))


def fillers():
    for name, sx, sy in CORNERS:
        start(f"filler_{name}")
        f = filler_tet(sx, sy)
        occ.cut(f, kick_vanes(only=[math.atan2(sy, sx)]))
        save(f"filler_{name}", "ply", "filler", explode=(sx * 230, sy * 230, 0), group="cabinet",
             note=f"corner filler: right-angle triangle, {K.filler(0)*MM:.0f} mm legs at the baffle to nothing at "
                  f"the mouth; hardwood or 18 mm ply laminations; slot for the vane")
    c0 = K.filler(0.0) * MM
    lam = []
    z = 0.0
    while z < Z_M - 1:
        z1 = min(z + T, Z_M)
        lam.append(dict(z0_mm=round(z, 1), z1_mm=round(z1, 1), leg_at_z0_mm=round(K.filler(z / MM) * MM, 1)))
        z = z1
    return dict(leg_at_baffle_mm=round(c0, 1), length_mm=round(Z_M, 1), face_width_at_baffle_mm=round(c0 * math.sqrt(2), 1),
                note="Rip a square hardwood blank of the leg size on its diagonal for two fillers, then taper each "
                     "to a point at the mouth. Or stack 18 mm ply triangles (laminations below) and plane the face.",
                laminations=lam)


# ---------------------------------------------------------------- supports
def kick_vanes(only=None):
    """Four 6 mm streamlined vanes from the centre body to the corner fillers (built in the current model)."""
    out = []
    z0, z1 = Z_VANE - VANE_CHORD / 2, Z_VANE + VANE_CHORD / 2
    zz = np.linspace(z0, z1, 7)
    r0 = min(K.body_radius(z / MM) for z in zz) * MM - 15
    r1 = max(hw(z) * math.sqrt(2) - K.filler(z / MM) * MM / math.sqrt(2) for z in zz) + 10
    for az in [math.radians(45 + 90 * k) for k in range(4)]:
        if only is not None and not any(abs(math.remainder(az - o, 2 * math.pi)) < 1e-6 for o in only):
            continue
        rect = occ.addRectangle(-VANE_T / 2, -VANE_CHORD / 2, 0, VANE_T, VANE_CHORD, roundedRadius=VANE_T / 2 - 0.01)
        ex = occ.extrude([(2, rect)], 0, 0, r1 - r0)
        v = [e for e in ex if e[0] == 3]
        # local (x thickness, y chord, z radial) -> global (radial, tangential, z)
        occ.affineTransform(v, [0, 0, 1, r0, 1, 0, 0, 0, 0, 1, 0, Z_VANE])
        occ.rotate(v, 0, 0, 0, 0, 0, 1, az)
        box = frustum(0.0, Z_M, hw)
        res, _ = occ.intersect(v, box)
        out += res
    return out


def standoff_shape(i):
    az = STANDOFF_AZ[i]
    s_ = [cyl_at(R_STANDOFF, az, 0.0, Z_TURN + SOCKET, STANDOFF_R)]
    if i == DUCT:
        s_, _ = occ.cut(s_, [cyl_at(R_STANDOFF, az, -1, Z_TURN + SOCKET + 1, 6.0)])
    return s_


def supports():
    for i, az in enumerate(STANDOFF_AZ):
        name = "standoff_duct" if i == DUCT else f"standoff_{i + 1}"
        start(name)
        standoff_shape(i)
        c, s = math.cos(az), math.sin(az)
        save(name, "aluminium", "support", explode=(c * 160, s * 160, -60), group="supports",
             note=f"20 mm aluminium round, {Z_TURN + SOCKET:.0f} long, M8 into the baffle, {SOCKET:.0f} mm into "
                  f"the shroud" + ("; 12 mm bore: cable duct from the chamber to the centre body" if i == DUCT else ""))
    for k, (name, sx, sy) in enumerate(CORNERS):
        start(f"vane_{name}")
        kick_vanes(only=[math.atan2(sy, sx)])
        save(f"vane_{name}", "aluminium", "support", explode=(sx * 260, sy * 260, 60), group="supports",
             note=f"{VANE_T:.0f} mm plate, {VANE_CHORD:.0f} mm chord with round edges, body to corner filler, "
                  f"centred {Z_M - Z_VANE:.0f} mm behind the kick mouth")


# ---------------------------------------------------------------- centre body
def lip_arc(n=24):
    """Rolled lip at the mid mouth: from the bore over the front, round the outside, to the back of the roll."""
    rho = M.lip_radius * MM
    al = math.radians(M.half_angle_deg)
    R = M.radius(M.length) * MM
    cr, cz = R + rho * math.cos(al), Z_MM - rho * math.sin(al)
    psi = np.linspace(-math.pi / 2, math.pi - al, n)
    return [(cr + rho * math.cos(p), cz + rho * math.sin(p)) for p in psi], (cr, cz - rho)


def s_of(z):
    return (z - Z_MT) / MM


def body_outer(z):
    """Outside radius of the centre body at z (mm), ignoring the lip roll."""
    if z <= Z_TURN:
        return K.body_radius(K.turn_z) * MM
    if z <= Z_M:
        return K.body_radius(z / MM) * MM
    return M.outer(min(s_of(z), M.length)) * MM


def body_inner(z):
    """Largest cavity or bore radius at z (mm)."""
    if z < J1:
        return R_CS
    if z < Z_MOTOR - WALL_B:
        return R_MC
    if z < Z_MOTOR:
        return 0.0
    if z <= Z_MT:
        return R_POD
    return M.radius(min(s_of(z), M.length)) * MM


def ring_z(r):
    """Underside of the lead-in ring over the outer mid cone (r in m), absolute z in mm."""
    return zs(float(M.s_cone(r)) + M.gap)


def body_profile():
    d, m = KD, M
    cs, rim = K.channel_start, d.rim_r
    segs = []
    rr = np.linspace(cs, rim, 40)
    segs.append(("spline", [(r * MM, float(K.shroud_z(r)) * MM) for r in rr]))
    zr = float(K.shroud_z(rim)) * MM
    lip = [(rim * MM, zr)]
    if Z_TURN - zr > 0.1:
        lip.append((rim * MM, Z_TURN))
    lip.append((body_outer(Z_TURN), Z_TURN))
    segs.append(("line", lip))
    u = np.linspace(0, 1, 70)
    zz = K.turn_z + (K.mouth_z - K.turn_z) * (1 - (1 - u) ** 1.6)
    segs.append(("spline", [(K.body_radius(z) * MM, z * MM) for z in zz]))
    arc, back = lip_arc()
    z_lb = back[1]
    s_m, s_lb = s_of(Z_M), s_of(z_lb)
    if s_lb <= s_m + 0.005:
        warnings.append("rolled lip reaches behind the kick mouth")
        s_lb = s_m + 0.005
    ss = np.linspace(s_m, s_lb, 30)
    segs.append(("spline", [(m.outer(s) * MM, zs(s)) for s in ss]))
    segs.append(("line", [(m.outer(s_lb) * MM, zs(s_lb)), arc[0]]))
    segs.append(("spline", arc))
    sb = np.linspace(m.length, 0, 60)
    segs.append(("spline", [(m.radius(s) * MM, zs(s)) for s in sb]))
    r_ot = m.radius(0.0)
    md_rim = MD.rim_r
    rr = np.linspace(r_ot, md_rim, 20)
    segs.append(("line", [(r_ot * MM, Z_MT), (r_ot * MM, ring_z(r_ot))]))
    segs.append(("spline", [(r * MM, ring_z(r)) for r in rr]))
    r_fl = MD.frame_od / 2 * MM - 12
    z_sur = Z_MF + M.gap * MM          # over the surround: the cone gap above a surround flush with the flange face
    segs.append(("line", [(md_rim * MM, ring_z(md_rim)), (md_rim * MM + 2, z_sur), (r_fl, z_sur), (r_fl, Z_MF),
                          (R_POD, Z_MF), (R_POD, Z_MOTOR), (0, Z_MOTOR), (0, Z_MOTOR - WALL_B),
                          (R_MC, Z_MOTOR - WALL_B), (R_MC, J1), (R_CS, J1)]))
    # the loop closes from (R_CS, J1) back to the first shroud point
    return segs


def body_features(body):
    """Standoff sockets, cable channels and vane slots cut from the whole body."""
    tools = []
    for az in STANDOFF_AZ:
        tools.append(cyl_at(R_STANDOFF, az, Z_TURN - 1, Z_TURN + SOCKET, STANDOFF_R + 0.25))
    az = STANDOFF_AZ[DUCT]
    tools.append(cyl_at(R_STANDOFF, az, Z_TURN + SOCKET - 2, Z_CAB, 6.0))
    tools.append(radial_hole(az, R_POD - 5, R_STANDOFF, Z_CAB, 6.0))
    # HF and mid cables: pod -> forward in the wall -> radially into the bore at a bullet vane
    z_h = zs(S_HF_CABLE)
    z_p = Z_MF - 30
    tools.append(cyl_at(R_HF_CH, BV_AZ[0], z_p, z_h, 4.0))
    tools.append(radial_hole(BV_AZ[0], R_POD - 5, R_HF_CH, z_p, 4.0))
    tools.append(radial_hole(BV_AZ[0], M.radius(S_HF_CABLE) * MM - 3, R_HF_CH, z_h, 4.0))
    tools += kick_vanes()
    tools += bullet_vanes(slots=True)
    res, _ = occ.cut(body, tools)
    return res


def build_body():
    start("body_full")
    body = revolve(body_profile())
    body_features(body)
    clean()
    TMP.mkdir(exist_ok=True)
    quiet_write(TMP / "body_full.step")
    v = sum(occ.getMass(*e) for e in gmsh.model.getEntities(3))
    return v


def joint_ring(zj):
    lo, hi = zj - 25, zj + 25
    zz = np.linspace(lo, hi, 21)
    r_in = max(body_inner(z) for z in zz)
    r_out = min(body_outer(z) for z in zz)
    if zj <= Z_MOTOR + 1e-6 and zj >= Z_MOTOR - 1e-6:
        r_in = max(r_in, R_POD)
    return r_in, r_out


def module(name, z0, z1, joints_lo, joints_hi, explode, note):
    start(name)
    src = occ.importShapes(str(TMP / "body_full.step"))
    res, _ = occ.intersect(src, slab(z0, z1))
    body = res
    for zj in joints_hi:            # spigot on top
        r_in, r_out = joint_ring(zj)
        if r_out - r_in < 25:
            warnings.append(f"{name}: wall only {r_out - r_in:.0f} mm at the joint z = {zj:.0f}")
            continue
        sp = ring(zj - 0.5, zj + SPIGOT_H, r_in + 5, r_in + 11)
        body, _ = occ.fuse(body, sp)
    tools = []
    for zj in joints_lo:            # recess underneath
        r_in, r_out = joint_ring(zj)
        if r_out - r_in < 25:
            continue
        tools += ring(zj - 1, zj + SPIGOT_H + 0.3, r_in + 4.8, r_in + 11.2)
    for zj in joints_lo + joints_hi:
        r_in, r_out = joint_ring(zj)
        if r_out - r_in < 25:
            continue
        rb = 0.5 * (r_in + 11 + r_out)
        for k in range(6):
            tools.append(cyl_at(rb, math.radians(60 * k), zj - 25, zj + 25, 2.75))
    if tools:
        body, _ = occ.cut(body, tools)
    return save(name, "print_black", "print", explode=explode, group="centre body", printable=True, note=note)


def centre_body():
    vol = build_body()
    zbot = Z_TURN - 1
    z_top = max(p[1] for p in lip_arc()[0]) + 1
    # mid horn joint: the first z behind the kick mouth where the wall is 40 mm thick over the bolt length,
    # clear of the vanes
    j4 = None
    for z in np.arange(Z_M - 5, Z_MT + 30, -2.0):
        r_in, r_out = joint_ring(z)
        if r_out - r_in >= 40 and abs(z - Z_VANE) > VANE_CHORD / 2 + 30:
            j4 = float(z)
            break
    j4 = j4 or (Z_MT + Z_M) / 2
    mods = [
        ("kick_shroud", zbot, J1, "underside follows the kick cone; lip; standoff sockets"),
        ("motor_fairing", J1, Z_MOTOR, f"covers the kick motor with {K.motor_clear*MM:.0f} mm clearance"),
        ("mid_pod", Z_MOTOR, Z_MF, f"sealed rear volume of the mid, {M.rear_volume_litres:.1f} L"),
        ("mid_horn_rear", Z_MF, j4, "lead-in ring over the mid cone, mid horn bore, vane slots; mid driver screws to its back face"),
        ("mid_horn_front", j4, z_top, f"mid horn to the mouth, rolled lip of {M.lip_radius*MM:.0f} mm radius"),
    ]
    joints = [J1, Z_MOTOR, Z_MF, j4]
    for i, (name, a, b, note) in enumerate(mods):
        lo = [a] if a in joints else []
        hi = [b] if b in joints else []
        module(name, a, b, lo, hi, explode=(0, 0, 120 + 90 * i), note=note)
    return dict(joints_mm=[round(z, 1) for z in joints], volume_cm3=round(vol / 1000, 1))


# ---------------------------------------------------------------- bullet
def bullet_vanes(slots=False):
    out = []
    s0, s1 = BV_S0, BV_S1
    r0 = min(M.plug(s) for s in np.linspace(s0, s1, 5)) * MM - 2
    r1 = max(M.radius(s) for s in np.linspace(s0, s1, 5)) * MM + 3
    for az in BV_AZ:
        chord = (s1 - s0) * MM
        t = BV_T + (0.4 if slots else 0.0)
        rect = occ.addRectangle(-t / 2, -chord / 2 - (0.2 if slots else 0), 0, t, chord + (0.4 if slots else 0),
                                roundedRadius=t / 2 - 0.01)
        ex = occ.extrude([(2, rect)], 0, 0, r1 - r0)
        v = [e for e in ex if e[0] == 3]
        occ.affineTransform(v, [0, 0, 1, r0, 1, 0, 0, 0, 0, 1, 0, zs((s0 + s1) / 2)])
        occ.rotate(v, 0, 0, 0, 0, 0, 1, az)
        out += v
    return out


def bullet_profile():
    m = M
    r_hb = TW.body_diameter / 2 * MM + 0.3
    s_cav = m.s_exit - m.hf_depth
    rr = np.linspace(0, m.plug_r0, 20)
    nose = [(r * MM, zs(float(m.nose(r)))) for r in rr]
    segs = [("spline", nose), ("line", [nose[-1], (m.plug_r0 * MM, Z_MT)])]
    ss = np.linspace(0, m.s_tip, 50)
    segs.append(("spline", [(m.plug(s) * MM, zs(s)) for s in ss]))
    sh = np.linspace(m.s_tip, m.s_exit, 30)
    segs.append(("line", [(m.tip_radius * MM, zs(m.s_tip)), (m.hf_horn_radius(m.s_tip) * MM, zs(m.s_tip))]))
    segs.append(("spline", [(m.hf_horn_radius(s) * MM, zs(s)) for s in sh]))
    segs.append(("line", [(m.hf_exit_diameter / 2 * MM, zs(m.s_exit)), (r_hb, zs(m.s_exit)), (r_hb, zs(s_cav)),
                          (0, zs(s_cav))]))
    return segs, s_cav


FELT = 3.0


def felt_shape():
    """Felt disc on the nose underside: its face is the designed nose surface, recessed FELT mm into the bullet."""
    rr = np.linspace(0, M.plug_r0 - 0.002, 20)
    face = [(r * MM, zs(float(M.nose(r)))) for r in rr]
    back = [(r, z + FELT) for r, z in face[::-1]]
    return revolve([("spline", face), ("line", [face[-1], back[0]]), ("spline", back)])


def bullet():
    segs, s_cav = bullet_profile()
    z_split = zs(s_cav)
    z_lo = min(p[1] for p in segs[0][1]) - 1
    r_hb = TW.body_diameter / 2 * MM + 0.3
    # nose cap: everything behind the HF cavity
    start("bullet_nose")
    b = revolve(segs)
    b, _ = occ.intersect(b, slab(z_lo, z_split))
    sp = ring(z_split - 0.5, z_split + 3, r_hb - 3.2, r_hb - 0.3)
    b, _ = occ.fuse(b, sp)
    occ.cut(b, felt_shape())
    save("bullet_nose", "print_orange", "print", explode=(0, 0, 640), group="bullet", printable=True,
         note=f"bullet nose: follows the mid cone at the gap; closes the HF cavity; 3 mm spigot; "
              f"{FELT:.0f} mm recess for the felt")
    start("felt")
    felt_shape()
    save("felt", "driver_cone", "felt", explode=(0, 0, 600), group="bullet",
         note=f"{FELT:.0f} mm wool felt on the bullet nose underside, set into a recess so its face is at the "
              f"designed gap from the mid cone (MidHorn.felt_loss models its damping)")
    start("bullet")
    b = revolve(segs)
    b, _ = occ.intersect(b, slab(z_split, zs(M.s_tip) + 1))
    holes = [radial_hole(BV_AZ[0], r_hb - 1, M.plug(S_HF_CABLE) * MM + 1, zs(S_HF_CABLE), 1.5)]
    slots = bullet_vanes(slots=True)
    occ.cut(b, holes + slots)
    save("bullet", "print_orange", "print", explode=(0, 0, 700), group="bullet", printable=True,
         note=f"smooth bullet round the {TW.model} body cylinder, printed HF horn "
              f"{M.hf_exit_diameter*MM:.1f} to {2*M.hf_mouth_radius*MM:.0f} mm over {M.hf_len*MM:.0f} mm")
    start("bullet_vanes")
    v = bullet_vanes()
    occ.cut(v, [radial_hole(BV_AZ[0], r_hb - 2, M.radius(S_HF_CABLE) * MM + 4, zs(S_HF_CABLE), 1.5)])
    save("bullet_vanes", "print_orange", "print", explode=(0, 0, 560), group="bullet", printable=True,
         note=f"three {BV_T:.0f} mm vanes hold the bullet in the mid throat; one carries the HF cable in a 3 mm bore")


# ---------------------------------------------------------------- drivers (proxies)
def cone_shell(drv, zmap, t=1.5, rim_out=None, roll_h=None):
    """Cone between vc_r and rim_r as a thin shell, plus dust cap and a half-roll surround."""
    rr = np.linspace(drv.vc_r, drv.rim_r, 30)
    top = [(r * MM, zmap(float(drv.cone_z(r)) * MM)) for r in rr]
    sgn = 1 if zmap(1.0) > zmap(0.0) else -1
    bot = [(r, z + sgn * t) for r, z in top[::-1]]
    segs = [("spline", top), ("line", [top[-1], bot[0]]), ("spline", bot), ("line", [bot[-1], top[0]])]
    # close: the last line ends at top[0]; drop it and let revolve close the loop
    segs = segs[:-1]
    out = revolve(segs)
    # dust cap: dome over the voice coil, bulging away from the motor
    rv = drv.vc_r * MM
    zn = zmap(drv.neck_z * MM)
    th = np.linspace(0, math.pi / 2, 12)
    dome = [(rv * math.sin(a), zn - sgn * 0.3 * rv * math.cos(a)) for a in th]
    out += revolve([("spline", dome), ("line", [dome[-1], (0, zn)])])
    # surround: half roll from the rim outward, bulging away from the motor
    w = (rim_out - drv.rim_r * MM) if rim_out else 0.11 * drv.rim_r * MM
    h = roll_h or w / 2
    rc, zc = drv.rim_r * MM + w / 2, zmap(drv.rim_z * MM)
    a = np.linspace(0, math.pi, 16)
    outer = [(rc - w / 2 * math.cos(x), zc - sgn * h * math.sin(x)) for x in a]
    inner = [(rc - (w / 2 - 1.2) * math.cos(x), zc - sgn * (h - 1.2) * math.sin(x)) for x in a[::-1]]
    out += revolve([("spline", outer), ("line", [outer[-1], inner[0]]), ("spline", inner)])
    return out


def basket(drv, zmap, arm, r_flange_in, z_flange_back, n=4, width_deg=10):
    """Basket arms following the cone at ``arm`` (mm) above it, from the flange to the top plate."""
    t = 6.0
    rr = np.linspace(drv.spider_r * MM + 4, drv.rim_r * MM, 16)
    sgn = 1 if zmap(1.0) > zmap(0.0) else -1
    lower = [(r, zmap(float(drv.cone_z(r / MM)) * MM) + sgn * arm) for r in rr]
    upper = [(r, z + sgn * t) for r, z in lower[::-1]]
    ztp = zmap(drv.top_plate_z * MM)
    pts = [(drv.spider_r * MM + 4, ztp)] + lower + [(r_flange_in, z_flange_back)] + \
          [(r_flange_in + 2 * t, z_flange_back)] + upper + [(drv.spider_r * MM + 4 + t, ztp)]
    width = 2 * drv.rim_r * MM * math.sin(math.radians(width_deg / 2))
    pts = _dedupe(pts)
    ids = [occ.addPoint(r, 0, z) for r, z in pts]
    lines = [occ.addLine(ids[i], ids[(i + 1) % len(ids)]) for i in range(len(ids))]
    surf = occ.addPlaneSurface([occ.addCurveLoop(lines)])
    occ.translate([(2, surf)], 0, -width / 2, 0)
    ex = occ.extrude([(2, surf)], 0, width, 0)
    arm = [e for e in ex if e[0] == 3]
    out = list(arm)
    for k in range(1, n):
        c = occ.copy(arm)
        occ.rotate(c, 0, 0, 0, 0, 0, 1, 2 * math.pi * k / n)
        out += c
    return out


def drivers():
    d = KD
    zk = lambda z: z  # noqa: E731  kick: cone_z is already in cabinet z (flange front face at z = 0)
    # kick frame: flange, basket arms at 0, 90, 180, 270 deg, spider housing
    start("drv_kick_frame")
    fl = ring(0.0, d.flange_t * MM, Z_CUT + 2, d.frame_od / 2 * MM)
    holes = [cyl_at(d.frame_od / 2 * MM - 9, math.radians(22.5 + 45 * k), -1, d.flange_t * MM + 1, 3.5) for k in range(8)]
    fl, _ = occ.cut(fl, holes)
    arms = basket(d, zk, K.arm * MM, Z_CUT + 4, d.flange_t * MM)
    hous = ring(d.neck_z * MM - 5, d.top_plate_z * MM, d.spider_r * MM + 2, d.spider_r * MM + 5)
    occ.fuse(fl, arms + hous)
    save("drv_kick_frame", "driver_metal", "driver", explode=(0, 0, -120), group="kick driver",
         note=f"proxy of the {D.label(D.KICK_RECORD)} frame, reversed: flange on the baffle front, basket into the horn")
    start("drv_kick_cone")
    cone_shell(d, zk, rim_out=Z_CUT - 3)
    save("drv_kick_cone", "driver_cone", "driver", explode=(0, 0, -120), group="kick driver",
         note="proxy cone (estimated profile), dust cap and surround")
    start("drv_kick_motor")
    mot = [(3, occ.addCylinder(0, 0, d.top_plate_z * MM, 0, 0, d.motor_h * MM, d.motor_r * MM))]
    occ.cut(mot, [(3, occ.addCylinder(0, 0, d.top_plate_z * MM - 1, 0, 0, d.motor_h * MM + 2, 0.5 * d.vc_r * MM))])
    save("drv_kick_motor", "driver_metal", "driver", explode=(0, 0, -120), group="kick driver",
         note=f"kick motor proxy, {2*d.motor_r*MM:.0f} x {d.motor_h*MM:.0f} mm")
    # mid: normal orientation, flange front face at Z_MF, motor behind
    md = MD
    zmid = lambda z: Z_MF - z  # noqa: E731
    start("drv_mid_frame")
    fl = ring(Z_MF - md.flange_t * MM, Z_MF, md.frame_od / 2 * MM - 12, md.frame_od / 2 * MM)
    arms = basket(md, zmid, md.xmech * MM + 8, md.frame_od / 2 * MM - 12, Z_MF - md.flange_t * MM)
    hous = ring(zmid(md.top_plate_z * MM), zmid(md.neck_z * MM - 4), md.spider_r * MM + 2, md.spider_r * MM + 5)
    occ.fuse(fl, arms + hous)
    save("drv_mid_frame", "driver_metal", "driver", explode=(0, 0, 330), group="mid driver",
         note=f"proxy of the {D.label(D.MID_RECORD)} frame")
    start("drv_mid_cone")
    cone_shell(md, zmid, t=1.0, rim_out=md.frame_od / 2 * MM - 13, roll_h=md.rim_z * MM)
    save("drv_mid_cone", "driver_cone", "driver", explode=(0, 0, 330), group="mid driver",
         note="proxy cone (estimated profile); surround assumed flush with the flange face")
    start("drv_mid_motor")
    z0 = zmid(md.top_plate_z * MM) - md.motor_h * MM
    occ.addCylinder(0, 0, z0, 0, 0, md.motor_h * MM, md.motor_r * MM)
    save("drv_mid_motor", "driver_metal", "driver", explode=(0, 0, 330), group="mid driver", note="mid motor proxy")
    start("drv_hf")
    se = zs(M.s_exit)
    rb = TW.body_diameter / 2 * MM
    body = [(3, occ.addCylinder(0, 0, se - TW.body_depth * MM, 0, 0, TW.body_depth * MM, rb))]
    occ.cut(body, [(3, occ.addCylinder(0, 0, se - 10, 0, 0, 11, M.hf_exit_diameter / 2 * MM))])
    occ.addBox(-8, -4, se - M.hf_depth * MM, 16, 8, (M.hf_depth - TW.body_depth) * MM + 0.5)
    save("drv_hf", "driver_metal", "driver", explode=(0, 0, 760), group="bullet",
         note=f"{TW.model}: {TW.body_diameter*MM:.0f} mm body cylinder, {TW.body_depth*MM:.1f} mm deep, flange removed")


# ---------------------------------------------------------------- area check
def _slice_area(vol, z, keep):
    big = 2000.0
    rect = occ.addRectangle(-big, -big, z, 2 * big, 2 * big)
    cp = occ.copy(vol)
    res, _ = occ.intersect([(2, rect)], cp)
    tot = 0.0
    for d, t in res:
        if d != 2:
            continue
        x0, y0, _, x1, y1, _ = occ.getBoundingBox(2, t)
        if keep(max(abs(x0), abs(x1), abs(y0), abs(y1))):
            tot += occ.getMass(2, t)
    occ.remove(res, recursive=True)
    return tot


def area_check(n=60):
    gmsh.clear()
    gmsh.model.add("area")
    global occ
    occ = gmsh.model.occ
    imp = lambda name: occ.importShapes(str(OUT / "step" / f"{name}.step"))  # noqa: E731
    body = []
    for name in ("kick_shroud", "motor_fairing", "mid_pod", "mid_horn_rear", "mid_horn_front"):
        body += imp(name)
    fill = []
    for name, _, _ in CORNERS:
        fill += imp(f"filler_{name}")
    walls_ = []
    for side in ("right", "left", "top", "bottom"):
        walls_ += imp(f"wall_{side}")
    stand = []
    for i in range(4):
        stand += imp("standoff_duct" if i == DUCT else f"standoff_{i + 1}")
    vanes = []
    for name, _, _ in CORNERS:
        vanes += imp(f"vane_{name}")
    bul = imp("bullet_nose") + imp("bullet")
    bv = imp("bullet_vanes")
    # kick: frustum just inside the wall thickness, less walls, fillers, body, standoffs and vanes (the real air).
    # The exposed part of each vane (outside the body and filler slots) is sliced separately: real air plus the
    # exposed vane section is the open area of walls, fillers and body alone, which is what Kick.area models.
    probe = frustum(0.0, Z_M, lambda z: hw(z) + T / 2)
    air, _ = occ.cut(probe, occ.copy(walls_ + fill + body + stand + vanes))
    # the open passage from the design: inside the walls, outside the filler tetrahedra and the body of revolution
    zz = np.linspace(K.turn_z, K.mouth_z, 80)
    solid_body = revolve([("line", [(0, Z_TURN), (body_outer(Z_TURN), Z_TURN)]),
                          ("spline", [(K.body_radius(z) * MM, z * MM) for z in zz]),
                          ("line", [(body_outer(Z_M), Z_M), (0, Z_M)])])
    tets = []
    for _, sx, sy in CORNERS:
        tets += filler_tet(sx, sy)
    passage, _ = occ.cut(frustum(0.0, Z_M, hw), solid_body + tets)
    vexp, _ = occ.intersect(occ.copy(vanes), passage)
    t0 = time.time()
    kick_litres = sum(occ.getMass(*e) for e in air) / 1e6
    rows = []
    for z in np.linspace(Z_TURN + 2, Z_M - 2, n):
        rb = K.body_radius(z / MM) * MM
        a = _slice_area(air, z, lambda r, rb=rb: r > rb + 1)
        v = _slice_area(vexp, z, lambda r: True)
        model = K.area(z / MM) * 1e6
        rows.append(dict(kind="kick", z_mm=round(z, 2), s_mm="", cad_area_cm2=(a + v) / 100, model_area_cm2=model / 100,
                         diff_pct=100 * (a + v - model) / model, real_air_cm2=a / 100, vane_blockage_cm2=v / 100,
                         vane_blockage_pct=100 * v / (a + v)))
    # mid: revolve just inside the wall, less the body, the bullet and its vanes
    ss = np.linspace(0, M.length, 80)
    prof = [("line", [(0, Z_MT), (M.radius(0) * MM + M.wall * MM / 2, Z_MT)]),
            ("spline", [(M.radius(s) * MM + M.wall * MM / 2, zs(s)) for s in ss]),
            ("line", [(M.radius(M.length) * MM + M.wall * MM / 2, Z_MM), (0, Z_MM)])]
    mprobe = revolve(prof)
    # the HF unit fills its cavity in the bullet; without it the cavity joins the mid air through the cable hole
    r_hb = TW.body_diameter / 2 * MM + 0.3
    s_cav = M.s_exit - M.hf_depth
    hf = [(3, occ.addCylinder(0, 0, zs(s_cav) - 0.1, 0, 0, (M.s_exit - s_cav) * MM + 0.2, r_hb + 0.1))]
    mair, _ = occ.cut(mprobe, occ.copy(body + bul + bv) + hf)
    ann = revolve([("spline", [(M.plug(s) * MM, zs(s)) for s in np.linspace(0, M.s_tip, 60)]),
                   ("line", [(M.tip_radius * MM, zs(M.s_tip)), (0, zs(M.s_tip)), (0, Z_MM)]),
                   ("line", [(0, Z_MM), (M.radius(M.length) * MM, Z_MM)]),
                   ("spline", [(M.radius(s) * MM, zs(s)) for s in np.linspace(M.length, 0, 80)])])
    bexp, _ = occ.intersect(occ.copy(bv), ann)
    for s in np.linspace(0.001, M.length - 0.001, n):
        rp = M.plug(s) * MM
        a = _slice_area(mair, zs(s), lambda r, rp=rp: r > rp + 0.5)
        v = _slice_area(bexp, zs(s), lambda r: True)
        model = M.area(s) * 1e6
        rows.append(dict(kind="mid", z_mm=round(zs(s), 2), s_mm=round(s * MM, 2), cad_area_cm2=(a + v) / 100,
                         model_area_cm2=model / 100, diff_pct=100 * (a + v - model) / model, real_air_cm2=a / 100,
                         vane_blockage_cm2=v / 100, vane_blockage_pct=100 * v / (a + v)))
    # standoffs block the turn under the shroud lip, which has no model area: report them against the turn
    turn_circ = 2 * math.pi * R_STANDOFF
    standoff = dict(width_fraction_of_turn_pct=100 * 4 * 2 * STANDOFF_R / turn_circ,
                    turn_height_mm=Z_TURN - KD.flange_t * MM, at_radius_mm=R_STANDOFF)
    RES.mkdir(parents=True, exist_ok=True)
    cols = list(rows[0].keys())
    with open(RES / "area-check.csv", "w") as f:
        f.write(",".join(cols) + "\n")
        for r in rows:
            f.write(",".join(f"{r[c]:.4f}" if isinstance(r[c], float) else str(r[c]) for c in cols) + "\n")
    out = {}
    for kind in ("kick", "mid"):
        rr = [r for r in rows if r["kind"] == kind]
        d = np.array([r["diff_pct"] for r in rr])
        b = np.array([r["vane_blockage_pct"] for r in rr])
        i = int(np.argmax(np.abs(d)))
        out[kind] = dict(max_abs_diff_pct=round(float(abs(d[i])), 3), at_z_mm=rr[i]["z_mm"],
                         rms_diff_pct=round(float(np.sqrt(np.mean(d**2))), 3), max_vane_blockage_pct=round(float(b.max()), 2))
    out["kick_air_litres"] = round(kick_litres, 2)
    out["standoffs"] = {k: round(v, 2) for k, v in standoff.items()}
    print(f"  area check in {time.time() - t0:.0f} s: {json.dumps(out)}", flush=True)
    return out


# ---------------------------------------------------------------- 3 x 3 array
def _Rax(u, t):
    u = np.asarray(u, float)
    u = u / np.linalg.norm(u)
    k = np.array([[0, -u[2], u[1]], [u[2], 0, -u[0]], [-u[1], u[0], 0]])
    return np.eye(3) + math.sin(t) * k + (1 - math.cos(t)) * k @ k


def _face(w):
    """Outer face of a wall (box local, metres): back 0, back 1, mouth 0, mouth 1, and its outward normal.

    Vertex 0 is at -y for the side walls and -x for the top and bottom.
    """
    zb, zm = Z_BACK / MM, K.mouth_z
    sx, sy = {"+x": (1, 0), "-x": (-1, 0), "+y": (0, 1), "-y": (0, -1)}[w]
    pts = []
    for z in (zb, zm):
        h = K.outer_half_width(z)
        pts += [(sx * h, -h, z), (sx * h, h, z)] if sx else [(-h, sy * h, z), (h, sy * h, z)]
    n = np.array([sx * math.cos(A), sy * math.cos(A), -math.sin(A)])
    return np.array(pts), n


def _box_vertices():
    out = []
    for z in (Z_BACK / MM, K.mouth_z):
        h = K.outer_half_width(z)
        out += [(sx * h, sy * h, z) for sx in (-1, 1) for sy in (-1, 1)]
    return np.array(out)


def _tr(P, p):
    return (P[0] @ np.asarray(p).T).T + P[1]


def _joint(P, wp, Q, wq):
    fp, n_p = _face(wp)
    fq, n_q = _face(wq)
    fp, fq = _tr(P, fp), _tr(Q, fq)
    n_q, n_p = Q[0] @ n_q, P[0] @ n_p
    d1 = (fp - fq[0]) @ n_q
    d2 = (fq - fp[0]) @ n_p
    sep = np.linalg.norm(fp[2:] - fq[2:], axis=1)
    return d1, d2, sep


def _inside(P, Q):
    """Largest outside-distance of each of P's corners from box Q (negative: inside Q)."""
    v = _tr(P, _box_vertices())
    faces = [_face(w) for w in ("+x", "-x", "+y", "-y")]
    zb, zm = Z_BACK / MM, K.mouth_z
    out = []
    for p in v:
        q = Q[0].T @ (p - Q[1])
        out.append(max([float((q - f[0]) @ n) for f, n in faces] + [zb - q[2], q[2] - zm]))
    return np.array(out)


def _layout(th, ph, tc):
    """Poses (R, t) of the centre, the +x edge box and the +x +y corner box."""
    I = (np.eye(3), np.zeros(3))
    # edge: hinge on the centre's +x mouth front edge
    R = _Rax((0, 1, 0), th)
    fc, _ = _face("+x")
    fe, _ = _face("-x")
    c = fc[2:].mean(0)
    Pe = (R, c - R @ fe[2:].mean(0))
    Pc = (_Rax((-1, 1, 0), ph), np.array([tc[0], tc[0], tc[1]]))
    return I, Pe, Pc


def _mirror(P, S):
    return (S @ P[0] @ S, S @ P[1])


def _all_boxes(I, Pe, Pc):
    Sx, Sy = np.diag([-1.0, 1, 1]), np.diag([1.0, -1, 1])
    W = np.array([[0.0, 1, 0], [1, 0, 0], [0, 0, 1]])
    boxes = {(0, 0): I, (1, 0): Pe, (0, 1): _mirror(Pe, W), (1, 1): Pc}
    boxes[(-1, 0)] = _mirror(boxes[(1, 0)], Sx)
    boxes[(0, -1)] = _mirror(boxes[(0, 1)], Sy)
    boxes[(-1, 1)] = _mirror(Pc, Sx)
    boxes[(1, -1)] = _mirror(Pc, Sy)
    boxes[(-1, -1)] = _mirror(boxes[(-1, 1)], Sy)
    return boxes


def _twist(Pe, Pc):
    _, n1 = _face("+y")
    _, n2 = _face("-y")
    c = float(np.clip((Pe[0] @ n1) @ -(Pc[0] @ n2), -1, 1))
    return math.degrees(math.acos(c))


def _report(boxes):
    joints, worst = [], (0.0, "")
    for (i, j), P in boxes.items():
        for di, dj, wp, wq in ((1, 0, "+x", "-x"), (0, 1, "+y", "-y")):
            if (i + di, j + dj) not in boxes:
                continue
            d1, d2, sep = _joint(P, wp, boxes[(i + di, j + dj)], wq)
            g = np.maximum(d1, d2) * MM
            f, _ = _face(wp)
            v = _tr(P, f)
            rad = np.hypot(v[:, 0], v[:, 1])
            lab = []
            for k in range(4):
                other = k ^ 1
                end = "either end" if abs(rad[k] - rad[other]) < 1e-3 else (
                    "inner end" if rad[k] < rad[other] else "outer end")
                lab.append(("back, " if k < 2 else "mouth, ") + end)
            k = int(np.argmax(g))
            joints.append(dict(a=[i, j], b=[i + di, j + dj], vertices=lab, gap_mm=[round(float(x), 1) for x in g],
                               penetration_mm=round(float(max(0.0, -min(d1.min(), d2.min())) * MM), 2),
                               mouth_edge_offset_mm=[round(float(x) * MM, 1) for x in sep]))
            if g[k] > worst[0]:
                worst = (float(g[k]), f"joint ({i},{j})-({i + di},{j + dj}), {lab[k]}")
    return joints, worst


def array_fit():
    from scipy.optimize import brentq, minimize
    s = math.radians(K.splay_deg)
    # corner tilt so that the corner sits at the full splay from each edge neighbour
    ez = np.array([math.sin(s), 0, math.cos(s)])

    def ang(ph):
        cz = _Rax((-1, 1, 0), ph)[:, 2]
        return math.acos(float(np.clip(cz @ ez, -1, 1))) - s
    ph = brentq(ang, s * 0.9, 2.5 * s)

    W = np.array([[0.0, 1, 0], [1, 0, 0], [0, 0, 1]])

    def ev(x):
        I, Pe, Pc = _layout(s, ph, x)
        d1, d2, sep = _joint(Pe, "+y", Pc, "-y")
        Pe2 = _mirror(Pe, W)
        con = np.r_[d1, d2, _inside(I, Pc), _inside(Pc, I), _inside(Pe, Pe2), _inside(Pe2, Pe)]
        return float(sep.max()), float(con.min())

    # start: corner rotated about the outer apex like the edges, then a grid over its two free coordinates
    apex = K.outer_half_width(K.mouth_z) / math.tan(A) - K.mouth_z
    c = np.array([0, 0, -apex])
    t0 = c - _Rax((-1, 1, 0), ph) @ c
    best = None
    for t in np.linspace(t0[0] - 0.3, t0[0] + 0.3, 121):
        for tz in np.linspace(t0[2] - 0.3, t0[2] + 0.3, 121):
            o, cmin = ev((t, tz))
            if cmin >= -1e-9 and (best is None or o < best[0]):
                best = (o, t, tz)
    if best is None:
        raise RuntimeError("no overlap-free corner position found")
    r = minimize(lambda x: ev(x)[0] + 1e3 * max(0.0, -ev(x)[1] - 1e-9), [best[1], best[2]], method="Nelder-Mead",
                 options=dict(xatol=1e-7, fatol=1e-9, maxiter=4000))

    def unpack(x):
        return _layout(s, ph, x)

    I, Pe, Pc = unpack(r.x[:2])
    boxes = _all_boxes(I, Pe, Pc)
    joints, worst = _report(boxes)
    # gnomonic layout from sunflower.geometry for comparison
    g = array_layout(K)
    gb = {}
    for b in g["boxes"]:
        Mx = np.array(b["matrix"])
        gb[(b["i"], -b["j"])] = (Mx[:3, :3], Mx[:3, 3])
    gj, gw = _report(gb)
    pen = max(j["penetration_mm"] for j in gj)
    out = dict(
        splay_deg=K.splay_deg, boxes_n=9,
        apex_behind_baffle_mm=round(apex * MM, 1),
        corner_tilt_deg=round(math.degrees(ph), 3),
        frame="Each matrix maps a box's own frame (metres; origin at the centre of its baffle front face, +z to the "
              "listener, +x right, +y up) to the array frame, which is the centre box's own frame.",
        method="Edge boxes hinge on the centre box's mouth front edges at the full splay, so those four joints are "
               "flush. Each corner box is tilted towards the diagonal so its axis is the full splay from both edge "
               "neighbours; its position is then solved (SLSQP) to bring its mouth front edges as close as possible "
               "to the edge boxes' without any overlap. Square frustums cannot also close the four inner vertices: "
               "the corner box is twisted against its neighbours, so the edge to corner joints open as a wedge.",
        optimiser=dict(success=bool(r.success), message=str(r.message),
                       max_mouth_edge_offset_mm=round(ev(r.x)[0] * MM, 1)),
        corner_joint_twist_deg=round(_twist(Pe, Pc), 2),
        max_gap_mm=round(worst[0], 1), max_gap_where=worst[1],
        max_penetration_mm=max(j["penetration_mm"] for j in joints),
        joints=joints,
        gnomonic=dict(note="geometry.array_layout: same splay at the four edge joints, but the corner boxes sit too "
                           "close and overlap their edge neighbours", max_gap_mm=round(gw[0], 1),
                      max_penetration_mm=pen),
        boxes=[dict(i=i, j=j, matrix=np.vstack([np.c_[P[0], P[1]], [0, 0, 0, 1]]).round(6).tolist())
               for (i, j), P in sorted(boxes.items(), key=lambda kv: (-kv[0][1], kv[0][0]))],
    )
    return out, boxes


def array_overlap(boxes):
    """Intersect the nine outer frustums in OCC and return the largest overlap volume (cm3)."""
    global occ
    gmsh.clear()
    gmsh.model.add("array")
    occ = gmsh.model.occ
    zb, zm = Z_BACK, Z_M
    tags = {}
    for key, P in boxes.items():
        v = _tr((P[0], P[1] * MM), np.array(square(ohw(zb), zb) + square(ohw(zm), zm)))
        a, b = [tuple(p) for p in v[:4]], [tuple(p) for p in v[4:]]
        tags[key] = hexa(a, b)
    worst = 0.0
    keys = list(boxes)
    for n, a in enumerate(keys):
        for b in keys[n + 1:]:
            if max(abs(a[0] - b[0]), abs(a[1] - b[1])) != 1:
                continue
            res, _ = occ.intersect(occ.copy(tags[a]), occ.copy(tags[b]))
            v = sum(occ.getMass(*e) for e in res if e[0] == 3)
            worst = max(worst, v / 1000)
            if res:
                occ.remove(res, recursive=True)
    return worst


def _box_solid(P):
    zb, zm = Z_BACK, Z_M
    v = _tr((P[0], P[1] * MM), np.array(square(ohw(zb), zb) + square(ohw(zm), zm)))
    return hexa([tuple(p) for p in v[:4]], [tuple(p) for p in v[4:]])


def _face_prism(P, w, depth):
    """The outer face of wall w of box P, swept outward by depth (mm), in the array frame."""
    f, n = _face(w)
    f = _tr((P[0], P[1] * MM), f * MM)[[0, 1, 3, 2]]
    n = P[0] @ n
    return hexa([tuple(p) for p in f], [tuple(p + depth * n) for p in f])


def array_spacers(boxes):
    """Tapered wedges that close the edge to corner joints: the space between the two facing walls, where the
    faces overlap, outside every box. Each is written to cad/step and cad/stl in the array frame."""
    global occ
    out, tris = [], []
    for (i, j), P in sorted(boxes.items()):
        for di, dj, wp, wq in ((1, 0, "+x", "-x"), (0, 1, "+y", "-y")):
            key = (i + di, j + dj)
            if key not in boxes or (i == 0 and j == 0) or (key == (0, 0)):
                continue
            if (i == 0) == (key[0] == 0) and (j == 0) == (key[1] == 0):
                continue
            Q = boxes[key]
            d1, d2, _ = _joint(P, wp, Q, wq)
            if max(d1.max(), d2.max()) * MM < 0.5:
                continue                      # flush joint, no spacer
            name = f"array_spacer_{i}_{j}__{key[0]}_{key[1]}".replace("-", "m")
            start(name)
            a = _face_prism(P, wp, 300.0)
            b = _face_prism(Q, wq, 300.0)
            sp, _ = occ.intersect(a, b)
            others = []
            for k2, B in boxes.items():
                if max(abs(k2[0] - i), abs(k2[1] - j)) <= 1 or max(abs(k2[0] - key[0]), abs(k2[1] - key[1])) <= 1:
                    others += _box_solid(B)
            occ.cut(sp, others)
            clean()
            vol = sum(occ.getMass(*e) for e in gmsh.model.getEntities(3))
            quiet_write(OUT / "step" / f"{name}.step")
            xyz = mesh_write(OUT / "stl" / f"{name}.stl", size=200.0)
            tris.append(_read_stl(OUT / "stl" / f"{name}.stl"))
            g = np.maximum(d1, d2) * MM
            f, _ = _face(wp)
            v = _tr(P, f)
            rad = np.hypot(v[:, 0], v[:, 1])
            lab = [("back" if k < 2 else "mouth") + ("_inner" if rad[k] < rad[k ^ 1] else "_outer") for k in range(4)]
            out.append(dict(name=name, between=[[i, j], list(key)], walls=[wp, wq],
                            thickness_mm={lab[k]: round(float(g[k]), 1) for k in range(4)},
                            max_thickness_mm=round(float(g.max()), 1), min_thickness_mm=round(float(g.min()), 1),
                            volume_litres=round(vol / 1e6, 2), bbox_mm=[round(float(v), 1) for v in [*xyz.min(0), *xyz.max(0)]],
                            files=dict(step=f"step/{name}.step", stl=f"stl/{name}.stl")))
    return out, (np.concatenate(tris) if tris else np.zeros((0, 3, 3)))


def _read_stl(path):
    b = path.read_bytes()
    n = struct.unpack("<I", b[80:84])[0]
    dt = np.dtype([("n", "<3f4"), ("v", "<9f4"), ("a", "<u2")])
    return np.frombuffer(b[84:84 + n * 50], dtype=dt)["v"].reshape(-1, 3, 3).astype(float)


def coarse_triangles(names, size=40.0):
    """Coarse surface mesh (mm) of the named parts, for the array preview."""
    global occ
    gmsh.clear()
    gmsh.model.add("coarse")
    occ = gmsh.model.occ
    for n in names:
        occ.importShapes(str(OUT / "step" / f"{n}.step"))
    occ.synchronize()
    gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", 0)
    gmsh.option.setNumber("Mesh.MeshSizeExtendFromBoundary", 0)
    gmsh.option.setNumber("Mesh.MeshSizeMax", size)
    gmsh.option.setNumber("Mesh.MeshSizeMin", size / 4)
    gmsh.model.mesh.generate(2)
    tags, xyz, _ = gmsh.model.mesh.getNodes()
    xyz = xyz.reshape(-1, 3)
    index = np.zeros(int(tags.max()) + 1, dtype=int)
    index[tags.astype(int)] = np.arange(len(tags))
    _, nodes = gmsh.model.mesh.getElementsByType(2)
    return xyz[index[nodes.astype(int)]].reshape(-1, 3, 3)


def write_array_stl(boxes, tri, extra=None):
    out = [] if extra is None else [extra]
    for P in boxes.values():
        v = tri.reshape(-1, 3) @ P[0].T + P[1] * MM
        out.append(v.reshape(-1, 3, 3))
    allv = np.concatenate(out).astype("<f4")
    nrm = np.cross(allv[:, 1] - allv[:, 0], allv[:, 2] - allv[:, 0])
    nrm /= np.maximum(np.linalg.norm(nrm, axis=1, keepdims=True), 1e-12)
    rec = np.zeros(len(allv), dtype=np.dtype([("n", "<3f4"), ("v", "<9f4"), ("a", "<u2")]))
    rec["n"] = nrm
    rec["v"] = allv.reshape(-1, 9)
    with open(OUT / "array.stl", "wb") as f:
        f.write(b"Sunflower 3 x 3 array, coarse preview, mm".ljust(80, b" "))
        f.write(struct.pack("<I", len(rec)))
        f.write(rec.tobytes())
    return len(rec)


# ---------------------------------------------------------------- outputs
def dxf(mitre, baf, back):
    import ezdxf
    (OUT / "dxf").mkdir(exist_ok=True)

    def doc_with(label):
        doc = ezdxf.new("R2010")
        doc.layers.add("CUT")
        doc.layers.add("INNER_FACE", linetype="DASHED")
        doc.layers.add("TEXT")
        return doc, doc.modelspace()

    # wall: outer face outline (cut) and the inner face (after the bevel)
    L = (Z_M - Z_WB) / math.cos(A)
    off = T * math.tan(A)
    wo0, wo1 = 2 * ohw(Z_WB), 2 * ohw(Z_M)
    wi0, wi1 = 2 * hw(Z_WB), 2 * hw(Z_M)
    doc, msp = doc_with("wall")
    msp.add_lwpolyline([(-wo0 / 2, 0), (wo0 / 2, 0), (wo1 / 2, L), (-wo1 / 2, L)], close=True, dxfattribs={"layer": "CUT"})
    msp.add_lwpolyline([(-wi0 / 2, off), (wi0 / 2, off), (wi1 / 2, L + off), (-wi1 / 2, L + off)], close=True,
                       dxfattribs={"layer": "INNER_FACE"})
    msp.add_text(f"Wall x4, 18 mm birch, outer face shown. Long edges: blade bevel {mitre['blade_bevel_deg']:.1f} deg, "
                 f"edge taper {mitre['edge_taper_deg']:.1f} deg. Ends bevelled {mitre['end_bevel_deg']:.1f} deg, parallel.",
                 dxfattribs={"height": 10, "layer": "TEXT"}).set_placement((-wo1 / 2, -30))
    doc.saveas(OUT / "dxf" / "wall.dxf")
    doc, msp = doc_with("baffle")
    a, b = baf["front_square_mm"] / 2, baf["back_square_mm"] / 2
    msp.add_lwpolyline([(-a, -a), (a, -a), (a, a), (-a, a)], close=True, dxfattribs={"layer": "CUT"})
    msp.add_lwpolyline([(-b, -b), (b, -b), (b, b), (-b, b)], close=True, dxfattribs={"layer": "INNER_FACE"})
    msp.add_circle((0, 0), baf["cutout_diameter_mm"] / 2, dxfattribs={"layer": "CUT"})
    for k in range(8):
        az = math.radians(22.5 + 45 * k)
        msp.add_circle((baf["flange_pcd_mm"] / 2 * math.cos(az), baf["flange_pcd_mm"] / 2 * math.sin(az)), 3.5,
                       dxfattribs={"layer": "CUT"})
    for i, az in enumerate(STANDOFF_AZ):
        msp.add_circle((R_STANDOFF * math.cos(az), R_STANDOFF * math.sin(az)), 6.0 if i == DUCT else 4.25,
                       dxfattribs={"layer": "CUT"})
    msp.add_text(f"Baffle, 25 mm birch, front face shown (back face dashed). Edges bevelled {K.half_angle_deg:.1f} deg.",
                 dxfattribs={"height": 10, "layer": "TEXT"}).set_placement((-a, -a - 30))
    doc.saveas(OUT / "dxf" / "baffle.dxf")
    doc, msp = doc_with("back")
    a, b = back["front_square_mm"] / 2, back["back_square_mm"] / 2
    msp.add_lwpolyline([(-a, -a), (a, -a), (a, a), (-a, a)], close=True, dxfattribs={"layer": "CUT"})
    msp.add_lwpolyline([(-b, -b), (b, -b), (b, b), (-b, b)], close=True, dxfattribs={"layer": "INNER_FACE"})
    msp.add_circle((0, 0), 32, dxfattribs={"layer": "CUT"})
    h = hw(Z_WB)
    for g in (h + 2, h + 9):
        msp.add_lwpolyline([(-g, -g), (g, -g), (g, g), (-g, g)], close=True, dxfattribs={"layer": "INNER_FACE"})
    rs = h + 13.5
    for x, y in [(rs, t * h) for t in (-0.6, 0, 0.6)] + [(-rs, t * h) for t in (-0.6, 0, 0.6)] + \
                [(t * h, rs) for t in (-0.6, 0, 0.6)] + [(t * h, -rs) for t in (-0.6, 0, 0.6)]:
        msp.add_circle((x, y), 2.75, dxfattribs={"layer": "CUT"})
    msp.add_text(f"Back panel, 18 mm birch, front face shown. Edges bevelled {K.half_angle_deg:.1f} deg. "
                 f"Gasket groove 7 x 2 mm.", dxfattribs={"height": 10, "layer": "TEXT"}).set_placement((-a, -a - 30))
    doc.saveas(OUT / "dxf" / "back.dxf")
    for old in ("side.dxf", "horn_side_wall.dxf", "horn_top_wall.dxf", "horn_wall_inner_face.dxf"):
        (OUT / "dxf" / old).unlink(missing_ok=True)


def assembly():
    start("assembly")
    for p in parts:
        occ.importShapes(str(OUT / "step" / f"{p['name']}.step"))
    occ.synchronize()
    quiet_write(OUT / "sunflower-assembly.step")


def main():
    t0 = time.time()
    for sub in ("step", "stl", "print", "dxf"):
        (OUT / sub).mkdir(parents=True, exist_ok=True)
    for sub in ("step", "stl", "print"):
        for f in (OUT / sub).glob("*"):
            f.unlink()
    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 0)
    gmsh.option.setString("Geometry.OCCTargetUnit", "MM")
    print("cabinet", flush=True)
    walls()
    baf = baffle()
    back = back_panel()
    fill = fillers()
    print("supports", flush=True)
    supports()
    print("centre body", flush=True)
    body = centre_body()
    print("bullet", flush=True)
    bullet()
    print("drivers", flush=True)
    drivers()
    assembly()
    print("area check", flush=True)
    area = area_check()
    print("array", flush=True)
    arr, boxes = array_fit()
    arr["max_overlap_cm3_occ"] = round(array_overlap(boxes), 3)
    spacers, sp_tri = array_spacers(boxes)
    arr["spacers"] = spacers
    arr["spacer_note"] = ("Tapered wedges, part of the rigging frame, that close the edge to corner joints. Each is the "
                          "space between the two facing walls where they overlap, so both boxes sit flush on it. "
                          "Coordinates are the array frame in mm (the centre box's own frame).")
    tri = coarse_triangles([p["name"] for p in parts if p["category"] in ("cabinet", "filler", "print", "support")])
    gmsh.finalize()
    arr["preview"] = dict(file="array.stl", triangles=write_array_stl(boxes, tri, sp_tri),
                          note="coarse mesh of the cabinet, fillers, supports and printed parts of each box, placed, "
                               "plus the array spacers")
    mitre = mitre_angles()
    dxf(mitre, baf, back)
    ext = dict(width=round(2 * ohw(Z_M), 1), height_front=round(2 * ohw(Z_M), 1), height_back=round(2 * ohw(Z_BACK), 1),
               depth=round(Z_M - Z_BACK, 1), overall_depth=round(max(p[1] for p in lip_arc()[0]) - Z_BACK, 1))
    L = (Z_M - Z_WB) / math.cos(A)
    cut = dict(
        units="mm",
        external_mm=ext,
        walls=dict(qty=4, ply_mm=T, slant_length_mm=round(L, 1),
                   inner_face=dict(back_edge=round(2 * hw(Z_WB), 1), mouth_edge=round(2 * hw(Z_M), 1)),
                   outer_face=dict(back_edge=round(2 * ohw(Z_WB), 1), mouth_edge=round(2 * ohw(Z_M), 1)),
                   **mitre, dxf="dxf/wall.dxf",
                   note="Four identical panels. Cut the outer face to size, tilt the blade to the bevel along both "
                        "long edges so the inner face is the narrower one, and bevel both ends at the half angle so "
                        "the back and the mouth are square to the axis."),
        baffle=dict(**baf, z_mm=[round(Z_BAF, 1), 0.0], dxf="dxf/baffle.dxf",
                    note="Glued inside the frustum; the kick flange screws to its front face"),
        back=dict(**back, z_mm=[round(Z_BACK, 1), round(Z_WB, 1)], dxf="dxf/back.dxf"),
        corner_filler=dict(qty=4, **fill),
        chamber=dict(available_litres=round(K.chamber_litres, 2),
                     frustum_litres=round(float(np.trapezoid([(2 * hw(z)) ** 2 for z in np.linspace(Z_WB, Z_BAF, 50)],
                                                          np.linspace(Z_WB, Z_BAF, 50))) / 1e6, 2),
                     note="Kick.chamber_litres: frustum between the baffle and the back, the baffle cutout, the cone"),
        supports=dict(standoffs=dict(qty=4, diameter_mm=2 * STANDOFF_R, pcd_mm=round(2 * R_STANDOFF, 1),
                                     length_mm=round(Z_TURN + SOCKET, 1), duct_bore_mm=12,
                                     azimuth_deg=[45, 135, 225, 315]),
                      vanes=dict(qty=4, thickness_mm=VANE_T, chord_mm=VANE_CHORD, z_centre_mm=round(Z_VANE, 1))),
        centre_body=body,
        stations_mm=dict(back=round(Z_BACK, 1), wall_back=round(Z_WB, 1), baffle_back=round(Z_BAF, 1), baffle_front=0.0, throat=0.0, turn=round(Z_TURN, 1),
                         motor_end=round(Z_MOTOR, 1), mid_driver_back=round(Z_MF - MD.depth * MM, 1),
                         mid_flange=round(Z_MF, 1), mid_throat=round(Z_MT, 1), hf_exit=round(zs(M.s_exit), 1),
                         hf_mouth=round(zs(M.s_tip), 1), mouth=round(Z_M, 1), mid_mouth=round(Z_MM, 1),
                         vanes=round(Z_VANE, 1)),
        array_spacers=dict(
            qty=len(spacers), max_thickness_mm=max((x["max_thickness_mm"] for x in spacers), default=0.0),
            min_thickness_mm=min((x["min_thickness_mm"] for x in spacers), default=0.0),
            pieces=[dict(name=x["name"], between=x["between"], max_thickness_mm=x["max_thickness_mm"],
                         min_thickness_mm=x["min_thickness_mm"], volume_litres=x["volume_litres"]) for x in spacers],
            note="Rigging frame, not part of a single box. Each spacer is a twisted wedge between the facing walls of "
                 "an edge box and a corner box. Make it as an 18 mm birch ply frame (a perimeter of strips "
                 "planed to the taper, with a ply skin on each face) or as a solid laminated wedge; four are mirror "
                 "images of the other four. STEP and STL in the array frame: cad/step and cad/stl/array_spacer_*."),
        area_check=area,
        print_files={p["name"]: p.get("print_files", []) for p in parts if p.get("print_files")},
        warnings=warnings,
    )
    stack = dict(splay_deg=K.splay_deg, boxes=3,
                 note="Three boxes in a vertical column, each turned by the splay about the mouth front edge it "
                      "shares with its neighbour. See array.json for the 3 x 3 array.",
                 pivot_mm=dict(y=round(ohw(Z_M), 1), z=round(Z_M, 1)))
    (OUT / "parts.json").write_text(json.dumps(parts, indent=1) + "\n")
    (OUT / "cutlist.json").write_text(json.dumps(cut, indent=1) + "\n")
    (OUT / "stack.json").write_text(json.dumps(stack, indent=1) + "\n")
    (OUT / "array.json").write_text(json.dumps(arr, indent=1) + "\n")
    for f in TMP.glob("*"):
        f.unlink()
    TMP.rmdir()
    print(json.dumps(dict(area_check=area, array=dict(max_gap_mm=arr["max_gap_mm"], where=arr["max_gap_where"],
                                                      overlap_cm3=arr["max_overlap_cm3_occ"]),
                          pieces=sum(len(p.get("print_files", [])) for p in parts), warnings=warnings), indent=1))
    print(f"done in {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
