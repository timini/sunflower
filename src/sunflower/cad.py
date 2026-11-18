"""Build every Sunflower part as STEP and STL (millimetres) from sunflower.design.

Coordinates follow sunflower.geometry, in millimetres: z = 0 is the front face
of the baffle (where the reversed kick flange sits), z is positive towards the
listener, x is to the right when facing the cabinet, y is up.

The box is single-ply 18 mm birch: two side walls angled at the half angle,
flat top and bottom panels, a rectangular baffle and back. Tapered wedges under
the top and over the bottom bring the horn to a square frustum, and four corner
fillers make the throat an octagon.

The two phase plugs are sized from slices of the model itself. Every part
except the kick centre body's outer surface and the mid bullet is built first.
At each z the open area inside the walls, less every part, is measured from the
part meshes (sunflower.slicer), and the plug radius is solved so the air left
is the horn law. The shroud over the reversed kick cone is sized the same way
from cylinder slices through the cone and the basket. The assembled model is
then sliced again and the plugs corrected until each slab matches its law.

Outputs:
  cad/step/<part>.step, cad/stl/<part>.stl   one file per part, in place
  cad/print/<part>-<n>[-s<k>].stl             printable pieces (z slab n, sector k), each fits a 250 mm bed
  cad/sunflower-assembly.step                 everything in place
  cad/parts.json                              part list: material, category, explode vector, bbox, print files
  cad/cutlist.json                            panels, bevels, wedges, fillers, chamber, stations
  cad/plug_profiles.json                      slice-solved centre body, bullet and shroud profiles
  cad/stack.json                              a single column (renderer)
  cad/array.json, cad/array.stl               3 x 3 array: placement of each box, joint check
  cad/dxf/<panel>.dxf                         flat patterns
  results/cad/slices.csv                      the assembled model sliced every 1 cm, against each law
  results/cad/area-check.csv                  the same, in the older long format (plots, site)
  results/cad/cone-channel.csv                open area over the reversed kick cone on cylinders r = const

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
import trimesh
from scipy.interpolate import PchipInterpolator
from scipy.optimize import brentq
from shapely.geometry import Point
from shapely.ops import unary_union

from sunflower import design as D
from sunflower import slicer as SL
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
E = T / math.cos(A)                                # horizontal thickness of an 18 mm panel at the half angle
HH = K.half_height * MM                            # inside half-height (flat top and bottom)

# ---------------------------------------------------------------- stations (mm)
Z_BAF = -K.baffle * MM                             # baffle back face
Z_WB = -(K.baffle + K.chamber_depth) * MM          # wall back ends, back panel front face
Z_BACK = Z_WB - T                                  # back panel back face
Z_M = K.mouth_z * MM                               # kick mouth (and mid mouth: no protrusion)
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


def s_of(z):
    return (z - Z_MT) / MM


# centre body radii (mm)
R_CS = K.channel_start * MM                        # inner edge of the shroud
R_MC = (KD.motor_r + K.motor_clear) * MM           # motor cavity
R_POD = MD.frame_od / 2 * MM + 0.5                 # mid pod cavity (frame plus 0.5 mm)
Z_CUT = KD.frame_od / 2 * MM - 17                  # baffle cutout radius (as in Kick.chamber_litres)

# supports
STANDOFF_R = 10.0
STANDOFF_AZ = [math.radians(45 + 90 * k) for k in range(4)]
DUCT = 0                                           # index of the hollow standoff (cable duct)
R_STANDOFF = KD.frame_od / 2 * MM + 12             # just outside the kick flange
SOCKET = 25.0                                      # standoff depth into the shroud
VANE_T, VANE_CHORD = 6.0, 60.0
Z_VANE = Z_M - 100.0
# Bullet vanes: in front of the HF unit, where the bullet has room to slim down for them. At the HF body
# (s = 5 to 51 mm) the bullet cannot give up their area and still hold the body cylinder.
BV_T, BV_S0, BV_S1 = 4.0, 0.058, 0.090             # thickness mm, s range m
BV_AZ = [math.radians(45 + 120 * k) for k in range(3)]
S_HF_CABLE = 0.5 * (BV_S0 + BV_S1)
R_CABLE_B = round(0.5 * (M.plug_radius + M.hf_horn_radius(S_HF_CABLE) + M.hf_wall) * MM, 1)   # HF cable bore in the
                                                   # bullet wall, midway between the HF horn wall and the bullet;
                                                   # runs back to the HF cavity
assert BV_S0 > M.s_exit + 0.005, "bullet vanes must sit in front of the HF unit"
R_HF_CH = R_POD + 20                               # HF cable channel radius in the body
Z_CAB = Z_MOTOR + 30                               # duct channel enters the pod here
SPIGOT_H = 6.0
WEDGE_RIBS = (-0.6, 0.0, 0.6)                      # rib positions as a fraction of the baffle half-width

DZ_SOLVE = 2.5                                     # plug solve station spacing, mm
occ = None
parts: list[dict] = []
warnings: list[str] = []
FINAL = False                                      # print pieces only on the final pass


# ---------------------------------------------------------------- profiles
class Prof:
    """Piecewise cubic (PCHIP) spline through solved stations: C1, no overshoot."""

    def __init__(self, x, y):
        self.x, self.y = np.asarray(x, float), np.asarray(y, float)
        self.f = PchipInterpolator(self.x, self.y)

    def __call__(self, v):
        return float(self.f(min(max(v, self.x[0]), self.x[-1])))


def _grid(a, b, d):
    n = max(2, int(math.ceil((b - a) / d)) + 1)
    return np.linspace(a, b, n)


P = dict(
    body=Prof(_grid(Z_TURN, Z_M, DZ_SOLVE), [K.body_radius(z / MM) * MM for z in _grid(Z_TURN, Z_M, DZ_SOLVE)]),
    bullet=Prof(_grid(0.0, M.s_tip * MM, DZ_SOLVE), [M.plug(s / MM) * MM for s in _grid(0.0, M.s_tip * MM, DZ_SOLVE)]),
    shroud=Prof(_grid(R_CS, KD.rim_r * MM, 3.0), [float(K.shroud_z(r / MM)) * MM for r in _grid(R_CS, KD.rim_r * MM, 3.0)]),
)


def body_r(z):
    return P["body"](z)


def bullet_r(s):
    """Bullet outer radius (mm) at mid horn coordinate s (m)."""
    return P["bullet"](s * MM) if 0 <= s <= M.s_tip + 1e-9 else 0.0


def shroud_z(r):
    """Shroud underside (mm) at radius r (mm)."""
    return P["shroud"](r)


def j1():
    """Shroud to fairing joint."""
    return max(KD.top_plate_z * MM, shroud_z(R_CS) + 10)


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

    def Pt(p):
        if p not in tags:
            tags[p] = occ.addPoint(p[0], 0, p[1])
        return tags[p]

    curves = []
    for kind, pts in pieces:
        ids = [Pt(p) for p in pts]
        if kind == "spline" and len(ids) > 2:
            curves.append(occ.addSpline(ids))
        else:
            curves += [occ.addLine(ids[k], ids[k + 1]) for k in range(len(ids) - 1) if ids[k] != ids[k + 1]]
    surf = occ.addPlaneSurface([occ.addCurveLoop(curves)])
    out = occ.revolve([(2, surf)], 0, 0, 0, 0, 0, 1, angle)
    return [e for e in out if e[0] == 3]


def poly_volume(faces_pts):
    cache, edges = {}, {}

    def Pt(p):
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
        ids = [Pt(p) for p in face]
        ids = [i for k, i in enumerate(ids) if i != ids[k - 1]]
        if len(ids) < 3:
            continue
        loop = [L(ids[i], ids[(i + 1) % len(ids)]) for i in range(len(ids))]
        surfs.append(occ.addPlaneSurface([occ.addCurveLoop(loop)]))
    return [(3, occ.addVolume([occ.addSurfaceLoop(surfs)]))]


def hexa(a, b):
    """Solid between two matching quads a (one end) and b (other end), each 4 xyz points in order."""
    faces = [a, b[::-1]] + [[a[i], b[i], b[(i + 1) % 4], a[(i + 1) % 4]] for i in range(4)]
    return poly_volume(faces)


def square(h, z):
    return [(-h, -h, z), (h, -h, z), (h, h, z), (-h, h, z)]


def rect(hx, hy, z):
    return [(-hx, -hy, z), (hx, -hy, z), (hx, hy, z), (-hx, hy, z)]


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


def mesh_write(path, size=12.0, curv=120, smin=0.5):
    """Surface mesh to STL. ``curv`` elements per full turn of a curved edge: 120 keeps the area error of a
    sliced circle under 0.05 %."""
    gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", curv)
    gmsh.option.setNumber("Mesh.MeshSizeMax", size)
    gmsh.option.setNumber("Mesh.MeshSizeMin", smin)
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
    return gmsh.model.mesh.getNodesByElementType(2)[1].reshape(-1, 3)     # nodes of the written triangles only


def save(name, material, category, explode=(0, 0, 0), group="", note="", printable=False, qty=1, curv=120,
         smin=0.5, size=12.0):
    clean()
    gmsh.model.removePhysicalGroups()
    quiet_write(OUT / "step" / f"{name}.step")
    vol = sum(occ.getMass(*d) for d in gmsh.model.getEntities(3))
    xyz = mesh_write(OUT / "stl" / f"{name}.stl", size=size, curv=curv, smin=smin)
    bb = [*xyz.min(0), *xyz.max(0)]
    rec = dict(name=name, material=material, category=category, qty=qty, explode_mm=[float(v) for v in explode],
               group=group, volume_cm3=round(vol / 1000, 1), bbox_mm=[round(float(v), 1) for v in bb], note=note)
    if printable and FINAL:
        rec["print_files"] = split_for_print(name, bb)
    for i, p in enumerate(parts):
        if p["name"] == name:
            parts[i] = rec
            break
    else:
        parts.append(rec)
    print(f"  {name}: {rec['volume_cm3']} cm3" + (f", {len(rec['print_files'])} print pieces"
                                                  if rec.get("print_files") else ""), flush=True)
    return rec


def _fits(xyz):
    s = np.sort(xyz.max(0) - xyz.min(0))
    return s[0] <= BED and s[1] <= BED and s[2] <= 1.3 * BED


def _sector_fits(rmax, ns, h):
    """Quick bound: can an annulus sector of outer radius rmax, ns sectors, height h fit at all?"""
    if ns == 1:
        dims = [2 * rmax, 2 * rmax, h]
    elif ns == 2:
        dims = [2 * rmax, rmax, h]
    else:
        a = 2 * math.pi / ns
        dims = [rmax, rmax * math.sin(a) if a < math.pi / 2 else rmax, h]
    s = sorted(dims)
    return s[0] <= BED and s[1] <= BED and s[2] <= 1.3 * BED


def split_for_print(name, bb):
    """Cut a part into z slabs of at most SLAB mm, then into equal sectors about the axis until every piece fits.

    Sectors start at azimuth 0, so with 8 or more each piece lies inside one
    octant. Pieces stay in assembly coordinates.
    """
    z0, z1 = bb[2], bb[5]
    if _fits(np.array([bb[:3], bb[3:]])) and z1 - z0 <= SLAB:
        fn = f"{name}-1.stl"                       # fits whole: print the part as it is
        (OUT / "print" / fn).write_bytes((OUT / "stl" / f"{name}.stl").read_bytes())
        return [fn]
    nz = max(1, math.ceil((z1 - z0 - 0.5) / SLAB))
    big = 2 * max(abs(v) for v in bb) + 100
    rmax = max(abs(v) for v in (bb[0], bb[1], bb[3], bb[4]))      # parts split this way are bodies of revolution
    files = []
    for ns in (1, 2, 4, 8, 12, 16, 24):
        if ns > 1 and not _sector_fits(rmax, ns, (z1 - z0) / nz) and ns < 24:
            continue
        ok = True
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
                    xyz = mesh_write(OUT / "print" / fn, size=8.0, curv=48)
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
def walls():
    """Side walls at the half angle, full inside height; flat top and bottom over them, full depth."""
    for side, sx in (("right", 1), ("left", -1)):
        start(f"wall_{side}")
        q = []
        for z in (Z_WB, Z_M):
            q.append([(sx * hw(z), -HH, z), (sx * hw(z), HH, z), (sx * ohw(z), HH, z), (sx * ohw(z), -HH, z)])
        hexa(*q)
        save(f"wall_{side}", "ply", "cabinet", explode=(sx * 350, 0, 0), group="cabinet",
             note=f"18 mm birch rectangle, {2*HH:.0f} high, {(Z_M - Z_WB)/math.cos(A):.0f} long on the slant; "
                  f"both ends bevelled {K.half_angle_deg:.1f} deg, long edges square (they butt under the top "
                  f"and over the bottom)")
    for name, sy in (("top", 1), ("bottom", -1)):
        start(name)
        q = []
        for z in (Z_WB, Z_M):
            q.append([(-ohw(z), sy * HH, z), (ohw(z), sy * HH, z), (ohw(z), sy * (HH + T), z), (-ohw(z), sy * (HH + T), z)])
        hexa(*q)
        save(name, "ply", "cabinet", explode=(0, sy * 350, 0), group="cabinet",
             note=f"18 mm birch, flat, trapezoid {2*ohw(Z_WB):.0f} to {2*ohw(Z_M):.0f} mm over "
                  f"{Z_M - Z_WB:.0f} mm; square edges (the side walls' outer faces run on from them)")


def wedge_solid(sy):
    """The whole wedge region: from the sloping horn face y = sy hw(z) to the flat panel, side wall to side wall."""
    h0, zm = hw(0.0), Z_M
    a = [(-h0, sy * h0, 0.0), (h0, sy * h0, 0.0), (h0, sy * HH, 0.0), (-h0, sy * HH, 0.0)]
    tip = [(-hw(zm), sy * HH, zm), (hw(zm), sy * HH, zm)]
    faces = [a, [a[0], a[1], tip[1], tip[0]], [a[3], a[2], tip[1], tip[0]], [a[1], a[2], tip[1]], [a[0], a[3], tip[0]]]
    return poly_volume(faces)


def wedge_cavity(sy):
    """Air sealed inside the wedge: behind the 18 mm sloping panel, inside the 18 mm side closures."""
    h0 = hw(0.0)
    ze = Z_M - E / math.tan(A)                      # the panel's back face reaches the flat panel here
    a = [(-(h0 - E), sy * (h0 + E), 0.0), ((h0 - E), sy * (h0 + E), 0.0), ((h0 - E), sy * HH, 0.0),
         (-(h0 - E), sy * HH, 0.0)]
    tip = [(-(hw(ze) - E), sy * HH, ze), ((hw(ze) - E), sy * HH, ze)]
    faces = [a, [a[0], a[1], tip[1], tip[0]], [a[3], a[2], tip[1], tip[0]], [a[1], a[2], tip[1]], [a[0], a[3], tip[0]]]
    return poly_volume(faces), ze


def wedges():
    out = {}
    for name, sy in (("wedge_top", 1), ("wedge_bottom", -1)):
        start(name)
        w = wedge_solid(sy)
        cav, ze = wedge_cavity(sy)
        ribs = [(3, occ.addBox(f * hw(0.0) - T / 2, -HH - 10, -10, T, 2 * HH + 20, Z_M + 20)) for f in WEDGE_RIBS]
        cells, _ = occ.cut(cav, ribs)
        occ.cut(w, cells)
        save(name, "ply", "cabinet", explode=(0, sy * 240, 120), group="cabinet",
             note=f"tapered wedge {'under the top' if sy > 0 else 'over the bottom'}: 18 mm sloping panel at the half "
                  f"angle (its face is the horn wall), 18 mm side closures against the side walls and "
                  f"{len(WEDGE_RIBS)} ribs, glued to the flat panel and the baffle front face and sealed; "
                  f"{2 * (HH - hw(0)):.0f} mm deep at the baffle, nothing at the mouth")
        out[name] = ze
    return out


def baffle():
    start("baffle")
    b = hexa(rect(hw(Z_BAF), HH, Z_BAF), rect(hw(0.0), HH, 0.0))
    holes = [(3, occ.addCylinder(0, 0, Z_BAF - 1, 0, 0, -Z_BAF + 2, Z_CUT))]
    pcd = KD.frame_od / 2 * MM - 9
    for k in range(8):
        az = math.radians(22.5 + 45 * k)
        holes.append(cyl_at(pcd, az, Z_BAF - 1, 1, 3.5))
    for i, az in enumerate(STANDOFF_AZ):
        holes.append(cyl_at(R_STANDOFF, az, Z_BAF - 1, 1, 6.0 if i == DUCT else 4.25))
    occ.cut(b, holes)
    save("baffle", "ply", "cabinet", explode=(0, 0, -260), group="cabinet",
         note=f"25 mm birch rectangle, {2*hw(0):.0f} wide at the front face by {2*HH:.0f} high, side edges "
              f"bevelled {K.half_angle_deg:.1f} deg, {2*Z_CUT:.0f} mm cutout, 8 flange holes on {2*pcd:.0f} PCD, "
              f"4 standoff bolts (one 12 mm duct hole)")
    return dict(front_width_mm=round(2 * hw(0), 1), back_width_mm=round(2 * hw(Z_BAF), 1), height_mm=round(2 * HH, 1),
                front_square_mm=round(2 * hw(0), 1), thickness_mm=K.baffle * MM,
                cutout_diameter_mm=round(2 * Z_CUT, 1), flange_pcd_mm=round(2 * pcd, 1), flange_holes=8,
                standoff_pcd_mm=round(2 * R_STANDOFF, 1), edge_bevel_deg=K.half_angle_deg)


def back_panel():
    start("back")
    Y = HH + T
    b = hexa(rect(ohw(Z_BACK), Y, Z_BACK), rect(ohw(Z_WB), Y, Z_WB))
    hx = hw(Z_WB)
    go = [(3, occ.addBox(-(hx + 9), -(HH + 9), Z_WB - 2, 2 * (hx + 9), 2 * (HH + 9), 3))]
    gi = [(3, occ.addBox(-(hx + 2), -(HH + 2), Z_WB - 3, 2 * (hx + 2), 2 * (HH + 2), 5))]
    groove, _ = occ.cut(go, gi)
    tools = list(groove)
    screws = [(s * (hx + 13.5), t * HH) for s in (-1, 1) for t in (-0.6, 0.0, 0.6)] + \
             [(t * hx, s * (HH + 12)) for s in (-1, 1) for t in (-0.6, 0.0, 0.6)]
    for x, y in screws:
        tools.append((3, occ.addCylinder(x, y, Z_BACK - 1, 0, 0, T + 2, 2.75)))
    tools.append((3, occ.addCylinder(0, 0, Z_BACK - 1, 0, 0, T + 2, 32)))
    occ.cut(b, tools)
    save("back", "ply", "cabinet", explode=(0, 0, -480), group="cabinet",
         note=f"18 mm birch rectangle, removable, {2*ohw(Z_WB):.0f} x {2*Y:.0f} at the front face, side edges "
              f"bevelled {K.half_angle_deg:.1f} deg to continue the walls; 2 mm deep gasket groove; 12 M5 screws "
              f"into inserts in the wall, top and bottom ends; 64 mm terminal cup")
    return dict(front_width_mm=round(2 * ohw(Z_WB), 1), back_width_mm=round(2 * ohw(Z_BACK), 1), height_mm=round(2 * Y, 1),
                front_square_mm=round(2 * ohw(Z_WB), 1), thickness_mm=T, edge_bevel_deg=K.half_angle_deg,
                screws="12 x M5 into threaded inserts in the wall, top and bottom end grain",
                gasket="3 mm closed-cell foam in a 7 mm wide, 2 mm deep groove over the panel ends",
                screw_xy_mm=[[round(x, 1), round(y, 1)] for x, y in screws])


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
                  f"the mouth, in the corner of a side wall and a wedge face; hardwood or 18 mm ply laminations; "
                  f"slot for the vane")
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
    zz = np.linspace(z0, z1, 13)
    r0 = min(body_r(z) for z in zz) - 15
    r1 = max(hw(z) * math.sqrt(2) - K.filler(z / MM) * MM / math.sqrt(2) for z in zz) + 10
    for az in [math.radians(45 + 90 * k) for k in range(4)]:
        if only is not None and not any(abs(math.remainder(az - o, 2 * math.pi)) < 1e-6 for o in only):
            continue
        rc = occ.addRectangle(-VANE_T / 2, -VANE_CHORD / 2, 0, VANE_T, VANE_CHORD, roundedRadius=VANE_T / 2 - 0.01)
        ex = occ.extrude([(2, rc)], 0, 0, r1 - r0)
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


def standoffs():
    for i, az in enumerate(STANDOFF_AZ):
        name = "standoff_duct" if i == DUCT else f"standoff_{i + 1}"
        start(name)
        standoff_shape(i)
        c, s = math.cos(az), math.sin(az)
        save(name, "aluminium", "support", explode=(c * 160, s * 160, -60), group="supports", curv=48,
             note=f"20 mm aluminium round, {Z_TURN + SOCKET:.0f} long, M8 into the baffle, {SOCKET:.0f} mm into "
                  f"the shroud" + ("; 12 mm bore: cable duct from the chamber to the centre body" if i == DUCT else ""))


def vanes():
    for name, sx, sy in CORNERS:
        start(f"vane_{name}")
        kick_vanes(only=[math.atan2(sy, sx)])
        save(f"vane_{name}", "aluminium", "support", explode=(sx * 260, sy * 260, 60), group="supports",
             curv=24, smin=1.0,
             note=f"{VANE_T:.0f} mm plate, {VANE_CHORD:.0f} mm chord with round edges, body to corner filler, "
                  f"centred {Z_M - Z_VANE:.0f} mm behind the kick mouth")


# ---------------------------------------------------------------- centre body
def mouth_round():
    """Rounded edge where the mid horn bore meets the centre body's outside, in the kick mouth plane.

    One arc tangent to the bore and to the body, with its front at the mouth plane (nothing sticks out), radius
    at most MidHorn.lip_radius. Returns (radius, arc points from the body side over the front to the bore side)."""
    rw, rb = M.radius(M.length) * MM, body_r(Z_M)
    tw = (M.radius(M.length) - M.radius(M.length - 0.004)) / 0.004
    tb = (body_r(Z_M) - body_r(Z_M - 4.0)) / 4.0
    rho = (rb - rw) / (math.sqrt(1 + tw * tw) + math.sqrt(1 + tb * tb) - tw + tb)
    rho = min(rho, M.lip_radius * MM)
    zc = Z_M - rho
    rc = rw - tw * rho + rho * math.sqrt(1 + tw * tw)
    pb, pw = math.atan2(-tb, 1.0), math.atan2(tw, -1.0)
    phi = np.linspace(pb, pw, 24)
    return rho, [(rc + rho * math.cos(p), zc + rho * math.sin(p)) for p in phi]


def body_outer(z):
    """Outside radius of the centre body at z (mm)."""
    if z <= Z_TURN:
        return body_r(Z_TURN)
    return body_r(min(z, Z_M))


def body_inner(z):
    """Largest cavity or bore radius at z (mm)."""
    if z < j1():
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
    J1 = j1()
    segs = []
    rr = np.linspace(R_CS, d.rim_r * MM, 40)
    segs.append(("spline", [(r, shroud_z(r)) for r in rr]))
    zr = shroud_z(d.rim_r * MM)
    lip = [(d.rim_r * MM, zr)]
    if Z_TURN - zr > 0.1:
        lip.append((d.rim_r * MM, Z_TURN))
    elif zr > Z_TURN + 0.1:
        warnings.append(f"shroud at the rim ({zr:.1f} mm) is above the turn plane ({Z_TURN:.1f} mm)")
    lip.append((body_outer(Z_TURN), Z_TURN))
    segs.append(("line", lip))
    rho, arc = mouth_round()
    zz = np.r_[P["body"].x[P["body"].x < arc[0][1] - 0.3]]
    segs.append(("spline", [(body_r(z), z) for z in zz] + [arc[0]]))
    segs.append(("spline", arc))
    s_w = s_of(arc[-1][1])
    sb = np.r_[np.arange(math.floor(s_w * MM / 2.5) * 2.5 / MM, 0, -0.0025), 0.0]
    sb = sb[sb < s_w - 2e-4]
    segs.append(("spline", [arc[-1]] + [(m.radius(s) * MM, zs(s)) for s in sb]))
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
    if not FINAL:
        mesh_write(TMP / "body_full.stl", size=12.0, curv=120)
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
    if not FINAL:
        return dict(volume_cm3=round(vol / 1000, 1))
    J1 = j1()
    zbot = Z_TURN - 1
    # mid horn joint: the first z behind the kick mouth where the wall is 40 mm thick over the bolt length,
    # clear of the kick vanes and the bullet vane slots
    j4 = None
    for z in np.arange(Z_M - 5, Z_MT + 30, -2.0):
        r_in, r_out = joint_ring(z)
        if r_out - r_in >= 40 and abs(z - Z_VANE) > VANE_CHORD / 2 + 30 and not zs(BV_S0) - 30 < z < zs(BV_S1) + 30:
            j4 = float(z)
            break
    j4 = j4 or (Z_MT + Z_M) / 2
    rho, _ = mouth_round()
    mods = [
        ("kick_shroud", zbot, J1, "underside follows the kick cone, sized from cylinder slices; lip; standoff sockets"),
        ("motor_fairing", J1, Z_MOTOR, f"covers the kick motor with {K.motor_clear*MM:.0f} mm clearance"),
        ("mid_pod", Z_MOTOR, Z_MF, f"sealed rear volume of the mid, {M.rear_volume_litres:.1f} L"),
        ("mid_horn_rear", Z_MF, j4, "lead-in ring over the mid cone, mid horn bore, vane slots; mid driver screws to its back face"),
        ("mid_horn_front", j4, Z_M + 1, f"mid horn to the mouth, flush with the kick mouth, {rho:.1f} mm rounded edge"),
    ]
    joints = [J1, Z_MOTOR, Z_MF, j4]
    for i, (name, a, b, note) in enumerate(mods):
        lo = [a] if a in joints else []
        hi = [b] if b in joints else []
        module(name, a, b, lo, hi, explode=(0, 0, 120 + 90 * i), note=note)
    return dict(joints_mm=[round(z, 1) for z in joints], volume_cm3=round(vol / 1000, 1), mouth_edge_radius_mm=round(rho, 2))


# ---------------------------------------------------------------- bullet
def bullet_vanes(slots=False):
    out = []
    s0, s1 = BV_S0, BV_S1
    r0 = min(bullet_r(s) for s in np.linspace(s0, s1, 9)) - 2
    r1 = max(M.radius(s) for s in np.linspace(s0, s1, 5)) * MM + 3
    for az in BV_AZ:
        chord = (s1 - s0) * MM
        t = BV_T + (0.4 if slots else 0.0)
        rc = occ.addRectangle(-t / 2, -chord / 2 - (0.2 if slots else 0), 0, t, chord + (0.4 if slots else 0),
                              roundedRadius=t / 2 - 0.01)
        ex = occ.extrude([(2, rc)], 0, 0, r1 - r0)
        v = [e for e in ex if e[0] == 3]
        occ.affineTransform(v, [0, 0, 1, r0, 1, 0, 0, 0, 0, 1, 0, zs((s0 + s1) / 2)])
        occ.rotate(v, 0, 0, 0, 0, 0, 1, az)
        out += v
    return out


def bullet_profile():
    m = M
    r_hb = TW.body_diameter / 2 * MM + 0.3
    s_cav = m.s_exit - m.hf_depth
    r0 = bullet_r(0.0) / MM
    rr = np.linspace(0, r0, 20)
    nose = [(r * MM, zs(float(m.nose(r)))) for r in rr]
    segs = [("spline", nose), ("line", [nose[-1], (r0 * MM, Z_MT)])]
    ss = P["bullet"].x
    segs.append(("spline", [(bullet_r(s / MM), zs(s / MM)) for s in ss]))
    sh = np.linspace(m.s_tip, m.s_exit, 30)
    segs.append(("line", [(bullet_r(m.s_tip), zs(m.s_tip)), (m.hf_horn_radius(m.s_tip) * MM, zs(m.s_tip))]))
    segs.append(("spline", [(m.hf_horn_radius(s) * MM, zs(s)) for s in sh]))
    segs.append(("line", [(m.hf_exit_diameter / 2 * MM, zs(m.s_exit)), (r_hb, zs(m.s_exit)), (r_hb, zs(s_cav)),
                          (0, zs(s_cav))]))
    return segs, s_cav


FELT = 3.0


def felt_shape():
    """Felt disc on the nose underside: its face is the designed nose surface, recessed FELT mm into the bullet."""
    rr = np.linspace(0, bullet_r(0.0) / MM - 0.002, 20)
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
    save("felt", "driver_cone", "felt", explode=(0, 0, 600), group="bullet", curv=48,
         note=f"{FELT:.0f} mm wool felt on the bullet nose underside, set into a recess so its face is at the "
              f"designed gap from the mid cone (MidHorn.felt_loss models its damping)")
    start("bullet")
    b = revolve(segs)
    b, _ = occ.intersect(b, slab(z_split, zs(M.s_tip) + 1))
    az = BV_AZ[0]
    c, s_ = math.cos(az), math.sin(az)
    zh = zs(S_HF_CABLE)
    holes = [radial_hole(az, R_CABLE_B, bullet_r(S_HF_CABLE) + 1, zh, 1.5),
             (3, occ.addCylinder(R_CABLE_B * c, R_CABLE_B * s_, zs(M.s_exit) - 1, 0, 0, zh - zs(M.s_exit) + 1, 1.5))]
    slots = bullet_vanes(slots=True)
    occ.cut(b, holes + slots)
    save("bullet", "print_orange", "print", explode=(0, 0, 700), group="bullet", printable=True,
         note=f"bullet round the {TW.model} body cylinder, outside sized from slices of the mid horn, printed HF "
              f"horn {M.hf_exit_diameter*MM:.1f} to {2*M.hf_mouth_radius*MM:.0f} mm over {M.hf_len*MM:.0f} mm; "
              f"3 mm HF cable bore from the vane back to the HF cavity")
    start("bullet_vanes")
    v = bullet_vanes()
    occ.cut(v, [radial_hole(BV_AZ[0], bullet_r(S_HF_CABLE) - 2, M.radius(S_HF_CABLE) * MM + 4, zh, 1.5)])
    save("bullet_vanes", "print_orange", "print", explode=(0, 0, 560), group="bullet", printable=True, curv=48,
         note=f"three {BV_T:.0f} mm vanes hold the bullet in the mid horn, {BV_S0*MM:.0f} to {BV_S1*MM:.0f} mm past "
              f"the throat; one carries the HF cable in a 3 mm bore")


# ---------------------------------------------------------------- drivers (proxies)
def cone_shell(drv, zmap, t=1.5, rim_out=None, roll_h=None):
    """Cone between vc_r and rim_r as a thin shell, plus dust cap and a half-roll surround."""
    rr = np.linspace(drv.vc_r, drv.rim_r, 30)
    top = [(r * MM, zmap(float(drv.cone_z(r)) * MM)) for r in rr]
    sgn = 1 if zmap(1.0) > zmap(0.0) else -1
    bot = [(r, z + sgn * t) for r, z in top[::-1]]
    segs = [("spline", top), ("line", [top[-1], bot[0]]), ("spline", bot)]
    out = revolve(segs)
    rv = drv.vc_r * MM
    zn = zmap(drv.neck_z * MM)
    th = np.linspace(0, math.pi / 2, 12)
    dome = [(rv * math.sin(a), zn - sgn * 0.3 * rv * math.cos(a)) for a in th]
    out += revolve([("spline", dome), ("line", [dome[-1], (0, zn)])])
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
    kw = dict(curv=36, smin=1.0)
    start("drv_kick_frame")
    fl = ring(0.0, d.flange_t * MM, Z_CUT + 2, d.frame_od / 2 * MM)
    holes = [cyl_at(d.frame_od / 2 * MM - 9, math.radians(22.5 + 45 * k), -1, d.flange_t * MM + 1, 3.5) for k in range(8)]
    fl, _ = occ.cut(fl, holes)
    arms = basket(d, zk, K.arm * MM, Z_CUT + 4, d.flange_t * MM)
    hous = ring(d.neck_z * MM - 5, d.top_plate_z * MM, d.spider_r * MM + 2, d.spider_r * MM + 5)
    occ.fuse(fl, arms + hous)
    save("drv_kick_frame", "driver_metal", "driver", explode=(0, 0, -120), group="kick driver", **kw,
         note=f"proxy of the {D.label(D.KICK_RECORD)} frame, reversed: flange on the baffle front, basket into the horn")
    start("drv_kick_cone")
    cone_shell(d, zk, rim_out=Z_CUT - 3)
    save("drv_kick_cone", "driver_cone", "driver", explode=(0, 0, -120), group="kick driver", **kw,
         note="proxy cone (estimated profile), dust cap and surround")
    start("drv_kick_motor")
    mot = [(3, occ.addCylinder(0, 0, d.top_plate_z * MM, 0, 0, d.motor_h * MM, d.motor_r * MM))]
    occ.cut(mot, [(3, occ.addCylinder(0, 0, d.top_plate_z * MM - 1, 0, 0, d.motor_h * MM + 2, 0.5 * d.vc_r * MM))])
    save("drv_kick_motor", "driver_metal", "driver", explode=(0, 0, -120), group="kick driver", **kw,
         note=f"kick motor proxy, {2*d.motor_r*MM:.0f} x {d.motor_h*MM:.0f} mm")
    md = MD
    zmid = lambda z: Z_MF - z  # noqa: E731
    start("drv_mid_frame")
    fl = ring(Z_MF - md.flange_t * MM, Z_MF, md.frame_od / 2 * MM - 12, md.frame_od / 2 * MM)
    arms = basket(md, zmid, md.xmech * MM + 8, md.frame_od / 2 * MM - 12, Z_MF - md.flange_t * MM)
    hous = ring(zmid(md.top_plate_z * MM), zmid(md.neck_z * MM - 4), md.spider_r * MM + 2, md.spider_r * MM + 5)
    occ.fuse(fl, arms + hous)
    save("drv_mid_frame", "driver_metal", "driver", explode=(0, 0, 330), group="mid driver", **kw,
         note=f"proxy of the {D.label(D.MID_RECORD)} frame")
    start("drv_mid_cone")
    cone_shell(md, zmid, t=1.0, rim_out=md.frame_od / 2 * MM - 13, roll_h=md.rim_z * MM)
    save("drv_mid_cone", "driver_cone", "driver", explode=(0, 0, 330), group="mid driver", **kw,
         note="proxy cone (estimated profile); surround assumed flush with the flange face")
    start("drv_mid_motor")
    z0 = zmid(md.top_plate_z * MM) - md.motor_h * MM
    occ.addCylinder(0, 0, z0, 0, 0, md.motor_h * MM, md.motor_r * MM)
    save("drv_mid_motor", "driver_metal", "driver", explode=(0, 0, 330), group="mid driver", note="mid motor proxy", **kw)
    start("drv_hf")
    se = zs(M.s_exit)
    rb = TW.body_diameter / 2 * MM
    body = [(3, occ.addCylinder(0, 0, se - TW.body_depth * MM, 0, 0, TW.body_depth * MM, rb))]
    occ.cut(body, [(3, occ.addCylinder(0, 0, se - 10, 0, 0, 11, M.hf_exit_diameter / 2 * MM))])
    if M.hf_depth - TW.body_depth > 0.001:            # terminals behind the body
        occ.addBox(-8, -4, se - M.hf_depth * MM, 16, 8, (M.hf_depth - TW.body_depth) * MM + 0.5)
    save("drv_hf", "driver_metal", "driver", explode=(0, 0, 760), group="bullet", **kw,
         note=f"{TW.model}: {TW.body_diameter*MM:.0f} mm body cylinder, {TW.body_depth*MM:.1f} mm deep, flange removed")


# ---------------------------------------------------------------- plug design from slices
def load_meshes(names, extra=None):
    out = {}
    for n in names:
        f = OUT / "stl" / f"{n}.stl"
        cat = next((p["category"] for p in parts if p["name"] == n), "")
        out[n] = (cat, trimesh.load_mesh(f, process=True))
    for n, f in (extra or {}).items():
        out[n] = ("print", trimesh.load_mesh(f, process=True))
    return out


def disc(r):
    return Point(0, 0).buffer(r, quad_segs=512)


def envelope_mm(z):
    """Smallest centre body radius that still holds what is inside it at z, with the cable and socket features."""
    e = K.envelope(z / MM) * MM
    if z <= Z_TURN + SOCKET + 3:
        e = max(e, R_STANDOFF + STANDOFF_R + 4)
    if z <= Z_CAB + 8:
        e = max(e, R_STANDOFF + 6 + 4)
    if zs(0) - 30 <= z <= zs(S_HF_CABLE) + 5:
        e = max(e, R_HF_CH + 4 + 4)
    return e


def hold_mm(s):
    """Smallest bullet radius that still holds the HF body cylinder (with its sleeve) or the HF horn and its wall."""
    if s <= M.s_exit + 0.002:
        return M.plug_radius * MM
    h = (M.hf_horn_radius(s) + M.hf_wall) * MM
    if s <= S_HF_CABLE + 0.002:
        h = max(h, R_CABLE_B + 1.5 + 1.5)
    return h


def solve_plugs(meshes):
    """Slice every built part and solve the body and bullet radii so the air left follows each law."""
    rows_k, rows_m = [], []
    zk = _grid(Z_TURN, Z_M, DZ_SOLVE)
    for z in zk:
        zq = min(max(z, Z_TURN + 0.05), Z_M - 0.05)
        inside, solid, _ = SL.inside_and_solid(meshes, zq)
        free = inside.difference(solid)
        law = K.law_area(z / MM) * 1e6
        lo = envelope_mm(z) + (3.0 if z < Z_M - 10 else 0.0)
        hi = hw(z) - 1.0

        def f(R):
            return free.difference(disc(R)).area - law
        if f(lo) <= 0:
            R, bind = lo, True
        else:
            R, bind = brentq(f, lo, hi, xtol=1e-4), False
        exposed = inside.difference(disc(R)).area - free.difference(disc(R)).area
        extra = exposed - (inside.area - K.wall_area(z / MM) * 1e6)       # beyond the wedges and fillers
        rows_k.append(dict(z_mm=z, r_mm=R, r_equation_mm=K.body_radius(z / MM) * MM, envelope_mm=envelope_mm(z),
                           law_cm2=law / 100, inside_cm2=inside.area / 100, obstruction_cm2=extra / 100,
                           shortfall_cm2=-f(R) / 100 if bind else 0.0, held_by_envelope=bind))
    sm = _grid(0.0, M.s_tip * MM, DZ_SOLVE)
    for s_mm in sm:
        s = s_mm / MM
        z = min(max(zs(s), Z_MT + 0.05), zs(M.s_tip) - 0.05)
        inside, solid, _ = SL.inside_and_solid(meshes, z)
        Rw = M.radius(s) * MM
        wall = disc(Rw)
        free = wall.difference(solid) if solid is not None else wall
        law = M.area(s) * 1e6
        lo, hi = hold_mm(s), Rw - 1.0

        def g(R):
            return free.difference(disc(R)).area - law
        if g(lo) <= 0:
            R, bind = lo, True
        else:
            R, bind = brentq(g, lo, hi, xtol=1e-4), False
        exposed = wall.difference(disc(R)).area - free.difference(disc(R)).area
        rows_m.append(dict(s_mm=s_mm, z_mm=zs(s), r_mm=R, r_equation_mm=M.plug(s) * MM, hold_mm=hold_mm(s),
                           wall_mm=Rw, law_cm2=law / 100, obstruction_cm2=exposed / 100,
                           shortfall_cm2=-g(R) / 100 if bind else 0.0, held_by_hf=bind))
    return rows_k, rows_m


def correct(prof, xs_mm, err_mm2, lo_fn):
    """Grow (or shrink) a plug so a measured area excess err (mm2, measured minus law) at stations xs goes."""
    x = prof.x
    e = np.interp(x, xs_mm, err_mm2, left=0.0, right=0.0)
    r2 = np.maximum(prof.y ** 2 + e / math.pi, 0.0)
    r = np.sqrt(r2)
    r = np.maximum(r, [lo_fn(v) for v in x])
    return Prof(x, r)


# ---------------------------------------------------------------- cone channel (cylinder slices)
def cone_channel(with_shroud, radii=None):
    """Open area over the reversed kick cone on cylinders r = const, measured with OCC from the part solids.

    A 0.5 mm cylindrical shell at each radius is cut by the cone, the basket and (if built) the shroud; the shell
    left between the cone and the shroud, over its thickness, is the open area on that cylinder."""
    start("channel")
    frame = occ.importShapes(str(OUT / "step" / "drv_kick_frame.step"))
    cone = occ.importShapes(str(OUT / "step" / "drv_kick_cone.step"))
    shroud = occ.importShapes(str(OUT / "step" / "kick_shroud.step")) if with_shroud else []
    radii = np.linspace(R_CS + 2, KD.rim_r * MM - 2, 12) if radii is None else radii
    rows = []
    dr = 0.5
    for r in radii:
        sh = ring(-1.0, j1() - 1.0, r, r + dr)         # to inside the shroud, above its underside
        a, _ = occ.intersect(occ.copy(sh), occ.copy(frame))
        blocked = sum(occ.getMass(*e) for e in a if e[0] == 3) / dr
        arm_top = max((occ.getBoundingBox(*e)[5] for e in a if e[0] == 3), default=None)
        if a:
            occ.remove(a, recursive=True)
        c, _ = occ.intersect(occ.copy(sh), occ.copy(cone))
        cone_top = max(occ.getBoundingBox(*e)[5] for e in c if e[0] == 3)
        cone_bot = min(occ.getBoundingBox(*e)[2] for e in c if e[0] == 3)
        occ.remove(c, recursive=True)
        open_ = None
        if with_shroud:
            air, _ = occ.cut(occ.copy(sh), occ.copy(frame + cone + shroud))
            open_ = sum(occ.getMass(*e) for e in air if e[0] == 3 and occ.getBoundingBox(*e)[2] > cone_bot + 0.5) / dr
            occ.remove(air, recursive=True)
        occ.remove(sh, recursive=True)
        rm = r + dr / 2
        psi = math.atan(float(KD.cone_slope(rm / MM)))
        rows.append(dict(r_mm=rm, psi_deg=math.degrees(psi), cone_top_mm=cone_top, arm_top_mm=arm_top,
                         arms_cyl_cm2=blocked / 100, open_cyl_cm2=None if open_ is None else open_ / 100,
                         target_normal_cm2=math.pi * rm**2 / K.compression / 100))
    return rows


def design_shroud(rows):
    """Shroud underside from the cylinder slices: the open area normal to the cone is cone area inside r over the
    compression, after the basket arms; no lower than the arms plus 3 mm or the cone travel plus 3 mm."""
    r = np.array([x["r_mm"] for x in rows])
    ct = PchipInterpolator(r, [x["cone_top_mm"] for x in rows], extrapolate=True)
    ab = PchipInterpolator(r, [x["arms_cyl_cm2"] * 100 for x in rows], extrapolate=True)
    at = PchipInterpolator(r, [x["arm_top_mm"] if x["arm_top_mm"] is not None else x["cone_top_mm"] for x in rows],
                           extrapolate=True)
    rr = _grid(R_CS, KD.rim_r * MM, 3.0)
    zz = []
    for v in rr:
        cos = math.cos(math.atan(float(KD.cone_slope(v / MM))))
        need = math.pi * v**2 / K.compression / cos + float(ab(v))
        z = float(ct(v)) + need / (2 * math.pi * v)
        z = max(z, float(at(v)) + 3.0, float(ct(v)) + (KD.xmech * MM + 3) / cos)
        zz.append(z)
    return Prof(rr, zz)


# ---------------------------------------------------------------- 3 x 3 array
def _outer_face(w):
    """Outer face of a box (mm, box frame): four corners, and the outward normal."""
    Y = HH + T
    zb, zm = Z_BACK, Z_M
    if w in ("+x", "-x"):
        s = 1 if w == "+x" else -1
        pts = [(s * ohw(zb), -Y, zb), (s * ohw(zb), Y, zb), (s * ohw(zm), Y, zm), (s * ohw(zm), -Y, zm)]
        n = np.array([s * math.cos(A), 0, -math.sin(A)])
    else:
        s = 1 if w == "+y" else -1
        pts = [(-ohw(zb), s * Y, zb), (ohw(zb), s * Y, zb), (ohw(zm), s * Y, zm), (-ohw(zm), s * Y, zm)]
        n = np.array([0, s, 0.0])
    return np.array(pts), n


def _box_solid(R, t):
    Y = HH + T
    v = np.array(rect(ohw(Z_BACK), Y, Z_BACK) + rect(ohw(Z_M), Y, Z_M)) @ R.T + t
    return hexa([tuple(p) for p in v[:4]], [tuple(p) for p in v[4:]])


def array_check():
    lay = array_layout(K)
    boxes = {}
    for b in lay["boxes"]:
        Mx = np.array(b["matrix"])
        boxes[(b["i"], b["j"])] = (Mx[:3, :3], Mx[:3, 3] * MM)
    joints = []
    for (i, j), (R, t) in boxes.items():
        for di, dj, wa, wb in ((1, 0, "+x", "-x"), (0, 1, "+y", "-y")):
            key = (i + di, j + dj)
            if key not in boxes:
                continue
            R2, t2 = boxes[key]
            fa, na = _outer_face(wa)
            fb, _ = _outer_face(wb)
            pa, pb = fa @ R.T + t, fb @ R2.T + t2
            na = R @ na
            d = (pb - pa[0]) @ na                      # > 0: gap, < 0: penetration
            mis = max(float(np.min(np.linalg.norm(pb - p, axis=1))) for p in pa)
            joints.append(dict(a=[i, j], b=list(key), kind="column" if di else "row",
                               max_gap_mm=round(float(max(d.max(), 0.0)), 4),
                               max_penetration_mm=round(float(max(-d.min(), 0.0)), 4),
                               corner_mismatch_mm=round(mis, 4)))
    start("array")
    tags = {k: _box_solid(*v) for k, v in boxes.items()}
    worst, keys = 0.0, list(boxes)
    for n, a in enumerate(keys):
        for b in keys[n + 1:]:
            if max(abs(a[0] - b[0]), abs(a[1] - b[1])) != 1:
                continue
            res, _ = occ.intersect(occ.copy(tags[a]), occ.copy(tags[b]))
            v = sum(occ.getMass(*e) for e in res if e[0] == 3)
            worst = max(worst, v / 1000)
            if res:
                occ.remove(res, recursive=True)
    out = dict(
        splay_deg=K.splay_deg, boxes_n=len(boxes), apex_behind_baffle_mm=round(lay["apex_behind_baffle_m"] * MM, 1),
        row_pitch_mm=round(lay["row_pitch_m"] * MM, 2),
        frame="Each matrix maps a box's own frame (metres; origin at the centre of its baffle front face, +z to the "
              "listener, +x right, +y up) to the array frame, which is the centre box's own frame.",
        method="geometry.array_layout: three columns fanned by the splay about the vertical line where the outer "
               "side faces of the centre box meet behind it; rows stacked flat at the external height. The angled "
               "side faces of neighbouring columns coincide, and the flat tops and bottoms of neighbouring rows "
               "coincide, so nine identical boxes tile with no spacers.",
        max_gap_mm=max(j["max_gap_mm"] for j in joints),
        max_penetration_mm=max(j["max_penetration_mm"] for j in joints),
        max_corner_mismatch_mm=max(j["corner_mismatch_mm"] for j in joints),
        max_overlap_cm3_occ=round(worst, 4),
        joints=joints,
        boxes=[dict(i=b["i"], j=b["j"], yaw_deg=b["yaw_deg"], pitch_deg=b["pitch_deg"], matrix=b["matrix"])
               for b in sorted(lay["boxes"], key=lambda b: (-b["j"], b["i"]))],
    )
    return out, boxes


def coarse_triangles(names, size=40.0):
    """Coarse surface mesh (mm) of the named parts, for the array preview."""
    start("coarse")
    for n in names:
        occ.importShapes(str(OUT / "step" / f"{n}.step"))
    occ.synchronize()
    gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", 0)
    gmsh.option.setNumber("Mesh.MeshSizeExtendFromBoundary", 0)
    gmsh.option.setNumber("Mesh.MeshSizeMax", size)
    gmsh.option.setNumber("Mesh.MeshSizeMin", size / 4)
    gmsh.model.mesh.generate(2)
    gmsh.option.setNumber("Mesh.MeshSizeExtendFromBoundary", 1)
    tags, xyz, _ = gmsh.model.mesh.getNodes()
    xyz = xyz.reshape(-1, 3)
    index = np.zeros(int(tags.max()) + 1, dtype=int)
    index[tags.astype(int)] = np.arange(len(tags))
    _, nodes = gmsh.model.mesh.getElementsByType(2)
    return xyz[index[nodes.astype(int)]].reshape(-1, 3, 3)


def write_array_stl(boxes, tri):
    out = []
    for R, t in boxes.values():
        v = tri.reshape(-1, 3) @ R.T + t
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


# ---------------------------------------------------------------- panels
def panel_bevels():
    """Saw settings for the wedge panel: its long edges sit on the side walls (compound), its back end on the baffle."""
    n_p = np.array([0, 1, -math.tan(A)])
    n_p /= np.linalg.norm(n_p)                              # wedge face (top wedge), towards the flat top
    n_s = np.array([1, 0, -math.tan(A)])
    n_s /= np.linalg.norm(n_s)                              # right side wall inner face, outward
    e = np.cross(n_s, n_p)
    e /= np.linalg.norm(e)                                  # the corner line, along the panel's long edge
    u = np.cross(e, n_p)
    u /= np.linalg.norm(u)                                  # square-cut edge face normal
    bevel = math.degrees(math.acos(min(1.0, abs(float(u @ n_s)))))
    length_dir = np.array([0, math.sin(A), math.cos(A)])
    taper = math.degrees(math.acos(min(1.0, abs(float(e @ length_dir)))))
    return dict(blade_bevel_deg=round(bevel, 2), edge_taper_deg=round(taper, 2), end_bevel_deg=round(K.half_angle_deg, 2),
                corner_dihedral_deg=round(math.degrees(math.pi - math.acos(float(n_p @ n_s))), 2))


def dxf(baf, back, bev, ze):
    import ezdxf
    (OUT / "dxf").mkdir(exist_ok=True)
    for f in (OUT / "dxf").glob("*.dxf"):
        f.unlink()

    def doc_with():
        doc = ezdxf.new("R2010")
        doc.layers.add("CUT")
        doc.layers.add("INNER_FACE", linetype="DASHED")
        doc.layers.add("TEXT")
        return doc, doc.modelspace()

    def text(msp, s, x, y):
        msp.add_text(s, dxfattribs={"height": 10, "layer": "TEXT"}).set_placement((x, y))

    # side wall: a rectangle on the slant; the outer face is longer than the inner by the end bevels
    Ls = (Z_M - Z_WB) / math.cos(A)               # same on both faces: the end bevels are parallel
    off = T * math.tan(A)
    doc, msp = doc_with()
    msp.add_lwpolyline([(0, -HH), (Ls, -HH), (Ls, HH), (0, HH)], close=True, dxfattribs={"layer": "CUT"})
    msp.add_lwpolyline([(off, -HH), (Ls + off, -HH), (Ls + off, HH), (off, HH)], close=True, dxfattribs={"layer": "INNER_FACE"})
    text(msp, f"Side wall x2, 18 mm birch, outer face shown. Ends bevelled {K.half_angle_deg:.1f} deg, parallel. "
              f"Long edges square.", 0, -HH - 30)
    doc.saveas(OUT / "dxf" / "side_wall.dxf")
    # top and bottom: flat trapezoids, square edges
    doc, msp = doc_with()
    L = Z_M - Z_WB
    msp.add_lwpolyline([(-ohw(Z_WB), 0), (ohw(Z_WB), 0), (ohw(Z_M), L), (-ohw(Z_M), L)], close=True, dxfattribs={"layer": "CUT"})
    msp.add_lwpolyline([(-hw(Z_WB), 0), (hw(Z_WB), 0), (hw(Z_M), L), (-hw(Z_M), L)], close=True, dxfattribs={"layer": "INNER_FACE"})
    text(msp, "Top and bottom x2, 18 mm birch, flat. All edges square. Dashed: the side walls' inner faces.",
         -ohw(Z_M), -30)
    doc.saveas(OUT / "dxf" / "top_bottom.dxf")
    # wedge panel: on its slope, from the baffle to the mouth
    doc, msp = doc_with()
    Ls = Z_M / math.cos(A)
    msp.add_lwpolyline([(-hw(0), 0), (hw(0), 0), (hw(Z_M), Ls), (-hw(Z_M), Ls)], close=True, dxfattribs={"layer": "CUT"})
    zl = (ze) / math.cos(A)
    msp.add_line((-hw(ze), zl), (hw(ze), zl), dxfattribs={"layer": "INNER_FACE"})
    text(msp, f"Wedge panel x2, 18 mm birch, horn face shown. Long edges: blade bevel {bev['blade_bevel_deg']:.1f} deg, "
              f"edge taper {bev['edge_taper_deg']:.1f} deg. Back end bevelled {bev['end_bevel_deg']:.1f} deg on the "
              f"baffle. Plane the back face to a feather edge from the dashed line to the mouth.", -hw(Z_M), -30)
    doc.saveas(OUT / "dxf" / "wedge_panel.dxf")
    # wedge rib: triangle in the y-z plane
    doc, msp = doc_with()
    h = HH - hw(0) - E
    msp.add_lwpolyline([(0, 0), (ze, 0), (0, h)], close=True, dxfattribs={"layer": "CUT"})
    text(msp, f"Wedge rib x{2*len(WEDGE_RIBS)}, 18 mm birch. Long leg on the flat panel, short leg on the baffle, "
              f"hypotenuse on the wedge panel's back face.", 0, -30)
    doc.saveas(OUT / "dxf" / "wedge_rib.dxf")
    # baffle
    doc, msp = doc_with()
    a, b = baf["front_width_mm"] / 2, baf["back_width_mm"] / 2
    msp.add_lwpolyline([(-a, -HH), (a, -HH), (a, HH), (-a, HH)], close=True, dxfattribs={"layer": "CUT"})
    msp.add_lwpolyline([(-b, -HH), (b, -HH), (b, HH), (-b, HH)], close=True, dxfattribs={"layer": "INNER_FACE"})
    msp.add_circle((0, 0), baf["cutout_diameter_mm"] / 2, dxfattribs={"layer": "CUT"})
    for k in range(8):
        az = math.radians(22.5 + 45 * k)
        msp.add_circle((baf["flange_pcd_mm"] / 2 * math.cos(az), baf["flange_pcd_mm"] / 2 * math.sin(az)), 3.5,
                       dxfattribs={"layer": "CUT"})
    for i, az in enumerate(STANDOFF_AZ):
        msp.add_circle((R_STANDOFF * math.cos(az), R_STANDOFF * math.sin(az)), 6.0 if i == DUCT else 4.25,
                       dxfattribs={"layer": "CUT"})
    text(msp, f"Baffle, 25 mm birch, front face shown (back face dashed). Side edges bevelled {K.half_angle_deg:.1f} deg, "
              f"top and bottom edges square.", -a, -HH - 30)
    doc.saveas(OUT / "dxf" / "baffle.dxf")
    # back
    doc, msp = doc_with()
    a, b, Y = back["front_width_mm"] / 2, back["back_width_mm"] / 2, back["height_mm"] / 2
    msp.add_lwpolyline([(-a, -Y), (a, -Y), (a, Y), (-a, Y)], close=True, dxfattribs={"layer": "CUT"})
    msp.add_lwpolyline([(-b, -Y), (b, -Y), (b, Y), (-b, Y)], close=True, dxfattribs={"layer": "INNER_FACE"})
    msp.add_circle((0, 0), 32, dxfattribs={"layer": "CUT"})
    hx = hw(Z_WB)
    for g in (2, 9):
        msp.add_lwpolyline([(-(hx + g), -(HH + g)), (hx + g, -(HH + g)), (hx + g, HH + g), (-(hx + g), HH + g)],
                           close=True, dxfattribs={"layer": "INNER_FACE"})
    for x, y in back["screw_xy_mm"]:
        msp.add_circle((x, y), 2.75, dxfattribs={"layer": "CUT"})
    text(msp, f"Back panel, 18 mm birch, front face shown. Side edges bevelled {K.half_angle_deg:.1f} deg. "
              f"Gasket groove 7 x 2 mm.", -a, -Y - 30)
    doc.saveas(OUT / "dxf" / "back.dxf")


def assembly():
    start("assembly")
    for p in parts:
        occ.importShapes(str(OUT / "step" / f"{p['name']}.step"))
    occ.synchronize()
    quiet_write(OUT / "sunflower-assembly.step")


# ---------------------------------------------------------------- verification
OBSTRUCTIONS = None


def model_meshes():
    """Every built part, plus the whole centre body, for slicing (the body modules are only cut on the final pass)."""
    names = [p["name"] for p in parts if p["category"] != "print" or p["name"].startswith("bullet")]
    names = [n for n in names if not n.startswith(("kick_shroud", "motor_fairing", "mid_pod", "mid_horn"))]
    return load_meshes(names, extra={"centre_body": TMP / "body_full.stl"})


def slab_errors(df):
    k = df[(df.z_mm > Z_TURN + 5) & (df.z_mm < Z_M - 1) & df.kick_model_cm2.notna()]
    m = df[df.mid_model_cm2.notna() & (df.z_mm > Z_MT + 1) & (df.z_mm < Z_M - 1)]
    return k, m


def summary_stats(df):
    k, m = slab_errors(df)
    tip = zs(M.s_tip)
    mb = m[m.z_mm < tip - 1]
    return dict(kick_slabs=len(k), kick_max_abs_pct=round(float(k.kick_diff_pct.abs().max()), 3),
                kick_rms_pct=round(float(np.sqrt((k.kick_diff_pct ** 2).mean())), 3),
                kick_worst_z_mm=float(k.z_mm.iloc[int(k.kick_diff_pct.abs().argmax())]),
                mid_slabs=len(m), mid_max_abs_pct=round(float(m.mid_diff_pct.abs().max()), 3),
                mid_rms_pct=round(float(np.sqrt((m.mid_diff_pct ** 2).mean())), 3),
                mid_worst_z_mm=float(m.z_mm.iloc[int(m.mid_diff_pct.abs().argmax())]),
                mid_bullet_max_abs_pct=round(float(mb.mid_diff_pct.abs().max()), 3))


# ---------------------------------------------------------------- outputs
def write_profiles(rows_k, rows_m, ch_eq, ch_new, rho):
    pk = P["body"]
    pb = P["bullet"]
    ps = P["shroud"]
    zz = pk.x
    data = dict(
        units="mm; z from the baffle front face, s from the mid throat plane",
        note="Plug radii solved from slices of the assembled model (every part subtracted), then corrected from a "
             "second slicing of the model with the plugs in place. Each profile is a PCHIP spline through these "
             "stations.",
        kick_body=dict(
            z_mm=[round(float(v), 3) for v in zz], r_mm=[round(float(v), 3) for v in pk.y],
            r_equation_mm=[round(K.body_radius(z / MM) * MM, 3) for z in zz],
            envelope_mm=[round(envelope_mm(z), 3) for z in zz],
            law_area_cm2=[round(K.law_area(z / MM) * 1e4, 3) for z in zz],
            obstruction_cm2=[round(r["obstruction_cm2"], 3) for r in rows_k],
            held_by_envelope=[bool(r["held_by_envelope"]) for r in rows_k]),
        bullet=dict(
            s_mm=[round(float(v), 3) for v in pb.x], z_mm=[round(zs(v / MM), 3) for v in pb.x],
            r_mm=[round(float(v), 3) for v in pb.y], r_equation_mm=[round(M.plug(v / MM) * MM, 3) for v in pb.x],
            hold_mm=[round(hold_mm(v / MM), 3) for v in pb.x], wall_mm=[round(M.radius(v / MM) * MM, 3) for v in pb.x],
            obstruction_cm2=[round(r["obstruction_cm2"], 3) for r in rows_m],
            held_by_hf=[bool(r["held_by_hf"]) for r in rows_m]),
        shroud=dict(r_mm=[round(float(v), 3) for v in ps.x], z_mm=[round(float(v), 3) for v in ps.y],
                    z_equation_mm=[round(float(K.shroud_z(v / MM)) * MM, 3) for v in ps.x]),
        mouth_edge_radius_mm=round(rho, 3),
    )
    (OUT / "plug_profiles.json").write_text(json.dumps(data, indent=1) + "\n")
    return data


def channel_check():
    """Cylinder slices through the built shroud, cone and basket, against the target and the equation shroud."""
    rows = cone_channel(True)
    for r in rows:
        rm = r["r_mm"]
        cos = math.cos(math.radians(r["psi_deg"]))
        r["open_normal_cm2"] = r["open_cyl_cm2"] * cos
        r["diff_pct"] = 100 * (r["open_normal_cm2"] / r["target_normal_cm2"] - 1)
        eq_cyl = (2 * math.pi * rm * (float(K.shroud_z(rm / MM)) * MM - r["cone_top_mm"]) - r["arms_cyl_cm2"] * 100) / 100
        r["equation_shroud_open_cyl_cm2"] = eq_cyl
        r["equation_shroud_open_normal_cm2"] = eq_cyl * cos
        r["equation_shroud_diff_pct"] = 100 * (eq_cyl * cos / r["target_normal_cm2"] - 1)
        r["shroud_z_mm"] = shroud_z(rm)
        r["equation_shroud_z_mm"] = float(K.shroud_z(rm / MM)) * MM
    import pandas as pd
    RES.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(RES / "cone-channel.csv", index=False, float_format="%.4g")
    return rows


def write_area_check(df, rows_k, rows_m):
    """results/cad/area-check.csv: the slices in the older long format, with the vane blockage from the plug solve
    (the vane area left exposed in the passage, averaged over the slab)."""
    import pandas as pd
    zk = np.array([r["z_mm"] for r in rows_k])
    bk = np.array([max(r["obstruction_cm2"], 0.0) for r in rows_k])
    zm = np.array([r["z_mm"] for r in rows_m])
    bm = np.array([max(r["obstruction_cm2"], 0.0) for r in rows_m])

    def avg(z, h, zz, bb):
        x = np.linspace(z - h / 2, z + h / 2, 21)
        return float(np.mean(np.interp(x, zz, bb, left=0.0, right=0.0)))
    ac = []
    for _, r in df.iterrows():
        if not math.isnan(r.kick_model_cm2):
            v = avg(r.z_mm, r.slab_mm, zk, bk)
            ac.append(dict(kind="kick", z_mm=r.z_mm, s_mm="", cad_area_cm2=r.kick_cm2, model_area_cm2=r.kick_model_cm2,
                           diff_pct=r.kick_diff_pct, real_air_cm2=r.kick_cm2, vane_blockage_cm2=v,
                           vane_blockage_pct=100 * v / (r.kick_cm2 + v)))
    for _, r in df.iterrows():
        if not math.isnan(r.mid_model_cm2):
            v = avg(r.z_mm, r.slab_mm, zm, bm)
            ac.append(dict(kind="mid", z_mm=r.z_mm, s_mm=round(r.z_mm - Z_MT, 2), cad_area_cm2=r.mid_cm2,
                           model_area_cm2=r.mid_model_cm2, diff_pct=r.mid_diff_pct, real_air_cm2=r.mid_cm2,
                           vane_blockage_cm2=v, vane_blockage_pct=100 * v / (r.mid_cm2 + v)))
    RES.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(ac).to_csv(RES / "area-check.csv", index=False, float_format="%.4f")


def main():
    global FINAL
    t0 = time.time()
    for sub in ("step", "stl", "print", "dxf"):
        (OUT / sub).mkdir(parents=True, exist_ok=True)
    for sub in ("step", "stl", "print"):
        for f in (OUT / sub).glob("*"):
            f.unlink()
    TMP.mkdir(exist_ok=True)
    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 0)
    gmsh.option.setString("Geometry.OCCTargetUnit", "MM")
    print("cabinet", flush=True)
    walls()
    ze = wedges()
    baf = baffle()
    back = back_panel()
    fill = fillers()
    print("supports and drivers", flush=True)
    standoffs()
    drivers()

    # 1. shroud over the reversed cone, from cylinder slices of the cone and basket
    print("cone channel", flush=True)
    ch_design = cone_channel(False)
    P["shroud"] = design_shroud(ch_design)

    # 2. plugs from slices of everything else (twice: the vanes reach into the body and follow its radius)
    rows_k = rows_m = None
    for it in range(3):
        print(f"plug solve, pass {it + 1}", flush=True)
        vanes()
        start("bv_tmp")
        bullet_vanes()
        clean()
        mesh_write(TMP / "bullet_vanes.stl", size=12.0, curv=48)
        names = [p["name"] for p in parts]
        meshes = load_meshes(names, extra={"bullet_vanes": TMP / "bullet_vanes.stl"})
        rows_k, rows_m = solve_plugs(meshes)
        nk = Prof([r["z_mm"] for r in rows_k], [r["r_mm"] for r in rows_k])
        nm = Prof([r["s_mm"] for r in rows_m], [r["r_mm"] for r in rows_m])
        dk, dm = float(np.abs(nk.y - P["body"].y).max()), float(np.abs(nm.y - P["bullet"].y).max())
        P["body"], P["bullet"] = nk, nm
        print(f"  change: body {dk:.3f} mm, bullet {dm:.3f} mm", flush=True)
        if it > 0 and dk < 0.05 and dm < 0.05:
            break
    solved = dict(body=P["body"], bullet=P["bullet"])

    # 3. build the plugs, slice the whole model, correct, repeat
    history = []
    for it in range(3):
        print(f"verify, pass {it + 1}", flush=True)
        vanes()
        centre_body()
        bullet()
        df, _ = SL.run(dz=10.0, meshes=model_meshes(), write=False)
        st = summary_stats(df)
        history.append(st)
        print("  " + json.dumps(st), flush=True)
        if st["kick_max_abs_pct"] <= 0.5 and st["mid_bullet_max_abs_pct"] <= 0.5:
            break
        k, m = slab_errors(df)
        P["body"] = correct(P["body"], k.z_mm.values, (k.kick_cm2 - k.kick_model_cm2).values * 100,
                            lambda z: envelope_mm(z) + (3.0 if z < Z_M - 10 else 0.0))
        mb = m[m.z_mm < zs(M.s_tip) - 1]
        P["bullet"] = correct(P["bullet"], (mb.z_mm.values - Z_MT), (mb.mid_cm2 - mb.mid_model_cm2).values * 100,
                              lambda s: hold_mm(s / MM))

    # 4. final build
    FINAL = True
    print("final build", flush=True)
    vanes()
    body = centre_body()
    bullet()
    print("cone channel check", flush=True)
    ch_new = channel_check()
    print("drivers, assembly", flush=True)
    assembly()
    rho, _ = mouth_round()
    prof = write_profiles(rows_k, rows_m, ch_design, ch_new, rho)

    # 5. the official slices of the assembled model (cad/parts.json)
    (OUT / "parts.json").write_text(json.dumps(parts, indent=1) + "\n")
    print("slices", flush=True)
    df, meshes = SL.run(dz=10.0)
    SL.figure(df, meshes)
    final = summary_stats(df)
    print("  " + json.dumps(final), flush=True)
    RES.mkdir(parents=True, exist_ok=True)
    write_area_check(df, rows_k, rows_m)

    # 6. array
    print("array", flush=True)
    arr, boxes = array_check()
    tri = coarse_triangles([p["name"] for p in parts if p["category"] in ("cabinet", "filler", "print", "support")],
                           size=60.0)
    gmsh.finalize()
    arr["preview"] = dict(file="array.stl", triangles=write_array_stl(boxes, tri),
                          note="coarse mesh of the cabinet, wedges, fillers, supports and printed parts of each box, placed")
    bev = panel_bevels()
    dxf(baf, back, bev, ze["wedge_top"])
    ext = dict(width=round(2 * ohw(Z_M), 1), height_front=round(2 * (HH + T), 1), height_back=round(2 * (HH + T), 1),
               width_back=round(2 * ohw(Z_BACK), 1), depth=round(Z_M - Z_BACK, 1), overall_depth=round(Z_M - Z_BACK, 1))
    L = (Z_M - Z_WB) / math.cos(A)
    kk = [r for r in prof["kick_body"]["r_mm"]]
    cut = dict(
        units="mm",
        external_mm=ext,
        walls=dict(qty=2, ply_mm=T, slant_length_mm=round(L, 1), height_mm=round(2 * HH, 1),
                   inner_face=dict(back_edge=round(L, 1), mouth_edge=round(L, 1)),
                   outer_face=dict(back_edge=round(2 * HH, 1), mouth_edge=round(2 * HH, 1)),
                   blade_bevel_deg=0.0, edge_taper_deg=0.0, end_bevel_deg=round(K.half_angle_deg, 2),
                   dxf="dxf/side_wall.dxf",
                   note="Two side walls: 18 mm rectangles, both ends bevelled at the half angle and parallel so the "
                        "back and the mouth are square to the axis. Long edges square; they butt under the top and "
                        "over the bottom."),
        top_bottom=dict(qty=2, ply_mm=T, length_mm=round(Z_M - Z_WB, 1), back_width_mm=round(2 * ohw(Z_WB), 1),
                        mouth_width_mm=round(2 * ohw(Z_M), 1), dxf="dxf/top_bottom.dxf",
                        note="Flat trapezoids, all edges square: the side walls' outer faces run on from the "
                             "angled edges. Full depth, from the back panel to the mouth."),
        wedges=dict(qty=2, ply_mm=T, depth_at_baffle_mm=round(HH - hw(0), 1), length_mm=round(Z_M, 1),
                    panel_slant_length_mm=round(Z_M / math.cos(A), 1), panel_back_width_mm=round(2 * hw(0), 1),
                    panel_mouth_width_mm=round(2 * hw(Z_M), 1), feather_from_z_mm=round(ze["wedge_top"], 1),
                    ribs_per_wedge=len(WEDGE_RIBS), rib_x_mm=[round(f * hw(0), 1) for f in WEDGE_RIBS],
                    sealed_litres_each=round(K.wedge_litres / 2, 2), **bev,
                    dxf=["dxf/wedge_panel.dxf", "dxf/wedge_rib.dxf"],
                    note="Under the flat top and over the flat bottom, from the baffle front face to the mouth. An "
                         "18 mm sloping panel at the half angle is the horn wall; 18 mm side closures along the side "
                         "walls and ribs carry it from the flat panel. Glue and seal: the wedges are not vented."),
        baffle=dict(**baf, z_mm=[round(Z_BAF, 1), 0.0], dxf="dxf/baffle.dxf",
                    note="Glued between the side walls, top and bottom; the kick flange screws to its front face"),
        back=dict(**{k: v for k, v in back.items() if k != "screw_xy_mm"}, z_mm=[round(Z_BACK, 1), round(Z_WB, 1)],
                  dxf="dxf/back.dxf"),
        corner_filler=dict(qty=4, **fill),
        chamber=dict(available_litres=round(K.chamber_litres, 2),
                     box_litres=round(float(np.trapezoid([2 * hw(z) * 2 * HH for z in np.linspace(Z_WB, Z_BAF, 50)],
                                                        np.linspace(Z_WB, Z_BAF, 50))) / 1e6, 2),
                     note="Kick.chamber_litres: the full-height box between the baffle and the back, the baffle "
                          "cutout, the cone"),
        supports=dict(standoffs=dict(qty=4, diameter_mm=2 * STANDOFF_R, pcd_mm=round(2 * R_STANDOFF, 1),
                                     length_mm=round(Z_TURN + SOCKET, 1), duct_bore_mm=12,
                                     azimuth_deg=[45, 135, 225, 315]),
                      vanes=dict(qty=4, thickness_mm=VANE_T, chord_mm=VANE_CHORD, z_centre_mm=round(Z_VANE, 1)),
                      bullet_vanes=dict(qty=3, thickness_mm=BV_T, s_mm=[BV_S0 * MM, BV_S1 * MM])),
        centre_body=dict(**body, max_radius_mm=round(max(kk), 1)),
        cone_channel=dict(file="../results/cad/cone-channel.csv",
                          max_abs_diff_pct=round(max(abs(r["diff_pct"]) for r in ch_new), 2),
                          equation_shroud_max_diff_pct=round(max(r["equation_shroud_diff_pct"] for r in ch_new), 2),
                          arms_cm2=round(ch_new[0]["arms_cyl_cm2"], 2),
                          note="Open area on cylinders r = const between the cone and the shroud, less the basket "
                               "arms, projected normal to the cone, against the cone area inside r over the "
                               "compression. The equation shroud assumed 12 % arm blockage; the real arms block "
                               "about 2 %, so the shroud is lower by 4 to 6 mm."),
        plugs=dict(profiles="plug_profiles.json", verify_passes=history, final_slices=final,
                   mouth_edge_radius_mm=round(rho, 2)),
        stations_mm=dict(back=round(Z_BACK, 1), wall_back=round(Z_WB, 1), baffle_back=round(Z_BAF, 1), baffle_front=0.0,
                         throat=0.0, turn=round(Z_TURN, 1), motor_end=round(Z_MOTOR, 1),
                         mid_driver_back=round(Z_MF - MD.depth * MM, 1), mid_flange=round(Z_MF, 1),
                         mid_throat=round(Z_MT, 1), hf_exit=round(zs(M.s_exit), 1), hf_mouth=round(zs(M.s_tip), 1),
                         mouth=round(Z_M, 1), mid_mouth=round(Z_MM, 1), vanes=round(Z_VANE, 1)),
        array=dict(max_gap_mm=arr["max_gap_mm"], max_penetration_mm=arr["max_penetration_mm"],
                   max_overlap_cm3=arr["max_overlap_cm3_occ"]),
        array_spacers=dict(qty=0, max_thickness_mm=0.0, min_thickness_mm=0.0, pieces=[],
                           note="None: the boxes tile exactly (see array.json)."),
        print_files={p["name"]: p.get("print_files", []) for p in parts if p.get("print_files")},
        warnings=warnings,
    )
    stack = dict(splay_deg=0.0, boxes=3, row_pitch_mm=round(2 * (HH + T), 1),
                 note="A column of the 3 x 3 array: rows stack flat, flat tops on flat bottoms. The columns fan by "
                      f"{K.splay_deg:.0f} deg about a vertical axis (array.json).")
    (OUT / "parts.json").write_text(json.dumps(parts, indent=1) + "\n")
    (OUT / "cutlist.json").write_text(json.dumps(cut, indent=1) + "\n")
    (OUT / "stack.json").write_text(json.dumps(stack, indent=1) + "\n")
    (OUT / "array.json").write_text(json.dumps(arr, indent=1) + "\n")
    for f in TMP.glob("*"):
        f.unlink()
    TMP.rmdir()
    print(json.dumps(dict(slices=final, array=dict(max_gap_mm=arr["max_gap_mm"],
                                                   max_penetration_mm=arr["max_penetration_mm"],
                                                   corner_mismatch_mm=arr["max_corner_mismatch_mm"],
                                                   overlap_cm3=arr["max_overlap_cm3_occ"]),
                          external_mm=ext, pieces=sum(len(p.get("print_files", [])) for p in parts),
                          warnings=warnings), indent=1))
    print(f"done in {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
