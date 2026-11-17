"""Build every Sunflower v0.2 part as STEP and STL (millimetres) from sunflower.design.

Coordinates: z is the horn axis towards the listener, z = 0 is the 15 inch
cone rim and the back face of the baffle. x is to the right when facing the
cabinet, y is up.

Outputs (cad/):
  step/<part>.step, stl/<part>.stl     one file per part, in place
  print/<part>-<n>[-left|-right].stl   printable sections, each within a 250 mm bed
  sunflower-assembly.step               everything in place
  parts.json, cutlist.json, stack.json  part list, panel sizes and volumes, stacking angles
  dxf/<panel>.dxf                       flat patterns

Driver bodies are dimensioned proxies for renders and clearance checks.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import gmsh
import numpy as np

from sunflower import design as D

ROOT = D.ROOT
OUT = ROOT / "cad"
MM = 1000.0
BED = 250.0

LOW, MID, TW = D.LOW, D.MID, D.TW
T = D.PLY * MM
BAF = LOW.baffle * MM
WM, HM, WT = LOW.mouth_width * MM, LOW.mouth_height * MM, LOW.throat_width * MM
Z_M = LOW.mouth_z * MM
Z_BI = -LOW.back_box * MM
Z_BO = Z_BI - T
BH = LOW.back_half_height * MM

Z_MID_TH = Z_M - MID.length * MM
Z_MID_BACK = Z_MID_TH - MID.driver_depth * MM
Z_CAN = LOW.can_start * MM
R_CAN = MID.can_diameter / 2 * MM
R_CONE_MID = MID.cone_diameter / 2 * MM
Z_ROD = Z_M - 0.5 * MID.length * MM
ROD_R = 6.0

occ = None
parts: list[dict] = []


# ------------------------------------------------------------------ helpers
def start(name):
    global occ
    gmsh.clear()
    gmsh.model.add(name)
    occ = gmsh.model.occ


def revolve(profile):
    pts = []
    for r, z in profile:
        if not pts or math.hypot(r - pts[-1][0], z - pts[-1][1]) > 1e-6:
            pts.append((max(r, 0.0), z))
    if math.hypot(pts[0][0] - pts[-1][0], pts[0][1] - pts[-1][1]) < 1e-6:
        pts.pop()
    tags = [occ.addPoint(r, 0, z) for r, z in pts]
    lines = [occ.addLine(tags[i], tags[(i + 1) % len(tags)]) for i in range(len(tags))]
    surf = occ.addPlaneSurface([occ.addCurveLoop(lines)])
    out = occ.revolve([(2, surf)], 0, 0, 0, 0, 0, 1, 2 * math.pi)
    return [e for e in out if e[0] == 3]


def ellipse(r0, z0, z1, n=24):
    t = np.linspace(0, 1, n)
    return [(r0 * math.sqrt(max(0.0, 1 - s * s)), z0 + (z1 - z0) * s) for s in t]


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


def hexa(inner, outer):
    """Slab from two matching quads (inner face, outer face), each 4 xyz points in order."""
    a, b = inner, outer
    faces = [a, b[::-1]] + [[a[i], a[(i + 1) % 4], b[(i + 1) % 4], b[i]] for i in range(4)]
    return poly_volume(faces)


def offset(quad, n, t):
    return [tuple(p[i] + n[i] * t for i in range(3)) for p in quad]


def unit(v):
    m = math.sqrt(sum(x * x for x in v))
    return tuple(x / m for x in v)


def clean():
    occ.synchronize()
    vols = gmsh.model.getEntities(3)
    keep = set(gmsh.model.getBoundary(vols, combined=False, oriented=False, recursive=True))
    for dim in (2, 1):
        keep |= set(gmsh.model.getBoundary([e for e in keep if e[0] == dim + 1], combined=False,
                                           oriented=False, recursive=False))
    stray = [e for d in (0, 1, 2) for e in gmsh.model.getEntities(d) if e not in keep]
    if stray:
        occ.remove(stray)
        occ.synchronize()


def mesh_write(path):
    gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", 36)
    gmsh.option.setNumber("Mesh.MeshSizeMax", 12)
    gmsh.option.setNumber("Mesh.MeshSizeMin", 0.5)
    gmsh.model.mesh.clear()
    gmsh.model.mesh.generate(2)
    gmsh.option.setNumber("Mesh.Binary", 1)
    gmsh.write(str(path))


def save(name, material, explode=(0, 0, 0), group="", note="", printable=False):
    clean()
    (OUT / "step").mkdir(parents=True, exist_ok=True)
    (OUT / "stl").mkdir(parents=True, exist_ok=True)
    gmsh.model.removePhysicalGroups()
    gmsh.write(str(OUT / "step" / f"{name}.step"))
    mesh_write(OUT / "stl" / f"{name}.stl")
    vol = sum(occ.getMass(*d) for d in gmsh.model.getEntities(3))
    xyz = gmsh.model.mesh.getNodes()[1].reshape(-1, 3)
    bb = [*xyz.min(0), *xyz.max(0)]
    rec = dict(name=name, material=material, qty=1, explode_mm=list(explode), group=group,
               volume_cm3=round(vol / 1000, 1), bbox_mm=[round(v, 1) for v in bb], note=note)
    if printable:
        rec["print_files"] = split_for_print(name, bb)
    parts.append(rec)


def split_for_print(name, bb):
    """Cut a part into z-sections of at most 220 mm, and into halves or quarters to fit the bed.

    A half of a round part is still its full diameter in one direction, so
    parts much wider than the bed are cut into quarters.
    """
    (OUT / "print").mkdir(parents=True, exist_ok=True)
    z0, z1 = bb[2], bb[5]
    n = max(1, math.ceil((z1 - z0) / 220.0))
    wide = max(bb[3] - bb[0], bb[4] - bb[1]) > BED
    if wide:         # quarters: each quadrant is half the diameter square
        cuts = [(f"q{i + 1}", xa, xb, ya, yb) for i, (xa, xb, ya, yb) in
                enumerate(((0, 1e4, 0, 1e4), (-1e4, 0, 0, 1e4), (-1e4, 0, -1e4, 0), (0, 1e4, -1e4, 0)))]
    else:
        cuts = [("", -1e4, 1e4, -1e4, 1e4)]
    files = []
    for k in range(n):
        za, zb = z0 + (z1 - z0) * k / n, z0 + (z1 - z0) * (k + 1) / n
        for side, xa, xb, ya, yb in cuts:
            gmsh.model.add(f"{name}-{k}-{side}")
            o = gmsh.model.occ
            src = o.importShapes(str(OUT / "step" / f"{name}.step"))
            box = o.addBox(xa, ya, za, xb - xa, yb - ya, zb - za)
            res, _ = o.intersect(src, [(3, box)])
            o.synchronize()
            if res:
                fn = f"{name}-{k + 1}" + (f"-{side}" if side else "") + ".stl"
                mesh_write(OUT / "print" / fn)
                files.append(fn)
            gmsh.model.remove()
    gmsh.model.setCurrent(name)
    return files


def y_in(z):
    """Inside face of the shell top at z (mm)."""
    return LOW.shell_half_height(z / MM) * MM


# ------------------------------------------------------------------ cabinet shell
def shell():
    angle = math.atan((y_in(Z_M) - y_in(Z_BI)) / (Z_M - Z_BI))
    yo = lambda z: y_in(z) + T / math.cos(angle)  # noqa: E731
    for sx, name in ((1, "side_right"), (-1, "side_left")):
        start(name)
        x0 = sx * WM / 2
        inner = [(x0, -yo(Z_BO), Z_BO), (x0, yo(Z_BO), Z_BO), (x0, yo(Z_M), Z_M), (x0, -yo(Z_M), Z_M)]
        hexa(inner, offset(inner, (sx, 0, 0), T))
        save(name, "ply", explode=(sx * 300, 0, 0), group="shell",
             note=f"vertical side, {Z_M - Z_BO:.0f} long, {2*yo(Z_BO):.0f} tall at the back, {2*yo(Z_M):.0f} at the front")
    for sy, name in ((1, "top"), (-1, "bottom")):
        start(name)
        inner = [(-WM / 2, sy * y_in(Z_BO), Z_BO), (WM / 2, sy * y_in(Z_BO), Z_BO),
                 (WM / 2, sy * y_in(Z_M), Z_M), (-WM / 2, sy * y_in(Z_M), Z_M)]
        n = unit((0, sy * math.cos(angle), -math.sin(angle)))
        hexa(inner, offset(inner, n, T))
        save(name, "ply", explode=(0, sy * 300, 0), group="shell",
             note=f"angled {math.degrees(angle):.1f} deg, {WM:.0f} wide, {(Z_M - Z_BO) / math.cos(angle):.0f} long")
    start("back")
    occ.addBox(-WM / 2, -y_in(Z_BI), Z_BO, WM, 2 * y_in(Z_BI), T)
    save("back", "ply", explode=(0, 0, -400), group="shell", note="terminal cup")
    start("baffle")
    h = y_in(0)
    b = occ.addBox(-WM / 2, -h, 0, WM, 2 * h, BAF)
    holes = [(3, occ.addCylinder(0, 0, -1, 0, 0, BAF + 2, LOW.cutout_radius * MM))]
    for sx in (-1, 1):
        for sy in (-1, 1):
            holes.append((3, occ.addCylinder(sx * (WM / 2 - 45), sy * (h - 35), -1, 0, 0, BAF + 2, 20)))
    occ.cut([(3, b)], holes)
    save("baffle", "ply", explode=(0, 0, -250), group="shell",
         note=f"{WM:.0f} x {2*h:.0f} x {BAF:.0f}, {2*LOW.cutout_radius*MM:.0f} mm cutout, four 40 mm chamber ports")
    return math.degrees(angle)


def horn_walls():
    zb, zm = BAF, Z_M
    W = lambda z: LOW.width(z / MM) * MM  # noqa: E731
    H = lambda z: LOW.height(z / MM) * MM  # noqa: E731
    out = {}
    for sx, name in ((1, "horn_wall_right"), (-1, "horn_wall_left")):
        start(name)
        inner = [(sx * W(zb) / 2, -H(zb) / 2, zb), (sx * W(zb) / 2, H(zb) / 2, zb),
                 (sx * W(zm) / 2, H(zm) / 2, zm), (sx * W(zm) / 2, -H(zm) / 2, zm)]
        n = unit((sx * (zm - zb), 0, -(W(zm) - W(zb)) / 2))
        hexa(inner, offset(inner, n, T))
        save(name, "ply", explode=(sx * 170, 0, 0), group="horn", note="internal side wall")
        out["side_wall"] = dict(throat_edge_mm=H(zb), mouth_edge_mm=H(zm),
                                slope_length_mm=math.hypot(zm - zb, (W(zm) - W(zb)) / 2),
                                flare_deg=math.degrees(math.atan((W(zm) - W(zb)) / 2 / (zm - zb))))
    for sy, name in ((1, "horn_wall_top"), (-1, "horn_wall_bottom")):
        start(name)
        inner = [(-W(zb) / 2 - T, sy * H(zb) / 2, zb), (W(zb) / 2 + T, sy * H(zb) / 2, zb),
                 (W(zm) / 2 + T, sy * H(zm) / 2, zm), (-W(zm) / 2 - T, sy * H(zm) / 2, zm)]
        n = unit((0, sy * (zm - zb), -(H(zm) - H(zb)) / 2))
        hexa(inner, offset(inner, n, T))
        save(name, "ply", explode=(0, sy * 170, 0), group="horn", note="internal top or bottom wall")
        out["top_wall"] = dict(throat_edge_mm=W(zb) + 2 * T, mouth_edge_mm=W(zm) + 2 * T,
                               slope_length_mm=math.hypot(zm - zb, (H(zm) - H(zb)) / 2),
                               flare_deg=math.degrees(math.atan((H(zm) - H(zb)) / 2 / (zm - zb))))
    return out


def corner_fillers():
    c0 = LOW.chamfer(LOW.baffle) * MM
    zb, zm = BAF, Z_M
    W0, H0 = WT, WT
    for sx, sy, name in ((1, 1, "tr"), (-1, 1, "tl"), (-1, -1, "bl"), (1, -1, "br")):
        start(f"corner_{name}")
        a = (sx * W0 / 2, sy * H0 / 2, zb)
        b = (sx * (W0 / 2 - c0), sy * H0 / 2, zb)
        c = (sx * W0 / 2, sy * (H0 / 2 - c0), zb)
        apex = (sx * WM / 2, sy * HM / 2, zm)
        poly_volume([[a, b, c], [a, b, apex], [b, c, apex], [c, a, apex]])
        save(f"corner_{name}", "print_orange", explode=(sx * 110, sy * 110, 0), group="horn",
             note="corner filler: 12 mm ply face on three ribs, painted")
    return dict(throat_leg_mm=c0, face_base_mm=c0 * math.sqrt(2))


# ------------------------------------------------------------------ centre body and mid
def centre_body():
    """One piece from the 15 inch cone (rear face follows the cone) round the motor to the mid can."""
    b = LOW.body
    a, cd = b.cone_radius * MM, b.cone_depth * MM
    clear = (b.xmax + LOW.gap) * MM
    rm3 = b.motor_radius * MM + 3
    me = LOW.motor_end * MM + 5
    r_th = LOW.throat_radius * MM
    zface = lambda r: cd * (1 - r / a) + clear  # noqa: E731
    prof = [(0, me), (rm3, me), (rm3, zface(rm3))]
    prof += [(r, zface(r)) for r in np.linspace(rm3, r_th, 12)[1:]]
    zs = np.linspace(max(zface(r_th), BAF), Z_CAN, 60)
    prof += [(LOW.inner_radius(z / MM) * MM, z) for z in zs]
    prof += [(R_CAN, Z_CAN), (R_CAN, Z_MID_BACK), (0, Z_MID_BACK)]
    start("centre_body")
    revolve(prof)
    save("centre_body", "print_black", explode=(0, 0, 150), group="low", printable=True,
         note="one piece: cone-following phase plug, motor cover, fairing, mid rear cap. Clears the motor by 3 mm")


def mid_parts():
    zs = np.linspace(0, MID.length, 80)
    bore = [(MID.radius(s) * MM, Z_MID_TH + s * MM) for s in zs]
    start("mid_horn")
    revolve([(R_CONE_MID + 1.5, Z_MID_TH)] + bore[1:] + [(R_CAN, Z_M), (R_CAN, Z_MID_TH)])
    save("mid_horn", "print_black", explode=(0, 0, 330), group="mid", printable=True,
         note=f"{MID.law} open-area law round the phase plug, {MID.mouth_diameter*1000:.0f} mm mouth, {MID.length*1000:.0f} mm long")
    start("mid_sleeve")
    r_in = MID.frame_diameter / 2 * MM + 2
    revolve([(r_in, Z_MID_BACK), (R_CAN, Z_MID_BACK), (R_CAN, Z_MID_TH), (r_in, Z_MID_TH)])
    save("mid_sleeve", "print_black", explode=(0, 0, 240), group="mid", printable=True,
         note=f"holds the {MID.frame_diameter*1000:.0f} mm driver frame")
    # phase plug with the HF horn inside it
    nose = MID.nose_length * MM
    s0 = Z_MID_TH
    se, st = s0 + MID.s_exit * MM, s0 + MID.s_tip * MM
    rb = TW.body_radius * MM + 0.5
    prof = [(0, s0 - nose)] + [(r, z) for r, z in reversed(ellipse(MID.plug_radius * MM, s0, s0 - nose))]
    ss = np.linspace(MID.s_exit, MID.s_tip, 20)
    prof += [(MID.plug(s) * MM, s0 + s * MM) for s in ss]
    prof += [(MID.hf_horn_radius(s) * MM, s0 + s * MM) for s in ss[::-1]]
    prof += [(rb, se), (rb, se - TW.depth_with_terminals * MM), (0, se - TW.depth_with_terminals * MM)]
    start("phase_plug")
    body = revolve(prof)
    vanes = []
    rt = MID.radius(0) * MM
    for k in range(3):
        v = occ.addBox(MID.plug_radius * MM - 1, -1.25, s0, rt + 6 - MID.plug_radius * MM, 2.5, 0.6 * MID.s_exit * MM)
        occ.rotate([(3, v)], 0, 0, 0, 0, 0, 1, k * 2 * math.pi / 3 + math.pi / 2)
        vanes.append((3, v))
    occ.fuse(body, vanes)
    save("phase_plug", "print_grey", explode=(0, 0, 520), group="mid", printable=True,
         note=f"{2*MID.plug_radius*1000:.0f} mm plug round the {TW.model}, tapering to its printed HF horn "
              f"({MID.hf_exit_diameter*1000:.0f} to {2*MID.hf_mouth_radius*1000:.0f} mm over {MID.hf_len*1000:.0f} mm)")


def rods():
    W = LOW.width(Z_ROD / MM) * MM
    H = LOW.height(Z_ROD / MM) * MM
    for name, (sx, sy) in dict(rod_right=(1, 0), rod_left=(-1, 0), rod_top=(0, 1), rod_bottom=(0, -1)).items():
        start(name)
        reach = (W if sx else H) / 2 + T * 0.8
        length = reach - (R_CAN - 8)
        occ.addCylinder(sx * (R_CAN - 8), sy * (R_CAN - 8), Z_ROD, sx * length, sy * length, 0, ROD_R)
        save(name, "aluminium", explode=(sx * 200, sy * 200, 0), group="mid", note=f"12 mm aluminium tube, {length:.0f} mm")


def driver_proxies():
    b = LOW.body
    a = b.cone_radius * MM
    env = D.LOW_RECORD.get("physical_envelope_mm", {})
    flange_d, flange_t = env.get("overall_diameter", 2 * a + 40), env.get("flange_thickness", 11)
    start("drv_low_frame")
    flange = occ.addCylinder(0, 0, -flange_t, 0, 0, flange_t, flange_d / 2)
    hole = occ.addCylinder(0, 0, -flange_t - 1, 0, 0, flange_t + 3, a + 2)
    occ.cut([(3, flange)], [(3, hole)])
    zc = b.cone_depth * MM
    spokes = []
    for k in range(8):
        ang = k * math.pi / 4 + math.pi / 8
        x0, y0 = math.cos(ang) * (a + 4), math.sin(ang) * (a + 4)
        x1, y1 = math.cos(ang) * (b.motor_radius * MM - 5), math.sin(ang) * (b.motor_radius * MM - 5)
        spokes.append((3, occ.addCylinder(x0, y0, 0, x1 - x0, y1 - y0, zc, 6)))
    motor = occ.addCylinder(0, 0, zc, 0, 0, b.motor_length * MM, b.motor_radius * MM)
    occ.fuse([(3, motor)], spokes)
    save("drv_low_frame", "driver_metal", explode=(0, 0, -150), group="low", note=f"proxy of the {D.label(D.LOW_RECORD)}")
    start("drv_low_cone")
    revolve([(a, 0), (40, zc), (36, zc), (a - 3, 0)])
    save("drv_low_cone", "driver_cone", explode=(0, 0, -150), group="low", note="proxy cone")
    fr = MID.frame_diameter / 2 * MM
    start("drv_mid")
    frame = occ.addCylinder(0, 0, Z_MID_TH - 6, 0, 0, 6, fr)
    can = occ.addCylinder(0, 0, Z_MID_BACK, 0, 0, Z_MID_TH - 6 - Z_MID_BACK, 0.6 * fr)
    occ.fuse([(3, frame)], [(3, can)])
    save("drv_mid", "driver_metal", explode=(0, 0, 240), group="mid", note=f"proxy of the {D.label(D.MID_RECORD)}")
    start("drv_mid_cone")
    rc = R_CONE_MID
    revolve([(rc, Z_MID_TH), (0.3 * rc, Z_MID_TH - 0.4 * rc), (0.26 * rc, Z_MID_TH - 0.4 * rc), (rc - 3, Z_MID_TH)])
    save("drv_mid_cone", "driver_cone", explode=(0, 0, 240), group="mid", note="proxy cone")
    se = Z_MID_TH + MID.s_exit * MM
    rb = TW.body_radius * MM
    start("drv_hf")
    prof = [(0, se - TW.body_depth * MM), (rb, se - TW.body_depth * MM), (rb, se - 2), (MID.hf_exit_diameter * 500 + 2, se),
            (0, se)]
    revolve(prof)
    save("drv_hf", "driver_metal", explode=(0, 0, 620), group="mid",
         note=f"{TW.model}: {TW.body_diameter*1000:.0f} mm body cylinder, {TW.body_depth*1000:.1f} mm deep")


# ------------------------------------------------------------------ outputs
def dxf(walls):
    import ezdxf
    (OUT / "dxf").mkdir(exist_ok=True)

    def write(name, pts, label, circles=()):
        doc = ezdxf.new("R2010")
        msp = doc.modelspace()
        msp.add_lwpolyline(pts, close=True)
        for (cx, cy, r) in circles:
            msp.add_circle((cx, cy), r)
        msp.add_text(label, dxfattribs={"height": 12}).set_placement((10, -30))
        doc.saveas(OUT / "dxf" / f"{name}.dxf")

    for key, w in walls.items():
        a, b, s = w["throat_edge_mm"], w["mouth_edge_mm"], w["slope_length_mm"]
        write(f"horn_{key}", [(-a / 2, 0), (a / 2, 0), (b / 2, s), (-b / 2, s)],
              f"Horn {key.replace('_', ' ')} x2, inner face, 18 mm ply, flare {w['flare_deg']:.1f} deg")
    h = y_in(0)
    write("baffle", [(-WM / 2, -h), (WM / 2, -h), (WM / 2, h), (-WM / 2, h)],
          "Baffle 25 mm ply", circles=[(0, 0, LOW.cutout_radius * MM)] +
          [(sx * (WM / 2 - 45), sy * (h - 35), 20) for sx in (-1, 1) for sy in (-1, 1)])
    write("side", [(0, -y_in(Z_BO) - T), (Z_M - Z_BO, -y_in(Z_M) - T), (Z_M - Z_BO, y_in(Z_M) + T), (0, y_in(Z_BO) + T)],
          "Side x2, 18 mm ply")


def assembly():
    start("assembly")
    for p in parts:
        occ.importShapes(str(OUT / "step" / f"{p['name']}.step"))
    occ.synchronize()
    gmsh.write(str(OUT / "sunflower-assembly.step"))


def main():
    OUT.mkdir(exist_ok=True)
    for sub in ("step", "stl", "print"):
        for f in (OUT / sub).glob("*"):
            f.unlink()
    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 0)
    gmsh.option.setString("Geometry.OCCTargetUnit", "MM")
    shell_angle = shell()
    walls = horn_walls()
    corner = corner_fillers()
    centre_body()
    mid_parts()
    rods()
    driver_proxies()
    assembly()
    gmsh.finalize()
    dxf(walls)
    avail = LOW.chamber_litres()
    cut = dict(walls=walls, corner_filler=corner, shell_angle_deg=shell_angle,
               chamber=dict(available_litres=avail, target_litres=D.LOW_REAR_LITRES,
                            fill_litres=max(0.0, avail - D.LOW_REAR_LITRES)),
               external_mm=dict(width=WM + 2 * T, height_front=HM + 2 * T, height_back=2 * (BH + T),
                                depth=Z_M - Z_BO),
               stations_mm=dict(back=Z_BO, baffle_back=0, baffle_front=BAF, throat=BAF,
                                motor_end=LOW.motor_end * MM, can_back=Z_CAN, mid_driver_back=Z_MID_BACK,
                                mid_throat=Z_MID_TH, hf_exit=Z_MID_TH + MID.s_exit * MM,
                                hf_mouth=Z_MID_TH + MID.s_tip * MM, mouth=Z_M, support_rods=Z_ROD),
               print_files={p["name"]: p.get("print_files", []) for p in parts if p.get("print_files")})
    stack = dict(splay_deg=LOW.splay_deg, boxes=3,
                 note="Each box sits on the one below with its bottom panel on the neighbour's top panel; "
                      "the angle between neighbours is the splay.",
                 pivot_mm=dict(y=HM / 2 + T, z=Z_M))
    (OUT / "parts.json").write_text(json.dumps(parts, indent=1) + "\n")
    (OUT / "cutlist.json").write_text(json.dumps(cut, indent=1) + "\n")
    (OUT / "stack.json").write_text(json.dumps(stack, indent=1) + "\n")
    print(json.dumps(cut, indent=1))


if __name__ == "__main__":
    main()
