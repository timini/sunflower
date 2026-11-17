"""Build every Sunflower part as STEP and STL (millimetres) from sunflower.design.

Coordinates: z is the horn axis towards the listener, z = 0 is the cone rim
of the 15 inch driver and the back face of the baffle. x is to the right
when facing the cabinet, y is up.

Outputs (cad/):
  step/<part>.step, stl/<part>.stl     one file per unique part
  sunflower-assembly.step               everything in place
  parts.json                            part list: material, quantity, explode vector
  dxf/<panel>.dxf                       flat patterns for CNC or a printed template
  cutlist.json                          panel sizes, bevels, rear chamber volume

Driver bodies are dimensioned proxies for renders and clearance checks,
not models of the real drivers.
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

LOW, MID, TW = D.LOW, D.MID, D.TW
T = D.PLY * MM
BAF = D.BAFFLE * MM
WM = LOW.mouth_width * MM
WT = LOW.throat_width * MM
Z_MOUTH = LOW.mouth_z * MM
Z_BAFFLE_FRONT = BAF
W_EXT = WM + 2 * T
Z_BACK_IN = -D.REAR_CHAMBER_DEPTH * MM
Z_BACK_OUT = Z_BACK_IN - T

# mid can stations
Z_MID_THROAT = Z_MOUTH - MID.length * MM
Z_MID_RIM = Z_MID_THROAT - MID.adapter_length * MM
Z_MID_BACK = Z_MID_RIM - MID.driver_depth * MM
Z_CAN = LOW.can_start * MM
Z_TAIL = LOW.tail_start * MM
R_CAN = MID.can_diameter / 2 * MM
Z_TW_FACE = Z_MID_THROAT + MID.tweeter_face * MM
Z_TW_BACK = Z_TW_FACE - TW.body_depth * MM
Z_TW_TERM = Z_TW_FACE - TW.depth_with_terminals * MM
R_PLUG = MID.plug_radius * MM
R_CONE_MID = math.sqrt(MID.sd / math.pi) * MM
NOSE = min(0.3 * R_CONE_MID, 30.0)     # phase plug nose behind the mid throat, towards the dust cap
Z_ROD = 0.5 * (Z_CAN + Z_MOUTH)
ROD_R = 6.0           # 12 mm aluminium tube, 10 mm bore for cables

occ = None
parts: list[dict] = []


# ------------------------------------------------------------------ helpers
def start(name):
    global occ
    gmsh.clear()
    gmsh.model.add(name)
    occ = gmsh.model.occ


def revolve(profile):
    """Solid of revolution about z from a closed (r, z) polygon."""
    pts = []
    for r, z in profile:
        if not pts or math.hypot(r - pts[-1][0], z - pts[-1][1]) > 1e-6:
            pts.append((r, z))
    if math.hypot(pts[0][0] - pts[-1][0], pts[0][1] - pts[-1][1]) < 1e-6:
        pts.pop()
    tags = [occ.addPoint(r, 0, z) for r, z in pts]
    lines = [occ.addLine(tags[i], tags[(i + 1) % len(tags)]) for i in range(len(tags))]
    surf = occ.addPlaneSurface([occ.addCurveLoop(lines)])
    out = occ.revolve([(2, surf)], 0, 0, 0, 0, 0, 1, 2 * math.pi)
    return [e for e in out if e[0] == 3]


def ellipse(r0, z0, z1, n=24, closing=True):
    """Quarter-ellipse from radius r0 at z0 to the axis at z1."""
    t = np.linspace(0, 1, n)
    return [(r0 * math.sqrt(max(0.0, 1 - s * s)), z0 + (z1 - z0) * s) for s in t]


def poly_volume(faces_pts):
    """Closed polyhedron from a list of planar faces (each a list of xyz)."""
    cache = {}

    def P(p):
        key = tuple(round(v, 6) for v in p)
        if key not in cache:
            cache[key] = occ.addPoint(*p)
        return cache[key]

    edges = {}

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


def save(name, dims, material, qty=1, explode=(0, 0, 0), group="", note=""):
    occ.synchronize()
    # drop construction wires and faces that are not part of a solid
    vols = gmsh.model.getEntities(3)
    keep = set(gmsh.model.getBoundary(vols, combined=False, oriented=False, recursive=True))
    for dim in (2, 1):
        keep |= set(gmsh.model.getBoundary([e for e in keep if e[0] == dim + 1], combined=False,
                                           oriented=False, recursive=False))
    stray = [e for d in (0, 1, 2) for e in gmsh.model.getEntities(d) if e not in keep]
    if stray:
        occ.remove(stray)
        occ.synchronize()
    (OUT / "step").mkdir(parents=True, exist_ok=True)
    (OUT / "stl").mkdir(parents=True, exist_ok=True)
    gmsh.model.removePhysicalGroups()
    gmsh.write(str(OUT / "step" / f"{name}.step"))
    gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", 36)
    gmsh.option.setNumber("Mesh.MeshSizeMax", 12)
    gmsh.option.setNumber("Mesh.MeshSizeMin", 0.5)
    gmsh.model.mesh.generate(2)
    gmsh.option.setNumber("Mesh.Binary", 1)
    gmsh.write(str(OUT / "stl" / f"{name}.stl"))
    vol = sum(occ.getMass(*d) for d in gmsh.model.getEntities(3))
    xyz = gmsh.model.mesh.getNodes()[1].reshape(-1, 3)
    bb = [*xyz.min(0), *xyz.max(0)]
    parts.append(dict(name=name, material=material, qty=qty, explode_mm=list(explode), group=group,
                      volume_cm3=round(vol / 1000, 1), bbox_mm=[round(v, 1) for v in bb], note=note))


# ------------------------------------------------------------------ cabinet
def cabinet():
    h = W_EXT / 2
    depth = Z_MOUTH - Z_BACK_OUT
    start("side_left")
    occ.addBox(-h, -h, Z_BACK_OUT, T, W_EXT, depth)
    save("side_left", None, "ply", explode=(-260, 0, 0), group="cabinet",
         note=f"{W_EXT:.0f} x {depth:.0f} x {T:.0f} mm")
    start("side_right")
    occ.addBox(h - T, -h, Z_BACK_OUT, T, W_EXT, depth)
    save("side_right", None, "ply", explode=(260, 0, 0), group="cabinet",
         note=f"{W_EXT:.0f} x {depth:.0f} x {T:.0f} mm")
    start("top")
    occ.addBox(-h + T, h - T, Z_BACK_OUT, WM, T, depth)
    save("top", None, "ply", explode=(0, 260, 0), group="cabinet", note=f"{WM:.0f} x {depth:.0f} x {T:.0f} mm")
    start("bottom")
    occ.addBox(-h + T, -h, Z_BACK_OUT, WM, T, depth)
    save("bottom", None, "ply", explode=(0, -260, 0), group="cabinet", note=f"{WM:.0f} x {depth:.0f} x {T:.0f} mm")
    start("back")
    occ.addBox(-WM / 2, -WM / 2, Z_BACK_OUT, WM, WM, T)
    save("back", None, "ply", explode=(0, 0, -420), group="cabinet", note=f"{WM:.0f} x {WM:.0f} x {T:.0f} mm, terminal cup")
    start("baffle")
    b = occ.addBox(-WM / 2, -WM / 2, 0, WM, WM, BAF)
    hole = occ.addCylinder(0, 0, -1, 0, 0, BAF + 2, LOW.cutout_radius * MM)
    occ.cut([(3, b)], [(3, hole)])
    save("baffle", None, "ply", explode=(0, 0, -260), group="cabinet",
         note=f"{WM:.0f} x {WM:.0f} x {BAF:.0f} mm, {2*LOW.cutout_radius*MM:.0f} mm cutout")


def horn_panels():
    tan_a = (WM - WT) / 2 / (LOW.length * MM)
    cos_a = 1 / math.sqrt(1 + tan_a**2)
    z0, z1 = Z_BAFFLE_FRONT, Z_MOUTH
    ext = 60.0

    def square_wire(half, z):
        pts = [occ.addPoint(sx * half, sy * half, z) for sx, sy in ((1, 1), (-1, 1), (-1, -1), (1, -1))]
        lines = [occ.addLine(pts[i], pts[(i + 1) % 4]) for i in range(4)]
        return occ.addCurveLoop(lines)

    def frustum(offset):
        za, zb = z0 - ext, z1 + ext
        ha = WT / 2 - tan_a * ext + offset / cos_a
        hb = WM / 2 + tan_a * ext + offset / cos_a
        return occ.addThruSections([square_wire(ha, za), square_wire(hb, zb)], makeSolid=True, makeRuled=True)

    names = dict(right=(1, 0), left=(-1, 0), top=(0, 1), bottom=(0, -1))
    for side, (sx, sy) in names.items():
        start(f"horn_wall_{side}")
        outer = frustum(T)
        inner = frustum(0.0)
        shell, _ = occ.cut(outer, inner)
        R = 2000
        if sx:
            tri = [(0, 0), (sx * R, R), (sx * R, -R)]
        else:
            tri = [(0, 0), (R, sy * R), (-R, sy * R)]
        pts = [occ.addPoint(x, y, z0) for x, y in tri]
        loop = occ.addCurveLoop([occ.addLine(pts[i], pts[(i + 1) % 3]) for i in range(3)])
        wedge = occ.extrude([(2, occ.addPlaneSurface([loop]))], 0, 0, z1 - z0)
        wedge = [e for e in wedge if e[0] == 3]
        clip = occ.addBox(-WM / 2, -WM / 2, z0, WM, WM, z1 - z0)
        panel, _ = occ.intersect(shell, wedge, removeObject=True, removeTool=True)
        occ.intersect(panel, [(3, clip)])
        save(f"horn_wall_{side}", None, "ply", explode=(sx * 170, sy * 170, 0), group="horn")
    dihedral = math.degrees(math.pi - math.acos(math.sin(math.atan(tan_a)) ** 2))
    return dict(flare_half_angle_deg=math.degrees(math.atan(tan_a)),
                long_edge_bevel_deg=90 - dihedral / 2, end_bevel_deg=math.degrees(math.atan(tan_a)),
                slope_length_mm=(LOW.length * MM) / cos_a)


def corner_fillers():
    c0 = LOW.chamfer(LOW.baffle) * MM
    z0, z1 = Z_BAFFLE_FRONT, Z_MOUTH
    out = []
    for sx, sy, name in ((1, 1, "tr"), (-1, 1, "tl"), (-1, -1, "bl"), (1, -1, "br")):
        start(f"corner_{name}")
        a = (sx * WT / 2, sy * WT / 2, z0)
        b = (sx * (WT / 2 - c0), sy * WT / 2, z0)
        c = (sx * WT / 2, sy * (WT / 2 - c0), z0)
        apex = (sx * WM / 2, sy * WM / 2, z1)
        poly_volume([[a, b, c], [a, b, apex], [b, c, apex], [c, a, apex]])
        save(f"corner_{name}", None, "print_orange", explode=(sx * 120, sy * 120, 0), group="horn",
             note="corner filler: 12 mm ply face on three ribs, painted")
    leg = c0
    hyp = leg * math.sqrt(2)
    slant = math.dist(((WT / 2 - c0 / 2), (WT / 2 - c0 / 2), z0), (WM / 2, WM / 2, z1))
    return dict(throat_leg_mm=leg, face_base_mm=hyp, face_height_mm=slant)


# ------------------------------------------------------------------ centre body
def low_nose():
    b = LOW.body
    z_motor = (b.cone_depth + b.motor_length) * MM
    rm = b.motor_radius * MM
    start("low_nose")
    prof = [(0, z_motor - 15), (rm - 3, z_motor - 15), (rm, z_motor)] + ellipse(rm, z_motor, z_motor + b.nose_length * MM)[1:]
    nose = revolve(prof)
    vent = occ.addCylinder(0, 0, z_motor - 20, 0, 0, b.nose_length * MM + 40, 10)
    occ.cut(nose, [(3, vent)])
    save("low_nose", None, "print_black", explode=(0, 0, 160), group="low",
         note=f"bonds to the {D.label(D.LOW_RECORD)} back plate, 15 mm spigot, 20 mm bore keeps the pole vent open")


def mid_can():
    rt = MID.outer_throat_radius * MM
    zs = np.linspace(0, MID.length, 60)
    bore = [(MID.radius(z) * MM, Z_MID_THROAT + z * MM) for z in zs]
    r_lead = R_CONE_MID + 1.5
    # A: horn with a cone-following lead-in from the cone rim to the annular throat
    start("mid_horn")
    prof = [(r_lead, Z_MID_RIM)] + bore + [(R_CAN, Z_MOUTH), (R_CAN, Z_MID_RIM)]
    revolve(prof)
    save("mid_horn", None, "print_black", explode=(0, 0, 330), group="mid",
         note=f"{MID.profile} bore, {MID.mouth_diameter*1000:.0f} mm mouth, {MID.length*1000:.0f} mm long, "
              f"lead-in to the {2*R_CONE_MID:.0f} mm cone at 1.5 mm clearance")
    # B: driver sleeve
    r_in = MID.frame_diameter / 2 * MM + 2
    start("mid_sleeve")
    revolve([(r_in, Z_MID_BACK), (R_CAN, Z_MID_BACK), (R_CAN, Z_MID_RIM), (r_in, Z_MID_RIM)])
    save("mid_sleeve", None, "print_black", explode=(0, 0, 230), group="mid",
         note=f"holds the {MID.frame_diameter*1000:.0f} mm driver frame")
    # C: rear cap + elliptical tail, hollow: part of the mid rear volume
    start("mid_tail")
    prof = [(0, Z_MID_BACK), (R_CAN, Z_MID_BACK), (R_CAN, Z_CAN)]
    prof += [(r, z) for r, z in ellipse(R_CAN, Z_CAN, Z_TAIL)][1:]
    revolve(prof)
    save("mid_tail", None, "print_black", explode=(0, 0, 130), group="mid",
         note=f"{MID.rear_cap*1000:.0f} mm rear cap with elliptical tail, print hollow")
    # D: phase plug: sleeve round the tweeter, nose towards the mid dust cap, three vanes
    start("phase_plug")
    prof = [(0, Z_MID_THROAT - NOSE)] + [(r, z) for r, z in reversed(ellipse(R_PLUG, Z_MID_THROAT, Z_MID_THROAT - NOSE))]
    prof += [(R_PLUG, Z_TW_FACE), (TW.body_radius * MM + 0.5, Z_TW_FACE),
             (TW.body_radius * MM + 0.5, Z_TW_TERM - 3), (0, Z_TW_TERM - 3)]
    body = revolve(prof)
    vanes = []
    for k in range(3):
        ang = k * 2 * math.pi / 3 + math.pi / 2
        v = occ.addBox(R_PLUG - 1, -1.25, Z_MID_THROAT, rt + 6 - R_PLUG, 2.5, 0.35 * (Z_TW_FACE - Z_MID_THROAT))
        occ.rotate([(3, v)], 0, 0, 0, 0, 0, 1, ang)
        vanes.append((3, v))
    occ.fuse(body, vanes)
    save("phase_plug", None, "print_grey", explode=(0, 0, 520), group="mid",
         note=f"{2*R_PLUG:.0f} mm sleeve for the {TW.model}, nose towards the mid dust cap, three 2.5 mm vanes")


def rods():
    for name, (sx, sy) in dict(rod_right=(1, 0), rod_left=(-1, 0), rod_top=(0, 1), rod_bottom=(0, -1)).items():
        start(name)
        r_out = LOW.width(Z_ROD / MM) / 2 * MM + T * 0.8
        length = r_out - (R_CAN - 8)
        c = occ.addCylinder(sx * (R_CAN - 8), sy * (R_CAN - 8), Z_ROD, sx * length, sy * length, 0, ROD_R)
        save(name, None, "aluminium", explode=(sx * 200, sy * 200, 0), group="mid",
             note=f"12 mm aluminium tube, {length:.0f} mm")


# ------------------------------------------------------------------ drivers (proxies)
def driver_proxies():
    b = LOW.body
    a = b.cone_radius * MM
    env = D.LOW_RECORD.get("physical_envelope_mm", {})
    flange_d = env.get("overall_diameter", 2 * a + 40)
    flange_t = env.get("flange_thickness", 11)
    start("drv_low_frame")
    flange = occ.addCylinder(0, 0, -flange_t, 0, 0, flange_t, flange_d / 2)
    hole = occ.addCylinder(0, 0, -flange_t - 1, 0, 0, flange_t + 3, a + 2)
    occ.cut([(3, flange)], [(3, hole)])
    spokes = []
    zc = b.cone_depth * MM
    for k in range(8):
        ang = k * math.pi / 4 + math.pi / 8
        x0, y0 = math.cos(ang) * (a + 4), math.sin(ang) * (a + 4)
        x1, y1 = math.cos(ang) * (b.motor_radius * MM - 5), math.sin(ang) * (b.motor_radius * MM - 5)
        spokes.append((3, occ.addCylinder(x0, y0, 0, x1 - x0, y1 - y0, zc, 6)))
    motor = occ.addCylinder(0, 0, zc, 0, 0, b.motor_length * MM, b.motor_radius * MM)
    occ.fuse([(3, motor)], spokes)
    save("drv_low_frame", None, "driver_metal", explode=(0, 0, -150), group="low",
         note=f"proxy of the {D.label(D.LOW_RECORD)}: flange, basket, motor")
    start("drv_low_cone")
    revolve([(a, 0), (40, zc), (36, zc), (a - 3, 0)])
    save("drv_low_cone", None, "driver_cone", explode=(0, 0, -150), group="low", note="proxy cone")
    # mid driver
    fr = MID.frame_diameter / 2 * MM
    start("drv_mid")
    frame = occ.addCylinder(0, 0, Z_MID_RIM - 6, 0, 0, 6, fr)
    can = occ.addCylinder(0, 0, Z_MID_BACK, 0, 0, Z_MID_RIM - 6 - Z_MID_BACK, 0.7 * fr)
    occ.fuse([(3, frame)], [(3, can)])
    save("drv_mid", None, "driver_metal", explode=(0, 0, 230), group="mid", note=f"proxy of the {D.label(D.MID_RECORD)}")
    start("drv_mid_cone")
    rc = R_CONE_MID
    revolve([(rc, Z_MID_RIM), (0.3 * rc, Z_MID_RIM - 0.4 * rc), (0.26 * rc, Z_MID_RIM - 0.4 * rc), (rc - 3, Z_MID_RIM)])
    save("drv_mid_cone", None, "driver_cone", explode=(0, 0, 230), group="mid", note="proxy cone")
    # bullet tweeter, from the datasheet drawing
    rb = TW.body_radius * MM
    rmouth = TW.mouth_diameter / 2 * MM
    rbul = TW.bullet_base / 2 * MM
    hd = TW.horn_depth * MM
    start("drv_tweeter_body")
    t = np.linspace(0, 1, 16)
    horn = [(rmouth * math.exp(math.log((rbul + 3) / rmouth) * s), Z_TW_FACE - hd * s) for s in t]
    prof = [(0, Z_TW_BACK), (rb, Z_TW_BACK), (rb, Z_TW_FACE - 2), (rb - 2, Z_TW_FACE)] + horn + [(0, Z_TW_FACE - hd)]
    revolve(prof)
    for k, sx in enumerate((-1, 1)):
        c = occ.addCylinder(sx * rb * 0.45, 0, Z_TW_TERM, 0, 0, Z_TW_BACK - Z_TW_TERM, 4)
    save("drv_tweeter_body", None, "driver_metal", explode=(0, 0, 620), group="mid",
         note=f"{TW.model}: {TW.body_diameter*1000:.0f} x {TW.body_depth*1000:.1f} mm body, "
              f"{TW.mouth_diameter*1000:.0f} mm horn mouth")
    start("drv_tweeter_bullet")
    prof = [(0, Z_TW_FACE - hd), (rbul, Z_TW_FACE - hd), (0, Z_TW_FACE - hd + 0.9 * hd)]
    revolve(prof)
    save("drv_tweeter_bullet", None, "aluminium", explode=(0, 0, 620), group="mid", note="bullet phase plug")


# ------------------------------------------------------------------ outputs
def dxf_panels(bevels):
    import ezdxf
    (OUT / "dxf").mkdir(exist_ok=True)
    s = bevels["slope_length_mm"]

    def write(name, pts, label):
        doc = ezdxf.new("R2010")
        msp = doc.modelspace()
        msp.add_lwpolyline(pts, close=True)
        msp.add_text(label, dxfattribs={"height": 12}).set_placement((10, -30))
        doc.saveas(OUT / "dxf" / f"{name}.dxf")

    # Horn wall inner face, as a trapezoid: throat edge WT, mouth edge WM.
    write("horn_wall_inner_face", [(-WT / 2, 0), (WT / 2, 0), (WM / 2, s), (-WM / 2, s)],
          f"Horn wall x4, inner face. Long edges bevel {bevels['long_edge_bevel_deg']:.1f} deg, "
          f"ends {bevels['end_bevel_deg']:.1f} deg. 18 mm ply.")
    write("baffle", [(-WM / 2, -WM / 2), (WM / 2, -WM / 2), (WM / 2, WM / 2), (-WM / 2, WM / 2)],
          "Baffle 25 mm, cut 353 mm hole on centre, 8 x M6 inserts on 370 mm PCD")
    doc = ezdxf.readfile(OUT / "dxf" / "baffle.dxf")
    doc.modelspace().add_circle((0, 0), LOW.cutout_radius * MM)
    for k in range(8):
        ang = k * math.pi / 4 + math.pi / 8
        doc.modelspace().add_circle((185 * math.cos(ang), 185 * math.sin(ang)), 4.0)
    doc.saveas(OUT / "dxf" / "baffle.dxf")


def assembly():
    start("assembly")
    for p in parts:
        occ.importShapes(str(OUT / "step" / f"{p['name']}.step"))
    occ.synchronize()
    gmsh.write(str(OUT / "sunflower-assembly.step"))


def rear_chamber_litres():
    box = (WM / MM) ** 2 * D.REAR_CHAMBER_DEPTH
    cone = math.pi * LOW.body.cone_radius**2 * LOW.body.cone_depth / 3   # concave side of the cone
    flange = math.pi * ((0.387 / 2) ** 2 - LOW.cutout_radius**2) * 0.011
    return dict(box_litres=box * 1000, cone_concave_litres=cone * 1000,
                flange_displacement_litres=flange * 1000,
                net_litres=(box + cone - flange) * 1000, target_litres=D.LOW_REAR_LITRES)


def main():
    OUT.mkdir(exist_ok=True)
    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 0)
    gmsh.option.setString("Geometry.OCCTargetUnit", "MM")
    cabinet()
    bevels = horn_panels()
    corner = corner_fillers()
    low_nose()
    mid_can()
    rods()
    driver_proxies()
    assembly()
    gmsh.finalize()
    dxf_panels(bevels)
    cut = dict(panels=bevels, corner_filler=corner, rear_chamber=rear_chamber_litres(),
               external_mm=dict(width=W_EXT, height=W_EXT, depth=Z_MOUTH - Z_BACK_OUT),
               stations_mm=dict(baffle_back=0, baffle_front=BAF, mid_tail_tip=Z_TAIL, mid_can_back=Z_CAN,
                                mid_driver_back=Z_MID_BACK, mid_cone_rim=Z_MID_RIM, mid_throat=Z_MID_THROAT,
                                tweeter_face=Z_TW_FACE, mouth=Z_MOUTH, support_rods=Z_ROD))
    (OUT / "parts.json").write_text(json.dumps(parts, indent=1) + "\n")
    (OUT / "cutlist.json").write_text(json.dumps(cut, indent=1) + "\n")
    print(json.dumps(cut, indent=1))


if __name__ == "__main__":
    main()
