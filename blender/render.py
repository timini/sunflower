"""Headless Blender renders and animations of the Sunflower assembly from cad/stl.

    blender -b -P blender/render.py -- --out docs/img/renders [--shots hero,front] [--samples 160]

Stills: hero, front, section, exploded, detail, rear, array (3 x 3: stack.jpg,
stack-front.jpg, stack-side.jpg and the plan stack-top.jpg). Animations:
turntable, orbit (the array), assemble (exploded and back). Animations write
PNG frames to --frames/<name>/; the justfile encodes them.

Reads cad/parts.json for categories and explode vectors and cad/array.json for
the array (nine identical boxes that tile exactly, no spacers). CAD coordinates
are millimetres with y up and z towards the listener; they map to Blender
metres, Z up, front facing -Y.
"""
import argparse
import json
import math
import sys
import time
from pathlib import Path

import bmesh
import bpy
from mathutils import Matrix, Vector

ROOT = Path(__file__).resolve().parents[1]
STILLS = ("hero", "front", "section", "exploded", "detail", "rear", "array")
ANIMS = ("turntable", "orbit", "assemble")
SHOTS = STILLS + ANIMS

# CAD frame to Blender frame: (x, y, z) mm -> (x, -z, y) m
C = Matrix.Rotation(math.radians(90), 4, "X")

MATERIALS = {
    #               base colour (linear)        metallic roughness
    "birch":        ((0.52, 0.36, 0.20, 1),    0.0, 0.45),
    "print_dark":   ((0.010, 0.010, 0.011, 1), 0.0, 0.55),
    "print_bullet": ((0.035, 0.036, 0.040, 1), 0.0, 0.40),   # dark too, a shade off the horn so it reads
    "aluminium":    ((0.90, 0.91, 0.92, 1),    1.0, 0.24),
    "driver_metal": ((0.16, 0.16, 0.17, 1),    1.0, 0.38),
    "driver_cone":  ((0.018, 0.018, 0.018, 1), 0.0, 0.85),
    "felt":         ((0.075, 0.075, 0.080, 1), 0.0, 1.0),    # dark grey wool
}


def args():
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "docs/img/renders"))
    ap.add_argument("--frames", default=str(ROOT / ".tmp/frames"))
    ap.add_argument("--shots", default=",".join(SHOTS))
    ap.add_argument("--samples", type=int, default=160, help="stills")
    ap.add_argument("--vsamples", type=int, default=96, help="animation frames")
    ap.add_argument("--res", default="1800x1200")
    ap.add_argument("--vres", default="1280x720")
    ap.add_argument("--frame-step", type=int, default=1, help="render every nth frame (previews)")
    ap.add_argument("--device", default="GPU", choices=("GPU", "CPU"))
    return ap.parse_args(argv)


def pick_material(p):
    cat, name = p.get("category", ""), p["name"]
    if cat == "felt" or "felt" in name:
        return "felt"
    if cat in ("cabinet", "filler"):
        return "birch"
    if cat == "print":
        return "print_bullet" if name.startswith("bullet") else "print_dark"
    if cat == "support":
        return "aluminium"
    if cat == "driver":
        return "driver_cone" if "cone" in name else "driver_metal"
    return {"ply": "birch", "print_orange": "print_bullet", "aluminium": "aluminium",
            "driver_cone": "driver_cone"}.get(p.get("material"), "print_dark")


# ---------------------------------------------------------------- scene setup

def reset(device="GPU"):
    bpy.ops.wm.read_factory_settings(use_empty=True)
    sc = bpy.context.scene
    sc.render.engine = "CYCLES"
    prefs = bpy.context.preferences.addons["cycles"].preferences
    try:
        if device != "GPU":
            raise RuntimeError("CPU requested")
        prefs.compute_device_type = "METAL"
        prefs.get_devices()
        for d in prefs.devices:
            d.use = d.type == "METAL"          # the CPU only slows a Metal render down
        sc.cycles.device = "GPU"
    except Exception:
        sc.cycles.device = "CPU"
    cy = sc.cycles
    cy.use_denoising = True
    cy.denoiser = "OPENIMAGEDENOISE"
    cy.denoising_input_passes = "RGB_ALBEDO_NORMAL"
    cy.denoising_prefilter = "ACCURATE"
    cy.denoising_use_gpu = True
    cy.seed = 7
    cy.use_animated_seed = False
    cy.max_bounces = 6
    cy.diffuse_bounces = 3
    cy.glossy_bounces = 3
    cy.transmission_bounces = 2
    cy.volume_bounces = 0
    cy.sample_clamp_indirect = 4.0
    sc.render.use_persistent_data = True
    sc.view_settings.view_transform = "AgX"
    sc.view_settings.look = "AgX - Medium High Contrast"
    sc.view_settings.exposure = -0.6
    sc.render.film_transparent = False
    # linear keys for loops, unless a shot sets Bezier
    bpy.context.preferences.edit.keyframe_new_interpolation_type = "LINEAR"
    return sc


def material(name):
    if name in bpy.data.materials:
        return bpy.data.materials[name]
    col, metal, rough = MATERIALS[name]
    m = bpy.data.materials.new(name)
    m.use_nodes = True
    nt = m.node_tree
    b = nt.nodes["Principled BSDF"]
    b.inputs["Base Color"].default_value = col
    b.inputs["Metallic"].default_value = metal
    b.inputs["Roughness"].default_value = rough
    if name == "birch":
        # pale birch with a fine grain along the cabinet depth (Blender Y), satin lacquer
        coord = nt.nodes.new("ShaderNodeTexCoord")
        mapping = nt.nodes.new("ShaderNodeMapping")
        mapping.inputs["Scale"].default_value = (70.0, 1.5, 70.0)
        noise = nt.nodes.new("ShaderNodeTexNoise")
        noise.inputs["Scale"].default_value = 5.0
        noise.inputs["Detail"].default_value = 8.0
        noise.inputs["Distortion"].default_value = 0.5
        ramp = nt.nodes.new("ShaderNodeValToRGB")
        ramp.color_ramp.elements[0].position = 0.3
        ramp.color_ramp.elements[0].color = (0.40, 0.25, 0.12, 1)
        ramp.color_ramp.elements[1].position = 0.75
        ramp.color_ramp.elements[1].color = (0.58, 0.40, 0.22, 1)
        nt.links.new(coord.outputs["Object"], mapping.inputs["Vector"])
        nt.links.new(mapping.outputs["Vector"], noise.inputs["Vector"])
        nt.links.new(noise.outputs["Fac"], ramp.inputs["Fac"])
        nt.links.new(ramp.outputs["Color"], b.inputs["Base Color"])
        b.inputs["Coat Weight"].default_value = 0.3
        b.inputs["Coat Roughness"].default_value = 0.25
    elif name == "print_bullet":
        b.inputs["Coat Weight"].default_value = 0.35
        b.inputs["Coat Roughness"].default_value = 0.2
    elif name == "print_dark":
        b.inputs["Specular IOR Level"].default_value = 0.3
        # faint print layer lines along the axis
        coord = nt.nodes.new("ShaderNodeTexCoord")
        wave = nt.nodes.new("ShaderNodeTexWave")
        wave.wave_type = "BANDS"
        wave.bands_direction = "Y"
        wave.inputs["Scale"].default_value = 300.0
        bump = nt.nodes.new("ShaderNodeBump")
        bump.inputs["Strength"].default_value = 0.04
        nt.links.new(coord.outputs["Object"], wave.inputs["Vector"])
        nt.links.new(wave.outputs["Fac"], bump.inputs["Height"])
        nt.links.new(bump.outputs["Normal"], b.inputs["Normal"])
    add_cut(m, cap_colour(name, col))
    return m


def cap_colour(name, col):
    # cut faces: prints a dark slate, the bullet a bluer slate, drivers steel, so each part reads
    return {"birch": (0.62, 0.45, 0.27, 1), "print_dark": (0.060, 0.062, 0.068, 1),
            "print_bullet": (0.050, 0.075, 0.120, 1), "felt": (0.12, 0.12, 0.125, 1),
            "aluminium": (0.55, 0.56, 0.58, 1), "driver_metal": (0.36, 0.37, 0.40, 1),
            "driver_cone": (0.015, 0.015, 0.015, 1)}.get(name, col)


def add_cut(m, cap):
    """Shader cutaway, off unless section() turns the 'cut' value on."""
    nt = m.node_tree
    out = nt.nodes["Material Output"]
    surf = out.inputs["Surface"].links[0].from_socket
    geo = nt.nodes.new("ShaderNodeNewGeometry")
    cut = nt.nodes.new("ShaderNodeValue")
    cut.name = "cut"
    cut.outputs[0].default_value = 0.0
    sep = nt.nodes.new("ShaderNodeSeparateXYZ")
    gt = nt.nodes.new("ShaderNodeMath")
    gt.operation = "GREATER_THAN"
    gt.inputs[1].default_value = 0.0
    m1 = nt.nodes.new("ShaderNodeMath")
    m1.operation = "MULTIPLY"
    m2 = nt.nodes.new("ShaderNodeMath")
    m2.operation = "MULTIPLY"
    # flat, slightly self-lit cap so cut faces read the same whatever the light
    capd = nt.nodes.new("ShaderNodeBsdfDiffuse")
    capd.inputs["Color"].default_value = cap
    cape = nt.nodes.new("ShaderNodeEmission")
    cape.inputs["Color"].default_value = cap
    cape.inputs["Strength"].default_value = 0.8
    capn = nt.nodes.new("ShaderNodeAddShader")
    nt.links.new(capd.outputs[0], capn.inputs[0])
    nt.links.new(cape.outputs[0], capn.inputs[1])
    transp = nt.nodes.new("ShaderNodeBsdfTransparent")
    mixa = nt.nodes.new("ShaderNodeMixShader")
    mixb = nt.nodes.new("ShaderNodeMixShader")
    L = nt.links.new
    L(geo.outputs["Position"], sep.inputs[0])
    L(sep.outputs["X"], gt.inputs[0])
    L(gt.outputs[0], m1.inputs[0])
    L(cut.outputs[0], m1.inputs[1])
    L(geo.outputs["Backfacing"], m2.inputs[0])
    L(cut.outputs[0], m2.inputs[1])
    L(m2.outputs[0], mixa.inputs[0])
    L(surf, mixa.inputs[1])
    L(capn.outputs[0], mixa.inputs[2])
    L(m1.outputs[0], mixb.inputs[0])
    L(mixa.outputs[0], mixb.inputs[1])
    L(transp.outputs[0], mixb.inputs[2])
    L(mixb.outputs[0], out.inputs["Surface"])


def smooth(ob, angle=35):
    bpy.context.view_layer.objects.active = ob
    ob.select_set(True)
    try:
        bpy.ops.object.shade_smooth_by_angle(angle=math.radians(angle))
    except Exception:
        try:
            bpy.ops.object.shade_auto_smooth(angle=math.radians(angle))
        except Exception:
            pass
    ob.select_set(False)


def import_stl(path, name, coll):
    bpy.ops.wm.stl_import(filepath=str(path))
    ob = bpy.context.selected_objects[0]
    ob.name = name
    ob.data.transform(C @ Matrix.Scale(0.001, 4))
    bm = bmesh.new()                       # STL triangles are loose: weld them into a closed mesh
    bm.from_mesh(ob.data)
    bmesh.ops.remove_doubles(bm, verts=bm.verts, dist=1e-6)
    bm.to_mesh(ob.data)
    bm.free()
    ob.data.update()
    for c in list(ob.users_collection):
        c.objects.unlink(ob)
    coll.objects.link(ob)
    return ob


def load_parts():
    """One box in the collection 'box', in the box frame."""
    parts = json.loads((ROOT / "cad/parts.json").read_text())
    coll = bpy.data.collections.new("box")
    bpy.context.scene.collection.children.link(coll)
    objs = {}
    for p in parts:
        f = ROOT / "cad/stl" / f"{p['name']}.stl"
        if not f.exists():
            print("missing", f, flush=True)
            continue
        ob = import_stl(f, p["name"], coll)
        ob.data.materials.append(material(pick_material(p)))
        ob["explode"] = p["explode_mm"]
        ob["category"] = p.get("category", "")
        ob["base_loc"] = list(ob.location)
        smooth(ob, 20 if p.get("category") in ("cabinet", "filler") else 35)
        objs[p["name"]] = ob
    return objs, coll


def world_and_lights():
    w = bpy.data.worlds.new("w")
    bpy.context.scene.world = w
    w.use_nodes = True
    nt = w.node_tree
    bg = nt.nodes["Background"]
    # soft studio gradient: brighter overhead, darker towards the floor
    tc = nt.nodes.new("ShaderNodeTexCoord")
    grad = nt.nodes.new("ShaderNodeSeparateXYZ")
    ramp = nt.nodes.new("ShaderNodeValToRGB")
    ramp.color_ramp.elements[0].position = 0.0
    ramp.color_ramp.elements[0].color = (0.30, 0.31, 0.33, 1)
    ramp.color_ramp.elements[1].position = 1.0
    ramp.color_ramp.elements[1].color = (0.88, 0.89, 0.91, 1)
    mr = nt.nodes.new("ShaderNodeMapRange")          # spread the gradient over a level view
    mr.inputs["From Min"].default_value = -0.35
    mr.inputs["From Max"].default_value = 0.45
    nt.links.new(tc.outputs["Generated"], grad.inputs["Vector"])
    nt.links.new(grad.outputs["Z"], mr.inputs["Value"])
    nt.links.new(mr.outputs["Result"], ramp.inputs["Fac"])
    nt.links.new(ramp.outputs["Color"], bg.inputs["Color"])
    bg.inputs["Strength"].default_value = 0.6

    bpy.ops.mesh.primitive_plane_add(size=400, location=(0, 0, -0.4))
    floor = bpy.context.active_object
    floor.name = "floor"
    fm = bpy.data.materials.new("floor")
    fm.use_nodes = True
    fb = fm.node_tree.nodes["Principled BSDF"]
    fb.inputs["Base Color"].default_value = (0.62, 0.63, 0.65, 1)
    fb.inputs["Roughness"].default_value = 0.85
    floor.data.materials.append(fm)

    rig = bpy.data.objects.new("lights", None)
    bpy.context.scene.collection.objects.link(rig)
    for loc, energy, size in (((-2.6, -3.0, 2.8), 1100, 2.5),   # key, front left, high
                              ((3.2, -1.6, 1.0), 380, 2.0),     # fill, front right
                              ((1.0, 3.0, 3.2), 800, 3.0),      # rim, behind
                              ((0.0, -3.0, -0.1), 140, 2.5)):   # low front fill into the mouth
        bpy.ops.object.light_add(type="AREA", location=loc)
        L = bpy.context.active_object
        L.data.energy = energy
        L.data.size = size
        L.rotation_euler = (Vector((0, -0.3, 0)) - Vector(loc)).to_track_quat("-Z", "Y").to_euler()
        L.parent = rig
        L.visible_camera = False
        L.visible_glossy = True
    return floor, rig


# ------------------------------------------------------------------ helpers

def bbox_points(objs, inst=None):
    """World corners of the bounding boxes of the given objects (optionally through instance matrices)."""
    dg = bpy.context.evaluated_depsgraph_get()
    pts = []
    mats = inst or [Matrix.Identity(4)]
    for ob in objs:
        if ob.hide_render:
            continue
        ev = ob.evaluated_get(dg)
        for M in mats:
            pts += [M @ ev.matrix_world @ Vector(c) for c in ev.bound_box]
    return pts


def bounds(pts):
    lo = Vector((min(p.x for p in pts), min(p.y for p in pts), min(p.z for p in pts)))
    hi = Vector((max(p.x for p in pts), max(p.y for p in pts), max(p.z for p in pts)))
    return lo, hi


def get_camera():
    cam = bpy.data.objects.get("cam")
    if cam is None:
        data = bpy.data.cameras.new("cam")
        cam = bpy.data.objects.new("cam", data)
        bpy.context.scene.collection.objects.link(cam)
        bpy.context.scene.camera = cam
        data.clip_start = 0.05
        data.clip_end = 200
    cam.parent = None
    cam.animation_data_clear()
    return cam


def aim(direction, target, pts, lens=50, margin=1.1, ortho=None, res=None):
    """Point the camera along direction at target and back it off until pts fit, times margin."""
    sc = bpy.context.scene
    if res:
        sc.render.resolution_x, sc.render.resolution_y = res
    cam = get_camera()
    d = Vector(direction).normalized()
    cam.rotation_euler = d.to_track_quat("-Z", "Y").to_euler()
    cam.data.lens = lens
    target = Vector(target)
    if ortho:
        cam.data.type = "ORTHO"
        cam.data.ortho_scale = ortho
        cam.location = target - d * 10
        return cam
    cam.data.type = "PERSP"
    # fit: distance where every point is inside the frustum about the target
    w, h = sc.render.resolution_x, sc.render.resolution_y
    sensor = cam.data.sensor_width
    fx = (sensor / 2) / lens                          # tan of the half angle on the wide side
    tx, ty = (fx, fx * h / w) if w >= h else (fx * w / h, fx)
    R = cam.rotation_euler.to_matrix()
    right, up = R.col[0], R.col[1]
    dist = 0.1
    for p in pts:
        v = p - target
        along = v.dot(d)
        need = max(abs(v.dot(right)) / tx, abs(v.dot(up)) / ty) - along
        dist = max(dist, need)
    cam.location = target - d * dist * margin
    return cam


def render(path, samples, res):
    sc = bpy.context.scene
    w, h = (int(v) for v in res.split("x")) if isinstance(res, str) else res
    sc.render.resolution_x, sc.render.resolution_y = w, h
    sc.render.resolution_percentage = 100
    sc.cycles.samples = samples
    sc.render.image_settings.file_format = "JPEG"
    sc.render.image_settings.quality = 90
    sc.render.filepath = str(path)
    t = time.time()
    bpy.ops.render.render(write_still=True)
    print(f"wrote {path} in {time.time() - t:.1f} s", flush=True)


def render_anim(folder, samples, res, frames, step=1):
    sc = bpy.context.scene
    folder.mkdir(parents=True, exist_ok=True)
    for f in folder.glob("f*.png"):
        f.unlink()
    sc.render.resolution_x, sc.render.resolution_y = res
    sc.render.resolution_percentage = 100
    sc.cycles.samples = samples
    sc.cycles.max_bounces, sc.cycles.diffuse_bounces, sc.cycles.glossy_bounces = 4, 2, 2
    sc.cycles.adaptive_threshold = 0.02
    sc.render.image_settings.file_format = "PNG"
    sc.render.image_settings.color_depth = "8"
    sc.render.image_settings.compression = 15
    sc.render.fps = 30
    sc.frame_start, sc.frame_end, sc.frame_step = 1, frames, step
    sc.render.filepath = str(folder / "f")
    t = time.time()
    bpy.ops.render.render(animation=True)
    n = len(list(folder.glob("f*.png")))
    print(f"wrote {n} frames to {folder} in {time.time() - t:.1f} s", flush=True)


def explode(objs, factor):
    for ob in objs.values():
        e = ob["explode"]
        ob.location = Vector(ob["base_loc"]) + C @ Vector(e) * (factor / 1000)


def section(objs, on=True):
    """Cut away the right half (x > 0 in CAD) in the shaders: that half turns transparent and the
    inside faces seen through the cut take a flat cap colour, so solid parts read as cut solids."""
    for m in bpy.data.materials:
        node = m.node_tree.nodes.get("cut") if m.node_tree else None
        if node:
            node.outputs[0].default_value = 1.0 if on else 0.0
    bpy.context.scene.cycles.transparent_max_bounces = 64 if on else 8


def set_floor(floor, z):
    floor.location.z = z


def key_interp(kind):
    bpy.context.preferences.edit.keyframe_new_interpolation_type = kind


# ------------------------------------------------------------------ array

def array_setup(box_coll):
    """Nine collection instances placed by cad/array.json. The boxes tile exactly: no spacers."""
    arr = json.loads((ROOT / "cad/array.json").read_text())
    insts = []
    for b in arr["boxes"]:
        M = Matrix(b["matrix"])
        Mb = C @ M @ C.inverted()          # box frame to array frame, in Blender axes
        inst = bpy.data.objects.new(f"box_{b['i']}_{b['j']}", None)
        inst.instance_type = "COLLECTION"
        inst.instance_collection = box_coll
        inst.matrix_world = Mb
        inst["ij"] = (b["i"], b["j"])
        bpy.context.scene.collection.objects.link(inst)
        insts.append(inst)
    print(f"array: {len(insts)} boxes", flush=True)
    return arr, insts


def show_box(box_coll, on):
    lc = bpy.context.view_layer.layer_collection.children[box_coll.name]
    lc.exclude = not on


# ------------------------------------------------------------------ main

def main():
    a = args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    fr = Path(a.frames)
    shots = [s.strip() for s in a.shots.split(",") if s.strip()]
    shots = ["array" if s == "stack" else s for s in shots]
    sc = reset(a.device)
    objs, box = load_parts()
    floor, rig = world_and_lights()
    res = tuple(int(v) for v in a.res.split("x"))
    vres = tuple(int(v) for v in a.vres.split("x"))

    cut = json.loads((ROOT / "cad/cutlist.json").read_text())
    ext = cut["external_mm"]
    st = cut["stations_mm"]
    W, Hf = ext["width"] / 1000, ext["height_front"] / 1000
    pts_box = bbox_points(objs.values())
    lo, hi = bounds(pts_box)
    centre = (lo + hi) / 2                          # Blender, middle of the whole assembly
    floor_z = lo.z - 0.002
    set_floor(floor, floor_z)
    cams = {}
    timing = {}

    def save_cams():
        old = json.loads((out / "cameras.json").read_text()) if (out / "cameras.json").exists() else {}
        old.update(cams)
        (out / "cameras.json").write_text(json.dumps(old, indent=1))

    def timed(name, fn):
        t = time.time()
        fn()
        if cams:
            save_cams()                     # straight away, so annotate can run before the videos finish
        timing[name] = time.time() - t
        print(f"TIME {name} {timing[name]:.1f} s", flush=True)

    hero_dir = Vector((-1.25, 2.0, -0.62))           # from front right, above, looking back and down
    if "hero" in shots:
        def f():
            aim(hero_dir, centre, pts_box, lens=50, margin=1.06, res=res)
            render(out / "hero.jpg", a.samples, res)
        timed("hero", f)
    if "front" in shots:
        def f():
            s = 1.32 * max(W, Hf)
            aim((0, 1, 0), (0, centre.y, 0), pts_box, ortho=s, res=(1400, 1400))
            render(out / "front.jpg", a.samples, (1400, 1400))
            cams["front"] = dict(res=[1400, 1400], ortho=s, centre_h=0.0, centre_v=0.0, axes="x,y")
        timed("front", f)
    if "detail" in shots:
        def f():
            zm = st["mid_mouth"] / 1000
            tgt = Vector((0, -zm + 0.2, 0))
            near = [Vector((x, -zm, z)) for x in (-0.16, 0.16) for z in (-0.16, 0.16)]
            aim((-0.42, 1.0, -0.28), tgt, near, lens=70, margin=1.22, res=res)
            render(out / "detail.jpg", a.samples, res)
        timed("detail", f)
    if "rear" in shots:
        def f():
            objs["back"].hide_render = True
            aim((0.75, -1.0, -0.45), centre, pts_box, lens=50, margin=0.95, res=res)
            render(out / "rear-open.jpg", a.samples, res)
            objs["back"].hide_render = False
        timed("rear", f)
    if "exploded" in shots:
        def f():
            explode(objs, 1.0)
            pts = bbox_points(objs.values())
            l2, h2 = bounds(pts)
            set_floor(floor, l2.z - 0.002)
            aim((-1.3, 1.1, -0.65), (l2 + h2) / 2, pts, lens=45, margin=1.02, res=res)
            render(out / "exploded.jpg", a.samples, res)
            explode(objs, 0.0)
            set_floor(floor, floor_z)
        timed("exploded", f)
    if "section" in shots:
        def f():
            section(objs, True)
            half = [Vector((min(p.x, 0.0), p.y, p.z)) for p in pts_box]   # fit the half that is left
            aim((-1.25, 0.45, -0.5), (bounds(half)[0] + bounds(half)[1]) / 2, half, lens=45, margin=1.04, res=res)
            render(out / "section.jpg", a.samples, res)
            s = 1.18 * max(hi.y - lo.y, (hi.z - lo.z) * 1600 / 1200)
            aim((-1, 0, 0), (0, centre.y, 0), pts_box, ortho=s, res=(1600, 1200))
            render(out / "section-side.jpg", a.samples, (1600, 1200))
            cams["section-side"] = dict(res=[1600, 1200], ortho=s, centre_h=centre.y * 1000, centre_v=0.0, axes="-z,y")
            section(objs, False)
        timed("section", f)

    if "turntable" in shots:
        def f():
            key_interp("LINEAR")
            sc.frame_set(1)
            piv = bpy.data.objects.new("pivot", None)
            sc.collection.objects.link(piv)
            piv.location = centre
            # fit at a few headings so nothing clips anywhere on the turn
            elev = Vector((0, 2.35, -0.75)).normalized()
            dist = 0
            for k in range(8):
                d = Matrix.Rotation(k * math.pi / 4, 3, "Z") @ elev
                cam = aim(d, centre, pts_box, lens=45, margin=1.08, res=vres)
                dist = max(dist, (cam.location - centre).length)
            cam = aim(elev, centre, pts_box, lens=45, res=vres)
            cam.location = centre - elev * dist
            bpy.context.view_layer.update()            # else the pivot's matrix_world is stale and the camera orbits off centre
            cam.parent = piv
            cam.matrix_parent_inverse = piv.matrix_world.inverted()
            rig.parent = None
            N = 240
            piv.rotation_euler = (0, 0, 0)
            piv.keyframe_insert("rotation_euler", index=2, frame=1)
            piv.rotation_euler = (0, 0, 2 * math.pi)
            piv.keyframe_insert("rotation_euler", index=2, frame=N + 1)
            render_anim(fr / "turntable", a.vsamples, vres, N, a.frame_step)
            bpy.data.objects.remove(piv)
        timed("turntable", f)

    if "assemble" in shots:
        def f():
            key_interp("BEZIER")
            N = 150
            explode(objs, 0.85)
            pts = bbox_points(objs.values())
            l2, h2 = bounds(pts)
            explode(objs, 0.0)
            set_floor(floor, l2.z - 0.002)
            aim((-1.3, 1.1, -0.65), (l2 + h2) / 2, pts, lens=45, margin=1.04, res=vres)
            # held assembled, explode, hold, assemble, hold; loops
            for fac, frame in ((0, 1), (0, 20), (0.85, 65), (0.85, 86), (0, 131), (0, N + 1)):
                explode(objs, fac)
                for ob in objs.values():
                    ob.keyframe_insert("location", frame=frame)
            render_anim(fr / "assemble", a.vsamples, vres, N, a.frame_step)
            for ob in objs.values():
                ob.animation_data_clear()
            explode(objs, 0.0)
            set_floor(floor, floor_z)
        timed("assemble", f)

    if "array" in shots or "orbit" in shots:
        arr, insts = array_setup(box)
        show_box(box, False)
        mats = [i.matrix_world.copy() for i in insts]
        pts = bbox_points(objs.values(), mats)
        al, ah = bounds(pts)
        ac = (al + ah) / 2
        floor.hide_render = True                 # flown: no floor, the gradient is the backdrop
        rig.location = ac
        rig.scale = (1.8, 1.8, 1.8)
        for L in rig.children:
            L.data.energy *= 1.8 ** 2
        if "array" in shots:
            def f():
                fwd = ac - Vector((0, 0.25, 0))          # weight the framing towards the mouths
                aim((-0.62, 1.6, -0.42), fwd, pts, lens=40, margin=1.15, res=res)
                render(out / "stack.jpg", a.samples, res)
                aim((0, 1, -0.06), ac, pts, lens=50, margin=1.12, res=res)
                render(out / "stack-front.jpg", a.samples, res)
                # the centre column from the side, cut on its axis: rows stacked flat
                for i in insts:
                    i.hide_render = i["ij"][0] != 0
                section(objs, True)
                cpts = bbox_points(objs.values(), [i.matrix_world for i in insts if i["ij"][0] == 0])
                cl, ch = bounds(cpts)
                cc = (cl + ch) / 2
                s = 1.18 * max(ch.y - cl.y, (ch.z - cl.z) * 1600 / 1400)
                aim((-1, 0, 0), (0, cc.y, cc.z), cpts, ortho=s, res=(1600, 1400))
                render(out / "stack-side.jpg", a.samples, (1600, 1400))
                cams["stack-side"] = dict(res=[1600, 1400], ortho=s, centre_h=cc.y * 1000, centre_v=cc.z * 1000,
                                          axes="-z,y")
                section(objs, False)
                for i in insts:
                    i.hide_render = False
                # plan from above, mouths towards the bottom of the frame: the columns fan in this view
                apex_y = arr["apex_behind_baffle_mm"] / 1000      # Blender y of the fan axis
                y0, y1 = al.y - 0.12, max(ah.y, apex_y) + 0.22
                s = 1.08 * max(ah.x - al.x, (y1 - y0) * 1600 / 1400)
                tc = Vector(((al.x + ah.x) / 2, (y0 + y1) / 2, ac.z))
                aim((0, 0, -1), tc, pts, ortho=s, res=(1600, 1400))
                render(out / "stack-top.jpg", a.samples, (1600, 1400))
                cams["stack-top"] = dict(res=[1600, 1400], ortho=s, centre_h=tc.x * 1000, centre_v=tc.y * 1000,
                                         axes="x,-z")
            timed("array", f)
        if "orbit" in shots:
            def f():
                key_interp("BEZIER")
                N = 240
                piv = bpy.data.objects.new("pivot", None)
                sc.collection.objects.link(piv)
                piv.location = ac - Vector((0, 0.25, 0))   # a little forward, nearer the mouths
                dist = 0
                base = Vector((0, 1.0, -0.22)).normalized()
                for deg in (-60, -30, 0, 30, 60):
                    d = Matrix.Rotation(math.radians(deg), 3, "Z") @ base
                    cam = aim(d, piv.location, pts, lens=40, margin=1.04, res=vres)
                    dist = max(dist, (cam.location - piv.location).length)
                cam = aim(base, piv.location, pts, lens=40, res=vres)
                cam.location = piv.location - base * dist
                bpy.context.view_layer.update()
                cam.parent = piv
                cam.matrix_parent_inverse = piv.matrix_world.inverted()
                # -60 to +60 and back, eased at both ends; frame N+1 equals frame 1
                for ang, frame in ((-60, 1), (60, N // 2 + 1), (-60, N + 1)):
                    piv.rotation_euler = (0, 0, math.radians(ang))
                    piv.keyframe_insert("rotation_euler", index=2, frame=frame)
                render_anim(fr / "orbit", a.vsamples, vres, N, a.frame_step)
            timed("orbit", f)

    print("TIMES", json.dumps({k: round(v, 1) for k, v in timing.items()}), flush=True)


main()
