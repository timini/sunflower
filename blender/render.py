"""Headless Blender renders of the Sunflower assembly from cad/stl.

    blender -b -P blender/render.py -- --out docs/img/renders [--shots hero,front] [--samples 128]

Reads cad/parts.json for materials and explode vectors. CAD coordinates are
millimetres with y up and z towards the listener; they are mapped to
Blender metres, Z up, front facing -Y.
"""
import argparse
import json
import math
import sys
from pathlib import Path

import bpy
from mathutils import Matrix, Vector

ROOT = Path(__file__).resolve().parents[1]
SHOTS = ("hero", "front", "section", "exploded", "detail", "rear", "turntable")

MATERIALS = {
    #             base colour (linear)        metallic roughness
    "ply":          ((0.045, 0.048, 0.052, 1), 0.0, 0.55),
    "print_orange": ((0.80, 0.42, 0.02, 1),    0.0, 0.45),   # sunflower yellow
    "print_black":  ((0.012, 0.012, 0.013, 1), 0.0, 0.6),
    "print_grey":   ((0.80, 0.42, 0.02, 1),    0.0, 0.45),
    "aluminium":    ((0.85, 0.86, 0.88, 1),    1.0, 0.28),
    "driver_metal": ((0.08, 0.08, 0.09, 1),    0.9, 0.35),
    "driver_cone":  ((0.02, 0.02, 0.02, 1),    0.0, 0.85),
}


def args():
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "docs/img/renders"))
    ap.add_argument("--shots", default=",".join(SHOTS))
    ap.add_argument("--samples", type=int, default=160)
    ap.add_argument("--res", default="1800x1200")
    ap.add_argument("--device", default="GPU", choices=("GPU", "CPU"))
    return ap.parse_args(argv)


def cad_to_blender(v):
    x, y, z = v
    return Vector((x / 1000, -z / 1000, y / 1000))


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
            d.use = True
        sc.cycles.device = "GPU"
    except Exception:
        sc.cycles.device = "CPU"
    sc.cycles.use_denoising = True
    sc.cycles.seed = 7
    sc.view_settings.view_transform = "AgX"
    sc.view_settings.look = "AgX - Medium High Contrast"
    sc.render.film_transparent = False
    sc.view_settings.exposure = -0.7
    return sc


def material(name):
    if name in bpy.data.materials:
        return bpy.data.materials[name]
    col, metal, rough = MATERIALS[name]
    m = bpy.data.materials.new(name)
    m.use_nodes = True
    b = m.node_tree.nodes["Principled BSDF"]
    b.inputs["Base Color"].default_value = col
    b.inputs["Metallic"].default_value = metal
    b.inputs["Roughness"].default_value = rough
    if name == "ply":
        # birch ply: pale wood with a fine grain along the panel, satin lacquer
        nt = m.node_tree
        coord = nt.nodes.new("ShaderNodeTexCoord")
        mapping = nt.nodes.new("ShaderNodeMapping")
        mapping.inputs["Scale"].default_value = (60.0, 1.2, 60.0)   # grain runs along the depth (Y)
        noise = nt.nodes.new("ShaderNodeTexNoise")
        noise.inputs["Scale"].default_value = 6.0
        noise.inputs["Detail"].default_value = 6.0
        noise.inputs["Distortion"].default_value = 0.6
        ramp = nt.nodes.new("ShaderNodeValToRGB")
        ramp.color_ramp.elements[0].position = 0.35
        ramp.color_ramp.elements[0].color = (0.36, 0.22, 0.11, 1)
        ramp.color_ramp.elements[1].position = 0.75
        ramp.color_ramp.elements[1].color = (0.56, 0.38, 0.21, 1)
        nt.links.new(coord.outputs["Object"], mapping.inputs["Vector"])
        nt.links.new(mapping.outputs["Vector"], noise.inputs["Vector"])
        nt.links.new(noise.outputs["Fac"], ramp.inputs["Fac"])
        nt.links.new(ramp.outputs["Color"], b.inputs["Base Color"])
        b.inputs["Roughness"].default_value = 0.42
        b.inputs["Coat Weight"].default_value = 0.25
        b.inputs["Coat Roughness"].default_value = 0.3
    return m


def load_parts():
    parts = json.loads((ROOT / "cad/parts.json").read_text())
    objs = {}
    for p in parts:
        bpy.ops.wm.stl_import(filepath=str(ROOT / "cad/stl" / f"{p['name']}.stl"))
        ob = bpy.context.selected_objects[0]
        ob.name = p["name"]
        ob.data.transform(Matrix.Rotation(math.radians(90), 4, "X") @ Matrix.Scale(0.001, 4))
        ob.data.update()
        ob.data.materials.append(material(p["material"]))
        for poly in ob.data.polygons:
            poly.use_smooth = p["material"] not in ("ply",)
        ob["explode"] = p["explode_mm"]
        ob["group"] = p["group"]
        objs[p["name"]] = ob
    for ob in objs.values():
        try:
            bpy.context.view_layer.objects.active = ob
            bpy.ops.object.shade_auto_smooth(angle=math.radians(35))
        except Exception:
            pass
    return objs


def world_and_lights():
    w = bpy.data.worlds.new("w")
    bpy.context.scene.world = w
    w.use_nodes = True
    bg = w.node_tree.nodes["Background"]
    bg.inputs["Color"].default_value = (0.82, 0.83, 0.85, 1)
    bg.inputs["Strength"].default_value = 0.55
    # floor with shadow catcher look
    bpy.ops.mesh.primitive_plane_add(size=200, location=(0, 0, -0.368))
    floor = bpy.context.active_object
    fm = bpy.data.materials.new("floor")
    fm.use_nodes = True
    fm.node_tree.nodes["Principled BSDF"].inputs["Base Color"].default_value = (0.78, 0.79, 0.8, 1)
    fm.node_tree.nodes["Principled BSDF"].inputs["Roughness"].default_value = 0.9
    floor.data.materials.append(fm)
    for loc, energy, size in (((-2.5, -3.0, 2.5), 900, 2.5), ((3.0, -1.5, 1.2), 350, 2.0),
                              ((0.5, 2.5, 3.0), 600, 3.0), ((0.0, -2.5, -0.2), 120, 2.0)):
        bpy.ops.object.light_add(type="AREA", location=loc)
        L = bpy.context.active_object
        L.data.energy = energy
        L.data.size = size
        direction = Vector((0, -0.4, 0)) - Vector(loc)
        L.rotation_euler = direction.to_track_quat("-Z", "Y").to_euler()
    return floor


def camera(loc, target=(0, -0.4, 0.0), lens=50, ortho=None):
    cam = bpy.data.objects.get("cam")
    if cam is None:
        bpy.ops.object.camera_add()
        cam = bpy.context.active_object
        cam.name = "cam"
        bpy.context.scene.camera = cam
    cam.location = loc
    d = Vector(target) - Vector(loc)
    cam.rotation_euler = d.to_track_quat("-Z", "Y").to_euler()
    cam.data.lens = lens
    if ortho:
        cam.data.type = "ORTHO"
        cam.data.ortho_scale = ortho
    else:
        cam.data.type = "PERSP"
    return cam


def render(sc, path, samples, res):
    w, h = (int(v) for v in res.split("x"))
    sc.render.resolution_x, sc.render.resolution_y = w, h
    sc.cycles.samples = samples
    if path.suffix == ".jpg":
        sc.render.image_settings.file_format = "JPEG"
        sc.render.image_settings.quality = 88
    else:
        sc.render.image_settings.file_format = "PNG"
    sc.render.filepath = str(path)
    bpy.ops.render.render(write_still=True)
    print("wrote", path, flush=True)


def explode(objs, factor):
    for ob in objs.values():
        base = ob.get("base_loc")
        if base is None:
            ob["base_loc"] = list(ob.location)
            base = ob["base_loc"]
        e = ob["explode"]
        ob.location = Vector(base) + cad_to_blender((e[0] * factor, e[1] * factor, e[2] * factor))


def section(objs, on=True):
    """Cut away the right half (x > 0 in CAD) with a boolean on every part."""
    cutter = bpy.data.objects.get("cutter")
    if cutter is None:
        bpy.ops.mesh.primitive_cube_add(size=1, location=(0.6, -0.4, 0))
        cutter = bpy.context.active_object
        cutter.name = "cutter"
        cutter.scale = (1.2, 2.0, 1.2)
        cutter.hide_render = True
        cutter.display_type = "WIRE"
    for ob in objs.values():
        mod = ob.modifiers.get("section")
        if on and mod is None:
            mod = ob.modifiers.new("section", "BOOLEAN")
            mod.operation = "DIFFERENCE"
            mod.object = cutter
            mod.solver = "EXACT" if hasattr(mod, "solver") else mod.solver
        elif not on and mod is not None:
            ob.modifiers.remove(mod)


def main():
    a = args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    shots = a.shots.split(",")
    sc = reset(a.device)
    objs = load_parts()
    world_and_lights()
    hero_cam = (1.55, -2.35, 0.75)
    if "hero" in shots:
        camera(hero_cam, lens=45)
        render(sc, out / "hero.jpg", a.samples, a.res)
    if "front" in shots:
        camera((0, -4.0, 0), target=(0, 0, 0), lens=50, ortho=0.95)
        render(sc, out / "front.jpg", a.samples, "1400x1400")
    if "detail" in shots:
        camera((0.28, -1.45, 0.16), target=(0, -0.76, 0), lens=60)
        render(sc, out / "detail.jpg", a.samples, a.res)
    if "rear" in shots:
        for n in ("back",):
            objs[n].hide_render = True
        camera((-1.2, 1.9, 0.9), target=(0, -0.2, 0), lens=45)
        render(sc, out / "rear-open.jpg", a.samples, a.res)
        objs["back"].hide_render = False
    if "exploded" in shots:
        explode(objs, 1.0)
        camera((2.15, -2.75, 1.15), target=(0, -0.42, 0.0), lens=40)
        render(sc, out / "exploded.jpg", a.samples, a.res)
        explode(objs, 0.0)
    if "section" in shots:
        section(objs, True)
        camera((2.3, -1.2, 0.85), target=(0, -0.38, 0), lens=42)
        render(sc, out / "section.jpg", a.samples, a.res)
        camera((3.5, -0.38, 0.0), target=(0, -0.40, 0), lens=50, ortho=1.5)
        render(sc, out / "section-side.jpg", a.samples, "1800x1000")
        section(objs, False)
    if "turntable" in shots:
        frames = out / "turntable"
        frames.mkdir(exist_ok=True)
        r = math.hypot(*hero_cam[:2])
        for i in range(36):
            ang = math.radians(-35 + i * 10)
            camera((r * math.sin(ang), -r * math.cos(ang), 0.75), lens=45)
            render(sc, frames / f"f{i:03d}.png", max(24, a.samples // 6), "720x480")


main()
