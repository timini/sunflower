"""Slice the assembled CAD model (the same part meshes the renderer uses) to measure the real horn expansion.

Every part in cad/parts.json is loaded from cad/stl/. At each plane z the walls give the inside of the
cabinet; every other part (wedges, corner fillers, centre body sections, phase plug, felt, vanes,
standoffs, driver cones, baskets and motors) is cut out of it. What is left is air. Air touching the
cabinet walls is the kick horn; air inside the mid horn wall is the mid horn; anything else is a sealed
cavity (motor enclosure, mid pod) and is reported separately. Each 1 cm slab is measured with three
planes (Simpson's rule), so slab volume / 1 cm is the open area of that slab.

Writes results/cad/slices.csv and docs/img/plots/slices.png.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import trimesh
from shapely.geometry import Point, Polygon
from shapely.ops import unary_union

ROOT = Path(__file__).resolve().parents[2]
CAD = None  # set from design at run time


def load_parts():
    from sunflower import design as D
    global CAD
    CAD = D.CAD_DIR
    parts = json.loads((CAD / "parts.json").read_text())
    meshes = {}
    for p in parts:
        f = CAD / "stl" / f"{p['name']}.stl"
        if f.exists():
            meshes[p["name"]] = (p.get("category", ""), trimesh.load_mesh(f, process=True))
    return meshes


def section(mesh, z):
    """Cross-section of a mesh at plane z as a shapely region. Nested loops combine by the even-odd rule,
    so a hollow part (a printed shell) comes out as a ring."""
    s = mesh.section(plane_origin=[0, 0, z], plane_normal=[0, 0, 1])
    if s is None:
        return None
    region = None
    for loop in s.discrete:
        if len(loop) < 4:
            continue
        poly = Polygon(loop[:, :2]).buffer(0)
        if poly.is_empty or poly.area < 1e-3:
            continue
        region = poly if region is None else region.symmetric_difference(poly)
    return region


def is_wall(name):
    return name.startswith("wall") or name in ("top", "bottom")


def inside_and_solid(meshes, z):
    """Inside of the cabinet walls at plane z, and the union of every other part's section.

    The tapered wedges under the flat top and over the flat bottom are sealed boxes (sloping panel, ribs and
    side closures against the top or bottom panel), so each wedge section is taken as its convex hull: the
    air sealed inside a wedge is not horn air."""
    walls, solids, wedges = [], [], []
    for name, (cat, m) in meshes.items():
        lo, hi = m.bounds[:, 2]
        if not lo < z < hi:
            continue
        sec = section(m, z)
        if sec is None:
            continue
        if name.startswith("wedge"):
            sec = sec.convex_hull
            wedges.append(sec)
        (walls if is_wall(name) else solids).append(sec)
    if not walls:
        return None, None, None
    w = unary_union(walls)
    if w.geom_type == "MultiPolygon":
        w = unary_union([p.buffer(0.5) for p in w.geoms]).buffer(-0.5)
    rings = [Polygon(r) for poly in getattr(w, "geoms", [w]) for r in poly.interiors]
    if not rings:
        return None, None, None
    inside = max(rings, key=lambda r: r.area)
    solid = unary_union(solids) if solids else Polygon()
    horn = inside.difference(unary_union(wedges)) if wedges else inside
    if horn.geom_type != "Polygon":
        horn = max(horn.geoms, key=lambda g: g.area)
    return inside, solid, horn


def air_at(meshes, z, mid_radius, plug_radius=0.0):
    inside, solid, horn = inside_and_solid(meshes, z)
    if inside is None:
        return None
    air = inside.difference(solid)
    kick = mid = hf = sealed = 0.0
    edge = horn.exterior.buffer(2.0)          # the horn walls: side walls and wedge faces
    kind = []
    for piece in getattr(air, "geoms", [air]):
        if piece.area < 1.0:
            continue
        rc = piece.centroid.distance(Point(0, 0))
        if piece.intersects(edge):
            kick += piece.area
            kind.append((piece, "kick"))
        elif mid_radius and plug_radius and piece.within(Point(0, 0).buffer(plug_radius - 0.5)):
            hf += piece.area                       # the HF unit's own horn bore inside the bullet
            kind.append((piece, "hf"))
        elif mid_radius and rc < mid_radius and piece.intersects(Point(0, 0).buffer(mid_radius).difference(
                Point(0, 0).buffer(plug_radius))):
            # only the air inside the mid horn wall is mid horn: a cable hole can join the bore to the rear
            # chamber in the hollow body, and that chamber is sealed air
            inner = piece.intersection(Point(0, 0).buffer(mid_radius + 0.5, quad_segs=256))
            mid += inner.area
            sealed += piece.area - inner.area
            kind.append((piece, "mid"))
        else:
            sealed += piece.area
            kind.append((piece, "sealed"))
    return dict(kick_mm2=kick, mid_mm2=mid, hf_mm2=hf, sealed_mm2=sealed, inside_mm2=inside.area, solid=solid, air=air,
                inside=inside, kind=kind)


def run(dz=10.0, z0=None, z1=None, meshes=None, write=True):
    from sunflower import design as D
    K, M = D.KICK, D.MID
    meshes = load_parts() if meshes is None else meshes
    z0 = 0.5 if z0 is None else z0
    z1 = (K.mouth_z * 1000 - 0.5) if z1 is None else z1
    rows = []
    z = z0
    while z < z1 - 1e-6:
        h = min(dz, z1 - z)
        vals = []
        for zz in (z + 0.137 * h, z + 0.513 * h, z + 0.871 * h):     # off the round-number joint faces
            s = (zz / 1000 - K.mid_throat_z)
            ok = 0 <= s <= M.length
            mr = M.radius(min(max(s, 0), M.length)) * 1000 if ok else 0.0
            pr = M.plug(s) * 1000 if ok else 0.0
            vals.append(air_at(meshes, zz, mr, pr))
        if all(v is not None for v in vals):
            avg = {k: sum(v[k] for v in vals) / 3 for k in ("kick_mm2", "mid_mm2", "hf_mm2", "sealed_mm2", "inside_mm2")}
            zc = (z + h / 2) / 1000
            s = zc - K.mid_throat_z
            rows.append(dict(z_mm=z + h / 2, slab_mm=h, kick_cm2=avg["kick_mm2"] / 100, mid_cm2=avg["mid_mm2"] / 100,
                             hf_bore_cm2=avg["hf_mm2"] / 100, sealed_cm2=avg["sealed_mm2"] / 100, inside_walls_cm2=avg["inside_mm2"] / 100,
                             kick_model_cm2=K.area(zc) * 1e4 if zc >= K.turn_z else np.nan,
                             mid_model_cm2=M.area(s) * 1e4 if 0 <= s <= M.length else np.nan))
        z += h
    df = pd.DataFrame(rows)
    df["kick_diff_pct"] = 100 * (df.kick_cm2 / df.kick_model_cm2 - 1)
    df["mid_diff_pct"] = 100 * (df.mid_cm2 / df.mid_model_cm2 - 1)
    if write:
        out = D.RESULTS / "cad"
        out.mkdir(parents=True, exist_ok=True)
        df.to_csv(out / "slices.csv", index=False, float_format="%.4g")
    return df, meshes


def figure(df, meshes):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sunflower import design as D
    K, M = D.KICK, D.MID
    fig = plt.figure(figsize=(14, 10.5))
    gs = fig.add_gridspec(2, 4, height_ratios=(1.15, 1))
    ax = fig.add_subplot(gs[0, :])
    ax.bar(df.z_mm, df.kick_cm2, width=df.slab_mm * 0.9, color="#2a78d6", alpha=0.6, label="Kick horn air, sliced from the model")
    ax.bar(df.z_mm, df.mid_cm2, width=df.slab_mm * 0.9, color="#eb6834", alpha=0.75, label="Mid horn air, sliced from the model")
    ax.axvspan(0, K.turn_z * 1000, color="#eeeeee", lw=0)
    ax.text(5, ax.get_ylim()[1] * 0.92 if ax.get_ylim()[1] > 0 else 1, "cone channel\nand turn", fontsize=8, color="#52514e")
    ax.plot(df.z_mm, df.kick_model_cm2, color="#0b3d7a", lw=1.4, label="Kick design law")
    ax.plot(df.z_mm, df.mid_model_cm2, color="#8a3510", lw=1.4, label="Mid design law")
    ax.set(xlabel="z, mm from the baffle front face", ylabel="open area (cm²), slab volume / slab length",
           title="The assembled model sliced every 1 cm: air left after every part is subtracted")
    ax.legend(loc="upper left", fontsize=9)
    picks = [K.turn_z * 1000 + 10, (K.motor_front + 0.01) * 1000, (K.mid_throat_z + 0.05) * 1000, K.mouth_z * 1000 - 10]
    for i, z in enumerate(picks):
        s = z / 1000 - K.mid_throat_z
        ok = 0 <= s <= M.length
        mr = M.radius(min(max(s, 0), M.length)) * 1000 if ok else 0.0
        a = air_at(meshes, z, mr, M.plug(s) * 1000 if ok else 0.0)
        ax = fig.add_subplot(gs[1, i])
        if a is None:
            continue
        x, y = a["inside"].exterior.xy
        ax.fill(x, y, color="#2b2b2b", lw=0)
        colours = dict(kick="#2a78d6", mid="#eb6834", hf="#1baf7a", sealed="#b9b6ae")
        for piece, kind in a["kind"]:
            x, y = piece.exterior.xy
            ax.fill(x, y, color=colours[kind], lw=0)
            for r in piece.interiors:
                x, y = r.xy
                ax.fill(x, y, color="#2b2b2b", lw=0)
        ax.set(aspect="equal", title=f"z = {z:.0f} mm", xticks=[], yticks=[])
    fig.text(0.01, 0.01, "Bottom: slices through the model, looking into the mouth. Blue: kick horn air. Orange: mid horn air. "
             "Green: the HF horn inside the bullet. Grey: sealed cavities. Black: solid parts (walls, fillers, centre body, "
             "bullet, vanes, drivers).", fontsize=9, color="#52514e")
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(D.IMG / "plots/slices.png", dpi=100)


if __name__ == "__main__":
    df, meshes = run()
    figure(df, meshes)
    k = df.dropna(subset=["kick_model_cm2"])
    k = k[k.kick_cm2 > 0]
    m = df.dropna(subset=["mid_model_cm2"])
    print(f"{len(df)} slabs; kick vs law worst {np.max(np.abs(k.kick_cm2 / k.kick_model_cm2 - 1)) * 100:.1f}%; "
          f"mid vs law worst {np.max(np.abs(m.mid_cm2 / m.mid_model_cm2 - 1)) * 100:.1f}%")
