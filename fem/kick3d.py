"""3D quarter-symmetry Helmholtz FEM of the kick horn, run inside dolfinx/dolfinx:v0.8.0.

Usage (inside the container, complex PETSc):
    python3 fem/kick3d.py case.json out.csv [chunk nchunks]

It checks the equal-area axisymmetric kick model against the real square
frustum with its four corner fillers. Quarter symmetry: x >= 0, y >= 0,
with rigid (natural) symmetry planes.

Air = (frustum interior less the corner filler tetrahedra, 0 <= z <=
mouth_z) + (quarter hemisphere of radius R over the kick mouth plane), less
one solid of revolution: the driver below the cone, surround and flange,
the motor column inside the voice coil, the centre body over the shroud and
the protruding mid horn with its lip (closed at the mid mouth). That solid
comes from ``case["solid"]``, an (r, z) outline in segments.

The cone and surround (half velocity) move along +z; dp/dn = -i w rho v n_z.
The hemisphere carries a first-order Bayliss-Turkel condition. The kick
mouth plane outside the frustum is a rigid baffle.

Outputs per frequency, for unit cone velocity: zrad (N s/m, whole cone) and
the on-axis far field at 1 m from a Kirchhoff-Helmholtz integral over a
hemisphere of radius kh_R with the half-space Green's function (h0).
"""
import json
import sys
import time

import numpy as np
import gmsh
import ufl
from mpi4py import MPI
from petsc4py import PETSc
from dolfinx import fem, geometry
from dolfinx.fem.petsc import assemble_matrix, assemble_vector
from dolfinx.io import gmshio

RHO, C = 1.225, 343.0
WALL, FAR, CONE, SURROUND = 2, 4, 10, 11


def on_polyline(r, z, poly, tol):
    P = np.asarray(poly)
    a, b = P[:-1], P[1:]
    d = b - a
    L2 = np.maximum((d**2).sum(1), 1e-30)
    t = np.clip(((r - a[:, 0]) * d[:, 0] + (z - a[:, 1]) * d[:, 1]) / L2, 0, 1)
    q = a + t[:, None] * d
    return np.min(np.hypot(q[:, 0] - r, q[:, 1] - z)) < tol


def build(case):
    occ = gmsh.model.occ
    zm, R = case["mouth_z"], case["R"]
    hw0, hwm, c0 = case["hw0"], case["hwm"], case["c0"]
    # frustum quarter
    w0 = occ.addWire([occ.addLine(*p) for p in pairs([occ.addPoint(0, 0, 0), occ.addPoint(hw0, 0, 0),
                                                       occ.addPoint(hw0, hw0, 0), occ.addPoint(0, hw0, 0)])])
    w1 = occ.addWire([occ.addLine(*p) for p in pairs([occ.addPoint(0, 0, zm), occ.addPoint(hwm, 0, zm),
                                                       occ.addPoint(hwm, hwm, zm), occ.addPoint(0, hwm, zm)])])
    fr = occ.addThruSections([w0, w1], makeSolid=True, makeRuled=True)
    # corner filler tetrahedron
    A, B, Cc, P = (occ.addPoint(hw0 - c0, hw0, 0), occ.addPoint(hw0, hw0 - c0, 0), occ.addPoint(hw0, hw0, 0),
                   occ.addPoint(hwm, hwm, zm))
    faces = []
    for tri in ((A, B, Cc), (A, B, P), (B, Cc, P), (Cc, A, P)):
        faces.append(occ.addPlaneSurface([occ.addCurveLoop([occ.addLine(*p) for p in pairs(list(tri))])]))
    tet = occ.addVolume([occ.addSurfaceLoop(faces)])
    horn, _ = occ.cut(fr, [(3, tet)])
    ext = occ.addSphere(0, 0, zm, R, -1, 0, np.pi / 2, np.pi / 2)
    air, _ = occ.fuse(horn, [(3, ext)])
    # solid of revolution
    pts, curves = [], []
    first = None
    for seg in case["solid"]:
        tags = []
        for i, (r, z) in enumerate(seg["pts"]):
            if i == 0 and pts:
                tags.append(pts[-1])
                continue
            tags.append(occ.addPoint(r, 0, z))
        pts.append(tags[-1])
        if first is None:
            first = tags[0]
        if seg["kind"] == "spline":
            curves.append(occ.addSpline(tags))
        else:
            curves += [occ.addLine(a, b) for a, b in zip(tags[:-1], tags[1:])]
    curves.append(occ.addLine(pts[-1], first))
    surf = occ.addPlaneSurface([occ.addCurveLoop(curves)])
    rev = occ.revolve([(2, surf)], 0, 0, 0, 0, 0, 1, np.pi / 2)
    vol = [d for d in rev if d[0] == 3]
    out, _ = occ.cut(air, vol)
    occ.synchronize()
    return out


def pairs(tags):
    return list(zip(tags, tags[1:] + tags[:1]))


def classify(case, vol_tags):
    zm, R = case["mouth_z"], case["R"]
    cone, sur = case["cone"], case["surround"]
    groups = {WALL: [], FAR: [], CONE: [], SURROUND: []}
    for _, s in gmsh.model.getBoundary(vol_tags, oriented=False):
        lo, hi = gmsh.model.getParametrizationBounds(2, s)
        uv = []
        for a in (0.2, 0.5, 0.8):
            for b in (0.2, 0.5, 0.8):
                uv += [lo[0] + a * (hi[0] - lo[0]), lo[1] + b * (hi[1] - lo[1])]
        xyz = np.array(gmsh.model.getValue(2, s, uv)).reshape(-1, 3)
        r, z = np.hypot(xyz[:, 0], xyz[:, 1]), xyz[:, 2]
        if np.all(np.abs(np.sqrt(r**2 + (z - zm) ** 2) - R) < 1e-3 * R) and np.all(z > zm - 1e-6):
            groups[FAR].append(s)
        elif all(on_polyline(ri, zi, cone, 2e-4) for ri, zi in zip(r, z)):
            groups[CONE].append(s)
        elif np.all(np.abs(z - sur[0][1]) < 1e-6) and abs(gmsh.model.occ.getCenterOfMass(2, s)[2] - sur[0][1]) < 1e-6:
            groups[SURROUND].append(s)   # the only flat face at the surround height
        else:
            groups[WALL].append(s)
    return groups


def mesh_sizes(case):
    f = gmsh.model.mesh.field
    ids = []
    for i, b in enumerate(case["boxes"]):
        t = f.add("Box")
        f.setNumber(t, "VIn", b["h"])
        f.setNumber(t, "VOut", case["h_far"])
        for k, v in zip(("XMin", "XMax", "YMin", "YMax", "ZMin", "ZMax"), b["box"]):
            f.setNumber(t, k, v)
        f.setNumber(t, "Thickness", b.get("thickness", 0.05))
        ids.append(t)
    m = f.add("Min")
    f.setNumbers(m, "FieldsList", ids)
    f.setAsBackgroundMesh(m)
    gmsh.option.setNumber("Mesh.MeshSizeExtendFromBoundary", 0)
    gmsh.option.setNumber("Mesh.MeshSizeFromPoints", 0)
    gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", 0)
    gmsh.option.setNumber("Mesh.MeshSizeMax", case["h_far"])


def main(case_path, out_path, chunk=0, nchunks=1):
    t0 = time.time()
    case = json.load(open(case_path))
    sc = case.get("scale", 1.0)
    for b in case["boxes"]:
        b["h"] *= sc
    case["h_far"] *= sc
    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 0)
    vols = build(case)
    groups = classify(case, vols)
    gmsh.model.addPhysicalGroup(3, [v[1] for v in vols], 1)
    for t, ss in groups.items():
        if ss:
            gmsh.model.addPhysicalGroup(2, ss, t)
    mesh_sizes(case)
    gmsh.option.setNumber("Mesh.Algorithm3D", 10)
    gmsh.model.mesh.generate(3)
    mesh, _, facets = gmshio.model_to_mesh(gmsh.model, MPI.COMM_SELF, 0, gdim=3)
    gmsh.finalize()
    t_mesh = time.time() - t0
    R, zm = case["R"], case["mouth_z"]
    V = fem.functionspace(mesh, ("Lagrange", 2))
    G = fem.functionspace(mesh, ("DG", 1, (3,)))
    p, q = ufl.TrialFunction(V), ufl.TestFunction(V)
    nrm = ufl.FacetNormal(mesh)
    ds = ufl.Measure("ds", domain=mesh, subdomain_data=facets)
    K = assemble_matrix(fem.form(ufl.inner(ufl.grad(p), ufl.grad(q)) * ufl.dx))
    M = assemble_matrix(fem.form(ufl.inner(p, q) * ufl.dx))
    B = assemble_matrix(fem.form(ufl.inner(p, q) * ds(FAR)))
    for A_ in (K, M, B):
        A_.assemble()
    b0 = assemble_vector(fem.form(-nrm[2] * ufl.conj(q) * ds(CONE) - 0.5 * nrm[2] * ufl.conj(q) * ds(SURROUND)))
    b0arr = b0.array.copy()
    area = fem.assemble_scalar(fem.form(-nrm[2] * ds(CONE) - 0.5 * nrm[2] * ds(SURROUND)))
    ph = fem.Function(V)
    gh = fem.Function(G)
    gexpr = fem.Expression(ufl.grad(ph), G.element.interpolation_points())
    # quarter hemisphere quadrature at kh_R: Gauss in polar angle, Gauss in azimuth
    Rk = case["kh_R"]
    gx, gw = np.polynomial.legendre.leggauss(60)
    psi, wpsi = (gx + 1) * np.pi / 4, gw * np.pi / 4
    ax, aw = np.polynomial.legendre.leggauss(30)
    phi, wphi = (ax + 1) * np.pi / 4, aw * np.pi / 4
    PS, PH = np.meshgrid(psi, phi, indexing="ij")
    W = np.outer(wpsi, wphi) * Rk**2 * np.sin(PS)
    nx, ny, nz = np.sin(PS) * np.cos(PH), np.sin(PS) * np.sin(PH), np.cos(PS)
    pts = np.column_stack([(Rk * nx).ravel(), (Rk * ny).ravel(), zm + (Rk * nz).ravel()])
    tree = geometry.bb_tree(mesh, 3)
    coll = geometry.compute_colliding_cells(mesh, geometry.compute_collisions_points(tree, pts), pts)
    cells = np.array([coll.links(i)[0] for i in range(len(pts))], dtype=np.int32)
    zp = (Rk * nz).ravel()
    nzr, Wr = nz.ravel(), W.ravel()
    ksp = PETSc.KSP().create(MPI.COMM_SELF)
    ksp.setType("preonly")
    ksp.getPC().setType("lu")
    ksp.getPC().setFactorSolverType("mumps")
    b = b0.duplicate()
    rows = []
    for f in case["freqs"][chunk::nchunks]:
        w = 2 * np.pi * f
        k = w / C
        A = K.copy()
        A.axpy(-k * k, M, structure=PETSc.Mat.Structure.SUBSET_NONZERO_PATTERN)
        A.axpy(1j * k + 1.0 / R, B, structure=PETSc.Mat.Structure.SUBSET_NONZERO_PATTERN)
        b.array[:] = 1j * w * RHO * b0arr
        ksp.setOperators(A)
        ksp.solve(b, ph.x.petsc_vec)
        ph.x.scatter_forward()
        A.destroy()
        zrad = 4 * np.dot(b0arr, ph.x.array)
        gh.interpolate(gexpr)
        pv = ph.eval(pts, cells).ravel()
        g = gh.eval(pts, cells)
        dpdn = g[:, 0] * nx.ravel() + g[:, 1] * ny.ravel() + g[:, 2] * nzr
        ep, em = np.exp(1j * k * zp), np.exp(-1j * k * zp)
        h0 = 4 / (4 * np.pi) * np.sum((pv * 1j * k * nzr * (ep - em) - dpdn * (ep + em)) * Wr)
        vn = -dpdn / (1j * w * RHO)
        parc = 4 * 0.5 * np.real(np.sum(pv * np.conj(vn) * Wr))
        rows.append(dict(f=f, zrad_re=zrad.real, zrad_im=zrad.imag, h0_re=h0.real, h0_im=h0.imag, power_arc=parc))
        print(f"{f:8.1f} Hz |Z|={abs(zrad):.4g} {time.time() - t0:.0f}s", flush=True)
    keys = list(rows[0]) if rows else ["f"]
    with open(out_path, "w") as fh:
        fh.write(",".join(keys) + "\n")
        for row in rows:
            fh.write(",".join(f"{row[k]:.10g}" for k in keys) + "\n")
    meta = dict(cells=int(mesh.topology.index_map(3).size_local), dofs=int(V.dofmap.index_map.size_global),
                driven_area_m2=4 * float(np.real(area)), mesh_s=t_mesh, total_s=time.time() - t0,
                faces={str(k): len(v) for k, v in groups.items()})
    with open(out_path + ".meta.json", "w") as fh:
        json.dump(meta, fh)
    print(json.dumps(meta))


if __name__ == "__main__":
    a = sys.argv[1:]
    main(a[0], a[1], *(int(v) for v in a[2:4]))
