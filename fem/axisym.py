"""Axisymmetric FEM of one horn section, run inside dolfinx/dolfinx:v0.8.0.

Usage (inside the container, complex PETSc):
    source /usr/local/bin/dolfinx-complex-mode
    python3 axisym.py case.json out.csv

case.json:
    z, r_outer, r_inner : horn profile samples (m), throat at z[0], mouth at z[-1]
    freqs               : frequencies (Hz)
    radius              : exterior half-space radius (m)
    h_horn, h_far       : element sizes (m) for P2 elements
    angles              : far-field angles (deg)

The horn mouth sits in an infinite rigid baffle at z = z[-1]. Unit normal
velocity drives the annular throat. The outer boundary of the half-space
carries a first-order Bayliss-Turkel absorbing condition. The far field at
1 m is the Rayleigh integral of the normal velocity across the mouth.

Output CSV: f, z_in (acoustic, Pa s/m^3), and p(theta)/U_in for each angle.
"""
import json
import sys

import numpy as np
import gmsh
import ufl
from mpi4py import MPI
from petsc4py import PETSc
from dolfinx import fem, geometry
from dolfinx.fem.petsc import LinearProblem
from dolfinx.io import gmshio
from scipy.special import j0

RHO, C = 1.225, 343.0
INLET, FAR = 1, 2


def build_mesh(case):
    z = np.asarray(case["z"])
    ro = np.asarray(case["r_outer"])
    ri = np.asarray(case["r_inner"])
    zm, R = z[-1], case["radius"]
    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 0)
    occ = gmsh.model.occ
    # horn interior polygon: inner profile up, outer profile down
    pts = [(ri[0], z[0])] + [(ro[i], z[i]) for i in range(len(z))] + \
          [(ri[i], z[i]) for i in range(len(z) - 1, 0, -1)]
    clean = []
    for p in pts:
        if not clean or np.hypot(p[0] - clean[-1][0], p[1] - clean[-1][1]) > 1e-6:
            clean.append(p)
    tags = [occ.addPoint(x, y, 0) for x, y in clean]
    lines = [occ.addLine(tags[i], tags[(i + 1) % len(tags)]) for i in range(len(tags))]
    horn = occ.addPlaneSurface([occ.addCurveLoop(lines)])
    disk = occ.addDisk(0, zm, 0, R, R)
    box = occ.addRectangle(0, zm, 0, R, R)
    outer, _ = occ.intersect([(2, disk)], [(2, box)])
    occ.fragment([(2, horn)], outer)
    occ.synchronize()
    surfaces = [s[1] for s in gmsh.model.getEntities(2)]
    gmsh.model.addPhysicalGroup(2, surfaces, 1)
    inlet, far = [], []
    for _, c in gmsh.model.getEntities(1):
        x0, y0, _, x1, y1, _ = gmsh.model.getBoundingBox(1, c)
        cx, cy, _ = gmsh.model.occ.getCenterOfMass(1, c)
        if abs(cy - z[0]) < 1e-6 and (y1 - y0) < 1e-5 and ri[0] - 1e-6 < cx < ro[0] + 1e-6:
            inlet.append(c)
        if abs(np.hypot(cx, cy - zm) - R) < 0.2 * R and y1 > zm + 0.5 * R and x1 > 0.5 * R:
            far.append(c)
    gmsh.model.addPhysicalGroup(1, inlet, INLET)
    gmsh.model.addPhysicalGroup(1, far, FAR)
    gmsh.model.mesh.field.add("Distance", 1)
    gmsh.model.mesh.field.setNumbers(1, "PointsList", tags)
    gmsh.model.mesh.field.add("Threshold", 2)
    gmsh.model.mesh.field.setNumber(2, "InField", 1)
    gmsh.model.mesh.field.setNumber(2, "SizeMin", case["h_horn"])
    gmsh.model.mesh.field.setNumber(2, "SizeMax", case["h_far"])
    gmsh.model.mesh.field.setNumber(2, "DistMin", 0.05 * R)
    gmsh.model.mesh.field.setNumber(2, "DistMax", 0.6 * R)
    gmsh.model.mesh.field.setAsBackgroundMesh(2)
    gmsh.option.setNumber("Mesh.MeshSizeExtendFromBoundary", 0)
    gmsh.model.mesh.generate(2)
    mesh, _, facets = gmshio.model_to_mesh(gmsh.model, MPI.COMM_WORLD, 0, gdim=2)
    gmsh.finalize()
    return mesh, facets


def main(case_path, out_path):
    case = json.load(open(case_path))
    mesh, facets = build_mesh(case)
    z = np.asarray(case["z"]); ro = np.asarray(case["r_outer"]); ri = np.asarray(case["r_inner"])
    zm, R = z[-1], case["radius"]
    V = fem.functionspace(mesh, ("Lagrange", 2))
    W = fem.functionspace(mesh, ("DG", 1))
    p, q = ufl.TrialFunction(V), ufl.TestFunction(V)
    x = ufl.SpatialCoordinate(mesh)
    r = x[0]
    ds = ufl.Measure("ds", domain=mesh, subdomain_data=facets)
    k_c = fem.Constant(mesh, PETSc.ScalarType(1.0))
    w_c = fem.Constant(mesh, PETSc.ScalarType(1.0))
    a = (ufl.inner(ufl.grad(p), ufl.grad(q)) - k_c**2 * ufl.inner(p, q)) * r * ufl.dx \
        + (1j * k_c + 1.0 / R) * ufl.inner(p, q) * r * ds(FAR)
    L = 1j * w_c * RHO * ufl.conj(q) * r * ds(INLET)   # unit inflow velocity

    s_in = np.pi * (ro[0] ** 2 - ri[0] ** 2)
    ones = fem.form(r * ds(INLET))
    r_in = MPI.COMM_WORLD.allreduce(fem.assemble_scalar(ones), op=MPI.SUM)
    # Gauss points across the mouth aperture, a hair outside the mouth plane
    gx, gw = np.polynomial.legendre.leggauss(80)
    a0, a1 = ri[-1], ro[-1]
    rr = a0 + (a1 - a0) * (gx + 1) / 2
    ww = gw * (a1 - a0) / 2
    pts = np.zeros((len(rr), 3)); pts[:, 0] = rr; pts[:, 1] = zm + 1e-4
    tree = geometry.bb_tree(mesh, mesh.topology.dim)
    coll = geometry.compute_colliding_cells(mesh, geometry.compute_collisions_points(tree, pts), pts)
    cells = np.array([coll.links(i)[0] for i in range(len(pts))], dtype=np.int32)

    rows = []
    for f in case["freqs"]:
        k = 2 * np.pi * f / C
        k_c.value = k
        w_c.value = 2 * np.pi * f
        prob = LinearProblem(a, L, petsc_options={"ksp_type": "preonly", "pc_type": "lu",
                                                  "pc_factor_mat_solver_type": "mumps"})
        ph = prob.solve()
        p_in = fem.assemble_scalar(fem.form(ph * r * ds(INLET))) / r_in
        u_in = 1.0 * s_in
        dpdz = fem.Function(W)
        dpdz.interpolate(fem.Expression(ph.dx(1), W.element.interpolation_points()))
        vz = -dpdz.eval(pts, cells).ravel() / (1j * 2 * np.pi * f * RHO)
        u_ratio = np.sum(vz * 2 * np.pi * rr * ww) / u_in
        row = dict(f=f, z_in_re=(p_in / u_in).real, z_in_im=(p_in / u_in).imag,
                   u_ratio_re=u_ratio.real, u_ratio_im=u_ratio.imag)
        for th in case["angles"]:
            integ = np.sum(vz * j0(k * rr * np.sin(np.radians(th))) * 2 * np.pi * rr * ww)
            pf = 1j * 2 * np.pi * f * RHO / (2 * np.pi * 1.0) * integ / u_in
            row[f"h{int(th)}_re"], row[f"h{int(th)}_im"] = pf.real, pf.imag
        rows.append(row)
        print(f"{f:8.1f} Hz  |Zin|={abs(p_in/u_in):.4g}", flush=True)
    keys = list(rows[0])
    with open(out_path, "w") as fh:
        fh.write(",".join(keys) + "\n")
        for row in rows:
            fh.write(",".join(f"{row[k]:.9g}" for k in keys) + "\n")
    print(f"cells {mesh.topology.index_map(2).size_local}, dofs {V.dofmap.index_map.size_global}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
