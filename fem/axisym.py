"""Axisymmetric Helmholtz FEM of one Sunflower case, run inside dolfinx/dolfinx:v0.8.0.

Usage (inside the container, complex PETSc):
    source /usr/local/bin/dolfinx-complex-mode
    python3 fem/axisym.py case.json out.csv [chunk nchunks]

With ``chunk`` and ``nchunks`` the process solves every nchunks-th frequency
starting at ``chunk``, so several processes can share one case.

case.json (written by sunflower.fem):
    points   : [[r, z, h], ...] closed outline of the air domain in the (r, z)
               half plane, with the target element size h at each point
    tags     : tag of the edge from point i to point i + 1 (the last edge
               closes the loop). 1 axis, 2 rigid wall, 3 rigid baffle,
               4 far boundary (a circular arc about far.centre), >= 10 driven
    drive    : {tag: weight}. A driven edge moves along +z with velocity
               weight x v, v = 1 m/s. The normal velocity into the air is
               -weight n_z, with n the outward normal of the air domain, so a
               curved cone is driven correctly.
    far      : {"zb": z of the baffle plane, "R": radius of the arc}
    kh       : {"R": radius of the Kirchhoff-Helmholtz arc, "n": Gauss points}
    flux     : [r0, r1, z]: volume velocity through this line (mouth plane)
    felt     : optional {"eta", "r_max", "z_max"}: a lossy layer (wavenumber
               k (1 - j eta), rho c unchanged) where r < r_max and z < z_max
    freqs, angles, scale (multiplies every h), h_max

Physics: (grad p . grad q - k^2 p q) r dx over the half plane, with
dp/dn = -(ik + 1/R) p on the arc (first-order Bayliss-Turkel), dp/dn = 0 on
rigid walls and the baffle, and dp/dn = -i w rho (v . n) on driven edges.
Time convention exp(+i w t).

Outputs per frequency:
    zrad  : mechanical radiation impedance on the driven surface,
            Z = -(1/v) sum_tags weight * integral(p n_z dS)  (N s/m). This is
            the force the air puts on the moving parts, as in the network.
    umouth: volume velocity through the flux line (m^3/s per m/s).
    power_arc: active power through the Kirchhoff-Helmholtz arc for unit
            velocity (W). Without losses it equals Re(zrad) / 2.
    h<deg>: far-field pressure at 1 m, exp(-ikr) removed, phase referenced to
            the point (0, zb), per unit velocity. It comes from the
            Kirchhoff-Helmholtz integral over a hemispherical arc of radius
            kh.R with the half-space (rigid baffle) Green's function, so a
            body that protrudes past the baffle is allowed.
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
from scipy.special import j0, j1

RHO, C = 1.225, 343.0
AXIS, WALL, BAFFLE, FAR = 1, 2, 3, 4


def build_mesh(case):
    pts, tags = case["points"], case["tags"]
    zb = case["far"]["zb"]
    sc = case.get("scale", 1.0)
    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 0)
    geo = gmsh.model.geo
    ids = [geo.addPoint(r, z, 0, h * sc) for r, z, h in pts]
    cen = geo.addPoint(0, zb, 0, case["h_max"] * sc)
    by_tag = {}
    lines = []
    n = len(ids)
    for i in range(n):
        a, b = ids[i], ids[(i + 1) % n]
        ln = geo.addCircleArc(a, cen, b) if tags[i] == FAR else geo.addLine(a, b)
        lines.append(ln)
        by_tag.setdefault(int(tags[i]), []).append(ln)
    surf = geo.addPlaneSurface([geo.addCurveLoop(lines)])
    geo.synchronize()
    gmsh.model.addPhysicalGroup(2, [surf], 1)
    for t, ls in by_tag.items():
        gmsh.model.addPhysicalGroup(1, ls, t)
    gmsh.option.setNumber("Mesh.MeshSizeMax", case["h_max"] * sc)
    gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", 0)
    gmsh.option.setNumber("Mesh.Algorithm", 6)
    gmsh.model.mesh.generate(2)
    mesh, _, facets = gmshio.model_to_mesh(gmsh.model, MPI.COMM_SELF, 0, gdim=2)
    gmsh.finalize()
    return mesh, facets


class Probe:
    """Evaluate p and grad p at fixed points."""

    def __init__(self, mesh, pts2):
        self.pts = np.zeros((len(pts2), 3))
        self.pts[:, :2] = pts2
        tree = geometry.bb_tree(mesh, mesh.topology.dim)
        coll = geometry.compute_colliding_cells(mesh, geometry.compute_collisions_points(tree, self.pts), self.pts)
        cells = []
        for i in range(len(self.pts)):
            c = coll.links(i)
            if len(c) == 0:
                raise RuntimeError(f"probe point {self.pts[i, :2]} is outside the mesh")
            cells.append(c[0])
        self.cells = np.array(cells, dtype=np.int32)

    def __call__(self, fn):
        return fn.eval(self.pts, self.cells)


def main(case_path, out_path, chunk=0, nchunks=1):
    t0 = time.time()
    case = json.load(open(case_path))
    mesh, facets = build_mesh(case)
    t_mesh = time.time() - t0
    zb, R = case["far"]["zb"], case["far"]["R"]
    V = fem.functionspace(mesh, ("Lagrange", 2))
    G = fem.functionspace(mesh, ("DG", 1, (2,)))
    p, q = ufl.TrialFunction(V), ufl.TestFunction(V)
    x = ufl.SpatialCoordinate(mesh)
    r = x[0]
    nrm = ufl.FacetNormal(mesh)
    ds = ufl.Measure("ds", domain=mesh, subdomain_data=facets)
    felt = case.get("felt")
    if felt and felt["eta"] > 0:
        # lossy layer: rho_e = rho (1 - j eta), K_e = K / (1 - j eta), so k_e = k (1 - j eta) and rho c is kept,
        # as in the network's lossy ducts. Multiplied through by rho: stiffness / (1 - j eta), mass x (1 - j eta).
        inside = ufl.And(ufl.lt(x[0], felt["r_max"]), ufl.lt(x[1], felt["z_max"]))
        e = PETSc.ScalarType(1 - 1j * felt["eta"])
        alpha = ufl.conditional(inside, 1 / e, 1.0)
        beta = ufl.conditional(inside, e, 1.0)
    else:
        alpha = beta = 1.0
    K = assemble_matrix(fem.form(alpha * ufl.inner(ufl.grad(p), ufl.grad(q)) * r * ufl.dx))
    M = assemble_matrix(fem.form(beta * ufl.inner(p, q) * r * ufl.dx))
    B = assemble_matrix(fem.form(ufl.inner(p, q) * r * ds(FAR)))
    for A_ in (K, M, B):
        A_.assemble()
    drive = {int(k): float(w) for k, w in case["drive"].items()}
    L0 = sum(-w * nrm[1] * ufl.conj(q) * r * ds(t) for t, w in drive.items())
    b0 = assemble_vector(fem.form(L0))
    b0.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
    b0arr = b0.array.copy()

    ph = fem.Function(V)
    gh = fem.Function(G)
    gexpr = fem.Expression(ufl.grad(ph), G.element.interpolation_points())

    # Kirchhoff-Helmholtz arc about (0, zb)
    Rk, nk = case["kh"]["R"], case["kh"]["n"]
    gx, gw = np.polynomial.legendre.leggauss(nk)
    psi = (gx + 1) * np.pi / 4
    wpsi = gw * np.pi / 4
    arc_r, arc_z = Rk * np.sin(psi), Rk * np.cos(psi)
    arc = Probe(mesh, np.column_stack([arc_r, zb + arc_z]))
    # flux line
    r0, r1, zf = case["flux"]
    fx, fw = np.polynomial.legendre.leggauss(80)
    fr = r0 + (r1 - r0) * (fx + 1) / 2
    fwt = fw * (r1 - r0) / 2 * 2 * np.pi * fr
    flux = Probe(mesh, np.column_stack([fr, np.full_like(fr, zf)]))

    freqs = case["freqs"][chunk::nchunks]
    rows = []
    ksp = PETSc.KSP().create(MPI.COMM_SELF)
    ksp.setType("preonly")
    ksp.getPC().setType("lu")
    ksp.getPC().setFactorSolverType("mumps")
    b = b0.duplicate()
    for f in freqs:
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
        zrad = 2 * np.pi * np.dot(b0arr, ph.x.array)
        gh.interpolate(gexpr)
        # mouth flux
        dpdz = flux(gh)[:, 1]
        umouth = np.sum(-dpdz / (1j * w * RHO) * fwt)
        # far field
        pv = arc(ph).ravel()
        g = arc(gh)
        nr, nz = np.sin(psi), np.cos(psi)
        dpdn = g[:, 0] * nr + g[:, 1] * nz
        dl = Rk * wpsi
        # active power through the arc, for unit velocity (compare with Re(zrad) / 2)
        vn = -dpdn / (1j * w * RHO)
        parc = 0.5 * np.real(np.sum(pv * np.conj(vn) * 2 * np.pi * arc_r * dl))
        row = dict(f=f, zrad_re=zrad.real, zrad_im=zrad.imag, umouth_re=umouth.real, umouth_im=umouth.imag,
                   power_arc=parc)
        for th in case["angles"]:
            st, ct = np.sin(np.radians(th)), np.cos(np.radians(th))
            a = k * arc_r * st
            J0, J1 = j0(a), j1(a)
            ep, em = np.exp(1j * k * arc_z * ct), np.exp(-1j * k * arc_z * ct)
            term = pv * 1j * k * (ep * (1j * nr * st * J1 + nz * ct * J0) + em * (1j * nr * st * J1 - nz * ct * J0)) \
                - dpdn * J0 * (ep + em)
            P = 0.5 * np.sum(term * arc_r * dl)
            row[f"h{int(th)}_re"], row[f"h{int(th)}_im"] = P.real, P.imag
        rows.append(row)
        print(f"{f:8.1f} Hz  |Z|={abs(zrad):.4g}", flush=True)
    keys = list(rows[0]) if rows else ["f"]
    with open(out_path, "w") as fh:
        fh.write(",".join(keys) + "\n")
        for row in rows:
            fh.write(",".join(f"{row[k]:.10g}" for k in keys) + "\n")
    hmin = mesh.h(2, np.arange(mesh.topology.index_map(2).size_local, dtype=np.int32))
    meta = dict(cells=int(mesh.topology.index_map(2).size_local), dofs=int(V.dofmap.index_map.size_global),
                h_min_m=float(hmin.min()), h_max_m=float(hmin.max()), mesh_s=t_mesh, total_s=time.time() - t0,
                n_freqs=len(freqs))
    with open(out_path + ".meta.json", "w") as fh:
        json.dump(meta, fh)
    print(json.dumps(meta))


if __name__ == "__main__":
    a = sys.argv[1:]
    main(a[0], a[1], *(int(v) for v in a[2:4]))
