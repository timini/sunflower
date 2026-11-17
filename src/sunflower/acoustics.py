"""Plane-wave acoustic network for the three Sunflower sections.

Each horn is a chain of thin cylindrical (or annular) duct segments, cascaded
as 2x2 transfer matrices from throat to mouth, with a flanged-piston
radiation load at the mouth. The driver is a lumped Thiele-Small motor with
an optional sealed rear volume and an optional shunt front-chamber volume.

This is the same model family as the horn-simulation Webster/TMM prescreen.
It ignores cross modes, wall losses, cone breakup and diffraction. The FEM
check in ``sunflower.fem`` exists to measure how much that matters.
"""
from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np
from scipy.special import j1, struve

RHO = 1.225
C = 343.0
P_REF = 20e-6


@dataclass(frozen=True)
class Driver:
    driver_id: str
    label: str
    re: float
    le: float
    bl: float
    sd: float
    mms: float
    cms: float
    rms: float
    xmax: float | None = None
    power_w: float | None = None
    fs: float | None = None
    source: str | None = None
    factory_sealed: bool = False

    @classmethod
    def from_record(cls, r: dict) -> "Driver":
        p = r["parameters"]
        ws = 2 * np.pi * p["fs_hz"]
        mms = p["mms_kg"]
        qms = p.get("qms")
        if qms is None and p.get("qes") and p.get("qts") and p["qes"] > p["qts"]:
            qms = p["qes"] * p["qts"] / (p["qes"] - p["qts"])
        return cls(
            driver_id=r["driver_id"],
            label=f"{r['manufacturer']} {r['model_name']}",
            re=p["re_ohm"], le=p["le_h"], bl=p["bl_tm"], sd=p["sd_m2"], mms=mms,
            cms=1.0 / (mms * ws**2), rms=(ws * mms / qms) if qms else 0.0,
            xmax=p.get("xmax_m"), power_w=p.get("power_w"), fs=p["fs_hz"],
            source=r.get("parameter_source"), factory_sealed=bool(r.get("factory_sealed_back")),
        )

    def with_rear_volume(self, litres: float | None) -> "Driver":
        if litres is None:
            return self
        k_box = RHO * C**2 * self.sd**2 / (litres * 1e-3)
        return replace(self, cms=1.0 / (1.0 / self.cms + k_box))


def piston_load(f: np.ndarray, radius: float) -> np.ndarray:
    """Acoustic radiation impedance (Pa s/m^3) of a flanged circular piston."""
    k = 2 * np.pi * np.asarray(f) / C
    x = 2 * k * radius
    r1 = 1 - 2 * j1(x) / x
    x1 = 2 * struve(1, x) / x
    return (r1 + 1j * x1) * RHO * C / (np.pi * radius**2)


def chain(f: np.ndarray, areas: np.ndarray, lengths: np.ndarray):
    """Cascade plane-wave segments. Returns (A, B, Cm, D) arrays over f."""
    k = 2 * np.pi * np.asarray(f) / C
    A = np.ones_like(k, dtype=complex)
    B = np.zeros_like(A)
    Cm = np.zeros_like(A)
    D = np.ones_like(A)
    for S, dz in zip(areas, lengths):
        if S <= 0:
            raise ValueError("Blocked or negative horn section")
        co, si = np.cos(k * dz), np.sin(k * dz)
        zc = RHO * C / S
        A, B, Cm, D = (A * co + B * 1j * si / zc, A * 1j * zc * si + B * co,
                       Cm * co + D * 1j * si / zc, Cm * 1j * zc * si + D * co)
    return A, B, Cm, D


def sample(area_fn, length: float, segments: int, breaks=()):
    """Midpoint sampling of area_fn on [0, length], honouring discontinuities."""
    edges = sorted({0.0, length, *[b for b in breaks if 0 < b < length]})
    areas, lengths = [], []
    for a, b in zip(edges[:-1], edges[1:]):
        n = max(1, int(np.ceil(segments * (b - a) / length)))
        dz = (b - a) / n
        for z in a + (np.arange(n) + 0.5) * dz:
            areas.append(area_fn(z))
            lengths.append(dz)
    return np.array(areas), np.array(lengths)


@dataclass
class HornTransfer:
    f: np.ndarray
    z_in: np.ndarray        # acoustic input impedance at the throat (Pa s/m^3)
    u_ratio: np.ndarray     # U_mouth / U_throat
    load: np.ndarray        # mouth radiation impedance
    throat_area: float
    mouth_area: float


def horn_transfer(f, area_fn, length, segments=300, breaks=(), mouth_area=None) -> HornTransfer:
    areas, lengths = sample(area_fn, length, segments, breaks)
    A, B, Cm, D = chain(f, areas, lengths)
    s_mouth = area_fn(length) if mouth_area is None else mouth_area
    load = piston_load(f, np.sqrt(s_mouth / np.pi))
    pin = A * load + B
    uin = Cm * load + D
    return HornTransfer(np.asarray(f), pin / uin, 1 / uin, load, area_fn(0.0), s_mouth)


_GAUSS = np.polynomial.legendre.leggauss(96)


def far_field(f, u_mouth, mouth_radius, theta_deg=0.0, r=1.0, inner_radius=0.0, wavefront_radius=None):
    """Far-field pressure of a baffled circular or annular aperture.

    With ``wavefront_radius`` the aperture velocity carries the phase of a
    spherical wavefront of that radius (the curved wavefront at a conical
    or pyramidal horn mouth), integrated numerically over the aperture.
    Without it the aperture is a flat piston.

    Phase is referenced to the aperture plane, so exp(-jkr) is removed. The
    caller adds the axial position of each aperture when summing sections.
    """
    f = np.asarray(f)
    k = 2 * np.pi * f / C
    s = np.sin(np.radians(theta_deg))
    if wavefront_radius is not None:
        from scipy.special import j0
        x, w = _GAUSS
        ri = inner_radius + (mouth_radius - inner_radius) * (x + 1) / 2
        wr = w * (mouth_radius - inner_radius) / 2 * 2 * np.pi * ri
        sag = wavefront_radius - np.sqrt(wavefront_radius**2 - ri**2)
        kern = j0(np.outer(k * s, ri)) * np.exp(-1j * np.outer(k, sag))
        weight = (kern @ wr) / (np.pi * (mouth_radius**2 - inner_radius**2))
        return 1j * 2 * np.pi * f * RHO * u_mouth / (2 * np.pi * r) * weight

    def dirc(a):
        x = k * a * s
        with np.errstate(invalid="ignore", divide="ignore"):
            d = np.where(np.abs(x) < 1e-9, 1.0, 2 * j1(x) / x)
        return d

    ao, ai = mouth_radius, inner_radius
    w = (ao**2 * dirc(ao) - ai**2 * (dirc(ai) if ai > 0 else 0)) / (ao**2 - ai**2)
    return 1j * 2 * np.pi * f * RHO * u_mouth / (2 * np.pi * r) * w


def spl(p):
    return 20 * np.log10(np.maximum(np.abs(p), 1e-30) / P_REF)


def solve(driver: Driver, tr: HornTransfer, *, v=2.83, front_m3=0.0, rear_litres=None):
    """Couple a driver to a horn transfer. Returns throat/mouth flows and motor state."""
    f = tr.f
    w = 2 * np.pi * f
    drv = driver.with_rear_volume(rear_litres)
    shunt = 1 + 1j * w * (front_m3 / (RHO * C**2)) * tr.z_in
    z_front = tr.z_in / shunt
    zm = drv.rms + 1j * w * drv.mms + 1 / (1j * w * drv.cms) + drv.sd**2 * z_front
    ze = drv.re + 1j * w * drv.le
    den = ze * zm + drv.bl**2
    vel = drv.bl * v / den
    cur = v * zm / den
    u_cone = vel * drv.sd
    u_throat = u_cone / shunt
    u_mouth = u_throat * tr.u_ratio
    return dict(
        u_cone=u_cone, u_throat=u_throat, u_mouth=u_mouth,
        z_e=v / cur, x_peak_m=np.sqrt(2) * np.abs(vel) / w,
        input_power_w=np.real(v * np.conj(cur)),
        radiated_power_w=np.abs(u_mouth) ** 2 * tr.load.real,
    )


@dataclass
class DriverSet:
    """Many drivers as arrays, for vectorised screening. Shapes (D, 1)."""
    ids: list
    labels: list
    re: np.ndarray
    le: np.ndarray
    bl: np.ndarray
    sd: np.ndarray
    mms: np.ndarray
    cms: np.ndarray
    rms: np.ndarray
    xmax: np.ndarray
    sealed: np.ndarray

    @classmethod
    def from_drivers(cls, drivers):
        col = lambda a: np.array(a, float)[:, None]  # noqa: E731
        return cls([d.driver_id for d in drivers], [d.label for d in drivers],
                   col([d.re for d in drivers]), col([d.le for d in drivers]), col([d.bl for d in drivers]),
                   col([d.sd for d in drivers]), col([d.mms for d in drivers]), col([d.cms for d in drivers]),
                   col([d.rms for d in drivers]), col([d.xmax or np.nan for d in drivers]),
                   np.array([d.factory_sealed for d in drivers])[:, None])

    def subset(self, mask):
        m = np.asarray(mask)
        pick = lambda a: a[m]  # noqa: E731
        return DriverSet([i for i, k in zip(self.ids, m) if k], [i for i, k in zip(self.labels, m) if k],
                         pick(self.re), pick(self.le), pick(self.bl), pick(self.sd), pick(self.mms),
                         pick(self.cms), pick(self.rms), pick(self.xmax), pick(self.sealed))


def solve_many(ds: DriverSet, tr: HornTransfer, *, v=2.83, front_m3=0.0, rear_litres=None):
    """Vectorised ``solve`` over a DriverSet. Returns (D, F) arrays."""
    w = 2 * np.pi * tr.f[None, :]
    cms = ds.cms
    if rear_litres is not None:
        k_box = RHO * C**2 * ds.sd**2 / (rear_litres * 1e-3)
        cms = np.where(ds.sealed, ds.cms, 1 / (1 / ds.cms + k_box))
    shunt = 1 + 1j * w * (front_m3 / (RHO * C**2)) * tr.z_in[None, :]
    z_front = tr.z_in[None, :] / shunt
    zm = ds.rms + 1j * w * ds.mms + 1 / (1j * w * cms) + ds.sd**2 * z_front
    den = (ds.re + 1j * w * ds.le) * zm + ds.bl**2
    vel = ds.bl * v / den
    u_mouth = vel * ds.sd / shunt * tr.u_ratio[None, :]
    return dict(u_mouth=u_mouth, x_peak_m=np.sqrt(2) * np.abs(vel) / w)


def beamwidth(f, polar_fn, angles=np.arange(0, 91, 2.5)):
    """-6 dB full angle (deg) at each f, from a function theta -> complex pressure over f."""
    ref = np.abs(polar_fn(0.0))
    out = np.full(len(np.atleast_1d(f)), 180.0)
    done = np.zeros_like(out, bool)
    for th in angles[1:]:
        lvl = np.abs(polar_fn(th)) / np.maximum(ref, 1e-30)
        hit = (lvl < 0.5) & ~done
        out[hit] = 2 * th
        done |= hit
    return out


def group_delay(f, p):
    ph = np.unwrap(np.angle(p))
    return -np.gradient(ph, 2 * np.pi * np.asarray(f))


# ---------------------------------------------------------------- nodal network
class Network:
    """Plane-wave duct network solved by nodal analysis, with distributed cone sources.

    Nodes carry pressure; ducts (lossless, exact for a uniform segment)
    join two nodes; a shunt volume at a node is an air compliance; a source
    at a node is a share of the cone's projected area, so a rigid cone moving
    at velocity v injects v x area there. One node radiates into a load.
    For unit cone velocity the solve gives every nodal pressure. The
    mechanical radiation impedance on the cone is sum(area_i p_i), and the
    mouth volume velocity is p_mouth / Z_load. This replaces a lumped piston
    and front volume: path-length spread across the cone, the cavity under
    a phase plug and the bends all come out of the same equations.
    """

    def __init__(self):
        self.n = 0
        self.ducts = []
        self.shunts = {}
        self.sources = {}
        self.mouth = None
        self.mouth_radius = None
        self.mouth_load = None

    def node(self):
        self.n += 1
        return self.n - 1

    def duct(self, a, b, area, length, loss=0.0):
        """``loss`` > 0 makes the wavenumber k (1 - j loss): a lumped stand-in for a damping layer."""
        if area <= 0 or length <= 0:
            raise ValueError(f"bad duct {area} {length}")
        self.ducts.append((a, b, float(area), float(length), float(loss)))

    def path(self, a, areas, lengths):
        """Chain of ducts from node a; returns the node list including a."""
        nodes = [a]
        for S, L in zip(areas, lengths):
            nxt = self.node()
            self.duct(nodes[-1], nxt, S, L)
            nodes.append(nxt)
        return nodes

    def volume(self, node, v):
        self.shunts[node] = self.shunts.get(node, 0.0) + float(v)

    def source(self, node, area):
        self.sources[node] = self.sources.get(node, 0.0) + float(area)

    def radiate(self, node, radius, load=None):
        self.mouth, self.mouth_radius, self.mouth_load = node, radius, load

    @property
    def source_area(self):
        return sum(self.sources.values())

    def solve(self, f):
        f = np.asarray(f, float)
        w = 2 * np.pi * f
        k = w / C
        F, N = len(f), self.n
        Y = np.zeros((F, N, N), complex)
        for a, b, S, L, eta in self.ducts:
            zc = RHO * C / S
            kk = k * (1 - 1j * eta)
            s, c = np.sin(kk * L), np.cos(kk * L)
            y0 = 1 / (1j * zc * s)
            Y[:, a, a] += y0 * c
            Y[:, b, b] += y0 * c
            Y[:, a, b] -= y0
            Y[:, b, a] -= y0
        for n, v in self.shunts.items():
            Y[:, n, n] += 1j * w * v / (RHO * C**2)
        zl = self.mouth_load(f) if self.mouth_load is not None else piston_load(f, self.mouth_radius)
        Y[:, self.mouth, self.mouth] += 1 / zl
        q = np.zeros((F, N), complex)
        for n, a in self.sources.items():
            q[:, n] = a
        p = np.linalg.solve(Y, q[..., None])[..., 0]
        z_rad = (p * q).sum(1)
        return dict(f=f, p=p, z_rad=z_rad, u_mouth=p[:, self.mouth] / zl, load=zl)


def couple(driver: Driver, net: dict, *, v=2.83, rear_litres=None, mmd=None):
    """Drive a solved network (unit cone velocity) with a Thiele-Small motor.

    ``mmd`` is the moving mass without air load; the network supplies the
    front air load. Without it, Mms less 1.5 times the one-side baffled air
    mass is used (the datasheet Mms carries air on both sides, and the
    sealed rear chamber adds back about half of one side).
    """
    f = net["f"]
    w = 2 * np.pi * f
    drv = driver.with_rear_volume(rear_litres)
    if mmd is None:
        a = np.sqrt(drv.sd / np.pi)
        mmd = drv.mms - 1.5 * 8 * RHO * a**3 / 3
    zm = drv.rms + 1j * w * mmd + 1 / (1j * w * drv.cms) + net["z_rad"]
    ze = drv.re + 1j * w * drv.le
    den = ze * zm + drv.bl**2
    vel = drv.bl * v / den
    cur = v * zm / den
    return dict(vel=vel, u_mouth=vel * net["u_mouth"], z_e=v / cur, x_peak_m=np.sqrt(2) * np.abs(vel) / w,
                input_power_w=np.real(v * np.conj(cur)),
                radiated_power_w=np.abs(vel * net["u_mouth"]) ** 2 * net["load"].real)


def baffle_step(f, width):
    """Free-space box: first-order shelf from half space (above) to -6 dB (below), f3 near 115/width."""
    x = 1j * np.asarray(f) / (115.0 / width)
    return (x + 0.5) / (x + 1)


def array_load(f, radius, n):
    """Per-box mouth load in a tight cluster of n coherent boxes: n times the load of one piston of n times the area."""
    return n * piston_load(f, radius * np.sqrt(n))
