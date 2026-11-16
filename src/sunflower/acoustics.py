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
        x, w = np.polynomial.legendre.leggauss(96)
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


def group_delay(f, p):
    ph = np.unwrap(np.angle(p))
    return -np.gradient(ph, 2 * np.pi * np.asarray(f))
