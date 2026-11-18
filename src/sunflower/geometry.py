"""Sunflower geometry, built from the inside out.

HF unit body cylinder -> mid phase plug -> mid driver -> mid throat -> mid
horn -> mid pod -> kick centre body -> kick horn -> cabinet.

Every horn is an open-area law S plus an inner body, and the inner body is
always subtracted.

* Mid: the phase plug is a smooth bullet. Its nose follows the mid cone at
  a constant gap, it encloses the HF body cylinder, and it tapers to the HF
  horn mouth just inside the mid mouth. The outer wall comes from the law:
  Ro = sqrt(S/pi + Ri^2).
* Kick: the 15 inch is mounted reversed, with its flange on the front face
  of the baffle and its motor pointing into the horn. A one-piece centre
  body sits over it.
  - Its underside is a shroud that follows the cone, outside the basket
    arms. The gap g(r) keeps the channel area at each radius equal to the
    cone area inside that radius divided by the compression ratio.
  - The air leaves at the cone rim, turns under the shroud lip, and runs
    forward between the body and the four cabinet walls.
  - The walls are a square frustum. The four corner fillers make the throat
    a regular octagon and taper to nothing at the mouth, so the mouth is
    fully square.
  - From the throat to the mouth, the body radius is solved from the law:
    Rb = sqrt((A_walls - fillers - S) / pi). The body encloses the kick
    motor, the mid pod and the mid horn, and meets the mid horn's outside at
    the kick mouth. The mid horn then protrudes past the kick mouth.

Coordinates are in metres. z = 0 is the front face of the baffle (where the
kick flange sits), and z is positive towards the listener. Mid horn
coordinate: s = 0 at the mid throat plane.
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from functools import cached_property
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
SQ2 = math.sqrt(2.0)
LAWS = ("exponential", "hypex", "conical", "os")
LAW_NAMES = {"conical": "conical", "exponential": "exponential", "hypex": "hyperbolic-exponential (T = 0.6)",
             "os": "oblate spheroidal (OS)"}
RHO = 1.225


def law_name(law):
    return LAW_NAMES.get(law, law)


def profile_radius(law, rt, rm, length, x):
    """Equivalent radius of an area law from rt at x = 0 to rm at x = length."""
    t = min(max(x / length, 0.0), 1.0)
    if law == "conical":
        return rt + (rm - rt) * t
    if law == "exponential":
        return rt * (rm / rt) ** t
    if law == "os":
        return math.sqrt(rt * rt + t * t * max(rm * rm - rt * rt, 0.0))
    if law == "hypex":
        T, target = 0.6, rm / rt
        lo, hi = 1e-6, 50.0
        for _ in range(60):
            m = 0.5 * (lo + hi)
            lo, hi = (m, hi) if math.cosh(m) + T * math.sinh(m) < target else (lo, m)
        m = 0.5 * (lo + hi)
        return rt * (math.cosh(m * t) + T * math.sinh(m * t))
    raise ValueError(law)


def law_area(law, s0, s1, length, x):
    r = profile_radius(law, math.sqrt(s0 / math.pi), math.sqrt(s1 / math.pi), length, x)
    return math.pi * r * r


# ---------------------------------------------------------------- drivers
@dataclass(frozen=True)
class ConeDriver:
    """Physical geometry of a cone driver.

    z is measured from the flange front face towards the motor. ``cone_z(r)``
    is the cone surface from the rim (r = rim_r, z = rim_z) to the neck
    (r = vc_r, z = neck_z): a curvilinear profile z = rim_z + depth (1 - u)^p.
    No maker publishes cone sections, so the cone, spider and basket are
    estimates from dimensioned drawings. ``estimated`` names the guessed
    fields.
    """
    driver_id: str
    label: str
    record: dict = field(repr=False, compare=False, hash=False)
    sd: float = 0.085
    xmax: float = 0.006
    frame_od: float = 0.387
    flange_t: float = 0.012
    depth: float = 0.168
    vc_r: float = 0.0375
    rim_r: float = 0.157
    rim_z: float = 0.008
    neck_z: float = 0.105
    top_plate_z: float = 0.125
    motor_r: float = 0.100
    motor_h: float = 0.043
    spider_r: float = 0.090
    profile_p: float = 1.1
    estimated: tuple = ()

    @classmethod
    def from_record(cls, rec: dict, kind: str) -> "ConeDriver":
        p, env = rec["parameters"], rec.get("physical_envelope_mm") or {}
        g = rec.get("geometry_mm") or {}
        mm = lambda v: None if v is None else v / 1000  # noqa: E731
        kick = kind == "kick"
        est = []
        sd = p["sd_m2"]
        r_eff = math.sqrt(sd / math.pi)
        surround = mm(g.get("surround")) or (0.022 if kick else 0.011)
        rim_r = mm(g.get("cone_rim")) / 2 if g.get("cone_rim") else r_eff - surround / 2
        if not g.get("cone_rim"):
            est.append("cone rim (from Sd)")
        frame_od = mm(env.get("maximum_width")) or (0.387 if kick else 0.22)
        depth = mm(env.get("depth")) or (0.168 if kick else 0.1)
        flange_t = mm(env.get("flange_thickness")) or (0.012 if kick else 0.010)
        vc_r = (mm(g.get("voice_coil")) or (0.075 if kick else 0.05)) / 2
        motor_h = mm(g.get("motor_height"))
        if motor_h is None:
            motor_h = 0.05 if kick else 0.04
            est.append("motor height")
        motor_r = mm(g.get("motor_diameter"))
        if motor_r is None:
            motor_r = 0.20 if kick else 0.45 * frame_od
            est.append("motor diameter")
        motor_r /= 2
        top = mm(g.get("top_plate_from_front"))
        if top is None:
            top = depth - motor_h
            est.append("top plate position")
        neck_z = top - (0.020 if kick else 0.012)
        est += ["cone profile and neck position", "spider diameter", "basket arm envelope"]
        return cls(rec["driver_id"], f"{rec['manufacturer']} {rec['model_name']}", rec, sd=sd,
                   xmax=p.get("xmax_m") or 0.005, frame_od=frame_od, flange_t=flange_t, depth=depth,
                   vc_r=vc_r, rim_r=rim_r, rim_z=0.008 if kick else 0.005, neck_z=neck_z, top_plate_z=top,
                   motor_r=motor_r, motor_h=motor_h, spider_r=min(2.4 * vc_r, 0.12, 0.70 * rim_r), estimated=tuple(est))

    @property
    def xmech(self):
        """One-way mechanical travel assumed for clearances (no datasheet gives it): 2 x Xmax, at least Xmax + 6 mm."""
        return max(2 * self.xmax, self.xmax + 0.006)

    @property
    def cone_depth(self):
        return self.neck_z - self.rim_z

    def cone_z(self, r):
        r = np.clip(r, self.vc_r, self.rim_r)
        u = (r - self.vc_r) / (self.rim_r - self.vc_r)
        return self.rim_z + self.cone_depth * (1 - u) ** self.profile_p

    def cone_slope(self, r):
        """|dz/dr| of the cone surface."""
        r = np.clip(r, self.vc_r, self.rim_r - 1e-6)
        u = (r - self.vc_r) / (self.rim_r - self.vc_r)
        return self.cone_depth * self.profile_p * (1 - u) ** (self.profile_p - 1) / (self.rim_r - self.vc_r)

    @property
    def cone_volume(self):
        """Air on the concave side of the cone, below the rim plane."""
        r = np.linspace(self.vc_r, self.rim_r, 200)
        return float(np.trapezoid(2 * np.pi * r * (self.cone_z(r) - self.rim_z), r) + math.pi * self.vc_r**2 * self.cone_depth)

    @property
    def mmr_free(self):
        """Air mass on one side of a baffled piston of area Sd."""
        a = math.sqrt(self.sd / math.pi)
        return 8 * RHO * a**3 / 3

    @property
    def price_usd(self):
        return self.record.get("price_usd")


def load_drivers(kind: str):
    recs = json.loads((ROOT / f"data/drivers/{kind}-candidates.json").read_text())["drivers"]
    return {r["driver_id"]: r for r in recs}


# ---------------------------------------------------------------- mid
@dataclass(frozen=True)
class MidHorn:
    drv: ConeDriver
    hf_id: str = ""
    hf_kind: str = "compression"
    hf_body_diameter: float = 0.065
    hf_depth: float = 0.046
    hf_exit_diameter: float = 0.0254
    hf_mouth_ratio: float = 1.8
    law: str = "os"
    compression: float = 2.0
    mouth_diameter: float = 0.40
    length: float = 0.30
    sleeve: float = 0.002
    hf_wall: float = 0.003
    wall: float = 0.006
    throat_offset: float = 0.005      # throat plane in front of the mid flange face
    pod_extra: float = 0.020          # pod length beyond the driver depth
    lip_radius: float = 0.015         # rolled lip at the mid mouth
    felt_loss: float = 0.3            # felt on the bullet underside damps the centre cavity (estimate; measure it)
    gap_override: float = 0.0         # cone to plug gap, m; 0 = 1.15 Xmax + 3 mm
    hf_mount: str = "bullet"          # "throat": HF unit sits in a big plug at the mid throat and fires down the mid horn
    plug_cover: float = 0.0           # with hf_mount "throat": plug radius as a fraction of the cone rim radius
    hf_law: str = "exponential"       # "os": constant-directivity HF waveguide of hf_half_angle_deg
    hf_half_angle_deg: float = 30.0

    # ------------------------------------------------ cone and front cavity
    @property
    def gap(self):
        """Cone to plug clearance: 1.15 Xmax + 3 mm (a DSP high-pass and limiter keep the mid inside it)."""
        return self.gap_override or 1.15 * self.drv.xmax + 0.003

    def s_cone(self, r):
        """Mid cone surface in horn coordinates (negative: behind the throat plane)."""
        d = self.drv
        r = np.asarray(r, float)
        cap = d.neck_z - 0.3 * d.vc_r * np.sqrt(np.clip(1 - (r / d.vc_r) ** 2, 0, 1))
        return -(self.throat_offset + np.where(r < d.vc_r, cap, d.cone_z(r)))

    # ------------------------------------------------ plug
    @property
    def plug_radius(self):
        return self.hf_body_diameter / 2 + self.sleeve

    @property
    def is_bullet(self):
        """A bullet tweeter carries its own phase plug, so it sits at the mid mouth on a slim nose."""
        return self.hf_kind == "bullet"

    @property
    def throat_mounted(self):
        return self.hf_mount == "throat"

    @property
    def s_face(self):
        """Throat mount: the HF face, far enough forward that the HF body clears the cone behind it."""
        r = self.hf_body_diameter / 2 + self.sleeve
        return max(0.010, float(self.s_cone(r)) + self.gap + self.hf_depth + 0.003)

    @property
    def s_exit(self):
        if self.throat_mounted:
            return self.s_face
        return self.s_tip if self.is_bullet else self.hf_depth + 0.005

    @property
    def s_tip(self):
        return self.s_face + 0.012 if self.throat_mounted else self.length - 0.010

    @property
    def hf_len(self):
        return self.s_tip - self.s_exit

    @property
    def hf_mouth_radius(self):
        return self.hf_horn_radius(self.s_tip)

    @property
    def tip_radius(self):
        return self.plug_radius if self.is_bullet else self.hf_mouth_radius + self.hf_wall

    @property
    def nose_length(self):
        return 0.3 * math.sqrt(self.drv.sd / math.pi)

    @property
    def _ogive_end(self):
        """Radius the holding taper heads for at the HF mouth: the old exponential tip, or the bare HF exit."""
        if self.hf_law == "os":
            return self.hf_exit_diameter / 2 + self.hf_wall
        return self.tip_radius

    NOSE = 0.016                                  # slim nose of a bullet-tweeter holder: a cable conduit

    @property
    def plug_r0(self):
        """Bullet radius at the throat plane, so the taper still clears the HF body at its front."""
        if self.throat_mounted:
            return max(self.plug_cover * self.drv.rim_r, self.plug_radius)
        if self.is_bullet:
            return self.NOSE
        u = self.s_exit / self.s_tip
        c = math.cos(math.pi * u / 2)
        e = self._ogive_end
        return e + (self.plug_radius + 0.001 - e) / c

    def plug(self, s):
        """Bullet outer radius at s >= 0. A smooth taper holds the HF body; with an OS HF waveguide the bullet
        then follows the waveguide's outside, joined to the taper by a smooth maximum (no corner)."""
        if s < 0 or s > self.s_tip:
            return 0.0
        if self.throat_mounted:                # big plug: full radius, rounds in to the HF body by its face,
            r0, rb = self.plug_r0, self.plug_radius        # then a short taper in front of the face
            rf = min(self.hf_mouth_radius + self.hf_wall, rb)
            s1 = max(self.s_face - 0.03, 0.0)
            if s <= s1:
                return r0
            if s <= self.s_face:
                t = (s - s1) / max(self.s_face - s1, 1e-6)
                return rb + (r0 - rb) * 0.5 * (1 + math.cos(math.pi * t))
            t = (s - self.s_face) / max(self.s_tip - self.s_face, 1e-6)
            return rf + (rb - rf) * 0.5 * (1 + math.cos(math.pi * min(t, 1.0)))
        if self.is_bullet:                     # slim nose rising smoothly to hold the tweeter body at the mouth
            s1 = self.s_tip - self.hf_depth    # back of the tweeter body
            s0 = max(s1 - 0.06, 0.0)
            if s >= s1:
                return self.plug_radius
            t = min(max((s - s0) / (s1 - s0), 0.0), 1.0)
            return self.NOSE + (self.plug_radius - self.NOSE) * 0.5 * (1 - math.cos(math.pi * t))
        e = self._ogive_end
        a = e + (self.plug_r0 - e) * math.cos(math.pi * s / (2 * self.s_tip))
        if self.hf_law != "os" or s < self.s_exit:
            return a
        b = self.hf_horn_radius(s) + self.hf_wall
        k = 0.004                                   # blend width, m
        return k * math.log(math.exp(a / k) + math.exp(b / k))

    def nose(self, r):
        """Underside of the bullet nose (r < plug_r0): the cone surface plus the gap."""
        return self.s_cone(r) + self.gap

    def hf_horn_radius(self, s):
        if self.throat_mounted:                # the HF face: a short barrel, the mid horn is its waveguide
            return self.hf_exit_diameter / 2
        x = min(max(s - self.s_exit, 0.0), self.hf_len)
        r0 = self.hf_exit_diameter / 2
        if self.hf_law == "os":
            return math.sqrt(r0 * r0 + (x * math.tan(math.radians(self.hf_half_angle_deg))) ** 2)
        return r0 * self.hf_mouth_ratio ** (x / self.hf_len)

    # ------------------------------------------------ open area
    @property
    def throat_area(self):
        return self.drv.sd / self.compression

    @property
    def outer_throat_radius(self):
        return math.sqrt(self.throat_area / math.pi + self.plug_r0**2)

    def eq_radius(self, s):
        return profile_radius(self.law, math.sqrt(self.throat_area / math.pi), self.mouth_diameter / 2, self.length, s)

    @cached_property
    def _wall(self):
        zz = np.linspace(0.0, self.length, 801)
        ro = np.array([math.sqrt(self.eq_radius(z) ** 2 + self.plug(z) ** 2) for z in zz])
        return zz, np.maximum.accumulate(ro)

    def radius(self, s):
        zz, ro = self._wall
        return float(np.interp(s, zz, ro))

    def area(self, s):
        return math.pi * (self.radius(s) ** 2 - self.plug(s) ** 2)

    def outer(self, s):
        """Outside of the mid horn wall (what the kick air sees)."""
        return self.radius(s) + self.wall

    @property
    def half_angle_deg(self):
        """Wall angle at the mouth: the mid horn's nominal coverage is about twice this."""
        ds = 0.01
        return math.degrees(math.atan((self.radius(self.length) - self.radius(self.length - ds)) / ds))

    @property
    def wavefront_radius(self):
        rm = self.mouth_diameter / 2
        return rm / math.sin(max(math.radians(self.half_angle_deg), math.radians(2)))

    # ------------------------------------------------ pod
    @property
    def pod_radius(self):
        return self.drv.frame_od / 2 + self.wall

    @property
    def pod_length(self):
        return self.drv.depth + self.pod_extra

    @property
    def rear_volume_litres(self):
        rc = self.pod_radius - self.wall
        d = self.drv
        displaced = math.pi * d.motor_r**2 * d.motor_h + 0.25 * math.pi * rc**2 * (d.depth - d.motor_h)
        return (math.pi * rc**2 * self.pod_length - displaced) * 1000

    @property
    def front_cavity_litres(self):
        r = np.linspace(0, self.drv.rim_r, 300)
        return float(np.trapezoid(2 * np.pi * r * self.gap, r)) * 1000

    def check(self):
        p = []
        d = self.drv
        if 2 * d.rim_r < 1.6 * 2 * self.plug_radius:
            p.append("mid cone smaller than 1.6 x the plug")
        if self.outer_throat_radius > d.rim_r - 0.004:
            p.append("throat annulus wider than the cone")
        if self.throat_mounted:
            if self.plug_r0 < self.plug_radius - 1e-9:
                p.append("plug too small to hold the HF unit")
        elif not self.is_bullet:
            if self.hf_len < 0.05:
                p.append("HF horn shorter than 50 mm")
            ss = np.linspace(self.s_exit, self.s_tip, 50)
            if min(self.plug(s) - self.hf_horn_radius(s) for s in ss) < self.hf_wall - 1e-4:
                p.append("bullet wall thinner than the HF horn wall")
        if self.half_angle_deg > 40:
            p.append("mid flare above 40 degrees")
        if not 1.3 <= self.compression <= 4:
            p.append("mid compression outside 1.3 to 4")
        return p


# ---------------------------------------------------------------- kick
@dataclass(frozen=True)
class Kick:
    drv: ConeDriver
    mid: MidHorn
    compression: float = 2.0
    law: str = "exponential"
    half_angle_deg: float = 12.0      # each wall; the splay between stacked boxes is twice this
    throat_width: float = 0.48        # inside, across flats, at the baffle front face
    protrusion: float = 0.15          # mid mouth in front of the kick mouth
    chamber_depth: float = 0.10       # sealed chamber, baffle back face to back panel
    blockage: float = 0.12            # basket arms in the cone channel
    ply: float = 0.018
    baffle: float = 0.025
    motor_clear: float = 0.010
    body_wall: float = 0.004
    arm_offset: float = 0.0           # basket arm surface above the cone; 0 = xmech + 10 mm
    arm_thickness: float = 0.006
    wedge_chamber: bool = False       # hollow top and bottom wedges vented into the sealed chamber
    shell: float = 0.006              # centre body printed as a ribbed shell this thick; its inside is the mid's rear chamber

    # ------------------------------------------------ positions along z
    @property
    def tan(self):
        return math.tan(math.radians(self.half_angle_deg))

    @property
    def splay_deg(self):
        return 2 * self.half_angle_deg

    @property
    def motor_front(self):
        """Front of the motor enclosure inside the centre body."""
        d = self.drv
        return d.top_plate_z + d.motor_h + self.motor_clear + self.body_wall

    @property
    def mid_flange_z(self):
        return self.motor_front + self.mid.pod_length

    @property
    def mid_throat_z(self):
        return self.mid_flange_z + self.mid.throat_offset

    @property
    def mid_mouth_z(self):
        return self.mid_throat_z + self.mid.length

    @property
    def mouth_z(self):
        return self.mid_mouth_z - self.protrusion

    @property
    def depth(self):
        """Cabinet depth, back panel to kick mouth (the mid horn protrudes past it)."""
        return self.ply + self.chamber_depth + self.baffle + self.mouth_z

    @property
    def overall_depth(self):
        return self.depth + (self.protrusion + self.mid.lip_radius if self.protrusion > 0 else 0.0)

    # ------------------------------------------------ walls
    def half_width(self, z):
        """Inside half-width of the frustum (all four walls), valid behind the baffle too."""
        return self.throat_width / 2 + z * self.tan

    @property
    def mouth_width(self):
        return 2 * self.half_width(self.mouth_z)

    def outer_half_width(self, z):
        return self.half_width(z) + self.ply / math.cos(math.radians(self.half_angle_deg))

    @property
    def half_height(self):
        """Inside half-height of the box: flat top and bottom, level with the square mouth."""
        return self.mouth_width / 2

    def wedge(self, z):
        """Thickness of the tapered wedge under the flat top (and over the flat bottom).

        Its sloping face is at half_width(z), so inside the horn is the same square frustum as the
        side walls, and the wedge vanishes at the square mouth."""
        return max(self.half_height - self.half_width(z), 0.0) if z >= 0 else 0.0

    @property
    def external_mouth(self):
        return 2 * self.outer_half_width(self.mouth_z)

    @property
    def external_height(self):
        return 2 * (self.half_height + self.ply)

    @property
    def external_back(self):
        return 2 * self.outer_half_width(-self.baffle - self.chamber_depth - self.ply)

    @property
    def wedge_litres(self):
        zz = np.linspace(0, self.mouth_z, 200)
        inner = np.array([2 * self.half_width(z) * 2 * self.wedge(z) for z in zz])
        return float(np.trapezoid(inner, zz) * 1000)

    def filler(self, z):
        """Corner filler leg: a regular octagon at the throat, nothing at the mouth."""
        c0 = self.throat_width / (2 + SQ2)
        return c0 * min(max(1 - z / self.mouth_z, 0.0), 1.0)

    def wall_area(self, z):
        w, c = 2 * self.half_width(z), self.filler(z)
        return w * w - 2 * c * c

    # ------------------------------------------------ cone channel (shroud over the reversed cone)
    @property
    def arm(self):
        return self.arm_offset or self.drv.xmech + 0.010

    def gap(self, r):
        """Shroud to cone gap, normal to the cone."""
        cr = self.compression * (1 - self.blockage)
        return np.maximum.reduce([np.asarray(r) / (2 * cr) + 0 * np.asarray(r),
                                  np.full_like(np.asarray(r, float), self.arm + self.arm_thickness + 0.003),
                                  np.full_like(np.asarray(r, float), self.drv.xmech + 0.003)])

    def shroud_z(self, r):
        psi = np.arctan(self.drv.cone_slope(r))
        return self.drv.cone_z(r) + self.gap(r) / np.cos(psi)

    @property
    def motor_body_radius(self):
        return self.drv.motor_r + self.motor_clear + self.body_wall

    @property
    def channel_start(self):
        """Inner end of the controlled channel: outside the spider platform and the motor enclosure."""
        return max(self.drv.spider_r + 0.008, self.motor_body_radius)

    @property
    def rim_open_area(self):
        """Cone area that the compression ratio refers to (the cone inside its rim)."""
        return math.pi * self.drv.rim_r**2

    @property
    def throat_area(self):
        return self.rim_open_area / self.compression

    @property
    def turn_z(self):
        """Underside of the shroud lip: high enough over the flange for the throat area to pass."""
        d = self.drv
        need = d.flange_t + 1.1 * self.throat_area / (2 * math.pi * d.frame_od / 2)
        return max(float(self.shroud_z(d.rim_r)), need)

    @property
    def trapped_litres(self):
        """Air between the cone and the spider and platform, inside the channel start (a lumped cavity)."""
        d = self.drv
        r = np.linspace(d.vc_r, self.channel_start, 100)
        roof = np.minimum(d.top_plate_z - 0.004, self.shroud_z(self.channel_start))
        return float(np.trapezoid(2 * np.pi * r * np.maximum(roof - d.cone_z(r), 0.003), r)) * 1000

    # ------------------------------------------------ forward horn and centre body
    @property
    def horn_length(self):
        return self.mouth_z - self.turn_z

    @property
    def mouth_body_radius(self):
        return self.mid.outer(self.mouth_z - self.mid_throat_z)

    @property
    def mouth_area(self):
        return self.wall_area(self.mouth_z) - math.pi * self.mouth_body_radius**2

    def law_area(self, z):
        return law_area(self.law, self.throat_area, self.mouth_area, self.horn_length, z - self.turn_z)

    def body_radius(self, z):
        """Centre body radius from the area law (forward horn, turn_z <= z <= mouth_z)."""
        return math.sqrt(max(self.wall_area(z) - self.law_area(z), 0.0) / math.pi)

    def envelope(self, z):
        """Smallest body radius that still holds what is inside it at z."""
        d, m = self.drv, self.mid
        if z <= self.turn_z + 0.03:
            return d.frame_od / 2 + 0.010
        if z <= self.motor_front:
            return self.motor_body_radius
        if z <= self.mid_throat_z:
            return m.pod_radius
        return m.outer(min(z - self.mid_throat_z, m.length))

    def area(self, z):
        return self.wall_area(z) - math.pi * self.body_radius(z) ** 2

    # ------------------------------------------------ hollow centre body
    @property
    def mid_rear_litres(self):
        """Sealed air inside the shell body, from the kick motor enclosure's front wall to the mouth, outside the
        mid horn wall, less the mid driver. This is the mid's rear chamber."""
        m, d = self.mid, self.mid.drv
        zz = np.linspace(self.motor_front, self.mouth_z, 300)
        inner = np.array([m.outer(z - self.mid_throat_z) if z >= self.mid_throat_z else 0.0 for z in zz])
        outer = np.array([max(self.body_radius(z) if z >= self.turn_z else 0.0, self.envelope(z)) for z in zz]) - self.shell
        v = float(np.trapezoid(np.pi * np.clip(outer**2 - inner**2, 0, None), zz))
        displaced = math.pi * d.motor_r**2 * d.motor_h + 0.25 * math.pi * (d.frame_od / 2) ** 2 * (d.depth - d.motor_h)
        return max(v - displaced, 0.0) * 1000

    # ------------------------------------------------ chamber
    @property
    def chamber_litres(self):
        """Sealed chamber: the box behind the baffle (full flat height), the baffle cutout, the cone's
        concave side, and the hollow wedges when they are vented into it (less their 18 mm panels)."""
        zz = np.linspace(-self.baffle - self.chamber_depth, -self.baffle, 100)
        box = np.trapezoid(2 * self.half_width(zz) * 2 * self.half_height, zz)
        cut = math.pi * (self.drv.frame_od / 2 - 0.017) ** 2 * self.baffle
        v = (box + cut + self.drv.cone_volume) * 1000
        if self.wedge_chamber:
            v += 0.8 * self.wedge_litres
        return float(v)

    # ------------------------------------------------ checks
    def check(self):
        p = []
        d = self.drv
        if self.throat_width < d.frame_od + 0.04:
            p.append("throat too narrow for the kick flange")
        if self.channel_start > d.rim_r - 0.02:
            p.append("motor enclosure covers the cone: no channel left")
            return p
        zz = np.linspace(self.turn_z, self.mouth_z, 300)
        rb = np.array([self.body_radius(z) for z in zz])
        env = np.array([self.envelope(z) for z in zz])
        inside = zz < self.mouth_z - 0.01
        if (rb - env)[inside].min() < 0.003:
            i = int(np.argmin(np.where(inside, rb - env, 1.0)))
            p.append(f"centre body from the law does not hold its contents at z = {zz[i]*1000:.0f} mm "
                     f"({rb[i]*1000:.0f} < {env[i]*1000:.0f} mm)")
        flats = np.array([self.half_width(z) for z in zz]) - rb
        if flats.min() < 0.008:
            p.append(f"passage at the flats only {flats.min()*1000:.0f} mm")
        if self.horn_length < 0.2:
            p.append("kick horn shorter than 200 mm")
        if self.mouth_area <= self.throat_area:
            p.append("mouth smaller than throat")
        if self.protrusion > 1e-9:
            p.append("mid horn protrudes past the kick mouth")
        if self.chamber_depth < 0.03:
            p.append("no room behind the baffle")
        return p

    def summary(self):
        return dict(throat_area_m2=self.throat_area, mouth_area_m2=self.mouth_area,
                    compression=self.compression, cone_to_throat=self.drv.sd / self.throat_area,
                    horn_length_m=self.horn_length, mouth_width_m=self.mouth_width,
                    external_mouth_m=self.external_mouth, external_back_m=self.external_back,
                    external_height_m=self.external_height, wedge_litres=self.wedge_litres,
                    wedge_chamber=self.wedge_chamber,
                    depth_m=self.depth, overall_depth_m=self.overall_depth, splay_deg=self.splay_deg,
                    chamber_litres=self.chamber_litres, trapped_litres=self.trapped_litres,
                    mid_rear_litres=self.mid_rear_litres,
                    turn_z_m=self.turn_z, max_body_radius_m=max(self.body_radius(z) for z in
                                                                 np.linspace(self.turn_z, self.mouth_z, 200)))


# ---------------------------------------------------------------- array
def array_layout(kick: Kick, n=3):
    """n x n array: n columns fanned by the splay about a common vertical axis, rows stacked flat.

    Identical boxes tile exactly: the angled sides meet flush between columns, the flat tops and
    bottoms meet flush between rows. Matrices map each box's frame (metres; origin at the baffle
    front face centre, +x right, +y up, +z forward) to the array frame (the centre box's frame)."""
    s = math.radians(kick.splay_deg)
    apex = (kick.external_mouth / 2) / math.tan(math.radians(kick.half_angle_deg)) - kick.mouth_z
    h = kick.external_height
    k = (n - 1) // 2
    out = []
    for j in range(-k, k + 1):
        for i in range(-k, k + 1):
            c, sn = math.cos(i * s), math.sin(i * s)
            R = np.array([[c, 0, sn], [0, 1, 0], [-sn, 0, c]])
            M = np.eye(4)
            M[:3, :3] = R
            M[:3, 3] = R @ np.array([0, 0, apex]) - np.array([0, 0, apex]) + np.array([0, j * h, 0])
            out.append(dict(i=i, j=j, yaw_deg=math.degrees(i * s), pitch_deg=0.0, matrix=M.round(6).tolist()))
    return dict(apex_behind_baffle_m=apex, row_pitch_m=h, splay_deg=kick.splay_deg, boxes=out)
