"""Nested geometry of Sunflower v0.2, built from the inside out.

HF unit body cylinder -> mid phase plug -> mid driver -> mid throat -> mid
horn -> mid can -> low centre body -> low horn.

Every horn is an open-area law S(z) plus an inner body. The inner body is
always subtracted: open area = outer area - pi Ri(z)^2.
  * Mid: the phase plug (the HF unit in a sleeve, tapering to the HF horn
    mouth) is fixed by the HF unit; the outer wall follows from the law,
    Ro = sqrt(S/pi + Ri^2).
  * Low: the walls are flat panels; the centre body follows from the law,
    Ri = sqrt((A_walls - S)/pi). The centre body is one piece from the 15
    inch cone (its rear face follows the cone as a phase plug) round the
    motor to the mid can.

Coordinates, metres. Low horn: z = 0 at the 15 inch cone rim (back face of
the baffle), positive towards the listener. Mid horn: s = 0 at the mid
throat. Both mouths are in one plane.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

SQ2 = math.sqrt(2.0)
LAWS = ("conical", "exponential", "hypex", "os")
LAW_NAMES = {"conical": "conical", "exponential": "exponential", "hypex": "hyperbolic-exponential",
             "os": "oblate spheroidal (OS)"}


def law_name(law):
    return LAW_NAMES.get(law, law)
PROFILES = LAWS

# Frame diameter and depth by nominal size, used when a record has no envelope.
SIZE_ENVELOPE = {"6in": (0.160, 0.070), "6.5in": (0.174, 0.075), "8in": (0.220, 0.095), "10in": (0.262, 0.115)}
# Highest crossover trusted for a cone of each size (cone breakup is not modelled).
SIZE_MAX_XO = {"6in": 7000.0, "6.5in": 7000.0, "8in": 5500.0, "10in": 3600.0}


def profile_radius(profile, rt, rm, length, z):
    """Radius of a flare from rt at z = 0 to rm at z = length."""
    t = min(max(z / length, 0.0), 1.0)
    z = t * length
    if profile == "conical":
        return rt + (rm - rt) * t
    if profile == "exponential":
        return rt * (rm / rt) ** t
    if profile == "os":
        tan = math.sqrt(max(rm * rm - rt * rt, 0.0)) / length
        return math.sqrt(rt * rt + (z * tan) ** 2)
    if profile == "hypex":                      # T = 0.6
        T = 0.6
        target = rm / rt
        lo, hi = 1e-6, 50.0
        for _ in range(60):
            m = 0.5 * (lo + hi)
            if math.cosh(m) + T * math.sinh(m) < target:
                lo = m
            else:
                hi = m
        m = 0.5 * (lo + hi)
        return rt * (math.cosh(m * t) + T * math.sinh(m * t))
    raise ValueError(profile)


def _cos_taper(r0, r1, t):
    t = min(max(t, 0.0), 1.0)
    return r1 + (r0 - r1) * 0.5 * (1 + math.cos(math.pi * t))


# ---------------------------------------------------------------- mid
@dataclass(frozen=True)
class MidHorn:
    # mid driver
    driver_id: str = ""
    nominal: str = "8in"
    sd: float = 0.022
    xmax: float = 0.003
    frame_diameter: float = 0.220
    driver_depth: float = 0.095
    factory_sealed: bool = False
    # HF unit in the middle
    hf_id: str = ""
    hf_kind: str = "compression"
    hf_body_diameter: float = 0.065
    hf_depth: float = 0.046          # with terminals
    hf_exit_diameter: float = 0.0254
    hf_horn_length: float = 0.0      # printed HF horn, exit to tip; 0 = run it to 10 mm short of the mid mouth
    hf_mouth_ratio: float = 1.6      # HF horn mouth / exit diameter
    # horn design variables
    law: str = "exponential"
    compression: float = 2.2
    mouth_diameter: float = 0.28
    length: float = 0.20
    # construction
    sleeve: float = 0.002
    hf_wall: float = 0.003
    wall: float = 0.006
    rear_cap: float = 0.018
    extra_rear_litres: float = 2.0   # hollow fairing behind the can, shared with the centre body

    # ------------------------------------------------ plug
    @property
    def plug_radius(self):
        return self.hf_body_diameter / 2 + self.sleeve

    @property
    def s_exit(self):
        return self.hf_depth + 0.005

    @property
    def hf_len(self):
        """Printed HF horn length. By default the HF mouth sits 10 mm inside the mid mouth: FEM shows a
        deep mouth inside the mid horn notches the HF on axis (the mid annulus behind it is a side branch)."""
        return self.hf_horn_length if self.hf_horn_length > 0 else self.length - 0.010 - self.s_exit

    @property
    def s_tip(self):
        return self.s_exit + self.hf_len

    @property
    def hf_mouth_radius(self):
        return self.hf_exit_diameter / 2 * self.hf_mouth_ratio

    @property
    def tip_radius(self):
        return self.hf_mouth_radius + self.hf_wall

    @property
    def nose_length(self):
        return 0.3 * math.sqrt(self.sd / math.pi)

    def plug(self, s):
        """Outer radius of the phase plug at s (0 beyond the HF mouth)."""
        if s < 0:
            t = -s / self.nose_length
            return self.plug_radius * math.sqrt(max(0.0, 1 - t * t))
        if s <= self.s_exit:
            return self.plug_radius
        if s <= self.s_tip:
            return _cos_taper(self.plug_radius, self.tip_radius, (s - self.s_exit) / self.hf_len)
        return 0.0

    def hf_horn_radius(self, s):
        """Inner radius of the printed HF horn, exit to mouth (exponential)."""
        t = (s - self.s_exit) / self.hf_len
        return self.hf_exit_diameter / 2 * self.hf_mouth_ratio ** min(max(t, 0.0), 1.0)

    # ------------------------------------------------ open area
    @property
    def throat_area(self):
        return self.sd / self.compression

    def eq_radius(self, s):
        rt = math.sqrt(self.throat_area / math.pi)
        return profile_radius(self.law, rt, self.mouth_diameter / 2, self.length, s)

    def _wall(self):
        """Outer wall on a grid: law area plus the plug, never narrowing."""
        key = "_wall_cache"
        if key not in self.__dict__:
            zz = np.linspace(0.0, self.length, 801)
            ro = np.array([math.sqrt(self.eq_radius(z) ** 2 + self.plug(z) ** 2) for z in zz])
            object.__setattr__(self, key, (zz, np.maximum.accumulate(ro)))
        return self.__dict__[key]

    def radius(self, s):
        """Outer wall radius. The plug is always inside it and always subtracted."""
        zz, ro = self._wall()
        return float(np.interp(s, zz, ro))

    def area(self, s):
        """Open area: outer wall minus the phase plug."""
        return math.pi * (self.radius(s) ** 2 - self.plug(s) ** 2)

    @property
    def outer_throat_radius(self):
        return self.radius(0.0)

    @property
    def compression_ratio(self):
        return self.compression

    @property
    def front_chamber_m3(self):
        """Air between cone and plug or lead-in: Sd times Xmax plus 1.5 mm."""
        return self.sd * (self.xmax + 0.0015)

    @property
    def front_chamber_cc(self):
        return self.front_chamber_m3 * 1e6

    @property
    def wavefront_radius(self):
        rm = self.mouth_diameter / 2
        dz = self.length * 1e-3
        slope = (self.radius(self.length) - self.radius(self.length - dz)) / dz
        return rm / math.sin(max(math.atan(slope), math.radians(2)))

    # ------------------------------------------------ can
    @property
    def can_diameter(self):
        return max(self.radius(self.length) * 2, self.frame_diameter) + 2 * self.wall

    @property
    def can_length(self):
        return self.length + self.driver_depth + self.rear_cap

    @property
    def rear_volume_litres(self):
        rc = self.can_diameter / 2 - self.wall
        return math.pi * rc**2 * (self.driver_depth + self.rear_cap) * 500 + self.extra_rear_litres

    @property
    def cone_diameter(self):
        return 2 * math.sqrt(self.sd / math.pi)

    def check(self):
        p = []
        if self.cone_diameter < 1.6 * 2 * self.plug_radius:
            p.append(f"cone {self.cone_diameter*1000:.0f} mm smaller than 1.6 x plug {2*self.plug_radius*1000:.0f} mm")
        if self.s_tip > self.length - 0.005:
            p.append("HF horn mouth beyond the mid mouth")
        if self.hf_len < 0.03:
            p.append("printed HF horn shorter than 30 mm")
        if self.radius(0) - self.plug_radius < 0.006:
            p.append("throat annulus narrower than 6 mm")
        zz = np.linspace(0, self.length, 200)
        ro = np.array([self.radius(z) for z in zz])
        half = math.degrees(math.atan((ro[-1] - ro[0]) / self.length))
        if half > 35:
            p.append(f"mean flare {half:.0f} deg above 35 deg")
        step = math.pi * self.tip_radius**2 / (math.pi * self.radius(self.s_tip) ** 2)
        if step > 0.10:
            p.append(f"HF mouth step {100*step:.0f}% of the mid area")
        if not 1.4 <= self.compression <= 6.0:
            p.append("compression outside 1.4 to 6")
        return p


# ---------------------------------------------------------------- low
@dataclass(frozen=True)
class LowDriverBody:
    cone_radius: float = 0.165
    cone_depth: float = 0.100
    motor_radius: float = 0.090
    motor_length: float = 0.061
    xmax: float = 0.005


@dataclass(frozen=True)
class LowHorn:
    """Flat-panel kick horn round a one-piece centre body.

    Width and height go linearly from an octagonal throat (across flats
    ``throat_width``) at the baffle front to a rectangle at the mouth. The
    centre body radius is solved from the area law up to the mid can, then
    is the can itself to the mouth.
    """
    throat_width: float = 0.36
    mouth_width: float = 0.60
    mouth_height: float = 0.55
    length: float = 0.45
    law: str = "exponential"
    gap: float = 0.003               # cone to plug clearance on top of Xmax
    baffle: float = 0.025
    cutout_radius: float = 0.1765
    body: LowDriverBody = field(default_factory=LowDriverBody)
    mid: MidHorn = field(default_factory=MidHorn)

    # ------------------------------------------------ walls
    @property
    def mouth_z(self):
        return self.baffle + self.length

    @property
    def throat_z(self):
        return self.baffle

    def _t(self, z):
        return min(max((z - self.baffle) / self.length, 0.0), 1.0)

    def width(self, z):
        return self.throat_width + (self.mouth_width - self.throat_width) * self._t(z)

    def height(self, z):
        return self.throat_width + (self.mouth_height - self.throat_width) * self._t(z)

    def chamfer(self, z):
        return self.throat_width / (2 + SQ2) * (1 - self._t(z))

    def outer_area(self, z):
        if z < self.baffle:
            return math.pi * self.cutout_radius**2
        c = self.chamfer(z)
        return self.width(z) * self.height(z) - 2 * c * c

    # ------------------------------------------------ centre body
    @property
    def can_radius(self):
        return self.mid.can_diameter / 2

    @property
    def can_start(self):
        return self.mouth_z - self.mid.can_length

    @property
    def motor_end(self):
        return self.body.cone_depth + self.body.motor_length

    def cone_radius_at(self, z):
        b = self.body
        return b.cone_radius * max(0.0, 1 - z / b.cone_depth)

    @property
    def throat_radius(self):
        """Centre body radius at the throat: it must cover the cone in front of it."""
        b = self.body
        r = self.cone_radius_at(self.throat_z - b.xmax - self.gap) + 0.010
        return max(r, b.motor_radius + 0.006)

    @property
    def throat_area(self):
        return self.outer_area(self.throat_z) - math.pi * self.throat_radius**2

    @property
    def compression(self):
        return math.pi * self.body.cone_radius**2 / self.throat_area

    @property
    def end_area(self):
        return self.outer_area(self.can_start) - math.pi * self.can_radius**2

    def law_area(self, z):
        rt = math.sqrt(self.throat_area / math.pi)
        re = math.sqrt(max(self.end_area, 1e-6) / math.pi)
        r = profile_radius(self.law, rt, re, self.can_start - self.throat_z, z - self.throat_z)
        return math.pi * r * r

    def inner_radius(self, z):
        if z >= self.can_start:
            return self.can_radius
        if z < self.throat_z:
            return self.throat_radius
        return math.sqrt(max(self.outer_area(z) - self.law_area(z), 0.0) / math.pi)

    def area(self, z):
        return self.outer_area(z) - math.pi * self.inner_radius(z) ** 2

    def acoustic_area(self, s):
        return self.area(s + self.throat_z)

    @property
    def acoustic_length(self):
        return self.mouth_z - self.throat_z

    @property
    def breaks(self):
        return (self.can_start - self.throat_z,)

    @property
    def front_chamber_m3(self):
        return math.pi * self.body.cone_radius**2 * (self.body.xmax + self.gap)

    @property
    def nose_end(self):
        return self.can_start

    @property
    def tail_start(self):
        return self.can_start

    @property
    def wavefront_radius(self):
        at = math.sqrt(self.outer_area(self.baffle) / math.pi)
        am = math.sqrt(self.outer_area(self.mouth_z) / math.pi)
        return am / math.sin(max(math.atan((am - at) / self.length), math.radians(2)))

    # ------------------------------------------------ cabinet
    # Shell: vertical sides; top and bottom are flat panels from a rear pod
    # (tall enough for the 15 inch flange) to the mouth. The internal horn
    # walls are separate panels; everything between them and the shell is
    # the sealed chamber, joined to the box behind the baffle.
    back_box: float = 0.045
    ply: float = 0.018

    @property
    def back_half_height(self):
        return self.cutout_radius + 0.045

    @property
    def depth(self):
        return self.ply + self.back_box + self.baffle + self.length

    @property
    def splay_deg(self):
        """Angle between neighbouring boxes in a vertical stack: top plus bottom of the shell."""
        rise = self.mouth_height / 2 - self.back_half_height
        return 2 * math.degrees(math.atan(rise / (self.depth - self.ply)))

    def shell_half_height(self, z):
        """Inside face of the shell top, from the back box to the mouth."""
        z0 = -self.back_box
        t = min(max((z - z0) / (self.mouth_z - z0), 0.0), 1.0)
        return self.back_half_height + (self.mouth_height / 2 - self.back_half_height) * t

    def chamber_litres(self):
        """Sealed volume available: shell interior minus baffle, horn and its walls, plus the cone's concave side."""
        t = self.ply
        zz = np.linspace(-self.back_box, self.mouth_z, 600)
        inside = np.array([self.mouth_width * 2 * self.shell_half_height(z) for z in zz])
        horn = np.array([0.0 if z < self.baffle else (self.width(z) + 2 * t) * (self.height(z) + 2 * t) for z in zz])
        horn = np.minimum(horn, inside)
        baffle = np.array([self.mouth_width * 2 * self.shell_half_height(z) if 0 <= z < self.baffle else 0.0 for z in zz])
        vol = np.trapezoid(inside - horn - baffle, zz)
        cone = math.pi * self.body.cone_radius**2 * self.body.cone_depth / 3
        return float((vol + cone) * 1000)

    def void_litres(self, ply=0.018):
        return self.chamber_litres()

    def check(self):
        p = []
        b = self.body
        if self.throat_width < 2 * self.cutout_radius:
            p.append("octagon throat smaller than the baffle cutout")
        if self.can_start < self.motor_end + 0.02:
            p.append("mid can overlaps the 15 inch motor")
        if self.end_area <= 0 or self.throat_area <= 0:
            p.append("blocked")
            return p
        zz = np.linspace(self.throat_z, self.mouth_z, 300)
        ri = np.array([self.inner_radius(z) for z in zz])
        rw = np.array([math.sqrt(self.outer_area(z) / math.pi) for z in zz])
        m = (zz >= b.cone_depth) & (zz <= self.motor_end)
        if m.any() and ri[m].min() < b.motor_radius + 0.006:
            p.append("centre body does not clear the 15 inch motor")
        if (rw - ri).min() < 0.03:
            p.append("low passage narrower than 30 mm")
        if self.mouth_height / 2 < self.back_half_height:
            p.append("mouth lower than the rear pod")
        if not 1.3 <= self.compression <= 4.0:
            p.append(f"low throat compression {self.compression:.2f} outside 1.3 to 4")
        a = np.array([self.area(z) for z in zz])
        if np.max(np.abs(np.diff(a)) / a[:-1]) > 0.05:
            p.append("step in low open area")
        return p

    def summary(self):
        zz = np.linspace(self.throat_z, self.mouth_z, 801)
        a = np.array([self.area(z) for z in zz])
        return dict(throat_area_m2=float(a[0]), mouth_area_m2=float(a[-1]), min_area_m2=float(a.min()),
                    compression=self.compression, expansion=float(a[-1] / a[0]),
                    front_chamber_litres=self.front_chamber_m3 * 1000, splay_deg=self.splay_deg,
                    chamber_available_litres=self.chamber_litres(), depth_m=self.depth)
