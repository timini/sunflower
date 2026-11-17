"""Nested geometry of the Sunflower horn. All lengths in metres.

Coordinates: z is the horn axis, positive towards the listener. For the low
horn z = 0 is the 15 inch cone rim (the back face of the baffle); for the
mid horn z = 0 is the mid throat. The mid horn mouth is flush with the low
horn mouth.

Nesting:
  * the tweeter body is the phase plug of the mid horn;
  * the mid horn can (mid horn, mid driver, rear cap and tail) and the
    reversed 15 inch motor together form the phase plug of the low horn.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict, field
import math

import numpy as np

SQ2 = math.sqrt(2.0)


PROFILES = ("conical", "exponential", "hypex", "os")

# Frame diameter and depth by nominal size, used when a catalogue record has
# no published envelope. Typical pro midrange values; check the chosen driver.
SIZE_ENVELOPE = {"6in": (0.160, 0.070), "6.5in": (0.174, 0.075), "8in": (0.220, 0.095), "10in": (0.262, 0.115)}
# Highest crossover trusted for a cone of each size (cone breakup is not modelled).
SIZE_MAX_XO = {"6in": 7000.0, "6.5in": 7000.0, "8in": 5000.0, "10in": 3600.0}


def profile_radius(profile, rt, rm, length, z):
    """Outer radius of a flare from rt at z = 0 to rm at z = length."""
    t = z / length
    if profile == "conical":
        return rt + (rm - rt) * t
    if profile == "exponential":
        return rt * (rm / rt) ** t
    if profile == "os":
        tan = math.sqrt(rm * rm - rt * rt) / length
        return math.sqrt(rt * rt + (z * tan) ** 2)
    if profile == "hypex":                      # T = 0.6, radius = rt (cosh mz + T sinh mz)
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


@dataclass(frozen=True)
class MidHorn:
    """Cone midrange into an annular horn around the tweeter body.

    z = 0 is the mid throat. The tweeter body, in a printed sleeve, runs from
    the throat to its face at ``tweeter_face``, so the mid throat is an
    annulus. The tweeter face is never behind its own depth.
    """
    driver_id: str = "beyma-6mcf200nd"
    nominal: str = "6.5in"
    sd: float = 0.014
    frame_diameter: float = 0.174
    driver_depth: float = 0.075
    profile: str = "conical"
    equivalent_throat_diameter: float = 0.076
    mouth_diameter: float = 0.190
    length: float = 0.125
    tweeter_id: str = "fostex-t90a"
    tweeter_face: float = 0.100           # from the mid throat
    tweeter_body_diameter: float = 0.060
    sleeve: float = 0.002
    front_chamber_cc: float = 25.0
    wall: float = 0.006
    adapter_length: float = 0.012
    rear_cap: float = 0.018
    tail_length: float = 0.120

    @property
    def plug_radius(self) -> float:
        return self.tweeter_body_diameter / 2 + self.sleeve

    @property
    def outer_throat_radius(self) -> float:
        return math.hypot(self.equivalent_throat_diameter / 2, self.plug_radius)

    @property
    def throat_area(self) -> float:
        return math.pi * (self.equivalent_throat_diameter / 2) ** 2

    @property
    def compression_ratio(self) -> float:
        return self.sd / self.throat_area

    def radius(self, z: float) -> float:
        return profile_radius(self.profile, self.outer_throat_radius, self.mouth_diameter / 2, self.length,
                              min(max(z, 0.0), self.length))

    def area(self, z: float) -> float:
        inner = self.plug_radius if z < self.tweeter_face else 0.0
        return math.pi * (self.radius(z) ** 2 - inner**2)

    @property
    def wavefront_radius(self) -> float:
        """Radius of the mouth wavefront, from the wall angle at the mouth."""
        rm = self.mouth_diameter / 2
        dz = self.length * 1e-3
        slope = (self.radius(self.length) - self.radius(self.length - dz)) / dz
        return rm / math.sin(max(math.atan(slope), math.radians(2)))

    @property
    def can_diameter(self) -> float:
        return max(self.mouth_diameter, self.frame_diameter) + 2 * self.wall

    @property
    def can_length(self) -> float:
        return self.length + self.adapter_length + self.driver_depth + self.rear_cap

    @property
    def rear_volume_litres(self) -> float:
        """Sealed volume behind the cone: half the sleeve plus the hollow tail."""
        rc = self.can_diameter / 2 - self.wall
        sleeve = math.pi * rc**2 * (self.driver_depth + self.rear_cap) * 0.5
        tail = 2 / 3 * math.pi * rc**2 * self.tail_length
        return (sleeve + tail) * 1000

    def check(self):
        problems = []
        if self.mouth_diameter / 2 < self.outer_throat_radius + 0.01:
            problems.append("mouth not larger than the throat")
        if not 1.4 <= self.compression_ratio <= 7.0:
            problems.append(f"compression ratio {self.compression_ratio:.1f} outside 1.4 to 7")
        if self.tweeter_face > self.length:
            problems.append("tweeter face outside the mid mouth")
        if self.radius(self.tweeter_face) - self.plug_radius < 0.006:
            problems.append("annulus too narrow at the tweeter face")
        half = math.degrees(math.atan((self.mouth_diameter / 2 - self.outer_throat_radius) / self.length))
        if half > 35:
            problems.append(f"mean flare half-angle {half:.0f} deg above 35 deg")
        return problems


@dataclass(frozen=True)
class LowDriverBody:
    """Reversed 15 inch: magnet forward, inside the low horn throat."""
    cone_radius: float = 0.165          # sqrt(Sd/pi) for a 15 inch
    cone_depth: float = 0.110           # rim plane to voice-coil plane
    motor_radius: float = 0.090         # magnet / motor outer radius
    motor_length: float = 0.070         # motor beyond the voice-coil plane
    nose_length: float = 0.130          # printed fairing on the back plate


@dataclass(frozen=True)
class LowHorn:
    """Straight-sided pyramid: octagonal throat to square mouth, flat panels.

    z = 0 is the cone rim, which sits at the back face of the baffle because
    the reversed driver is rear-mounted through the baffle cutout. The horn
    walls start at the baffle front face (z = baffle) and run ``length`` to
    the mouth. Width W and corner chamfer leg c vary linearly, so every wall
    (four side panels, four corner fillers) is a flat plate.
    """
    throat_width: float = 0.40          # across flats at the baffle front face
    mouth_width: float = 0.62
    length: float = 0.80                # baffle front face to mouth
    baffle: float = 0.025
    cutout_radius: float = 0.1765       # 15NMB420 baffle cutout 353 mm
    body: LowDriverBody = field(default_factory=LowDriverBody)
    mid: MidHorn = field(default_factory=MidHorn)

    @property
    def mouth_z(self):
        return self.baffle + self.length

    def _t(self, z):
        return min(max((z - self.baffle) / self.length, 0.0), 1.0)

    def width(self, z):
        return self.throat_width + (self.mouth_width - self.throat_width) * self._t(z)

    def chamfer(self, z):
        c0 = self.throat_width / (2 + SQ2)  # regular octagon at the baffle
        return c0 * (1 - self._t(z))

    def outer_area(self, z):
        if z < self.baffle:
            return math.pi * self.cutout_radius**2
        w, c = self.width(z), self.chamfer(z)
        return w * w - 2 * c * c

    @property
    def wavefront_radius(self):
        """Mouth wavefront radius of the equivalent cone (equal-area radii)."""
        at = math.sqrt(self.outer_area(self.baffle) / math.pi)
        am = math.sqrt(self.outer_area(self.mouth_z) / math.pi)
        return am / math.sin(math.atan((am - at) / self.length))

    @property
    def front_chamber_m3(self):
        """Air between the convex back of the cone and the throat plane."""
        b = self.body
        zz = np.linspace(0, b.cone_depth, 400)
        rc = b.cone_radius * (1 - zz / b.cone_depth)
        a = np.array([self.outer_area(z) for z in zz]) - np.pi * rc**2
        return float(np.trapezoid(a, zz))

    # centre body --------------------------------------------------------
    @property
    def throat_z(self):
        return self.body.cone_depth

    @property
    def can_start(self):
        return self.mouth_z - self.mid.can_length

    @property
    def tail_start(self):
        return self.can_start - self.mid.tail_length

    @property
    def nose_end(self):
        b = self.body
        return b.cone_depth + b.motor_length + b.nose_length

    def inner_radius(self, z):
        b, rc = self.body, self.mid.can_diameter / 2
        r = 0.0
        zm = b.cone_depth + b.motor_length
        if b.cone_depth <= z <= zm:
            r = b.motor_radius
        elif zm < z <= zm + b.nose_length:
            t = (z - zm) / b.nose_length          # elliptical nose on the motor
            r = b.motor_radius * math.sqrt(max(0.0, 1 - t * t))
        if z >= self.can_start:
            r = max(r, rc)
        elif z >= self.tail_start:
            t = (self.can_start - z) / self.mid.tail_length
            r = max(r, rc * math.sqrt(max(0.0, 1 - t * t)))
        return r

    def area(self, z):
        return self.outer_area(z) - math.pi * self.inner_radius(z) ** 2

    def acoustic_area(self, s):
        """Area at distance s from the throat plane (z = cone_depth)."""
        return self.area(s + self.throat_z)

    @property
    def acoustic_length(self):
        return self.mouth_z - self.throat_z

    @property
    def breaks(self):
        b = self.body
        zs = [b.cone_depth + b.motor_length, self.nose_end, self.tail_start, self.can_start]
        return tuple(z - self.throat_z for z in zs)

    def check(self):
        problems = []
        if self.throat_width < 2 * self.cutout_radius:
            problems.append("octagon throat smaller than the baffle cutout")
        if self.tail_start < self.nose_end:
            problems.append("mid can tail overlaps the 15 inch nose fairing")
        zz = np.linspace(self.throat_z, self.mouth_z, 400)
        if min(self.area(z) for z in zz) <= 0:
            problems.append("blocked section")
        clear = min(self.width(z) / 2 - self.inner_radius(z) for z in zz)
        if clear < 0.05:
            problems.append(f"radial clearance {clear*1000:.0f} mm < 50 mm")
        return problems

    def summary(self):
        zz = np.linspace(self.throat_z, self.mouth_z, 801)
        a = np.array([self.area(z) for z in zz])
        return dict(throat_area_m2=float(a[0]), mouth_area_m2=float(a[-1]),
                    min_area_m2=float(a.min()), min_area_z_m=float(zz[a.argmin()]),
                    expansion=float(a[-1] / a[0]), front_chamber_litres=self.front_chamber_m3 * 1000)


def as_dict(obj):
    return asdict(obj)
