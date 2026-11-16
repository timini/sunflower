"""Nested geometry of the Sunflower horn. All lengths in metres.

Coordinates: z is the horn axis, positive towards the listener. z = 0 is the
front face of the 15 inch mounting baffle. The low horn mouth is at
z = low.length. The mid horn and the tweeter sit on the axis, and the mid
horn mouth is flush with the low horn mouth.

Nesting:
  * the tweeter body is the phase plug of the mid horn;
  * the mid horn can (mid horn + 6.5 inch driver + rear cap) and the reversed
    15 inch motor together form the phase plug of the low horn.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict, field
import math

import numpy as np

SQ2 = math.sqrt(2.0)


@dataclass(frozen=True)
class Tweeter:
    """Fostex T90A, horn super tweeter. Body sits in a printed bullet housing."""
    model: str = "Fostex T90A"
    housing_diameter: float = 0.064      # bullet housing OD (T90A horn is 60 mm)
    face_from_mid_throat: float = 0.100  # tweeter mouth plane, measured from mid throat
    body_depth: float = 0.0878           # T90A envelope depth (horn-simulation study)
    nose_length: float = 0.040           # ogive tail of the bullet, behind the mid throat
    sensitivity_db: float = 106.0        # 1 W / 1 m, 8 ohm (Fostex)
    horn_cutoff_hz: float = 3600.0       # Fostex published cut-off
    upper_hz: float = 35000.0
    impedance_ohm: float = 8.0


@dataclass(frozen=True)
class MidHorn:
    """Beyma 6MCF200Nd into an annular conical horn around the tweeter."""
    driver_id: str = "beyma-6mcf200nd"
    equivalent_throat_diameter: float = 0.076
    mouth_diameter: float = 0.190
    length: float = 0.125
    front_chamber_cc: float = 20.0
    wall: float = 0.005                 # printed horn wall
    driver_frame_diameter: float = 0.174
    driver_depth: float = 0.075
    adapter_length: float = 0.012       # cone-to-annulus adapter / front chamber ring
    rear_cap: float = 0.018
    can_diameter: float = 0.200         # outer diameter of the mid can

    @property
    def outer_throat_radius(self) -> float:
        return math.hypot(self.equivalent_throat_diameter, Tweeter().housing_diameter) / 2

    @property
    def throat_area(self) -> float:
        return math.pi * (self.equivalent_throat_diameter / 2) ** 2

    def radius(self, z: float) -> float:
        rt, rm = self.outer_throat_radius, self.mouth_diameter / 2
        return rt + (rm - rt) * z / self.length

    def area(self, z: float) -> float:
        tw = Tweeter()
        inner = tw.housing_diameter / 2 if z < tw.face_from_mid_throat else 0.0
        return math.pi * (self.radius(z) ** 2 - inner**2)

    @property
    def wavefront_radius(self) -> float:
        """Spherical wavefront radius at the mouth (distance to the cone apex)."""
        rt, rm = self.outer_throat_radius, self.mouth_diameter / 2
        return rm / math.sin(math.atan((rm - rt) / self.length))

    @property
    def can_length(self) -> float:
        return self.length + self.adapter_length + self.driver_depth + self.rear_cap


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
    tail_length: float = 0.120          # ogive tail on the mid can
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
        return self.can_start - self.tail_length

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
            t = (self.can_start - z) / self.tail_length
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
