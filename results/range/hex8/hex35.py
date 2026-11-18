"""Hex 10 hybrid: 350 mm cube limit. Short front horn on the cone; the rear chamber radiates through a slot at the horn mouth
(a slot port). Half space (on a wall), 2.83 V."""
import math, numpy as np
from sunflower.geometry import load_drivers
from sunflower.acoustics import Driver, RHO, C, spl
recs = {k: r for k, r in load_drivers("mid").items() if r["nominal_diameter"] in ("10in", "8in")}
f = np.geomspace(20, 500, 400); w = 2 * np.pi * f; s = 1j * w
def vented(d, vb, fb, v=2.83, ql=7.0, qp=30.0):
    cab = vb / (RHO * C**2); map_ = 1 / ((2 * np.pi * fb) ** 2 * cab)
    zp = s * map_ + 2 * np.pi * fb * map_ / qp
    zb = 1 / (s * cab + 1 / zp + 2 * np.pi * fb * cab / ql)
    zm = d.rms + s * d.mms + 1 / (s * d.cms) + d.sd**2 * zb
    vel = d.bl * v / ((d.re + s * d.le) * zm + d.bl**2)
    ud = vel * d.sd
    return spl(1j * w * RHO * ud * (1 - zb / zp) / (2 * np.pi)), np.abs(vel) / w * math.sqrt(2), np.abs(ud * zb / zp), map_
# 350 mm cube: hex 350 across flats, 350 deep, 15 mm ply; less driver, horn flare, plug and port
F, Dp, t = 0.35, 0.35, 0.015
gross = 0.866 * (F - 2 * t) ** 2 * (Dp - 2 * t)
VB = gross - 0.006
print(f"hex {F*1000:.0f} mm across flats x {Dp*1000:.0f} deep: gross {gross*1000:.1f} L, net about {VB*1000:.1f} L")
rows = []
for k, r in recs.items():
    d = Driver.from_record(r); p = r["parameters"]
    if (r.get("physical_envelope_mm") or {}).get("maximum_width", 999) > 270: continue
    best = None
    for fb in np.arange(40, 80, 1.0):
        l, x, up, mp = vented(d, VB, fb)
        ref = l[(f > 150) & (f < 300)].mean()
        f3 = f[np.argmax(l > ref - 3)]; f6 = f[np.argmax(l > ref - 6)]; f10 = f[np.argmax(l > ref - 10)]
        rip = np.ptp(l[(f > f3) & (f < 200)])
        sc = f6 + 4 * max(0, rip - 3)
        if best is None or sc < best[0]: best = (sc, fb, f3, f6, f10, rip, ref, x, up, mp)
    _, fb, f3, f6, f10, rip, ref, x, up, mp = best
    vmax = math.sqrt((p["power_w"] or 200) * 8)
    vx = 2.83 * p["xmax_m"] / x[f > 0.9 * fb].max()
    vuse = min(vx, vmax)
    sp = 0.0045                                  # slot 300 x 15 mm
    lp = mp * sp / RHO - 0.85 * math.sqrt(sp)    # physical length less end corrections
    vport = up.max() * vuse / 2.83 / sp
    rows.append((f6, f"{k:18s} {r['nominal_diameter']:5s} tune {fb:.0f} Hz: -3 dB {f3:.0f}, -6 dB {f6:.0f}, -10 dB {f10:.0f} Hz | {ref:.1f} dB/2.83 V | "
                 f"max {ref + 20*math.log10(vuse/2.83):.0f} dB | slot 300x15 mm, {lp*1000:.0f} mm long, air {vport:.0f} m/s at max | cone usable to {r.get('usable_f_high_hz')}"))
for _, t_ in sorted(rows): print(t_)
