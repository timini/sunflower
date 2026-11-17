"""Acoustic networks built from the actual Sunflower geometry.

Kick: the cone is split into rings. The inner rings, behind the spider and
motor, feed a lumped trapped cavity. The rest feed a radial channel whose
flow-normal area is 2 pi r g, with g the real shroud gap, less the basket
arm blockage. At the rim the flow turns under the shroud lip and runs
forward through the open area between the frustum walls (less the corner
fillers) and the centre body, slice by slice, to the square mouth.

Mid: the cone under the bullet nose feeds an inner radial channel, the
cone under the lead-in ring feeds an outer one, and both meet at the
throat annulus. The cone between them injects there directly. The horn
from the throat to the mouth uses the open area between the wall and the
bullet, which is always subtracted.
"""
from __future__ import annotations

import math

import numpy as np

from sunflower.acoustics import Network, array_load
from sunflower.geometry import Kick, MidHorn


def _rings(r0, r1, n):
    e = np.linspace(r0, r1, n + 1)
    return e, 0.5 * (e[1:] + e[:-1])


def kick_network(k: Kick, *, n_channel=16, dz=0.012, cluster=1):
    d = k.drv
    net = Network()
    cav = net.node()
    net.volume(cav, k.trapped_litres / 1000)
    rs = k.channel_start
    net.source(cav, math.pi * rs**2)
    edges, mids = _rings(rs, d.rim_r, n_channel)
    nodes = [cav]
    for i, rm in enumerate(mids):
        slant = (edges[i + 1] - edges[i]) * math.sqrt(1 + float(d.cone_slope(rm)) ** 2)
        area = 2 * math.pi * rm * float(k.gap(rm)) * (1 - k.blockage)
        nxt = net.node()
        net.duct(nodes[-1], nxt, area, slant)
        net.source(nxt, math.pi * (edges[i + 1] ** 2 - edges[i] ** 2))
        nodes.append(nxt)
    net.source(nodes[-1], max(d.sd - math.pi * d.rim_r**2, 0.0))  # surround
    # turn: radially over the surround and flange under the lip, then forward
    r_lip = k.body_radius(k.turn_z)
    rr = np.linspace(d.rim_r, r_lip, 5)
    last = nodes[-1]
    for a, b in zip(rr[:-1], rr[1:]):
        rm = 0.5 * (a + b)
        floor = d.rim_z if rm < d.frame_od / 2 else d.flange_t
        nxt = net.node()
        net.duct(last, nxt, 2 * math.pi * rm * (k.turn_z - floor), b - a)
        last = nxt
    # dead volume beside the lip, between the baffle and the lip underside, outside the flange
    dead = (k.wall_area(0.0) - math.pi * r_lip**2) * k.turn_z
    net.volume(last, max(dead, 0.0))
    zz = np.arange(k.turn_z, k.mouth_z + 1e-9, dz)
    zz = np.append(zz[:-1], k.mouth_z) if zz[-1] < k.mouth_z - 1e-6 else zz
    turn_len = 0.5 * math.pi * 0.5 * (k.turn_z - d.flange_t)
    areas = [k.area(0.5 * (a + b)) for a, b in zip(zz[:-1], zz[1:])]
    lens = list(np.diff(zz))
    lens[0] += turn_len
    path = net.path(last, areas, lens)
    r_eq = math.sqrt(k.wall_area(k.mouth_z) / math.pi)
    load = (lambda f: array_load(f, r_eq, cluster)) if cluster > 1 else None
    net.radiate(path[-1], r_eq, load)
    net.mouth_inner = k.mouth_body_radius
    return net


def mid_network(m: MidHorn, *, n_branch=10, ds=0.005, felt=None):
    """``felt``: loss factor of the damping layer on the bullet underside (default m.felt_loss)."""
    felt = m.felt_loss if felt is None else felt
    d = m.drv
    net = Network()
    rp, ro = m.plug_r0, m.outer_throat_radius
    g = m.gap
    # inner branch under the bullet nose: centre (closed) to the plug edge
    e, mids = _rings(0.0, rp, n_branch)
    inner = net.node()
    net.source(inner, math.pi * e[1] ** 2)
    last = inner
    for i in range(1, n_branch):
        nxt = net.node()
        net.duct(last, nxt, 2 * math.pi * e[i] * g, e[1] - e[0], loss=felt)
        net.source(nxt, math.pi * (e[i + 1] ** 2 - e[i] ** 2))
        last = nxt
    throat = net.node()
    net.duct(last, throat, 2 * math.pi * rp * g, 0.5 * (e[1] - e[0]), loss=felt)
    # outer branch under the lead-in ring: rim (closed, surround) to the throat annulus
    e, mids = _rings(ro, d.rim_r, n_branch)
    outer = net.node()
    net.source(outer, math.pi * (e[-1] ** 2 - e[-2] ** 2) + max(d.sd - math.pi * d.rim_r**2, 0.0))  # surround
    last = outer
    for i in range(n_branch - 1, 0, -1):
        nxt = net.node()
        slant = (e[i + 1] - e[i]) * math.sqrt(1 + float(d.cone_slope(e[i])) ** 2)
        net.duct(last, nxt, 2 * math.pi * e[i] * g, slant)
        if i > 0:
            net.source(nxt, math.pi * (e[i] ** 2 - e[i - 1] ** 2))
        last = nxt
    net.duct(last, throat, 2 * math.pi * ro * g, 0.5 * (e[1] - e[0]))
    # cone under the annulus injects at the throat; a short duct up to the throat plane
    net.source(throat, math.pi * (ro**2 - rp**2) - 0.0)
    depth = float(np.mean(-m.s_cone(np.linspace(rp, ro, 10))))
    start = net.node()
    net.duct(throat, start, m.throat_area, max(depth, 0.003))
    edges = np.arange(0.0, m.length + 1e-9, ds)
    if edges[-1] < m.length - 1e-6:
        edges = np.append(edges, m.length)
    edges = edges[np.abs(edges - m.s_tip) > 0.001]
    edges = np.array(sorted([*edges, m.s_tip]))
    areas = [m.area(0.5 * (a + b)) for a, b in zip(edges[:-1], edges[1:])]
    path = net.path(start, areas, np.diff(edges))
    net.radiate(path[-1], m.mouth_diameter / 2)
    return net
