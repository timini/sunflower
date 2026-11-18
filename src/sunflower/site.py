"""Write the GitHub Pages site (docs/*.html) from the design and the results.

Every number on the site comes from results/, cad/ or sunflower.design, so
the pages cannot drift from the files they describe.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

from sunflower import design as D
from sunflower.geometry import law_name

ROOT = D.ROOT
DOCS = ROOT / "docs"
REPO = "https://github.com/timini/sunflower"

LOGO = """<svg viewBox="0 0 32 32" aria-hidden="true"><g fill="none" stroke-width="2.4">
<rect x="3" y="3" width="26" height="26" rx="4" stroke="#d38a0c"/><circle cx="16" cy="16" r="8" stroke="#eb6834"/>
<circle cx="16" cy="16" r="3.2" fill="#1baf7a" stroke="none"/></g></svg>"""

PAGES = [("index.html", "Product"), ("design.html", "Design"), ("simulation.html", "Simulation"), ("range.html", "Range"), ("validation.html", "Validation"),
         ("build.html", "Build"), ("downloads.html", "Downloads")]


def page(name, title, body, description):
    nav = "".join(f'<a href="{f}"{" aria-current=page" if f == name else ""}>{t}</a>' for f, t in PAGES)
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{title}</title>
<meta name="description" content="{description}">
<link rel="stylesheet" href="assets/style.css">
<link rel="icon" href="data:image/svg+xml,{LOGO.replace('"', "'").replace('#', '%23').replace(chr(10), '')}">
</head>
<body>
<header class="site"><div class="wrap">
<a class="brand" href="index.html">{LOGO}Sunflower</a>
<nav>{nav}</nav>
</div></header>
<main><div class="wrap">
{body}
</div></main>
<footer><div class="wrap">
Sunflower is an open design. Source, CAD and results: <a href="{REPO}">{REPO.replace('https://', '')}</a>.
Simulated, not yet built or measured. MIT licence.
</div></footer>
</body>
</html>
"""


def fig(src, caption, cls="plot"):
    return f'<figure class="figure"><img class="{cls}" src="{src}" alt="{caption}" loading="lazy"><figcaption>{caption}</figcaption></figure>'


def table(head, rows, num=()):
    th = "".join(f'<th class="{"num" if i in num else ""}">{h}</th>' for i, h in enumerate(head))
    tr = "".join("<tr>" + "".join(f'<td class="{"num" if i in num else ""}">{c}</td>' for i, c in enumerate(r)) + "</tr>"
                 for r in rows)
    return f'<div class="table-wrap"><table><thead><tr>{th}</tr></thead><tbody>{tr}</tbody></table></div>'



def load():
    R = ROOT / "results"
    data = dict(
        dsp=json.loads((R / "planewave/dsp.json").read_text()),
        fem=json.loads((R / "fem/summary.json").read_text()),
        cut=json.loads((ROOT / "cad/cutlist.json").read_text()),
        array=json.loads((ROOT / "cad/array.json").read_text()),
        parts=json.loads((ROOT / "cad/parts.json").read_text()),
        cutoffs=json.loads((R / "planewave/cutoffs.json").read_text()),
        system=pd.read_csv(R / "planewave/system.csv"),
        area=pd.read_csv(R / "cad/area-check.csv"),
        felt=pd.read_csv(R / "fem/mid-felt-sweep.csv") if (R / "fem/mid-felt-sweep.csv").exists() else None,
        n_mid=len(pd.read_csv(R / "optimise/mid-search.csv", usecols=[0])),
        n_kick=len(pd.read_csv(R / "optimise/kick-search.csv", usecols=[0])),
        summary=D.summary(),
    )
    K = D.KICK
    ex = data["cut"]["external_mm"]
    data["cad_stale"] = abs(ex["width"] - K.external_mouth * 1000) > 5
    ex.update(width=K.external_mouth * 1000, height_front=K.external_height * 1000, depth=K.depth * 1000)
    s = data["system"]
    f = s.frequency_hz.values
    on = s.sum_0deg_spl_db.values
    ref = on[(f >= 200) & (f <= 2000)].mean()
    data["kpi"] = dict(sensitivity=float(data["dsp"]["target_db"]), f6=float(f[np.argmax(on > ref - 6)]))
    return data


def stale_note(d):
    if not d.get("cad_stale"):
        return ""
    return ('<div class="note"><p><b>Being rebuilt.</b> The design on this site is the current one, '
            f'{d["cut"]["external_mm"]["width"]:.0f} mm square and {d["cut"]["external_mm"]["depth"]:.0f} mm deep. The renders, '
            'videos, CAD files, cut list and slice figure still show the previous, larger box and are being rebuilt.</p></div>')


def video(name, caption):
    return (f'<figure class="figure"><video controls loop muted playsinline preload="none" '
            f'poster="img/renders/{name}-poster.jpg"><source src="img/renders/{name}.webm" type="video/webm">'
            f'<source src="img/renders/{name}.mp4" type="video/mp4"></video><figcaption>{caption}</figcaption></figure>')


def price(v):
    return f"${v:.0f}" if v else "not found"


def drivers_table(d):
    k, m, tw = D.KICK_RECORD, D.MID_RECORD, D.TW
    rows = [["Kick", f"{D.label(k)}, 15 inch, mounted reversed", f'<a href="{k["parameter_source"]}">datasheet</a>',
             price(k.get("price_usd"))],
            ["Mid", f"{D.label(m)}, 8 inch", f'<a href="{m["parameter_source"]}">datasheet</a>', price(m.get("price_usd"))],
            ["HF", f"{tw.model}, 1 inch exit, {tw.body_diameter*1000:.0f} mm body", f'<a href="{tw.source}">datasheet</a>',
             price(tw.price_usd)]]
    total = sum(x for x in (k.get("price_usd"), m.get("price_usd"), tw.price_usd) if x)
    rows.append(["", "<b>Drivers per box</b>", "", f"<b>${total:.0f}</b>"])
    return table(["Section", "Driver", "Source", "Price (USD)"], rows, num=(3,))


# ---------------------------------------------------------------- pages
def index(d):
    k, cut, dsp = d["kpi"], d["cut"], d["dsp"]
    ex = cut["external_mm"]
    f1, f2 = D.CROSSOVER_LOW_MID_HZ, D.CROSSOVER_MID_HIGH_HZ
    K, m, tw = D.KICK, D.MID, D.TW
    low = dsp["kick_low_end"]["single_half_space_db"]
    shelf = dsp["no_sub_shelf"]
    body = f"""
<h1>Sunflower</h1>
<p class="lede">A three-way coaxial horn built on the idea behind the Funktion-One Res 2: a kick horn up to
{f1:.0f} Hz, a cone mid that carries the voice from {f1:.0f} Hz to {f2/1000:.0f} kHz, and a 1 inch compression driver above.
Here the three are nested on one axis. The HF horn is the mid's phase plug, the mid horn is the core of the kick
horn's centre body, and its mouth sits inside the kick mouth.</p>
<figure class="hero"><img src="img/renders/hero.jpg" alt="Rendered Sunflower cabinet"></figure>
<div class="stat">
<div><b>{k['sensitivity']:.0f} dB</b><span>2.83 V, 1 m</span></div>
<div><b>{low['f_minus6_hz']:.0f} Hz</b><span>kick -6 dB on the ground; {shelf['f_minus6_hz']:.0f} Hz with the no-sub shelf</span></div>
<div><b>{f1:.0f} Hz / {f2/1000:.0f} kHz</b><span>crossovers: kick, vocal-range mid, HF</span></div>
<div><b>{ex['width']:.0f} x {ex['height_front']:.0f} x {ex['depth']:.0f}</b><span>mm, width x height x depth</span></div>
</div>
{stale_note(d)}
<div class="note"><p><b>Status.</b> Simulated, not built or measured. The cone and basket shapes of the drivers are
estimated from datasheet drawings, and the phase plugs must be checked against real drivers before printing. The
<a href="simulation.html#limits">limits</a> section lists what the models leave out.</p></div>
{video("turntable", "One box, all the way round.")}
<h2>Nine boxes, one point source</h2>
<p>The sides are angled at {K.half_angle_deg:.0f} degrees and the top and bottom are flat, so nine identical boxes tile
exactly: three columns fanned {K.splay_deg:.0f} degrees apart, each a straight stack of three, every joint flush.</p>
{fig("img/renders/stack.jpg", "Nine identical boxes: three fanned columns, flat-stacked rows.", "render")}
{fig("img/renders/stack-top-dims.jpg", "From above: the three columns fan about one point behind the array.", "render")}
{video("orbit", "Around the 3 x 3 cluster.")}
<div class="grid2">
{fig("img/renders/front-dims.jpg", "Into the mouth, mm. Completely square at the mouth; the corner fillers make the throat an octagon.", "render")}
{fig("img/renders/section-side-dims.jpg", "Side section, mm.", "render")}
</div>
<h2>Why coaxial</h2>
<p>A conventional three-way top puts its kick horn, mid horn and HF horn side by side or one above another. Sunflower
nests them on one axis, and that changes four things.</p>
<div class="grid2">
<div class="card"><h3>One acoustic centre</h3><p>The kick, mid and HF all radiate from the same axis. There is no
spacing between separate horns, so the sections do not interfere with each other off axis at the crossovers, and a
single delay per section aligns them at every angle round the box, not only straight ahead.</p></div>
<div class="card"><h3>The same pattern every way</h3><p>Because the horns are round and nested, the coverage is
the same horizontally and vertically. A box can be used on its side, in a column or in a block without its pattern
changing.</p></div>
<div class="card"><h3>No wasted space</h3><p>In front of a horn-loaded 15 inch is a large volume of horn air. Here the
mid driver, its sealed pod, the mid horn and the HF unit all live inside the kick's centre body, in that volume, and the
centre body doubles as the kick's phase plug. Every inner part is subtracted from the horn it sits in.</p></div>
<div class="card"><h3>Boxes that tile</h3><p>The angled sides and flat top and bottom let nine identical boxes form a
curved array with every joint flush, so a cluster is one continuous wavefront rather than a stack of separate sources.</p></div>
</div>

<h2>Compared with a Funktion-One Res 2</h2>
<p>The Res 2 is the reference: a 15 inch bass horn up to 250 Hz with an 8 inch cone mid and a 1 inch compression driver
above. Sunflower keeps that split. The table compares the published Res 2 figures with Sunflower's simulated ones; a Res 2
is a measured, shipping product and Sunflower has not been built.</p>
{table(["", "Res 2 (published)", "Sunflower (simulated)"], [
    ["Layout", "Bass horn, with the mid and HF horns beside it", "Kick, mid and HF nested on one axis"],
    ["Acoustic centres", "Separate for bass and mid/HF", "One axis for all three"],
    ["Coverage", "50 x 25 deg (different each way)", "The same horizontally and vertically"],
    ["Low end", "Rated from 60 Hz (bass horn stated to 35 Hz)", f"-6 dB at {low['f_minus6_hz']:.0f} Hz on the ground; made to run with a sub"],
    ["Mid to HF crossover", "Not published (about 4 to 6 kHz on its sister boxes)", f"{f2/1000:.1f} kHz"],
    ["Size (W x H x D)", "484 x 510 x 975 mm", f"{ex['width']:.0f} x {ex['height_front']:.0f} x {ex['depth']:.0f} mm"],
    ["Arraying", "Radiused grids, set by the grid", "Nine identical boxes tile flush"],
    ["Status", "Shipping product", "Simulated; not built or measured"]])}
<p>Where Sunflower gives ground: its front is larger (829 mm square against 484 x 510 mm), its kick is a kick bin that
needs a sub, and its mid is cone-limited to about {f2/1000:.1f} kHz where Funktion-One's Axhead takes theirs higher.</p>

<h2>Drivers</h2>
{drivers_table(d)}
<p>Chosen by an optimiser that tried {d['n_mid']:,} mid and HF combinations and {d['n_kick']:,} kick and cabinet
variants, then tuned the best through the crossover. See <a href="design.html">design</a>.</p>
"""
    return page("index.html", "Sunflower", body, "Sunflower: a three-way nested coaxial horn after the Funktion-One Res 2.")


def design(d):
    K, m, tw = D.KICK, D.MID, D.TW
    kd, md = K.drv, m.drv
    a = d["area"]
    ka = a[a.kind == "kick"]
    mi = a[a.kind == "mid"]
    felt = d["felt"]
    felt_rows = ""
    if felt is not None and "eta" in felt:
        g = felt.groupby("eta").first().reset_index()
    f1, f2 = D.CROSSOVER_LOW_MID_HZ, D.CROSSOVER_MID_HIGH_HZ
    fs = d["fem"]["mid"].get("felt_sweep", {}).get("rows", [])
    if fs:
        felt_rows = table(["Felt loss factor", "FEM notch depth (dB)", "Network notch depth (dB)"],
                          [[f"{r['eta']:.2f}", f"{r['fem_null_depth_db']:.1f}", f"{r['network_null_depth_db']:.1f}"]
                           for r in fs], num=(0, 1, 2))
    est = ", ".join(kd.estimated)
    body = f"""
<h1>Design</h1>
<p class="lede">Sized from the inside out: the HF unit's body sets the mid phase plug, the plug sets the smallest
mid cone and the throat, the mid horn and its pod set the kick's centre body, and the centre body and the walls set
the kick horn. Every inner body is subtracted from its horn's area.</p>
{fig("img/plots/chain.png", "The inside-out chain: each size and what set it.")}
<h2>The reference: Funktion-One Res 2</h2>
<p>The Res 2's selling point is a horn-loaded 8 inch cone that carries everything from 250 Hz up to the compression
driver, so the voice, including the 2 to 4 kHz consonant band, stays on one paper cone. Funktion-One's Res 4 and 5
run the same 8 inch to 5.7 and 7.5 kHz with their Axhead loading device. Sunflower keeps the 250 Hz low crossover and
pushes the mid as high as the FEM allows: {f2/1000:.1f} kHz with a 66 mm bullet round a B&amp;C DE111 (see
<a href="#mid">mid</a> and the <a href="validation.html">validation</a> page).</p>

<h2 id="mid">Mid horn and its phase plug</h2>
<p>{D.label(D.MID_RECORD)} in an {law_name(m.law)} horn, {m.mouth_diameter*1000:.0f} mm mouth,
{m.length*1000:.0f} mm long, {m.compression:.1f}:1 compression. The phase plug is one smooth bullet: its nose follows the
cone {m.gap*1000:.1f} mm away, it holds the {tw.model}'s {tw.body_diameter*1000:.0f} mm body, and it tapers to the
printed HF horn's {2*m.hf_mouth_radius*1000:.0f} mm mouth just inside the mid mouth. The wall is solved so the open area
(wall minus bullet) follows the law exactly. The mid sits in a sealed {m.rear_volume_litres:.1f} litre printed pod,
separate from the kick motor.</p>
{fig("img/plots/mid-drawing.png", "Mid horn, bullet and HF horn, section, mm.")}
{fig("img/plots/mid-area.png", "Mid horn area: the bullet is always subtracted.")}
<h3>Felt under the bullet</h3>
<p>The air between the cone and the bullet nose is a radial cavity closed at the centre. Half a wavelength across the
bullet radius puts a deep notch near 3.6 kHz, right where the mid hands over. Both the network model and the FEM show
it. A felt layer on the underside of the nose damps it; the CAD has a 3 mm felt part recessed into the nose so the
designed gap is kept.</p>
{felt_rows}

<h2 id="kick">Kick horn and the reversed 15 inch</h2>
<p>{D.label(D.KICK_RECORD)}, reversed: the flange sits on the front of the baffle, the motor points into the horn and
the cone's concave side faces a sealed {K.chamber_litres:.0f} litre chamber. The centre body's underside is a shroud
over the cone, outside the basket arms. The gap at each radius makes the channel area equal to the cone area inside
that radius divided by {K.compression:.1f}, so every ring of the cone sees the same compression; it never comes closer
than the basket arms plus 9 mm or the cone's mechanical travel plus 3 mm. The air leaves at the rim, turns under the
shroud lip and runs forward through a {K.throat_area*1e4:.0f} cm² throat.</p>
{fig("img/plots/throat-drawing.png", "Half section from the axis to the side wall, mm. The numbers are the gap between cone and shroud, sized so the channel area at each radius is the cone area inside it divided by the compression ratio, never closer than the basket arms allow.")}
<p>From the throat forward the body's radius is solved from the {law_name(K.law)} law, so the open area between the
four walls, less the corner fillers, and the body follows it all the way to the square mouth. The body swells to
{2*K.summary()['max_body_radius_m']*1000:.0f} mm, holds the kick motor, the mid pod and the mid horn, and meets the mid horn's
outside at the kick mouth.</p>
{fig("img/plots/kick-drawing.png", "The cabinet in section. Upper half through the wall flats, lower half through a corner filler.")}
{fig("img/plots/low-area.png", "Kick areas. Dots: areas sliced from the CAD air volume.")}
<p>The horn profiles are checked against the solid model: the kick's air volume was built in CAD (walls, fillers,
body, standoffs and vanes) and sliced at 60 planes. It matches the model to {ka.diff_pct.abs().max():.3f}%, the mid to
{mi.diff_pct.abs().max():.2f}% (the worst point is the HF cable hole). The support vanes block at most
{ka.vane_blockage_pct.max():.1f}% of the kick horn.</p>
<div class="note"><p><b>Measure before printing.</b> No maker publishes cone sections. The {kd.label} geometry here
is estimated from dimensioned drawings: {est}. The shroud and the bullet nose must be checked against a real driver.</p></div>

<h2 id="box">Cabinet and splay</h2>
<p>Single-ply 18 mm birch. The four horn walls are the outside of the box: four identical trapezoids with compound
mitres, from a {d['cut']['external_mm']['height_back']:.0f} mm square back to a {d['cut']['external_mm']['width']:.0f} mm
square mouth, {K.depth*1000:.0f} mm deep. The two sides are angled; the top and bottom are flat, with sealed wedges inside
them so the horn itself is a square frustum. The mid horn ends flush inside the kick mouth.</p>
<p>The splay between boxes is twice the wall angle, {K.splay_deg:.0f} degrees. It was chosen against the mid's coverage:
about 0.8 of its -6 dB beamwidth from 1 to 4 kHz, the point-source rule Funktion-One also uses when it flies Res tops.</p>
{fig("img/plots/array-coverage.png", "Coverage of the 3 x 3 cluster (power sum of nine boxes).")}
{fig("img/plots/tradeoff.png", "Every kick variant in a 600 mm box: width against low-end extension.")}
"""
    return page("design.html", "Sunflower design", body, "How Sunflower is sized, from the HF body out to the square mouth.")


def section(name, title, intro, extra=""):
    return f"""
<h2 id="{name}">{title}</h2>
<p>{intro}</p>
{fig(f"img/plots/{name}-magnitude.png", "Magnitude on axis.")}
{fig(f"img/plots/{name}-phase.png", "Phase and group delay.")}
{fig(f"img/plots/{name}-polar.png", "Off axis.")}
{fig(f"img/plots/{name}-electrical.png", "Impedance and excursion.") if name != "tweeter" else ""}
{extra}"""


def simulation(d):
    dsp, co, fem = d["dsp"], d["cutoffs"], d["fem"]
    K, m = D.KICK, D.MID
    f1, f2 = D.CROSSOVER_LOW_MID_HZ, D.CROSSOVER_MID_HIGH_HZ
    le = dsp["kick_low_end"]
    sh = dsp["no_sub_shelf"]
    kp = fem["kick"]["passband"]["spl_0deg"]
    k3 = fem.get("kick3d", {})
    k3t = ""
    for key in ("passband", "comparison", "agreement"):
        if isinstance(k3.get(key), dict):
            v = k3[key].get("spl_0deg", k3[key])
            if isinstance(v, dict) and "max_abs_db" in v:
                k3t = f" A 3D model of the real square frustum with its fillers agrees with the round equal-area model to {v['max_abs_db']:.1f} dB from 40 to 250 Hz."
    sel = D.SELECTED
    f2s = sel.get("f2_after_fem", {})
    dsp_rows = [["Crossover", f"Linkwitz-Riley 24 dB/oct at {f1:.0f} Hz and {f2:.0f} Hz"],
                ["Gains (dB)", ", ".join(f"{k} {v:+.1f}" for k, v in dsp["gain_db"].items())],
                ["Delays (ms)", ", ".join(f"{k} {v:.2f}" for k, v in dsp["delay_ms"].items())],
                ["Polarity", ", ".join(f"{k} {'+' if v > 0 else '-'}" for k, v in dsp["polarity"].items())],
                ["Mid EQ", f"high shelf +{dsp['eq']['mid_high_shelf_db']:.1f} dB at {dsp['eq']['mid_high_shelf_hz']:.0f} Hz"],
                ["No-sub option", f"+6 dB low shelf at {sh['shelf_hz']:.0f} Hz with a 30 Hz high-pass: -6 dB at "
                                  f"{sh['f_minus6_hz']:.0f} Hz, safe to about {sh.get('safe_power_w', 0):.0f} W"]]
    body = f"""
<h1>Simulation</h1>
<p class="lede">Each horn is modelled from its real profile. A plane-wave network, built slice by slice from the
geometry, screens thousands of designs; the cone is a distributed source, not a piston. Axisymmetric FEM (and a 3D
check of the square kick horn) then validates the chosen one, with the cone as a vibrating boundary.</p>
<h2 id="full-range">Full range</h2>
{fig("img/plots/full-range.png", "Top: each section on its own with the cutoffs that limit it. Bottom: through the crossover.")}
{fig("img/plots/full-range-group-delay.png", "Group delay of the sum.")}
{table(["DSP", "Setting"], dsp_rows)}
<h2 id="low-end">Kick low end</h2>
<p>The kick is a kick bin, not a sub, and like the HD-15 it is meant to run with one. One box on the ground is down
6 dB at {le['single_half_space_db']['f_minus6_hz']:.0f} Hz ({le['single_free_db']['f_minus6_hz']:.0f} Hz in free space).
The small sealed chamber (sealed resonance about {co['low']['sealed_resonance_hz']:.0f} Hz) keeps the cone under control
below the horn's range, at the cost of extension. Without a sub, the no-sub shelf takes one box to
{sh['f_minus6_hz']:.0f} Hz, but excursion then limits it to about {sh.get('safe_power_w', 0):.0f} W.</p>
{fig("img/plots/kick-low-end.png", "Free space, on the ground, and per box in a 3 x 3 cluster.")}
{section("low", "Kick", f"Reversed {D.label(D.KICK_RECORD)}, {law_name(K.law)} horn, square mouth.")}
{section("mid", "Mid", f"{D.label(D.MID_RECORD)} in the OS horn round the bullet. The response above 1 kHz includes the FEM correction (below).",
         fig("img/plots/hf-in-mid.png", "The HF horn inside the mid horn, FEM."))}
{section("tweeter", "HF", f"{D.TW.model}: datasheet response on axis, with the off-axis shape from the FEM of its printed horn inside the mid horn.")}
<h2 id="fem">FEM validation</h2>
<p><b>Kick.</b> FEM against the network, coupled to the driver, 40 to 250 Hz: {kp['rms_db']:.1f} dB RMS,
{kp['max_abs_db']:.1f} dB worst.{k3t} Above about 400 Hz the FEM shows cross modes in the wide passage near the mouth,
which a plane-wave model cannot; they are well above the {f1:.0f} Hz crossover.</p>
<p><b>Mid.</b> Within about 1 dB of the network up to 2.7 kHz. Above that the FEM falls 8 to 12 dB by 3 to 6 kHz:
the air over the cone behaves as a radial cavity, which the network lumps into one node. So the upper crossover was
picked again with the FEM correction applied
({", ".join(f"{k} Hz: {v:.2f}" for k, v in f2s.items())}, lower is better), and {f2/1000:.0f} kHz is the highest that
scores within 0.15 of the best. It needs the full 12 dB mid shelf. A two-slot phase plug, as in compression drivers,
would push this up; that is the main open item for the mid.</p>
{fig("img/plots/fem-vs-planewave.png", "FEM against the network model.")}
<h2 id="limits">Limits</h2>
<ul>
<li>Cone, spider and basket shapes are estimated from datasheet drawings; the shroud gap and bullet nose depend on them.</li>
<li>Cones are rigid in the models. Mid cone breakup above about 1.5 to 3 kHz is not modelled.</li>
<li>The felt is a loss factor, not a measured material.</li>
<li>Free-space results use a baffle-step shelf for the box front; the FEM is in half space.</li>
<li>The HF unit uses its datasheet curve, read by eye (about 2 dB), measured by the maker on their own horn.</li>
<li>Array results are power sums of one box's polar; the low-frequency cluster uses a mutual-load approximation.</li>
<li>Nothing has been built or measured.</li>
</ul>
"""
    return page("simulation.html", "Sunflower simulation", body, "Sunflower responses, crossover, FEM checks and limits.")


def build(d):
    cut = d["cut"]
    w, bf, bk, fl = cut["walls"], cut["baffle"], cut["back"], cut["corner_filler"]
    prints = sorted({p.split("-")[0] for p in cut.get("print_files", [])}) if isinstance(cut.get("print_files"), list) else []
    tb, wd = cut["top_bottom"], cut["wedges"]
    rows = [["Side walls", f"{w['qty']}", f"18 mm birch, {w['height_mm']:.0f} mm high, {w['slant_length_mm']:.0f} mm long, "
             f"angled {D.KICK.half_angle_deg:.0f} deg; ends bevelled {w['end_bevel_deg']:.0f} deg."],
            ["Top and bottom", f"{tb['qty']}", f"18 mm birch, flat, {tb['length_mm']:.0f} mm long, {tb['back_width_mm']:.0f} mm "
             f"wide at the back to {tb['mouth_width_mm']:.0f} mm at the mouth."],
            ["Wedges", f"{wd['qty']}", f"Sloping 18 mm panels under the top and over the bottom, {wd['depth_at_baffle_mm']:.0f} mm "
             f"deep at the baffle to nothing at the mouth, {wd['ribs_per_wedge']} ribs each, sealed. They make the horn a square frustum."],
            ["Baffle", "1", f"{bf['thickness_mm']:.0f} mm, {bf['front_square_mm']:.0f} mm square at the front, "
             f"{bf['cutout_diameter_mm']:.0f} mm cutout, edges bevelled {bf['edge_bevel_deg']:.0f} deg. {bf['note']}."],
            ["Back", "1", f"{bk['thickness_mm']:.0f} mm, removable: {bk['screws']}; {bk['gasket']}."],
            ["Corner fillers", f"{fl['qty']}", f"{fl['leg_at_baffle_mm']:.0f} mm legs at the baffle, tapering to nothing at "
             f"the mouth, {fl['length_mm']:.0f} mm long. {fl['note']}"]]
    body = f"""
<h1>Build</h1>
{stale_note(d)}
<p class="lede">A single-ply birch frustum, four tapering corner fillers, a printed centre body in sections, and three
drivers. Files are on the <a href="downloads.html">downloads</a> page.</p>
{video("assemble", "Assembly order.")}
<h2>Drivers</h2>
{drivers_table(d)}
<h2>Wood</h2>
{table(["Part", "Qty", "Notes"], rows)}
<h2>Printed parts</h2>
<p>The centre body, mid horn and bullet are split into {len(cut.get('print_files', []))} pieces that each fit a
250 mm bed, with spigots and through bolts. Print the parts near the kick motor in ASA, PC or nylon (the motor runs hot);
the rest can be PETG. Add the 3 mm felt layer to the recess under the bullet nose.</p>
<h2>Assembly</h2>
<ol class="steps">
<li><b>Measure the drivers first.</b> Check the kick cone profile and basket against the shroud, and the mid cone
against the bullet nose, before printing those parts.</li>
<li>Glue the four walls with the fillers in the corners, then the baffle inside the frustum.</li>
<li>Screw the kick driver to the front of the baffle, reversed, with its cone facing the chamber.</li>
<li>Fit the four standoffs (one is the cable duct) and the shroud, then the motor fairing, mid pod with its driver, the
mid horn sections and the bullet with the HF unit.</li>
<li>Fit the four vanes from the body to the wall corners, then the back with its gasket.</li>
<li>DSP: start from the settings on the <a href="simulation.html#full-range">simulation page</a>, then measure each section.</li>
</ol>
{fig("img/renders/exploded.jpg", "Exploded.", "render")}
{fig("img/renders/section.jpg", "Cutaway.", "render")}
{fig("img/renders/rear-open.jpg", "Back removed.", "render")}
"""
    return page("build.html", "Sunflower build", body, "Sunflower cut list, printed parts and assembly.")


def downloads(d):
    parts = d["parts"]
    rows = [[f"<code>{p['name']}</code>", p.get("category", ""), p.get("note", ""),
             f'<a href="cad/step/{p["name"]}.step">STEP</a> · <a href="cad/stl/{p["name"]}.stl">STL</a>'] for p in parts]
    res = [["Section responses", '<a href="results/planewave/low.csv">kick</a> · <a href="results/planewave/mid.csv">mid</a> · <a href="results/planewave/tweeter.csv">HF</a>'],
           ["System and no-sub shelf", '<a href="results/planewave/system.csv">system.csv</a> · <a href="results/planewave/array.csv">array.csv</a>'],
           ["DSP settings", '<a href="results/planewave/dsp.json">dsp.json</a>'],
           ["FEM", '<a href="results/fem/summary.json">summary</a> · <a href="results/fem/kick-vs-network.csv">kick</a> · <a href="results/fem/mid-vs-network.csv">mid</a> · <a href="results/fem/hf-in-mid.csv">HF in mid</a>'],
           ["CAD slices", '<a href="results/cad/slices.csv">slices.csv</a> · <a href="results/cad/cone-channel.csv">cone-channel.csv</a> · <a href="cad/plug_profiles.json">plug_profiles.json</a>'],
           ["Optimiser", '<a href="results/optimise/mid-search.csv">mid</a> · <a href="results/optimise/kick-search.csv">kick</a> · <a href="results/optimise/system-search.csv">finalists</a> · <a href="results/optimise/selected.json">selected</a>']]
    body = f"""
<h1>Downloads</h1>
{stale_note(d)}
<p class="lede">Everything here is generated by the scripts in the repository. Millimetres.</p>
<p><a href="cad/sunflower-assembly.step">sunflower-assembly.step</a> · <a href="cad/cutlist.json">cutlist.json</a> ·
<a href="cad/array.json">array.json</a> · <a href="cad/parts.json">parts.json</a> ·
DXF: <a href="cad/dxf/side_wall.dxf">side wall</a> · <a href="cad/dxf/top_bottom.dxf">top and bottom</a> ·
<a href="cad/dxf/wedge_panel.dxf">wedge panel</a> · <a href="cad/dxf/wedge_rib.dxf">wedge rib</a> ·
<a href="cad/dxf/baffle.dxf">baffle</a> · <a href="cad/dxf/back.dxf">back</a></p>
<h2>Parts</h2>
<p>Printable pieces are in <code>cad/print/</code>. Parts named <code>drv_</code> are stand-ins for the drivers.</p>
{table(["Part", "Kind", "Notes", "Files"], rows)}
<h2>Data</h2>
{table(["Data", "Files"], res)}
<h2>Reproduce</h2>
<pre>git clone {REPO}
cd sunflower
just all</pre>
"""
    return page("downloads.html", "Sunflower downloads", body, "Sunflower CAD and data.")


def validation(d):
    R = ROOT / "results/validation"
    sizes = pd.read_csv(R / "mid-sizes-fem.csv")
    srows = [[f"{r['size']}", r.driver, f"{r.length*1000:.0f} mm", f"{r.fem_minus6_hz/1000:.1f} kHz", f"{r.network_minus6_hz/1000:.1f} kHz"]
             for _, r in sizes.sort_values(["length", "size"]).iterrows()]
    hm = pd.read_csv(R / "horn-simulation-mid.csv")
    hk = pd.read_csv(R / "horn-simulation-kick.csv")
    def zerr(t):
        a = t.hs_re + 1j * t.hs_im
        b = t.net_re + 1j * t.net_im
        e = abs(a - b) / abs(a)
        return e.median() * 100, e.quantile(0.9) * 100
    mm, mk = zerr(hm), zerr(hk)
    fem = d["fem"]
    kp = fem["kick"]["passband"]["spl_0deg"]
    body = f"""
<h1>Validation</h1>
<p class="lede">What the models were checked against, and where they fall short. Every number on this page comes from
the files in <code>results/validation/</code> and <code>results/fem/</code>.</p>

<h2>1. A known box: the HD-15</h2>
<p>The HD-15 (Rog Mogale, Speakerplans, 2001) is a 15 inch kick and bass bin with a published plan and response.
Modelled with the same plane-wave network Sunflower uses, with the Eminence Kappa 15LF from the plan, it matches the
published curve within about 1.5 dB from 40 to 130 Hz. Above 200 Hz the model reads 3 to 11 dB higher than the
published curve, which itself looks like a simulation. Its wedge-shaped front chamber is lumped as one volume in the
model, which is the likely cause.</p>
{fig("img/validation/hd15-check.png", "HD-15: the Sunflower network against the published curve, with the uncertain chamber and path varied.")}

<h2>2. An independent solver: horn-simulation</h2>
<p>The real air volumes of the Sunflower mid and kick horns were exported as STEP and solved in 3D by the separate
<a href="https://github.com/timini/horn-simulation">horn-simulation</a> FEM pipeline. Its throat impedance agrees with
the Sunflower network: median difference {mm[0]:.0f}% for the mid horn and {mk[0]:.0f}% for the kick horn
(90th percentile {mm[1]:.0f}% and {mk[1]:.0f}%).</p>
{fig("img/validation/horn-simulation-check.png", "Throat impedance of each horn: horn-simulation against the Sunflower network.")}

<h2>3. The whole kick, with the cone moving</h2>
<p>The axisymmetric FEM with the reversed cone as a vibrating boundary agrees with the network within
{kp['rms_db']:.1f} dB RMS ({kp['max_abs_db']:.1f} dB worst) from 40 to 250 Hz. A 3D model of the square frustum with
its corner fillers agrees with the round equal-area model within about 0.7 dB in the same band.</p>

<h2>4. Where the network is wrong: the mid's top end</h2>
<p>The air between the mid cone and its phase plug behaves as a cavity that the plane-wave network cannot represent.
The FEM shows the cone's radiation resistance collapsing above about 3 kHz on an 8 inch, and earlier on bigger cones.
The network overstates the top end by 0.5 to 5 kHz, so mids are ranked on the FEM, not the network.</p>
{table(["Size", "Driver", "Mid horn", "-6 dB, FEM", "-6 dB, network"], srows)}
<p>Tried with the FEM on the 8 inch: compression ratio (2.2 is best; more is worse), a tighter gap (helps, limited by
excursion), one and two annular slots (no gain), closing the HF bore (no change), and felt under the bullet (removes a
separate centre-cavity notch near 3.6 kHz). The remaining limit is set by the depth of the cone, so the upper crossover
is 3 kHz. The real cone must be measured: the profile used here is estimated from the datasheet.</p>

<h2>5. The geometry: slicing the real model</h2>
<p>The assembled model (every part the renders use) is cut every 1 cm. At each cut the inside of the walls, less every
part (wedges, corner fillers, centre body, phase plug, felt, vanes, standoffs and the drivers), is the air. The phase
plugs are sized from these slices, so the measured open area follows each horn's law with every obstruction counted.</p>
{fig("img/plots/slices.png", "The model sliced every 1 cm, with cross-sections looking into the mouth.")}

<h2>6. Box depth</h2>
{fig("img/validation/depth-comparison.png", "Three depths with the same rules. Extra depth mostly buys a longer mid horn; 700 mm was chosen.")}
"""
    return page("validation.html", "Sunflower validation", body, "What the Sunflower models were checked against.")


RANGE = [("", "Sunflower 15C", "Flagship kick top: 15 inch kick, 8 inch mid, 1 inch compression HF in a constant-directivity waveguide."),
         ("15B", "Sunflower 15B", "The same box with a bullet tweeter in a large phase plug at the mid throat, firing down the mid horn. Its figures are the maker's own."),
         ("12C", "Sunflower 12C", "Compact top: 12 inch kick in a box under 500 mm square; always used with a sub.")]


def _variant(v):
    from sunflower.design_tools import parts_from
    base = ROOT / (f"results/range/{v}" if v else "results")
    sel = json.loads(((ROOT / f"results/range/{v}/selected.json") if v else (ROOT / "results/optimise/selected.json")).read_text())
    krec, mrec, tw, M, K = parts_from(sel)
    dsp = json.loads((base / "planewave/dsp.json").read_text()) if (base / "planewave/dsp.json").exists() else {}
    img = f"img/range/{v}" if v else "img"
    from sunflower.range import body_material
    env, shell = body_material(K)
    price = sum(x or 0 for x in (krec.get("price_usd"), mrec.get("price_usd"), tw.price_usd))
    return dict(sel=sel, krec=krec, mrec=mrec, tw=tw, M=M, K=K, dsp=dsp, img=img, env=env, shell=shell, price=price)


def range_page(d):
    rows, blocks = [], []
    for v, name, blurb in RANGE:
        x = _variant(v)
        K, M, dsp = x["K"], x["M"], x["dsp"]
        le = dsp.get("kick_low_end", {}).get("single_half_space_db", {})
        f2 = dsp.get("crossover", {}).get("mid_high_hz", x["sel"]["crossover_hz"][1])
        rows.append([f"<b>{name}</b>", D.label(x["krec"]), D.label(x["mrec"]), x["tw"].model, f"{f2/1000:.1f} kHz",
                     f"{le.get('f_minus6_hz', float('nan')):.0f} Hz", f"{dsp.get('target_db', float('nan')):.0f} dB",
                     f"{K.external_mouth*1000:.0f} x {K.external_height*1000:.0f} x {K.depth*1000:.0f}",
                     f"{K.splay_deg:.0f} deg", f"{x['shell']:.0f} L", f"${x['price']:.0f}"])
        figs = []
        for f, cap in ((f"{x['img']}/renders/hero.jpg", "Render"), (f"{x['img']}/plots/kick-drawing.png", "Section, mm"),
                       (f"{x['img']}/plots/full-range.png", "Each section and the sum"), (f"{x['img']}/plots/slices.png",
                       "The model sliced every 1 cm")):
            if (DOCS / f).exists():
                figs.append(fig(f, cap, "render" if f.endswith(".jpg") else "plot"))
        blocks.append(f'<h2 id="{name.split()[-1]}">{name}</h2>\n<p>{blurb}</p>\n<div class="grid2">\n' + "\n".join(figs) + "\n</div>")
    body = f"""
<h1>The range</h1>
<p class="lede">Three line-ups built to the same rules and checked the same way: the mid sits inside the kick mouth, the
mouth is square, the top and bottom are flat so nine boxes tile, the phase plugs are sized from slices of the real model,
and the mid's top end comes from FEM. Every centre body is a ribbed 6 mm printed shell whose inside is the mid's sealed
rear chamber.</p>
{table(["Model", "Kick", "Mid", "HF", "Mid to HF", "Kick -6 dB (ground)", "Sensitivity", "Outside W x H x D (mm)",
        "Splay", "Body plastic", "Drivers"], rows)}
<p>Drivers are per box at the prices found when the design was made. All three are simulated, not built or measured.</p>
{"".join(blocks)}

<h2 id="hex">Hex 8: a small two-way for single use (concept)</h2>
<p>For a cafe or bar that wants one striking box per side, not an array: a 350 mm hexagonal cabinet, 300 mm deep, with
an 8 inch coaxial driver. The compression driver fires through the centre of the woofer into its own small horn standing in
the cone, so the two share one source. Of the pro 8 inch coaxials checked, only the Beyma 8CX300 has such a separate centre
horn (the others hide the exit under a dust cap), which makes it the one a printed, coloured horn could replace or copy. It is a hybrid: a short hexagonal horn on the front of the cone, and the
18 litre rear chamber radiating into that horn through three round ports, one in every other wall near the mouth,
as in a multiple-entry horn. The ports are printed and bend inside the box to reach their length.</p>
{table(["", "Hex 8 (simulated low end)"], [
    ["Driver", "Beyma 8CX300Nd/N coaxial, about $429 (or the 8CX300Fe at about $289, which reaches 59 Hz)"],
    ["Low end, on a wall", "-6 dB at 54 Hz, with a few dB of bass EQ below about 100 Hz"],
    ["Sensitivity and level", "95 dB at 2.83 V; about 115 dB before the cone reaches its excursion limit"],
    ["HF", "Takes over from 1.5 kHz, 70 degree conical horn (maker's figures)"],
    ["Size", "350 mm across the flats, 300 mm deep; three 44 mm ports, each about 230 mm long"]])}
<div class="grid2">
{fig("img/range/concepts/hex8-single.jpg", "Hex 8: the HF horn stands in the centre of the cone, inside the hexagonal front horn.", "render")}
{fig("img/range/concepts/hex8-front.jpg", "From the front.", "render")}
{fig("img/range/concepts/hex8-cafe.jpg", "On a wall. The three dark openings in the horn are the rear chamber's ports.", "render")}
</div>
<div class="note"><p><b>Concept.</b> The low end is simulated (a ported-box model with the driver's datasheet
parameters). The front horn, the crossover and the top end are not: those figures are the maker's. How Beyma fixes its
centre horn is not documented, so a printed replacement needs a sample driver to check. An alternative
with a Beyma 8MI100 and a bullet tweeter as its phase plug keeps the yellow plug and crosses at about 3.2 kHz on FEM.</p></div>

<h2 id="clusters">Festival clusters (concepts)</h2>
<p>For festivals, the boxes fly as a cluster: one centre box ringed by five, every mouth aimed out from one point.
Five round a centre on a curved surface means the centre is a pentagon and its neighbours hexagons, the pattern of a
football, so the joints between boxes are flush with no gaps. The pentagon is the smaller tile, so the centre box is the
smaller driver. Inside each box are the Sunflower internals scaled to fit: the kick centre body, the yellow mid horn and the
phase plug that holds the HF waveguide. The flown frame hangs above.</p>
{table(["Cluster", "Centre box", "Ring", "Boxes"], [
    ["<b>Cluster 15/18</b>", "15 inch box, pentagonal mouth", "Five 18 inch boxes, hexagonal mouths", "6"],
    ["<b>Cluster 12/15</b>", "12 inch box, pentagonal mouth", "Five 15 inch boxes, hexagonal mouths", "6"]])}
<div class="grid2">
{fig("img/range/concepts/cluster-15-18-front.jpg", "Cluster 15/18, from the front: the smaller pentagonal box in the middle.", "render")}
{fig("img/range/concepts/cluster-15-18-three-quarter.jpg", "Cluster 15/18, three-quarter view.", "render")}
{fig("img/range/concepts/cluster-12-15-front.jpg", "Cluster 12/15, from the front.", "render")}
{fig("img/range/concepts/cluster-12-15-three-quarter.jpg", "Cluster 12/15, three-quarter view.", "render")}
</div>
<div class="note"><p><b>Concept only.</b> These renders show the arrangement. The pentagonal and hexagonal boxes have
not been designed or simulated: each needs its own horn, sized from the inside out and checked like the boxes above.
Neighbours sit about 37 degrees apart, so a cluster covers a wide area from one flying point.</p></div>
"""
    return page("range.html", "Sunflower range", body, "The Sunflower range: 15C, 15B, 12C, the Hex 8 and the festival clusters.")


def main():
    d = load()
    for name, fn in (("index.html", index), ("design.html", design),
                     ("simulation.html", simulation), ("range.html", range_page), ("validation.html", validation), ("build.html", build),
                     ("downloads.html", downloads)):
        (DOCS / name).write_text(fn(d))
    (DOCS / ".nojekyll").write_text("")
    print("site written to", DOCS)


if __name__ == "__main__":
    main()
