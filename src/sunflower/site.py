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

PAGES = [("index.html", "Product"), ("design.html", "Design"), ("simulation.html", "Simulation"),
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
    s = data["system"]
    f = s.frequency_hz.values
    on = s.sum_0deg_spl_db.values
    ref = on[(f >= 200) & (f <= 2000)].mean()
    data["kpi"] = dict(sensitivity=float(data["dsp"]["target_db"]), f6=float(f[np.argmax(on > ref - 6)]))
    return data


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
Here the three are nested on one axis. The HF horn is the mid's phase plug, and the mid horn is the core of the kick
horn's centre body.</p>
<figure class="hero"><img src="img/renders/hero.jpg" alt="Rendered Sunflower cabinet"></figure>
<div class="stat">
<div><b>{k['sensitivity']:.0f} dB</b><span>2.83 V, 1 m</span></div>
<div><b>{low['f_minus6_hz']:.0f} Hz</b><span>kick -6 dB on the ground; {shelf['f_minus6_hz']:.0f} Hz with the no-sub shelf</span></div>
<div><b>{f1:.0f} Hz / {f2/1000:.0f} kHz</b><span>crossovers: kick, vocal-range mid, HF</span></div>
<div><b>{ex['width']:.0f} x {ex['height_front']:.0f} x {ex['depth']:.0f}</b><span>mm at the mouth; the mid horn stands
{K.protrusion*1000:.0f} mm proud</span></div>
</div>
<div class="note"><p><b>Status.</b> Simulated, not built or measured. The cone and basket shapes of the drivers are
estimated from datasheet drawings, and the phase plugs must be checked against real drivers before printing. The
<a href="simulation.html#limits">limits</a> section lists what the models leave out.</p></div>
{video("turntable", "One box, all the way round.")}
<h2>Nine boxes, one point source</h2>
<p>All four walls are angled at {K.half_angle_deg:.0f} degrees, so neighbouring boxes splay {K.splay_deg:.0f} degrees in
both directions and a 3 x 3 block forms a curved point source. The edge joints close flush. Identical square boxes
cannot also close at the four corners, so tapered spacers fill those joints (up to
{d['cut']['array_spacers']['max_thickness_mm']:.0f} mm thick).</p>
{fig("img/renders/stack.jpg", "Nine boxes as a curved point source, with the corner spacers.", "render")}
{video("orbit", "Around the 3 x 3 cluster.")}
<div class="grid2">
{fig("img/renders/front-dims.jpg", "Into the mouth, mm. Completely square at the mouth; the corner fillers make the throat an octagon.", "render")}
{fig("img/renders/section-side-dims.jpg", "Side section, mm.", "render")}
</div>
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
pushes the mid as high as the models allow: {f2/1000:.0f} kHz (see <a href="#mid">mid</a>).</p>

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
{fig("img/plots/throat-drawing.png", "The reversed cone, the shroud and the gap, mm.")}
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
square mouth, {K.depth*1000:.0f} mm deep. The mid horn protrudes {K.protrusion*1000:.0f} mm beyond the kick mouth on a
rolled lip, which keeps the box within 600 mm and keeps mid sound from diffracting into the kick horn.</p>
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
    sp = cut["array_spacers"]
    prints = sorted({p.split("-")[0] for p in cut.get("print_files", [])}) if isinstance(cut.get("print_files"), list) else []
    rows = [["Walls", f"{w['qty']}", f"18 mm birch trapezoids, outer face {w['outer_face']['back_edge']:.0f} to "
             f"{w['outer_face']['mouth_edge']:.0f} mm, {w['slant_length_mm']:.0f} mm long. Blade bevel "
             f"{w['blade_bevel_deg']:.1f} deg on both long edges, ends bevelled {w['end_bevel_deg']:.0f} deg."],
            ["Baffle", "1", f"{bf['thickness_mm']:.0f} mm, {bf['front_square_mm']:.0f} mm square at the front, "
             f"{bf['cutout_diameter_mm']:.0f} mm cutout, edges bevelled {bf['edge_bevel_deg']:.0f} deg. {bf['note']}."],
            ["Back", "1", f"{bk['thickness_mm']:.0f} mm, removable: {bk['screws']}; {bk['gasket']}."],
            ["Corner fillers", f"{fl['qty']}", f"{fl['leg_at_baffle_mm']:.0f} mm legs at the baffle, tapering to nothing at "
             f"the mouth, {fl['length_mm']:.0f} mm long. {fl['note']}"],
            ["Array spacers", f"{sp['qty']}", f"Tapered, {sp['min_thickness_mm']:.0f} to {sp['max_thickness_mm']:.0f} mm, "
             "for the 3 x 3 cluster's corner joints."]]
    body = f"""
<h1>Build</h1>
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
           ["CAD area check", '<a href="results/cad/area-check.csv">area-check.csv</a>'],
           ["Optimiser", '<a href="results/optimise/mid-search.csv">mid</a> · <a href="results/optimise/kick-search.csv">kick</a> · <a href="results/optimise/system-search.csv">finalists</a> · <a href="results/optimise/selected.json">selected</a>']]
    body = f"""
<h1>Downloads</h1>
<p class="lede">Everything here is generated by the scripts in the repository. Millimetres.</p>
<p><a href="cad/sunflower-assembly.step">sunflower-assembly.step</a> · <a href="cad/cutlist.json">cutlist.json</a> ·
<a href="cad/array.json">array.json</a> · <a href="cad/parts.json">parts.json</a> ·
<a href="cad/dxf/wall.dxf">wall.dxf</a> · <a href="cad/dxf/baffle.dxf">baffle.dxf</a> · <a href="cad/dxf/back.dxf">back.dxf</a></p>
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


def main():
    d = load()
    for name, fn in (("index.html", index), ("design.html", design),
                     ("simulation.html", simulation), ("build.html", build), ("downloads.html", downloads)):
        (DOCS / name).write_text(fn(d))
    (DOCS / ".nojekyll").write_text("")
    print("site written to", DOCS)


if __name__ == "__main__":
    main()
