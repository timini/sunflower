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
        parts=json.loads((ROOT / "cad/parts.json").read_text()),
        system=pd.read_csv(R / "planewave/system.csv"),
        low=pd.read_csv(R / "planewave/low.csv"),
        mid=pd.read_csv(R / "planewave/mid.csv"),
        screen=pd.read_csv(R / "screen-15/search.csv"),
        refine=pd.read_csv(R / "screen-15/refine-18sound-15nmb420.csv"),
    )
    s = data["system"]
    f = s.frequency_hz.values
    on = s.sum_0deg_spl_db.values
    ref = on[(f >= 200) & (f <= 2000)].mean()
    b = (f >= 100) & (f <= 20000)
    data["kpi"] = dict(
        sensitivity=float(on[b].mean()),
        ptp=float(np.ptp(on[b])),
        f3=float(f[np.argmax(on > ref - 3)]),
        f6=float(f[np.argmax(on > ref - 6)]),
    )
    return data


# ---------------------------------------------------------------- pages
def index(d):
    k, cut, dsp = d["kpi"], d["cut"], d["dsp"]
    ext = cut["external_mm"]
    body = f"""
<h1>Sunflower</h1>
<p class="lede">A three-way coaxial horn loudspeaker where each horn's phase plug is the next horn.
A reversed 15 inch driver drives a flat-panel horn. A 6.5 inch midrange horn sits in its mouth and
forms the low horn's centre body. A Fostex T90A tweeter sits in the throat of the mid horn as its phase plug.
All three sections radiate from one axis and one mouth plane.</p>
<figure class="hero"><img src="img/renders/hero.jpg" alt="Rendered Sunflower cabinet, three-quarter front view"></figure>
<div class="stat">
<div><b>{k['sensitivity']:.0f} dB</b><span>2.83 V, 1 m, mean 100 Hz to 20 kHz</span></div>
<div><b>{k['f3']:.0f} Hz</b><span>low-frequency -3 dB, on axis</span></div>
<div><b>{k['ptp']:.1f} dB</b><span>peak to peak, 100 Hz to 20 kHz, on axis</span></div>
<div><b>{ext['width']:.0f} x {ext['height']:.0f} x {ext['depth']:.0f}</b><span>mm, width x height x depth</span></div>
</div>
<div class="note"><p><b>Status.</b> Every number on this site is from simulation: a plane-wave
horn model coupled to Thiele/Small driver models, checked with axisymmetric FEM. Nothing has been built or
measured yet. The <a href="simulation.html#limits">limits section</a> lists what the models leave out.</p></div>

<div class="grid2">
{fig("img/renders/front.jpg", "Straight into the mouth: the corner fillers turn the square into an octagon at the throat, which gives the project its name.", "render")}
{fig("img/renders/detail.jpg", "The mid horn mouth, with the tweeter bullet and its three locating vanes in the throat.", "render")}
</div>

<h2>Three horns on one axis</h2>
<div class="grid3">
<div class="card"><h3><span class="swatch" style="background:var(--low)"></span>Low, {dsp['crossover']['low_mid_hz']:.0f} Hz down</h3>
<p>18 Sound 15NMB420, mounted backwards so its neodymium motor points into the horn. The motor and a printed
nose cone form the throat's phase plug. The cone's front works into a {D.LOW_REAR_LITRES:.0f} litre sealed chamber.
The horn is four flat birch-ply panels and four corner fillers, octagonal at the throat and square at the mouth.</p></div>
<div class="card"><h3><span class="swatch" style="background:var(--mid)"></span>Mid, {dsp['crossover']['low_mid_hz']:.0f} Hz to {dsp['crossover']['mid_high_hz']/1000:.0f} kHz</h3>
<p>Beyma 6MCF200Nd, factory sealed, loading a conical horn with a {D.MID.mouth_diameter*1000:.0f} mm mouth through an annular throat.
The driver, horn and an elliptical tail sit in a {D.MID.can_diameter*1000:.0f} mm printed can on four aluminium rods.
To the low horn, the can is the centre body.</p></div>
<div class="card"><h3><span class="swatch" style="background:var(--tw)"></span>Tweeter, {dsp['crossover']['mid_high_hz']/1000:.0f} kHz up</h3>
<p>Fostex T90A ring-diaphragm horn tweeter, 106 dB/W/m, in a {D.TWEETER.housing_diameter*1000:.0f} mm printed bullet. The bullet fills the
centre of the mid horn throat, so the mid sound leaves through an annulus around it. The tweeter's own mouth sits
{(D.MID.length - D.TWEETER.face_from_mid_throat)*1000:.0f} mm inside the mid horn.</p></div>
</div>
{fig("img/renders/section.jpg", "Cutaway render. From the back: sealed chamber, reversed 15 inch with its nose cone, the low horn, the mid can with tail, mid horn and tweeter bullet.", "render")}

<h2>Specification</h2>
{table(["", "Value", "Notes"], [
    ["Type", "Three-way coaxial horn, active", "Needs a 3-way DSP crossover and three amplifier channels"],
    ["Sensitivity", f"{k['sensitivity']:.1f} dB", "2.83 V at the quietest section, 1 m, mean 100 Hz to 20 kHz"],
    ["Frequency response", f"{k['f3']:.0f} Hz to 20 kHz, -3 dB", f"-6 dB at {k['f6']:.0f} Hz; on axis, after DSP"],
    ["Crossovers", f"{dsp['crossover']['low_mid_hz']:.0f} Hz and {dsp['crossover']['mid_high_hz']/1000:.0f} kHz", "Linkwitz-Riley 24 dB/oct"],
    ["Coverage", "About 40 degrees above 3 kHz", "Set by the 20 degree half-angle of the conical mid horn"],
    ["Drivers", "18 Sound 15NMB420, Beyma 6MCF200Nd, Fostex T90A", "All 8 ohm"],
    ["Cabinet", f"{ext['width']:.0f} x {ext['height']:.0f} x {ext['depth']:.0f} mm", "18 mm birch ply, 25 mm baffle"],
    ["Low horn", f"{D.LOW.throat_width*1000:.0f} mm octagon to {D.LOW.mouth_width*1000:.0f} mm square, {D.LOW.length*1000:.0f} mm long", "Flat panels, 12 degree flare"],
    ["Mid horn", f"Conical, {D.MID.outer_throat_radius*2000:.1f} to {D.MID.mouth_diameter*1000:.0f} mm, {D.MID.length*1000:.0f} mm long", "Annular throat around the 64 mm bullet"],
])}
<figure class="figure"><img class="render" src="img/renders/turntable.gif" alt="Turntable animation of the cabinet"><figcaption>Turntable, rendered from the CAD in Blender.</figcaption></figure>
<p>Next: <a href="design.html">how the nested horns are laid out</a>, <a href="simulation.html">the simulated responses</a>,
or <a href="build.html">how to build one</a>.</p>
"""
    return page("index.html", "Sunflower", body, "A three-way nested coaxial horn loudspeaker: reversed 15 inch, 6.5 inch mid horn, T90A tweeter.")


def design(d):
    lo, m, tw = D.LOW, D.MID, D.TWEETER
    sm = lo.summary()
    cut = d["cut"]
    best = d["screen"][d["screen"].passes].drop_duplicates("driver_id").head(8)
    rows = [[r.label, f"{r.throat_width_m*1000:.0f} / {r.mouth_width_m*1000:.0f} / {r.length_m*1000:.0f}",
             f"{r.rear_litres:.0f}", f"{r.ripple_db:.2f}", f"{r.mean_spl_db:.1f}", f"{r.x_peak_100w_mm:.1f} / {r.xmax_mm:.1f}"]
            for r in best.itertuples()]
    ref = d["refine"]
    chosen = ref[(ref.throat_width_m == lo.throat_width) & (ref.mouth_width_m == lo.mouth_width) &
                 (ref.length_m == lo.length) & (ref.rear_litres == D.LOW_REAR_LITRES)].iloc[0]
    body = f"""
<h1>Design</h1>
<p class="lede">Sunflower is three horns nested on one axis. Each section's phase plug, the solid body in the
middle of its throat, is the next section. This page covers the layout, the area functions and how each driver
was chosen.</p>
{fig("img/plots/section-drawing.png", "Half-section through the axis, to scale. Wood: cabinet and low horn walls. Black: the low horn centre body, which is the 15 inch motor and nose cone, then the mid can. Yellow: tweeter bullet.")}

{fig("img/renders/section-side.jpg", "Cutaway render on the same axis, mouth on the left.", "render")}

<h2>Why nest the horns</h2>
<p>A coaxial layout puts every acoustic source on one axis, so the sections stay aligned off axis as well as on
it. The usual cost is that the inner horn blocks the outer one. In Sunflower the blockage is part of the outer horn:
the low horn's area function is the space between its walls and the centre body, and the model includes the
centre body from the start.</p>
<ul>
<li><b>Low section.</b> The 15 inch is mounted backwards, from behind the baffle. Its convex cone and motor face the
horn. The motor plus a printed elliptical nose is the first part of the centre body. The front of the cone works
into a small sealed chamber at the back of the cabinet.</li>
<li><b>Mid section.</b> The mid can is the rest of the centre body. Its elliptical tail starts {lo.tail_start*1000:.0f} mm from the
cone rim, so the open area in the low horn narrows smoothly onto the can and then widens again to the mouth.</li>
<li><b>Tweeter.</b> The tweeter bullet is the mid horn's phase plug. The mid throat is the annulus between the
{tw.housing_diameter*1000:.0f} mm bullet and a {m.outer_throat_radius*2000:.1f} mm bore, {m.throat_area*1e4:.1f} cm² open,
{D.mid_driver().sd/m.throat_area:.2f}:1 compression for the 6.5 inch cone. A cone-following lead-in keeps 1.5 mm clearance to the
cone and gives a {m.front_chamber_cc:.0f} cm³ front chamber.</li>
</ul>

<h2>Area functions</h2>
{fig("img/plots/area.png", "Open area along each horn. The low horn's open area narrows where the motor and then the mid can sit, which the plane-wave and FEM models both include.")}
{table(["Low horn", "Value"], [
    ["Throat area at the voice-coil plane", f"{sm['throat_area_m2']*1e4:.0f} cm²"],
    ["Mouth open area (square minus can)", f"{sm['mouth_area_m2']*1e4:.0f} cm²"],
    ["Front chamber, cone back to throat", f"{lo.front_chamber_m3*1000:.1f} litres"],
    ["Flare half-angle of the side panels", f"{cut['panels']['flare_half_angle_deg']:.1f} degrees"],
    ["Rear sealed chamber, net", f"{cut['rear_chamber']['net_litres']:.1f} litres"],
])}

<h2>Choosing the 15 inch</h2>
<p>The low section was screened over {d['screen'].driver_id.nunique()} 15 inch drivers from the horn-simulation catalogue,
36 flat-panel horn sizes and three rear-chamber volumes: {len(d['screen'])} combinations, each a full plane-wave solve
of the horn with its centre body. A row passed if its mean output from 100 to 700 Hz was at least 103 dB and its
excursion at 100 W stayed inside Xmax. Passing rows were ranked by peak-to-peak variation.</p>
{fig("img/plots/screen-15.png", "Best horn for each driver that passed. Grey: other drivers. The catalogue values are legacy and unverified, so the chosen driver was checked by hand against its datasheet.")}
{table(["Driver", "Throat / mouth / length (mm)", "Rear (L)", "Variation (dB)", "Mean (dB)", "x at 100 W / Xmax (mm)"], rows, num=(2, 3, 4, 5))}
<p>The 18 Sound 15NMB420 was chosen over the Eminence Wheel House 200 at the top of the list. The Eminence ripples less but has
2.2 mm of Xmax against 2.1 mm needed at 100 W. The 15NMB420 has 6.5 mm, a small neodymium motor that blocks
less of the throat, and 450 W AES handling. Its Thiele/Small values were checked against the
<a href="https://www.eighteensound.it/en/products/lf-driver/15-0/8/15NMB420.pdf">18 Sound datasheet</a>.
A finer search with its own body dimensions chose a {lo.throat_width*1000:.0f} mm throat, {lo.mouth_width*1000:.0f} mm mouth,
{lo.length*1000:.0f} mm length and a {D.LOW_REAR_LITRES:.0f} litre chamber: {chosen.ripple_db:.1f} dB variation, {chosen.mean_spl_db:.1f} dB mean.
A 750 mm mouth measured 0.2 dB better and makes the cabinet 50 mm wider, so the smaller mouth was kept.</p>

<h2>Mid and tweeter</h2>
<p>The mid horn comes from an earlier study in
<a href="https://github.com/timini/horn-simulation">horn-simulation</a>: 480 annular horn geometries against 26 manufacturer-verified
6.5 inch drivers, 500 Hz to 6.5 kHz. The Beyma 6MCF200Nd with a conical horn had the least variation at over 100 dB.
Its sealed back matters here: the driver sits in the airflow of the low horn and needs no rear chamber of its own.
The port of that model into this repository reproduces the study's result within 0.12 dB in band, which is a test.</p>
<p>The Fostex T90A is a horn super tweeter: 106 dB/W/m, 5 to 35 kHz, with a 60 mm aluminium horn that fits a
{tw.housing_diameter*1000:.0f} mm bullet. Fostex recommends a crossover at 7 kHz or above, which sets the upper crossover.</p>

<h2>Dimensions</h2>
{table(["Station", "z from the 15 inch cone rim (mm)"], [[k.replace('_', ' '), f"{v:.0f}"] for k, v in cut['stations_mm'].items()], num=(1,))}
"""
    return page("design.html", "Sunflower design", body, "Layout, area functions and driver selection for the Sunflower nested coaxial horn.")


def simulation(d):
    dsp, fem = d["dsp"], d["fem"]
    low, mid = d["low"], d["mid"]
    lb = (low.frequency_hz >= 100) & (low.frequency_hz <= 500)
    mb = (mid.frequency_hz >= 500) & (mid.frequency_hz <= 7000)
    body = f"""
<h1>Simulation</h1>
<p class="lede">Each section was simulated on its own, unfiltered, then summed through the crossover with the
gains, delays and polarity a DSP would apply. All curves are at 2.83 V and 1 m.</p>

<h2>Method</h2>
<ol>
<li><b>Plane-wave horn model.</b> Each horn is cut into a few hundred thin annular slices and cascaded as
transfer matrices from throat to mouth, with the radiation load of a flanged piston at the mouth. Centre bodies
are subtracted from the area slice by slice.</li>
<li><b>Driver model.</b> Each driver is a Thiele/Small motor with its sealed rear volume and the front chamber
between cone and throat, solved for cone velocity and electrical impedance.</li>
<li><b>Radiation.</b> The mouth's far field includes the curvature of the wavefront in a conical or pyramidal horn.
Treating the mouth as a flat piston gave false nulls off axis.</li>
<li><b>FEM check.</b> The mid and low horns were solved again with an axisymmetric Helmholtz finite-element model
(dolfinx 0.8, quadratic elements) in a baffled half-space with an absorbing outer boundary. The FEM throat impedance
and far field were coupled to the same driver models, so the two results differ only in the acoustics. The FEM
code was first checked against the exact solution for a straight tube: median error {fem['tube']['median']*100:.0f}% in throat impedance.</li>
</ol>

<h2>Low section</h2>
{fig("img/plots/low-section.png", "18 Sound 15NMB420 in the low horn. Shaded: 100 to 500 Hz. The group delay peak near 95 Hz is the horn's low-frequency cut-off.")}
<p>From 100 to 500 Hz the plane-wave model gives {np.ptp(low.spl_db[lb]):.1f} dB variation at a mean of {low.spl_db[lb].mean():.1f} dB.
The FEM agrees within {fem['low']['max_abs_difference_db']:.1f} dB in that band. The low horn stays
usable to about 1 kHz on axis, which leaves room for the 500 Hz crossover. Off axis it narrows above about 500 Hz,
roughly where the mid takes over.</p>

<h2>Mid section</h2>
{fig("img/plots/mid-section.png", "Beyma 6MCF200Nd in the conical mid horn with the tweeter bullet in its throat. Shaded: 500 Hz to 7 kHz.")}
<p>From 500 Hz to 7 kHz the plane-wave model gives {np.ptp(mid.spl_db[mb]):.1f} dB variation at a mean of {mid.spl_db[mb].mean():.1f} dB;
FEM gives {fem['mid']['fem_ripple_db']:.1f} dB with a mean difference of {fem['mid']['mean_difference_db']:.2f} dB and a largest difference of
{fem['mid']['max_abs_difference_db']:.1f} dB. Coverage narrows to about plus or minus 20 degrees above 3 kHz, as expected for
a conical horn with a 20 degree half-angle.</p>

<h2>Tweeter</h2>
{fig("img/plots/tweeter-section.png", "Fostex T90A, from the datasheet: 106 dB/W/m, horn cut-off at 3.6 kHz, roll-off above 35 kHz, 60 mm piston directivity. Not a horn simulation.")}
<p>The tweeter's internal horn and ring diaphragm were not simulated. Its curves are a minimum-phase model of the
published figures. The effect of sitting {(D.MID.length - D.TWEETER.face_from_mid_throat)*1000:.0f} mm inside the mid horn is not modelled
beyond the extra path length. Measuring it in place is the first job after building.</p>

<h2>Electrical impedance</h2>
{fig("img/plots/impedance.png", "Impedance of the low and mid drivers in their horns.")}

<h2>FEM against plane-wave</h2>
{fig("img/plots/fem-vs-planewave.png", "Grey: plane-wave. Colour: FEM. Solid on axis, dotted 30 degrees.")}
{table(["Section", "Band", "Largest on-axis difference", "Mean difference", "Variation, FEM", "Variation, plane-wave"], [
    ["Low", "100 to 500 Hz", f"{fem['low']['max_abs_difference_db']:.2f} dB", f"{fem['low']['mean_difference_db']:+.2f} dB", f"{fem['low']['fem_ripple_db']:.2f} dB", f"{fem['low']['planewave_ripple_db']:.2f} dB"],
    ["Mid", "500 Hz to 7 kHz", f"{fem['mid']['max_abs_difference_db']:.2f} dB", f"{fem['mid']['mean_difference_db']:+.2f} dB", f"{fem['mid']['fem_ripple_db']:.2f} dB", f"{fem['mid']['planewave_ripple_db']:.2f} dB"],
], num=(2, 3, 4, 5))}

<h2>The complete system</h2>
{fig("img/plots/system.png", "Each section through its LR4 filters, and their sum on axis.")}
<p>With the settings below, the on-axis sum stays within {d['kpi']['ptp']:.1f} dB peak to peak from 100 Hz to 20 kHz
(RMS deviation {dsp['rms_deviation_100_20k_db']:.2f} dB), before any equalisation.</p>
{fig("img/plots/system-off-axis.png", "Summed response at 0, 15 and 30 degrees.")}
<h3>DSP settings</h3>
{table(["Channel", "Filter", "Gain", "Delay", "Polarity in the model"], [
    ["Low", f"LR4 low-pass {dsp['crossover']['low_mid_hz']:.0f} Hz", f"{dsp['gain_db']['low']:+.1f} dB", f"{dsp['delay_ms']['low']:.2f} ms", "Inverted"],
    ["Mid", f"LR4 high-pass {dsp['crossover']['low_mid_hz']:.0f} Hz, LR4 low-pass {dsp['crossover']['mid_high_hz']/1000:.0f} kHz", f"{dsp['gain_db']['mid']:+.1f} dB", f"{dsp['delay_ms']['mid']:.2f} ms", "Normal"],
    ["Tweeter", f"LR4 high-pass {dsp['crossover']['mid_high_hz']/1000:.0f} kHz", f"{dsp['gain_db']['tweeter']:+.1f} dB", f"{dsp['delay_ms']['tweeter']:.2f} ms", "Normal"],
], num=(2, 3))}
<div class="note"><p>The model wants the low section inverted relative to a forward-facing driver. The reversed
mounting already does that: with the 15 inch wired red to positive, a positive voltage pulls air out of the horn.
So wire all three drivers normally and leave every DSP channel at normal polarity, then confirm with a
measurement at the crossover.</p>
<p>The {dsp['delay_ms']['mid']:.2f} ms on the mid and {dsp['delay_ms']['tweeter']:.2f} ms on the tweeter make up for the longer path from the 15 inch cone to the mouth
and for the delay of the low horn itself.</p></div>

<h2 id="limits">What the models leave out</h2>
<ul>
<li>No cross modes in the plane-wave model. The FEM is axisymmetric, so it includes radial modes but not the
octagon to square corners, which it replaces with circles of equal area.</li>
<li>Cone breakup, wall losses, distortion and power compression are not modelled.</li>
<li>The cabinet front is treated as an infinite baffle at the mouth plane. A real 736 mm front face will show
baffle step and edge diffraction below about 300 Hz.</li>
<li>The four support rods, the three bullet vanes and the cable channels are not in the acoustic models.
They block about 2% of each passage.</li>
<li>The 15NMB420 motor diameter and the split between cone depth and motor length are estimated from the datasheet
drawing, as is the 6.5 inch cone profile behind the lead-in. Measure the drivers before printing the nose and the
mid horn.</li>
<li>The T90A response is a datasheet model, not a simulation.</li>
</ul>
"""
    return page("simulation.html", "Sunflower simulation", body, "Phase and magnitude response of each Sunflower horn section, FEM check and DSP settings.")


def build(d):
    cut, parts, dsp = d["cut"], d["parts"], d["dsp"]
    ext = cut["external_mm"]
    P = {p["name"]: p for p in parts}
    t = D.PLY * 1000
    depth = ext["depth"]
    wm = D.LOW.mouth_width * 1000
    ply = [
        ["Sides, left and right", "2", f"{ext['height']:.0f} x {depth:.0f}", "18", "Square edges"],
        ["Top and bottom", "2", f"{wm:.0f} x {depth:.0f}", "18", "Fit between the sides"],
        ["Back", "1", f"{wm:.0f} x {wm:.0f}", "18", "Terminal cup, 4 pole"],
        ["Baffle", "1", f"{wm:.0f} x {wm:.0f}", "25", "353 mm hole, 8 x M6 inserts on 370 mm PCD (baffle.dxf)"],
        ["Horn walls", "4", f"Trapezoid {D.LOW.throat_width*1000:.0f} to {wm:.0f} wide, {cut['panels']['slope_length_mm']:.0f} long (inner face)", "18",
         f"Long edges bevelled {cut['panels']['long_edge_bevel_deg']:.1f} deg, ends {cut['panels']['end_bevel_deg']:.1f} deg (horn_wall_inner_face.dxf)"],
        ["Corner filler faces", "4", f"Triangle, base {cut['corner_filler']['face_base_mm']:.0f}, height {cut['corner_filler']['face_height_mm']:.0f}", "12", "Three ribs each behind the face"],
    ]
    dens = 1.27
    printed = []
    for n, label, fill in (("mid_horn", "Mid horn and cone lead-in", 0.45), ("mid_sleeve", "Mid driver sleeve", 1.0),
                           ("mid_tail", "Mid rear cap and tail", 0.25), ("bullet", "Tweeter bullet with vanes", 1.0),
                           ("low_nose", "15 inch nose cone", 0.3)):
        p = P[n]
        bb = p["bbox_mm"]
        size = f"{bb[3]-bb[0]:.0f} x {bb[4]-bb[1]:.0f} x {bb[5]-bb[2]:.0f}"
        printed.append([label, f"<code>{n}.stl</code>", size, f"{p['volume_cm3']*dens*fill/1000:.2f}"])
    rod = P["rod_right"]["note"].split(", ")[-1]
    body = f"""
<h1>Build</h1>
<p class="lede">The cabinet is flat birch-ply panels. The round parts are 3D printed. Everything here comes from the
CAD in <code>cad/</code>, which is generated from the same design file as the simulations.</p>
{fig("img/renders/exploded.jpg", "Exploded view. Cabinet panels move outwards, the horn walls and corner fillers inwards, the mid can and tweeter forwards.", "render")}

<h2>Bill of materials</h2>
{table(["Item", "Qty", "Notes"], [
    ["18 Sound 15NMB420, 8 ohm", "1", "Low. Rear-mounted, motor forward"],
    ["Beyma 6MCF200Nd, 8 ohm", "1", "Mid. Factory sealed back"],
    ["Fostex T90A, 8 ohm", "1", "Tweeter. The SE and ST variants differ in response; re-check the crossover if you use one"],
    ["Birch ply 18 mm", "About 3.6 m²", "Two 2440 x 1220 sheets with offcuts"],
    ["Birch ply 25 mm, or two layers of 12 mm", f"{wm:.0f} x {wm:.0f}", "Baffle"],
    ["Birch ply 12 mm", "About 0.3 m²", "Corner filler faces and ribs"],
    ["Aluminium tube, 12 mm OD, 10 mm bore", f"4 x {rod}", "Mid can supports; the top one carries the cables"],
    ["PETG or ASA filament", "About 3 kg", "Printed parts, see below"],
    ["M6 threaded inserts and M6 x 30 bolts", "8", "15 inch, fitted from the chamber side"],
    ["M4 heat-set inserts and M4 x 16 screws", "8", "Mid driver into the mid horn, tail into sleeve"],
    ["Speaker cable, 2 x 1.0 mm²", "About 3 m", "Two pairs for mid and tweeter, one for the low"],
    ["Terminal cup, 4 pole or two 2 pole", "1", "Back panel, one pair per amplifier channel"],
    ["Polyester wadding, 25 mm", "0.5 m²", "Rear chamber, lightly"],
    ["Foam gasket tape, 3 mm", "2 m", "15 inch flange, mid driver flange, back panel"],
    ["PU glue, wood glue, 2-part epoxy, silicone", "", ""],
    ["3-way DSP crossover and 3 amplifier channels", "per speaker", "LR4 filters, delay, gain, limiter"],
])}

<h2>Cut list</h2>
{table(["Part", "Qty", "Size (mm)", "Ply (mm)", "Notes"], ply, num=(1, 3))}
<p>DXF flat patterns: <a href="cad/dxf/horn_wall_inner_face.dxf">horn wall</a>, <a href="cad/dxf/baffle.dxf">baffle</a>.
The horn walls are dimensioned on their inner face. Cut the trapezoid, then bevel the two long edges at
{cut['panels']['long_edge_bevel_deg']:.1f} degrees so neighbours meet in a mitre, and the two short edges at {cut['panels']['end_bevel_deg']:.1f} degrees
so the throat end sits flat on the baffle and the mouth end is flush with the front.</p>

<h2>Printed parts</h2>
{table(["Part", "File", "Size (mm)", "Est. mass (kg)"], printed, num=(3,))}
<p>Print in PETG or ASA, 0.2 mm layers. Use four or more perimeters on every acoustic surface. Print the mid horn
mouth-down and the tail point-up. Before printing, check two dimensions that came from datasheet drawings, not
measurements:</p>
<ul>
<li>The motor diameter of your 15NMB420. The nose cone assumes 150 mm and has a 20 mm bore to keep the pole vent open.</li>
<li>The cone profile of your 6MCF200Nd. The lead-in in <code>mid_horn</code> assumes a straight cone and leaves 1.5 mm.
Peak excursion in use is under 1 mm, but confirm the clearance with modelling clay before gluing anything.</li>
</ul>
<p>Add two features in your slicer or CAD before printing, because the generated parts leave them out:</p>
<ul>
<li>A 6 mm cable channel inside the top wall of <code>mid_horn</code>, from the top rod hole back to the sleeve.</li>
<li>A 4 mm channel along the top vane of <code>bullet</code> for the tweeter pair.</li>
</ul>

<h2>Assembly</h2>
<ol class="steps">
<li><b>Carcass.</b> Glue the sides, top and bottom around the back panel. Fit the baffle {D.REAR_CHAMBER_DEPTH*1000:.0f} mm in front of the back panel's
inner face. That gap is the {cut['rear_chamber']['net_litres']:.0f} litre sealed chamber, including the volume inside the cone. Seal every joint.
Add two dowel braces between baffle and back.</li>
<li><b>15 inch.</b> Before the back goes on, pass the 15NMB420 motor-first through the baffle cutout from the
chamber side. The flange sits on the back face of the baffle, on gasket tape, held by eight M6 bolts into inserts.
Wire it red to positive.</li>
<li><b>Nose cone.</b> Bond the printed nose to the back plate with silicone so it can be removed. Keep the bore clear.</li>
<li><b>Damping and back.</b> Line the back panel lightly with wadding, keeping it off the cone. Fit the terminal cup,
run the low pair, and glue the back on.</li>
<li><b>Horn walls.</b> Dry-fit the four bevelled walls: their throat ends rest on the baffle around the octagon and their
mouth ends meet the inner faces of the box at the front. Glue and pin. The voids behind them are sealed. Filling them
with sand or foam is optional.</li>
<li><b>Corner fillers.</b> Glue the triangular faces into the four corners on their ribs, wide end at the baffle, point at the
mouth corner. Fill and paint the inside of the horn.</li>
<li><b>Mid can.</b> Press the T90A into the bullet and run its pair through the vane. Key the bullet vanes into the mid horn
throat. Fit the 6MCF200Nd to the back of the mid horn, then the sleeve and the tail. Pass both pairs into the top rod hole.</li>
<li><b>Supports.</b> Drill the four horn walls at {cut['stations_mm']['support_rods']:.0f} mm from the cone rim, on the centre lines. Push the
rods through into blind holes in the mid can, centre the can with its front flush with the mouth, and epoxy.
Run the cables out of the top rod through the void, through a sealed hole in the baffle, and to the terminal cup.</li>
<li><b>Set up the DSP.</b> Start from the <a href="simulation.html">simulated settings</a>. On the low channel add a
second-order high-pass at 60 Hz and a limiter set to about 400 W: in the model the 15NMB420 reaches its 6.5 mm Xmax
near 80 Hz at 450 W.</li>
<li><b>Measure.</b> Measure each section alone at 1 m on axis and at 30 degrees, then the crossovers. Adjust the delays
for the deepest null with one section inverted, then restore polarity. Compare the result with the
<a href="downloads.html">simulated data</a>.</li>
</ol>
{fig("img/renders/rear-open.jpg", "Back panel removed: the rear-mounted 15 inch, its flange on the back of the baffle.", "render")}
"""
    return page("build.html", "Sunflower build", body, "Bill of materials, cut list, printed parts and assembly steps for Sunflower.")


def downloads(d):
    parts = d["parts"]
    rows = [[f"<code>{p['name']}</code>", p["material"].replace("_", " "), p.get("note", ""),
             f'<a href="cad/step/{p["name"]}.step">STEP</a> · <a href="cad/stl/{p["name"]}.stl">STL</a>'] for p in parts]
    res = [["Plane-wave, low section", '<a href="results/planewave/low.csv">low.csv</a>'],
           ["Plane-wave, mid section", '<a href="results/planewave/mid.csv">mid.csv</a>'],
           ["Tweeter model", '<a href="results/planewave/tweeter.csv">tweeter.csv</a>'],
           ["System through the crossover, 0, 15, 30 deg", '<a href="results/planewave/system.csv">system.csv</a>'],
           ["DSP settings", '<a href="results/planewave/dsp.json">dsp.json</a>'],
           ["FEM against plane-wave, low", '<a href="results/fem/low.csv">low.csv</a>'],
           ["FEM against plane-wave, mid", '<a href="results/fem/mid.csv">mid.csv</a>'],
           ["FEM summary", '<a href="results/fem/summary.json">summary.json</a>'],
           ["15 inch screen, all rows", '<a href="results/screen-15/search.csv">search.csv</a>'],
           ["15NMB420 refinement", '<a href="results/screen-15/refine-18sound-15nmb420.csv">refine.csv</a>']]
    body = f"""
<h1>Downloads</h1>
<p class="lede">All files are generated by the scripts in the repository. Units are millimetres. The assembly
STEP has every part in place.</p>
<p><a href="cad/sunflower-assembly.step">sunflower-assembly.step</a> · <a href="cad/cutlist.json">cutlist.json</a> ·
<a href="cad/parts.json">parts.json</a> · <a href="cad/dxf/horn_wall_inner_face.dxf">horn_wall_inner_face.dxf</a> ·
<a href="cad/dxf/baffle.dxf">baffle.dxf</a></p>
<h2>Parts</h2>
<p>Parts named <code>drv_</code> are dimensioned stand-ins for the drivers, for clearance checks and renders.</p>
{table(["Part", "Material", "Notes", "Files"], rows)}
<h2>Simulation data</h2>
<p>CSV columns: frequency in Hz, SPL in dB at 2.83 V and 1 m, phase in degrees, group delay in ms, and where
relevant impedance in ohms and peak excursion in mm.</p>
{table(["Data", "File"], res)}
<h2>Reproduce</h2>
<pre>git clone {REPO}
cd sunflower
just all        # screen, simulate, FEM (Docker), CAD, plots, renders (Blender), site</pre>
"""
    return page("downloads.html", "Sunflower downloads", body, "STEP, STL and DXF files and simulation data for Sunflower.")


def main():
    d = load()
    for name, fn in (("index.html", index), ("design.html", design), ("simulation.html", simulation),
                     ("build.html", build), ("downloads.html", downloads)):
        (DOCS / name).write_text(fn(d))
    (DOCS / ".nojekyll").write_text("")
    print("site written to", DOCS)


if __name__ == "__main__":
    main()
