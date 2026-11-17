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
        cutoffs=json.loads((R / "planewave/cutoffs.json").read_text()),
        system=pd.read_csv(R / "planewave/system.csv"),
        low=pd.read_csv(R / "planewave/low.csv"),
        mid=pd.read_csv(R / "planewave/mid.csv"),
        tweeter=pd.read_csv(R / "planewave/tweeter.csv"),
        mids=pd.read_csv(R / "optimise/mid-shortlist.csv"),
        midsearch_n=len(pd.read_csv(R / "optimise/mid-search.csv.gz", usecols=[0])),
        lowsearch_n=len(pd.read_csv(R / "optimise/low-search.csv.gz", usecols=[0])),
        sysdf=pd.read_csv(R / "optimise/system-search.csv"),
        summary=D.summary(),
    )
    s = data["system"]
    f = s.frequency_hz.values
    on = s.sum_0deg_spl_db.values
    ref = on[(f >= 200) & (f <= 2000)].mean()
    b = (f >= 100) & (f <= 16000)
    data["kpi"] = dict(sensitivity=float(data["dsp"]["target_db"]), ptp=float(np.ptp(on[b])),
                       f3=float(f[np.argmax(on > ref - 3)]), f6=float(f[np.argmax(on > ref - 6)]))
    return data


def section_band(df, lo, hi):
    b = (df.frequency_hz >= lo) & (df.frequency_hz <= hi)
    return float(np.ptp(df.spl_db[b])), float(df.spl_db[b].mean())


# ---------------------------------------------------------------- pages
def index(d):
    k, cut, dsp, sm = d["kpi"], d["cut"], d["dsp"], d["summary"]
    ext = cut["external_mm"]
    f1, f2 = D.CROSSOVER_LOW_MID_HZ, D.CROSSOVER_MID_HIGH_HZ
    m, tw = D.MID, D.TW
    body = f"""
<h1>Sunflower</h1>
<p class="lede">A three-way coaxial horn loudspeaker where each horn's phase plug is the next horn.
A reversed 15 inch driver drives a flat-panel horn. A {m.nominal.replace('in', ' inch')} midrange horn sits in its mouth and
forms the low horn's centre body. A {tw.model} bullet tweeter sits in the throat of the mid horn as its phase plug.
All three sections radiate from one axis and one mouth plane.</p>
<figure class="hero"><img src="img/renders/hero.jpg" alt="Rendered Sunflower cabinet, three-quarter front view"></figure>
<div class="stat">
<div><b>{k['sensitivity']:.0f} dB</b><span>2.83 V, 1 m</span></div>
<div><b>{k['f3']:.0f} Hz</b><span>low-frequency -3 dB, on axis</span></div>
<div><b>{k['ptp']:.1f} dB</b><span>peak to peak, 100 Hz to 16 kHz, on axis, no EQ</span></div>
<div><b>{ext['width']:.0f} x {ext['height']:.0f} x {ext['depth']:.0f}</b><span>mm, width x height x depth</span></div>
</div>
<div class="note"><p><b>Status.</b> Every number here is from simulation: plane-wave horn models coupled to
Thiele/Small driver models, checked with axisymmetric FEM, and the tweeter's published response. Every horn size
came out of an optimisation with explicit size limits (see <a href="design.html">Design</a>). Nothing has been
built or measured yet.</p></div>
{fig("img/renders/front-dims.jpg", "Straight into the mouth, with the main dimensions in millimetres. The corner fillers turn the square into an octagon at the throat, which gives the project its name.", "render")}

<h2>Three horns on one axis</h2>
<div class="grid3">
<div class="card"><h3><span class="swatch" style="background:var(--low)"></span>Low, up to {f1:.0f} Hz</h3>
<p>{sm['low']['driver']}, mounted backwards so its motor points into the horn. The motor and a printed nose
are the throat's phase plug. Its cone works into a {D.LOW_REAR_LITRES:.0f} litre sealed chamber. The horn is four flat
birch-ply panels and four corner fillers: {D.LOW.throat_width*1000:.0f} mm octagon to {D.LOW.mouth_width*1000:.0f} mm square over
{D.LOW.length*1000:.0f} mm.</p></div>
<div class="card"><h3><span class="swatch" style="background:var(--mid)"></span>Mid, {f1:.0f} Hz to {f2/1000:.1f} kHz</h3>
<p>{sm['mid']['driver']} ({m.nominal.replace('in', ' inch')}) into {'an' if m.profile[0] in 'aeiou' else 'a'} {m.profile} horn, {m.mouth_diameter*1000:.0f} mm mouth,
{m.length*1000:.0f} mm long, through an annular throat around the tweeter. It sits in a {m.can_diameter*1000:.0f} mm printed can on four
aluminium rods. To the low horn, the can is the centre body.</p></div>
<div class="card"><h3><span class="swatch" style="background:var(--tw)"></span>Tweeter, {f2/1000:.1f} kHz up</h3>
<p>{tw.model}: a ring-diaphragm horn tweeter with a bullet phase plug in its own mouth,
{tw.body_diameter*1000:.0f} mm across. Its body, in a 2 mm printed sleeve, fills the middle of the mid horn throat, so
the mid's sound leaves through the annulus around it.</p></div>
</div>
{fig("img/renders/section-side-dims.jpg", "Cutaway on the axis, mouth on the left, dimensions in millimetres.", "render")}
<div class="grid2">
{fig("img/renders/section.jpg", "Cutaway: sealed chamber, reversed 15 inch and nose, low horn, the mid can with its tail, mid horn and tweeter.", "render")}
{fig("img/renders/detail.jpg", "The mid horn mouth with the bullet tweeter in its throat.", "render")}
</div>

<h2>Specification</h2>
{table(["", "Value", "Notes"], [
    ["Type", "Three-way coaxial horn, active", "3-way DSP crossover and three amplifier channels"],
    ["Sensitivity", f"{k['sensitivity']:.1f} dB", "2.83 V, 1 m, set by the quietest section"],
    ["Frequency response", f"{k['f3']:.0f} Hz to 16 kHz", f"-3 dB on axis after DSP; -6 dB at {k['f6']:.0f} Hz"],
    ["Crossovers", f"{f1:.0f} Hz and {f2/1000:.1f} kHz", "Linkwitz-Riley 24 dB/oct"],
    ["Drivers", f"{sm['low']['driver']}, {sm['mid']['driver']}, {tw.model}", "All 8 ohm"],
    ["Cabinet", f"{ext['width']:.0f} x {ext['height']:.0f} x {ext['depth']:.0f} mm", "18 mm birch ply, 25 mm baffle"],
    ["Low horn", f"{D.LOW.throat_width*1000:.0f} mm octagon to {D.LOW.mouth_width*1000:.0f} mm square, {D.LOW.length*1000:.0f} mm", f"Flat panels, {cut['panels']['flare_half_angle_deg']:.1f} degree flare"],
    ["Mid horn", f"{m.profile.capitalize()}, {m.outer_throat_radius*2000:.0f} to {m.mouth_diameter*1000:.0f} mm, {m.length*1000:.0f} mm", f"Annular throat, {m.compression_ratio:.1f}:1 compression"],
])}
<figure class="figure"><img class="render" src="img/renders/turntable.gif" alt="Turntable animation of the cabinet"><figcaption>Turntable, rendered from the CAD in Blender.</figcaption></figure>
"""
    return page("index.html", "Sunflower", body, "A three-way nested coaxial horn loudspeaker with a reversed 15 inch, a mid horn and a bullet tweeter.")


def design(d):
    lo, m, tw, sel = D.LOW, D.MID, D.TW, D.SELECTED
    sysdf, mids = d["sysdf"], d["mids"].set_index("key")
    sysdf = sysdf.assign(tweeter=mids.loc[sysdf.mid_key, "tweeter"].values,
                         mid_label=mids.loc[sysdf.mid_key, "label"].values,
                         mid_mouth=mids.loc[sysdf.mid_key, "mouth"].values,
                         f2=mids.loc[sysdf.mid_key, "f2"].values, f1=mids.loc[sysdf.mid_key, "f1"].values)
    from sunflower import tweeters as TWS
    rows = []
    for tid, g in sysdf.groupby("tweeter"):
        r = g.sort_values("score").iloc[0]
        rows.append([TWS.load(tid).model, f"{r.f1:.0f} / {r.f2:.0f}", f"{r.mid_label}, {r.mid_mouth*1000:.0f} mm mouth",
                     f"{r.width*1000:.0f} x {r.depth*1000:.0f}", f"{r.rms_on_db:.2f}", f"{r.spread_30_db:.2f}", f"{r.f3_hz:.0f}", f"{r.score:.2f}"])
    feas = sysdf[(sysdf.width <= 0.85) & (sysdf.depth <= 1.2)].sort_values("volume_m3")
    size_rows = []
    for label, g in (("Smallest", feas.head(1)), ("Chosen", feas[np.isclose(feas.score, sel["scores"]["score"])].head(1)),
                     ("Best score", feas.sort_values("score").head(1))):
        if len(g):
            r = g.iloc[0]
            size_rows.append([label, f"{r.width*1000:.0f} x {r.depth*1000:.0f}", f"{r.volume_m3*1000:.0f}",
                              f"{r.score:.2f}", f"{r.f3_hz:.0f}", f"{r.rms_on_db:.2f}"])
    body = f"""
<h1>Design</h1>
<p class="lede">Every horn size, both crossover points, all three drivers and the tweeter were chosen by
simulation. This page covers the nesting, how the optimiser searched, and the trade-off between size and
performance.</p>
{fig("img/plots/low-drawing.png", "Cabinet section, mm. Wood: cabinet and low horn walls. Black: the low horn centre body (15 inch motor and nose, then the mid can). Yellow: the tweeter sleeve, which is the mid phase plug.")}

<h2>Nesting</h2>
<ul>
<li><b>Low.</b> The 15 inch is mounted backwards from behind the baffle. Its convex cone and motor face the horn,
and the motor plus a printed nose is the start of the centre body. The low horn's area is the space between the
walls and that centre body, slice by slice.</li>
<li><b>Mid.</b> The mid can is the rest of the centre body. Its elliptical tail starts {(lo.tail_start - lo.nose_end)*1000:.0f} mm in front of the
15 inch nose, so the open area narrows smoothly onto the can.</li>
<li><b>Tweeter.</b> The {tw.model} body is the mid horn's phase plug. The mid throat is the annulus between the
{m.plug_radius*2000:.0f} mm sleeve and a {m.outer_throat_radius*2000:.0f} mm bore, {m.throat_area*1e4:.0f} cm² open, {m.compression_ratio:.1f}:1
compression for the mid cone.</li>
</ul>

<h2>How the sizes were chosen</h2>
<p>The optimiser in <code>src/sunflower/optimise.py</code> works in three stages. Each candidate is a full
plane-wave solve of the horn with its centre body, coupled to the driver's Thiele/Small model.</p>
<ol>
<li><b>Mid and tweeter.</b> Three bullet tweeters, each with the lowest crossover its maker allows. Mid horns in four
profiles (conical, exponential, hyperbolic-exponential, oblate spheroidal), five throats, six mouths from 190 to
460 mm, seven lengths from 120 to 500 mm, and two tweeter depths. Every horn was coupled to every 6, 6.5, 8 and 10 inch
cone in the catalogue that is trusted to the upper crossover; the best four drivers per horn were kept,
{d['midsearch_n']:,} rows. Two rules apply before scoring. The mid mouth must be at least c / (pi f) across at the lower
crossover f, so the horn still loads its driver there: 218 mm at 500 Hz, 273 mm at 400 Hz. Horns flaring wider than
35 degrees were excluded because the plane-wave model is not reliable there. Score: in-band variation, coverage
mismatch with the tweeter at the upper crossover, and an output floor.</li>
<li><b>Low.</b> The best mid of each can size was placed in 128 flat-panel low horns (two throats, four mouths,
four lengths, four sealed volumes), each coupled to every 15 inch driver with at least 4 mm Xmax that is not an
instrument speaker (bass and guitar cabinet drivers are designed to break up): {d['lowsearch_n']:,} combinations passed the
output and excursion limits. Score: variation up to 1.4 times the lower crossover, coverage match with the mid
there, and bass extension.</li>
<li><b>System.</b> The best 20 combinations for each tweeter, {len(sysdf)} in all, went through the full crossover with
gains, delays and polarity tuned for each. Score: on-axis RMS deviation from 100 Hz to 16 kHz; plus half the roughness
of the 30 degree response relative to on axis, measured after removing a straight-line trend so that steady narrowing is
allowed but steps at the crossovers are not; plus a penalty for a -3 dB point above 70 Hz.</li>
</ol>

<h3>Best design with each tweeter</h3>
{table(["Tweeter", "Crossovers (Hz)", "Mid", "Cabinet W x D (mm)", "On-axis RMS (dB)", "30 deg roughness (dB)", "-3 dB (Hz)", "Score"], rows, num=(4, 5, 6, 7))}
<p>{sel['rule'].capitalize()}. That picked the {tw.model}, crossing at {D.CROSSOVER_LOW_MID_HZ:.0f} Hz and
{D.CROSSOVER_MID_HIGH_HZ:.0f} Hz.</p>

<h3>Size against performance</h3>
{fig("img/plots/tradeoff.png", "Every finalist. Dotted line: the largest cabinet inside the 850 mm wide by 1200 mm deep limit.")}
{table(["", "W x D (mm)", "Volume (L)", "Score", "-3 dB (Hz)", "On-axis RMS (dB)"], size_rows, num=(2, 3, 4, 5))}

<h2>The chosen parts</h2>
{table(["", "Choice", "Why"], [
    ["Tweeter", tw.model, f"Body {tw.body_diameter*1000:.0f} x {tw.body_depth*1000:.0f} mm, {tw.sensitivity_db:.0f} dB/W/m, crossover from {tw.crossover_min_hz/1000:.1f} kHz. <a href=\"{tw.source}\">Datasheet</a>"],
    ["Mid driver", d['summary']['mid']['driver'], f"{m.nominal.replace('in', ' inch')}, best score for this tweeter and can size"],
    ["Mid horn", f"{m.profile}, throat {m.outer_throat_radius*2000:.0f} mm, mouth {m.mouth_diameter*1000:.0f} mm, {m.length*1000:.0f} mm long", f"Tweeter face {m.tweeter_face*1000:.0f} mm in front of the throat"],
    ["Low driver", d['summary']['low']['driver'], "Best low score with this mid can"],
    ["Low horn", f"{lo.throat_width*1000:.0f} to {lo.mouth_width*1000:.0f} mm, {lo.length*1000:.0f} mm long", f"{D.LOW_REAR_LITRES:.0f} litre sealed chamber"],
])}
<p>Catalogue values for drivers other than the 6.5 inch set are legacy and unverified. The chosen drivers were
checked by hand against their datasheets; the records in <code>data/drivers/verified/</code> say what was checked.</p>
"""
    return page("design.html", "Sunflower design", body, "How Sunflower's nested horns were sized: the optimisation, the tweeter choice and the size trade-off.")


def horn_section(name, d, title, intro):
    figs = [f"img/plots/{name}-drawing.png", f"img/plots/{name}-area.png", f"img/plots/{name}-magnitude.png",
            f"img/plots/{name}-phase.png", f"img/plots/{name}-polar.png", f"img/plots/{name}-electrical.png"]
    caps = ["Dimensioned section, mm.", "Open area along the horn.", "Magnitude on axis, unfiltered, with its cutoffs.",
            "Phase and group delay.", "Off-axis responses and polar map.", "Impedance and cone excursion."]
    blocks = "".join(fig(f, c) for f, c in zip(figs, caps) if (DOCS / f).exists())
    return f'<h2 id="{name}">{title}</h2>\n{intro}\n{blocks}'


def simulation(d):
    dsp, fem, co = d["dsp"], d["fem"], d["cutoffs"]
    f1, f2 = D.CROSSOVER_LOW_MID_HZ, D.CROSSOVER_MID_HIGH_HZ
    lr, lm = section_band(d["low"], 100, f1)
    mr, mm = section_band(d["mid"], f1, f2)
    tr, tm = section_band(d["tweeter"], f2, 16000)
    sm = d["summary"]
    m, lo, tw = D.MID, D.LOW, D.TW
    fmt = lambda v: f"{v:.0f} Hz" if v else "outside the plotted range"  # noqa: E731
    low_intro = f"""<p>{sm['low']['driver']} in the flat-panel low horn: {lo.throat_width*1000:.0f} mm octagon to
{lo.mouth_width*1000:.0f} mm square over {lo.length*1000:.0f} mm, {D.LOW_REAR_LITRES:.0f} litre sealed chamber. From 100 Hz to {f1:.0f} Hz:
{lr:.1f} dB variation at {lm:.1f} dB mean. Lower -3 dB at {fmt(co['low']['f_low_3db'])}; the mouth's own cutoff, where its
circumference equals the wavelength, is {co['low']['mouth_cutoff_hz']:.0f} Hz. The air between the cone and the throat gives a
high-frequency corner near {co['low']['front_chamber_corner_hz']:.0f} Hz. FEM agrees within {fem['low']['max_abs_difference_db']:.1f} dB in band.</p>"""
    mid_intro = f"""<p>{sm['mid']['driver']} in {'an' if m.profile[0] in 'aeiou' else 'a'} {m.profile} horn: {m.outer_throat_radius*2000:.0f} mm annular throat around
the {m.plug_radius*2000:.0f} mm tweeter sleeve, {m.mouth_diameter*1000:.0f} mm mouth, {m.length*1000:.0f} mm long. From {f1:.0f} Hz to
{f2:.0f} Hz: {mr:.1f} dB variation at {mm:.1f} dB mean. Lower -3 dB at {fmt(co['mid']['f_low_3db'])}, upper -3 dB at
{fmt(co['mid']['f_high_3db'])}, mouth cutoff {co['mid']['mouth_cutoff_hz']:.0f} Hz. FEM agrees within
{fem['mid']['max_abs_difference_db']:.1f} dB in band, mean difference {fem['mid']['mean_difference_db']:+.2f} dB.</p>"""
    tw_intro = f"""<p>{tw.model}, not simulated: the curves are its published on-axis and off-axis response, read from the
<a href="{tw.source}">datasheet</a> graph to about 2 dB, with minimum phase reconstructed from the magnitude. Its own
horn cuts off at {tw.cutoff_hz:.0f} Hz. From {f2:.0f} Hz to 16 kHz: {tr:.1f} dB variation at {tm:.1f} dB. The tweeter's
diaphragm sits {(m.length - m.tweeter_face + tw.horn_depth)*1000:.0f} mm behind the mouth plane, which the system sum includes as a
delay. The mid horn's effect on the tweeter's coverage is not modelled.</p>"""
    body = f"""
<h1>Simulation</h1>
<p class="lede">Each horn on its own first, with all its plots, then the full-range system with the cutoff
frequencies marked. All curves are at 2.83 V and 1 m.</p>
<p><a href="#low">Low horn</a> · <a href="#mid">Mid horn</a> · <a href="#tweeter">Tweeter</a> ·
<a href="#full-range">Full range</a> · <a href="#method">Method</a> · <a href="#limits">Limits</a></p>
{horn_section("low", d, "Low horn", low_intro)}
{horn_section("mid", d, "Mid horn", mid_intro)}
{horn_section("tweeter", d, "Tweeter", tw_intro)}

<h2 id="full-range">Full range</h2>
{fig("img/plots/full-range.png", "Top: each section unfiltered and level-matched, with the frequencies that limit it. Bottom: through the crossover, on axis and at 30 degrees.")}
{table(["Section", "Lower -3 dB", "Upper -3 dB", "Other limit"], [
    ["Low", fmt(co['low']['f_low_3db']), fmt(co['low']['f_high_3db']), f"Mouth cutoff {co['low']['mouth_cutoff_hz']:.0f} Hz"],
    ["Mid", fmt(co['mid']['f_low_3db']), fmt(co['mid']['f_high_3db']), f"Mouth cutoff {co['mid']['mouth_cutoff_hz']:.0f} Hz"],
    ["Tweeter", fmt(co['tweeter']['f_low_3db']), fmt(co['tweeter']['f_high_3db']), f"Horn cutoff {tw.cutoff_hz:.0f} Hz (datasheet)"],
    ["Crossovers", f"{f1:.0f} Hz", f"{f2:.0f} Hz", "LR4"],
])}
<p>Through the crossover the on-axis sum stays within {d['kpi']['ptp']:.1f} dB peak to peak from 100 Hz to 16 kHz
(RMS deviation {dsp['rms_deviation_100_16k_db']:.2f} dB), before any equalisation.</p>
{fig("img/plots/full-range-group-delay.png", "Group delay of the summed response.")}
<h3>DSP settings</h3>
{table(["Channel", "Filter", "Gain", "Delay", "Polarity in the model"], [
    ["Low", f"LR4 low-pass {f1:.0f} Hz", f"{dsp['gain_db']['low']:+.1f} dB", f"{dsp['delay_ms']['low']:.2f} ms", "Inverted" if dsp['polarity']['low'] < 0 else "Normal"],
    ["Mid", f"LR4 high-pass {f1:.0f} Hz, LR4 low-pass {f2:.0f} Hz", f"{dsp['gain_db']['mid']:+.1f} dB", f"{dsp['delay_ms']['mid']:.2f} ms", "Normal"],
    ["Tweeter", f"LR4 high-pass {f2:.0f} Hz", f"{dsp['gain_db']['tweeter']:+.1f} dB", f"{dsp['delay_ms']['tweeter']:.2f} ms", "Inverted" if dsp['polarity']['tweeter'] < 0 else "Normal"],
], num=(2, 3))}
<div class="note"><p>In the model, positive voltage pushes air out of every horn. The reversed 15 inch does the
opposite when wired red to positive: it pulls air out of the horn. So where the table says the low section is
inverted, wire it normally and leave the DSP channel at normal polarity; where it says normal, invert it. Confirm with
a measurement at the crossover.</p></div>

<h2 id="method">Method</h2>
<ol>
<li><b>Plane-wave horn model.</b> Each horn is cut into a few hundred thin annular slices and cascaded as
transfer matrices, with a flanged-piston radiation load at the mouth. Centre bodies are subtracted slice by slice.</li>
<li><b>Driver model.</b> Thiele/Small motor with its sealed rear volume and the front chamber between cone and throat.</li>
<li><b>Radiation.</b> The far field includes the curved wavefront at a flared mouth, which matters off axis.</li>
<li><b>FEM check.</b> Mid and low horns solved again with an axisymmetric Helmholtz finite-element model (dolfinx 0.8)
in a baffled half-space, coupled to the same driver models. The FEM code matches the exact straight-tube solution to a
median {fem['tube']['median']*100:.0f}% in throat impedance.</li>
</ol>
{fig("img/plots/fem-vs-planewave.png", "FEM against plane-wave. Solid on axis, dotted 30 degrees.")}

<h2 id="limits">What the models leave out</h2>
<ul>
<li>No cross modes in the plane-wave model. The FEM is axisymmetric: it replaces the octagon and square of the low horn
with circles of equal area.</li>
<li>Cone breakup, wall losses, distortion and power compression. Cone sizes are only trusted to a crossover of
7 kHz for 6 inch, 5 kHz for 8 inch and 3.6 kHz for 10 inch.</li>
<li>The cabinet front is an infinite baffle at the mouth plane. A real front face shows baffle step and edge
diffraction in the low and mid bands.</li>
<li>The tweeter is its datasheet response. How the mid horn around it changes its coverage is not modelled.</li>
<li>Support rods, vanes and cable channels block about 2% of each passage and are not in the acoustic models.</li>
<li>Motor and cone dimensions of the drivers not on a datasheet drawing are estimates; measure before printing.</li>
</ul>
"""
    return page("simulation.html", "Sunflower simulation", body, "Each Sunflower horn section with all its plots, and the full-range response with its cutoff frequencies.")


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
        ["Baffle", "1", f"{wm:.0f} x {wm:.0f}", "25", f"{2*D.LOW.cutout_radius*1000:.0f} mm hole, 8 x M6 inserts (baffle.dxf)"],
        ["Horn walls", "4", f"Trapezoid {D.LOW.throat_width*1000:.0f} to {wm:.0f} wide, {cut['panels']['slope_length_mm']:.0f} long (inner face)", "18",
         f"Long edges bevelled {cut['panels']['long_edge_bevel_deg']:.1f} deg, ends {cut['panels']['end_bevel_deg']:.1f} deg (horn_wall_inner_face.dxf)"],
        ["Corner filler faces", "4", f"Triangle, base {cut['corner_filler']['face_base_mm']:.0f}, height {cut['corner_filler']['face_height_mm']:.0f}", "12", "Three ribs each behind the face"],
    ]
    dens = 1.27
    printed = []
    for n, label, fill in (("mid_horn", "Mid horn and cone lead-in", 0.45), ("mid_sleeve", "Mid driver sleeve", 1.0),
                           ("mid_tail", "Mid rear cap and tail", 0.25), ("phase_plug", "Tweeter sleeve, nose and vanes", 1.0),
                           ("low_nose", "15 inch nose cone", 0.3)):
        p = P[n]
        bb = p["bbox_mm"]
        size = f"{bb[3]-bb[0]:.0f} x {bb[4]-bb[1]:.0f} x {bb[5]-bb[2]:.0f}"
        printed.append([label, f"<code>{n}.stl</code>", size, f"{p['volume_cm3']*dens*fill/1000:.2f}"])
    rod = P["rod_right"]["note"].split(", ")[-1]
    sm = d["summary"]
    body = f"""
<h1>Build</h1>
<p class="lede">The cabinet is flat birch-ply panels. The round parts are 3D printed. Everything here comes from the
CAD in <code>cad/</code>, which is generated from the same design file as the simulations.</p>
{fig("img/plots/low-drawing.png", "Cabinet section with dimensions, mm.")}
{fig("img/plots/mid-drawing.png", "Mid horn section with dimensions, mm.")}
{fig("img/renders/exploded.jpg", "Exploded view. Cabinet panels move outwards, the horn walls and corner fillers inwards, the mid can and tweeter forwards.", "render")}

<h2>Bill of materials</h2>
{table(["Item", "Qty", "Notes"], [
    [f"{sm['low']['driver']}, 8 ohm", "1", "Low. Rear-mounted, motor forward"],
    [f"{sm['mid']['driver']}, 8 ohm", "1", f"Mid, {D.MID.nominal.replace('in', ' inch')}"],
    [f"{D.TW.model}, 8 ohm", "1", "Bullet tweeter"],
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
<li>The motor diameter of your {sm['low']['driver']}. The nose cone assumes {D.LOW.body.motor_radius*2000:.0f} mm and has a 20 mm bore to keep the pole vent open.</li>
<li>The cone profile of your {sm['mid']['driver']}. The lead-in in <code>mid_horn</code> assumes a straight cone and leaves 1.5 mm.
Confirm the clearance with modelling clay before gluing anything.</li>
<li>Your {D.TW.model}'s body. The sleeve in <code>phase_plug</code> is bored to {D.TW.body_diameter*1000 + 1:.0f} mm for a {D.TW.body_diameter*1000:.0f} mm body.</li>
</ul>
<p>Add two features in your slicer or CAD before printing, because the generated parts leave them out:</p>
<ul>
<li>A 6 mm cable channel inside the top wall of <code>mid_horn</code>, from the top rod hole back to the sleeve.</li>
<li>A 4 mm channel along the top vane of <code>phase_plug</code> for the tweeter pair.</li>
</ul>

<h2>Assembly</h2>
<ol class="steps">
<li><b>Carcass.</b> Glue the sides, top and bottom around the back panel. Fit the baffle {D.REAR_CHAMBER_DEPTH*1000:.0f} mm in front of the back panel's
inner face. That gap is the {cut['rear_chamber']['net_litres']:.0f} litre sealed chamber, including the volume inside the cone. Seal every joint.
Add two dowel braces between baffle and back.</li>
<li><b>15 inch.</b> Before the back goes on, pass the 15 inch motor-first through the baffle cutout from the
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
<li><b>Mid can.</b> Slide the {D.TW.model} into the phase-plug sleeve, face flush with its front, and run its pair through the
top vane. Key the three vanes into the mid horn throat: the tweeter face ends up {D.MID.tweeter_face*1000:.0f} mm in front of the throat.
Fit the mid driver to the back of the mid horn, then the sleeve and the tail. Pass both pairs into the top rod hole.</li>
<li><b>Supports.</b> Drill the four horn walls at {cut['stations_mm']['support_rods']:.0f} mm from the cone rim, on the centre lines. Push the
rods through into blind holes in the mid can, centre the can with its front flush with the mouth, and epoxy.
Run the cables out of the top rod through the void, through a sealed hole in the baffle, and to the terminal cup.</li>
<li><b>Set up the DSP.</b> Start from the <a href="simulation.html#full-range">simulated settings</a>. On the low channel add a
second-order high-pass at 60 Hz and a limiter set from the excursion plot in the <a href="simulation.html#low">low horn section</a>.</li>
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
           ["Section cutoffs", '<a href="results/planewave/cutoffs.json">cutoffs.json</a>'],
           ["Optimiser: mid search (kept rows)", '<a href="results/optimise/mid-search.csv.gz">mid-search.csv.gz</a>'],
           ["Optimiser: mid shortlist", '<a href="results/optimise/mid-shortlist.csv">mid-shortlist.csv</a>'],
           ["Optimiser: low search", '<a href="results/optimise/low-search.csv.gz">low-search.csv.gz</a>'],
           ["Optimiser: system finalists", '<a href="results/optimise/system-search.csv">system-search.csv</a>'],
           ["Selected design", '<a href="results/optimise/selected.json">selected.json</a>']]
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
