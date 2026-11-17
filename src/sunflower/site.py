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

PAGES = [("index.html", "Product"), ("review.html", "v0.2 review"), ("design.html", "Design"), ("simulation.html", "Simulation"),
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
        stack=json.loads((ROOT / "cad/stack.json").read_text()),
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


def an(word):
    return "an" if word[0] in "aeiou" else "a"


def inch(nominal):
    return nominal.replace("in", " inch")


# ---------------------------------------------------------------- pages
def index(d):
    k, cut, sm, st = d["kpi"], d["cut"], d["summary"], d["stack"]
    ex = cut["external_mm"]
    f1, f2 = D.CROSSOVER_LOW_MID_HZ, D.CROSSOVER_MID_HIGH_HZ
    m, lo, tw = D.MID, D.LOW, D.TW
    shelf = d["dsp"].get("no_sub_shelf", {})
    body = f"""
<h1>Sunflower</h1>
<p class="lede">A three-way coaxial horn where each horn's phase plug is the next horn. A {tw.model} on a small
printed horn sits in the middle of {an(law_name(m.law))} {law_name(m.law)} mid horn and is its phase plug. The mid horn and its
{inch(m.nominal)} driver sit in a can that is the end of a one-piece centre body, which is the phase plug of a reversed
15 inch kick horn. It follows the Funktion-One Res layout (kick, wide-band mid, small HF) on one axis.</p>
<figure class="hero"><img src="img/renders/hero.jpg" alt="Rendered Sunflower cabinet"></figure>
<div class="stat">
<div><b>{k['sensitivity']:.0f} dB</b><span>2.83 V, 1 m</span></div>
<div><b>{k['f3']:.0f} Hz</b><span>low -3 dB; pair with a sub</span></div>
<div><b>{f1:.0f} Hz / {f2/1000:.1f} kHz</b><span>crossovers: kick, wide-band mid, HF</span></div>
<div><b>{ex['width']:.0f} x {ex['height_front']:.0f} x {ex['depth']:.0f}</b><span>mm, width x height x depth</span></div>
</div>
<div class="note"><p><b>Status.</b> Version 0.2, simulated only. What changed from v0.1 and what is still missing is on the
<a href="review.html">v0.2 review</a> page.</p></div>
<h2>Stacks</h2>
<p>The top and bottom of the cabinet are angled, so boxes stack into a splayed column: {st['splay_deg']:.1f} degrees
between neighbours.</p>
{fig("img/renders/stack.jpg", "Three cabinets splayed in a vertical stack.", "render")}
<div class="grid2">
{fig("img/renders/front-dims.jpg", "Into the mouth, mm. The corner fillers turn the rectangle into an octagon at the throat.", "render")}
{fig("img/renders/stack-side-dims.jpg", "The stack from the side.", "render")}
</div>

<h2>Three horns, one axis, designed from the inside out</h2>
<div class="grid3">
<div class="card"><h3><span class="swatch" style="background:var(--tw)"></span>HF, {f2/1000:.1f} kHz up</h3>
<p>{tw.model} ({tw.kind}), {tw.body_diameter*1000:.0f} mm body cylinder. A printed horn, {2*m.hf_mouth_radius*1000:.0f} mm at
the mouth, runs to just inside the mid mouth. Its outside is the tapering tip of the mid phase plug.</p></div>
<div class="card"><h3><span class="swatch" style="background:var(--mid)"></span>Mid, {f1:.0f} Hz to {f2/1000:.1f} kHz</h3>
<p>{sm['mid']['driver']} ({inch(m.nominal)}). Its cone is at least 1.6 times the {2*m.plug_radius*1000:.0f} mm plug.
{law_name(m.law).capitalize()} open-area law from a {m.compression:.1f}:1 annular throat to a {m.mouth_diameter*1000:.0f} mm mouth over
{m.length*1000:.0f} mm, with the plug always subtracted.</p></div>
<div class="card"><h3><span class="swatch" style="background:var(--low)"></span>Kick, up to {f1:.0f} Hz</h3>
<p>{sm['low']['driver']}, mounted backwards. The one-piece centre body's rear face follows the cone, so the throat is a
narrow {lo.compression:.1f}:1 annulus. A tube-like kick horn, {lo.length*1000:.0f} mm long, {lo.mouth_width*1000:.0f} x
{lo.mouth_height*1000:.0f} mm mouth.</p></div>
</div>
{fig("img/renders/section-side-dims.jpg", "Cutaway on the axis, mouth on the left, mm.", "render")}
<div class="grid2">
{fig("img/renders/section.jpg", "Cutaway: chamber, reversed 15 inch, centre body, mid can, phase plug and HF horn.", "render")}
{fig("img/renders/detail.jpg", "The mid mouth with the HF horn at its centre.", "render")}
</div>

<h2>Specification</h2>
{table(["", "Value", "Notes"], [
    ["Type", "Three-way coaxial horn, active", "Kick bin with wide-band mid and HF; use with a sub"],
    ["Sensitivity", f"{k['sensitivity']:.1f} dB", "2.83 V, 1 m"],
    ["Response", f"{k['f3']:.0f} Hz to 16 kHz, -3 dB", f"On axis after DSP. No-sub shelf: -6 dB at {shelf.get('f_minus6_hz', float('nan')):.0f} Hz, up to about {shelf.get('safe_power_w', float('nan')):.0f} W"],
    ["Crossovers", f"{f1:.0f} Hz and {f2/1000:.1f} kHz", "Linkwitz-Riley 24 dB/oct"],
    ["Drivers", f"{sm['low']['driver']}, {sm['mid']['driver']}, {tw.model}", ""],
    ["Cabinet", f"{ex['width']:.0f} wide, {ex['height_front']:.0f} high at the front, {ex['height_back']:.0f} at the back, {ex['depth']:.0f} deep", f"18 mm birch ply; {st['splay_deg']:.1f} degree splay"],
    ["HF unit price", f"${tw.price_usd:.0f}" if tw.price_usd else "", "Retail, see the Design page"],
])}
<figure class="figure"><img class="render" src="img/renders/turntable.gif" alt="Turntable animation"><figcaption>Turntable, rendered in Blender from the CAD.</figcaption></figure>
"""
    return page("index.html", "Sunflower", body, "A three-way nested coaxial horn: kick, wide-band mid and HF on one axis.")


GAPS = [
    ("Design order", "Tweeter, mid and low searched on separate grids. The PRV cone was barely wider than the 86 mm plug",
     "Inside out: HF body cylinder, then plug, then mid cones at least 1.6 times the plug, then throat, mouth, can, centre body, kick horn"),
    ("Mid phase plug", "A cylinder that stopped dead at the tweeter face: a step in open area",
     "The plug tapers to the HF horn mouth and the wall follows the area law, so open area has no step"),
    ("HF unit", "Fostex T925A super tweeter, $453, read by its outside envelope",
     "Five HF units compared, including 1 inch compression drivers on our own printed horn (Funktion-One style), sized by body cylinder only, with prices"),
    ("HF in the mid horn", "Not modelled", "Axisymmetric FEM of the HF mouth inside the mid horn. It showed a deep notch when the mouth was deep inside, so the HF horn now runs to 10 mm inside the mid mouth"),
    ("Low centre body", "15 inch nose and mid-can tail separate, with a gap between", "One piece from the cone to the mid can"),
    ("Low throat", "No compression (throat bigger than the cone)", "The centre body's rear face follows the cone, giving a narrow compressed annulus"),
    ("Low front chamber", "8.4 litres, roll-off near 660 Hz", "Cone area times (Xmax + 3 mm), well under a litre"),
    ("Depth", "982 mm", "At most 600 mm"),
    ("Cabinet", "Square box with dead voids", "Angled top and bottom, voids are the sealed chamber, splayed stacking"),
    ("Band split", "Kick to 400 Hz, mid to 6 kHz", "Crossovers chosen by the optimiser from 250 to 500 Hz and the HF unit's range, rewarding a wide mid band"),
    ("Print sizes", "302 mm can, larger than common beds", "Every printed part cut into sections and quarters within 250 mm"),
    ("Cost", "None", "HF price is the tie-breaker among near-equal designs; prices listed"),
]
REMAINING = [
    "Nothing is built or measured.",
    "HF responses are datasheet curves on the maker's horn, read by eye; the printed HF horn changes them. Only its coverage is modelled (FEM).",
    "Low and mid FEM are axisymmetric: the rectangular kick horn becomes a circle of equal area.",
    "Thiele/Small data for most catalogue drivers are unverified; the chosen ones are checked by hand.",
    "The 15 inch cone profile and motor size, and the mid cone profile, are estimated. Measure before printing the centre body and plug.",
    "Mid and low driver prices are not in the cost score; only the HF unit's.",
    "Cone breakup is not modelled; cone sizes are only trusted to fixed crossover limits.",
]


def review(d):
    body = f"""
<h1>v0.2 review</h1>
<p class="lede">v0.1 was reviewed against the design brief. v0.2 closes these gaps. The v0.1 site and files are kept at
the <a href="{REPO}/tree/v0.1">v0.1 tag</a>.</p>
<h2>Gaps and what v0.2 does about them</h2>
{table(["Area", "v0.1", "v0.2"], GAPS)}
<h2>References</h2>
{table(["Reference", "Layout", "Use here"], [
    ["Funktion-One Res 2", "15 inch kick horn to 250 Hz, 8 inch mid with Axhead from 250 Hz up, 1 inch HF. 102 dB bass, 106 dB mid/HF, 60 Hz to 18 kHz", "Overall band split"],
    ["Funktion-One Res 4T", "12 inch 114 to 445 Hz, 8 inch 445 Hz to 5.77 kHz, 1 inch compression driver 5.55 to 18 kHz", "HF type and crossover region"],
    ["HD-15 kick bin", "Front-loaded 15 inch straight horn, about 578 x 578 x 472 mm outside, useful from about 90 Hz", "Depth target and kick-horn character"],
])}
<p>Funktion-One figures are from <a href="https://funktion-one.com/product/res-2">funktion-one.com</a>. HD-15 figures
are from a <a href="https://www.diyaudio.com/community/posts/7749785">diyAudio thread</a>; the plans themselves are on
Speakerplans and are not reproduced here.</p>
<h2>Still open</h2>
<ul>{"".join(f"<li>{r}</li>" for r in REMAINING)}</ul>
"""
    return page("review.html", "Sunflower v0.2 review", body, "Gap analysis of Sunflower v0.1 and what v0.2 changed.")


def design(d):
    lo, m, tw, sel = D.LOW, D.MID, D.TW, D.SELECTED
    sysdf = d["sysdf"]
    from sunflower import tweeters as TWS
    hf_rows = []
    for tid in TWS.all_ids(include_excluded=True):
        t = TWS.load(tid)
        g = sysdf[sysdf.tweeter == tid]
        best = g.sort_values("score").iloc[0] if len(g) else None
        excluded = json.loads((ROOT / "data/drivers/excluded.json").read_text())["drivers"].get(tid)
        hf_rows.append([t.model, t.kind, f"{t.body_diameter*1000:.0f}", f"{t.crossover_min_hz/1000:.1f}",
                        f"${t.price_usd:.0f}" if t.price_usd else "none found",
                        f"{best.score:.2f}" if best is not None else ("excluded: " + excluded if excluded else "no feasible design")])
    chain = [[n, f"{v:.0f}", note] for n, v, note in D.chain()]
    body = f"""
<h1>Design</h1>
<p class="lede">Every size comes from the one inside it. The optimiser searched that chain for every HF unit, mid
driver and kick driver in the catalogue, within a 600 mm depth limit.</p>
{fig("img/plots/chain.png", "The design chain, from the HF body cylinder out to the kick horn mouth.")}
{table(["Step", "mm (cm² for areas)", "Set by"], chain, num=(1,))}
{fig("img/plots/low-drawing.png", "Cabinet section, mm. Shaded: sealed chamber. Black: the one-piece centre body. Yellow: the mid phase plug.")}

<h2>Phase plugs shape the expansion</h2>
<p>Each horn is designed as an open-area law. The inner body is always subtracted. In the mid, the plug is fixed by the
HF unit (its body plus a 2 mm sleeve) and tapers to the HF horn mouth; the outer wall is solved so that wall minus plug
follows the law. In the kick horn the walls are flat panels, so the centre body is solved instead: its radius is whatever
leaves the open area on the law, from the throat to the mid can.</p>
<div class="grid2">
{fig("img/plots/mid-area.png", "Mid: wall, plug and open area.")}
{fig("img/plots/low-area.png", "Kick: walls, centre body and open area.")}
</div>

<h2>HF units</h2>
<p>Bullet super tweeters and 1 inch compression drivers, sized by their body cylinder only (flanges are cut off or
ignored). A compression driver gets a small printed horn whose outside is the plug's tapering tip, the way Funktion-One
puts the HF at the centre of the mid.</p>
{table(["Unit", "Type", "Body (mm)", "Lowest crossover (kHz)", "Price each", "Best system score"], hf_rows, num=(2, 3))}
{fig("img/plots/tradeoff.png", "Every finalist, by HF unit: depth against score.")}
<p>Rule: {sel['rule']}. Chosen: {tw.model} with the {d['summary']['mid']['driver']} and the {d['summary']['low']['driver']}.</p>

<h2>The search</h2>
<ol>
<li><b>HF and mid.</b> Four area laws, three throat compressions, five mouths (220 to 380 mm), five lengths (120 to 260 mm),
crossovers f1 of 250, 315, 400 and 500 Hz, and each HF unit's upper crossovers. Only cones at least 1.6 times the plug, and
the 60 most efficient per pool plus every verified driver. {d['midsearch_n']:,} rows kept. Score: in-band variation, coverage
mismatch with the HF unit, a loading penalty when the mouth is smaller than c / (pi f1), an output floor, and a reward for a
wider mid band.</li>
<li><b>Kick.</b> The best mids became the end of the centre body in kick horns of three laws, three mouth widths, three
heights and three lengths, at most 600 mm deep. Every 15 inch with at least 4 mm Xmax that is not an instrument or coaxial
driver: {d['lowsearch_n']:,} rows passing -3 dB at or below 80 Hz and at least 96 dB. Score: variation from 80 Hz to 1.4 f1,
coverage match with the mid, low Qes for horn loading, and an output penalty below 100 dB.</li>
<li><b>System.</b> The best combinations for each HF unit through the full crossover, scored on on-axis flatness, off-axis
smoothness and bass extension.</li>
</ol>
"""
    return page("design.html", "Sunflower design", body, "How Sunflower v0.2 was sized from the inside out.")


def horn_section(name, title, intro):
    figs = [(f"img/plots/{name}-drawing.png", "Dimensioned section, mm."),
            (f"img/plots/{name}-area.png", "Open area along the horn."),
            (f"img/plots/{name}-magnitude.png", "Magnitude on axis, unfiltered, with its cutoffs."),
            (f"img/plots/{name}-phase.png", "Phase and group delay."),
            (f"img/plots/{name}-polar.png", "Off-axis responses and polar map."),
            (f"img/plots/{name}-electrical.png", "Impedance and cone excursion.")]
    blocks = "".join(fig(f, c) for f, c in figs if (DOCS / f).exists())
    return f'<h2 id="{name}">{title}</h2>\n{intro}\n{blocks}'


def simulation(d):
    dsp, fem, co = d["dsp"], d["fem"], d["cutoffs"]
    f1, f2 = D.CROSSOVER_LOW_MID_HZ, D.CROSSOVER_MID_HIGH_HZ
    lr, lm = section_band(d["low"], 80, f1)
    mr, mm = section_band(d["mid"], f1, f2)
    tr, tm = section_band(d["tweeter"], f2, 16000)
    sm = d["summary"]
    m, lo, tw = D.MID, D.LOW, D.TW
    fmt = lambda v: f"{v:.0f} Hz" if v else "outside the plotted range"  # noqa: E731
    shelf = dsp.get("no_sub_shelf", {})
    hf = fem.get("hf", {})
    low_intro = f"""<p>{sm['low']['driver']} in the kick horn: {law_name(lo.law)} law to the mid can, {lo.compression:.1f}:1 throat,
{lo.length*1000:.0f} mm long, {D.LOW_REAR_LITRES:.0f} litre sealed chamber. From 80 Hz to {f1:.0f} Hz: {lr:.1f} dB variation at
{lm:.1f} dB. Lower -3 dB at {fmt(co['low']['f_low_3db'])}; the mouth cutoff is {co['low']['mouth_cutoff_hz']:.0f} Hz. FEM
agrees within {fem['low']['max_abs_difference_db']:.1f} dB in band.</p>"""
    mid_intro = f"""<p>{sm['mid']['driver']} in {an(law_name(m.law))} {law_name(m.law)} horn round the tapering plug: {m.outer_throat_radius*2000:.0f} mm
throat, {m.mouth_diameter*1000:.0f} mm mouth, {m.length*1000:.0f} mm long. From {f1:.0f} Hz to {f2:.0f} Hz: {mr:.1f} dB variation
at {mm:.1f} dB. Lower -3 dB at {fmt(co['mid']['f_low_3db'])}, upper at {fmt(co['mid']['f_high_3db'])}, mouth cutoff
{co['mid']['mouth_cutoff_hz']:.0f} Hz. FEM agrees within {fem['mid']['max_abs_difference_db']:.1f} dB in band.</p>"""
    tw_intro = f"""<p>{tw.model}: on-axis response from the <a href="{tw.source}">datasheet</a>, read by eye to about 2 dB, with
minimum phase. Off axis from the FEM of its printed horn mouth inside the mid horn. From {f2:.0f} Hz to 16 kHz: {tr:.1f} dB
variation at {tm:.1f} dB.</p>
{fig("img/plots/hf-in-mid.png", "HF mouth inside the mid horn, axisymmetric FEM, against a free piston of the same size.")}
<p>In the FEM, sitting in the mid horn changes the HF's on-axis level by {hf.get('onaxis_change_db_mean', float('nan')):+.1f} dB on
average from {f2/1000:.1f} to 16 kHz and narrows it: {hf.get('rel_30deg_fem_mean_db', float('nan')):.1f} dB at 30 degrees against
{hf.get('rel_30deg_piston_mean_db', float('nan')):.1f} dB for the free piston.</p>"""
    body = f"""
<h1>Simulation</h1>
<p class="lede">Each horn on its own with all its plots, then the full range with its cutoffs. All at 2.83 V, 1 m.</p>
<p><a href="#low">Kick horn</a> · <a href="#mid">Mid horn</a> · <a href="#tweeter">HF</a> · <a href="#full-range">Full range</a> ·
<a href="#method">Method</a></p>
{horn_section("low", "Kick horn", low_intro)}
{horn_section("mid", "Mid horn", mid_intro)}
{horn_section("tweeter", "HF", tw_intro)}

<h2 id="full-range">Full range</h2>
{fig("img/plots/full-range.png", "Top: each section unfiltered and level-matched, with its cutoffs. Bottom: through the crossover; dashed blue is the no-sub shelf.")}
{table(["Section", "Lower -3 dB", "Upper -3 dB", "Other limit"], [
    ["Kick", fmt(co['low']['f_low_3db']), fmt(co['low']['f_high_3db']), f"Mouth cutoff {co['low']['mouth_cutoff_hz']:.0f} Hz"],
    ["Mid", fmt(co['mid']['f_low_3db']), fmt(co['mid']['f_high_3db']), f"Mouth cutoff {co['mid']['mouth_cutoff_hz']:.0f} Hz"],
    ["HF", fmt(co['tweeter']['f_low_3db']), fmt(co['tweeter']['f_high_3db']), f"Datasheet cutoff {tw.cutoff_hz:.0f} Hz"],
    ["Crossovers", f"{f1:.0f} Hz", f"{f2:.0f} Hz", "LR4"],
])}
<p>On axis the sum stays within {d['kpi']['ptp']:.1f} dB peak to peak from 100 Hz to 16 kHz (RMS {dsp['rms_deviation_100_16k_db']:.2f} dB),
before any EQ.</p>
{fig("img/plots/full-range-group-delay.png", "Group delay of the sum.")}
<h3>DSP settings</h3>
{table(["Channel", "Filter", "Gain", "Delay", "Polarity in the model"], [
    ["Kick", f"LR4 low-pass {f1:.0f} Hz", f"{dsp['gain_db']['low']:+.1f} dB", f"{dsp['delay_ms']['low']:.2f} ms", "Inverted" if dsp['polarity']['low'] < 0 else "Normal"],
    ["Mid", f"LR4 high-pass {f1:.0f} Hz, LR4 low-pass {f2:.0f} Hz", f"{dsp['gain_db']['mid']:+.1f} dB", f"{dsp['delay_ms']['mid']:.2f} ms", "Normal"],
    ["HF", f"LR4 high-pass {f2:.0f} Hz", f"{dsp['gain_db']['tweeter']:+.1f} dB", f"{dsp['delay_ms']['tweeter']:.2f} ms", "Inverted" if dsp['polarity']['tweeter'] < 0 else "Normal"],
], num=(2, 3))}
<div class="note"><p>The reversed 15 inch pulls air out of the horn when wired red to positive, the opposite of the model's
convention. Where the table says the kick is normal, invert that DSP channel; where it says inverted, leave it normal.</p>
<p><b>Without a sub.</b> A +6 dB second-order low shelf at {shelf.get('shelf_hz', float('nan')):.0f} Hz on the kick channel, with a
30 Hz LR4 high-pass, moves -6 dB to {shelf.get('f_minus6_hz', float('nan')):.0f} Hz. It needs {shelf.get('max_excursion_mm', float('nan')):.1f} mm
of excursion at the rated {shelf.get('rated_power_w', float('nan')):.0f} W against {shelf.get('xmax_mm', float('nan')):.1f} mm Xmax, so limit the kick
channel to about {shelf.get('safe_power_w', float('nan')):.0f} W when the shelf is on.</p></div>

<h2 id="method">Method</h2>
<ol>
<li><b>Plane-wave horns.</b> Thin annular slices cascaded as transfer matrices, open area = wall minus inner body.</li>
<li><b>Drivers.</b> Thiele/Small motors with their sealed volumes and front chambers (cone area times Xmax plus clearance).</li>
<li><b>Radiation.</b> Curved-wavefront far field at each mouth.</li>
<li><b>FEM.</b> Axisymmetric Helmholtz (dolfinx 0.8) for the kick horn, the mid horn and the HF mouth in the mid horn, in a
baffled half-space. Straight-tube check: median {fem['tube']['median']*100:.0f}% error in throat impedance.</li>
</ol>
{fig("img/plots/fem-vs-planewave.png", "FEM against plane-wave, kick and mid. Solid on axis, dotted 30 degrees.")}
"""
    return page("simulation.html", "Sunflower simulation", body, "Sunflower v0.2: each horn with all its plots, and the full range.")


def build(d):
    cut, parts, stack = d["cut"], d["parts"], d["stack"]
    ex = cut["external_mm"]
    P = {p["name"]: p for p in parts}
    sm = d["summary"]
    tw, m, lo = D.TW, D.MID, D.LOW
    w = cut["walls"]
    ply = [
        ["Sides", "2", f"{ex['depth']:.0f} long, {ex['height_back']:.0f} tall at the back, {ex['height_front']:.0f} at the front", "18", "side.dxf"],
        ["Top and bottom (shell)", "2", f"{lo.mouth_width*1000:.0f} wide, angled {cut['shell_angle_deg']:.1f} deg", "18", "Fit between the sides"],
        ["Back", "1", f"{lo.mouth_width*1000:.0f} x {2*lo.back_half_height*1000:.0f}", "18", "Terminal cup"],
        ["Baffle", "1", f"{lo.mouth_width*1000:.0f} x {2*lo.shell_half_height(0)*1000:.0f}", "25", "baffle.dxf: cutout and four 40 mm chamber ports"],
        ["Horn side walls", "2", f"Trapezoid {w['side_wall']['throat_edge_mm']:.0f} to {w['side_wall']['mouth_edge_mm']:.0f}, {w['side_wall']['slope_length_mm']:.0f} long", "18", f"flare {w['side_wall']['flare_deg']:.1f} deg"],
        ["Horn top and bottom walls", "2", f"Trapezoid {w['top_wall']['throat_edge_mm']:.0f} to {w['top_wall']['mouth_edge_mm']:.0f}, {w['top_wall']['slope_length_mm']:.0f} long", "18", f"flare {w['top_wall']['flare_deg']:.1f} deg"],
        ["Corner filler faces", "4", f"Triangle, base {cut['corner_filler']['face_base_mm']:.0f}", "12", "Three ribs each"],
    ]
    printed = []
    for n, label, fill in (("centre_body", "One-piece centre body", 0.25), ("mid_horn", "Mid horn", 0.45),
                           ("mid_sleeve", "Mid driver sleeve", 1.0), ("phase_plug", "Phase plug and HF horn", 1.0)):
        p = P[n]
        files = ", ".join(f"<code>{f}</code>" for f in p.get("print_files", []))
        printed.append([label, files, f"{p['volume_cm3']*1.27*fill/1000:.2f}"])
    hf_price = f"${tw.price_usd:.0f} each" if tw.price_usd else ""
    body = f"""
<h1>Build</h1>
<p class="lede">Flat birch-ply panels and printed round parts. Everything here is generated from the same design file as the
simulations.</p>
{fig("img/plots/low-drawing.png", "Cabinet section, mm.")}
{fig("img/plots/mid-drawing.png", "Mid horn, phase plug and HF horn, mm.")}
{fig("img/plots/tweeter-drawing.png", "HF unit and its printed horn, mm.")}
{fig("img/renders/exploded.jpg", "Exploded view.", "render")}

<h2>Bill of materials</h2>
{table(["Item", "Qty", "Notes"], [
    [f"{sm['low']['driver']}", "1", "Kick, rear-mounted, motor forward"],
    [f"{sm['mid']['driver']}", "1", f"Mid, {inch(m.nominal)}"],
    [f"{tw.model}", "1", f"HF, {hf_price}"],
    ["Birch ply 18 mm", "About 3 m²", ""],
    ["Birch ply 25 mm", "1 baffle", ""],
    ["Birch ply 12 mm", "About 0.3 m²", "Corner filler faces and ribs"],
    ["Aluminium tube, 12 mm OD", "4", "Centre body supports; the top one carries the mid and HF cables"],
    ["PETG or ASA filament", "About 3 to 4 kg", "Printed parts below"],
    ["M6 inserts and bolts", "8", "15 inch, from the chamber side"],
    ["M4 heat-set inserts and screws", "12", "Mid driver, HF unit, part joints"],
    ["Speaker cable 2 x 1.0 mm²", "About 3 m", ""],
    ["Terminal cup", "1", "One pair per amplifier channel"],
    ["Polyester wadding", "0.5 m²", "Chamber"],
    ["Closed-cell foam blocks", f"{cut['chamber']['fill_litres']:.0f} litres", f"Fill the chamber down from {cut['chamber']['available_litres']:.0f} to {cut['chamber']['target_litres']:.0f} litres"],
    ["3-way DSP and 3 amplifier channels", "per box", "LR4, delay, gain, shelf, limiter"],
])}

<h2>Cut list</h2>
{table(["Part", "Qty", "Size (mm)", "Ply (mm)", "Notes"], ply, num=(1, 3))}
<p>DXF: <a href="cad/dxf/side.dxf">side</a>, <a href="cad/dxf/baffle.dxf">baffle</a>,
<a href="cad/dxf/horn_side_wall.dxf">horn side wall</a>, <a href="cad/dxf/horn_top_wall.dxf">horn top wall</a>.</p>

<h2>Printed parts</h2>
<p>Each part is already cut into sections and quarters that fit a 250 mm bed, in <code>cad/print/</code>.</p>
{table(["Part", "Print files", "Est. mass (kg)"], printed, num=(2,))}
<p>Before printing, measure: the 15 inch cone profile and motor (the centre body's rear face follows the cone at Xmax plus
3 mm and clears the motor by 3 mm), the mid cone profile (lead-in at 1.5 mm), and the HF unit's body cylinder (the plug is bored
to {tw.body_diameter*1000 + 1:.0f} mm).</p>

<h2>Assembly</h2>
<ol class="steps">
<li><b>Shell.</b> Glue the sides to the angled top and bottom and the back. Fit the baffle {lo.back_box*1000:.0f} mm in front of the back.
Seal every joint.</li>
<li><b>Kick driver.</b> Pass it motor-first through the baffle from behind; flange on the back face, on gasket, eight M6 bolts.</li>
<li><b>Horn walls.</b> Glue in the four internal horn walls and the corner fillers. The space outside them is the sealed chamber,
joined to the box behind the baffle through the four baffle ports. Add foam blocks to bring it to {cut['chamber']['target_litres']:.0f} litres.</li>
<li><b>Centre body.</b> Glue the printed sections into one piece. Its rear face sits {(lo.body.xmax + lo.gap)*1000:.0f} mm in front
of the 15 inch cone and does not touch the motor; it hangs on the rods, with a foam ring at the motor.</li>
<li><b>Mid and HF.</b> Fit the HF unit into the phase plug, the plug's vanes into the mid horn throat, the mid driver to the back of
the mid horn, then the sleeve and the centre body.</li>
<li><b>Supports.</b> Four 12 mm tubes through the horn walls into the can; cables out through the top tube, through the chamber,
to the terminal cup.</li>
<li><b>DSP.</b> Start from the settings on the <a href="simulation.html#full-range">simulation page</a>, then measure each section.</li>
<li><b>Stacking.</b> Boxes sit splayed at {stack['splay_deg']:.1f} degrees, angled panel on angled panel; fly or strap them as a column.</li>
</ol>
{fig("img/renders/rear-open.jpg", "Back removed.", "render")}
"""
    return page("build.html", "Sunflower build", body, "Sunflower v0.2 bill of materials, cut list, printed parts and assembly.")


def downloads(d):
    parts = d["parts"]
    rows = [[f"<code>{p['name']}</code>", p["material"].replace("_", " "), p.get("note", ""),
             f'<a href="cad/step/{p["name"]}.step">STEP</a> · <a href="cad/stl/{p["name"]}.stl">STL</a>'] for p in parts]
    res = [["Section responses", '<a href="results/planewave/low.csv">low</a> · <a href="results/planewave/mid.csv">mid</a> · <a href="results/planewave/tweeter.csv">HF</a>'],
           ["System, 0/15/30 deg and no-sub shelf", '<a href="results/planewave/system.csv">system.csv</a>'],
           ["DSP settings", '<a href="results/planewave/dsp.json">dsp.json</a>'],
           ["Cutoffs", '<a href="results/planewave/cutoffs.json">cutoffs.json</a>'],
           ["FEM", '<a href="results/fem/summary.json">summary.json</a> · <a href="results/fem/low.csv">low</a> · <a href="results/fem/mid.csv">mid</a> · <a href="results/fem/hf-in-mid.csv">HF in mid</a>'],
           ["Optimiser", '<a href="results/optimise/mid-search.csv.gz">mid search</a> · <a href="results/optimise/mid-shortlist.csv">shortlist</a> · <a href="results/optimise/low-search.csv.gz">low search</a> · <a href="results/optimise/system-search.csv">finalists</a> · <a href="results/optimise/selected.json">selected</a>']]
    body = f"""
<h1>Downloads</h1>
<p class="lede">All generated by the scripts in the repository. Millimetres.</p>
<p><a href="cad/sunflower-assembly.step">sunflower-assembly.step</a> · <a href="cad/cutlist.json">cutlist.json</a> ·
<a href="cad/stack.json">stack.json</a> · <a href="cad/parts.json">parts.json</a></p>
<h2>Parts</h2>
<p>Printable sections are in <code>cad/print/</code>. Parts named <code>drv_</code> are dimensioned stand-ins for the drivers.</p>
{table(["Part", "Material", "Notes", "Files"], rows)}
<h2>Data</h2>
{table(["Data", "Files"], res)}
<h2>Reproduce</h2>
<pre>git clone {REPO}
cd sunflower
just all</pre>
"""
    return page("downloads.html", "Sunflower downloads", body, "Sunflower v0.2 CAD and data.")


def main():
    d = load()
    for name, fn in (("index.html", index), ("review.html", review), ("design.html", design),
                     ("simulation.html", simulation), ("build.html", build), ("downloads.html", downloads)):
        (DOCS / name).write_text(fn(d))
    (DOCS / ".nojekyll").write_text("")
    print("site written to", DOCS)


if __name__ == "__main__":
    main()
