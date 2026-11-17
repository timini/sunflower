# Sunflower

A three-way coaxial horn where each horn's phase plug is the next horn, laid out like a Funktion-One Res top (kick, wide-band mid, small HF) on one axis.

![Sunflower stack, rendered from the CAD](docs/img/renders/stack.jpg)

Designed from the inside out:

1. **HF:** an 18 Sound ND1030 compression driver (65 mm body cylinder, about $150) on a small printed horn.
2. **Mid phase plug:** that body in a 2 mm sleeve, tapering to the HF horn mouth.
3. **Mid driver:** a Beyma 6MCF200Nd (6.5 inch), whose cone is at least 1.6 times the plug.
4. **Mid horn:** an oblate spheroidal (OS) open-area law from a 2.2:1 annular throat to a 220 mm mouth over 150 mm, with the plug always subtracted.
5. **Centre body:** the mid can ends a one-piece centre body whose rear face follows the reversed Eminence DELTA-15LFA cone. That gives a 2.3:1 kick-horn throat, an HD-15 style tube-like kick horn, 500 mm long.

Site with renders, each horn's plots, the full-range response, the v0.2 review and build instructions: **https://timini.github.io/sunflower/**

## Simulated performance (v0.2)

| | |
|---|---|
| Sensitivity | 103.5 dB, 2.83 V, 1 m |
| Response | 81 Hz to 16 kHz, -3 dB, on axis; use with a sub |
| No-sub shelf | -6 dB at 61 Hz, kick channel limited to about 48 W |
| Crossovers | LR4 at 500 Hz and 3500 Hz |
| Cabinet | 596 wide, 636 high at the front, 479 at the back, 588 deep; 15.7 degree splay when stacked |

These are plane-wave horn models coupled to Thiele/Small driver models, checked with axisymmetric FEM:
- kick within 1.2 dB;
- mid within 0.7 dB;
- the HF mouth inside the mid horn, a separate case.

Nothing has been built or measured. v0.1 is kept at the `v0.1` tag.

## Repository

| Path | What it is |
|---|---|
| `src/sunflower/optimise.py` | Inside-out search over HF unit, mid driver and horn, kick driver and horn, crossovers, size and HF cost |
| `src/sunflower/geometry.py` | Area-law horns with phase plugs and the one-piece centre body |
| `src/sunflower/design.py` | The selected design, from `results/optimise/selected.json` |
| `src/sunflower/acoustics.py`, `simulate.py` | Plane-wave horns, driver models, crossover tuning, the no-sub shelf |
| `src/sunflower/tweeters.py`, `data/tweeters/` | HF units from their datasheets, with prices |
| `src/sunflower/fem.py`, `fem/axisym.py` | Axisymmetric FEM in the dolfinx Docker image |
| `src/sunflower/cad.py` | STEP, STL, print-bed sections, DXF, cut list, stack angles |
| `blender/render.py`, `src/sunflower/annotate.py` | Cycles renders, splayed stack, dimension overlays |
| `src/sunflower/plots.py`, `src/sunflower/site.py` | Figures and the GitHub Pages site |
| `data/drivers/` | Catalogues; `verified/` checked by hand against datasheets; `excluded.json` lists removed parts and why |

## Reproduce

Needs [uv](https://docs.astral.sh/uv/), [just](https://just.systems), Docker for FEM, Blender 4.2 or later, ffmpeg.

```sh
just all
```

## Licence

MIT
