# Sunflower

A three-way coaxial horn loudspeaker where each horn's phase plug is the next horn.

![Sunflower, rendered from the CAD](docs/img/renders/hero.jpg)

- **Low:** an 18 Sound 15NMB420, mounted backwards, drives a flat-panel horn that goes from an octagonal throat to a 700 mm square mouth. The driver's motor and a printed nose cone are the phase plug.
- **Mid:** a Beyma 6MCF200Nd loads a conical horn through an annular throat. The horn, the driver and an elliptical tail sit in a 200 mm can in the low horn's mouth, where the can is the low horn's centre body.
- **Tweeter:** a Fostex T90A sits in a printed bullet in the mid horn throat, and the bullet is the mid horn's phase plug.

Site with renders, response plots and build instructions: **https://timini.github.io/sunflower/**

## Simulated performance

| | |
|---|---|
| Sensitivity | 103.9 dB, 2.83 V, 1 m |
| Response | 81 Hz to 20 kHz, -3 dB, on axis after DSP |
| Flatness | 3.6 dB peak to peak, 100 Hz to 20 kHz, before EQ |
| Crossovers | LR4 at 500 Hz and 7 kHz, active |
| Cabinet | 736 x 736 x 888 mm, 18 mm birch ply |

These figures come from a plane-wave horn model coupled to Thiele/Small driver models. Axisymmetric FEM of the mid and low horns agrees within 1.9 dB and 0.8 dB on axis in their bands. Nothing has been built or measured yet. The site's simulation page lists what the models leave out.

## Repository

| Path | What it is |
|---|---|
| `src/sunflower/design.py` | Every dimension, driver and crossover point. All other scripts read from here |
| `src/sunflower/geometry.py` | Nested area functions of the low and mid horns |
| `src/sunflower/acoustics.py` | Plane-wave horn network, driver model, curved-wavefront radiation |
| `src/sunflower/screen_15.py` | 15 inch driver and low horn screen |
| `src/sunflower/simulate.py` | Per-section responses, crossover, delays and sum |
| `src/sunflower/fem.py`, `fem/axisym.py` | Axisymmetric FEM check in the dolfinx Docker image |
| `src/sunflower/cad.py` | STEP, STL and DXF for every part, cut list |
| `blender/render.py` | Cycles renders from the STL files |
| `src/sunflower/plots.py`, `src/sunflower/site.py` | Figures and the GitHub Pages site in `docs/` |
| `data/drivers/` | Driver data. `verified/` holds the two records checked against datasheets |
| `results/`, `cad/` | Generated outputs, committed |

## Reproduce

Needs [uv](https://docs.astral.sh/uv/), [just](https://just.systems), Docker for the FEM step, Blender 4.2 or later for renders, and ffmpeg for the turntable GIF.

```sh
just all     # screen, sim, fem, cad, plots, render, site, test
just test    # tests only
just serve   # preview the site at http://localhost:8000
```

The mid horn comes from a study in [horn-simulation](https://github.com/timini/horn-simulation). `tests/` checks that the port here reproduces that study's response within 0.5 dB.

## Licence

MIT
