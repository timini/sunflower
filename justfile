# Sunflower: three-way nested coaxial horn

default:
    @just --list

# Unit and consistency tests
test:
    uv run pytest -q

# 15 inch driver and low horn screen, then the refinement for the chosen driver
screen:
    uv run python -m sunflower.screen_15
    uv run python -m sunflower.screen_15 --refine

# Plane-wave simulation of all sections, crossover and sum
sim:
    uv run python -m sunflower.simulate

# Axisymmetric FEM check (needs Docker, pulls dolfinx/dolfinx:v0.8.0)
fem:
    uv run python -m sunflower.fem

# STEP, STL and DXF files, part list and cut list
cad:
    uv run python -m sunflower.cad

# Figures for the site
plots:
    uv run python -m sunflower.plots

# Blender renders (needs Blender 4.2 or later on PATH)
render *ARGS:
    blender -b -P blender/render.py -- --out docs/img/renders {{ARGS}}
    ffmpeg -y -loglevel error -framerate 10 -i docs/img/renders/turntable/f%03d.png -vf "split[a][b];[a]palettegen=max_colors=128[p];[b][p]paletteuse" docs/img/renders/turntable.gif

# HTML pages from the results
site:
    uv run python -m sunflower.site

# Everything, in order
all: screen sim fem cad plots render site test

# Preview the site with the CAD and results folders in place, as Pages serves it
serve:
    rm -rf .tmp/site && mkdir -p .tmp/site && cp -r docs/. .tmp/site && cp -r cad results .tmp/site/
    python3 -m http.server -d .tmp/site 8000
