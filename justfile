# Sunflower: three-way nested coaxial horn

default:
    @just --list

# Unit and consistency tests
test:
    uv run pytest -q

# Inside-out optimisation of HF unit, mid and kick (about 15 minutes); add --from-low to redo only stages 2 and 3
optimise *ARGS:
    uv run python -m sunflower.optimise {{ARGS}}

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

# Blender stills and animations, then dimensions and video encoding (Blender 4.5 or later; about 40 minutes on an M3 Pro)
render *ARGS:
    blender -b -P blender/render.py -- --out docs/img/renders {{ARGS}}
    uv run python -m sunflower.annotate
    just encode

# Encode the animation frames in .tmp/frames to mp4 (H.264), webm (VP9) and a poster jpg, then delete the frames
encode:
    #!/usr/bin/env bash
    set -euo pipefail
    out=docs/img/renders
    for name in turntable orbit assemble; do
        d=.tmp/frames/$name
        [ -e "$d/f0001.png" ] || continue
        ffmpeg -y -loglevel error -framerate 30 -i "$d/f%04d.png" -c:v libx264 -preset slow -crf 18 -pix_fmt yuv420p -movflags +faststart "$out/$name.mp4"
        ffmpeg -y -loglevel error -framerate 30 -i "$d/f%04d.png" -c:v libvpx-vp9 -crf 30 -b:v 0 -row-mt 1 -deadline good -cpu-used 2 -pix_fmt yuv420p "$out/$name.webm"
        ffmpeg -y -loglevel error -i "$d/f0001.png" -q:v 3 "$out/$name-poster.jpg"
        rm -rf "$d"
        ls -l "$out/$name".* "$out/$name-poster.jpg"
    done

# HTML pages from the results
site:
    uv run python -m sunflower.site

# Everything, in order
all: optimise sim cad fem
    uv run python -m sunflower.optimise --fem-f2
    uv run python -m sunflower.simulate
    just plots render site test

# Preview the site with the CAD and results folders in place, as Pages serves it
serve:
    rm -rf .tmp/site && mkdir -p .tmp/site && cp -r docs/. .tmp/site && cp -r cad results .tmp/site/
    python3 -m http.server -d .tmp/site 8000
