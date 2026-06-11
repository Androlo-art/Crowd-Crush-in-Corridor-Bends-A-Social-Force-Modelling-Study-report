# Crowd dynamics in bend-fed corridors — thesis support repository

This repository supports the submitted thesis by providing the thesis-specific analysis
and plotting code, processed simulation results, report figures, provenance documentation
and representative videos. The simulation core is based on the Farkas–Helbing–Vicsek /
Helbing escape-panic package; third-party licensing and attribution are documented
separately (see [`LICENSE_OR_NOTICE.md`](LICENSE_OR_NOTICE.md) and
[`code/model/c_core/README.md`](code/model/c_core/README.md)). The repository is intended
to support inspection and reproducibility of the reported figures, **not** to serve as a
general-purpose crowd-simulation package.

The study uses a social-force model to examine how **turn angle**, **exit width**,
**inner-corner rounding** and a **placed obstacle** govern where dense-crowd crush
concentrates in a bend-fed corridor.

## What is here

```
code/        thesis-specific Python: model driver + navigation, experiment runners, plotting
results/     the exact result folders behind each reported figure (CSV + run_info + grids)
figures/     the report figures, organised by thesis section
videos/      a representative simulation video per scenario
docs/        figure provenance + how-to guides
licenses/    third-party (panic) non-profit and commercial licence texts
```

> **Note on the simulation core.** The C escape-panic core is **third-party** (see below);
> the folder `code/model/c_core/` documents its upstream source and the thesis
> modifications. The included results, figures and Python code regenerate the reported
> figures without it.

`results/`, `figures/` and `videos/` are organised by thesis section:

| Section folder | Thesis content |
|---|---|
| `01_faster_is_slower` | faster-is-slower benchmark (model validation) |
| `02_open_and_restricted_bend` | open-bend capacity and the restricted-exit crush |
| `03_rounded_corner` | inner-corner rounding study |
| `04_turn_exit_coupling` | coupled turn-angle × exit-width study |
| `05_obstacle_mitigation` | pillar/gate obstacle study |
| `00_model` (figures/results) | the navigation floor-field model figure |

`code/plotting/` is one flat folder (the plotting modules import each other by name), with
a README mapping every figure to the script and section that produce it.

## Install

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt          # numpy, scipy, matplotlib, scikit-fmm
```

## Regenerate the figures

The figures regenerate from the included `results/` — **no simulation rerun needed** (and
no C core needed). See `docs/how_to_regenerate_figures.md`. Quick example:

```bash
cd code/plotting/report_figures
python plot_fis_report_figures.py        # reads results/01_…, writes figures/_regenerated/
```

## Provenance

- `docs/figure_provenance.md` — every figure → the script and result folder that produce it.

## Simulation core (third-party)

The simulation core is based on the pedestrian escape-panic package of Farkas, Helbing &
Vicsek (Nature 407:487–490, 2000; upstream: https://github.com/fij/panic). It is **not my
work** and is **not relicensed**. `code/model/c_core/README.md` documents the base package
and the thesis modifications (parameterised bend geometry, variable exit width, inner-corner
rounding, optional column contact force); the panic non-profit and commercial licence texts
are in `licenses/`. My own Python scripts, processed results, figures and documentation are
the author's own work (see `LICENSE_OR_NOTICE.md`).

## Limitations

Research code for one corridor family and one model. Numerical thresholds (e.g. the
saturation width d*(θ)) are internal to these simulations and are **not** real-world design
figures. See the thesis for the full discussion.
