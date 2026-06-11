# Code

- `model/` — the simulation (C core + `sfm` Python driver + fast-marching navigation).
- `experiments/` — one run script per study (need built binaries; see model/c_core/README.md).
- `plotting/` — regenerate the thesis figures from the included `results/` (no rerun needed).

See `docs/figure_provenance.md` and `docs/how_to_regenerate_figures.md`.
