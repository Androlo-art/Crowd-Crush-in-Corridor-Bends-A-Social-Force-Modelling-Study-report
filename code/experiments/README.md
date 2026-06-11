# Experiments (simulation run scripts)

Run scripts for the geometry studies. Each launches the C simulation binary (via
`sfm.backends.c_exe`) over its parameter sweep and writes results under `results/`.

⚠️ They require the third-party simulation core (obtain from upstream; see `code/model/c_core/README.md`) and a precomputed
floor field for the geometry; re-running the full sweeps is hundreds of CPU-hours. The
included `results/` already contain the final data, so **re-running is not needed to
reproduce the figures**. See `docs/how_to_run_representative_simulations.md`.

| Folder | Study |
|---|---|
| `02_open_and_restricted_bend/run_open_and_restricted_bend.py` | open + restricted bend (`--exe` selects the binary) |
| `03_rounded_corner/run_rounded_corner_sweep.py` | inner-corner radius sweep |
| `04_turn_exit_coupling/run_turn_exit_coupling.py` | turn × exit sweep (chamfered) |
| `05_obstacle_mitigation/run_obstacle_mitigation.py` | obstacle layouts |
| `05_obstacle_mitigation/run_obstacle_navigation_check.py` | navigation-aware robustness check |
| `05_obstacle_mitigation/extract_obstacle_transit.py` | obstacle throughput extraction |

**Faster-is-slower** is the model-validation benchmark: its final raw results and
metadata are in `results/01_faster_is_slower/`, and the figures regenerate from them via
`code/plotting/report_figures/plot_fis_report_figures.py`. Representative single-room
runs can be produced from the model code (`docs/how_to_run_representative_simulations.md`).

Note: the `EXE` / `FIELD_DIR` constants at the top of each script assume the simulation
binary and a precomputed floor field exist locally — adjust them to your paths.
