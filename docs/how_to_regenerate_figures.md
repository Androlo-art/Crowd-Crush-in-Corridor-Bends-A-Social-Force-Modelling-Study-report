# How to regenerate the figures

All figures regenerate from the included `results/` — **no simulation rerun is needed**.
Install the Python deps first (`pip install -r requirements.txt`).

Run the commands from `code/plotting/` so the shared modules import correctly. Figures
are written to `figures/_regenerated/` (a scratch folder); the canonical thesis figures
are the committed ones under `figures/<section>/`.

```bash
cd code/plotting
```

## The report figures (most of the thesis figures)

These three entry scripts take no arguments and read the release `results/` directly:

```bash
python report_figures/plot_fis_report_figures.py        # FIS: 4-panel, doorwidth, variant overlay
python report_figures/plot_bend_report_figures.py       # bend open-capacity, crush, density, pressure; capacity phase/saturation/throughput
python report_figures/plot_density_legend_bottom.py     # capacity density montage; obstacle config grid
```

## The remaining figures (generators with explicit data paths)

```bash
# Ch2 — navigation floor-field figure
python plot_navigation_field.py        # uses results/00_model/navigation_fields_d12 by default

# Ch5 — rounded-corner line trends (bottleneck + open chamfer runs)
python plot_chamfer.py --bn ../../results/03_rounded_corner/chamfer_bottleneck \
                       --open-dir ../../results/03_rounded_corner/chamfer_open \
                       --outdir ../../figures/_regenerated

# Ch5/App — corner contact + open-density heatmap strips (and the appendix crowd-pressure strip)
python plot_report_heatmaps.py --chamfer-bn ../../results/03_rounded_corner/chamfer_bottleneck \
                               --chamfer-open ../../results/03_rounded_corner/chamfer_open \
                               --pressure-run ../../results/02_open_and_restricted_bend/open_bend_capacity \
                               --outdir ../../figures/_regenerated --suffix ''

# Ch4 — bottleneck crush + open capacity (run the module directly with a run folder)
python plot_bottleneck.py ../../results/02_open_and_restricted_bend/restricted_bend_bottleneck
python plot_bottleneck.py ../../results/02_open_and_restricted_bend/open_bend_capacity --open

# Ch6/App — coupling capacity line figures + density montage
python analyze_capacity.py --run ../../results/04_turn_exit_coupling/capacity_chamfered \
                           --open ../../results/02_open_and_restricted_bend/open_bend_capacity \
                           --outdir ../../figures/_regenerated
python maps_capacity.py --run ../../results/04_turn_exit_coupling/capacity_chamfered \
                        --outdir ../../figures/_regenerated --montage --field rho --radius 0.25

# Ch7/App — obstacle fields, profiles, config grid
python analyze_obstacle.py --run ../../results/05_obstacle_mitigation/obstacle_main \
                           --outdir ../../figures/_regenerated --field both

# App — obstacle throughput (reads the transit summary)
python plot_transit.py ../../results/05_obstacle_mitigation/obstacle_transit --out ../../figures/_regenerated/obstacle_throughput
```

Notes
- The mechanism-snapshot figures (`plot_fis_mechanism_snapshots.py`, `fis_snapshot_review.py`)
  **re-run one short trajectory** and therefore need the built FIS room binary
  (`code/model/c_core/room_fis`); the committed PDFs are included, so this is optional.
- A few generators resolve relative `--run` arguments against `code/plotting/`; pass the
  `../../results/...` paths shown above (or absolute paths) and they work as listed.
- `plot_bottleneck.py` / `plot_chamfer.py` print their output filenames; the report uses
  the `_revA`/`_revB` versions produced by the `report_figures/` scripts.
