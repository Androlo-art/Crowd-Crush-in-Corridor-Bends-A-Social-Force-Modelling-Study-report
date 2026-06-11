# Plotting

These scripts regenerate the thesis figures from the included `results/`. **No
simulation rerun is needed** for the figures. See `docs/how_to_regenerate_figures.md`
for exact commands, and `docs/figure_provenance.md` for the full figure→script→data map.

## Why this folder is flat

The plotting modules `import` each other by name (e.g. `plot_report_heatmaps` imports
`plot_bottleneck`, `plot_chamfer`, `thesis_style`; the `report_figures/` scripts import
several of them). Splitting them into per-section subfolders would break those imports,
so the shared modules are kept flat here under their original importable names. The
per-section organisation lives in `results/`, `figures/`, `videos/` and `docs/`.

## Entry points (what to run)

| Script | Produces (thesis figure) |
|---|---|
| `report_figures/plot_fis_report_figures.py` | `fis_standard_4panel`, `fis_doorwidth`, `fis_variant_overlay` |
| `report_figures/plot_bend_report_figures.py` | `bend_open_capacity`, `bend_bottleneck_crush`*, `bend_open_vs_restricted_density`, `bend_pressure_appendix`, `bend_capacity_phase/saturation/throughput` |
| `report_figures/plot_density_legend_bottom.py` | `bend_capacity_density_montage`, `obstacle_grid_density` |
| `plot_navigation_field.py` | `bend_floorfield` (Ch2) |
| `plot_chamfer.py` | `bend_chamfer_lines` (Ch5) |
| `plot_report_heatmaps.py` | `bend_chamfer_contact`, `bend_chamfer_density_open` |
| `plot_bottleneck.py` | `bend_open_capacity`, `bend_bottleneck_crush` (base versions) |
| `analyze_capacity.py` | capacity phase/saturation/throughput (base versions) |
| `maps_capacity.py` | capacity density montage (base) |
| `analyze_obstacle.py` | `obstacle_contact_90`, `obstacle_profile_grid`, `obstacle_grid_density` |
| `plot_transit.py` | `obstacle_throughput` |
| `plot_fis_mechanism_snapshots.py`, `fis_snapshot_review.py` | FIS mechanism snapshots / contact sheet (re-run a short trajectory; need the FIS room binary) |

\* `report_figures/plot_bend_report_figures.py` regenerates the formatting-final
(`_revA`) versions used in the thesis; `plot_bottleneck.py` etc. are the underlying base
generators those scripts import.

## Shared modules (imported, usually not run directly)
`thesis_style.py` (styling); `make_field_bottleneck.py`, `make_field_chamfer.py`,
`make_field_capacity_chamfer.py`, `make_field_obstacle.py` (geometry/overlays used by the
heatmap plotters).

Output of the `report_figures/` scripts goes to `figures/_regenerated/` (scratch); the
committed figures under `figures/<section>/` are the canonical thesis versions.
