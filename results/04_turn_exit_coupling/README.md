# results / 04_turn_exit_coupling

Final coupled turn-angle × exit-width sweep (Ch6 + App), chamfered r=0.25, 6 angles ×
8 exit widths × 15 seeds. Contains `aggregated.csv`, `run_info.txt`, `grids/` (the source
for the density montage).

| Folder | Used for |
|---|---|
| `capacity_chamfered/` | phase map, occupancy saturation, outflow, density montage |

Figures: `report_figures/plot_bend_report_figures.py` (phase/saturation/throughput),
`report_figures/plot_density_legend_bottom.py` (montage), or the `analyze_capacity.py` /
`maps_capacity.py` generators. No rerun needed.
