# Figure provenance

Each figure in the thesis, the script that produces it, and the result folder it reads.

| Thesis section | Figure | Figure file | Plotting script | Result folder | Notes |
|---|---|---|---|---|---|
| Model (Ch2) | navigation field | `figures/00_model/bend_floorfield.pdf` | `code/plotting/plot_navigation_field.py` | `results/00_model/navigation_fields_d12/` | fast-marching navigation field |
| FIS (Ch3) | benchmark (4-panel) | `figures/01_faster_is_slower/fis_standard_4panel.pdf` | `code/plotting/report_figures/plot_fis_report_figures.py` | `results/01_faster_is_slower/benchmark_standard/` (+ `helbing_reference/`) | leaving time, flow, variability, casualties vs desired speed |
| FIS (Ch3) | mechanism snapshots | `figures/01_faster_is_slower/fis_mechanism_snapshots.pdf` | `code/plotting/plot_fis_mechanism_snapshots.py` | single illustrative run | one high-speed run; needs the simulation binary to regenerate |
| FIS (App B) | door-width check | `figures/01_faster_is_slower/fis_doorwidth.pdf` | `code/plotting/report_figures/plot_fis_report_figures.py` | `door_width_1.05m/` + `benchmark_standard/` | leaving time vs desired speed, two door widths |
| FIS (App B) | casualty-rule comparison | `figures/01_faster_is_slower/fis_variant_overlay.pdf` | `code/plotting/report_figures/plot_fis_report_figures.py` | `benchmark_{standard,soft_radius,non_interacting}/` | leaving time + casualties for the three casualty rules |
| FIS (App B) | mechanism contact sheet | `figures/01_faster_is_slower/fis_mechanism_contactsheet.pdf` | `code/plotting/fis_snapshot_review.py` | single illustrative run | frame grid of one run; needs the simulation binary to regenerate |
| Bend (Ch4) | open-bend capacity | `figures/02_open_and_restricted_bend/bend_open_capacity_revA.pdf` | `code/plotting/report_figures/plot_bend_report_figures.py` | `results/02_open_and_restricted_bend/open_bend_capacity/` | capacity and inner-wall density vs turn angle |
| Bend (Ch4) | crush vs angle | `figures/02_open_and_restricted_bend/bend_bottleneck_crush.pdf` | `code/plotting/plot_bottleneck.py` | `results/02_open_and_restricted_bend/restricted_bend_bottleneck/` | door/corner density and casualties vs turn angle |
| Bend (Ch4) | open vs restricted density | `figures/02_open_and_restricted_bend/bend_open_vs_restricted_density_revA.pdf` | `code/plotting/report_figures/plot_bend_report_figures.py` | `open_bend_capacity/` + `restricted_bend_bottleneck/` | steady-state density fields, open vs 1.2 m door |
| Bend (App) | crowd pressure | `figures/02_open_and_restricted_bend/bend_pressure_appendix_revA.pdf` | `code/plotting/report_figures/plot_bend_report_figures.py` | `open_bend_capacity/grids` | crowd-pressure fields across turn angle |
| Corner (Ch5) | trends vs radius | `figures/03_rounded_corner/bend_chamfer_lines.pdf` | `code/plotting/plot_chamfer.py` | `chamfer_bottleneck/` + `chamfer_open/` | density/contact/casualties vs inner-corner radius |
| Corner (Ch5) | contact fields | `figures/03_rounded_corner/bend_chamfer_contact.pdf` | `code/plotting/plot_report_heatmaps.py` | `chamfer_bottleneck/grids` | contact-force fields vs inner-corner radius |
| Corner (App) | open density fields | `figures/03_rounded_corner/bend_chamfer_density_open.pdf` | `code/plotting/plot_report_heatmaps.py` | `chamfer_open/grids` | density fields vs inner-corner radius (open) |
| Coupling (Ch6) | crush phase map | `figures/04_turn_exit_coupling/bend_capacity_phase_revA.pdf` | `code/plotting/report_figures/plot_bend_report_figures.py` | `capacity_chamfered/` | mean casualties over the turn-angle × exit-width plane |
| Coupling (App) | occupancy transition | `figures/04_turn_exit_coupling/bend_capacity_saturation_revA.pdf` | `code/plotting/report_figures/plot_bend_report_figures.py` | `capacity_chamfered/` | corridor occupancy vs exit width |
| Coupling (App) | realised outflow | `figures/04_turn_exit_coupling/bend_capacity_throughput_revA.pdf` | `code/plotting/report_figures/plot_bend_report_figures.py` | `capacity_chamfered/` | realised agent outflow vs exit width |
| Coupling (App) | density montage | `figures/04_turn_exit_coupling/bend_capacity_density_montage_revB.pdf` | `code/plotting/report_figures/plot_density_legend_bottom.py` | `capacity_chamfered/grids` | density fields over the turn-angle × exit-width grid |
| Obstacle (Ch7) | contact fields (90°) | `figures/05_obstacle_mitigation/obstacle_contact_90.pdf` | `code/plotting/analyze_obstacle.py` | `obstacle_main/` | contact fields: no obstacle vs offset pillar vs gate |
| Obstacle (Ch7) | crush location | `figures/05_obstacle_mitigation/obstacle_profile_grid.pdf` | `code/plotting/analyze_obstacle.py` | `obstacle_main/` | where crush sits along the corridor |
| Obstacle (App) | configuration grid | `figures/05_obstacle_mitigation/obstacle_grid_density_revA.pdf` | `code/plotting/report_figures/plot_density_legend_bottom.py` | `obstacle_main/` | density fields across all obstacle layouts |
| Obstacle (App) | throughput | `figures/05_obstacle_mitigation/obstacle_throughput.pdf` | `code/plotting/plot_transit.py` | `obstacle_transit/transit_summary.csv` | agent throughput vs turn angle |

Note: the model schematic figures in Ch2 / Appendix A (the pedestrian-interaction geometry,
the update-loop diagram, and the parameter tables) are drawn directly in the thesis with
TikZ; they have no separate plotting script or data file and so are not part of this repository.
