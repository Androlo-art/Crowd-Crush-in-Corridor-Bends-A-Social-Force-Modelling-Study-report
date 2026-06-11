# results / 01_faster_is_slower

Final faster-is-slower benchmark data (Ch3 + App B). Each folder has `raw_results.csv`
(per-seed rows) and `run_info.txt` (exact parameters).

| Folder | Casualty rule / variant |
|---|---|
| `benchmark_standard/` | full-obstruction |
| `benchmark_soft_radius/` | reduced-radius r_s=0.6 |
| `benchmark_non_interacting/` | non-interacting control |
| `door_width_1.05m/` | door-width calibration |
| `helbing_reference/` | digitised Helbing et al. (2000) curves |

The thesis faster-is-slower figures are generated directly from these `raw_results.csv`
files by `code/plotting/report_figures/plot_fis_report_figures.py` — no simulation rerun
is needed.
