#!/usr/bin/env python3
"""
Regenerate the two density-grid figures with the colourbar along the bottom (so the panel
grid is wider): the obstacle configuration grid and the coupled turn-angle x exit-width
density montage. Reads the saved density grids in results/ only; runs no simulation.
Outputs go to figures/_regenerated/.

Run from code/plotting/report_figures/:
  python plot_density_legend_bottom.py
"""
import os, sys, glob, shutil, datetime
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

_HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(_HERE, '..', '..', '..'))
sys.path.insert(0, os.path.dirname(_HERE))                        # shared plotting modules
import thesis_style as TS
import analyze_obstacle as AO     # fig 1 helpers (appendix_grid internals)
import maps_capacity as MC        # fig 2 helpers (montage internals)
TS.apply()

THESIS_FIG = os.path.join(REPO, 'figures', '_regenerated')        # regeneration target (NOT the thesis)
PREVIEW = os.path.join(THESIS_FIG, 'previews')
os.makedirs(PREVIEW, exist_ok=True)
STAMP = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')

OBS   = os.path.join(REPO, 'results/05_obstacle_mitigation/obstacle_main')        # fig 1 data
CAPCH = os.path.join(REPO, 'results/04_turn_exit_coupling/capacity_chamfered')    # fig 2 data


def backup_current(ref_name):
    """Back up (once) the CURRENT referenced thesis PDF before anything else."""
    src = os.path.join(THESIS_FIG, ref_name)
    if not os.path.exists(src):
        return
    stem = ref_name[:-4]
    if glob.glob(os.path.join(THESIS_FIG, '%s_backup_*.pdf' % stem)):
        print('  backup  %-40s -> (already exists, skipped)' % ref_name); return
    dst = os.path.join(THESIS_FIG, '%s_backup_%s.pdf' % (stem, STAMP))
    shutil.copy2(src, dst)
    print('  backup  %-40s -> %s' % (ref_name, os.path.basename(dst)))


def save(fig, out_name):
    pdf = os.path.join(THESIS_FIG, out_name + '.pdf')
    fig.savefig(pdf, bbox_inches='tight')
    fig.savefig(os.path.join(PREVIEW, out_name + '.png'), dpi=150, bbox_inches='tight')
    plt.close(fig)
    print('  wrote   %s' % pdf)


# ===== Figure 1: obstacle density grid (colourbar -> bottom) =================
def fig_obstacle_grid():
    backup_current('obstacle_grid_density.pdf')
    field = 'rho'; cmap = TS.CMAP
    D, angles, labels = AO.load_aggregated(OBS)         # same data + panel order as the original
    cols = labels
    grids = {}
    for a in angles:
        for l in cols:
            cand = glob.glob(os.path.join(OBS, 'grids', f'a{a}_{l}_lam*.csv'))
            if cand:
                G = AO.load_grid(cand[0])
                if G is not None:
                    grids[(a, l)] = G
    vmax = max(np.nanpercentile(G[field], 99) for G in grids.values())
    if not (np.isfinite(vmax) and vmax > 0):
        vmax = 1.0
    nrow, ncol = len(angles), len(cols)
    # colourbar now lives along the bottom -> the whole width goes to panels (wider panels).
    fig, axes = plt.subplots(nrow, ncol, squeeze=False, constrained_layout=True,
                             figsize=(1.5 * ncol, 1.45 * nrow + 1.0))
    funit = r'crowd density (ped m$^{-2}$)'
    im = None
    for i, a in enumerate(angles):
        xl, yl = AO._crop(a)
        for j, l in enumerate(cols):
            ax = axes[i][j]; G = grids.get((a, l))
            if i == 0: ax.set_title(AO._short(l), fontsize=10, fontweight='bold')   # was 8
            if j == 0: ax.set_ylabel(f'θ = {a}°', fontsize=12, fontweight='bold')    # was 10
            ax.set_xticks([]); ax.set_yticks([])
            if G is None:
                continue
            im = AO._imshow_clipped(ax, G[field], a, cmap, vmax)
            AO._overlay(ax, a, D.get((a, l)), wall_lw=1.5)
            ax.set_xlim(*xl); ax.set_ylim(*yl); ax.set_aspect('equal')
    if im is not None:
        cb = fig.colorbar(im, ax=axes, location='bottom', shrink=0.5, aspect=45, pad=0.02)
        cb.set_label(funit, fontsize=14, fontweight='bold')
        cb.ax.tick_params(labelsize=11)
    fig.suptitle('Pillar configurations, steady-state crowd density', fontsize=14, fontweight='bold')
    save(fig, 'obstacle_grid_density_revA')


# ===== Figure 2: coupled density montage (colourbar -> bottom; revA preserved) =
def fig_montage_revB():
    backup_current('bend_capacity_density_montage_revA.pdf')
    MC.OVERLAY_R = 0.25                     # chamfered overlay (capovCH geometry), as in revA
    field = 'rho'
    cells = MC._cells_from_grids(CAPCH)
    angles = sorted({a for a, d, p in cells}); doors = sorted({d for a, d, p in cells})
    vmax = 0.0; grids = {}
    for a, d, p in cells:
        Gg = MC.load_grid(p); grids[(a, d)] = Gg; vmax = max(vmax, np.nanpercentile(Gg[field], 99))
    nr, nc = len(angles), len(doors)
    # colourbar along the bottom -> wider panels (was a right-hand vertical bar in revA)
    fig, axes = plt.subplots(nr, nc, figsize=(1.75 * nc, 2.4 * nr + 0.7), constrained_layout=True)
    SUB = 13                                # bold larger row/column labels (preserved from revA)
    im = None
    for i, a in enumerate(angles):
        for j, d in enumerate(doors):
            ax = axes[i, j]; Gg = grids.get((a, d))
            if Gg is not None:
                im = ax.imshow(Gg[field], origin='lower', extent=[0, MC.DOMX, 0, MC.DOMY],
                               cmap=TS.CMAP, vmin=0, vmax=vmax)
                MC._overlay(ax, a, d)
            ax.set_xlim(0, MC.DOMX); ax.set_ylim(0, MC.DOMY); ax.set_aspect('equal')
            ax.set_xticks([]); ax.set_yticks([])
            if i == 0: ax.set_title(f'$d={d:g}$ m', fontsize=SUB, fontweight='bold')
            if j == 0: ax.set_ylabel(f'${int(a)}^\\circ$', fontsize=SUB, fontweight='bold')
    cb = fig.colorbar(im, ax=axes, location='bottom', shrink=0.5, aspect=45, pad=0.02)
    cb.set_label(r'density (ped m$^{-2}$)', fontsize=14, fontweight='bold')
    cb.ax.tick_params(labelsize=11)
    save(fig, 'bend_capacity_density_montage_revB')      # NO suptitle (preserved from revA)


def main():
    print('=== batch 2: move colourbar to bottom on two density grids ===')
    fig_obstacle_grid()
    fig_montage_revB()
    print('done. revA/revB PDFs (+ backups) in:\n  %s\nPNG previews in:\n  %s' % (THESIS_FIG, PREVIEW))


if __name__ == '__main__':
    main()
