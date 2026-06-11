"""
plot_open_vs_restricted_density.py  (NEW, non-destructive helper for the v2 report)
===================================================================================
Report-ready, SHARED-COLOUR-SCALE comparison of the steady-state crowd-density
field in the OPEN bend versus the bend feeding a 1.2 m DOOR (restricted exit), for
Chapter 4.

Layout (stacked, full width):
  (a) Open bend / freely draining outlet     -- one row, all six angles
  (b) Restricted exit / 1.2 m door           -- one row, all six angles
  shared HORIZONTAL colourbar along the bottom

Row titles sit ABOVE each row; angle headings sit above each panel; the colourbar
is horizontal at the bottom so both rows use the FULL figure width (panels stay as
large as the original single-row density strips). One shared colour scale across
all panels. The open row draws NOTHING across its outlet (the open end is
unobstructed); only the restricted row draws the green 1.2 m door.

Why this script exists: the per-run density heatmaps the study already produces each
auto-scale to their OWN vmax, so they are not comparable by eye. This loads the
precomputed grids from BOTH runs and renders them on one shared scale. Using the
SAME demand (lambda) for both rows makes it a controlled comparison: only the open
end vs the 1.2 m door changes.

Reuses load_grid / geometry / wall-drawing / styling from plot_bottleneck.py and
make_field_bottleneck.py. READS ONLY existing grids/*.csv; runs no simulation. Does
NOT modify any existing project code or data. Writes into the report figures folder.

Run from the repo root:
  .venv/bin/python -u bend_bottleneck/plot_open_vs_restricted_density.py \
      --open-run       results/bend_bottleneck/openbendCAP_20260605_123719 \
      --restricted-run results/bend_bottleneck/phaseB_20260605_112610

Optional (defaults shown):
  --angles 0,45,60,90,120,135    --lam 8    --open-lam 8
  --out <path WITHOUT extension>   (default: the report figures folder)
  --fig-width 7.2 --row-title-size 13 --angle-size 12 --axis-size 10
  --tick-size 9 --cbar-size 12
"""
import os, sys, argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.path import Path as MPath
from matplotlib.patches import PathPatch
from matplotlib.ticker import FormatStrFormatter

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
import plot_bottleneck as PB          # reuse load_grid, _crop, _walls, constants, TS
import make_field_bottleneck as F
import thesis_style as TS

OUT_DEFAULT = os.path.join(ROOT, '..', '..', 'figures', '_regenerated', 'bend_open_vs_restricted_density')


def load_row(folder, angles, lam):
    grids, geoms = {}, {}
    for a in angles:
        ai = int(a)
        p = os.path.join(folder, 'grids', f'a{ai}_lam{lam}.csv')
        if os.path.exists(p):
            grids[ai] = PB.load_grid(p)
            geoms[ai] = F.geometry(ai, PB.WIDTH)
        else:
            print(f'  [warn] missing grid {p}')
    return grids, geoms


def panel(ax, g, gr, vmax, bx, by, dw, dh, open_mode, XX, YY):
    """One shared-scale density panel. open_mode=True draws NO line across the
    outlet (the open end is unobstructed); open_mode=False draws the green door."""
    walk = MPath(g['poly']).contains_points(
        np.column_stack([XX.ravel(), YY.ravel()])).reshape(PB.NY, PB.NX)
    Z = np.where(walk, gr['rho'], np.nan)
    cf = ax.contourf(PB.XC - bx, PB.YC - by, np.where(np.isfinite(Z), Z, 0.0),
                     levels=np.linspace(0, vmax, 25), cmap=TS.CMAP, extend='max')
    clip = PathPatch(MPath([(px - bx, py - by) for px, py in g['poly']]),
                     transform=ax.transData, fc='none', ec='none')
    ax.add_patch(clip)
    try:
        cf.set_clip_path(clip)
    except AttributeError:
        for c in cf.collections:
            c.set_clip_path(clip)
    for wpts in PB._walls(g):
        ax.plot([p[0] - bx for p in wpts], [p[1] - by for p in wpts],
                'k-', lw=1.6, zorder=5)
    if not open_mode:                                # restricted: walls flank the door
        ax.plot([g['O_in'][0] - bx, g['door_hi'][0] - bx],
                [g['O_in'][1] - by, g['door_hi'][1] - by], 'k-', lw=1.6, zorder=5)
        ax.plot([g['O_out'][0] - bx, g['door_lo'][0] - bx],
                [g['O_out'][1] - by, g['door_lo'][1] - by], 'k-', lw=1.6, zorder=5)
        ax.plot([g['door_lo'][0] - bx, g['door_hi'][0] - bx],
                [g['door_lo'][1] - by, g['door_hi'][1] - by], 'lime', lw=3.0, zorder=6)
    # (open_mode: draw nothing at the outlet -- the full-width end is unobstructed)
    if g['Vin'] is not None:
        vx, vy = g['Vin']
        ax.add_patch(plt.Rectangle((vx - PB.CORNER_HALF - bx, vy - PB.CORNER_HALF - by),
                                   2 * PB.CORNER_HALF, 2 * PB.CORNER_HALF, fill=False,
                                   ec='magenta', lw=1.3, ls='--', zorder=6))
    ax.set(xlim=(0, dw), ylim=(0, dh))
    ax.set_aspect('equal')
    return cf


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--open-run', required=True)
    ap.add_argument('--restricted-run', required=True)
    ap.add_argument('--angles', default='0,45,60,90,120,135')
    ap.add_argument('--lam', type=float, default=8.0,
                    help='demand for the restricted row (over-capacity level)')
    ap.add_argument('--open-lam', type=float, default=None,
                    help='demand for the open row (default = --lam, same demand)')
    ap.add_argument('--out', default=OUT_DEFAULT, help='output path WITHOUT extension')
    ap.add_argument('--panel-w', type=float, default=2.2,
                    help='native width (in) of ONE panel; ~2.2 matches the original density strips '
                         '(larger panels = higher-resolution feel; figure width = panel-w x ncol)')
    ap.add_argument('--tick-step', type=float, default=5.0, help='axis tick spacing in metres (0,5,10,...)')
    # Native sizes are large because the wide figure scales to ~0.45x at \textwidth;
    # these render near report-text size on the page. Titles/y-ticks/colourbar do not
    # narrow the panels (titles on top, y-ticks leftmost-only, colourbar at bottom).
    ap.add_argument('--row-title-size', type=float, default=22.0, help='(a)/(b) row-title size')
    ap.add_argument('--angle-size', type=float, default=20.0, help='per-panel angle heading size')
    ap.add_argument('--axis-size', type=float, default=17.0, help='x/y axis-label size')
    ap.add_argument('--tick-size', type=float, default=16.0, help='y-tick + colourbar-tick label size')
    ap.add_argument('--xtick-size', type=float, default=14.0,
                    help='x-tick label size (kept a little smaller to avoid crowding the bottom row)')
    ap.add_argument('--cbar-size', type=float, default=21.0, help='colourbar label size')
    args = ap.parse_args()

    TS.apply()
    angles = [int(float(x)) for x in args.angles.split(',')]
    open_lam = args.open_lam if args.open_lam is not None else args.lam
    openF = (args.open_run if os.path.isabs(args.open_run) else os.path.join(ROOT, args.open_run))
    restF = (args.restricted_run if os.path.isabs(args.restricted_run)
             else os.path.join(ROOT, args.restricted_run))

    og, ogeo = load_row(openF, angles, open_lam)
    rg, rgeo = load_row(restF, angles, args.lam)
    keys = [a for a in angles if a in og and a in rg]      # preserves requested order
    if not keys:
        print('ERROR: no angle has grids in BOTH runs at the requested demand.'); sys.exit(1)

    # one shared vmax across BOTH rows (99.5th percentile of density)
    vmax = 0.0
    for grids in (og, rg):
        for a in keys:
            v = grids[a]['rho'][np.isfinite(grids[a]['rho'])]
            if v.size:
                vmax = max(vmax, float(np.nanpercentile(v, 99.5)))
    vmax = max(vmax, 1e-6)

    # full-corridor crop, sized like the original single-row strips: a common box
    # (max extent), each angle's own crop centred within it.
    allgeo = {**rgeo, **ogeo}
    crops = {a: PB._crop(allgeo[a]) for a in keys}
    dw = max(x1 - x0 for x0, x1, y0, y1 in crops.values())
    dh = max(y1 - y0 for x0, x1, y0, y1 in crops.values())
    boxes = {a: (0.5 * (x0 + x1) - dw / 2, 0.5 * (y0 + y1) - dh / 2)
             for a, (x0, x1, y0, y1) in crops.items()}

    from matplotlib.ticker import MultipleLocator
    ncol = len(keys)
    panel_w = args.panel_w                           # native width of one panel (~original strips)
    panel_h = panel_w * (dh / dw)                    # full-corridor height at equal aspect
    figw = panel_w * ncol + 0.7                      # + left margin for the y-axis labels
    figh = 2.0 * panel_h + 2.0                       # two rows + (larger) titles + bottom colourbar
    fig = plt.figure(figsize=(figw, figh))
    sfs = fig.subfigures(3, 1, height_ratios=[panel_h, panel_h, 0.9], hspace=0.0)
    XX, YY = np.meshgrid(PB.XC, PB.YC)

    # (suptitle = row title on top; x-tick labels only on the bottom row, since both
    #  rows share the same x-box per column; y-tick labels only on the leftmost panel.)
    rows = [('(a) Open bend / freely draining outlet', og, ogeo, True, sfs[0], False),
            ('(b) Restricted exit / 1.2 m door',        rg, rgeo, False, sfs[1], True)]
    cf = None
    for rlab, grids, geoms, open_mode, sf, show_x in rows:
        sf.suptitle(rlab, y=0.99, fontsize=args.row_title_size, fontweight='bold')
        axs = np.atleast_1d(sf.subplots(1, ncol))
        sf.subplots_adjust(left=0.06, right=0.995, top=0.85,
                           bottom=(0.17 if show_x else 0.03), wspace=0.07)
        for ci, a in enumerate(keys):
            ax = axs[ci]
            bx, by = boxes[a]
            cf = panel(ax, geoms[a], grids[a], vmax, bx, by, dw, dh, open_mode, XX, YY)
            ax.set_title(f'${a}^\\circ$', fontsize=args.angle_size, fontweight='bold', pad=4)
            ax.xaxis.set_major_locator(MultipleLocator(args.tick_step))
            ax.yaxis.set_major_locator(MultipleLocator(args.tick_step))
            ax.tick_params(axis='y', labelsize=args.tick_size, labelleft=(ci == 0))
            ax.tick_params(axis='x', labelsize=args.xtick_size, labelbottom=show_x)
            if ci == 0:
                ax.set_ylabel('$y$ (m)', fontsize=args.axis_size)
            if show_x:
                ax.set_xlabel('$x$ (m)', fontsize=args.axis_size)

    cax = sfs[2].add_axes([0.25, 0.55, 0.5, 0.30])
    cb = fig.colorbar(cf, cax=cax, orientation='horizontal',
                      ticks=np.linspace(0.0, vmax, 9))           # 0.00 .. vmax, even spacing
    cb.ax.xaxis.set_major_formatter(FormatStrFormatter('%.2f'))  # 2 decimal places
    cb.set_label(r'Crowd density $\rho$ (ped m$^{-2}$)', fontsize=args.cbar_size, fontweight='bold')
    cb.ax.tick_params(labelsize=args.tick_size)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    for ext in ('pdf', 'png', 'svg'):
        fig.savefig(f'{args.out}.{ext}', bbox_inches='tight')
    plt.close(fig)
    print(f'wrote {args.out}.pdf / .png / .svg')
    print(f'  shared vmax = {vmax:.2f} ped/m^2 | angles = {keys} | '
          f'open lam={open_lam:.0f}, restricted lam={args.lam:.0f}')


if __name__ == '__main__':
    main()
