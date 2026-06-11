"""OPTION-D spatial diagnostics from SAVED grids (NO rerun).

Each capov run already stores steady-window (t in [200,400] s) 2-D fields per cell
in grids/a{ang}_d{dtag}_lam{lam}.csv  (cols x,y,rho,contact,casualty,pressure).
This script rasterises them and overlays the corridor walls + door so we can SEE,
without recomputing anything:

  * montage  : density (or contact/pressure) across the whole theta x door grid
               -> visually confirm the saturation/phase-boundary picture
               (door-limited cells pile density AT the door; bend-limited cells
               are dilute / free-flowing).
  * detail   : per representative cell, density + contact + pressure side by side
               -> judge whether contact maps are informative or too noisy, and
               whether the bend vs door crush location matches the story.

Writes only to --outdir (a NEW folder). Nothing is overwritten.

Usage:
  python -u bend_bottleneck/maps_capacity.py --run results/bend_capacity/capov_<stamp> \
      --outdir results/bend_capacity/capov_<stamp>/diag --montage --field rho
  python -u bend_bottleneck/maps_capacity.py --run results/bend_capacity/capov_<stamp> \
      --outdir results/bend_capacity/capov_<stamp>/diag \
      --cells "0,0.8;0,2.5;0,3.5;90,1.6;90,2.0;90,4.0;135,1.2"
"""
import os, sys, csv, glob, argparse, re
import numpy as np

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
import make_field_bottleneck as B          # geometry overlay (UNTOUCHED)

RES = 0.25; DOMX, DOMY = 18.0, 32.0
NX, NY = int(DOMX/RES), int(DOMY/RES)
WIDTH = 5.0
OVERLAY_R = 0.0          # >0 -> draw chamfered (filleted) walls (set by --radius for capovCH runs)


def _dtag(d): return int(round(d*10))


def load_grid(path):
    """Sparse grid csv -> full NY x NX arrays (NaN off-corridor)."""
    F = {k: np.full((NY, NX), np.nan) for k in ('rho', 'contact', 'casualty', 'pressure')}
    with open(path) as f:
        for r in csv.DictReader(f):
            ix = min(NX-1, int(float(r['x'])/RES)); iy = min(NY-1, int(float(r['y'])/RES))
            for k in F: F[k][iy, ix] = float(r[k])
    return F


def _overlay(ax, angle, door):
    B.DOOR_W = float(door)
    if OVERLAY_R > 0.05:
        import make_field_capacity_chamfer as FC
        g = FC.geometry_rounded(int(angle), WIDTH, OVERLAY_R)   # chamfered walls
    else:
        g = B.geometry(int(angle), WIDTH)
    poly = g['poly'] + [g['poly'][0]]; px, py = zip(*poly)
    ax.plot(px, py, 'k-', lw=1.0)
    dlo, dhi = g['door_lo'], g['door_hi']
    ax.plot([dlo[0], dhi[0]], [dlo[1], dhi[1]], color='lime', lw=2.5)   # the door
    if g['Vin'] is not None:
        vx, vy = g['Vin']
        ax.add_patch(__import__('matplotlib').patches.Rectangle(
            (vx-1.6, vy-1.6), 3.2, 3.2, fill=False, ec='magenta', lw=0.8, ls='--'))


def _cells_from_grids(run):
    cells = []
    for p in sorted(glob.glob(os.path.join(run, 'grids', 'a*_d*_lam*.csv'))):
        m = re.search(r'a(\d+)_d(\d+)_lam', os.path.basename(p))
        if m: cells.append((int(m.group(1)), int(m.group(2))/10.0, p))
    return cells


def montage(run, outdir, field='rho'):
    import matplotlib; matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import thesis_style as TS; TS.apply()
    cells = _cells_from_grids(run)
    angles = sorted({a for a, d, p in cells}); doors = sorted({d for a, d, p in cells})
    # global colour scale (99th pct) for fair comparison
    vmax = 0.0
    grids = {}
    for a, d, p in cells:
        G = load_grid(p); grids[(a, d)] = G
        vmax = max(vmax, np.nanpercentile(G[field], 99))
    os.makedirs(outdir, exist_ok=True)
    nr, nc = len(angles), len(doors)
    fig, axes = plt.subplots(nr, nc, figsize=(1.55*nc, 2.4*nr), constrained_layout=True)
    cmap = TS.CMAP
    for i, a in enumerate(angles):
        for j, d in enumerate(doors):
            ax = axes[i, j]; G = grids.get((a, d))
            if G is not None:
                im = ax.imshow(G[field], origin='lower', extent=[0, DOMX, 0, DOMY],
                               cmap=cmap, vmin=0, vmax=vmax)
                _overlay(ax, a, d)
            ax.set_xlim(0, DOMX); ax.set_ylim(0, DOMY); ax.set_aspect('equal')
            ax.set_xticks([]); ax.set_yticks([])
            if i == 0: ax.set_title(f'd={d:g} m', fontsize=9)
            if j == 0: ax.set_ylabel(f'{int(a)}°', fontsize=10)
    cb = fig.colorbar(im, ax=axes, shrink=0.6, location='right')
    lab = {'rho': r'density (ped m$^{-2}$)', 'contact': 'contact force (N kg$^{-1}$)',
           'pressure': r'pressure (ped m$^{-2}$ (m s$^{-1}$)$^2$)'}.get(field, field)
    cb.set_label(lab)
    fig.suptitle(f'{field} over the angle x door grid (steady window)  — door = lime', fontsize=11)
    out = os.path.join(outdir, f'map_montage_{field}')
    for ext in ('png', 'pdf'): fig.savefig(f'{out}.{ext}', bbox_inches='tight', dpi=140)
    plt.close(fig); print(f'wrote {out}.png/.pdf  (vmax={vmax:.2f})')


def detail(run, cells, outdir):
    import matplotlib; matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import thesis_style as TS; TS.apply()
    os.makedirs(outdir, exist_ok=True)
    fields = [('rho', r'density (ped m$^{-2}$)', 'turbo'),
              ('contact', 'contact (N kg$^{-1}$)', 'turbo'),
              ('pressure', 'pressure', 'turbo')]
    for (a, d) in cells:
        p = os.path.join(run, 'grids', f'a{int(a)}_d{_dtag(d)}_lam12.0.csv')
        if not os.path.exists(p): print(f'  missing grid {p}'); continue
        G = load_grid(p)
        fig, axes = plt.subplots(1, 3, figsize=(TS.FIGW, 0.5*TS.FIGW), constrained_layout=True)
        for ax, (k, lab, cm) in zip(axes, fields):
            im = ax.imshow(G[k], origin='lower', extent=[0, DOMX, 0, DOMY], cmap=cm,
                           vmin=0, vmax=np.nanpercentile(G[k], 99) or 1)
            _overlay(ax, a, d)
            ax.set_xlim(0, DOMX); ax.set_ylim(0, DOMY); ax.set_aspect('equal')
            ax.set_xticks([]); ax.set_yticks([]); ax.set_title(lab, fontsize=9)
            fig.colorbar(im, ax=ax, shrink=0.55)
        fig.suptitle(f'θ={int(a)}°  d={d:g} m  (steady window)', fontsize=11)
        out = os.path.join(outdir, f'map_detail_a{int(a)}_d{_dtag(d)}')
        for ext in ('png', 'pdf'): fig.savefig(f'{out}.{ext}', bbox_inches='tight', dpi=140)
        plt.close(fig); print(f'wrote {out}.png/.pdf')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--run', required=True)
    ap.add_argument('--outdir', required=True)
    ap.add_argument('--montage', action='store_true')
    ap.add_argument('--field', default='rho', choices=['rho', 'contact', 'pressure', 'casualty'])
    ap.add_argument('--cells', default=None, help='detail cells "ang,door;ang,door;..."')
    ap.add_argument('--radius', type=float, default=0.0, help='draw chamfered walls of this radius (e.g. 0.25 for capovCH)')
    args = ap.parse_args()
    globals()['OVERLAY_R'] = args.radius
    run = args.run if os.path.isabs(args.run) else os.path.join(ROOT, args.run)
    out = args.outdir if os.path.isabs(args.outdir) else os.path.join(ROOT, args.outdir)
    if args.montage:
        montage(run, out, args.field)
    if args.cells:
        cells = [(float(c.split(',')[0]), float(c.split(',')[1])) for c in args.cells.split(';') if c.strip()]
        detail(run, cells, out)
    if not args.montage and not args.cells:
        montage(run, out, args.field)


if __name__ == '__main__':
    main()
