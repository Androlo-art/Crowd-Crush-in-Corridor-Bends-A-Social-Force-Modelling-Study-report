"""Floor-field (fast-marching navigation) figure for the report (v2).

NON-DESTRUCTIVE. Reads existing precomputed navigation fields and reuses the existing
chamfer-geometry builder; it WRITES ONLY a figure (pdf+png) to the report's v2 figures
folder. It does not modify, rename, or delete any simulation code, parameter files, or
field data.

It renders the desired-direction field  e^0 = -grad(D)/|grad(D)|  that the C solver
queries for ScenarioID=6 (ff_load_file in sd_lib_*.c), on the SAME chamfered bend
geometry used by the coupled/obstacle experiments (r=0.25 m inner fillet, 1.2 m door).
The arrows show the shortest-path field hugging the inner wall and turning the crowd
around the inner corner -- the mechanism behind the inner-wall density band.

The fields read here are exactly the CSVs the solver loads via SFM_BEND_FIELD_DIR,
written by make_field_capacity_chamfer.py with skfmm.distance (Eikonal solver).

Usage (defaults reproduce the report figure):
  .venv/bin/python -u bend_bottleneck/plot_floorfield_for_report.py
  # options:
  #   --angles 45 90 135
  #   --fielddir results/bend_capacity_fields/d12   (chamfered, 1.2 m door)
  #   --radius 0.25
"""
import os, sys, argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
import make_field_capacity_chamfer as CH          # geometry_rounded(ang, W, r) -- read-only use

OUT_DEFAULT = os.path.join(ROOT, '..', '..', 'figures', '_regenerated', 'bend_floorfield')


def load_field(path):
    """Read the field CSV: header 'Nx Ny xmin ymin H' then Ny*Nx rows 'inside U V'."""
    with open(path) as f:
        h = f.readline().split()
        Nx, Ny = int(h[0]), int(h[1])
        xmin, ymin, H = float(h[2]), float(h[3]), float(h[4])
        ins = np.zeros((Ny, Nx), bool); U = np.zeros((Ny, Nx)); V = np.zeros((Ny, Nx))
        for iy in range(Ny):
            for ix in range(Nx):
                a, u, v = f.readline().split()
                ins[iy, ix] = int(a) != 0; U[iy, ix] = float(u); V[iy, ix] = float(v)
    xs = xmin + (np.arange(Nx) + 0.5) * H
    ys = ymin + (np.arange(Ny) + 0.5) * H
    return xs, ys, ins, U, V, H


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--angles', type=int, nargs='+', default=[45, 90, 135])
    ap.add_argument('--fielddir', default=os.path.join(ROOT, '..', '..', 'results', '00_model', 'navigation_fields_d12'))
    ap.add_argument('--radius', type=float, default=0.25)
    ap.add_argument('--width', type=float, default=5.0)
    ap.add_argument('--crop', type=float, default=6.0)
    ap.add_argument('--out', default=OUT_DEFAULT)
    args = ap.parse_args()
    try:
        import thesis_style as TS; TS.apply(); figw = TS.FIGW
    except Exception:
        figw = 6.3
    rtag = int(args.radius * 10 + 0.5)
    n = len(args.angles)
    fig, axs = plt.subplots(1, n, figsize=(figw, figw / n * 1.55 + 0.5),
                            constrained_layout=True, squeeze=False)
    axs = axs[0]
    for k, (ax, ang) in enumerate(zip(axs, args.angles)):
        path = os.path.join(args.fielddir, f'ff_field_a{ang}_r{rtag}.csv')
        if not os.path.exists(path):
            ax.set_title(f'{ang} deg (field missing)'); ax.axis('off'); continue
        xs, ys, ins, U, V, H = load_field(path)
        g = CH.geometry_rounded(ang, args.width, args.radius)
        O = np.array(g['O']); n2 = np.array(g['n2'])
        # inner-corner vertex = crop centre; geometry_rounded stores it as a NumPy
        # array (so `a or b` is ambiguous). Use an explicit key check, and fall back
        # to the outlet centre O for a straight corridor (no inner vertex).
        vin = g.get('Vin_sharp')
        if vin is None:
            vin = g.get('Vin')
        vx, vy = (float(O[0]), float(O[1])) if vin is None else (float(vin[0]), float(vin[1]))
        XX, YY = np.meshgrid(xs, ys)
        m = ins & (np.abs(XX - vx) <= args.crop) & (np.abs(YY - vy) <= args.crop)
        Um = np.where(m, U, np.nan); Vm = np.where(m, V, np.nan)
        st = max(1, int(round(0.6 / H)))                       # ~0.6 m arrow spacing
        ax.quiver(XX[::st, ::st], YY[::st, ::st], Um[::st, ::st], Vm[::st, ::st],
                  scale=28, width=0.005, color='#1F3A60', pivot='mid', zorder=4)
        poly = list(g['poly']) + [g['poly'][0]]; px, py = zip(*poly)
        ax.plot(px, py, 'k-', lw=1.6, zorder=5)                # corridor walls + fillet arc
        dlo = O - 0.6 * n2; dhi = O + 0.6 * n2                  # 1.2 m door at the outlet cap
        ax.plot([dlo[0], dhi[0]], [dlo[1], dhi[1]], color='limegreen', lw=3.2, zorder=6)
        ax.set(xlim=(vx - args.crop, vx + args.crop), ylim=(vy - args.crop, vy + args.crop),
               aspect='equal')
        ax.set_title(rf'$\theta = {ang}^\circ$', fontsize=11)
        ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle(r'Fast-marching navigation field $\hat{e}^{\,0}=-\nabla D/|\nabla D|$ '
                 r'(chamfered bend; $1.2$\,m door in green)', fontsize=11)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    for ext in ('pdf', 'png'):
        fig.savefig(f'{args.out}.{ext}', bbox_inches='tight', dpi=150)
    plt.close(fig)
    print('wrote', args.out + '.pdf  and  .png')


if __name__ == '__main__':
    main()
