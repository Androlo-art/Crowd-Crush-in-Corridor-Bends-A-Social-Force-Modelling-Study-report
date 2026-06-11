"""Floor-field generator for the ROUNDED / CHAMFERED inner-corner study (90 deg bend).

Natural continuation of make_field_bottleneck.py: the SAME long-inlet / long-outlet
bend geometry (HB=18, LB=12, width 5 m, 1.2 m door OR open cap), but with the INNER
CORNER replaced by a quarter-circle fillet of radius r. The matching C wall arc is
ALREADY built by sd_lib_bottleneck.c / sd_lib_openbend.c when BendCornerR>0 at 90 deg,
and the solver loads exactly the file this script writes:

    r == 0 :  ff_field_a90.csv            (sharp baseline; identical to make_field_bottleneck)
    r >  0 :  ff_field_a90_r{int(r*10+0.5)}.csv

so r=0.5 -> ff_field_a90_r5.csv, r=1.0 -> _r10, r=1.5 -> _r15, r=2.0 -> _r20.

ONLY 90 deg is supported (the C arc is 90-deg-only). The originals
(make_field_bottleneck.py, sd_lib_*.c) are untouched; this file imports the former
to reuse its geometry constants and helpers.

Usage:
  # BOTTLENECK (1.2 m door) fields, radii 0..2.0:
  python -u bend_bottleneck/make_field_chamfer.py \
      --radii 0 0.5 1.0 1.5 2.0 --width 5.0 \
      --outdir results/bend_chamfer_fields_bottleneck --preview
  # OPEN (full-cap egress) fields:
  python -u bend_bottleneck/make_field_chamfer.py \
      --radii 0 0.5 1.0 1.5 2.0 --width 5.0 --open \
      --outdir results/bend_chamfer_fields_open --preview
"""
import os, sys, argparse, json
import numpy as np
import skfmm
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.path import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import make_field_bottleneck as B          # reuse constants + helpers (UNTOUCHED)

ANGLE = 90                                  # the C rounded-corner arc is 90-deg-only
ARC_N = 33                                  # points sampled along the quarter-arc


def rtag(r):
    """C field-file radius tag: int(r*10+0.5) — matches sd_lib ff_load_file()."""
    return int(r * 10.0 + 0.5)


def field_name(r):
    return f'ff_field_a{ANGLE}.csv' if r <= 0.05 else f'ff_field_a{ANGLE}_r{rtag(r)}.csv'


def geometry_rounded(W, r):
    """90 deg bend geometry with the inner corner replaced by a radius-r fillet.

    Returns B.geometry(90, W) with 'poly' rounded, plus:
      Vin_sharp  : the ORIGINAL sharp inner vertex (used as the fixed corner ROI)
      corner_r   : r
      arc_centre : fillet centre (None if sharp)
      t1, t2     : arc tangent points on the two inner walls
    The fillet matches the C arc in sd_lib_*.c exactly: centre (Vinx+r, Viny-r),
    quarter-arc from angle pi (t1 on arm-1 inner) to pi/2 (t2 on arm-2 inner).
    """
    g = B.geometry(ANGLE, W)
    Vin = np.array(g['Vin']); d2 = np.array(g['d2'])      # 90 deg: d2 = (1, 0)
    d1 = np.array([0.0, 1.0])                             # arm-1 inner runs +y up to Vin
    g['Vin_sharp'] = (float(Vin[0]), float(Vin[1]))
    if r <= 0.05:
        g['corner_r'] = 0.0; g['arc_centre'] = None
        return g
    t1 = Vin - r * d1                       # on arm-1 inner wall: (Vinx,   Viny-r)
    t2 = Vin + r * d2                       # on arm-2 inner wall: (Vinx+r, Viny)
    centre = Vin + r * (d2 - d1)            # 90 deg -> (Vinx+r, Viny-r)  == C arc centre
    phi = np.linspace(np.pi, 0.5 * np.pi, ARC_N)
    arc = [(centre[0] + r * np.cos(p), centre[1] + r * np.sin(p)) for p in phi]
    p = g['poly']                          # [I_in, Vin, O_in, O_out, Vout, I_out]
    g['poly'] = [p[0]] + arc + [p[2], p[3], p[4], p[5]]
    g['corner_r'] = float(r); g['arc_centre'] = (float(centre[0]), float(centre[1]))
    g['t1'] = (float(t1[0]), float(t1[1])); g['t2'] = (float(t2[0]), float(t2[1]))
    return g


def build(W, r, outdir, door_w, preview=False):
    g = geometry_rounded(W, r); O = g['O']; d2 = g['d2']; n2 = g['n2']
    Nx = int(round(B.XSIZE / B.H)); Ny = int(round(B.YSIZE / B.H))
    xs = (np.arange(Nx) + 0.5) * B.H; ys = (np.arange(Ny) + 0.5) * B.H
    XC, YC = np.meshgrid(xs, ys); P = np.stack([XC, YC], -1)
    inside = Path(g['poly']).contains_points(P.reshape(-1, 2)).reshape(Ny, Nx)
    if inside[:, 0].any() or inside[:, -1].any() or inside[-1, :].any():
        print(f"  WARNING r{r}: corridor reaches a non-inlet grid edge — enlarge domain.")
    proj = (P - O) @ d2                     # along-outlet distance (0 at cap)
    lat = (P - O) @ n2                      # lateral position across the cap
    exitm = inside & (proj >= -2 * B.H) & (np.abs(lat) <= door_w / 2)
    if not exitm.any():
        print(f"  ERROR r{r}: empty exit mask!"); return g
    phi = np.ones((Ny, Nx)); phi[exitm] = -1.0
    phi = np.ma.MaskedArray(phi, ~inside)
    D = np.abs(np.ma.filled(skfmm.distance(phi, dx=B.H), np.nan))
    Df = np.where(inside & np.isfinite(D), D, 0.0)
    gx, gy = B.masked_grad(Df, inside & np.isfinite(D), B.H)
    U, V = -gx, -gy
    nrm = np.hypot(U, V); nrm[nrm == 0] = 1.0
    U, V = U / nrm, V / nrm
    U[exitm], V[exitm] = d2[0], d2[1]
    bad = (~inside) | (~np.isfinite(D))
    U[bad] = 0.0; V[bad] = 0.0

    os.makedirs(outdir, exist_ok=True)
    path = os.path.join(outdir, field_name(r))
    with open(path, 'w') as f:
        f.write(f'{Nx} {Ny} 0.0 0.0 {B.H}\n')
        for iy in range(Ny):
            for ix in range(Nx):
                f.write(f'{int(inside[iy, ix])} {U[iy, ix]:.4f} {V[iy, ix]:.4f}\n')
    print(f'wrote {path}  inside={int(inside.sum())} cells  exit={int(exitm.sum())} cells  '
          f'r={r}  arc_centre={g.get("arc_centre")}')
    if preview:
        _preview(W, r, g, inside, U, V, exitm, xs, ys, outdir, door_w)
    return g


def _preview(W, r, g, inside, U, V, exitm, xs, ys, outdir, door_w):
    fig, ax = plt.subplots(figsize=(6, 9))
    ax.imshow(np.where(inside, 0.3, np.nan), origin='lower', extent=[0, B.XSIZE, 0, B.YSIZE],
              cmap='Blues', vmin=0, vmax=1, alpha=0.5)
    st = 6; Xc, Yc = np.meshgrid(xs, ys); m = inside & ~exitm
    ax.quiver(Xc[::st, ::st], Yc[::st, ::st], np.where(m, U, 0)[::st, ::st],
              np.where(m, V, 0)[::st, ::st], scale=30, width=0.003, color='navy')
    poly = g['poly'] + [g['poly'][0]]; px, py = zip(*poly)
    ax.plot(px, py, 'k-', lw=1.5)
    O_in, O_out, dlo, dhi = g['O_in'], g['O_out'], g['door_lo'], g['door_hi']
    if door_w >= W - 1e-6:                                  # OPEN cap
        ax.plot([O_in[0], O_out[0]], [O_in[1], O_out[1]], '--', color='0.45', lw=1.6)
    else:                                                   # 1.2 m door
        ax.plot([O_in[0], dhi[0]], [O_in[1], dhi[1]], 'r-', lw=4)
        ax.plot([O_out[0], dlo[0]], [O_out[1], dlo[1]], 'r-', lw=4)
        ax.plot([dlo[0], dhi[0]], [dlo[1], dhi[1]], 'lime', lw=4)
    vx, vy = g['Vin_sharp']
    ax.add_patch(plt.Rectangle((vx - 1.6, vy - 1.6), 3.2, 3.2, fill=False, ec='magenta', lw=1.2, ls='--'))
    if g['arc_centre'] is not None:
        ax.plot([g['arc_centre'][0]], [g['arc_centre'][1]], 'mx', ms=7)
    ax.set(xlim=(0, B.XSIZE), ylim=(0, B.YSIZE), aspect='equal',
           title=f'{"open" if door_w>=W-1e-6 else "bottleneck"} 90 deg bend  r={r} m')
    ax.set_xlabel('x (m)'); ax.set_ylabel('y (m)')
    p = os.path.join(outdir, f'preview_a{ANGLE}_r{rtag(r)}.png')
    fig.savefig(p, dpi=110, bbox_inches='tight'); plt.close(fig)
    print(f'   preview -> {p}')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--radii', type=float, nargs='+', default=[0.0, 0.5, 1.0, 1.5, 2.0])
    ap.add_argument('--width', type=float, default=5.0)
    ap.add_argument('--outdir', default='results/bend_chamfer_fields_bottleneck')
    ap.add_argument('--open', action='store_true',
                    help='OPEN bend: navigation target = the WHOLE outlet cap (egress=width).')
    ap.add_argument('--preview', action='store_true')
    args = ap.parse_args()
    door_w = args.width if args.open else B.DOOR_W
    print(f'{"OPEN" if args.open else "BOTTLENECK"} chamfer fields (90 deg): '
          f'radii={args.radii} egress={door_w}m width={args.width} HB={B.HB} LB={B.LB} '
          f'domain={B.XSIZE}x{B.YSIZE}')
    geo = {}
    for r in args.radii:
        g = build(args.width, r, args.outdir, door_w, preview=args.preview)
        geo[f'{r:g}'] = {k: (v.tolist() if hasattr(v, 'tolist') else v)
                         for k, v in g.items()
                         if k in ('O', 'O_in', 'O_out', 'door_lo', 'door_hi', 'd2', 'n2',
                                  'Vin_sharp', 'arc_centre', 'corner_r', 't1', 't2')}
    with open(os.path.join(args.outdir, 'geometry.json'), 'w') as f:
        json.dump(dict(angle=ANGLE, HB=B.HB, LB=B.LB, CX=B.CX, XSIZE=B.XSIZE, YSIZE=B.YSIZE,
                       egress=door_w, width=args.width, radii=args.radii, fields=geo), f, indent=2)
    print('wrote geometry.json  (C walls mirror these; r=0 -> ff_field_a90.csv, r>0 -> _r{R*10})')


if __name__ == '__main__':
    main()
