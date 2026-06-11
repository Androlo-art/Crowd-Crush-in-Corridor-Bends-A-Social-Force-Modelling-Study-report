"""PILLAR-AWARE floor fields for the obstacle-mitigation ROBUSTNESS CHECK.

Identical to make_field_capacity_chamfer.build EXCEPT each configuration's pillar disc(s)
are masked OUT of the walkable region, so the fast-marching (FMM) navigation field routes
agents AROUND the pillars instead of straight at the door. The physical column force is
unchanged (still applied by sd_crunch_capacity_obs) — the ONLY difference from the
physical-only study is this field. This is the "agents can see the pillars" model; the
physical-only study is the dense-crush limit where navigation breaks down.

Geometry, door (1.2 m) and chamfer (r=0.25) are IDENTICAL to the physical study; only the
field changes. One field per (config, angle), written to
  results/bend_obstacle_navfields/<label>/ff_field_a<ang>_r<rtag>.csv
so the nav runner (run_capacity_obstacle_nav.py) selects the matching field per cell.
The pillar geometry per (config, angle) is taken VERBATIM from run_capacity_obstacle so the
masked discs sit exactly where the physical columns are. ctrl (no pillar) reproduces the
chamfered field byte-for-byte (a built-in equivalence check).

Nothing existing is touched; this writes only under --outroot.

Usage:
  full : python -u bend_bottleneck/make_field_obstacle.py --angles 0 45 60 90 120 135
  smoke: python -u bend_bottleneck/make_field_obstacle.py --angles 90 --cfgs ctrl gate
"""
import os, sys, argparse, json
import numpy as np
import skfmm
from matplotlib.path import Path

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT); sys.path.insert(0, ROOT)
import make_field_bottleneck as B            # FMM helpers + grid constants (UNTOUCHED)
import make_field_capacity_chamfer as FC     # geometry_rounded / field_name / rtag (UNTOUCHED)
import run_capacity_obstacle as R            # COLCFGS + cfg_centers (exact pillar geometry)

WIDTH = 5.0; CORNER_R = 0.25; DOOR = 1.2
OUTROOT = os.path.join(ROOT, 'results', 'bend_obstacle_navfields')


def build_nav(ang, cols, outdir, inflate=0.3):
    """FC.build, but with the pillar discs cols=[(cx,cy,D),...] removed from the walkable
    region so the field routes around them. `inflate` (m) grows each masked disc beyond the
    physical radius D/2:
      inflate = Dmean/2 = 0.3  -> a 'see-and-avoid' pedestrian who keeps a BODY radius clear
                                  of the pillar (the realistic navigation case, because the
                                  Helbing contact already engages at 0.5*(D_i+D_c));
      inflate = 0              -> the conservative 'route-around-but-graze' lower bound that
                                  lets agent centroids reach the bare disc edge.
    Returns (geometry, n_unreachable_cells)."""
    g = FC.geometry_rounded(ang, WIDTH, CORNER_R); O = g['O']; d2 = g['d2']; n2 = g['n2']
    Nx = int(round(B.XSIZE / B.H)); Ny = int(round(B.YSIZE / B.H))
    xs = (np.arange(Nx) + 0.5) * B.H; ys = (np.arange(Ny) + 0.5) * B.H
    XC, YC = np.meshgrid(xs, ys); P = np.stack([XC, YC], -1)
    inside = Path(g['poly']).contains_points(P.reshape(-1, 2)).reshape(Ny, Nx)
    colm = np.zeros((Ny, Nx), bool)
    for (cx, cy, Dc) in cols:
        rad = Dc / 2.0 + inflate
        colm |= (XC - cx)**2 + (YC - cy)**2 <= rad**2
    inside_nav = inside & ~colm
    proj = (P - O) @ d2; lat = (P - O) @ n2
    door_band = (proj >= -2 * B.H) & (np.abs(lat) <= DOOR / 2)
    if (colm & inside & door_band).any():
        print(f'  WARNING a{ang}: a masked pillar overlaps the exit band — door target shrunk!', flush=True)
    exitm = inside_nav & door_band
    if not exitm.any():
        print(f'  ERROR a{ang}: empty exit mask'); return g, -1
    phi = np.ones((Ny, Nx)); phi[exitm] = -1.0
    phi = np.ma.MaskedArray(phi, ~inside_nav)
    Dd = np.abs(np.ma.filled(skfmm.distance(phi, dx=B.H), np.nan))
    reach = inside_nav & np.isfinite(Dd)
    unreached = int((inside_nav & ~np.isfinite(Dd)).sum())
    if unreached > 0.01 * max(int(inside_nav.sum()), 1):
        print(f'  WARNING a{ang}: {unreached} walkable cells cannot reach the exit '
              f'(does a pillar block a route?) — check placement', flush=True)
    Df = np.where(reach, Dd, 0.0)
    gx, gy = B.masked_grad(Df, reach, B.H)
    U, V = -gx, -gy; nrm = np.hypot(U, V); nrm[nrm == 0] = 1.0; U, V = U / nrm, V / nrm
    U[exitm], V[exitm] = d2[0], d2[1]
    bad = (~inside_nav) | (~np.isfinite(Dd)); U[bad] = 0.0; V[bad] = 0.0
    os.makedirs(outdir, exist_ok=True)
    path = os.path.join(outdir, FC.field_name(ang, CORNER_R))
    with open(path, 'w') as f:
        f.write(f'{Nx} {Ny} 0.0 0.0 {B.H}\n')
        for iy in range(Ny):
            for ix in range(Nx):
                f.write(f'{int(reach[iy, ix])} {U[iy, ix]:.4f} {V[iy, ix]:.4f}\n')
    return g, unreached


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--angles', type=float, nargs='+', default=[0, 45, 60, 90, 120, 135])
    ap.add_argument('--cfgs', nargs='+', default=None, help='config labels (default: all)')
    ap.add_argument('--inflate', type=float, default=0.3,
                    help='grow each masked disc by this many m beyond D/2 (0.3 = body radius / '
                         'see-and-avoid [default]; 0 = conservative grazing lower bound)')
    ap.add_argument('--outroot', default=OUTROOT)
    args = ap.parse_args()
    cfgs = [c for c in R.COLCFGS if (args.cfgs is None or c['label'] in args.cfgs)]
    print(f'PILLAR-AWARE fields  cfgs={[c["label"] for c in cfgs]}  angles={args.angles}\n'
          f'  door={DOOR} corner_r={CORNER_R} width={WIDTH} inflate={args.inflate} m '
          f'-> {args.outroot}', flush=True)
    for cfg in cfgs:
        outdir = os.path.join(args.outroot, cfg['label']); os.makedirs(outdir, exist_ok=True)
        for a in args.angles:
            ai = int(a) if float(a).is_integer() else a
            cols = R.cfg_centers(ai, cfg)              # [] for ctrl
            g, unr = build_nav(ai, cols, outdir, inflate=args.inflate)
            tag = 'OK' if unr == 0 else (f'unreachable={unr}' if unr > 0 else 'ERROR')
            print(f'  {cfg["label"]:<15} a{ai}: {len(cols)} pillar(s) masked  [{tag}]', flush=True)
        json.dump({'egress': float(DOOR), 'radius': float(CORNER_R), 'config': cfg['label'],
                   'pillar_aware': True, 'inflate': float(args.inflate)},
                  open(os.path.join(outdir, 'geometry.json'), 'w'), indent=2)
    print('done. nav runner: run_capacity_obstacle_nav.py points SFM_BEND_FIELD_DIR at '
          'navfields/<label>/ per cell.', flush=True)


if __name__ == '__main__':
    main()
