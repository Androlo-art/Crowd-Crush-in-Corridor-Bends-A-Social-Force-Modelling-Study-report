"""GENERAL-ANGLE inner-corner fillet floor fields for the chamfered Option-D study.

Unlike make_field_chamfer.py (90 deg ONLY), this rounds the inner corner at ANY
turn angle, using the analytic fillet tangent to both inner walls:

    tangent distance  t = r*tan(theta/2)         (along each inner wall from Vin)
    fillet centre     C = (Vinx + r, Viny - t)
    arc span          [pi - theta, pi]            (radius r, centred at C)
    tangent points    t1 = (Vinx, Viny - t)   on arm-1 inner wall
                      t2 = Vin + t*d2          on arm-2 inner wall

These MUST match the C walls in sd_lib_capacity_ch.c (same formula) so the
navigation field and the physical wall/contact geometry agree.

Door width is carried by the per-door FOLDER (d{int(d*10)}/), exactly like
make_field_capacity.py; the corner radius is carried by the FILENAME tag
ff_field_a{ang}_r{int(r*10+0.5)}.csv (what the C ff_load_file expects when
BendCornerR>0.05). Straight (ang<0.5) writes the flat field under the _r tag too.

Usage:
  python -u bend_bottleneck/make_field_capacity_chamfer.py \
      --angles 0 45 60 90 120 135 --doors 0.8 1.2 1.6 2.0 2.5 3.0 3.5 4.0 \
      --radius 0.25 --width 5.0 --outroot results/bend_capacity_fields --preview
"""
import os, sys, argparse, json
import numpy as np
import skfmm
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.path import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import make_field_bottleneck as B          # FMM helpers + constants (UNTOUCHED)

ARC_N = 41                                  # arc sample points (smooth even at 135 deg)


def rtag(r):
    return int(r * 10.0 + 0.5)              # matches C ff_load_file int(BendCornerR*10+0.5)


def dtag(d):
    return int(round(d * 10.0))


def field_name(ang, r):
    return f'ff_field_a{ang}_r{rtag(r)}.csv'


def geometry_rounded(ang, W, r):
    """B.geometry(ang,W) with the inner corner replaced by a general-angle fillet."""
    g = B.geometry(ang, W)
    g['Vin_sharp'] = g['Vin']
    if ang < 0.5 or r <= 0.05 or g['Vin'] is None:
        g['corner_r'] = 0.0; g['arc_centre'] = None
        return g                            # straight / no fillet -> flat geometry
    th = np.radians(ang)
    d2 = np.array([np.sin(th), np.cos(th)])
    Vin = np.array(g['Vin'], float)
    t = r * np.tan(th / 2.0)                # tangent distance
    C = np.array([Vin[0] + r, Vin[1] - t])  # fillet centre
    t1 = np.array([Vin[0], Vin[1] - t])     # tangent point on arm-1 inner (x = Vinx)
    t2 = Vin + t * d2                        # tangent point on arm-2 inner
    phi = np.linspace(np.pi, np.pi - th, ARC_N)            # t1 (pi) -> t2 (pi-th)
    arc = [(C[0] + r * np.cos(p), C[1] + r * np.sin(p)) for p in phi]
    p = g['poly']                           # [I_in, Vin, O_in, O_out, Vout, I_out]
    g['poly'] = [p[0]] + arc + [p[2], p[3], p[4], p[5]]
    g['corner_r'] = float(r); g['arc_centre'] = (float(C[0]), float(C[1]))
    g['t1'] = (float(t1[0]), float(t1[1])); g['t2'] = (float(t2[0]), float(t2[1]))
    g['tan_dist'] = float(t)
    return g


def build(ang, W, r, door_w, outdir, preview=False):
    g = geometry_rounded(ang, W, r); O = g['O']; d2 = g['d2']; n2 = g['n2']
    Nx = int(round(B.XSIZE / B.H)); Ny = int(round(B.YSIZE / B.H))
    xs = (np.arange(Nx) + 0.5) * B.H; ys = (np.arange(Ny) + 0.5) * B.H
    XC, YC = np.meshgrid(xs, ys); P = np.stack([XC, YC], -1)
    inside = Path(g['poly']).contains_points(P.reshape(-1, 2)).reshape(Ny, Nx)
    if inside[:, 0].any() or inside[:, -1].any() or inside[-1, :].any():
        print(f"  WARNING a{ang} r{r}: corridor reaches a non-inlet grid edge.")
    proj = (P - O) @ d2; lat = (P - O) @ n2
    exitm = inside & (proj >= -2 * B.H) & (np.abs(lat) <= door_w / 2)
    if not exitm.any():
        print(f"  ERROR a{ang} r{r}: empty exit mask!"); return g
    phi = np.ones((Ny, Nx)); phi[exitm] = -1.0
    phi = np.ma.MaskedArray(phi, ~inside)
    D = np.abs(np.ma.filled(skfmm.distance(phi, dx=B.H), np.nan))
    Df = np.where(inside & np.isfinite(D), D, 0.0)
    gx, gy = B.masked_grad(Df, inside & np.isfinite(D), B.H)
    U, V = -gx, -gy; nrm = np.hypot(U, V); nrm[nrm == 0] = 1.0
    U, V = U / nrm, V / nrm
    U[exitm], V[exitm] = d2[0], d2[1]
    bad = (~inside) | (~np.isfinite(D)); U[bad] = 0.0; V[bad] = 0.0
    os.makedirs(outdir, exist_ok=True)
    path = os.path.join(outdir, field_name(ang, r))
    with open(path, 'w') as f:
        f.write(f'{Nx} {Ny} 0.0 0.0 {B.H}\n')
        for iy in range(Ny):
            for ix in range(Nx):
                f.write(f'{int(inside[iy, ix])} {U[iy, ix]:.4f} {V[iy, ix]:.4f}\n')
    if preview:
        _preview(ang, r, g, inside, U, V, exitm, xs, ys, outdir, door_w)
    return g


def _preview(ang, r, g, inside, U, V, exitm, xs, ys, outdir, door_w):
    fig, ax = plt.subplots(figsize=(6, 9))
    ax.imshow(np.where(inside, 0.3, np.nan), origin='lower', extent=[0, B.XSIZE, 0, B.YSIZE],
              cmap='Blues', vmin=0, vmax=1, alpha=0.5)
    poly = g['poly'] + [g['poly'][0]]; px, py = zip(*poly)
    ax.plot(px, py, 'k-', lw=1.4)
    if g['arc_centre'] is not None:
        ax.plot([g['arc_centre'][0]], [g['arc_centre'][1]], 'mx', ms=7)
        vx, vy = g['Vin_sharp']
        ax.add_patch(plt.Rectangle((vx - 1.6, vy - 1.6), 3.2, 3.2, fill=False, ec='magenta', lw=1, ls='--'))
    ax.set(xlim=(0, B.XSIZE), ylim=(0, B.YSIZE), aspect='equal',
           title=f'a={ang} r={r} m door={door_w} m  (general fillet)')
    p = os.path.join(outdir, f'preview_a{ang}_r{rtag(r)}.png')
    fig.savefig(p, dpi=110, bbox_inches='tight'); plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--angles', type=float, nargs='+', default=[0, 45, 60, 90, 120, 135])
    ap.add_argument('--doors', type=float, nargs='+', default=[0.8, 1.2, 1.6, 2.0, 2.5, 3.0, 3.5, 4.0])
    ap.add_argument('--radius', type=float, default=0.25)
    ap.add_argument('--width', type=float, default=5.0)
    ap.add_argument('--outroot', default='results/bend_capacity_fields')
    ap.add_argument('--preview', action='store_true')
    args = ap.parse_args()
    W = args.width; r = args.radius
    print(f'CHAMFER fields r={r} (tag {rtag(r)}): angles={args.angles} doors={args.doors} width={W}')
    for d in args.doors:
        outdir = os.path.join(args.outroot, f'd{dtag(d)}'); os.makedirs(outdir, exist_ok=True)
        geo = {}
        for a in args.angles:
            ai = int(a) if float(a).is_integer() else a
            g = build(ai, W, r, d, outdir, preview=args.preview)
            geo[str(ai)] = {k: (v if not hasattr(v, 'tolist') else v.tolist())
                            for k, v in g.items()
                            if k in ('arc_centre', 'corner_r', 't1', 't2', 'tan_dist', 'Vin_sharp')}
        # merge into the door folder's geometry.json (keep egress already there if present)
        gj = os.path.join(outdir, 'geometry.json')
        base = json.load(open(gj)) if os.path.exists(gj) else {}
        base.setdefault('egress', float(d)); base['radius'] = float(r); base['chamfer_angles'] = geo
        json.dump(base, open(gj, 'w'), indent=2)
        print(f'  door {d} m -> {outdir}  ({field_name(int(args.angles[-1]), r)} etc.)', flush=True)
    print('done. C (sd_crunch_capacity_ch, BendCornerR=r) loads ff_field_a{ang}_r{tag}.csv from d{tag}/.')


if __name__ == '__main__':
    main()
