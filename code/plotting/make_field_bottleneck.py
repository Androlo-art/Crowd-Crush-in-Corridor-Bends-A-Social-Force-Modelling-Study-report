"""Floor-field generator for the BEND + DOWNSTREAM BOTTLENECK scenario.

Copy of make_bend_field_param.py's logic, with TWO deliberate changes for the
crush study (originals untouched):

  1. LONG INLET: HB raised 8 -> 18 m so the straight approach before the bend is
     long at EVERY angle (the original geometry shrinks the inner inlet wall to
     ~2 m at 135 deg). Domain raised to 14 x 28 m to fit it (the C neighbour grid
     auto-resizes from XS/YS, and the field grid is written from the file header).
  2. DOOR EXIT: the FMM target is a central 1.2 m door at the outlet cap (not the
     whole open outlet), so agents funnel to a bottleneck and pressure builds.

It also PRINTS the exact outlet/door geometry per angle (O, n2, door endpoints,
Vin) — those numbers are what the C walls in sd_lib_bottleneck.c must mirror.

Usage:
  python -u bend_bottleneck/make_field_bottleneck.py --angles 0 45 90 135 \
         --width 5.0 --outdir results/bend_bottleneck_fields --preview
"""
import os, argparse, json
import numpy as np
import skfmm
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.path import Path

# Geometry — MUST match the C Init_Scenario (bottleneck). LONG inlet + tall domain.
HB = 18.0; LB = 12.0; CX = 5.0; H = 0.1   # LB 8->12: long outlet keeps the door jam clear of the inner-corner ROI at sharp angles
XSIZE = 18.0; YSIZE = 32.0                 # widened+heightened for the longer folded outlet
DOOR_W = 1.2                      # bottleneck door width (matches FIS room)


def line_intersect(P1, D1, P2, D2):
    A = np.array([[D1[0], -D2[0]], [D1[1], -D2[1]]])
    ab = np.linalg.solve(A, np.array(P2) - np.array(P1))
    return np.array(P1) + ab[0] * np.array(D1)


def masked_grad(D, inside, h):
    gx = np.zeros_like(D); gy = np.zeros_like(D)
    L = np.zeros_like(inside); R = np.zeros_like(inside)
    L[:, 1:] = inside[:, :-1]; R[:, :-1] = inside[:, 1:]
    Dl = np.zeros_like(D); Dr = np.zeros_like(D)
    Dl[:, 1:] = D[:, :-1]; Dr[:, :-1] = D[:, 1:]
    both = inside & L & R; onlyR = inside & R & ~L; onlyL = inside & L & ~R
    gx[both] = (Dr[both] - Dl[both]) / (2*h)
    gx[onlyR] = (Dr[onlyR] - D[onlyR]) / h
    gx[onlyL] = (D[onlyL] - Dl[onlyL]) / h
    Bm = np.zeros_like(inside); Tm = np.zeros_like(inside)
    Bm[1:, :] = inside[:-1, :]; Tm[:-1, :] = inside[1:, :]
    Db = np.zeros_like(D); Dt = np.zeros_like(D)
    Db[1:, :] = D[:-1, :]; Dt[:-1, :] = D[1:, :]
    bothy = inside & Bm & Tm; onlyT = inside & Tm & ~Bm; onlyB = inside & Bm & ~Tm
    gy[bothy] = (Dt[bothy] - Db[bothy]) / (2*h)
    gy[onlyT] = (Dt[onlyT] - D[onlyT]) / h
    gy[onlyB] = (D[onlyB] - Db[onlyB]) / h
    return gx, gy


def geometry(ang, W):
    """Return polygon + key points for turn angle `ang`, width W, long inlet."""
    th = np.radians(ang)
    d1 = np.array([0.0, 1.0]); d2 = np.array([np.sin(th), np.cos(th)])
    I = np.array([CX, 0.0]); B = np.array([CX, HB]); O = B + LB * d2
    n1 = np.array([1.0, 0.0]); n2 = np.array([np.cos(th), -np.sin(th)])  # inner-side lateral
    I_in, I_out = I + (W/2)*n1, I - (W/2)*n1
    O_in, O_out = O + (W/2)*n2, O - (W/2)*n2
    if ang < 0.5:
        O = np.array([CX, HB+LB]); d2 = np.array([0.0, 1.0]); n2 = np.array([1.0, 0.0])
        O_in = np.array([CX+W/2, HB+LB]); O_out = np.array([CX-W/2, HB+LB])
        poly = [(CX+W/2, 0.0), tuple(O_in), tuple(O_out), (CX-W/2, 0.0)]
        Vin = None
    else:
        Vin = line_intersect(I_in, d1, O_in, d2)
        Vout = line_intersect(I_out, d1, O_out, d2)
        poly = [tuple(I_in), tuple(Vin), tuple(O_in), tuple(O_out), tuple(Vout), tuple(I_out)]
    door_hi = O + (DOOR_W/2)*n2          # door endpoints (central 1.2 m of the cap)
    door_lo = O - (DOOR_W/2)*n2
    return dict(th=th, d2=d2, n2=n2, O=O, O_in=O_in, O_out=O_out,
                door_hi=door_hi, door_lo=door_lo, Vin=Vin, poly=poly)


def build(ang, W, outdir, preview=False):
    g = geometry(ang, W); O = g['O']; d2 = g['d2']; n2 = g['n2']
    Nx = int(round(XSIZE / H)); Ny = int(round(YSIZE / H))
    xs = (np.arange(Nx)+0.5)*H; ys = (np.arange(Ny)+0.5)*H
    XC, YC = np.meshgrid(xs, ys); P = np.stack([XC, YC], -1)
    inside = Path(g['poly']).contains_points(P.reshape(-1, 2)).reshape(Ny, Nx)
    if inside[:, 0].any() or inside[:, -1].any() or inside[-1, :].any():
        print(f"  WARNING a{ang}: corridor reaches a non-inlet grid edge — enlarge domain.")

    proj = (P - O) @ d2                      # along-outlet distance (0 at cap)
    lat = (P - O) @ n2                       # lateral position across the cap
    exitm = inside & (proj >= -2*H) & (np.abs(lat) <= DOOR_W/2)   # 1.2 m DOOR only
    if not exitm.any():
        print(f"  ERROR a{ang}: empty door exit mask!"); return g

    phi = np.ones((Ny, Nx)); phi[exitm] = -1.0
    phi = np.ma.MaskedArray(phi, ~inside)
    D = np.abs(np.ma.filled(skfmm.distance(phi, dx=H), np.nan))
    Df = np.where(inside & np.isfinite(D), D, 0.0)
    gx, gy = masked_grad(Df, inside & np.isfinite(D), H)
    U, V = -gx, -gy
    nrm = np.hypot(U, V); nrm[nrm == 0] = 1.0
    U, V = U/nrm, V/nrm
    U[exitm], V[exitm] = d2[0], d2[1]
    bad = (~inside) | (~np.isfinite(D))
    U[bad] = 0.0; V[bad] = 0.0

    os.makedirs(outdir, exist_ok=True)
    path = os.path.join(outdir, f'ff_field_a{ang}.csv')
    with open(path, 'w') as f:
        f.write(f'{Nx} {Ny} 0.0 0.0 {H}\n')
        for iy in range(Ny):
            for ix in range(Nx):
                f.write(f'{int(inside[iy, ix])} {U[iy, ix]:.4f} {V[iy, ix]:.4f}\n')
    print(f'wrote {path}  inside={int(inside.sum())} cells  exit(door)={int(exitm.sum())} cells')
    print(f'   a{ang}: O=({O[0]:.3f},{O[1]:.3f}) n2=({n2[0]:.3f},{n2[1]:.3f}) '
          f'O_in=({g["O_in"][0]:.3f},{g["O_in"][1]:.3f}) O_out=({g["O_out"][0]:.3f},{g["O_out"][1]:.3f})')
    print(f'        DOOR endpoints: lo=({g["door_lo"][0]:.3f},{g["door_lo"][1]:.3f}) '
          f'hi=({g["door_hi"][0]:.3f},{g["door_hi"][1]:.3f})'
          + (f'  Vin=({g["Vin"][0]:.3f},{g["Vin"][1]:.3f}) -> inner inlet wall length={g["Vin"][1]:.1f} m'
             if g['Vin'] is not None else ''))

    if preview:
        _preview(ang, W, g, inside, U, V, exitm, xs, ys, outdir)
    return g


def _preview(ang, W, g, inside, U, V, exitm, xs, ys, outdir):
    fig, ax = plt.subplots(figsize=(6, 9))
    ax.imshow(np.where(inside, 0.3, np.nan), origin='lower', extent=[0, XSIZE, 0, YSIZE],
              cmap='Blues', vmin=0, vmax=1, alpha=0.5)
    st = 6
    Xc, Yc = np.meshgrid(xs, ys)
    m = inside & ~exitm
    ax.quiver(Xc[::st, ::st], Yc[::st, ::st], np.where(m, U, 0)[::st, ::st],
              np.where(m, V, 0)[::st, ::st], scale=30, width=0.003, color='navy')
    # walls of the corridor poly
    poly = g['poly'] + [g['poly'][0]]
    px, py = zip(*poly)
    ax.plot(px, py, 'k-', lw=1.5)
    # outlet wall with the door gap (the bottleneck)
    O_in, O_out, dlo, dhi = g['O_in'], g['O_out'], g['door_lo'], g['door_hi']
    ax.plot([O_in[0], dhi[0]], [O_in[1], dhi[1]], 'r-', lw=4)
    ax.plot([O_out[0], dlo[0]], [O_out[1], dlo[1]], 'r-', lw=4)
    ax.plot([dlo[0], dhi[0]], [dlo[1], dhi[1]], 'lime', lw=4)   # the 1.2 m door
    ax.plot([g['O'][0]], [g['O'][1]], 'go', ms=6)
    ax.set(xlim=(0, XSIZE), ylim=(0, YSIZE), aspect='equal',
           title=f'bottleneck bend a={ang}°  (long inlet HB={HB} m, 1.2 m door)')
    ax.set_xlabel('x (m)'); ax.set_ylabel('y (m)')
    p = os.path.join(outdir, f'preview_a{ang}.png')
    fig.savefig(p, dpi=110, bbox_inches='tight'); plt.close(fig)
    print(f'   preview -> {p}')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--angles', type=float, nargs='+', default=[0, 45, 90, 135])
    ap.add_argument('--width', type=float, default=5.0)
    ap.add_argument('--outdir', default='results/bend_bottleneck_fields')
    ap.add_argument('--preview', action='store_true')
    ap.add_argument('--open', action='store_true',
                    help='OPEN bend: navigation target = the WHOLE outlet cap (egress width = corridor width), '
                         'i.e. the same geometry as the bottleneck but with the door removed (for the comparison).')
    args = ap.parse_args()
    if args.open:                       # door opens to the full corridor width -> open outlet
        globals()['DOOR_W'] = args.width
    print(f'{"OPEN" if args.open else "BOTTLENECK"} fields: HB={HB} LB={LB} domain={XSIZE}x{YSIZE} '
          f'egress={DOOR_W}m width={args.width}')
    geo = {}
    for a in args.angles:
        ai = int(a) if float(a).is_integer() else a
        g = build(ai, args.width, args.outdir, preview=args.preview)
        geo[str(ai)] = {k: (v.tolist() if hasattr(v, 'tolist') else v)
                        for k, v in g.items() if k in ('O', 'O_in', 'O_out', 'door_lo', 'door_hi', 'd2', 'n2')}
    with open(os.path.join(args.outdir, 'geometry.json'), 'w') as f:
        json.dump(dict(HB=HB, LB=LB, CX=CX, XSIZE=XSIZE, YSIZE=YSIZE, DOOR_W=DOOR_W,
                       width=args.width, angles=geo), f, indent=2)
    print('wrote geometry.json (C walls must mirror these numbers).')


if __name__ == '__main__':
    main()
