"""Parametrised Fast-Marching (Eikonal) floor-field generator for the bend
corridor (ScenarioID=6).

This is an ADDITIVE companion to make_bend_field.py: it does NOT touch the
existing 3 m field set (results/bend_study/ff_field_a*.csv). It lets you
generate fields for an arbitrary corridor WIDTH into a SEPARATE directory, so
the canonical 5 m study and the legacy 3 m fields coexist.

The field grid (XSIZE x YSIZE, h) is identical to the C agent domain
(Init_Scenario_Bend: XS=14, YS=16). Width only changes the corridor mask and
the navigation gradient — not the grid — so the C floor-field loader needs no
change; point it at this output dir via SFM_BEND_FIELD_DIR.

Geometry MUST match Init_Scenario_Bend: HB=8, LB=8, Cx=5, domain 14x16, h=0.1.

Usage:
  # 5 m fields (angles 45..135 + rounded 90 deg) into results/bend_study_w5/
  python -u make_bend_field_param.py --width 5.0 --outdir results/bend_study_w5 \
         --angles 45 60 90 120 135 --round90 1.0 2.0
  # reproduce the legacy 3 m set (sanity check; writes elsewhere)
  python -u make_bend_field_param.py --width 3.0 --outdir /tmp/ff_check
"""
import os
import argparse
import numpy as np
import skfmm
from matplotlib.path import Path

# Fixed grid + arm geometry — MUST match C Init_Scenario_Bend.
HB = 8.0; LB = 8.0; CX = 5.0; H = 0.1
XSIZE = 14.0; YSIZE = 16.0


def line_intersect(P1, D1, P2, D2):
    A = np.array([[D1[0], -D2[0]], [D1[1], -D2[1]]])
    ab = np.linalg.solve(A, np.array(P2) - np.array(P1))
    return np.array(P1) + ab[0] * np.array(D1)


def masked_grad(D, inside, h):
    """Mask-aware gradient (never references an outside-corridor cell), so
    boundary cells stay wall-tangent instead of pointing into the wall."""
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


def build(ang, width, outdir, round_r=0.0):
    """Build & write the FMM field for turn angle `ang` at corridor `width`.
    Filenames are width-independent (ff_field_a{ang}[_r{R*10}].csv) so the C
    loader finds them via SFM_BEND_FIELD_DIR; width is encoded by the OUTPUT
    DIRECTORY, not the filename."""
    W = float(width)
    th = np.radians(ang)
    d1 = np.array([0.0, 1.0]); d2 = np.array([np.sin(th), np.cos(th)])
    I = np.array([CX, 0.0]); B = np.array([CX, HB]); O = B + LB * d2
    n1 = np.array([1.0, 0.0]); n2 = np.array([np.cos(th), -np.sin(th)])
    I_in, I_out = I + (W/2)*n1, I - (W/2)*n1
    O_in, O_out = O + (W/2)*n2, O - (W/2)*n2
    Vin = None
    if ang < 0.5:
        O = np.array([CX, HB+LB]); d2 = np.array([0.0, 1.0])
        O_in = np.array([CX+W/2, HB+LB]); O_out = np.array([CX-W/2, HB+LB])
        poly = [(CX+W/2, 0), O_in, O_out, (CX-W/2, 0)]
    else:
        Vin = line_intersect(I_in, d1, O_in, d2)
        Vout = line_intersect(I_out, d1, O_out, d2)
        poly = [tuple(I_in), tuple(Vin), tuple(O_in), tuple(O_out), tuple(Vout), tuple(I_out)]

    Nx = int(round(XSIZE / H)); Ny = int(round(YSIZE / H))
    xs = (np.arange(Nx)+0.5)*H; ys = (np.arange(Ny)+0.5)*H
    XC, YC = np.meshgrid(xs, ys); P = np.stack([XC, YC], -1)
    inside = Path(poly).contains_points(P.reshape(-1, 2)).reshape(Ny, Nx)

    # Sanity: warn only if the corridor reaches the LEFT/RIGHT/TOP edge (true
    # clipping). The BOTTOM row (y=0) is the open inlet and always touches.
    if inside[:, 0].any() or inside[:, -1].any() or inside[-1, :].any():
        print(f"  WARNING a{ang} w{W}: corridor reaches a non-inlet grid edge — "
              f"domain (14x16) may be too small for this width/angle; check geometry.")

    if round_r > 0.0 and Vin is not None:
        ccx, ccy = Vin[0] + round_r, Vin[1] - round_r
        box = ((XC >= Vin[0]) & (XC <= Vin[0]+round_r) &
               (YC >= Vin[1]-round_r) & (YC <= Vin[1]))
        wedge = box & (((XC-ccx)**2 + (YC-ccy)**2) >= round_r**2)
        inside = inside | wedge
    proj = (P - O) @ d2
    exitm = inside & (proj >= -2*H)

    phi = np.ones((Ny, Nx)); phi[exitm] = -1.0
    phi = np.ma.MaskedArray(phi, ~inside)
    D = np.abs(np.ma.filled(skfmm.distance(phi, dx=H), np.nan))
    Df = np.where(inside & np.isfinite(D), D, 0.0)
    gx, gy = masked_grad(Df, inside & np.isfinite(D), H)
    U, V = -gx, -gy
    n = np.hypot(U, V); n[n == 0] = 1.0
    U, V = U/n, V/n
    U[exitm], V[exitm] = d2[0], d2[1]
    bad = (~inside) | (~np.isfinite(D))
    U[bad] = 0.0; V[bad] = 0.0

    suffix = f'_r{int(round(round_r*10))}' if round_r > 0.0 else ''
    os.makedirs(outdir, exist_ok=True)
    path = os.path.join(outdir, f'ff_field_a{ang}{suffix}.csv')
    with open(path, 'w') as f:
        f.write(f'{Nx} {Ny} 0.0 0.0 {H}\n')
        for iy in range(Ny):
            for ix in range(Nx):
                f.write(f'{int(inside[iy, ix])} {U[iy, ix]:.4f} {V[iy, ix]:.4f}\n')
    print(f'wrote {path}  (width={W} m, inside={int(inside.sum())} cells)')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--width', type=float, default=5.0,
                    help='corridor width (m). 5.0 = Moussaid Fig.S4 match (default).')
    ap.add_argument('--outdir', default='results/bend_study_w5',
                    help='output directory (point SFM_BEND_FIELD_DIR here at run time).')
    ap.add_argument('--angles', type=int, nargs='+', default=[45, 60, 90, 120, 135],
                    help='turn angles (deg) to generate sharp-corner fields for.')
    ap.add_argument('--round90', type=float, nargs='*', default=[1.0, 2.0],
                    help='inner-corner rounding radii (m) for the 90 deg field (mitigation study).')
    args = ap.parse_args()

    root = os.path.dirname(os.path.abspath(__file__))
    outdir = args.outdir if os.path.isabs(args.outdir) else os.path.join(root, args.outdir)
    print(f'Generating bend floor fields: width={args.width} m -> {outdir}')
    for ang in args.angles:
        build(ang, args.width, outdir)
    for r in (args.round90 or []):
        build(90, args.width, outdir, round_r=r)
    print('done.')


if __name__ == '__main__':
    main()
