"""Fast-Marching (Eikonal) floor field for the bend corridor — the proper fix
for the Dijkstra metrication / corner-singularity artifacts.

For each turn angle: build the corridor (constant-width tube around the bent
centreline), solve |grad D| = 1 with D=0 at the outlet and walls masked
(skfmm.distance), then desired direction = -grad(D)/|grad(D)|. The gradient of
the true geodesic distance is continuous and isotropic, so arrows point at ANY
angle (track a 50 deg corridor exactly) with no banding and no corner swirl.

Top row: Dijkstra + min-delta (current C method).  Bottom row: FMM.
"""
import os, heapq
import numpy as np
import matplotlib.pyplot as plt
import skfmm

W = 3.0; HB = 8.0; LB = 8.0; CX = 5.0; H = 0.1
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results', 'bend_study')
os.makedirs(OUT, exist_ok=True)
ANGLES = [45, 90, 135]
N8 = [(1,0),(-1,0),(0,1),(0,-1),(1,1),(1,-1),(-1,1),(-1,-1)]


def grids(theta):
    th = np.radians(theta); d2 = np.array([np.sin(th), np.cos(th)])
    I = np.array([CX, 0.0]); B = np.array([CX, HB]); O = B + LB*d2
    pts = np.array([I, B, O]); pad = W
    xmin, xmax = pts[:,0].min()-pad, pts[:,0].max()+pad
    ymin, ymax = pts[:,1].min()-pad, pts[:,1].max()+pad
    Nx = int(round((xmax-xmin)/H)); Ny = int(round((ymax-ymin)/H))
    xs = xmin+(np.arange(Nx)+0.5)*H; ys = ymin+(np.arange(Ny)+0.5)*H
    XC, YC = np.meshgrid(xs, ys); P = np.stack([XC, YC], -1)

    def sd2(a, b):
        ab = b-a; L2 = ab@ab+1e-12
        t = np.clip(((P-a)@ab)/L2, 0, 1); pr = a+t[..., None]*ab
        return ((P-pr)**2).sum(-1)
    inside = np.minimum(sd2(I, B), sd2(B, O)) <= (W/2)**2     # corridor tube (clean)
    proj = (P - O) @ d2
    exitm = inside & (proj >= -H)                             # outlet end
    return XC, YC, inside, exitm, d2, (I, B, O)


def fmm_field(inside, exitm):
    phi = np.ones(inside.shape); phi[exitm] = -1.0
    phi = np.ma.MaskedArray(phi, ~inside)        # mask walls -> distance wraps around them
    D = np.abs(np.ma.filled(skfmm.distance(phi, dx=H), np.nan))
    gy, gx = np.gradient(np.nan_to_num(D, nan=0.0), H)
    U, V = -gx, -gy; n = np.hypot(U, V); n[n == 0] = 1.0
    U, V = U/n, V/n
    bad = ~inside | ~np.isfinite(D)
    U[bad] = np.nan; V[bad] = np.nan
    return D, U, V


def dijkstra_mindelta(inside, exitm):
    Ny, Nx = inside.shape; INF = 1e30; dist = np.full((Ny, Nx), INF); pq = []
    for iy in range(Ny):
        for ix in range(Nx):
            if exitm[iy, ix]: dist[iy, ix] = 0.0; heapq.heappush(pq, (0.0, iy, ix))
    while pq:
        d, iy, ix = heapq.heappop(pq)
        if d > dist[iy, ix]: continue
        for dx, dy in N8:
            jx, jy = ix+dx, iy+dy
            if not (0 <= jx < Nx and 0 <= jy < Ny) or not inside[jy, jx]: continue
            if dx and dy and not (inside[iy, jx] and inside[jy, ix]): continue
            nd = d+(H*2**.5 if dx and dy else H)
            if nd < dist[jy, jx]: dist[jy, jx] = nd; heapq.heappush(pq, (nd, jy, jx))
    U = np.full((Ny, Nx), np.nan); V = np.full((Ny, Nx), np.nan)
    for iy in range(Ny):
        for ix in range(Nx):
            if not inside[iy, ix] or dist[iy, ix] >= INF: continue
            vx = vy = 0.0
            for dx, dy in N8:
                jx, jy = ix+dx, iy+dy
                if not (0 <= jx < Nx and 0 <= jy < Ny) or not inside[jy, jx] or dist[jy, jx] >= INF: continue
                dd = dist[iy, ix]-dist[jy, jx]
                if dd > 0: nn = (dx*dx+dy*dy)**.5; vx += dd*dx/nn; vy += dd*dy/nn
            m = (vx*vx+vy*vy)**.5
            if m > 1e-9: U[iy, ix], V[iy, ix] = vx/m, vy/m
    return dist, U, V


fig, axes = plt.subplots(2, 3, figsize=(19, 13), constrained_layout=True)
for col, th in enumerate(ANGLES):
    XC, YC, inside, exitm, d2, (I, B, O) = grids(th)
    for row, (name, fn) in enumerate([('Dijkstra + min-delta (current)', dijkstra_mindelta),
                                       ('Fast Marching (Eikonal) -grad D', fmm_field)]):
        D, U, V = fn(inside, exitm)
        ax = axes[row][col]
        ax.contourf(XC, YC, np.where(inside, np.where(np.isfinite(D) & (D < 1e29), D, np.nan), np.nan),
                    levels=25, cmap='viridis', alpha=0.5)
        s = 3; good = inside & np.isfinite(U)
        ax.quiver(XC[::s, ::s], YC[::s, ::s], np.where(good, U, np.nan)[::s, ::s],
                  np.where(good, V, np.nan)[::s, ::s], color='k', scale=35, width=0.0035)
        ax.plot(*O, 'g^', ms=11); ax.plot(*I, 'co', ms=9)
        ax.set_aspect('equal'); ax.set_title(f'theta={th} - {name}', fontsize=11)
fig.suptitle('Dijkstra+min-delta (top) vs Fast-Marching Eikonal (bottom) on the SAME corridor tube.\n'
             'FMM tracks the diagonal exactly and curves through corners; no banding, no swirl.',
             fontsize=13, fontweight='bold')
p = os.path.join(OUT, 'floorfield_fmm_vs_dijkstra.png')
plt.savefig(p, dpi=105); plt.close(fig)
print('Saved', p)
