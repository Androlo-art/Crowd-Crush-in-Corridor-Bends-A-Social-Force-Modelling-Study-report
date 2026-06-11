"""Carefully construct PROPER wall-based bend-corridor geometry (like the
initial scenarios: explicit inner/outer walls) for 45, 90, 135 deg, and
compute the Dijkstra floor field (wall-clearance mask + MIN-DELTA direction)
on it. VERIFY here in Python before porting wall coordinates to C.

Corridor: constant width w, arm 1 vertical (inlet at bottom), bend at B,
arm 2 leaving at turn angle theta toward +x; outlet at arm-2 end.
Inner wall = right-offset polyline I_in -> Vin -> O_in.
Outer wall = left-offset  polyline I_out -> Vout -> O_out.
Vin/Vout via exact line intersection (no hand-derived formula).
"""
import os, heapq
import numpy as np
import matplotlib.pyplot as plt

W = 3.0; HB = 8.0; LB = 8.0; CX = 5.0; H = 0.1; CLEAR = 0.15
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results', 'bend_study')
os.makedirs(OUT, exist_ok=True)
ANGLES = [45, 90, 135]
N8 = [(1,0),(-1,0),(0,1),(0,-1),(1,1),(1,-1),(-1,1),(-1,-1)]


def line_intersect(P1, D1, P2, D2):
    """intersection of line P1+a*D1 and P2+b*D2."""
    A = np.array([[D1[0], -D2[0]], [D1[1], -D2[1]]])
    ab = np.linalg.solve(A, np.array(P2) - np.array(P1))
    return np.array(P1) + ab[0]*np.array(D1)


def geometry(theta):
    th = np.radians(theta)
    d1 = np.array([0.0, 1.0]); d2 = np.array([np.sin(th), np.cos(th)])
    I = np.array([CX, 0.0]); B = np.array([CX, HB]); O = B + LB*d2
    n1 = np.array([1.0, 0.0])               # right of d1 (inner side)
    n2 = np.array([np.cos(th), -np.sin(th)])  # right of d2 (inner side)
    I_in, O_in = I + (W/2)*n1, O + (W/2)*n2
    I_out, O_out = I - (W/2)*n1, O - (W/2)*n2
    Vin = line_intersect(I_in, d1, O_in, d2)
    Vout = line_intersect(I_out, d1, O_out, d2)
    inner = [I_in, Vin, O_in]      # polyline
    outer = [I_out, Vout, O_out]
    walls = []                      # list of (x1,y1,x2,y2)
    for poly in (inner, outer):
        for a, b in zip(poly[:-1], poly[1:]):
            walls.append((a[0], a[1], b[0], b[1]))
    return dict(I=I, B=B, O=O, d2=d2, inner=inner, outer=outer, walls=walls,
                O_in=O_in, O_out=O_out)


def seg_d2(P, x1, y1, x2, y2):
    ax = np.array([x1, y1]); bx = np.array([x2, y2]); ab = bx-ax
    L2 = ab@ab + 1e-12
    t = np.clip(((P-ax)@ab)/L2, 0, 1)
    proj = ax + t[..., None]*ab
    return ((P-proj)**2).sum(-1)


def field(g):
    allp = np.array(g['inner']+g['outer']+[g['I'], g['O']])
    xmin, xmax = allp[:,0].min()-1, allp[:,0].max()+1
    ymin, ymax = allp[:,1].min()-1, allp[:,1].max()+1
    Nx = int(round((xmax-xmin)/H)); Ny = int(round((ymax-ymin)/H))
    xs = xmin+(np.arange(Nx)+0.5)*H; ys = ymin+(np.arange(Ny)+0.5)*H
    XC, YC = np.meshgrid(xs, ys); P = np.stack([XC, YC], -1)
    # wall-clearance mask
    inside = np.ones((Ny, Nx), bool)
    for (x1, y1, x2, y2) in g['walls']:
        inside &= (seg_d2(P, x1, y1, x2, y2) >= CLEAR**2)
    # exit cells: near the outlet end-cap segment (O_in - O_out)
    oi, oo = g['O_in'], g['O_out']
    exitm = inside & (seg_d2(P, oi[0], oi[1], oo[0], oo[1]) <= (1.5*H)**2)
    # Dijkstra from exit
    INF = 1e30; dist = np.full((Ny, Nx), INF); pq = []
    for iy in range(Ny):
        for ix in range(Nx):
            if exitm[iy, ix]:
                dist[iy, ix] = 0.0; heapq.heappush(pq, (0.0, iy, ix))
    while pq:
        d, iy, ix = heapq.heappop(pq)
        if d > dist[iy, ix]: continue
        for dx, dy in N8:
            jx, jy = ix+dx, iy+dy
            if not (0 <= jx < Nx and 0 <= jy < Ny) or not inside[jy, jx]: continue
            if dx and dy and not (inside[iy, jx] and inside[jy, ix]): continue
            nd = d+(H*1.41421356 if dx and dy else H)
            if nd < dist[jy, jx]:
                dist[jy, jx] = nd; heapq.heappush(pq, (nd, jy, jx))
    # min-delta direction
    GX = np.zeros((Ny, Nx)); GY = np.zeros((Ny, Nx))
    for iy in range(Ny):
        for ix in range(Nx):
            if not inside[iy, ix] or dist[iy, ix] >= INF: continue
            vx = vy = 0.0
            for dx, dy in N8:
                jx, jy = ix+dx, iy+dy
                if not (0 <= jx < Nx and 0 <= jy < Ny) or not inside[jy, jx]: continue
                if dist[jy, jx] >= INF: continue
                dd = dist[iy, ix]-dist[jy, jx]
                if dd > 0:
                    n = (dx*dx+dy*dy)**0.5; vx += dd*dx/n; vy += dd*dy/n
            m = (vx*vx+vy*vy)**0.5
            if m > 1e-9: GX[iy, ix], GY[iy, ix] = vx/m, vy/m
    return XC, YC, inside, dist, GX, GY, (xmin, xmax, ymin, ymax)


fig, axes = plt.subplots(2, 3, figsize=(19, 13), constrained_layout=True)
for col, th in enumerate(ANGLES):
    g = geometry(th)
    # row 0: geometry (walls)
    ax = axes[0][col]
    for (x1, y1, x2, y2) in g['walls']:
        ax.plot([x1, x2], [y1, y2], 'k-', lw=3)
    ax.plot([g['I'][0]-W/2, g['I'][0]+W/2], [0, 0], 'c-', lw=4)          # inlet
    ax.plot([g['O_in'][0], g['O_out'][0]], [g['O_in'][1], g['O_out'][1]], 'g-', lw=4)  # outlet
    cl = np.array([g['I'], g['B'], g['O']])
    ax.plot(cl[:,0], cl[:,1], 'b--', lw=1, alpha=0.6)
    ax.set_aspect('equal'); ax.set_title(f'theta={th} - PROPER walls', fontsize=12); ax.grid(alpha=0.2)
    # row 1: field
    XC, YC, inside, dist, GX, GY, bb = field(g)
    ax2 = axes[1][col]
    ax2.contourf(XC, YC, np.where(inside & (dist < 1e29), dist, np.nan), levels=25, cmap='viridis', alpha=0.55)
    for (x1, y1, x2, y2) in g['walls']:
        ax2.plot([x1, x2], [y1, y2], 'w-', lw=2)
    s = 3; good = inside & (np.hypot(GX, GY) > 0.1)
    ax2.quiver(XC[::s, ::s], YC[::s, ::s], np.where(good, GX, np.nan)[::s, ::s],
               np.where(good, GY, np.nan)[::s, ::s], color='k', scale=35, width=0.0035)
    ax2.set_aspect('equal'); ax2.set_title(f'theta={th} - Dijkstra min-delta field', fontsize=12)
fig.suptitle('PROPER wall-based bend corridors + Dijkstra min-delta floor field (Python verify, no sim)',
             fontsize=14, fontweight='bold')
p = os.path.join(OUT, 'bend_proper_geom_field.png')
plt.savefig(p, dpi=105); plt.close(fig)
print('Saved', p)
# print wall coords for the C port
for th in ANGLES:
    g = geometry(th)
    print(f'--- theta={th} walls (x1,y1,x2,y2) ---')
    for w in g['walls']:
        print('   ' + ', '.join(f'{v:.3f}' for v in w))
