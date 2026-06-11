"""Single source of truth for all scenario geometries.

Each geometry is (bounds, walls, door, walkable_rects). DoorWidth (DW) is a
runtime override applied by builder functions so the registry can vary it
per-sweep (e.g. moussaid_room).
"""

DW_DEFAULT = 1.0


def fis_room(DW=DW_DEFAULT, Lx=15.0, Ly=15.0):
    """15x15 m room, east-wall door."""
    cy = Ly / 2
    return dict(
        bounds=(0, Lx, 0, Ly),
        walls=[(0, 0, Lx, 0), (0, 0, 0, Ly), (0, Ly, Lx, Ly),
               (Lx, 0, Lx, cy - DW / 2), (Lx, cy + DW / 2, Lx, Ly)],
        door=(Lx, cy - DW / 2, Lx, cy + DW / 2),
        walkable_rects=[(0, Lx, 0, Ly)],
    )


def moussaid_room(DW=1.0, Lx=10.0, Ly=4.0):
    """10 m × 4 m corridor (Kretz 2006), east-wall door — Moussaid 2011 Fig.S5.
    Agents walk eastward (+x) toward the exit centered on the east wall."""
    cy = Ly / 2
    return dict(
        bounds=(0, Lx, 0, Ly),
        walls=[(0, 0, Lx, 0), (0, 0, 0, Ly), (0, Ly, Lx, Ly),
               (Lx, 0, Lx, cy - DW / 2), (Lx, cy + DW / 2, Lx, Ly)],
        door=(Lx, cy - DW / 2, Lx, cy + DW / 2),
        walkable_rects=[(0, Lx, 0, Ly)],
    )


def l_narrow(DW=DW_DEFAULT):
    """Symmetric T-merge, narrow arms. C exe ScenarioID=1."""
    return dict(
        bounds=(0, 20, 0, 10),
        walls=[
            (0, 0, 20, 0), (0, 0, 0, 10),
            (0, 10, 9 - DW / 2, 10), (9 + DW / 2, 10, 20, 10),
            (20, 0, 20, 10),
            (0, 2, 8, 2), (10, 2, 20, 2),
            (8, 2, 8, 10), (10, 2, 10, 10),
        ],
        door=(9 - DW / 2, 10, 9 + DW / 2, 10),
        walkable_rects=[(0, 20, 0, 2), (8, 10, 2, 10)],
    )


def l_wide(DW=DW_DEFAULT):
    """Symmetric T-merge, wide arms. C exe ScenarioID=2."""
    return dict(
        bounds=(0, 20, 0, 10),
        walls=[
            (0, 0, 20, 0), (0, 0, 0, 10),
            (0, 10, 10 - DW / 2, 10), (10 + DW / 2, 10, 20, 10),
            (20, 0, 20, 10),
            (0, 4, 8, 4), (12, 4, 20, 4),
            (8, 4, 8, 10), (12, 4, 12, 10),
        ],
        door=(10 - DW / 2, 10, 10 + DW / 2, 10),
        walkable_rects=[(0, 20, 0, 4), (8, 12, 4, 10)],
    )


def t_corridor(DW=None, TStemHalfWidth=2.0):
    """True T-shape: 40 m h-arm + stem of half-width TStemHalfWidth.
    Default 2.0 → 4 m wide stem. Mitigation: TStemHalfWidth=4 → 8 m wide stem.
    DW = exit width on TOP of stem (Y=12); default = full stem-top open."""
    half = max(0.5, min(9.0, float(TStemHalfWidth)))
    st_lo = 20.0 - half
    st_hi = 20.0 + half
    full_dw = 2 * half
    if DW is None or DW <= 0:
        DW = full_dw
    door_clip = 0.05
    d_lo = max(st_lo + door_clip, 20 - DW / 2)
    d_hi = min(st_hi - door_clip, 20 + DW / 2)
    walls = [
        (0, 0, 40, 0),                                    # south of h-arm
        (0, 0, 0, 4),                                     # west of h-arm
        (40, 0, 40, 4),                                   # east of h-arm
        (0, 4, st_lo, 4), (st_hi, 4, 40, 4),              # top of h-arm
        (st_lo, 4, st_lo, 12), (st_hi, 4, st_hi, 12),     # stem sides
    ]
    if d_lo > st_lo + door_clip + 1e-3:
        walls.append((st_lo, 12, d_lo, 12))
    if d_hi < st_hi - door_clip - 1e-3:
        walls.append((d_hi, 12, st_hi, 12))
    return dict(
        bounds=(0, 40, 0, 12),
        walls=walls,
        door=(d_lo, 12, d_hi, 12),
        walkable_rects=[(0, 40, 0, 4), (st_lo, st_hi, 4, 12)],
    )


def bidir_corridor(DW=None, Lx=16.0, Ly=4.0):
    """Bidirectional corridor (Moussaid 2011 Fig.S2): 16 m × 4 m,
    periodic BC on east/west, walls top/bottom only. ScenarioID=4."""
    return dict(
        bounds=(0, Lx, 0, Ly),
        walls=[(0, 0, Lx, 0), (0, Ly, Lx, Ly)],
        door=(Lx, 0, Lx, Ly),   # cosmetic only — wraps periodically
        walkable_rects=[(0, Lx, 0, Ly)],
    )


def moussaid_corner(DW=5.0, ChamferLen=0.0, DiagonalChamfer=0.0, CurvedChamferR=0.0):
    """90° corner — Moussaid 2011 Fig.S4 paper geometry. 10 × 10 m L-shape.
    Walkable: LEFT arm [0,5]×[0,10] + TOP arm [0,10]×[5,10].
    Cut-out (solid): [5,10]×[0,5]. Inner concave corner at (5, 5).

    DW: exit width on east wall of upper arm (5 = full open).
    ChamferLen: inner-corner chamfer length (m). 0 = sharp 90°.
        When > 0, the inner corner is replaced by a 45° chamfer
        from (5, 5-d) to (5+d, 5). Walls 5 and 6 are shortened and
        a straight diagonal segment is drawn (in C it is a K=5
        staircase approximation; matches geometrically)."""
    d_lo = max(5.05, 7.5 - DW / 2)
    d_hi = min(9.95, 7.5 + DW / 2)
    d = max(0.0, min(4.5, float(ChamferLen)))
    diag = float(DiagonalChamfer) > 0.5
    r = max(0.0, min(4.5, float(CurvedChamferR)))

    walls = [
        (0, 10, 10, 10),           # north
        (0, 0,  0, 10),            # west
    ]
    if r > 0.05:
        # Quarter-arc inner corner from (5, 5-r) to (5+r, 5), polygon of K segments
        import math as _math
        walls += [
            (0, 0, 5, 0),                 # south of left arm
            (5 + r, 5, 10, 5),            # top of cut-out (shortened)
            (5, 0, 5, 5 - r),             # left of cut-out (shortened)
        ]
        K = 48; cx, cy = 5 + r, 5 - r        # 48-segment polygon for smooth visual arc
        for k in range(K):
            t0 = 0.5 * _math.pi * k / K
            t1 = 0.5 * _math.pi * (k + 1) / K
            x0 = cx - r * _math.cos(t0); y0 = cy + r * _math.sin(t0)
            x1 = cx - r * _math.cos(t1); y1 = cy + r * _math.sin(t1)
            walls.append((x0, y0, x1, y1))
    elif diag:
        walls += [
            (0, 0, 3, 0),          # south of left arm (shortened to X∈[0,3])
            (7, 5, 10, 5),         # top of cut-out (shortened to X∈[7,10])
            (3, 0, 7, 5),          # single diagonal chamfer
        ]
    else:
        walls.append((0, 0, 5, 0))   # south of left arm
        if d < 0.01:
            walls += [
                (5, 5, 10, 5),         # top of cut-out (full)
                (5, 0, 5, 5),          # left of cut-out (full)
            ]
        else:
            walls += [
                (5 + d, 5, 10, 5),     # top of cut-out (shortened)
                (5, 0, 5, 5 - d),      # left of cut-out (shortened)
                (5, 5 - d, 5 + d, 5),  # chamfer diagonal
            ]
    if d_lo > 5.05 + 1e-3:
        walls.append((10, 5, 10, d_lo))
    if d_hi < 9.95 - 1e-3:
        walls.append((10, d_hi, 10, 10))

    return dict(
        bounds=(0, 10, 0, 10),
        walls=walls,
        door=(10, d_lo, 10, d_hi),
        walkable_rects=[(0, 5, 0, 10), (0, 10, 5, 10)],
    )


# Convenience lookup
BUILDERS = {
    'fis_room':         fis_room,
    'moussaid_room':    moussaid_room,
    'l_narrow':         l_narrow,
    'l_wide':           l_wide,
    't_corridor':       t_corridor,
    'moussaid_corner':  moussaid_corner,
    'bidir_corridor':   bidir_corridor,
}


def build(name, DW=DW_DEFAULT, **kw):
    if name not in BUILDERS:
        raise KeyError(f"Unknown geometry '{name}'. Known: {list(BUILDERS)}")
    # moussaid_corner accepts ChamferLen; other builders ignore it.
    if name == 'moussaid_corner':
        return BUILDERS[name](DW, ChamferLen=kw.get('ChamferLen', 0.0),
                              DiagonalChamfer=kw.get('DiagonalChamfer', 0.0),
                              CurvedChamferR=kw.get('CurvedChamferR', 0.0))
    if name == 't_corridor':
        return BUILDERS[name](DW, TStemHalfWidth=kw.get('TStemHalfWidth', 2.0))
    return BUILDERS[name](DW)


def draw(ax, geo, wall_color='k', door_color='lime'):
    for x1, y1, x2, y2 in geo['walls']:
        ax.plot([x1, x2], [y1, y2], color=wall_color, linewidth=1.8, zorder=3)
    dx1, dy1, dx2, dy2 = geo['door']
    ax.plot([dx1, dx2], [dy1, dy2], color=door_color, linewidth=4,
            label='Exit', zorder=4)
