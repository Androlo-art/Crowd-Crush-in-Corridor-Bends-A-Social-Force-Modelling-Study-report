"""Analysis + figures for the OBSTACLE-MITIGATION study (run AFTER the sweep finishes).

Reads a results/bend_obstacle/obs_<stamp>/ run (aggregated.csv + grids/) and produces,
config-agnostically (whatever column configs are present):

  1) benefit_vs_angle : the QUANTITATIVE headline. For each harm metric (casualties,
     peak door density, door contact p95) and the reliable throughput (inflow_rate),
     one line per obstacle config vs turn angle, with the no-column control bold black.
     -> answers "does an obstacle help, which config, by how much, and does the benefit
        depend on the turn angle?" and exposes any crush-relief-vs-throughput tradeoff.

  2) density_maps_a<angle> : the MECHANISM visual. A montage of steady-window DENSITY
     fields [ctrl | each config] at one angle, cropped to the door region, SHARED colour
     scale, with the chamfered walls + door + column(s) drawn. -> the reader SEES the
     door arch in the control and how each obstacle breaks/reshapes it.
     (Density, not pressure: crowd pressure rho*Var(v) collapses to ~0 exactly where the
     crowd locks, so it understates the worst crush; density and contact are the honest
     harm fields. Use --field contact for the contact-force montage.)

  3) obstacle_summary.csv + stdout table : per (angle, config) metric means with the
     percentage change vs the control, so the effect size is explicit.

This ONLY reads the run folder and writes to --outdir (default <run>/figs). It produces
no new simulations and overwrites nothing outside --outdir.

Usage:
  python -u bend_bottleneck/analyze_obstacle.py --run results/bend_obstacle/obs_<stamp>
  python -u bend_bottleneck/analyze_obstacle.py --run results/bend_obstacle/obs_<stamp> --field contact
  python -u bend_bottleneck/analyze_obstacle.py --run results/bend_obstacle/obs_<stamp> --angle 90
"""
import os, sys, csv, glob, argparse
import numpy as np
from matplotlib.path import Path as MPath

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
import make_field_bottleneck as B          # geometry()/door endpoints (UNTOUCHED)

RES = 0.25; DOMX, DOMY = 18.0, 32.0
NX, NY = int(DOMX/RES), int(DOMY/RES)
WIDTH = 5.0; CORNER_R = 0.25
DOOR_BACK = 2.0       # door ROI = this many m of approach in front of the cap (matches the runner)
AGENT_R = 0.3         # agent radius (~D_mean/2); cells within (D_c/2 + AGENT_R) of a column are excluded
CROP_BACK = 2.0       # crop box centred this many m upstream of the door (into the arch)
CROP_HALF = 3.6       # half-size of the square door-region crop (m)

# door_contact_excol = peak door-ROI contact with the COLUMN footprints masked out, so it
# measures agent-agent crush, not agent-column reaction (computed from grids; ctrl excludes
# nothing -> identical to capov). casualties is the unconfounded harm headline regardless.
HARM = [('casualties', 'casualties (mean in window)'),
        ('door_rho_peak', r'peak door density (ped m$^{-2}$)'),
        ('door_outflow', r'agent exit outflow (ag s$^{-1}$)')]   # contact dropped from this summary
        # (peak contact is a noisy max -> non-monotonic across angles; it is shown spatially
        #  in the contact maps + harm profiles instead. door_contact_excol stays in the table.)

# Central display metadata so EVERY figure uses the same human-readable names + colours.
# internal label -> (display name, colour, arrangement text for caption/key, short tag)
# display name encodes the distinguishing parameter (distances are UPSTREAM of the exit).
# display name (legends/panel titles) | colour | full arrangement (caption key) | 2-line grid header.
# distances are measured from the exit, back along the corridor; all pillars 0.9 m wide unless noted.
CFG_META = {
    'ctrl':           ('no obstacle',                    'black',   'none (baseline)',                                                 'none'),
    'pillar_axis':    ('single pillar, centred (1.5 m)',  '#d62728', '1 pillar on the centreline, 1.5 m from the exit, 0.9 m wide',     'centred\n1.5 m from exit'),
    'pillar_off':     ('single pillar, offset (1.5 m)',   '#ff7f0e', '1 pillar 0.8 m off-centre, 1.5 m from the exit, 0.9 m wide',      'offset 0.8 m\n1.5 m from exit'),
    'pillar_off_far': ('single pillar, offset (2.2 m)',   '#8c564b', '1 pillar 0.8 m off-centre, 2.2 m from the exit, 0.9 m wide',      'offset 0.8 m\n2.2 m from exit'),
    'pillar_big':     ('single pillar, large (2.0 m)',    '#9467bd', '1 pillar 0.8 m off-centre, 2.0 m from the exit, 1.1 m wide (+22%)','large +22%\n2.0 m from exit'),
    'gate':           ('two-pillar gate (1.1 m gap)',     '#1f77b4', '2 pillars, 1.1 m gap, 1.8 m from the exit, 0.9 m wide',           'gate 1.1 m gap\n1.8 m from exit'),
    'gate_wide':      ('two-pillar gate (1.6 m gap)',     '#17becf', '2 pillars, 1.6 m gap, 1.8 m from the exit, 0.9 m wide',           'gate 1.6 m gap\n1.8 m from exit'),
    'gate_close':     ('two-pillar gate (near, 1.2 m)',   '#3b5bdb', '2 pillars, 1.2 m gap, 1.2 m from the exit, 0.8 m wide',           'gate 1.2 m gap\n1.2 m from exit'),
}
CURATED = ['ctrl', 'pillar_off_far', 'gate']      # logical 0 -> 1 pillar -> 2 pillars for the main figure


def _disp(lab):  return CFG_META.get(lab, (lab, 'gray', '', lab))[0]
def _color(lab): return CFG_META.get(lab, (lab, 'gray', '', lab))[1]
def _arr(lab):   return CFG_META.get(lab, (lab, 'gray', '', lab))[2]
def _short(lab): return CFG_META.get(lab, (lab, 'gray', '', lab))[3]


def _f(x):
    try: return float(x)
    except (ValueError, TypeError): return float('nan')


def load_aggregated(run):
    """-> dict[(angle,label)] = row(dict); plus ordered angle/label lists."""
    p = os.path.join(run, 'aggregated.csv')
    if not os.path.exists(p):
        raise SystemExit(f'no aggregated.csv in {run} (run the sweep to completion first)')
    rows = list(csv.DictReader(open(p)))
    D = {}
    angles, labels = [], []
    for r in rows:
        a = int(round(_f(r['angle']))); lab = r['col_label']
        D[(a, lab)] = r
        if a not in angles: angles.append(a)
        if lab not in labels: labels.append(lab)
    angles = sorted(angles)
    # canonical label order: ctrl first, then as encountered
    order = {l: i for i, l in enumerate(CFG_META)}        # logical: ctrl -> pillars -> gates
    labels = sorted(labels, key=lambda l: (order.get(l, 99), l))
    return D, angles, labels


def _palette(labels):
    """(colour, linestyle, linewidth, marker) per config, from the central CFG_META."""
    col = {}
    for l in labels:
        c = _color(l)
        if l == 'ctrl':
            col[l] = (c, '-', 2.6, 'o')
        else:
            col[l] = (c, '-', 1.8, ('s' if l.startswith('gate') else '^'))
    return col


def benefit_vs_angle(D, angles, labels, outdir):
    import matplotlib; matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    try:
        import thesis_style as TS; TS.apply(); figw = TS.FIGW
    except Exception:
        figw = 6.3
    col = _palette(labels)
    n = len(HARM); nrow = 1 if n <= 3 else 2; ncol = int(np.ceil(n/nrow))
    legrows = int(np.ceil(len(labels) / 4.0))
    panel_h = (figw/ncol)*0.95*nrow; leg_h = 0.32*legrows + 0.22; H = panel_h + leg_h + 0.35
    fig, axes = plt.subplots(nrow, ncol, squeeze=False, figsize=(figw, H))
    axes = axes.ravel()
    handles = lbls = None
    for j, (metric, ylab) in enumerate(HARM):
        ax = axes[j]
        for lab in labels:
            xs, ys, es = [], [], []
            for a in angles:
                r = D.get((a, lab))
                if not r: continue
                xs.append(a); ys.append(_f(r.get(f'{metric}_mean'))); es.append(_f(r.get(f'{metric}_ci95')))
            if not xs: continue
            xs = np.array(xs, float); ys = np.array(ys, float); es = np.array(es, float)
            c, ls, lw, mk = col[lab]
            lw = 1.8 if lab == 'ctrl' else 1.2          # thinner lines so curves are readable
            ax.fill_between(xs, ys-es, ys+es, color=c, alpha=0.13, lw=0,
                            zorder=2 if lab == 'ctrl' else 1)          # shaded 95% CI of the mean
            ax.plot(xs, ys, color=c, ls=ls, lw=lw, marker=mk, ms=4, label=_disp(lab),
                    zorder=6 if lab == 'ctrl' else 4)
        ax.set_xlabel('turn angle (degrees)'); ax.set_ylabel(ylab); ax.grid(alpha=0.3)
        ax.set_xticks(angles)
        ax.set_title(f'({chr(97+j)})', fontsize=11, fontweight='bold', loc='left')
        if j == 0: handles, lbls = ax.get_legend_handles_labels()
    for k in range(n, len(axes)): axes[k].axis('off')
    # reserve a bottom strip for the legend so it can never overlap the panels
    fig.subplots_adjust(left=0.085, right=0.98, top=1 - 0.32/H, bottom=(leg_h + 0.18)/H,
                        wspace=0.30, hspace=0.5)
    if handles:
        fig.legend(handles, lbls, loc='lower center', ncol=min(4, len(labels)), fontsize=8.5,
                   frameon=False, bbox_to_anchor=(0.5, 0.012),
                   title='shaded band = 95% CI', title_fontsize=8.5)
    os.makedirs(outdir, exist_ok=True)
    out = os.path.join(outdir, 'obstacle_benefit_vs_angle')
    for ext in ('pdf', 'png'): fig.savefig(f'{out}.{ext}', dpi=150, bbox_inches='tight')
    plt.close(fig); print(f'wrote {out}.pdf/.png', flush=True)


def load_grid(path):
    G = {k: np.full((NY, NX), np.nan) for k in ('rho', 'contact', 'casualty', 'pressure')}
    if not os.path.exists(path): return None
    for r in csv.DictReader(open(path)):
        ix = min(NX-1, int(_f(r['x'])/RES)); iy = min(NY-1, int(_f(r['y'])/RES))
        for k in G: G[k][iy, ix] = _f(r[k])
    return G


def _door_and_halo(angle, agg_row):
    """Door ROI (full-width approach band, identical to the runner) and the column-halo mask
    (cells within D_c/2 + AGENT_R of any column centre). Returns (door_bool, halo_bool)."""
    g = B.geometry(int(angle), WIDTH)
    O = np.array(g['O']); d2 = np.array(g['d2']); n2 = np.array(g['n2'])
    xc = (np.arange(NX)+0.5)*RES; yc = (np.arange(NY)+0.5)*RES
    XX, YY = np.meshgrid(xc, yc)
    walk = MPath(g['poly']).contains_points(np.column_stack([XX.ravel(), YY.ravel()])).reshape(NY, NX)
    p = (XX-O[0])*d2[0]+(YY-O[1])*d2[1]; lat = (XX-O[0])*n2[0]+(YY-O[1])*n2[1]
    door = walk & (p <= 0) & (p >= -DOOR_BACK) & (np.abs(lat) <= WIDTH/2)
    halo = np.zeros((NY, NX), bool)
    for (xk, yk, dk) in (('col_x', 'col_y', 'col_d'), ('col2_x', 'col2_y', 'col2_d')):
        cx, cy, dd = _f(agg_row.get(xk)), _f(agg_row.get(yk)), _f(agg_row.get(dk))
        if np.isfinite(cx) and np.isfinite(cy) and np.isfinite(dd) and dd > 0:
            halo |= ((XX-cx)**2 + (YY-cy)**2) <= (dd/2 + AGENT_R)**2
    return door, halo


def excluded_contact(run, angle, label, agg_row):
    """Peak steady-window contact in the door ROI with the column footprints masked out
    (from the grid CSV). For ctrl nothing is masked -> identical to the capov contact field."""
    cand = glob.glob(os.path.join(run, 'grids', f'a{angle}_{label}_lam*.csv'))
    if not cand:
        return float('nan')
    G = load_grid(cand[0])
    if G is None:
        return float('nan')
    door, halo = _door_and_halo(angle, agg_row)
    roi = door & ~halo & np.isfinite(G['contact'])
    return float(np.nanmax(G['contact'][roi])) if roi.any() else float('nan')


def _geom(angle):
    try:
        import make_field_capacity_chamfer as FC
        return FC.geometry_rounded(int(angle), WIDTH, CORNER_R)
    except Exception:
        return B.geometry(int(angle), WIDTH)


def _imshow_clipped(ax, arr, angle, cmap, vmax):
    """Render a field smoothly: NaN cells filled from the nearest valid cell, bilinearly
    interpolated for display, then HARD-CLIPPED to the corridor polygon so colour never
    bleeds across the diagonal walls and no interior cell stays white. Rendering only —
    the underlying gridded data is unchanged (no effect on any reported number)."""
    import matplotlib.patches as mp
    from scipy.ndimage import distance_transform_edt
    a = arr.copy(); m = ~np.isfinite(a)
    if m.any() and (~m).any():
        idx = distance_transform_edt(m, return_distances=False, return_indices=True)
        a = a[tuple(idx)]
    g = B.geometry(int(angle), WIDTH)   # clip to the SHARP polygon the grids were written on
    im = ax.imshow(a, origin='lower', extent=[0, DOMX, 0, DOMY], cmap=cmap, vmin=0, vmax=vmax,
                   interpolation='bilinear')
    clip = mp.Polygon(np.array(g['poly']), closed=True, fc='none', ec='none')
    ax.add_patch(clip); im.set_clip_path(clip)
    return im


def _overlay(ax, angle, agg_row, wall_lw=2.0):
    """Thick corridor walls + door + column(s)."""
    import matplotlib.patches as mp
    g = _geom(angle)
    poly = g['poly'] + [g['poly'][0]]; px, py = zip(*poly)
    ax.plot(px, py, 'k-', lw=wall_lw, zorder=6)
    dlo, dhi = g['door_lo'], g['door_hi']
    ax.plot([dlo[0], dhi[0]], [dlo[1], dhi[1]], color='limegreen', lw=3.0, zorder=7)
    if agg_row:
        for (xk, yk, dk) in (('col_x', 'col_y', 'col_d'), ('col2_x', 'col2_y', 'col2_d')):
            cx, cy, dd = _f(agg_row.get(xk)), _f(agg_row.get(yk)), _f(agg_row.get(dk))
            if np.isfinite(cx) and np.isfinite(cy) and np.isfinite(dd) and dd > 0:
                ax.add_patch(mp.Circle((cx, cy), dd/2, color='darkorange', alpha=0.95, zorder=8))
                ax.add_patch(mp.Circle((cx, cy), dd/2, fill=False, ec='k', lw=1.0, zorder=9))


def _crop(angle):
    g = B.geometry(int(angle), WIDTH)
    O = np.array(g['O']); d2 = np.array(g['d2'])
    c = O - CROP_BACK*d2          # box centre = into the corridor from the door
    return (c[0]-CROP_HALF, c[0]+CROP_HALF), (c[1]-CROP_HALF, c[1]+CROP_HALF)


def density_maps(run, D, angles, labels, outdir, field='rho', only_angle=None, cfgs=None):
    """Per-angle heatmap montage at the exit. Default = curated configs (no obstacle ->
    single pillar -> gate) for the main text; pass cfgs=labels to show all (auto 2 rows).
    Panels: bold (a),(b),... + plain-English title; shared horizontal colourbar."""
    import matplotlib; matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    try:
        import thesis_style as TS; TS.apply(); cmap = TS.CMAP; figw = TS.FIGW
    except Exception:
        cmap = 'turbo'; figw = 6.3
    fname = {'rho': 'density', 'contact': 'contact force'}.get(field, field)
    funit = {'rho': r'crowd density (ped m$^{-2}$)',
             'contact': r'contact force (N kg$^{-1}$)'}.get(field, field)
    want = cfgs if cfgs is not None else CURATED
    use_angles = [only_angle] if only_angle is not None else angles
    for angle in use_angles:
        sel = []; grids = {}
        for lab in want:
            cand = glob.glob(os.path.join(run, 'grids', f'a{angle}_{lab}_lam*.csv'))
            if cand:
                G = load_grid(cand[0])
                if G is not None: grids[lab] = G; sel.append(lab)
        if not sel:
            print(f'  (no grids for angle {angle}; skip {field} map)'); continue
        vmax = max(np.nanpercentile(grids[l][field], 99) for l in sel)
        if not (np.isfinite(vmax) and vmax > 0): vmax = 1.0      # guard all-zero/all-NaN field
        xl, yl = _crop(angle)
        n = len(sel); nrow = 1 if n <= 4 else 2; ncol = int(np.ceil(n/nrow))
        fig, axes = plt.subplots(nrow, ncol, squeeze=False, constrained_layout=True,
                                 figsize=(figw, (figw/ncol)*1.18*nrow + 0.6))
        axes = axes.ravel()
        for j, lab in enumerate(sel):
            ax = axes[j]
            im = _imshow_clipped(ax, grids[lab][field], angle, cmap, vmax)
            _overlay(ax, angle, D.get((angle, lab)))
            ax.set_xlim(*xl); ax.set_ylim(*yl); ax.set_aspect('equal')
            ax.set_xticks([]); ax.set_yticks([])
            ax.set_title(f'({chr(97+j)}) {_disp(lab)}', fontsize=10, fontweight='bold')
        for k in range(len(sel), len(axes)): axes[k].axis('off')
        cb = fig.colorbar(im, ax=axes.tolist(), location='bottom', shrink=0.8, aspect=42, pad=0.03)
        cb.set_label(funit, fontsize=11)
        fig.suptitle(f'Steady-state crowd {fname} at exit, θ = {angle}°', fontsize=12)
        os.makedirs(outdir, exist_ok=True)
        out = os.path.join(outdir, f'obstacle_{field}_maps_a{angle}')
        for ext in ('pdf', 'png'): fig.savefig(f'{out}.{ext}', bbox_inches='tight', dpi=150)
        plt.close(fig); print(f'wrote {out}.pdf/.png  (vmax={vmax:.2f})', flush=True)


def summary_table(D, angles, labels, outdir):
    metrics = ['casualties', 'door_rho_peak', 'door_rho_steady', 'door_contact_excol',
               'door_contact_p95', 'crush_frac', 'inflow_rate', 'door_outflow']
    rows = []
    print('\n=== obstacle mitigation summary (Δ% vs control) ===', flush=True)
    for a in angles:
        ctrl = D.get((a, 'ctrl'))
        print(f'\n  θ = {a}°' + ('' if ctrl else '   [no control!]'), flush=True)
        for lab in labels:
            r = D.get((a, lab))
            if not r: continue
            rec = dict(angle=a, col_label=lab, n_seeds=r.get('n_seeds', ''))
            bits = [f'{lab:<14}']
            for m in metrics:
                v = _f(r.get(f'{m}_mean')); rec[f'{m}_mean'] = round(v, 3) if np.isfinite(v) else ''
                if ctrl and lab != 'ctrl':
                    c = _f(ctrl.get(f'{m}_mean'))
                    d = 100.0*(v-c)/c if (np.isfinite(v) and np.isfinite(c) and abs(c) > 1e-9) else float('nan')
                    rec[f'{m}_pct'] = round(d, 1) if np.isfinite(d) else ''
                    if m in ('casualties', 'door_rho_peak', 'door_contact_excol', 'door_outflow') and np.isfinite(d):
                        tag = 'contact' if m == 'door_contact_excol' else m.split('_')[0]
                        bits.append(f'{tag} {v:6.2f} ({d:+5.1f}%)')
                else:
                    if m in ('casualties', 'door_rho_peak', 'door_contact_excol', 'door_outflow'):
                        bits.append(f'{m.split("_")[0]} {v:6.2f}')
            rows.append(rec)
            print('    ' + '  '.join(bits), flush=True)
    if rows:
        os.makedirs(outdir, exist_ok=True)
        # union of keys across ALL rows (ctrl rows lack the _pct columns that column rows have)
        keys = ['angle', 'col_label', 'n_seeds']
        for r in rows:
            for k in r:
                if k not in keys: keys.append(k)
        with open(os.path.join(outdir, 'obstacle_summary.csv'), 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=keys, extrasaction='ignore'); w.writeheader()
            for r in rows: w.writerow(r)
        print(f'\nwrote {os.path.join(outdir, "obstacle_summary.csv")}', flush=True)


def _pillar_p(r, O, d2):
    cx, cy = _f(r.get('col_x')), _f(r.get('col_y'))
    if np.isfinite(cx) and np.isfinite(cy):
        return (cx-O[0])*d2[0] + (cy-O[1])*d2[1]
    return -1.8   # reference split for ctrl / no-column


def harm_profiles(run, D, angles, labels, outdir, sel=None):
    """Peak density, peak contact and casualty as a function of ALONG-OUTLET distance
    from the door (p=0 door, negative=upstream) -- shows whether an obstacle relieves the
    door crush or merely relocates it upstream. The single most important diagnostic here."""
    import matplotlib; matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    try:
        import thesis_style as TS; TS.apply(); figw = TS.FIGW
    except Exception:
        figw = 6.3
    if sel is None:
        sel = [l for l in ['ctrl', 'gate', 'pillar_off_far', 'pillar_axis'] if l in labels] or labels[:4]
    col = _palette(labels)
    xc = (np.arange(NX)+0.5)*RES; yc = (np.arange(NY)+0.5)*RES; XX, YY = np.meshgrid(xc, yc)
    edges = np.arange(-6.0, 0.001, 0.25); cen = 0.5*(edges[:-1]+edges[1:])
    panels = [('rho', 'peak density\n(ped m$^{-2}$)'),
              ('contact', 'peak contact\n(N kg$^{-1}$)'),
              ('casualty', 'casualties\n(relative)')]
    for angle in angles:
        g = B.geometry(int(angle), WIDTH); O = np.array(g['O']); d2 = np.array(g['d2']); n2 = np.array(g['n2'])
        walk = MPath(g['poly']).contains_points(np.column_stack([XX.ravel(), YY.ravel()])).reshape(NY, NX)
        p = (XX-O[0])*d2[0]+(YY-O[1])*d2[1]; lat = (XX-O[0])*n2[0]+(YY-O[1])*n2[1]
        band = walk & (np.abs(lat) <= WIDTH/2)
        fig, axes = plt.subplots(3, 1, figsize=(figw, figw*1.0), constrained_layout=True, sharex=True)
        for lab in sel:
            cand = glob.glob(os.path.join(run, 'grids', f'a{angle}_{lab}_lam*.csv'))
            if not cand: continue
            G = load_grid(cand[0])
            if G is None: continue
            ser = {k: [] for k, _ in panels}
            for i in range(len(cen)):
                m = band & (p >= edges[i]) & (p < edges[i+1])
                for k, _ in panels:
                    mm = m & np.isfinite(G[k])
                    ser[k].append((float(np.nansum(G[k][mm])) if k == 'casualty'
                                   else float(np.nanmax(G[k][mm]))) if mm.any() else np.nan)
            c, ls, lw, mk = col.get(lab, ('gray', '-', 1.6, 'o'))
            for ax, (k, _) in zip(axes, panels):
                ax.plot(cen, ser[k], color=c, lw=lw, label=_disp(lab), zorder=5 if lab == 'ctrl' else 3)
            # shade THIS config's pillar extent(s) in its own colour (true ~0.9 m width)
            r = D.get((angle, lab))
            if r:
                for (xk, yk, dk) in (('col_x', 'col_y', 'col_d'), ('col2_x', 'col2_y', 'col2_d')):
                    cx, cy, dd = _f(r.get(xk)), _f(r.get(yk)), _f(r.get(dk))
                    if np.isfinite(cx) and np.isfinite(cy) and np.isfinite(dd) and dd > 0:
                        pp = (cx-O[0])*d2[0] + (cy-O[1])*d2[1]
                        for ax in axes:
                            ax.axvspan(pp-dd/2, pp+dd/2, color=c, alpha=0.14, lw=0, zorder=0)
        for ax, (k, ylab) in zip(axes, panels):
            ax.axvline(0.0, color='limegreen', lw=3, zorder=1)        # the exit
            ax.set_ylabel(ylab); ax.grid(alpha=0.3, zorder=0)
        axes[0].text(0.0, 1.03, 'exit', transform=axes[0].get_xaxis_transform(), ha='center',
                     va='bottom', color='green', fontsize=9, fontweight='bold')
        axes[-1].legend(loc='upper left', fontsize=10, frameon=True, framealpha=0.92,
                        title='line = setup;   shading = pillar width', title_fontsize=9)
        axes[-1].set_xlabel('distance from exit along the corridor (m)')
        fig.suptitle(f'Where the crush sits along the corridor, θ = {angle}°', fontsize=12)
        os.makedirs(outdir, exist_ok=True)
        out = os.path.join(outdir, f'obstacle_harm_profile_a{angle}')
        for ext in ('pdf', 'png'): fig.savefig(f'{out}.{ext}', bbox_inches='tight', dpi=150)
        plt.close(fig); print(f'wrote {out}.pdf/.png', flush=True)


def profile_grid(run, D, angles, labels, outdir, which=(0, 90, 135), sel=None):
    """MAIN-TEXT multi-angle harm profile: peak density (top row) and peak contact (bottom
    row) vs along-corridor distance from the exit, ONE COLUMN PER TURN ANGLE (default
    0/90/135 = straight / right-angle bend / sharpest). A single clean panel that shows the
    relief -> relocation -> washout progression and that the result is defensible at every
    angle. Curated configs only; each row shares a y-scale so the angles are directly
    comparable. Companion to harm_profiles() (which keeps the full per-angle, 3-metric
    figure for the appendix); both read the same grids, nothing here is overwritten."""
    import matplotlib; matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    try:
        import thesis_style as TS; TS.apply(); figw = TS.FIGW
    except Exception:
        figw = 6.3
    if sel is None:
        sel = [l for l in CURATED if l in labels] or labels[:3]
    use = [a for a in which if a in angles]
    if not use:                                            # fall back to a spread of what's present
        use = sorted({angles[0], angles[len(angles)//2], angles[-1]})
    col = _palette(labels)
    xc = (np.arange(NX)+0.5)*RES; yc = (np.arange(NY)+0.5)*RES; XX, YY = np.meshgrid(xc, yc)
    edges = np.arange(-6.0, 0.001, 0.25); cen = 0.5*(edges[:-1]+edges[1:])
    rows_meta = [('rho', 'peak density\n(ped m$^{-2}$)'),
                 ('contact', 'peak contact\n(N kg$^{-1}$)')]
    ncol = len(use); nrow = len(rows_meta)
    panel_h = (figw/ncol)*0.92*nrow; leg_h = 0.54; H = panel_h + leg_h + 0.55
    fig, axes = plt.subplots(nrow, ncol, squeeze=False, figsize=(figw, H),
                             sharex=True, sharey='row')
    any_data = False
    for ci, a in enumerate(use):
        g = B.geometry(int(a), WIDTH); O = np.array(g['O']); d2 = np.array(g['d2']); n2 = np.array(g['n2'])
        walk = MPath(g['poly']).contains_points(np.column_stack([XX.ravel(), YY.ravel()])).reshape(NY, NX)
        p = (XX-O[0])*d2[0]+(YY-O[1])*d2[1]; lat = (XX-O[0])*n2[0]+(YY-O[1])*n2[1]
        band = walk & (np.abs(lat) <= WIDTH/2)
        for lab in sel:
            cand = glob.glob(os.path.join(run, 'grids', f'a{a}_{lab}_lam*.csv'))
            if not cand: continue
            G = load_grid(cand[0])
            if G is None: continue
            any_data = True
            ser = {k: [] for k, _ in rows_meta}
            for i in range(len(cen)):
                m = band & (p >= edges[i]) & (p < edges[i+1])
                for k, _ in rows_meta:
                    mm = m & np.isfinite(G[k])
                    ser[k].append(float(np.nanmax(G[k][mm])) if mm.any() else np.nan)
            c, ls, lw, mk = col.get(lab, ('gray', '-', 1.6, 'o'))
            lw = 2.2 if lab == 'ctrl' else 1.5
            for ri, (k, _) in enumerate(rows_meta):
                axes[ri][ci].plot(cen, ser[k], color=c, lw=lw, zorder=5 if lab == 'ctrl' else 3)
            r = D.get((a, lab))                            # shade THIS config's pillar(s) in its colour
            if r:
                for (xk, yk, dk) in (('col_x', 'col_y', 'col_d'), ('col2_x', 'col2_y', 'col2_d')):
                    cx, cy, dd = _f(r.get(xk)), _f(r.get(yk)), _f(r.get(dk))
                    if np.isfinite(cx) and np.isfinite(cy) and np.isfinite(dd) and dd > 0:
                        pp = (cx-O[0])*d2[0] + (cy-O[1])*d2[1]
                        for ri in range(nrow):
                            axes[ri][ci].axvspan(pp-dd/2, pp+dd/2, color=c, alpha=0.13, lw=0, zorder=0)
        axes[0][ci].set_title(f'θ = {a}°', fontsize=11, fontweight='bold')
        for ri in range(nrow):
            ax = axes[ri][ci]
            ax.axvline(0.0, color='limegreen', lw=2.5, zorder=1)       # the exit
            ax.grid(alpha=0.3, zorder=0); ax.set_xlim(-5.0, 0.3)
            pidx = ri*ncol + ci
            ax.text(0.035, 0.93, f'({chr(97+pidx)})', transform=ax.transAxes, fontsize=9,
                    fontweight='bold', va='top',
                    bbox=dict(boxstyle='round,pad=0.15', fc='white', ec='none', alpha=0.7))
    for ri, (k, ylab) in enumerate(rows_meta):
        axes[ri][0].set_ylabel(ylab)
    for ci in range(ncol):
        axes[-1][ci].set_xlabel('distance from exit (m)')
    handles = [Line2D([0], [0], color=_color(l), lw=2.2 if l == 'ctrl' else 1.6) for l in sel]
    lbls = [_disp(l) for l in sel]
    fig.subplots_adjust(left=0.115, right=0.985, top=1 - 0.62/H, bottom=(leg_h + 0.55)/H,
                        wspace=0.10, hspace=0.16)
    fig.legend(handles, lbls, loc='lower center', ncol=min(3, len(sel)), fontsize=8.5,
               frameon=False, bbox_to_anchor=(0.5, 0.012),
               title='shaded band = each setup’s pillar(s);    green line = exit',
               title_fontsize=8.5)
    fig.suptitle('Where the crush sits along the corridor', fontsize=12)
    if not any_data:
        print('  (profile_grid: no grids found for the requested angles; skipped)'); plt.close(fig); return
    os.makedirs(outdir, exist_ok=True)
    out = os.path.join(outdir, 'obstacle_profile_grid')
    for ext in ('pdf', 'png'): fig.savefig(f'{out}.{ext}', bbox_inches='tight', dpi=150)
    plt.close(fig); print(f'wrote {out}.pdf/.png  (angles={use}, configs={sel})', flush=True)


def zone_table(run, D, angles, labels, outdir):
    """Casualties + peak contact split into BEFORE-pillar (upstream) vs DOOR (pillar->door)
    zones, per config, with Δ vs control -- quantifies relief vs relocation."""
    xc = (np.arange(NX)+0.5)*RES; yc = (np.arange(NY)+0.5)*RES; XX, YY = np.meshgrid(xc, yc)
    rows = []
    print('\n=== spatial harm split: UPSTREAM (before pillar) vs DOOR (pillar→door), from grids ===', flush=True)
    for a in angles:
        g = B.geometry(int(a), WIDTH); O = np.array(g['O']); d2 = np.array(g['d2']); n2 = np.array(g['n2'])
        walk = MPath(g['poly']).contains_points(np.column_stack([XX.ravel(), YY.ravel()])).reshape(NY, NX)
        p = (XX-O[0])*d2[0]+(YY-O[1])*d2[1]; lat = (XX-O[0])*n2[0]+(YY-O[1])*n2[1]
        band = walk & (np.abs(lat) <= WIDTH/2)
        ctrl = D.get((a, 'ctrl')); cu = cd = None
        print(f'\n  θ={a}°   (casualty = grid sum; contact = peak N/kg)', flush=True)
        for lab in labels:
            r = D.get((a, lab))
            if not r: continue
            cand = glob.glob(os.path.join(run, 'grids', f'a{a}_{lab}_lam*.csv'))
            if not cand: continue
            G = load_grid(cand[0])
            pp = _pillar_p(r, O, d2)
            door = band & (p >= pp) & (p <= 0.01) & np.isfinite(G['casualty'])
            up = band & (p >= pp-3.0) & (p < pp) & np.isfinite(G['casualty'])
            dcas = float(np.nansum(G['casualty'][door])); ucas = float(np.nansum(G['casualty'][up]))
            dC = float(np.nanmax(G['contact'][door])) if door.any() else 0.0
            uC = float(np.nanmax(G['contact'][up])) if up.any() else 0.0
            rec = dict(angle=a, col_label=lab, pillar_p=round(pp, 2),
                       door_cas=round(dcas, 2), up_cas=round(ucas, 2),
                       door_peakC=round(dC, 1), up_peakC=round(uC, 1))
            rows.append(rec)
            if lab == 'ctrl': cu, cd = ucas, dcas
            du = f'{100*(ucas-cu)/cu:+4.0f}%' if (cu and lab != 'ctrl') else '  ref' if lab == 'ctrl' else '   --'
            dd = f'{100*(dcas-cd)/cd:+4.0f}%' if (cd and lab != 'ctrl') else '  ref' if lab == 'ctrl' else '   --'
            print(f'    {lab:<15} DOOR cas={dcas:6.1f}({dd}) peakC={dC:4.0f}   |   UPSTREAM cas={ucas:6.1f}({du}) peakC={uC:4.0f}', flush=True)
    if rows:
        os.makedirs(outdir, exist_ok=True)
        with open(os.path.join(outdir, 'obstacle_zone_split.csv'), 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader()
            for r in rows: w.writerow(r)
        print(f'\nwrote {os.path.join(outdir, "obstacle_zone_split.csv")}', flush=True)


def appendix_grid(run, D, angles, labels, outdir, field='rho'):
    """All configurations (columns) x all turn angles (rows) in one grid — appendix reference."""
    import matplotlib; matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    try:
        import thesis_style as TS; TS.apply(); cmap = TS.CMAP
    except Exception:
        cmap = 'turbo'
    cols = labels
    grids = {}
    for a in angles:
        for l in cols:
            cand = glob.glob(os.path.join(run, 'grids', f'a{a}_{l}_lam*.csv'))
            if cand:
                G = load_grid(cand[0])
                if G is not None: grids[(a, l)] = G
    if not grids:
        return
    vmax = max(np.nanpercentile(G[field], 99) for G in grids.values())
    if not (np.isfinite(vmax) and vmax > 0): vmax = 1.0          # guard all-zero/all-NaN field
    nrow, ncol = len(angles), len(cols)
    fig, axes = plt.subplots(nrow, ncol, squeeze=False, constrained_layout=True,
                             figsize=(1.35*ncol, 1.45*nrow + 0.4))
    funit = {'rho': r'crowd density (ped m$^{-2}$)', 'contact': r'contact force (N kg$^{-1}$)'}.get(field, field)
    fname = 'density' if field == 'rho' else 'contact force'
    im = None
    for i, a in enumerate(angles):
        xl, yl = _crop(a)
        for j, l in enumerate(cols):
            ax = axes[i][j]; G = grids.get((a, l))
            # set the row/column labels BEFORE any skip, so a config missing at angle-0 still
            # gets its column header (otherwise the whole column is left anonymous)
            if i == 0: ax.set_title(_short(l), fontsize=8, fontweight='bold')
            if j == 0: ax.set_ylabel(f'θ = {a}°', fontsize=10, fontweight='bold')
            ax.set_xticks([]); ax.set_yticks([])
            if G is None: continue
            im = _imshow_clipped(ax, G[field], a, cmap, vmax)
            _overlay(ax, a, D.get((a, l)), wall_lw=1.5)
            ax.set_xlim(*xl); ax.set_ylim(*yl); ax.set_aspect('equal')
    if im is not None:
        cb = fig.colorbar(im, ax=axes, location='right', shrink=0.55, aspect=30); cb.set_label(funit, fontsize=10)
    fig.suptitle(f'Pillar configurations, steady-state crowd {fname}', fontsize=13, fontweight='bold')
    os.makedirs(outdir, exist_ok=True)
    out = os.path.join(outdir, f'obstacle_appendix_grid_{field}')
    for ext in ('pdf', 'png'): fig.savefig(f'{out}.{ext}', bbox_inches='tight', dpi=150)
    plt.close(fig); print(f'wrote {out}.pdf/.png', flush=True)


def write_key(labels, outdir):
    """Caption key: maps each display name -> exact pillar arrangement (CSV + stdout)."""
    os.makedirs(outdir, exist_ok=True)
    with open(os.path.join(outdir, 'config_key.csv'), 'w', newline='') as f:
        w = csv.writer(f); w.writerow(['label', 'display_name', 'short', 'arrangement'])
        for l in labels: w.writerow([l, _disp(l), _short(l), _arr(l)])
    print('\n=== configuration key (for figure captions) ===', flush=True)
    for l in labels:
        print(f'  {_disp(l):<26} : {_arr(l)}', flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--run', required=True)
    ap.add_argument('--outdir', default=None)
    ap.add_argument('--field', default='both', choices=['rho', 'contact', 'both'])
    ap.add_argument('--maps-all', action='store_true',
                    help='per-angle maps show ALL configs (two rows) instead of the curated 3')
    ap.add_argument('--angle', type=int, default=None, help='only this angle for the maps')
    ap.add_argument('--profile-angles', type=int, nargs='+', default=[0, 90, 135],
                    help='angles (columns) of the multi-angle profile_grid main figure')
    args = ap.parse_args()
    run = args.run if os.path.isabs(args.run) else os.path.join(ROOT, args.run)
    outdir = args.outdir or os.path.join(run, 'figs')
    D, angles, labels = load_aggregated(run)
    # inject the column-excluded door contact (clean metric; ctrl is unmasked == capov field)
    for (a, l), r in list(D.items()):
        ec = excluded_contact(run, a, l, r)
        r['door_contact_excol_mean'] = round(ec, 3) if ec == ec else ''
        r['door_contact_excol_ci95'] = 0.0
    print(f'run={run}\n  angles={angles}\n  configs={labels}', flush=True)
    write_key(labels, outdir)
    summary_table(D, angles, labels, outdir)
    zone_table(run, D, angles, labels, outdir)
    benefit_vs_angle(D, angles, labels, outdir)
    harm_profiles(run, D, angles, labels, outdir)
    profile_grid(run, D, angles, labels, outdir, which=args.profile_angles)
    maps_cfgs = labels if args.maps_all else CURATED
    for fld in (['rho', 'contact'] if args.field == 'both' else [args.field]):
        density_maps(run, D, angles, labels, outdir, field=fld, only_angle=args.angle, cfgs=maps_cfgs)
        appendix_grid(run, D, angles, labels, outdir, field=fld)
    print(f'\nfigures -> {outdir}', flush=True)


if __name__ == '__main__':
    main()
