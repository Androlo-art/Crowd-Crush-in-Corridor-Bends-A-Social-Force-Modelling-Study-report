"""OBSTACLE-MITIGATION study: can an obstacle relieve the bend-fed exit crush?

Closes the thesis mitigation loop. The chamfer chapter showed the lethal crush is
owned by the RESTRICTED EXIT and that rounding the corner does not relieve it. Here
we place an obstacle (a single asymmetric pillar, or a two-column "gate") UPSTREAM of
the 1.2 m bend-fed door and ask:

  * does an obstacle reduce the peak door density / contact / casualties at the exit?
  * does the benefit depend on the TURN ANGLE (the bend already meters demand to the
    door -- capov d*(theta) -- so an obstacle should help most where the turn delivers
    the most demand, i.e. straight/gentle, and less as the turn throttles the flow)?
  * single pillar vs gate, and where (the room pillar study found a GATE beats a single
    column and that tandem in-line FAILS -- does that transfer to the bend-fed door?)

WHY THESE PLACEMENTS (not arbitrary): an obstacle relieves a bottleneck by breaking the
PRESSURE ARCH that forms in the converging zone ~1-2 door-widths UPSTREAM of the exit.
So obstacles live in that arch zone (d_up ~ 1.2-2.4 m), NOT at the door mouth (a column
0.5 m from a 1.2 m door merely throttles it). We SWEEP position/size/count over that
zone to MAP the response and find the optimum -- the optimum (and its shift with angle)
is the result, so no single position is asserted.

SIZES: 0.8-1.1 m = a realistic large-building structural column / deliberate crowd
wave-breaker. Gates straddle the centreline with an inter-column GAP ~= the door width
(the room-study winner geometry); a gap >> door does little, a gap << door becomes a
second bottleneck. Corridor half-width 2.5 m, door 1.2 m.

MODEL CHOICE -- purely physical column (NOT in the FMM field): this is the
Helbing-Farkas-Vicsek (2000) treatment and is correct for the dense PUSHING/crush regime
we study -- the arch-breaking is a CONTACT-mechanics effect, not a wayfinding one.
Masking the column into the navigation field would model a calm informed crowd that
pre-avoids it and would not capture arch-breaking. Validity requires the column to sit
in the door's convergence zone (verified in smoke: agents deflect smoothly around it,
no back-face pile, no field-shadow stall) -- which the arch-zone placement guarantees.

Physics: ScenarioID==6 with the Helbing column un-gated (sd_lib_capacity_obs.c). Chamfered
inner corner r=0.25 (free of the sharp-vertex artefact). Door fixed at 1.2 m. Saturating
inflow lambda=12. Measurement (masks/ROIs/metrics/window/aggregation) is IDENTICAL to
run_capacity_chamfer.py, so the no-column control reproduces capovCH and with-vs-without
is directly comparable. Steady window [200,400] s. Nothing original is touched.

Usage:
  smoke : python -u bend_bottleneck/run_capacity_obstacle.py --smoke
  gate-smoke: python -u bend_bottleneck/run_capacity_obstacle.py --smoke --cfgs ctrl,gate
  full  : python -u bend_bottleneck/run_capacity_obstacle.py --angles 0,45,90,135 --seeds 12 --jobs 12
"""
import os, sys, csv, time, math, json, argparse, datetime, platform
import multiprocessing as mp
import numpy as np
from scipy.ndimage import gaussian_filter
from matplotlib.path import Path as MPath

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, os.path.join(ROOT, 'model')); sys.path.insert(0, os.path.join(ROOT, 'plotting'))
from sfm.backends import c_exe
import make_field_bottleneck as F          # geometry() for masks + column placement

EXE       = os.path.join(ROOT, 'model', 'c_core', 'bend', 'sd_crunch_capacity_obs')  # column un-gated into ID6
CAP_PAR   = os.path.join(ROOT, 'model', 'c_core', 'bend', 'sd_capacity.par')          # has Column* + BendDoorWidth tokens
FIELD_ROOT = os.path.join(ROOT, '..', 'results', 'bend_capacity_fields')           # reuse chamfered fields (column is physical-only)
OUT_ROOT  = os.path.join(ROOT, '..', 'results', 'bend_obstacle')
CORNER_R  = 0.25
RTAG      = int(CORNER_R*10 + 0.5)           # = 3
DOOR      = 1.2                              # validated lethal door (fixed)

WIDTH = 5.0; V0 = 1.3; RES = 0.25; DOMX, DOMY = 18.0, 32.0   # identical to capov/phaseB
NX, NY = int(DOMX/RES), int(DOMY/RES)
XC = (np.arange(NX)+0.5)*RES; YC = (np.arange(NY)+0.5)*RES
CORNER_HALF = 1.6; DOOR_BACK = 2.0
F_CRUSH = 200.0
INFLOW_CAP = 1000
T_TOTAL = 400.0; T_MIN = 200.0
CELL_TIMEOUT = 600

ANGLES = [0, 45, 90, 135]                    # span the turn range -> angle-dependence of the benefit
LAMBDAS = [12.0]

# Each config: cols = list of (d_up [m upstream of door], offset [m lateral], D_c [m]).
#   len(cols) = ColumnSwitch (0 / 1 / 2). Centre = O - d_up*d2 + offset*n2 (angle-general).
# Placement principle: arch zone ~1-2 door-widths upstream; sizes 0.8-1.1 m; gate gap ~= door.
COLCFGS = [
    dict(label='ctrl',          cols=[]),                                       # no column -> reproduces capovCH
    dict(label='pillar_axis',   cols=[(1.5,  0.0, 0.9)]),                        # symmetric single pillar, arch zone
    dict(label='pillar_off',    cols=[(1.5,  0.8, 0.9)]),                        # Helbing asymmetric (toward inner wall)
    dict(label='pillar_off_far',cols=[(2.2,  0.8, 0.9)]),                        # asymmetric, further upstream
    dict(label='pillar_big',    cols=[(2.0,  0.8, 1.1)]),                        # larger pillar
    dict(label='gate',          cols=[(1.8,  1.0, 0.9), (1.8, -1.0, 0.9)]),      # 2-col gate, gap ~1.1 m ~= door
    dict(label='gate_wide',     cols=[(1.8,  1.25,0.9), (1.8, -1.25,0.9)]),      # wider gate, gap ~1.6 m
    dict(label='gate_close',    cols=[(1.2,  1.0, 0.8), (1.2, -1.0, 0.8)]),      # gate closer to door, gap ~1.2 m
]

ROW = ['angle','col_label','col_switch','col_x','col_y','col_d','col2_x','col2_y','col2_d',
       'door_width','lam','seed','status','timed_out','sim_t','mean_n','steady','door_rho_steady',
       'door_rho_peak','corner_rho','door_contact_p95','door_contact_max','crush_frac',
       'casualties','door_outflow','inflow_rate','n_end']


def _dtag(d): return int(round(d * 10.0))


def col_center(angle, d_up, offset):
    """Column centre d_up m upstream of the door, offset m laterally -- angle-general."""
    g = F.geometry(int(angle), WIDTH)
    O = np.array(g['O'], float); d2 = np.array(g['d2'], float); n2 = np.array(g['n2'], float)
    c = O - d_up*d2 + offset*n2
    return float(c[0]), float(c[1])


def cfg_centers(angle, cfg):
    """List of (cx, cy, D) for every column in cfg (empty for ctrl)."""
    return [(*col_center(angle, du, off), D) for (du, off, D) in cfg['cols']]


def _check_field(angle):
    fdir = os.path.join(FIELD_ROOT, f'd{_dtag(DOOR)}')
    gj = os.path.join(fdir, 'geometry.json')
    if not os.path.isdir(fdir) or not os.path.exists(gj):
        raise ValueError(f'missing field folder/geometry.json for door {DOOR}: {fdir}')
    egress = json.load(open(gj)).get('egress', None)
    if egress is None or abs(float(egress) - float(DOOR)) > 0.05:
        raise ValueError(f'egress mismatch in {gj}: {egress} != {DOOR}')
    ff = os.path.join(fdir, f'ff_field_a{int(round(angle))}_r{RTAG}.csv')
    if not os.path.exists(ff):
        raise ValueError(f'missing field {ff} for angle {angle}')
    return fdir, ff


def _winit(field_root, t_total, t_min, cell_timeout, exe, cap_par):
    globals()['FIELD_ROOT'] = field_root
    globals()['T_TOTAL'] = t_total; globals()['T_MIN'] = t_min
    globals()['CELL_TIMEOUT'] = cell_timeout; globals()['EXE'] = exe
    c_exe.PAR_FILE = cap_par
    os.environ.setdefault('SFM_BEND_FIELD_DIR', field_root)


def _masks(angle):
    """IDENTICAL to run_capacity_chamfer._masks: full-width door approach band + corner ROI."""
    g = F.geometry(angle, WIDTH)
    walk = MPath(g['poly']).contains_points(
        np.column_stack([np.meshgrid(XC, YC)[0].ravel(),
                         np.meshgrid(XC, YC)[1].ravel()])).reshape(NY, NX)
    O = np.array(g['O']); d2 = np.array(g['d2']); n2 = np.array(g['n2'])
    XX, YY = np.meshgrid(XC, YC)
    p = (XX-O[0])*d2[0]+(YY-O[1])*d2[1]
    lat = (XX-O[0])*n2[0]+(YY-O[1])*n2[1]
    door = walk & (p <= 0) & (p >= -DOOR_BACK) & (np.abs(lat) <= WIDTH/2)
    if g['Vin'] is not None:
        vx, vy = g['Vin']
        corner = walk & (np.abs(XX-vx) <= CORNER_HALF) & (np.abs(YY-vy) <= CORNER_HALF)
    else:
        corner = np.zeros((NY, NX), bool)
    return g, walk, door, corner


def _smooth(field, walk):
    wf = walk.astype(float); sw = gaussian_filter(wf, 1.0)
    return gaussian_filter(np.where(walk, field, 0.0), 1.0) / np.maximum(sw, 1e-6)


def _nan(angle, cfg, lam, seed, status, sim_t):
    r = {k: '' for k in ROW}
    r.update(angle=angle, col_label=cfg['label'], col_switch=len(cfg['cols']),
             door_width=DOOR, lam=lam, seed=seed, status=status,
             timed_out=0, sim_t=round(sim_t, 2), mean_n=0)
    return r


def run_one(task, snapdir=None):
    angle, ci, lam, seed = task
    cfg = COLCFGS[ci]; label = cfg['label']
    try:
        field_dir, ff = _check_field(angle)
    except ValueError as ex:
        print(f'  !! {ex}', flush=True)
        return _nan(angle, cfg, lam, seed, 'no_field', 0.0), None
    os.environ['SFM_BEND_FIELD_DIR'] = field_dir

    centers = cfg_centers(angle, cfg); sw = len(centers)
    g, walk, door_m, corner = _masks(angle)
    ov = dict(InjurySwitch=5, CasualtyRadiusScale=0.6, CasualtyForceScale=1.0,
              CasualtyContactScale=1.0, CasualtyFrictionScale=1.0, StallKickEnable=0.0,
              UseFloorField=1.0, BendAngle=float(angle), BendWidth=WIDTH, BendCornerR=CORNER_R,
              BendDoorWidth=DOOR, InflowRate=float(lam), InitialN=0.0, ColumnSwitch=int(sw))
    if sw >= 1:
        ov.update(ColumnCenterX=centers[0][0], ColumnCenterY=centers[0][1], ColumnD=centers[0][2])
    if sw == 2:
        ov.update(Column2CenterX=centers[1][0], Column2CenterY=centers[1][1], Column2D=centers[1][2])
    case = {'label': 'obs', 'geometry': 'fis_room', 'scenario_id': 6,
            'params': dict(N0=INFLOW_CAP, MaxSimTime=T_TOTAL, V_ChangeLimit=0.05, DoorWidth=1.0)}
    stats = {}; t0 = time.time()
    try:
        _t, _f, frames, _e = c_exe.run_with_trajectory(case, V0, seed, traj_dt=0.5,
            max_sim_time=T_TOTAL, overrides=ov, exe=EXE, timeout_s=CELL_TIMEOUT,
            full_traj=True, stats=stats)
    except Exception as ex:
        print(f'  !! run error a{angle} {label} s{seed}: {ex}', flush=True)
        return _nan(angle, cfg, lam, seed, 'error', time.time()-t0), None
    sim_t = time.time()-t0; timed_out = int(sim_t >= CELL_TIMEOUT-1)
    win = [(t, a) for t, a in frames if t >= T_MIN and a]
    acc = dict(count=np.zeros((NY, NX)), csum=np.zeros((NY, NX)), inj=np.zeros((NY, NX)),
               ns=np.zeros((NY, NX)), vx=np.zeros((NY, NX)), vx2=np.zeros((NY, NX)),
               vy=np.zeros((NY, NX)), vy2=np.zeros((NY, NX)), nfr=0, angle=angle, label=label)
    if snapdir:
        _snapshot(frames, angle, cfg, centers,
                  os.path.join(snapdir, f'snap_a{int(angle)}_{label}_s{seed}.png'))
    cxyz = [round(v, 3) for v in centers[0]] if sw >= 1 else ['', '', '']
    c2xyz = [round(v, 3) for v in centers[1]] if sw == 2 else ['', '', '']
    if not win:
        r = _nan(angle, cfg, lam, seed, 'locked' if timed_out else 'empty', sim_t)
        r.update(col_x=cxyz[0], col_y=cxyz[1], col_d=cxyz[2], col2_x=c2xyz[0], col2_y=c2xyz[1], col2_d=c2xyz[2])
        return r, acc
    door_peak = 0.0; cas = []; dcontacts = []; dcrush = 0; dn = 0; ns_in = []
    for t, ags in win:
        ns_in.append(len(ags))
        H, _, _ = np.histogram2d([a[1] for a in ags], [a[2] for a in ags],
                                 bins=[NX, NY], range=[[0, DOMX], [0, DOMY]])
        acc['count'] += H.T
        fr = _smooth(H.T/(RES*RES), walk)
        door_peak = max(door_peak, float(np.nanmax(fr[door_m])) if door_m.any() else 0.0)
        cas.append(sum(1 for a in ags if a[0] == 1))
        for a in ags:
            inj, x, y, d, vx, vy, tp, contact = a
            if not (0 <= x <= DOMX and 0 <= y <= DOMY): continue
            jx = min(NX-1, int(x/RES)); jy = min(NY-1, int(y/RES))
            acc['ns'][jy, jx] += 1; acc['csum'][jy, jx] += contact
            acc['vx'][jy, jx] += vx; acc['vx2'][jy, jx] += vx*vx
            acc['vy'][jy, jx] += vy; acc['vy2'][jy, jx] += vy*vy
            if inj == 1: acc['inj'][jy, jx] += 1
            if door_m[jy, jx]:
                dcontacts.append(contact); dn += 1
                if contact > F_CRUSH*math.pi*d: dcrush += 1
        acc['nfr'] += 1
    rho_s = _smooth(acc['count']/(acc['nfr']*RES*RES), walk)
    door_steady = float(np.nanmax(rho_s[door_m])) if door_m.any() else 0.0
    corner_rho = float(np.nanmax(rho_s[corner])) if corner.any() else float('nan')
    n_end = len(win[-1][1]); succ = stats.get('success', 0)
    if len(ns_in) >= 4:
        h = len(ns_in)//2; f1 = float(np.mean(ns_in[:h])); f2 = float(np.mean(ns_in[h:]))
        steady = int(abs(f2-f1)/max(f1, 1e-6) < 0.15)
    else:
        steady = 0
    row = dict(angle=angle, col_label=label, col_switch=sw,
               col_x=cxyz[0], col_y=cxyz[1], col_d=cxyz[2], col2_x=c2xyz[0], col2_y=c2xyz[1], col2_d=c2xyz[2],
               door_width=DOOR, lam=lam, seed=seed, status='ok', timed_out=timed_out, sim_t=round(sim_t, 2),
               mean_n=round(float(np.mean(ns_in)), 1), steady=steady,
               door_rho_steady=round(door_steady, 3), door_rho_peak=round(door_peak, 3),
               corner_rho=round(corner_rho, 3) if np.isfinite(corner_rho) else '',
               door_contact_p95=round(float(np.percentile(dcontacts, 95)), 1) if dcontacts else 0.0,
               door_contact_max=round(float(np.max(dcontacts)), 1) if dcontacts else 0.0,
               crush_frac=round(dcrush/max(dn, 1), 4), casualties=round(float(np.mean(cas)), 2),
               door_outflow=round(max(0, succ-n_end)/T_TOTAL, 3),
               inflow_rate=round(succ/T_TOTAL, 3), n_end=n_end)
    return row, acc


def _snapshot(frames, angle, cfg, centers, outpath):
    """Last-frame scatter + corridor walls + door + column(s) -- geometry sanity (smoke only)."""
    import matplotlib; matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    if not frames:
        return
    t, ags = frames[-1]
    g = F.geometry(int(angle), WIDTH)
    poly = g['poly'] + [g['poly'][0]]
    dlo, dhi = g['door_lo'], g['door_hi']
    fig, ax = plt.subplots(figsize=(5.2, 8.6))
    ax.plot([p[0] for p in poly], [p[1] for p in poly], 'k-', lw=1.4)
    ax.plot([dlo[0], dhi[0]], [dlo[1], dhi[1]], color='lime', lw=4)
    lx = [a[1] for a in ags if a[0] == 0]; ly = [a[2] for a in ags if a[0] == 0]
    ix = [a[1] for a in ags if a[0] == 1]; iy = [a[2] for a in ags if a[0] == 1]
    if lx: ax.scatter(lx, ly, s=12, c='steelblue', edgecolor='none', zorder=3)
    if ix: ax.scatter(ix, iy, s=26, c='red', marker='x', zorder=4)
    for (cx, cy, Dc) in centers:
        ax.add_patch(plt.Circle((cx, cy), Dc/2, color='darkorange', alpha=0.9, zorder=5))
        ax.add_patch(plt.Circle((cx, cy), Dc/2, fill=False, ec='k', lw=1.2, zorder=6))
    ax.set(xlim=(0, DOMX), ylim=(0, DOMY), aspect='equal')
    ax.set_title(f'θ={int(angle)}°  {cfg["label"]}  t={t:.0f}s  N={len(ags)}  inj={len(ix)}', fontsize=10)
    os.makedirs(os.path.dirname(outpath), exist_ok=True)
    fig.savefig(outpath, dpi=130, bbox_inches='tight'); plt.close(fig)
    print(f'      snapshot -> {os.path.relpath(outpath, ROOT)}', flush=True)


def _fnum(r, k):
    try: return float(r[k])
    except (ValueError, TypeError, KeyError): return float('nan')


def _aggregate(rows, outdir):
    from collections import defaultdict
    grp = defaultdict(list)
    for r in rows:
        if r['status'] == 'ok': grp[(r['angle'], r['col_label'], r['lam'])].append(r)
    metrics = ['inflow_rate', 'door_outflow', 'door_rho_steady', 'door_rho_peak', 'corner_rho',
               'door_contact_p95', 'door_contact_max', 'crush_frac', 'casualties', 'mean_n']
    out = []
    for (a, lab, lm), rs in sorted(grp.items(), key=lambda kv: (kv[0][0], str(kv[0][1]))):
        rec = dict(angle=a, col_label=lab, lam=lm, n_seeds=len(rs), col_switch=rs[0]['col_switch'],
                   col_x=rs[0].get('col_x', ''), col_y=rs[0].get('col_y', ''), col_d=rs[0].get('col_d', ''),
                   col2_x=rs[0].get('col2_x', ''), col2_y=rs[0].get('col2_y', ''), col2_d=rs[0].get('col2_d', ''))
        for m in metrics:
            v = np.array([_fnum(r, m) for r in rs]); v = v[np.isfinite(v)]
            rec[f'{m}_mean'] = round(float(v.mean()), 4) if v.size else ''
            rec[f'{m}_ci95'] = round(float(1.96*v.std(ddof=1)/math.sqrt(len(v))), 4) if v.size > 1 else 0.0
        out.append(rec)
    if out:
        with open(os.path.join(outdir, 'aggregated.csv'), 'w', newline='') as f:
            wa = csv.DictWriter(f, fieldnames=list(out[0].keys())); wa.writeheader()
            for rec in out: wa.writerow(rec)


def _dump_grids(cond_acc, outdir):
    gd = os.path.join(outdir, 'grids'); os.makedirs(gd, exist_ok=True)
    for (angle, label, lam), C in cond_acc.items():
        g, walk, door_m, corner = _masks(angle)
        nfr = max(C['nfr'], 1)
        rho = _smooth(C['count']/(nfr*RES*RES), walk)
        contact = _smooth(np.where(C['ns'] > 0, C['csum']/np.maximum(C['ns'], 1), 0.0), walk)
        cas = _smooth(C['inj']/(nfr*RES*RES), walk)
        with np.errstate(invalid='ignore'):
            vxm = np.where(C['ns'] > 0, C['vx']/np.maximum(C['ns'], 1), 0.0)
            vym = np.where(C['ns'] > 0, C['vy']/np.maximum(C['ns'], 1), 0.0)
            var = np.maximum(np.where(C['ns'] > 1, C['vx2']/np.maximum(C['ns'], 1)-vxm**2, 0) +
                             np.where(C['ns'] > 1, C['vy2']/np.maximum(C['ns'], 1)-vym**2, 0), 0)
        pres = _smooth(C['count']/(nfr*RES*RES)*var, walk)
        with open(os.path.join(gd, f'a{angle}_{label}_lam{lam}.csv'), 'w', newline='') as f:
            wr = csv.writer(f); wr.writerow(['x', 'y', 'rho', 'contact', 'casualty', 'pressure'])
            for iy in range(NY):
                for ix in range(NX):
                    if walk[iy, ix]:
                        wr.writerow([f'{XC[ix]:.2f}', f'{YC[iy]:.2f}', f'{rho[iy, ix]:.4f}',
                                     f'{contact[iy, ix]:.4f}', f'{cas[iy, ix]:.5f}', f'{pres[iy, ix]:.5f}'])


def _runinfo(outdir, angles, lambdas, seeds, t_all):
    with open(os.path.join(outdir, 'run_info.txt'), 'w') as f:
        f.write(f'OBSTACLE-MITIGATION study  {datetime.datetime.now().isoformat(timespec="seconds")}\n')
        f.write(f'python={platform.python_version()} platform={platform.platform()}\n')
        f.write(f'width={WIDTH} door={DOOR} corner_r={CORNER_R} V0={V0} T_TOTAL={T_TOTAL} '
                f'T_MIN={T_MIN} RES={RES} domain={DOMX}x{DOMY} cell_timeout={CELL_TIMEOUT}\n')
        f.write(f'angles={angles}\nlambdas={lambdas}\nseeds={seeds}\n')
        f.write('colcfgs (label : cols [(d_up,offset,D), ...]):\n')
        for c in COLCFGS: f.write(f'  {c["label"]:<14}: {c["cols"]}\n')
        f.write(f'exe={EXE}\npar={CAP_PAR}\nfield_root={FIELD_ROOT}\n')
        f.write('model: ScenarioID=6, chamfered corner r=0.25, Helbing column un-gated into ID6 '
                '(sd_lib_capacity_obs.c), purely physical (no FMM mask), InjurySwitch=5 soft-radius 0.6, '
                'StallKickEnable=0, controlled inflow lambda, door=1.2 m, steady window [200,400]s.\n')
        f.write(f'wall_time={time.time()-t_all:.0f}s\n')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--seeds', type=int, default=12)
    ap.add_argument('--jobs', type=int, default=os.cpu_count())
    ap.add_argument('--angles', default=None, help='comma list, e.g. 0,45,90,135')
    ap.add_argument('--lambdas', default=None, help='comma list, e.g. 12')
    ap.add_argument('--cfgs', default=None, help='comma list of COLCFG labels (default: all)')
    ap.add_argument('--smoke', action='store_true',
                    help='tiny set + geometry snapshots (default: a{90,0} x {ctrl,pillar_off,gate} x seed0)')
    ap.add_argument('--t-total', type=float, default=T_TOTAL)
    ap.add_argument('--t-min', type=float, default=T_MIN)
    ap.add_argument('--cell-timeout', type=float, default=CELL_TIMEOUT)
    ap.add_argument('--field-root', default=FIELD_ROOT)
    ap.add_argument('--exe', default=EXE)
    ap.add_argument('--par', default=CAP_PAR)
    ap.add_argument('--tag', default='obs')
    args = ap.parse_args()
    globals()['EXE'] = os.path.abspath(args.exe if os.path.isabs(args.exe) else os.path.join(ROOT, args.exe))
    globals()['CAP_PAR'] = os.path.abspath(args.par)
    globals()['FIELD_ROOT'] = os.path.abspath(args.field_root)
    globals()['T_TOTAL'] = args.t_total; globals()['T_MIN'] = args.t_min
    globals()['CELL_TIMEOUT'] = args.cell_timeout
    c_exe.PAR_FILE = CAP_PAR

    angles = ANGLES; lambdas = LAMBDAS; seeds = list(range(args.seeds))
    cfg_idx = list(range(len(COLCFGS)))
    if args.angles:  angles = [int(float(x)) for x in args.angles.split(',')]
    if args.lambdas: lambdas = [float(x) for x in args.lambdas.split(',')]
    if args.cfgs:
        want = {s.strip() for s in args.cfgs.split(',')}
        cfg_idx = [i for i, c in enumerate(COLCFGS) if c['label'] in want]

    problems = []
    for a in angles:
        try: _check_field(a)
        except ValueError as ex: problems.append(str(ex))
    if problems:
        raise SystemExit('FIELD PRE-FLIGHT FAILED:\n  ' + '\n  '.join(problems))

    stamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')

    # ---------- SMOKE: single-process, snapshots, print metrics ----------
    if args.smoke:
        sangles = [a for a in (90, 0) if a in angles] or angles[:2]
        slabels = ['ctrl', 'pillar_off', 'gate'] if not args.cfgs else [COLCFGS[i]['label'] for i in cfg_idx]
        sidx = [i for i, c in enumerate(COLCFGS) if c['label'] in slabels]
        snapdir = os.path.join(OUT_ROOT, f'smoke_{stamp}')
        os.makedirs(snapdir, exist_ok=True)
        print(f'SMOKE  exe={os.path.basename(EXE)}  angles={sangles}  cfgs={slabels}  -> {snapdir}', flush=True)
        recs = []
        for a in sangles:
            for ci in sidx:
                t0 = time.time()
                row, _ = run_one((a, ci, 12.0, 0), snapdir=snapdir)
                recs.append(row)
                print(f'  a{a:>3} {row["col_label"]:<14} sw={row.get("col_switch")} | {row["status"]:>6} '
                      f'casualties={row.get("casualties")} door_peak={row.get("door_rho_peak")} '
                      f'door_steady={row.get("door_rho_steady")} contact_p95={row.get("door_contact_p95")} '
                      f'mean_n={row.get("mean_n")} ({time.time()-t0:.0f}s)', flush=True)
        with open(os.path.join(snapdir, 'smoke_metrics.csv'), 'w', newline='') as fh:
            w = csv.DictWriter(fh, fieldnames=ROW); w.writeheader()
            for r in recs: w.writerow(r)
        print(f'\nSMOKE done -> {snapdir}', flush=True)
        return

    # ---------- FULL SWEEP (you run this; heavy) ----------
    tasks = [(a, ci, lm, s) for a in angles for ci in cfg_idx for lm in lambdas for s in seeds]
    outdir = os.path.join(OUT_ROOT, f'{args.tag}_{stamp}'); os.makedirs(outdir, exist_ok=True)
    raw = os.path.join(outdir, 'raw_results.csv')
    print(f'OBSTACLE STUDY  angles={angles} cfgs={[COLCFGS[i]["label"] for i in cfg_idx]} '
          f'lam={lambdas} seeds={len(seeds)} -> {len(tasks)} runs  jobs={args.jobs}\n'
          f'  exe={EXE}\n  field_root={FIELD_ROOT}\n  out={outdir}', flush=True)
    rows = []; cond_acc = {}
    fh = open(raw, 'w', newline=''); w = csv.DictWriter(fh, fieldnames=ROW); w.writeheader(); fh.flush()
    t_all = time.time(); n = 0
    with mp.Pool(args.jobs, initializer=_winit,
                 initargs=(FIELD_ROOT, args.t_total, args.t_min, args.cell_timeout, EXE, CAP_PAR)) as pool:
        for row, acc in pool.imap_unordered(run_one, tasks):
            w.writerow(row); fh.flush(); rows.append(row); n += 1
            if acc is not None and acc['nfr'] > 0:
                key = (acc['angle'], acc['label'], row['lam'])
                C = cond_acc.setdefault(key, {k: np.zeros((NY, NX)) for k in
                                              ('count', 'csum', 'inj', 'ns', 'vx', 'vx2', 'vy', 'vy2')})
                C['nfr'] = C.get('nfr', 0)+acc['nfr']
                for kk in ('count', 'csum', 'inj', 'ns', 'vx', 'vx2', 'vy', 'vy2'): C[kk] += acc[kk]
            print(f'  [{n}/{len(tasks)}] a{row["angle"]} {row["col_label"]} s{row["seed"]} '
                  f'{row["status"]} cas={row.get("casualties")} door_peak={row.get("door_rho_peak")} '
                  f'({row.get("sim_t")}s)', flush=True)
    fh.close()
    _aggregate(rows, outdir); _dump_grids(cond_acc, outdir); _runinfo(outdir, angles, lambdas, seeds, t_all)
    print(f'\nDONE -> {outdir}', flush=True)


if __name__ == '__main__':
    main()
