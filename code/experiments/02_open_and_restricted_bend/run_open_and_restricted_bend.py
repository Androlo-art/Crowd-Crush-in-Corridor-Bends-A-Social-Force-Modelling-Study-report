"""Stage-2 bend + 1.2 m door BOTTLENECK study (crush regime).

Sweeps turn angle (incl. 0 = straight control) under saturating inflow, measures
crush at the DOOR and at the inner CORNER, and asks: does an upstream bend make
the crush worse / more asymmetric than a straight approach to the same door?

Self-contained; uses the NEW binary bend_bottleneck/sd_crunch_bottleneck via
c_exe(exe=...) and the door floor-fields in results/bend_bottleneck_fields/.
NOTHING in the original code is touched. Drive the crush with INFLOW (not dense
pre-seed, which hangs in placement).

Outputs: results/bend_bottleneck/phaseB_<YYYYMMDD_HHMMSS>/
  raw_results.csv  aggregated.csv  run_info.txt  grids/  fig_*.{png,pdf,svg}

Usage (M-series, many workers):
  smoke : python -u bend_bottleneck/run_bottleneck_study.py --reduced --jobs 8
  full  : python -u bend_bottleneck/run_bottleneck_study.py --seeds 12 --jobs 14
"""
import os, sys, csv, time, math, argparse, datetime, platform
import multiprocessing as mp
import numpy as np
from scipy.ndimage import gaussian_filter
from matplotlib.path import Path as MPath

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, os.path.join(ROOT, 'model')); sys.path.insert(0, os.path.join(ROOT, 'plotting'))
from sfm.backends import c_exe
import make_field_bottleneck as F
import plot_bottleneck as P

EXE = os.path.join(ROOT, 'model', 'c_core', 'bend', 'sd_crunch_bottleneck')
FIELD_DIR = os.path.join(ROOT, '..', 'results', 'bend_bottleneck_fields')
OUT_ROOT = os.path.join(ROOT, '..', 'results', 'bend_bottleneck')

WIDTH = 5.0; V0 = 1.3; RES = 0.25; DOMX, DOMY = 18.0, 32.0   # match make_field_bottleneck.py (LB=12, long outlet)
NX, NY = int(DOMX/RES), int(DOMY/RES)
XC = (np.arange(NX)+0.5)*RES; YC = (np.arange(NY)+0.5)*RES
CORNER_HALF = 1.6; DOOR_BACK = 2.0           # door ROI: 2 m upstream of the cap
F_CRUSH = 200.0                              # crush when contact > F_CRUSH*pi*D
INFLOW_CAP = 1000
# Longer than before: at sharp angles the door jam backs up late, and the longer
# outlet (LB=12) lengthens the corridor further, so the steady jam forms even later.
# Measure a long steady window after full backup; the `steady` flag below verifies it.
T_TOTAL = 400.0; T_MIN = 200.0   # measure [200, 400] s: a 200 s steady window after the late high-angle backup along the longer corridor
CELL_TIMEOUT = 420               # wall-clock cap (s); a 400 s sim runs in ~110 s wall on the M5, so this is generous headroom

ANGLES = [0, 45, 60, 90, 120, 135]
LAMBDAS = [4.0, 8.0]

ROW = ['angle','lam','seed','status','timed_out','sim_t','mean_n','steady',
       'door_rho_steady','door_rho_peak','corner_rho','door_contact_p95',
       'door_contact_max','crush_frac','casualties','door_outflow','inflow_rate','n_end']


def _winit(field_dir, t_total, t_min, cell_timeout, exe):
    os.environ['SFM_BEND_FIELD_DIR'] = field_dir
    globals()['T_TOTAL'] = t_total
    globals()['T_MIN'] = t_min
    globals()['CELL_TIMEOUT'] = cell_timeout
    globals()['EXE'] = exe          # allow the open-bend binary to be swapped in (same measurement code)


def _masks(angle):
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


def run_one(task):
    angle, lam, seed = task
    g, walk, door, corner = _masks(angle)
    ov = dict(InjurySwitch=5, CasualtyRadiusScale=0.6, CasualtyForceScale=1.0,
              CasualtyContactScale=1.0, CasualtyFrictionScale=1.0, StallKickEnable=0.0,
              UseFloorField=1.0, BendAngle=float(angle), BendWidth=WIDTH, BendCornerR=0.0,
              InflowRate=float(lam), InitialN=0.0)
    case = {'label':'bn','geometry':'fis_room','scenario_id':6,
            'params':dict(N0=INFLOW_CAP, MaxSimTime=T_TOTAL, V_ChangeLimit=0.05, DoorWidth=1.0)}
    stats = {}; t0 = time.time()
    try:
        _t,_f,frames,_e = c_exe.run_with_trajectory(case, V0, seed, traj_dt=0.5,
            max_sim_time=T_TOTAL, overrides=ov, exe=EXE, timeout_s=CELL_TIMEOUT,
            full_traj=True, stats=stats)
    except Exception as ex:
        return _nan(angle, lam, seed, 'error', time.time()-t0), None
    sim_t = time.time()-t0; timed_out = int(sim_t >= CELL_TIMEOUT-1)
    win = [(t,a) for t,a in frames if t >= T_MIN and a]
    acc = dict(count=np.zeros((NY,NX)), csum=np.zeros((NY,NX)), inj=np.zeros((NY,NX)),
               ns=np.zeros((NY,NX)), vx=np.zeros((NY,NX)), vx2=np.zeros((NY,NX)),
               vy=np.zeros((NY,NX)), vy2=np.zeros((NY,NX)), nfr=0, angle=angle)
    if not win:
        return _nan(angle, lam, seed, 'locked' if timed_out else 'empty', sim_t), acc
    door_peak = 0.0; cas = []; dcontacts = []; dcrush = 0; dn = 0; ns_in = []
    for t, ags in win:
        ns_in.append(len(ags))
        H,_,_ = np.histogram2d([a[1] for a in ags],[a[2] for a in ags],
                               bins=[NX,NY], range=[[0,DOMX],[0,DOMY]])
        acc['count'] += H.T
        fr = _smooth(H.T/(RES*RES), walk)
        door_peak = max(door_peak, float(np.nanmax(fr[door])) if door.any() else 0.0)
        cas.append(sum(1 for a in ags if a[0] == 1))
        for a in ags:
            inj,x,y,d,vx,vy,tp,contact = a
            if not (0<=x<=DOMX and 0<=y<=DOMY): continue
            jx=min(NX-1,int(x/RES)); jy=min(NY-1,int(y/RES))
            acc['count']; acc['ns'][jy,jx]+=1; acc['csum'][jy,jx]+=contact
            acc['vx'][jy,jx]+=vx; acc['vx2'][jy,jx]+=vx*vx
            acc['vy'][jy,jx]+=vy; acc['vy2'][jy,jx]+=vy*vy
            if inj==1: acc['inj'][jy,jx]+=1
            if door[jy,jx]:
                dcontacts.append(contact); dn+=1
                if contact > F_CRUSH*math.pi*d: dcrush+=1
        acc['nfr'] += 1
    rho_s = _smooth(acc['count']/(acc['nfr']*RES*RES), walk)
    door_steady = float(np.nanmax(rho_s[door])) if door.any() else 0.0
    corner_rho = float(np.nanmax(rho_s[corner])) if corner.any() else float('nan')
    n_end = len(win[-1][1])
    succ = stats.get('success', 0)
    # stationarity over the window: 1st-half vs 2nd-half mean occupancy within 15%
    if len(ns_in) >= 4:
        h = len(ns_in)//2; f1 = float(np.mean(ns_in[:h])); f2 = float(np.mean(ns_in[h:]))
        steady = int(abs(f2-f1)/max(f1,1e-6) < 0.15)
    else:
        steady = 0
    row = dict(angle=angle, lam=lam, seed=seed, status='ok', timed_out=timed_out,
               sim_t=round(sim_t,2), mean_n=round(float(np.mean(ns_in)),1), steady=steady,
               door_rho_steady=round(door_steady,3), door_rho_peak=round(door_peak,3),
               corner_rho=round(corner_rho,3) if np.isfinite(corner_rho) else '',
               door_contact_p95=round(float(np.percentile(dcontacts,95)),1) if dcontacts else 0.0,
               door_contact_max=round(float(np.max(dcontacts)),1) if dcontacts else 0.0,
               crush_frac=round(dcrush/max(dn,1),4),
               casualties=round(float(np.mean(cas)),2),
               door_outflow=round(max(0,succ-n_end)/T_TOTAL,3),
               inflow_rate=round(succ/T_TOTAL,3), n_end=n_end)
    return row, acc


def _nan(angle, lam, seed, status, sim_t):
    r = {k:'' for k in ROW}
    r.update(angle=angle, lam=lam, seed=seed, status=status, timed_out=0,
             sim_t=round(sim_t,2), mean_n=0)
    return r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--seeds', type=int, default=12)
    ap.add_argument('--jobs', type=int, default=os.cpu_count())
    ap.add_argument('--lambdas', default=None, help='comma list, e.g. 4,8')
    ap.add_argument('--angles', default=None, help='comma list, e.g. 0,45,60,90,120,135')
    ap.add_argument('--reduced', action='store_true', help='smoke: 0/90/135 x λ8 x 3 seeds')
    ap.add_argument('--t-total', type=float, default=T_TOTAL, help='sim duration (s)')
    ap.add_argument('--t-min', type=float, default=T_MIN, help='start of steady measurement window (s)')
    ap.add_argument('--cell-timeout', type=float, default=CELL_TIMEOUT, help='per-run wall-clock cap (s)')
    ap.add_argument('--plot-only', default=None)
    ap.add_argument('--exe', default=EXE, help='simulation binary (default: bottleneck; pass the openbend binary for the open comparison)')
    ap.add_argument('--field-dir', default=FIELD_DIR, help='floor-field directory (door fields by default; open-cap fields for the open run)')
    ap.add_argument('--tag', default='phaseB', help='output sub-folder prefix (e.g. openbendB for the open run)')
    ap.add_argument('--open', action='store_true', help='label/metadata only: this is the open-outlet comparison run')
    args = ap.parse_args()
    if args.plot_only:
        P.regenerate(args.plot_only); return
    args.exe = os.path.abspath(args.exe)
    globals()['EXE'] = args.exe
    globals()['FIELD_DIR'] = os.path.abspath(args.field_dir) if not os.path.isabs(args.field_dir) else args.field_dir

    angles = ANGLES; lambdas = LAMBDAS; seeds = list(range(args.seeds))
    if args.reduced:
        angles = [0, 90, 135]; lambdas = [8.0]; seeds = [0, 1, 2]
    if args.angles:  angles = [int(float(x)) for x in args.angles.split(',')]
    if args.lambdas: lambdas = [float(x) for x in args.lambdas.split(',')]

    # field pre-flight
    missing = [a for a in angles if not os.path.exists(os.path.join(FIELD_DIR, f'ff_field_a{a}.csv'))]
    if missing:
        raise SystemExit(f'missing door fields for angles {missing} — run:\n'
                         f'  python -u bend_bottleneck/make_field_bottleneck.py '
                         f'--angles {" ".join(map(str,angles))} --width {WIDTH} --outdir {os.path.relpath(FIELD_DIR,ROOT)}')

    tasks = [(a, lm, s) for a in angles for lm in lambdas for s in seeds]
    tasks.sort(key=lambda t: (t[1], t[2], t[0]))
    stamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    outdir = os.path.join(OUT_ROOT, f'{args.tag}_{stamp}'); os.makedirs(outdir, exist_ok=True)
    raw = os.path.join(outdir, 'raw_results.csv')
    print(f'BOTTLENECK STUDY  angles={angles} λ={lambdas} seeds={len(seeds)} '
          f'-> {len(tasks)} runs  jobs={args.jobs}\n  out={outdir}', flush=True)

    rows = []; cond_acc = {}
    fh = open(raw, 'w', newline=''); w = csv.DictWriter(fh, fieldnames=ROW); w.writeheader(); fh.flush()
    t_all = time.time(); n = 0
    with mp.Pool(args.jobs, initializer=_winit,
                 initargs=(FIELD_DIR, args.t_total, args.t_min, args.cell_timeout, args.exe)) as pool:
        for row, acc in pool.imap_unordered(run_one, tasks):
            w.writerow(row); fh.flush(); rows.append(row); n += 1
            if acc is not None and acc['nfr'] > 0:
                key = (acc['angle'], row['lam'])
                C = cond_acc.setdefault(key, {k:np.zeros((NY,NX)) for k in
                                              ('count','csum','inj','ns','vx','vx2','vy','vy2')})
                C['nfr'] = C.get('nfr',0)+acc['nfr']
                for kk in ('count','csum','inj','ns','vx','vx2','vy','vy2'): C[kk]+=acc[kk]
            print(f'  [{n}/{len(tasks)}] a{row["angle"]} λ{row["lam"]} s{row["seed"]} '
                  f'{row["status"]} door_steady={row.get("door_rho_steady")} '
                  f'door_peak={row.get("door_rho_peak")} corner={row.get("corner_rho")} '
                  f'cas={row.get("casualties")} ({row.get("sim_t")}s)', flush=True)
    fh.close()
    _aggregate(rows, outdir)
    _dump_grids(cond_acc, outdir)
    _runinfo(outdir, angles, lambdas, seeds, t_all)
    print('\nGenerating figures...', flush=True)
    P.regenerate(outdir)
    print(f'DONE -> {outdir}', flush=True)


def _fnum(r, k):
    try: return float(r[k])
    except (ValueError, TypeError, KeyError): return float('nan')


def _aggregate(rows, outdir):
    from collections import defaultdict
    grp = defaultdict(list)
    for r in rows:
        if r['status'] == 'ok': grp[(r['angle'], r['lam'])].append(r)
    metrics = ['door_rho_steady','door_rho_peak','corner_rho','door_contact_p95',
               'crush_frac','casualties','door_outflow','inflow_rate','mean_n']
    out = []
    for (a, lm), rs in sorted(grp.items()):
        d = dict(angle=a, lam=lm, n_seeds=len(rs))
        for m in metrics:
            v = np.array([_fnum(r, m) for r in rs]); v = v[np.isfinite(v)]
            d[f'{m}_mean'] = round(float(v.mean()),4) if v.size else ''
            d[f'{m}_ci95'] = round(float(1.96*v.std(ddof=1)/math.sqrt(len(v))),4) if v.size > 1 else 0.0
        out.append(d)
    if out:
        with open(os.path.join(outdir,'aggregated.csv'),'w',newline='') as f:
            wa = csv.DictWriter(f, fieldnames=list(out[0].keys())); wa.writeheader()
            for d in out: wa.writerow(d)


def _dump_grids(cond_acc, outdir):
    gd = os.path.join(outdir,'grids'); os.makedirs(gd, exist_ok=True)
    for (angle, lam), C in cond_acc.items():
        g, walk, door, corner = _masks(angle)
        nfr = max(C['nfr'],1)
        rho = _smooth(C['count']/(nfr*RES*RES), walk)
        contact = _smooth(np.where(C['ns']>0, C['csum']/np.maximum(C['ns'],1), 0.0), walk)
        cas = _smooth(C['inj']/(nfr*RES*RES), walk)
        with np.errstate(invalid='ignore'):
            vxm=np.where(C['ns']>0,C['vx']/np.maximum(C['ns'],1),0.0)
            vym=np.where(C['ns']>0,C['vy']/np.maximum(C['ns'],1),0.0)
            var=np.maximum(np.where(C['ns']>1,C['vx2']/np.maximum(C['ns'],1)-vxm**2,0)+
                           np.where(C['ns']>1,C['vy2']/np.maximum(C['ns'],1)-vym**2,0),0)
        pres = _smooth(C['count']/(nfr*RES*RES)*var, walk)
        with open(os.path.join(gd, f'a{angle}_lam{lam}.csv'),'w',newline='') as f:
            wr=csv.writer(f); wr.writerow(['x','y','rho','contact','casualty','pressure'])
            for iy in range(NY):
                for ix in range(NX):
                    if walk[iy,ix]:
                        wr.writerow([f'{XC[ix]:.2f}',f'{YC[iy]:.2f}',f'{rho[iy,ix]:.4f}',
                                     f'{contact[iy,ix]:.4f}',f'{cas[iy,ix]:.5f}',f'{pres[iy,ix]:.5f}'])


def _runinfo(outdir, angles, lambdas, seeds, t_all):
    with open(os.path.join(outdir,'run_info.txt'),'w') as f:
        f.write(f'BEND+BOTTLENECK study  {datetime.datetime.now().isoformat(timespec="seconds")}\n')
        f.write(f'python={platform.python_version()} platform={platform.platform()}\n')
        f.write(f'width={WIDTH} V0={V0} T_TOTAL={T_TOTAL} T_MIN={T_MIN} RES={RES} '
                f'door=1.2m HB=18(long inlet) domain={DOMX}x{DOMY} cell_timeout={CELL_TIMEOUT}\n')
        f.write(f'angles={angles}\nlambdas={lambdas}\nseeds={seeds}\nexe={EXE}\nfield_dir={FIELD_DIR}\n')
        f.write('model: InjurySwitch=5 soft-radius 0.6, StallKickEnable=0, inflow-driven crush.\n')
        f.write(f'wall_time={time.time()-t_all:.0f}s\n')


if __name__ == '__main__':
    main()
