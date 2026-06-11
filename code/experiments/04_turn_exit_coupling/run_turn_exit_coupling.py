"""OPTION-D: bend-to-exit CAPACITY-AND-OVERLOAD study.

Central question
---------------
Is the bend-to-exit corridor a CAPACITY-LIMITED SYSTEM IN SERIES, where the
realised flow is the smaller of the bend capacity Q_bend(theta) and the exit
capacity Q_door(d), and crush risk is governed by the OVERLOAD ratio
Phi = Q_bend(theta) / Q_door(d) (how far the bend's delivered demand exceeds
what the door can pass)?

Design
------
Sweep turn angle theta x DOOR WIDTH d at a saturating inflow. Then:
  * theta = 0 (straight) column     -> Q_door(d)            (exit capacity)
  * open-bend study (separate)      -> Q_bend(theta)        (bend capacity)
  * measured throughput inflow_rate -> Q_sys(theta,d)       (test vs min(.,.))
  * door density / casualties vs Phi -> risk law / collapse
  * crossover Q_door(d*)=Q_bend(theta) -> critical width d*(theta) phase boundary

This reuses the EXACT metric/mask/aggregation code of run_bottleneck_study.py so
the d=1.2 m column reproduces the phaseB bottleneck baseline. The ONLY physics
change is the parameterised door width, carried by:
  * the C param BendDoorWidth (binary sd_crunch_capacity, par sd_capacity.par), and
  * a per-door floor-field folder results/bend_capacity_fields/d{int(d*10)}/.
Both are asserted to agree (geometry.json egress == BendDoorWidth) before each run.

Nothing in the original code/binaries/fields/results is touched.

Usage:
  smoke : python -u bend_bottleneck/run_capacity_overload.py --reduced --jobs 6
  full  : python -u bend_bottleneck/run_capacity_overload.py \
            --angles 0,45,90,135 --doors 0.8,1.2,1.6,2.0,2.5,3.0,3.5,4.0 \
            --lambdas 12 --seeds 15 --jobs 14
"""
import os, sys, csv, time, math, json, argparse, datetime, platform
import multiprocessing as mp
import numpy as np
from scipy.ndimage import gaussian_filter
from matplotlib.path import Path as MPath

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, os.path.join(ROOT, 'model')); sys.path.insert(0, os.path.join(ROOT, 'plotting'))
from sfm.backends import c_exe
import make_field_bottleneck as F          # geometry() for the measurement masks

EXE       = os.path.join(ROOT, 'model', 'c_core', 'bend', 'sd_crunch_capacity_ch')  # general-angle chamfer binary
CAP_PAR   = os.path.join(ROOT, 'model', 'c_core', 'bend', 'sd_capacity.par')   # has BendDoorWidth token
FIELD_ROOT = os.path.join(ROOT, '..', 'results', 'bend_capacity_fields')     # d{tag}/ holds chamfered ff_field_a{ang}_r{tag}.csv
OUT_ROOT  = os.path.join(ROOT, '..', 'results', 'bend_capacity')
CORNER_R  = 0.25                             # inner-corner fillet radius (chamfered Option D)
RTAG      = int(CORNER_R*10 + 0.5)           # matches C ff_load_file int(BendCornerR*10+0.5)

WIDTH = 5.0; V0 = 1.3; RES = 0.25; DOMX, DOMY = 18.0, 32.0   # identical to phaseB
NX, NY = int(DOMX/RES), int(DOMY/RES)
XC = (np.arange(NX)+0.5)*RES; YC = (np.arange(NY)+0.5)*RES
CORNER_HALF = 1.6; DOOR_BACK = 2.0           # door ROI: 2 m of approach in front of the cap (full width)
F_CRUSH = 200.0                              # crush when contact > F_CRUSH*pi*D
INFLOW_CAP = 1000
T_TOTAL = 400.0; T_MIN = 200.0               # steady window [200,400] s (as phaseB)
CELL_TIMEOUT = 600                           # wall cap (s); narrow door + lam12 holds more agents -> headroom

ANGLES = [0, 45, 60, 90, 120, 135]          # 6 angles — consistent with open-bend/phaseB studies
DOORS  = [0.8, 1.2, 1.6, 2.0, 2.5, 3.0, 3.5, 4.0]
LAMBDAS = [12.0]                             # saturating: offered load to the door = Q_bend(theta)

ROW = ['angle','door_width','lam','seed','status','timed_out','sim_t','mean_n','steady',
       'door_rho_steady','door_rho_peak','corner_rho','door_contact_p95',
       'door_contact_max','crush_frac','casualties','door_outflow','inflow_rate','n_end']


def _dtag(d):                       # MUST match make_field_capacity.dtag
    return int(round(d * 10.0))


def _door_dir(door):
    return os.path.join(FIELD_ROOT, f'd{_dtag(door)}')


def _check_field(angle, door):
    """Assert the per-door field folder exists, its egress matches `door`, and the
    angle field is present. Returns (field_dir, ff_path). Raises ValueError on any
    mismatch -> field/wall disagreement can never silently happen."""
    fdir = _door_dir(door)
    gj = os.path.join(fdir, 'geometry.json')
    if not os.path.isdir(fdir) or not os.path.exists(gj):
        raise ValueError(f'missing field folder/geometry.json for door {door}: {fdir}')
    egress = json.load(open(gj)).get('egress', None)
    if egress is None or abs(float(egress) - float(door)) > 0.05:
        raise ValueError(f'egress mismatch in {gj}: geometry.json egress={egress} != door {door}')
    ai = int(round(angle))
    ff = os.path.join(fdir, f'ff_field_a{ai}_r{RTAG}.csv')   # chamfered field
    if not os.path.exists(ff):
        raise ValueError(f'missing field {ff} for angle {angle} door {door}')
    return fdir, ff


def _winit(field_root, t_total, t_min, cell_timeout, exe, cap_par):
    globals()['FIELD_ROOT'] = field_root
    globals()['T_TOTAL'] = t_total
    globals()['T_MIN'] = t_min
    globals()['CELL_TIMEOUT'] = cell_timeout
    globals()['EXE'] = exe
    c_exe.PAR_FILE = cap_par              # use OUR par (with BendDoorWidth) — spawn re-imports c_exe
    os.environ.setdefault('SFM_BEND_FIELD_DIR', field_root)   # overridden per task in run_one


def _masks(angle):
    """Measurement masks — IDENTICAL to run_bottleneck_study._masks. The door ROI is
    the full-width approach band (door-width-independent) so the standing jam is
    measured comparably across door widths; the corner ROI is fixed at the sharp
    vertex Vin."""
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
    angle, door, lam, seed = task
    # --- per-task field selection + sanity check (no silent field/wall mismatch) ---
    try:
        field_dir, ff = _check_field(angle, door)
    except ValueError as ex:
        r = _nan(angle, door, lam, seed, 'no_field', 0.0); print(f'  !! {ex}', flush=True)
        return r, None
    os.environ['SFM_BEND_FIELD_DIR'] = field_dir
    print(f'  START a{angle} d{door} lam{lam} s{seed}  BendDoorWidth={door}  '
          f'field={os.path.relpath(ff, ROOT)}  exe={os.path.basename(EXE)}', flush=True)

    g, walk, door_m, corner = _masks(angle)
    ov = dict(InjurySwitch=5, CasualtyRadiusScale=0.6, CasualtyForceScale=1.0,
              CasualtyContactScale=1.0, CasualtyFrictionScale=1.0, StallKickEnable=0.0,
              UseFloorField=1.0, BendAngle=float(angle), BendWidth=WIDTH, BendCornerR=CORNER_R,
              BendDoorWidth=float(door), InflowRate=float(lam), InitialN=0.0)
    case = {'label':'cap','geometry':'fis_room','scenario_id':6,
            'params':dict(N0=INFLOW_CAP, MaxSimTime=T_TOTAL, V_ChangeLimit=0.05, DoorWidth=1.0)}
    stats = {}; t0 = time.time()
    try:
        _t,_f,frames,_e = c_exe.run_with_trajectory(case, V0, seed, traj_dt=0.5,
            max_sim_time=T_TOTAL, overrides=ov, exe=EXE, timeout_s=CELL_TIMEOUT,
            full_traj=True, stats=stats)
    except Exception:
        return _nan(angle, door, lam, seed, 'error', time.time()-t0), None
    sim_t = time.time()-t0; timed_out = int(sim_t >= CELL_TIMEOUT-1)
    win = [(t,a) for t,a in frames if t >= T_MIN and a]
    acc = dict(count=np.zeros((NY,NX)), csum=np.zeros((NY,NX)), inj=np.zeros((NY,NX)),
               ns=np.zeros((NY,NX)), vx=np.zeros((NY,NX)), vx2=np.zeros((NY,NX)),
               vy=np.zeros((NY,NX)), vy2=np.zeros((NY,NX)), nfr=0, angle=angle, door=door)
    if not win:
        return _nan(angle, door, lam, seed, 'locked' if timed_out else 'empty', sim_t), acc
    door_peak = 0.0; cas = []; dcontacts = []; dcrush = 0; dn = 0; ns_in = []
    for t, ags in win:
        ns_in.append(len(ags))
        H,_,_ = np.histogram2d([a[1] for a in ags],[a[2] for a in ags],
                               bins=[NX,NY], range=[[0,DOMX],[0,DOMY]])
        acc['count'] += H.T
        fr = _smooth(H.T/(RES*RES), walk)
        door_peak = max(door_peak, float(np.nanmax(fr[door_m])) if door_m.any() else 0.0)
        cas.append(sum(1 for a in ags if a[0] == 1))
        for a in ags:
            inj,x,y,d,vx,vy,tp,contact = a
            if not (0<=x<=DOMX and 0<=y<=DOMY): continue
            jx=min(NX-1,int(x/RES)); jy=min(NY-1,int(y/RES))
            acc['ns'][jy,jx]+=1; acc['csum'][jy,jx]+=contact
            acc['vx'][jy,jx]+=vx; acc['vx2'][jy,jx]+=vx*vx
            acc['vy'][jy,jx]+=vy; acc['vy2'][jy,jx]+=vy*vy
            if inj==1: acc['inj'][jy,jx]+=1
            if door_m[jy,jx]:
                dcontacts.append(contact); dn+=1
                if contact > F_CRUSH*math.pi*d: dcrush+=1
        acc['nfr'] += 1
    rho_s = _smooth(acc['count']/(acc['nfr']*RES*RES), walk)
    door_steady = float(np.nanmax(rho_s[door_m])) if door_m.any() else 0.0
    corner_rho = float(np.nanmax(rho_s[corner])) if corner.any() else float('nan')
    n_end = len(win[-1][1])
    succ = stats.get('success', 0)
    if len(ns_in) >= 4:
        h = len(ns_in)//2; f1 = float(np.mean(ns_in[:h])); f2 = float(np.mean(ns_in[h:]))
        steady = int(abs(f2-f1)/max(f1,1e-6) < 0.15)
    else:
        steady = 0
    row = dict(angle=angle, door_width=door, lam=lam, seed=seed, status='ok', timed_out=timed_out,
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


def _nan(angle, door, lam, seed, status, sim_t):
    r = {k:'' for k in ROW}
    r.update(angle=angle, door_width=door, lam=lam, seed=seed, status=status, timed_out=0,
             sim_t=round(sim_t,2), mean_n=0)
    return r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--seeds', type=int, default=15)
    ap.add_argument('--jobs', type=int, default=os.cpu_count())
    ap.add_argument('--angles', default=None, help='comma list, e.g. 0,45,90,135')
    ap.add_argument('--doors', default=None, help='comma list, e.g. 0.8,1.2,1.6,2.0,2.5,3.0,3.5,4.0')
    ap.add_argument('--lambdas', default=None, help='comma list, e.g. 12')
    ap.add_argument('--reduced', action='store_true', help='smoke: a{0,90} x d{0.8,2.5} x 1 seed')
    ap.add_argument('--t-total', type=float, default=T_TOTAL)
    ap.add_argument('--t-min', type=float, default=T_MIN)
    ap.add_argument('--cell-timeout', type=float, default=CELL_TIMEOUT)
    ap.add_argument('--field-root', default=FIELD_ROOT)
    ap.add_argument('--exe', default=EXE)
    ap.add_argument('--par', default=CAP_PAR)
    ap.add_argument('--tag', default='capovCH')
    args = ap.parse_args()
    globals()['EXE'] = os.path.abspath(args.exe)
    globals()['CAP_PAR'] = os.path.abspath(args.par)
    globals()['FIELD_ROOT'] = os.path.abspath(args.field_root)

    angles = ANGLES; doors = DOORS; lambdas = LAMBDAS; seeds = list(range(args.seeds))
    if args.reduced:
        angles = [0, 90]; doors = [0.8, 2.5]; lambdas = [12.0]; seeds = [0]
    if args.angles:  angles = [int(float(x)) for x in args.angles.split(',')]
    if args.doors:   doors  = [float(x) for x in args.doors.split(',')]
    if args.lambdas: lambdas = [float(x) for x in args.lambdas.split(',')]

    # ---- pre-flight: every (angle,door) field must exist + egress must match ----
    problems = []
    for a in angles:
        for d in doors:
            try: _check_field(a, d)
            except ValueError as ex: problems.append(str(ex))
    if problems:
        msg = '\n  '.join(problems[:12])
        raise SystemExit(
            f'FIELD PRE-FLIGHT FAILED ({len(problems)} issue(s)):\n  {msg}\n'
            f'Generate the fields first:\n'
            f'  python -u bend_bottleneck/make_field_capacity.py --angles {" ".join(map(str,angles))} '
            f'--doors {" ".join(map(str,doors))} --width {WIDTH} '
            f'--outroot {os.path.relpath(FIELD_ROOT, ROOT)} --preview')

    tasks = [(a, d, lm, s) for a in angles for d in doors for lm in lambdas for s in seeds]
    tasks.sort(key=lambda t: (t[2], t[1], t[0], t[3]))      # lam, door, angle, seed
    stamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    outdir = os.path.join(OUT_ROOT, f'{args.tag}_{stamp}'); os.makedirs(outdir, exist_ok=True)
    raw = os.path.join(outdir, 'raw_results.csv')
    print(f'CAPACITY-OVERLOAD STUDY  angles={angles} doors={doors} lam={lambdas} '
          f'seeds={len(seeds)} -> {len(tasks)} runs  jobs={args.jobs}\n'
          f'  exe={EXE}\n  par={CAP_PAR}\n  field_root={FIELD_ROOT}\n  out={outdir}', flush=True)

    rows = []; cond_acc = {}
    fh = open(raw, 'w', newline=''); w = csv.DictWriter(fh, fieldnames=ROW); w.writeheader(); fh.flush()
    t_all = time.time(); n = 0
    with mp.Pool(args.jobs, initializer=_winit,
                 initargs=(FIELD_ROOT, args.t_total, args.t_min, args.cell_timeout, EXE, CAP_PAR)) as pool:
        for row, acc in pool.imap_unordered(run_one, tasks):
            w.writerow(row); fh.flush(); rows.append(row); n += 1
            if acc is not None and acc['nfr'] > 0:
                key = (acc['angle'], acc['door'], row['lam'])
                C = cond_acc.setdefault(key, {k:np.zeros((NY,NX)) for k in
                                              ('count','csum','inj','ns','vx','vx2','vy','vy2')})
                C['nfr'] = C.get('nfr',0)+acc['nfr']
                for kk in ('count','csum','inj','ns','vx','vx2','vy','vy2'): C[kk]+=acc[kk]
            print(f'  [{n}/{len(tasks)}] a{row["angle"]} d{row["door_width"]} lam{row["lam"]} s{row["seed"]} '
                  f'{row["status"]} Q=inflow{row.get("inflow_rate")} dout={row.get("door_outflow")} '
                  f'door_steady={row.get("door_rho_steady")} corner={row.get("corner_rho")} '
                  f'cas={row.get("casualties")} ({row.get("sim_t")}s)', flush=True)
    fh.close()
    _aggregate(rows, outdir)
    _dump_grids(cond_acc, outdir)
    _runinfo(outdir, angles, doors, lambdas, seeds, t_all)
    print(f'\nDONE -> {outdir}\nNow make the figures (separate, you run):\n'
          f'  python -u bend_bottleneck/analyze_capacity.py --run {os.path.relpath(outdir,ROOT)} '
          f'--outdir {os.path.relpath(outdir,ROOT)}/figs', flush=True)


def _fnum(r, k):
    try: return float(r[k])
    except (ValueError, TypeError, KeyError): return float('nan')


def _aggregate(rows, outdir):
    from collections import defaultdict
    grp = defaultdict(list)
    for r in rows:
        if r['status'] == 'ok': grp[(r['angle'], r['door_width'], r['lam'])].append(r)
    metrics = ['inflow_rate','door_outflow','door_rho_steady','door_rho_peak','corner_rho',
               'door_contact_p95','door_contact_max','crush_frac','casualties','mean_n']
    out = []
    for (a, d, lm), rs in sorted(grp.items()):
        rec = dict(angle=a, door_width=d, lam=lm, n_seeds=len(rs))
        for m in metrics:
            v = np.array([_fnum(r, m) for r in rs]); v = v[np.isfinite(v)]
            rec[f'{m}_mean'] = round(float(v.mean()),4) if v.size else ''
            rec[f'{m}_ci95'] = round(float(1.96*v.std(ddof=1)/math.sqrt(len(v))),4) if v.size > 1 else 0.0
        out.append(rec)
    if out:
        with open(os.path.join(outdir,'aggregated.csv'),'w',newline='') as f:
            wa = csv.DictWriter(f, fieldnames=list(out[0].keys())); wa.writeheader()
            for rec in out: wa.writerow(rec)


def _dump_grids(cond_acc, outdir):
    gd = os.path.join(outdir,'grids'); os.makedirs(gd, exist_ok=True)
    for (angle, door, lam), C in cond_acc.items():
        g, walk, door_m, corner = _masks(angle)
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
        with open(os.path.join(gd, f'a{angle}_d{_dtag(door)}_lam{lam}.csv'),'w',newline='') as f:
            wr=csv.writer(f); wr.writerow(['x','y','rho','contact','casualty','pressure'])
            for iy in range(NY):
                for ix in range(NX):
                    if walk[iy,ix]:
                        wr.writerow([f'{XC[ix]:.2f}',f'{YC[iy]:.2f}',f'{rho[iy,ix]:.4f}',
                                     f'{contact[iy,ix]:.4f}',f'{cas[iy,ix]:.5f}',f'{pres[iy,ix]:.5f}'])


def _runinfo(outdir, angles, doors, lambdas, seeds, t_all):
    with open(os.path.join(outdir,'run_info.txt'),'w') as f:
        f.write(f'OPTION-D capacity-overload study  {datetime.datetime.now().isoformat(timespec="seconds")}\n')
        f.write(f'python={platform.python_version()} platform={platform.platform()}\n')
        f.write(f'width={WIDTH} V0={V0} T_TOTAL={T_TOTAL} T_MIN={T_MIN} RES={RES} '
                f'HB=18 domain={DOMX}x{DOMY} cell_timeout={CELL_TIMEOUT} F_CRUSH={F_CRUSH}\n')
        f.write(f'angles={angles}\ndoors={doors}\nlambdas={lambdas}\nseeds={seeds}\n')
        f.write(f'exe={EXE}\npar={CAP_PAR}\nfield_root={FIELD_ROOT}\n')
        f.write('model: ScenarioID=6, sharp corner (BendCornerR=0), InjurySwitch=5 soft-radius 0.6, '
                'StallKickEnable=0, controlled inflow, parameterised BendDoorWidth.\n')
        f.write(f'wall_time={time.time()-t_all:.0f}s\n')


if __name__ == '__main__':
    main()
