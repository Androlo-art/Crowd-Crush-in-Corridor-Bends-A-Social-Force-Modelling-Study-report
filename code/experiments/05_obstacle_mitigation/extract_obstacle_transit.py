"""Per-agent TRANSIT-TIME extractor for the obstacle study (efficiency coda).

Uses the TRANSIT-instrumented binary sd_crunch_capacity_obs_t, which stamps each agent's
spawn time and, on every door exit, appends one line "(spawn_t exit_t)" to sd.transit_*.
The physics is bit-identical to sd_crunch_capacity_obs (the change is additive logging
only — verified by diff), so transit and the obstacle metrics come from the same dynamics.

Each line of sd.transit_* is ONE completed agent transit:
    transit = exit_t - spawn_t        (spawn at the inlet -> exit at the door)
There is no id-tracking or censoring ambiguity — only agents that actually reached the
door appear; injured (pinned) agents never do, by construction.

RIGOUR (steady state): the per-cell summary is the DEPARTURE-based steady-state sojourn
time — agents that EXIT within [T_MIN, T_TOTAL] (the same steady window used everywhere
else), excluding the fast fill-transient leavers — and reports mean/median AND the TAIL
(p90/p95/max), the safety-relevant, non-redundant part. The FULL per-agent table
(seed, spawn_t, exit_t, transit) is saved per (angle,config) so any other filter or
relationship (distribution & tail, transit-vs-spawn-time stationarity, transit-vs-angle,
ctrl-vs-obstacle) can be computed afterwards WITHOUT re-running.

Same geometry/physics/door(1.2)/inflow(12) as the sweeps; only the FLOOR FIELD differs:
  --mode physical (default): column-blind chamfered field (results/bend_capacity_fields/d12)  [matches obs_*]
  --mode nav                : pillar-aware per-config field (results/bend_obstacle_navfields/<label>) [matches obsnav_*]
TrajSaveDt is set to 0 (the transit log is independent of the trajectory, so we skip the
large traj file). Writes only under results/bend_obstacle_transit/.

Usage:
  smoke: python -u bend_bottleneck/extract_transit.py --angles 90 --cfgs ctrl gate --seeds 1 --jobs 2
  run  : python -u bend_bottleneck/extract_transit.py --angles 0 45 60 90 120 135 \
            --cfgs ctrl gate pillar_off_far pillar_axis --seeds 12 --jobs 12 --mode physical
"""
import os, sys, csv, glob, time, argparse, tempfile, shutil, datetime
import multiprocessing as mp
import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, os.path.join(ROOT, 'model')); sys.path.insert(0, os.path.join(ROOT, 'plotting'))
from sfm.backends import c_exe
import run_capacity_obstacle as R          # COLCFGS, cfg_centers (exact pillar geometry)

EXE        = os.path.join(ROOT, 'model', 'c_core', 'bend', 'sd_crunch_capacity_obs_t')  # TRANSIT-instrumented
CAP_PAR    = os.path.join(ROOT, 'model', 'c_core', 'bend', 'sd_capacity.par')
PHYS_FIELD = os.path.join(ROOT, '..', 'results', 'bend_capacity_fields')      # /d12  (column-blind chamfered)
NAV_FIELD  = os.path.join(ROOT, '..', 'results', 'bend_obstacle_navfields')   # /<label>  (pillar-aware)
OUT_ROOT   = os.path.join(ROOT, '..', 'results', 'bend_obstacle_transit')
WIDTH = 5.0; V0 = 1.3; CORNER_R = 0.25; DOOR = 1.2; RTAG = 3
T_TOTAL = 400.0; T_MIN = 200.0; CELL_TIMEOUT = 600; INFLOW_CAP = 1000


def _dtag(d): return int(round(d * 10))


def _field_dir(mode, label):
    return os.path.join(PHYS_FIELD, f'd{_dtag(DOOR)}') if mode == 'physical' else os.path.join(NAV_FIELD, label)


def parse_transit(path):
    """sd.transit_* -> list of (spawn_t, exit_t). Each line is one completed door exit."""
    out = []
    with open(path) as f:
        for line in f:
            if line.startswith('#'):
                continue
            p = line.split()
            if len(p) >= 2:
                out.append((float(p[0]), float(p[1])))
    return out


def run_cell(task):
    angle, ci, seed, mode = task
    cfg = R.COLCFGS[ci]; label = cfg['label']
    fdir = _field_dir(mode, label)
    if not os.path.exists(os.path.join(fdir, f'ff_field_a{int(angle)}_r{RTAG}.csv')):
        return label, angle, seed, None, f'missing field in {fdir}'
    os.environ['SFM_BEND_FIELD_DIR'] = fdir
    c_exe.PAR_FILE = CAP_PAR
    centers = R.cfg_centers(angle, cfg); sw = len(centers)
    ov = dict(InjurySwitch=5, CasualtyRadiusScale=0.6, CasualtyForceScale=1.0, CasualtyContactScale=1.0,
              CasualtyFrictionScale=1.0, StallKickEnable=0.0, UseFloorField=1.0, BendAngle=float(angle),
              BendWidth=WIDTH, BendCornerR=CORNER_R, BendDoorWidth=DOOR, InflowRate=12.0, InitialN=0.0,
              ColumnSwitch=int(sw))
    if sw >= 1:
        ov.update(ColumnCenterX=centers[0][0], ColumnCenterY=centers[0][1], ColumnD=centers[0][2])
    if sw == 2:
        ov.update(Column2CenterX=centers[1][0], Column2CenterY=centers[1][1], Column2D=centers[1][2])
    base = dict(N0=INFLOW_CAP, V0=V0, RndSeed=seed, ScenarioID=6, ColumnSwitch=int(sw), InjurySwitch=5,
                DoorWidth=1.0, DefaultDeltaT=0.01, V_ChangeLimit=0.05, C_NS=0.95,
                MaxSimTime=T_TOTAL, TrajSaveDt=0.0)        # transit log is independent of the traj
    base.update(ov)
    tmpl = c_exe.load_par_template(); tmp = tempfile.mkdtemp(suffix=f'_tr_a{int(angle)}_{label}_s{seed}')
    try:
        tp = os.path.join(tmp, 'sd.par')
        with open(tp, 'w') as fh:
            fh.writelines(c_exe._patch(tmpl, base))
        c_exe._run_exe(tp, tmp, EXE, timeout_s=CELL_TIMEOUT)
        tg = glob.glob(os.path.join(tmp, 'sd.transit_*'))
        if not tg:
            return label, angle, seed, [], 'no transit file (no exits?)'
        recs = parse_transit(tg[0])
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    return label, angle, seed, recs, 'ok'


def _stats(tr):
    tr = np.asarray(tr, float)
    return dict(n=len(tr), mean=round(float(tr.mean()), 2), median=round(float(np.median(tr)), 2),
                p90=round(float(np.percentile(tr, 90)), 2), p95=round(float(np.percentile(tr, 95)), 2),
                max=round(float(tr.max()), 2))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--angles', type=float, nargs='+', default=[0, 45, 60, 90, 120, 135])
    ap.add_argument('--cfgs', nargs='+', default=['ctrl', 'gate', 'pillar_off_far', 'pillar_axis'])
    ap.add_argument('--seeds', type=int, default=12)
    ap.add_argument('--mode', choices=['physical', 'nav'], default='physical')
    ap.add_argument('--jobs', type=int, default=os.cpu_count())
    ap.add_argument('--tag', default=None)
    args = ap.parse_args()
    idx = [i for i, c in enumerate(R.COLCFGS) if c['label'] in args.cfgs]
    if not idx:
        raise SystemExit(f'no matching configs in {args.cfgs}')
    stamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    outdir = os.path.join(OUT_ROOT, f'{args.tag or ("transit_"+args.mode)}_{stamp}')
    os.makedirs(os.path.join(outdir, 'records'), exist_ok=True)
    tasks = [(int(a), ci, s, args.mode) for a in args.angles for ci in idx for s in range(args.seeds)]
    print(f'TRANSIT  mode={args.mode}  exe={os.path.basename(EXE)}  '
          f'angles={[int(a) for a in args.angles]}  cfgs={[R.COLCFGS[i]["label"] for i in idx]}  '
          f'seeds={args.seeds}  -> {len(tasks)} cells\n  out={outdir}', flush=True)
    results = {}; t0 = time.time(); n = 0
    with mp.Pool(args.jobs) as pool:
        for label, angle, seed, recs, status in pool.imap_unordered(run_cell, tasks):
            n += 1
            if recs is None:
                print(f'  [{n}/{len(tasks)}] !! a{angle} {label} s{seed}: {status}', flush=True); continue
            results.setdefault((angle, label), []).extend([(seed, sp, ex, round(ex-sp, 3)) for (sp, ex) in recs])
            inw = sum(1 for (sp, ex) in recs if T_MIN <= ex <= T_TOTAL)
            print(f'  [{n}/{len(tasks)}] a{angle} {label} s{seed}: {len(recs)} exits ({inw} in window)', flush=True)
    summ = []
    for (angle, label), rows in sorted(results.items()):
        with open(os.path.join(outdir, 'records', f'a{angle}_{label}.csv'), 'w', newline='') as f:
            w = csv.writer(f); w.writerow(['seed', 'spawn_t', 'exit_t', 'transit'])
            for r in rows:
                w.writerow(r)
        win = [r[3] for r in rows if T_MIN <= r[2] <= T_TOTAL]      # exit-in-window steady departures
        rec = dict(angle=angle, col_label=label, mode=args.mode, n_exits_total=len(rows))
        if win:
            s = _stats(win)
            rec.update(n_window=s['n'], transit_mean=s['mean'], transit_median=s['median'],
                       transit_p90=s['p90'], transit_p95=s['p95'], transit_max=s['max'])
        summ.append(rec)
    if summ:
        keys = ['angle', 'col_label', 'mode', 'n_window', 'transit_mean', 'transit_median',
                'transit_p90', 'transit_p95', 'transit_max', 'n_exits_total']
        with open(os.path.join(outdir, 'transit_summary.csv'), 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=keys, extrasaction='ignore'); w.writeheader()
            for r in summ: w.writerow(r)
    print(f'\nDONE ({time.time()-t0:.0f}s) -> {outdir}\n  transit_summary.csv + records/a<ang>_<cfg>.csv '
          f'(full per-agent spawn/exit/transit table for any later analysis)', flush=True)


if __name__ == '__main__':
    main()
