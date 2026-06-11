"""OPTION-D analysis & thesis figures: capacity, risk surface, phase boundary.

What the full run (capov, 15 seeds) supports — and what it does NOT
------------------------------------------------------------------
ROBUST (steady-window quantities, reproduce phaseB at d=1.2):
  * casualties(theta,d) and door_density(theta,d)  -> the RISK surface
  * mean occupancy mean_n(theta,d)                  -> the SATURATION boundary
    (corridor packs <-> free-flows) = the capacity crossover d*(theta), which
    DECREASES with angle (sharper bend -> narrower critical exit).
  * the bend lowers crush at every door width (explains casualties-fall-with-angle).

USE WITH CARE (whole-run-average throughput):
  * door_outflow = exited/T_TOTAL underestimates the steady rate by the fill
    fraction (~10% for saturated cells) and is NOISY for wide/transition doors
    (mean_n ~ 180-350; e.g. the d=3.5 spike). It is reliable on the door-limited
    branch (narrow/medium doors) and where it cleanly PLATEAUS (90/135 deg), which
    cross-checks the independent open-bend capacity. We therefore show the capacity
    decomposition but anchor Q_bend on the OPEN study, not on the noisy wide doors.

NOT SUPPORTED (do not claim):
  * a clean universal Q_sys=min() fit at all angles (0/45 wide-door throughput noisy)
  * a universal risk-vs-Phi collapse (Q_door->0 at the d=0.8 clog makes Phi blow up)

Throughput metric = door_outflow (exit flow). Q_bend(theta) from --open (max over
lam of inflow_rate_mean = the published open-bend capacity).

Modes:
  --check   : print numbers (no figures).
  (default) : write thesis figures to --outdir (NEW folder; nothing overwritten).
"""
import os, sys, csv, argparse
import numpy as np

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

THROUGHPUT = 'door_outflow_mean'        # exit throughput
SAT_FREE_FRAC = 0.5                     # mean_n midpoint (packed<->free) for d*(theta)


def _load_agg(path):
    rows = []
    with open(path) as f:
        for r in csv.DictReader(f):
            d = {}
            for k, v in r.items():
                try: d[k] = float(v)
                except (ValueError, TypeError): d[k] = v
            rows.append(d)
    return rows


def _sat(rows):
    best = {}
    for r in rows:
        key = (r['angle'], r['door_width'])
        if key not in best or r['lam'] > best[key]['lam']:
            best[key] = r
    return best


def _Qbend_open(open_dir):
    """Open-bend capacity per angle = max over lam of inflow_rate_mean (published capacity)."""
    p = os.path.join(open_dir, 'aggregated.csv') if open_dir else None
    if not p or not os.path.exists(p):
        return None
    q = {}
    for r in _load_agg(p):
        v = r.get('inflow_rate_mean', '')
        if isinstance(v, float) and np.isfinite(v):
            q[r['angle']] = max(q.get(r['angle'], 0.0), v)
    return q


def derive(run_dir, open_dir=None):
    rows = _load_agg(os.path.join(run_dir, 'aggregated.csv'))
    sat = _sat(rows)
    angles = sorted({a for (a, d) in sat}); doors = sorted({d for (a, d) in sat})
    G = lambda a, d, k: (sat[(a, d)].get(k, np.nan) if (a, d) in sat else np.nan)

    Qsys   = {(a, d): G(a, d, THROUGHPUT) for a in angles for d in doors}
    meann  = {(a, d): G(a, d, 'mean_n_mean') for a in angles for d in doors}
    Qdoor  = {d: Qsys.get((0.0, d), Qsys.get((0, d), np.nan)) for d in doors}
    Qbend_open = _Qbend_open(open_dir)

    # saturation boundary d*(theta): door width where mean_n crosses the packed<->free midpoint
    dd = np.array(doors); dstar = {}; saturated = {}
    for a in angles:
        mn = np.array([meann[(a, d)] for d in doors])
        packed, free = np.nanmax(mn), np.nanmin(mn)
        thr = free + SAT_FREE_FRAC * (packed - free)
        saturated[a] = {d: (meann[(a, d)] >= thr) for d in doors}
        below = np.where(mn < thr)[0]
        if below.size and below[0] > 0:
            i = below[0]
            dstar[a] = float(np.interp(thr, [mn[i], mn[i-1]], [dd[i], dd[i-1]]))
        else:
            dstar[a] = float('nan')
    return dict(rows=rows, sat=sat, angles=angles, doors=doors, G=G,
                Qsys=Qsys, meann=meann, Qdoor=Qdoor, Qbend_open=Qbend_open,
                dstar=dstar, saturated=saturated)


def check(run_dir, open_dir=None):
    D = derive(run_dir, open_dir)
    print(f'== OPTION-D check :: {os.path.relpath(run_dir, ROOT)} ==')
    print(f'angles={D["angles"]}  doors={D["doors"]}  throughput={THROUGHPUT}')
    print('\nSaturation boundary d*(theta) [door stops binding; corridor stops packing]:')
    for a in D['angles']:
        qb = D['Qbend_open'].get(a) if D['Qbend_open'] else float('nan')
        print(f'   a={a:>5}  d*={D["dstar"][a]:.2f} m   Q_bend(open)={qb:.2f} ped/s')
    print('\n  angle door  Qexit  mean_n  sat?  door_rho   cas')
    for a in D['angles']:
        for d in D['doors']:
            r = D['sat'].get((a, d))
            if not r: continue
            print(f'  {a:>5} {d:>4} {D["Qsys"][(a,d)]:>6.2f} {D["meann"][(a,d)]:>6.0f}  '
                  f'{"Y" if D["saturated"][a][d] else "-":>3}  {r.get("door_rho_steady_mean","?"):>8}  '
                  f'{r.get("casualties_mean","?"):>5}')
    print('\nGUIDE: Qexit rises with d then plateaus at ~Q_bend(open); sat=Y => door-limited/crush-prone.')
    return D


# --------------------------------------------------------------------------- figures
def figures(run_dir, outdir, open_dir=None):
    import matplotlib.pyplot as plt
    import thesis_style as TS
    TS.apply()
    D = derive(run_dir, open_dir); os.makedirs(outdir, exist_ok=True)
    angles, doors = D['angles'], D['doors']; dd = np.array(doors)
    G = D['G']

    # ---- FIG 1: risk surface — casualties & door density vs door width, per angle ----
    fig, ax = plt.subplots(1, 2, figsize=(TS.FIGW, 0.5*TS.FIGW), constrained_layout=True)
    for a in angles:
        col = TS.angle_color(a)
        ax[0].plot(dd, [G(a, d, 'casualties_mean') for d in doors], '-o', color=col, label=f'{int(a)}°')
        ax[1].plot(dd, [G(a, d, 'door_rho_steady_mean') for d in doors], '-o', color=col)
    ax[1].axhline(TS.LETHAL_RHO, color=TS.CRIMSON, ls='--', lw=1.2)
    ax[0].set_ylabel('casualties (mean per frame)')
    ax[1].set_ylabel(r'door density (ped m$^{-2}$)')
    for a_ in ax: a_.set_xlabel('exit width $d$ (m)'); TS.xfmt(a_, 1)
    ax[0].set_title('(a) Casualty risk'); ax[1].set_title('(b) Door crush density')
    ax[0].legend(title='turn angle', ncol=2)
    TS.save(fig, os.path.join(outdir, 'fig_cap_risk'))

    # ---- FIG 2: capacity decomposition — exit flow vs door width + open Q_bend refs ---
    fig, ax = plt.subplots(figsize=(TS.FIGW, 0.62*TS.FIGW), constrained_layout=True)
    for a in angles:
        col = TS.angle_color(a)
        ys = np.array([D['Qsys'][(a, d)] for d in doors])
        satm = np.array([D['saturated'][a][d] for d in doors])
        ax.plot(dd, ys, '-', color=col, label=f'{int(a)}°', zorder=2)
        ax.plot(dd[satm], ys[satm], 'o', color=col, mfc=col, ms=5, zorder=3)        # filled = door-limited
        ax.plot(dd[~satm], ys[~satm], 'o', color=col, mfc='white', ms=5, zorder=3)  # open = bend-limited
        if D['Qbend_open'] and a in D['Qbend_open']:
            ax.axhline(D['Qbend_open'][a], color=col, ls=':', lw=1.0, alpha=0.8)
    ax.set_xlabel('exit width $d$ (m)')
    ax.set_ylabel(r'realised exit flow $Q_{\mathrm{sys}}$ (ped s$^{-1}$)')
    ax.set_title(r'Exit flow vs door width: sharp bends CAP throughput; '
                 r'gentle ones overshoot ($\cdots$ = open-bend $Q_{\mathrm{bend}}$)')
    ax.legend(title='turn angle', ncol=2); TS.yfmt(ax, 1); TS.xfmt(ax, 1)
    TS.save(fig, os.path.join(outdir, 'fig_cap_law'))

    # ---- FIG 3: phase diagram — theta-d plane, d*(theta) boundary, colour=casualties ---
    fig, ax = plt.subplots(figsize=(TS.FIGW, 0.62*TS.FIGW), constrained_layout=True)
    AA, DDg, CC = [], [], []
    for a in angles:
        for d in doors:
            AA.append(float(a)); DDg.append(d); CC.append(G(a, d, 'casualties_mean'))
    sc = ax.scatter(AA, DDg, c=np.nan_to_num(CC), cmap='turbo', s=140, edgecolor='k', lw=0.5)
    aa = np.array([float(a) for a in angles])
    ds = np.array([D['dstar'][a] for a in angles])
    ok = np.isfinite(ds)
    if ok.sum() >= 2:
        ax.plot(aa[ok], ds[ok], 'k-o', lw=2.2, label=r'crossover $d^*(\theta)$ (door $\to$ bend limited)')
        ax.fill_between(aa[ok], ds[ok], dd.max()+0.2, color='0.85', alpha=0.5, zorder=0)
        ax.text(aa[ok].mean(), min(ds[ok])+0.2, 'exit-limited\n(crush-prone)', ha='center', fontsize=9)
        ax.text(aa[ok].mean(), dd.max()-0.2, 'bend-limited (relieved)', ha='center', va='top', fontsize=9)
    ax.set_xlabel('turn angle $\\theta$ (deg)'); ax.set_ylabel('exit width $d$ (m)')
    ax.set_ylim(dd.min()-0.2, dd.max()+0.2)
    ax.set_title('Crush concentrates at STRAIGHT/gentle $\\times$ narrow exit ($d^*$ from occupancy)')
    cb = fig.colorbar(sc, ax=ax); cb.set_label('casualties (mean per frame)')
    if ok.sum() >= 2: ax.legend(loc='upper right', fontsize=9)
    TS.save(fig, os.path.join(outdir, 'fig_cap_phase'))

    # ---- FIG 4 (diagnostic): saturation curve that defines d* + throughput caveat -----
    fig, ax = plt.subplots(figsize=(TS.FIGW, 0.62*TS.FIGW), constrained_layout=True)
    for a in angles:
        col = TS.angle_color(a)
        ax.plot(dd, [D['meann'][(a, d)] for d in doors], '-o', color=col, label=f'{int(a)}°')
        if np.isfinite(D['dstar'][a]):
            ax.axvline(D['dstar'][a], color=col, ls=':', lw=1.0, alpha=0.7)
    ax.set_xlabel('exit width $d$ (m)'); ax.set_ylabel('mean occupancy $\\bar N$ (agents)')
    ax.set_title('Diagnostic: corridor packs (high $\\bar N$) until $d>d^*$ — defines the crossover')
    ax.legend(title='turn angle', ncol=2); TS.xfmt(ax, 1)
    TS.save(fig, os.path.join(outdir, 'fig_cap_diag_saturation'))
    print(f'wrote 4 figures (fig_cap_risk, fig_cap_law, fig_cap_phase, fig_cap_diag_saturation) -> {outdir}')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--run', required=True)
    ap.add_argument('--open', default=None, help='open-bend study dir (Q_bend reference)')
    ap.add_argument('--outdir', default=None, help='figures folder (NEW)')
    ap.add_argument('--check', action='store_true', help='numbers only, no figures')
    args = ap.parse_args()
    run = args.run if os.path.isabs(args.run) else os.path.join(ROOT, args.run)
    opn = (args.open if (args.open is None or os.path.isabs(args.open)) else os.path.join(ROOT, args.open))
    if args.check or not args.outdir:
        check(run, opn)
    else:
        out = args.outdir if os.path.isabs(args.outdir) else os.path.join(ROOT, args.outdir)
        figures(run, out, opn)


if __name__ == '__main__':
    main()
