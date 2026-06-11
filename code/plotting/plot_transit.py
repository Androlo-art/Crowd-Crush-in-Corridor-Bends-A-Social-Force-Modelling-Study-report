"""Transit-time + exit-throughput figures for the obstacle study.

Reads an extract_transit.py run (results/bend_obstacle_transit/<run>/records/), computes for
each (angle, config) the per-SEED steady-window mean transit and exit rate, and plots both
vs turn angle with 95% CI bands taken BETWEEN seeds (seed is the unit of replication, because
throughput/transit vary strongly seed-to-seed via stochastic clogging). Two panels:
  (a) mean transit time vs angle   -- expected ~flat / overlapping (no clean effect)
  (b) steady exit rate vs angle     -- the gate sits above control at the bends (Helbing eases outflow)

Steady window [200,400] s (departure-based: agents that EXIT in the window).

Usage:
  python -u bend_bottleneck/plot_transit.py --run results/bend_obstacle_transit/transit_physical_<stamp>
"""
import os, sys, csv, glob, re, argparse
import numpy as np
from collections import defaultdict

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT); sys.path.insert(0, ROOT)
import analyze_obstacle as A          # _palette / _disp (shared colour scheme + names)

T_MIN, T_TOT = 200.0, 400.0; WIN = T_TOT - T_MIN


def perseed(run, angle, cfg):
    """-> {seed: (mean_transit_s, exit_rate_per_s)} over in-window departures."""
    p = os.path.join(run, 'records', f'a{angle}_{cfg}.csv')
    if not os.path.exists(p):
        return {}
    bys = defaultdict(list)
    for r in csv.DictReader(open(p)):
        ex = float(r['exit_t'])
        if T_MIN <= ex <= T_TOT:
            bys[int(r['seed'])].append(float(r['transit']))
    return {s: (float(np.mean(v)), len(v) / WIN) for s, v in bys.items() if v}


def ci(x):
    x = np.asarray(x, float)
    if len(x) < 2:
        return (float(x.mean()) if len(x) else float('nan')), 0.0
    return float(x.mean()), float(1.96 * x.std(ddof=1) / np.sqrt(len(x)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--run', required=True)
    ap.add_argument('--cfgs', nargs='+', default=['ctrl', 'gate', 'pillar_off_far', 'pillar_axis'])
    ap.add_argument('--outdir', default=None)
    args = ap.parse_args()
    run = args.run if os.path.isabs(args.run) else os.path.join(ROOT, args.run)
    outdir = args.outdir or run
    angs = sorted({int(re.search(r'a(\d+)_', os.path.basename(f)).group(1))
                   for f in glob.glob(os.path.join(run, 'records', 'a*_*.csv'))})
    import matplotlib; matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    try:
        import thesis_style as TS; TS.apply(); figw = TS.FIGW
    except Exception:
        figw = 6.3
    col = A._palette(['ctrl'] + [c for c in args.cfgs if c != 'ctrl'])
    fig, axes = plt.subplots(1, 2, figsize=(figw, 0.52*figw), constrained_layout=True)
    panels = [(0, 'mean transit time (s)'), (1, r'steady exit rate (ag s$^{-1}$)')]
    print(f'angles={angs}  cfgs={args.cfgs}')
    for ax, (mi, ylab) in zip(axes, panels):
        for c in args.cfgs:
            xs, ys, es = [], [], []
            for a in angs:
                ps = perseed(run, a, c)
                if not ps:
                    continue
                m, e = ci([v[mi] for v in ps.values()])
                xs.append(a); ys.append(m); es.append(e)
            if not xs:
                continue
            xs = np.array(xs); ys = np.array(ys); es = np.array(es)
            cc, ls, lw, mk = col.get(c, ('gray', '-', 1.4, 'o'))
            lw = 1.9 if c == 'ctrl' else 1.3
            ax.fill_between(xs, ys-es, ys+es, color=cc, alpha=0.13, lw=0,
                            zorder=2 if c == 'ctrl' else 1)
            ax.plot(xs, ys, color=cc, lw=lw, marker=mk, ms=4, label=A._disp(c),
                    zorder=6 if c == 'ctrl' else 4)
        ax.set_xlabel('turn angle (degrees)'); ax.set_ylabel(ylab)
        ax.grid(alpha=0.3); ax.set_xticks(angs)
    axes[0].set_title('(a)', loc='left', fontsize=11, fontweight='bold')
    axes[1].set_title('(b)', loc='left', fontsize=11, fontweight='bold')
    axes[0].legend(fontsize=7.5, frameon=True, framealpha=0.9)
    fig.suptitle(r'Steady-state transit time and exit throughput (mean $\pm$ 95\% CI over seeds)',
                 fontsize=11)
    out = os.path.join(outdir, 'transit_vs_angle')
    for ext in ('pdf', 'png'):
        fig.savefig(f'{out}.{ext}', dpi=150, bbox_inches='tight')
    plt.close(fig)
    print('wrote', out + '.pdf/.png')


if __name__ == '__main__':
    main()
