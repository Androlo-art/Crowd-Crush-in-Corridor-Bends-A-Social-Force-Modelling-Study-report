"""
plot_fis_snapshots_for_report.py   (NEW, non-destructive helper for the v2 report)
==================================================================================
Mechanism snapshot of ONE faster-is-slower run (single 15x15 m room, one east-wall
door), for Chapter 3. Renders N chosen time points (2-6 panels) of the same
evacuation as true-size discs: ACTIVE (uninjured) agents in blue, INJURED agents
(casualties) in red, exit door in green.

The faster-is-slower sweep deleted its trajectories, so this RE-RUNS one (v0, seed)
with trajectory capture, using the SAME room/config/injury rule as the Chapter 3
'standard' (full-obstruction) sweep -- so the picture is a faithful, representative
member of that ensemble (state in the caption that it is one illustrative
trajectory-capture run; the quantitative validation comes from the full 20-seed
sweep). Choose the seed and time points with fis_snapshot_candidates.py first.

Does NOT modify any existing code/data. Re-runs the released binary (read-only) and
writes only the report figure.

Run from the repo root. NOTE: plain numbers for --times, no 't=' and no '<...>':
  .venv/bin/python -u bend_bottleneck/plot_fis_snapshots_for_report.py --v0 5.0 --seed 1 --times 0,51,204,280

Optional (defaults shown):
  --door 1.2 --injury 1 --tsim 400 --traj-dt 0.5 --door-R 2.0
  --labels "early;clogging;draining;late"   (optional phase words, one per panel)
  --blue-label active --red-label injured
  --t-early 4 --t-late <last>               (used only when --times is omitted: auto early/clog/late)
  --out <path WITHOUT extension>            (default: the report figures folder)
"""
import os, sys, argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.collections import EllipseCollection
from matplotlib.lines import Line2D

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, '..', 'model'))
from sfm.backends import c_exe
from sfm import cases, geometry
import thesis_style as TS

OUT_DEFAULT = os.path.join(ROOT, '..', '..', 'figures', '_regenerated', 'fis_mechanism_snapshots')


def draw_discs(ax, xs, ys, ds, color, ec='white', alpha=0.9, z=4):
    """Render agents as their true-diameter discs (data units)."""
    if not len(xs):
        return
    coll = EllipseCollection(widths=np.asarray(ds), heights=np.asarray(ds),
                             angles=np.zeros(len(xs)), units='xy',
                             offsets=np.c_[xs, ys], offset_transform=ax.transData,
                             facecolors=color, edgecolors=ec, linewidths=0.4,
                             alpha=alpha)
    coll.set_zorder(z)
    ax.add_collection(coll)


def split(ags):
    """Agent tuple is (injured, x, y, d, vx, vy, type[, contact])."""
    ok = [(a[1], a[2], a[3]) for a in ags if a[0] == 0]
    ij = [(a[1], a[2], a[3]) for a in ags if a[0] == 1]
    okx = [p[0] for p in ok]; oky = [p[1] for p in ok]; okd = [p[2] for p in ok]
    ijx = [p[0] for p in ij]; ijy = [p[1] for p in ij]; ijd = [p[2] for p in ij]
    return (okx, oky, okd), (ijx, ijy, ijd)


def door_region_count(ags, cx, cy, R):
    return sum(1 for a in ags if (a[1] - cx) ** 2 + (a[2] - cy) ** 2 <= R * R)


def parse_times(s):
    """Tolerant parse of --times: strips 't=', spaces and stray '<>' so e.g.
    '<t=0,t=51,t=280>' or '0, 51, 280' both work."""
    s = s.replace('<', '').replace('>', '').replace('t=', '').replace(' ', '')
    return [float(x) for x in s.split(',') if x != '']


def render_contact_sheet(frames, geo, n_frames, out_noext, blue_label, red_label, Lx, Ly):
    """Full-evacuation contact sheet: n_frames evenly spaced, each labelled t + injured.
    For the appendix (supplementary visual evidence of the whole sequence)."""
    ncol = 4
    nrow = int(np.ceil(n_frames / ncol))
    idxs = np.linspace(0, len(frames) - 1, n_frames).round().astype(int)
    fig, axs = plt.subplots(nrow, ncol, figsize=(1.9 * ncol, 1.9 * nrow + 0.5),
                            squeeze=False, layout='compressed')
    for k, ax in enumerate(axs.flat):
        if k >= len(idxs):
            ax.axis('off'); continue
        t, ags = frames[idxs[k]]
        geometry.draw(ax, geo)
        (okx, oky, okd), (ijx, ijy, ijd) = split(ags)
        draw_discs(ax, okx, oky, okd, 'tab:blue', z=4)
        draw_discs(ax, ijx, ijy, ijd, 'crimson', z=5)
        ax.set(xlim=(-0.5, Lx + 0.5), ylim=(-0.5, Ly + 0.5)); ax.set_aspect('equal')
        ax.set_title(f'$t={t:.0f}$ s,  injured $={len(ijx)}$', fontsize=9)
        ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle(f'Full evacuation sequence  ({blue_label} = blue, {red_label} = red)',
                 fontsize=11, fontweight='bold')
    for ext in ('pdf', 'png', 'svg'):
        fig.savefig(f'{out_noext}.{ext}', bbox_inches='tight')
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--v0', type=float, default=5.0)
    ap.add_argument('--seed', type=int, default=1)
    ap.add_argument('--door', type=float, default=1.2,
                    help='door width (m); 1.2 = the width adopted in Chapter 3')
    ap.add_argument('--injury', type=int, default=1,
                    help='InjurySwitch: 1 = standard full-obstruction rule (Ch3 reference)')
    ap.add_argument('--tsim', type=float, default=400.0)
    ap.add_argument('--traj-dt', type=float, default=0.5)
    ap.add_argument('--t-early', type=float, default=4.0)
    ap.add_argument('--t-late', type=float, default=None, help='auto mode only; default last frame')
    ap.add_argument('--door-R', type=float, default=2.0,
                    help='radius (m) of the door region used to locate the clog frame (auto mode)')
    ap.add_argument('--times', default=None,
                    help='MANUAL: 2-6 comma-separated times in seconds, e.g. --times 0,51,204,280. '
                         "Tolerant of 't=' and spaces. Overrides the automatic early/clog/late choice.")
    ap.add_argument('--labels', default=None,
                    help='optional phase words, one per panel, semicolon-separated, '
                         'e.g. "early;clogging;draining;late"')
    ap.add_argument('--blue-label', default='active', help='legend label for uninjured agents')
    ap.add_argument('--red-label', default='injured', help='legend label for casualties')
    ap.add_argument('--contact-frames', type=int, default=0,
                    help='if >0, also write a full-evacuation contact sheet with this many '
                         'frames (e.g. 12) as fis_mechanism_contactsheet.* (for the appendix)')
    ap.add_argument('--out', default=OUT_DEFAULT, help='output path WITHOUT extension')
    args = ap.parse_args()

    TS.apply()
    case = cases.get('fis_room')
    Lx, Ly = 15.0, 15.0
    cx, cy = Lx, Ly / 2.0
    ov = dict(InjurySwitch=int(args.injury), DoorWidth=float(args.door))
    print(f'[fis-snap] re-running FIS room: v0={args.v0} seed={args.seed} '
          f'door={args.door} m InjurySwitch={args.injury} (ONE representative run)...', flush=True)
    _times, _final, frames, elapsed = c_exe.run_with_trajectory(
        case, args.v0, args.seed, traj_dt=args.traj_dt, max_sim_time=args.tsim,
        overrides=ov, exe=c_exe.EXE_SCENARIO, timeout_s=900, full_traj=True)
    if not frames:
        print('[fis-snap] ERROR: no trajectory frames returned. Binary:', c_exe.EXE_SCENARIO)
        sys.exit(1)

    ts = np.array([t for t, _ in frames])
    print(f'[fis-snap] {len(frames)} frames, t in [{ts.min():.1f}, {ts.max():.1f}] s, wall={elapsed:.1f}s')

    counts = [door_region_count(ags, cx, cy, args.door_R) for _, ags in frames]
    phase_labels = None
    if args.times:                                  # MANUAL selection (after review)
        want = parse_times(args.times)
        if not (2 <= len(want) <= 6):
            print('[fis-snap] ERROR: --times needs 2 to 6 comma-separated values'); sys.exit(1)
        idxs = [int(np.argmin(np.abs(ts - w))) for w in want]
        if args.labels:
            phase_labels = [s.strip() for s in args.labels.split(';')]
            if len(phase_labels) != len(idxs):
                print('[fis-snap] WARNING: --labels count != --times count; ignoring --labels')
                phase_labels = None
    else:                                           # AUTOMATIC fallback (3 panels)
        i_early = int(np.argmin(np.abs(ts - args.t_early)))
        i_clog = int(np.argmax(counts))
        i_late = (len(frames) - 1 if args.t_late is None
                  else int(np.argmin(np.abs(ts - args.t_late))))
        idxs = [i_early, i_clog, i_late]
        phase_labels = ['early evacuation', 'clogging at the door', 'late evacuation']

    npanel = len(idxs)
    tags = 'abcdefgh'
    geo = geometry.fis_room(DW=args.door, Lx=Lx, Ly=Ly)
    fig, axs = plt.subplots(1, npanel, figsize=(1.9 * npanel, 3.1),
                            squeeze=False, layout='compressed')
    axs = axs[0]
    for k, (ax, i) in enumerate(zip(axs, idxs)):
        t, ags = frames[i]
        geometry.draw(ax, geo)
        (okx, oky, okd), (ijx, ijy, ijd) = split(ags)
        draw_discs(ax, okx, oky, okd, 'tab:blue', z=4)
        draw_discs(ax, ijx, ijy, ijd, 'crimson', z=5)
        ax.set(xlim=(-0.5, Lx + 0.5), ylim=(-0.5, Ly + 0.5))
        ax.set_aspect('equal')
        head = f'({tags[k]}) {phase_labels[k]}\n' if phase_labels else f'({tags[k]})  '
        ax.set_title(f'{head}$t={t:.0f}$ s,  injured $={len(ijx)}$', fontsize=10)
        ax.set_xlabel('$x$ (m)', fontsize=10)
        ax.tick_params(labelsize=9)
    axs[0].set_ylabel('$y$ (m)', fontsize=10)

    handles = [Line2D([0], [0], marker='o', ls='', mfc='tab:blue', mec='white', ms=8,
                      label=f'{args.blue_label} agent'),
               Line2D([0], [0], marker='o', ls='', mfc='crimson', mec='white', ms=8,
                      label=f'{args.red_label} agent'),
               Line2D([0], [0], color='lime', lw=4, label='exit door')]
    fig.legend(handles=handles, loc='lower center', ncol=3, fontsize=9.5, frameon=False,
               bbox_to_anchor=(0.5, -0.05))

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    for ext in ('pdf', 'png', 'svg'):
        fig.savefig(f'{args.out}.{ext}', bbox_inches='tight')
    plt.close(fig)

    if args.contact_frames > 0:
        cs_out = os.path.join(os.path.dirname(args.out), 'fis_mechanism_contactsheet')
        render_contact_sheet(frames, geo, args.contact_frames, cs_out,
                             args.blue_label, args.red_label, Lx, Ly)
        print(f'[fis-snap] wrote contact sheet {cs_out}.pdf / .png / .svg')

    i_clog_auto = int(np.argmax(counts))
    print(f'[fis-snap] panel times = {", ".join(f"{ts[i]:.1f}s" for i in idxs)}'
          f'{"  [manual]" if args.times else "  [auto]"}')
    print(f'[fis-snap] (reference) peak door-region crowding at t={ts[i_clog_auto]:.1f}s '
          f'({counts[i_clog_auto]} agents within {args.door_R} m of the door)')
    print(f'[fis-snap] wrote {args.out}.pdf / .png / .svg')


if __name__ == '__main__':
    main()
