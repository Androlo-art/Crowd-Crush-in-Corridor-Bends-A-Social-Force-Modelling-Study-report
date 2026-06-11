"""
fis_snapshot_candidates.py   (NEW, non-destructive helper for the v2 report)
============================================================================
FIS snapshot CANDIDATE-REVIEW tool, for Chapter 3.

Faster-is-slower has high seed-to-seed variability, so the snapshot run should be
chosen by inspection -- not auto-picked. This tool RE-RUNS several FIS trials
(single 15x15 m room, one east-wall door) with trajectory capture and, for each
candidate seed, produces:

  * a CONTACT SHEET PNG (a grid of frames across the whole evacuation, each labelled
    with time and injured count) -- the primary review artifact;
  * optionally a short MP4 (--video, if ffmpeg is present);
  * METADATA to judge representativeness: seed, v0, evacuation time, casualties,
    whether the door blocked (mobile agents trapped behind a jam), time of peak
    door-region crowding, and persistent-blockage behaviour.

It also reads the EXISTING 20-seed sweep CSV and prints, at the chosen v0, the
distribution of casualties / leaving time across all seeds, the median, and -- per
candidate -- how the re-run compares to that seed's sweep row (a reproducibility
check) and whether the seed sits near the median (representative) or in the tail
(extreme). This lets you pick a representative run, not a cherry-picked extreme.

After reviewing, render the final figure with your chosen seed + three time points:
  plot_fis_snapshots_for_report.py --seed <S> --times t1,t2,t3

Uses the SAME room/config/injury rule as the Chapter 3 'standard' (full-obstruction)
sweep, so candidates are genuine members of that ensemble. Reads the released binary
(read-only) and the existing sweep CSV; runs N short sims; writes a NEW timestamped
folder. Does NOT modify any existing project code or data.

Run from the repo root (this RE-RUNS simulations -- roughly 0.5-3 min per seed at
high v0, since clogged runs go to the time cap):
  .venv/bin/python -u bend_bottleneck/fis_snapshot_candidates.py --v0 5.0 --seeds 0,1,2,3,4,5

Optional (defaults shown):
  --door 1.2 --injury 1 --tsim 400 --traj-dt 0.5 --grid-frames 12 --door-R 2.0
  --sweep-csv <standard run raw_results.csv>   --video   --out <dir>
"""
import os, sys, csv, math, argparse, datetime
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, '..', 'model'))
from sfm.backends import c_exe
from sfm import cases, geometry
import thesis_style as TS
import plot_fis_snapshots_for_report as snap   # reuse draw_discs / split / door_region_count

SWEEP_DEFAULT = os.path.join(
    ROOT, 'results', 'faster_is_slower',
    'fis_final_dw1.2_standard_20260602_091219', 'raw_results.csv')


def read_sweep(csv_path, v0):
    """Return (T_list, inj_list, row_by_seed) for the chosen v0 from the sweep CSV."""
    if not os.path.exists(csv_path):
        return [], [], {}
    Ts, injs, by_seed = [], [], {}
    with open(csv_path) as fh:
        for d in csv.DictReader(fh):
            try:
                if abs(float(d['v0']) - v0) > 1e-6:
                    continue
            except (ValueError, KeyError):
                continue
            s = int(float(d['seed']))
            T = float(d['T_evac']); inj = float(d['N_injured'])
            Ts.append(T); injs.append(inj); by_seed[s] = d
    return Ts, injs, by_seed


def rerun_fis(v0, seed, door, injury, tsim, traj_dt):
    case = cases.get('fis_room')
    ov = dict(InjurySwitch=int(injury), DoorWidth=float(door))
    times, _final, frames, elapsed = c_exe.run_with_trajectory(
        case, v0, seed, traj_dt=traj_dt, max_sim_time=tsim,
        overrides=ov, exe=c_exe.EXE_SCENARIO, timeout_s=900, full_traj=True)
    return times, frames, elapsed


def contact_sheet(frames, v0, seed, door, n_frames, door_R, outpath, Lx=15.0, Ly=15.0):
    """Grid of evenly-spaced frames across the run; each labelled t + injured."""
    TS.apply()
    geo = geometry.fis_room(DW=door, Lx=Lx, Ly=Ly)
    ncol = 4
    nrow = int(math.ceil(n_frames / ncol))
    idxs = np.linspace(0, len(frames) - 1, n_frames).round().astype(int)
    fig, axs = plt.subplots(nrow, ncol, figsize=(2.1 * ncol, 2.1 * nrow + 0.4),
                            squeeze=False, layout='compressed')
    for k, ax in enumerate(axs.flat):
        if k >= len(idxs):
            ax.axis('off'); continue
        t, ags = frames[idxs[k]]
        geometry.draw(ax, geo)
        (okx, oky, okd), (ijx, ijy, ijd) = snap.split(ags)
        snap.draw_discs(ax, okx, oky, okd, 'tab:blue', z=4)
        snap.draw_discs(ax, ijx, ijy, ijd, 'crimson', z=5)
        ax.set(xlim=(-0.5, Lx + 0.5), ylim=(-0.5, Ly + 0.5)); ax.set_aspect('equal')
        ax.set_title(f't={t:.0f}s  N={len(ags)}  inj={len(ijx)}', fontsize=9)
        ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle(f'FIS candidate: v0={v0:.1f} m/s, seed={seed}, door={door} m '
                 f'(blue = active, red = injured)', fontsize=12, fontweight='bold')
    fig.savefig(outpath, dpi=140, bbox_inches='tight')
    plt.close(fig)


def maybe_video(frames, v0, seed, door, outpath, Lx=15.0, Ly=15.0, fps=20):
    import shutil
    if not shutil.which('ffmpeg'):
        print('    [video] ffmpeg not found -- skipping MP4 (contact sheet still written)')
        return False
    import matplotlib.animation as manim
    geo = geometry.fis_room(DW=door, Lx=Lx, Ly=Ly)
    fig, ax = plt.subplots(figsize=(5, 5))
    def upd(k):
        ax.clear(); geometry.draw(ax, geo)
        t, ags = frames[k]
        (okx, oky, okd), (ijx, ijy, ijd) = snap.split(ags)
        snap.draw_discs(ax, okx, oky, okd, 'tab:blue', z=4)
        snap.draw_discs(ax, ijx, ijy, ijd, 'crimson', z=5)
        ax.set(xlim=(-0.5, Lx + 0.5), ylim=(-0.5, Ly + 0.5)); ax.set_aspect('equal')
        ax.set_title(f'v0={v0:.1f} seed={seed}  t={t:.1f}s  inj={len(ijx)}', fontsize=10)
    ani = manim.FuncAnimation(fig, upd, frames=len(frames), interval=1000 / fps)
    ani.save(outpath, writer=manim.FFMpegWriter(fps=fps, bitrate=2400))
    plt.close(fig)
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--v0', type=float, default=5.0)
    ap.add_argument('--seeds', default='0,1,2,3,4,5')
    ap.add_argument('--door', type=float, default=1.2)
    ap.add_argument('--injury', type=int, default=1,
                    help='InjurySwitch: 1 = standard full-obstruction (Ch3 reference sweep)')
    ap.add_argument('--tsim', type=float, default=400.0)
    ap.add_argument('--traj-dt', type=float, default=0.5)
    ap.add_argument('--grid-frames', type=int, default=12)
    ap.add_argument('--door-R', type=float, default=2.0)
    ap.add_argument('--sweep-csv', default=SWEEP_DEFAULT,
                    help='existing 20-seed sweep CSV (for representativeness/reproducibility)')
    ap.add_argument('--video', action='store_true', help='also write an MP4 per candidate (needs ffmpeg)')
    ap.add_argument('--out', default=None, help='output dir (default: timestamped under results/)')
    args = ap.parse_args()

    seeds = [int(x) for x in args.seeds.split(',')]
    if args.out:
        outdir = args.out
    else:
        ts = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        outdir = os.path.join(ROOT, 'results', 'fis_snapshot_candidates', f'cand_{ts}')
    os.makedirs(outdir, exist_ok=True)

    # sweep distribution at this v0 (for representativeness)
    Ts, injs, by_seed = read_sweep(args.sweep_csv, args.v0)
    if injs:
        med_inj = float(np.median(injs)); med_T = float(np.median(Ts))
        print(f'[sweep] v0={args.v0}: {len(injs)} seeds | casualties median={med_inj:.1f} '
              f'[{min(injs):.0f},{max(injs):.0f}] | T_evac median={med_T:.0f}s '
              f'[{min(Ts):.0f},{max(Ts):.0f}]')
        # seed closest to the median casualties (a natural "representative" suggestion)
        cand_for_med = sorted(by_seed.items(),
                              key=lambda kv: abs(float(kv[1]['N_injured']) - med_inj))
        print(f'[sweep] seed closest to median casualties: '
              f'{cand_for_med[0][0]} (N_injured={float(cand_for_med[0][1]["N_injured"]):.0f})')
    else:
        med_inj = med_T = float('nan')
        print(f'[sweep] WARNING: no sweep rows at v0={args.v0} in {args.sweep_csv}')

    meta_rows = []
    for s in seeds:
        print(f'[rerun] seed={s} v0={args.v0} door={args.door} InjurySwitch={args.injury} ...', flush=True)
        times, frames, elapsed = rerun_fis(args.v0, s, args.door, args.injury,
                                           args.tsim, args.traj_dt)
        if not frames:
            print(f'  seed={s}: NO frames returned -- skipping'); continue
        tarr = np.array([t for t, _ in frames])
        n_exit = len(times)
        T_evac_re = (max(times.values()) if n_exit >= cases.get('fis_room')['params']['N0']
                     else args.tsim)
        _t_last, ags_last = frames[-1]
        n_room_final = len(ags_last)
        n_inj_final = sum(1 for a in ags_last if a[0] == 1)
        n_mobile_trapped = n_room_final - n_inj_final
        blocked = n_mobile_trapped >= 5            # mobile agents stuck behind a jam
        counts = [snap.door_region_count(a, 15.0, 7.5, args.door_R) for _, a in frames]
        i_peak = int(np.argmax(counts))
        t_peak = float(tarr[i_peak])

        # sweep comparison (reproducibility + representativeness)
        srow = by_seed.get(s)
        T_csv = float(srow['T_evac']) if srow else float('nan')
        inj_csv = float(srow['N_injured']) if srow else float('nan')
        repr_tag = ''
        if injs and not math.isnan(inj_csv):
            lo, hi = np.percentile(injs, 25), np.percentile(injs, 75)
            repr_tag = 'REPRESENTATIVE (near median)' if lo <= inj_csv <= hi else 'tail/extreme'

        sheet = os.path.join(outdir, f'candidate_v{args.v0:.1f}_seed{s}.png')
        contact_sheet(frames, args.v0, s, args.door, args.grid_frames, args.door_R, sheet)
        if args.video:
            maybe_video(frames, args.v0, s, args.door,
                        os.path.join(outdir, f'candidate_v{args.v0:.1f}_seed{s}.mp4'))

        meta_rows.append(dict(
            seed=s, v0=args.v0, T_evac_rerun=round(T_evac_re, 1), T_evac_sweepCSV=round(T_csv, 1),
            casualties_rerun=n_inj_final, casualties_sweepCSV=round(inj_csv, 1),
            door_blocked=int(blocked), n_in_room_final=n_room_final,
            n_mobile_trapped=n_mobile_trapped, t_peak_door_crowd=round(t_peak, 1),
            peak_door_count=counts[i_peak], representativeness=repr_tag,
            wall_s=round(elapsed, 1)))
        print(f'  seed={s}: T_evac~{T_evac_re:.0f}s (CSV {T_csv:.0f}) | casualties={n_inj_final} '
              f'(CSV {inj_csv:.0f}) | blocked={blocked} (trapped mobile={n_mobile_trapped}) | '
              f'peak door crowd t={t_peak:.0f}s | {repr_tag}')

    # write metadata CSV + a run_info.txt
    if meta_rows:
        cols = list(meta_rows[0].keys())
        with open(os.path.join(outdir, 'candidates_metadata.csv'), 'w', newline='') as fh:
            w = csv.DictWriter(fh, fieldnames=cols); w.writeheader(); w.writerows(meta_rows)
    with open(os.path.join(outdir, 'run_info.txt'), 'w') as fh:
        fh.write('FIS snapshot candidate review (trajectory-capture re-runs; non-destructive)\n')
        fh.write(f'v0={args.v0} seeds={seeds} door={args.door} InjurySwitch={args.injury} '
                 f'tsim={args.tsim} traj_dt={args.traj_dt}\n')
        fh.write(f'sweep_csv={args.sweep_csv}\n')
        fh.write('Reproducibility check: T_evac_rerun / casualties_rerun should match the '
                 'sweep CSV columns for the same (v0, seed).\n')
    print(f'\n[done] wrote contact sheets + candidates_metadata.csv to:\n  {outdir}')
    print('Review the contact sheets, then render the final figure with e.g.:')
    print('  .venv/bin/python -u bend_bottleneck/plot_fis_snapshots_for_report.py '
          f'--v0 {args.v0:.1f} --seed <CHOSEN> --times <t1,t2,t3>')


if __name__ == '__main__':
    main()
