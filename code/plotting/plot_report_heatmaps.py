"""
plot_report_heatmaps.py   (NEW, non-destructive report-formatting helper)
=========================================================================
Re-renders three existing single-row heatmap strips in the SAME visual style as
bend_open_vs_restricted_density.pdf, for readability/consistency in the v2 report:

  1. bend_chamfer_contact        -- contact-force field vs inner-corner radius (90 deg bottleneck)
  2. bend_chamfer_density_open   -- time-averaged density vs inner-corner radius (90 deg open)
  3. bend_pressure_appendix      -- crowd-pressure field across turn angle (open bend, lam=12)

Shared style: full-corridor panels; bold suptitle + bold per-panel subtitles on top;
HORIZONTAL colourbar at the bottom with 2-decimal ticks (0.00..vmax); ticks every 5 m;
y-tick labels only on the leftmost panel; x-ticks on all; walls/door/box line widths as
in the open-vs-restricted figure; the OPEN outlet draws nothing across the end.

The three remain SEPARATE figures (they live in different places in the thesis) -- only the
formatting is unified. Subtitles/contents are preserved; only y-extent follows each geometry
(90 deg chamfer crops to ~20 m; the bend crops to ~30 m).

This does NOT modify any original code: it imports plot_chamfer.py and plot_bottleneck.py to
reuse their geometry builders / grid loaders, reads grids/*.csv only, and writes NEW
"*_formatted" files (it never overwrites the current PDFs).

Run from the repo root (writes _formatted PDFs next to the originals for comparison):
  .venv/bin/python -u bend_bottleneck/plot_report_heatmaps.py

Optional: per-run overrides + style knobs (defaults shown in --help). Drop --suffix to overwrite.
"""
import os, sys, glob, argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.path import Path as MPath
from matplotlib.patches import PathPatch
from matplotlib.ticker import FormatStrFormatter, MultipleLocator

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
import plot_bottleneck as PB     # F=make_field_bottleneck, load_grid, _crop, _walls (bend), consts
import plot_chamfer as PCH       # FC=make_field_chamfer, _walls (chamfer w/ arc)
import thesis_style as TS

FIGS_DEFAULT = os.path.join(ROOT, '..', '..', 'figures', '_regenerated')
TAGS = 'abcdefgh'


def _door_or_cap(ax, g, sx, sy, draw_door):
    """Restricted: black walls flanking a green door. Open: nothing at the outlet."""
    if draw_door:
        ax.plot([g['O_in'][0]-sx, g['door_hi'][0]-sx], [g['O_in'][1]-sy, g['door_hi'][1]-sy],
                'k-', lw=1.6, zorder=5)
        ax.plot([g['O_out'][0]-sx, g['door_lo'][0]-sx], [g['O_out'][1]-sy, g['door_lo'][1]-sy],
                'k-', lw=1.6, zorder=5)
        ax.plot([g['door_lo'][0]-sx, g['door_hi'][0]-sx], [g['door_lo'][1]-sy, g['door_hi'][1]-sy],
                'lime', lw=3.0, zorder=6)


def render_strip(kind, panels, key, suptitle, clabel, out, args, draw_door):
    """panels = list of (subtitle_label, grid_dict, geometry_dict)."""
    if not panels:
        print(f'  [skip] no panels for {out}'); return
    XX, YY = np.meshgrid(PB.XC, PB.YC)
    vmax = 0.0
    for _, gr, _ in panels:
        v = gr[key][np.isfinite(gr[key])]
        if v.size:
            vmax = max(vmax, float(np.nanpercentile(v, 99.5)))
    vmax = max(vmax, 1e-6)

    crops = {i: PB._crop(g) for i, (_, _, g) in enumerate(panels)}
    dw = max(x1-x0 for x0, x1, y0, y1 in crops.values())
    dh = max(y1-y0 for x0, x1, y0, y1 in crops.values())
    ncol = len(panels)
    panel_w = args.panel_w
    panel_h = panel_w * (dh/dw)
    figw = panel_w * ncol + 0.7
    figh = panel_h + 1.9                                # one row + suptitle + bottom colourbar
    fig = plt.figure(figsize=(figw, figh))
    sf_p, sf_c = fig.subfigures(2, 1, height_ratios=[panel_h, 0.6], hspace=0.0)
    sf_p.suptitle(suptitle, y=0.99, fontsize=args.sup_size, fontweight='bold')
    axs = np.atleast_1d(sf_p.subplots(1, ncol))
    sf_p.subplots_adjust(left=0.055, right=0.995, top=0.85, bottom=0.16, wspace=0.07)

    walls_fn = PCH._walls if kind == 'chamfer' else PB._walls
    box_key = 'Vin_sharp' if kind == 'chamfer' else 'Vin'
    cf = None
    for i, (label, gr, g) in enumerate(panels):
        ax = axs[i]
        x0, x1, y0, y1 = crops[i]
        sx = 0.5*(x0+x1) - dw/2
        sy = 0.5*(y0+y1) - dh/2
        walk = MPath(g['poly']).contains_points(
            np.column_stack([XX.ravel(), YY.ravel()])).reshape(PB.NY, PB.NX)
        Z = np.where(walk, gr[key], np.nan)
        cf = ax.contourf(PB.XC-sx, PB.YC-sy, np.where(np.isfinite(Z), Z, 0.0),
                         levels=np.linspace(0, vmax, 25), cmap=TS.CMAP, extend='max')
        clip = PathPatch(MPath([(px-sx, py-sy) for px, py in g['poly']]),
                         transform=ax.transData, fc='none', ec='none')
        ax.add_patch(clip)
        try:
            cf.set_clip_path(clip)
        except AttributeError:
            for c in cf.collections:
                c.set_clip_path(clip)
        for wpts in walls_fn(g):
            ax.plot([p[0]-sx for p in wpts], [p[1]-sy for p in wpts], 'k-', lw=1.6, zorder=5)
        _door_or_cap(ax, g, sx, sy, draw_door)
        box = g.get(box_key)
        if box is not None:
            vx, vy = box
            ax.add_patch(plt.Rectangle((vx-PB.CORNER_HALF-sx, vy-PB.CORNER_HALF-sy),
                                       2*PB.CORNER_HALF, 2*PB.CORNER_HALF, fill=False,
                                       ec='magenta', lw=1.3, ls='--', zorder=6))
        ax.set(xlim=(0, dw), ylim=(0, dh))
        ax.set_aspect('equal')
        ax.set_title(f'({TAGS[i]}) {label}', fontsize=args.subtitle_size, fontweight='bold', pad=4)
        ax.xaxis.set_major_locator(MultipleLocator(5))
        ax.yaxis.set_major_locator(MultipleLocator(5))
        ax.tick_params(axis='y', labelsize=args.tick_size, labelleft=(i == 0))
        ax.tick_params(axis='x', labelsize=args.xtick_size)
        ax.set_xlabel('$x$ (m)', fontsize=args.axis_size)
        if i == 0:
            ax.set_ylabel('$y$ (m)', fontsize=args.axis_size)

    cax = sf_c.add_axes([0.25, 0.55, 0.5, 0.30])
    cb = fig.colorbar(cf, cax=cax, orientation='horizontal', ticks=np.linspace(0.0, vmax, 9))
    cb.ax.xaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    cb.set_label(clabel, fontsize=args.cbar_size, fontweight='bold')
    cb.ax.tick_params(labelsize=args.tick_size)

    os.makedirs(os.path.dirname(out), exist_ok=True)
    for ext in ('pdf', 'png', 'svg'):
        fig.savefig(f'{out}.{ext}', bbox_inches='tight')
    plt.close(fig)
    print(f'wrote {out}.pdf / .png / .svg   (vmax={vmax:.2f}, {ncol} panels, y up to ~{dh:.0f} m)')


def build_chamfer_panels(folder, radii, key):
    out = []
    for r in radii:
        cands = glob.glob(os.path.join(folder, 'grids', f'a{PCH.FC.ANGLE}_r{PCH.FC.rtag(r)}_lam*.csv'))
        if not cands:
            print(f'  [warn] no chamfer grid r={r} in {folder}'); continue
        out.append((f'$r={r:g}$ m', PB.load_grid(cands[0]), PCH.FC.geometry_rounded(PB.WIDTH, r)))
    return out


def build_bend_panels(folder, angles, lam):
    out = []
    for a in angles:
        p = os.path.join(folder, 'grids', f'a{int(a)}_lam{lam}.csv')
        if not os.path.exists(p):
            print(f'  [warn] no bend grid a{a} lam{lam} in {folder}'); continue
        out.append((f'${int(a)}^\\circ$', PB.load_grid(p), PB.F.geometry(int(a), PB.WIDTH)))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--chamfer-bn', default='results/bend_chamfer/chamferBN_20260605_192423')
    ap.add_argument('--chamfer-open', default='results/bend_chamfer/chamferOpen_20260605_193647')
    ap.add_argument('--pressure-run', default='results/bend_bottleneck/openbendCAP_20260605_123719')
    ap.add_argument('--pressure-lam', default='12.0', help='lam tag in the pressure grid filenames')
    ap.add_argument('--outdir', default=FIGS_DEFAULT)
    ap.add_argument('--suffix', default='_formatted',
                    help='output suffix; "" overwrites the current PDFs (default keeps originals)')
    ap.add_argument('--panel-w', type=float, default=2.2)
    ap.add_argument('--sup-size', type=float, default=21.0)
    ap.add_argument('--subtitle-size', type=float, default=20.0)
    ap.add_argument('--axis-size', type=float, default=17.0)
    ap.add_argument('--tick-size', type=float, default=16.0, help='y-tick + colourbar-tick size')
    ap.add_argument('--xtick-size', type=float, default=14.0)
    ap.add_argument('--cbar-size', type=float, default=21.0)
    a = ap.parse_args()
    TS.apply()

    def absdir(p):
        return p if os.path.isabs(p) else os.path.join(ROOT, p)

    radii = [0.0, 0.25, 0.5, 1.0, 1.5, 2.0]
    angles = [0, 45, 60, 90, 120, 135]
    o = lambda name: os.path.join(a.outdir, f'{name}{a.suffix}')

    # 1) contact field (chamfer bottleneck)
    render_strip('chamfer', build_chamfer_panels(absdir(a.chamfer_bn), radii, 'contact'),
                 'contact', 'Contact force vs inner-corner radius (bottlenecked corridor)',
                 r'Contact force (N kg$^{-1}$)', o('bend_chamfer_contact'), a, draw_door=True)

    # 2) time-averaged density (chamfer open)
    render_strip('chamfer', build_chamfer_panels(absdir(a.chamfer_open), radii, 'rho'),
                 'rho', 'Time-averaged density vs inner-corner radius (open corridor)',
                 r'Density $\rho$ (ped m$^{-2}$)', o('bend_chamfer_density_open'), a, draw_door=False)

    # 3) crowd-pressure field (open bend)
    render_strip('bend', build_bend_panels(absdir(a.pressure_run), angles, a.pressure_lam),
                 'pressure', r'Crowd pressure $\rho\,$Var$(v)$ (appendix diagnostic)',
                 r'$\rho\,$Var$(v)$ (ped s$^{-2}$)', o('bend_pressure_appendix'), a, draw_door=False)


if __name__ == '__main__':
    main()
