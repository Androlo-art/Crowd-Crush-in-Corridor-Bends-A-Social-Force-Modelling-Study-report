#!/usr/bin/env python3
"""
Regenerate the bend and capacity report figures from the processed results in results/:
open-bend capacity, the restricted-exit crush, the open-vs-restricted density comparison,
the crowd-pressure fields, and the coupled turn-angle x exit-width figures (phase map,
occupancy saturation, realised outflow). Reads result tables and saved grids only; runs no
simulation. Outputs go to figures/_regenerated/.

Run from code/plotting/report_figures/:
  python plot_bend_report_figures.py
"""
import os, sys, glob, shutil, datetime
from types import SimpleNamespace
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator, FormatStrFormatter

_HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(_HERE, '..', '..', '..'))
sys.path.insert(0, os.path.dirname(_HERE))                        # shared plotting modules

# project modules (read-only re-use; guarantees identical data + styling)
import thesis_style as TS
import plot_bottleneck as PB                  # fig 1 helpers (read_rows, _f, CAP_GREEN, XC/YC, _crop, _walls); runs TS.apply()
import analyze_capacity as AC                 # figs 4,5,6 data (derive)
import maps_capacity as MC                    # fig 7 helpers (_cells_from_grids, load_grid, _overlay, DOMX/Y)
import plot_report_heatmaps as PRH            # fig 3 (build_bend_panels, render_strip)
import plot_open_vs_restricted_density as PD  # fig 2 (load_row, panel)
TS.apply()

# ---- paths ----------------------------------------------------------------
THESIS_FIG = os.path.join(REPO, 'figures', '_regenerated')       # regeneration target (NOT the thesis)
PREVIEW = os.path.join(THESIS_FIG, 'previews')
os.makedirs(PREVIEW, exist_ok=True)
STAMP = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')

# ---- verified data folders ------------------------------------------------
OPEN  = os.path.join(REPO, 'results/02_open_and_restricted_bend/open_bend_capacity')        # fig1 data, fig3 grids, fig4-6 --open
CAPCH = os.path.join(REPO, 'results/04_turn_exit_coupling/capacity_chamfered')              # figs 4-7 data
RESTR = os.path.join(REPO, 'results/02_open_and_restricted_bend/restricted_bend_bottleneck')# fig2 restricted row
ANGLES6 = [0, 45, 60, 90, 120, 135]


def backup_original(name):
    src = os.path.join(THESIS_FIG, name + '.pdf')
    if not os.path.exists(src):
        return
    if glob.glob(os.path.join(THESIS_FIG, '%s_backup_*.pdf' % name)):   # back up the original ONCE
        print('  backup  %-34s -> (already exists, skipped)' % (name + '.pdf'))
        return
    dst = os.path.join(THESIS_FIG, '%s_backup_%s.pdf' % (name, STAMP))
    shutil.copy2(src, dst)
    print('  backup  %-34s -> %s' % (name + '.pdf', os.path.basename(dst)))


def save_revA(fig, name):
    pdf = os.path.join(THESIS_FIG, '%s_revA.pdf' % name)
    fig.savefig(pdf, bbox_inches='tight')
    fig.savefig(os.path.join(PREVIEW, '%s_revA.png' % name), dpi=150, bbox_inches='tight')
    plt.close(fig)
    print('  wrote   %s' % pdf)


# ===== Figure 1: bend_open_capacity (readability: larger legend/titles/axes) ===
def fig1_open_capacity():
    name = 'bend_open_capacity'; backup_original(name)
    agg = PB.read_rows(os.path.join(OPEN, 'aggregated.csv')); _f = PB._f
    angles = sorted({int(_f(r, 'angle')) for r in agg}); lams = sorted({_f(r, 'lam') for r in agg})
    lo, hi = min(lams), max(lams)
    LEG, SUB, AX, TICK = 11.5, 14, 12.5, 10.5    # bumped from 8 / ~11 / ~11 / 9 (titles wrapped to fit)
    fig, ax = plt.subplots(1, 3, figsize=(10.5, 4.1), constrained_layout=True)
    # (a) outflow vs inflow per angle + identity line
    maxflow = {}
    for a in angles:
        rs = sorted([r for r in agg if int(_f(r, 'angle')) == a], key=lambda r: _f(r, 'lam'))
        xs = np.array([_f(r, 'lam') for r in rs]); ys = np.array([_f(r, 'inflow_rate_mean') for r in rs])
        ax[0].plot(xs, ys, 'o-', color=TS.angle_color(a), ms=4, label=f'${a}^\\circ$'); maxflow[a] = np.nanmax(ys)
    ax[0].plot([lo, hi], [lo, hi], 'k:', lw=1, label='outflow = inflow')
    ax[0].set_xlabel('Agent inflow rate (ag s$^{-1}$)', fontsize=AX)
    ax[0].set_ylabel('Agent outflow rate (ag s$^{-1}$)', fontsize=AX)
    ax[0].set_xticks(lams); ax[0].tick_params(axis='both', labelsize=TICK)
    ax[0].set_title('(a) Outflow rate at different\nbend angles', loc='left', fontweight='bold', fontsize=SUB)
    ax[0].legend(fontsize=LEG, ncol=1, loc='upper left', framealpha=0.85, handlelength=1.4, labelspacing=0.3)
    ax[0].grid(alpha=0.3)
    # (b) maximum outflow vs angle
    aa = sorted(maxflow); ax[1].plot(aa, [maxflow[a] for a in aa], 'o-', color=PB.CAP_GREEN)
    ax[1].set_xlabel('Turn angle (deg)', fontsize=AX)
    ax[1].set_ylabel('Maximum agent outflow (ag s$^{-1}$)', fontsize=AX)
    ax[1].set_xticks(aa); ax[1].tick_params(axis='both', labelsize=TICK)
    ax[1].set_title('(b) Maximum agent outflow\nvs bend angle', loc='left', fontweight='bold', fontsize=SUB)
    ax[1].grid(alpha=0.3); TS.yfmt(ax[1], 1)
    # (c) inner-corner density vs inflow per angle
    for a in angles:
        rs = sorted([r for r in agg if int(_f(r, 'angle')) == a], key=lambda r: _f(r, 'lam'))
        xs = np.array([_f(r, 'lam') for r in rs]); cs = np.array([_f(r, 'corner_rho_mean') for r in rs])
        es = np.array([_f(r, 'corner_rho_ci95') for r in rs]); ok = np.isfinite(cs)
        if ok.any():
            col = TS.angle_color(a); eok = np.where(np.isfinite(es[ok]), es[ok], 0.0)
            ax[2].fill_between(xs[ok], cs[ok] - eok, cs[ok] + eok, color=col, alpha=0.18, lw=0)
            ax[2].plot(xs[ok], cs[ok], 'o-', color=col, ms=4, label=f'${a}^\\circ$')
    ax[2].set_xlabel('Agent inflow rate (ag s$^{-1}$)', fontsize=AX)
    ax[2].set_ylabel('Inner-corner density (ped m$^{-2}$)', fontsize=AX)
    ax[2].set_xticks(lams); ax[2].tick_params(axis='both', labelsize=TICK); TS.yfmt(ax[2], 1); ax[2].grid(alpha=0.3)
    ax[2].set_title('(c) Inner-corner density\nvs inflow rate', loc='left', fontweight='bold', fontsize=SUB)
    ax[2].legend(fontsize=LEG, ncol=1, loc='lower right', framealpha=0.85, handlelength=1.4, labelspacing=0.3)
    save_revA(fig, name)


# ===== Figure 2: open vs restricted density (new row titles only) ===========
def fig2_density():
    name = 'bend_open_vs_restricted_density'; backup_original(name)
    angles, lam, open_lam = ANGLES6, 8.0, 8.0
    og, ogeo = PD.load_row(OPEN, angles, open_lam)
    rg, rgeo = PD.load_row(RESTR, angles, lam)
    keys = [a for a in angles if a in og and a in rg]
    vmax = 0.0
    for grids in (og, rg):
        for a in keys:
            v = grids[a]['rho'][np.isfinite(grids[a]['rho'])]
            if v.size: vmax = max(vmax, float(np.nanpercentile(v, 99.5)))
    vmax = max(vmax, 1e-6)
    allgeo = {**rgeo, **ogeo}
    crops = {a: PB._crop(allgeo[a]) for a in keys}
    dw = max(x1 - x0 for x0, x1, y0, y1 in crops.values())
    dh = max(y1 - y0 for x0, x1, y0, y1 in crops.values())
    boxes = {a: (0.5 * (x0 + x1) - dw / 2, 0.5 * (y0 + y1) - dh / 2)
             for a, (x0, x1, y0, y1) in crops.items()}
    # sizes IDENTICAL to plot_open_vs_restricted_density defaults (figure otherwise unchanged)
    panel_w, ROW, ANG, AXs, TICK, XTICK, CBAR = 2.2, 22.0, 20.0, 17.0, 16.0, 14.0, 21.0
    ncol = len(keys); panel_h = panel_w * (dh / dw)
    figw = panel_w * ncol + 0.7; figh = 2.0 * panel_h + 2.0
    fig = plt.figure(figsize=(figw, figh))
    sfs = fig.subfigures(3, 1, height_ratios=[panel_h, panel_h, 0.9], hspace=0.0)
    XX, YY = np.meshgrid(PB.XC, PB.YC)
    rows = [('(a) Open bend',                         og, ogeo, True,  sfs[0], False),
            ('(b) Bottleneck bend (restricted exit)', rg, rgeo, False, sfs[1], True)]
    cf = None
    for rlab, grids, geoms, open_mode, sf, show_x in rows:
        sf.suptitle(rlab, y=0.99, fontsize=ROW, fontweight='bold')
        axs = np.atleast_1d(sf.subplots(1, ncol))
        sf.subplots_adjust(left=0.06, right=0.995, top=0.85,
                           bottom=(0.17 if show_x else 0.03), wspace=0.07)
        for ci, a in enumerate(keys):
            ax = axs[ci]; bx, by = boxes[a]
            cf = PD.panel(ax, geoms[a], grids[a], vmax, bx, by, dw, dh, open_mode, XX, YY)
            ax.set_title(f'${a}^\\circ$', fontsize=ANG, fontweight='bold', pad=4)
            ax.xaxis.set_major_locator(MultipleLocator(5.0)); ax.yaxis.set_major_locator(MultipleLocator(5.0))
            ax.tick_params(axis='y', labelsize=TICK, labelleft=(ci == 0))
            ax.tick_params(axis='x', labelsize=XTICK, labelbottom=show_x)
            if ci == 0: ax.set_ylabel('$y$ (m)', fontsize=AXs)
            if show_x: ax.set_xlabel('$x$ (m)', fontsize=AXs)
    cax = sfs[2].add_axes([0.25, 0.55, 0.5, 0.30])
    cb = fig.colorbar(cf, cax=cax, orientation='horizontal', ticks=np.linspace(0.0, vmax, 9))
    cb.ax.xaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    cb.set_label(r'Crowd density $\rho$ (ped m$^{-2}$)', fontsize=CBAR, fontweight='bold')
    cb.ax.tick_params(labelsize=TICK)
    save_revA(fig, name)


# ===== Figure 3: crowd-pressure heatmaps (drop "appendix diagnostic") ========
def fig3_pressure():
    name = 'bend_pressure_appendix'; backup_original(name)
    args_ns = SimpleNamespace(panel_w=2.2, sup_size=21.0, subtitle_size=20.0, axis_size=17.0,
                              tick_size=16.0, xtick_size=14.0, cbar_size=21.0)
    panels = PRH.build_bend_panels(OPEN, ANGLES6, '12.0')
    out_prev = os.path.join(PREVIEW, '%s_revA' % name)           # render_strip writes pdf/png/svg here
    PRH.render_strip('bend', panels, 'pressure',
                     r'Crowd pressure $\rho\,\mathrm{Var}(v)$ across turn angle',
                     r'$\rho\,$Var$(v)$ (ped s$^{-2}$)', out_prev, args_ns, draw_door=False)
    shutil.copy2(out_prev + '.pdf', os.path.join(THESIS_FIG, '%s_revA.pdf' % name))
    print('  wrote   %s' % os.path.join(THESIS_FIG, '%s_revA.pdf' % name))


# ===== Figures 4/5/6: capacity line figures (title/legend wording only) ======
def fig4_throughput(D):
    name = 'bend_capacity_throughput'; backup_original(name)
    angles, doors = D['angles'], D['doors']; dd = np.array(doors)
    fig, ax = plt.subplots(figsize=(TS.FIGW, 0.62 * TS.FIGW), constrained_layout=True)
    for a in angles:
        col = TS.angle_color(a)
        ys = np.array([D['Qsys'][(a, d)] for d in doors]); satm = np.array([D['saturated'][a][d] for d in doors])
        ax.plot(dd, ys, '-', color=col, label=f'{int(a)}°', zorder=2)
        ax.plot(dd[satm], ys[satm], 'o', color=col, mfc=col, ms=5, zorder=3)
        ax.plot(dd[~satm], ys[~satm], 'o', color=col, mfc='white', ms=5, zorder=3)
        if D['Qbend_open'] and a in D['Qbend_open']:
            ax.axhline(D['Qbend_open'][a], color=col, ls=':', lw=1.0, alpha=0.8)
    ax.set_xlabel('exit width $d$ (m)')
    ax.set_ylabel(r'realised exit flow $Q_{\mathrm{sys}}$ (ped s$^{-1}$)')
    ax.set_title('Realised outflow versus exit width')
    ax.legend(title='turn angle', ncol=2); TS.yfmt(ax, 1); TS.xfmt(ax, 1)
    save_revA(fig, name)


def fig5_saturation(D):
    name = 'bend_capacity_saturation'; backup_original(name)
    angles, doors = D['angles'], D['doors']; dd = np.array(doors)
    fig, ax = plt.subplots(figsize=(TS.FIGW, 0.62 * TS.FIGW), constrained_layout=True)
    for a in angles:
        col = TS.angle_color(a)
        ax.plot(dd, [D['meann'][(a, d)] for d in doors], '-o', color=col, label=f'{int(a)}°')
        if np.isfinite(D['dstar'][a]):
            ax.axvline(D['dstar'][a], color=col, ls=':', lw=1.0, alpha=0.7)
    ax.set_xlabel('exit width $d$ (m)'); ax.set_ylabel('mean occupancy $\\bar N$ (agents)')
    ax.set_title('Occupancy transition versus exit width')
    ax.legend(title='turn angle', ncol=2); TS.xfmt(ax, 1)
    save_revA(fig, name)


def fig6_phase(D):
    name = 'bend_capacity_phase'; backup_original(name)
    angles, doors = D['angles'], D['doors']; dd = np.array(doors); G = D['G']
    fig, ax = plt.subplots(figsize=(TS.FIGW, 0.62 * TS.FIGW), constrained_layout=True)
    AA, DDg, CC = [], [], []
    for a in angles:
        for d in doors:
            AA.append(float(a)); DDg.append(d); CC.append(G(a, d, 'casualties_mean'))
    sc = ax.scatter(AA, DDg, c=np.nan_to_num(CC), cmap='turbo', s=140, edgecolor='k', lw=0.5)
    aa = np.array([float(a) for a in angles]); ds = np.array([D['dstar'][a] for a in angles]); ok = np.isfinite(ds)
    if ok.sum() >= 2:
        ax.plot(aa[ok], ds[ok], 'k-o', lw=2.2, label=r'critical exit width $d^\ast(\theta)$')
        ax.fill_between(aa[ok], ds[ok], dd.max() + 0.2, color='0.85', alpha=0.5, zorder=0)
        ax.text(aa[ok].mean(), min(ds[ok]) + 0.2, 'exit-limited\n(crush-prone)', ha='center', fontsize=9)
        ax.text(aa[ok].mean(), dd.max() - 0.2, 'bend-limited (relieved)', ha='center', va='top', fontsize=9)
    ax.set_xlabel('turn angle $\\theta$ (deg)'); ax.set_ylabel('exit width $d$ (m)')
    ax.set_ylim(dd.min() - 0.2, dd.max() + 0.2)
    ax.set_title('Crush across the turn-angle–exit-width plane')
    cb = fig.colorbar(sc, ax=ax); cb.set_label('mean casualties per run')
    if ok.sum() >= 2: ax.legend(loc='upper right', fontsize=9)
    save_revA(fig, name)


# ===== Figure 7: density montage (no suptitle; bold+larger row/col labels) ====
def fig7_montage():
    name = 'bend_capacity_density_montage'; backup_original(name)
    MC.OVERLAY_R = 0.25                      # chamfered wall overlay (capovCH geometry)
    field = 'rho'
    cells = MC._cells_from_grids(CAPCH)
    angles = sorted({a for a, d, p in cells}); doors = sorted({d for a, d, p in cells})
    vmax = 0.0; grids = {}
    for a, d, p in cells:
        Gg = MC.load_grid(p); grids[(a, d)] = Gg; vmax = max(vmax, np.nanpercentile(Gg[field], 99))
    nr, nc = len(angles), len(doors)
    fig, axes = plt.subplots(nr, nc, figsize=(1.55 * nc, 2.4 * nr), constrained_layout=True)
    SUB = 13                                 # column (d) headers + row (angle) labels: larger + bold
    im = None
    for i, a in enumerate(angles):
        for j, d in enumerate(doors):
            ax = axes[i, j]; Gg = grids.get((a, d))
            if Gg is not None:
                im = ax.imshow(Gg[field], origin='lower', extent=[0, MC.DOMX, 0, MC.DOMY],
                               cmap=TS.CMAP, vmin=0, vmax=vmax)
                MC._overlay(ax, a, d)
            ax.set_xlim(0, MC.DOMX); ax.set_ylim(0, MC.DOMY); ax.set_aspect('equal')
            ax.set_xticks([]); ax.set_yticks([])
            if i == 0: ax.set_title(f'$d={d:g}$ m', fontsize=SUB, fontweight='bold')
            if j == 0: ax.set_ylabel(f'${int(a)}^\\circ$', fontsize=SUB, fontweight='bold')
    cb = fig.colorbar(im, ax=axes, shrink=0.6, location='right')
    cb.set_label(r'density (ped m$^{-2}$)')
    save_revA(fig, name)
    # (overall suptitle intentionally removed)


def main():
    print('=== batch 1: bend/capacity report figures (revA) ===')
    D = AC.derive(CAPCH, OPEN)               # data for figs 4,5,6 (Q_bend dotted lines from --open)
    fig1_open_capacity()
    fig2_density()
    fig3_pressure()
    fig4_throughput(D)
    fig5_saturation(D)
    fig6_phase(D)
    fig7_montage()
    print('done. _revA PDFs (+ timestamped backups) in:\n  %s\nPNG previews in:\n  %s' % (THESIS_FIG, PREVIEW))


if __name__ == '__main__':
    main()
