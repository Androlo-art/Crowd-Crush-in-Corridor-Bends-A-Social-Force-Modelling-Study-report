"""Thesis-grade figures for the ROUNDED INNER-CORNER (chamfer) study at 90 deg.

Companion to plot_bottleneck.py; the swept variable is the inner-corner fillet
RADIUS r. Same styling (thesis_style), larger axis/title fonts for the report.

Writes ONLY into --outdir (a folder you choose) so it never overwrites figures that
already sit in the run folders. Reads aggregated.csv + grids/ — NO re-simulation.

Figures produced (into --outdir):
  fig_chamfer_lines.*               1x3 line figure:
                                    (a) inner-corner PEAK CONTACT vs r  [open + bottleneck, crush line]
                                    (b) inner-corner PEAK DENSITY vs r  [open + bottleneck, lethal line]
                                    (c) DOOR density (steady + peak) vs r [bottleneck control, lethal line]
  fig_chamfer_contact_bottleneck.*  contact-force heatmap strip per radius (the vertex
                                    hotspot at r=0 and its removal; wall/jam contact).
  fig_chamfer_density_open.*        density heatmap strip per radius (inner-wall band relaxing).
  fig_chamfer_density_bottleneck.*  density heatmap strip per radius (uniform jam; appendix).

Why these: contact force is nonzero only where bodies compress, so it is the crush-
relevant metric in the JAMMED bottleneck and ~0 in the free-flowing OPEN corridor;
density is the metric that still discriminates in the open corridor. The door panel
is bottleneck-only (the open corridor has no door) and serves as a locality control.

Run (does NOT re-simulate; writes to a NEW folder):
  python -u bend_bottleneck/plot_chamfer.py \
      --bn results/bend_chamfer/chamferBN_<stamp> \
      --open-dir results/bend_chamfer/chamferOpen_<stamp> \
      --outdir results/bend_chamfer/figs_final
"""
import os, sys, csv, glob, math, argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.path import Path as MPath
from matplotlib.patches import PathPatch
from matplotlib.ticker import FormatStrFormatter
from matplotlib.lines import Line2D

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
import make_field_chamfer as FC
import thesis_style as TS
TS.apply()

WIDTH = 5.0; RES = 0.25; DOMX, DOMY = 18.0, 32.0
NX, NY = int(DOMX/RES), int(DOMY/RES)
XC = (np.arange(NX)+0.5)*RES; YC = (np.arange(NY)+0.5)*RES
CORNER_HALF = 1.6
CRUSH_REF = 200.0*math.pi*0.6                 # crush threshold at MEAN diameter 0.6 m (~377 N/kg; per-agent 314-440 for D in [0.5,0.7])
LETHAL = 6.0
BN_COL    = '#2E8B57'        # bottleneck = green (matches the bottleneck door colour used before)
OPEN_COL  = TS.NAVY          # open corridor = navy (distinct from green/red)
THRESH_COL = TS.CRIMSON      # lethal / crush threshold lines = RED (consistent with the earlier figures)
DOOR_COL  = '#2E8B57'        # panel (c) is the bottleneck door -> same green

# Large fonts: sized so that at width=\textwidth (~6.3 in) the titles/axes render
# ~11 pt on the page. Tweak here if you include the figure at another width.
AXLAB = 19; PANEL = 18; TICK = 15; LEG = 14   # LEG = legend + threshold label (smaller)
STRIP_PANEL_W = 2.3; STRIP_LABEL_SIZE = 12; STRIP_TITLE_SIZE = 12
STRIP_SUP_SIZE = 16; STRIP_TICK_SIZE = 10.5


def _f(r, k):
    try: return float(r[k])
    except (ValueError, TypeError, KeyError): return float('nan')


def read_rows(p):
    with open(p) as fh: return list(csv.DictReader(fh))


def load_grid(path):
    out = {k: np.full((NY, NX), np.nan) for k in ('rho','contact','casualty','pressure')}
    with open(path) as fh:
        for d in csv.DictReader(fh):
            ix = int(round(float(d['x'])/RES-0.5)); iy = int(round(float(d['y'])/RES-0.5))
            if 0 <= ix < NX and 0 <= iy < NY:
                for k in out:
                    if d.get(k, '') != '': out[k][iy, ix] = float(d[k])
    return out


def parse_info(folder):
    info = dict(radii=[0.0,0.25,0.5,1.0,1.5,2.0], lambdas=[8.0])
    p = os.path.join(folder,'run_info.txt')
    if os.path.exists(p):
        for ln in open(p):
            for key in ('radii','lambdas'):
                if ln.startswith(key+'='):
                    try: info[key] = [float(x) for x in ln.split('=')[1].strip().strip('[]').split(',') if x.strip()]
                    except ValueError: pass
    return info


def _series(rows, lam, key):
    rs = sorted([r for r in rows if abs(_f(r,'lam')-lam) < 1e-6], key=lambda r:_f(r,'corner_r'))
    return (np.array([_f(r,'corner_r') for r in rs]),
            np.array([_f(r,f'{key}_mean') for r in rs]),
            np.array([_f(r,f'{key}_ci95') for r in rs]))


def _style(ax, xlabel, ylabel, tag):
    ax.set_xlabel(xlabel, fontsize=AXLAB); ax.set_ylabel(ylabel, fontsize=AXLAB)
    ax.set_title(tag, loc='center', fontweight='bold', fontsize=PANEL)   # centred over the panel
    ax.tick_params(labelsize=TICK); ax.grid(alpha=0.3)


def _band(ax, x, y, e, color, marker, label):
    ax.plot(x, y, marker+'-', color=color, lw=2, label=label, ms=6)
    ax.fill_between(x, y-e, y+e, color=color, alpha=0.15)


# ── line figure: both conditions on (a) and (b); door control on (c) ──────────
def _thresh(ax, yval, label, va='bottom'):
    """Red threshold line + label; va='bottom' puts the label ABOVE the line,
    va='top' puts it BELOW the line (both kept inside the axes)."""
    ax.axhline(yval, color=THRESH_COL, ls='--', lw=1.6, zorder=1)
    ax.text(0.98, yval, label, color=THRESH_COL, fontsize=LEG, ha='right', va=va,
            transform=ax.get_yaxis_transform(), zorder=6,
            bbox=dict(fc='white', ec='none', alpha=0.85, pad=1.5))


def fig_chamfer_lines(bn_folder, open_folder, outdir):
    bn = read_rows(os.path.join(bn_folder,'aggregated.csv'))
    op = read_rows(os.path.join(open_folder,'aggregated.csv'))
    lam_b = max({_f(r,'lam') for r in bn}); lam_o = max({_f(r,'lam') for r in op})
    XLAB = 'Inner-corner radius $r$ (m)'
    fig, ax = plt.subplots(1, 3, figsize=(10.6, 4.6))   # bottom strip reserved for a shared legend

    # (a) peak contact — bottleneck (green) + open (navy, ~0: free flow can't compress); crush line above.
    xb,cb,eb = _series(bn, lam_b, 'corner_contact_max'); _band(ax[0], xb,cb,eb, BN_COL, 'o', None)
    xo,co,eo = _series(op, lam_o, 'corner_contact_max'); _band(ax[0], xo,co,eo, OPEN_COL, 's', None)
    _thresh(ax[0], CRUSH_REF, 'crush threshold', va='bottom')
    m=float(np.nanmax(cb)); ax[0].set_ylim(-0.03*m, m*1.07)            # no top headroom: keeps the open line visible
    _style(ax[0], XLAB, 'Peak contact force (N kg$^{-1}$)', '(a) Inner-corner\npeak contact force')

    # (b) peak density — bottleneck (green) + open (navy); lethal line, label below.
    xb,db,eb = _series(bn, lam_b, 'corner_rho'); _band(ax[1], xb,db,eb, BN_COL, 'o', None)
    xo,do,eo = _series(op, lam_o, 'corner_rho'); _band(ax[1], xo,do,eo, OPEN_COL, 's', None)
    _thresh(ax[1], LETHAL, 'lethal $\\approx$ 6', va='top')
    ax[1].set_ylim(min(1.9, float(np.nanmin(do))-0.3), LETHAL*1.06)
    _style(ax[1], XLAB, 'Peak density (ped m$^{-2}$)', '(b) Inner-corner\npeak density')

    # (c) door density (bottleneck only) — green; steady + peak labelled INLINE (no legend box).
    xd,yd,ed = _series(bn, lam_b, 'door_rho_steady'); _band(ax[2], xd, yd, ed, DOOR_COL, 'o', None)
    xp,yp,ep = _series(bn, lam_b, 'door_rho_peak')
    ax[2].plot(xp, yp, 's--', color=DOOR_COL, lw=2, ms=6)
    ax[2].fill_between(xp, yp-ep, yp+ep, color=DOOR_COL, alpha=0.10)
    _thresh(ax[2], LETHAL, 'lethal $\\approx$ 6', va='top')
    ax[2].set_ylim(5.5, float(np.nanmax(yp))*1.07)
    j = int(np.argmin(np.abs(xp-1.5)))
    ax[2].annotate('peak', (xp[j], yp[j]), textcoords='offset points', xytext=(0,7),
                   ha='center', va='bottom', color=DOOR_COL, fontsize=LEG, fontweight='bold')
    ax[2].annotate('steady mean', (xd[j], yd[j]), textcoords='offset points', xytext=(0,-7),
                   ha='center', va='top', color=DOOR_COL, fontsize=LEG, fontweight='bold')
    _style(ax[2], XLAB, 'Door density (ped m$^{-2}$)', '(c) Bottleneck\ndoor density')

    # one shared horizontal legend along the bottom (applies to a & b; c is labelled inline)
    handles=[Line2D([0],[0], color=BN_COL,   marker='o', lw=2, ms=6, label='bottleneck'),
             Line2D([0],[0], color=OPEN_COL, marker='s', lw=2, ms=6, label='open corridor')]
    fig.tight_layout(rect=[0, 0.10, 1, 1])
    fig.legend(handles=handles, loc='lower center', ncol=2, fontsize=LEG, frameon=False,
               bbox_to_anchor=(0.5, 0.005))
    TS.save(fig, os.path.join(outdir,'fig_chamfer_lines')); print('  wrote fig_chamfer_lines')


# ── heatmap strip across RADII (saves into outdir) ────────────────────────────
def _crop(g):
    xs=[p[0] for p in g['poly']]; ys=[p[1] for p in g['poly']]
    return min(xs)-0.5, max(xs)+0.5, min(ys)-0.5, max(ys)+0.5


def _walls(g):
    p = g['poly']; return p[:-3], p[-3:]   # inner=[I_in,*arc,O_in]; outer=[O_out,Vout,I_out]


def heat_strip(folder, lam, radii, key, clabel, fname, outdir, suptitle='', open_mode=False):
    grids={}; geoms={}
    for r in radii:
        cands = glob.glob(os.path.join(folder,'grids',f'a{FC.ANGLE}_r{FC.rtag(r)}_lam*.csv'))
        if cands: grids[r]=load_grid(cands[0]); geoms[r]=FC.geometry_rounded(WIDTH,r)
    if not grids: print(f'  no grids in {folder}'); return
    vmax=0.0
    for r,gr in grids.items():
        v=gr[key][np.isfinite(gr[key])]
        if v.size: vmax=max(vmax,float(np.nanpercentile(v,99.5)))
    vmax=max(vmax,1e-6)
    keys=sorted(grids); na=len(keys); tags='abcdefghijkl'
    crops={r:_crop(geoms[r]) for r in keys}
    dw=max(x1-x0 for x0,x1,y0,y1 in crops.values()); dh=max(y1-y0 for x0,x1,y0,y1 in crops.values())
    fig,axs=plt.subplots(1,na,figsize=(STRIP_PANEL_W*na, STRIP_PANEL_W*(dh/dw)+0.9),
                         constrained_layout=True, squeeze=False)
    XX,YY=np.meshgrid(XC,YC); cf=None
    for idx,r in enumerate(keys):
        ax=axs.flat[idx]; g=geoms[r]; gr=grids[r]; x0,x1,y0,y1=crops[r]
        bx=0.5*(x0+x1)-dw/2; by=0.5*(y0+y1)-dh/2
        walk=MPath(g['poly']).contains_points(np.column_stack([XX.ravel(),YY.ravel()])).reshape(NY,NX)
        Z=np.where(walk,gr[key],np.nan)
        cf=ax.contourf(XC-bx,YC-by,np.where(np.isfinite(Z),Z,0.0),levels=np.linspace(0,vmax,25),
                       cmap=TS.CMAP,extend='max')
        clip=PathPatch(MPath([(px-bx,py-by) for px,py in g['poly']]),transform=ax.transData,fc='none',ec='none')
        ax.add_patch(clip)
        try: cf.set_clip_path(clip)
        except AttributeError:
            for c in cf.collections: c.set_clip_path(clip)
        for wpts in _walls(g):
            ax.plot([p[0]-bx for p in wpts],[p[1]-by for p in wpts],'k-',lw=1.8,zorder=5)
        if open_mode:
            ax.plot([g['O_in'][0]-bx,g['O_out'][0]-bx],[g['O_in'][1]-by,g['O_out'][1]-by],
                    '--',color='0.45',lw=1.4,zorder=5)
        else:
            ax.plot([g['O_in'][0]-bx,g['door_hi'][0]-bx],[g['O_in'][1]-by,g['door_hi'][1]-by],'k-',lw=1.8,zorder=5)
            ax.plot([g['O_out'][0]-bx,g['door_lo'][0]-bx],[g['O_out'][1]-by,g['door_lo'][1]-by],'k-',lw=1.8,zorder=5)
            ax.plot([g['door_lo'][0]-bx,g['door_hi'][0]-bx],[g['door_lo'][1]-by,g['door_hi'][1]-by],'lime',lw=2.5,zorder=6)
        vx,vy=g['Vin_sharp']
        ax.add_patch(plt.Rectangle((vx-CORNER_HALF-bx,vy-CORNER_HALF-by),2*CORNER_HALF,2*CORNER_HALF,
                                   fill=False,ec='magenta',lw=1.1,ls='--',zorder=6))
        ax.set(xlim=(0,dw),ylim=(0,dh)); ax.set_aspect('equal')
        ax.set_title(f'({tags[idx]}) $r={r:g}$ m', fontsize=STRIP_TITLE_SIZE, fontweight='bold')
        ax.tick_params(labelsize=STRIP_TICK_SIZE)
        ax.set_xlabel('x (m)', fontsize=STRIP_LABEL_SIZE)
        if idx==0: ax.set_ylabel('y (m)', fontsize=STRIP_LABEL_SIZE)
    cb=fig.colorbar(cf,ax=axs,shrink=0.85,pad=0.02)
    cb.set_label(clabel, fontsize=STRIP_LABEL_SIZE); cb.ax.tick_params(labelsize=STRIP_TICK_SIZE)
    cb.ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    if suptitle: fig.suptitle(suptitle, fontsize=STRIP_SUP_SIZE, fontweight='bold')
    TS.save(fig, os.path.join(outdir,fname)); print(f'  wrote {fname} (vmax={vmax:.2f})')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--bn', required=True, help='bottleneck chamfer run folder')
    ap.add_argument('--open-dir', dest='open_dir', required=True, help='open chamfer run folder')
    ap.add_argument('--outdir', required=True, help='NEW output folder for figures (never overwrites run figures)')
    a = ap.parse_args()
    os.makedirs(a.outdir, exist_ok=True)
    ib = parse_info(a.bn); io = parse_info(a.open_dir)
    lam_b = max(ib['lambdas']); lam_o = max(io['lambdas'])
    print(f'chamfer figures -> {a.outdir}')
    fig_chamfer_lines(a.bn, a.open_dir, a.outdir)
    heat_strip(a.bn, lam_b, ib['radii'], 'contact', r'Contact force (N kg$^{-1}$)',
               'fig_chamfer_contact_bottleneck', a.outdir,
               suptitle='Contact force vs inner-corner radius (bottlenecked corridor)', open_mode=False)
    heat_strip(a.open_dir, lam_o, io['radii'], 'rho', r'Density $\rho$ (ped m$^{-2}$)',
               'fig_chamfer_density_open', a.outdir,
               suptitle='Time-averaged density vs inner-corner radius (open corridor)', open_mode=True)
    heat_strip(a.bn, lam_b, ib['radii'], 'rho', r'Density $\rho$ (ped m$^{-2}$)',
               'fig_chamfer_density_bottleneck', a.outdir,
               suptitle='Time-averaged density vs inner-corner radius (bottlenecked corridor)', open_mode=False)
    print('done.')


if __name__ == '__main__':
    main()
