"""Thesis-grade figures for the bend study (door BOTTLENECK and OPEN comparison).

Both experiments share this pipeline so the figures are directly comparable:
same geometry, same measurement window, same colours (thesis_style), same crops.

Bottleneck folder (door 1.2 m):
  fig_crush_vs_angle.*           door density (steady, peak), inner-corner density,
                                 casualties vs turn angle, per demand λ (95% CI).
Open folder (egress = full width, detected from the folder name 'openbend'):
  fig_open_vs_angle.*            throughput capacity + inner-corner density vs angle.
Both:
  fig_density_heatmaps_lam<λ>.*  shared-scale density per angle (bend-cropped).
  appendix_pressure_heatmaps_lam<λ>.*  ρ·Var(v) (appendix-only diagnostic).

Standalone replot:
  python -u bend_bottleneck/plot_bottleneck.py <run_folder> [--open]
"""
import os, sys, csv, glob, math, argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.path import Path as MPath
from matplotlib.patches import PathPatch
from matplotlib.ticker import FormatStrFormatter

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
import make_field_bottleneck as F
import thesis_style as TS
TS.apply()

WIDTH = 5.0; RES = 0.25; DOMX, DOMY = 18.0, 32.0   # match make_field_bottleneck.py (LB=12, long outlet)
NX, NY = int(DOMX/RES), int(DOMY/RES)
XC = (np.arange(NX)+0.5)*RES; YC = (np.arange(NY)+0.5)*RES
CORNER_HALF = 1.6
CAP_GREEN = '#2E8B57'   # distinct green for the capacity (max-outflow) line — NOT an angle/0° colour

# ── heat-strip layout knobs (override from the CLI to "play around") ──────────
STRIP_ROWS      = 1     # 1 = all six scenarios side by side in a single row
STRIP_PANEL_W   = 2.3   # inches per panel (figure width = STRIP_PANEL_W * ncols)
STRIP_LABEL_SIZE = 11   # axis-title (x/y label) font size  — matches fig_capacity
STRIP_TITLE_SIZE = 11   # per-panel angle title font size   — matches fig_capacity panel titles
STRIP_SUP_SIZE   = 15   # overall figure title font size     — matches fig_capacity suptitle
STRIP_TICK_SIZE  = 10   # tick-label font size


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
    info = dict(angles=[0,45,60,90,120,135], lambdas=[4.0,8.0])
    p = os.path.join(folder,'run_info.txt')
    if os.path.exists(p):
        for ln in open(p):
            for key in ('angles','lambdas'):
                if ln.startswith(key+'='):
                    try: info[key] = [float(x) for x in ln.split('=')[1].strip().strip('[]').split(',') if x.strip()]
                    except ValueError: pass
    if not glob.glob(os.path.join(folder,'grids','*.csv')):
        info['lambdas'] = info.get('lambdas', [])
    return info


# ── line figure: BOTTLENECK ──────────────────────────────────────────────────
def fig_crush_vs_angle(folder):
    agg = read_rows(os.path.join(folder,'aggregated.csv'))
    if not agg: print('  no aggregated rows'); return
    lams = sorted({_f(r,'lam') for r in agg})
    panels = [('door_rho_steady','Door density, steady mean (ped m$^{-2}$)', True, 1),
              ('door_rho_peak','Door density, peak (ped m$^{-2}$)', True, 1),
              ('corner_rho','Inner-corner density (ped m$^{-2}$)', True, 1),
              ('casualties','Casualties (mean in window)', False, 0)]
    fig, ax = plt.subplots(2, 2, figsize=(TS.FIGW, 0.92*TS.FIGW), constrained_layout=True)
    tags = ['(a)','(b)','(c)','(d)']
    for k,(m,lab,sixline,dec) in enumerate(panels):
        a = ax.flat[k]
        for j,lm in enumerate(lams):
            rs = sorted([r for r in agg if abs(_f(r,'lam')-lm)<1e-6], key=lambda r:_f(r,'angle'))
            xs=np.array([_f(r,'angle') for r in rs]); ys=np.array([_f(r,f'{m}_mean') for r in rs])
            es=np.array([_f(r,f'{m}_ci95') for r in rs]); ok=np.isfinite(ys)
            if ok.any():
                a.plot(xs[ok],ys[ok],'o-',color=TS.lam_color(j,len(lams)),label=f'$\\lambda={lm:.0f}$')
                a.fill_between(xs[ok],ys[ok]-es[ok],ys[ok]+es[ok],color=TS.lam_color(j,len(lams)),alpha=0.15)
        if sixline:
            a.axhline(TS.LETHAL_RHO,color=TS.CRIMSON,ls='--',lw=1.4)
            a.text(135, TS.LETHAL_RHO+0.05, 'lethal $\\approx$ 6', color=TS.CRIMSON,
                   fontsize=11, fontweight='bold', ha='right', va='bottom',
                   bbox=dict(fc='white', ec='none', alpha=0.75, pad=1.0))
        a.set_xlabel('Turn angle (deg)'); a.set_ylabel(lab); a.set_xticks([0,45,60,90,120,135])
        a.grid(alpha=0.3); a.set_title(tags[k],loc='left',fontweight='bold'); TS.yfmt(a,dec)
        if k==0: a.legend(title='Inflow $\\lambda$ (ag s$^{-1}$)')
    TS.save(fig, os.path.join(folder,'fig_crush_vs_angle')); print('  wrote fig_crush_vs_angle')


# ── line figure: OPEN comparison ─────────────────────────────────────────────
def fig_open_vs_angle(folder):
    agg = read_rows(os.path.join(folder,'aggregated.csv'))
    if not agg: print('  no aggregated rows'); return
    lams = sorted({_f(r,'lam') for r in agg}); lam_hi = max(lams)
    rs = sorted([r for r in agg if abs(_f(r,'lam')-lam_hi)<1e-6], key=lambda r:_f(r,'angle'))
    ang = np.array([_f(r,'angle') for r in rs])
    # capacity = saturating throughput (max achieved outflow over demand, per angle)
    cap=[]; corner=[]; corner_e=[]
    for a in ang:
        outs=[_f(r,'inflow_rate_mean') for r in agg if abs(_f(r,'angle')-a)<1e-6]
        cap.append(np.nanmax(outs))
        hit=[r for r in rs if abs(_f(r,'angle')-a)<1e-6][0]
        corner.append(_f(hit,'corner_rho_mean')); corner_e.append(_f(hit,'corner_rho_ci95'))
    cap=np.array(cap); corner=np.array(corner); corner_e=np.array(corner_e)
    fig, ax = plt.subplots(1, 2, figsize=(TS.FIGW, 0.46*TS.FIGW), constrained_layout=True)
    ax[0].plot(ang,cap,'o-',color=TS.NAVY)
    ax[0].set(xlabel='Turn angle (deg)', ylabel='Throughput capacity (ag s$^{-1}$)')
    ax[0].set_title('(a)',loc='left',fontweight='bold')
    ax[1].plot(ang,corner,'o-',color=TS.CRIMSON)
    ax[1].fill_between(ang,corner-corner_e,corner+corner_e,color=TS.CRIMSON,alpha=0.15)
    ax[1].set(xlabel='Turn angle (deg)', ylabel='Inner-corner density (ped m$^{-2}$)')
    ax[1].set_title('(b)',loc='left',fontweight='bold')
    for a in ax: a.set_xticks([0,45,60,90,120,135]); a.grid(alpha=0.3); TS.yfmt(a,1)
    fig.suptitle('Open bend: throughput capacity and inner-corner loading versus turn angle',
                 fontweight='bold', fontsize=15)
    TS.save(fig, os.path.join(folder,'fig_open_vs_angle')); print('  wrote fig_open_vs_angle')


# ── capacity figure (needs a multi-λ sweep; the "fundamental diagram" view) ────
def fig_capacity(folder):
    agg = read_rows(os.path.join(folder,'aggregated.csv'))
    if not agg: print('  no aggregated rows'); return
    angles=sorted({int(_f(r,'angle')) for r in agg}); lams=sorted({_f(r,'lam') for r in agg})
    if len(lams) < 3:
        print(f'  [capacity] only {len(lams)} inflow levels — skip (needs a multi-λ run)'); return
    lo,hi=min(lams),max(lams)
    fig, ax = plt.subplots(1, 3, figsize=(10.2, 3.5), constrained_layout=True)
    # (a) flow that actually gets through vs the rate we feed in -> levels off
    maxflow={}
    for a in angles:
        rs=sorted([r for r in agg if int(_f(r,'angle'))==a], key=lambda r:_f(r,'lam'))
        xs=np.array([_f(r,'lam') for r in rs]); ys=np.array([_f(r,'inflow_rate_mean') for r in rs])
        ax[0].plot(xs,ys,'o-',color=TS.angle_color(a),ms=4,label=f'${a}^\\circ$'); maxflow[a]=np.nanmax(ys)
    ax[0].plot([lo,hi],[lo,hi],'k:',lw=1,label='outflow = inflow')
    ax[0].set(xlabel='Agent inflow rate (ag s$^{-1}$)', ylabel='Agent outflow rate (ag s$^{-1}$)')
    ax[0].set_xticks(lams); ax[0].tick_params(axis='x', labelsize=9)
    ax[0].set_title('(a) Outflow rate at different bend angles',loc='left',fontweight='bold')
    ax[0].legend(fontsize=8, ncol=1, loc='upper left', framealpha=0.85, handlelength=1.4, labelspacing=0.3)
    ax[0].grid(alpha=0.3)
    # (b) maximum agent outflow (= the bend's capacity) vs angle — distinct GREEN
    aa=sorted(maxflow); ax[1].plot(aa,[maxflow[a] for a in aa],'o-',color=CAP_GREEN)
    ax[1].set(xlabel='Turn angle (deg)', ylabel='Maximum agent outflow (ag s$^{-1}$)')
    ax[1].set_xticks(aa); ax[1].set_title('(b) Maximum agent outflow vs bend angle',loc='left',fontweight='bold')
    ax[1].grid(alpha=0.3); TS.yfmt(ax[1],1)
    # (c) inner-corner density vs agent inflow rate -> each angle rises then plateaus (saturates)
    for a in angles:
        rs=sorted([r for r in agg if int(_f(r,'angle'))==a], key=lambda r:_f(r,'lam'))
        xs=np.array([_f(r,'lam') for r in rs]); cs=np.array([_f(r,'corner_rho_mean') for r in rs])
        es=np.array([_f(r,'corner_rho_ci95') for r in rs])
        ok=np.isfinite(cs)
        if ok.any():
            col=TS.angle_color(a)
            eok=np.where(np.isfinite(es[ok]),es[ok],0.0)
            ax[2].fill_between(xs[ok],cs[ok]-eok,cs[ok]+eok,color=col,alpha=0.18,lw=0)  # 95% CI band, line colour
            ax[2].plot(xs[ok],cs[ok],'o-',color=col,ms=4,label=f'${a}^\\circ$')
    ax[2].set(xlabel='Agent inflow rate (ag s$^{-1}$)', ylabel='Inner-corner density (ped m$^{-2}$)')
    ax[2].set_xticks(lams); ax[2].tick_params(axis='x', labelsize=9); TS.yfmt(ax[2],1); ax[2].grid(alpha=0.3)
    ax[2].set_title('(c) Inner-corner density vs inflow rate',loc='left',fontweight='bold')
    ax[2].legend(fontsize=8, ncol=1, loc='lower right', framealpha=0.85, handlelength=1.4, labelspacing=0.3)
    TS.save(fig, os.path.join(folder,'fig_capacity')); print('  wrote fig_capacity')


# ── heatmap strip (shared) ────────────────────────────────────────────────────
def _crop(g):
    xs=[p[0] for p in g['poly']]; ys=[p[1] for p in g['poly']]
    return min(xs)-0.5, max(xs)+0.5, min(ys)-0.5, max(ys)+0.5


def _walls(g):
    """Return (inner_wall_pts, outer_wall_pts) as the SOLID corridor walls,
    excluding the open inlet (bottom) and the outlet cap (handled separately)."""
    p = g['poly']
    if len(p) == 6:                 # bend: I_in,Vin,O_in | O_out,Vout,I_out
        return [p[0],p[1],p[2]], [p[3],p[4],p[5]]
    return [p[0],p[1]], [p[2],p[3]] # straight: right wall | left wall


def heat_strip(folder, lam, angles, key, clabel, fname, suptitle='', open_mode=False):
    grids={}; geoms={}
    for a in angles:
        ai=int(a); p=os.path.join(folder,'grids',f'a{ai}_lam{lam}.csv')
        if os.path.exists(p): grids[ai]=load_grid(p); geoms[ai]=F.geometry(ai,WIDTH)
    if not grids: print(f'  no grids λ={lam}'); return
    vmax=0.0
    for a,gr in grids.items():
        v=gr[key][np.isfinite(gr[key])]
        if v.size: vmax=max(vmax,float(np.nanpercentile(v,99.5)))
    vmax=max(vmax,1e-6)
    keys=sorted(grids); na=len(keys)
    # IDENTICAL-size box for every panel -> equal widths (with aspect 'equal').
    crops={a:_crop(geoms[a]) for a in keys}
    dw=max(x1-x0 for x0,x1,y0,y1 in crops.values())
    dh=max(y1-y0 for x0,x1,y0,y1 in crops.values())
    tags='abcdefghijkl'
    nrow=max(1,STRIP_ROWS); ncol=int(math.ceil(na/nrow))
    fig,axs=plt.subplots(nrow,ncol,figsize=(STRIP_PANEL_W*ncol, STRIP_PANEL_W*(dh/dw)*nrow+0.9),
                         constrained_layout=True, squeeze=False)
    XX,YY=np.meshgrid(XC,YC); cf=None
    for idx,a in enumerate(keys):
        ax=axs.flat[idx]; g=geoms[a]; gr=grids[a]; x0,x1,y0,y1=crops[a]
        bx=0.5*(x0+x1)-dw/2; by=0.5*(y0+y1)-dh/2     # fixed-size box, centred on the corridor
        walk=MPath(g['poly']).contains_points(np.column_stack([XX.ravel(),YY.ravel()])).reshape(NY,NX)
        Z=np.where(walk,gr[key],np.nan)
        cf=ax.contourf(XC-bx,YC-by,np.where(np.isfinite(Z),Z,0.0),levels=np.linspace(0,vmax,25),
                       cmap=TS.CMAP,extend='max')
        clip=PathPatch(MPath([(px-bx,py-by) for px,py in g['poly']]),transform=ax.transData,fc='none',ec='none')
        ax.add_patch(clip)
        try: cf.set_clip_path(clip)
        except AttributeError:
            for c in cf.collections: c.set_clip_path(clip)
        # solid black corridor walls (inner + outer), so walls vs openings are clear
        for wpts in _walls(g):
            ax.plot([p[0]-bx for p in wpts],[p[1]-by for p in wpts],'k-',lw=1.8,zorder=5)
        if open_mode:
            ax.plot([g['O_in'][0]-bx,g['O_out'][0]-bx],[g['O_in'][1]-by,g['O_out'][1]-by],
                    '--',color='0.45',lw=1.4,zorder=5)            # open exit cap
        else:
            ax.plot([g['O_in'][0]-bx,g['door_hi'][0]-bx],[g['O_in'][1]-by,g['door_hi'][1]-by],'k-',lw=1.8,zorder=5)
            ax.plot([g['O_out'][0]-bx,g['door_lo'][0]-bx],[g['O_out'][1]-by,g['door_lo'][1]-by],'k-',lw=1.8,zorder=5)
            ax.plot([g['door_lo'][0]-bx,g['door_hi'][0]-bx],[g['door_lo'][1]-by,g['door_hi'][1]-by],'lime',lw=2.5,zorder=6)
        if g['Vin'] is not None:
            vx,vy=g['Vin']
            ax.add_patch(plt.Rectangle((vx-CORNER_HALF-bx,vy-CORNER_HALF-by),2*CORNER_HALF,2*CORNER_HALF,
                                       fill=False,ec='magenta',lw=1.1,ls='--',zorder=6))
        ax.set(xlim=(0,dw),ylim=(0,dh)); ax.set_aspect('equal')
        ax.set_title(f'({tags[idx]}) ${a}^\\circ$', fontsize=STRIP_TITLE_SIZE, fontweight='bold')
        ax.tick_params(labelsize=STRIP_TICK_SIZE)
        ax.set_xlabel('x (m)', fontsize=STRIP_LABEL_SIZE)
        if idx % ncol == 0: ax.set_ylabel('y (m)', fontsize=STRIP_LABEL_SIZE)
    for j in range(na, nrow*ncol): axs.flat[j].axis('off')
    cb=fig.colorbar(cf,ax=axs,shrink=0.85,pad=0.02)
    cb.set_label(clabel, fontsize=STRIP_LABEL_SIZE); cb.ax.tick_params(labelsize=STRIP_TICK_SIZE)
    cb.ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    if suptitle: fig.suptitle(suptitle, fontsize=STRIP_SUP_SIZE, fontweight='bold')
    TS.save(fig, os.path.join(folder,fname)); print(f'  wrote {fname} (vmax={vmax:.2f})')


def regenerate(folder, open_mode=None):
    if open_mode is None:
        open_mode = 'openbend' in os.path.basename(os.path.normpath(folder)).lower()
    info=parse_info(folder); angles=info['angles']; lams=info['lambdas']
    print(f'{"OPEN" if open_mode else "BOTTLENECK"} figures in {folder}  angles={angles} lambdas={lams}')
    if open_mode:
        fig_open_vs_angle(folder)
        fig_capacity(folder)          # only writes if >=3 demand levels are present
    else:
        fig_crush_vs_angle(folder)
    if lams:
        lam_hi=max(lams)
        rho_sup = ('Time-averaged crowd density at open corridor'
                   if open_mode else 'Time-averaged crowd density at bottlenecked corridor')
        heat_strip(folder,lam_hi,angles,'rho',r'Density $\rho$ (ped m$^{-2}$)',
                   f'fig_density_heatmaps_lam{lam_hi:.0f}', suptitle=rho_sup, open_mode=open_mode)
        heat_strip(folder,lam_hi,angles,'pressure',r'$\rho\,$Var$(v)$ (ped s$^{-2}$)',
                   f'appendix_pressure_heatmaps_lam{lam_hi:.0f}',
                   suptitle='Crowd pressure $\\rho\\,$Var$(v)$ (appendix diagnostic)', open_mode=open_mode)
    print('done.')


if __name__ == '__main__':
    ap=argparse.ArgumentParser()
    ap.add_argument('folder')
    ap.add_argument('--open', dest='open_mode', action='store_true', default=None,
                    help='force open-comparison line figure (else auto-detected from folder name)')
    ap.add_argument('--rows', type=int, default=STRIP_ROWS, help='heat-strip rows (1 = all six side by side)')
    ap.add_argument('--panel-w', type=float, default=STRIP_PANEL_W, help='inches per heat-strip panel')
    ap.add_argument('--label-size', type=float, default=STRIP_LABEL_SIZE, help='axis-title font size')
    ap.add_argument('--title-size', type=float, default=STRIP_TITLE_SIZE, help='per-panel angle title size')
    ap.add_argument('--sup-size', type=float, default=STRIP_SUP_SIZE, help='overall figure title size')
    ap.add_argument('--tick-size', type=float, default=STRIP_TICK_SIZE, help='tick-label size')
    a=ap.parse_args()
    globals().update(STRIP_ROWS=a.rows, STRIP_PANEL_W=a.panel_w, STRIP_LABEL_SIZE=a.label_size,
                     STRIP_TITLE_SIZE=a.title_size, STRIP_SUP_SIZE=a.sup_size, STRIP_TICK_SIZE=a.tick_size)
    regenerate(a.folder, a.open_mode)
