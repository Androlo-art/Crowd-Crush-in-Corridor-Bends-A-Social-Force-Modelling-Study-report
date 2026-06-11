"""
plot_fis_results.py
===================
Generates the two FIS *results* figures for the thesis (Helbing dashed reference
on every panel). Reads existing raw_results.csv files only — runs no simulation.

Outputs (timestamped, png+pdf+svg) into:
    results/faster_is_slower/fis_results_figs_<YYYYMMDD_HHMMSS>/

  fig1_fis_variant_overlay.*   -- (a) T_evac and (b) casualties vs v0 for the
                                   three injury models (Standard / Soft-radius /
                                   Ghost) overlaid on Helbing 2000.
  fig2_fis_doorwidth.*         -- T_evac vs v0, door calibration 1.05 vs 1.2 m.

Run from the repo root:
    .venv/bin/python -u plot_fis_results.py
"""
import os, csv, datetime
import numpy as np
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.abspath(__file__))
FIS  = os.path.join(ROOT, 'results', 'faster_is_slower')

# ----- Helbing 2000 digitized reference (v0, T_evac, injured) ---------------
hv, hT, hInj = [], [], []
with open(os.path.join(ROOT, 'figure_c_digitized.csv')) as fh:
    r = csv.reader(fh); next(r)
    for row in r:
        hv.append(float(row[0])); hT.append(float(row[1])); hInj.append(float(row[2]))
hv = np.array(hv); hT = np.array(hT); hInj = np.array(hInj)

CENSOR = 400.0
VMIN_PLOT = 0.6            # exclude censored low-speed points from the trend plots

RUNS = {
    'standard'  : 'fis_final_dw1.2_standard_20260602_091219',
    'soft_radius': 'fis_final_dw1.2_soft_radius_20260602_091737',
    'ghost'     : 'fis_final_dw1.2_ghost_20260602_091559',
    'dw1.05'    : 'fis_20260601_185954',
}

def load(folder):
    """Return dict v0 -> (mean_T_all, sd_T_all, mean_injured)."""
    rows = {}
    with open(os.path.join(FIS, folder, 'raw_results.csv')) as fh:
        for d in csv.DictReader(fh):
            v0 = float(d['v0'])
            rows.setdefault(v0, {'T': [], 'inj': []})
            rows[v0]['T'].append(float(d['T_evac']))
            rows[v0]['inj'].append(float(d['N_injured']))
    out = {}
    for v0, x in rows.items():
        T = np.array(x['T']); inj = np.array(x['inj'])
        out[v0] = (T.mean(), T.std(ddof=0), inj.mean())
    return out

def series(folder, vmin=VMIN_PLOT):
    d = load(folder)
    vs = np.array(sorted(v for v in d if v >= vmin))
    mT = np.array([d[v][0] for v in vs])
    sT = np.array([d[v][1] for v in vs])
    mI = np.array([d[v][2] for v in vs])
    return vs, mT, sT, mI

# style per model
STYLE = {
    'standard'   : dict(color='navy',      marker='o', ls='-',  label='Standard (Helbing rule)'),
    'soft_radius': dict(color='darkorange',marker='s', ls='-',  label='Soft-radius ($r_s{=}0.6$, project model)'),
    'ghost'      : dict(color='firebrick', marker='^', ls='-',  label='Ghost (non-interacting casualties)'),
}
HELB = dict(color='k', ls='--', lw=2.2, label='Helbing et al. (2000), $N=200$')

# ---------------------------------------------------------------------------
ts  = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
OUT = os.path.join(FIS, f'fis_results_figs_{ts}')
os.makedirs(OUT, exist_ok=True)

def savefig(fig, name):
    for ext in ('png', 'pdf', 'svg'):
        fig.savefig(os.path.join(OUT, f'{name}.{ext}'),
                    dpi=140 if ext == 'png' else None, bbox_inches='tight')

# ===== Figure 1: variant overlay (T_evac + casualties) =====================
fig1, (axT, axC) = plt.subplots(1, 2, figsize=(13, 5.0), constrained_layout=True)

# Helbing reference, clipped to plotted v0 range
hmask = (hv >= VMIN_PLOT) & (hv <= 8.0)
axT.plot(hv[hmask], hT[hmask],   **HELB)
axC.plot(hv[hmask], hInj[hmask], **HELB)

for key in ('standard', 'soft_radius', 'ghost'):
    vs, mT, sT, mI = series(RUNS[key])
    st = STYLE[key]
    axT.plot(vs, mT, color=st['color'], marker=st['marker'], ls=st['ls'],
             lw=2, ms=5, label=st['label'])
    axC.plot(vs, mI, color=st['color'], marker=st['marker'], ls=st['ls'],
             lw=2, ms=5, label=st['label'])

axT.axhline(CENSOR, color='grey', ls=':', lw=1, zorder=0)
axT.text(0.62, CENSOR-12, '$T_{\\max}=400$ s (cap)', fontsize=8, color='grey', va='top')
axT.set_xlabel('Desired speed $v_0$ (m s$^{-1}$)')
axT.set_ylabel('Leaving time $T_{\\mathrm{evac}}$ (s)')
axT.set_title('(a) Faster-is-slower: leaving time')
axT.set_xlim(0.5, 8.1); axT.grid(True, ls=':', alpha=0.5)
axT.legend(fontsize=8, loc='upper left', framealpha=0.92)

axC.set_xlabel('Desired speed $v_0$ (m s$^{-1}$)')
axC.set_ylabel('Number of casualties')
axC.set_title('(b) Casualty count')
axC.set_xlim(0.5, 8.1); axC.grid(True, ls=':', alpha=0.5)
axC.legend(fontsize=8, loc='upper left', framealpha=0.92)

savefig(fig1, 'fig1_fis_variant_overlay')

# ===== Figure 2: door-width calibration ====================================
fig2, ax = plt.subplots(1, 1, figsize=(6.6, 5.0), constrained_layout=True)
ax.plot(hv[hmask], hT[hmask], **HELB)
for key, col, mk, lab in (('dw1.05', 'tab:red',  'v', '$d_w=1.05$ m (paper text $\\approx$1 m)'),
                          ('standard', 'navy',   'o', '$d_w=1.2$ m (released \\texttt{sd.par})')):
    vs, mT, sT, mI = series(RUNS[key])
    ax.plot(vs, mT, color=col, marker=mk, ls='-', lw=2, ms=5, label=lab)
ax.axhline(CENSOR, color='grey', ls=':', lw=1, zorder=0)
ax.set_xlabel('Desired speed $v_0$ (m s$^{-1}$)')
ax.set_ylabel('Leaving time $T_{\\mathrm{evac}}$ (s)')
ax.set_title('Door-width calibration (standard injury rule)')
ax.set_xlim(0.5, 8.1); ax.grid(True, ls=':', alpha=0.5)
ax.legend(fontsize=8, loc='upper left', framealpha=0.92)
savefig(fig2, 'fig2_fis_doorwidth')

with open(os.path.join(OUT, 'run_info.txt'), 'w') as fh:
    fh.write('FIS results figures (derived from existing raw_results.csv; no new sim runs)\n')
    fh.write(f'timestamp={ts}\n')
    fh.write('Helbing reference: figure_c_digitized.csv (Fig.1 leaving time + injured, N=200)\n')
    for k, v in RUNS.items():
        fh.write(f'{k}: {v}\n')
    fh.write(f'trend plots exclude v0 < {VMIN_PLOT} (censored low-speed runs)\n')

print('wrote figures to', OUT)
