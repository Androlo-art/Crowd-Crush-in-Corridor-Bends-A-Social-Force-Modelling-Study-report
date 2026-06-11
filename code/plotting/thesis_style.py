"""Shared thesis-grade figure styling for the bend study (open + bottleneck).

Goal: figures whose text matches the report body (11 pt, Times-like serif) when
included at width=\\textwidth on the A4/2.5 cm-margin layout (text width = 16 cm =
6.3 in), with a single fixed colour mapping so a given turn angle / demand is the
SAME colour in every figure of both experiments.

Usage:
    import thesis_style as TS
    TS.apply()
    fig, ax = plt.subplots(figsize=(TS.FIGW, 0.62*TS.FIGW), constrained_layout=True)
    ...
    TS.yfmt(ax, 1)          # 1-decimal y ticks (consistent sig figs)
    TS.save(fig, path)      # writes .pdf + .png + .svg
"""
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter

# Final on-page width (16 cm text block) and a comfortable half width.
FIGW = 6.3
FIGW_HALF = 3.05

# Fixed angle order used in BOTH experiments -> identical colours per angle.
ANGLES = [0, 45, 60, 90, 120, 135]
NAVY = '#1F3A60'
CRIMSON = '#B22222'
CMAP = 'turbo'                       # density / contact heatmaps (shared)
LETHAL_RHO = 6.0                     # ped m^-2 reference line


def angle_color(a):
    """Stable colour for a turn angle, consistent across every figure."""
    i = ANGLES.index(int(round(a))) if int(round(a)) in ANGLES else 0
    return plt.cm.plasma(np.linspace(0.05, 0.85, len(ANGLES)))[i]


def lam_color(j, n):
    """Stable colour for the j-th demand level (of n)."""
    return plt.cm.viridis(np.linspace(0.15, 0.78, max(n, 1)))[j]


def apply():
    plt.rcParams.update({
        'font.family': 'serif',
        'font.serif': ['Times New Roman', 'Times', 'DejaVu Serif'],
        'mathtext.fontset': 'stix',
        'font.size': 11,             # = report body text
        'axes.titlesize': 11,
        'axes.labelsize': 11,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'legend.fontsize': 9.5,
        'legend.title_fontsize': 9.5,
        'axes.linewidth': 0.8,
        'lines.linewidth': 1.8,
        'lines.markersize': 5,
        'figure.dpi': 140,
        'savefig.dpi': 140,
        'svg.fonttype': 'none',
    })


def yfmt(ax, dec=1):
    ax.yaxis.set_major_formatter(FormatStrFormatter(f'%.{dec}f'))


def xfmt(ax, dec=0):
    ax.xaxis.set_major_formatter(FormatStrFormatter(f'%.{dec}f'))


def save(fig, path_noext):
    for ext in ('pdf', 'png', 'svg'):
        fig.savefig(f'{path_noext}.{ext}', bbox_inches='tight')
    plt.close(fig)
