#!/usr/bin/env python3
"""
Regenerate the faster-is-slower benchmark figures from the processed results in
results/01_faster_is_slower/: the 4-panel benchmark, the door-width calibration and the
casualty-rule comparison, each against the digitised Helbing et al. (2000) reference.
Reads the result tables only; runs no simulation. Outputs go to figures/_regenerated/.

Run from code/plotting/report_figures/:
  python plot_fis_report_figures.py
"""
import os, csv, shutil, datetime
import numpy as np
import matplotlib
matplotlib.use("Agg")          # no display; deterministic vector output
import matplotlib.pyplot as plt

# ---- paths -----------------------------------------------------------------
_HERE      = os.path.dirname(os.path.abspath(__file__))
_RELEASE   = os.path.abspath(os.path.join(_HERE, "..", "..", ".."))
DATA_ROOT  = _RELEASE
FIS        = os.path.join(_RELEASE, "results", "01_faster_is_slower")
THESIS_FIG = os.path.join(_RELEASE, "figures", "_regenerated")   # regeneration target (NOT the thesis)
PREVIEW    = os.path.join(THESIS_FIG, "previews")
os.makedirs(THESIS_FIG, exist_ok=True)
os.makedirs(PREVIEW, exist_ok=True)
TS = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

RUNS = {
    "standard"   : "benchmark_standard",
    "soft_radius": "benchmark_soft_radius",
    "ghost"      : "benchmark_non_interacting",
    "dw1.05"     : "door_width_1.05m",
}
CENSOR     = 400.0   # MaxSimTime cap (s)
VMIN_TREND = 0.6     # trend panels exclude censored low-speed runs (matches originals)

# ---- consistent report styling shared by all three figures -----------------
# Figures are authored close to their physical display width (~6.5 in at
# \textwidth, ~4.3 in at 0.62\textwidth) so 11 pt text imports near body size.
plt.rcParams.update({
    "font.size": 11, "axes.titlesize": 11.5, "axes.labelsize": 11,
    "xtick.labelsize": 9.5, "ytick.labelsize": 9.5, "legend.fontsize": 9,
    "lines.linewidth": 1.8, "axes.grid": True, "grid.linestyle": ":",
    "grid.alpha": 0.5, "pdf.fonttype": 42, "savefig.bbox": "tight",
})
XLAB = "Desired speed $v_0$ (m s$^{-1}$)"
TLAB = "Leaving time $T_{\\mathrm{evac}}$ (s)"
HELB = dict(color="k", ls="--", lw=1.8, label="Helbing et al. (2000), $N=200$")


# ---- data loading ----------------------------------------------------------
def load_run(folder):
    """folder -> {v0: {'T':arr, 'inj':arr, 'Jov':arr}} over all seeds."""
    rows = {}
    with open(os.path.join(FIS, folder, "raw_results.csv")) as fh:
        for d in csv.DictReader(fh):
            v0 = float(d["v0"])
            r = rows.setdefault(v0, {"T": [], "inj": [], "Jov": []})
            r["T"].append(float(d["T_evac"]))
            r["inj"].append(float(d["N_injured"]))
            r["Jov"].append(float(d["J_over_v0"]))
    return {v0: {k: np.asarray(v) for k, v in d.items()} for v0, d in rows.items()}


def load_helbing():
    """Digitised Helbing 2000 references; columns indexed by position."""
    c = np.genfromtxt(os.path.join(FIS, "helbing_reference", "figure_c_digitized.csv"), delimiter=",", names=True)
    d = np.genfromtxt(os.path.join(FIS, "helbing_reference", "figure_d_digitized.csv"), delimiter=",", names=True)
    cn, dn = c.dtype.names, d.dtype.names
    return (c[cn[0]], c[cn[1]], c[cn[2]], d[dn[0]], d[dn[1]])   # v,T,inj, v,Jov


HV, HT, HI, HDV, HDJ = load_helbing()
runs = {k: load_run(v) for k, v in RUNS.items()}


def trend(run, vmin=VMIN_TREND):
    """Seed-mean trend, censored low-speed runs excluded (as in the originals)."""
    vs = np.array(sorted(v for v in run if v >= vmin))
    return (vs,
            np.array([run[v]["T"].mean()   for v in vs]),
            np.array([run[v]["inj"].mean() for v in vs]))


def full_stats(run):
    """Per-speed mean and std (ddof=0) over the full v0 range."""
    vs = np.array(sorted(run))

    def ms(k):
        return (np.array([run[v][k].mean()      for v in vs]),
                np.array([run[v][k].std(ddof=0) for v in vs]))

    return vs, ms("T"), ms("inj"), ms("Jov")


def scatter_pts(ax, run, key, color, vs):
    """Faint individual-trial markers behind the seed-mean line."""
    for v in vs:
        y = run[v][key]
        ax.scatter(np.full_like(y, v), y, s=10, color=color, alpha=0.18,
                   edgecolors="none", zorder=1)


def save(fig, name):
    """Back up the existing thesis PDF, write the new PDF there, write a PNG preview."""
    pdf = os.path.join(THESIS_FIG, name + ".pdf")
    if os.path.exists(pdf):
        bak = os.path.join(THESIS_FIG, "%s_backup_%s.pdf" % (name, TS))
        shutil.copy2(pdf, bak)
        print("  backed up %-22s -> %s" % (name + ".pdf", os.path.basename(bak)))
    fig.savefig(pdf)                                          # vector PDF -> thesis folder
    fig.savefig(os.path.join(PREVIEW, name + ".png"), dpi=200)  # PNG preview -> repo folder
    plt.close(fig)
    print("  wrote %s" % pdf)


# ===== Figure 1: variant overlay (T_evac + casualties) -- \textwidth ========
# Both panels share the same four series, so a single legend below the panels
# keeps the data in both panels fully unobstructed.
fig, (axT, axC) = plt.subplots(1, 2, figsize=(7.0, 3.4), constrained_layout=True)
hm = (HV >= VMIN_TREND) & (HV <= 8.0)
axT.plot(HV[hm], HT[hm], **HELB)
axC.plot(HV[hm], HI[hm], color="k", ls="--", lw=1.8)
STYLE = {
    "standard"   : dict(color="navy",       marker="o", label="Standard (full-obstruction)"),
    "soft_radius": dict(color="darkorange", marker="s", label="Reduced-radius ($r_s{=}0.6$)"),
    "ghost"      : dict(color="firebrick",  marker="^", label="Non-interacting casualties"),
}
for k in ("standard", "soft_radius", "ghost"):
    vs, mT, mI = trend(runs[k])
    s = STYLE[k]
    axT.plot(vs, mT, color=s["color"], marker=s["marker"], ms=4.5, label=s["label"])
    axC.plot(vs, mI, color=s["color"], marker=s["marker"], ms=4.5)
axT.set(xlabel=XLAB, ylabel=TLAB, title="(a) Leaving time", xlim=(0.5, 8.1))
axC.set(xlabel=XLAB, ylabel="Number of casualties", title="(b) Casualty count", xlim=(0.5, 8.1))
handles, labels = axT.get_legend_handles_labels()
fig.legend(handles, labels, loc="outside lower center", ncol=2, framealpha=0.92)
save(fig, "fis_variant_overlay")

# ===== Figure 2: door-width calibration -- 0.62\textwidth ===================
fig, ax = plt.subplots(figsize=(4.3, 3.3), constrained_layout=True)
ax.plot(HV[hm], HT[hm], **HELB)
for k, col, mk, lab in (("dw1.05",   "tab:red", "v", "$d_w=1.05$ m (paper text $\\approx$1 m)"),
                        ("standard", "navy",    "o", "$d_w=1.2$ m (adopted)")):
    vs, mT, _ = trend(runs[k])
    ax.plot(vs, mT, color=col, marker=mk, ms=4.5, label=lab)
ax.axhline(CENSOR, color="grey", ls=":", lw=1.1, zorder=0)
ax.text(7.9, CENSOR - 6, "$T_{\\max}=400$ s (cap)", fontsize=8, color="grey", va="top", ha="right")
ax.set(xlabel=XLAB, ylabel=TLAB, xlim=(0.5, 8.1))
ax.legend(loc="upper left", framealpha=0.92)
save(fig, "fis_doorwidth")

# ===== Figure 3: standard 4-panel -- 2x2, \textwidth =======================
std = runs["standard"]
vs, (mT, sT), (mI, sI), (mJ, sJ) = full_stats(std)
fig, ((aT, aJ), (aS, aC)) = plt.subplots(2, 2, figsize=(7.0, 5.8), constrained_layout=True)
hmA = (HV  >= 0.2) & (HV  <= 8.0)
hmD = (HDV >= 0.2) & (HDV <= 8.0)

# (a) leaving time
scatter_pts(aT, std, "T", "navy", vs)
aT.fill_between(vs, mT - sT, mT + sT, color="navy", alpha=0.15, zorder=2)
aT.plot(vs, mT, color="navy", marker="o", ms=3.5, zorder=3, label="Seed mean")
aT.plot(HV[hmA], HT[hmA], **HELB)
aT.axhline(CENSOR, color="grey", ls=":", lw=1.1, zorder=0)
aT.set(ylabel=TLAB, title="(a) Leaving time")
aT.legend(loc="upper right", framealpha=0.92)

# (b) flow efficiency J/v0  -- Helbing flow curve from figure_d (kept; see caption)
scatter_pts(aJ, std, "Jov", "seagreen", vs)
aJ.fill_between(vs, mJ - sJ, mJ + sJ, color="seagreen", alpha=0.15, zorder=2)
aJ.plot(vs, mJ, color="seagreen", marker="o", ms=3.5, zorder=3, label="Seed mean")
aJ.plot(HDV[hmD], HDJ[hmD], **HELB)
aJ.set(ylabel="Flow efficiency $J/v_0$", title="(b) Flow efficiency")
aJ.legend(loc="upper right", framealpha=0.92)

# (c) trial-to-trial variability sigma(T_evac)
aS.plot(vs, sT, color="crimson", marker="s", ms=3.5, label="$\\sigma$ across seeds")
aS.set(ylabel="$\\sigma(T_{\\mathrm{evac}})$ (s)", title="(c) Trial-to-trial variability")
aS.legend(loc="upper left", framealpha=0.92)

# (d) casualties
scatter_pts(aC, std, "inj", "crimson", vs)
aC.fill_between(vs, np.clip(mI - sI, 0, None), mI + sI, color="crimson", alpha=0.15, zorder=2)
aC.plot(vs, mI, color="crimson", marker="o", ms=3.5, zorder=3, label="Seed mean")
aC.plot(HV[hmA], HI[hmA], **HELB)
aC.set(ylabel="Number of casualties", title="(d) Casualties")
aC.legend(loc="upper left", framealpha=0.92)

for a in (aT, aJ, aS, aC):
    a.set_xlabel(XLAB)
    a.set_xlim(0.0, 8.2)
save(fig, "fis_standard_4panel")

print("done. PDFs (+ timestamped backups) in:\n  %s\nPNG previews in:\n  %s" % (THESIS_FIG, PREVIEW))
