# =========================================================
# pca_fig.py — figures of the closed-form (PCA) version of the letter (2026-11-03), from stored results.
#   fig_cl/fig2_pca_snr.pdf   per-user CosSim vs SNR: Proposed at U = k against learned Static ODS-k
#                             (three-seed means), i.e. the closed-form nested code against learned
#                             partitions at equal channel uses and energy
#   fig_cl/fig3_pca_load.pdf  per-user CosSim vs load at 0, 10, 20 dB: Proposed against the fixed
#                             partitions with the same code (Static ODS-k, PCA), drawn up to U = k
#   fig_cl/fig4_pca_agg.pdf   aggregate fidelity S(U) at 10 dB (letter only)
# Inputs: data/extra/pca_eval.json (Code/pca_eval.py) and data/ortho_U<k>_s*.json. Styles follow
# rev_dynamic_fig.py: the proposal solid with filled stars, Learned ODS-k dash-dot with open circles,
# Static ODS-k (same PCA code) dash-dot with open squares, k > K dashed (below d_b per user), the
# overload U > K shaded. pdf.fonttype 42 embeds TrueType fonts (no Type 3).
# =========================================================
import glob, json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
DATA = os.path.join(ROOT, "data")
FIG = os.path.join(ROOT, "fig_cl")
SNRS = [0, 5, 10, 15, 20, 25, 30]
LOADS = [1, 2, 3, 4, 5, 6]
K = 4
plt.rcParams.update({"font.size": 20, "axes.linewidth": 1.2, "lines.markersize": 11, "mathtext.fontset": "cm",
                     "pdf.fonttype": 42})                     # TrueType, no Type 3 fonts in the PDFs
FIGSIZE = (7.5, 5.55)
AX_RECT = [0.155, 0.14, 0.76, 0.7703]
PROP = dict(color="#ff7f0e", lw=2.2, ls="-", marker="*", zorder=4)
LEG = 16
KCOL = {1: "#08306b", 2: "#2171b5", 3: "#41ab5d", 4: "#7f7f7f", 5: "#c994c7", 6: "#ce1256"}
E = json.load(open(os.path.join(DATA, "extra", "pca_eval.json")))


def ods(k, c):
    """Learned ODS-k: open circles, dash-dot (dashed for k > K)."""
    return dict(color=c, lw=1.3, ls="--" if k > K else "-.", marker="o", mfc="white", mew=1.3)


def static(k, c):
    """Static ODS-k (the same PCA code on k fixed parts): open squares, so it never shares a style with
    Learned ODS-k."""
    return {**ods(k, c), "marker": "s"}


def proposal(U):
    return np.array([E["proposal"]["opt"][str(U)][str(s)] for s in SNRS])


def pca_static(k):
    return np.array([E["static"][str(k)][str(s)] for s in SNRS])


def learned(k):
    R = [json.load(open(f)) for f in sorted(glob.glob(os.path.join(DATA, f"ortho_U{k}_s[0-9].json")))]
    return np.array([[r["eval"][str(k)][str(s)]["cos"] for s in SNRS] for r in R]).mean(0)


def save(fig, name):
    out = os.path.join(FIG, name)
    fig.savefig(out); fig.savefig(out.replace(".pdf", ".png"), dpi=100)
    print("wrote", out)


def fig_snr():
    fig = plt.figure(figsize=FIGSIZE)
    ax = fig.add_axes(AX_RECT)
    for k in LOADS:
        ax.plot(SNRS, learned(k), **ods(k, KCOL[k]), zorder=2)
        ax.plot(SNRS, proposal(k), **{**PROP, "color": KCOL[k]})
    ax.set_xlabel("SNR (dB)"); ax.set_ylabel("Per-user CosSim"); ax.grid(alpha=0.3)
    h = [Line2D([0], [0], **{**PROP, "color": "k"}, label="Proposed, $U=k$")]
    h += [Line2D([0], [0], **ods(k, KCOL[k]), label=f"Learned ODS-{k}") for k in LOADS]
    ax.legend(handles=h, loc="lower right", fontsize=LEG, handlelength=3.0, framealpha=0.9)
    save(fig, "fig2_pca_snr.pdf")


def fig_load(snrs=(0, 10, 20)):
    fig = plt.figure(figsize=FIGSIZE)
    ax = fig.add_axes(AX_RECT)
    ax.axvspan(K + 0.5, 6.6, color="0.92", zorder=0)
    scol = {s: c for s, c in zip(snrs, ["#4d4d4d", "#2166ac", "#b2182b"])}
    for s in snrs:
        i = SNRS.index(s)
        for k in LOADS:
            y = pca_static(k)[i]
            ax.plot([1, k], [y, y], color=scol[s], lw=1.3, ls=ods(k, scol[s])["ls"], alpha=0.7, zorder=2)
            ax.plot([k], [y], ls="none", marker="s", color=scol[s], mfc="white", mew=1.3, zorder=3)
        ax.plot(LOADS, [proposal(U)[i] for U in LOADS], **{**PROP, "color": scol[s]})
        ax.text(6.63, proposal(6)[i], f"{s} dB", va="center", ha="left", fontsize=14, color=scol[s], clip_on=False)
    ax.set_xlim(0.6, 6.6); ax.set_ylim(0.6, 1.005); ax.set_xticks(LOADS)
    ax.set_xlabel("Load $U$ (active users)"); ax.set_ylabel("Per-user CosSim"); ax.grid(alpha=0.3)
    ax.text(5.55, 0.86, "overload", ha="center", va="center", fontsize=15, color="0.35")
    h = [Line2D([0], [0], **{**PROP, "color": "k"}, label="Proposed"),
         Line2D([0], [0], **static(1, "k"), label="Static ODS-$k$, $k\\leq K$"),
         Line2D([0], [0], **static(5, "k"), label="Static ODS-5, 6")]
    ax.legend(handles=h, loc="lower left", fontsize=LEG, handlelength=3.0, framealpha=0.9)
    save(fig, "fig3_pca_load.pdf")


def fig_agg(snr=10):
    i = SNRS.index(snr)
    fig = plt.figure(figsize=FIGSIZE)
    ax = fig.add_axes(AX_RECT)
    ax.axvspan(K + 0.5, 6.6, color="0.92", zorder=0)
    for k in LOADS:
        ax.plot(LOADS, [min(U, k) * pca_static(k)[i] for U in LOADS], **static(k, KCOL[k]), zorder=2)
    ax.plot(LOADS, [U * proposal(U)[i] for U in LOADS], **PROP)
    ax.set_xlim(0.6, 6.6); ax.set_ylim(0.6, 6.4); ax.set_xticks(LOADS)
    ax.set_xlabel("Load $U$ (active users)"); ax.set_ylabel(f"Aggregate fidelity $S(U)$ at {snr} dB"); ax.grid(alpha=0.3)
    ax.text(5.55, 5.75, "overload", ha="center", va="bottom", fontsize=15, color="0.35")
    h = [Line2D([0], [0], **{**PROP, "label": "Proposed"})]
    h += [Line2D([0], [0], **static(k, KCOL[k]), label=f"Static ODS-{k}") for k in LOADS]
    ax.legend(handles=h, loc="upper left", fontsize=LEG, handlelength=3.0, framealpha=0.9)
    save(fig, "fig4_pca_agg.pdf")


if __name__ == "__main__":
    fig_snr(); fig_load(); fig_agg()
