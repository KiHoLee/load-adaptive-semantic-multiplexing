# =========================================================
# rev_dynamic_fig.py — figures for the dynamic-mask framing (2026-11-03).
# Power model: every user transmits at unit average power on each channel use it occupies,
# so a user of Static ODS-k (k fixed blocks of d_s/k) spends d_s/k at every load U <= k and its
# CosSim does not depend on U, while the one-model proposal and Static ODS-U occupy the whole
# block and spend d_s/U. Static ODS-5 and ODS-6 compress every user below d_b (K = 4 blocks of
# d_b); they are drawn dashed with open markers, and the overload region U > K is shaded.
#
#   fig_cl/fig2_dyn_snr.pdf    per-user CosSim vs SNR at every load against Static ODS-k
#   fig_cl/fig3_dyn_load.pdf   per-user CosSim vs load U at several SNRs
#   fig_cl/fig4_dyn_agg.pdf    aggregate fidelity S(U) per block vs load at one SNR
#
# Usage: python rev_dynamic_fig.py [proposal run name] [data_dir]
# Reads stored results only: data/<name>_s*.json and data/ortho_U<k>_s*.json (Static ODS-k at
# its design load; with orthogonal blocks a user's CosSim does not depend on the other users).
# =========================================================
import glob, json, os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
NAME = sys.argv[1] if len(sys.argv) > 1 else "shared_dyns_clamp"
DATA = sys.argv[2] if len(sys.argv) > 2 else os.path.join(ROOT, "data")
FIG = os.path.join(ROOT, "fig_cl")
SNRS = [0, 5, 10, 15, 20, 25, 30]
LOADS = [1, 2, 3, 4, 5, 6]
K = 4
plt.rcParams.update({"font.size": 20, "axes.linewidth": 1.2})
FIGSIZE = (7.5, 5.55)
AX_RECT = [0.155, 0.14, 0.76, 0.7804]       # 6.0 x 4.5 in axes box (8:6), as rev_replot.py
PROP = dict(color="#ff7f0e", lw=3.0, marker="*", ms=15, zorder=4)
KCOL = {1: "#08306b", 2: "#2171b5", 3: "#41ab5d", 4: "#7f7f7f", 5: "#c994c7", 6: "#ce1256"}


def runs(pattern):
    R = [json.load(open(f)) for f in sorted(glob.glob(os.path.join(DATA, pattern)))]
    assert R, pattern
    return R


def proposal(U):
    v = np.array([[R["eval"][str(U)][str(s)]["cos"] for s in SNRS] for R in runs(f"{NAME}_s[0-9].json")])
    return v.mean(0), v.std(0), len(v)


def static(k):
    """Static ODS-k at unit power per occupied channel use: its value at any load U <= k."""
    v = np.array([[R["eval"][str(k)][str(s)]["cos"] for s in SNRS] for R in runs(f"ortho_U{k}_s[0-9].json")])
    return v.mean(0), v.std(0), len(v)


def report():
    print(f"proposal {NAME}: {proposal(1)[2]} seed(s); static ODS-k: {static(1)[2]} seeds")
    for U in LOADS:
        p, sd, _ = proposal(U)
        print(f"U={U} prop " + " ".join(f"{x:.3f}" for x in p) + f"  (seed sd max {sd.max():.3f})")
        for k in range(U, 7):
            print(f"    -ODS{k} " + " ".join(f"{x:+.3f}" for x in p - static(k)[0]))


def fig_load(snrs=(0, 10, 20), path="fig3_dyn_load.pdf"):
    fig = plt.figure(figsize=FIGSIZE)
    ax = fig.add_axes(AX_RECT)
    ax.axvspan(K + 0.5, 6.6, color="0.92", zorder=0)
    scol = {s: c for s, c in zip(snrs, ["#4d4d4d", "#2166ac", "#b2182b"])}
    for s in snrs:
        i = SNRS.index(s)
        for k in range(1, 7):
            y = static(k)[0][i]
            hyp = k > K
            ax.plot([1, k], [y, y], color=scol[s], lw=1.3, ls="--" if hyp else "-", alpha=0.7, zorder=2)
            ax.plot([k], [y], color=scol[s], marker="o", ms=7, mfc="white" if hyp else scol[s], zorder=3)
        ax.plot(LOADS, [proposal(U)[0][i] for U in LOADS], **{**PROP, "color": scol[s]})
        ax.text(6.55, proposal(6)[0][i], f"{s} dB", va="center", ha="left", fontsize=16, color=scol[s])
    ax.set_xlim(0.6, 6.6)
    ax.set_ylim(0.58, 1.005)
    ax.set_xticks(LOADS)
    ax.set_xlabel("Load $U$ (active users)")
    ax.set_ylabel("Per-user CosSim")
    ax.grid(alpha=0.3)
    ax.text(5.55, 0.595, "overload", ha="center", va="bottom", fontsize=15, color="0.35")
    h = [Line2D([0], [0], color="k", lw=3, marker="*", ms=15, label="Proposed (one model)"),
         Line2D([0], [0], color="k", lw=1.3, marker="o", ms=7, label="Static ODS-$k$, $k\\leq K$ (end at $U=k$)"),
         Line2D([0], [0], color="k", lw=1.3, ls="--", marker="o", ms=7, mfc="white",
                label="Static ODS-5, 6 (below $d_b$ per user)")]
    ax.legend(handles=h, loc="lower left", fontsize=14, framealpha=0.9)
    out = os.path.join(FIG, path)
    fig.savefig(out); fig.savefig(out.replace(".pdf", ".png"), dpi=100)
    print("wrote", out)


def fig_snr(loads=(1, 2, 3, 4, 5, 6), path="fig2_dyn_snr.pdf", distortion=False):
    fig = plt.figure(figsize=FIGSIZE)
    ax = fig.add_axes(AX_RECT)
    f = (lambda y: 1 - y) if distortion else (lambda y: y)
    for k in range(1, 7):
        hyp = k > K
        ax.plot(SNRS, f(static(k)[0]), color=KCOL[k], lw=1.3, ls="--" if hyp else "-", marker="o", ms=7,
                mfc="white" if hyp else KCOL[k], zorder=2)
    for U in loads:
        ax.plot(SNRS, f(proposal(U)[0]), **{**PROP, "color": KCOL[U], "lw": 2.2, "ms": 12, "ls": ":"})
    if distortion:
        ax.set_yscale("log")
        ax.set_ylabel("Semantic distortion $1-$CosSim")
    else:
        ax.set_ylabel("Per-user CosSim")
    ax.set_xlabel("SNR (dB)")
    ax.grid(alpha=0.3, which="both")
    h = [Line2D([0], [0], color=KCOL[k], lw=1.3, ls="--" if k > K else "-", marker="o", ms=7,
                mfc="white" if k > K else KCOL[k], label=f"Static ODS-{k}") for k in range(1, 7)]
    h.append(Line2D([0], [0], color="k", lw=2.2, ls=":", marker="*", ms=12, label="Proposed, $U=k$"))
    ax.legend(handles=h, loc="upper right" if distortion else "lower right", fontsize=13, ncol=1, framealpha=0.9)
    out = os.path.join(FIG, path)
    fig.savefig(out); fig.savefig(out.replace(".pdf", ".png"), dpi=100)
    print("wrote", out)


def fig_agg(snr=10, path="fig4_dyn_agg.pdf"):
    """S(U) = sum of the per-user CosSim of the served users; Static ODS-k serves min(U, k) users."""
    i = SNRS.index(snr)
    fig = plt.figure(figsize=FIGSIZE)
    ax = fig.add_axes(AX_RECT)
    ax.axvspan(K + 0.5, 6.6, color="0.92", zorder=0)
    for k in range(1, 7):
        hyp = k > K
        c = static(k)[0][i]
        ax.plot(LOADS, [min(U, k) * c for U in LOADS], color=KCOL[k], lw=1.3, ls="--" if hyp else "-",
                marker="o", ms=7, mfc="white" if hyp else KCOL[k], zorder=2)
    ax.plot(LOADS, [U * proposal(U)[0][i] for U in LOADS], **PROP)
    ax.set_xlim(0.6, 6.6)
    ax.set_xticks(LOADS)
    ax.set_xlabel("Load $U$ (active users)")
    ax.set_ylabel(f"Aggregate fidelity $S(U)$ at {snr} dB")
    ax.grid(alpha=0.3)
    ax.text(5.55, 5.75, "overload", ha="center", va="bottom", fontsize=15, color="0.35")
    h = [Line2D([0], [0], **{**PROP, "label": "Proposed (one model)"})]
    h += [Line2D([0], [0], color=KCOL[k], lw=1.3, ls="--" if k > K else "-", marker="o", ms=7,
                 mfc="white" if k > K else KCOL[k], label=f"Static ODS-{k}") for k in range(1, 7)]
    ax.legend(handles=h, loc="upper left", fontsize=13, framealpha=0.9)
    out = os.path.join(FIG, path)
    fig.savefig(out); fig.savefig(out.replace(".pdf", ".png"), dpi=100)
    print("wrote", out)


if __name__ == "__main__":
    report()
    fig_load()
    fig_snr()
    fig_snr(path="draft_dyn_snr_dist.pdf", distortion=True)   # letter only
    fig_agg()
