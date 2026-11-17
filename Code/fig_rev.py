# =========================================================
# fig_rev.py — result figures of the general-mask revision (2026-11-17), from stored results only.
#   fig_cl/fig2_load_rev.pdf  per-user CosSim versus the load U = 1..8 (light load, full load U = K = 4, overload)
#                             at 0, 10 and 20 dB: Proposed (load masks, one transceiver for every load), Static ODS-k
#                             (the same code on k fixed parts, drawn flat up to its design load k, open squares at k)
#                             and Learned ODS-U (the learned partition trained for the load U, open circles; three-seed
#                             means for U <= 6, one run at U = 8, none at U = 7). AG News, independent Rayleigh fading.
#   fig_cl/fig3_overload_coh.pdf  two stacked panels:
#     (a) observers of one scene from light load through overload (COCO caption mixtures, K = 4, independent Rayleigh
#         fading, known gains): NMSE reduction of the overlapping masks of Proposition 2 against the load masks, both
#         with the joint LMMSE receiver, versus U = 2..8. Lines: Gaussian model with the fitted statistics (exact NMSE,
#         100 gain draws, shared_theory.py) at kappa = 0.8 (solid) and 0.4 (dashed). Markers: real caption embeddings
#         at kappa = 0.8 (shared_masks.py, 2000 frames) with paired bootstrap 95% intervals (1000 resamples of the
#         frames, default_rng(555 + SNR)); open marker: kappa = 0 (independent captions) at U = 8.
#     (b) channel coherence (panel_coh()).
#   fig_cl/fig3_overload.pdf  panel (a) alone (fig_overload()).
# 2026-11-17: save() runs a layout guard (no curve or marker under a legend, no text off the canvas;
#   reference and boundary lines carry gid "ref" and are exempt); Fig. 3(a) legend in one column in the empty upper
#   left (it covered the U = 7, 20 dB caption marker), caption markers staggered by SNR (DODGE), markers named
#   "Mixtures" (caption mixtures), region labels 18 pt on the canvas ("full load"), SNR tags 16 pt.
# Inputs: data/extra/pca_eval.json, data/extra/pca_eval_load.json (U = 7), data/ortho_U<k>_s*.json,
#         data/shared/theory_coco_mix<k>_K4_U<U>.json, data/shared/shared_coco_mix<k>_K4_U<U>[_ol].json/.npz.
# Drawn at 7.5 in width for print at 0.85 columnwidth (fonts about 8 pt in print). pdf.fonttype 42 (no Type 3).
# The manuscript's Fig. 2 and Fig. 3 are one panel each, fig_cl/fig2_load_one.pdf (fig_load_one(), nines axis;
#   the linear version is fig_cl/fig2_alt.pdf) and fig_cl/fig3_scene_one.pdf (fig_scene_one()). The column figures
#   fig_cl/fig2_load_col.pdf and fig_cl/fig3_scene_col.pdf (fig_load_col(), fig_scene_col(), drawn at print size
#   under the guard col_guard()) are figures of the response letter; the script also draws the earlier versions
#   above (fig2_load_rev.pdf, fig3_overload_coh.pdf, fig3_overload.pdf), which neither document includes.
# =========================================================
import glob, json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.legend
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
DATA = os.path.join(ROOT, "data")
FIG = os.path.join(ROOT, "fig_cl")
K = 4
LOADS = [1, 2, 3, 4, 5, 6, 7, 8]
SNR3 = (0, 10, 20)
DODGE = {0: -0.14, 10: 0.14, 20: 0.0}                         # Fig. 3(a): caption markers staggered by SNR (no overlap at U = 8)
SCOL = {0: "#4d4d4d", 10: "#2166ac", 20: "#b2182b"}            # one color per SNR in both figures
plt.rcParams.update({"font.size": 20, "axes.linewidth": 1.2, "lines.markersize": 10, "mathtext.fontset": "cm",
                     "pdf.fonttype": 42})
LEG = 16
PROP = dict(lw=2.2, ls="-", marker="*", markersize=13, zorder=5)
E = json.load(open(os.path.join(DATA, "extra", "pca_eval.json")))
EL = json.load(open(os.path.join(DATA, "extra", "pca_eval_load.json")))


def proposal(U, s):
    if U == 7:
        return EL["proposal_opt"]["7"][str(s)]
    return E["proposal"]["opt"][str(U)][str(s)]


def static(k, s):
    if k == 7:
        return EL["static"]["7"][str(s)]
    return E["static"][str(k)][str(s)]


def learned(k, s):
    R = [json.load(open(f)) for f in sorted(glob.glob(os.path.join(DATA, f"ortho_U{k}_s[0-9].json")))]
    return float(np.mean([r["eval"][str(k)][str(s)]["cos"] for r in R])) if R else None


def regions(ax, x0, x1, fs=18):
    """light load U < K, full load U = K (dotted line), overload U > K (gray); labels above the axes."""
    ax.axvspan(K + 0.5, x1, color="0.92", zorder=0)
    ax.axvline(K, color="0.55", lw=1.0, ls=":", zorder=1, gid="ref")
    tr = ax.get_xaxis_transform()
    ax.text((x0 + K - 0.5) / 2, 1.025, "light load", transform=tr, ha="center", va="bottom", fontsize=fs, color="0.25")
    ax.text(K, 1.025, "full load", transform=tr, ha="center", va="bottom", fontsize=fs, color="0.25")
    ax.text((K + 0.5 + x1) / 2, 1.025, "overload", transform=tr, ha="center", va="bottom", fontsize=fs, color="0.25")


def guard(fig):
    """no curve point or marker inside a legend box (inflated by the marker radius), no text
    outside the canvas. Lines with gid 'ref' (reference and boundary lines) are exempt."""
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    W, H = fig.bbox.width, fig.bbox.height
    for ax in fig.axes:
        legs = [a for a in ax.get_children() if isinstance(a, matplotlib.legend.Legend)]
        for t in ax.texts + [ax.xaxis.label, ax.yaxis.label]:
            e = t.get_window_extent(r)
            assert e.x0 >= 0 and e.y0 >= 0 and e.x1 <= W and e.y1 <= H, ("text leaves the canvas", t.get_text())
        for lg in legs:
            e = lg.get_window_extent(r)
            assert e.x0 >= 0 and e.y0 >= 0 and e.x1 <= W and e.y1 <= H, "legend leaves the canvas"
            for ln in ax.lines:
                if ln.get_gid() == "ref":
                    continue
                xy = ax.transData.transform(np.column_stack(ln.get_data()).astype(float))
                pad = 0.5 * ln.get_markersize() * fig.dpi / 72.0 + ln.get_linewidth() * fig.dpi / 72.0
                pts = [xy]
                if ln.get_linestyle() not in ("None", "none", "", " ") and len(xy) > 1:
                    tt = np.linspace(0, 1, 60)[:, None]
                    pts += [a + tt * (b - a) for a, b in zip(xy[:-1], xy[1:])]
                pts = np.vstack(pts)
                inside = ((pts[:, 0] > e.x0 - pad) & (pts[:, 0] < e.x1 + pad) & (pts[:, 1] > e.y0 - pad) & (pts[:, 1] < e.y1 + pad))
                assert not inside.any(), ("curve or marker under the legend", ln.get_label(), ln.get_color())


def save(fig, name, strict=True):
    """strict: the guard raises (manuscript figures); otherwise it prints a warning (letter-only figure)."""
    try:
        guard(fig)
    except AssertionError as e:
        if strict:
            raise
        print("WARNING (letter figure", name + "):", e)
    out = os.path.join(FIG, name)
    fig.savefig(out, metadata={"CreationDate": None}); fig.savefig(out.replace(".pdf", ".png"), dpi=100)
    print("wrote", out)


def fig_load():
    H = 3.4          # same axes box as Fig. 3
    fig = plt.figure(figsize=(7.5, H))
    ax = fig.add_axes([0.13, 0.777 / H, 0.75, (H - 0.777 - 0.42) / H])
    regions(ax, 0.6, 8.6)
    for s in SNR3:
        c = SCOL[s]
        for k in LOADS:
            y = static(k, s)
            ax.plot([1, k], [y, y], color=c, lw=1.2, ls="--" if k > K else "-.", alpha=0.65, zorder=2)
            ax.plot([k], [y], ls="none", marker="s", color=c, mfc="white", mew=1.3, zorder=3)
        L = [(U, learned(U, s)) for U in LOADS]
        L = [(U, v) for U, v in L if v is not None]
        ax.plot([u for u, _ in L], [v for _, v in L], ls="none", marker="o", color=c, mfc="white", mew=1.6,
                markersize=8, zorder=4)
        ax.plot(LOADS, [proposal(U, s) for U in LOADS], color=c, **PROP)
        ax.text(8.68, proposal(8, s), f"{s} dB", va="center", ha="left", fontsize=16, color=c, clip_on=False)
    ax.set_xlim(0.6, 8.6); ax.set_ylim(0.47, 1.005); ax.set_xticks(LOADS)    # 0.53 before the 3.4 in canvas (no other free region for the legend)
    ax.set_xlabel("Load $U$ (active users)"); ax.set_ylabel("Per-user CosSim"); ax.grid(alpha=0.3)
    h = [Line2D([0], [0], color="k", **PROP, label="Proposed"),
         Line2D([0], [0], color="k", lw=1.2, ls="-.", marker="s", mfc="white", mew=1.3, label="Static ODS-$k$, $k\\leq K$"),
         Line2D([0], [0], color="k", lw=1.2, ls="--", marker="s", mfc="white", mew=1.3, label="Static ODS-$k$, $k>K$"),
         Line2D([0], [0], color="k", ls="none", marker="o", mfc="white", mew=1.6, markersize=8, label="Learned ODS-$U$")]
    ax.legend(handles=h, loc="lower center", ncol=2, fontsize=LEG - 2, handlelength=2.4, columnspacing=1.0,
              framealpha=0.95, bbox_to_anchor=(0.5, 0.0))
    save(fig, "fig2_load_rev.pdf")


# ----------------------------------------------------------------- Fig. 3(a): one scene, light load to overload
def theory_red(mix, U, s):
    f = os.path.join(DATA, "shared", f"theory_coco_mix{mix:g}_K4_U{U}.json")
    if not os.path.exists(f):
        return None
    t = json.load(open(f))
    snrs = [float(x) for x in t["meta"]["snrs"]]
    if float(s) not in snrs:
        return None
    i = snrs.index(float(s))
    R = t["results"]["full"]["rayleigh"]
    return 100.0 * (1.0 - R["gen_joint"][i]["nmse"] / R["load_joint"][i]["nmse"])


def real_red(mix, U, s, nb=1000):
    """NMSE reduction (%) of gen_joint against load_joint on the same frames, with a paired bootstrap 95% CI."""
    base = os.path.join(DATA, "shared", f"shared_coco_mix{mix:g}_K4_U{U}")
    for suf in ("_ol", ""):
        if os.path.exists(base + suf + ".npz"):
            z = np.load(base + suf + ".npz")
            break
    else:
        return None
    eg, el = z[f"err_gen_joint_{s}"].sum(1), z[f"err_load_joint_{s}"].sum(1)
    r = 100.0 * (1.0 - eg.sum() / el.sum())
    rng = np.random.default_rng(555 + int(s))
    n = len(eg)
    bs = []
    for _ in range(nb):
        i = rng.integers(0, n, n)
        bs.append(100.0 * (1.0 - eg[i].sum() / el[i].sum()))
    return r, np.percentile(bs, 2.5), np.percentile(bs, 97.5)


def panel_overload(ax):
    UU = [2, 3, 4, 5, 6, 7, 8]
    regions(ax, 1.6, 8.6, fs=14)
    for s in SNR3:
        c = SCOL[s]
        for mix, ls in ((0.8, "-"), (0.4, "--")):
            pts = [(U, theory_red(mix, U, s)) for U in UU]
            pts = [(U, v) for U, v in pts if v is not None]
            ax.plot([u for u, _ in pts], [v for _, v in pts], color=c, lw=1.8, ls=ls, zorder=3)
        pts = [(U, real_red(0.8, U, s)) for U in UU]
        pts = [(U, v) for U, v in pts if v is not None]
        if pts:
            x = np.array([u for u, _ in pts]); m = np.array([v[0] for _, v in pts])
            lo = np.array([v[1] for _, v in pts]); hi = np.array([v[2] for _, v in pts])
            ax.errorbar(x + DODGE[s], m, yerr=[m - lo, hi - m], ls="none", marker="o", color=c, mfc=c, ms=8, capsize=3,
                        elinewidth=1.3, zorder=5)
        z0 = real_red(0.0, 8, s)
        if z0 is not None:
            ax.plot([8 + DODGE[s]], [z0[0]], ls="none", marker="D", color=c, mfc="white", mew=1.5, ms=8, zorder=5)
        yl = theory_red(0.8, 8, s)
        if yl is not None:
            ax.text(8.68, yl + {0: 2.6, 10: -2.6, 20: 0.0}[s], f"{s} dB", va="center", ha="left", fontsize=16, color=c,
                    clip_on=False)
    ax.axhline(0, color="0.3", lw=0.9, zorder=1, gid="ref")
    ax.set_xlim(1.6, 8.6); ax.set_ylim(-3, 73); ax.set_xticks(UU); ax.set_yticks([0, 20, 40, 60])     # 58 before 2026-11-17 (legend headroom)
    ax.set_xlabel("Load $U$ (active users)"); ax.set_ylabel("NMSE reduction (%)"); ax.grid(alpha=0.3)
    h = [Line2D([0], [0], color="k", lw=1.8, ls="-", label="Model, $\\bar\\kappa=0.8$"),
         Line2D([0], [0], color="k", lw=1.8, ls="--", label="Model, $\\bar\\kappa=0.4$"),
         Line2D([0], [0], color="k", ls="none", marker="o", ms=8, label="Mixtures, $\\bar\\kappa=0.8$"),
         Line2D([0], [0], color="k", ls="none", marker="D", mfc="white", mew=1.5, ms=8, label="Mixtures, $\\bar\\kappa\\approx0$")]
    ax.legend(handles=h, loc="upper left", fontsize=LEG - 2, handlelength=2.2, framealpha=0.95, ncol=1,
              labelspacing=0.25, borderpad=0.35, bbox_to_anchor=(0.0, 1.0))  # one column in the empty upper left (2026-11-17 fix)


def fig_overload_only():
    H = 4.0
    fig = plt.figure(figsize=(7.5, H))
    ax = fig.add_axes([0.13, 0.777 / H, 0.75, (H - 0.777 - 0.42) / H])
    panel_overload(ax)
    save(fig, "draft_fig3a_overload.pdf")


# ----------------------------------------------------------------- Fig. 3(b): coherence bandwidth and time
ETAS = ["1.0", "0.9", "0.7", "0.5", "0.3", "0.0"]
UCOL = {1: "#1b9e77", 2: "#7570b3", 4: "#d95f02", 6: "#e7298a"}


def coh(rho, eta, U, mode, snr=0, nf="_nf6"):
    f = os.path.join(DATA, "pilot_corr", f"pca_csi_coh_rho{rho}_eta{eta}_U{U}{nf}.json")
    if not os.path.exists(f):
        return None
    r = json.load(open(f))["res"]
    return r[mode][str(snr)]["cos"] - r["fixed"][str(snr)]["cos"]


def panel_coh(ax, snr=0):
    x = [float(e) for e in ETAS]
    for U, c in UCOL.items():
        lo = [coh("0.3", e, U, "csiq", snr) for e in ETAS]
        hi = [coh("0.95", e, U, "csiq", snr) for e in ETAS]
        if None not in lo and None not in hi:
            ax.fill_between(x, lo, hi, color=c, alpha=0.18, lw=0, zorder=2)
        y = [coh("0.7", e, U, "csiq", snr) for e in ETAS]
        ax.plot(x, y, color=c, lw=2.0, ls="-", zorder=3)
        ys = [coh("0.7", e, U, "switch", snr) for e in ETAS]
        ax.plot(x, ys, ls="none", marker="o", color=c, mfc="white", mew=1.6, ms=8, zorder=4)
    ax.axhline(0, color="0.3", lw=0.9, zorder=1)
    ax.set_xlim(1.03, -0.03)
    ax.set_xticks([1.0, 0.9, 0.7, 0.5, 0.3, 0.0]); ax.set_xticklabels(["1", "0.9", "0.7", "0.5", "0.3", "0"])
    ax.set_ylim(-0.06, 0.175)
    ax.set_xlabel("Gain correlation $\\eta$ after one block"); ax.set_ylabel("CosSim gain"); ax.grid(alpha=0.3)
    h = [Line2D([0], [0], color="k", lw=2.0, label="Channel-aware, $\\rho_f\\!=\\!0.7$"),
         Patch(facecolor="0.6", alpha=0.35, label="$\\rho_f$ from 0.3 to 0.95"),
         Line2D([0], [0], color="k", ls="none", marker="o", mfc="white", mew=1.6, ms=8, label="Coherence-aware choice")]
    l1 = ax.legend(handles=h, loc="upper right", fontsize=LEG - 2, handlelength=2.0, framealpha=0.95)
    ax.add_artist(l1)
    hu = [Line2D([0], [0], color=c, lw=2.4, label=f"$U\\!=\\!{U}$") for U, c in UCOL.items()]
    ax.legend(handles=hu, loc="lower left", ncol=2, fontsize=LEG - 2, handlelength=1.6, columnspacing=0.9,
              framealpha=0.95)


def fig_overload_coh():
    """two stacked panels; inches: x label 0.777 under each axes, 0.42 above (a) for the load regions."""
    ha, hb, xl, tl = 2.95, 2.95, 0.777, 0.42
    H = xl + hb + xl + ha + tl
    fig = plt.figure(figsize=(7.5, H))
    top = fig.add_axes([0.18, (xl + hb + xl) / H, 0.70, ha / H])
    bot = fig.add_axes([0.18, xl / H, 0.70, hb / H])
    panel_overload(top)
    panel_coh(bot)
    fig.align_ylabels([top, bot])
    for ax, lab in ((top, "(a)"), (bot, "(b)")):
        ax.text(-0.235, 1.03, lab, transform=ax.transAxes, fontsize=19, va="bottom", ha="left")
    save(fig, "fig3_overload_coh.pdf", strict=False)     # letter only; its panel (b) legends are revised there

def fig_overload():
    """panel (a) of fig3_overload_coh.pdf alone, one scene from light load through overload; the coherence panel
    (b) stays in fig3_overload_coh.pdf."""
    H = 3.4          # same axes box as Fig. 2
    fig = plt.figure(figsize=(7.5, H))
    ax = fig.add_axes([0.13, 0.777 / H, 0.75, (H - 0.777 - 0.42) / H])
    panel_overload(ax)
    save(fig, "fig3_overload.pdf")


# =================================================================================================================
# Column figures (2026-11-17). Native print size: the canvas width is \columnwidth (252 pt = 3.487 in) and the manuscript includes both PDFs
# at width=\columnwidth, so canvas pt = printed pt. Two figures, two panels each, one 8:6 axes box for every panel, one
# canvas for both figures, one framed 6.5 pt legend band above the panels (legends hang from the top edge, the first
# flush left, the last flush right), ticks and axis labels 8 pt, region labels and tags 7 pt, no sub- or superscripts
# in any figure text.
#   fig_cl/fig2_load_col.pdf   (a) per-user CosSim versus the load U = 1..8 at 0, 10, 20 dB (Proposed, Static ODS-k
#                              ending at its design load k, Learned ODS-U); (b) margin of Proposed over Learned ODS-U
#                              (circles) and over Static ODS-U (squares), markers staggered by SNR around each load.
#   fig_cl/fig3_scene_col.pdf  (a) CosSim gain over the load masks at 0 dB versus the load U = 1, 2, 4, 6 of the
#                              channel-aware masks (lines rho_f = 0.7; bands from the least to the largest gain over
#                              every tested rho_f from 0.3 to 0.95) and of the coherence-aware choice (triangles), line
#                              style per gain correlation eta = 1, 0.5, 0 with the eta written at the right end;
#                              (b) overlap NMSE reduction versus the load (the data of fig3_overload.pdf), SNR key
#                              inside the light-load region. The paired bootstrap 95% intervals of the mixtures are
#                              drawn as error bars; at print size they are at most 0.6 pt long on either side, inside
#                              the 2 pt marker radius, so the text quotes the interval it needs.
# Font: Arial; CL_COL_FONT="DejaVu Sans" gives the default font, which then needs a lower left end of the load axes for
# the 7 pt light-load labels (the guard fails otherwise). Run: python3 fig_rev.py (from Code/).
# =================================================================================================================
import matplotlib.collections
from matplotlib.markers import MarkerStyle
from matplotlib.path import Path as MPath
from matplotlib.backend_bases import _get_renderer
from matplotlib.backends.backend_pdf import FigureCanvasPdf
from matplotlib.ticker import FixedLocator

COL_W = 252.0 / 72.27                   # \columnwidth in inches
COL_FS, COL_FS_REG, COL_FS_LEG = 8.0, 7.0, 6.5
COL_MS, COL_MEW = 4.0, 0.8              # one marker size and edge width for every marker
COL_LW, COL_LW_SEG, COL_LW_B = 1.1, 0.8, 1.3
COL_BCOL = "#7f3b08"                    # full-load boundary and its label; no curve uses this colour
COL_DASHDOT, COL_DASHED = (0, (3.5, 1.2, 1.0, 1.2)), (0, (3.0, 1.5))
COL_LEGKW = dict(fontsize=COL_FS_LEG, handlelength=1.6, labelspacing=0.12, borderpad=0.25, columnspacing=0.8,
                 handletextpad=0.4, borderaxespad=0.0, frameon=True, fancybox=False, framealpha=1.0, edgecolor="0.6")
COL_L, COL_R, COL_B = 0.47, 0.035, 0.335    # left margin of each panel, right margin, bottom margin (ticks + x label)
COL_AW = 1.15                                # axes width of every panel
COL_AH = 0.75 * COL_AW                       # 8:6 box
COL_GAP = COL_W - 2 * COL_L - 2 * COL_AW - COL_R + COL_L   # left edge of panel (b) minus right edge of panel (a)
COL_TOP = 0.14                               # row of the full-load label and the panel tags above the axes
COL_BAND = 0.30                              # legend band; one canvas for both figures
COL_X0, COL_X1 = 0.4, 8.75                   # every load axis (no marker within its ink radius of a frame edge)
COL_DODGE = {0: -0.2, 10: 0.0, 20: 0.2}      # markers of the three SNRs staggered around each load
COL_DODGE_D = {0: -0.3, 10: 0.0, 20: 0.3}    # the three coincident kappa ~ 0 diamonds at U = 8, spread wider
COL_ETA = (("1.0", "$\\eta\\!=\\!1$", 0.0), ("0.5", "$\\eta\\!=\\!0.5$", 0.0), ("0.0", "$\\eta\\!=\\!0$", -0.021))
                                             # gain correlation, its tag, tag offset (the eta = 0 tag sits below the
                                             # zero line, which the guard checks)
COL_RHO = ("0.3", "0.5", "0.7", "0.9", "0.95")   # rho_f values spanned by the bands (each where it was run)
COL_UCOH = (1, 2, 4, 6)
COL_FONT = os.environ.get("CL_COL_FONT", "Arial")
COL_MINUS = "−"
# one style per scheme: colour = SNR, marker and line style = scheme; eta sets the line style of the
# channel-aware masks
COL_STY = {
    "Proposed": dict(lw=COL_LW, ls="-", marker="*", zorder=5),
    "Static":   dict(lw=COL_LW_SEG, ls=COL_DASHDOT, alpha=0.7, zorder=2),
    "StaticMk": dict(ls="none", marker="s", mfc="white", zorder=3),
    "Learned":  dict(ls="none", marker="o", mfc="white", zorder=4),
    "Model08":  dict(lw=COL_LW, ls="-", zorder=3),
    "Model04":  dict(lw=COL_LW, ls=COL_DASHED, zorder=3),
    "Mix08":    dict(ls="none", marker="o", zorder=5),
    "Mix0":     dict(ls="none", marker="D", mfc="none", zorder=5),
    "Chan":     {"1.0": dict(lw=COL_LW, ls="-", zorder=3), "0.5": dict(lw=COL_LW, ls=COL_DASHDOT, zorder=3),
                 "0.0": dict(lw=COL_LW, ls=COL_DASHED, zorder=3)},
    "Choice":   dict(ls="none", marker="^", mfc="white", zorder=4),
}


def col_rc():
    return {"font.size": COL_FS, "axes.labelsize": COL_FS, "xtick.labelsize": COL_FS, "ytick.labelsize": COL_FS,
            "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6, "xtick.major.size": 2.5,
            "ytick.major.size": 2.5, "xtick.minor.size": 1.5, "xtick.major.pad": 1.5, "ytick.major.pad": 1.5,
            "axes.labelpad": 1.5, "lines.markersize": COL_MS, "lines.markeredgewidth": COL_MEW,
            "mathtext.fontset": "cm", "pdf.fonttype": 42, "grid.linewidth": 0.4, "legend.fontsize": COL_FS_LEG,
            "font.family": "sans-serif", "font.sans-serif": [COL_FONT, "DejaVu Sans"]}


def col_canvas():
    """two panels side by side under the legend band; returns the figure, the two axes and the canvas height."""
    H = COL_B + COL_AH + COL_TOP + COL_BAND
    fig = plt.figure(figsize=(COL_W, H))
    axs = [fig.add_axes([(COL_L + j * (COL_AW + COL_GAP)) / COL_W, COL_B / H, COL_AW / COL_W, COL_AH / H])
           for j in (0, 1)]
    return fig, axs, H


def col_regions(ax, light_xy, over_xy):
    """light load U < K (label inside), full load U = K (dashed boundary in its own colour, label at its top),
    overload U > K (light grey, label inside)."""
    ax.axvspan(K + 0.5, COL_X1, color="0.91", lw=0, zorder=0)
    ax.axvline(K, color=COL_BCOL, lw=COL_LW_B, ls=(0, (3.0, 1.6)), zorder=1.5, gid="ref")
    ax.text(K, 1.0 + 2.0 / (COL_AH * 72), "full load", transform=ax.get_xaxis_transform(), ha="center",
            va="bottom", fontsize=COL_FS_REG, color=COL_BCOL, gid="boundary")
    ax.text(*light_xy, "light load", ha="center", va="center", fontsize=COL_FS_REG, color="0.2", gid="region")
    ax.text(*over_xy, "overload", ha="center", va="center", fontsize=COL_FS_REG, color="0.2", gid="region")


def col_load_axis(ax):
    ax.set_xlim(COL_X0, COL_X1); ax.set_xticks(LOADS)
    ax.set_xlabel("Load $U$ (active users)"); ax.grid(alpha=0.3)


def col_yticks(ax, ticks, labels):
    ax.yaxis.set_major_locator(FixedLocator(ticks)); ax.set_yticklabels(labels)


def col_tag(fig, ax, lab, H):
    """panel tag at the top left of the panel, in the row of the full-load label."""
    fig.text(ax.get_position().x0 - (COL_L - 0.02) / COL_W, (COL_B + COL_AH) / H + 2.0 / 72 / H, lab,
             fontsize=COL_FS, ha="left", va="bottom", gid="tag")


def col_key(**kw):
    """SNR key: the colour alone, written as coloured text without a handle, so that it cannot read as a scheme."""
    h = [Line2D([], [], ls="none", label=f"{s} dB") for s in SNR3]
    return dict(handles=h, labelcolor=[SCOL[s] for s in SNR3], **{**COL_LEGKW, "handlelength": 0.0,
                                                                   "handletextpad": 0.0, **kw})


def col_band(fig, H, left, right):
    """the one placement rule of the band legends in both figures: each legend hangs from the top edge of the canvas,
    the first flush with the left edge of the canvas, the last flush with the right edge of the panels. left and right
    are (handles, legend keyword) pairs."""
    ytop = 1.0 - 0.01 / H
    for (h, kw), loc, x in ((left, "upper left", 0.02 / COL_W), (right, "upper right", 1.0 - COL_R / COL_W)):
        fig.legend(handles=h, loc=loc, bbox_to_anchor=(x, ytop), **kw)


def fig_load_col():
    with plt.rc_context(col_rc()):
        fig, (a, b), H = col_canvas()
        # (a) per-user CosSim
        col_regions(a, light_xy=(2.3, 0.6785), over_xy=(6.6, 0.836))
        for s in SNR3:
            c = SCOL[s]
            for k in LOADS:
                y = static(k, s)
                a.plot([1, k], [y, y], color=c, **COL_STY["Static"])
                a.plot([k], [y], color=c, **COL_STY["StaticMk"])
            Lr = [(U, learned(U, s)) for U in LOADS if learned(U, s) is not None]
            a.plot([u for u, _ in Lr], [v for _, v in Lr], color=c, **COL_STY["Learned"])
            a.plot(LOADS, [proposal(U, s) for U in LOADS], color=c, **COL_STY["Proposed"])
        col_load_axis(a); a.set_ylim(0.645, 1.018)
        col_yticks(a, [0.7, 0.8, 0.9, 1.0], ["0.7", "0.8", "0.9", "1"]); a.set_ylabel("Per-user CosSim")
        # (b) margin of Proposed over Learned ODS-U and over Static ODS-U
        col_regions(b, light_xy=(2.3, 0.0512), over_xy=(6.6, 0.027))
        b.axhline(0, color="0.35", lw=0.6, zorder=1, gid="ref")
        for s in SNR3:
            c = SCOL[s]
            b.plot([U + COL_DODGE[s] for U in LOADS], [proposal(U, s) - static(U, s) for U in LOADS], color=c,
                   **COL_STY["StaticMk"])
            Lr = [(U, proposal(U, s) - learned(U, s)) for U in LOADS if learned(U, s) is not None]
            b.plot([u + COL_DODGE[s] for u, _ in Lr], [v for _, v in Lr], color=c, **COL_STY["Learned"])
        col_load_axis(b); b.set_ylim(-0.0075, 0.0572)
        col_yticks(b, [0, 0.02, 0.04], ["0", "0.02", "0.04"]); b.set_ylabel("CosSim margin")
        # legend band: schemes in one framed box, the SNR key in a second one beside it; every label
        # is set through mathtext so that the three labels share one baseline
        h = [Line2D([0], [0], color="k", label="$\\mathregular{Proposed}$", **COL_STY["Proposed"]),
             Line2D([0], [0], color="k", lw=COL_LW_SEG, ls=COL_DASHDOT, marker="s", mfc="white",
                    label="Static ODS-$k$"),
             Line2D([0], [0], color="k", label="Learned ODS-$U$", **COL_STY["Learned"])]
        key = col_key(ncol=3)
        col_band(fig, H, (h, dict(ncol=3, **COL_LEGKW)), (key.pop("handles"), key))
        col_tag(fig, a, "(a)", H); col_tag(fig, b, "(b)", H)
        col_save(fig, "fig2_load_col.pdf")


def fig_scene_col():
    with plt.rc_context(col_rc()):
        fig, (a, b), H = col_canvas()
        # (a) channel-aware masks and the coherence-aware choice over the load masks, 0 dB, offsets within +-6 dB
        c = SCOL[0]
        col_regions(a, light_xy=(2.05, 0.165), over_xy=(6.65, 0.092))
        for e, lab, dy in COL_ETA:
            runs = [[coh(rho, e, U, "csiq") for U in COL_UCOH] for rho in COL_RHO]
            runs = [g for g in runs if None not in g]
            a.fill_between(COL_UCOH, np.min(runs, axis=0), np.max(runs, axis=0), color=c, alpha=0.16, lw=0,
                           zorder=2)
            y = [coh("0.7", e, U, "csiq") for U in COL_UCOH]
            a.plot(COL_UCOH, y, color=c, **COL_STY["Chan"][e])
            a.plot(COL_UCOH, [coh("0.7", e, U, "switch") for U in COL_UCOH], color=c, **COL_STY["Choice"])
            a.text(6.5, y[-1] + dy, lab, ha="left", va="center", fontsize=COL_FS_REG, color=c, gid="region")
        a.axhline(0, color="0.35", lw=0.6, zorder=1, gid="ref")
        col_load_axis(a); a.set_ylim(-0.062, 0.19)
        col_yticks(a, [-0.05, 0, 0.05, 0.1, 0.15], [COL_MINUS + "0.05", "0", "0.05", "0.1", "0.15"])
        a.set_ylabel("CosSim gain")
        # (b) one scene: overlap NMSE reduction versus the load
        UU = [2, 3, 4, 5, 6, 7, 8]
        col_regions(b, light_xy=(2.3, 68.0), over_xy=(6.6, 67.6))
        b.legend(loc="upper center", bbox_to_anchor=(2.3, 59.5), bbox_transform=b.transData, ncol=1, **col_key())
        for s in SNR3:
            c = SCOL[s]
            for mix, key in ((0.8, "Model08"), (0.4, "Model04")):
                pts = [(U, theory_red(mix, U, s)) for U in UU]
                b.plot([u for u, _ in pts], [v for _, v in pts], color=c, **COL_STY[key])
            pts = [(U, real_red(0.8, U, s)) for U in UU]
            xx = np.array([u for u, _ in pts]) + COL_DODGE[s]; m = np.array([v[0] for _, v in pts])
            lo = np.array([v[1] for _, v in pts]); hi = np.array([v[2] for _, v in pts])
            b.errorbar(xx, m, yerr=[m - lo, hi - m], color=c, mfc=c, capsize=1.5, elinewidth=0.7, capthick=0.7,
                       **COL_STY["Mix08"])
            b.plot([8 + COL_DODGE_D[s]], [real_red(0.0, 8, s)[0]], color=c, **COL_STY["Mix0"])
        b.axhline(0, color="0.35", lw=0.6, zorder=1, gid="ref")
        col_load_axis(b); b.set_ylim(-6, 75); col_yticks(b, [0, 20, 40, 60], ["0", "20", "40", "60"])
        b.set_ylabel("Reduction (%)")
        # legend band: (a) left, (b) right
        ha = [(Patch(facecolor=SCOL[0], alpha=0.16, lw=0), Line2D([0], [0], color=SCOL[0], lw=COL_LW)),
              Line2D([0], [0], color=SCOL[0], **COL_STY["Choice"])]
        la = dict(labels=["Channel-aware", "Coherence-aware choice"], ncol=1, **COL_LEGKW)
        hb = [Line2D([0], [0], color="k", label="Model, $\\bar\\kappa\\!=\\!0.8$", **COL_STY["Model08"]),
              Line2D([0], [0], color="k", label="Model, $\\bar\\kappa\\!=\\!0.4$", **COL_STY["Model04"]),
              Line2D([0], [0], color="k", label="Mixtures, $\\bar\\kappa\\!=\\!0.8$", **COL_STY["Mix08"]),
              Line2D([0], [0], color="k", label="Mixtures, $\\bar\\kappa\\!\\approx\\!0$", **COL_STY["Mix0"])]
        col_band(fig, H, (ha, la), (hb, dict(ncol=2, **COL_LEGKW)))
        col_tag(fig, a, "(a)", H); col_tag(fig, b, "(b)", H)
        col_save(fig, "fig3_scene_col.pdf")


def col_ink(ln):
    """half extents (left, right, down, up) in pt of the ink of the marker of ln: the marker path at its size, its
    edge, and for miter joins the miter tips (a diamond reaches 0.707 ms + 0.71 mew from its centre)."""
    ms = MarkerStyle(ln.get_marker())
    p = ms.get_path().transformed(ms.get_transform())
    v = p.vertices * ln.get_markersize()
    hw = 0.5 * ln.get_markeredgewidth()
    if ms.get_joinstyle() == "miter" and p.codes is not None and p.codes[-1] == MPath.CLOSEPOLY:
        poly = v[:-1]
        if np.allclose(poly[0], poly[-1]):
            poly = poly[:-1]
        sgn = 1.0 if 0.5 * np.sum(poly[:, 0] * np.roll(poly[:, 1], -1) - np.roll(poly[:, 0], -1) * poly[:, 1]) > 0 \
            else -1.0
        tips = []
        for i in range(len(poly)):
            a, b, c = poly[i - 1], poly[i], poly[(i + 1) % len(poly)]
            e1, e2 = (b - a) / np.hypot(*(b - a)), (c - b) / np.hypot(*(c - b))
            n1, n2 = sgn * np.array([e1[1], -e1[0]]), sgn * np.array([e2[1], -e2[0]])
            tips.append(b + (n1 + n2) * hw / (1.0 + n1 @ n2))
        t = np.array(tips)
        return -t[:, 0].min(), t[:, 0].max(), -t[:, 1].min(), t[:, 1].max()
    return -v[:, 0].min() + hw, v[:, 0].max() + hw, -v[:, 1].min() + hw, v[:, 1].max() + hw


def col_pdf_layout(fig):
    """lay the figure out with the renderer of the PDF backend at 72 dpi (1 px = 1 pt) and return that renderer. The
    PDF is what is printed, and its text metrics differ from those of the Agg canvas (whose mathtext boxes carry one
    pixel of padding at the canvas dpi), so every distance of the guard is measured on this layout."""
    r = _get_renderer(fig, FigureCanvasPdf(fig).print_pdf)
    fig.draw(r)
    return r


def col_guard(fig, clear_pt=2.0, marker_clear_pt=0.5, leg_fs=None):
    """strict layout guard of the column figures, every panel (all distances in printed pt):
    1. every text and every legend inside the canvas; any two texts at least clear_pt apart, except an axis label and
       the tick labels of its own axis (set by the label pad); every y-axis label within the height of its axes;
    2. no figure legend overlaps an axes box, and every figure legend keeps clear_pt from every text and every other
       legend;
    3. every axes legend inside its axes and at least clear_pt from every text;
    4. no ink of a curve, marker, error bar or band within 0.5 pt of an axes legend or within clear_pt of a label
       inside the axes (region labels, eta tags); horizontal lines with gid 'ref' (zero lines) keep clear_pt from
       those labels but may pass under a legend; the full-load boundary is checked by 5.;
    5. every label inside the axes keeps clear_pt from the frame and from the full-load boundary;
    6. the ink of every marker (marker path at its size, edge, miter tips) at least marker_clear_pt inside the inner
       edge of the axes frame;
    7. printed sizes: legends 6.5 pt (leg_fs when given), every other text at least 7 pt, no sub- or superscript in
       any text.
    The layout is that of the PDF backend (col_pdf_layout)."""
    leg_fs = COL_FS_LEG if leg_fs is None else leg_fs
    r = col_pdf_layout(fig)
    Wp, Hp = fig.bbox.width, fig.bbox.height
    k = fig.dpi / 72.0
    cp = clear_pt * k
    legs_ax = [(ax, c) for ax in fig.axes for c in ax.get_children() if isinstance(c, matplotlib.legend.Legend)]

    def on_canvas(e, what):
        assert e.x0 >= -0.5 and e.y0 >= -0.5 and e.x1 <= Wp + 0.5 and e.y1 <= Hp + 0.5, ("leaves the canvas", what)

    def ov(e, f, pad=0.0):
        return e.x0 - pad < f.x1 and f.x0 - pad < e.x1 and e.y0 - pad < f.y1 and f.y0 - pad < e.y1

    texts = []                                                               # (text, name, kind, axes)
    for ax in fig.axes:
        for kind, tl in (("x", ax.get_xticklabels()), ("y", ax.get_yticklabels())):
            for t in tl:
                if t.get_visible() and t.get_text():
                    texts.append((t, "tick " + t.get_text(), kind + "tick", ax))
        texts.append((ax.xaxis.label, ax.xaxis.label.get_text(), "xlabel", ax))
        texts.append((ax.yaxis.label, ax.yaxis.label.get_text(), "ylabel", ax))
        texts += [(t, t.get_text(), "text", ax) for t in ax.texts]
    texts += [(t, t.get_text(), "fig", None) for t in fig.texts]
    for t, s, _, _ in texts:                                                 # 7. sizes and scripts
        assert t.get_fontsize() >= 7.0 - 1e-9, ("text below 7 pt", s, t.get_fontsize())
        assert "_" not in s and "^" not in s, ("sub- or superscript", s)
    for lg in list(fig.legends) + [l for _, l in legs_ax]:
        for t in lg.get_texts():
            assert abs(t.get_fontsize() - leg_fs) < 1e-9, ("legend size", t.get_text())
            assert "_" not in t.get_text() and "^" not in t.get_text(), ("sub- or superscript", t.get_text())
    boxes = [(t.get_window_extent(r), s, kind, ax) for t, s, kind, ax in texts]
    for e, s, _, _ in boxes:                                                 # 1.
        on_canvas(e, s)
    for i in range(len(boxes)):
        for j in range(i + 1, len(boxes)):
            (e, s, ki, ai), (f, s2, kj, aj) = boxes[i], boxes[j]
            own = ai is aj and {ki, kj} in ({"xtick", "xlabel"}, {"ytick", "ylabel"})
            assert own or not ov(e, f, cp), ("texts closer than clear_pt", s, s2)
    for e, s, kind, ax in boxes:
        if kind == "ylabel":
            axe = ax.get_window_extent(r)
            assert e.y0 >= axe.y0 - 0.5 and e.y1 <= axe.y1 + 0.5, ("y-axis label longer than its axes", s)
    for lg in fig.legends:                                                   # 2.
        e = lg.get_window_extent(r); on_canvas(e, "legend")
        for ax in fig.axes:
            assert not ov(e, ax.get_window_extent(r)), "figure legend over an axes box"
        for f, s, _, _ in boxes:
            assert not ov(e, f, cp), ("figure legend within clear_pt of a text", s)
        for lg2 in fig.legends:
            if lg2 is not lg:
                assert not ov(e, lg2.get_window_extent(r), cp), "legends closer than clear_pt"
    for ax, lg in legs_ax:                                                   # 3.
        e, axe = lg.get_window_extent(r), ax.get_window_extent(r)
        assert e.x0 >= axe.x0 - 0.5 and e.y0 >= axe.y0 - 0.5 and e.x1 <= axe.x1 + 0.5 and e.y1 <= axe.y1 + 0.5, \
            "axes legend leaves its axes"
        for f, s, _, _ in boxes:
            assert not ov(e, f, cp), ("axes legend within clear_pt of a text", s)
    for ax in fig.axes:                                                      # 6.
        axe = ax.get_window_extent(r)
        sp = 0.5 * max(s.get_linewidth() for s in ax.spines.values())
        for ln in ax.lines:
            if ln.get_gid() == "ref" or ln.get_marker() in (None, "None", "none", ""):
                continue
            d = np.column_stack(ln.get_data()).astype(float)
            if len(d) == 0:
                continue
            xy = ax.transData.transform(d)
            l_, r_, d_, u_ = [(v + sp + marker_clear_pt) * k for v in col_ink(ln)]
            ok = ((xy[:, 0] - l_ >= axe.x0) & (xy[:, 0] + r_ <= axe.x1) & (xy[:, 1] - d_ >= axe.y0)
                  & (xy[:, 1] + u_ <= axe.y1))
            assert ok.all(), ("marker ink within marker_clear_pt of the frame", ln.get_color(), fig.axes.index(ax),
                              d[~ok][0])
    for ax in fig.axes:                                                      # 4. and 5.
        axe = ax.get_window_extent(r)
        items = [(t.get_window_extent(r), t.get_text(), True) for t in ax.texts if t.get_gid() == "region"]
        items += [(lg.get_window_extent(r), "axes legend", False) for a2, lg in legs_ax if a2 is ax]
        xb = ax.transData.transform((K, 0))[0]
        for e, s, isreg in items:
            c = clear_pt * k
            if isreg:
                assert e.x0 >= axe.x0 + c and e.x1 <= axe.x1 - c and e.y0 >= axe.y0 + c and e.y1 <= axe.y1 - c, \
                    ("label off its axes or within clear_pt of the frame", s)
            half = 0.5 * COL_LW_B * k
            assert e.x1 <= xb - half - c or e.x0 >= xb + half + c, ("label or legend touches the boundary", s)
            for ln in ax.lines:
                d = np.column_stack(ln.get_data()).astype(float)
                if len(d) == 0:
                    continue
                if ln.get_gid() == "ref":
                    if isreg and np.allclose(d[:, 0], [0, 1]) and np.ptp(d[:, 1]) == 0:     # zero line (axhline)
                        yl = ln.get_transform().transform((0.0, d[0, 1]))[1]
                        pad = (0.5 * ln.get_linewidth() + clear_pt) * k
                        assert not (e.y0 - pad < yl < e.y1 + pad), ("label within clear_pt of a zero line", s)
                    continue
                xy = ax.transData.transform(d)
                cl = clear_pt if isreg else 0.5
                has_m = ln.get_marker() not in (None, "None", "none", "")
                has_l = ln.get_linestyle() not in ("None", "none", "", " ") and len(xy) > 1
                groups = []                                  # (points, ink half-width in pt)
                if has_m:
                    groups.append((xy, max(col_ink(ln))))
                if has_l:
                    tt = np.linspace(0, 1, 120)[:, None]
                    groups.append((np.vstack([p + tt * (q - p) for p, q in zip(xy[:-1], xy[1:])]),
                                   0.5 * ln.get_linewidth()))
                for pts, ink in groups:
                    pad = (ink + cl) * k
                    bad = ((pts[:, 0] > e.x0 - pad) & (pts[:, 0] < e.x1 + pad) & (pts[:, 1] > e.y0 - pad)
                           & (pts[:, 1] < e.y1 + pad))
                    assert not bad.any(), ("curve or marker under", s, ln.get_color(), fig.axes.index(ax),
                                           ax.transData.inverted().transform(pts[bad][0]))
            pad = (clear_pt if isreg else 0.5) * k
            corners = np.array([[e.x0 - pad, e.y0 - pad], [e.x1 + pad, e.y0 - pad], [e.x0 - pad, e.y1 + pad],
                                [e.x1 + pad, e.y1 + pad], [0.5 * (e.x0 + e.x1), 0.5 * (e.y0 + e.y1)]])
            for coll in ax.collections:                      # error bars and bands: edges sampled, box corners
                for path in coll.get_paths():                # tested against filled bands
                    if len(path.vertices) == 0:
                        continue
                    dp = path.transformed(coll.get_transform())
                    v = dp.interpolated(60).vertices
                    bad = ((v[:, 0] > e.x0 - pad) & (v[:, 0] < e.x1 + pad) & (v[:, 1] > e.y0 - pad)
                           & (v[:, 1] < e.y1 + pad))
                    assert not bad.any(), ("band or error bar under", s, fig.axes.index(ax))
                    if isinstance(coll, matplotlib.collections.PolyCollection):
                        assert not dp.contains_points(corners).any(), ("label inside a band", s)


def col_save(fig, name):
    for lg in list(fig.legends) + [c for ax in fig.axes for c in ax.get_children()
                                   if isinstance(c, matplotlib.legend.Legend)]:
        lg.get_frame().set_linewidth(0.5)
    out = os.path.join(FIG, name)
    col_guard(fig)
    fig.savefig(out, metadata={"CreationDate": None})
    fig.savefig(out.replace(".pdf", ".png"), dpi=300)
    plt.close(fig)
    print("wrote", out)


# =================================================================================================================
# One-panel figures (2026-11-17), one representative panel per result figure of the manuscript, so that each
# prints large: Fig. 2 = per-user CosSim versus the load U = 1..8 at 0, 10, 20 dB (fig_load_one(), a
# "nines" and a linear version), Fig. 3 = overlap NMSE reduction versus the load (fig_scene_one()). Native print
# size as above: canvas \columnwidth wide, both PDFs included at width=\columnwidth. One canvas and one 8:6 axes box
# for both figures; ticks and axis labels 8 pt, region labels and SNR tags 7 pt, legends 7 pt
# (ONE_FS_LEG); the three region labels in one row inside the top of the axes in both figures (light load, full load
# in the colour of its boundary, which stops below it, overload), the legend below that row, on a background that hides
# the grid behind each label (one_label_bg()); the SNR written in the colour of its curves right of the axes beside the
# curve ends at U = 8 (plain tags: each tag names the colour of a set of curves, not one curve);
# the styles COL_STY. The margin panel (fig_load_col (b)) and the coherence panel (fig_scene_col (a)) stay in
# the code for the letter.
#   fig_cl/fig2_load_one_nines.pdf  y drawn as -log10(1 - CosSim), ticks labelled as CosSim at half decades
#                                   (0.7, 0.9, 0.97, 0.99, 0.997), unlabelled minor ticks at 0.8, 0.95, 0.98, 0.995
#   fig_cl/fig2_load_one_lin.pdf    linear CosSim axis
#   fig_cl/fig2_load_one.pdf        the manuscript version (ONE_FIG2), the other one copied to fig2_alt.pdf
#   fig_cl/fig3_scene_one.pdf       overlap NMSE reduction (%) of the overlapping masks (18) over the load masks
# Static ODS-k segments are drawn opaque in the 70 percent tint of their SNR colour (the alpha of COL_STY["Static"]
# blended two coincident segments into a third colour: 10 dB ODS-2 at 0.9618 and 20 dB ODS-8 at 0.9614), the
# shorter segment on top, so each segment keeps its own colour up to its square.
# fig_load_one() prints the vertical gap in pt between Proposed and Learned ODS-U at every load with a Learned point.
# =================================================================================================================
ONE_FS_LEG = 7.0
ONE_LM, ONE_RM = 0.50, 0.31          # inches: least left margin (ticks + y label), least right margin (SNR tags)
ONE_B, ONE_T = 0.30, 0.02            # inches: bottom (ticks + x label), top (half the frame line; labels inside)
ONE_AW = float(os.environ.get("CL_ONE_AW", 2.4))   # axes width in inches
ONE_AH = 0.75 * ONE_AW                               # 8:6 box
ONE_L = ONE_LM + 0.5 * (COL_W - ONE_LM - ONE_AW - ONE_RM)   # block of labels, axes and tags centred
ONE_X0, ONE_X1 = 0.25, 8.6           # load axis of both figures
ONE_LEGKW = {**COL_LEGKW, "fontsize": ONE_FS_LEG}
ONE_NINES_MINOR = (0.8, 0.95, 0.98, 0.995)
ONE_SHADE = "0.91"                   # overload shading (as COL_ regions)
ONE_BG_PAD = 0.4                     # pt of background around each label of the region row


def one_canvas():
    r"""one axes box, canvas \columnwidth wide; the SNR tags use the margin at the right."""
    H = ONE_B + ONE_AH + ONE_T
    fig = plt.figure(figsize=(COL_W, H))
    ax = fig.add_axes([ONE_L / COL_W, ONE_B / H, ONE_AW / COL_W, ONE_AH / H])
    return fig, ax, H


def one_load_axis(ax):
    ax.set_xlim(ONE_X0, ONE_X1); ax.set_xticks(LOADS)
    ax.set_xlabel("Load $U$ (active users)"); ax.grid(alpha=0.3)


ONE_LAB_TOP = 2.3         # pt from the top edge of the axes to the top of the row of region labels
ONE_LAB_H = 9.0           # pt from the top edge of the axes to the bottom of that row (7 pt text; checked by one_guard)
ONE_B_GAP = 3.0           # pt between the bottom of the full-load label and the top of its boundary line
ONE_LEG_TOP = ONE_LAB_H + 3.0   # pt from the top edge of the axes to the top of a legend below the label row
ONE_LIGHT_X, ONE_OVER_X = 2.45, 0.5 * (K + 0.5 + ONE_X1)   # centres of the light-load and overload labels


def one_regions(ax):
    """overload shaded; the three region labels in one row inside the top of the axes, the same in both figures:
    light load over U = 1..3, full load on its boundary in the colour of the boundary, overload over its region; the
    dashed full-load boundary stops ONE_B_GAP below its label (a boundary may stop short of a label)."""
    ax.axvspan(K + 0.5, ONE_X1, color=ONE_SHADE, lw=0, zorder=0)
    AHp = ONE_AH * 72
    tr = ax.get_xaxis_transform()
    ax.plot([K, K], [0.0, 1.0 - (ONE_LAB_H + ONE_B_GAP) / AHp], transform=tr, color=COL_BCOL, lw=COL_LW_B,
            ls=(0, (3.0, 1.6)), zorder=1.5, gid="ref")
    y = 1.0 - ONE_LAB_TOP / AHp
    ax.text(K, y, "full load", transform=tr, ha="center", va="top", fontsize=COL_FS_REG, color=COL_BCOL,
            gid="boundary")
    for x, s in ((ONE_LIGHT_X, "light load"), (ONE_OVER_X, "overload")):
        ax.text(x, y, s, transform=tr, ha="center", va="top", fontsize=COL_FS_REG, color="0.2", gid="region")


def one_guard(fig, clear_pt=2.0):
    """additions to col_guard for the row of region labels inside the axes: every label of the row (region labels and
    the full-load label) ends at most ONE_LAB_H below the top edge and keeps clear_pt from every curve and marker ink
    (the full-load label is not checked by col_guard, which keeps the region labels clear of the boundary), and the
    boundary line ends at least clear_pt below the full-load label."""
    r = col_pdf_layout(fig)
    k = fig.dpi / 72.0
    for ax in fig.axes:
        axe = ax.get_window_extent(r)
        labs = [t for t in ax.texts if t.get_gid() in ("region", "boundary")]
        for t in labs:
            e = t.get_window_extent(r)
            assert axe.y1 - e.y0 <= ONE_LAB_H * k + 1e-6, ("label row deeper than ONE_LAB_H", t.get_text(),
                                                             (axe.y1 - e.y0) / k)
            assert axe.y1 - e.y1 >= clear_pt * k - 1e-6, ("label within clear_pt of the top edge", t.get_text())
            if t.get_gid() != "boundary":
                continue
            for ln in ax.lines:
                d = np.column_stack(ln.get_data()).astype(float)
                if ln.get_gid() == "ref":
                    top = ln.get_transform().transform(d).max(axis=0)[1]
                    assert e.y0 - top >= clear_pt * k, ("boundary line within clear_pt of its label", (e.y0 - top) / k)
                    continue
                xy = ax.transData.transform(d)
                pts = [(xy, max(col_ink(ln)))] if ln.get_marker() not in (None, "None", "none", "") else []
                if ln.get_linestyle() not in ("None", "none", "", " ") and len(xy) > 1:
                    tt = np.linspace(0, 1, 120)[:, None]
                    pts.append((np.vstack([p + tt * (q - p) for p, q in zip(xy[:-1], xy[1:])]),
                                0.5 * ln.get_linewidth()))
                for P, ink in pts:
                    pad = (ink + clear_pt) * k
                    bad = (P[:, 0] > e.x0 - pad) & (P[:, 0] < e.x1 + pad) & (P[:, 1] > e.y0 - pad) & \
                          (P[:, 1] < e.y1 + pad)
                    assert not bad.any(), ("curve or marker under the full-load label", ln.get_color())


def one_tag(ax, y, s):
    """SNR tag in the colour of its curves, 3 pt right of the frame at the height y (data) of its curve set."""
    ax.annotate(f"{s} dB", xy=(1.0, y), xycoords=("axes fraction", "data"), xytext=(3.0, 0.0),
                textcoords="offset points", ha="left", va="center", fontsize=COL_FS_REG, color=SCOL[s], gid="tag")


def nines_fwd(c):
    return -np.log10(np.clip(1.0 - np.asarray(c, dtype=float), 1e-9, None))


def nines_inv(y):
    return 1.0 - 10.0 ** (-np.asarray(y, dtype=float))


def one_tint(c, a=0.7):
    """the opaque colour of c drawn at alpha a on white."""
    rgb = np.array(matplotlib.colors.to_rgb(c))
    return tuple(a * rgb + (1.0 - a))


ONE_LOAD = {   # per version: y limits, labelled ticks, minor ticks, legend anchor (axes fraction)
    "nines": dict(ylim=(0.63, 0.9974), ticks=[0.7, 0.9, 0.97, 0.99, 0.997], minor=ONE_NINES_MINOR, leg="top"),
    "lin":   dict(ylim=(0.67, 1.05), ticks=[0.7, 0.8, 0.9, 1.0], minor=(), leg=(1.0, 0.55)),
}


def one_gaps(fig, ax):
    """printed vertical gap (pt) between the Proposed star and the Learned ODS-U circle at every load with a Learned
    ODS-U point (U = 1..6 and 8); a gap below the 4 pt marker (COL_MS) means the circle touches the star."""
    col_pdf_layout(fig)
    k = fig.dpi / 72.0
    out = {}
    for s in SNR3:
        for U in [u for u in LOADS if learned(u, s) is not None]:
            yp = ax.transData.transform((U, proposal(U, s)))[1]
            yl = ax.transData.transform((U, learned(U, s)))[1]
            out[(s, U)] = (yp - yl) / k
    return out


def fig_load_one(scale="nines"):
    cfg = ONE_LOAD[scale]
    with plt.rc_context(col_rc()):
        fig, a, H = one_canvas()
        if scale == "nines":
            a.set_yscale("function", functions=(nines_fwd, nines_inv))
        one_regions(a)
        seg = {**COL_STY["Static"], "alpha": 1.0}
        for s in SNR3:
            c = SCOL[s]
            for k in LOADS:
                y = static(k, s)
                a.plot([1, k], [y, y], color=one_tint(c, COL_STY["Static"]["alpha"]),
                       **{**seg, "zorder": seg["zorder"] + 0.01 * (len(LOADS) - k)})
                a.plot([k], [y], color=c, **COL_STY["StaticMk"])
            Lr = [(U, learned(U, s)) for U in LOADS if learned(U, s) is not None]
            a.plot([u for u, _ in Lr], [v for _, v in Lr], color=c, **COL_STY["Learned"])
            a.plot(LOADS, [proposal(U, s) for U in LOADS], color=c, **COL_STY["Proposed"])
            one_tag(a, proposal(8, s), s)
        one_load_axis(a); a.set_ylim(*cfg["ylim"])
        col_yticks(a, cfg["ticks"], [("%g" % t) if t < 1 else "1" for t in cfg["ticks"]])
        a.yaxis.set_minor_locator(FixedLocator(list(cfg["minor"])))
        a.tick_params(axis="y", which="minor", length=1.5, width=0.5)
        a.set_ylabel("Per-user CosSim")
        h = [Line2D([0], [0], color="k", label="$\\mathregular{Proposed}$", **COL_STY["Proposed"]),
             Line2D([0], [0], color="k", lw=COL_LW_SEG, ls=COL_DASHDOT, marker="s", mfc="white",
                    label="Static ODS-$k$"),
             Line2D([0], [0], color="k", label="Learned ODS-$U$", **COL_STY["Learned"])]
        leg = (1.0, 1.0 - ONE_LEG_TOP / (ONE_AH * 72)) if cfg["leg"] == "top" else cfg["leg"]
        a.legend(handles=h, loc="upper right", bbox_to_anchor=leg, ncol=1, **{**ONE_LEGKW, "borderaxespad": 0.2})
        gaps = one_gaps(fig, a)
        print(f"Fig. 2 {scale}: printed gap Proposed - Learned ODS-U (pt):",
              ", ".join(f"{s} dB U={U}: {g:.2f}" for (s, U), g in gaps.items()))
        small = [f"{s} dB U={U}" for (s, U), g in gaps.items() if g < COL_MS]
        print(f"Fig. 2 {scale}: gap below the {COL_MS:g} pt marker at", ", ".join(small) if small else "no load")
        one_save(fig, f"fig2_load_one_{scale}.pdf")
        return gaps


ONE_SCENE_TAG = {0: 33.6, 10: 27.8, 20: 53.0}    # heights of the SNR tags of Fig. 3 beside the curve ends at U = 8:
                                                 # 10 dB at its model end (27.81), 0 dB just above its marker (32.89)
                                                 # and 2 pt clear of the 10 dB tag, 20 dB between model and marker


def fig_scene_one():
    with plt.rc_context(col_rc()):
        fig, b, H = one_canvas()
        UU = [2, 3, 4, 5, 6, 7, 8]
        one_regions(b)
        for s in SNR3:
            c = SCOL[s]
            for mix, key in ((0.8, "Model08"), (0.4, "Model04")):
                pts = [(U, theory_red(mix, U, s)) for U in UU]
                b.plot([u for u, _ in pts], [v for _, v in pts], color=c, **COL_STY[key])
            pts = [(U, real_red(0.8, U, s)) for U in UU]
            xx = np.array([u for u, _ in pts]) + COL_DODGE[s]; m = np.array([v[0] for _, v in pts])
            lo = np.array([v[1] for _, v in pts]); hi = np.array([v[2] for _, v in pts])
            b.errorbar(xx, m, yerr=[m - lo, hi - m], color=c, mfc=c, capsize=1.5, elinewidth=0.7, capthick=0.7,
                       **COL_STY["Mix08"])
            b.plot([8 + COL_DODGE_D[s]], [real_red(0.0, 8, s)[0]], color=c, **COL_STY["Mix0"])
            one_tag(b, ONE_SCENE_TAG[s], s)
        b.axhline(0, color="0.35", lw=0.6, zorder=1, gid="ref")
        one_load_axis(b); b.set_ylim(-5, 64); col_yticks(b, [0, 20, 40, 60], ["0", "20", "40", "60"])
        b.set_ylabel("NMSE reduction (%)")
        hb = [Line2D([0], [0], color="k", label="Model, $\\bar\\kappa\\!=\\!0.8$", **COL_STY["Model08"]),
              Line2D([0], [0], color="k", label="Model, $\\bar\\kappa\\!=\\!0.4$", **COL_STY["Model04"]),
              Line2D([0], [0], color="k", label="Mixtures, $\\bar\\kappa\\!=\\!0.8$", **COL_STY["Mix08"]),
              Line2D([0], [0], color="k", label="Mixtures, $\\bar\\kappa\\!\\approx\\!0$", **COL_STY["Mix0"])]
        b.legend(handles=hb, loc="upper left", bbox_to_anchor=(0.0, 1.0 - ONE_LEG_TOP / (ONE_AH * 72)), ncol=1,
                 **{**ONE_LEGKW, "borderaxespad": 0.2})
        one_save(fig, "fig3_scene_one.pdf")


ONE_FIG2 = "nines"     # the Fig. 2 version of the manuscript (fig2_load_one.pdf); the other is fig2_alt.pdf. Why: the
                       # nines axis spreads the 10 and 20 dB curve sets over the height (the linear axis packs them
                       # into its top fifth), so at the 2.4 x 1.8 in box the printed Proposed-Learned gap at U = 2, 4, 6
                       # grows at 20 dB from 2.4-3.8 pt to 7.0-18.7 pt and the Static ODS-k segments separate at 10
                       # and 20 dB; at 10 dB it is 2.2-4.4 pt against 2.4-2.5 pt, and at 0 dB it shrinks from 10-15 pt
                       # to 4.5-4.7 pt, still more than the 4 pt marker (fig_load_one() prints the gaps).
                       # Over every load with a Learned point (U = 1..6, 8), the nines gap stays below the 4 pt
                       # marker at 10 dB for U = 4, 5, 6, 8 (1.9-2.9 pt) and at U = 8 for 0 dB (3.9 pt) and 20 dB
                       # (3.7 pt): there the circle touches the star and the text carries the margin (0.0055 to
                       # 0.047 over all loads and SNRs); the linear axis is below 4 pt at every 10 and 20 dB load.


def fig2_choose():
    """copy the chosen Fig. 2 version to fig2_load_one.pdf and the other to fig2_alt.pdf (PDF and PNG)."""
    import shutil
    other = "lin" if ONE_FIG2 == "nines" else "nines"
    for src, dst in ((ONE_FIG2, "fig2_load_one"), (other, "fig2_alt")):
        for ext in (".pdf", ".png"):
            shutil.copyfile(os.path.join(FIG, f"fig2_load_one_{src}{ext}"), os.path.join(FIG, dst + ext))
    print("Fig. 2 build:", ONE_FIG2, "| fig2_alt:", other)


def one_label_bg(fig):
    """a background behind every label of the region row, so that no gridline runs behind its text: white over the
    unshaded part of the label and ONE_SHADE over its part in the overload shading (the full-load label crosses the
    edge of the shading), ONE_BG_PAD pt around the text, above the grid and the boundary line (zorder 1.5) and below
    every curve (zorder 2 and up). Called once the layout is final; the guards measure the text, not this patch."""
    from matplotlib.patches import Rectangle
    r = col_pdf_layout(fig)
    k = fig.dpi / 72.0
    for ax in fig.axes:
        inv = ax.transAxes.inverted()
        xs = inv.transform(ax.transData.transform((K + 0.5, 0.0)))[0]        # edge of the shading (axes fraction)
        for t in [t for t in ax.texts if t.get_gid() in ("region", "boundary")]:
            e = t.get_window_extent(r).padded(ONE_BG_PAD * k)
            (x0, y0), (x1, y1) = inv.transform([[e.x0, e.y0], [e.x1, e.y1]])
            for a, b, fc in ((x0, min(x1, xs), "white"), (max(x0, xs), x1, ONE_SHADE)):
                if b > a:
                    ax.add_patch(Rectangle((a, y0), b - a, y1 - y0, transform=ax.transAxes, facecolor=fc, lw=0,
                                           zorder=1.6, gid="labelbg"))


def one_save(fig, name):
    for lg in [c for ax in fig.axes for c in ax.get_children() if isinstance(c, matplotlib.legend.Legend)]:
        lg.get_frame().set_linewidth(0.5)
    one_label_bg(fig)
    out = os.path.join(FIG, name)
    if os.environ.get("CL_DRAFT"):       # iteration only: write a draft PNG and PDF before the guard runs
        fig.savefig(out.replace(".pdf", "_draft.png"), dpi=300)
        fig.savefig(out.replace(".pdf", "_draft.pdf"), metadata={"CreationDate": None})
    col_guard(fig, leg_fs=ONE_FS_LEG)
    one_guard(fig)
    fig.savefig(out, metadata={"CreationDate": None})
    fig.savefig(out.replace(".pdf", ".png"), dpi=300)
    plt.close(fig)
    print("wrote", out)


if __name__ == "__main__":
    import sys
    if "--one" not in sys.argv:          # the letter figures and the two-panel column figures
        fig_load()
        fig_overload_only()
        fig_overload_coh()
        fig_overload()
        fig_load_col()
        fig_scene_col()
    fig_load_one("nines")
    fig_load_one("lin")
    fig2_choose()
    fig_scene_one()
