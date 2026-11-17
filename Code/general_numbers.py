# =========================================================
# general_numbers.py — the numbers of the manuscript and of its response letter about the real-valued masks,
# printed from the stored results into data/general_numbers.md (2026-11-17).
# Read-only: reads data/ only, runs no simulation, trains nothing, writes only data/general_numbers.md (and stdout).
# A missing input file is listed as a missing input (the script never fails on a missing file).
# --public: print without the optional local annotations (Code/general_numbers_notes.json, not distributed).
# Without that file the output is always the public one; every table is the same either way.
# Bootstrap convention (as Code/pca_csi.py, Code/shared_pilot.py and Code/fig_rev.py): paired 95% CIs over frames,
# 1000 resamples idx = numpy default_rng(555 + SNR).integers(0, F, (1000, F)), the same resamples for every scheme
# and metric of one point; CIs of a mean difference = percentiles of the resampled means of the per-frame
# differences; CIs of an NMSE reduction 1 - A/B = percentiles of 1 - sum_f A_f / sum_f B_f over the resampled frames
# (A_f, B_f = per-frame total squared error err_<mode>_<snr>.sum(1)); same-frame NMSE = sum err / sum ||c||^2, so
# the ratio of sums is the ratio of the NMSEs.
#
# Sections and inputs:
#   G  index (printed first): every section with its inputs, frames and seeds, and the list of missing inputs.
#   A  Load adaptivity, AG News, independent content, independent Rayleigh block fading (Fig. 2):
#        data/extra/pca_eval.json       proposal.opt[U][snr] = Proposed (PCA nested code, load masks, allocation (12),
#                                       blind-gain Wiener receiver), per-user CosSim, U = 1..6, 8; static[k][snr] =
#                                       Static ODS-k (the same closed-form code on k fixed parts, n = E = d_s/k, any
#                                       U <= k), k = 1..6, 8. 2000 frames per point and user, eval seed 123 + SNR.
#        data/extra/pca_eval_load.json  proposal_opt and static at U = k = 7 (also 6 and 8, checked against pca_eval).
#        data/ortho_U<k>_s<seed>.json   Learned ODS-U (learned partition trained for the load U), eval[U][snr].cos;
#                                       three seeds for U = 1..6, one run at U = 8, none at U = 7.
#   B  Natural COCO captions of one image (shared content), known gains (Code/shared_masks.py):
#        data/shared/shared_coco_K{4,2}_U{2..5}.json/.npz and every other shared_coco_* / shared_flickr_* file
#        without _mix (controls _ctrl/_ctrlfit, mismatched C _cf<N|other>, residual phase _ph<deg><abs|eff>, rank
#        rule _rank, dense random masks _rnd), plus shared_agnews_*_rnd (independent content). Schemes: load_pu = load
#        masks + per-user Wiener receiver; load_joint = load masks + joint LMMSE receiver with the content covariance;
#        gen_joint = overlapping masks of Proposition 2 + joint LMMSE; gen_eq = all energy in the common modes;
#        gen_k0 = Proposition 2 with kappa forced to 0 (dense random rotation); gen_pc = Proposition 2 masks with the
#        per-coefficient (A1) receiver; rnd = dense random masks + joint LMMSE. JSON: cos, weakest, nmse per mode and
#        SNR; kappa (768 per-coefficient kappa_i of the design C), fit (kappa_bar, group profile), design (uses,
#        overlap index t, pruning). npz: cos_<mode>_<snr> (F, U), err_<mode>_<snr> (F, U), nrm_<snr> (F, U).
#        Labels checked 2026-11-17: the load masks of B and C use the equal (spec) split, not the rank split of A
#        (they differ at K = 4, U = 5, 7 and K = 2, U = 5); gen_joint at U > K is the pruned Proposition 2; the gain
#        knowledge of every run is printed from its arguments (_ph<deg>abs: receivers use |h|, gains NOT known);
#        mixture kappa 0 has the fitted kappa_bar 0.002 (not C = 0), small reductions print with four decimals.
#        data/shared/content_stats.json (Code/content_stats.py): kappa_bar with its bootstrap CI over fit images,
#        kappa per 128-group and of the leading coefficients, controls of different images, mixture fits.
#   C  One scene from light load through overload, K = 4 (Fig. 3(a)):
#        data/shared/theory_coco_mix<k>_K4_U<U>.json (Code/shared_theory.py, Gaussian model with the fitted
#          statistics, exact NMSE, 100 Rayleigh draws default_rng(7)): results.full.rayleigh (the model of the
#          figure), results.a1.rayleigh (diagonal surrogate (A1): approximation gap), diag (pruning, active modes,
#          overlap index t_bar_c, decomposition b_over_a = receiver, d_over_b = pruned kappa = 0 design, c_over_d =
#          overlap), meta (kappa_bar of the fit, slope gains).
#        data/shared/shared_coco_mix<k>_K4_U<U>_ol.json/.npz (real caption mixtures b_u = normalize(sqrt(kappa) b_s
#          + sqrt(1 - kappa) b_{p,u}), 2000 frames, load_pu, load_joint, gen_joint): kappa 0.8 at U = 2..8, kappa 0 at
#          U = 8; shared_coco_mix0.8_K4_U4.json/.npz (non-_ol run of the same point, checked frame by frame).
#        kappa sweep at U = 4: theory_coco_mix{0,0.2,0.4,0.6,0.8,0.9,0.95}_K{2,4}_U4.json and the real mixtures
#          shared_coco_mix<k>_K{2,4}_U4.json/.npz (0 and 10 dB).
#        natural-caption model runs: theory_coco_K{2,4}_U{2..5}.json, theory_flickr_K{2,4}_U{2..5}.json, next to the
#          measured reductions of section B.
#   D  Coherence bandwidth and time (Code/pca_csi.py --coh, AG News, independent content, 2000 frames, N = 24 subbands
#        with AR(1) correlation rho_f, gains reported one block earlier with correlation eta, near-far offsets
#        uniform in +-6 dB where the name has _nf6, k = U):
#        data/pilot_corr/pca_csi_coh_rho<rho>_eta<eta>_U<U>[_nf6].json (+ .npz with cos_<mode>_<snr> (F, U)).
#        Modes: fixed = load masks (no CSI at the transmitter); csiq = channel-aware masks (greedy subband counts,
#        allocation (13)); switch = coherence-aware choice between csiq and fixed from the reported gains;
#        csiqp = csiq with the conditional-mean gains; ods = Static ODS-U placed by the same gains (Hungarian);
#        csiqb = channel-aware support with equal values; csiqw = channel-aware support with (12) at unit gain.
#        Energy: csiq family n_u x 128 per user (total d_s), fixed and ods d_s/U, printed in the D intro.
#        Stored per mode and SNR: cos, weakest, gain_fixed (mean CosSim minus fixed) with gain_fixed_ci, the same
#        for the weakest user; switch: frac_csiq, rule_loss with rule_loss_ci, decision_loss, oracle_cf_cos.
#        The stored CIs are recomputed from the npz here (check printed).
#   E  Search versus closed forms (Code/overlap_bound.py): data/extra/overlap_bound2_<content>_<gains>_K<K>_U<U>
#        [_nf6].json (restricted model, joint LMMSE receiver, Adam search from the closed form and 3 random starts,
#        3000 steps; equal gains: 1 draw; rayleigh-D20 / slot-rho0.7-D20: 20 gain draws) and the original
#        overlap_bound_U<U>_nf<X>.json (AG News, load masks vs any real overlapping masks, deterministic gains).
#        gap = 100 (closed - search) / closed of the sum NMSE (positive: the search is lower).
#   F  Gain estimation (Code/shared_pilot.py): data/shared/pilot/shared_pilot_<tag>[_snr<S>].json, tags of
#        Code/pilot_queue.sh.
# Usage: python3 Code/general_numbers.py [--public]      (about ten seconds on a laptop, CPU, numpy only)
# Testing only: GENERAL_NUMBERS_DATA=<dir> reads (and writes general_numbers.md into) another data tree, e.g. a
# scratch copy with files removed, to check that missing inputs are listed instead of failing.
# =========================================================
import glob, json, os, re, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
D = os.environ.get("GENERAL_NUMBERS_DATA") or os.path.join(os.path.dirname(HERE), "data")   # override: tests only
OUT = os.path.join(D, "general_numbers.md")
NBOOT = 1000
MISSING, ANOM, INDEX, MISM = [], [], [], []
_NOTES = os.path.join(HERE, "general_numbers_notes.json")      # optional local annotations, not distributed
NOTES = None if "--public" in sys.argv[1:] or not os.path.exists(_NOTES) else json.load(open(_NOTES, encoding="utf-8"))
PUBLIC = NOTES is None


def note(key, *args, public=""):
    """optional local annotation (formatted with args), or the public text."""
    return public if PUBLIC else NOTES[key].format(*args)


MISS = note("missing", public="missing input")
SEC = {}                      # section letter -> list of lines
_IDX = {}


def P(rel):
    return os.path.join(D, rel)


def ex(rel):
    return os.path.exists(P(rel))


def ldj(rel):
    return json.load(open(P(rel)))


def rel(f):
    return os.path.relpath(f, D)


def pend(w, sec, files, what):
    """record a missing input (one entry per call; a .json/.npz pair is one entry) and print it in place."""
    files = [files] if isinstance(files, str) else list(files)
    MISSING.append((sec, tuple(files), what))
    if w is not None:
        w(f"- {MISS}: {what}: " + ", ".join(f"data/{f}" for f in files))


def anom(sec, txt):
    ANOM.append((sec, txt))


def bidx(F, snr):
    key = (F, int(snr))
    if key not in _IDX:
        _IDX[key] = np.random.default_rng(555 + int(snr)).integers(0, F, (NBOOT, F))
    return _IDX[key]


def ci_mean(d, idx):
    m = d[idx].mean(1)
    return float(d.mean()), float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def ci_red(num, den, idx):
    """100 (1 - sum num / sum den) with the bootstrap CI of the ratio of sums over the resampled frames."""
    b = 1.0 - num[idx].sum(1) / den[idx].sum(1)
    return 100.0 * (1.0 - num.sum() / den.sum()), 100.0 * float(np.percentile(b, 2.5)), 100.0 * float(np.percentile(b, 97.5))


def fci(t, nd=4):
    return f"{t[0]:+.{nd}f} [{t[1]:+.{nd}f}, {t[2]:+.{nd}f}]"


def fpc(t, nd=2):
    return f"{t[0]:+.{nd}f} % [{t[1]:+.{nd}f}, {t[2]:+.{nd}f}]"


def fpca(t):
    """fpc with four decimals when every entry is below 0.05 in magnitude, so a small nonzero reduction never
    prints as 0.00 (e.g. the mixture kappa 0 rows, fitted kappa_bar 0.002, not C = 0)."""
    return fpc(t, 4 if max(abs(x) for x in t) < 0.05 else 2)


def fxa(x):
    """one percentage with four decimals below 0.05 in magnitude, two otherwise."""
    return f"{x:.4f}" if abs(x) < 0.05 else f"{x:.2f}"


def load_split(U, K, rule):
    """coefficients per user of the load masks: spec = equal split min(floor(d_s/U), d_b) (shared_masks.py and
    shared_theory.py default, sections B and C); rank = the manuscript's Proposed of section A and Fig. 2
    (pca_eval.per_load): user j sends min(ceil((N - j)/U) 128, d_b), N = 6K slots of 128."""
    d_s, N = K * 768, 6 * K
    if rule == "spec":
        return [min(d_s // U, 768)] * U
    return [min(-(-(N - j) // U) * 128, 768) for j in range(U)]


def rules_differ(K, Us):
    return [U for U in Us if load_split(U, K, "spec") != load_split(U, K, "rank")]


def gain_note(a):
    """what the receivers of a shared_masks.py run know about the gains (from its arguments)."""
    if a.get("pilot"):
        return ("pilot run: the overlapping masks use the pilot gain estimate, the load masks the effective gain "
                "(receivers do NOT know the gains of the overlapping masks)")
    if a.get("phase"):
        if a.get("rxgain") == "abs":
            return "receivers use |h|: the effective gain g cos(phi) is NOT known" + note("phase_abs")
        return "receivers use the effective gain g cos(phi) (effective gains known)"
    return "receivers know the gains"


def ovl(U, K, short=False):
    """label of a load U on a block of K d_b uses; at U > K the overlapping masks are the pruned Proposition 2."""
    if U > K:
        return " (overload, pruned Prop. 2)" if short else " (overload; gen_joint = pruned Proposition 2)"
    return " (full load)" if U == K else " (light load)"


def hdr(w, cols):
    w("| " + " | ".join(cols) + " |")
    w("|" + "---|" * len(cols))


def row(w, cells):
    w("| " + " | ".join(str(c) for c in cells) + " |")


# ================================================================== A. load adaptivity (AG News)
def sec_A():
    L = SEC.setdefault("A", []); w = L.append
    S = ["0", "5", "10", "15", "20", "25", "30"]
    SD = [f"{s} dB" for s in S]
    w("## A. Load adaptivity from light load through overload (AG News, independent content, Fig. 2)\n")
    w("Per-user CosSim, independent Rayleigh block fading, blind-gain receiver of the manuscript, d_s = K d_b with "
      "K = 4 (full load U = K = 4; light load U < 4; overload U > 4). Proposed = PCA nested code with the load masks "
      "and the allocation (12), one untrained transceiver for every load. Static ODS-k = the same closed-form code on "
      "k fixed parts (n = E = d_s/k), able to serve at most k users; for U <= k each served user gets the value of "
      "Static ODS-k at its design load (flat in U). Learned ODS-U = the learned partition retrained for the load U.\n")
    if not ex("extra/pca_eval.json"):
        pend(w, "A", "extra/pca_eval.json", "proposal and Static ODS-k (all of section A)")
        return
    E = ldj("extra/pca_eval.json")
    EL = ldj("extra/pca_eval_load.json") if ex("extra/pca_eval_load.json") else None
    pr, st = {}, {}
    for U in E["proposal"]["opt"]:
        pr[int(U)] = np.array([E["proposal"]["opt"][U][s] for s in S])
    for k in E["static"]:
        st[int(k)] = np.array([E["static"][k][s] for s in S])
    if EL is None:
        pend(w, "A", "extra/pca_eval_load.json", "Proposed and Static ODS-7 at U = 7")
    else:
        pr[7] = np.array([EL["proposal_opt"]["7"][s] for s in S])
        st[7] = np.array([EL["static"]["7"][s] for s in S])
        for U in ("6", "8"):
            if U in EL["proposal_opt"] and int(U) in pr:
                d1 = np.abs(np.array([EL["proposal_opt"][U][s] for s in S]) - pr[int(U)]).max()
                d2 = np.abs(np.array([EL["static"][U][s] for s in S]) - st[int(U)]).max()
                w(f"- check: pca_eval_load.json minus pca_eval.json at U = k = {U}, largest |difference| over the "
                  f"seven SNRs: Proposed {d1:.1e}, Static ODS-{U} {d2:.1e} (stored check value "
                  f"{EL.get('check_max_abs_diff_vs_pca_eval_json', float('nan')):.1e})")
                if max(d1, d2) > 1e-6:
                    anom("A", f"pca_eval_load.json differs from pca_eval.json at U = {U} by up to {max(d1, d2):.1e}")
    LO = {}
    for U in range(1, 9):
        fs = sorted(glob.glob(P(f"ortho_U{U}_s[0-9].json")))
        if fs:
            LO[U] = np.array([[json.load(open(f))["eval"][str(U)][s]["cos"] for s in S] for f in fs])
    w(f"- protocol: {E.get('protocol', '')}")
    w(f"- Learned ODS-U files: " + "; ".join(f"U = {U}: {len(v)} seed(s)" for U, v in sorted(LO.items()))
      + "; U = 7: no file data/ortho_U7_s*.json (not run, Fig. 2 draws no Learned ODS-7)")
    Us = sorted(pr)
    w("\n### A.1 Proposed, per-user CosSim versus the load U (K = 4)\n")
    hdr(w, ["load U"] + SD)
    for U in Us:
        row(w, [f"Proposed, U = {U}" + (" (pca_eval_load.json)" if U == 7 else "")] + [f"{x:.4f}" for x in pr[U]])
    w("\n### A.2 Static ODS-k, per-user CosSim of every served user (U <= k), K = 4\n")
    hdr(w, ["k"] + SD)
    for k in sorted(st):
        row(w, [f"Static ODS-{k}" + (" (pca_eval_load.json)" if k == 7 else "")] + [f"{x:.4f}" for x in st[k]])
    mono = all((st[k] >= st[k + 1] - 1e-12).all() for k in sorted(st) if k + 1 in st)
    w(f"\n- Static ODS-k decreases with k at every SNR (k = 1..8): {mono}; so the best fixed partition that serves "
      f"all U users at load U is Static ODS-U (upper envelope of the fixed partitions)")
    if not mono:
        anom("A", "Static ODS-k is not monotone in k")
    w("\n### A.3 Learned ODS-U at its load U: mean over seeds (sample SD, seeds)\n")
    hdr(w, ["load U"] + SD)
    for U in sorted(LO):
        v = LO[U]
        sd = v.std(0, ddof=1) if len(v) > 1 else np.full(len(S), np.nan)
        row(w, [f"Learned ODS-{U}"] + [f"{m:.4f} ({s_:.4f}, {len(v)})" if len(v) > 1 else f"{m:.4f} (-, 1)"
                                        for m, s_ in zip(v.mean(0), sd)])
    w("\n### A.4 Proposed minus Learned ODS-U at the same load U (CosSim)\n")
    hdr(w, ["load U"] + SD + ["min over 5-30 dB", "max over 5-30 dB", "min over 0-30 dB of Proposed minus the best seed"])
    allm, all5 = [], []
    for U in sorted(LO):
        if U not in pr:
            continue
        d = pr[U] - LO[U].mean(0)
        db = pr[U] - LO[U].max(0)
        allm += [(x, U, s) for x, s in zip(d, S)]
        all5 += [(x, U, s) for x, s in zip(d[1:], S[1:])]
        row(w, [f"U = {U} ({len(LO[U])} seed(s))"] + [f"{x:+.4f}" for x in d]
            + [f"{d[1:].min():+.4f}", f"{d[1:].max():+.4f}", f"{db.min():+.4f}"])
    if allm:
        mn, mx = min(allm), max(allm)
        w(f"\n- Proposed minus Learned ODS-U over U = {', '.join(str(U) for U in sorted(LO))} and 0-30 dB: "
          f"{mn[0]:+.4f} (U = {mn[1]}, {mn[2]} dB) to {mx[0]:+.4f} (U = {mx[1]}, {mx[2]} dB); from 5 dB on: "
          f"{min(all5)[0]:+.4f} (U = {min(all5)[1]}, {min(all5)[2]} dB) to {max(all5)[0]:+.4f}")
        w(f"- Proposed exceeds Learned ODS-U at every listed load and SNR: {mn[0] > 0}")
        if mn[0] <= 0:
            anom("A", f"Proposed is not above Learned ODS-U at U = {mn[1]}, {mn[2]} dB ({mn[0]:+.4f})")
    w("\n### A.5 Proposed minus Static ODS-U at its design load U = k (same code, same uses; differences = sampling "
      "and the slot granularity at U = 5 and 7)\n")
    hdr(w, ["U = k"] + SD + ["largest |difference|"])
    for U in Us:
        if U in st:
            d = pr[U] - st[U]
            row(w, [f"U = k = {U}"] + [f"{x:+.4f}" for x in d] + [f"{np.abs(d).max():.4f}"])
    dd = [(pr[U][i] - st[U][i], U, S[i]) for U in Us if U in st and U > 1 for i in range(len(S))]
    if dd:
        neg = [x for x in dd if x[0] < 0]
        w(f"\n- Proposed minus Static ODS-U at U = k = 2..8: {len(neg)} of {len(dd)} cells negative; range "
          f"{min(dd)[0]:+.4f} (U = {min(dd)[1]}, {min(dd)[2]} dB) to {max(dd)[0]:+.4f} (U = {max(dd)[1]}, {max(dd)[2]} dB): "
          f"equal within these values at the design load, not above")
        anom("A", f"Proposed minus Static ODS-U at its design load U = k is negative in {len(neg)} of {len(dd)} cells "
             f"(down to {min(dd)[0]:+.4f} at U = {min(dd)[1]}, {min(dd)[2]} dB)" + note("A_eq", abs(min(dd)[0])))
    w("\n### A.6 Margins over Static ODS-k below its design load (U < k): Proposed at load U minus Static ODS-k "
      "(fixed parts dimensioned for k users)\n")
    w("'Light load' " + note("light", public="means U < K = 4") + "; rows with U > 4 are overload of the block even "
      "though they lie below the design load k of Static ODS-6 or ODS-8.\n")
    hdr(w, ["k, U"] + SD + ["min", "max"])
    for k in (4, 6, 8):
        if k not in st:
            continue
        for U in range(1, k):
            if U in pr:
                d = pr[U] - st[k]
                row(w, [f"Static ODS-{k}, U = {U}" + (" (overload of K = 4)" if U > 4 else "")]
                    + [f"{x:+.4f}" for x in d] + [f"{d.min():+.4f}", f"{d.max():+.4f}"])
    w("\n### A.7 Overload, U = 5..8 on the block dimensioned for K = 4\n")
    hdr(w, ["quantity, U"] + SD)
    for U in (5, 6, 7, 8):
        if U in pr:
            row(w, [f"Proposed CosSim, U = {U}"] + [f"{x:.4f}" for x in pr[U]])
    for U in (5, 6, 7, 8):
        if U in pr and U in LO:
            row(w, [f"Proposed minus Learned ODS-{U} ({len(LO[U])} seed(s))"] + [f"{x:+.4f}" for x in pr[U] - LO[U].mean(0)])
        elif U in pr:
            row(w, [f"Proposed minus Learned ODS-{U}"] + ["no run"] * len(S))
    for U in (5, 6, 7, 8):
        if U in pr and U in st:
            row(w, [f"Proposed minus Static ODS-{U} (U = k)"] + [f"{x:+.4f}" for x in pr[U] - st[U]])
    for U in (5, 6, 7):
        if U in pr and 8 in st:
            row(w, [f"Proposed minus Static ODS-8, U = {U}"] + [f"{x:+.4f}" for x in pr[U] - st[8]])
    ov = [(x, U, s) for U in (5, 6, 7, 8) if U in pr and U in LO for x, s in zip(pr[U] - LO[U].mean(0), S)]
    if ov:
        mn = min(ov)
        mn5 = min(o for o in ov if o[2] != "0")
        w(f"\n- minimum margin of Proposed over Learned ODS-U in overload (U = "
          f"{', '.join(str(U) for U in (5, 6, 7, 8) if U in LO)}; none at U = 7), 0-30 dB: {mn[0]:+.4f} "
          f"(U = {mn[1]}, {mn[2]} dB); from 5 dB on: {mn5[0]:+.4f} (U = {mn5[1]}, {mn5[2]} dB); largest "
          f"{max(ov)[0]:+.4f} (U = {max(ov)[1]}, {max(ov)[2]} dB)")
    w("- Static ODS-k with k < U serves only k of the U users (U - k users get nothing); e.g. Static ODS-4 at U = 8 "
      "serves 4 of 8. Proposed serves all U users at every load.")
    for U in (1, 2, 3, 4, 5, 6, 7, 8):
        if U in pr:
            w(f"- Proposed, U = {U}, at 0 / 10 / 20 dB: {pr[U][0]:.4f} / {pr[U][2]:.4f} / {pr[U][4]:.4f}; "
              f"change from U = 1 (Proposed at U minus Proposed at U = 1; negative = drop): "
              f"{pr[U][0] - pr[1][0]:+.4f} / {pr[U][2] - pr[1][2]:+.4f} / {pr[U][4] - pr[1][4]:+.4f}")
    INDEX.append(("A", "extra/pca_eval.json, extra/pca_eval_load.json, ortho_U<k>_s*.json",
                  "2000 frames per point and user (pca_eval), eval seed 123 + SNR; Learned ODS-U seeds s1-s3 (U = 1..6), s1 (U = 8)"))


# ================================================================== B. natural captions
VAR = [("_ctrlfit", "control: captions of U distinct images (true C = 0), design and receiver use the fitted C (mismatch)"),
       ("_ctrl", "control: captions of U distinct images (true C = 0), design and receiver use C = 0"),
       ("_cf100", "design and receiver C fitted on the first 100 fit images only"),
       ("_cf1000", "design and receiver C fitted on the first 1000 fit images only"),
       ("_cfother", "design and receiver C fitted on the other corpus (Flickr30k) in the COCO basis"),
       ("_ph10abs", "residual phase N(0, (10 deg)^2), receivers use |h|"),
       ("_ph10eff", "residual phase N(0, (10 deg)^2), receivers use g cos(phi)"),
       ("_ph30abs", "residual phase N(0, (30 deg)^2), receivers use |h|"),
       ("_ph30eff", "residual phase N(0, (30 deg)^2), receivers use g cos(phi)"),
       ("_rank", "load masks with the rank rule of the manuscript (user j sends min(ceil((N - j)/U) 128, d_b)), load modes only"),
       ("_rnd", "dense random masks (rnd) next to the load masks and Proposition 2, 500 frames")]
EXTRA_MODES = ("gen_eq", "gen_k0", "gen_pc", "rnd")


def variant_desc(a):
    """one-line description of a shared_masks.py run from its arguments."""
    p = []
    ds = {"coco": "COCO captions of one image", "flickr": "Flickr30k captions of one image",
          "agnews": "AG News, independent content (C = 0)"}[a["dataset"]]
    if a.get("control"):
        p.append("control: captions of U distinct images (true C = 0), design and receiver use "
                 + ("C = 0" if a.get("control_c") == "zero" else "the fitted C (mismatch)"))
    cf = str(a.get("c_from", "self"))
    if cf in ("other", "other_corpus"):
        p.append("design and receiver C fitted on the other caption corpus, in this corpus' basis (mismatch)")
    elif cf.isdigit():
        p.append(f"design and receiver C fitted on the first {cf} fit images only (estimation noise)")
    if a.get("phase"):
        p.append(f"residual phase N(0, ({a['phase']:g} deg)^2), receivers use "
                 + ("|h| (phase not known)" if a.get("rxgain") == "abs" else "the effective gain g cos(phi)"))
    if a.get("load_rule") == "rank":
        p.append("load masks with the rank rule of the manuscript (user j sends min(ceil((N - j)/U) 128, d_b)), load modes only")
    if "rnd" in a.get("modes", []):
        p.append("dense random masks (rnd) next to the load masks and Proposition 2")
    if a.get("nearfar"):
        p.append(f"near-far offsets +-{a['nearfar']:g} dB")
    if a.get("pilot"):
        p.append("pilot-charged overlapping masks")
    over = "; overload, gen_joint = pruned Proposition 2" if a["U"] > a["K"] else ""
    return f"{ds}, K = {a['K']}, U = {a['U']}" + over + ("; " + "; ".join(p) if p else "")


def shared_block(w, fjs, sec, title, full=True):
    """tables of one shared_masks.py run: per-mode metrics, paired differences with CIs, design info."""
    r = json.load(open(fjs))
    fnz = fjs[:-5] + ".npz"
    a = r["args"]
    F = r["frames"]
    snrs = [int(s) for s in r["res"][list(r["res"])[0]]]
    modes = list(r["res"])
    fi = r["fit"]
    w(f"\n#### {title}: data/{rel(fjs)}\n")
    kap = np.array(r["kappa"]) if r.get("kappa") is not None else None
    w(f"- dataset {a['dataset']}" + (f", mixture kappa {a['mix']:g}" if a.get("mix") is not None else "")
      + f", K = {a['K']}, U = {a['U']}, {F} frames, gains {a.get('gains', 'rayleigh')}"
      + (f", near-far +-{a['nearfar']:g} dB" if a.get("nearfar") else "") + ", SNRs " + "/".join(str(s) for s in snrs)
      + " dB, " + gain_note(a) + f"; load masks: {a.get('load_rule', 'spec')} rule ("
      + "/".join(str(x) for x in load_split(a["U"], a["K"], a.get("load_rule", "spec"))) + " coefficients per user)")
    if kap is not None:
        grp = [kap[g * 128:(g + 1) * 128].mean() for g in range(6)]
        w(f"- content statistics of the design C (fit): kappa_bar = {fi.get('kappa_bar', float('nan')):.4f}"
          f" (tr C_model {fi.get('trC_model', float('nan')):.4f}, tr C of the content {fi.get('trC_self', float('nan')):.4f}, "
          f"model is the true content: {fi.get('model_is_true')}); kappa_1 = {kap[0]:.4f}; lam-weighted kappa per "
          f"128-group 1..6: " + "/".join(f"{x:.3f}" for x in fi.get("kappa_group_lamw", [])) +
          "; mean kappa_i per group: " + "/".join(f"{x:.3f}" for x in grp)
          + "; kappa_i of coefficients 1..10: " + "/".join(f"{x:.3f}" for x in kap[:10]))
    if full:
        w("")
        hdr(w, ["mode", "metric"] + [f"{s} dB" for s in snrs])
        for m in modes:
            for key, lab in (("cos", "CosSim"), ("weakest", "weakest user CosSim"), ("nmse", "NMSE")):
                row(w, [m, lab] + [f"{r['res'][m][str(s)][key]:.4f}" if str(s) in r["res"][m] else "-" for s in snrs])
    if not os.path.exists(fnz):
        pend(w, sec, rel(fnz), "per-frame arrays (CIs)")
        return r, None
    z = np.load(fnz)
    out = {}
    w("")
    cols = ["SNR"]
    if "gen_joint" in modes:
        cols += ["gen_joint - load_joint CosSim [95% CI]", "gen_joint - load_joint weakest [95% CI]",
                 "overlap NMSE reduction 1 - gen_joint/load_joint [95% CI]"]
    if "load_joint" in modes and "load_pu" in modes:
        cols += ["load_joint - load_pu CosSim [95% CI]", "receiver NMSE reduction 1 - load_joint/load_pu [95% CI]"]
    if "gen_joint" in modes and "load_pu" in modes:
        cols += ["total NMSE reduction 1 - gen_joint/load_pu [95% CI]"]
    hdr(w, cols)
    for s in snrs:
        idx = bidx(F, s)
        c = {m: z[f"cos_{m}_{s}"].astype(np.float64) for m in modes if f"cos_{m}_{s}" in z}
        e = {m: z[f"err_{m}_{s}"].astype(np.float64).sum(1) for m in modes if f"err_{m}_{s}" in z}
        cells = [f"{s} dB"]
        o = {}
        if "gen_joint" in c:
            o["dcos_gl"] = ci_mean(c["gen_joint"].mean(1) - c["load_joint"].mean(1), idx)
            o["dwk_gl"] = ci_mean(c["gen_joint"].min(1) - c["load_joint"].min(1), idx)
            o["red_gl"] = ci_red(e["gen_joint"], e["load_joint"], idx)
            cells += [fci(o["dcos_gl"]), fci(o["dwk_gl"]), fpc(o["red_gl"])]
        if "load_joint" in c and "load_pu" in c:
            o["dcos_lp"] = ci_mean(c["load_joint"].mean(1) - c["load_pu"].mean(1), idx)
            o["red_lp"] = ci_red(e["load_joint"], e["load_pu"], idx)
            cells += [fci(o["dcos_lp"]), fpc(o["red_lp"])]
        if "gen_joint" in c and "load_pu" in c:
            o["red_gp"] = ci_red(e["gen_joint"], e["load_pu"], idx)
            cells += [fpc(o["red_gp"])]
        row(w, cells)
        out[s] = o
        if "red_gl" in o and o["red_gl"][2] < 0:
            if fi.get("model_is_true", True):
                anom(sec, f"{rel(fjs)} {s} dB: overlap NMSE reduction CI entirely below 0 ({fpc(o['red_gl'])})")
            else:
                MISM.append((rel(fjs), s, o["red_gl"]))
    ext = [m for m in EXTRA_MODES if m in modes]
    if ext and full and "load_joint" in modes:
        w("")
        hdr(w, ["SNR"] + [f"{m} - load_joint CosSim [95% CI]; NMSE reduction 1 - {m}/load_joint [95% CI]" for m in ext])
        for s in snrs:
            idx = bidx(F, s)
            cl = z[f"cos_load_joint_{s}"].astype(np.float64).mean(1)
            el = z[f"err_load_joint_{s}"].astype(np.float64).sum(1)
            cells = [f"{s} dB"]
            for m in ext:
                cm = z[f"cos_{m}_{s}"].astype(np.float64).mean(1)
                em = z[f"err_{m}_{s}"].astype(np.float64).sum(1)
                cells.append(fci(ci_mean(cm - cl, idx)) + "; " + fpc(ci_red(em, el, idx)))
            row(w, cells)
    dg = r.get("design", {})
    if dg and full:
        w("")
        hdr(w, ["SNR", "load masks: uses / coefficients sent", "gen_joint: uses", "gen_joint overlap index t",
                "gen_joint t_overlap_modes", "modes active / pruned", "partial coefficients", "n_data"])
        for s in snrs:
            d = dg.get(str(s), {})
            ld_, gj = d.get("load", {}), d.get("gen_joint", {})
            row(w, [f"{s} dB", f"{ld_.get('n_uses', '-')} / {ld_.get('n_coef_active', '-')}", gj.get("n_uses", "-"),
                    f"{gj['t_overlap']:.4f}" if "t_overlap" in gj else "-",
                    f"{gj['t_overlap_modes']:.4f}" if gj.get("t_overlap_modes") is not None else "-",
                    f"{gj.get('n_modes_active', '-')} / {gj.get('n_pruned', '-')}", gj.get("n_partial_coef", "-"),
                    gj.get("n_data", "-")])
    return r, out


def sec_B():
    L = SEC.setdefault("B", []); w = L.append
    w("## B. Natural COCO captions of one image (shared content), known gains (Code/shared_masks.py)\n")
    w("Every user of a frame sends a different caption of the same test image (natural shared content), independent "
      "Rayleigh block fading, the receiver knows the gains (except the residual-phase runs _ph<deg>abs of B.3, whose "
      "receivers use |h| and do not know the effective gain g cos(phi)), per-user energy d_s/U. Paired differences on "
      "the same frames, captions, gains and noise. Overlap NMSE reduction = gain of the overlapping masks of "
      "Proposition 2 (pruned Proposition 2 at U > K, at most d_s active modes) over the load masks, both with the joint "
      "LMMSE receiver; receiver NMSE reduction = gain of the joint receiver over the per-user receiver, both with the "
      "load masks (this part is a receiver gain, not a mask gain).\n")
    w(f"Load masks of sections B and C (load_pu, load_joint): the equal split, every user sends min(floor(d_s/U), d_b) "
      f"coefficients (--load_rule spec). The Proposed of section A and Fig. 2 uses the rank split (user j sends "
      f"min(ceil((N - j)/U) 128, d_b)). The two differ at K = 4, U = "
      f"{', '.join(str(U) for U in rules_differ(4, range(1, 9)))} and at K = 2, U = "
      f"{', '.join(str(U) for U in rules_differ(2, range(1, 6)))} (e.g. K = 4, U = 5: spec "
      f"{'/'.join(str(x) for x in load_split(5, 4, 'spec'))} = {sum(load_split(5, 4, 'spec'))} uses, rank "
      f"{'/'.join(str(x) for x in load_split(5, 4, 'rank'))} = {sum(load_split(5, 4, 'rank'))} uses) and coincide "
      "elsewhere; the runs with suffix _rank (B.3) use the rank split.\n")
    req = [f"shared/shared_coco_K{K}_U{U}.json" for K in (4, 2) for U in (2, 3, 4, 5)]
    for f in req:
        if not ex(f):
            pend(w, "B", f, "natural-caption run")
    fs = sorted(set(glob.glob(P("shared/shared_coco_*.json")) + glob.glob(P("shared/shared_flickr_*.json"))))
    fs = [f for f in fs if "_mix" not in os.path.basename(f)]
    base = [P(f) for f in req if ex(f)]
    rest = [f for f in fs if f not in base]
    nat = {}
    w("### B.1 Natural captions, K = 4 and K = 2, U = 2..5 (U > K is overload of the block)\n")
    for f in base:
        nm = os.path.basename(f)[7:-5]
        K, U = map(int, re.match(r"coco_K(\d)_U(\d)", nm).groups())
        r, o = shared_block(w, f, "B", f"COCO natural captions, K = {K}, U = {U}" + ovl(U, K))
        nat[(K, U)] = (r, o)
    w("\n### B.2 Summary: overlap and receiver NMSE reductions of the natural captions (paired bootstrap 95% CI)\n")
    snr_all = sorted({s for (K, U), (r, o) in nat.items() if o for s in o})
    hdr(w, ["K, U", "quantity"] + [f"{s} dB" for s in snr_all])
    for (K, U), (r, o) in sorted(nat.items(), key=lambda t: (-t[0][0], t[0][1])):
        if not o:
            continue
        for key, lab in (("red_gl", "overlap 1 - gen_joint/load_joint"), ("red_lp", "receiver 1 - load_joint/load_pu"),
                         ("dcos_gl", "CosSim gen_joint - load_joint"), ("dcos_lp", "CosSim load_joint - load_pu")):
            row(w, [f"K = {K}, U = {U}" + (ovl(U, K, True) if U > K else ""), lab]
                + [(fpc(o[s][key]) if key.startswith("red") else fci(o[s][key]))
                                                 if s in o and key in o[s] else "-" for s in snr_all])
    w("\n### B.3 Variants and controls (same point unless stated)\n")
    ag = sorted(glob.glob(P("shared/shared_agnews_*.json")))
    for f in rest + ag:
        shared_block(w, f, "B", variant_desc(json.load(open(f))["args"]))
    if MISM:
        w("\n- mismatched-design runs (design C differs from the true content) with an overlap NMSE reduction CI entirely "
          "below 0: " + "; ".join(f"{f} {s} dB {fpc(t)}" for f, s, t in MISM))
        fl = sorted({f for f, s, t in MISM})
        anom("B", "expected sign, not an error: the control with the fitted C on captions of different images "
             f"({', '.join(fl)}) loses with the overlapping masks at every SNR (worst {fpc(min(MISM, key=lambda t: t[2][0])[2])} at "
             f"{min(MISM, key=lambda t: t[2][0])[1]} dB): overlap pays only when the content is really shared")
    # content_stats.json
    w("\n### B.4 Content statistics (data/shared/content_stats.json, Code/content_stats.py)\n")
    INDEX.append(("B", "shared/shared_coco_K{4,2}_U{2..5}, other shared_coco_*/shared_flickr_* (no _mix), shared_agnews_*_rnd, content_stats.json",
                  "2000 frames (COCO), 1000 (Flickr), 500 (_rnd); captions default_rng(123 + SNR + 1000 U), gains 5123 + SNR + 1000 U, "
                  "noise 9123 + SNR + 1000 U; bootstrap 555 + SNR"))
    if not ex("shared/content_stats.json"):
        pend(w, "B", "shared/content_stats.json", "content statistics")
        return nat
    C = ldj("shared/content_stats.json")
    for ds in ("coco", "flickr"):
        if ds not in C:
            continue
        si, ts, cd = C[ds]["same_image"], C[ds].get("test_same_image", {}), C[ds].get("control_diff_images", {})
        b = si.get("bootstrap", {})
        w(f"- {ds}, five captions of one fit image ({si['n_tuples']} images): kappa_bar = {si['kappa_bar']:.4f} "
          f"[{b.get('kappa_bar_ci95', [np.nan, np.nan])[0]:.4f}, {b.get('kappa_bar_ci95', [np.nan, np.nan])[1]:.4f}] "
          f"(bootstrap over fit images, {b.get('resamples', '')} resamples); test images ({ts.get('n_images', '-')}): "
          f"{ts.get('kappa_bar', np.nan):.4f} [{ts.get('kappa_bar_ci95', [np.nan, np.nan])[0]:.4f}, "
          f"{ts.get('kappa_bar_ci95', [np.nan, np.nan])[1]:.4f}]; kappa_1 = {si['kappa_1']:.4f}")
        w(f"  - lam-weighted kappa per 128-group 1..6: " + "/".join(f"{x:.3f}" for x in si["kappa_group_lamw"])
          + "; mean kappa_i per group: " + "/".join(f"{x:.3f}" for x in si["kappa_group_mean"])
          + "; kappa_i of coefficients 1..10: " + "/".join(f"{x:.3f}" for x in si["kappa_lead10"]))
        w(f"  - share of tr C in group 1: {si['trC_share_group1']:.3f}; share of ||C||_F^2 in block 1: "
          f"{si['frob_share_block1']:.4f}; off-diagonal energy share of C: {si['offdiag'].get('share_sq', np.nan):.3f} "
          f"(noise corrected {si.get('offdiag_share_sq_noise_corrected', np.nan):.3f}); spectrum participation ratio "
          f"{C[ds]['spectrum']['participation_ratio']:.1f} of 768, energy in group 1 {C[ds]['spectrum']['group_energy'][0]:.3f}")
        if cd:
            w(f"  - control, captions of five different images: kappa_bar = {cd['kappa_bar']:.4f} "
              f"[{cd['bootstrap']['kappa_bar_ci95'][0]:.4f}, {cd['bootstrap']['kappa_bar_ci95'][1]:.4f}]; "
              f"||C_ctrl||_F / ||C_same||_F = {cd.get('frob_ratio_to_same', np.nan):.3f}")
    mx = C.get("mix", {})
    if mx:
        w(f"- caption mixtures (coco, U = {mx.get('U')} users per tuple, fit {mx.get('mix_fit')}): kappa_bar of the fit "
          "[bootstrap 95% CI]; lam-weighted kappa groups 1..3; ||C - kappa Lam||_F / ||kappa Lam||_F")
        for k, v in mx["fits"].items():
            bb = v.get("bootstrap", {})
            ci = bb.get("kappa_bar_ci95", [np.nan, np.nan])
            w(f"  - kappa {k}: kappa_bar = {v['kappa_bar']:.4f} [{ci[0]:.4f}, {ci[1]:.4f}]; groups "
              + "/".join(f"{x:.3f}" for x in v["kappa_group_lamw"][:3])
              + "; rel. deviation " + (f"{v['rel_dev_C_from_kappaLam']:.4f}" if v.get("rel_dev_C_from_kappaLam") is not None else "-"))
    return nat


# ================================================================== C. one scene through overload
def th_at(t, model, scheme, snr, gains="rayleigh"):
    sn = [float(x) for x in t["meta"]["snrs"]]
    if float(snr) not in sn:
        return None
    return t["results"][model][gains][scheme][sn.index(float(snr))]


def th_diag(t, snr):
    sn = [float(x) for x in t["meta"]["snrs"]]
    return t["diag"][sn.index(float(snr))] if float(snr) in sn else None


def th_red(t, snr, model="full"):
    g, l, p = (th_at(t, model, m, snr) for m in ("gen_joint", "load_joint", "load_pu"))
    if g is None:
        return None
    return {"pu": p["nmse"], "lj": l["nmse"], "gj": g["nmse"], "ov": 100 * (1 - g["nmse"] / l["nmse"]),
            "rx": 100 * (1 - l["nmse"] / p["nmse"]), "tot": 100 * (1 - g["nmse"] / p["nmse"]),
            "se_ov": 100 * g["minus_load_joint"]["se"] / l["nmse"] if g["minus_load_joint"].get("se") else None,
            "cos_pu": p["cos_proxy"], "cos_lj": l["cos_proxy"], "cos_gj": g["cos_proxy"]}


def real_pt(rel_json):
    """per SNR: NMSE of the three schemes, reductions and CosSim differences with paired bootstrap CIs."""
    if not ex(rel_json) or not ex(rel_json[:-5] + ".npz"):
        return None
    r = ldj(rel_json)
    z = np.load(P(rel_json[:-5] + ".npz"))
    F = r["frames"]
    res = {}
    for s in [int(x) for x in r["res"]["load_joint"]]:
        idx = bidx(F, s)
        c = {m: z[f"cos_{m}_{s}"].astype(np.float64) for m in ("load_pu", "load_joint", "gen_joint")}
        e = {m: z[f"err_{m}_{s}"].astype(np.float64).sum(1) for m in ("load_pu", "load_joint", "gen_joint")}
        res[s] = {"nmse": {m: r["res"][m][str(s)]["nmse"] for m in c}, "cos": {m: r["res"][m][str(s)]["cos"] for m in c},
                  "weakest": {m: r["res"][m][str(s)]["weakest"] for m in c},
                  "ov": ci_red(e["gen_joint"], e["load_joint"], idx), "rx": ci_red(e["load_joint"], e["load_pu"], idx),
                  "tot": ci_red(e["gen_joint"], e["load_pu"], idx),
                  "dc_gl": ci_mean(c["gen_joint"].mean(1) - c["load_joint"].mean(1), idx),
                  "dc_lp": ci_mean(c["load_joint"].mean(1) - c["load_pu"].mean(1), idx),
                  "dw_gl": ci_mean(c["gen_joint"].min(1) - c["load_joint"].min(1), idx),
                  "design": r.get("design", {}).get(str(s), {})}
    return {"F": F, "kappa_bar": r["fit"].get("kappa_bar"), "res": res}


def sec_C(nat):
    L = SEC.setdefault("C", []); w = L.append
    w("## C. Observers of one scene from light load through overload, K = 4 (Fig. 3(a))\n")
    w("Overlapping masks of Proposition 2 (gen_joint) against the load masks (load_joint), both with the joint LMMSE "
      "receiver and KNOWN gains; receiver gain = load_pu -> load_joint (load masks, per-user Wiener -> joint LMMSE). "
      "Full load U = K = 4 (d_s = 3072 uses); U > 4 is overload. Model = Gaussian model with the fitted statistics "
      "(results.full.rayleigh, exact NMSE, mean over 100 Rayleigh draws; SE = paired Monte Carlo standard error over "
      "the draws). Real captions = caption mixtures with mean content correlation kappa_bar, 2000 frames, paired "
      "bootstrap 95% CIs.\n")
    w("- overlapping masks at U > K: gen_joint is the PRUNED Proposition 2 (greedy pruning to at most d_s active modes, "
      "partial coefficients; prop2_pruned of section E), labelled 'overload, pruned Prop. 2' in every row below. "
      "Pruning acts only where more than d_s modes would be active (pruned modes > 0 in C.1c and C.2c; the "
      "SNRs are listed below); at lower SNR the design equals plain Proposition 2.")
    w(f"- load masks = equal split min(floor(d_s/U), d_b) per user (--load_rule spec); at U = "
      f"{', '.join(str(U) for U in rules_differ(4, range(2, 9)))} (K = 4) this differs from the rank split of the "
      f"Proposed of Fig. 2 (U = 5: spec {'/'.join(str(x) for x in load_split(5, 4, 'spec'))}, rank "
      f"{'/'.join(str(x) for x in load_split(5, 4, 'rank'))}; U = 7: spec {load_split(7, 4, 'spec')[0]} each, rank "
      f"{'/'.join(str(x) for x in load_split(7, 4, 'rank'))}). So at U = 5 and 7 the Fig. 3(a) reference is NOT "
      "'the proposal of Fig. 2'; it coincides with it at U = 2, 3, 4, 6, 8.")
    if ex("shared/shared_coco_K4_U5_rank.json") and ex("shared/shared_coco_K4_U5.json") \
            and ex("shared/shared_coco_K4_U5_rank.npz") and ex("shared/shared_coco_K4_U5.npz") \
            and ldj("shared/shared_coco_K4_U5_rank.json")["frames"] == ldj("shared/shared_coco_K4_U5.json")["frames"]:
        ra, rb = ldj("shared/shared_coco_K4_U5_rank.json"), ldj("shared/shared_coco_K4_U5.json")
        za, zb = np.load(P("shared/shared_coco_K4_U5_rank.npz")), np.load(P("shared/shared_coco_K4_U5.npz"))
        cells = []
        for s in [int(x) for x in ra["res"]["load_joint"]]:
            if str(s) not in rb["res"]["load_joint"] or f"err_load_joint_{s}" not in za or f"err_load_joint_{s}" not in zb:
                continue
            idx = bidx(ra["frames"], s)
            t = ci_red(za[f"err_load_joint_{s}"].astype(np.float64).sum(1), zb[f"err_load_joint_{s}"].astype(np.float64).sum(1), idx)
            cells.append(f"{s} dB {ra['res']['load_joint'][str(s)]['nmse']:.4f} / {rb['res']['load_joint'][str(s)]['nmse']:.4f} "
                         f"(1 - rank/spec {fpca(t)})")
        w("- rank against spec split, natural COCO captions, K = 4, U = 5 (B.3 shared_coco_K4_U5_rank against "
          "shared_coco_K4_U5, same frames), load_joint NMSE rank / spec: " + "; ".join(cells))
    UU = (2, 3, 4, 5, 6, 7, 8)
    TH = {}
    for mix in (0.8, 0.4):
        for U in UU:
            f = f"shared/theory_coco_mix{mix:g}_K4_U{U}.json"
            if ex(f):
                TH[(mix, U)] = ldj(f)
    RL = {}
    for U in UU:
        f = f"shared/shared_coco_mix0.8_K4_U{U}_ol.json"
        RL[(0.8, U)] = real_pt(f)
    RL[(0.0, 8)] = real_pt("shared/shared_coco_mix0_K4_U8_ol.json")
    pr_ = []
    for (mix, U), t in sorted(TH.items()):
        if U > 4:
            sn = [f"{float(s):g}" for s, d in zip(t["meta"]["snrs"], t["diag"]) if d.get("drops_c", 0) > 0]
            pr_.append(f"model kappa {mix:g}, U = {U}: " + ("/".join(sn) + " dB" if sn else "none"))
    for (mix, U), rr in sorted(RL.items()):
        if rr and U > 4:
            sn = [str(s) for s, v in sorted(rr["res"].items()) if (v["design"].get("gen_joint", {}).get("n_pruned") or 0) > 0]
            pr_.append(f"captions kappa {mix:g}, U = {U}: " + ("/".join(sn) + " dB" if sn else "none"))
    if pr_:
        w("- SNRs at which the pruning removes modes (U > K): " + "; ".join(pr_))
    # ---- C.0 compact table of the figure
    w("\n### C.0 Fig. 3(a) at a glance: overlap NMSE reduction 1 - gen_joint/load_joint (%)\n")
    hdr(w, ["U", "0 dB model kb 0.8", "0 dB model kb 0.4", "0 dB captions kb 0.8 [CI]", "10 dB model kb 0.8",
            "10 dB model kb 0.4", "10 dB captions kb 0.8 [CI]", "20 dB model kb 0.8", "20 dB model kb 0.4",
            "20 dB captions kb 0.8 [CI]"])
    for U in UU:
        cells = [f"U = {U}" + (ovl(U, 4, True) if U > 4 else (" (full)" if U == 4 else " (light)"))]
        for s in (0, 10, 20):
            for mix in (0.8, 0.4):
                t = TH.get((mix, U))
                v = th_red(t, s) if t else None
                cells.append(fxa(v["ov"]) if v else MISS)
            rr = RL.get((0.8, U))
            cells.append(fpca(rr["res"][s]["ov"]) if rr and s in rr["res"] else MISS)
        row(w, cells)
    w("\n(values below 0.05 % print with four decimals, so a small nonzero reduction never reads as 0.00)")
    rr0 = RL.get((0.0, 8))
    kb0 = None
    if ex("shared/content_stats.json"):
        kb0 = ldj("shared/content_stats.json").get("mix", {}).get("fits", {}).get("0.0", {}).get("kappa_bar")
    lab0 = ("mixture kappa 0 (captions of different images; the design and the receiver use the FITTED C, "
            + (f"kappa_bar {kb0:.4f}" if kb0 is not None else "fitted kappa_bar") + ", not C = 0)")
    w(f"\n- captions with {lab0}, U = 8: "
      + ("; ".join(f"{s} dB {fpca(rr0['res'][s]['ov'])}" for s in sorted(rr0["res"])) if rr0 else MISS)
      + "; this is the same kind of point as the mixture kappa 0 rows of C.3 (fitted C, not C = 0)" + note("C_mix0_U8"))
    for U in UU:
        if (0.8, U) not in TH:
            pend(w, "C", f"shared/theory_coco_mix0.8_K4_U{U}.json", f"model kappa_bar 0.8, U = {U}")
        if (0.4, U) not in TH:
            pend(w, "C", f"shared/theory_coco_mix0.4_K4_U{U}.json", f"model kappa_bar 0.4, U = {U}")
        if RL.get((0.8, U)) is None:
            pend(w, "C", [f"shared/shared_coco_mix0.8_K4_U{U}_ol.json", f"shared/shared_coco_mix0.8_K4_U{U}_ol.npz"],
                 f"real captions kappa_bar 0.8, U = {U}")
    if rr0 is None:
        pend(w, "C", ["shared/shared_coco_mix0_K4_U8_ol.json", "shared/shared_coco_mix0_K4_U8_ol.npz"],
             "real captions kappa_bar 0, U = 8")
    w("\n- " + note("C0_label", public="summary") + " lines (overlap NMSE reduction 1 - gen_joint/load_joint, K = 4, "
      "known gains):")
    for s in (0, 10, 20):
        for mix in (0.8, 0.4):
            vals = [(U, th_red(TH[(mix, U)], s)) for U in UU if (mix, U) in TH]
            w(f"  - model kappa_bar {mix:g}, {s} dB: " + " / ".join(f"{v['ov']:.2f}" for U, v in vals) + " % at U = "
              + " / ".join(str(U) for U, v in vals) + "; receiver reduction (load_pu -> load_joint) "
              + " / ".join(f"{v['rx']:.2f}" for U, v in vals) + " %")
        vals = [(U, RL[(0.8, U)]["res"][s]) for U in UU if RL.get((0.8, U)) and s in RL[(0.8, U)]["res"]]
        if vals:
            w(f"  - captions kappa_bar 0.8, {s} dB: " + " / ".join(f"{v['ov'][0]:.2f}" for U, v in vals) + " % at U = "
              + " / ".join(str(U) for U, v in vals) + "; receiver reduction " + " / ".join(f"{v['rx'][0]:.2f}" for U, v in vals)
              + " %; CosSim gen_joint - load_joint " + " / ".join(f"{v['dc_gl'][0]:+.4f}" for U, v in vals))
    # ---- C.1 model
    w("\n### C.1 Model (Gaussian, fitted statistics, 100 Rayleigh draws, known gains), K = 4\n")
    for mix in (0.8, 0.4):
        w(f"\n#### mixture kappa {mix:g}\n")
        hdr(w, ["U", "SNR", "kappa_bar (fit)", "NMSE load_pu", "NMSE load_joint", "NMSE gen_joint",
                "overlap reduction % (SE)", "receiver reduction %", "total reduction 1 - gen_joint/load_pu %",
                "CosSim proxy load_pu / load_joint / gen_joint"])
        for U in UU:
            t = TH.get((mix, U))
            if t is None:
                row(w, [f"U = {U}", "all", MISS, "", "", "", "", "", "", ""])
                continue
            for s in t["meta"]["snrs"]:
                v = th_red(t, s)
                row(w, [f"U = {U}" + (ovl(U, 4, True) if U > 4 else ""), f"{float(s):g} dB",
                        f"{t['meta']['kappa_bar']:.4f}", f"{v['pu']:.5f}",
                        f"{v['lj']:.5f}", f"{v['gj']:.5f}",
                        f"{v['ov']:.2f} ({v['se_ov']:.2f})" if v["se_ov"] is not None else f"{v['ov']:.2f}",
                        f"{v['rx']:.2f}", f"{v['tot']:.2f}",
                        f"{v['cos_pu']:.4f} / {v['cos_lj']:.4f} / {v['cos_gj']:.4f}"])
    # A1 approximation gap and decomposition
    w("\n#### C.1b A1 approximation gap (diagonal surrogate C = diag(kappa_i lam_i) against the full C) and the "
      "decomposition of the full model (diag.decomp_rel.full_rayleigh)\n")
    hdr(w, ["kappa, U", "SNR", "overlap red. full %", "overlap red. A1 %", "A1 - full (pp)",
            "rel. NMSE gap gen_joint A1/full - 1", "rel. NMSE gap load_joint A1/full - 1",
            "b_over_a (receiver)", "d_over_b (pruned kappa = 0 design)", "c_over_d (overlap)"])
    gaps = []
    for mix in (0.8, 0.4):
        for U in UU:
            t = TH.get((mix, U))
            if t is None:
                continue
            for s in t["meta"]["snrs"]:
                vf, va = th_red(t, s, "full"), th_red(t, s, "a1")
                dg = th_diag(t, s).get("decomp_rel", {}).get("full_rayleigh", {})
                gaps.append((abs(va["ov"] - vf["ov"]), mix, U, s))
                row(w, [f"{mix:g}, U = {U}" + (ovl(U, 4, True) if U > 4 else ""), f"{float(s):g} dB",
                        f"{vf['ov']:.3f}", f"{va['ov']:.3f}",
                        f"{va['ov'] - vf['ov']:+.3f}", f"{va['gj'] / vf['gj'] - 1:+.2e}", f"{va['lj'] / vf['lj'] - 1:+.2e}",
                        f"{dg.get('b_over_a', np.nan):+.4f}", f"{dg.get('d_over_b', np.nan):+.1e}",
                        f"{dg.get('c_over_d', np.nan):+.4f}"])
    if gaps:
        g = max(gaps)
        w(f"\n- largest |A1 - full| of the overlap reduction over kappa 0.8/0.4, U = 2..8 present, all SNRs: {g[0]:.3f} "
          f"percentage points (kappa {g[1]:g}, U = {g[2]}, {float(g[3]):g} dB)")
    w("\n#### C.1c Design of the overlapping masks in the model (diag): active modes, pruning, overlap index\n")
    hdr(w, ["kappa, U", "SNR", "load coefficients per user", "active modes (common)", "pruned modes", "uses of gen_joint / d_s",
            "overlap index t_bar_c", "low-SNR slope gain G0 over load_joint / over load_pu"])
    for mix in (0.8, 0.4):
        for U in UU:
            t = TH.get((mix, U))
            if t is None:
                continue
            sg = t["meta"].get("slope_gains", {})
            for s in t["meta"]["snrs"]:
                d = th_diag(t, s)
                row(w, [f"{mix:g}, U = {U}" + (ovl(U, 4, True) if U > 4 else ""), f"{float(s):g} dB",
                        t["meta"].get("load_coefficients", ["-"])[0],
                        f"{d['active_modes_c']} ({d['active_common_c']})", d["drops_c"], f"{d['uses_c']} / {d['d_s']}",
                        f"{d['t_bar_c']:.4f}", f"{sg.get('G0', np.nan):.3f} / {sg.get('over_per_user', np.nan):.3f}"])
    # ---- C.2 real captions
    w("\n### C.2 Real caption mixtures, K = 4, known gains (shared_coco_mix<k>_K4_U<U>_ol, 2000 frames)\n")
    hdr(w, ["kappa, U", "SNR", "kappa_bar (fit)", "NMSE load_pu / load_joint / gen_joint",
            "overlap reduction [95% CI]", "receiver reduction [95% CI]", "total reduction 1 - gen_joint/load_pu [95% CI]",
            "CosSim gen_joint - load_joint [95% CI]", "CosSim load_joint - load_pu [95% CI]",
            "weakest gen_joint - load_joint [95% CI]"])
    keys = [(0.8, U) for U in UU] + [(0.0, 8)]
    for k in keys:
        rr = RL.get(k)
        if rr is None:
            row(w, [f"{k[0]:g}, U = {k[1]}", "all", MISS] + [""] * 7)
            continue
        for s, v in sorted(rr["res"].items()):
            row(w, [f"{k[0]:g}, U = {k[1]}" + (ovl(k[1], 4, True) if k[1] > 4 else ""), f"{s} dB", f"{rr['kappa_bar']:.4f}",
                    " / ".join(f"{v['nmse'][m]:.5f}" for m in ("load_pu", "load_joint", "gen_joint")),
                    fpca(v["ov"]), fpc(v["rx"]), fpc(v["tot"]), fci(v["dc_gl"]), fci(v["dc_lp"]), fci(v["dw_gl"])])
    w("\n#### C.2b CosSim of every scheme with real captions (mean / weakest user), K = 4\n")
    hdr(w, ["kappa, U", "SNR", "load_pu", "load_joint", "gen_joint"])
    for k in keys:
        rr = RL.get(k)
        if rr is None:
            continue
        for s, v in sorted(rr["res"].items()):
            row(w, [f"{k[0]:g}, U = {k[1]}" + (ovl(k[1], 4, True) if k[1] > 4 else ""), f"{s} dB"]
                + [f"{v['cos'][m]:.4f} / {v['weakest'][m]:.4f}"
                                                          for m in ("load_pu", "load_joint", "gen_joint")])
    w("\n#### C.2c Design of the overlapping masks with real captions (JSON design.gen_joint and design.load)\n")
    hdr(w, ["kappa, U", "SNR", "load masks: uses / coefficients per frame", "gen_joint uses", "active modes / pruned",
            "partial coefficients", "overlap index t", "t_overlap_modes"])
    for k in keys:
        rr = RL.get(k)
        if rr is None:
            continue
        for s, v in sorted(rr["res"].items()):
            d = v["design"]; l_, g_ = d.get("load", {}), d.get("gen_joint", {})
            row(w, [f"{k[0]:g}, U = {k[1]}" + (ovl(k[1], 4, True) if k[1] > 4 else ""), f"{s} dB",
                    f"{l_.get('n_uses', '-')} / {l_.get('n_coef_active', '-')}",
                    g_.get("n_uses", "-"), f"{g_.get('n_modes_active', '-')} / {g_.get('n_pruned', '-')}",
                    g_.get("n_partial_coef", "-"), f"{g_.get('t_overlap', np.nan):.4f}",
                    f"{g_['t_overlap_modes']:.4f}" if g_.get("t_overlap_modes") is not None else "-"])
    # model vs captions at kappa 0.8
    w("\n#### C.2d Model against real captions, kappa_bar 0.8, K = 4: overlap and receiver NMSE reductions (%)\n")
    hdr(w, ["U", "SNR", "overlap: model", "overlap: captions [95% CI]", "overlap: model - captions (pp)",
            "model inside the caption CI", "receiver: model", "receiver: captions [95% CI]", "receiver: model - captions (pp)"])
    rxd = []
    for U in UU:
        t, rr = TH.get((0.8, U)), RL.get((0.8, U))
        if t is None or rr is None:
            row(w, [f"U = {U}", "all", MISS if t is None else "", MISS if rr is None else "", "", "", "", "", ""])
            continue
        for s, v in sorted(rr["res"].items()):
            m = th_red(t, s)
            if m is None:
                continue
            row(w, [f"U = {U}" + (ovl(U, 4, True) if U > 4 else ""), f"{s} dB", f"{m['ov']:.2f}", fpc(v["ov"]),
                    f"{m['ov'] - v['ov'][0]:+.2f}",
                    v["ov"][1] <= m["ov"] <= v["ov"][2], f"{m['rx']:.2f}", fpc(v["rx"]), f"{m['rx'] - v['rx'][0]:+.2f}"])
            rxd.append((abs(m["rx"] - v["rx"][0]), U, s, m["rx"], v["rx"]))
    if rxd:
        m_ = max(rxd)
        anom("C", f"receiver NMSE reduction (load_pu -> load_joint), model against real captions, kappa_bar 0.8, K = 4: "
             f"differs by up to {m_[0]:.2f} percentage points (U = {m_[1]}, {m_[2]} dB: model {m_[3]:.2f} %, captions "
             f"{fpc(m_[4])})" + note("C_rx"))
    # consistency of the non-_ol U = 4 run
    if ex("shared/shared_coco_mix0.8_K4_U4.npz") and ex("shared/shared_coco_mix0.8_K4_U4_ol.npz"):
        za, zb = np.load(P("shared/shared_coco_mix0.8_K4_U4.npz")), np.load(P("shared/shared_coco_mix0.8_K4_U4_ol.npz"))
        dm = max(float(np.abs(za[k] - zb[k]).max()) for k in za.files if k in zb.files and k.split("_")[0] in ("err", "cos"))
        w(f"\n- check: shared_coco_mix0.8_K4_U4 (all six modes, 0/10 dB) against the _ol run of the same point, largest "
          f"frame-by-frame |difference| of the shared arrays: {dm:.1e}")
        if dm > 1e-6:
            anom("C", f"shared_coco_mix0.8_K4_U4 and _ol differ frame by frame ({dm:.1e})")
    elif not ex("shared/shared_coco_mix0.8_K4_U4.json"):
        pend(w, "C", "shared/shared_coco_mix0.8_K4_U4.json", "non-_ol run at U = 4 (cross-check)")
    # kappa_bar of model fit vs caption fit
    kk = [(abs(TH[(0.8, U)]["meta"]["kappa_bar"] - RL[(0.8, U)]["kappa_bar"]), U) for U in UU
          if (0.8, U) in TH and RL.get((0.8, U))]
    if kk:
        w(f"- kappa_bar of the model fit against the fit of the caption runs, kappa 0.8: largest |difference| "
          f"{max(kk)[0]:.1e} (U = {max(kk)[1]})")
    md = [(abs(th_red(TH[(0.8, U)], s)["ov"] - v["ov"][0]), U, s, th_red(TH[(0.8, U)], s)["ov"], v["ov"])
          for U in UU if (0.8, U) in TH and RL.get((0.8, U)) for s, v in RL[(0.8, U)]["res"].items()
          if th_red(TH[(0.8, U)], s)]
    if md:
        m = max(md)
        anom("C", f"model and real captions (kappa_bar 0.8, K = 4) differ in the overlap NMSE reduction by up to "
             f"{m[0]:.2f} percentage points (U = {m[1]}, {m[2]} dB: model {m[3]:.2f} %, captions {fpc(m[4])}); "
             "the model lies outside most caption CIs (see C.2d)" + note("C_model"))
    # ---- C.3 kappa sweep at U = 4
    w("\n### C.3 Content correlation sweep at U = 4 (full load at K = 4; overload at K = 2, gen_joint = pruned "
      "Proposition 2)\n")
    w("Mixture kappa 0 = captions of different images; its fitted kappa_bar is about 0.002 (not 0), and the design "
      "and the receiver use that fitted C. Values below 0.05 % print with four decimals.\n")
    KAP = ("0", "0.2", "0.4", "0.6", "0.8", "0.9", "0.95")
    z0m, z0r = [], []            # mixture kappa 0: |model overlap reduction|, caption reductions with CIs
    for K in (4, 2):
        w(f"\n#### K = {K}: model (Gaussian, 100 Rayleigh draws), overlap reduction % (SE) and receiver reduction %\n")
        snr_th = None
        rows = []
        for k in KAP:
            f = f"shared/theory_coco_mix{k}_K{K}_U4.json"
            if not ex(f):
                rows.append((k, None)); pend(w, "C", f, f"kappa sweep model, K = {K}"); continue
            t = ldj(f); snr_th = snr_th or t["meta"]["snrs"]
            rows.append((k, t))
        if snr_th:
            hdr(w, ["mixture kappa (kappa_bar)"] + [f"{float(s):g} dB overlap (SE) / receiver" for s in snr_th])
            for k, t in rows:
                if t is None:
                    continue
                cells = [f"{k} ({t['meta']['kappa_bar']:.3f})"]
                for s in snr_th:
                    v = th_red(t, s)
                    nd_ = 4 if v and abs(v["ov"]) < 0.05 else 2
                    cells.append(f"{v['ov']:.{nd_}f} ({v['se_ov']:.{nd_}f}) / {fxa(v['rx'])}" if v and v["se_ov"] is not None
                                 else (f"{fxa(v['ov'])} / {fxa(v['rx'])}" if v else "-"))
                    if v and k == "0":
                        z0m.append((abs(v["ov"]), K, float(s), v["ov"]))
                row(w, cells)
        w(f"\n#### K = {K}: real caption mixtures (shared_coco_mix<k>_K{K}_U4, 2000 frames)\n")
        hdr(w, ["mixture kappa (kappa_bar)", "SNR", "overlap reduction [95% CI]", "receiver reduction [95% CI]",
                "CosSim gen_joint - load_joint [95% CI]", "CosSim gen_eq - load_joint [95% CI]"])
        miss = []
        for k in KAP:
            f = f"shared/shared_coco_mix{k}_K{K}_U4.json"
            if not ex(f) or not ex(f[:-5] + ".npz"):
                miss.append(f); continue
            r = ldj(f); z = np.load(P(f[:-5] + ".npz")); F = r["frames"]
            for s in [int(x) for x in r["res"]["load_joint"]]:
                idx = bidx(F, s)
                e = {m: z[f"err_{m}_{s}"].astype(np.float64).sum(1) for m in ("load_pu", "load_joint", "gen_joint")}
                c = {m: z[f"cos_{m}_{s}"].astype(np.float64).mean(1) for m in ("load_joint", "gen_joint", "gen_eq")
                     if f"cos_{m}_{s}" in z}
                rx = ci_red(e["load_joint"], e["load_pu"], idx)
                ov_ = ci_red(e["gen_joint"], e["load_joint"], idx)
                dgl = ci_mean(c["gen_joint"] - c["load_joint"], idx)
                deq = ci_mean(c["gen_eq"] - c["load_joint"], idx) if "gen_eq" in c else None
                fca = lambda t: fci(t, 6 if max(abs(x) for x in t) < 5e-4 else 4)
                row(w, [f"{k} ({r['fit']['kappa_bar']:.3f})", f"{s} dB", fpca(ov_), fpca(rx), fca(dgl),
                        fca(deq) if deq else "-"])
                if k == "0":
                    z0r.append((max(abs(x) for x in ov_), K, s, ov_))
                if ov_[2] < 0:
                    anom("C", f"{f} {s} dB (kappa_bar {r['fit']['kappa_bar']:.4f}): overlap NMSE reduction CI entirely "
                         f"below 0 ({fpca(ov_)}): the overlapping masks are slightly worse than the load masks here"
                         + (note("C_mix0_neg") if k == "0" else ""))
                elif k == "0" and ov_[1] > 0:
                    anom("C", f"{f} {s} dB (kappa_bar {r['fit']['kappa_bar']:.4f}): overlap NMSE reduction CI entirely "
                         f"above 0 ({fpca(ov_)}) although the content is independent: the fitted C (kappa_bar about "
                         "0.002, not 0) gives a tiny nonzero overlap effect" + note("C_mix0_pos"))
                if rx[2] < 0:
                    anom("C", f"{f} {s} dB (kappa_bar {r['fit']['kappa_bar']:.4f}): the joint receiver with the fitted C is "
                         f"slightly worse than the per-user receiver, receiver NMSE reduction {fpc(rx)} (C fitted from "
                         "independent captions is estimation noise)" + note("C_rx0"))
        for f in miss:
            pend(w, "C", [f, f[:-5] + ".npz"], f"kappa sweep real captions, K = {K}")
    if z0m or z0r:
        mm = max(z0m) if z0m else None
        mr = max(z0r) if z0r else None
        bd = max([t[0] for t in z0m] + [t[0] for t in z0r])
        bound = float(np.ceil(bd * 100.0) / 100.0) if bd > 0 else 0.01
        w(f"\n- mixture kappa 0 (fitted kappa_bar about 0.002, design and receiver use the fitted C), K = 4 and 2, every "
          "SNR: largest |overlap NMSE reduction| "
          + (f"model {mm[3]:+.4f} % (K = {mm[1]}, {mm[2]:g} dB)" if mm else "model -")
          + (f"; captions, largest |CI end| {mr[0]:.4f} % (K = {mr[1]}, {mr[2]} dB: {fpca(mr[3])})" if mr else "")
          + note("C3_bound", bound) + "; at K = 4 alone (full load) the largest is "
          + f"{max([t[0] for t in z0m + z0r if t[1] == 4] or [float('nan')]):.4f} %")
    # ---- C.4 natural-caption model runs
    w("\n### C.4 Natural captions: model (Gaussian with the fitted statistics) against the measured reduction of section B\n")
    for ds in ("coco", "flickr"):
        for K in (4, 2):
            w(f"\n#### {ds}, K = {K}\n")
            hdr(w, ["U", "SNR", "kappa_bar", "model overlap red. % (SE)", "model receiver red. %", "measured overlap red. [95% CI]",
                    "measured receiver red. [95% CI]", "pruned modes (model)", "t_bar_c (model)"])
            miss = []
            for U in (2, 3, 4, 5):
                f = f"shared/theory_{ds}_K{K}_U{U}.json"
                if not ex(f):
                    miss.append(f); continue
                t = ldj(f)
                meas = {}
                if ds == "coco" and nat and (K, U) in nat and nat[(K, U)][1]:
                    meas = nat[(K, U)][1]
                elif ds == "flickr" and ex(f"shared/shared_flickr_K{K}_U{U}.json"):
                    rr = real_pt(f"shared/shared_flickr_K{K}_U{U}.json")
                    meas = {s: {"red_gl": v["ov"], "red_lp": v["rx"]} for s, v in rr["res"].items()} if rr else {}
                for s in t["meta"]["snrs"]:
                    v = th_red(t, s); d = th_diag(t, s)
                    mm = meas.get(int(float(s)), {})
                    row(w, [f"U = {U}" + (ovl(U, K, True) if U > K else ""), f"{float(s):g} dB", f"{t['meta']['kappa_bar']:.4f}",
                            f"{v['ov']:.2f} ({v['se_ov']:.2f})" if v["se_ov"] is not None else f"{v['ov']:.2f}",
                            f"{v['rx']:.2f}", fpc(mm["red_gl"]) if "red_gl" in mm else "-",
                            fpc(mm["red_lp"]) if "red_lp" in mm else "-", d["drops_c"], f"{d['t_bar_c']:.4f}"])
            for f in miss:
                pend(w, "C", f, f"{ds} natural-caption model")
    INDEX.append(("C", "shared/theory_coco_mix<k>_K4_U<U>, shared_coco_mix<k>_K4_U<U>_ol(.npz), theory_coco_mix*_K{2,4}_U4, "
                  "shared_coco_mix*_K{2,4}_U4, theory_coco_K*_U*, theory_flickr_K*_U*",
                  "model: 100 Rayleigh draws default_rng(7); captions: 2000 frames, seeds as section B; bootstrap 555 + SNR"))


# ================================================================== D. coherence
RX = re.compile(r"pca_csi_coh_rho([\d.]+)_eta([\d.]+)_U(\d+)(?:_nf(\d+))?\.json$")
DMODES = ("csiq", "csiqp", "switch", "ods", "csiqb", "csiqw")


def sec_D():
    L = SEC.setdefault("D", []); w = L.append
    w("## D. Coherence bandwidth and time (Code/pca_csi.py --coh, Fig. 3(b))\n")
    w("AG News, independent content, K = 4 (N = 24 subbands of 128 uses), 2000 frames, k = U (ods = Static ODS-U). "
      "Gains AR(1) over the subbands with adjacent correlation rho_f, reported one block earlier with correlation eta "
      "(eta = 1: current gains; eta = 0: no information), near-far offsets uniform in +-6 dB where nf = 6. "
      "Gain = CosSim of the mode minus CosSim of fixed (the load masks) on the same frames, with the stored paired "
      "bootstrap 95% CI over frames (1000 resamples, seed 555 + SNR). Modes: csiq = channel-aware masks; csiqp = csiq "
      "with conditional-mean gains; switch = coherence-aware choice; ods = Static ODS-U placed by the same gains; "
      "csiqb = channel-aware support with equal values; csiqw = channel-aware support with (12) at unit gain.\n")
    w("Energy per user (Code/pca_csi.py, E = n * SEG for the csiq family): csiq, csiqp, csiqb, csiqw (and switch when it "
      "picks csiq) give user u the energy n_u x 128 for its n_u subbands chosen per frame (unit power per occupied use; "
      "only the total over the users equals d_s); fixed (the load masks) gives d_s/U to every user and ods d_s/k = "
      "d_s/U. So for U >= 2 the csiq gain over the load masks includes energy moved between users" + note("D_energy")
      + "; at U = 1 the one user receives all 24 subbands, so both have the energy d_s.\n")
    fs = sorted(glob.glob(P("pilot_corr/pca_csi_coh_*.json")))
    R, other = {}, []
    for f in fs:
        m = RX.search(os.path.basename(f))
        if not m:
            other.append(rel(f)); continue
        rho, eta, U, nf = m.group(1), m.group(2), int(m.group(3)), int(m.group(4) or 0)
        R[(nf, rho, eta, U)] = (f, json.load(open(f)))
    if not R:
        pend(w, "D", "pilot_corr/pca_csi_coh_rho<rho>_eta<eta>_U<U>_nf6.json", "coherence runs")
        return
    if other:
        w("- other coherence files (not parsed here): " + ", ".join(other))
    # recompute CIs from the npz
    dmax, nchk, kmis = 0.0, 0, []
    for key, (f, r) in R.items():
        if r["args"].get("k") != key[3]:
            kmis.append(rel(f))
        fn = f[:-5] + ".npz"
        if not os.path.exists(fn):
            pend(w, "D", rel(fn), "per-frame arrays of a coherence run"); continue
        z = np.load(fn)
        F = r["args"]["frames"]
        for s in r["res"]["fixed"]:
            idx = bidx(F, s)
            cf = z[f"cos_fixed_{s}"].astype(np.float64)
            for mo in r["res"]:
                if mo == "fixed" or f"cos_{mo}_{s}" not in z:
                    continue
                cm = z[f"cos_{mo}_{s}"].astype(np.float64)
                t = ci_mean(cm.mean(1) - cf.mean(1), idx)
                tw = ci_mean(cm.min(1) - cf.min(1), idx)
                st_ = r["res"][mo][s]
                dmax = max(dmax, abs(t[0] - st_["gain_fixed"]), abs(t[1] - st_["gain_fixed_ci"][0]),
                           abs(t[2] - st_["gain_fixed_ci"][1]), abs(tw[1] - st_["gain_fixed_weakest_ci"][0]),
                           abs(tw[2] - st_["gain_fixed_weakest_ci"][1]))
                nchk += 1
    w(f"- check: the stored gain_fixed / gain_fixed_weakest and their CIs recomputed from the npz (float32 per-frame "
      f"CosSim, default_rng(555 + SNR)): {nchk} mode-SNR points, largest |difference| {dmax:.1e}")
    if dmax > 5e-5:
        anom("D", f"stored coherence CIs differ from the npz recomputation by up to {dmax:.1e}")
    if kmis:
        anom("D", "coherence runs with k != U: " + ", ".join(kmis))
        w("- runs with k != U: " + ", ".join(kmis))
    else:
        w("- every coherence run has k = U (ods = Static ODS-U)")
    nfs = sorted({k[0] for k in R})
    w(f"- files: {len(R)} (" + ", ".join(f"nf = {n}: {sum(1 for k in R if k[0] == n)}" for n in nfs) + ")")

    def g(key, mo, s, f="gain_fixed"):
        v = R.get(key)
        if v is None or mo not in v[1]["res"] or str(s) not in v[1]["res"][mo]:
            return None
        o = v[1]["res"][mo][str(s)]
        return (o[f], o[f + "_ci"][0], o[f + "_ci"][1]) if f + "_ci" in o else None

    def gs(key, mo, s, f="gain_fixed"):
        t = g(key, mo, s, f)
        return fci(t) if t else "-"
    ETAS = ["1.0", "0.9", "0.7", "0.5", "0.3", "0.0"]
    # ---- D.1 grid
    w("\n### D.1 Gain over the load masks versus the gain correlation eta, offsets +-6 dB (nf = 6)\n")
    for s in (0, 10, 20):
        for mo, lab in (("csiq", "channel-aware masks (csiq)"), ("switch", "coherence-aware choice (switch)"),
                        ("csiqp", "csiq with conditional-mean gains (csiqp)")):
            w(f"\n#### {s} dB, {lab}: mean CosSim gain over the load masks [95% CI]\n")
            hdr(w, ["U, rho_f"] + [f"eta = {e}" for e in ETAS])
            for U in (1, 2, 4, 6):
                for rho in ("0.3", "0.7", "0.95"):
                    row(w, [f"U = {U}, rho_f = {rho}"] + [gs((6, rho, e, U), mo, s) if (6, rho, e, U) in R
                                                         else MISS for e in ETAS])
        w(f"\n#### {s} dB, coherence-aware choice: share of frames choosing csiq; rule_loss [95% CI]; decision_loss; "
          "oracle_cf_cos - cos (value of the current gains)\n")
        hdr(w, ["U, rho_f"] + [f"eta = {e}" for e in ETAS])
        for U in (1, 2, 4, 6):
            for rho in ("0.3", "0.7", "0.95"):
                cells = [f"U = {U}, rho_f = {rho}"]
                for e in ETAS:
                    v = R.get((6, rho, e, U))
                    if v is None:
                        cells.append(MISS); continue
                    o = v[1]["res"]["switch"][str(s)]
                    cells.append(f"{o['frac_csiq']:.3f}; {o['rule_loss']:+.4f} [{o['rule_loss_ci'][0]:+.4f}, "
                                 f"{o['rule_loss_ci'][1]:+.4f}]; {o['decision_loss']:.4f}; {o['oracle_cf_cos'] - o['cos']:+.4f}")
                row(w, cells)
    for U in (1, 2, 4, 6):
        for rho in ("0.3", "0.7", "0.95"):
            for e in ETAS:
                if (6, rho, e, U) not in R:
                    pend(w, "D", [f"pilot_corr/pca_csi_coh_rho{rho}_eta{e}_U{U}_nf6.json",
                                  f"pilot_corr/pca_csi_coh_rho{rho}_eta{e}_U{U}_nf6.npz"], "coherence grid point")
    # ---- D.2 summary statements
    w("\n### D.2 Summary at rho_f = 0.7, nf = 6\n")
    for s in (0, 10, 20):
        for e in ("1.0", "0.0"):
            w(f"- {s} dB, eta = {e}: csiq gain " + "; ".join(f"U = {U} {gs((6, '0.7', e, U), 'csiq', s)}" for U in (1, 2, 4, 6)))
            w(f"- {s} dB, eta = {e}: switch gain " + "; ".join(f"U = {U} {gs((6, '0.7', e, U), 'switch', s)}" for U in (1, 2, 4, 6)))
            w(f"- {s} dB, eta = {e}: weakest-user csiq gain " + "; ".join(
                f"U = {U} {gs((6, '0.7', e, U), 'csiq', s, 'gain_fixed_weakest')}" for U in (1, 2, 4, 6)))
    sw0 = [(abs(g(k, "switch", s)[0]), k, s) for k in R if k[2] == "0.0" and k[0] == 6 for s in (0, 10, 20) if g(k, "switch", s)]
    if sw0:
        m = max(sw0)
        w(f"- eta = 0, nf = 6, every rho_f, U and SNR: largest |switch gain| {m[0]:.4f} (rho_f = {m[1][1]}, U = {m[1][3]}, {m[2]} dB)")
    swn = [(g(k, "switch", s), k, s) for k in R for s in (0, 10, 20) if g(k, "switch", s)]
    neg = [(t, k, s) for t, k, s in swn if t[2] < 0]
    if neg:
        anom("D", f"the coherence-aware choice falls below the load masks (CI entirely below 0) at {len(neg)} of {len(swn)} "
             f"points, by at most {-min(neg)[0][0]:.4f} CosSim (nf {min(neg)[1][0]}, rho_f {min(neg)[1][1]}, eta "
             f"{min(neg)[1][2]}, U = {min(neg)[1][3]}, {min(neg)[2]} dB)" + note("D_switch", -min(neg)[0][0]))
    w(f"- switch below the load masks with the CI entirely below 0: {len(neg)} of {len(swn)} points"
      + ("; worst " + ", ".join(f"{fci(t)} (nf {k[0]}, rho_f {k[1]}, eta {k[2]}, U = {k[3]}, {s} dB)"
                                for t, k, s in sorted(neg)[:5]) if neg else ""))
    rl = [(R[k][1]["res"]["switch"][str(s)]["rule_loss"], k, s) for k in R for s in (0, 10, 20)
          if str(s) in R[k][1]["res"]["switch"]]
    m = max(rl)
    w(f"- largest rule_loss over all runs: {m[0]:+.4f} (nf {m[1][0]}, rho_f {m[1][1]}, eta {m[1][2]}, U = {m[1][3]}, {m[2]} dB)")
    w("- first eta (from 1 down to 0) at which the csiq gain CI lies entirely below 0 (nf = 6):")
    for s in (0, 10, 20):
        cells = []
        for U in (1, 2, 4, 6):
            for rho in ("0.3", "0.7", "0.95"):
                first = next((e for e in ETAS if g((6, rho, e, U), "csiq", s) and g((6, rho, e, U), "csiq", s)[2] < 0), "none")
                cells.append(f"U = {U}, rho_f {rho}: {first}")
        w(f"  - {s} dB: " + "; ".join(cells))
    # ---- D.3 rho sweep at eta = 1
    w("\n### D.3 Subband correlation rho_f at eta = 1 (current gains), nf = 6: gain over the load masks [95% CI]\n")
    RHOS = sorted({k[1] for k in R if k[0] == 6 and k[2] == "1.0"}, key=float)
    for s in (0, 10, 20):
        for mo in ("csiq", "ods", "csiqb", "csiqw"):
            w(f"\n#### {s} dB, {mo}\n")
            hdr(w, ["rho_f"] + [f"U = {U}" for U in (1, 2, 4, 6)])
            for rho in RHOS:
                row(w, [rho] + [gs((6, rho, "1.0", U), mo, s) if (6, rho, "1.0", U) in R else MISS for U in (1, 2, 4, 6)])
    for rho in ("0.3", "0.5", "0.7", "0.9", "0.95", "1.0"):
        for U in (1, 2, 4, 6):
            if (6, rho, "1.0", U) not in R:
                pend(w, "D", f"pilot_corr/pca_csi_coh_rho{rho}_eta1.0_U{U}_nf6.json", "rho_f sweep point")
    RH = [r_ for r_ in RHOS if r_ != "1.0"]
    if len(RH) >= 2:
        w(f"\n- dependence of the csiq gain on rho_f (eta = 1, nf = 6, rho_f {RH[0]} .. {RH[-1]}; rho_f = 1 is the "
          "one-gain case of D.4): 'decreasing' = the gain falls at every step of rho_f; 'flat' = the CIs at the two "
          "ends overlap")
        summ_ = []
        for s in (0, 10, 20):
            cells, dl, fl = [], [], []
            for U in (1, 2, 4, 6):
                vals = [g((6, r_, "1.0", U), "csiq", s) for r_ in RH]
                if any(v is None for v in vals):
                    cells.append(f"U = {U}: -"); continue
                dec = all(a_[0] > b_[0] for a_, b_ in zip(vals, vals[1:]))
                flat = vals[0][1] <= vals[-1][2] and vals[-1][1] <= vals[0][2]
                (fl if flat else dl if dec else []).append(str(U))
                cells.append(f"U = {U}: {vals[0][0]:+.4f} -> {vals[-1][0]:+.4f}, "
                             + ("flat" if flat else ("decreasing" if dec else "not monotone")))
            w(f"  - {s} dB: " + "; ".join(cells))
            summ_.append(f"{s} dB: " + (f"decreasing at U = {', '.join(dl)}" if dl else "decreasing at no load")
                         + (f", flat at U = {', '.join(fl)}" if fl else ", flat at no load"))
        w("  - so 'the gain grows as rho_f decreases' holds only for the loads listed as decreasing (" + "; ".join(summ_)
          + "); at a flat load the subband correlation does not change the gain within the CIs")
    # ---- D.4 rho = 1
    w("\n### D.4 rho_f = 1 (one gain per user and block), eta = 1, nf = 6: CosSim of fixed and gain of every mode [95% CI]\n")
    hdr(w, ["U", "SNR", "fixed CosSim / weakest"] + list(DMODES))
    for U in (1, 2, 4, 6):
        k = (6, "1.0", "1.0", U)
        if k not in R:
            continue
        for s in (0, 10, 20):
            fx = R[k][1]["res"]["fixed"][str(s)]
            row(w, [f"U = {U}", f"{s} dB", f"{fx['cos']:.4f} / {fx['weakest']:.4f}"] + [gs(k, mo, s) for mo in DMODES])
    k1 = (6, "1.0", "1.0", 1)
    if k1 in R and g(k1, "csiq", 0) and g(k1, "csiqw", 0):
        q, qw, qb = g(k1, "csiq", 0), g(k1, "csiqw", 0), g(k1, "csiqb", 0)
        k7 = (6, "0.7", "1.0", 1)
        w(f"\n- U = 1, rho_f = 1 (one gain per block, so the subbands cannot be chosen), 0 dB: csiq gains {fci(q)} over "
          f"the load masks with the same energy d_s; the csiq support with (12) at unit gain (csiqw) gains {fci(qw)}"
          + (f" and with equal values (csiqb) {fci(qb)}" if qb else "")
          + f"; the remaining {q[0] - qw[0]:+.4f} comes from the transmitter knowing the gain in (13), not from choosing "
          "subbands" + (f". At rho_f = 0.7 the U = 1 gain is {fci(g(k7, 'csiq', 0))}, so at U = 1 only part of the "
                        "channel-aware gain is a subband choice" if g(k7, "csiq", 0) else ""))
    # ---- D.5 no offsets
    w("\n### D.5 Without near-far offsets (nf = 0) against nf = 6, rho_f = 0.7: csiq gain [95% CI] and fixed CosSim\n")
    hdr(w, ["U, eta", "SNR", "nf = 0 csiq gain", "nf = 6 csiq gain", "nf = 0 switch gain", "nf = 6 switch gain",
            "fixed CosSim nf = 0 / nf = 6"])
    for k in sorted(k for k in R if k[0] == 0):
        k6 = (6,) + k[1:]
        for s in (0, 10, 20):
            f0 = R[k][1]["res"]["fixed"][str(s)]["cos"]
            f6 = R[k6][1]["res"]["fixed"][str(s)]["cos"] if k6 in R else np.nan
            row(w, [f"rho_f {k[1]}, U = {k[3]}, eta = {k[2]}", f"{s} dB", gs(k, "csiq", s), gs(k6, "csiq", s) if k6 in R else "-",
                    gs(k, "switch", s), gs(k6, "switch", s) if k6 in R else "-", f"{f0:.4f} / {f6:.4f}"])
    # ---- D.6 full listing
    w("\n### D.6 Full listing: every coherence file, SNR and mode\n")
    w("Columns: mean CosSim; gain over fixed [95% CI]; weakest-user CosSim; weakest-user gain over fixed [95% CI]. "
      "Switch rows add frac_csiq, rule_loss [CI], decision_loss, oracle_cf_cos - cos, frac_csiq_cf.\n")
    hdr(w, ["nf", "rho_f", "eta", "U", "SNR", "mode", "CosSim", "gain [95% CI]", "weakest", "weakest gain [95% CI]", "switch extras"])
    for k in sorted(R, key=lambda k: (k[0], float(k[1]), -float(k[2]), k[3])):
        r = R[k][1]
        for s in r["res"]["fixed"]:
            fx = r["res"]["fixed"][s]
            row(w, [k[0], k[1], k[2], k[3], f"{s} dB", "fixed", f"{fx['cos']:.4f}", "-", f"{fx['weakest']:.4f}", "-", ""])
            for mo in DMODES:
                if mo not in r["res"] or s not in r["res"][mo]:
                    continue
                o = r["res"][mo][s]
                ex_ = ""
                if mo == "switch":
                    ex_ = (f"frac {o['frac_csiq']:.3f}; rule_loss {o['rule_loss']:+.4f} [{o['rule_loss_ci'][0]:+.4f}, "
                           f"{o['rule_loss_ci'][1]:+.4f}]; decision_loss {o['decision_loss']:.4f}; oracle_cf - cos "
                           f"{o['oracle_cf_cos'] - o['cos']:+.4f}; frac_cf {o.get('frac_csiq_cf', np.nan):.3f}")
                row(w, [k[0], k[1], k[2], k[3], f"{s} dB", mo, f"{o['cos']:.4f}", gs(k, mo, s), f"{o['weakest']:.4f}",
                        gs(k, mo, s, "gain_fixed_weakest"), ex_])
    INDEX.append(("D", "pilot_corr/pca_csi_coh_rho<rho>_eta<eta>_U<U>[_nf6].json/.npz",
                  "2000 frames per run, draws seed 123 + SNR (shared by all modes of a run), switch samples seed 999 + SNR, "
                  "bootstrap 555 + SNR"))


# ================================================================== E. search versus closed forms
def fg(x):
    """gap in percent: scientific below 1e-3 so that numerical zeros stay visible as such."""
    return f"{x:+.1e}" if abs(x) < 1e-3 else f"{x:+.3f}"


def sec_E():
    L = SEC.setdefault("E", []); w = L.append
    w("## E. Numerical search against the closed-form masks (Code/overlap_bound.py)\n")
    w("Restricted model (use i of every slot mixes only the coefficients i of the groups: 128 problems of S = 6K uses), "
      "joint LMMSE receiver with known gains, per-user energy. closed = the closed form of the condition (load12 = load "
      "masks with (12); prop2 = Proposition 2; prop2_pruned = pruned Proposition 2 at U > K; csiq13 = channel-aware "
      "masks with (13) on the greedy support); support-restricted search = values optimized on the support of the "
      "load masks (csiq support for slot gains), the non-overlapping instance; search = any real S x 6U transmit "
      "matrix per use index (Adam from the closed form and 3 random starts, 3000 steps, best kept). gap = "
      "100 (closed - search) / closed of the sum NMSE (positive: the search is lower). With Rayleigh draws the closed "
      "forms are designed at the unit gain while the search knows the drawn gains at the transmitter (the gap then "
      "contains transmitter gain knowledge).\n")
    fs = sorted(glob.glob(P("extra/overlap_bound2_*.json")))
    if not fs:
        pend(w, "E", "extra/overlap_bound2_*.json", "search checks")
    w("### E.1 overlap_bound2 (generalized masks)\n")
    hdr(w, ["setting", "SNR", "closed form", "draws", "sum NMSE load / closed / support-restricted search / search",
            "gap closed -> search: mean / max / min over draws %", "gap of the means %",
            "advantage of closed over the support-restricted search 100 (support - closed)/support: mean / min %",
            "free search beyond the support-restricted search 100 (support - search)/support: mean %", "load masks above closed 100 (load - closed)/closed: mean %",
            "overlap index t closed / search", "search energy on shared slots / outside the closed support",
            "closed-form checks"])
    summ = {}
    for f in fs:
        r = json.load(open(f))
        case = r["case"]
        for s, o in r["res"].items():
            cs_, ds_, ss_ = np.array(o["closed_sum"]), np.array(o["disjoint_sum"]), np.array(o["search_sum"])
            dg = 100 * (ds_ - cs_) / ds_
            op = 100 * (ds_ - ss_) / ds_
            gp = np.array(o["gap_pct"])
            chk = ""
            if "prop2" in o:
                pp = o["prop2"]
                chk = f"modal vs trace rel {pp.get('modal_rel_diff', np.nan):.1e}; energy dev {pp.get('user_energy_dev', np.nan):.1e}"
            if "prune" in o:
                pp = o["prune"]
                chk = (f"split {pp.get('split')}; modes kept {pp.get('modes_kept')}; drops {pp.get('drops_global')}; "
                       f"energy dev before scaling {pp.get('user_energy_dev_prescale', np.nan):.1e}")
            if "csiq_counts_mean" in o:
                chk = "csiq subbands per user " + "/".join(f"{x:.2f}" for x in o["csiq_counts_mean"])
            row(w, [case, f"{s} dB", o["closed_name"], o["draws"],
                    f"{o['load_mean_sum']:.5f} / {o['closed_mean_sum']:.5f} / {o['disjoint_mean_sum']:.5f} / {o['search_mean_sum']:.5f}",
                    f"{fg(gp.mean())} / {fg(gp.max())} / {fg(gp.min())}", fg(o["gap_pct_of_means"]),
                    f"{fg(dg.mean())} / {fg(dg.min())}", fg(op.mean()), fg(o["load_minus_closed_pct_mean"]),
                    f"{np.mean(o['t_closed']):.3f} / {np.mean(o['t_search']):.3f}",
                    f"{np.mean(o['search_shared_energy']):.3f} / {np.mean(o['search_tx_outside_closed']):.3f}", chk])
            kind = case.split("_")[1].split("-")[0]
            summ.setdefault(kind, []).append((gp.max(), gp.mean(), case, s, op.mean(), np.mean(o["search_shared_energy"]),
                                              dg.mean(), 100 * (o["closed_mean_sum"] - o["disjoint_mean_sum"]) / o["closed_mean_sum"]))
            if gp.min() < -0.01:
                anom("E", f"{case} {s} dB: the search ends above the closed form by {-gp.min():.3f} % in a draw")
    w("")
    for kind, v in summ.items():
        lab = {"equal": "equal (deterministic) gains, closed form designed for the actual gains",
               "rayleigh": "Rayleigh draws, closed form at the unit design gain, search with the drawn gains known at the transmitter",
               "slot": "AR(1) subband gains, channel-aware closed form csiq13 against the search with the same gains"}.get(kind, kind)
        mx, mn = max(v), max(v, key=lambda t: t[1])
        w(f"- {lab}: largest per-draw gap {fg(mx[0])} % ({mx[2]}, {mx[3]} dB); largest mean gap {fg(mn[1])} % ({mn[2]}, {mn[3]} dB); "
          f"largest mean gain of the free search over the support-restricted search {fg(max(t[4] for t in v))} %; "
          f"settings x SNRs: {len(v)}")
        if kind == "slot":
            anom("E", f"channel-aware closed form csiq13 against the search with the same subband gains: mean gap up to "
                 f"{fg(mn[1])} % ({mn[2]}, {mn[3]} dB), per draw up to {fg(mx[0])} %; the search shares little energy "
                 f"(shared-slot energy {mn[5]:.3f}), so the gap is mostly a better subband assignment than the greedy support, "
                 "not overlap" + note("E_slot"))
        if kind == "rayleigh":
            dmin = min(v, key=lambda t: t[6])
            anom("E", f"Rayleigh draws: the search knows the drawn gains at the transmitter while the closed forms use the "
                 f"unit design gain, so its gap (mean up to {fg(mn[1])} %) is mostly transmitter gain knowledge: the "
                 f"non-overlapping support-restricted search with the same knowledge is already lower than the closed "
                 f"form by up to {-dmin[6]:.3f} % of its own value (mean over draws of 100 (closed - support)/support; "
                 f"{dmin[2]}, {dmin[3]} dB; as a ratio of the mean sums {max(t[7] for t in v):.3f} % of the closed form "
                 f"at most), and the free search beyond it is lower by at most {max(t[4] for t in v):.3f} % of the "
                 "support-restricted value (mean over draws)" + note("E_rayleigh"))
    w("\n### E.2 overlap_bound (AG News, load masks against any real overlapping masks, deterministic gains)\n")
    fs0 = sorted(glob.glob(P("extra/overlap_bound_U*_nf*.json")))
    if not fs0:
        pend(w, "E", "extra/overlap_bound_U<U>_nf<X>.json", "original search")
    hdr(w, ["U, offsets", "SNR", "per-user NMSE mean: closed / support-restricted / search", "worst user: closed / search",
            "gap mean %", "gap worst user %", "search energy on shared slots (max over users)"])
    g0 = []
    for f in fs0:
        r = json.load(open(f))
        a = r["args"]
        for s, o in r["res"].items():
            if s == "gain_db":
                continue
            gm = 100 * (o["closed_mean"] - o["overlap_mean"]) / o["closed_mean"]
            gw = 100 * (o["closed_worst"] - o["overlap_worst"]) / o["closed_worst"]
            g0.append((gm, a["U"], a["nearfar"], s, gw))
            row(w, [f"U = {a['U']}, +-{a['nearfar']:g} dB" + (" (overload)" if a["U"] > 4 else ""), f"{s} dB",
                    f"{o['closed_mean']:.6f} / {o['disjoint_mean']:.6f} / {o['overlap_mean']:.6f}",
                    f"{o['closed_worst']:.6f} / {o['overlap_worst']:.6f}", fg(gm), fg(gw),
                    f"{max(o['overlap_shared_energy']):.2e}"])
    if g0:
        w(f"\n- largest mean gap closed -> search over U = 2, 4, 6, 8, offsets 0 / +-6 dB, 0/10/20 dB: {fg(max(g0)[0])} % "
          f"(U = {max(g0)[1]}, +-{max(g0)[2]:g} dB, {max(g0)[3]} dB); smallest {fg(min(g0)[0])} %")
        lo = [t for t in g0 if t[1] <= 4]
        hi = [t for t in g0 if t[1] > 4 and t[0] > 1e-3]
        if lo:
            w(f"- up to full load (U = 2, 4), every offset and SNR: |gap| <= {max(abs(t[0]) for t in lo):.1e} % (the load "
              "masks are the optimum of the search for independent content)")
        if hi:
            w("- overload with +-6 dB offsets: " + "; ".join(f"U = {t[1]}, {t[3]} dB: mean gap {fg(t[0])} %, worst user "
                                                        f"{fg(t[4])} %" for t in hi))
            hm = max(hi)
            ww = [t for t in g0 if t[1] > 4 and t[4] < -1e-3]
            wm = min(ww, key=lambda t: t[4]) if ww else None
            snr_m = sorted({t[3] for t in hi}, key=float)
            snr_w = sorted({t[3] for t in ww}, key=float)
            anom("E", f"independent content (AG News), overload U = {', '.join(str(U) for U in sorted({t[1] for t in hi}))} "
                 f"with +-6 dB offsets: searched overlapping masks give a mean NMSE lower than the load masks by up to "
                 f"{hm[0]:.3f} % (U = {hm[1]}, {hm[3]} dB; nonzero at {'/'.join(snr_m)} dB)"
                 + (f"; the worst user is worse by up to {-wm[4]:.3f} % (U = {wm[1]}) at {'/'.join(snr_w)} dB only "
                    "(unchanged within 1e-3 % at the other SNRs)" if wm else "; the worst user is never worse by more than 1e-3 %")
                 + "; with equal gains the gap is 0 at every load. So 'load masks optimal for independent content' holds "
                 "up to full load (and in overload only with equal gains)")
    INDEX.append(("E", "extra/overlap_bound2_*.json, extra/overlap_bound_U<U>_nf<X>.json",
                  "equal gains: 1 draw; rayleigh-D20 and slot-rho0.7-D20: 20 draws default_rng(123 + SNR); search torch seed 1000 + SNR "
                  "(original: seed SNR), 3000 Adam steps, 3 random starts"))


# ================================================================== F. gain estimation
def sec_F():
    L = SEC.setdefault("F", []); w = L.append
    w("## F. Gain estimation for the overlapping masks (Code/shared_pilot.py)\n")
    fs = sorted(glob.glob(P("shared/pilot/shared_pilot_*.json")))
    tags = [f"coco_K{K}_U{U}{ph}_E{E}_emauto0_fs_it30" for (U, K) in ((4, 4), (4, 2), (2, 4), (2, 2)) for E in (1, 8)
            for ph in ([""] if not (U == 4 and K == 4 and E == 1) else ["", "_ph10"])]
    if not fs:
        w("No file in data/shared/pilot/ yet (" + note("F_queue", public="the queue Code/pilot_queue.sh writes")
          + " shared_pilot_<tag>.json, or "
          "shared_pilot_<tag>_snr<S>.json with SPLIT=1; 2000 frames, SNR -5/0/10/20 dB).")
        for t in tags:
            pend(w, "F", [f"shared/pilot/shared_pilot_{t}.json", f"shared/pilot/shared_pilot_{t}.npz"],
                 "gain estimation (final, 2000 frames; with SPLIT=1 one file per SNR, suffix _snr<S>)")
    else:
        groups = {}
        for f in fs:
            b = os.path.basename(f)[len("shared_pilot_"):-5]
            base = re.sub(r"_snr-?\d+", "", b)
            groups.setdefault(base, []).append(f)
        for t in tags:
            if t not in groups:
                pend(w, "F", [f"shared/pilot/shared_pilot_{t}.json", f"shared/pilot/shared_pilot_{t}.npz"],
                     "gain estimation point (final, 2000 frames)")
        for base, fl in sorted(groups.items()):
            rs = [json.load(open(f)) for f in fl]
            frames = sorted({r.get("frames", "?") for r in rs}, key=str)
            trial = "_fr" in base or any(isinstance(x, int) and x < 2000 for x in frames)
            w(f"\n### {base}" + (" (TRIAL RUN, not final)" if trial else "") + f": frames {'/'.join(str(x) for x in frames)}; "
              + ", ".join(f"data/{rel(f)}" for f in fl) + "\n")
            hdr(w, ["mode", "SNR", "CosSim", "weakest", "NMSE", "gain rel. RMSE", "- load_joint_blind CosSim [95% CI]",
                    "- load_joint (true g) CosSim [95% CI]", "- load_joint_blind NMSE [95% CI]"])
            rows = []
            for r in rs:
                for m, d in r["res"].items():
                    for s_, o in d.items():
                        rows.append((m, int(s_), o))
            for m, s_, o in sorted(rows, key=lambda t: (t[0], t[1])):
                def bb(key):
                    v = o.get(key)
                    return f"{v['mean']:+.4f} [{v['lo']:+.4f}, {v['hi']:+.4f}]" if v else "-"
                row(w, [m, f"{s_} dB", f"{o['cos']:.4f}", f"{o['weakest']:.4f}", f"{o['nmse']:.5f}",
                        f"{o['gain_rel_rmse']:.4f}" if "gain_rel_rmse" in o else "-", bb("d_vs_load_joint_blind"),
                        bb("d_vs_load_joint"), bb("dnmse_vs_load_joint_blind")])
    if not PUBLIC:
        for t in NOTES["F_extra"]:
            w(t)
    INDEX.append(("F", "shared/pilot/shared_pilot_<tag>[_snr<S>].json" + note("F_index_queue"),
                  "2000 frames, seeds as section B, bootstrap 555 + SNR" + note("F_index_extra")))


# ================================================================== assemble
def main():
    sec_A()
    nat = sec_B()
    sec_C(nat)
    sec_D()
    sec_E()
    sec_F()
    out = ["# Numbers of the general-mask version (Code/general_numbers.py, read-only on data/)\n",
           "Every number below is computed from stored results" + note("intro_f1") + "; CIs are paired bootstrap 95% "
           "intervals over frames "
           "(1000 resamples, numpy default_rng(555 + SNR)) unless marked SE (Monte Carlo standard error over draws). "
           "CosSim = per-user cosine similarity of the recovered embedding; NMSE = sum of squared errors / sum of "
           "squared norms (PCA coefficients); reductions in percent of the reference NMSE. K = 4 (d_s = 4 d_b = 3072 "
           "uses, full load U = 4) unless stated.\n",
           "## G. Index\n"]
    cnt = {"A": len(glob.glob(P("extra/pca_eval*.json"))) + len(glob.glob(P("ortho_U[0-9]_s[0-9].json"))),
           "B": len([f for f in glob.glob(P("shared/shared_*.json")) if "_mix" not in f]) + ex("shared/content_stats.json"),
           "C": len(glob.glob(P("shared/theory_*.json"))) + len(glob.glob(P("shared/shared_coco_mix*.json"))),
           "D": len(glob.glob(P("pilot_corr/pca_csi_coh_*.json"))),
           "E": len(glob.glob(P("extra/overlap_bound*.json"))),
           "F": len(glob.glob(P("shared/pilot/shared_pilot_*.json")))}
    out.append("| section | content | inputs | JSON files found | frames and seeds |")
    out.append("|---|---|---|---|---|")
    names = {"A": "load adaptivity, AG News (Fig. 2)", "B": "natural captions of one image, known gains",
             "C": "one scene from light load through overload (Fig. 3(a))", "D": "coherence bandwidth and time (Fig. 3(b))",
             "E": "search against the closed forms", "F": "gain estimation"}
    for s, inp, fr in sorted(INDEX):
        out.append(f"| {s} | {names[s]} | {inp} | {cnt.get(s, '-')} | {fr} |")
    seen = []
    for s, fs, what in MISSING:
        if (s, fs) not in [(a, b) for a, b, c in seen]:
            seen.append((s, fs, what))
    if seen or not PUBLIC:
        out.append("\n" + note("missing_head", len(seen), public=f"### Missing inputs ({len(seen)})") + "\n")
    for s, fs, what in seen:
        out.append(f"- {MISS} [{s}] {what}: " + ", ".join(f"data/{f}" for f in fs))
    if not MISSING and not PUBLIC:
        out.append("- none")
    out.append(f"\n### Data anomalies and checks ({len(ANOM)})\n")
    for s, t in ANOM:
        out.append(f"- [{s}] {t}")
    if not ANOM:
        out.append("- none flagged automatically")
    out.append("")
    for s in "ABCDEF":
        out += SEC.get(s, []) + [""]
    open(OUT, "w").write("\n".join(out) + "\n")
    print("\n".join(out[:80]))
    print(f"... wrote {OUT} ({len(out)} lines){' (public)' if PUBLIC else ''}; missing inputs {len(seen)}; "
          f"anomalies {len(ANOM)}")


if __name__ == "__main__":
    main()
