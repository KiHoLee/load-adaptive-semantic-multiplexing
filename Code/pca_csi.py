# =========================================================
# pca_csi.py — channel-aware dynamic masks for the closed-form PCA code (2026-11-04).
# Fading varies over the N = 24 slots of a block: every user has its own AR(1) complex Gaussian gain
# process over the slots (correlation rho between adjacent slots, CN(0,1) marginals), independent
# across users, with an optional near-far offset uniform in [-X, X] dB per user and frame. The
# receiver knows the gains of the active users, either of the current block (perfect)
# or of the previous block (--delay rho_b: h_now = rho_b h_prev + sqrt(1 - rho_b^2) w per slot), and
# announces the placement. Each user scales every occupied slot to its planned energy, so the
# receiver estimates each slot gain blindly from the received energy of that slot and applies the
# LMMSE (Wiener) estimate per coefficient, combining the copies of a coefficient.
# Schemes (coefficient block b = coefficients 128 b .. 128 b + 127, b = 0..5):
#   fixed     the masks of the manuscript (rank s mod U), coefficients repeated cyclically, power of
#             the allocation (12) for unit gain split equally over the copies (no CSI at the transmitter)
#   csi       channel-aware masks: Hungarian slot sets with the quotas of the fixed rule, block b on the
#             b-th strongest slot of the user, allocation with the slot gains
#             P_i = max(0, sqrt(lambda_i s2 / (nu g_i^2)) - s2 / g_i^2), energy d_s/U per user
#   csiq      csi with the slot COUNT of every user set per frame: from one slot per user, every
#             remaining slot goes to the user whose MSE drops most; energy n_j x 128 (unit power per
#             occupied use)
#   csiqe     csiq with every count capped at that of the load masks, so the counts are those of csi
#             and only the placement is greedy (isolates the slot counts at a load U that divides N)
#   csic      csiq with the greedy criterion sqrt(1 - MSE / sum lambda), a proxy of the per-user CosSim of
#             the LMMSE estimate, instead of the MSE
#   ods       Static ODS-k placed by the same gains at a load U <= k: every active user gets d_s/k uses
#             (N/k slots chosen by the Hungarian assignment among all free slots), energy d_s/k
#   odsm      ods with the energy d_s/U of the proposal (the matched-energy reference of the manuscript)
#   odsfix    Static ODS-k with fixed contiguous parts and no CSI at the transmitter (the classical one)
# Mask values (2026-11-04): a mask is its slot support times its amplitudes. These modes keep a support and
# replace the amplitudes, on the same draws and, for the csiq family, the same greedy subband counts:
#   fixedb    support of fixed, every used entry at one amplitude (equal values, P_i ~ lambda_i m_i)
#   csiqw     support of csiq, amplitudes of the allocation (12) for unit gain (SNR-aware, gain-blind)
#   csiqb     support of csiq, every used entry at one amplitude (equal values, P_i ~ lambda_i)
# Coherence (2026-11-16): gains reported one block earlier, h | h' ~ CN(eta h', (1 - eta^2) R_F) per user,
# [R_F]_{ss'} = rho^{|s-s'|} (times the user offset), eta = --delay (alias --eta).
#   channel   the delayed-CSI innovation v is drawn as before (same complex normal draws, same order) and
#             then passed through the same AR(1) filter over the subbands as h, so the current block keeps
#             the adjacent-subband correlation rho (stationary). The old i.i.d. innovation, which gave the
#             current block the correlation rho eta^2, stays behind --legacy_delay. eta = 1 draws no v, so
#             those runs are bit-identical to the old code; eta = 0 makes h independent of h'; rho = 1 gives
#             one gain per user and block.
#   switch    per frame and user set, the receiver (which knows h', eta, rho, offsets) draws --nsamp (32)
#             conditional samples of h | h' with a separate generator (seed 999 + SNR), evaluates for the
#             csiq plan and for the fixed (load-mask) plan the closed-form sum MSE with known gains
#               csiq : sum_i lam_i s2 / (g_{slot(i)}^2 P_i + s2)   (unsent coefficients add lam_i)
#               fixed: sum_i lam_i s2 / (sum_copies g_s^2 P_i / m_i + s2)
#             averaged over the samples and announces the lower one. The realized result is that plan's
#             (the csiq and fixed receivers on the same draws). Stored per SNR:
#               frac_csiq        share of frames choosing csiq
#               rule_loss        max(cos fixed, cos csiq) - cos switch, the loss of the rule against the better
#                                member at scheme level (negative: the switch beats both), with rule_loss_ci, a
#                                paired bootstrap 95% CI over frames (1000 resamples, seed 555 + SNR)
#               oracle_cf_cos    CosSim when choosing per frame by the same closed forms at the TRUE current
#                                gains (frac_csiq_cf); oracle_cf_cos - cos is the value of current-gain knowledge
#                                (staleness), which no rule built on h' can recover
#               oracle_cos       CosSim of the per-frame better plan by the realized outcome; decision_loss =
#                                oracle_cos - cos. It contains the staleness
#                                share above plus a receiver-noise share (oracle_cos - oracle_cf_cos), so it is
#                                NOT the error of the rule given h'
#               pred_mse_*, true_mse_*  closed-form sum MSE averaged over the samples and at the true gains
#   csiqp     csiq with the conditional-mean gains g_bar^2 = eta^2 g'^2 + (1 - eta^2) offset^2 in the greedy
#             subband counts and in (13); slots are still ordered by g' (equal to csiq at eta = 1).
# Every mode except fixed also stores, when fixed is run, gain_fixed / gain_fixed_weakest (mean and weakest
# CosSim minus those of fixed) with paired bootstrap 95% CIs (same resamples as rule_loss_ci).
# Output: ../data/pilot_corr/pca_csi_<tag><suffix>.json; with --coh
#   ../data/pilot_corr/pca_csi_coh_rho<rho>_eta<eta>_U<U>[_nf<X>].json. Without --coh, a delayed run with the
#   stationary innovation adds _stat to the old tag so the files of the legacy runs are not overwritten.
#   Next to the JSON, an .npz with the per-frame data (paired across the schemes of a run): cos_<mode>_<snr>
#   (F,U) float32 per-user CosSim, pos_<snr> (F,U) test indices, and for switch choose_<snr>, choose_cf_<snr>
#   (F,) and the per-frame closed forms mq_, mf_ (sample mean) and tq_, tf_ (true gains) of csiq / fixed.
#   With --coh an existing JSON/npz of the same tag is moved to ../data/pilot_corr/old/ (time stamp) first.
# Usage: python pca_csi.py --U 2 --k 6 --nearfar 6
#        python pca_csi.py --coh --U 4 --k 4 --rho 0.7 --eta 0.7 --nearfar 6 --snrs 0 10 20 \
#               --modes fixed csiq switch csiqp ods csiqb
# =========================================================
import argparse, json, math, os, time
import numpy as np
import torch
from scipy.optimize import linear_sum_assignment

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(os.path.dirname(HERE), "data")
SNRS = (0, 5, 10, 15, 20, 25, 30)
S, SEG, D_B = 24, 128, 768
NB = D_B // SEG


def ar1(w, rho):
    """AR(1) filter over the last axis (subbands): CN(0,1) marginals, adjacent correlation rho (rho = 1:
    one value per row, sqrt(1 - rho^2) = 0)."""
    h = np.empty_like(w); h[..., 0] = w[..., 0]
    a = math.sqrt(max(0.0, 1 - rho ** 2))
    for s in range(1, S):
        h[..., s] = rho * h[..., s - 1] + a * w[..., s]
    return h


def channel(rng, F, U, rho, nearfar, delay, legacy=False, full=False):
    """returns (true gains, reported gains), both (F,U,S) magnitudes. The delayed-CSI innovation passes
    through the same AR(1) filter as h (stationary); legacy=True keeps the old i.i.d. innovation. With
    full=True also returns the complex current gains, the complex reported gains and the offsets."""
    w = (rng.standard_normal((F, U, S)) + 1j * rng.standard_normal((F, U, S))) / math.sqrt(2)
    h = ar1(w, rho)
    off = 10 ** ((2 * rng.random((F, U, 1)) - 1) * nearfar / 20.0) if nearfar > 0 else 1.0
    if delay >= 1.0:
        out = (np.abs(h) * off, np.abs(h) * off)
        return out + (h, h, off) if full else out
    v = (rng.standard_normal((F, U, S)) + 1j * rng.standard_normal((F, U, S))) / math.sqrt(2)
    if not legacy:
        v = ar1(v, rho)
    now = delay * h + math.sqrt(1 - delay ** 2) * v
    out = (np.abs(now) * off, np.abs(h) * off)
    return out + (now, h, off) if full else out


def wf(lam, g2, E, s2, iters=60):
    """allocation with per-coefficient gains: lam (k,), g2 (F,k) > 0, E (F,) -> P (F,k)."""
    lo, hi = np.full(g2.shape[0], -60.0), np.full(g2.shape[0], 30.0)
    for _ in range(iters):
        mid = (lo + hi) / 2
        P = np.maximum(0.0, np.sqrt(lam * s2 / (np.exp(mid)[:, None] * g2)) - s2 / g2)
        big = P.sum(1) > E
        lo, hi = np.where(big, mid, lo), np.where(big, hi, mid)
    return np.maximum(0.0, np.sqrt(lam * s2 / (np.exp(hi)[:, None] * g2)) - s2 / g2)


def mse_of(lam, g2, P, s2):
    return (lam * s2 / (g2 * P + s2)).sum(1) + 0.0


class Plan:
    """per user: copies[f][i] = list of (slot, share) for coefficient i; P[f, i] total power."""


def place_sorted(G_sched_j, slots, E, lam, s2, G_val_j=None):
    """blocks on the strongest slots of `slots` (list per frame); returns slot_of_block (F,NB) (-1 unused), P (F,D_B).
    G_val_j (csiqp): gains used in the allocation (13) instead of G_sched_j; the slot order stays by G_sched_j."""
    F = len(slots)
    Gv = G_sched_j if G_val_j is None else G_val_j
    sob = -np.ones((F, NB), int)
    g2 = np.full((F, D_B), 1e-12); m = np.zeros((F, D_B), bool)
    for f in range(F):
        sl = sorted(slots[f], key=lambda x: -G_sched_j[f, x])[:NB]
        for b, x in enumerate(sl):
            sob[f, b] = x; g2[f, b * SEG:(b + 1) * SEG] = Gv[f, x] ** 2; m[f, b * SEG:(b + 1) * SEG] = True
    P = wf(lam, g2, E, s2) * m
    return sob, P


def receive_sorted(c, Gt_j, sob, P, lam, V, s2, rng, genie):
    """one slot per block, slot energy normalized to its plan, blind per-slot gain estimate, LMMSE."""
    F = c.shape[0]
    a = np.sqrt(P / lam)
    x = c * a                                                    # (F, D_B)
    bh = np.zeros_like(c)
    for b in range(NB):
        sl = sob[:, b]; used = sl >= 0
        idx = slice(b * SEG, (b + 1) * SEG)
        Es = P[:, idx].sum(1)
        xb = x[:, idx]
        xb = xb * np.sqrt(Es / ((xb ** 2).sum(1) + 1e-12))[:, None]   # slot energy = plan
        g = np.where(used, Gt_j[np.arange(F), np.maximum(sl, 0)], 0.0)
        y = g[:, None] * xb + math.sqrt(s2) * rng.standard_normal(xb.shape)
        if genie:
            gh2 = g ** 2
        else:
            gh2 = np.maximum(((y ** 2).sum(1) - SEG * s2) / np.maximum(Es, 1e-12), 1e-6)
        Pb = P[:, idx]
        w = gh2[:, None] * Pb / (gh2[:, None] * Pb + s2)
        ab = a[:, idx]
        cb = np.where(ab > 0, y / np.where(ab > 0, ab * np.sqrt(gh2)[:, None], 1.0) * w, 0.0)
        cb = np.where((Es > 0)[:, None] & used[:, None], cb, 0.0)
        bh[:, idx] = cb
    return bh @ V.T


def receive_fixed(c, Gt_j, j, U, n, P, lam, V, s2, rng, genie):
    """the manuscript rule: segment r of rank j on slot j + r U carries block r mod NB at power P_i / m_i."""
    F = c.shape[0]
    a = np.sqrt(P / lam)
    num = np.zeros_like(c); den = np.zeros_like(c)
    for b in range(min(n, NB) if n < NB else NB):
        segs = [r for r in range(n) if r % NB == b]
        m = len(segs)
        idx = slice(b * SEG, (b + 1) * SEG)
        for r in segs:
            s = j + r * U
            Es = P[:, idx].sum(1) / m
            xb = c[:, idx] * a[:, idx] / math.sqrt(m)
            xb = xb * np.sqrt(Es / ((xb ** 2).sum(1) + 1e-12))[:, None]
            g = Gt_j[:, s]
            y = g[:, None] * xb + math.sqrt(s2) * rng.standard_normal(xb.shape)
            gh2 = g ** 2 if genie else np.maximum(((y ** 2).sum(1) - SEG * s2) / np.maximum(Es, 1e-12), 1e-6)
            amp = a[:, idx] / math.sqrt(m)
            num[:, idx] += np.sqrt(gh2)[:, None] * amp * y               # LMMSE numerator (without lambda)
            den[:, idx] += gh2[:, None] * amp ** 2
    chat = np.where(den > 0, lam * num / (lam * den + s2), 0.0)
    return chat @ V.T


def hungarian_sets(util, quotas):
    """util (F,U,S); returns per frame the slot list of every rank (quotas[j] slots)."""
    F, U, _ = util.shape
    jq = np.concatenate([[j] * q for j, q in enumerate(quotas)])
    out = []
    for f in range(F):
        rows, cols = linear_sum_assignment(-util[f][jq])               # (pairs x slots), pairs <= S
        sets = [[] for _ in range(U)]
        for r, ccol in zip(rows, cols):
            sets[jq[r]].append(ccol)
        out.append(sets)
    return out


def greedy_counts(Gs, lam, s2, crit="mse", quotas=None, Gval=None):
    """csiq (crit mse) and csic (crit cos): slot count per user set greedily by the total MSE or by the
    sum of sqrt(1 - MSE / sum lambda). Gs (F,U,S) reported gains. With quotas (csiqe), user j receives
    at most quotas[j] slots, so the counts equal those of the load masks and only the placement is greedy.
    Gval (csiqp): gains used in the MSE of (13) (conditional-mean gains); candidates and order still by Gs."""
    F, U, _ = Gs.shape
    cap = np.array(quotas if quotas is not None else [S] * U)
    tot = lam.sum()
    score = (lambda m: m) if crit == "mse" else (lambda m: -np.sqrt(np.maximum(0.0, 1.0 - m / tot)))
    avail = np.ones((F, S), bool)
    sets = [[[] for _ in range(U)] for _ in range(F)]
    ar = np.arange(F)

    def user_mse(j, extra=None):
        g2 = np.full((F, D_B), 1e-12); m = np.zeros((F, D_B), bool); n = np.zeros(F)
        for f in range(F):
            if Gval is None:
                gl = sorted([Gs[f, j, x] for x in sets[f][j]] + ([Gs[f, j, extra[f]]] if extra is not None else []), reverse=True)
            else:
                xs = sorted(sets[f][j] + ([extra[f]] if extra is not None else []), key=lambda x: -Gs[f, j, x])
                gl = [Gval[f, j, x] for x in xs]
            n[f] = len(gl)
            for b in range(min(len(gl), NB)):
                g2[f, b * SEG:(b + 1) * SEG] = gl[b] ** 2; m[f, b * SEG:(b + 1) * SEG] = True
        P = wf(lam, g2, n * SEG, s2) * m
        return (lam * s2 / (g2 * P + s2) * m).sum(1) + (lam * ~m).sum(1)
    for j in range(U):
        s0 = np.where(avail, Gs[:, j], -1.0).argmax(1); avail[ar, s0] = False
        for f in range(F): sets[f][j].append(s0[f])
    cur = np.stack([user_mse(j) for j in range(U)], 1)
    for _ in range(S - U):
        gain = np.full((F, U), -np.inf); cand = np.zeros((F, U), int)
        for j in range(U):
            sj = np.where(avail, Gs[:, j], -1.0).argmax(1); cand[:, j] = sj
            gain[:, j] = score(cur[:, j]) - score(user_mse(j, sj))
            full = np.array([len(sets[f][j]) >= cap[j] for f in range(F)])
            gain[full, j] = -np.inf
        jb = gain.argmax(1)
        for f in range(F):
            sets[f][jb[f]].append(cand[f, jb[f]]); avail[f, cand[f, jb[f]]] = False
        cur = np.stack([user_mse(j) for j in range(U)], 1)
    return sets


def plan_mse_sorted(gj, sob, P, lam, s2):
    """closed-form MSE of a sorted plan (csiq family, one copy per block) with known slot gains gj (F,S);
    blocks without a slot (sob = -1) and coefficients with P_i = 0 add lam_i."""
    gb = np.where(sob >= 0, np.take_along_axis(gj, np.maximum(sob, 0), 1), 0.0)
    g2 = np.repeat(gb ** 2, SEG, axis=1)
    return (lam * s2 / (g2 * P + s2)).sum(1)


def plan_mse_fixed(gj, j, U, n, P, lam, s2):
    """closed-form MSE of the load masks of rank j with known slot gains gj (F,S): block b on the m_b slots
    j + r U (r mod NB = b) at P_i / m_b each, maximum-ratio combined: lam_i s2 / (sum_copies g_s^2 P_i / m_b + s2)."""
    g2 = np.zeros((gj.shape[0], D_B))
    for b in range(min(n, NB)):
        segs = [r for r in range(n) if r % NB == b]
        g2[:, b * SEG:(b + 1) * SEG] = (sum(gj[:, j + r * U] ** 2 for r in segs) / len(segs))[:, None]
    return (lam * s2 / (g2 * P + s2)).sum(1)


def boot_ci(d, idx):
    """paired bootstrap 95% CI of the mean of the per-frame differences d (F,) over the resamples idx (B,F)."""
    m = d[idx].mean(1)
    return [float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))]


def run(a, lam, V, ct, Ete, store=None):
    """means per mode and SNR; with a dict `store`, also fills it with the per-frame arrays of the .npz."""
    U, k = a.U, a.k
    F = a.frames
    snrs = tuple(getattr(a, "snrs", None) or SNRS)
    legacy = getattr(a, "legacy_delay", False)
    nsamp = getattr(a, "nsamp", 32)
    res = {}
    count_cache = {}                                   # greedy supports per (criterion, snr), shared by the csiq family
    percs = {}                                    # per-frame per-user CosSim (U,F) of fixed and csiq (for switch)
    allcs = {}                                    # per-frame per-user CosSim (U,F) of every mode run
    swx = {}                                      # per-frame arrays of switch

    def draws(snr):
        rng = np.random.default_rng(123 + snr)
        Gt, Gs, _, hp, off = channel(rng, F, U, a.rho, a.nearfar, a.delay, legacy=legacy, full=True)
        pos = rng.integers(0, ct.shape[0], (F, U))
        return rng, Gt, Gs, hp, off, pos

    def counts_of(key, Gs, s2, Gval=None):
        if key not in count_cache:
            count_cache[key] = greedy_counts(Gs, lam, s2, "cos" if key[0] == "cos" else "mse", Gval=Gval)
        return count_cache[key]

    def fixed_P(n, s2):
        kk = min(n * SEG, D_B)
        P = np.zeros((F, D_B))
        P[:, :kk] = wf(lam[:kk], np.ones((F, kk)), np.full(F, 3072.0 / U), s2)
        return P

    def realize(mode, snr):
        """per-frame per-user CosSim (U,F) of one scheme on the draws of seed 123 + snr."""
        rng, Gt, Gs, hp, off, pos = draws(snr)
        s2 = 10 ** (-snr / 10.0)
        util = np.log1p(10 ** (snr / 10.0) * Gs ** 2)
        quotas = [math.ceil((S - j) / U) for j in range(U)]
        Gv = None
        if mode == "csi":
            sets = hungarian_sets(util, quotas)
        elif mode in ("csiq", "csic", "csiqw", "csiqb"):
            sets = counts_of(("cos" if mode == "csic" else "mse", snr), Gs, s2)
        elif mode == "csiqp":
            if a.delay >= 1.0:                    # g_bar = g': the supports and (13) of csiq
                sets = counts_of(("mse", snr), Gs, s2)
            else:
                Gv = np.sqrt(a.delay ** 2 * Gs ** 2 + (1 - a.delay ** 2) * off ** 2)
                sets = counts_of(("mse_p", snr), Gs, s2, Gval=Gv)
        elif mode == "csiqe":
            sets = greedy_counts(Gs, lam, s2, "mse", quotas)
        elif mode in ("ods", "odsm"):
            sets = hungarian_sets(util, [S // k] * U)
        cs = []
        for j in range(U):
            c = ct[pos[:, j]]
            if mode in ("fixed", "fixedb"):
                n = quotas[j]; kk = min(n * SEG, D_B)
                if mode == "fixed":
                    P = fixed_P(n, s2)
                else:
                    P = np.zeros((F, D_B))
                    mc = np.array([len([r for r in range(n) if r % NB == i // SEG]) for i in range(kk)], float)
                    P[:, :kk] = (3072.0 / U) * lam[:kk] * mc / (lam[:kk] * mc).sum()
                bh = receive_fixed(c, Gt[:, j], j, U, n, P, lam, V, s2, rng, a.genie)
            elif mode == "odsfix":
                n = S // k; kk = min(n * SEG, D_B)
                P = np.zeros((F, D_B))
                P[:, :kk] = wf(lam[:kk], np.ones((F, kk)), np.full(F, 3072.0 / k), s2)
                # contiguous part j: slots j n .. j n + n - 1, read with the same receiver (one copy per block)
                sob = -np.ones((F, NB), int)
                for b in range(min(n, NB)):
                    sob[:, b] = j * n + b
                bh = receive_sorted(c, Gt[:, j], sob, P, lam, V, s2, rng, a.genie)
            else:
                sl = [sets[f][j] for f in range(F)]
                n = np.array([len(x) for x in sl])
                E = n * SEG if mode in ("csiq", "csic", "csiqe", "csiqw", "csiqb", "csiqp") else (np.full(F, 3072.0 / k) if mode == "ods" else np.full(F, 3072.0 / U))
                sob, P = place_sorted(Gs[:, j], sl, E, lam, s2, None if Gv is None else Gv[:, j])
                if mode in ("csiqw", "csiqb"):
                    sent = np.repeat(sob >= 0, SEG, axis=1)          # (F, D_B) coefficients of the used groups
                    if mode == "csiqw":
                        P = wf(lam, np.where(sent, 1.0, 1e-12), E, s2) * sent
                    else:
                        P = E[:, None] * lam * sent / (lam * sent).sum(1, keepdims=True)
                bh = receive_sorted(c, Gt[:, j], sob, P, lam, V, s2, rng, a.genie)
            b = Ete[pos[:, j]]
            cs.append((bh * b).sum(1) / (np.linalg.norm(bh, axis=1) * np.linalg.norm(b, axis=1) + 1e-12))
        cs = np.stack(cs)
        if mode in ("fixed", "csiq"):
            percs[(mode, snr)] = cs
        return cs

    def switch(snr):
        """coherence-aware choice between the csiq plan (built on g') and the load masks, per frame."""
        rng, Gt, Gs, hp, off, pos = draws(snr)
        s2 = 10 ** (-snr / 10.0)
        quotas = [math.ceil((S - j) / U) for j in range(U)]
        sets = counts_of(("mse", snr), Gs, s2)
        plans, Pfix = [], []
        for j in range(U):
            sl = [sets[f][j] for f in range(F)]
            plans.append(place_sorted(Gs[:, j], sl, np.array([len(x) for x in sl]) * SEG, lam, s2))
            Pfix.append(fixed_P(quotas[j], s2))
        eta = min(a.delay, 1.0); bsd = math.sqrt(max(0.0, 1 - eta ** 2))
        T = nsamp if eta < 1.0 else 1             # eta = 1: every sample equals h'
        rng2 = np.random.default_rng(999 + snr)   # separate generator: the main draws stay paired
        m_q, m_f = np.zeros(F), np.zeros(F)
        for _ in range(T):
            w = (rng2.standard_normal((F, U, S)) + 1j * rng2.standard_normal((F, U, S))) / math.sqrt(2)
            g = np.abs(eta * hp + bsd * (w if legacy else ar1(w, a.rho))) * off
            for j in range(U):
                m_q += plan_mse_sorted(g[:, j], *plans[j], lam, s2)
                m_f += plan_mse_fixed(g[:, j], j, U, quotas[j], Pfix[j], lam, s2)
        choose = m_q < m_f                        # announce csiq iff its conditional expected sum MSE is lower
        t_q, t_f = np.zeros(F), np.zeros(F)       # closed forms at the true current gains Gt (diagnostic only)
        for j in range(U):
            t_q += plan_mse_sorted(Gt[:, j], *plans[j], lam, s2)
            t_f += plan_mse_fixed(Gt[:, j], j, U, quotas[j], Pfix[j], lam, s2)
        choose_cf = t_q < t_f
        cq = percs[("csiq", snr)] if ("csiq", snr) in percs else realize("csiq", snr)
        cf = percs[("fixed", snr)] if ("fixed", snr) in percs else realize("fixed", snr)
        cs = np.where(choose[None, :], cq, cf)
        better = np.maximum(cq.mean(0), cf.mean(0))
        swx[snr] = {"choose": choose, "choose_cf": choose_cf, "mq": m_q / T, "mf": m_f / T, "tq": t_q, "tf": t_f}
        return cs, {"frac_csiq": float(choose.mean()),
                    "best_member_cos": float(max(cq.mean(), cf.mean())),
                    "rule_loss": float(max(cq.mean(), cf.mean()) - cs.mean()),
                    "oracle_cf_cos": float(np.where(choose_cf[None, :], cq, cf).mean()),
                    "frac_csiq_cf": float(choose_cf.mean()),
                    "decision_loss": float((better - cs.mean(0)).mean()),
                    "oracle_cos": float(better.mean()), "pred_mse_csiq": float(m_q.mean() / T),
                    "pred_mse_fixed": float(m_f.mean() / T), "true_mse_csiq": float(t_q.mean()),
                    "true_mse_fixed": float(t_f.mean()), "nsamp": T}

    for mode in a.modes:
        if mode in ("ods", "odsm", "odsfix") and (k is None or U > k):
            continue
        out = {}
        for snr in snrs:
            if mode == "switch":
                cs, extra = switch(snr)
            else:
                cs, extra = realize(mode, snr), {}
            allcs[(mode, snr)] = cs
            out[snr] = {"cos": float(cs.mean()), "weakest": float(cs.min(0).mean()), **extra}
            print(f"U={U} k={k} {mode} snr={snr} cos={out[snr]['cos']:.4f} weakest={out[snr]['weakest']:.4f}"
                  + (f" frac_csiq={extra['frac_csiq']:.3f} rule_loss={extra['rule_loss']:.5f}"
                     f" oracle_cf={extra['oracle_cf_cos']:.4f} loss={extra['decision_loss']:.5f}" if extra else ""), flush=True)
        res[mode] = out
    # paired bootstrap CIs over frames (same resamples for every mode of one SNR; separate generator)
    for snr in snrs:
        idx = np.random.default_rng(555 + snr).integers(0, F, (1000, F))
        if "fixed" in res:
            cf = allcs[("fixed", snr)]
            for mode in res:
                if mode != "fixed":
                    cm = allcs[(mode, snr)]
                    res[mode][snr].update(
                        gain_fixed=float(cm.mean() - cf.mean()), gain_fixed_ci=boot_ci(cm.mean(0) - cf.mean(0), idx),
                        gain_fixed_weakest=float(cm.min(0).mean() - cf.min(0).mean()),
                        gain_fixed_weakest_ci=boot_ci(cm.min(0) - cf.min(0), idx))
        if "switch" in res:
            fq, ff, fs = percs[("csiq", snr)].mean(0), percs[("fixed", snr)].mean(0), allcs[("switch", snr)].mean(0)
            mb = np.maximum(fq[idx].mean(1), ff[idx].mean(1)) - fs[idx].mean(1)
            res["switch"][snr]["rule_loss_ci"] = [float(np.percentile(mb, 2.5)), float(np.percentile(mb, 97.5))]
    if store is not None:
        for (mode, snr), cs in allcs.items():
            store[f"cos_{mode}_{snr}"] = cs.T.astype(np.float32)
        for snr in snrs:
            store[f"pos_{snr}"] = draws(snr)[5].astype(np.int32)
        for snr, d in swx.items():
            for key, v in d.items():
                store[f"{key}_{snr}"] = v
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--U", type=int, required=True)
    ap.add_argument("--k", type=int, default=None)
    ap.add_argument("--rho", type=float, default=0.7)
    ap.add_argument("--nearfar", type=float, default=0.0)
    ap.add_argument("--delay", "--eta", dest="delay", type=float, default=1.0,
                    help="gain age eta: h = eta h' + sqrt(1 - eta^2) v (1 = current-block gains)")
    ap.add_argument("--legacy_delay", action="store_true",
                    help="old i.i.d. innovation over the subbands (current-block correlation rho eta^2)")
    ap.add_argument("--genie", action="store_true")
    ap.add_argument("--frames", type=int, default=2000)
    ap.add_argument("--snrs", type=int, nargs="+", default=list(SNRS))
    ap.add_argument("--nsamp", type=int, default=32, help="conditional samples of h | h' for mode switch")
    ap.add_argument("--modes", nargs="+", default=["fixed", "csi", "csiq", "ods", "odsfix"])
    ap.add_argument("--coh", action="store_true", help="coherence naming pca_csi_coh_rho<rho>_eta<eta>_U<U>[_nf<X>]")
    ap.add_argument("--suffix", default="", help="appended to the output tag (extra modes of a finished run)")
    a = ap.parse_args()
    C = torch.load(os.path.join(DATA, "emb_ag_news_bert-base-uncased.pt"), map_location="cpu", weights_only=False)
    Etr, Ete = C["train"].double().numpy(), C["test"].double().numpy()
    lam, V = np.linalg.eigh(Etr.T @ Etr / Etr.shape[0]); o = np.argsort(lam)[::-1]
    lam, V = np.maximum(lam[o], 1e-12), V[:, o]
    store = {}
    res = run(a, lam, V, Ete @ V, Ete, store)
    if a.coh:
        tag = f"coh_rho{a.rho}_eta{a.delay}_U{a.U}" + (f"_nf{int(a.nearfar)}" if a.nearfar else "") \
            + ("_legacy" if a.legacy_delay and a.delay < 1 else "") + ("_genie" if a.genie else "") + a.suffix
    else:
        tag = f"U{a.U}" + (f"_k{a.k}" if a.k else "") + f"_rho{a.rho}" + (f"_nf{int(a.nearfar)}" if a.nearfar else "") \
            + ((f"_d{a.delay}" + ("" if a.legacy_delay else "_stat")) if a.delay < 1 else "") + ("_genie" if a.genie else "") + a.suffix
    out_dir = os.path.join(DATA, "pilot_corr")
    os.makedirs(out_dir, exist_ok=True)
    base = os.path.join(out_dir, f"pca_csi_{tag}")
    if a.coh:                                     # never overwrite a coherence result: move it to old/ first
        stamp = time.strftime("%Y%m%d-%H%M%S")
        for ext in (".json", ".npz"):
            if os.path.exists(base + ext):
                os.makedirs(os.path.join(out_dir, "old"), exist_ok=True)
                os.replace(base + ext, os.path.join(out_dir, "old", f"pca_csi_{tag}.{stamp}{ext}"))
                print(f"moved the existing {base + ext} to old/", flush=True)
    np.savez_compressed(base + ".npz", **store)
    json.dump({"args": vars(a), "res": res}, open(base + ".json", "w"), indent=1)


if __name__ == "__main__":
    main()
