# =========================================================
# pca_csi.py — channel-aware dynamic masks for the closed-form PCA code (2026-11-04).
# Fading varies over the N = 24 slots of a block: every user has its own AR(1) complex Gaussian gain
# process over the slots (correlation rho between adjacent slots, CN(0,1) marginals), independent
# across users, with an optional near-far offset uniform in [-X, X] dB per user and frame. The
# scheduler (receiver) knows the gains of the active users, either of the current block (perfect)
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
#   ods       Static ODS-k with the same scheduler at a load U <= k: every active user gets d_s/k uses
#             (N/k slots chosen by the Hungarian assignment among all free slots), energy d_s/k
#   odsm      ods with the energy d_s/U of the proposal (the matched-energy reference of the manuscript)
#   odsfix    Static ODS-k with fixed contiguous parts and no CSI at the transmitter (the classical one)
# Mask values (2026-11-04): a mask is its slot support times its amplitudes. These modes keep a support and
# replace the amplitudes, on the same draws and, for the csiq family, the same greedy grant:
#   fixedb    support of fixed, every used entry at one amplitude (binary mask, P_i ~ lambda_i m_i)
#   csiqw     support of csiq, amplitudes of the allocation (12) for unit gain (SNR-aware, gain-blind)
#   csiqb     support of csiq, every used entry at one amplitude (binary mask, P_i ~ lambda_i)
# Output: ../data/pilot_corr/pca_csi_<tag><suffix>.json.  Usage: python pca_csi.py --U 2 --k 6 --nearfar 6
# =========================================================
import argparse, json, math, os
import numpy as np
import torch
from scipy.optimize import linear_sum_assignment

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(os.path.dirname(HERE), "data")
SNRS = (0, 5, 10, 15, 20, 25, 30)
S, SEG, D_B = 24, 128, 768
NB = D_B // SEG


def channel(rng, F, U, rho, nearfar, delay):
    """returns (true gains, scheduler gains), both (F,U,S) magnitudes."""
    w = (rng.standard_normal((F, U, S)) + 1j * rng.standard_normal((F, U, S))) / math.sqrt(2)
    h = np.empty_like(w); h[..., 0] = w[..., 0]
    a = math.sqrt(max(0.0, 1 - rho ** 2))
    for s in range(1, S):
        h[..., s] = rho * h[..., s - 1] + a * w[..., s]
    off = 10 ** ((2 * rng.random((F, U, 1)) - 1) * nearfar / 20.0) if nearfar > 0 else 1.0
    if delay >= 1.0:
        return np.abs(h) * off, np.abs(h) * off
    v = (rng.standard_normal((F, U, S)) + 1j * rng.standard_normal((F, U, S))) / math.sqrt(2)
    now = delay * h + math.sqrt(1 - delay ** 2) * v
    return np.abs(now) * off, np.abs(h) * off


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


def place_sorted(G_sched_j, slots, E, lam, s2):
    """blocks on the strongest slots of `slots` (list per frame); returns slot_of_block (F,NB) (-1 unused), P (F,D_B)."""
    F = len(slots)
    sob = -np.ones((F, NB), int)
    g2 = np.full((F, D_B), 1e-12); m = np.zeros((F, D_B), bool)
    for f in range(F):
        sl = sorted(slots[f], key=lambda x: -G_sched_j[f, x])[:NB]
        for b, x in enumerate(sl):
            sob[f, b] = x; g2[f, b * SEG:(b + 1) * SEG] = G_sched_j[f, x] ** 2; m[f, b * SEG:(b + 1) * SEG] = True
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


def greedy_counts(Gs, lam, s2, crit="mse", quotas=None):
    """csiq (crit mse) and csic (crit cos): slot count per user set greedily by the total MSE or by the
    sum of sqrt(1 - MSE / sum lambda). Gs (F,U,S) scheduler gains. With quotas (csiqe), user j receives
    at most quotas[j] slots, so the counts equal those of the load masks and only the placement is greedy."""
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
            gl = sorted([Gs[f, j, x] for x in sets[f][j]] + ([Gs[f, j, extra[f]]] if extra is not None else []), reverse=True)
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


def run(a, lam, V, ct, Ete):
    U, k = a.U, a.k
    res = {}
    grants = {}                                   # greedy grant per (criterion, snr), shared by the csiq family
    for mode in a.modes:
        if mode in ("ods", "odsm", "odsfix") and (k is None or U > k):
            continue
        out = {}
        for snr in SNRS:
            rng = np.random.default_rng(123 + snr)
            s2 = 10 ** (-snr / 10.0)
            Gt, Gs = channel(rng, a.frames, U, a.rho, a.nearfar, a.delay)
            pos = rng.integers(0, ct.shape[0], (a.frames, U))
            util = np.log1p(10 ** (snr / 10.0) * Gs ** 2)
            quotas = [math.ceil((S - j) / U) for j in range(U)]
            if mode == "csi":
                sets = hungarian_sets(util, quotas)
            elif mode in ("csiq", "csic", "csiqw", "csiqb"):
                key = ("cos" if mode == "csic" else "mse", snr)
                if key not in grants:
                    grants[key] = greedy_counts(Gs, lam, s2, key[0])
                sets = grants[key]
            elif mode == "csiqe":
                sets = greedy_counts(Gs, lam, s2, "mse", quotas)
            elif mode in ("ods", "odsm"):
                sets = hungarian_sets(util, [S // k] * U)
            cs = []
            for j in range(U):
                c = ct[pos[:, j]]
                if mode in ("fixed", "fixedb"):
                    n = quotas[j]; kk = min(n * SEG, D_B)
                    P = np.zeros((a.frames, D_B))
                    if mode == "fixed":
                        P[:, :kk] = wf(lam[:kk], np.ones((a.frames, kk)), np.full(a.frames, 3072.0 / U), s2)
                    else:
                        mc = np.array([len([r for r in range(n) if r % NB == i // SEG]) for i in range(kk)], float)
                        P[:, :kk] = (3072.0 / U) * lam[:kk] * mc / (lam[:kk] * mc).sum()
                    bh = receive_fixed(c, Gt[:, j], j, U, n, P, lam, V, s2, rng, a.genie)
                elif mode == "odsfix":
                    n = S // k; kk = min(n * SEG, D_B)
                    P = np.zeros((a.frames, D_B))
                    P[:, :kk] = wf(lam[:kk], np.ones((a.frames, kk)), np.full(a.frames, 3072.0 / k), s2)
                    # contiguous part j: slots j n .. j n + n - 1, read with the same receiver (one copy per block)
                    sob = -np.ones((a.frames, NB), int)
                    for b in range(min(n, NB)):
                        sob[:, b] = j * n + b
                    bh = receive_sorted(c, Gt[:, j], sob, P, lam, V, s2, rng, a.genie)
                else:
                    sl = [sets[f][j] for f in range(a.frames)]
                    n = np.array([len(x) for x in sl])
                    E = n * SEG if mode in ("csiq", "csic", "csiqe", "csiqw", "csiqb") else (np.full(a.frames, 3072.0 / k) if mode == "ods" else np.full(a.frames, 3072.0 / U))
                    sob, P = place_sorted(Gs[:, j], sl, E, lam, s2)
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
            out[snr] = {"cos": float(cs.mean()), "weakest": float(cs.min(0).mean())}
            print(f"U={U} k={k} {mode} snr={snr} cos={out[snr]['cos']:.4f} weakest={out[snr]['weakest']:.4f}", flush=True)
        res[mode] = out
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--U", type=int, required=True)
    ap.add_argument("--k", type=int, default=None)
    ap.add_argument("--rho", type=float, default=0.7)
    ap.add_argument("--nearfar", type=float, default=0.0)
    ap.add_argument("--delay", type=float, default=1.0)
    ap.add_argument("--genie", action="store_true")
    ap.add_argument("--frames", type=int, default=2000)
    ap.add_argument("--modes", nargs="+", default=["fixed", "csi", "csiq", "ods", "odsfix"])
    ap.add_argument("--suffix", default="", help="appended to the output tag (extra modes of a finished run)")
    a = ap.parse_args()
    C = torch.load(os.path.join(DATA, "emb_ag_news_bert-base-uncased.pt"), map_location="cpu", weights_only=False)
    Etr, Ete = C["train"].double().numpy(), C["test"].double().numpy()
    lam, V = np.linalg.eigh(Etr.T @ Etr / Etr.shape[0]); o = np.argsort(lam)[::-1]
    lam, V = np.maximum(lam[o], 1e-12), V[:, o]
    res = run(a, lam, V, Ete @ V, Ete)
    tag = f"U{a.U}" + (f"_k{a.k}" if a.k else "") + f"_rho{a.rho}" + (f"_nf{int(a.nearfar)}" if a.nearfar else "") \
        + (f"_d{a.delay}" if a.delay < 1 else "") + ("_genie" if a.genie else "") + a.suffix
    os.makedirs(os.path.join(DATA, "pilot_corr"), exist_ok=True)
    json.dump({"args": vars(a), "res": res}, open(os.path.join(DATA, "pilot_corr", f"pca_csi_{tag}.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
