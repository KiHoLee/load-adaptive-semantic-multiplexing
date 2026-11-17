# =========================================================
# overlap_bound.py — can overlapping real-valued masks beat the non-overlapping load masks when the receiver is the
# joint (multi-user) LMMSE estimator? (2026-11-04). For a Gaussian source and linear analog transmission the
# joint LMMSE estimator is the MMSE estimator, so no receiver, SIC included, does better for a given
# transmit design: the comparison bounds what overlap with SIC can gain.
# Model: the block of d_s = 24 x 128 real uses, every user with the PCA spectrum lam of the AG News BERT
# embeddings in 6 groups of 128, per-user energy d_s/U, gain g_u per user (equal, or near-far spread from
# +nf to -nf dB over the users), noise s2 = 10^(-SNR/10). Restriction (keeps the problem small): use i of
# every slot mixes only the coefficients i of the groups (128 independent problems of 24 uses x 6U
# coefficients, coupled by the per-user energy). Schemes, all with the joint LMMSE receiver:
#   closed    the load masks (4) with the allocation (12), as in the paper
#   nonoverlap  the non-overlapping member (JSON key 'disjoint'): the uses of each user in the load masks, any
#             linear code of the user on those uses, optimized
#   overlap   any real 24 x 6U transmit matrix per use index, i.e. real-valued overlapping masks, optimized
#             (from the closed form and from random starts; the problem is not convex)
# Output: ../data/extra/overlap_bound_U<U>_nf<X>.json.  Usage: python overlap_bound.py --U 6 --nearfar 6
#
# Extension (2026-11-16, numerical search checks of the generalized real-valued masks). Given any of
# --content, --gains or --K, the script runs run2 below and writes ../data/extra/overlap_bound2_<case>.json;
# without them it runs the original search above unchanged (same seeds, same output file and content).
# Same restricted model with S = 6K slots of 128 uses (K = 4: 24 slots, K = 2: 12 slots, d_s = K d_b),
# 128 problems of S uses x 6U coefficients, per-user energy, joint LMMSE receiver with a known gain.
#   --content indep        AG News spectrum, independent users (kappa = 0): the original model
#             coco|flickr  caption embeddings ../data/emb_<ds>_caps_bert-base-uncased.pt (train: 8000 images x 5
#                          captions): lam, V from the second moment of the 40000 fit captions; per image
#                          S = sum of the 5 coefficient vectors, C_ii = (mean S_i^2 - 5 lam_i)/20,
#                          kappa_i = clip(C_ii/lam_i, 0, 0.99)
#             mix:KAPPA[:ds] mixtures b_u = normalize(sqrt(KAPPA) b_s + sqrt(1-KAPPA) b_{p,u}) of fit captions of ds
#                          (default coco): b_s one caption of the image, b_{p,u} captions of U other, distinct
#                          images; one tuple of U users per fit image (default_rng(0)); lam, kappa fitted on the
#                          8000 U mixtures as above with U in place of 5
#           The U users' coefficient i follows (A1): covariance lam_i[(1 - kappa_i) I + kappa_i 1 1^T], in the
#           search objective and in the joint LMMSE receiver (exact inverse per coefficient).
#   --gains equal          one known gain per user (1, or the --nearfar spread of the original script)
#           rayleigh:D     D draws of independent |h_u| ~ |CN(0,1)| per user and block (offsets uniform in +-nearfar
#                          dB if given), numpy default_rng(123 + SNR); closed forms at the unit design gain, the
#                          search with the drawn gains known at the transmitter, the receiver knows the gains
#           slot:RHO[:D]   D draws (default --draws) of per-slot gains, AR(1) over the 24 slots with correlation RHO
#                          (pca_csi.channel with perfect CSI, offsets uniform in +-nearfar dB), K = 4 only
# Members (all with the joint LMMSE receiver):
#   load      the load masks (4) with (12) at the design gain, d_s/U per user: the user gain for --gains equal with
#             indep content, the unit gain otherwise (shared content: the design gain of Proposition 2, also with
#             --nearfar); at U > K the leading d_s/U coefficients per user
#   closed    the closed form of the condition: load (indep content, block fading); Proposition 2 masks
#             (shared content, U <= K: modes common / difference, p_{i,m} = max(0, sqrt(lam_i s2/nu) - s2/a_{i,m}),
#             W_i = lam_i^{-1/2}(sqrt(p_{i,1}) I + (sqrt(p_{i,0}) - sqrt(p_{i,1})) 1 1^T/U) on U aligned uses of
#             coefficient i, split evenly over the copies of the load masks); pruned Proposition 2 (shared
#             content, U > K: water-filling over all U d_b modes, the active mode with the smallest
#             MSE reduction lam a - lam s2/(p + s2/a) dropped one at a time with nu re-solved until at most d_s modes
#             are active (free placement); coefficient i needs m_i uses (its kept modes) in its use-index problem;
#             if a problem of the original split would need more than S uses, the coefficients are relabeled over
#             the 128 problems (6 each, longest-first to the least loaded problem; under (A1) and block fading the
#             coefficients are independent, so any such split is an equally valid restricted model with the same
#             objective and per-user energies), and closed and search run in that split (load and the non-overlapping
#             member keep the original split; sum NMSE does not depend on the split); only if the relabeling fails, the pruning
#             with at most S modes per problem of the original split is used (reported as split); difference
#             modes from a fixed orthonormal complement (Hadamard for U = 2^n, real DFT otherwise) with the user
#             index shifted cyclically by i and the kept columns rotated cyclically with i, each user then scaled
#             exactly to d_s/U); channel-aware masks (13) on the greedy support (csiq of pca_csi.py, energy
#             n_u x 128) for --gains slot
#   nonoverlap  (JSON key 'disjoint') values optimized on the uses of each user in load (in csiq for --gains slot),
#             the non-overlapping member
#   search    any real S x 6U transmit matrix per use index, Adam --steps from closed (lr 0.01) and from
#             --restarts random starts (N(0,1) in energy units, lr 0.02), each user normalized to its energy in
#             closed (--search_energy equal: d_s/U); Adam runs in energy units W = A sqrt(lam) (amplitudes A of
#             weak coefficients are ~100x larger, and Adam moves every entry by about lr per step); members and
#             starts of --chunk draws run as one batch (independent problems: identical updates to separate runs,
#             same result for any --chunk), the best iterate per problem is kept; the gradient of diag(M^{-1}) is
#             the exact -C diag(g) C (DiagInv), cheaper than autograd through the Cholesky
# Output name: overlap_bound2_<content>_<gains>_K<K>_U<U>[_nf<X>][_s<steps>][_r<restarts>][_Eeq]<tag>.json with
# <gains> = equal | rayleigh-D<D> | slot-rho<RHO>-D<D> (non-default --steps/--restarts/--search_energy in the
# name, so smoke runs never overwrite a full run).
# Reported per SNR (per draw and mean over draws): sum NMSE = sum_u ||c_u - c_hat_u||^2 / sum_i lam_i of every
# member, gap_pct = 100 (closed - search)/closed (positive: the search is lower), the energy-weighted overlap
# index t of the per-coefficient Gram blocks, the search's share of each user's received energy on slots shared
# with another user (search_shared_energy) and of its transmit energy on slots where closed gives the user none
# (search_tx_outside_closed; with slot gains a positive gap with ~0 sharing is a better slot assignment than the
# greedy support, not overlap), Proposition-2 checks (modal formula vs the joint LMMSE trace in the model,
# per-user energy deviation from d_s/U), pruning bookkeeping, csiq slot counts, runtime.
# Usage: python overlap_bound.py --content coco --gains equal --K 4 --U 4 --snrs -5 0 10
#        python overlap_bound.py --content indep --gains slot:0.7 --U 4 --nearfar 6 --snrs 0 10 20
# =========================================================
import argparse, json, math, os, time
import numpy as np
import torch
from pca_csi import DATA, wf

S, SEG, NB = 24, 128, 6


def mse_users(A, g, lam_g, s2, U):
    """A (SEG, S, NB*U) transmit amplitudes; g (U,) gains; lam_g (SEG, NB*U) variances -> per-user MSE (U,)."""
    B = A * g.repeat_interleave(NB)[None, None, :]
    M = torch.diag_embed(1.0 / lam_g) + B.transpose(1, 2) @ B / s2
    C = torch.cholesky_inverse(torch.linalg.cholesky(M))
    return torch.diagonal(C, dim1=1, dim2=2).reshape(SEG, U, NB).sum((0, 2))


def energy_users(A, lam_g, U):
    return (A ** 2 * lam_g[:, None, :]).sum(1).reshape(SEG, U, NB).sum((0, 2))


def normalize(W, lam_g, E, U):
    sc = torch.sqrt(E / energy_users(W, lam_g, U))
    return W * sc.repeat_interleave(NB)[None, None, :]


def closed_form(lam, g, s2, U, S_=S):
    """the paper's scheme: load masks, groups repeated cyclically, allocation (12) with the user gain.
    S_ slots (default 24); with fewer slots than groups a user sends its leading groups (U > K)."""
    A = torch.zeros(SEG, S_, NB * U, dtype=torch.float64)
    lam_g = torch.tensor(lam.reshape(NB, SEG).T.copy()).repeat(1, U)
    for j in range(U):
        slots = [s for s in range(S_) if s % U == j]
        nb = min(len(slots), NB)
        P = wf(lam[:nb * SEG], np.full((1, nb * SEG), float(g[j]) ** 2), np.array([S_ * SEG / U]), s2)[0]
        for b in range(nb):
            own = [s for t, s in enumerate(slots) if t % NB == b]          # copies of group b
            p = torch.tensor(P[b * SEG:(b + 1) * SEG] / len(own))         # energy split over the copies
            for s in own:
                A[:, s, j * NB + b] = torch.sqrt(p / lam_g[:, j * NB + b])
    return A, lam_g


def optimize(W0, mask, g, lam_g, s2, U, E, steps, lr):
    W = W0.clone().requires_grad_(True)
    opt = torch.optim.Adam([W], lr=lr)
    best, bestA = float("inf"), None
    for it in range(steps):
        A = normalize(W * mask, lam_g, E, U)
        m = mse_users(A, g, lam_g, s2, U)
        loss = m.sum()
        opt.zero_grad(); loss.backward(); opt.step()
        if loss.item() < best:
            best, bestA = loss.item(), A.detach().clone()
    return bestA


# ---------------------------------------------------------------------------------------------------------
# extension 2026-11-16 (run2): shared content (A1), random and per-slot gains, K = 2, closed forms
# ---------------------------------------------------------------------------------------------------------
D_B = NB * SEG
CAPS = {"coco": "emb_coco_caps_bert-base-uncased.pt", "flickr": "emb_flickr_caps_bert-base-uncased.pt"}


def content_tag(content):
    p = content.split(":")
    return p[0] if p[0] != "mix" else "mix" + p[1] + ("-" + p[2] if len(p) > 2 else "")


def gains_tag(gains, draws):
    """one separated field per parameter, draw count always: equal, rayleigh-D20, slot-rho0.7-D20."""
    p = gains.split(":")
    if p[0] == "rayleigh":
        return f"rayleigh-D{int(p[1])}"
    if p[0] == "slot":
        return f"slot-rho{float(p[1]):g}-D{int(p[2]) if len(p) > 2 else draws}"
    return p[0]


def fit_content(content, U):
    """-> lam (D_B,) descending, kappa (D_B,), info dict."""
    if content == "indep":
        C = torch.load(os.path.join(DATA, "emb_ag_news_bert-base-uncased.pt"), map_location="cpu", weights_only=False)
        Etr = C["train"].double().numpy()
        lam = np.maximum(np.linalg.eigvalsh(Etr.T @ Etr / Etr.shape[0])[::-1], 1e-12).copy()   # as the original
        return lam, np.zeros_like(lam), {"source": "ag_news", "kappa": "0"}
    p = content.split(":")
    ds = p[2] if (p[0] == "mix" and len(p) > 2) else ("coco" if p[0] == "mix" else p[0])
    X = torch.load(os.path.join(DATA, CAPS[ds]), map_location="cpu", weights_only=False)["train"].double().numpy()
    X = X.reshape(-1, 5, X.shape[1])                                         # (images, 5 captions, d_b)
    info = {"source": ds}
    if p[0] == "mix":
        kap = float(p[1])
        rng = np.random.default_rng(0)
        n = X.shape[0]
        sh = X[np.arange(n), rng.integers(0, 5, n)]                          # shared caption of the image
        oth = np.stack([o + (o >= m) for m, o in ((m, rng.choice(n - 1, U, replace=False)) for m in range(n))])
        pv = X[oth, rng.integers(0, 5, (n, U))]                              # captions of U other images
        T = math.sqrt(kap) * sh[:, None, :] + math.sqrt(1.0 - kap) * pv
        T = T / np.linalg.norm(T, axis=2, keepdims=True)
        info.update({"mix_kappa": kap, "tuple": U, "seed": 0})
    else:
        T = X
    m = T.shape[1]
    Y = T.reshape(-1, T.shape[2])
    lam, V = np.linalg.eigh(Y.T @ Y / Y.shape[0])
    o = np.argsort(lam)[::-1]
    lam, V = np.maximum(lam[o], 1e-12), V[:, o]
    c = T @ V
    Cd = ((c.sum(1) ** 2).mean(0) - m * lam) / (m * (m - 1))                  # diag of C
    kappa = np.clip(Cd / lam, 0.0, 0.99)
    info.update({"n_fit_vectors": int(Y.shape[0]), "lam_sum": float(lam.sum()),
                 "kappa_bar": float((lam * kappa).sum() / lam.sum()), "kappa_bar_unclipped": float(Cd.sum() / lam.sum()),
                 "kappa_group_mean": kappa.reshape(NB, SEG).mean(1).tolist(),
                 "kappa_group_lamw": ((lam * kappa).reshape(NB, SEG).sum(1) / lam.reshape(NB, SEG).sum(1)).tolist(),
                 "kappa_lead10": kappa[:10].tolist()})
    return lam, kappa, info


def prior_inv(lam_g, kap_g, U):
    """(SEG, NB*U, NB*U) inverse of the (A1) prior; diag(1/lam) when kappa = 0."""
    if not (kap_g > 0).any():
        return torch.diag_embed(1.0 / lam_g)
    Gi = torch.zeros(SEG, NB * U, NB * U, dtype=torch.float64)
    I, J = torch.eye(U, dtype=torch.float64), torch.ones(U, U, dtype=torch.float64) / U
    for b in range(NB):
        lb, kb = lam_g[:, b], kap_g[:, b]
        a0, a1 = 1 + (U - 1) * kb, 1 - kb
        blk = (I[None] / a1[:, None, None] + (1 / a0 - 1 / a1)[:, None, None] * J[None]) / lb[:, None, None]
        cols = torch.arange(U) * NB + b
        Gi[:, cols[:, None], cols[None, :]] = blk
    return Gi


class DiagInv(torch.autograd.Function):
    """diag(M^{-1}) of a batch of SPD matrices (Cholesky). Backward: the exact gradient -C diag(g) C with
    C = M^{-1} (two batched products; about half the time of autograd through the factorization, gradients
    equal to 4e-15 relative on a test batch)."""
    @staticmethod
    def forward(ctx, M):
        C = torch.cholesky_inverse(torch.linalg.cholesky(M))
        ctx.save_for_backward(C)
        return torch.diagonal(C, dim1=-2, dim2=-1).clone()

    @staticmethod
    def backward(ctx, g):
        C, = ctx.saved_tensors
        return -(C * g[..., None, :]) @ C


def mse_batch(A, Gf, Ginv, s2, U):
    """A (Bt, SEG, S_, NB*U), Gf (Bt, S_, NB*U) gains per slot and column, Ginv (SEG, n, n) shared or
    (Bt, SEG, n, n) per problem -> per-user MSE (Bt, U)."""
    Bm = A * Gf[:, None]
    M = (Ginv if Ginv.dim() == 4 else Ginv[None]) + Bm.transpose(-1, -2) @ Bm / s2
    return DiagInv.apply(M).reshape(A.shape[0], SEG, U, NB).sum((1, 3))


def energy_batch(A, lam_g, U):
    return (A ** 2 * lam_g[None, :, None, :]).sum(2).reshape(A.shape[0], SEG, U, NB).sum((1, 3))


def normalize_batch(W, lam_g, E, U):
    sc = torch.sqrt(E / (energy_batch(W, lam_g, U) + 1e-300))
    return W * sc.repeat_interleave(NB, dim=1)[:, None, None, :]


def overlap_index(A, Gf, lam_g, U):
    """energy-weighted mean over coefficients of the mean normalized off-diagonal of the U x U received Gram
    block of the coefficient (t_i of Proposition 2) -> (Bt,)."""
    if U < 2:
        return torch.zeros(A.shape[0], dtype=torch.float64)
    Bm = A * Gf[:, None]
    num, den = torch.zeros(A.shape[0], dtype=torch.float64), torch.zeros(A.shape[0], dtype=torch.float64)
    off = 1 - torch.eye(U, dtype=torch.float64)
    for b in range(NB):
        cols = torch.arange(U) * NB + b
        X = Bm[..., cols]                                                   # (Bt, SEG, S_, U)
        G = X.transpose(-1, -2) @ X                                         # (Bt, SEG, U, U)
        d = torch.diagonal(G, dim1=-2, dim2=-1)
        sd = torch.sqrt(d.clamp_min(0.0))
        t = (G / (sd[..., :, None] * sd[..., None, :]).clamp_min(1e-300) * off).sum((-1, -2)) / (U * (U - 1))
        w = (d * lam_g[None, :, cols]).sum(-1)                              # received energy of the coefficient
        num += (w * t).sum(1); den += w.sum(1)
    return num / den.clamp_min(1e-300)


def shared_energy(A, Gf, lam_g, U, S_):
    """share of each user's received energy on slots where another user has >= 1 percent -> (Bt, U)."""
    e = ((A * Gf[:, None]) ** 2 * lam_g[None, :, None, :]).sum(1).reshape(A.shape[0], S_, U, NB).sum(3)
    sh = e / e.sum(2, keepdim=True).clamp_min(1e-300)
    shared = ((sh > 0.01).sum(2, keepdim=True) > 1).double()
    return (e * shared).sum(1) / e.sum(1).clamp_min(1e-300)


def outside_share(A, A_ref, lam_g, U, S_):
    """share of each user's transmit energy on slots where A_ref gives that user no energy -> (Bt, U). With
    shared_energy it separates a different slot assignment (outside > 0, shared ~ 0) from overlap (shared > 0)."""
    def tx(X):
        return (X ** 2 * lam_g[None, :, None, :]).sum(1).reshape(X.shape[0], S_, U, NB).sum(3)   # (Bt, S_, U)
    e = tx(A)
    return (e * (tx(A_ref) <= 0)).sum(1) / e.sum(1).clamp_min(1e-300)


def mode_wf(lam, a, w, E, s2, iters=100):
    """mode water-filling p = max(0, sqrt(lam s2/nu) - s2/a), sum w p = E (bisection on log nu)."""
    lo, hi = -120.0, 60.0
    for _ in range(iters):
        mid = (lo + hi) / 2
        p = np.maximum(0.0, np.sqrt(lam * s2 / math.exp(mid)) - s2 / a)
        if (w * p).sum() > E:
            lo = mid
        else:
            hi = mid
    return np.maximum(0.0, np.sqrt(lam * s2 / math.exp(hi)) - s2 / a)


def load_copies(U, S_):
    """copies[u][b] = slots of group b of user u in the load masks."""
    out = []
    for u in range(U):
        slots = [s for s in range(S_) if s % U == u]
        out.append([[s for t, s in enumerate(slots) if t % NB == b] for b in range(NB)])
    return out


def prop2(lam, kappa, s2, U, S_):
    """Proposition 2 masks (U <= K) -> A (SEG, S_, NB*U), info."""
    E = S_ * SEG                                                            # U users x d_s/U
    a0, a1 = 1 + (U - 1) * kappa, 1 - kappa
    p = mode_wf(np.concatenate([lam, lam]), np.concatenate([a0, a1]),
                np.concatenate([np.ones_like(lam), np.full_like(lam, U - 1.0)]), E, s2)
    p0, p1 = p[:D_B], p[D_B:]
    cp = load_copies(U, S_)
    A = torch.zeros(SEG, S_, NB * U, dtype=torch.float64)
    I, J = np.eye(U), np.ones((U, U)) / U
    for b in range(NB):
        m = min(len(cp[u][b]) for u in range(U))
        assert m >= 1, "Proposition 2 needs U <= K"
        i = b * SEG + np.arange(SEG)
        W = (np.sqrt(p1[i])[:, None, None] * I + (np.sqrt(p0[i]) - np.sqrt(p1[i]))[:, None, None] * J) \
            / np.sqrt(lam[i])[:, None, None] / math.sqrt(m)                  # (SEG, U uses, U users)
        cols = [u * NB + b for u in range(U)]
        for r in range(m):
            for v in range(U):                                              # use v = copy r of user v's slot
                A[:, cp[v][b][r], cols] = torch.tensor(W[:, v, :])
    modal = float((lam * s2 / (p0 + s2 / a0) + (U - 1) * lam * s2 / (p1 + s2 / a1)).sum())
    info = {"modal_sum_mse": modal, "t_bar": float((p0 - p1).sum() / (p0 + (U - 1) * p1).sum()),
            "active_common": int((p0 > 0).sum()), "active_diff_coefs": int((p1 > 0).sum()),
            "mode_energy_sum": float(p0.sum() + (U - 1) * p1.sum())}
    return A, info


def complement_basis(U):
    """fixed orthonormal basis (U, U-1) of the complement of 1: Hadamard for U = 2^n, real DFT otherwise."""
    if U & (U - 1) == 0:
        H = np.ones((1, 1))
        while H.shape[0] < U:
            H = np.block([[H, H], [H, -H]])
        return H[:, 1:] / math.sqrt(U)
    u = np.arange(U)
    cols = []
    for k in range(1, (U - 1) // 2 + 1):
        cols += [math.sqrt(2.0 / U) * np.cos(2 * math.pi * k * u / U), math.sqrt(2.0 / U) * np.sin(2 * math.pi * k * u / U)]
    if U % 2 == 0:
        cols.append((-1.0) ** u / math.sqrt(U))
    return np.stack(cols, 1)


def prune_modes(lam, kappa, s2, U, S_, restricted):
    """greedy pruning of the Proposition-2 modes: water-fill all U d_b modes (sum p = d_s); while the active
    modes exceed the budget (restricted: S_ per use index; else d_s in total) drop the active mode with the
    smallest MSE reduction and re-solve nu. -> p (U, D_B) mode energies (0 for dropped), drops."""
    E = S_ * SEG
    coef = np.tile(np.arange(D_B), U)
    mode = np.repeat(np.arange(U), D_B)
    a = np.where(mode == 0, 1 + (U - 1) * kappa[coef], 1 - kappa[coef])
    lm = lam[coef]
    alive = np.ones(U * D_B, bool)
    drops = 0
    while True:
        p = np.zeros(U * D_B)
        p[alive] = mode_wf(lm[alive], a[alive], 1.0, E, s2)
        act = p > 0
        if restricted:
            over = np.bincount(coef[act] % SEG, minlength=SEG) > S_
            if not over.any():
                break
            cand = act & over[coef % SEG]
        else:
            if act.sum() <= E:
                break
            cand = act
        red = lm * a - lm * s2 / (p + s2 / a)
        alive[int(np.argmin(np.where(cand, red, np.inf)))] = False
        drops += 1
    modal = float(np.where(act, lm * s2 / (p + s2 / a), lm * a).sum())
    return p.reshape(U, D_B), a.reshape(U, D_B), modal, drops


def identity_split():
    """the original split: use-index problem jj holds coefficient b SEG + jj of group b -> (SEG, NB)."""
    return np.arange(D_B).reshape(NB, SEG).T.copy()


def lpt_split(m, S_):
    """relabel the coefficients over the SEG use-index problems (NB coefficients each): coefficients by kept modes
    m_i descending (stable), each to the open problem (< NB coefficients) with the lowest load (lowest index on
    ties) -> perm (SEG, NB) coefficient per (problem, position), coefficients ascending in a problem, or None if
    a problem would need more than S_ uses; loads (SEG,)."""
    load, cnt = np.zeros(SEG, int), np.zeros(SEG, int)
    members = [[] for _ in range(SEG)]
    for i in np.argsort(-m, kind="stable"):
        op = np.nonzero(cnt < NB)[0]
        j = int(op[np.argmin(load[op])])
        load[j] += int(m[i]); cnt[j] += 1; members[j].append(int(i))
    if load.max() > S_:
        return None, load
    return np.array([sorted(x) for x in members]), load


def pruned_prop2(lam, kappa, s2, U, S_):
    """pruned Proposition 2 (U > K): water-filling over all U d_b modes, pruned until at most d_s modes
    are active (free placement). The kept modes of coefficient i need m_i uses in its use-index problem; with the
    original split some problem may need more than S_ uses, then the coefficients are relabeled over the problems
    (lpt_split; under (A1) and block fading the coefficients are independent, so any split into SEG problems of NB
    coefficients is an equally valid restricted model with the same objective and per-user energies). Only if that
    fails the per-use-index pruning (at most S_ modes per problem of the original split) is used.
    -> A (SEG, S_, NB*U) in the split perm, info, perm (SEG, NB)."""
    t0 = time.time()
    p, a, modal_g, drops_g = prune_modes(lam, kappa, s2, U, S_, False)
    _, _, modal_r, drops_r = prune_modes(lam, kappa, s2, U, S_, True)       # reference: fixed-split pruning
    m = (p > 0).sum(0)
    ident = identity_split()
    load_id = m[ident].sum(1)
    if load_id.max() <= S_:
        perm, split = ident, "identity"
    else:
        perm, split = lpt_split(m, S_)[0], "lpt"
        if perm is None:                                                    # fallback: restricted pruning
            p, a, _, _ = prune_modes(lam, kappa, s2, U, S_, True)
            perm, split = ident, "restricted_fallback"
    Qc = complement_basis(U)
    A = torch.zeros(SEG, S_, NB * U, dtype=torch.float64)
    nxt = np.zeros(SEG, int)
    q0 = np.full(U, 1 / math.sqrt(U))
    for jj in range(SEG):
        for b in range(NB):
            i = int(perm[jj, b])                                            # masks per coefficient do not depend on the split
            rows = []
            if p[0, i] > 0:
                rows.append(math.sqrt(p[0, i] / lam[i]) * q0)
            act = np.nonzero(p[1:, i] > 0)[0]
            for r in range(len(act)):
                col = (i + r) % (U - 1)
                q = Qc[(np.arange(U) + i) % U, col]                         # user index shifted by i
                rows.append(math.sqrt(p[1 + act[r], i] / lam[i]) * q)
            for row in rows:
                A[jj, nxt[jj], [u * NB + b for u in range(U)]] = torch.tensor(row)
                nxt[jj] += 1
    assert nxt.max() <= S_
    modal = modal_r if split == "restricted_fallback" else modal_g
    info = {"split": split, "modal_sum_mse": modal, "modal_sum_mse_global": modal_g,
            "modal_sum_mse_restricted": modal_r, "drops_global": drops_g, "drops_restricted": drops_r,
            "modes_kept": int((p > 0).sum()), "common_kept": int((p[0] > 0).sum()),
            "modes_per_index_max_identity_split": int(load_id.max()), "modes_per_index_max": int(nxt.max()),
            "modes_per_index_min": int(nxt.min()), "prune_time_s": time.time() - t0}
    return A, info, perm


def csiq_masks(G, lam, s2, U):
    """channel-aware masks (13) on the greedy support (csiq of pca_csi.py) per draw -> A (D, SEG, S, NB*U),
    own (D, 1, S, NB*U) support, counts (D, U)."""
    from pca_csi import greedy_counts, place_sorted
    D = G.shape[0]
    sets = greedy_counts(G, lam, s2)
    A = torch.zeros(D, SEG, S, NB * U, dtype=torch.float64)
    own = torch.zeros(D, 1, S, NB * U, dtype=torch.float64)
    cnt = np.zeros((D, U), int)
    for j in range(U):
        sl = [sets[f][j] for f in range(D)]
        n = np.array([len(x) for x in sl]); cnt[:, j] = n
        sob, P = place_sorted(G[:, j], sl, n * SEG, lam, s2)
        for f in range(D):
            own[f, 0, sl[f], j * NB:(j + 1) * NB] = 1.0
            for b in range(NB):
                if sob[f, b] >= 0:
                    i = slice(b * SEG, (b + 1) * SEG)
                    A[f, :, sob[f, b], j * NB + b] = torch.tensor(np.sqrt(P[f, i] / lam[i]))
    return A, own, cnt


def draw_gains(gains, U, S_, snr, nearfar, draws):
    """-> G (D, U, S_) gain magnitudes, design gain per user (U,) used by load (None: unit)."""
    p = gains.split(":")
    if p[0] == "equal":
        gdb = np.linspace(nearfar, -nearfar, U) if nearfar > 0 else np.zeros(U)
        g = 10 ** (gdb / 20.0)
        return np.repeat(g[None, :, None], S_, axis=2), g
    rng = np.random.default_rng(123 + snr)
    if p[0] == "rayleigh":
        D = int(p[1])
        h = np.sqrt((rng.standard_normal((D, U)) ** 2 + rng.standard_normal((D, U)) ** 2) / 2.0)
        off = 10 ** ((2 * rng.random((D, U)) - 1) * nearfar / 20.0) if nearfar > 0 else 1.0
        return np.repeat((h * off)[:, :, None], S_, axis=2), None
    if p[0] == "slot":
        from pca_csi import channel
        assert S_ == S, "slot gains need K = 4 (24 slots)"
        D = int(p[2]) if len(p) > 2 else draws
        return channel(rng, D, U, float(p[1]), nearfar, 1.0)[0], None
    raise ValueError(gains)


def search_batch(A_cl, A_dis0, own, Gf, Ginv, lam_g, s2, U, E, steps, restarts, chunk=4, Ginv_dis=None, lam_dis=None):
    """non-overlapping member (load-mask support, from A_dis0), from closed, `restarts` random starts; batches of independent
    problems (`chunk` draws at a time: smaller batches stay in cache, about 1.75x faster per problem at U = 4;
    all random starts are drawn first, so the result does not depend on `chunk`). Adam runs in energy units
    W = A sqrt(lam) (Adam moves every entry by about lr per step, so in amplitude units the weak coefficients,
    whose amplitudes are up to ~100x larger, barely move); the starts are the same points as before (closed and
    non-overlapping member + the same small amplitude noise), random starts N(0, 1) in W. The non-overlapping member
    may use a split of its own (Ginv_dis, lam_dis; default the split of closed)
    -> best A (B, ...), best losses (B,), B = (2 + restarts) D, index row * D + draw (row 0 non-overlapping,
    1 from closed, 2 + r random start r)."""
    D = A_cl.shape[0]
    R = restarts
    nd = 1e-3 * torch.randn_like(A_dis0)
    nc = 1e-2 * torch.randn_like(A_cl)
    nr = torch.randn((R, D) + A_cl.shape[1:], dtype=torch.float64)
    sq_c = torch.sqrt(lam_g)[None, :, None, :]                              # (1, SEG, 1, NB*U)
    if lam_dis is None:
        sq_d, two = sq_c, False
    else:
        sq_d, two = torch.sqrt(lam_dis)[None, :, None, :], True
    Bt = (2 + R) * D
    best = torch.full((Bt,), float("inf"), dtype=torch.float64)
    bestA = torch.zeros((Bt,) + A_cl.shape[1:], dtype=torch.float64)
    for c0 in range(0, D, chunk):
        sl = slice(c0, min(D, c0 + chunk))
        dc = sl.stop - c0
        own_c = own[sl]
        Wd = ((A_dis0[sl] + nd[sl] * own_c) * sq_d).requires_grad_(True)
        Wc = ((A_cl[sl] + nc[sl]) * sq_c).requires_grad_(True)
        Wr = nr[:, sl].reshape((R * dc,) + A_cl.shape[1:]).clone().requires_grad_(True)
        opt = torch.optim.Adam([{"params": [Wd], "lr": 0.01}, {"params": [Wc], "lr": 0.01}, {"params": [Wr], "lr": 0.02}])
        Gall = torch.cat([Gf[sl], Gf[sl], Gf[sl].repeat(R, 1, 1)])
        Eall = torch.cat([E[sl], E[sl], E[sl].repeat(R, 1)])
        nrow = (2 + R) * dc
        if two:                                                             # per-row split: non-overlapping rows first
            SQ = torch.cat([sq_d.expand(dc, -1, -1, -1), sq_c.expand((1 + R) * dc, -1, -1, -1)])
            GI = torch.cat([Ginv_dis[None].expand(dc, -1, -1, -1), Ginv[None].expand((1 + R) * dc, -1, -1, -1)])
        else:
            SQ, GI = sq_c, Ginv
        gidx = (torch.arange(2 + R)[:, None] * D + c0 + torch.arange(dc)[None, :]).reshape(-1)   # local -> global
        bl = torch.full((gidx.numel(),), float("inf"), dtype=torch.float64)
        bA = torch.zeros((gidx.numel(),) + A_cl.shape[1:], dtype=torch.float64)
        for it in range(steps):
            Wall = torch.cat([Wd * own_c, Wc, Wr])
            e = (Wall ** 2).sum(2).reshape(nrow, SEG, U, NB).sum((1, 3))   # per-user energy
            sc = torch.sqrt(Eall / (e + 1e-300))
            A = Wall * sc.repeat_interleave(NB, dim=1)[:, None, None, :] / SQ
            loss = mse_batch(A, Gall, GI, s2, U).sum(1)
            opt.zero_grad(); loss.sum().backward(); opt.step()
            with torch.no_grad():
                imp = loss.detach() < bl
                bl = torch.where(imp, loss.detach(), bl)
                bA[imp] = A.detach()[imp]
        best[gidx], bestA[gidx] = bl, bA
    return bestA, best


def run2(a):
    t_all = time.time()
    torch.set_num_threads(2)
    content = a.content or "indep"
    gains = a.gains or "equal"
    K, U = a.K or 4, a.U
    S_ = NB * K
    shared = content != "indep"
    lam, kappa, cinfo = fit_content(content, U)
    tot = float(lam.sum())
    lam_g = torch.tensor(lam.reshape(NB, SEG).T.copy()).repeat(1, U)
    kap_g = torch.tensor(kappa.reshape(NB, SEG).T.copy()).repeat(1, U)
    Ginv = prior_inv(lam_g, kap_g, U)
    case = f"{content_tag(content)}_{gains_tag(gains, a.draws)}_K{K}_U{U}" + (f"_nf{a.nearfar:g}" if a.nearfar else "") \
        + (f"_s{a.steps}" if a.steps != 3000 else "") + (f"_r{a.restarts}" if a.restarts != 3 else "") \
        + ("_Eeq" if a.search_energy != "closed" else "") + a.tag
    kind = gains.split(":")[0]
    res = {}
    print(f"[{case}] lam sum {tot:.4f} kappa_bar {cinfo.get('kappa_bar', 0.0):.4f} S={S_} d_s={S_ * SEG}", flush=True)
    for snr in a.snrs:
        t0 = time.time()
        s2 = 10 ** (-snr / 10.0)
        torch.manual_seed(1000 + snr)
        G, gdes = draw_gains(gains, U, S_, snr, a.nearfar, a.draws)
        D = G.shape[0]
        Gf = torch.tensor(G).permute(0, 2, 1).repeat_interleave(NB, dim=2).contiguous()   # (D, S_, NB*U)
        gd = gdes if (gdes is not None and not shared) else np.ones(U)       # shared content: unit design gain as Prop. 2
        A_load = closed_form(lam, gd, s2, U, S_)[0][None].repeat(D, 1, 1, 1)
        own = torch.zeros(1, 1, S_, NB * U, dtype=torch.float64)
        for j in range(U):
            own[0, 0, [s for s in range(S_) if s % U == j], j * NB:(j + 1) * NB] = 1.0
        own = own.repeat(D, 1, 1, 1)
        out, extra = {}, {}
        lam_c, Ginv_c, two = lam_g, Ginv, False                             # split of closed / search (default: original)
        if kind == "slot":
            A_cl, own, cnt = csiq_masks(G, lam, s2, U)
            A_dis0, closed_name = A_cl, "csiq13"
            extra["csiq_counts_mean"] = cnt.mean(0).tolist()
        elif shared and U <= K:
            A1, info = prop2(lam, kappa, s2, U, S_)
            A_cl, A_dis0, closed_name = A1[None].repeat(D, 1, 1, 1), A_load, "prop2"
            with torch.no_grad():
                one = torch.ones(1, S_, NB * U, dtype=torch.float64)
                inside = float(mse_batch(A1[None], one, Ginv, s2, U).sum())
                eu = energy_batch(A1[None], lam_g, U)[0].numpy()
            info.update({"inside_sum_mse": inside, "modal_rel_diff": abs(inside - info["modal_sum_mse"]) / info["modal_sum_mse"],
                         "user_energy": eu.tolist(), "user_energy_dev": float(np.abs(eu / (S_ * SEG / U) - 1).max()),
                         "modal_sum_nmse": info["modal_sum_mse"] / tot, "inside_sum_nmse": inside / tot})
            extra["prop2"] = info
        elif shared:
            A1, info, perm = pruned_prop2(lam, kappa, s2, U, S_)
            if info["split"] == "lpt":                                      # closed and search in the relabeled split
                lam_c = torch.tensor(lam[perm]).repeat(1, U)
                Ginv_c = prior_inv(lam_c, torch.tensor(kappa[perm]).repeat(1, U), U)
                two = True
                info["perm"] = perm.tolist()
            with torch.no_grad():
                one = torch.ones(1, S_, NB * U, dtype=torch.float64)
                eu = energy_batch(A1[None], lam_c, U)[0].numpy()
                unscaled = float(mse_batch(A1[None], one, Ginv_c, s2, U).sum())
                A1 = normalize_batch(A1[None], lam_c, torch.full((1, U), S_ * SEG / U, dtype=torch.float64), U)[0]
                scaled = float(mse_batch(A1[None], one, Ginv_c, s2, U).sum())
            info.update({"user_energy_prescale": eu.tolist(), "user_energy_dev_prescale": float(np.abs(eu / (S_ * SEG / U) - 1).max()),
                         "inside_sum_mse_prescale": unscaled,
                         "modal_rel_diff_prescale": abs(unscaled - info["modal_sum_mse"]) / info["modal_sum_mse"],
                         "inside_sum_nmse": scaled / tot, "modal_sum_nmse": info["modal_sum_mse"] / tot,
                         "modal_sum_nmse_restricted": info["modal_sum_mse_restricted"] / tot,
                         "modal_sum_nmse_global": info["modal_sum_mse_global"] / tot})
            A_cl, A_dis0, closed_name = A1[None].repeat(D, 1, 1, 1), A_load, "prop2_pruned"
            extra["prune"] = info
        else:
            A_cl, A_dis0, closed_name = A_load, A_load, "load12"
        with torch.no_grad():
            E_cl = energy_batch(A_cl, lam_c, U)
            E = E_cl if a.search_energy == "closed" else torch.full((D, U), S_ * SEG / U, dtype=torch.float64)
            m_load = mse_batch(A_load, Gf, Ginv, s2, U) / tot
            m_cl = mse_batch(A_cl, Gf, Ginv_c, s2, U) / tot
        bestA, best = search_batch(A_cl, A_dis0, own, Gf, Ginv_c, lam_c, s2, U, E, a.steps, a.restarts, a.chunk,
                                   Ginv if two else None, lam_g if two else None)
        R = a.restarts
        with torch.no_grad():
            m_all = torch.cat([mse_batch(bestA[:D], Gf, Ginv, s2, U),                       # non-overlapping member: original split
                               mse_batch(bestA[D:], Gf.repeat(1 + R, 1, 1), Ginv_c, s2, U)]) / tot   # (B, U)
            sums = m_all.sum(1).reshape(2 + R, D)                                # rows: non-overlapping, closed start, random
            k = sums[1:].argmin(0)                                               # best overlap start per draw
            idx = (1 + k) * D + torch.arange(D)
            m_srch, A_srch = m_all[idx], bestA[idx]
            t_cl, t_srch = overlap_index(A_cl, Gf, lam_c, U), overlap_index(A_srch, Gf, lam_c, U)
            sh_srch = shared_energy(A_srch, Gf, lam_c, U, S_)
            out_srch = outside_share(A_srch, A_cl, lam_c, U, S_)
        cs, ss = m_cl.sum(1).numpy(), m_srch.sum(1).numpy()
        gap = 100 * (cs - ss) / cs
        out = {"closed_name": closed_name, "draws": D,
               "load_sum": m_load.sum(1).tolist(), "closed_sum": cs.tolist(), "disjoint_sum": sums[0].tolist(),
               "search_sum": ss.tolist(), "search_all_sums": sums[1:].T.tolist(),
               "closed_users": m_cl.tolist(), "search_users": m_srch.tolist(), "load_users": m_load.tolist(),
               "gap_pct": gap.tolist(), "gap_pct_mean": float(gap.mean()), "gap_pct_max": float(gap.max()),
               "gap_pct_of_means": float(100 * (cs.mean() - ss.mean()) / cs.mean()),
               "load_minus_closed_pct_mean": float((100 * (m_load.sum(1).numpy() - cs) / cs).mean()),
               "t_closed": t_cl.tolist(), "t_search": t_srch.tolist(),
               "search_shared_energy": sh_srch.tolist(), "search_tx_outside_closed": out_srch.tolist(),
               "energy_closed": E_cl.tolist(), "search_energy": a.search_energy}
        for kk in ("load", "closed", "disjoint", "search"):
            out[kk + "_mean_sum"] = float(np.mean(out[kk + "_sum"]))
        if kind != "equal":
            out["gains"] = G[:, :, 0].tolist() if kind == "rayleigh" else G.tolist()
        out.update(extra)
        out["runtime_s"] = time.time() - t0
        res[str(snr)] = out
        chk = ""
        if "prop2" in extra:
            chk = f" modal {extra['prop2']['modal_sum_nmse']:.6f} inside {extra['prop2']['inside_sum_nmse']:.6f} " \
                  f"rel {extra['prop2']['modal_rel_diff']:.1e} Edev {extra['prop2']['user_energy_dev']:.1e} t {extra['prop2']['t_bar']:.3f}"
        if "prune" in extra:
            pi = extra["prune"]
            chk = f" pruned split {pi['split']} modal {pi['modal_sum_nmse']:.5f} inside {pi['inside_sum_nmse']:.5f} " \
                  f"rel {pi['modal_rel_diff_prescale']:.1e} (fixed-split pruning {pi['modal_sum_nmse_restricted']:.5f}) " \
                  f"Edev {pi['user_energy_dev_prescale']:.1e} drops {pi['drops_global']}/{pi['drops_restricted']} " \
                  f"modes/index {pi['modes_per_index_max_identity_split']}->{pi['modes_per_index_max']}"
        if "csiq_counts_mean" in extra:
            chk = f" counts {np.round(extra['csiq_counts_mean'], 2).tolist()}"
        print(f"[{case}] snr={snr} D={D} sumNMSE load {out['load_mean_sum']:.5f} {closed_name} {out['closed_mean_sum']:.5f} "
              f"disjoint {out['disjoint_mean_sum']:.5f} search {out['search_mean_sum']:.5f} gap {out['gap_pct_mean']:+.2e}% "
              f"(max {out['gap_pct_max']:+.2e}%) t {float(t_cl.mean()):.3f}/{float(t_srch.mean()):.3f} "
              f"search energy outside closed slots {float(out_srch.mean()):.3f} shared {float(sh_srch.mean()):.3f}{chk} "
              f"{out['runtime_s']:.1f}s", flush=True)
    os.makedirs(os.path.join(DATA, "extra"), exist_ok=True)
    fn = os.path.join(DATA, "extra", f"overlap_bound2_{case}.json")
    json.dump({"args": vars(a), "case": case, "K": K, "U": U, "S": S_, "d_s": S_ * SEG, "content": cinfo,
               "res": res, "runtime_s": time.time() - t_all}, open(fn, "w"), indent=1)
    print(f"[{case}] wrote {fn} total {time.time() - t_all:.1f}s", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--U", type=int, required=True)
    ap.add_argument("--nearfar", type=float, default=0.0)
    ap.add_argument("--snrs", type=int, nargs="+", default=[0, 10, 20])
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--restarts", type=int, default=3)
    # extension (run2); none given -> the original search below, unchanged
    ap.add_argument("--content", default=None, help="indep | coco | flickr | mix:KAPPA[:ds]")
    ap.add_argument("--gains", default=None, help="equal | rayleigh:DRAWS | slot:RHO_F[:DRAWS]")
    ap.add_argument("--K", type=int, default=None, choices=[2, 4])
    ap.add_argument("--draws", type=int, default=20, help="draws of slot gains without :DRAWS")
    ap.add_argument("--search_energy", default="closed", choices=["closed", "equal"])
    ap.add_argument("--chunk", type=int, default=4, help="draws optimized together (speed only, same result)")
    ap.add_argument("--tag", default="", help="appended to the run2 output name (e.g. _smoke)")
    a = ap.parse_args()
    if a.content is not None or a.gains is not None or a.K is not None:
        return run2(a)
    old_args = {k: getattr(a, k) for k in ("U", "nearfar", "snrs", "steps", "restarts")}
    torch.set_num_threads(2)
    C = torch.load(os.path.join(DATA, "emb_ag_news_bert-base-uncased.pt"), map_location="cpu", weights_only=False)
    Etr = C["train"].double().numpy()
    lam = np.maximum(np.linalg.eigvalsh(Etr.T @ Etr / Etr.shape[0])[::-1], 1e-12).copy()
    U = a.U
    gdb = np.linspace(a.nearfar, -a.nearfar, U) if a.nearfar > 0 else np.zeros(U)
    g = torch.tensor(10 ** (gdb / 20.0))
    E = torch.full((U,), S * SEG / U, dtype=torch.float64)
    tot = float(lam.sum())
    res = {"gain_db": gdb.tolist()}
    for snr in a.snrs:
        s2 = 10 ** (-snr / 10.0)
        torch.manual_seed(snr)
        A0, lam_g = closed_form(lam, g.numpy(), s2, U)
        own = torch.zeros(1, S, NB * U, dtype=torch.float64)
        for j in range(U):
            own[0, [s for s in range(S) if s % U == j], j * NB:(j + 1) * NB] = 1.0
        full = torch.ones_like(own)
        out = {}
        with torch.no_grad():
            out["closed"] = (mse_users(A0, g, lam_g, s2, U) / tot).tolist()
        Ad = optimize(A0 + 1e-3 * torch.randn_like(A0) * own, own, g, lam_g, s2, U, E, a.steps, 0.01)
        cands = [optimize(A0 + 1e-2 * torch.randn_like(A0), full, g, lam_g, s2, U, E, a.steps, 0.01)]
        for r in range(a.restarts):
            cands.append(optimize(torch.randn_like(A0) * A0.abs().max(), full, g, lam_g, s2, U, E, a.steps, 0.02))
        with torch.no_grad():
            out["disjoint"] = (mse_users(Ad, g, lam_g, s2, U) / tot).tolist()
            ov = [mse_users(c, g, lam_g, s2, U) / tot for c in cands]
            k = int(np.argmin([o.sum().item() for o in ov]))
            out["overlap"] = ov[k].tolist()
            out["overlap_all_sums"] = [o.sum().item() for o in ov]
            Ak = cands[k] * g.repeat_interleave(NB)[None, None, :]
            # how much the users overlap: share of the received energy of user u on slots where another user
            # carries at least 1 percent of that slot's received energy
            e = (Ak ** 2 * lam_g[:, None, :]).sum(0).reshape(S, U, NB).sum(2)     # (S,U) received energy
            sh = e / e.sum(1, keepdim=True)
            shared = ((sh > 0.01).sum(1, keepdim=True) > 1).double()
            out["overlap_shared_energy"] = ((e * shared).sum(0) / e.sum(0)).tolist()
        for kk in ("closed", "disjoint", "overlap"):
            out[kk + "_mean"] = float(np.mean(out[kk])); out[kk + "_worst"] = float(np.max(out[kk]))
        res[str(snr)] = out
        print(f"U={U} nf={a.nearfar} snr={snr} NMSE mean closed {out['closed_mean']:.5f} disjoint {out['disjoint_mean']:.5f} "
              f"overlap {out['overlap_mean']:.5f} (worst {out['closed_worst']:.5f}/{out['disjoint_worst']:.5f}/"
              f"{out['overlap_worst']:.5f}) restarts {np.round(out['overlap_all_sums'], 5).tolist()} "
              f"shared {np.round(out['overlap_shared_energy'], 3).tolist()}", flush=True)
    os.makedirs(os.path.join(DATA, "extra"), exist_ok=True)
    json.dump({"args": old_args, "res": res},
              open(os.path.join(DATA, "extra", f"overlap_bound_U{U}_nf{int(a.nearfar)}.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
