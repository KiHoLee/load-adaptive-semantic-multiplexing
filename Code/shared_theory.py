# =========================================================
# shared_theory.py — analysis lines for the shared-content runs (2026-11-16): exact NMSE of the load masks and
# of the generalized real-valued masks of Proposition 2 in the Gaussian model of the caption embeddings.
# Model: the U users of one scene send coefficient vectors c_u = V^T b_u with the joint covariance
#   full   Gamma = I_U (x) (Lam - C) + 1 1^T (x) C, lam, V, C fitted on the fit images (content_stats.fit_content:
#          captions of one image, or --mix KAPPA mixtures, seed 0: --mix_fit spec (default) U users per tuple in the
#          order of overlap_bound.fit_content; --mix_fit shared_masks the fit of shared_masks.Content,
#          5 per tuple in its order; the choice is stored in meta.mix_fit)
#   a1     the diagonal surrogate (A1): C = diag(kappa_i lam_i), kappa_i = clip(C_ii / lam_i, 0, 0.99), so the U
#          users' coefficient i has covariance lam_i [(1 - kappa_i) I + kappa_i 1 1^T]
# Block of d_s = K d_b real uses (N = 6K slots of 128), per-user energy d_s/U, noise s2 = 10^(-SNR/10) per use,
# gain g_u per user known at the receiver: equal (g = 1) or 100 independent Rayleigh draws |h_u| ~ |CN(0,1)|
# (default_rng(7), the same draws at every SNR and for every scheme). The transmitter designs at the unit gain.
# A mask set is summarized by the U x U transmit Gram block Omega_i = W_i^T W_i of every coefficient i
# (W_i: uses x users); the receiver sees y_i = W_i G c_i + n_i, so M = Psi^T Psi / s2 has the blocks
# G Omega_i G / s2 and the joint LMMSE error is tr[(Gamma^{-1} + M)^{-1}] (unsent coefficients from the prior).
# Schemes:
#   load_pu    (a) load masks: every user sends its leading d_u = min(floor(d_s/U), d_b) coefficients with (12) at
#              energy d_s/U and the unit gain (--load_rule spec, default: the equal split of shared_masks.design_load);
#              --load_rule rank: rank j sends min(ceil((N - j)/U) * 128, d_b) coefficients (pca_eval.per_load,
#              overlap_bound; output suffix _rank). The rules differ only when U does not divide N = 6K (U = 5).
#              + per-user Wiener with the true gain: sum_i lam_i s2 / (g_u^2 P_i + s2) (closed form, both models)
#   load_joint (b) load masks + joint LMMSE with the model covariance
#   gen_joint  (c) Proposition 2 masks (kappa_i of the fit, unit design gain): modes common / difference,
#              p_{i,m} = max(0, sqrt(lam_i s2/nu) - s2/a_{i,m}), a_{i,0} = 1 + (U-1) kappa_i, a_{i,m>0} = 1 - kappa_i,
#              sum p = d_s; Omega_i = lam_i^{-1} (p_{i,1} I + (p_{i,0} - p_{i,1}) 1 1^T / U) at U <= K. At U > K greedy
#              pruning to at most d_s active modes (drop the active mode with the smallest MSE reduction
#              lam a - lam s2/(p + s2/a), re-solve nu), difference modes from the fixed complement basis of
#              overlap_bound.py (Hadamard for U = 2^n, real DFT otherwise) with the user index shifted by i and the
#              kept columns rotated cyclically with i, then every user scaled exactly to d_s/U; + joint LMMSE
#   gen_eq     (c') equal-value full overlap (t_i = 1 for every i): only the common modes, p_{i,0} water-filled with
#              a_{i,0} and sum p = d_s, Omega_i = lam_i^{-1} p_{i,0} 1 1^T / U, + joint LMMSE
#   gen_k0     (d) Proposition 2 with kappa = 0 (pruned at U > K), realized as W_i = sqrt(P_i/lam_i) Q_i with a random
#              orthogonal Q_i per coefficient (rows kept = active modes, default_rng(11)), users scaled to d_s/U;
#              + joint LMMSE. The evaluator sees only Omega_i = W_i^T W_i, so at U <= K Omega_i = (P_i/lam_i) I is
#              the Gram block of (b) identically: (d) = (b) there is an identity of the Gram model (it checks
#              Q^T Q = I only), not evidence for the rotation equivalence of real masks (that comes from the
#              explicit-mask runs of shared_masks.py). At U > K (d) is the pruned kappa = 0 design and differs from
#              (b) by the pruning (spec rule: a few modes; rank rule: also the unequal split): the overlap
#              (shared-content) gain is (d) -> (c), stored per SNR in diag[].decomp_rel with (b) -> (d) and (b) -> (c).
# Exact evaluation (float64): a1 per coefficient (U x U blocks, all draws batched); full: dense Cholesky of the
# information matrix Gamma^{-1} + M (dimension U d_b, Gamma^{-1} = I (x) B1 + (1/U) 1 1^T (x) (B0 - B1)) for
# gen_joint, gen_k0 and load_joint when the users' allocations differ (only --load_rule rank at U = 5); load_joint with equal allocations
# by simultaneous diagonalization of B1 and the load information plus Woodbury over the rank-d_b common part;
# gen_eq by Woodbury over its rank-d_b information. The structured paths are checked against the dense solve on
# the first draw of every SNR (stored), and --validate compares the dense solve with the covariance form
# tr Gamma - ||L^{-1} Psi Gamma||_F^2 (L L^T = Psi Gamma Psi^T + s2 I). A relative ridge 1e-10 lam_1 is added
# to Lam only if a Cholesky of Lam - C or Lam + (U-1) C fails (reported).
# Also stored: the closed forms at equal gains in (A1) (Proposition 2 modal MSE; load_joint and gen_eq modal),
# their relative difference to the a1 trace (must be < 1e-8; Proposition 2 with the unscaled masks), the
# per-user energy deviation before scaling, the energy-weighted overlap index t of Omega, the interior formula
# t_i = kappa s2 / [P_i (1 - kappa)(1 + (U-1) kappa)], and the low-SNR slope gains (meta.slope_gains): the exact
# first-mode predictions in (A1), G0 = pred_c_over_b = max_i lam_i a_{i,0}^2 / (lam_1 (1 + (U-1) kappa_1^2)) (overlap
# over load + joint receiver) and pred_c_over_a = max_i lam_i a_{i,0}^2 / lam_1 (over per-user reception), also for
# U = 2..5; the formulas (1 + (U-1) k)^2 / (1 + (U-1) k^2) and (1 + (U-1) k)^2 with k = kappa_1 equal them only when
# coefficient 1 is the first active common mode (coef1_is_first_common_mode; true for COCO, false for Flickr), so
# they are stored as *_formula_kappa1; numeric check at -70 dB (equal gains, a1 and full).
# Metrics per scheme, model, gains and SNR: NMSE = E sum_u ||c_u - c_hat_u||^2 / (U sum lam) (mean over draws), its
# spread over draws and Monte Carlo standard error nmse_se = sd / sqrt(draws) (100 draws: about 3-7% of the
# absolute NMSE), the paired differences to (b) and (a) on the same draws with their standard errors
# (minus_load_joint, minus_load_pu: much tighter than the absolute NMSE), per-user NMSE, and the proxies
# cos = sqrt(1 - NMSE_u) (mean, and mean of the per-draw minimum over users). Per-draw per-user MSE in the .npz.
# Output: ../data/shared/theory_<ds>[_mix<k>]_K<K>_U<U>[_rank].json (+ .npz).
# Usage: python shared_theory.py --ds coco --K 4 --U 4
#        python shared_theory.py --ds coco --mix 0.6 --K 2 --U 4
#        python shared_theory.py --ds coco --K 2 --U 5 --load_rule rank        (manuscript rank counts, suffix _rank)
#        python shared_theory.py --ds coco --K 4 --U 4 --snrs 0 --draws 3 --validate --outdir smoke   (smoke)
# =========================================================
import argparse, json, math, os, time
import numpy as np
from scipy.linalg import lapack, solve_triangular, cholesky, cho_factor, cho_solve
import content_stats as cs

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(os.path.dirname(HERE), "data")
SNRS = (-10, -5, 0, 5, 10, 15, 20)
SEG, NB = 128, 6
SCHEMES = ("load_pu", "load_joint", "gen_joint", "gen_eq", "gen_k0")


# ------------------------------------------------------------------ allocations and masks
def modes_wf(lam, a, E, s2, w=None):
    """exact reverse water-filling p_k = max(0, sqrt(lam_k s2 / nu) - s2 / a_k), sum_k w_k p_k = E."""
    w = np.ones_like(lam) if w is None else w
    al, be = np.sqrt(lam * s2), s2 / a
    tau = be / al
    o = np.argsort(tau, kind="stable")
    cA, cB = np.cumsum(w[o] * al[o]), np.cumsum(w[o] * be[o])
    f = np.maximum.accumulate(tau[o] * cA - cB)                               # total at the activation of mode o_j
    j = int(np.searchsorted(f, E, side="left")) - 1
    t = (E + cB[j]) / cA[j]
    return np.maximum(0.0, al * t - be)


LOAD_RULES = ("spec", "rank")


def load_counts(U, K, rule="spec"):
    """coefficients sent by each user of the load masks. spec: d_u = min(floor(d_s/U), d_b) for every user (equal split,
    shared_masks.design_load); rank: rank j sends min(ceil((N - j)/U) * 128, d_b) (pca_eval.per_load, overlap_bound).
    Equal whenever U divides N = 6K."""
    N, d = NB * K, NB * SEG
    if rule == "rank":
        return [min(math.ceil((N - j) / U) * SEG, d) for j in range(U)]
    assert rule == "spec", rule
    return [min(K * d // U, d)] * U


def load_powers(lam, U, K, s2, rule="spec"):
    d = lam.size
    P = np.zeros((U, d))
    E = K * d / U
    for j, n in enumerate(load_counts(U, K, rule)):
        P[j, :n] = modes_wf(lam[:n], np.ones(n), E, s2)
    return P


def prune(lam, kappa, U, s2, d_s):
    """Proposition-2 mode energies with greedy pruning to at most d_s active modes (mode-major arrays as in
    overlap_bound.prune_modes, ties to the lowest index) -> p (U, d), a (U, d), drops."""
    d = lam.size
    coef, mode = np.tile(np.arange(d), U), np.repeat(np.arange(U), d)
    a = np.where(mode == 0, 1 + (U - 1) * kappa[coef], 1 - kappa[coef])
    lm = lam[coef]
    alive = np.ones(U * d, bool)
    drops = 0
    while True:
        p = np.zeros(U * d)
        p[alive] = modes_wf(lm[alive], a[alive], d_s, s2)
        act = p > 0
        if act.sum() <= d_s:
            break
        red = lm * a - lm * s2 / (p + s2 / a)
        alive[int(np.argmin(np.where(act, red, np.inf)))] = False
        drops += 1
    return p.reshape(U, d), a.reshape(U, d), drops


def complement_basis(U):
    """fixed orthonormal basis (U, U-1) of the complement of 1 (as overlap_bound.complement_basis)."""
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


def omega_prop2(lam, p, U):
    """Gram blocks of the Proposition-2 masks: common mode 1/sqrt(U) with p_{i,0}; the r-th active difference mode of
    coefficient i is column (i + r) mod (U-1) of the complement basis with the user index shifted by i."""
    d = lam.size
    Om = p[0][:, None, None] * np.full((U, U), 1.0 / U)[None]
    if U > 1:
        Qc = complement_basis(U)
        k = (p[1:] > 0).sum(0)
        p1 = p[1:].max(0)
        assert np.allclose(np.where(p[1:] > 0, p[1:], p1[None]), p1[None], rtol=1e-12, atol=0)
        sh = (np.arange(U)[None, :] + np.arange(d)[:, None]) % U
        for r in range(U - 1):
            q = Qc[sh, ((np.arange(d) + r) % (U - 1))[:, None]]               # (d, U)
            Om += np.where(k > r, p1, 0.0)[:, None, None] * q[:, :, None] * q[:, None, :]
    return Om / lam[:, None, None]


def omega_k0(lam, p, U, seed=11):
    """rotation sanity: W_i = sqrt(P_i/lam_i) Q_i (rows = uses), random orthogonal Q_i, rows kept = active modes."""
    d = lam.size
    Q = np.linalg.qr(np.random.default_rng(seed).standard_normal((d, U, U)))[0]
    n = (p > 0).sum(0)
    W = np.sqrt(p.max(0) / lam)[:, None, None] * Q * (np.arange(U)[None, :, None] < n[:, None, None])
    return W.transpose(0, 2, 1) @ W


def omega_load(lam, P):
    U, d = P.shape
    Om = np.zeros((d, U, U))
    Om[:, np.arange(U), np.arange(U)] = (P / lam[None]).T
    return Om


def user_energy(lam, Om):
    return (lam[:, None] * np.diagonal(Om, axis1=1, axis2=2)).sum(0)


def scale_users(lam, Om, E):
    s = np.sqrt(E / user_energy(lam, Om))
    return Om * s[None, :, None] * s[None, None, :]


def overlap_index(lam, Om):
    """energy-weighted mean over coefficients of the mean normalized off-diagonal of Omega_i (t_i)."""
    U = Om.shape[1]
    dg = np.diagonal(Om, axis1=1, axis2=2)
    sd = np.sqrt(np.maximum(dg, 0.0))
    nz = (sd[:, :, None] * sd[:, None, :]) > 0
    R = np.where(nz, Om / np.where(nz, sd[:, :, None] * sd[:, None, :], 1.0), 0.0)
    t = (R.sum((1, 2)) - np.trace(R, axis1=1, axis2=2)) / (U * (U - 1))
    w = lam * dg.sum(1)
    return float((w * t).sum() / w.sum())


def make_masks(lam, kappa, U, K, s2, rule="spec"):
    d = lam.size
    d_s, E = K * d, K * d / U
    P = load_powers(lam, U, K, s2, rule)
    p, a, drops = prune(lam, kappa, U, s2, d_s)
    Om_c0 = omega_prop2(lam, p, U)
    a0 = 1 + (U - 1) * kappa
    p0 = modes_wf(lam, a0, d_s, s2)
    pk, _, drops_k = prune(lam, np.zeros_like(kappa), U, s2, d_s)
    Om_d0 = omega_k0(lam, pk, U)
    e_c, e_d = user_energy(lam, Om_c0), user_energy(lam, Om_d0)
    m = {"P": P, "p": p, "a": a, "p0": p0, "Om_c_unscaled": Om_c0,
         "load_joint": omega_load(lam, P), "gen_joint": scale_users(lam, Om_c0, E),
         "gen_eq": p0[:, None, None] * np.full((U, U), 1.0 / U)[None] / lam[:, None, None],
         "gen_k0": scale_users(lam, Om_d0, E)}
    act = p > 0
    info = {"drops_c": drops, "drops_k0": drops_k, "active_modes_c": int(act.sum()), "active_common_c": int(act[0].sum()),
            "uses_c": int(act.sum()), "d_s": d_s,
            "energy_dev_prescale_c": float(np.abs(e_c / E - 1).max()),
            "energy_dev_prescale_k0": float(np.abs(e_d / E - 1).max()),
            "energy_users_c": user_energy(lam, m["gen_joint"]).tolist(),
            "energy_users_load": (P.sum(1)).tolist(), "energy_users_eq": user_energy(lam, m["gen_eq"]).tolist(),
            "t_bar_c": overlap_index(lam, m["gen_joint"]), "t_bar_eq": overlap_index(lam, m["gen_eq"]),
            "t_bar_k0": overlap_index(lam, m["gen_k0"]),
            "load_coefficients": load_counts(U, K, rule), "load_active": [int((P[j] > 0).sum()) for j in range(U)],
            "uses_load": int(sum(load_counts(U, K, rule))), "active_modes_k0": int((pk > 0).sum()),
            "common_active_eq": int((p0 > 0).sum())}
    if U <= K:                                                                # interior formula for t_i
        i = (p[0] > 0) & (p[1] > 0) & (kappa > 0)
        if i.any():
            p0i, p1i, ki = p[0][i], p[1][i], kappa[i]
            t = (p0i - p1i) / (p0i + (U - 1) * p1i)
            tf = ki * s2 / ((p0i + (U - 1) * p1i) / U * (1 - ki) * (1 + (U - 1) * ki))
            info["t_interior_maxrel"] = float(np.abs(t / tf - 1).max())
            info["n_interior"] = int(i.sum())
    return m, info


# ------------------------------------------------------------------ evaluators
def pu_closed(lam, P, Gs, s2):
    """per-user Wiener with the true gain: (D, U) MSE."""
    return (lam[None, None] * s2 / (Gs[:, :, None] ** 2 * P[None] + s2)).sum(2)


def a1_trace(lam, kappa, Om, Gs, s2):
    """joint LMMSE in the (A1) model: per coefficient U x U, all draws batched -> (D, U) MSE."""
    U = Om.shape[1]
    I, J = np.eye(U), np.full((U, U), 1.0 / U)
    Gi = ((I - J)[None] / (1 - kappa)[:, None, None] + J[None] / (1 + (U - 1) * kappa)[:, None, None]) / lam[:, None, None]
    out = []
    for g in Gs:
        A = Gi + g[None, :, None] * g[None, None, :] * Om / s2
        out.append(np.diagonal(np.linalg.inv(A), axis1=1, axis2=2).sum(0))
    return np.array(out)


class Full:
    """Gaussian model with the full C (user-major stacking, index u d + i)."""

    def __init__(self, lam, C, U):
        d = lam.size
        self.lam, self.C, self.U, self.d, self.n = lam, C, U, d, U * d
        self.ridge = 0.0
        while True:
            G1 = np.diag(lam) - C + self.ridge * np.eye(d)
            G0 = G1 + U * C
            try:
                self.L1, self.L0 = cholesky(G1, lower=True), cholesky(G0, lower=True)
                break
            except np.linalg.LinAlgError:
                assert self.ridge == 0.0, "Cholesky failed with the ridge"
                self.ridge = 1e-10 * lam[0]
        self.G1, self.G0 = G1, G0
        self.trLam = float(np.trace(G1 + C))
        self._Ginv = None
        self._Gam = None

    @property
    def Ginv(self):
        if self._Ginv is None:
            I = np.eye(self.d)
            B1 = cho_solve((self.L1, True), I)
            B0 = cho_solve((self.L0, True), I)
            B1, B0 = (B1 + B1.T) / 2, (B0 + B0.T) / 2
            self._Ginv = np.asfortranarray(np.kron(np.eye(self.U), B1) + np.kron(np.full((self.U, self.U), 1.0 / self.U), B0 - B1))
        return self._Ginv

    @property
    def Gam(self):
        if self._Gam is None:
            self._Gam = np.kron(np.eye(self.U), self.G1) + np.kron(np.ones((self.U, self.U)), self.C)
        return self._Gam

    def dense(self, Om, g, s2):
        """exact per-user MSE (U,): Cholesky of Gamma^{-1} + M, diag of the inverse from L^{-1}."""
        U, d = self.U, self.d
        A = self.Ginv.copy(order="F")
        ar = np.arange(d)
        for u in range(U):
            for v in range(U):
                A[u * d + ar, v * d + ar] += g[u] * g[v] * Om[:, u, v] / s2
        L, info = lapack.dpotrf(A, lower=1, clean=1, overwrite_a=1)
        assert info == 0, f"dpotrf info {info}"
        Li, info = lapack.dtrtri(L, lower=1, overwrite_c=1)
        assert info == 0, f"dtrtri info {info}"
        return np.einsum("ij,ij->j", Li, Li).reshape(U, d).sum(1)

    def cov(self, Om, g, s2):
        """covariance form (validation): tr Gamma - ||L^{-1} Psi Gamma||_F^2, Psi_i = Omega_i^{1/2} G."""
        U, d, n = self.U, self.d, self.n
        ev, Q = np.linalg.eigh(Om)
        W = (Q * np.sqrt(np.maximum(ev, 0.0))[:, None, :]) @ Q.transpose(0, 2, 1)      # (d, U, U) symmetric root
        Gam3 = self.Gam.reshape(U, d, n)
        PG = np.zeros((U, d, n))
        for r in range(U):
            for u in range(U):
                PG[r] += (W[:, r, u] * g[u])[:, None] * Gam3[u]
        PG = PG.reshape(n, n)
        PG3 = PG.reshape(n, U, d)
        S = np.zeros((n, U, d))
        for r in range(U):
            for u in range(U):
                S[:, r, :] += PG3[:, u, :] * (W[:, r, u] * g[u])[None, :]
        S = S.reshape(n, n) + s2 * np.eye(n)
        L = cholesky((S + S.T) / 2, lower=True)
        X = solve_triangular(L, PG, lower=True)
        return (np.diag(self.Gam) - (X ** 2).sum(0)).reshape(U, d).sum(1)

    def load_setup(self, Pvec, s2):
        """simultaneous diagonalization for load masks with one allocation for all users."""
        Dm = Pvec / (self.lam * s2)
        F = self.L1
        lk, O = np.linalg.eigh(F.T @ (Dm[:, None] * F))
        T = F @ O
        Z = solve_triangular(self.L0, T, lower=True)
        Rp = (Z.T @ Z - np.eye(self.d)) / self.U
        W = T.T @ T
        return {"lk": np.maximum(lk, 0.0), "W": (W + W.T) / 2, "Rp": (Rp + Rp.T) / 2}

    def load_fast(self, st, g):
        nn = 1.0 / (1.0 + g[:, None] ** 2 * st["lk"][None, :])                 # (U, d) diag of N_u'^{-1}
        Y = nn.sum(0)
        X = np.linalg.solve(np.eye(self.d) + st["Rp"] * Y[None, :], st["Rp"])
        XW = X * st["W"]
        return nn @ np.diag(st["W"]) - ((nn @ XW.T) * nn).sum(1)

    def eq_woodbury(self, p0, g, s2):
        """gen_eq: M = (g g^T) (x) diag(p0 / (lam U s2)), rank d_b -> exact per-user MSE (U,)."""
        dh = np.sqrt(p0 / (self.lam * self.U * s2))
        sg, g2 = g.sum(), (g ** 2).sum()
        H = g2 * self.G1 + sg ** 2 * self.C
        L = cholesky(np.eye(self.d) + dh[:, None] * H * dh[None, :], lower=True)
        out = []
        for u in range(self.U):
            X = solve_triangular(L, dh[:, None] * (g[u] * self.G1 + sg * self.C), lower=True)
            out.append(self.trLam - (X ** 2).sum())
        return np.array(out)


# ------------------------------------------------------------------ main
def g0(k, U):
    """(1 + (U-1) k)^2 / (1 + (U-1) k^2): the overlap slope gain IF the first active common mode has kappa = k."""
    return (1 + (U - 1) * k) ** 2 / (1 + (U - 1) * k ** 2)


def slope_pred(lam, kappa, U):
    """exact first-mode low-SNR slope gains in (A1). The first active Proposition-2 mode is the common mode of
    i0 = argmax_i lam_i a_{i,0}^2 (MSE reduction lam_i0 a_{i0,0}^2 / s2 per unit energy); the load masks put their first
    energy on coefficient 1 (lam_1 (1 + (U-1) kappa_1^2) / s2 with the joint receiver, lam_1 / s2 per-user Wiener)."""
    a0 = 1 + (U - 1) * kappa
    i0 = int(np.argmax(lam * a0 ** 2))
    num = float(lam[i0] * a0[i0] ** 2)
    return {"first_common_mode": i0, "kappa_first_common_mode": float(kappa[i0]),
            "coef1_is_first_common_mode": bool(i0 == 0),
            "pred_c_over_b": num / float(lam[0] * (1 + (U - 1) * kappa[0] ** 2)), "pred_c_over_a": num / float(lam[0])}


def summarize(mse, tot):
    """mse (D, U) -> metrics; nmse_se = Monte Carlo standard error of the mean over the D gain draws."""
    D, U = mse.shape
    nu = mse / tot
    c = np.sqrt(np.maximum(0.0, 1 - nu))
    m = nu.sum(1) / U
    return {"nmse": float(m.mean()), "nmse_sd": float(m.std()),
            "nmse_se": float(m.std(ddof=1) / math.sqrt(D)) if D > 1 else None, "draws": int(D),
            "nmse_users": nu.mean(0).tolist(), "cos_proxy": float(c.mean()), "weakest_proxy": float(c.min(1).mean())}


def paired(x, y, tot):
    """NMSE of x minus NMSE of y on the same gain draws (the first min(D_x, D_y)), mean and standard error."""
    k = min(x.shape[0], y.shape[0])
    dl = (x[:k].sum(1) - y[:k].sum(1)) / (x.shape[1] * tot)
    return {"mean": float(dl.mean()), "se": float(dl.std(ddof=1) / math.sqrt(k)) if k > 1 else None, "draws": int(k)}


def decomp(out, mo, gs):
    """relative changes of the mean NMSE: (b) over (a), (c) over (b), (d) over (b) (U <= K: Gram identity, 0; U > K:
    pruning/split effect of the kappa = 0 design), (c) over (d) (U > K: the shared-content overlap gain)."""
    a_, b, c, d_ = (out[(mo, gs, s)].sum(1) for s in ("load_pu", "load_joint", "gen_joint", "gen_k0"))
    k = d_.shape[0]
    return {"b_over_a": float(b.mean() / a_.mean() - 1), "c_over_b": float(c.mean() / b.mean() - 1),
            "d_over_b": float(d_.mean() / b[:k].mean() - 1), "c_over_d": float(c[:k].mean() / d_.mean() - 1),
            "draws_d": int(k)}


def rel(x, y):
    return float(np.abs(np.asarray(x) / np.asarray(y) - 1).max())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ds", default="coco", choices=["coco", "flickr"])
    ap.add_argument("--mix", type=float, default=None)
    ap.add_argument("--K", type=int, default=4)
    ap.add_argument("--U", type=int, default=4)
    ap.add_argument("--snrs", type=float, nargs="+", default=list(SNRS))
    ap.add_argument("--draws", type=int, default=100)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--k0_draws", type=int, default=-1, help="Rayleigh draws of gen_k0 in the full model (-1: all)")
    ap.add_argument("--models", nargs="+", default=["a1", "full"])
    ap.add_argument("--load_rule", default="spec", choices=list(LOAD_RULES),
                    help="spec: floor(d_s/U) coefficients per user (equal split, shared_masks); rank: manuscript rank counts")
    ap.add_argument("--mix_fit", default="spec", choices=["spec", "shared_masks"],
                    help="mixture fit: spec U per tuple (overlap_bound order); shared_masks 5 per tuple (its order)")
    ap.add_argument("--validate", action="store_true")
    ap.add_argument("--outdir", default=os.path.join(DATA, "shared"))
    ap.add_argument("--tag", default="")
    a = ap.parse_args()
    t_all = time.time()
    K, U = a.K, a.U
    assert U >= 2
    F = cs.fit_content(a.ds, a.mix, U, a.mix_fit)
    lam, C, kappa = F["lam"], F["C"], F["kappa"]
    d = lam.size
    tot = float(lam.sum())
    d_s = K * d
    rule = a.load_rule
    name = a.ds + (f"_mix{a.mix:g}" if a.mix is not None else "") + f"_K{K}_U{U}" + ("_rank" if rule == "rank" else "") + a.tag
    rng = np.random.default_rng(a.seed)
    Gs = np.sqrt((rng.standard_normal((a.draws, U)) ** 2 + rng.standard_normal((a.draws, U)) ** 2) / 2.0)
    gainsets = {"equal": np.ones((1, U)), "rayleigh": Gs}
    k0d = a.draws if a.k0_draws < 0 else min(a.k0_draws, a.draws)
    t0 = time.time()
    full = Full(lam, C, U) if "full" in a.models else None
    kb = float((lam * kappa).sum() / tot)
    od = cs.offdiag(C)
    meta = {"ds": a.ds, "mix": a.mix, "mix_fit": F["mix_fit"], "fit_tuple": F["tuple"], "K": K, "U": U, "d_b": d,
            "d_s": d_s, "slots": NB * K,
            "energy_per_user": d_s / U, "snrs": a.snrs, "draws": a.draws, "seed": a.seed, "k0_draws_full": k0d,
            "models": a.models, "lam_sum": tot, "kappa_bar": kb, "kappa_1": float(kappa[0]),
            "kappa_group_lamw": ((lam * kappa).reshape(NB, SEG).sum(1) / lam.reshape(NB, SEG).sum(1)).tolist(),
            "offdiag_share_sq": od["share_sq"], "offdiag_share_norm": od["share_norm"], "ridge": full.ridge if full else None,
            "load_rule": rule, "load_coefficients": load_counts(U, K, rule), "setup_s": time.time() - t0,
            "gains_rayleigh_mean_g2": float((Gs ** 2).mean()),
            "checks_note": "U <= K: (d) = (b) is an identity of the Gram model (checks Q^T Q = I only; keys "
                           "*_k0_gram_identity_rel). U > K: (d) - (b) is the effect of the pruned kappa = 0 design "
                           "(and, with --load_rule rank, of the unequal split), not a check; the shared-content "
                           "(overlap) gain is (d) -> (c): diag[].decomp_rel.c_over_d. nmse_se: Monte Carlo SE over "
                           "the gain draws; scheme differences: minus_load_joint / minus_load_pu (paired)"}
    sp = slope_pred(lam, kappa, U)
    meta["slope_gains"] = {"G0": sp["pred_c_over_b"], "over_per_user": sp["pred_c_over_a"], **sp,
                           "G0_by_U": {str(v): slope_pred(lam, kappa, v)["pred_c_over_b"] for v in (2, 3, 4, 5)},
                           "over_per_user_by_U": {str(v): slope_pred(lam, kappa, v)["pred_c_over_a"] for v in (2, 3, 4, 5)},
                           "G0_formula_kappa1": g0(kappa[0], U),
                           "over_per_user_formula_kappa1": (1 + (U - 1) * kappa[0]) ** 2,
                           "G0_formula_kappa1_by_U": {str(v): g0(kappa[0], v) for v in (2, 3, 4, 5)},
                           "G0_formula_kappa_bar": g0(kb, U),
                           "note": "G0 / over_per_user (= pred_c_over_b / pred_c_over_a) are the low-SNR slope gains; "
                                   "the *_formula_kappa1 values equal them only if coef1_is_first_common_mode; "
                                   "G0_formula_kappa_bar is a summary number, not a slope gain"}
    print(f"[{name}] kappa_bar {kb:.4f} kappa_1 {kappa[0]:.3f} offdiag {od['share_sq']:.3f} ridge {meta['ridge']} "
          f"load_rule {rule} {meta['load_coefficients']} mix_fit {meta['mix_fit']} fit_tuple {meta['fit_tuple']}; "
          f"G0 {sp['pred_c_over_b']:.4f} (first common mode {sp['first_common_mode']}, kappa "
          f"{sp['kappa_first_common_mode']:.3f}; formula with kappa_1 {meta['slope_gains']['G0_formula_kappa1']:.4f})",
          flush=True)
    # numeric slope check at -70 dB, equal gains (a1 closed forms; full model by the dense solve)
    s2 = 1e7
    m, _ = make_masks(lam, kappa, U, K, s2, rule)
    one = np.ones((1, U))
    red = {"load_pu": U * tot - pu_closed(lam, m["P"], one, s2).sum()}
    for k in ("load_joint", "gen_joint"):
        red[k] = U * tot - a1_trace(lam, kappa, m[k], one, s2).sum()
    meta["slope_gains"]["num_a1_c_over_b"] = float(red["gen_joint"] / red["load_joint"])
    meta["slope_gains"]["num_a1_c_over_a"] = float(red["gen_joint"] / red["load_pu"])
    if full is not None:
        rf = {k: U * full.trLam - full.dense(m[k], one[0], s2).sum() for k in ("load_joint", "gen_joint")}
        meta["slope_gains"]["num_full_c_over_b"] = float(rf["gen_joint"] / rf["load_joint"])
        meta["slope_gains"]["num_full_c_over_a"] = float(rf["gen_joint"] / red["load_pu"])
    print(f"   slope gains: G0 pred {meta['slope_gains']['pred_c_over_b']:.4f} num a1 "
          f"{meta['slope_gains']['num_a1_c_over_b']:.4f} full {meta['slope_gains'].get('num_full_c_over_b', float('nan')):.4f}; "
          f"over per-user pred {meta['slope_gains']['pred_c_over_a']:.4f} num a1 {meta['slope_gains']['num_a1_c_over_a']:.4f}",
          flush=True)

    res = {mo: {gs: {s: [] for s in SCHEMES} for gs in gainsets} for mo in a.models}
    raw = {}
    closed, diag = [], []
    for snr in a.snrs:
        ts = time.time()
        s2 = 10 ** (-snr / 10.0)
        m, info = make_masks(lam, kappa, U, K, s2, rule)
        tm = {}
        # closed forms at equal gains in (A1)
        p, aa, P, p0 = m["p"], m["a"], m["P"], m["p0"]
        a0, a1 = 1 + (U - 1) * kappa, 1 - kappa
        cf = {"gen_joint_modal": float((lam[None] * s2 / (p + s2 / aa)).sum())}
        cf["gen_eq_modal"] = float((lam * s2 / (p0 + s2 / a0) + (U - 1) * lam * a1).sum())
        same_alloc = bool(np.all(P == P[0]))
        if same_alloc:
            cf["load_joint_modal"] = float((lam * s2 / (P[0] + s2 / a0) + (U - 1) * lam * s2 / (P[0] + s2 / a1)).sum())
        tr_unscaled = float(a1_trace(lam, kappa, m["Om_c_unscaled"], one, s2).sum())
        info["modal_vs_a1_trace_rel"] = abs(tr_unscaled / cf["gen_joint_modal"] - 1)
        out = {}
        for mo in a.models:
            for gs, G in gainsets.items():
                key = (mo, gs)
                t1 = time.time()
                out[key + ("load_pu",)] = pu_closed(lam, P, G, s2)
                tm[f"{mo}_{gs}_load_pu"] = time.time() - t1
                if mo == "a1":
                    for s in SCHEMES[1:]:
                        t1 = time.time()
                        out[key + (s,)] = a1_trace(lam, kappa, m[s], G, s2)
                        tm[f"{mo}_{gs}_{s}"] = time.time() - t1
                    continue
                t1 = time.time()
                if same_alloc:
                    st = full.load_setup(P[0], s2)
                    out[key + ("load_joint",)] = np.array([full.load_fast(st, g) for g in G])
                else:
                    out[key + ("load_joint",)] = np.array([full.dense(m["load_joint"], g, s2) for g in G])
                tm[f"{mo}_{gs}_load_joint"] = time.time() - t1
                t1 = time.time()
                out[key + ("gen_joint",)] = np.array([full.dense(m["gen_joint"], g, s2) for g in G])
                tm[f"{mo}_{gs}_gen_joint"] = time.time() - t1
                t1 = time.time()
                out[key + ("gen_eq",)] = np.array([full.eq_woodbury(p0, g, s2) for g in G])
                tm[f"{mo}_{gs}_gen_eq"] = time.time() - t1
                t1 = time.time()
                Gk = G if gs == "equal" else G[:k0d]
                out[key + ("gen_k0",)] = np.array([full.dense(m["gen_k0"], g, s2) for g in Gk])
                tm[f"{mo}_{gs}_gen_k0"] = time.time() - t1
        # checks
        if "a1" in a.models:
            for gs in gainsets:
                if U <= K:                                                    # Gram identity (Q^T Q = I), not a test of (d)
                    info[f"a1_{gs}_k0_gram_identity_rel"] = rel(out[("a1", gs, "gen_k0")].sum(1), out[("a1", gs, "load_joint")].sum(1))
            if same_alloc:
                info["a1_load_joint_modal_rel"] = abs(out[("a1", "equal", "load_joint")].sum() / cf["load_joint_modal"] - 1)
            info["a1_gen_eq_modal_rel"] = abs(out[("a1", "equal", "gen_eq")].sum() / cf["gen_eq_modal"] - 1)
            info["a1_gen_joint_scaled_vs_modal"] = float(out[("a1", "equal", "gen_joint")].sum() / cf["gen_joint_modal"] - 1)
        if full is not None:
            g = Gs[0]
            if same_alloc:
                info["full_load_fast_vs_dense_rel"] = rel(out[("full", "rayleigh", "load_joint")][0].sum(),
                                                          full.dense(m["load_joint"], g, s2).sum())
            info["full_eq_woodbury_vs_dense_rel"] = rel(out[("full", "rayleigh", "gen_eq")][0].sum(),
                                                        full.dense(m["gen_eq"], g, s2).sum())
            if U <= K:                                                        # Gram identity (Q^T Q = I), not a test of (d)
                kk = out[("full", "rayleigh", "gen_k0")].shape[0]
                dk = out[("full", "rayleigh", "gen_k0")].sum(1) / out[("full", "rayleigh", "load_joint")][:kk].sum(1) - 1
                info["full_rayleigh_k0_gram_identity_rel_max"] = float(np.abs(dk).max())
                info["full_equal_k0_gram_identity_rel"] = abs(float(out[("full", "equal", "gen_k0")].sum() /
                                                                    out[("full", "equal", "load_joint")].sum() - 1))
            if a.validate:
                t1 = time.time()
                v = {}
                for s in SCHEMES[1:]:
                    v[s] = rel(full.cov(m[s], g, s2).sum(), full.dense(m[s], g, s2).sum())
                fa1 = Full(lam, np.diag(kappa * lam), U)
                v["a1_dense_vs_percoef_gen_joint"] = rel(fa1.dense(m["gen_joint"], g, s2).sum(),
                                                         a1_trace(lam, kappa, m["gen_joint"], g[None], s2).sum())
                v["a1_dense_vs_percoef_gen_k0"] = rel(fa1.dense(m["gen_k0"], g, s2).sum(),
                                                      a1_trace(lam, kappa, m["gen_k0"], g[None], s2).sum())
                from pca_csi import wf
                n0 = load_counts(U, K, rule)[0]
                Pw = wf(lam[:n0], np.ones((1, n0)), np.array([d_s / U]), s2)[0]
                v["load_alloc_vs_pca_csi_wf_rel"] = float(np.abs(Pw - P[0, :Pw.size]).max() / P[0].max())
                v["time_s"] = time.time() - t1
                info["validate_cov_vs_dense_rel"] = v
        for (mo, gs, s), x in out.items():
            sm = summarize(x, tot)
            sm["minus_load_joint"] = paired(x, out[(mo, gs, "load_joint")], tot)
            sm["minus_load_pu"] = paired(x, out[(mo, gs, "load_pu")], tot)
            if s == "gen_joint":
                sm["minus_gen_k0"] = paired(x, out[(mo, gs, "gen_k0")], tot)
            res[mo][gs][s].append(sm)
            raw[f"{mo}_{gs}_{s}_snr{snr:g}"] = x
        info["decomp_rel"] = {f"{mo}_{gs}": decomp(out, mo, gs) for mo in a.models for gs in gainsets}
        info["time_s"] = tm
        info["snr_time_s"] = time.time() - ts
        closed.append({k: v / (U * tot) for k, v in cf.items()})
        diag.append(info)
        line = "  ".join(f"{s}={res['full' if 'full' in a.models else 'a1']['rayleigh'][s][-1]['nmse']:.4f}" for s in SCHEMES)
        line_e = "  ".join(f"{s}={res['a1']['equal'][s][-1]['nmse']:.4f}" for s in SCHEMES) if "a1" in a.models else ""
        print(f"  SNR {snr:g}: rayleigh {'full' if 'full' in a.models else 'a1'}: {line}", flush=True)
        if line_e:
            print(f"          equal a1: {line_e}  modal(c) {closed[-1]['gen_joint_modal']:.4f}", flush=True)
        if "full" in a.models:
            print(f"          equal full: " + "  ".join(f"{s}={res['full']['equal'][s][-1]['nmse']:.4f}" for s in SCHEMES), flush=True)
        chk = {k: v for k, v in info.items() if (k.endswith("_rel") or k.endswith("_rel_max") or k.startswith("energy_dev")
               or k in ("t_bar_c", "drops_c", "drops_k0", "active_modes_c", "active_modes_k0", "uses_load",
                        "t_interior_maxrel", "a1_gen_joint_scaled_vs_modal")) and k != "decomp_rel"}
        print(f"          checks {json.dumps({k: (round(v, 12) if isinstance(v, float) else v) for k, v in chk.items()})}", flush=True)
        if "validate_cov_vs_dense_rel" in info:
            print(f"          validate {json.dumps(info['validate_cov_vs_dense_rel'])}", flush=True)
        print(f"          decomp_rel (b/a, c/b, d/b, c/d) " + "  ".join(
            f"{k}: " + "/".join(f"{v[q]:+.4f}" for q in ("b_over_a", "c_over_b", "d_over_b", "c_over_d"))
            for k, v in info["decomp_rel"].items()), flush=True)
        if "full" in a.models:
            r_ = res["full"]["rayleigh"]
            print(f"          full rayleigh SE: " + "  ".join(f"{s}={r_[s][-1]['nmse_se']:.4f}" for s in SCHEMES
                                                          if r_[s][-1]["nmse_se"] is not None)
                  + f"; paired c-b {r_['gen_joint'][-1]['minus_load_joint']['mean']:+.5f} (se "
                  f"{r_['gen_joint'][-1]['minus_load_joint']['se'] or 0:.5f}), c-d {r_['gen_joint'][-1]['minus_gen_k0']['mean']:+.5f} "
                  f"(se {r_['gen_joint'][-1]['minus_gen_k0']['se'] or 0:.5f})", flush=True)
        print(f"          time {info['snr_time_s']:.1f} s  " + json.dumps({k: round(v, 2) for k, v in tm.items()}), flush=True)
    R = {"meta": meta, "schemes": list(SCHEMES), "results": res, "closed_equal_a1_nmse": closed, "diag": diag,
         "runtime_s": time.time() - t_all}
    os.makedirs(a.outdir, exist_ok=True)
    fn = os.path.join(a.outdir, f"theory_{name}.json")
    json.dump(R, open(fn, "w"), indent=1)
    np.savez_compressed(fn[:-5] + ".npz", gains=Gs, snrs=np.array(a.snrs), **raw)
    print(f"wrote {fn} in {R['runtime_s']:.0f} s", flush=True)


if __name__ == "__main__":
    main()
