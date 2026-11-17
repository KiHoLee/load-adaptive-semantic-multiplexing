# =========================================================
# shared_masks.py — real-valued masks for users that describe one scene (shared content), 2026-11-16.
# Content: BERT caption embeddings, five captions per image (emb_captions.py). The U users of a frame send U
# distinct captions of one test image, so their PCA coefficients c_u = V^T b_u are correlated. Fit on the training
# captions only: lam, V from the second moment of all 40,000 fit captions, C = mean over images and ordered pairs
# u != v of c_u c_v^T (symmetrized), kappa_i = clip(C_ii / lam_i, 0, 0.99), joint covariance (user-major)
#   Gamma = I_U (x) (Lam - C) + 1 1^T (x) C.
# --dataset agnews: independent content (C = 0), AG News test texts, loaded as pca_eval.Source does.
# --mix KAPPA: b_u = normalize(sqrt(kappa) b_s + sqrt(1 - kappa) b_{p,u}), b_s one caption of the frame's image and
#   b_{p,u} captions of U other, distinct images; lam, V, C fitted on U-user mixtures of the fit images, one tuple per
#   fit image, seed 0, with content_stats.mix_tuples and content_stats.fit (the same fit as content_stats.py and
#   shared_theory.py --mix_fit spec).
# --control: captions of U distinct images (true C = 0); the design and receiver use C = 0 (--control_c zero) or
#   the fitted C (--control_c fit, mismatch).
# --gauss: the captions are replaced by Gaussian draws with the fitted (true) Gamma; the exact NMSE trace of every
#   scheme is computed per frame next to the Monte Carlo NMSE (validation). --a1 (with --gauss): the content model
#   and the receivers use the diagonal surrogate C = diag(kappa_i lam_i), so with --gains equal the Monte Carlo
#   NMSE of Proposition 2 is compared with its closed form (mse_design_equal_gain).
# Channel: block fading, independent Rayleigh |h_u| ~ |CN(0,1)| per user and frame (--gains equal: |h_u| = 1),
# optional near-far offsets uniform in +-X dB, known at the receiver; the transmitter knows only the SNR (unit design
# gain). Noise s2 = 10^(-SNR/10) per real use, d_s = K d_b uses, per-user energy d_s/U. Options: --phase DEG
# (residual phase phi ~ N(0, DEG^2): effective gain g cos(phi); the receiver uses |h| (--rxgain abs) or g cos(phi)
# (--rxgain eff)); --pilot: the overlapping members (gen_*, rnd) need known gains, so each user sends one real pilot
# use per block, taken from the block (d_s - U data uses), with pilot energy --pilot_energy Ep (default 1, unit
# amplitude) taken from the user's energy (d_s/U - Ep for the data); gain estimate = received pilot value / sqrt(Ep)
# (error variance s2/Ep), used by the receiver as the known gain. The non-overlapping load masks (load_pu,
# load_joint) are not charged: they keep d_s uses and d_s/U, and their receivers use the effective gain (the
# manuscript's blind estimate (8) from the user's own uses adds < 0.001), so they equal the no-pilot run frame by
# frame. gen_joint_true: the masks of gen_joint (pilot-charged under --pilot) with the true effective gain at the
# receiver (genie reference of the gain estimate in the same run).
# --c_from self | other | N (C of the design and receiver fitted on the other caption corpus, in this corpus' V basis,
# or on the first N fit images only; a C that makes Gamma indefinite is projected, see fix_gamma).
# Schemes (the same frames, captions, gains and noise draws for every scheme of one point, so they are paired):
#   load_pu    (a) load masks: user u sends its leading d_u coefficients with (12) at E = d_s/U (copies act as one
#              use), z_{u,i} = g_u a_i c_{u,i} + xi; per-user Wiener receiver with the receiver gain.
#              --load_rule spec (default, SPEC.md): d_u = min(floor(d_s/U), d_b) for every user; --load_rule rank
#              (the manuscript's Proposed, pca_eval.per_load, shared_theory --load_rule rank): user j sends
#              min(ceil((N - j)/U) * 128, d_b), N = 6K slots. The rules differ only at U = 5 (output suffix _rank).
#   load_joint (b) load masks + joint LMMSE with the full C (unsent coefficients from the joint prior)
#   gen_joint  (c) Proposition 2 masks W_i = lam_i^{-1/2} (sqrt(p_i1) I + (sqrt(p_i0) - sqrt(p_i1)) 1 1^T / U) on U uses,
#              p water-filled over common (a0 = 1 + (U-1) kappa) and difference (a1 = 1 - kappa) modes, sum p = U E;
#              if more than n_data modes are active (U > K or pilot), greedy pruning (drop the active mode with the
#              smallest MSE reduction, re-solve nu). When U d_b > n_data (overload, or pilot at U = K) every
#              coefficient is sent in modal form, one use per active mode (rows diag(sqrt(p/lam)) Q^T of the active
#              modes; W_i above when all U modes are active), so the masks use at most n_data uses (info n_uses); a
#              coefficient with m_i < U kept modes sends them on m_i uses (difference modes from the real-DFT
#              complement basis, index rotated cyclically with i), and each user is then scaled exactly to its
#              energy (deviation before scaling stored); joint LMMSE with the full C
#   gen_eq     (c') all energy in the common modes: every user sends coefficient i with one value on one common use
#              (water-filling over the common modes only) + joint LMMSE
#   gen_k0     (d) kappa forced to 0 (p_i0 = p_i1 = P_i of (12)), W_i = sqrt(P_i / lam_i) Q_i with a random orthogonal
#              Q_i per coefficient (seed 4242 + U): dense overlap with the Gram matrix of (b) + joint LMMSE
#   gen_pc     (e) the masks of (c) + the per-coefficient receiver with (A1) only (U x U per coefficient, kappa_i)
#   rnd        dense random masks x_u = R_u diag(sqrt(P / lam)) c_u, R_u (n_data x d_b) random orthonormal columns
#              fixed per user (seed 31 + u), P of (12) at E, received sum over users + noise, joint LMMSE with the
#              dataset's Gamma by an exact dense solve (precision form); U <= K only
# Joint LMMSE: c_hat = Gamma Psi^T (Psi Gamma Psi^T + s2 I)^{-1} y, Cholesky per frame in float64 (torch), with only
# the active observations (equal to (Gamma^{-1} + Psi^T Psi / s2)^{-1} Psi^T y / s2; --check compares both dense forms
# on frame 0). Metrics: per-user CosSim(b_u, V c_hat_u), weakest user per frame, NMSE = sum ||c - c_hat||^2 / sum ||c||^2.
# Validation (trace computed): mc_over_theory = mean_f err_f / mean_f tr_f (err_f, tr_f the frame's total squared
# error and exact Gaussian MSE), with the paired standard error sd(err_f - tr_f) / sqrt(F) / mean(tr_f) and z.
# Frames and seeds: one test image per frame (frame f -> image f mod n_test), U distinct captions with
# rng = default_rng(123 + SNR + 1000 U); gains default_rng(5123 + SNR + 1000 U), one (U, 5) block of normals per
# frame (Re h, Im h, offset via the normal CDF, phase, pilot noise; drawn whatever the options, so the gain, phase,
# pilot and near-far runs share h); noise default_rng(9123 + SNR + 1000 U) with max(5, U, K) noise rows per
# coefficient (5 whenever U, K <= 5, so those draws did not change when U > 5 was allowed on 2026-11-17);
# Gaussian draws default_rng(7123 + SNR + 1000 U). The noise, captions and gains are therefore the same for K = 4 and
# K = 2 at fixed U and SNR, and differ between U. All draws are made frame by frame (prefix-stable): a run with F
# frames uses the first F frames of a longer run.
# Output: ../data/shared/shared_<tag>.json and shared_<tag>.npz (per-frame per-user CosSim, squared error and ||c||^2
# of every scheme, true and receiver gains, per-frame Gaussian trace when computed), tag =
# <dataset>[_mix<k>]_K<K>_U<U>[_ctrl|_ctrlfit][_nf<X>][_ph<deg><rxgain>][_pilot[E<Ep>]][_cf<other|N>][_rank][_geq][_a1][_gauss]<suffix>.
# Usage: python shared_masks.py --dataset coco --U 4 --K 4 --frames 2000
#        python shared_masks.py --dataset coco --U 4 --K 2 --gauss --snrs 0 --frames 300 --outdir ../data/shared/smoke
#        python shared_masks.py --dataset coco --U 4 --K 4 --gauss --a1 --gains equal --snrs 0 --frames 1000 \
#               --modes gen_joint gen_k0 --outdir ../data/shared/smoke      (Proposition 2 closed form vs Monte Carlo)
# =========================================================
import argparse, json, math, os, time
import numpy as np
import torch
from scipy.special import ndtr
import content_stats as cs

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(os.path.dirname(HERE), "data")
D_B, NCAP, R_NOISE = 768, 5, 5
SNRS = (-10, -5, 0, 5, 10, 20)
MODES = ("load_pu", "load_joint", "gen_joint", "gen_eq", "gen_k0", "gen_pc")
JOINT = ("load_joint", "gen_joint", "gen_eq", "gen_k0", "gen_joint_true")
OVERLAP = ("gen_joint", "gen_eq", "gen_k0", "gen_pc", "rnd")       # need known gains: charged the pilot (--pilot)
OTHER = {"coco": "flickr", "flickr": "coco"}
TD = torch.float64


# ----------------------------------------------------------------- content
def load_caps(ds):
    C = torch.load(os.path.join(DATA, f"emb_{ds}_caps_bert-base-uncased.pt"), map_location="cpu", weights_only=False)
    return C["train"].double().numpy().reshape(-1, NCAP, D_B), C["test"].double().numpy()


def eig_fit(X):
    """second moment of the rows of X -> (lam descending, clamped at 1e-12, V), as pca_eval.Source."""
    lam, V = np.linalg.eigh(X.T @ X / X.shape[0])
    o = np.argsort(lam)[::-1]
    return np.maximum(lam[o], 1e-12), V[:, o]


def cross_moment(Ct):
    """Ct (N, m, d) coefficients of the m users of N scenes -> mean over scenes and ordered pairs u != v of c_u c_v^T."""
    N, m, d = Ct.shape
    S = Ct.sum(1)
    X = Ct.reshape(-1, d)
    C = (S.T @ S - X.T @ X) / (N * m * (m - 1))
    return (C + C.T) / 2


def pick_others(rng, idx, n, m):
    """m distinct items out of n, all different from idx[f], for every frame f (frame by frame: prefix-stable)."""
    if m == 0:
        return np.zeros((len(idx), 0), int)
    o = np.stack([rng.choice(n - 1, m, replace=False) for _ in idx])
    return o + (o >= idx[:, None])


def mixtures(caps, kappa, U, rng, idx, n):
    """test frames of --mix, frame by frame (prefix-stable), draws per frame in the order of content_stats.mix_tuples:
    user u of scene f: normalize(sqrt(kappa) b_s + sqrt(1 - kappa) b_{p,u}), b_s a random caption of image idx[f],
    b_{p,u} random captions of U other distinct images (from the same split)."""
    x = np.empty((len(idx), U, caps.shape[-1]))
    for k, f in enumerate(idx):
        sc = rng.integers(0, NCAP)
        o = rng.choice(n - 1, U, replace=False)
        o = o + (o >= f)
        pc = rng.integers(0, NCAP, U)
        x[k] = math.sqrt(kappa) * caps[f, sc][None, :] + math.sqrt(1.0 - kappa) * caps[o, pc]
    return x / np.linalg.norm(x, axis=-1, keepdims=True)


def fix_gamma(lam, C, U, name):
    """Cholesky of Lam - C and Lam + (U-1) C; relative ridge 1e-10 lam_max if one fails; if it still fails (a mismatched
    C), clip the eigenvalues of Lam^{-1/2} C Lam^{-1/2} to [-1/(U-1) + 1e-6, 1 - 1e-6]. Returns lam, C, notes."""
    notes = {}

    def ok(l, c):
        try:
            np.linalg.cholesky(np.diag(l) - c)
            if U > 1:
                np.linalg.cholesky(np.diag(l) + (U - 1) * c)
            return True
        except np.linalg.LinAlgError:
            return False
    if not ok(lam, C):
        lam = lam + 1e-10 * lam.max()
        notes["ridge"] = 1e-10 * float(lam.max())
        if not ok(lam, C):
            s = 1.0 / np.sqrt(lam)
            mu, E = np.linalg.eigh(s[:, None] * C * s[None, :])
            lo = -1.0 / (U - 1) + 1e-6 if U > 1 else -1.0
            notes["projected_eigs"] = int(((mu < lo) | (mu > 1 - 1e-6)).sum())
            mu = np.clip(mu, lo, 1 - 1e-6)
            C = (E * mu) @ E.T / (s[:, None] * s[None, :])
            C = (C + C.T) / 2
    if notes:
        print(f"fix_gamma {name}: {notes}", flush=True)
    return lam, C, notes


class Content:
    def __init__(self, a):
        self.a, U = a, a.U
        self.kind = "indep" if a.dataset == "agnews" else ("mix" if a.mix is not None else ("control" if a.control else "shared"))
        info = {"kind": self.kind}
        if a.dataset == "agnews":
            C = torch.load(os.path.join(DATA, "emb_ag_news_bert-base-uncased.pt"), map_location="cpu", weights_only=False)
            Etr, self.test = C["train"].double().numpy(), C["test"].double().numpy()
            self.lam, self.V = eig_fit(Etr)
            Cself = np.zeros((D_B, D_B))
        else:
            tr, self.test = load_caps(a.dataset)
            if self.kind == "mix":
                T = cs.mix_tuples(tr, a.mix, U, seed=0)                 # U users per fit image, seed 0
                Fm = cs.fit(T)                                          # the fit of content_stats / shared_theory
                self.lam, self.V = Fm["lam"], Fm["V"]
                Cfit = T @ self.V
                Cself = Fm["C"]
                info.update({"fit_tuple": int(T.shape[1]), "mix_fit": "content_stats.mix_tuples + fit (spec)"})
            else:
                self.lam, self.V = eig_fit(tr.reshape(-1, D_B))
                Cfit = tr @ self.V
                Cself = cross_moment(Cfit)
                info["fit_tuple"] = NCAP                               # the five captions of a fit image
        self.n_test = self.test.shape[0]
        Ctrue = np.zeros((D_B, D_B)) if self.kind in ("indep", "control") else Cself
        if self.kind == "indep" or (self.kind == "control" and a.control_c == "zero"):
            Cmod = np.zeros((D_B, D_B))
        elif a.c_from == "self":
            Cmod = Cself
        elif a.c_from in ("other", "other_corpus"):
            tro, _ = load_caps(OTHER[a.dataset])
            Cmod = cross_moment(tro @ self.V)
        else:
            Cmod = cross_moment(Cfit[:int(a.c_from)])
        if getattr(a, "a1", False):                                     # diagonal surrogate (A1) as truth and model
            k1 = np.clip(np.diag(Cself) / self.lam, 0.0, 0.99)
            Ctrue = Cmod = np.diag(k1 * self.lam)
            info["a1"] = True
        self.lam_true, self.C_true, n1 = fix_gamma(self.lam.copy(), Ctrue, U, "true")
        self.lam_m, self.C, n2 = fix_gamma(self.lam.copy(), Cmod, U, "model")
        self.D = np.diag(self.lam_m) - self.C
        self.kap = np.clip(np.diag(self.C) / self.lam_m, 0.0, 0.99)
        self.model_true = bool(np.array_equal(self.C_true, self.C) and np.array_equal(self.lam_true, self.lam_m))
        lam = self.lam_m
        info.update({"n_test": self.n_test, "lam_min": float(lam.min()), "lam_max": float(lam.max()),
                     "lam_sum": float(lam.sum()), "participation_ratio": float(lam.sum() ** 2 / (lam ** 2).sum()),
                     "trC_model": float(np.trace(self.C)), "trC_self": float(np.trace(Cself)),
                     "kappa_bar": float((lam * self.kap).sum()),
                     "kappa_group_mean": [float(self.kap[g * 128:(g + 1) * 128].mean()) for g in range(6)],
                     "kappa_group_lamw": [float((lam * self.kap)[g * 128:(g + 1) * 128].sum() / lam[g * 128:(g + 1) * 128].sum()) for g in range(6)],
                     "kappa_clipped_low": int((np.diag(self.C) / lam < 0).sum()),
                     "kappa_clipped_high": int((np.diag(self.C) / lam > 0.99).sum()),
                     "offdiag_frob2_share_C": float(((self.C - np.diag(np.diag(self.C))) ** 2).sum() / max((self.C ** 2).sum(), 1e-300)),
                     "fix_true": n1, "fix_model": n2, "model_is_true": self.model_true})
        self.info = info

    def frames(self, rng, F, U):
        """coefficients (F, U, d_b) of the U users of every frame."""
        n = self.n_test
        idx = np.arange(F) % n
        if self.kind == "indep":
            b = self.test[np.concatenate([idx[:, None], pick_others(rng, idx, n, U - 1)], 1)]
        elif self.kind == "control":                                     # frame by frame: images, then caption slots
            img, cap = np.empty((F, U), int), np.empty((F, U), int)
            for f in range(F):
                img[f] = np.concatenate([idx[f:f + 1], pick_others(rng, idx[f:f + 1], n, U - 1)[0]])
                cap[f] = rng.integers(0, NCAP, U)
            b = self.test[img, cap]
        elif self.kind == "mix":
            b = mixtures(self.test, self.a.mix, U, rng, idx, n)
        else:
            assert U <= NCAP, "at most five captions per image"
            caps = rng.random((F, NCAP)).argsort(1)[:, :U]
            b = self.test[idx[:, None], caps]
        return b @ self.V

    def gauss(self, rng, F, U, Q):
        """Gaussian draws with the true Gamma: mode 0 ~ N(0, Lam + (U-1) C), modes 1.. ~ N(0, Lam - C)."""
        L0 = np.linalg.cholesky(np.diag(self.lam_true) + (U - 1) * self.C_true)
        L1 = np.linalg.cholesky(np.diag(self.lam_true) - self.C_true)
        w = rng.standard_normal((F, U, D_B))
        z = np.empty_like(w)
        z[:, 0] = w[:, 0] @ L0.T
        z[:, 1:] = w[:, 1:] @ L1.T
        return np.einsum("um,fmd->fud", Q, z)


# ----------------------------------------------------------------- designs
def mode_basis(U):
    """U x U orthonormal: column 0 = 1/sqrt(U) (common mode), columns 1..U-1 the real-DFT complement
    (cos and sin of frequency k = 1..floor((U-1)/2), and (-1)^u for even U)."""
    u = np.arange(U)
    cols = [np.ones(U) / math.sqrt(U)]
    for k in range(1, (U - 1) // 2 + 1):
        cols += [np.cos(2 * math.pi * k * u / U) * math.sqrt(2.0 / U), np.sin(2 * math.pi * k * u / U) * math.sqrt(2.0 / U)]
    if U % 2 == 0 and U > 1:
        cols.append((-1.0) ** u / math.sqrt(U))
    Q = np.stack(cols, 1)
    assert np.allclose(Q.T @ Q, np.eye(U), atol=1e-12)
    return Q


def wfill(w, o, n, E):
    """exact water-filling: p_j = max(0, w_j t - o_j) with sum_j n_j p_j = E (n_j = multiplicity)."""
    p = np.zeros_like(w)
    m = (n > 0) & (w > 0)
    if E <= 0 or not m.any():
        return p
    idx = np.flatnonzero(m)
    tau = o[idx] / w[idx]
    od = np.argsort(tau, kind="stable")
    idx, tau = idx[od], tau[od]
    t = (E + np.cumsum(n[idx] * o[idx])) / np.cumsum(n[idx] * w[idx])
    k = np.flatnonzero(t > tau)[-1]
    p[idx] = np.maximum(0.0, w[idx] * t[k] - o[idx])
    assert abs((n * p).sum() - E) <= 1e-9 * E, ((n * p).sum(), E)
    return p


def finalize(name, Tx, Rot, valid, lam, E, rescale, extra):
    """per-user energy check (unit gain), optional exact per-user scaling, receiver observation matrix, overlap index."""
    U = Tx.shape[2]
    n_uses = int((np.abs(Tx) > 0).any(2).sum())                 # physical uses carrying a value of some user
    Eu = np.einsum("i,iru->u", lam, Tx ** 2)
    dev0 = float(np.max(np.abs(Eu / E - 1.0)))
    if rescale:
        Tx = Tx * np.sqrt(E / Eu)[None, None, :]
    A0 = Tx if Rot is None else np.einsum("irs,isu->iru", Rot, Tx)
    Eu1 = np.einsum("i,iru->u", lam, Tx ** 2)
    ii, rr = np.nonzero(valid)
    G = np.einsum("iru,irv->iuv", A0, A0)                       # unit-gain Gram matrix per coefficient
    dg = np.diagonal(G, 0, 1, 2)
    e = lam * dg.sum(1)
    with np.errstate(invalid="ignore", divide="ignore"):
        t = (G.sum((1, 2)) - dg.sum(1)) / max(U * (U - 1), 1) / dg.mean(1)
    t = np.where(e > 0, t, 0.0)
    info = {"energy_dev_before": dev0, "energy_dev": float(np.max(np.abs(Eu1 / E - 1.0))), "rescaled": bool(rescale),
            "energy_per_user": Eu1.tolist(), "n_obs": int(len(ii)), "n_uses": n_uses,
            "n_coef_active": int(valid.any(1).sum()),
            "t_overlap": float((e * t).sum() / e.sum()) if e.sum() > 0 else 0.0, **extra}
    return {"name": name, "Tx": Tx, "Rot": Rot, "A0": A0, "valid": valid, "ii": ii.astype(np.int64), "rr": rr,
            "A0obs": A0[ii, rr], "info": info}


def load_counts(U, n_data, rule="spec"):
    """coefficients sent by each user of the load masks (as shared_theory.load_counts). spec: min(floor(n_data/U), d_b)
    for every user; rank: user j sends min(ceil((N - j)/U) * 128, d_b), N = n_data / 128 slots (pca_eval.per_load)."""
    if rule == "rank":
        N = n_data // 128
        return [min(math.ceil((N - j) / U) * 128, D_B) for j in range(U)]
    assert rule == "spec", rule
    return [min(n_data // U, D_B)] * U


def design_load(lam, U, E, n_data, s2, rule="spec"):
    """load masks: user u sends its leading d_u coefficients with (12) at energy E; P, a are (d_b, U)."""
    du = load_counts(U, n_data, rule)
    P = np.zeros((D_B, U))
    for u, d in enumerate(du):
        P[:d, u] = wfill(np.sqrt(lam[:d] * s2), np.full(d, s2), np.ones(d), E)
    a = np.sqrt(P / lam[:, None])
    Tx = np.zeros((D_B, U, U))
    Tx[:, np.arange(U), np.arange(U)] = a
    valid = a > 0
    sch = finalize("load", Tx, None, valid, lam, E, False, {"d_u": du, "load_rule": rule,
                                                            "n_active_P": [int(x) for x in (P > 0).sum(0)]})
    sch.update(P=P, a=a)
    return sch


def prune_modes(lam, a0, a1, U, E_tot, n_data, s2, common_only=False):
    """water-filling over common and difference modes (multiplicities 1 and U-1); greedy pruning while more than n_data
    modes are active. Returns p0, p1, kept counts n0 (0/1), n1 (0..U-1), number of pruned modes."""
    w = np.sqrt(np.concatenate([lam, lam]) * s2)
    A2 = np.concatenate([a0, a1])
    L2 = np.concatenate([lam, lam])
    o = s2 / A2
    N = np.concatenate([np.ones(D_B), np.zeros(D_B) if common_only else np.full(D_B, U - 1.0)])
    pruned = 0
    while True:
        p = wfill(w, o, N, E_tot)
        act = (p > 0) & (N > 0)
        if (N * act).sum() <= n_data:
            break
        red = np.where(act, L2 * A2 - L2 * s2 / (p + o), np.inf)
        j = int(np.argmin(red))
        N[j] -= 1
        pruned += 1
    return p[:D_B], p[D_B:], N[:D_B], N[D_B:], pruned


def design_prop2(lam, kap, U, E, n_data, s2, kind="gen_joint", qseed=None):
    """kind gen_joint (Proposition 2, pruned when needed), gen_eq (common modes only), gen_k0 (kappa = 0, random Q)."""
    Q = mode_basis(U)
    if kind == "gen_k0":
        kap = np.zeros_like(kap)
    a0, a1 = 1.0 + (U - 1) * kap, 1.0 - kap
    p0, p1, n0, n1, pruned = prune_modes(lam, a0, a1, U, U * E, n_data, s2, common_only=(kind == "gen_eq"))
    Tx = np.zeros((D_B, U, U))
    Rot = np.repeat(np.eye(U)[None], D_B, 0)
    valid = np.zeros((D_B, U), bool)
    full = (n0 == 1) & (n1 == U - 1)
    sym_ok = U * D_B <= n_data                                     # U uses for every coefficient fit in the block
    partial, compact = 0, 0
    rq = np.random.default_rng(qseed) if kind == "gen_k0" else None
    for i in range(D_B):
        if kind == "gen_k0":
            Z = rq.standard_normal((U, U))
            Qr, Rr = np.linalg.qr(Z)
            Qr = Qr * np.sign(np.diag(Rr))                        # Haar orthogonal (drawn for every i)
            pk = p0[i] if n0[i] > 0 else p1[i]                     # kappa = 0: p0 = p1 = P_i of (12)
            m = int(n0[i] + n1[i]) if pk > 0 else 0
            Tx[i, :m] = math.sqrt(pk / lam[i]) * Qr[:m]
            valid[i, :m] = True
            partial += 0 < m < U
        elif full[i] and (sym_ok or p1[i] > 0):
            sp = np.sqrt(np.concatenate([[p0[i]], np.full(U - 1, p1[i])]) / lam[i])
            Tx[i] = (Q * sp) @ Q.T                                 # = lam^{-1/2}(sqrt(p1) I + (sqrt(p0) - sqrt(p1)) 11^T/U)
            Rot[i] = Q.T                                           # receiver: modal observations, inactive modes dropped
            valid[i] = sp > 0
        else:                                                      # modal form: one use per active kept mode
            rows = []
            if n0[i] > 0 and p0[i] > 0:
                rows.append(math.sqrt(p0[i] / lam[i]) * Q[:, 0])
            if p1[i] > 0:
                rows += [math.sqrt(p1[i] / lam[i]) * Q[:, 1 + (i + k) % (U - 1)] for k in range(int(n1[i]))]
            if rows:
                Tx[i, :len(rows)] = np.stack(rows)
                valid[i, :len(rows)] = True
            if full[i]:                                            # all modes kept, difference modes inactive (p1 = 0):
                compact += len(rows) > 0                           # common mode only, equal per-user energy p0/U
            else:
                partial += 0 < len(rows) and not (kind == "gen_eq")
    act_modes = int(((p0 > 0) * n0).sum() + ((p1 > 0) * n1).sum())
    with np.errstate(invalid="ignore", divide="ignore"):
        tm = np.where(p0 + (U - 1) * p1 > 0, (p0 - p1) / (p0 + (U - 1) * p1), 0.0)
    e = p0 + (U - 1) * p1
    extra = {"n_modes_active": act_modes, "n_pruned": pruned, "n_partial_coef": int(partial),
             "n_compact_coef": int(compact), "n_data": int(n_data),
             "t_overlap_modes": float((e * tm).sum() / e.sum()) if kind == "gen_joint" and e.sum() > 0 else None,
             "mse_design_equal_gain": float((lam * s2 / (p0 + s2 / a0) * (n0 > 0) + lam * a0 * (n0 == 0)).sum()
                                            + (lam * s2 / (p1 + s2 / a1) * n1 + lam * a1 * (U - 1 - n1)).sum())
             if kind != "gen_eq" else float((lam * s2 / (p0 + s2 / a0)).sum() + (U - 1) * (lam * a1).sum())}
    sch = finalize(kind, Tx, Rot if kind == "gen_joint" else None, valid, lam, E, partial > 0, extra)
    sch.update(p0=p0, p1=p1)
    return sch


# ----------------------------------------------------------------- receivers
def observe(sch, c, g, noise, s2):
    """y_i = Tx_i G c_i + n_i on the R rows of coefficient i, then the receiver rotation: (F, d_b, R)."""
    Tx = sch["Tx"]
    R = Tx.shape[1]
    y = np.einsum("iru,fiu->fir", Tx, (g[:, :, None] * c).transpose(0, 2, 1)) + math.sqrt(s2) * noise[:, :, :R]
    if sch["Rot"] is not None:
        y = np.einsum("irs,fis->fir", sch["Rot"], y)
    return y


def rx_joint(sch, gh, y, Dt, Ct, s2, lam_sum, trace):
    """joint LMMSE on the active observations: S = Psi Gamma Psi^T + s2 I, c_hat = Gamma Psi^T S^{-1} y.
    Psi is block diagonal over the coefficient index, so S = K_D o (A A^T) + K_C o (beta beta^T) + s2 I."""
    F, U = gh.shape
    ii = sch["ii"]
    n_obs = len(ii)
    const = bool(np.all(gh == gh[0]))
    chat = np.zeros((F, U, D_B))
    tr = np.full(F, U * lam_sum)
    if n_obs == 0:
        return chat, tr, 0.0
    iit = torch.from_numpy(ii)
    KD, KC = Dt[iit][:, iit], Ct[iit][:, iit]
    Dr, Cr = Dt[iit], Ct[iit]
    A0 = torch.from_numpy(np.ascontiguousarray(sch["A0obs"]))
    yo = torch.from_numpy(np.ascontiguousarray(y[:, ii, sch["rr"]]))
    ght = torch.from_numpy(gh)
    tsolve = 0.0
    for f in range(F):
        t0 = time.time()
        A = A0 * ght[f]
        beta = A.sum(1)
        S = (A @ A.T).mul_(KD).addcmul_(KC, torch.outer(beta, beta))
        S.diagonal().add_(s2)
        L = torch.linalg.cholesky(S)
        w = torch.cholesky_solve(yo[f][:, None], L)[:, 0]
        vt = torch.zeros(D_B, U, dtype=TD).index_add_(0, iit, A * w[:, None])
        chat[f] = ((Dt @ vt).T + (Ct @ vt.sum(1))[None]).numpy()
        tsolve += time.time() - t0
        if trace and f > 0 and const:
            tr[f] = tr[0]                                     # equal gains in every frame: the same trace
        elif trace:
            X = (A[:, :, None] * Dr[:, None, :] + beta[:, None, None] * Cr[:, None, :]).reshape(n_obs, U * D_B)
            Z = torch.linalg.solve_triangular(L, X, upper=False)
            tr[f] = U * lam_sum - float((Z ** 2).sum())
    return chat, tr, tsolve / F


def dense_check(sch, gh_f, y_f, chat_f, cont, s2):
    """frame 0: the structured solve against the dense covariance form and the dense precision form of the spec."""
    U = gh_f.shape[0]
    ii, A = sch["ii"], sch["A0obs"] * gh_f[None, :]
    n = U * D_B
    Psi = np.zeros((len(ii), n))
    for u in range(U):
        Psi[np.arange(len(ii)), u * D_B + ii] = A[:, u]
    G = np.kron(np.eye(U), cont.D) + np.kron(np.ones((U, U)), cont.C)
    yo = y_f[ii, sch["rr"]]
    c1 = G @ Psi.T @ np.linalg.solve(Psi @ G @ Psi.T + s2 * np.eye(len(ii)), yo)
    Gi = gamma_inv(cont, U)
    c2 = np.linalg.solve(Gi + Psi.T @ Psi / s2, Psi.T @ yo / s2)
    ref = chat_f.reshape(-1)
    nr = np.linalg.norm(ref) + 1e-300
    return float(np.linalg.norm(c1 - ref) / nr), float(np.linalg.norm(c2 - ref) / nr)


def gamma_inv(cont, U):
    """Gamma^{-1} = I (x) B1 + (1/U) 1 1^T (x) (B0 - B1), B1 = (Lam - C)^{-1}, B0 = (Lam + (U-1) C)^{-1} (user-major)."""
    def pinv(M):
        L = np.linalg.cholesky(M)
        Li = np.linalg.solve(L, np.eye(M.shape[0]))
        return Li.T @ Li
    B1 = pinv(cont.D)
    B0 = pinv(np.diag(cont.lam_m) + (U - 1) * cont.C)
    return np.kron(np.eye(U), B1) + np.kron(np.ones((U, U)), (B0 - B1) / U)


def rx_pu(sch, gh, y, lam, s2, g_true, lam_true, trace):
    """per-user Wiener with the receiver gain; unsent coefficients are zero. P, a: (d_b, U)."""
    P, a = sch["P"], sch["a"]
    U = gh.shape[1]
    z = y[:, :, :U]                                                   # row u = user u
    w = gh[:, None, :] * a[None] * lam[None, :, None] / (gh[:, None, :] ** 2 * P[None] + s2)
    w = np.where((a > 0)[None], w, 0.0)
    chat = (w * z).transpose(0, 2, 1)
    tr = None
    if trace:
        sent = (a > 0)[None]
        e = np.where(sent, lam_true[None, :, None] * (1 - w * g_true[:, None, :] * a[None]) ** 2 + w ** 2 * s2,
                     lam_true[None, :, None])
        tr = e.sum((1, 2))
    return chat, tr


def rx_pc(sch, gh, y, lam, kap, s2, g_true, Cdiag_true, lam_true, trace, chunk=100):
    """per-coefficient LMMSE with (A1): Gamma_i = lam_i ((1 - kappa_i) I + kappa_i 1 1^T); exact Gaussian MSE of this
    linear receiver under the true per-coefficient covariance (lam_i - C_ii) I + C_ii 1 1^T when trace is set."""
    F, U = gh.shape
    A0 = sch["A0"]
    R = A0.shape[1]
    I_U, J = np.eye(U), np.ones((U, U))
    Gi = lam[:, None, None] * ((1 - kap)[:, None, None] * I_U + kap[:, None, None] * J)
    Gt = (lam_true - Cdiag_true)[:, None, None] * I_U + Cdiag_true[:, None, None] * J
    chat = np.zeros((F, U, D_B))
    tr = np.zeros(F) if trace else None
    for s in range(0, F, chunk):
        sl = slice(s, min(F, s + chunk))
        A = A0[None] * gh[sl][:, None, None, :]                       # (Fc, d_b, R, U)
        GA = Gi[None] @ A.transpose(0, 1, 3, 2)                      # Gamma_i A^T  (Fc, d_b, U, R)
        S = A @ GA + s2 * np.eye(R)
        K = np.linalg.solve(S, GA.transpose(0, 1, 3, 2)).transpose(0, 1, 3, 2)   # (Fc, d_b, U, R)
        chat[sl] = np.einsum("fiur,fir->fui", K, y[sl])
        if trace:
            B = np.eye(U) - K @ (A0[None] * g_true[sl][:, None, None, :])
            tr[sl] = (np.einsum("fiuv,ivw,fiuw->f", B, Gt, B) + s2 * (K ** 2).sum((1, 2, 3)))
    return chat, tr


def rx_rnd(cont, U, n_data, a_amp, c, g, gh, noise, s2, Gi, trace, check):
    """dense random masks: y = sum_u g_u R_u diag(a) c_u + n; precision-form joint LMMSE per frame."""
    F = c.shape[0]
    R = []
    for u in range(U):
        Qu, Ru = np.linalg.qr(np.random.default_rng(31 + u).standard_normal((n_data, D_B)))
        R.append(Qu * np.sign(np.diag(Ru)))
    T = [[R[u].T @ R[v] for v in range(U)] for u in range(U)]
    nz = noise.reshape(F, -1)[:, :n_data]
    X = c * a_amp[None, None, :]
    RTy = np.zeros((F, U, D_B))
    for u in range(U):
        RTy[:, u] = sum((X[:, v] * g[:, v, None]) @ T[u][v].T for v in range(U)) + math.sqrt(s2) * nz @ R[u]
    TA = [[torch.from_numpy(a_amp[:, None] * T[u][v] * a_amp[None, :] / s2) for v in range(U)] for u in range(U)]
    Git = torch.from_numpy(Gi)
    n = U * D_B
    chat = np.zeros((F, U, D_B))
    tr = np.zeros(F) if trace else None
    tsolve, chk = 0.0, None
    for f in range(F):
        t0 = time.time()
        A = Git.clone()
        for u in range(U):
            for v in range(U):
                A[u * D_B:(u + 1) * D_B, v * D_B:(v + 1) * D_B] += float(gh[f, u] * gh[f, v]) * TA[u][v]
        L = torch.linalg.cholesky(A)
        rhs = torch.from_numpy((gh[f][:, None] * a_amp[None, :] * RTy[f] / s2).reshape(-1))
        chat[f] = torch.cholesky_solve(rhs[:, None], L)[:, 0].numpy().reshape(U, D_B)
        tsolve += time.time() - t0
        if trace:
            Z = torch.linalg.solve_triangular(L, torch.eye(n, dtype=TD), upper=False)
            tr[f] = float((Z ** 2).sum())
        if check and f == 0:
            Psi = np.concatenate([gh[0, u] * R[u] * a_amp[None, :] for u in range(U)], 1)
            y0 = sum(g[0, u] * R[u] @ X[0, u] for u in range(U)) + math.sqrt(s2) * nz[0]
            G = np.kron(np.eye(U), cont.D) + np.kron(np.ones((U, U)), cont.C)
            c1 = G @ Psi.T @ np.linalg.solve(Psi @ G @ Psi.T + s2 * np.eye(n_data), y0)
            chk = float(np.linalg.norm(c1 - chat[0].reshape(-1)) / np.linalg.norm(c1))
    return chat, tr, tsolve / F, chk


# ----------------------------------------------------------------- main
def cos_rows(c, ch):
    return (c * ch).sum(-1) / (np.linalg.norm(c, axis=-1) * np.linalg.norm(ch, axis=-1) + 1e-12)


def tag_of(a):
    t = a.dataset + (f"_mix{a.mix:g}" if a.mix is not None else "") + f"_K{a.K}_U{a.U}"
    if a.control:
        t += "_ctrl" if a.control_c == "zero" else "_ctrlfit"
    if a.nearfar:
        t += f"_nf{a.nearfar:g}"
    if a.phase:
        t += f"_ph{a.phase:g}{a.rxgain}"
    if a.pilot:
        t += "_pilot" + (f"E{a.pilot_energy:g}" if a.pilot_energy != 1.0 else "")
    if a.c_from != "self":
        t += "_cf" + ("other" if a.c_from.startswith("other") else a.c_from)
    if a.load_rule != "spec":
        t += "_" + a.load_rule
    if a.gains != "rayleigh":
        t += "_geq"
    if a.a1:
        t += "_a1"
    if a.gauss:
        t += "_gauss"
    return t + a.suffix


def paired_ratio(err_f, tr_f):
    """mean_f err_f / mean_f tr_f with the paired standard error sd(err_f - tr_f) / sqrt(F) / mean(tr_f)."""
    m = float(tr_f.mean())
    r = float(err_f.mean()) / m
    se = float((err_f - tr_f).std(ddof=1) / math.sqrt(len(tr_f)) / m) if len(tr_f) > 1 else float("nan")
    return r, se


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["coco", "flickr", "agnews"], default="coco")
    ap.add_argument("--mix", type=float, default=None)
    ap.add_argument("--U", type=int, required=True)
    ap.add_argument("--K", type=int, choices=[4, 2], default=4)
    ap.add_argument("--snrs", type=int, nargs="+", default=list(SNRS))
    ap.add_argument("--frames", type=int, default=None, help="default: every test image (coco 2000, flickr 1000)")
    ap.add_argument("--nearfar", type=float, default=0.0)
    ap.add_argument("--gains", choices=["rayleigh", "equal"], default="rayleigh", help="|h_u|: Rayleigh or 1")
    ap.add_argument("--phase", type=float, default=0.0)
    ap.add_argument("--rxgain", choices=["abs", "eff"], default="abs")
    ap.add_argument("--pilot", action="store_true", help="pilot-based gains for the overlapping members (gen_*, rnd)")
    ap.add_argument("--pilot_energy", type=float, default=1.0, help="energy Ep of the one-use pilot (taken from d_s/U)")
    ap.add_argument("--c_from", default="self", help="self | other | N (first N fit images)")
    ap.add_argument("--control", action="store_true")
    ap.add_argument("--control_c", choices=["zero", "fit"], default="zero")
    ap.add_argument("--load_rule", choices=["spec", "rank"], default="spec", help="coefficients per user of the load masks")
    ap.add_argument("--gauss", action="store_true")
    ap.add_argument("--a1", action="store_true", help="with --gauss: content model C = diag(kappa lam) (A1)")
    ap.add_argument("--modes", nargs="+", default=list(MODES))
    ap.add_argument("--check", action="store_true", help="dense-solve check of the joint receivers on frame 0")
    ap.add_argument("--theory", action="store_true", help="exact Gaussian NMSE trace per frame (always on with --gauss)")
    ap.add_argument("--outdir", default=os.path.join(DATA, "shared"))
    ap.add_argument("--suffix", default="")
    a = ap.parse_args()
    assert not (a.mix is not None and (a.control or a.dataset == "agnews")), "--mix needs caption content"
    assert a.c_from in ("self", "other", "other_corpus") or a.c_from.isdigit()
    assert not a.a1 or (a.gauss and a.c_from == "self" and not a.control and a.dataset != "agnews"), \
        "--a1 is a Gaussian-model check of shared content (--gauss, own C)"
    assert a.pilot_energy > 0
    torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "2")))
    T0 = time.time()
    U, K = a.U, a.K
    cont = Content(a)
    F = a.frames or cont.n_test
    d_s = K * D_B
    n_load, E_load = d_s, d_s / U                                    # load masks: never charged a pilot
    n_gen = d_s - (U if a.pilot else 0)                              # overlapping members: one pilot use per user
    E_gen = d_s / U - (a.pilot_energy if a.pilot else 0.0)
    trace = a.gauss or a.theory
    lam = cont.lam_m
    lam_sum_true = float(cont.lam_true.sum())
    Dt, Ct = torch.from_numpy(cont.D), torch.from_numpy(cont.C)
    Q = mode_basis(U)
    modes = [m for m in a.modes if not (m == "rnd" and U > K)]
    assert all(m in MODES + ("gen_joint_true", "rnd") for m in modes), modes
    Gi = gamma_inv(cont, U) if "rnd" in modes else None
    print(f"{tag_of(a)}: F={F} n_data load/gen={n_load}/{n_gen} E load/gen={E_load:.1f}/{E_gen:.1f} "
          f"kappa_bar={cont.info['kappa_bar']:.4f} trC={cont.info['trC_model']:.4f} modes={modes}", flush=True)
    res, design, arrays = {m: {} for m in modes}, {}, {}
    for snr in a.snrs:
        s2 = 10 ** (-snr / 10.0)
        off = 1000 * U
        rng_c = np.random.default_rng(123 + snr + off)
        c = None if a.gauss else cont.frames(rng_c, F, U)
        # per frame and user: Re h, Im h, offset (normal CDF -> uniform), phase, pilot noise (prefix-stable in F)
        Z = np.random.default_rng(5123 + snr + off).standard_normal((F, U, 5))
        h = np.abs(Z[..., 0] + 1j * Z[..., 1]) / math.sqrt(2) if a.gains == "rayleigh" else np.ones((F, U))
        offs = 10 ** ((2 * ndtr(Z[..., 2]) - 1) * a.nearfar / 20.0)
        phi = np.radians(a.phase) * Z[..., 3]
        g_abs = h * offs
        g = g_abs * np.cos(phi)                                      # effective (true) gain
        gh_load = g if a.pilot else (g_abs if a.rxgain == "abs" else g)
        gh_gen = g + math.sqrt(s2 / a.pilot_energy) * Z[..., 4] if a.pilot else gh_load
        noise = np.random.default_rng(9123 + snr + off).standard_normal((F, D_B, max(R_NOISE, U, K)))
        if a.gauss:
            c = cont.gauss(np.random.default_rng(7123 + snr + off), F, U, Q)
        nrm = (c ** 2).sum(-1)
        arrays[f"nrm_{snr}"] = nrm.astype(np.float32)
        arrays[f"g_{snr}"] = g.astype(np.float32)
        arrays[f"gh_{snr}"] = gh_gen.astype(np.float32)              # receiver gain of the overlapping members
        arrays[f"ghload_{snr}"] = gh_load.astype(np.float32)         # receiver gain of the load masks
        sch = {}
        if any(m.startswith("load") for m in modes):
            sch["load"] = design_load(lam, U, E_load, n_load, s2, a.load_rule)
        if "rnd" in modes:
            sch["load_rnd"] = design_load(lam, U, E_gen, n_gen, s2)    # (12) at the energy and uses of rnd
        if any(m in modes for m in ("gen_joint", "gen_pc", "gen_joint_true")):
            sch["gen_joint"] = design_prop2(lam, cont.kap, U, E_gen, n_gen, s2, "gen_joint")
        if "gen_eq" in modes:
            sch["gen_eq"] = design_prop2(lam, cont.kap, U, E_gen, n_gen, s2, "gen_eq")
        if "gen_k0" in modes:
            sch["gen_k0"] = design_prop2(lam, cont.kap, U, E_gen, n_gen, s2, "gen_k0", qseed=4242 + U)
        design[snr] = {k: v["info"] for k, v in sch.items() if k != "load_rnd"}
        ys = {k: observe(v, c, g, noise, s2) for k, v in sch.items() if k != "load_rnd"}
        for m in modes:
            t0 = time.time()
            tr, tsolve, chk = None, None, None
            gh = gh_gen if m in OVERLAP else (g if m == "gen_joint_true" else gh_load)
            exact = bool(np.array_equal(gh, g))
            theory_joint = trace and exact and cont.model_true       # trace[(Gamma^-1 + M)^-1] is the MSE only then
            if m == "load_pu":
                ch, tr = rx_pu(sch["load"], gh, ys["load"], lam, s2, g, cont.lam_true, trace)
                tsolve = (time.time() - t0) / F
            elif m in JOINT:
                key = "load" if m == "load_joint" else ("gen_joint" if m == "gen_joint_true" else m)
                ch, tr, tsolve = rx_joint(sch[key], gh, ys[key], Dt, Ct, s2, float(lam.sum()), theory_joint)
                if a.check:
                    chk = dense_check(sch[key], gh[0], ys[key][0], ch[0], cont, s2)
            elif m == "gen_pc":
                ch, tr = rx_pc(sch["gen_joint"], gh, ys["gen_joint"], lam, cont.kap, s2, g, np.diag(cont.C_true),
                               cont.lam_true, trace)
                tsolve = (time.time() - t0) / F
            elif m == "rnd":
                ld = sch["load_rnd"]
                ch, tr, tsolve, chk = rx_rnd(cont, U, n_gen, ld["a"][:, 0], c, g, gh, noise[:, :, :K], s2, Gi,
                                             theory_joint, a.check)
            else:
                raise ValueError(m)
            err = ((c - ch) ** 2).sum(-1)
            cz = cos_rows(c, ch)
            out = {"cos": float(cz.mean()), "weakest": float(cz.min(1).mean()), "nmse": float(err.sum() / nrm.sum()),
                   "rx_gain": "pilot" if gh is gh_gen and a.pilot else ("true" if exact else a.rxgain),
                   "sec_per_frame_rx": float(tsolve), "sec_total": float(time.time() - t0)}
            if tr is not None and (m in ("load_pu", "gen_pc") or theory_joint):
                out["nmse_theory"] = float(tr.mean() / (U * lam_sum_true))
                r, se = paired_ratio(err.sum(1), tr)
                out.update({"mc_over_theory": r, "mc_over_theory_se": se, "mc_over_theory_z": (r - 1) / se})
            key = "load" if m.startswith("load") else ("gen_joint" if m in ("gen_pc", "gen_joint_true") else m)
            dsg = design[snr].get(key, {})
            if a.a1 and a.gains == "equal" and exact and dsg.get("mse_design_equal_gain") is not None \
                    and m in ("gen_joint", "gen_eq") and dsg.get("n_partial_coef", 1) == 0:
                cf = dsg["mse_design_equal_gain"]                 # Proposition 2 closed form (A1, unit gains)
                e = err.sum(1)
                out.update({"closed_form_nmse": cf / (U * lam_sum_true), "mc_over_closed_form": float(e.mean() / cf),
                            "mc_over_closed_form_se": float(e.std(ddof=1) / math.sqrt(F) / cf)})
                if tr is not None and "nmse_theory" in out:
                    out["closed_form_over_trace"] = float(cf / tr[0])
            if chk is not None:
                out["check_rel"] = chk
            res[m][snr] = out
            arrays[f"cos_{m}_{snr}"] = cz.astype(np.float32)
            arrays[f"err_{m}_{snr}"] = err.astype(np.float32)
            if "nmse_theory" in out:
                arrays[f"tr_{m}_{snr}"] = tr                          # per-frame exact Gaussian MSE (float64)
            msg = ""
            if "nmse_theory" in out:
                msg += (f" theory={out['nmse_theory']:.5f} ratio={out['mc_over_theory']:.4f}"
                        f"+-{out['mc_over_theory_se']:.4f}")
            if "mc_over_closed_form" in out:
                msg += (f" closed={out['closed_form_nmse']:.5f} mc/closed={out['mc_over_closed_form']:.4f}"
                        f"+-{out['mc_over_closed_form_se']:.4f}")
                if "closed_form_over_trace" in out:
                    msg += f" closed/trace-1={out['closed_form_over_trace'] - 1:.1e}"
            print(f"{tag_of(a)} snr={snr} {m:14s} cos={out['cos']:.4f} weakest={out['weakest']:.4f} "
                  f"nmse={out['nmse']:.5f}" + msg + f" rx={tsolve * 1e3:.1f}ms/frame"
                  + (f" check={chk}" if chk is not None else ""), flush=True)
        for k, v in design[snr].items():
            print(f"   design {k}: n_obs={v['n_obs']} n_uses={v['n_uses']} energy_dev_before={v['energy_dev_before']:.2e} "
                  f"energy_dev={v['energy_dev']:.2e} t={v['t_overlap']:.4f}"
                  + (f" pruned={v['n_pruned']} partial={v['n_partial_coef']} compact={v['n_compact_coef']}"
                     if "n_pruned" in v else ""), flush=True)
    os.makedirs(a.outdir, exist_ok=True)
    tag = tag_of(a)
    R = {"args": vars(a), "frames": F, "n_data": {"load": n_load, "gen": n_gen},
         "energy_per_user": {"load": E_load, "gen": E_gen},
         "pilot": {"charged_to": list(OVERLAP), "pilot_energy": a.pilot_energy, "gain_error_var": "s2 / pilot_energy",
                   "load_masks": "not charged, effective gain"} if a.pilot else None,
         "fit": cont.info, "kappa": cont.kap.tolist(), "design": design, "res": res, "runtime_s": time.time() - T0}
    json.dump(R, open(os.path.join(a.outdir, f"shared_{tag}.json"), "w"), indent=1)
    np.savez_compressed(os.path.join(a.outdir, f"shared_{tag}.npz"), **arrays)
    print(f"wrote shared_{tag}.json ({time.time() - T0:.0f} s)", flush=True)


if __name__ == "__main__":
    main()
