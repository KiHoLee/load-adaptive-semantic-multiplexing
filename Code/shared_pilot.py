# =========================================================
# shared_pilot.py — receiver gain estimation for the overlapping (Proposition 2) masks and the load masks,
# 2026-11-16 (rows of (8), transmit normalization of every
# user, energy-matched EM, converged EM with a line search, bounded M-step, per-iteration CosSim / NMSE, paired CIs
# of NMSE and gain error, tag and code identity).
# Why: shared_masks.py --pilot gives the overlapping members one real pilot use per user and uses the raw pilot
# estimate as the known gain, while the load masks get the exact gain. Here every receiver estimates its gains
# (fair setting), and the overlapping members estimate theirs from the whole frame by EM, started from the pilot.
# Model: shared_masks.py (imported as sm): sm.Content (lam, V, C, Gamma = I_U (x) (Lam - C) + 1 1^T (x) C), the frames,
# captions, gains, phases, pilot noise Z[..., 4] and noise rows are the same draws as shared_masks for the same
# (dataset, U, K, SNR, frames): captions default_rng(123 + SNR + 1000 U) (sm.Content.frames), gains
# default_rng(5123 + SNR + 1000 U) with one (U, 5) block of normals per frame (Re h, Im h, offset, phase, pilot
# noise), noise default_rng(9123 + SNR + 1000 U) with 5 rows per coefficient, --gauss draws
# default_rng(7123 + SNR + 1000 U) (sm.Content.gauss), all frame by frame (prefix-stable in the frame count).
# Gains: Rayleigh |h_u| (--gains equal: 1), --nearfar X, --phase DEG (transmitter pre-compensates the phase, residual
# phi ~ N(0, DEG^2)); the true (effective) gain is g = |h| offs cos(phi). Noise s2 = 10^(-SNR/10) per real use,
# d_s = K d_b uses.
# Members and their resources:
#   load    load masks (sm.design_load, spec rule): d_s uses, d_s/U per user, never charged a pilot.
#   gen     Proposition 2 masks (sm.design_prop2 gen_joint), always pilot-charged: n_gen = d_s - U data uses,
#           E_gen = d_s/U - Ep, one real pilot use per user y_p,u = sqrt(Ep) g_u + sqrt(s2) Z[f, u, 4].
#   gen0    Proposition 2 masks without pilot (d_s uses, d_s/U), as shared_masks without --pilot.
# Transmit normalization (eq. (5) of the manuscript, modes *_blind and *_rho): user u scales its unit-gain data
# values x_u = sum_i Tx_i[:, u] c_{u,i} to exactly the member's data energy E per frame, rho_u = sqrt(E / ||x_u||^2)
# (the same rule for every member; the pilot use is not scaled); the receiver treats rho_u as 1, so its estimate of
# rho_u c_u is compared with c_u (CosSim is scale-free; nmse includes the rho scale error, nmse_rho is measured
# against rho_u c_u). Without the suffix _rho (and outside *_blind) the users transmit unnormalized, as in shared_masks.
# Modes (all on the same frames, paired):
#   load_pu, load_joint     load masks, true g at the receiver (sm.rx_pu, sm.rx_joint), = shared_masks.
#   gen_joint_true          gen masks, true g (= shared_masks --pilot --pilot_energy Ep gen_joint_true).
#   gen_joint_true0         gen0 masks, true g (= shared_masks gen_joint without --pilot; genie reference of em0).
#   load_pu_rho,            load masks with the transmit normalization, true g at the receiver (genie references of
#   load_joint_rho          load_*_blind and load_joint_em_rho: the normalization changes each user's received energy
#                           (rho 0.78-1.32 at 10 dB) and the NMSE (the receiver treats rho as 1), so load_joint is not the
#                           genie of the normalized system; CosSim differs by < 0.0002 on 200 frames at 0 and 10 dB).
#   gen_joint_true_rho,     gen / gen0 masks with the transmit normalization, true g at the receiver (genie
#   gen_joint_true0_rho     references of gen_joint_em_rho / gen_joint_em0_rho).
#   load_pu_blind,          the manuscript's receiver: load masks with the transmit normalization; the receiver
#   load_joint_blind        estimates g_u^2 by (8), ghat_u^2 = max((||z_u||^2 - d_u s2) / E_load, 1e-6), z_u the d_u
#                           received (combined) entries of user u, i = 1..d_u, including the entries with P_i = 0 that
#                           carry only noise (as the manuscript and pca_eval.user), and uses ghat in the Wiener factor
#                           (load_pu_blind, sm.rx_pu) or in the joint LMMSE (load_joint_blind, sm.rx_joint).
#   load_pu_blindA,         variant of (8) on the n_u entries with P_i > 0 only (not the manuscript; it improves on it
#   load_joint_blindA       at low SNR); optional modes, not in the default --modes.
#   gen_joint_pilot         gen masks with the raw pilot estimate max(y_p / sqrt(Ep), g_min) (= shared_masks
#                           --pilot gen_joint except where the clamp g_min acts).
#   gen_joint_em            gen masks, Gaussian-ML gain: EM started from the clamped pilot estimate, pilot term in the
#                           M-step.
#   gen_joint_em0           gen0 masks (no pilot), Gaussian-ML gain: EM started from the common blind gain
#                           ghat_u = sqrt(max((||y_obs||^2 - n_obs s2) / e1, 1e-6)), e1 = sum_i tr(A_i Gamma_i A_i^T) the
#                           model received signal energy at unit gains (all n_obs active observations).
#   load_joint_em           load masks (unnormalized), Gaussian-ML gain: EM started from (8) computed without rho.
#   gen_joint_em_rho,       energy-matched EM (below) with the transmit normalization: gen masks (pilot start and pilot
#   gen_joint_em0_rho,      term), gen0 masks (common blind start) and load masks (started from (8) = the estimate of
#   load_joint_em_rho       load_joint_blind); the like-for-like pair is gen_joint_em(0)_rho vs load_joint_em_rho, and
#                           gen_joint_em(0) vs load_joint_em.
# EM (g deterministic, c Gaussian with the model Gamma; observations of sm.rx_joint: y_r = sum_u A0obs[r, u] g_u
# c_u[i_r] + n_r, n_r ~ N(0, s2) i.i.d., the receiver rotation is orthonormal). Iteration t (--em_iters, default 3):
#   E-step: chat = joint LMMSE with gh = ghat^(t) (the operations of sm.rx_joint, plus the observed-data log-likelihood
#     log N(y_obs; 0, S) from its Cholesky factor, + the pilot term); per-coefficient posterior covariance Sigma_i:
#     --em_cov a1 (default): Sigma_i = (Gamma_i^{-1} + G A_i^T A_i G / s2)^{-1}, Gamma_i = (lam_i - C_ii) I + C_ii 1 1^T,
#       A_i the A0 rows of coefficient i (inactive rows zero), G = diag(ghat), computed in the covariance form
#       Gamma_i - Gamma_i Psi_i^T (Psi_i Gamma_i Psi_i^T + s2 I)^{-1} Psi_i Gamma_i, Psi_i = A_i G;
#     --em_cov exact: the U x U diagonal blocks of Gamma - Gamma Psi^T S^{-1} Psi Gamma (Z = L^{-1} X as in the trace
#       path of sm.rx_joint); the a1 term is computed next to it and its relative difference stored;
#     --em_cov auto: exact at SNR <= --em_auto_snr (default 0 dB), a1 above. Smoke (U4 K4 captions, Rayleigh, 8 frames):
#       the a1 fixed point lies 2-7% (relative RMSE) from the exact one at -5 and 0 dB and about 1% at 10 dB for the
#       Gaussian-ML modes, 2% and 0.2% for the energy-matched modes; the exact blocks cost about 0.1 s per iteration at
#       0 dB, little at -5 dB, and about 0.6 s at 10-20 dB (where a1 is kept).
#   M-step (maximizer of E[log p(y, y_p, c; g)]): R_uv = sum_r A0obs[r,u] A0obs[r,v] (chat_u chat_v + Sigma_{i_r}[u,v])
#     + Ep [u = v], b_u = sum_r A0obs[r,u] chat_u[i_r] y_r + sqrt(Ep) y_p,u, ghat^(t+1) = argmin_{g >= gm}
#     (g^T R g / 2 - b^T g) (bounded M-step, exact over the active sets; = R^{-1} b where that is >= gm), gm =
#     --em_g_min (default 0.05 = the g_min of --g_min; also the lower bound of the EM start). Smoke (U4 K4 captions, Rayleigh,
#     200 frames): gm = 1e-3 (the floor sqrt(1e-6) of (8)) raised the mean relative gain RMSE of every EM mode at 0 dB
#     by 0.004-0.006 and lowered its CosSim by 0.0004 (deep fades then follow the noise), and changed 10 dB by
#     <= 0.0002, so the default stays 0.05;
#     em0 and load_joint_em: Ep = 0. A frame stops when max_u |ghat^(t+1) - ghat^(t)| / ghat^(t) < --em_tol (1e-4).
#   The EM step is ghat + s2 R^{-1} score, score = (b - R ghat) / s2 the observed-data score (Fisher identity), a
#   gradient step with the complete-data information R / s2; at high SNR the observed information is a small part of
#   it, so plain EM moves slowly and stops far from its fixed point after a few iterations. --em_step fisher: Fisher
#   scoring ghat + I^{-1} score with the expected Fisher information I of the per-coefficient (A1) model (+ Ep / s2 I),
#   bounded to [max(ghat/2, gm), 2 ghat] per user (bounded quadratic step), then a line search on the exact
#   observed-data log-likelihood ll (from the Cholesky factor of the E-step): with the slope d0 = score^T step and ll at
#   the full step, the quadratic fit gives the step length al*; the full step is kept if it increases ll and al* >= 0.75
#   (or if it is below --em_tol), otherwise al = clip(al*, 0.1, 0.75) is tried, then halved, --em_ls (2) trials in all
#   while ll decreases (the full step is kept when it increased ll more), and a frame without an ascent step stops
#   (this removes the 0 dB two-cycle and the zigzag of the scoring step; --em_ls 0: every scoring step is taken; with a1
#   at 10-20 dB most frames stop this way after 3-5 steps). The same fixed point R g = b (output tag _fs):
#   the Gaussian-model ML of g with --em_cov exact (ll is monotone); with a1 the step direction is approximate, so a
#   frame stops where it no longer increases the exact ll, within the a1-vs-exact gap of the ML. It does not use the
#   known transmit energy.
#   Energy-matched EM (modes *_em_rho, any --em_step): each user's data energy is known exactly (rho), and for the
#   true content sum_r A0obs[r,u]^2 c_u[i_r]^2 = ||x_u||^2 = E (A0 = Rot Tx, Rot orthonormal). The E-step moments are
#   scaled to it: r_u = sqrt(E / R_uu), R <- diag(r) R diag(r), b <- r o b (the posterior of each user's content
#   rescaled to the known energy), then the pilot term and the bounded M-step as above. For the load masks
#   (R diagonal) the update is ghat <- b_u / sqrt(E R_uu), which at high SNR is sqrt(||z_u||^2 / E) whatever the start
#   (the analogue of (8)); with overlap the cross moments R_uv carry the separation of the users. There is no
#   likelihood to search on; at low SNR the plain step contracts slowly (rate -> 1), so the step in log g is
#   extrapolated, log ghat <- log ghat + om d, d = log T(ghat) - log ghat the plain step, om = min(1 / (1 - q), --emr_omax
#   (8)) with q = ||d|| / ||d_prev|| when the plain steps contract (q < 0.95) without a sign flip (d^T d_prev > 0), om = 1
#   otherwise (the same fixed point; --emr_plain: om = 1 always).
#   Per iteration t = 0..T: ghat^(t), relative gain RMSE, and CosSim / squared error of chat(ghat^(t)) (npz per frame,
#   JSON means), so the result after any number of iterations up to T is in the same run; final chat = chat(ghat^(T)).
# Metrics: per-user CosSim(c_u, chat_u) in the PCA basis (= CosSim(b_u, V chat_u), V orthonormal), weakest user per
# frame, NMSE = sum ||c - chat||^2 / sum ||c||^2, relative gain RMSE per frame sqrt(mean_u ((ghat - g) / g)^2) against
# the true g (gain_rel_rmse = its mean over frames), seconds per frame of every mode; paired bootstrap 95% CIs over
# frames (1000 resamples, default_rng(555 + SNR), the same resamples for every mode and metric) of the per-frame
# differences to the references present in the run (load_joint_blind, load_joint, load_joint_rho, load_joint_em,
# load_joint_em_rho):
# user-mean CosSim (d_vs_<ref>), per-frame NMSE err_f / nrm_f (dnmse_vs_<ref>) and relative gain RMSE
# (dgrmse_vs_<ref>, modes and references with estimated gains). The Ep-independent modes give the same numbers frame
# by frame for every Ep (same draws), so files of different Ep of one point can be paired frame by frame.
# Consistency (--check_pilot / --check_nopilot: npz of shared_masks.py on the same point with / without --pilot):
# frame-by-frame max |cos| and relative err differences of load_pu, load_joint, gen_joint_true (vs gen_joint_true),
# gen_joint_pilot (vs gen_joint, frames where the clamp acts excluded) and gen_joint_true0 (vs gen_joint, no pilot);
# printed and stored in the JSON (consistency).
# Output: ../data/shared/pilot/shared_pilot_<tag>.json and .npz (per-frame per-user cos and err of every mode, nrm, g,
# rho per member, raw pilot estimate, every ghat, EM per-iteration ghat, gain RMSE, CosSim, squared error, iterations
# and joint solves), JSON code = md5 of shared_pilot.py and shared_masks.py, tag =
# <dataset>[_mix<k>]_K<K>_U<U>[_nf<X>][_ph<deg>]_E<Ep>[_geq][_gauss][_emexact|_emauto<snr>][_fs][_it<T>][_tol<x>]
# [_gmin<x>][_emgmin<x>][_ls<n>][_emrplain|_om<x>]<suffix> (_it, _tol, _gmin, _emgmin, _ls, _om only when they differ
# from 3, 1e-4, 0.05, 0.05, 2, 8).
# Usage: python shared_pilot.py --dataset coco --U 4 --K 4 --pilot_energy 1 --snrs -5 0 10 20 --em_step fisher \
#               --em_iters 30 --em_cov auto                (the options of pilot_queue.sh, tag ..._E1_emauto0_fs_it30)
#        python shared_pilot.py --dataset coco --U 4 --K 4 --snrs 0 10 --frames 200 --outdir ../data/shared/pilot_smoke \
#               --check_pilot ../data/shared/pilot_smoke/shared_coco_K4_U4_pilot_chk.npz
#        python shared_pilot.py --dataset coco --U 4 --K 4 --gauss --snrs -5 0 --frames 20 --em_cov exact \
#               --modes gen_joint_true gen_joint_em gen_joint_em0 --outdir ../data/shared/pilot_smoke
# =========================================================
import argparse, hashlib, itertools, json, math, os, time
import numpy as np
import torch
from scipy.special import ndtr
import shared_masks as sm

D_B, R_NOISE = sm.D_B, sm.R_NOISE
SNRS = (-5, 0, 10, 20)
MODES = ("load_pu", "load_joint", "load_pu_rho", "load_joint_rho", "load_pu_blind", "load_joint_blind",
         "gen_joint_true", "gen_joint_true_rho", "gen_joint_pilot", "gen_joint_em", "gen_joint_em0", "load_joint_em",
         "gen_joint_em_rho", "gen_joint_em0_rho", "load_joint_em_rho")
ALL = MODES + ("gen_joint_true0", "gen_joint_true0_rho", "load_pu_blindA", "load_joint_blindA")
SCHEME = {"load_pu": "load", "load_joint": "load", "load_pu_rho": "load", "load_joint_rho": "load",
          "load_pu_blind": "load", "load_joint_blind": "load", "load_pu_blindA": "load", "load_joint_blindA": "load",
          "load_joint_em": "load", "load_joint_em_rho": "load",
          "gen_joint_true": "gen", "gen_joint_true_rho": "gen", "gen_joint_pilot": "gen", "gen_joint_em": "gen",
          "gen_joint_em_rho": "gen", "gen_joint_true0": "gen0", "gen_joint_true0_rho": "gen0", "gen_joint_em0": "gen0",
          "gen_joint_em0_rho": "gen0"}
RHO = ("load_pu_rho", "load_joint_rho", "load_pu_blind", "load_joint_blind", "load_pu_blindA", "load_joint_blindA",
       "gen_joint_true_rho", "gen_joint_true0_rho", "gen_joint_em_rho", "gen_joint_em0_rho",
       "load_joint_em_rho")                                                                     # transmit normalization
TRUE = ("load_joint", "load_joint_rho", "gen_joint_true", "gen_joint_true0", "gen_joint_true_rho", "gen_joint_true0_rho")
EM = ("gen_joint_em", "gen_joint_em0", "load_joint_em", "gen_joint_em_rho", "gen_joint_em0_rho", "load_joint_em_rho")
EMR = ("gen_joint_em_rho", "gen_joint_em0_rho", "load_joint_em_rho")                          # energy-matched
REFS = ("load_joint_blind", "load_joint", "load_joint_rho", "load_joint_em", "load_joint_em_rho")   # bootstrap refs
TD = sm.TD
DEF = {"em_iters": 3, "em_tol": 1e-4, "g_min": 0.05, "em_g_min": 0.05, "em_ls": 2, "emr_omax": 8.0}   # tag defaults


# ----------------------------------------------------------------- gain estimates
def rel_rmse(gh, g):
    """per frame sqrt(mean_u ((ghat - g) / g)^2); gh (..., U) broadcast against g (F, U) or (F, 1, U)."""
    return np.sqrt((((gh - g) / g) ** 2).mean(-1))


def tx_rho(sch, c, E):
    """rho_u = sqrt(E / ||x_u||^2), x_u = sum_i Tx_i[:, u] c_{u,i} the unit-gain data values of user u (eq. (5)): (F, U).
    Load masks: ||x_u||^2 = sum_i a_iu^2 c_ui^2 (copies combined as one use)."""
    e = (sch["Tx"] ** 2).sum(1)                                       # (d_b, U)
    return np.sqrt(E / (e.T[None] * c ** 2).sum(-1))


def blind_load(sch, y, s2, E, rows="du"):
    """(8): ghat_u^2 = max((||z_u||^2 - n_u s2) / E, 1e-6), z_u = row u of the load observations. rows du (the
    manuscript, pca_eval.user): the d_u received entries i = 1..d_u, including P_i = 0 (noise only), n_u = d_u;
    rows active (variant): the entries with P_i > 0 only."""
    U = sch["a"].shape[1]
    if rows == "du":
        v = np.arange(D_B)[:, None] < np.array(sch["info"]["d_u"])[None]
    else:
        v = sch["valid"]                                              # (d_b, U): user u sends coefficient i
    z2 = (y[:, :, :U] ** 2 * v[None]).sum(1)
    return np.sqrt(np.maximum((z2 - v.sum(0)[None] * s2) / E, 1e-6))


def blind_common(sch, y, s2, M, Gb):
    """common blind gain of em0: sqrt(max((||y_obs||^2 - n_obs s2) / e1, 1e-6)) for every user, e1 = sum_i
    tr(A_i Gamma_i A_i^T) = sum_i sum_uv M_i[u, v] Gamma_i[u, v] the model signal energy at unit gains."""
    U = M.shape[1]
    e1 = float(np.einsum("iuv,iuv->", M, Gb))
    yo = y[:, sch["ii"], sch["rr"]]
    e = np.sqrt(np.maximum(((yo ** 2).sum(1) - len(sch["ii"]) * s2) / e1, 1e-6))
    return np.repeat(e[:, None], U, 1)


# ----------------------------------------------------------------- EM
def gamma_blocks(cont, U):
    """Gamma_i = (lam_i - C_ii) I + C_ii 1 1^T (model): (d_b, U, U)."""
    Cd = np.diag(cont.C)
    return (cont.lam_m - Cd)[:, None, None] * np.eye(U) + Cd[:, None, None] * np.ones((U, U))


def joint_post(sch, gh, y, Dt, Ct, s2):
    """the joint LMMSE of sm.rx_joint (the same operations, so the same chat) and the observed-data log-likelihood
    log N(y_obs; 0, S) per frame without the constant, -sum log diag L - y^T S^{-1} y / 2: chat (F, U, d_b), ll (F)."""
    F, U = gh.shape
    ii = sch["ii"]
    chat = np.zeros((F, U, D_B))
    ll = np.zeros(F)
    if len(ii) == 0:
        return chat, ll
    if "_kk" not in sch:                                              # cached per design
        iit = torch.from_numpy(ii)
        sch["_kk"] = (iit, Dt[iit][:, iit], Ct[iit][:, iit], torch.from_numpy(np.ascontiguousarray(sch["A0obs"])))
    iit, KD, KC, A0 = sch["_kk"]
    yo = torch.from_numpy(np.ascontiguousarray(y[:, ii, sch["rr"]]))
    ght = torch.from_numpy(np.ascontiguousarray(gh))
    for f in range(F):
        A = A0 * ght[f]
        beta = A.sum(1)
        S = (A @ A.T).mul_(KD).addcmul_(KC, torch.outer(beta, beta))
        S.diagonal().add_(s2)
        L = torch.linalg.cholesky(S)
        w = torch.cholesky_solve(yo[f][:, None], L)[:, 0]
        vt = torch.zeros(D_B, U, dtype=TD).index_add_(0, iit, A * w[:, None])
        chat[f] = ((Dt @ vt).T + (Ct @ vt.sum(1))[None]).numpy()
        ll[f] = -float(torch.log(L.diagonal()).sum()) - 0.5 * float(yo[f] @ w)
    return chat, ll


def sigma_term_a1(A0v, M, gh, Gb, s2, chunk=100):
    """sum_i M_i o Sigma_i with Sigma_i = Gamma_i - Gamma_i Psi_i^T (Psi_i Gamma_i Psi_i^T + s2 I)^{-1} Psi_i Gamma_i,
    Psi_i = A_i diag(gh) (= (Gamma_i^{-1} + G A_i^T A_i G / s2)^{-1}): (F, U, U)."""
    F, U = gh.shape
    R = A0v.shape[1]
    out = np.zeros((F, U, U))
    for s in range(0, F, chunk):
        sl = slice(s, min(F, s + chunk))
        Psi = A0v[None] * gh[sl][:, None, None, :]                    # (Fc, d_b, R, U)
        GP = Gb[None] @ Psi.transpose(0, 1, 3, 2)                     # Gamma_i Psi_i^T (Fc, d_b, U, R)
        S = Psi @ GP + s2 * np.eye(R)
        Sig = Gb[None] - GP @ np.linalg.solve(S, GP.transpose(0, 1, 3, 2))
        out[sl] = np.einsum("iuv,fiuv->fuv", M, Sig)
    return out


def sigma_term_exact(sch, gh, Dt, Ct, s2, M, Gb):
    """sum_i M_i o Sigma_i with the U x U diagonal blocks of the exact joint posterior covariance
    Gamma - (L^{-1} Psi Gamma)^T (L^{-1} Psi Gamma), S = L L^T as in sm.rx_joint; per frame (slow): (F, U, U)."""
    F, U = gh.shape
    ii = sch["ii"]
    n_obs = len(ii)
    out = np.zeros((F, U, U))
    if n_obs == 0:
        return out
    obs = np.unique(ii)                                               # M_i = 0 for the other coefficients
    iit, ot = torch.from_numpy(ii), torch.from_numpy(obs)
    KD, KC = Dt[iit][:, iit], Ct[iit][:, iit]
    Dr, Cr = Dt[iit][:, ot], Ct[iit][:, ot]
    A0 = torch.from_numpy(np.ascontiguousarray(sch["A0obs"]))
    ght = torch.from_numpy(np.ascontiguousarray(gh))
    Mo, Go = M[obs], Gb[obs]
    for f in range(F):
        A = A0 * ght[f]
        beta = A.sum(1)
        S = (A @ A.T).mul_(KD).addcmul_(KC, torch.outer(beta, beta))
        S.diagonal().add_(s2)
        L = torch.linalg.cholesky(S)
        X = (A[:, :, None] * Dr[:, None, :] + beta[:, None, None] * Cr[:, None, :]).reshape(n_obs, U * len(obs))
        Zt = torch.linalg.solve_triangular(L, X, upper=False).reshape(n_obs, U, len(obs))
        ZZ = torch.einsum("rui,rvi->iuv", Zt, Zt).numpy()
        out[f] = np.einsum("iuv,iuv->uv", Mo, Go - ZZ)
    return out


def fisher_a1(A0v, gh, Gb, s2, chunk=100):
    """expected Fisher information of g under the per-coefficient (A1) model y_i ~ N(0, S_i), S_i = Psi_i Gamma_i
    Psi_i^T + s2 I, Psi_i = A_i diag(gh): I = sum_i (AA o VV + AV o AV^T), AA = A_i^T S_i^-1 A_i,
    VV = V_i^T S_i^-1 V_i, AV = A_i^T S_i^-1 V_i, V_i = Psi_i Gamma_i (= (1/2) tr(S^-1 dS_u S^-1 dS_v) with
    dS_u = a_u v_u^T + v_u a_u^T): (F, U, U)."""
    F, U = gh.shape
    R = A0v.shape[1]
    out = np.zeros((F, U, U))
    for s in range(0, F, chunk):
        sl = slice(s, min(F, s + chunk))
        Psi = A0v[None] * gh[sl][:, None, None, :]                    # (Fc, d_b, R, U)
        V = Psi @ Gb[None]
        S = V @ Psi.transpose(0, 1, 3, 2) + s2 * np.eye(R)
        Ab = np.broadcast_to(A0v[None], Psi.shape)
        PX = np.linalg.solve(S, np.concatenate([Ab, V], -1))          # S^-1 [A, V]
        At, Vt = Ab.transpose(0, 1, 3, 2), V.transpose(0, 1, 3, 2)
        AA, VV, AV = At @ PX[..., :U], Vt @ PX[..., U:], At @ PX[..., U:]
        out[sl] = (AA * VV + AV * AV.transpose(0, 1, 3, 2)).sum(1)
    return out


def solve_batch(Rm, b):
    try:
        return np.linalg.solve(Rm, b[..., None])[..., 0]
    except np.linalg.LinAlgError:
        return np.stack([np.linalg.lstsq(R, v, rcond=None)[0] for R, v in zip(Rm, b)])


def bqp(Rm, b, lo, hi=None):
    """argmin_x x^T R x / 2 - b^T x subject to lo <= x (<= hi) per frame (R symmetric positive definite): R^{-1} b where
    it is feasible, otherwise the best feasible stationary point over all active sets (each user free, at the lower or
    at the upper bound; exact for the convex problem, 2^U or 3^U small solves, U <= 5): (F, U)."""
    x = solve_batch(Rm, b)
    viol = (x < lo).any(1) if hi is None else ((x < lo) | (x > hi)).any(1)
    if not viol.any():
        return x
    U = b.shape[1]
    states = [np.array(s) for s in itertools.product((0, 1) if hi is None else (0, 1, 2), repeat=U)]
    for f in np.flatnonzero(viol):
        R_, b_, lo_ = Rm[f], b[f], lo[f]
        hi_ = hi[f] if hi is not None else np.full(U, np.inf)
        best, bobj = None, np.inf
        for st in states:
            fr = st == 0
            xf = np.where(st == 1, lo_, np.where(st == 2, hi_, 0.0))
            if fr.any():
                try:
                    xf[fr] = np.linalg.solve(R_[np.ix_(fr, fr)], b_[fr] - R_[np.ix_(fr, ~fr)] @ xf[~fr])
                except np.linalg.LinAlgError:
                    continue
                if (xf[fr] < lo_[fr]).any() or (xf[fr] > hi_[fr]).any():
                    continue
            obj = 0.5 * xf @ R_ @ xf - b_ @ xf
            if obj < bobj:
                best, bobj = xf, obj
        x[f] = best
    return x


def em_gains(sch, y, c, gh0, s2, Dt, Ct, Gb, a, cov, yp=None, Ep=0.0, energy=None):
    """EM of the gains (see header): Gaussian-ML (energy None; plain EM or Fisher scoring with the line search) or
    energy-matched (energy = the member's data energy). c is used only for the per-iteration metrics. Returns a dict:
    ch (final chat), gh, hist (F, T+1, U), hcos (F, T+1) user-mean CosSim, herr (F, T+1) squared error per frame (frames
    that stopped carry their last value), used (M-steps per frame), nsolve (joint solves per frame), nrej (line-search
    trials per frame), nacc (extrapolated energy-matched steps per frame), cov_diff (exact: mean relative difference of
    the a1 and exact covariance terms per iteration)."""
    F, U = gh0.shape
    T = a.em_iters
    A0v = sch["A0"] * sch["valid"][:, :, None]                       # inactive rows zero (they are already)
    M = np.einsum("iru,irv->iuv", A0v, A0v)                            # sum over coefficient i's rows
    fisher = a.em_step == "fisher" and energy is None
    accel = energy is not None and not a.emr_plain
    sEp = math.sqrt(Ep)

    def post(gv, idx):
        ch, ll = joint_post(sch, gv, y[idx], Dt, Ct, s2)
        if Ep > 0:
            ll = ll - ((yp[idx] - sEp * gv) ** 2).sum(1) / (2 * s2)
        return ch, ll

    gh = np.maximum(gh0, a.em_g_min)
    allf = np.arange(F)
    ch, ll = post(gh, allf)
    hist, hcos, herr = np.empty((F, T + 1, U)), np.empty((F, T + 1)), np.empty((F, T + 1))
    hist[:, 0], hcos[:, 0], herr[:, 0] = gh, sm.cos_rows(c, ch).mean(1), ((c - ch) ** 2).sum((1, 2))
    used, nsolve, nrej, nacc = np.zeros(F, int), np.ones(F, int), np.zeros(F, int), np.zeros(F, int)
    dprev = np.zeros((F, U))
    act = allf
    cov_diff = []
    for t in range(T):
        if act.size == 0:
            break
        gt, cht = gh[act], ch[act]
        Rm = np.einsum("iuv,fui,fvi->fuv", M, cht, cht)
        if cov == "a1":
            Rm += sigma_term_a1(A0v, M, gt, Gb, s2)
        else:
            Rx = sigma_term_exact(sch, gt, Dt, Ct, s2, M, Gb)
            Ra = sigma_term_a1(A0v, M, gt, Gb, s2)
            cov_diff.append(float(np.mean(np.linalg.norm(Ra - Rx, axis=(1, 2)) / np.linalg.norm(Rx, axis=(1, 2)))))
            Rm += Rx
        b = np.einsum("iru,fir,fui->fu", A0v, y[act], cht)
        if energy is not None:                                         # moments scaled to the known data energy
            r = np.sqrt(energy / np.maximum(np.einsum("fuu->fu", Rm), 1e-300))
            Rm = Rm * r[:, :, None] * r[:, None, :]
            b = b * r
        if Ep > 0:
            Rm += Ep * np.eye(U)
            b += sEp * yp[act]
        if fisher:
            score = (b - np.einsum("fuv,fv->fu", Rm, gt)) / s2        # = E[complete-data score | y] (Fisher identity)
            Inf = fisher_a1(A0v, gt, Gb, s2) + (Ep / s2) * np.eye(U)
            new = bqp(Inf, np.einsum("fuv,fv->fu", Inf, gt) + score, np.maximum(gt / 2, a.em_g_min), 2 * gt)
        else:
            new = bqp(Rm, b, np.full_like(gt, a.em_g_min))
            if accel:                                                  # extrapolated energy-matched step (Aitken rate)
                d = np.log(new) - np.log(gt)                           # plain step in log g
                dp = dprev[act]
                with np.errstate(divide="ignore", invalid="ignore"):
                    q = np.linalg.norm(d, axis=1) / np.linalg.norm(dp, axis=1)
                ok = np.isfinite(q) & (q < 0.95) & ((d * dp).sum(1) > 0)   # contracting without a sign flip
                om = np.where(ok, np.minimum(1.0 / (1.0 - np.where(ok, q, 0.0)), a.emr_omax), 1.0)
                dprev[act] = d
                nacc[act] += ok
                new = np.maximum(gt * np.exp(om[:, None] * d), a.em_g_min)
        chn, lln = post(new, act)
        nsolve[act] += 1
        stay = np.zeros(act.size, bool)
        if fisher and a.em_ls > 0:                                     # line search on the exact log-likelihood
            step, ll0 = new - gt, ll[act]
            tol = 1e-10 * np.abs(ll0)
            d0 = (score * step).sum(1)                                 # model slope along the step (>= 0)
            k2 = lln - ll0 - d0                                        # ll(al) ~ ll0 + d0 al + k2 al^2
            with np.errstate(divide="ignore", invalid="ignore"):
                ast = np.where(k2 < 0, d0 / (-2 * k2), np.inf)         # maximizer of the fit
            small = (np.abs(step) / gt).max(1) < a.em_tol              # converged step: taken without a search
            ok1 = (lln >= ll0 - tol) | small                           # the full step is an ascent step
            redo = ~ok1 | ((ast < 0.75) & ~small)                      # overshoot (zigzag) or descent: shorter step
            al = np.where(redo, np.clip(np.nan_to_num(ast, posinf=0.5), 0.1, 0.75), 1.0)
            best = np.where(ok1, lln, -np.inf)
            for _ in range(a.em_ls):
                if not redo.any():
                    break
                bi = np.flatnonzero(redo)
                trial = gt[bi] + al[bi, None] * step[bi]
                chb, llb = post(trial, act[bi])
                nsolve[act[bi]] += 1
                nrej[act[bi]] += 1
                take = (llb >= ll0[bi] - tol[bi]) & (llb > best[bi])
                tk = bi[take]
                new[tk], chn[tk], lln[tk], best[tk] = trial[take], chb[take], llb[take], llb[take]
                redo[bi] = ~take & ~ok1[bi]                            # keep the full step if it was an ascent step
                al[bi] /= 2
            stay = ~np.isfinite(best)                                  # no ascent step: keep ghat^(t), the frame stops
            new[stay], chn[stay], lln[stay] = gt[stay], cht[stay], ll0[stay]
        conv = ((np.abs(new - gt) / gt).max(1) < a.em_tol) | stay
        gh[act], ch[act], ll[act] = new, chn, lln
        used[act] += 1
        hist[act, t + 1] = new
        hcos[act, t + 1] = sm.cos_rows(c[act], chn).mean(1)
        herr[act, t + 1] = ((c[act] - chn) ** 2).sum((1, 2))
        act = act[~conv]
    for t in range(1, T + 1):
        st = used < t
        hist[st, t], hcos[st, t], herr[st, t] = hist[st, t - 1], hcos[st, t - 1], herr[st, t - 1]
    return {"ch": ch, "gh": gh, "hist": hist, "hcos": hcos, "herr": herr, "used": used, "nsolve": nsolve,
            "nrej": nrej, "nacc": nacc, "cov_diff": cov_diff}


# ----------------------------------------------------------------- checks and statistics
def boot(dF, idx):
    b = dF[idx].mean(1)
    lo, hi = np.percentile(b, [2.5, 97.5])
    return {"mean": float(dF.mean()), "lo": float(lo), "hi": float(hi)}


def consistency(path, arrays, snrs, g_min, nopilot):
    """frame-by-frame comparison with a shared_masks.py npz of the same point (float32 arrays)."""
    S = np.load(path)
    pairs = [("load_pu", "load_pu"), ("load_joint", "load_joint")]
    pairs += [("gen_joint_true0", "gen_joint")] if nopilot else [("gen_joint_true", "gen_joint_true"),
                                                                  ("gen_joint_pilot", "gen_joint")]
    out = {"file": os.path.basename(path)}
    for snr in snrs:
        if f"g_{snr}" not in S:
            continue
        F = min(len(S[f"g_{snr}"]), len(arrays[f"g_{snr}"]))
        o = {"frames": F, "g_maxabs": float(np.abs(S[f"g_{snr}"][:F] - arrays[f"g_{snr}"][:F].astype(np.float32)).max())}
        if not nopilot:
            o["pilot_raw_maxabs"] = float(np.abs(S[f"gh_{snr}"][:F] - arrays[f"pilraw_{snr}"][:F].astype(np.float32)).max())
        for m, ms in pairs:
            k1, k2 = f"cos_{m}_{snr}", f"cos_{ms}_{snr}"
            if k1 not in arrays or k2 not in S:
                continue
            keep = np.ones(F, bool)
            if m == "gen_joint_pilot":
                keep = ~(arrays[f"pilraw_{snr}"][:F] < g_min).any(1)
            c1, c2 = arrays[k1][:F][keep], S[k2][:F][keep]
            e1, e2 = arrays[f"err_{m}_{snr}"][:F][keep], S[f"err_{ms}_{snr}"][:F][keep]
            o[m] = {"vs": ms, "frames_compared": int(keep.sum()), "frames_clamp_excluded": int((~keep).sum()),
                    "cos_maxabs": float(np.abs(c1 - c2).max()) if keep.any() else None,
                    "err_maxrel": float((np.abs(e1 - e2) / np.maximum(np.abs(e2), 1e-30)).max()) if keep.any() else None,
                    "cos_mean": [float(c1.mean()), float(c2.mean())] if keep.any() else None}
        out[str(snr)] = o
    return out


def tag_of(a):
    t = a.dataset + (f"_mix{a.mix:g}" if a.mix is not None else "") + f"_K{a.K}_U{a.U}"
    if a.nearfar:
        t += f"_nf{a.nearfar:g}"
    if a.phase:
        t += f"_ph{a.phase:g}"
    t += f"_E{a.pilot_energy:g}"
    if a.gains != "rayleigh":
        t += "_geq"
    if a.gauss:
        t += "_gauss"
    if a.em_cov == "exact":
        t += "_emexact"
    elif a.em_cov == "auto":
        t += f"_emauto{a.em_auto_snr:g}"
    if a.em_step == "fisher":
        t += "_fs"
    if a.em_iters != DEF["em_iters"]:
        t += f"_it{a.em_iters}"
    if a.em_tol != DEF["em_tol"]:
        t += f"_tol{a.em_tol:g}"
    if a.g_min != DEF["g_min"]:
        t += f"_gmin{a.g_min:g}"
    if a.em_g_min != DEF["em_g_min"]:
        t += f"_emgmin{a.em_g_min:g}"
    if a.em_ls != DEF["em_ls"]:
        t += f"_ls{a.em_ls}"
    if a.emr_plain:
        t += "_emrplain"
    elif a.emr_omax != DEF["emr_omax"]:
        t += f"_om{a.emr_omax:g}"
    return t + a.suffix


def md5(path):
    with open(path, "rb") as fh:
        return hashlib.md5(fh.read()).hexdigest()


# ----------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["coco", "flickr", "agnews"], default="coco")
    ap.add_argument("--mix", type=float, default=None)
    ap.add_argument("--U", type=int, required=True)
    ap.add_argument("--K", type=int, choices=[4, 2], default=4)
    ap.add_argument("--snrs", type=int, nargs="+", default=list(SNRS))
    ap.add_argument("--frames", type=int, default=None, help="default: every test image (coco 2000, flickr 1000)")
    ap.add_argument("--nearfar", type=float, default=0.0)
    ap.add_argument("--gains", choices=["rayleigh", "equal"], default="rayleigh")
    ap.add_argument("--phase", type=float, default=0.0)
    ap.add_argument("--pilot_energy", type=float, default=1.0, help="Ep of the one-use pilot of the gen masks")
    ap.add_argument("--g_min", type=float, default=DEF["g_min"], help="clamp of the raw pilot estimate")
    ap.add_argument("--em_g_min", type=float, default=DEF["em_g_min"],
                    help="lower bound of the EM start and iterates (default 0.05; 1e-3 = the floor sqrt(1e-6) of (8))")
    ap.add_argument("--em_iters", type=int, default=DEF["em_iters"])
    ap.add_argument("--em_cov", choices=["a1", "exact", "auto"], default="a1")
    ap.add_argument("--em_auto_snr", type=float, default=0.0, help="--em_cov auto: exact at SNR <= this, a1 above")
    ap.add_argument("--em_tol", type=float, default=DEF["em_tol"])
    ap.add_argument("--em_step", choices=["em", "fisher"], default="em",
                    help="em: ghat = argmin_{g >= em_g_min} (R, b) (plain EM); fisher: scoring step + line search, same fixed point")
    ap.add_argument("--em_ls", type=int, default=DEF["em_ls"],
                    help="line-search trials of the Fisher step (0: no line search, every scoring step taken)")
    ap.add_argument("--emr_plain", action="store_true", help="energy-matched EM without the extrapolated step")
    ap.add_argument("--emr_omax", type=float, default=DEF["emr_omax"],
                    help="largest extrapolation factor of the energy-matched step")
    ap.add_argument("--gauss", action="store_true", help="Gaussian content with the fitted (true) Gamma")
    ap.add_argument("--modes", nargs="+", default=list(MODES))
    ap.add_argument("--check_pilot", default=None, help="shared_masks.py --pilot npz of the same point")
    ap.add_argument("--check_nopilot", default=None, help="shared_masks.py npz (no pilot) of the same point")
    ap.add_argument("--n_boot", type=int, default=1000)
    ap.add_argument("--outdir", default=os.path.join(sm.DATA, "shared", "pilot"))
    ap.add_argument("--suffix", default="")
    a = ap.parse_args()
    a.control, a.control_c, a.c_from, a.a1 = False, "zero", "self", False      # fixed (sm.Content options)
    assert not (a.mix is not None and a.dataset == "agnews"), "--mix needs caption content"
    assert a.U <= R_NOISE and a.K <= R_NOISE and a.pilot_energy > 0 and a.em_iters >= 0 and a.g_min > 0 < a.em_g_min
    assert all(m in ALL for m in a.modes), a.modes
    torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "2")))
    T0 = time.time()
    code = {"shared_pilot.py": md5(os.path.abspath(__file__)), "shared_masks.py": md5(os.path.abspath(sm.__file__))}
    U, K, Ep = a.U, a.K, a.pilot_energy
    cont = sm.Content(a)
    F = a.frames or cont.n_test
    d_s = K * D_B
    n_load, E_load = d_s, d_s / U
    n_gen, E_gen = d_s - U, d_s / U - Ep
    assert E_gen > 0
    Emem = {"load": E_load, "gen": E_gen, "gen0": E_load}
    lam = cont.lam_m
    lam_sum = float(lam.sum())
    Dt, Ct = torch.from_numpy(cont.D), torch.from_numpy(cont.C)
    Gb = gamma_blocks(cont, U)
    Q = sm.mode_basis(U)
    modes = list(a.modes)
    tag = tag_of(a)
    print(f"{tag}: F={F} n_data load/gen/gen0={n_load}/{n_gen}/{n_load} E load/gen/gen0={E_load:.1f}/{E_gen:.1f}/"
          f"{E_load:.1f} kappa_bar={cont.info['kappa_bar']:.4f} em={a.em_cov} {a.em_step} x{a.em_iters} tol={a.em_tol:g} "
          f"modes={modes} code md5 {code['shared_pilot.py'][:8]} sm {code['shared_masks.py'][:8]}", flush=True)
    res, design, arrays, em_cov_diff = {m: {} for m in modes}, {}, {}, {}
    for snr in a.snrs:
        s2 = 10 ** (-snr / 10.0)
        cov = "exact" if a.em_cov == "exact" or (a.em_cov == "auto" and snr <= a.em_auto_snr) else "a1"
        off = 1000 * U
        rng_c = np.random.default_rng(123 + snr + off)
        c = None if a.gauss else cont.frames(rng_c, F, U)
        Z = np.random.default_rng(5123 + snr + off).standard_normal((F, U, 5))
        h = np.abs(Z[..., 0] + 1j * Z[..., 1]) / math.sqrt(2) if a.gains == "rayleigh" else np.ones((F, U))
        offs = 10 ** ((2 * ndtr(Z[..., 2]) - 1) * a.nearfar / 20.0)
        phi = np.radians(a.phase) * Z[..., 3]
        g = h * offs * np.cos(phi)                                    # effective (true) gain
        pil = g + math.sqrt(s2 / Ep) * Z[..., 4]                       # y_p / sqrt(Ep) (= sm gh_gen under --pilot)
        yp = math.sqrt(Ep) * pil                                       # y_p = sqrt(Ep) g + sqrt(s2) Z4
        gh_pil = np.maximum(pil, a.g_min)
        noise = np.random.default_rng(9123 + snr + off).standard_normal((F, D_B, R_NOISE))
        if a.gauss:
            c = cont.gauss(np.random.default_rng(7123 + snr + off), F, U, Q)
        nrm = (c ** 2).sum(-1)
        arrays[f"nrm_{snr}"] = nrm.astype(np.float32)
        arrays[f"g_{snr}"] = g
        arrays[f"pilraw_{snr}"] = pil
        sch = {}
        for k in {SCHEME[m] for m in modes}:
            if k == "load":
                sch[k] = sm.design_load(lam, U, E_load, n_load, s2)
            else:
                sch[k] = sm.design_prop2(lam, cont.kap, U, Emem[k], n_load if k == "gen0" else n_gen, s2, "gen_joint")
        design[snr] = {k: v["info"] for k, v in sch.items()}
        ys, rho = {}, {}
        for k in sorted(sch):
            if any(SCHEME[m] == k and m not in RHO for m in modes):
                ys[k] = sm.observe(sch[k], c, g, noise, s2)
            if any(SCHEME[m] == k and m in RHO for m in modes):
                rho[k] = tx_rho(sch[k], c, Emem[k])                    # every user scaled to its data energy (5)
                ys[k + "_rho"] = sm.observe(sch[k], c, g * rho[k], noise, s2)
                arrays[f"rho_{k}_{snr}"] = rho[k]
                design[snr][f"rho_{k}"] = {"mean": float(rho[k].mean()), "std": float(rho[k].std()),
                                           "min": float(rho[k].min()), "max": float(rho[k].max())}
        ghb = {}
        if "load_rho" in ys:
            ghb = {r: blind_load(sch["load"], ys["load_rho"], s2, E_load, r) for r in ("du", "active")}
        em_cov_diff[snr] = {}
        cosF, nmF, grF = {}, {}, {}
        for m in modes:
            t0 = time.time()
            k = SCHEME[m]
            yk = ys[k + ("_rho" if m in RHO else "")]
            gh, emo = None, None
            if m in ("load_pu", "load_pu_rho"):
                ch = sm.rx_pu(sch[k], g, yk, lam, s2, g, cont.lam_true, False)[0]
            elif m in TRUE:
                ch = sm.rx_joint(sch[k], g, yk, Dt, Ct, s2, lam_sum, False)[0]
            elif m in ("load_pu_blind", "load_pu_blindA"):
                gh = ghb["du" if m == "load_pu_blind" else "active"]
                ch = sm.rx_pu(sch[k], gh, yk, lam, s2, g, cont.lam_true, False)[0]
            elif m in ("load_joint_blind", "load_joint_blindA"):
                gh = ghb["du" if m == "load_joint_blind" else "active"]
                ch = sm.rx_joint(sch[k], gh, yk, Dt, Ct, s2, lam_sum, False)[0]
            elif m == "gen_joint_pilot":
                gh = gh_pil
                ch = sm.rx_joint(sch[k], gh, yk, Dt, Ct, s2, lam_sum, False)[0]
            elif m in EM:
                if k == "gen":
                    gh0 = gh_pil
                elif k == "gen0":
                    A0v = sch[k]["A0"] * sch[k]["valid"][:, :, None]
                    gh0 = blind_common(sch[k], yk, s2, np.einsum("iru,irv->iuv", A0v, A0v), Gb)
                else:
                    gh0 = blind_load(sch[k], yk, s2, E_load)          # (8) on the observations of this mode
                Epm = Ep if k == "gen" else 0.0
                emo = em_gains(sch[k], yk, c, gh0, s2, Dt, Ct, Gb, a, cov, yp if Epm > 0 else None, Epm,
                               Emem[k] if m in EMR else None)
                ch, gh = emo["ch"], emo["gh"]
            else:
                raise ValueError(m)
            sec = (time.time() - t0) / F
            err = ((c - ch) ** 2).sum(-1)
            cz = sm.cos_rows(c, ch)
            cosF[m] = cz.mean(1)
            nmF[m] = err.sum(1) / nrm.sum(1)
            out = {"cos": float(cz.mean()), "weakest": float(cz.min(1).mean()), "nmse": float(err.sum() / nrm.sum()),
                   "nmse_frame_mean": float(nmF[m].mean()), "sec_per_frame": float(sec)}
            msg = ""
            if m in RHO:
                rk = rho[k][..., None]
                out["nmse_rho"] = float(((rk * c - ch) ** 2).sum() / (rk[..., 0] ** 2 * nrm).sum())
            if gh is not None:
                rr = rel_rmse(gh, g)
                grF[m] = rr
                floor = a.em_g_min if m in EM else (a.g_min if m == "gen_joint_pilot" else 1e-3)
                out.update({"gain_rel_rmse": float(rr.mean()), "gain_rel_rmse_median": float(np.median(rr)),
                            "gain_rel_rmse_pooled": float(np.sqrt((((gh - g) / g) ** 2).mean())),
                            "gain_floor": floor, "gain_at_floor_frac": float((gh <= floor * (1 + 1e-9)).mean())})
                arrays[f"gh_{m}_{snr}"] = gh
                msg += f" grmse={out['gain_rel_rmse']:.4f} (med {out['gain_rel_rmse_median']:.4f})"
            if emo is not None:
                er = rel_rmse(emo["hist"], g[:, None, :])               # (F, T+1)
                out.update({"em_cov": cov, "em_rmse_per_iter": er.mean(0).tolist(),
                            "em_rmse_median_per_iter": np.median(er, 0).tolist(),
                            "em_cos_per_iter": emo["hcos"].mean(0).tolist(),
                            "em_nmse_per_iter": (emo["herr"].sum(0) / nrm.sum()).tolist(),
                            "em_iters_used": float(emo["used"].mean()),
                            "em_converged_frac": float((emo["used"] < a.em_iters).mean()),
                            "em_solves_per_frame": float(emo["nsolve"].mean()),
                            "em_ls_trials_per_frame": float(emo["nrej"].mean()),
                            "emr_extrapolated_per_frame": float(emo["nacc"].mean())})
                if emo["cov_diff"]:
                    out["em_cov_a1_vs_exact_reldiff_per_iter"] = emo["cov_diff"]
                    em_cov_diff[snr][m] = emo["cov_diff"]
                arrays[f"emgh_{m}_{snr}"] = emo["hist"].astype(np.float32)
                arrays[f"emrmse_{m}_{snr}"] = er
                arrays[f"emcos_{m}_{snr}"] = emo["hcos"].astype(np.float32)
                arrays[f"emerr_{m}_{snr}"] = emo["herr"].astype(np.float32)
                arrays[f"emit_{m}_{snr}"] = emo["used"].astype(np.int16)
                arrays[f"emns_{m}_{snr}"] = emo["nsolve"].astype(np.int16)
                msg += (" em=" + "/".join(f"{x:.4f}" for x in out["em_rmse_per_iter"])
                        + " cos=" + "/".join(f"{x:.4f}" for x in out["em_cos_per_iter"])
                        + f" it={out['em_iters_used']:.2f} solves={out['em_solves_per_frame']:.2f}"
                        + (f" ls={out['em_ls_trials_per_frame']:.2f}" if a.em_step == "fisher" and m not in EMR else "")
                        + (f" xtr={out['emr_extrapolated_per_frame']:.2f}" if m in EMR else "") + f" [{cov}]")
                if emo["cov_diff"]:
                    msg += " a1-vs-exact=" + "/".join(f"{x:.2e}" for x in emo["cov_diff"])
            res[m][snr] = out
            arrays[f"cos_{m}_{snr}"] = cz.astype(np.float32)
            arrays[f"err_{m}_{snr}"] = err.astype(np.float32)
            print(f"{tag} snr={snr} {m:19s} cos={out['cos']:.4f} weakest={out['weakest']:.4f} nmse={out['nmse']:.5f}"
                  + (f" nmse_rho={out['nmse_rho']:.5f}" if "nmse_rho" in out else "") + msg
                  + f" {sec * 1e3:.1f}ms/frame", flush=True)
        idx = np.random.default_rng(555 + snr).integers(0, F, (a.n_boot, F))
        for m in modes:
            for r in REFS:
                if r not in cosF:
                    continue
                res[m][snr][f"d_vs_{r}"] = boot(cosF[m] - cosF[r], idx)
                res[m][snr][f"dnmse_vs_{r}"] = boot(nmF[m] - nmF[r], idx)
                if m in grF and r in grF:
                    res[m][snr][f"dgrmse_vs_{r}"] = boot(grF[m] - grF[r], idx)
        for k, v in design[snr].items():
            if k.startswith("rho_"):
                print(f"   {k}: mean={v['mean']:.4f} std={v['std']:.4f} min={v['min']:.4f} max={v['max']:.4f}", flush=True)
                continue
            print(f"   design {k}: n_obs={v['n_obs']} n_uses={v['n_uses']} energy_dev={v['energy_dev']:.2e}"
                  + (f" pruned={v['n_pruned']} partial={v['n_partial_coef']}" if "n_pruned" in v else ""), flush=True)
    cons = {}
    for key, path, nop in (("pilot", a.check_pilot, False), ("nopilot", a.check_nopilot, True)):
        if path:
            cons[key] = consistency(path, arrays, a.snrs, a.g_min, nop)
            for snr in a.snrs:
                o = cons[key].get(str(snr))
                if o is None:
                    print(f"consistency {key} snr={snr}: not in {path}", flush=True)
                    continue
                print(f"consistency {key} snr={snr}: F={o['frames']} g {o['g_maxabs']:.1e}"
                      + (f" pilot {o['pilot_raw_maxabs']:.1e}" if "pilot_raw_maxabs" in o else ""), flush=True)
                for m, v in o.items():
                    if isinstance(v, dict):
                        print(f"   {m:16s} vs {v['vs']:15s} frames={v['frames_compared']} (clamp excl. "
                              f"{v['frames_clamp_excluded']}) max|dcos|={v['cos_maxabs']} max rel derr={v['err_maxrel']}"
                              f" cos mine/sm={v['cos_mean']}", flush=True)
    os.makedirs(a.outdir, exist_ok=True)
    R = {"args": vars(a), "frames": F, "code": code,
         "n_data": {"load": n_load, "gen": n_gen, "gen0": n_load},
         "energy_per_user": {"load": E_load, "gen": E_gen, "gen0": E_load},
         "pilot": {"pilot_energy": Ep, "uses_per_user": 1,
                   "charged_to": [m for m in ALL if SCHEME[m] == "gen"],
                   "g_min": a.g_min, "estimate": "max(y_p / sqrt(Ep), g_min)", "em_g_min": a.em_g_min},
         "fit": cont.info, "design": design, "res": res, "consistency": cons, "em_cov_diff": em_cov_diff,
         "runtime_s": time.time() - T0}
    json.dump(R, open(os.path.join(a.outdir, f"shared_pilot_{tag}.json"), "w"), indent=1)
    np.savez_compressed(os.path.join(a.outdir, f"shared_pilot_{tag}.npz"), **arrays)
    print(f"wrote shared_pilot_{tag}.json ({time.time() - T0:.0f} s)", flush=True)


if __name__ == "__main__":
    main()
