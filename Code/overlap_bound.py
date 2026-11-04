# =========================================================
# overlap_bound.py — can overlapping real-valued masks beat the disjoint masks when the receiver is the
# joint (multi-user) LMMSE estimator? (2026-11-04). For a Gaussian source and linear analog transmission the
# joint LMMSE estimator is the MMSE estimator, so no receiver, SIC included, does better for a given
# transmit design: the comparison bounds what overlap with SIC can gain.
# Model: the block of d_s = 24 x 128 real uses, every user with the PCA spectrum lam of the AG News BERT
# embeddings in 6 groups of 128, per-user energy d_s/U, gain g_u per user (equal, or near-far spread from
# +nf to -nf dB over the users), noise s2 = 10^(-SNR/10). Restriction (keeps the problem small): use i of
# every slot mixes only the coefficients i of the groups (128 independent problems of 24 uses x 6U
# coefficients, coupled by the per-user energy). Schemes, all with the joint LMMSE receiver:
#   closed    the load masks (4) with the allocation (12), as in the paper
#   disjoint  every slot owned by one user (load masks), any linear code of the user on its own slots, optimized
#   overlap   any real 24 x 6U transmit matrix per use index, i.e. real-valued overlapping masks, optimized
#             (from the closed form and from random starts; the problem is not convex)
# Output: ../data/extra/overlap_bound_U<U>_nf<X>.json.  Usage: python overlap_bound.py --U 6 --nearfar 6
# =========================================================
import argparse, json, math, os
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


def closed_form(lam, g, s2, U):
    """the paper's scheme: load masks, groups repeated cyclically, allocation (12) with the user gain."""
    A = torch.zeros(SEG, S, NB * U, dtype=torch.float64)
    lam_g = torch.tensor(lam.reshape(NB, SEG).T.copy()).repeat(1, U)
    for j in range(U):
        slots = [s for s in range(S) if s % U == j]
        nb = min(len(slots), NB)
        P = wf(lam[:nb * SEG], np.full((1, nb * SEG), float(g[j]) ** 2), np.array([S * SEG / U]), s2)[0]
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--U", type=int, required=True)
    ap.add_argument("--nearfar", type=float, default=0.0)
    ap.add_argument("--snrs", type=int, nargs="+", default=[0, 10, 20])
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--restarts", type=int, default=3)
    a = ap.parse_args()
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
    json.dump({"args": vars(a), "res": res},
              open(os.path.join(DATA, "extra", f"overlap_bound_U{U}_nf{int(a.nearfar)}.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
