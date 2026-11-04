# =========================================================
# pca_csi_ls.py — is the greedy slot grant of the channel-aware masks near the best slot positions?
# (2026-11-04). Same channel, draws, placement, allocation (13), and receiver as mode csiq of pca_csi.py.
# After the greedy grant, a local search moves single slots between users and swaps pairs of slots of two
# users, per frame, as long as the sum of the per-user MSEs under (13) decreases (best move per frame and
# round, at most --maxit rounds). Every user keeps at least one slot, and a user still sends each group once
# on its strongest slots with the energy d_s/N per held slot. The search optimizes both the slot counts and
# the slot positions; it is a local optimum, not a proven global one.
# Output: ../data/pilot_corr/pca_csi_ls_U<U>[_nf<X>].json.  Usage: python pca_csi_ls.py --U 4 --nearfar 6
# =========================================================
import argparse, json, os
import numpy as np
import torch
from pca_csi import DATA, S, SEG, D_B, NB, channel, wf, greedy_counts, place_sorted, receive_sorted


def mse_sel(Gsel, lam, s2):
    """Gsel (F,S): gains of the held slots, -1 elsewhere -> sum MSE of the user under (13) per frame."""
    F = Gsel.shape[0]
    srt = -np.sort(-Gsel, axis=1)
    n = (Gsel >= 0).sum(1)
    g2 = np.full((F, D_B), 1e-12); m = np.zeros((F, D_B), bool)
    for b in range(NB):
        has = n > b
        g2[:, b * SEG:(b + 1) * SEG] = np.where(has[:, None], np.maximum(srt[:, b:b + 1], 1e-6) ** 2, 1e-12)
        m[:, b * SEG:(b + 1) * SEG] = has[:, None]
    P = wf(lam, g2, n * SEG, s2) * m
    return (lam * s2 / (g2 * P + s2) * m).sum(1) + (lam * ~m).sum(1)


def held(Gs, owner, j_of_frame):
    """gains of the slots held by user j_of_frame[f] in frame f, -1 elsewhere."""
    ar = np.arange(Gs.shape[0])
    G = Gs[ar, j_of_frame, :]
    return np.where(owner == j_of_frame[:, None], G, -1.0)


def local_search(Gs_all, sets, lam, s2, maxit):
    F, U, _ = Gs_all.shape
    owner_all = np.zeros((F, S), int)
    for f in range(F):
        for j in range(U):
            owner_all[f, sets[f][j]] = j
    cur_all = np.stack([mse_sel(held(Gs_all, owner_all, np.full(F, j)), lam, s2) for j in range(U)], 1)
    start = cur_all.sum(1).copy()
    changed = np.zeros(F, bool); alive = np.ones(F, bool); rounds = 0; active = []
    for it in range(maxit):
        idx = np.where(alive)[0]                             # frames not yet at a local optimum
        if idx.size == 0:
            break
        Gs, owner, cur = Gs_all[idx], owner_all[idx], cur_all[idx]
        n = idx.size; ar = np.arange(n)
        best = np.zeros(n); kind = np.zeros(n, int); bx = np.zeros(n, int); by = np.zeros(n, int)
        bma = np.zeros(n); bmb = np.zeros(n); bjb = np.zeros(n, int)
        for x in range(S):                                   # move slot x from its owner a to user b
            a = owner[:, x]
            Ga = held(Gs, owner, a); Ga[:, x] = -1.0
            ok_a = (Ga >= 0).sum(1) >= 1
            ma = mse_sel(Ga, lam, s2)
            for b in range(U):
                jb = np.full(n, b)
                Gb = held(Gs, owner, jb); Gb[:, x] = Gs[:, b, x]
                mb = mse_sel(Gb, lam, s2)
                gain = np.where(ok_a & (a != b), cur[ar, a] + cur[:, b] - ma - mb, -np.inf)
                upd = gain > best
                best[upd] = gain[upd]; kind[upd] = 1; bx[upd] = x; bjb[upd] = b; bma[upd] = ma[upd]; bmb[upd] = mb[upd]
        for x in range(S):                                   # swap slot x (owner a) with slot y (owner b)
            for y in range(x + 1, S):
                a = owner[:, x]; b = owner[:, y]
                Ga = held(Gs, owner, a); Ga[:, x] = -1.0; Ga[ar, y] = Gs[ar, a, y]
                Gb = held(Gs, owner, b); Gb[:, y] = -1.0; Gb[ar, x] = Gs[ar, b, x]
                ma = mse_sel(Ga, lam, s2); mb = mse_sel(Gb, lam, s2)
                gain = np.where(a != b, cur[ar, a] + cur[ar, b] - ma - mb, -np.inf)
                upd = gain > best
                best[upd] = gain[upd]; kind[upd] = 2; bx[upd] = x; by[upd] = y; bma[upd] = ma[upd]; bmb[upd] = mb[upd]
        go = best > 1e-9 * start[idx]
        active.append(int(go.sum()))
        alive[idx[~go]] = False
        if not go.any():
            break
        rounds += 1
        for k in np.where(go)[0]:
            f = idx[k]
            if kind[k] == 1:
                a, b = owner_all[f, bx[k]], bjb[k]
                owner_all[f, bx[k]] = b
            else:
                a, b = owner_all[f, bx[k]], owner_all[f, by[k]]
                owner_all[f, bx[k]], owner_all[f, by[k]] = b, a
            cur_all[f, a], cur_all[f, b] = bma[k], bmb[k]
        changed[idx[go]] = True
        print(f"  round {it + 1}: {go.sum()} frames improved, mean sum MSE {cur_all.sum(1).mean():.5f}", flush=True)
    new_sets = [[list(np.where(owner_all[f] == j)[0]) for j in range(U)] for f in range(F)]
    return new_sets, float(start.mean()), float(cur_all.sum(1).mean()), float(changed.mean()), rounds, active


def evaluate(a, lam, V, ct, Ete, snr, sets_fn):
    """reproduces the draws and the receiver of mode csiq of pca_csi.run for one SNR."""
    U = a.U
    rng = np.random.default_rng(123 + snr)
    s2 = 10 ** (-snr / 10.0)
    Gt, Gs = channel(rng, a.frames, U, a.rho, a.nearfar, a.delay)
    pos = rng.integers(0, ct.shape[0], (a.frames, U))
    sets, info = sets_fn(Gs, s2)
    cs, counts = [], []
    for j in range(U):
        c = ct[pos[:, j]]
        sl = [sets[f][j] for f in range(a.frames)]
        n = np.array([len(x) for x in sl]); counts.append(n)
        sob, P = place_sorted(Gs[:, j], sl, n * SEG, lam, s2)
        bh = receive_sorted(c, Gt[:, j], sob, P, lam, V, s2, rng, False)
        b = Ete[pos[:, j]]
        cs.append((bh * b).sum(1) / (np.linalg.norm(bh, axis=1) * np.linalg.norm(b, axis=1) + 1e-12))
    cs = np.stack(cs)
    return {"cos": float(cs.mean()), "weakest": float(cs.min(0).mean()), **info}, np.stack(counts, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--U", type=int, required=True)
    ap.add_argument("--rho", type=float, default=0.7)
    ap.add_argument("--nearfar", type=float, default=0.0)
    ap.add_argument("--delay", type=float, default=1.0)
    ap.add_argument("--frames", type=int, default=2000)
    ap.add_argument("--snrs", type=int, nargs="+", default=[0, 10, 20])
    ap.add_argument("--maxit", type=int, default=12)
    a = ap.parse_args()
    C = torch.load(os.path.join(DATA, "emb_ag_news_bert-base-uncased.pt"), map_location="cpu", weights_only=False)
    Etr, Ete = C["train"].double().numpy(), C["test"].double().numpy()
    lam, V = np.linalg.eigh(Etr.T @ Etr / Etr.shape[0]); o = np.argsort(lam)[::-1]
    lam, V = np.maximum(lam[o], 1e-12), V[:, o]
    ct = Ete @ V
    tag = f"U{a.U}" + (f"_nf{int(a.nearfar)}" if a.nearfar else "")
    out_path = os.path.join(DATA, "pilot_corr", f"pca_csi_ls_{tag}" + ("_" + "_".join(map(str, a.snrs)) if a.snrs != [0, 10, 20] else "") + ".json")
    res = {}
    cache = {}
    def grant(Gs, s2):
        if s2 not in cache:
            cache[s2] = greedy_counts(Gs, lam, s2, "mse")
        return cache[s2]
    for snr in a.snrs:
        def greedy(Gs, s2):
            return grant(Gs, s2), {}
        def searched(Gs, s2):
            sets = grant(Gs, s2)
            ns, m0, m1, ch, rounds, active = local_search(Gs, sets, lam, s2, a.maxit)
            return ns, {"mse_greedy": m0, "mse_search": m1, "frames_changed": ch, "rounds": rounds, "improved_per_round": active}
        r0, n0 = evaluate(a, lam, V, ct, Ete, snr, greedy)
        r1, n1 = evaluate(a, lam, V, ct, Ete, snr, searched)
        r1["counts_changed"] = float((n0 != n1).any(1).mean())
        res[str(snr)] = {"csiq": r0, "csiql": r1}
        print(f"U={a.U} snr={snr} greedy cos={r0['cos']:.4f} weakest={r0['weakest']:.4f} | search cos={r1['cos']:.4f} "
              f"weakest={r1['weakest']:.4f} sumMSE {r1['mse_greedy']:.5f}->{r1['mse_search']:.5f} "
              f"changed {r1['frames_changed']:.3f} counts changed {r1['counts_changed']:.3f}", flush=True)
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        json.dump({"args": vars(a), "res": res}, open(out_path, "w"), indent=1)


if __name__ == "__main__":
    main()
