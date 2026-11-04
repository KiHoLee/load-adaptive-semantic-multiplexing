# =========================================================
# pca_csi_counts.py — who receives the slots of the greedy grant (csiq of pca_csi.py), 2026-11-04.
# Same channel and draws as pca_csi.py (seed 123 + SNR, AR(1) over the 24 slots, offsets of up to
# +-nearfar dB). Users of each frame are ranked by their mean slot gain g^2 (rank 0 strongest), and the
# script stores the mean slot count per rank, the share of frames in which the weakest user receives more,
# as many as, or fewer slots than the equal count N/U, and the rank correlation of count and gain.
# Output: ../data/pilot_corr/pca_csi_counts_U<U>_nf<X>.json.  Usage: python pca_csi_counts.py --U 6 --nearfar 6
# =========================================================
import argparse, json, os
import numpy as np
import torch
from scipy.stats import spearmanr
from pca_csi import DATA, S, channel, greedy_counts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--U", type=int, required=True)
    ap.add_argument("--rho", type=float, default=0.7)
    ap.add_argument("--nearfar", type=float, default=6.0)
    ap.add_argument("--frames", type=int, default=2000)
    ap.add_argument("--snrs", type=int, nargs="+", default=[0, 10, 20])
    a = ap.parse_args()
    C = torch.load(os.path.join(DATA, "emb_ag_news_bert-base-uncased.pt"), map_location="cpu", weights_only=False)
    Etr = C["train"].double().numpy()
    lam = np.linalg.eigvalsh(Etr.T @ Etr / Etr.shape[0])[::-1]
    lam = np.maximum(lam, 1e-12)
    res = {}
    for snr in a.snrs:
        rng = np.random.default_rng(123 + snr)
        s2 = 10 ** (-snr / 10.0)
        Gt, Gs = channel(rng, a.frames, a.U, a.rho, a.nearfar, 1.0)
        sets = greedy_counts(Gs, lam, s2, "mse")
        n = np.array([[len(sets[f][j]) for j in range(a.U)] for f in range(a.frames)], float)
        strength = (Gs ** 2).mean(2)                          # (F,U) mean slot gain of every user
        order = np.argsort(-strength, 1)                       # rank 0 = strongest user of the frame
        by_rank = np.take_along_axis(n, order, 1).mean(0)
        weak = n[np.arange(a.frames), order[:, -1]]
        eq = S / a.U
        rho_s = [spearmanr(strength[f], n[f]).correlation for f in range(a.frames) if np.ptp(n[f]) > 0]
        res[str(snr)] = {"count_by_rank": by_rank.tolist(), "equal_count": eq,
                         "weakest_more": float((weak > eq + 1e-9).mean()), "weakest_equal": float((abs(weak - eq) < 1e-9).mean()),
                         "weakest_fewer": float((weak < eq - 1e-9).mean()),
                         "spearman_mean": float(np.mean(rho_s)) if rho_s else float("nan"),
                         "count_min_max": [float(n.min()), float(n.max())]}
        print(f"U={a.U} snr={snr} by_rank={np.round(by_rank, 2).tolist()} eq={eq:.2f} "
              f"weakest more/equal/fewer={res[str(snr)]['weakest_more']:.2f}/{res[str(snr)]['weakest_equal']:.2f}/"
              f"{res[str(snr)]['weakest_fewer']:.2f} spearman={res[str(snr)]['spearman_mean']:+.2f}", flush=True)
    os.makedirs(os.path.join(DATA, "pilot_corr"), exist_ok=True)
    json.dump({"args": vars(a), "res": res},
              open(os.path.join(DATA, "pilot_corr", f"pca_csi_counts_U{a.U}_nf{int(a.nearfar)}.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
