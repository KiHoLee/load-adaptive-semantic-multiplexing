# =========================================================
# content_stats.py — statistics of shared content in the caption embeddings (2026-11-16).
# Users that describe one scene (the five captions of one image, written by different people) send
# correlated embeddings. This script fits the content model (8) of the manuscript on the fit images only and
# measures how much of the content is shared, where it sits in the spectrum and how well the diagonal
# surrogate (A1) describes it.
# Model (content with shared components, (8) of the manuscript): lam, V eigenvalues / eigenvectors of the second moment of all
# fit vectors (as pca_eval.Source: descending, clamped at 1e-12), c = V^T b. For tuples of m vectors that
# share one scene (m = 5 captions of one image, or m = U mixtures),
#   C = mean over tuples and ordered pairs u != v of c_u c_v^T = (E[S S^T] - m Lam_emp) / (m (m - 1)),
#   S = sum of the m coefficient vectors, Lam_emp = V^T (second moment) V (diagonal up to rounding),
#   kappa_i = clip(C_ii / lam_i, 0, 0.99),  kappa_bar = sum_i lam_i kappa_i / sum_i lam_i.
# tr C / sum lam is the mean cosine of two captions of one image (unit embeddings); kappa_bar differs from it
# only by the clipping.
# Data: ../data/emb_<ds>_caps_bert-base-uncased.pt (Code/emb_captions.py): train (8000*5, 768), five
# consecutive rows per fit image; test (n_test, 5, 768).
# Measured, for coco and flickr:
#   spectrum      participation ratio, energy per 128-group, leading eigenvalues, smallest eigenvalue
#   kappa         per 128-group (mean and lam-weighted), leading coefficients, kappa_bar with a bootstrap
#                 95% CI over fit images (2000 resamples, V fixed, lam and C_ii re-estimated), the mean
#                 pair CosSim (= tr C / sum lam) with its CI, the same on the test images (V of the fit)
#   C             off-diagonal Frobenius share ||C - diag C||_F^2 / ||C||_F^2 (and the norm ratio), on the two
#                 halves of the fit images, the share of tr C and of ||C||_F^2 in the first group, smallest
#                 eigenvalues of C, Lam - C and Lam + 4 C (relative to lam_1)
#   control       captions of five different images: tuple k holds, for user u, image j = sigma[(k + u) mod n] with
#                 caption slot (u + r_j) mod 5 (sigma a random permutation of the fit images, r_j a random slot
#                 offset per image, default_rng(0), permutation first); every caption is used once, so lam, V are
#                 those of the fit, and the users' slots are independent and uniform, as in the measured control
#                 frames of shared_masks.py (random slot per user). The same statistics; ||C_ctrl||_F is the
#                 estimation-noise floor of C plus the rank-one term of the overall caption mean, and the
#                 off-diagonal share after subtracting it is an approximate correction. Also stored: the
#                 slot-aligned control of the first version (user u always slot u; 2026-11-16 16:52), whose pair
#                 cosine carries the caption-slot means (Flickr: slots 0 and 4), and the slot means themselves
#                 (||mu_s||^2, ||mean||^2, mean over s != t of mu_s.mu_t)
#   mix fits      controlled mixtures (--mix), coco, U = 4 users per tuple, kappa in
#                 {0, 0.2, 0.4, 0.6, 0.8, 0.9, 0.95}: b_u = normalize(sqrt(kappa) b_s + sqrt(1 - kappa) b_{p,u}),
#                 b_s one caption of the fit image, b_{p,u} captions of U other, distinct fit images, one tuple
#                 per fit image, default_rng(0) (the same draws, in the same order, as
#                 overlap_bound.fit_content); lam, V, C fitted on the 8000 U mixtures with m = U; the
#                 statistics above plus ||C - kappa Lam||_F / ||kappa Lam||_F and the deviation of kappa_i from
#                 kappa; and the difference to the fit of shared_masks.Content (mix_tuples_sm: 5 users per tuple,
#                 its draw order), ||V C V^T - V' C' V'^T||_F / ||V C V^T||_F and its kappa_bar
# Importable helpers (used by shared_theory.py): load_caps, mix_tuples, mix_tuples_sm, control_tuples, fit,
# fit_content (mix_fit "spec": U per tuple as above; "shared_masks": the fit mixtures of shared_masks.py).
# Output: ../data/shared/content_stats.json (and a printed summary). CPU only, about a minute.
# Usage: python content_stats.py            (all of the above)
#        python content_stats.py --ds coco --no_mix
# =========================================================
import argparse, hashlib, json, math, os, time
import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(os.path.dirname(HERE), "data")
CAPS = {"coco": "emb_coco_caps_bert-base-uncased.pt", "flickr": "emb_flickr_caps_bert-base-uncased.pt"}
SEG, NB = 128, 6
MIX_KAPPAS = (0.0, 0.2, 0.4, 0.6, 0.8, 0.9, 0.95)
CLIP = 0.99


def load_caps(ds, split="train"):
    """-> (n_images, 5, d) float64 unit caption embeddings of the fit (train) or test images."""
    X = torch.load(os.path.join(DATA, CAPS[ds]), map_location="cpu", weights_only=False)[split].double().numpy()
    return X.reshape(-1, 5, X.shape[-1]) if split == "train" else X


def mix_tuples(X, kappa, U, seed=0):
    """controlled mixtures, one tuple of U users per image of X (n, 5, d) -> (n, U, d). The draws and their order
    are those of overlap_bound.fit_content: shared caption index, U other distinct images per image, their
    caption indices."""
    rng = np.random.default_rng(seed)
    n = X.shape[0]
    sh = X[np.arange(n), rng.integers(0, 5, n)]                              # shared caption of the image
    oth = np.stack([o + (o >= m) for m, o in ((m, rng.choice(n - 1, U, replace=False)) for m in range(n))])
    pv = X[oth, rng.integers(0, 5, (n, U))]                                  # captions of U other images
    T = math.sqrt(kappa) * sh[:, None, :] + math.sqrt(1.0 - kappa) * pv
    return T / np.linalg.norm(T, axis=2, keepdims=True)


def mix_tuples_sm(X, kappa, m=5, seed=0):
    """the fit mixtures of shared_masks.Content (mixtures(tr, kappa, 5, default_rng(0), arange(n), n)): m users per
    image, draws in its order (m other distinct images per image, shared caption index, other caption indices)."""
    rng = np.random.default_rng(seed)
    n = X.shape[0]
    idx = np.arange(n)
    o = np.stack([rng.choice(n - 1, m, replace=False) for _ in idx])
    oth = o + (o >= idx[:, None])
    sc = rng.integers(0, 5, n)
    pc = rng.integers(0, 5, (n, m))
    T = math.sqrt(kappa) * X[idx, sc][:, None, :] + math.sqrt(1.0 - kappa) * X[oth, pc]
    return T / np.linalg.norm(T, axis=-1, keepdims=True)


def control_tuples(X, seed=0, slots="random"):
    """captions of five different images -> (n, 5, d). Tuple k, user u: image j = sigma[(k + u) mod n]; caption slot
    (u + r_j) mod 5 with a random offset r_j per image (slots="random": independent uniform slots, every caption once)
    or slot u (slots="aligned", the first version: carries the caption-slot means into the pair cosine)."""
    n = X.shape[0]
    rng = np.random.default_rng(seed)
    sig = rng.permutation(n)
    k = np.arange(n)
    if slots == "aligned":
        return np.stack([X[sig[(k + u) % n], u] for u in range(5)], 1)
    r = rng.integers(0, 5, n)
    return np.stack([X[sig[(k + u) % n], (u + r[sig[(k + u) % n]]) % 5] for u in range(5)], 1)


def slot_means(X):
    """caption-slot means mu_s (over images) of X (n, 5, d): the pair cosine of captions of different images is
    mean_{s != t} mu_s.mu_t for slot-aligned users and ||mean_s mu_s||^2 for independent uniform slots."""
    mu = X.mean(0)
    G = mu @ mu.T
    mb = mu.mean(0)
    return {"slot_mean_sq": np.diag(G).tolist(), "overall_mean_sq": float(mb @ mb),
            "pair_term_aligned_slots": float((G.sum() - np.trace(G)) / 20.0), "pair_term_random_slots": float(mb @ mb)}


def fit(T, V=None):
    """second-order fit of tuples T (n, m, d) -> dict lam (d,), V (d, d), C (d, d), kappa (d,), m, lam_raw.
    With V given, the eigenbasis is kept and lam = diag of the second moment in it (control tuples)."""
    n, m, d = T.shape
    Y = T.reshape(-1, d)
    Sig = Y.T @ Y / Y.shape[0]
    if V is None:
        lam_raw, V = np.linalg.eigh(Sig)
        o = np.argsort(lam_raw)[::-1]
        lam_raw, V = lam_raw[o], V[:, o]
    Lemp = V.T @ Sig @ V
    lam_raw = np.diag(Lemp).copy()
    S = T.sum(1) @ V                                                         # (n, d) summed coefficients
    C = (S.T @ S / n - m * Lemp) / (m * (m - 1))
    C = (C + C.T) / 2
    lam = np.maximum(lam_raw, 1e-12)
    kappa = np.clip(np.diag(C) / lam, 0.0, CLIP)
    return {"lam": lam, "lam_raw": lam_raw, "V": V, "C": C, "kappa": kappa, "m": m, "n": n}


def fit_content(ds="coco", mix=None, U=4, mix_fit="spec"):
    """fit used by the analysis lines: same-image captions of ds (m = 5), or mixtures of ds: mix_fit "spec" U users per
    tuple (mix_tuples, the order of overlap_bound.fit_content), "shared_masks" 5 per tuple (mix_tuples_sm)."""
    X = load_caps(ds)
    if mix is None:
        T = X
    elif mix_fit == "shared_masks":
        T = mix_tuples_sm(X, float(mix))
    else:
        assert mix_fit == "spec", mix_fit
        T = mix_tuples(X, float(mix), U)
    F = fit(T)
    F.update({"ds": ds, "mix": mix, "tuple": T.shape[1], "mix_fit": mix_fit if mix is not None else None})
    return F


def per_tuple(T, V):
    """per tuple: mean c_i^2 over its m vectors, mean c_{u,i} c_{v,i} over ordered pairs, mean pair cosine."""
    m = T.shape[1]
    c = T @ V
    q = (c ** 2).sum(1)
    s = c.sum(1)
    G = np.einsum("nud,nvd->nuv", T, T)
    pc = (G.sum((1, 2)) - np.trace(G, axis1=1, axis2=2)) / (m * (m - 1))
    return q / m, (s ** 2 - q) / (m * (m - 1)), pc


def kbar_of(lam_i, Cd):
    lam_i = np.maximum(lam_i, 1e-12)
    return (lam_i * np.clip(Cd / lam_i, 0.0, CLIP)).sum(-1) / lam_i.sum(-1)


def bootstrap(T, V, B=2000, seed=1):
    """kappa_bar and mean pair CosSim with 95% percentile CIs over tuples (V fixed)."""
    lt, ct, pc = per_tuple(T, V)
    n = lt.shape[0]
    rng = np.random.default_rng(seed)
    kb, pm = [], []
    for b0 in range(0, B, 250):
        W = rng.multinomial(n, np.full(n, 1.0 / n), size=min(250, B - b0)).astype(float) / n
        kb.append(kbar_of(W @ lt, W @ ct))
        pm.append(W @ pc)
    kb, pm = np.concatenate(kb), np.concatenate(pm)
    return {"kappa_bar": float(kbar_of(lt.mean(0), ct.mean(0))),
            "kappa_bar_ci95": [float(np.percentile(kb, 2.5)), float(np.percentile(kb, 97.5))],
            "pair_cos": float(pc.mean()),
            "pair_cos_ci95": [float(np.percentile(pm, 2.5)), float(np.percentile(pm, 97.5))], "resamples": B}


def offdiag(C):
    f2 = float((C ** 2).sum())
    o2 = f2 - float((np.diag(C) ** 2).sum())
    return {"frob": math.sqrt(f2), "frob_offdiag": math.sqrt(max(o2, 0.0)), "share_sq": o2 / f2,
            "share_norm": math.sqrt(max(o2, 0.0) / f2)}


def summary(F, T, boot=True):
    lam, C, kap = F["lam"], F["C"], F["kappa"]
    d = lam.size
    g = lambda x: x.reshape(NB, SEG)
    tot = float(lam.sum())
    trC = float(np.trace(C))
    od = offdiag(C)
    blk1 = float((C[:SEG, :SEG] ** 2).sum()) / od["frob"] ** 2
    Lam = np.diag(F["lam_raw"])
    m = F["m"]
    R = {"n_tuples": int(F["n"]), "m": int(m), "lam_sum": tot,
         "kappa_bar": float((lam * kap).sum() / tot), "trC_over_lam": trC / tot,
         "kappa_group_mean": g(kap).mean(1).tolist(),
         "kappa_group_lamw": (g(lam * kap).sum(1) / g(lam).sum(1)).tolist(),
         "kappa_lead10": kap[:10].tolist(), "kappa_1": float(kap[0]),
         "n_kappa_clipped_low": int((np.diag(C) / lam < 0).sum()), "n_kappa_clipped_high": int((np.diag(C) / lam > CLIP).sum()),
         "trC_share_group1": float(np.trace(C[:SEG, :SEG])) / trC if trC != 0 else None,
         "frob_share_block1": blk1, "offdiag": od,
         "eig_min_rel": {"C": float(np.linalg.eigvalsh(C)[0] / lam[0]),
                         "Lam_minus_C": float(np.linalg.eigvalsh(Lam - C)[0] / lam[0]),
                         "Lam_clamped_minus_C": float(np.linalg.eigvalsh(np.diag(lam) - C)[0] / lam[0]),
                         "Lam_plus_m1_C": float(np.linalg.eigvalsh(Lam + (m - 1) * C)[0] / lam[0])},
         "eig_max_rel_C": float(np.linalg.eigvalsh(C)[-1] / lam[0])}
    if boot:
        R["bootstrap"] = bootstrap(T, F["V"])
    return R


def spectrum(lam):
    tot = lam.sum()
    return {"participation_ratio": float(tot ** 2 / (lam ** 2).sum()), "lam_sum": float(tot),
            "group_energy": (lam.reshape(NB, SEG).sum(1) / tot).tolist(), "lead5": lam[:5].tolist(),
            "lam_min": float(lam[-1]), "lam_tail5_rel": (lam[-5:] / lam[0]).tolist(),
            "cum_energy_128_256_384": [float(lam[:k].sum() / tot) for k in (128, 256, 384)]}


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for blk in iter(lambda: f.read(1 << 22), b""):
            h.update(blk)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ds", nargs="+", default=["coco", "flickr"])
    ap.add_argument("--no_mix", action="store_true")
    ap.add_argument("--mix_ds", default="coco")
    ap.add_argument("--mix_U", type=int, default=4)
    ap.add_argument("--out", default=os.path.join(DATA, "shared", "content_stats.json"))
    a = ap.parse_args()
    t0 = time.time()
    torch.set_num_threads(2)
    R = {"protocol": "fit images only (8000 x 5 captions); lam, V from the second moment of all fit vectors; "
                     "C over ordered pairs of one tuple; kappa_i = clip(C_ii/lam_i, 0, 0.99); bootstrap over tuples "
                     "with V fixed, 2000 resamples, seed 1; control: captions of 5 different images with random "
                     "independent slots (seed 0); mix: U per tuple (overlap_bound order), seed 0",
         "version": "2026-11-16 fix: control with random slots, slot means, aligned control, shared_masks mix fit",
         "provenance": {}}
    for ds in a.ds:
        p = os.path.join(DATA, CAPS[ds])
        st = os.stat(p)
        R["provenance"][ds] = {"file": CAPS[ds], "bytes": st.st_size, "mtime": time.ctime(st.st_mtime), "sha256": sha256(p)}
        X = load_caps(ds)
        F = fit(X)
        r = {"spectrum": spectrum(F["lam"]), "same_image": summary(F, X)}
        h = X.shape[0] // 2
        r["same_image"]["offdiag_halves"] = [offdiag(fit(X[:h], F["V"])["C"])["share_sq"],
                                             offdiag(fit(X[h:], F["V"])["C"])["share_sq"]]
        Tc = control_tuples(X)
        Fc = fit(Tc, F["V"])
        r["control_diff_images"] = summary(Fc, Tc)
        r["control_diff_images"]["slots"] = "random (u + r_j) mod 5, independent uniform per user"
        fs, fc = r["same_image"]["offdiag"], r["control_diff_images"]["offdiag"]
        r["control_diff_images"]["frob_ratio_to_same"] = fc["frob"] / fs["frob"]
        r["same_image"]["offdiag_share_sq_noise_corrected"] = (fs["frob_offdiag"] ** 2 - fc["frob_offdiag"] ** 2) / \
            (fs["frob"] ** 2 - fc["frob"] ** 2)
        Ta = control_tuples(X, slots="aligned")
        Fa = fit(Ta, F["V"])
        ba = bootstrap(Ta, F["V"])
        r["control_aligned_slots"] = {"slots": "user u always slot u (first version)", "pair_cos": ba["pair_cos"],
                                      "pair_cos_ci95": ba["pair_cos_ci95"],
                                      "frob_ratio_to_same": offdiag(Fa["C"])["frob"] / fs["frob"]}
        r["slot_means"] = slot_means(X)
        r["notes"] = {"control": "pair cosine of different images = noise + overall-mean term ||mean||^2 (random slots); "
                                 "quote the unclipped pair_cos for the control: the clipped kappa_bar is biased "
                                 "upward near 0 and its percentile CI can exclude the point estimate",
                      "offdiag_share_sq_noise_corrected": "approximate: subtracts ||C_ctrl||_F^2, i.e. the estimation "
                                                          "noise plus the rank-one overall-mean term",
                      "offdiag": "share_sq = ||C - diag C||_F^2 / ||C||_F^2 (energy share); share_norm = its square "
                                 "root (norm ratio ||C - diag C||_F / ||C||_F)"}
        Xt = load_caps(ds, "test")
        r["test_same_image"] = {"n_images": int(Xt.shape[0]), **bootstrap(Xt, F["V"])}
        R[ds] = r
        s, b = r["same_image"], r["same_image"]["bootstrap"]
        print(f"[{ds}] lam tail / lam_1 {np.array(r['spectrum']['lam_tail5_rel'])}", flush=True)
        print(f"[{ds}] PR {r['spectrum']['participation_ratio']:.1f}  kappa_bar {b['kappa_bar']:.4f} "
              f"CI [{b['kappa_bar_ci95'][0]:.4f}, {b['kappa_bar_ci95'][1]:.4f}]  pair cos {b['pair_cos']:.4f} "
              f"(test {r['test_same_image']['pair_cos']:.4f})  kappa_1 {s['kappa_1']:.3f}", flush=True)
        print(f"   kappa per group mean {np.round(s['kappa_group_mean'], 3).tolist()}  lam-w "
              f"{np.round(s['kappa_group_lamw'], 3).tolist()}", flush=True)
        print(f"   offdiag share {fs['share_sq']:.3f} (halves {np.round(s['offdiag_halves'], 3).tolist()}, noise-corrected "
              f"{s['offdiag_share_sq_noise_corrected']:.3f})  trC in group 1 {s['trC_share_group1']:.3f}  "
              f"eig_min_rel {s['eig_min_rel']}", flush=True)
        cb = r["control_diff_images"]["bootstrap"]
        print(f"   control: kappa_bar {cb['kappa_bar']:.4f} CI [{cb['kappa_bar_ci95'][0]:.4f}, {cb['kappa_bar_ci95'][1]:.4f}] "
              f"pair cos {cb['pair_cos']:.4f} CI [{cb['pair_cos_ci95'][0]:.4f}, {cb['pair_cos_ci95'][1]:.4f}]  "
              f"||C_ctrl||/||C|| {r['control_diff_images']['frob_ratio_to_same']:.3f}", flush=True)
        ca, sm = r["control_aligned_slots"], r["slot_means"]
        print(f"   aligned-slot control: pair cos {ca['pair_cos']:.5f} CI [{ca['pair_cos_ci95'][0]:.5f}, "
              f"{ca['pair_cos_ci95'][1]:.5f}] ||C||-ratio {ca['frob_ratio_to_same']:.3f}; slot means |mu_s|^2 "
              f"{np.round(sm['slot_mean_sq'], 4).tolist()} pair term aligned {sm['pair_term_aligned_slots']:.5f} "
              f"random {sm['pair_term_random_slots']:.5f}", flush=True)
    if not a.no_mix:
        X = load_caps(a.mix_ds)
        R["mix"] = {"ds": a.mix_ds, "U": a.mix_U, "seed": 0, "tuple": a.mix_U, "mix_fit": "spec", "fits": {}}
        for k in MIX_KAPPAS:
            T = mix_tuples(X, k, a.mix_U)
            F = fit(T)
            s = summary(F, T)
            Lam = np.diag(F["lam"])
            s["rel_dev_C_from_kappaLam_over_Lam"] = float(np.linalg.norm(F["C"] - k * Lam) / np.linalg.norm(Lam))
            s["rel_dev_C_from_kappaLam"] = float(np.linalg.norm(F["C"] - k * Lam) / np.linalg.norm(k * Lam)) if k > 0 else None
            s["kappa_dev_lamw"] = float((F["lam"] * np.abs(F["kappa"] - k)).sum() / F["lam"].sum())
            s["kappa_dev_max_group1"] = float(np.abs(F["kappa"][:SEG] - k).max())
            s["spectrum"] = spectrum(F["lam"])
            Fs = fit(mix_tuples_sm(X, k))                                    # the fit of shared_masks.Content
            A1 = F["V"] @ F["C"] @ F["V"].T
            A2 = Fs["V"] @ Fs["C"] @ Fs["V"].T
            s["vs_shared_masks_fit"] = {"tuple": int(Fs["m"]), "kappa_bar": float((Fs["lam"] * Fs["kappa"]).sum() / Fs["lam"].sum()),
                                        "rel_dev_C_embedding_space": float(np.linalg.norm(A1 - A2) / np.linalg.norm(A1))}
            R["mix"]["fits"][str(k)] = s
            print(f"[mix {k}] kappa_bar {s['kappa_bar']:.4f} CI {np.round(s['bootstrap']['kappa_bar_ci95'], 4).tolist()} "
                  f"groups {np.round(s['kappa_group_lamw'], 3).tolist()} offdiag {s['offdiag']['share_sq']:.3f} "
                  f"||C-kLam||/||Lam|| {s['rel_dev_C_from_kappaLam_over_Lam']:.4f}; shared_masks fit kappa_bar "
                  f"{s['vs_shared_masks_fit']['kappa_bar']:.4f} ||dC||/||C|| {s['vs_shared_masks_fit']['rel_dev_C_embedding_space']:.4f}",
                  flush=True)
    R["runtime_s"] = time.time() - t0
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    json.dump(R, open(a.out, "w"), indent=1)
    print(f"wrote {a.out} in {R['runtime_s']:.0f} s", flush=True)


if __name__ == "__main__":
    main()
