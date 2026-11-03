# =========================================================
# pca_eval.py — every evaluation of the closed-form proposal (2026-11-03): a principal-component (PCA)
# nested code of the centered sentence embedding, cut by the dynamic masks of the manuscript, with
# MSE-optimal power allocation per load and SNR and a blind-gain LMMSE (Wiener) receiver.
#
# Protocol of the manuscript: test split, 2,000 frames per point and user, independent Rayleigh
# |h_u| ~ CN(0,1) magnitude per user and block, per-user energy, noise variance 1/SNR per real use,
# evaluation seed 123 + SNR (the same draws for every variant at one load and SNR, so the
# comparisons are paired). A user with n uses and energy E sends its leading k = min(n, d_b)
# coefficients; sending a coefficient over several uses at the same total energy gives the same
# LMMSE estimate, so k <= n uses carry all of it. The receiver estimates |h_u|^2 from the received
# energy of the k entries that carry the code and applies the Wiener factor per coefficient; no
# other channel knowledge is used.
#
# Power allocation (sum P_i = E):
#   opt    reverse water-filling P_i = max(0, sqrt(lambda_i s2 / mu) - s2), the LMMSE optimum at the
#          announced SNR for |h|^2 = 1
#   prop   P_i proportional to lambda_i (no SNR knowledge)
# Receivers: wiener (blind gain), genie (known gain), zf (blind gain, no Wiener factor).
#
# Usage: python pca_eval.py   ->  ../data/extra/pca_eval.json
# =========================================================
import json, math, os, time
import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(os.path.dirname(HERE), "data")
EVAL_SNRS = (0, 5, 10, 15, 20, 25, 30)
D_B, K, N_SLOTS, FRAMES = 768, 4, 24, 2000


class Source:
    def __init__(self, dataset="ag_news", model="bert-base-uncased"):
        tag = model.replace("/", "_")
        C = torch.load(os.path.join(DATA, f"emb_{dataset}_{tag}.pt"), map_location="cpu", weights_only=False)
        self.C = C
        self.Etr, self.Ete = C["train"].double().numpy(), C["test"].double().numpy()
        lam, V = np.linalg.eigh(self.Etr.T @ self.Etr / self.Etr.shape[0])   # training pool only
        o = np.argsort(lam)[::-1]
        self.lam, self.V = np.maximum(lam[o], 1e-12), V[:, o]
        self.ct = self.Ete @ self.V
        self.d_b = self.V.shape[0]


def alloc(lmb, E, s2, mode):
    if mode == "prop":
        return E * lmb / lmb.sum()
    lo, hi = 1e-12, 1e12
    for _ in range(200):
        mu = math.sqrt(lo * hi)
        P = np.maximum(0.0, np.sqrt(lmb * s2 / mu) - s2)
        lo, hi = (mu, hi) if P.sum() > E else (lo, mu)
    return P


def user(src, n, E, snr, rng, mode="opt", rx="wiener", h=None, phase_deg=0.0, offset_db=0.0, pos=None):
    """returns (b_hat, pos) for FRAMES frames of one user with n uses and energy E."""
    s2 = 10 ** (-snr / 10.0)
    k = min(n, src.d_b)
    lmb = src.lam[:k]
    P = alloc(lmb, E, s2, mode)
    a = np.sqrt(P / lmb)
    if pos is None:
        pos = rng.integers(0, src.ct.shape[0], FRAMES)
    x = src.ct[pos, :k] * a
    x = x * np.sqrt(E / ((x ** 2).sum(1, keepdims=True) + 1e-12))             # energy E per frame
    if h is None:
        h = np.sqrt((rng.standard_normal(FRAMES) ** 2 + rng.standard_normal(FRAMES) ** 2) / 2.0)
    g = h * 10 ** (offset_db / 20.0)
    if phase_deg > 0:
        g = g * np.cos(np.radians(phase_deg) * rng.standard_normal(FRAMES))
    y = g[:, None] * x + math.sqrt(s2) * rng.standard_normal(x.shape)
    if rx == "genie":
        h2, den = g ** 2, g
    else:                                     # blind estimate from the received energy of the k used entries
        h2 = np.maximum(((y ** 2).sum(1) - k * s2) / E, 1e-6)
        den = np.sqrt(h2)
    w = (a > 0).astype(float)[None, :] if rx == "zf" else (h2[:, None] * P) / (h2[:, None] * P + s2)
    chat = np.where(a > 0, y / np.where(a > 0, a, 1.0) * w / den[:, None], 0.0)
    return chat @ src.V[:, :k].T, pos


def cos(bh, b):
    return (bh * b).sum(1) / (np.linalg.norm(bh, axis=1) * np.linalg.norm(b, axis=1) + 1e-12)


def per_load(src, U, snr, d_s, **kw):
    """mean per-user CosSim of the proposal at load U: rank j has ceil((N - j)/U) slots."""
    rng = np.random.default_rng(123 + snr)
    seg = d_s // N_SLOTS
    cs = []
    for j in range(U):
        n = math.ceil((N_SLOTS - j) / U) * seg
        bh, pos = user(src, n, d_s / U, snr, rng, **kw)
        cs.append(cos(bh, src.Ete[pos]))
    return float(np.mean(np.concatenate(cs)))


def static(src, k, snr, d_s, E=None, **kw):
    """Static PCA-k: n = d_s/k uses, energy d_s/k (or E), the same at every load U <= k."""
    rng = np.random.default_rng(123 + snr)
    bh, pos = user(src, d_s // k, E if E is not None else d_s / k, snr, rng, **kw)
    return float(np.mean(cos(bh, src.Ete[pos])))


def width(src, n, E, snr):
    s2 = 10 ** (-snr / 10.0)
    P = alloc(src.lam[:min(n, src.d_b)], E, s2, "opt")
    return int((P > 0).sum()), float(P.sum() ** 2 / (P ** 2).sum())


def classifier(src, seed=0, epochs=300, lr=1e-2):
    torch.manual_seed(seed)
    X = torch.tensor(src.Etr, dtype=torch.float32); y = src.C["train_labels"]
    W = torch.nn.Linear(X.shape[1], int(y.max()) + 1)
    opt = torch.optim.Adam(W.parameters(), lr=lr)
    for _ in range(epochs):
        opt.zero_grad(); torch.nn.functional.cross_entropy(W(X), y).backward(); opt.step()
    with torch.no_grad():
        clean = (W(torch.tensor(src.Ete, dtype=torch.float32)).argmax(-1) == src.C["test_labels"]).float().mean().item()
    return W, clean


def accuracy(src, W, U, snr, d_s):
    rng = np.random.default_rng(123 + snr)
    seg = d_s // N_SLOTS
    hits = []
    for j in range(U):
        bh, pos = user(src, math.ceil((N_SLOTS - j) / U) * seg, d_s / U, snr, rng)
        with torch.no_grad():
            pred = W(torch.tensor(bh, dtype=torch.float32)).argmax(-1)
        hits.append((pred == src.C["test_labels"][torch.as_tensor(pos)]).float().numpy())
    return float(np.mean(np.concatenate(hits)))


def main():
    t0 = time.time()
    d_s = K * D_B
    src = Source()
    R = {"protocol": "test split, 2000 frames per point and user, independent Rayleigh, eval seed 123+snr",
         "lambda_participation_ratio": float(src.lam.sum() ** 2 / (src.lam ** 2).sum())}
    # proposal and its variants at every load
    R["proposal"] = {}
    for name, kw in {"opt": {}, "prop": {"mode": "prop"}, "genie": {"rx": "genie"}, "zf": {"rx": "zf"}}.items():
        R["proposal"][name] = {U: {s: per_load(src, U, s, d_s, **kw) for s in EVAL_SNRS} for U in (1, 2, 3, 4, 5, 6, 8)}
    # static PCA-k (n = E = d_s/k), and the matched-energy Static PCA-6 / PCA-4 (energy d_s/U on its part)
    R["static"] = {k: {s: static(src, k, s, d_s) for s in EVAL_SNRS} for k in (1, 2, 3, 4, 5, 6, 8)}
    R["static_prop"] = {k: {s: static(src, k, s, d_s, mode="prop") for s in EVAL_SNRS} for k in (4, 6)}
    R["matched"] = {kk: {U: {s: static(src, kk, s, d_s, E=d_s / U) for s in EVAL_SNRS} for U in range(1, kk)} for kk in (4, 6)}
    # expansion factor K (d_s = K d_b) at U = 1 and 6
    R["K"] = {k2: {U: {s: per_load(src, U, s, k2 * D_B) for s in EVAL_SNRS} for U in (1, 6)} for k2 in (1, 2, 8)}
    # effective width of the allocation (active coefficients, participation ratio) at U = 1, 4, 6
    R["width"] = {U: {s: width(src, d_s // U, d_s / U, s) for s in EVAL_SNRS} for U in (1, 4, 6)}
    # robustness at U = 4 (the static partition Static PCA-4 is identical at U = 4)
    R["robust"] = {
        "phase30": {s: per_load(src, 4, s, d_s, phase_deg=30.0) for s in EVAL_SNRS},
        "phase30_static4": {s: static(src, 4, s, d_s, phase_deg=30.0) for s in EVAL_SNRS},
    }
    nf = {}
    for s in EVAL_SNRS:                       # offsets -6, -2, +2, +6 dB over the four users
        rng = np.random.default_rng(123 + s)
        per = []
        for j, off in enumerate((-6.0, -2.0, 2.0, 6.0)):
            bh, pos = user(src, d_s // 4, d_s / 4, s, rng, offset_db=off)
            per.append(float(np.mean(cos(bh, src.Ete[pos]))))
        nf[s] = {"mean": float(np.mean(per)), "weakest": per[0], "per_user": per}
    R["robust"]["nearfar6"] = nf
    cf = {}
    for s in EVAL_SNRS:                       # common fading: one |h| for the four users of a frame
        rng = np.random.default_rng(123 + s)
        h = np.sqrt((rng.standard_normal(FRAMES) ** 2 + rng.standard_normal(FRAMES) ** 2) / 2.0)
        cs = []
        for _ in range(4):
            bh, pos = user(src, d_s // 4, d_s / 4, s, rng, h=h)
            cs.append(cos(bh, src.Ete[pos]))
        cf[s] = float(np.mean(np.concatenate(cs)))
    R["robust"]["common"] = cf
    # topic classifier on the recovered embeddings (U = 4)
    W, clean = classifier(src)
    R["classifier_topic"] = {"clean": clean, **{s: accuracy(src, W, 4, s, d_s) for s in (0, 10, 20)}}
    # other datasets and encoders (U = 1, 4, 6), sentiment classifier on SST-2
    for key, (ds, mdl) in {"sst2": ("sst2", "bert-base-uncased"), "distil": ("ag_news", "distilbert-base-uncased")}.items():
        s2c = Source(ds, mdl)
        R[key] = {U: {s: per_load(s2c, U, s, K * s2c.d_b) for s in EVAL_SNRS} for U in (1, 4, 6)}
        R[key + "_static6"] = {s: static(s2c, 6, s, K * s2c.d_b) for s in EVAL_SNRS}
        if key == "sst2":
            W2, clean2 = classifier(s2c)
            R["classifier_sentiment"] = {"clean": clean2, **{s: accuracy(s2c, W2, 4, s, K * s2c.d_b) for s in (0, 10, 20)}}
    # cost: stored coefficients and CPU time per frame and user at U = 4 (transmitter + receiver)
    x = src.ct[:256, :D_B]
    t1 = time.time()
    for _ in range(20):
        c = src.Ete[:256] @ src.V; _ = (c * 0.5) @ src.V.T
    R["cost"] = {"stored_coefficients": int(src.V.size + src.lam.size),
                 "cpu_ms_per_user_frame": (time.time() - t1) / (20 * 256) * 1e3}
    R["runtime_s"] = time.time() - t0
    os.makedirs(os.path.join(DATA, "extra"), exist_ok=True)
    json.dump(R, open(os.path.join(DATA, "extra", "pca_eval.json"), "w"), indent=1)
    for name in ("opt", "prop", "genie", "zf"):
        print(name, {U: round(v[10], 4) for U, v in R["proposal"][name].items()})
    print("static", {k: round(v[10], 4) for k, v in R["static"].items()})
    print("matched6", {U: round(v[0], 4) for U, v in R["matched"][6].items()})
    print("robust", {k: (v[10] if not isinstance(v[10], dict) else v[10]["weakest"]) for k, v in R["robust"].items()})
    print("topic", R["classifier_topic"], "sent", R.get("classifier_sentiment"))
    print("sst2", {U: round(v[10], 4) for U, v in R["sst2"].items()}, "distil", {U: round(v[10], 4) for U, v in R["distil"].items()})
    print("time", round(R["runtime_s"]), "s")


if __name__ == "__main__":
    main()
