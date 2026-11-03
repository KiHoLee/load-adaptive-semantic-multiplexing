# =========================================================
# pca_mc.py — Monte Carlo check of the closed forms (7), (10) and (12) of the letter (2026-11-03).
# For U = 1, 4, 6 and 0, 10, 20 dB, the user of rank 0 sends its expanded code e_u = p_U(gamma) (1_K x c_u)
# on its own slots (copies of coefficient i at amplitude sqrt(P_i/(m_i lambda_i)), P_i from the allocation
# (12)), each entry passes a known gain g = |h_u| (Rayleigh) and white noise of variance 1/SNR, the
# receiver adds the m_i copies and divides by sqrt(m_i) (eq. 7) and applies the LMMSE estimate with the
# known gain (eq. 9 at h_hat = g, zero for P_i = 0). The measured MSE of the d_u sent coefficients is
# compared with the mean over frames of the sum of (10). The energy normalization rho_u = 1 here, so g is
# independent of the coefficients, as assumed by (10).
#   source "train": coefficients of the training pool (second moments exactly lambda_i), checks (10)
#   source "test":  coefficients of the test set (training eigenvalues used by transmitter and receiver)
# Seeded after the model (V, lambda) is built, from a generator of its own (seed 999), 20,000 frames.
# Usage: python3 Code/pca_mc.py  ->  data/extra/pca_mc.json
# =========================================================
import json, math, os, sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pca_eval import Source, alloc, D_B, K, N_SLOTS, DATA  # noqa: E402

FR = 20000


def check(src, coeff, U, snr, rng):
    d_s = K * D_B
    seg = d_s // N_SLOTS
    M = math.ceil(N_SLOTS / U) * seg                 # entries of e_u sent by rank 0
    k = min(M, D_B)
    s2 = 10 ** (-snr / 10.0)
    lmb = src.lam[:k]
    P = alloc(lmb, d_s / U, s2, "opt")
    idx = np.arange(M) % D_B                          # coefficient carried by each sent entry
    m = np.bincount(idx, minlength=D_B)[:k]           # copies m_i
    amp = np.sqrt(P / (m * lmb))[idx]                 # entry t of p_U(gamma)
    pos = rng.integers(0, coeff.shape[0], FR)
    c = coeff[pos, :k]
    g = np.sqrt((rng.standard_normal(FR) ** 2 + rng.standard_normal(FR) ** 2) / 2.0)
    y = g[:, None] * (c[:, idx] * amp) + math.sqrt(s2) * rng.standard_normal((FR, M))
    z = np.zeros((FR, k))
    np.add.at(z.T, idx, y.T)                          # sum of the copies of each coefficient
    z = z / np.sqrt(m)
    a = np.sqrt(P / lmb)
    g2 = g[:, None] ** 2
    w = g2 * P / (g2 * P + s2)
    chat = np.where(a > 0, w * z / (np.where(a > 0, a, 1.0) * g[:, None]), 0.0)
    emp = float(((chat - c) ** 2).sum(1).mean())
    pred = float((lmb * s2 / (g2 * P + s2)).sum(1).mean())
    return {"d_u": int(k), "active": int((P > 0).sum()), "mse_measured": emp, "mse_closed_form": pred,
            "rel_gap": emp / pred - 1.0}


def main():
    src = Source()
    rng = np.random.default_rng(999)
    coeffs = {"train": src.Etr @ src.V, "test": src.ct}
    R = {"protocol": "rank-0 user, known Rayleigh gain, rho_u = 1, 20000 frames, own generator seed 999",
         "frames": FR}
    for name, cf in coeffs.items():
        R[name] = {U: {s: check(src, cf, U, s, rng) for s in (0, 10, 20)} for U in (1, 4, 6)}
    R["max_abs_rel_gap"] = {name: max(abs(v["rel_gap"]) for U in R[name].values() for v in U.values())
                            for name in coeffs}
    os.makedirs(os.path.join(DATA, "extra"), exist_ok=True)
    json.dump(R, open(os.path.join(DATA, "extra", "pca_mc.json"), "w"), indent=1)
    for name in coeffs:
        for U, v in R[name].items():
            print(name, U, {s: (round(x["mse_measured"], 5), round(x["mse_closed_form"], 5), f"{x['rel_gap']:+.4f}")
                            for s, x in v.items()})
    print("max |rel gap|", R["max_abs_rel_gap"])


if __name__ == "__main__":
    main()
