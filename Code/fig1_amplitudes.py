"""fig1_amplitudes.py - bar heights of the load-mask rows of Fig. 1 (fig1_system_rev.tex), 2026-11-17.

Computes the RMS amplitude of every slot of the two load-mask frames of the mask map in Fig. 1, from
the stored held-out AG News spectrum only (no simulation, no training):

  data/effdim_spectrum_test.npy  cumulative variance of the centered, l2-normalized BERT-base test
                                 embeddings (d_b = 768); lambda_i = its first differences, sum 1.
                                 (The training-pool eigenvalues of data/emb_ag_news_bert-base-uncased.pt,
                                 which define Lambda and (12) in the letter, give 1.708, 1.086, 0.853,
                                 0.720, 0.621, 0.521 for frame tau and 1.502, 0.936, 0.719, 0.592 for
                                 frame tau+1 and do NOT reproduce the figure; the test-split second
                                 moment of the same file matches the stored spectrum to 2e-4. The largest
                                 print difference is 0.08 mm (smallest tau bar) and 0.03 mm (tau+1) at
                                 0.85 columnwidth, below one 300-dpi pixel, so both rows stay on the
                                 held-out spectrum.)

Block d_s = K d_b = 3072 (K = 4), N = 24 slots of G = 128 uses, gamma = 10 dB (sigma^2 = 0.1).
At load U the user of rank j uses the slots s with s mod U = j, slot s carries segment floor(s/U) of
r_u = 1_K (x) c_u, that is, group q = floor(s/U) mod 6 of 128 coefficients. The allocation (12) of the
letter (reverse water-filling P_i = max(0, sqrt(lambda_i sigma^2/nu) - sigma^2), sum_{i<=d_u} P_i = d_s/U)
is split equally over the m_i copies of coefficient i, so a use of coefficient i has the mean power
P_i/m_i (mask value sqrt(P_i/(m_i lambda_i)) times the RMS sqrt(lambda_i) of c_{u,i}). The bar of a slot
carrying group q has the height sqrt(mean_{i in group q} P_i/m_i).

  frame tau,   U = 2 (light load): d_u = 768, m_i = 2, energy 1536, groups 0-5 ->
               1.746, 1.097, 0.851, 0.699, 0.577, 0.449   (reproduces \\ampT of fig1_system_rev.tex)
  frame tau+1, U = 6 (overload, U > K): d_u = 512 (4 slots per user), m_i = 1, energy 512, groups 0-3 ->
               1.519, 0.935, 0.708, 0.564                  (\\ampO of fig1_system_rev.tex)
               every code is cut after its leading 512 coefficients (groups 4 and 5 are not sent).

Usage: python3 Code/fig1_amplitudes.py   (prints both rows, checks both drawn rows \ampT and \ampO to 5e-4
       and checks that the mean power per used entry is 1, i.e. energy d_s/U on (N/U) G uses)
"""
import math
import os
import re

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
D_B, K, N_SLOTS, SNR_DB = 768, 4, 24, 10
D_S = K * D_B
G = D_S // N_SLOTS                       # uses per slot = coefficients per group (128)
AMP_T_FIG = (1.746, 1.097, 0.851, 0.699, 0.577, 0.449)   # \ampT as drawn in fig1_system_rev.tex
AMP_O_FIG = (1.519, 0.935, 0.708, 0.564)                 # \ampO as drawn in fig1_system_rev.tex


def spectrum():
    cum = np.load(os.path.join(ROOT, "data", "effdim_spectrum_test.npy")).astype(np.float64)
    return np.maximum(np.diff(np.concatenate([[0.0], cum])), 0.0)


def alloc(lmb, E, s2):
    """allocation (12): reverse water-filling with sum P_i = E, nu found by bisection."""
    lo, hi = 1e-12, 1e12
    for _ in range(300):
        nu = math.sqrt(lo * hi)
        P = np.maximum(0.0, np.sqrt(lmb * s2 / nu) - s2)
        lo, hi = (nu, hi) if P.sum() > E else (lo, nu)
    return P


def row(lam, U):
    """per-group RMS amplitude of the load masks at load U (every rank has N/U slots here)."""
    s2 = 10 ** (-SNR_DB / 10.0)
    M = (N_SLOTS // U) * G                     # entries of r_u sent by one user
    d_u = min(M, D_B)
    m = np.array([(M - i) // D_B + 1 for i in range(1, d_u + 1)], dtype=float)   # copies m_{u,i}
    P = alloc(lam[:d_u], D_S / U, s2)
    amp = [math.sqrt(np.mean(P[q * G:(q + 1) * G] / m[q * G:(q + 1) * G])) for q in range(d_u // G)]
    return d_u, m, P, amp


def drawn(name):
    """the bar heights as defined in fig1_system_rev.tex (\\def\\<name>{{...}}), so the constants above are checked
    against the figure source itself."""
    src = open(os.path.join(ROOT, "fig1_system_rev.tex")).read()
    mt = re.search(r"\\def\\" + name + r"\{\{([^}]*)\}\}", src)
    return tuple(float(x) for x in mt.group(1).split(",")) if mt else None


def main():
    for nm, const in (("ampT", AMP_T_FIG), ("ampO", AMP_O_FIG)):
        assert drawn(nm) == const, f"\\{nm} of fig1_system_rev.tex {drawn(nm)} differs from the constant {const}"
    print("constants AMP_T_FIG and AMP_O_FIG equal \\ampT and \\ampO of fig1_system_rev.tex")
    lam = spectrum()
    print(f"spectrum: {lam.size} eigenvalues, sum {lam.sum():.6f}")
    for U, name in ((2, "frame tau,   U = 2"), (6, "frame tau+1, U = 6")):
        d_u, m, P, amp = row(lam, U)
        used = (N_SLOTS // U) * G                      # entries of r_u sent by one user
        assert int(m.sum()) == used, "copies do not fill the user's uses"
        mean_pow = float(P.sum() / used)               # each copy of coefficient i carries P_i/m_i
        print(f"{name}: d_u = {d_u}, copies {sorted(set(int(x) for x in m))}, "
              f"energy {P.sum():.1f}, mean power per use {mean_pow:.6f}, active {int((P > 0).sum())}")
        print("   amplitudes per group:", ", ".join(f"{a:.3f}" for a in amp))
        assert abs(mean_pow - 1.0) < 1e-6, "mean power per use is not 1 (energy d_s/U on (N/U) G uses)"
        if U == 2:
            assert all(abs(a - b) < 5e-4 for a, b in zip(amp, AMP_T_FIG)), "ampT not reproduced"
            print("   reproduces \\ampT of fig1_system_rev.tex to 3 decimals")
        else:
            assert d_u == 512 and set(m) == {1.0}, "overload row: 512 coefficients, one copy each"
            assert len(amp) == len(AMP_O_FIG) and all(abs(a - b) < 5e-4 for a, b in zip(amp, AMP_O_FIG)), \
                "ampO not reproduced"
            print("   reproduces \\ampO of fig1_system_rev.tex to 3 decimals")


if __name__ == "__main__":
    main()
