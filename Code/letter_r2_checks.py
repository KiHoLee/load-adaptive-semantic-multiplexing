"""Letter-only values of Comments 2.1-2.5 of the response letter, read from stored results only.

No training, no simulation: every number is a difference or a copy of values in data/.
Usage (from the project root):  python3 -I Code/letter_r2_checks.py [data_dir]

Prints
  1. Table R2.3: Proposed minus Static ODS-k at light load (k = 4, 6, 8) and minus the variants with
     energy d_s/U (k = 4, 6), from data/extra/pca_eval.json (proposal.opt, static, matched).
  2. Comment 2.5: Learned nested code without MAML (shared_dyns_joint_s1) against the MAML run
     (shared_dyns_clamp_s1) and against Proposed, U = 1..6, 0-30 dB (seed 1).
  3. Table R2.5a: Learned ODS-U (seed 1) trained on 0-25 dB (default), 0-10 dB (trlow), 15-25 dB (trhigh).
  4. Comment 2.4: Learned nested code with the load masks under subband gains (pilot_corr/prop_fixed_nf6_s1)
     minus the closed-form load masks (pilot_corr/pca_csi_U*_k6_rho0.7_nf6, mode fixed), and the
     channel-aware masks (mode csiq) minus that learned run.
"""
import json, os, sys

D = sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
S = ["0", "5", "10", "15", "20", "25", "30"]
ld = lambda p: json.load(open(os.path.join(D, p)))
v = lambda x: x["cos"] if isinstance(x, dict) else x

E = ld("extra/pca_eval.json")
P = E["proposal"]["opt"]

print("1. Table R2.3: Proposed minus fixed partition (3 decimals)")
for k, Us in (("4", "123"), ("6", "12345"), ("8", "1234")):
    for U in Us:
        print(f"  Static ODS-{k}, U={U}:", " ".join(f"{P[U][s] - v(E['static'][k][s]):+.3f}" for s in S))
for k, Us in (("4", "123"), ("6", "12345")):
    for U in Us:
        print(f"  Static ODS-{k} energy d_s/U, U={U}:", " ".join(f"{P[U][s] - v(E['matched'][k][U][s]):+.3f}" for s in S))
print(f"  Static ODS-8, U=1, 0 dB, 4 decimals: {P['1']['0'] - v(E['static']['8']['0']):+.4f}")

print("2. Learned nested code without MAML (seed 1)")
j = ld("shared_dyns_joint_s1.json")["eval"]
c = ld("shared_dyns_clamp_s1.json")["eval"]
gap = [(P[U][s] - j[U][s]["cos"], U, s) for U in j for s in S]
chg = [(c[U][s]["cos"] - j[U][s]["cos"], U, s) for U in j for s in S]
print("  Proposed minus no-MAML run: min %+.4f (U=%s, %s dB), max %+.4f (U=%s, %s dB)" % (min(gap) + max(gap)))
print("  MAML run minus no-MAML run: min %+.4f (U=%s, %s dB), max %+.4f (U=%s, %s dB)" % (min(chg) + max(chg)))

print("3. Table R2.5a: Learned ODS-U (seed 1) by training SNR range")
for U in ("1", "4", "6"):
    for tag in ("", "_trlow", "_trhigh"):
        e = ld(f"ortho_U{U}{tag}_s1.json")["eval"][U]
        print(f"  U={U} {tag or 'default'}:", " ".join(f"{e[s]['cos']:.3f}" for s in S))

print("4. Learned nested code with the load masks under subband gains (rho_f 0.7, +-6 dB, seed 1)")
pf = ld("pilot_corr/prop_fixed_nf6_s1.json")["cos"]["0.7"]
for U in "123456":
    r = ld(f"pilot_corr/pca_csi_U{U}_k6_rho0.7_nf6.json")["res"]
    print(f"  U={U} learned minus closed-form load masks:", " ".join(f"{pf[U][s] - v(r['fixed'][s]):+.4f}" for s in S),
          f"| channel-aware minus learned at 0 dB {v(r['csiq']['0']) - pf[U]['0']:+.4f}")
