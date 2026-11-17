"""Letter values that no other script prints, read from stored results only (no training, no simulation).

Usage (from the repository root):  python3 -I Code/letter_checks/letter_tables.py data

Prints
  1. Table R1.1: aggregate fidelity S(U) = U times the per-user CosSim of the served users at 0, 10 and 20 dB,
     computed as Section 9 of data/pca_numbers.md (10 dB): Proposed (extra/pca_eval.json, proposal.opt),
     Learned ODS-U (three-seed mean of ortho_U<U>_s<n>.json), Static ODS-4 (serves min(U, 4) users) and Static ODS-6.
  2. Table R2.7a: per-user CosSim of Proposed minus that of each retrained reference (seed 1) on AG News with BERT,
     SST-2 with BERT and AG News with DistilBERT (extra/pca_eval.json keys proposal.opt, sst2, distil, static,
     sst2_static6, distil_static6; ortho_U4_s1, ortho_U6_s1, sst2_ortho_U4_s1, sst2_ortho_U6_s1,
     distil_ortho_U4_s1, shared_dyns_clamp_s1, sst2_shared_dyns_s1, distil_shared_dyns_s1).
  3. Comment 2.1: the earlier nested variant trained for 300 epochs (shared_nest_e300_s1) minus its run of
     150 epochs (shared_nest_s1), seed 1, over every load and SNR of both files.
  4. Comment 2.6: share of frames in which the weakest user of a frame holds more than N/U subbands under the
     greedy subband counts of the channel-aware masks (field weakest_more of pilot_corr/pca_csi_counts_U<U>_nf6.json).
  5. Table R2.5b: training configuration stored in the field cfg of ortho_U4_s1.json (Learned ODS-k),
     shared_dyns_s1.json (Learned nested code) and shared_agn_s1.json (Learned shared embedding). The loss weight
     (lam, 0.5) and the 256 adaptation frames are defaults of Code/rev_lib.py (semantic_loss, adapt).
  6. Comment 2.6 (estimation errors at the receiver): CosSim of the overlapping masks with known gains minus that with
     gains estimated by EM from the frame without pilots (modes gen_joint_true0 and gen_joint_em0 of
     shared/pilot/shared_pilot_coco_K<K>_U<U>_E1_emauto0_fs_it30_snr<S>.json, natural captions), and the loss of the
     raw estimate from one pilot use per user (gen_joint_true minus gen_joint_pilot). Where the per-frame arrays
     (.npz) of a point are present (K = 4, U = 4, 0 dB), the paired bootstrap 95% CI is recomputed as in
     Code/shared_pilot.py: per-frame mean over the users, 1000 resamples numpy default_rng(555 + SNR). Part 6 needs
     numpy, parts 1 to 5 only the standard library.
"""
import glob, json, os, statistics as st, sys

D = sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))), "data")
S = ["0", "5", "10", "15", "20", "25", "30"]
ld = lambda p: json.load(open(os.path.join(D, p)))
cos = lambda x: x["cos"] if isinstance(x, dict) else x

E = ld("extra/pca_eval.json")
P = E["proposal"]["opt"]


def ods_mean(U, s):
    fs = sorted(glob.glob(os.path.join(D, f"ortho_U{U}_s[0-9].json")))
    return st.mean(json.load(open(f))["eval"][str(U)][s]["cos"] for f in fs), len(fs)


print("1. Table R1.1: S(U) at 0, 10 and 20 dB (U = 1..6)")
for s in ("0", "10", "20"):
    print(f"  {s} dB")
    print("    Proposed       ", " ".join(f"{U * P[str(U)][s]:.2f}" for U in range(1, 7)))
    print("    Learned ODS-U  ", " ".join(f"{U * ods_mean(U, s)[0]:.2f}" for U in range(1, 7)),
          f"(seeds per load: {[ods_mean(U, s)[1] for U in range(1, 7)]})")
    print("    Static ODS-4   ", " ".join(f"{min(U, 4) * cos(E['static']['4'][s]):.2f}" for U in range(1, 7)))
    print("    Static ODS-6   ", " ".join(f"{U * cos(E['static']['6'][s]):.2f}" for U in range(1, 7)))

print("\n2. Table R2.7a: Proposed minus reference (seed 1), 0 to 30 dB, three decimals")


def ev(name, U):
    e = ld(f"{name}_s1.json")["eval"][str(U)]
    return {s: e[s]["cos"] for s in S if s in e}


def line(label, prop, ref):
    print(f"  {label:<48}", " ".join(f"{prop[s] - ref[s]:+.3f}" if s in ref else "     -" for s in S))


line("Learned ODS-4, AG News, BERT, U = 4", P["4"], ev("ortho_U4", 4))
line("Learned ODS-4, SST-2, BERT, U = 4", E["sst2"]["4"], ev("sst2_ortho_U4", 4))
line("Learned ODS-4, AG News, DistilBERT, U = 4", E["distil"]["4"], ev("distil_ortho_U4", 4))
line("Learned ODS-6, AG News, BERT, U = 6", P["6"], ev("ortho_U6", 6))
line("Learned ODS-6, SST-2, BERT, U = 6", E["sst2"]["6"], ev("sst2_ortho_U6", 6))
line("Learned nested code, AG News, BERT, U = 4", P["4"], ev("shared_dyns_clamp", 4))
line("Learned nested code, SST-2, BERT, U = 4", E["sst2"]["4"], ev("sst2_shared_dyns", 4))
line("Learned nested code, AG News, DistilBERT, U = 4", E["distil"]["4"], ev("distil_shared_dyns", 4))
print("  (the letter prints a hyphen at 30 dB for the Learned nested code on SST-2 and with DistilBERT,"
      " above its training range of 0 to 25 dB)")
line("Static ODS-6, AG News, BERT, U = 1", P["1"], {s: cos(E["static"]["6"][s]) for s in S})
line("Static ODS-6, SST-2, BERT, U = 1", E["sst2"]["1"], {s: cos(E["sst2_static6"][s]) for s in S})
line("Static ODS-6, AG News, DistilBERT, U = 1", E["distil"]["1"], {s: cos(E["distil_static6"][s]) for s in S})

print("\n3. Comment 2.1: shared_nest_e300_s1 minus shared_nest_s1 (seed 1)")
a, b = ld("shared_nest_e300_s1.json"), ld("shared_nest_s1.json")
d = [a["eval"][U][s]["cos"] - b["eval"][U][s]["cos"] for U in a["eval"] if U in b["eval"]
     for s in S if s in a["eval"][U] and s in b["eval"][U]]
print(f"  epochs {a['cfg'].get('epochs')} against {b['cfg'].get('epochs')}: min {min(d):+.4f}, max {max(d):+.4f} "
      f"over {len(d)} points (loads {sorted(a['eval'], key=int)})")

print("\n4. Comment 2.6: weakest user holds more than N/U subbands (share of frames, offsets of up to 6 dB)")
for U in (2, 3, 4, 6):
    r = ld(f"pilot_corr/pca_csi_counts_U{U}_nf6.json")["res"]
    print(f"  U = {U}: " + ", ".join(f"{s} dB {r[s]['weakest_more']:.3f}" for s in sorted(r, key=float)))

print("\n5. Table R2.5b: stored training configuration (field cfg)")
K5 = ("mode", "inner_steps", "alpha", "beta", "epochs", "steps", "meta_batch", "batch", "train_snrs", "adapt_steps", "seed")
for f in ("ortho_U4_s1.json", "shared_dyns_s1.json", "shared_agn_s1.json"):
    c = ld(f)["cfg"]
    print(f"  {f}: " + ", ".join(f"{k} {c.get(k)}" for k in K5))

print("\n6. Comment 2.6: known gains minus EM gains without pilots, overlapping masks, natural captions (K, U, SNR)")
PT = "shared/pilot/shared_pilot_coco_K{K}_U{U}_E1_emauto0_fs_it30_snr{S}"
cost = {}
for K in (4, 2):
    for U in (4, 2):
        cells = []
        for S6 in ("-5", "0", "10", "20"):
            f = PT.format(K=K, U=U, S=S6)
            if not os.path.exists(os.path.join(D, f + ".json")):
                cells.append(f"{S6} dB missing"); continue
            r = ld(f + ".json")["res"]
            cost[(K, U, int(S6))] = r["gen_joint_true0"][S6]["cos"] - r["gen_joint_em0"][S6]["cos"]
            cells.append(f"{S6} dB {cost[(K, U, int(S6))]:+.5f}")
        print(f"  K = {K}, U = {U}: " + ", ".join(cells))
if cost:
    m = max(cost, key=cost.get)
    hi = [abs(v) for (K, U, s6), v in cost.items() if s6 >= 10]
    print(f"  largest cost {cost[m]:+.5f} (K = {m[0]}, U = {m[1]}, {m[2]} dB); from 10 dB on |cost| <= {max(hi):.5f}")
f = PT.format(K=4, U=4, S="0")
if os.path.exists(os.path.join(D, f + ".json")):
    r = ld(f + ".json")["res"]
    print(f"  raw pilot estimate, K = 4, U = 4, 0 dB: gen_joint_true minus gen_joint_pilot "
          f"{r['gen_joint_true']['0']['cos'] - r['gen_joint_pilot']['0']['cos']:+.4f}")
npz = [k for k in sorted(cost) if os.path.exists(os.path.join(D, PT.format(K=k[0], U=k[1], S=k[2]) + ".npz"))]
if npz:
    import numpy as np
    for K, U, s6 in npz:
        z = np.load(os.path.join(D, PT.format(K=K, U=U, S=s6) + ".npz"))
        dF = (z[f"cos_gen_joint_true0_{s6}"].astype(np.float64).mean(1)
              - z[f"cos_gen_joint_em0_{s6}"].astype(np.float64).mean(1))
        idx = np.random.default_rng(555 + s6).integers(0, dF.size, (1000, dF.size))
        lo, hi_ = np.percentile(dF[idx].mean(1), [2.5, 97.5])
        print(f"  paired bootstrap, K = {K}, U = {U}, {s6} dB, {dF.size} frames: {dF.mean():+.5f} [{lo:+.5f}, {hi_:+.5f}], "
              f"four decimals {dF.mean():.4f} [{lo:.4f}, {hi_:.4f}]")
else:
    print("  no per-frame arrays (.npz) present, so no CI")
