# =========================================================
# pca_numbers.py — every number of the closed-form (PCA) version of the letter (2026-11-03), printed
# from the stored results into data/pca_numbers.md. Reads data/ only.
#   Proposal: data/extra/pca_eval.json (Code/pca_eval.py), PCA nested code + dynamic masks +
#   MSE-optimal power + blind-gain LMMSE receiver. Learned references (three-seed means unless
#   noted): learned nested code shared_dyns_clamp_s*, the same started from PCA shared_dyns_pcainit_s1,
#   Static ODS-k ortho_U<k>_s*, Overlapping masks shared_agn_s*, Static ODS-6 with energy d_s/U
#   extra/energy_ortho_U6_s*, ToDMA extra/todma_redim.json. Monte Carlo check of the closed forms:
#   extra/pca_mc.json (Code/pca_mc.py).
# =========================================================
import glob, json, os, statistics as st
HERE = os.path.dirname(os.path.abspath(__file__))
D = os.path.join(os.path.dirname(HERE), "data")
S = ["0", "5", "10", "15", "20", "25", "30"]
out = []
w = out.append


def ld(p):
    return json.load(open(os.path.join(D, p)))


def have(pat):
    return sorted(glob.glob(os.path.join(D, pat)))


def evals(name, U):
    return [[ld(os.path.relpath(f, D))["eval"][str(U)][s]["cos"] for s in S] for f in have(f"{name}_s[0-9].json")]


def mean(v):
    return [st.mean(c) for c in zip(*v)]


def row(label, v, fmt="{:.3f}"):
    w(f"| {label} | " + " | ".join(fmt.format(x) for x in v) + " |")


def hdr(first=""):
    w(f"| {first} | " + " | ".join(f"{c} dB" for c in S) + " |")
    w("|---" * (len(S) + 1) + "|")


def diff(a, b):
    return [x - y for x, y in zip(a, b)]


def rng_(v):
    return f"{min(v):+.3f} to {max(v):+.3f}"


E = ld("extra/pca_eval.json")
pr = {int(U): [E["proposal"]["opt"][U][s] for s in S] for U in E["proposal"]["opt"]}
var = {n: {int(U): [E["proposal"][n][U][s] for s in S] for U in E["proposal"][n]} for n in ("prop", "genie", "zf")}
stat = {int(k): [E["static"][k][s] for s in S] for k in E["static"]}
ods = {k: mean(evals(f"ortho_U{k}", k)) for k in range(1, 7)}
lrn = {U: mean(evals("shared_dyns_clamp", U)) for U in range(1, 7)}
pci = {U: [ld("shared_dyns_pcainit_s1.json")["eval"][str(U)][s]["cos"] for s in S] for U in range(1, 7)}
agn = {U: mean(evals("shared_agn", U)) for U in range(1, 7)}

w("# Numbers of the closed-form (PCA) version\n")
w(f"Participation ratio of the PCA spectrum of the training embeddings: {E['lambda_participation_ratio']:.1f} of 768\n")
w("## 1. Proposal (PCA nested code + dynamic masks), per-user CosSim\n")
hdr("load")
for U in sorted(pr):
    row(f"U={U}", pr[U])
w("\n## 2. Proposal minus learned references at load U (learned: three-seed means)\n")
hdr("reference, U")
allo, alll = [], []
for U in range(1, 7):
    d = diff(pr[U], ods[U]); allo += d; row(f"learned Static ODS-U, U={U}", d, "{:+.3f}")
for U in range(1, 7):
    d = diff(pr[U], lrn[U]); alll += d; row(f"learned nested code, U={U}", d, "{:+.3f}")
for U in range(1, 7):
    row(f"learned nested code from PCA (seed 1), U={U}", diff(pr[U], pci[U]), "{:+.3f}")
for U in range(1, 7):
    row(f"Overlapping masks, U={U}", diff(pr[U], agn[U]), "{:+.3f}")
w(f"\n- proposal minus learned Static ODS-U over U=1..6 and all SNRs: {rng_(allo)}")
w(f"- proposal minus learned nested code over U=1..6 and all SNRs: {rng_(alll)}")
w(f"- proposal minus learned nested code from PCA over U=1..6: {rng_(sum((diff(pr[U], pci[U]) for U in range(1, 7)), []))}")
w(f"- proposal minus Overlapping masks over U=1..6: {rng_(sum((diff(pr[U], agn[U]) for U in range(1, 7)), []))}")
w("\n## 3. Static partitions with the same closed-form code (Static PCA-k, n = E = d_s/k, any U <= k)\n")
hdr("k")
for k in sorted(stat):
    row(f"Static PCA-{k}", stat[k])
w("\n### proposal minus Static PCA-U at its design load (identical code; differences are sampling and the slot granularity at U=5)")
hdr("U")
for U in (1, 2, 3, 4, 5, 6, 8):
    row(f"U={U}", diff(pr[U], stat[U]), "{:+.4f}")
w("\n## 4. Light load: proposal minus fixed partitions at U < k\n")
hdr("reference, U")
for k in (6, 4):
    for U in range(1, k):
        row(f"Static PCA-{k}, U={U}", diff(pr[U], stat[k]), "{:+.3f}")
for k in (6, 4):
    for U in range(1, k):
        row(f"learned Static ODS-{k}, U={U}", diff(pr[U], ods[k]), "{:+.3f}")
w("\n## 5. Matched energy: proposal minus the fixed partition whose user spends d_s/U on its own part\n")
hdr("reference, U")
for kk in ("6", "4"):
    for U, v in E["matched"][kk].items():
        row(f"Static PCA-{kk} energy d_s/U, U={U}", diff(pr[int(U)], [v[s] for s in S]), "{:+.3f}")
fs = have("extra/energy_ortho_U6_s[0-9].json")
for U in range(1, 6):
    m = mean([[ld(os.path.relpath(f, D))["U"][str(U)]["per_user"][s]["cos"] for s in S] for f in fs])
    row(f"learned Static ODS-6 energy d_s/U, U={U}", diff(pr[U], m), "{:+.3f}")
w("\n## 6. Variants of the proposal (minus the proposal)\n")
hdr("variant, U")
for n, lab in (("prop", "power proportional to variance (no SNR)"), ("zf", "no Wiener factor"), ("genie", "known gain")):
    for U in (1, 4, 6):
        row(f"{lab}, U={U}", diff(var[n][U], pr[U]), "{:+.4f}")
w("\n### expansion factor K (d_s = K d_b)")
hdr("K, U")
for k2, d in E["K"].items():
    for U, v in d.items():
        row(f"K={k2}, U={U}", [v[s] for s in S])
row("K=4, U=1", pr[1]); row("K=4, U=6", pr[6])
w("\n### allocation width at U=1, 4, 6: active coefficients / participation ratio of the powers")
for U, d in E["width"].items():
    w(f"- U={U}: " + ", ".join(f"{s} dB {d[s][0]}/{d[s][1]:.0f}" for s in S))
w("\n## 7. ToDMA re-dimensioned (best T per load and SNR, single runs of 500 frames) minus proposal\n")
t = ld("extra/todma_redim.json")
hdr("U")
for U in (1, 2, 4, 6):
    best = [max(v[s]["cos"] for k, v in t.items() if k.startswith(f"U{U}_") and s in v) for s in S]
    row(f"U={U}", diff(best, pr[U]), "{:+.3f}")
w("\n## 8. Robustness and generality at U=4 (proposal; learned Static ODS-4 from ortho_U4_s1, seed 1)\n")
o = ld("ortho_U4_s1.json")
R = E["robust"]
hdr("condition")
row("proposal default", pr[4]); row("learned ODS-4 default", [o["eval"]["4"][s]["cos"] for s in S])
row("proposal common fading", [R["common"][s] for s in S]); row("learned ODS-4 common fading", [o["robust"]["test_common"][s] for s in S])
row("proposal phase 30 deg", [R["phase30"][s] for s in S]); row("learned ODS-4 phase 30 deg", [o["robust"]["phase30"][s] for s in S])
row("proposal near-far 6 dB weakest", [R["nearfar6"][s]["weakest"] for s in S]); row("learned ODS-4 near-far 6 dB weakest", [o["robust"]["nearfar6"][s]["worst"] for s in S])
row("proposal near-far 6 dB mean", [R["nearfar6"][s]["mean"] for s in S]); row("learned ODS-4 near-far 6 dB mean", [o["robust"]["nearfar6"][s]["cos"] for s in S])
for key, lab, of in (("sst2", "SST-2", "sst2_ortho_U4_s1.json"), ("distil", "DistilBERT", "distil_ortho_U4_s1.json")):
    row(f"proposal {lab}", [E[key]["4"][s] for s in S]); row(f"learned ODS-4 {lab}", [ld(of)["eval"]["4"][s]["cos"] for s in S])
row("proposal U=8", pr[8]); row("learned ODS-8 (s1)", [ld("ortho_U8_s1.json")["eval"]["8"][s]["cos"] for s in S])
row("Static PCA-8", stat[8])
w(f"\n- topic classifier: clean {E['classifier_topic']['clean']:.3f}; proposal U=4 at 0/10/20 dB "
  f"{E['classifier_topic']['0']:.3f}/{E['classifier_topic']['10']:.3f}/{E['classifier_topic']['20']:.3f}")
w(f"- sentiment classifier (SST-2): clean {E['classifier_sentiment']['clean']:.3f}; proposal U=4 at 0/10/20 dB "
  f"{E['classifier_sentiment']['0']:.3f}/{E['classifier_sentiment']['10']:.3f}/{E['classifier_sentiment']['20']:.3f}")
for k in ("20", "10"):
    if "probe_acc" in o["eval"]["4"][k]:
        w(f"- learned ODS-4 topic probe at {k} dB: {o['eval']['4'][k]['probe_acc']:.3f}")
# classifier accuracies of the proposal and of Learned ODS-4 (seed 1) at 0 / 10 / 20 dB, and their differences
o2 = ld("sst2_ortho_U4_s1.json")
C3 = ("0", "10", "20")
for lab, ce, oo in (("topic", E["classifier_topic"], o), ("sentiment (SST-2)", E["classifier_sentiment"], o2)):
    lp = [oo["eval"]["4"][k]["probe_acc"] for k in C3]
    w(f"- {lab} accuracy at 0/10/20 dB: proposal " + "/".join(f"{ce[k]:.3f}" for k in C3)
      + "; learned ODS-4 (seed 1) " + "/".join(f"{x:.3f}" for x in lp)
      + "; proposal minus learned " + "/".join(f"{ce[k] - x:+.4f}" for k, x in zip(C3, lp)))
w(f"- learned ODS-4 parameters (ortho_U4_s1): {o['params']}; proposal stored values / learned ODS-4 parameters: "
  f"{E['cost']['stored_coefficients'] / o['params']:.3f}")
w(f"- cost: {E['cost']['stored_coefficients']} stored coefficients (basis and variances); "
  f"{E['cost']['cpu_ms_per_user_frame']:.4f} ms CPU per user and frame for the two transforms")
MP = ld(os.path.join("extra", "mask_partition.json"))
w(f"- transmitted energy of a user in entries that carry at least 90 percent of their energy from that user "
  f"(three-seed means, data/extra/mask_partition.json): Overlapping masks {MP['shared_agn']['tx_own_share_mean']:.3f}; "
  f"random frozen masks, four users {MP['shared_U4_randmask']['tx_own_share_mean']:.3f}")
mrg = []
for a_, b_ in ((pr[4], [o["eval"]["4"][s]["cos"] for s in S]),
               ([R["common"][s] for s in S], [o["robust"]["test_common"][s] for s in S]),
               ([R["phase30"][s] for s in S], [o["robust"]["phase30"][s] for s in S]),
               ([R["nearfar6"][s]["weakest"] for s in S], [o["robust"]["nearfar6"][s]["worst"] for s in S]),
               ([E["sst2"]["4"][s] for s in S], [ld("sst2_ortho_U4_s1.json")["eval"]["4"][s]["cos"] for s in S]),
               ([E["distil"]["4"][s] for s in S], [ld("distil_ortho_U4_s1.json")["eval"]["4"][s]["cos"] for s in S]),
               (pr[8], [ld("ortho_U8_s1.json")["eval"]["8"][s]["cos"] for s in S])):
    mrg += [a_[S.index(k)] - b_[S.index(k)] for k in ("10", "20")]
w(f"- proposal minus learned over the listed conditions (default, common fading, phase 30 deg, near-far weakest, "
  f"SST-2, DistilBERT, U=8) at 10 and 20 dB: {min(mrg):+.4f} to {max(mrg):+.4f}")
w("\n## 9. Aggregate fidelity S(U) at 10 dB\n")
w("| U | proposal | learned ODS-U | Static PCA-4 (min(U,4) users) | Static PCA-6 |")
w("|---|---|---|---|---|")
for U in range(1, 7):
    w(f"| {U} | {U * pr[U][2]:.2f} | {U * ods[U][2]:.2f} | {min(U, 4) * stat[4][2]:.2f} | {U * stat[6][2]:.2f} |")
w("\n## 10. Table I of the letter: per-user CosSim at U=1 and U=6, 0 / 10 / 20 dB (absolute values)\n")
w("| row | U=1 0 dB | U=1 10 dB | U=1 20 dB | U=6 0 dB | U=6 10 dB | U=6 20 dB |")
w("|---|---|---|---|---|---|---|")
T3 = ("0", "10", "20")
I3 = [S.index(s) for s in T3]


def row2(label, a, b):
    w(f"| {label} | " + " | ".join(f"{x:.3f}" for x in list(a) + list(b)) + " |")


row2("Proposed", [pr[1][i] for i in I3], [pr[6][i] for i in I3])
row2("Static ODS-6 energy d_s/U (U=6: Static ODS-6)", [E["matched"]["6"]["1"][s] for s in T3], [stat[6][i] for i in I3])
row2("power proportional to variance", [var["prop"][1][i] for i in I3], [var["prop"][6][i] for i in I3])
row2("no Wiener factor", [var["zf"][1][i] for i in I3], [var["zf"][6][i] for i in I3])
row2("K=1", [E["K"]["1"]["1"][s] for s in T3], [E["K"]["1"]["6"][s] for s in T3])
row2("K=8", [E["K"]["8"]["1"][s] for s in T3], [E["K"]["8"]["6"][s] for s in T3])
row2("Learned nested code (three seeds)", [lrn[1][i] for i in I3], [lrn[6][i] for i in I3])
row2("Learned nested code from PCA (seed 1)", [pci[1][i] for i in I3], [pci[6][i] for i in I3])
row2("Learned ODS-U (three seeds)", [ods[1][i] for i in I3], [ods[6][i] for i in I3])
row2("Overlapping masks (three seeds)", [agn[1][i] for i in I3], [agn[6][i] for i in I3])
tb = {U: [max(v[s]["cos"] for k, v in t.items() if k.startswith(f"U{U}_") and s in v) for s in T3] for U in (1, 6)}
row2("ToDMA best T (single run)", tb[1], tb[6])
sds = []
for name, loads in (("ortho_U{}", None), ("shared_dyns_clamp", None), ("shared_agn", None)):
    for U in range(1, 7):
        v = evals(name.format(U) if "{}" in name else name, U)
        sds.append(max(st.stdev(c) for c in zip(*v)))
w(f"\n- largest sample SD over three seeds (Learned ODS-k at its load, learned nested code, Overlapping masks; U=1..6, all SNRs): {max(sds):.4f}")
w(f"- K=1 minus proposal at U=6, 10 dB: {E['K']['1']['6']['10'] - pr[6][2]:+.3f}; K=8 minus proposal at U=6, 10 dB: {E['K']['8']['6']['10'] - pr[6][2]:+.3f}")
w("\n## 11. Monte Carlo check of (7), (10), (12) (Code/pca_mc.py, rank-0 user, known gain, rho_u = 1, 20,000 frames, seed 999)\n")
MC = ld(os.path.join("extra", "pca_mc.json"))
w("| source, U | 0 dB measured / closed form | 10 dB | 20 dB |")
w("|---|---|---|---|")
for src_name in ("train", "test"):
    for U in ("1", "4", "6"):
        w(f"| {src_name}, U={U} | " + " | ".join(f"{MC[src_name][U][s]['mse_measured']:.5f} / {MC[src_name][U][s]['mse_closed_form']:.5f} ({MC[src_name][U][s]['rel_gap']:+.2%})" for s in ("0", "10", "20")) + " |")
w(f"\n- largest |measured / closed form - 1|: training sentences {MC['max_abs_rel_gap']['train']:.2%}, test sentences {MC['max_abs_rel_gap']['test']:.2%}")

# ---- 12. channel-aware masks under time-correlated fading (Code/pca_csi.py, data/pilot_corr/pca_csi_*.json)
w("\n## 12. Channel-aware masks under time-correlated fading (Code/pca_csi.py)\n")
w("AR(1) complex Gaussian gain per user over the 24 slots (rho between adjacent slots), independent across users, "
  "near-far offset uniform in [-6, 6] dB per user and frame where marked nf6; the scheduler knows the slot gains "
  "(perfect, or delayed with correlation rho_b); blind per-slot gain estimate; 2,000 frames per user, draws shared "
  "by all schemes of a scenario (seed 123 + SNR). Files without the suffix _x hold fixed, csi, csiq, ods, odsfix; "
  "files with _x hold odsm (and csic), files with _e hold csiqe.")
w("Schemes: csiq = Proposed, channel-aware masks (greedy slot counts by the total MSE, energy 128 per granted slot); "
  "csi = channel-aware placement with the slot counts of the load masks (energy d_s/U); fixed = Proposed with the "
  "load masks (4), no CSI at the transmitter; ods = Static ODS-k with known gains (the same slot gains as the proposal, "
  "not its grant: N/k slots per user by the Hungarian assignment on log(1 + gamma g^2), energy d_s/k); "
  "odsm = the same with energy d_s/U; odsfix = Static ODS-k with fixed "
  "contiguous parts, no CSI; csiqe = csiq with every count capped at that of the load masks (equal counts at U=6, "
  "greedy placement only); weakest = mean over frames of the smallest per-user CosSim of the frame.\n")
PC = os.path.join(D, "pilot_corr")


def csi_ld(tag):
    r = {}
    for suf in ("", "_x", "_e"):
        p = os.path.join(PC, f"pca_csi_{tag}{suf}.json")
        if os.path.exists(p):
            r.update(json.load(open(p))["res"])
    return r


def cval(r, m, s, key="cos"):
    return r[m][s][key] if m in r and s in r[m] else float("nan")


T3 = ["0", "10", "20"]
for scen, k in (("rho0.7_nf6", 6), ("rho0.7", 6), ("rho0.7_nf6", 4)):
    w(f"\n### {scen}, Static ODS-{k} ({', '.join(T3)} dB; mean, then weakest in parentheses)\n")
    w("| U | fixed | csi | csiq | csiqe | ods | odsm | odsfix |")
    w("|---|---|---|---|---|---|---|---|")
    for U in range(1, 7):
        r = csi_ld(f"U{U}_k{k}_{scen}")
        if k == 4:
            r6 = csi_ld(f"U{U}_k6_{scen}")
            r = {**{m: r6[m] for m in ("fixed", "csi", "csiq") if m in r6}, **r}
        if not r:
            continue
        cells = []
        for m in ("fixed", "csi", "csiq", "csiqe", "ods", "odsm", "odsfix"):
            if m not in r:
                cells.append("-"); continue
            cells.append(" / ".join(f"{cval(r, m, s):.3f} ({cval(r, m, s, 'weakest'):.3f})" for s in T3))
        w(f"| {U} | " + " | ".join(cells) + " |")
nf = {U: csi_ld(f"U{U}_k6_rho0.7_nf6") for U in range(1, 7)}
nf4 = {U: csi_ld(f"U{U}_k4_rho0.7_nf6") for U in range(1, 5)}
w("\n### Differences (rho0.7_nf6, all seven SNRs unless stated)\n")
for a_, b_, lab, Us, src in (("csiq", "fixed", "Proposed channel-aware minus load masks", range(1, 7), nf),
                             ("csiq", "csi", "slot counts: csiq minus csi", range(1, 7), nf),
                             ("csiq", "csiqe", "free counts: csiq minus csiqe (equal counts, greedy placement)", range(2, 7), nf),
                             ("csiqe", "ods", "greedy placement at the design load: csiqe minus Static ODS-6 with known gains", range(6, 7), nf),
                             ("csiq", "ods", "csiq minus Static ODS-6 with known gains", range(1, 7), nf),
                             ("csiq", "odsm", "csiq minus Static ODS-6 with known gains, energy d_s/U", range(1, 6), nf),
                             ("ods", "odsfix", "Static ODS-6: known gains minus fixed parts", range(1, 7), nf)):
    for U in Us:
        r = src[U]
        if a_ in r and b_ in r:
            d0 = [cval(r, a_, s) - cval(r, b_, s) for s in S]
            dw = [cval(r, a_, s, "weakest") - cval(r, b_, s, "weakest") for s in S]
            w(f"- {lab}, U={U}: mean {rng_(d0)} (0 dB {d0[0]:+.3f}); weakest {rng_(dw)} (0 dB {dw[0]:+.3f})")
for U in range(1, 5):
    r, r6 = nf4[U], nf[U]
    for b_ in ("ods", "odsm"):
        if b_ in r and "csiq" in r6:
            d0 = [cval(r6, "csiq", s) - cval(r, b_, s) for s in S]
            w(f"- csiq minus Static ODS-4 with known gains ({b_}), U={U}: {rng_(d0)} (0 dB {d0[0]:+.3f})")
w("\n### Delayed CSI at the scheduler (rho0.7_nf6, mean at 0 / 10 / 20 dB)\n")
w("| rho_b | U | fixed | csi | csiq | ods | odsm |")
w("|---|---|---|---|---|---|---|")
for d in ("0.9", "0.7"):
    for U in range(1, 7):
        r = csi_ld(f"U{U}_k6_rho0.7_nf6_d{d}")
        if r:
            w(f"| {d} | {U} | " + " | ".join(" / ".join(f"{cval(r, m, s):.3f}" for s in T3) if m in r else "-"
                                            for m in ("fixed", "csi", "csiq", "ods", "odsm")) + " |")
w("\n- delayed CSI, the previous-block gains follow the AR(1) process and every current slot coefficient is "
  "rho_b times its previous value plus an independent CN(0, 1 - rho_b^2) term (adjacent-slot correlation of the current "
  "block 0.7 rho_b^2); the delayed runs use their own draws of sentences and noise")
for d in ("0.9", "0.7"):
    loss, paired = [], []
    for U in range(1, 7):
        r = csi_ld(f"U{U}_k6_rho0.7_nf6_d{d}")
        if not r:
            continue
        l_ = [cval(nf[U], "csiq", s) - cval(r, "csiq", s) for s in S]
        f_ = [cval(nf[U], "fixed", s) - cval(r, "fixed", s) for s in S]
        loss += [(x, U, s) for x, s in zip(l_, S)]
        paired += [(x - y, U, s) for x, y, s in zip(l_, f_, S)]
        g_ = [cval(r, "csiq", s) - cval(r, "fixed", s) for s in S]
        w(f"- rho_b={d}, U={U}: csiq minus load masks " + " / ".join(f"{x:+.4f}" for x in g_) + " (0 to 30 dB)")
    m1, m2 = max(loss), max(paired)
    w(f"- rho_b={d}: largest loss of csiq, perfect minus delayed, {m1[0]:.4f} (U={m1[1]}, {m1[2]} dB); "
      f"corrected by the load-mask difference between the draws {m2[0]:.4f} (U={m2[1]}, {m2[2]} dB)")
w("\n### Fading correlation and the gain estimate at U=4 (nf6, mean at 0 / 10 / 20 dB)\n")
for lab, tag in (("rho=0.3", "U4_k6_rho0.3_nf6"), ("rho=0.7", "U4_k6_rho0.7_nf6"), ("rho=0.95", "U4_k6_rho0.95_nf6"),
                 ("rho=0.7, known slot gains (genie)", "U4_k6_rho0.7_nf6_genie")):
    r = csi_ld(tag)
    if r:
        w(f"- {lab}: " + "; ".join(f"{m} " + "/".join(f"{cval(r, m, s):.3f}" for s in T3) for m in r))
        w(f"  - {lab}: csiq minus load masks at 0 dB {cval(r, 'csiq', '0') - cval(r, 'fixed', '0'):+.4f}")
gn = csi_ld("U4_k6_rho0.7_nf6_genie")
if gn:
    w("- known slot gains minus blind estimates, U=4, rho=0.7 (all seven SNRs): " + "; ".join(
        f"{m} {rng_([cval(gn, m, s) - cval(nf[4], m, s) for s in S])} (0 dB {cval(gn, m, '0') - cval(nf[4], m, '0'):+.4f})"
        for m in gn))
w("\n### Learned references with CSI (Code/pilot_corr.py, seed 1, rho=0.7 nf6, mean at 0 / 10 / 20 dB)\n")
for f in sorted(glob.glob(os.path.join(PC, "*_nf6_s1.json"))):
    r = json.load(open(f))["cos"].get("0.7", {})
    w(f"- {os.path.basename(f)[:-8]}: " + "; ".join(f"U{U} " + "/".join(f"{r[U][s]:.3f}" for s in T3) for U in sorted(r, key=int)))
w("\n### Learned references with CSI without SNR offsets (seed 1, rho=0.7, mean at 0 / 10 / 20 dB; "
  "last column: largest learned minus csiq of pca_csi_U*_k6_rho0.7 over the seven SNRs)\n")
for f in sorted(glob.glob(os.path.join(PC, "*_s1.json"))):
    if "_nf6_" in f:
        continue
    r = json.load(open(f))["cos"].get("0.7", {})
    best = max((r[U][s] - cval(csi_ld(f"U{U}_k6_rho0.7"), "csiq", s), U, s) for U in r for s in S)
    w(f"- {os.path.basename(f)[:-8]}: " + "; ".join(f"U{U} " + "/".join(f"{r[U][s]:.3f}" for s in T3) for U in sorted(r, key=int))
      + f"; largest minus csiq {best[0]:+.4f} (U={best[1]}, {best[2]} dB)")
w("- csiq without SNR offsets: " + "; ".join(
    f"U{U} " + "/".join(f"{cval(csi_ld(f'U{U}_k6_rho0.7'), 'csiq', s):.3f}" for s in T3) for U in range(1, 7)))
open(os.path.join(D, "pca_numbers.md"), "w").write("\n".join(out) + "\n")
print("\n".join(out))
