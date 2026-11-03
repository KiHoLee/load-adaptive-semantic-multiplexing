# =========================================================
# rev_dyn_numbers.py — every number of the dynamic-mask revision (2026-11-03), printed from the
# stored results into data/dyn_numbers.md. Reads data/ only; sections whose files are missing say so.
#   Proposal: shared_dyns_clamp_s{1,2,3} (shared_dyns checkpoints re-evaluated with the profile held
#   at its 25 dB value above the training range). Static ODS-k: ortho_U<k>_s{1,2,3} at its design load
#   (unit average power over its d_s/k occupied uses, so its CosSim holds at every U <= k).
# =========================================================
import glob, json, os, statistics as st
HERE = os.path.dirname(os.path.abspath(__file__))
D = os.path.join(os.path.dirname(HERE), "data")
S = ["0", "5", "10", "15", "20", "25", "30"]
P = "shared_dyns_clamp"
out = []
w = out.append


def ld(p):
    return json.load(open(os.path.join(D, p)))


def have(pat):
    return sorted(glob.glob(os.path.join(D, pat)))


def evals(name, U, key="cos"):
    fs = have(f"{name}_s[0-9].json")
    return [[ld(os.path.relpath(f, D))["eval"][str(U)][s][key] for s in S] for f in fs]


def mean(v):
    return [st.mean(c) for c in zip(*v)]


def sd(v):
    return [st.pstdev(c) for c in zip(*v)] if len(v) > 1 else [0.0] * len(S)


def row(label, v, fmt="{:.3f}"):
    w(f"| {label} | " + " | ".join(fmt.format(x) for x in v) + " |")


def hdr(first="", cols=S):
    w(f"| {first} | " + " | ".join(f"{c} dB" for c in cols) + " |")
    w("|---" * (len(cols) + 1) + "|")


ods = {k: mean(evals(f"ortho_U{k}", k)) for k in range(1, 7)}
prop = {U: mean(evals(P, U)) for U in range(1, 7)}
psd = {U: sd(evals(P, U)) for U in range(1, 7)}

w("# Numbers of the dynamic-mask revision\n")
w("## 1. Per-user CosSim, three-seed means (proposal sd in brackets as max over SNR)\n")
hdr("scheme")
for k in range(1, 7):
    row(f"Static ODS-{k} (any U<=k)", ods[k])
for U in range(1, 7):
    row(f"Proposed one model, U={U} (sd<= {max(psd[U]):.3f})", prop[U])

w("\n## 2. Proposal minus Static ODS-k at load U (k >= U)\n")
hdr("U, k")
for U in range(1, 7):
    for k in range(U, 7):
        row(f"U={U}, ODS-{k}", [a - b for a, b in zip(prop[U], ods[k])], "{:+.3f}")
gaps = [p - o for U in range(1, 7) for p, o in zip(prop[U], ods[U])]
w(f"\nProposal minus best partition ODS-U over all loads and SNRs: min {min(gaps):+.3f}, max {max(gaps):+.3f}")
g0 = [prop[U][0] - ods[U][0] for U in range(1, 7)]
w("At 0 dB, proposal minus ODS-U for U=1..6: " + " ".join(f"{x:+.3f}" for x in g0))
g5 = [p - o for U in range(1, 7) for p, o in zip(prop[U][1:], ods[U][1:])]
w(f"From 5 dB on, proposal minus ODS-U: min {min(g5):+.3f}, max {max(g5):+.3f}")

w("\n## 3. Matched-energy alternative: a static user raises its power on its own block to spend d_s/U\n")
for kk in (4, 6):
    fs = have(f"extra/energy_ortho_U{kk}_s[0-9].json")
    if not fs:
        continue
    hdr(f"proposal - boosted ODS-{kk}")
    for U in range(1, kk + 1):
        b = mean([[ld(os.path.relpath(f, D))["U"][str(U)]["per_user"][s]["cos"] for s in S] for f in fs])
        row(f"U={U}", [p - x for p, x in zip(prop[U], b)], "{:+.3f}")

w("\n## 4. Aggregate fidelity S(U) per block (served users only), 10 dB and 0 dB\n")
for snr in ("0", "10", "20"):
    i = S.index(snr)
    w(f"\n{snr} dB:")
    w("| U | proposal | " + " | ".join(f"ODS-{k}" for k in range(1, 7)) + " |")
    w("|---" * 8 + "|")
    for U in range(1, 7):
        w(f"| {U} | {U * prop[U][i]:.2f} | " + " | ".join(f"{min(U, k) * ods[k][i]:.2f}" for k in range(1, 7)) + " |")

w("\n## 5. Design variants and history (per-user CosSim)\n")
hdr("variant")
for name, label in [("shared_agn", "superposed overlapping masks + attention, one model (first revision)"),
                    ("shared_abs", "overlapping masks absorbing idle masks (seed 1)"),
                    ("shared_absblk", "fixed 0/1 blocks absorbing idle blocks (seed 1)"),
                    ("shared_dyn", "nested code + dynamic masks, no SNR profile"),
                    ("shared_dyns", "with SNR profile, profile extrapolated above 25 dB"),
                    (P, "with SNR profile held above 25 dB (final)")]:
    for U in (1, 4, 6):
        v = evals(name, U)
        if v:
            row(f"{label}, U={U} ({len(v)} seeds)", mean(v))

w("\n## 6. Effective width of a user's transmitted energy (participation ratio, dims), three seeds\n")
fs = have("extra/width_shared_dyns_s[0-9].json")
if fs:
    hdr("U (dims owned)")
    for U in range(1, 7):
        v = [[ld(os.path.relpath(f, D))["U"][str(U)][s]["eff_width"] for s in S] for f in fs]
        row(f"U={U} ({3072 // U})", mean(v), "{:.0f}")

w("\n## 7. ToDMA re-dimensioned (T slots of L uses, T*L = 3072; single run, 500 frames per point)\n")
f = os.path.join(D, "extra", "todma_redim.json")
if os.path.exists(f):
    t = json.load(open(f))
    hdr("config")
    for k, v in t.items():
        row(k, [v[s]["cos"] if s in v else float("nan") for s in S])
    for U in (1, 2, 4, 6):
        best = [max(v[s]["cos"] for k, v in t.items() if k.startswith(f"U{U}_") and s in v) for s in S]
        row(f"U={U}: proposal minus best ToDMA", [a - b for a, b in zip(prop[U], best)], "{:+.3f}")

w("\n## 8. Round-4 evaluations and ablations of the final scheme (seed 1 unless noted)\n")
for name, label in [("shared_dyns_freezetx", "random transmit projection (frozen)"),
                    ("shared_dyns_uniform", "uniform transmit and receive profiles (frozen)"),
                    ("shared_dyns_joint", "training without MAML"),
                    ("shared_dyns_K1", "K=1 (d_s=768)"), ("shared_dyns_K8", "K=8 (d_s=6144)"),
                    ("shared_dyns_slots12", "12 slots"), ("shared_dyns_slots48", "48 slots"),
                    ("sst2_shared_dyns", "SST-2"), ("distil_shared_dyns", "DistilBERT")]:
    fs2 = have(f"{name}_s[0-9].json")
    if not fs2:
        w(f"- {label}: not available yet")
        continue
    for U in (1, 4, 6):
        row(f"{label}, U={U}", mean(evals(name, U)))
fs2 = have("shared_dyns_U8_s[0-9].json")
if fs2:
    for U in (1, 4, 6, 8):
        row(f"one model up to 8 users, U={U}", mean(evals("shared_dyns_U8", U)))
    row("Static ODS-8 (ortho_U8 s1)", [ld("ortho_U8_s1.json")["eval"]["8"][s]["cos"] for s in S])
w("\n### 8a. Proposal (seed 1, clamp) at U=1 and 6 for reference in Table II")
for U in (1, 4, 6):
    row(f"proposal seed 1, U={U}", [ld(f"{P}_s1.json")["eval"][str(U)][s]["cos"] for s in S])

fs2 = have("extra/noadapt_shared_dyns_s[0-9].json")
if fs2:
    w("\n### 8b. Without the one-step adaptation (three seeds): noadapt minus adapted")
    worst = 0.0
    for U in range(1, 7):
        na = mean([[ld(os.path.relpath(f, D))["U"][str(U)][s]["cos"] for s in S] for f in fs2])
        d = [a - b for a, b in zip(na, prop[U])]
        worst = max(worst, max(abs(x) for x in d))
        row(f"U={U} noadapt mean", na)
        row(f"U={U} noadapt - adapted", d, "{:+.3f}")
    w(f"max |noadapt - adapted| over U and SNR: {worst:.3f}")

f = os.path.join(D, "extra", "robust_shared_dyns_s1.json")
if os.path.exists(f):
    r = json.load(open(f))["U"]
    o = ld("ortho_U4_s1.json")["robust"]
    o_def = [ld("ortho_U4_s1.json")["eval"]["4"][s]["cos"] for s in S]
    w("\n### 8c. Channel conditions at U=4, seed 1: proposal vs Static ODS-4 (ortho_U4_s1)")
    hdr("condition")
    row("proposal default", [r["4"]["default"][s] for s in S]); row("ODS-4 default", o_def)
    for k in ("phase10", "phase20", "phase30", "test_common"):
        row(f"proposal {k}", [r["4"][k][s] for s in S]); row(f"ODS-4 {k}", [o[k][s] for s in S])
    for D_ in (3, 6):
        row(f"proposal nearfar{D_} mean", [r["4"][f"nearfar{D_}"][s]["cos"] for s in S])
        row(f"ODS-4 nearfar{D_} mean", [o[f"nearfar{D_}"][s]["cos"] for s in S])
        row(f"proposal nearfar{D_} worst", [r["4"][f"nearfar{D_}"][s]["worst"] for s in S])
        row(f"ODS-4 nearfar{D_} worst", [o[f"nearfar{D_}"][s]["worst"] for s in S])

f = os.path.join(D, "extra", "probe_shared_dyns_s1.json")
if os.path.exists(f):
    pr = json.load(open(f))
    w(f"\n### 8d. Topic probe accuracy (seed 1); clean test accuracy {pr['probe_clean_acc']:.4f}")
    hdr("scheme")
    for U, v in pr["U"].items():
        row(f"proposal U={U}", [v[s]["probe_acc"] for s in S])
    for k in (4, 6):
        e = ld(f"ortho_U{k}_s1.json")["eval"][str(k)]
        if "probe_acc" in e["20"]:
            row(f"Static ODS-{k} at U={k}", [e[s]["probe_acc"] for s in S])

for n in ("shared_dyns", "ortho_U4", "ortho_U6", "shared_agn"):
    f = os.path.join(D, "extra", f"cost_{n}_s1.json")
    if os.path.exists(f):
        c = json.load(open(f))
        w(f"- cost {n}: {c['params']} parameters, {c['latency_ms']} ms per frame (device {c['device']})")

w("\n### 8e. Generality at full load, seed 1: proposal vs Static ODS-k at its design load")
for pre, lab in (("sst2_", "SST-2"), ("distil_", "DistilBERT")):
    for U in (4, 6):
        fo = os.path.join(D, f"{pre}ortho_U{U}_s1.json")
        fp = os.path.join(D, f"{pre}shared_dyns_s1.json")
        if os.path.exists(fo) and os.path.exists(fp):
            a = [ld(f"{pre}shared_dyns_s1.json")["eval"][str(U)][s]["cos"] for s in S]
            b = [ld(f"{pre}ortho_U{U}_s1.json")["eval"][str(U)][s]["cos"] for s in S]
            row(f"{lab} proposal U={U}", a); row(f"{lab} Static ODS-{U}", b)
            row(f"{lab} difference U={U}", [x - y for x, y in zip(a, b)], "{:+.3f}")
    fp = os.path.join(D, f"{pre}shared_dyns_s1.json")
    if os.path.exists(fp) and "probe_acc" in ld(f"{pre}shared_dyns_s1.json")["eval"]["4"]["20"]:
        row(f"{lab} proposal probe U=4", [ld(f"{pre}shared_dyns_s1.json")["eval"]["4"][s]["probe_acc"] for s in S])
        fo = os.path.join(D, f"{pre}ortho_U4_s1.json")
        if os.path.exists(fo) and "probe_acc" in ld(f"{pre}ortho_U4_s1.json")["eval"]["4"]["20"]:
            row(f"{lab} Static ODS-4 probe", [ld(f"{pre}ortho_U4_s1.json")["eval"]["4"][s]["probe_acc"] for s in S])

open(os.path.join(D, "dyn_numbers.md"), "w").write("\n".join(out) + "\n")
print("\n".join(out[:60]))
