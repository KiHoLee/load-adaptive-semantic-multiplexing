# =========================================================
# pca_eval_load.py — 2026-11-17: the load axis of Fig. 2 from light load through overload (U = 1..8), with the
# closed-form proposal of pca_eval.py unchanged (same functions, same seeds 123 + SNR, same 2,000 test frames).
# Adds only the loads and design loads that pca_eval.json does not hold: U = 7 for the proposal (opt allocation,
# blind-gain Wiener receiver) and Static ODS-7 (the same code on 7 fixed parts of floor(d_s/7) uses at the energy
# d_s/7). It also recomputes U = 6 and 8 and Static ODS-6 and -8 and checks them against pca_eval.json (must be
# identical), so the new points come from the same code and draws as the stored ones.
# Usage (CPU): python pca_eval_load.py  ->  ../data/extra/pca_eval_load.json
# =========================================================
import json, os, time
import pca_eval as pe

def main():
    t0 = time.time()
    d_s = pe.K * pe.D_B
    src = pe.Source()
    ref = json.load(open(os.path.join(pe.DATA, "extra", "pca_eval.json")))
    R = {"protocol": "as pca_eval.py: test split, 2000 frames per point and user, independent Rayleigh, eval seed 123+snr",
         "proposal_opt": {U: {s: pe.per_load(src, U, s, d_s) for s in pe.EVAL_SNRS} for U in (6, 7, 8)},
         "static": {k: {s: pe.static(src, k, s, d_s) for s in pe.EVAL_SNRS} for k in (6, 7, 8)}}
    chk = 0.0
    for U in (6, 8):
        for s in pe.EVAL_SNRS:
            chk = max(chk, abs(R["proposal_opt"][U][s] - ref["proposal"]["opt"][str(U)][str(s)]),
                      abs(R["static"][U][s] - ref["static"][str(U)][str(s)]))
    R["check_max_abs_diff_vs_pca_eval_json"] = chk
    R["runtime_s"] = time.time() - t0
    json.dump(R, open(os.path.join(pe.DATA, "extra", "pca_eval_load.json"), "w"), indent=1)
    print("check vs pca_eval.json (U = 6, 8 and Static ODS-6, -8):", chk)
    for U in (6, 7, 8):
        print(f"U={U} proposal", {s: round(v, 4) for s, v in R["proposal_opt"][U].items()})
        print(f"Static ODS-{U}", {s: round(v, 4) for s, v in R["static"][U].items()})

if __name__ == "__main__":
    main()
