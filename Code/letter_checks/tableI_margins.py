# Table I margins: Proposed minus each learned scheme at U = 1, 6 and 0/10/20 dB, and over all U=1..6 and SNRs.
import json, os, sys, glob, statistics as st
D = sys.argv[1]; S = ['0','5','10','15','20','25','30']
E = json.load(open(os.path.join(D, 'extra/pca_eval.json')))['proposal']['opt']
def means(pattern, U):
    fs = sorted(glob.glob(os.path.join(D, pattern)))
    L = [json.load(open(f))['eval'][str(U)] for f in fs]
    return {s: st.mean(l[s]['cos'] for l in L) for s in S}, len(L), {s: (st.stdev([l[s]['cos'] for l in L]) if len(L)>1 else 0) for s in S}
for name, pat in (('Learned nested code', 'shared_dyns_clamp_s[0-9].json'), ('Learned shared embedding', 'shared_agn_s[0-9].json'), ('Learned ODS-U', None)):
    allr = []; t1 = []
    for U in range(1, 7):
        m, n, sd = means(pat if pat else f'ortho_U{U}_s[0-9].json', U)
        for s in S:
            d = E[str(U)][s] - m[s]; allr.append((d, U, s))
            if U in (1, 6) and s in ('0','10','20'): t1.append((d, U, s))
    print(f'{name} ({n} seeds): all U<=6 {min(allr)[0]:.4f} {min(allr)[1:]} to {max(allr)[0]:.4f} {max(allr)[1:]}; Table I points {min(t1)[0]:.5f} {min(t1)[1:]} to {max(t1)[0]:.5f} {max(t1)[1:]}')
# PCA-start learned nested code (seed 1)
f = os.path.join(D, 'shared_dyns_pcainit_s1.json')
if os.path.exists(f):
    L = json.load(open(f))['eval']
    r = [(E[str(U)][s] - L[str(U)][s]['cos'], U, s) for U in range(1, 7) for s in S if str(U) in L]
    print(f'PCA-start nested code (seed 1): {min(r)[0]:.4f} {min(r)[1:]} to {max(r)[0]:.4f} {max(r)[1:]}')
