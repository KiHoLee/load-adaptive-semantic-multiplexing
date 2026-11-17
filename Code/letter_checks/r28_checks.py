"""Read-only checks for Comment 2.8 of the response letter. Usage: python3 -I Code/letter_checks/r28_checks.py data
1) seed SD maxima and smallest margin of Proposed over any single seed (U = 1..6, 0..30 dB)
2) Table R2.8 (margin over Learned ODS-U with 95% CI half-width 1.96 sqrt(2 s^2 + sigma^2/3),
   s = mean over seeds of cos_frame_std/sqrt(n_frames), sigma = seed SD)
3) ToDMA leads and their z-values with the Table R2.8 standard error of Proposed."""
import json, glob, math, os, sys, statistics as st
D = sys.argv[1]; os.chdir(D)
S = ['0', '5', '10', '15', '20', '25', '30']
E = json.load(open('extra/pca_eval.json'))['proposal']['opt']
def runs(pat, U): return [json.load(open(f))['eval'][str(U)] for f in sorted(glob.glob(pat))]
for name, pat in (('Learned nested code', 'shared_dyns_clamp_s[0-9].json'),
                  ('Learned shared embedding', 'shared_agn_s[0-9].json'), ('Learned ODS-U', None)):
    mins, sds = [], []
    for U in range(1, 7):
        L = runs(pat if pat else f'ortho_U{U}_s[0-9].json', U)
        for s in S:
            v = [l[s]['cos'] for l in L]; sds.append((st.stdev(v), U, s))
            mins += [(E[str(U)][s] - x, U, s) for x in v]
    print(f'{name}: smallest margin over a single seed {min(mins)}, largest seed SD {max(sds)}')
sU = {}
for U in range(1, 7):
    L = runs(f'ortho_U{U}_s[0-9].json', U); row = []
    for s in S:
        v = [l[s]['cos'] for l in L]; sd = st.stdev(v)
        se = st.mean(l[s]['cos_frame_std'] / math.sqrt(l[s]['n_frames']) for l in L); sU[(U, s)] = se
        row.append('%+.3f+-%.3f' % (E[str(U)][s] - st.mean(v), 1.96 * math.sqrt(2 * se ** 2 + sd ** 2 / 3)))
    print('Table R2.8 U=%d' % U, ' '.join(row))
T = json.load(open('extra/todma_redim.json'))
for U, s in ((1, '15'), (1, '25'), (1, '30'), (6, '20')):
    best = max((T[k][s]['cos'], k) for k in T if k.startswith(f'U{U}_'))
    lead = best[0] - E[str(U)][s]
    print(f'ToDMA U={U} {s} dB lead {lead:.4f} z={lead / math.sqrt(T[best[1]][s]["se"] ** 2 + sU[(U, s)] ** 2):.2f}')
