# Proposed (pca_eval.json opt / pca_eval_load.json) minus Learned ODS-U seed means (data/ortho_U<U>_s<n>.json). Reads only.
import json, os, sys, glob, statistics as st
D = sys.argv[1]; S = ['0','5','10','15','20','25','30']
E = json.load(open(os.path.join(D, 'extra/pca_eval.json')))['proposal']['opt']
rows = []
for U in (1,2,3,4,5,6,8):
    fs = sorted(glob.glob(os.path.join(D, f'ortho_U{U}_s[0-9].json')))
    L = [json.load(open(f))['eval'][str(U)] for f in fs]
    for s in S:
        m = st.mean(l[s]['cos'] for l in L)
        sd = st.stdev([l[s]['cos'] for l in L]) if len(L) > 1 else float('nan')
        rows.append((E[str(U)][s] - m, U, s, len(L), sd))
mn = min(rows); mx = max(rows)
print(f'Proposed minus Learned ODS-U: min {mn[0]:.5f} at U={mn[1]} {mn[2]} dB ({mn[3]} seeds); max {mx[0]:.5f} at U={mx[1]} {mx[2]} dB')
print('10 dB:', ' '.join(f'U{r[1]}:{r[0]:.5f}' for r in rows if r[2]=='10'))
print('0 dB:', ' '.join(f'U{r[1]}:{r[0]:.4f}' for r in rows if r[2]=='0'))
print('5-30 dB range over U<=6:', min(r[0] for r in rows if r[2]!='0' and r[1]<=6), max(r[0] for r in rows if r[2]!='0' and r[1]<=6))
print('largest seed SD over U<=6:', max(r[4] for r in rows if r[1]<=6))
# Table I learned rows min over U=1,6 and 0/10/20 dB (Learned ODS-U only here)
print('Table I Learned ODS-U U=1,6 at 0/10/20:', [(r[1], r[2], round(r[0],5)) for r in rows if r[1] in (1,6) and r[2] in ('0','10','20')])
