# Reads stored block-fading results only (data/extra/pca_eval.json, data/extra/todma_redim.json, data/ortho_U*_s*.json).
# Usage (from the repository root): python3 -I Code/letter_checks/block_fading_checks.py data
import json, os, sys, glob, statistics as st
D = sys.argv[1]; S = ['0','5','10','15','20','25','30']
E = json.load(open(os.path.join(D, 'extra/pca_eval.json')))
P = E['proposal']['opt']
# matched energy: Proposed minus Static ODS-k with energy d_s/U
for k in ('4', '6'):
    Us = sorted(E['matched'][k], key=int)
    d = [(P[U][s] - E['matched'][k][U][s], U, s) for U in Us for s in S]
    lead = [x for x in d if x[0] < 0]
    print(f'Static ODS-{k} energy ds/U, U in {Us}: Proposed minus it from {min(d)[0]:+.4f} ({min(d)[1:]}) to {max(d)[0]:+.4f} ({max(d)[1:]}); its leads at', [(U, s, round(-v, 4)) for v, U, s in lead])
    if k == '6':
        hi = [x for x in d if int(x[2]) >= 10]
        print('   from 10 dB on: Proposed minus it max', max(hi)[0], max(hi)[1:])
# no-Wiener and known gain and prop
for key in ('prop', 'genie', 'zf'):
    for U in ('1', '4', '6'):
        if U in E['proposal'][key]:
            dd = [E['proposal'][key][U][s] - P[U][s] for s in S]
            print(f'{key} minus opt U={U}:', ' '.join(f'{x:+.4f}' for x in dd))
# ToDMA
t = json.load(open(os.path.join(D, 'extra/todma_redim.json')))
for U in ('1', '2', '4', '6'):
    best = []
    for s in S:
        c = [(v[s]['cos'], k) for k, v in t.items() if k.startswith(f'U{U}_') and s in v]
        best.append(max(c))
    print(f'ToDMA best T minus Proposed U={U}:', ' '.join(f'{b[0]-P[U][s]:+.4f}({b[1]})' for b, s in zip(best, S)))
print('ToDMA keys', sorted(t.keys()))
