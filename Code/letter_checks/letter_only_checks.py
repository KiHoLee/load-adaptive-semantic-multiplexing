# Re-checks values given in the response letter only, from stored results only.
# Usage: python3 -I Code/letter_checks/letter_only_checks.py data
import json, os, sys, glob, statistics as st
D = sys.argv[1]; S = ['0','5','10','15','20','25','30']
ld = lambda p: json.load(open(os.path.join(D, p)))
P = ld('extra/pca_eval.json')['proposal']['opt']
# 1. SNR-specialized Learned ODS-U (seed 1): trlow (0-10 dB), trhigh (15-25 dB) vs default seed 1 and vs Proposed
for U in (1, 4, 6):
    dflt = ld(f'ortho_U{U}_s1.json')['eval'][str(U)]; lo = ld(f'ortho_U{U}_trlow_s1.json')['eval'][str(U)]; hi = ld(f'ortho_U{U}_trhigh_s1.json')['eval'][str(U)]
    print(f'U={U} default s1:', ' '.join(f"{dflt[s]['cos']:.3f}" for s in S), '| trlow:', ' '.join(f"{lo[s]['cos']:.3f}" for s in S), '| trhigh:', ' '.join(f"{hi[s]['cos']:.3f}" for s in S))
    print(f'   trhigh minus Proposed 15-30 dB:', ' '.join(f"{hi[s]['cos']-P[str(U)][s]:+.4f}" for s in S[3:]), '| trlow minus Proposed 0-10 dB:', ' '.join(f"{lo[s]['cos']-P[str(U)][s]:+.4f}" for s in S[:3]))
    print(f'   trlow minus default:', ' '.join(f"{lo[s]['cos']-dflt[s]['cos']:+.4f}" for s in S), '| trhigh minus default:', ' '.join(f"{hi[s]['cos']-dflt[s]['cos']:+.4f}" for s in S))
# 2. MAML ablations on the four-user version of Learned shared embedding (seed 1)
base = ld('shared_U4_s1.json')['eval']['4']
for tag in ('alpha1e-4', 'alpha2e-3', 'inner3', 'mb2', 'mb6'):
    v = ld(f'shared_U4_{tag}_s1.json')['eval']['4']
    dd = [v[s]['cos'] - base[s]['cos'] for s in S]
    print(f'shared_U4_{tag} minus shared_U4_s1: max |d| {max(abs(x) for x in dd):.4f}; min gap to Proposed {min(P["4"][s]-v[s]["cos"] for s in S):.4f}')
# 3. Learned nested code without MAML (shared_dyns_joint_s1) vs clamp seed 1
for f in sorted(glob.glob(os.path.join(D, 'shared_dyns_*_s1.json'))):
    print('  available:', os.path.basename(f), list(json.load(open(f))['eval'].keys()))
