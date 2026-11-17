# Reads stored coherence results only. Loss of the channel-aware masks from fresh (eta=1) to aged gains (eta<1).
# Usage: python3 -I Code/letter_checks/delayed_loss.py data/pilot_corr
import json, os, sys
d = sys.argv[1]
def J(rho, eta, U): return json.load(open(os.path.join(d, f'pca_csi_coh_rho{rho}_eta{eta}_U{U}_nf6.json')))['res']
for rho in ('0.3', '0.7', '0.95'):
    for U in (1, 2, 4, 6):
        for snr in ('0', '10', '20'):
            a = J(rho, '1.0', U)
            out = []
            for eta in ('0.9', '0.7', '0.5', '0.3', '0.0'):
                b = J(rho, eta, U)
                out.append(f"eta {eta}: abs {a['csiq'][snr]['cos']-b['csiq'][snr]['cos']:+.4f} gain-diff {a['csiq'][snr]['gain_fixed']-b['csiq'][snr]['gain_fixed']:+.4f}")
            if snr == '0' or rho == '0.7':
                print(f'rho_f {rho} U {U} {snr} dB | ' + ' | '.join(out))
