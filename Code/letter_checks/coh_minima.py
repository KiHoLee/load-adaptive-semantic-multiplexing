# Reads stored coherence results only (no simulation). Usage: python3 -I Code/letter_checks/coh_minima.py data/pilot_corr
import json, glob, os, sys, re
d = sys.argv[1]
rows = []
for f in sorted(glob.glob(os.path.join(d, 'pca_csi_coh_rho*_eta*_U*.json'))):
    b = os.path.basename(f)
    m = re.match(r'pca_csi_coh_rho([\d.]+)_eta([\d.]+)_U(\d+)(_nf6)?(_genie)?\.json', b)
    if not m or m.group(5): continue
    J = json.load(open(f))
    nf = 6 if m.group(4) else 0
    for mode in ('switch', 'csiq'):
        if mode not in J['res']: continue
        for snr, v in J['res'][mode].items():
            rows.append((mode, nf, float(m.group(1)), float(m.group(2)), int(m.group(3)), int(snr), v['gain_fixed'], v['gain_fixed_ci'], v.get('gain_fixed_weakest'), b))
for nf in (6, 0):
    sw = [r for r in rows if r[0] == 'switch' and r[1] == nf]
    mn = min(sw, key=lambda r: r[6])
    print(f'nf={nf}: switch points {len(sw)}; min switch gain {mn[6]:.6f} CI {mn[7]} at rho_f {mn[2]}, eta {mn[3]}, U {mn[4]}, {mn[5]} dB ({mn[9]})')
    below = [r for r in sw if r[7][1] < 0]
    print(f'   switch CI entirely below 0: {len(below)} points; worst {min(below,key=lambda r:r[6])[6]:.4f}' if below else '   none below')
# csiq at eta 0, 0 dB, rho 0.7 nf6
for U in (1,2,4,6):
    r=[x for x in rows if x[0]=='csiq' and x[1]==6 and x[2]==0.7 and x[3]==0.0 and x[4]==U and x[5]==0]
    print('csiq eta0 rho0.7 nf6 0dB U',U, round(r[0][6],4), [round(c,4) for c in r[0][7]])
