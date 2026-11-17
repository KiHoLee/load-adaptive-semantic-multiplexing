"""Read-only checks of values in the response letter (Comments 1.4, 1.5, 2.1, 2.7, 2.8), 2026-11-17.
Usage (from the repository root): python3 -I Code/letter_checks/comment_checks.py data
- four-user submitted design, three-seed means (R1.5); common vs independent fading (R1.4)
- same-code check of Comment 2.8: difference Proposed - Static ODS-U at U=k, predicted SE s*sqrt(2(U-1)/U)
- U=8 CI half-widths with sigma = largest seed SD of Learned ODS-U at U<=6 (Comment 2.8)
- training curves (field conv) of the 19 Learned ODS-U runs (Comment 2.1)
- Proposed minus Learned nested code at U=1,4,6 on AG News, SST-2, DistilBERT (Comment 2.7)"""
import json, os, sys, math, statistics as st
D = sys.argv[1]
J = lambda p: json.load(open(os.path.join(D, p)))
SN = ['0','5','10','15','20','25','30']
def cos(f, U, s): return J(f)['eval'][str(U)][s]['cos']
print("== R1-4: four-user submitted design, 3-seed means at U=4")
for s in SN:
    a = st.mean(cos('shared_U4_s%d.json'%k,4,s) for k in (1,2,3))
    r = st.mean(cos('shared_U4_randmask_s%d.json'%k,4,s) for k in (1,2,3))
    o = st.mean(cos('ortho_U4_s%d.json'%k,4,s) for k in (1,2,3))
    print(s, 'learned %.4f rand %.4f ods4 %.4f  rand-learned %+.4f rand-ods %+.4f learned-ods %+.4f'%(a,r,o,r-a,r-o,a-o))
print("== R1-6: common vs independent fading, seed 1, U=4")
for s in SN:
    sc, oc = cos('shared_U4_common_s1.json',4,s), cos('ortho_U4_common_s1.json',4,s)
    si, oi = cos('shared_U4_s1.json',4,s), cos('ortho_U4_s1.json',4,s)
    print(s, 'common: shared %.4f ods %.4f diff %+.4f | indep: shared %.4f ods %.4f diff %+.4f'%(sc,oc,sc-oc,si,oi,si-oi))
c = J('ortho_U4_common_s1.json')['cfg']; print('ortho_U4_common cfg fading', c.get('fading'), '| shared_U4_common fading', J('shared_U4_common_s1.json')['cfg'].get('fading'))
print("== R2-1: same-code check")
pe = J('extra/pca_eval.json'); print('pca_eval keys', list(pe.keys())[:12])
import glob
P = pe['proposal']['opt']; Sst = pe['static']
print('static keys', list(Sst.keys())[:10])
def sU(U, s):
    L = [json.load(open(f))['eval'][str(U)][s] for f in sorted(glob.glob(os.path.join(D, 'ortho_U%d_s[0-9].json' % U)))]
    return st.mean(l['cos_frame_std'] / math.sqrt(l['n_frames']) for l in L)
z_old, z_new, neg, n = [], [], 0, 0
for U in (2, 3, 4, 6, 8):
    for s in SN:
        st_v = Sst[str(U)][s] if isinstance(Sst[str(U)], dict) and s in Sst[str(U)] else None
        if st_v is None: continue
        if isinstance(st_v, dict): st_v = st_v.get('cos', st_v)
        d = P[str(U)][s] - st_v; se = sU(U, s); n += 1; neg += d < 0
        z_old.append(d / (math.sqrt(U - 1) * se)); z_new.append(d / (math.sqrt(2 * (U - 1) / U) * se))
rms = lambda z: math.sqrt(sum(x * x for x in z) / len(z))
print('points', n, 'negative', neg, 'rms old %.3f new %.3f mean new %.3f max|d|' % (rms(z_old), rms(z_new), st.mean(z_new)))
print('cos_frame_std U=1,4 at 0 dB', J('ortho_U1_s1.json')['eval']['1']['0']['cos_frame_std'], J('ortho_U4_s1.json')['eval']['4']['0']['cos_frame_std'])
print("== R2-3: U=8 margin over Learned ODS-8 (one seed) with sigma = largest seed SD of Learned ODS-U, U<=6, at that SNR")
o8 = J('ortho_U8_s1.json')['eval']['8']
for s in SN:
    sig = max(st.stdev([J('ortho_U%d_s%d.json' % (U, k))['eval'][str(U)][s]['cos'] for k in (1, 2, 3)]) for U in range(1, 7))
    se = o8[s]['cos_frame_std'] / math.sqrt(o8[s]['n_frames'])
    m = P['8'][s] - o8[s]['cos']
    print(s, 'margin %+.4f  half %.4f  (sigma %.4f, s %.4f)' % (m, 1.96 * math.sqrt(2 * se ** 2 + sig ** 2), sig, se))
print('ortho_U8 robust:', J('ortho_U8_s1.json').get('robust'))
print("== R2-6: training curves of Learned ODS-U (field conv: epoch, loss, CosSim@10 dB)")
fl = sorted(glob.glob(os.path.join(D, 'ortho_U[0-9]_s[0-9].json')))
lo, hi, tr = [], [], []
for f in fl:
    c = J(os.path.basename(f))['conv']
    L = {e: (l, cs) for e, l, cs in c}
    a = st.mean(L[e][0] for e in range(41, 51)); b = st.mean(L[e][0] for e in range(141, 151))
    xs = [e for e in range(50, 151)]; ys = [L[e][1] for e in xs]
    mx, my = st.mean(xs), st.mean(ys)
    slope = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sum((x - mx) ** 2 for x in xs)
    res = st.pstdev([y - (my + slope * (x - mx)) for x, y in zip(xs, ys)])
    lo.append(100 * (b - a) / a); tr.append(100 * slope); hi.append(res)
    print(os.path.basename(f), 'loss 141-150 vs 41-50 %+.2f %%' % lo[-1], 'trend %+.4f per 100 ep' % tr[-1], 'resid SD %.4f' % res, 'epochs', len(c))
print('runs', len(fl), 'loss change %.2f to %.2f %%' % (min(lo), max(lo)), 'trend %.4f to %.4f mean %.4f' % (min(tr), max(tr), st.mean(tr)), 'resid SD mean %.4f' % st.mean(hi))
print("== R2-3 precise half-widths at U=8")
for s in SN:
    sig = max(st.stdev([J('ortho_U%d_s%d.json' % (U, k))['eval'][str(U)][s]['cos'] for k in (1, 2, 3)]) for U in range(1, 7))
    se = o8[s]['cos_frame_std'] / math.sqrt(o8[s]['n_frames'])
    print(s, '%.5f' % (1.96 * math.sqrt(2 * se ** 2 + sig ** 2)), 'margin %.5f' % (P['8'][s] - o8[s]['cos']))
print("== R2-14: Proposed minus Learned nested code (seed 1), U = 1, 4, 6, 0-25 dB")
for name, prop, f in (('AG News', P, 'shared_dyns_clamp_s1.json'), ('SST-2', pe['sst2'], 'sst2_shared_dyns_s1.json'), ('DistilBERT', pe.get('distil'), 'distil_shared_dyns_s1.json')):
    if prop is None: print(name, 'no Proposed key; pca_eval keys', list(pe.keys())); continue
    if 'opt' in prop: prop = prop['opt']
    ev = J(f)['eval']; v = []
    for U in (1, 4, 6):
        row = []
        for s in SN[:6]:
            if str(U) in ev and s in ev[str(U)]:
                dd = prop[str(U)][s] - ev[str(U)][s]['cos']; v.append(dd); row.append('%+.4f' % dd)
        print(' ', name, 'U=%d' % U, ' '.join(row))
    print(' ', name, 'min %.4f max %.4f' % (min(v), max(v)))
