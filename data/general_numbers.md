# Numbers of the general-mask version (Code/general_numbers.py, read-only on data/)

Every number below is computed from stored results; CIs are paired bootstrap 95% intervals over frames (1000 resamples, numpy default_rng(555 + SNR)) unless marked SE (Monte Carlo standard error over draws). CosSim = per-user cosine similarity of the recovered embedding; NMSE = sum of squared errors / sum of squared norms (PCA coefficients); reductions in percent of the reference NMSE. K = 4 (d_s = 4 d_b = 3072 uses, full load U = 4) unless stated.

## G. Index

| section | content | inputs | JSON files found | frames and seeds |
|---|---|---|---|---|
| A | load adaptivity, AG News (Fig. 2) | extra/pca_eval.json, extra/pca_eval_load.json, ortho_U<k>_s*.json | 21 | 2000 frames per point and user (pca_eval), eval seed 123 + SNR; Learned ODS-U seeds s1-s3 (U = 1..6), s1 (U = 8) |
| B | natural captions of one image, known gains | shared/shared_coco_K{4,2}_U{2..5}, other shared_coco_*/shared_flickr_* (no _mix), shared_agnews_*_rnd, content_stats.json | 35 | 2000 frames (COCO), 1000 (Flickr), 500 (_rnd); captions default_rng(123 + SNR + 1000 U), gains 5123 + SNR + 1000 U, noise 9123 + SNR + 1000 U; bootstrap 555 + SNR |
| C | one scene from light load through overload (Fig. 3(a)) | shared/theory_coco_mix<k>_K4_U<U>, shared_coco_mix<k>_K4_U<U>_ol(.npz), theory_coco_mix*_K{2,4}_U4, shared_coco_mix*_K{2,4}_U4, theory_coco_K*_U*, theory_flickr_K*_U* | 64 | model: 100 Rayleigh draws default_rng(7); captions: 2000 frames, seeds as section B; bootstrap 555 + SNR |
| D | coherence bandwidth and time (Fig. 3(b)) | pilot_corr/pca_csi_coh_rho<rho>_eta<eta>_U<U>[_nf6].json/.npz | 100 | 2000 frames per run, draws seed 123 + SNR (shared by all modes of a run), switch samples seed 999 + SNR, bootstrap 555 + SNR |
| E | search against the closed forms | extra/overlap_bound2_*.json, extra/overlap_bound_U<U>_nf<X>.json | 22 | equal gains: 1 draw; rayleigh-D20 and slot-rho0.7-D20: 20 draws default_rng(123 + SNR); search torch seed 1000 + SNR (original: seed SNR), 3000 Adam steps, 3 random starts |
| F | gain estimation | shared/pilot/shared_pilot_<tag>[_snr<S>].json | 36 | 2000 frames, seeds as section B, bootstrap 555 + SNR |

### Data anomalies and checks (14)

- [A] Proposed minus Static ODS-U at its design load U = k is negative in 37 of 49 cells (down to -0.0034 at U = 3, 5 dB)
- [B] expected sign, not an error: the control with the fitted C on captions of different images (shared/shared_coco_K2_U4_ctrlfit.json, shared/shared_coco_K4_U4_ctrlfit.json) loses with the overlapping masks at every SNR (worst -2.98 % [-3.13, -2.83] at -10 dB): overlap pays only when the content is really shared
- [C] receiver NMSE reduction (load_pu -> load_joint), model against real captions, kappa_bar 0.8, K = 4: differs by up to 14.13 percentage points (U = 2, 20 dB: model 32.11 %, captions +17.98 % [+14.85, +21.07])
- [C] model and real captions (kappa_bar 0.8, K = 4) differ in the overlap NMSE reduction by up to 8.84 percentage points (U = 5, 20 dB: model 22.21 %, captions +31.04 % [+30.34, +31.75]); the model lies outside most caption CIs (see C.2d)
- [C] shared/shared_coco_mix0_K4_U4.json 0 dB (kappa_bar 0.0023): the joint receiver with the fitted C is slightly worse than the per-user receiver, receiver NMSE reduction -0.28 % [-0.31, -0.25] (C fitted from independent captions is estimation noise)
- [C] shared/shared_coco_mix0_K4_U4.json 10 dB (kappa_bar 0.0023): the joint receiver with the fitted C is slightly worse than the per-user receiver, receiver NMSE reduction -0.69 % [-0.77, -0.61] (C fitted from independent captions is estimation noise)
- [C] shared/shared_coco_mix0_K2_U4.json 0 dB (kappa_bar 0.0023): overlap NMSE reduction CI entirely below 0 (-0.0006 % [-0.0011, -0.0001]): the overlapping masks are slightly worse than the load masks here
- [C] shared/shared_coco_mix0_K2_U4.json 0 dB (kappa_bar 0.0023): the joint receiver with the fitted C is slightly worse than the per-user receiver, receiver NMSE reduction -0.18 % [-0.20, -0.16] (C fitted from independent captions is estimation noise)
- [C] shared/shared_coco_mix0_K2_U4.json 10 dB (kappa_bar 0.0023): overlap NMSE reduction CI entirely above 0 (+0.0057 % [+0.0014, +0.0104]) although the content is independent: the fitted C (kappa_bar about 0.002, not 0) gives a tiny nonzero overlap effect
- [C] shared/shared_coco_mix0_K2_U4.json 10 dB (kappa_bar 0.0023): the joint receiver with the fitted C is slightly worse than the per-user receiver, receiver NMSE reduction -0.47 % [-0.52, -0.43] (C fitted from independent captions is estimation noise)
- [D] the coherence-aware choice falls below the load masks (CI entirely below 0) at 14 of 294 points, by at most 0.0027 CosSim (nf 0, rho_f 0.7, eta 0.0, U = 4, 0 dB)
- [E] Rayleigh draws: the search knows the drawn gains at the transmitter while the closed forms use the unit design gain, so its gap (mean up to +5.837 %) is mostly transmitter gain knowledge: the non-overlapping support-restricted search with the same knowledge is already lower than the closed form by up to 5.669 % of its own value (mean over draws of 100 (closed - support)/support; coco_rayleigh-D20_K4_U2, 0 dB; as a ratio of the mean sums 6.989 % of the closed form at most), and the free search beyond it is lower by at most 2.297 % of the support-restricted value (mean over draws)
- [E] channel-aware closed form csiq13 against the search with the same subband gains: mean gap up to +5.277 % (indep_slot-rho0.7-D20_K4_U4_nf6, 20 dB), per draw up to +19.942 %; the search shares little energy (shared-slot energy 0.101), so the gap is mostly a better subband assignment than the greedy support, not overlap
- [E] independent content (AG News), overload U = 6, 8 with +-6 dB offsets: searched overlapping masks give a mean NMSE lower than the load masks by up to 2.200 % (U = 6, 10 dB; nonzero at 0/10 dB); the worst user is worse by up to 3.213 % (U = 8) at 10 dB only (unchanged within 1e-3 % at the other SNRs); with equal gains the gap is 0 at every load. So 'load masks optimal for independent content' holds up to full load (and in overload only with equal gains)

## A. Load adaptivity from light load through overload (AG News, independent content, Fig. 2)

Per-user CosSim, independent Rayleigh block fading, blind-gain receiver of the manuscript, d_s = K d_b with K = 4 (full load U = K = 4; light load U < 4; overload U > 4). Proposed = PCA nested code with the load masks and the allocation (12), one untrained transceiver for every load. Static ODS-k = the same closed-form code on k fixed parts (n = E = d_s/k), able to serve at most k users; for U <= k each served user gets the value of Static ODS-k at its design load (flat in U). Learned ODS-U = the learned partition retrained for the load U.

- check: pca_eval_load.json minus pca_eval.json at U = k = 6, largest |difference| over the seven SNRs: Proposed 0.0e+00, Static ODS-6 2.2e-16 (stored check value 2.2e-16)
- check: pca_eval_load.json minus pca_eval.json at U = k = 8, largest |difference| over the seven SNRs: Proposed 1.1e-16, Static ODS-8 1.1e-16 (stored check value 2.2e-16)
- protocol: test split, 2000 frames per point and user, independent Rayleigh, eval seed 123+snr
- Learned ODS-U files: U = 1: 3 seed(s); U = 2: 3 seed(s); U = 3: 3 seed(s); U = 4: 3 seed(s); U = 5: 3 seed(s); U = 6: 3 seed(s); U = 8: 1 seed(s); U = 7: no file data/ortho_U7_s*.json (not run, Fig. 2 draws no Learned ODS-7)

### A.1 Proposed, per-user CosSim versus the load U (K = 4)

| load U | 0 dB | 5 dB | 10 dB | 15 dB | 20 dB | 25 dB | 30 dB |
|---|---|---|---|---|---|---|---|
| Proposed, U = 1 | 0.8948 | 0.9495 | 0.9769 | 0.9915 | 0.9966 | 0.9986 | 0.9995 |
| Proposed, U = 2 | 0.8458 | 0.9175 | 0.9623 | 0.9844 | 0.9934 | 0.9972 | 0.9991 |
| Proposed, U = 3 | 0.8153 | 0.8943 | 0.9505 | 0.9776 | 0.9908 | 0.9961 | 0.9988 |
| Proposed, U = 4 | 0.7915 | 0.8768 | 0.9391 | 0.9720 | 0.9886 | 0.9952 | 0.9985 |
| Proposed, U = 5 | 0.7748 | 0.8645 | 0.9276 | 0.9645 | 0.9808 | 0.9888 | 0.9918 |
| Proposed, U = 6 | 0.7572 | 0.8528 | 0.9190 | 0.9572 | 0.9746 | 0.9821 | 0.9854 |
| Proposed, U = 7 (pca_eval_load.json) | 0.7418 | 0.8399 | 0.9089 | 0.9484 | 0.9665 | 0.9746 | 0.9781 |
| Proposed, U = 8 | 0.7282 | 0.8299 | 0.9024 | 0.9424 | 0.9609 | 0.9689 | 0.9726 |

### A.2 Static ODS-k, per-user CosSim of every served user (U <= k), K = 4

| k | 0 dB | 5 dB | 10 dB | 15 dB | 20 dB | 25 dB | 30 dB |
|---|---|---|---|---|---|---|---|
| Static ODS-1 | 0.8948 | 0.9495 | 0.9769 | 0.9915 | 0.9966 | 0.9986 | 0.9995 |
| Static ODS-2 | 0.8490 | 0.9199 | 0.9618 | 0.9850 | 0.9938 | 0.9973 | 0.9991 |
| Static ODS-3 | 0.8178 | 0.8977 | 0.9494 | 0.9793 | 0.9913 | 0.9963 | 0.9987 |
| Static ODS-4 | 0.7937 | 0.8798 | 0.9387 | 0.9743 | 0.9891 | 0.9952 | 0.9983 |
| Static ODS-5 | 0.7744 | 0.8660 | 0.9292 | 0.9661 | 0.9818 | 0.9886 | 0.9920 |
| Static ODS-6 | 0.7575 | 0.8531 | 0.9192 | 0.9584 | 0.9748 | 0.9818 | 0.9853 |
| Static ODS-7 (pca_eval_load.json) | 0.7428 | 0.8414 | 0.9108 | 0.9507 | 0.9677 | 0.9752 | 0.9788 |
| Static ODS-8 | 0.7290 | 0.8302 | 0.9018 | 0.9436 | 0.9614 | 0.9688 | 0.9727 |

- Static ODS-k decreases with k at every SNR (k = 1..8): True; so the best fixed partition that serves all U users at load U is Static ODS-U (upper envelope of the fixed partitions)

### A.3 Learned ODS-U at its load U: mean over seeds (sample SD, seeds)

| load U | 0 dB | 5 dB | 10 dB | 15 dB | 20 dB | 25 dB | 30 dB |
|---|---|---|---|---|---|---|---|
| Learned ODS-1 | 0.8703 (0.0013, 3) | 0.9416 (0.0042, 3) | 0.9712 (0.0020, 3) | 0.9854 (0.0004, 3) | 0.9905 (0.0006, 3) | 0.9930 (0.0005, 3) | 0.9936 (0.0005, 3) |
| Learned ODS-2 | 0.8162 (0.0014, 3) | 0.9101 (0.0027, 3) | 0.9554 (0.0010, 3) | 0.9774 (0.0004, 3) | 0.9865 (0.0005, 3) | 0.9903 (0.0005, 3) | 0.9914 (0.0007, 3) |
| Learned ODS-3 | 0.7788 (0.0015, 3) | 0.8877 (0.0019, 3) | 0.9422 (0.0006, 3) | 0.9703 (0.0004, 3) | 0.9821 (0.0005, 3) | 0.9869 (0.0006, 3) | 0.9884 (0.0006, 3) |
| Learned ODS-4 | 0.7503 (0.0012, 3) | 0.8685 (0.0009, 3) | 0.9319 (0.0008, 3) | 0.9638 (0.0006, 3) | 0.9776 (0.0005, 3) | 0.9832 (0.0006, 3) | 0.9852 (0.0007, 3) |
| Learned ODS-5 | 0.7278 (0.0018, 3) | 0.8523 (0.0005, 3) | 0.9221 (0.0001, 3) | 0.9574 (0.0004, 3) | 0.9729 (0.0005, 3) | 0.9790 (0.0006, 3) | 0.9814 (0.0006, 3) |
| Learned ODS-6 | 0.7118 (0.0010, 3) | 0.8388 (0.0007, 3) | 0.9120 (0.0005, 3) | 0.9498 (0.0003, 3) | 0.9668 (0.0006, 3) | 0.9733 (0.0006, 3) | 0.9759 (0.0006, 3) |
| Learned ODS-8 | 0.6851 (-, 1) | 0.8178 (-, 1) | 0.8951 (-, 1) | 0.9368 (-, 1) | 0.9550 (-, 1) | 0.9623 (-, 1) | 0.9654 (-, 1) |

### A.4 Proposed minus Learned ODS-U at the same load U (CosSim)

| load U | 0 dB | 5 dB | 10 dB | 15 dB | 20 dB | 25 dB | 30 dB | min over 5-30 dB | max over 5-30 dB | min over 0-30 dB of Proposed minus the best seed |
|---|---|---|---|---|---|---|---|---|---|---|
| U = 1 (3 seed(s)) | +0.0245 | +0.0079 | +0.0057 | +0.0061 | +0.0060 | +0.0055 | +0.0059 | +0.0055 | +0.0079 | +0.0042 |
| U = 2 (3 seed(s)) | +0.0296 | +0.0074 | +0.0069 | +0.0070 | +0.0069 | +0.0069 | +0.0078 | +0.0069 | +0.0078 | +0.0058 |
| U = 3 (3 seed(s)) | +0.0364 | +0.0066 | +0.0083 | +0.0073 | +0.0088 | +0.0092 | +0.0104 | +0.0066 | +0.0104 | +0.0051 |
| U = 4 (3 seed(s)) | +0.0412 | +0.0084 | +0.0072 | +0.0081 | +0.0110 | +0.0120 | +0.0132 | +0.0072 | +0.0132 | +0.0066 |
| U = 5 (3 seed(s)) | +0.0470 | +0.0122 | +0.0055 | +0.0071 | +0.0079 | +0.0098 | +0.0104 | +0.0055 | +0.0122 | +0.0054 |
| U = 6 (3 seed(s)) | +0.0454 | +0.0140 | +0.0070 | +0.0074 | +0.0078 | +0.0088 | +0.0095 | +0.0070 | +0.0140 | +0.0065 |
| U = 8 (1 seed(s)) | +0.0432 | +0.0121 | +0.0073 | +0.0057 | +0.0059 | +0.0067 | +0.0072 | +0.0057 | +0.0121 | +0.0057 |

- Proposed minus Learned ODS-U over U = 1, 2, 3, 4, 5, 6, 8 and 0-30 dB: +0.0055 (U = 1, 25 dB) to +0.0470 (U = 5, 0 dB); from 5 dB on: +0.0055 (U = 1, 25 dB) to +0.0140
- Proposed exceeds Learned ODS-U at every listed load and SNR: True

### A.5 Proposed minus Static ODS-U at its design load U = k (same code, same uses; differences = sampling and the slot granularity at U = 5 and 7)

| U = k | 0 dB | 5 dB | 10 dB | 15 dB | 20 dB | 25 dB | 30 dB | largest |difference| |
|---|---|---|---|---|---|---|---|---|
| U = k = 1 | +0.0000 | +0.0000 | +0.0000 | +0.0000 | +0.0000 | +0.0000 | +0.0000 | 0.0000 |
| U = k = 2 | -0.0032 | -0.0024 | +0.0005 | -0.0006 | -0.0004 | -0.0001 | +0.0000 | 0.0032 |
| U = k = 3 | -0.0025 | -0.0034 | +0.0011 | -0.0017 | -0.0005 | -0.0002 | +0.0001 | 0.0034 |
| U = k = 4 | -0.0021 | -0.0030 | +0.0004 | -0.0023 | -0.0005 | -0.0000 | +0.0002 | 0.0030 |
| U = k = 5 | +0.0004 | -0.0016 | -0.0016 | -0.0016 | -0.0010 | +0.0002 | -0.0002 | 0.0016 |
| U = k = 6 | -0.0003 | -0.0003 | -0.0002 | -0.0012 | -0.0002 | +0.0003 | +0.0000 | 0.0012 |
| U = k = 7 | -0.0010 | -0.0016 | -0.0019 | -0.0023 | -0.0012 | -0.0005 | -0.0008 | 0.0023 |
| U = k = 8 | -0.0008 | -0.0003 | +0.0006 | -0.0011 | -0.0005 | +0.0001 | -0.0001 | 0.0011 |

- Proposed minus Static ODS-U at U = k = 2..8: 37 of 49 cells negative; range -0.0034 (U = 3, 5 dB) to +0.0011 (U = 3, 10 dB): equal within these values at the design load, not above

### A.6 Margins over Static ODS-k below its design load (U < k): Proposed at load U minus Static ODS-k (fixed parts dimensioned for k users)

'Light load' means U < K = 4; rows with U > 4 are overload of the block even though they lie below the design load k of Static ODS-6 or ODS-8.

| k, U | 0 dB | 5 dB | 10 dB | 15 dB | 20 dB | 25 dB | 30 dB | min | max |
|---|---|---|---|---|---|---|---|---|---|
| Static ODS-4, U = 1 | +0.1012 | +0.0697 | +0.0382 | +0.0173 | +0.0075 | +0.0033 | +0.0012 | +0.0012 | +0.1012 |
| Static ODS-4, U = 2 | +0.0521 | +0.0377 | +0.0236 | +0.0102 | +0.0043 | +0.0020 | +0.0008 | +0.0008 | +0.0521 |
| Static ODS-4, U = 3 | +0.0216 | +0.0145 | +0.0118 | +0.0033 | +0.0018 | +0.0009 | +0.0005 | +0.0005 | +0.0216 |
| Static ODS-6, U = 1 | +0.1373 | +0.0964 | +0.0577 | +0.0331 | +0.0218 | +0.0167 | +0.0142 | +0.0142 | +0.1373 |
| Static ODS-6, U = 2 | +0.0883 | +0.0644 | +0.0430 | +0.0260 | +0.0186 | +0.0154 | +0.0138 | +0.0138 | +0.0883 |
| Static ODS-6, U = 3 | +0.0578 | +0.0412 | +0.0313 | +0.0192 | +0.0160 | +0.0143 | +0.0135 | +0.0135 | +0.0578 |
| Static ODS-6, U = 4 | +0.0340 | +0.0237 | +0.0199 | +0.0135 | +0.0138 | +0.0134 | +0.0132 | +0.0132 | +0.0340 |
| Static ODS-6, U = 5 (overload of K = 4) | +0.0173 | +0.0114 | +0.0084 | +0.0061 | +0.0060 | +0.0070 | +0.0065 | +0.0060 | +0.0173 |
| Static ODS-8, U = 1 | +0.1658 | +0.1193 | +0.0751 | +0.0480 | +0.0351 | +0.0297 | +0.0268 | +0.0268 | +0.1658 |
| Static ODS-8, U = 2 | +0.1168 | +0.0873 | +0.0605 | +0.0408 | +0.0320 | +0.0284 | +0.0264 | +0.0264 | +0.1168 |
| Static ODS-8, U = 3 | +0.0863 | +0.0641 | +0.0488 | +0.0340 | +0.0294 | +0.0272 | +0.0261 | +0.0261 | +0.0863 |
| Static ODS-8, U = 4 | +0.0625 | +0.0466 | +0.0373 | +0.0284 | +0.0272 | +0.0264 | +0.0257 | +0.0257 | +0.0625 |
| Static ODS-8, U = 5 (overload of K = 4) | +0.0458 | +0.0343 | +0.0258 | +0.0210 | +0.0194 | +0.0199 | +0.0191 | +0.0191 | +0.0458 |
| Static ODS-8, U = 6 (overload of K = 4) | +0.0282 | +0.0226 | +0.0172 | +0.0136 | +0.0131 | +0.0132 | +0.0126 | +0.0126 | +0.0282 |
| Static ODS-8, U = 7 (overload of K = 4) | +0.0128 | +0.0097 | +0.0071 | +0.0048 | +0.0051 | +0.0058 | +0.0053 | +0.0048 | +0.0128 |

### A.7 Overload, U = 5..8 on the block dimensioned for K = 4

| quantity, U | 0 dB | 5 dB | 10 dB | 15 dB | 20 dB | 25 dB | 30 dB |
|---|---|---|---|---|---|---|---|
| Proposed CosSim, U = 5 | 0.7748 | 0.8645 | 0.9276 | 0.9645 | 0.9808 | 0.9888 | 0.9918 |
| Proposed CosSim, U = 6 | 0.7572 | 0.8528 | 0.9190 | 0.9572 | 0.9746 | 0.9821 | 0.9854 |
| Proposed CosSim, U = 7 | 0.7418 | 0.8399 | 0.9089 | 0.9484 | 0.9665 | 0.9746 | 0.9781 |
| Proposed CosSim, U = 8 | 0.7282 | 0.8299 | 0.9024 | 0.9424 | 0.9609 | 0.9689 | 0.9726 |
| Proposed minus Learned ODS-5 (3 seed(s)) | +0.0470 | +0.0122 | +0.0055 | +0.0071 | +0.0079 | +0.0098 | +0.0104 |
| Proposed minus Learned ODS-6 (3 seed(s)) | +0.0454 | +0.0140 | +0.0070 | +0.0074 | +0.0078 | +0.0088 | +0.0095 |
| Proposed minus Learned ODS-7 | no run | no run | no run | no run | no run | no run | no run |
| Proposed minus Learned ODS-8 (1 seed(s)) | +0.0432 | +0.0121 | +0.0073 | +0.0057 | +0.0059 | +0.0067 | +0.0072 |
| Proposed minus Static ODS-5 (U = k) | +0.0004 | -0.0016 | -0.0016 | -0.0016 | -0.0010 | +0.0002 | -0.0002 |
| Proposed minus Static ODS-6 (U = k) | -0.0003 | -0.0003 | -0.0002 | -0.0012 | -0.0002 | +0.0003 | +0.0000 |
| Proposed minus Static ODS-7 (U = k) | -0.0010 | -0.0016 | -0.0019 | -0.0023 | -0.0012 | -0.0005 | -0.0008 |
| Proposed minus Static ODS-8 (U = k) | -0.0008 | -0.0003 | +0.0006 | -0.0011 | -0.0005 | +0.0001 | -0.0001 |
| Proposed minus Static ODS-8, U = 5 | +0.0458 | +0.0343 | +0.0258 | +0.0210 | +0.0194 | +0.0199 | +0.0191 |
| Proposed minus Static ODS-8, U = 6 | +0.0282 | +0.0226 | +0.0172 | +0.0136 | +0.0131 | +0.0132 | +0.0126 |
| Proposed minus Static ODS-8, U = 7 | +0.0128 | +0.0097 | +0.0071 | +0.0048 | +0.0051 | +0.0058 | +0.0053 |

- minimum margin of Proposed over Learned ODS-U in overload (U = 5, 6, 8; none at U = 7), 0-30 dB: +0.0055 (U = 5, 10 dB); from 5 dB on: +0.0055 (U = 5, 10 dB); largest +0.0470 (U = 5, 0 dB)
- Static ODS-k with k < U serves only k of the U users (U - k users get nothing); e.g. Static ODS-4 at U = 8 serves 4 of 8. Proposed serves all U users at every load.
- Proposed, U = 1, at 0 / 10 / 20 dB: 0.8948 / 0.9769 / 0.9966; change from U = 1 (Proposed at U minus Proposed at U = 1; negative = drop): +0.0000 / +0.0000 / +0.0000
- Proposed, U = 2, at 0 / 10 / 20 dB: 0.8458 / 0.9623 / 0.9934; change from U = 1 (Proposed at U minus Proposed at U = 1; negative = drop): -0.0490 / -0.0147 / -0.0032
- Proposed, U = 3, at 0 / 10 / 20 dB: 0.8153 / 0.9505 / 0.9908; change from U = 1 (Proposed at U minus Proposed at U = 1; negative = drop): -0.0796 / -0.0264 / -0.0057
- Proposed, U = 4, at 0 / 10 / 20 dB: 0.7915 / 0.9391 / 0.9886; change from U = 1 (Proposed at U minus Proposed at U = 1; negative = drop): -0.1033 / -0.0378 / -0.0080
- Proposed, U = 5, at 0 / 10 / 20 dB: 0.7748 / 0.9276 / 0.9808; change from U = 1 (Proposed at U minus Proposed at U = 1; negative = drop): -0.1200 / -0.0493 / -0.0158
- Proposed, U = 6, at 0 / 10 / 20 dB: 0.7572 / 0.9190 / 0.9746; change from U = 1 (Proposed at U minus Proposed at U = 1; negative = drop): -0.1377 / -0.0579 / -0.0220
- Proposed, U = 7, at 0 / 10 / 20 dB: 0.7418 / 0.9089 / 0.9665; change from U = 1 (Proposed at U minus Proposed at U = 1; negative = drop): -0.1531 / -0.0680 / -0.0300
- Proposed, U = 8, at 0 / 10 / 20 dB: 0.7282 / 0.9024 / 0.9609; change from U = 1 (Proposed at U minus Proposed at U = 1; negative = drop): -0.1666 / -0.0745 / -0.0357

## B. Natural COCO captions of one image (shared content), known gains (Code/shared_masks.py)

Every user of a frame sends a different caption of the same test image (natural shared content), independent Rayleigh block fading, the receiver knows the gains (except the residual-phase runs _ph<deg>abs of B.3, whose receivers use |h| and do not know the effective gain g cos(phi)), per-user energy d_s/U. Paired differences on the same frames, captions, gains and noise. Overlap NMSE reduction = gain of the overlapping masks of Proposition 2 (pruned Proposition 2 at U > K, at most d_s active modes) over the load masks, both with the joint LMMSE receiver; receiver NMSE reduction = gain of the joint receiver over the per-user receiver, both with the load masks (this part is a receiver gain, not a mask gain).

Load masks of sections B and C (load_pu, load_joint): the equal split, every user sends min(floor(d_s/U), d_b) coefficients (--load_rule spec). The Proposed of section A and Fig. 2 uses the rank split (user j sends min(ceil((N - j)/U) 128, d_b)). The two differ at K = 4, U = 5, 7 and at K = 2, U = 5 (e.g. K = 4, U = 5: spec 614/614/614/614/614 = 3070 uses, rank 640/640/640/640/512 = 3072 uses) and coincide elsewhere; the runs with suffix _rank (B.3) use the rank split.

### B.1 Natural captions, K = 4 and K = 2, U = 2..5 (U > K is overload of the block)


#### COCO natural captions, K = 4, U = 2 (light load): data/shared/shared_coco_K4_U2.json

- dataset coco, K = 4, U = 2, 2000 frames, gains rayleigh, SNRs -10/-5/0/5/10/20 dB, receivers know the gains; load masks: spec rule (768/768 coefficients per user)
- content statistics of the design C (fit): kappa_bar = 0.3935 (tr C_model 0.3935, tr C of the content 0.3935, model is the true content: True); kappa_1 = 0.7416; lam-weighted kappa per 128-group 1..6: 0.434/0.151/0.107/0.089/0.075/0.066; mean kappa_i per group: 0.307/0.146/0.106/0.088/0.075/0.065; kappa_i of coefficients 1..10: 0.742/0.154/0.326/0.687/0.630/0.445/0.779/0.596/0.425/0.476

| mode | metric | -10 dB | -5 dB | 0 dB | 5 dB | 10 dB | 20 dB |
|---|---|---|---|---|---|---|---|
| load_pu | CosSim | 0.6940 | 0.7948 | 0.8760 | 0.9351 | 0.9683 | 0.9948 |
| load_pu | weakest user CosSim | 0.6055 | 0.7212 | 0.8212 | 0.8998 | 0.9480 | 0.9909 |
| load_pu | NMSE | 0.4942 | 0.3487 | 0.2197 | 0.1190 | 0.0585 | 0.0099 |
| load_joint | CosSim | 0.7276 | 0.8167 | 0.8889 | 0.9409 | 0.9710 | 0.9951 |
| load_joint | weakest user CosSim | 0.6663 | 0.7619 | 0.8456 | 0.9110 | 0.9532 | 0.9914 |
| load_joint | NMSE | 0.4595 | 0.3228 | 0.2034 | 0.1111 | 0.0551 | 0.0096 |
| gen_joint | CosSim | 0.7337 | 0.8185 | 0.8893 | 0.9409 | 0.9710 | 0.9951 |
| gen_joint | weakest user CosSim | 0.6731 | 0.7639 | 0.8460 | 0.9110 | 0.9532 | 0.9914 |
| gen_joint | NMSE | 0.4509 | 0.3201 | 0.2027 | 0.1110 | 0.0551 | 0.0096 |
| gen_eq | CosSim | 0.6902 | 0.7467 | 0.7803 | 0.8036 | 0.8123 | 0.8191 |
| gen_eq | weakest user CosSim | 0.6117 | 0.6599 | 0.6865 | 0.7054 | 0.7122 | 0.7198 |
| gen_eq | NMSE | 0.5109 | 0.4275 | 0.3755 | 0.3379 | 0.3230 | 0.3124 |
| gen_k0 | CosSim | 0.7281 | 0.8176 | 0.8884 | 0.9409 | 0.9709 | 0.9950 |
| gen_k0 | weakest user CosSim | 0.6673 | 0.7634 | 0.8449 | 0.9112 | 0.9532 | 0.9912 |
| gen_k0 | NMSE | 0.4585 | 0.3220 | 0.2039 | 0.1108 | 0.0552 | 0.0096 |
| gen_pc | CosSim | 0.7238 | 0.8106 | 0.8843 | 0.9381 | 0.9699 | 0.9950 |
| gen_pc | weakest user CosSim | 0.6561 | 0.7496 | 0.8368 | 0.9056 | 0.9512 | 0.9911 |
| gen_pc | NMSE | 0.4630 | 0.3307 | 0.2101 | 0.1152 | 0.0567 | 0.0097 |

| SNR | gen_joint - load_joint CosSim [95% CI] | gen_joint - load_joint weakest [95% CI] | overlap NMSE reduction 1 - gen_joint/load_joint [95% CI] | load_joint - load_pu CosSim [95% CI] | receiver NMSE reduction 1 - load_joint/load_pu [95% CI] | total NMSE reduction 1 - gen_joint/load_pu [95% CI] |
|---|---|---|---|---|---|---|
| -10 dB | +0.0061 [+0.0058, +0.0064] | +0.0068 [+0.0063, +0.0072] | +1.87 % [+1.77, +1.97] | +0.0335 [+0.0314, +0.0360] | +7.02 % [+6.72, +7.37] | +8.76 % [+8.44, +9.13] |
| -5 dB | +0.0018 [+0.0017, +0.0019] | +0.0020 [+0.0019, +0.0022] | +0.85 % [+0.81, +0.90] | +0.0220 [+0.0202, +0.0236] | +7.41 % [+7.01, +7.78] | +8.20 % [+7.80, +8.58] |
| 0 dB | +0.0004 [+0.0003, +0.0004] | +0.0004 [+0.0004, +0.0005] | +0.31 % [+0.29, +0.33] | +0.0129 [+0.0117, +0.0145] | +7.44 % [+7.01, +7.95] | +7.72 % [+7.28, +8.24] |
| 5 dB | +0.0000 [+0.0000, +0.0001] | +0.0001 [+0.0001, +0.0001] | +0.07 % [+0.07, +0.08] | +0.0058 [+0.0050, +0.0067] | +6.71 % [+6.12, +7.28] | +6.77 % [+6.19, +7.35] |
| 10 dB | +0.0000 [+0.0000, +0.0000] | +0.0000 [+0.0000, +0.0000] | +0.01 % [+0.01, +0.02] | +0.0026 [+0.0020, +0.0034] | +5.77 % [+4.85, +6.79] | +5.79 % [+4.86, +6.81] |
| 20 dB | +0.0000 [-0.0000, +0.0000] | +0.0000 [-0.0000, +0.0000] | +0.00 % [-0.00, +0.00] | +0.0003 [+0.0001, +0.0004] | +3.72 % [+2.28, +5.16] | +3.72 % [+2.28, +5.16] |

| SNR | gen_eq - load_joint CosSim [95% CI]; NMSE reduction 1 - gen_eq/load_joint [95% CI] | gen_k0 - load_joint CosSim [95% CI]; NMSE reduction 1 - gen_k0/load_joint [95% CI] | gen_pc - load_joint CosSim [95% CI]; NMSE reduction 1 - gen_pc/load_joint [95% CI] |
|---|---|---|---|
| -10 dB | -0.0374 [-0.0402, -0.0346]; -11.18 % [-12.07, -10.32] | +0.0005 [-0.0012, +0.0024]; +0.21 % [-0.23, +0.64] | -0.0038 [-0.0046, -0.0030]; -0.75 % [-0.94, -0.57] |
| -5 dB | -0.0700 [-0.0730, -0.0674]; -32.41 % [-34.09, -30.93] | +0.0009 [-0.0002, +0.0021]; +0.24 % [-0.16, +0.66] | -0.0062 [-0.0069, -0.0055]; -2.43 % [-2.67, -2.20] |
| 0 dB | -0.1086 [-0.1117, -0.1055]; -84.63 % [-88.34, -81.12] | -0.0005 [-0.0012, +0.0003]; -0.27 % [-0.77, +0.22] | -0.0046 [-0.0052, -0.0041]; -3.30 % [-3.59, -3.01] |
| 5 dB | -0.1373 [-0.1403, -0.1344]; -204.24 % [-213.89, -195.34] | +0.0000 [-0.0005, +0.0005]; +0.21 % [-0.43, +0.85] | -0.0028 [-0.0032, -0.0024]; -3.74 % [-4.09, -3.37] |
| 10 dB | -0.1587 [-0.1617, -0.1559]; -485.88 % [-514.40, -458.77] | -0.0000 [-0.0003, +0.0004]; -0.15 % [-0.99, +0.78] | -0.0010 [-0.0013, -0.0008]; -2.88 % [-3.39, -2.35] |
| 20 dB | -0.1760 [-0.1788, -0.1731]; -3169.20 % [-3533.51, -2815.37] | -0.0001 [-0.0002, +0.0001]; -0.65 % [-2.82, +1.12] | -0.0001 [-0.0002, -0.0001]; -1.83 % [-2.56, -1.04] |

| SNR | load masks: uses / coefficients sent | gen_joint: uses | gen_joint overlap index t | gen_joint t_overlap_modes | modes active / pruned | partial coefficients | n_data |
|---|---|---|---|---|---|---|---|
| -10 dB | 256 / 128 | 328 | 0.3110 | 0.3110 | 262 / 0 | 0 | 3072 |
| -5 dB | 602 / 301 | 694 | 0.1489 | 0.1489 | 613 / 0 | 0 | 3072 |
| 0 dB | 1312 / 656 | 1380 | 0.0669 | 0.0669 | 1317 / 0 | 0 | 3072 |
| 5 dB | 1534 / 767 | 1534 | 0.0226 | 0.0226 | 1534 / 0 | 0 | 3072 |
| 10 dB | 1534 / 767 | 1534 | 0.0072 | 0.0072 | 1534 / 0 | 0 | 3072 |
| 20 dB | 1534 / 767 | 1534 | 0.0007 | 0.0007 | 1534 / 0 | 0 | 3072 |

#### COCO natural captions, K = 4, U = 3 (light load): data/shared/shared_coco_K4_U3.json

- dataset coco, K = 4, U = 3, 2000 frames, gains rayleigh, SNRs -10/-5/0/5/10/20 dB, receivers know the gains; load masks: spec rule (768/768/768 coefficients per user)
- content statistics of the design C (fit): kappa_bar = 0.3935 (tr C_model 0.3935, tr C of the content 0.3935, model is the true content: True); kappa_1 = 0.7416; lam-weighted kappa per 128-group 1..6: 0.434/0.151/0.107/0.089/0.075/0.066; mean kappa_i per group: 0.307/0.146/0.106/0.088/0.075/0.065; kappa_i of coefficients 1..10: 0.742/0.154/0.326/0.687/0.630/0.445/0.779/0.596/0.425/0.476

| mode | metric | -10 dB | -5 dB | 0 dB | 5 dB | 10 dB | 20 dB |
|---|---|---|---|---|---|---|---|
| load_pu | CosSim | 0.6485 | 0.7607 | 0.8504 | 0.9167 | 0.9606 | 0.9924 |
| load_pu | weakest user CosSim | 0.4986 | 0.6287 | 0.7466 | 0.8425 | 0.9180 | 0.9814 |
| load_pu | NMSE | 0.5559 | 0.4006 | 0.2622 | 0.1504 | 0.0735 | 0.0145 |
| load_joint | CosSim | 0.7005 | 0.7961 | 0.8710 | 0.9274 | 0.9646 | 0.9930 |
| load_joint | weakest user CosSim | 0.6074 | 0.7085 | 0.7971 | 0.8704 | 0.9292 | 0.9834 |
| load_joint | NMSE | 0.5004 | 0.3583 | 0.2353 | 0.1359 | 0.0676 | 0.0135 |
| gen_joint | CosSim | 0.7134 | 0.8000 | 0.8719 | 0.9275 | 0.9647 | 0.9930 |
| gen_joint | weakest user CosSim | 0.6225 | 0.7132 | 0.7982 | 0.8706 | 0.9292 | 0.9834 |
| gen_joint | NMSE | 0.4827 | 0.3523 | 0.2338 | 0.1356 | 0.0676 | 0.0135 |
| gen_eq | CosSim | 0.6611 | 0.7050 | 0.7325 | 0.7465 | 0.7531 | 0.7550 |
| gen_eq | weakest user CosSim | 0.5543 | 0.5915 | 0.6142 | 0.6243 | 0.6298 | 0.6305 |
| gen_eq | NMSE | 0.5533 | 0.4920 | 0.4513 | 0.4296 | 0.4196 | 0.4161 |
| gen_k0 | CosSim | 0.6999 | 0.7959 | 0.8712 | 0.9273 | 0.9647 | 0.9931 |
| gen_k0 | weakest user CosSim | 0.6060 | 0.7084 | 0.7966 | 0.8702 | 0.9292 | 0.9835 |
| gen_k0 | NMSE | 0.5013 | 0.3585 | 0.2352 | 0.1360 | 0.0676 | 0.0135 |
| gen_pc | CosSim | 0.7010 | 0.7900 | 0.8652 | 0.9236 | 0.9632 | 0.9928 |
| gen_pc | weakest user CosSim | 0.5999 | 0.6939 | 0.7834 | 0.8610 | 0.9253 | 0.9827 |
| gen_pc | NMSE | 0.4983 | 0.3663 | 0.2438 | 0.1416 | 0.0699 | 0.0139 |

| SNR | gen_joint - load_joint CosSim [95% CI] | gen_joint - load_joint weakest [95% CI] | overlap NMSE reduction 1 - gen_joint/load_joint [95% CI] | load_joint - load_pu CosSim [95% CI] | receiver NMSE reduction 1 - load_joint/load_pu [95% CI] | total NMSE reduction 1 - gen_joint/load_pu [95% CI] |
|---|---|---|---|---|---|---|
| -10 dB | +0.0130 [+0.0125, +0.0134] | +0.0151 [+0.0144, +0.0158] | +3.54 % [+3.42, +3.66] | +0.0520 [+0.0497, +0.0541] | +9.98 % [+9.66, +10.28] | +13.16 % [+12.82, +13.50] |
| -5 dB | +0.0039 [+0.0037, +0.0040] | +0.0048 [+0.0045, +0.0050] | +1.66 % [+1.60, +1.72] | +0.0354 [+0.0335, +0.0372] | +10.57 % [+10.23, +10.93] | +12.05 % [+11.70, +12.43] |
| 0 dB | +0.0009 [+0.0008, +0.0009] | +0.0011 [+0.0010, +0.0011] | +0.62 % [+0.59, +0.65] | +0.0206 [+0.0192, +0.0220] | +10.26 % [+9.81, +10.69] | +10.82 % [+10.37, +11.25] |
| 5 dB | +0.0001 [+0.0001, +0.0002] | +0.0002 [+0.0002, +0.0002] | +0.18 % [+0.17, +0.19] | +0.0107 [+0.0097, +0.0117] | +9.65 % [+9.13, +10.15] | +9.81 % [+9.30, +10.32] |
| 10 dB | +0.0000 [+0.0000, +0.0000] | +0.0000 [+0.0000, +0.0000] | +0.04 % [+0.04, +0.04] | +0.0041 [+0.0035, +0.0048] | +8.09 % [+7.40, +8.88] | +8.12 % [+7.44, +8.92] |
| 20 dB | +0.0000 [-0.0000, +0.0000] | +0.0000 [-0.0000, +0.0000] | +0.00 % [-0.00, +0.00] | +0.0006 [+0.0004, +0.0009] | +6.45 % [+4.79, +8.03] | +6.45 % [+4.79, +8.03] |

| SNR | gen_eq - load_joint CosSim [95% CI]; NMSE reduction 1 - gen_eq/load_joint [95% CI] | gen_k0 - load_joint CosSim [95% CI]; NMSE reduction 1 - gen_k0/load_joint [95% CI] | gen_pc - load_joint CosSim [95% CI]; NMSE reduction 1 - gen_pc/load_joint [95% CI] |
|---|---|---|---|
| -10 dB | -0.0393 [-0.0420, -0.0366]; -10.56 % [-11.32, -9.80] | -0.0006 [-0.0024, +0.0009]; -0.17 % [-0.59, +0.18] | +0.0006 [-0.0003, +0.0014]; +0.42 % [+0.23, +0.60] |
| -5 dB | -0.0911 [-0.0939, -0.0882]; -37.32 % [-38.70, -35.94] | -0.0002 [-0.0012, +0.0008]; -0.07 % [-0.44, +0.31] | -0.0061 [-0.0066, -0.0056]; -2.24 % [-2.43, -2.05] |
| 0 dB | -0.1386 [-0.1416, -0.1356]; -91.79 % [-94.71, -89.05] | +0.0001 [-0.0005, +0.0008]; +0.05 % [-0.35, +0.43] | -0.0059 [-0.0063, -0.0054]; -3.62 % [-3.83, -3.39] |
| 5 dB | -0.1809 [-0.1840, -0.1777]; -216.13 % [-223.25, -208.93] | -0.0001 [-0.0005, +0.0003]; -0.07 % [-0.47, +0.36] | -0.0038 [-0.0042, -0.0035]; -4.21 % [-4.50, -3.93] |
| 10 dB | -0.2116 [-0.2146, -0.2085]; -520.82 % [-542.21, -498.65] | +0.0000 [-0.0002, +0.0003]; +0.02 % [-0.54, +0.55] | -0.0015 [-0.0017, -0.0013]; -3.50 % [-3.83, -3.20] |
| 20 dB | -0.2380 [-0.2412, -0.2347]; -2973.44 % [-3205.04, -2764.51] | +0.0001 [-0.0000, +0.0002]; +0.66 % [-0.42, +1.84] | -0.0002 [-0.0003, -0.0002]; -2.61 % [-3.29, -1.94] |

| SNR | load masks: uses / coefficients sent | gen_joint: uses | gen_joint overlap index t | gen_joint t_overlap_modes | modes active / pruned | partial coefficients | n_data |
|---|---|---|---|---|---|---|---|
| -10 dB | 285 / 95 | 477 | 0.3366 | 0.3366 | 293 / 0 | 0 | 3072 |
| -5 dB | 669 / 223 | 912 | 0.1648 | 0.1648 | 692 / 0 | 0 | 3072 |
| 0 dB | 1548 / 516 | 1794 | 0.0779 | 0.0779 | 1564 / 0 | 0 | 3072 |
| 5 dB | 2301 / 767 | 2301 | 0.0292 | 0.0292 | 2299 / 0 | 0 | 3072 |
| 10 dB | 2301 / 767 | 2301 | 0.0092 | 0.0092 | 2301 / 0 | 0 | 3072 |
| 20 dB | 2301 / 767 | 2301 | 0.0009 | 0.0009 | 2301 / 0 | 0 | 3072 |

#### COCO natural captions, K = 4, U = 4 (full load): data/shared/shared_coco_K4_U4.json

- dataset coco, K = 4, U = 4, 2000 frames, gains rayleigh, SNRs -10/-5/0/5/10/20 dB, receivers know the gains; load masks: spec rule (768/768/768/768 coefficients per user)
- content statistics of the design C (fit): kappa_bar = 0.3935 (tr C_model 0.3935, tr C of the content 0.3935, model is the true content: True); kappa_1 = 0.7416; lam-weighted kappa per 128-group 1..6: 0.434/0.151/0.107/0.089/0.075/0.066; mean kappa_i per group: 0.307/0.146/0.106/0.088/0.075/0.065; kappa_i of coefficients 1..10: 0.742/0.154/0.326/0.687/0.630/0.445/0.779/0.596/0.425/0.476

| mode | metric | -10 dB | -5 dB | 0 dB | 5 dB | 10 dB | 20 dB |
|---|---|---|---|---|---|---|---|
| load_pu | CosSim | 0.6167 | 0.7352 | 0.8304 | 0.9012 | 0.9514 | 0.9911 |
| load_pu | weakest user CosSim | 0.4138 | 0.5578 | 0.6845 | 0.7870 | 0.8836 | 0.9742 |
| load_pu | NMSE | 0.5939 | 0.4372 | 0.2944 | 0.1769 | 0.0899 | 0.0171 |
| load_joint | CosSim | 0.6844 | 0.7806 | 0.8562 | 0.9152 | 0.9571 | 0.9917 |
| load_joint | weakest user CosSim | 0.5670 | 0.6717 | 0.7562 | 0.8308 | 0.9030 | 0.9765 |
| load_joint | NMSE | 0.5241 | 0.3836 | 0.2605 | 0.1576 | 0.0817 | 0.0162 |
| gen_joint | CosSim | 0.7032 | 0.7866 | 0.8577 | 0.9155 | 0.9571 | 0.9917 |
| gen_joint | weakest user CosSim | 0.5887 | 0.6789 | 0.7580 | 0.8312 | 0.9030 | 0.9765 |
| gen_joint | NMSE | 0.4987 | 0.3744 | 0.2581 | 0.1572 | 0.0816 | 0.0162 |
| gen_eq | CosSim | 0.6481 | 0.6861 | 0.7052 | 0.7177 | 0.7218 | 0.7234 |
| gen_eq | weakest user CosSim | 0.5179 | 0.5517 | 0.5649 | 0.5740 | 0.5826 | 0.5806 |
| gen_eq | NMSE | 0.5721 | 0.5203 | 0.4927 | 0.4738 | 0.4682 | 0.4660 |
| gen_k0 | CosSim | 0.6845 | 0.7803 | 0.8560 | 0.9153 | 0.9569 | 0.9917 |
| gen_k0 | weakest user CosSim | 0.5659 | 0.6712 | 0.7556 | 0.8311 | 0.9022 | 0.9766 |
| gen_k0 | NMSE | 0.5238 | 0.3840 | 0.2606 | 0.1575 | 0.0819 | 0.0161 |
| gen_pc | CosSim | 0.6896 | 0.7753 | 0.8500 | 0.9107 | 0.9551 | 0.9915 |
| gen_pc | weakest user CosSim | 0.5653 | 0.6568 | 0.7405 | 0.8178 | 0.8968 | 0.9757 |
| gen_pc | NMSE | 0.5159 | 0.3903 | 0.2695 | 0.1645 | 0.0848 | 0.0165 |

| SNR | gen_joint - load_joint CosSim [95% CI] | gen_joint - load_joint weakest [95% CI] | overlap NMSE reduction 1 - gen_joint/load_joint [95% CI] | load_joint - load_pu CosSim [95% CI] | receiver NMSE reduction 1 - load_joint/load_pu [95% CI] | total NMSE reduction 1 - gen_joint/load_pu [95% CI] |
|---|---|---|---|---|---|---|
| -10 dB | +0.0188 [+0.0183, +0.0195] | +0.0216 [+0.0207, +0.0226] | +4.86 % [+4.71, +5.01] | +0.0677 [+0.0656, +0.0699] | +11.76 % [+11.48, +12.03] | +16.04 % [+15.72, +16.36] |
| -5 dB | +0.0060 [+0.0058, +0.0062] | +0.0072 [+0.0069, +0.0076] | +2.38 % [+2.31, +2.46] | +0.0454 [+0.0436, +0.0473] | +12.28 % [+11.97, +12.61] | +14.36 % [+14.04, +14.71] |
| 0 dB | +0.0015 [+0.0014, +0.0015] | +0.0018 [+0.0017, +0.0020] | +0.93 % [+0.89, +0.97] | +0.0258 [+0.0245, +0.0270] | +11.52 % [+11.14, +11.87] | +12.33 % [+11.94, +12.70] |
| 5 dB | +0.0003 [+0.0002, +0.0003] | +0.0004 [+0.0004, +0.0004] | +0.29 % [+0.27, +0.30] | +0.0140 [+0.0131, +0.0150] | +10.92 % [+10.49, +11.38] | +11.17 % [+10.74, +11.63] |
| 10 dB | +0.0000 [+0.0000, +0.0000] | +0.0001 [+0.0000, +0.0001] | +0.06 % [+0.06, +0.07] | +0.0057 [+0.0051, +0.0063] | +9.19 % [+8.62, +9.80] | +9.24 % [+8.68, +9.86] |
| 20 dB | +0.0000 [+0.0000, +0.0000] | +0.0000 [-0.0000, +0.0000] | +0.00 % [+0.00, +0.00] | +0.0006 [+0.0005, +0.0008] | +5.59 % [+4.72, +6.56] | +5.59 % [+4.72, +6.56] |

| SNR | gen_eq - load_joint CosSim [95% CI]; NMSE reduction 1 - gen_eq/load_joint [95% CI] | gen_k0 - load_joint CosSim [95% CI]; NMSE reduction 1 - gen_k0/load_joint [95% CI] | gen_pc - load_joint CosSim [95% CI]; NMSE reduction 1 - gen_pc/load_joint [95% CI] |
|---|---|---|---|
| -10 dB | -0.0363 [-0.0388, -0.0338]; -9.15 % [-9.78, -8.53] | +0.0001 [-0.0012, +0.0015]; +0.05 % [-0.22, +0.35] | +0.0051 [+0.0042, +0.0060]; +1.58 % [+1.37, +1.77] |
| -5 dB | -0.0945 [-0.0971, -0.0918]; -35.65 % [-36.82, -34.45] | -0.0003 [-0.0012, +0.0005]; -0.10 % [-0.40, +0.17] | -0.0054 [-0.0059, -0.0049]; -1.76 % [-1.92, -1.59] |
| 0 dB | -0.1510 [-0.1539, -0.1483]; -89.15 % [-91.44, -86.92] | -0.0002 [-0.0008, +0.0004]; -0.04 % [-0.35, +0.25] | -0.0062 [-0.0066, -0.0058]; -3.48 % [-3.65, -3.29] |
| 5 dB | -0.1975 [-0.2004, -0.1946]; -200.55 % [-205.93, -195.16] | +0.0001 [-0.0003, +0.0004]; +0.10 % [-0.24, +0.42] | -0.0045 [-0.0048, -0.0042]; -4.34 % [-4.55, -4.13] |
| 10 dB | -0.2353 [-0.2384, -0.2322]; -473.47 % [-488.81, -459.20] | -0.0002 [-0.0004, +0.0000]; -0.32 % [-0.73, +0.10] | -0.0020 [-0.0022, -0.0018]; -3.88 % [-4.16, -3.61] |
| 20 dB | -0.2684 [-0.2715, -0.2654]; -2783.53 % [-2939.45, -2636.23] | +0.0000 [-0.0001, +0.0001]; +0.21 % [-0.62, +1.07] | -0.0002 [-0.0003, -0.0002]; -2.27 % [-2.75, -1.78] |

| SNR | load masks: uses / coefficients sent | gen_joint: uses | gen_joint overlap index t | gen_joint t_overlap_modes | modes active / pruned | partial coefficients | n_data |
|---|---|---|---|---|---|---|---|
| -10 dB | 304 / 76 | 640 | 0.3464 | 0.3464 | 307 / 0 | 0 | 3072 |
| -5 dB | 716 / 179 | 1140 | 0.1736 | 0.1736 | 756 / 0 | 0 | 3072 |
| 0 dB | 1692 / 423 | 2176 | 0.0840 | 0.0840 | 1711 / 0 | 0 | 3072 |
| 5 dB | 3048 / 762 | 3064 | 0.0344 | 0.0344 | 3037 / 0 | 0 | 3072 |
| 10 dB | 3068 / 767 | 3068 | 0.0109 | 0.0109 | 3068 / 0 | 0 | 3072 |
| 20 dB | 3068 / 767 | 3068 | 0.0011 | 0.0011 | 3068 / 0 | 0 | 3072 |

#### COCO natural captions, K = 4, U = 5 (overload; gen_joint = pruned Proposition 2): data/shared/shared_coco_K4_U5.json

- dataset coco, K = 4, U = 5, 2000 frames, gains rayleigh, SNRs -10/-5/0/5/10/20 dB, receivers know the gains; load masks: spec rule (614/614/614/614/614 coefficients per user)
- content statistics of the design C (fit): kappa_bar = 0.3935 (tr C_model 0.3935, tr C of the content 0.3935, model is the true content: True); kappa_1 = 0.7416; lam-weighted kappa per 128-group 1..6: 0.434/0.151/0.107/0.089/0.075/0.066; mean kappa_i per group: 0.307/0.146/0.106/0.088/0.075/0.065; kappa_i of coefficients 1..10: 0.742/0.154/0.326/0.687/0.630/0.445/0.779/0.596/0.425/0.476

| mode | metric | -10 dB | -5 dB | 0 dB | 5 dB | 10 dB | 20 dB |
|---|---|---|---|---|---|---|---|
| load_pu | CosSim | 0.5966 | 0.7124 | 0.8148 | 0.8868 | 0.9435 | 0.9865 |
| load_pu | weakest user CosSim | 0.3613 | 0.4965 | 0.6349 | 0.7365 | 0.8525 | 0.9631 |
| load_pu | NMSE | 0.6195 | 0.4701 | 0.3191 | 0.2007 | 0.1045 | 0.0261 |
| load_joint | CosSim | 0.6724 | 0.7664 | 0.8469 | 0.9049 | 0.9502 | 0.9873 |
| load_joint | weakest user CosSim | 0.5379 | 0.6377 | 0.7328 | 0.8006 | 0.8795 | 0.9669 |
| load_joint | NMSE | 0.5411 | 0.4059 | 0.2770 | 0.1762 | 0.0947 | 0.0249 |
| gen_joint | CosSim | 0.6969 | 0.7746 | 0.8490 | 0.9053 | 0.9504 | 0.9874 |
| gen_joint | weakest user CosSim | 0.5654 | 0.6475 | 0.7354 | 0.8012 | 0.8795 | 0.9668 |
| gen_joint | NMSE | 0.5082 | 0.3935 | 0.2735 | 0.1754 | 0.0944 | 0.0247 |
| gen_eq | CosSim | 0.6417 | 0.6727 | 0.6909 | 0.6985 | 0.7024 | 0.7037 |
| gen_eq | weakest user CosSim | 0.4935 | 0.5217 | 0.5374 | 0.5389 | 0.5443 | 0.5435 |
| gen_eq | NMSE | 0.5813 | 0.5399 | 0.5146 | 0.5030 | 0.4976 | 0.4957 |
| gen_k0 | CosSim | 0.6734 | 0.7660 | 0.8469 | 0.9048 | 0.9504 | 0.9873 |
| gen_k0 | weakest user CosSim | 0.5383 | 0.6377 | 0.7331 | 0.8018 | 0.8807 | 0.9667 |
| gen_k0 | NMSE | 0.5398 | 0.4066 | 0.2771 | 0.1764 | 0.0945 | 0.0249 |
| gen_pc | CosSim | 0.6835 | 0.7625 | 0.8405 | 0.9004 | 0.9482 | 0.9871 |
| gen_pc | weakest user CosSim | 0.5419 | 0.6233 | 0.7156 | 0.7870 | 0.8716 | 0.9653 |
| gen_pc | NMSE | 0.5252 | 0.4104 | 0.2863 | 0.1831 | 0.0979 | 0.0252 |

| SNR | gen_joint - load_joint CosSim [95% CI] | gen_joint - load_joint weakest [95% CI] | overlap NMSE reduction 1 - gen_joint/load_joint [95% CI] | load_joint - load_pu CosSim [95% CI] | receiver NMSE reduction 1 - load_joint/load_pu [95% CI] | total NMSE reduction 1 - gen_joint/load_pu [95% CI] |
|---|---|---|---|---|---|---|
| -10 dB | +0.0245 [+0.0238, +0.0252] | +0.0274 [+0.0261, +0.0286] | +6.07 % [+5.89, +6.24] | +0.0758 [+0.0737, +0.0779] | +12.66 % [+12.38, +12.93] | +17.96 % [+17.63, +18.29] |
| -5 dB | +0.0083 [+0.0080, +0.0085] | +0.0098 [+0.0094, +0.0103] | +3.06 % [+2.97, +3.15] | +0.0539 [+0.0522, +0.0557] | +13.65 % [+13.36, +13.96] | +16.30 % [+15.97, +16.62] |
| 0 dB | +0.0021 [+0.0020, +0.0022] | +0.0026 [+0.0024, +0.0028] | +1.25 % [+1.20, +1.30] | +0.0321 [+0.0308, +0.0336] | +13.19 % [+12.84, +13.51] | +14.27 % [+13.90, +14.61] |
| 5 dB | +0.0004 [+0.0004, +0.0005] | +0.0006 [+0.0005, +0.0007] | +0.45 % [+0.42, +0.47] | +0.0181 [+0.0170, +0.0191] | +12.18 % [+11.76, +12.60] | +12.57 % [+12.14, +13.00] |
| 10 dB | +0.0002 [+0.0002, +0.0002] | +0.0001 [+0.0000, +0.0001] | +0.38 % [+0.36, +0.40] | +0.0067 [+0.0062, +0.0072] | +9.33 % [+8.89, +9.77] | +9.67 % [+9.23, +10.12] |
| 20 dB | +0.0001 [+0.0001, +0.0001] | -0.0002 [-0.0002, -0.0001] | +0.96 % [+0.89, +1.03] | +0.0008 [+0.0006, +0.0010] | +4.74 % [+4.09, +5.48] | +5.66 % [+5.00, +6.38] |

| SNR | gen_eq - load_joint CosSim [95% CI]; NMSE reduction 1 - gen_eq/load_joint [95% CI] | gen_k0 - load_joint CosSim [95% CI]; NMSE reduction 1 - gen_k0/load_joint [95% CI] | gen_pc - load_joint CosSim [95% CI]; NMSE reduction 1 - gen_pc/load_joint [95% CI] |
|---|---|---|---|
| -10 dB | -0.0306 [-0.0330, -0.0282]; -7.43 % [-8.00, -6.82] | +0.0011 [-0.0002, +0.0024]; +0.24 % [-0.03, +0.53] | +0.0111 [+0.0102, +0.0120]; +2.92 % [+2.71, +3.12] |
| -5 dB | -0.0937 [-0.0961, -0.0910]; -33.01 % [-34.01, -32.05] | -0.0004 [-0.0012, +0.0004]; -0.17 % [-0.45, +0.09] | -0.0039 [-0.0044, -0.0034]; -1.10 % [-1.27, -0.95] |
| 0 dB | -0.1561 [-0.1590, -0.1532]; -85.78 % [-87.79, -83.81] | -0.0001 [-0.0006, +0.0005]; -0.05 % [-0.33, +0.21] | -0.0064 [-0.0068, -0.0061]; -3.36 % [-3.51, -3.20] |
| 5 dB | -0.2063 [-0.2093, -0.2033]; -185.44 % [-190.01, -180.82] | -0.0001 [-0.0005, +0.0003]; -0.10 % [-0.41, +0.19] | -0.0045 [-0.0047, -0.0042]; -3.91 % [-4.09, -3.72] |
| 10 dB | -0.2478 [-0.2507, -0.2450]; -425.39 % [-435.96, -414.72] | +0.0002 [-0.0001, +0.0004]; +0.23 % [-0.12, +0.58] | -0.0020 [-0.0022, -0.0018]; -3.37 % [-3.59, -3.15] |
| 20 dB | -0.2837 [-0.2867, -0.2807]; -1891.68 % [-1956.33, -1832.31] | -0.0000 [-0.0001, +0.0000]; -0.10 % [-0.59, +0.40] | -0.0002 [-0.0003, -0.0001]; -1.07 % [-1.41, -0.72] |

| SNR | load masks: uses / coefficients sent | gen_joint: uses | gen_joint overlap index t | gen_joint t_overlap_modes | modes active / pruned | partial coefficients | n_data |
|---|---|---|---|---|---|---|---|
| -10 dB | 325 / 65 | 341 | 0.3525 | 0.3525 | 341 / 0 | 0 | 3072 |
| -5 dB | 760 / 152 | 798 | 0.1791 | 0.1791 | 798 / 0 | 0 | 3072 |
| 0 dB | 1795 / 359 | 1833 | 0.0877 | 0.0877 | 1833 / 0 | 0 | 3072 |
| 5 dB | 3070 / 614 | 3072 | 0.0391 | 0.0391 | 3072 / 536 | 116 | 3072 |
| 10 dB | 3070 / 614 | 3072 | 0.0177 | 0.0176 | 3072 / 763 | 86 | 3072 |
| 20 dB | 3070 / 614 | 3072 | 0.0083 | 0.0082 | 3072 / 763 | 76 | 3072 |

#### COCO natural captions, K = 2, U = 2 (full load): data/shared/shared_coco_K2_U2.json

- dataset coco, K = 2, U = 2, 2000 frames, gains rayleigh, SNRs -10/-5/0/5/10/20 dB, receivers know the gains; load masks: spec rule (768/768 coefficients per user)
- content statistics of the design C (fit): kappa_bar = 0.3935 (tr C_model 0.3935, tr C of the content 0.3935, model is the true content: True); kappa_1 = 0.7416; lam-weighted kappa per 128-group 1..6: 0.434/0.151/0.107/0.089/0.075/0.066; mean kappa_i per group: 0.307/0.146/0.106/0.088/0.075/0.065; kappa_i of coefficients 1..10: 0.742/0.154/0.326/0.687/0.630/0.445/0.779/0.596/0.425/0.476

| mode | metric | -10 dB | -5 dB | 0 dB | 5 dB | 10 dB | 20 dB |
|---|---|---|---|---|---|---|---|
| load_pu | CosSim | 0.6231 | 0.7354 | 0.8296 | 0.9024 | 0.9494 | 0.9909 |
| load_pu | weakest user CosSim | 0.5270 | 0.6529 | 0.7639 | 0.8551 | 0.9195 | 0.9843 |
| load_pu | NMSE | 0.5864 | 0.4369 | 0.2955 | 0.1755 | 0.0925 | 0.0172 |
| load_joint | CosSim | 0.6627 | 0.7635 | 0.8477 | 0.9120 | 0.9540 | 0.9915 |
| load_joint | weakest user CosSim | 0.5980 | 0.7044 | 0.7976 | 0.8734 | 0.9284 | 0.9854 |
| load_joint | NMSE | 0.5488 | 0.4056 | 0.2733 | 0.1627 | 0.0865 | 0.0164 |
| gen_joint | CosSim | 0.6739 | 0.7675 | 0.8487 | 0.9122 | 0.9540 | 0.9915 |
| gen_joint | weakest user CosSim | 0.6104 | 0.7090 | 0.7988 | 0.8736 | 0.9284 | 0.9854 |
| gen_joint | NMSE | 0.5343 | 0.3997 | 0.2717 | 0.1624 | 0.0864 | 0.0164 |
| gen_eq | CosSim | 0.6456 | 0.7146 | 0.7594 | 0.7928 | 0.8077 | 0.8186 |
| gen_eq | weakest user CosSim | 0.5711 | 0.6329 | 0.6693 | 0.6969 | 0.7086 | 0.7194 |
| gen_eq | NMSE | 0.5709 | 0.4749 | 0.4084 | 0.3555 | 0.3306 | 0.3133 |
| gen_k0 | CosSim | 0.6633 | 0.7646 | 0.8469 | 0.9121 | 0.9539 | 0.9915 |
| gen_k0 | weakest user CosSim | 0.5992 | 0.7060 | 0.7966 | 0.8738 | 0.9283 | 0.9853 |
| gen_k0 | NMSE | 0.5478 | 0.4048 | 0.2742 | 0.1623 | 0.0867 | 0.0164 |
| gen_pc | CosSim | 0.6641 | 0.7583 | 0.8420 | 0.9078 | 0.9521 | 0.9913 |
| gen_pc | weakest user CosSim | 0.5939 | 0.6929 | 0.7867 | 0.8653 | 0.9247 | 0.9849 |
| gen_pc | NMSE | 0.5455 | 0.4114 | 0.2812 | 0.1688 | 0.0894 | 0.0167 |

| SNR | gen_joint - load_joint CosSim [95% CI] | gen_joint - load_joint weakest [95% CI] | overlap NMSE reduction 1 - gen_joint/load_joint [95% CI] | load_joint - load_pu CosSim [95% CI] | receiver NMSE reduction 1 - load_joint/load_pu [95% CI] | total NMSE reduction 1 - gen_joint/load_pu [95% CI] |
|---|---|---|---|---|---|---|
| -10 dB | +0.0112 [+0.0106, +0.0118] | +0.0124 [+0.0115, +0.0131] | +2.65 % [+2.50, +2.78] | +0.0395 [+0.0373, +0.0421] | +6.40 % [+6.13, +6.71] | +8.88 % [+8.55, +9.22] |
| -5 dB | +0.0040 [+0.0038, +0.0042] | +0.0047 [+0.0044, +0.0049] | +1.45 % [+1.38, +1.52] | +0.0281 [+0.0262, +0.0300] | +7.15 % [+6.80, +7.49] | +8.50 % [+8.13, +8.85] |
| 0 dB | +0.0010 [+0.0009, +0.0010] | +0.0011 [+0.0011, +0.0012] | +0.57 % [+0.54, +0.60] | +0.0181 [+0.0167, +0.0198] | +7.49 % [+7.10, +7.93] | +8.02 % [+7.62, +8.46] |
| 5 dB | +0.0002 [+0.0002, +0.0002] | +0.0002 [+0.0002, +0.0002] | +0.18 % [+0.16, +0.19] | +0.0096 [+0.0086, +0.0107] | +7.27 % [+6.76, +7.78] | +7.43 % [+6.92, +7.94] |
| 10 dB | +0.0000 [+0.0000, +0.0000] | +0.0000 [+0.0000, +0.0000] | +0.04 % [+0.03, +0.05] | +0.0046 [+0.0038, +0.0055] | +6.48 % [+5.71, +7.25] | +6.52 % [+5.75, +7.29] |
| 20 dB | +0.0000 [-0.0000, +0.0000] | +0.0000 [-0.0000, +0.0000] | +0.00 % [+0.00, +0.00] | +0.0006 [+0.0003, +0.0008] | +4.50 % [+3.08, +6.02] | +4.50 % [+3.09, +6.02] |

| SNR | gen_eq - load_joint CosSim [95% CI]; NMSE reduction 1 - gen_eq/load_joint [95% CI] | gen_k0 - load_joint CosSim [95% CI]; NMSE reduction 1 - gen_k0/load_joint [95% CI] | gen_pc - load_joint CosSim [95% CI]; NMSE reduction 1 - gen_pc/load_joint [95% CI] |
|---|---|---|---|
| -10 dB | -0.0171 [-0.0199, -0.0144]; -4.02 % [-4.67, -3.40] | +0.0007 [-0.0018, +0.0030]; +0.18 % [-0.29, +0.62] | +0.0014 [+0.0003, +0.0024]; +0.61 % [+0.41, +0.80] |
| -5 dB | -0.0489 [-0.0517, -0.0463]; -17.10 % [-18.18, -16.08] | +0.0010 [-0.0004, +0.0026]; +0.20 % [-0.19, +0.62] | -0.0052 [-0.0060, -0.0044]; -1.42 % [-1.63, -1.22] |
| 0 dB | -0.0883 [-0.0914, -0.0853]; -49.43 % [-51.72, -47.35] | -0.0007 [-0.0018, +0.0003]; -0.31 % [-0.79, +0.17] | -0.0057 [-0.0063, -0.0051]; -2.90 % [-3.14, -2.66] |
| 5 dB | -0.1192 [-0.1222, -0.1164]; -118.46 % [-123.80, -113.41] | +0.0001 [-0.0006, +0.0008]; +0.24 % [-0.33, +0.78] | -0.0042 [-0.0047, -0.0038]; -3.72 % [-4.02, -3.40] |
| 10 dB | -0.1463 [-0.1493, -0.1435]; -282.34 % [-296.80, -268.75] | -0.0001 [-0.0006, +0.0004]; -0.22 % [-0.94, +0.59] | -0.0019 [-0.0022, -0.0016]; -3.35 % [-3.79, -2.91] |
| 20 dB | -0.1729 [-0.1758, -0.1700]; -1812.82 % [-1968.63, -1648.82] | -0.0000 [-0.0002, +0.0001]; -0.15 % [-1.55, +1.09] | -0.0002 [-0.0003, -0.0002]; -2.24 % [-2.89, -1.57] |

| SNR | load masks: uses / coefficients sent | gen_joint: uses | gen_joint overlap index t | gen_joint t_overlap_modes | modes active / pruned | partial coefficients | n_data |
|---|---|---|---|---|---|---|---|
| -10 dB | 152 / 76 | 216 | 0.4533 | 0.4533 | 153 / 0 | 0 | 1536 |
| -5 dB | 358 / 179 | 446 | 0.2336 | 0.2336 | 371 / 0 | 0 | 1536 |
| 0 dB | 846 / 423 | 942 | 0.1099 | 0.1099 | 856 / 0 | 0 | 1536 |
| 5 dB | 1524 / 762 | 1528 | 0.0451 | 0.0451 | 1521 / 0 | 0 | 1536 |
| 10 dB | 1534 / 767 | 1534 | 0.0143 | 0.0143 | 1534 / 0 | 0 | 1536 |
| 20 dB | 1534 / 767 | 1534 | 0.0014 | 0.0014 | 1534 / 0 | 0 | 1536 |

#### COCO natural captions, K = 2, U = 3 (overload; gen_joint = pruned Proposition 2): data/shared/shared_coco_K2_U3.json

- dataset coco, K = 2, U = 3, 2000 frames, gains rayleigh, SNRs -10/-5/0/5/10/20 dB, receivers know the gains; load masks: spec rule (512/512/512 coefficients per user)
- content statistics of the design C (fit): kappa_bar = 0.3935 (tr C_model 0.3935, tr C of the content 0.3935, model is the true content: True); kappa_1 = 0.7416; lam-weighted kappa per 128-group 1..6: 0.434/0.151/0.107/0.089/0.075/0.066; mean kappa_i per group: 0.307/0.146/0.106/0.088/0.075/0.065; kappa_i of coefficients 1..10: 0.742/0.154/0.326/0.687/0.630/0.445/0.779/0.596/0.425/0.476

| mode | metric | -10 dB | -5 dB | 0 dB | 5 dB | 10 dB | 20 dB |
|---|---|---|---|---|---|---|---|
| load_pu | CosSim | 0.5747 | 0.6965 | 0.7992 | 0.8795 | 0.9363 | 0.9807 |
| load_pu | weakest user CosSim | 0.4160 | 0.5514 | 0.6792 | 0.7876 | 0.8797 | 0.9647 |
| load_pu | NMSE | 0.6459 | 0.4919 | 0.3434 | 0.2139 | 0.1174 | 0.0369 |
| load_joint | CosSim | 0.6356 | 0.7421 | 0.8280 | 0.8957 | 0.9432 | 0.9819 |
| load_joint | weakest user CosSim | 0.5394 | 0.6497 | 0.7466 | 0.8282 | 0.8980 | 0.9679 |
| load_joint | NMSE | 0.5868 | 0.4408 | 0.3075 | 0.1925 | 0.1078 | 0.0354 |
| gen_joint | CosSim | 0.6581 | 0.7504 | 0.8301 | 0.8962 | 0.9434 | 0.9820 |
| gen_joint | weakest user CosSim | 0.5645 | 0.6597 | 0.7492 | 0.8288 | 0.8980 | 0.9676 |
| gen_joint | NMSE | 0.5583 | 0.4289 | 0.3041 | 0.1916 | 0.1073 | 0.0352 |
| gen_eq | CosSim | 0.6249 | 0.6809 | 0.7177 | 0.7393 | 0.7504 | 0.7547 |
| gen_eq | weakest user CosSim | 0.5214 | 0.5711 | 0.6025 | 0.6190 | 0.6278 | 0.6303 |
| gen_eq | NMSE | 0.6004 | 0.5262 | 0.4734 | 0.4406 | 0.4238 | 0.4166 |
| gen_k0 | CosSim | 0.6344 | 0.7418 | 0.8282 | 0.8956 | 0.9432 | 0.9819 |
| gen_k0 | weakest user CosSim | 0.5369 | 0.6492 | 0.7458 | 0.8281 | 0.8979 | 0.9682 |
| gen_k0 | NMSE | 0.5882 | 0.4411 | 0.3074 | 0.1925 | 0.1077 | 0.0353 |
| gen_pc | CosSim | 0.6458 | 0.7387 | 0.8213 | 0.8905 | 0.9410 | 0.9816 |
| gen_pc | weakest user CosSim | 0.5427 | 0.6383 | 0.7311 | 0.8160 | 0.8918 | 0.9666 |
| gen_pc | NMSE | 0.5728 | 0.4443 | 0.3165 | 0.2000 | 0.1112 | 0.0358 |

| SNR | gen_joint - load_joint CosSim [95% CI] | gen_joint - load_joint weakest [95% CI] | overlap NMSE reduction 1 - gen_joint/load_joint [95% CI] | load_joint - load_pu CosSim [95% CI] | receiver NMSE reduction 1 - load_joint/load_pu [95% CI] | total NMSE reduction 1 - gen_joint/load_pu [95% CI] |
|---|---|---|---|---|---|---|
| -10 dB | +0.0225 [+0.0216, +0.0234] | +0.0251 [+0.0238, +0.0263] | +4.85 % [+4.67, +5.03] | +0.0610 [+0.0586, +0.0633] | +9.15 % [+8.87, +9.44] | +13.56 % [+13.19, +13.91] |
| -5 dB | +0.0083 [+0.0080, +0.0086] | +0.0100 [+0.0095, +0.0104] | +2.70 % [+2.61, +2.79] | +0.0456 [+0.0435, +0.0476] | +10.39 % [+10.09, +10.71] | +12.80 % [+12.47, +13.15] |
| 0 dB | +0.0022 [+0.0021, +0.0022] | +0.0026 [+0.0024, +0.0028] | +1.12 % [+1.08, +1.17] | +0.0287 [+0.0270, +0.0304] | +10.44 % [+10.04, +10.82] | +11.45 % [+11.06, +11.83] |
| 5 dB | +0.0005 [+0.0005, +0.0005] | +0.0006 [+0.0005, +0.0006] | +0.45 % [+0.43, +0.48] | +0.0162 [+0.0150, +0.0175] | +10.00 % [+9.56, +10.44] | +10.41 % [+9.97, +10.85] |
| 10 dB | +0.0002 [+0.0002, +0.0002] | +0.0000 [-0.0000, +0.0000] | +0.38 % [+0.35, +0.40] | +0.0068 [+0.0061, +0.0078] | +8.23 % [+7.66, +8.87] | +8.57 % [+8.01, +9.23] |
| 20 dB | +0.0001 [+0.0001, +0.0001] | -0.0003 [-0.0003, -0.0002] | +0.64 % [+0.56, +0.71] | +0.0011 [+0.0008, +0.0015] | +4.17 % [+3.20, +5.13] | +4.78 % [+3.83, +5.72] |

| SNR | gen_eq - load_joint CosSim [95% CI]; NMSE reduction 1 - gen_eq/load_joint [95% CI] | gen_k0 - load_joint CosSim [95% CI]; NMSE reduction 1 - gen_k0/load_joint [95% CI] | gen_pc - load_joint CosSim [95% CI]; NMSE reduction 1 - gen_pc/load_joint [95% CI] |
|---|---|---|---|
| -10 dB | -0.0107 [-0.0135, -0.0080]; -2.32 % [-2.90, -1.74] | -0.0013 [-0.0036, +0.0008]; -0.23 % [-0.66, +0.13] | +0.0102 [+0.0091, +0.0114]; +2.38 % [+2.16, +2.61] |
| -5 dB | -0.0611 [-0.0639, -0.0583]; -19.37 % [-20.30, -18.46] | -0.0003 [-0.0016, +0.0011]; -0.07 % [-0.45, +0.30] | -0.0034 [-0.0041, -0.0027]; -0.80 % [-0.98, -0.62] |
| 0 dB | -0.1103 [-0.1131, -0.1075]; -53.94 % [-55.80, -52.21] | +0.0002 [-0.0006, +0.0010]; +0.03 % [-0.35, +0.39] | -0.0066 [-0.0071, -0.0061]; -2.93 % [-3.13, -2.73] |
| 5 dB | -0.1563 [-0.1593, -0.1532]; -128.87 % [-132.89, -124.71] | -0.0001 [-0.0006, +0.0005]; -0.01 % [-0.38, +0.40] | -0.0051 [-0.0056, -0.0047]; -3.89 % [-4.12, -3.66] |
| 10 dB | -0.1928 [-0.1959, -0.1896]; -293.31 % [-303.47, -283.07] | +0.0000 [-0.0003, +0.0003]; +0.07 % [-0.38, +0.50] | -0.0022 [-0.0025, -0.0020]; -3.17 % [-3.46, -2.91] |
| 20 dB | -0.2272 [-0.2303, -0.2240]; -1077.22 % [-1118.27, -1037.66] | +0.0001 [-0.0000, +0.0002]; +0.37 % [-0.17, +0.94] | -0.0003 [-0.0004, -0.0002]; -1.03 % [-1.39, -0.68] |

| SNR | load masks: uses / coefficients sent | gen_joint: uses | gen_joint overlap index t | gen_joint t_overlap_modes | modes active / pruned | partial coefficients | n_data |
|---|---|---|---|---|---|---|---|
| -10 dB | 171 / 57 | 171 | 0.4764 | 0.4764 | 171 / 0 | 0 | 1536 |
| -5 dB | 396 / 132 | 413 | 0.2548 | 0.2548 | 413 / 0 | 0 | 1536 |
| 0 dB | 939 / 313 | 970 | 0.1230 | 0.1230 | 970 / 0 | 0 | 1536 |
| 5 dB | 1536 / 512 | 1536 | 0.0567 | 0.0567 | 1536 / 496 | 96 | 1536 |
| 10 dB | 1536 / 512 | 1536 | 0.0263 | 0.0263 | 1536 / 765 | 63 | 1536 |
| 20 dB | 1536 / 512 | 1536 | 0.0142 | 0.0142 | 1536 / 765 | 60 | 1536 |

#### COCO natural captions, K = 2, U = 4 (overload; gen_joint = pruned Proposition 2): data/shared/shared_coco_K2_U4.json

- dataset coco, K = 2, U = 4, 2000 frames, gains rayleigh, SNRs -10/-5/0/5/10/20 dB, receivers know the gains; load masks: spec rule (384/384/384/384 coefficients per user)
- content statistics of the design C (fit): kappa_bar = 0.3935 (tr C_model 0.3935, tr C of the content 0.3935, model is the true content: True); kappa_1 = 0.7416; lam-weighted kappa per 128-group 1..6: 0.434/0.151/0.107/0.089/0.075/0.066; mean kappa_i per group: 0.307/0.146/0.106/0.088/0.075/0.065; kappa_i of coefficients 1..10: 0.742/0.154/0.326/0.687/0.630/0.445/0.779/0.596/0.425/0.476

| mode | metric | -10 dB | -5 dB | 0 dB | 5 dB | 10 dB | 20 dB |
|---|---|---|---|---|---|---|---|
| load_pu | CosSim | 0.5412 | 0.6679 | 0.7758 | 0.8603 | 0.9224 | 0.9720 |
| load_pu | weakest user CosSim | 0.3300 | 0.4765 | 0.6107 | 0.7246 | 0.8397 | 0.9491 |
| load_pu | NMSE | 0.6818 | 0.5299 | 0.3788 | 0.2455 | 0.1426 | 0.0543 |
| load_joint | CosSim | 0.6192 | 0.7259 | 0.8119 | 0.8812 | 0.9312 | 0.9731 |
| load_joint | weakest user CosSim | 0.4979 | 0.6125 | 0.7045 | 0.7862 | 0.8683 | 0.9529 |
| load_joint | NMSE | 0.6088 | 0.4658 | 0.3337 | 0.2177 | 0.1301 | 0.0526 |
| gen_joint | CosSim | 0.6511 | 0.7384 | 0.8154 | 0.8821 | 0.9317 | 0.9734 |
| gen_joint | weakest user CosSim | 0.5337 | 0.6270 | 0.7088 | 0.7872 | 0.8684 | 0.9525 |
| gen_joint | NMSE | 0.5693 | 0.4480 | 0.3282 | 0.2161 | 0.1291 | 0.0520 |
| gen_eq | CosSim | 0.6176 | 0.6664 | 0.6935 | 0.7125 | 0.7198 | 0.7231 |
| gen_eq | weakest user CosSim | 0.4901 | 0.5348 | 0.5558 | 0.5700 | 0.5811 | 0.5804 |
| gen_eq | NMSE | 0.6114 | 0.5476 | 0.5096 | 0.4816 | 0.4712 | 0.4663 |
| gen_k0 | CosSim | 0.6194 | 0.7255 | 0.8116 | 0.8813 | 0.9309 | 0.9731 |
| gen_k0 | weakest user CosSim | 0.4975 | 0.6117 | 0.7036 | 0.7864 | 0.8672 | 0.9530 |
| gen_k0 | NMSE | 0.6085 | 0.4663 | 0.3339 | 0.2175 | 0.1304 | 0.0526 |
| gen_pc | CosSim | 0.6376 | 0.7254 | 0.8054 | 0.8755 | 0.9286 | 0.9730 |
| gen_pc | weakest user CosSim | 0.5113 | 0.6037 | 0.6882 | 0.7700 | 0.8595 | 0.9511 |
| gen_pc | NMSE | 0.5852 | 0.4652 | 0.3425 | 0.2261 | 0.1340 | 0.0528 |

| SNR | gen_joint - load_joint CosSim [95% CI] | gen_joint - load_joint weakest [95% CI] | overlap NMSE reduction 1 - gen_joint/load_joint [95% CI] | load_joint - load_pu CosSim [95% CI] | receiver NMSE reduction 1 - load_joint/load_pu [95% CI] | total NMSE reduction 1 - gen_joint/load_pu [95% CI] |
|---|---|---|---|---|---|---|
| -10 dB | +0.0319 [+0.0309, +0.0330] | +0.0359 [+0.0343, +0.0375] | +6.49 % [+6.29, +6.70] | +0.0780 [+0.0757, +0.0805] | +10.70 % [+10.43, +10.96] | +16.50 % [+16.14, +16.82] |
| -5 dB | +0.0125 [+0.0121, +0.0129] | +0.0145 [+0.0138, +0.0151] | +3.81 % [+3.69, +3.91] | +0.0580 [+0.0561, +0.0601] | +12.10 % [+11.82, +12.41] | +15.44 % [+15.14, +15.79] |
| 0 dB | +0.0035 [+0.0034, +0.0036] | +0.0043 [+0.0041, +0.0046] | +1.65 % [+1.59, +1.71] | +0.0360 [+0.0345, +0.0375] | +11.89 % [+11.54, +12.21] | +13.35 % [+12.98, +13.69] |
| 5 dB | +0.0009 [+0.0009, +0.0010] | +0.0010 [+0.0009, +0.0012] | +0.75 % [+0.71, +0.78] | +0.0208 [+0.0197, +0.0221] | +11.29 % [+10.90, +11.71] | +11.95 % [+11.55, +12.37] |
| 10 dB | +0.0005 [+0.0005, +0.0005] | +0.0001 [+0.0000, +0.0002] | +0.73 % [+0.69, +0.78] | +0.0088 [+0.0080, +0.0095] | +8.78 % [+8.32, +9.27] | +9.45 % [+8.99, +9.94] |
| 20 dB | +0.0003 [+0.0003, +0.0003] | -0.0004 [-0.0005, -0.0004] | +1.08 % [+0.99, +1.16] | +0.0011 [+0.0009, +0.0013] | +3.09 % [+2.69, +3.55] | +4.14 % [+3.72, +4.58] |

| SNR | gen_eq - load_joint CosSim [95% CI]; NMSE reduction 1 - gen_eq/load_joint [95% CI] | gen_k0 - load_joint CosSim [95% CI]; NMSE reduction 1 - gen_k0/load_joint [95% CI] | gen_pc - load_joint CosSim [95% CI]; NMSE reduction 1 - gen_pc/load_joint [95% CI] |
|---|---|---|---|
| -10 dB | -0.0016 [-0.0042, +0.0009]; -0.41 % [-0.92, +0.07] | +0.0003 [-0.0014, +0.0020]; +0.06 % [-0.24, +0.38] | +0.0184 [+0.0171, +0.0197]; +3.89 % [+3.65, +4.12] |
| -5 dB | -0.0595 [-0.0620, -0.0569]; -17.56 % [-18.36, -16.77] | -0.0004 [-0.0016, +0.0006]; -0.12 % [-0.43, +0.16] | -0.0005 [-0.0012, +0.0001]; +0.12 % [-0.06, +0.29] |
| 0 dB | -0.1183 [-0.1212, -0.1157]; -52.70 % [-54.16, -51.29] | -0.0002 [-0.0010, +0.0005]; -0.05 % [-0.34, +0.23] | -0.0065 [-0.0069, -0.0060]; -2.62 % [-2.79, -2.44] |
| 5 dB | -0.1687 [-0.1716, -0.1659]; -121.16 % [-124.32, -118.16] | +0.0001 [-0.0004, +0.0005]; +0.11 % [-0.20, +0.40] | -0.0057 [-0.0060, -0.0054]; -3.82 % [-4.01, -3.64] |
| 10 dB | -0.2114 [-0.2143, -0.2084]; -262.31 % [-269.12, -255.66] | -0.0003 [-0.0005, +0.0000]; -0.27 % [-0.57, +0.04] | -0.0026 [-0.0028, -0.0023]; -2.99 % [-3.21, -2.78] |
| 20 dB | -0.2500 [-0.2530, -0.2472]; -786.45 % [-804.34, -768.54] | +0.0000 [-0.0001, +0.0001]; +0.05 % [-0.26, +0.36] | -0.0001 [-0.0002, -0.0001]; -0.36 % [-0.58, -0.15] |

| SNR | load masks: uses / coefficients sent | gen_joint: uses | gen_joint overlap index t | gen_joint t_overlap_modes | modes active / pruned | partial coefficients | n_data |
|---|---|---|---|---|---|---|---|
| -10 dB | 188 / 47 | 188 | 0.4906 | 0.4906 | 188 / 0 | 0 | 1536 |
| -5 dB | 428 / 107 | 456 | 0.2666 | 0.2666 | 456 / 0 | 0 | 1536 |
| 0 dB | 1008 / 252 | 1051 | 0.1303 | 0.1303 | 1051 / 0 | 0 | 1536 |
| 5 dB | 1536 / 384 | 1536 | 0.0619 | 0.0618 | 1536 / 839 | 115 | 1536 |
| 10 dB | 1536 / 384 | 1536 | 0.0328 | 0.0326 | 1536 / 1532 | 91 | 1536 |
| 20 dB | 1536 / 384 | 1536 | 0.0192 | 0.0190 | 1536 / 1532 | 79 | 1536 |

#### COCO natural captions, K = 2, U = 5 (overload; gen_joint = pruned Proposition 2): data/shared/shared_coco_K2_U5.json

- dataset coco, K = 2, U = 5, 2000 frames, gains rayleigh, SNRs -10/-5/0/5/10/20 dB, receivers know the gains; load masks: spec rule (307/307/307/307/307 coefficients per user)
- content statistics of the design C (fit): kappa_bar = 0.3935 (tr C_model 0.3935, tr C of the content 0.3935, model is the true content: True); kappa_1 = 0.7416; lam-weighted kappa per 128-group 1..6: 0.434/0.151/0.107/0.089/0.075/0.066; mean kappa_i per group: 0.307/0.146/0.106/0.088/0.075/0.065; kappa_i of coefficients 1..10: 0.742/0.154/0.326/0.687/0.630/0.445/0.779/0.596/0.425/0.476

| mode | metric | -10 dB | -5 dB | 0 dB | 5 dB | 10 dB | 20 dB |
|---|---|---|---|---|---|---|---|
| load_pu | CosSim | 0.5203 | 0.6434 | 0.7577 | 0.8431 | 0.9111 | 0.9635 |
| load_pu | weakest user CosSim | 0.2769 | 0.4144 | 0.5576 | 0.6693 | 0.8024 | 0.9320 |
| load_pu | NMSE | 0.7055 | 0.5623 | 0.4059 | 0.2729 | 0.1629 | 0.0707 |
| load_joint | CosSim | 0.6075 | 0.7115 | 0.8021 | 0.8692 | 0.9214 | 0.9650 |
| load_joint | weakest user CosSim | 0.4693 | 0.5789 | 0.6815 | 0.7555 | 0.8413 | 0.9382 |
| load_joint | NMSE | 0.6241 | 0.4870 | 0.3505 | 0.2387 | 0.1482 | 0.0684 |
| gen_joint | CosSim | 0.6482 | 0.7281 | 0.8069 | 0.8707 | 0.9222 | 0.9655 |
| gen_joint | weakest user CosSim | 0.5133 | 0.5982 | 0.6873 | 0.7572 | 0.8416 | 0.9379 |
| gen_joint | NMSE | 0.5739 | 0.4637 | 0.3429 | 0.2363 | 0.1466 | 0.0674 |
| gen_eq | CosSim | 0.6153 | 0.6557 | 0.6814 | 0.6943 | 0.7008 | 0.7035 |
| gen_eq | weakest user CosSim | 0.4697 | 0.5074 | 0.5299 | 0.5359 | 0.5431 | 0.5433 |
| gen_eq | NMSE | 0.6150 | 0.5630 | 0.5280 | 0.5092 | 0.4999 | 0.4960 |
| gen_k0 | CosSim | 0.6086 | 0.7109 | 0.8019 | 0.8691 | 0.9216 | 0.9649 |
| gen_k0 | weakest user CosSim | 0.4704 | 0.5789 | 0.6816 | 0.7569 | 0.8429 | 0.9380 |
| gen_k0 | NMSE | 0.6229 | 0.4879 | 0.3508 | 0.2389 | 0.1479 | 0.0684 |
| gen_pc | CosSim | 0.6352 | 0.7144 | 0.7962 | 0.8638 | 0.9189 | 0.9648 |
| gen_pc | weakest user CosSim | 0.4906 | 0.5730 | 0.6652 | 0.7394 | 0.8310 | 0.9356 |
| gen_pc | NMSE | 0.5894 | 0.4817 | 0.3585 | 0.2467 | 0.1519 | 0.0685 |

| SNR | gen_joint - load_joint CosSim [95% CI] | gen_joint - load_joint weakest [95% CI] | overlap NMSE reduction 1 - gen_joint/load_joint [95% CI] | load_joint - load_pu CosSim [95% CI] | receiver NMSE reduction 1 - load_joint/load_pu [95% CI] | total NMSE reduction 1 - gen_joint/load_pu [95% CI] |
|---|---|---|---|---|---|---|
| -10 dB | +0.0407 [+0.0395, +0.0419] | +0.0440 [+0.0420, +0.0460] | +8.05 % [+7.82, +8.28] | +0.0872 [+0.0849, +0.0896] | +11.53 % [+11.27, +11.78] | +18.66 % [+18.30, +19.01] |
| -5 dB | +0.0166 [+0.0161, +0.0170] | +0.0193 [+0.0184, +0.0201] | +4.78 % [+4.64, +4.92] | +0.0681 [+0.0662, +0.0700] | +13.39 % [+13.12, +13.67] | +17.53 % [+17.21, +17.86] |
| 0 dB | +0.0049 [+0.0047, +0.0050] | +0.0058 [+0.0055, +0.0061] | +2.17 % [+2.10, +2.24] | +0.0444 [+0.0428, +0.0460] | +13.63 % [+13.31, +13.94] | +15.51 % [+15.16, +15.84] |
| 5 dB | +0.0014 [+0.0014, +0.0015] | +0.0017 [+0.0015, +0.0019] | +1.01 % [+0.97, +1.06] | +0.0261 [+0.0249, +0.0274] | +12.53 % [+12.15, +12.90] | +13.42 % [+13.03, +13.79] |
| 10 dB | +0.0008 [+0.0008, +0.0009] | +0.0003 [+0.0002, +0.0004] | +1.03 % [+0.98, +1.08] | +0.0102 [+0.0096, +0.0109] | +9.03 % [+8.64, +9.40] | +9.96 % [+9.58, +10.35] |
| 20 dB | +0.0005 [+0.0005, +0.0005] | -0.0003 [-0.0004, -0.0002] | +1.46 % [+1.37, +1.56] | +0.0015 [+0.0013, +0.0017] | +3.27 % [+2.92, +3.66] | +4.69 % [+4.32, +5.07] |

| SNR | gen_eq - load_joint CosSim [95% CI]; NMSE reduction 1 - gen_eq/load_joint [95% CI] | gen_k0 - load_joint CosSim [95% CI]; NMSE reduction 1 - gen_k0/load_joint [95% CI] | gen_pc - load_joint CosSim [95% CI]; NMSE reduction 1 - gen_pc/load_joint [95% CI] |
|---|---|---|---|
| -10 dB | +0.0078 [+0.0054, +0.0102]; +1.46 % [+1.01, +1.92] | +0.0011 [-0.0005, +0.0029]; +0.19 % [-0.10, +0.50] | +0.0277 [+0.0264, +0.0289]; +5.56 % [+5.31, +5.80] |
| -5 dB | -0.0557 [-0.0582, -0.0533]; -15.61 % [-16.32, -14.89] | -0.0006 [-0.0017, +0.0005]; -0.19 % [-0.47, +0.09] | +0.0030 [+0.0023, +0.0037]; +1.09 % [+0.90, +1.27] |
| 0 dB | -0.1207 [-0.1234, -0.1181]; -50.63 % [-51.93, -49.38] | -0.0002 [-0.0009, +0.0005]; -0.08 % [-0.35, +0.18] | -0.0059 [-0.0064, -0.0055]; -2.26 % [-2.43, -2.11] |
| 5 dB | -0.1750 [-0.1779, -0.1720]; -113.31 % [-116.04, -110.68] | -0.0002 [-0.0006, +0.0003]; -0.10 % [-0.38, +0.16] | -0.0054 [-0.0057, -0.0051]; -3.36 % [-3.52, -3.18] |
| 10 dB | -0.2206 [-0.2234, -0.2177]; -237.36 % [-242.16, -232.47] | +0.0002 [-0.0001, +0.0005]; +0.16 % [-0.12, +0.42] | -0.0025 [-0.0027, -0.0023]; -2.54 % [-2.72, -2.38] |
| 20 dB | -0.2615 [-0.2644, -0.2586]; -625.14 % [-635.72, -615.27] | -0.0000 [-0.0001, +0.0001]; +0.01 % [-0.20, +0.22] | -0.0001 [-0.0002, -0.0001]; -0.14 % [-0.32, +0.03] |

| SNR | load masks: uses / coefficients sent | gen_joint: uses | gen_joint overlap index t | gen_joint t_overlap_modes | modes active / pruned | partial coefficients | n_data |
|---|---|---|---|---|---|---|---|
| -10 dB | 205 / 41 | 207 | 0.5044 | 0.5044 | 207 / 0 | 0 | 1536 |
| -5 dB | 450 / 90 | 479 | 0.2746 | 0.2746 | 479 / 0 | 0 | 1536 |
| 0 dB | 1070 / 214 | 1124 | 0.1349 | 0.1349 | 1124 / 0 | 0 | 1536 |
| 5 dB | 1535 / 307 | 1536 | 0.0646 | 0.0645 | 1536 / 1089 | 125 | 1536 |
| 10 dB | 1535 / 307 | 1536 | 0.0350 | 0.0349 | 1536 / 2299 | 99 | 1536 |
| 20 dB | 1535 / 307 | 1536 | 0.0221 | 0.0219 | 1536 / 2299 | 90 | 1536 |

### B.2 Summary: overlap and receiver NMSE reductions of the natural captions (paired bootstrap 95% CI)

| K, U | quantity | -10 dB | -5 dB | 0 dB | 5 dB | 10 dB | 20 dB |
|---|---|---|---|---|---|---|---|
| K = 4, U = 2 | overlap 1 - gen_joint/load_joint | +1.87 % [+1.77, +1.97] | +0.85 % [+0.81, +0.90] | +0.31 % [+0.29, +0.33] | +0.07 % [+0.07, +0.08] | +0.01 % [+0.01, +0.02] | +0.00 % [-0.00, +0.00] |
| K = 4, U = 2 | receiver 1 - load_joint/load_pu | +7.02 % [+6.72, +7.37] | +7.41 % [+7.01, +7.78] | +7.44 % [+7.01, +7.95] | +6.71 % [+6.12, +7.28] | +5.77 % [+4.85, +6.79] | +3.72 % [+2.28, +5.16] |
| K = 4, U = 2 | CosSim gen_joint - load_joint | +0.0061 [+0.0058, +0.0064] | +0.0018 [+0.0017, +0.0019] | +0.0004 [+0.0003, +0.0004] | +0.0000 [+0.0000, +0.0001] | +0.0000 [+0.0000, +0.0000] | +0.0000 [-0.0000, +0.0000] |
| K = 4, U = 2 | CosSim load_joint - load_pu | +0.0335 [+0.0314, +0.0360] | +0.0220 [+0.0202, +0.0236] | +0.0129 [+0.0117, +0.0145] | +0.0058 [+0.0050, +0.0067] | +0.0026 [+0.0020, +0.0034] | +0.0003 [+0.0001, +0.0004] |
| K = 4, U = 3 | overlap 1 - gen_joint/load_joint | +3.54 % [+3.42, +3.66] | +1.66 % [+1.60, +1.72] | +0.62 % [+0.59, +0.65] | +0.18 % [+0.17, +0.19] | +0.04 % [+0.04, +0.04] | +0.00 % [-0.00, +0.00] |
| K = 4, U = 3 | receiver 1 - load_joint/load_pu | +9.98 % [+9.66, +10.28] | +10.57 % [+10.23, +10.93] | +10.26 % [+9.81, +10.69] | +9.65 % [+9.13, +10.15] | +8.09 % [+7.40, +8.88] | +6.45 % [+4.79, +8.03] |
| K = 4, U = 3 | CosSim gen_joint - load_joint | +0.0130 [+0.0125, +0.0134] | +0.0039 [+0.0037, +0.0040] | +0.0009 [+0.0008, +0.0009] | +0.0001 [+0.0001, +0.0002] | +0.0000 [+0.0000, +0.0000] | +0.0000 [-0.0000, +0.0000] |
| K = 4, U = 3 | CosSim load_joint - load_pu | +0.0520 [+0.0497, +0.0541] | +0.0354 [+0.0335, +0.0372] | +0.0206 [+0.0192, +0.0220] | +0.0107 [+0.0097, +0.0117] | +0.0041 [+0.0035, +0.0048] | +0.0006 [+0.0004, +0.0009] |
| K = 4, U = 4 | overlap 1 - gen_joint/load_joint | +4.86 % [+4.71, +5.01] | +2.38 % [+2.31, +2.46] | +0.93 % [+0.89, +0.97] | +0.29 % [+0.27, +0.30] | +0.06 % [+0.06, +0.07] | +0.00 % [+0.00, +0.00] |
| K = 4, U = 4 | receiver 1 - load_joint/load_pu | +11.76 % [+11.48, +12.03] | +12.28 % [+11.97, +12.61] | +11.52 % [+11.14, +11.87] | +10.92 % [+10.49, +11.38] | +9.19 % [+8.62, +9.80] | +5.59 % [+4.72, +6.56] |
| K = 4, U = 4 | CosSim gen_joint - load_joint | +0.0188 [+0.0183, +0.0195] | +0.0060 [+0.0058, +0.0062] | +0.0015 [+0.0014, +0.0015] | +0.0003 [+0.0002, +0.0003] | +0.0000 [+0.0000, +0.0000] | +0.0000 [+0.0000, +0.0000] |
| K = 4, U = 4 | CosSim load_joint - load_pu | +0.0677 [+0.0656, +0.0699] | +0.0454 [+0.0436, +0.0473] | +0.0258 [+0.0245, +0.0270] | +0.0140 [+0.0131, +0.0150] | +0.0057 [+0.0051, +0.0063] | +0.0006 [+0.0005, +0.0008] |
| K = 4, U = 5 (overload, pruned Prop. 2) | overlap 1 - gen_joint/load_joint | +6.07 % [+5.89, +6.24] | +3.06 % [+2.97, +3.15] | +1.25 % [+1.20, +1.30] | +0.45 % [+0.42, +0.47] | +0.38 % [+0.36, +0.40] | +0.96 % [+0.89, +1.03] |
| K = 4, U = 5 (overload, pruned Prop. 2) | receiver 1 - load_joint/load_pu | +12.66 % [+12.38, +12.93] | +13.65 % [+13.36, +13.96] | +13.19 % [+12.84, +13.51] | +12.18 % [+11.76, +12.60] | +9.33 % [+8.89, +9.77] | +4.74 % [+4.09, +5.48] |
| K = 4, U = 5 (overload, pruned Prop. 2) | CosSim gen_joint - load_joint | +0.0245 [+0.0238, +0.0252] | +0.0083 [+0.0080, +0.0085] | +0.0021 [+0.0020, +0.0022] | +0.0004 [+0.0004, +0.0005] | +0.0002 [+0.0002, +0.0002] | +0.0001 [+0.0001, +0.0001] |
| K = 4, U = 5 (overload, pruned Prop. 2) | CosSim load_joint - load_pu | +0.0758 [+0.0737, +0.0779] | +0.0539 [+0.0522, +0.0557] | +0.0321 [+0.0308, +0.0336] | +0.0181 [+0.0170, +0.0191] | +0.0067 [+0.0062, +0.0072] | +0.0008 [+0.0006, +0.0010] |
| K = 2, U = 2 | overlap 1 - gen_joint/load_joint | +2.65 % [+2.50, +2.78] | +1.45 % [+1.38, +1.52] | +0.57 % [+0.54, +0.60] | +0.18 % [+0.16, +0.19] | +0.04 % [+0.03, +0.05] | +0.00 % [+0.00, +0.00] |
| K = 2, U = 2 | receiver 1 - load_joint/load_pu | +6.40 % [+6.13, +6.71] | +7.15 % [+6.80, +7.49] | +7.49 % [+7.10, +7.93] | +7.27 % [+6.76, +7.78] | +6.48 % [+5.71, +7.25] | +4.50 % [+3.08, +6.02] |
| K = 2, U = 2 | CosSim gen_joint - load_joint | +0.0112 [+0.0106, +0.0118] | +0.0040 [+0.0038, +0.0042] | +0.0010 [+0.0009, +0.0010] | +0.0002 [+0.0002, +0.0002] | +0.0000 [+0.0000, +0.0000] | +0.0000 [-0.0000, +0.0000] |
| K = 2, U = 2 | CosSim load_joint - load_pu | +0.0395 [+0.0373, +0.0421] | +0.0281 [+0.0262, +0.0300] | +0.0181 [+0.0167, +0.0198] | +0.0096 [+0.0086, +0.0107] | +0.0046 [+0.0038, +0.0055] | +0.0006 [+0.0003, +0.0008] |
| K = 2, U = 3 (overload, pruned Prop. 2) | overlap 1 - gen_joint/load_joint | +4.85 % [+4.67, +5.03] | +2.70 % [+2.61, +2.79] | +1.12 % [+1.08, +1.17] | +0.45 % [+0.43, +0.48] | +0.38 % [+0.35, +0.40] | +0.64 % [+0.56, +0.71] |
| K = 2, U = 3 (overload, pruned Prop. 2) | receiver 1 - load_joint/load_pu | +9.15 % [+8.87, +9.44] | +10.39 % [+10.09, +10.71] | +10.44 % [+10.04, +10.82] | +10.00 % [+9.56, +10.44] | +8.23 % [+7.66, +8.87] | +4.17 % [+3.20, +5.13] |
| K = 2, U = 3 (overload, pruned Prop. 2) | CosSim gen_joint - load_joint | +0.0225 [+0.0216, +0.0234] | +0.0083 [+0.0080, +0.0086] | +0.0022 [+0.0021, +0.0022] | +0.0005 [+0.0005, +0.0005] | +0.0002 [+0.0002, +0.0002] | +0.0001 [+0.0001, +0.0001] |
| K = 2, U = 3 (overload, pruned Prop. 2) | CosSim load_joint - load_pu | +0.0610 [+0.0586, +0.0633] | +0.0456 [+0.0435, +0.0476] | +0.0287 [+0.0270, +0.0304] | +0.0162 [+0.0150, +0.0175] | +0.0068 [+0.0061, +0.0078] | +0.0011 [+0.0008, +0.0015] |
| K = 2, U = 4 (overload, pruned Prop. 2) | overlap 1 - gen_joint/load_joint | +6.49 % [+6.29, +6.70] | +3.81 % [+3.69, +3.91] | +1.65 % [+1.59, +1.71] | +0.75 % [+0.71, +0.78] | +0.73 % [+0.69, +0.78] | +1.08 % [+0.99, +1.16] |
| K = 2, U = 4 (overload, pruned Prop. 2) | receiver 1 - load_joint/load_pu | +10.70 % [+10.43, +10.96] | +12.10 % [+11.82, +12.41] | +11.89 % [+11.54, +12.21] | +11.29 % [+10.90, +11.71] | +8.78 % [+8.32, +9.27] | +3.09 % [+2.69, +3.55] |
| K = 2, U = 4 (overload, pruned Prop. 2) | CosSim gen_joint - load_joint | +0.0319 [+0.0309, +0.0330] | +0.0125 [+0.0121, +0.0129] | +0.0035 [+0.0034, +0.0036] | +0.0009 [+0.0009, +0.0010] | +0.0005 [+0.0005, +0.0005] | +0.0003 [+0.0003, +0.0003] |
| K = 2, U = 4 (overload, pruned Prop. 2) | CosSim load_joint - load_pu | +0.0780 [+0.0757, +0.0805] | +0.0580 [+0.0561, +0.0601] | +0.0360 [+0.0345, +0.0375] | +0.0208 [+0.0197, +0.0221] | +0.0088 [+0.0080, +0.0095] | +0.0011 [+0.0009, +0.0013] |
| K = 2, U = 5 (overload, pruned Prop. 2) | overlap 1 - gen_joint/load_joint | +8.05 % [+7.82, +8.28] | +4.78 % [+4.64, +4.92] | +2.17 % [+2.10, +2.24] | +1.01 % [+0.97, +1.06] | +1.03 % [+0.98, +1.08] | +1.46 % [+1.37, +1.56] |
| K = 2, U = 5 (overload, pruned Prop. 2) | receiver 1 - load_joint/load_pu | +11.53 % [+11.27, +11.78] | +13.39 % [+13.12, +13.67] | +13.63 % [+13.31, +13.94] | +12.53 % [+12.15, +12.90] | +9.03 % [+8.64, +9.40] | +3.27 % [+2.92, +3.66] |
| K = 2, U = 5 (overload, pruned Prop. 2) | CosSim gen_joint - load_joint | +0.0407 [+0.0395, +0.0419] | +0.0166 [+0.0161, +0.0170] | +0.0049 [+0.0047, +0.0050] | +0.0014 [+0.0014, +0.0015] | +0.0008 [+0.0008, +0.0009] | +0.0005 [+0.0005, +0.0005] |
| K = 2, U = 5 (overload, pruned Prop. 2) | CosSim load_joint - load_pu | +0.0872 [+0.0849, +0.0896] | +0.0681 [+0.0662, +0.0700] | +0.0444 [+0.0428, +0.0460] | +0.0261 [+0.0249, +0.0274] | +0.0102 [+0.0096, +0.0109] | +0.0015 [+0.0013, +0.0017] |

### B.3 Variants and controls (same point unless stated)


#### COCO captions of one image, K = 2, U = 4; overload, gen_joint = pruned Proposition 2; design and receiver C fitted on the first 100 fit images only (estimation noise): data/shared/shared_coco_K2_U4_cf100.json

- dataset coco, K = 2, U = 4, 2000 frames, gains rayleigh, SNRs -5/0/10 dB, receivers know the gains; load masks: spec rule (384/384/384/384 coefficients per user)
- content statistics of the design C (fit): kappa_bar = 0.1322 (tr C_model 0.1319, tr C of the content 0.3935, model is the true content: False); kappa_1 = 0.3283; lam-weighted kappa per 128-group 1..6: 0.149/0.034/0.015/0.012/0.007/0.005; mean kappa_i per group: 0.095/0.030/0.014/0.012/0.007/0.005; kappa_i of coefficients 1..10: 0.328/0.020/0.099/0.259/0.227/0.132/0.303/0.166/0.096/0.195

| mode | metric | -5 dB | 0 dB | 10 dB |
|---|---|---|---|---|
| load_pu | CosSim | 0.6679 | 0.7758 | 0.9224 |
| load_pu | weakest user CosSim | 0.4765 | 0.6107 | 0.8397 |
| load_pu | NMSE | 0.5299 | 0.3788 | 0.1426 |
| load_joint | CosSim | 0.6962 | 0.7857 | 0.9161 |
| load_joint | weakest user CosSim | 0.5561 | 0.6448 | 0.8285 |
| load_joint | NMSE | 0.5033 | 0.3698 | 0.1542 |
| gen_joint | CosSim | 0.7020 | 0.7870 | 0.9161 |
| gen_joint | weakest user CosSim | 0.5629 | 0.6463 | 0.8282 |
| gen_joint | NMSE | 0.4955 | 0.3677 | 0.1541 |
| gen_eq | CosSim | 0.6405 | 0.6466 | 0.5794 |
| gen_eq | weakest user CosSim | 0.4829 | 0.4490 | 0.2362 |
| gen_eq | NMSE | 0.5823 | 0.5678 | 0.6833 |
| gen_k0 | CosSim | 0.6961 | 0.7854 | 0.9159 |
| gen_k0 | weakest user CosSim | 0.5556 | 0.6438 | 0.8274 |
| gen_k0 | NMSE | 0.5038 | 0.3699 | 0.1543 |
| gen_pc | CosSim | 0.7053 | 0.7951 | 0.9263 |
| gen_pc | weakest user CosSim | 0.5696 | 0.6660 | 0.8532 |
| gen_pc | NMSE | 0.4926 | 0.3577 | 0.1378 |

| SNR | gen_joint - load_joint CosSim [95% CI] | gen_joint - load_joint weakest [95% CI] | overlap NMSE reduction 1 - gen_joint/load_joint [95% CI] | load_joint - load_pu CosSim [95% CI] | receiver NMSE reduction 1 - load_joint/load_pu [95% CI] | total NMSE reduction 1 - gen_joint/load_pu [95% CI] |
|---|---|---|---|---|---|---|
| -5 dB | +0.0058 [+0.0056, +0.0060] | +0.0069 [+0.0065, +0.0072] | +1.55 % [+1.51, +1.60] | +0.0284 [+0.0269, +0.0300] | +5.01 % [+4.83, +5.21] | +6.48 % [+6.30, +6.69] |
| 0 dB | +0.0013 [+0.0013, +0.0014] | +0.0015 [+0.0013, +0.0016] | +0.55 % [+0.53, +0.57] | +0.0099 [+0.0090, +0.0108] | +2.39 % [+2.19, +2.60] | +2.93 % [+2.73, +3.15] |
| 10 dB | +0.0000 [-0.0000, +0.0001] | -0.0004 [-0.0005, -0.0002] | +0.05 % [+0.02, +0.07] | -0.0063 [-0.0068, -0.0058] | -8.14 % [-8.51, -7.77] | -8.08 % [-8.47, -7.71] |

| SNR | gen_eq - load_joint CosSim [95% CI]; NMSE reduction 1 - gen_eq/load_joint [95% CI] | gen_k0 - load_joint CosSim [95% CI]; NMSE reduction 1 - gen_k0/load_joint [95% CI] | gen_pc - load_joint CosSim [95% CI]; NMSE reduction 1 - gen_pc/load_joint [95% CI] |
|---|---|---|---|
| -5 dB | -0.0557 [-0.0588, -0.0527]; -15.68 % [-16.45, -14.86] | -0.0001 [-0.0015, +0.0012]; -0.09 % [-0.39, +0.18] | +0.0091 [+0.0084, +0.0098]; +2.12 % [+1.98, +2.25] |
| 0 dB | -0.1391 [-0.1423, -0.1359]; -53.57 % [-55.05, -52.06] | -0.0003 [-0.0013, +0.0007]; -0.04 % [-0.33, +0.25] | +0.0094 [+0.0088, +0.0100]; +3.27 % [+3.11, +3.43] |
| 10 dB | -0.3367 [-0.3410, -0.3328]; -343.17 % [-353.00, -334.03] | -0.0002 [-0.0006, +0.0002]; -0.08 % [-0.42, +0.23] | +0.0102 [+0.0097, +0.0107]; +10.65 % [+10.36, +10.95] |

| SNR | load masks: uses / coefficients sent | gen_joint: uses | gen_joint overlap index t | gen_joint t_overlap_modes | modes active / pruned | partial coefficients | n_data |
|---|---|---|---|---|---|---|---|
| -5 dB | 428 / 107 | 421 | 0.0800 | 0.0800 | 421 / 0 | 0 | 1536 |
| 0 dB | 1008 / 252 | 1013 | 0.0364 | 0.0364 | 1013 / 0 | 0 | 1536 |
| 10 dB | 1536 / 384 | 1536 | 0.0064 | 0.0064 | 1536 / 1532 | 14 | 1536 |

#### COCO captions of one image, K = 2, U = 4; overload, gen_joint = pruned Proposition 2; design and receiver C fitted on the first 1000 fit images only (estimation noise): data/shared/shared_coco_K2_U4_cf1000.json

- dataset coco, K = 2, U = 4, 2000 frames, gains rayleigh, SNRs -5/0/10 dB, receivers know the gains; load masks: spec rule (384/384/384/384 coefficients per user)
- content statistics of the design C (fit): kappa_bar = 0.3608 (tr C_model 0.3608, tr C of the content 0.3935, model is the true content: False); kappa_1 = 0.6555; lam-weighted kappa per 128-group 1..6: 0.397/0.147/0.105/0.089/0.073/0.065; mean kappa_i per group: 0.291/0.141/0.103/0.088/0.073/0.063; kappa_i of coefficients 1..10: 0.656/0.147/0.284/0.618/0.552/0.399/0.686/0.513/0.382/0.441

| mode | metric | -5 dB | 0 dB | 10 dB |
|---|---|---|---|---|
| load_pu | CosSim | 0.6679 | 0.7758 | 0.9224 |
| load_pu | weakest user CosSim | 0.4765 | 0.6107 | 0.8397 |
| load_pu | NMSE | 0.5299 | 0.3788 | 0.1426 |
| load_joint | CosSim | 0.7215 | 0.8067 | 0.9268 |
| load_joint | weakest user CosSim | 0.6061 | 0.6951 | 0.8595 |
| load_joint | NMSE | 0.4716 | 0.3415 | 0.1376 |
| gen_joint | CosSim | 0.7331 | 0.8099 | 0.9273 |
| gen_joint | weakest user CosSim | 0.6192 | 0.6991 | 0.8591 |
| gen_joint | NMSE | 0.4553 | 0.3364 | 0.1367 |
| gen_eq | CosSim | 0.6621 | 0.6866 | 0.7043 |
| gen_eq | weakest user CosSim | 0.5275 | 0.5436 | 0.5508 |
| gen_eq | NMSE | 0.5528 | 0.5183 | 0.4922 |
| gen_k0 | CosSim | 0.7211 | 0.8064 | 0.9266 |
| gen_k0 | weakest user CosSim | 0.6052 | 0.6945 | 0.8585 |
| gen_k0 | NMSE | 0.4723 | 0.3417 | 0.1380 |
| gen_pc | CosSim | 0.7244 | 0.8052 | 0.9286 |
| gen_pc | weakest user CosSim | 0.6026 | 0.6880 | 0.8594 |
| gen_pc | NMSE | 0.4666 | 0.3430 | 0.1341 |

| SNR | gen_joint - load_joint CosSim [95% CI] | gen_joint - load_joint weakest [95% CI] | overlap NMSE reduction 1 - gen_joint/load_joint [95% CI] | load_joint - load_pu CosSim [95% CI] | receiver NMSE reduction 1 - load_joint/load_pu [95% CI] | total NMSE reduction 1 - gen_joint/load_pu [95% CI] |
|---|---|---|---|---|---|---|
| -5 dB | +0.0116 [+0.0112, +0.0119] | +0.0131 [+0.0125, +0.0138] | +3.47 % [+3.36, +3.57] | +0.0537 [+0.0518, +0.0557] | +10.99 % [+10.72, +11.28] | +14.08 % [+13.77, +14.40] |
| 0 dB | +0.0033 [+0.0031, +0.0034] | +0.0039 [+0.0036, +0.0042] | +1.50 % [+1.44, +1.55] | +0.0309 [+0.0294, +0.0322] | +9.85 % [+9.51, +10.15] | +11.20 % [+10.85, +11.52] |
| 10 dB | +0.0004 [+0.0004, +0.0005] | -0.0004 [-0.0006, -0.0003] | +0.62 % [+0.57, +0.68] | +0.0044 [+0.0038, +0.0051] | +3.49 % [+3.01, +3.98] | +4.09 % [+3.61, +4.57] |

| SNR | gen_eq - load_joint CosSim [95% CI]; NMSE reduction 1 - gen_eq/load_joint [95% CI] | gen_k0 - load_joint CosSim [95% CI]; NMSE reduction 1 - gen_k0/load_joint [95% CI] | gen_pc - load_joint CosSim [95% CI]; NMSE reduction 1 - gen_pc/load_joint [95% CI] |
|---|---|---|---|
| -5 dB | -0.0594 [-0.0621, -0.0568]; -17.21 % [-17.99, -16.40] | -0.0004 [-0.0016, +0.0006]; -0.13 % [-0.43, +0.14] | +0.0029 [+0.0023, +0.0035]; +1.06 % [+0.90, +1.22] |
| 0 dB | -0.1200 [-0.1229, -0.1172]; -51.77 % [-53.19, -50.38] | -0.0003 [-0.0010, +0.0005]; -0.07 % [-0.36, +0.20] | -0.0015 [-0.0019, -0.0010]; -0.43 % [-0.58, -0.26] |
| 10 dB | -0.2225 [-0.2257, -0.2195]; -257.71 % [-264.09, -251.22] | -0.0003 [-0.0006, +0.0000]; -0.27 % [-0.59, +0.03] | +0.0017 [+0.0015, +0.0020]; +2.59 % [+2.35, +2.83] |

| SNR | load masks: uses / coefficients sent | gen_joint: uses | gen_joint overlap index t | gen_joint t_overlap_modes | modes active / pruned | partial coefficients | n_data |
|---|---|---|---|---|---|---|---|
| -5 dB | 428 / 107 | 444 | 0.2430 | 0.2430 | 444 / 0 | 0 | 1536 |
| 0 dB | 1008 / 252 | 1051 | 0.1222 | 0.1222 | 1051 / 0 | 0 | 1536 |
| 10 dB | 1536 / 384 | 1536 | 0.0314 | 0.0314 | 1536 / 1532 | 88 | 1536 |

#### COCO captions of one image, K = 2, U = 4; overload, gen_joint = pruned Proposition 2; design and receiver C fitted on the other caption corpus, in this corpus' basis (mismatch): data/shared/shared_coco_K2_U4_cfother.json

- dataset coco, K = 2, U = 4, 2000 frames, gains rayleigh, SNRs -5/0/10 dB, receivers know the gains; load masks: spec rule (384/384/384/384 coefficients per user)
- content statistics of the design C (fit): kappa_bar = 0.1985 (tr C_model 0.1985, tr C of the content 0.3935, model is the true content: False); kappa_1 = 0.0707; lam-weighted kappa per 128-group 1..6: 0.200/0.209/0.179/0.165/0.156/0.155; mean kappa_i per group: 0.237/0.206/0.178/0.164/0.156/0.154; kappa_i of coefficients 1..10: 0.071/0.035/0.140/0.172/0.289/0.187/0.171/0.247/0.245/0.254

| mode | metric | -5 dB | 0 dB | 10 dB |
|---|---|---|---|---|
| load_pu | CosSim | 0.6679 | 0.7758 | 0.9224 |
| load_pu | weakest user CosSim | 0.4765 | 0.6107 | 0.8397 |
| load_pu | NMSE | 0.5299 | 0.3788 | 0.1426 |
| load_joint | CosSim | 0.6976 | 0.7916 | 0.9234 |
| load_joint | weakest user CosSim | 0.5540 | 0.6580 | 0.8483 |
| load_joint | NMSE | 0.5012 | 0.3618 | 0.1426 |
| gen_joint | CosSim | 0.7080 | 0.7950 | 0.9241 |
| gen_joint | weakest user CosSim | 0.5668 | 0.6630 | 0.8482 |
| gen_joint | NMSE | 0.4871 | 0.3566 | 0.1412 |
| gen_eq | CosSim | 0.6490 | 0.6761 | 0.7017 |
| gen_eq | weakest user CosSim | 0.5032 | 0.5221 | 0.5436 |
| gen_eq | NMSE | 0.5745 | 0.5377 | 0.5031 |
| gen_k0 | CosSim | 0.6973 | 0.7909 | 0.9233 |
| gen_k0 | weakest user CosSim | 0.5540 | 0.6555 | 0.8477 |
| gen_k0 | NMSE | 0.5018 | 0.3625 | 0.1428 |
| gen_pc | CosSim | 0.7110 | 0.7988 | 0.9274 |
| gen_pc | weakest user CosSim | 0.5742 | 0.6718 | 0.8549 |
| gen_pc | NMSE | 0.4840 | 0.3518 | 0.1358 |

| SNR | gen_joint - load_joint CosSim [95% CI] | gen_joint - load_joint weakest [95% CI] | overlap NMSE reduction 1 - gen_joint/load_joint [95% CI] | load_joint - load_pu CosSim [95% CI] | receiver NMSE reduction 1 - load_joint/load_pu [95% CI] | total NMSE reduction 1 - gen_joint/load_pu [95% CI] |
|---|---|---|---|---|---|---|
| -5 dB | +0.0104 [+0.0100, +0.0107] | +0.0128 [+0.0121, +0.0135] | +2.81 % [+2.71, +2.90] | +0.0298 [+0.0283, +0.0311] | +5.42 % [+5.24, +5.59] | +8.07 % [+7.87, +8.28] |
| 0 dB | +0.0035 [+0.0033, +0.0036] | +0.0050 [+0.0047, +0.0054] | +1.44 % [+1.38, +1.50] | +0.0158 [+0.0148, +0.0167] | +4.48 % [+4.28, +4.67] | +5.86 % [+5.63, +6.06] |
| 10 dB | +0.0007 [+0.0006, +0.0007] | -0.0002 [-0.0004, +0.0000] | +0.94 % [+0.87, +1.00] | +0.0010 [+0.0006, +0.0014] | +0.00 % [-0.30, +0.30] | +0.94 % [+0.63, +1.24] |

| SNR | gen_eq - load_joint CosSim [95% CI]; NMSE reduction 1 - gen_eq/load_joint [95% CI] | gen_k0 - load_joint CosSim [95% CI]; NMSE reduction 1 - gen_k0/load_joint [95% CI] | gen_pc - load_joint CosSim [95% CI]; NMSE reduction 1 - gen_pc/load_joint [95% CI] |
|---|---|---|---|
| -5 dB | -0.0486 [-0.0515, -0.0456]; -14.64 % [-15.38, -13.84] | -0.0003 [-0.0017, +0.0009]; -0.14 % [-0.43, +0.13] | +0.0134 [+0.0127, +0.0140]; +3.43 % [+3.29, +3.57] |
| 0 dB | -0.1155 [-0.1184, -0.1126]; -48.60 % [-50.00, -47.22] | -0.0007 [-0.0017, +0.0003]; -0.18 % [-0.49, +0.13] | +0.0072 [+0.0067, +0.0077]; +2.78 % [+2.64, +2.90] |
| 10 dB | -0.2217 [-0.2246, -0.2186]; -252.85 % [-259.49, -246.41] | -0.0001 [-0.0005, +0.0003]; -0.12 % [-0.47, +0.21] | +0.0040 [+0.0038, +0.0042]; +4.78 % [+4.61, +4.95] |

| SNR | load masks: uses / coefficients sent | gen_joint: uses | gen_joint overlap index t | gen_joint t_overlap_modes | modes active / pruned | partial coefficients | n_data |
|---|---|---|---|---|---|---|---|
| -5 dB | 428 / 107 | 438 | 0.1886 | 0.1886 | 438 / 0 | 0 | 1536 |
| 0 dB | 1008 / 252 | 1012 | 0.1271 | 0.1271 | 1012 / 0 | 0 | 1536 |
| 10 dB | 1536 / 384 | 1536 | 0.0454 | 0.0453 | 1536 / 1532 | 154 | 1536 |

#### COCO captions of one image, K = 2, U = 4; overload, gen_joint = pruned Proposition 2; control: captions of U distinct images (true C = 0), design and receiver use C = 0: data/shared/shared_coco_K2_U4_ctrl.json

- dataset coco, K = 2, U = 4, 2000 frames, gains rayleigh, SNRs -10/-5/0/5/10/20 dB, receivers know the gains; load masks: spec rule (384/384/384/384 coefficients per user)
- content statistics of the design C (fit): kappa_bar = 0.0000 (tr C_model 0.0000, tr C of the content 0.3935, model is the true content: True); kappa_1 = 0.0000; lam-weighted kappa per 128-group 1..6: 0.000/0.000/0.000/0.000/0.000/0.000; mean kappa_i per group: 0.000/0.000/0.000/0.000/0.000/0.000; kappa_i of coefficients 1..10: 0.000/0.000/0.000/0.000/0.000/0.000/0.000/0.000/0.000/0.000

| mode | metric | -10 dB | -5 dB | 0 dB | 5 dB | 10 dB | 20 dB |
|---|---|---|---|---|---|---|---|
| load_pu | CosSim | 0.5434 | 0.6690 | 0.7766 | 0.8608 | 0.9221 | 0.9719 |
| load_pu | weakest user CosSim | 0.3288 | 0.4762 | 0.6134 | 0.7263 | 0.8381 | 0.9479 |
| load_pu | NMSE | 0.6802 | 0.5288 | 0.3782 | 0.2454 | 0.1427 | 0.0544 |
| load_joint | CosSim | 0.5434 | 0.6690 | 0.7766 | 0.8608 | 0.9221 | 0.9719 |
| load_joint | weakest user CosSim | 0.3288 | 0.4762 | 0.6134 | 0.7263 | 0.8381 | 0.9479 |
| load_joint | NMSE | 0.6802 | 0.5288 | 0.3782 | 0.2454 | 0.1427 | 0.0544 |
| gen_joint | CosSim | 0.5434 | 0.6690 | 0.7766 | 0.8608 | 0.9221 | 0.9719 |
| gen_joint | weakest user CosSim | 0.3288 | 0.4762 | 0.6134 | 0.7263 | 0.8381 | 0.9479 |
| gen_joint | NMSE | 0.6802 | 0.5288 | 0.3782 | 0.2454 | 0.1427 | 0.0544 |
| gen_eq | CosSim | 0.3430 | 0.3890 | 0.4190 | 0.4396 | 0.4510 | 0.4530 |
| gen_eq | weakest user CosSim | 0.0970 | 0.1406 | 0.1622 | 0.1743 | 0.1889 | 0.1833 |
| gen_eq | NMSE | 0.8560 | 0.8166 | 0.7878 | 0.7650 | 0.7556 | 0.7516 |
| gen_k0 | CosSim | 0.5430 | 0.6689 | 0.7763 | 0.8607 | 0.9223 | 0.9721 |
| gen_k0 | weakest user CosSim | 0.3289 | 0.4759 | 0.6108 | 0.7263 | 0.8383 | 0.9486 |
| gen_k0 | NMSE | 0.6801 | 0.5294 | 0.3784 | 0.2455 | 0.1426 | 0.0542 |
| gen_pc | CosSim | 0.5434 | 0.6690 | 0.7766 | 0.8608 | 0.9221 | 0.9719 |
| gen_pc | weakest user CosSim | 0.3288 | 0.4762 | 0.6134 | 0.7263 | 0.8381 | 0.9479 |
| gen_pc | NMSE | 0.6802 | 0.5288 | 0.3782 | 0.2454 | 0.1427 | 0.0544 |

| SNR | gen_joint - load_joint CosSim [95% CI] | gen_joint - load_joint weakest [95% CI] | overlap NMSE reduction 1 - gen_joint/load_joint [95% CI] | load_joint - load_pu CosSim [95% CI] | receiver NMSE reduction 1 - load_joint/load_pu [95% CI] | total NMSE reduction 1 - gen_joint/load_pu [95% CI] |
|---|---|---|---|---|---|---|
| -10 dB | +0.0000 [+0.0000, +0.0000] | +0.0000 [+0.0000, +0.0000] | +0.00 % [+0.00, +0.00] | +0.0000 [+0.0000, +0.0000] | +0.00 % [+0.00, +0.00] | +0.00 % [+0.00, +0.00] |
| -5 dB | +0.0000 [+0.0000, +0.0000] | +0.0000 [+0.0000, +0.0000] | +0.00 % [+0.00, +0.00] | +0.0000 [+0.0000, +0.0000] | +0.00 % [+0.00, +0.00] | +0.00 % [+0.00, +0.00] |
| 0 dB | +0.0000 [+0.0000, +0.0000] | +0.0000 [+0.0000, +0.0000] | +0.00 % [+0.00, +0.00] | +0.0000 [+0.0000, +0.0000] | +0.00 % [+0.00, +0.00] | +0.00 % [+0.00, +0.00] |
| 5 dB | +0.0000 [+0.0000, +0.0000] | +0.0000 [+0.0000, +0.0000] | +0.00 % [+0.00, +0.00] | +0.0000 [+0.0000, +0.0000] | +0.00 % [+0.00, +0.00] | +0.00 % [+0.00, +0.00] |
| 10 dB | +0.0000 [+0.0000, +0.0000] | +0.0000 [+0.0000, +0.0000] | +0.00 % [+0.00, +0.00] | +0.0000 [+0.0000, +0.0000] | +0.00 % [+0.00, +0.00] | +0.00 % [+0.00, +0.00] |
| 20 dB | +0.0000 [+0.0000, +0.0000] | +0.0000 [+0.0000, +0.0000] | +0.00 % [+0.00, +0.00] | +0.0000 [+0.0000, +0.0000] | +0.00 % [+0.00, +0.00] | +0.00 % [+0.00, +0.00] |

| SNR | gen_eq - load_joint CosSim [95% CI]; NMSE reduction 1 - gen_eq/load_joint [95% CI] | gen_k0 - load_joint CosSim [95% CI]; NMSE reduction 1 - gen_k0/load_joint [95% CI] | gen_pc - load_joint CosSim [95% CI]; NMSE reduction 1 - gen_pc/load_joint [95% CI] |
|---|---|---|---|
| -10 dB | -0.2004 [-0.2044, -0.1964]; -25.83 % [-26.48, -25.17] | -0.0004 [-0.0031, +0.0020]; +0.01 % [-0.28, +0.28] | +0.0000 [+0.0000, +0.0000]; +0.00 % [+0.00, +0.00] |
| -5 dB | -0.2799 [-0.2834, -0.2760]; -54.42 % [-55.53, -53.17] | -0.0001 [-0.0018, +0.0017]; -0.11 % [-0.38, +0.18] | +0.0000 [+0.0000, +0.0000]; +0.00 % [+0.00, +0.00] |
| 0 dB | -0.3576 [-0.3609, -0.3541]; -108.30 % [-110.79, -106.07] | -0.0003 [-0.0015, +0.0009]; -0.06 % [-0.38, +0.24] | +0.0000 [+0.0000, +0.0000]; +0.00 % [+0.00, +0.00] |
| 5 dB | -0.4212 [-0.4242, -0.4181]; -211.73 % [-216.32, -206.71] | -0.0001 [-0.0009, +0.0007]; -0.04 % [-0.38, +0.29] | +0.0000 [+0.0000, +0.0000]; +0.00 % [+0.00, +0.00] |
| 10 dB | -0.4712 [-0.4739, -0.4685]; -429.46 % [-441.11, -418.20] | +0.0002 [-0.0003, +0.0007]; +0.10 % [-0.28, +0.48] | +0.0000 [+0.0000, +0.0000]; +0.00 % [+0.00, +0.00] |
| 20 dB | -0.5189 [-0.5215, -0.5164]; -1281.90 % [-1314.27, -1250.66] | +0.0002 [+0.0000, +0.0004]; +0.35 % [+0.01, +0.73] | +0.0000 [+0.0000, +0.0000]; +0.00 % [+0.00, +0.00] |

| SNR | load masks: uses / coefficients sent | gen_joint: uses | gen_joint overlap index t | gen_joint t_overlap_modes | modes active / pruned | partial coefficients | n_data |
|---|---|---|---|---|---|---|---|
| -10 dB | 188 / 47 | 188 | -0.0000 | 0.0000 | 188 / 0 | 0 | 1536 |
| -5 dB | 428 / 107 | 428 | -0.0000 | 0.0000 | 428 / 0 | 0 | 1536 |
| 0 dB | 1008 / 252 | 1008 | -0.0000 | 0.0000 | 1008 / 0 | 0 | 1536 |
| 5 dB | 1536 / 384 | 1536 | -0.0000 | 0.0000 | 1536 / 824 | 0 | 1536 |
| 10 dB | 1536 / 384 | 1536 | -0.0000 | 0.0000 | 1536 / 1532 | 0 | 1536 |
| 20 dB | 1536 / 384 | 1536 | -0.0000 | 0.0000 | 1536 / 1532 | 0 | 1536 |

#### COCO captions of one image, K = 2, U = 4; overload, gen_joint = pruned Proposition 2; control: captions of U distinct images (true C = 0), design and receiver use the fitted C (mismatch): data/shared/shared_coco_K2_U4_ctrlfit.json

- dataset coco, K = 2, U = 4, 2000 frames, gains rayleigh, SNRs -10/-5/0/5/10/20 dB, receivers know the gains; load masks: spec rule (384/384/384/384 coefficients per user)
- content statistics of the design C (fit): kappa_bar = 0.3935 (tr C_model 0.3935, tr C of the content 0.3935, model is the true content: False); kappa_1 = 0.7416; lam-weighted kappa per 128-group 1..6: 0.434/0.151/0.107/0.089/0.075/0.066; mean kappa_i per group: 0.307/0.146/0.106/0.088/0.075/0.065; kappa_i of coefficients 1..10: 0.742/0.154/0.326/0.687/0.630/0.445/0.779/0.596/0.425/0.476

| mode | metric | -10 dB | -5 dB | 0 dB | 5 dB | 10 dB | 20 dB |
|---|---|---|---|---|---|---|---|
| load_pu | CosSim | 0.5434 | 0.6690 | 0.7766 | 0.8608 | 0.9221 | 0.9719 |
| load_pu | weakest user CosSim | 0.3288 | 0.4762 | 0.6134 | 0.7263 | 0.8381 | 0.9479 |
| load_pu | NMSE | 0.6802 | 0.5288 | 0.3782 | 0.2454 | 0.1427 | 0.0544 |
| load_joint | CosSim | 0.4661 | 0.6008 | 0.7246 | 0.8284 | 0.9063 | 0.9693 |
| load_joint | weakest user CosSim | 0.2051 | 0.3502 | 0.5002 | 0.6460 | 0.7931 | 0.9401 |
| load_joint | NMSE | 0.7594 | 0.6089 | 0.4458 | 0.2917 | 0.1677 | 0.0591 |
| gen_joint | CosSim | 0.4485 | 0.5937 | 0.7223 | 0.8279 | 0.9062 | 0.9691 |
| gen_joint | weakest user CosSim | 0.1999 | 0.3489 | 0.4990 | 0.6454 | 0.7924 | 0.9393 |
| gen_joint | NMSE | 0.7820 | 0.6196 | 0.4497 | 0.2928 | 0.1679 | 0.0594 |
| gen_eq | CosSim | 0.3369 | 0.3833 | 0.4135 | 0.4347 | 0.4447 | 0.4474 |
| gen_eq | weakest user CosSim | 0.0948 | 0.1336 | 0.1514 | 0.1632 | 0.1756 | 0.1704 |
| gen_eq | NMSE | 0.8834 | 0.8413 | 0.8109 | 0.7871 | 0.7775 | 0.7736 |
| gen_k0 | CosSim | 0.4659 | 0.6002 | 0.7251 | 0.8280 | 0.9064 | 0.9694 |
| gen_k0 | weakest user CosSim | 0.2045 | 0.3504 | 0.5004 | 0.6452 | 0.7930 | 0.9405 |
| gen_k0 | NMSE | 0.7596 | 0.6098 | 0.4452 | 0.2922 | 0.1677 | 0.0590 |
| gen_pc | CosSim | 0.4667 | 0.6194 | 0.7480 | 0.8466 | 0.9165 | 0.9712 |
| gen_pc | weakest user CosSim | 0.2204 | 0.3832 | 0.5427 | 0.6850 | 0.8188 | 0.9448 |
| gen_pc | NMSE | 0.7620 | 0.5878 | 0.4141 | 0.2641 | 0.1507 | 0.0556 |

| SNR | gen_joint - load_joint CosSim [95% CI] | gen_joint - load_joint weakest [95% CI] | overlap NMSE reduction 1 - gen_joint/load_joint [95% CI] | load_joint - load_pu CosSim [95% CI] | receiver NMSE reduction 1 - load_joint/load_pu [95% CI] | total NMSE reduction 1 - gen_joint/load_pu [95% CI] |
|---|---|---|---|---|---|---|
| -10 dB | -0.0176 [-0.0187, -0.0166] | -0.0052 [-0.0075, -0.0027] | -2.98 % [-3.13, -2.83] | -0.0773 [-0.0798, -0.0747] | -11.63 % [-11.95, -11.30] | -14.96 % [-15.35, -14.57] |
| -5 dB | -0.0071 [-0.0076, -0.0067] | -0.0013 [-0.0023, -0.0004] | -1.76 % [-1.85, -1.67] | -0.0682 [-0.0701, -0.0659] | -15.15 % [-15.48, -14.74] | -17.18 % [-17.54, -16.77] |
| 0 dB | -0.0023 [-0.0024, -0.0021] | -0.0012 [-0.0016, -0.0009] | -0.89 % [-0.93, -0.84] | -0.0520 [-0.0536, -0.0503] | -17.87 % [-18.24, -17.46] | -18.91 % [-19.30, -18.50] |
| 5 dB | -0.0006 [-0.0006, -0.0005] | -0.0007 [-0.0008, -0.0004] | -0.38 % [-0.40, -0.35] | -0.0324 [-0.0337, -0.0312] | -18.86 % [-19.32, -18.46] | -19.31 % [-19.77, -18.90] |
| 10 dB | -0.0001 [-0.0002, -0.0001] | -0.0007 [-0.0008, -0.0006] | -0.12 % [-0.15, -0.10] | -0.0158 [-0.0167, -0.0150] | -17.51 % [-18.05, -16.96] | -17.65 % [-18.21, -17.10] |
| 20 dB | -0.0001 [-0.0002, -0.0001] | -0.0008 [-0.0008, -0.0007] | -0.42 % [-0.48, -0.36] | -0.0026 [-0.0029, -0.0023] | -8.69 % [-9.23, -8.14] | -9.15 % [-9.67, -8.61] |

| SNR | gen_eq - load_joint CosSim [95% CI]; NMSE reduction 1 - gen_eq/load_joint [95% CI] | gen_k0 - load_joint CosSim [95% CI]; NMSE reduction 1 - gen_k0/load_joint [95% CI] | gen_pc - load_joint CosSim [95% CI]; NMSE reduction 1 - gen_pc/load_joint [95% CI] |
|---|---|---|---|
| -10 dB | -0.1292 [-0.1323, -0.1258]; -16.33 % [-16.82, -15.84] | -0.0003 [-0.0025, +0.0022]; -0.03 % [-0.30, +0.25] | +0.0006 [-0.0008, +0.0020]; -0.34 % [-0.52, -0.16] |
| -5 dB | -0.2175 [-0.2210, -0.2138]; -38.17 % [-39.08, -37.14] | -0.0006 [-0.0022, +0.0010]; -0.14 % [-0.41, +0.12] | +0.0186 [+0.0175, +0.0196]; +3.47 % [+3.28, +3.64] |
| 0 dB | -0.3111 [-0.3150, -0.3073]; -81.91 % [-83.95, -80.05] | +0.0005 [-0.0007, +0.0016]; +0.14 % [-0.14, +0.41] | +0.0233 [+0.0224, +0.0242]; +7.11 % [+6.90, +7.32] |
| 5 dB | -0.3938 [-0.3971, -0.3902]; -169.87 % [-173.85, -165.50] | -0.0004 [-0.0012, +0.0005]; -0.18 % [-0.49, +0.14] | +0.0182 [+0.0176, +0.0188]; +9.46 % [+9.23, +9.70] |
| 10 dB | -0.4616 [-0.4646, -0.4584]; -363.65 % [-374.04, -353.89] | +0.0000 [-0.0004, +0.0006]; +0.02 % [-0.31, +0.39] | +0.0101 [+0.0096, +0.0107]; +10.14 % [+9.83, +10.44] |
| 20 dB | -0.5219 [-0.5247, -0.5193]; -1208.64 % [-1242.58, -1175.12] | +0.0001 [-0.0000, +0.0003]; +0.18 % [-0.16, +0.51] | +0.0019 [+0.0017, +0.0021]; +6.02 % [+5.70, +6.35] |

| SNR | load masks: uses / coefficients sent | gen_joint: uses | gen_joint overlap index t | gen_joint t_overlap_modes | modes active / pruned | partial coefficients | n_data |
|---|---|---|---|---|---|---|---|
| -10 dB | 188 / 47 | 188 | 0.4906 | 0.4906 | 188 / 0 | 0 | 1536 |
| -5 dB | 428 / 107 | 456 | 0.2666 | 0.2666 | 456 / 0 | 0 | 1536 |
| 0 dB | 1008 / 252 | 1051 | 0.1303 | 0.1303 | 1051 / 0 | 0 | 1536 |
| 5 dB | 1536 / 384 | 1536 | 0.0619 | 0.0618 | 1536 / 839 | 115 | 1536 |
| 10 dB | 1536 / 384 | 1536 | 0.0328 | 0.0326 | 1536 / 1532 | 91 | 1536 |
| 20 dB | 1536 / 384 | 1536 | 0.0192 | 0.0190 | 1536 / 1532 | 79 | 1536 |

#### COCO captions of one image, K = 2, U = 4; overload, gen_joint = pruned Proposition 2; residual phase N(0, (10 deg)^2), receivers use |h| (phase not known): data/shared/shared_coco_K2_U4_ph10abs.json

- dataset coco, K = 2, U = 4, 2000 frames, gains rayleigh, SNRs -5/0/10 dB, receivers use |h|: the effective gain g cos(phi) is NOT known; load masks: spec rule (384/384/384/384 coefficients per user)
- content statistics of the design C (fit): kappa_bar = 0.3935 (tr C_model 0.3935, tr C of the content 0.3935, model is the true content: True); kappa_1 = 0.7416; lam-weighted kappa per 128-group 1..6: 0.434/0.151/0.107/0.089/0.075/0.066; mean kappa_i per group: 0.307/0.146/0.106/0.088/0.075/0.065; kappa_i of coefficients 1..10: 0.742/0.154/0.326/0.687/0.630/0.445/0.779/0.596/0.425/0.476

| mode | metric | -5 dB | 0 dB | 10 dB |
|---|---|---|---|---|
| load_pu | CosSim | 0.6644 | 0.7727 | 0.9211 |
| load_pu | weakest user CosSim | 0.4719 | 0.6062 | 0.8370 |
| load_pu | NMSE | 0.5344 | 0.3834 | 0.1453 |
| load_joint | CosSim | 0.7233 | 0.8096 | 0.9301 |
| load_joint | weakest user CosSim | 0.6095 | 0.7017 | 0.8664 |
| load_joint | NMSE | 0.4696 | 0.3375 | 0.1324 |
| gen_joint | CosSim | 0.7359 | 0.8131 | 0.9306 |
| gen_joint | weakest user CosSim | 0.6242 | 0.7061 | 0.8665 |
| gen_joint | NMSE | 0.4517 | 0.3319 | 0.1315 |
| gen_eq | CosSim | 0.6653 | 0.6928 | 0.7197 |
| gen_eq | weakest user CosSim | 0.5339 | 0.5553 | 0.5812 |
| gen_eq | NMSE | 0.5491 | 0.5107 | 0.4717 |
| gen_k0 | CosSim | 0.7229 | 0.8093 | 0.9298 |
| gen_k0 | weakest user CosSim | 0.6088 | 0.7008 | 0.8653 |
| gen_k0 | NMSE | 0.4700 | 0.3377 | 0.1328 |
| gen_pc | CosSim | 0.7228 | 0.8030 | 0.9274 |
| gen_pc | weakest user CosSim | 0.6008 | 0.6852 | 0.8574 |
| gen_pc | NMSE | 0.4689 | 0.3464 | 0.1365 |

| SNR | gen_joint - load_joint CosSim [95% CI] | gen_joint - load_joint weakest [95% CI] | overlap NMSE reduction 1 - gen_joint/load_joint [95% CI] | load_joint - load_pu CosSim [95% CI] | receiver NMSE reduction 1 - load_joint/load_pu [95% CI] | total NMSE reduction 1 - gen_joint/load_pu [95% CI] |
|---|---|---|---|---|---|---|
| -5 dB | +0.0126 [+0.0122, +0.0130] | +0.0148 [+0.0141, +0.0154] | +3.81 % [+3.70, +3.92] | +0.0589 [+0.0570, +0.0610] | +12.13 % [+11.86, +12.43] | +15.48 % [+15.17, +15.82] |
| 0 dB | +0.0036 [+0.0034, +0.0037] | +0.0044 [+0.0042, +0.0047] | +1.65 % [+1.59, +1.71] | +0.0368 [+0.0353, +0.0383] | +11.96 % [+11.61, +12.28] | +13.42 % [+13.06, +13.76] |
| 10 dB | +0.0005 [+0.0005, +0.0005] | +0.0001 [+0.0000, +0.0002] | +0.72 % [+0.68, +0.76] | +0.0090 [+0.0083, +0.0098] | +8.85 % [+8.40, +9.33] | +9.51 % [+9.06, +10.00] |

| SNR | gen_eq - load_joint CosSim [95% CI]; NMSE reduction 1 - gen_eq/load_joint [95% CI] | gen_k0 - load_joint CosSim [95% CI]; NMSE reduction 1 - gen_k0/load_joint [95% CI] | gen_pc - load_joint CosSim [95% CI]; NMSE reduction 1 - gen_pc/load_joint [95% CI] |
|---|---|---|---|
| -5 dB | -0.0580 [-0.0605, -0.0554]; -16.95 % [-17.73, -16.15] | -0.0004 [-0.0016, +0.0006]; -0.10 % [-0.42, +0.18] | -0.0005 [-0.0012, +0.0001]; +0.13 % [-0.04, +0.30] |
| 0 dB | -0.1167 [-0.1196, -0.1141]; -51.32 % [-52.77, -49.94] | -0.0002 [-0.0010, +0.0005]; -0.05 % [-0.34, +0.23] | -0.0066 [-0.0070, -0.0061]; -2.63 % [-2.80, -2.46] |
| 10 dB | -0.2104 [-0.2134, -0.2074]; -256.16 % [-262.81, -249.46] | -0.0003 [-0.0005, +0.0000]; -0.25 % [-0.56, +0.06] | -0.0026 [-0.0029, -0.0024]; -3.03 % [-3.25, -2.82] |

| SNR | load masks: uses / coefficients sent | gen_joint: uses | gen_joint overlap index t | gen_joint t_overlap_modes | modes active / pruned | partial coefficients | n_data |
|---|---|---|---|---|---|---|---|
| -5 dB | 428 / 107 | 456 | 0.2666 | 0.2666 | 456 / 0 | 0 | 1536 |
| 0 dB | 1008 / 252 | 1051 | 0.1303 | 0.1303 | 1051 / 0 | 0 | 1536 |
| 10 dB | 1536 / 384 | 1536 | 0.0328 | 0.0326 | 1536 / 1532 | 91 | 1536 |

#### COCO captions of one image, K = 2, U = 4; overload, gen_joint = pruned Proposition 2; residual phase N(0, (10 deg)^2), receivers use the effective gain g cos(phi): data/shared/shared_coco_K2_U4_ph10eff.json

- dataset coco, K = 2, U = 4, 2000 frames, gains rayleigh, SNRs -5/0/10 dB, receivers use the effective gain g cos(phi) (effective gains known); load masks: spec rule (384/384/384/384 coefficients per user)
- content statistics of the design C (fit): kappa_bar = 0.3935 (tr C_model 0.3935, tr C of the content 0.3935, model is the true content: True); kappa_1 = 0.7416; lam-weighted kappa per 128-group 1..6: 0.434/0.151/0.107/0.089/0.075/0.066; mean kappa_i per group: 0.307/0.146/0.106/0.088/0.075/0.065; kappa_i of coefficients 1..10: 0.742/0.154/0.326/0.687/0.630/0.445/0.779/0.596/0.425/0.476

| mode | metric | -5 dB | 0 dB | 10 dB |
|---|---|---|---|---|
| load_pu | CosSim | 0.6644 | 0.7728 | 0.9211 |
| load_pu | weakest user CosSim | 0.4719 | 0.6062 | 0.8370 |
| load_pu | NMSE | 0.5343 | 0.3832 | 0.1448 |
| load_joint | CosSim | 0.7233 | 0.8096 | 0.9301 |
| load_joint | weakest user CosSim | 0.6095 | 0.7016 | 0.8664 |
| load_joint | NMSE | 0.4694 | 0.3373 | 0.1320 |
| gen_joint | CosSim | 0.7360 | 0.8132 | 0.9306 |
| gen_joint | weakest user CosSim | 0.6242 | 0.7060 | 0.8665 |
| gen_joint | NMSE | 0.4515 | 0.3318 | 0.1310 |
| gen_eq | CosSim | 0.6653 | 0.6928 | 0.7197 |
| gen_eq | weakest user CosSim | 0.5339 | 0.5553 | 0.5811 |
| gen_eq | NMSE | 0.5490 | 0.5106 | 0.4715 |
| gen_k0 | CosSim | 0.7229 | 0.8094 | 0.9298 |
| gen_k0 | weakest user CosSim | 0.6088 | 0.7007 | 0.8653 |
| gen_k0 | NMSE | 0.4699 | 0.3375 | 0.1323 |
| gen_pc | CosSim | 0.7228 | 0.8030 | 0.9275 |
| gen_pc | weakest user CosSim | 0.6008 | 0.6851 | 0.8574 |
| gen_pc | NMSE | 0.4688 | 0.3462 | 0.1360 |

| SNR | gen_joint - load_joint CosSim [95% CI] | gen_joint - load_joint weakest [95% CI] | overlap NMSE reduction 1 - gen_joint/load_joint [95% CI] | load_joint - load_pu CosSim [95% CI] | receiver NMSE reduction 1 - load_joint/load_pu [95% CI] | total NMSE reduction 1 - gen_joint/load_pu [95% CI] |
|---|---|---|---|---|---|---|
| -5 dB | +0.0126 [+0.0122, +0.0130] | +0.0147 [+0.0141, +0.0153] | +3.81 % [+3.70, +3.92] | +0.0589 [+0.0570, +0.0610] | +12.14 % [+11.87, +12.45] | +15.49 % [+15.18, +15.83] |
| 0 dB | +0.0036 [+0.0034, +0.0037] | +0.0044 [+0.0042, +0.0047] | +1.66 % [+1.59, +1.71] | +0.0368 [+0.0353, +0.0383] | +11.97 % [+11.62, +12.29] | +13.43 % [+13.06, +13.77] |
| 10 dB | +0.0005 [+0.0005, +0.0005] | +0.0001 [+0.0000, +0.0002] | +0.73 % [+0.69, +0.77] | +0.0090 [+0.0083, +0.0098] | +8.87 % [+8.42, +9.35] | +9.53 % [+9.08, +10.02] |

| SNR | gen_eq - load_joint CosSim [95% CI]; NMSE reduction 1 - gen_eq/load_joint [95% CI] | gen_k0 - load_joint CosSim [95% CI]; NMSE reduction 1 - gen_k0/load_joint [95% CI] | gen_pc - load_joint CosSim [95% CI]; NMSE reduction 1 - gen_pc/load_joint [95% CI] |
|---|---|---|---|
| -5 dB | -0.0580 [-0.0605, -0.0554]; -16.96 % [-17.74, -16.16] | -0.0004 [-0.0016, +0.0006]; -0.11 % [-0.42, +0.17] | -0.0005 [-0.0012, +0.0001]; +0.13 % [-0.04, +0.30] |
| 0 dB | -0.1167 [-0.1196, -0.1142]; -51.35 % [-52.80, -49.97] | -0.0002 [-0.0010, +0.0005]; -0.05 % [-0.34, +0.24] | -0.0066 [-0.0070, -0.0061]; -2.63 % [-2.80, -2.45] |
| 10 dB | -0.2104 [-0.2134, -0.2074]; -257.18 % [-263.87, -250.45] | -0.0003 [-0.0005, +0.0000]; -0.26 % [-0.56, +0.05] | -0.0026 [-0.0029, -0.0024]; -3.04 % [-3.25, -2.82] |

| SNR | load masks: uses / coefficients sent | gen_joint: uses | gen_joint overlap index t | gen_joint t_overlap_modes | modes active / pruned | partial coefficients | n_data |
|---|---|---|---|---|---|---|---|
| -5 dB | 428 / 107 | 456 | 0.2666 | 0.2666 | 456 / 0 | 0 | 1536 |
| 0 dB | 1008 / 252 | 1051 | 0.1303 | 0.1303 | 1051 / 0 | 0 | 1536 |
| 10 dB | 1536 / 384 | 1536 | 0.0328 | 0.0326 | 1536 / 1532 | 91 | 1536 |

#### COCO captions of one image, K = 2, U = 4; overload, gen_joint = pruned Proposition 2; residual phase N(0, (30 deg)^2), receivers use |h| (phase not known): data/shared/shared_coco_K2_U4_ph30abs.json

- dataset coco, K = 2, U = 4, 2000 frames, gains rayleigh, SNRs -5/0/10 dB, receivers use |h|: the effective gain g cos(phi) is NOT known; load masks: spec rule (384/384/384/384 coefficients per user)
- content statistics of the design C (fit): kappa_bar = 0.3935 (tr C_model 0.3935, tr C of the content 0.3935, model is the true content: True); kappa_1 = 0.7416; lam-weighted kappa per 128-group 1..6: 0.434/0.151/0.107/0.089/0.075/0.066; mean kappa_i per group: 0.307/0.146/0.106/0.088/0.075/0.065; kappa_i of coefficients 1..10: 0.742/0.154/0.326/0.687/0.630/0.445/0.779/0.596/0.425/0.476

| mode | metric | -5 dB | 0 dB | 10 dB |
|---|---|---|---|---|
| load_pu | CosSim | 0.6250 | 0.7342 | 0.8977 |
| load_pu | weakest user CosSim | 0.4056 | 0.5336 | 0.7768 |
| load_pu | NMSE | 0.5810 | 0.4372 | 0.1973 |
| load_joint | CosSim | 0.6935 | 0.7804 | 0.9108 |
| load_joint | weakest user CosSim | 0.5642 | 0.6521 | 0.8188 |
| load_joint | NMSE | 0.5110 | 0.3845 | 0.1804 |
| gen_joint | CosSim | 0.7079 | 0.7846 | 0.9112 |
| gen_joint | weakest user CosSim | 0.5827 | 0.6582 | 0.8190 |
| gen_joint | NMSE | 0.4921 | 0.3784 | 0.1795 |
| gen_eq | CosSim | 0.6524 | 0.6834 | 0.7150 |
| gen_eq | weakest user CosSim | 0.5174 | 0.5430 | 0.5714 |
| gen_eq | NMSE | 0.5713 | 0.5314 | 0.4897 |
| gen_k0 | CosSim | 0.6935 | 0.7800 | 0.9106 |
| gen_k0 | weakest user CosSim | 0.5650 | 0.6505 | 0.8179 |
| gen_k0 | NMSE | 0.5110 | 0.3847 | 0.1807 |
| gen_pc | CosSim | 0.6930 | 0.7720 | 0.9064 |
| gen_pc | weakest user CosSim | 0.5555 | 0.6307 | 0.8048 |
| gen_pc | NMSE | 0.5101 | 0.3948 | 0.1862 |

| SNR | gen_joint - load_joint CosSim [95% CI] | gen_joint - load_joint weakest [95% CI] | overlap NMSE reduction 1 - gen_joint/load_joint [95% CI] | load_joint - load_pu CosSim [95% CI] | receiver NMSE reduction 1 - load_joint/load_pu [95% CI] | total NMSE reduction 1 - gen_joint/load_pu [95% CI] |
|---|---|---|---|---|---|---|
| -5 dB | +0.0144 [+0.0140, +0.0149] | +0.0185 [+0.0177, +0.0193] | +3.70 % [+3.60, +3.80] | +0.0685 [+0.0665, +0.0706] | +12.05 % [+11.79, +12.32] | +15.31 % [+15.03, +15.62] |
| 0 dB | +0.0043 [+0.0041, +0.0044] | +0.0061 [+0.0058, +0.0064] | +1.59 % [+1.54, +1.64] | +0.0462 [+0.0445, +0.0479] | +12.05 % [+11.75, +12.35] | +13.45 % [+13.13, +13.76] |
| 10 dB | +0.0004 [+0.0004, +0.0005] | +0.0001 [+0.0000, +0.0002] | +0.54 % [+0.52, +0.57] | +0.0131 [+0.0121, +0.0140] | +8.53 % [+8.17, +8.94] | +9.03 % [+8.66, +9.44] |

| SNR | gen_eq - load_joint CosSim [95% CI]; NMSE reduction 1 - gen_eq/load_joint [95% CI] | gen_k0 - load_joint CosSim [95% CI]; NMSE reduction 1 - gen_k0/load_joint [95% CI] | gen_pc - load_joint CosSim [95% CI]; NMSE reduction 1 - gen_pc/load_joint [95% CI] |
|---|---|---|---|
| -5 dB | -0.0411 [-0.0439, -0.0381]; -11.80 % [-12.49, -11.05] | +0.0000 [-0.0014, +0.0015]; -0.01 % [-0.34, +0.33] | -0.0004 [-0.0012, +0.0003]; +0.17 % [+0.01, +0.32] |
| 0 dB | -0.0970 [-0.1001, -0.0940]; -38.21 % [-39.59, -37.00] | -0.0004 [-0.0013, +0.0006]; -0.04 % [-0.38, +0.28] | -0.0084 [-0.0089, -0.0078]; -2.67 % [-2.82, -2.51] |
| 10 dB | -0.1957 [-0.1992, -0.1922]; -171.41 % [-176.60, -166.46] | -0.0001 [-0.0005, +0.0003]; -0.16 % [-0.47, +0.16] | -0.0043 [-0.0047, -0.0040]; -3.19 % [-3.37, -3.01] |

| SNR | load masks: uses / coefficients sent | gen_joint: uses | gen_joint overlap index t | gen_joint t_overlap_modes | modes active / pruned | partial coefficients | n_data |
|---|---|---|---|---|---|---|---|
| -5 dB | 428 / 107 | 456 | 0.2666 | 0.2666 | 456 / 0 | 0 | 1536 |
| 0 dB | 1008 / 252 | 1051 | 0.1303 | 0.1303 | 1051 / 0 | 0 | 1536 |
| 10 dB | 1536 / 384 | 1536 | 0.0328 | 0.0326 | 1536 / 1532 | 91 | 1536 |

#### COCO captions of one image, K = 2, U = 4; overload, gen_joint = pruned Proposition 2; residual phase N(0, (30 deg)^2), receivers use the effective gain g cos(phi): data/shared/shared_coco_K2_U4_ph30eff.json

- dataset coco, K = 2, U = 4, 2000 frames, gains rayleigh, SNRs -5/0/10 dB, receivers use the effective gain g cos(phi) (effective gains known); load masks: spec rule (384/384/384/384 coefficients per user)
- content statistics of the design C (fit): kappa_bar = 0.3935 (tr C_model 0.3935, tr C of the content 0.3935, model is the true content: True); kappa_1 = 0.7416; lam-weighted kappa per 128-group 1..6: 0.434/0.151/0.107/0.089/0.075/0.066; mean kappa_i per group: 0.307/0.146/0.106/0.088/0.075/0.065; kappa_i of coefficients 1..10: 0.742/0.154/0.326/0.687/0.630/0.445/0.779/0.596/0.425/0.476

| mode | metric | -5 dB | 0 dB | 10 dB |
|---|---|---|---|---|
| load_pu | CosSim | 0.6279 | 0.7376 | 0.9018 |
| load_pu | weakest user CosSim | 0.4110 | 0.5409 | 0.7905 |
| load_pu | NMSE | 0.5754 | 0.4287 | 0.1750 |
| load_joint | CosSim | 0.6993 | 0.7864 | 0.9162 |
| load_joint | weakest user CosSim | 0.5771 | 0.6669 | 0.8368 |
| load_joint | NMSE | 0.5021 | 0.3723 | 0.1560 |
| gen_joint | CosSim | 0.7130 | 0.7904 | 0.9168 |
| gen_joint | weakest user CosSim | 0.5934 | 0.6720 | 0.8370 |
| gen_joint | NMSE | 0.4832 | 0.3663 | 0.1550 |
| gen_eq | CosSim | 0.6541 | 0.6852 | 0.7165 |
| gen_eq | weakest user CosSim | 0.5203 | 0.5463 | 0.5741 |
| gen_eq | NMSE | 0.5633 | 0.5207 | 0.4753 |
| gen_k0 | CosSim | 0.6989 | 0.7864 | 0.9161 |
| gen_k0 | weakest user CosSim | 0.5768 | 0.6664 | 0.8360 |
| gen_k0 | NMSE | 0.5025 | 0.3722 | 0.1562 |
| gen_pc | CosSim | 0.6983 | 0.7783 | 0.9123 |
| gen_pc | weakest user CosSim | 0.5672 | 0.6469 | 0.8245 |
| gen_pc | NMSE | 0.5017 | 0.3828 | 0.1618 |

| SNR | gen_joint - load_joint CosSim [95% CI] | gen_joint - load_joint weakest [95% CI] | overlap NMSE reduction 1 - gen_joint/load_joint [95% CI] | load_joint - load_pu CosSim [95% CI] | receiver NMSE reduction 1 - load_joint/load_pu [95% CI] | total NMSE reduction 1 - gen_joint/load_pu [95% CI] |
|---|---|---|---|---|---|---|
| -5 dB | +0.0138 [+0.0133, +0.0142] | +0.0163 [+0.0157, +0.0170] | +3.75 % [+3.64, +3.85] | +0.0713 [+0.0691, +0.0737] | +12.75 % [+12.47, +13.06] | +16.02 % [+15.73, +16.36] |
| 0 dB | +0.0040 [+0.0039, +0.0041] | +0.0051 [+0.0048, +0.0053] | +1.63 % [+1.57, +1.68] | +0.0489 [+0.0469, +0.0508] | +13.15 % [+12.80, +13.49] | +14.57 % [+14.20, +14.92] |
| 10 dB | +0.0005 [+0.0005, +0.0006] | +0.0002 [+0.0001, +0.0003] | +0.65 % [+0.61, +0.69] | +0.0144 [+0.0133, +0.0156] | +10.84 % [+10.35, +11.32] | +11.42 % [+10.92, +11.91] |

| SNR | gen_eq - load_joint CosSim [95% CI]; NMSE reduction 1 - gen_eq/load_joint [95% CI] | gen_k0 - load_joint CosSim [95% CI]; NMSE reduction 1 - gen_k0/load_joint [95% CI] | gen_pc - load_joint CosSim [95% CI]; NMSE reduction 1 - gen_pc/load_joint [95% CI] |
|---|---|---|---|
| -5 dB | -0.0451 [-0.0476, -0.0424]; -12.19 % [-12.89, -11.43] | -0.0003 [-0.0017, +0.0009]; -0.08 % [-0.41, +0.22] | -0.0009 [-0.0017, -0.0002]; +0.06 % [-0.13, +0.23] |
| 0 dB | -0.1012 [-0.1040, -0.0985]; -39.85 % [-41.23, -38.62] | -0.0000 [-0.0010, +0.0009]; +0.04 % [-0.28, +0.34] | -0.0081 [-0.0086, -0.0075]; -2.82 % [-2.99, -2.64] |
| 10 dB | -0.1998 [-0.2028, -0.1965]; -204.67 % [-210.17, -198.76] | -0.0001 [-0.0005, +0.0002]; -0.11 % [-0.44, +0.23] | -0.0039 [-0.0042, -0.0036]; -3.69 % [-3.92, -3.46] |

| SNR | load masks: uses / coefficients sent | gen_joint: uses | gen_joint overlap index t | gen_joint t_overlap_modes | modes active / pruned | partial coefficients | n_data |
|---|---|---|---|---|---|---|---|
| -5 dB | 428 / 107 | 456 | 0.2666 | 0.2666 | 456 / 0 | 0 | 1536 |
| 0 dB | 1008 / 252 | 1051 | 0.1303 | 0.1303 | 1051 / 0 | 0 | 1536 |
| 10 dB | 1536 / 384 | 1536 | 0.0328 | 0.0326 | 1536 / 1532 | 91 | 1536 |

#### COCO captions of one image, K = 2, U = 5; overload, gen_joint = pruned Proposition 2; load masks with the rank rule of the manuscript (user j sends min(ceil((N - j)/U) 128, d_b)), load modes only: data/shared/shared_coco_K2_U5_rank.json

- dataset coco, K = 2, U = 5, 2000 frames, gains rayleigh, SNRs -10/-5/0/5/10/20 dB, receivers know the gains; load masks: rank rule (384/384/256/256/256 coefficients per user)
- content statistics of the design C (fit): kappa_bar = 0.3935 (tr C_model 0.3935, tr C of the content 0.3935, model is the true content: True); kappa_1 = 0.7416; lam-weighted kappa per 128-group 1..6: 0.434/0.151/0.107/0.089/0.075/0.066; mean kappa_i per group: 0.307/0.146/0.106/0.088/0.075/0.065; kappa_i of coefficients 1..10: 0.742/0.154/0.326/0.687/0.630/0.445/0.779/0.596/0.425/0.476

| mode | metric | -10 dB | -5 dB | 0 dB | 5 dB | 10 dB | 20 dB |
|---|---|---|---|---|---|---|---|
| load_pu | CosSim | 0.5203 | 0.6434 | 0.7577 | 0.8430 | 0.9105 | 0.9623 |
| load_pu | weakest user CosSim | 0.2769 | 0.4144 | 0.5576 | 0.6700 | 0.8023 | 0.9286 |
| load_pu | NMSE | 0.7055 | 0.5623 | 0.4059 | 0.2733 | 0.1642 | 0.0730 |
| load_joint | CosSim | 0.6075 | 0.7115 | 0.8021 | 0.8690 | 0.9208 | 0.9640 |
| load_joint | weakest user CosSim | 0.4693 | 0.5789 | 0.6815 | 0.7558 | 0.8410 | 0.9351 |
| load_joint | NMSE | 0.6241 | 0.4870 | 0.3505 | 0.2393 | 0.1493 | 0.0702 |

| SNR | load_joint - load_pu CosSim [95% CI] | receiver NMSE reduction 1 - load_joint/load_pu [95% CI] |
|---|---|---|
| -10 dB | +0.0872 [+0.0849, +0.0896] | +11.53 % [+11.27, +11.78] |
| -5 dB | +0.0681 [+0.0662, +0.0700] | +13.39 % [+13.12, +13.67] |
| 0 dB | +0.0444 [+0.0428, +0.0460] | +13.63 % [+13.31, +13.94] |
| 5 dB | +0.0260 [+0.0248, +0.0273] | +12.46 % [+12.08, +12.83] |
| 10 dB | +0.0103 [+0.0097, +0.0109] | +9.04 % [+8.67, +9.41] |
| 20 dB | +0.0017 [+0.0015, +0.0019] | +3.74 % [+3.40, +4.11] |

| SNR | load masks: uses / coefficients sent | gen_joint: uses | gen_joint overlap index t | gen_joint t_overlap_modes | modes active / pruned | partial coefficients | n_data |
|---|---|---|---|---|---|---|---|
| -10 dB | 205 / 41 | - | - | - | - / - | - | - |
| -5 dB | 450 / 90 | - | - | - | - / - | - | - |
| 0 dB | 1070 / 214 | - | - | - | - / - | - | - |
| 5 dB | 1536 / 384 | - | - | - | - / - | - | - |
| 10 dB | 1536 / 384 | - | - | - | - / - | - | - |
| 20 dB | 1536 / 384 | - | - | - | - / - | - | - |

#### COCO captions of one image, K = 4, U = 2; dense random masks (rnd) next to the load masks and Proposition 2: data/shared/shared_coco_K4_U2_rnd.json

- dataset coco, K = 4, U = 2, 500 frames, gains rayleigh, SNRs 0/10/20 dB, receivers know the gains; load masks: spec rule (768/768 coefficients per user)
- content statistics of the design C (fit): kappa_bar = 0.3935 (tr C_model 0.3935, tr C of the content 0.3935, model is the true content: True); kappa_1 = 0.7416; lam-weighted kappa per 128-group 1..6: 0.434/0.151/0.107/0.089/0.075/0.066; mean kappa_i per group: 0.307/0.146/0.106/0.088/0.075/0.065; kappa_i of coefficients 1..10: 0.742/0.154/0.326/0.687/0.630/0.445/0.779/0.596/0.425/0.476

| mode | metric | 0 dB | 10 dB | 20 dB |
|---|---|---|---|---|
| load_pu | CosSim | 0.8752 | 0.9693 | 0.9952 |
| load_pu | weakest user CosSim | 0.8201 | 0.9493 | 0.9916 |
| load_pu | NMSE | 0.2204 | 0.0566 | 0.0092 |
| load_joint | CosSim | 0.8903 | 0.9722 | 0.9955 |
| load_joint | weakest user CosSim | 0.8487 | 0.9549 | 0.9921 |
| load_joint | NMSE | 0.2015 | 0.0529 | 0.0088 |
| gen_joint | CosSim | 0.8907 | 0.9722 | 0.9955 |
| gen_joint | weakest user CosSim | 0.8491 | 0.9549 | 0.9921 |
| gen_joint | NMSE | 0.2009 | 0.0529 | 0.0088 |
| rnd | CosSim | 0.8847 | 0.9673 | 0.9941 |
| rnd | weakest user CosSim | 0.8404 | 0.9473 | 0.9899 |
| rnd | NMSE | 0.2109 | 0.0624 | 0.0115 |

| SNR | gen_joint - load_joint CosSim [95% CI] | gen_joint - load_joint weakest [95% CI] | overlap NMSE reduction 1 - gen_joint/load_joint [95% CI] | load_joint - load_pu CosSim [95% CI] | receiver NMSE reduction 1 - load_joint/load_pu [95% CI] | total NMSE reduction 1 - gen_joint/load_pu [95% CI] |
|---|---|---|---|---|---|---|
| 0 dB | +0.0004 [+0.0003, +0.0004] | +0.0004 [+0.0004, +0.0005] | +0.30 % [+0.26, +0.34] | +0.0151 [+0.0120, +0.0189] | +8.59 % [+7.49, +9.80] | +8.86 % [+7.74, +10.07] |
| 10 dB | +0.0000 [+0.0000, +0.0000] | +0.0000 [+0.0000, +0.0000] | +0.01 % [+0.01, +0.02] | +0.0028 [+0.0016, +0.0043] | +6.58 % [+4.61, +8.64] | +6.59 % [+4.62, +8.65] |
| 20 dB | -0.0000 [-0.0000, +0.0000] | -0.0000 [-0.0000, +0.0000] | -0.00 % [-0.00, +0.00] | +0.0003 [+0.0001, +0.0005] | +4.64 % [+1.57, +7.65] | +4.64 % [+1.57, +7.65] |

| SNR | rnd - load_joint CosSim [95% CI]; NMSE reduction 1 - rnd/load_joint [95% CI] |
|---|---|
| 0 dB | -0.0056 [-0.0070, -0.0039]; -4.66 % [-5.80, -3.34] |
| 10 dB | -0.0049 [-0.0058, -0.0040]; -18.03 % [-21.20, -14.94] |
| 20 dB | -0.0014 [-0.0017, -0.0012]; -30.30 % [-35.82, -25.28] |

| SNR | load masks: uses / coefficients sent | gen_joint: uses | gen_joint overlap index t | gen_joint t_overlap_modes | modes active / pruned | partial coefficients | n_data |
|---|---|---|---|---|---|---|---|
| 0 dB | 1312 / 656 | 1380 | 0.0669 | 0.0669 | 1317 / 0 | 0 | 3072 |
| 10 dB | 1534 / 767 | 1534 | 0.0072 | 0.0072 | 1534 / 0 | 0 | 3072 |
| 20 dB | 1534 / 767 | 1534 | 0.0007 | 0.0007 | 1534 / 0 | 0 | 3072 |

#### COCO captions of one image, K = 4, U = 4; design and receiver C fitted on the first 100 fit images only (estimation noise): data/shared/shared_coco_K4_U4_cf100.json

- dataset coco, K = 4, U = 4, 2000 frames, gains rayleigh, SNRs -5/0/10 dB, receivers know the gains; load masks: spec rule (768/768/768/768 coefficients per user)
- content statistics of the design C (fit): kappa_bar = 0.1322 (tr C_model 0.1319, tr C of the content 0.3935, model is the true content: False); kappa_1 = 0.3283; lam-weighted kappa per 128-group 1..6: 0.149/0.034/0.015/0.012/0.007/0.005; mean kappa_i per group: 0.095/0.030/0.014/0.012/0.007/0.005; kappa_i of coefficients 1..10: 0.328/0.020/0.099/0.259/0.227/0.132/0.303/0.166/0.096/0.195

| mode | metric | -5 dB | 0 dB | 10 dB |
|---|---|---|---|---|
| load_pu | CosSim | 0.7352 | 0.8304 | 0.9514 |
| load_pu | weakest user CosSim | 0.5578 | 0.6845 | 0.8836 |
| load_pu | NMSE | 0.4372 | 0.2944 | 0.0899 |
| load_joint | CosSim | 0.7530 | 0.8316 | 0.9349 |
| load_joint | weakest user CosSim | 0.6137 | 0.6947 | 0.8407 |
| load_joint | NMSE | 0.4205 | 0.2956 | 0.1192 |
| gen_joint | CosSim | 0.7555 | 0.8320 | 0.9349 |
| gen_joint | weakest user CosSim | 0.6166 | 0.6950 | 0.8407 |
| gen_joint | NMSE | 0.4169 | 0.2949 | 0.1191 |
| gen_eq | CosSim | 0.6498 | 0.6342 | 0.5564 |
| gen_eq | weakest user CosSim | 0.4729 | 0.3948 | 0.1831 |
| gen_eq | NMSE | 0.5670 | 0.5807 | 0.7511 |
| gen_k0 | CosSim | 0.7528 | 0.8314 | 0.9347 |
| gen_k0 | weakest user CosSim | 0.6133 | 0.6940 | 0.8394 |
| gen_k0 | NMSE | 0.4209 | 0.2957 | 0.1194 |
| gen_pc | CosSim | 0.7615 | 0.8432 | 0.9539 |
| gen_pc | weakest user CosSim | 0.6300 | 0.7240 | 0.8925 |
| gen_pc | NMSE | 0.4100 | 0.2798 | 0.0869 |

| SNR | gen_joint - load_joint CosSim [95% CI] | gen_joint - load_joint weakest [95% CI] | overlap NMSE reduction 1 - gen_joint/load_joint [95% CI] | load_joint - load_pu CosSim [95% CI] | receiver NMSE reduction 1 - load_joint/load_pu [95% CI] | total NMSE reduction 1 - gen_joint/load_pu [95% CI] |
|---|---|---|---|---|---|---|
| -5 dB | +0.0025 [+0.0024, +0.0026] | +0.0028 [+0.0026, +0.0031] | +0.86 % [+0.84, +0.89] | +0.0178 [+0.0166, +0.0192] | +3.83 % [+3.63, +4.05] | +4.66 % [+4.45, +4.89] |
| 0 dB | +0.0005 [+0.0004, +0.0005] | +0.0003 [+0.0002, +0.0004] | +0.26 % [+0.25, +0.27] | +0.0011 [+0.0005, +0.0017] | -0.42 % [-0.66, -0.19] | -0.16 % [-0.39, +0.07] |
| 10 dB | +0.0000 [-0.0000, +0.0000] | -0.0000 [-0.0001, -0.0000] | +0.00 % [+0.00, +0.00] | -0.0165 [-0.0171, -0.0159] | -32.52 % [-33.50, -31.53] | -32.52 % [-33.49, -31.52] |

| SNR | gen_eq - load_joint CosSim [95% CI]; NMSE reduction 1 - gen_eq/load_joint [95% CI] | gen_k0 - load_joint CosSim [95% CI]; NMSE reduction 1 - gen_k0/load_joint [95% CI] | gen_pc - load_joint CosSim [95% CI]; NMSE reduction 1 - gen_pc/load_joint [95% CI] |
|---|---|---|---|
| -5 dB | -0.1032 [-0.1062, -0.1001]; -34.83 % [-35.90, -33.70] | -0.0001 [-0.0013, +0.0009]; -0.08 % [-0.39, +0.18] | +0.0085 [+0.0079, +0.0092]; +2.49 % [+2.34, +2.63] |
| 0 dB | -0.1974 [-0.2008, -0.1939]; -96.42 % [-98.86, -93.94] | -0.0002 [-0.0009, +0.0007]; -0.02 % [-0.32, +0.30] | +0.0117 [+0.0111, +0.0123]; +5.35 % [+5.16, +5.53] |
| 10 dB | -0.3786 [-0.3830, -0.3745]; -530.39 % [-549.18, -511.39] | -0.0003 [-0.0007, +0.0001]; -0.20 % [-0.63, +0.20] | +0.0189 [+0.0182, +0.0197]; +27.07 % [+26.53, +27.63] |

| SNR | load masks: uses / coefficients sent | gen_joint: uses | gen_joint overlap index t | gen_joint t_overlap_modes | modes active / pruned | partial coefficients | n_data |
|---|---|---|---|---|---|---|---|
| -5 dB | 716 / 179 | 800 | 0.0515 | 0.0515 | 725 / 0 | 0 | 3072 |
| 0 dB | 1692 / 423 | 1748 | 0.0212 | 0.0212 | 1679 / 0 | 0 | 3072 |
| 10 dB | 3068 / 767 | 3068 | 0.0024 | 0.0024 | 3068 / 0 | 0 | 3072 |

#### COCO captions of one image, K = 4, U = 4; design and receiver C fitted on the first 1000 fit images only (estimation noise): data/shared/shared_coco_K4_U4_cf1000.json

- dataset coco, K = 4, U = 4, 2000 frames, gains rayleigh, SNRs -5/0/10 dB, receivers know the gains; load masks: spec rule (768/768/768/768 coefficients per user)
- content statistics of the design C (fit): kappa_bar = 0.3608 (tr C_model 0.3608, tr C of the content 0.3935, model is the true content: False); kappa_1 = 0.6555; lam-weighted kappa per 128-group 1..6: 0.397/0.147/0.105/0.089/0.073/0.065; mean kappa_i per group: 0.291/0.141/0.103/0.088/0.073/0.063; kappa_i of coefficients 1..10: 0.656/0.147/0.284/0.618/0.552/0.399/0.686/0.513/0.382/0.441

| mode | metric | -5 dB | 0 dB | 10 dB |
|---|---|---|---|---|
| load_pu | CosSim | 0.7352 | 0.8304 | 0.9514 |
| load_pu | weakest user CosSim | 0.5578 | 0.6845 | 0.8836 |
| load_pu | NMSE | 0.4372 | 0.2944 | 0.0899 |
| load_joint | CosSim | 0.7757 | 0.8506 | 0.9513 |
| load_joint | weakest user CosSim | 0.6637 | 0.7451 | 0.8897 |
| load_joint | NMSE | 0.3906 | 0.2693 | 0.0920 |
| gen_joint | CosSim | 0.7813 | 0.8519 | 0.9514 |
| gen_joint | weakest user CosSim | 0.6703 | 0.7464 | 0.8898 |
| gen_joint | NMSE | 0.3822 | 0.2671 | 0.0919 |
| gen_eq | CosSim | 0.6801 | 0.6955 | 0.7048 |
| gen_eq | weakest user CosSim | 0.5408 | 0.5467 | 0.5493 |
| gen_eq | NMSE | 0.5279 | 0.5052 | 0.4915 |
| gen_k0 | CosSim | 0.7754 | 0.8503 | 0.9511 |
| gen_k0 | weakest user CosSim | 0.6630 | 0.7444 | 0.8888 |
| gen_k0 | NMSE | 0.3912 | 0.2695 | 0.0923 |
| gen_pc | CosSim | 0.7747 | 0.8499 | 0.9551 |
| gen_pc | weakest user CosSim | 0.6563 | 0.7405 | 0.8967 |
| gen_pc | NMSE | 0.3911 | 0.2698 | 0.0849 |

| SNR | gen_joint - load_joint CosSim [95% CI] | gen_joint - load_joint weakest [95% CI] | overlap NMSE reduction 1 - gen_joint/load_joint [95% CI] | load_joint - load_pu CosSim [95% CI] | receiver NMSE reduction 1 - load_joint/load_pu [95% CI] | total NMSE reduction 1 - gen_joint/load_pu [95% CI] |
|---|---|---|---|---|---|---|
| -5 dB | +0.0056 [+0.0054, +0.0058] | +0.0067 [+0.0063, +0.0070] | +2.16 % [+2.09, +2.23] | +0.0405 [+0.0388, +0.0424] | +10.66 % [+10.36, +10.98] | +12.59 % [+12.27, +12.93] |
| 0 dB | +0.0013 [+0.0012, +0.0014] | +0.0013 [+0.0012, +0.0015] | +0.81 % [+0.77, +0.85] | +0.0201 [+0.0189, +0.0212] | +8.52 % [+8.16, +8.87] | +9.27 % [+8.89, +9.61] |
| 10 dB | +0.0001 [+0.0001, +0.0001] | +0.0001 [+0.0001, +0.0001] | +0.12 % [+0.11, +0.12] | -0.0001 [-0.0006, +0.0004] | -2.31 % [-2.99, -1.62] | -2.18 % [-2.87, -1.50] |

| SNR | gen_eq - load_joint CosSim [95% CI]; NMSE reduction 1 - gen_eq/load_joint [95% CI] | gen_k0 - load_joint CosSim [95% CI]; NMSE reduction 1 - gen_k0/load_joint [95% CI] | gen_pc - load_joint CosSim [95% CI]; NMSE reduction 1 - gen_pc/load_joint [95% CI] |
|---|---|---|---|
| -5 dB | -0.0957 [-0.0983, -0.0929]; -35.12 % [-36.26, -33.97] | -0.0004 [-0.0013, +0.0004]; -0.13 % [-0.43, +0.14] | -0.0010 [-0.0015, -0.0005]; -0.13 % [-0.28, +0.03] |
| 0 dB | -0.1550 [-0.1580, -0.1521]; -87.60 % [-89.80, -85.42] | -0.0003 [-0.0009, +0.0004]; -0.07 % [-0.37, +0.21] | -0.0007 [-0.0010, -0.0003]; -0.19 % [-0.35, +0.00] |
| 10 dB | -0.2465 [-0.2497, -0.2433]; -434.29 % [-447.90, -421.15] | -0.0002 [-0.0005, +0.0000]; -0.33 % [-0.72, +0.08] | +0.0038 [+0.0036, +0.0040]; +7.74 % [+7.35, +8.10] |

| SNR | load masks: uses / coefficients sent | gen_joint: uses | gen_joint overlap index t | gen_joint t_overlap_modes | modes active / pruned | partial coefficients | n_data |
|---|---|---|---|---|---|---|---|
| -5 dB | 716 / 179 | 1112 | 0.1614 | 0.1614 | 743 / 0 | 0 | 3072 |
| 0 dB | 1692 / 423 | 2164 | 0.0798 | 0.0798 | 1714 / 0 | 0 | 3072 |
| 10 dB | 3068 / 767 | 3068 | 0.0104 | 0.0104 | 3068 / 0 | 0 | 3072 |

#### COCO captions of one image, K = 4, U = 4; design and receiver C fitted on the other caption corpus, in this corpus' basis (mismatch): data/shared/shared_coco_K4_U4_cfother.json

- dataset coco, K = 4, U = 4, 2000 frames, gains rayleigh, SNRs -5/0/10 dB, receivers know the gains; load masks: spec rule (768/768/768/768 coefficients per user)
- content statistics of the design C (fit): kappa_bar = 0.1985 (tr C_model 0.1985, tr C of the content 0.3935, model is the true content: False); kappa_1 = 0.0707; lam-weighted kappa per 128-group 1..6: 0.200/0.209/0.179/0.165/0.156/0.155; mean kappa_i per group: 0.237/0.206/0.178/0.164/0.156/0.154; kappa_i of coefficients 1..10: 0.071/0.035/0.140/0.172/0.289/0.187/0.171/0.247/0.245/0.254

| mode | metric | -5 dB | 0 dB | 10 dB |
|---|---|---|---|---|
| load_pu | CosSim | 0.7352 | 0.8304 | 0.9514 |
| load_pu | weakest user CosSim | 0.5578 | 0.6845 | 0.8836 |
| load_pu | NMSE | 0.4372 | 0.2944 | 0.0899 |
| load_joint | CosSim | 0.7570 | 0.8400 | 0.9494 |
| load_joint | weakest user CosSim | 0.6193 | 0.7168 | 0.8843 |
| load_joint | NMSE | 0.4148 | 0.2841 | 0.0948 |
| gen_joint | CosSim | 0.7624 | 0.8418 | 0.9496 |
| gen_joint | weakest user CosSim | 0.6263 | 0.7196 | 0.8848 |
| gen_joint | NMSE | 0.4070 | 0.2812 | 0.0944 |
| gen_eq | CosSim | 0.6686 | 0.6877 | 0.7037 |
| gen_eq | weakest user CosSim | 0.5183 | 0.5298 | 0.5449 |
| gen_eq | NMSE | 0.5483 | 0.5218 | 0.5003 |
| gen_k0 | CosSim | 0.7569 | 0.8393 | 0.9494 |
| gen_k0 | weakest user CosSim | 0.6196 | 0.7143 | 0.8841 |
| gen_k0 | NMSE | 0.4153 | 0.2847 | 0.0949 |
| gen_pc | CosSim | 0.7658 | 0.8459 | 0.9543 |
| gen_pc | weakest user CosSim | 0.6349 | 0.7291 | 0.8938 |
| gen_pc | NMSE | 0.4031 | 0.2755 | 0.0861 |

| SNR | gen_joint - load_joint CosSim [95% CI] | gen_joint - load_joint weakest [95% CI] | overlap NMSE reduction 1 - gen_joint/load_joint [95% CI] | load_joint - load_pu CosSim [95% CI] | receiver NMSE reduction 1 - load_joint/load_pu [95% CI] | total NMSE reduction 1 - gen_joint/load_pu [95% CI] |
|---|---|---|---|---|---|---|
| -5 dB | +0.0054 [+0.0051, +0.0056] | +0.0069 [+0.0065, +0.0074] | +1.88 % [+1.81, +1.95] | +0.0219 [+0.0206, +0.0231] | +5.12 % [+4.93, +5.32] | +6.91 % [+6.69, +7.13] |
| 0 dB | +0.0018 [+0.0017, +0.0019] | +0.0029 [+0.0027, +0.0031] | +1.02 % [+0.98, +1.06] | +0.0095 [+0.0087, +0.0103] | +3.51 % [+3.28, +3.72] | +4.49 % [+4.25, +4.71] |
| 10 dB | +0.0002 [+0.0002, +0.0002] | +0.0005 [+0.0004, +0.0005] | +0.39 % [+0.39, +0.40] | -0.0021 [-0.0024, -0.0018] | -5.46 % [-5.97, -4.95] | -5.04 % [-5.55, -4.54] |

| SNR | gen_eq - load_joint CosSim [95% CI]; NMSE reduction 1 - gen_eq/load_joint [95% CI] | gen_k0 - load_joint CosSim [95% CI]; NMSE reduction 1 - gen_k0/load_joint [95% CI] | gen_pc - load_joint CosSim [95% CI]; NMSE reduction 1 - gen_pc/load_joint [95% CI] |
|---|---|---|---|
| -5 dB | -0.0885 [-0.0913, -0.0853]; -32.17 % [-33.28, -31.03] | -0.0001 [-0.0013, +0.0009]; -0.10 % [-0.41, +0.17] | +0.0087 [+0.0082, +0.0093]; +2.82 % [+2.69, +2.96] |
| 0 dB | -0.1523 [-0.1552, -0.1494]; -83.70 % [-85.80, -81.58] | -0.0006 [-0.0014, +0.0001]; -0.23 % [-0.57, +0.09] | +0.0059 [+0.0055, +0.0063]; +3.01 % [+2.86, +3.15] |
| 10 dB | -0.2457 [-0.2488, -0.2427]; -427.70 % [-441.36, -414.79] | +0.0000 [-0.0003, +0.0004]; -0.05 % [-0.50, +0.40] | +0.0049 [+0.0047, +0.0052]; +9.22 % [+8.93, +9.52] |

| SNR | load masks: uses / coefficients sent | gen_joint: uses | gen_joint overlap index t | gen_joint t_overlap_modes | modes active / pruned | partial coefficients | n_data |
|---|---|---|---|---|---|---|---|
| -5 dB | 716 / 179 | 1276 | 0.1497 | 0.1497 | 721 / 0 | 0 | 3072 |
| 0 dB | 1692 / 423 | 2500 | 0.0959 | 0.0959 | 1681 / 0 | 0 | 3072 |
| 10 dB | 3068 / 767 | 3068 | 0.0144 | 0.0144 | 3068 / 0 | 0 | 3072 |

#### COCO captions of one image, K = 4, U = 4; control: captions of U distinct images (true C = 0), design and receiver use C = 0: data/shared/shared_coco_K4_U4_ctrl.json

- dataset coco, K = 4, U = 4, 2000 frames, gains rayleigh, SNRs -10/-5/0/5/10/20 dB, receivers know the gains; load masks: spec rule (768/768/768/768 coefficients per user)
- content statistics of the design C (fit): kappa_bar = 0.0000 (tr C_model 0.0000, tr C of the content 0.3935, model is the true content: True); kappa_1 = 0.0000; lam-weighted kappa per 128-group 1..6: 0.000/0.000/0.000/0.000/0.000/0.000; mean kappa_i per group: 0.000/0.000/0.000/0.000/0.000/0.000; kappa_i of coefficients 1..10: 0.000/0.000/0.000/0.000/0.000/0.000/0.000/0.000/0.000/0.000

| mode | metric | -10 dB | -5 dB | 0 dB | 5 dB | 10 dB | 20 dB |
|---|---|---|---|---|---|---|---|
| load_pu | CosSim | 0.6190 | 0.7360 | 0.8311 | 0.9016 | 0.9512 | 0.9910 |
| load_pu | weakest user CosSim | 0.4152 | 0.5578 | 0.6867 | 0.7887 | 0.8825 | 0.9736 |
| load_pu | NMSE | 0.5923 | 0.4363 | 0.2938 | 0.1768 | 0.0902 | 0.0173 |
| load_joint | CosSim | 0.6190 | 0.7360 | 0.8311 | 0.9016 | 0.9512 | 0.9910 |
| load_joint | weakest user CosSim | 0.4152 | 0.5578 | 0.6867 | 0.7887 | 0.8825 | 0.9736 |
| load_joint | NMSE | 0.5923 | 0.4363 | 0.2938 | 0.1768 | 0.0902 | 0.0173 |
| gen_joint | CosSim | 0.6190 | 0.7360 | 0.8311 | 0.9016 | 0.9512 | 0.9910 |
| gen_joint | weakest user CosSim | 0.4152 | 0.5578 | 0.6867 | 0.7887 | 0.8825 | 0.9736 |
| gen_joint | NMSE | 0.5923 | 0.4363 | 0.2938 | 0.1768 | 0.0902 | 0.0173 |
| gen_eq | CosSim | 0.3718 | 0.4100 | 0.4328 | 0.4462 | 0.4536 | 0.4533 |
| gen_eq | weakest user CosSim | 0.1218 | 0.1573 | 0.1726 | 0.1790 | 0.1909 | 0.1835 |
| gen_eq | NMSE | 0.8317 | 0.7967 | 0.7739 | 0.7580 | 0.7528 | 0.7512 |
| gen_k0 | CosSim | 0.6186 | 0.7361 | 0.8308 | 0.9016 | 0.9513 | 0.9911 |
| gen_k0 | weakest user CosSim | 0.4144 | 0.5579 | 0.6848 | 0.7888 | 0.8827 | 0.9741 |
| gen_k0 | NMSE | 0.5922 | 0.4366 | 0.2940 | 0.1767 | 0.0901 | 0.0171 |
| gen_pc | CosSim | 0.6190 | 0.7360 | 0.8311 | 0.9016 | 0.9512 | 0.9910 |
| gen_pc | weakest user CosSim | 0.4152 | 0.5578 | 0.6867 | 0.7887 | 0.8825 | 0.9736 |
| gen_pc | NMSE | 0.5923 | 0.4363 | 0.2938 | 0.1768 | 0.0902 | 0.0173 |

| SNR | gen_joint - load_joint CosSim [95% CI] | gen_joint - load_joint weakest [95% CI] | overlap NMSE reduction 1 - gen_joint/load_joint [95% CI] | load_joint - load_pu CosSim [95% CI] | receiver NMSE reduction 1 - load_joint/load_pu [95% CI] | total NMSE reduction 1 - gen_joint/load_pu [95% CI] |
|---|---|---|---|---|---|---|
| -10 dB | +0.0000 [+0.0000, +0.0000] | +0.0000 [+0.0000, +0.0000] | +0.00 % [+0.00, +0.00] | +0.0000 [+0.0000, +0.0000] | +0.00 % [+0.00, +0.00] | +0.00 % [+0.00, +0.00] |
| -5 dB | +0.0000 [+0.0000, +0.0000] | +0.0000 [+0.0000, +0.0000] | +0.00 % [+0.00, +0.00] | +0.0000 [+0.0000, +0.0000] | +0.00 % [+0.00, +0.00] | +0.00 % [+0.00, +0.00] |
| 0 dB | +0.0000 [+0.0000, +0.0000] | +0.0000 [+0.0000, +0.0000] | +0.00 % [+0.00, +0.00] | +0.0000 [+0.0000, +0.0000] | +0.00 % [+0.00, +0.00] | +0.00 % [+0.00, +0.00] |
| 5 dB | +0.0000 [+0.0000, +0.0000] | +0.0000 [+0.0000, +0.0000] | +0.00 % [+0.00, +0.00] | +0.0000 [+0.0000, +0.0000] | +0.00 % [+0.00, +0.00] | +0.00 % [+0.00, +0.00] |
| 10 dB | +0.0000 [+0.0000, +0.0000] | +0.0000 [+0.0000, +0.0000] | +0.00 % [+0.00, +0.00] | +0.0000 [+0.0000, +0.0000] | +0.00 % [+0.00, +0.00] | +0.00 % [+0.00, +0.00] |
| 20 dB | +0.0000 [+0.0000, +0.0000] | +0.0000 [+0.0000, +0.0000] | +0.00 % [+0.00, +0.00] | +0.0000 [+0.0000, +0.0000] | +0.00 % [+0.00, +0.00] | +0.00 % [+0.00, +0.00] |

| SNR | gen_eq - load_joint CosSim [95% CI]; NMSE reduction 1 - gen_eq/load_joint [95% CI] | gen_k0 - load_joint CosSim [95% CI]; NMSE reduction 1 - gen_k0/load_joint [95% CI] | gen_pc - load_joint CosSim [95% CI]; NMSE reduction 1 - gen_pc/load_joint [95% CI] |
|---|---|---|---|
| -10 dB | -0.2472 [-0.2510, -0.2434]; -40.42 % [-41.31, -39.50] | -0.0004 [-0.0026, +0.0016]; +0.02 % [-0.27, +0.29] | +0.0000 [+0.0000, +0.0000]; +0.00 % [+0.00, +0.00] |
| -5 dB | -0.3260 [-0.3293, -0.3223]; -82.61 % [-84.27, -80.78] | +0.0001 [-0.0013, +0.0015]; -0.07 % [-0.35, +0.22] | +0.0000 [+0.0000, +0.0000]; +0.00 % [+0.00, +0.00] |
| 0 dB | -0.3983 [-0.4014, -0.3949]; -163.39 % [-167.22, -159.86] | -0.0002 [-0.0012, +0.0007]; -0.05 % [-0.37, +0.27] | +0.0000 [+0.0000, +0.0000]; +0.00 % [+0.00, +0.00] |
| 5 dB | -0.4554 [-0.4583, -0.4525]; -328.75 % [-336.73, -320.14] | -0.0000 [-0.0007, +0.0006]; +0.03 % [-0.35, +0.42] | +0.0000 [+0.0000, +0.0000]; +0.00 % [+0.00, +0.00] |
| 10 dB | -0.4975 [-0.5001, -0.4950]; -734.71 % [-760.44, -711.52] | +0.0001 [-0.0003, +0.0005]; +0.14 % [-0.32, +0.61] | +0.0000 [+0.0000, +0.0000]; +0.00 % [+0.00, +0.00] |
| 20 dB | -0.5377 [-0.5402, -0.5352]; -4249.80 % [-4523.71, -3995.08] | +0.0001 [+0.0000, +0.0003]; +0.89 % [+0.06, +1.84] | +0.0000 [+0.0000, +0.0000]; +0.00 % [+0.00, +0.00] |

| SNR | load masks: uses / coefficients sent | gen_joint: uses | gen_joint overlap index t | gen_joint t_overlap_modes | modes active / pruned | partial coefficients | n_data |
|---|---|---|---|---|---|---|---|
| -10 dB | 304 / 76 | 304 | -0.0000 | 0.0000 | 304 / 0 | 0 | 3072 |
| -5 dB | 716 / 179 | 716 | -0.0000 | 0.0000 | 716 / 0 | 0 | 3072 |
| 0 dB | 1692 / 423 | 1692 | -0.0000 | 0.0000 | 1692 / 0 | 0 | 3072 |
| 5 dB | 3048 / 762 | 3048 | -0.0000 | 0.0000 | 3048 / 0 | 0 | 3072 |
| 10 dB | 3068 / 767 | 3068 | -0.0000 | 0.0000 | 3068 / 0 | 0 | 3072 |
| 20 dB | 3068 / 767 | 3068 | -0.0000 | 0.0000 | 3068 / 0 | 0 | 3072 |

#### COCO captions of one image, K = 4, U = 4; control: captions of U distinct images (true C = 0), design and receiver use the fitted C (mismatch): data/shared/shared_coco_K4_U4_ctrlfit.json

- dataset coco, K = 4, U = 4, 2000 frames, gains rayleigh, SNRs -10/-5/0/5/10/20 dB, receivers know the gains; load masks: spec rule (768/768/768/768 coefficients per user)
- content statistics of the design C (fit): kappa_bar = 0.3935 (tr C_model 0.3935, tr C of the content 0.3935, model is the true content: False); kappa_1 = 0.7416; lam-weighted kappa per 128-group 1..6: 0.434/0.151/0.107/0.089/0.075/0.066; mean kappa_i per group: 0.307/0.146/0.106/0.088/0.075/0.065; kappa_i of coefficients 1..10: 0.742/0.154/0.326/0.687/0.630/0.445/0.779/0.596/0.425/0.476

| mode | metric | -10 dB | -5 dB | 0 dB | 5 dB | 10 dB | 20 dB |
|---|---|---|---|---|---|---|---|
| load_pu | CosSim | 0.6190 | 0.7360 | 0.8311 | 0.9016 | 0.9512 | 0.9910 |
| load_pu | weakest user CosSim | 0.4152 | 0.5578 | 0.6867 | 0.7887 | 0.8825 | 0.9736 |
| load_pu | NMSE | 0.5923 | 0.4363 | 0.2938 | 0.1768 | 0.0902 | 0.0173 |
| load_joint | CosSim | 0.5461 | 0.6781 | 0.7903 | 0.8780 | 0.9404 | 0.9896 |
| load_joint | weakest user CosSim | 0.2877 | 0.4409 | 0.5904 | 0.7252 | 0.8494 | 0.9685 |
| load_joint | NMSE | 0.6733 | 0.5095 | 0.3501 | 0.2123 | 0.1079 | 0.0197 |
| gen_joint | CosSim | 0.5351 | 0.6746 | 0.7891 | 0.8777 | 0.9403 | 0.9896 |
| gen_joint | weakest user CosSim | 0.2860 | 0.4396 | 0.5891 | 0.7248 | 0.8492 | 0.9685 |
| gen_joint | NMSE | 0.6888 | 0.5153 | 0.3523 | 0.2129 | 0.1081 | 0.0197 |
| gen_eq | CosSim | 0.3666 | 0.4045 | 0.4270 | 0.4411 | 0.4473 | 0.4476 |
| gen_eq | weakest user CosSim | 0.1176 | 0.1485 | 0.1603 | 0.1672 | 0.1773 | 0.1705 |
| gen_eq | NMSE | 0.8570 | 0.8204 | 0.7967 | 0.7800 | 0.7747 | 0.7733 |
| gen_k0 | CosSim | 0.5460 | 0.6778 | 0.7906 | 0.8778 | 0.9403 | 0.9897 |
| gen_k0 | weakest user CosSim | 0.2870 | 0.4413 | 0.5904 | 0.7248 | 0.8489 | 0.9690 |
| gen_k0 | NMSE | 0.6734 | 0.5101 | 0.3496 | 0.2127 | 0.1080 | 0.0196 |
| gen_pc | CosSim | 0.5585 | 0.7010 | 0.8115 | 0.8924 | 0.9476 | 0.9906 |
| gen_pc | weakest user CosSim | 0.3147 | 0.4796 | 0.6328 | 0.7599 | 0.8700 | 0.9721 |
| gen_pc | NMSE | 0.6611 | 0.4800 | 0.3193 | 0.1894 | 0.0954 | 0.0179 |

| SNR | gen_joint - load_joint CosSim [95% CI] | gen_joint - load_joint weakest [95% CI] | overlap NMSE reduction 1 - gen_joint/load_joint [95% CI] | load_joint - load_pu CosSim [95% CI] | receiver NMSE reduction 1 - load_joint/load_pu [95% CI] | total NMSE reduction 1 - gen_joint/load_pu [95% CI] |
|---|---|---|---|---|---|---|
| -10 dB | -0.0110 [-0.0117, -0.0103] | -0.0018 [-0.0031, -0.0004] | -2.31 % [-2.43, -2.19] | -0.0729 [-0.0752, -0.0706] | -13.68 % [-14.03, -13.33] | -16.31 % [-16.70, -15.91] |
| -5 dB | -0.0035 [-0.0037, -0.0033] | -0.0013 [-0.0018, -0.0008] | -1.12 % [-1.18, -1.06] | -0.0580 [-0.0597, -0.0560] | -16.79 % [-17.14, -16.35] | -18.10 % [-18.48, -17.66] |
| 0 dB | -0.0013 [-0.0013, -0.0012] | -0.0013 [-0.0015, -0.0011] | -0.65 % [-0.68, -0.62] | -0.0407 [-0.0422, -0.0393] | -19.13 % [-19.55, -18.68] | -19.91 % [-20.33, -19.46] |
| 5 dB | -0.0003 [-0.0003, -0.0003] | -0.0005 [-0.0005, -0.0004] | -0.30 % [-0.31, -0.29] | -0.0236 [-0.0247, -0.0225] | -20.09 % [-20.61, -19.62] | -20.45 % [-20.97, -19.97] |
| 10 dB | -0.0001 [-0.0001, -0.0001] | -0.0002 [-0.0002, -0.0001] | -0.13 % [-0.13, -0.12] | -0.0108 [-0.0115, -0.0101] | -19.69 % [-20.35, -19.01] | -19.84 % [-20.50, -19.16] |
| 20 dB | -0.0000 [-0.0000, -0.0000] | -0.0000 [-0.0000, -0.0000] | -0.01 % [-0.01, -0.01] | -0.0014 [-0.0016, -0.0011] | -14.28 % [-15.54, -13.04] | -14.29 % [-15.55, -13.06] |

| SNR | gen_eq - load_joint CosSim [95% CI]; NMSE reduction 1 - gen_eq/load_joint [95% CI] | gen_k0 - load_joint CosSim [95% CI]; NMSE reduction 1 - gen_k0/load_joint [95% CI] | gen_pc - load_joint CosSim [95% CI]; NMSE reduction 1 - gen_pc/load_joint [95% CI] |
|---|---|---|---|
| -10 dB | -0.1795 [-0.1828, -0.1759]; -27.28 % [-28.00, -26.58] | -0.0001 [-0.0020, +0.0018]; -0.01 % [-0.29, +0.26] | +0.0124 [+0.0113, +0.0136]; +1.81 % [+1.65, +1.99] |
| -5 dB | -0.2736 [-0.2772, -0.2698]; -61.01 % [-62.41, -59.47] | -0.0002 [-0.0015, +0.0011]; -0.10 % [-0.37, +0.16] | +0.0229 [+0.0220, +0.0238]; +5.79 % [+5.59, +5.97] |
| 0 dB | -0.3633 [-0.3670, -0.3596]; -127.59 % [-130.82, -124.60] | +0.0003 [-0.0006, +0.0012]; +0.13 % [-0.17, +0.42] | +0.0212 [+0.0203, +0.0219]; +8.78 % [+8.54, +9.01] |
| 5 dB | -0.4369 [-0.4401, -0.4337]; -267.40 % [-274.25, -259.74] | -0.0002 [-0.0009, +0.0005]; -0.17 % [-0.52, +0.19] | +0.0144 [+0.0138, +0.0149]; +10.78 % [+10.50, +11.05] |
| 10 dB | -0.4931 [-0.4959, -0.4901]; -617.71 % [-640.57, -596.98] | -0.0001 [-0.0004, +0.0003]; -0.08 % [-0.50, +0.35] | +0.0072 [+0.0068, +0.0077]; +11.59 % [+11.20, +11.97] |
| 20 dB | -0.5420 [-0.5447, -0.5393]; -3817.89 % [-4082.06, -3571.91] | +0.0001 [-0.0000, +0.0002]; +0.48 % [-0.33, +1.27] | +0.0010 [+0.0008, +0.0011]; +9.41 % [+8.70, +10.14] |

| SNR | load masks: uses / coefficients sent | gen_joint: uses | gen_joint overlap index t | gen_joint t_overlap_modes | modes active / pruned | partial coefficients | n_data |
|---|---|---|---|---|---|---|---|
| -10 dB | 304 / 76 | 640 | 0.3464 | 0.3464 | 307 / 0 | 0 | 3072 |
| -5 dB | 716 / 179 | 1140 | 0.1736 | 0.1736 | 756 / 0 | 0 | 3072 |
| 0 dB | 1692 / 423 | 2176 | 0.0840 | 0.0840 | 1711 / 0 | 0 | 3072 |
| 5 dB | 3048 / 762 | 3064 | 0.0344 | 0.0344 | 3037 / 0 | 0 | 3072 |
| 10 dB | 3068 / 767 | 3068 | 0.0109 | 0.0109 | 3068 / 0 | 0 | 3072 |
| 20 dB | 3068 / 767 | 3068 | 0.0011 | 0.0011 | 3068 / 0 | 0 | 3072 |

#### COCO captions of one image, K = 4, U = 4; residual phase N(0, (10 deg)^2), receivers use |h| (phase not known): data/shared/shared_coco_K4_U4_ph10abs.json

- dataset coco, K = 4, U = 4, 2000 frames, gains rayleigh, SNRs -5/0/10 dB, receivers use |h|: the effective gain g cos(phi) is NOT known; load masks: spec rule (768/768/768/768 coefficients per user)
- content statistics of the design C (fit): kappa_bar = 0.3935 (tr C_model 0.3935, tr C of the content 0.3935, model is the true content: True); kappa_1 = 0.7416; lam-weighted kappa per 128-group 1..6: 0.434/0.151/0.107/0.089/0.075/0.066; mean kappa_i per group: 0.307/0.146/0.106/0.088/0.075/0.065; kappa_i of coefficients 1..10: 0.742/0.154/0.326/0.687/0.630/0.445/0.779/0.596/0.425/0.476

| mode | metric | -5 dB | 0 dB | 10 dB |
|---|---|---|---|---|
| load_pu | CosSim | 0.7320 | 0.8278 | 0.9503 |
| load_pu | weakest user CosSim | 0.5533 | 0.6804 | 0.8813 |
| load_pu | NMSE | 0.4418 | 0.2987 | 0.0923 |
| load_joint | CosSim | 0.7782 | 0.8542 | 0.9562 |
| load_joint | weakest user CosSim | 0.6687 | 0.7536 | 0.9013 |
| load_joint | NMSE | 0.3874 | 0.2641 | 0.0838 |
| gen_joint | CosSim | 0.7843 | 0.8557 | 0.9562 |
| gen_joint | weakest user CosSim | 0.6761 | 0.7554 | 0.9013 |
| gen_joint | NMSE | 0.3782 | 0.2616 | 0.0838 |
| gen_eq | CosSim | 0.6853 | 0.7047 | 0.7217 |
| gen_eq | weakest user CosSim | 0.5510 | 0.5645 | 0.5827 |
| gen_eq | NMSE | 0.5216 | 0.4936 | 0.4686 |
| gen_k0 | CosSim | 0.7779 | 0.8540 | 0.9560 |
| gen_k0 | weakest user CosSim | 0.6684 | 0.7529 | 0.9004 |
| gen_k0 | NMSE | 0.3877 | 0.2642 | 0.0841 |
| gen_pc | CosSim | 0.7728 | 0.8478 | 0.9542 |
| gen_pc | weakest user CosSim | 0.6538 | 0.7376 | 0.8949 |
| gen_pc | NMSE | 0.3942 | 0.2733 | 0.0871 |

| SNR | gen_joint - load_joint CosSim [95% CI] | gen_joint - load_joint weakest [95% CI] | overlap NMSE reduction 1 - gen_joint/load_joint [95% CI] | load_joint - load_pu CosSim [95% CI] | receiver NMSE reduction 1 - load_joint/load_pu [95% CI] | total NMSE reduction 1 - gen_joint/load_pu [95% CI] |
|---|---|---|---|---|---|---|
| -5 dB | +0.0061 [+0.0059, +0.0063] | +0.0074 [+0.0070, +0.0077] | +2.38 % [+2.31, +2.46] | +0.0462 [+0.0444, +0.0481] | +12.32 % [+12.02, +12.65] | +14.41 % [+14.09, +14.75] |
| 0 dB | +0.0015 [+0.0014, +0.0016] | +0.0019 [+0.0017, +0.0020] | +0.93 % [+0.89, +0.97] | +0.0264 [+0.0251, +0.0276] | +11.59 % [+11.22, +11.95] | +12.41 % [+12.02, +12.77] |
| 10 dB | +0.0000 [+0.0000, +0.0000] | +0.0001 [+0.0000, +0.0001] | +0.06 % [+0.06, +0.07] | +0.0059 [+0.0052, +0.0064] | +9.22 % [+8.66, +9.82] | +9.27 % [+8.72, +9.88] |

| SNR | gen_eq - load_joint CosSim [95% CI]; NMSE reduction 1 - gen_eq/load_joint [95% CI] | gen_k0 - load_joint CosSim [95% CI]; NMSE reduction 1 - gen_k0/load_joint [95% CI] | gen_pc - load_joint CosSim [95% CI]; NMSE reduction 1 - gen_pc/load_joint [95% CI] |
|---|---|---|---|
| -5 dB | -0.0929 [-0.0956, -0.0902]; -34.65 % [-35.79, -33.48] | -0.0003 [-0.0012, +0.0005]; -0.08 % [-0.39, +0.19] | -0.0054 [-0.0059, -0.0049]; -1.75 % [-1.92, -1.60] |
| 0 dB | -0.1495 [-0.1524, -0.1467]; -86.89 % [-89.14, -84.71] | -0.0002 [-0.0008, +0.0004]; -0.04 % [-0.35, +0.25] | -0.0064 [-0.0067, -0.0060]; -3.50 % [-3.67, -3.31] |
| 10 dB | -0.2345 [-0.2376, -0.2314]; -459.07 % [-473.83, -445.43] | -0.0002 [-0.0004, +0.0000]; -0.31 % [-0.71, +0.11] | -0.0020 [-0.0022, -0.0019]; -3.90 % [-4.17, -3.63] |

| SNR | load masks: uses / coefficients sent | gen_joint: uses | gen_joint overlap index t | gen_joint t_overlap_modes | modes active / pruned | partial coefficients | n_data |
|---|---|---|---|---|---|---|---|
| -5 dB | 716 / 179 | 1140 | 0.1736 | 0.1736 | 756 / 0 | 0 | 3072 |
| 0 dB | 1692 / 423 | 2176 | 0.0840 | 0.0840 | 1711 / 0 | 0 | 3072 |
| 10 dB | 3068 / 767 | 3068 | 0.0109 | 0.0109 | 3068 / 0 | 0 | 3072 |

#### COCO captions of one image, K = 4, U = 4; residual phase N(0, (10 deg)^2), receivers use the effective gain g cos(phi): data/shared/shared_coco_K4_U4_ph10eff.json

- dataset coco, K = 4, U = 4, 2000 frames, gains rayleigh, SNRs -5/0/10 dB, receivers use the effective gain g cos(phi) (effective gains known); load masks: spec rule (768/768/768/768 coefficients per user)
- content statistics of the design C (fit): kappa_bar = 0.3935 (tr C_model 0.3935, tr C of the content 0.3935, model is the true content: True); kappa_1 = 0.7416; lam-weighted kappa per 128-group 1..6: 0.434/0.151/0.107/0.089/0.075/0.066; mean kappa_i per group: 0.307/0.146/0.106/0.088/0.075/0.065; kappa_i of coefficients 1..10: 0.742/0.154/0.326/0.687/0.630/0.445/0.779/0.596/0.425/0.476

| mode | metric | -5 dB | 0 dB | 10 dB |
|---|---|---|---|---|
| load_pu | CosSim | 0.7320 | 0.8278 | 0.9504 |
| load_pu | weakest user CosSim | 0.5534 | 0.6804 | 0.8814 |
| load_pu | NMSE | 0.4417 | 0.2985 | 0.0918 |
| load_joint | CosSim | 0.7783 | 0.8542 | 0.9562 |
| load_joint | weakest user CosSim | 0.6687 | 0.7535 | 0.9013 |
| load_joint | NMSE | 0.3872 | 0.2638 | 0.0833 |
| gen_joint | CosSim | 0.7844 | 0.8557 | 0.9562 |
| gen_joint | weakest user CosSim | 0.6761 | 0.7554 | 0.9013 |
| gen_joint | NMSE | 0.3779 | 0.2614 | 0.0833 |
| gen_eq | CosSim | 0.6853 | 0.7048 | 0.7217 |
| gen_eq | weakest user CosSim | 0.5510 | 0.5646 | 0.5827 |
| gen_eq | NMSE | 0.5214 | 0.4934 | 0.4684 |
| gen_k0 | CosSim | 0.7780 | 0.8540 | 0.9560 |
| gen_k0 | weakest user CosSim | 0.6684 | 0.7529 | 0.9004 |
| gen_k0 | NMSE | 0.3875 | 0.2639 | 0.0836 |
| gen_pc | CosSim | 0.7728 | 0.8478 | 0.9542 |
| gen_pc | weakest user CosSim | 0.6538 | 0.7375 | 0.8949 |
| gen_pc | NMSE | 0.3940 | 0.2731 | 0.0866 |

| SNR | gen_joint - load_joint CosSim [95% CI] | gen_joint - load_joint weakest [95% CI] | overlap NMSE reduction 1 - gen_joint/load_joint [95% CI] | load_joint - load_pu CosSim [95% CI] | receiver NMSE reduction 1 - load_joint/load_pu [95% CI] | total NMSE reduction 1 - gen_joint/load_pu [95% CI] |
|---|---|---|---|---|---|---|
| -5 dB | +0.0061 [+0.0059, +0.0063] | +0.0073 [+0.0070, +0.0077] | +2.38 % [+2.31, +2.46] | +0.0462 [+0.0444, +0.0481] | +12.34 % [+12.03, +12.66] | +14.43 % [+14.10, +14.77] |
| 0 dB | +0.0015 [+0.0014, +0.0016] | +0.0019 [+0.0017, +0.0020] | +0.93 % [+0.89, +0.97] | +0.0264 [+0.0251, +0.0276] | +11.60 % [+11.22, +11.95] | +12.42 % [+12.02, +12.79] |
| 10 dB | +0.0000 [+0.0000, +0.0000] | +0.0001 [+0.0000, +0.0001] | +0.06 % [+0.06, +0.07] | +0.0058 [+0.0052, +0.0064] | +9.25 % [+8.70, +9.86] | +9.31 % [+8.76, +9.92] |

| SNR | gen_eq - load_joint CosSim [95% CI]; NMSE reduction 1 - gen_eq/load_joint [95% CI] | gen_k0 - load_joint CosSim [95% CI]; NMSE reduction 1 - gen_k0/load_joint [95% CI] | gen_pc - load_joint CosSim [95% CI]; NMSE reduction 1 - gen_pc/load_joint [95% CI] |
|---|---|---|---|
| -5 dB | -0.0930 [-0.0956, -0.0903]; -34.68 % [-35.83, -33.50] | -0.0003 [-0.0012, +0.0005]; -0.09 % [-0.39, +0.18] | -0.0054 [-0.0059, -0.0049]; -1.76 % [-1.92, -1.59] |
| 0 dB | -0.1495 [-0.1524, -0.1467]; -87.00 % [-89.26, -84.80] | -0.0002 [-0.0008, +0.0004]; -0.04 % [-0.34, +0.26] | -0.0064 [-0.0067, -0.0060]; -3.50 % [-3.68, -3.31] |
| 10 dB | -0.2345 [-0.2375, -0.2314]; -462.14 % [-477.08, -448.29] | -0.0002 [-0.0004, +0.0000]; -0.32 % [-0.72, +0.10] | -0.0020 [-0.0022, -0.0019]; -3.91 % [-4.18, -3.63] |

| SNR | load masks: uses / coefficients sent | gen_joint: uses | gen_joint overlap index t | gen_joint t_overlap_modes | modes active / pruned | partial coefficients | n_data |
|---|---|---|---|---|---|---|---|
| -5 dB | 716 / 179 | 1140 | 0.1736 | 0.1736 | 756 / 0 | 0 | 3072 |
| 0 dB | 1692 / 423 | 2176 | 0.0840 | 0.0840 | 1711 / 0 | 0 | 3072 |
| 10 dB | 3068 / 767 | 3068 | 0.0109 | 0.0109 | 3068 / 0 | 0 | 3072 |

#### COCO captions of one image, K = 4, U = 4; residual phase N(0, (30 deg)^2), receivers use |h| (phase not known): data/shared/shared_coco_K4_U4_ph30abs.json

- dataset coco, K = 4, U = 4, 2000 frames, gains rayleigh, SNRs -5/0/10 dB, receivers use |h|: the effective gain g cos(phi) is NOT known; load masks: spec rule (768/768/768/768 coefficients per user)
- content statistics of the design C (fit): kappa_bar = 0.3935 (tr C_model 0.3935, tr C of the content 0.3935, model is the true content: True); kappa_1 = 0.7416; lam-weighted kappa per 128-group 1..6: 0.434/0.151/0.107/0.089/0.075/0.066; mean kappa_i per group: 0.307/0.146/0.106/0.088/0.075/0.065; kappa_i of coefficients 1..10: 0.742/0.154/0.326/0.687/0.630/0.445/0.779/0.596/0.425/0.476

| mode | metric | -5 dB | 0 dB | 10 dB |
|---|---|---|---|---|
| load_pu | CosSim | 0.6934 | 0.7917 | 0.9296 |
| load_pu | weakest user CosSim | 0.4834 | 0.6069 | 0.8251 |
| load_pu | NMSE | 0.4925 | 0.3544 | 0.1446 |
| load_joint | CosSim | 0.7490 | 0.8264 | 0.9386 |
| load_joint | weakest user CosSim | 0.6208 | 0.7017 | 0.8551 |
| load_joint | NMSE | 0.4319 | 0.3131 | 0.1329 |
| gen_joint | CosSim | 0.7562 | 0.8282 | 0.9386 |
| gen_joint | weakest user CosSim | 0.6305 | 0.7045 | 0.8552 |
| gen_joint | NMSE | 0.4220 | 0.3104 | 0.1328 |
| gen_eq | CosSim | 0.6746 | 0.6972 | 0.7177 |
| gen_eq | weakest user CosSim | 0.5358 | 0.5534 | 0.5733 |
| gen_eq | NMSE | 0.5427 | 0.5130 | 0.4862 |
| gen_k0 | CosSim | 0.7492 | 0.8260 | 0.9385 |
| gen_k0 | weakest user CosSim | 0.6219 | 0.7003 | 0.8545 |
| gen_k0 | NMSE | 0.4319 | 0.3133 | 0.1331 |
| gen_pc | CosSim | 0.7424 | 0.8177 | 0.9351 |
| gen_pc | weakest user CosSim | 0.6031 | 0.6794 | 0.8443 |
| gen_pc | NMSE | 0.4394 | 0.3243 | 0.1377 |

| SNR | gen_joint - load_joint CosSim [95% CI] | gen_joint - load_joint weakest [95% CI] | overlap NMSE reduction 1 - gen_joint/load_joint [95% CI] | load_joint - load_pu CosSim [95% CI] | receiver NMSE reduction 1 - load_joint/load_pu [95% CI] | total NMSE reduction 1 - gen_joint/load_pu [95% CI] |
|---|---|---|---|---|---|---|
| -5 dB | +0.0071 [+0.0069, +0.0074] | +0.0097 [+0.0092, +0.0101] | +2.31 % [+2.24, +2.37] | +0.0556 [+0.0537, +0.0576] | +12.31 % [+12.04, +12.59] | +14.33 % [+14.04, +14.64] |
| 0 dB | +0.0019 [+0.0018, +0.0019] | +0.0029 [+0.0027, +0.0031] | +0.88 % [+0.85, +0.92] | +0.0346 [+0.0332, +0.0361] | +11.64 % [+11.32, +11.95] | +12.42 % [+12.08, +12.74] |
| 10 dB | +0.0001 [+0.0000, +0.0001] | +0.0001 [+0.0001, +0.0001] | +0.06 % [+0.06, +0.07] | +0.0090 [+0.0082, +0.0097] | +8.08 % [+7.66, +8.52] | +8.13 % [+7.71, +8.58] |

| SNR | gen_eq - load_joint CosSim [95% CI]; NMSE reduction 1 - gen_eq/load_joint [95% CI] | gen_k0 - load_joint CosSim [95% CI]; NMSE reduction 1 - gen_k0/load_joint [95% CI] | gen_pc - load_joint CosSim [95% CI]; NMSE reduction 1 - gen_pc/load_joint [95% CI] |
|---|---|---|---|
| -5 dB | -0.0744 [-0.0774, -0.0712]; -25.63 % [-26.61, -24.62] | +0.0001 [-0.0010, +0.0012]; +0.01 % [-0.32, +0.32] | -0.0066 [-0.0072, -0.0060]; -1.74 % [-1.89, -1.58] |
| 0 dB | -0.1292 [-0.1323, -0.1260]; -63.84 % [-65.88, -62.10] | -0.0003 [-0.0011, +0.0004]; -0.04 % [-0.38, +0.27] | -0.0087 [-0.0091, -0.0082]; -3.57 % [-3.73, -3.40] |
| 10 dB | -0.2209 [-0.2244, -0.2174]; -265.81 % [-275.32, -257.06] | -0.0001 [-0.0004, +0.0003]; -0.16 % [-0.51, +0.20] | -0.0034 [-0.0037, -0.0032]; -3.59 % [-3.80, -3.40] |

| SNR | load masks: uses / coefficients sent | gen_joint: uses | gen_joint overlap index t | gen_joint t_overlap_modes | modes active / pruned | partial coefficients | n_data |
|---|---|---|---|---|---|---|---|
| -5 dB | 716 / 179 | 1140 | 0.1736 | 0.1736 | 756 / 0 | 0 | 3072 |
| 0 dB | 1692 / 423 | 2176 | 0.0840 | 0.0840 | 1711 / 0 | 0 | 3072 |
| 10 dB | 3068 / 767 | 3068 | 0.0109 | 0.0109 | 3068 / 0 | 0 | 3072 |

#### COCO captions of one image, K = 4, U = 4; residual phase N(0, (30 deg)^2), receivers use the effective gain g cos(phi): data/shared/shared_coco_K4_U4_ph30eff.json

- dataset coco, K = 4, U = 4, 2000 frames, gains rayleigh, SNRs -5/0/10 dB, receivers use the effective gain g cos(phi) (effective gains known); load masks: spec rule (768/768/768/768 coefficients per user)
- content statistics of the design C (fit): kappa_bar = 0.3935 (tr C_model 0.3935, tr C of the content 0.3935, model is the true content: True); kappa_1 = 0.7416; lam-weighted kappa per 128-group 1..6: 0.434/0.151/0.107/0.089/0.075/0.066; mean kappa_i per group: 0.307/0.146/0.106/0.088/0.075/0.065; kappa_i of coefficients 1..10: 0.742/0.154/0.326/0.687/0.630/0.445/0.779/0.596/0.425/0.476

| mode | metric | -5 dB | 0 dB | 10 dB |
|---|---|---|---|---|
| load_pu | CosSim | 0.6968 | 0.7956 | 0.9340 |
| load_pu | weakest user CosSim | 0.4910 | 0.6166 | 0.8402 |
| load_pu | NMSE | 0.4851 | 0.3430 | 0.1184 |
| load_joint | CosSim | 0.7552 | 0.8326 | 0.9440 |
| load_joint | weakest user CosSim | 0.6356 | 0.7185 | 0.8738 |
| load_joint | NMSE | 0.4208 | 0.2982 | 0.1051 |
| gen_joint | CosSim | 0.7620 | 0.8343 | 0.9440 |
| gen_joint | weakest user CosSim | 0.6440 | 0.7208 | 0.8739 |
| gen_joint | NMSE | 0.4110 | 0.2955 | 0.1051 |
| gen_eq | CosSim | 0.6764 | 0.6990 | 0.7191 |
| gen_eq | weakest user CosSim | 0.5391 | 0.5568 | 0.5760 |
| gen_eq | NMSE | 0.5330 | 0.5010 | 0.4714 |
| gen_k0 | CosSim | 0.7549 | 0.8325 | 0.9439 |
| gen_k0 | weakest user CosSim | 0.6356 | 0.7181 | 0.8732 |
| gen_k0 | NMSE | 0.4212 | 0.2981 | 0.1052 |
| gen_pc | CosSim | 0.7486 | 0.8244 | 0.9409 |
| gen_pc | weakest user CosSim | 0.6184 | 0.6985 | 0.8645 |
| gen_pc | NMSE | 0.4288 | 0.3095 | 0.1098 |

| SNR | gen_joint - load_joint CosSim [95% CI] | gen_joint - load_joint weakest [95% CI] | overlap NMSE reduction 1 - gen_joint/load_joint [95% CI] | load_joint - load_pu CosSim [95% CI] | receiver NMSE reduction 1 - load_joint/load_pu [95% CI] | total NMSE reduction 1 - gen_joint/load_pu [95% CI] |
|---|---|---|---|---|---|---|
| -5 dB | +0.0068 [+0.0066, +0.0070] | +0.0083 [+0.0080, +0.0087] | +2.35 % [+2.28, +2.42] | +0.0583 [+0.0562, +0.0605] | +13.25 % [+12.93, +13.59] | +15.28 % [+14.96, +15.63] |
| 0 dB | +0.0017 [+0.0016, +0.0018] | +0.0023 [+0.0021, +0.0024] | +0.91 % [+0.87, +0.95] | +0.0370 [+0.0353, +0.0387] | +13.07 % [+12.66, +13.45] | +13.86 % [+13.45, +14.25] |
| 10 dB | +0.0000 [+0.0000, +0.0000] | +0.0001 [+0.0001, +0.0001] | +0.07 % [+0.06, +0.07] | +0.0100 [+0.0090, +0.0110] | +11.23 % [+10.64, +11.84] | +11.30 % [+10.70, +11.90] |

| SNR | gen_eq - load_joint CosSim [95% CI]; NMSE reduction 1 - gen_eq/load_joint [95% CI] | gen_k0 - load_joint CosSim [95% CI]; NMSE reduction 1 - gen_k0/load_joint [95% CI] | gen_pc - load_joint CosSim [95% CI]; NMSE reduction 1 - gen_pc/load_joint [95% CI] |
|---|---|---|---|
| -5 dB | -0.0788 [-0.0814, -0.0760]; -26.66 % [-27.68, -25.57] | -0.0003 [-0.0013, +0.0007]; -0.08 % [-0.40, +0.20] | -0.0066 [-0.0073, -0.0060]; -1.89 % [-2.07, -1.72] |
| 0 dB | -0.1336 [-0.1366, -0.1307]; -67.99 % [-70.02, -66.06] | -0.0000 [-0.0008, +0.0006]; +0.05 % [-0.29, +0.35] | -0.0082 [-0.0086, -0.0077]; -3.80 % [-3.98, -3.60] |
| 10 dB | -0.2249 [-0.2281, -0.2216]; -348.43 % [-359.39, -336.86] | -0.0001 [-0.0003, +0.0002]; -0.07 % [-0.46, +0.34] | -0.0030 [-0.0033, -0.0028]; -4.46 % [-4.75, -4.19] |

| SNR | load masks: uses / coefficients sent | gen_joint: uses | gen_joint overlap index t | gen_joint t_overlap_modes | modes active / pruned | partial coefficients | n_data |
|---|---|---|---|---|---|---|---|
| -5 dB | 716 / 179 | 1140 | 0.1736 | 0.1736 | 756 / 0 | 0 | 3072 |
| 0 dB | 1692 / 423 | 2176 | 0.0840 | 0.0840 | 1711 / 0 | 0 | 3072 |
| 10 dB | 3068 / 767 | 3068 | 0.0109 | 0.0109 | 3068 / 0 | 0 | 3072 |

#### COCO captions of one image, K = 4, U = 4; dense random masks (rnd) next to the load masks and Proposition 2: data/shared/shared_coco_K4_U4_rnd.json

- dataset coco, K = 4, U = 4, 500 frames, gains rayleigh, SNRs 0/10/20 dB, receivers know the gains; load masks: spec rule (768/768/768/768 coefficients per user)
- content statistics of the design C (fit): kappa_bar = 0.3935 (tr C_model 0.3935, tr C of the content 0.3935, model is the true content: True); kappa_1 = 0.7416; lam-weighted kappa per 128-group 1..6: 0.434/0.151/0.107/0.089/0.075/0.066; mean kappa_i per group: 0.307/0.146/0.106/0.088/0.075/0.065; kappa_i of coefficients 1..10: 0.742/0.154/0.326/0.687/0.630/0.445/0.779/0.596/0.425/0.476

| mode | metric | 0 dB | 10 dB | 20 dB |
|---|---|---|---|---|
| load_pu | CosSim | 0.8317 | 0.9524 | 0.9919 |
| load_pu | weakest user CosSim | 0.6870 | 0.8881 | 0.9771 |
| load_pu | NMSE | 0.2931 | 0.0894 | 0.0158 |
| load_joint | CosSim | 0.8570 | 0.9575 | 0.9923 |
| load_joint | weakest user CosSim | 0.7567 | 0.9048 | 0.9788 |
| load_joint | NMSE | 0.2593 | 0.0814 | 0.0151 |
| gen_joint | CosSim | 0.8585 | 0.9575 | 0.9923 |
| gen_joint | weakest user CosSim | 0.7585 | 0.9049 | 0.9788 |
| gen_joint | NMSE | 0.2569 | 0.0813 | 0.0151 |
| rnd | CosSim | 0.8477 | 0.9320 | 0.9758 |
| rnd | weakest user CosSim | 0.7430 | 0.8564 | 0.9398 |
| rnd | NMSE | 0.2745 | 0.1283 | 0.0472 |

| SNR | gen_joint - load_joint CosSim [95% CI] | gen_joint - load_joint weakest [95% CI] | overlap NMSE reduction 1 - gen_joint/load_joint [95% CI] | load_joint - load_pu CosSim [95% CI] | receiver NMSE reduction 1 - load_joint/load_pu [95% CI] | total NMSE reduction 1 - gen_joint/load_pu [95% CI] |
|---|---|---|---|---|---|---|
| 0 dB | +0.0015 [+0.0013, +0.0016] | +0.0018 [+0.0015, +0.0021] | +0.93 % [+0.86, +1.01] | +0.0253 [+0.0231, +0.0276] | +11.54 % [+10.79, +12.27] | +12.36 % [+11.61, +13.13] |
| 10 dB | +0.0000 [+0.0000, +0.0000] | +0.0001 [+0.0000, +0.0001] | +0.07 % [+0.06, +0.08] | +0.0050 [+0.0042, +0.0059] | +8.91 % [+7.87, +9.92] | +8.97 % [+7.93, +9.98] |
| 20 dB | +0.0000 [-0.0000, +0.0000] | +0.0000 [-0.0000, +0.0000] | +0.00 % [-0.00, +0.00] | +0.0004 [+0.0003, +0.0006] | +4.64 % [+3.32, +5.92] | +4.64 % [+3.32, +5.92] |

| SNR | rnd - load_joint CosSim [95% CI]; NMSE reduction 1 - rnd/load_joint [95% CI] |
|---|---|
| 0 dB | -0.0093 [-0.0107, -0.0079]; -5.86 % [-6.72, -5.03] |
| 10 dB | -0.0255 [-0.0263, -0.0246]; -57.58 % [-60.70, -54.31] |
| 20 dB | -0.0165 [-0.0171, -0.0160]; -212.04 % [-229.20, -197.14] |

| SNR | load masks: uses / coefficients sent | gen_joint: uses | gen_joint overlap index t | gen_joint t_overlap_modes | modes active / pruned | partial coefficients | n_data |
|---|---|---|---|---|---|---|---|
| 0 dB | 1692 / 423 | 2176 | 0.0840 | 0.0840 | 1711 / 0 | 0 | 3072 |
| 10 dB | 3068 / 767 | 3068 | 0.0109 | 0.0109 | 3068 / 0 | 0 | 3072 |
| 20 dB | 3068 / 767 | 3068 | 0.0011 | 0.0011 | 3068 / 0 | 0 | 3072 |

#### COCO captions of one image, K = 4, U = 5; overload, gen_joint = pruned Proposition 2; load masks with the rank rule of the manuscript (user j sends min(ceil((N - j)/U) 128, d_b)), load modes only: data/shared/shared_coco_K4_U5_rank.json

- dataset coco, K = 4, U = 5, 2000 frames, gains rayleigh, SNRs -10/-5/0/5/10/20 dB, receivers know the gains; load masks: rank rule (640/640/640/640/512 coefficients per user)
- content statistics of the design C (fit): kappa_bar = 0.3935 (tr C_model 0.3935, tr C of the content 0.3935, model is the true content: True); kappa_1 = 0.7416; lam-weighted kappa per 128-group 1..6: 0.434/0.151/0.107/0.089/0.075/0.066; mean kappa_i per group: 0.307/0.146/0.106/0.088/0.075/0.065; kappa_i of coefficients 1..10: 0.742/0.154/0.326/0.687/0.630/0.445/0.779/0.596/0.425/0.476

| mode | metric | -10 dB | -5 dB | 0 dB | 5 dB | 10 dB | 20 dB |
|---|---|---|---|---|---|---|---|
| load_pu | CosSim | 0.5966 | 0.7124 | 0.8148 | 0.8868 | 0.9435 | 0.9864 |
| load_pu | weakest user CosSim | 0.3613 | 0.4965 | 0.6349 | 0.7366 | 0.8525 | 0.9624 |
| load_pu | NMSE | 0.6195 | 0.4701 | 0.3191 | 0.2007 | 0.1046 | 0.0265 |
| load_joint | CosSim | 0.6724 | 0.7664 | 0.8469 | 0.9048 | 0.9501 | 0.9872 |
| load_joint | weakest user CosSim | 0.5379 | 0.6377 | 0.7328 | 0.8007 | 0.8794 | 0.9663 |
| load_joint | NMSE | 0.5411 | 0.4059 | 0.2770 | 0.1763 | 0.0949 | 0.0252 |

| SNR | load_joint - load_pu CosSim [95% CI] | receiver NMSE reduction 1 - load_joint/load_pu [95% CI] |
|---|---|---|
| -10 dB | +0.0758 [+0.0737, +0.0779] | +12.66 % [+12.38, +12.93] |
| -5 dB | +0.0539 [+0.0522, +0.0557] | +13.65 % [+13.36, +13.96] |
| 0 dB | +0.0321 [+0.0308, +0.0336] | +13.19 % [+12.84, +13.51] |
| 5 dB | +0.0180 [+0.0170, +0.0191] | +12.17 % [+11.75, +12.59] |
| 10 dB | +0.0067 [+0.0062, +0.0072] | +9.32 % [+8.88, +9.76] |
| 20 dB | +0.0008 [+0.0007, +0.0010] | +4.85 % [+4.22, +5.59] |

| SNR | load masks: uses / coefficients sent | gen_joint: uses | gen_joint overlap index t | gen_joint t_overlap_modes | modes active / pruned | partial coefficients | n_data |
|---|---|---|---|---|---|---|---|
| -10 dB | 325 / 65 | - | - | - | - / - | - | - |
| -5 dB | 760 / 152 | - | - | - | - / - | - | - |
| 0 dB | 1795 / 359 | - | - | - | - / - | - | - |
| 5 dB | 3072 / 640 | - | - | - | - / - | - | - |
| 10 dB | 3072 / 640 | - | - | - | - / - | - | - |
| 20 dB | 3072 / 640 | - | - | - | - / - | - | - |

#### Flickr30k captions of one image, K = 2, U = 4; overload, gen_joint = pruned Proposition 2: data/shared/shared_flickr_K2_U4.json

- dataset flickr, K = 2, U = 4, 1000 frames, gains rayleigh, SNRs -5/0/10 dB, receivers know the gains; load masks: spec rule (384/384/384/384 coefficients per user)
- content statistics of the design C (fit): kappa_bar = 0.3207 (tr C_model 0.3207, tr C of the content 0.3207, model is the true content: True); kappa_1 = 0.1387; lam-weighted kappa per 128-group 1..6: 0.349/0.166/0.122/0.102/0.089/0.079; mean kappa_i per group: 0.290/0.161/0.121/0.102/0.088/0.077; kappa_i of coefficients 1..10: 0.139/0.330/0.322/0.607/0.403/0.413/0.406/0.596/0.454/0.407

| mode | metric | -5 dB | 0 dB | 10 dB |
|---|---|---|---|---|
| load_pu | CosSim | 0.6686 | 0.7756 | 0.9208 |
| load_pu | weakest user CosSim | 0.4829 | 0.6155 | 0.8347 |
| load_pu | NMSE | 0.5296 | 0.3800 | 0.1448 |
| load_joint | CosSim | 0.7098 | 0.8020 | 0.9276 |
| load_joint | weakest user CosSim | 0.5801 | 0.6850 | 0.8575 |
| load_joint | NMSE | 0.4856 | 0.3479 | 0.1358 |
| gen_joint | CosSim | 0.7204 | 0.8054 | 0.9282 |
| gen_joint | weakest user CosSim | 0.5932 | 0.6896 | 0.8577 |
| gen_joint | NMSE | 0.4708 | 0.3425 | 0.1347 |
| gen_eq | CosSim | 0.6160 | 0.6501 | 0.6767 |
| gen_eq | weakest user CosSim | 0.4637 | 0.4900 | 0.5123 |
| gen_eq | NMSE | 0.6086 | 0.5640 | 0.5276 |
| gen_k0 | CosSim | 0.7111 | 0.8019 | 0.9278 |
| gen_k0 | weakest user CosSim | 0.5838 | 0.6832 | 0.8580 |
| gen_k0 | NMSE | 0.4843 | 0.3481 | 0.1354 |
| gen_pc | CosSim | 0.7102 | 0.7983 | 0.9260 |
| gen_pc | weakest user CosSim | 0.5727 | 0.6736 | 0.8512 |
| gen_pc | NMSE | 0.4838 | 0.3525 | 0.1379 |

| SNR | gen_joint - load_joint CosSim [95% CI] | gen_joint - load_joint weakest [95% CI] | overlap NMSE reduction 1 - gen_joint/load_joint [95% CI] | load_joint - load_pu CosSim [95% CI] | receiver NMSE reduction 1 - load_joint/load_pu [95% CI] | total NMSE reduction 1 - gen_joint/load_pu [95% CI] |
|---|---|---|---|---|---|---|
| -5 dB | +0.0106 [+0.0101, +0.0112] | +0.0131 [+0.0122, +0.0141] | +3.04 % [+2.88, +3.19] | +0.0412 [+0.0387, +0.0437] | +8.31 % [+7.92, +8.67] | +11.09 % [+10.62, +11.52] |
| 0 dB | +0.0035 [+0.0033, +0.0037] | +0.0047 [+0.0043, +0.0050] | +1.55 % [+1.46, +1.63] | +0.0264 [+0.0246, +0.0281] | +8.46 % [+8.07, +8.84] | +9.88 % [+9.45, +10.28] |
| 10 dB | +0.0006 [+0.0005, +0.0006] | +0.0002 [+0.0001, +0.0003] | +0.80 % [+0.75, +0.86] | +0.0068 [+0.0060, +0.0078] | +6.27 % [+5.76, +6.86] | +7.03 % [+6.52, +7.60] |

| SNR | gen_eq - load_joint CosSim [95% CI]; NMSE reduction 1 - gen_eq/load_joint [95% CI] | gen_k0 - load_joint CosSim [95% CI]; NMSE reduction 1 - gen_k0/load_joint [95% CI] | gen_pc - load_joint CosSim [95% CI]; NMSE reduction 1 - gen_pc/load_joint [95% CI] |
|---|---|---|---|
| -5 dB | -0.0937 [-0.0983, -0.0891]; -25.33 % [-26.69, -24.00] | +0.0013 [-0.0005, +0.0032]; +0.27 % [-0.14, +0.70] | +0.0004 [-0.0005, +0.0015]; +0.37 % [+0.15, +0.62] |
| 0 dB | -0.1519 [-0.1567, -0.1480]; -62.13 % [-64.52, -59.96] | -0.0001 [-0.0013, +0.0011]; -0.07 % [-0.51, +0.36] | -0.0036 [-0.0042, -0.0030]; -1.32 % [-1.52, -1.12] |
| 10 dB | -0.2509 [-0.2554, -0.2463]; -288.61 % [-298.73, -277.80] | +0.0002 [-0.0003, +0.0008]; +0.29 % [-0.21, +0.79] | -0.0016 [-0.0019, -0.0013]; -1.59 % [-1.87, -1.32] |

| SNR | load masks: uses / coefficients sent | gen_joint: uses | gen_joint overlap index t | gen_joint t_overlap_modes | modes active / pruned | partial coefficients | n_data |
|---|---|---|---|---|---|---|---|
| -5 dB | 436 / 109 | 443 | 0.2409 | 0.2409 | 443 / 0 | 0 | 1536 |
| 0 dB | 1028 / 257 | 1061 | 0.1285 | 0.1285 | 1061 / 0 | 0 | 1536 |
| 10 dB | 1536 / 384 | 1536 | 0.0346 | 0.0344 | 1536 / 1532 | 99 | 1536 |

#### Flickr30k captions of one image, K = 4, U = 4: data/shared/shared_flickr_K4_U4.json

- dataset flickr, K = 4, U = 4, 1000 frames, gains rayleigh, SNRs -5/0/10 dB, receivers know the gains; load masks: spec rule (768/768/768/768 coefficients per user)
- content statistics of the design C (fit): kappa_bar = 0.3207 (tr C_model 0.3207, tr C of the content 0.3207, model is the true content: True); kappa_1 = 0.1387; lam-weighted kappa per 128-group 1..6: 0.349/0.166/0.122/0.102/0.089/0.079; mean kappa_i per group: 0.290/0.161/0.121/0.102/0.088/0.077; kappa_i of coefficients 1..10: 0.139/0.330/0.322/0.607/0.403/0.413/0.406/0.596/0.454/0.407

| mode | metric | -5 dB | 0 dB | 10 dB |
|---|---|---|---|---|
| load_pu | CosSim | 0.7347 | 0.8303 | 0.9504 |
| load_pu | weakest user CosSim | 0.5613 | 0.6884 | 0.8797 |
| load_pu | NMSE | 0.4383 | 0.2952 | 0.0914 |
| load_joint | CosSim | 0.7674 | 0.8492 | 0.9548 |
| load_joint | weakest user CosSim | 0.6435 | 0.7415 | 0.8952 |
| load_joint | NMSE | 0.4008 | 0.2710 | 0.0853 |
| gen_joint | CosSim | 0.7730 | 0.8508 | 0.9548 |
| gen_joint | weakest user CosSim | 0.6508 | 0.7437 | 0.8953 |
| gen_joint | NMSE | 0.3927 | 0.2685 | 0.0853 |
| gen_eq | CosSim | 0.6370 | 0.6627 | 0.6788 |
| gen_eq | weakest user CosSim | 0.4809 | 0.4999 | 0.5138 |
| gen_eq | NMSE | 0.5813 | 0.5467 | 0.5245 |
| gen_k0 | CosSim | 0.7686 | 0.8493 | 0.9550 |
| gen_k0 | weakest user CosSim | 0.6476 | 0.7408 | 0.8960 |
| gen_k0 | NMSE | 0.3996 | 0.2710 | 0.0850 |
| gen_pc | CosSim | 0.7642 | 0.8454 | 0.9535 |
| gen_pc | weakest user CosSim | 0.6318 | 0.7305 | 0.8908 |
| gen_pc | NMSE | 0.4043 | 0.2763 | 0.0874 |

| SNR | gen_joint - load_joint CosSim [95% CI] | gen_joint - load_joint weakest [95% CI] | overlap NMSE reduction 1 - gen_joint/load_joint [95% CI] | load_joint - load_pu CosSim [95% CI] | receiver NMSE reduction 1 - load_joint/load_pu [95% CI] | total NMSE reduction 1 - gen_joint/load_pu [95% CI] |
|---|---|---|---|---|---|---|
| -5 dB | +0.0055 [+0.0053, +0.0058] | +0.0073 [+0.0068, +0.0079] | +2.03 % [+1.93, +2.14] | +0.0328 [+0.0306, +0.0349] | +8.56 % [+8.16, +8.96] | +10.42 % [+9.96, +10.85] |
| 0 dB | +0.0016 [+0.0015, +0.0017] | +0.0022 [+0.0021, +0.0024] | +0.94 % [+0.89, +0.99] | +0.0188 [+0.0174, +0.0203] | +8.17 % [+7.75, +8.58] | +9.04 % [+8.60, +9.46] |
| 10 dB | +0.0000 [+0.0000, +0.0000] | +0.0001 [+0.0001, +0.0001] | +0.09 % [+0.08, +0.09] | +0.0044 [+0.0037, +0.0052] | +6.61 % [+5.97, +7.32] | +6.70 % [+6.05, +7.41] |

| SNR | gen_eq - load_joint CosSim [95% CI]; NMSE reduction 1 - gen_eq/load_joint [95% CI] | gen_k0 - load_joint CosSim [95% CI]; NMSE reduction 1 - gen_k0/load_joint [95% CI] | gen_pc - load_joint CosSim [95% CI]; NMSE reduction 1 - gen_pc/load_joint [95% CI] |
|---|---|---|---|
| -5 dB | -0.1304 [-0.1352, -0.1258]; -45.03 % [-46.95, -43.12] | +0.0012 [-0.0002, +0.0026]; +0.29 % [-0.13, +0.72] | -0.0032 [-0.0039, -0.0024]; -0.87 % [-1.08, -0.65] |
| 0 dB | -0.1865 [-0.1914, -0.1826]; -101.72 % [-105.42, -98.14] | +0.0001 [-0.0009, +0.0010]; +0.02 % [-0.44, +0.45] | -0.0038 [-0.0042, -0.0033]; -1.92 % [-2.14, -1.71] |
| 10 dB | -0.2760 [-0.2805, -0.2715]; -514.70 % [-537.62, -490.76] | +0.0003 [-0.0001, +0.0007]; +0.44 % [-0.23, +1.09] | -0.0013 [-0.0016, -0.0011]; -2.39 % [-2.74, -2.04] |

| SNR | load masks: uses / coefficients sent | gen_joint: uses | gen_joint overlap index t | gen_joint t_overlap_modes | modes active / pruned | partial coefficients | n_data |
|---|---|---|---|---|---|---|---|
| -5 dB | 728 / 182 | 1176 | 0.1663 | 0.1663 | 750 / 0 | 0 | 3072 |
| 0 dB | 1712 / 428 | 2264 | 0.0856 | 0.0856 | 1733 / 0 | 0 | 3072 |
| 10 dB | 3068 / 767 | 3068 | 0.0114 | 0.0114 | 3068 / 0 | 0 | 3072 |

#### AG News, independent content (C = 0), K = 4, U = 2; dense random masks (rnd) next to the load masks and Proposition 2: data/shared/shared_agnews_K4_U2_rnd.json

- dataset agnews, K = 4, U = 2, 500 frames, gains rayleigh, SNRs 0/10/20 dB, receivers know the gains; load masks: spec rule (768/768 coefficients per user)
- content statistics of the design C (fit): kappa_bar = 0.0000 (tr C_model 0.0000, tr C of the content 0.0000, model is the true content: True); kappa_1 = 0.0000; lam-weighted kappa per 128-group 1..6: 0.000/0.000/0.000/0.000/0.000/0.000; mean kappa_i per group: 0.000/0.000/0.000/0.000/0.000/0.000; kappa_i of coefficients 1..10: 0.000/0.000/0.000/0.000/0.000/0.000/0.000/0.000/0.000/0.000

| mode | metric | 0 dB | 10 dB | 20 dB |
|---|---|---|---|---|
| load_pu | CosSim | 0.8457 | 0.9626 | 0.9941 |
| load_pu | weakest user CosSim | 0.7812 | 0.9385 | 0.9896 |
| load_pu | NMSE | 0.2667 | 0.0683 | 0.0113 |
| load_joint | CosSim | 0.8457 | 0.9626 | 0.9941 |
| load_joint | weakest user CosSim | 0.7812 | 0.9385 | 0.9896 |
| load_joint | NMSE | 0.2667 | 0.0683 | 0.0113 |
| gen_joint | CosSim | 0.8457 | 0.9626 | 0.9941 |
| gen_joint | weakest user CosSim | 0.7812 | 0.9385 | 0.9896 |
| gen_joint | NMSE | 0.2667 | 0.0683 | 0.0113 |
| rnd | CosSim | 0.8378 | 0.9539 | 0.9921 |
| rnd | weakest user CosSim | 0.7696 | 0.9241 | 0.9861 |
| rnd | NMSE | 0.2798 | 0.0833 | 0.0151 |

| SNR | gen_joint - load_joint CosSim [95% CI] | gen_joint - load_joint weakest [95% CI] | overlap NMSE reduction 1 - gen_joint/load_joint [95% CI] | load_joint - load_pu CosSim [95% CI] | receiver NMSE reduction 1 - load_joint/load_pu [95% CI] | total NMSE reduction 1 - gen_joint/load_pu [95% CI] |
|---|---|---|---|---|---|---|
| 0 dB | +0.0000 [+0.0000, +0.0000] | +0.0000 [+0.0000, +0.0000] | +0.00 % [+0.00, +0.00] | +0.0000 [+0.0000, +0.0000] | +0.00 % [+0.00, +0.00] | +0.00 % [+0.00, +0.00] |
| 10 dB | +0.0000 [+0.0000, +0.0000] | +0.0000 [+0.0000, +0.0000] | +0.00 % [+0.00, +0.00] | +0.0000 [+0.0000, +0.0000] | +0.00 % [+0.00, +0.00] | +0.00 % [+0.00, +0.00] |
| 20 dB | +0.0000 [+0.0000, +0.0000] | +0.0000 [+0.0000, +0.0000] | +0.00 % [+0.00, +0.00] | +0.0000 [+0.0000, +0.0000] | +0.00 % [+0.00, +0.00] | +0.00 % [+0.00, +0.00] |

| SNR | rnd - load_joint CosSim [95% CI]; NMSE reduction 1 - rnd/load_joint [95% CI] |
|---|---|
| 0 dB | -0.0079 [-0.0104, -0.0052]; -4.91 % [-6.05, -3.79] |
| 10 dB | -0.0087 [-0.0104, -0.0073]; -21.88 % [-24.56, -19.31] |
| 20 dB | -0.0020 [-0.0025, -0.0017]; -33.48 % [-37.80, -29.77] |

| SNR | load masks: uses / coefficients sent | gen_joint: uses | gen_joint overlap index t | gen_joint t_overlap_modes | modes active / pruned | partial coefficients | n_data |
|---|---|---|---|---|---|---|---|
| 0 dB | 1350 / 675 | 1350 | 0.0000 | 0.0000 | 1350 / 0 | 0 | 3072 |
| 10 dB | 1534 / 767 | 1534 | 0.0000 | 0.0000 | 1534 / 0 | 0 | 3072 |
| 20 dB | 1534 / 767 | 1534 | 0.0000 | 0.0000 | 1534 / 0 | 0 | 3072 |

#### AG News, independent content (C = 0), K = 4, U = 4; dense random masks (rnd) next to the load masks and Proposition 2: data/shared/shared_agnews_K4_U4_rnd.json

- dataset agnews, K = 4, U = 4, 500 frames, gains rayleigh, SNRs 0/10/20 dB, receivers know the gains; load masks: spec rule (768/768/768/768 coefficients per user)
- content statistics of the design C (fit): kappa_bar = 0.0000 (tr C_model 0.0000, tr C of the content 0.0000, model is the true content: True); kappa_1 = 0.0000; lam-weighted kappa per 128-group 1..6: 0.000/0.000/0.000/0.000/0.000/0.000; mean kappa_i per group: 0.000/0.000/0.000/0.000/0.000/0.000; kappa_i of coefficients 1..10: 0.000/0.000/0.000/0.000/0.000/0.000/0.000/0.000/0.000/0.000

| mode | metric | 0 dB | 10 dB | 20 dB |
|---|---|---|---|---|
| load_pu | CosSim | 0.7935 | 0.9408 | 0.9899 |
| load_pu | weakest user CosSim | 0.6301 | 0.8620 | 0.9713 |
| load_pu | NMSE | 0.3509 | 0.1092 | 0.0196 |
| load_joint | CosSim | 0.7935 | 0.9408 | 0.9899 |
| load_joint | weakest user CosSim | 0.6301 | 0.8620 | 0.9713 |
| load_joint | NMSE | 0.3509 | 0.1092 | 0.0196 |
| gen_joint | CosSim | 0.7935 | 0.9408 | 0.9899 |
| gen_joint | weakest user CosSim | 0.6301 | 0.8620 | 0.9713 |
| gen_joint | NMSE | 0.3509 | 0.1092 | 0.0196 |
| rnd | CosSim | 0.7766 | 0.8988 | 0.9649 |
| rnd | weakest user CosSim | 0.6004 | 0.7745 | 0.9094 |
| rnd | NMSE | 0.3759 | 0.1807 | 0.0665 |

| SNR | gen_joint - load_joint CosSim [95% CI] | gen_joint - load_joint weakest [95% CI] | overlap NMSE reduction 1 - gen_joint/load_joint [95% CI] | load_joint - load_pu CosSim [95% CI] | receiver NMSE reduction 1 - load_joint/load_pu [95% CI] | total NMSE reduction 1 - gen_joint/load_pu [95% CI] |
|---|---|---|---|---|---|---|
| 0 dB | +0.0000 [+0.0000, +0.0000] | +0.0000 [+0.0000, +0.0000] | +0.00 % [+0.00, +0.00] | +0.0000 [+0.0000, +0.0000] | +0.00 % [+0.00, +0.00] | +0.00 % [+0.00, +0.00] |
| 10 dB | +0.0000 [+0.0000, +0.0000] | +0.0000 [+0.0000, +0.0000] | +0.00 % [+0.00, +0.00] | +0.0000 [+0.0000, +0.0000] | +0.00 % [+0.00, +0.00] | +0.00 % [+0.00, +0.00] |
| 20 dB | +0.0000 [+0.0000, +0.0000] | +0.0000 [+0.0000, +0.0000] | +0.00 % [+0.00, +0.00] | +0.0000 [+0.0000, +0.0000] | +0.00 % [+0.00, +0.00] | +0.00 % [+0.00, +0.00] |

| SNR | rnd - load_joint CosSim [95% CI]; NMSE reduction 1 - rnd/load_joint [95% CI] |
|---|---|
| 0 dB | -0.0170 [-0.0195, -0.0146]; -7.13 % [-7.89, -6.38] |
| 10 dB | -0.0421 [-0.0435, -0.0407]; -65.48 % [-68.71, -62.25] |
| 20 dB | -0.0250 [-0.0260, -0.0241]; -239.29 % [-258.94, -221.60] |

| SNR | load masks: uses / coefficients sent | gen_joint: uses | gen_joint overlap index t | gen_joint t_overlap_modes | modes active / pruned | partial coefficients | n_data |
|---|---|---|---|---|---|---|---|
| 0 dB | 1824 / 456 | 1824 | -0.0000 | 0.0000 | 1824 / 0 | 0 | 3072 |
| 10 dB | 3068 / 767 | 3068 | -0.0000 | 0.0000 | 3068 / 0 | 0 | 3072 |
| 20 dB | 3068 / 767 | 3068 | -0.0000 | 0.0000 | 3068 / 0 | 0 | 3072 |

- mismatched-design runs (design C differs from the true content) with an overlap NMSE reduction CI entirely below 0: shared/shared_coco_K2_U4_ctrlfit.json -10 dB -2.98 % [-3.13, -2.83]; shared/shared_coco_K2_U4_ctrlfit.json -5 dB -1.76 % [-1.85, -1.67]; shared/shared_coco_K2_U4_ctrlfit.json 0 dB -0.89 % [-0.93, -0.84]; shared/shared_coco_K2_U4_ctrlfit.json 5 dB -0.38 % [-0.40, -0.35]; shared/shared_coco_K2_U4_ctrlfit.json 10 dB -0.12 % [-0.15, -0.10]; shared/shared_coco_K2_U4_ctrlfit.json 20 dB -0.42 % [-0.48, -0.36]; shared/shared_coco_K4_U4_ctrlfit.json -10 dB -2.31 % [-2.43, -2.19]; shared/shared_coco_K4_U4_ctrlfit.json -5 dB -1.12 % [-1.18, -1.06]; shared/shared_coco_K4_U4_ctrlfit.json 0 dB -0.65 % [-0.68, -0.62]; shared/shared_coco_K4_U4_ctrlfit.json 5 dB -0.30 % [-0.31, -0.29]; shared/shared_coco_K4_U4_ctrlfit.json 10 dB -0.13 % [-0.13, -0.12]; shared/shared_coco_K4_U4_ctrlfit.json 20 dB -0.01 % [-0.01, -0.01]

### B.4 Content statistics (data/shared/content_stats.json, Code/content_stats.py)

- coco, five captions of one fit image (8000 images): kappa_bar = 0.3935 [0.3910, 0.3960] (bootstrap over fit images, 2000 resamples); test images (2000): 0.3937 [0.3888, 0.3989]; kappa_1 = 0.7416
  - lam-weighted kappa per 128-group 1..6: 0.434/0.151/0.107/0.089/0.075/0.066; mean kappa_i per group: 0.307/0.146/0.106/0.088/0.075/0.065; kappa_i of coefficients 1..10: 0.742/0.154/0.326/0.687/0.630/0.445/0.779/0.596/0.425/0.476
  - share of tr C in group 1: 0.958; share of ||C||_F^2 in block 1: 0.9985; off-diagonal energy share of C: 0.176 (noise corrected 0.175); spectrum participation ratio 41.0 of 768, energy in group 1 0.868
  - control, captions of five different images: kappa_bar = 0.0029 [0.0024, 0.0043]; ||C_ctrl||_F / ||C_same||_F = 0.031
- flickr, five captions of one fit image (8000 images): kappa_bar = 0.3207 [0.3181, 0.3232] (bootstrap over fit images, 2000 resamples); test images (1000): 0.3201 [0.3129, 0.3273]; kappa_1 = 0.1387
  - lam-weighted kappa per 128-group 1..6: 0.349/0.166/0.122/0.102/0.089/0.079; mean kappa_i per group: 0.290/0.161/0.121/0.102/0.088/0.077; kappa_i of coefficients 1..10: 0.139/0.330/0.322/0.607/0.403/0.413/0.406/0.596/0.454/0.407
  - share of tr C in group 1: 0.941; share of ||C||_F^2 in block 1: 0.9971; off-diagonal energy share of C: 0.326 (noise corrected 0.325); spectrum participation ratio 42.4 of 768, energy in group 1 0.863
  - control, captions of five different images: kappa_bar = 0.0016 [0.0015, 0.0030]; ||C_ctrl||_F / ||C_same||_F = 0.044
- caption mixtures (coco, U = 4 users per tuple, fit spec): kappa_bar of the fit [bootstrap 95% CI]; lam-weighted kappa groups 1..3; ||C - kappa Lam||_F / ||kappa Lam||_F
  - kappa 0.0: kappa_bar = 0.0023 [0.0020, 0.0041]; groups 0.002/0.002/0.002; rel. deviation -
  - kappa 0.2: kappa_bar = 0.1960 [0.1945, 0.1975]; groups 0.196/0.200/0.197; rel. deviation 0.1406
  - kappa 0.4: kappa_bar = 0.3965 [0.3952, 0.3977]; groups 0.397/0.398/0.394; rel. deviation 0.0672
  - kappa 0.6: kappa_bar = 0.5995 [0.5987, 0.6004]; groups 0.601/0.597/0.592; rel. deviation 0.0368
  - kappa 0.8: kappa_bar = 0.8018 [0.8014, 0.8022]; groups 0.803/0.798/0.794; rel. deviation 0.0201
  - kappa 0.9: kappa_bar = 0.9016 [0.9014, 0.9018]; groups 0.903/0.899/0.896; rel. deviation 0.0112
  - kappa 0.95: kappa_bar = 0.9510 [0.9509, 0.9511]; groups 0.952/0.949/0.948; rel. deviation 0.0059

## C. Observers of one scene from light load through overload, K = 4 (Fig. 3(a))

Overlapping masks of Proposition 2 (gen_joint) against the load masks (load_joint), both with the joint LMMSE receiver and KNOWN gains; receiver gain = load_pu -> load_joint (load masks, per-user Wiener -> joint LMMSE). Full load U = K = 4 (d_s = 3072 uses); U > 4 is overload. Model = Gaussian model with the fitted statistics (results.full.rayleigh, exact NMSE, mean over 100 Rayleigh draws; SE = paired Monte Carlo standard error over the draws). Real captions = caption mixtures with mean content correlation kappa_bar, 2000 frames, paired bootstrap 95% CIs.

- overlapping masks at U > K: gen_joint is the PRUNED Proposition 2 (greedy pruning to at most d_s active modes, partial coefficients; prop2_pruned of section E), labelled 'overload, pruned Prop. 2' in every row below. Pruning acts only where more than d_s modes would be active (pruned modes > 0 in C.1c and C.2c; the SNRs are listed below); at lower SNR the design equals plain Proposition 2.
- load masks = equal split min(floor(d_s/U), d_b) per user (--load_rule spec); at U = 5, 7 (K = 4) this differs from the rank split of the Proposed of Fig. 2 (U = 5: spec 614/614/614/614/614, rank 640/640/640/640/512; U = 7: spec 438 each, rank 512/512/512/384/384/384/384). So at U = 5 and 7 the Fig. 3(a) reference is NOT 'the proposal of Fig. 2'; it coincides with it at U = 2, 3, 4, 6, 8.
- rank against spec split, natural COCO captions, K = 4, U = 5 (B.3 shared_coco_K4_U5_rank against shared_coco_K4_U5, same frames), load_joint NMSE rank / spec: -10 dB 0.5411 / 0.5411 (1 - rank/spec +0.0000 % [+0.0000, +0.0000]); -5 dB 0.4059 / 0.4059 (1 - rank/spec +0.0000 % [+0.0000, +0.0000]); 0 dB 0.2770 / 0.2770 (1 - rank/spec +0.0000 % [+0.0000, +0.0000]); 5 dB 0.1763 / 0.1762 (1 - rank/spec -0.0225 % [-0.0372, -0.0081]); 10 dB 0.0949 / 0.0947 (1 - rank/spec -0.17 % [-0.22, -0.13]); 20 dB 0.0252 / 0.0249 (1 - rank/spec -1.15 % [-1.31, -1.00])
- SNRs at which the pruning removes modes (U > K): model kappa 0.4, U = 5: 10/20 dB; model kappa 0.4, U = 6: 10/20 dB; model kappa 0.4, U = 7: 10/20 dB; model kappa 0.4, U = 8: 10/20 dB; model kappa 0.8, U = 5: 10/20 dB; model kappa 0.8, U = 6: 10/20 dB; model kappa 0.8, U = 7: 10/20 dB; model kappa 0.8, U = 8: 10/20 dB; captions kappa 0, U = 8: 10/20 dB; captions kappa 0.8, U = 5: 10/20 dB; captions kappa 0.8, U = 6: 10/20 dB; captions kappa 0.8, U = 7: 10/20 dB; captions kappa 0.8, U = 8: 10/20 dB

### C.0 Fig. 3(a) at a glance: overlap NMSE reduction 1 - gen_joint/load_joint (%)

| U | 0 dB model kb 0.8 | 0 dB model kb 0.4 | 0 dB captions kb 0.8 [CI] | 10 dB model kb 0.8 | 10 dB model kb 0.4 | 10 dB captions kb 0.8 [CI] | 20 dB model kb 0.8 | 20 dB model kb 0.4 | 20 dB captions kb 0.8 [CI] |
|---|---|---|---|---|---|---|---|---|---|
| U = 2 (light) | 10.78 | 1.85 | +11.78 % [+11.58, +12.00] | 2.94 | 0.28 | +2.98 % [+2.89, +3.06] | 0.19 | 0.0158 | +0.19 % [+0.18, +0.20] |
| U = 3 (light) | 16.33 | 3.18 | +18.39 % [+18.16, +18.61] | 5.58 | 0.74 | +5.25 % [+5.15, +5.34] | 0.46 | 0.0473 | +0.39 % [+0.37, +0.40] |
| U = 4 (full) | 20.71 | 4.31 | +22.98 % [+22.74, +23.21] | 6.97 | 1.04 | +7.01 % [+6.90, +7.10] | 0.68 | 0.09 | +0.61 % [+0.59, +0.62] |
| U = 5 (overload, pruned Prop. 2) | 24.31 | 5.32 | +26.22 % [+25.97, +26.49] | 12.39 | 2.88 | +15.07 % [+14.89, +15.26] | 22.21 | 7.26 | +31.04 % [+30.34, +31.75] |
| U = 6 (overload, pruned Prop. 2) | 26.71 | 6.10 | +28.67 % [+28.43, +28.92] | 17.87 | 4.74 | +21.78 % [+21.50, +22.07] | 35.37 | 11.83 | +44.03 % [+43.31, +44.70] |
| U = 7 (overload, pruned Prop. 2) | 29.12 | 6.93 | +30.93 % [+30.66, +31.22] | 23.15 | 6.57 | +26.92 % [+26.59, +27.27] | 44.15 | 14.90 | +50.89 % [+50.25, +51.55] |
| U = 8 (overload, pruned Prop. 2) | 31.26 | 7.70 | +32.89 % [+32.62, +33.18] | 27.81 | 8.31 | +31.05 % [+30.67, +31.41] | 50.93 | 17.85 | +55.18 % [+54.64, +55.75] |

(values below 0.05 % print with four decimals, so a small nonzero reduction never reads as 0.00)

- captions with mixture kappa 0 (captions of different images; the design and the receiver use the FITTED C, kappa_bar 0.0023, not C = 0), U = 8: -5 dB -0.0004 % [-0.0009, -0.0000]; 0 dB -0.0006 % [-0.0009, -0.0002]; 10 dB +0.0019 % [-0.0001, +0.0040]; 20 dB +0.0101 % [+0.0067, +0.0143]; this is the same kind of point as the mixture kappa 0 rows of C.3 (fitted C, not C = 0)

- summary lines (overlap NMSE reduction 1 - gen_joint/load_joint, K = 4, known gains):
  - model kappa_bar 0.8, 0 dB: 10.78 / 16.33 / 20.71 / 24.31 / 26.71 / 29.12 / 31.26 % at U = 2 / 3 / 4 / 5 / 6 / 7 / 8; receiver reduction (load_pu -> load_joint) 23.36 / 33.41 / 36.08 / 39.08 / 40.60 / 41.37 / 41.20 %
  - model kappa_bar 0.4, 0 dB: 1.85 / 3.18 / 4.31 / 5.32 / 6.10 / 6.93 / 7.70 % at U = 2 / 3 / 4 / 5 / 6 / 7 / 8; receiver reduction (load_pu -> load_joint) 5.35 / 8.25 / 9.58 / 11.03 / 11.85 / 12.45 / 12.60 %
  - captions kappa_bar 0.8, 0 dB: 11.78 / 18.39 / 22.98 / 26.22 / 28.67 / 30.93 / 32.89 % at U = 2 / 3 / 4 / 5 / 6 / 7 / 8; receiver reduction 23.09 / 30.92 / 34.62 / 36.99 / 38.50 / 39.59 / 40.17 %; CosSim gen_joint - load_joint +0.0113 / +0.0187 / +0.0248 / +0.0297 / +0.0343 / +0.0381 / +0.0420
  - model kappa_bar 0.8, 10 dB: 2.94 / 5.58 / 6.97 / 12.39 / 17.87 / 23.15 / 27.81 % at U = 2 / 3 / 4 / 5 / 6 / 7 / 8; receiver reduction (load_pu -> load_joint) 27.63 / 33.73 / 39.87 / 40.84 / 39.55 / 37.97 / 34.96 %
  - model kappa_bar 0.4, 10 dB: 0.28 / 0.74 / 1.04 / 2.88 / 4.74 / 6.57 / 8.31 % at U = 2 / 3 / 4 / 5 / 6 / 7 / 8; receiver reduction (load_pu -> load_joint) 6.48 / 7.92 / 10.44 / 11.48 / 11.30 / 11.24 / 10.27 %
  - captions kappa_bar 0.8, 10 dB: 2.98 / 5.25 / 7.01 / 15.07 / 21.78 / 26.92 / 31.05 % at U = 2 / 3 / 4 / 5 / 6 / 7 / 8; receiver reduction 24.25 / 31.89 / 36.63 / 36.14 / 35.81 / 34.18 / 33.57 %; CosSim gen_joint - load_joint +0.0007 / +0.0014 / +0.0021 / +0.0053 / +0.0088 / +0.0123 / +0.0158
  - model kappa_bar 0.8, 20 dB: 0.19 / 0.46 / 0.68 / 22.21 / 35.37 / 44.15 / 50.93 % at U = 2 / 3 / 4 / 5 / 6 / 7 / 8; receiver reduction (load_pu -> load_joint) 32.11 / 21.07 / 31.94 / 27.86 / 21.25 / 18.86 / 12.18 %
  - model kappa_bar 0.4, 20 dB: 0.02 / 0.05 / 0.09 / 7.26 / 11.83 / 14.90 / 17.85 % at U = 2 / 3 / 4 / 5 / 6 / 7 / 8; receiver reduction (load_pu -> load_joint) 8.06 / 4.35 / 7.76 / 7.76 / 6.17 / 5.80 / 3.68 %
  - captions kappa_bar 0.8, 20 dB: 0.19 / 0.39 / 0.61 / 31.04 / 44.03 / 50.89 / 55.18 % at U = 2 / 3 / 4 / 5 / 6 / 7 / 8; receiver reduction 17.98 / 25.82 / 26.88 / 19.07 / 14.36 / 12.69 / 11.28 %; CosSim gen_joint - load_joint +0.0000 / +0.0000 / +0.0000 / +0.0034 / +0.0069 / +0.0106 / +0.0140

### C.1 Model (Gaussian, fitted statistics, 100 Rayleigh draws, known gains), K = 4


#### mixture kappa 0.8

| U | SNR | kappa_bar (fit) | NMSE load_pu | NMSE load_joint | NMSE gen_joint | overlap reduction % (SE) | receiver reduction % | total reduction 1 - gen_joint/load_pu % | CosSim proxy load_pu / load_joint / gen_joint |
|---|---|---|---|---|---|---|---|---|---|
| U = 2 | -10 dB | 0.8016 | 0.51503 | 0.43893 | 0.38713 | 11.80 (0.30) | 14.78 | 24.83 | 0.6780 / 0.7437 / 0.7779 |
| U = 2 | -5 dB | 0.8016 | 0.36350 | 0.29443 | 0.26029 | 11.60 (0.28) | 19.00 | 28.40 | 0.7832 / 0.8364 / 0.8569 |
| U = 2 | 0 dB | 0.8016 | 0.23073 | 0.17683 | 0.15777 | 10.78 (0.45) | 23.36 | 31.62 | 0.8666 / 0.9052 / 0.9159 |
| U = 2 | 10 dB | 0.8016 | 0.06376 | 0.04614 | 0.04478 | 2.94 (0.32) | 27.63 | 29.76 | 0.9638 / 0.9761 / 0.9769 |
| U = 2 | 20 dB | 0.8016 | 0.01470 | 0.00998 | 0.00996 | 0.19 (0.04) | 32.11 | 32.23 | 0.9912 / 0.9949 / 0.9949 |
| U = 3 | -10 dB | 0.8019 | 0.58505 | 0.45614 | 0.36647 | 19.66 (0.23) | 22.03 | 37.36 | 0.6208 / 0.7341 / 0.7932 |
| U = 3 | -5 dB | 0.8019 | 0.43598 | 0.31380 | 0.25480 | 18.80 (0.21) | 28.02 | 41.56 | 0.7329 / 0.8260 / 0.8614 |
| U = 3 | 0 dB | 0.8019 | 0.29541 | 0.19672 | 0.16459 | 16.33 (0.44) | 33.41 | 44.29 | 0.8274 / 0.8946 / 0.9128 |
| U = 3 | 10 dB | 0.8019 | 0.09006 | 0.05968 | 0.05635 | 5.58 (0.42) | 33.73 | 37.43 | 0.9514 / 0.9692 / 0.9710 |
| U = 3 | 20 dB | 0.8019 | 0.01574 | 0.01243 | 0.01237 | 0.46 (0.06) | 21.07 | 21.44 | 0.9919 / 0.9937 / 0.9937 |
| U = 4 | -10 dB | 0.8018 | 0.60804 | 0.46240 | 0.34150 | 26.15 (0.29) | 23.95 | 43.84 | 0.6022 / 0.7303 / 0.8092 |
| U = 4 | -5 dB | 0.8018 | 0.45765 | 0.31972 | 0.24087 | 24.66 (0.23) | 30.14 | 47.37 | 0.7162 / 0.8227 / 0.8697 |
| U = 4 | 0 dB | 0.8018 | 0.31499 | 0.20135 | 0.15965 | 20.71 (0.46) | 36.08 | 49.31 | 0.8124 / 0.8922 / 0.9156 |
| U = 4 | 5 dB | 0.8018 | 0.19505 | 0.11619 | 0.10026 | 13.71 (0.74) | 40.43 | 48.60 | 0.8873 / 0.9392 / 0.9478 |
| U = 4 | 10 dB | 0.8018 | 0.10614 | 0.06382 | 0.05937 | 6.97 (0.56) | 39.87 | 44.06 | 0.9404 / 0.9670 / 0.9694 |
| U = 4 | 15 dB | 0.8018 | 0.05170 | 0.03274 | 0.03199 | 2.30 (0.23) | 36.68 | 38.13 | 0.9718 / 0.9832 / 0.9836 |
| U = 4 | 20 dB | 0.8018 | 0.02304 | 0.01568 | 0.01558 | 0.68 (0.09) | 31.94 | 32.40 | 0.9878 / 0.9920 / 0.9921 |
| U = 5 (overload, pruned Prop. 2) | -10 dB | 0.8018 | 0.63855 | 0.47109 | 0.32346 | 31.34 (0.19) | 26.23 | 49.35 | 0.5777 / 0.7260 / 0.8217 |
| U = 5 (overload, pruned Prop. 2) | -5 dB | 0.8018 | 0.48835 | 0.32779 | 0.23155 | 29.36 (0.20) | 32.88 | 52.59 | 0.6950 / 0.8188 / 0.8759 |
| U = 5 (overload, pruned Prop. 2) | 0 dB | 0.8018 | 0.34176 | 0.20822 | 0.15760 | 24.31 (0.36) | 39.08 | 53.88 | 0.7956 / 0.8890 / 0.9172 |
| U = 5 (overload, pruned Prop. 2) | 10 dB | 0.8018 | 0.12018 | 0.07110 | 0.06229 | 12.39 (0.26) | 40.84 | 48.17 | 0.9323 / 0.9634 / 0.9680 |
| U = 5 (overload, pruned Prop. 2) | 20 dB | 0.8018 | 0.03178 | 0.02292 | 0.01783 | 22.21 (0.04) | 27.86 | 43.88 | 0.9831 / 0.9883 / 0.9909 |
| U = 6 (overload, pruned Prop. 2) | -10 dB | 0.8015 | 0.66166 | 0.48041 | 0.31204 | 35.05 (0.17) | 27.39 | 52.84 | 0.5574 / 0.7196 / 0.8287 |
| U = 6 (overload, pruned Prop. 2) | -5 dB | 0.8015 | 0.51354 | 0.33749 | 0.22742 | 32.62 (0.18) | 34.28 | 55.72 | 0.6765 / 0.8130 / 0.8784 |
| U = 6 (overload, pruned Prop. 2) | 0 dB | 0.8015 | 0.36546 | 0.21706 | 0.15908 | 26.71 (0.33) | 40.60 | 56.47 | 0.7805 / 0.8840 / 0.9165 |
| U = 6 (overload, pruned Prop. 2) | 10 dB | 0.8015 | 0.13368 | 0.08081 | 0.06637 | 17.87 (0.20) | 39.55 | 50.35 | 0.9254 / 0.9583 / 0.9659 |
| U = 6 (overload, pruned Prop. 2) | 20 dB | 0.8015 | 0.03956 | 0.03115 | 0.02013 | 35.37 (0.04) | 21.25 | 49.11 | 0.9791 / 0.9842 / 0.9898 |
| U = 7 (overload, pruned Prop. 2) | -10 dB | 0.8019 | 0.67368 | 0.48389 | 0.29686 | 38.65 (0.15) | 28.17 | 55.93 | 0.5478 / 0.7176 / 0.8381 |
| U = 7 (overload, pruned Prop. 2) | -5 dB | 0.8019 | 0.52569 | 0.34159 | 0.21931 | 35.80 (0.16) | 35.02 | 58.28 | 0.6683 / 0.8107 / 0.8831 |
| U = 7 (overload, pruned Prop. 2) | 0 dB | 0.8019 | 0.37592 | 0.22042 | 0.15623 | 29.12 (0.29) | 41.37 | 58.44 | 0.7742 / 0.8823 / 0.9181 |
| U = 7 (overload, pruned Prop. 2) | 10 dB | 0.8019 | 0.14076 | 0.08730 | 0.06709 | 23.15 (0.15) | 37.97 | 52.34 | 0.9213 / 0.9550 / 0.9655 |
| U = 7 (overload, pruned Prop. 2) | 20 dB | 0.8019 | 0.04804 | 0.03898 | 0.02177 | 44.15 (0.03) | 18.86 | 54.69 | 0.9748 / 0.9802 / 0.9889 |
| U = 8 (overload, pruned Prop. 2) | -10 dB | 0.8018 | 0.68324 | 0.49088 | 0.28676 | 41.58 (0.14) | 28.15 | 58.03 | 0.5413 / 0.7128 / 0.8442 |
| U = 8 (overload, pruned Prop. 2) | -5 dB | 0.8018 | 0.53466 | 0.34785 | 0.21437 | 38.37 (0.17) | 34.94 | 59.91 | 0.6636 / 0.8069 / 0.8860 |
| U = 8 (overload, pruned Prop. 2) | 0 dB | 0.8018 | 0.38278 | 0.22509 | 0.15474 | 31.26 (0.29) | 41.20 | 59.58 | 0.7716 / 0.8797 / 0.9190 |
| U = 8 (overload, pruned Prop. 2) | 10 dB | 0.8018 | 0.14351 | 0.09334 | 0.06738 | 27.81 (0.12) | 34.96 | 53.04 | 0.9213 / 0.9518 / 0.9654 |
| U = 8 (overload, pruned Prop. 2) | 20 dB | 0.8018 | 0.05184 | 0.04553 | 0.02234 | 50.93 (0.03) | 12.18 | 56.91 | 0.9734 / 0.9769 / 0.9887 |

#### mixture kappa 0.4

| U | SNR | kappa_bar (fit) | NMSE load_pu | NMSE load_joint | NMSE gen_joint | overlap reduction % (SE) | receiver reduction % | total reduction 1 - gen_joint/load_pu % | CosSim proxy load_pu / load_joint / gen_joint |
|---|---|---|---|---|---|---|---|---|---|
| U = 2 | -10 dB | 0.3961 | 0.51692 | 0.49998 | 0.48848 | 2.30 (0.05) | 3.28 | 5.50 | 0.6766 / 0.6947 / 0.7032 |
| U = 2 | -5 dB | 0.3961 | 0.36513 | 0.34964 | 0.34224 | 2.11 (0.04) | 4.24 | 6.27 | 0.7821 / 0.7969 / 0.8018 |
| U = 2 | 0 dB | 0.3961 | 0.23196 | 0.21956 | 0.21551 | 1.85 (0.08) | 5.35 | 7.09 | 0.8659 / 0.8769 / 0.8794 |
| U = 2 | 10 dB | 0.3961 | 0.06413 | 0.05997 | 0.05980 | 0.28 (0.04) | 6.48 | 6.74 | 0.9636 / 0.9675 / 0.9676 |
| U = 2 | 20 dB | 0.3961 | 0.01478 | 0.01359 | 0.01358 | 0.02 (0.00) | 8.06 | 8.07 | 0.9912 / 0.9924 / 0.9924 |
| U = 3 | -10 dB | 0.3970 | 0.58720 | 0.55513 | 0.53263 | 4.05 (0.05) | 5.46 | 9.29 | 0.6190 / 0.6539 / 0.6717 |
| U = 3 | -5 dB | 0.3970 | 0.43800 | 0.40762 | 0.39273 | 3.65 (0.04) | 6.94 | 10.34 | 0.7315 / 0.7592 / 0.7694 |
| U = 3 | 0 dB | 0.3970 | 0.29714 | 0.27261 | 0.26393 | 3.18 (0.08) | 8.25 | 11.17 | 0.8263 / 0.8456 / 0.8511 |
| U = 3 | 10 dB | 0.3970 | 0.09075 | 0.08356 | 0.08295 | 0.74 (0.06) | 7.92 | 8.60 | 0.9510 / 0.9556 / 0.9559 |
| U = 3 | 20 dB | 0.3970 | 0.01588 | 0.01519 | 0.01518 | 0.05 (0.01) | 4.35 | 4.40 | 0.9918 / 0.9922 / 0.9922 |
| U = 4 | -10 dB | 0.3965 | 0.61019 | 0.57172 | 0.53932 | 5.67 (0.08) | 6.30 | 11.61 | 0.6004 / 0.6429 / 0.6684 |
| U = 4 | -5 dB | 0.3965 | 0.45972 | 0.42312 | 0.40159 | 5.09 (0.05) | 7.96 | 12.64 | 0.7148 / 0.7496 / 0.7644 |
| U = 4 | 0 dB | 0.3965 | 0.31679 | 0.28642 | 0.27407 | 4.31 (0.08) | 9.58 | 13.48 | 0.8113 / 0.8371 / 0.8448 |
| U = 4 | 5 dB | 0.3965 | 0.19639 | 0.17527 | 0.17025 | 2.86 (0.16) | 10.76 | 13.31 | 0.8865 / 0.9029 / 0.9060 |
| U = 4 | 10 dB | 0.3965 | 0.10691 | 0.09574 | 0.09475 | 1.04 (0.09) | 10.44 | 11.38 | 0.9399 / 0.9481 / 0.9486 |
| U = 4 | 15 dB | 0.3965 | 0.05210 | 0.04724 | 0.04709 | 0.31 (0.04) | 9.34 | 9.63 | 0.9715 / 0.9748 / 0.9749 |
| U = 4 | 20 dB | 0.3965 | 0.02323 | 0.02143 | 0.02141 | 0.09 (0.02) | 7.76 | 7.85 | 0.9877 / 0.9888 / 0.9888 |
| U = 5 (overload, pruned Prop. 2) | -10 dB | 0.3964 | 0.64072 | 0.59376 | 0.55190 | 7.05 (0.06) | 7.33 | 13.86 | 0.5759 / 0.6282 / 0.6618 |
| U = 5 (overload, pruned Prop. 2) | -5 dB | 0.3964 | 0.49053 | 0.44523 | 0.41719 | 6.30 (0.04) | 9.24 | 14.95 | 0.6934 / 0.7366 / 0.7561 |
| U = 5 (overload, pruned Prop. 2) | 0 dB | 0.3964 | 0.34372 | 0.30580 | 0.28953 | 5.32 (0.07) | 11.03 | 15.77 | 0.7943 / 0.8265 / 0.8368 |
| U = 5 (overload, pruned Prop. 2) | 10 dB | 0.3964 | 0.12120 | 0.10728 | 0.10419 | 2.88 (0.05) | 11.48 | 14.03 | 0.9317 / 0.9420 / 0.9437 |
| U = 5 (overload, pruned Prop. 2) | 20 dB | 0.3964 | 0.03229 | 0.02978 | 0.02762 | 7.26 (0.04) | 7.76 | 14.45 | 0.9829 / 0.9845 / 0.9856 |
| U = 6 (overload, pruned Prop. 2) | -10 dB | 0.3956 | 0.66397 | 0.61128 | 0.56110 | 8.21 (0.06) | 7.94 | 15.49 | 0.5554 / 0.6145 / 0.6553 |
| U = 6 (overload, pruned Prop. 2) | -5 dB | 0.3956 | 0.51589 | 0.46436 | 0.43058 | 7.27 (0.04) | 9.99 | 16.54 | 0.6747 / 0.7237 / 0.7475 |
| U = 6 (overload, pruned Prop. 2) | 0 dB | 0.3956 | 0.36761 | 0.32406 | 0.30428 | 6.10 (0.06) | 11.85 | 17.23 | 0.7791 / 0.8155 / 0.8280 |
| U = 6 (overload, pruned Prop. 2) | 10 dB | 0.3956 | 0.13494 | 0.11969 | 0.11403 | 4.74 (0.04) | 11.30 | 15.50 | 0.9247 / 0.9356 / 0.9387 |
| U = 6 (overload, pruned Prop. 2) | 20 dB | 0.3956 | 0.04034 | 0.03785 | 0.03338 | 11.83 (0.06) | 6.17 | 17.27 | 0.9787 / 0.9804 / 0.9827 |
| U = 7 (overload, pruned Prop. 2) | -10 dB | 0.3968 | 0.67583 | 0.61877 | 0.56063 | 9.40 (0.05) | 8.44 | 17.05 | 0.5459 / 0.6096 / 0.6569 |
| U = 7 (overload, pruned Prop. 2) | -5 dB | 0.3968 | 0.52797 | 0.47230 | 0.43299 | 8.32 (0.04) | 10.54 | 17.99 | 0.6666 / 0.7191 / 0.7468 |
| U = 7 (overload, pruned Prop. 2) | 0 dB | 0.3968 | 0.37805 | 0.33099 | 0.30804 | 6.93 (0.06) | 12.45 | 18.52 | 0.7727 / 0.8118 / 0.8264 |
| U = 7 (overload, pruned Prop. 2) | 10 dB | 0.3968 | 0.14214 | 0.12616 | 0.11787 | 6.57 (0.04) | 11.24 | 17.07 | 0.9205 / 0.9322 / 0.9367 |
| U = 7 (overload, pruned Prop. 2) | 20 dB | 0.3968 | 0.04905 | 0.04620 | 0.03932 | 14.90 (0.06) | 5.80 | 19.84 | 0.9743 / 0.9761 / 0.9796 |
| U = 8 (overload, pruned Prop. 2) | -10 dB | 0.3966 | 0.68554 | 0.62641 | 0.56094 | 10.45 (0.05) | 8.62 | 18.18 | 0.5392 / 0.6042 / 0.6575 |
| U = 8 (overload, pruned Prop. 2) | -5 dB | 0.3966 | 0.53709 | 0.47945 | 0.43509 | 9.25 (0.04) | 10.73 | 18.99 | 0.6617 / 0.7149 / 0.7461 |
| U = 8 (overload, pruned Prop. 2) | 0 dB | 0.3966 | 0.38505 | 0.33653 | 0.31060 | 7.70 (0.07) | 12.60 | 19.34 | 0.7700 / 0.8090 / 0.8254 |
| U = 8 (overload, pruned Prop. 2) | 10 dB | 0.3966 | 0.14504 | 0.13015 | 0.11933 | 8.31 (0.04) | 10.27 | 17.72 | 0.9205 / 0.9306 / 0.9364 |
| U = 8 (overload, pruned Prop. 2) | 20 dB | 0.3966 | 0.05302 | 0.05106 | 0.04195 | 17.85 (0.05) | 3.68 | 20.88 | 0.9728 / 0.9739 / 0.9786 |

#### C.1b A1 approximation gap (diagonal surrogate C = diag(kappa_i lam_i) against the full C) and the decomposition of the full model (diag.decomp_rel.full_rayleigh)

| kappa, U | SNR | overlap red. full % | overlap red. A1 % | A1 - full (pp) | rel. NMSE gap gen_joint A1/full - 1 | rel. NMSE gap load_joint A1/full - 1 | b_over_a (receiver) | d_over_b (pruned kappa = 0 design) | c_over_d (overlap) |
|---|---|---|---|---|---|---|---|---|---|
| 0.8, U = 2 | -10 dB | 11.801 | 11.818 | +0.017 | +2.38e-04 | +4.31e-04 | -0.1478 | -1.1e-16 | -0.1180 |
| 0.8, U = 2 | -5 dB | 11.598 | 11.638 | +0.041 | +7.92e-04 | +1.25e-03 | -0.1900 | -1.7e-15 | -0.1160 |
| 0.8, U = 2 | 0 dB | 10.776 | 10.878 | +0.102 | +2.31e-03 | +3.45e-03 | -0.2336 | +1.1e-15 | -0.1078 |
| 0.8, U = 2 | 10 dB | 2.939 | 2.964 | +0.025 | +1.05e-02 | +1.08e-02 | -0.2763 | -1.0e-15 | -0.0294 |
| 0.8, U = 2 | 20 dB | 0.185 | 0.183 | -0.002 | +1.29e-02 | +1.29e-02 | -0.3211 | -6.7e-16 | -0.0019 |
| 0.8, U = 3 | -10 dB | 19.658 | 19.661 | +0.003 | +2.48e-04 | +2.87e-04 | -0.2203 | -1.1e-16 | -0.1966 |
| 0.8, U = 3 | -5 dB | 18.802 | 18.815 | +0.013 | +6.73e-04 | +8.35e-04 | -0.2802 | -3.3e-16 | -0.1880 |
| 0.8, U = 3 | 0 dB | 16.335 | 16.388 | +0.054 | +1.67e-03 | +2.31e-03 | -0.3341 | +4.4e-16 | -0.1633 |
| 0.8, U = 3 | 10 dB | 5.581 | 5.643 | +0.062 | +7.31e-03 | +7.97e-03 | -0.3373 | +2.2e-16 | -0.0558 |
| 0.8, U = 3 | 20 dB | 0.461 | 0.458 | -0.003 | +9.18e-03 | +9.15e-03 | -0.2107 | +0.0e+00 | -0.0046 |
| 0.8, U = 4 | -10 dB | 26.147 | 26.147 | -0.000 | +2.39e-04 | +2.32e-04 | -0.2395 | -4.4e-16 | -0.2615 |
| 0.8, U = 4 | -5 dB | 24.664 | 24.670 | +0.007 | +5.59e-04 | +6.50e-04 | -0.3014 | +2.2e-16 | -0.2466 |
| 0.8, U = 4 | 0 dB | 20.710 | 20.746 | +0.036 | +1.28e-03 | +1.74e-03 | -0.3608 | +8.9e-16 | -0.2071 |
| 0.8, U = 4 | 5 dB | 13.711 | 13.809 | +0.098 | +2.90e-03 | +4.04e-03 | -0.4043 | +2.9e-15 | -0.1371 |
| 0.8, U = 4 | 10 dB | 6.970 | 7.028 | +0.057 | +5.57e-03 | +6.19e-03 | -0.3987 | -4.4e-16 | -0.0697 |
| 0.8, U = 4 | 15 dB | 2.300 | 2.303 | +0.003 | +7.04e-03 | +7.07e-03 | -0.3668 | +0.0e+00 | -0.0230 |
| 0.8, U = 4 | 20 dB | 0.676 | 0.675 | -0.001 | +7.26e-03 | +7.26e-03 | -0.3194 | +8.9e-16 | -0.0068 |
| 0.8, U = 5 (overload, pruned Prop. 2) | -10 dB | 31.339 | 31.336 | -0.003 | +2.43e-04 | +1.95e-04 | -0.2623 | +6.7e-16 | -0.3134 |
| 0.8, U = 5 (overload, pruned Prop. 2) | -5 dB | 29.361 | 29.364 | +0.003 | +4.98e-04 | +5.42e-04 | -0.3288 | +2.2e-16 | -0.2936 |
| 0.8, U = 5 (overload, pruned Prop. 2) | 0 dB | 24.308 | 24.336 | +0.028 | +1.09e-03 | +1.46e-03 | -0.3908 | +6.7e-16 | -0.2431 |
| 0.8, U = 5 (overload, pruned Prop. 2) | 10 dB | 12.389 | 12.412 | +0.024 | +4.90e-03 | +5.17e-03 | -0.4084 | -1.8e-04 | -0.1237 |
| 0.8, U = 5 (overload, pruned Prop. 2) | 20 dB | 22.208 | 22.054 | -0.154 | +9.00e-03 | +7.01e-03 | -0.2786 | -1.3e-03 | -0.2211 |
| 0.8, U = 6 (overload, pruned Prop. 2) | -10 dB | 35.047 | 35.043 | -0.005 | +2.39e-04 | +1.68e-04 | -0.2739 | -6.7e-16 | -0.3505 |
| 0.8, U = 6 (overload, pruned Prop. 2) | -5 dB | 32.616 | 32.618 | +0.002 | +4.39e-04 | +4.62e-04 | -0.3428 | -6.7e-16 | -0.3262 |
| 0.8, U = 6 (overload, pruned Prop. 2) | 0 dB | 26.712 | 26.734 | +0.022 | +9.33e-04 | +1.24e-03 | -0.4060 | +2.2e-16 | -0.2671 |
| 0.8, U = 6 (overload, pruned Prop. 2) | 10 dB | 17.868 | 17.872 | +0.004 | +4.27e-03 | +4.31e-03 | -0.3955 | +0.0e+00 | -0.1787 |
| 0.8, U = 6 (overload, pruned Prop. 2) | 20 dB | 35.370 | 35.166 | -0.204 | +9.62e-03 | +6.45e-03 | -0.2125 | -4.4e-16 | -0.3537 |
| 0.8, U = 7 (overload, pruned Prop. 2) | -10 dB | 38.651 | 38.646 | -0.005 | +2.37e-04 | +1.56e-04 | -0.2817 | -1.0e-15 | -0.3865 |
| 0.8, U = 7 (overload, pruned Prop. 2) | -5 dB | 35.798 | 35.799 | +0.001 | +4.03e-04 | +4.21e-04 | -0.3502 | -3.3e-16 | -0.3580 |
| 0.8, U = 7 (overload, pruned Prop. 2) | 0 dB | 29.119 | 29.139 | +0.020 | +8.39e-04 | +1.12e-03 | -0.4137 | -1.3e-15 | -0.2912 |
| 0.8, U = 7 (overload, pruned Prop. 2) | 10 dB | 23.154 | 23.144 | -0.011 | +3.94e-03 | +3.80e-03 | -0.3797 | -9.5e-04 | -0.2308 |
| 0.8, U = 7 (overload, pruned Prop. 2) | 20 dB | 44.154 | 43.938 | -0.216 | +9.60e-03 | +5.71e-03 | -0.1886 | -2.8e-03 | -0.4400 |
| 0.8, U = 8 (overload, pruned Prop. 2) | -10 dB | 41.582 | 41.577 | -0.005 | +2.18e-04 | +1.40e-04 | -0.2815 | +0.0e+00 | -0.4158 |
| 0.8, U = 8 (overload, pruned Prop. 2) | -5 dB | 38.373 | 38.374 | +0.002 | +3.46e-04 | +3.77e-04 | -0.3494 | -7.8e-16 | -0.3837 |
| 0.8, U = 8 (overload, pruned Prop. 2) | 0 dB | 31.255 | 31.273 | +0.018 | +7.50e-04 | +1.01e-03 | -0.4120 | +2.2e-16 | -0.3126 |
| 0.8, U = 8 (overload, pruned Prop. 2) | 10 dB | 27.807 | 27.787 | -0.020 | +3.68e-03 | +3.40e-03 | -0.3496 | +4.4e-16 | -0.2781 |
| 0.8, U = 8 (overload, pruned Prop. 2) | 20 dB | 50.933 | 50.714 | -0.219 | +9.73e-03 | +5.25e-03 | -0.1218 | +0.0e+00 | -0.5093 |
| 0.4, U = 2 | -10 dB | 2.302 | 2.309 | +0.008 | +1.30e-03 | +1.38e-03 | -0.0328 | +0.0e+00 | -0.0230 |
| 0.4, U = 2 | -5 dB | 2.114 | 2.130 | +0.016 | +3.04e-03 | +3.20e-03 | -0.0424 | -7.8e-16 | -0.0211 |
| 0.4, U = 2 | 0 dB | 1.846 | 1.880 | +0.034 | +6.54e-03 | +6.89e-03 | -0.0535 | -1.1e-15 | -0.0185 |
| 0.4, U = 2 | 10 dB | 0.282 | 0.281 | -0.001 | +1.36e-02 | +1.36e-02 | -0.0648 | +2.2e-15 | -0.0028 |
| 0.4, U = 2 | 20 dB | 0.016 | 0.016 | -0.000 | +1.74e-02 | +1.74e-02 | -0.0806 | -5.6e-16 | -0.0002 |
| 0.4, U = 3 | -10 dB | 4.053 | 4.041 | -0.012 | +9.97e-04 | +8.68e-04 | -0.0546 | +0.0e+00 | -0.0405 |
| 0.4, U = 3 | -5 dB | 3.653 | 3.635 | -0.017 | +2.16e-03 | +1.98e-03 | -0.0694 | -5.6e-16 | -0.0365 |
| 0.4, U = 3 | 0 dB | 3.182 | 3.175 | -0.007 | +4.27e-03 | +4.19e-03 | -0.0825 | +1.3e-15 | -0.0318 |
| 0.4, U = 3 | 10 dB | 0.737 | 0.744 | +0.007 | +7.51e-03 | +7.59e-03 | -0.0792 | -1.6e-15 | -0.0074 |
| 0.4, U = 3 | 20 dB | 0.047 | 0.047 | -0.000 | +5.18e-03 | +5.18e-03 | -0.0435 | +0.0e+00 | -0.0005 |
| 0.4, U = 4 | -10 dB | 5.666 | 5.648 | -0.018 | +8.43e-04 | +6.47e-04 | -0.0630 | -2.2e-16 | -0.0567 |
| 0.4, U = 4 | -5 dB | 5.087 | 5.063 | -0.024 | +1.67e-03 | +1.42e-03 | -0.0796 | +2.2e-16 | -0.0509 |
| 0.4, U = 4 | 0 dB | 4.311 | 4.297 | -0.014 | +3.04e-03 | +2.90e-03 | -0.0958 | -3.3e-16 | -0.0431 |
| 0.4, U = 4 | 5 dB | 2.865 | 2.887 | +0.022 | +4.65e-03 | +4.87e-03 | -0.1076 | +4.7e-15 | -0.0286 |
| 0.4, U = 4 | 10 dB | 1.041 | 1.048 | +0.007 | +5.50e-03 | +5.57e-03 | -0.1044 | -3.3e-16 | -0.0104 |
| 0.4, U = 4 | 15 dB | 0.312 | 0.314 | +0.002 | +5.38e-03 | +5.40e-03 | -0.0934 | +2.4e-15 | -0.0031 |
| 0.4, U = 4 | 20 dB | 0.090 | 0.090 | +0.000 | +4.85e-03 | +4.86e-03 | -0.0776 | -1.0e-15 | -0.0009 |
| 0.4, U = 5 (overload, pruned Prop. 2) | -10 dB | 7.050 | 7.027 | -0.023 | +7.83e-04 | +5.31e-04 | -0.0733 | +0.0e+00 | -0.0705 |
| 0.4, U = 5 (overload, pruned Prop. 2) | -5 dB | 6.298 | 6.269 | -0.029 | +1.46e-03 | +1.15e-03 | -0.0924 | +8.9e-16 | -0.0630 |
| 0.4, U = 5 (overload, pruned Prop. 2) | 0 dB | 5.322 | 5.306 | -0.016 | +2.50e-03 | +2.33e-03 | -0.1103 | +2.2e-16 | -0.0532 |
| 0.4, U = 5 (overload, pruned Prop. 2) | 10 dB | 2.879 | 2.896 | +0.017 | +4.35e-03 | +4.52e-03 | -0.1148 | -3.9e-05 | -0.0288 |
| 0.4, U = 5 (overload, pruned Prop. 2) | 20 dB | 7.255 | 7.355 | +0.100 | +6.41e-03 | +7.49e-03 | -0.0776 | -6.2e-04 | -0.0720 |
| 0.4, U = 6 (overload, pruned Prop. 2) | -10 dB | 8.209 | 8.184 | -0.025 | +7.27e-04 | +4.51e-04 | -0.0794 | +2.2e-16 | -0.0821 |
| 0.4, U = 6 (overload, pruned Prop. 2) | -5 dB | 7.275 | 7.244 | -0.030 | +1.29e-03 | +9.64e-04 | -0.0999 | +1.3e-15 | -0.0727 |
| 0.4, U = 6 (overload, pruned Prop. 2) | 0 dB | 6.102 | 6.087 | -0.015 | +2.09e-03 | +1.93e-03 | -0.1185 | -2.2e-16 | -0.0610 |
| 0.4, U = 6 (overload, pruned Prop. 2) | 10 dB | 4.735 | 4.773 | +0.038 | +3.60e-03 | +4.01e-03 | -0.1130 | -7.8e-16 | -0.0474 |
| 0.4, U = 6 (overload, pruned Prop. 2) | 20 dB | 11.825 | 11.977 | +0.152 | +6.37e-03 | +8.11e-03 | -0.0617 | +0.0e+00 | -0.1183 |
| 0.4, U = 7 (overload, pruned Prop. 2) | -10 dB | 9.395 | 9.368 | -0.027 | +7.04e-04 | +4.04e-04 | -0.0844 | -2.2e-16 | -0.0940 |
| 0.4, U = 7 (overload, pruned Prop. 2) | -5 dB | 8.324 | 8.294 | -0.030 | +1.18e-03 | +8.55e-04 | -0.1054 | +0.0e+00 | -0.0832 |
| 0.4, U = 7 (overload, pruned Prop. 2) | 0 dB | 6.935 | 6.925 | -0.010 | +1.80e-03 | +1.70e-03 | -0.1245 | -2.2e-16 | -0.0693 |
| 0.4, U = 7 (overload, pruned Prop. 2) | 10 dB | 6.569 | 6.628 | +0.058 | +3.12e-03 | +3.74e-03 | -0.1124 | -3.8e-04 | -0.0653 |
| 0.4, U = 7 (overload, pruned Prop. 2) | 20 dB | 14.902 | 15.067 | +0.166 | +6.01e-03 | +7.98e-03 | -0.0580 | -2.2e-03 | -0.1472 |
| 0.4, U = 8 (overload, pruned Prop. 2) | -10 dB | 10.453 | 10.426 | -0.027 | +6.69e-04 | +3.70e-04 | -0.0862 | +0.0e+00 | -0.1045 |
| 0.4, U = 8 (overload, pruned Prop. 2) | -5 dB | 9.252 | 9.225 | -0.027 | +1.07e-03 | +7.71e-04 | -0.1073 | +4.4e-16 | -0.0925 |
| 0.4, U = 8 (overload, pruned Prop. 2) | 0 dB | 7.704 | 7.702 | -0.003 | +1.56e-03 | +1.53e-03 | -0.1260 | -3.3e-16 | -0.0770 |
| 0.4, U = 8 (overload, pruned Prop. 2) | 10 dB | 8.309 | 8.386 | +0.077 | +2.74e-03 | +3.59e-03 | -0.1027 | +8.9e-16 | -0.0831 |
| 0.4, U = 8 (overload, pruned Prop. 2) | 20 dB | 17.854 | 18.027 | +0.173 | +5.74e-03 | +7.87e-03 | -0.0368 | -1.1e-16 | -0.1785 |

- largest |A1 - full| of the overlap reduction over kappa 0.8/0.4, U = 2..8 present, all SNRs: 0.219 percentage points (kappa 0.8, U = 8, 20 dB)

#### C.1c Design of the overlapping masks in the model (diag): active modes, pruning, overlap index

| kappa, U | SNR | load coefficients per user | active modes (common) | pruned modes | uses of gen_joint / d_s | overlap index t_bar_c | low-SNR slope gain G0 over load_joint / over load_pu |
|---|---|---|---|---|---|---|---|
| 0.8, U = 2 | -10 dB | 768 | 325 (313) | 0 | 325 / 3072 | 0.8704 | 1.980 / 3.301 |
| 0.8, U = 2 | -5 dB | 768 | 662 (620) | 0 | 662 / 3072 | 0.6898 | 1.980 / 3.301 |
| 0.8, U = 2 | 0 dB | 768 | 896 (767) | 0 | 896 / 3072 | 0.4999 | 1.980 / 3.301 |
| 0.8, U = 2 | 10 dB | 768 | 1534 (767) | 0 | 1534 / 3072 | 0.1026 | 1.980 / 3.301 |
| 0.8, U = 2 | 20 dB | 768 | 1534 (767) | 0 | 1534 / 3072 | 0.0103 | 1.980 / 3.301 |
| 0.8, U = 3 | -10 dB | 768 | 419 (405) | 0 | 419 / 3072 | 0.8928 | 2.971 / 6.931 |
| 0.8, U = 3 | -5 dB | 768 | 771 (705) | 0 | 771 / 3072 | 0.6817 | 2.971 / 6.931 |
| 0.8, U = 3 | 0 dB | 768 | 971 (767) | 0 | 971 / 3072 | 0.4673 | 2.971 / 6.931 |
| 0.8, U = 3 | 10 dB | 768 | 2275 (767) | 0 | 2275 / 3072 | 0.1069 | 2.971 / 6.931 |
| 0.8, U = 3 | 20 dB | 768 | 2301 (767) | 0 | 2301 / 3072 | 0.0107 | 2.971 / 6.931 |
| 0.8, U = 4 | -10 dB | 768 | 497 (482) | 0 | 497 / 3072 | 0.9059 | 3.966 / 11.880 |
| 0.8, U = 4 | -5 dB | 768 | 837 (753) | 0 | 837 / 3072 | 0.6727 | 3.966 / 11.880 |
| 0.8, U = 4 | 0 dB | 768 | 1019 (767) | 0 | 1019 / 3072 | 0.4421 | 3.966 / 11.880 |
| 0.8, U = 4 | 5 dB | 768 | 1511 (767) | 0 | 1511 / 3072 | 0.2502 | 3.966 / 11.880 |
| 0.8, U = 4 | 10 dB | 768 | 2765 (767) | 0 | 2765 / 3072 | 0.1080 | 3.966 / 11.880 |
| 0.8, U = 4 | 15 dB | 768 | 3068 (767) | 0 | 3068 / 3072 | 0.0346 | 3.966 / 11.880 |
| 0.8, U = 4 | 20 dB | 768 | 3068 (767) | 0 | 3068 / 3072 | 0.0109 | 3.966 / 11.880 |
| 0.8, U = 5 (overload, pruned Prop. 2) | -10 dB | 614 | 569 (553) | 0 | 569 / 3072 | 0.9167 | 4.964 / 18.239 |
| 0.8, U = 5 (overload, pruned Prop. 2) | -5 dB | 614 | 857 (765) | 0 | 857 / 3072 | 0.6631 | 4.964 / 18.239 |
| 0.8, U = 5 (overload, pruned Prop. 2) | 0 dB | 614 | 1059 (767) | 0 | 1059 / 3072 | 0.4216 | 4.964 / 18.239 |
| 0.8, U = 5 (overload, pruned Prop. 2) | 10 dB | 614 | 3072 (767) | 23 | 3072 / 3072 | 0.1069 | 4.964 / 18.239 |
| 0.8, U = 5 (overload, pruned Prop. 2) | 20 dB | 614 | 3072 (767) | 763 | 3072 / 3072 | 0.0256 | 4.964 / 18.239 |
| 0.8, U = 6 (overload, pruned Prop. 2) | -10 dB | 512 | 629 (609) | 0 | 629 / 3072 | 0.9232 | 5.961 / 25.806 |
| 0.8, U = 6 (overload, pruned Prop. 2) | -5 dB | 512 | 872 (767) | 0 | 872 / 3072 | 0.6546 | 5.961 / 25.806 |
| 0.8, U = 6 (overload, pruned Prop. 2) | 0 dB | 512 | 1087 (767) | 0 | 1087 / 3072 | 0.4047 | 5.961 / 25.806 |
| 0.8, U = 6 (overload, pruned Prop. 2) | 10 dB | 512 | 3072 (767) | 240 | 3072 / 3072 | 0.1052 | 5.961 / 25.806 |
| 0.8, U = 6 (overload, pruned Prop. 2) | 20 dB | 512 | 3072 (767) | 1530 | 3072 / 3072 | 0.0349 | 5.961 / 25.806 |
| 0.8, U = 7 (overload, pruned Prop. 2) | -10 dB | 438 | 675 (657) | 0 | 675 / 3072 | 0.9305 | 6.959 / 34.776 |
| 0.8, U = 7 (overload, pruned Prop. 2) | -5 dB | 438 | 881 (767) | 0 | 881 / 3072 | 0.6467 | 6.959 / 34.776 |
| 0.8, U = 7 (overload, pruned Prop. 2) | 0 dB | 438 | 1121 (767) | 0 | 1121 / 3072 | 0.3908 | 6.959 / 34.776 |
| 0.8, U = 7 (overload, pruned Prop. 2) | 10 dB | 438 | 3072 (767) | 437 | 3072 / 3072 | 0.1034 | 6.959 / 34.776 |
| 0.8, U = 7 (overload, pruned Prop. 2) | 20 dB | 438 | 3072 (767) | 2297 | 3072 / 3072 | 0.0407 | 6.959 / 34.776 |
| 0.8, U = 8 (overload, pruned Prop. 2) | -10 dB | 384 | 717 (696) | 0 | 717 / 3072 | 0.9377 | 7.959 / 45.154 |
| 0.8, U = 8 (overload, pruned Prop. 2) | -5 dB | 384 | 893 (767) | 0 | 893 / 3072 | 0.6383 | 7.959 / 45.154 |
| 0.8, U = 8 (overload, pruned Prop. 2) | 0 dB | 384 | 1152 (767) | 0 | 1152 / 3072 | 0.3784 | 7.959 / 45.154 |
| 0.8, U = 8 (overload, pruned Prop. 2) | 10 dB | 384 | 3072 (767) | 600 | 3072 / 3072 | 0.1013 | 7.959 / 45.154 |
| 0.8, U = 8 (overload, pruned Prop. 2) | 20 dB | 384 | 3072 (767) | 3064 | 3072 / 3072 | 0.0444 | 7.959 / 45.154 |
| 0.4, U = 2 | -10 dB | 768 | 271 (206) | 0 | 271 / 3072 | 0.3505 | 1.687 / 1.953 |
| 0.4, U = 2 | -5 dB | 768 | 614 (448) | 0 | 614 / 3072 | 0.2624 | 1.687 / 1.953 |
| 0.4, U = 2 | 0 dB | 768 | 1186 (758) | 0 | 1186 / 3072 | 0.1769 | 1.687 / 1.953 |
| 0.4, U = 2 | 10 dB | 768 | 1534 (767) | 0 | 1534 / 3072 | 0.0225 | 1.687 / 1.953 |
| 0.4, U = 2 | 20 dB | 768 | 1534 (767) | 0 | 1534 / 3072 | 0.0022 | 1.687 / 1.953 |
| 0.4, U = 3 | -10 dB | 768 | 315 (219) | 0 | 315 / 3072 | 0.3633 | 2.438 / 3.192 |
| 0.4, U = 3 | -5 dB | 768 | 704 (458) | 0 | 704 / 3072 | 0.2647 | 2.438 / 3.192 |
| 0.4, U = 3 | 0 dB | 768 | 1391 (757) | 0 | 1391 / 3072 | 0.1804 | 2.438 / 3.192 |
| 0.4, U = 3 | 10 dB | 768 | 2301 (767) | 0 | 2301 / 3072 | 0.0265 | 2.438 / 3.192 |
| 0.4, U = 3 | 20 dB | 768 | 2301 (767) | 0 | 2301 / 3072 | 0.0027 | 2.438 / 3.192 |
| 0.4, U = 4 | -10 dB | 768 | 352 (238) | 0 | 352 / 3072 | 0.3710 | 3.239 / 4.728 |
| 0.4, U = 4 | -5 dB | 768 | 780 (480) | 0 | 780 / 3072 | 0.2646 | 3.239 / 4.728 |
| 0.4, U = 4 | 0 dB | 768 | 1517 (761) | 0 | 1517 / 3072 | 0.1792 | 3.239 / 4.728 |
| 0.4, U = 4 | 5 dB | 768 | 2624 (767) | 0 | 2624 / 3072 | 0.0886 | 3.239 / 4.728 |
| 0.4, U = 4 | 10 dB | 768 | 3068 (767) | 0 | 3068 / 3072 | 0.0292 | 3.239 / 4.728 |
| 0.4, U = 4 | 15 dB | 768 | 3068 (767) | 0 | 3068 / 3072 | 0.0092 | 3.239 / 4.728 |
| 0.4, U = 4 | 20 dB | 768 | 3068 (767) | 0 | 3068 / 3072 | 0.0029 | 3.239 / 4.728 |
| 0.4, U = 5 (overload, pruned Prop. 2) | -10 dB | 614 | 396 (260) | 0 | 396 / 3072 | 0.3771 | 4.121 / 6.756 |
| 0.4, U = 5 (overload, pruned Prop. 2) | -5 dB | 614 | 835 (507) | 0 | 835 / 3072 | 0.2634 | 4.121 / 6.756 |
| 0.4, U = 5 (overload, pruned Prop. 2) | 0 dB | 614 | 1616 (764) | 0 | 1616 / 3072 | 0.1762 | 4.121 / 6.756 |
| 0.4, U = 5 (overload, pruned Prop. 2) | 10 dB | 614 | 3072 (767) | 763 | 3072 / 3072 | 0.0402 | 4.121 / 6.756 |
| 0.4, U = 5 (overload, pruned Prop. 2) | 20 dB | 614 | 3072 (767) | 763 | 3072 / 3072 | 0.0192 | 4.121 / 6.756 |
| 0.4, U = 6 (overload, pruned Prop. 2) | -10 dB | 512 | 436 (281) | 0 | 436 / 3072 | 0.3831 | 4.963 / 8.807 |
| 0.4, U = 6 (overload, pruned Prop. 2) | -5 dB | 512 | 902 (532) | 0 | 902 / 3072 | 0.2619 | 4.963 / 8.807 |
| 0.4, U = 6 (overload, pruned Prop. 2) | 0 dB | 512 | 1699 (764) | 0 | 1699 / 3072 | 0.1727 | 4.963 / 8.807 |
| 0.4, U = 6 (overload, pruned Prop. 2) | 10 dB | 512 | 3072 (766) | 1530 | 3072 / 3072 | 0.0474 | 4.963 / 8.807 |
| 0.4, U = 6 (overload, pruned Prop. 2) | 20 dB | 512 | 3072 (764) | 1530 | 3072 / 3072 | 0.0293 | 4.963 / 8.807 |
| 0.4, U = 7 (overload, pruned Prop. 2) | -10 dB | 438 | 466 (304) | 0 | 466 / 3072 | 0.3882 | 5.882 / 11.461 |
| 0.4, U = 7 (overload, pruned Prop. 2) | -5 dB | 438 | 951 (555) | 0 | 951 / 3072 | 0.2606 | 5.882 / 11.461 |
| 0.4, U = 7 (overload, pruned Prop. 2) | 0 dB | 438 | 1767 (765) | 0 | 1767 / 3072 | 0.1696 | 5.882 / 11.461 |
| 0.4, U = 7 (overload, pruned Prop. 2) | 10 dB | 438 | 3072 (765) | 2291 | 3072 / 3072 | 0.0519 | 5.882 / 11.461 |
| 0.4, U = 7 (overload, pruned Prop. 2) | 20 dB | 438 | 3072 (756) | 2297 | 3072 / 3072 | 0.0352 | 5.882 / 11.461 |
| 0.4, U = 8 (overload, pruned Prop. 2) | -10 dB | 384 | 486 (325) | 0 | 486 / 3072 | 0.3918 | 6.793 / 14.301 |
| 0.4, U = 8 (overload, pruned Prop. 2) | -5 dB | 384 | 1000 (580) | 0 | 1000 / 3072 | 0.2589 | 6.793 / 14.301 |
| 0.4, U = 8 (overload, pruned Prop. 2) | 0 dB | 384 | 1844 (766) | 0 | 1844 / 3072 | 0.1663 | 6.793 / 14.301 |
| 0.4, U = 8 (overload, pruned Prop. 2) | 10 dB | 384 | 3072 (761) | 3050 | 3072 / 3072 | 0.0544 | 6.793 / 14.301 |
| 0.4, U = 8 (overload, pruned Prop. 2) | 20 dB | 384 | 3072 (740) | 3064 | 3072 / 3072 | 0.0383 | 6.793 / 14.301 |

### C.2 Real caption mixtures, K = 4, known gains (shared_coco_mix<k>_K4_U<U>_ol, 2000 frames)

| kappa, U | SNR | kappa_bar (fit) | NMSE load_pu / load_joint / gen_joint | overlap reduction [95% CI] | receiver reduction [95% CI] | total reduction 1 - gen_joint/load_pu [95% CI] | CosSim gen_joint - load_joint [95% CI] | CosSim load_joint - load_pu [95% CI] | weakest gen_joint - load_joint [95% CI] |
|---|---|---|---|---|---|---|---|---|---|
| 0.8, U = 2 | -5 dB | 0.8016 | 0.35156 / 0.28340 / 0.24848 | +12.32 % [+12.07, +12.59] | +19.39 % [+18.92, +19.90] | +29.32 % [+28.88, +29.77] | +0.0208 [+0.0203, +0.0211] | +0.0508 [+0.0484, +0.0532] | +0.0239 [+0.0234, +0.0243] |
| 0.8, U = 2 | 0 dB | 0.8016 | 0.22237 / 0.17102 / 0.15086 | +11.78 % [+11.58, +12.00] | +23.09 % [+22.48, +23.70] | +32.16 % [+31.65, +32.66] | +0.0113 [+0.0110, +0.0115] | +0.0351 [+0.0331, +0.0374] | +0.0137 [+0.0134, +0.0140] |
| 0.8, U = 2 | 10 dB | 0.8016 | 0.05852 / 0.04433 / 0.04301 | +2.98 % [+2.89, +3.06] | +24.25 % [+22.91, +25.65] | +26.50 % [+25.23, +27.85] | +0.0007 [+0.0007, +0.0007] | +0.0089 [+0.0078, +0.0102] | +0.0011 [+0.0010, +0.0011] |
| 0.8, U = 2 | 20 dB | 0.8016 | 0.00981 / 0.00805 / 0.00803 | +0.19 % [+0.18, +0.20] | +17.98 % [+14.85, +21.07] | +18.14 % [+15.02, +21.22] | +0.0000 [+0.0000, +0.0000] | +0.0010 [+0.0007, +0.0014] | +0.0000 [+0.0000, +0.0000] |
| 0.8, U = 3 | -5 dB | 0.8019 | 0.40527 / 0.29934 / 0.23770 | +20.59 % [+20.30, +20.87] | +26.14 % [+25.76, +26.50] | +41.35 % [+41.05, +41.65] | +0.0365 [+0.0359, +0.0371] | +0.0780 [+0.0756, +0.0804] | +0.0425 [+0.0418, +0.0432] |
| 0.8, U = 3 | 0 dB | 0.8019 | 0.26489 / 0.18298 / 0.14934 | +18.39 % [+18.16, +18.61] | +30.92 % [+30.46, +31.36] | +43.62 % [+43.28, +43.96] | +0.0187 [+0.0184, +0.0190] | +0.0548 [+0.0527, +0.0569] | +0.0231 [+0.0228, +0.0235] |
| 0.8, U = 3 | 10 dB | 0.8019 | 0.07377 / 0.05024 / 0.04761 | +5.25 % [+5.15, +5.34] | +31.89 % [+30.92, +32.90] | +35.46 % [+34.58, +36.40] | +0.0014 [+0.0013, +0.0014] | +0.0140 [+0.0130, +0.0153] | +0.0026 [+0.0025, +0.0027] |
| 0.8, U = 3 | 20 dB | 0.8019 | 0.01445 / 0.01072 / 0.01068 | +0.39 % [+0.37, +0.40] | +25.82 % [+23.15, +28.31] | +26.10 % [+23.45, +28.58] | +0.0000 [+0.0000, +0.0000] | +0.0022 [+0.0018, +0.0027] | +0.0001 [+0.0000, +0.0001] |
| 0.8, U = 4 | -5 dB | 0.8018 | 0.44087 / 0.31081 / 0.22896 | +26.34 % [+26.02, +26.64] | +29.50 % [+29.21, +29.80] | +48.07 % [+47.80, +48.33] | +0.0484 [+0.0476, +0.0491] | +0.0967 [+0.0944, +0.0992] | +0.0571 [+0.0563, +0.0580] |
| 0.8, U = 4 | 0 dB | 0.8018 | 0.29748 / 0.19450 / 0.14981 | +22.98 % [+22.74, +23.21] | +34.62 % [+34.26, +35.01] | +49.64 % [+49.38, +49.93] | +0.0248 [+0.0244, +0.0252] | +0.0686 [+0.0667, +0.0706] | +0.0306 [+0.0301, +0.0310] |
| 0.8, U = 4 | 10 dB | 0.8018 | 0.09070 / 0.05747 / 0.05345 | +7.01 % [+6.90, +7.10] | +36.63 % [+35.83, +37.41] | +41.07 % [+40.34, +41.76] | +0.0021 [+0.0020, +0.0022] | +0.0200 [+0.0188, +0.0212] | +0.0043 [+0.0042, +0.0044] |
| 0.8, U = 4 | 20 dB | 0.8018 | 0.01727 / 0.01263 / 0.01255 | +0.61 % [+0.59, +0.62] | +26.88 % [+24.94, +28.86] | +27.32 % [+25.40, +29.29] | +0.0000 [+0.0000, +0.0000] | +0.0026 [+0.0023, +0.0031] | +0.0001 [+0.0001, +0.0001] |
| 0.8, U = 5 (overload, pruned Prop. 2) | -5 dB | 0.8018 | 0.47377 / 0.32378 / 0.22526 | +30.43 % [+30.12, +30.75] | +31.66 % [+31.39, +31.95] | +52.45 % [+52.25, +52.66] | +0.0583 [+0.0575, +0.0592] | +0.1127 [+0.1104, +0.1150] | +0.0689 [+0.0679, +0.0700] |
| 0.8, U = 5 (overload, pruned Prop. 2) | 0 dB | 0.8018 | 0.32382 / 0.20405 / 0.15054 | +26.22 % [+25.97, +26.49] | +36.99 % [+36.65, +37.30] | +53.51 % [+53.27, +53.74] | +0.0297 [+0.0293, +0.0301] | +0.0807 [+0.0786, +0.0829] | +0.0366 [+0.0361, +0.0371] |
| 0.8, U = 5 (overload, pruned Prop. 2) | 10 dB | 0.8018 | 0.10630 / 0.06788 / 0.05765 | +15.07 % [+14.89, +15.26] | +36.14 % [+35.53, +36.79] | +45.76 % [+45.27, +46.28] | +0.0053 [+0.0052, +0.0054] | +0.0231 [+0.0221, +0.0243] | +0.0078 [+0.0077, +0.0080] |
| 0.8, U = 5 (overload, pruned Prop. 2) | 20 dB | 0.8018 | 0.02699 / 0.02184 / 0.01506 | +31.04 % [+30.34, +31.75] | +19.07 % [+17.70, +20.48] | +44.19 % [+43.43, +44.96] | +0.0034 [+0.0034, +0.0035] | +0.0029 [+0.0026, +0.0033] | +0.0035 [+0.0034, +0.0035] |
| 0.8, U = 6 (overload, pruned Prop. 2) | -5 dB | 0.8015 | 0.49671 / 0.33366 / 0.21977 | +34.13 % [+33.81, +34.45] | +32.83 % [+32.57, +33.09] | +55.75 % [+55.56, +55.95] | +0.0675 [+0.0666, +0.0684] | +0.1234 [+0.1211, +0.1258] | +0.0809 [+0.0798, +0.0821] |
| 0.8, U = 6 (overload, pruned Prop. 2) | 0 dB | 0.8015 | 0.34879 / 0.21451 / 0.15301 | +28.67 % [+28.43, +28.92] | +38.50 % [+38.22, +38.78] | +56.13 % [+55.92, +56.35] | +0.0343 [+0.0338, +0.0348] | +0.0911 [+0.0890, +0.0932] | +0.0420 [+0.0414, +0.0426] |
| 0.8, U = 6 (overload, pruned Prop. 2) | 10 dB | 0.8015 | 0.12148 / 0.07797 / 0.06099 | +21.78 % [+21.50, +22.07] | +35.81 % [+35.18, +36.41] | +49.79 % [+49.35, +50.21] | +0.0088 [+0.0087, +0.0090] | +0.0265 [+0.0253, +0.0277] | +0.0115 [+0.0114, +0.0117] |
| 0.8, U = 6 (overload, pruned Prop. 2) | 20 dB | 0.8015 | 0.03629 / 0.03108 / 0.01740 | +44.03 % [+43.31, +44.70] | +14.36 % [+13.27, +15.46] | +52.07 % [+51.54, +52.56] | +0.0069 [+0.0068, +0.0070] | +0.0030 [+0.0027, +0.0033] | +0.0071 [+0.0070, +0.0072] |
| 0.8, U = 7 (overload, pruned Prop. 2) | -5 dB | 0.8019 | 0.51890 / 0.34334 / 0.21722 | +36.73 % [+36.41, +37.01] | +33.83 % [+33.61, +34.07] | +58.14 % [+57.98, +58.30] | +0.0749 [+0.0738, +0.0760] | +0.1340 [+0.1318, +0.1362] | +0.0900 [+0.0887, +0.0912] |
| 0.8, U = 7 (overload, pruned Prop. 2) | 0 dB | 0.8019 | 0.36585 / 0.22100 / 0.15264 | +30.93 % [+30.66, +31.22] | +39.59 % [+39.35, +39.84] | +58.28 % [+58.09, +58.48] | +0.0381 [+0.0376, +0.0387] | +0.0988 [+0.0969, +0.1008] | +0.0462 [+0.0456, +0.0469] |
| 0.8, U = 7 (overload, pruned Prop. 2) | 10 dB | 0.8019 | 0.13339 / 0.08779 / 0.06416 | +26.92 % [+26.59, +27.27] | +34.18 % [+33.61, +34.71] | +51.90 % [+51.53, +52.25] | +0.0123 [+0.0122, +0.0125] | +0.0278 [+0.0267, +0.0289] | +0.0152 [+0.0150, +0.0154] |
| 0.8, U = 7 (overload, pruned Prop. 2) | 20 dB | 0.8019 | 0.04674 / 0.04081 / 0.02004 | +50.89 % [+50.25, +51.55] | +12.69 % [+11.74, +13.65] | +57.12 % [+56.65, +57.56] | +0.0106 [+0.0104, +0.0107] | +0.0036 [+0.0032, +0.0040] | +0.0109 [+0.0107, +0.0111] |
| 0.8, U = 8 (overload, pruned Prop. 2) | -5 dB | 0.8018 | 0.53131 / 0.34914 / 0.21353 | +38.84 % [+38.55, +39.15] | +34.29 % [+34.08, +34.51] | +59.81 % [+59.66, +59.98] | +0.0805 [+0.0795, +0.0817] | +0.1389 [+0.1370, +0.1410] | +0.0978 [+0.0965, +0.0992] |
| 0.8, U = 8 (overload, pruned Prop. 2) | 0 dB | 0.8018 | 0.38153 / 0.22827 / 0.15319 | +32.89 % [+32.62, +33.18] | +40.17 % [+39.93, +40.40] | +59.85 % [+59.66, +60.02] | +0.0420 [+0.0414, +0.0425] | +0.1048 [+0.1029, +0.1066] | +0.0509 [+0.0502, +0.0515] |
| 0.8, U = 8 (overload, pruned Prop. 2) | 10 dB | 0.8018 | 0.14651 / 0.09732 / 0.06710 | +31.05 % [+30.67, +31.41] | +33.57 % [+33.07, +34.05] | +54.20 % [+53.87, +54.51] | +0.0158 [+0.0156, +0.0160] | +0.0302 [+0.0291, +0.0312] | +0.0189 [+0.0186, +0.0191] |
| 0.8, U = 8 (overload, pruned Prop. 2) | 20 dB | 0.8018 | 0.05601 / 0.04969 / 0.02227 | +55.18 % [+54.64, +55.75] | +11.28 % [+10.49, +12.08] | +60.24 % [+59.86, +60.63] | +0.0140 [+0.0138, +0.0142] | +0.0037 [+0.0034, +0.0041] | +0.0146 [+0.0143, +0.0148] |
| 0, U = 8 (overload, pruned Prop. 2) | -5 dB | 0.0018 | 0.52949 / 0.52975 / 0.52976 | -0.0004 % [-0.0009, -0.0000] | -0.05 % [-0.06, -0.04] | -0.05 % [-0.06, -0.04] | -0.0000 [-0.0000, +0.0000] | -0.0003 [-0.0004, -0.0002] | +0.0000 [-0.0000, +0.0000] |
| 0, U = 8 (overload, pruned Prop. 2) | 0 dB | 0.0018 | 0.37694 / 0.37729 / 0.37730 | -0.0006 % [-0.0009, -0.0002] | -0.09 % [-0.10, -0.08] | -0.09 % [-0.10, -0.08] | -0.0000 [-0.0000, -0.0000] | -0.0004 [-0.0005, -0.0003] | -0.0000 [-0.0000, -0.0000] |
| 0, U = 8 (overload, pruned Prop. 2) | 10 dB | 0.0018 | 0.14415 / 0.14451 / 0.14450 | +0.0019 % [-0.0001, +0.0040] | -0.25 % [-0.27, -0.22] | -0.24 % [-0.27, -0.22] | -0.0000 [-0.0000, +0.0000] | -0.0003 [-0.0003, -0.0002] | -0.0000 [-0.0001, -0.0000] |
| 0, U = 8 (overload, pruned Prop. 2) | 20 dB | 0.0018 | 0.05421 / 0.05443 / 0.05442 | +0.0101 % [+0.0067, +0.0143] | -0.40 % [-0.42, -0.38] | -0.39 % [-0.41, -0.37] | +0.0000 [+0.0000, +0.0000] | -0.0001 [-0.0001, -0.0001] | -0.0000 [-0.0000, -0.0000] |

#### C.2b CosSim of every scheme with real captions (mean / weakest user), K = 4

| kappa, U | SNR | load_pu | load_joint | gen_joint |
|---|---|---|---|---|
| 0.8, U = 2 | -5 dB | 0.7933 / 0.7205 | 0.8440 / 0.8124 | 0.8648 / 0.8363 |
| 0.8, U = 2 | 0 dB | 0.8740 / 0.8185 | 0.9091 / 0.8828 | 0.9204 / 0.8965 |
| 0.8, U = 2 | 10 dB | 0.9684 / 0.9481 | 0.9773 / 0.9652 | 0.9780 / 0.9663 |
| 0.8, U = 2 | 20 dB | 0.9949 / 0.9910 | 0.9959 / 0.9931 | 0.9959 / 0.9931 |
| 0.8, U = 3 | -5 dB | 0.7580 / 0.6293 | 0.8360 / 0.7938 | 0.8725 / 0.8364 |
| 0.8, U = 3 | 0 dB | 0.8485 / 0.7441 | 0.9033 / 0.8661 | 0.9220 / 0.8892 |
| 0.8, U = 3 | 10 dB | 0.9604 / 0.9175 | 0.9744 / 0.9536 | 0.9758 / 0.9562 |
| 0.8, U = 3 | 20 dB | 0.9924 / 0.9814 | 0.9946 / 0.9879 | 0.9946 / 0.9879 |
| 0.8, U = 4 | -5 dB | 0.7329 / 0.5578 | 0.8296 / 0.7817 | 0.8780 / 0.8389 |
| 0.8, U = 4 | 0 dB | 0.8285 / 0.6839 | 0.8971 / 0.8539 | 0.9219 / 0.8844 |
| 0.8, U = 4 | 10 dB | 0.9507 / 0.8813 | 0.9707 / 0.9422 | 0.9728 / 0.9466 |
| 0.8, U = 4 | 20 dB | 0.9910 / 0.9737 | 0.9937 / 0.9835 | 0.9937 / 0.9836 |
| 0.8, U = 5 (overload, pruned Prop. 2) | -5 dB | 0.7092 / 0.4923 | 0.8219 / 0.7703 | 0.8803 / 0.8392 |
| 0.8, U = 5 (overload, pruned Prop. 2) | 0 dB | 0.8113 / 0.6291 | 0.8920 / 0.8441 | 0.9217 / 0.8807 |
| 0.8, U = 5 (overload, pruned Prop. 2) | 10 dB | 0.9423 / 0.8493 | 0.9654 / 0.9310 | 0.9707 / 0.9388 |
| 0.8, U = 5 (overload, pruned Prop. 2) | 20 dB | 0.9861 / 0.9625 | 0.9890 / 0.9759 | 0.9924 / 0.9793 |
| 0.8, U = 6 (overload, pruned Prop. 2) | -5 dB | 0.6926 / 0.4457 | 0.8160 / 0.7607 | 0.8835 / 0.8416 |
| 0.8, U = 6 (overload, pruned Prop. 2) | 0 dB | 0.7951 / 0.5796 | 0.8862 / 0.8360 | 0.9205 / 0.8780 |
| 0.8, U = 6 (overload, pruned Prop. 2) | 10 dB | 0.9336 / 0.8168 | 0.9602 / 0.9214 | 0.9690 / 0.9329 |
| 0.8, U = 6 (overload, pruned Prop. 2) | 20 dB | 0.9813 / 0.9534 | 0.9843 / 0.9691 | 0.9913 / 0.9762 |
| 0.8, U = 7 (overload, pruned Prop. 2) | -5 dB | 0.6762 / 0.4000 | 0.8102 / 0.7525 | 0.8851 / 0.8425 |
| 0.8, U = 7 (overload, pruned Prop. 2) | 0 dB | 0.7838 / 0.5409 | 0.8826 / 0.8311 | 0.9207 / 0.8773 |
| 0.8, U = 7 (overload, pruned Prop. 2) | 10 dB | 0.9273 / 0.7949 | 0.9551 / 0.9133 | 0.9674 / 0.9285 |
| 0.8, U = 7 (overload, pruned Prop. 2) | 20 dB | 0.9758 / 0.9391 | 0.9794 / 0.9610 | 0.9899 / 0.9719 |
| 0.8, U = 8 (overload, pruned Prop. 2) | -5 dB | 0.6678 / 0.3763 | 0.8066 / 0.7470 | 0.8872 / 0.8448 |
| 0.8, U = 8 (overload, pruned Prop. 2) | 0 dB | 0.7737 / 0.5037 | 0.8785 / 0.8249 | 0.9205 / 0.8758 |
| 0.8, U = 8 (overload, pruned Prop. 2) | 10 dB | 0.9199 / 0.7666 | 0.9501 / 0.9050 | 0.9659 / 0.9239 |
| 0.8, U = 8 (overload, pruned Prop. 2) | 20 dB | 0.9711 / 0.9292 | 0.9748 / 0.9544 | 0.9888 / 0.9690 |
| 0, U = 8 (overload, pruned Prop. 2) | -5 dB | 0.6684 / 0.3667 | 0.6680 / 0.3652 | 0.6680 / 0.3652 |
| 0, U = 8 (overload, pruned Prop. 2) | 0 dB | 0.7767 / 0.5067 | 0.7763 / 0.5052 | 0.7763 / 0.5051 |
| 0, U = 8 (overload, pruned Prop. 2) | 10 dB | 0.9212 / 0.7683 | 0.9210 / 0.7672 | 0.9210 / 0.7672 |
| 0, U = 8 (overload, pruned Prop. 2) | 20 dB | 0.9720 / 0.9288 | 0.9719 / 0.9285 | 0.9719 / 0.9285 |

#### C.2c Design of the overlapping masks with real captions (JSON design.gen_joint and design.load)

| kappa, U | SNR | load masks: uses / coefficients per frame | gen_joint uses | active modes / pruned | partial coefficients | overlap index t | t_overlap_modes |
|---|---|---|---|---|---|---|---|
| 0.8, U = 2 | -5 dB | 600 / 300 | 1240 | 662 / 0 | 0 | 0.6898 | 0.6898 |
| 0.8, U = 2 | 0 dB | 1278 / 639 | 1534 | 896 / 0 | 0 | 0.4999 | 0.4999 |
| 0.8, U = 2 | 10 dB | 1534 / 767 | 1534 | 1534 / 0 | 0 | 0.1026 | 0.1026 |
| 0.8, U = 2 | 20 dB | 1534 / 767 | 1534 | 1534 / 0 | 0 | 0.0103 | 0.0103 |
| 0.8, U = 3 | -5 dB | 666 / 222 | 2115 | 771 / 0 | 0 | 0.6817 | 0.6817 |
| 0.8, U = 3 | 0 dB | 1521 / 507 | 2301 | 971 / 0 | 0 | 0.4673 | 0.4673 |
| 0.8, U = 3 | 10 dB | 2301 / 767 | 2301 | 2275 / 0 | 0 | 0.1069 | 0.1069 |
| 0.8, U = 3 | 20 dB | 2301 / 767 | 2301 | 2301 / 0 | 0 | 0.0107 | 0.0107 |
| 0.8, U = 4 | -5 dB | 716 / 179 | 3012 | 837 / 0 | 0 | 0.6727 | 0.6727 |
| 0.8, U = 4 | 0 dB | 1676 / 419 | 3068 | 1019 / 0 | 0 | 0.4421 | 0.4421 |
| 0.8, U = 4 | 10 dB | 3068 / 767 | 3068 | 2765 / 0 | 0 | 0.1080 | 0.1080 |
| 0.8, U = 4 | 20 dB | 3068 / 767 | 3068 | 3068 / 0 | 0 | 0.0109 | 0.0109 |
| 0.8, U = 5 (overload, pruned Prop. 2) | -5 dB | 765 / 153 | 857 | 857 / 0 | 0 | 0.6631 | 0.6631 |
| 0.8, U = 5 (overload, pruned Prop. 2) | 0 dB | 1790 / 358 | 1059 | 1059 / 0 | 0 | 0.4216 | 0.4216 |
| 0.8, U = 5 (overload, pruned Prop. 2) | 10 dB | 3070 / 614 | 3072 | 3072 / 23 | 6 | 0.1069 | 0.1069 |
| 0.8, U = 5 (overload, pruned Prop. 2) | 20 dB | 3070 / 614 | 3072 | 3072 / 763 | 191 | 0.0256 | 0.0255 |
| 0.8, U = 6 (overload, pruned Prop. 2) | -5 dB | 798 / 133 | 872 | 872 / 0 | 0 | 0.6546 | 0.6546 |
| 0.8, U = 6 (overload, pruned Prop. 2) | 0 dB | 1872 / 312 | 1087 | 1087 / 0 | 0 | 0.4047 | 0.4047 |
| 0.8, U = 6 (overload, pruned Prop. 2) | 10 dB | 3072 / 512 | 3072 | 3072 / 240 | 48 | 0.1052 | 0.1052 |
| 0.8, U = 6 (overload, pruned Prop. 2) | 20 dB | 3072 / 512 | 3072 | 3072 / 1530 | 306 | 0.0349 | 0.0349 |
| 0.8, U = 7 (overload, pruned Prop. 2) | -5 dB | 833 / 119 | 881 | 881 / 0 | 0 | 0.6467 | 0.6467 |
| 0.8, U = 7 (overload, pruned Prop. 2) | 0 dB | 1960 / 280 | 1121 | 1121 / 0 | 0 | 0.3908 | 0.3908 |
| 0.8, U = 7 (overload, pruned Prop. 2) | 10 dB | 3066 / 438 | 3072 | 3072 / 437 | 73 | 0.1034 | 0.1033 |
| 0.8, U = 7 (overload, pruned Prop. 2) | 20 dB | 3066 / 438 | 3072 | 3072 / 2297 | 383 | 0.0407 | 0.0406 |
| 0.8, U = 8 (overload, pruned Prop. 2) | -5 dB | 864 / 108 | 893 | 893 / 0 | 0 | 0.6383 | 0.6383 |
| 0.8, U = 8 (overload, pruned Prop. 2) | 0 dB | 2024 / 253 | 1152 | 1152 / 0 | 0 | 0.3784 | 0.3784 |
| 0.8, U = 8 (overload, pruned Prop. 2) | 10 dB | 3072 / 384 | 3072 | 3072 / 600 | 86 | 0.1013 | 0.1013 |
| 0.8, U = 8 (overload, pruned Prop. 2) | 20 dB | 3072 / 384 | 3072 | 3072 / 3064 | 438 | 0.0444 | 0.0443 |
| 0, U = 8 (overload, pruned Prop. 2) | -5 dB | 856 / 107 | 857 | 857 / 0 | 0 | 0.0008 | 0.0008 |
| 0, U = 8 (overload, pruned Prop. 2) | 0 dB | 2016 / 252 | 2010 | 2010 / 0 | 0 | 0.0006 | 0.0006 |
| 0, U = 8 (overload, pruned Prop. 2) | 10 dB | 3072 / 384 | 3072 | 3072 / 3064 | 2 | 0.0001 | 0.0001 |
| 0, U = 8 (overload, pruned Prop. 2) | 20 dB | 3072 / 384 | 3072 | 3072 / 3064 | 2 | 0.0000 | 0.0000 |

#### C.2d Model against real captions, kappa_bar 0.8, K = 4: overlap and receiver NMSE reductions (%)

| U | SNR | overlap: model | overlap: captions [95% CI] | overlap: model - captions (pp) | model inside the caption CI | receiver: model | receiver: captions [95% CI] | receiver: model - captions (pp) |
|---|---|---|---|---|---|---|---|---|
| U = 2 | -5 dB | 11.60 | +12.32 % [+12.07, +12.59] | -0.72 | False | 19.00 | +19.39 % [+18.92, +19.90] | -0.39 |
| U = 2 | 0 dB | 10.78 | +11.78 % [+11.58, +12.00] | -1.01 | False | 23.36 | +23.09 % [+22.48, +23.70] | +0.27 |
| U = 2 | 10 dB | 2.94 | +2.98 % [+2.89, +3.06] | -0.04 | True | 27.63 | +24.25 % [+22.91, +25.65] | +3.39 |
| U = 2 | 20 dB | 0.19 | +0.19 % [+0.18, +0.20] | -0.01 | True | 32.11 | +17.98 % [+14.85, +21.07] | +14.13 |
| U = 3 | -5 dB | 18.80 | +20.59 % [+20.30, +20.87] | -1.79 | False | 28.02 | +26.14 % [+25.76, +26.50] | +1.89 |
| U = 3 | 0 dB | 16.33 | +18.39 % [+18.16, +18.61] | -2.05 | False | 33.41 | +30.92 % [+30.46, +31.36] | +2.49 |
| U = 3 | 10 dB | 5.58 | +5.25 % [+5.15, +5.34] | +0.34 | False | 33.73 | +31.89 % [+30.92, +32.90] | +1.84 |
| U = 3 | 20 dB | 0.46 | +0.39 % [+0.37, +0.40] | +0.07 | False | 21.07 | +25.82 % [+23.15, +28.31] | -4.75 |
| U = 4 | -5 dB | 24.66 | +26.34 % [+26.02, +26.64] | -1.67 | False | 30.14 | +29.50 % [+29.21, +29.80] | +0.64 |
| U = 4 | 0 dB | 20.71 | +22.98 % [+22.74, +23.21] | -2.27 | False | 36.08 | +34.62 % [+34.26, +35.01] | +1.46 |
| U = 4 | 10 dB | 6.97 | +7.01 % [+6.90, +7.10] | -0.04 | True | 39.87 | +36.63 % [+35.83, +37.41] | +3.24 |
| U = 4 | 20 dB | 0.68 | +0.61 % [+0.59, +0.62] | +0.07 | False | 31.94 | +26.88 % [+24.94, +28.86] | +5.06 |
| U = 5 (overload, pruned Prop. 2) | -5 dB | 29.36 | +30.43 % [+30.12, +30.75] | -1.07 | False | 32.88 | +31.66 % [+31.39, +31.95] | +1.22 |
| U = 5 (overload, pruned Prop. 2) | 0 dB | 24.31 | +26.22 % [+25.97, +26.49] | -1.91 | False | 39.08 | +36.99 % [+36.65, +37.30] | +2.09 |
| U = 5 (overload, pruned Prop. 2) | 10 dB | 12.39 | +15.07 % [+14.89, +15.26] | -2.68 | False | 40.84 | +36.14 % [+35.53, +36.79] | +4.70 |
| U = 5 (overload, pruned Prop. 2) | 20 dB | 22.21 | +31.04 % [+30.34, +31.75] | -8.84 | False | 27.86 | +19.07 % [+17.70, +20.48] | +8.80 |
| U = 6 (overload, pruned Prop. 2) | -5 dB | 32.62 | +34.13 % [+33.81, +34.45] | -1.52 | False | 34.28 | +32.83 % [+32.57, +33.09] | +1.45 |
| U = 6 (overload, pruned Prop. 2) | 0 dB | 26.71 | +28.67 % [+28.43, +28.92] | -1.96 | False | 40.60 | +38.50 % [+38.22, +38.78] | +2.11 |
| U = 6 (overload, pruned Prop. 2) | 10 dB | 17.87 | +21.78 % [+21.50, +22.07] | -3.91 | False | 39.55 | +35.81 % [+35.18, +36.41] | +3.74 |
| U = 6 (overload, pruned Prop. 2) | 20 dB | 35.37 | +44.03 % [+43.31, +44.70] | -8.66 | False | 21.25 | +14.36 % [+13.27, +15.46] | +6.89 |
| U = 7 (overload, pruned Prop. 2) | -5 dB | 35.80 | +36.73 % [+36.41, +37.01] | -0.93 | False | 35.02 | +33.83 % [+33.61, +34.07] | +1.19 |
| U = 7 (overload, pruned Prop. 2) | 0 dB | 29.12 | +30.93 % [+30.66, +31.22] | -1.81 | False | 41.37 | +39.59 % [+39.35, +39.84] | +1.77 |
| U = 7 (overload, pruned Prop. 2) | 10 dB | 23.15 | +26.92 % [+26.59, +27.27] | -3.76 | False | 37.97 | +34.18 % [+33.61, +34.71] | +3.79 |
| U = 7 (overload, pruned Prop. 2) | 20 dB | 44.15 | +50.89 % [+50.25, +51.55] | -6.73 | False | 18.86 | +12.69 % [+11.74, +13.65] | +6.17 |
| U = 8 (overload, pruned Prop. 2) | -5 dB | 38.37 | +38.84 % [+38.55, +39.15] | -0.47 | False | 34.94 | +34.29 % [+34.08, +34.51] | +0.65 |
| U = 8 (overload, pruned Prop. 2) | 0 dB | 31.26 | +32.89 % [+32.62, +33.18] | -1.64 | False | 41.20 | +40.17 % [+39.93, +40.40] | +1.03 |
| U = 8 (overload, pruned Prop. 2) | 10 dB | 27.81 | +31.05 % [+30.67, +31.41] | -3.24 | False | 34.96 | +33.57 % [+33.07, +34.05] | +1.39 |
| U = 8 (overload, pruned Prop. 2) | 20 dB | 50.93 | +55.18 % [+54.64, +55.75] | -4.25 | False | 12.18 | +11.28 % [+10.49, +12.08] | +0.90 |

- check: shared_coco_mix0.8_K4_U4 (all six modes, 0/10 dB) against the _ol run of the same point, largest frame-by-frame |difference| of the shared arrays: 0.0e+00
- kappa_bar of the model fit against the fit of the caption runs, kappa 0.8: largest |difference| 8.0e-13 (U = 7)

### C.3 Content correlation sweep at U = 4 (full load at K = 4; overload at K = 2, gen_joint = pruned Proposition 2)

Mixture kappa 0 = captions of different images; its fitted kappa_bar is about 0.002 (not 0), and the design and the receiver use that fitted C. Values below 0.05 % print with four decimals.


#### K = 4: model (Gaussian, 100 Rayleigh draws), overlap reduction % (SE) and receiver reduction %

| mixture kappa (kappa_bar) | -10 dB overlap (SE) / receiver | -5 dB overlap (SE) / receiver | 0 dB overlap (SE) / receiver | 5 dB overlap (SE) / receiver | 10 dB overlap (SE) / receiver | 15 dB overlap (SE) / receiver | 20 dB overlap (SE) / receiver |
|---|---|---|---|---|---|---|---|
| 0 (0.002) | 0.0004 (0.0000) / 0.05 | 0.0007 (0.0000) / 0.12 | 0.0008 (0.0000) / 0.27 | 0.0007 (0.0001) / 0.50 | 0.0001 (0.0000) / 0.63 | 0.0000 (0.0000) / 0.62 | 0.0000 (0.0000) / 0.54 |
| 0.2 (0.196) | 1.50 (0.02) / 1.75 | 1.37 (0.02) / 2.29 | 1.20 (0.02) / 2.87 | 0.84 (0.05) / 3.38 | 0.29 (0.03) / 3.37 | 0.08 (0.01) / 3.04 | 0.0241 (0.0051) / 2.52 |
| 0.4 (0.396) | 5.67 (0.08) / 6.30 | 5.09 (0.05) / 7.96 | 4.31 (0.08) / 9.58 | 2.86 (0.16) / 10.76 | 1.04 (0.09) / 10.44 | 0.31 (0.04) / 9.34 | 0.09 (0.02) / 7.76 |
| 0.6 (0.600) | 13.11 (0.16) / 13.55 | 11.81 (0.11) / 16.95 | 9.85 (0.20) / 20.19 | 6.43 (0.35) / 22.46 | 2.61 (0.22) / 21.79 | 0.80 (0.09) / 19.62 | 0.23 (0.04) / 16.55 |
| 0.8 (0.802) | 26.15 (0.29) / 23.95 | 24.66 (0.23) / 30.14 | 20.71 (0.46) / 36.08 | 13.71 (0.74) / 40.43 | 6.97 (0.56) / 39.87 | 2.30 (0.23) / 36.68 | 0.68 (0.09) / 31.94 |
| 0.9 (0.902) | 35.47 (0.43) / 30.81 | 36.93 (0.37) / 39.26 | 32.38 (0.80) / 47.57 | 22.28 (1.26) / 54.08 | 13.13 (1.03) / 54.49 | 5.29 (0.49) / 51.31 | 1.61 (0.18) / 46.05 |
| 0.95 (0.951) | 40.32 (0.53) / 34.86 | 45.76 (0.45) / 44.94 | 43.75 (1.21) / 55.14 | 32.05 (1.97) / 63.64 | 20.48 (1.69) / 65.49 | 10.49 (0.94) / 63.10 | 3.54 (0.35) / 58.16 |

#### K = 4: real caption mixtures (shared_coco_mix<k>_K4_U4, 2000 frames)

| mixture kappa (kappa_bar) | SNR | overlap reduction [95% CI] | receiver reduction [95% CI] | CosSim gen_joint - load_joint [95% CI] | CosSim gen_eq - load_joint [95% CI] |
|---|---|---|---|---|---|
| 0 (0.002) | 0 dB | -0.0006 % [-0.0013, +0.0001] | -0.28 % [-0.31, -0.25] | -0.000003 [-0.000006, +0.000000] | -0.4002 [-0.4033, -0.3969] |
| 0 (0.002) | 10 dB | +0.0000 % [-0.0000, +0.0001] | -0.69 % [-0.77, -0.61] | +0.000000 [-0.000000, +0.000000] | -0.5071 [-0.5097, -0.5045] |
| 0.2 (0.196) | 0 dB | +1.30 % [+1.27, +1.33] | +2.21 % [+2.11, +2.32] | +0.0024 [+0.0023, +0.0024] | -0.2579 [-0.2602, -0.2554] |
| 0.2 (0.196) | 10 dB | +0.30 % [+0.29, +0.30] | +2.08 % [+1.92, +2.23] | +0.000154 [+0.000147, +0.000161] | -0.3591 [-0.3614, -0.3569] |
| 0.4 (0.396) | 0 dB | +4.87 % [+4.78, +4.96] | +8.70 % [+8.52, +8.90] | +0.0081 [+0.0080, +0.0083] | -0.1491 [-0.1509, -0.1473] |
| 0.4 (0.396) | 10 dB | +1.06 % [+1.04, +1.08] | +8.57 % [+8.21, +8.90] | +0.0005 [+0.0005, +0.0005] | -0.2403 [-0.2419, -0.2387] |
| 0.6 (0.600) | 0 dB | +11.11 % [+10.95, +11.26] | +18.99 % [+18.73, +19.29] | +0.0157 [+0.0155, +0.0160] | -0.0632 [-0.0645, -0.0619] |
| 0.6 (0.600) | 10 dB | +2.65 % [+2.61, +2.69] | +19.08 % [+18.49, +19.65] | +0.0011 [+0.0010, +0.0011] | -0.1403 [-0.1414, -0.1392] |
| 0.8 (0.802) | 0 dB | +22.98 % [+22.74, +23.21] | +34.62 % [+34.26, +35.01] | +0.0248 [+0.0244, +0.0252] | +0.0022 [+0.0014, +0.0030] |
| 0.8 (0.802) | 10 dB | +7.01 % [+6.90, +7.10] | +36.63 % [+35.83, +37.41] | +0.0021 [+0.0020, +0.0022] | -0.0558 [-0.0564, -0.0552] |
| 0.9 (0.902) | 0 dB | +35.25 % [+34.95, +35.54] | +46.25 % [+45.85, +46.66] | +0.0303 [+0.0299, +0.0307] | +0.0258 [+0.0252, +0.0264] |
| 0.9 (0.902) | 10 dB | +13.11 % [+12.95, +13.27] | +51.33 % [+50.56, +52.08] | +0.0029 [+0.0029, +0.0030] | -0.0201 [-0.0205, -0.0198] |
| 0.95 (0.951) | 0 dB | +47.08 % [+46.76, +47.39] | +54.04 % [+53.65, +54.43] | +0.0340 [+0.0335, +0.0345] | +0.0339 [+0.0333, +0.0344] |
| 0.95 (0.951) | 10 dB | +20.23 % [+20.00, +20.45] | +63.08 % [+62.37, +63.75] | +0.0034 [+0.0033, +0.0035] | -0.0051 [-0.0053, -0.0049] |

#### K = 2: model (Gaussian, 100 Rayleigh draws), overlap reduction % (SE) and receiver reduction %

| mixture kappa (kappa_bar) | -10 dB overlap (SE) / receiver | -5 dB overlap (SE) / receiver | 0 dB overlap (SE) / receiver | 5 dB overlap (SE) / receiver | 10 dB overlap (SE) / receiver | 15 dB overlap (SE) / receiver | 20 dB overlap (SE) / receiver |
|---|---|---|---|---|---|---|---|
| 0 (0.002) | 0.0004 (0.0000) / 0.0332 | 0.0004 (0.0000) / 0.08 | 0.0008 (0.0000) / 0.17 | 0.0033 (0.0002) / 0.32 | 0.0096 (0.0007) / 0.44 | 0.0149 (0.0015) / 0.56 | 0.0154 (0.0022) / 0.73 |
| 0.2 (0.196) | 1.58 (0.03) / 1.45 | 1.45 (0.02) / 1.96 | 1.31 (0.01) / 2.52 | 1.25 (0.02) / 2.96 | 1.68 (0.03) / 2.80 | 2.34 (0.05) / 2.38 | 3.06 (0.07) / 1.91 |
| 0.4 (0.396) | 5.96 (0.09) / 5.31 | 5.45 (0.07) / 6.96 | 4.81 (0.05) / 8.62 | 4.32 (0.07) / 9.69 | 5.54 (0.05) / 8.71 | 7.99 (0.08) / 6.81 | 11.02 (0.12) / 4.58 |
| 0.6 (0.600) | 13.60 (0.19) / 11.49 | 12.65 (0.13) / 14.91 | 11.15 (0.13) / 18.27 | 9.68 (0.17) / 20.33 | 12.08 (0.08) / 18.11 | 17.42 (0.10) / 13.92 | 24.19 (0.15) / 8.91 |
| 0.8 (0.802) | 25.63 (0.37) / 20.24 | 25.84 (0.24) / 26.42 | 23.45 (0.29) / 32.55 | 20.22 (0.44) / 36.51 | 23.03 (0.16) / 33.00 | 32.25 (0.08) / 25.63 | 43.93 (0.11) / 16.43 |
| 0.9 (0.902) | 32.59 (0.53) / 25.86 | 36.72 (0.35) / 34.15 | 35.86 (0.50) / 42.61 | 31.59 (0.81) / 48.53 | 34.58 (0.41) / 44.89 | 44.28 (0.08) / 35.62 | 57.61 (0.06) / 23.32 |
| 0.95 (0.951) | 36.12 (0.61) / 29.08 | 42.81 (0.45) / 38.82 | 46.34 (0.70) / 49.02 | 43.04 (1.26) / 56.69 | 46.20 (0.74) / 53.68 | 55.18 (0.16) / 43.58 | 67.59 (0.05) / 29.20 |

#### K = 2: real caption mixtures (shared_coco_mix<k>_K2_U4, 2000 frames)

| mixture kappa (kappa_bar) | SNR | overlap reduction [95% CI] | receiver reduction [95% CI] | CosSim gen_joint - load_joint [95% CI] | CosSim gen_eq - load_joint [95% CI] |
|---|---|---|---|---|---|
| 0 (0.002) | 0 dB | -0.0006 % [-0.0011, -0.0001] | -0.18 % [-0.20, -0.16] | -0.000002 [-0.000004, +0.000000] | -0.3578 [-0.3611, -0.3543] |
| 0 (0.002) | 10 dB | +0.0057 % [+0.0014, +0.0104] | -0.47 % [-0.52, -0.43] | +0.000002 [-0.000005, +0.000008] | -0.4796 [-0.4823, -0.4770] |
| 0.2 (0.196) | 0 dB | +1.36 % [+1.33, +1.40] | +2.07 % [+1.98, +2.18] | +0.0034 [+0.0034, +0.0035] | -0.2190 [-0.2215, -0.2164] |
| 0.2 (0.196) | 10 dB | +2.06 % [+2.00, +2.12] | +1.56 % [+1.44, +1.69] | +0.0016 [+0.0015, +0.0016] | -0.3321 [-0.3344, -0.3299] |
| 0.4 (0.396) | 0 dB | +5.20 % [+5.11, +5.29] | +8.00 % [+7.83, +8.17] | +0.0119 [+0.0117, +0.0121] | -0.1144 [-0.1162, -0.1125] |
| 0.4 (0.396) | 10 dB | +6.93 % [+6.76, +7.10] | +6.61 % [+6.31, +6.89] | +0.0051 [+0.0050, +0.0052] | -0.2141 [-0.2158, -0.2124] |
| 0.6 (0.600) | 0 dB | +12.16 % [+11.98, +12.34] | +17.43 % [+17.19, +17.69] | +0.0236 [+0.0233, +0.0240] | -0.0334 [-0.0347, -0.0321] |
| 0.6 (0.600) | 10 dB | +15.07 % [+14.75, +15.37] | +14.85 % [+14.35, +15.32] | +0.0100 [+0.0099, +0.0102] | -0.1156 [-0.1167, -0.1145] |
| 0.8 (0.802) | 0 dB | +25.51 % [+25.21, +25.78] | +31.54 % [+31.21, +31.86] | +0.0384 [+0.0378, +0.0390] | +0.0262 [+0.0253, +0.0270] |
| 0.8 (0.802) | 10 dB | +27.54 % [+27.10, +27.95] | +28.57 % [+27.85, +29.23] | +0.0150 [+0.0148, +0.0153] | -0.0335 [-0.0341, -0.0329] |
| 0.9 (0.902) | 0 dB | +38.56 % [+38.20, +38.91] | +41.70 % [+41.34, +42.06] | +0.0476 [+0.0469, +0.0483] | +0.0467 [+0.0459, +0.0474] |
| 0.9 (0.902) | 10 dB | +39.59 % [+39.13, +40.04] | +39.90 % [+39.13, +40.62] | +0.0179 [+0.0177, +0.0182] | +0.0003 [-0.0001, +0.0007] |
| 0.95 (0.951) | 0 dB | +49.52 % [+49.11, +49.91] | +48.23 % [+47.87, +48.60] | +0.0532 [+0.0524, +0.0539] | +0.0533 [+0.0526, +0.0541] |
| 0.95 (0.951) | 10 dB | +51.32 % [+50.87, +51.76] | +48.75 % [+47.98, +49.48] | +0.0196 [+0.0194, +0.0199] | +0.0141 [+0.0138, +0.0144] |

- mixture kappa 0 (fitted kappa_bar about 0.002, design and receiver use the fitted C), K = 4 and 2, every SNR: largest |overlap NMSE reduction| model +0.0154 % (K = 2, 20 dB); captions, largest |CI end| 0.0104 % (K = 2, 10 dB: +0.0057 % [+0.0014, +0.0104]); at K = 4 alone (full load) the largest is 0.0013 %

### C.4 Natural captions: model (Gaussian with the fitted statistics) against the measured reduction of section B


#### coco, K = 4

| U | SNR | kappa_bar | model overlap red. % (SE) | model receiver red. % | measured overlap red. [95% CI] | measured receiver red. [95% CI] | pruned modes (model) | t_bar_c (model) |
|---|---|---|---|---|---|---|---|---|
| U = 2 | -10 dB | 0.3935 | 1.86 (0.06) | 6.77 | +1.87 % [+1.77, +1.97] | +7.02 % [+6.72, +7.37] | 0 | 0.3110 |
| U = 2 | -5 dB | 0.3935 | 0.83 (0.03) | 7.45 | +0.85 % [+0.81, +0.90] | +7.41 % [+7.01, +7.78] | 0 | 0.1489 |
| U = 2 | 0 dB | 0.3935 | 0.30 (0.02) | 7.91 | +0.31 % [+0.29, +0.33] | +7.44 % [+7.01, +7.95] | 0 | 0.0669 |
| U = 2 | 5 dB | 0.3935 | 0.08 (0.01) | 8.19 | +0.07 % [+0.07, +0.08] | +6.71 % [+6.12, +7.28] | 0 | 0.0226 |
| U = 2 | 10 dB | 0.3935 | 0.02 (0.00) | 8.36 | +0.01 % [+0.01, +0.02] | +5.77 % [+4.85, +6.79] | 0 | 0.0072 |
| U = 2 | 15 dB | 0.3935 | 0.00 (0.00) | 9.11 | - | - | 0 | 0.0023 |
| U = 2 | 20 dB | 0.3935 | -0.00 (0.00) | 11.09 | +0.00 % [-0.00, +0.00] | +3.72 % [+2.28, +5.16] | 0 | 0.0007 |
| U = 3 | -10 dB | 0.3935 | 3.43 (0.07) | 10.61 | +3.54 % [+3.42, +3.66] | +9.98 % [+9.66, +10.28] | 0 | 0.3366 |
| U = 3 | -5 dB | 0.3935 | 1.62 (0.05) | 11.64 | +1.66 % [+1.60, +1.72] | +10.57 % [+10.23, +10.93] | 0 | 0.1648 |
| U = 3 | 0 dB | 0.3935 | 0.61 (0.03) | 11.81 | +0.62 % [+0.59, +0.65] | +10.26 % [+9.81, +10.69] | 0 | 0.0779 |
| U = 3 | 5 dB | 0.3935 | 0.18 (0.01) | 11.04 | +0.18 % [+0.17, +0.19] | +9.65 % [+9.13, +10.15] | 0 | 0.0292 |
| U = 3 | 10 dB | 0.3935 | 0.04 (0.00) | 8.87 | +0.04 % [+0.04, +0.04] | +8.09 % [+7.40, +8.88] | 0 | 0.0092 |
| U = 3 | 15 dB | 0.3935 | 0.01 (0.00) | 6.31 | - | - | 0 | 0.0029 |
| U = 3 | 20 dB | 0.3935 | 0.00 (0.00) | 4.04 | +0.00 % [-0.00, +0.00] | +6.45 % [+4.79, +8.03] | 0 | 0.0009 |
| U = 4 | -10 dB | 0.3935 | 4.76 (0.07) | 11.78 | +4.86 % [+4.71, +5.01] | +11.76 % [+11.48, +12.03] | 0 | 0.3464 |
| U = 4 | -5 dB | 0.3935 | 2.33 (0.06) | 12.74 | +2.38 % [+2.31, +2.46] | +12.28 % [+11.97, +12.61] | 0 | 0.1736 |
| U = 4 | 0 dB | 0.3935 | 0.92 (0.04) | 12.99 | +0.93 % [+0.89, +0.97] | +11.52 % [+11.14, +11.87] | 0 | 0.0840 |
| U = 4 | 5 dB | 0.3935 | 0.29 (0.02) | 12.92 | +0.29 % [+0.27, +0.30] | +10.92 % [+10.49, +11.38] | 0 | 0.0344 |
| U = 4 | 10 dB | 0.3935 | 0.06 (0.01) | 12.04 | +0.06 % [+0.06, +0.07] | +9.19 % [+8.62, +9.80] | 0 | 0.0109 |
| U = 4 | 15 dB | 0.3935 | 0.01 (0.00) | 10.42 | - | - | 0 | 0.0034 |
| U = 4 | 20 dB | 0.3935 | 0.00 (0.00) | 8.20 | +0.00 % [+0.00, +0.00] | +5.59 % [+4.72, +6.56] | 0 | 0.0011 |
| U = 5 (overload, pruned Prop. 2) | -10 dB | 0.3935 | 6.02 (0.07) | 13.22 | +6.07 % [+5.89, +6.24] | +12.66 % [+12.38, +12.93] | 0 | 0.3525 |
| U = 5 (overload, pruned Prop. 2) | -5 dB | 0.3935 | 3.03 (0.05) | 14.33 | +3.06 % [+2.97, +3.15] | +13.65 % [+13.36, +13.96] | 0 | 0.1791 |
| U = 5 (overload, pruned Prop. 2) | 0 dB | 0.3935 | 1.23 (0.03) | 14.40 | +1.25 % [+1.20, +1.30] | +13.19 % [+12.84, +13.51] | 0 | 0.0877 |
| U = 5 (overload, pruned Prop. 2) | 5 dB | 0.3935 | 0.42 (0.01) | 14.04 | +0.45 % [+0.42, +0.47] | +12.18 % [+11.76, +12.60] | 536 | 0.0391 |
| U = 5 (overload, pruned Prop. 2) | 10 dB | 0.3935 | 0.31 (0.01) | 12.71 | +0.38 % [+0.36, +0.40] | +9.33 % [+8.89, +9.77] | 763 | 0.0177 |
| U = 5 (overload, pruned Prop. 2) | 15 dB | 0.3935 | 0.41 (0.02) | 10.77 | - | - | 763 | 0.0106 |
| U = 5 (overload, pruned Prop. 2) | 20 dB | 0.3935 | 0.58 (0.02) | 8.35 | +0.96 % [+0.89, +1.03] | +4.74 % [+4.09, +5.48] | 763 | 0.0083 |

#### coco, K = 2

| U | SNR | kappa_bar | model overlap red. % (SE) | model receiver red. % | measured overlap red. [95% CI] | measured receiver red. [95% CI] | pruned modes (model) | t_bar_c (model) |
|---|---|---|---|---|---|---|---|---|
| U = 2 | -10 dB | 0.3935 | 2.62 (0.07) | 6.14 | +2.65 % [+2.50, +2.78] | +6.40 % [+6.13, +6.71] | 0 | 0.4533 |
| U = 2 | -5 dB | 0.3935 | 1.39 (0.05) | 7.09 | +1.45 % [+1.38, +1.52] | +7.15 % [+6.80, +7.49] | 0 | 0.2336 |
| U = 2 | 0 dB | 0.3935 | 0.56 (0.03) | 7.64 | +0.57 % [+0.54, +0.60] | +7.49 % [+7.10, +7.93] | 0 | 0.1099 |
| U = 2 | 5 dB | 0.3935 | 0.19 (0.02) | 8.06 | +0.18 % [+0.16, +0.19] | +7.27 % [+6.76, +7.78] | 0 | 0.0451 |
| U = 2 | 10 dB | 0.3935 | 0.04 (0.01) | 8.24 | +0.04 % [+0.03, +0.05] | +6.48 % [+5.71, +7.25] | 0 | 0.0143 |
| U = 2 | 15 dB | 0.3935 | 0.01 (0.00) | 8.55 | - | - | 0 | 0.0045 |
| U = 2 | 20 dB | 0.3935 | 0.00 (0.00) | 9.74 | +0.00 % [+0.00, +0.00] | +4.50 % [+3.08, +6.02] | 0 | 0.0014 |
| U = 3 (overload, pruned Prop. 2) | -10 dB | 0.3935 | 4.64 (0.06) | 9.51 | +4.85 % [+4.67, +5.03] | +9.15 % [+8.87, +9.44] | 0 | 0.4764 |
| U = 3 (overload, pruned Prop. 2) | -5 dB | 0.3935 | 2.62 (0.06) | 11.13 | +2.70 % [+2.61, +2.79] | +10.39 % [+10.09, +10.71] | 0 | 0.2548 |
| U = 3 (overload, pruned Prop. 2) | 0 dB | 0.3935 | 1.13 (0.04) | 11.80 | +1.12 % [+1.08, +1.17] | +10.44 % [+10.04, +10.82] | 0 | 0.1230 |
| U = 3 (overload, pruned Prop. 2) | 5 dB | 0.3935 | 0.44 (0.02) | 11.54 | +0.45 % [+0.43, +0.48] | +10.00 % [+9.56, +10.44] | 496 | 0.0567 |
| U = 3 (overload, pruned Prop. 2) | 10 dB | 0.3935 | 0.34 (0.01) | 9.49 | +0.38 % [+0.35, +0.40] | +8.23 % [+7.66, +8.87] | 765 | 0.0263 |
| U = 3 (overload, pruned Prop. 2) | 15 dB | 0.3935 | 0.48 (0.02) | 6.52 | - | - | 765 | 0.0170 |
| U = 3 (overload, pruned Prop. 2) | 20 dB | 0.3935 | 0.61 (0.03) | 3.91 | +0.64 % [+0.56, +0.71] | +4.17 % [+3.20, +5.13] | 765 | 0.0142 |
| U = 4 (overload, pruned Prop. 2) | -10 dB | 0.3935 | 6.37 (0.08) | 10.64 | +6.49 % [+6.29, +6.70] | +10.70 % [+10.43, +10.96] | 0 | 0.4906 |
| U = 4 (overload, pruned Prop. 2) | -5 dB | 0.3935 | 3.70 (0.07) | 12.29 | +3.81 % [+3.69, +3.91] | +12.10 % [+11.82, +12.41] | 0 | 0.2666 |
| U = 4 (overload, pruned Prop. 2) | 0 dB | 0.3935 | 1.65 (0.05) | 12.89 | +1.65 % [+1.59, +1.71] | +11.89 % [+11.54, +12.21] | 0 | 0.1303 |
| U = 4 (overload, pruned Prop. 2) | 5 dB | 0.3935 | 0.72 (0.03) | 12.76 | +0.75 % [+0.71, +0.78] | +11.29 % [+10.90, +11.71] | 839 | 0.0619 |
| U = 4 (overload, pruned Prop. 2) | 10 dB | 0.3935 | 0.65 (0.02) | 11.18 | +0.73 % [+0.69, +0.78] | +8.78 % [+8.32, +9.27] | 1532 | 0.0328 |
| U = 4 (overload, pruned Prop. 2) | 15 dB | 0.3935 | 0.79 (0.03) | 8.66 | - | - | 1532 | 0.0224 |
| U = 4 (overload, pruned Prop. 2) | 20 dB | 0.3935 | 0.92 (0.04) | 5.83 | +1.08 % [+0.99, +1.16] | +3.09 % [+2.69, +3.55] | 1532 | 0.0192 |
| U = 5 (overload, pruned Prop. 2) | -10 dB | 0.3935 | 7.93 (0.07) | 11.89 | +8.05 % [+7.82, +8.28] | +11.53 % [+11.27, +11.78] | 0 | 0.5044 |
| U = 5 (overload, pruned Prop. 2) | -5 dB | 0.3935 | 4.73 (0.07) | 13.83 | +4.78 % [+4.64, +4.92] | +13.39 % [+13.12, +13.67] | 0 | 0.2746 |
| U = 5 (overload, pruned Prop. 2) | 0 dB | 0.3935 | 2.16 (0.04) | 14.43 | +2.17 % [+2.10, +2.24] | +13.63 % [+13.31, +13.94] | 0 | 0.1349 |
| U = 5 (overload, pruned Prop. 2) | 5 dB | 0.3935 | 1.00 (0.02) | 13.92 | +1.01 % [+0.97, +1.06] | +12.53 % [+12.15, +12.90] | 1089 | 0.0647 |
| U = 5 (overload, pruned Prop. 2) | 10 dB | 0.3935 | 0.93 (0.02) | 11.72 | +1.03 % [+0.98, +1.08] | +9.03 % [+8.64, +9.40] | 2299 | 0.0350 |
| U = 5 (overload, pruned Prop. 2) | 15 dB | 0.3935 | 1.13 (0.03) | 8.85 | - | - | 2299 | 0.0251 |
| U = 5 (overload, pruned Prop. 2) | 20 dB | 0.3935 | 1.28 (0.03) | 6.05 | +1.46 % [+1.37, +1.56] | +3.27 % [+2.92, +3.66] | 2299 | 0.0222 |

#### flickr, K = 4

| U | SNR | kappa_bar | model overlap red. % (SE) | model receiver red. % | measured overlap red. [95% CI] | measured receiver red. [95% CI] | pruned modes (model) | t_bar_c (model) |
|---|---|---|---|---|---|---|---|---|
| U = 2 | -10 dB | 0.3207 | 1.41 (0.04) | 4.53 | - | - | 0 | 0.2732 |
| U = 2 | -5 dB | 0.3207 | 0.71 (0.03) | 5.06 | - | - | 0 | 0.1439 |
| U = 2 | 0 dB | 0.3207 | 0.29 (0.02) | 5.41 | - | - | 0 | 0.0686 |
| U = 2 | 5 dB | 0.3207 | 0.09 (0.01) | 5.62 | - | - | 0 | 0.0233 |
| U = 2 | 10 dB | 0.3207 | 0.02 (0.00) | 5.74 | - | - | 0 | 0.0074 |
| U = 2 | 15 dB | 0.3207 | 0.00 (0.00) | 6.24 | - | - | 0 | 0.0023 |
| U = 2 | 20 dB | 0.3207 | 0.00 (0.00) | 7.61 | - | - | 0 | 0.0007 |
| U = 3 | -10 dB | 0.3207 | 2.64 (0.04) | 7.26 | - | - | 0 | 0.2930 |
| U = 3 | -5 dB | 0.3207 | 1.42 (0.04) | 8.18 | - | - | 0 | 0.1585 |
| U = 3 | 0 dB | 0.3207 | 0.62 (0.03) | 8.45 | - | - | 0 | 0.0797 |
| U = 3 | 5 dB | 0.3207 | 0.21 (0.02) | 7.97 | - | - | 0 | 0.0304 |
| U = 3 | 10 dB | 0.3207 | 0.05 (0.01) | 6.41 | - | - | 0 | 0.0096 |
| U = 3 | 15 dB | 0.3207 | 0.01 (0.00) | 4.55 | - | - | 0 | 0.0030 |
| U = 3 | 20 dB | 0.3207 | 0.00 (0.00) | 2.90 | - | - | 0 | 0.0010 |
| U = 4 | -10 dB | 0.3207 | 3.73 (0.05) | 8.21 | - | - | 0 | 0.3035 |
| U = 4 | -5 dB | 0.3207 | 2.06 (0.04) | 9.11 | +2.03 % [+1.93, +2.14] | +8.56 % [+8.16, +8.96] | 0 | 0.1663 |
| U = 4 | 0 dB | 0.3207 | 0.93 (0.03) | 9.46 | +0.94 % [+0.89, +0.99] | +8.17 % [+7.75, +8.58] | 0 | 0.0856 |
| U = 4 | 5 dB | 0.3207 | 0.34 (0.02) | 9.52 | - | - | 0 | 0.0360 |
| U = 4 | 10 dB | 0.3207 | 0.09 (0.01) | 8.90 | +0.09 % [+0.08, +0.09] | +6.61 % [+5.97, +7.32] | 0 | 0.0114 |
| U = 4 | 15 dB | 0.3207 | 0.02 (0.00) | 7.71 | - | - | 0 | 0.0036 |
| U = 4 | 20 dB | 0.3207 | 0.00 (0.00) | 6.06 | - | - | 0 | 0.0011 |
| U = 5 (overload, pruned Prop. 2) | -10 dB | 0.3207 | 4.75 (0.05) | 9.34 | - | - | 0 | 0.3094 |
| U = 5 (overload, pruned Prop. 2) | -5 dB | 0.3207 | 2.70 (0.04) | 10.40 | - | - | 0 | 0.1709 |
| U = 5 (overload, pruned Prop. 2) | 0 dB | 0.3207 | 1.24 (0.03) | 10.69 | - | - | 0 | 0.0892 |
| U = 5 (overload, pruned Prop. 2) | 5 dB | 0.3207 | 0.50 (0.02) | 10.55 | - | - | 534 | 0.0412 |
| U = 5 (overload, pruned Prop. 2) | 10 dB | 0.3207 | 0.39 (0.01) | 9.59 | - | - | 763 | 0.0196 |
| U = 5 (overload, pruned Prop. 2) | 15 dB | 0.3207 | 0.52 (0.02) | 8.17 | - | - | 763 | 0.0122 |
| U = 5 (overload, pruned Prop. 2) | 20 dB | 0.3207 | 0.73 (0.03) | 6.40 | - | - | 763 | 0.0097 |

#### flickr, K = 2

| U | SNR | kappa_bar | model overlap red. % (SE) | model receiver red. % | measured overlap red. [95% CI] | measured receiver red. [95% CI] | pruned modes (model) | t_bar_c (model) |
|---|---|---|---|---|---|---|---|---|
| U = 2 | -10 dB | 0.3207 | 1.88 (0.05) | 4.05 | - | - | 0 | 0.3804 |
| U = 2 | -5 dB | 0.3207 | 1.11 (0.03) | 4.77 | - | - | 0 | 0.2143 |
| U = 2 | 0 dB | 0.3207 | 0.51 (0.02) | 5.21 | - | - | 0 | 0.1090 |
| U = 2 | 5 dB | 0.3207 | 0.19 (0.02) | 5.53 | - | - | 0 | 0.0465 |
| U = 2 | 10 dB | 0.3207 | 0.05 (0.01) | 5.66 | - | - | 0 | 0.0148 |
| U = 2 | 15 dB | 0.3207 | 0.01 (0.00) | 5.86 | - | - | 0 | 0.0047 |
| U = 2 | 20 dB | 0.3207 | 0.00 (0.00) | 6.67 | - | - | 0 | 0.0015 |
| U = 3 (overload, pruned Prop. 2) | -10 dB | 0.3207 | 3.40 (0.04) | 6.39 | - | - | 0 | 0.4025 |
| U = 3 (overload, pruned Prop. 2) | -5 dB | 0.3207 | 2.12 (0.04) | 7.71 | - | - | 0 | 0.2319 |
| U = 3 (overload, pruned Prop. 2) | 0 dB | 0.3207 | 1.04 (0.03) | 8.36 | - | - | 0 | 0.1218 |
| U = 3 (overload, pruned Prop. 2) | 5 dB | 0.3207 | 0.48 (0.02) | 8.28 | - | - | 519 | 0.0586 |
| U = 3 (overload, pruned Prop. 2) | 10 dB | 0.3207 | 0.44 (0.02) | 6.83 | - | - | 765 | 0.0296 |
| U = 3 (overload, pruned Prop. 2) | 15 dB | 0.3207 | 0.60 (0.03) | 4.72 | - | - | 765 | 0.0196 |
| U = 3 (overload, pruned Prop. 2) | 20 dB | 0.3207 | 0.77 (0.03) | 2.91 | - | - | 765 | 0.0163 |
| U = 4 (overload, pruned Prop. 2) | -10 dB | 0.3207 | 4.74 (0.06) | 7.29 | - | - | 0 | 0.4082 |
| U = 4 (overload, pruned Prop. 2) | -5 dB | 0.3207 | 3.03 (0.05) | 8.65 | +3.04 % [+2.88, +3.19] | +8.31 % [+7.92, +8.67] | 0 | 0.2409 |
| U = 4 (overload, pruned Prop. 2) | 0 dB | 0.3207 | 1.53 (0.04) | 9.30 | +1.55 % [+1.46, +1.63] | +8.46 % [+8.07, +8.84] | 0 | 0.1285 |
| U = 4 (overload, pruned Prop. 2) | 5 dB | 0.3207 | 0.77 (0.03) | 9.32 | - | - | 850 | 0.0639 |
| U = 4 (overload, pruned Prop. 2) | 10 dB | 0.3207 | 0.75 (0.02) | 8.20 | +0.80 % [+0.75, +0.86] | +6.27 % [+5.76, +6.86] | 1532 | 0.0346 |
| U = 4 (overload, pruned Prop. 2) | 15 dB | 0.3207 | 0.94 (0.04) | 6.40 | - | - | 1532 | 0.0244 |
| U = 4 (overload, pruned Prop. 2) | 20 dB | 0.3207 | 1.11 (0.05) | 4.40 | - | - | 1532 | 0.0214 |
| U = 5 (overload, pruned Prop. 2) | -10 dB | 0.3207 | 5.96 (0.05) | 8.23 | - | - | 0 | 0.4134 |
| U = 5 (overload, pruned Prop. 2) | -5 dB | 0.3207 | 3.90 (0.05) | 9.88 | - | - | 0 | 0.2467 |
| U = 5 (overload, pruned Prop. 2) | 0 dB | 0.3207 | 2.03 (0.04) | 10.59 | - | - | 0 | 0.1326 |
| U = 5 (overload, pruned Prop. 2) | 5 dB | 0.3207 | 1.08 (0.02) | 10.34 | - | - | 1096 | 0.0667 |
| U = 5 (overload, pruned Prop. 2) | 10 dB | 0.3207 | 1.12 (0.02) | 8.75 | - | - | 2299 | 0.0381 |
| U = 5 (overload, pruned Prop. 2) | 15 dB | 0.3207 | 1.38 (0.03) | 6.67 | - | - | 2299 | 0.0286 |
| U = 5 (overload, pruned Prop. 2) | 20 dB | 0.3207 | 1.56 (0.04) | 4.65 | - | - | 2299 | 0.0252 |

## D. Coherence bandwidth and time (Code/pca_csi.py --coh, Fig. 3(b))

AG News, independent content, K = 4 (N = 24 subbands of 128 uses), 2000 frames, k = U (ods = Static ODS-U). Gains AR(1) over the subbands with adjacent correlation rho_f, reported one block earlier with correlation eta (eta = 1: current gains; eta = 0: no information), near-far offsets uniform in +-6 dB where nf = 6. Gain = CosSim of the mode minus CosSim of fixed (the load masks) on the same frames, with the stored paired bootstrap 95% CI over frames (1000 resamples, seed 555 + SNR). Modes: csiq = channel-aware masks; csiqp = csiq with conditional-mean gains; switch = coherence-aware choice; ods = Static ODS-U placed by the same gains; csiqb = channel-aware support with equal values; csiqw = channel-aware support with (12) at unit gain.

Energy per user (Code/pca_csi.py, E = n * SEG for the csiq family): csiq, csiqp, csiqb, csiqw (and switch when it picks csiq) give user u the energy n_u x 128 for its n_u subbands chosen per frame (unit power per occupied use; only the total over the users equals d_s); fixed (the load masks) gives d_s/U to every user and ods d_s/k = d_s/U. So for U >= 2 the csiq gain over the load masks includes energy moved between users; at U = 1 the one user receives all 24 subbands, so both have the energy d_s.

- other coherence files (not parsed here): pilot_corr/pca_csi_coh_rho1.0_eta1.0_U1_genie.json, pilot_corr/pca_csi_coh_rho1.0_eta1.0_U4_genie.json
- check: the stored gain_fixed / gain_fixed_weakest and their CIs recomputed from the npz (float32 per-frame CosSim, default_rng(555 + SNR)): 1764 mode-SNR points, largest |difference| 2.8e-09
- every coherence run has k = U (ods = Static ODS-U)
- files: 98 (nf = 0: 14, nf = 6: 84)

### D.1 Gain over the load masks versus the gain correlation eta, offsets +-6 dB (nf = 6)


#### 0 dB, channel-aware masks (csiq): mean CosSim gain over the load masks [95% CI]

| U, rho_f | eta = 1.0 | eta = 0.9 | eta = 0.7 | eta = 0.5 | eta = 0.3 | eta = 0.0 |
|---|---|---|---|---|---|---|
| U = 1, rho_f = 0.3 | +0.0442 [+0.0427, +0.0456] | +0.0378 [+0.0364, +0.0392] | +0.0177 [+0.0157, +0.0197] | -0.0070 [-0.0101, -0.0039] | -0.0311 [-0.0356, -0.0271] | -0.0470 [-0.0516, -0.0425] |
| U = 1, rho_f = 0.7 | +0.0428 [+0.0413, +0.0443] | +0.0360 [+0.0345, +0.0374] | +0.0137 [+0.0110, +0.0162] | -0.0109 [-0.0145, -0.0076] | -0.0322 [-0.0364, -0.0284] | -0.0435 [-0.0480, -0.0395] |
| U = 1, rho_f = 0.95 | +0.0430 [+0.0411, +0.0451] | +0.0297 [+0.0271, +0.0321] | +0.0060 [+0.0029, +0.0090] | -0.0143 [-0.0180, -0.0103] | -0.0279 [-0.0325, -0.0239] | -0.0374 [-0.0422, -0.0327] |
| U = 2, rho_f = 0.3 | +0.0787 [+0.0766, +0.0809] | +0.0707 [+0.0685, +0.0728] | +0.0430 [+0.0404, +0.0456] | +0.0065 [+0.0033, +0.0095] | -0.0226 [-0.0261, -0.0191] | -0.0418 [-0.0455, -0.0376] |
| U = 2, rho_f = 0.7 | +0.0752 [+0.0731, +0.0773] | +0.0667 [+0.0647, +0.0686] | +0.0371 [+0.0345, +0.0394] | +0.0034 [+0.0001, +0.0067] | -0.0243 [-0.0282, -0.0202] | -0.0434 [-0.0475, -0.0390] |
| U = 2, rho_f = 0.95 | +0.0676 [+0.0653, +0.0698] | +0.0542 [+0.0520, +0.0565] | +0.0223 [+0.0195, +0.0252] | -0.0033 [-0.0066, +0.0000] | -0.0228 [-0.0265, -0.0189] | -0.0310 [-0.0348, -0.0274] |
| U = 4, rho_f = 0.3 | +0.1406 [+0.1375, +0.1439] | +0.1309 [+0.1280, +0.1341] | +0.0937 [+0.0904, +0.0972] | +0.0487 [+0.0450, +0.0528] | +0.0111 [+0.0073, +0.0153] | -0.0099 [-0.0138, -0.0056] |
| U = 4, rho_f = 0.7 | +0.1359 [+0.1327, +0.1392] | +0.1251 [+0.1221, +0.1283] | +0.0898 [+0.0864, +0.0932] | +0.0491 [+0.0452, +0.0528] | +0.0151 [+0.0111, +0.0193] | -0.0051 [-0.0095, -0.0012] |
| U = 4, rho_f = 0.95 | +0.1177 [+0.1146, +0.1209] | +0.1005 [+0.0976, +0.1035] | +0.0615 [+0.0581, +0.0650] | +0.0313 [+0.0275, +0.0350] | +0.0094 [+0.0055, +0.0131] | -0.0042 [-0.0082, -0.0004] |
| U = 6, rho_f = 0.3 | +0.1581 [+0.1554, +0.1608] | +0.1437 [+0.1410, +0.1465] | +0.1036 [+0.1006, +0.1068] | +0.0561 [+0.0529, +0.0596] | +0.0168 [+0.0134, +0.0204] | -0.0069 [-0.0103, -0.0034] |
| U = 6, rho_f = 0.7 | +0.1515 [+0.1487, +0.1541] | +0.1371 [+0.1343, +0.1398] | +0.0976 [+0.0945, +0.1006] | +0.0506 [+0.0469, +0.0539] | +0.0140 [+0.0101, +0.0178] | -0.0071 [-0.0108, -0.0032] |
| U = 6, rho_f = 0.95 | +0.1270 [+0.1247, +0.1298] | +0.1098 [+0.1071, +0.1123] | +0.0696 [+0.0665, +0.0726] | +0.0355 [+0.0320, +0.0387] | +0.0107 [+0.0073, +0.0140] | -0.0032 [-0.0065, +0.0000] |

#### 0 dB, coherence-aware choice (switch): mean CosSim gain over the load masks [95% CI]

| U, rho_f | eta = 1.0 | eta = 0.9 | eta = 0.7 | eta = 0.5 | eta = 0.3 | eta = 0.0 |
|---|---|---|---|---|---|---|
| U = 1, rho_f = 0.3 | +0.0442 [+0.0427, +0.0456] | +0.0378 [+0.0364, +0.0392] | +0.0168 [+0.0148, +0.0188] | +0.0010 [-0.0004, +0.0024] | -0.0004 [-0.0009, +0.0001] | +0.0000 [+0.0000, +0.0000] |
| U = 1, rho_f = 0.7 | +0.0428 [+0.0413, +0.0443] | +0.0360 [+0.0345, +0.0374] | +0.0143 [+0.0119, +0.0165] | -0.0003 [-0.0020, +0.0011] | -0.0001 [-0.0003, +0.0001] | +0.0000 [+0.0000, +0.0000] |
| U = 1, rho_f = 0.95 | +0.0430 [+0.0411, +0.0451] | +0.0296 [+0.0270, +0.0320] | +0.0081 [+0.0058, +0.0106] | +0.0002 [-0.0015, +0.0019] | -0.0001 [-0.0008, +0.0004] | -0.0007 [-0.0017, +0.0000] |
| U = 2, rho_f = 0.3 | +0.0787 [+0.0766, +0.0809] | +0.0707 [+0.0685, +0.0728] | +0.0429 [+0.0403, +0.0455] | +0.0063 [+0.0036, +0.0088] | -0.0014 [-0.0024, -0.0006] | -0.0001 [-0.0002, +0.0000] |
| U = 2, rho_f = 0.7 | +0.0752 [+0.0731, +0.0773] | +0.0667 [+0.0647, +0.0686] | +0.0369 [+0.0344, +0.0393] | +0.0054 [+0.0027, +0.0080] | -0.0011 [-0.0021, -0.0001] | -0.0002 [-0.0006, +0.0001] |
| U = 2, rho_f = 0.95 | +0.0676 [+0.0653, +0.0698] | +0.0542 [+0.0520, +0.0565] | +0.0225 [+0.0199, +0.0253] | +0.0019 [-0.0003, +0.0042] | -0.0010 [-0.0020, +0.0000] | -0.0003 [-0.0005, -0.0001] |
| U = 4, rho_f = 0.3 | +0.1406 [+0.1375, +0.1439] | +0.1309 [+0.1280, +0.1341] | +0.0937 [+0.0904, +0.0972] | +0.0486 [+0.0449, +0.0527] | +0.0082 [+0.0046, +0.0121] | -0.0018 [-0.0038, +0.0002] |
| U = 4, rho_f = 0.7 | +0.1359 [+0.1327, +0.1392] | +0.1251 [+0.1221, +0.1283] | +0.0898 [+0.0864, +0.0932] | +0.0489 [+0.0449, +0.0526] | +0.0112 [+0.0076, +0.0150] | -0.0004 [-0.0026, +0.0016] |
| U = 4, rho_f = 0.95 | +0.1177 [+0.1146, +0.1209] | +0.1005 [+0.0976, +0.1035] | +0.0614 [+0.0581, +0.0650] | +0.0308 [+0.0269, +0.0344] | +0.0071 [+0.0037, +0.0102] | +0.0008 [-0.0017, +0.0032] |
| U = 6, rho_f = 0.3 | +0.1581 [+0.1554, +0.1608] | +0.1437 [+0.1410, +0.1465] | +0.1036 [+0.1006, +0.1068] | +0.0561 [+0.0529, +0.0596] | +0.0152 [+0.0119, +0.0188] | -0.0004 [-0.0021, +0.0013] |
| U = 6, rho_f = 0.7 | +0.1515 [+0.1487, +0.1541] | +0.1371 [+0.1343, +0.1398] | +0.0976 [+0.0945, +0.1006] | +0.0506 [+0.0469, +0.0539] | +0.0133 [+0.0098, +0.0166] | -0.0011 [-0.0030, +0.0007] |
| U = 6, rho_f = 0.95 | +0.1270 [+0.1247, +0.1298] | +0.1098 [+0.1071, +0.1123] | +0.0696 [+0.0665, +0.0726] | +0.0355 [+0.0320, +0.0386] | +0.0100 [+0.0070, +0.0129] | -0.0003 [-0.0023, +0.0018] |

#### 0 dB, csiq with conditional-mean gains (csiqp): mean CosSim gain over the load masks [95% CI]

| U, rho_f | eta = 1.0 | eta = 0.9 | eta = 0.7 | eta = 0.5 | eta = 0.3 | eta = 0.0 |
|---|---|---|---|---|---|---|
| U = 1, rho_f = 0.3 | +0.0442 [+0.0427, +0.0456] | +0.0378 [+0.0364, +0.0392] | +0.0183 [+0.0163, +0.0203] | -0.0047 [-0.0076, -0.0018] | -0.0264 [-0.0307, -0.0226] | -0.0399 [-0.0443, -0.0357] |
| U = 1, rho_f = 0.7 | +0.0428 [+0.0413, +0.0443] | +0.0360 [+0.0346, +0.0374] | +0.0144 [+0.0117, +0.0168] | -0.0088 [-0.0123, -0.0056] | -0.0279 [-0.0318, -0.0243] | -0.0374 [-0.0415, -0.0335] |
| U = 1, rho_f = 0.95 | +0.0430 [+0.0411, +0.0451] | +0.0297 [+0.0270, +0.0321] | +0.0065 [+0.0034, +0.0095] | -0.0128 [-0.0166, -0.0090] | -0.0252 [-0.0297, -0.0213] | -0.0334 [-0.0383, -0.0289] |
| U = 2, rho_f = 0.3 | +0.0787 [+0.0766, +0.0809] | +0.0707 [+0.0685, +0.0728] | +0.0442 [+0.0416, +0.0467] | +0.0104 [+0.0072, +0.0134] | -0.0150 [-0.0184, -0.0117] | -0.0311 [-0.0348, -0.0271] |
| U = 2, rho_f = 0.7 | +0.0752 [+0.0731, +0.0773] | +0.0667 [+0.0647, +0.0687] | +0.0382 [+0.0357, +0.0406] | +0.0070 [+0.0039, +0.0103] | -0.0172 [-0.0211, -0.0133] | -0.0332 [-0.0374, -0.0290] |
| U = 2, rho_f = 0.95 | +0.0676 [+0.0653, +0.0698] | +0.0542 [+0.0520, +0.0565] | +0.0229 [+0.0201, +0.0259] | -0.0008 [-0.0040, +0.0024] | -0.0179 [-0.0217, -0.0142] | -0.0239 [-0.0275, -0.0204] |
| U = 4, rho_f = 0.3 | +0.1406 [+0.1375, +0.1439] | +0.1310 [+0.1281, +0.1342] | +0.0954 [+0.0921, +0.0989] | +0.0540 [+0.0504, +0.0581] | +0.0210 [+0.0173, +0.0251] | +0.0035 [-0.0003, +0.0077] |
| U = 4, rho_f = 0.7 | +0.1359 [+0.1327, +0.1392] | +0.1251 [+0.1222, +0.1284] | +0.0914 [+0.0880, +0.0948] | +0.0543 [+0.0504, +0.0579] | +0.0241 [+0.0203, +0.0283] | +0.0070 [+0.0029, +0.0108] |
| U = 4, rho_f = 0.95 | +0.1177 [+0.1146, +0.1209] | +0.1006 [+0.0976, +0.1036] | +0.0625 [+0.0592, +0.0661] | +0.0348 [+0.0309, +0.0384] | +0.0157 [+0.0120, +0.0193] | +0.0041 [+0.0002, +0.0077] |
| U = 6, rho_f = 0.3 | +0.1581 [+0.1554, +0.1608] | +0.1439 [+0.1412, +0.1467] | +0.1055 [+0.1025, +0.1086] | +0.0616 [+0.0583, +0.0650] | +0.0269 [+0.0236, +0.0304] | +0.0067 [+0.0034, +0.0101] |
| U = 6, rho_f = 0.7 | +0.1515 [+0.1487, +0.1541] | +0.1372 [+0.1344, +0.1399] | +0.0994 [+0.0962, +0.1024] | +0.0559 [+0.0524, +0.0591] | +0.0232 [+0.0194, +0.0269] | +0.0055 [+0.0018, +0.0092] |
| U = 6, rho_f = 0.95 | +0.1270 [+0.1247, +0.1298] | +0.1098 [+0.1070, +0.1124] | +0.0708 [+0.0677, +0.0738] | +0.0391 [+0.0356, +0.0422] | +0.0170 [+0.0135, +0.0203] | +0.0057 [+0.0025, +0.0088] |

#### 0 dB, coherence-aware choice: share of frames choosing csiq; rule_loss [95% CI]; decision_loss; oracle_cf_cos - cos (value of the current gains)

| U, rho_f | eta = 1.0 | eta = 0.9 | eta = 0.7 | eta = 0.5 | eta = 0.3 | eta = 0.0 |
|---|---|---|---|---|---|---|
| U = 1, rho_f = 0.3 | 1.000; +0.0000 [+0.0000, +0.0000]; 0.0000; +0.0000 | 1.000; +0.0000 [+0.0000, +0.0000]; 0.0001; +0.0000 | 0.880; +0.0009 [+0.0005, +0.0013]; 0.0070; +0.0068 | 0.213; -0.0010 [-0.0024, +0.0004]; 0.0133; +0.0130 | 0.009; +0.0004 [-0.0001, +0.0009]; 0.0098; +0.0095 | 0.000; +0.0000 [+0.0000, +0.0000]; 0.0072; +0.0069 |
| U = 1, rho_f = 0.7 | 1.000; +0.0000 [+0.0000, +0.0000]; 0.0000; +0.0000 | 1.000; +0.0000 [+0.0000, +0.0000]; 0.0004; +0.0003 | 0.812; -0.0006 [-0.0016, +0.0002]; 0.0086; +0.0084 | 0.182; +0.0003 [-0.0011, +0.0020]; 0.0145; +0.0141 | 0.007; +0.0001 [-0.0001, +0.0003]; 0.0094; +0.0090 | 0.001; -0.0000 [-0.0000, +0.0000]; 0.0067; +0.0064 |
| U = 1, rho_f = 0.95 | 1.000; +0.0000 [+0.0000, +0.0000]; 0.0000; +0.0000 | 0.993; +0.0000 [-0.0001, +0.0002]; 0.0037; +0.0037 | 0.676; -0.0021 [-0.0039, -0.0004]; 0.0133; +0.0129 | 0.193; -0.0002 [-0.0019, +0.0015]; 0.0146; +0.0142 | 0.030; +0.0001 [-0.0004, +0.0008]; 0.0113; +0.0109 | 0.007; +0.0007 [-0.0000, +0.0017]; 0.0101; +0.0096 |
| U = 2, rho_f = 0.3 | 1.000; +0.0000 [+0.0000, +0.0000]; 0.0000; +0.0000 | 1.000; +0.0000 [+0.0000, +0.0000]; 0.0000; +0.0000 | 0.994; +0.0001 [-0.0000, +0.0003]; 0.0053; +0.0051 | 0.650; +0.0002 [-0.0016, +0.0020]; 0.0224; +0.0218 | 0.064; +0.0014 [+0.0006, +0.0024]; 0.0190; +0.0184 | 0.002; +0.0001 [-0.0000, +0.0002]; 0.0132; +0.0125 |
| U = 2, rho_f = 0.7 | 1.000; +0.0000 [+0.0000, +0.0000]; 0.0000; +0.0000 | 1.000; +0.0000 [+0.0000, +0.0000]; 0.0000; -0.0000 | 0.989; +0.0002 [-0.0001, +0.0005]; 0.0071; +0.0069 | 0.579; -0.0020 [-0.0038, -0.0000]; 0.0219; +0.0214 | 0.052; +0.0011 [+0.0001, +0.0021]; 0.0193; +0.0186 | 0.004; +0.0002 [-0.0001, +0.0006]; 0.0137; +0.0131 |
| U = 2, rho_f = 0.95 | 1.000; +0.0000 [+0.0000, +0.0000]; 0.0000; +0.0000 | 1.000; +0.0000 [+0.0000, +0.0000]; 0.0009; +0.0007 | 0.906; -0.0003 [-0.0012, +0.0006]; 0.0119; +0.0115 | 0.400; -0.0019 [-0.0040, +0.0003]; 0.0214; +0.0205 | 0.051; +0.0010 [-0.0000, +0.0020]; 0.0178; +0.0170 | 0.009; +0.0003 [+0.0001, +0.0005]; 0.0145; +0.0136 |
| U = 4, rho_f = 0.3 | 1.000; +0.0000 [+0.0000, +0.0000]; 0.0000; +0.0000 | 1.000; +0.0000 [+0.0000, +0.0000]; 0.0000; -0.0000 | 1.000; +0.0000 [+0.0000, +0.0000]; 0.0021; +0.0019 | 0.999; +0.0001 [+0.0000, +0.0002]; 0.0128; +0.0120 | 0.812; +0.0029 [+0.0014, +0.0045]; 0.0317; +0.0307 | 0.274; +0.0018 [-0.0002, +0.0038]; 0.0325; +0.0312 |
| U = 4, rho_f = 0.7 | 1.000; +0.0000 [+0.0000, +0.0000]; 0.0000; +0.0000 | 1.000; +0.0000 [+0.0000, +0.0000]; 0.0000; +0.0000 | 1.000; +0.0000 [+0.0000, +0.0000]; 0.0031; +0.0028 | 0.997; +0.0002 [+0.0000, +0.0005]; 0.0139; +0.0130 | 0.793; +0.0038 [+0.0022, +0.0055]; 0.0308; +0.0298 | 0.281; +0.0004 [-0.0016, +0.0026]; 0.0332; +0.0321 |
| U = 4, rho_f = 0.95 | 1.000; +0.0000 [+0.0000, +0.0000]; 0.0000; +0.0000 | 1.000; +0.0000 [+0.0000, +0.0000]; 0.0004; +0.0004 | 1.000; +0.0000 [+0.0000, +0.0000]; 0.0074; +0.0070 | 0.981; +0.0006 [+0.0001, +0.0010]; 0.0182; +0.0173 | 0.717; +0.0023 [+0.0002, +0.0044]; 0.0302; +0.0291 | 0.390; -0.0008 [-0.0032, +0.0017]; 0.0304; +0.0293 |
| U = 6, rho_f = 0.3 | 1.000; +0.0000 [+0.0000, +0.0000]; 0.0000; +0.0000 | 1.000; +0.0000 [+0.0000, +0.0000]; 0.0000; +0.0000 | 1.000; +0.0000 [+0.0000, +0.0000]; 0.0013; +0.0009 | 1.000; +0.0000 [+0.0000, +0.0000]; 0.0102; +0.0092 | 0.870; +0.0016 [+0.0004, +0.0028]; 0.0258; +0.0239 | 0.233; +0.0004 [-0.0013, +0.0021]; 0.0300; +0.0278 |
| U = 6, rho_f = 0.7 | 1.000; +0.0000 [+0.0000, +0.0000]; 0.0000; +0.0000 | 1.000; +0.0000 [+0.0000, +0.0000]; 0.0000; -0.0000 | 1.000; +0.0000 [+0.0000, +0.0000]; 0.0021; +0.0018 | 1.000; +0.0000 [+0.0000, +0.0000]; 0.0110; +0.0099 | 0.847; +0.0007 [-0.0009, +0.0022]; 0.0267; +0.0246 | 0.251; +0.0011 [-0.0007, +0.0030]; 0.0312; +0.0292 |
| U = 6, rho_f = 0.95 | 1.000; +0.0000 [+0.0000, +0.0000]; 0.0000; +0.0000 | 1.000; +0.0000 [+0.0000, +0.0000]; 0.0001; +0.0000 | 1.000; +0.0000 [+0.0000, +0.0000]; 0.0048; +0.0039 | 0.992; +0.0001 [-0.0002, +0.0003]; 0.0144; +0.0125 | 0.785; +0.0007 [-0.0008, +0.0022]; 0.0251; +0.0229 | 0.358; +0.0003 [-0.0018, +0.0023]; 0.0276; +0.0252 |

#### 10 dB, channel-aware masks (csiq): mean CosSim gain over the load masks [95% CI]

| U, rho_f | eta = 1.0 | eta = 0.9 | eta = 0.7 | eta = 0.5 | eta = 0.3 | eta = 0.0 |
|---|---|---|---|---|---|---|
| U = 1, rho_f = 0.3 | +0.0067 [+0.0064, +0.0070] | +0.0055 [+0.0053, +0.0058] | +0.0007 [+0.0001, +0.0012] | -0.0063 [-0.0075, -0.0052] | -0.0141 [-0.0159, -0.0125] | -0.0158 [-0.0176, -0.0141] |
| U = 1, rho_f = 0.7 | +0.0066 [+0.0063, +0.0068] | +0.0052 [+0.0049, +0.0054] | +0.0001 [-0.0006, +0.0006] | -0.0065 [-0.0075, -0.0056] | -0.0124 [-0.0140, -0.0110] | -0.0162 [-0.0180, -0.0144] |
| U = 1, rho_f = 0.95 | +0.0069 [+0.0065, +0.0072] | +0.0042 [+0.0032, +0.0049] | -0.0032 [-0.0045, -0.0021] | -0.0080 [-0.0095, -0.0065] | -0.0117 [-0.0134, -0.0101] | -0.0145 [-0.0167, -0.0125] |
| U = 2, rho_f = 0.3 | +0.0184 [+0.0177, +0.0191] | +0.0166 [+0.0159, +0.0173] | +0.0086 [+0.0077, +0.0094] | -0.0031 [-0.0044, -0.0019] | -0.0126 [-0.0143, -0.0111] | -0.0180 [-0.0199, -0.0162] |
| U = 2, rho_f = 0.7 | +0.0179 [+0.0172, +0.0186] | +0.0155 [+0.0148, +0.0161] | +0.0065 [+0.0054, +0.0076] | -0.0042 [-0.0056, -0.0029] | -0.0131 [-0.0148, -0.0115] | -0.0183 [-0.0202, -0.0165] |
| U = 2, rho_f = 0.95 | +0.0173 [+0.0165, +0.0181] | +0.0126 [+0.0118, +0.0133] | +0.0015 [+0.0004, +0.0028] | -0.0091 [-0.0108, -0.0074] | -0.0146 [-0.0166, -0.0128] | -0.0172 [-0.0189, -0.0154] |
| U = 4, rho_f = 0.3 | +0.0526 [+0.0509, +0.0543] | +0.0502 [+0.0484, +0.0519] | +0.0372 [+0.0356, +0.0389] | +0.0204 [+0.0185, +0.0222] | +0.0060 [+0.0039, +0.0081] | -0.0047 [-0.0068, -0.0026] |
| U = 4, rho_f = 0.7 | +0.0522 [+0.0506, +0.0538] | +0.0479 [+0.0462, +0.0496] | +0.0352 [+0.0334, +0.0368] | +0.0182 [+0.0164, +0.0202] | +0.0048 [+0.0028, +0.0067] | -0.0036 [-0.0057, -0.0015] |
| U = 4, rho_f = 0.95 | +0.0456 [+0.0440, +0.0471] | +0.0394 [+0.0377, +0.0411] | +0.0240 [+0.0222, +0.0256] | +0.0102 [+0.0084, +0.0119] | +0.0025 [+0.0006, +0.0046] | -0.0011 [-0.0030, +0.0009] |
| U = 6, rho_f = 0.3 | +0.0595 [+0.0580, +0.0610] | +0.0563 [+0.0547, +0.0578] | +0.0416 [+0.0401, +0.0432] | +0.0223 [+0.0205, +0.0241] | +0.0078 [+0.0059, +0.0098] | -0.0036 [-0.0056, -0.0016] |
| U = 6, rho_f = 0.7 | +0.0588 [+0.0572, +0.0605] | +0.0525 [+0.0511, +0.0539] | +0.0378 [+0.0363, +0.0393] | +0.0199 [+0.0182, +0.0217] | +0.0054 [+0.0036, +0.0074] | -0.0039 [-0.0059, -0.0020] |
| U = 6, rho_f = 0.95 | +0.0505 [+0.0491, +0.0519] | +0.0428 [+0.0414, +0.0443] | +0.0261 [+0.0245, +0.0274] | +0.0121 [+0.0103, +0.0138] | +0.0031 [+0.0013, +0.0049] | -0.0035 [-0.0053, -0.0014] |

#### 10 dB, coherence-aware choice (switch): mean CosSim gain over the load masks [95% CI]

| U, rho_f | eta = 1.0 | eta = 0.9 | eta = 0.7 | eta = 0.5 | eta = 0.3 | eta = 0.0 |
|---|---|---|---|---|---|---|
| U = 1, rho_f = 0.3 | +0.0067 [+0.0064, +0.0070] | +0.0055 [+0.0053, +0.0058] | +0.0011 [+0.0008, +0.0014] | -0.0001 [-0.0002, +0.0000] | +0.0000 [+0.0000, +0.0000] | +0.0000 [+0.0000, +0.0000] |
| U = 1, rho_f = 0.7 | +0.0066 [+0.0063, +0.0068] | +0.0052 [+0.0049, +0.0054] | +0.0006 [+0.0001, +0.0009] | -0.0000 [-0.0001, +0.0001] | +0.0000 [+0.0000, +0.0000] | +0.0000 [+0.0000, +0.0000] |
| U = 1, rho_f = 0.95 | +0.0069 [+0.0065, +0.0072] | +0.0040 [+0.0031, +0.0047] | -0.0001 [-0.0006, +0.0005] | -0.0001 [-0.0004, +0.0001] | -0.0000 [-0.0000, +0.0000] | -0.0000 [-0.0001, +0.0000] |
| U = 2, rho_f = 0.3 | +0.0184 [+0.0177, +0.0191] | +0.0166 [+0.0159, +0.0173] | +0.0085 [+0.0076, +0.0093] | -0.0000 [-0.0007, +0.0006] | -0.0001 [-0.0002, +0.0000] | -0.0000 [-0.0000, +0.0000] |
| U = 2, rho_f = 0.7 | +0.0179 [+0.0172, +0.0186] | +0.0155 [+0.0148, +0.0161] | +0.0065 [+0.0055, +0.0075] | -0.0008 [-0.0015, -0.0000] | -0.0001 [-0.0002, -0.0000] | -0.0000 [-0.0000, +0.0000] |
| U = 2, rho_f = 0.95 | +0.0173 [+0.0165, +0.0181] | +0.0126 [+0.0119, +0.0134] | +0.0022 [+0.0012, +0.0032] | -0.0002 [-0.0007, +0.0002] | -0.0004 [-0.0009, -0.0000] | -0.0000 [-0.0001, +0.0000] |
| U = 4, rho_f = 0.3 | +0.0526 [+0.0509, +0.0543] | +0.0502 [+0.0484, +0.0519] | +0.0372 [+0.0356, +0.0389] | +0.0203 [+0.0183, +0.0221] | +0.0046 [+0.0027, +0.0063] | -0.0015 [-0.0027, -0.0003] |
| U = 4, rho_f = 0.7 | +0.0522 [+0.0506, +0.0538] | +0.0479 [+0.0462, +0.0496] | +0.0352 [+0.0334, +0.0368] | +0.0181 [+0.0163, +0.0201] | +0.0029 [+0.0013, +0.0045] | -0.0013 [-0.0025, -0.0002] |
| U = 4, rho_f = 0.95 | +0.0456 [+0.0440, +0.0471] | +0.0394 [+0.0377, +0.0411] | +0.0240 [+0.0222, +0.0256] | +0.0097 [+0.0079, +0.0113] | +0.0015 [+0.0000, +0.0030] | -0.0005 [-0.0017, +0.0007] |
| U = 6, rho_f = 0.3 | +0.0595 [+0.0580, +0.0610] | +0.0563 [+0.0547, +0.0578] | +0.0416 [+0.0401, +0.0432] | +0.0223 [+0.0205, +0.0241] | +0.0063 [+0.0046, +0.0081] | -0.0009 [-0.0020, +0.0001] |
| U = 6, rho_f = 0.7 | +0.0588 [+0.0572, +0.0605] | +0.0525 [+0.0511, +0.0539] | +0.0378 [+0.0363, +0.0393] | +0.0199 [+0.0182, +0.0217] | +0.0044 [+0.0027, +0.0062] | -0.0009 [-0.0019, +0.0002] |
| U = 6, rho_f = 0.95 | +0.0505 [+0.0491, +0.0519] | +0.0428 [+0.0414, +0.0443] | +0.0261 [+0.0245, +0.0274] | +0.0119 [+0.0101, +0.0136] | +0.0026 [+0.0011, +0.0041] | -0.0011 [-0.0024, +0.0002] |

#### 10 dB, csiq with conditional-mean gains (csiqp): mean CosSim gain over the load masks [95% CI]

| U, rho_f | eta = 1.0 | eta = 0.9 | eta = 0.7 | eta = 0.5 | eta = 0.3 | eta = 0.0 |
|---|---|---|---|---|---|---|
| U = 1, rho_f = 0.3 | +0.0067 [+0.0064, +0.0070] | +0.0055 [+0.0053, +0.0058] | +0.0007 [+0.0001, +0.0012] | -0.0061 [-0.0072, -0.0050] | -0.0133 [-0.0150, -0.0118] | -0.0147 [-0.0164, -0.0132] |
| U = 1, rho_f = 0.7 | +0.0066 [+0.0063, +0.0068] | +0.0052 [+0.0049, +0.0054] | +0.0001 [-0.0006, +0.0007] | -0.0062 [-0.0072, -0.0053] | -0.0118 [-0.0133, -0.0104] | -0.0153 [-0.0170, -0.0136] |
| U = 1, rho_f = 0.95 | +0.0069 [+0.0065, +0.0072] | +0.0042 [+0.0032, +0.0049] | -0.0031 [-0.0043, -0.0020] | -0.0077 [-0.0092, -0.0063] | -0.0113 [-0.0129, -0.0098] | -0.0139 [-0.0160, -0.0119] |
| U = 2, rho_f = 0.3 | +0.0184 [+0.0177, +0.0191] | +0.0166 [+0.0159, +0.0173] | +0.0087 [+0.0078, +0.0094] | -0.0024 [-0.0036, -0.0012] | -0.0109 [-0.0125, -0.0095] | -0.0156 [-0.0174, -0.0139] |
| U = 2, rho_f = 0.7 | +0.0179 [+0.0172, +0.0186] | +0.0155 [+0.0148, +0.0161] | +0.0067 [+0.0056, +0.0077] | -0.0035 [-0.0048, -0.0022] | -0.0116 [-0.0132, -0.0101] | -0.0161 [-0.0178, -0.0143] |
| U = 2, rho_f = 0.95 | +0.0173 [+0.0165, +0.0181] | +0.0126 [+0.0118, +0.0133] | +0.0017 [+0.0006, +0.0030] | -0.0083 [-0.0099, -0.0067] | -0.0131 [-0.0149, -0.0113] | -0.0152 [-0.0168, -0.0135] |
| U = 4, rho_f = 0.3 | +0.0526 [+0.0509, +0.0543] | +0.0502 [+0.0484, +0.0519] | +0.0376 [+0.0360, +0.0392] | +0.0221 [+0.0202, +0.0239] | +0.0094 [+0.0074, +0.0113] | +0.0003 [-0.0018, +0.0023] |
| U = 4, rho_f = 0.7 | +0.0522 [+0.0506, +0.0538] | +0.0480 [+0.0463, +0.0496] | +0.0356 [+0.0338, +0.0372] | +0.0197 [+0.0179, +0.0216] | +0.0079 [+0.0061, +0.0099] | +0.0009 [-0.0011, +0.0029] |
| U = 4, rho_f = 0.95 | +0.0456 [+0.0440, +0.0471] | +0.0394 [+0.0377, +0.0411] | +0.0243 [+0.0226, +0.0259] | +0.0113 [+0.0095, +0.0130] | +0.0045 [+0.0026, +0.0065] | +0.0015 [-0.0005, +0.0035] |
| U = 6, rho_f = 0.3 | +0.0595 [+0.0580, +0.0610] | +0.0563 [+0.0547, +0.0579] | +0.0421 [+0.0406, +0.0436] | +0.0240 [+0.0221, +0.0257] | +0.0111 [+0.0092, +0.0131] | +0.0011 [-0.0009, +0.0030] |
| U = 6, rho_f = 0.7 | +0.0588 [+0.0572, +0.0605] | +0.0525 [+0.0512, +0.0539] | +0.0382 [+0.0368, +0.0398] | +0.0214 [+0.0198, +0.0232] | +0.0086 [+0.0067, +0.0104] | +0.0006 [-0.0013, +0.0025] |
| U = 6, rho_f = 0.95 | +0.0505 [+0.0491, +0.0519] | +0.0428 [+0.0414, +0.0443] | +0.0264 [+0.0249, +0.0278] | +0.0132 [+0.0115, +0.0149] | +0.0050 [+0.0033, +0.0067] | -0.0009 [-0.0027, +0.0011] |

#### 10 dB, coherence-aware choice: share of frames choosing csiq; rule_loss [95% CI]; decision_loss; oracle_cf_cos - cos (value of the current gains)

| U, rho_f | eta = 1.0 | eta = 0.9 | eta = 0.7 | eta = 0.5 | eta = 0.3 | eta = 0.0 |
|---|---|---|---|---|---|---|
| U = 1, rho_f = 0.3 | 1.000; +0.0000 [+0.0000, +0.0000]; 0.0000; +0.0000 | 1.000; +0.0000 [+0.0000, +0.0000]; 0.0000; +0.0000 | 0.621; -0.0005 [-0.0010, -0.0000]; 0.0017; +0.0017 | 0.021; +0.0001 [-0.0000, +0.0002]; 0.0014; +0.0013 | 0.000; +0.0000 [+0.0000, +0.0000]; 0.0007; +0.0007 | 0.000; +0.0000 [+0.0000, +0.0000]; 0.0005; +0.0005 |
| U = 1, rho_f = 0.7 | 1.000; +0.0000 [+0.0000, +0.0000]; 0.0000; +0.0000 | 0.999; +0.0000 [+0.0000, +0.0000]; 0.0001; +0.0001 | 0.529; -0.0005 [-0.0008, -0.0001]; 0.0022; +0.0021 | 0.029; +0.0000 [-0.0001, +0.0001]; 0.0014; +0.0014 | 0.000; +0.0000 [+0.0000, +0.0000]; 0.0008; +0.0008 | 0.000; +0.0000 [+0.0000, +0.0000]; 0.0005; +0.0005 |
| U = 1, rho_f = 0.95 | 1.000; +0.0000 [+0.0000, +0.0000]; 0.0000; +0.0000 | 0.965; +0.0002 [+0.0001, +0.0004]; 0.0012; +0.0012 | 0.374; +0.0001 [-0.0005, +0.0006]; 0.0030; +0.0030 | 0.046; +0.0001 [-0.0001, +0.0004]; 0.0020; +0.0019 | 0.003; +0.0000 [-0.0000, +0.0000]; 0.0012; +0.0012 | 0.001; +0.0000 [+0.0000, +0.0001]; 0.0010; +0.0010 |
| U = 2, rho_f = 0.3 | 1.000; +0.0000 [+0.0000, +0.0000]; 0.0000; +0.0000 | 1.000; +0.0000 [+0.0000, +0.0000]; 0.0000; +0.0000 | 0.957; +0.0001 [-0.0001, +0.0003]; 0.0020; +0.0019 | 0.288; +0.0000 [-0.0006, +0.0007]; 0.0059; +0.0058 | 0.008; +0.0001 [-0.0000, +0.0002]; 0.0036; +0.0035 | 0.001; +0.0000 [+0.0000, +0.0000]; 0.0030; +0.0029 |
| U = 2, rho_f = 0.7 | 1.000; +0.0000 [+0.0000, +0.0000]; 0.0000; +0.0000 | 1.000; +0.0000 [+0.0000, +0.0000]; 0.0000; +0.0000 | 0.911; -0.0000 [-0.0004, +0.0003]; 0.0031; +0.0030 | 0.243; +0.0008 [+0.0000, +0.0015]; 0.0062; +0.0061 | 0.009; +0.0001 [+0.0000, +0.0002]; 0.0036; +0.0035 | 0.001; +0.0000 [+0.0000, +0.0000]; 0.0028; +0.0027 |
| U = 2, rho_f = 0.95 | 1.000; +0.0000 [+0.0000, +0.0000]; 0.0000; +0.0000 | 0.997; -0.0000 [-0.0001, +0.0000]; 0.0005; +0.0005 | 0.682; -0.0006 [-0.0014, +0.0001]; 0.0055; +0.0054 | 0.128; +0.0002 [-0.0002, +0.0007]; 0.0051; +0.0050 | 0.012; +0.0004 [+0.0000, +0.0009]; 0.0041; +0.0040 | 0.003; +0.0000 [-0.0000, +0.0001]; 0.0031; +0.0030 |
| U = 4, rho_f = 0.3 | 1.000; +0.0000 [+0.0000, +0.0000]; 0.0000; +0.0000 | 1.000; +0.0000 [+0.0000, +0.0000]; 0.0000; +0.0000 | 1.000; +0.0000 [+0.0000, +0.0000]; 0.0011; +0.0011 | 0.991; +0.0002 [+0.0001, +0.0003]; 0.0068; +0.0067 | 0.735; +0.0015 [+0.0004, +0.0026]; 0.0145; +0.0144 | 0.311; +0.0015 [+0.0003, +0.0027]; 0.0156; +0.0154 |
| U = 4, rho_f = 0.7 | 1.000; +0.0000 [+0.0000, +0.0000]; 0.0000; +0.0000 | 1.000; +0.0000 [+0.0000, +0.0000]; 0.0000; +0.0000 | 1.000; +0.0000 [+0.0000, +0.0000]; 0.0014; +0.0014 | 0.992; +0.0001 [-0.0000, +0.0003]; 0.0066; +0.0065 | 0.734; +0.0018 [+0.0009, +0.0029]; 0.0145; +0.0144 | 0.325; +0.0013 [+0.0002, +0.0025]; 0.0155; +0.0153 |
| U = 4, rho_f = 0.95 | 1.000; +0.0000 [+0.0000, +0.0000]; 0.0000; +0.0000 | 1.000; +0.0000 [+0.0000, +0.0000]; 0.0006; +0.0006 | 0.999; -0.0000 [-0.0001, +0.0000]; 0.0035; +0.0034 | 0.942; +0.0005 [+0.0000, +0.0009]; 0.0092; +0.0090 | 0.677; +0.0011 [-0.0001, +0.0022]; 0.0138; +0.0136 | 0.433; +0.0005 [-0.0005, +0.0017]; 0.0141; +0.0140 |
| U = 6, rho_f = 0.3 | 1.000; +0.0000 [+0.0000, +0.0000]; 0.0000; +0.0000 | 1.000; +0.0000 [+0.0000, +0.0000]; 0.0000; +0.0000 | 1.000; +0.0000 [+0.0000, +0.0000]; 0.0008; +0.0007 | 0.999; +0.0000 [-0.0000, +0.0000]; 0.0055; +0.0052 | 0.805; +0.0015 [+0.0007, +0.0022]; 0.0133; +0.0129 | 0.283; +0.0009 [-0.0001, +0.0020]; 0.0156; +0.0152 |
| U = 6, rho_f = 0.7 | 1.000; +0.0000 [+0.0000, +0.0000]; 0.0000; +0.0000 | 1.000; +0.0000 [+0.0000, +0.0000]; 0.0000; +0.0000 | 1.000; +0.0000 [+0.0000, +0.0000]; 0.0012; +0.0011 | 0.998; +0.0000 [-0.0000, +0.0001]; 0.0059; +0.0058 | 0.782; +0.0011 [+0.0002, +0.0018]; 0.0140; +0.0136 | 0.301; +0.0009 [-0.0002, +0.0019]; 0.0151; +0.0148 |
| U = 6, rho_f = 0.95 | 1.000; +0.0000 [+0.0000, +0.0000]; 0.0000; +0.0000 | 1.000; +0.0000 [+0.0000, +0.0000]; 0.0002; +0.0001 | 1.000; +0.0000 [+0.0000, +0.0000]; 0.0035; +0.0033 | 0.969; +0.0002 [-0.0000, +0.0005]; 0.0087; +0.0084 | 0.703; +0.0005 [-0.0004, +0.0014]; 0.0138; +0.0136 | 0.392; +0.0011 [-0.0002, +0.0024]; 0.0144; +0.0140 |

#### 20 dB, channel-aware masks (csiq): mean CosSim gain over the load masks [95% CI]

| U, rho_f | eta = 1.0 | eta = 0.9 | eta = 0.7 | eta = 0.5 | eta = 0.3 | eta = 0.0 |
|---|---|---|---|---|---|---|
| U = 1, rho_f = 0.3 | +0.0007 [+0.0007, +0.0008] | +0.0006 [+0.0006, +0.0006] | -0.0001 [-0.0002, -0.0000] | -0.0015 [-0.0023, -0.0011] | -0.0030 [-0.0039, -0.0022] | -0.0032 [-0.0036, -0.0028] |
| U = 1, rho_f = 0.7 | +0.0007 [+0.0007, +0.0007] | +0.0006 [+0.0005, +0.0006] | -0.0004 [-0.0007, -0.0003] | -0.0019 [-0.0024, -0.0015] | -0.0024 [-0.0030, -0.0020] | -0.0033 [-0.0040, -0.0027] |
| U = 1, rho_f = 0.95 | +0.0008 [+0.0007, +0.0008] | +0.0004 [+0.0003, +0.0005] | -0.0010 [-0.0015, -0.0006] | -0.0018 [-0.0024, -0.0014] | -0.0020 [-0.0024, -0.0017] | -0.0030 [-0.0037, -0.0023] |
| U = 2, rho_f = 0.3 | +0.0024 [+0.0023, +0.0025] | +0.0021 [+0.0020, +0.0022] | +0.0008 [+0.0006, +0.0009] | -0.0014 [-0.0019, -0.0010] | -0.0033 [-0.0039, -0.0027] | -0.0049 [-0.0056, -0.0043] |
| U = 2, rho_f = 0.7 | +0.0023 [+0.0022, +0.0024] | +0.0020 [+0.0019, +0.0021] | +0.0005 [+0.0003, +0.0007] | -0.0017 [-0.0022, -0.0013] | -0.0033 [-0.0039, -0.0029] | -0.0043 [-0.0049, -0.0037] |
| U = 2, rho_f = 0.95 | +0.0023 [+0.0021, +0.0024] | +0.0016 [+0.0015, +0.0018] | -0.0004 [-0.0008, -0.0001] | -0.0021 [-0.0026, -0.0018] | -0.0032 [-0.0038, -0.0027] | -0.0047 [-0.0055, -0.0040] |
| U = 4, rho_f = 0.3 | +0.0117 [+0.0111, +0.0124] | +0.0109 [+0.0104, +0.0116] | +0.0086 [+0.0080, +0.0092] | +0.0044 [+0.0037, +0.0051] | +0.0007 [-0.0001, +0.0015] | -0.0013 [-0.0021, -0.0004] |
| U = 4, rho_f = 0.7 | +0.0114 [+0.0109, +0.0121] | +0.0105 [+0.0099, +0.0111] | +0.0078 [+0.0071, +0.0084] | +0.0039 [+0.0032, +0.0046] | +0.0012 [+0.0005, +0.0020] | -0.0008 [-0.0017, -0.0001] |
| U = 4, rho_f = 0.95 | +0.0104 [+0.0098, +0.0110] | +0.0090 [+0.0084, +0.0097] | +0.0052 [+0.0045, +0.0060] | +0.0024 [+0.0017, +0.0032] | +0.0007 [-0.0001, +0.0015] | -0.0006 [-0.0015, +0.0002] |
| U = 6, rho_f = 0.3 | +0.0131 [+0.0126, +0.0136] | +0.0122 [+0.0116, +0.0128] | +0.0096 [+0.0090, +0.0103] | +0.0049 [+0.0043, +0.0056] | +0.0009 [+0.0002, +0.0016] | -0.0018 [-0.0026, -0.0010] |
| U = 6, rho_f = 0.7 | +0.0131 [+0.0125, +0.0138] | +0.0118 [+0.0113, +0.0124] | +0.0094 [+0.0087, +0.0100] | +0.0045 [+0.0038, +0.0052] | +0.0009 [+0.0001, +0.0016] | -0.0005 [-0.0012, +0.0002] |
| U = 6, rho_f = 0.95 | +0.0119 [+0.0113, +0.0125] | +0.0100 [+0.0095, +0.0106] | +0.0060 [+0.0054, +0.0067] | +0.0030 [+0.0023, +0.0038] | +0.0009 [+0.0002, +0.0017] | -0.0008 [-0.0015, -0.0001] |

#### 20 dB, coherence-aware choice (switch): mean CosSim gain over the load masks [95% CI]

| U, rho_f | eta = 1.0 | eta = 0.9 | eta = 0.7 | eta = 0.5 | eta = 0.3 | eta = 0.0 |
|---|---|---|---|---|---|---|
| U = 1, rho_f = 0.3 | +0.0007 [+0.0007, +0.0008] | +0.0006 [+0.0006, +0.0006] | +0.0000 [-0.0000, +0.0001] | -0.0000 [-0.0000, +0.0000] | +0.0000 [+0.0000, +0.0000] | +0.0000 [+0.0000, +0.0000] |
| U = 1, rho_f = 0.7 | +0.0007 [+0.0007, +0.0007] | +0.0006 [+0.0005, +0.0006] | -0.0000 [-0.0001, +0.0000] | +0.0000 [-0.0000, +0.0000] | +0.0000 [+0.0000, +0.0000] | +0.0000 [+0.0000, +0.0000] |
| U = 1, rho_f = 0.95 | +0.0008 [+0.0007, +0.0008] | +0.0004 [+0.0003, +0.0005] | -0.0003 [-0.0006, -0.0000] | -0.0000 [-0.0000, +0.0000] | -0.0000 [-0.0000, +0.0000] | +0.0000 [+0.0000, +0.0000] |
| U = 2, rho_f = 0.3 | +0.0024 [+0.0023, +0.0025] | +0.0021 [+0.0020, +0.0022] | +0.0008 [+0.0007, +0.0009] | -0.0001 [-0.0002, +0.0000] | -0.0000 [-0.0000, +0.0000] | -0.0000 [-0.0000, +0.0000] |
| U = 2, rho_f = 0.7 | +0.0023 [+0.0022, +0.0024] | +0.0020 [+0.0019, +0.0021] | +0.0005 [+0.0004, +0.0006] | -0.0001 [-0.0001, -0.0000] | -0.0000 [-0.0000, +0.0000] | +0.0000 [+0.0000, +0.0000] |
| U = 2, rho_f = 0.95 | +0.0023 [+0.0021, +0.0024] | +0.0016 [+0.0015, +0.0018] | +0.0000 [-0.0003, +0.0002] | -0.0001 [-0.0001, -0.0000] | -0.0000 [-0.0000, +0.0000] | -0.0000 [-0.0000, +0.0000] |
| U = 4, rho_f = 0.3 | +0.0117 [+0.0111, +0.0124] | +0.0109 [+0.0104, +0.0116] | +0.0086 [+0.0080, +0.0092] | +0.0042 [+0.0035, +0.0048] | +0.0004 [-0.0002, +0.0009] | -0.0005 [-0.0010, +0.0000] |
| U = 4, rho_f = 0.7 | +0.0114 [+0.0109, +0.0121] | +0.0105 [+0.0099, +0.0111] | +0.0078 [+0.0071, +0.0084] | +0.0036 [+0.0029, +0.0044] | +0.0008 [+0.0001, +0.0015] | -0.0003 [-0.0008, +0.0001] |
| U = 4, rho_f = 0.95 | +0.0104 [+0.0098, +0.0110] | +0.0090 [+0.0084, +0.0097] | +0.0051 [+0.0044, +0.0058] | +0.0020 [+0.0013, +0.0027] | +0.0002 [-0.0004, +0.0008] | +0.0001 [-0.0005, +0.0006] |
| U = 6, rho_f = 0.3 | +0.0131 [+0.0126, +0.0136] | +0.0122 [+0.0116, +0.0128] | +0.0096 [+0.0090, +0.0103] | +0.0049 [+0.0043, +0.0055] | +0.0009 [+0.0003, +0.0015] | -0.0002 [-0.0005, +0.0002] |
| U = 6, rho_f = 0.7 | +0.0131 [+0.0125, +0.0138] | +0.0118 [+0.0113, +0.0124] | +0.0094 [+0.0087, +0.0100] | +0.0045 [+0.0038, +0.0052] | +0.0007 [+0.0001, +0.0013] | -0.0005 [-0.0009, -0.0001] |
| U = 6, rho_f = 0.95 | +0.0119 [+0.0113, +0.0125] | +0.0100 [+0.0095, +0.0106] | +0.0060 [+0.0053, +0.0067] | +0.0029 [+0.0023, +0.0035] | +0.0005 [-0.0001, +0.0011] | -0.0003 [-0.0008, +0.0002] |

#### 20 dB, csiq with conditional-mean gains (csiqp): mean CosSim gain over the load masks [95% CI]

| U, rho_f | eta = 1.0 | eta = 0.9 | eta = 0.7 | eta = 0.5 | eta = 0.3 | eta = 0.0 |
|---|---|---|---|---|---|---|
| U = 1, rho_f = 0.3 | +0.0007 [+0.0007, +0.0008] | +0.0006 [+0.0006, +0.0006] | -0.0001 [-0.0002, -0.0000] | -0.0015 [-0.0020, -0.0011] | -0.0029 [-0.0038, -0.0022] | -0.0031 [-0.0035, -0.0027] |
| U = 1, rho_f = 0.7 | +0.0007 [+0.0007, +0.0007] | +0.0006 [+0.0005, +0.0006] | -0.0004 [-0.0007, -0.0003] | -0.0018 [-0.0023, -0.0014] | -0.0023 [-0.0029, -0.0020] | -0.0031 [-0.0038, -0.0026] |
| U = 1, rho_f = 0.95 | +0.0008 [+0.0007, +0.0008] | +0.0004 [+0.0003, +0.0005] | -0.0010 [-0.0014, -0.0006] | -0.0018 [-0.0024, -0.0014] | -0.0020 [-0.0023, -0.0017] | -0.0029 [-0.0036, -0.0023] |
| U = 2, rho_f = 0.3 | +0.0024 [+0.0023, +0.0025] | +0.0021 [+0.0020, +0.0022] | +0.0008 [+0.0006, +0.0009] | -0.0014 [-0.0019, -0.0009] | -0.0031 [-0.0037, -0.0026] | -0.0046 [-0.0051, -0.0040] |
| U = 2, rho_f = 0.7 | +0.0023 [+0.0022, +0.0024] | +0.0020 [+0.0019, +0.0021] | +0.0005 [+0.0003, +0.0006] | -0.0017 [-0.0021, -0.0013] | -0.0031 [-0.0036, -0.0027] | -0.0040 [-0.0045, -0.0035] |
| U = 2, rho_f = 0.95 | +0.0023 [+0.0021, +0.0024] | +0.0016 [+0.0015, +0.0018] | -0.0004 [-0.0008, -0.0001] | -0.0020 [-0.0025, -0.0017] | -0.0031 [-0.0036, -0.0025] | -0.0044 [-0.0051, -0.0037] |
| U = 4, rho_f = 0.3 | +0.0117 [+0.0111, +0.0124] | +0.0110 [+0.0104, +0.0116] | +0.0086 [+0.0080, +0.0093] | +0.0048 [+0.0041, +0.0054] | +0.0014 [+0.0006, +0.0022] | -0.0001 [-0.0009, +0.0007] |
| U = 4, rho_f = 0.7 | +0.0114 [+0.0109, +0.0121] | +0.0105 [+0.0099, +0.0111] | +0.0079 [+0.0072, +0.0085] | +0.0042 [+0.0035, +0.0049] | +0.0019 [+0.0011, +0.0027] | +0.0002 [-0.0007, +0.0009] |
| U = 4, rho_f = 0.95 | +0.0104 [+0.0098, +0.0110] | +0.0091 [+0.0085, +0.0097] | +0.0054 [+0.0047, +0.0061] | +0.0028 [+0.0020, +0.0035] | +0.0012 [+0.0004, +0.0020] | -0.0000 [-0.0009, +0.0008] |
| U = 6, rho_f = 0.3 | +0.0131 [+0.0126, +0.0136] | +0.0122 [+0.0116, +0.0128] | +0.0097 [+0.0092, +0.0104] | +0.0053 [+0.0048, +0.0060] | +0.0018 [+0.0011, +0.0025] | -0.0006 [-0.0013, +0.0002] |
| U = 6, rho_f = 0.7 | +0.0131 [+0.0125, +0.0138] | +0.0118 [+0.0113, +0.0124] | +0.0095 [+0.0089, +0.0102] | +0.0048 [+0.0042, +0.0056] | +0.0016 [+0.0009, +0.0023] | +0.0006 [-0.0001, +0.0013] |
| U = 6, rho_f = 0.95 | +0.0119 [+0.0113, +0.0125] | +0.0101 [+0.0095, +0.0107] | +0.0063 [+0.0056, +0.0070] | +0.0034 [+0.0027, +0.0041] | +0.0015 [+0.0008, +0.0022] | -0.0000 [-0.0007, +0.0006] |

#### 20 dB, coherence-aware choice: share of frames choosing csiq; rule_loss [95% CI]; decision_loss; oracle_cf_cos - cos (value of the current gains)

| U, rho_f | eta = 1.0 | eta = 0.9 | eta = 0.7 | eta = 0.5 | eta = 0.3 | eta = 0.0 |
|---|---|---|---|---|---|---|
| U = 1, rho_f = 0.3 | 1.000; +0.0000 [+0.0000, +0.0000]; 0.0000; +0.0000 | 1.000; +0.0000 [+0.0000, +0.0000]; 0.0000; +0.0000 | 0.421; -0.0000 [-0.0001, +0.0000]; 0.0003; +0.0003 | 0.004; +0.0000 [-0.0000, +0.0000]; 0.0001; +0.0001 | 0.000; +0.0000 [+0.0000, +0.0000]; 0.0001; +0.0001 | 0.000; +0.0000 [+0.0000, +0.0000]; 0.0000; +0.0000 |
| U = 1, rho_f = 0.7 | 1.000; +0.0000 [+0.0000, +0.0000]; 0.0000; +0.0000 | 0.998; -0.0000 [-0.0000, +0.0000]; 0.0000; +0.0000 | 0.376; +0.0000 [-0.0000, +0.0001]; 0.0003; +0.0003 | 0.008; -0.0000 [-0.0000, +0.0000]; 0.0001; +0.0001 | 0.000; +0.0000 [+0.0000, +0.0000]; 0.0001; +0.0001 | 0.000; +0.0000 [+0.0000, +0.0000]; 0.0000; +0.0000 |
| U = 1, rho_f = 0.95 | 1.000; +0.0000 [+0.0000, +0.0000]; 0.0000; +0.0000 | 0.930; -0.0000 [-0.0000, +0.0000]; 0.0002; +0.0002 | 0.297; +0.0003 [+0.0000, +0.0006]; 0.0006; +0.0006 | 0.026; +0.0000 [-0.0000, +0.0000]; 0.0002; +0.0002 | 0.001; +0.0000 [+0.0000, +0.0000]; 0.0001; +0.0001 | 0.000; +0.0000 [+0.0000, +0.0000]; 0.0001; +0.0001 |
| U = 2, rho_f = 0.3 | 1.000; +0.0000 [+0.0000, +0.0000]; 0.0000; +0.0000 | 1.000; +0.0000 [+0.0000, +0.0000]; 0.0000; +0.0000 | 0.838; +0.0000 [-0.0001, +0.0001]; 0.0005; +0.0005 | 0.105; +0.0001 [-0.0000, +0.0002]; 0.0008; +0.0008 | 0.003; +0.0000 [-0.0000, +0.0000]; 0.0005; +0.0005 | 0.002; +0.0000 [-0.0000, +0.0000]; 0.0003; +0.0003 |
| U = 2, rho_f = 0.7 | 1.000; +0.0000 [+0.0000, +0.0000]; 0.0000; +0.0000 | 0.999; +0.0000 [+0.0000, +0.0000]; 0.0000; +0.0000 | 0.753; -0.0000 [-0.0002, +0.0001]; 0.0006; +0.0006 | 0.086; +0.0001 [+0.0000, +0.0001]; 0.0007; +0.0007 | 0.002; +0.0000 [-0.0000, +0.0000]; 0.0005; +0.0005 | 0.001; -0.0000 [-0.0000, +0.0000]; 0.0003; +0.0003 |
| U = 2, rho_f = 0.95 | 1.000; +0.0000 [+0.0000, +0.0000]; 0.0000; +0.0000 | 0.984; +0.0000 [+0.0000, +0.0000]; 0.0002; +0.0002 | 0.458; -0.0000 [-0.0002, +0.0003]; 0.0010; +0.0009 | 0.051; +0.0001 [+0.0000, +0.0001]; 0.0007; +0.0007 | 0.005; +0.0000 [-0.0000, +0.0000]; 0.0005; +0.0005 | 0.003; +0.0000 [-0.0000, +0.0000]; 0.0004; +0.0004 |
| U = 4, rho_f = 0.3 | 1.000; +0.0000 [+0.0000, +0.0000]; 0.0000; +0.0000 | 1.000; +0.0000 [+0.0000, +0.0000]; 0.0000; +0.0000 | 1.000; +0.0000 [+0.0000, +0.0000]; 0.0004; +0.0004 | 0.943; +0.0002 [+0.0001, +0.0004]; 0.0023; +0.0023 | 0.632; +0.0003 [-0.0002, +0.0008]; 0.0044; +0.0044 | 0.347; +0.0005 [-0.0000, +0.0010]; 0.0046; +0.0046 |
| U = 4, rho_f = 0.7 | 1.000; +0.0000 [+0.0000, +0.0000]; 0.0000; +0.0000 | 1.000; +0.0000 [+0.0000, +0.0000]; 0.0000; +0.0000 | 0.999; +0.0000 [+0.0000, +0.0000]; 0.0005; +0.0005 | 0.929; +0.0002 [+0.0001, +0.0004]; 0.0024; +0.0024 | 0.614; +0.0004 [+0.0000, +0.0009]; 0.0044; +0.0043 | 0.364; +0.0003 [-0.0001, +0.0008]; 0.0045; +0.0045 |
| U = 4, rho_f = 0.95 | 1.000; +0.0000 [+0.0000, +0.0000]; 0.0000; +0.0000 | 1.000; +0.0000 [+0.0000, +0.0000]; 0.0001; +0.0001 | 0.972; +0.0001 [+0.0001, +0.0002]; 0.0018; +0.0018 | 0.827; +0.0004 [+0.0000, +0.0007]; 0.0035; +0.0035 | 0.567; +0.0005 [+0.0000, +0.0010]; 0.0043; +0.0043 | 0.394; -0.0001 [-0.0006, +0.0005]; 0.0039; +0.0039 |
| U = 6, rho_f = 0.3 | 1.000; +0.0000 [+0.0000, +0.0000]; 0.0000; +0.0000 | 1.000; +0.0000 [+0.0000, +0.0000]; 0.0000; +0.0000 | 1.000; +0.0000 [+0.0000, +0.0000]; 0.0003; +0.0003 | 0.970; +0.0000 [-0.0002, +0.0002]; 0.0020; +0.0020 | 0.685; +0.0000 [-0.0004, +0.0004]; 0.0041; +0.0040 | 0.322; +0.0002 [-0.0002, +0.0005]; 0.0042; +0.0042 |
| U = 6, rho_f = 0.7 | 1.000; +0.0000 [+0.0000, +0.0000]; 0.0000; +0.0000 | 1.000; +0.0000 [+0.0000, +0.0000]; 0.0000; +0.0000 | 1.000; +0.0000 [+0.0000, +0.0000]; 0.0005; +0.0004 | 0.959; -0.0000 [-0.0002, +0.0001]; 0.0024; +0.0024 | 0.676; +0.0001 [-0.0003, +0.0006]; 0.0044; +0.0044 | 0.358; +0.0005 [+0.0001, +0.0009]; 0.0052; +0.0051 |
| U = 6, rho_f = 0.95 | 1.000; +0.0000 [+0.0000, +0.0000]; 0.0000; +0.0000 | 1.000; +0.0000 [+0.0000, +0.0000]; 0.0002; +0.0002 | 0.991; +0.0000 [-0.0000, +0.0001]; 0.0018; +0.0018 | 0.872; +0.0001 [-0.0002, +0.0004]; 0.0031; +0.0030 | 0.570; +0.0004 [-0.0000, +0.0009]; 0.0046; +0.0045 | 0.392; +0.0003 [-0.0001, +0.0008]; 0.0046; +0.0045 |

### D.2 Summary at rho_f = 0.7, nf = 6

- 0 dB, eta = 1.0: csiq gain U = 1 +0.0428 [+0.0413, +0.0443]; U = 2 +0.0752 [+0.0731, +0.0773]; U = 4 +0.1359 [+0.1327, +0.1392]; U = 6 +0.1515 [+0.1487, +0.1541]
- 0 dB, eta = 1.0: switch gain U = 1 +0.0428 [+0.0413, +0.0443]; U = 2 +0.0752 [+0.0731, +0.0773]; U = 4 +0.1359 [+0.1327, +0.1392]; U = 6 +0.1515 [+0.1487, +0.1541]
- 0 dB, eta = 1.0: weakest-user csiq gain U = 1 +0.0428 [+0.0413, +0.0443]; U = 2 +0.1068 [+0.1032, +0.1105]; U = 4 +0.2669 [+0.2591, +0.2754]; U = 6 +0.3463 [+0.3370, +0.3545]
- 0 dB, eta = 0.0: csiq gain U = 1 -0.0435 [-0.0480, -0.0395]; U = 2 -0.0434 [-0.0475, -0.0390]; U = 4 -0.0051 [-0.0095, -0.0012]; U = 6 -0.0071 [-0.0108, -0.0032]
- 0 dB, eta = 0.0: switch gain U = 1 +0.0000 [+0.0000, +0.0000]; U = 2 -0.0002 [-0.0006, +0.0001]; U = 4 -0.0004 [-0.0026, +0.0016]; U = 6 -0.0011 [-0.0030, +0.0007]
- 0 dB, eta = 0.0: weakest-user csiq gain U = 1 -0.0435 [-0.0480, -0.0395]; U = 2 -0.0629 [-0.0703, -0.0555]; U = 4 -0.0039 [-0.0145, +0.0060]; U = 6 -0.0063 [-0.0189, +0.0055]
- 10 dB, eta = 1.0: csiq gain U = 1 +0.0066 [+0.0063, +0.0068]; U = 2 +0.0179 [+0.0172, +0.0186]; U = 4 +0.0522 [+0.0506, +0.0538]; U = 6 +0.0588 [+0.0572, +0.0605]
- 10 dB, eta = 1.0: switch gain U = 1 +0.0066 [+0.0063, +0.0068]; U = 2 +0.0179 [+0.0172, +0.0186]; U = 4 +0.0522 [+0.0506, +0.0538]; U = 6 +0.0588 [+0.0572, +0.0605]
- 10 dB, eta = 1.0: weakest-user csiq gain U = 1 +0.0066 [+0.0063, +0.0068]; U = 2 +0.0271 [+0.0258, +0.0284]; U = 4 +0.1190 [+0.1139, +0.1239]; U = 6 +0.1622 [+0.1559, +0.1691]
- 10 dB, eta = 0.0: csiq gain U = 1 -0.0162 [-0.0180, -0.0144]; U = 2 -0.0183 [-0.0202, -0.0165]; U = 4 -0.0036 [-0.0057, -0.0015]; U = 6 -0.0039 [-0.0059, -0.0020]
- 10 dB, eta = 0.0: switch gain U = 1 +0.0000 [+0.0000, +0.0000]; U = 2 -0.0000 [-0.0000, +0.0000]; U = 4 -0.0013 [-0.0025, -0.0002]; U = 6 -0.0009 [-0.0019, +0.0002]
- 10 dB, eta = 0.0: weakest-user csiq gain U = 1 -0.0162 [-0.0180, -0.0144]; U = 2 -0.0284 [-0.0319, -0.0251]; U = 4 -0.0099 [-0.0161, -0.0029]; U = 6 -0.0154 [-0.0230, -0.0075]
- 20 dB, eta = 1.0: csiq gain U = 1 +0.0007 [+0.0007, +0.0007]; U = 2 +0.0023 [+0.0022, +0.0024]; U = 4 +0.0114 [+0.0109, +0.0121]; U = 6 +0.0131 [+0.0125, +0.0138]
- 20 dB, eta = 1.0: switch gain U = 1 +0.0007 [+0.0007, +0.0007]; U = 2 +0.0023 [+0.0022, +0.0024]; U = 4 +0.0114 [+0.0109, +0.0121]; U = 6 +0.0131 [+0.0125, +0.0138]
- 20 dB, eta = 1.0: weakest-user csiq gain U = 1 +0.0007 [+0.0007, +0.0007]; U = 2 +0.0033 [+0.0031, +0.0035]; U = 4 +0.0288 [+0.0267, +0.0311]; U = 6 +0.0420 [+0.0391, +0.0454]
- 20 dB, eta = 0.0: csiq gain U = 1 -0.0033 [-0.0040, -0.0027]; U = 2 -0.0043 [-0.0049, -0.0037]; U = 4 -0.0008 [-0.0017, -0.0001]; U = 6 -0.0005 [-0.0012, +0.0002]
- 20 dB, eta = 0.0: switch gain U = 1 +0.0000 [+0.0000, +0.0000]; U = 2 +0.0000 [+0.0000, +0.0000]; U = 4 -0.0003 [-0.0008, +0.0001]; U = 6 -0.0005 [-0.0009, -0.0001]
- 20 dB, eta = 0.0: weakest-user csiq gain U = 1 -0.0033 [-0.0040, -0.0027]; U = 2 -0.0072 [-0.0084, -0.0061]; U = 4 -0.0030 [-0.0062, -0.0001]; U = 6 -0.0015 [-0.0053, +0.0022]
- eta = 0, nf = 6, every rho_f, U and SNR: largest |switch gain| 0.0018 (rho_f = 0.3, U = 4, 0 dB)
- switch below the load masks with the CI entirely below 0: 14 of 294 points; worst -0.0027 [-0.0044, -0.0010] (nf 0, rho_f 0.7, eta 0.0, U = 4, 0 dB), -0.0015 [-0.0027, -0.0003] (nf 6, rho_f 0.3, eta 0.0, U = 4, 10 dB), -0.0014 [-0.0024, -0.0006] (nf 6, rho_f 0.3, eta 0.3, U = 2, 0 dB), -0.0013 [-0.0025, -0.0002] (nf 6, rho_f 0.7, eta 0.0, U = 4, 10 dB), -0.0013 [-0.0023, -0.0003] (nf 0, rho_f 0.7, eta 0.0, U = 4, 10 dB)
- largest rule_loss over all runs: +0.0038 (nf 6, rho_f 0.7, eta 0.3, U = 4, 0 dB)
- first eta (from 1 down to 0) at which the csiq gain CI lies entirely below 0 (nf = 6):
  - 0 dB: U = 1, rho_f 0.3: 0.5; U = 1, rho_f 0.7: 0.5; U = 1, rho_f 0.95: 0.5; U = 2, rho_f 0.3: 0.3; U = 2, rho_f 0.7: 0.3; U = 2, rho_f 0.95: 0.3; U = 4, rho_f 0.3: 0.0; U = 4, rho_f 0.7: 0.0; U = 4, rho_f 0.95: 0.0; U = 6, rho_f 0.3: 0.0; U = 6, rho_f 0.7: 0.0; U = 6, rho_f 0.95: none
  - 10 dB: U = 1, rho_f 0.3: 0.5; U = 1, rho_f 0.7: 0.5; U = 1, rho_f 0.95: 0.7; U = 2, rho_f 0.3: 0.5; U = 2, rho_f 0.7: 0.5; U = 2, rho_f 0.95: 0.5; U = 4, rho_f 0.3: 0.0; U = 4, rho_f 0.7: 0.0; U = 4, rho_f 0.95: none; U = 6, rho_f 0.3: 0.0; U = 6, rho_f 0.7: 0.0; U = 6, rho_f 0.95: 0.0
  - 20 dB: U = 1, rho_f 0.3: 0.7; U = 1, rho_f 0.7: 0.7; U = 1, rho_f 0.95: 0.7; U = 2, rho_f 0.3: 0.5; U = 2, rho_f 0.7: 0.5; U = 2, rho_f 0.95: 0.7; U = 4, rho_f 0.3: 0.0; U = 4, rho_f 0.7: 0.0; U = 4, rho_f 0.95: none; U = 6, rho_f 0.3: 0.0; U = 6, rho_f 0.7: none; U = 6, rho_f 0.95: 0.0

### D.3 Subband correlation rho_f at eta = 1 (current gains), nf = 6: gain over the load masks [95% CI]


#### 0 dB, csiq

| rho_f | U = 1 | U = 2 | U = 4 | U = 6 |
|---|---|---|---|---|
| 0.3 | +0.0442 [+0.0427, +0.0456] | +0.0787 [+0.0766, +0.0809] | +0.1406 [+0.1375, +0.1439] | +0.1581 [+0.1554, +0.1608] |
| 0.5 | +0.0437 [+0.0422, +0.0452] | +0.0772 [+0.0750, +0.0794] | +0.1392 [+0.1361, +0.1424] | +0.1563 [+0.1537, +0.1589] |
| 0.7 | +0.0428 [+0.0413, +0.0443] | +0.0752 [+0.0731, +0.0773] | +0.1359 [+0.1327, +0.1392] | +0.1515 [+0.1487, +0.1541] |
| 0.9 | +0.0422 [+0.0404, +0.0439] | +0.0707 [+0.0684, +0.0728] | +0.1264 [+0.1235, +0.1296] | +0.1379 [+0.1356, +0.1406] |
| 0.95 | +0.0430 [+0.0411, +0.0451] | +0.0676 [+0.0653, +0.0698] | +0.1177 [+0.1146, +0.1209] | +0.1270 [+0.1247, +0.1298] |
| 1.0 | +0.0187 [+0.0164, +0.0209] | +0.0246 [+0.0227, +0.0267] | +0.0246 [+0.0233, +0.0259] | +0.0240 [+0.0228, +0.0253] |

#### 0 dB, ods

| rho_f | U = 1 | U = 2 | U = 4 | U = 6 |
|---|---|---|---|---|
| 0.3 | +0.0442 [+0.0427, +0.0456] | +0.0770 [+0.0749, +0.0791] | +0.1385 [+0.1354, +0.1417] | +0.1560 [+0.1533, +0.1587] |
| 0.5 | +0.0437 [+0.0422, +0.0452] | +0.0755 [+0.0733, +0.0776] | +0.1369 [+0.1337, +0.1400] | +0.1540 [+0.1515, +0.1567] |
| 0.7 | +0.0428 [+0.0413, +0.0443] | +0.0733 [+0.0712, +0.0754] | +0.1333 [+0.1301, +0.1365] | +0.1491 [+0.1463, +0.1517] |
| 0.9 | +0.0422 [+0.0404, +0.0439] | +0.0683 [+0.0660, +0.0705] | +0.1232 [+0.1202, +0.1264] | +0.1348 [+0.1323, +0.1375] |
| 0.95 | +0.0430 [+0.0411, +0.0451] | +0.0648 [+0.0624, +0.0669] | +0.1138 [+0.1107, +0.1170] | +0.1232 [+0.1209, +0.1259] |
| 1.0 | +0.0187 [+0.0164, +0.0209] | +0.0229 [+0.0209, +0.0249] | +0.0240 [+0.0227, +0.0253] | +0.0244 [+0.0232, +0.0256] |

#### 0 dB, csiqb

| rho_f | U = 1 | U = 2 | U = 4 | U = 6 |
|---|---|---|---|---|
| 0.3 | +0.0305 [+0.0291, +0.0319] | +0.0635 [+0.0614, +0.0657] | +0.1257 [+0.1225, +0.1289] | +0.1441 [+0.1414, +0.1468] |
| 0.5 | +0.0300 [+0.0286, +0.0314] | +0.0621 [+0.0599, +0.0642] | +0.1243 [+0.1212, +0.1275] | +0.1423 [+0.1397, +0.1450] |
| 0.7 | +0.0292 [+0.0278, +0.0306] | +0.0601 [+0.0581, +0.0623] | +0.1212 [+0.1179, +0.1244] | +0.1376 [+0.1349, +0.1402] |
| 0.9 | +0.0285 [+0.0268, +0.0302] | +0.0559 [+0.0536, +0.0580] | +0.1119 [+0.1089, +0.1151] | +0.1242 [+0.1219, +0.1269] |
| 0.95 | +0.0293 [+0.0275, +0.0313] | +0.0529 [+0.0506, +0.0551] | +0.1032 [+0.1001, +0.1065] | +0.1133 [+0.1110, +0.1161] |
| 1.0 | +0.0034 [+0.0013, +0.0055] | +0.0057 [+0.0041, +0.0075] | +0.0023 [+0.0016, +0.0029] | +0.0006 [-0.0007, +0.0018] |

#### 0 dB, csiqw

| rho_f | U = 1 | U = 2 | U = 4 | U = 6 |
|---|---|---|---|---|
| 0.3 | +0.0431 [+0.0417, +0.0446] | +0.0746 [+0.0725, +0.0768] | +0.1342 [+0.1310, +0.1374] | +0.1508 [+0.1480, +0.1535] |
| 0.5 | +0.0426 [+0.0412, +0.0441] | +0.0732 [+0.0710, +0.0754] | +0.1328 [+0.1296, +0.1360] | +0.1490 [+0.1464, +0.1517] |
| 0.7 | +0.0418 [+0.0403, +0.0433] | +0.0713 [+0.0691, +0.0734] | +0.1297 [+0.1265, +0.1330] | +0.1444 [+0.1417, +0.1471] |
| 0.9 | +0.0412 [+0.0394, +0.0429] | +0.0669 [+0.0646, +0.0690] | +0.1205 [+0.1175, +0.1237] | +0.1313 [+0.1289, +0.1340] |
| 0.95 | +0.0418 [+0.0399, +0.0438] | +0.0636 [+0.0613, +0.0658] | +0.1117 [+0.1086, +0.1149] | +0.1203 [+0.1180, +0.1231] |
| 1.0 | +0.0053 [+0.0037, +0.0067] | +0.0061 [+0.0047, +0.0076] | +0.0019 [+0.0016, +0.0023] | +0.0005 [-0.0008, +0.0017] |

#### 10 dB, csiq

| rho_f | U = 1 | U = 2 | U = 4 | U = 6 |
|---|---|---|---|---|
| 0.3 | +0.0067 [+0.0064, +0.0070] | +0.0184 [+0.0177, +0.0191] | +0.0526 [+0.0509, +0.0543] | +0.0595 [+0.0580, +0.0610] |
| 0.5 | +0.0067 [+0.0064, +0.0070] | +0.0182 [+0.0176, +0.0189] | +0.0526 [+0.0509, +0.0543] | +0.0595 [+0.0580, +0.0611] |
| 0.7 | +0.0066 [+0.0063, +0.0068] | +0.0179 [+0.0172, +0.0186] | +0.0522 [+0.0506, +0.0538] | +0.0588 [+0.0572, +0.0605] |
| 0.9 | +0.0065 [+0.0062, +0.0068] | +0.0170 [+0.0163, +0.0177] | +0.0483 [+0.0467, +0.0498] | +0.0543 [+0.0528, +0.0557] |
| 0.95 | +0.0069 [+0.0065, +0.0072] | +0.0173 [+0.0165, +0.0181] | +0.0456 [+0.0440, +0.0471] | +0.0505 [+0.0491, +0.0519] |
| 1.0 | +0.0026 [+0.0018, +0.0036] | +0.0086 [+0.0076, +0.0097] | +0.0105 [+0.0097, +0.0113] | +0.0094 [+0.0087, +0.0101] |

#### 10 dB, ods

| rho_f | U = 1 | U = 2 | U = 4 | U = 6 |
|---|---|---|---|---|
| 0.3 | +0.0067 [+0.0064, +0.0070] | +0.0180 [+0.0173, +0.0186] | +0.0527 [+0.0511, +0.0544] | +0.0599 [+0.0583, +0.0614] |
| 0.5 | +0.0067 [+0.0064, +0.0070] | +0.0178 [+0.0171, +0.0184] | +0.0526 [+0.0510, +0.0543] | +0.0598 [+0.0584, +0.0615] |
| 0.7 | +0.0066 [+0.0063, +0.0068] | +0.0174 [+0.0167, +0.0181] | +0.0522 [+0.0506, +0.0539] | +0.0591 [+0.0576, +0.0608] |
| 0.9 | +0.0065 [+0.0062, +0.0068] | +0.0163 [+0.0156, +0.0170] | +0.0483 [+0.0466, +0.0498] | +0.0546 [+0.0532, +0.0561] |
| 0.95 | +0.0069 [+0.0065, +0.0072] | +0.0163 [+0.0155, +0.0171] | +0.0454 [+0.0438, +0.0470] | +0.0509 [+0.0495, +0.0523] |
| 1.0 | +0.0026 [+0.0018, +0.0036] | +0.0055 [+0.0046, +0.0064] | +0.0084 [+0.0077, +0.0092] | +0.0081 [+0.0074, +0.0087] |

#### 10 dB, csiqb

| rho_f | U = 1 | U = 2 | U = 4 | U = 6 |
|---|---|---|---|---|
| 0.3 | +0.0016 [+0.0014, +0.0018] | +0.0105 [+0.0099, +0.0111] | +0.0414 [+0.0398, +0.0431] | +0.0496 [+0.0481, +0.0511] |
| 0.5 | +0.0015 [+0.0013, +0.0017] | +0.0103 [+0.0097, +0.0109] | +0.0413 [+0.0397, +0.0430] | +0.0496 [+0.0481, +0.0512] |
| 0.7 | +0.0013 [+0.0011, +0.0015] | +0.0099 [+0.0092, +0.0105] | +0.0410 [+0.0394, +0.0426] | +0.0488 [+0.0473, +0.0505] |
| 0.9 | +0.0008 [+0.0006, +0.0010] | +0.0085 [+0.0079, +0.0092] | +0.0370 [+0.0354, +0.0384] | +0.0441 [+0.0427, +0.0455] |
| 0.95 | +0.0007 [+0.0004, +0.0009] | +0.0083 [+0.0076, +0.0090] | +0.0340 [+0.0324, +0.0355] | +0.0400 [+0.0386, +0.0414] |
| 1.0 | -0.0062 [-0.0070, -0.0053] | -0.0030 [-0.0040, -0.0020] | -0.0025 [-0.0031, -0.0017] | -0.0030 [-0.0036, -0.0024] |

#### 10 dB, csiqw

| rho_f | U = 1 | U = 2 | U = 4 | U = 6 |
|---|---|---|---|---|
| 0.3 | +0.0066 [+0.0063, +0.0069] | +0.0182 [+0.0175, +0.0188] | +0.0519 [+0.0503, +0.0536] | +0.0589 [+0.0574, +0.0604] |
| 0.5 | +0.0066 [+0.0063, +0.0069] | +0.0180 [+0.0173, +0.0186] | +0.0519 [+0.0503, +0.0536] | +0.0589 [+0.0574, +0.0605] |
| 0.7 | +0.0064 [+0.0062, +0.0067] | +0.0177 [+0.0169, +0.0184] | +0.0516 [+0.0500, +0.0532] | +0.0582 [+0.0567, +0.0599] |
| 0.9 | +0.0064 [+0.0061, +0.0067] | +0.0168 [+0.0162, +0.0176] | +0.0479 [+0.0462, +0.0494] | +0.0538 [+0.0524, +0.0553] |
| 0.95 | +0.0068 [+0.0064, +0.0071] | +0.0171 [+0.0163, +0.0179] | +0.0452 [+0.0436, +0.0467] | +0.0500 [+0.0486, +0.0514] |
| 1.0 | +0.0009 [+0.0004, +0.0015] | +0.0049 [+0.0043, +0.0056] | +0.0039 [+0.0036, +0.0042] | +0.0013 [+0.0007, +0.0018] |

#### 20 dB, csiq

| rho_f | U = 1 | U = 2 | U = 4 | U = 6 |
|---|---|---|---|---|
| 0.3 | +0.0007 [+0.0007, +0.0008] | +0.0024 [+0.0023, +0.0025] | +0.0117 [+0.0111, +0.0124] | +0.0131 [+0.0126, +0.0136] |
| 0.5 | +0.0007 [+0.0007, +0.0007] | +0.0023 [+0.0022, +0.0024] | +0.0113 [+0.0108, +0.0119] | +0.0131 [+0.0125, +0.0136] |
| 0.7 | +0.0007 [+0.0007, +0.0007] | +0.0023 [+0.0022, +0.0024] | +0.0114 [+0.0109, +0.0121] | +0.0131 [+0.0125, +0.0138] |
| 0.9 | +0.0007 [+0.0007, +0.0008] | +0.0022 [+0.0021, +0.0024] | +0.0110 [+0.0104, +0.0116] | +0.0123 [+0.0117, +0.0129] |
| 0.95 | +0.0008 [+0.0007, +0.0008] | +0.0023 [+0.0021, +0.0024] | +0.0104 [+0.0098, +0.0110] | +0.0119 [+0.0113, +0.0125] |
| 1.0 | +0.0003 [+0.0002, +0.0005] | +0.0017 [+0.0014, +0.0021] | +0.0014 [+0.0011, +0.0016] | +0.0011 [+0.0009, +0.0013] |

#### 20 dB, ods

| rho_f | U = 1 | U = 2 | U = 4 | U = 6 |
|---|---|---|---|---|
| 0.3 | +0.0007 [+0.0007, +0.0008] | +0.0023 [+0.0022, +0.0024] | +0.0118 [+0.0112, +0.0126] | +0.0133 [+0.0127, +0.0138] |
| 0.5 | +0.0007 [+0.0007, +0.0007] | +0.0023 [+0.0022, +0.0024] | +0.0114 [+0.0109, +0.0119] | +0.0132 [+0.0127, +0.0138] |
| 0.7 | +0.0007 [+0.0007, +0.0007] | +0.0023 [+0.0022, +0.0024] | +0.0115 [+0.0110, +0.0122] | +0.0133 [+0.0127, +0.0140] |
| 0.9 | +0.0007 [+0.0007, +0.0008] | +0.0022 [+0.0020, +0.0023] | +0.0111 [+0.0105, +0.0118] | +0.0126 [+0.0120, +0.0132] |
| 0.95 | +0.0008 [+0.0007, +0.0008] | +0.0022 [+0.0020, +0.0023] | +0.0105 [+0.0099, +0.0111] | +0.0122 [+0.0116, +0.0128] |
| 1.0 | +0.0003 [+0.0002, +0.0005] | +0.0006 [+0.0004, +0.0009] | +0.0009 [+0.0007, +0.0011] | +0.0008 [+0.0007, +0.0010] |

#### 20 dB, csiqb

| rho_f | U = 1 | U = 2 | U = 4 | U = 6 |
|---|---|---|---|---|
| 0.3 | -0.0008 [-0.0009, -0.0007] | +0.0003 [+0.0002, +0.0004] | +0.0077 [+0.0071, +0.0085] | +0.0098 [+0.0093, +0.0104] |
| 0.5 | -0.0008 [-0.0009, -0.0007] | +0.0003 [+0.0002, +0.0004] | +0.0073 [+0.0068, +0.0078] | +0.0098 [+0.0092, +0.0103] |
| 0.7 | -0.0008 [-0.0009, -0.0008] | +0.0002 [+0.0001, +0.0003] | +0.0074 [+0.0068, +0.0080] | +0.0098 [+0.0092, +0.0104] |
| 0.9 | -0.0009 [-0.0010, -0.0008] | -0.0001 [-0.0002, +0.0001] | +0.0067 [+0.0061, +0.0073] | +0.0087 [+0.0082, +0.0093] |
| 0.95 | -0.0010 [-0.0011, -0.0009] | -0.0002 [-0.0003, -0.0001] | +0.0058 [+0.0053, +0.0064] | +0.0081 [+0.0075, +0.0087] |
| 1.0 | -0.0029 [-0.0031, -0.0027] | -0.0026 [-0.0029, -0.0023] | -0.0047 [-0.0049, -0.0044] | -0.0044 [-0.0046, -0.0042] |

#### 20 dB, csiqw

| rho_f | U = 1 | U = 2 | U = 4 | U = 6 |
|---|---|---|---|---|
| 0.3 | +0.0007 [+0.0007, +0.0007] | +0.0023 [+0.0022, +0.0024] | +0.0115 [+0.0109, +0.0122] | +0.0128 [+0.0123, +0.0134] |
| 0.5 | +0.0007 [+0.0007, +0.0007] | +0.0023 [+0.0022, +0.0024] | +0.0111 [+0.0105, +0.0116] | +0.0128 [+0.0123, +0.0134] |
| 0.7 | +0.0007 [+0.0007, +0.0007] | +0.0023 [+0.0022, +0.0024] | +0.0112 [+0.0107, +0.0119] | +0.0129 [+0.0123, +0.0136] |
| 0.9 | +0.0007 [+0.0007, +0.0007] | +0.0022 [+0.0021, +0.0023] | +0.0108 [+0.0102, +0.0114] | +0.0121 [+0.0115, +0.0127] |
| 0.95 | +0.0008 [+0.0007, +0.0008] | +0.0023 [+0.0021, +0.0024] | +0.0102 [+0.0096, +0.0108] | +0.0118 [+0.0111, +0.0124] |
| 1.0 | -0.0000 [-0.0002, +0.0001] | +0.0013 [+0.0010, +0.0015] | +0.0008 [+0.0007, +0.0009] | +0.0002 [+0.0000, +0.0004] |

- dependence of the csiq gain on rho_f (eta = 1, nf = 6, rho_f 0.3 .. 0.95; rho_f = 1 is the one-gain case of D.4): 'decreasing' = the gain falls at every step of rho_f; 'flat' = the CIs at the two ends overlap
  - 0 dB: U = 1: +0.0442 -> +0.0430, flat; U = 2: +0.0787 -> +0.0676, decreasing; U = 4: +0.1406 -> +0.1177, decreasing; U = 6: +0.1581 -> +0.1270, decreasing
  - 10 dB: U = 1: +0.0067 -> +0.0069, flat; U = 2: +0.0184 -> +0.0173, flat; U = 4: +0.0526 -> +0.0456, decreasing; U = 6: +0.0595 -> +0.0505, decreasing
  - 20 dB: U = 1: +0.0007 -> +0.0008, flat; U = 2: +0.0024 -> +0.0023, flat; U = 4: +0.0117 -> +0.0104, not monotone; U = 6: +0.0131 -> +0.0119, not monotone
  - so 'the gain grows as rho_f decreases' holds only for the loads listed as decreasing (0 dB: decreasing at U = 2, 4, 6, flat at U = 1; 10 dB: decreasing at U = 4, 6, flat at U = 1, 2; 20 dB: decreasing at no load, flat at U = 1, 2); at a flat load the subband correlation does not change the gain within the CIs

### D.4 rho_f = 1 (one gain per user and block), eta = 1, nf = 6: CosSim of fixed and gain of every mode [95% CI]

| U | SNR | fixed CosSim / weakest | csiq | csiqp | switch | ods | csiqb | csiqw |
|---|---|---|---|---|---|---|---|---|
| U = 1 | 0 dB | 0.8804 / 0.8804 | +0.0187 [+0.0164, +0.0209] | +0.0187 [+0.0164, +0.0209] | +0.0187 [+0.0164, +0.0209] | +0.0187 [+0.0164, +0.0209] | +0.0034 [+0.0013, +0.0055] | +0.0053 [+0.0037, +0.0067] |
| U = 1 | 10 dB | 0.9739 / 0.9739 | +0.0026 [+0.0018, +0.0036] | +0.0026 [+0.0018, +0.0036] | +0.0026 [+0.0018, +0.0036] | +0.0026 [+0.0018, +0.0036] | -0.0062 [-0.0070, -0.0053] | +0.0009 [+0.0004, +0.0015] |
| U = 1 | 20 dB | 0.9938 / 0.9938 | +0.0003 [+0.0002, +0.0005] | +0.0003 [+0.0002, +0.0005] | +0.0003 [+0.0002, +0.0005] | +0.0003 [+0.0002, +0.0005] | -0.0029 [-0.0031, -0.0027] | -0.0000 [-0.0002, +0.0001] |
| U = 2 | 0 dB | 0.8359 / 0.7587 | +0.0246 [+0.0227, +0.0267] | +0.0246 [+0.0227, +0.0267] | +0.0246 [+0.0227, +0.0267] | +0.0229 [+0.0209, +0.0249] | +0.0057 [+0.0041, +0.0075] | +0.0061 [+0.0047, +0.0076] |
| U = 2 | 10 dB | 0.9527 / 0.9223 | +0.0086 [+0.0076, +0.0097] | +0.0086 [+0.0076, +0.0097] | +0.0086 [+0.0076, +0.0097] | +0.0055 [+0.0046, +0.0064] | -0.0030 [-0.0040, -0.0020] | +0.0049 [+0.0043, +0.0056] |
| U = 2 | 20 dB | 0.9903 / 0.9837 | +0.0017 [+0.0014, +0.0021] | +0.0017 [+0.0014, +0.0021] | +0.0017 [+0.0014, +0.0021] | +0.0006 [+0.0004, +0.0009] | -0.0026 [-0.0029, -0.0023] | +0.0013 [+0.0010, +0.0015] |
| U = 4 | 0 dB | 0.7776 / 0.5787 | +0.0246 [+0.0233, +0.0259] | +0.0246 [+0.0233, +0.0259] | +0.0246 [+0.0233, +0.0259] | +0.0240 [+0.0227, +0.0253] | +0.0023 [+0.0016, +0.0029] | +0.0019 [+0.0016, +0.0023] |
| U = 4 | 10 dB | 0.9288 / 0.8255 | +0.0105 [+0.0097, +0.0113] | +0.0105 [+0.0097, +0.0113] | +0.0105 [+0.0097, +0.0113] | +0.0084 [+0.0077, +0.0092] | -0.0025 [-0.0031, -0.0017] | +0.0039 [+0.0036, +0.0042] |
| U = 4 | 20 dB | 0.9853 / 0.9597 | +0.0014 [+0.0011, +0.0016] | +0.0014 [+0.0011, +0.0016] | +0.0014 [+0.0011, +0.0016] | +0.0009 [+0.0007, +0.0011] | -0.0047 [-0.0049, -0.0044] | +0.0008 [+0.0007, +0.0009] |
| U = 6 | 0 dB | 0.7424 / 0.4591 | +0.0240 [+0.0228, +0.0253] | +0.0240 [+0.0228, +0.0253] | +0.0240 [+0.0228, +0.0253] | +0.0244 [+0.0232, +0.0256] | +0.0006 [-0.0007, +0.0018] | +0.0005 [-0.0008, +0.0017] |
| U = 6 | 10 dB | 0.9088 / 0.7482 | +0.0094 [+0.0087, +0.0101] | +0.0094 [+0.0087, +0.0101] | +0.0094 [+0.0087, +0.0101] | +0.0081 [+0.0074, +0.0087] | -0.0030 [-0.0036, -0.0024] | +0.0013 [+0.0007, +0.0018] |
| U = 6 | 20 dB | 0.9709 / 0.9265 | +0.0011 [+0.0009, +0.0013] | +0.0011 [+0.0009, +0.0013] | +0.0011 [+0.0009, +0.0013] | +0.0008 [+0.0007, +0.0010] | -0.0044 [-0.0046, -0.0042] | +0.0002 [+0.0000, +0.0004] |

- U = 1, rho_f = 1 (one gain per block, so the subbands cannot be chosen), 0 dB: csiq gains +0.0187 [+0.0164, +0.0209] over the load masks with the same energy d_s; the csiq support with (12) at unit gain (csiqw) gains +0.0053 [+0.0037, +0.0067] and with equal values (csiqb) +0.0034 [+0.0013, +0.0055]; the remaining +0.0134 comes from the transmitter knowing the gain in (13), not from choosing subbands. At rho_f = 0.7 the U = 1 gain is +0.0428 [+0.0413, +0.0443], so at U = 1 only part of the channel-aware gain is a subband choice

### D.5 Without near-far offsets (nf = 0) against nf = 6, rho_f = 0.7: csiq gain [95% CI] and fixed CosSim

| U, eta | SNR | nf = 0 csiq gain | nf = 6 csiq gain | nf = 0 switch gain | nf = 6 switch gain | fixed CosSim nf = 0 / nf = 6 |
|---|---|---|---|---|---|---|
| rho_f 0.7, U = 1, eta = 0.0 | 0 dB | -0.0529 [-0.0570, -0.0486] | -0.0435 [-0.0480, -0.0395] | +0.0000 [+0.0000, +0.0000] | +0.0000 [+0.0000, +0.0000] | 0.9383 / 0.9262 |
| rho_f 0.7, U = 1, eta = 0.0 | 10 dB | -0.0148 [-0.0165, -0.0131] | -0.0162 [-0.0180, -0.0144] | +0.0000 [+0.0000, +0.0000] | +0.0000 [+0.0000, +0.0000] | 0.9908 / 0.9885 |
| rho_f 0.7, U = 1, eta = 0.0 | 20 dB | -0.0023 [-0.0026, -0.0020] | -0.0033 [-0.0040, -0.0027] | +0.0000 [+0.0000, +0.0000] | +0.0000 [+0.0000, +0.0000] | 0.9976 / 0.9973 |
| rho_f 0.7, U = 4, eta = 0.0 | 0 dB | -0.0119 [-0.0160, -0.0081] | -0.0051 [-0.0095, -0.0012] | -0.0027 [-0.0044, -0.0010] | -0.0004 [-0.0026, +0.0016] | 0.7923 / 0.7805 |
| rho_f 0.7, U = 4, eta = 0.0 | 10 dB | -0.0033 [-0.0053, -0.0013] | -0.0036 [-0.0057, -0.0015] | -0.0013 [-0.0023, -0.0003] | -0.0013 [-0.0025, -0.0002] | 0.9397 / 0.9336 |
| rho_f 0.7, U = 4, eta = 0.0 | 20 dB | -0.0006 [-0.0014, +0.0001] | -0.0008 [-0.0017, -0.0001] | -0.0000 [-0.0005, +0.0005] | -0.0003 [-0.0008, +0.0001] | 0.9875 / 0.9854 |
| rho_f 0.7, U = 1, eta = 0.3 | 0 dB | -0.0351 [-0.0391, -0.0314] | -0.0322 [-0.0364, -0.0284] | -0.0000 [-0.0001, +0.0000] | -0.0001 [-0.0003, +0.0001] | 0.9383 / 0.9266 |
| rho_f 0.7, U = 1, eta = 0.3 | 10 dB | -0.0120 [-0.0136, -0.0105] | -0.0124 [-0.0140, -0.0110] | +0.0000 [+0.0000, +0.0000] | +0.0000 [+0.0000, +0.0000] | 0.9908 / 0.9885 |
| rho_f 0.7, U = 1, eta = 0.3 | 20 dB | -0.0022 [-0.0026, -0.0018] | -0.0024 [-0.0030, -0.0020] | +0.0000 [+0.0000, +0.0000] | +0.0000 [+0.0000, +0.0000] | 0.9976 / 0.9973 |
| rho_f 0.7, U = 4, eta = 0.3 | 0 dB | +0.0054 [+0.0013, +0.0093] | +0.0151 [+0.0111, +0.0193] | +0.0036 [+0.0003, +0.0069] | +0.0112 [+0.0076, +0.0150] | 0.7942 / 0.7800 |
| rho_f 0.7, U = 4, eta = 0.3 | 10 dB | +0.0042 [+0.0023, +0.0061] | +0.0048 [+0.0028, +0.0067] | +0.0027 [+0.0010, +0.0042] | +0.0029 [+0.0013, +0.0045] | 0.9397 / 0.9337 |
| rho_f 0.7, U = 4, eta = 0.3 | 20 dB | +0.0009 [+0.0001, +0.0016] | +0.0012 [+0.0005, +0.0020] | +0.0004 [-0.0001, +0.0010] | +0.0008 [+0.0001, +0.0015] | 0.9871 / 0.9850 |
| rho_f 0.7, U = 1, eta = 0.5 | 0 dB | -0.0147 [-0.0179, -0.0115] | -0.0109 [-0.0145, -0.0076] | -0.0009 [-0.0019, +0.0001] | -0.0003 [-0.0020, +0.0011] | 0.9385 / 0.9266 |
| rho_f 0.7, U = 1, eta = 0.5 | 10 dB | -0.0067 [-0.0078, -0.0057] | -0.0065 [-0.0075, -0.0056] | -0.0000 [-0.0001, +0.0000] | -0.0000 [-0.0001, +0.0001] | 0.9908 / 0.9885 |
| rho_f 0.7, U = 1, eta = 0.5 | 20 dB | -0.0014 [-0.0017, -0.0011] | -0.0019 [-0.0024, -0.0015] | -0.0000 [-0.0000, +0.0000] | +0.0000 [-0.0000, +0.0000] | 0.9976 / 0.9973 |
| rho_f 0.7, U = 4, eta = 0.5 | 0 dB | +0.0383 [+0.0344, +0.0417] | +0.0491 [+0.0452, +0.0528] | +0.0380 [+0.0340, +0.0414] | +0.0489 [+0.0449, +0.0526] | 0.7944 / 0.7799 |
| rho_f 0.7, U = 4, eta = 0.5 | 10 dB | +0.0153 [+0.0136, +0.0173] | +0.0182 [+0.0164, +0.0202] | +0.0152 [+0.0135, +0.0172] | +0.0181 [+0.0163, +0.0201] | 0.9398 / 0.9326 |
| rho_f 0.7, U = 4, eta = 0.5 | 20 dB | +0.0032 [+0.0025, +0.0038] | +0.0039 [+0.0032, +0.0046] | +0.0030 [+0.0023, +0.0036] | +0.0036 [+0.0029, +0.0044] | 0.9875 / 0.9854 |
| rho_f 0.7, U = 1, eta = 0.7 | 0 dB | +0.0114 [+0.0098, +0.0132] | +0.0137 [+0.0110, +0.0162] | +0.0114 [+0.0101, +0.0128] | +0.0143 [+0.0119, +0.0165] | 0.9388 / 0.9267 |
| rho_f 0.7, U = 1, eta = 0.7 | 10 dB | -0.0005 [-0.0011, -0.0001] | +0.0001 [-0.0006, +0.0006] | +0.0002 [-0.0001, +0.0005] | +0.0006 [+0.0001, +0.0009] | 0.9908 / 0.9885 |
| rho_f 0.7, U = 1, eta = 0.7 | 20 dB | -0.0003 [-0.0005, -0.0001] | -0.0004 [-0.0007, -0.0003] | +0.0000 [+0.0000, +0.0001] | -0.0000 [-0.0001, +0.0000] | 0.9976 / 0.9973 |
| rho_f 0.7, U = 4, eta = 0.7 | 0 dB | +0.0813 [+0.0781, +0.0844] | +0.0898 [+0.0864, +0.0932] | +0.0813 [+0.0781, +0.0844] | +0.0898 [+0.0864, +0.0932] | 0.7956 / 0.7810 |
| rho_f 0.7, U = 4, eta = 0.7 | 10 dB | +0.0310 [+0.0295, +0.0327] | +0.0352 [+0.0334, +0.0368] | +0.0310 [+0.0295, +0.0327] | +0.0352 [+0.0334, +0.0368] | 0.9391 / 0.9314 |
| rho_f 0.7, U = 4, eta = 0.7 | 20 dB | +0.0063 [+0.0058, +0.0069] | +0.0078 [+0.0071, +0.0084] | +0.0063 [+0.0058, +0.0069] | +0.0078 [+0.0071, +0.0084] | 0.9877 / 0.9853 |
| rho_f 0.7, U = 1, eta = 0.9 | 0 dB | +0.0307 [+0.0299, +0.0316] | +0.0360 [+0.0345, +0.0374] | +0.0307 [+0.0299, +0.0316] | +0.0360 [+0.0345, +0.0374] | 0.9387 / 0.9268 |
| rho_f 0.7, U = 1, eta = 0.9 | 10 dB | +0.0040 [+0.0038, +0.0041] | +0.0052 [+0.0049, +0.0054] | +0.0040 [+0.0038, +0.0041] | +0.0052 [+0.0049, +0.0054] | 0.9908 / 0.9884 |
| rho_f 0.7, U = 1, eta = 0.9 | 20 dB | +0.0004 [+0.0004, +0.0004] | +0.0006 [+0.0005, +0.0006] | +0.0004 [+0.0004, +0.0004] | +0.0006 [+0.0005, +0.0006] | 0.9976 / 0.9973 |
| rho_f 0.7, U = 4, eta = 0.9 | 0 dB | +0.1187 [+0.1160, +0.1218] | +0.1251 [+0.1221, +0.1283] | +0.1187 [+0.1160, +0.1218] | +0.1251 [+0.1221, +0.1283] | 0.7960 / 0.7833 |
| rho_f 0.7, U = 4, eta = 0.9 | 10 dB | +0.0432 [+0.0418, +0.0446] | +0.0479 [+0.0462, +0.0496] | +0.0432 [+0.0418, +0.0446] | +0.0479 [+0.0462, +0.0496] | 0.9400 / 0.9320 |
| rho_f 0.7, U = 4, eta = 0.9 | 20 dB | +0.0091 [+0.0086, +0.0097] | +0.0105 [+0.0099, +0.0111] | +0.0091 [+0.0086, +0.0097] | +0.0105 [+0.0099, +0.0111] | 0.9874 / 0.9854 |
| rho_f 0.7, U = 1, eta = 1.0 | 0 dB | +0.0370 [+0.0362, +0.0378] | +0.0428 [+0.0413, +0.0443] | +0.0370 [+0.0362, +0.0378] | +0.0428 [+0.0413, +0.0443] | 0.9387 / 0.9272 |
| rho_f 0.7, U = 1, eta = 1.0 | 10 dB | +0.0051 [+0.0049, +0.0052] | +0.0066 [+0.0063, +0.0068] | +0.0051 [+0.0049, +0.0052] | +0.0066 [+0.0063, +0.0068] | 0.9908 / 0.9885 |
| rho_f 0.7, U = 1, eta = 1.0 | 20 dB | +0.0005 [+0.0005, +0.0005] | +0.0007 [+0.0007, +0.0007] | +0.0005 [+0.0005, +0.0005] | +0.0007 [+0.0007, +0.0007] | 0.9976 / 0.9972 |
| rho_f 0.7, U = 4, eta = 1.0 | 0 dB | +0.1295 [+0.1269, +0.1325] | +0.1359 [+0.1327, +0.1392] | +0.1295 [+0.1269, +0.1325] | +0.1359 [+0.1327, +0.1392] | 0.7982 / 0.7862 |
| rho_f 0.7, U = 4, eta = 1.0 | 10 dB | +0.0475 [+0.0461, +0.0489] | +0.0522 [+0.0506, +0.0538] | +0.0475 [+0.0461, +0.0489] | +0.0522 [+0.0506, +0.0538] | 0.9394 / 0.9319 |
| rho_f 0.7, U = 4, eta = 1.0 | 20 dB | +0.0099 [+0.0094, +0.0105] | +0.0114 [+0.0109, +0.0121] | +0.0099 [+0.0094, +0.0105] | +0.0114 [+0.0109, +0.0121] | 0.9871 / 0.9852 |
| rho_f 1.0, U = 1, eta = 1.0 | 0 dB | +0.0156 [+0.0135, +0.0176] | +0.0187 [+0.0164, +0.0209] | +0.0156 [+0.0135, +0.0176] | +0.0187 [+0.0164, +0.0209] | 0.8918 / 0.8804 |
| rho_f 1.0, U = 1, eta = 1.0 | 10 dB | +0.0017 [+0.0011, +0.0024] | +0.0026 [+0.0018, +0.0036] | +0.0017 [+0.0011, +0.0024] | +0.0026 [+0.0018, +0.0036] | 0.9779 / 0.9739 |
| rho_f 1.0, U = 1, eta = 1.0 | 20 dB | +0.0003 [+0.0001, +0.0006] | +0.0003 [+0.0002, +0.0005] | +0.0003 [+0.0001, +0.0006] | +0.0003 [+0.0002, +0.0005] | 0.9948 / 0.9938 |
| rho_f 1.0, U = 4, eta = 1.0 | 0 dB | +0.0196 [+0.0185, +0.0208] | +0.0246 [+0.0233, +0.0259] | +0.0196 [+0.0185, +0.0208] | +0.0246 [+0.0233, +0.0259] | 0.7892 / 0.7776 |
| rho_f 1.0, U = 4, eta = 1.0 | 10 dB | +0.0082 [+0.0075, +0.0089] | +0.0105 [+0.0097, +0.0113] | +0.0082 [+0.0075, +0.0089] | +0.0105 [+0.0097, +0.0113] | 0.9367 / 0.9288 |
| rho_f 1.0, U = 4, eta = 1.0 | 20 dB | +0.0012 [+0.0009, +0.0015] | +0.0014 [+0.0011, +0.0016] | +0.0012 [+0.0009, +0.0015] | +0.0014 [+0.0011, +0.0016] | 0.9870 / 0.9853 |

### D.6 Full listing: every coherence file, SNR and mode

Columns: mean CosSim; gain over fixed [95% CI]; weakest-user CosSim; weakest-user gain over fixed [95% CI]. Switch rows add frac_csiq, rule_loss [CI], decision_loss, oracle_cf_cos - cos, frac_csiq_cf.

| nf | rho_f | eta | U | SNR | mode | CosSim | gain [95% CI] | weakest | weakest gain [95% CI] | switch extras |
|---|---|---|---|---|---|---|---|---|---|---|
| 0 | 0.7 | 1.0 | 1 | 0 dB | fixed | 0.9387 | - | 0.9387 | - |  |
| 0 | 0.7 | 1.0 | 1 | 0 dB | csiq | 0.9757 | +0.0370 [+0.0362, +0.0378] | 0.9757 | +0.0370 [+0.0362, +0.0378] |  |
| 0 | 0.7 | 1.0 | 1 | 0 dB | csiqp | 0.9757 | +0.0370 [+0.0362, +0.0378] | 0.9757 | +0.0370 [+0.0362, +0.0378] |  |
| 0 | 0.7 | 1.0 | 1 | 0 dB | switch | 0.9757 | +0.0370 [+0.0362, +0.0378] | 0.9757 | +0.0370 [+0.0362, +0.0378] | frac 1.000; rule_loss +0.0000 [+0.0000, +0.0000]; decision_loss 0.0000; oracle_cf - cos +0.0000; frac_cf 1.000 |
| 0 | 0.7 | 1.0 | 1 | 0 dB | ods | 0.9757 | +0.0370 [+0.0362, +0.0378] | 0.9757 | +0.0370 [+0.0362, +0.0378] |  |
| 0 | 0.7 | 1.0 | 1 | 0 dB | csiqb | 0.9612 | +0.0225 [+0.0217, +0.0233] | 0.9612 | +0.0225 [+0.0217, +0.0233] |  |
| 0 | 0.7 | 1.0 | 1 | 0 dB | csiqw | 0.9744 | +0.0357 [+0.0349, +0.0366] | 0.9744 | +0.0357 [+0.0349, +0.0366] |  |
| 0 | 0.7 | 1.0 | 1 | 10 dB | fixed | 0.9908 | - | 0.9908 | - |  |
| 0 | 0.7 | 1.0 | 1 | 10 dB | csiq | 0.9958 | +0.0051 [+0.0049, +0.0052] | 0.9958 | +0.0051 [+0.0049, +0.0052] |  |
| 0 | 0.7 | 1.0 | 1 | 10 dB | csiqp | 0.9958 | +0.0051 [+0.0049, +0.0052] | 0.9958 | +0.0051 [+0.0049, +0.0052] |  |
| 0 | 0.7 | 1.0 | 1 | 10 dB | switch | 0.9958 | +0.0051 [+0.0049, +0.0052] | 0.9958 | +0.0051 [+0.0049, +0.0052] | frac 1.000; rule_loss +0.0000 [+0.0000, +0.0000]; decision_loss 0.0000; oracle_cf - cos +0.0000; frac_cf 1.000 |
| 0 | 0.7 | 1.0 | 1 | 10 dB | ods | 0.9958 | +0.0051 [+0.0049, +0.0052] | 0.9958 | +0.0051 [+0.0049, +0.0052] |  |
| 0 | 0.7 | 1.0 | 1 | 10 dB | csiqb | 0.9911 | +0.0003 [+0.0002, +0.0005] | 0.9911 | +0.0003 [+0.0002, +0.0005] |  |
| 0 | 0.7 | 1.0 | 1 | 10 dB | csiqw | 0.9957 | +0.0050 [+0.0049, +0.0051] | 0.9957 | +0.0050 [+0.0049, +0.0051] |  |
| 0 | 0.7 | 1.0 | 1 | 20 dB | fixed | 0.9976 | - | 0.9976 | - |  |
| 0 | 0.7 | 1.0 | 1 | 20 dB | csiq | 0.9981 | +0.0005 [+0.0005, +0.0005] | 0.9981 | +0.0005 [+0.0005, +0.0005] |  |
| 0 | 0.7 | 1.0 | 1 | 20 dB | csiqp | 0.9981 | +0.0005 [+0.0005, +0.0005] | 0.9981 | +0.0005 [+0.0005, +0.0005] |  |
| 0 | 0.7 | 1.0 | 1 | 20 dB | switch | 0.9981 | +0.0005 [+0.0005, +0.0005] | 0.9981 | +0.0005 [+0.0005, +0.0005] | frac 1.000; rule_loss +0.0000 [+0.0000, +0.0000]; decision_loss 0.0000; oracle_cf - cos +0.0000; frac_cf 1.000 |
| 0 | 0.7 | 1.0 | 1 | 20 dB | ods | 0.9981 | +0.0005 [+0.0005, +0.0005] | 0.9981 | +0.0005 [+0.0005, +0.0005] |  |
| 0 | 0.7 | 1.0 | 1 | 20 dB | csiqb | 0.9967 | -0.0009 [-0.0010, -0.0008] | 0.9967 | -0.0009 [-0.0010, -0.0008] |  |
| 0 | 0.7 | 1.0 | 1 | 20 dB | csiqw | 0.9981 | +0.0005 [+0.0005, +0.0005] | 0.9981 | +0.0005 [+0.0005, +0.0005] |  |
| 0 | 0.7 | 1.0 | 4 | 0 dB | fixed | 0.7982 | - | 0.6576 | - |  |
| 0 | 0.7 | 1.0 | 4 | 0 dB | csiq | 0.9278 | +0.1295 [+0.1269, +0.1325] | 0.9082 | +0.2506 [+0.2437, +0.2582] |  |
| 0 | 0.7 | 1.0 | 4 | 0 dB | csiqp | 0.9278 | +0.1295 [+0.1269, +0.1325] | 0.9082 | +0.2506 [+0.2437, +0.2582] |  |
| 0 | 0.7 | 1.0 | 4 | 0 dB | switch | 0.9278 | +0.1295 [+0.1269, +0.1325] | 0.9082 | +0.2506 [+0.2437, +0.2582] | frac 1.000; rule_loss +0.0000 [+0.0000, +0.0000]; decision_loss 0.0000; oracle_cf - cos +0.0000; frac_cf 1.000 |
| 0 | 0.7 | 1.0 | 4 | 0 dB | ods | 0.9273 | +0.1290 [+0.1264, +0.1320] | 0.9018 | +0.2442 [+0.2373, +0.2517] |  |
| 0 | 0.7 | 1.0 | 4 | 0 dB | csiqb | 0.9126 | +0.1143 [+0.1117, +0.1173] | 0.8908 | +0.2332 [+0.2262, +0.2408] |  |
| 0 | 0.7 | 1.0 | 4 | 0 dB | csiqw | 0.9230 | +0.1248 [+0.1221, +0.1277] | 0.9034 | +0.2458 [+0.2389, +0.2534] |  |
| 0 | 0.7 | 1.0 | 4 | 10 dB | fixed | 0.9394 | - | 0.8794 | - |  |
| 0 | 0.7 | 1.0 | 4 | 10 dB | csiq | 0.9869 | +0.0475 [+0.0461, +0.0489] | 0.9815 | +0.1021 [+0.0975, +0.1063] |  |
| 0 | 0.7 | 1.0 | 4 | 10 dB | csiqp | 0.9869 | +0.0475 [+0.0461, +0.0489] | 0.9815 | +0.1021 [+0.0975, +0.1063] |  |
| 0 | 0.7 | 1.0 | 4 | 10 dB | switch | 0.9869 | +0.0475 [+0.0461, +0.0489] | 0.9815 | +0.1021 [+0.0975, +0.1063] | frac 1.000; rule_loss +0.0000 [+0.0000, +0.0000]; decision_loss 0.0000; oracle_cf - cos +0.0000; frac_cf 1.000 |
| 0 | 0.7 | 1.0 | 4 | 10 dB | ods | 0.9872 | +0.0478 [+0.0464, +0.0492] | 0.9814 | +0.1020 [+0.0973, +0.1063] |  |
| 0 | 0.7 | 1.0 | 4 | 10 dB | csiqb | 0.9755 | +0.0360 [+0.0347, +0.0374] | 0.9664 | +0.0870 [+0.0825, +0.0913] |  |
| 0 | 0.7 | 1.0 | 4 | 10 dB | csiqw | 0.9862 | +0.0468 [+0.0454, +0.0482] | 0.9805 | +0.1011 [+0.0965, +0.1053] |  |
| 0 | 0.7 | 1.0 | 4 | 20 dB | fixed | 0.9871 | - | 0.9708 | - |  |
| 0 | 0.7 | 1.0 | 4 | 20 dB | csiq | 0.9971 | +0.0099 [+0.0094, +0.0105] | 0.9949 | +0.0241 [+0.0221, +0.0263] |  |
| 0 | 0.7 | 1.0 | 4 | 20 dB | csiqp | 0.9971 | +0.0099 [+0.0094, +0.0105] | 0.9949 | +0.0241 [+0.0221, +0.0263] |  |
| 0 | 0.7 | 1.0 | 4 | 20 dB | switch | 0.9971 | +0.0099 [+0.0094, +0.0105] | 0.9949 | +0.0241 [+0.0221, +0.0263] | frac 1.000; rule_loss +0.0000 [+0.0000, +0.0000]; decision_loss 0.0000; oracle_cf - cos +0.0000; frac_cf 1.000 |
| 0 | 0.7 | 1.0 | 4 | 20 dB | ods | 0.9972 | +0.0100 [+0.0095, +0.0106] | 0.9950 | +0.0242 [+0.0223, +0.0264] |  |
| 0 | 0.7 | 1.0 | 4 | 20 dB | csiqb | 0.9934 | +0.0063 [+0.0057, +0.0068] | 0.9893 | +0.0185 [+0.0165, +0.0206] |  |
| 0 | 0.7 | 1.0 | 4 | 20 dB | csiqw | 0.9969 | +0.0097 [+0.0092, +0.0103] | 0.9946 | +0.0238 [+0.0218, +0.0259] |  |
| 0 | 0.7 | 0.9 | 1 | 0 dB | fixed | 0.9387 | - | 0.9387 | - |  |
| 0 | 0.7 | 0.9 | 1 | 0 dB | csiq | 0.9694 | +0.0307 [+0.0299, +0.0316] | 0.9694 | +0.0307 [+0.0299, +0.0316] |  |
| 0 | 0.7 | 0.9 | 1 | 0 dB | csiqp | 0.9694 | +0.0307 [+0.0299, +0.0316] | 0.9694 | +0.0307 [+0.0299, +0.0316] |  |
| 0 | 0.7 | 0.9 | 1 | 0 dB | switch | 0.9694 | +0.0307 [+0.0299, +0.0316] | 0.9694 | +0.0307 [+0.0299, +0.0316] | frac 1.000; rule_loss +0.0000 [+0.0000, +0.0000]; decision_loss 0.0002; oracle_cf - cos +0.0002; frac_cf 0.989 |
| 0 | 0.7 | 0.9 | 1 | 0 dB | ods | 0.9694 | +0.0307 [+0.0299, +0.0316] | 0.9694 | +0.0307 [+0.0299, +0.0316] |  |
| 0 | 0.7 | 0.9 | 1 | 0 dB | csiqb | 0.9550 | +0.0163 [+0.0155, +0.0171] | 0.9550 | +0.0163 [+0.0155, +0.0171] |  |
| 0 | 0.7 | 0.9 | 1 | 0 dB | csiqw | 0.9685 | +0.0298 [+0.0289, +0.0306] | 0.9685 | +0.0298 [+0.0289, +0.0306] |  |
| 0 | 0.7 | 0.9 | 1 | 10 dB | fixed | 0.9908 | - | 0.9908 | - |  |
| 0 | 0.7 | 0.9 | 1 | 10 dB | csiq | 0.9948 | +0.0040 [+0.0038, +0.0041] | 0.9948 | +0.0040 [+0.0038, +0.0041] |  |
| 0 | 0.7 | 0.9 | 1 | 10 dB | csiqp | 0.9948 | +0.0040 [+0.0038, +0.0041] | 0.9948 | +0.0040 [+0.0038, +0.0041] |  |
| 0 | 0.7 | 0.9 | 1 | 10 dB | switch | 0.9948 | +0.0040 [+0.0038, +0.0041] | 0.9948 | +0.0040 [+0.0038, +0.0041] | frac 0.999; rule_loss +0.0000 [+0.0000, +0.0000]; decision_loss 0.0001; oracle_cf - cos +0.0001; frac_cf 0.984 |
| 0 | 0.7 | 0.9 | 1 | 10 dB | ods | 0.9948 | +0.0040 [+0.0038, +0.0041] | 0.9948 | +0.0040 [+0.0038, +0.0041] |  |
| 0 | 0.7 | 0.9 | 1 | 10 dB | csiqb | 0.9892 | -0.0016 [-0.0018, -0.0015] | 0.9892 | -0.0016 [-0.0018, -0.0015] |  |
| 0 | 0.7 | 0.9 | 1 | 10 dB | csiqw | 0.9946 | +0.0038 [+0.0037, +0.0040] | 0.9946 | +0.0038 [+0.0037, +0.0040] |  |
| 0 | 0.7 | 0.9 | 1 | 20 dB | fixed | 0.9976 | - | 0.9976 | - |  |
| 0 | 0.7 | 0.9 | 1 | 20 dB | csiq | 0.9980 | +0.0004 [+0.0004, +0.0004] | 0.9980 | +0.0004 [+0.0004, +0.0004] |  |
| 0 | 0.7 | 0.9 | 1 | 20 dB | csiqp | 0.9980 | +0.0004 [+0.0004, +0.0004] | 0.9980 | +0.0004 [+0.0004, +0.0004] |  |
| 0 | 0.7 | 0.9 | 1 | 20 dB | switch | 0.9980 | +0.0004 [+0.0004, +0.0004] | 0.9980 | +0.0004 [+0.0004, +0.0004] | frac 0.998; rule_loss +0.0000 [+0.0000, +0.0000]; decision_loss 0.0000; oracle_cf - cos +0.0000; frac_cf 0.986 |
| 0 | 0.7 | 0.9 | 1 | 20 dB | ods | 0.9980 | +0.0004 [+0.0004, +0.0004] | 0.9980 | +0.0004 [+0.0004, +0.0004] |  |
| 0 | 0.7 | 0.9 | 1 | 20 dB | csiqb | 0.9964 | -0.0012 [-0.0013, -0.0011] | 0.9964 | -0.0012 [-0.0013, -0.0011] |  |
| 0 | 0.7 | 0.9 | 1 | 20 dB | csiqw | 0.9980 | +0.0004 [+0.0004, +0.0004] | 0.9980 | +0.0004 [+0.0004, +0.0004] |  |
| 0 | 0.7 | 0.9 | 4 | 0 dB | fixed | 0.7960 | - | 0.6535 | - |  |
| 0 | 0.7 | 0.9 | 4 | 0 dB | csiq | 0.9148 | +0.1187 [+0.1160, +0.1218] | 0.8829 | +0.2294 [+0.2223, +0.2369] |  |
| 0 | 0.7 | 0.9 | 4 | 0 dB | csiqp | 0.9149 | +0.1188 [+0.1161, +0.1219] | 0.8830 | +0.2295 [+0.2223, +0.2369] |  |
| 0 | 0.7 | 0.9 | 4 | 0 dB | switch | 0.9148 | +0.1187 [+0.1160, +0.1218] | 0.8829 | +0.2294 [+0.2223, +0.2369] | frac 1.000; rule_loss +0.0000 [+0.0000, +0.0000]; decision_loss 0.0000; oracle_cf - cos +0.0000; frac_cf 0.999 |
| 0 | 0.7 | 0.9 | 4 | 0 dB | ods | 0.9144 | +0.1184 [+0.1156, +0.1214] | 0.8777 | +0.2241 [+0.2170, +0.2320] |  |
| 0 | 0.7 | 0.9 | 4 | 0 dB | csiqb | 0.9011 | +0.1051 [+0.1023, +0.1081] | 0.8698 | +0.2163 [+0.2093, +0.2237] |  |
| 0 | 0.7 | 0.9 | 4 | 0 dB | csiqw | 0.9118 | +0.1158 [+0.1130, +0.1188] | 0.8816 | +0.2281 [+0.2211, +0.2356] |  |
| 0 | 0.7 | 0.9 | 4 | 10 dB | fixed | 0.9400 | - | 0.8810 | - |  |
| 0 | 0.7 | 0.9 | 4 | 10 dB | csiq | 0.9833 | +0.0432 [+0.0418, +0.0446] | 0.9744 | +0.0933 [+0.0890, +0.0978] |  |
| 0 | 0.7 | 0.9 | 4 | 10 dB | csiqp | 0.9833 | +0.0433 [+0.0418, +0.0447] | 0.9743 | +0.0933 [+0.0890, +0.0977] |  |
| 0 | 0.7 | 0.9 | 4 | 10 dB | switch | 0.9833 | +0.0432 [+0.0418, +0.0446] | 0.9744 | +0.0933 [+0.0890, +0.0978] | frac 1.000; rule_loss +0.0000 [+0.0000, +0.0000]; decision_loss 0.0000; oracle_cf - cos +0.0000; frac_cf 1.000 |
| 0 | 0.7 | 0.9 | 4 | 10 dB | ods | 0.9834 | +0.0434 [+0.0420, +0.0448] | 0.9741 | +0.0931 [+0.0888, +0.0975] |  |
| 0 | 0.7 | 0.9 | 4 | 10 dB | csiqb | 0.9715 | +0.0315 [+0.0301, +0.0328] | 0.9594 | +0.0784 [+0.0740, +0.0827] |  |
| 0 | 0.7 | 0.9 | 4 | 10 dB | csiqw | 0.9827 | +0.0427 [+0.0413, +0.0441] | 0.9738 | +0.0928 [+0.0885, +0.0972] |  |
| 0 | 0.7 | 0.9 | 4 | 20 dB | fixed | 0.9874 | - | 0.9718 | - |  |
| 0 | 0.7 | 0.9 | 4 | 20 dB | csiq | 0.9965 | +0.0091 [+0.0086, +0.0097] | 0.9939 | +0.0220 [+0.0202, +0.0243] |  |
| 0 | 0.7 | 0.9 | 4 | 20 dB | csiqp | 0.9965 | +0.0091 [+0.0086, +0.0097] | 0.9939 | +0.0220 [+0.0202, +0.0243] |  |
| 0 | 0.7 | 0.9 | 4 | 20 dB | switch | 0.9965 | +0.0091 [+0.0086, +0.0097] | 0.9939 | +0.0220 [+0.0202, +0.0243] | frac 1.000; rule_loss +0.0000 [+0.0000, +0.0000]; decision_loss 0.0000; oracle_cf - cos +0.0000; frac_cf 0.999 |
| 0 | 0.7 | 0.9 | 4 | 20 dB | ods | 0.9966 | +0.0092 [+0.0087, +0.0098] | 0.9940 | +0.0222 [+0.0204, +0.0245] |  |
| 0 | 0.7 | 0.9 | 4 | 20 dB | csiqb | 0.9922 | +0.0049 [+0.0044, +0.0055] | 0.9873 | +0.0155 [+0.0137, +0.0178] |  |
| 0 | 0.7 | 0.9 | 4 | 20 dB | csiqw | 0.9963 | +0.0089 [+0.0084, +0.0095] | 0.9935 | +0.0217 [+0.0199, +0.0240] |  |
| 0 | 0.7 | 0.7 | 1 | 0 dB | fixed | 0.9388 | - | 0.9388 | - |  |
| 0 | 0.7 | 0.7 | 1 | 0 dB | csiq | 0.9502 | +0.0114 [+0.0098, +0.0132] | 0.9502 | +0.0114 [+0.0098, +0.0132] |  |
| 0 | 0.7 | 0.7 | 1 | 0 dB | csiqp | 0.9507 | +0.0119 [+0.0103, +0.0136] | 0.9507 | +0.0119 [+0.0103, +0.0136] |  |
| 0 | 0.7 | 0.7 | 1 | 0 dB | switch | 0.9502 | +0.0114 [+0.0101, +0.0128] | 0.9502 | +0.0114 [+0.0101, +0.0128] | frac 0.812; rule_loss +0.0000 [-0.0010, +0.0011]; decision_loss 0.0073; oracle_cf - cos +0.0071; frac_cf 0.792 |
| 0 | 0.7 | 0.7 | 1 | 0 dB | ods | 0.9502 | +0.0114 [+0.0098, +0.0132] | 0.9502 | +0.0114 [+0.0098, +0.0132] |  |
| 0 | 0.7 | 0.7 | 1 | 0 dB | csiqb | 0.9384 | -0.0004 [-0.0018, +0.0010] | 0.9384 | -0.0004 [-0.0018, +0.0010] |  |
| 0 | 0.7 | 0.7 | 1 | 0 dB | csiqw | 0.9507 | +0.0119 [+0.0104, +0.0136] | 0.9507 | +0.0119 [+0.0104, +0.0136] |  |
| 0 | 0.7 | 0.7 | 1 | 10 dB | fixed | 0.9908 | - | 0.9908 | - |  |
| 0 | 0.7 | 0.7 | 1 | 10 dB | csiq | 0.9903 | -0.0005 [-0.0011, -0.0001] | 0.9903 | -0.0005 [-0.0011, -0.0001] |  |
| 0 | 0.7 | 0.7 | 1 | 10 dB | csiqp | 0.9903 | -0.0005 [-0.0010, -0.0001] | 0.9903 | -0.0005 [-0.0010, -0.0001] |  |
| 0 | 0.7 | 0.7 | 1 | 10 dB | switch | 0.9910 | +0.0002 [-0.0001, +0.0005] | 0.9910 | +0.0002 [-0.0001, +0.0005] | frac 0.524; rule_loss -0.0002 [-0.0005, +0.0001]; decision_loss 0.0018; oracle_cf - cos +0.0018; frac_cf 0.691 |
| 0 | 0.7 | 0.7 | 1 | 10 dB | ods | 0.9903 | -0.0005 [-0.0011, -0.0001] | 0.9903 | -0.0005 [-0.0011, -0.0001] |  |
| 0 | 0.7 | 0.7 | 1 | 10 dB | csiqb | 0.9834 | -0.0074 [-0.0079, -0.0070] | 0.9834 | -0.0074 [-0.0079, -0.0070] |  |
| 0 | 0.7 | 0.7 | 1 | 10 dB | csiqw | 0.9902 | -0.0006 [-0.0011, -0.0002] | 0.9902 | -0.0006 [-0.0011, -0.0002] |  |
| 0 | 0.7 | 0.7 | 1 | 20 dB | fixed | 0.9976 | - | 0.9976 | - |  |
| 0 | 0.7 | 0.7 | 1 | 20 dB | csiq | 0.9973 | -0.0003 [-0.0005, -0.0001] | 0.9973 | -0.0003 [-0.0005, -0.0001] |  |
| 0 | 0.7 | 0.7 | 1 | 20 dB | csiqp | 0.9973 | -0.0003 [-0.0005, -0.0001] | 0.9973 | -0.0003 [-0.0005, -0.0001] |  |
| 0 | 0.7 | 0.7 | 1 | 20 dB | switch | 0.9976 | +0.0000 [+0.0000, +0.0001] | 0.9976 | +0.0000 [+0.0000, +0.0001] | frac 0.365; rule_loss -0.0000 [-0.0001, -0.0000]; decision_loss 0.0002; oracle_cf - cos +0.0002; frac_cf 0.684 |
| 0 | 0.7 | 0.7 | 1 | 20 dB | ods | 0.9973 | -0.0003 [-0.0005, -0.0001] | 0.9973 | -0.0003 [-0.0005, -0.0001] |  |
| 0 | 0.7 | 0.7 | 1 | 20 dB | csiqb | 0.9952 | -0.0024 [-0.0026, -0.0022] | 0.9952 | -0.0024 [-0.0026, -0.0022] |  |
| 0 | 0.7 | 0.7 | 1 | 20 dB | csiqw | 0.9973 | -0.0003 [-0.0005, -0.0001] | 0.9973 | -0.0003 [-0.0005, -0.0001] |  |
| 0 | 0.7 | 0.7 | 4 | 0 dB | fixed | 0.7956 | - | 0.6516 | - |  |
| 0 | 0.7 | 0.7 | 4 | 0 dB | csiq | 0.8769 | +0.0813 [+0.0781, +0.0844] | 0.7999 | +0.1483 [+0.1393, +0.1573] |  |
| 0 | 0.7 | 0.7 | 4 | 0 dB | csiqp | 0.8785 | +0.0829 [+0.0797, +0.0860] | 0.8029 | +0.1513 [+0.1421, +0.1601] |  |
| 0 | 0.7 | 0.7 | 4 | 0 dB | switch | 0.8769 | +0.0813 [+0.0781, +0.0844] | 0.7999 | +0.1483 [+0.1393, +0.1573] | frac 1.000; rule_loss +0.0000 [+0.0000, +0.0000]; decision_loss 0.0032; oracle_cf - cos +0.0030; frac_cf 0.914 |
| 0 | 0.7 | 0.7 | 4 | 0 dB | ods | 0.8765 | +0.0809 [+0.0777, +0.0841] | 0.7960 | +0.1444 [+0.1355, +0.1532] |  |
| 0 | 0.7 | 0.7 | 4 | 0 dB | csiqb | 0.8694 | +0.0738 [+0.0706, +0.0768] | 0.8029 | +0.1513 [+0.1423, +0.1598] |  |
| 0 | 0.7 | 0.7 | 4 | 0 dB | csiqw | 0.8786 | +0.0830 [+0.0798, +0.0860] | 0.8080 | +0.1564 [+0.1474, +0.1653] |  |
| 0 | 0.7 | 0.7 | 4 | 10 dB | fixed | 0.9391 | - | 0.8784 | - |  |
| 0 | 0.7 | 0.7 | 4 | 10 dB | csiq | 0.9701 | +0.0310 [+0.0295, +0.0327] | 0.9426 | +0.0643 [+0.0594, +0.0693] |  |
| 0 | 0.7 | 0.7 | 4 | 10 dB | csiqp | 0.9705 | +0.0314 [+0.0299, +0.0331] | 0.9436 | +0.0652 [+0.0603, +0.0703] |  |
| 0 | 0.7 | 0.7 | 4 | 10 dB | switch | 0.9701 | +0.0310 [+0.0295, +0.0327] | 0.9426 | +0.0643 [+0.0594, +0.0693] | frac 1.000; rule_loss +0.0000 [+0.0000, +0.0000]; decision_loss 0.0018; oracle_cf - cos +0.0018; frac_cf 0.916 |
| 0 | 0.7 | 0.7 | 4 | 10 dB | ods | 0.9700 | +0.0309 [+0.0294, +0.0325] | 0.9414 | +0.0630 [+0.0579, +0.0682] |  |
| 0 | 0.7 | 0.7 | 4 | 10 dB | csiqb | 0.9595 | +0.0204 [+0.0189, +0.0219] | 0.9341 | +0.0557 [+0.0509, +0.0604] |  |
| 0 | 0.7 | 0.7 | 4 | 10 dB | csiqw | 0.9705 | +0.0314 [+0.0299, +0.0330] | 0.9448 | +0.0664 [+0.0616, +0.0714] |  |
| 0 | 0.7 | 0.7 | 4 | 20 dB | fixed | 0.9877 | - | 0.9732 | - |  |
| 0 | 0.7 | 0.7 | 4 | 20 dB | csiq | 0.9940 | +0.0063 [+0.0058, +0.0069] | 0.9880 | +0.0148 [+0.0129, +0.0169] |  |
| 0 | 0.7 | 0.7 | 4 | 20 dB | csiqp | 0.9941 | +0.0064 [+0.0059, +0.0069] | 0.9882 | +0.0150 [+0.0131, +0.0170] |  |
| 0 | 0.7 | 0.7 | 4 | 20 dB | switch | 0.9940 | +0.0063 [+0.0058, +0.0069] | 0.9880 | +0.0148 [+0.0129, +0.0169] | frac 1.000; rule_loss +0.0000 [+0.0000, +0.0000]; decision_loss 0.0004; oracle_cf - cos +0.0004; frac_cf 0.907 |
| 0 | 0.7 | 0.7 | 4 | 20 dB | ods | 0.9940 | +0.0063 [+0.0058, +0.0069] | 0.9878 | +0.0146 [+0.0127, +0.0167] |  |
| 0 | 0.7 | 0.7 | 4 | 20 dB | csiqb | 0.9891 | +0.0014 [+0.0009, +0.0019] | 0.9809 | +0.0077 [+0.0059, +0.0096] |  |
| 0 | 0.7 | 0.7 | 4 | 20 dB | csiqw | 0.9940 | +0.0063 [+0.0058, +0.0068] | 0.9881 | +0.0149 [+0.0131, +0.0169] |  |
| 0 | 0.7 | 0.5 | 1 | 0 dB | fixed | 0.9385 | - | 0.9385 | - |  |
| 0 | 0.7 | 0.5 | 1 | 0 dB | csiq | 0.9238 | -0.0147 [-0.0179, -0.0115] | 0.9238 | -0.0147 [-0.0179, -0.0115] |  |
| 0 | 0.7 | 0.5 | 1 | 0 dB | csiqp | 0.9256 | -0.0128 [-0.0160, -0.0097] | 0.9256 | -0.0128 [-0.0160, -0.0097] |  |
| 0 | 0.7 | 0.5 | 1 | 0 dB | switch | 0.9376 | -0.0009 [-0.0019, +0.0001] | 0.9376 | -0.0009 [-0.0019, +0.0001] | frac 0.133; rule_loss +0.0009 [-0.0001, +0.0019]; decision_loss 0.0117; oracle_cf - cos +0.0115; frac_cf 0.540 |
| 0 | 0.7 | 0.5 | 1 | 0 dB | ods | 0.9238 | -0.0147 [-0.0179, -0.0115] | 0.9238 | -0.0147 [-0.0179, -0.0115] |  |
| 0 | 0.7 | 0.5 | 1 | 0 dB | csiqb | 0.9166 | -0.0219 [-0.0245, -0.0192] | 0.9166 | -0.0219 [-0.0245, -0.0192] |  |
| 0 | 0.7 | 0.5 | 1 | 0 dB | csiqw | 0.9264 | -0.0121 [-0.0152, -0.0091] | 0.9264 | -0.0121 [-0.0152, -0.0091] |  |
| 0 | 0.7 | 0.5 | 1 | 10 dB | fixed | 0.9908 | - | 0.9908 | - |  |
| 0 | 0.7 | 0.5 | 1 | 10 dB | csiq | 0.9841 | -0.0067 [-0.0078, -0.0057] | 0.9841 | -0.0067 [-0.0078, -0.0057] |  |
| 0 | 0.7 | 0.5 | 1 | 10 dB | csiqp | 0.9843 | -0.0065 [-0.0076, -0.0055] | 0.9843 | -0.0065 [-0.0076, -0.0055] |  |
| 0 | 0.7 | 0.5 | 1 | 10 dB | switch | 0.9908 | -0.0000 [-0.0001, +0.0000] | 0.9908 | -0.0000 [-0.0001, +0.0000] | frac 0.025; rule_loss +0.0000 [-0.0000, +0.0001]; decision_loss 0.0010; oracle_cf - cos +0.0010; frac_cf 0.406 |
| 0 | 0.7 | 0.5 | 1 | 10 dB | ods | 0.9841 | -0.0067 [-0.0078, -0.0057] | 0.9841 | -0.0067 [-0.0078, -0.0057] |  |
| 0 | 0.7 | 0.5 | 1 | 10 dB | csiqb | 0.9771 | -0.0138 [-0.0146, -0.0129] | 0.9771 | -0.0138 [-0.0146, -0.0129] |  |
| 0 | 0.7 | 0.5 | 1 | 10 dB | csiqw | 0.9844 | -0.0065 [-0.0075, -0.0055] | 0.9844 | -0.0065 [-0.0075, -0.0055] |  |
| 0 | 0.7 | 0.5 | 1 | 20 dB | fixed | 0.9976 | - | 0.9976 | - |  |
| 0 | 0.7 | 0.5 | 1 | 20 dB | csiq | 0.9962 | -0.0014 [-0.0017, -0.0011] | 0.9962 | -0.0014 [-0.0017, -0.0011] |  |
| 0 | 0.7 | 0.5 | 1 | 20 dB | csiqp | 0.9962 | -0.0014 [-0.0017, -0.0011] | 0.9962 | -0.0014 [-0.0017, -0.0011] |  |
| 0 | 0.7 | 0.5 | 1 | 20 dB | switch | 0.9976 | -0.0000 [-0.0000, +0.0000] | 0.9976 | -0.0000 [-0.0000, +0.0000] | frac 0.007; rule_loss +0.0000 [-0.0000, +0.0000]; decision_loss 0.0001; oracle_cf - cos +0.0001; frac_cf 0.373 |
| 0 | 0.7 | 0.5 | 1 | 20 dB | ods | 0.9962 | -0.0014 [-0.0017, -0.0011] | 0.9962 | -0.0014 [-0.0017, -0.0011] |  |
| 0 | 0.7 | 0.5 | 1 | 20 dB | csiqb | 0.9936 | -0.0040 [-0.0043, -0.0037] | 0.9936 | -0.0040 [-0.0043, -0.0037] |  |
| 0 | 0.7 | 0.5 | 1 | 20 dB | csiqw | 0.9962 | -0.0014 [-0.0017, -0.0011] | 0.9962 | -0.0014 [-0.0017, -0.0011] |  |
| 0 | 0.7 | 0.5 | 4 | 0 dB | fixed | 0.7944 | - | 0.6496 | - |  |
| 0 | 0.7 | 0.5 | 4 | 0 dB | csiq | 0.8327 | +0.0383 [+0.0344, +0.0417] | 0.7088 | +0.0592 [+0.0488, +0.0688] |  |
| 0 | 0.7 | 0.5 | 4 | 0 dB | csiqp | 0.8376 | +0.0432 [+0.0393, +0.0466] | 0.7179 | +0.0683 [+0.0579, +0.0780] |  |
| 0 | 0.7 | 0.5 | 4 | 0 dB | switch | 0.8324 | +0.0380 [+0.0340, +0.0414] | 0.7084 | +0.0587 [+0.0482, +0.0685] | frac 0.993; rule_loss +0.0003 [+0.0001, +0.0006]; decision_loss 0.0167; oracle_cf - cos +0.0161; frac_cf 0.711 |
| 0 | 0.7 | 0.5 | 4 | 0 dB | ods | 0.8328 | +0.0384 [+0.0348, +0.0420] | 0.7071 | +0.0575 [+0.0479, +0.0674] |  |
| 0 | 0.7 | 0.5 | 4 | 0 dB | csiqb | 0.8324 | +0.0380 [+0.0343, +0.0413] | 0.7267 | +0.0770 [+0.0671, +0.0865] |  |
| 0 | 0.7 | 0.5 | 4 | 0 dB | csiqw | 0.8388 | +0.0444 [+0.0406, +0.0478] | 0.7239 | +0.0743 [+0.0639, +0.0838] |  |
| 0 | 0.7 | 0.5 | 4 | 10 dB | fixed | 0.9398 | - | 0.8817 | - |  |
| 0 | 0.7 | 0.5 | 4 | 10 dB | csiq | 0.9551 | +0.0153 [+0.0136, +0.0173] | 0.9084 | +0.0267 [+0.0210, +0.0329] |  |
| 0 | 0.7 | 0.5 | 4 | 10 dB | csiqp | 0.9566 | +0.0167 [+0.0150, +0.0187] | 0.9124 | +0.0307 [+0.0251, +0.0369] |  |
| 0 | 0.7 | 0.5 | 4 | 10 dB | switch | 0.9550 | +0.0152 [+0.0135, +0.0172] | 0.9083 | +0.0266 [+0.0210, +0.0329] | frac 0.990; rule_loss +0.0001 [-0.0001, +0.0003]; decision_loss 0.0071; oracle_cf - cos +0.0070; frac_cf 0.727 |
| 0 | 0.7 | 0.5 | 4 | 10 dB | ods | 0.9551 | +0.0153 [+0.0135, +0.0173] | 0.9082 | +0.0264 [+0.0209, +0.0322] |  |
| 0 | 0.7 | 0.5 | 4 | 10 dB | csiqb | 0.9469 | +0.0071 [+0.0055, +0.0090] | 0.9087 | +0.0270 [+0.0221, +0.0328] |  |
| 0 | 0.7 | 0.5 | 4 | 10 dB | csiqw | 0.9569 | +0.0171 [+0.0155, +0.0191] | 0.9142 | +0.0325 [+0.0270, +0.0387] |  |
| 0 | 0.7 | 0.5 | 4 | 20 dB | fixed | 0.9875 | - | 0.9723 | - |  |
| 0 | 0.7 | 0.5 | 4 | 20 dB | csiq | 0.9907 | +0.0032 [+0.0025, +0.0038] | 0.9792 | +0.0069 [+0.0045, +0.0092] |  |
| 0 | 0.7 | 0.5 | 4 | 20 dB | csiqp | 0.9910 | +0.0035 [+0.0028, +0.0041] | 0.9802 | +0.0079 [+0.0055, +0.0102] |  |
| 0 | 0.7 | 0.5 | 4 | 20 dB | switch | 0.9904 | +0.0030 [+0.0023, +0.0036] | 0.9788 | +0.0065 [+0.0041, +0.0089] | frac 0.923; rule_loss +0.0002 [+0.0001, +0.0003]; decision_loss 0.0022; oracle_cf - cos +0.0022; frac_cf 0.698 |
| 0 | 0.7 | 0.5 | 4 | 20 dB | ods | 0.9907 | +0.0032 [+0.0025, +0.0038] | 0.9791 | +0.0068 [+0.0044, +0.0092] |  |
| 0 | 0.7 | 0.5 | 4 | 20 dB | csiqb | 0.9858 | -0.0016 [-0.0022, -0.0011] | 0.9741 | +0.0018 [-0.0003, +0.0040] |  |
| 0 | 0.7 | 0.5 | 4 | 20 dB | csiqw | 0.9910 | +0.0035 [+0.0029, +0.0041] | 0.9805 | +0.0082 [+0.0058, +0.0105] |  |
| 0 | 0.7 | 0.3 | 1 | 0 dB | fixed | 0.9383 | - | 0.9383 | - |  |
| 0 | 0.7 | 0.3 | 1 | 0 dB | csiq | 0.9032 | -0.0351 [-0.0391, -0.0314] | 0.9032 | -0.0351 [-0.0391, -0.0314] |  |
| 0 | 0.7 | 0.3 | 1 | 0 dB | csiqp | 0.9071 | -0.0312 [-0.0351, -0.0276] | 0.9071 | -0.0312 [-0.0351, -0.0276] |  |
| 0 | 0.7 | 0.3 | 1 | 0 dB | switch | 0.9383 | -0.0000 [-0.0001, +0.0000] | 0.9383 | -0.0000 [-0.0001, +0.0000] | frac 0.003; rule_loss +0.0000 [-0.0000, +0.0001]; decision_loss 0.0066; oracle_cf - cos +0.0063; frac_cf 0.371 |
| 0 | 0.7 | 0.3 | 1 | 0 dB | ods | 0.9032 | -0.0351 [-0.0391, -0.0314] | 0.9032 | -0.0351 [-0.0391, -0.0314] |  |
| 0 | 0.7 | 0.3 | 1 | 0 dB | csiqb | 0.9000 | -0.0383 [-0.0416, -0.0353] | 0.9000 | -0.0383 [-0.0416, -0.0353] |  |
| 0 | 0.7 | 0.3 | 1 | 0 dB | csiqw | 0.9076 | -0.0306 [-0.0345, -0.0271] | 0.9076 | -0.0306 [-0.0345, -0.0271] |  |
| 0 | 0.7 | 0.3 | 1 | 10 dB | fixed | 0.9908 | - | 0.9908 | - |  |
| 0 | 0.7 | 0.3 | 1 | 10 dB | csiq | 0.9788 | -0.0120 [-0.0136, -0.0105] | 0.9788 | -0.0120 [-0.0136, -0.0105] |  |
| 0 | 0.7 | 0.3 | 1 | 10 dB | csiqp | 0.9794 | -0.0115 [-0.0129, -0.0100] | 0.9794 | -0.0115 [-0.0129, -0.0100] |  |
| 0 | 0.7 | 0.3 | 1 | 10 dB | switch | 0.9908 | +0.0000 [+0.0000, +0.0000] | 0.9908 | +0.0000 [+0.0000, +0.0000] | frac 0.000; rule_loss +0.0000 [+0.0000, +0.0000]; decision_loss 0.0005; oracle_cf - cos +0.0005; frac_cf 0.245 |
| 0 | 0.7 | 0.3 | 1 | 10 dB | ods | 0.9788 | -0.0120 [-0.0136, -0.0105] | 0.9788 | -0.0120 [-0.0136, -0.0105] |  |
| 0 | 0.7 | 0.3 | 1 | 10 dB | csiqb | 0.9721 | -0.0188 [-0.0201, -0.0176] | 0.9721 | -0.0188 [-0.0201, -0.0176] |  |
| 0 | 0.7 | 0.3 | 1 | 10 dB | csiqw | 0.9795 | -0.0114 [-0.0129, -0.0100] | 0.9795 | -0.0114 [-0.0129, -0.0100] |  |
| 0 | 0.7 | 0.3 | 1 | 20 dB | fixed | 0.9976 | - | 0.9976 | - |  |
| 0 | 0.7 | 0.3 | 1 | 20 dB | csiq | 0.9954 | -0.0022 [-0.0026, -0.0018] | 0.9954 | -0.0022 [-0.0026, -0.0018] |  |
| 0 | 0.7 | 0.3 | 1 | 20 dB | csiqp | 0.9955 | -0.0021 [-0.0025, -0.0018] | 0.9955 | -0.0021 [-0.0025, -0.0018] |  |
| 0 | 0.7 | 0.3 | 1 | 20 dB | switch | 0.9976 | +0.0000 [+0.0000, +0.0000] | 0.9976 | +0.0000 [+0.0000, +0.0000] | frac 0.000; rule_loss +0.0000 [+0.0000, +0.0000]; decision_loss 0.0001; oracle_cf - cos +0.0000; frac_cf 0.220 |
| 0 | 0.7 | 0.3 | 1 | 20 dB | ods | 0.9954 | -0.0022 [-0.0026, -0.0018] | 0.9954 | -0.0022 [-0.0026, -0.0018] |  |
| 0 | 0.7 | 0.3 | 1 | 20 dB | csiqb | 0.9927 | -0.0049 [-0.0052, -0.0046] | 0.9927 | -0.0049 [-0.0052, -0.0046] |  |
| 0 | 0.7 | 0.3 | 1 | 20 dB | csiqw | 0.9955 | -0.0021 [-0.0025, -0.0018] | 0.9955 | -0.0021 [-0.0025, -0.0018] |  |
| 0 | 0.7 | 0.3 | 4 | 0 dB | fixed | 0.7942 | - | 0.6522 | - |  |
| 0 | 0.7 | 0.3 | 4 | 0 dB | csiq | 0.7996 | +0.0054 [+0.0013, +0.0093] | 0.6505 | -0.0017 [-0.0114, +0.0090] |  |
| 0 | 0.7 | 0.3 | 4 | 0 dB | csiqp | 0.8088 | +0.0146 [+0.0107, +0.0185] | 0.6682 | +0.0160 [+0.0062, +0.0266] |  |
| 0 | 0.7 | 0.3 | 4 | 0 dB | switch | 0.7978 | +0.0036 [+0.0003, +0.0069] | 0.6497 | -0.0025 [-0.0106, +0.0063] | frac 0.713; rule_loss +0.0018 [-0.0003, +0.0040]; decision_loss 0.0333; oracle_cf - cos +0.0326; frac_cf 0.530 |
| 0 | 0.7 | 0.3 | 4 | 0 dB | ods | 0.8000 | +0.0058 [+0.0016, +0.0098] | 0.6525 | +0.0003 [-0.0099, +0.0109] |  |
| 0 | 0.7 | 0.3 | 4 | 0 dB | csiqb | 0.8055 | +0.0113 [+0.0076, +0.0149] | 0.6801 | +0.0279 [+0.0183, +0.0382] |  |
| 0 | 0.7 | 0.3 | 4 | 0 dB | csiqw | 0.8093 | +0.0152 [+0.0112, +0.0190] | 0.6708 | +0.0186 [+0.0090, +0.0292] |  |
| 0 | 0.7 | 0.3 | 4 | 10 dB | fixed | 0.9397 | - | 0.8798 | - |  |
| 0 | 0.7 | 0.3 | 4 | 10 dB | csiq | 0.9439 | +0.0042 [+0.0023, +0.0061] | 0.8856 | +0.0057 [-0.0005, +0.0118] |  |
| 0 | 0.7 | 0.3 | 4 | 10 dB | csiqp | 0.9467 | +0.0070 [+0.0052, +0.0088] | 0.8926 | +0.0128 [+0.0067, +0.0187] |  |
| 0 | 0.7 | 0.3 | 4 | 10 dB | switch | 0.9423 | +0.0027 [+0.0010, +0.0042] | 0.8831 | +0.0032 [-0.0020, +0.0079] | frac 0.685; rule_loss +0.0016 [+0.0006, +0.0027]; decision_loss 0.0137; oracle_cf - cos +0.0136; frac_cf 0.573 |
| 0 | 0.7 | 0.3 | 4 | 10 dB | ods | 0.9443 | +0.0047 [+0.0029, +0.0067] | 0.8868 | +0.0069 [+0.0009, +0.0131] |  |
| 0 | 0.7 | 0.3 | 4 | 10 dB | csiqb | 0.9378 | -0.0019 [-0.0036, -0.0001] | 0.8918 | +0.0119 [+0.0065, +0.0172] |  |
| 0 | 0.7 | 0.3 | 4 | 10 dB | csiqw | 0.9469 | +0.0072 [+0.0055, +0.0091] | 0.8936 | +0.0137 [+0.0077, +0.0197] |  |
| 0 | 0.7 | 0.3 | 4 | 20 dB | fixed | 0.9871 | - | 0.9707 | - |  |
| 0 | 0.7 | 0.3 | 4 | 20 dB | csiq | 0.9880 | +0.0009 [+0.0001, +0.0016] | 0.9724 | +0.0018 [-0.0010, +0.0043] |  |
| 0 | 0.7 | 0.3 | 4 | 20 dB | csiqp | 0.9887 | +0.0015 [+0.0008, +0.0022] | 0.9745 | +0.0038 [+0.0013, +0.0063] |  |
| 0 | 0.7 | 0.3 | 4 | 20 dB | switch | 0.9876 | +0.0004 [-0.0001, +0.0010] | 0.9715 | +0.0009 [-0.0013, +0.0031] | frac 0.611; rule_loss +0.0005 [+0.0001, +0.0009]; decision_loss 0.0038; oracle_cf - cos +0.0038; frac_cf 0.544 |
| 0 | 0.7 | 0.3 | 4 | 20 dB | ods | 0.9885 | +0.0013 [+0.0006, +0.0020] | 0.9740 | +0.0033 [+0.0007, +0.0058] |  |
| 0 | 0.7 | 0.3 | 4 | 20 dB | csiqb | 0.9835 | -0.0036 [-0.0043, -0.0030] | 0.9691 | -0.0015 [-0.0039, +0.0007] |  |
| 0 | 0.7 | 0.3 | 4 | 20 dB | csiqw | 0.9887 | +0.0016 [+0.0009, +0.0022] | 0.9747 | +0.0041 [+0.0015, +0.0065] |  |
| 0 | 0.7 | 0.0 | 1 | 0 dB | fixed | 0.9383 | - | 0.9383 | - |  |
| 0 | 0.7 | 0.0 | 1 | 0 dB | csiq | 0.8854 | -0.0529 [-0.0570, -0.0486] | 0.8854 | -0.0529 [-0.0570, -0.0486] |  |
| 0 | 0.7 | 0.0 | 1 | 0 dB | csiqp | 0.8913 | -0.0470 [-0.0509, -0.0429] | 0.8913 | -0.0470 [-0.0509, -0.0429] |  |
| 0 | 0.7 | 0.0 | 1 | 0 dB | switch | 0.9383 | +0.0000 [+0.0000, +0.0000] | 0.9383 | +0.0000 [+0.0000, +0.0000] | frac 0.000; rule_loss +0.0000 [+0.0000, +0.0000]; decision_loss 0.0045; oracle_cf - cos +0.0042; frac_cf 0.274 |
| 0 | 0.7 | 0.0 | 1 | 0 dB | ods | 0.8854 | -0.0529 [-0.0570, -0.0486] | 0.8854 | -0.0529 [-0.0570, -0.0486] |  |
| 0 | 0.7 | 0.0 | 1 | 0 dB | csiqb | 0.8864 | -0.0519 [-0.0552, -0.0484] | 0.8864 | -0.0519 [-0.0552, -0.0484] |  |
| 0 | 0.7 | 0.0 | 1 | 0 dB | csiqw | 0.8913 | -0.0470 [-0.0509, -0.0429] | 0.8913 | -0.0470 [-0.0509, -0.0429] |  |
| 0 | 0.7 | 0.0 | 1 | 10 dB | fixed | 0.9908 | - | 0.9908 | - |  |
| 0 | 0.7 | 0.0 | 1 | 10 dB | csiq | 0.9761 | -0.0148 [-0.0165, -0.0131] | 0.9761 | -0.0148 [-0.0165, -0.0131] |  |
| 0 | 0.7 | 0.0 | 1 | 10 dB | csiqp | 0.9769 | -0.0140 [-0.0157, -0.0124] | 0.9769 | -0.0140 [-0.0157, -0.0124] |  |
| 0 | 0.7 | 0.0 | 1 | 10 dB | switch | 0.9908 | +0.0000 [+0.0000, +0.0000] | 0.9908 | +0.0000 [+0.0000, +0.0000] | frac 0.000; rule_loss +0.0000 [+0.0000, +0.0000]; decision_loss 0.0003; oracle_cf - cos +0.0003; frac_cf 0.178 |
| 0 | 0.7 | 0.0 | 1 | 10 dB | ods | 0.9761 | -0.0148 [-0.0165, -0.0131] | 0.9761 | -0.0148 [-0.0165, -0.0131] |  |
| 0 | 0.7 | 0.0 | 1 | 10 dB | csiqb | 0.9697 | -0.0211 [-0.0224, -0.0199] | 0.9697 | -0.0211 [-0.0224, -0.0199] |  |
| 0 | 0.7 | 0.0 | 1 | 10 dB | csiqw | 0.9769 | -0.0140 [-0.0157, -0.0124] | 0.9769 | -0.0140 [-0.0157, -0.0124] |  |
| 0 | 0.7 | 0.0 | 1 | 20 dB | fixed | 0.9976 | - | 0.9976 | - |  |
| 0 | 0.7 | 0.0 | 1 | 20 dB | csiq | 0.9953 | -0.0023 [-0.0026, -0.0020] | 0.9953 | -0.0023 [-0.0026, -0.0020] |  |
| 0 | 0.7 | 0.0 | 1 | 20 dB | csiqp | 0.9954 | -0.0022 [-0.0024, -0.0020] | 0.9954 | -0.0022 [-0.0024, -0.0020] |  |
| 0 | 0.7 | 0.0 | 1 | 20 dB | switch | 0.9976 | +0.0000 [+0.0000, +0.0000] | 0.9976 | +0.0000 [+0.0000, +0.0000] | frac 0.000; rule_loss +0.0000 [+0.0000, +0.0000]; decision_loss 0.0000; oracle_cf - cos +0.0000; frac_cf 0.148 |
| 0 | 0.7 | 0.0 | 1 | 20 dB | ods | 0.9953 | -0.0023 [-0.0026, -0.0020] | 0.9953 | -0.0023 [-0.0026, -0.0020] |  |
| 0 | 0.7 | 0.0 | 1 | 20 dB | csiqb | 0.9925 | -0.0051 [-0.0054, -0.0049] | 0.9925 | -0.0051 [-0.0054, -0.0049] |  |
| 0 | 0.7 | 0.0 | 1 | 20 dB | csiqw | 0.9954 | -0.0022 [-0.0024, -0.0020] | 0.9954 | -0.0022 [-0.0024, -0.0020] |  |
| 0 | 0.7 | 0.0 | 4 | 0 dB | fixed | 0.7923 | - | 0.6464 | - |  |
| 0 | 0.7 | 0.0 | 4 | 0 dB | csiq | 0.7804 | -0.0119 [-0.0160, -0.0081] | 0.6178 | -0.0286 [-0.0394, -0.0187] |  |
| 0 | 0.7 | 0.0 | 4 | 0 dB | csiqp | 0.7928 | +0.0005 [-0.0036, +0.0043] | 0.6413 | -0.0052 [-0.0162, +0.0048] |  |
| 0 | 0.7 | 0.0 | 4 | 0 dB | switch | 0.7896 | -0.0027 [-0.0044, -0.0010] | 0.6403 | -0.0061 [-0.0104, -0.0019] | frac 0.195; rule_loss +0.0027 [+0.0010, +0.0044]; decision_loss 0.0319; oracle_cf - cos +0.0310; frac_cf 0.434 |
| 0 | 0.7 | 0.0 | 4 | 0 dB | ods | 0.7803 | -0.0120 [-0.0161, -0.0082] | 0.6190 | -0.0274 [-0.0378, -0.0174] |  |
| 0 | 0.7 | 0.0 | 4 | 0 dB | csiqb | 0.7892 | -0.0031 [-0.0070, +0.0006] | 0.6510 | +0.0045 [-0.0064, +0.0141] |  |
| 0 | 0.7 | 0.0 | 4 | 0 dB | csiqw | 0.7922 | -0.0001 [-0.0043, +0.0036] | 0.6403 | -0.0061 [-0.0170, +0.0037] |  |
| 0 | 0.7 | 0.0 | 4 | 10 dB | fixed | 0.9397 | - | 0.8790 | - |  |
| 0 | 0.7 | 0.0 | 4 | 10 dB | csiq | 0.9364 | -0.0033 [-0.0053, -0.0013] | 0.8712 | -0.0078 [-0.0147, -0.0012] |  |
| 0 | 0.7 | 0.0 | 4 | 10 dB | csiqp | 0.9405 | +0.0008 [-0.0012, +0.0027] | 0.8815 | +0.0024 [-0.0040, +0.0086] |  |
| 0 | 0.7 | 0.0 | 4 | 10 dB | switch | 0.9384 | -0.0013 [-0.0023, -0.0003] | 0.8762 | -0.0028 [-0.0062, +0.0005] | frac 0.275; rule_loss +0.0013 [+0.0003, +0.0023]; decision_loss 0.0151; oracle_cf - cos +0.0149; frac_cf 0.475 |
| 0 | 0.7 | 0.0 | 4 | 10 dB | ods | 0.9370 | -0.0027 [-0.0046, -0.0006] | 0.8724 | -0.0066 [-0.0132, -0.0002] |  |
| 0 | 0.7 | 0.0 | 4 | 10 dB | csiqb | 0.9321 | -0.0076 [-0.0093, -0.0058] | 0.8824 | +0.0034 [-0.0026, +0.0092] |  |
| 0 | 0.7 | 0.0 | 4 | 10 dB | csiqw | 0.9403 | +0.0006 [-0.0013, +0.0025] | 0.8811 | +0.0021 [-0.0045, +0.0085] |  |
| 0 | 0.7 | 0.0 | 4 | 20 dB | fixed | 0.9875 | - | 0.9723 | - |  |
| 0 | 0.7 | 0.0 | 4 | 20 dB | csiq | 0.9869 | -0.0006 [-0.0014, +0.0001] | 0.9698 | -0.0024 [-0.0053, +0.0003] |  |
| 0 | 0.7 | 0.0 | 4 | 20 dB | csiqp | 0.9877 | +0.0002 [-0.0005, +0.0009] | 0.9726 | +0.0003 [-0.0025, +0.0029] |  |
| 0 | 0.7 | 0.0 | 4 | 20 dB | switch | 0.9875 | -0.0000 [-0.0005, +0.0005] | 0.9721 | -0.0002 [-0.0021, +0.0019] | frac 0.370; rule_loss +0.0000 [-0.0005, +0.0005]; decision_loss 0.0037; oracle_cf - cos +0.0037; frac_cf 0.469 |
| 0 | 0.7 | 0.0 | 4 | 20 dB | ods | 0.9871 | -0.0005 [-0.0012, +0.0003] | 0.9704 | -0.0019 [-0.0048, +0.0009] |  |
| 0 | 0.7 | 0.0 | 4 | 20 dB | csiqb | 0.9825 | -0.0050 [-0.0057, -0.0044] | 0.9673 | -0.0050 [-0.0073, -0.0027] |  |
| 0 | 0.7 | 0.0 | 4 | 20 dB | csiqw | 0.9877 | +0.0002 [-0.0005, +0.0009] | 0.9726 | +0.0003 [-0.0024, +0.0029] |  |
| 0 | 1.0 | 1.0 | 1 | 0 dB | fixed | 0.8918 | - | 0.8918 | - |  |
| 0 | 1.0 | 1.0 | 1 | 0 dB | csiq | 0.9075 | +0.0156 [+0.0135, +0.0176] | 0.9075 | +0.0156 [+0.0135, +0.0176] |  |
| 0 | 1.0 | 1.0 | 1 | 0 dB | csiqp | 0.9075 | +0.0156 [+0.0135, +0.0176] | 0.9075 | +0.0156 [+0.0135, +0.0176] |  |
| 0 | 1.0 | 1.0 | 1 | 0 dB | switch | 0.9075 | +0.0156 [+0.0135, +0.0176] | 0.9075 | +0.0156 [+0.0135, +0.0176] | frac 1.000; rule_loss +0.0000 [+0.0000, +0.0000]; decision_loss 0.0012; oracle_cf - cos +0.0000; frac_cf 1.000 |
| 0 | 1.0 | 1.0 | 1 | 0 dB | ods | 0.9075 | +0.0156 [+0.0135, +0.0176] | 0.9075 | +0.0156 [+0.0135, +0.0176] |  |
| 0 | 1.0 | 1.0 | 1 | 0 dB | csiqb | 0.8920 | +0.0002 [-0.0017, +0.0020] | 0.8920 | +0.0002 [-0.0017, +0.0020] |  |
| 0 | 1.0 | 1.0 | 1 | 0 dB | csiqw | 0.8978 | +0.0060 [+0.0045, +0.0074] | 0.8978 | +0.0060 [+0.0045, +0.0074] |  |
| 0 | 1.0 | 1.0 | 1 | 10 dB | fixed | 0.9779 | - | 0.9779 | - |  |
| 0 | 1.0 | 1.0 | 1 | 10 dB | csiq | 0.9796 | +0.0017 [+0.0011, +0.0024] | 0.9796 | +0.0017 [+0.0011, +0.0024] |  |
| 0 | 1.0 | 1.0 | 1 | 10 dB | csiqp | 0.9796 | +0.0017 [+0.0011, +0.0024] | 0.9796 | +0.0017 [+0.0011, +0.0024] |  |
| 0 | 1.0 | 1.0 | 1 | 10 dB | switch | 0.9796 | +0.0017 [+0.0011, +0.0024] | 0.9796 | +0.0017 [+0.0011, +0.0024] | frac 1.000; rule_loss +0.0000 [+0.0000, +0.0000]; decision_loss 0.0004; oracle_cf - cos +0.0000; frac_cf 1.000 |
| 0 | 1.0 | 1.0 | 1 | 10 dB | ods | 0.9796 | +0.0017 [+0.0011, +0.0024] | 0.9796 | +0.0017 [+0.0011, +0.0024] |  |
| 0 | 1.0 | 1.0 | 1 | 10 dB | csiqb | 0.9708 | -0.0070 [-0.0076, -0.0063] | 0.9708 | -0.0070 [-0.0076, -0.0063] |  |
| 0 | 1.0 | 1.0 | 1 | 10 dB | csiqw | 0.9783 | +0.0005 [+0.0001, +0.0009] | 0.9783 | +0.0005 [+0.0001, +0.0009] |  |
| 0 | 1.0 | 1.0 | 1 | 20 dB | fixed | 0.9948 | - | 0.9948 | - |  |
| 0 | 1.0 | 1.0 | 1 | 20 dB | csiq | 0.9950 | +0.0003 [+0.0001, +0.0006] | 0.9950 | +0.0003 [+0.0001, +0.0006] |  |
| 0 | 1.0 | 1.0 | 1 | 20 dB | csiqp | 0.9950 | +0.0003 [+0.0001, +0.0006] | 0.9950 | +0.0003 [+0.0001, +0.0006] |  |
| 0 | 1.0 | 1.0 | 1 | 20 dB | switch | 0.9950 | +0.0003 [+0.0001, +0.0006] | 0.9950 | +0.0003 [+0.0001, +0.0006] | frac 1.000; rule_loss +0.0000 [+0.0000, +0.0000]; decision_loss 0.0001; oracle_cf - cos +0.0000; frac_cf 1.000 |
| 0 | 1.0 | 1.0 | 1 | 20 dB | ods | 0.9950 | +0.0003 [+0.0001, +0.0006] | 0.9950 | +0.0003 [+0.0001, +0.0006] |  |
| 0 | 1.0 | 1.0 | 1 | 20 dB | csiqb | 0.9920 | -0.0027 [-0.0030, -0.0024] | 0.9920 | -0.0027 [-0.0030, -0.0024] |  |
| 0 | 1.0 | 1.0 | 1 | 20 dB | csiqw | 0.9948 | +0.0001 [-0.0000, +0.0001] | 0.9948 | +0.0001 [-0.0000, +0.0001] |  |
| 0 | 1.0 | 1.0 | 4 | 0 dB | fixed | 0.7892 | - | 0.6143 | - |  |
| 0 | 1.0 | 1.0 | 4 | 0 dB | csiq | 0.8088 | +0.0196 [+0.0185, +0.0208] | 0.6721 | +0.0578 [+0.0541, +0.0612] |  |
| 0 | 1.0 | 1.0 | 4 | 0 dB | csiqp | 0.8088 | +0.0196 [+0.0185, +0.0208] | 0.6721 | +0.0578 [+0.0541, +0.0612] |  |
| 0 | 1.0 | 1.0 | 4 | 0 dB | switch | 0.8088 | +0.0196 [+0.0185, +0.0208] | 0.6721 | +0.0578 [+0.0541, +0.0612] | frac 1.000; rule_loss +0.0000 [+0.0000, +0.0000]; decision_loss 0.0003; oracle_cf - cos +0.0000; frac_cf 1.000 |
| 0 | 1.0 | 1.0 | 4 | 0 dB | ods | 0.8085 | +0.0193 [+0.0182, +0.0204] | 0.6700 | +0.0557 [+0.0519, +0.0595] |  |
| 0 | 1.0 | 1.0 | 4 | 0 dB | csiqb | 0.7888 | -0.0004 [-0.0010, +0.0002] | 0.6357 | +0.0214 [+0.0194, +0.0233] |  |
| 0 | 1.0 | 1.0 | 4 | 0 dB | csiqw | 0.7908 | +0.0016 [+0.0013, +0.0020] | 0.6213 | +0.0070 [+0.0057, +0.0084] |  |
| 0 | 1.0 | 1.0 | 4 | 10 dB | fixed | 0.9367 | - | 0.8491 | - |  |
| 0 | 1.0 | 1.0 | 4 | 10 dB | csiq | 0.9448 | +0.0082 [+0.0075, +0.0089] | 0.8888 | +0.0397 [+0.0371, +0.0424] |  |
| 0 | 1.0 | 1.0 | 4 | 10 dB | csiqp | 0.9448 | +0.0082 [+0.0075, +0.0089] | 0.8888 | +0.0397 [+0.0371, +0.0424] |  |
| 0 | 1.0 | 1.0 | 4 | 10 dB | switch | 0.9448 | +0.0082 [+0.0075, +0.0089] | 0.8888 | +0.0397 [+0.0371, +0.0424] | frac 1.000; rule_loss +0.0000 [+0.0000, +0.0000]; decision_loss 0.0000; oracle_cf - cos +0.0000; frac_cf 1.000 |
| 0 | 1.0 | 1.0 | 4 | 10 dB | ods | 0.9430 | +0.0063 [+0.0057, +0.0069] | 0.8716 | +0.0224 [+0.0203, +0.0248] |  |
| 0 | 1.0 | 1.0 | 4 | 10 dB | csiqb | 0.9316 | -0.0050 [-0.0056, -0.0044] | 0.8716 | +0.0225 [+0.0202, +0.0248] |  |
| 0 | 1.0 | 1.0 | 4 | 10 dB | csiqw | 0.9399 | +0.0033 [+0.0031, +0.0035] | 0.8716 | +0.0225 [+0.0214, +0.0234] |  |
| 0 | 1.0 | 1.0 | 4 | 20 dB | fixed | 0.9870 | - | 0.9650 | - |  |
| 0 | 1.0 | 1.0 | 4 | 20 dB | csiq | 0.9882 | +0.0012 [+0.0009, +0.0015] | 0.9724 | +0.0074 [+0.0061, +0.0088] |  |
| 0 | 1.0 | 1.0 | 4 | 20 dB | csiqp | 0.9882 | +0.0012 [+0.0009, +0.0015] | 0.9724 | +0.0074 [+0.0061, +0.0088] |  |
| 0 | 1.0 | 1.0 | 4 | 20 dB | switch | 0.9882 | +0.0012 [+0.0009, +0.0015] | 0.9724 | +0.0074 [+0.0061, +0.0088] | frac 1.000; rule_loss +0.0000 [+0.0000, +0.0000]; decision_loss 0.0000; oracle_cf - cos +0.0000; frac_cf 1.000 |
| 0 | 1.0 | 1.0 | 4 | 20 dB | ods | 0.9878 | +0.0008 [+0.0006, +0.0011] | 0.9682 | +0.0032 [+0.0023, +0.0041] |  |
| 0 | 1.0 | 1.0 | 4 | 20 dB | csiqb | 0.9824 | -0.0046 [-0.0049, -0.0043] | 0.9616 | -0.0034 [-0.0045, -0.0020] |  |
| 0 | 1.0 | 1.0 | 4 | 20 dB | csiqw | 0.9877 | +0.0007 [+0.0006, +0.0008] | 0.9704 | +0.0054 [+0.0047, +0.0062] |  |
| 6 | 0.3 | 1.0 | 1 | 0 dB | fixed | 0.9279 | - | 0.9279 | - |  |
| 6 | 0.3 | 1.0 | 1 | 0 dB | csiq | 0.9721 | +0.0442 [+0.0427, +0.0456] | 0.9721 | +0.0442 [+0.0427, +0.0456] |  |
| 6 | 0.3 | 1.0 | 1 | 0 dB | csiqp | 0.9721 | +0.0442 [+0.0427, +0.0456] | 0.9721 | +0.0442 [+0.0427, +0.0456] |  |
| 6 | 0.3 | 1.0 | 1 | 0 dB | switch | 0.9721 | +0.0442 [+0.0427, +0.0456] | 0.9721 | +0.0442 [+0.0427, +0.0456] | frac 1.000; rule_loss +0.0000 [+0.0000, +0.0000]; decision_loss 0.0000; oracle_cf - cos +0.0000; frac_cf 1.000 |
| 6 | 0.3 | 1.0 | 1 | 0 dB | ods | 0.9721 | +0.0442 [+0.0427, +0.0456] | 0.9721 | +0.0442 [+0.0427, +0.0456] |  |
| 6 | 0.3 | 1.0 | 1 | 0 dB | csiqb | 0.9584 | +0.0305 [+0.0291, +0.0319] | 0.9584 | +0.0305 [+0.0291, +0.0319] |  |
| 6 | 0.3 | 1.0 | 1 | 0 dB | csiqw | 0.9710 | +0.0431 [+0.0417, +0.0446] | 0.9710 | +0.0431 [+0.0417, +0.0446] |  |
| 6 | 0.3 | 1.0 | 1 | 10 dB | fixed | 0.9886 | - | 0.9886 | - |  |
| 6 | 0.3 | 1.0 | 1 | 10 dB | csiq | 0.9953 | +0.0067 [+0.0064, +0.0070] | 0.9953 | +0.0067 [+0.0064, +0.0070] |  |
| 6 | 0.3 | 1.0 | 1 | 10 dB | csiqp | 0.9953 | +0.0067 [+0.0064, +0.0070] | 0.9953 | +0.0067 [+0.0064, +0.0070] |  |
| 6 | 0.3 | 1.0 | 1 | 10 dB | switch | 0.9953 | +0.0067 [+0.0064, +0.0070] | 0.9953 | +0.0067 [+0.0064, +0.0070] | frac 1.000; rule_loss +0.0000 [+0.0000, +0.0000]; decision_loss 0.0000; oracle_cf - cos +0.0000; frac_cf 1.000 |
| 6 | 0.3 | 1.0 | 1 | 10 dB | ods | 0.9953 | +0.0067 [+0.0064, +0.0070] | 0.9953 | +0.0067 [+0.0064, +0.0070] |  |
| 6 | 0.3 | 1.0 | 1 | 10 dB | csiqb | 0.9901 | +0.0016 [+0.0014, +0.0018] | 0.9901 | +0.0016 [+0.0014, +0.0018] |  |
| 6 | 0.3 | 1.0 | 1 | 10 dB | csiqw | 0.9952 | +0.0066 [+0.0063, +0.0069] | 0.9952 | +0.0066 [+0.0063, +0.0069] |  |
| 6 | 0.3 | 1.0 | 1 | 20 dB | fixed | 0.9972 | - | 0.9972 | - |  |
| 6 | 0.3 | 1.0 | 1 | 20 dB | csiq | 0.9980 | +0.0007 [+0.0007, +0.0008] | 0.9980 | +0.0007 [+0.0007, +0.0008] |  |
| 6 | 0.3 | 1.0 | 1 | 20 dB | csiqp | 0.9980 | +0.0007 [+0.0007, +0.0008] | 0.9980 | +0.0007 [+0.0007, +0.0008] |  |
| 6 | 0.3 | 1.0 | 1 | 20 dB | switch | 0.9980 | +0.0007 [+0.0007, +0.0008] | 0.9980 | +0.0007 [+0.0007, +0.0008] | frac 1.000; rule_loss +0.0000 [+0.0000, +0.0000]; decision_loss 0.0000; oracle_cf - cos +0.0000; frac_cf 1.000 |
| 6 | 0.3 | 1.0 | 1 | 20 dB | ods | 0.9980 | +0.0007 [+0.0007, +0.0008] | 0.9980 | +0.0007 [+0.0007, +0.0008] |  |
| 6 | 0.3 | 1.0 | 1 | 20 dB | csiqb | 0.9965 | -0.0008 [-0.0009, -0.0007] | 0.9965 | -0.0008 [-0.0009, -0.0007] |  |
| 6 | 0.3 | 1.0 | 1 | 20 dB | csiqw | 0.9979 | +0.0007 [+0.0007, +0.0007] | 0.9979 | +0.0007 [+0.0007, +0.0007] |  |
| 6 | 0.3 | 1.0 | 2 | 0 dB | fixed | 0.8771 | - | 0.8339 | - |  |
| 6 | 0.3 | 1.0 | 2 | 0 dB | csiq | 0.9558 | +0.0787 [+0.0766, +0.0809] | 0.9453 | +0.1114 [+0.1078, +0.1148] |  |
| 6 | 0.3 | 1.0 | 2 | 0 dB | csiqp | 0.9558 | +0.0787 [+0.0766, +0.0809] | 0.9453 | +0.1114 [+0.1078, +0.1148] |  |
| 6 | 0.3 | 1.0 | 2 | 0 dB | switch | 0.9558 | +0.0787 [+0.0766, +0.0809] | 0.9453 | +0.1114 [+0.1078, +0.1148] | frac 1.000; rule_loss +0.0000 [+0.0000, +0.0000]; decision_loss 0.0000; oracle_cf - cos +0.0000; frac_cf 1.000 |
| 6 | 0.3 | 1.0 | 2 | 0 dB | ods | 0.9541 | +0.0770 [+0.0749, +0.0791] | 0.9378 | +0.1039 [+0.1004, +0.1074] |  |
| 6 | 0.3 | 1.0 | 2 | 0 dB | csiqb | 0.9407 | +0.0635 [+0.0614, +0.0657] | 0.9291 | +0.0953 [+0.0917, +0.0987] |  |
| 6 | 0.3 | 1.0 | 2 | 0 dB | csiqw | 0.9518 | +0.0746 [+0.0725, +0.0768] | 0.9426 | +0.1088 [+0.1051, +0.1122] |  |
| 6 | 0.3 | 1.0 | 2 | 10 dB | fixed | 0.9740 | - | 0.9631 | - |  |
| 6 | 0.3 | 1.0 | 2 | 10 dB | csiq | 0.9924 | +0.0184 [+0.0177, +0.0191] | 0.9906 | +0.0275 [+0.0264, +0.0286] |  |
| 6 | 0.3 | 1.0 | 2 | 10 dB | csiqp | 0.9924 | +0.0184 [+0.0177, +0.0191] | 0.9906 | +0.0275 [+0.0264, +0.0286] |  |
| 6 | 0.3 | 1.0 | 2 | 10 dB | switch | 0.9924 | +0.0184 [+0.0177, +0.0191] | 0.9906 | +0.0275 [+0.0264, +0.0286] | frac 1.000; rule_loss +0.0000 [+0.0000, +0.0000]; decision_loss 0.0000; oracle_cf - cos +0.0000; frac_cf 1.000 |
| 6 | 0.3 | 1.0 | 2 | 10 dB | ods | 0.9920 | +0.0180 [+0.0173, +0.0186] | 0.9889 | +0.0259 [+0.0248, +0.0269] |  |
| 6 | 0.3 | 1.0 | 2 | 10 dB | csiqb | 0.9845 | +0.0105 [+0.0099, +0.0111] | 0.9810 | +0.0180 [+0.0169, +0.0190] |  |
| 6 | 0.3 | 1.0 | 2 | 10 dB | csiqw | 0.9922 | +0.0182 [+0.0175, +0.0188] | 0.9903 | +0.0272 [+0.0261, +0.0283] |  |
| 6 | 0.3 | 1.0 | 2 | 20 dB | fixed | 0.9953 | - | 0.9934 | - |  |
| 6 | 0.3 | 1.0 | 2 | 20 dB | csiq | 0.9977 | +0.0024 [+0.0023, +0.0025] | 0.9968 | +0.0034 [+0.0032, +0.0036] |  |
| 6 | 0.3 | 1.0 | 2 | 20 dB | csiqp | 0.9977 | +0.0024 [+0.0023, +0.0025] | 0.9968 | +0.0034 [+0.0032, +0.0036] |  |
| 6 | 0.3 | 1.0 | 2 | 20 dB | switch | 0.9977 | +0.0024 [+0.0023, +0.0025] | 0.9968 | +0.0034 [+0.0032, +0.0036] | frac 1.000; rule_loss +0.0000 [+0.0000, +0.0000]; decision_loss 0.0000; oracle_cf - cos +0.0000; frac_cf 1.000 |
| 6 | 0.3 | 1.0 | 2 | 20 dB | ods | 0.9977 | +0.0023 [+0.0022, +0.0024] | 0.9967 | +0.0033 [+0.0031, +0.0035] |  |
| 6 | 0.3 | 1.0 | 2 | 20 dB | csiqb | 0.9956 | +0.0003 [+0.0002, +0.0004] | 0.9940 | +0.0006 [+0.0004, +0.0008] |  |
| 6 | 0.3 | 1.0 | 2 | 20 dB | csiqw | 0.9977 | +0.0023 [+0.0022, +0.0024] | 0.9967 | +0.0033 [+0.0031, +0.0035] |  |
| 6 | 0.3 | 1.0 | 4 | 0 dB | fixed | 0.7853 | - | 0.6139 | - |  |
| 6 | 0.3 | 1.0 | 4 | 0 dB | csiq | 0.9259 | +0.1406 [+0.1375, +0.1439] | 0.8917 | +0.2778 [+0.2697, +0.2863] |  |
| 6 | 0.3 | 1.0 | 4 | 0 dB | csiqp | 0.9259 | +0.1406 [+0.1375, +0.1439] | 0.8917 | +0.2778 [+0.2697, +0.2863] |  |
| 6 | 0.3 | 1.0 | 4 | 0 dB | switch | 0.9259 | +0.1406 [+0.1375, +0.1439] | 0.8917 | +0.2778 [+0.2697, +0.2863] | frac 1.000; rule_loss +0.0000 [+0.0000, +0.0000]; decision_loss 0.0000; oracle_cf - cos +0.0000; frac_cf 1.000 |
| 6 | 0.3 | 1.0 | 4 | 0 dB | ods | 0.9238 | +0.1385 [+0.1354, +0.1417] | 0.8727 | +0.2588 [+0.2507, +0.2673] |  |
| 6 | 0.3 | 1.0 | 4 | 0 dB | csiqb | 0.9110 | +0.1257 [+0.1225, +0.1289] | 0.8761 | +0.2622 [+0.2541, +0.2707] |  |
| 6 | 0.3 | 1.0 | 4 | 0 dB | csiqw | 0.9195 | +0.1342 [+0.1310, +0.1374] | 0.8886 | +0.2747 [+0.2666, +0.2832] |  |
| 6 | 0.3 | 1.0 | 4 | 10 dB | fixed | 0.9324 | - | 0.8544 | - |  |
| 6 | 0.3 | 1.0 | 4 | 10 dB | csiq | 0.9851 | +0.0526 [+0.0509, +0.0543] | 0.9747 | +0.1203 [+0.1151, +0.1255] |  |
| 6 | 0.3 | 1.0 | 4 | 10 dB | csiqp | 0.9851 | +0.0526 [+0.0509, +0.0543] | 0.9747 | +0.1203 [+0.1151, +0.1255] |  |
| 6 | 0.3 | 1.0 | 4 | 10 dB | switch | 0.9851 | +0.0526 [+0.0509, +0.0543] | 0.9747 | +0.1203 [+0.1151, +0.1255] | frac 1.000; rule_loss +0.0000 [+0.0000, +0.0000]; decision_loss 0.0000; oracle_cf - cos +0.0000; frac_cf 1.000 |
| 6 | 0.3 | 1.0 | 4 | 10 dB | ods | 0.9852 | +0.0527 [+0.0511, +0.0544] | 0.9728 | +0.1184 [+0.1133, +0.1237] |  |
| 6 | 0.3 | 1.0 | 4 | 10 dB | csiqb | 0.9738 | +0.0414 [+0.0398, +0.0431] | 0.9588 | +0.1044 [+0.0992, +0.1096] |  |
| 6 | 0.3 | 1.0 | 4 | 10 dB | csiqw | 0.9844 | +0.0519 [+0.0503, +0.0536] | 0.9739 | +0.1195 [+0.1143, +0.1247] |  |
| 6 | 0.3 | 1.0 | 4 | 20 dB | fixed | 0.9851 | - | 0.9644 | - |  |
| 6 | 0.3 | 1.0 | 4 | 20 dB | csiq | 0.9968 | +0.0117 [+0.0111, +0.0124] | 0.9943 | +0.0299 [+0.0277, +0.0326] |  |
| 6 | 0.3 | 1.0 | 4 | 20 dB | csiqp | 0.9968 | +0.0117 [+0.0111, +0.0124] | 0.9943 | +0.0299 [+0.0277, +0.0326] |  |
| 6 | 0.3 | 1.0 | 4 | 20 dB | switch | 0.9968 | +0.0117 [+0.0111, +0.0124] | 0.9943 | +0.0299 [+0.0277, +0.0326] | frac 1.000; rule_loss +0.0000 [+0.0000, +0.0000]; decision_loss 0.0000; oracle_cf - cos +0.0000; frac_cf 1.000 |
| 6 | 0.3 | 1.0 | 4 | 20 dB | ods | 0.9969 | +0.0118 [+0.0112, +0.0126] | 0.9945 | +0.0300 [+0.0278, +0.0327] |  |
| 6 | 0.3 | 1.0 | 4 | 20 dB | csiqb | 0.9928 | +0.0077 [+0.0071, +0.0085] | 0.9876 | +0.0232 [+0.0210, +0.0259] |  |
| 6 | 0.3 | 1.0 | 4 | 20 dB | csiqw | 0.9966 | +0.0115 [+0.0109, +0.0122] | 0.9939 | +0.0295 [+0.0272, +0.0321] |  |
| 6 | 0.3 | 1.0 | 6 | 0 dB | fixed | 0.7452 | - | 0.4874 | - |  |
| 6 | 0.3 | 1.0 | 6 | 0 dB | csiq | 0.9032 | +0.1581 [+0.1554, +0.1608] | 0.8482 | +0.3608 [+0.3517, +0.3696] |  |
| 6 | 0.3 | 1.0 | 6 | 0 dB | csiqp | 0.9032 | +0.1581 [+0.1554, +0.1608] | 0.8482 | +0.3608 [+0.3517, +0.3696] |  |
| 6 | 0.3 | 1.0 | 6 | 0 dB | switch | 0.9032 | +0.1581 [+0.1554, +0.1608] | 0.8482 | +0.3608 [+0.3517, +0.3696] | frac 1.000; rule_loss +0.0000 [+0.0000, +0.0000]; decision_loss 0.0000; oracle_cf - cos +0.0000; frac_cf 1.000 |
| 6 | 0.3 | 1.0 | 6 | 0 dB | ods | 0.9012 | +0.1560 [+0.1533, +0.1587] | 0.8220 | +0.3345 [+0.3253, +0.3435] |  |
| 6 | 0.3 | 1.0 | 6 | 0 dB | csiqb | 0.8893 | +0.1441 [+0.1414, +0.1468] | 0.8338 | +0.3463 [+0.3372, +0.3550] |  |
| 6 | 0.3 | 1.0 | 6 | 0 dB | csiqw | 0.8959 | +0.1508 [+0.1480, +0.1535] | 0.8451 | +0.3577 [+0.3486, +0.3665] |  |
| 6 | 0.3 | 1.0 | 6 | 10 dB | fixed | 0.9118 | - | 0.7908 | - |  |
| 6 | 0.3 | 1.0 | 6 | 10 dB | csiq | 0.9713 | +0.0595 [+0.0580, +0.0610] | 0.9523 | +0.1616 [+0.1555, +0.1676] |  |
| 6 | 0.3 | 1.0 | 6 | 10 dB | csiqp | 0.9713 | +0.0595 [+0.0580, +0.0610] | 0.9523 | +0.1616 [+0.1555, +0.1676] |  |
| 6 | 0.3 | 1.0 | 6 | 10 dB | switch | 0.9713 | +0.0595 [+0.0580, +0.0610] | 0.9523 | +0.1616 [+0.1555, +0.1676] | frac 1.000; rule_loss +0.0000 [+0.0000, +0.0000]; decision_loss 0.0000; oracle_cf - cos +0.0000; frac_cf 1.000 |
| 6 | 0.3 | 1.0 | 6 | 10 dB | ods | 0.9717 | +0.0599 [+0.0583, +0.0614] | 0.9526 | +0.1619 [+0.1557, +0.1679] |  |
| 6 | 0.3 | 1.0 | 6 | 10 dB | csiqb | 0.9615 | +0.0496 [+0.0481, +0.0511] | 0.9368 | +0.1461 [+0.1399, +0.1521] |  |
| 6 | 0.3 | 1.0 | 6 | 10 dB | csiqw | 0.9708 | +0.0589 [+0.0574, +0.0604] | 0.9515 | +0.1607 [+0.1546, +0.1668] |  |
| 6 | 0.3 | 1.0 | 6 | 20 dB | fixed | 0.9709 | - | 0.9353 | - |  |
| 6 | 0.3 | 1.0 | 6 | 20 dB | csiq | 0.9840 | +0.0131 [+0.0126, +0.0136] | 0.9764 | +0.0411 [+0.0384, +0.0439] |  |
| 6 | 0.3 | 1.0 | 6 | 20 dB | csiqp | 0.9840 | +0.0131 [+0.0126, +0.0136] | 0.9764 | +0.0411 [+0.0384, +0.0439] |  |
| 6 | 0.3 | 1.0 | 6 | 20 dB | switch | 0.9840 | +0.0131 [+0.0126, +0.0136] | 0.9764 | +0.0411 [+0.0384, +0.0439] | frac 1.000; rule_loss +0.0000 [+0.0000, +0.0000]; decision_loss 0.0000; oracle_cf - cos +0.0000; frac_cf 1.000 |
| 6 | 0.3 | 1.0 | 6 | 20 dB | ods | 0.9842 | +0.0133 [+0.0127, +0.0138] | 0.9770 | +0.0417 [+0.0390, +0.0445] |  |
| 6 | 0.3 | 1.0 | 6 | 20 dB | csiqb | 0.9807 | +0.0098 [+0.0093, +0.0104] | 0.9701 | +0.0348 [+0.0322, +0.0376] |  |
| 6 | 0.3 | 1.0 | 6 | 20 dB | csiqw | 0.9838 | +0.0128 [+0.0123, +0.0134] | 0.9759 | +0.0406 [+0.0379, +0.0434] |  |
| 6 | 0.3 | 0.9 | 1 | 0 dB | fixed | 0.9280 | - | 0.9280 | - |  |
| 6 | 0.3 | 0.9 | 1 | 0 dB | csiq | 0.9657 | +0.0378 [+0.0364, +0.0392] | 0.9657 | +0.0378 [+0.0364, +0.0392] |  |
| 6 | 0.3 | 0.9 | 1 | 0 dB | csiqp | 0.9657 | +0.0378 [+0.0364, +0.0392] | 0.9657 | +0.0378 [+0.0364, +0.0392] |  |
| 6 | 0.3 | 0.9 | 1 | 0 dB | switch | 0.9657 | +0.0378 [+0.0364, +0.0392] | 0.9657 | +0.0378 [+0.0364, +0.0392] | frac 1.000; rule_loss +0.0000 [+0.0000, +0.0000]; decision_loss 0.0001; oracle_cf - cos +0.0000; frac_cf 0.994 |
| 6 | 0.3 | 0.9 | 1 | 0 dB | ods | 0.9657 | +0.0378 [+0.0364, +0.0392] | 0.9657 | +0.0378 [+0.0364, +0.0392] |  |
| 6 | 0.3 | 0.9 | 1 | 0 dB | csiqb | 0.9522 | +0.0242 [+0.0228, +0.0256] | 0.9522 | +0.0242 [+0.0228, +0.0256] |  |
| 6 | 0.3 | 0.9 | 1 | 0 dB | csiqw | 0.9648 | +0.0368 [+0.0354, +0.0382] | 0.9648 | +0.0368 [+0.0354, +0.0382] |  |
| 6 | 0.3 | 0.9 | 1 | 10 dB | fixed | 0.9885 | - | 0.9885 | - |  |
| 6 | 0.3 | 0.9 | 1 | 10 dB | csiq | 0.9941 | +0.0055 [+0.0053, +0.0058] | 0.9941 | +0.0055 [+0.0053, +0.0058] |  |
| 6 | 0.3 | 0.9 | 1 | 10 dB | csiqp | 0.9941 | +0.0055 [+0.0053, +0.0058] | 0.9941 | +0.0055 [+0.0053, +0.0058] |  |
| 6 | 0.3 | 0.9 | 1 | 10 dB | switch | 0.9941 | +0.0055 [+0.0053, +0.0058] | 0.9941 | +0.0055 [+0.0053, +0.0058] | frac 1.000; rule_loss +0.0000 [+0.0000, +0.0000]; decision_loss 0.0000; oracle_cf - cos +0.0000; frac_cf 0.990 |
| 6 | 0.3 | 0.9 | 1 | 10 dB | ods | 0.9941 | +0.0055 [+0.0053, +0.0058] | 0.9941 | +0.0055 [+0.0053, +0.0058] |  |
| 6 | 0.3 | 0.9 | 1 | 10 dB | csiqb | 0.9881 | -0.0005 [-0.0007, -0.0003] | 0.9881 | -0.0005 [-0.0007, -0.0003] |  |
| 6 | 0.3 | 0.9 | 1 | 10 dB | csiqw | 0.9939 | +0.0054 [+0.0052, +0.0056] | 0.9939 | +0.0054 [+0.0052, +0.0056] |  |
| 6 | 0.3 | 0.9 | 1 | 20 dB | fixed | 0.9973 | - | 0.9973 | - |  |
| 6 | 0.3 | 0.9 | 1 | 20 dB | csiq | 0.9979 | +0.0006 [+0.0006, +0.0006] | 0.9979 | +0.0006 [+0.0006, +0.0006] |  |
| 6 | 0.3 | 0.9 | 1 | 20 dB | csiqp | 0.9979 | +0.0006 [+0.0006, +0.0006] | 0.9979 | +0.0006 [+0.0006, +0.0006] |  |
| 6 | 0.3 | 0.9 | 1 | 20 dB | switch | 0.9979 | +0.0006 [+0.0006, +0.0006] | 0.9979 | +0.0006 [+0.0006, +0.0006] | frac 1.000; rule_loss +0.0000 [+0.0000, +0.0000]; decision_loss 0.0000; oracle_cf - cos +0.0000; frac_cf 0.988 |
| 6 | 0.3 | 0.9 | 1 | 20 dB | ods | 0.9979 | +0.0006 [+0.0006, +0.0006] | 0.9979 | +0.0006 [+0.0006, +0.0006] |  |
| 6 | 0.3 | 0.9 | 1 | 20 dB | csiqb | 0.9961 | -0.0012 [-0.0013, -0.0011] | 0.9961 | -0.0012 [-0.0013, -0.0011] |  |
| 6 | 0.3 | 0.9 | 1 | 20 dB | csiqw | 0.9979 | +0.0006 [+0.0006, +0.0006] | 0.9979 | +0.0006 [+0.0006, +0.0006] |  |
| 6 | 0.3 | 0.9 | 2 | 0 dB | fixed | 0.8758 | - | 0.8322 | - |  |
| 6 | 0.3 | 0.9 | 2 | 0 dB | csiq | 0.9465 | +0.0707 [+0.0685, +0.0728] | 0.9326 | +0.1004 [+0.0967, +0.1039] |  |
| 6 | 0.3 | 0.9 | 2 | 0 dB | csiqp | 0.9465 | +0.0707 [+0.0685, +0.0728] | 0.9326 | +0.1003 [+0.0966, +0.1038] |  |
| 6 | 0.3 | 0.9 | 2 | 0 dB | switch | 0.9465 | +0.0707 [+0.0685, +0.0728] | 0.9326 | +0.1004 [+0.0967, +0.1039] | frac 1.000; rule_loss +0.0000 [+0.0000, +0.0000]; decision_loss 0.0000; oracle_cf - cos +0.0000; frac_cf 0.999 |
| 6 | 0.3 | 0.9 | 2 | 0 dB | ods | 0.9447 | +0.0689 [+0.0667, +0.0710] | 0.9246 | +0.0923 [+0.0888, +0.0960] |  |
| 6 | 0.3 | 0.9 | 2 | 0 dB | csiqb | 0.9320 | +0.0562 [+0.0540, +0.0583] | 0.9176 | +0.0853 [+0.0817, +0.0889] |  |
| 6 | 0.3 | 0.9 | 2 | 0 dB | csiqw | 0.9431 | +0.0673 [+0.0651, +0.0694] | 0.9305 | +0.0982 [+0.0945, +0.1018] |  |
| 6 | 0.3 | 0.9 | 2 | 10 dB | fixed | 0.9738 | - | 0.9627 | - |  |
| 6 | 0.3 | 0.9 | 2 | 10 dB | csiq | 0.9904 | +0.0166 [+0.0159, +0.0173] | 0.9879 | +0.0252 [+0.0239, +0.0264] |  |
| 6 | 0.3 | 0.9 | 2 | 10 dB | csiqp | 0.9904 | +0.0166 [+0.0159, +0.0173] | 0.9878 | +0.0251 [+0.0239, +0.0263] |  |
| 6 | 0.3 | 0.9 | 2 | 10 dB | switch | 0.9904 | +0.0166 [+0.0159, +0.0173] | 0.9879 | +0.0252 [+0.0239, +0.0264] | frac 1.000; rule_loss +0.0000 [+0.0000, +0.0000]; decision_loss 0.0000; oracle_cf - cos +0.0000; frac_cf 0.999 |
| 6 | 0.3 | 0.9 | 2 | 10 dB | ods | 0.9899 | +0.0161 [+0.0154, +0.0167] | 0.9859 | +0.0232 [+0.0220, +0.0244] |  |
| 6 | 0.3 | 0.9 | 2 | 10 dB | csiqb | 0.9816 | +0.0078 [+0.0072, +0.0085] | 0.9774 | +0.0147 [+0.0135, +0.0159] |  |
| 6 | 0.3 | 0.9 | 2 | 10 dB | csiqw | 0.9901 | +0.0163 [+0.0156, +0.0170] | 0.9875 | +0.0248 [+0.0236, +0.0261] |  |
| 6 | 0.3 | 0.9 | 2 | 20 dB | fixed | 0.9954 | - | 0.9935 | - |  |
| 6 | 0.3 | 0.9 | 2 | 20 dB | csiq | 0.9975 | +0.0021 [+0.0020, +0.0022] | 0.9965 | +0.0030 [+0.0028, +0.0032] |  |
| 6 | 0.3 | 0.9 | 2 | 20 dB | csiqp | 0.9975 | +0.0021 [+0.0020, +0.0022] | 0.9965 | +0.0030 [+0.0028, +0.0032] |  |
| 6 | 0.3 | 0.9 | 2 | 20 dB | switch | 0.9975 | +0.0021 [+0.0020, +0.0022] | 0.9965 | +0.0030 [+0.0028, +0.0032] | frac 1.000; rule_loss +0.0000 [+0.0000, +0.0000]; decision_loss 0.0000; oracle_cf - cos +0.0000; frac_cf 0.997 |
| 6 | 0.3 | 0.9 | 2 | 20 dB | ods | 0.9974 | +0.0020 [+0.0019, +0.0021] | 0.9963 | +0.0028 [+0.0027, +0.0030] |  |
| 6 | 0.3 | 0.9 | 2 | 20 dB | csiqb | 0.9950 | -0.0004 [-0.0005, -0.0003] | 0.9932 | -0.0003 [-0.0005, -0.0001] |  |
| 6 | 0.3 | 0.9 | 2 | 20 dB | csiqw | 0.9974 | +0.0020 [+0.0019, +0.0022] | 0.9965 | +0.0030 [+0.0028, +0.0031] |  |
| 6 | 0.3 | 0.9 | 4 | 0 dB | fixed | 0.7824 | - | 0.6079 | - |  |
| 6 | 0.3 | 0.9 | 4 | 0 dB | csiq | 0.9133 | +0.1309 [+0.1280, +0.1341] | 0.8704 | +0.2625 [+0.2551, +0.2707] |  |
| 6 | 0.3 | 0.9 | 4 | 0 dB | csiqp | 0.9134 | +0.1310 [+0.1281, +0.1342] | 0.8703 | +0.2624 [+0.2550, +0.2707] |  |
| 6 | 0.3 | 0.9 | 4 | 0 dB | switch | 0.9133 | +0.1309 [+0.1280, +0.1341] | 0.8704 | +0.2625 [+0.2551, +0.2707] | frac 1.000; rule_loss +0.0000 [+0.0000, +0.0000]; decision_loss 0.0000; oracle_cf - cos -0.0000; frac_cf 1.000 |
| 6 | 0.3 | 0.9 | 4 | 0 dB | ods | 0.9109 | +0.1285 [+0.1257, +0.1317] | 0.8503 | +0.2424 [+0.2348, +0.2507] |  |
| 6 | 0.3 | 0.9 | 4 | 0 dB | csiqb | 0.8995 | +0.1171 [+0.1143, +0.1203] | 0.8571 | +0.2492 [+0.2417, +0.2574] |  |
| 6 | 0.3 | 0.9 | 4 | 0 dB | csiqw | 0.9081 | +0.1257 [+0.1228, +0.1289] | 0.8682 | +0.2603 [+0.2527, +0.2684] |  |
| 6 | 0.3 | 0.9 | 4 | 10 dB | fixed | 0.9310 | - | 0.8523 | - |  |
| 6 | 0.3 | 0.9 | 4 | 10 dB | csiq | 0.9812 | +0.0502 [+0.0484, +0.0519] | 0.9675 | +0.1152 [+0.1094, +0.1199] |  |
| 6 | 0.3 | 0.9 | 4 | 10 dB | csiqp | 0.9812 | +0.0502 [+0.0484, +0.0519] | 0.9675 | +0.1152 [+0.1095, +0.1199] |  |
| 6 | 0.3 | 0.9 | 4 | 10 dB | switch | 0.9812 | +0.0502 [+0.0484, +0.0519] | 0.9675 | +0.1152 [+0.1094, +0.1199] | frac 1.000; rule_loss +0.0000 [+0.0000, +0.0000]; decision_loss 0.0000; oracle_cf - cos +0.0000; frac_cf 0.999 |
| 6 | 0.3 | 0.9 | 4 | 10 dB | ods | 0.9812 | +0.0502 [+0.0483, +0.0518] | 0.9651 | +0.1127 [+0.1070, +0.1175] |  |
| 6 | 0.3 | 0.9 | 4 | 10 dB | csiqb | 0.9697 | +0.0387 [+0.0368, +0.0403] | 0.9520 | +0.0997 [+0.0939, +0.1045] |  |
| 6 | 0.3 | 0.9 | 4 | 10 dB | csiqw | 0.9807 | +0.0497 [+0.0478, +0.0513] | 0.9671 | +0.1147 [+0.1090, +0.1194] |  |
| 6 | 0.3 | 0.9 | 4 | 20 dB | fixed | 0.9851 | - | 0.9645 | - |  |
| 6 | 0.3 | 0.9 | 4 | 20 dB | csiq | 0.9961 | +0.0109 [+0.0104, +0.0116] | 0.9928 | +0.0283 [+0.0262, +0.0307] |  |
| 6 | 0.3 | 0.9 | 4 | 20 dB | csiqp | 0.9961 | +0.0110 [+0.0104, +0.0116] | 0.9928 | +0.0284 [+0.0263, +0.0308] |  |
| 6 | 0.3 | 0.9 | 4 | 20 dB | switch | 0.9961 | +0.0109 [+0.0104, +0.0116] | 0.9928 | +0.0283 [+0.0262, +0.0307] | frac 1.000; rule_loss +0.0000 [+0.0000, +0.0000]; decision_loss 0.0000; oracle_cf - cos +0.0000; frac_cf 0.997 |
| 6 | 0.3 | 0.9 | 4 | 20 dB | ods | 0.9962 | +0.0110 [+0.0104, +0.0117] | 0.9929 | +0.0285 [+0.0264, +0.0309] |  |
| 6 | 0.3 | 0.9 | 4 | 20 dB | csiqb | 0.9915 | +0.0064 [+0.0058, +0.0070] | 0.9851 | +0.0206 [+0.0185, +0.0230] |  |
| 6 | 0.3 | 0.9 | 4 | 20 dB | csiqw | 0.9958 | +0.0107 [+0.0101, +0.0114] | 0.9924 | +0.0279 [+0.0259, +0.0303] |  |
| 6 | 0.3 | 0.9 | 6 | 0 dB | fixed | 0.7457 | - | 0.4883 | - |  |
| 6 | 0.3 | 0.9 | 6 | 0 dB | csiq | 0.8894 | +0.1437 [+0.1410, +0.1465] | 0.8199 | +0.3315 [+0.3229, +0.3403] |  |
| 6 | 0.3 | 0.9 | 6 | 0 dB | csiqp | 0.8896 | +0.1439 [+0.1412, +0.1467] | 0.8202 | +0.3319 [+0.3231, +0.3406] |  |
| 6 | 0.3 | 0.9 | 6 | 0 dB | switch | 0.8894 | +0.1437 [+0.1410, +0.1465] | 0.8199 | +0.3315 [+0.3229, +0.3403] | frac 1.000; rule_loss +0.0000 [+0.0000, +0.0000]; decision_loss 0.0000; oracle_cf - cos +0.0000; frac_cf 1.000 |
| 6 | 0.3 | 0.9 | 6 | 0 dB | ods | 0.8869 | +0.1412 [+0.1386, +0.1440] | 0.7926 | +0.3042 [+0.2955, +0.3126] |  |
| 6 | 0.3 | 0.9 | 6 | 0 dB | csiqb | 0.8768 | +0.1311 [+0.1283, +0.1339] | 0.8092 | +0.3209 [+0.3124, +0.3296] |  |
| 6 | 0.3 | 0.9 | 6 | 0 dB | csiqw | 0.8835 | +0.1378 [+0.1351, +0.1406] | 0.8178 | +0.3295 [+0.3208, +0.3381] |  |
| 6 | 0.3 | 0.9 | 6 | 10 dB | fixed | 0.9108 | - | 0.7844 | - |  |
| 6 | 0.3 | 0.9 | 6 | 10 dB | csiq | 0.9671 | +0.0563 [+0.0547, +0.0578] | 0.9427 | +0.1583 [+0.1517, +0.1647] |  |
| 6 | 0.3 | 0.9 | 6 | 10 dB | csiqp | 0.9671 | +0.0563 [+0.0547, +0.0579] | 0.9431 | +0.1587 [+0.1521, +0.1651] |  |
| 6 | 0.3 | 0.9 | 6 | 10 dB | switch | 0.9671 | +0.0563 [+0.0547, +0.0578] | 0.9427 | +0.1583 [+0.1517, +0.1647] | frac 1.000; rule_loss +0.0000 [+0.0000, +0.0000]; decision_loss 0.0000; oracle_cf - cos +0.0000; frac_cf 1.000 |
| 6 | 0.3 | 0.9 | 6 | 10 dB | ods | 0.9674 | +0.0566 [+0.0550, +0.0582] | 0.9422 | +0.1578 [+0.1512, +0.1642] |  |
| 6 | 0.3 | 0.9 | 6 | 10 dB | csiqb | 0.9569 | +0.0461 [+0.0445, +0.0476] | 0.9280 | +0.1436 [+0.1370, +0.1501] |  |
| 6 | 0.3 | 0.9 | 6 | 10 dB | csiqw | 0.9666 | +0.0559 [+0.0542, +0.0574] | 0.9422 | +0.1578 [+0.1512, +0.1643] |  |
| 6 | 0.3 | 0.9 | 6 | 20 dB | fixed | 0.9711 | - | 0.9363 | - |  |
| 6 | 0.3 | 0.9 | 6 | 20 dB | csiq | 0.9833 | +0.0122 [+0.0116, +0.0128] | 0.9752 | +0.0389 [+0.0361, +0.0417] |  |
| 6 | 0.3 | 0.9 | 6 | 20 dB | csiqp | 0.9833 | +0.0122 [+0.0116, +0.0128] | 0.9755 | +0.0391 [+0.0364, +0.0420] |  |
| 6 | 0.3 | 0.9 | 6 | 20 dB | switch | 0.9833 | +0.0122 [+0.0116, +0.0128] | 0.9752 | +0.0389 [+0.0361, +0.0417] | frac 1.000; rule_loss +0.0000 [+0.0000, +0.0000]; decision_loss 0.0000; oracle_cf - cos +0.0000; frac_cf 1.000 |
| 6 | 0.3 | 0.9 | 6 | 20 dB | ods | 0.9835 | +0.0124 [+0.0118, +0.0130] | 0.9760 | +0.0396 [+0.0368, +0.0425] |  |
| 6 | 0.3 | 0.9 | 6 | 20 dB | csiqb | 0.9796 | +0.0085 [+0.0079, +0.0090] | 0.9681 | +0.0318 [+0.0289, +0.0346] |  |
| 6 | 0.3 | 0.9 | 6 | 20 dB | csiqw | 0.9831 | +0.0120 [+0.0114, +0.0125] | 0.9747 | +0.0384 [+0.0356, +0.0413] |  |
| 6 | 0.3 | 0.7 | 1 | 0 dB | fixed | 0.9279 | - | 0.9279 | - |  |
| 6 | 0.3 | 0.7 | 1 | 0 dB | csiq | 0.9457 | +0.0177 [+0.0157, +0.0197] | 0.9457 | +0.0177 [+0.0157, +0.0197] |  |
| 6 | 0.3 | 0.7 | 1 | 0 dB | csiqp | 0.9463 | +0.0183 [+0.0163, +0.0203] | 0.9463 | +0.0183 [+0.0163, +0.0203] |  |
| 6 | 0.3 | 0.7 | 1 | 0 dB | switch | 0.9447 | +0.0168 [+0.0148, +0.0188] | 0.9447 | +0.0168 [+0.0148, +0.0188] | frac 0.880; rule_loss +0.0009 [+0.0005, +0.0013]; decision_loss 0.0070; oracle_cf - cos +0.0068; frac_cf 0.811 |
| 6 | 0.3 | 0.7 | 1 | 0 dB | ods | 0.9457 | +0.0177 [+0.0157, +0.0197] | 0.9457 | +0.0177 [+0.0157, +0.0197] |  |
| 6 | 0.3 | 0.7 | 1 | 0 dB | csiqb | 0.9346 | +0.0067 [+0.0050, +0.0084] | 0.9346 | +0.0067 [+0.0050, +0.0084] |  |
| 6 | 0.3 | 0.7 | 1 | 0 dB | csiqw | 0.9457 | +0.0177 [+0.0158, +0.0197] | 0.9457 | +0.0177 [+0.0158, +0.0197] |  |
| 6 | 0.3 | 0.7 | 1 | 10 dB | fixed | 0.9885 | - | 0.9885 | - |  |
| 6 | 0.3 | 0.7 | 1 | 10 dB | csiq | 0.9892 | +0.0007 [+0.0001, +0.0012] | 0.9892 | +0.0007 [+0.0001, +0.0012] |  |
| 6 | 0.3 | 0.7 | 1 | 10 dB | csiqp | 0.9892 | +0.0007 [+0.0001, +0.0012] | 0.9892 | +0.0007 [+0.0001, +0.0012] |  |
| 6 | 0.3 | 0.7 | 1 | 10 dB | switch | 0.9896 | +0.0011 [+0.0008, +0.0014] | 0.9896 | +0.0011 [+0.0008, +0.0014] | frac 0.621; rule_loss -0.0005 [-0.0010, -0.0000]; decision_loss 0.0017; oracle_cf - cos +0.0017; frac_cf 0.715 |
| 6 | 0.3 | 0.7 | 1 | 10 dB | ods | 0.9892 | +0.0007 [+0.0001, +0.0012] | 0.9892 | +0.0007 [+0.0001, +0.0012] |  |
| 6 | 0.3 | 0.7 | 1 | 10 dB | csiqb | 0.9821 | -0.0064 [-0.0069, -0.0060] | 0.9821 | -0.0064 [-0.0069, -0.0060] |  |
| 6 | 0.3 | 0.7 | 1 | 10 dB | csiqw | 0.9891 | +0.0006 [+0.0000, +0.0011] | 0.9891 | +0.0006 [+0.0000, +0.0011] |  |
| 6 | 0.3 | 0.7 | 1 | 20 dB | fixed | 0.9973 | - | 0.9973 | - |  |
| 6 | 0.3 | 0.7 | 1 | 20 dB | csiq | 0.9972 | -0.0001 [-0.0002, -0.0000] | 0.9972 | -0.0001 [-0.0002, -0.0000] |  |
| 6 | 0.3 | 0.7 | 1 | 20 dB | csiqp | 0.9972 | -0.0001 [-0.0002, -0.0000] | 0.9972 | -0.0001 [-0.0002, -0.0000] |  |
| 6 | 0.3 | 0.7 | 1 | 20 dB | switch | 0.9973 | +0.0000 [-0.0000, +0.0001] | 0.9973 | +0.0000 [-0.0000, +0.0001] | frac 0.421; rule_loss -0.0000 [-0.0001, +0.0000]; decision_loss 0.0003; oracle_cf - cos +0.0003; frac_cf 0.694 |
| 6 | 0.3 | 0.7 | 1 | 20 dB | ods | 0.9972 | -0.0001 [-0.0002, -0.0000] | 0.9972 | -0.0001 [-0.0002, -0.0000] |  |
| 6 | 0.3 | 0.7 | 1 | 20 dB | csiqb | 0.9947 | -0.0026 [-0.0027, -0.0025] | 0.9947 | -0.0026 [-0.0027, -0.0025] |  |
| 6 | 0.3 | 0.7 | 1 | 20 dB | csiqw | 0.9972 | -0.0002 [-0.0003, -0.0001] | 0.9972 | -0.0002 [-0.0003, -0.0001] |  |
| 6 | 0.3 | 0.7 | 2 | 0 dB | fixed | 0.8750 | - | 0.8311 | - |  |
| 6 | 0.3 | 0.7 | 2 | 0 dB | csiq | 0.9180 | +0.0430 [+0.0404, +0.0456] | 0.8913 | +0.0602 [+0.0556, +0.0646] |  |
| 6 | 0.3 | 0.7 | 2 | 0 dB | csiqp | 0.9192 | +0.0442 [+0.0416, +0.0467] | 0.8929 | +0.0618 [+0.0573, +0.0661] |  |
| 6 | 0.3 | 0.7 | 2 | 0 dB | switch | 0.9179 | +0.0429 [+0.0403, +0.0455] | 0.8912 | +0.0601 [+0.0556, +0.0644] | frac 0.994; rule_loss +0.0001 [-0.0000, +0.0003]; decision_loss 0.0053; oracle_cf - cos +0.0051; frac_cf 0.875 |
| 6 | 0.3 | 0.7 | 2 | 0 dB | ods | 0.9163 | +0.0413 [+0.0387, +0.0439] | 0.8835 | +0.0524 [+0.0478, +0.0569] |  |
| 6 | 0.3 | 0.7 | 2 | 0 dB | csiqb | 0.9080 | +0.0330 [+0.0304, +0.0354] | 0.8837 | +0.0526 [+0.0483, +0.0569] |  |
| 6 | 0.3 | 0.7 | 2 | 0 dB | csiqw | 0.9172 | +0.0422 [+0.0396, +0.0447] | 0.8923 | +0.0611 [+0.0566, +0.0654] |  |
| 6 | 0.3 | 0.7 | 2 | 10 dB | fixed | 0.9738 | - | 0.9625 | - |  |
| 6 | 0.3 | 0.7 | 2 | 10 dB | csiq | 0.9823 | +0.0086 [+0.0077, +0.0094] | 0.9758 | +0.0133 [+0.0117, +0.0147] |  |
| 6 | 0.3 | 0.7 | 2 | 10 dB | csiqp | 0.9824 | +0.0087 [+0.0078, +0.0094] | 0.9759 | +0.0134 [+0.0118, +0.0148] |  |
| 6 | 0.3 | 0.7 | 2 | 10 dB | switch | 0.9823 | +0.0085 [+0.0076, +0.0093] | 0.9757 | +0.0131 [+0.0115, +0.0146] | frac 0.957; rule_loss +0.0001 [-0.0001, +0.0003]; decision_loss 0.0020; oracle_cf - cos +0.0019; frac_cf 0.839 |
| 6 | 0.3 | 0.7 | 2 | 10 dB | ods | 0.9817 | +0.0079 [+0.0070, +0.0087] | 0.9733 | +0.0108 [+0.0093, +0.0124] |  |
| 6 | 0.3 | 0.7 | 2 | 10 dB | csiqb | 0.9730 | -0.0008 [-0.0015, +0.0000] | 0.9655 | +0.0030 [+0.0015, +0.0044] |  |
| 6 | 0.3 | 0.7 | 2 | 10 dB | csiqw | 0.9823 | +0.0086 [+0.0077, +0.0093] | 0.9760 | +0.0134 [+0.0119, +0.0149] |  |
| 6 | 0.3 | 0.7 | 2 | 20 dB | fixed | 0.9953 | - | 0.9934 | - |  |
| 6 | 0.3 | 0.7 | 2 | 20 dB | csiq | 0.9961 | +0.0008 [+0.0006, +0.0009] | 0.9945 | +0.0011 [+0.0008, +0.0013] |  |
| 6 | 0.3 | 0.7 | 2 | 20 dB | csiqp | 0.9961 | +0.0008 [+0.0006, +0.0009] | 0.9945 | +0.0011 [+0.0008, +0.0013] |  |
| 6 | 0.3 | 0.7 | 2 | 20 dB | switch | 0.9961 | +0.0008 [+0.0007, +0.0009] | 0.9945 | +0.0011 [+0.0009, +0.0014] | frac 0.838; rule_loss +0.0000 [-0.0001, +0.0001]; decision_loss 0.0005; oracle_cf - cos +0.0005; frac_cf 0.789 |
| 6 | 0.3 | 0.7 | 2 | 20 dB | ods | 0.9960 | +0.0007 [+0.0005, +0.0008] | 0.9942 | +0.0008 [+0.0004, +0.0010] |  |
| 6 | 0.3 | 0.7 | 2 | 20 dB | csiqb | 0.9926 | -0.0027 [-0.0028, -0.0025] | 0.9899 | -0.0034 [-0.0037, -0.0032] |  |
| 6 | 0.3 | 0.7 | 2 | 20 dB | csiqw | 0.9960 | +0.0007 [+0.0006, +0.0009] | 0.9944 | +0.0010 [+0.0007, +0.0013] |  |
| 6 | 0.3 | 0.7 | 4 | 0 dB | fixed | 0.7819 | - | 0.6075 | - |  |
| 6 | 0.3 | 0.7 | 4 | 0 dB | csiq | 0.8756 | +0.0937 [+0.0904, +0.0972] | 0.7915 | +0.1840 [+0.1750, +0.1925] |  |
| 6 | 0.3 | 0.7 | 4 | 0 dB | csiqp | 0.8773 | +0.0954 [+0.0921, +0.0989] | 0.7943 | +0.1868 [+0.1777, +0.1954] |  |
| 6 | 0.3 | 0.7 | 4 | 0 dB | switch | 0.8756 | +0.0937 [+0.0904, +0.0972] | 0.7915 | +0.1840 [+0.1750, +0.1925] | frac 1.000; rule_loss +0.0000 [+0.0000, +0.0000]; decision_loss 0.0021; oracle_cf - cos +0.0019; frac_cf 0.940 |
| 6 | 0.3 | 0.7 | 4 | 0 dB | ods | 0.8733 | +0.0914 [+0.0881, +0.0949] | 0.7738 | +0.1663 [+0.1571, +0.1753] |  |
| 6 | 0.3 | 0.7 | 4 | 0 dB | csiqb | 0.8672 | +0.0854 [+0.0821, +0.0888] | 0.7915 | +0.1841 [+0.1754, +0.1927] |  |
| 6 | 0.3 | 0.7 | 4 | 0 dB | csiqw | 0.8742 | +0.0923 [+0.0890, +0.0957] | 0.7945 | +0.1871 [+0.1783, +0.1956] |  |
| 6 | 0.3 | 0.7 | 4 | 10 dB | fixed | 0.9310 | - | 0.8522 | - |  |
| 6 | 0.3 | 0.7 | 4 | 10 dB | csiq | 0.9681 | +0.0372 [+0.0356, +0.0389] | 0.9375 | +0.0854 [+0.0803, +0.0907] |  |
| 6 | 0.3 | 0.7 | 4 | 10 dB | csiqp | 0.9685 | +0.0376 [+0.0360, +0.0392] | 0.9390 | +0.0869 [+0.0819, +0.0922] |  |
| 6 | 0.3 | 0.7 | 4 | 10 dB | switch | 0.9681 | +0.0372 [+0.0356, +0.0389] | 0.9375 | +0.0854 [+0.0803, +0.0907] | frac 1.000; rule_loss +0.0000 [+0.0000, +0.0000]; decision_loss 0.0011; oracle_cf - cos +0.0011; frac_cf 0.939 |
| 6 | 0.3 | 0.7 | 4 | 10 dB | ods | 0.9679 | +0.0370 [+0.0354, +0.0387] | 0.9348 | +0.0826 [+0.0776, +0.0881] |  |
| 6 | 0.3 | 0.7 | 4 | 10 dB | csiqb | 0.9578 | +0.0268 [+0.0252, +0.0285] | 0.9280 | +0.0758 [+0.0709, +0.0809] |  |
| 6 | 0.3 | 0.7 | 4 | 10 dB | csiqw | 0.9684 | +0.0374 [+0.0358, +0.0391] | 0.9391 | +0.0869 [+0.0819, +0.0922] |  |
| 6 | 0.3 | 0.7 | 4 | 20 dB | fixed | 0.9849 | - | 0.9644 | - |  |
| 6 | 0.3 | 0.7 | 4 | 20 dB | csiq | 0.9935 | +0.0086 [+0.0080, +0.0092] | 0.9866 | +0.0223 [+0.0201, +0.0247] |  |
| 6 | 0.3 | 0.7 | 4 | 20 dB | csiqp | 0.9936 | +0.0086 [+0.0080, +0.0093] | 0.9869 | +0.0225 [+0.0204, +0.0249] |  |
| 6 | 0.3 | 0.7 | 4 | 20 dB | switch | 0.9935 | +0.0086 [+0.0080, +0.0092] | 0.9866 | +0.0223 [+0.0201, +0.0247] | frac 1.000; rule_loss +0.0000 [+0.0000, +0.0000]; decision_loss 0.0004; oracle_cf - cos +0.0004; frac_cf 0.925 |
| 6 | 0.3 | 0.7 | 4 | 20 dB | ods | 0.9936 | +0.0086 [+0.0081, +0.0093] | 0.9868 | +0.0224 [+0.0202, +0.0248] |  |
| 6 | 0.3 | 0.7 | 4 | 20 dB | csiqb | 0.9882 | +0.0033 [+0.0027, +0.0039] | 0.9784 | +0.0141 [+0.0120, +0.0164] |  |
| 6 | 0.3 | 0.7 | 4 | 20 dB | csiqw | 0.9934 | +0.0085 [+0.0079, +0.0091] | 0.9867 | +0.0223 [+0.0202, +0.0248] |  |
| 6 | 0.3 | 0.7 | 6 | 0 dB | fixed | 0.7441 | - | 0.4852 | - |  |
| 6 | 0.3 | 0.7 | 6 | 0 dB | csiq | 0.8477 | +0.1036 [+0.1006, +0.1068] | 0.7133 | +0.2281 [+0.2186, +0.2383] |  |
| 6 | 0.3 | 0.7 | 6 | 0 dB | csiqp | 0.8496 | +0.1055 [+0.1025, +0.1086] | 0.7173 | +0.2321 [+0.2228, +0.2424] |  |
| 6 | 0.3 | 0.7 | 6 | 0 dB | switch | 0.8477 | +0.1036 [+0.1006, +0.1068] | 0.7133 | +0.2281 [+0.2186, +0.2383] | frac 1.000; rule_loss +0.0000 [+0.0000, +0.0000]; decision_loss 0.0013; oracle_cf - cos +0.0009; frac_cf 0.967 |
| 6 | 0.3 | 0.7 | 6 | 0 dB | ods | 0.8446 | +0.1005 [+0.0975, +0.1036] | 0.6873 | +0.2021 [+0.1922, +0.2128] |  |
| 6 | 0.3 | 0.7 | 6 | 0 dB | csiqb | 0.8406 | +0.0965 [+0.0937, +0.0996] | 0.7192 | +0.2340 [+0.2248, +0.2446] |  |
| 6 | 0.3 | 0.7 | 6 | 0 dB | csiqw | 0.8460 | +0.1019 [+0.0989, +0.1050] | 0.7173 | +0.2321 [+0.2226, +0.2425] |  |
| 6 | 0.3 | 0.7 | 6 | 10 dB | fixed | 0.9117 | - | 0.7884 | - |  |
| 6 | 0.3 | 0.7 | 6 | 10 dB | csiq | 0.9533 | +0.0416 [+0.0401, +0.0432] | 0.9025 | +0.1141 [+0.1072, +0.1204] |  |
| 6 | 0.3 | 0.7 | 6 | 10 dB | csiqp | 0.9538 | +0.0421 [+0.0406, +0.0436] | 0.9044 | +0.1160 [+0.1092, +0.1222] |  |
| 6 | 0.3 | 0.7 | 6 | 10 dB | switch | 0.9533 | +0.0416 [+0.0401, +0.0432] | 0.9025 | +0.1141 [+0.1072, +0.1204] | frac 1.000; rule_loss +0.0000 [+0.0000, +0.0000]; decision_loss 0.0008; oracle_cf - cos +0.0007; frac_cf 0.960 |
| 6 | 0.3 | 0.7 | 6 | 10 dB | ods | 0.9533 | +0.0417 [+0.0401, +0.0432] | 0.9015 | +0.1131 [+0.1062, +0.1196] |  |
| 6 | 0.3 | 0.7 | 6 | 10 dB | csiqb | 0.9441 | +0.0325 [+0.0309, +0.0340] | 0.8962 | +0.1078 [+0.1013, +0.1139] |  |
| 6 | 0.3 | 0.7 | 6 | 10 dB | csiqw | 0.9536 | +0.0419 [+0.0403, +0.0434] | 0.9045 | +0.1161 [+0.1093, +0.1224] |  |
| 6 | 0.3 | 0.7 | 6 | 20 dB | fixed | 0.9707 | - | 0.9345 | - |  |
| 6 | 0.3 | 0.7 | 6 | 20 dB | csiq | 0.9804 | +0.0096 [+0.0090, +0.0103] | 0.9664 | +0.0318 [+0.0288, +0.0351] |  |
| 6 | 0.3 | 0.7 | 6 | 20 dB | csiqp | 0.9805 | +0.0097 [+0.0092, +0.0104] | 0.9670 | +0.0325 [+0.0294, +0.0358] |  |
| 6 | 0.3 | 0.7 | 6 | 20 dB | switch | 0.9804 | +0.0096 [+0.0090, +0.0103] | 0.9664 | +0.0318 [+0.0288, +0.0351] | frac 1.000; rule_loss +0.0000 [+0.0000, +0.0000]; decision_loss 0.0003; oracle_cf - cos +0.0003; frac_cf 0.934 |
| 6 | 0.3 | 0.7 | 6 | 20 dB | ods | 0.9806 | +0.0099 [+0.0093, +0.0105] | 0.9675 | +0.0330 [+0.0299, +0.0363] |  |
| 6 | 0.3 | 0.7 | 6 | 20 dB | csiqb | 0.9760 | +0.0053 [+0.0047, +0.0059] | 0.9591 | +0.0245 [+0.0215, +0.0278] |  |
| 6 | 0.3 | 0.7 | 6 | 20 dB | csiqw | 0.9803 | +0.0096 [+0.0090, +0.0102] | 0.9666 | +0.0320 [+0.0289, +0.0354] |  |
| 6 | 0.3 | 0.5 | 1 | 0 dB | fixed | 0.9274 | - | 0.9274 | - |  |
| 6 | 0.3 | 0.5 | 1 | 0 dB | csiq | 0.9205 | -0.0070 [-0.0101, -0.0039] | 0.9205 | -0.0070 [-0.0101, -0.0039] |  |
| 6 | 0.3 | 0.5 | 1 | 0 dB | csiqp | 0.9227 | -0.0047 [-0.0076, -0.0018] | 0.9227 | -0.0047 [-0.0076, -0.0018] |  |
| 6 | 0.3 | 0.5 | 1 | 0 dB | switch | 0.9285 | +0.0010 [-0.0004, +0.0024] | 0.9285 | +0.0010 [-0.0004, +0.0024] | frac 0.213; rule_loss -0.0010 [-0.0024, +0.0004]; decision_loss 0.0133; oracle_cf - cos +0.0130; frac_cf 0.537 |
| 6 | 0.3 | 0.5 | 1 | 0 dB | ods | 0.9205 | -0.0070 [-0.0101, -0.0039] | 0.9205 | -0.0070 [-0.0101, -0.0039] |  |
| 6 | 0.3 | 0.5 | 1 | 0 dB | csiqb | 0.9139 | -0.0136 [-0.0161, -0.0111] | 0.9139 | -0.0136 [-0.0161, -0.0111] |  |
| 6 | 0.3 | 0.5 | 1 | 0 dB | csiqw | 0.9221 | -0.0053 [-0.0083, -0.0024] | 0.9221 | -0.0053 [-0.0083, -0.0024] |  |
| 6 | 0.3 | 0.5 | 1 | 10 dB | fixed | 0.9885 | - | 0.9885 | - |  |
| 6 | 0.3 | 0.5 | 1 | 10 dB | csiq | 0.9822 | -0.0063 [-0.0075, -0.0052] | 0.9822 | -0.0063 [-0.0075, -0.0052] |  |
| 6 | 0.3 | 0.5 | 1 | 10 dB | csiqp | 0.9824 | -0.0061 [-0.0072, -0.0050] | 0.9824 | -0.0061 [-0.0072, -0.0050] |  |
| 6 | 0.3 | 0.5 | 1 | 10 dB | switch | 0.9884 | -0.0001 [-0.0002, +0.0000] | 0.9884 | -0.0001 [-0.0002, +0.0000] | frac 0.021; rule_loss +0.0001 [-0.0000, +0.0002]; decision_loss 0.0014; oracle_cf - cos +0.0013; frac_cf 0.386 |
| 6 | 0.3 | 0.5 | 1 | 10 dB | ods | 0.9822 | -0.0063 [-0.0075, -0.0052] | 0.9822 | -0.0063 [-0.0075, -0.0052] |  |
| 6 | 0.3 | 0.5 | 1 | 10 dB | csiqb | 0.9751 | -0.0134 [-0.0144, -0.0125] | 0.9751 | -0.0134 [-0.0144, -0.0125] |  |
| 6 | 0.3 | 0.5 | 1 | 10 dB | csiqw | 0.9825 | -0.0060 [-0.0072, -0.0050] | 0.9825 | -0.0060 [-0.0072, -0.0050] |  |
| 6 | 0.3 | 0.5 | 1 | 20 dB | fixed | 0.9973 | - | 0.9973 | - |  |
| 6 | 0.3 | 0.5 | 1 | 20 dB | csiq | 0.9958 | -0.0015 [-0.0023, -0.0011] | 0.9958 | -0.0015 [-0.0023, -0.0011] |  |
| 6 | 0.3 | 0.5 | 1 | 20 dB | csiqp | 0.9959 | -0.0015 [-0.0020, -0.0011] | 0.9959 | -0.0015 [-0.0020, -0.0011] |  |
| 6 | 0.3 | 0.5 | 1 | 20 dB | switch | 0.9973 | -0.0000 [-0.0000, +0.0000] | 0.9973 | -0.0000 [-0.0000, +0.0000] | frac 0.004; rule_loss +0.0000 [-0.0000, +0.0000]; decision_loss 0.0001; oracle_cf - cos +0.0001; frac_cf 0.367 |
| 6 | 0.3 | 0.5 | 1 | 20 dB | ods | 0.9958 | -0.0015 [-0.0023, -0.0011] | 0.9958 | -0.0015 [-0.0023, -0.0011] |  |
| 6 | 0.3 | 0.5 | 1 | 20 dB | csiqb | 0.9931 | -0.0042 [-0.0046, -0.0039] | 0.9931 | -0.0042 [-0.0046, -0.0039] |  |
| 6 | 0.3 | 0.5 | 1 | 20 dB | csiqw | 0.9959 | -0.0014 [-0.0019, -0.0011] | 0.9959 | -0.0014 [-0.0019, -0.0011] |  |
| 6 | 0.3 | 0.5 | 2 | 0 dB | fixed | 0.8745 | - | 0.8300 | - |  |
| 6 | 0.3 | 0.5 | 2 | 0 dB | csiq | 0.8810 | +0.0065 [+0.0033, +0.0095] | 0.8361 | +0.0061 [+0.0004, +0.0115] |  |
| 6 | 0.3 | 0.5 | 2 | 0 dB | csiqp | 0.8849 | +0.0104 [+0.0072, +0.0134] | 0.8417 | +0.0117 [+0.0059, +0.0169] |  |
| 6 | 0.3 | 0.5 | 2 | 0 dB | switch | 0.8808 | +0.0063 [+0.0036, +0.0088] | 0.8373 | +0.0073 [+0.0023, +0.0118] | frac 0.650; rule_loss +0.0002 [-0.0016, +0.0020]; decision_loss 0.0224; oracle_cf - cos +0.0218; frac_cf 0.617 |
| 6 | 0.3 | 0.5 | 2 | 0 dB | ods | 0.8798 | +0.0053 [+0.0019, +0.0084] | 0.8292 | -0.0008 [-0.0067, +0.0050] |  |
| 6 | 0.3 | 0.5 | 2 | 0 dB | csiqb | 0.8777 | +0.0032 [+0.0004, +0.0059] | 0.8395 | +0.0095 [+0.0043, +0.0143] |  |
| 6 | 0.3 | 0.5 | 2 | 0 dB | csiqw | 0.8832 | +0.0087 [+0.0057, +0.0115] | 0.8409 | +0.0109 [+0.0051, +0.0162] |  |
| 6 | 0.3 | 0.5 | 2 | 10 dB | fixed | 0.9739 | - | 0.9627 | - |  |
| 6 | 0.3 | 0.5 | 2 | 10 dB | csiq | 0.9708 | -0.0031 [-0.0044, -0.0019] | 0.9583 | -0.0044 [-0.0068, -0.0021] |  |
| 6 | 0.3 | 0.5 | 2 | 10 dB | csiqp | 0.9715 | -0.0024 [-0.0036, -0.0012] | 0.9595 | -0.0032 [-0.0054, -0.0010] |  |
| 6 | 0.3 | 0.5 | 2 | 10 dB | switch | 0.9739 | -0.0000 [-0.0007, +0.0006] | 0.9629 | +0.0002 [-0.0010, +0.0014] | frac 0.288; rule_loss +0.0000 [-0.0006, +0.0007]; decision_loss 0.0059; oracle_cf - cos +0.0058; frac_cf 0.532 |
| 6 | 0.3 | 0.5 | 2 | 10 dB | ods | 0.9700 | -0.0039 [-0.0051, -0.0026] | 0.9552 | -0.0075 [-0.0099, -0.0051] |  |
| 6 | 0.3 | 0.5 | 2 | 10 dB | csiqb | 0.9628 | -0.0111 [-0.0121, -0.0101] | 0.9513 | -0.0114 [-0.0131, -0.0096] |  |
| 6 | 0.3 | 0.5 | 2 | 10 dB | csiqw | 0.9716 | -0.0023 [-0.0034, -0.0011] | 0.9599 | -0.0027 [-0.0049, -0.0006] |  |
| 6 | 0.3 | 0.5 | 2 | 20 dB | fixed | 0.9952 | - | 0.9933 | - |  |
| 6 | 0.3 | 0.5 | 2 | 20 dB | csiq | 0.9939 | -0.0014 [-0.0019, -0.0010] | 0.9908 | -0.0025 [-0.0035, -0.0017] |  |
| 6 | 0.3 | 0.5 | 2 | 20 dB | csiqp | 0.9939 | -0.0014 [-0.0019, -0.0009] | 0.9908 | -0.0024 [-0.0035, -0.0016] |  |
| 6 | 0.3 | 0.5 | 2 | 20 dB | switch | 0.9952 | -0.0001 [-0.0002, +0.0000] | 0.9931 | -0.0001 [-0.0003, +0.0001] | frac 0.105; rule_loss +0.0001 [-0.0000, +0.0002]; decision_loss 0.0008; oracle_cf - cos +0.0008; frac_cf 0.469 |
| 6 | 0.3 | 0.5 | 2 | 20 dB | ods | 0.9937 | -0.0016 [-0.0020, -0.0012] | 0.9903 | -0.0030 [-0.0039, -0.0022] |  |
| 6 | 0.3 | 0.5 | 2 | 20 dB | csiqb | 0.9900 | -0.0053 [-0.0056, -0.0050] | 0.9860 | -0.0072 [-0.0079, -0.0066] |  |
| 6 | 0.3 | 0.5 | 2 | 20 dB | csiqw | 0.9939 | -0.0013 [-0.0018, -0.0010] | 0.9909 | -0.0023 [-0.0032, -0.0016] |  |
| 6 | 0.3 | 0.5 | 4 | 0 dB | fixed | 0.7821 | - | 0.6090 | - |  |
| 6 | 0.3 | 0.5 | 4 | 0 dB | csiq | 0.8308 | +0.0487 [+0.0450, +0.0528] | 0.7013 | +0.0923 [+0.0819, +0.1024] |  |
| 6 | 0.3 | 0.5 | 4 | 0 dB | csiqp | 0.8362 | +0.0540 [+0.0504, +0.0581] | 0.7101 | +0.1011 [+0.0910, +0.1111] |  |
| 6 | 0.3 | 0.5 | 4 | 0 dB | switch | 0.8308 | +0.0486 [+0.0449, +0.0527] | 0.7011 | +0.0921 [+0.0817, +0.1023] | frac 0.999; rule_loss +0.0001 [+0.0000, +0.0002]; decision_loss 0.0128; oracle_cf - cos +0.0120; frac_cf 0.756 |
| 6 | 0.3 | 0.5 | 4 | 0 dB | ods | 0.8297 | +0.0475 [+0.0438, +0.0514] | 0.6876 | +0.0786 [+0.0690, +0.0885] |  |
| 6 | 0.3 | 0.5 | 4 | 0 dB | csiqb | 0.8297 | +0.0476 [+0.0442, +0.0513] | 0.7163 | +0.1073 [+0.0975, +0.1169] |  |
| 6 | 0.3 | 0.5 | 4 | 0 dB | csiqw | 0.8337 | +0.0516 [+0.0480, +0.0555] | 0.7104 | +0.1014 [+0.0915, +0.1112] |  |
| 6 | 0.3 | 0.5 | 4 | 10 dB | fixed | 0.9311 | - | 0.8533 | - |  |
| 6 | 0.3 | 0.5 | 4 | 10 dB | csiq | 0.9515 | +0.0204 [+0.0185, +0.0222] | 0.8982 | +0.0449 [+0.0388, +0.0509] |  |
| 6 | 0.3 | 0.5 | 4 | 10 dB | csiqp | 0.9532 | +0.0221 [+0.0202, +0.0239] | 0.9036 | +0.0503 [+0.0442, +0.0563] |  |
| 6 | 0.3 | 0.5 | 4 | 10 dB | switch | 0.9513 | +0.0203 [+0.0183, +0.0221] | 0.8977 | +0.0444 [+0.0383, +0.0506] | frac 0.991; rule_loss +0.0002 [+0.0001, +0.0003]; decision_loss 0.0068; oracle_cf - cos +0.0067; frac_cf 0.758 |
| 6 | 0.3 | 0.5 | 4 | 10 dB | ods | 0.9515 | +0.0204 [+0.0185, +0.0222] | 0.8968 | +0.0435 [+0.0376, +0.0495] |  |
| 6 | 0.3 | 0.5 | 4 | 10 dB | csiqb | 0.9439 | +0.0128 [+0.0110, +0.0145] | 0.8983 | +0.0449 [+0.0394, +0.0505] |  |
| 6 | 0.3 | 0.5 | 4 | 10 dB | csiqw | 0.9531 | +0.0220 [+0.0201, +0.0238] | 0.9029 | +0.0496 [+0.0435, +0.0556] |  |
| 6 | 0.3 | 0.5 | 4 | 20 dB | fixed | 0.9851 | - | 0.9655 | - |  |
| 6 | 0.3 | 0.5 | 4 | 20 dB | csiq | 0.9895 | +0.0044 [+0.0037, +0.0051] | 0.9758 | +0.0104 [+0.0080, +0.0128] |  |
| 6 | 0.3 | 0.5 | 4 | 20 dB | csiqp | 0.9899 | +0.0048 [+0.0041, +0.0054] | 0.9770 | +0.0115 [+0.0093, +0.0140] |  |
| 6 | 0.3 | 0.5 | 4 | 20 dB | switch | 0.9893 | +0.0042 [+0.0035, +0.0048] | 0.9752 | +0.0098 [+0.0074, +0.0121] | frac 0.943; rule_loss +0.0002 [+0.0001, +0.0004]; decision_loss 0.0023; oracle_cf - cos +0.0023; frac_cf 0.729 |
| 6 | 0.3 | 0.5 | 4 | 20 dB | ods | 0.9895 | +0.0044 [+0.0037, +0.0051] | 0.9756 | +0.0102 [+0.0077, +0.0128] |  |
| 6 | 0.3 | 0.5 | 4 | 20 dB | csiqb | 0.9844 | -0.0007 [-0.0013, -0.0001] | 0.9697 | +0.0043 [+0.0022, +0.0064] |  |
| 6 | 0.3 | 0.5 | 4 | 20 dB | csiqw | 0.9899 | +0.0047 [+0.0041, +0.0054] | 0.9772 | +0.0118 [+0.0095, +0.0142] |  |
| 6 | 0.3 | 0.5 | 6 | 0 dB | fixed | 0.7433 | - | 0.4841 | - |  |
| 6 | 0.3 | 0.5 | 6 | 0 dB | csiq | 0.7995 | +0.0561 [+0.0529, +0.0596] | 0.6027 | +0.1185 [+0.1082, +0.1300] |  |
| 6 | 0.3 | 0.5 | 6 | 0 dB | csiqp | 0.8049 | +0.0616 [+0.0583, +0.0650] | 0.6134 | +0.1292 [+0.1187, +0.1404] |  |
| 6 | 0.3 | 0.5 | 6 | 0 dB | switch | 0.7995 | +0.0561 [+0.0529, +0.0596] | 0.6027 | +0.1185 [+0.1082, +0.1300] | frac 1.000; rule_loss +0.0000 [+0.0000, +0.0000]; decision_loss 0.0102; oracle_cf - cos +0.0092; frac_cf 0.800 |
| 6 | 0.3 | 0.5 | 6 | 0 dB | ods | 0.7979 | +0.0545 [+0.0511, +0.0581] | 0.5866 | +0.1025 [+0.0918, +0.1137] |  |
| 6 | 0.3 | 0.5 | 6 | 0 dB | csiqb | 0.7989 | +0.0556 [+0.0524, +0.0589] | 0.6217 | +0.1376 [+0.1273, +0.1489] |  |
| 6 | 0.3 | 0.5 | 6 | 0 dB | csiqw | 0.8018 | +0.0585 [+0.0553, +0.0619] | 0.6115 | +0.1273 [+0.1172, +0.1389] |  |
| 6 | 0.3 | 0.5 | 6 | 10 dB | fixed | 0.9116 | - | 0.7887 | - |  |
| 6 | 0.3 | 0.5 | 6 | 10 dB | csiq | 0.9339 | +0.0223 [+0.0205, +0.0241] | 0.8428 | +0.0541 [+0.0466, +0.0612] |  |
| 6 | 0.3 | 0.5 | 6 | 10 dB | csiqp | 0.9356 | +0.0240 [+0.0221, +0.0257] | 0.8484 | +0.0597 [+0.0522, +0.0667] |  |
| 6 | 0.3 | 0.5 | 6 | 10 dB | switch | 0.9339 | +0.0223 [+0.0205, +0.0241] | 0.8428 | +0.0541 [+0.0466, +0.0612] | frac 0.999; rule_loss +0.0000 [-0.0000, +0.0000]; decision_loss 0.0055; oracle_cf - cos +0.0052; frac_cf 0.779 |
| 6 | 0.3 | 0.5 | 6 | 10 dB | ods | 0.9340 | +0.0224 [+0.0204, +0.0242] | 0.8417 | +0.0530 [+0.0448, +0.0602] |  |
| 6 | 0.3 | 0.5 | 6 | 10 dB | csiqb | 0.9277 | +0.0161 [+0.0144, +0.0178] | 0.8510 | +0.0623 [+0.0550, +0.0688] |  |
| 6 | 0.3 | 0.5 | 6 | 10 dB | csiqw | 0.9355 | +0.0239 [+0.0221, +0.0256] | 0.8486 | +0.0598 [+0.0523, +0.0669] |  |
| 6 | 0.3 | 0.5 | 6 | 20 dB | fixed | 0.9711 | - | 0.9365 | - |  |
| 6 | 0.3 | 0.5 | 6 | 20 dB | csiq | 0.9761 | +0.0049 [+0.0043, +0.0056] | 0.9517 | +0.0151 [+0.0120, +0.0183] |  |
| 6 | 0.3 | 0.5 | 6 | 20 dB | csiqp | 0.9765 | +0.0053 [+0.0048, +0.0060] | 0.9535 | +0.0170 [+0.0138, +0.0201] |  |
| 6 | 0.3 | 0.5 | 6 | 20 dB | switch | 0.9760 | +0.0049 [+0.0043, +0.0055] | 0.9520 | +0.0155 [+0.0125, +0.0185] | frac 0.970; rule_loss +0.0000 [-0.0002, +0.0002]; decision_loss 0.0020; oracle_cf - cos +0.0020; frac_cf 0.719 |
| 6 | 0.3 | 0.5 | 6 | 20 dB | ods | 0.9765 | +0.0054 [+0.0048, +0.0060] | 0.9538 | +0.0173 [+0.0141, +0.0204] |  |
| 6 | 0.3 | 0.5 | 6 | 20 dB | csiqb | 0.9718 | +0.0007 [+0.0001, +0.0013] | 0.9472 | +0.0107 [+0.0077, +0.0137] |  |
| 6 | 0.3 | 0.5 | 6 | 20 dB | csiqw | 0.9764 | +0.0053 [+0.0047, +0.0059] | 0.9535 | +0.0170 [+0.0138, +0.0201] |  |
| 6 | 0.3 | 0.3 | 1 | 0 dB | fixed | 0.9270 | - | 0.9270 | - |  |
| 6 | 0.3 | 0.3 | 1 | 0 dB | csiq | 0.8959 | -0.0311 [-0.0356, -0.0271] | 0.8959 | -0.0311 [-0.0356, -0.0271] |  |
| 6 | 0.3 | 0.3 | 1 | 0 dB | csiqp | 0.9005 | -0.0264 [-0.0307, -0.0226] | 0.9005 | -0.0264 [-0.0307, -0.0226] |  |
| 6 | 0.3 | 0.3 | 1 | 0 dB | switch | 0.9266 | -0.0004 [-0.0009, +0.0001] | 0.9266 | -0.0004 [-0.0009, +0.0001] | frac 0.009; rule_loss +0.0004 [-0.0001, +0.0009]; decision_loss 0.0098; oracle_cf - cos +0.0095; frac_cf 0.368 |
| 6 | 0.3 | 0.3 | 1 | 0 dB | ods | 0.8959 | -0.0311 [-0.0356, -0.0271] | 0.8959 | -0.0311 [-0.0356, -0.0271] |  |
| 6 | 0.3 | 0.3 | 1 | 0 dB | csiqb | 0.8937 | -0.0332 [-0.0368, -0.0298] | 0.8937 | -0.0332 [-0.0368, -0.0298] |  |
| 6 | 0.3 | 0.3 | 1 | 0 dB | csiqw | 0.8991 | -0.0279 [-0.0323, -0.0240] | 0.8991 | -0.0279 [-0.0323, -0.0240] |  |
| 6 | 0.3 | 0.3 | 1 | 10 dB | fixed | 0.9885 | - | 0.9885 | - |  |
| 6 | 0.3 | 0.3 | 1 | 10 dB | csiq | 0.9744 | -0.0141 [-0.0159, -0.0125] | 0.9744 | -0.0141 [-0.0159, -0.0125] |  |
| 6 | 0.3 | 0.3 | 1 | 10 dB | csiqp | 0.9752 | -0.0133 [-0.0150, -0.0118] | 0.9752 | -0.0133 [-0.0150, -0.0118] |  |
| 6 | 0.3 | 0.3 | 1 | 10 dB | switch | 0.9885 | +0.0000 [+0.0000, +0.0000] | 0.9885 | +0.0000 [+0.0000, +0.0000] | frac 0.000; rule_loss +0.0000 [+0.0000, +0.0000]; decision_loss 0.0007; oracle_cf - cos +0.0007; frac_cf 0.243 |
| 6 | 0.3 | 0.3 | 1 | 10 dB | ods | 0.9744 | -0.0141 [-0.0159, -0.0125] | 0.9744 | -0.0141 [-0.0159, -0.0125] |  |
| 6 | 0.3 | 0.3 | 1 | 10 dB | csiqb | 0.9683 | -0.0202 [-0.0216, -0.0190] | 0.9683 | -0.0202 [-0.0216, -0.0190] |  |
| 6 | 0.3 | 0.3 | 1 | 10 dB | csiqw | 0.9753 | -0.0132 [-0.0150, -0.0118] | 0.9753 | -0.0132 [-0.0150, -0.0118] |  |
| 6 | 0.3 | 0.3 | 1 | 20 dB | fixed | 0.9973 | - | 0.9973 | - |  |
| 6 | 0.3 | 0.3 | 1 | 20 dB | csiq | 0.9944 | -0.0030 [-0.0039, -0.0022] | 0.9944 | -0.0030 [-0.0039, -0.0022] |  |
| 6 | 0.3 | 0.3 | 1 | 20 dB | csiqp | 0.9944 | -0.0029 [-0.0038, -0.0022] | 0.9944 | -0.0029 [-0.0038, -0.0022] |  |
| 6 | 0.3 | 0.3 | 1 | 20 dB | switch | 0.9973 | +0.0000 [+0.0000, +0.0000] | 0.9973 | +0.0000 [+0.0000, +0.0000] | frac 0.000; rule_loss +0.0000 [+0.0000, +0.0000]; decision_loss 0.0001; oracle_cf - cos +0.0001; frac_cf 0.208 |
| 6 | 0.3 | 0.3 | 1 | 20 dB | ods | 0.9944 | -0.0030 [-0.0039, -0.0022] | 0.9944 | -0.0030 [-0.0039, -0.0022] |  |
| 6 | 0.3 | 0.3 | 1 | 20 dB | csiqb | 0.9914 | -0.0059 [-0.0067, -0.0053] | 0.9914 | -0.0059 [-0.0067, -0.0053] |  |
| 6 | 0.3 | 0.3 | 1 | 20 dB | csiqw | 0.9944 | -0.0029 [-0.0038, -0.0022] | 0.9944 | -0.0029 [-0.0038, -0.0022] |  |
| 6 | 0.3 | 0.3 | 2 | 0 dB | fixed | 0.8749 | - | 0.8310 | - |  |
| 6 | 0.3 | 0.3 | 2 | 0 dB | csiq | 0.8523 | -0.0226 [-0.0261, -0.0191] | 0.7953 | -0.0357 [-0.0418, -0.0296] |  |
| 6 | 0.3 | 0.3 | 2 | 0 dB | csiqp | 0.8599 | -0.0150 [-0.0184, -0.0117] | 0.8059 | -0.0251 [-0.0307, -0.0191] |  |
| 6 | 0.3 | 0.3 | 2 | 0 dB | switch | 0.8735 | -0.0014 [-0.0024, -0.0006] | 0.8293 | -0.0017 [-0.0034, -0.0000] | frac 0.064; rule_loss +0.0014 [+0.0006, +0.0024]; decision_loss 0.0190; oracle_cf - cos +0.0184; frac_cf 0.423 |
| 6 | 0.3 | 0.3 | 2 | 0 dB | ods | 0.8523 | -0.0226 [-0.0258, -0.0190] | 0.7901 | -0.0409 [-0.0466, -0.0345] |  |
| 6 | 0.3 | 0.3 | 2 | 0 dB | csiqb | 0.8545 | -0.0204 [-0.0235, -0.0173] | 0.8069 | -0.0241 [-0.0295, -0.0188] |  |
| 6 | 0.3 | 0.3 | 2 | 0 dB | csiqw | 0.8574 | -0.0175 [-0.0207, -0.0141] | 0.8039 | -0.0271 [-0.0329, -0.0212] |  |
| 6 | 0.3 | 0.3 | 2 | 10 dB | fixed | 0.9742 | - | 0.9631 | - |  |
| 6 | 0.3 | 0.3 | 2 | 10 dB | csiq | 0.9615 | -0.0126 [-0.0143, -0.0111] | 0.9438 | -0.0193 [-0.0224, -0.0166] |  |
| 6 | 0.3 | 0.3 | 2 | 10 dB | csiqp | 0.9632 | -0.0109 [-0.0125, -0.0095] | 0.9468 | -0.0163 [-0.0191, -0.0137] |  |
| 6 | 0.3 | 0.3 | 2 | 10 dB | switch | 0.9741 | -0.0001 [-0.0002, +0.0000] | 0.9630 | -0.0002 [-0.0004, +0.0000] | frac 0.008; rule_loss +0.0001 [-0.0000, +0.0002]; decision_loss 0.0036; oracle_cf - cos +0.0035; frac_cf 0.340 |
| 6 | 0.3 | 0.3 | 2 | 10 dB | ods | 0.9606 | -0.0136 [-0.0152, -0.0120] | 0.9402 | -0.0229 [-0.0262, -0.0201] |  |
| 6 | 0.3 | 0.3 | 2 | 10 dB | csiqb | 0.9550 | -0.0191 [-0.0204, -0.0180] | 0.9399 | -0.0232 [-0.0255, -0.0211] |  |
| 6 | 0.3 | 0.3 | 2 | 10 dB | csiqw | 0.9632 | -0.0110 [-0.0125, -0.0096] | 0.9467 | -0.0164 [-0.0192, -0.0138] |  |
| 6 | 0.3 | 0.3 | 2 | 20 dB | fixed | 0.9952 | - | 0.9932 | - |  |
| 6 | 0.3 | 0.3 | 2 | 20 dB | csiq | 0.9919 | -0.0033 [-0.0039, -0.0027] | 0.9877 | -0.0055 [-0.0068, -0.0044] |  |
| 6 | 0.3 | 0.3 | 2 | 20 dB | csiqp | 0.9921 | -0.0031 [-0.0037, -0.0026] | 0.9881 | -0.0051 [-0.0064, -0.0041] |  |
| 6 | 0.3 | 0.3 | 2 | 20 dB | switch | 0.9952 | -0.0000 [-0.0000, +0.0000] | 0.9932 | -0.0000 [-0.0000, +0.0000] | frac 0.003; rule_loss +0.0000 [-0.0000, +0.0000]; decision_loss 0.0005; oracle_cf - cos +0.0005; frac_cf 0.301 |
| 6 | 0.3 | 0.3 | 2 | 20 dB | ods | 0.9917 | -0.0035 [-0.0042, -0.0029] | 0.9870 | -0.0062 [-0.0075, -0.0051] |  |
| 6 | 0.3 | 0.3 | 2 | 20 dB | csiqb | 0.9880 | -0.0072 [-0.0078, -0.0068] | 0.9832 | -0.0101 [-0.0112, -0.0092] |  |
| 6 | 0.3 | 0.3 | 2 | 20 dB | csiqw | 0.9921 | -0.0031 [-0.0037, -0.0026] | 0.9881 | -0.0051 [-0.0064, -0.0041] |  |
| 6 | 0.3 | 0.3 | 4 | 0 dB | fixed | 0.7821 | - | 0.6101 | - |  |
| 6 | 0.3 | 0.3 | 4 | 0 dB | csiq | 0.7932 | +0.0111 [+0.0073, +0.0153] | 0.6347 | +0.0245 [+0.0145, +0.0349] |  |
| 6 | 0.3 | 0.3 | 4 | 0 dB | csiqp | 0.8030 | +0.0210 [+0.0173, +0.0251] | 0.6508 | +0.0407 [+0.0303, +0.0509] |  |
| 6 | 0.3 | 0.3 | 4 | 0 dB | switch | 0.7902 | +0.0082 [+0.0046, +0.0121] | 0.6265 | +0.0164 [+0.0072, +0.0258] | frac 0.812; rule_loss +0.0029 [+0.0014, +0.0045]; decision_loss 0.0317; oracle_cf - cos +0.0307; frac_cf 0.560 |
| 6 | 0.3 | 0.3 | 4 | 0 dB | ods | 0.7932 | +0.0111 [+0.0075, +0.0154] | 0.6246 | +0.0145 [+0.0048, +0.0248] |  |
| 6 | 0.3 | 0.3 | 4 | 0 dB | csiqb | 0.7985 | +0.0165 [+0.0129, +0.0206] | 0.6615 | +0.0514 [+0.0416, +0.0615] |  |
| 6 | 0.3 | 0.3 | 4 | 0 dB | csiqw | 0.7996 | +0.0176 [+0.0139, +0.0217] | 0.6484 | +0.0382 [+0.0282, +0.0484] |  |
| 6 | 0.3 | 0.3 | 4 | 10 dB | fixed | 0.9322 | - | 0.8567 | - |  |
| 6 | 0.3 | 0.3 | 4 | 10 dB | csiq | 0.9383 | +0.0060 [+0.0039, +0.0081] | 0.8700 | +0.0133 [+0.0066, +0.0197] |  |
| 6 | 0.3 | 0.3 | 4 | 10 dB | csiqp | 0.9416 | +0.0094 [+0.0074, +0.0113] | 0.8800 | +0.0233 [+0.0168, +0.0297] |  |
| 6 | 0.3 | 0.3 | 4 | 10 dB | switch | 0.9368 | +0.0046 [+0.0027, +0.0063] | 0.8665 | +0.0097 [+0.0038, +0.0152] | frac 0.735; rule_loss +0.0015 [+0.0004, +0.0026]; decision_loss 0.0145; oracle_cf - cos +0.0144; frac_cf 0.560 |
| 6 | 0.3 | 0.3 | 4 | 10 dB | ods | 0.9383 | +0.0061 [+0.0039, +0.0081] | 0.8693 | +0.0126 [+0.0058, +0.0189] |  |
| 6 | 0.3 | 0.3 | 4 | 10 dB | csiqb | 0.9334 | +0.0011 [-0.0008, +0.0030] | 0.8775 | +0.0207 [+0.0146, +0.0268] |  |
| 6 | 0.3 | 0.3 | 4 | 10 dB | csiqw | 0.9411 | +0.0089 [+0.0068, +0.0108] | 0.8770 | +0.0203 [+0.0138, +0.0267] |  |
| 6 | 0.3 | 0.3 | 4 | 20 dB | fixed | 0.9855 | - | 0.9663 | - |  |
| 6 | 0.3 | 0.3 | 4 | 20 dB | csiq | 0.9861 | +0.0007 [-0.0001, +0.0015] | 0.9669 | +0.0006 [-0.0023, +0.0036] |  |
| 6 | 0.3 | 0.3 | 4 | 20 dB | csiqp | 0.9869 | +0.0014 [+0.0006, +0.0022] | 0.9692 | +0.0030 [+0.0002, +0.0059] |  |
| 6 | 0.3 | 0.3 | 4 | 20 dB | switch | 0.9858 | +0.0004 [-0.0002, +0.0009] | 0.9663 | +0.0000 [-0.0022, +0.0022] | frac 0.632; rule_loss +0.0003 [-0.0002, +0.0008]; decision_loss 0.0044; oracle_cf - cos +0.0044; frac_cf 0.557 |
| 6 | 0.3 | 0.3 | 4 | 20 dB | ods | 0.9865 | +0.0010 [+0.0003, +0.0017] | 0.9679 | +0.0017 [-0.0008, +0.0043] |  |
| 6 | 0.3 | 0.3 | 4 | 20 dB | csiqb | 0.9814 | -0.0040 [-0.0047, -0.0033] | 0.9629 | -0.0033 [-0.0058, -0.0008] |  |
| 6 | 0.3 | 0.3 | 4 | 20 dB | csiqw | 0.9869 | +0.0014 [+0.0007, +0.0022] | 0.9694 | +0.0031 [+0.0003, +0.0060] |  |
| 6 | 0.3 | 0.3 | 6 | 0 dB | fixed | 0.7440 | - | 0.4839 | - |  |
| 6 | 0.3 | 0.3 | 6 | 0 dB | csiq | 0.7608 | +0.0168 [+0.0134, +0.0204] | 0.5229 | +0.0390 [+0.0281, +0.0512] |  |
| 6 | 0.3 | 0.3 | 6 | 0 dB | csiqp | 0.7709 | +0.0269 [+0.0236, +0.0304] | 0.5415 | +0.0576 [+0.0461, +0.0695] |  |
| 6 | 0.3 | 0.3 | 6 | 0 dB | switch | 0.7592 | +0.0152 [+0.0119, +0.0188] | 0.5200 | +0.0361 [+0.0257, +0.0467] | frac 0.870; rule_loss +0.0016 [+0.0004, +0.0028]; decision_loss 0.0258; oracle_cf - cos +0.0239; frac_cf 0.578 |
| 6 | 0.3 | 0.3 | 6 | 0 dB | ods | 0.7598 | +0.0158 [+0.0125, +0.0193] | 0.5098 | +0.0259 [+0.0134, +0.0374] |  |
| 6 | 0.3 | 0.3 | 6 | 0 dB | csiqb | 0.7660 | +0.0220 [+0.0189, +0.0255] | 0.5518 | +0.0679 [+0.0568, +0.0795] |  |
| 6 | 0.3 | 0.3 | 6 | 0 dB | csiqw | 0.7668 | +0.0228 [+0.0195, +0.0263] | 0.5367 | +0.0528 [+0.0414, +0.0643] |  |
| 6 | 0.3 | 0.3 | 6 | 10 dB | fixed | 0.9110 | - | 0.7839 | - |  |
| 6 | 0.3 | 0.3 | 6 | 10 dB | csiq | 0.9188 | +0.0078 [+0.0059, +0.0098] | 0.8046 | +0.0207 [+0.0124, +0.0291] |  |
| 6 | 0.3 | 0.3 | 6 | 10 dB | csiqp | 0.9220 | +0.0111 [+0.0092, +0.0131] | 0.8150 | +0.0311 [+0.0230, +0.0394] |  |
| 6 | 0.3 | 0.3 | 6 | 10 dB | switch | 0.9173 | +0.0063 [+0.0046, +0.0081] | 0.8002 | +0.0163 [+0.0089, +0.0241] | frac 0.805; rule_loss +0.0015 [+0.0007, +0.0022]; decision_loss 0.0133; oracle_cf - cos +0.0129; frac_cf 0.597 |
| 6 | 0.3 | 0.3 | 6 | 10 dB | ods | 0.9188 | +0.0078 [+0.0060, +0.0097] | 0.8029 | +0.0189 [+0.0113, +0.0273] |  |
| 6 | 0.3 | 0.3 | 6 | 10 dB | csiqb | 0.9150 | +0.0040 [+0.0022, +0.0059] | 0.8204 | +0.0365 [+0.0288, +0.0448] |  |
| 6 | 0.3 | 0.3 | 6 | 10 dB | csiqw | 0.9215 | +0.0105 [+0.0087, +0.0124] | 0.8130 | +0.0290 [+0.0209, +0.0374] |  |
| 6 | 0.3 | 0.3 | 6 | 20 dB | fixed | 0.9713 | - | 0.9379 | - |  |
| 6 | 0.3 | 0.3 | 6 | 20 dB | csiq | 0.9723 | +0.0009 [+0.0002, +0.0016] | 0.9392 | +0.0013 [-0.0024, +0.0050] |  |
| 6 | 0.3 | 0.3 | 6 | 20 dB | csiqp | 0.9731 | +0.0018 [+0.0011, +0.0025] | 0.9426 | +0.0048 [+0.0013, +0.0084] |  |
| 6 | 0.3 | 0.3 | 6 | 20 dB | switch | 0.9723 | +0.0009 [+0.0003, +0.0015] | 0.9403 | +0.0024 [-0.0006, +0.0055] | frac 0.685; rule_loss +0.0000 [-0.0004, +0.0004]; decision_loss 0.0041; oracle_cf - cos +0.0040; frac_cf 0.555 |
| 6 | 0.3 | 0.3 | 6 | 20 dB | ods | 0.9728 | +0.0015 [+0.0008, +0.0022] | 0.9414 | +0.0035 [+0.0000, +0.0074] |  |
| 6 | 0.3 | 0.3 | 6 | 20 dB | csiqb | 0.9684 | -0.0029 [-0.0035, -0.0022] | 0.9379 | +0.0000 [-0.0031, +0.0034] |  |
| 6 | 0.3 | 0.3 | 6 | 20 dB | csiqw | 0.9731 | +0.0017 [+0.0011, +0.0024] | 0.9426 | +0.0048 [+0.0012, +0.0084] |  |
| 6 | 0.3 | 0.0 | 1 | 0 dB | fixed | 0.9267 | - | 0.9267 | - |  |
| 6 | 0.3 | 0.0 | 1 | 0 dB | csiq | 0.8796 | -0.0470 [-0.0516, -0.0425] | 0.8796 | -0.0470 [-0.0516, -0.0425] |  |
| 6 | 0.3 | 0.0 | 1 | 0 dB | csiqp | 0.8867 | -0.0399 [-0.0443, -0.0357] | 0.8867 | -0.0399 [-0.0443, -0.0357] |  |
| 6 | 0.3 | 0.0 | 1 | 0 dB | switch | 0.9267 | +0.0000 [+0.0000, +0.0000] | 0.9267 | +0.0000 [+0.0000, +0.0000] | frac 0.000; rule_loss +0.0000 [+0.0000, +0.0000]; decision_loss 0.0072; oracle_cf - cos +0.0069; frac_cf 0.270 |
| 6 | 0.3 | 0.0 | 1 | 0 dB | ods | 0.8796 | -0.0470 [-0.0516, -0.0425] | 0.8796 | -0.0470 [-0.0516, -0.0425] |  |
| 6 | 0.3 | 0.0 | 1 | 0 dB | csiqb | 0.8812 | -0.0454 [-0.0494, -0.0417] | 0.8812 | -0.0454 [-0.0494, -0.0417] |  |
| 6 | 0.3 | 0.0 | 1 | 0 dB | csiqw | 0.8841 | -0.0425 [-0.0469, -0.0382] | 0.8841 | -0.0425 [-0.0469, -0.0382] |  |
| 6 | 0.3 | 0.0 | 1 | 10 dB | fixed | 0.9885 | - | 0.9885 | - |  |
| 6 | 0.3 | 0.0 | 1 | 10 dB | csiq | 0.9727 | -0.0158 [-0.0176, -0.0141] | 0.9727 | -0.0158 [-0.0176, -0.0141] |  |
| 6 | 0.3 | 0.0 | 1 | 10 dB | csiqp | 0.9738 | -0.0147 [-0.0164, -0.0132] | 0.9738 | -0.0147 [-0.0164, -0.0132] |  |
| 6 | 0.3 | 0.0 | 1 | 10 dB | switch | 0.9885 | +0.0000 [+0.0000, +0.0000] | 0.9885 | +0.0000 [+0.0000, +0.0000] | frac 0.000; rule_loss +0.0000 [+0.0000, +0.0000]; decision_loss 0.0005; oracle_cf - cos +0.0005; frac_cf 0.165 |
| 6 | 0.3 | 0.0 | 1 | 10 dB | ods | 0.9727 | -0.0158 [-0.0176, -0.0141] | 0.9727 | -0.0158 [-0.0176, -0.0141] |  |
| 6 | 0.3 | 0.0 | 1 | 10 dB | csiqb | 0.9664 | -0.0221 [-0.0235, -0.0208] | 0.9664 | -0.0221 [-0.0235, -0.0208] |  |
| 6 | 0.3 | 0.0 | 1 | 10 dB | csiqw | 0.9737 | -0.0148 [-0.0165, -0.0132] | 0.9737 | -0.0148 [-0.0165, -0.0132] |  |
| 6 | 0.3 | 0.0 | 1 | 20 dB | fixed | 0.9973 | - | 0.9973 | - |  |
| 6 | 0.3 | 0.0 | 1 | 20 dB | csiq | 0.9941 | -0.0032 [-0.0036, -0.0028] | 0.9941 | -0.0032 [-0.0036, -0.0028] |  |
| 6 | 0.3 | 0.0 | 1 | 20 dB | csiqp | 0.9943 | -0.0031 [-0.0035, -0.0027] | 0.9943 | -0.0031 [-0.0035, -0.0027] |  |
| 6 | 0.3 | 0.0 | 1 | 20 dB | switch | 0.9973 | +0.0000 [+0.0000, +0.0000] | 0.9973 | +0.0000 [+0.0000, +0.0000] | frac 0.000; rule_loss +0.0000 [+0.0000, +0.0000]; decision_loss 0.0000; oracle_cf - cos +0.0000; frac_cf 0.149 |
| 6 | 0.3 | 0.0 | 1 | 20 dB | ods | 0.9941 | -0.0032 [-0.0036, -0.0028] | 0.9941 | -0.0032 [-0.0036, -0.0028] |  |
| 6 | 0.3 | 0.0 | 1 | 20 dB | csiqb | 0.9910 | -0.0063 [-0.0067, -0.0060] | 0.9910 | -0.0063 [-0.0067, -0.0060] |  |
| 6 | 0.3 | 0.0 | 1 | 20 dB | csiqw | 0.9943 | -0.0031 [-0.0035, -0.0027] | 0.9943 | -0.0031 [-0.0035, -0.0027] |  |
| 6 | 0.3 | 0.0 | 2 | 0 dB | fixed | 0.8756 | - | 0.8319 | - |  |
| 6 | 0.3 | 0.0 | 2 | 0 dB | csiq | 0.8338 | -0.0418 [-0.0455, -0.0376] | 0.7708 | -0.0611 [-0.0676, -0.0538] |  |
| 6 | 0.3 | 0.0 | 2 | 0 dB | csiqp | 0.8445 | -0.0311 [-0.0348, -0.0271] | 0.7860 | -0.0460 [-0.0525, -0.0389] |  |
| 6 | 0.3 | 0.0 | 2 | 0 dB | switch | 0.8755 | -0.0001 [-0.0002, +0.0000] | 0.8318 | -0.0001 [-0.0003, +0.0001] | frac 0.002; rule_loss +0.0001 [-0.0000, +0.0002]; decision_loss 0.0132; oracle_cf - cos +0.0125; frac_cf 0.304 |
| 6 | 0.3 | 0.0 | 2 | 0 dB | ods | 0.8340 | -0.0416 [-0.0451, -0.0374] | 0.7663 | -0.0657 [-0.0722, -0.0587] |  |
| 6 | 0.3 | 0.0 | 2 | 0 dB | csiqb | 0.8397 | -0.0359 [-0.0392, -0.0322] | 0.7875 | -0.0444 [-0.0502, -0.0376] |  |
| 6 | 0.3 | 0.0 | 2 | 0 dB | csiqw | 0.8404 | -0.0352 [-0.0390, -0.0312] | 0.7808 | -0.0512 [-0.0576, -0.0441] |  |
| 6 | 0.3 | 0.0 | 2 | 10 dB | fixed | 0.9739 | - | 0.9625 | - |  |
| 6 | 0.3 | 0.0 | 2 | 10 dB | csiq | 0.9559 | -0.0180 [-0.0199, -0.0162] | 0.9358 | -0.0267 [-0.0299, -0.0234] |  |
| 6 | 0.3 | 0.0 | 2 | 10 dB | csiqp | 0.9583 | -0.0156 [-0.0174, -0.0139] | 0.9398 | -0.0227 [-0.0258, -0.0196] |  |
| 6 | 0.3 | 0.0 | 2 | 10 dB | switch | 0.9739 | -0.0000 [-0.0000, +0.0000] | 0.9625 | -0.0000 [-0.0000, +0.0000] | frac 0.001; rule_loss +0.0000 [+0.0000, +0.0000]; decision_loss 0.0030; oracle_cf - cos +0.0029; frac_cf 0.258 |
| 6 | 0.3 | 0.0 | 2 | 10 dB | ods | 0.9552 | -0.0188 [-0.0206, -0.0169] | 0.9323 | -0.0303 [-0.0336, -0.0271] |  |
| 6 | 0.3 | 0.0 | 2 | 10 dB | csiqb | 0.9500 | -0.0239 [-0.0254, -0.0225] | 0.9329 | -0.0296 [-0.0322, -0.0270] |  |
| 6 | 0.3 | 0.0 | 2 | 10 dB | csiqw | 0.9579 | -0.0160 [-0.0178, -0.0143] | 0.9392 | -0.0234 [-0.0264, -0.0202] |  |
| 6 | 0.3 | 0.0 | 2 | 20 dB | fixed | 0.9953 | - | 0.9934 | - |  |
| 6 | 0.3 | 0.0 | 2 | 20 dB | csiq | 0.9904 | -0.0049 [-0.0056, -0.0043] | 0.9850 | -0.0084 [-0.0096, -0.0071] |  |
| 6 | 0.3 | 0.0 | 2 | 20 dB | csiqp | 0.9907 | -0.0046 [-0.0051, -0.0040] | 0.9858 | -0.0076 [-0.0087, -0.0064] |  |
| 6 | 0.3 | 0.0 | 2 | 20 dB | switch | 0.9953 | -0.0000 [-0.0000, +0.0000] | 0.9934 | -0.0000 [-0.0000, +0.0000] | frac 0.002; rule_loss +0.0000 [-0.0000, +0.0000]; decision_loss 0.0003; oracle_cf - cos +0.0003; frac_cf 0.206 |
| 6 | 0.3 | 0.0 | 2 | 20 dB | ods | 0.9901 | -0.0052 [-0.0059, -0.0045] | 0.9843 | -0.0091 [-0.0104, -0.0078] |  |
| 6 | 0.3 | 0.0 | 2 | 20 dB | csiqb | 0.9867 | -0.0086 [-0.0090, -0.0081] | 0.9813 | -0.0121 [-0.0129, -0.0112] |  |
| 6 | 0.3 | 0.0 | 2 | 20 dB | csiqw | 0.9907 | -0.0046 [-0.0051, -0.0040] | 0.9857 | -0.0076 [-0.0088, -0.0065] |  |
| 6 | 0.3 | 0.0 | 4 | 0 dB | fixed | 0.7816 | - | 0.6088 | - |  |
| 6 | 0.3 | 0.0 | 4 | 0 dB | csiq | 0.7717 | -0.0099 [-0.0138, -0.0056] | 0.6010 | -0.0079 [-0.0178, +0.0028] |  |
| 6 | 0.3 | 0.0 | 4 | 0 dB | csiqp | 0.7852 | +0.0035 [-0.0003, +0.0077] | 0.6229 | +0.0140 [+0.0040, +0.0248] |  |
| 6 | 0.3 | 0.0 | 4 | 0 dB | switch | 0.7798 | -0.0018 [-0.0038, +0.0002] | 0.6077 | -0.0012 [-0.0066, +0.0043] | frac 0.274; rule_loss +0.0018 [-0.0002, +0.0038]; decision_loss 0.0325; oracle_cf - cos +0.0312; frac_cf 0.441 |
| 6 | 0.3 | 0.0 | 4 | 0 dB | ods | 0.7730 | -0.0086 [-0.0124, -0.0046] | 0.5954 | -0.0134 [-0.0233, -0.0027] |  |
| 6 | 0.3 | 0.0 | 4 | 0 dB | csiqb | 0.7802 | -0.0014 [-0.0051, +0.0027] | 0.6312 | +0.0224 [+0.0123, +0.0327] |  |
| 6 | 0.3 | 0.0 | 4 | 0 dB | csiqw | 0.7801 | -0.0016 [-0.0053, +0.0026] | 0.6164 | +0.0076 [-0.0023, +0.0182] |  |
| 6 | 0.3 | 0.0 | 4 | 10 dB | fixed | 0.9334 | - | 0.8582 | - |  |
| 6 | 0.3 | 0.0 | 4 | 10 dB | csiq | 0.9287 | -0.0047 [-0.0068, -0.0026] | 0.8478 | -0.0104 [-0.0175, -0.0035] |  |
| 6 | 0.3 | 0.0 | 4 | 10 dB | csiqp | 0.9337 | +0.0003 [-0.0018, +0.0023] | 0.8625 | +0.0043 [-0.0026, +0.0108] |  |
| 6 | 0.3 | 0.0 | 4 | 10 dB | switch | 0.9318 | -0.0015 [-0.0027, -0.0003] | 0.8551 | -0.0031 [-0.0072, +0.0010] | frac 0.311; rule_loss +0.0015 [+0.0003, +0.0027]; decision_loss 0.0156; oracle_cf - cos +0.0154; frac_cf 0.457 |
| 6 | 0.3 | 0.0 | 4 | 10 dB | ods | 0.9285 | -0.0049 [-0.0070, -0.0026] | 0.8457 | -0.0125 [-0.0195, -0.0055] |  |
| 6 | 0.3 | 0.0 | 4 | 10 dB | csiqb | 0.9260 | -0.0074 [-0.0093, -0.0054] | 0.8616 | +0.0034 [-0.0030, +0.0097] |  |
| 6 | 0.3 | 0.0 | 4 | 10 dB | csiqw | 0.9325 | -0.0009 [-0.0030, +0.0012] | 0.8568 | -0.0014 [-0.0084, +0.0054] |  |
| 6 | 0.3 | 0.0 | 4 | 20 dB | fixed | 0.9855 | - | 0.9660 | - |  |
| 6 | 0.3 | 0.0 | 4 | 20 dB | csiq | 0.9842 | -0.0013 [-0.0021, -0.0004] | 0.9623 | -0.0037 [-0.0067, -0.0008] |  |
| 6 | 0.3 | 0.0 | 4 | 20 dB | csiqp | 0.9854 | -0.0001 [-0.0009, +0.0007] | 0.9658 | -0.0002 [-0.0030, +0.0026] |  |
| 6 | 0.3 | 0.0 | 4 | 20 dB | switch | 0.9850 | -0.0005 [-0.0010, +0.0000] | 0.9644 | -0.0016 [-0.0036, +0.0002] | frac 0.347; rule_loss +0.0005 [-0.0000, +0.0010]; decision_loss 0.0046; oracle_cf - cos +0.0046; frac_cf 0.466 |
| 6 | 0.3 | 0.0 | 4 | 20 dB | ods | 0.9847 | -0.0008 [-0.0016, +0.0000] | 0.9635 | -0.0025 [-0.0054, +0.0005] |  |
| 6 | 0.3 | 0.0 | 4 | 20 dB | csiqb | 0.9800 | -0.0055 [-0.0062, -0.0048] | 0.9599 | -0.0061 [-0.0087, -0.0035] |  |
| 6 | 0.3 | 0.0 | 4 | 20 dB | csiqw | 0.9853 | -0.0001 [-0.0009, +0.0007] | 0.9657 | -0.0003 [-0.0031, +0.0026] |  |
| 6 | 0.3 | 0.0 | 6 | 0 dB | fixed | 0.7453 | - | 0.4871 | - |  |
| 6 | 0.3 | 0.0 | 6 | 0 dB | csiq | 0.7384 | -0.0069 [-0.0103, -0.0034] | 0.4908 | +0.0037 [-0.0078, +0.0149] |  |
| 6 | 0.3 | 0.0 | 6 | 0 dB | csiqp | 0.7520 | +0.0067 [+0.0034, +0.0101] | 0.5137 | +0.0266 [+0.0153, +0.0380] |  |
| 6 | 0.3 | 0.0 | 6 | 0 dB | switch | 0.7449 | -0.0004 [-0.0021, +0.0013] | 0.4911 | +0.0039 [-0.0012, +0.0092] | frac 0.233; rule_loss +0.0004 [-0.0013, +0.0021]; decision_loss 0.0300; oracle_cf - cos +0.0278; frac_cf 0.459 |
| 6 | 0.3 | 0.0 | 6 | 0 dB | ods | 0.7386 | -0.0067 [-0.0102, -0.0033] | 0.4820 | -0.0051 [-0.0164, +0.0070] |  |
| 6 | 0.3 | 0.0 | 6 | 0 dB | csiqb | 0.7467 | +0.0015 [-0.0019, +0.0047] | 0.5221 | +0.0350 [+0.0238, +0.0464] |  |
| 6 | 0.3 | 0.0 | 6 | 0 dB | csiqw | 0.7461 | +0.0008 [-0.0027, +0.0043] | 0.5046 | +0.0175 [+0.0057, +0.0291] |  |
| 6 | 0.3 | 0.0 | 6 | 10 dB | fixed | 0.9116 | - | 0.7887 | - |  |
| 6 | 0.3 | 0.0 | 6 | 10 dB | csiq | 0.9080 | -0.0036 [-0.0056, -0.0016] | 0.7783 | -0.0105 [-0.0190, -0.0018] |  |
| 6 | 0.3 | 0.0 | 6 | 10 dB | csiqp | 0.9127 | +0.0011 [-0.0009, +0.0030] | 0.7924 | +0.0037 [-0.0046, +0.0117] |  |
| 6 | 0.3 | 0.0 | 6 | 10 dB | switch | 0.9107 | -0.0009 [-0.0020, +0.0001] | 0.7860 | -0.0027 [-0.0070, +0.0018] | frac 0.283; rule_loss +0.0009 [-0.0001, +0.0020]; decision_loss 0.0156; oracle_cf - cos +0.0152; frac_cf 0.468 |
| 6 | 0.3 | 0.0 | 6 | 10 dB | ods | 0.9083 | -0.0034 [-0.0054, -0.0013] | 0.7777 | -0.0110 [-0.0192, -0.0026] |  |
| 6 | 0.3 | 0.0 | 6 | 10 dB | csiqb | 0.9064 | -0.0052 [-0.0071, -0.0035] | 0.8000 | +0.0113 [+0.0038, +0.0188] |  |
| 6 | 0.3 | 0.0 | 6 | 10 dB | csiqw | 0.9116 | +0.0000 [-0.0020, +0.0019] | 0.7879 | -0.0008 [-0.0093, +0.0075] |  |
| 6 | 0.3 | 0.0 | 6 | 20 dB | fixed | 0.9715 | - | 0.9381 | - |  |
| 6 | 0.3 | 0.0 | 6 | 20 dB | csiq | 0.9697 | -0.0018 [-0.0026, -0.0010] | 0.9298 | -0.0083 [-0.0122, -0.0045] |  |
| 6 | 0.3 | 0.0 | 6 | 20 dB | csiqp | 0.9709 | -0.0006 [-0.0013, +0.0002] | 0.9351 | -0.0030 [-0.0068, +0.0006] |  |
| 6 | 0.3 | 0.0 | 6 | 20 dB | switch | 0.9713 | -0.0002 [-0.0005, +0.0002] | 0.9377 | -0.0004 [-0.0022, +0.0014] | frac 0.322; rule_loss +0.0002 [-0.0002, +0.0005]; decision_loss 0.0042; oracle_cf - cos +0.0042; frac_cf 0.451 |
| 6 | 0.3 | 0.0 | 6 | 20 dB | ods | 0.9701 | -0.0014 [-0.0022, -0.0007] | 0.9312 | -0.0069 [-0.0108, -0.0031] |  |
| 6 | 0.3 | 0.0 | 6 | 20 dB | csiqb | 0.9662 | -0.0052 [-0.0059, -0.0046] | 0.9309 | -0.0071 [-0.0105, -0.0040] |  |
| 6 | 0.3 | 0.0 | 6 | 20 dB | csiqw | 0.9708 | -0.0007 [-0.0014, +0.0001] | 0.9345 | -0.0036 [-0.0074, +0.0001] |  |
| 6 | 0.5 | 1.0 | 1 | 0 dB | fixed | 0.9278 | - | 0.9278 | - |  |
| 6 | 0.5 | 1.0 | 1 | 0 dB | csiq | 0.9715 | +0.0437 [+0.0422, +0.0452] | 0.9715 | +0.0437 [+0.0422, +0.0452] |  |
| 6 | 0.5 | 1.0 | 1 | 0 dB | csiqp | 0.9715 | +0.0437 [+0.0422, +0.0452] | 0.9715 | +0.0437 [+0.0422, +0.0452] |  |
| 6 | 0.5 | 1.0 | 1 | 0 dB | switch | 0.9715 | +0.0437 [+0.0422, +0.0452] | 0.9715 | +0.0437 [+0.0422, +0.0452] | frac 1.000; rule_loss +0.0000 [+0.0000, +0.0000]; decision_loss 0.0000; oracle_cf - cos +0.0000; frac_cf 1.000 |
| 6 | 0.5 | 1.0 | 1 | 0 dB | ods | 0.9715 | +0.0437 [+0.0422, +0.0452] | 0.9715 | +0.0437 [+0.0422, +0.0452] |  |
| 6 | 0.5 | 1.0 | 1 | 0 dB | csiqb | 0.9579 | +0.0300 [+0.0286, +0.0314] | 0.9579 | +0.0300 [+0.0286, +0.0314] |  |
| 6 | 0.5 | 1.0 | 1 | 0 dB | csiqw | 0.9705 | +0.0426 [+0.0412, +0.0441] | 0.9705 | +0.0426 [+0.0412, +0.0441] |  |
| 6 | 0.5 | 1.0 | 1 | 10 dB | fixed | 0.9885 | - | 0.9885 | - |  |
| 6 | 0.5 | 1.0 | 1 | 10 dB | csiq | 0.9952 | +0.0067 [+0.0064, +0.0070] | 0.9952 | +0.0067 [+0.0064, +0.0070] |  |
| 6 | 0.5 | 1.0 | 1 | 10 dB | csiqp | 0.9952 | +0.0067 [+0.0064, +0.0070] | 0.9952 | +0.0067 [+0.0064, +0.0070] |  |
| 6 | 0.5 | 1.0 | 1 | 10 dB | switch | 0.9952 | +0.0067 [+0.0064, +0.0070] | 0.9952 | +0.0067 [+0.0064, +0.0070] | frac 1.000; rule_loss +0.0000 [+0.0000, +0.0000]; decision_loss 0.0000; oracle_cf - cos +0.0000; frac_cf 1.000 |
| 6 | 0.5 | 1.0 | 1 | 10 dB | ods | 0.9952 | +0.0067 [+0.0064, +0.0070] | 0.9952 | +0.0067 [+0.0064, +0.0070] |  |
| 6 | 0.5 | 1.0 | 1 | 10 dB | csiqb | 0.9901 | +0.0015 [+0.0013, +0.0017] | 0.9901 | +0.0015 [+0.0013, +0.0017] |  |
| 6 | 0.5 | 1.0 | 1 | 10 dB | csiqw | 0.9951 | +0.0066 [+0.0063, +0.0069] | 0.9951 | +0.0066 [+0.0063, +0.0069] |  |
| 6 | 0.5 | 1.0 | 1 | 20 dB | fixed | 0.9972 | - | 0.9972 | - |  |
| 6 | 0.5 | 1.0 | 1 | 20 dB | csiq | 0.9980 | +0.0007 [+0.0007, +0.0007] | 0.9980 | +0.0007 [+0.0007, +0.0007] |  |
| 6 | 0.5 | 1.0 | 1 | 20 dB | csiqp | 0.9980 | +0.0007 [+0.0007, +0.0007] | 0.9980 | +0.0007 [+0.0007, +0.0007] |  |
| 6 | 0.5 | 1.0 | 1 | 20 dB | switch | 0.9980 | +0.0007 [+0.0007, +0.0007] | 0.9980 | +0.0007 [+0.0007, +0.0007] | frac 1.000; rule_loss +0.0000 [+0.0000, +0.0000]; decision_loss 0.0000; oracle_cf - cos +0.0000; frac_cf 1.000 |
| 6 | 0.5 | 1.0 | 1 | 20 dB | ods | 0.9980 | +0.0007 [+0.0007, +0.0007] | 0.9980 | +0.0007 [+0.0007, +0.0007] |  |
| 6 | 0.5 | 1.0 | 1 | 20 dB | csiqb | 0.9964 | -0.0008 [-0.0009, -0.0007] | 0.9964 | -0.0008 [-0.0009, -0.0007] |  |
| 6 | 0.5 | 1.0 | 1 | 20 dB | csiqw | 0.9979 | +0.0007 [+0.0007, +0.0007] | 0.9979 | +0.0007 [+0.0007, +0.0007] |  |
| 6 | 0.5 | 1.0 | 2 | 0 dB | fixed | 0.8778 | - | 0.8349 | - |  |
| 6 | 0.5 | 1.0 | 2 | 0 dB | csiq | 0.9551 | +0.0772 [+0.0750, +0.0794] | 0.9442 | +0.1093 [+0.1057, +0.1129] |  |
| 6 | 0.5 | 1.0 | 2 | 0 dB | csiqp | 0.9551 | +0.0772 [+0.0750, +0.0794] | 0.9442 | +0.1093 [+0.1057, +0.1129] |  |
| 6 | 0.5 | 1.0 | 2 | 0 dB | switch | 0.9551 | +0.0772 [+0.0750, +0.0794] | 0.9442 | +0.1093 [+0.1057, +0.1129] | frac 1.000; rule_loss +0.0000 [+0.0000, +0.0000]; decision_loss 0.0000; oracle_cf - cos +0.0000; frac_cf 1.000 |
| 6 | 0.5 | 1.0 | 2 | 0 dB | ods | 0.9533 | +0.0755 [+0.0733, +0.0776] | 0.9366 | +0.1016 [+0.0980, +0.1052] |  |
| 6 | 0.5 | 1.0 | 2 | 0 dB | csiqb | 0.9399 | +0.0621 [+0.0599, +0.0642] | 0.9281 | +0.0932 [+0.0896, +0.0966] |  |
| 6 | 0.5 | 1.0 | 2 | 0 dB | csiqw | 0.9510 | +0.0732 [+0.0710, +0.0754] | 0.9416 | +0.1066 [+0.1030, +0.1102] |  |
| 6 | 0.5 | 1.0 | 2 | 10 dB | fixed | 0.9741 | - | 0.9631 | - |  |
| 6 | 0.5 | 1.0 | 2 | 10 dB | csiq | 0.9923 | +0.0182 [+0.0176, +0.0189] | 0.9904 | +0.0273 [+0.0262, +0.0284] |  |
| 6 | 0.5 | 1.0 | 2 | 10 dB | csiqp | 0.9923 | +0.0182 [+0.0176, +0.0189] | 0.9904 | +0.0273 [+0.0262, +0.0284] |  |
| 6 | 0.5 | 1.0 | 2 | 10 dB | switch | 0.9923 | +0.0182 [+0.0176, +0.0189] | 0.9904 | +0.0273 [+0.0262, +0.0284] | frac 1.000; rule_loss +0.0000 [+0.0000, +0.0000]; decision_loss 0.0000; oracle_cf - cos +0.0000; frac_cf 1.000 |
| 6 | 0.5 | 1.0 | 2 | 10 dB | ods | 0.9918 | +0.0178 [+0.0171, +0.0184] | 0.9887 | +0.0256 [+0.0245, +0.0267] |  |
| 6 | 0.5 | 1.0 | 2 | 10 dB | csiqb | 0.9844 | +0.0103 [+0.0097, +0.0109] | 0.9808 | +0.0177 [+0.0167, +0.0188] |  |
| 6 | 0.5 | 1.0 | 2 | 10 dB | csiqw | 0.9921 | +0.0180 [+0.0173, +0.0186] | 0.9901 | +0.0270 [+0.0259, +0.0282] |  |
| 6 | 0.5 | 1.0 | 2 | 20 dB | fixed | 0.9954 | - | 0.9934 | - |  |
| 6 | 0.5 | 1.0 | 2 | 20 dB | csiq | 0.9977 | +0.0023 [+0.0022, +0.0024] | 0.9968 | +0.0033 [+0.0032, +0.0035] |  |
| 6 | 0.5 | 1.0 | 2 | 20 dB | csiqp | 0.9977 | +0.0023 [+0.0022, +0.0024] | 0.9968 | +0.0033 [+0.0032, +0.0035] |  |
| 6 | 0.5 | 1.0 | 2 | 20 dB | switch | 0.9977 | +0.0023 [+0.0022, +0.0024] | 0.9968 | +0.0033 [+0.0032, +0.0035] | frac 1.000; rule_loss +0.0000 [+0.0000, +0.0000]; decision_loss 0.0000; oracle_cf - cos +0.0000; frac_cf 1.000 |
| 6 | 0.5 | 1.0 | 2 | 20 dB | ods | 0.9976 | +0.0023 [+0.0022, +0.0024] | 0.9967 | +0.0032 [+0.0031, +0.0034] |  |
| 6 | 0.5 | 1.0 | 2 | 20 dB | csiqb | 0.9956 | +0.0003 [+0.0002, +0.0004] | 0.9940 | +0.0006 [+0.0004, +0.0008] |  |
| 6 | 0.5 | 1.0 | 2 | 20 dB | csiqw | 0.9977 | +0.0023 [+0.0022, +0.0024] | 0.9967 | +0.0033 [+0.0031, +0.0035] |  |
| 6 | 0.5 | 1.0 | 4 | 0 dB | fixed | 0.7856 | - | 0.6168 | - |  |
| 6 | 0.5 | 1.0 | 4 | 0 dB | csiq | 0.9249 | +0.1392 [+0.1361, +0.1424] | 0.8897 | +0.2729 [+0.2651, +0.2815] |  |
| 6 | 0.5 | 1.0 | 4 | 0 dB | csiqp | 0.9249 | +0.1392 [+0.1361, +0.1424] | 0.8897 | +0.2729 [+0.2651, +0.2815] |  |
| 6 | 0.5 | 1.0 | 4 | 0 dB | switch | 0.9249 | +0.1392 [+0.1361, +0.1424] | 0.8897 | +0.2729 [+0.2651, +0.2815] | frac 1.000; rule_loss +0.0000 [+0.0000, +0.0000]; decision_loss 0.0000; oracle_cf - cos +0.0000; frac_cf 1.000 |
| 6 | 0.5 | 1.0 | 4 | 0 dB | ods | 0.9226 | +0.1369 [+0.1337, +0.1400] | 0.8699 | +0.2532 [+0.2452, +0.2616] |  |
| 6 | 0.5 | 1.0 | 4 | 0 dB | csiqb | 0.9100 | +0.1243 [+0.1212, +0.1275] | 0.8742 | +0.2574 [+0.2495, +0.2660] |  |
| 6 | 0.5 | 1.0 | 4 | 0 dB | csiqw | 0.9185 | +0.1328 [+0.1296, +0.1360] | 0.8867 | +0.2699 [+0.2619, +0.2784] |  |
| 6 | 0.5 | 1.0 | 4 | 10 dB | fixed | 0.9322 | - | 0.8544 | - |  |
| 6 | 0.5 | 1.0 | 4 | 10 dB | csiq | 0.9848 | +0.0526 [+0.0509, +0.0543] | 0.9743 | +0.1198 [+0.1145, +0.1252] |  |
| 6 | 0.5 | 1.0 | 4 | 10 dB | csiqp | 0.9848 | +0.0526 [+0.0509, +0.0543] | 0.9743 | +0.1198 [+0.1145, +0.1252] |  |
| 6 | 0.5 | 1.0 | 4 | 10 dB | switch | 0.9848 | +0.0526 [+0.0509, +0.0543] | 0.9743 | +0.1198 [+0.1145, +0.1252] | frac 1.000; rule_loss +0.0000 [+0.0000, +0.0000]; decision_loss 0.0000; oracle_cf - cos +0.0000; frac_cf 1.000 |
| 6 | 0.5 | 1.0 | 4 | 10 dB | ods | 0.9849 | +0.0526 [+0.0510, +0.0543] | 0.9720 | +0.1175 [+0.1123, +0.1229] |  |
| 6 | 0.5 | 1.0 | 4 | 10 dB | csiqb | 0.9736 | +0.0413 [+0.0397, +0.0430] | 0.9584 | +0.1040 [+0.0988, +0.1094] |  |
| 6 | 0.5 | 1.0 | 4 | 10 dB | csiqw | 0.9842 | +0.0519 [+0.0503, +0.0536] | 0.9735 | +0.1191 [+0.1138, +0.1245] |  |
| 6 | 0.5 | 1.0 | 4 | 20 dB | fixed | 0.9855 | - | 0.9658 | - |  |
| 6 | 0.5 | 1.0 | 4 | 20 dB | csiq | 0.9968 | +0.0113 [+0.0108, +0.0119] | 0.9942 | +0.0284 [+0.0264, +0.0303] |  |
| 6 | 0.5 | 1.0 | 4 | 20 dB | csiqp | 0.9968 | +0.0113 [+0.0108, +0.0119] | 0.9942 | +0.0284 [+0.0264, +0.0303] |  |
| 6 | 0.5 | 1.0 | 4 | 20 dB | switch | 0.9968 | +0.0113 [+0.0108, +0.0119] | 0.9942 | +0.0284 [+0.0264, +0.0303] | frac 1.000; rule_loss +0.0000 [+0.0000, +0.0000]; decision_loss 0.0000; oracle_cf - cos +0.0000; frac_cf 1.000 |
| 6 | 0.5 | 1.0 | 4 | 20 dB | ods | 0.9969 | +0.0114 [+0.0109, +0.0119] | 0.9944 | +0.0285 [+0.0266, +0.0305] |  |
| 6 | 0.5 | 1.0 | 4 | 20 dB | csiqb | 0.9928 | +0.0073 [+0.0068, +0.0078] | 0.9875 | +0.0216 [+0.0197, +0.0236] |  |
| 6 | 0.5 | 1.0 | 4 | 20 dB | csiqw | 0.9965 | +0.0111 [+0.0105, +0.0116] | 0.9938 | +0.0280 [+0.0260, +0.0299] |  |
| 6 | 0.5 | 1.0 | 6 | 0 dB | fixed | 0.7457 | - | 0.4889 | - |  |
| 6 | 0.5 | 1.0 | 6 | 0 dB | csiq | 0.9020 | +0.1563 [+0.1537, +0.1589] | 0.8454 | +0.3565 [+0.3487, +0.3648] |  |
| 6 | 0.5 | 1.0 | 6 | 0 dB | csiqp | 0.9020 | +0.1563 [+0.1537, +0.1589] | 0.8454 | +0.3565 [+0.3487, +0.3648] |  |
| 6 | 0.5 | 1.0 | 6 | 0 dB | switch | 0.9020 | +0.1563 [+0.1537, +0.1589] | 0.8454 | +0.3565 [+0.3487, +0.3648] | frac 1.000; rule_loss +0.0000 [+0.0000, +0.0000]; decision_loss 0.0000; oracle_cf - cos +0.0000; frac_cf 1.000 |
| 6 | 0.5 | 1.0 | 6 | 0 dB | ods | 0.8997 | +0.1540 [+0.1515, +0.1567] | 0.8182 | +0.3293 [+0.3214, +0.3380] |  |
| 6 | 0.5 | 1.0 | 6 | 0 dB | csiqb | 0.8880 | +0.1423 [+0.1397, +0.1450] | 0.8309 | +0.3420 [+0.3340, +0.3503] |  |
| 6 | 0.5 | 1.0 | 6 | 0 dB | csiqw | 0.8947 | +0.1490 [+0.1464, +0.1517] | 0.8422 | +0.3533 [+0.3454, +0.3615] |  |
| 6 | 0.5 | 1.0 | 6 | 10 dB | fixed | 0.9116 | - | 0.7894 | - |  |
| 6 | 0.5 | 1.0 | 6 | 10 dB | csiq | 0.9710 | +0.0595 [+0.0580, +0.0611] | 0.9514 | +0.1619 [+0.1558, +0.1682] |  |
| 6 | 0.5 | 1.0 | 6 | 10 dB | csiqp | 0.9710 | +0.0595 [+0.0580, +0.0611] | 0.9514 | +0.1619 [+0.1558, +0.1682] |  |
| 6 | 0.5 | 1.0 | 6 | 10 dB | switch | 0.9710 | +0.0595 [+0.0580, +0.0611] | 0.9514 | +0.1619 [+0.1558, +0.1682] | frac 1.000; rule_loss +0.0000 [+0.0000, +0.0000]; decision_loss 0.0000; oracle_cf - cos +0.0000; frac_cf 1.000 |
| 6 | 0.5 | 1.0 | 6 | 10 dB | ods | 0.9714 | +0.0598 [+0.0584, +0.0615] | 0.9517 | +0.1623 [+0.1561, +0.1684] |  |
| 6 | 0.5 | 1.0 | 6 | 10 dB | csiqb | 0.9611 | +0.0496 [+0.0481, +0.0512] | 0.9360 | +0.1465 [+0.1405, +0.1528] |  |
| 6 | 0.5 | 1.0 | 6 | 10 dB | csiqw | 0.9704 | +0.0589 [+0.0574, +0.0605] | 0.9505 | +0.1611 [+0.1550, +0.1674] |  |
| 6 | 0.5 | 1.0 | 6 | 20 dB | fixed | 0.9709 | - | 0.9351 | - |  |
| 6 | 0.5 | 1.0 | 6 | 20 dB | csiq | 0.9839 | +0.0131 [+0.0125, +0.0136] | 0.9763 | +0.0412 [+0.0384, +0.0439] |  |
| 6 | 0.5 | 1.0 | 6 | 20 dB | csiqp | 0.9839 | +0.0131 [+0.0125, +0.0136] | 0.9763 | +0.0412 [+0.0384, +0.0439] |  |
| 6 | 0.5 | 1.0 | 6 | 20 dB | switch | 0.9839 | +0.0131 [+0.0125, +0.0136] | 0.9763 | +0.0412 [+0.0384, +0.0439] | frac 1.000; rule_loss +0.0000 [+0.0000, +0.0000]; decision_loss 0.0000; oracle_cf - cos +0.0000; frac_cf 1.000 |
| 6 | 0.5 | 1.0 | 6 | 20 dB | ods | 0.9841 | +0.0132 [+0.0127, +0.0138] | 0.9769 | +0.0418 [+0.0389, +0.0446] |  |
| 6 | 0.5 | 1.0 | 6 | 20 dB | csiqb | 0.9807 | +0.0098 [+0.0092, +0.0103] | 0.9700 | +0.0349 [+0.0320, +0.0376] |  |
| 6 | 0.5 | 1.0 | 6 | 20 dB | csiqw | 0.9837 | +0.0128 [+0.0123, +0.0134] | 0.9758 | +0.0407 [+0.0379, +0.0434] |  |
| 6 | 0.7 | 1.0 | 1 | 0 dB | fixed | 0.9272 | - | 0.9272 | - |  |
| 6 | 0.7 | 1.0 | 1 | 0 dB | csiq | 0.9701 | +0.0428 [+0.0413, +0.0443] | 0.9701 | +0.0428 [+0.0413, +0.0443] |  |
| 6 | 0.7 | 1.0 | 1 | 0 dB | csiqp | 0.9701 | +0.0428 [+0.0413, +0.0443] | 0.9701 | +0.0428 [+0.0413, +0.0443] |  |
| 6 | 0.7 | 1.0 | 1 | 0 dB | switch | 0.9701 | +0.0428 [+0.0413, +0.0443] | 0.9701 | +0.0428 [+0.0413, +0.0443] | frac 1.000; rule_loss +0.0000 [+0.0000, +0.0000]; decision_loss 0.0000; oracle_cf - cos +0.0000; frac_cf 1.000 |
| 6 | 0.7 | 1.0 | 1 | 0 dB | ods | 0.9701 | +0.0428 [+0.0413, +0.0443] | 0.9701 | +0.0428 [+0.0413, +0.0443] |  |
| 6 | 0.7 | 1.0 | 1 | 0 dB | csiqb | 0.9564 | +0.0292 [+0.0278, +0.0306] | 0.9564 | +0.0292 [+0.0278, +0.0306] |  |
| 6 | 0.7 | 1.0 | 1 | 0 dB | csiqw | 0.9690 | +0.0418 [+0.0403, +0.0433] | 0.9690 | +0.0418 [+0.0403, +0.0433] |  |
| 6 | 0.7 | 1.0 | 1 | 10 dB | fixed | 0.9885 | - | 0.9885 | - |  |
| 6 | 0.7 | 1.0 | 1 | 10 dB | csiq | 0.9950 | +0.0066 [+0.0063, +0.0068] | 0.9950 | +0.0066 [+0.0063, +0.0068] |  |
| 6 | 0.7 | 1.0 | 1 | 10 dB | csiqp | 0.9950 | +0.0066 [+0.0063, +0.0068] | 0.9950 | +0.0066 [+0.0063, +0.0068] |  |
| 6 | 0.7 | 1.0 | 1 | 10 dB | switch | 0.9950 | +0.0066 [+0.0063, +0.0068] | 0.9950 | +0.0066 [+0.0063, +0.0068] | frac 1.000; rule_loss +0.0000 [+0.0000, +0.0000]; decision_loss 0.0000; oracle_cf - cos +0.0000; frac_cf 1.000 |
| 6 | 0.7 | 1.0 | 1 | 10 dB | ods | 0.9950 | +0.0066 [+0.0063, +0.0068] | 0.9950 | +0.0066 [+0.0063, +0.0068] |  |
| 6 | 0.7 | 1.0 | 1 | 10 dB | csiqb | 0.9898 | +0.0013 [+0.0011, +0.0015] | 0.9898 | +0.0013 [+0.0011, +0.0015] |  |
| 6 | 0.7 | 1.0 | 1 | 10 dB | csiqw | 0.9949 | +0.0064 [+0.0062, +0.0067] | 0.9949 | +0.0064 [+0.0062, +0.0067] |  |
| 6 | 0.7 | 1.0 | 1 | 20 dB | fixed | 0.9972 | - | 0.9972 | - |  |
| 6 | 0.7 | 1.0 | 1 | 20 dB | csiq | 0.9979 | +0.0007 [+0.0007, +0.0007] | 0.9979 | +0.0007 [+0.0007, +0.0007] |  |
| 6 | 0.7 | 1.0 | 1 | 20 dB | csiqp | 0.9979 | +0.0007 [+0.0007, +0.0007] | 0.9979 | +0.0007 [+0.0007, +0.0007] |  |
| 6 | 0.7 | 1.0 | 1 | 20 dB | switch | 0.9979 | +0.0007 [+0.0007, +0.0007] | 0.9979 | +0.0007 [+0.0007, +0.0007] | frac 1.000; rule_loss +0.0000 [+0.0000, +0.0000]; decision_loss 0.0000; oracle_cf - cos +0.0000; frac_cf 1.000 |
| 6 | 0.7 | 1.0 | 1 | 20 dB | ods | 0.9979 | +0.0007 [+0.0007, +0.0007] | 0.9979 | +0.0007 [+0.0007, +0.0007] |  |
| 6 | 0.7 | 1.0 | 1 | 20 dB | csiqb | 0.9964 | -0.0008 [-0.0009, -0.0008] | 0.9964 | -0.0008 [-0.0009, -0.0008] |  |
| 6 | 0.7 | 1.0 | 1 | 20 dB | csiqw | 0.9979 | +0.0007 [+0.0007, +0.0007] | 0.9979 | +0.0007 [+0.0007, +0.0007] |  |
| 6 | 0.7 | 1.0 | 2 | 0 dB | fixed | 0.8779 | - | 0.8348 | - |  |
| 6 | 0.7 | 1.0 | 2 | 0 dB | csiq | 0.9531 | +0.0752 [+0.0731, +0.0773] | 0.9416 | +0.1068 [+0.1032, +0.1105] |  |
| 6 | 0.7 | 1.0 | 2 | 0 dB | csiqp | 0.9531 | +0.0752 [+0.0731, +0.0773] | 0.9416 | +0.1068 [+0.1032, +0.1105] |  |
| 6 | 0.7 | 1.0 | 2 | 0 dB | switch | 0.9531 | +0.0752 [+0.0731, +0.0773] | 0.9416 | +0.1068 [+0.1032, +0.1105] | frac 1.000; rule_loss +0.0000 [+0.0000, +0.0000]; decision_loss 0.0000; oracle_cf - cos +0.0000; frac_cf 1.000 |
| 6 | 0.7 | 1.0 | 2 | 0 dB | ods | 0.9512 | +0.0733 [+0.0712, +0.0754] | 0.9335 | +0.0987 [+0.0951, +0.1022] |  |
| 6 | 0.7 | 1.0 | 2 | 0 dB | csiqb | 0.9381 | +0.0601 [+0.0581, +0.0623] | 0.9256 | +0.0908 [+0.0872, +0.0944] |  |
| 6 | 0.7 | 1.0 | 2 | 0 dB | csiqw | 0.9492 | +0.0713 [+0.0691, +0.0734] | 0.9391 | +0.1042 [+0.1006, +0.1079] |  |
| 6 | 0.7 | 1.0 | 2 | 10 dB | fixed | 0.9741 | - | 0.9629 | - |  |
| 6 | 0.7 | 1.0 | 2 | 10 dB | csiq | 0.9920 | +0.0179 [+0.0172, +0.0186] | 0.9899 | +0.0271 [+0.0258, +0.0284] |  |
| 6 | 0.7 | 1.0 | 2 | 10 dB | csiqp | 0.9920 | +0.0179 [+0.0172, +0.0186] | 0.9899 | +0.0271 [+0.0258, +0.0284] |  |
| 6 | 0.7 | 1.0 | 2 | 10 dB | switch | 0.9920 | +0.0179 [+0.0172, +0.0186] | 0.9899 | +0.0271 [+0.0258, +0.0284] | frac 1.000; rule_loss +0.0000 [+0.0000, +0.0000]; decision_loss 0.0000; oracle_cf - cos +0.0000; frac_cf 1.000 |
| 6 | 0.7 | 1.0 | 2 | 10 dB | ods | 0.9914 | +0.0174 [+0.0167, +0.0181] | 0.9881 | +0.0252 [+0.0240, +0.0266] |  |
| 6 | 0.7 | 1.0 | 2 | 10 dB | csiqb | 0.9839 | +0.0099 [+0.0092, +0.0105] | 0.9803 | +0.0174 [+0.0162, +0.0187] |  |
| 6 | 0.7 | 1.0 | 2 | 10 dB | csiqw | 0.9917 | +0.0177 [+0.0169, +0.0184] | 0.9897 | +0.0268 [+0.0256, +0.0282] |  |
| 6 | 0.7 | 1.0 | 2 | 20 dB | fixed | 0.9953 | - | 0.9934 | - |  |
| 6 | 0.7 | 1.0 | 2 | 20 dB | csiq | 0.9977 | +0.0023 [+0.0022, +0.0024] | 0.9967 | +0.0033 [+0.0031, +0.0035] |  |
| 6 | 0.7 | 1.0 | 2 | 20 dB | csiqp | 0.9977 | +0.0023 [+0.0022, +0.0024] | 0.9967 | +0.0033 [+0.0031, +0.0035] |  |
| 6 | 0.7 | 1.0 | 2 | 20 dB | switch | 0.9977 | +0.0023 [+0.0022, +0.0024] | 0.9967 | +0.0033 [+0.0031, +0.0035] | frac 1.000; rule_loss +0.0000 [+0.0000, +0.0000]; decision_loss 0.0000; oracle_cf - cos +0.0000; frac_cf 1.000 |
| 6 | 0.7 | 1.0 | 2 | 20 dB | ods | 0.9976 | +0.0023 [+0.0022, +0.0024] | 0.9966 | +0.0032 [+0.0030, +0.0034] |  |
| 6 | 0.7 | 1.0 | 2 | 20 dB | csiqb | 0.9955 | +0.0002 [+0.0001, +0.0003] | 0.9939 | +0.0005 [+0.0003, +0.0007] |  |
| 6 | 0.7 | 1.0 | 2 | 20 dB | csiqw | 0.9976 | +0.0023 [+0.0022, +0.0024] | 0.9967 | +0.0033 [+0.0031, +0.0035] |  |
| 6 | 0.7 | 1.0 | 4 | 0 dB | fixed | 0.7862 | - | 0.6183 | - |  |
| 6 | 0.7 | 1.0 | 4 | 0 dB | csiq | 0.9221 | +0.1359 [+0.1327, +0.1392] | 0.8851 | +0.2669 [+0.2591, +0.2754] |  |
| 6 | 0.7 | 1.0 | 4 | 0 dB | csiqp | 0.9221 | +0.1359 [+0.1327, +0.1392] | 0.8851 | +0.2669 [+0.2591, +0.2754] |  |
| 6 | 0.7 | 1.0 | 4 | 0 dB | switch | 0.9221 | +0.1359 [+0.1327, +0.1392] | 0.8851 | +0.2669 [+0.2591, +0.2754] | frac 1.000; rule_loss +0.0000 [+0.0000, +0.0000]; decision_loss 0.0000; oracle_cf - cos +0.0000; frac_cf 1.000 |
| 6 | 0.7 | 1.0 | 4 | 0 dB | ods | 0.9195 | +0.1333 [+0.1301, +0.1365] | 0.8642 | +0.2459 [+0.2384, +0.2544] |  |
| 6 | 0.7 | 1.0 | 4 | 0 dB | csiqb | 0.9073 | +0.1212 [+0.1179, +0.1244] | 0.8697 | +0.2515 [+0.2438, +0.2599] |  |
| 6 | 0.7 | 1.0 | 4 | 0 dB | csiqw | 0.9159 | +0.1297 [+0.1265, +0.1330] | 0.8820 | +0.2638 [+0.2560, +0.2722] |  |
| 6 | 0.7 | 1.0 | 4 | 10 dB | fixed | 0.9319 | - | 0.8541 | - |  |
| 6 | 0.7 | 1.0 | 4 | 10 dB | csiq | 0.9841 | +0.0522 [+0.0506, +0.0538] | 0.9731 | +0.1190 [+0.1139, +0.1239] |  |
| 6 | 0.7 | 1.0 | 4 | 10 dB | csiqp | 0.9841 | +0.0522 [+0.0506, +0.0538] | 0.9731 | +0.1190 [+0.1139, +0.1239] |  |
| 6 | 0.7 | 1.0 | 4 | 10 dB | switch | 0.9841 | +0.0522 [+0.0506, +0.0538] | 0.9731 | +0.1190 [+0.1139, +0.1239] | frac 1.000; rule_loss +0.0000 [+0.0000, +0.0000]; decision_loss 0.0000; oracle_cf - cos +0.0000; frac_cf 1.000 |
| 6 | 0.7 | 1.0 | 4 | 10 dB | ods | 0.9842 | +0.0522 [+0.0506, +0.0539] | 0.9704 | +0.1162 [+0.1112, +0.1211] |  |
| 6 | 0.7 | 1.0 | 4 | 10 dB | csiqb | 0.9729 | +0.0410 [+0.0394, +0.0426] | 0.9573 | +0.1031 [+0.0980, +0.1082] |  |
| 6 | 0.7 | 1.0 | 4 | 10 dB | csiqw | 0.9836 | +0.0516 [+0.0500, +0.0532] | 0.9725 | +0.1183 [+0.1132, +0.1233] |  |
| 6 | 0.7 | 1.0 | 4 | 20 dB | fixed | 0.9852 | - | 0.9652 | - |  |
| 6 | 0.7 | 1.0 | 4 | 20 dB | csiq | 0.9967 | +0.0114 [+0.0109, +0.0121] | 0.9940 | +0.0288 [+0.0267, +0.0311] |  |
| 6 | 0.7 | 1.0 | 4 | 20 dB | csiqp | 0.9967 | +0.0114 [+0.0109, +0.0121] | 0.9940 | +0.0288 [+0.0267, +0.0311] |  |
| 6 | 0.7 | 1.0 | 4 | 20 dB | switch | 0.9967 | +0.0114 [+0.0109, +0.0121] | 0.9940 | +0.0288 [+0.0267, +0.0311] | frac 1.000; rule_loss +0.0000 [+0.0000, +0.0000]; decision_loss 0.0000; oracle_cf - cos +0.0000; frac_cf 1.000 |
| 6 | 0.7 | 1.0 | 4 | 20 dB | ods | 0.9968 | +0.0115 [+0.0110, +0.0122] | 0.9942 | +0.0290 [+0.0268, +0.0313] |  |
| 6 | 0.7 | 1.0 | 4 | 20 dB | csiqb | 0.9927 | +0.0074 [+0.0068, +0.0080] | 0.9872 | +0.0220 [+0.0198, +0.0243] |  |
| 6 | 0.7 | 1.0 | 4 | 20 dB | csiqw | 0.9965 | +0.0112 [+0.0107, +0.0119] | 0.9936 | +0.0284 [+0.0263, +0.0307] |  |
| 6 | 0.7 | 1.0 | 6 | 0 dB | fixed | 0.7472 | - | 0.4924 | - |  |
| 6 | 0.7 | 1.0 | 6 | 0 dB | csiq | 0.8987 | +0.1515 [+0.1487, +0.1541] | 0.8386 | +0.3463 [+0.3370, +0.3545] |  |
| 6 | 0.7 | 1.0 | 6 | 0 dB | csiqp | 0.8987 | +0.1515 [+0.1487, +0.1541] | 0.8386 | +0.3463 [+0.3370, +0.3545] |  |
| 6 | 0.7 | 1.0 | 6 | 0 dB | switch | 0.8987 | +0.1515 [+0.1487, +0.1541] | 0.8386 | +0.3463 [+0.3370, +0.3545] | frac 1.000; rule_loss +0.0000 [+0.0000, +0.0000]; decision_loss 0.0000; oracle_cf - cos +0.0000; frac_cf 1.000 |
| 6 | 0.7 | 1.0 | 6 | 0 dB | ods | 0.8963 | +0.1491 [+0.1463, +0.1517] | 0.8105 | +0.3181 [+0.3088, +0.3265] |  |
| 6 | 0.7 | 1.0 | 6 | 0 dB | csiqb | 0.8848 | +0.1376 [+0.1349, +0.1402] | 0.8243 | +0.3319 [+0.3227, +0.3401] |  |
| 6 | 0.7 | 1.0 | 6 | 0 dB | csiqw | 0.8916 | +0.1444 [+0.1417, +0.1471] | 0.8353 | +0.3429 [+0.3336, +0.3511] |  |
| 6 | 0.7 | 1.0 | 6 | 10 dB | fixed | 0.9114 | - | 0.7876 | - |  |
| 6 | 0.7 | 1.0 | 6 | 10 dB | csiq | 0.9702 | +0.0588 [+0.0572, +0.0605] | 0.9498 | +0.1622 [+0.1559, +0.1691] |  |
| 6 | 0.7 | 1.0 | 6 | 10 dB | csiqp | 0.9702 | +0.0588 [+0.0572, +0.0605] | 0.9498 | +0.1622 [+0.1559, +0.1691] |  |
| 6 | 0.7 | 1.0 | 6 | 10 dB | switch | 0.9702 | +0.0588 [+0.0572, +0.0605] | 0.9498 | +0.1622 [+0.1559, +0.1691] | frac 1.000; rule_loss +0.0000 [+0.0000, +0.0000]; decision_loss 0.0000; oracle_cf - cos +0.0000; frac_cf 1.000 |
| 6 | 0.7 | 1.0 | 6 | 10 dB | ods | 0.9706 | +0.0591 [+0.0576, +0.0608] | 0.9494 | +0.1619 [+0.1555, +0.1687] |  |
| 6 | 0.7 | 1.0 | 6 | 10 dB | csiqb | 0.9603 | +0.0488 [+0.0473, +0.0505] | 0.9344 | +0.1468 [+0.1405, +0.1536] |  |
| 6 | 0.7 | 1.0 | 6 | 10 dB | csiqw | 0.9697 | +0.0582 [+0.0567, +0.0599] | 0.9491 | +0.1615 [+0.1552, +0.1684] |  |
| 6 | 0.7 | 1.0 | 6 | 20 dB | fixed | 0.9707 | - | 0.9341 | - |  |
| 6 | 0.7 | 1.0 | 6 | 20 dB | csiq | 0.9838 | +0.0131 [+0.0125, +0.0138] | 0.9761 | +0.0420 [+0.0391, +0.0454] |  |
| 6 | 0.7 | 1.0 | 6 | 20 dB | csiqp | 0.9838 | +0.0131 [+0.0125, +0.0138] | 0.9761 | +0.0420 [+0.0391, +0.0454] |  |
| 6 | 0.7 | 1.0 | 6 | 20 dB | switch | 0.9838 | +0.0131 [+0.0125, +0.0138] | 0.9761 | +0.0420 [+0.0391, +0.0454] | frac 1.000; rule_loss +0.0000 [+0.0000, +0.0000]; decision_loss 0.0000; oracle_cf - cos +0.0000; frac_cf 1.000 |
| 6 | 0.7 | 1.0 | 6 | 20 dB | ods | 0.9840 | +0.0133 [+0.0127, +0.0140] | 0.9767 | +0.0426 [+0.0397, +0.0460] |  |
| 6 | 0.7 | 1.0 | 6 | 20 dB | csiqb | 0.9805 | +0.0098 [+0.0092, +0.0104] | 0.9696 | +0.0356 [+0.0327, +0.0389] |  |
| 6 | 0.7 | 1.0 | 6 | 20 dB | csiqw | 0.9836 | +0.0129 [+0.0123, +0.0136] | 0.9756 | +0.0415 [+0.0386, +0.0449] |  |
| 6 | 0.7 | 0.9 | 1 | 0 dB | fixed | 0.9268 | - | 0.9268 | - |  |
| 6 | 0.7 | 0.9 | 1 | 0 dB | csiq | 0.9628 | +0.0360 [+0.0345, +0.0374] | 0.9628 | +0.0360 [+0.0345, +0.0374] |  |
| 6 | 0.7 | 0.9 | 1 | 0 dB | csiqp | 0.9628 | +0.0360 [+0.0346, +0.0374] | 0.9628 | +0.0360 [+0.0346, +0.0374] |  |
| 6 | 0.7 | 0.9 | 1 | 0 dB | switch | 0.9628 | +0.0360 [+0.0345, +0.0374] | 0.9628 | +0.0360 [+0.0345, +0.0374] | frac 1.000; rule_loss +0.0000 [+0.0000, +0.0000]; decision_loss 0.0004; oracle_cf - cos +0.0003; frac_cf 0.986 |
| 6 | 0.7 | 0.9 | 1 | 0 dB | ods | 0.9628 | +0.0360 [+0.0345, +0.0374] | 0.9628 | +0.0360 [+0.0345, +0.0374] |  |
| 6 | 0.7 | 0.9 | 1 | 0 dB | csiqb | 0.9495 | +0.0227 [+0.0213, +0.0241] | 0.9495 | +0.0227 [+0.0213, +0.0241] |  |
| 6 | 0.7 | 0.9 | 1 | 0 dB | csiqw | 0.9619 | +0.0350 [+0.0336, +0.0364] | 0.9619 | +0.0350 [+0.0336, +0.0364] |  |
| 6 | 0.7 | 0.9 | 1 | 10 dB | fixed | 0.9884 | - | 0.9884 | - |  |
| 6 | 0.7 | 0.9 | 1 | 10 dB | csiq | 0.9936 | +0.0052 [+0.0049, +0.0054] | 0.9936 | +0.0052 [+0.0049, +0.0054] |  |
| 6 | 0.7 | 0.9 | 1 | 10 dB | csiqp | 0.9936 | +0.0052 [+0.0049, +0.0054] | 0.9936 | +0.0052 [+0.0049, +0.0054] |  |
| 6 | 0.7 | 0.9 | 1 | 10 dB | switch | 0.9936 | +0.0052 [+0.0049, +0.0054] | 0.9936 | +0.0052 [+0.0049, +0.0054] | frac 0.999; rule_loss +0.0000 [+0.0000, +0.0000]; decision_loss 0.0001; oracle_cf - cos +0.0001; frac_cf 0.983 |
| 6 | 0.7 | 0.9 | 1 | 10 dB | ods | 0.9936 | +0.0052 [+0.0049, +0.0054] | 0.9936 | +0.0052 [+0.0049, +0.0054] |  |
| 6 | 0.7 | 0.9 | 1 | 10 dB | csiqb | 0.9875 | -0.0009 [-0.0011, -0.0007] | 0.9875 | -0.0009 [-0.0011, -0.0007] |  |
| 6 | 0.7 | 0.9 | 1 | 10 dB | csiqw | 0.9935 | +0.0051 [+0.0048, +0.0053] | 0.9935 | +0.0051 [+0.0048, +0.0053] |  |
| 6 | 0.7 | 0.9 | 1 | 20 dB | fixed | 0.9973 | - | 0.9973 | - |  |
| 6 | 0.7 | 0.9 | 1 | 20 dB | csiq | 0.9979 | +0.0006 [+0.0005, +0.0006] | 0.9979 | +0.0006 [+0.0005, +0.0006] |  |
| 6 | 0.7 | 0.9 | 1 | 20 dB | csiqp | 0.9979 | +0.0006 [+0.0005, +0.0006] | 0.9979 | +0.0006 [+0.0005, +0.0006] |  |
| 6 | 0.7 | 0.9 | 1 | 20 dB | switch | 0.9979 | +0.0006 [+0.0005, +0.0006] | 0.9979 | +0.0006 [+0.0005, +0.0006] | frac 0.998; rule_loss -0.0000 [-0.0000, +0.0000]; decision_loss 0.0000; oracle_cf - cos +0.0000; frac_cf 0.975 |
| 6 | 0.7 | 0.9 | 1 | 20 dB | ods | 0.9979 | +0.0006 [+0.0005, +0.0006] | 0.9979 | +0.0006 [+0.0005, +0.0006] |  |
| 6 | 0.7 | 0.9 | 1 | 20 dB | csiqb | 0.9961 | -0.0013 [-0.0014, -0.0012] | 0.9961 | -0.0013 [-0.0014, -0.0012] |  |
| 6 | 0.7 | 0.9 | 1 | 20 dB | csiqw | 0.9979 | +0.0005 [+0.0005, +0.0006] | 0.9979 | +0.0005 [+0.0005, +0.0006] |  |
| 6 | 0.7 | 0.9 | 2 | 0 dB | fixed | 0.8766 | - | 0.8326 | - |  |
| 6 | 0.7 | 0.9 | 2 | 0 dB | csiq | 0.9433 | +0.0667 [+0.0647, +0.0686] | 0.9280 | +0.0954 [+0.0919, +0.0988] |  |
| 6 | 0.7 | 0.9 | 2 | 0 dB | csiqp | 0.9433 | +0.0667 [+0.0647, +0.0687] | 0.9279 | +0.0953 [+0.0919, +0.0988] |  |
| 6 | 0.7 | 0.9 | 2 | 0 dB | switch | 0.9433 | +0.0667 [+0.0647, +0.0686] | 0.9280 | +0.0954 [+0.0919, +0.0988] | frac 1.000; rule_loss +0.0000 [+0.0000, +0.0000]; decision_loss 0.0000; oracle_cf - cos -0.0000; frac_cf 0.996 |
| 6 | 0.7 | 0.9 | 2 | 0 dB | ods | 0.9410 | +0.0644 [+0.0623, +0.0663] | 0.9189 | +0.0863 [+0.0829, +0.0897] |  |
| 6 | 0.7 | 0.9 | 2 | 0 dB | csiqb | 0.9291 | +0.0525 [+0.0505, +0.0545] | 0.9133 | +0.0807 [+0.0772, +0.0842] |  |
| 6 | 0.7 | 0.9 | 2 | 0 dB | csiqw | 0.9401 | +0.0635 [+0.0614, +0.0654] | 0.9260 | +0.0934 [+0.0899, +0.0968] |  |
| 6 | 0.7 | 0.9 | 2 | 10 dB | fixed | 0.9742 | - | 0.9630 | - |  |
| 6 | 0.7 | 0.9 | 2 | 10 dB | csiq | 0.9897 | +0.0155 [+0.0148, +0.0161] | 0.9867 | +0.0237 [+0.0225, +0.0249] |  |
| 6 | 0.7 | 0.9 | 2 | 10 dB | csiqp | 0.9896 | +0.0155 [+0.0148, +0.0161] | 0.9866 | +0.0236 [+0.0224, +0.0248] |  |
| 6 | 0.7 | 0.9 | 2 | 10 dB | switch | 0.9897 | +0.0155 [+0.0148, +0.0161] | 0.9867 | +0.0237 [+0.0225, +0.0249] | frac 1.000; rule_loss +0.0000 [+0.0000, +0.0000]; decision_loss 0.0000; oracle_cf - cos +0.0000; frac_cf 0.997 |
| 6 | 0.7 | 0.9 | 2 | 10 dB | ods | 0.9889 | +0.0148 [+0.0141, +0.0154] | 0.9844 | +0.0213 [+0.0201, +0.0225] |  |
| 6 | 0.7 | 0.9 | 2 | 10 dB | csiqb | 0.9808 | +0.0066 [+0.0060, +0.0072] | 0.9761 | +0.0131 [+0.0119, +0.0142] |  |
| 6 | 0.7 | 0.9 | 2 | 10 dB | csiqw | 0.9894 | +0.0152 [+0.0145, +0.0159] | 0.9864 | +0.0234 [+0.0222, +0.0246] |  |
| 6 | 0.7 | 0.9 | 2 | 20 dB | fixed | 0.9953 | - | 0.9934 | - |  |
| 6 | 0.7 | 0.9 | 2 | 20 dB | csiq | 0.9974 | +0.0020 [+0.0019, +0.0021] | 0.9964 | +0.0030 [+0.0028, +0.0032] |  |
| 6 | 0.7 | 0.9 | 2 | 20 dB | csiqp | 0.9974 | +0.0020 [+0.0019, +0.0021] | 0.9964 | +0.0030 [+0.0028, +0.0032] |  |
| 6 | 0.7 | 0.9 | 2 | 20 dB | switch | 0.9974 | +0.0020 [+0.0019, +0.0021] | 0.9964 | +0.0030 [+0.0028, +0.0032] | frac 0.999; rule_loss +0.0000 [+0.0000, +0.0000]; decision_loss 0.0000; oracle_cf - cos +0.0000; frac_cf 0.995 |
| 6 | 0.7 | 0.9 | 2 | 20 dB | ods | 0.9973 | +0.0020 [+0.0019, +0.0021] | 0.9962 | +0.0028 [+0.0026, +0.0030] |  |
| 6 | 0.7 | 0.9 | 2 | 20 dB | csiqb | 0.9948 | -0.0006 [-0.0007, -0.0004] | 0.9929 | -0.0005 [-0.0007, -0.0003] |  |
| 6 | 0.7 | 0.9 | 2 | 20 dB | csiqw | 0.9973 | +0.0020 [+0.0019, +0.0021] | 0.9963 | +0.0029 [+0.0027, +0.0031] |  |
| 6 | 0.7 | 0.9 | 4 | 0 dB | fixed | 0.7833 | - | 0.6112 | - |  |
| 6 | 0.7 | 0.9 | 4 | 0 dB | csiq | 0.9083 | +0.1251 [+0.1221, +0.1283] | 0.8599 | +0.2487 [+0.2413, +0.2566] |  |
| 6 | 0.7 | 0.9 | 4 | 0 dB | csiqp | 0.9084 | +0.1251 [+0.1222, +0.1284] | 0.8597 | +0.2485 [+0.2410, +0.2565] |  |
| 6 | 0.7 | 0.9 | 4 | 0 dB | switch | 0.9083 | +0.1251 [+0.1221, +0.1283] | 0.8599 | +0.2487 [+0.2413, +0.2566] | frac 1.000; rule_loss +0.0000 [+0.0000, +0.0000]; decision_loss 0.0000; oracle_cf - cos +0.0000; frac_cf 0.999 |
| 6 | 0.7 | 0.9 | 4 | 0 dB | ods | 0.9054 | +0.1221 [+0.1192, +0.1253] | 0.8380 | +0.2268 [+0.2192, +0.2348] |  |
| 6 | 0.7 | 0.9 | 4 | 0 dB | csiqb | 0.8949 | +0.1116 [+0.1087, +0.1148] | 0.8475 | +0.2363 [+0.2288, +0.2441] |  |
| 6 | 0.7 | 0.9 | 4 | 0 dB | csiqw | 0.9033 | +0.1200 [+0.1171, +0.1232] | 0.8575 | +0.2462 [+0.2388, +0.2541] |  |
| 6 | 0.7 | 0.9 | 4 | 10 dB | fixed | 0.9320 | - | 0.8558 | - |  |
| 6 | 0.7 | 0.9 | 4 | 10 dB | csiq | 0.9800 | +0.0479 [+0.0462, +0.0496] | 0.9649 | +0.1091 [+0.1042, +0.1140] |  |
| 6 | 0.7 | 0.9 | 4 | 10 dB | csiqp | 0.9800 | +0.0480 [+0.0463, +0.0496] | 0.9649 | +0.1091 [+0.1042, +0.1140] |  |
| 6 | 0.7 | 0.9 | 4 | 10 dB | switch | 0.9800 | +0.0479 [+0.0462, +0.0496] | 0.9649 | +0.1091 [+0.1042, +0.1140] | frac 1.000; rule_loss +0.0000 [+0.0000, +0.0000]; decision_loss 0.0000; oracle_cf - cos +0.0000; frac_cf 1.000 |
| 6 | 0.7 | 0.9 | 4 | 10 dB | ods | 0.9798 | +0.0478 [+0.0461, +0.0494] | 0.9615 | +0.1057 [+0.1008, +0.1106] |  |
| 6 | 0.7 | 0.9 | 4 | 10 dB | csiqb | 0.9685 | +0.0365 [+0.0348, +0.0381] | 0.9496 | +0.0938 [+0.0888, +0.0987] |  |
| 6 | 0.7 | 0.9 | 4 | 10 dB | csiqw | 0.9795 | +0.0475 [+0.0458, +0.0491] | 0.9645 | +0.1087 [+0.1038, +0.1136] |  |
| 6 | 0.7 | 0.9 | 4 | 20 dB | fixed | 0.9854 | - | 0.9654 | - |  |
| 6 | 0.7 | 0.9 | 4 | 20 dB | csiq | 0.9959 | +0.0105 [+0.0099, +0.0111] | 0.9924 | +0.0270 [+0.0249, +0.0292] |  |
| 6 | 0.7 | 0.9 | 4 | 20 dB | csiqp | 0.9959 | +0.0105 [+0.0099, +0.0111] | 0.9924 | +0.0270 [+0.0249, +0.0292] |  |
| 6 | 0.7 | 0.9 | 4 | 20 dB | switch | 0.9959 | +0.0105 [+0.0099, +0.0111] | 0.9924 | +0.0270 [+0.0249, +0.0292] | frac 1.000; rule_loss +0.0000 [+0.0000, +0.0000]; decision_loss 0.0000; oracle_cf - cos +0.0000; frac_cf 1.000 |
| 6 | 0.7 | 0.9 | 4 | 20 dB | ods | 0.9960 | +0.0106 [+0.0100, +0.0112] | 0.9924 | +0.0270 [+0.0250, +0.0292] |  |
| 6 | 0.7 | 0.9 | 4 | 20 dB | csiqb | 0.9913 | +0.0059 [+0.0053, +0.0065] | 0.9845 | +0.0191 [+0.0170, +0.0212] |  |
| 6 | 0.7 | 0.9 | 4 | 20 dB | csiqw | 0.9957 | +0.0103 [+0.0097, +0.0109] | 0.9920 | +0.0266 [+0.0245, +0.0288] |  |
| 6 | 0.7 | 0.9 | 6 | 0 dB | fixed | 0.7471 | - | 0.4946 | - |  |
| 6 | 0.7 | 0.9 | 6 | 0 dB | csiq | 0.8842 | +0.1371 [+0.1343, +0.1398] | 0.8080 | +0.3133 [+0.3044, +0.3212] |  |
| 6 | 0.7 | 0.9 | 6 | 0 dB | csiqp | 0.8843 | +0.1372 [+0.1344, +0.1399] | 0.8078 | +0.3131 [+0.3041, +0.3211] |  |
| 6 | 0.7 | 0.9 | 6 | 0 dB | switch | 0.8842 | +0.1371 [+0.1343, +0.1398] | 0.8080 | +0.3133 [+0.3044, +0.3212] | frac 1.000; rule_loss +0.0000 [+0.0000, +0.0000]; decision_loss 0.0000; oracle_cf - cos -0.0000; frac_cf 1.000 |
| 6 | 0.7 | 0.9 | 6 | 0 dB | ods | 0.8814 | +0.1342 [+0.1315, +0.1369] | 0.7781 | +0.2835 [+0.2742, +0.2922] |  |
| 6 | 0.7 | 0.9 | 6 | 0 dB | csiqb | 0.8718 | +0.1246 [+0.1218, +0.1273] | 0.7974 | +0.3027 [+0.2938, +0.3106] |  |
| 6 | 0.7 | 0.9 | 6 | 0 dB | csiqw | 0.8785 | +0.1314 [+0.1286, +0.1341] | 0.8051 | +0.3105 [+0.3014, +0.3183] |  |
| 6 | 0.7 | 0.9 | 6 | 10 dB | fixed | 0.9130 | - | 0.7946 | - |  |
| 6 | 0.7 | 0.9 | 6 | 10 dB | csiq | 0.9655 | +0.0525 [+0.0511, +0.0539] | 0.9385 | +0.1440 [+0.1385, +0.1498] |  |
| 6 | 0.7 | 0.9 | 6 | 10 dB | csiqp | 0.9655 | +0.0525 [+0.0512, +0.0539] | 0.9387 | +0.1442 [+0.1386, +0.1499] |  |
| 6 | 0.7 | 0.9 | 6 | 10 dB | switch | 0.9655 | +0.0525 [+0.0511, +0.0539] | 0.9385 | +0.1440 [+0.1385, +0.1498] | frac 1.000; rule_loss +0.0000 [+0.0000, +0.0000]; decision_loss 0.0000; oracle_cf - cos +0.0000; frac_cf 1.000 |
| 6 | 0.7 | 0.9 | 6 | 10 dB | ods | 0.9658 | +0.0528 [+0.0514, +0.0542] | 0.9375 | +0.1430 [+0.1375, +0.1488] |  |
| 6 | 0.7 | 0.9 | 6 | 10 dB | csiqb | 0.9553 | +0.0424 [+0.0410, +0.0437] | 0.9243 | +0.1297 [+0.1241, +0.1356] |  |
| 6 | 0.7 | 0.9 | 6 | 10 dB | csiqw | 0.9651 | +0.0521 [+0.0507, +0.0535] | 0.9381 | +0.1436 [+0.1381, +0.1494] |  |
| 6 | 0.7 | 0.9 | 6 | 20 dB | fixed | 0.9713 | - | 0.9366 | - |  |
| 6 | 0.7 | 0.9 | 6 | 20 dB | csiq | 0.9831 | +0.0118 [+0.0113, +0.0124] | 0.9746 | +0.0380 [+0.0354, +0.0410] |  |
| 6 | 0.7 | 0.9 | 6 | 20 dB | csiqp | 0.9831 | +0.0118 [+0.0113, +0.0124] | 0.9750 | +0.0384 [+0.0358, +0.0414] |  |
| 6 | 0.7 | 0.9 | 6 | 20 dB | switch | 0.9831 | +0.0118 [+0.0113, +0.0124] | 0.9746 | +0.0380 [+0.0354, +0.0410] | frac 1.000; rule_loss +0.0000 [+0.0000, +0.0000]; decision_loss 0.0000; oracle_cf - cos +0.0000; frac_cf 0.999 |
| 6 | 0.7 | 0.9 | 6 | 20 dB | ods | 0.9833 | +0.0120 [+0.0115, +0.0126] | 0.9753 | +0.0387 [+0.0361, +0.0417] |  |
| 6 | 0.7 | 0.9 | 6 | 20 dB | csiqb | 0.9793 | +0.0080 [+0.0075, +0.0086] | 0.9674 | +0.0308 [+0.0282, +0.0338] |  |
| 6 | 0.7 | 0.9 | 6 | 20 dB | csiqw | 0.9829 | +0.0116 [+0.0111, +0.0122] | 0.9742 | +0.0376 [+0.0350, +0.0406] |  |
| 6 | 0.7 | 0.7 | 1 | 0 dB | fixed | 0.9267 | - | 0.9267 | - |  |
| 6 | 0.7 | 0.7 | 1 | 0 dB | csiq | 0.9405 | +0.0137 [+0.0110, +0.0162] | 0.9405 | +0.0137 [+0.0110, +0.0162] |  |
| 6 | 0.7 | 0.7 | 1 | 0 dB | csiqp | 0.9411 | +0.0144 [+0.0117, +0.0168] | 0.9411 | +0.0144 [+0.0117, +0.0168] |  |
| 6 | 0.7 | 0.7 | 1 | 0 dB | switch | 0.9411 | +0.0143 [+0.0119, +0.0165] | 0.9411 | +0.0143 [+0.0119, +0.0165] | frac 0.812; rule_loss -0.0006 [-0.0016, +0.0002]; decision_loss 0.0086; oracle_cf - cos +0.0084; frac_cf 0.787 |
| 6 | 0.7 | 0.7 | 1 | 0 dB | ods | 0.9405 | +0.0137 [+0.0110, +0.0162] | 0.9405 | +0.0137 [+0.0110, +0.0162] |  |
| 6 | 0.7 | 0.7 | 1 | 0 dB | csiqb | 0.9301 | +0.0034 [+0.0011, +0.0055] | 0.9301 | +0.0034 [+0.0011, +0.0055] |  |
| 6 | 0.7 | 0.7 | 1 | 0 dB | csiqw | 0.9403 | +0.0136 [+0.0109, +0.0160] | 0.9403 | +0.0136 [+0.0109, +0.0160] |  |
| 6 | 0.7 | 0.7 | 1 | 10 dB | fixed | 0.9885 | - | 0.9885 | - |  |
| 6 | 0.7 | 0.7 | 1 | 10 dB | csiq | 0.9885 | +0.0001 [-0.0006, +0.0006] | 0.9885 | +0.0001 [-0.0006, +0.0006] |  |
| 6 | 0.7 | 0.7 | 1 | 10 dB | csiqp | 0.9886 | +0.0001 [-0.0006, +0.0007] | 0.9886 | +0.0001 [-0.0006, +0.0007] |  |
| 6 | 0.7 | 0.7 | 1 | 10 dB | switch | 0.9890 | +0.0006 [+0.0001, +0.0009] | 0.9890 | +0.0006 [+0.0001, +0.0009] | frac 0.529; rule_loss -0.0005 [-0.0008, -0.0001]; decision_loss 0.0022; oracle_cf - cos +0.0021; frac_cf 0.696 |
| 6 | 0.7 | 0.7 | 1 | 10 dB | ods | 0.9885 | +0.0001 [-0.0006, +0.0006] | 0.9885 | +0.0001 [-0.0006, +0.0006] |  |
| 6 | 0.7 | 0.7 | 1 | 10 dB | csiqb | 0.9816 | -0.0069 [-0.0075, -0.0064] | 0.9816 | -0.0069 [-0.0075, -0.0064] |  |
| 6 | 0.7 | 0.7 | 1 | 10 dB | csiqw | 0.9885 | +0.0001 [-0.0006, +0.0006] | 0.9885 | +0.0001 [-0.0006, +0.0006] |  |
| 6 | 0.7 | 0.7 | 1 | 20 dB | fixed | 0.9973 | - | 0.9973 | - |  |
| 6 | 0.7 | 0.7 | 1 | 20 dB | csiq | 0.9969 | -0.0004 [-0.0007, -0.0003] | 0.9969 | -0.0004 [-0.0007, -0.0003] |  |
| 6 | 0.7 | 0.7 | 1 | 20 dB | csiqp | 0.9969 | -0.0004 [-0.0007, -0.0003] | 0.9969 | -0.0004 [-0.0007, -0.0003] |  |
| 6 | 0.7 | 0.7 | 1 | 20 dB | switch | 0.9973 | -0.0000 [-0.0001, +0.0000] | 0.9973 | -0.0000 [-0.0001, +0.0000] | frac 0.376; rule_loss +0.0000 [-0.0000, +0.0001]; decision_loss 0.0003; oracle_cf - cos +0.0003; frac_cf 0.671 |
| 6 | 0.7 | 0.7 | 1 | 20 dB | ods | 0.9969 | -0.0004 [-0.0007, -0.0003] | 0.9969 | -0.0004 [-0.0007, -0.0003] |  |
| 6 | 0.7 | 0.7 | 1 | 20 dB | csiqb | 0.9944 | -0.0029 [-0.0031, -0.0027] | 0.9944 | -0.0029 [-0.0031, -0.0027] |  |
| 6 | 0.7 | 0.7 | 1 | 20 dB | csiqw | 0.9969 | -0.0005 [-0.0007, -0.0003] | 0.9969 | -0.0005 [-0.0007, -0.0003] |  |
| 6 | 0.7 | 0.7 | 2 | 0 dB | fixed | 0.8764 | - | 0.8320 | - |  |
| 6 | 0.7 | 0.7 | 2 | 0 dB | csiq | 0.9135 | +0.0371 [+0.0345, +0.0394] | 0.8834 | +0.0513 [+0.0468, +0.0556] |  |
| 6 | 0.7 | 0.7 | 2 | 0 dB | csiqp | 0.9146 | +0.0382 [+0.0357, +0.0406] | 0.8848 | +0.0528 [+0.0483, +0.0571] |  |
| 6 | 0.7 | 0.7 | 2 | 0 dB | switch | 0.9133 | +0.0369 [+0.0344, +0.0393] | 0.8832 | +0.0511 [+0.0467, +0.0555] | frac 0.989; rule_loss +0.0002 [-0.0001, +0.0005]; decision_loss 0.0071; oracle_cf - cos +0.0069; frac_cf 0.838 |
| 6 | 0.7 | 0.7 | 2 | 0 dB | ods | 0.9114 | +0.0350 [+0.0323, +0.0374] | 0.8748 | +0.0428 [+0.0380, +0.0472] |  |
| 6 | 0.7 | 0.7 | 2 | 0 dB | csiqb | 0.9040 | +0.0276 [+0.0252, +0.0298] | 0.8767 | +0.0447 [+0.0405, +0.0488] |  |
| 6 | 0.7 | 0.7 | 2 | 0 dB | csiqw | 0.9125 | +0.0361 [+0.0337, +0.0385] | 0.8842 | +0.0521 [+0.0476, +0.0564] |  |
| 6 | 0.7 | 0.7 | 2 | 10 dB | fixed | 0.9741 | - | 0.9627 | - |  |
| 6 | 0.7 | 0.7 | 2 | 10 dB | csiq | 0.9806 | +0.0065 [+0.0054, +0.0076] | 0.9726 | +0.0099 [+0.0077, +0.0120] |  |
| 6 | 0.7 | 0.7 | 2 | 10 dB | csiqp | 0.9807 | +0.0067 [+0.0056, +0.0077] | 0.9728 | +0.0100 [+0.0079, +0.0121] |  |
| 6 | 0.7 | 0.7 | 2 | 10 dB | switch | 0.9806 | +0.0065 [+0.0055, +0.0075] | 0.9727 | +0.0100 [+0.0080, +0.0119] | frac 0.911; rule_loss -0.0000 [-0.0004, +0.0003]; decision_loss 0.0031; oracle_cf - cos +0.0030; frac_cf 0.814 |
| 6 | 0.7 | 0.7 | 2 | 10 dB | ods | 0.9797 | +0.0056 [+0.0045, +0.0067] | 0.9695 | +0.0068 [+0.0046, +0.0089] |  |
| 6 | 0.7 | 0.7 | 2 | 10 dB | csiqb | 0.9716 | -0.0025 [-0.0034, -0.0016] | 0.9628 | +0.0001 [-0.0016, +0.0018] |  |
| 6 | 0.7 | 0.7 | 2 | 10 dB | csiqw | 0.9807 | +0.0066 [+0.0055, +0.0077] | 0.9729 | +0.0101 [+0.0080, +0.0122] |  |
| 6 | 0.7 | 0.7 | 2 | 20 dB | fixed | 0.9954 | - | 0.9935 | - |  |
| 6 | 0.7 | 0.7 | 2 | 20 dB | csiq | 0.9959 | +0.0005 [+0.0003, +0.0007] | 0.9940 | +0.0005 [+0.0001, +0.0008] |  |
| 6 | 0.7 | 0.7 | 2 | 20 dB | csiqp | 0.9959 | +0.0005 [+0.0003, +0.0006] | 0.9940 | +0.0005 [+0.0002, +0.0008] |  |
| 6 | 0.7 | 0.7 | 2 | 20 dB | switch | 0.9959 | +0.0005 [+0.0004, +0.0006] | 0.9941 | +0.0007 [+0.0004, +0.0009] | frac 0.753; rule_loss -0.0000 [-0.0002, +0.0001]; decision_loss 0.0006; oracle_cf - cos +0.0006; frac_cf 0.767 |
| 6 | 0.7 | 0.7 | 2 | 20 dB | ods | 0.9957 | +0.0003 [+0.0001, +0.0005] | 0.9937 | +0.0002 [-0.0002, +0.0005] |  |
| 6 | 0.7 | 0.7 | 2 | 20 dB | csiqb | 0.9924 | -0.0029 [-0.0031, -0.0028] | 0.9896 | -0.0039 [-0.0042, -0.0036] |  |
| 6 | 0.7 | 0.7 | 2 | 20 dB | csiqw | 0.9958 | +0.0004 [+0.0003, +0.0006] | 0.9940 | +0.0005 [+0.0001, +0.0008] |  |
| 6 | 0.7 | 0.7 | 4 | 0 dB | fixed | 0.7810 | - | 0.6042 | - |  |
| 6 | 0.7 | 0.7 | 4 | 0 dB | csiq | 0.8708 | +0.0898 [+0.0864, +0.0932] | 0.7809 | +0.1767 [+0.1678, +0.1863] |  |
| 6 | 0.7 | 0.7 | 4 | 0 dB | csiqp | 0.8725 | +0.0914 [+0.0880, +0.0948] | 0.7834 | +0.1792 [+0.1704, +0.1888] |  |
| 6 | 0.7 | 0.7 | 4 | 0 dB | switch | 0.8708 | +0.0898 [+0.0864, +0.0932] | 0.7809 | +0.1767 [+0.1678, +0.1863] | frac 1.000; rule_loss +0.0000 [+0.0000, +0.0000]; decision_loss 0.0031; oracle_cf - cos +0.0028; frac_cf 0.926 |
| 6 | 0.7 | 0.7 | 4 | 0 dB | ods | 0.8678 | +0.0868 [+0.0833, +0.0904] | 0.7608 | +0.1566 [+0.1477, +0.1663] |  |
| 6 | 0.7 | 0.7 | 4 | 0 dB | csiqb | 0.8629 | +0.0818 [+0.0786, +0.0852] | 0.7822 | +0.1780 [+0.1695, +0.1876] |  |
| 6 | 0.7 | 0.7 | 4 | 0 dB | csiqw | 0.8692 | +0.0881 [+0.0848, +0.0915] | 0.7832 | +0.1790 [+0.1702, +0.1888] |  |
| 6 | 0.7 | 0.7 | 4 | 10 dB | fixed | 0.9314 | - | 0.8538 | - |  |
| 6 | 0.7 | 0.7 | 4 | 10 dB | csiq | 0.9666 | +0.0352 [+0.0334, +0.0368] | 0.9329 | +0.0792 [+0.0736, +0.0842] |  |
| 6 | 0.7 | 0.7 | 4 | 10 dB | csiqp | 0.9670 | +0.0356 [+0.0338, +0.0372] | 0.9342 | +0.0804 [+0.0749, +0.0856] |  |
| 6 | 0.7 | 0.7 | 4 | 10 dB | switch | 0.9666 | +0.0352 [+0.0334, +0.0368] | 0.9329 | +0.0792 [+0.0736, +0.0842] | frac 1.000; rule_loss +0.0000 [+0.0000, +0.0000]; decision_loss 0.0014; oracle_cf - cos +0.0014; frac_cf 0.931 |
| 6 | 0.7 | 0.7 | 4 | 10 dB | ods | 0.9664 | +0.0350 [+0.0332, +0.0366] | 0.9301 | +0.0763 [+0.0705, +0.0814] |  |
| 6 | 0.7 | 0.7 | 4 | 10 dB | csiqb | 0.9563 | +0.0249 [+0.0232, +0.0265] | 0.9236 | +0.0698 [+0.0645, +0.0749] |  |
| 6 | 0.7 | 0.7 | 4 | 10 dB | csiqw | 0.9669 | +0.0354 [+0.0336, +0.0371] | 0.9343 | +0.0805 [+0.0750, +0.0856] |  |
| 6 | 0.7 | 0.7 | 4 | 20 dB | fixed | 0.9853 | - | 0.9653 | - |  |
| 6 | 0.7 | 0.7 | 4 | 20 dB | csiq | 0.9930 | +0.0078 [+0.0071, +0.0084] | 0.9850 | +0.0196 [+0.0173, +0.0220] |  |
| 6 | 0.7 | 0.7 | 4 | 20 dB | csiqp | 0.9931 | +0.0079 [+0.0072, +0.0085] | 0.9853 | +0.0200 [+0.0177, +0.0224] |  |
| 6 | 0.7 | 0.7 | 4 | 20 dB | switch | 0.9930 | +0.0078 [+0.0071, +0.0084] | 0.9849 | +0.0196 [+0.0173, +0.0220] | frac 0.999; rule_loss +0.0000 [+0.0000, +0.0000]; decision_loss 0.0005; oracle_cf - cos +0.0005; frac_cf 0.909 |
| 6 | 0.7 | 0.7 | 4 | 20 dB | ods | 0.9930 | +0.0078 [+0.0071, +0.0084] | 0.9848 | +0.0195 [+0.0172, +0.0218] |  |
| 6 | 0.7 | 0.7 | 4 | 20 dB | csiqb | 0.9878 | +0.0025 [+0.0019, +0.0031] | 0.9769 | +0.0115 [+0.0093, +0.0138] |  |
| 6 | 0.7 | 0.7 | 4 | 20 dB | csiqw | 0.9930 | +0.0077 [+0.0071, +0.0084] | 0.9852 | +0.0199 [+0.0176, +0.0222] |  |
| 6 | 0.7 | 0.7 | 6 | 0 dB | fixed | 0.7446 | - | 0.4868 | - |  |
| 6 | 0.7 | 0.7 | 6 | 0 dB | csiq | 0.8421 | +0.0976 [+0.0945, +0.1006] | 0.6997 | +0.2129 [+0.2025, +0.2226] |  |
| 6 | 0.7 | 0.7 | 6 | 0 dB | csiqp | 0.8439 | +0.0994 [+0.0962, +0.1024] | 0.7034 | +0.2166 [+0.2062, +0.2265] |  |
| 6 | 0.7 | 0.7 | 6 | 0 dB | switch | 0.8421 | +0.0976 [+0.0945, +0.1006] | 0.6997 | +0.2129 [+0.2025, +0.2226] | frac 1.000; rule_loss +0.0000 [+0.0000, +0.0000]; decision_loss 0.0021; oracle_cf - cos +0.0018; frac_cf 0.955 |
| 6 | 0.7 | 0.7 | 6 | 0 dB | ods | 0.8394 | +0.0949 [+0.0918, +0.0979] | 0.6742 | +0.1873 [+0.1767, +0.1972] |  |
| 6 | 0.7 | 0.7 | 6 | 0 dB | csiqb | 0.8351 | +0.0905 [+0.0873, +0.0935] | 0.7050 | +0.2182 [+0.2080, +0.2278] |  |
| 6 | 0.7 | 0.7 | 6 | 0 dB | csiqw | 0.8402 | +0.0957 [+0.0925, +0.0987] | 0.7025 | +0.2157 [+0.2053, +0.2257] |  |
| 6 | 0.7 | 0.7 | 6 | 10 dB | fixed | 0.9129 | - | 0.7935 | - |  |
| 6 | 0.7 | 0.7 | 6 | 10 dB | csiq | 0.9507 | +0.0378 [+0.0363, +0.0393] | 0.8940 | +0.1005 [+0.0939, +0.1068] |  |
| 6 | 0.7 | 0.7 | 6 | 10 dB | csiqp | 0.9511 | +0.0382 [+0.0368, +0.0398] | 0.8954 | +0.1018 [+0.0952, +0.1082] |  |
| 6 | 0.7 | 0.7 | 6 | 10 dB | switch | 0.9507 | +0.0378 [+0.0363, +0.0393] | 0.8940 | +0.1005 [+0.0939, +0.1068] | frac 1.000; rule_loss +0.0000 [+0.0000, +0.0000]; decision_loss 0.0012; oracle_cf - cos +0.0011; frac_cf 0.946 |
| 6 | 0.7 | 0.7 | 6 | 10 dB | ods | 0.9508 | +0.0379 [+0.0364, +0.0394] | 0.8921 | +0.0986 [+0.0921, +0.1052] |  |
| 6 | 0.7 | 0.7 | 6 | 10 dB | csiqb | 0.9417 | +0.0288 [+0.0274, +0.0303] | 0.8885 | +0.0949 [+0.0886, +0.1012] |  |
| 6 | 0.7 | 0.7 | 6 | 10 dB | csiqw | 0.9510 | +0.0381 [+0.0366, +0.0396] | 0.8959 | +0.1023 [+0.0959, +0.1087] |  |
| 6 | 0.7 | 0.7 | 6 | 20 dB | fixed | 0.9704 | - | 0.9329 | - |  |
| 6 | 0.7 | 0.7 | 6 | 20 dB | csiq | 0.9798 | +0.0094 [+0.0087, +0.0100] | 0.9641 | +0.0313 [+0.0281, +0.0347] |  |
| 6 | 0.7 | 0.7 | 6 | 20 dB | csiqp | 0.9799 | +0.0095 [+0.0089, +0.0102] | 0.9649 | +0.0320 [+0.0288, +0.0354] |  |
| 6 | 0.7 | 0.7 | 6 | 20 dB | switch | 0.9798 | +0.0094 [+0.0087, +0.0100] | 0.9641 | +0.0313 [+0.0281, +0.0347] | frac 1.000; rule_loss +0.0000 [+0.0000, +0.0000]; decision_loss 0.0005; oracle_cf - cos +0.0004; frac_cf 0.907 |
| 6 | 0.7 | 0.7 | 6 | 20 dB | ods | 0.9799 | +0.0095 [+0.0089, +0.0102] | 0.9644 | +0.0315 [+0.0283, +0.0350] |  |
| 6 | 0.7 | 0.7 | 6 | 20 dB | csiqb | 0.9755 | +0.0050 [+0.0044, +0.0057] | 0.9571 | +0.0242 [+0.0211, +0.0275] |  |
| 6 | 0.7 | 0.7 | 6 | 20 dB | csiqw | 0.9798 | +0.0094 [+0.0088, +0.0100] | 0.9646 | +0.0317 [+0.0286, +0.0351] |  |
| 6 | 0.7 | 0.5 | 1 | 0 dB | fixed | 0.9266 | - | 0.9266 | - |  |
| 6 | 0.7 | 0.5 | 1 | 0 dB | csiq | 0.9157 | -0.0109 [-0.0145, -0.0076] | 0.9157 | -0.0109 [-0.0145, -0.0076] |  |
| 6 | 0.7 | 0.5 | 1 | 0 dB | csiqp | 0.9178 | -0.0088 [-0.0123, -0.0056] | 0.9178 | -0.0088 [-0.0123, -0.0056] |  |
| 6 | 0.7 | 0.5 | 1 | 0 dB | switch | 0.9263 | -0.0003 [-0.0020, +0.0011] | 0.9263 | -0.0003 [-0.0020, +0.0011] | frac 0.182; rule_loss +0.0003 [-0.0011, +0.0020]; decision_loss 0.0145; oracle_cf - cos +0.0141; frac_cf 0.552 |
| 6 | 0.7 | 0.5 | 1 | 0 dB | ods | 0.9157 | -0.0109 [-0.0145, -0.0076] | 0.9157 | -0.0109 [-0.0145, -0.0076] |  |
| 6 | 0.7 | 0.5 | 1 | 0 dB | csiqb | 0.9100 | -0.0166 [-0.0195, -0.0140] | 0.9100 | -0.0166 [-0.0195, -0.0140] |  |
| 6 | 0.7 | 0.5 | 1 | 0 dB | csiqw | 0.9170 | -0.0097 [-0.0132, -0.0064] | 0.9170 | -0.0097 [-0.0132, -0.0064] |  |
| 6 | 0.7 | 0.5 | 1 | 10 dB | fixed | 0.9885 | - | 0.9885 | - |  |
| 6 | 0.7 | 0.5 | 1 | 10 dB | csiq | 0.9820 | -0.0065 [-0.0075, -0.0056] | 0.9820 | -0.0065 [-0.0075, -0.0056] |  |
| 6 | 0.7 | 0.5 | 1 | 10 dB | csiqp | 0.9823 | -0.0062 [-0.0072, -0.0053] | 0.9823 | -0.0062 [-0.0072, -0.0053] |  |
| 6 | 0.7 | 0.5 | 1 | 10 dB | switch | 0.9885 | -0.0000 [-0.0001, +0.0001] | 0.9885 | -0.0000 [-0.0001, +0.0001] | frac 0.029; rule_loss +0.0000 [-0.0001, +0.0001]; decision_loss 0.0014; oracle_cf - cos +0.0014; frac_cf 0.425 |
| 6 | 0.7 | 0.5 | 1 | 10 dB | ods | 0.9820 | -0.0065 [-0.0075, -0.0056] | 0.9820 | -0.0065 [-0.0075, -0.0056] |  |
| 6 | 0.7 | 0.5 | 1 | 10 dB | csiqb | 0.9750 | -0.0135 [-0.0143, -0.0127] | 0.9750 | -0.0135 [-0.0143, -0.0127] |  |
| 6 | 0.7 | 0.5 | 1 | 10 dB | csiqw | 0.9823 | -0.0062 [-0.0071, -0.0053] | 0.9823 | -0.0062 [-0.0071, -0.0053] |  |
| 6 | 0.7 | 0.5 | 1 | 20 dB | fixed | 0.9973 | - | 0.9973 | - |  |
| 6 | 0.7 | 0.5 | 1 | 20 dB | csiq | 0.9954 | -0.0019 [-0.0024, -0.0015] | 0.9954 | -0.0019 [-0.0024, -0.0015] |  |
| 6 | 0.7 | 0.5 | 1 | 20 dB | csiqp | 0.9955 | -0.0018 [-0.0023, -0.0014] | 0.9955 | -0.0018 [-0.0023, -0.0014] |  |
| 6 | 0.7 | 0.5 | 1 | 20 dB | switch | 0.9973 | +0.0000 [-0.0000, +0.0000] | 0.9973 | +0.0000 [-0.0000, +0.0000] | frac 0.008; rule_loss -0.0000 [-0.0000, +0.0000]; decision_loss 0.0001; oracle_cf - cos +0.0001; frac_cf 0.393 |
| 6 | 0.7 | 0.5 | 1 | 20 dB | ods | 0.9954 | -0.0019 [-0.0024, -0.0015] | 0.9954 | -0.0019 [-0.0024, -0.0015] |  |
| 6 | 0.7 | 0.5 | 1 | 20 dB | csiqb | 0.9927 | -0.0046 [-0.0050, -0.0042] | 0.9927 | -0.0046 [-0.0050, -0.0042] |  |
| 6 | 0.7 | 0.5 | 1 | 20 dB | csiqw | 0.9955 | -0.0018 [-0.0023, -0.0014] | 0.9955 | -0.0018 [-0.0023, -0.0014] |  |
| 6 | 0.7 | 0.5 | 2 | 0 dB | fixed | 0.8761 | - | 0.8317 | - |  |
| 6 | 0.7 | 0.5 | 2 | 0 dB | csiq | 0.8795 | +0.0034 [+0.0001, +0.0067] | 0.8327 | +0.0010 [-0.0047, +0.0067] |  |
| 6 | 0.7 | 0.5 | 2 | 0 dB | csiqp | 0.8832 | +0.0070 [+0.0039, +0.0103] | 0.8378 | +0.0061 [+0.0004, +0.0117] |  |
| 6 | 0.7 | 0.5 | 2 | 0 dB | switch | 0.8815 | +0.0054 [+0.0027, +0.0080] | 0.8367 | +0.0050 [+0.0003, +0.0098] | frac 0.579; rule_loss -0.0020 [-0.0038, -0.0000]; decision_loss 0.0219; oracle_cf - cos +0.0214; frac_cf 0.599 |
| 6 | 0.7 | 0.5 | 2 | 0 dB | ods | 0.8780 | +0.0018 [-0.0015, +0.0053] | 0.8252 | -0.0065 [-0.0128, -0.0003] |  |
| 6 | 0.7 | 0.5 | 2 | 0 dB | csiqb | 0.8759 | -0.0002 [-0.0030, +0.0028] | 0.8356 | +0.0040 [-0.0013, +0.0093] |  |
| 6 | 0.7 | 0.5 | 2 | 0 dB | csiqw | 0.8813 | +0.0052 [+0.0021, +0.0084] | 0.8369 | +0.0052 [-0.0004, +0.0109] |  |
| 6 | 0.7 | 0.5 | 2 | 10 dB | fixed | 0.9742 | - | 0.9629 | - |  |
| 6 | 0.7 | 0.5 | 2 | 10 dB | csiq | 0.9700 | -0.0042 [-0.0056, -0.0029] | 0.9562 | -0.0067 [-0.0093, -0.0042] |  |
| 6 | 0.7 | 0.5 | 2 | 10 dB | csiqp | 0.9707 | -0.0035 [-0.0048, -0.0022] | 0.9574 | -0.0055 [-0.0081, -0.0031] |  |
| 6 | 0.7 | 0.5 | 2 | 10 dB | switch | 0.9734 | -0.0008 [-0.0015, -0.0000] | 0.9616 | -0.0013 [-0.0028, +0.0001] | frac 0.243; rule_loss +0.0008 [+0.0000, +0.0015]; decision_loss 0.0062; oracle_cf - cos +0.0061; frac_cf 0.531 |
| 6 | 0.7 | 0.5 | 2 | 10 dB | ods | 0.9691 | -0.0051 [-0.0065, -0.0037] | 0.9531 | -0.0098 [-0.0125, -0.0072] |  |
| 6 | 0.7 | 0.5 | 2 | 10 dB | csiqb | 0.9619 | -0.0122 [-0.0133, -0.0111] | 0.9489 | -0.0140 [-0.0161, -0.0121] |  |
| 6 | 0.7 | 0.5 | 2 | 10 dB | csiqw | 0.9707 | -0.0034 [-0.0048, -0.0022] | 0.9577 | -0.0052 [-0.0078, -0.0029] |  |
| 6 | 0.7 | 0.5 | 2 | 20 dB | fixed | 0.9953 | - | 0.9934 | - |  |
| 6 | 0.7 | 0.5 | 2 | 20 dB | csiq | 0.9936 | -0.0017 [-0.0022, -0.0013] | 0.9902 | -0.0031 [-0.0041, -0.0024] |  |
| 6 | 0.7 | 0.5 | 2 | 20 dB | csiqp | 0.9937 | -0.0017 [-0.0021, -0.0013] | 0.9904 | -0.0030 [-0.0039, -0.0022] |  |
| 6 | 0.7 | 0.5 | 2 | 20 dB | switch | 0.9953 | -0.0001 [-0.0001, -0.0000] | 0.9933 | -0.0001 [-0.0002, -0.0000] | frac 0.086; rule_loss +0.0001 [+0.0000, +0.0001]; decision_loss 0.0007; oracle_cf - cos +0.0007; frac_cf 0.448 |
| 6 | 0.7 | 0.5 | 2 | 20 dB | ods | 0.9934 | -0.0020 [-0.0024, -0.0015] | 0.9897 | -0.0037 [-0.0046, -0.0029] |  |
| 6 | 0.7 | 0.5 | 2 | 20 dB | csiqb | 0.9898 | -0.0055 [-0.0059, -0.0052] | 0.9856 | -0.0078 [-0.0085, -0.0072] |  |
| 6 | 0.7 | 0.5 | 2 | 20 dB | csiqw | 0.9937 | -0.0017 [-0.0021, -0.0013] | 0.9904 | -0.0030 [-0.0038, -0.0022] |  |
| 6 | 0.7 | 0.5 | 4 | 0 dB | fixed | 0.7799 | - | 0.6049 | - |  |
| 6 | 0.7 | 0.5 | 4 | 0 dB | csiq | 0.8290 | +0.0491 [+0.0452, +0.0528] | 0.6966 | +0.0917 [+0.0811, +0.1016] |  |
| 6 | 0.7 | 0.5 | 4 | 0 dB | csiqp | 0.8341 | +0.0543 [+0.0504, +0.0579] | 0.7060 | +0.1011 [+0.0905, +0.1111] |  |
| 6 | 0.7 | 0.5 | 4 | 0 dB | switch | 0.8288 | +0.0489 [+0.0449, +0.0526] | 0.6960 | +0.0911 [+0.0803, +0.1010] | frac 0.997; rule_loss +0.0002 [+0.0000, +0.0005]; decision_loss 0.0139; oracle_cf - cos +0.0130; frac_cf 0.757 |
| 6 | 0.7 | 0.5 | 4 | 0 dB | ods | 0.8271 | +0.0472 [+0.0433, +0.0510] | 0.6802 | +0.0753 [+0.0645, +0.0857] |  |
| 6 | 0.7 | 0.5 | 4 | 0 dB | csiqb | 0.8275 | +0.0476 [+0.0439, +0.0511] | 0.7110 | +0.1062 [+0.0959, +0.1159] |  |
| 6 | 0.7 | 0.5 | 4 | 0 dB | csiqw | 0.8311 | +0.0512 [+0.0475, +0.0550] | 0.7043 | +0.0994 [+0.0886, +0.1094] |  |
| 6 | 0.7 | 0.5 | 4 | 10 dB | fixed | 0.9326 | - | 0.8581 | - |  |
| 6 | 0.7 | 0.5 | 4 | 10 dB | csiq | 0.9508 | +0.0182 [+0.0164, +0.0202] | 0.8956 | +0.0375 [+0.0314, +0.0435] |  |
| 6 | 0.7 | 0.5 | 4 | 10 dB | csiqp | 0.9523 | +0.0197 [+0.0179, +0.0216] | 0.9003 | +0.0423 [+0.0363, +0.0481] |  |
| 6 | 0.7 | 0.5 | 4 | 10 dB | switch | 0.9507 | +0.0181 [+0.0163, +0.0201] | 0.8954 | +0.0373 [+0.0313, +0.0433] | frac 0.992; rule_loss +0.0001 [-0.0000, +0.0003]; decision_loss 0.0066; oracle_cf - cos +0.0065; frac_cf 0.746 |
| 6 | 0.7 | 0.5 | 4 | 10 dB | ods | 0.9503 | +0.0177 [+0.0158, +0.0197] | 0.8921 | +0.0340 [+0.0278, +0.0400] |  |
| 6 | 0.7 | 0.5 | 4 | 10 dB | csiqb | 0.9432 | +0.0106 [+0.0090, +0.0124] | 0.8959 | +0.0378 [+0.0324, +0.0433] |  |
| 6 | 0.7 | 0.5 | 4 | 10 dB | csiqw | 0.9522 | +0.0196 [+0.0178, +0.0215] | 0.8998 | +0.0418 [+0.0359, +0.0476] |  |
| 6 | 0.7 | 0.5 | 4 | 20 dB | fixed | 0.9854 | - | 0.9656 | - |  |
| 6 | 0.7 | 0.5 | 4 | 20 dB | csiq | 0.9892 | +0.0039 [+0.0032, +0.0046] | 0.9746 | +0.0091 [+0.0064, +0.0116] |  |
| 6 | 0.7 | 0.5 | 4 | 20 dB | csiqp | 0.9896 | +0.0042 [+0.0035, +0.0049] | 0.9758 | +0.0102 [+0.0076, +0.0127] |  |
| 6 | 0.7 | 0.5 | 4 | 20 dB | switch | 0.9890 | +0.0036 [+0.0029, +0.0044] | 0.9740 | +0.0085 [+0.0058, +0.0109] | frac 0.929; rule_loss +0.0002 [+0.0001, +0.0004]; decision_loss 0.0024; oracle_cf - cos +0.0024; frac_cf 0.715 |
| 6 | 0.7 | 0.5 | 4 | 20 dB | ods | 0.9895 | +0.0041 [+0.0034, +0.0048] | 0.9753 | +0.0097 [+0.0071, +0.0121] |  |
| 6 | 0.7 | 0.5 | 4 | 20 dB | csiqb | 0.9841 | -0.0013 [-0.0020, -0.0006] | 0.9681 | +0.0026 [+0.0002, +0.0049] |  |
| 6 | 0.7 | 0.5 | 4 | 20 dB | csiqw | 0.9896 | +0.0042 [+0.0035, +0.0049] | 0.9759 | +0.0104 [+0.0078, +0.0128] |  |
| 6 | 0.7 | 0.5 | 6 | 0 dB | fixed | 0.7439 | - | 0.4879 | - |  |
| 6 | 0.7 | 0.5 | 6 | 0 dB | csiq | 0.7945 | +0.0506 [+0.0469, +0.0539] | 0.5853 | +0.0974 [+0.0865, +0.1081] |  |
| 6 | 0.7 | 0.5 | 6 | 0 dB | csiqp | 0.7998 | +0.0559 [+0.0524, +0.0591] | 0.5955 | +0.1076 [+0.0966, +0.1183] |  |
| 6 | 0.7 | 0.5 | 6 | 0 dB | switch | 0.7945 | +0.0506 [+0.0469, +0.0539] | 0.5853 | +0.0974 [+0.0865, +0.1081] | frac 1.000; rule_loss +0.0000 [+0.0000, +0.0000]; decision_loss 0.0110; oracle_cf - cos +0.0099; frac_cf 0.775 |
| 6 | 0.7 | 0.5 | 6 | 0 dB | ods | 0.7931 | +0.0492 [+0.0458, +0.0525] | 0.5693 | +0.0814 [+0.0703, +0.0919] |  |
| 6 | 0.7 | 0.5 | 6 | 0 dB | csiqb | 0.7939 | +0.0500 [+0.0465, +0.0532] | 0.6046 | +0.1167 [+0.1059, +0.1274] |  |
| 6 | 0.7 | 0.5 | 6 | 0 dB | csiqw | 0.7963 | +0.0524 [+0.0488, +0.0557] | 0.5921 | +0.1042 [+0.0931, +0.1151] |  |
| 6 | 0.7 | 0.5 | 6 | 10 dB | fixed | 0.9126 | - | 0.7923 | - |  |
| 6 | 0.7 | 0.5 | 6 | 10 dB | csiq | 0.9325 | +0.0199 [+0.0182, +0.0217] | 0.8395 | +0.0473 [+0.0398, +0.0546] |  |
| 6 | 0.7 | 0.5 | 6 | 10 dB | csiqp | 0.9340 | +0.0214 [+0.0198, +0.0232] | 0.8445 | +0.0522 [+0.0450, +0.0595] |  |
| 6 | 0.7 | 0.5 | 6 | 10 dB | switch | 0.9325 | +0.0199 [+0.0182, +0.0217] | 0.8395 | +0.0473 [+0.0398, +0.0545] | frac 0.998; rule_loss +0.0000 [-0.0000, +0.0001]; decision_loss 0.0059; oracle_cf - cos +0.0058; frac_cf 0.751 |
| 6 | 0.7 | 0.5 | 6 | 10 dB | ods | 0.9323 | +0.0197 [+0.0179, +0.0216] | 0.8368 | +0.0446 [+0.0372, +0.0524] |  |
| 6 | 0.7 | 0.5 | 6 | 10 dB | csiqb | 0.9263 | +0.0137 [+0.0121, +0.0154] | 0.8467 | +0.0544 [+0.0477, +0.0615] |  |
| 6 | 0.7 | 0.5 | 6 | 10 dB | csiqw | 0.9340 | +0.0213 [+0.0197, +0.0232] | 0.8447 | +0.0524 [+0.0451, +0.0596] |  |
| 6 | 0.7 | 0.5 | 6 | 20 dB | fixed | 0.9707 | - | 0.9342 | - |  |
| 6 | 0.7 | 0.5 | 6 | 20 dB | csiq | 0.9752 | +0.0045 [+0.0038, +0.0052] | 0.9475 | +0.0133 [+0.0099, +0.0171] |  |
| 6 | 0.7 | 0.5 | 6 | 20 dB | csiqp | 0.9756 | +0.0048 [+0.0042, +0.0056] | 0.9494 | +0.0152 [+0.0119, +0.0190] |  |
| 6 | 0.7 | 0.5 | 6 | 20 dB | switch | 0.9752 | +0.0045 [+0.0038, +0.0052] | 0.9479 | +0.0137 [+0.0105, +0.0176] | frac 0.959; rule_loss -0.0000 [-0.0002, +0.0001]; decision_loss 0.0024; oracle_cf - cos +0.0024; frac_cf 0.720 |
| 6 | 0.7 | 0.5 | 6 | 20 dB | ods | 0.9755 | +0.0048 [+0.0042, +0.0056] | 0.9489 | +0.0147 [+0.0113, +0.0186] |  |
| 6 | 0.7 | 0.5 | 6 | 20 dB | csiqb | 0.9711 | +0.0004 [-0.0003, +0.0010] | 0.9439 | +0.0097 [+0.0067, +0.0132] |  |
| 6 | 0.7 | 0.5 | 6 | 20 dB | csiqw | 0.9756 | +0.0049 [+0.0042, +0.0056] | 0.9496 | +0.0154 [+0.0121, +0.0191] |  |
| 6 | 0.7 | 0.3 | 1 | 0 dB | fixed | 0.9266 | - | 0.9266 | - |  |
| 6 | 0.7 | 0.3 | 1 | 0 dB | csiq | 0.8943 | -0.0322 [-0.0364, -0.0284] | 0.8943 | -0.0322 [-0.0364, -0.0284] |  |
| 6 | 0.7 | 0.3 | 1 | 0 dB | csiqp | 0.8987 | -0.0279 [-0.0318, -0.0243] | 0.8987 | -0.0279 [-0.0318, -0.0243] |  |
| 6 | 0.7 | 0.3 | 1 | 0 dB | switch | 0.9265 | -0.0001 [-0.0003, +0.0001] | 0.9265 | -0.0001 [-0.0003, +0.0001] | frac 0.007; rule_loss +0.0001 [-0.0001, +0.0003]; decision_loss 0.0094; oracle_cf - cos +0.0090; frac_cf 0.384 |
| 6 | 0.7 | 0.3 | 1 | 0 dB | ods | 0.8943 | -0.0322 [-0.0364, -0.0284] | 0.8943 | -0.0322 [-0.0364, -0.0284] |  |
| 6 | 0.7 | 0.3 | 1 | 0 dB | csiqb | 0.8921 | -0.0344 [-0.0379, -0.0312] | 0.8921 | -0.0344 [-0.0379, -0.0312] |  |
| 6 | 0.7 | 0.3 | 1 | 0 dB | csiqw | 0.8971 | -0.0295 [-0.0334, -0.0258] | 0.8971 | -0.0295 [-0.0334, -0.0258] |  |
| 6 | 0.7 | 0.3 | 1 | 10 dB | fixed | 0.9885 | - | 0.9885 | - |  |
| 6 | 0.7 | 0.3 | 1 | 10 dB | csiq | 0.9760 | -0.0124 [-0.0140, -0.0110] | 0.9760 | -0.0124 [-0.0140, -0.0110] |  |
| 6 | 0.7 | 0.3 | 1 | 10 dB | csiqp | 0.9767 | -0.0118 [-0.0133, -0.0104] | 0.9767 | -0.0118 [-0.0133, -0.0104] |  |
| 6 | 0.7 | 0.3 | 1 | 10 dB | switch | 0.9885 | +0.0000 [+0.0000, +0.0000] | 0.9885 | +0.0000 [+0.0000, +0.0000] | frac 0.000; rule_loss +0.0000 [+0.0000, +0.0000]; decision_loss 0.0008; oracle_cf - cos +0.0008; frac_cf 0.258 |
| 6 | 0.7 | 0.3 | 1 | 10 dB | ods | 0.9760 | -0.0124 [-0.0140, -0.0110] | 0.9760 | -0.0124 [-0.0140, -0.0110] |  |
| 6 | 0.7 | 0.3 | 1 | 10 dB | csiqb | 0.9695 | -0.0190 [-0.0203, -0.0177] | 0.9695 | -0.0190 [-0.0203, -0.0177] |  |
| 6 | 0.7 | 0.3 | 1 | 10 dB | csiqw | 0.9767 | -0.0118 [-0.0133, -0.0104] | 0.9767 | -0.0118 [-0.0133, -0.0104] |  |
| 6 | 0.7 | 0.3 | 1 | 20 dB | fixed | 0.9973 | - | 0.9973 | - |  |
| 6 | 0.7 | 0.3 | 1 | 20 dB | csiq | 0.9949 | -0.0024 [-0.0030, -0.0020] | 0.9949 | -0.0024 [-0.0030, -0.0020] |  |
| 6 | 0.7 | 0.3 | 1 | 20 dB | csiqp | 0.9950 | -0.0023 [-0.0029, -0.0020] | 0.9950 | -0.0023 [-0.0029, -0.0020] |  |
| 6 | 0.7 | 0.3 | 1 | 20 dB | switch | 0.9973 | +0.0000 [+0.0000, +0.0000] | 0.9973 | +0.0000 [+0.0000, +0.0000] | frac 0.000; rule_loss +0.0000 [+0.0000, +0.0000]; decision_loss 0.0001; oracle_cf - cos +0.0001; frac_cf 0.235 |
| 6 | 0.7 | 0.3 | 1 | 20 dB | ods | 0.9949 | -0.0024 [-0.0030, -0.0020] | 0.9949 | -0.0024 [-0.0030, -0.0020] |  |
| 6 | 0.7 | 0.3 | 1 | 20 dB | csiqb | 0.9919 | -0.0054 [-0.0058, -0.0050] | 0.9919 | -0.0054 [-0.0058, -0.0050] |  |
| 6 | 0.7 | 0.3 | 1 | 20 dB | csiqw | 0.9950 | -0.0023 [-0.0029, -0.0020] | 0.9950 | -0.0023 [-0.0029, -0.0020] |  |
| 6 | 0.7 | 0.3 | 2 | 0 dB | fixed | 0.8757 | - | 0.8308 | - |  |
| 6 | 0.7 | 0.3 | 2 | 0 dB | csiq | 0.8514 | -0.0243 [-0.0282, -0.0202] | 0.7932 | -0.0375 [-0.0442, -0.0302] |  |
| 6 | 0.7 | 0.3 | 2 | 0 dB | csiqp | 0.8585 | -0.0172 [-0.0211, -0.0133] | 0.8033 | -0.0275 [-0.0343, -0.0204] |  |
| 6 | 0.7 | 0.3 | 2 | 0 dB | switch | 0.8746 | -0.0011 [-0.0021, -0.0001] | 0.8295 | -0.0012 [-0.0028, +0.0005] | frac 0.052; rule_loss +0.0011 [+0.0001, +0.0021]; decision_loss 0.0193; oracle_cf - cos +0.0186; frac_cf 0.416 |
| 6 | 0.7 | 0.3 | 2 | 0 dB | ods | 0.8509 | -0.0247 [-0.0287, -0.0206] | 0.7873 | -0.0435 [-0.0505, -0.0365] |  |
| 6 | 0.7 | 0.3 | 2 | 0 dB | csiqb | 0.8533 | -0.0224 [-0.0258, -0.0187] | 0.8046 | -0.0262 [-0.0321, -0.0200] |  |
| 6 | 0.7 | 0.3 | 2 | 0 dB | csiqw | 0.8557 | -0.0199 [-0.0238, -0.0160] | 0.8006 | -0.0301 [-0.0368, -0.0232] |  |
| 6 | 0.7 | 0.3 | 2 | 10 dB | fixed | 0.9741 | - | 0.9628 | - |  |
| 6 | 0.7 | 0.3 | 2 | 10 dB | csiq | 0.9610 | -0.0131 [-0.0148, -0.0115] | 0.9422 | -0.0207 [-0.0238, -0.0178] |  |
| 6 | 0.7 | 0.3 | 2 | 10 dB | csiqp | 0.9625 | -0.0116 [-0.0132, -0.0101] | 0.9449 | -0.0179 [-0.0210, -0.0151] |  |
| 6 | 0.7 | 0.3 | 2 | 10 dB | switch | 0.9740 | -0.0001 [-0.0002, -0.0000] | 0.9626 | -0.0002 [-0.0004, -0.0001] | frac 0.009; rule_loss +0.0001 [+0.0000, +0.0002]; decision_loss 0.0036; oracle_cf - cos +0.0035; frac_cf 0.353 |
| 6 | 0.7 | 0.3 | 2 | 10 dB | ods | 0.9604 | -0.0137 [-0.0154, -0.0122] | 0.9395 | -0.0234 [-0.0267, -0.0206] |  |
| 6 | 0.7 | 0.3 | 2 | 10 dB | csiqb | 0.9542 | -0.0199 [-0.0213, -0.0186] | 0.9376 | -0.0252 [-0.0277, -0.0229] |  |
| 6 | 0.7 | 0.3 | 2 | 10 dB | csiqw | 0.9624 | -0.0117 [-0.0133, -0.0102] | 0.9447 | -0.0181 [-0.0211, -0.0153] |  |
| 6 | 0.7 | 0.3 | 2 | 20 dB | fixed | 0.9953 | - | 0.9933 | - |  |
| 6 | 0.7 | 0.3 | 2 | 20 dB | csiq | 0.9919 | -0.0033 [-0.0039, -0.0029] | 0.9877 | -0.0056 [-0.0065, -0.0047] |  |
| 6 | 0.7 | 0.3 | 2 | 20 dB | csiqp | 0.9922 | -0.0031 [-0.0036, -0.0027] | 0.9882 | -0.0051 [-0.0059, -0.0043] |  |
| 6 | 0.7 | 0.3 | 2 | 20 dB | switch | 0.9953 | -0.0000 [-0.0000, +0.0000] | 0.9933 | +0.0000 [-0.0000, +0.0000] | frac 0.002; rule_loss +0.0000 [-0.0000, +0.0000]; decision_loss 0.0005; oracle_cf - cos +0.0005; frac_cf 0.297 |
| 6 | 0.7 | 0.3 | 2 | 20 dB | ods | 0.9918 | -0.0035 [-0.0040, -0.0030] | 0.9873 | -0.0060 [-0.0071, -0.0051] |  |
| 6 | 0.7 | 0.3 | 2 | 20 dB | csiqb | 0.9881 | -0.0072 [-0.0076, -0.0068] | 0.9832 | -0.0101 [-0.0108, -0.0094] |  |
| 6 | 0.7 | 0.3 | 2 | 20 dB | csiqw | 0.9921 | -0.0031 [-0.0036, -0.0027] | 0.9881 | -0.0051 [-0.0060, -0.0044] |  |
| 6 | 0.7 | 0.3 | 4 | 0 dB | fixed | 0.7800 | - | 0.6074 | - |  |
| 6 | 0.7 | 0.3 | 4 | 0 dB | csiq | 0.7951 | +0.0151 [+0.0111, +0.0193] | 0.6336 | +0.0262 [+0.0158, +0.0364] |  |
| 6 | 0.7 | 0.3 | 4 | 0 dB | csiqp | 0.8041 | +0.0241 [+0.0203, +0.0283] | 0.6500 | +0.0425 [+0.0323, +0.0530] |  |
| 6 | 0.7 | 0.3 | 4 | 0 dB | switch | 0.7912 | +0.0112 [+0.0076, +0.0150] | 0.6257 | +0.0183 [+0.0087, +0.0275] | frac 0.793; rule_loss +0.0038 [+0.0022, +0.0055]; decision_loss 0.0308; oracle_cf - cos +0.0298; frac_cf 0.586 |
| 6 | 0.7 | 0.3 | 4 | 0 dB | ods | 0.7944 | +0.0144 [+0.0103, +0.0184] | 0.6227 | +0.0152 [+0.0043, +0.0257] |  |
| 6 | 0.7 | 0.3 | 4 | 0 dB | csiqb | 0.7987 | +0.0187 [+0.0150, +0.0229] | 0.6569 | +0.0494 [+0.0394, +0.0595] |  |
| 6 | 0.7 | 0.3 | 4 | 0 dB | csiqw | 0.8001 | +0.0201 [+0.0163, +0.0243] | 0.6447 | +0.0373 [+0.0272, +0.0471] |  |
| 6 | 0.7 | 0.3 | 4 | 10 dB | fixed | 0.9337 | - | 0.8610 | - |  |
| 6 | 0.7 | 0.3 | 4 | 10 dB | csiq | 0.9385 | +0.0048 [+0.0028, +0.0067] | 0.8691 | +0.0081 [+0.0020, +0.0143] |  |
| 6 | 0.7 | 0.3 | 4 | 10 dB | csiqp | 0.9416 | +0.0079 [+0.0061, +0.0099] | 0.8787 | +0.0177 [+0.0119, +0.0237] |  |
| 6 | 0.7 | 0.3 | 4 | 10 dB | switch | 0.9366 | +0.0029 [+0.0013, +0.0045] | 0.8646 | +0.0036 [-0.0014, +0.0087] | frac 0.734; rule_loss +0.0018 [+0.0009, +0.0029]; decision_loss 0.0145; oracle_cf - cos +0.0144; frac_cf 0.567 |
| 6 | 0.7 | 0.3 | 4 | 10 dB | ods | 0.9378 | +0.0041 [+0.0021, +0.0061] | 0.8656 | +0.0046 [-0.0017, +0.0109] |  |
| 6 | 0.7 | 0.3 | 4 | 10 dB | csiqb | 0.9333 | -0.0004 [-0.0021, +0.0013] | 0.8762 | +0.0151 [+0.0098, +0.0209] |  |
| 6 | 0.7 | 0.3 | 4 | 10 dB | csiqw | 0.9410 | +0.0073 [+0.0054, +0.0092] | 0.8756 | +0.0146 [+0.0087, +0.0208] |  |
| 6 | 0.7 | 0.3 | 4 | 20 dB | fixed | 0.9850 | - | 0.9643 | - |  |
| 6 | 0.7 | 0.3 | 4 | 20 dB | csiq | 0.9863 | +0.0012 [+0.0005, +0.0020] | 0.9667 | +0.0024 [-0.0005, +0.0052] |  |
| 6 | 0.7 | 0.3 | 4 | 20 dB | csiqp | 0.9869 | +0.0019 [+0.0011, +0.0027] | 0.9690 | +0.0047 [+0.0019, +0.0074] |  |
| 6 | 0.7 | 0.3 | 4 | 20 dB | switch | 0.9858 | +0.0008 [+0.0001, +0.0015] | 0.9660 | +0.0017 [-0.0009, +0.0043] | frac 0.614; rule_loss +0.0004 [+0.0000, +0.0009]; decision_loss 0.0044; oracle_cf - cos +0.0043; frac_cf 0.572 |
| 6 | 0.7 | 0.3 | 4 | 20 dB | ods | 0.9864 | +0.0014 [+0.0006, +0.0022] | 0.9671 | +0.0028 [-0.0001, +0.0057] |  |
| 6 | 0.7 | 0.3 | 4 | 20 dB | csiqb | 0.9815 | -0.0035 [-0.0042, -0.0028] | 0.9624 | -0.0019 [-0.0045, +0.0006] |  |
| 6 | 0.7 | 0.3 | 4 | 20 dB | csiqw | 0.9870 | +0.0020 [+0.0012, +0.0027] | 0.9691 | +0.0048 [+0.0020, +0.0075] |  |
| 6 | 0.7 | 0.3 | 6 | 0 dB | fixed | 0.7432 | - | 0.4867 | - |  |
| 6 | 0.7 | 0.3 | 6 | 0 dB | csiq | 0.7572 | +0.0140 [+0.0101, +0.0178] | 0.5152 | +0.0285 [+0.0167, +0.0396] |  |
| 6 | 0.7 | 0.3 | 6 | 0 dB | csiqp | 0.7665 | +0.0232 [+0.0194, +0.0269] | 0.5326 | +0.0459 [+0.0342, +0.0575] |  |
| 6 | 0.7 | 0.3 | 6 | 0 dB | switch | 0.7565 | +0.0133 [+0.0098, +0.0166] | 0.5148 | +0.0280 [+0.0176, +0.0383] | frac 0.847; rule_loss +0.0007 [-0.0009, +0.0022]; decision_loss 0.0267; oracle_cf - cos +0.0246; frac_cf 0.561 |
| 6 | 0.7 | 0.3 | 6 | 0 dB | ods | 0.7564 | +0.0132 [+0.0097, +0.0170] | 0.5050 | +0.0183 [+0.0069, +0.0302] |  |
| 6 | 0.7 | 0.3 | 6 | 0 dB | csiqb | 0.7614 | +0.0181 [+0.0144, +0.0217] | 0.5402 | +0.0535 [+0.0418, +0.0648] |  |
| 6 | 0.7 | 0.3 | 6 | 0 dB | csiqw | 0.7618 | +0.0186 [+0.0148, +0.0223] | 0.5240 | +0.0373 [+0.0255, +0.0489] |  |
| 6 | 0.7 | 0.3 | 6 | 10 dB | fixed | 0.9122 | - | 0.7909 | - |  |
| 6 | 0.7 | 0.3 | 6 | 10 dB | csiq | 0.9176 | +0.0054 [+0.0036, +0.0074] | 0.7999 | +0.0090 [+0.0012, +0.0176] |  |
| 6 | 0.7 | 0.3 | 6 | 10 dB | csiqp | 0.9208 | +0.0086 [+0.0067, +0.0104] | 0.8094 | +0.0185 [+0.0108, +0.0267] |  |
| 6 | 0.7 | 0.3 | 6 | 10 dB | switch | 0.9166 | +0.0044 [+0.0027, +0.0062] | 0.7991 | +0.0082 [+0.0011, +0.0159] | frac 0.782; rule_loss +0.0011 [+0.0002, +0.0018]; decision_loss 0.0140; oracle_cf - cos +0.0136; frac_cf 0.568 |
| 6 | 0.7 | 0.3 | 6 | 10 dB | ods | 0.9178 | +0.0056 [+0.0038, +0.0076] | 0.7999 | +0.0090 [+0.0016, +0.0176] |  |
| 6 | 0.7 | 0.3 | 6 | 10 dB | csiqb | 0.9138 | +0.0016 [-0.0001, +0.0034] | 0.8152 | +0.0243 [+0.0166, +0.0321] |  |
| 6 | 0.7 | 0.3 | 6 | 10 dB | csiqw | 0.9202 | +0.0080 [+0.0061, +0.0099] | 0.8075 | +0.0166 [+0.0090, +0.0250] |  |
| 6 | 0.7 | 0.3 | 6 | 20 dB | fixed | 0.9711 | - | 0.9366 | - |  |
| 6 | 0.7 | 0.3 | 6 | 20 dB | csiq | 0.9720 | +0.0009 [+0.0001, +0.0016] | 0.9370 | +0.0004 [-0.0035, +0.0043] |  |
| 6 | 0.7 | 0.3 | 6 | 20 dB | csiqp | 0.9728 | +0.0016 [+0.0009, +0.0023] | 0.9403 | +0.0037 [-0.0000, +0.0075] |  |
| 6 | 0.7 | 0.3 | 6 | 20 dB | switch | 0.9718 | +0.0007 [+0.0001, +0.0013] | 0.9374 | +0.0008 [-0.0026, +0.0041] | frac 0.676; rule_loss +0.0001 [-0.0003, +0.0006]; decision_loss 0.0044; oracle_cf - cos +0.0044; frac_cf 0.549 |
| 6 | 0.7 | 0.3 | 6 | 20 dB | ods | 0.9725 | +0.0014 [+0.0007, +0.0021] | 0.9388 | +0.0022 [-0.0016, +0.0060] |  |
| 6 | 0.7 | 0.3 | 6 | 20 dB | csiqb | 0.9682 | -0.0029 [-0.0035, -0.0022] | 0.9359 | -0.0008 [-0.0041, +0.0027] |  |
| 6 | 0.7 | 0.3 | 6 | 20 dB | csiqw | 0.9728 | +0.0016 [+0.0009, +0.0023] | 0.9404 | +0.0037 [+0.0000, +0.0075] |  |
| 6 | 0.7 | 0.0 | 1 | 0 dB | fixed | 0.9262 | - | 0.9262 | - |  |
| 6 | 0.7 | 0.0 | 1 | 0 dB | csiq | 0.8827 | -0.0435 [-0.0480, -0.0395] | 0.8827 | -0.0435 [-0.0480, -0.0395] |  |
| 6 | 0.7 | 0.0 | 1 | 0 dB | csiqp | 0.8889 | -0.0374 [-0.0415, -0.0335] | 0.8889 | -0.0374 [-0.0415, -0.0335] |  |
| 6 | 0.7 | 0.0 | 1 | 0 dB | switch | 0.9263 | +0.0000 [+0.0000, +0.0000] | 0.9263 | +0.0000 [+0.0000, +0.0000] | frac 0.001; rule_loss -0.0000 [-0.0000, +0.0000]; decision_loss 0.0067; oracle_cf - cos +0.0064; frac_cf 0.281 |
| 6 | 0.7 | 0.0 | 1 | 0 dB | ods | 0.8827 | -0.0435 [-0.0480, -0.0395] | 0.8827 | -0.0435 [-0.0480, -0.0395] |  |
| 6 | 0.7 | 0.0 | 1 | 0 dB | csiqb | 0.8824 | -0.0438 [-0.0475, -0.0403] | 0.8824 | -0.0438 [-0.0475, -0.0403] |  |
| 6 | 0.7 | 0.0 | 1 | 0 dB | csiqw | 0.8860 | -0.0403 [-0.0445, -0.0363] | 0.8860 | -0.0403 [-0.0445, -0.0363] |  |
| 6 | 0.7 | 0.0 | 1 | 10 dB | fixed | 0.9885 | - | 0.9885 | - |  |
| 6 | 0.7 | 0.0 | 1 | 10 dB | csiq | 0.9723 | -0.0162 [-0.0180, -0.0144] | 0.9723 | -0.0162 [-0.0180, -0.0144] |  |
| 6 | 0.7 | 0.0 | 1 | 10 dB | csiqp | 0.9733 | -0.0153 [-0.0170, -0.0136] | 0.9733 | -0.0153 [-0.0170, -0.0136] |  |
| 6 | 0.7 | 0.0 | 1 | 10 dB | switch | 0.9885 | +0.0000 [+0.0000, +0.0000] | 0.9885 | +0.0000 [+0.0000, +0.0000] | frac 0.000; rule_loss +0.0000 [+0.0000, +0.0000]; decision_loss 0.0005; oracle_cf - cos +0.0005; frac_cf 0.176 |
| 6 | 0.7 | 0.0 | 1 | 10 dB | ods | 0.9723 | -0.0162 [-0.0180, -0.0144] | 0.9723 | -0.0162 [-0.0180, -0.0144] |  |
| 6 | 0.7 | 0.0 | 1 | 10 dB | csiqb | 0.9661 | -0.0224 [-0.0240, -0.0210] | 0.9661 | -0.0224 [-0.0240, -0.0210] |  |
| 6 | 0.7 | 0.0 | 1 | 10 dB | csiqw | 0.9732 | -0.0153 [-0.0171, -0.0136] | 0.9732 | -0.0153 [-0.0171, -0.0136] |  |
| 6 | 0.7 | 0.0 | 1 | 20 dB | fixed | 0.9973 | - | 0.9973 | - |  |
| 6 | 0.7 | 0.0 | 1 | 20 dB | csiq | 0.9940 | -0.0033 [-0.0040, -0.0027] | 0.9940 | -0.0033 [-0.0040, -0.0027] |  |
| 6 | 0.7 | 0.0 | 1 | 20 dB | csiqp | 0.9942 | -0.0031 [-0.0038, -0.0026] | 0.9942 | -0.0031 [-0.0038, -0.0026] |  |
| 6 | 0.7 | 0.0 | 1 | 20 dB | switch | 0.9973 | +0.0000 [+0.0000, +0.0000] | 0.9973 | +0.0000 [+0.0000, +0.0000] | frac 0.000; rule_loss +0.0000 [+0.0000, +0.0000]; decision_loss 0.0000; oracle_cf - cos +0.0000; frac_cf 0.157 |
| 6 | 0.7 | 0.0 | 1 | 20 dB | ods | 0.9940 | -0.0033 [-0.0040, -0.0027] | 0.9940 | -0.0033 [-0.0040, -0.0027] |  |
| 6 | 0.7 | 0.0 | 1 | 20 dB | csiqb | 0.9912 | -0.0062 [-0.0067, -0.0057] | 0.9912 | -0.0062 [-0.0067, -0.0057] |  |
| 6 | 0.7 | 0.0 | 1 | 20 dB | csiqw | 0.9942 | -0.0032 [-0.0038, -0.0026] | 0.9942 | -0.0032 [-0.0038, -0.0026] |  |
| 6 | 0.7 | 0.0 | 2 | 0 dB | fixed | 0.8752 | - | 0.8300 | - |  |
| 6 | 0.7 | 0.0 | 2 | 0 dB | csiq | 0.8318 | -0.0434 [-0.0475, -0.0390] | 0.7671 | -0.0629 [-0.0703, -0.0555] |  |
| 6 | 0.7 | 0.0 | 2 | 0 dB | csiqp | 0.8420 | -0.0332 [-0.0374, -0.0290] | 0.7818 | -0.0482 [-0.0553, -0.0412] |  |
| 6 | 0.7 | 0.0 | 2 | 0 dB | switch | 0.8750 | -0.0002 [-0.0006, +0.0001] | 0.8297 | -0.0003 [-0.0011, +0.0003] | frac 0.004; rule_loss +0.0002 [-0.0001, +0.0006]; decision_loss 0.0137; oracle_cf - cos +0.0131; frac_cf 0.317 |
| 6 | 0.7 | 0.0 | 2 | 0 dB | ods | 0.8320 | -0.0433 [-0.0475, -0.0390] | 0.7618 | -0.0682 [-0.0755, -0.0611] |  |
| 6 | 0.7 | 0.0 | 2 | 0 dB | csiqb | 0.8373 | -0.0379 [-0.0418, -0.0342] | 0.7833 | -0.0467 [-0.0530, -0.0401] |  |
| 6 | 0.7 | 0.0 | 2 | 0 dB | csiqw | 0.8377 | -0.0375 [-0.0416, -0.0332] | 0.7763 | -0.0538 [-0.0609, -0.0465] |  |
| 6 | 0.7 | 0.0 | 2 | 10 dB | fixed | 0.9739 | - | 0.9625 | - |  |
| 6 | 0.7 | 0.0 | 2 | 10 dB | csiq | 0.9556 | -0.0183 [-0.0202, -0.0165] | 0.9341 | -0.0284 [-0.0319, -0.0251] |  |
| 6 | 0.7 | 0.0 | 2 | 10 dB | csiqp | 0.9579 | -0.0161 [-0.0178, -0.0143] | 0.9381 | -0.0244 [-0.0278, -0.0212] |  |
| 6 | 0.7 | 0.0 | 2 | 10 dB | switch | 0.9739 | -0.0000 [-0.0000, +0.0000] | 0.9625 | -0.0000 [-0.0001, +0.0000] | frac 0.001; rule_loss +0.0000 [+0.0000, +0.0000]; decision_loss 0.0028; oracle_cf - cos +0.0027; frac_cf 0.271 |
| 6 | 0.7 | 0.0 | 2 | 10 dB | ods | 0.9552 | -0.0187 [-0.0205, -0.0170] | 0.9317 | -0.0308 [-0.0340, -0.0275] |  |
| 6 | 0.7 | 0.0 | 2 | 10 dB | csiqb | 0.9498 | -0.0242 [-0.0257, -0.0228] | 0.9313 | -0.0312 [-0.0338, -0.0287] |  |
| 6 | 0.7 | 0.0 | 2 | 10 dB | csiqw | 0.9574 | -0.0166 [-0.0184, -0.0149] | 0.9371 | -0.0253 [-0.0287, -0.0222] |  |
| 6 | 0.7 | 0.0 | 2 | 20 dB | fixed | 0.9953 | - | 0.9933 | - |  |
| 6 | 0.7 | 0.0 | 2 | 20 dB | csiq | 0.9909 | -0.0043 [-0.0049, -0.0037] | 0.9861 | -0.0072 [-0.0084, -0.0061] |  |
| 6 | 0.7 | 0.0 | 2 | 20 dB | csiqp | 0.9913 | -0.0040 [-0.0045, -0.0035] | 0.9868 | -0.0065 [-0.0075, -0.0055] |  |
| 6 | 0.7 | 0.0 | 2 | 20 dB | switch | 0.9953 | +0.0000 [+0.0000, +0.0000] | 0.9933 | +0.0000 [+0.0000, +0.0000] | frac 0.001; rule_loss -0.0000 [-0.0000, +0.0000]; decision_loss 0.0003; oracle_cf - cos +0.0003; frac_cf 0.215 |
| 6 | 0.7 | 0.0 | 2 | 20 dB | ods | 0.9905 | -0.0048 [-0.0055, -0.0041] | 0.9850 | -0.0083 [-0.0097, -0.0070] |  |
| 6 | 0.7 | 0.0 | 2 | 20 dB | csiqb | 0.9872 | -0.0081 [-0.0085, -0.0077] | 0.9818 | -0.0115 [-0.0122, -0.0107] |  |
| 6 | 0.7 | 0.0 | 2 | 20 dB | csiqw | 0.9913 | -0.0040 [-0.0045, -0.0035] | 0.9867 | -0.0066 [-0.0076, -0.0056] |  |
| 6 | 0.7 | 0.0 | 4 | 0 dB | fixed | 0.7805 | - | 0.6071 | - |  |
| 6 | 0.7 | 0.0 | 4 | 0 dB | csiq | 0.7754 | -0.0051 [-0.0095, -0.0012] | 0.6032 | -0.0039 [-0.0145, +0.0060] |  |
| 6 | 0.7 | 0.0 | 4 | 0 dB | csiqp | 0.7876 | +0.0070 [+0.0029, +0.0108] | 0.6242 | +0.0171 [+0.0066, +0.0268] |  |
| 6 | 0.7 | 0.0 | 4 | 0 dB | switch | 0.7801 | -0.0004 [-0.0026, +0.0016] | 0.6083 | +0.0012 [-0.0042, +0.0065] | frac 0.281; rule_loss +0.0004 [-0.0016, +0.0026]; decision_loss 0.0332; oracle_cf - cos +0.0321; frac_cf 0.475 |
| 6 | 0.7 | 0.0 | 4 | 0 dB | ods | 0.7762 | -0.0043 [-0.0084, -0.0004] | 0.5964 | -0.0106 [-0.0213, -0.0007] |  |
| 6 | 0.7 | 0.0 | 4 | 0 dB | csiqb | 0.7823 | +0.0018 [-0.0023, +0.0053] | 0.6308 | +0.0237 [+0.0134, +0.0332] |  |
| 6 | 0.7 | 0.0 | 4 | 0 dB | csiqw | 0.7820 | +0.0015 [-0.0027, +0.0052] | 0.6150 | +0.0080 [-0.0026, +0.0179] |  |
| 6 | 0.7 | 0.0 | 4 | 10 dB | fixed | 0.9336 | - | 0.8594 | - |  |
| 6 | 0.7 | 0.0 | 4 | 10 dB | csiq | 0.9300 | -0.0036 [-0.0057, -0.0015] | 0.8495 | -0.0099 [-0.0161, -0.0029] |  |
| 6 | 0.7 | 0.0 | 4 | 10 dB | csiqp | 0.9345 | +0.0009 [-0.0011, +0.0029] | 0.8629 | +0.0035 [-0.0026, +0.0101] |  |
| 6 | 0.7 | 0.0 | 4 | 10 dB | switch | 0.9323 | -0.0013 [-0.0025, -0.0002] | 0.8567 | -0.0028 [-0.0064, +0.0011] | frac 0.325; rule_loss +0.0013 [+0.0002, +0.0025]; decision_loss 0.0155; oracle_cf - cos +0.0153; frac_cf 0.488 |
| 6 | 0.7 | 0.0 | 4 | 10 dB | ods | 0.9298 | -0.0039 [-0.0059, -0.0018] | 0.8469 | -0.0126 [-0.0191, -0.0056] |  |
| 6 | 0.7 | 0.0 | 4 | 10 dB | csiqb | 0.9266 | -0.0071 [-0.0087, -0.0052] | 0.8615 | +0.0020 [-0.0035, +0.0081] |  |
| 6 | 0.7 | 0.0 | 4 | 10 dB | csiqw | 0.9332 | -0.0005 [-0.0024, +0.0016] | 0.8572 | -0.0022 [-0.0082, +0.0045] |  |
| 6 | 0.7 | 0.0 | 4 | 20 dB | fixed | 0.9854 | - | 0.9659 | - |  |
| 6 | 0.7 | 0.0 | 4 | 20 dB | csiq | 0.9846 | -0.0008 [-0.0017, -0.0001] | 0.9629 | -0.0030 [-0.0062, -0.0001] |  |
| 6 | 0.7 | 0.0 | 4 | 20 dB | csiqp | 0.9856 | +0.0002 [-0.0007, +0.0009] | 0.9662 | +0.0003 [-0.0026, +0.0031] |  |
| 6 | 0.7 | 0.0 | 4 | 20 dB | switch | 0.9851 | -0.0003 [-0.0008, +0.0001] | 0.9647 | -0.0012 [-0.0030, +0.0005] | frac 0.364; rule_loss +0.0003 [-0.0001, +0.0008]; decision_loss 0.0045; oracle_cf - cos +0.0045; frac_cf 0.479 |
| 6 | 0.7 | 0.0 | 4 | 20 dB | ods | 0.9849 | -0.0006 [-0.0014, +0.0002] | 0.9636 | -0.0024 [-0.0054, +0.0004] |  |
| 6 | 0.7 | 0.0 | 4 | 20 dB | csiqb | 0.9801 | -0.0053 [-0.0060, -0.0047] | 0.9602 | -0.0058 [-0.0084, -0.0033] |  |
| 6 | 0.7 | 0.0 | 4 | 20 dB | csiqw | 0.9856 | +0.0002 [-0.0007, +0.0009] | 0.9662 | +0.0002 [-0.0027, +0.0029] |  |
| 6 | 0.7 | 0.0 | 6 | 0 dB | fixed | 0.7431 | - | 0.4845 | - |  |
| 6 | 0.7 | 0.0 | 6 | 0 dB | csiq | 0.7360 | -0.0071 [-0.0108, -0.0032] | 0.4782 | -0.0063 [-0.0189, +0.0055] |  |
| 6 | 0.7 | 0.0 | 6 | 0 dB | csiqp | 0.7486 | +0.0055 [+0.0018, +0.0092] | 0.5013 | +0.0168 [+0.0040, +0.0291] |  |
| 6 | 0.7 | 0.0 | 6 | 0 dB | switch | 0.7420 | -0.0011 [-0.0030, +0.0007] | 0.4854 | +0.0009 [-0.0051, +0.0064] | frac 0.251; rule_loss +0.0011 [-0.0007, +0.0030]; decision_loss 0.0312; oracle_cf - cos +0.0292; frac_cf 0.456 |
| 6 | 0.7 | 0.0 | 6 | 0 dB | ods | 0.7352 | -0.0079 [-0.0114, -0.0043] | 0.4669 | -0.0176 [-0.0296, -0.0062] |  |
| 6 | 0.7 | 0.0 | 6 | 0 dB | csiqb | 0.7435 | +0.0004 [-0.0031, +0.0043] | 0.5094 | +0.0249 [+0.0119, +0.0373] |  |
| 6 | 0.7 | 0.0 | 6 | 0 dB | csiqw | 0.7427 | -0.0004 [-0.0040, +0.0034] | 0.4904 | +0.0059 [-0.0073, +0.0180] |  |
| 6 | 0.7 | 0.0 | 6 | 10 dB | fixed | 0.9123 | - | 0.7938 | - |  |
| 6 | 0.7 | 0.0 | 6 | 10 dB | csiq | 0.9084 | -0.0039 [-0.0059, -0.0020] | 0.7784 | -0.0154 [-0.0230, -0.0075] |  |
| 6 | 0.7 | 0.0 | 6 | 10 dB | csiqp | 0.9130 | +0.0006 [-0.0013, +0.0025] | 0.7922 | -0.0017 [-0.0091, +0.0059] |  |
| 6 | 0.7 | 0.0 | 6 | 10 dB | switch | 0.9115 | -0.0009 [-0.0019, +0.0002] | 0.7900 | -0.0039 [-0.0082, +0.0006] | frac 0.301; rule_loss +0.0009 [-0.0002, +0.0019]; decision_loss 0.0151; oracle_cf - cos +0.0148; frac_cf 0.462 |
| 6 | 0.7 | 0.0 | 6 | 10 dB | ods | 0.9085 | -0.0038 [-0.0058, -0.0019] | 0.7767 | -0.0171 [-0.0254, -0.0090] |  |
| 6 | 0.7 | 0.0 | 6 | 10 dB | csiqb | 0.9065 | -0.0059 [-0.0077, -0.0041] | 0.7994 | +0.0056 [-0.0014, +0.0129] |  |
| 6 | 0.7 | 0.0 | 6 | 10 dB | csiqw | 0.9117 | -0.0006 [-0.0025, +0.0013] | 0.7877 | -0.0062 [-0.0136, +0.0016] |  |
| 6 | 0.7 | 0.0 | 6 | 20 dB | fixed | 0.9709 | - | 0.9348 | - |  |
| 6 | 0.7 | 0.0 | 6 | 20 dB | csiq | 0.9704 | -0.0005 [-0.0012, +0.0002] | 0.9333 | -0.0015 [-0.0053, +0.0022] |  |
| 6 | 0.7 | 0.0 | 6 | 20 dB | csiqp | 0.9715 | +0.0006 [-0.0001, +0.0013] | 0.9380 | +0.0031 [-0.0004, +0.0068] |  |
| 6 | 0.7 | 0.0 | 6 | 20 dB | switch | 0.9704 | -0.0005 [-0.0009, -0.0001] | 0.9327 | -0.0022 [-0.0044, -0.0001] | frac 0.358; rule_loss +0.0005 [+0.0001, +0.0009]; decision_loss 0.0052; oracle_cf - cos +0.0051; frac_cf 0.477 |
| 6 | 0.7 | 0.0 | 6 | 20 dB | ods | 0.9707 | -0.0002 [-0.0009, +0.0006] | 0.9344 | -0.0004 [-0.0042, +0.0033] |  |
| 6 | 0.7 | 0.0 | 6 | 20 dB | csiqb | 0.9669 | -0.0041 [-0.0047, -0.0034] | 0.9333 | -0.0016 [-0.0048, +0.0017] |  |
| 6 | 0.7 | 0.0 | 6 | 20 dB | csiqw | 0.9715 | +0.0005 [-0.0002, +0.0012] | 0.9375 | +0.0027 [-0.0009, +0.0063] |  |
| 6 | 0.9 | 1.0 | 1 | 0 dB | fixed | 0.9222 | - | 0.9222 | - |  |
| 6 | 0.9 | 1.0 | 1 | 0 dB | csiq | 0.9643 | +0.0422 [+0.0404, +0.0439] | 0.9643 | +0.0422 [+0.0404, +0.0439] |  |
| 6 | 0.9 | 1.0 | 1 | 0 dB | csiqp | 0.9643 | +0.0422 [+0.0404, +0.0439] | 0.9643 | +0.0422 [+0.0404, +0.0439] |  |
| 6 | 0.9 | 1.0 | 1 | 0 dB | switch | 0.9643 | +0.0422 [+0.0404, +0.0439] | 0.9643 | +0.0422 [+0.0404, +0.0439] | frac 1.000; rule_loss +0.0000 [+0.0000, +0.0000]; decision_loss 0.0000; oracle_cf - cos +0.0000; frac_cf 1.000 |
| 6 | 0.9 | 1.0 | 1 | 0 dB | ods | 0.9643 | +0.0422 [+0.0404, +0.0439] | 0.9643 | +0.0422 [+0.0404, +0.0439] |  |
| 6 | 0.9 | 1.0 | 1 | 0 dB | csiqb | 0.9507 | +0.0285 [+0.0268, +0.0302] | 0.9507 | +0.0285 [+0.0268, +0.0302] |  |
| 6 | 0.9 | 1.0 | 1 | 0 dB | csiqw | 0.9633 | +0.0412 [+0.0394, +0.0429] | 0.9633 | +0.0412 [+0.0394, +0.0429] |  |
| 6 | 0.9 | 1.0 | 1 | 10 dB | fixed | 0.9876 | - | 0.9876 | - |  |
| 6 | 0.9 | 1.0 | 1 | 10 dB | csiq | 0.9941 | +0.0065 [+0.0062, +0.0068] | 0.9941 | +0.0065 [+0.0062, +0.0068] |  |
| 6 | 0.9 | 1.0 | 1 | 10 dB | csiqp | 0.9941 | +0.0065 [+0.0062, +0.0068] | 0.9941 | +0.0065 [+0.0062, +0.0068] |  |
| 6 | 0.9 | 1.0 | 1 | 10 dB | switch | 0.9941 | +0.0065 [+0.0062, +0.0068] | 0.9941 | +0.0065 [+0.0062, +0.0068] | frac 1.000; rule_loss +0.0000 [+0.0000, +0.0000]; decision_loss 0.0000; oracle_cf - cos +0.0000; frac_cf 1.000 |
| 6 | 0.9 | 1.0 | 1 | 10 dB | ods | 0.9941 | +0.0065 [+0.0062, +0.0068] | 0.9941 | +0.0065 [+0.0062, +0.0068] |  |
| 6 | 0.9 | 1.0 | 1 | 10 dB | csiqb | 0.9884 | +0.0008 [+0.0006, +0.0010] | 0.9884 | +0.0008 [+0.0006, +0.0010] |  |
| 6 | 0.9 | 1.0 | 1 | 10 dB | csiqw | 0.9940 | +0.0064 [+0.0061, +0.0067] | 0.9940 | +0.0064 [+0.0061, +0.0067] |  |
| 6 | 0.9 | 1.0 | 1 | 20 dB | fixed | 0.9971 | - | 0.9971 | - |  |
| 6 | 0.9 | 1.0 | 1 | 20 dB | csiq | 0.9978 | +0.0007 [+0.0007, +0.0008] | 0.9978 | +0.0007 [+0.0007, +0.0008] |  |
| 6 | 0.9 | 1.0 | 1 | 20 dB | csiqp | 0.9978 | +0.0007 [+0.0007, +0.0008] | 0.9978 | +0.0007 [+0.0007, +0.0008] |  |
| 6 | 0.9 | 1.0 | 1 | 20 dB | switch | 0.9978 | +0.0007 [+0.0007, +0.0008] | 0.9978 | +0.0007 [+0.0007, +0.0008] | frac 1.000; rule_loss +0.0000 [+0.0000, +0.0000]; decision_loss 0.0000; oracle_cf - cos +0.0000; frac_cf 1.000 |
| 6 | 0.9 | 1.0 | 1 | 20 dB | ods | 0.9978 | +0.0007 [+0.0007, +0.0008] | 0.9978 | +0.0007 [+0.0007, +0.0008] |  |
| 6 | 0.9 | 1.0 | 1 | 20 dB | csiqb | 0.9962 | -0.0009 [-0.0010, -0.0008] | 0.9962 | -0.0009 [-0.0010, -0.0008] |  |
| 6 | 0.9 | 1.0 | 1 | 20 dB | csiqw | 0.9978 | +0.0007 [+0.0007, +0.0007] | 0.9978 | +0.0007 [+0.0007, +0.0007] |  |
| 6 | 0.9 | 1.0 | 2 | 0 dB | fixed | 0.8742 | - | 0.8274 | - |  |
| 6 | 0.9 | 1.0 | 2 | 0 dB | csiq | 0.9449 | +0.0707 [+0.0684, +0.0728] | 0.9306 | +0.1032 [+0.0993, +0.1069] |  |
| 6 | 0.9 | 1.0 | 2 | 0 dB | csiqp | 0.9449 | +0.0707 [+0.0684, +0.0728] | 0.9306 | +0.1032 [+0.0993, +0.1069] |  |
| 6 | 0.9 | 1.0 | 2 | 0 dB | switch | 0.9449 | +0.0707 [+0.0684, +0.0728] | 0.9306 | +0.1032 [+0.0993, +0.1069] | frac 1.000; rule_loss +0.0000 [+0.0000, +0.0000]; decision_loss 0.0000; oracle_cf - cos +0.0000; frac_cf 1.000 |
| 6 | 0.9 | 1.0 | 2 | 0 dB | ods | 0.9426 | +0.0683 [+0.0660, +0.0705] | 0.9210 | +0.0936 [+0.0897, +0.0973] |  |
| 6 | 0.9 | 1.0 | 2 | 0 dB | csiqb | 0.9301 | +0.0559 [+0.0536, +0.0580] | 0.9150 | +0.0877 [+0.0837, +0.0913] |  |
| 6 | 0.9 | 1.0 | 2 | 0 dB | csiqw | 0.9411 | +0.0669 [+0.0646, +0.0690] | 0.9280 | +0.1006 [+0.0966, +0.1043] |  |
| 6 | 0.9 | 1.0 | 2 | 10 dB | fixed | 0.9734 | - | 0.9609 | - |  |
| 6 | 0.9 | 1.0 | 2 | 10 dB | csiq | 0.9904 | +0.0170 [+0.0163, +0.0177] | 0.9879 | +0.0270 [+0.0257, +0.0283] |  |
| 6 | 0.9 | 1.0 | 2 | 10 dB | csiqp | 0.9904 | +0.0170 [+0.0163, +0.0177] | 0.9879 | +0.0270 [+0.0257, +0.0283] |  |
| 6 | 0.9 | 1.0 | 2 | 10 dB | switch | 0.9904 | +0.0170 [+0.0163, +0.0177] | 0.9879 | +0.0270 [+0.0257, +0.0283] | frac 1.000; rule_loss +0.0000 [+0.0000, +0.0000]; decision_loss 0.0000; oracle_cf - cos +0.0000; frac_cf 1.000 |
| 6 | 0.9 | 1.0 | 2 | 10 dB | ods | 0.9896 | +0.0163 [+0.0156, +0.0170] | 0.9853 | +0.0244 [+0.0232, +0.0257] |  |
| 6 | 0.9 | 1.0 | 2 | 10 dB | csiqb | 0.9819 | +0.0085 [+0.0079, +0.0092] | 0.9776 | +0.0167 [+0.0155, +0.0180] |  |
| 6 | 0.9 | 1.0 | 2 | 10 dB | csiqw | 0.9902 | +0.0168 [+0.0162, +0.0176] | 0.9877 | +0.0268 [+0.0255, +0.0281] |  |
| 6 | 0.9 | 1.0 | 2 | 20 dB | fixed | 0.9952 | - | 0.9932 | - |  |
| 6 | 0.9 | 1.0 | 2 | 20 dB | csiq | 0.9975 | +0.0022 [+0.0021, +0.0024] | 0.9965 | +0.0033 [+0.0031, +0.0036] |  |
| 6 | 0.9 | 1.0 | 2 | 20 dB | csiqp | 0.9975 | +0.0022 [+0.0021, +0.0024] | 0.9965 | +0.0033 [+0.0031, +0.0036] |  |
| 6 | 0.9 | 1.0 | 2 | 20 dB | switch | 0.9975 | +0.0022 [+0.0021, +0.0024] | 0.9965 | +0.0033 [+0.0031, +0.0036] | frac 1.000; rule_loss +0.0000 [+0.0000, +0.0000]; decision_loss 0.0000; oracle_cf - cos +0.0000; frac_cf 1.000 |
| 6 | 0.9 | 1.0 | 2 | 20 dB | ods | 0.9974 | +0.0022 [+0.0020, +0.0023] | 0.9963 | +0.0032 [+0.0029, +0.0034] |  |
| 6 | 0.9 | 1.0 | 2 | 20 dB | csiqb | 0.9952 | -0.0001 [-0.0002, +0.0001] | 0.9935 | +0.0003 [+0.0001, +0.0006] |  |
| 6 | 0.9 | 1.0 | 2 | 20 dB | csiqw | 0.9975 | +0.0022 [+0.0021, +0.0023] | 0.9965 | +0.0033 [+0.0031, +0.0036] |  |
| 6 | 0.9 | 1.0 | 4 | 0 dB | fixed | 0.7842 | - | 0.6084 | - |  |
| 6 | 0.9 | 1.0 | 4 | 0 dB | csiq | 0.9106 | +0.1264 [+0.1235, +0.1296] | 0.8653 | +0.2569 [+0.2492, +0.2655] |  |
| 6 | 0.9 | 1.0 | 4 | 0 dB | csiqp | 0.9106 | +0.1264 [+0.1235, +0.1296] | 0.8653 | +0.2569 [+0.2492, +0.2655] |  |
| 6 | 0.9 | 1.0 | 4 | 0 dB | switch | 0.9106 | +0.1264 [+0.1235, +0.1296] | 0.8653 | +0.2569 [+0.2492, +0.2655] | frac 1.000; rule_loss +0.0000 [+0.0000, +0.0000]; decision_loss 0.0000; oracle_cf - cos +0.0000; frac_cf 1.000 |
| 6 | 0.9 | 1.0 | 4 | 0 dB | ods | 0.9073 | +0.1232 [+0.1202, +0.1264] | 0.8420 | +0.2336 [+0.2264, +0.2422] |  |
| 6 | 0.9 | 1.0 | 4 | 0 dB | csiqb | 0.8960 | +0.1119 [+0.1089, +0.1151] | 0.8504 | +0.2420 [+0.2344, +0.2506] |  |
| 6 | 0.9 | 1.0 | 4 | 0 dB | csiqw | 0.9046 | +0.1205 [+0.1175, +0.1237] | 0.8612 | +0.2528 [+0.2452, +0.2614] |  |
| 6 | 0.9 | 1.0 | 4 | 10 dB | fixed | 0.9330 | - | 0.8546 | - |  |
| 6 | 0.9 | 1.0 | 4 | 10 dB | csiq | 0.9813 | +0.0483 [+0.0467, +0.0498] | 0.9680 | +0.1135 [+0.1083, +0.1184] |  |
| 6 | 0.9 | 1.0 | 4 | 10 dB | csiqp | 0.9813 | +0.0483 [+0.0467, +0.0498] | 0.9680 | +0.1135 [+0.1083, +0.1184] |  |
| 6 | 0.9 | 1.0 | 4 | 10 dB | switch | 0.9813 | +0.0483 [+0.0467, +0.0498] | 0.9680 | +0.1135 [+0.1083, +0.1184] | frac 1.000; rule_loss +0.0000 [+0.0000, +0.0000]; decision_loss 0.0000; oracle_cf - cos +0.0000; frac_cf 1.000 |
| 6 | 0.9 | 1.0 | 4 | 10 dB | ods | 0.9812 | +0.0483 [+0.0466, +0.0498] | 0.9639 | +0.1093 [+0.1042, +0.1141] |  |
| 6 | 0.9 | 1.0 | 4 | 10 dB | csiqb | 0.9699 | +0.0370 [+0.0354, +0.0384] | 0.9520 | +0.0974 [+0.0923, +0.1024] |  |
| 6 | 0.9 | 1.0 | 4 | 10 dB | csiqw | 0.9808 | +0.0479 [+0.0462, +0.0494] | 0.9675 | +0.1130 [+0.1078, +0.1179] |  |
| 6 | 0.9 | 1.0 | 4 | 20 dB | fixed | 0.9853 | - | 0.9643 | - |  |
| 6 | 0.9 | 1.0 | 4 | 20 dB | csiq | 0.9963 | +0.0110 [+0.0104, +0.0116] | 0.9932 | +0.0288 [+0.0267, +0.0311] |  |
| 6 | 0.9 | 1.0 | 4 | 20 dB | csiqp | 0.9963 | +0.0110 [+0.0104, +0.0116] | 0.9932 | +0.0288 [+0.0267, +0.0311] |  |
| 6 | 0.9 | 1.0 | 4 | 20 dB | switch | 0.9963 | +0.0110 [+0.0104, +0.0116] | 0.9932 | +0.0288 [+0.0267, +0.0311] | frac 1.000; rule_loss +0.0000 [+0.0000, +0.0000]; decision_loss 0.0000; oracle_cf - cos +0.0000; frac_cf 1.000 |
| 6 | 0.9 | 1.0 | 4 | 20 dB | ods | 0.9964 | +0.0111 [+0.0105, +0.0118] | 0.9934 | +0.0290 [+0.0269, +0.0313] |  |
| 6 | 0.9 | 1.0 | 4 | 20 dB | csiqb | 0.9920 | +0.0067 [+0.0061, +0.0073] | 0.9857 | +0.0214 [+0.0193, +0.0237] |  |
| 6 | 0.9 | 1.0 | 4 | 20 dB | csiqw | 0.9961 | +0.0108 [+0.0102, +0.0114] | 0.9928 | +0.0285 [+0.0263, +0.0307] |  |
| 6 | 0.9 | 1.0 | 6 | 0 dB | fixed | 0.7475 | - | 0.4942 | - |  |
| 6 | 0.9 | 1.0 | 6 | 0 dB | csiq | 0.8854 | +0.1379 [+0.1356, +0.1406] | 0.8134 | +0.3192 [+0.3113, +0.3282] |  |
| 6 | 0.9 | 1.0 | 6 | 0 dB | csiqp | 0.8854 | +0.1379 [+0.1356, +0.1406] | 0.8134 | +0.3192 [+0.3113, +0.3282] |  |
| 6 | 0.9 | 1.0 | 6 | 0 dB | switch | 0.8854 | +0.1379 [+0.1356, +0.1406] | 0.8134 | +0.3192 [+0.3113, +0.3282] | frac 1.000; rule_loss +0.0000 [+0.0000, +0.0000]; decision_loss 0.0000; oracle_cf - cos +0.0000; frac_cf 1.000 |
| 6 | 0.9 | 1.0 | 6 | 0 dB | ods | 0.8822 | +0.1348 [+0.1323, +0.1375] | 0.7820 | +0.2879 [+0.2797, +0.2966] |  |
| 6 | 0.9 | 1.0 | 6 | 0 dB | csiqb | 0.8717 | +0.1242 [+0.1219, +0.1269] | 0.7992 | +0.3050 [+0.2970, +0.3138] |  |
| 6 | 0.9 | 1.0 | 6 | 0 dB | csiqw | 0.8788 | +0.1313 [+0.1289, +0.1340] | 0.8083 | +0.3142 [+0.3062, +0.3229] |  |
| 6 | 0.9 | 1.0 | 6 | 10 dB | fixed | 0.9124 | - | 0.7892 | - |  |
| 6 | 0.9 | 1.0 | 6 | 10 dB | csiq | 0.9667 | +0.0543 [+0.0528, +0.0557] | 0.9419 | +0.1528 [+0.1471, +0.1588] |  |
| 6 | 0.9 | 1.0 | 6 | 10 dB | csiqp | 0.9667 | +0.0543 [+0.0528, +0.0557] | 0.9419 | +0.1528 [+0.1471, +0.1588] |  |
| 6 | 0.9 | 1.0 | 6 | 10 dB | switch | 0.9667 | +0.0543 [+0.0528, +0.0557] | 0.9419 | +0.1528 [+0.1471, +0.1588] | frac 1.000; rule_loss +0.0000 [+0.0000, +0.0000]; decision_loss 0.0000; oracle_cf - cos +0.0000; frac_cf 1.000 |
| 6 | 0.9 | 1.0 | 6 | 10 dB | ods | 0.9671 | +0.0546 [+0.0532, +0.0561] | 0.9402 | +0.1510 [+0.1454, +0.1571] |  |
| 6 | 0.9 | 1.0 | 6 | 10 dB | csiqb | 0.9565 | +0.0441 [+0.0427, +0.0455] | 0.9264 | +0.1372 [+0.1315, +0.1433] |  |
| 6 | 0.9 | 1.0 | 6 | 10 dB | csiqw | 0.9662 | +0.0538 [+0.0524, +0.0553] | 0.9411 | +0.1519 [+0.1463, +0.1580] |  |
| 6 | 0.9 | 1.0 | 6 | 20 dB | fixed | 0.9710 | - | 0.9352 | - |  |
| 6 | 0.9 | 1.0 | 6 | 20 dB | csiq | 0.9833 | +0.0123 [+0.0117, +0.0129] | 0.9749 | +0.0397 [+0.0367, +0.0428] |  |
| 6 | 0.9 | 1.0 | 6 | 20 dB | csiqp | 0.9833 | +0.0123 [+0.0117, +0.0129] | 0.9749 | +0.0397 [+0.0367, +0.0428] |  |
| 6 | 0.9 | 1.0 | 6 | 20 dB | switch | 0.9833 | +0.0123 [+0.0117, +0.0129] | 0.9749 | +0.0397 [+0.0367, +0.0428] | frac 1.000; rule_loss +0.0000 [+0.0000, +0.0000]; decision_loss 0.0000; oracle_cf - cos +0.0000; frac_cf 1.000 |
| 6 | 0.9 | 1.0 | 6 | 20 dB | ods | 0.9836 | +0.0126 [+0.0120, +0.0132] | 0.9758 | +0.0407 [+0.0377, +0.0438] |  |
| 6 | 0.9 | 1.0 | 6 | 20 dB | csiqb | 0.9798 | +0.0087 [+0.0082, +0.0093] | 0.9678 | +0.0327 [+0.0298, +0.0358] |  |
| 6 | 0.9 | 1.0 | 6 | 20 dB | csiqw | 0.9831 | +0.0121 [+0.0115, +0.0127] | 0.9744 | +0.0393 [+0.0363, +0.0424] |  |
| 6 | 0.95 | 1.0 | 1 | 0 dB | fixed | 0.9155 | - | 0.9155 | - |  |
| 6 | 0.95 | 1.0 | 1 | 0 dB | csiq | 0.9585 | +0.0430 [+0.0411, +0.0451] | 0.9585 | +0.0430 [+0.0411, +0.0451] |  |
| 6 | 0.95 | 1.0 | 1 | 0 dB | csiqp | 0.9585 | +0.0430 [+0.0411, +0.0451] | 0.9585 | +0.0430 [+0.0411, +0.0451] |  |
| 6 | 0.95 | 1.0 | 1 | 0 dB | switch | 0.9585 | +0.0430 [+0.0411, +0.0451] | 0.9585 | +0.0430 [+0.0411, +0.0451] | frac 1.000; rule_loss +0.0000 [+0.0000, +0.0000]; decision_loss 0.0000; oracle_cf - cos +0.0000; frac_cf 1.000 |
| 6 | 0.95 | 1.0 | 1 | 0 dB | ods | 0.9585 | +0.0430 [+0.0411, +0.0451] | 0.9585 | +0.0430 [+0.0411, +0.0451] |  |
| 6 | 0.95 | 1.0 | 1 | 0 dB | csiqb | 0.9449 | +0.0293 [+0.0275, +0.0313] | 0.9449 | +0.0293 [+0.0275, +0.0313] |  |
| 6 | 0.95 | 1.0 | 1 | 0 dB | csiqw | 0.9573 | +0.0418 [+0.0399, +0.0438] | 0.9573 | +0.0418 [+0.0399, +0.0438] |  |
| 6 | 0.95 | 1.0 | 1 | 10 dB | fixed | 0.9862 | - | 0.9862 | - |  |
| 6 | 0.95 | 1.0 | 1 | 10 dB | csiq | 0.9931 | +0.0069 [+0.0065, +0.0072] | 0.9931 | +0.0069 [+0.0065, +0.0072] |  |
| 6 | 0.95 | 1.0 | 1 | 10 dB | csiqp | 0.9931 | +0.0069 [+0.0065, +0.0072] | 0.9931 | +0.0069 [+0.0065, +0.0072] |  |
| 6 | 0.95 | 1.0 | 1 | 10 dB | switch | 0.9931 | +0.0069 [+0.0065, +0.0072] | 0.9931 | +0.0069 [+0.0065, +0.0072] | frac 1.000; rule_loss +0.0000 [+0.0000, +0.0000]; decision_loss 0.0000; oracle_cf - cos +0.0000; frac_cf 1.000 |
| 6 | 0.95 | 1.0 | 1 | 10 dB | ods | 0.9931 | +0.0069 [+0.0065, +0.0072] | 0.9931 | +0.0069 [+0.0065, +0.0072] |  |
| 6 | 0.95 | 1.0 | 1 | 10 dB | csiqb | 0.9869 | +0.0007 [+0.0004, +0.0009] | 0.9869 | +0.0007 [+0.0004, +0.0009] |  |
| 6 | 0.95 | 1.0 | 1 | 10 dB | csiqw | 0.9930 | +0.0068 [+0.0064, +0.0071] | 0.9930 | +0.0068 [+0.0064, +0.0071] |  |
| 6 | 0.95 | 1.0 | 1 | 20 dB | fixed | 0.9969 | - | 0.9969 | - |  |
| 6 | 0.95 | 1.0 | 1 | 20 dB | csiq | 0.9977 | +0.0008 [+0.0007, +0.0008] | 0.9977 | +0.0008 [+0.0007, +0.0008] |  |
| 6 | 0.95 | 1.0 | 1 | 20 dB | csiqp | 0.9977 | +0.0008 [+0.0007, +0.0008] | 0.9977 | +0.0008 [+0.0007, +0.0008] |  |
| 6 | 0.95 | 1.0 | 1 | 20 dB | switch | 0.9977 | +0.0008 [+0.0007, +0.0008] | 0.9977 | +0.0008 [+0.0007, +0.0008] | frac 1.000; rule_loss +0.0000 [+0.0000, +0.0000]; decision_loss 0.0000; oracle_cf - cos +0.0000; frac_cf 1.000 |
| 6 | 0.95 | 1.0 | 1 | 20 dB | ods | 0.9977 | +0.0008 [+0.0007, +0.0008] | 0.9977 | +0.0008 [+0.0007, +0.0008] |  |
| 6 | 0.95 | 1.0 | 1 | 20 dB | csiqb | 0.9960 | -0.0010 [-0.0011, -0.0009] | 0.9960 | -0.0010 [-0.0011, -0.0009] |  |
| 6 | 0.95 | 1.0 | 1 | 20 dB | csiqw | 0.9977 | +0.0008 [+0.0007, +0.0008] | 0.9977 | +0.0008 [+0.0007, +0.0008] |  |
| 6 | 0.95 | 1.0 | 2 | 0 dB | fixed | 0.8694 | - | 0.8183 | - |  |
| 6 | 0.95 | 1.0 | 2 | 0 dB | csiq | 0.9371 | +0.0676 [+0.0653, +0.0698] | 0.9199 | +0.1016 [+0.0973, +0.1054] |  |
| 6 | 0.95 | 1.0 | 2 | 0 dB | csiqp | 0.9371 | +0.0676 [+0.0653, +0.0698] | 0.9199 | +0.1016 [+0.0973, +0.1054] |  |
| 6 | 0.95 | 1.0 | 2 | 0 dB | switch | 0.9371 | +0.0676 [+0.0653, +0.0698] | 0.9199 | +0.1016 [+0.0973, +0.1054] | frac 1.000; rule_loss +0.0000 [+0.0000, +0.0000]; decision_loss 0.0000; oracle_cf - cos +0.0000; frac_cf 1.000 |
| 6 | 0.95 | 1.0 | 2 | 0 dB | ods | 0.9342 | +0.0648 [+0.0624, +0.0669] | 0.9089 | +0.0906 [+0.0864, +0.0942] |  |
| 6 | 0.95 | 1.0 | 2 | 0 dB | csiqb | 0.9223 | +0.0529 [+0.0506, +0.0551] | 0.9045 | +0.0863 [+0.0820, +0.0901] |  |
| 6 | 0.95 | 1.0 | 2 | 0 dB | csiqw | 0.9331 | +0.0636 [+0.0613, +0.0658] | 0.9167 | +0.0984 [+0.0942, +0.1022] |  |
| 6 | 0.95 | 1.0 | 2 | 10 dB | fixed | 0.9714 | - | 0.9571 | - |  |
| 6 | 0.95 | 1.0 | 2 | 10 dB | csiq | 0.9887 | +0.0173 [+0.0165, +0.0181] | 0.9855 | +0.0284 [+0.0269, +0.0299] |  |
| 6 | 0.95 | 1.0 | 2 | 10 dB | csiqp | 0.9887 | +0.0173 [+0.0165, +0.0181] | 0.9855 | +0.0284 [+0.0269, +0.0299] |  |
| 6 | 0.95 | 1.0 | 2 | 10 dB | switch | 0.9887 | +0.0173 [+0.0165, +0.0181] | 0.9855 | +0.0284 [+0.0269, +0.0299] | frac 1.000; rule_loss +0.0000 [+0.0000, +0.0000]; decision_loss 0.0000; oracle_cf - cos +0.0000; frac_cf 1.000 |
| 6 | 0.95 | 1.0 | 2 | 10 dB | ods | 0.9877 | +0.0163 [+0.0155, +0.0171] | 0.9822 | +0.0251 [+0.0237, +0.0266] |  |
| 6 | 0.95 | 1.0 | 2 | 10 dB | csiqb | 0.9797 | +0.0083 [+0.0076, +0.0090] | 0.9747 | +0.0176 [+0.0162, +0.0190] |  |
| 6 | 0.95 | 1.0 | 2 | 10 dB | csiqw | 0.9885 | +0.0171 [+0.0163, +0.0179] | 0.9853 | +0.0282 [+0.0268, +0.0298] |  |
| 6 | 0.95 | 1.0 | 2 | 20 dB | fixed | 0.9950 | - | 0.9927 | - |  |
| 6 | 0.95 | 1.0 | 2 | 20 dB | csiq | 0.9973 | +0.0023 [+0.0021, +0.0024] | 0.9963 | +0.0035 [+0.0033, +0.0038] |  |
| 6 | 0.95 | 1.0 | 2 | 20 dB | csiqp | 0.9973 | +0.0023 [+0.0021, +0.0024] | 0.9963 | +0.0035 [+0.0033, +0.0038] |  |
| 6 | 0.95 | 1.0 | 2 | 20 dB | switch | 0.9973 | +0.0023 [+0.0021, +0.0024] | 0.9963 | +0.0035 [+0.0033, +0.0038] | frac 1.000; rule_loss +0.0000 [+0.0000, +0.0000]; decision_loss 0.0000; oracle_cf - cos +0.0000; frac_cf 1.000 |
| 6 | 0.95 | 1.0 | 2 | 20 dB | ods | 0.9972 | +0.0022 [+0.0020, +0.0023] | 0.9960 | +0.0033 [+0.0031, +0.0035] |  |
| 6 | 0.95 | 1.0 | 2 | 20 dB | csiqb | 0.9948 | -0.0002 [-0.0003, -0.0001] | 0.9930 | +0.0002 [+0.0000, +0.0005] |  |
| 6 | 0.95 | 1.0 | 2 | 20 dB | csiqw | 0.9973 | +0.0023 [+0.0021, +0.0024] | 0.9962 | +0.0035 [+0.0033, +0.0038] |  |
| 6 | 0.95 | 1.0 | 4 | 0 dB | fixed | 0.7818 | - | 0.5973 | - |  |
| 6 | 0.95 | 1.0 | 4 | 0 dB | csiq | 0.8995 | +0.1177 [+0.1146, +0.1209] | 0.8462 | +0.2489 [+0.2414, +0.2580] |  |
| 6 | 0.95 | 1.0 | 4 | 0 dB | csiqp | 0.8995 | +0.1177 [+0.1146, +0.1209] | 0.8462 | +0.2489 [+0.2414, +0.2580] |  |
| 6 | 0.95 | 1.0 | 4 | 0 dB | switch | 0.8995 | +0.1177 [+0.1146, +0.1209] | 0.8462 | +0.2489 [+0.2414, +0.2580] | frac 1.000; rule_loss +0.0000 [+0.0000, +0.0000]; decision_loss 0.0000; oracle_cf - cos +0.0000; frac_cf 1.000 |
| 6 | 0.95 | 1.0 | 4 | 0 dB | ods | 0.8956 | +0.1138 [+0.1107, +0.1170] | 0.8209 | +0.2236 [+0.2160, +0.2328] |  |
| 6 | 0.95 | 1.0 | 4 | 0 dB | csiqb | 0.8851 | +0.1032 [+0.1001, +0.1065] | 0.8315 | +0.2342 [+0.2266, +0.2433] |  |
| 6 | 0.95 | 1.0 | 4 | 0 dB | csiqw | 0.8935 | +0.1117 [+0.1086, +0.1149] | 0.8403 | +0.2430 [+0.2354, +0.2520] |  |
| 6 | 0.95 | 1.0 | 4 | 10 dB | fixed | 0.9326 | - | 0.8492 | - |  |
| 6 | 0.95 | 1.0 | 4 | 10 dB | csiq | 0.9782 | +0.0456 [+0.0440, +0.0471] | 0.9625 | +0.1132 [+0.1081, +0.1184] |  |
| 6 | 0.95 | 1.0 | 4 | 10 dB | csiqp | 0.9782 | +0.0456 [+0.0440, +0.0471] | 0.9625 | +0.1132 [+0.1081, +0.1184] |  |
| 6 | 0.95 | 1.0 | 4 | 10 dB | switch | 0.9782 | +0.0456 [+0.0440, +0.0471] | 0.9625 | +0.1132 [+0.1081, +0.1184] | frac 1.000; rule_loss +0.0000 [+0.0000, +0.0000]; decision_loss 0.0000; oracle_cf - cos +0.0000; frac_cf 1.000 |
| 6 | 0.95 | 1.0 | 4 | 10 dB | ods | 0.9780 | +0.0454 [+0.0438, +0.0470] | 0.9569 | +0.1077 [+0.1026, +0.1127] |  |
| 6 | 0.95 | 1.0 | 4 | 10 dB | csiqb | 0.9666 | +0.0340 [+0.0324, +0.0355] | 0.9463 | +0.0971 [+0.0920, +0.1021] |  |
| 6 | 0.95 | 1.0 | 4 | 10 dB | csiqw | 0.9778 | +0.0452 [+0.0436, +0.0467] | 0.9619 | +0.1127 [+0.1076, +0.1179] |  |
| 6 | 0.95 | 1.0 | 4 | 20 dB | fixed | 0.9854 | - | 0.9641 | - |  |
| 6 | 0.95 | 1.0 | 4 | 20 dB | csiq | 0.9958 | +0.0104 [+0.0098, +0.0110] | 0.9921 | +0.0280 [+0.0260, +0.0301] |  |
| 6 | 0.95 | 1.0 | 4 | 20 dB | csiqp | 0.9958 | +0.0104 [+0.0098, +0.0110] | 0.9921 | +0.0280 [+0.0260, +0.0301] |  |
| 6 | 0.95 | 1.0 | 4 | 20 dB | switch | 0.9958 | +0.0104 [+0.0098, +0.0110] | 0.9921 | +0.0280 [+0.0260, +0.0301] | frac 1.000; rule_loss +0.0000 [+0.0000, +0.0000]; decision_loss 0.0000; oracle_cf - cos +0.0000; frac_cf 1.000 |
| 6 | 0.95 | 1.0 | 4 | 20 dB | ods | 0.9959 | +0.0105 [+0.0099, +0.0111] | 0.9924 | +0.0283 [+0.0263, +0.0304] |  |
| 6 | 0.95 | 1.0 | 4 | 20 dB | csiqb | 0.9912 | +0.0058 [+0.0053, +0.0064] | 0.9841 | +0.0200 [+0.0180, +0.0222] |  |
| 6 | 0.95 | 1.0 | 4 | 20 dB | csiqw | 0.9956 | +0.0102 [+0.0096, +0.0108] | 0.9917 | +0.0277 [+0.0256, +0.0298] |  |
| 6 | 0.95 | 1.0 | 6 | 0 dB | fixed | 0.7457 | - | 0.4846 | - |  |
| 6 | 0.95 | 1.0 | 6 | 0 dB | csiq | 0.8727 | +0.1270 [+0.1247, +0.1298] | 0.7888 | +0.3042 [+0.2966, +0.3128] |  |
| 6 | 0.95 | 1.0 | 6 | 0 dB | csiqp | 0.8727 | +0.1270 [+0.1247, +0.1298] | 0.7888 | +0.3042 [+0.2966, +0.3128] |  |
| 6 | 0.95 | 1.0 | 6 | 0 dB | switch | 0.8727 | +0.1270 [+0.1247, +0.1298] | 0.7888 | +0.3042 [+0.2966, +0.3128] | frac 1.000; rule_loss +0.0000 [+0.0000, +0.0000]; decision_loss 0.0000; oracle_cf - cos +0.0000; frac_cf 1.000 |
| 6 | 0.95 | 1.0 | 6 | 0 dB | ods | 0.8689 | +0.1232 [+0.1209, +0.1259] | 0.7555 | +0.2709 [+0.2632, +0.2797] |  |
| 6 | 0.95 | 1.0 | 6 | 0 dB | csiqb | 0.8590 | +0.1133 [+0.1110, +0.1161] | 0.7742 | +0.2895 [+0.2820, +0.2982] |  |
| 6 | 0.95 | 1.0 | 6 | 0 dB | csiqw | 0.8660 | +0.1203 [+0.1180, +0.1231] | 0.7806 | +0.2959 [+0.2885, +0.3044] |  |
| 6 | 0.95 | 1.0 | 6 | 10 dB | fixed | 0.9126 | - | 0.7864 | - |  |
| 6 | 0.95 | 1.0 | 6 | 10 dB | csiq | 0.9631 | +0.0505 [+0.0491, +0.0519] | 0.9341 | +0.1477 [+0.1419, +0.1536] |  |
| 6 | 0.95 | 1.0 | 6 | 10 dB | csiqp | 0.9631 | +0.0505 [+0.0491, +0.0519] | 0.9341 | +0.1477 [+0.1419, +0.1536] |  |
| 6 | 0.95 | 1.0 | 6 | 10 dB | switch | 0.9631 | +0.0505 [+0.0491, +0.0519] | 0.9341 | +0.1477 [+0.1419, +0.1536] | frac 1.000; rule_loss +0.0000 [+0.0000, +0.0000]; decision_loss 0.0000; oracle_cf - cos +0.0000; frac_cf 1.000 |
| 6 | 0.95 | 1.0 | 6 | 10 dB | ods | 0.9634 | +0.0509 [+0.0495, +0.0523] | 0.9308 | +0.1444 [+0.1387, +0.1503] |  |
| 6 | 0.95 | 1.0 | 6 | 10 dB | csiqb | 0.9525 | +0.0400 [+0.0386, +0.0414] | 0.9184 | +0.1321 [+0.1263, +0.1379] |  |
| 6 | 0.95 | 1.0 | 6 | 10 dB | csiqw | 0.9626 | +0.0500 [+0.0486, +0.0514] | 0.9329 | +0.1465 [+0.1407, +0.1523] |  |
| 6 | 0.95 | 1.0 | 6 | 20 dB | fixed | 0.9709 | - | 0.9335 | - |  |
| 6 | 0.95 | 1.0 | 6 | 20 dB | csiq | 0.9828 | +0.0119 [+0.0113, +0.0125] | 0.9738 | +0.0403 [+0.0374, +0.0433] |  |
| 6 | 0.95 | 1.0 | 6 | 20 dB | csiqp | 0.9828 | +0.0119 [+0.0113, +0.0125] | 0.9738 | +0.0403 [+0.0374, +0.0433] |  |
| 6 | 0.95 | 1.0 | 6 | 20 dB | switch | 0.9828 | +0.0119 [+0.0113, +0.0125] | 0.9738 | +0.0403 [+0.0374, +0.0433] | frac 1.000; rule_loss +0.0000 [+0.0000, +0.0000]; decision_loss 0.0000; oracle_cf - cos +0.0000; frac_cf 1.000 |
| 6 | 0.95 | 1.0 | 6 | 20 dB | ods | 0.9831 | +0.0122 [+0.0116, +0.0128] | 0.9747 | +0.0413 [+0.0383, +0.0443] |  |
| 6 | 0.95 | 1.0 | 6 | 20 dB | csiqb | 0.9790 | +0.0081 [+0.0075, +0.0087] | 0.9661 | +0.0326 [+0.0297, +0.0355] |  |
| 6 | 0.95 | 1.0 | 6 | 20 dB | csiqw | 0.9826 | +0.0118 [+0.0111, +0.0124] | 0.9734 | +0.0399 [+0.0370, +0.0429] |  |
| 6 | 0.95 | 0.9 | 1 | 0 dB | fixed | 0.9155 | - | 0.9155 | - |  |
| 6 | 0.95 | 0.9 | 1 | 0 dB | csiq | 0.9452 | +0.0297 [+0.0271, +0.0321] | 0.9452 | +0.0297 [+0.0271, +0.0321] |  |
| 6 | 0.95 | 0.9 | 1 | 0 dB | csiqp | 0.9452 | +0.0297 [+0.0270, +0.0321] | 0.9452 | +0.0297 [+0.0270, +0.0321] |  |
| 6 | 0.95 | 0.9 | 1 | 0 dB | switch | 0.9452 | +0.0296 [+0.0270, +0.0320] | 0.9452 | +0.0296 [+0.0270, +0.0320] | frac 0.993; rule_loss +0.0000 [-0.0001, +0.0002]; decision_loss 0.0037; oracle_cf - cos +0.0037; frac_cf 0.947 |
| 6 | 0.95 | 0.9 | 1 | 0 dB | ods | 0.9452 | +0.0297 [+0.0271, +0.0321] | 0.9452 | +0.0297 [+0.0271, +0.0321] |  |
| 6 | 0.95 | 0.9 | 1 | 0 dB | csiqb | 0.9329 | +0.0174 [+0.0149, +0.0197] | 0.9329 | +0.0174 [+0.0149, +0.0197] |  |
| 6 | 0.95 | 0.9 | 1 | 0 dB | csiqw | 0.9439 | +0.0283 [+0.0258, +0.0307] | 0.9439 | +0.0283 [+0.0258, +0.0307] |  |
| 6 | 0.95 | 0.9 | 1 | 10 dB | fixed | 0.9858 | - | 0.9858 | - |  |
| 6 | 0.95 | 0.9 | 1 | 10 dB | csiq | 0.9900 | +0.0042 [+0.0032, +0.0049] | 0.9900 | +0.0042 [+0.0032, +0.0049] |  |
| 6 | 0.95 | 0.9 | 1 | 10 dB | csiqp | 0.9900 | +0.0042 [+0.0032, +0.0049] | 0.9900 | +0.0042 [+0.0032, +0.0049] |  |
| 6 | 0.95 | 0.9 | 1 | 10 dB | switch | 0.9898 | +0.0040 [+0.0031, +0.0047] | 0.9898 | +0.0040 [+0.0031, +0.0047] | frac 0.965; rule_loss +0.0002 [+0.0001, +0.0004]; decision_loss 0.0012; oracle_cf - cos +0.0012; frac_cf 0.939 |
| 6 | 0.95 | 0.9 | 1 | 10 dB | ods | 0.9900 | +0.0042 [+0.0032, +0.0049] | 0.9900 | +0.0042 [+0.0032, +0.0049] |  |
| 6 | 0.95 | 0.9 | 1 | 10 dB | csiqb | 0.9832 | -0.0026 [-0.0035, -0.0019] | 0.9832 | -0.0026 [-0.0035, -0.0019] |  |
| 6 | 0.95 | 0.9 | 1 | 10 dB | csiqw | 0.9900 | +0.0042 [+0.0032, +0.0048] | 0.9900 | +0.0042 [+0.0032, +0.0048] |  |
| 6 | 0.95 | 0.9 | 1 | 20 dB | fixed | 0.9970 | - | 0.9970 | - |  |
| 6 | 0.95 | 0.9 | 1 | 20 dB | csiq | 0.9974 | +0.0004 [+0.0003, +0.0005] | 0.9974 | +0.0004 [+0.0003, +0.0005] |  |
| 6 | 0.95 | 0.9 | 1 | 20 dB | csiqp | 0.9974 | +0.0004 [+0.0003, +0.0005] | 0.9974 | +0.0004 [+0.0003, +0.0005] |  |
| 6 | 0.95 | 0.9 | 1 | 20 dB | switch | 0.9974 | +0.0004 [+0.0003, +0.0005] | 0.9974 | +0.0004 [+0.0003, +0.0005] | frac 0.930; rule_loss -0.0000 [-0.0000, +0.0000]; decision_loss 0.0002; oracle_cf - cos +0.0002; frac_cf 0.929 |
| 6 | 0.95 | 0.9 | 1 | 20 dB | ods | 0.9974 | +0.0004 [+0.0003, +0.0005] | 0.9974 | +0.0004 [+0.0003, +0.0005] |  |
| 6 | 0.95 | 0.9 | 1 | 20 dB | csiqb | 0.9953 | -0.0018 [-0.0019, -0.0016] | 0.9953 | -0.0018 [-0.0019, -0.0016] |  |
| 6 | 0.95 | 0.9 | 1 | 20 dB | csiqw | 0.9974 | +0.0004 [+0.0003, +0.0005] | 0.9974 | +0.0004 [+0.0003, +0.0005] |  |
| 6 | 0.95 | 0.9 | 2 | 0 dB | fixed | 0.8693 | - | 0.8184 | - |  |
| 6 | 0.95 | 0.9 | 2 | 0 dB | csiq | 0.9235 | +0.0542 [+0.0520, +0.0565] | 0.8984 | +0.0800 [+0.0763, +0.0839] |  |
| 6 | 0.95 | 0.9 | 2 | 0 dB | csiqp | 0.9235 | +0.0542 [+0.0520, +0.0565] | 0.8982 | +0.0798 [+0.0759, +0.0837] |  |
| 6 | 0.95 | 0.9 | 2 | 0 dB | switch | 0.9235 | +0.0542 [+0.0520, +0.0565] | 0.8984 | +0.0800 [+0.0763, +0.0839] | frac 1.000; rule_loss +0.0000 [+0.0000, +0.0000]; decision_loss 0.0009; oracle_cf - cos +0.0007; frac_cf 0.975 |
| 6 | 0.95 | 0.9 | 2 | 0 dB | ods | 0.9212 | +0.0518 [+0.0498, +0.0541] | 0.8887 | +0.0703 [+0.0666, +0.0742] |  |
| 6 | 0.95 | 0.9 | 2 | 0 dB | csiqb | 0.9104 | +0.0411 [+0.0389, +0.0434] | 0.8858 | +0.0674 [+0.0637, +0.0714] |  |
| 6 | 0.95 | 0.9 | 2 | 0 dB | csiqw | 0.9199 | +0.0506 [+0.0484, +0.0529] | 0.8954 | +0.0770 [+0.0731, +0.0809] |  |
| 6 | 0.95 | 0.9 | 2 | 10 dB | fixed | 0.9720 | - | 0.9583 | - |  |
| 6 | 0.95 | 0.9 | 2 | 10 dB | csiq | 0.9845 | +0.0126 [+0.0118, +0.0133] | 0.9786 | +0.0203 [+0.0189, +0.0217] |  |
| 6 | 0.95 | 0.9 | 2 | 10 dB | csiqp | 0.9845 | +0.0126 [+0.0118, +0.0133] | 0.9785 | +0.0202 [+0.0188, +0.0216] |  |
| 6 | 0.95 | 0.9 | 2 | 10 dB | switch | 0.9846 | +0.0126 [+0.0119, +0.0134] | 0.9787 | +0.0203 [+0.0189, +0.0217] | frac 0.997; rule_loss -0.0000 [-0.0001, +0.0000]; decision_loss 0.0005; oracle_cf - cos +0.0005; frac_cf 0.964 |
| 6 | 0.95 | 0.9 | 2 | 10 dB | ods | 0.9834 | +0.0114 [+0.0106, +0.0121] | 0.9750 | +0.0167 [+0.0152, +0.0181] |  |
| 6 | 0.95 | 0.9 | 2 | 10 dB | csiqb | 0.9752 | +0.0032 [+0.0026, +0.0040] | 0.9676 | +0.0092 [+0.0079, +0.0106] |  |
| 6 | 0.95 | 0.9 | 2 | 10 dB | csiqw | 0.9844 | +0.0125 [+0.0117, +0.0132] | 0.9785 | +0.0202 [+0.0188, +0.0216] |  |
| 6 | 0.95 | 0.9 | 2 | 20 dB | fixed | 0.9950 | - | 0.9927 | - |  |
| 6 | 0.95 | 0.9 | 2 | 20 dB | csiq | 0.9966 | +0.0016 [+0.0015, +0.0018] | 0.9953 | +0.0026 [+0.0023, +0.0029] |  |
| 6 | 0.95 | 0.9 | 2 | 20 dB | csiqp | 0.9966 | +0.0016 [+0.0015, +0.0018] | 0.9952 | +0.0026 [+0.0023, +0.0029] |  |
| 6 | 0.95 | 0.9 | 2 | 20 dB | switch | 0.9966 | +0.0016 [+0.0015, +0.0018] | 0.9952 | +0.0025 [+0.0023, +0.0028] | frac 0.984; rule_loss +0.0000 [+0.0000, +0.0000]; decision_loss 0.0002; oracle_cf - cos +0.0002; frac_cf 0.952 |
| 6 | 0.95 | 0.9 | 2 | 20 dB | ods | 0.9964 | +0.0014 [+0.0013, +0.0016] | 0.9948 | +0.0021 [+0.0018, +0.0024] |  |
| 6 | 0.95 | 0.9 | 2 | 20 dB | csiqb | 0.9936 | -0.0014 [-0.0016, -0.0012] | 0.9911 | -0.0015 [-0.0018, -0.0012] |  |
| 6 | 0.95 | 0.9 | 2 | 20 dB | csiqw | 0.9966 | +0.0016 [+0.0015, +0.0018] | 0.9952 | +0.0025 [+0.0023, +0.0028] |  |
| 6 | 0.95 | 0.9 | 4 | 0 dB | fixed | 0.7802 | - | 0.5946 | - |  |
| 6 | 0.95 | 0.9 | 4 | 0 dB | csiq | 0.8807 | +0.1005 [+0.0976, +0.1035] | 0.8043 | +0.2097 [+0.2023, +0.2180] |  |
| 6 | 0.95 | 0.9 | 4 | 0 dB | csiqp | 0.8808 | +0.1006 [+0.0976, +0.1036] | 0.8037 | +0.2090 [+0.2016, +0.2174] |  |
| 6 | 0.95 | 0.9 | 4 | 0 dB | switch | 0.8807 | +0.1005 [+0.0976, +0.1035] | 0.8043 | +0.2097 [+0.2023, +0.2180] | frac 1.000; rule_loss +0.0000 [+0.0000, +0.0000]; decision_loss 0.0004; oracle_cf - cos +0.0004; frac_cf 0.984 |
| 6 | 0.95 | 0.9 | 4 | 0 dB | ods | 0.8763 | +0.0961 [+0.0931, +0.0992] | 0.7789 | +0.1843 [+0.1767, +0.1931] |  |
| 6 | 0.95 | 0.9 | 4 | 0 dB | csiqb | 0.8684 | +0.0882 [+0.0853, +0.0912] | 0.7946 | +0.2000 [+0.1925, +0.2083] |  |
| 6 | 0.95 | 0.9 | 4 | 0 dB | csiqw | 0.8754 | +0.0952 [+0.0923, +0.0982] | 0.7981 | +0.2035 [+0.1957, +0.2120] |  |
| 6 | 0.95 | 0.9 | 4 | 10 dB | fixed | 0.9319 | - | 0.8473 | - |  |
| 6 | 0.95 | 0.9 | 4 | 10 dB | csiq | 0.9713 | +0.0394 [+0.0377, +0.0411] | 0.9442 | +0.0970 [+0.0916, +0.1027] |  |
| 6 | 0.95 | 0.9 | 4 | 10 dB | csiqp | 0.9713 | +0.0394 [+0.0377, +0.0411] | 0.9438 | +0.0966 [+0.0912, +0.1023] |  |
| 6 | 0.95 | 0.9 | 4 | 10 dB | switch | 0.9713 | +0.0394 [+0.0377, +0.0411] | 0.9442 | +0.0970 [+0.0916, +0.1027] | frac 1.000; rule_loss +0.0000 [+0.0000, +0.0000]; decision_loss 0.0006; oracle_cf - cos +0.0006; frac_cf 0.979 |
| 6 | 0.95 | 0.9 | 4 | 10 dB | ods | 0.9710 | +0.0391 [+0.0374, +0.0408] | 0.9387 | +0.0914 [+0.0860, +0.0972] |  |
| 6 | 0.95 | 0.9 | 4 | 10 dB | csiqb | 0.9602 | +0.0282 [+0.0266, +0.0300] | 0.9309 | +0.0836 [+0.0783, +0.0893] |  |
| 6 | 0.95 | 0.9 | 4 | 10 dB | csiqw | 0.9710 | +0.0391 [+0.0374, +0.0408] | 0.9437 | +0.0965 [+0.0911, +0.1021] |  |
| 6 | 0.95 | 0.9 | 4 | 20 dB | fixed | 0.9852 | - | 0.9640 | - |  |
| 6 | 0.95 | 0.9 | 4 | 20 dB | csiq | 0.9943 | +0.0090 [+0.0084, +0.0097] | 0.9877 | +0.0237 [+0.0216, +0.0260] |  |
| 6 | 0.95 | 0.9 | 4 | 20 dB | csiqp | 0.9943 | +0.0091 [+0.0085, +0.0097] | 0.9878 | +0.0238 [+0.0218, +0.0261] |  |
| 6 | 0.95 | 0.9 | 4 | 20 dB | switch | 0.9943 | +0.0090 [+0.0084, +0.0097] | 0.9877 | +0.0237 [+0.0216, +0.0260] | frac 1.000; rule_loss +0.0000 [+0.0000, +0.0000]; decision_loss 0.0001; oracle_cf - cos +0.0001; frac_cf 0.974 |
| 6 | 0.95 | 0.9 | 4 | 20 dB | ods | 0.9944 | +0.0091 [+0.0085, +0.0098] | 0.9878 | +0.0239 [+0.0218, +0.0261] |  |
| 6 | 0.95 | 0.9 | 4 | 20 dB | csiqb | 0.9892 | +0.0040 [+0.0034, +0.0046] | 0.9789 | +0.0150 [+0.0130, +0.0173] |  |
| 6 | 0.95 | 0.9 | 4 | 20 dB | csiqw | 0.9942 | +0.0089 [+0.0083, +0.0096] | 0.9876 | +0.0236 [+0.0216, +0.0259] |  |
| 6 | 0.95 | 0.9 | 6 | 0 dB | fixed | 0.7445 | - | 0.4815 | - |  |
| 6 | 0.95 | 0.9 | 6 | 0 dB | csiq | 0.8543 | +0.1098 [+0.1071, +0.1123] | 0.7380 | +0.2565 [+0.2469, +0.2657] |  |
| 6 | 0.95 | 0.9 | 6 | 0 dB | csiqp | 0.8543 | +0.1098 [+0.1070, +0.1124] | 0.7369 | +0.2554 [+0.2458, +0.2645] |  |
| 6 | 0.95 | 0.9 | 6 | 0 dB | switch | 0.8543 | +0.1098 [+0.1071, +0.1123] | 0.7380 | +0.2565 [+0.2469, +0.2657] | frac 1.000; rule_loss +0.0000 [+0.0000, +0.0000]; decision_loss 0.0001; oracle_cf - cos +0.0000; frac_cf 0.997 |
| 6 | 0.95 | 0.9 | 6 | 0 dB | ods | 0.8501 | +0.1056 [+0.1029, +0.1081] | 0.7052 | +0.2237 [+0.2139, +0.2330] |  |
| 6 | 0.95 | 0.9 | 6 | 0 dB | csiqb | 0.8425 | +0.0980 [+0.0953, +0.1005] | 0.7283 | +0.2468 [+0.2375, +0.2558] |  |
| 6 | 0.95 | 0.9 | 6 | 0 dB | csiqw | 0.8483 | +0.1038 [+0.1011, +0.1064] | 0.7281 | +0.2467 [+0.2371, +0.2560] |  |
| 6 | 0.95 | 0.9 | 6 | 10 dB | fixed | 0.9128 | - | 0.7861 | - |  |
| 6 | 0.95 | 0.9 | 6 | 10 dB | csiq | 0.9556 | +0.0428 [+0.0414, +0.0443] | 0.9096 | +0.1234 [+0.1169, +0.1298] |  |
| 6 | 0.95 | 0.9 | 6 | 10 dB | csiqp | 0.9556 | +0.0428 [+0.0414, +0.0443] | 0.9089 | +0.1227 [+0.1164, +0.1291] |  |
| 6 | 0.95 | 0.9 | 6 | 10 dB | switch | 0.9556 | +0.0428 [+0.0414, +0.0443] | 0.9096 | +0.1234 [+0.1169, +0.1298] | frac 1.000; rule_loss +0.0000 [+0.0000, +0.0000]; decision_loss 0.0002; oracle_cf - cos +0.0001; frac_cf 0.991 |
| 6 | 0.95 | 0.9 | 6 | 10 dB | ods | 0.9557 | +0.0429 [+0.0414, +0.0443] | 0.9047 | +0.1186 [+0.1122, +0.1248] |  |
| 6 | 0.95 | 0.9 | 6 | 10 dB | csiqb | 0.9457 | +0.0329 [+0.0315, +0.0343] | 0.8982 | +0.1121 [+0.1058, +0.1184] |  |
| 6 | 0.95 | 0.9 | 6 | 10 dB | csiqw | 0.9552 | +0.0424 [+0.0410, +0.0439] | 0.9084 | +0.1222 [+0.1158, +0.1286] |  |
| 6 | 0.95 | 0.9 | 6 | 20 dB | fixed | 0.9711 | - | 0.9348 | - |  |
| 6 | 0.95 | 0.9 | 6 | 20 dB | csiq | 0.9811 | +0.0100 [+0.0095, +0.0106] | 0.9678 | +0.0330 [+0.0302, +0.0363] |  |
| 6 | 0.95 | 0.9 | 6 | 20 dB | csiqp | 0.9812 | +0.0101 [+0.0095, +0.0107] | 0.9685 | +0.0337 [+0.0308, +0.0370] |  |
| 6 | 0.95 | 0.9 | 6 | 20 dB | switch | 0.9811 | +0.0100 [+0.0095, +0.0106] | 0.9678 | +0.0330 [+0.0302, +0.0363] | frac 1.000; rule_loss +0.0000 [+0.0000, +0.0000]; decision_loss 0.0002; oracle_cf - cos +0.0002; frac_cf 0.973 |
| 6 | 0.95 | 0.9 | 6 | 20 dB | ods | 0.9814 | +0.0103 [+0.0098, +0.0110] | 0.9691 | +0.0343 [+0.0315, +0.0376] |  |
| 6 | 0.95 | 0.9 | 6 | 20 dB | csiqb | 0.9770 | +0.0059 [+0.0053, +0.0065] | 0.9598 | +0.0249 [+0.0221, +0.0283] |  |
| 6 | 0.95 | 0.9 | 6 | 20 dB | csiqw | 0.9810 | +0.0099 [+0.0094, +0.0105] | 0.9677 | +0.0329 [+0.0300, +0.0362] |  |
| 6 | 0.95 | 0.7 | 1 | 0 dB | fixed | 0.9161 | - | 0.9161 | - |  |
| 6 | 0.95 | 0.7 | 1 | 0 dB | csiq | 0.9221 | +0.0060 [+0.0029, +0.0090] | 0.9221 | +0.0060 [+0.0029, +0.0090] |  |
| 6 | 0.95 | 0.7 | 1 | 0 dB | csiqp | 0.9226 | +0.0065 [+0.0034, +0.0095] | 0.9226 | +0.0065 [+0.0034, +0.0095] |  |
| 6 | 0.95 | 0.7 | 1 | 0 dB | switch | 0.9242 | +0.0081 [+0.0058, +0.0106] | 0.9242 | +0.0081 [+0.0058, +0.0106] | frac 0.676; rule_loss -0.0021 [-0.0039, -0.0004]; decision_loss 0.0133; oracle_cf - cos +0.0129; frac_cf 0.746 |
| 6 | 0.95 | 0.7 | 1 | 0 dB | ods | 0.9221 | +0.0060 [+0.0029, +0.0090] | 0.9221 | +0.0060 [+0.0029, +0.0090] |  |
| 6 | 0.95 | 0.7 | 1 | 0 dB | csiqb | 0.9134 | -0.0027 [-0.0053, +0.0001] | 0.9134 | -0.0027 [-0.0053, +0.0001] |  |
| 6 | 0.95 | 0.7 | 1 | 0 dB | csiqw | 0.9211 | +0.0050 [+0.0018, +0.0081] | 0.9211 | +0.0050 [+0.0018, +0.0081] |  |
| 6 | 0.95 | 0.7 | 1 | 10 dB | fixed | 0.9861 | - | 0.9861 | - |  |
| 6 | 0.95 | 0.7 | 1 | 10 dB | csiq | 0.9829 | -0.0032 [-0.0045, -0.0021] | 0.9829 | -0.0032 [-0.0045, -0.0021] |  |
| 6 | 0.95 | 0.7 | 1 | 10 dB | csiqp | 0.9830 | -0.0031 [-0.0043, -0.0020] | 0.9830 | -0.0031 [-0.0043, -0.0020] |  |
| 6 | 0.95 | 0.7 | 1 | 10 dB | switch | 0.9861 | -0.0001 [-0.0006, +0.0005] | 0.9861 | -0.0001 [-0.0006, +0.0005] | frac 0.374; rule_loss +0.0001 [-0.0005, +0.0006]; decision_loss 0.0030; oracle_cf - cos +0.0030; frac_cf 0.698 |
| 6 | 0.95 | 0.7 | 1 | 10 dB | ods | 0.9829 | -0.0032 [-0.0045, -0.0021] | 0.9829 | -0.0032 [-0.0045, -0.0021] |  |
| 6 | 0.95 | 0.7 | 1 | 10 dB | csiqb | 0.9762 | -0.0100 [-0.0110, -0.0090] | 0.9762 | -0.0100 [-0.0110, -0.0090] |  |
| 6 | 0.95 | 0.7 | 1 | 10 dB | csiqw | 0.9830 | -0.0031 [-0.0043, -0.0020] | 0.9830 | -0.0031 [-0.0043, -0.0020] |  |
| 6 | 0.95 | 0.7 | 1 | 20 dB | fixed | 0.9970 | - | 0.9970 | - |  |
| 6 | 0.95 | 0.7 | 1 | 20 dB | csiq | 0.9960 | -0.0010 [-0.0015, -0.0006] | 0.9960 | -0.0010 [-0.0015, -0.0006] |  |
| 6 | 0.95 | 0.7 | 1 | 20 dB | csiqp | 0.9960 | -0.0010 [-0.0014, -0.0006] | 0.9960 | -0.0010 [-0.0014, -0.0006] |  |
| 6 | 0.95 | 0.7 | 1 | 20 dB | switch | 0.9967 | -0.0003 [-0.0006, -0.0000] | 0.9967 | -0.0003 [-0.0006, -0.0000] | frac 0.297; rule_loss +0.0003 [+0.0000, +0.0006]; decision_loss 0.0006; oracle_cf - cos +0.0006; frac_cf 0.678 |
| 6 | 0.95 | 0.7 | 1 | 20 dB | ods | 0.9960 | -0.0010 [-0.0015, -0.0006] | 0.9960 | -0.0010 [-0.0015, -0.0006] |  |
| 6 | 0.95 | 0.7 | 1 | 20 dB | csiqb | 0.9934 | -0.0036 [-0.0040, -0.0033] | 0.9934 | -0.0036 [-0.0040, -0.0033] |  |
| 6 | 0.95 | 0.7 | 1 | 20 dB | csiqw | 0.9960 | -0.0010 [-0.0014, -0.0006] | 0.9960 | -0.0010 [-0.0014, -0.0006] |  |
| 6 | 0.95 | 0.7 | 2 | 0 dB | fixed | 0.8703 | - | 0.8199 | - |  |
| 6 | 0.95 | 0.7 | 2 | 0 dB | csiq | 0.8926 | +0.0223 [+0.0195, +0.0252] | 0.8498 | +0.0299 [+0.0249, +0.0352] |  |
| 6 | 0.95 | 0.7 | 2 | 0 dB | csiqp | 0.8932 | +0.0229 [+0.0201, +0.0259] | 0.8501 | +0.0302 [+0.0252, +0.0357] |  |
| 6 | 0.95 | 0.7 | 2 | 0 dB | switch | 0.8929 | +0.0225 [+0.0199, +0.0253] | 0.8510 | +0.0311 [+0.0262, +0.0363] | frac 0.906; rule_loss -0.0003 [-0.0012, +0.0006]; decision_loss 0.0119; oracle_cf - cos +0.0115; frac_cf 0.783 |
| 6 | 0.95 | 0.7 | 2 | 0 dB | ods | 0.8907 | +0.0204 [+0.0176, +0.0232] | 0.8413 | +0.0214 [+0.0161, +0.0268] |  |
| 6 | 0.95 | 0.7 | 2 | 0 dB | csiqb | 0.8842 | +0.0139 [+0.0113, +0.0167] | 0.8453 | +0.0255 [+0.0208, +0.0304] |  |
| 6 | 0.95 | 0.7 | 2 | 0 dB | csiqw | 0.8906 | +0.0203 [+0.0176, +0.0232] | 0.8484 | +0.0286 [+0.0236, +0.0339] |  |
| 6 | 0.95 | 0.7 | 2 | 10 dB | fixed | 0.9719 | - | 0.9582 | - |  |
| 6 | 0.95 | 0.7 | 2 | 10 dB | csiq | 0.9734 | +0.0015 [+0.0004, +0.0028] | 0.9597 | +0.0015 [-0.0007, +0.0037] |  |
| 6 | 0.95 | 0.7 | 2 | 10 dB | csiqp | 0.9736 | +0.0017 [+0.0006, +0.0030] | 0.9599 | +0.0017 [-0.0005, +0.0039] |  |
| 6 | 0.95 | 0.7 | 2 | 10 dB | switch | 0.9740 | +0.0022 [+0.0012, +0.0032] | 0.9610 | +0.0028 [+0.0008, +0.0046] | frac 0.682; rule_loss -0.0006 [-0.0014, +0.0001]; decision_loss 0.0055; oracle_cf - cos +0.0054; frac_cf 0.724 |
| 6 | 0.95 | 0.7 | 2 | 10 dB | ods | 0.9724 | +0.0005 [-0.0006, +0.0017] | 0.9563 | -0.0019 [-0.0041, +0.0004] |  |
| 6 | 0.95 | 0.7 | 2 | 10 dB | csiqb | 0.9648 | -0.0071 [-0.0080, -0.0060] | 0.9506 | -0.0076 [-0.0093, -0.0057] |  |
| 6 | 0.95 | 0.7 | 2 | 10 dB | csiqw | 0.9736 | +0.0017 [+0.0006, +0.0029] | 0.9601 | +0.0019 [-0.0003, +0.0041] |  |
| 6 | 0.95 | 0.7 | 2 | 20 dB | fixed | 0.9950 | - | 0.9928 | - |  |
| 6 | 0.95 | 0.7 | 2 | 20 dB | csiq | 0.9946 | -0.0004 [-0.0008, -0.0001] | 0.9918 | -0.0010 [-0.0018, -0.0003] |  |
| 6 | 0.95 | 0.7 | 2 | 20 dB | csiqp | 0.9946 | -0.0004 [-0.0008, -0.0001] | 0.9918 | -0.0010 [-0.0018, -0.0004] |  |
| 6 | 0.95 | 0.7 | 2 | 20 dB | switch | 0.9951 | +0.0000 [-0.0003, +0.0002] | 0.9927 | -0.0001 [-0.0007, +0.0004] | frac 0.458; rule_loss -0.0000 [-0.0002, +0.0003]; decision_loss 0.0010; oracle_cf - cos +0.0009; frac_cf 0.707 |
| 6 | 0.95 | 0.7 | 2 | 20 dB | ods | 0.9943 | -0.0007 [-0.0012, -0.0004] | 0.9910 | -0.0017 [-0.0026, -0.0010] |  |
| 6 | 0.95 | 0.7 | 2 | 20 dB | csiqb | 0.9909 | -0.0041 [-0.0045, -0.0038] | 0.9868 | -0.0059 [-0.0068, -0.0053] |  |
| 6 | 0.95 | 0.7 | 2 | 20 dB | csiqw | 0.9946 | -0.0004 [-0.0008, -0.0001] | 0.9918 | -0.0009 [-0.0017, -0.0003] |  |
| 6 | 0.95 | 0.7 | 4 | 0 dB | fixed | 0.7785 | - | 0.5926 | - |  |
| 6 | 0.95 | 0.7 | 4 | 0 dB | csiq | 0.8399 | +0.0615 [+0.0581, +0.0650] | 0.7098 | +0.1172 [+0.1077, +0.1278] |  |
| 6 | 0.95 | 0.7 | 4 | 0 dB | csiqp | 0.8410 | +0.0625 [+0.0592, +0.0661] | 0.7114 | +0.1188 [+0.1096, +0.1294] |  |
| 6 | 0.95 | 0.7 | 4 | 0 dB | switch | 0.8399 | +0.0614 [+0.0581, +0.0650] | 0.7098 | +0.1172 [+0.1077, +0.1278] | frac 1.000; rule_loss +0.0000 [+0.0000, +0.0000]; decision_loss 0.0074; oracle_cf - cos +0.0070; frac_cf 0.846 |
| 6 | 0.95 | 0.7 | 4 | 0 dB | ods | 0.8355 | +0.0570 [+0.0533, +0.0606] | 0.6875 | +0.0949 [+0.0852, +0.1054] |  |
| 6 | 0.95 | 0.7 | 4 | 0 dB | csiqb | 0.8331 | +0.0546 [+0.0512, +0.0580] | 0.7138 | +0.1212 [+0.1121, +0.1316] |  |
| 6 | 0.95 | 0.7 | 4 | 0 dB | csiqw | 0.8371 | +0.0586 [+0.0552, +0.0621] | 0.7076 | +0.1150 [+0.1058, +0.1254] |  |
| 6 | 0.95 | 0.7 | 4 | 10 dB | fixed | 0.9332 | - | 0.8516 | - |  |
| 6 | 0.95 | 0.7 | 4 | 10 dB | csiq | 0.9571 | +0.0240 [+0.0222, +0.0256] | 0.9069 | +0.0553 [+0.0494, +0.0605] |  |
| 6 | 0.95 | 0.7 | 4 | 10 dB | csiqp | 0.9575 | +0.0243 [+0.0226, +0.0259] | 0.9074 | +0.0558 [+0.0499, +0.0609] |  |
| 6 | 0.95 | 0.7 | 4 | 10 dB | switch | 0.9571 | +0.0240 [+0.0222, +0.0256] | 0.9069 | +0.0553 [+0.0494, +0.0605] | frac 0.999; rule_loss -0.0000 [-0.0001, +0.0000]; decision_loss 0.0035; oracle_cf - cos +0.0034; frac_cf 0.838 |
| 6 | 0.95 | 0.7 | 4 | 10 dB | ods | 0.9565 | +0.0233 [+0.0216, +0.0249] | 0.9011 | +0.0495 [+0.0438, +0.0549] |  |
| 6 | 0.95 | 0.7 | 4 | 10 dB | csiqb | 0.9478 | +0.0146 [+0.0129, +0.0162] | 0.9008 | +0.0492 [+0.0436, +0.0540] |  |
| 6 | 0.95 | 0.7 | 4 | 10 dB | csiqw | 0.9573 | +0.0241 [+0.0223, +0.0257] | 0.9075 | +0.0559 [+0.0500, +0.0610] |  |
| 6 | 0.95 | 0.7 | 4 | 20 dB | fixed | 0.9848 | - | 0.9622 | - |  |
| 6 | 0.95 | 0.7 | 4 | 20 dB | csiq | 0.9900 | +0.0052 [+0.0045, +0.0060] | 0.9747 | +0.0125 [+0.0098, +0.0154] |  |
| 6 | 0.95 | 0.7 | 4 | 20 dB | csiqp | 0.9902 | +0.0054 [+0.0047, +0.0061] | 0.9754 | +0.0132 [+0.0106, +0.0160] |  |
| 6 | 0.95 | 0.7 | 4 | 20 dB | switch | 0.9899 | +0.0051 [+0.0044, +0.0058] | 0.9744 | +0.0122 [+0.0095, +0.0150] | frac 0.972; rule_loss +0.0001 [+0.0001, +0.0002]; decision_loss 0.0018; oracle_cf - cos +0.0018; frac_cf 0.820 |
| 6 | 0.95 | 0.7 | 4 | 20 dB | ods | 0.9902 | +0.0053 [+0.0046, +0.0060] | 0.9749 | +0.0127 [+0.0101, +0.0154] |  |
| 6 | 0.95 | 0.7 | 4 | 20 dB | csiqb | 0.9850 | +0.0002 [-0.0004, +0.0009] | 0.9675 | +0.0053 [+0.0027, +0.0078] |  |
| 6 | 0.95 | 0.7 | 4 | 20 dB | csiqw | 0.9902 | +0.0054 [+0.0047, +0.0061] | 0.9755 | +0.0133 [+0.0107, +0.0160] |  |
| 6 | 0.95 | 0.7 | 6 | 0 dB | fixed | 0.7418 | - | 0.4740 | - |  |
| 6 | 0.95 | 0.7 | 6 | 0 dB | csiq | 0.8114 | +0.0696 [+0.0665, +0.0726] | 0.6199 | +0.1460 [+0.1356, +0.1563] |  |
| 6 | 0.95 | 0.7 | 6 | 0 dB | csiqp | 0.8126 | +0.0708 [+0.0677, +0.0738] | 0.6215 | +0.1476 [+0.1373, +0.1582] |  |
| 6 | 0.95 | 0.7 | 6 | 0 dB | switch | 0.8114 | +0.0696 [+0.0665, +0.0726] | 0.6199 | +0.1460 [+0.1356, +0.1563] | frac 1.000; rule_loss +0.0000 [+0.0000, +0.0000]; decision_loss 0.0048; oracle_cf - cos +0.0039; frac_cf 0.905 |
| 6 | 0.95 | 0.7 | 6 | 0 dB | ods | 0.8087 | +0.0669 [+0.0638, +0.0699] | 0.6016 | +0.1277 [+0.1177, +0.1384] |  |
| 6 | 0.95 | 0.7 | 6 | 0 dB | csiqb | 0.8046 | +0.0628 [+0.0599, +0.0658] | 0.6247 | +0.1508 [+0.1408, +0.1614] |  |
| 6 | 0.95 | 0.7 | 6 | 0 dB | csiqw | 0.8079 | +0.0661 [+0.0631, +0.0691] | 0.6140 | +0.1400 [+0.1298, +0.1508] |  |
| 6 | 0.95 | 0.7 | 6 | 10 dB | fixed | 0.9124 | - | 0.7830 | - |  |
| 6 | 0.95 | 0.7 | 6 | 10 dB | csiq | 0.9384 | +0.0261 [+0.0245, +0.0274] | 0.8517 | +0.0687 [+0.0614, +0.0755] |  |
| 6 | 0.95 | 0.7 | 6 | 10 dB | csiqp | 0.9388 | +0.0264 [+0.0249, +0.0278] | 0.8515 | +0.0684 [+0.0613, +0.0752] |  |
| 6 | 0.95 | 0.7 | 6 | 10 dB | switch | 0.9384 | +0.0261 [+0.0245, +0.0274] | 0.8517 | +0.0687 [+0.0614, +0.0755] | frac 1.000; rule_loss +0.0000 [+0.0000, +0.0000]; decision_loss 0.0035; oracle_cf - cos +0.0033; frac_cf 0.856 |
| 6 | 0.95 | 0.7 | 6 | 10 dB | ods | 0.9383 | +0.0260 [+0.0245, +0.0274] | 0.8469 | +0.0639 [+0.0569, +0.0706] |  |
| 6 | 0.95 | 0.7 | 6 | 10 dB | csiqb | 0.9306 | +0.0182 [+0.0167, +0.0196] | 0.8515 | +0.0684 [+0.0617, +0.0750] |  |
| 6 | 0.95 | 0.7 | 6 | 10 dB | csiqw | 0.9386 | +0.0263 [+0.0248, +0.0276] | 0.8523 | +0.0693 [+0.0620, +0.0759] |  |
| 6 | 0.95 | 0.7 | 6 | 20 dB | fixed | 0.9705 | - | 0.9321 | - |  |
| 6 | 0.95 | 0.7 | 6 | 20 dB | csiq | 0.9765 | +0.0060 [+0.0054, +0.0067] | 0.9493 | +0.0172 [+0.0136, +0.0211] |  |
| 6 | 0.95 | 0.7 | 6 | 20 dB | csiqp | 0.9768 | +0.0063 [+0.0056, +0.0070] | 0.9509 | +0.0188 [+0.0154, +0.0227] |  |
| 6 | 0.95 | 0.7 | 6 | 20 dB | switch | 0.9765 | +0.0060 [+0.0053, +0.0067] | 0.9492 | +0.0171 [+0.0135, +0.0211] | frac 0.991; rule_loss +0.0000 [-0.0000, +0.0001]; decision_loss 0.0018; oracle_cf - cos +0.0018; frac_cf 0.812 |
| 6 | 0.95 | 0.7 | 6 | 20 dB | ods | 0.9767 | +0.0062 [+0.0056, +0.0070] | 0.9499 | +0.0178 [+0.0143, +0.0220] |  |
| 6 | 0.95 | 0.7 | 6 | 20 dB | csiqb | 0.9724 | +0.0019 [+0.0013, +0.0026] | 0.9438 | +0.0117 [+0.0086, +0.0154] |  |
| 6 | 0.95 | 0.7 | 6 | 20 dB | csiqw | 0.9767 | +0.0062 [+0.0056, +0.0069] | 0.9507 | +0.0185 [+0.0151, +0.0224] |  |
| 6 | 0.95 | 0.5 | 1 | 0 dB | fixed | 0.9161 | - | 0.9161 | - |  |
| 6 | 0.95 | 0.5 | 1 | 0 dB | csiq | 0.9018 | -0.0143 [-0.0180, -0.0103] | 0.9018 | -0.0143 [-0.0180, -0.0103] |  |
| 6 | 0.95 | 0.5 | 1 | 0 dB | csiqp | 0.9032 | -0.0128 [-0.0166, -0.0090] | 0.9032 | -0.0128 [-0.0166, -0.0090] |  |
| 6 | 0.95 | 0.5 | 1 | 0 dB | switch | 0.9163 | +0.0002 [-0.0015, +0.0019] | 0.9163 | +0.0002 [-0.0015, +0.0019] | frac 0.193; rule_loss -0.0002 [-0.0019, +0.0015]; decision_loss 0.0146; oracle_cf - cos +0.0142; frac_cf 0.570 |
| 6 | 0.95 | 0.5 | 1 | 0 dB | ods | 0.9018 | -0.0143 [-0.0180, -0.0103] | 0.9018 | -0.0143 [-0.0180, -0.0103] |  |
| 6 | 0.95 | 0.5 | 1 | 0 dB | csiqb | 0.8959 | -0.0202 [-0.0236, -0.0165] | 0.8959 | -0.0202 [-0.0236, -0.0165] |  |
| 6 | 0.95 | 0.5 | 1 | 0 dB | csiqw | 0.9015 | -0.0146 [-0.0184, -0.0107] | 0.9015 | -0.0146 [-0.0184, -0.0107] |  |
| 6 | 0.95 | 0.5 | 1 | 10 dB | fixed | 0.9862 | - | 0.9862 | - |  |
| 6 | 0.95 | 0.5 | 1 | 10 dB | csiq | 0.9782 | -0.0080 [-0.0095, -0.0065] | 0.9782 | -0.0080 [-0.0095, -0.0065] |  |
| 6 | 0.95 | 0.5 | 1 | 10 dB | csiqp | 0.9784 | -0.0077 [-0.0092, -0.0063] | 0.9784 | -0.0077 [-0.0092, -0.0063] |  |
| 6 | 0.95 | 0.5 | 1 | 10 dB | switch | 0.9860 | -0.0001 [-0.0004, +0.0001] | 0.9860 | -0.0001 [-0.0004, +0.0001] | frac 0.046; rule_loss +0.0001 [-0.0001, +0.0004]; decision_loss 0.0020; oracle_cf - cos +0.0019; frac_cf 0.500 |
| 6 | 0.95 | 0.5 | 1 | 10 dB | ods | 0.9782 | -0.0080 [-0.0095, -0.0065] | 0.9782 | -0.0080 [-0.0095, -0.0065] |  |
| 6 | 0.95 | 0.5 | 1 | 10 dB | csiqb | 0.9714 | -0.0148 [-0.0161, -0.0136] | 0.9714 | -0.0148 [-0.0161, -0.0136] |  |
| 6 | 0.95 | 0.5 | 1 | 10 dB | csiqw | 0.9785 | -0.0077 [-0.0092, -0.0063] | 0.9785 | -0.0077 [-0.0092, -0.0063] |  |
| 6 | 0.95 | 0.5 | 1 | 20 dB | fixed | 0.9970 | - | 0.9970 | - |  |
| 6 | 0.95 | 0.5 | 1 | 20 dB | csiq | 0.9952 | -0.0018 [-0.0024, -0.0014] | 0.9952 | -0.0018 [-0.0024, -0.0014] |  |
| 6 | 0.95 | 0.5 | 1 | 20 dB | csiqp | 0.9952 | -0.0018 [-0.0024, -0.0014] | 0.9952 | -0.0018 [-0.0024, -0.0014] |  |
| 6 | 0.95 | 0.5 | 1 | 20 dB | switch | 0.9970 | -0.0000 [-0.0000, +0.0000] | 0.9970 | -0.0000 [-0.0000, +0.0000] | frac 0.026; rule_loss +0.0000 [-0.0000, +0.0000]; decision_loss 0.0002; oracle_cf - cos +0.0002; frac_cf 0.485 |
| 6 | 0.95 | 0.5 | 1 | 20 dB | ods | 0.9952 | -0.0018 [-0.0024, -0.0014] | 0.9952 | -0.0018 [-0.0024, -0.0014] |  |
| 6 | 0.95 | 0.5 | 1 | 20 dB | csiqb | 0.9923 | -0.0047 [-0.0052, -0.0043] | 0.9923 | -0.0047 [-0.0052, -0.0043] |  |
| 6 | 0.95 | 0.5 | 1 | 20 dB | csiqw | 0.9952 | -0.0018 [-0.0024, -0.0014] | 0.9952 | -0.0018 [-0.0024, -0.0014] |  |
| 6 | 0.95 | 0.5 | 2 | 0 dB | fixed | 0.8696 | - | 0.8184 | - |  |
| 6 | 0.95 | 0.5 | 2 | 0 dB | csiq | 0.8663 | -0.0033 [-0.0066, +0.0000] | 0.8094 | -0.0090 [-0.0152, -0.0028] |  |
| 6 | 0.95 | 0.5 | 2 | 0 dB | csiqp | 0.8688 | -0.0008 [-0.0040, +0.0024] | 0.8128 | -0.0057 [-0.0117, +0.0003] |  |
| 6 | 0.95 | 0.5 | 2 | 0 dB | switch | 0.8715 | +0.0019 [-0.0003, +0.0042] | 0.8206 | +0.0022 [-0.0020, +0.0064] | frac 0.400; rule_loss -0.0019 [-0.0040, +0.0003]; decision_loss 0.0214; oracle_cf - cos +0.0205; frac_cf 0.577 |
| 6 | 0.95 | 0.5 | 2 | 0 dB | ods | 0.8663 | -0.0033 [-0.0066, -0.0001] | 0.8054 | -0.0130 [-0.0191, -0.0070] |  |
| 6 | 0.95 | 0.5 | 2 | 0 dB | csiqb | 0.8632 | -0.0064 [-0.0093, -0.0035] | 0.8140 | -0.0044 [-0.0099, +0.0011] |  |
| 6 | 0.95 | 0.5 | 2 | 0 dB | csiqw | 0.8661 | -0.0035 [-0.0069, -0.0003] | 0.8104 | -0.0080 [-0.0140, -0.0017] |  |
| 6 | 0.95 | 0.5 | 2 | 10 dB | fixed | 0.9717 | - | 0.9579 | - |  |
| 6 | 0.95 | 0.5 | 2 | 10 dB | csiq | 0.9626 | -0.0091 [-0.0108, -0.0074] | 0.9415 | -0.0164 [-0.0195, -0.0131] |  |
| 6 | 0.95 | 0.5 | 2 | 10 dB | csiqp | 0.9634 | -0.0083 [-0.0099, -0.0067] | 0.9428 | -0.0150 [-0.0180, -0.0119] |  |
| 6 | 0.95 | 0.5 | 2 | 10 dB | switch | 0.9715 | -0.0002 [-0.0007, +0.0002] | 0.9573 | -0.0005 [-0.0013, +0.0003] | frac 0.128; rule_loss +0.0002 [-0.0002, +0.0007]; decision_loss 0.0051; oracle_cf - cos +0.0050; frac_cf 0.511 |
| 6 | 0.95 | 0.5 | 2 | 10 dB | ods | 0.9620 | -0.0097 [-0.0112, -0.0081] | 0.9390 | -0.0188 [-0.0217, -0.0157] |  |
| 6 | 0.95 | 0.5 | 2 | 10 dB | csiqb | 0.9552 | -0.0165 [-0.0179, -0.0151] | 0.9351 | -0.0227 [-0.0253, -0.0199] |  |
| 6 | 0.95 | 0.5 | 2 | 10 dB | csiqw | 0.9631 | -0.0086 [-0.0102, -0.0069] | 0.9425 | -0.0153 [-0.0184, -0.0121] |  |
| 6 | 0.95 | 0.5 | 2 | 20 dB | fixed | 0.9951 | - | 0.9928 | - |  |
| 6 | 0.95 | 0.5 | 2 | 20 dB | csiq | 0.9929 | -0.0021 [-0.0026, -0.0018] | 0.9889 | -0.0039 [-0.0048, -0.0031] |  |
| 6 | 0.95 | 0.5 | 2 | 20 dB | csiqp | 0.9930 | -0.0020 [-0.0025, -0.0017] | 0.9891 | -0.0037 [-0.0046, -0.0030] |  |
| 6 | 0.95 | 0.5 | 2 | 20 dB | switch | 0.9950 | -0.0001 [-0.0001, -0.0000] | 0.9927 | -0.0001 [-0.0003, -0.0000] | frac 0.051; rule_loss +0.0001 [+0.0000, +0.0001]; decision_loss 0.0007; oracle_cf - cos +0.0007; frac_cf 0.503 |
| 6 | 0.95 | 0.5 | 2 | 20 dB | ods | 0.9927 | -0.0024 [-0.0029, -0.0020] | 0.9883 | -0.0045 [-0.0055, -0.0037] |  |
| 6 | 0.95 | 0.5 | 2 | 20 dB | csiqb | 0.9890 | -0.0061 [-0.0065, -0.0057] | 0.9838 | -0.0090 [-0.0098, -0.0083] |  |
| 6 | 0.95 | 0.5 | 2 | 20 dB | csiqw | 0.9930 | -0.0021 [-0.0025, -0.0017] | 0.9891 | -0.0037 [-0.0046, -0.0030] |  |
| 6 | 0.95 | 0.5 | 4 | 0 dB | fixed | 0.7794 | - | 0.5978 | - |  |
| 6 | 0.95 | 0.5 | 4 | 0 dB | csiq | 0.8107 | +0.0313 [+0.0275, +0.0350] | 0.6556 | +0.0578 [+0.0469, +0.0683] |  |
| 6 | 0.95 | 0.5 | 4 | 0 dB | csiqp | 0.8142 | +0.0348 [+0.0309, +0.0384] | 0.6620 | +0.0642 [+0.0534, +0.0746] |  |
| 6 | 0.95 | 0.5 | 4 | 0 dB | switch | 0.8101 | +0.0308 [+0.0269, +0.0344] | 0.6542 | +0.0564 [+0.0458, +0.0666] | frac 0.981; rule_loss +0.0006 [+0.0001, +0.0010]; decision_loss 0.0182; oracle_cf - cos +0.0173; frac_cf 0.686 |
| 6 | 0.95 | 0.5 | 4 | 0 dB | ods | 0.8077 | +0.0283 [+0.0244, +0.0320] | 0.6407 | +0.0429 [+0.0315, +0.0535] |  |
| 6 | 0.95 | 0.5 | 4 | 0 dB | csiqb | 0.8082 | +0.0289 [+0.0253, +0.0324] | 0.6685 | +0.0707 [+0.0604, +0.0807] |  |
| 6 | 0.95 | 0.5 | 4 | 0 dB | csiqw | 0.8100 | +0.0306 [+0.0268, +0.0343] | 0.6559 | +0.0581 [+0.0472, +0.0686] |  |
| 6 | 0.95 | 0.5 | 4 | 10 dB | fixed | 0.9341 | - | 0.8554 | - |  |
| 6 | 0.95 | 0.5 | 4 | 10 dB | csiq | 0.9443 | +0.0102 [+0.0084, +0.0119] | 0.8746 | +0.0192 [+0.0130, +0.0250] |  |
| 6 | 0.95 | 0.5 | 4 | 10 dB | csiqp | 0.9454 | +0.0113 [+0.0095, +0.0130] | 0.8773 | +0.0219 [+0.0159, +0.0277] |  |
| 6 | 0.95 | 0.5 | 4 | 10 dB | switch | 0.9438 | +0.0097 [+0.0079, +0.0113] | 0.8733 | +0.0179 [+0.0119, +0.0236] | frac 0.942; rule_loss +0.0005 [+0.0000, +0.0009]; decision_loss 0.0092; oracle_cf - cos +0.0090; frac_cf 0.673 |
| 6 | 0.95 | 0.5 | 4 | 10 dB | ods | 0.9438 | +0.0097 [+0.0079, +0.0114] | 0.8699 | +0.0145 [+0.0085, +0.0204] |  |
| 6 | 0.95 | 0.5 | 4 | 10 dB | csiqb | 0.9372 | +0.0031 [+0.0014, +0.0046] | 0.8765 | +0.0211 [+0.0157, +0.0265] |  |
| 6 | 0.95 | 0.5 | 4 | 10 dB | csiqw | 0.9451 | +0.0110 [+0.0092, +0.0127] | 0.8768 | +0.0214 [+0.0153, +0.0272] |  |
| 6 | 0.95 | 0.5 | 4 | 20 dB | fixed | 0.9849 | - | 0.9626 | - |  |
| 6 | 0.95 | 0.5 | 4 | 20 dB | csiq | 0.9874 | +0.0024 [+0.0017, +0.0032] | 0.9675 | +0.0049 [+0.0021, +0.0078] |  |
| 6 | 0.95 | 0.5 | 4 | 20 dB | csiqp | 0.9877 | +0.0028 [+0.0020, +0.0035] | 0.9686 | +0.0060 [+0.0033, +0.0088] |  |
| 6 | 0.95 | 0.5 | 4 | 20 dB | switch | 0.9870 | +0.0020 [+0.0013, +0.0027] | 0.9666 | +0.0041 [+0.0014, +0.0066] | frac 0.827; rule_loss +0.0004 [+0.0000, +0.0007]; decision_loss 0.0035; oracle_cf - cos +0.0035; frac_cf 0.647 |
| 6 | 0.95 | 0.5 | 4 | 20 dB | ods | 0.9874 | +0.0025 [+0.0017, +0.0032] | 0.9673 | +0.0048 [+0.0019, +0.0077] |  |
| 6 | 0.95 | 0.5 | 4 | 20 dB | csiqb | 0.9824 | -0.0025 [-0.0033, -0.0019] | 0.9611 | -0.0015 [-0.0040, +0.0011] |  |
| 6 | 0.95 | 0.5 | 4 | 20 dB | csiqw | 0.9877 | +0.0028 [+0.0021, +0.0036] | 0.9689 | +0.0063 [+0.0036, +0.0091] |  |
| 6 | 0.95 | 0.5 | 6 | 0 dB | fixed | 0.7405 | - | 0.4716 | - |  |
| 6 | 0.95 | 0.5 | 6 | 0 dB | csiq | 0.7760 | +0.0355 [+0.0320, +0.0387] | 0.5401 | +0.0685 [+0.0565, +0.0799] |  |
| 6 | 0.95 | 0.5 | 6 | 0 dB | csiqp | 0.7795 | +0.0391 [+0.0356, +0.0422] | 0.5468 | +0.0752 [+0.0639, +0.0868] |  |
| 6 | 0.95 | 0.5 | 6 | 0 dB | switch | 0.7760 | +0.0355 [+0.0320, +0.0386] | 0.5397 | +0.0681 [+0.0563, +0.0795] | frac 0.992; rule_loss +0.0001 [-0.0002, +0.0003]; decision_loss 0.0144; oracle_cf - cos +0.0125; frac_cf 0.729 |
| 6 | 0.95 | 0.5 | 6 | 0 dB | ods | 0.7745 | +0.0340 [+0.0305, +0.0370] | 0.5292 | +0.0576 [+0.0460, +0.0687] |  |
| 6 | 0.95 | 0.5 | 6 | 0 dB | csiqb | 0.7737 | +0.0332 [+0.0299, +0.0363] | 0.5538 | +0.0822 [+0.0705, +0.0937] |  |
| 6 | 0.95 | 0.5 | 6 | 0 dB | csiqw | 0.7751 | +0.0346 [+0.0312, +0.0378] | 0.5391 | +0.0675 [+0.0562, +0.0788] |  |
| 6 | 0.95 | 0.5 | 6 | 10 dB | fixed | 0.9123 | - | 0.7843 | - |  |
| 6 | 0.95 | 0.5 | 6 | 10 dB | csiq | 0.9244 | +0.0121 [+0.0103, +0.0138] | 0.8107 | +0.0264 [+0.0192, +0.0340] |  |
| 6 | 0.95 | 0.5 | 6 | 10 dB | csiqp | 0.9256 | +0.0132 [+0.0115, +0.0149] | 0.8131 | +0.0289 [+0.0217, +0.0364] |  |
| 6 | 0.95 | 0.5 | 6 | 10 dB | switch | 0.9242 | +0.0119 [+0.0101, +0.0136] | 0.8103 | +0.0260 [+0.0189, +0.0335] | frac 0.969; rule_loss +0.0002 [-0.0000, +0.0005]; decision_loss 0.0087; oracle_cf - cos +0.0084; frac_cf 0.684 |
| 6 | 0.95 | 0.5 | 6 | 10 dB | ods | 0.9247 | +0.0123 [+0.0106, +0.0140] | 0.8086 | +0.0243 [+0.0168, +0.0321] |  |
| 6 | 0.95 | 0.5 | 6 | 10 dB | csiqb | 0.9186 | +0.0062 [+0.0046, +0.0079] | 0.8191 | +0.0348 [+0.0280, +0.0422] |  |
| 6 | 0.95 | 0.5 | 6 | 10 dB | csiqw | 0.9252 | +0.0129 [+0.0111, +0.0145] | 0.8128 | +0.0286 [+0.0214, +0.0361] |  |
| 6 | 0.95 | 0.5 | 6 | 20 dB | fixed | 0.9705 | - | 0.9319 | - |  |
| 6 | 0.95 | 0.5 | 6 | 20 dB | csiq | 0.9736 | +0.0030 [+0.0023, +0.0038] | 0.9397 | +0.0078 [+0.0038, +0.0116] |  |
| 6 | 0.95 | 0.5 | 6 | 20 dB | csiqp | 0.9740 | +0.0034 [+0.0027, +0.0041] | 0.9416 | +0.0097 [+0.0057, +0.0135] |  |
| 6 | 0.95 | 0.5 | 6 | 20 dB | switch | 0.9735 | +0.0029 [+0.0023, +0.0035] | 0.9400 | +0.0081 [+0.0049, +0.0114] | frac 0.872; rule_loss +0.0001 [-0.0002, +0.0004]; decision_loss 0.0031; oracle_cf - cos +0.0030; frac_cf 0.675 |
| 6 | 0.95 | 0.5 | 6 | 20 dB | ods | 0.9738 | +0.0033 [+0.0025, +0.0040] | 0.9405 | +0.0086 [+0.0045, +0.0125] |  |
| 6 | 0.95 | 0.5 | 6 | 20 dB | csiqb | 0.9694 | -0.0011 [-0.0018, -0.0005] | 0.9351 | +0.0031 [-0.0005, +0.0068] |  |
| 6 | 0.95 | 0.5 | 6 | 20 dB | csiqw | 0.9740 | +0.0034 [+0.0027, +0.0041] | 0.9415 | +0.0096 [+0.0056, +0.0134] |  |
| 6 | 0.95 | 0.3 | 1 | 0 dB | fixed | 0.9153 | - | 0.9153 | - |  |
| 6 | 0.95 | 0.3 | 1 | 0 dB | csiq | 0.8874 | -0.0279 [-0.0325, -0.0239] | 0.8874 | -0.0279 [-0.0325, -0.0239] |  |
| 6 | 0.95 | 0.3 | 1 | 0 dB | csiqp | 0.8901 | -0.0252 [-0.0297, -0.0213] | 0.8901 | -0.0252 [-0.0297, -0.0213] |  |
| 6 | 0.95 | 0.3 | 1 | 0 dB | switch | 0.9152 | -0.0001 [-0.0008, +0.0004] | 0.9152 | -0.0001 [-0.0008, +0.0004] | frac 0.030; rule_loss +0.0001 [-0.0004, +0.0008]; decision_loss 0.0113; oracle_cf - cos +0.0109; frac_cf 0.442 |
| 6 | 0.95 | 0.3 | 1 | 0 dB | ods | 0.8874 | -0.0279 [-0.0325, -0.0239] | 0.8874 | -0.0279 [-0.0325, -0.0239] |  |
| 6 | 0.95 | 0.3 | 1 | 0 dB | csiqb | 0.8844 | -0.0309 [-0.0348, -0.0274] | 0.8844 | -0.0309 [-0.0348, -0.0274] |  |
| 6 | 0.95 | 0.3 | 1 | 0 dB | csiqw | 0.8879 | -0.0274 [-0.0319, -0.0235] | 0.8879 | -0.0274 [-0.0319, -0.0235] |  |
| 6 | 0.95 | 0.3 | 1 | 10 dB | fixed | 0.9860 | - | 0.9860 | - |  |
| 6 | 0.95 | 0.3 | 1 | 10 dB | csiq | 0.9744 | -0.0117 [-0.0134, -0.0101] | 0.9744 | -0.0117 [-0.0134, -0.0101] |  |
| 6 | 0.95 | 0.3 | 1 | 10 dB | csiqp | 0.9748 | -0.0113 [-0.0129, -0.0098] | 0.9748 | -0.0113 [-0.0129, -0.0098] |  |
| 6 | 0.95 | 0.3 | 1 | 10 dB | switch | 0.9860 | -0.0000 [-0.0000, +0.0000] | 0.9860 | -0.0000 [-0.0000, +0.0000] | frac 0.003; rule_loss +0.0000 [-0.0000, +0.0000]; decision_loss 0.0012; oracle_cf - cos +0.0012; frac_cf 0.368 |
| 6 | 0.95 | 0.3 | 1 | 10 dB | ods | 0.9744 | -0.0117 [-0.0134, -0.0101] | 0.9744 | -0.0117 [-0.0134, -0.0101] |  |
| 6 | 0.95 | 0.3 | 1 | 10 dB | csiqb | 0.9678 | -0.0182 [-0.0196, -0.0169] | 0.9678 | -0.0182 [-0.0196, -0.0169] |  |
| 6 | 0.95 | 0.3 | 1 | 10 dB | csiqw | 0.9748 | -0.0113 [-0.0129, -0.0098] | 0.9748 | -0.0113 [-0.0129, -0.0098] |  |
| 6 | 0.95 | 0.3 | 1 | 20 dB | fixed | 0.9970 | - | 0.9970 | - |  |
| 6 | 0.95 | 0.3 | 1 | 20 dB | csiq | 0.9950 | -0.0020 [-0.0024, -0.0017] | 0.9950 | -0.0020 [-0.0024, -0.0017] |  |
| 6 | 0.95 | 0.3 | 1 | 20 dB | csiqp | 0.9950 | -0.0020 [-0.0023, -0.0017] | 0.9950 | -0.0020 [-0.0023, -0.0017] |  |
| 6 | 0.95 | 0.3 | 1 | 20 dB | switch | 0.9970 | -0.0000 [-0.0000, +0.0000] | 0.9970 | -0.0000 [-0.0000, +0.0000] | frac 0.001; rule_loss +0.0000 [+0.0000, +0.0000]; decision_loss 0.0001; oracle_cf - cos +0.0001; frac_cf 0.364 |
| 6 | 0.95 | 0.3 | 1 | 20 dB | ods | 0.9950 | -0.0020 [-0.0024, -0.0017] | 0.9950 | -0.0020 [-0.0024, -0.0017] |  |
| 6 | 0.95 | 0.3 | 1 | 20 dB | csiqb | 0.9919 | -0.0051 [-0.0054, -0.0047] | 0.9919 | -0.0051 [-0.0054, -0.0047] |  |
| 6 | 0.95 | 0.3 | 1 | 20 dB | csiqw | 0.9950 | -0.0020 [-0.0023, -0.0017] | 0.9950 | -0.0020 [-0.0023, -0.0017] |  |
| 6 | 0.95 | 0.3 | 2 | 0 dB | fixed | 0.8696 | - | 0.8189 | - |  |
| 6 | 0.95 | 0.3 | 2 | 0 dB | csiq | 0.8468 | -0.0228 [-0.0265, -0.0189] | 0.7809 | -0.0380 [-0.0451, -0.0313] |  |
| 6 | 0.95 | 0.3 | 2 | 0 dB | csiqp | 0.8517 | -0.0179 [-0.0217, -0.0142] | 0.7885 | -0.0304 [-0.0374, -0.0239] |  |
| 6 | 0.95 | 0.3 | 2 | 0 dB | switch | 0.8686 | -0.0010 [-0.0020, +0.0000] | 0.8176 | -0.0013 [-0.0029, +0.0005] | frac 0.051; rule_loss +0.0010 [-0.0000, +0.0020]; decision_loss 0.0178; oracle_cf - cos +0.0170; frac_cf 0.438 |
| 6 | 0.95 | 0.3 | 2 | 0 dB | ods | 0.8478 | -0.0218 [-0.0254, -0.0182] | 0.7789 | -0.0400 [-0.0469, -0.0339] |  |
| 6 | 0.95 | 0.3 | 2 | 0 dB | csiqb | 0.8467 | -0.0229 [-0.0264, -0.0194] | 0.7900 | -0.0289 [-0.0354, -0.0226] |  |
| 6 | 0.95 | 0.3 | 2 | 0 dB | csiqw | 0.8480 | -0.0216 [-0.0253, -0.0178] | 0.7837 | -0.0352 [-0.0422, -0.0286] |  |
| 6 | 0.95 | 0.3 | 2 | 10 dB | fixed | 0.9719 | - | 0.9580 | - |  |
| 6 | 0.95 | 0.3 | 2 | 10 dB | csiq | 0.9572 | -0.0146 [-0.0166, -0.0128] | 0.9332 | -0.0248 [-0.0285, -0.0214] |  |
| 6 | 0.95 | 0.3 | 2 | 10 dB | csiqp | 0.9588 | -0.0131 [-0.0149, -0.0113] | 0.9362 | -0.0218 [-0.0252, -0.0185] |  |
| 6 | 0.95 | 0.3 | 2 | 10 dB | switch | 0.9714 | -0.0004 [-0.0009, -0.0000] | 0.9572 | -0.0008 [-0.0018, -0.0000] | frac 0.012; rule_loss +0.0004 [+0.0000, +0.0009]; decision_loss 0.0041; oracle_cf - cos +0.0040; frac_cf 0.378 |
| 6 | 0.95 | 0.3 | 2 | 10 dB | ods | 0.9569 | -0.0149 [-0.0168, -0.0130] | 0.9314 | -0.0267 [-0.0304, -0.0231] |  |
| 6 | 0.95 | 0.3 | 2 | 10 dB | csiqb | 0.9504 | -0.0215 [-0.0230, -0.0199] | 0.9281 | -0.0299 [-0.0328, -0.0271] |  |
| 6 | 0.95 | 0.3 | 2 | 10 dB | csiqw | 0.9581 | -0.0138 [-0.0156, -0.0120] | 0.9348 | -0.0233 [-0.0268, -0.0199] |  |
| 6 | 0.95 | 0.3 | 2 | 20 dB | fixed | 0.9950 | - | 0.9927 | - |  |
| 6 | 0.95 | 0.3 | 2 | 20 dB | csiq | 0.9917 | -0.0032 [-0.0038, -0.0027] | 0.9870 | -0.0056 [-0.0067, -0.0045] |  |
| 6 | 0.95 | 0.3 | 2 | 20 dB | csiqp | 0.9919 | -0.0031 [-0.0036, -0.0025] | 0.9874 | -0.0053 [-0.0064, -0.0042] |  |
| 6 | 0.95 | 0.3 | 2 | 20 dB | switch | 0.9950 | -0.0000 [-0.0000, +0.0000] | 0.9927 | -0.0000 [-0.0000, +0.0000] | frac 0.005; rule_loss +0.0000 [-0.0000, +0.0000]; decision_loss 0.0005; oracle_cf - cos +0.0005; frac_cf 0.363 |
| 6 | 0.95 | 0.3 | 2 | 20 dB | ods | 0.9914 | -0.0036 [-0.0042, -0.0030] | 0.9862 | -0.0065 [-0.0077, -0.0053] |  |
| 6 | 0.95 | 0.3 | 2 | 20 dB | csiqb | 0.9877 | -0.0073 [-0.0079, -0.0068] | 0.9818 | -0.0108 [-0.0119, -0.0099] |  |
| 6 | 0.95 | 0.3 | 2 | 20 dB | csiqw | 0.9919 | -0.0031 [-0.0037, -0.0026] | 0.9873 | -0.0054 [-0.0065, -0.0043] |  |
| 6 | 0.95 | 0.3 | 4 | 0 dB | fixed | 0.7793 | - | 0.5988 | - |  |
| 6 | 0.95 | 0.3 | 4 | 0 dB | csiq | 0.7887 | +0.0094 [+0.0055, +0.0131] | 0.6148 | +0.0160 [+0.0056, +0.0267] |  |
| 6 | 0.95 | 0.3 | 4 | 0 dB | csiqp | 0.7950 | +0.0157 [+0.0120, +0.0193] | 0.6280 | +0.0293 [+0.0189, +0.0397] |  |
| 6 | 0.95 | 0.3 | 4 | 0 dB | switch | 0.7863 | +0.0071 [+0.0037, +0.0102] | 0.6110 | +0.0122 [+0.0029, +0.0207] | frac 0.717; rule_loss +0.0023 [+0.0002, +0.0044]; decision_loss 0.0302; oracle_cf - cos +0.0291; frac_cf 0.565 |
| 6 | 0.95 | 0.3 | 4 | 0 dB | ods | 0.7879 | +0.0086 [+0.0047, +0.0123] | 0.6057 | +0.0070 [-0.0043, +0.0176] |  |
| 6 | 0.95 | 0.3 | 4 | 0 dB | csiqb | 0.7897 | +0.0104 [+0.0068, +0.0141] | 0.6350 | +0.0362 [+0.0254, +0.0466] |  |
| 6 | 0.95 | 0.3 | 4 | 0 dB | csiqw | 0.7902 | +0.0109 [+0.0071, +0.0145] | 0.6197 | +0.0210 [+0.0102, +0.0314] |  |
| 6 | 0.95 | 0.3 | 4 | 10 dB | fixed | 0.9340 | - | 0.8551 | - |  |
| 6 | 0.95 | 0.3 | 4 | 10 dB | csiq | 0.9365 | +0.0025 [+0.0006, +0.0046] | 0.8570 | +0.0020 [-0.0048, +0.0084] |  |
| 6 | 0.95 | 0.3 | 4 | 10 dB | csiqp | 0.9385 | +0.0045 [+0.0026, +0.0065] | 0.8625 | +0.0074 [+0.0007, +0.0137] |  |
| 6 | 0.95 | 0.3 | 4 | 10 dB | switch | 0.9354 | +0.0015 [+0.0000, +0.0030] | 0.8555 | +0.0005 [-0.0046, +0.0056] | frac 0.677; rule_loss +0.0011 [-0.0001, +0.0022]; decision_loss 0.0138; oracle_cf - cos +0.0136; frac_cf 0.559 |
| 6 | 0.95 | 0.3 | 4 | 10 dB | ods | 0.9363 | +0.0023 [+0.0001, +0.0042] | 0.8526 | -0.0024 [-0.0096, +0.0040] |  |
| 6 | 0.95 | 0.3 | 4 | 10 dB | csiqb | 0.9305 | -0.0035 [-0.0053, -0.0017] | 0.8623 | +0.0072 [+0.0008, +0.0132] |  |
| 6 | 0.95 | 0.3 | 4 | 10 dB | csiqw | 0.9378 | +0.0038 [+0.0018, +0.0057] | 0.8601 | +0.0050 [-0.0016, +0.0115] |  |
| 6 | 0.95 | 0.3 | 4 | 20 dB | fixed | 0.9851 | - | 0.9632 | - |  |
| 6 | 0.95 | 0.3 | 4 | 20 dB | csiq | 0.9858 | +0.0007 [-0.0001, +0.0015] | 0.9639 | +0.0007 [-0.0023, +0.0034] |  |
| 6 | 0.95 | 0.3 | 4 | 20 dB | csiqp | 0.9863 | +0.0012 [+0.0004, +0.0020] | 0.9655 | +0.0023 [-0.0005, +0.0051] |  |
| 6 | 0.95 | 0.3 | 4 | 20 dB | switch | 0.9853 | +0.0002 [-0.0004, +0.0008] | 0.9629 | -0.0003 [-0.0025, +0.0018] | frac 0.567; rule_loss +0.0005 [+0.0000, +0.0010]; decision_loss 0.0043; oracle_cf - cos +0.0043; frac_cf 0.559 |
| 6 | 0.95 | 0.3 | 4 | 20 dB | ods | 0.9863 | +0.0012 [+0.0005, +0.0019] | 0.9653 | +0.0021 [-0.0006, +0.0049] |  |
| 6 | 0.95 | 0.3 | 4 | 20 dB | csiqb | 0.9810 | -0.0041 [-0.0049, -0.0034] | 0.9582 | -0.0050 [-0.0076, -0.0024] |  |
| 6 | 0.95 | 0.3 | 4 | 20 dB | csiqw | 0.9863 | +0.0012 [+0.0004, +0.0020] | 0.9656 | +0.0024 [-0.0005, +0.0051] |  |
| 6 | 0.95 | 0.3 | 6 | 0 dB | fixed | 0.7410 | - | 0.4762 | - |  |
| 6 | 0.95 | 0.3 | 6 | 0 dB | csiq | 0.7518 | +0.0107 [+0.0073, +0.0140] | 0.4920 | +0.0158 [+0.0045, +0.0281] |  |
| 6 | 0.95 | 0.3 | 6 | 0 dB | csiqp | 0.7580 | +0.0170 [+0.0135, +0.0203] | 0.5061 | +0.0299 [+0.0190, +0.0423] |  |
| 6 | 0.95 | 0.3 | 6 | 0 dB | switch | 0.7510 | +0.0100 [+0.0070, +0.0129] | 0.4936 | +0.0174 [+0.0081, +0.0289] | frac 0.785; rule_loss +0.0007 [-0.0008, +0.0022]; decision_loss 0.0251; oracle_cf - cos +0.0229; frac_cf 0.570 |
| 6 | 0.95 | 0.3 | 6 | 0 dB | ods | 0.7508 | +0.0097 [+0.0062, +0.0129] | 0.4820 | +0.0058 [-0.0055, +0.0173] |  |
| 6 | 0.95 | 0.3 | 6 | 0 dB | csiqb | 0.7529 | +0.0118 [+0.0085, +0.0150] | 0.5133 | +0.0371 [+0.0260, +0.0493] |  |
| 6 | 0.95 | 0.3 | 6 | 0 dB | csiqw | 0.7528 | +0.0118 [+0.0084, +0.0151] | 0.4955 | +0.0193 [+0.0084, +0.0318] |  |
| 6 | 0.95 | 0.3 | 6 | 10 dB | fixed | 0.9112 | - | 0.7794 | - |  |
| 6 | 0.95 | 0.3 | 6 | 10 dB | csiq | 0.9143 | +0.0031 [+0.0013, +0.0049] | 0.7820 | +0.0026 [-0.0053, +0.0108] |  |
| 6 | 0.95 | 0.3 | 6 | 10 dB | csiqp | 0.9162 | +0.0050 [+0.0033, +0.0067] | 0.7868 | +0.0074 [-0.0003, +0.0153] |  |
| 6 | 0.95 | 0.3 | 6 | 10 dB | switch | 0.9138 | +0.0026 [+0.0011, +0.0041] | 0.7837 | +0.0043 [-0.0023, +0.0113] | frac 0.703; rule_loss +0.0005 [-0.0004, +0.0014]; decision_loss 0.0138; oracle_cf - cos +0.0136; frac_cf 0.564 |
| 6 | 0.95 | 0.3 | 6 | 10 dB | ods | 0.9153 | +0.0041 [+0.0024, +0.0058] | 0.7834 | +0.0040 [-0.0035, +0.0121] |  |
| 6 | 0.95 | 0.3 | 6 | 10 dB | csiqb | 0.9099 | -0.0013 [-0.0029, +0.0004] | 0.7959 | +0.0165 [+0.0090, +0.0242] |  |
| 6 | 0.95 | 0.3 | 6 | 10 dB | csiqw | 0.9155 | +0.0043 [+0.0026, +0.0060] | 0.7852 | +0.0058 [-0.0021, +0.0139] |  |
| 6 | 0.95 | 0.3 | 6 | 20 dB | fixed | 0.9706 | - | 0.9325 | - |  |
| 6 | 0.95 | 0.3 | 6 | 20 dB | csiq | 0.9715 | +0.0009 [+0.0002, +0.0017] | 0.9331 | +0.0006 [-0.0031, +0.0045] |  |
| 6 | 0.95 | 0.3 | 6 | 20 dB | csiqp | 0.9721 | +0.0015 [+0.0008, +0.0022] | 0.9356 | +0.0032 [-0.0004, +0.0071] |  |
| 6 | 0.95 | 0.3 | 6 | 20 dB | switch | 0.9710 | +0.0005 [-0.0001, +0.0011] | 0.9321 | -0.0004 [-0.0031, +0.0027] | frac 0.570; rule_loss +0.0004 [-0.0000, +0.0009]; decision_loss 0.0046; oracle_cf - cos +0.0045; frac_cf 0.565 |
| 6 | 0.95 | 0.3 | 6 | 20 dB | ods | 0.9716 | +0.0010 [+0.0002, +0.0017] | 0.9328 | +0.0004 [-0.0034, +0.0043] |  |
| 6 | 0.95 | 0.3 | 6 | 20 dB | csiqb | 0.9675 | -0.0031 [-0.0038, -0.0025] | 0.9298 | -0.0026 [-0.0059, +0.0007] |  |
| 6 | 0.95 | 0.3 | 6 | 20 dB | csiqw | 0.9720 | +0.0015 [+0.0008, +0.0022] | 0.9355 | +0.0031 [-0.0005, +0.0069] |  |
| 6 | 0.95 | 0.0 | 1 | 0 dB | fixed | 0.9147 | - | 0.9147 | - |  |
| 6 | 0.95 | 0.0 | 1 | 0 dB | csiq | 0.8773 | -0.0374 [-0.0422, -0.0327] | 0.8773 | -0.0374 [-0.0422, -0.0327] |  |
| 6 | 0.95 | 0.0 | 1 | 0 dB | csiqp | 0.8812 | -0.0334 [-0.0383, -0.0289] | 0.8812 | -0.0334 [-0.0383, -0.0289] |  |
| 6 | 0.95 | 0.0 | 1 | 0 dB | switch | 0.9140 | -0.0007 [-0.0017, +0.0000] | 0.9140 | -0.0007 [-0.0017, +0.0000] | frac 0.007; rule_loss +0.0007 [-0.0000, +0.0017]; decision_loss 0.0101; oracle_cf - cos +0.0096; frac_cf 0.370 |
| 6 | 0.95 | 0.0 | 1 | 0 dB | ods | 0.8773 | -0.0374 [-0.0422, -0.0327] | 0.8773 | -0.0374 [-0.0422, -0.0327] |  |
| 6 | 0.95 | 0.0 | 1 | 0 dB | csiqb | 0.8764 | -0.0382 [-0.0426, -0.0343] | 0.8764 | -0.0382 [-0.0426, -0.0343] |  |
| 6 | 0.95 | 0.0 | 1 | 0 dB | csiqw | 0.8787 | -0.0360 [-0.0406, -0.0314] | 0.8787 | -0.0360 [-0.0406, -0.0314] |  |
| 6 | 0.95 | 0.0 | 1 | 10 dB | fixed | 0.9860 | - | 0.9860 | - |  |
| 6 | 0.95 | 0.0 | 1 | 10 dB | csiq | 0.9715 | -0.0145 [-0.0167, -0.0125] | 0.9715 | -0.0145 [-0.0167, -0.0125] |  |
| 6 | 0.95 | 0.0 | 1 | 10 dB | csiqp | 0.9721 | -0.0139 [-0.0160, -0.0119] | 0.9721 | -0.0139 [-0.0160, -0.0119] |  |
| 6 | 0.95 | 0.0 | 1 | 10 dB | switch | 0.9860 | -0.0000 [-0.0001, +0.0000] | 0.9860 | -0.0000 [-0.0001, +0.0000] | frac 0.001; rule_loss +0.0000 [+0.0000, +0.0001]; decision_loss 0.0010; oracle_cf - cos +0.0010; frac_cf 0.300 |
| 6 | 0.95 | 0.0 | 1 | 10 dB | ods | 0.9715 | -0.0145 [-0.0167, -0.0125] | 0.9715 | -0.0145 [-0.0167, -0.0125] |  |
| 6 | 0.95 | 0.0 | 1 | 10 dB | csiqb | 0.9654 | -0.0206 [-0.0224, -0.0189] | 0.9654 | -0.0206 [-0.0224, -0.0189] |  |
| 6 | 0.95 | 0.0 | 1 | 10 dB | csiqw | 0.9720 | -0.0140 [-0.0161, -0.0120] | 0.9720 | -0.0140 [-0.0161, -0.0120] |  |
| 6 | 0.95 | 0.0 | 1 | 20 dB | fixed | 0.9970 | - | 0.9970 | - |  |
| 6 | 0.95 | 0.0 | 1 | 20 dB | csiq | 0.9941 | -0.0030 [-0.0037, -0.0023] | 0.9941 | -0.0030 [-0.0037, -0.0023] |  |
| 6 | 0.95 | 0.0 | 1 | 20 dB | csiqp | 0.9941 | -0.0029 [-0.0036, -0.0023] | 0.9941 | -0.0029 [-0.0036, -0.0023] |  |
| 6 | 0.95 | 0.0 | 1 | 20 dB | switch | 0.9970 | +0.0000 [+0.0000, +0.0000] | 0.9970 | +0.0000 [+0.0000, +0.0000] | frac 0.000; rule_loss +0.0000 [+0.0000, +0.0000]; decision_loss 0.0001; oracle_cf - cos +0.0001; frac_cf 0.290 |
| 6 | 0.95 | 0.0 | 1 | 20 dB | ods | 0.9941 | -0.0030 [-0.0037, -0.0023] | 0.9941 | -0.0030 [-0.0037, -0.0023] |  |
| 6 | 0.95 | 0.0 | 1 | 20 dB | csiqb | 0.9910 | -0.0060 [-0.0066, -0.0055] | 0.9910 | -0.0060 [-0.0066, -0.0055] |  |
| 6 | 0.95 | 0.0 | 1 | 20 dB | csiqw | 0.9941 | -0.0029 [-0.0036, -0.0023] | 0.9941 | -0.0029 [-0.0036, -0.0023] |  |
| 6 | 0.95 | 0.0 | 2 | 0 dB | fixed | 0.8684 | - | 0.8169 | - |  |
| 6 | 0.95 | 0.0 | 2 | 0 dB | csiq | 0.8373 | -0.0310 [-0.0348, -0.0274] | 0.7697 | -0.0473 [-0.0541, -0.0409] |  |
| 6 | 0.95 | 0.0 | 2 | 0 dB | csiqp | 0.8444 | -0.0239 [-0.0275, -0.0204] | 0.7811 | -0.0358 [-0.0423, -0.0295] |  |
| 6 | 0.95 | 0.0 | 2 | 0 dB | switch | 0.8681 | -0.0003 [-0.0005, -0.0001] | 0.8165 | -0.0004 [-0.0008, -0.0001] | frac 0.009; rule_loss +0.0003 [+0.0001, +0.0005]; decision_loss 0.0145; oracle_cf - cos +0.0136; frac_cf 0.357 |
| 6 | 0.95 | 0.0 | 2 | 0 dB | ods | 0.8377 | -0.0307 [-0.0344, -0.0270] | 0.7664 | -0.0505 [-0.0572, -0.0442] |  |
| 6 | 0.95 | 0.0 | 2 | 0 dB | csiqb | 0.8395 | -0.0289 [-0.0320, -0.0256] | 0.7820 | -0.0349 [-0.0407, -0.0294] |  |
| 6 | 0.95 | 0.0 | 2 | 0 dB | csiqw | 0.8395 | -0.0289 [-0.0325, -0.0252] | 0.7736 | -0.0433 [-0.0499, -0.0370] |  |
| 6 | 0.95 | 0.0 | 2 | 10 dB | fixed | 0.9720 | - | 0.9580 | - |  |
| 6 | 0.95 | 0.0 | 2 | 10 dB | csiq | 0.9548 | -0.0172 [-0.0189, -0.0154] | 0.9293 | -0.0287 [-0.0320, -0.0253] |  |
| 6 | 0.95 | 0.0 | 2 | 10 dB | csiqp | 0.9568 | -0.0152 [-0.0168, -0.0135] | 0.9331 | -0.0249 [-0.0279, -0.0218] |  |
| 6 | 0.95 | 0.0 | 2 | 10 dB | switch | 0.9719 | -0.0000 [-0.0001, +0.0000] | 0.9579 | -0.0000 [-0.0002, +0.0000] | frac 0.003; rule_loss +0.0000 [-0.0000, +0.0001]; decision_loss 0.0031; oracle_cf - cos +0.0030; frac_cf 0.323 |
| 6 | 0.95 | 0.0 | 2 | 10 dB | ods | 0.9546 | -0.0174 [-0.0192, -0.0156] | 0.9276 | -0.0303 [-0.0339, -0.0270] |  |
| 6 | 0.95 | 0.0 | 2 | 10 dB | csiqb | 0.9483 | -0.0237 [-0.0250, -0.0222] | 0.9249 | -0.0330 [-0.0357, -0.0303] |  |
| 6 | 0.95 | 0.0 | 2 | 10 dB | csiqw | 0.9558 | -0.0162 [-0.0179, -0.0145] | 0.9310 | -0.0269 [-0.0301, -0.0237] |  |
| 6 | 0.95 | 0.0 | 2 | 20 dB | fixed | 0.9949 | - | 0.9925 | - |  |
| 6 | 0.95 | 0.0 | 2 | 20 dB | csiq | 0.9902 | -0.0047 [-0.0055, -0.0040] | 0.9843 | -0.0082 [-0.0096, -0.0069] |  |
| 6 | 0.95 | 0.0 | 2 | 20 dB | csiqp | 0.9905 | -0.0044 [-0.0051, -0.0037] | 0.9850 | -0.0075 [-0.0089, -0.0064] |  |
| 6 | 0.95 | 0.0 | 2 | 20 dB | switch | 0.9949 | -0.0000 [-0.0000, +0.0000] | 0.9925 | -0.0000 [-0.0001, +0.0000] | frac 0.003; rule_loss +0.0000 [-0.0000, +0.0000]; decision_loss 0.0004; oracle_cf - cos +0.0004; frac_cf 0.285 |
| 6 | 0.95 | 0.0 | 2 | 20 dB | ods | 0.9901 | -0.0048 [-0.0056, -0.0041] | 0.9839 | -0.0086 [-0.0101, -0.0073] |  |
| 6 | 0.95 | 0.0 | 2 | 20 dB | csiqb | 0.9863 | -0.0085 [-0.0091, -0.0080] | 0.9797 | -0.0128 [-0.0139, -0.0118] |  |
| 6 | 0.95 | 0.0 | 2 | 20 dB | csiqw | 0.9904 | -0.0045 [-0.0052, -0.0039] | 0.9847 | -0.0078 [-0.0093, -0.0066] |  |
| 6 | 0.95 | 0.0 | 4 | 0 dB | fixed | 0.7785 | - | 0.5989 | - |  |
| 6 | 0.95 | 0.0 | 4 | 0 dB | csiq | 0.7743 | -0.0042 [-0.0082, -0.0004] | 0.5889 | -0.0099 [-0.0213, +0.0005] |  |
| 6 | 0.95 | 0.0 | 4 | 0 dB | csiqp | 0.7826 | +0.0041 [+0.0002, +0.0077] | 0.6055 | +0.0067 [-0.0043, +0.0170] |  |
| 6 | 0.95 | 0.0 | 4 | 0 dB | switch | 0.7793 | +0.0008 [-0.0017, +0.0032] | 0.6013 | +0.0025 [-0.0046, +0.0091] | frac 0.390; rule_loss -0.0008 [-0.0032, +0.0017]; decision_loss 0.0304; oracle_cf - cos +0.0293; frac_cf 0.484 |
| 6 | 0.95 | 0.0 | 4 | 0 dB | ods | 0.7751 | -0.0034 [-0.0070, +0.0005] | 0.5840 | -0.0149 [-0.0260, -0.0046] |  |
| 6 | 0.95 | 0.0 | 4 | 0 dB | csiqb | 0.7775 | -0.0010 [-0.0047, +0.0024] | 0.6119 | +0.0130 [+0.0022, +0.0230] |  |
| 6 | 0.95 | 0.0 | 4 | 0 dB | csiqw | 0.7766 | -0.0019 [-0.0057, +0.0017] | 0.5940 | -0.0048 [-0.0161, +0.0054] |  |
| 6 | 0.95 | 0.0 | 4 | 10 dB | fixed | 0.9338 | - | 0.8541 | - |  |
| 6 | 0.95 | 0.0 | 4 | 10 dB | csiq | 0.9326 | -0.0011 [-0.0030, +0.0009] | 0.8509 | -0.0033 [-0.0098, +0.0032] |  |
| 6 | 0.95 | 0.0 | 4 | 10 dB | csiqp | 0.9352 | +0.0015 [-0.0005, +0.0035] | 0.8584 | +0.0042 [-0.0023, +0.0106] |  |
| 6 | 0.95 | 0.0 | 4 | 10 dB | switch | 0.9332 | -0.0005 [-0.0017, +0.0007] | 0.8532 | -0.0009 [-0.0049, +0.0032] | frac 0.433; rule_loss +0.0005 [-0.0005, +0.0017]; decision_loss 0.0141; oracle_cf - cos +0.0140; frac_cf 0.489 |
| 6 | 0.95 | 0.0 | 4 | 10 dB | ods | 0.9332 | -0.0006 [-0.0026, +0.0014] | 0.8506 | -0.0035 [-0.0100, +0.0030] |  |
| 6 | 0.95 | 0.0 | 4 | 10 dB | csiqb | 0.9272 | -0.0066 [-0.0084, -0.0048] | 0.8574 | +0.0032 [-0.0028, +0.0092] |  |
| 6 | 0.95 | 0.0 | 4 | 10 dB | csiqw | 0.9341 | +0.0003 [-0.0017, +0.0023] | 0.8543 | +0.0001 [-0.0064, +0.0066] |  |
| 6 | 0.95 | 0.0 | 4 | 20 dB | fixed | 0.9855 | - | 0.9643 | - |  |
| 6 | 0.95 | 0.0 | 4 | 20 dB | csiq | 0.9849 | -0.0006 [-0.0015, +0.0002] | 0.9612 | -0.0031 [-0.0062, -0.0001] |  |
| 6 | 0.95 | 0.0 | 4 | 20 dB | csiqp | 0.9854 | -0.0000 [-0.0009, +0.0008] | 0.9630 | -0.0013 [-0.0044, +0.0016] |  |
| 6 | 0.95 | 0.0 | 4 | 20 dB | switch | 0.9856 | +0.0001 [-0.0005, +0.0006] | 0.9643 | -0.0000 [-0.0019, +0.0018] | frac 0.394; rule_loss -0.0001 [-0.0006, +0.0005]; decision_loss 0.0039; oracle_cf - cos +0.0039; frac_cf 0.510 |
| 6 | 0.95 | 0.0 | 4 | 20 dB | ods | 0.9851 | -0.0003 [-0.0011, +0.0005] | 0.9620 | -0.0023 [-0.0054, +0.0007] |  |
| 6 | 0.95 | 0.0 | 4 | 20 dB | csiqb | 0.9802 | -0.0053 [-0.0060, -0.0046] | 0.9564 | -0.0080 [-0.0106, -0.0054] |  |
| 6 | 0.95 | 0.0 | 4 | 20 dB | csiqw | 0.9854 | -0.0000 [-0.0009, +0.0008] | 0.9630 | -0.0013 [-0.0043, +0.0017] |  |
| 6 | 0.95 | 0.0 | 6 | 0 dB | fixed | 0.7423 | - | 0.4804 | - |  |
| 6 | 0.95 | 0.0 | 6 | 0 dB | csiq | 0.7390 | -0.0032 [-0.0065, +0.0000] | 0.4640 | -0.0164 [-0.0274, -0.0047] |  |
| 6 | 0.95 | 0.0 | 6 | 0 dB | csiqp | 0.7480 | +0.0057 [+0.0025, +0.0088] | 0.4867 | +0.0063 [-0.0047, +0.0181] |  |
| 6 | 0.95 | 0.0 | 6 | 0 dB | switch | 0.7419 | -0.0003 [-0.0023, +0.0018] | 0.4771 | -0.0033 [-0.0104, +0.0038] | frac 0.358; rule_loss +0.0003 [-0.0018, +0.0023]; decision_loss 0.0276; oracle_cf - cos +0.0252; frac_cf 0.473 |
| 6 | 0.95 | 0.0 | 6 | 0 dB | ods | 0.7391 | -0.0032 [-0.0066, -0.0001] | 0.4616 | -0.0188 [-0.0293, -0.0073] |  |
| 6 | 0.95 | 0.0 | 6 | 0 dB | csiqb | 0.7418 | -0.0004 [-0.0036, +0.0026] | 0.4889 | +0.0085 [-0.0026, +0.0201] |  |
| 6 | 0.95 | 0.0 | 6 | 0 dB | csiqw | 0.7411 | -0.0012 [-0.0045, +0.0020] | 0.4682 | -0.0122 [-0.0232, -0.0008] |  |
| 6 | 0.95 | 0.0 | 6 | 10 dB | fixed | 0.9117 | - | 0.7831 | - |  |
| 6 | 0.95 | 0.0 | 6 | 10 dB | csiq | 0.9083 | -0.0035 [-0.0053, -0.0014] | 0.7658 | -0.0173 [-0.0252, -0.0095] |  |
| 6 | 0.95 | 0.0 | 6 | 10 dB | csiqp | 0.9108 | -0.0009 [-0.0027, +0.0011] | 0.7724 | -0.0106 [-0.0186, -0.0028] |  |
| 6 | 0.95 | 0.0 | 6 | 10 dB | switch | 0.9106 | -0.0011 [-0.0024, +0.0002] | 0.7778 | -0.0053 [-0.0109, +0.0002] | frac 0.392; rule_loss +0.0011 [-0.0002, +0.0024]; decision_loss 0.0144; oracle_cf - cos +0.0140; frac_cf 0.474 |
| 6 | 0.95 | 0.0 | 6 | 10 dB | ods | 0.9092 | -0.0026 [-0.0044, -0.0006] | 0.7672 | -0.0158 [-0.0243, -0.0081] |  |
| 6 | 0.95 | 0.0 | 6 | 10 dB | csiqb | 0.9048 | -0.0070 [-0.0087, -0.0051] | 0.7826 | -0.0004 [-0.0080, +0.0066] |  |
| 6 | 0.95 | 0.0 | 6 | 10 dB | csiqw | 0.9097 | -0.0021 [-0.0039, -0.0000] | 0.7691 | -0.0140 [-0.0222, -0.0062] |  |
| 6 | 0.95 | 0.0 | 6 | 20 dB | fixed | 0.9710 | - | 0.9336 | - |  |
| 6 | 0.95 | 0.0 | 6 | 20 dB | csiq | 0.9702 | -0.0008 [-0.0015, -0.0001] | 0.9285 | -0.0051 [-0.0089, -0.0015] |  |
| 6 | 0.95 | 0.0 | 6 | 20 dB | csiqp | 0.9709 | -0.0000 [-0.0007, +0.0006] | 0.9318 | -0.0018 [-0.0055, +0.0017] |  |
| 6 | 0.95 | 0.0 | 6 | 20 dB | switch | 0.9707 | -0.0003 [-0.0008, +0.0002] | 0.9314 | -0.0022 [-0.0048, +0.0001] | frac 0.392; rule_loss +0.0003 [-0.0001, +0.0008]; decision_loss 0.0046; oracle_cf - cos +0.0045; frac_cf 0.490 |
| 6 | 0.95 | 0.0 | 6 | 20 dB | ods | 0.9705 | -0.0004 [-0.0012, +0.0003] | 0.9296 | -0.0041 [-0.0080, -0.0004] |  |
| 6 | 0.95 | 0.0 | 6 | 20 dB | csiqb | 0.9664 | -0.0046 [-0.0053, -0.0040] | 0.9267 | -0.0070 [-0.0104, -0.0037] |  |
| 6 | 0.95 | 0.0 | 6 | 20 dB | csiqw | 0.9709 | -0.0001 [-0.0008, +0.0006] | 0.9314 | -0.0022 [-0.0059, +0.0013] |  |
| 6 | 1.0 | 1.0 | 1 | 0 dB | fixed | 0.8804 | - | 0.8804 | - |  |
| 6 | 1.0 | 1.0 | 1 | 0 dB | csiq | 0.8991 | +0.0187 [+0.0164, +0.0209] | 0.8991 | +0.0187 [+0.0164, +0.0209] |  |
| 6 | 1.0 | 1.0 | 1 | 0 dB | csiqp | 0.8991 | +0.0187 [+0.0164, +0.0209] | 0.8991 | +0.0187 [+0.0164, +0.0209] |  |
| 6 | 1.0 | 1.0 | 1 | 0 dB | switch | 0.8991 | +0.0187 [+0.0164, +0.0209] | 0.8991 | +0.0187 [+0.0164, +0.0209] | frac 1.000; rule_loss +0.0000 [+0.0000, +0.0000]; decision_loss 0.0011; oracle_cf - cos +0.0000; frac_cf 1.000 |
| 6 | 1.0 | 1.0 | 1 | 0 dB | ods | 0.8991 | +0.0187 [+0.0164, +0.0209] | 0.8991 | +0.0187 [+0.0164, +0.0209] |  |
| 6 | 1.0 | 1.0 | 1 | 0 dB | csiqb | 0.8838 | +0.0034 [+0.0013, +0.0055] | 0.8838 | +0.0034 [+0.0013, +0.0055] |  |
| 6 | 1.0 | 1.0 | 1 | 0 dB | csiqw | 0.8857 | +0.0053 [+0.0037, +0.0067] | 0.8857 | +0.0053 [+0.0037, +0.0067] |  |
| 6 | 1.0 | 1.0 | 1 | 10 dB | fixed | 0.9739 | - | 0.9739 | - |  |
| 6 | 1.0 | 1.0 | 1 | 10 dB | csiq | 0.9765 | +0.0026 [+0.0018, +0.0036] | 0.9765 | +0.0026 [+0.0018, +0.0036] |  |
| 6 | 1.0 | 1.0 | 1 | 10 dB | csiqp | 0.9765 | +0.0026 [+0.0018, +0.0036] | 0.9765 | +0.0026 [+0.0018, +0.0036] |  |
| 6 | 1.0 | 1.0 | 1 | 10 dB | switch | 0.9765 | +0.0026 [+0.0018, +0.0036] | 0.9765 | +0.0026 [+0.0018, +0.0036] | frac 1.000; rule_loss +0.0000 [+0.0000, +0.0000]; decision_loss 0.0004; oracle_cf - cos +0.0000; frac_cf 1.000 |
| 6 | 1.0 | 1.0 | 1 | 10 dB | ods | 0.9765 | +0.0026 [+0.0018, +0.0036] | 0.9765 | +0.0026 [+0.0018, +0.0036] |  |
| 6 | 1.0 | 1.0 | 1 | 10 dB | csiqb | 0.9676 | -0.0062 [-0.0070, -0.0053] | 0.9676 | -0.0062 [-0.0070, -0.0053] |  |
| 6 | 1.0 | 1.0 | 1 | 10 dB | csiqw | 0.9747 | +0.0009 [+0.0004, +0.0015] | 0.9747 | +0.0009 [+0.0004, +0.0015] |  |
| 6 | 1.0 | 1.0 | 1 | 20 dB | fixed | 0.9938 | - | 0.9938 | - |  |
| 6 | 1.0 | 1.0 | 1 | 20 dB | csiq | 0.9941 | +0.0003 [+0.0002, +0.0005] | 0.9941 | +0.0003 [+0.0002, +0.0005] |  |
| 6 | 1.0 | 1.0 | 1 | 20 dB | csiqp | 0.9941 | +0.0003 [+0.0002, +0.0005] | 0.9941 | +0.0003 [+0.0002, +0.0005] |  |
| 6 | 1.0 | 1.0 | 1 | 20 dB | switch | 0.9941 | +0.0003 [+0.0002, +0.0005] | 0.9941 | +0.0003 [+0.0002, +0.0005] | frac 1.000; rule_loss +0.0000 [+0.0000, +0.0000]; decision_loss 0.0001; oracle_cf - cos +0.0000; frac_cf 1.000 |
| 6 | 1.0 | 1.0 | 1 | 20 dB | ods | 0.9941 | +0.0003 [+0.0002, +0.0005] | 0.9941 | +0.0003 [+0.0002, +0.0005] |  |
| 6 | 1.0 | 1.0 | 1 | 20 dB | csiqb | 0.9908 | -0.0029 [-0.0031, -0.0027] | 0.9908 | -0.0029 [-0.0031, -0.0027] |  |
| 6 | 1.0 | 1.0 | 1 | 20 dB | csiqw | 0.9937 | -0.0000 [-0.0002, +0.0001] | 0.9937 | -0.0000 [-0.0002, +0.0001] |  |
| 6 | 1.0 | 1.0 | 2 | 0 dB | fixed | 0.8359 | - | 0.7587 | - |  |
| 6 | 1.0 | 1.0 | 2 | 0 dB | csiq | 0.8605 | +0.0246 [+0.0227, +0.0267] | 0.8070 | +0.0483 [+0.0448, +0.0521] |  |
| 6 | 1.0 | 1.0 | 2 | 0 dB | csiqp | 0.8605 | +0.0246 [+0.0227, +0.0267] | 0.8070 | +0.0483 [+0.0448, +0.0521] |  |
| 6 | 1.0 | 1.0 | 2 | 0 dB | switch | 0.8605 | +0.0246 [+0.0227, +0.0267] | 0.8070 | +0.0483 [+0.0448, +0.0521] | frac 1.000; rule_loss +0.0000 [+0.0000, +0.0000]; decision_loss 0.0009; oracle_cf - cos +0.0000; frac_cf 1.000 |
| 6 | 1.0 | 1.0 | 2 | 0 dB | ods | 0.8588 | +0.0229 [+0.0209, +0.0249] | 0.7980 | +0.0394 [+0.0358, +0.0430] |  |
| 6 | 1.0 | 1.0 | 2 | 0 dB | csiqb | 0.8416 | +0.0057 [+0.0041, +0.0075] | 0.7835 | +0.0248 [+0.0216, +0.0284] |  |
| 6 | 1.0 | 1.0 | 2 | 0 dB | csiqw | 0.8420 | +0.0061 [+0.0047, +0.0076] | 0.7760 | +0.0174 [+0.0147, +0.0203] |  |
| 6 | 1.0 | 1.0 | 2 | 10 dB | fixed | 0.9527 | - | 0.9223 | - |  |
| 6 | 1.0 | 1.0 | 2 | 10 dB | csiq | 0.9612 | +0.0086 [+0.0076, +0.0097] | 0.9429 | +0.0206 [+0.0185, +0.0227] |  |
| 6 | 1.0 | 1.0 | 2 | 10 dB | csiqp | 0.9612 | +0.0086 [+0.0076, +0.0097] | 0.9429 | +0.0206 [+0.0185, +0.0227] |  |
| 6 | 1.0 | 1.0 | 2 | 10 dB | switch | 0.9612 | +0.0086 [+0.0076, +0.0097] | 0.9429 | +0.0206 [+0.0185, +0.0227] | frac 1.000; rule_loss +0.0000 [+0.0000, +0.0000]; decision_loss 0.0002; oracle_cf - cos +0.0000; frac_cf 1.000 |
| 6 | 1.0 | 1.0 | 2 | 10 dB | ods | 0.9582 | +0.0055 [+0.0046, +0.0064] | 0.9330 | +0.0107 [+0.0090, +0.0126] |  |
| 6 | 1.0 | 1.0 | 2 | 10 dB | csiqb | 0.9496 | -0.0030 [-0.0040, -0.0020] | 0.9292 | +0.0069 [+0.0049, +0.0089] |  |
| 6 | 1.0 | 1.0 | 2 | 10 dB | csiqw | 0.9576 | +0.0049 [+0.0043, +0.0056] | 0.9360 | +0.0137 [+0.0122, +0.0151] |  |
| 6 | 1.0 | 1.0 | 2 | 20 dB | fixed | 0.9903 | - | 0.9837 | - |  |
| 6 | 1.0 | 1.0 | 2 | 20 dB | csiq | 0.9921 | +0.0017 [+0.0014, +0.0021] | 0.9877 | +0.0040 [+0.0033, +0.0048] |  |
| 6 | 1.0 | 1.0 | 2 | 20 dB | csiqp | 0.9921 | +0.0017 [+0.0014, +0.0021] | 0.9877 | +0.0040 [+0.0033, +0.0048] |  |
| 6 | 1.0 | 1.0 | 2 | 20 dB | switch | 0.9921 | +0.0017 [+0.0014, +0.0021] | 0.9877 | +0.0040 [+0.0033, +0.0048] | frac 1.000; rule_loss +0.0000 [+0.0000, +0.0000]; decision_loss 0.0000; oracle_cf - cos +0.0000; frac_cf 1.000 |
| 6 | 1.0 | 1.0 | 2 | 20 dB | ods | 0.9909 | +0.0006 [+0.0004, +0.0009] | 0.9849 | +0.0012 [+0.0007, +0.0017] |  |
| 6 | 1.0 | 1.0 | 2 | 20 dB | csiqb | 0.9877 | -0.0026 [-0.0029, -0.0023] | 0.9818 | -0.0018 [-0.0024, -0.0012] |  |
| 6 | 1.0 | 1.0 | 2 | 20 dB | csiqw | 0.9916 | +0.0013 [+0.0010, +0.0015] | 0.9868 | +0.0031 [+0.0026, +0.0037] |  |
| 6 | 1.0 | 1.0 | 4 | 0 dB | fixed | 0.7776 | - | 0.5787 | - |  |
| 6 | 1.0 | 1.0 | 4 | 0 dB | csiq | 0.8023 | +0.0246 [+0.0233, +0.0259] | 0.6456 | +0.0668 [+0.0629, +0.0708] |  |
| 6 | 1.0 | 1.0 | 4 | 0 dB | csiqp | 0.8023 | +0.0246 [+0.0233, +0.0259] | 0.6456 | +0.0668 [+0.0629, +0.0708] |  |
| 6 | 1.0 | 1.0 | 4 | 0 dB | switch | 0.8023 | +0.0246 [+0.0233, +0.0259] | 0.6456 | +0.0668 [+0.0629, +0.0708] | frac 1.000; rule_loss +0.0000 [+0.0000, +0.0000]; decision_loss 0.0004; oracle_cf - cos +0.0000; frac_cf 1.000 |
| 6 | 1.0 | 1.0 | 4 | 0 dB | ods | 0.8016 | +0.0240 [+0.0227, +0.0253] | 0.6440 | +0.0653 [+0.0610, +0.0695] |  |
| 6 | 1.0 | 1.0 | 4 | 0 dB | csiqb | 0.7799 | +0.0023 [+0.0016, +0.0029] | 0.6026 | +0.0238 [+0.0217, +0.0260] |  |
| 6 | 1.0 | 1.0 | 4 | 0 dB | csiqw | 0.7796 | +0.0019 [+0.0016, +0.0023] | 0.5852 | +0.0065 [+0.0047, +0.0081] |  |
| 6 | 1.0 | 1.0 | 4 | 10 dB | fixed | 0.9288 | - | 0.8255 | - |  |
| 6 | 1.0 | 1.0 | 4 | 10 dB | csiq | 0.9393 | +0.0105 [+0.0097, +0.0113] | 0.8742 | +0.0487 [+0.0457, +0.0517] |  |
| 6 | 1.0 | 1.0 | 4 | 10 dB | csiqp | 0.9393 | +0.0105 [+0.0097, +0.0113] | 0.8742 | +0.0487 [+0.0457, +0.0517] |  |
| 6 | 1.0 | 1.0 | 4 | 10 dB | switch | 0.9393 | +0.0105 [+0.0097, +0.0113] | 0.8742 | +0.0487 [+0.0457, +0.0517] | frac 1.000; rule_loss +0.0000 [+0.0000, +0.0000]; decision_loss 0.0000; oracle_cf - cos +0.0000; frac_cf 1.000 |
| 6 | 1.0 | 1.0 | 4 | 10 dB | ods | 0.9372 | +0.0084 [+0.0077, +0.0092] | 0.8549 | +0.0294 [+0.0267, +0.0323] |  |
| 6 | 1.0 | 1.0 | 4 | 10 dB | csiqb | 0.9264 | -0.0025 [-0.0031, -0.0017] | 0.8562 | +0.0307 [+0.0280, +0.0334] |  |
| 6 | 1.0 | 1.0 | 4 | 10 dB | csiqw | 0.9327 | +0.0039 [+0.0036, +0.0042] | 0.8513 | +0.0257 [+0.0246, +0.0270] |  |
| 6 | 1.0 | 1.0 | 4 | 20 dB | fixed | 0.9853 | - | 0.9597 | - |  |
| 6 | 1.0 | 1.0 | 4 | 20 dB | csiq | 0.9866 | +0.0014 [+0.0011, +0.0016] | 0.9683 | +0.0086 [+0.0074, +0.0097] |  |
| 6 | 1.0 | 1.0 | 4 | 20 dB | csiqp | 0.9866 | +0.0014 [+0.0011, +0.0016] | 0.9683 | +0.0086 [+0.0074, +0.0097] |  |
| 6 | 1.0 | 1.0 | 4 | 20 dB | switch | 0.9866 | +0.0014 [+0.0011, +0.0016] | 0.9683 | +0.0086 [+0.0074, +0.0097] | frac 1.000; rule_loss +0.0000 [+0.0000, +0.0000]; decision_loss 0.0000; oracle_cf - cos +0.0000; frac_cf 1.000 |
| 6 | 1.0 | 1.0 | 4 | 20 dB | ods | 0.9861 | +0.0009 [+0.0007, +0.0011] | 0.9631 | +0.0034 [+0.0027, +0.0041] |  |
| 6 | 1.0 | 1.0 | 4 | 20 dB | csiqb | 0.9806 | -0.0047 [-0.0049, -0.0044] | 0.9571 | -0.0026 [-0.0037, -0.0015] |  |
| 6 | 1.0 | 1.0 | 4 | 20 dB | csiqw | 0.9860 | +0.0008 [+0.0007, +0.0009] | 0.9661 | +0.0064 [+0.0057, +0.0072] |  |
| 6 | 1.0 | 1.0 | 6 | 0 dB | fixed | 0.7424 | - | 0.4591 | - |  |
| 6 | 1.0 | 1.0 | 6 | 0 dB | csiq | 0.7664 | +0.0240 [+0.0228, +0.0253] | 0.5314 | +0.0722 [+0.0664, +0.0779] |  |
| 6 | 1.0 | 1.0 | 6 | 0 dB | csiqp | 0.7664 | +0.0240 [+0.0228, +0.0253] | 0.5314 | +0.0722 [+0.0664, +0.0779] |  |
| 6 | 1.0 | 1.0 | 6 | 0 dB | switch | 0.7664 | +0.0240 [+0.0228, +0.0253] | 0.5314 | +0.0722 [+0.0664, +0.0779] | frac 1.000; rule_loss +0.0000 [+0.0000, +0.0000]; decision_loss 0.0015; oracle_cf - cos +0.0000; frac_cf 1.000 |
| 6 | 1.0 | 1.0 | 6 | 0 dB | ods | 0.7668 | +0.0244 [+0.0232, +0.0256] | 0.5463 | +0.0871 [+0.0811, +0.0927] |  |
| 6 | 1.0 | 1.0 | 6 | 0 dB | csiqb | 0.7430 | +0.0006 [-0.0007, +0.0018] | 0.4696 | +0.0105 [+0.0042, +0.0160] |  |
| 6 | 1.0 | 1.0 | 6 | 0 dB | csiqw | 0.7429 | +0.0005 [-0.0008, +0.0017] | 0.4520 | -0.0072 [-0.0138, -0.0014] |  |
| 6 | 1.0 | 1.0 | 6 | 10 dB | fixed | 0.9088 | - | 0.7482 | - |  |
| 6 | 1.0 | 1.0 | 6 | 10 dB | csiq | 0.9182 | +0.0094 [+0.0087, +0.0101] | 0.8069 | +0.0587 [+0.0552, +0.0625] |  |
| 6 | 1.0 | 1.0 | 6 | 10 dB | csiqp | 0.9182 | +0.0094 [+0.0087, +0.0101] | 0.8069 | +0.0587 [+0.0552, +0.0625] |  |
| 6 | 1.0 | 1.0 | 6 | 10 dB | switch | 0.9182 | +0.0094 [+0.0087, +0.0101] | 0.8069 | +0.0587 [+0.0552, +0.0625] | frac 1.000; rule_loss +0.0000 [+0.0000, +0.0000]; decision_loss 0.0004; oracle_cf - cos +0.0000; frac_cf 1.000 |
| 6 | 1.0 | 1.0 | 6 | 10 dB | ods | 0.9169 | +0.0081 [+0.0074, +0.0087] | 0.7876 | +0.0393 [+0.0361, +0.0429] |  |
| 6 | 1.0 | 1.0 | 6 | 10 dB | csiqb | 0.9058 | -0.0030 [-0.0036, -0.0024] | 0.7860 | +0.0377 [+0.0344, +0.0413] |  |
| 6 | 1.0 | 1.0 | 6 | 10 dB | csiqw | 0.9101 | +0.0013 [+0.0007, +0.0018] | 0.7679 | +0.0197 [+0.0167, +0.0228] |  |
| 6 | 1.0 | 1.0 | 6 | 20 dB | fixed | 0.9709 | - | 0.9265 | - |  |
| 6 | 1.0 | 1.0 | 6 | 20 dB | csiq | 0.9720 | +0.0011 [+0.0009, +0.0013] | 0.9363 | +0.0098 [+0.0085, +0.0114] |  |
| 6 | 1.0 | 1.0 | 6 | 20 dB | csiqp | 0.9720 | +0.0011 [+0.0009, +0.0013] | 0.9363 | +0.0098 [+0.0085, +0.0114] |  |
| 6 | 1.0 | 1.0 | 6 | 20 dB | switch | 0.9720 | +0.0011 [+0.0009, +0.0013] | 0.9363 | +0.0098 [+0.0085, +0.0114] | frac 1.000; rule_loss +0.0000 [+0.0000, +0.0000]; decision_loss 0.0002; oracle_cf - cos +0.0000; frac_cf 1.000 |
| 6 | 1.0 | 1.0 | 6 | 20 dB | ods | 0.9718 | +0.0008 [+0.0007, +0.0010] | 0.9313 | +0.0048 [+0.0039, +0.0060] |  |
| 6 | 1.0 | 1.0 | 6 | 20 dB | csiqb | 0.9666 | -0.0044 [-0.0046, -0.0042] | 0.9245 | -0.0020 [-0.0033, -0.0005] |  |
| 6 | 1.0 | 1.0 | 6 | 20 dB | csiqw | 0.9711 | +0.0002 [+0.0000, +0.0004] | 0.9313 | +0.0048 [+0.0037, +0.0059] |  |

## E. Numerical search against the closed-form masks (Code/overlap_bound.py)

Restricted model (use i of every slot mixes only the coefficients i of the groups: 128 problems of S = 6K uses), joint LMMSE receiver with known gains, per-user energy. closed = the closed form of the condition (load12 = load masks with (12); prop2 = Proposition 2; prop2_pruned = pruned Proposition 2 at U > K; csiq13 = channel-aware masks with (13) on the greedy support); support-restricted search = values optimized on the support of the load masks (csiq support for slot gains), the non-overlapping instance; search = any real S x 6U transmit matrix per use index (Adam from the closed form and 3 random starts, 3000 steps, best kept). gap = 100 (closed - search) / closed of the sum NMSE (positive: the search is lower). With Rayleigh draws the closed forms are designed at the unit gain while the search knows the drawn gains at the transmitter (the gap then contains transmitter gain knowledge).

### E.1 overlap_bound2 (generalized masks)

| setting | SNR | closed form | draws | sum NMSE load / closed / support-restricted search / search | gap closed -> search: mean / max / min over draws % | gap of the means % | advantage of closed over the support-restricted search 100 (support - closed)/support: mean / min % | free search beyond the support-restricted search 100 (support - search)/support: mean % | load masks above closed 100 (load - closed)/closed: mean % | overlap index t closed / search | search energy on shared slots / outside the closed support | closed-form checks |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| coco_equal_K2_U3 | -5 dB | prop2_pruned | 1 | 1.13308 / 1.10275 / 1.13162 / 1.10275 | -1.5e-11 / -1.5e-11 / -1.5e-11 | -1.5e-11 | +2.551 / +2.551 | +2.551 | +2.750 | 0.255 / 0.255 | 1.000 / 0.004 | split identity; modes kept 413; drops 0; energy dev before scaling 2.2e-16 |
| coco_equal_K2_U3 | 0 dB | prop2_pruned | 1 | 0.69697 / 0.68984 / 0.69661 / 0.68984 | -1.8e-10 / -1.8e-10 / -1.8e-10 | -1.8e-10 | +0.972 / +0.972 | +0.972 | +1.034 | 0.123 / 0.123 | 1.000 / 0.001 | split identity; modes kept 970; drops 0; energy dev before scaling 0.0e+00 |
| coco_equal_K2_U3 | 10 dB | prop2_pruned | 1 | 0.16569 / 0.16512 / 0.16569 / 0.16512 | -1.6e-11 / -1.6e-11 / -1.6e-11 | -1.6e-11 | +0.342 / +0.342 | +0.342 | +0.345 | 0.026 / 0.026 | 1.000 / 0.000 | split lpt; modes kept 1536; drops 765; energy dev before scaling 2.2e-16 |
| coco_equal_K2_U4 | -5 dB | prop2_pruned | 1 | 1.65145 / 1.58685 / 1.64736 / 1.58685 | -1.9e-10 / -1.9e-10 / -1.9e-10 | -1.9e-10 | +3.673 / +3.673 | +3.673 | +4.071 | 0.267 / 0.267 | 1.000 / 0.005 | split identity; modes kept 456; drops 0; energy dev before scaling 8.9e-16 |
| coco_equal_K2_U4 | 0 dB | prop2_pruned | 1 | 1.05715 / 1.04034 / 1.05608 / 1.04034 | -9.6e-09 / -9.6e-09 / -9.6e-09 | -9.6e-09 | +1.490 / +1.490 | +1.490 | +1.615 | 0.130 / 0.130 | 1.000 / 0.003 | split identity; modes kept 1051; drops 0; energy dev before scaling 1.3e-15 |
| coco_equal_K2_U4 | 10 dB | prop2_pruned | 1 | 0.29988 / 0.29770 / 0.29987 / 0.29770 | -1.9e-08 / -1.9e-08 / -1.9e-08 | -1.9e-08 | +0.723 / +0.723 | +0.723 | +0.731 | 0.033 / 0.033 | 1.000 / 0.000 | split lpt; modes kept 1536; drops 1532; energy dev before scaling 8.9e-16 |
| coco_equal_K4_U2 | -5 dB | prop2 | 1 | 0.48102 / 0.47762 / 0.48090 / 0.47762 | -6.6e-09 / -6.6e-09 / -6.6e-09 | -6.6e-09 | +0.684 / +0.684 | +0.684 | +0.712 | 0.149 / 0.149 | 0.140 / 0.000 | modal vs trace rel 3.5e-16; energy dev 1.4e-15 |
| coco_equal_K4_U2 | 0 dB | prop2 | 1 | 0.24855 / 0.24798 / 0.24853 / 0.24798 | -7.9e-11 / -7.9e-11 / -7.9e-11 | -7.9e-11 | +0.222 / +0.222 | +0.222 | +0.230 | 0.067 / 0.067 | 0.017 / 0.000 | modal vs trace rel 4.5e-16; energy dev 1.0e-15 |
| coco_equal_K4_U2 | 10 dB | prop2 | 1 | 0.03570 / 0.03569 / 0.03570 / 0.03569 | +5.8e-14 / +5.8e-14 / +5.8e-14 | +5.8e-14 | +0.005 / +0.005 | +0.005 | +0.005 | 0.007 / 0.007 | 0.000 / 0.000 | modal vs trace rel 0.0e+00; energy dev 8.9e-16 |
| coco_equal_K4_U4 | -5 dB | prop2 | 1 | 1.28399 / 1.25427 / 1.28208 / 1.25427 | -2.0e-09 / -2.0e-09 / -2.0e-09 | -2.0e-09 | +2.169 / +2.169 | +2.169 | +2.369 | 0.174 / 0.174 | 0.057 / 0.002 | modal vs trace rel 3.5e-16; energy dev 1.3e-15 |
| coco_equal_K4_U4 | 0 dB | prop2 | 1 | 0.74900 / 0.74238 / 0.74861 / 0.74238 | -8.4e-09 / -8.4e-09 / -8.4e-09 | -8.4e-09 | +0.833 / +0.833 | +0.833 | +0.892 | 0.084 / 0.084 | 0.059 / 0.001 | modal vs trace rel 7.5e-16; energy dev 1.0e-15 |
| coco_equal_K4_U4 | 10 dB | prop2 | 1 | 0.13574 / 0.13569 / 0.13573 / 0.13569 | +0.0e+00 / +0.0e+00 / +0.0e+00 | +0.0e+00 | +0.034 / +0.034 | +0.034 | +0.035 | 0.011 / 0.011 | 0.000 / 0.000 | modal vs trace rel 4.1e-16; energy dev 4.4e-16 |
| coco_rayleigh-D20_K4_U2 | -5 dB | prop2 | 20 | 0.62393 / 0.61955 / 0.59473 / 0.58929 | +4.276 / +9.854 / +0.025 | +4.884 | -3.716 / -10.317 | +0.824 | +0.711 | 0.149 / 0.385 | 0.811 / 0.433 | modal vs trace rel 3.5e-16; energy dev 1.4e-15 |
| coco_rayleigh-D20_K4_U2 | 0 dB | prop2 | 20 | 0.45499 / 0.45382 / 0.42210 / 0.41949 | +5.612 / +13.695 / +0.121 | +7.564 | -5.669 / -14.642 | +0.484 | +0.245 | 0.067 / 0.251 | 0.722 / 0.000 | modal vs trace rel 4.5e-16; energy dev 1.0e-15 |
| coco_rayleigh-D20_K4_U2 | 10 dB | prop2 | 20 | 0.10789 / 0.10789 / 0.10082 / 0.10074 | +2.940 / +16.409 / +0.007 | +6.626 | -3.227 / -19.607 | +0.033 | +0.002 | 0.007 / 0.049 | 0.551 / 0.000 | modal vs trace rel 0.0e+00; energy dev 8.9e-16 |
| coco_rayleigh-D20_K4_U4 | -5 dB | prop2 | 20 | 1.46829 / 1.43529 / 1.41992 / 1.38570 | +3.418 / +7.067 / +1.446 | +3.455 | -1.182 / -6.497 | +2.297 | +2.303 | 0.174 / 0.337 | 0.981 / 0.478 | modal vs trace rel 3.5e-16; energy dev 1.3e-15 |
| coco_rayleigh-D20_K4_U4 | 0 dB | prop2 | 20 | 1.22195 / 1.20983 / 1.15192 / 1.12767 | +5.837 / +12.177 / +1.743 | +6.791 | -4.392 / -8.871 | +1.775 | +0.971 | 0.084 / 0.266 | 0.957 / 0.159 | modal vs trace rel 7.5e-16; energy dev 1.0e-15 |
| coco_rayleigh-D20_K4_U4 | 10 dB | prop2 | 20 | 0.31687 / 0.31673 / 0.29817 / 0.29757 | +4.242 / +11.515 / +0.045 | +6.047 | -4.427 / -12.611 | +0.152 | +0.038 | 0.011 / 0.060 | 0.900 / 0.000 | modal vs trace rel 4.1e-16; energy dev 4.4e-16 |
| indep_equal_K4_U2 | -5 dB | load12 | 1 | 0.59908 / 0.59908 / 0.59908 / 0.59908 | -5.8e-10 / -5.8e-10 / -5.8e-10 | -5.8e-10 | +2.8e-10 / +2.8e-10 | -2.9e-10 | +0.0e+00 | 0.000 / -0.000 | 0.001 / 0.001 |  |
| indep_equal_K4_U2 | 0 dB | load12 | 1 | 0.31017 / 0.31017 / 0.31017 / 0.31017 | -2.2e-12 / -2.2e-12 / -2.2e-12 | -2.2e-12 | +1.8e-14 / +1.8e-14 | -2.2e-12 | +0.0e+00 | 0.000 / 0.000 | 0.001 / 0.000 |  |
| indep_equal_K4_U2 | 10 dB | load12 | 1 | 0.04426 / 0.04426 / 0.04426 / 0.04426 | +1.7e-13 / +1.7e-13 / +1.7e-13 | +1.7e-13 | -1.4e-13 / -1.4e-13 | +3.1e-14 | +0.0e+00 | 0.000 / -0.000 | 1.000 / 0.504 |  |
| indep_equal_K4_U4 | -5 dB | load12 | 1 | 1.61698 / 1.61698 / 1.61698 / 1.61698 | +1.4e-14 / +1.4e-14 / +1.4e-14 | +1.4e-14 | -2.7e-14 / -2.7e-14 | -1.4e-14 | +0.0e+00 | 0.000 / -0.000 | 0.004 / 0.005 |  |
| indep_equal_K4_U4 | 0 dB | load12 | 1 | 0.94664 / 0.94664 / 0.94664 / 0.94664 | +5.9e-14 / +5.9e-14 / +5.9e-14 | +5.9e-14 | -5.9e-14 / -5.9e-14 | +0.0e+00 | +0.0e+00 | 0.000 / -0.000 | 0.010 / 0.003 |  |
| indep_equal_K4_U4 | 10 dB | load12 | 1 | 0.16898 / 0.16898 / 0.16898 / 0.16898 | +4.9e-14 / +4.9e-14 / +4.9e-14 | +4.9e-14 | -1.6e-14 / -1.6e-14 | +3.3e-14 | +0.0e+00 | 0.000 / -0.000 | 0.000 / 0.000 |  |
| indep_slot-rho0.7-D20_K4_U2 | 0 dB | csiq13 | 20 | 0.38971 / 0.14966 / 0.14966 / 0.14958 | +0.052 / +0.300 / -6.5e-10 | +0.049 | +4.0e-14 / -1.4e-13 | +0.052 | +169.074 | 0.000 / -0.000 | 0.004 / 0.015 | csiq subbands per user 12.15/11.85 |
| indep_slot-rho0.7-D20_K4_U2 | 10 dB | csiq13 | 20 | 0.10147 / 0.02010 / 0.02010 / 0.02005 | +0.241 / +4.813 / -5.4e-12 | +0.257 | +1.3e-13 / -1.4e-13 | +0.241 | +375.834 | 0.000 / -0.000 | 0.023 / 0.010 | csiq subbands per user 11.95/12.05 |
| indep_slot-rho0.7-D20_K4_U2 | 20 dB | csiq13 | 20 | 0.00837 / 0.00196 / 0.00196 / 0.00196 | +0.034 / +0.674 / -3.6e-10 | +0.034 | +1.9e-11 / -1.1e-13 | +0.034 | +325.497 | 0.000 / -0.000 | 0.001 / 0.004 | csiq subbands per user 11.95/12.05 |
| indep_slot-rho0.7-D20_K4_U2_nf6 | 0 dB | csiq13 | 20 | 0.39348 / 0.15230 / 0.15230 / 0.15211 | +0.126 / +1.310 / -5.6e-10 | +0.128 | +9.3e-14 / -1.4e-13 | +0.126 | +168.243 | 0.000 / -0.000 | 0.010 / 0.020 | csiq subbands per user 12.05/11.95 |
| indep_slot-rho0.7-D20_K4_U2_nf6 | 10 dB | csiq13 | 20 | 0.10717 / 0.02473 / 0.02473 / 0.02436 | +1.409 / +18.519 / -1.3e-11 | +1.506 | +2.8e-13 / -1.5e-13 | +1.409 | +359.275 | 0.000 / -0.000 | 0.034 / 0.032 | csiq subbands per user 10.65/13.35 |
| indep_slot-rho0.7-D20_K4_U2_nf6 | 20 dB | csiq13 | 20 | 0.01166 / 0.00239 / 0.00239 / 0.00238 | +0.510 / +5.607 / -5.4e-09 | +0.396 | +1.2e-10 / -1.2e-13 | +0.510 | +349.812 | 0.000 / 0.000 | 0.001 / 0.015 | csiq subbands per user 11.90/12.10 |
| indep_slot-rho0.7-D20_K4_U4 | 0 dB | csiq13 | 20 | 1.23242 / 0.50963 / 0.50963 / 0.50769 | +0.384 / +1.552 / -1.1e-08 | +0.380 | +3.2e-12 / -7.0e-14 | +0.384 | +144.231 | 0.000 / -0.000 | 0.078 / 0.047 | csiq subbands per user 6.15/5.95/6.15/5.75 |
| indep_slot-rho0.7-D20_K4_U4 | 10 dB | csiq13 | 20 | 0.45327 / 0.08599 / 0.08599 / 0.08468 | +1.518 / +4.772 / -3.6e-09 | +1.528 | +4.7e-12 / -8.7e-14 | +1.518 | +425.885 | 0.000 / -0.000 | 0.034 / 0.056 | csiq subbands per user 6.05/5.95/6.00/6.00 |
| indep_slot-rho0.7-D20_K4_U4 | 20 dB | csiq13 | 20 | 0.07646 / 0.00933 / 0.00933 / 0.00899 | +3.152 / +10.525 / -4.2e-12 | +3.628 | +8.2e-11 / -9.8e-14 | +3.152 | +696.455 | 0.000 / -0.000 | 0.017 / 0.071 | csiq subbands per user 6.00/6.00/6.00/6.00 |
| indep_slot-rho0.7-D20_K4_U4_nf6 | 0 dB | csiq13 | 20 | 1.22996 / 0.50709 / 0.50709 / 0.50207 | +1.053 / +3.091 / +0.003 | +0.991 | +5.7e-12 / -8.6e-14 | +1.053 | +150.740 | 0.000 / 0.000 | 0.083 / 0.071 | csiq subbands per user 6.20/5.70/5.60/6.50 |
| indep_slot-rho0.7-D20_K4_U4_nf6 | 10 dB | csiq13 | 20 | 0.49450 / 0.10366 / 0.10366 / 0.10049 | +2.732 / +11.342 / -2.2e-10 | +3.058 | +1.0e-11 / -1.0e-13 | +2.732 | +361.791 | 0.000 / 0.000 | 0.054 / 0.073 | csiq subbands per user 6.15/6.05/6.00/5.80 |
| indep_slot-rho0.7-D20_K4_U4_nf6 | 20 dB | csiq13 | 20 | 0.09547 / 0.01322 / 0.01322 / 0.01248 | +5.277 / +19.942 / +7.8e-14 | +5.567 | +7.7e-11 / -1.0e-13 | +5.277 | +642.841 | 0.000 / -0.000 | 0.101 / 0.096 | csiq subbands per user 6.00/6.00/6.00/6.00 |
| mix0.5_equal_K4_U2 | -5 dB | prop2 | 1 | 0.47331 / 0.45050 / 0.47280 / 0.45050 | -1.2e-08 / -1.2e-08 / -1.2e-08 | -1.2e-08 | +4.718 / +4.718 | +4.718 | +5.064 | 0.342 / 0.342 | 1.000 / 0.000 | modal vs trace rel 8.6e-16; energy dev 1.3e-15 |
| mix0.5_equal_K4_U2 | 0 dB | prop2 | 1 | 0.24180 / 0.23246 / 0.24164 / 0.23246 | -1.1e-08 / -1.1e-08 / -1.1e-08 | -1.1e-08 | +3.800 / +3.800 | +3.800 | +4.020 | 0.233 / 0.233 | 1.000 / 0.000 | modal vs trace rel 1.2e-16; energy dev 1.7e-15 |
| mix0.5_equal_K4_U2 | 10 dB | prop2 | 1 | 0.03541 / 0.03536 / 0.03541 / 0.03536 | +9.8e-14 / +9.8e-14 / +9.8e-14 | +9.8e-14 | +0.148 / +0.148 | +0.148 | +0.156 | 0.031 / 0.031 | 0.000 / 0.000 | modal vs trace rel 2.0e-16; energy dev 3.3e-16 |
| mix0.5_equal_K4_U4 | -5 dB | prop2 | 1 | 1.25061 / 1.12002 / 1.24254 / 1.12002 | -8.1e-08 / -8.1e-08 / -8.1e-08 | -8.1e-08 | +9.860 / +9.860 | +9.860 | +11.660 | 0.337 / 0.337 | 1.000 / 0.000 | modal vs trace rel 5.9e-16; energy dev 3.3e-16 |
| mix0.5_equal_K4_U4 | 0 dB | prop2 | 1 | 0.71296 / 0.64956 / 0.70915 / 0.64956 | -1.4e-07 / -1.4e-07 / -1.4e-07 | -1.4e-07 | +8.404 / +8.404 | +8.404 | +9.761 | 0.224 / 0.224 | 1.000 / 0.000 | modal vs trace rel 0.0e+00; energy dev 4.4e-16 |
| mix0.5_equal_K4_U4 | 10 dB | prop2 | 1 | 0.13187 / 0.13091 / 0.13178 / 0.13091 | +8.5e-14 / +8.5e-14 / +8.5e-14 | +8.5e-14 | +0.658 / +0.658 | +0.658 | +0.731 | 0.038 / 0.038 | 1.000 / 0.000 | modal vs trace rel 4.2e-16; energy dev 3.3e-16 |

- equal (deterministic) gains, closed form designed for the actual gains: largest per-draw gap +1.7e-13 % (indep_equal_K4_U2, 10 dB); largest mean gap +1.7e-13 % (indep_equal_K4_U2, 10 dB); largest mean gain of the free search over the support-restricted search +9.860 %; settings x SNRs: 24
- Rayleigh draws, closed form at the unit design gain, search with the drawn gains known at the transmitter: largest per-draw gap +16.409 % (coco_rayleigh-D20_K4_U2, 10 dB); largest mean gap +5.837 % (coco_rayleigh-D20_K4_U4, 0 dB); largest mean gain of the free search over the support-restricted search +2.297 %; settings x SNRs: 6
- AR(1) subband gains, channel-aware closed form csiq13 against the search with the same gains: largest per-draw gap +19.942 % (indep_slot-rho0.7-D20_K4_U4_nf6, 20 dB); largest mean gap +5.277 % (indep_slot-rho0.7-D20_K4_U4_nf6, 20 dB); largest mean gain of the free search over the support-restricted search +5.277 %; settings x SNRs: 12

### E.2 overlap_bound (AG News, load masks against any real overlapping masks, deterministic gains)

| U, offsets | SNR | per-user NMSE mean: closed / support-restricted / search | worst user: closed / search | gap mean % | gap worst user % | search energy on shared slots (max over users) |
|---|---|---|---|---|---|---|
| U = 2, +-0 dB | 0 dB | 0.155084 / 0.155084 / 0.155084 | 0.155084 / 0.155084 | -3.1e-10 | -3.5e-10 | 0.00e+00 |
| U = 2, +-0 dB | 10 dB | 0.022128 / 0.022128 / 0.022128 | 0.022128 / 0.022128 | +1.6e-13 | +1.4e-13 | 0.00e+00 |
| U = 2, +-0 dB | 20 dB | 0.002312 / 0.002312 / 0.002312 | 0.002312 / 0.002312 | -9.0e-09 | -9.8e-09 | 0.00e+00 |
| U = 2, +-6 dB | 0 dB | 0.192579 / 0.192579 / 0.192579 | 0.333303 / 0.333303 | -3.2e-12 | -3.7e-12 | 4.33e-07 |
| U = 2, +-6 dB | 10 dB | 0.041458 / 0.041458 / 0.041458 | 0.077153 / 0.077153 | -2.4e-12 | +5.4e-14 | 0.00e+00 |
| U = 2, +-6 dB | 20 dB | 0.004826 / 0.004826 / 0.004826 | 0.009069 / 0.009069 | -2.1e-08 | -9.2e-11 | 0.00e+00 |
| U = 4, +-0 dB | 0 dB | 0.236659 / 0.236659 / 0.236659 | 0.236659 / 0.236659 | -4.4e-11 | -5.7e-11 | 9.71e-07 |
| U = 4, +-0 dB | 10 dB | 0.042246 / 0.042246 / 0.042246 | 0.042246 / 0.042246 | +8.2e-14 | +6.6e-14 | 0.00e+00 |
| U = 4, +-0 dB | 20 dB | 0.004601 / 0.004601 / 0.004601 | 0.004601 / 0.004601 | -7.5e-12 | -1.5e-11 | 0.00e+00 |
| U = 4, +-6 dB | 0 dB | 0.253510 / 0.253510 / 0.253510 | 0.440406 / 0.440406 | -2.3e-11 | -1.9e-12 | 1.08e-06 |
| U = 4, +-6 dB | 10 dB | 0.058729 / 0.058729 / 0.058729 | 0.132372 / 0.132372 | -7.9e-12 | -1.4e-11 | 0.00e+00 |
| U = 4, +-6 dB | 20 dB | 0.007279 / 0.007279 / 0.007279 | 0.017791 / 0.017791 | -4.8e-10 | +1.2e-13 | 0.00e+00 |
| U = 6, +-0 dB (overload) | 0 dB | 0.291998 / 0.291998 / 0.291998 | 0.291998 / 0.291998 | -8.8e-12 | -1.2e-11 | 1.14e-06 |
| U = 6, +-0 dB (overload) | 10 dB | 0.069848 / 0.069848 / 0.069848 | 0.069848 / 0.069848 | +0.0e+00 | -2.0e-14 | 0.00e+00 |
| U = 6, +-0 dB (overload) | 20 dB | 0.026139 / 0.026139 / 0.026139 | 0.026139 / 0.026139 | +4.0e-14 | +0.0e+00 | 0.00e+00 |
| U = 6, +-6 dB (overload) | 0 dB | 0.303544 / 0.303544 / 0.303215 | 0.504078 / 0.504078 | +0.109 | -8.3e-09 | 7.63e-02 |
| U = 6, +-6 dB (overload) | 10 dB | 0.085944 / 0.085944 / 0.084054 | 0.174436 / 0.176872 | +2.200 | -1.397 | 1.00e+00 |
| U = 6, +-6 dB (overload) | 20 dB | 0.028717 / 0.028717 / 0.028717 | 0.041455 / 0.041455 | +0.0e+00 | -1.7e-14 | 0.00e+00 |
| U = 8, +-0 dB (overload) | 0 dB | 0.334012 / 0.334012 / 0.334012 | 0.334012 / 0.334012 | -6.8e-12 | -1.1e-11 | 0.00e+00 |
| U = 8, +-0 dB (overload) | 10 dB | 0.096278 / 0.096278 / 0.096278 | 0.096278 / 0.096278 | +4.3e-14 | +1.4e-14 | 0.00e+00 |
| U = 8, +-0 dB (overload) | 20 dB | 0.049329 / 0.049329 / 0.049329 | 0.049329 / 0.049329 | +0.0e+00 | -4.2e-14 | 0.00e+00 |
| U = 8, +-6 dB (overload) | 0 dB | 0.342588 / 0.342588 / 0.341910 | 0.548349 / 0.548349 | +0.198 | -3.7e-08 | 9.98e-01 |
| U = 8, +-6 dB (overload) | 10 dB | 0.112174 / 0.112174 / 0.110374 | 0.208620 / 0.215322 | +1.604 | -3.213 | 9.99e-01 |
| U = 8, +-6 dB (overload) | 20 dB | 0.051855 / 0.051855 / 0.051855 | 0.065780 / 0.065780 | +1.3e-14 | +0.0e+00 | 0.00e+00 |

- largest mean gap closed -> search over U = 2, 4, 6, 8, offsets 0 / +-6 dB, 0/10/20 dB: +2.200 % (U = 6, +-6 dB, 10 dB); smallest -2.1e-08 %
- up to full load (U = 2, 4), every offset and SNR: |gap| <= 2.1e-08 % (the load masks are the optimum of the search for independent content)
- overload with +-6 dB offsets: U = 6, 0 dB: mean gap +0.109 %, worst user -8.3e-09 %; U = 6, 10 dB: mean gap +2.200 %, worst user -1.397 %; U = 8, 0 dB: mean gap +0.198 %, worst user -3.7e-08 %; U = 8, 10 dB: mean gap +1.604 %, worst user -3.213 %

## F. Gain estimation for the overlapping masks (Code/shared_pilot.py)


### coco_K2_U2_E1_emauto0_fs_it30: frames 2000; data/shared/pilot/shared_pilot_coco_K2_U2_E1_emauto0_fs_it30_snr-5.json, data/shared/pilot/shared_pilot_coco_K2_U2_E1_emauto0_fs_it30_snr0.json, data/shared/pilot/shared_pilot_coco_K2_U2_E1_emauto0_fs_it30_snr10.json, data/shared/pilot/shared_pilot_coco_K2_U2_E1_emauto0_fs_it30_snr20.json

| mode | SNR | CosSim | weakest | NMSE | gain rel. RMSE | - load_joint_blind CosSim [95% CI] | - load_joint (true g) CosSim [95% CI] | - load_joint_blind NMSE [95% CI] |
|---|---|---|---|---|---|---|---|---|
| gen_joint_em | -5 dB | 0.7656 | 0.7054 | 0.40337 | 0.1804 | +0.0141 [+0.0125, +0.0157] | +0.0021 [+0.0014, +0.0026] | -0.0164 [-0.0181, -0.0146] |
| gen_joint_em | 0 dB | 0.8483 | 0.7977 | 0.27625 | 0.1437 | +0.0043 [+0.0034, +0.0051] | +0.0006 [+0.0003, +0.0009] | -0.0023 [-0.0034, -0.0013] |
| gen_joint_em | 10 dB | 0.9540 | 0.9283 | 0.10031 | 0.1278 | +0.0006 [+0.0003, +0.0008] | +0.0000 [-0.0001, +0.0001] | +0.0070 [+0.0063, +0.0077] |
| gen_joint_em | 20 dB | 0.9915 | 0.9854 | 0.03291 | 0.1153 | +0.0002 [+0.0001, +0.0004] | +0.0000 [-0.0000, +0.0000] | +0.0082 [+0.0076, +0.0088] |
| gen_joint_em0 | -5 dB | 0.7657 | 0.7056 | 0.40319 | 0.1820 | +0.0143 [+0.0126, +0.0159] | +0.0022 [+0.0016, +0.0027] | -0.0165 [-0.0183, -0.0147] |
| gen_joint_em0 | 0 dB | 0.8484 | 0.7978 | 0.27616 | 0.1420 | +0.0043 [+0.0035, +0.0052] | +0.0007 [+0.0004, +0.0010] | -0.0024 [-0.0035, -0.0014] |
| gen_joint_em0 | 10 dB | 0.9540 | 0.9284 | 0.10055 | 0.1291 | +0.0006 [+0.0003, +0.0009] | +0.0000 [-0.0001, +0.0002] | +0.0072 [+0.0065, +0.0080] |
| gen_joint_em0 | 20 dB | 0.9915 | 0.9854 | 0.03503 | 0.1229 | +0.0002 [+0.0001, +0.0004] | +0.0000 [-0.0000, +0.0000] | +0.0103 [+0.0096, +0.0110] |
| gen_joint_em0_rho | -5 dB | 0.7651 | 0.7046 | 0.40159 | 0.1638 | +0.0137 [+0.0121, +0.0153] | +0.0016 [+0.0009, +0.0022] | -0.0181 [-0.0198, -0.0165] |
| gen_joint_em0_rho | 0 dB | 0.8474 | 0.7956 | 0.27376 | 0.1010 | +0.0034 [+0.0026, +0.0042] | -0.0003 [-0.0007, +0.0002] | -0.0048 [-0.0058, -0.0039] |
| gen_joint_em0_rho | 10 dB | 0.9534 | 0.9273 | 0.09315 | 0.0373 | -0.0001 [-0.0002, +0.0001] | -0.0006 [-0.0009, -0.0003] | -0.0002 [-0.0003, +0.0000] |
| gen_joint_em0_rho | 20 dB | 0.9913 | 0.9850 | 0.02464 | 0.0089 | +0.0000 [-0.0000, +0.0001] | -0.0002 [-0.0003, -0.0001] | -0.0001 [-0.0002, +0.0000] |
| gen_joint_em_rho | -5 dB | 0.7650 | 0.7045 | 0.40174 | 0.1620 | +0.0136 [+0.0120, +0.0152] | +0.0015 [+0.0008, +0.0022] | -0.0180 [-0.0197, -0.0163] |
| gen_joint_em_rho | 0 dB | 0.8473 | 0.7955 | 0.27390 | 0.1028 | +0.0033 [+0.0025, +0.0041] | -0.0004 [-0.0008, +0.0001] | -0.0047 [-0.0057, -0.0038] |
| gen_joint_em_rho | 10 dB | 0.9533 | 0.9272 | 0.09322 | 0.0373 | -0.0001 [-0.0002, +0.0000] | -0.0006 [-0.0009, -0.0004] | -0.0001 [-0.0003, +0.0001] |
| gen_joint_em_rho | 20 dB | 0.9913 | 0.9850 | 0.02466 | 0.0089 | +0.0000 [-0.0001, +0.0001] | -0.0002 [-0.0003, -0.0001] | -0.0000 [-0.0001, +0.0000] |
| gen_joint_pilot | -5 dB | 0.7066 | 0.6314 | 0.57204 | 1.8996 | -0.0449 [-0.0477, -0.0424] | -0.0570 [-0.0592, -0.0548] | +0.1523 [+0.1470, +0.1583] |
| gen_joint_pilot | 0 dB | 0.8036 | 0.7348 | 0.46294 | 1.4048 | -0.0404 [-0.0425, -0.0383] | -0.0441 [-0.0461, -0.0421] | +0.1843 [+0.1738, +0.1952] |
| gen_joint_pilot | 10 dB | 0.9448 | 0.9127 | 0.24961 | 0.5194 | -0.0087 [-0.0096, -0.0078] | -0.0092 [-0.0102, -0.0083] | +0.1563 [+0.1431, +0.1699] |
| gen_joint_pilot | 20 dB | 0.9910 | 0.9844 | 0.05712 | 0.1587 | -0.0003 [-0.0005, -0.0001] | -0.0005 [-0.0007, -0.0004] | +0.0324 [+0.0272, +0.0380] |
| gen_joint_true | -5 dB | 0.7674 | 0.7089 | 0.39989 | - | +0.0160 [+0.0142, +0.0177] | +0.0039 [+0.0037, +0.0041] | -0.0198 [-0.0217, -0.0180] |
| gen_joint_true | 0 dB | 0.8486 | 0.7987 | 0.27188 | - | +0.0045 [+0.0037, +0.0055] | +0.0009 [+0.0008, +0.0010] | -0.0067 [-0.0078, -0.0057] |
| gen_joint_true | 10 dB | 0.9540 | 0.9284 | 0.08650 | - | +0.0005 [+0.0003, +0.0008] | -0.0000 [-0.0000, -0.0000] | -0.0068 [-0.0074, -0.0063] |
| gen_joint_true | 20 dB | 0.9915 | 0.9854 | 0.01640 | - | +0.0002 [+0.0001, +0.0003] | -0.0000 [-0.0000, -0.0000] | -0.0083 [-0.0087, -0.0079] |
| gen_joint_true0 | -5 dB | 0.7675 | 0.7090 | 0.39972 | - | +0.0161 [+0.0143, +0.0178] | +0.0040 [+0.0038, +0.0042] | -0.0200 [-0.0219, -0.0182] |
| gen_joint_true0 | 0 dB | 0.8487 | 0.7988 | 0.27175 | - | +0.0046 [+0.0038, +0.0055] | +0.0010 [+0.0009, +0.0010] | -0.0068 [-0.0079, -0.0059] |
| gen_joint_true0 | 10 dB | 0.9540 | 0.9284 | 0.08643 | - | +0.0006 [+0.0003, +0.0008] | +0.0000 [+0.0000, +0.0000] | -0.0069 [-0.0074, -0.0064] |
| gen_joint_true0 | 20 dB | 0.9915 | 0.9854 | 0.01638 | - | +0.0002 [+0.0001, +0.0004] | +0.0000 [-0.0000, +0.0000] | -0.0083 [-0.0087, -0.0079] |
| gen_joint_true0_rho | -5 dB | 0.7674 | 0.7086 | 0.40028 | - | +0.0160 [+0.0142, +0.0177] | +0.0039 [+0.0036, +0.0042] | -0.0194 [-0.0213, -0.0177] |
| gen_joint_true0_rho | 0 dB | 0.8483 | 0.7975 | 0.27370 | - | +0.0043 [+0.0035, +0.0052] | +0.0007 [+0.0004, +0.0010] | -0.0049 [-0.0059, -0.0040] |
| gen_joint_true0_rho | 10 dB | 0.9536 | 0.9278 | 0.09326 | - | +0.0002 [+0.0001, +0.0003] | -0.0004 [-0.0006, -0.0001] | -0.0001 [-0.0002, +0.0001] |
| gen_joint_true0_rho | 20 dB | 0.9913 | 0.9851 | 0.02470 | - | +0.0001 [-0.0000, +0.0001] | -0.0002 [-0.0003, -0.0001] | +0.0000 [-0.0001, +0.0001] |
| gen_joint_true_rho | -5 dB | 0.7673 | 0.7085 | 0.40045 | - | +0.0159 [+0.0141, +0.0176] | +0.0038 [+0.0035, +0.0041] | -0.0193 [-0.0211, -0.0175] |
| gen_joint_true_rho | 0 dB | 0.8483 | 0.7974 | 0.27383 | - | +0.0042 [+0.0034, +0.0051] | +0.0006 [+0.0003, +0.0009] | -0.0048 [-0.0057, -0.0039] |
| gen_joint_true_rho | 10 dB | 0.9536 | 0.9277 | 0.09332 | - | +0.0001 [+0.0000, +0.0003] | -0.0004 [-0.0006, -0.0002] | +0.0000 [-0.0002, +0.0002] |
| gen_joint_true_rho | 20 dB | 0.9913 | 0.9851 | 0.02472 | - | +0.0000 [-0.0000, +0.0001] | -0.0002 [-0.0003, -0.0001] | +0.0000 [-0.0001, +0.0001] |
| load_joint | -5 dB | 0.7635 | 0.7044 | 0.40560 | - | +0.0121 [+0.0104, +0.0138] | +0.0000 [+0.0000, +0.0000] | -0.0141 [-0.0159, -0.0124] |
| load_joint | 0 dB | 0.8477 | 0.7976 | 0.27331 | - | +0.0036 [+0.0028, +0.0045] | +0.0000 [+0.0000, +0.0000] | -0.0053 [-0.0064, -0.0043] |
| load_joint | 10 dB | 0.9540 | 0.9284 | 0.08647 | - | +0.0005 [+0.0003, +0.0008] | +0.0000 [+0.0000, +0.0000] | -0.0069 [-0.0074, -0.0063] |
| load_joint | 20 dB | 0.9915 | 0.9854 | 0.01638 | - | +0.0002 [+0.0001, +0.0004] | +0.0000 [+0.0000, +0.0000] | -0.0083 [-0.0087, -0.0079] |
| load_joint_blind | -5 dB | 0.7515 | 0.6822 | 0.41972 | 0.3205 | +0.0000 [+0.0000, +0.0000] | -0.0121 [-0.0138, -0.0104] | +0.0000 [+0.0000, +0.0000] |
| load_joint_blind | 0 dB | 0.8440 | 0.7903 | 0.27859 | 0.1489 | +0.0000 [+0.0000, +0.0000] | -0.0036 [-0.0045, -0.0028] | +0.0000 [+0.0000, +0.0000] |
| load_joint_blind | 10 dB | 0.9534 | 0.9274 | 0.09332 | 0.0287 | +0.0000 [+0.0000, +0.0000] | -0.0005 [-0.0008, -0.0003] | +0.0000 [+0.0000, +0.0000] |
| load_joint_blind | 20 dB | 0.9913 | 0.9850 | 0.02470 | 0.0076 | +0.0000 [+0.0000, +0.0000] | -0.0002 [-0.0004, -0.0001] | +0.0000 [+0.0000, +0.0000] |
| load_joint_em | -5 dB | 0.7615 | 0.7005 | 0.40909 | 0.1816 | +0.0100 [+0.0085, +0.0116] | -0.0021 [-0.0027, -0.0014] | -0.0106 [-0.0124, -0.0090] |
| load_joint_em | 0 dB | 0.8474 | 0.7967 | 0.27767 | 0.1411 | +0.0034 [+0.0026, +0.0043] | -0.0003 [-0.0006, +0.0000] | -0.0009 [-0.0020, +0.0002] |
| load_joint_em | 10 dB | 0.9540 | 0.9283 | 0.10055 | 0.1287 | +0.0006 [+0.0003, +0.0008] | +0.0000 [-0.0001, +0.0001] | +0.0072 [+0.0065, +0.0080] |
| load_joint_em | 20 dB | 0.9915 | 0.9854 | 0.03500 | 0.1228 | +0.0002 [+0.0001, +0.0004] | +0.0000 [-0.0000, +0.0000] | +0.0103 [+0.0096, +0.0110] |
| load_joint_em_rho | -5 dB | 0.7609 | 0.6995 | 0.40746 | 0.1627 | +0.0094 [+0.0079, +0.0110] | -0.0026 [-0.0034, -0.0019] | -0.0123 [-0.0139, -0.0107] |
| load_joint_em_rho | 0 dB | 0.8465 | 0.7946 | 0.27519 | 0.0995 | +0.0024 [+0.0016, +0.0033] | -0.0012 [-0.0016, -0.0008] | -0.0034 [-0.0044, -0.0025] |
| load_joint_em_rho | 10 dB | 0.9534 | 0.9273 | 0.09317 | 0.0373 | -0.0001 [-0.0002, +0.0000] | -0.0006 [-0.0009, -0.0004] | -0.0002 [-0.0003, +0.0001] |
| load_joint_em_rho | 20 dB | 0.9913 | 0.9850 | 0.02464 | 0.0089 | +0.0000 [-0.0000, +0.0001] | -0.0002 [-0.0003, -0.0001] | -0.0001 [-0.0002, +0.0000] |
| load_joint_rho | -5 dB | 0.7635 | 0.7041 | 0.40606 | - | +0.0120 [+0.0103, +0.0137] | -0.0000 [-0.0003, +0.0002] | -0.0137 [-0.0154, -0.0119] |
| load_joint_rho | 0 dB | 0.8474 | 0.7964 | 0.27516 | - | +0.0033 [+0.0026, +0.0042] | -0.0003 [-0.0006, -0.0000] | -0.0034 [-0.0044, -0.0026] |
| load_joint_rho | 10 dB | 0.9536 | 0.9277 | 0.09327 | - | +0.0002 [+0.0001, +0.0003] | -0.0004 [-0.0006, -0.0002] | -0.0001 [-0.0002, +0.0001] |
| load_joint_rho | 20 dB | 0.9913 | 0.9851 | 0.02470 | - | +0.0001 [-0.0000, +0.0001] | -0.0002 [-0.0003, -0.0001] | +0.0000 [-0.0001, +0.0001] |
| load_pu | -5 dB | 0.7354 | 0.6529 | 0.43685 | - | -0.0160 [-0.0183, -0.0136] | -0.0281 [-0.0300, -0.0262] | +0.0171 [+0.0149, +0.0194] |
| load_pu | 0 dB | 0.8296 | 0.7639 | 0.29545 | - | -0.0144 [-0.0162, -0.0129] | -0.0181 [-0.0198, -0.0167] | +0.0169 [+0.0153, +0.0185] |
| load_pu | 10 dB | 0.9494 | 0.9195 | 0.09246 | - | -0.0040 [-0.0050, -0.0032] | -0.0046 [-0.0055, -0.0038] | -0.0009 [-0.0019, +0.0002] |
| load_pu | 20 dB | 0.9909 | 0.9843 | 0.01715 | - | -0.0003 [-0.0006, -0.0001] | -0.0006 [-0.0008, -0.0003] | -0.0075 [-0.0080, -0.0070] |
| load_pu_blind | -5 dB | 0.7342 | 0.6507 | 0.45445 | 0.3205 | -0.0173 [-0.0196, -0.0149] | -0.0293 [-0.0313, -0.0274] | +0.0347 [+0.0325, +0.0367] |
| load_pu_blind | 0 dB | 0.8290 | 0.7620 | 0.30142 | 0.1489 | -0.0150 [-0.0167, -0.0136] | -0.0187 [-0.0204, -0.0173] | +0.0228 [+0.0212, +0.0245] |
| load_pu_blind | 10 dB | 0.9491 | 0.9190 | 0.09918 | 0.0287 | -0.0043 [-0.0053, -0.0035] | -0.0049 [-0.0058, -0.0041] | +0.0059 [+0.0050, +0.0067] |
| load_pu_blind | 20 dB | 0.9908 | 0.9840 | 0.02555 | 0.0076 | -0.0005 [-0.0008, -0.0003] | -0.0007 [-0.0010, -0.0005] | +0.0009 [+0.0006, +0.0012] |
| load_pu_rho | -5 dB | 0.7355 | 0.6529 | 0.43708 | - | -0.0159 [-0.0182, -0.0135] | -0.0280 [-0.0299, -0.0261] | +0.0174 [+0.0151, +0.0196] |
| load_pu_rho | 0 dB | 0.8295 | 0.7630 | 0.29692 | - | -0.0145 [-0.0162, -0.0131] | -0.0182 [-0.0199, -0.0168] | +0.0183 [+0.0169, +0.0199] |
| load_pu_rho | 10 dB | 0.9492 | 0.9191 | 0.09917 | - | -0.0043 [-0.0052, -0.0035] | -0.0048 [-0.0058, -0.0040] | +0.0058 [+0.0050, +0.0067] |
| load_pu_rho | 20 dB | 0.9908 | 0.9840 | 0.02552 | - | -0.0005 [-0.0008, -0.0003] | -0.0007 [-0.0010, -0.0005] | +0.0008 [+0.0006, +0.0011] |

### coco_K2_U2_E8_emauto0_fs_it30: frames 2000; data/shared/pilot/shared_pilot_coco_K2_U2_E8_emauto0_fs_it30_snr-5.json, data/shared/pilot/shared_pilot_coco_K2_U2_E8_emauto0_fs_it30_snr0.json, data/shared/pilot/shared_pilot_coco_K2_U2_E8_emauto0_fs_it30_snr10.json, data/shared/pilot/shared_pilot_coco_K2_U2_E8_emauto0_fs_it30_snr20.json

| mode | SNR | CosSim | weakest | NMSE | gain rel. RMSE | - load_joint_blind CosSim [95% CI] | - load_joint (true g) CosSim [95% CI] | - load_joint_blind NMSE [95% CI] |
|---|---|---|---|---|---|---|---|---|
| gen_joint_em | -5 dB | 0.7650 | 0.7049 | 0.40430 | 0.1741 | +0.0136 [+0.0120, +0.0152] | +0.0015 [+0.0009, +0.0020] | -0.0154 [-0.0172, -0.0136] |
| gen_joint_em | 0 dB | 0.8477 | 0.7970 | 0.27697 | 0.1436 | +0.0036 [+0.0028, +0.0045] | -0.0000 [-0.0003, +0.0003] | -0.0016 [-0.0027, -0.0006] |
| gen_joint_em | 10 dB | 0.9537 | 0.9280 | 0.09904 | 0.1210 | +0.0003 [+0.0000, +0.0006] | -0.0003 [-0.0004, -0.0001] | +0.0057 [+0.0051, +0.0063] |
| gen_joint_em | 20 dB | 0.9914 | 0.9853 | 0.02616 | 0.0881 | +0.0002 [+0.0000, +0.0003] | -0.0001 [-0.0001, -0.0000] | +0.0015 [+0.0011, +0.0018] |
| gen_joint_em_rho | -5 dB | 0.7645 | 0.7040 | 0.40270 | 0.1528 | +0.0131 [+0.0115, +0.0147] | +0.0010 [+0.0004, +0.0016] | -0.0170 [-0.0188, -0.0154] |
| gen_joint_em_rho | 0 dB | 0.8468 | 0.7949 | 0.27483 | 0.1046 | +0.0027 [+0.0019, +0.0036] | -0.0009 [-0.0013, -0.0005] | -0.0038 [-0.0047, -0.0029] |
| gen_joint_em_rho | 10 dB | 0.9531 | 0.9269 | 0.09368 | 0.0372 | -0.0004 [-0.0005, -0.0002] | -0.0009 [-0.0012, -0.0006] | +0.0004 [+0.0002, +0.0005] |
| gen_joint_em_rho | 20 dB | 0.9912 | 0.9849 | 0.02478 | 0.0090 | -0.0000 [-0.0001, +0.0000] | -0.0003 [-0.0004, -0.0002] | +0.0001 [-0.0000, +0.0002] |
| gen_joint_pilot | -5 dB | 0.7395 | 0.6691 | 0.47552 | 0.7970 | -0.0119 [-0.0140, -0.0097] | -0.0240 [-0.0257, -0.0224] | +0.0558 [+0.0514, +0.0601] |
| gen_joint_pilot | 0 dB | 0.8333 | 0.7742 | 0.33079 | 0.5753 | -0.0108 [-0.0121, -0.0094] | -0.0144 [-0.0157, -0.0133] | +0.0522 [+0.0485, +0.0559] |
| gen_joint_pilot | 10 dB | 0.9521 | 0.9250 | 0.10578 | 0.1993 | -0.0014 [-0.0017, -0.0010] | -0.0019 [-0.0022, -0.0016] | +0.0125 [+0.0112, +0.0138] |
| gen_joint_pilot | 20 dB | 0.9913 | 0.9851 | 0.02066 | 0.0586 | +0.0001 [-0.0001, +0.0002] | -0.0002 [-0.0002, -0.0001] | -0.0040 [-0.0046, -0.0034] |
| gen_joint_true | -5 dB | 0.7667 | 0.7082 | 0.40097 | - | +0.0153 [+0.0135, +0.0169] | +0.0032 [+0.0030, +0.0034] | -0.0187 [-0.0206, -0.0170] |
| gen_joint_true | 0 dB | 0.8480 | 0.7980 | 0.27285 | - | +0.0040 [+0.0031, +0.0049] | +0.0003 [+0.0003, +0.0004] | -0.0057 [-0.0068, -0.0048] |
| gen_joint_true | 10 dB | 0.9537 | 0.9280 | 0.08700 | - | +0.0003 [+0.0000, +0.0005] | -0.0003 [-0.0003, -0.0003] | -0.0063 [-0.0069, -0.0058] |
| gen_joint_true | 20 dB | 0.9914 | 0.9853 | 0.01651 | - | +0.0002 [+0.0000, +0.0003] | -0.0001 [-0.0001, -0.0001] | -0.0082 [-0.0086, -0.0078] |
| gen_joint_true_rho | -5 dB | 0.7666 | 0.7078 | 0.40153 | - | +0.0152 [+0.0134, +0.0168] | +0.0031 [+0.0028, +0.0034] | -0.0182 [-0.0200, -0.0165] |
| gen_joint_true_rho | 0 dB | 0.8477 | 0.7967 | 0.27478 | - | +0.0036 [+0.0029, +0.0045] | -0.0000 [-0.0003, +0.0003] | -0.0038 [-0.0048, -0.0029] |
| gen_joint_true_rho | 10 dB | 0.9533 | 0.9273 | 0.09381 | - | -0.0001 [-0.0002, +0.0000] | -0.0007 [-0.0009, -0.0004] | +0.0005 [+0.0003, +0.0006] |
| gen_joint_true_rho | 20 dB | 0.9913 | 0.9850 | 0.02483 | - | -0.0000 [-0.0001, +0.0001] | -0.0002 [-0.0003, -0.0002] | +0.0001 [+0.0000, +0.0002] |
| load_joint | -5 dB | 0.7635 | 0.7044 | 0.40560 | - | +0.0121 [+0.0104, +0.0138] | +0.0000 [+0.0000, +0.0000] | -0.0141 [-0.0159, -0.0124] |
| load_joint | 0 dB | 0.8477 | 0.7976 | 0.27331 | - | +0.0036 [+0.0028, +0.0045] | +0.0000 [+0.0000, +0.0000] | -0.0053 [-0.0064, -0.0043] |
| load_joint | 10 dB | 0.9540 | 0.9284 | 0.08647 | - | +0.0005 [+0.0003, +0.0008] | +0.0000 [+0.0000, +0.0000] | -0.0069 [-0.0074, -0.0063] |
| load_joint | 20 dB | 0.9915 | 0.9854 | 0.01638 | - | +0.0002 [+0.0001, +0.0004] | +0.0000 [+0.0000, +0.0000] | -0.0083 [-0.0087, -0.0079] |
| load_joint_blind | -5 dB | 0.7515 | 0.6822 | 0.41972 | 0.3205 | +0.0000 [+0.0000, +0.0000] | -0.0121 [-0.0138, -0.0104] | +0.0000 [+0.0000, +0.0000] |
| load_joint_blind | 0 dB | 0.8440 | 0.7903 | 0.27859 | 0.1489 | +0.0000 [+0.0000, +0.0000] | -0.0036 [-0.0045, -0.0028] | +0.0000 [+0.0000, +0.0000] |
| load_joint_blind | 10 dB | 0.9534 | 0.9274 | 0.09332 | 0.0287 | +0.0000 [+0.0000, +0.0000] | -0.0005 [-0.0008, -0.0003] | +0.0000 [+0.0000, +0.0000] |
| load_joint_blind | 20 dB | 0.9913 | 0.9850 | 0.02470 | 0.0076 | +0.0000 [+0.0000, +0.0000] | -0.0002 [-0.0004, -0.0001] | +0.0000 [+0.0000, +0.0000] |
| load_joint_rho | -5 dB | 0.7635 | 0.7041 | 0.40606 | - | +0.0120 [+0.0103, +0.0137] | -0.0000 [-0.0003, +0.0002] | -0.0137 [-0.0154, -0.0119] |
| load_joint_rho | 0 dB | 0.8474 | 0.7964 | 0.27516 | - | +0.0033 [+0.0026, +0.0042] | -0.0003 [-0.0006, -0.0000] | -0.0034 [-0.0044, -0.0026] |
| load_joint_rho | 10 dB | 0.9536 | 0.9277 | 0.09327 | - | +0.0002 [+0.0001, +0.0003] | -0.0004 [-0.0006, -0.0002] | -0.0001 [-0.0002, +0.0001] |
| load_joint_rho | 20 dB | 0.9913 | 0.9851 | 0.02470 | - | +0.0001 [-0.0000, +0.0001] | -0.0002 [-0.0003, -0.0001] | +0.0000 [-0.0001, +0.0001] |
| load_pu | -5 dB | 0.7354 | 0.6529 | 0.43685 | - | -0.0160 [-0.0183, -0.0136] | -0.0281 [-0.0300, -0.0262] | +0.0171 [+0.0149, +0.0194] |
| load_pu | 0 dB | 0.8296 | 0.7639 | 0.29545 | - | -0.0144 [-0.0162, -0.0129] | -0.0181 [-0.0198, -0.0167] | +0.0169 [+0.0153, +0.0185] |
| load_pu | 10 dB | 0.9494 | 0.9195 | 0.09246 | - | -0.0040 [-0.0050, -0.0032] | -0.0046 [-0.0055, -0.0038] | -0.0009 [-0.0019, +0.0002] |
| load_pu | 20 dB | 0.9909 | 0.9843 | 0.01715 | - | -0.0003 [-0.0006, -0.0001] | -0.0006 [-0.0008, -0.0003] | -0.0075 [-0.0080, -0.0070] |
| load_pu_blind | -5 dB | 0.7342 | 0.6507 | 0.45445 | 0.3205 | -0.0173 [-0.0196, -0.0149] | -0.0293 [-0.0313, -0.0274] | +0.0347 [+0.0325, +0.0367] |
| load_pu_blind | 0 dB | 0.8290 | 0.7620 | 0.30142 | 0.1489 | -0.0150 [-0.0167, -0.0136] | -0.0187 [-0.0204, -0.0173] | +0.0228 [+0.0212, +0.0245] |
| load_pu_blind | 10 dB | 0.9491 | 0.9190 | 0.09918 | 0.0287 | -0.0043 [-0.0053, -0.0035] | -0.0049 [-0.0058, -0.0041] | +0.0059 [+0.0050, +0.0067] |
| load_pu_blind | 20 dB | 0.9908 | 0.9840 | 0.02555 | 0.0076 | -0.0005 [-0.0008, -0.0003] | -0.0007 [-0.0010, -0.0005] | +0.0009 [+0.0006, +0.0012] |
| load_pu_rho | -5 dB | 0.7355 | 0.6529 | 0.43708 | - | -0.0159 [-0.0182, -0.0135] | -0.0280 [-0.0299, -0.0261] | +0.0174 [+0.0151, +0.0196] |
| load_pu_rho | 0 dB | 0.8295 | 0.7630 | 0.29692 | - | -0.0145 [-0.0162, -0.0131] | -0.0182 [-0.0199, -0.0168] | +0.0183 [+0.0169, +0.0199] |
| load_pu_rho | 10 dB | 0.9492 | 0.9191 | 0.09917 | - | -0.0043 [-0.0052, -0.0035] | -0.0048 [-0.0058, -0.0040] | +0.0058 [+0.0050, +0.0067] |
| load_pu_rho | 20 dB | 0.9908 | 0.9840 | 0.02552 | - | -0.0005 [-0.0008, -0.0003] | -0.0007 [-0.0010, -0.0005] | +0.0008 [+0.0006, +0.0011] |

### coco_K2_U4_E1_emauto0_fs_it30: frames 2000; data/shared/pilot/shared_pilot_coco_K2_U4_E1_emauto0_fs_it30_snr-5.json, data/shared/pilot/shared_pilot_coco_K2_U4_E1_emauto0_fs_it30_snr0.json, data/shared/pilot/shared_pilot_coco_K2_U4_E1_emauto0_fs_it30_snr10.json, data/shared/pilot/shared_pilot_coco_K2_U4_E1_emauto0_fs_it30_snr20.json

| mode | SNR | CosSim | weakest | NMSE | gain rel. RMSE | - load_joint_blind CosSim [95% CI] | - load_joint (true g) CosSim [95% CI] | - load_joint_blind NMSE [95% CI] |
|---|---|---|---|---|---|---|---|---|
| gen_joint_em | -5 dB | 0.7353 | 0.6186 | 0.45333 | 0.2480 | +0.0217 [+0.0205, +0.0229] | +0.0094 [+0.0089, +0.0100] | -0.0265 [-0.0278, -0.0252] |
| gen_joint_em | 0 dB | 0.8144 | 0.7051 | 0.33236 | 0.1524 | +0.0060 [+0.0054, +0.0065] | +0.0025 [+0.0023, +0.0028] | -0.0058 [-0.0065, -0.0051] |
| gen_joint_em | 10 dB | 0.9317 | 0.8681 | 0.13825 | 0.1161 | +0.0009 [+0.0007, +0.0011] | +0.0005 [+0.0004, +0.0006] | +0.0042 [+0.0037, +0.0047] |
| gen_joint_em | 20 dB | 0.9734 | 0.9525 | 0.06247 | 0.0979 | +0.0003 [+0.0003, +0.0004] | +0.0003 [+0.0002, +0.0003] | +0.0046 [+0.0043, +0.0050] |
| gen_joint_em0 | -5 dB | 0.7356 | 0.6192 | 0.45297 | 0.2480 | +0.0220 [+0.0208, +0.0232] | +0.0097 [+0.0092, +0.0102] | -0.0269 [-0.0282, -0.0255] |
| gen_joint_em0 | 0 dB | 0.8145 | 0.7052 | 0.33215 | 0.1535 | +0.0061 [+0.0056, +0.0066] | +0.0027 [+0.0024, +0.0029] | -0.0060 [-0.0067, -0.0053] |
| gen_joint_em0 | 10 dB | 0.9318 | 0.8683 | 0.13833 | 0.1179 | +0.0010 [+0.0008, +0.0012] | +0.0006 [+0.0005, +0.0007] | +0.0043 [+0.0038, +0.0048] |
| gen_joint_em0 | 20 dB | 0.9735 | 0.9527 | 0.06476 | 0.1087 | +0.0004 [+0.0003, +0.0005] | +0.0004 [+0.0003, +0.0004] | +0.0069 [+0.0065, +0.0074] |
| gen_joint_em0_rho | -5 dB | 0.7352 | 0.6186 | 0.45121 | 0.2339 | +0.0216 [+0.0205, +0.0228] | +0.0093 [+0.0087, +0.0098] | -0.0286 [-0.0299, -0.0274] |
| gen_joint_em0_rho | 0 dB | 0.8137 | 0.7030 | 0.33010 | 0.1311 | +0.0053 [+0.0048, +0.0058] | +0.0019 [+0.0015, +0.0022] | -0.0081 [-0.0086, -0.0075] |
| gen_joint_em0_rho | 10 dB | 0.9313 | 0.8664 | 0.13283 | 0.0458 | +0.0005 [+0.0004, +0.0007] | +0.0002 [-0.0000, +0.0003] | -0.0012 [-0.0013, -0.0011] |
| gen_joint_em0_rho | 20 dB | 0.9733 | 0.9520 | 0.05711 | 0.0131 | +0.0003 [+0.0003, +0.0004] | +0.0002 [+0.0002, +0.0003] | -0.0007 [-0.0008, -0.0007] |
| gen_joint_em_rho | -5 dB | 0.7349 | 0.6181 | 0.45154 | 0.2339 | +0.0213 [+0.0202, +0.0225] | +0.0090 [+0.0085, +0.0096] | -0.0283 [-0.0296, -0.0271] |
| gen_joint_em_rho | 0 dB | 0.8136 | 0.7029 | 0.33035 | 0.1304 | +0.0052 [+0.0047, +0.0056] | +0.0017 [+0.0014, +0.0021] | -0.0078 [-0.0084, -0.0072] |
| gen_joint_em_rho | 10 dB | 0.9312 | 0.8662 | 0.13302 | 0.0457 | +0.0004 [+0.0003, +0.0005] | +0.0001 [-0.0001, +0.0002] | -0.0010 [-0.0012, -0.0009] |
| gen_joint_em_rho | 20 dB | 0.9733 | 0.9519 | 0.05726 | 0.0131 | +0.0002 [+0.0002, +0.0003] | +0.0001 [+0.0001, +0.0002] | -0.0006 [-0.0006, -0.0005] |
| gen_joint_pilot | -5 dB | 0.6748 | 0.5378 | 0.59232 | 2.4056 | -0.0388 [-0.0408, -0.0368] | -0.0510 [-0.0528, -0.0492] | +0.1125 [+0.1084, +0.1167] |
| gen_joint_pilot | 0 dB | 0.7723 | 0.6363 | 0.46046 | 1.3114 | -0.0361 [-0.0376, -0.0346] | -0.0395 [-0.0410, -0.0380] | +0.1223 [+0.1179, +0.1273] |
| gen_joint_pilot | 10 dB | 0.9233 | 0.8467 | 0.23221 | 0.5027 | -0.0075 [-0.0080, -0.0070] | -0.0079 [-0.0084, -0.0073] | +0.0982 [+0.0928, +0.1034] |
| gen_joint_pilot | 20 dB | 0.9726 | 0.9500 | 0.08482 | 0.1779 | -0.0004 [-0.0006, -0.0003] | -0.0005 [-0.0006, -0.0004] | +0.0270 [+0.0245, +0.0299] |
| gen_joint_true | -5 dB | 0.7382 | 0.6268 | 0.44831 | - | +0.0245 [+0.0233, +0.0258] | +0.0123 [+0.0119, +0.0127] | -0.0315 [-0.0329, -0.0302] |
| gen_joint_true | 0 dB | 0.8152 | 0.7086 | 0.32850 | - | +0.0068 [+0.0062, +0.0073] | +0.0033 [+0.0032, +0.0035] | -0.0097 [-0.0103, -0.0090] |
| gen_joint_true | 10 dB | 0.9316 | 0.8682 | 0.12932 | - | +0.0008 [+0.0006, +0.0010] | +0.0004 [+0.0004, +0.0004] | -0.0047 [-0.0050, -0.0044] |
| gen_joint_true | 20 dB | 0.9733 | 0.9524 | 0.05221 | - | +0.0003 [+0.0002, +0.0004] | +0.0002 [+0.0002, +0.0002] | -0.0056 [-0.0058, -0.0054] |
| gen_joint_true0 | -5 dB | 0.7384 | 0.6270 | 0.44803 | - | +0.0247 [+0.0235, +0.0260] | +0.0125 [+0.0121, +0.0129] | -0.0318 [-0.0331, -0.0305] |
| gen_joint_true0 | 0 dB | 0.8154 | 0.7088 | 0.32823 | - | +0.0069 [+0.0064, +0.0075] | +0.0035 [+0.0034, +0.0036] | -0.0099 [-0.0106, -0.0093] |
| gen_joint_true0 | 10 dB | 0.9317 | 0.8684 | 0.12911 | - | +0.0009 [+0.0007, +0.0011] | +0.0005 [+0.0005, +0.0005] | -0.0049 [-0.0052, -0.0046] |
| gen_joint_true0 | 20 dB | 0.9734 | 0.9525 | 0.05204 | - | +0.0004 [+0.0003, +0.0004] | +0.0003 [+0.0003, +0.0003] | -0.0058 [-0.0060, -0.0056] |
| gen_joint_true0_rho | -5 dB | 0.7383 | 0.6267 | 0.44856 | - | +0.0247 [+0.0235, +0.0259] | +0.0124 [+0.0120, +0.0128] | -0.0313 [-0.0326, -0.0300] |
| gen_joint_true0_rho | 0 dB | 0.8152 | 0.7073 | 0.32936 | - | +0.0067 [+0.0062, +0.0072] | +0.0033 [+0.0031, +0.0035] | -0.0088 [-0.0094, -0.0082] |
| gen_joint_true0_rho | 10 dB | 0.9315 | 0.8671 | 0.13303 | - | +0.0007 [+0.0006, +0.0008] | +0.0004 [+0.0002, +0.0005] | -0.0010 [-0.0012, -0.0008] |
| gen_joint_true0_rho | 20 dB | 0.9734 | 0.9520 | 0.05730 | - | +0.0003 [+0.0003, +0.0004] | +0.0002 [+0.0002, +0.0003] | -0.0005 [-0.0006, -0.0005] |
| gen_joint_true_rho | -5 dB | 0.7381 | 0.6265 | 0.44884 | - | +0.0245 [+0.0233, +0.0257] | +0.0122 [+0.0118, +0.0126] | -0.0310 [-0.0323, -0.0297] |
| gen_joint_true_rho | 0 dB | 0.8150 | 0.7071 | 0.32962 | - | +0.0065 [+0.0061, +0.0071] | +0.0031 [+0.0029, +0.0033] | -0.0085 [-0.0091, -0.0080] |
| gen_joint_true_rho | 10 dB | 0.9314 | 0.8669 | 0.13322 | - | +0.0006 [+0.0005, +0.0007] | +0.0003 [+0.0001, +0.0004] | -0.0008 [-0.0010, -0.0007] |
| gen_joint_true_rho | 20 dB | 0.9733 | 0.9519 | 0.05746 | - | +0.0002 [+0.0002, +0.0003] | +0.0002 [+0.0001, +0.0002] | -0.0004 [-0.0005, -0.0003] |
| load_joint | -5 dB | 0.7259 | 0.6125 | 0.46576 | - | +0.0123 [+0.0112, +0.0134] | +0.0000 [+0.0000, +0.0000] | -0.0141 [-0.0154, -0.0129] |
| load_joint | 0 dB | 0.8119 | 0.7045 | 0.33375 | - | +0.0034 [+0.0029, +0.0040] | +0.0000 [+0.0000, +0.0000] | -0.0044 [-0.0051, -0.0038] |
| load_joint | 10 dB | 0.9312 | 0.8683 | 0.13007 | - | +0.0004 [+0.0002, +0.0006] | +0.0000 [+0.0000, +0.0000] | -0.0040 [-0.0043, -0.0036] |
| load_joint | 20 dB | 0.9731 | 0.9529 | 0.05261 | - | +0.0001 [+0.0000, +0.0001] | +0.0000 [+0.0000, +0.0000] | -0.0052 [-0.0054, -0.0050] |
| load_joint_blind | -5 dB | 0.7136 | 0.5820 | 0.47985 | 0.5192 | +0.0000 [+0.0000, +0.0000] | -0.0123 [-0.0134, -0.0112] | +0.0000 [+0.0000, +0.0000] |
| load_joint_blind | 0 dB | 0.8084 | 0.6927 | 0.33815 | 0.2140 | +0.0000 [+0.0000, +0.0000] | -0.0034 [-0.0040, -0.0029] | +0.0000 [+0.0000, +0.0000] |
| load_joint_blind | 10 dB | 0.9308 | 0.8661 | 0.13403 | 0.0469 | +0.0000 [+0.0000, +0.0000] | -0.0004 [-0.0006, -0.0002] | +0.0000 [+0.0000, +0.0000] |
| load_joint_blind | 20 dB | 0.9730 | 0.9524 | 0.05782 | 0.0115 | +0.0000 [+0.0000, +0.0000] | -0.0001 [-0.0001, -0.0000] | +0.0000 [+0.0000, +0.0000] |
| load_joint_em | -5 dB | 0.7230 | 0.6039 | 0.47035 | 0.2376 | +0.0094 [+0.0084, +0.0105] | -0.0029 [-0.0033, -0.0025] | -0.0095 [-0.0107, -0.0083] |
| load_joint_em | 0 dB | 0.8111 | 0.7010 | 0.33756 | 0.1500 | +0.0026 [+0.0022, +0.0032] | -0.0008 [-0.0010, -0.0005] | -0.0006 [-0.0012, +0.0001] |
| load_joint_em | 10 dB | 0.9313 | 0.8682 | 0.13941 | 0.1168 | +0.0005 [+0.0003, +0.0007] | +0.0001 [+0.0000, +0.0002] | +0.0054 [+0.0049, +0.0059] |
| load_joint_em | 20 dB | 0.9731 | 0.9530 | 0.06564 | 0.1093 | +0.0001 [+0.0000, +0.0002] | +0.0000 [+0.0000, +0.0001] | +0.0078 [+0.0073, +0.0083] |
| load_joint_em_rho | -5 dB | 0.7227 | 0.6043 | 0.46849 | 0.2223 | +0.0091 [+0.0081, +0.0101] | -0.0032 [-0.0037, -0.0027] | -0.0114 [-0.0125, -0.0103] |
| load_joint_em_rho | 0 dB | 0.8104 | 0.6990 | 0.33528 | 0.1263 | +0.0019 [+0.0015, +0.0024] | -0.0015 [-0.0018, -0.0012] | -0.0029 [-0.0034, -0.0024] |
| load_joint_em_rho | 10 dB | 0.9309 | 0.8663 | 0.13378 | 0.0451 | +0.0000 [-0.0001, +0.0002] | -0.0003 [-0.0005, -0.0001] | -0.0003 [-0.0004, -0.0001] |
| load_joint_em_rho | 20 dB | 0.9730 | 0.9524 | 0.05778 | 0.0127 | +0.0000 [-0.0000, +0.0001] | -0.0001 [-0.0001, -0.0000] | -0.0000 [-0.0001, -0.0000] |
| load_joint_rho | -5 dB | 0.7260 | 0.6131 | 0.46607 | - | +0.0124 [+0.0113, +0.0135] | +0.0001 [-0.0001, +0.0003] | -0.0138 [-0.0149, -0.0126] |
| load_joint_rho | 0 dB | 0.8117 | 0.7032 | 0.33469 | - | +0.0033 [+0.0028, +0.0037] | -0.0002 [-0.0004, +0.0000] | -0.0035 [-0.0040, -0.0029] |
| load_joint_rho | 10 dB | 0.9310 | 0.8670 | 0.13393 | - | +0.0002 [+0.0001, +0.0003] | -0.0001 [-0.0003, +0.0000] | -0.0001 [-0.0002, +0.0000] |
| load_joint_rho | 20 dB | 0.9731 | 0.9525 | 0.05786 | - | +0.0000 [-0.0000, +0.0001] | -0.0000 [-0.0001, +0.0000] | +0.0000 [-0.0000, +0.0001] |
| load_pu | -5 dB | 0.6679 | 0.4765 | 0.52987 | - | -0.0458 [-0.0479, -0.0436] | -0.0580 [-0.0601, -0.0561] | +0.0500 [+0.0481, +0.0518] |
| load_pu | 0 dB | 0.7758 | 0.6107 | 0.37880 | - | -0.0326 [-0.0341, -0.0311] | -0.0360 [-0.0375, -0.0345] | +0.0406 [+0.0391, +0.0421] |
| load_pu | 10 dB | 0.9224 | 0.8397 | 0.14258 | - | -0.0084 [-0.0092, -0.0077] | -0.0088 [-0.0095, -0.0080] | +0.0086 [+0.0076, +0.0095] |
| load_pu | 20 dB | 0.9720 | 0.9491 | 0.05428 | - | -0.0010 [-0.0012, -0.0008] | -0.0011 [-0.0013, -0.0009] | -0.0035 [-0.0039, -0.0032] |
| load_pu_blind | -5 dB | 0.6664 | 0.4747 | 0.54960 | 0.5192 | -0.0472 [-0.0493, -0.0451] | -0.0595 [-0.0616, -0.0575] | +0.0698 [+0.0679, +0.0717] |
| load_pu_blind | 0 dB | 0.7754 | 0.6089 | 0.38491 | 0.2140 | -0.0331 [-0.0346, -0.0316] | -0.0365 [-0.0380, -0.0349] | +0.0468 [+0.0451, +0.0484] |
| load_pu_blind | 10 dB | 0.9224 | 0.8388 | 0.14661 | 0.0469 | -0.0084 [-0.0092, -0.0077] | -0.0088 [-0.0096, -0.0081] | +0.0126 [+0.0117, +0.0135] |
| load_pu_blind | 20 dB | 0.9720 | 0.9487 | 0.05955 | 0.0115 | -0.0010 [-0.0012, -0.0008] | -0.0011 [-0.0013, -0.0009] | +0.0017 [+0.0015, +0.0020] |
| load_pu_rho | -5 dB | 0.6680 | 0.4772 | 0.52999 | - | -0.0456 [-0.0477, -0.0435] | -0.0579 [-0.0600, -0.0559] | +0.0501 [+0.0483, +0.0519] |
| load_pu_rho | 0 dB | 0.7758 | 0.6100 | 0.37930 | - | -0.0326 [-0.0341, -0.0312] | -0.0360 [-0.0375, -0.0345] | +0.0411 [+0.0397, +0.0426] |
| load_pu_rho | 10 dB | 0.9224 | 0.8389 | 0.14633 | - | -0.0084 [-0.0091, -0.0077] | -0.0088 [-0.0095, -0.0080] | +0.0123 [+0.0115, +0.0131] |
| load_pu_rho | 20 dB | 0.9720 | 0.9487 | 0.05959 | - | -0.0010 [-0.0012, -0.0008] | -0.0011 [-0.0013, -0.0009] | +0.0018 [+0.0015, +0.0021] |

### coco_K2_U4_E8_emauto0_fs_it30: frames 2000; data/shared/pilot/shared_pilot_coco_K2_U4_E8_emauto0_fs_it30_snr-5.json, data/shared/pilot/shared_pilot_coco_K2_U4_E8_emauto0_fs_it30_snr0.json, data/shared/pilot/shared_pilot_coco_K2_U4_E8_emauto0_fs_it30_snr10.json, data/shared/pilot/shared_pilot_coco_K2_U4_E8_emauto0_fs_it30_snr20.json

| mode | SNR | CosSim | weakest | NMSE | gain rel. RMSE | - load_joint_blind CosSim [95% CI] | - load_joint (true g) CosSim [95% CI] | - load_joint_blind NMSE [95% CI] |
|---|---|---|---|---|---|---|---|---|
| gen_joint_em | -5 dB | 0.7341 | 0.6176 | 0.45507 | 0.2454 | +0.0205 [+0.0193, +0.0216] | +0.0082 [+0.0076, +0.0087] | -0.0248 [-0.0261, -0.0235] |
| gen_joint_em | 0 dB | 0.8133 | 0.7041 | 0.33394 | 0.1487 | +0.0048 [+0.0043, +0.0054] | +0.0014 [+0.0012, +0.0017] | -0.0042 [-0.0049, -0.0035] |
| gen_joint_em | 10 dB | 0.9311 | 0.8671 | 0.13741 | 0.1058 | +0.0003 [+0.0001, +0.0005] | -0.0001 [-0.0002, -0.0000] | +0.0034 [+0.0030, +0.0038] |
| gen_joint_em | 20 dB | 0.9732 | 0.9522 | 0.05713 | 0.0680 | +0.0002 [+0.0001, +0.0003] | +0.0001 [+0.0001, +0.0001] | -0.0007 [-0.0009, -0.0005] |
| gen_joint_em_rho | -5 dB | 0.7337 | 0.6171 | 0.45339 | 0.2315 | +0.0201 [+0.0190, +0.0213] | +0.0078 [+0.0073, +0.0084] | -0.0265 [-0.0277, -0.0252] |
| gen_joint_em_rho | 0 dB | 0.8125 | 0.7018 | 0.33219 | 0.1284 | +0.0041 [+0.0036, +0.0045] | +0.0006 [+0.0003, +0.0009] | -0.0060 [-0.0065, -0.0054] |
| gen_joint_em_rho | 10 dB | 0.9306 | 0.8652 | 0.13411 | 0.0456 | -0.0002 [-0.0003, -0.0001] | -0.0005 [-0.0007, -0.0004] | +0.0001 [-0.0001, +0.0002] |
| gen_joint_em_rho | 20 dB | 0.9731 | 0.9515 | 0.05754 | 0.0130 | +0.0001 [+0.0000, +0.0001] | -0.0000 [-0.0001, +0.0001] | -0.0003 [-0.0004, -0.0002] |
| gen_joint_pilot | -5 dB | 0.7111 | 0.5833 | 0.50459 | 0.9722 | -0.0026 [-0.0041, -0.0010] | -0.0148 [-0.0161, -0.0136] | +0.0247 [+0.0224, +0.0272] |
| gen_joint_pilot | 0 dB | 0.8017 | 0.6821 | 0.36693 | 0.5420 | -0.0067 [-0.0076, -0.0058] | -0.0102 [-0.0109, -0.0094] | +0.0288 [+0.0271, +0.0303] |
| gen_joint_pilot | 10 dB | 0.9295 | 0.8626 | 0.14412 | 0.1980 | -0.0013 [-0.0016, -0.0011] | -0.0017 [-0.0019, -0.0015] | +0.0101 [+0.0093, +0.0108] |
| gen_joint_pilot | 20 dB | 0.9730 | 0.9516 | 0.05608 | 0.0665 | +0.0000 [-0.0001, +0.0001] | -0.0001 [-0.0001, -0.0000] | -0.0017 [-0.0021, -0.0014] |
| gen_joint_true | -5 dB | 0.7368 | 0.6254 | 0.45026 | - | +0.0232 [+0.0220, +0.0245] | +0.0109 [+0.0105, +0.0113] | -0.0296 [-0.0309, -0.0282] |
| gen_joint_true | 0 dB | 0.8140 | 0.7073 | 0.33042 | - | +0.0056 [+0.0051, +0.0061] | +0.0022 [+0.0020, +0.0023] | -0.0077 [-0.0084, -0.0071] |
| gen_joint_true | 10 dB | 0.9310 | 0.8672 | 0.13042 | - | +0.0002 [-0.0000, +0.0004] | -0.0002 [-0.0002, -0.0002] | -0.0036 [-0.0039, -0.0033] |
| gen_joint_true | 20 dB | 0.9732 | 0.9521 | 0.05249 | - | +0.0001 [+0.0001, +0.0002] | +0.0000 [+0.0000, +0.0001] | -0.0053 [-0.0056, -0.0051] |
| gen_joint_true_rho | -5 dB | 0.7368 | 0.6251 | 0.45079 | - | +0.0232 [+0.0219, +0.0244] | +0.0109 [+0.0105, +0.0113] | -0.0291 [-0.0304, -0.0277] |
| gen_joint_true_rho | 0 dB | 0.8138 | 0.7058 | 0.33152 | - | +0.0054 [+0.0049, +0.0059] | +0.0019 [+0.0017, +0.0022] | -0.0066 [-0.0072, -0.0061] |
| gen_joint_true_rho | 10 dB | 0.9308 | 0.8659 | 0.13431 | - | +0.0000 [-0.0001, +0.0001] | -0.0004 [-0.0005, -0.0002] | +0.0003 [+0.0001, +0.0004] |
| gen_joint_true_rho | 20 dB | 0.9731 | 0.9515 | 0.05774 | - | +0.0001 [+0.0000, +0.0001] | +0.0000 [-0.0001, +0.0001] | -0.0001 [-0.0002, -0.0000] |
| load_joint | -5 dB | 0.7259 | 0.6125 | 0.46576 | - | +0.0123 [+0.0112, +0.0134] | +0.0000 [+0.0000, +0.0000] | -0.0141 [-0.0154, -0.0129] |
| load_joint | 0 dB | 0.8119 | 0.7045 | 0.33375 | - | +0.0034 [+0.0029, +0.0040] | +0.0000 [+0.0000, +0.0000] | -0.0044 [-0.0051, -0.0038] |
| load_joint | 10 dB | 0.9312 | 0.8683 | 0.13007 | - | +0.0004 [+0.0002, +0.0006] | +0.0000 [+0.0000, +0.0000] | -0.0040 [-0.0043, -0.0036] |
| load_joint | 20 dB | 0.9731 | 0.9529 | 0.05261 | - | +0.0001 [+0.0000, +0.0001] | +0.0000 [+0.0000, +0.0000] | -0.0052 [-0.0054, -0.0050] |
| load_joint_blind | -5 dB | 0.7136 | 0.5820 | 0.47985 | 0.5192 | +0.0000 [+0.0000, +0.0000] | -0.0123 [-0.0134, -0.0112] | +0.0000 [+0.0000, +0.0000] |
| load_joint_blind | 0 dB | 0.8084 | 0.6927 | 0.33815 | 0.2140 | +0.0000 [+0.0000, +0.0000] | -0.0034 [-0.0040, -0.0029] | +0.0000 [+0.0000, +0.0000] |
| load_joint_blind | 10 dB | 0.9308 | 0.8661 | 0.13403 | 0.0469 | +0.0000 [+0.0000, +0.0000] | -0.0004 [-0.0006, -0.0002] | +0.0000 [+0.0000, +0.0000] |
| load_joint_blind | 20 dB | 0.9730 | 0.9524 | 0.05782 | 0.0115 | +0.0000 [+0.0000, +0.0000] | -0.0001 [-0.0001, -0.0000] | +0.0000 [+0.0000, +0.0000] |
| load_joint_rho | -5 dB | 0.7260 | 0.6131 | 0.46607 | - | +0.0124 [+0.0113, +0.0135] | +0.0001 [-0.0001, +0.0003] | -0.0138 [-0.0149, -0.0126] |
| load_joint_rho | 0 dB | 0.8117 | 0.7032 | 0.33469 | - | +0.0033 [+0.0028, +0.0037] | -0.0002 [-0.0004, +0.0000] | -0.0035 [-0.0040, -0.0029] |
| load_joint_rho | 10 dB | 0.9310 | 0.8670 | 0.13393 | - | +0.0002 [+0.0001, +0.0003] | -0.0001 [-0.0003, +0.0000] | -0.0001 [-0.0002, +0.0000] |
| load_joint_rho | 20 dB | 0.9731 | 0.9525 | 0.05786 | - | +0.0000 [-0.0000, +0.0001] | -0.0000 [-0.0001, +0.0000] | +0.0000 [-0.0000, +0.0001] |
| load_pu | -5 dB | 0.6679 | 0.4765 | 0.52987 | - | -0.0458 [-0.0479, -0.0436] | -0.0580 [-0.0601, -0.0561] | +0.0500 [+0.0481, +0.0518] |
| load_pu | 0 dB | 0.7758 | 0.6107 | 0.37880 | - | -0.0326 [-0.0341, -0.0311] | -0.0360 [-0.0375, -0.0345] | +0.0406 [+0.0391, +0.0421] |
| load_pu | 10 dB | 0.9224 | 0.8397 | 0.14258 | - | -0.0084 [-0.0092, -0.0077] | -0.0088 [-0.0095, -0.0080] | +0.0086 [+0.0076, +0.0095] |
| load_pu | 20 dB | 0.9720 | 0.9491 | 0.05428 | - | -0.0010 [-0.0012, -0.0008] | -0.0011 [-0.0013, -0.0009] | -0.0035 [-0.0039, -0.0032] |
| load_pu_blind | -5 dB | 0.6664 | 0.4747 | 0.54960 | 0.5192 | -0.0472 [-0.0493, -0.0451] | -0.0595 [-0.0616, -0.0575] | +0.0698 [+0.0679, +0.0717] |
| load_pu_blind | 0 dB | 0.7754 | 0.6089 | 0.38491 | 0.2140 | -0.0331 [-0.0346, -0.0316] | -0.0365 [-0.0380, -0.0349] | +0.0468 [+0.0451, +0.0484] |
| load_pu_blind | 10 dB | 0.9224 | 0.8388 | 0.14661 | 0.0469 | -0.0084 [-0.0092, -0.0077] | -0.0088 [-0.0096, -0.0081] | +0.0126 [+0.0117, +0.0135] |
| load_pu_blind | 20 dB | 0.9720 | 0.9487 | 0.05955 | 0.0115 | -0.0010 [-0.0012, -0.0008] | -0.0011 [-0.0013, -0.0009] | +0.0017 [+0.0015, +0.0020] |
| load_pu_rho | -5 dB | 0.6680 | 0.4772 | 0.52999 | - | -0.0456 [-0.0477, -0.0435] | -0.0579 [-0.0600, -0.0559] | +0.0501 [+0.0483, +0.0519] |
| load_pu_rho | 0 dB | 0.7758 | 0.6100 | 0.37930 | - | -0.0326 [-0.0341, -0.0312] | -0.0360 [-0.0375, -0.0345] | +0.0411 [+0.0397, +0.0426] |
| load_pu_rho | 10 dB | 0.9224 | 0.8389 | 0.14633 | - | -0.0084 [-0.0091, -0.0077] | -0.0088 [-0.0095, -0.0080] | +0.0123 [+0.0115, +0.0131] |
| load_pu_rho | 20 dB | 0.9720 | 0.9487 | 0.05959 | - | -0.0010 [-0.0012, -0.0008] | -0.0011 [-0.0013, -0.0009] | +0.0018 [+0.0015, +0.0021] |

### coco_K4_U2_E1_emauto0_fs_it30: frames 2000; data/shared/pilot/shared_pilot_coco_K4_U2_E1_emauto0_fs_it30_snr-5.json, data/shared/pilot/shared_pilot_coco_K4_U2_E1_emauto0_fs_it30_snr0.json, data/shared/pilot/shared_pilot_coco_K4_U2_E1_emauto0_fs_it30_snr10.json, data/shared/pilot/shared_pilot_coco_K4_U2_E1_emauto0_fs_it30_snr20.json

| mode | SNR | CosSim | weakest | NMSE | gain rel. RMSE | - load_joint_blind CosSim [95% CI] | - load_joint (true g) CosSim [95% CI] | - load_joint_blind NMSE [95% CI] |
|---|---|---|---|---|---|---|---|---|
| gen_joint_em | -5 dB | 0.8174 | 0.7616 | 0.32422 | 0.1436 | +0.0065 [+0.0055, +0.0075] | +0.0007 [+0.0002, +0.0011] | -0.0060 [-0.0071, -0.0049] |
| gen_joint_em | 0 dB | 0.8892 | 0.8455 | 0.20937 | 0.1300 | +0.0021 [+0.0015, +0.0026] | +0.0003 [+0.0000, +0.0005] | +0.0014 [+0.0006, +0.0022] |
| gen_joint_em | 10 dB | 0.9709 | 0.9531 | 0.07105 | 0.1294 | +0.0003 [+0.0001, +0.0006] | -0.0000 [-0.0001, +0.0001] | +0.0085 [+0.0078, +0.0091] |
| gen_joint_em | 20 dB | 0.9951 | 0.9914 | 0.02662 | 0.1161 | +0.0001 [+0.0001, +0.0002] | +0.0000 [-0.0000, +0.0000] | +0.0087 [+0.0081, +0.0092] |
| gen_joint_em0 | -5 dB | 0.8174 | 0.7615 | 0.32421 | 0.1449 | +0.0065 [+0.0055, +0.0075] | +0.0007 [+0.0002, +0.0011] | -0.0060 [-0.0071, -0.0049] |
| gen_joint_em0 | 0 dB | 0.8892 | 0.8455 | 0.20934 | 0.1292 | +0.0021 [+0.0016, +0.0027] | +0.0003 [+0.0001, +0.0006] | +0.0014 [+0.0006, +0.0022] |
| gen_joint_em0 | 10 dB | 0.9709 | 0.9531 | 0.07129 | 0.1304 | +0.0003 [+0.0001, +0.0006] | -0.0000 [-0.0001, +0.0001] | +0.0087 [+0.0080, +0.0094] |
| gen_joint_em0 | 20 dB | 0.9951 | 0.9914 | 0.02872 | 0.1236 | +0.0001 [+0.0001, +0.0002] | +0.0000 [-0.0000, +0.0000] | +0.0108 [+0.0101, +0.0115] |
| gen_joint_em0_rho | -5 dB | 0.8165 | 0.7600 | 0.32214 | 0.1169 | +0.0056 [+0.0046, +0.0067] | -0.0002 [-0.0008, +0.0003] | -0.0081 [-0.0092, -0.0071] |
| gen_joint_em0_rho | 0 dB | 0.8882 | 0.8433 | 0.20602 | 0.0734 | +0.0010 [+0.0006, +0.0015] | -0.0007 [-0.0012, -0.0003] | -0.0019 [-0.0024, -0.0015] |
| gen_joint_em0_rho | 10 dB | 0.9705 | 0.9524 | 0.06252 | 0.0290 | -0.0001 [-0.0002, +0.0001] | -0.0004 [-0.0006, -0.0002] | -0.0001 [-0.0002, +0.0001] |
| gen_joint_em0_rho | 20 dB | 0.9949 | 0.9911 | 0.01792 | 0.0071 | +0.0000 [-0.0000, +0.0001] | -0.0001 [-0.0002, -0.0001] | -0.0000 [-0.0001, +0.0000] |
| gen_joint_em_rho | -5 dB | 0.8165 | 0.7600 | 0.32216 | 0.1157 | +0.0056 [+0.0046, +0.0067] | -0.0002 [-0.0008, +0.0003] | -0.0081 [-0.0092, -0.0070] |
| gen_joint_em_rho | 0 dB | 0.8881 | 0.8433 | 0.20608 | 0.0735 | +0.0010 [+0.0005, +0.0015] | -0.0008 [-0.0012, -0.0004] | -0.0019 [-0.0024, -0.0014] |
| gen_joint_em_rho | 10 dB | 0.9705 | 0.9524 | 0.06255 | 0.0290 | -0.0001 [-0.0002, +0.0000] | -0.0004 [-0.0006, -0.0002] | -0.0000 [-0.0002, +0.0001] |
| gen_joint_em_rho | 20 dB | 0.9949 | 0.9911 | 0.01792 | 0.0071 | +0.0000 [-0.0000, +0.0001] | -0.0001 [-0.0002, -0.0001] | -0.0000 [-0.0001, +0.0000] |
| gen_joint_pilot | -5 dB | 0.7573 | 0.6843 | 0.53008 | 1.8996 | -0.0536 [-0.0561, -0.0514] | -0.0595 [-0.0617, -0.0573] | +0.1998 [+0.1922, +0.2088] |
| gen_joint_pilot | 0 dB | 0.8461 | 0.7835 | 0.46786 | 1.4048 | -0.0410 [-0.0430, -0.0390] | -0.0428 [-0.0448, -0.0407] | +0.2599 [+0.2429, +0.2794] |
| gen_joint_pilot | 10 dB | 0.9631 | 0.9393 | 0.31811 | 0.5194 | -0.0075 [-0.0084, -0.0067] | -0.0079 [-0.0087, -0.0071] | +0.2555 [+0.2313, +0.2811] |
| gen_joint_pilot | 20 dB | 0.9947 | 0.9907 | 0.06132 | 0.1587 | -0.0002 [-0.0004, -0.0001] | -0.0003 [-0.0005, -0.0003] | +0.0434 [+0.0355, +0.0526] |
| gen_joint_true | -5 dB | 0.8185 | 0.7639 | 0.32015 | - | +0.0076 [+0.0065, +0.0086] | +0.0017 [+0.0016, +0.0018] | -0.0101 [-0.0113, -0.0090] |
| gen_joint_true | 0 dB | 0.8892 | 0.8460 | 0.20280 | - | +0.0021 [+0.0016, +0.0027] | +0.0003 [+0.0003, +0.0004] | -0.0052 [-0.0058, -0.0045] |
| gen_joint_true | 10 dB | 0.9710 | 0.9531 | 0.05515 | - | +0.0004 [+0.0002, +0.0006] | -0.0000 [-0.0000, -0.0000] | -0.0074 [-0.0079, -0.0070] |
| gen_joint_true | 20 dB | 0.9951 | 0.9914 | 0.00956 | - | +0.0001 [+0.0001, +0.0002] | -0.0000 [-0.0000, -0.0000] | -0.0084 [-0.0088, -0.0080] |
| gen_joint_true0 | -5 dB | 0.8185 | 0.7639 | 0.32008 | - | +0.0076 [+0.0066, +0.0087] | +0.0018 [+0.0017, +0.0019] | -0.0102 [-0.0113, -0.0090] |
| gen_joint_true0 | 0 dB | 0.8893 | 0.8460 | 0.20274 | - | +0.0022 [+0.0016, +0.0027] | +0.0004 [+0.0003, +0.0004] | -0.0052 [-0.0059, -0.0045] |
| gen_joint_true0 | 10 dB | 0.9710 | 0.9532 | 0.05512 | - | +0.0004 [+0.0002, +0.0006] | +0.0000 [+0.0000, +0.0000] | -0.0075 [-0.0079, -0.0070] |
| gen_joint_true0 | 20 dB | 0.9951 | 0.9914 | 0.00956 | - | +0.0001 [+0.0001, +0.0002] | +0.0000 [-0.0000, +0.0000] | -0.0084 [-0.0088, -0.0080] |
| gen_joint_true0_rho | -5 dB | 0.8183 | 0.7631 | 0.32137 | - | +0.0074 [+0.0063, +0.0084] | +0.0015 [+0.0013, +0.0018] | -0.0089 [-0.0100, -0.0078] |
| gen_joint_true0_rho | 0 dB | 0.8888 | 0.8446 | 0.20625 | - | +0.0017 [+0.0013, +0.0022] | -0.0001 [-0.0004, +0.0003] | -0.0017 [-0.0022, -0.0013] |
| gen_joint_true0_rho | 10 dB | 0.9707 | 0.9527 | 0.06259 | - | +0.0001 [+0.0000, +0.0002] | -0.0003 [-0.0004, -0.0001] | +0.0000 [-0.0001, +0.0001] |
| gen_joint_true0_rho | 20 dB | 0.9950 | 0.9912 | 0.01795 | - | +0.0000 [-0.0000, +0.0001] | -0.0001 [-0.0002, -0.0001] | +0.0000 [-0.0001, +0.0001] |
| gen_joint_true_rho | -5 dB | 0.8182 | 0.7631 | 0.32144 | - | +0.0073 [+0.0063, +0.0084] | +0.0015 [+0.0012, +0.0017] | -0.0088 [-0.0100, -0.0077] |
| gen_joint_true_rho | 0 dB | 0.8888 | 0.8446 | 0.20631 | - | +0.0017 [+0.0012, +0.0022] | -0.0001 [-0.0004, +0.0002] | -0.0017 [-0.0022, -0.0012] |
| gen_joint_true_rho | 10 dB | 0.9707 | 0.9527 | 0.06261 | - | +0.0001 [+0.0000, +0.0002] | -0.0003 [-0.0004, -0.0001] | +0.0000 [-0.0001, +0.0001] |
| gen_joint_true_rho | 20 dB | 0.9950 | 0.9911 | 0.01795 | - | +0.0000 [-0.0000, +0.0001] | -0.0001 [-0.0002, -0.0001] | +0.0000 [-0.0001, +0.0001] |
| load_joint | -5 dB | 0.8167 | 0.7619 | 0.32283 | - | +0.0059 [+0.0048, +0.0069] | +0.0000 [+0.0000, +0.0000] | -0.0074 [-0.0086, -0.0063] |
| load_joint | 0 dB | 0.8889 | 0.8456 | 0.20336 | - | +0.0018 [+0.0012, +0.0024] | +0.0000 [+0.0000, +0.0000] | -0.0046 [-0.0053, -0.0039] |
| load_joint | 10 dB | 0.9710 | 0.9532 | 0.05513 | - | +0.0004 [+0.0002, +0.0006] | +0.0000 [+0.0000, +0.0000] | -0.0074 [-0.0079, -0.0070] |
| load_joint | 20 dB | 0.9951 | 0.9914 | 0.00956 | - | +0.0001 [+0.0001, +0.0002] | +0.0000 [+0.0000, +0.0000] | -0.0084 [-0.0088, -0.0080] |
| load_joint_blind | -5 dB | 0.8109 | 0.7506 | 0.33025 | 0.2032 | +0.0000 [+0.0000, +0.0000] | -0.0059 [-0.0069, -0.0048] | +0.0000 [+0.0000, +0.0000] |
| load_joint_blind | 0 dB | 0.8871 | 0.8418 | 0.20797 | 0.0923 | +0.0000 [+0.0000, +0.0000] | -0.0018 [-0.0024, -0.0012] | +0.0000 [+0.0000, +0.0000] |
| load_joint_blind | 10 dB | 0.9706 | 0.9525 | 0.06258 | 0.0184 | +0.0000 [+0.0000, +0.0000] | -0.0004 [-0.0006, -0.0002] | +0.0000 [+0.0000, +0.0000] |
| load_joint_blind | 20 dB | 0.9949 | 0.9911 | 0.01794 | 0.0052 | +0.0000 [+0.0000, +0.0000] | -0.0001 [-0.0002, -0.0001] | +0.0000 [+0.0000, +0.0000] |
| load_joint_em | -5 dB | 0.8156 | 0.7594 | 0.32696 | 0.1448 | +0.0047 [+0.0037, +0.0057] | -0.0012 [-0.0017, -0.0007] | -0.0033 [-0.0044, -0.0022] |
| load_joint_em | 0 dB | 0.8889 | 0.8451 | 0.20993 | 0.1289 | +0.0018 [+0.0012, +0.0023] | -0.0000 [-0.0003, +0.0002] | +0.0020 [+0.0012, +0.0028] |
| load_joint_em | 10 dB | 0.9709 | 0.9531 | 0.07127 | 0.1301 | +0.0003 [+0.0001, +0.0006] | -0.0000 [-0.0001, +0.0001] | +0.0087 [+0.0080, +0.0094] |
| load_joint_em | 20 dB | 0.9951 | 0.9914 | 0.02871 | 0.1235 | +0.0001 [+0.0001, +0.0002] | +0.0000 [-0.0000, +0.0000] | +0.0108 [+0.0101, +0.0115] |
| load_joint_em_rho | -5 dB | 0.8147 | 0.7577 | 0.32486 | 0.1162 | +0.0038 [+0.0028, +0.0048] | -0.0021 [-0.0027, -0.0015] | -0.0054 [-0.0065, -0.0044] |
| load_joint_em_rho | 0 dB | 0.8878 | 0.8429 | 0.20655 | 0.0730 | +0.0007 [+0.0002, +0.0012] | -0.0011 [-0.0015, -0.0007] | -0.0014 [-0.0019, -0.0010] |
| load_joint_em_rho | 10 dB | 0.9705 | 0.9524 | 0.06252 | 0.0290 | -0.0001 [-0.0002, +0.0001] | -0.0004 [-0.0006, -0.0002] | -0.0001 [-0.0002, +0.0001] |
| load_joint_em_rho | 20 dB | 0.9949 | 0.9911 | 0.01792 | 0.0071 | +0.0000 [-0.0000, +0.0001] | -0.0001 [-0.0002, -0.0001] | -0.0000 [-0.0001, +0.0000] |
| load_joint_rho | -5 dB | 0.8165 | 0.7612 | 0.32403 | - | +0.0056 [+0.0046, +0.0067] | -0.0002 [-0.0005, +0.0000] | -0.0062 [-0.0074, -0.0052] |
| load_joint_rho | 0 dB | 0.8885 | 0.8442 | 0.20680 | - | +0.0014 [+0.0009, +0.0019] | -0.0004 [-0.0008, -0.0001] | -0.0012 [-0.0017, -0.0007] |
| load_joint_rho | 10 dB | 0.9707 | 0.9527 | 0.06258 | - | +0.0001 [+0.0000, +0.0002] | -0.0003 [-0.0004, -0.0001] | +0.0000 [-0.0001, +0.0001] |
| load_joint_rho | 20 dB | 0.9950 | 0.9912 | 0.01794 | - | +0.0000 [-0.0000, +0.0001] | -0.0001 [-0.0002, -0.0001] | +0.0000 [-0.0001, +0.0001] |
| load_pu | -5 dB | 0.7948 | 0.7212 | 0.34866 | - | -0.0161 [-0.0181, -0.0142] | -0.0220 [-0.0236, -0.0202] | +0.0184 [+0.0166, +0.0202] |
| load_pu | 0 dB | 0.8760 | 0.8212 | 0.21971 | - | -0.0111 [-0.0126, -0.0099] | -0.0129 [-0.0145, -0.0117] | +0.0117 [+0.0104, +0.0131] |
| load_pu | 10 dB | 0.9683 | 0.9480 | 0.05851 | - | -0.0023 [-0.0031, -0.0016] | -0.0026 [-0.0034, -0.0020] | -0.0041 [-0.0049, -0.0032] |
| load_pu | 20 dB | 0.9948 | 0.9909 | 0.00993 | - | -0.0001 [-0.0003, +0.0000] | -0.0003 [-0.0004, -0.0001] | -0.0080 [-0.0084, -0.0075] |
| load_pu_blind | -5 dB | 0.7941 | 0.7197 | 0.35817 | 0.2032 | -0.0168 [-0.0188, -0.0149] | -0.0226 [-0.0243, -0.0209] | +0.0279 [+0.0260, +0.0298] |
| load_pu_blind | 0 dB | 0.8754 | 0.8196 | 0.22473 | 0.0923 | -0.0117 [-0.0131, -0.0106] | -0.0135 [-0.0150, -0.0122] | +0.0168 [+0.0155, +0.0183] |
| load_pu_blind | 10 dB | 0.9682 | 0.9478 | 0.06592 | 0.0184 | -0.0024 [-0.0032, -0.0018] | -0.0028 [-0.0036, -0.0021] | +0.0033 [+0.0027, +0.0040] |
| load_pu_blind | 20 dB | 0.9947 | 0.9906 | 0.01836 | 0.0052 | -0.0002 [-0.0004, -0.0001] | -0.0004 [-0.0005, -0.0002] | +0.0004 [+0.0002, +0.0006] |
| load_pu_rho | -5 dB | 0.7947 | 0.7208 | 0.34956 | - | -0.0162 [-0.0181, -0.0143] | -0.0220 [-0.0237, -0.0203] | +0.0193 [+0.0176, +0.0211] |
| load_pu_rho | 0 dB | 0.8757 | 0.8201 | 0.22278 | - | -0.0114 [-0.0128, -0.0103] | -0.0132 [-0.0147, -0.0120] | +0.0148 [+0.0137, +0.0161] |
| load_pu_rho | 10 dB | 0.9682 | 0.9478 | 0.06594 | - | -0.0024 [-0.0031, -0.0018] | -0.0028 [-0.0035, -0.0021] | +0.0034 [+0.0027, +0.0041] |
| load_pu_rho | 20 dB | 0.9947 | 0.9906 | 0.01835 | - | -0.0002 [-0.0004, -0.0001] | -0.0004 [-0.0005, -0.0002] | +0.0004 [+0.0003, +0.0006] |

### coco_K4_U2_E8_emauto0_fs_it30: frames 2000; data/shared/pilot/shared_pilot_coco_K4_U2_E8_emauto0_fs_it30_snr-5.json, data/shared/pilot/shared_pilot_coco_K4_U2_E8_emauto0_fs_it30_snr0.json, data/shared/pilot/shared_pilot_coco_K4_U2_E8_emauto0_fs_it30_snr10.json, data/shared/pilot/shared_pilot_coco_K4_U2_E8_emauto0_fs_it30_snr20.json

| mode | SNR | CosSim | weakest | NMSE | gain rel. RMSE | - load_joint_blind CosSim [95% CI] | - load_joint (true g) CosSim [95% CI] | - load_joint_blind NMSE [95% CI] |
|---|---|---|---|---|---|---|---|---|
| gen_joint_em | -5 dB | 0.8172 | 0.7614 | 0.32456 | 0.1393 | +0.0063 [+0.0053, +0.0073] | +0.0005 [-0.0000, +0.0009] | -0.0057 [-0.0068, -0.0045] |
| gen_joint_em | 0 dB | 0.8889 | 0.8451 | 0.20954 | 0.1298 | +0.0018 [+0.0013, +0.0024] | +0.0000 [-0.0003, +0.0003] | +0.0016 [+0.0008, +0.0024] |
| gen_joint_em | 10 dB | 0.9708 | 0.9529 | 0.06957 | 0.1234 | +0.0002 [+0.0000, +0.0005] | -0.0001 [-0.0002, -0.0000] | +0.0070 [+0.0064, +0.0076] |
| gen_joint_em | 20 dB | 0.9951 | 0.9913 | 0.01982 | 0.0894 | +0.0001 [+0.0000, +0.0002] | -0.0000 [-0.0000, +0.0000] | +0.0019 [+0.0015, +0.0023] |
| gen_joint_em_rho | -5 dB | 0.8163 | 0.7598 | 0.32258 | 0.1119 | +0.0054 [+0.0044, +0.0065] | -0.0004 [-0.0010, +0.0001] | -0.0077 [-0.0088, -0.0066] |
| gen_joint_em_rho | 0 dB | 0.8879 | 0.8430 | 0.20649 | 0.0746 | +0.0008 [+0.0003, +0.0012] | -0.0010 [-0.0015, -0.0006] | -0.0015 [-0.0020, -0.0010] |
| gen_joint_em_rho | 10 dB | 0.9704 | 0.9522 | 0.06271 | 0.0289 | -0.0002 [-0.0003, -0.0000] | -0.0005 [-0.0007, -0.0003] | +0.0001 [+0.0000, +0.0003] |
| gen_joint_em_rho | 20 dB | 0.9949 | 0.9911 | 0.01796 | 0.0071 | -0.0000 [-0.0001, +0.0000] | -0.0002 [-0.0002, -0.0001] | +0.0000 [-0.0000, +0.0001] |
| gen_joint_pilot | -5 dB | 0.7903 | 0.7227 | 0.42036 | 0.7970 | -0.0205 [-0.0223, -0.0187] | -0.0264 [-0.0280, -0.0247] | +0.0901 [+0.0842, +0.0963] |
| gen_joint_pilot | 0 dB | 0.8750 | 0.8226 | 0.28216 | 0.5753 | -0.0122 [-0.0134, -0.0111] | -0.0139 [-0.0152, -0.0129] | +0.0742 [+0.0697, +0.0794] |
| gen_joint_pilot | 10 dB | 0.9695 | 0.9504 | 0.08100 | 0.1993 | -0.0011 [-0.0015, -0.0008] | -0.0015 [-0.0018, -0.0013] | +0.0184 [+0.0166, +0.0205] |
| gen_joint_pilot | 20 dB | 0.9950 | 0.9912 | 0.01445 | 0.0586 | +0.0001 [-0.0000, +0.0002] | -0.0001 [-0.0001, -0.0001] | -0.0035 [-0.0042, -0.0027] |
| gen_joint_true | -5 dB | 0.8181 | 0.7635 | 0.32066 | - | +0.0073 [+0.0062, +0.0083] | +0.0014 [+0.0013, +0.0015] | -0.0096 [-0.0108, -0.0085] |
| gen_joint_true | 0 dB | 0.8890 | 0.8457 | 0.20323 | - | +0.0019 [+0.0013, +0.0025] | +0.0001 [+0.0001, +0.0001] | -0.0047 [-0.0054, -0.0040] |
| gen_joint_true | 10 dB | 0.9709 | 0.9530 | 0.05532 | - | +0.0003 [+0.0001, +0.0005] | -0.0001 [-0.0001, -0.0001] | -0.0073 [-0.0077, -0.0068] |
| gen_joint_true | 20 dB | 0.9951 | 0.9913 | 0.00960 | - | +0.0001 [+0.0000, +0.0002] | -0.0000 [-0.0000, -0.0000] | -0.0083 [-0.0087, -0.0079] |
| gen_joint_true_rho | -5 dB | 0.8179 | 0.7628 | 0.32194 | - | +0.0070 [+0.0060, +0.0081] | +0.0012 [+0.0009, +0.0014] | -0.0083 [-0.0095, -0.0072] |
| gen_joint_true_rho | 0 dB | 0.8885 | 0.8443 | 0.20673 | - | +0.0014 [+0.0010, +0.0019] | -0.0004 [-0.0007, -0.0000] | -0.0012 [-0.0018, -0.0008] |
| gen_joint_true_rho | 10 dB | 0.9706 | 0.9526 | 0.06278 | - | +0.0000 [-0.0001, +0.0001] | -0.0004 [-0.0005, -0.0002] | +0.0002 [+0.0001, +0.0003] |
| gen_joint_true_rho | 20 dB | 0.9949 | 0.9911 | 0.01798 | - | +0.0000 [-0.0000, +0.0001] | -0.0001 [-0.0002, -0.0001] | +0.0000 [-0.0000, +0.0001] |
| load_joint | -5 dB | 0.8167 | 0.7619 | 0.32283 | - | +0.0059 [+0.0048, +0.0069] | +0.0000 [+0.0000, +0.0000] | -0.0074 [-0.0086, -0.0063] |
| load_joint | 0 dB | 0.8889 | 0.8456 | 0.20336 | - | +0.0018 [+0.0012, +0.0024] | +0.0000 [+0.0000, +0.0000] | -0.0046 [-0.0053, -0.0039] |
| load_joint | 10 dB | 0.9710 | 0.9532 | 0.05513 | - | +0.0004 [+0.0002, +0.0006] | +0.0000 [+0.0000, +0.0000] | -0.0074 [-0.0079, -0.0070] |
| load_joint | 20 dB | 0.9951 | 0.9914 | 0.00956 | - | +0.0001 [+0.0001, +0.0002] | +0.0000 [+0.0000, +0.0000] | -0.0084 [-0.0088, -0.0080] |
| load_joint_blind | -5 dB | 0.8109 | 0.7506 | 0.33025 | 0.2032 | +0.0000 [+0.0000, +0.0000] | -0.0059 [-0.0069, -0.0048] | +0.0000 [+0.0000, +0.0000] |
| load_joint_blind | 0 dB | 0.8871 | 0.8418 | 0.20797 | 0.0923 | +0.0000 [+0.0000, +0.0000] | -0.0018 [-0.0024, -0.0012] | +0.0000 [+0.0000, +0.0000] |
| load_joint_blind | 10 dB | 0.9706 | 0.9525 | 0.06258 | 0.0184 | +0.0000 [+0.0000, +0.0000] | -0.0004 [-0.0006, -0.0002] | +0.0000 [+0.0000, +0.0000] |
| load_joint_blind | 20 dB | 0.9949 | 0.9911 | 0.01794 | 0.0052 | +0.0000 [+0.0000, +0.0000] | -0.0001 [-0.0002, -0.0001] | +0.0000 [+0.0000, +0.0000] |
| load_joint_rho | -5 dB | 0.8165 | 0.7612 | 0.32403 | - | +0.0056 [+0.0046, +0.0067] | -0.0002 [-0.0005, +0.0000] | -0.0062 [-0.0074, -0.0052] |
| load_joint_rho | 0 dB | 0.8885 | 0.8442 | 0.20680 | - | +0.0014 [+0.0009, +0.0019] | -0.0004 [-0.0008, -0.0001] | -0.0012 [-0.0017, -0.0007] |
| load_joint_rho | 10 dB | 0.9707 | 0.9527 | 0.06258 | - | +0.0001 [+0.0000, +0.0002] | -0.0003 [-0.0004, -0.0001] | +0.0000 [-0.0001, +0.0001] |
| load_joint_rho | 20 dB | 0.9950 | 0.9912 | 0.01794 | - | +0.0000 [-0.0000, +0.0001] | -0.0001 [-0.0002, -0.0001] | +0.0000 [-0.0001, +0.0001] |
| load_pu | -5 dB | 0.7948 | 0.7212 | 0.34866 | - | -0.0161 [-0.0181, -0.0142] | -0.0220 [-0.0236, -0.0202] | +0.0184 [+0.0166, +0.0202] |
| load_pu | 0 dB | 0.8760 | 0.8212 | 0.21971 | - | -0.0111 [-0.0126, -0.0099] | -0.0129 [-0.0145, -0.0117] | +0.0117 [+0.0104, +0.0131] |
| load_pu | 10 dB | 0.9683 | 0.9480 | 0.05851 | - | -0.0023 [-0.0031, -0.0016] | -0.0026 [-0.0034, -0.0020] | -0.0041 [-0.0049, -0.0032] |
| load_pu | 20 dB | 0.9948 | 0.9909 | 0.00993 | - | -0.0001 [-0.0003, +0.0000] | -0.0003 [-0.0004, -0.0001] | -0.0080 [-0.0084, -0.0075] |
| load_pu_blind | -5 dB | 0.7941 | 0.7197 | 0.35817 | 0.2032 | -0.0168 [-0.0188, -0.0149] | -0.0226 [-0.0243, -0.0209] | +0.0279 [+0.0260, +0.0298] |
| load_pu_blind | 0 dB | 0.8754 | 0.8196 | 0.22473 | 0.0923 | -0.0117 [-0.0131, -0.0106] | -0.0135 [-0.0150, -0.0122] | +0.0168 [+0.0155, +0.0183] |
| load_pu_blind | 10 dB | 0.9682 | 0.9478 | 0.06592 | 0.0184 | -0.0024 [-0.0032, -0.0018] | -0.0028 [-0.0036, -0.0021] | +0.0033 [+0.0027, +0.0040] |
| load_pu_blind | 20 dB | 0.9947 | 0.9906 | 0.01836 | 0.0052 | -0.0002 [-0.0004, -0.0001] | -0.0004 [-0.0005, -0.0002] | +0.0004 [+0.0002, +0.0006] |
| load_pu_rho | -5 dB | 0.7947 | 0.7208 | 0.34956 | - | -0.0162 [-0.0181, -0.0143] | -0.0220 [-0.0237, -0.0203] | +0.0193 [+0.0176, +0.0211] |
| load_pu_rho | 0 dB | 0.8757 | 0.8201 | 0.22278 | - | -0.0114 [-0.0128, -0.0103] | -0.0132 [-0.0147, -0.0120] | +0.0148 [+0.0137, +0.0161] |
| load_pu_rho | 10 dB | 0.9682 | 0.9478 | 0.06594 | - | -0.0024 [-0.0031, -0.0018] | -0.0028 [-0.0035, -0.0021] | +0.0034 [+0.0027, +0.0041] |
| load_pu_rho | 20 dB | 0.9947 | 0.9906 | 0.01835 | - | -0.0002 [-0.0004, -0.0001] | -0.0004 [-0.0005, -0.0002] | +0.0004 [+0.0003, +0.0006] |

### coco_K4_U4_E1_emauto0_fs_it30: frames 2000; data/shared/pilot/shared_pilot_coco_K4_U4_E1_emauto0_fs_it30_snr-5.json, data/shared/pilot/shared_pilot_coco_K4_U4_E1_emauto0_fs_it30_snr0.json, data/shared/pilot/shared_pilot_coco_K4_U4_E1_emauto0_fs_it30_snr10.json, data/shared/pilot/shared_pilot_coco_K4_U4_E1_emauto0_fs_it30_snr20.json

| mode | SNR | CosSim | weakest | NMSE | gain rel. RMSE | - load_joint_blind CosSim [95% CI] | - load_joint (true g) CosSim [95% CI] | - load_joint_blind NMSE [95% CI] |
|---|---|---|---|---|---|---|---|---|
| gen_joint_em | -5 dB | 0.7852 | 0.6737 | 0.37873 | 0.1850 | +0.0140 [+0.0130, +0.0150] | +0.0045 [+0.0042, +0.0049] | -0.0168 [-0.0180, -0.0155] |
| gen_joint_em | 0 dB | 0.8574 | 0.7561 | 0.26304 | 0.1288 | +0.0039 [+0.0035, +0.0044] | +0.0011 [+0.0010, +0.0013] | -0.0022 [-0.0029, -0.0016] |
| gen_joint_em | 10 dB | 0.9573 | 0.9031 | 0.09462 | 0.1285 | +0.0007 [+0.0004, +0.0009] | +0.0002 [+0.0001, +0.0002] | +0.0064 [+0.0058, +0.0069] |
| gen_joint_em | 20 dB | 0.9917 | 0.9766 | 0.03185 | 0.1201 | +0.0001 [+0.0001, +0.0002] | +0.0000 [-0.0000, +0.0000] | +0.0077 [+0.0072, +0.0081] |
| gen_joint_em0 | -5 dB | 0.7853 | 0.6741 | 0.37856 | 0.1852 | +0.0142 [+0.0132, +0.0152] | +0.0047 [+0.0043, +0.0050] | -0.0170 [-0.0182, -0.0158] |
| gen_joint_em0 | 0 dB | 0.8574 | 0.7562 | 0.26296 | 0.1292 | +0.0040 [+0.0036, +0.0045] | +0.0012 [+0.0010, +0.0014] | -0.0023 [-0.0029, -0.0016] |
| gen_joint_em0 | 10 dB | 0.9573 | 0.9032 | 0.09476 | 0.1298 | +0.0007 [+0.0005, +0.0009] | +0.0002 [+0.0001, +0.0003] | +0.0065 [+0.0059, +0.0071] |
| gen_joint_em0 | 20 dB | 0.9917 | 0.9766 | 0.03381 | 0.1280 | +0.0001 [+0.0001, +0.0002] | +0.0000 [+0.0000, +0.0000] | +0.0097 [+0.0091, +0.0102] |
| gen_joint_em0_rho | -5 dB | 0.7846 | 0.6725 | 0.37659 | 0.1673 | +0.0135 [+0.0125, +0.0145] | +0.0040 [+0.0036, +0.0043] | -0.0190 [-0.0201, -0.0178] |
| gen_joint_em0_rho | 0 dB | 0.8564 | 0.7532 | 0.26041 | 0.0928 | +0.0030 [+0.0026, +0.0034] | +0.0002 [-0.0001, +0.0005] | -0.0048 [-0.0053, -0.0044] |
| gen_joint_em0_rho | 10 dB | 0.9567 | 0.9012 | 0.08786 | 0.0348 | +0.0001 [-0.0000, +0.0002] | -0.0004 [-0.0006, -0.0002] | -0.0004 [-0.0006, -0.0003] |
| gen_joint_em0_rho | 20 dB | 0.9916 | 0.9761 | 0.02412 | 0.0100 | -0.0000 [-0.0000, +0.0000] | -0.0001 [-0.0002, -0.0001] | -0.0000 [-0.0001, -0.0000] |
| gen_joint_em_rho | -5 dB | 0.7844 | 0.6719 | 0.37684 | 0.1675 | +0.0133 [+0.0123, +0.0142] | +0.0038 [+0.0034, +0.0041] | -0.0187 [-0.0198, -0.0175] |
| gen_joint_em_rho | 0 dB | 0.8564 | 0.7531 | 0.26053 | 0.0930 | +0.0029 [+0.0025, +0.0033] | +0.0001 [-0.0002, +0.0004] | -0.0047 [-0.0052, -0.0042] |
| gen_joint_em_rho | 10 dB | 0.9567 | 0.9012 | 0.08792 | 0.0349 | +0.0000 [-0.0001, +0.0002] | -0.0005 [-0.0006, -0.0003] | -0.0003 [-0.0005, -0.0002] |
| gen_joint_em_rho | 20 dB | 0.9916 | 0.9761 | 0.02414 | 0.0100 | -0.0000 [-0.0000, -0.0000] | -0.0001 [-0.0002, -0.0001] | -0.0000 [-0.0000, +0.0000] |
| gen_joint_pilot | -5 dB | 0.7221 | 0.5814 | 0.55159 | 2.4056 | -0.0490 [-0.0509, -0.0470] | -0.0585 [-0.0602, -0.0567] | +0.1560 [+0.1505, +0.1620] |
| gen_joint_pilot | 0 dB | 0.8160 | 0.6846 | 0.42621 | 1.3114 | -0.0375 [-0.0390, -0.0360] | -0.0403 [-0.0417, -0.0388] | +0.1610 [+0.1545, +0.1684] |
| gen_joint_pilot | 10 dB | 0.9492 | 0.8814 | 0.22390 | 0.5027 | -0.0074 [-0.0079, -0.0069] | -0.0079 [-0.0084, -0.0074] | +0.1356 [+0.1275, +0.1433] |
| gen_joint_pilot | 20 dB | 0.9911 | 0.9742 | 0.05644 | 0.1779 | -0.0005 [-0.0006, -0.0004] | -0.0006 [-0.0007, -0.0005] | +0.0323 [+0.0285, +0.0365] |
| gen_joint_true | -5 dB | 0.7865 | 0.6787 | 0.37460 | - | +0.0154 [+0.0144, +0.0164] | +0.0059 [+0.0057, +0.0061] | -0.0210 [-0.0221, -0.0197] |
| gen_joint_true | 0 dB | 0.8576 | 0.7580 | 0.25821 | - | +0.0042 [+0.0037, +0.0047] | +0.0014 [+0.0013, +0.0015] | -0.0071 [-0.0077, -0.0065] |
| gen_joint_true | 10 dB | 0.9571 | 0.9029 | 0.08167 | - | +0.0005 [+0.0003, +0.0007] | -0.0000 [-0.0000, -0.0000] | -0.0066 [-0.0070, -0.0062] |
| gen_joint_true | 20 dB | 0.9917 | 0.9765 | 0.01618 | - | +0.0001 [+0.0000, +0.0002] | -0.0000 [-0.0000, -0.0000] | -0.0080 [-0.0083, -0.0077] |
| gen_joint_true0 | -5 dB | 0.7866 | 0.6789 | 0.37444 | - | +0.0155 [+0.0145, +0.0165] | +0.0060 [+0.0058, +0.0062] | -0.0211 [-0.0223, -0.0198] |
| gen_joint_true0 | 0 dB | 0.8577 | 0.7580 | 0.25808 | - | +0.0043 [+0.0038, +0.0047] | +0.0015 [+0.0014, +0.0015] | -0.0072 [-0.0078, -0.0066] |
| gen_joint_true0 | 10 dB | 0.9571 | 0.9030 | 0.08160 | - | +0.0005 [+0.0003, +0.0007] | +0.0000 [+0.0000, +0.0000] | -0.0067 [-0.0071, -0.0063] |
| gen_joint_true0 | 20 dB | 0.9917 | 0.9765 | 0.01616 | - | +0.0001 [+0.0000, +0.0002] | +0.0000 [+0.0000, +0.0000] | -0.0080 [-0.0083, -0.0077] |
| gen_joint_true0_rho | -5 dB | 0.7865 | 0.6778 | 0.37522 | - | +0.0154 [+0.0143, +0.0163] | +0.0059 [+0.0056, +0.0061] | -0.0203 [-0.0215, -0.0191] |
| gen_joint_true0_rho | 0 dB | 0.8573 | 0.7559 | 0.26025 | - | +0.0039 [+0.0035, +0.0043] | +0.0011 [+0.0009, +0.0013] | -0.0050 [-0.0055, -0.0045] |
| gen_joint_true0_rho | 10 dB | 0.9568 | 0.9017 | 0.08809 | - | +0.0002 [+0.0001, +0.0003] | -0.0003 [-0.0005, -0.0001] | -0.0002 [-0.0003, -0.0000] |
| gen_joint_true0_rho | 20 dB | 0.9916 | 0.9762 | 0.02418 | - | +0.0000 [-0.0000, +0.0000] | -0.0001 [-0.0002, -0.0000] | +0.0000 [-0.0000, +0.0001] |
| gen_joint_true_rho | -5 dB | 0.7864 | 0.6776 | 0.37538 | - | +0.0153 [+0.0142, +0.0163] | +0.0058 [+0.0055, +0.0060] | -0.0202 [-0.0213, -0.0190] |
| gen_joint_true_rho | 0 dB | 0.8572 | 0.7558 | 0.26038 | - | +0.0038 [+0.0034, +0.0042] | +0.0010 [+0.0008, +0.0012] | -0.0049 [-0.0054, -0.0043] |
| gen_joint_true_rho | 10 dB | 0.9568 | 0.9017 | 0.08815 | - | +0.0002 [+0.0001, +0.0003] | -0.0003 [-0.0005, -0.0002] | -0.0001 [-0.0003, +0.0000] |
| gen_joint_true_rho | 20 dB | 0.9916 | 0.9761 | 0.02420 | - | -0.0000 [-0.0000, +0.0000] | -0.0001 [-0.0002, -0.0001] | +0.0000 [+0.0000, +0.0001] |
| load_joint | -5 dB | 0.7806 | 0.6717 | 0.38357 | - | +0.0095 [+0.0085, +0.0104] | +0.0000 [+0.0000, +0.0000] | -0.0120 [-0.0131, -0.0108] |
| load_joint | 0 dB | 0.8562 | 0.7562 | 0.26049 | - | +0.0028 [+0.0023, +0.0033] | +0.0000 [+0.0000, +0.0000] | -0.0048 [-0.0054, -0.0042] |
| load_joint | 10 dB | 0.9571 | 0.9030 | 0.08165 | - | +0.0005 [+0.0003, +0.0007] | +0.0000 [+0.0000, +0.0000] | -0.0066 [-0.0070, -0.0062] |
| load_joint | 20 dB | 0.9917 | 0.9765 | 0.01616 | - | +0.0001 [+0.0000, +0.0002] | +0.0000 [+0.0000, +0.0000] | -0.0080 [-0.0083, -0.0077] |
| load_joint_blind | -5 dB | 0.7711 | 0.6453 | 0.39555 | 0.4069 | +0.0000 [+0.0000, +0.0000] | -0.0095 [-0.0104, -0.0085] | +0.0000 [+0.0000, +0.0000] |
| load_joint_blind | 0 dB | 0.8534 | 0.7458 | 0.26526 | 0.1560 | +0.0000 [+0.0000, +0.0000] | -0.0028 [-0.0033, -0.0023] | +0.0000 [+0.0000, +0.0000] |
| load_joint_blind | 10 dB | 0.9566 | 0.9010 | 0.08827 | 0.0348 | +0.0000 [+0.0000, +0.0000] | -0.0005 [-0.0007, -0.0003] | +0.0000 [+0.0000, +0.0000] |
| load_joint_blind | 20 dB | 0.9916 | 0.9762 | 0.02415 | 0.0080 | +0.0000 [+0.0000, +0.0000] | -0.0001 [-0.0002, -0.0000] | +0.0000 [+0.0000, +0.0000] |
| load_joint_em | -5 dB | 0.7793 | 0.6667 | 0.38751 | 0.1816 | +0.0082 [+0.0072, +0.0091] | -0.0013 [-0.0016, -0.0011] | -0.0080 [-0.0091, -0.0069] |
| load_joint_em | 0 dB | 0.8560 | 0.7545 | 0.26532 | 0.1282 | +0.0026 [+0.0022, +0.0031] | -0.0002 [-0.0004, -0.0000] | +0.0001 [-0.0006, +0.0007] |
| load_joint_em | 10 dB | 0.9573 | 0.9031 | 0.09485 | 0.1291 | +0.0007 [+0.0004, +0.0009] | +0.0002 [+0.0001, +0.0002] | +0.0066 [+0.0060, +0.0072] |
| load_joint_em | 20 dB | 0.9917 | 0.9766 | 0.03382 | 0.1280 | +0.0001 [+0.0001, +0.0002] | +0.0000 [+0.0000, +0.0000] | +0.0097 [+0.0092, +0.0102] |
| load_joint_em_rho | -5 dB | 0.7786 | 0.6653 | 0.38538 | 0.1625 | +0.0075 [+0.0065, +0.0084] | -0.0020 [-0.0023, -0.0016] | -0.0102 [-0.0112, -0.0090] |
| load_joint_em_rho | 0 dB | 0.8551 | 0.7517 | 0.26256 | 0.0906 | +0.0017 [+0.0012, +0.0020] | -0.0011 [-0.0014, -0.0009] | -0.0027 [-0.0031, -0.0022] |
| load_joint_em_rho | 10 dB | 0.9567 | 0.9012 | 0.08788 | 0.0348 | +0.0001 [-0.0000, +0.0002] | -0.0004 [-0.0006, -0.0003] | -0.0004 [-0.0005, -0.0002] |
| load_joint_em_rho | 20 dB | 0.9916 | 0.9761 | 0.02412 | 0.0100 | -0.0000 [-0.0000, +0.0000] | -0.0001 [-0.0002, -0.0001] | -0.0000 [-0.0001, -0.0000] |
| load_joint_rho | -5 dB | 0.7805 | 0.6710 | 0.38415 | - | +0.0094 [+0.0084, +0.0103] | -0.0001 [-0.0002, +0.0001] | -0.0114 [-0.0125, -0.0103] |
| load_joint_rho | 0 dB | 0.8559 | 0.7542 | 0.26251 | - | +0.0025 [+0.0020, +0.0029] | -0.0003 [-0.0006, -0.0001] | -0.0028 [-0.0032, -0.0022] |
| load_joint_rho | 10 dB | 0.9568 | 0.9017 | 0.08811 | - | +0.0002 [+0.0001, +0.0003] | -0.0003 [-0.0005, -0.0001] | -0.0002 [-0.0003, -0.0000] |
| load_joint_rho | 20 dB | 0.9916 | 0.9762 | 0.02418 | - | +0.0000 [-0.0000, +0.0000] | -0.0001 [-0.0002, -0.0000] | +0.0000 [-0.0000, +0.0001] |
| load_pu | -5 dB | 0.7352 | 0.5578 | 0.43725 | - | -0.0359 [-0.0378, -0.0342] | -0.0454 [-0.0473, -0.0436] | +0.0417 [+0.0400, +0.0435] |
| load_pu | 0 dB | 0.8304 | 0.6845 | 0.29439 | - | -0.0230 [-0.0242, -0.0217] | -0.0258 [-0.0270, -0.0245] | +0.0291 [+0.0277, +0.0304] |
| load_pu | 10 dB | 0.9514 | 0.8836 | 0.08991 | - | -0.0052 [-0.0058, -0.0046] | -0.0057 [-0.0063, -0.0051] | +0.0016 [+0.0008, +0.0025] |
| load_pu | 20 dB | 0.9911 | 0.9742 | 0.01712 | - | -0.0005 [-0.0007, -0.0003] | -0.0006 [-0.0008, -0.0005] | -0.0070 [-0.0074, -0.0066] |
| load_pu_blind | -5 dB | 0.7338 | 0.5548 | 0.45429 | 0.4069 | -0.0373 [-0.0392, -0.0355] | -0.0468 [-0.0487, -0.0450] | +0.0587 [+0.0569, +0.0607] |
| load_pu_blind | 0 dB | 0.8299 | 0.6822 | 0.30057 | 0.1560 | -0.0235 [-0.0248, -0.0222] | -0.0263 [-0.0276, -0.0250] | +0.0353 [+0.0339, +0.0367] |
| load_pu_blind | 10 dB | 0.9512 | 0.8826 | 0.09666 | 0.0348 | -0.0054 [-0.0060, -0.0048] | -0.0059 [-0.0065, -0.0053] | +0.0084 [+0.0077, +0.0091] |
| load_pu_blind | 20 dB | 0.9910 | 0.9739 | 0.02518 | 0.0080 | -0.0006 [-0.0007, -0.0005] | -0.0007 [-0.0009, -0.0005] | +0.0010 [+0.0009, +0.0012] |
| load_pu_rho | -5 dB | 0.7352 | 0.5576 | 0.43748 | - | -0.0359 [-0.0378, -0.0341] | -0.0454 [-0.0473, -0.0436] | +0.0419 [+0.0403, +0.0437] |
| load_pu_rho | 0 dB | 0.8303 | 0.6832 | 0.29597 | - | -0.0231 [-0.0244, -0.0219] | -0.0259 [-0.0272, -0.0247] | +0.0307 [+0.0294, +0.0320] |
| load_pu_rho | 10 dB | 0.9512 | 0.8827 | 0.09638 | - | -0.0054 [-0.0060, -0.0048] | -0.0059 [-0.0065, -0.0053] | +0.0081 [+0.0075, +0.0088] |
| load_pu_rho | 20 dB | 0.9910 | 0.9739 | 0.02521 | - | -0.0006 [-0.0007, -0.0005] | -0.0007 [-0.0009, -0.0005] | +0.0011 [+0.0009, +0.0013] |

### coco_K4_U4_E8_emauto0_fs_it30: frames 2000; data/shared/pilot/shared_pilot_coco_K4_U4_E8_emauto0_fs_it30_snr-5.json, data/shared/pilot/shared_pilot_coco_K4_U4_E8_emauto0_fs_it30_snr0.json, data/shared/pilot/shared_pilot_coco_K4_U4_E8_emauto0_fs_it30_snr10.json, data/shared/pilot/shared_pilot_coco_K4_U4_E8_emauto0_fs_it30_snr20.json

| mode | SNR | CosSim | weakest | NMSE | gain rel. RMSE | - load_joint_blind CosSim [95% CI] | - load_joint (true g) CosSim [95% CI] | - load_joint_blind NMSE [95% CI] |
|---|---|---|---|---|---|---|---|---|
| gen_joint_em | -5 dB | 0.7846 | 0.6734 | 0.37952 | 0.1837 | +0.0135 [+0.0125, +0.0145] | +0.0040 [+0.0037, +0.0043] | -0.0160 [-0.0172, -0.0148] |
| gen_joint_em | 0 dB | 0.8569 | 0.7555 | 0.26364 | 0.1263 | +0.0034 [+0.0030, +0.0039] | +0.0006 [+0.0004, +0.0008] | -0.0016 [-0.0022, -0.0010] |
| gen_joint_em | 10 dB | 0.9570 | 0.9026 | 0.09341 | 0.1209 | +0.0004 [+0.0002, +0.0006] | -0.0001 [-0.0002, -0.0000] | +0.0051 [+0.0047, +0.0056] |
| gen_joint_em | 20 dB | 0.9917 | 0.9764 | 0.02555 | 0.0928 | +0.0001 [+0.0000, +0.0001] | -0.0000 [-0.0001, -0.0000] | +0.0014 [+0.0011, +0.0016] |
| gen_joint_em_rho | -5 dB | 0.7838 | 0.6715 | 0.37775 | 0.1669 | +0.0127 [+0.0118, +0.0137] | +0.0032 [+0.0029, +0.0036] | -0.0178 [-0.0189, -0.0166] |
| gen_joint_em_rho | 0 dB | 0.8559 | 0.7526 | 0.26138 | 0.0922 | +0.0024 [+0.0020, +0.0028] | -0.0004 [-0.0007, -0.0001] | -0.0039 [-0.0043, -0.0034] |
| gen_joint_em_rho | 10 dB | 0.9564 | 0.9007 | 0.08838 | 0.0348 | -0.0002 [-0.0003, -0.0001] | -0.0007 [-0.0009, -0.0005] | +0.0001 [-0.0000, +0.0002] |
| gen_joint_em_rho | 20 dB | 0.9915 | 0.9759 | 0.02425 | 0.0099 | -0.0001 [-0.0001, -0.0001] | -0.0002 [-0.0002, -0.0001] | +0.0001 [+0.0001, +0.0001] |
| gen_joint_pilot | -5 dB | 0.7582 | 0.6285 | 0.44683 | 0.9722 | -0.0129 [-0.0143, -0.0115] | -0.0224 [-0.0237, -0.0213] | +0.0513 [+0.0487, +0.0542] |
| gen_joint_pilot | 0 dB | 0.8446 | 0.7301 | 0.30868 | 0.5420 | -0.0089 [-0.0097, -0.0080] | -0.0117 [-0.0125, -0.0109] | +0.0434 [+0.0414, +0.0455] |
| gen_joint_pilot | 10 dB | 0.9554 | 0.8980 | 0.09972 | 0.1980 | -0.0012 [-0.0014, -0.0009] | -0.0017 [-0.0019, -0.0015] | +0.0115 [+0.0104, +0.0124] |
| gen_joint_pilot | 20 dB | 0.9915 | 0.9759 | 0.02055 | 0.0665 | -0.0001 [-0.0001, -0.0000] | -0.0002 [-0.0002, -0.0001] | -0.0036 [-0.0041, -0.0031] |
| gen_joint_true | -5 dB | 0.7859 | 0.6780 | 0.37558 | - | +0.0148 [+0.0138, +0.0158] | +0.0053 [+0.0051, +0.0055] | -0.0200 [-0.0211, -0.0187] |
| gen_joint_true | 0 dB | 0.8571 | 0.7573 | 0.25910 | - | +0.0037 [+0.0032, +0.0041] | +0.0009 [+0.0008, +0.0009] | -0.0062 [-0.0068, -0.0056] |
| gen_joint_true | 10 dB | 0.9568 | 0.9025 | 0.08214 | - | +0.0002 [+0.0000, +0.0004] | -0.0003 [-0.0003, -0.0003] | -0.0061 [-0.0065, -0.0057] |
| gen_joint_true | 20 dB | 0.9917 | 0.9763 | 0.01629 | - | +0.0000 [-0.0000, +0.0001] | -0.0001 [-0.0001, -0.0001] | -0.0079 [-0.0082, -0.0075] |
| gen_joint_true_rho | -5 dB | 0.7858 | 0.6769 | 0.37634 | - | +0.0147 [+0.0136, +0.0156] | +0.0051 [+0.0049, +0.0054] | -0.0192 [-0.0203, -0.0180] |
| gen_joint_true_rho | 0 dB | 0.8567 | 0.7552 | 0.26125 | - | +0.0033 [+0.0029, +0.0037] | +0.0005 [+0.0003, +0.0007] | -0.0040 [-0.0045, -0.0035] |
| gen_joint_true_rho | 10 dB | 0.9565 | 0.9012 | 0.08861 | - | -0.0001 [-0.0002, +0.0000] | -0.0006 [-0.0007, -0.0004] | +0.0003 [+0.0002, +0.0005] |
| gen_joint_true_rho | 20 dB | 0.9916 | 0.9760 | 0.02431 | - | -0.0001 [-0.0001, -0.0001] | -0.0002 [-0.0002, -0.0001] | +0.0002 [+0.0001, +0.0002] |
| load_joint | -5 dB | 0.7806 | 0.6717 | 0.38357 | - | +0.0095 [+0.0085, +0.0104] | +0.0000 [+0.0000, +0.0000] | -0.0120 [-0.0131, -0.0108] |
| load_joint | 0 dB | 0.8562 | 0.7562 | 0.26049 | - | +0.0028 [+0.0023, +0.0033] | +0.0000 [+0.0000, +0.0000] | -0.0048 [-0.0054, -0.0042] |
| load_joint | 10 dB | 0.9571 | 0.9030 | 0.08165 | - | +0.0005 [+0.0003, +0.0007] | +0.0000 [+0.0000, +0.0000] | -0.0066 [-0.0070, -0.0062] |
| load_joint | 20 dB | 0.9917 | 0.9765 | 0.01616 | - | +0.0001 [+0.0000, +0.0002] | +0.0000 [+0.0000, +0.0000] | -0.0080 [-0.0083, -0.0077] |
| load_joint_blind | -5 dB | 0.7711 | 0.6453 | 0.39555 | 0.4069 | +0.0000 [+0.0000, +0.0000] | -0.0095 [-0.0104, -0.0085] | +0.0000 [+0.0000, +0.0000] |
| load_joint_blind | 0 dB | 0.8534 | 0.7458 | 0.26526 | 0.1560 | +0.0000 [+0.0000, +0.0000] | -0.0028 [-0.0033, -0.0023] | +0.0000 [+0.0000, +0.0000] |
| load_joint_blind | 10 dB | 0.9566 | 0.9010 | 0.08827 | 0.0348 | +0.0000 [+0.0000, +0.0000] | -0.0005 [-0.0007, -0.0003] | +0.0000 [+0.0000, +0.0000] |
| load_joint_blind | 20 dB | 0.9916 | 0.9762 | 0.02415 | 0.0080 | +0.0000 [+0.0000, +0.0000] | -0.0001 [-0.0002, -0.0000] | +0.0000 [+0.0000, +0.0000] |
| load_joint_rho | -5 dB | 0.7805 | 0.6710 | 0.38415 | - | +0.0094 [+0.0084, +0.0103] | -0.0001 [-0.0002, +0.0001] | -0.0114 [-0.0125, -0.0103] |
| load_joint_rho | 0 dB | 0.8559 | 0.7542 | 0.26251 | - | +0.0025 [+0.0020, +0.0029] | -0.0003 [-0.0006, -0.0001] | -0.0028 [-0.0032, -0.0022] |
| load_joint_rho | 10 dB | 0.9568 | 0.9017 | 0.08811 | - | +0.0002 [+0.0001, +0.0003] | -0.0003 [-0.0005, -0.0001] | -0.0002 [-0.0003, -0.0000] |
| load_joint_rho | 20 dB | 0.9916 | 0.9762 | 0.02418 | - | +0.0000 [-0.0000, +0.0000] | -0.0001 [-0.0002, -0.0000] | +0.0000 [-0.0000, +0.0001] |
| load_pu | -5 dB | 0.7352 | 0.5578 | 0.43725 | - | -0.0359 [-0.0378, -0.0342] | -0.0454 [-0.0473, -0.0436] | +0.0417 [+0.0400, +0.0435] |
| load_pu | 0 dB | 0.8304 | 0.6845 | 0.29439 | - | -0.0230 [-0.0242, -0.0217] | -0.0258 [-0.0270, -0.0245] | +0.0291 [+0.0277, +0.0304] |
| load_pu | 10 dB | 0.9514 | 0.8836 | 0.08991 | - | -0.0052 [-0.0058, -0.0046] | -0.0057 [-0.0063, -0.0051] | +0.0016 [+0.0008, +0.0025] |
| load_pu | 20 dB | 0.9911 | 0.9742 | 0.01712 | - | -0.0005 [-0.0007, -0.0003] | -0.0006 [-0.0008, -0.0005] | -0.0070 [-0.0074, -0.0066] |
| load_pu_blind | -5 dB | 0.7338 | 0.5548 | 0.45429 | 0.4069 | -0.0373 [-0.0392, -0.0355] | -0.0468 [-0.0487, -0.0450] | +0.0587 [+0.0569, +0.0607] |
| load_pu_blind | 0 dB | 0.8299 | 0.6822 | 0.30057 | 0.1560 | -0.0235 [-0.0248, -0.0222] | -0.0263 [-0.0276, -0.0250] | +0.0353 [+0.0339, +0.0367] |
| load_pu_blind | 10 dB | 0.9512 | 0.8826 | 0.09666 | 0.0348 | -0.0054 [-0.0060, -0.0048] | -0.0059 [-0.0065, -0.0053] | +0.0084 [+0.0077, +0.0091] |
| load_pu_blind | 20 dB | 0.9910 | 0.9739 | 0.02518 | 0.0080 | -0.0006 [-0.0007, -0.0005] | -0.0007 [-0.0009, -0.0005] | +0.0010 [+0.0009, +0.0012] |
| load_pu_rho | -5 dB | 0.7352 | 0.5576 | 0.43748 | - | -0.0359 [-0.0378, -0.0341] | -0.0454 [-0.0473, -0.0436] | +0.0419 [+0.0403, +0.0437] |
| load_pu_rho | 0 dB | 0.8303 | 0.6832 | 0.29597 | - | -0.0231 [-0.0244, -0.0219] | -0.0259 [-0.0272, -0.0247] | +0.0307 [+0.0294, +0.0320] |
| load_pu_rho | 10 dB | 0.9512 | 0.8827 | 0.09638 | - | -0.0054 [-0.0060, -0.0048] | -0.0059 [-0.0065, -0.0053] | +0.0081 [+0.0075, +0.0088] |
| load_pu_rho | 20 dB | 0.9910 | 0.9739 | 0.02521 | - | -0.0006 [-0.0007, -0.0005] | -0.0007 [-0.0009, -0.0005] | +0.0011 [+0.0009, +0.0013] |

### coco_K4_U4_ph10_E1_emauto0_fs_it30: frames 2000; data/shared/pilot/shared_pilot_coco_K4_U4_ph10_E1_emauto0_fs_it30_snr-5.json, data/shared/pilot/shared_pilot_coco_K4_U4_ph10_E1_emauto0_fs_it30_snr0.json, data/shared/pilot/shared_pilot_coco_K4_U4_ph10_E1_emauto0_fs_it30_snr10.json, data/shared/pilot/shared_pilot_coco_K4_U4_ph10_E1_emauto0_fs_it30_snr20.json

| mode | SNR | CosSim | weakest | NMSE | gain rel. RMSE | - load_joint_blind CosSim [95% CI] | - load_joint (true g) CosSim [95% CI] | - load_joint_blind NMSE [95% CI] |
|---|---|---|---|---|---|---|---|---|
| gen_joint_em | -5 dB | 0.7829 | 0.6708 | 0.38222 | 0.1873 | +0.0145 [+0.0134, +0.0155] | +0.0046 [+0.0042, +0.0049] | -0.0173 [-0.0185, -0.0160] |
| gen_joint_em | 0 dB | 0.8554 | 0.7533 | 0.26630 | 0.1298 | +0.0041 [+0.0036, +0.0045] | +0.0011 [+0.0010, +0.0013] | -0.0024 [-0.0031, -0.0018] |
| gen_joint_em | 10 dB | 0.9564 | 0.9014 | 0.09620 | 0.1286 | +0.0007 [+0.0005, +0.0009] | +0.0002 [+0.0001, +0.0002] | +0.0063 [+0.0057, +0.0069] |
| gen_joint_em | 20 dB | 0.9915 | 0.9760 | 0.03227 | 0.1204 | +0.0001 [+0.0001, +0.0002] | +0.0000 [+0.0000, +0.0000] | +0.0077 [+0.0073, +0.0081] |
| gen_joint_em0 | -5 dB | 0.7830 | 0.6713 | 0.38205 | 0.1875 | +0.0146 [+0.0136, +0.0156] | +0.0047 [+0.0044, +0.0050] | -0.0175 [-0.0187, -0.0162] |
| gen_joint_em0 | 0 dB | 0.8554 | 0.7534 | 0.26623 | 0.1302 | +0.0041 [+0.0037, +0.0046] | +0.0012 [+0.0010, +0.0014] | -0.0025 [-0.0031, -0.0018] |
| gen_joint_em0 | 10 dB | 0.9564 | 0.9015 | 0.09633 | 0.1299 | +0.0007 [+0.0005, +0.0009] | +0.0002 [+0.0001, +0.0003] | +0.0064 [+0.0058, +0.0070] |
| gen_joint_em0 | 20 dB | 0.9915 | 0.9761 | 0.03418 | 0.1281 | +0.0001 [+0.0001, +0.0002] | +0.0000 [+0.0000, +0.0001] | +0.0096 [+0.0091, +0.0101] |
| gen_joint_em0_rho | -5 dB | 0.7823 | 0.6696 | 0.38013 | 0.1701 | +0.0139 [+0.0129, +0.0149] | +0.0040 [+0.0037, +0.0044] | -0.0194 [-0.0206, -0.0182] |
| gen_joint_em0_rho | 0 dB | 0.8544 | 0.7505 | 0.26372 | 0.0944 | +0.0031 [+0.0027, +0.0035] | +0.0002 [-0.0001, +0.0005] | -0.0050 [-0.0055, -0.0045] |
| gen_joint_em0_rho | 10 dB | 0.9558 | 0.8995 | 0.08950 | 0.0355 | +0.0001 [-0.0000, +0.0002] | -0.0004 [-0.0006, -0.0002] | -0.0004 [-0.0006, -0.0003] |
| gen_joint_em0_rho | 20 dB | 0.9914 | 0.9756 | 0.02452 | 0.0102 | -0.0000 [-0.0000, +0.0000] | -0.0001 [-0.0002, -0.0001] | -0.0000 [-0.0001, -0.0000] |
| gen_joint_em_rho | -5 dB | 0.7821 | 0.6690 | 0.38038 | 0.1703 | +0.0137 [+0.0127, +0.0147] | +0.0038 [+0.0034, +0.0042] | -0.0192 [-0.0203, -0.0180] |
| gen_joint_em_rho | 0 dB | 0.8544 | 0.7504 | 0.26384 | 0.0946 | +0.0030 [+0.0026, +0.0034] | +0.0001 [-0.0002, +0.0004] | -0.0049 [-0.0054, -0.0044] |
| gen_joint_em_rho | 10 dB | 0.9557 | 0.8994 | 0.08957 | 0.0355 | +0.0000 [-0.0001, +0.0002] | -0.0005 [-0.0007, -0.0003] | -0.0004 [-0.0005, -0.0002] |
| gen_joint_em_rho | 20 dB | 0.9914 | 0.9755 | 0.02454 | 0.0102 | -0.0000 [-0.0000, -0.0000] | -0.0001 [-0.0002, -0.0001] | -0.0000 [-0.0000, +0.0000] |
| gen_joint_pilot | -5 dB | 0.7194 | 0.5783 | 0.55410 | 2.4425 | -0.0490 [-0.0509, -0.0471] | -0.0589 [-0.0606, -0.0570] | +0.1546 [+0.1490, +0.1607] |
| gen_joint_pilot | 0 dB | 0.8135 | 0.6815 | 0.42898 | 1.3290 | -0.0378 [-0.0393, -0.0364] | -0.0407 [-0.0422, -0.0393] | +0.1603 [+0.1538, +0.1679] |
| gen_joint_pilot | 10 dB | 0.9481 | 0.8792 | 0.22682 | 0.5092 | -0.0076 [-0.0081, -0.0071] | -0.0081 [-0.0086, -0.0076] | +0.1369 [+0.1287, +0.1447] |
| gen_joint_pilot | 20 dB | 0.9909 | 0.9736 | 0.05775 | 0.1804 | -0.0005 [-0.0006, -0.0004] | -0.0006 [-0.0007, -0.0005] | +0.0332 [+0.0295, +0.0376] |
| gen_joint_true | -5 dB | 0.7843 | 0.6759 | 0.37810 | - | +0.0159 [+0.0148, +0.0169] | +0.0060 [+0.0058, +0.0062] | -0.0214 [-0.0226, -0.0202] |
| gen_joint_true | 0 dB | 0.8556 | 0.7553 | 0.26153 | - | +0.0043 [+0.0038, +0.0048] | +0.0014 [+0.0013, +0.0015] | -0.0072 [-0.0078, -0.0066] |
| gen_joint_true | 10 dB | 0.9562 | 0.9012 | 0.08334 | - | +0.0005 [+0.0003, +0.0007] | -0.0000 [-0.0000, -0.0000] | -0.0066 [-0.0070, -0.0062] |
| gen_joint_true | 20 dB | 0.9915 | 0.9760 | 0.01658 | - | +0.0001 [+0.0000, +0.0002] | -0.0000 [-0.0000, -0.0000] | -0.0080 [-0.0083, -0.0076] |
| gen_joint_true0 | -5 dB | 0.7844 | 0.6761 | 0.37794 | - | +0.0160 [+0.0149, +0.0170] | +0.0061 [+0.0059, +0.0063] | -0.0216 [-0.0228, -0.0203] |
| gen_joint_true0 | 0 dB | 0.8557 | 0.7554 | 0.26140 | - | +0.0044 [+0.0039, +0.0049] | +0.0015 [+0.0014, +0.0016] | -0.0073 [-0.0079, -0.0067] |
| gen_joint_true0 | 10 dB | 0.9562 | 0.9013 | 0.08327 | - | +0.0005 [+0.0003, +0.0007] | +0.0000 [+0.0000, +0.0000] | -0.0066 [-0.0070, -0.0062] |
| gen_joint_true0 | 20 dB | 0.9915 | 0.9760 | 0.01657 | - | +0.0001 [+0.0000, +0.0002] | +0.0000 [+0.0000, +0.0000] | -0.0080 [-0.0083, -0.0077] |
| gen_joint_true0_rho | -5 dB | 0.7842 | 0.6749 | 0.37870 | - | +0.0158 [+0.0148, +0.0168] | +0.0060 [+0.0057, +0.0062] | -0.0208 [-0.0220, -0.0196] |
| gen_joint_true0_rho | 0 dB | 0.8553 | 0.7532 | 0.26355 | - | +0.0040 [+0.0036, +0.0044] | +0.0011 [+0.0009, +0.0013] | -0.0052 [-0.0057, -0.0046] |
| gen_joint_true0_rho | 10 dB | 0.9559 | 0.9000 | 0.08973 | - | +0.0002 [+0.0001, +0.0003] | -0.0003 [-0.0005, -0.0001] | -0.0002 [-0.0003, -0.0000] |
| gen_joint_true0_rho | 20 dB | 0.9914 | 0.9756 | 0.02458 | - | +0.0000 [-0.0000, +0.0000] | -0.0001 [-0.0002, -0.0000] | +0.0000 [-0.0000, +0.0001] |
| gen_joint_true_rho | -5 dB | 0.7841 | 0.6748 | 0.37886 | - | +0.0157 [+0.0147, +0.0167] | +0.0058 [+0.0056, +0.0061] | -0.0207 [-0.0218, -0.0194] |
| gen_joint_true_rho | 0 dB | 0.8553 | 0.7532 | 0.26367 | - | +0.0039 [+0.0035, +0.0044] | +0.0010 [+0.0008, +0.0012] | -0.0050 [-0.0055, -0.0045] |
| gen_joint_true_rho | 10 dB | 0.9559 | 0.8999 | 0.08980 | - | +0.0002 [+0.0001, +0.0003] | -0.0003 [-0.0005, -0.0002] | -0.0001 [-0.0003, +0.0000] |
| gen_joint_true_rho | 20 dB | 0.9914 | 0.9756 | 0.02460 | - | -0.0000 [-0.0000, +0.0000] | -0.0001 [-0.0002, -0.0001] | +0.0000 [+0.0000, +0.0001] |
| load_joint | -5 dB | 0.7783 | 0.6687 | 0.38717 | - | +0.0099 [+0.0089, +0.0109] | +0.0000 [+0.0000, +0.0000] | -0.0124 [-0.0136, -0.0112] |
| load_joint | 0 dB | 0.8542 | 0.7535 | 0.26385 | - | +0.0029 [+0.0024, +0.0034] | +0.0000 [+0.0000, +0.0000] | -0.0049 [-0.0055, -0.0043] |
| load_joint | 10 dB | 0.9562 | 0.9013 | 0.08332 | - | +0.0005 [+0.0003, +0.0007] | +0.0000 [+0.0000, +0.0000] | -0.0066 [-0.0070, -0.0062] |
| load_joint | 20 dB | 0.9915 | 0.9760 | 0.01657 | - | +0.0001 [+0.0000, +0.0002] | +0.0000 [+0.0000, +0.0000] | -0.0080 [-0.0083, -0.0077] |
| load_joint_blind | -5 dB | 0.7684 | 0.6417 | 0.39954 | 0.4146 | +0.0000 [+0.0000, +0.0000] | -0.0099 [-0.0109, -0.0089] | +0.0000 [+0.0000, +0.0000] |
| load_joint_blind | 0 dB | 0.8513 | 0.7428 | 0.26872 | 0.1595 | +0.0000 [+0.0000, +0.0000] | -0.0029 [-0.0034, -0.0024] | +0.0000 [+0.0000, +0.0000] |
| load_joint_blind | 10 dB | 0.9557 | 0.8993 | 0.08992 | 0.0355 | +0.0000 [+0.0000, +0.0000] | -0.0005 [-0.0007, -0.0003] | +0.0000 [+0.0000, +0.0000] |
| load_joint_blind | 20 dB | 0.9914 | 0.9756 | 0.02455 | 0.0082 | +0.0000 [+0.0000, +0.0000] | -0.0001 [-0.0002, -0.0000] | +0.0000 [+0.0000, +0.0000] |
| load_joint_em | -5 dB | 0.7769 | 0.6637 | 0.39109 | 0.1838 | +0.0085 [+0.0075, +0.0095] | -0.0014 [-0.0017, -0.0011] | -0.0085 [-0.0096, -0.0073] |
| load_joint_em | 0 dB | 0.8540 | 0.7518 | 0.26861 | 0.1291 | +0.0027 [+0.0023, +0.0032] | -0.0002 [-0.0004, -0.0000] | -0.0001 [-0.0007, +0.0006] |
| load_joint_em | 10 dB | 0.9564 | 0.9014 | 0.09642 | 0.1291 | +0.0007 [+0.0005, +0.0009] | +0.0002 [+0.0001, +0.0002] | +0.0065 [+0.0059, +0.0071] |
| load_joint_em | 20 dB | 0.9915 | 0.9761 | 0.03420 | 0.1280 | +0.0001 [+0.0001, +0.0002] | +0.0000 [+0.0000, +0.0001] | +0.0096 [+0.0091, +0.0101] |
| load_joint_em_rho | -5 dB | 0.7762 | 0.6623 | 0.38902 | 0.1652 | +0.0078 [+0.0068, +0.0088] | -0.0020 [-0.0024, -0.0017] | -0.0105 [-0.0116, -0.0094] |
| load_joint_em_rho | 0 dB | 0.8531 | 0.7489 | 0.26590 | 0.0922 | +0.0017 [+0.0013, +0.0021] | -0.0012 [-0.0015, -0.0009] | -0.0028 [-0.0033, -0.0023] |
| load_joint_em_rho | 10 dB | 0.9558 | 0.8995 | 0.08953 | 0.0354 | +0.0001 [-0.0000, +0.0002] | -0.0004 [-0.0006, -0.0003] | -0.0004 [-0.0005, -0.0003] |
| load_joint_em_rho | 20 dB | 0.9914 | 0.9756 | 0.02452 | 0.0102 | -0.0000 [-0.0000, +0.0000] | -0.0001 [-0.0002, -0.0001] | -0.0000 [-0.0001, -0.0000] |
| load_joint_rho | -5 dB | 0.7782 | 0.6680 | 0.38774 | - | +0.0098 [+0.0088, +0.0108] | -0.0001 [-0.0002, +0.0001] | -0.0118 [-0.0129, -0.0106] |
| load_joint_rho | 0 dB | 0.8539 | 0.7515 | 0.26584 | - | +0.0026 [+0.0021, +0.0030] | -0.0003 [-0.0006, -0.0001] | -0.0029 [-0.0034, -0.0024] |
| load_joint_rho | 10 dB | 0.9559 | 0.8999 | 0.08976 | - | +0.0002 [+0.0001, +0.0003] | -0.0003 [-0.0005, -0.0001] | -0.0002 [-0.0003, -0.0000] |
| load_joint_rho | 20 dB | 0.9914 | 0.9756 | 0.02458 | - | +0.0000 [-0.0000, +0.0000] | -0.0001 [-0.0002, -0.0000] | +0.0000 [-0.0000, +0.0001] |
| load_pu | -5 dB | 0.7320 | 0.5534 | 0.44165 | - | -0.0364 [-0.0383, -0.0346] | -0.0462 [-0.0481, -0.0444] | +0.0421 [+0.0405, +0.0439] |
| load_pu | 0 dB | 0.8278 | 0.6804 | 0.29846 | - | -0.0235 [-0.0247, -0.0221] | -0.0264 [-0.0276, -0.0251] | +0.0297 [+0.0283, +0.0310] |
| load_pu | 10 dB | 0.9504 | 0.8814 | 0.09182 | - | -0.0053 [-0.0060, -0.0047] | -0.0058 [-0.0064, -0.0052] | +0.0019 [+0.0011, +0.0027] |
| load_pu | 20 dB | 0.9909 | 0.9736 | 0.01756 | - | -0.0005 [-0.0007, -0.0004] | -0.0006 [-0.0008, -0.0005] | -0.0070 [-0.0074, -0.0066] |
| load_pu_blind | -5 dB | 0.7306 | 0.5503 | 0.45925 | 0.4146 | -0.0378 [-0.0397, -0.0360] | -0.0476 [-0.0496, -0.0458] | +0.0597 [+0.0578, +0.0617] |
| load_pu_blind | 0 dB | 0.8273 | 0.6781 | 0.30479 | 0.1595 | -0.0240 [-0.0253, -0.0227] | -0.0269 [-0.0282, -0.0256] | +0.0361 [+0.0346, +0.0375] |
| load_pu_blind | 10 dB | 0.9501 | 0.8803 | 0.09854 | 0.0355 | -0.0056 [-0.0062, -0.0050] | -0.0061 [-0.0067, -0.0055] | +0.0086 [+0.0079, +0.0094] |
| load_pu_blind | 20 dB | 0.9908 | 0.9732 | 0.02561 | 0.0082 | -0.0006 [-0.0008, -0.0005] | -0.0007 [-0.0009, -0.0005] | +0.0011 [+0.0009, +0.0013] |
| load_pu_rho | -5 dB | 0.7321 | 0.5531 | 0.44186 | - | -0.0363 [-0.0382, -0.0345] | -0.0462 [-0.0481, -0.0444] | +0.0423 [+0.0407, +0.0441] |
| load_pu_rho | 0 dB | 0.8277 | 0.6791 | 0.29999 | - | -0.0236 [-0.0249, -0.0224] | -0.0265 [-0.0278, -0.0252] | +0.0313 [+0.0299, +0.0326] |
| load_pu_rho | 10 dB | 0.9501 | 0.8805 | 0.09826 | - | -0.0055 [-0.0062, -0.0050] | -0.0061 [-0.0067, -0.0054] | +0.0083 [+0.0077, +0.0090] |
| load_pu_rho | 20 dB | 0.9908 | 0.9733 | 0.02565 | - | -0.0006 [-0.0008, -0.0005] | -0.0007 [-0.0009, -0.0005] | +0.0011 [+0.0009, +0.0013] |

