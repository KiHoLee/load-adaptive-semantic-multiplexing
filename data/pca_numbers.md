# Numbers of the closed-form (PCA) version

Participation ratio of the PCA spectrum of the training embeddings: 61.9 of 768

## 1. Proposal (PCA nested code + dynamic masks), per-user CosSim

| load | 0 dB | 5 dB | 10 dB | 15 dB | 20 dB | 25 dB | 30 dB |
|---|---|---|---|---|---|---|---|
| U=1 | 0.895 | 0.950 | 0.977 | 0.992 | 0.997 | 0.999 | 1.000 |
| U=2 | 0.846 | 0.917 | 0.962 | 0.984 | 0.993 | 0.997 | 0.999 |
| U=3 | 0.815 | 0.894 | 0.951 | 0.978 | 0.991 | 0.996 | 0.999 |
| U=4 | 0.792 | 0.877 | 0.939 | 0.972 | 0.989 | 0.995 | 0.998 |
| U=5 | 0.775 | 0.864 | 0.928 | 0.965 | 0.981 | 0.989 | 0.992 |
| U=6 | 0.757 | 0.853 | 0.919 | 0.957 | 0.975 | 0.982 | 0.985 |
| U=8 | 0.728 | 0.830 | 0.902 | 0.942 | 0.961 | 0.969 | 0.973 |

## 2. Proposal minus learned references at load U (learned: three-seed means)

| reference, U | 0 dB | 5 dB | 10 dB | 15 dB | 20 dB | 25 dB | 30 dB |
|---|---|---|---|---|---|---|---|
| learned Static ODS-U, U=1 | +0.025 | +0.008 | +0.006 | +0.006 | +0.006 | +0.006 | +0.006 |
| learned Static ODS-U, U=2 | +0.030 | +0.007 | +0.007 | +0.007 | +0.007 | +0.007 | +0.008 |
| learned Static ODS-U, U=3 | +0.036 | +0.007 | +0.008 | +0.007 | +0.009 | +0.009 | +0.010 |
| learned Static ODS-U, U=4 | +0.041 | +0.008 | +0.007 | +0.008 | +0.011 | +0.012 | +0.013 |
| learned Static ODS-U, U=5 | +0.047 | +0.012 | +0.006 | +0.007 | +0.008 | +0.010 | +0.010 |
| learned Static ODS-U, U=6 | +0.045 | +0.014 | +0.007 | +0.007 | +0.008 | +0.009 | +0.009 |
| learned nested code, U=1 | +0.018 | +0.010 | +0.007 | +0.010 | +0.008 | +0.008 | +0.008 |
| learned nested code, U=2 | +0.015 | +0.007 | +0.008 | +0.009 | +0.006 | +0.008 | +0.007 |
| learned nested code, U=3 | +0.019 | +0.009 | +0.009 | +0.008 | +0.007 | +0.009 | +0.009 |
| learned nested code, U=4 | +0.019 | +0.009 | +0.007 | +0.007 | +0.006 | +0.008 | +0.008 |
| learned nested code, U=5 | +0.022 | +0.011 | +0.007 | +0.008 | +0.006 | +0.008 | +0.007 |
| learned nested code, U=6 | +0.022 | +0.012 | +0.008 | +0.008 | +0.006 | +0.009 | +0.009 |
| learned nested code from PCA (seed 1), U=1 | +0.020 | +0.011 | +0.008 | +0.011 | +0.009 | +0.010 | +0.010 |
| learned nested code from PCA (seed 1), U=2 | +0.012 | +0.006 | +0.007 | +0.007 | +0.003 | +0.004 | +0.003 |
| learned nested code from PCA (seed 1), U=3 | +0.016 | +0.006 | +0.006 | +0.005 | +0.004 | +0.007 | +0.007 |
| learned nested code from PCA (seed 1), U=4 | +0.016 | +0.007 | +0.007 | +0.006 | +0.004 | +0.005 | +0.005 |
| learned nested code from PCA (seed 1), U=5 | +0.019 | +0.010 | +0.007 | +0.007 | +0.003 | +0.003 | +0.003 |
| learned nested code from PCA (seed 1), U=6 | +0.019 | +0.011 | +0.007 | +0.007 | +0.003 | +0.004 | +0.004 |
| Overlapping masks, U=1 | +0.017 | +0.017 | +0.019 | +0.021 | +0.022 | +0.022 | +0.023 |
| Overlapping masks, U=2 | +0.024 | +0.013 | +0.017 | +0.020 | +0.022 | +0.022 | +0.024 |
| Overlapping masks, U=3 | +0.032 | +0.010 | +0.016 | +0.018 | +0.021 | +0.022 | +0.024 |
| Overlapping masks, U=4 | +0.038 | +0.012 | +0.014 | +0.017 | +0.020 | +0.022 | +0.024 |
| Overlapping masks, U=5 | +0.049 | +0.016 | +0.011 | +0.014 | +0.015 | +0.017 | +0.018 |
| Overlapping masks, U=6 | +0.054 | +0.021 | +0.011 | +0.011 | +0.011 | +0.011 | +0.012 |

- proposal minus learned Static ODS-U over U=1..6 and all SNRs: +0.006 to +0.047
- proposal minus learned nested code over U=1..6 and all SNRs: +0.006 to +0.022
- proposal minus learned nested code from PCA over U=1..6: +0.003 to +0.020
- proposal minus Overlapping masks over U=1..6: +0.010 to +0.054

## 3. Static partitions with the same closed-form code (Static PCA-k, n = E = d_s/k, any U <= k)

| k | 0 dB | 5 dB | 10 dB | 15 dB | 20 dB | 25 dB | 30 dB |
|---|---|---|---|---|---|---|---|
| Static PCA-1 | 0.895 | 0.950 | 0.977 | 0.992 | 0.997 | 0.999 | 1.000 |
| Static PCA-2 | 0.849 | 0.920 | 0.962 | 0.985 | 0.994 | 0.997 | 0.999 |
| Static PCA-3 | 0.818 | 0.898 | 0.949 | 0.979 | 0.991 | 0.996 | 0.999 |
| Static PCA-4 | 0.794 | 0.880 | 0.939 | 0.974 | 0.989 | 0.995 | 0.998 |
| Static PCA-5 | 0.774 | 0.866 | 0.929 | 0.966 | 0.982 | 0.989 | 0.992 |
| Static PCA-6 | 0.758 | 0.853 | 0.919 | 0.958 | 0.975 | 0.982 | 0.985 |
| Static PCA-8 | 0.729 | 0.830 | 0.902 | 0.944 | 0.961 | 0.969 | 0.973 |

### proposal minus Static PCA-U at its design load (identical code; differences are sampling and the slot granularity at U=5)
| U | 0 dB | 5 dB | 10 dB | 15 dB | 20 dB | 25 dB | 30 dB |
|---|---|---|---|---|---|---|---|
| U=1 | +0.0000 | +0.0000 | +0.0000 | +0.0000 | +0.0000 | +0.0000 | +0.0000 |
| U=2 | -0.0032 | -0.0024 | +0.0005 | -0.0006 | -0.0004 | -0.0001 | +0.0000 |
| U=3 | -0.0025 | -0.0034 | +0.0011 | -0.0017 | -0.0005 | -0.0002 | +0.0001 |
| U=4 | -0.0021 | -0.0030 | +0.0004 | -0.0023 | -0.0005 | -0.0000 | +0.0002 |
| U=5 | +0.0004 | -0.0016 | -0.0016 | -0.0016 | -0.0010 | +0.0002 | -0.0002 |
| U=6 | -0.0003 | -0.0003 | -0.0002 | -0.0012 | -0.0002 | +0.0003 | +0.0000 |
| U=8 | -0.0008 | -0.0003 | +0.0006 | -0.0011 | -0.0005 | +0.0001 | -0.0001 |

## 4. Light load: proposal minus fixed partitions at U < k

| reference, U | 0 dB | 5 dB | 10 dB | 15 dB | 20 dB | 25 dB | 30 dB |
|---|---|---|---|---|---|---|---|
| Static PCA-6, U=1 | +0.137 | +0.096 | +0.058 | +0.033 | +0.022 | +0.017 | +0.014 |
| Static PCA-6, U=2 | +0.088 | +0.064 | +0.043 | +0.026 | +0.019 | +0.015 | +0.014 |
| Static PCA-6, U=3 | +0.058 | +0.041 | +0.031 | +0.019 | +0.016 | +0.014 | +0.013 |
| Static PCA-6, U=4 | +0.034 | +0.024 | +0.020 | +0.014 | +0.014 | +0.013 | +0.013 |
| Static PCA-6, U=5 | +0.017 | +0.011 | +0.008 | +0.006 | +0.006 | +0.007 | +0.006 |
| Static PCA-4, U=1 | +0.101 | +0.070 | +0.038 | +0.017 | +0.007 | +0.003 | +0.001 |
| Static PCA-4, U=2 | +0.052 | +0.038 | +0.024 | +0.010 | +0.004 | +0.002 | +0.001 |
| Static PCA-4, U=3 | +0.022 | +0.014 | +0.012 | +0.003 | +0.002 | +0.001 | +0.000 |
| learned Static ODS-6, U=1 | +0.183 | +0.111 | +0.065 | +0.042 | +0.030 | +0.025 | +0.024 |
| learned Static ODS-6, U=2 | +0.134 | +0.079 | +0.050 | +0.035 | +0.027 | +0.024 | +0.023 |
| learned Static ODS-6, U=3 | +0.104 | +0.056 | +0.039 | +0.028 | +0.024 | +0.023 | +0.023 |
| learned Static ODS-6, U=4 | +0.080 | +0.038 | +0.027 | +0.022 | +0.022 | +0.022 | +0.023 |
| learned Static ODS-6, U=5 | +0.063 | +0.026 | +0.016 | +0.015 | +0.014 | +0.016 | +0.016 |
| learned Static ODS-4, U=1 | +0.145 | +0.081 | +0.045 | +0.028 | +0.019 | +0.015 | +0.014 |
| learned Static ODS-4, U=2 | +0.095 | +0.049 | +0.030 | +0.021 | +0.016 | +0.014 | +0.014 |
| learned Static ODS-4, U=3 | +0.065 | +0.026 | +0.019 | +0.014 | +0.013 | +0.013 | +0.014 |

## 5. Matched energy: proposal minus the fixed partition whose user spends d_s/U on its own part

| reference, U | 0 dB | 5 dB | 10 dB | 15 dB | 20 dB | 25 dB | 30 dB |
|---|---|---|---|---|---|---|---|
| Static PCA-6 energy d_s/U, U=1 | +0.001 | +0.004 | +0.009 | +0.011 | +0.012 | +0.012 | +0.013 |
| Static PCA-6 energy d_s/U, U=2 | -0.004 | -0.001 | +0.007 | +0.009 | +0.011 | +0.012 | +0.013 |
| Static PCA-6 energy d_s/U, U=3 | -0.003 | -0.004 | +0.006 | +0.007 | +0.011 | +0.012 | +0.013 |
| Static PCA-6 energy d_s/U, U=4 | -0.002 | -0.004 | +0.004 | +0.006 | +0.010 | +0.012 | +0.013 |
| Static PCA-6 energy d_s/U, U=5 | +0.001 | -0.002 | +0.001 | +0.002 | +0.004 | +0.006 | +0.006 |
| Static PCA-4 energy d_s/U, U=1 | +0.000 | +0.000 | +0.000 | +0.000 | +0.000 | +0.000 | +0.000 |
| Static PCA-4 energy d_s/U, U=2 | -0.003 | -0.002 | +0.000 | -0.001 | -0.000 | -0.000 | +0.000 |
| Static PCA-4 energy d_s/U, U=3 | -0.003 | -0.003 | +0.001 | -0.002 | -0.001 | -0.000 | +0.000 |
| learned Static ODS-6 energy d_s/U, U=1 | +0.009 | +0.012 | +0.016 | +0.020 | +0.021 | +0.022 | +0.022 |
| learned Static ODS-6 energy d_s/U, U=2 | +0.009 | +0.006 | +0.013 | +0.019 | +0.020 | +0.021 | +0.022 |
| learned Static ODS-6 energy d_s/U, U=3 | +0.019 | +0.006 | +0.012 | +0.015 | +0.019 | +0.021 | +0.022 |
| learned Static ODS-6 energy d_s/U, U=4 | +0.027 | +0.007 | +0.009 | +0.014 | +0.019 | +0.021 | +0.022 |
| learned Static ODS-6 energy d_s/U, U=5 | +0.039 | +0.011 | +0.007 | +0.010 | +0.013 | +0.015 | +0.015 |

## 6. Variants of the proposal (minus the proposal)

| variant, U | 0 dB | 5 dB | 10 dB | 15 dB | 20 dB | 25 dB | 30 dB |
|---|---|---|---|---|---|---|---|
| power proportional to variance (no SNR), U=1 | -0.0039 | -0.0071 | -0.0055 | -0.0034 | -0.0018 | -0.0008 | -0.0003 |
| power proportional to variance (no SNR), U=4 | -0.0016 | -0.0024 | -0.0060 | -0.0057 | -0.0038 | -0.0020 | -0.0010 |
| power proportional to variance (no SNR), U=6 | -0.0011 | -0.0020 | -0.0056 | -0.0056 | -0.0035 | -0.0017 | -0.0008 |
| no Wiener factor, U=1 | -0.0288 | -0.0077 | -0.0028 | -0.0006 | -0.0003 | -0.0001 | -0.0000 |
| no Wiener factor, U=4 | -0.3081 | -0.0465 | -0.0102 | -0.0035 | -0.0011 | -0.0004 | -0.0001 |
| no Wiener factor, U=6 | -0.1618 | -0.0338 | -0.0098 | -0.0031 | -0.0010 | -0.0004 | -0.0001 |
| known gain, U=1 | +0.0000 | +0.0000 | +0.0000 | -0.0000 | +0.0000 | -0.0000 | +0.0000 |
| known gain, U=4 | +0.0004 | +0.0002 | +0.0000 | -0.0000 | +0.0000 | -0.0000 | +0.0000 |
| known gain, U=6 | +0.0003 | +0.0001 | +0.0000 | +0.0000 | +0.0000 | +0.0000 | -0.0000 |

### expansion factor K (d_s = K d_b)
| K, U | 0 dB | 5 dB | 10 dB | 15 dB | 20 dB | 25 dB | 30 dB |
|---|---|---|---|---|---|---|---|
| K=1, U=1 | 0.794 | 0.880 | 0.939 | 0.974 | 0.989 | 0.995 | 0.998 |
| K=1, U=6 | 0.615 | 0.728 | 0.808 | 0.855 | 0.876 | 0.886 | 0.890 |
| K=2, U=1 | 0.849 | 0.920 | 0.962 | 0.985 | 0.994 | 0.997 | 0.999 |
| K=2, U=6 | 0.688 | 0.797 | 0.873 | 0.916 | 0.937 | 0.946 | 0.949 |
| K=8, U=1 | 0.930 | 0.970 | 0.986 | 0.995 | 0.998 | 0.999 | 1.000 |
| K=8, U=6 | 0.816 | 0.894 | 0.949 | 0.978 | 0.991 | 0.996 | 0.999 |
| K=4, U=1 | 0.895 | 0.950 | 0.977 | 0.992 | 0.997 | 0.999 | 1.000 |
| K=4, U=6 | 0.757 | 0.853 | 0.919 | 0.957 | 0.975 | 0.982 | 0.985 |

### allocation width at U=1, 4, 6: active coefficients / participation ratio of the powers
- U=1: 0 dB 766/274, 5 dB 767/328, 10 dB 767/347, 15 dB 767/354, 20 dB 767/356, 25 dB 767/357, 30 dB 767/357
- U=4: 0 dB 456/158, 5 dB 765/257, 10 dB 767/321, 15 dB 767/345, 20 dB 767/353, 25 dB 767/356, 30 dB 767/356
- U=6: 0 dB 348/129, 5 dB 512/212, 10 dB 512/258, 15 dB 512/274, 20 dB 512/280, 25 dB 512/281, 30 dB 512/282

## 7. ToDMA re-dimensioned (best T per load and SNR, single runs of 500 frames) minus proposal

| U | 0 dB | 5 dB | 10 dB | 15 dB | 20 dB | 25 dB | 30 dB |
|---|---|---|---|---|---|---|---|
| U=1 | -0.068 | -0.033 | -0.013 | +0.003 | -0.000 | +0.001 | +0.000 |
| U=2 | -0.210 | -0.052 | -0.029 | -0.003 | -0.001 | -0.001 | -0.000 |
| U=4 | -0.299 | -0.105 | -0.046 | -0.009 | -0.006 | -0.006 | -0.004 |
| U=6 | -0.406 | -0.174 | -0.073 | -0.019 | +0.001 | -0.000 | -0.000 |

## 8. Robustness and generality at U=4 (proposal; learned Static ODS-4 from ortho_U4_s1, seed 1)

| condition | 0 dB | 5 dB | 10 dB | 15 dB | 20 dB | 25 dB | 30 dB |
|---|---|---|---|---|---|---|---|
| proposal default | 0.792 | 0.877 | 0.939 | 0.972 | 0.989 | 0.995 | 0.998 |
| learned ODS-4 default | 0.749 | 0.868 | 0.932 | 0.964 | 0.978 | 0.984 | 0.986 |
| proposal common fading | 0.793 | 0.885 | 0.940 | 0.973 | 0.989 | 0.996 | 0.998 |
| learned ODS-4 common fading | 0.748 | 0.874 | 0.935 | 0.965 | 0.978 | 0.984 | 0.986 |
| proposal phase 30 deg | 0.753 | 0.849 | 0.917 | 0.958 | 0.980 | 0.988 | 0.991 |
| learned ODS-4 phase 30 deg | 0.703 | 0.834 | 0.909 | 0.951 | 0.969 | 0.977 | 0.979 |
| proposal near-far 6 dB weakest | 0.633 | 0.750 | 0.855 | 0.931 | 0.968 | 0.986 | 0.995 |
| learned ODS-4 near-far 6 dB weakest | 0.555 | 0.721 | 0.851 | 0.923 | 0.959 | 0.976 | 0.983 |
| proposal near-far 6 dB mean | 0.774 | 0.860 | 0.927 | 0.965 | 0.985 | 0.993 | 0.998 |
| learned ODS-4 near-far 6 dB mean | 0.729 | 0.848 | 0.919 | 0.957 | 0.975 | 0.983 | 0.985 |
| proposal SST-2 | 0.784 | 0.872 | 0.937 | 0.970 | 0.988 | 0.995 | 0.998 |
| learned ODS-4 SST-2 | 0.739 | 0.862 | 0.929 | 0.963 | 0.978 | 0.984 | 0.986 |
| proposal DistilBERT | 0.819 | 0.896 | 0.949 | 0.977 | 0.991 | 0.996 | 0.999 |
| learned ODS-4 DistilBERT | 0.785 | 0.890 | 0.943 | 0.969 | 0.980 | 0.985 | 0.986 |
| proposal U=8 | 0.728 | 0.830 | 0.902 | 0.942 | 0.961 | 0.969 | 0.973 |
| learned ODS-8 (s1) | 0.685 | 0.818 | 0.895 | 0.937 | 0.955 | 0.962 | 0.965 |
| Static PCA-8 | 0.729 | 0.830 | 0.902 | 0.944 | 0.961 | 0.969 | 0.973 |

- topic classifier: clean 0.855; proposal U=4 at 0/10/20 dB 0.778/0.841/0.846
- sentiment classifier (SST-2): clean 0.860; proposal U=4 at 0/10/20 dB 0.786/0.840/0.854
- learned ODS-4 topic probe at 20 dB: 0.846
- learned ODS-4 topic probe at 10 dB: 0.828
- topic accuracy at 0/10/20 dB: proposal 0.778/0.841/0.846; learned ODS-4 (seed 1) 0.757/0.828/0.846; proposal minus learned +0.0207/+0.0130/-0.0006
- sentiment (SST-2) accuracy at 0/10/20 dB: proposal 0.786/0.840/0.854; learned ODS-4 (seed 1) 0.782/0.849/0.855; proposal minus learned +0.0036/-0.0086/-0.0019
- learned ODS-4 parameters (ortho_U4_s1): 1184256; proposal stored values / learned ODS-4 parameters: 0.499
- cost: 590592 stored coefficients (basis and variances); 0.0133 ms CPU per user and frame for the two transforms
- transmitted energy of a user in entries that carry at least 90 percent of their energy from that user (three-seed means, data/extra/mask_partition.json): Overlapping masks 0.993; random frozen masks, four users 0.197
- proposal minus learned over the listed conditions (default, common fading, phase 30 deg, near-far weakest, SST-2, DistilBERT, U=8) at 10 and 20 dB: +0.0042 to +0.0114

## 9. Aggregate fidelity S(U) at 10 dB

| U | proposal | learned ODS-U | Static PCA-4 (min(U,4) users) | Static PCA-6 |
|---|---|---|---|---|
| 1 | 0.98 | 0.97 | 0.94 | 0.92 |
| 2 | 1.92 | 1.91 | 1.88 | 1.84 |
| 3 | 2.85 | 2.83 | 2.82 | 2.76 |
| 4 | 3.76 | 3.73 | 3.75 | 3.68 |
| 5 | 4.64 | 4.61 | 3.75 | 4.60 |
| 6 | 5.51 | 5.47 | 3.75 | 5.52 |

## 10. Table I of the letter: per-user CosSim at U=1 and U=6, 0 / 10 / 20 dB (absolute values)

| row | U=1 0 dB | U=1 10 dB | U=1 20 dB | U=6 0 dB | U=6 10 dB | U=6 20 dB |
|---|---|---|---|---|---|---|
| Proposed | 0.895 | 0.977 | 0.997 | 0.757 | 0.919 | 0.975 |
| Static ODS-6 energy d_s/U (U=6: Static ODS-6) | 0.894 | 0.968 | 0.984 | 0.758 | 0.919 | 0.975 |
| power proportional to variance | 0.891 | 0.971 | 0.995 | 0.756 | 0.913 | 0.971 |
| no Wiener factor | 0.866 | 0.974 | 0.996 | 0.595 | 0.909 | 0.974 |
| K=1 | 0.794 | 0.939 | 0.989 | 0.615 | 0.808 | 0.876 |
| K=8 | 0.930 | 0.986 | 0.998 | 0.816 | 0.949 | 0.991 |
| Learned nested code (three seeds) | 0.877 | 0.970 | 0.989 | 0.735 | 0.911 | 0.968 |
| Learned nested code from PCA (seed 1) | 0.875 | 0.969 | 0.988 | 0.739 | 0.912 | 0.972 |
| Learned ODS-U (three seeds) | 0.870 | 0.971 | 0.991 | 0.712 | 0.912 | 0.967 |
| Overlapping masks (three seeds) | 0.878 | 0.958 | 0.974 | 0.703 | 0.908 | 0.964 |
| ToDMA best T (single run) | 0.827 | 0.964 | 0.996 | 0.351 | 0.846 | 0.975 |

- largest sample SD over three seeds (Learned ODS-k at its load, learned nested code, Overlapping masks; U=1..6, all SNRs): 0.0046
- K=1 minus proposal at U=6, 10 dB: -0.111; K=8 minus proposal at U=6, 10 dB: +0.030

## 11. Monte Carlo check of (7), (10), (12) (Code/pca_mc.py, rank-0 user, known gain, rho_u = 1, 20,000 frames, seed 999)

| source, U | 0 dB measured / closed form | 10 dB | 20 dB |
|---|---|---|---|
| train, U=1 | 0.18685 / 0.18690 (-0.03%) | 0.04071 / 0.04069 (+0.05%) | 0.00670 / 0.00672 (-0.34%) |
| train, U=4 | 0.34550 / 0.34608 (-0.17%) | 0.10890 / 0.10866 (+0.22%) | 0.02034 / 0.02032 (+0.13%) |
| train, U=6 | 0.37746 / 0.37773 (-0.07%) | 0.12059 / 0.12034 (+0.21%) | 0.02386 / 0.02382 (+0.20%) |
| test, U=1 | 0.18678 / 0.18406 (+1.48%) | 0.04205 / 0.04169 (+0.86%) | 0.00686 / 0.00683 (+0.44%) |
| test, U=4 | 0.35248 / 0.34587 (+1.91%) | 0.11031 / 0.10901 (+1.19%) | 0.02115 / 0.02102 (+0.67%) |
| test, U=6 | 0.37834 / 0.37604 (+0.61%) | 0.12087 / 0.12053 (+0.29%) | 0.02304 / 0.02297 (+0.30%) |

- largest |measured / closed form - 1|: training sentences 0.34%, test sentences 1.91%

## 12. Channel-aware masks under time-correlated fading (Code/pca_csi.py)

AR(1) complex Gaussian gain per user over the 24 slots (rho between adjacent slots), independent across users, near-far offset uniform in [-6, 6] dB per user and frame where marked nf6; the scheduler knows the slot gains (perfect, or delayed with correlation rho_b); blind per-slot gain estimate; 2,000 frames per user, draws shared by all schemes of a scenario (seed 123 + SNR). Files without the suffix _x hold fixed, csi, csiq, ods, odsfix; files with _x hold odsm (and csic), files with _e hold csiqe.
Schemes: csiq = Proposed, channel-aware masks (greedy slot counts by the total MSE, energy 128 per granted slot); csi = channel-aware placement with the slot counts of the load masks (energy d_s/U); fixed = Proposed with the load masks (4), no CSI at the transmitter; ods = Static ODS-k with known gains (the same slot gains as the proposal, not its grant: N/k slots per user by the Hungarian assignment on log(1 + gamma g^2), energy d_s/k); odsm = the same with energy d_s/U; odsfix = Static ODS-k with fixed contiguous parts, no CSI; csiqe = csiq with every count capped at that of the load masks (equal counts at U=6, greedy placement only); weakest = mean over frames of the smallest per-user CosSim of the frame.


### rho0.7_nf6, Static ODS-6 (0, 10, 20 dB; mean, then weakest in parentheses)

| U | fixed | csi | csiq | csiqe | ods | odsm | odsfix |
|---|---|---|---|---|---|---|---|
| 1 | 0.927 (0.927) / 0.988 (0.988) / 0.997 (0.997) | 0.970 (0.970) / 0.995 (0.995) / 0.998 (0.998) | 0.970 (0.970) / 0.995 (0.995) / 0.998 (0.998) | - | 0.901 (0.901) / 0.972 (0.972) / 0.984 (0.984) | 0.964 (0.964) / 0.983 (0.983) / 0.985 (0.985) | 0.748 (0.748) / 0.916 (0.916) / 0.971 (0.971) |
| 2 | 0.878 (0.835) / 0.974 (0.963) / 0.995 (0.993) | 0.951 (0.934) / 0.991 (0.988) / 0.998 (0.997) | 0.953 (0.942) / 0.992 (0.990) / 0.998 (0.997) | 0.952 (0.936) / 0.992 (0.988) / 0.998 (0.997) | 0.903 (0.875) / 0.972 (0.965) / 0.984 (0.982) | 0.948 (0.933) / 0.981 (0.977) / 0.985 (0.983) | 0.748 (0.655) / 0.910 (0.870) / 0.971 (0.960) |
| 3 | 0.842 (0.754) / 0.962 (0.935) / 0.993 (0.987) | 0.933 (0.895) / 0.988 (0.980) / 0.997 (0.996) | 0.936 (0.911) / 0.989 (0.984) / 0.997 (0.996) | 0.934 (0.899) / 0.988 (0.980) / 0.997 (0.996) | 0.900 (0.852) / 0.972 (0.960) / 0.984 (0.980) | 0.932 (0.898) / 0.978 (0.971) / 0.985 (0.981) | 0.746 (0.597) / 0.910 (0.837) / 0.971 (0.953) |
| 4 | 0.786 (0.618) / 0.932 (0.854) / 0.985 (0.965) | 0.919 (0.864) / 0.984 (0.970) / 0.997 (0.994) | 0.922 (0.885) / 0.984 (0.973) / 0.997 (0.994) | 0.921 (0.871) / 0.984 (0.971) / 0.997 (0.994) | 0.899 (0.838) / 0.972 (0.956) / 0.984 (0.979) | 0.919 (0.868) / 0.976 (0.965) / 0.985 (0.980) | 0.746 (0.550) / 0.910 (0.817) / 0.971 (0.945) |
| 5 | 0.766 (0.550) / 0.922 (0.823) / 0.978 (0.951) | 0.908 (0.837) / 0.977 (0.959) / 0.990 (0.984) | 0.910 (0.862) / 0.977 (0.961) / 0.990 (0.983) | 0.908 (0.845) / 0.977 (0.960) / 0.990 (0.983) | 0.899 (0.827) / 0.971 (0.953) / 0.984 (0.978) | 0.908 (0.841) / 0.973 (0.958) / 0.984 (0.979) | 0.745 (0.509) / 0.911 (0.800) / 0.971 (0.941) |
| 6 | 0.747 (0.492) / 0.911 (0.788) / 0.971 (0.934) | 0.896 (0.811) / 0.971 (0.949) / 0.984 (0.977) | 0.899 (0.839) / 0.970 (0.950) / 0.984 (0.976) | 0.897 (0.821) / 0.970 (0.949) / 0.984 (0.976) | 0.896 (0.811) / 0.971 (0.949) / 0.984 (0.977) | - | 0.742 (0.476) / 0.911 (0.782) / 0.971 (0.934) |

### rho0.7, Static ODS-6 (0, 10, 20 dB; mean, then weakest in parentheses)

| U | fixed | csi | csiq | csiqe | ods | odsm | odsfix |
|---|---|---|---|---|---|---|---|
| 1 | 0.939 (0.939) / 0.991 (0.991) / 0.998 (0.998) | 0.976 (0.976) / 0.996 (0.996) / 0.998 (0.998) | 0.976 (0.976) / 0.996 (0.996) / 0.998 (0.998) | - | 0.910 (0.910) / 0.975 (0.975) / 0.985 (0.985) | 0.969 (0.969) / 0.984 (0.984) / 0.986 (0.986) | 0.764 (0.764) / 0.924 (0.924) / 0.973 (0.973) |
| 2 | 0.887 (0.859) / 0.979 (0.972) / 0.996 (0.995) | 0.957 (0.949) / 0.993 (0.991) / 0.998 (0.997) | 0.957 (0.951) / 0.993 (0.992) / 0.998 (0.997) | - | 0.909 (0.895) / 0.975 (0.971) / 0.985 (0.982) | 0.954 (0.946) / 0.982 (0.979) / 0.986 (0.983) | 0.757 (0.679) / 0.919 (0.887) / 0.973 (0.964) |
| 3 | 0.856 (0.800) / 0.967 (0.951) / 0.994 (0.990) | 0.941 (0.924) / 0.990 (0.987) / 0.998 (0.996) | 0.941 (0.928) / 0.991 (0.988) / 0.998 (0.996) | - | 0.908 (0.885) / 0.975 (0.969) / 0.985 (0.981) | 0.940 (0.924) / 0.980 (0.976) / 0.985 (0.982) | 0.758 (0.631) / 0.918 (0.858) / 0.974 (0.959) |
| 4 | 0.798 (0.658) / 0.939 (0.879) / 0.987 (0.971) | 0.927 (0.902) / 0.987 (0.981) / 0.997 (0.995) | 0.928 (0.908) / 0.987 (0.981) / 0.997 (0.995) | - | 0.907 (0.877) / 0.975 (0.967) / 0.985 (0.979) | 0.927 (0.903) / 0.978 (0.972) / 0.985 (0.980) | 0.757 (0.588) / 0.919 (0.843) / 0.973 (0.952) |
| 5 | 0.775 (0.589) / 0.929 (0.850) / 0.981 (0.959) | 0.915 (0.882) / 0.980 (0.971) / 0.991 (0.984) | 0.916 (0.890) / 0.980 (0.971) / 0.991 (0.984) | - | 0.906 (0.871) / 0.974 (0.965) / 0.985 (0.979) | 0.916 (0.883) / 0.976 (0.968) / 0.985 (0.979) | 0.755 (0.552) / 0.919 (0.827) / 0.974 (0.950) |
| 6 | 0.757 (0.539) / 0.919 (0.819) / 0.973 (0.944) | 0.905 (0.864) / 0.974 (0.963) / 0.984 (0.978) | 0.905 (0.872) / 0.973 (0.962) / 0.984 (0.977) | - | 0.905 (0.864) / 0.974 (0.963) / 0.984 (0.978) | - | 0.753 (0.520) / 0.918 (0.812) / 0.973 (0.943) |

### rho0.7_nf6, Static ODS-4 (0, 10, 20 dB; mean, then weakest in parentheses)

| U | fixed | csi | csiq | csiqe | ods | odsm | odsfix |
|---|---|---|---|---|---|---|---|
| 1 | 0.927 (0.927) / 0.988 (0.988) / 0.997 (0.997) | 0.970 (0.970) / 0.995 (0.995) / 0.998 (0.998) | 0.970 (0.970) / 0.995 (0.995) / 0.998 (0.998) | - | 0.922 (0.922) / 0.986 (0.986) / 0.997 (0.997) | 0.970 (0.970) / 0.995 (0.995) / 0.998 (0.998) | 0.785 (0.785) / 0.937 (0.937) / 0.985 (0.985) |
| 2 | 0.878 (0.835) / 0.974 (0.963) / 0.995 (0.993) | 0.951 (0.934) / 0.991 (0.988) / 0.998 (0.997) | 0.953 (0.942) / 0.992 (0.990) / 0.998 (0.997) | - | 0.924 (0.900) / 0.985 (0.980) / 0.997 (0.996) | 0.952 (0.935) / 0.992 (0.988) / 0.998 (0.997) | 0.784 (0.703) / 0.930 (0.895) / 0.985 (0.977) |
| 3 | 0.842 (0.754) / 0.962 (0.935) / 0.993 (0.987) | 0.933 (0.895) / 0.988 (0.980) / 0.997 (0.996) | 0.936 (0.911) / 0.989 (0.984) / 0.997 (0.996) | - | 0.921 (0.879) / 0.985 (0.975) / 0.997 (0.995) | 0.934 (0.897) / 0.988 (0.980) / 0.997 (0.996) | 0.782 (0.643) / 0.930 (0.867) / 0.986 (0.971) |
| 4 | 0.786 (0.618) / 0.932 (0.854) / 0.985 (0.965) | 0.919 (0.864) / 0.984 (0.970) / 0.997 (0.994) | 0.922 (0.885) / 0.984 (0.973) / 0.997 (0.994) | - | 0.919 (0.864) / 0.984 (0.970) / 0.997 (0.994) | - | 0.782 (0.603) / 0.930 (0.846) / 0.985 (0.964) |
| 5 | 0.766 (0.550) / 0.922 (0.823) / 0.978 (0.951) | 0.908 (0.837) / 0.977 (0.959) / 0.990 (0.984) | 0.910 (0.862) / 0.977 (0.961) / 0.990 (0.983) | - | - | - | - |
| 6 | 0.747 (0.492) / 0.911 (0.788) / 0.971 (0.934) | 0.896 (0.811) / 0.971 (0.949) / 0.984 (0.977) | 0.899 (0.839) / 0.970 (0.950) / 0.984 (0.976) | - | - | - | - |

### Differences (rho0.7_nf6, all seven SNRs unless stated)

- Proposed channel-aware minus load masks, U=1: mean +0.000 to +0.043 (0 dB +0.043); weakest +0.000 to +0.043 (0 dB +0.043)
- Proposed channel-aware minus load masks, U=2: mean +0.000 to +0.075 (0 dB +0.075); weakest +0.000 to +0.107 (0 dB +0.107)
- Proposed channel-aware minus load masks, U=3: mean +0.001 to +0.094 (0 dB +0.094); weakest +0.001 to +0.156 (0 dB +0.156)
- Proposed channel-aware minus load masks, U=4: mean +0.002 to +0.136 (0 dB +0.136); weakest +0.004 to +0.267 (0 dB +0.267)
- Proposed channel-aware minus load masks, U=5: mean +0.002 to +0.145 (0 dB +0.145); weakest +0.004 to +0.312 (0 dB +0.312)
- Proposed channel-aware minus load masks, U=6: mean +0.002 to +0.151 (0 dB +0.151); weakest +0.006 to +0.346 (0 dB +0.346)
- slot counts: csiq minus csi, U=1: mean +0.000 to +0.000 (0 dB +0.000); weakest +0.000 to +0.000 (0 dB +0.000)
- slot counts: csiq minus csi, U=2: mean +0.000 to +0.002 (0 dB +0.002); weakest +0.000 to +0.008 (0 dB +0.008)
- slot counts: csiq minus csi, U=3: mean +0.000 to +0.003 (0 dB +0.003); weakest +0.000 to +0.016 (0 dB +0.016)
- slot counts: csiq minus csi, U=4: mean -0.000 to +0.003 (0 dB +0.003); weakest -0.000 to +0.021 (0 dB +0.021)
- slot counts: csiq minus csi, U=5: mean -0.000 to +0.003 (0 dB +0.003); weakest -0.001 to +0.025 (0 dB +0.025)
- slot counts: csiq minus csi, U=6: mean -0.000 to +0.002 (0 dB +0.002); weakest -0.001 to +0.028 (0 dB +0.028)
- free counts: csiq minus csiqe (equal counts, greedy placement), U=2: mean +0.000 to +0.001 (0 dB +0.001); weakest +0.000 to +0.006 (0 dB +0.006)
- free counts: csiq minus csiqe (equal counts, greedy placement), U=3: mean +0.000 to +0.002 (0 dB +0.002); weakest +0.000 to +0.011 (0 dB +0.011)
- free counts: csiq minus csiqe (equal counts, greedy placement), U=4: mean -0.000 to +0.002 (0 dB +0.002); weakest -0.000 to +0.014 (0 dB +0.014)
- free counts: csiq minus csiqe (equal counts, greedy placement), U=5: mean +0.000 to +0.002 (0 dB +0.002); weakest -0.000 to +0.017 (0 dB +0.017)
- free counts: csiq minus csiqe (equal counts, greedy placement), U=6: mean +0.000 to +0.002 (0 dB +0.002); weakest -0.001 to +0.018 (0 dB +0.018)
- greedy placement at the design load: csiqe minus Static ODS-6 with known gains, U=6: mean -0.000 to +0.001 (0 dB +0.001); weakest -0.001 to +0.010 (0 dB +0.010)
- csiq minus Static ODS-6 with known gains, U=1: mean +0.013 to +0.070 (0 dB +0.070); weakest +0.013 to +0.070 (0 dB +0.070)
- csiq minus Static ODS-6 with known gains, U=2: mean +0.013 to +0.050 (0 dB +0.050); weakest +0.014 to +0.066 (0 dB +0.066)
- csiq minus Static ODS-6 with known gains, U=3: mean +0.013 to +0.036 (0 dB +0.036); weakest +0.015 to +0.059 (0 dB +0.059)
- csiq minus Static ODS-6 with known gains, U=4: mean +0.012 to +0.023 (0 dB +0.023); weakest +0.014 to +0.047 (0 dB +0.047)
- csiq minus Static ODS-6 with known gains, U=5: mean +0.006 to +0.012 (0 dB +0.012); weakest +0.005 to +0.036 (0 dB +0.036)
- csiq minus Static ODS-6 with known gains, U=6: mean -0.000 to +0.002 (0 dB +0.002); weakest -0.001 to +0.028 (0 dB +0.028)
- csiq minus Static ODS-6 with known gains, energy d_s/U, U=1: mean +0.006 to +0.013 (0 dB +0.006); weakest +0.006 to +0.013 (0 dB +0.006)
- csiq minus Static ODS-6 with known gains, energy d_s/U, U=2: mean +0.005 to +0.013 (0 dB +0.005); weakest +0.009 to +0.014 (0 dB +0.009)
- csiq minus Static ODS-6 with known gains, energy d_s/U, U=3: mean +0.004 to +0.013 (0 dB +0.004); weakest +0.011 to +0.015 (0 dB +0.013)
- csiq minus Static ODS-6 with known gains, energy d_s/U, U=4: mean +0.003 to +0.013 (0 dB +0.003); weakest +0.009 to +0.017 (0 dB +0.017)
- csiq minus Static ODS-6 with known gains, energy d_s/U, U=5: mean +0.002 to +0.006 (0 dB +0.002); weakest +0.003 to +0.021 (0 dB +0.021)
- Static ODS-6: known gains minus fixed parts, U=1: mean +0.002 to +0.153 (0 dB +0.153); weakest +0.002 to +0.153 (0 dB +0.153)
- Static ODS-6: known gains minus fixed parts, U=2: mean +0.002 to +0.155 (0 dB +0.155); weakest +0.003 to +0.220 (0 dB +0.220)
- Static ODS-6: known gains minus fixed parts, U=3: mean +0.002 to +0.154 (0 dB +0.154); weakest +0.004 to +0.255 (0 dB +0.255)
- Static ODS-6: known gains minus fixed parts, U=4: mean +0.002 to +0.154 (0 dB +0.154); weakest +0.004 to +0.288 (0 dB +0.288)
- Static ODS-6: known gains minus fixed parts, U=5: mean +0.002 to +0.153 (0 dB +0.153); weakest +0.005 to +0.317 (0 dB +0.317)
- Static ODS-6: known gains minus fixed parts, U=6: mean +0.002 to +0.154 (0 dB +0.154); weakest +0.005 to +0.335 (0 dB +0.335)
- csiq minus Static ODS-4 with known gains (ods), U=1: +0.000 to +0.048 (0 dB +0.048)
- csiq minus Static ODS-4 with known gains (odsm), U=1: +0.000 to +0.000 (0 dB +0.000)
- csiq minus Static ODS-4 with known gains (ods), U=2: +0.000 to +0.029 (0 dB +0.029)
- csiq minus Static ODS-4 with known gains (odsm), U=2: +0.000 to +0.001 (0 dB +0.001)
- csiq minus Static ODS-4 with known gains (ods), U=3: +0.000 to +0.015 (0 dB +0.015)
- csiq minus Static ODS-4 with known gains (odsm), U=3: +0.000 to +0.002 (0 dB +0.002)
- csiq minus Static ODS-4 with known gains (ods), U=4: -0.000 to +0.003 (0 dB +0.003)

### Delayed CSI at the scheduler (rho0.7_nf6, mean at 0 / 10 / 20 dB)

| rho_b | U | fixed | csi | csiq | ods | odsm |
|---|---|---|---|---|---|---|
| 0.9 | 1 | 0.927 / 0.988 / 0.997 | 0.963 / 0.994 / 0.998 | 0.963 / 0.994 / 0.998 | 0.885 / 0.968 / 0.984 | 0.958 / 0.982 / 0.985 |
| 0.9 | 2 | 0.876 / 0.974 / 0.995 | 0.941 / 0.989 / 0.997 | 0.943 / 0.990 / 0.997 | 0.887 / 0.968 / 0.984 | 0.939 / 0.979 / 0.985 |
| 0.9 | 3 | 0.842 / 0.962 / 0.993 | 0.921 / 0.985 / 0.997 | 0.924 / 0.986 / 0.997 | 0.885 / 0.967 / 0.984 | 0.921 / 0.976 / 0.985 |
| 0.9 | 4 | 0.784 / 0.932 / 0.985 | 0.906 / 0.980 / 0.996 | 0.909 / 0.980 / 0.996 | 0.883 / 0.967 / 0.983 | 0.906 / 0.973 / 0.984 |
| 0.9 | 5 | 0.766 / 0.920 / 0.978 | 0.894 / 0.973 / 0.990 | 0.897 / 0.973 / 0.990 | 0.884 / 0.966 / 0.983 | 0.895 / 0.969 / 0.984 |
| 0.9 | 6 | 0.747 / 0.911 / 0.971 | 0.881 / 0.966 / 0.983 | 0.884 / 0.966 / 0.983 | 0.881 / 0.966 / 0.983 | - |
| 0.7 | 1 | 0.928 / 0.988 / 0.997 | 0.941 / 0.989 / 0.997 | 0.941 / 0.989 / 0.997 | 0.843 / 0.954 / 0.980 | 0.938 / 0.978 / 0.984 |
| 0.7 | 2 | 0.875 / 0.974 / 0.995 | 0.911 / 0.980 / 0.996 | 0.913 / 0.981 / 0.996 | 0.844 / 0.953 / 0.981 | 0.910 / 0.972 / 0.984 |
| 0.7 | 3 | 0.842 / 0.963 / 0.993 | 0.886 / 0.973 / 0.994 | 0.889 / 0.975 / 0.995 | 0.841 / 0.952 / 0.980 | 0.886 / 0.966 / 0.983 |
| 0.7 | 4 | 0.782 / 0.932 / 0.985 | 0.868 / 0.966 / 0.993 | 0.871 / 0.966 / 0.993 | 0.841 / 0.951 / 0.980 | 0.868 / 0.960 / 0.982 |
| 0.7 | 5 | 0.765 / 0.921 / 0.978 | 0.855 / 0.958 / 0.986 | 0.858 / 0.959 / 0.987 | 0.843 / 0.952 / 0.980 | 0.856 / 0.956 / 0.981 |
| 0.7 | 6 | 0.745 / 0.913 / 0.971 | 0.839 / 0.951 / 0.980 | 0.841 / 0.951 / 0.980 | 0.839 / 0.951 / 0.980 | - |

- delayed CSI, the previous-block gains follow the AR(1) process and every current slot coefficient is rho_b times its previous value plus an independent CN(0, 1 - rho_b^2) term (adjacent-slot correlation of the current block 0.7 rho_b^2); the delayed runs use their own draws of sentences and noise
- rho_b=0.9, U=1: csiq minus load masks +0.0356 / +0.0144 / +0.0052 / +0.0017 / +0.0005 / +0.0002 / +0.0001 (0 to 30 dB)
- rho_b=0.9, U=2: csiq minus load masks +0.0670 / +0.0373 / +0.0159 / +0.0060 / +0.0020 / +0.0007 / +0.0002 (0 to 30 dB)
- rho_b=0.9, U=3: csiq minus load masks +0.0818 / +0.0492 / +0.0234 / +0.0108 / +0.0042 / +0.0016 / +0.0006 (0 to 30 dB)
- rho_b=0.9, U=4: csiq minus load masks +0.1247 / +0.0843 / +0.0483 / +0.0239 / +0.0106 / +0.0044 / +0.0019 (0 to 30 dB)
- rho_b=0.9, U=5: csiq minus load masks +0.1307 / +0.0885 / +0.0523 / +0.0264 / +0.0113 / +0.0047 / +0.0020 (0 to 30 dB)
- rho_b=0.9, U=6: csiq minus load masks +0.1368 / +0.0950 / +0.0543 / +0.0273 / +0.0121 / +0.0049 / +0.0020 (0 to 30 dB)
- rho_b=0.9: largest loss of csiq, perfect minus delayed, 0.0145 (U=6, 0 dB); corrected by the load-mask difference between the draws 0.0147 (U=6, 0 dB)
- rho_b=0.7, U=1: csiq minus load masks +0.0133 / +0.0025 / +0.0002 / -0.0006 / -0.0007 / -0.0002 / -0.0001 (0 to 30 dB)
- rho_b=0.7, U=2: csiq minus load masks +0.0378 / +0.0204 / +0.0075 / +0.0024 / +0.0006 / +0.0001 / -0.0000 (0 to 30 dB)
- rho_b=0.7, U=3: csiq minus load masks +0.0470 / +0.0277 / +0.0119 / +0.0050 / +0.0019 / +0.0007 / +0.0003 (0 to 30 dB)
- rho_b=0.7, U=4: csiq minus load masks +0.0881 / +0.0616 / +0.0342 / +0.0175 / +0.0083 / +0.0031 / +0.0012 (0 to 30 dB)
- rho_b=0.7, U=5: csiq minus load masks +0.0934 / +0.0627 / +0.0383 / +0.0180 / +0.0085 / +0.0033 / +0.0014 (0 to 30 dB)
- rho_b=0.7, U=6: csiq minus load masks +0.0969 / +0.0658 / +0.0382 / +0.0187 / +0.0089 / +0.0040 / +0.0013 (0 to 30 dB)
- rho_b=0.7: largest loss of csiq, perfect minus delayed, 0.0572 (U=6, 0 dB); corrected by the load-mask difference between the draws 0.0546 (U=6, 0 dB)

### Fading correlation and the gain estimate at U=4 (nf6, mean at 0 / 10 / 20 dB)

- rho=0.3: fixed 0.785/0.932/0.985; csi 0.924/0.985/0.997; csiq 0.926/0.985/0.997; ods 0.904/0.973/0.984
  - rho=0.3: csiq minus load masks at 0 dB +0.1406
- rho=0.7: fixed 0.786/0.932/0.985; csi 0.919/0.984/0.997; csiq 0.922/0.984/0.997; ods 0.899/0.972/0.984; odsfix 0.746/0.910/0.971; odsm 0.919/0.976/0.985; csiqe 0.921/0.984/0.997
  - rho=0.7: csiq minus load masks at 0 dB +0.1359
- rho=0.95: fixed 0.782/0.933/0.985; csi 0.896/0.978/0.996; csiq 0.900/0.978/0.996; ods 0.872/0.964/0.983
  - rho=0.95: csiq minus load masks at 0 dB +0.1177
- rho=0.7, known slot gains (genie): fixed 0.791/0.932/0.985; csi 0.920/0.984/0.997; csiq 0.922/0.984/0.997; ods 0.899/0.971/0.984; odsfix 0.751/0.911/0.971
  - rho=0.7, known slot gains (genie): csiq minus load masks at 0 dB +0.1315
- known slot gains minus blind estimates, U=4, rho=0.7 (all seven SNRs): fixed -0.000 to +0.005 (0 dB +0.0045); csi -0.000 to +0.000 (0 dB +0.0002); csiq -0.000 to +0.000 (0 dB +0.0001); ods -0.000 to +0.000 (0 dB +0.0002); odsfix -0.000 to +0.006 (0 dB +0.0059)

### Learned references with CSI (Code/pilot_corr.py, seed 1, rho=0.7 nf6, mean at 0 / 10 / 20 dB)

- ods1_block_nf6: U1 0.914/0.981/0.993
- ods1_netpw_nf6: U1 0.938/0.983/0.994
- ods2_block_nf6: U2 0.858/0.967/0.989
- ods2_netpw_nf6: U2 0.910/0.968/0.992
- ods3_block_nf6: U3 0.819/0.954/0.983
- ods3_netpw_nf6: U3 0.873/0.938/0.980
- ods4_block_nf6: U4 0.788/0.942/0.978
- ods4_csisort_nf6: U4 0.902/0.984/0.997
- ods4_netpw_all_nf6: U1 0.848/0.911/0.963; U2 0.844/0.911/0.966; U3 0.843/0.914/0.967; U4 0.840/0.916/0.967
- ods4_netpw_nf6: U4 0.861/0.919/0.967
- ods6_block_nf6: U6 0.741/0.920/0.965
- ods6_csisort_nf6: U6 0.878/0.970/0.984
- ods6_netpw_all_nf6: U1 0.846/0.895/0.943; U2 0.845/0.897/0.945; U3 0.843/0.897/0.945; U4 0.843/0.898/0.945; U5 0.842/0.899/0.944; U6 0.841/0.900/0.944
- ods6_netpw_nf6: U6 0.868/0.934/0.976
- prop_csisort_nf6: U1 0.963/0.988/0.992; U2 0.945/0.990/0.997; U3 0.924/0.986/0.996; U4 0.909/0.982/0.995; U5 0.896/0.975/0.990; U6 0.884/0.968/0.984
- prop_fixed_nf6: U1 0.903/0.971/0.986; U2 0.870/0.962/0.988; U3 0.832/0.951/0.984; U4 0.809/0.939/0.979; U5 0.789/0.928/0.970; U6 0.773/0.918/0.961
- prop_netpw_nf6: U1 0.935/0.975/0.991; U2 0.906/0.958/0.990; U3 0.872/0.932/0.984; U4 0.854/0.915/0.972; U5 0.832/0.899/0.956; U6 0.831/0.889/0.941
- prop_netq_nf6: U1 0.963/0.989/0.992; U2 0.947/0.989/0.996; U3 0.918/0.958/0.979; U4 0.877/0.789/0.926; U5 0.854/0.789/0.905; U6 0.839/0.774/0.899
- prop_netqpw_nf6: U1 0.966/0.991/0.994; U2 0.945/0.987/0.995; U3 0.906/0.967/0.983; U4 0.856/0.928/0.959; U5 0.832/0.918/0.955; U6 0.820/0.916/0.955

### Learned references with CSI without SNR offsets (seed 1, rho=0.7, mean at 0 / 10 / 20 dB; last column: largest learned minus csiq of pca_csi_U*_k6_rho0.7 over the seven SNRs)

- ods4_block: U4 0.802/0.949/0.980; largest minus csiq -0.0128 (U=4, 30 dB)
- ods4_netpw: U4 0.845/0.925/0.975; largest minus csiq -0.0098 (U=4, 30 dB)
- ods6_block: U6 0.754/0.928/0.968; largest minus csiq -0.0110 (U=6, 30 dB)
- ods6_netpw_all: U1 0.854/0.862/0.938; U2 0.854/0.880/0.956; U3 0.852/0.882/0.958; U4 0.850/0.884/0.960; U5 0.849/0.885/0.960; U6 0.850/0.886/0.961; largest minus csiq -0.0145 (U=6, 30 dB)
- prop_csisort: U1 0.971/0.992/0.995; U2 0.955/0.993/0.998; U3 0.936/0.989/0.997; U4 0.922/0.986/0.997; U5 0.909/0.979/0.991; U6 0.898/0.972/0.984; largest minus csiq +0.0003 (U=2, 30 dB)
- prop_fixed: U1 0.917/0.973/0.987; U2 0.885/0.968/0.990; U3 0.847/0.957/0.986; U4 0.823/0.947/0.983; U5 0.804/0.935/0.973; U6 0.788/0.926/0.964; largest minus csiq -0.0044 (U=2, 30 dB)
- prop_netqpw: U1 0.976/0.996/0.999; U2 0.944/0.991/0.998; U3 0.909/0.969/0.992; U4 0.852/0.913/0.968; U5 0.843/0.902/0.959; U6 0.832/0.900/0.958; largest minus csiq +0.0005 (U=2, 25 dB)
- csiq without SNR offsets: U1 0.976/0.996/0.998; U2 0.957/0.993/0.998; U3 0.941/0.991/0.998; U4 0.928/0.987/0.997; U5 0.916/0.980/0.991; U6 0.905/0.973/0.984

## 13. Mask support against mask values

A mask is its support (the slots of a user) times its values (the amplitudes). Block fading: the values with P_i proportional to lambda_i (proposal/prop of pca_eval.json) against the allocation (12) (proposal/opt). Slot-varying fading (rho=0.7): fixedb = load masks with equal values on all used entries, fixed = load masks with (12), csiqb = channel-aware support with equal values, csiqw = channel-aware support with (12) at unit gain, csiq = channel-aware masks with (13). The csiq family shares the greedy grant and all draws.

- block fading, (12) minus values proportional to lambda_i over U=1..6, 8 and all SNRs: +0.0003 to +0.0071 (largest at U=1, 5 dB)

### offsets up to +-6 dB: mean per-user CosSim at 0 / 10 / 20 dB

| U | fixedb | fixed | csiqb | csiqw | csiq |
|---|---|---|---|---|---|
| 1 | 0.915/0.981/0.995 | 0.927/0.988/0.997 | 0.956/0.990/0.996 | 0.969/0.995/0.998 | 0.970/0.995/0.998 |
| 2 | 0.871/0.964/0.992 | 0.878/0.974/0.995 | 0.938/0.984/0.996 | 0.949/0.992/0.998 | 0.953/0.992/0.998 |
| 3 | 0.837/0.947/0.984 | 0.842/0.962/0.993 | 0.921/0.979/0.995 | 0.930/0.988/0.997 | 0.936/0.989/0.997 |
| 4 | 0.786/0.925/0.980 | 0.786/0.932/0.985 | 0.907/0.973/0.993 | 0.916/0.984/0.996 | 0.922/0.984/0.997 |
| 5 | 0.765/0.916/0.973 | 0.766/0.922/0.978 | 0.896/0.966/0.987 | 0.904/0.977/0.990 | 0.910/0.977/0.990 |
| 6 | 0.747/0.906/0.966 | 0.747/0.911/0.971 | 0.885/0.960/0.981 | 0.892/0.970/0.984 | 0.899/0.970/0.984 |
- support with equal values, csiqb minus fixedb, 0 dB: +0.0416 (U=1, 0 dB) to +0.1379 (U=6, 0 dB)
- channel-aware masks minus load masks, csiq minus fixed, 0 dB: +0.0428 (U=1, 0 dB) to +0.1515 (U=6, 0 dB)
- values (13) on the channel-aware support, csiq minus csiqb, 0 dB: +0.0136 (U=1, 0 dB) to +0.0152 (U=3, 0 dB); from 10 dB on: +0.0010 (U=1, 30 dB) to +0.0112 (U=4, 10 dB)
- gain-aware part of the values, csiq minus csiqw, all SNRs: +0.0000 (U=1, 30 dB) to +0.0070 (U=6, 0 dB)
- values (12) on the load masks, fixed minus fixedb, all SNRs: +0.0003 (U=6, 0 dB) to +0.0149 (U=3, 10 dB)

### no offsets: mean per-user CosSim at 0 / 10 / 20 dB

| U | fixedb | fixed | csiqb | csiqw | csiq |
|---|---|---|---|---|---|
| 1 | 0.923/0.983/0.996 | 0.939/0.991/0.998 | 0.961/0.991/0.997 | 0.974/0.996/0.998 | 0.976/0.996/0.998 |
| 2 | 0.877/0.968/0.993 | 0.887/0.979/0.996 | 0.941/0.986/0.996 | 0.954/0.993/0.998 | 0.957/0.993/0.998 |
| 3 | 0.848/0.951/0.986 | 0.856/0.967/0.994 | 0.926/0.981/0.995 | 0.937/0.990/0.998 | 0.941/0.991/0.998 |
| 4 | 0.795/0.931/0.982 | 0.798/0.939/0.987 | 0.913/0.975/0.993 | 0.923/0.986/0.997 | 0.928/0.987/0.997 |
| 5 | 0.772/0.922/0.976 | 0.775/0.929/0.981 | 0.901/0.969/0.988 | 0.911/0.980/0.991 | 0.916/0.980/0.991 |
| 6 | 0.755/0.912/0.969 | 0.757/0.919/0.973 | 0.891/0.963/0.981 | 0.899/0.973/0.984 | 0.905/0.973/0.984 |
- support with equal values, csiqb minus fixedb, 0 dB: +0.0381 (U=1, 0 dB) to +0.1358 (U=6, 0 dB)
- channel-aware masks minus load masks, csiq minus fixed, 0 dB: +0.0370 (U=1, 0 dB) to +0.1475 (U=6, 0 dB)
- values (13) on the channel-aware support, csiq minus csiqb, 0 dB: +0.0141 (U=6, 0 dB) to +0.0159 (U=2, 0 dB); from 10 dB on: +0.0009 (U=1, 30 dB) to +0.0115 (U=4, 10 dB)
- gain-aware part of the values, csiq minus csiqw, all SNRs: +0.0000 (U=1, 30 dB) to +0.0055 (U=6, 0 dB)
- values (12) on the load masks, fixed minus fixedb, all SNRs: +0.0010 (U=1, 30 dB) to +0.0164 (U=3, 10 dB)

## 14. Greedy slot grant against a local search over slot counts and positions (rho=0.7, offsets up to +-6 dB)

csiq = greedy grant of the channel-aware masks; csiql = csiq followed by single-slot moves and pairwise swaps between users while the sum MSE under (13) decreases (best move per frame and round). Same draws, placement, allocation, and receiver.

| U | SNR | csiq | csiql | csiql - csiq | weakest csiq / csiql | sum MSE greedy -> search | frames changed | counts changed | rounds |
|---|---|---|---|---|---|---|---|---|---|
| 2 | 0 dB | 0.9531 | 0.9533 | +0.0002 | 0.9416 / 0.9418 | 0.17124 -> 0.17071 (-0.31 %) | 0.357 | 0.038 | 3 |
| 2 | 10 dB | 0.9920 | 0.9920 | +0.0000 | 0.9899 / 0.9900 | 0.02478 -> 0.02467 (-0.47 %) | 0.282 | 0.074 | 4 |
| 2 | 20 dB | 0.9977 | 0.9977 | +0.0000 | 0.9967 / 0.9967 | 0.00256 -> 0.00254 (-0.63 %) | 0.281 | 0.078 | 3 |
| 3 | 0 dB | 0.9360 | 0.9365 | +0.0005 | 0.9105 / 0.9106 | 0.34715 -> 0.34491 (-0.65 %) | 0.820 | 0.158 | 7 |
| 3 | 10 dB | 0.9888 | 0.9890 | +0.0001 | 0.9838 / 0.9841 | 0.05464 -> 0.05391 (-1.33 %) | 0.655 | 0.119 | 6 |
| 3 | 20 dB | 0.9974 | 0.9974 | +0.0000 | 0.9958 / 0.9959 | 0.00595 -> 0.00584 (-1.77 %) | 0.678 | 0.155 | 7 |
| 4 | 0 dB | 0.9221 | 0.9228 | +0.0007 | 0.8851 / 0.8847 | 0.56533 -> 0.56051 (-0.85 %) | 0.928 | 0.309 | 9 |
| 4 | 10 dB | 0.9841 | 0.9846 | +0.0005 | 0.9731 / 0.9739 | 0.10758 -> 0.10452 (-2.85 %) | 0.943 | 0.265 | 8 |
| 4 | 20 dB | 0.9967 | 0.9968 | +0.0002 | 0.9940 / 0.9944 | 0.01298 -> 0.01188 (-8.44 %) | 0.965 | 0.002 | 8 |
| 5 | 0 dB | 0.9103 | 0.9112 | +0.0009 | 0.8623 / 0.8618 | 0.81213 -> 0.80359 (-1.05 %) | 0.957 | 0.347 | 10 |
| 5 | 10 dB | 0.9771 | 0.9777 | +0.0006 | 0.9609 / 0.9631 | 0.19441 -> 0.18943 (-2.56 %) | 0.947 | 0.415 | 11 |
| 5 | 20 dB | 0.9904 | 0.9905 | +0.0001 | 0.9830 / 0.9836 | 0.06701 -> 0.06595 (-1.58 %) | 0.961 | 0.434 | 10 |
| 6 | 0 dB | 0.8987 | 0.8999 | +0.0012 | 0.8386 / 0.8385 | 1.09128 -> 1.07878 (-1.15 %) | 0.948 | 0.374 | 12 |
| 6 | 10 dB | 0.9702 | 0.9711 | +0.0009 | 0.9498 / 0.9524 | 0.30724 -> 0.29757 (-3.15 %) | 0.978 | 0.295 | 11 |
| 6 | 20 dB | 0.9838 | 0.9841 | +0.0003 | 0.9761 / 0.9769 | 0.14683 -> 0.14389 (-2.00 %) | 0.991 | 0.043 | 12 |

- mean CosSim, search minus greedy: +0.0000 to +0.0012; weakest user: -0.0005 to +0.0026; sum MSE change: -8.44 to -0.31 percent; points: 15
