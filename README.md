# Load-Adaptive Semantic Multiplexing via Nested Principal-Component Codes and Dynamic Masks

Code, stored results, and trained reference models for the IEEE Communications
Letters paper by Ki-Ho Lee, Hyun-Ho Choi, and Jung-Ryun Lee (under review).
Every figure and every number of the paper and of its supplementary results can
be regenerated from this repository.

## What the letter studies

`U` distributed users send sentences over one block of `d_s = K * d_b` real
channel uses (`d_b = 768`, `K = 4`, `d_s = 3072`, 24 slots of 128 uses), and the
load `U` changes from frame to frame between 1 and 6 (8 in a generality check).
The receiver announces the SNR, the load, and a rank for each active user.

* **Nested code.** Each user projects its centered BERT sentence embedding on
  the principal components of the training embeddings (PCA, eigenvalues in
  decreasing order) and repeats the 768 coefficients cyclically over the block.
  Leading coefficients carry the largest variance, so the code can be cut at
  any length.
* **Dynamic masks.** Each mask is a real-valued vector. Its support, the slots
  `s` with `s mod U = j` of the user of rank `j` at load `U`, assigns the
  channel uses, and its values, the amplitudes of the power allocation below,
  set the power. A user sends all coefficients, with repetitions, at light load
  and fewer coefficients under overload. Under block fading, for users with
  independent content and a linear receiver (successive interference
  cancellation with linear estimates included), no overlapping mask lowers the
  MSE at equal energy per coefficient up to the full load `U = K`, because the
  disjoint supports give every coefficient its own uses (Section III).
* **Closed-form power and receiver.** Coefficient `i` receives the power
  `P_i = max(0, sqrt(lambda_i s2 / nu) - s2)` with `sum P_i = d_s/U` at the
  announced noise variance `s2` (classical optimal linear coding,
  K.-H. Lee and D. P. Petersen, IEEE Trans. Commun., 1976). The receiver
  estimates the gain blindly from the received energy and applies the Wiener
  factor per coefficient. Nothing is trained.
* **Channel-aware masks (Section IV-E).** When the gain varies over the slots,
  as over subbands, and the receiver knows it, every used slot carries one group
  of 128 coefficients. The groups of a user occupy its slots in decreasing order
  of gain, and coefficient `i` on a slot of gain `g_i` receives
  `P_i = max(0, sqrt(lambda_i s2 / nu) / g_i - s2 / g_i^2)` (the same classical
  result). After a first slot for every user, the receiver grants each further
  slot greedily to the user whose MSE decreases most, at the energy `d_s/N` per
  slot, so the slot count and the energy of a user follow its channel. The
  greedy grant is a heuristic and is not shown to be optimal.

Main results (per-user cosine similarity, 2,000 frames per point and user):

* The proposal exceeds the learned partition of every load (Learned ODS-`U`,
  three-seed means) by 0.006 to 0.047, and a learned nested code with the same
  masks by 0.006 to 0.022.
* At its design load, a fixed partition with the same code (Static ODS-`k`)
  equals the proposal. At light load the proposal exceeds Static ODS-6 by up to
  0.137 and Static ODS-4 by up to 0.101, through the energy and, for short
  parts, the channel uses of absent users.
* Re-dimensioned ToDMA leads by at most 0.003 at four of 28 points. The masks
  need the announced active set.
* Under slot-varying fading with offsets of up to 6 dB, the channel-aware masks
  exceed the load masks at 0 dB by 0.043 (`U = 1`) to 0.151 (`U = 6`), and by
  up to 0.346 for the weakest user of a frame. At `U = 6`, Static ODS-6 with
  the same known gains lies within 0.003 in the mean. The channel-aware masks
  lead it by 0.028 for the weakest user at 0 dB, 0.018 of it from the slot
  counts and the energy that follows them. Below six users, the variants of
  Static ODS-4 and ODS-6 with known gains and the energy `d_s/U` lie within
  0.002 (`U <= 3`) and trail by 0.002 to 0.013, respectively. A user of the
  channel-aware masks spends `n * d_s/N` for its `n` granted slots, which
  concentrates the energy on fewer uses, with no peak-power limit.
* Learned transceivers with channel state (single runs, `Code/pilot_corr.py`)
  trail the channel-aware masks or tie them within 0.001, with leads of at most
  0.0006. With the load masks, the learned nested code of these runs exceeds
  the closed-form load masks at `U = 4` to 6 by up to 0.025 at 0 and 5 dB. Its
  receiver knows the slot gains and its draws differ, so the cause is not
  isolated.
* Slot counts of the greedy grant (`Code/pca_csi_counts.py`): the grant follows
  the MSE, not a rate, so it tends to give more slots to weaker users. At 0 dB
  with offsets of up to 6 dB, the weakest user of a frame receives more than
  `N/U` slots in 83 to 91 percent of the frames at `U = 2` to 6. At `U = 4` and
  6 the counts approach `N/U` as the SNR grows.
* Mask support against mask values (`Code/pca_csi.py` modes `fixedb`, `csiqw`,
  `csiqb`, files `*_vals.json`, Section 13 of `data/pca_numbers.md`): under
  block fading, values that ignore the SNR cost at most 0.007. Under
  slot-varying fading at 0 dB, the channel-aware supports with equal values
  gain 0.042 to 0.138 over the load masks with equal values, and on these
  supports the values (13) add 0.014 to 0.015.
* Slot positions of the greedy grant (`Code/pca_csi_ls.py`, files
  `pca_csi_ls_*.json`, Section 14 of `data/pca_numbers.md`): a local search
  that moves and swaps slots between users after the grant lowers the sum MSE
  by 0.3 to 8.4 percent but raises the mean CosSim by at most 0.0012 at
  `U = 2` to 6 and 0, 10, and 20 dB. It is a local optimum, not a bound.
* Real-valued overlapping masks with the joint LMMSE receiver
  (`Code/overlap_bound.py`, Gaussian model): at `U <= 4` every coefficient of
  the load masks has its own uses, and no overlapping design improves on them.
  At `U = 6` and 8, a gradient search finds no gain with equal gains and lowers
  the mean NMSE by at most 2.2 percent with gains spread over +-6 dB, for the
  strongest users and at the cost of the weakest ones. The search is not
  convex.

## Conventions

* Power: every user transmits at unit average power over the channel uses it
  occupies (energy `d_s/U`; at `U = 5` ranks 0 to 3 have 640 uses at average
  power 0.96 and rank 4 has 512 uses at 1.2). A Static ODS-`k` user spends
  `d_s/k`. Transmission is analog without a peak-power constraint.
* Channel: each user has its own Rayleigh coefficient `h_u ~ CN(0,1)`, constant
  within a block, with the phase pre-compensated at the transmitter, and the
  noise variance is `1/SNR` per real use. Section IV-E replaces it by
  slot-varying fading: per user `h_{u,s} = 0.7 h_{u,s-1} + sqrt(0.51) w_{u,s}`
  over the 24 slots, independent across users, with an offset uniform within
  +-6 dB per user and frame (tag `nf6`). The receiver knows the slot gains, or
  those of the previous block with correlation `eta` (`--delay`), and every
  user pre-compensates all slot phases.
* Metric: per-user CosSim between the centered BERT embedding and its estimate.
  The aggregate fidelity `S(U)` is compared only between schemes that serve the
  same `U` users.

## Schemes

| Name in the paper | Source | Description |
|---|---|---|
| Proposed | `Code/pca_eval.py`, `proposal/opt` in `data/extra/pca_eval.json` | PCA nested code, dynamic masks, closed-form power, blind-gain Wiener receiver |
| Static ODS-`k` | `static` in `pca_eval.json` | the same code on `k` fixed parts of `d_s/k` uses, energy `d_s/k`, parts of absent users silent |
| Static ODS-`k` with energy `d_s/U` | `matched` in `pca_eval.json` | a Static ODS-`k` user increases its power on its part to spend `d_s/U` |
| Learned ODS-`k` | `ortho_U<k>` | learned linear projection, LN, and linear decoder per part, SNR-aware MAML, one model per `k` |
| Learned nested code | `shared_dyns`, evaluated as `shared_dyns_clamp`; `shared_dyns_pcainit` | one learned transceiver with the supports of the dynamic masks and learned SNR-dependent mask values; also started from the PCA basis (`rev_run.py --init pca`) |
| Overlapping masks | `shared_agn` | learned overlapping masks and user-wise attention of the original submission, one model for all loads |
| ToDMA, best `T` | `extra/todma_redim.json` | single-antenna adaptation: first `T` BERT tokens over `T` slots of `3072/T` uses, OMP detection, genie token count per slot and user association |
| Channel-aware masks | `csiq` in `data/pilot_corr/pca_csi_*.json` | ordered groups on the strongest slots, power (13), greedy slot grant at `d_s/N` per slot, blind per-slot gain estimate |
| Load masks under slot-varying fading | `fixed` in `pca_csi_*.json` | the proposal with the masks (4), no channel state at the transmitter |
| Static ODS-`k` with known gains | `ods` (energy `d_s/k`), `odsm` (energy `d_s/U`, files `_x`) | `N/k` slots per user by the assignment maximizing the sum of `log(1 + gamma g^2)`, power (13) |
| Controls of Section IV-E | `csi` (slot counts of the load masks), `csiqe` (counts capped at the load masks, files `_e`), `odsfix` (fixed contiguous parts) | separate placement, slot counts, and channel state |
| Supports and values | `fixedb`, `csiqb` (equal values on all used entries), `csiqw` (values (12) on the channel-aware supports), files `_vals` | separate the supports of the masks from their values |
| Local search over slot positions | `csiql` in `pca_csi_ls_*.json` | the greedy grant followed by single-slot moves and pairwise swaps while the sum MSE under (13) decreases |
| Learned references with channel state | `data/pilot_corr/{prop,ods<k>}_<mode>[_nf6]_s1.json` | single runs (seed 1) of the learned nested code and Learned ODS-`k` with the placements `fixed`, `block`, `csisort`, `netq`, `netpw`, `netqpw` (mask network with slot counts and per-slot weights) |

## Layout

| Path | Content |
|---|---|
| `Code/pca_eval.py` | every evaluation of the closed-form proposal and of Static ODS-`k`: loads, SNRs, variants, `K`, robustness, SST-2, DistilBERT, classifiers |
| `Code/pca_mc.py` | Monte Carlo check of the closed-form error and allocation |
| `Code/pca_numbers.py` | prints every number of the paper and of its supplementary results into `data/pca_numbers.md`, from `data/` only |
| `Code/pca_fig.py` | regenerates Figs. 2 and 3 and the supplementary figure `fig4_pca_agg.pdf` from `data/` only |
| `Code/rev_lib.py`, `Code/rev_run.py` | learned references: data split and embedding cache, transceivers (`SharedMuxNest` is the learned nested code, `OrthoDS` the learned partition), MAML training, adaptation, evaluation |
| `Code/rev_queue.sh`, `Code/dyn_queue.sh`, `Code/extra_runs.sh`, `Code/rev_extra.py` | training and evaluation runs of the learned references |
| `Code/mask_partition.py` | own-energy share of the transmitted energy per mask (Overlapping masks) |
| `Code/todma_redim.py` | ToDMA re-dimensioned to 24, 32, and 48 slots of the same block |
| `Code/pca_csi.py` | closed-form channel-aware masks and their references under slot-varying fading (Section IV-E), CPU |
| `Code/csi_queue.sh`, `Code/csi_extra_queue.sh`, `Code/csi_equal_queue.sh` | every `pca_csi.py` run of the paper and of the supplementary results |
| `Code/pilot_corr.py`, `Code/corr_queue.sh` | learned references with channel state, one GPU |
| `Code/pca_csi_counts.py` | slot counts of the greedy grant per gain rank, on the draws of `pca_csi.py`, CPU |
| `Code/csi_vals_queue.sh` | the `pca_csi.py` runs that separate the supports of the masks from their values |
| `Code/pca_csi_ls.py`, `Code/csi_ls_queue.sh` | local search over the slot counts and positions after the greedy grant, on the draws of `pca_csi.py`, CPU |
| `Code/overlap_bound.py` | load masks against optimized disjoint and real-valued overlapping masks with the joint LMMSE receiver, Gaussian model, CPU |
| `data/pilot_corr/` | results of `pca_csi.py` (`pca_csi_*.json`), `pca_csi_counts.py` (`pca_csi_counts_*.json`), `pca_csi_ls.py` (`pca_csi_ls_*.json`), and `pilot_corr.py` (`*_s1.json`) |
| `Code/rev_cache.py` | builds the embedding caches `data/emb_*.pt` (not stored; AG News, SST-2, BERT-base, DistilBERT) |
| `Code/rev_dyn_numbers.py`, `Code/rev_dynamic_fig.py`, and the other `rev_*` scripts | numbers and figures of the earlier versions, kept with their results |
| `data/extra/pca_eval.json`, `data/extra/pca_mc.json` | results of the closed-form proposal |
| `data/*.json`, `data/extra/` | one file per learned run and the evaluations of stored checkpoints |
| `data/pca_numbers.md`, `data/dyn_numbers.md` | every number quoted in the paper (closed form) and of the learned references |
| `data/ckpt/` | checkpoints of the learned nested code for seeds 1 to 3, of Learned ODS-1 to 6 for seed 1, and of earlier designs |
| `fig_cl/` | Figs. 1 to 3 of the paper and the supplementary figure `fig4_pca_agg.pdf` |

## Figure and table map

| Item | Data | Script |
|---|---|---|
| Fig. 1 | transceiver diagram (TikZ): the real-valued mask and the support of the dynamic mask at two loads | none |
| Fig. 2, per-user CosSim versus SNR | `pca_eval.json` (proposal), `ortho_U{1..6}_s{1,2,3}` | `pca_fig.py` |
| Fig. 3, per-user CosSim versus load | `pca_eval.json` (proposal, static) | `pca_fig.py` |
| Supplementary `S(U)` figure | `pca_eval.json` | `pca_fig.py` |
| Table I, ablations and reference designs | `pca_eval.json` (variants, `K`, matched), `shared_dyns_clamp_s*`, `shared_dyns_pcainit_s1`, `ortho_U*_s*`, `shared_agn_s*`, `extra/todma_redim.json` | `pca_numbers.py` |
| Section IV-D, robustness and generality (text) | `pca_eval.json` (robust, sst2, distil, classifiers), `ortho_U4_s1`, `sst2_ortho_U4_s1`, `distil_ortho_U4_s1`, `ortho_U8_s1` | `pca_numbers.py` |
| Section IV-E, slot-varying fading (text) | `pilot_corr/pca_csi_U*_k{6,4}_rho*_nf6*.json` | `pca_csi.py`, `pca_numbers.py` (Section 12 of `pca_numbers.md`) |
| Learned references with channel state | `pilot_corr/*_s1.json` | `pilot_corr.py`, `pca_numbers.py` |
| Energy division of Overlapping masks | `extra/mask_partition.json` | `mask_partition.py` |
| Slot counts of the greedy grant | `pilot_corr/pca_csi_counts_U{2,3,4,6}_nf{0,6}.json` | `pca_csi_counts.py --U <U> --nearfar <0 or 6>` |
| Section IV-E, supports against values | `pilot_corr/pca_csi_U{1..6}_k6_rho0.7[_nf6]_vals.json` | `csi_vals_queue.sh`, `pca_numbers.py` (Section 13) |
| Section IV-E, local search over slot positions | `pilot_corr/pca_csi_ls_U{2..6}_nf6_{0,10,20}.json` | `csi_ls_queue.sh`, `pca_numbers.py` (Section 14) |
| Overlapping masks with the joint LMMSE receiver | `extra/overlap_bound_U{2,4,6,8}_nf{0,6}.json` | `overlap_bound.py --U <U> --nearfar <0 or 6>` |
| Monte Carlo check | `extra/pca_mc.json` | `pca_mc.py` |

## Reproduction

Requirements: Python 3.10 or later, PyTorch, `transformers`, `datasets`,
`numpy`, `scipy`, `matplotlib`. Run the scripts from `Code/`.

```bash
cd Code
python rev_cache.py          # embedding caches (AG News, SST-2, BERT-base, DistilBERT)
python pca_eval.py           # closed-form proposal and Static ODS-k, about two minutes on a CPU
python pca_mc.py             # Monte Carlo check of the closed forms
python pca_numbers.py        # data/pca_numbers.md
python pca_fig.py            # Figs. 2 and 3 and the S(U) figure
bash csi_queue.sh && bash csi_extra_queue.sh && bash csi_equal_queue.sh   # Section IV-E, CPU
python pca_numbers.py        # again, to add Section 12
```

The learned references with channel state take about one hour per run on one
GPU: `GPU=0 bash corr_queue.sh`.

The learned references are trained as in the earlier versions:

```bash
PY=python EP=150 BS=64 N_PAR=3 bash rev_queue.sh main_ortho      # Learned ODS-1 to 6, three seeds (also stage scale, general)
PY=python EXTRA="--per-user-rx" EP=150 BS=64 N_PAR=3 bash rev_queue.sh main   # Overlapping masks
PY=python bash dyn_queue.sh main && PY=python bash dyn_queue.sh clamp        # learned nested code
python rev_run.py --name shared_dyns_pcainit --scheme shared_dyns --init pca --seed 1 --K 4 --U 6 --U-max 6 \
    --agnostic --epochs 150 --steps 100 --batch 64 --trials 2000 --eval-U 1 2 3 4 5 6 --out-dir ../data
```

The closed-form evaluations use the evaluation seed 123 + SNR, so the variants
of one scenario share their draws. The runs with the gains of the previous block
draw their own, and `pca_numbers.md` also gives their losses corrected by the
load masks. The learned references use the training seeds 1, 2, and 3
and were trained on NVIDIA GPUs and an Apple M5 Max. GPU training is not
bit-exact, so a rerun reproduces their values within the seed spread of the
paper.

## Earlier versions

The original submission is kept at the tag `original-submission`, the revision
with overlapping masks at `parity-revision-20261021`, and the learned nested
code as the proposal at `learned-nested-20261103`.

## Citation

The letter is under review. Until it appears, please cite this repository and
the earlier shared-embedding paper:

> K.-H. Lee, H.-H. Choi, and J.-R. Lee, "Code and data for load-adaptive
> semantic multiplexing," 2026. [Online]. Available:
> https://github.com/KiHoLee/load-adaptive-semantic-multiplexing

> K.-H. Lee, H.-H. Choi, and J.-R. Lee, "Transformer-based shared embedding
> for multiple access in semantic communications," *IEEE J. Sel. Areas
> Commun.*, vol. 44, pp. 2622-2637, 2026.
