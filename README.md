# Load-Adaptive Semantic Multiplexing via Nested Principal-Component Codes and Dynamic Masks

Ki-Ho Lee, Hyun-Ho Choi, and Jung-Ryun Lee

Status: under review at IEEE Communications Letters.

This repository holds the seeded scripts, the stored results, the trained reference
models, and the run configurations of the paper. The scripts in the map below print
every figure, table, and number of the manuscript and of its response letter from the
stored results alone. No training, simulation, network access, or GPU is needed for that.

## What the paper studies

`U` users send sentence embeddings to one receiver over a block of `d_s` real channel
uses, and the load `U` changes from frame to frame. Real-valued masks are the general
description of how the users share the block. A mask weights every channel use of a
user, and the masks of two users may overlap. The best masks differ with the condition
of the frame, namely the load, the SNR, the subband gains and their age, and the content
the users share. Learned masks would need one model per condition. For a nested
principal-component code and a linear receiver of least MSE, the paper gives the masks
in closed form per condition.

* **Nested code.** Each user projects its centered BERT embedding (1) on the principal
  components of the training embeddings (2) and repeats the coefficients `K` times (3).
  The leading coefficients carry the most variance, so the code can be cut at any length.
* **One mask design.** The masks (4) of a frame solve the design (P), the least MSE of
  the joint LMMSE receiver (9) at the energy of each user.
* **Load masks (5).** They are one member of (4). Their supports do not overlap and
  cover the block. For independent content under block fading with known gains, the
  optimum of (P) up to the full load `U = K` needs no overlap, and the load masks with the
  allocation (16) attain it when every squared gain equals its mean. Proposed applies them
  with the blind gain estimate (11) and the Wiener estimate (12). Below full load the
  active users take over the uses and the energy of the absent ones. In overload every
  user keeps the leading part of its code.
* **Channel-aware masks (19).** With fresh subband gains, each mask orders the groups of
  the code by the gain of the subband. The coherence-aware choice compares them with the
  load masks over draws of the aged gains (7) and announces the masks of lower MSE.
* **Overlapping masks (18).** For users who describe one scene, the masks overlap
  wherever a coefficient has shared content, and they send the shared part together.
  The joint LMMSE receiver weighs each signal by the shared content. It is a linear form
  of user-wise attention, and real-valued masks are the means for learned attention to
  handle inter-user interference.

## System and energy conventions

* Block: `d_b = 768`, `K = 4`, `d_s = K d_b = 3072` real uses, `N = 24` subbands (slots)
  of `L = d_s/N = 128` uses. Light load is `U < K`, full load `U = K`, and overload `U > K`.
* Energy: user `u` spends `E_u = d_s/U` per block unless stated. Static ODS-`k` spends
  `d_s/k`, or `d_s/U` in its variant. There is no peak-power constraint.
* Channel (6): `g_{u,s} = beta_u |h_{u,s}|`, noise variance `1/SNR` per real use. The
  fading vector `h_u = [h_{u,0}, ..., h_{u,N-1}]` is `CN(0, R_F)` with
  `[R_F]_{ss'} = rho_f^{|s-s'|}`, independent across users. It ages over frames by (7)
  with the gain correlation `eta`. Every user pre-compensates the phase of each subband.
* Block fading is `rho_f = 1`, one gain per user and block. Section IV uses it with
  `beta_u = 1` and `eta = 0`, which is Rayleigh fading independent across users and
  frames, unless subband gains are stated.
* Near-far offsets: with subband gains, `20 log10 beta_u` is uniform within +-6 dB per
  user and frame (file tag `_nf6` in `data/pilot_corr/` and in
  `data/extra/overlap_bound2_*_nf6.json`). The near-far tests under block fading use
  fixed offsets spread evenly over +-6 dB, at `U = 4` the offsets -6, -2, 2, and 6 dB.
  These are the +-6 dB test of Section IV-B, its rows in Table R1.4 of the response
  letter, and `data/extra/overlap_bound_U*_nf6.json`.
* Metric: the per-user CosSim between the centered embedding and its estimate. The
  results for one scene report the NMSE with paired bootstrap 95 percent confidence
  intervals.
* Protocol: AG News gives 8,000 training and 2,000 test sentences, encoded by BERT-base
  (uncased). Every point evaluates 2,000 test frames per user at 0 to 30 dB in 5 dB steps.
  For one scene, the users of a frame send different COCO captions of one of 2,000 test
  images, with the statistics fitted on 8,000 training images.

## Schemes and labels in the result files

Some result files and number files predate the names of the manuscript. The response
letter lists this mapping, and the table repeats it. `data/general_numbers.md` and
`data/pca_numbers.md` number the allocations (16) and (19) as (12) and (13), and
`data/pca_numbers.md` numbers the load masks (5) as (4). Section 11 of
`data/pca_numbers.md` checks (10), (14), and (16) under the numbers (7), (10), and (12).
In `data/dyn_numbers.md`, Section 8a refers to Table I as Table II, and the rows labeled
Proposed one model are the Learned nested code. Sections C and D of
`data/general_numbers.md` carry the labels Fig. 3(a) and Fig. 3(b) of an earlier layout.
Section C holds Fig. 3, and Section D holds Table R2.6 and panel (a) of Fig. R2.6 of the
response letter. Several script comments use the same earlier numbers.

| Name in the manuscript | Result files and labels |
|---|---|
| Proposed | `data/extra/pca_eval.json` key `proposal.opt`, `data/extra/pca_eval_load.json` (`U = 7`), label `proposal` in `pca_numbers.md` |
| Static ODS-`k` | `pca_eval.json` key `static`, label Static PCA-`k` in `pca_numbers.md` |
| Static ODS-`k` with energy `d_s/U` | `pca_eval.json` key `matched` |
| Learned ODS-`U` | `data/ortho_U<U>_s<seed>.json`, label learned Static ODS-`k` in `pca_numbers.md` and Static ODS-`k` in `dyn_numbers.md` |
| Learned nested code | trained as `data/shared_dyns_s<seed>.json`, reported as its evaluation `data/shared_dyns_clamp_s<seed>.json`, label proposal in `dyn_numbers.md` |
| Learned shared embedding | `data/shared_agn_s<seed>.json`, label Overlapping masks in `pca_numbers.md` |
| ToDMA, best `T` | `data/extra/todma_redim.json` |
| load masks (5) | mode `fixed` in `data/pilot_corr/pca_csi_*.json`, modes `load_pu` (per-user receiver) and `load_joint` (joint LMMSE receiver) in `data/shared/`, `closed_name` `load12` in `data/extra/overlap_bound2_indep_equal_*.json` |
| channel-aware masks (19) | mode `csiq` in `data/pilot_corr/pca_csi_*.json`, `closed_name` `csiq13` in `data/extra/overlap_bound2_indep_slot-*.json` |
| Static ODS-`k` placed by the same gains | mode `ods` (energy `d_s/k`) and `odsm` (energy `d_s/U`, files `_x`) |
| coherence-aware choice | mode `switch` in `data/pilot_corr/pca_csi_coh_*.json` |
| overlapping masks (18) | mode `gen_joint` in `data/shared/`, `closed_name` `prop2` (pruned in overload: `prop2_pruned`) in `data/extra/overlap_bound2_*.json`, label Proposition 2 in `general_numbers.md` |
| overlapping masks with gains estimated without pilots | mode `gen_joint_em0` (EM from the frame) against `gen_joint_true0` (known gains) in `data/shared/pilot/` |
| Learned references with subband gains | `data/pilot_corr/{prop,ods<k>}_<mode>[_nf6]_s1.json` |

## Layout

| Path | Content |
|---|---|
| `Code/general_numbers.py` | prints `data/general_numbers.md`: load adaptivity (A), one scene (B, C), coherence (D), search against the closed forms (E), gain estimation (F), index (G) |
| `Code/pca_numbers.py`, `Code/rev_dyn_numbers.py` | print `data/pca_numbers.md` and `data/dyn_numbers.md` |
| `Code/fig_rev.py` | draws Figs. 2 and 3 of the manuscript and Figs. R1.2 and R2.6 of the response letter |
| `fig1_system_rev.tex`, `Code/fig1_amplitudes.py` | TikZ source of Fig. 1 and the check of its bar heights |
| `Code/letter_r2_checks.py`, `Code/letter_checks/*.py` | read-only checks that print the letter values not printed above |
| `Code/pca_eval.py`, `Code/pca_eval_load.py`, `Code/pca_mc.py` | closed-form evaluations of Proposed and Static ODS-`k`, and the Monte Carlo check |
| `Code/pca_csi.py`, `Code/pca_csi_counts.py`, `Code/pca_csi_ls.py` | load masks, channel-aware masks, and their references under subband gains, CPU |
| `Code/shared_masks.py`, `Code/shared_theory.py`, `Code/content_stats.py`, `Code/shared_pilot.py` | one scene: caption runs, Gaussian model, content statistics, gain estimation without pilots, CPU |
| `Code/overlap_bound.py` | numerical search against the closed-form masks with the joint LMMSE receiver, CPU |
| `Code/rev_lib.py`, `Code/rev_run.py`, `Code/rev_extra.py`, `Code/pilot_corr.py`, `Code/mask_partition.py` | learned references: models, training, evaluation, and evaluations of stored checkpoints |
| `Code/todma_redim.py` | ToDMA with the best number of token slots. It takes the data split and the encoder from `Code/cl_experiments.py`, which imports `Code/bert_semcom.py`. Both are code of the original submission, and their header comments describe the submitted design. |
| `Code/rev_cache.py`, `Code/emb_captions.py` | embedding caches `data/emb_*.pt` (not stored) |
| `Code/*_queue.sh`, `Code/extra_runs.sh` | run queues with generic paths |
| `data/` | one result file per run (`json`), per-frame arrays (`npz`) where a confidence interval needs them, the number files `*.md` |
| `data/ckpt/`, `data/pilot_corr/*.pt` | trained reference models (see Trained models) |
| `fig_cl/` | the figure PDFs of the manuscript and of the response letter |

## Figure and table map

Run every command from the repository root. "Module" means the code at the top level of
the script, and the section names the part of the printed file.

| Item | Script and function | Data files |
|---|---|---|
| Fig. 1 (transceiver and masks in four frames) | `fig1_system_rev.tex` (pdflatex). `Code/fig1_amplitudes.py`: `main()`, `row()`, `alloc()` recompute the bar heights of frames `tau` and `tau+1` and check them against the TikZ source, which stores the bar heights of all four frames | `data/effdim_spectrum_test.npy`, `fig1_system_rev.tex`, `data/shared/content_stats.json` (group energies behind frame `tau+3`) |
| Fig. 2 (per-user CosSim versus the load) | `Code/fig_rev.py --one`: `fig_load_one("nines")`, then `fig2_choose()` writes `fig_cl/fig2_load_one.pdf` | `data/extra/pca_eval.json`, `data/extra/pca_eval_load.json`, `data/ortho_U{1..6}_s{1,2,3}.json`, `data/ortho_U8_s1.json` |
| Fig. 3 (NMSE reduction of the overlapping masks, one scene) | `Code/fig_rev.py --one`: `fig_scene_one()` with `theory_red()` and `real_red()` | `data/shared/theory_coco_mix{0.4,0.8}_K4_U{2..8}.json`, `data/shared/shared_coco_mix0.8_K4_U{2..8}_ol.npz`, `data/shared/shared_coco_mix0_K4_U8_ol.npz` |
| Table I | `Code/pca_numbers.py`: module, Section 10. `Code/letter_checks/tableI_margins.py`: module | `pca_eval.json`, `data/shared_dyns_clamp_s{1,2,3}.json`, `data/ortho_U{1,6}_s{1,2,3}.json`, `data/shared_agn_s{1,2,3}.json`, `data/extra/todma_redim.json` |
| Section III-C, III-D (search against the closed forms) | `Code/general_numbers.py`: `sec_E()` | `data/extra/overlap_bound2_*.json`, `data/extra/overlap_bound_U{2,4,6,8}_nf{0,6}.json` |
| Section IV-A, IV-C (one scene) | `Code/general_numbers.py`: `sec_B()`, `sec_C()`, `sec_F()` | `data/shared/content_stats.json`, `data/shared/shared_*.json` and `.npz`, `data/shared/theory_*.json`, `data/shared/pilot/shared_pilot_*.json` |
| Section IV-B, paragraphs 1 and 2 | `Code/general_numbers.py`: `sec_A()`. `Code/pca_numbers.py`: module, Sections 1 to 8 and 11. `Code/letter_checks/learned_margins.py`, `block_fading_checks.py`: module | `pca_eval.json`, `pca_eval_load.json`, `pca_mc.json`, `todma_redim.json`, `data/ortho_U*_s*.json`, `data/sst2_ortho_U4_s1.json`, `data/distil_ortho_U4_s1.json` |
| Section IV-B, paragraph 3 (subband gains and their age) | `Code/general_numbers.py`: `sec_D()`. `Code/pca_numbers.py`: module, Section 12. `Code/letter_checks/coh_minima.py`: module | `data/pilot_corr/pca_csi_coh_*.json` and `.npz`, `data/pilot_corr/pca_csi_U*_k6_rho0.7_nf6.json` |
| Letter Table R1.1 | `Code/letter_checks/letter_tables.py`: module, part 1 (`pca_numbers.py` Section 9 gives 10 dB) | `pca_eval.json`, `data/ortho_U{1..6}_s{1,2,3}.json` |
| Letter Table R1.2 | `Code/general_numbers.py`: `sec_A()` (A.1 to A.3). `Code/rev_dyn_numbers.py`: module, Sections 1 and 8 | `pca_eval.json`, `pca_eval_load.json`, `data/ortho_U*_s*.json`, `data/shared_dyns_clamp_s*.json`, `data/shared_dyns_U8_s1.json` |
| Letter Fig. R1.2 | `Code/fig_rev.py`: `fig_load_col()` writes `fig_cl/fig2_load_col.pdf` | as Fig. 2 |
| Letter Table R1.4 | `Code/pca_numbers.py`: module, Section 8 | `pca_eval.json`, `data/ortho_U4_s1.json`, `data/sst2_ortho_U4_s1.json`, `data/distil_ortho_U4_s1.json`, `data/ortho_U8_s1.json` |
| Letter Table R1.5 | `Code/letter_checks/comment_checks.py`: module, block R1-4. `Code/rev_dyn_numbers.py`: module, Sections 1, 5, 8, 8a. `Code/general_numbers.py`: `sec_A()` (A.1, Proposed) | `data/shared_U4_s{1,2,3}.json`, `data/shared_U4_randmask_s{1,2,3}.json`, `data/shared_agn_s*.json`, `data/ortho_U4_s*.json`, `data/shared_dyns_clamp_s1.json`, `data/shared_dyns_freezetx_s1.json`, `pca_eval.json` |
| Letter Table R2.3 | `Code/letter_r2_checks.py`: module, part 1. `Code/pca_numbers.py`: Sections 4 and 5 | `pca_eval.json` |
| Letter Table R2.4 | `Code/pca_numbers.py`: module, Section 12 (learned references with subband gains) | `data/pilot_corr/pca_csi_U{4,6}_k6_rho0.7_nf6.json`, `pca_csi_U4_k4_rho0.7_nf6.json`, `prop_*_nf6_s1.json`, `ods{4,6}_*_nf6_s1.json` |
| Letter Table R2.5a | `Code/letter_r2_checks.py`: module, part 3. `Code/pca_numbers.py`: Sections 1 and 6 | `pca_eval.json`, `data/ortho_U{1,4,6}_s1.json`, `data/ortho_U{1,4,6}_{trlow,trhigh}_s1.json` |
| Letter Table R2.5b | `Code/letter_checks/letter_tables.py`: module, part 5 | field `cfg` of `data/ortho_U4_s1.json`, `data/shared_dyns_s1.json`, `data/shared_agn_s1.json` |
| Letter Table R2.6 | `Code/general_numbers.py`: `sec_D()` (D.1 to D.3) | `data/pilot_corr/pca_csi_coh_rho*_eta*_U*_nf6.json` and `.npz` |
| Letter Fig. R2.6 | `Code/fig_rev.py`: `fig_scene_col()` with `coh()` writes `fig_cl/fig3_scene_col.pdf` | `data/pilot_corr/pca_csi_coh_*_nf6.json`, panel (b) as Fig. 3 |
| Letter Table R2.7a | `Code/letter_checks/letter_tables.py`: module, part 2 | `pca_eval.json`, seed-1 files `ortho_U4`, `ortho_U6`, `sst2_ortho_U4`, `sst2_ortho_U6`, `distil_ortho_U4`, `shared_dyns_clamp`, `sst2_shared_dyns`, `distil_shared_dyns` in `data/` |
| Letter Table R2.7b | `Code/pca_numbers.py`: module, Sections 1 and 8 | `pca_eval.json` (classifiers), `data/ortho_U4_s1.json`, `data/sst2_ortho_U4_s1.json` |
| Letter Table R2.8 | `Code/letter_checks/r28_checks.py`: module | `pca_eval.json`, `data/ortho_U{1..6}_s{1,2,3}.json`, `todma_redim.json` |
| Letter, Comment 1.2 (overlap in overload) | `Code/general_numbers.py`: `sec_E()` (E.2) | `data/extra/overlap_bound_U{6,8}_nf6.json` |
| Letter, Comment 1.4 (common fading) | `Code/letter_checks/comment_checks.py`: module, block R1-6 | `data/shared_U4_common_s1.json`, `data/ortho_U4_common_s1.json`, `data/shared_U4_s1.json`, `data/ortho_U4_s1.json` |
| Letter, Comment 2.1 (training curves, 300 epochs) | `Code/letter_checks/comment_checks.py`: block R2-6. `Code/letter_checks/letter_tables.py`: part 3 | field `conv` of `data/ortho_U*_s*.json`, `data/shared_nest_s1.json`, `data/shared_nest_e300_s1.json` |
| Letter, Comment 2.4 (Learned nested code under subband gains) | `Code/letter_r2_checks.py`: part 4 | `data/pilot_corr/prop_fixed_nf6_s1.json`, `data/pilot_corr/pca_csi_U*_k6_rho0.7_nf6.json` |
| Letter, Comment 2.5 (MAML and adaptation) | `Code/letter_r2_checks.py`: part 2. `Code/letter_checks/letter_only_checks.py`: module. `Code/rev_dyn_numbers.py`: Section 8 | `data/shared_dyns_joint_s1.json`, `data/shared_U4_{alpha1e-4,alpha2e-3,inner3,mb2,mb6}_s1.json`, `data/extra/noadapt_shared_dyns_s*.json` |
| Letter, Comment 2.6 (subband gains) | `Code/letter_checks/delayed_loss.py`, `coh_minima.py`: module. `Code/letter_checks/letter_tables.py`: part 4. `Code/pca_numbers.py`: Sections 12 to 14 (subband counts at the end of Section 12) | `data/pilot_corr/pca_csi_coh_*.json`, `data/pilot_corr/pca_csi_counts_U*_nf*.json`, `data/pilot_corr/pca_csi_ls_*.json`, `data/pilot_corr/*_vals.json` |
| Letter, Comment 2.6 (estimation errors at the receiver) | `Code/letter_checks/letter_tables.py`: part 6. `Code/general_numbers.py`: `sec_F()`. `Code/pca_numbers.py`: Sections 6 and 12 (known gains) | `data/shared/pilot/shared_pilot_*.json`, `data/shared/pilot/shared_pilot_coco_K4_U4_E1_emauto0_fs_it30_snr0.npz` (per-frame arrays of the CI), `pca_eval.json`, `data/pilot_corr/pca_csi_U4_k6_rho0.7_nf6_genie.json` |

Tables R1.3 and R2.1 of the letter state the rules and the components of the schemes and
need no data. The comment block at the top of each script describes its inputs, and the
map lists the data files. `data/effdim_spectrum_test.npy`, the input of
`Code/fig1_amplitudes.py`, holds the cumulative eigenvalue share of the second moment of
the centered, l2-normalized test embeddings of `data/emb_ag_news_bert-base-uncased.pt`,
the cache that `Code/rev_cache.py` builds. No script of this repository writes it. The
docstring of `Code/fig1_amplitudes.py` describes it and compares it with the second
moment of the test split of that cache.

## Regenerate the numbers and figures

```bash
python Code/general_numbers.py --public      # data/general_numbers.md
python Code/pca_numbers.py                   # data/pca_numbers.md
python Code/rev_dyn_numbers.py               # data/dyn_numbers.md
python Code/fig_rev.py                       # Figs. 2 and 3, Figs. R1.2 and R2.6, and earlier variants
python Code/fig_rev.py --one                 # Figs. 2 and 3 only
python Code/fig1_amplitudes.py               # bar heights of Fig. 1
SOURCE_DATE_EPOCH=1794873600 FORCE_SOURCE_DATE=1 latexmk -pdf -outdir=out fig1_system_rev.tex \
  && cp out/fig1_system_rev.pdf fig_cl/       # Fig. 1
python3 -I Code/letter_r2_checks.py data
python3 -I Code/letter_checks/letter_tables.py data
for f in block_fading_checks comment_checks learned_margins letter_only_checks r28_checks tableI_margins; do
  python3 -I Code/letter_checks/$f.py data
done
python3 -I Code/letter_checks/coh_minima.py data/pilot_corr
python3 -I Code/letter_checks/delayed_loss.py data/pilot_corr
python3 -I Code/letter_checks/fig3_ci_vs_marker.py data/general_numbers.md
```

The three number files are regenerated byte for byte. `Code/fig_rev.py` draws with the
Arial font. With Arial and the versions below, it reproduces the stored PDFs byte for
byte. With the fixed source date above and pdfTeX 1.40.29, latexmk reproduces the stored
PDF of Fig. 1 byte for byte. Without it, the PDF differs only in its creation date and
file identifier. `Code/fig_rev.py` also writes PNG copies and earlier variants that neither
document includes, and `.gitignore` lists them.

## Rerun the experiments

Rerun in a separate clone. The runs write into `data/` and `data/ckpt/` and replace the
stored result files and checkpoints, and the number files then no longer match the stored
ones byte for byte. The queues change to the folder of the script, so they run from any
working directory. `PY` selects the interpreter (default `python`) and `GPU` the GPU index
(default 0). Training runs use CUDA, then Apple MPS, then the CPU, whichever is present
first. Every run writes one result file.

The one-scene, coherence, search, and gain-estimation queues take `N_PAR` parallel runs
and `THREADS` BLAS threads per run. The `csi_*` queues use one BLAS thread per run.
`csi_queue.sh` and `csi_extra_queue.sh` take `N_PAR` (default 14 and 12), and
`csi_equal_queue.sh`, `csi_vals_queue.sh`, and `csi_ls_queue.sh` start all their runs at
once.

`Code/rev_queue.sh` skips a run whose result file already exists in `OUT` (default
`data/`), and every reported learned run is stored there. To retrain them, give it an empty
folder, for example `OUT=../rerun bash Code/rev_queue.sh main_ortho` (`OUT` is relative to
`Code/`). It passes `--per-user-rx` to every run of Learned shared embedding and its
variants, as their field `cfg.per_user_rx` records. `Code/coherence_queue.sh` skips a run
whose stored result is complete, so move those files out of `data/pilot_corr/` to rerun
them. Every other queue reruns its jobs in a fresh clone and overwrites their outputs in
`data/`.

1. Embedding caches (not stored): `python Code/rev_cache.py` builds AG News with BERT-base
   and with DistilBERT and SST-2 with BERT-base. `python Code/emb_captions.py --dataset coco`
   and `--dataset flickr` read the caption sets from the default Hugging Face cache
   `~/.cache/huggingface/hub` and ignore `HF_HOME`. Download them first with
   `huggingface-cli download yerevann/coco-karpathy --repo-type dataset` and
   `huggingface-cli download nlphuji/flickr30k flickr_annotations_30k.csv --repo-type dataset`.
2. Closed forms: `python Code/pca_eval.py`, `python Code/pca_eval_load.py`,
   `python Code/pca_mc.py`. They run on the CPU and read the embedding cache with PyTorch.
3. Subband gains, CPU: `bash Code/csi_queue.sh`, `csi_extra_queue.sh`,
   `csi_equal_queue.sh`, `csi_vals_queue.sh`, `csi_ls_queue.sh`, and
   `bash Code/coherence_queue.sh`. Subband counts:
   `python Code/pca_csi_counts.py --U <U> --nearfar <0 or 6>` for `U` = 2, 3, 4, 6.
4. One scene, CPU: `bash Code/shared_theory_queue.sh` (runs `Code/content_stats.py`
   first when `data/shared/content_stats.json` is missing), `bash Code/shared_masks_queue.sh`,
   `bash Code/overload_queue.sh`, and `bash Code/pilot_queue.sh` (gain estimation).
5. Search against the closed forms, CPU: `bash Code/overlap_bound_queue.sh [original|run2|all]`.
6. Learned references, GPU optional: `OUT=<empty folder> bash Code/rev_queue.sh <stage>`
   with the stages `main_ortho`, `ortho_s3`, `scale`, `main`, `decomp`, `ablate`, `ablate_b`, `general`,
   and `snr`. `bash Code/dyn_queue.sh <stage>` with `main`, `clamp`, `ablate`, `eval`,
   `history`, `pcainit`, `todma`, and `report`. Then `bash Code/extra_runs.sh` and
   `bash Code/corr_queue.sh` (learned references with subband gains).

Seeds: the evaluations of Proposed and Static ODS-`k` draw with the seed 123 + SNR, so the
schemes of one point share their draws. The header of each script gives its other seeds,
for example those of the caption runs and of the gain estimation. The learned references
use the training seeds 1, 2, and 3. The bootstrap intervals resample frames with
`numpy.random.default_rng(555 + SNR)`.

Result files store their arguments in `args`, `cfg`, `meta`, or `protocol`. The
evaluations of stored checkpoints in `data/extra/` store the name, seed, and device of the
evaluated model, and `Code/dyn_queue.sh` gives the ToDMA settings. The learned references
were trained on NVIDIA GPUs and on an Apple M5 Max. Their field `cfg` stores the device and
the PyTorch version, except for the learned references with subband gains in
`data/pilot_corr/`. GPU training is not bit-exact, so a rerun reproduces their values
within the seed spread of the paper. The field `code` of the gain-estimation results
records the MD5 of `Code/shared_pilot.py` and `Code/shared_masks.py` at run time.
`Code/shared_masks.py` matches it. The released `Code/shared_pilot.py` differs from that
copy in comment lines and help texts only.

## Software

* Python 3.10, numpy 2.2.5, and matplotlib 3.10.8 for the number files and figures. The
  letter checks need only the Python standard library, except part 6 of
  `Code/letter_checks/letter_tables.py`, which reads per-frame arrays with numpy. The
  number files, figures, and letter checks need no PyTorch.
* PyTorch for every experiment script, which reads the embedding caches with torch, and
  for the learned references: 2.11.0 with CUDA 12.8 or 13.0 on NVIDIA GPUs, and 2.5.1 on
  Apple MPS.
* transformers 4.46.3 and datasets 3.1.0 for the embedding caches.
* scipy for `pca_csi.py` and the scripts that import it (`pca_csi_counts.py`,
  `pca_csi_ls.py`, `overlap_bound.py`), `pilot_corr.py`, `shared_masks.py`,
  `shared_theory.py`, and `shared_pilot.py`. pyarrow and pandas for `emb_captions.py`.
* A TeX installation with TikZ and the `standalone` class for Fig. 1.

## Datasets and models

The embeddings come from public datasets and encoders, which keep their licenses and terms
of use. Read them on the dataset and model cards before reuse.

* AG News (X. Zhang, J. Zhao, and Y. LeCun, 2015), Hugging Face `fancyzhx/ag_news`.
* COCO captions (T.-Y. Lin et al., 2014), Karpathy split `yerevann/coco-karpathy`. The
  8,000 fit images and 2,000 test images are drawn with seed 0 from the training part of
  the Karpathy split. The COCO annotations are released under CC BY 4.0, and the images
  under the Flickr terms.
* SST-2 (R. Socher et al., 2013), Hugging Face `stanfordnlp/sst2`.
* Flickr30k captions (P. Young et al., 2014), Hugging Face `nlphuji/flickr30k`, response
  letter only.
* BERT-base uncased (J. Devlin et al., 2019) and DistilBERT base uncased (V. Sanh et al.,
  2019), Hugging Face `bert-base-uncased` and `distilbert-base-uncased`, Apache 2.0.

## Trained models

Git tracks the checkpoints listed in `.gitignore`.

* Learned ODS-`U`: `data/ckpt/ortho_U{1..6}_s{1,2,3}.pt` and `data/ckpt/ortho_U8_s1.pt`.
* Learned nested code: `data/ckpt/shared_dyns_s{1,2,3}.pt`. The reported values are its
  evaluations `data/shared_dyns_clamp_s*.json` (field `reeval_of`).
* Learned shared embedding: `data/ckpt/shared_agn_s1.pt`.
* Learned ODS-4 and ODS-6 on SST-2 and Learned ODS-4 with DistilBERT:
  `data/ckpt/sst2_ortho_U4_s1.pt`, `sst2_ortho_U6_s1.pt`, `distil_ortho_U4_s1.pt`.
* Learned ODS-4 under common fading: `data/ckpt/ortho_U4_common_s1.pt`.
* The four-user version of Learned shared embedding: `data/ckpt/shared_U4_s1.pt`.
* Learned references with subband gains: `data/pilot_corr/*_s1.pt`.

Each seed of Learned shared embedding exceeds 50 MB, the largest file that GitHub accepts
without a warning. Seed 1 has been in the repository since an earlier version. To keep the
repository small, the other checkpoints of reported runs are in the release asset of this
version on the Releases page of the repository. These are seeds 2 and 3 of Learned shared
embedding, the four-user models of the response letter, and the learned runs that only the
response letter reports. The asset:

* file `load-adaptive-semantic-multiplexing-models.tar.gz`
* SHA-256 `3fb2a54ad54829aa95080e3dac019d3df052c740f02066e718de90d1b6b5317d`

Unpack it at the repository root with
`tar -xzf load-adaptive-semantic-multiplexing-models.tar.gz`. It writes these files to
`data/ckpt/`, which `.gitignore` keeps out of git:

* `shared_agn_s2.pt`, `shared_agn_s3.pt` (Learned shared embedding, Table I)
* `shared_U4_s2.pt`, `shared_U4_s3.pt`, `shared_U4_randmask_s{1,2,3}.pt` (Table R1.5)
* `shared_U4_common_s1.pt` (common fading, Comment 1.4)
* `shared_U4_{alpha1e-4,alpha2e-3,inner3,mb2,mb6}_s1.pt` (MAML configuration, Comment 2.5)
* `shared_dyns_{U8,joint,K1,K8,freezetx,pcainit}_s1.pt` (variants of the Learned nested
  code: eight users, joint training instead of meta-learning, `K = 1` and `K = 8`, a
  transmit projection frozen at random values, and the start from the principal components)
* `sst2_shared_dyns_s1.pt`, `distil_shared_dyns_s1.pt` (Learned nested code on SST-2 and
  with DistilBERT)
* `ortho_U{1,4,6}_{trlow,trhigh}_s1.pt` (Learned ODS-`U` trained on 0 to 10 dB or on 15 to
  25 dB, Table R2.5a)
* `shared_nest_s1.pt`, `shared_nest_e300_s1.pt` (training curves and the 300-epoch run,
  Comment 2.1)

No script that regenerates the numbers or figures reads a checkpoint. Each result file in
`data/` holds the training configuration of its run in the field `cfg`, and the queues
above retrain them.

## Earlier versions

The original submission is kept at the tag `original-submission`, the revision with
overlapping learned masks at `parity-revision-20261021`, the Learned nested code as the
proposal at `learned-nested-20261103`, and the closed-form version of 2026-11-04 at
`closed-form-pca-20261104`.

## Citation

The paper is under review at IEEE Communications Letters. Until it appears, please cite
this repository and the earlier shared-embedding paper.

```bibtex
@misc{lee2026code,
  author       = {Lee, Ki-Ho and Choi, Hyun-Ho and Lee, Jung-Ryun},
  title        = {Code and data for load-adaptive semantic multiplexing},
  howpublished = {[Online]. Available: \url{https://github.com/KiHoLee/load-adaptive-semantic-multiplexing}},
  year         = {2026}
}
```

> K.-H. Lee, H.-H. Choi, and J.-R. Lee, "Load-adaptive semantic multiplexing via nested
> principal-component codes and dynamic masks," under review at *IEEE Commun. Lett.*

> K.-H. Lee, H.-H. Choi, and J.-R. Lee, "Transformer-based shared embedding for multiple
> access in semantic communications," *IEEE J. Sel. Areas Commun.*, vol. 44,
> pp. 2622-2637, 2026.
