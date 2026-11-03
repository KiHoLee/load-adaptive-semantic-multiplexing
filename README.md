# Load-Adaptive Semantic Multiplexing via Nested Expanded BERT Embeddings and Dynamic Masks

Code, stored results, and trained models for the IEEE Communications Letters
paper by Ki-Ho Lee, Hyun-Ho Choi, and Jung-Ryun Lee (under review). Every
figure and every number of the paper and of its supplementary results can be
regenerated from this repository.

## What the letter studies

`U` distributed users send sentences over one block of `d_s = K * d_b` real
channel uses (`d_b = 768`, `K = 4`, `d_s = 3072`), and the load `U` changes from
frame to frame between 1 and `U_max = 6`. The receiver announces the load and a
rank for each active user before a frame.

* Every user applies one common transmitter, which expands its frozen BERT
  sentence embedding into a nested code read as `N = 24` ordered segments of
  128 uses. The leading segments stand alone, and later segments refine them.
* Dynamic masks set by the announced load give the user of rank `j` the slots
  `s` with `s mod U = j`, so the masks are disjoint and cover the block. A user
  sends all 24 segments when it is alone and 4 segments at `U = 6`.
* A learned per-load profile, log-linear in the SNR and held at its 25 dB value
  above the training range, weights the entries. The receiver gathers the slots
  of a user, normalizes them blindly, and applies one linear decoder shared by
  all users and loads.
* One transceiver is trained once for all loads by first-order SNR-aware MAML,
  with the load drawn at random at every step. No mask is learned.

Main results (three-seed means):

* At equal channel uses and per-user power under independent fading, one model
  lies within 0.005 of the best learned static partition, Static ODS-`U`, at
  every load from 5 dB on, and exceeds it by 0.006 to 0.025 at 0 dB through its
  SNR-dependent profile.
* At light load it exceeds a fixed six-part partition by up to 0.165, since a
  static user confined to its own part leaves the energy and the channel uses
  of absent users idle. With the energy `d_s/U`, Static ODS-6 leads by at most
  0.009 up to 10 dB and trails from 15 dB on at `U <= 5`.
* Re-dimensioned ToDMA leads by up to 0.013 at high SNR. The masks need the
  announced active set.

## Conventions

* Power: every user transmits at unit average power over the channel uses it
  occupies. A Static ODS-`k` user spends `d_s/k` at every load `U <= k`, and the
  proposal and Static ODS-`U` spend `d_s/U`. At `U = 5` the 128-use slots give
  ranks 0 to 3 of the proposal 640 uses at average power 0.96 and rank 4 512
  uses at 1.2. Transmission is analog without a peak-power constraint.
* Channel: each user has its own Rayleigh coefficient `h_u ~ CN(0,1)`, constant
  within a block, with the phase pre-compensated at the transmitter, and the
  noise variance is `1/SNR`. The receiver uses no estimate of the individual
  gains.
* Metric: per-user cosine similarity between the centered BERT embedding and
  its estimate. The aggregate fidelity `S(U)` is the sum over the active users
  and is compared only between schemes that serve the same `U` users.

## Schemes

| Name in the paper | Run names | Description |
|---|---|---|
| Proposed (one model) | `shared_dyns`, evaluated as `shared_dyns_clamp` | nested expanded embedding, dynamic slot masks, SNR-dependent profile held above 25 dB, one shared decoder |
| SNR-independent profile | `shared_dyn` | the same design with a per-load profile that ignores the SNR |
| Static ODS-`k` | `ortho_U<k>` | `k` exclusive parts of `floor(d_s/k)` uses, learned linear projection, LN, and linear decoder, one model per `k` |
| Static ODS-6, energy `d_s/U` | `ortho_U6` evaluated by `rev_extra.py energy` | a Static ODS-6 user increases its power on its own part to spend `d_s/U`, evaluated at the SNR plus `10 log10(6/U)` dB |
| Overlapping masks | `shared_agn` | learned overlapping masks and user-wise attention of the original submission, one model for all loads |
| idle masks reassigned | `shared_abs` | Overlapping masks with the masks of absent users reassigned to the active users |
| ToDMA, best `T` | `extra/todma_redim.json` | single-antenna adaptation: first `T` BERT tokens over `T` slots of `3072/T` uses, OMP detection, genie token count per slot and user association |
| per-load retrains of the submitted design | `shared_U<U>` | Overlapping masks retrained for each load `U` (supplementary results) |
| per-load receive offset | `shared_nest` | earlier variant of the proposal, removed after one seed failed at `U = 1` (not reported in the paper) |

## Layout

| Path | Content |
|---|---|
| `Code/rev_lib.py` | shared library: data split and embedding cache, transceivers (`SharedMuxNest` is the proposal), channel, MAML training, adaptation, evaluation |
| `Code/rev_run.py` | trains and evaluates one configuration, writes `data/<name>_s<seed>.json` and a checkpoint in `data/ckpt/` |
| `Code/dyn_queue.sh` | every run of the load-adaptive design, by stage: `main`, `clamp`, `ablate`, `eval`, `history`, `todma`, `report` |
| `Code/rev_queue.sh` | training runs of the static partitions, of Overlapping masks, and of the per-load retrains (stages `main`, `main_ortho`, `scale`, `general`) |
| `Code/rev_extra.py` | evaluations of stored checkpoints: `reeval`, `noadapt`, `robust`, `probe`, `cost`, `width`, `energy`, and the modes of the earlier revision |
| `Code/todma_redim.py` | ToDMA re-dimensioned to 24, 32, and 48 slots of the same block |
| `Code/rev_dyn_numbers.py` | prints every number of the paper and of its supplementary results into `data/dyn_numbers.md`, from `data/` only |
| `Code/rev_dynamic_fig.py` | regenerates Figs. 2 and 3 and the supplementary figure `fig4_dyn_agg.pdf` from `data/` only |
| `Code/mask_partition.py` | own-energy share of the transmitted energy per mask (Overlapping masks) |
| `Code/rev_cache.py` | builds the embedding caches `data/emb_*.pt` (not stored; AG News, SST-2, BERT-base, DistilBERT) |
| `Code/extra_runs.sh`, `Code/rev_collect.py`, `Code/rev_replot.py`, `Code/letter_mechanism.py`, `Code/rev_diag.py`, `Code/rev_latency.py`, `Code/rev_pilot*.sh` | scripts of the earlier revision, kept with its results. The current paper reuses only the runs of `extra_runs.sh` with the energy `d_s/U` at `k = 4` and 6 |
| `Code/bert_semcom.py`, `Code/cl_experiments.py`, `Code/cl_todma_load.py`, `Code/fig_cl/` | ToDMA reference of the original submission |
| `data/*.json` | one file per run: configuration, per-SNR per-user CosSim at every evaluated load, worst user, probe accuracy, robustness evaluations |
| `data/extra/` | evaluations of stored checkpoints (`rev_extra.py`, `todma_redim.py`, `mask_partition.py`) |
| `data/dyn_numbers.md` | every number quoted in the paper and in its supplementary results |
| `data/ckpt/` | checkpoints of Proposed (one model) for seeds 1 to 3, of Static ODS-1 to Static ODS-6 for seed 1, and of Overlapping masks and the per-load model at `U = 4` of the earlier revision |
| `fig_cl/` | Figs. 1 to 3 of the paper and the supplementary figure `fig4_dyn_agg.pdf` |

## Figure and table map

| Item | Runs | Script |
|---|---|---|
| Fig. 1 | transceiver diagram (TikZ) | none |
| Fig. 2, per-user CosSim versus SNR | `shared_dyns_clamp_s{1,2,3}`, `ortho_U{1..6}_s{1,2,3}` | `rev_dynamic_fig.py` |
| Fig. 3, per-user CosSim versus load | `shared_dyns_clamp_s{1,2,3}`, `ortho_U{1..6}_s{1,2,3}` | `rev_dynamic_fig.py` |
| Supplementary figure `fig4_dyn_agg.pdf`, `S(U)` at 10 dB | the runs of Fig. 3 | `rev_dynamic_fig.py` |
| Sec. IV-B, matched energy | `extra/energy_ortho_U{1..6}_s{1,2,3}` | `rev_extra.py energy` |
| Sec. IV-B, no adaptation and effective width | `extra/noadapt_shared_dyns_s*`, `extra/width_shared_dyn*_s1` | `rev_extra.py noadapt`, `width` |
| Sec. IV-B, ToDMA | `extra/todma_redim.json` | `todma_redim.py` |
| Table II | `shared_dyns_clamp_s*`, `ortho_U1_s*`, `ortho_U6_s*`, `shared_dyn_s*`, `shared_dyns_{uniform,freezetx,joint,K1,K8,slots12,slots48}_s1`, `shared_dyns_s*` (without the hold), `shared_agn_s*`, `shared_abs_s1`, `extra/energy_ortho_U6_s*`, `extra/todma_redim.json` | `rev_dyn_numbers.py` |
| Sec. IV-C, energy division of Overlapping masks | `extra/mask_partition.json` | `mask_partition.py` (needs checkpoints not stored here) |
| Table III | `extra/robust_shared_dyns_s1`, `ortho_U4_s1`, `sst2_shared_dyns_s1`, `sst2_ortho_U4_s1`, `distil_shared_dyns_s1`, `distil_ortho_U4_s1`, `shared_dyns_U8_s1`, `ortho_U8_s1`, `extra/probe_shared_dyns_s1`, `extra/cost_*_s1` | `rev_dyn_numbers.py` |

## Reproduction

Requirements: Python 3.10 or later, PyTorch, `transformers`, `datasets`,
`numpy`, `matplotlib`. Run the scripts from `Code/`.

```bash
cd Code
python rev_cache.py                                          # embedding caches
PY=python EP=150 BS=64 N_PAR=3 bash rev_queue.sh main_ortho  # Static ODS-1 to 6, three seeds
PY=python EXTRA="--per-user-rx" EP=150 BS=64 N_PAR=3 bash rev_queue.sh main   # Overlapping masks, per-load retrains
PY=python EP=150 BS=64 N_PAR=3 bash rev_queue.sh scale       # includes Static ODS-8 (Table III)
PY=python EP=150 BS=64 N_PAR=3 bash rev_queue.sh general     # includes Static ODS-4 and 6 on SST-2, Static ODS-4 with DistilBERT
PY=python bash dyn_queue.sh main                             # Proposed (one model), three seeds
PY=python bash dyn_queue.sh clamp
PY=python bash dyn_queue.sh ablate                           # also eval, history, todma
python rev_extra.py energy ../data/ckpt/ortho_U4_s{1,2,3}.pt ../data/ckpt/ortho_U6_s{1,2,3}.pt   # energy d_s/U at k = 4 and 6
PY=python bash dyn_queue.sh report                           # data/dyn_numbers.md and the figures
```

`rev_dyn_numbers.py` and `rev_dynamic_fig.py` read only the stored results, so
the numbers and figures can be regenerated without training. The evaluations of
`dyn_queue.sh eval` need the checkpoints of all seeds, which the training runs
write to `data/ckpt/`.

Training uses 150 epochs of 100 steps with 64 frames per task, four SNR tasks
per meta-batch drawn from 0 to 25 dB, an inner rate of 5e-4, and Adam at 1e-3.
Every point is evaluated on 2,000 held-out frames, after one inner step on 256
training frames unless stated otherwise. Training seeds are 1, 2, and 3, and
the evaluation seed is 123. The runs of the load-adaptive design and of
`shared_abs` used an NVIDIA RTX PRO 5000 Blackwell (CUDA, torch 2.11.0,
transformers 5.22.1). The static partitions used an NVIDIA RTX A4500 (CUDA,
torch 2.11, seeds 1 and 2) and an Apple M5 Max (MPS, torch 2.5.1, seed 3), and
Overlapping masks (`shared_agn_s*`) and the per-load retrains used the Apple
M5 Max. GPU training is not bit-exact, so a rerun reproduces the reported
values within the seed spread of the paper.

## Earlier versions

The code and results of the original submission are kept at the tag
`original-submission`, and those of the earlier revision with overlapping
masks at the tag `parity-revision-20261021`.

## Citation

The letter is under review. Until it appears, please cite this repository and
the earlier shared-embedding paper:

> K.-H. Lee, H.-H. Choi, and J.-R. Lee, "Code and data for load-adaptive
> semantic multiplexing," 2026. [Online]. Available:
> https://github.com/KiHoLee/load-adaptive-semantic-multiplexing

> K.-H. Lee, H.-H. Choi, and J.-R. Lee, "Transformer-based shared embedding
> for multiple access in semantic communications," *IEEE J. Sel. Areas
> Commun.*, vol. 44, pp. 2622-2637, 2026.
