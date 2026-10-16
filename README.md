# Multi-target QSAR pipeline for HDAC inhibitors

Multi-task regression of inhibitory potency (pIC50) against the **nine human HDAC
isoforms**, with a focus on Class I (HDAC1, 2, 3, 8). The pipeline predicts all nine
targets from a SMILES string, is evaluated **out-of-distribution** through a
cluster-based split, and ends with activity imputation and a Class I selectivity score.

The model is a **D-MPNN** (Yang et al. 2019) trained as a multi-task regressor with an
uncertainty-weighted loss and a self-supervised pre-trained encoder, compared against a
single-task LightGBM fingerprint baseline and fused with it by a per-isoform Ridge
meta-learner.

## Installation

```bash
pip install -r requirements.txt
```

Install `torch` and `torch-geometric` with the build matching your CUDA version. The
`hdbscan` package is optional: it is only used for the DBCV score when the split is
recomputed from scratch.

## Running

Open `main.ipynb` and run the cells in order. All required data ships with the
repository. The D-MPNN requires a GPU; everything else runs on CPU.

After the run, `figures.ipynb` regenerates the per-isoform figures and the validation
loss curves. All artefacts are written to `outputs/`, one subfolder per stage:

| Folder | Contents |
|---|---|
| `outputs/split/` | UMAP split visualisation, split-difficulty comparison, random-split baseline metrics. |
| `outputs/baseline/` | LightGBM test metrics and out-of-fold predictions. |
| `outputs/dmpnn/` | Per-seed CV history, CV results, OOF predictions, fold weights and task weights (`seed<N>_*`), `seed_summary.csv`, test metrics and parity plot of the selected seed, loss curves. |
| `outputs/stacking/` | Per-isoform metrics and parity plot of the stacked model. |
| `outputs/comparison/` | Per-isoform tables, ranking metrics and final comparison of the three models. |
| `outputs/screening/` | Scored dataset, BitterDB predictions and top-compound figures. |
| `outputs/applicability/` | Error-vs-similarity calibration table and figure, domain-annotated screening library. |
| `outputs/model/` | `inference_bundle.pt`. |

Standalone inference on new molecules:

```bash
python s7_inference.py --bundle outputs/model/inference_bundle.pt --in compounds.csv \
                       --smiles-col smiles --out predictions.csv --device cuda
```

## Data and the reference split

| Path | Contents |
|---|---|
| `data.xlsx` | 7,517 molecules with pIC50 on the nine HDAC isoforms (ChEMBL). |
| `bitter.csv` | 1,041 bitter compounds (BitterDB), external control library. |
| `split/*_set_multitarget.csv` | Reference partition: train 6,063 / val 794 / test 660, 1,242 groups. |
| `pretrained/ssl_encoder_dmpnn.pt` | Pre-trained D-MPNN encoder (11 tensors, `hidden_dim=160`). |

**The split is versioned on purpose.** UMAP + HDBSCAN do not return the same partition
across versions of numpy, scikit-learn and scipy: recomputing it on a different stack
yields 1,199 groups and 755 test molecules instead of 1,242 and 660, and every
downstream number changes. The first cell loads `split/` when present and only
recomputes the split in its absence.

The chemical-group id of each molecule is stored in the `Cluster` column.

## Repository layout

The modules are numbered in the order in which `main.ipynb` uses them; `main.ipynb`
itself holds no function definitions — every function lives in `s0`-`s9` and is just
called from the notebook. Inside each file the functions appear in call order, with the
entry point called by the notebook last.

| File | Role |
|---|---|
| `main.ipynb` | End-to-end pipeline: split, baseline, split comparison, D-MPNN, stacking, per-isoform analysis, inference, selectivity screening. |
| `figures.ipynb` | Regenerates the paper figures and the validation loss curves from the artefacts in `outputs/`. |
| `s0_config.py` | Shared constants: isoforms, target columns, SMILES and group columns, Class I isoforms, underpowered-n threshold. |
| `s1_split.py` | Morgan fingerprints, UMAP + HDBSCAN clustering, best-of-N KS split, distribution plots, split-vs-random UMAP visualization. |
| `s2_metrics.py` | Pooled and per-isoform metrics, parity plots, ranking metrics and EF@10%, the LightGBM/D-MPNN/stacking comparison table. |
| `s3_baseline.py` | LightGBM baseline (`run_lgbm_baseline`): test predictions and out-of-fold predictions. |
| `s4_dmpnn.py` | Graph featurisation, 25 global descriptors, augmentation, and the D-MPNN model. |
| `s5_training.py` | Grouped k-fold training with SWA and early stopping over several seeds (`train_dmpnn`), seed selection on CV, fold-ensemble prediction. |
| `s6_stacking.py` | Per-isoform non-negative Ridge meta-learner on out-of-fold predictions. |
| `s7_inference.py` | Inference bundle export and prediction on new SMILES (also a command-line tool). |
| `s8_screening.py` | Class I selectivity score, BitterDB cleaning, zinc-binding-group detection and drug-likeness, top-compound tables. |
| `s9_applicability.py` | Applicability domain: nearest-neighbour similarity, error-vs-similarity calibration, prediction clipping, screening-library annotation (`run_applicability`). |

## Method summary

- **Split.** Feature-based Morgan fingerprints (radius 3, 2048 bit) → UMAP on the
  Jaccard distance → HDBSCAN. Noise points become singleton groups, so no data is
  discarded and structurally isolated molecules cannot leak. Among candidate group
  splits, the one whose per-isoform pIC50 distributions overlap most closely is
  retained (largest aggregated Kolmogorov–Smirnov p-value). Median nearest-neighbour
  Tanimoto similarity from test to training: **0.63**, against 0.81 for a random split.
- **Baseline.** Nine independent single-task LightGBM regressors on Morgan fingerprints.
- **Graph model.** D-MPNN, `hidden_dim=160`, four message-passing layers, Set2Set
  readout, 25 RDKit global descriptors standardised per fold. Target z-scoring per
  isoform (estimated on the fold training set only) and seed-ensembling are active;
  the fingerprint is deliberately not appended to the global block. The global
  descriptors are concatenated to the graph embedding **after** the Set2Set readout,
  inside the prediction head, so no whole-molecule feature enters message passing.
- **Atom and bond features.** The descriptor list is the standard one: 26 atom features
  (atom type, degree, total H, aromaticity, in-ring, hybridisation, formal charge,
  chirality, membership of 3- to 7-membered rings, total valence, radical electrons) and
  8 bond features (bond type, conjugation, in-ring, stereochemistry). The implementation
  reports `node_in_features=29` and `edge_in_features=10` because `one_hot_encoding`
  appends a trailing "other" slot to each one-hot block — three on atoms (atom type,
  hybridisation, chirality) and two on bonds (bond type, stereochemistry). Since every
  block sums to exactly one, those five columns are linearly dependent on the rest and
  add no representational capacity: their contribution is absorbable into the bias, and
  the bond-type slot is identically zero across this dataset. The encoding is therefore
  informationally equivalent to the 26/8 formulation, only over-parameterised by five
  redundant columns.
- **Training.** AdamW (lr 5e-4, weight decay 0.01), gradient clipping at 1.0, plateau
  scheduler, at most 120 epochs with early stopping at 18, conditional SWA over the
  final quarter, edge dropout 0.10 and node masking 0.05 resampled every epoch.
- **Aggregation.** Fold predictions de-standardised and averaged with weights
  proportional to the inverse validation MSE, then a non-negative per-isoform Ridge
  meta-learner fitted on out-of-fold predictions.
- **Applicability domain.** Nearest-neighbour Tanimoto similarity to the training set on
  the same Morgan fingerprints used by the split, with the threshold calibrated on
  measured error rather than assumed. Predictions are clipped to the observed pIC50
  range and the clipped rows are flagged.

## Results

On the held-out out-of-distribution test set (published run):

| Model | R² (pooled) | MSE | Spearman | macro-R² |
|---|---|---|---|---|
| LightGBM (baseline) | 0.372 | 0.797 | 0.596 | 0.231 |
| D-MPNN | 0.494 | 0.643 | 0.696 | 0.394 |
| Stacking (D-MPNN + LightGBM) | 0.503 | 0.631 | 0.691 | 0.403 |

The D-MPNN is the headline result: it beats a strong single-task baseline by +0.12
pooled R² and rescues isoforms the baseline cannot predict at all (HDAC9: −0.078 →
+0.465). Stacking adds a marginal refinement.

**Why the split matters.** With the same baseline, the same molecules and the same
split sizes, changing only the partitioning criterion:

| | random split | cluster-based split |
|---|---|---|
| median NN similarity | ~0.81 | ~0.63 |
| R² (pooled, LightGBM) | **0.653** | **0.372** |
| MSE | 0.468 | 0.797 |
| Spearman | 0.811 | 0.596 |

A random split inflates R² by **+0.281**. This is reproduced by the split-comparison cell of `main.ipynb` (cell 5).

## Is the model more than a chemotype detector?

58% of the dataset carries a hydroxamic acid, so any model trained on it will rank
hydroxamates highly. Finding known zinc-binding groups among the top-ranked compounds is
therefore **not evidence of anything**: the same result is produced by a one-line
substructure match, so the test has no power to separate a good model from a trivial one.
Two tests that do have that power are reported instead.

**The chemotype null model.** A LightGBM given only the 17 binary "bears this ZBG"
features, on the same split and the same metrics, is an explicit formalisation of the
hypothesis that the model knows nothing but chemotype membership:

| model | features | R² (test) | Spearman |
|---|---|---|---|
| hydroxamate only | 1 | 0.054 | 0.225 |
| all 17 ZBG classes | 17 | **0.082** | 0.254 |
| LightGBM Morgan FP | 2048 | 0.372 | 0.596 |
| D-MPNN | graph | **0.494** | 0.696 |

Chemotype membership explains 8% of the variance. The relevant margin for the D-MPNN is
the 0.41 above this null, not the 0.49 above the global mean.

**Within-chemotype stratification.** Restricting to a single chemotype removes the
between-chemotype variance by construction, so any surviving performance must come from
the cap group, the linker and the substitution pattern. On out-of-fold predictions:

| subset | n | target SD | R² | Spearman | RMSE |
|---|---|---|---|---|---|
| all | 6857 | 1.147 | 0.388 | 0.630 | 0.897 |
| hydroxamates only | 4075 | 1.141 | **0.400** | **0.631** | 0.884 |
| aminoanilides only | 1208 | 0.917 | 0.449 | 0.673 | 0.681 |
| no recognised ZBG | 1136 | 1.233 | 0.267 | 0.549 | 1.055 |

Performance inside the hydroxamate subset is indistinguishable from performance on the
whole population. Restriction to a subset normally deflates R² by shrinking the target
variance, but here the subset SD is 1.141 against 1.147, so the two values are directly
comparable, and Spearman — which is insensitive to that effect — agrees. Had R² collapsed
to near zero within the subset, that would have falsified the model; it does not. This is
also the regime in which the model is actually used, since a medicinal-chemistry project
ranks analogues within a committed chemotype rather than across chemotypes.

## Applicability domain

The domain threshold is **measured, not assumed**. Out-of-fold and test predictions are
binned by nearest-neighbour Tanimoto similarity to the training set and the error is
computed in each bin (`outputs/applicability/calibration.csv` and `calibration.png`):

| NN-Tanimoto | n (observations) | RMSE | R² | Spearman |
|---|---|---|---|---|
| 0.15–0.25 | 91 | 1.427 | −0.043 | 0.236 |
| 0.25–0.35 | 713 | 1.228 | −0.001 | 0.306 |
| 0.35–0.45 | 2013 | 1.127 | 0.208 | 0.470 |
| 0.45–0.55 | 3135 | 0.865 | 0.414 | 0.637 |
| 0.55–0.65 | 3465 | 0.859 | 0.422 | 0.655 |
| 0.65–0.80 | 4344 | 0.769 | 0.515 | 0.724 |
| 0.80–1.00 | 1556 | 0.732 | 0.516 | 0.724 |

Below 0.45 the RMSE exceeds one pIC50 unit — a tenfold error in IC50 — and below 0.35
R² is indistinguishable from zero: the ranking is uninformative there. **The threshold is
set at 0.45.**

The two calibration sources are complementary and both are needed. The test set measures
error on a genuinely held-out split, but its similarity never drops below 0.27 so it
cannot populate the low-similarity bins; the out-of-fold predictions, scored against
neighbours outside their own cluster (which `GroupKFold` never splits across folds), reach
down to 0.14 and are what fixes the threshold.

**Applied to the BitterDB control library:**

| | value |
|---|---|
| compounds | 1041 |
| median NN-Tanimoto to training | **0.253** |
| inside the domain (≥ 0.45) | **63 (6.1%)** |
| predictions clipped to the observed range | 6 |
| predicted mean Class I > 6 (1 µM) | 4 |
| **in domain *and* predicted > 6** | **0** |

94% of the library sits outside the domain, and no compound is both inside the domain and
predicted active — the most potent in-domain prediction is 5.71. The unfiltered top-15 by
selectivity score has a median NN-Tanimoto of 0.26, roughly a third of the training
similarity the test set enjoys, so that ranking is not interpretable as a hit list.

Two artefacts of the unfiltered pipeline that the domain filter removes: predictions fell
as low as **pIC50 −9.11**, which is not physically meaningful, and the selectivity score
`mean(Class I) − max(others)` structurally penalises pan-HDAC chemotypes, since a
hydroxamate raises both terms and nets out near zero.

## Reproducibility notes

- The split, the LightGBM baseline and the split comparison are **deterministic** and
  reproduce exactly from the shipped artefacts.
- **D-MPNN training is not bit-reproducible**: the scatter operation in message passing
  uses `atomicAdd` on CUDA, and early stopping and SWA turn small numerical differences
  into different stopping epochs. Re-running the published configuration lands roughly
  0.015 pooled R² below the table above. Report **mean ± standard deviation over
  several seeds**: raise `N_SEEDS` in the configuration cell (cell 6); the training cell
  (cell 7) evaluates each seed separately, writes `outputs/dmpnn/seed_summary.csv`,
  and passes the seed with the best cross-validation R² to the following cells.
- The published loss-curve figure was produced with early stopping disabled for visual
  clarity only; a normal run yields truncated curves.
- Feature dimensions are `node_in_features=29` and `edge_in_features=10`, and the global
  block holds 25 RDKit descriptors. These are the dimensions encoded in the pre-trained
  checkpoint (`W_i` is `160 × 39`, `W_o` is `160 × 189`), so the trailing one-hot slots
  described under *Atom and bond features* cannot be dropped without re-deriving the
  checkpoint.
- **Zinc-binding-group detection** (`s8_screening.py`) assigns each molecule a single
  ZBG: `zbg_hit` returns the *first* match in the priority order of `ZBG_SMARTS`, so a
  molecule bearing several groups is counted once, under the highest-priority one. Use
  `zbg_all_hits` for per-chemotype prevalence, where a molecule may count in several
  groups. The two give materially different tables (e.g. catechol: 235 independent
  matches against 35 first-hits, the rest being masked by a hydroxamic acid elsewhere in
  the molecule), so any reported percentage must state which convention it uses.
