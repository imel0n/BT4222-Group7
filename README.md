# BT4222 Group 7 — Personalised Search Re-ranking on JDSearch

Ordinal ranking of the candidate slate shown for each user's test query (0 = none, 1 = click, 2 = cart, 3 = purchase), with the goal of raising session conversion.

## Pipeline

Run the notebooks in order. Each one reads the previous step's outputs from `processed/` next to the raw data.

| Notebook | What it does | Outputs |
|---|---|---|
| `step1_preprocess_and_data_understanding.ipynb` | Parses the raw file into flat tables; data findings (per-type history retention windows, in-window funnel, query-field checks, roles of each field) | `events`, `candidates`, `users`, `queries` `.parquet` |
| `step2_feature_engineering.ipynb` | Window-aware user / item / category / query features; user-level train/val/test split; leak-free LightGBM LambdaRank benchmark with ablation | `features.parquet` |
| `step3_text_relevance.ipynb` | Query→product TF-IDF text features over the anonymised term ids (name cosine, match fraction/IDF coverage, brand+category cosine); benchmark NDCG@10 0.408 → 0.414, cold-slate NDCG +23% relative | `features.parquet` (4 text columns added) |
| `step4_semantic_embeddings.ipynb` | 64-d LSA latent space (SVD of the TF-IDF name matrix) for 11.6M products; query projection + semantic user history profiles; NDCG@10 0.414 → 0.415, cold slates 0.100 → 0.105. Verdict: GBDT feature engineering has hit diminishing returns — embeddings should feed a neural ranker next | `features.parquet` (3 semantic columns added) |
| `step4b_domain_features.ipynb` | Repurchase-cycle timing, within-slate percentile ranks, cross-category order-transition lift; NDCG@10 0.4155 → 0.4159 but `slate_pct_pop` takes 48% of model gain and cold slates improve 0.104 → 0.114 — evidence the remaining headroom is representational (slate-aware model needed, not more scalar features) | `features.parquet` (6 columns added, 55 total) |
| `step4c_seasonality.ipynb` | Analysis only. Dates history events from the shared test day (2022-10-17) and confirms festival order spikes (618 2.7×, Double 11 2.6×). Sessions in promotion-dependent product types convert at 5.8% vs 14.4%, a category trait rather than pre-festival deferral. Seasonal features add nothing to ranking (NDCG@10 0.4144 → 0.4142), so they are **not** carried into the modelling feature set | – (its run wrote 4 columns into `features.parquet`; restore the 50-column file before re-running Step 4) |
| `step4d_feature_audit.ipynb` | Significance and redundancy audit of the 55 features: Spearman clustering plus permutation importance on test slates with Bonferroni-corrected CIs. 27 features significant; 15 drop candidates (brand/category cart/order counts, scalar semantic profiles, rebuy/cross-lift); 32-feature set selected for Step 5 | `feature_audit.csv` |
| `step4e_purchase_sequence_affinity.ipynb` | Independent local feature-only experiment following Step 4b's workflow: level-2 purchase episodes, training-only smoothed associations, three-fold cross-fitting, recency-decayed scores, correctness tests and sample diagnostics. Does not modify the main feature table or train models | Separate `purchase_sequence_affinity_sample/` artifacts |
| `step5_neural_ranker.ipynb` | Neural ranker (L4 GPU): feature MLP + fixed LSA embeddings + DIN history attention over the last 50 events + slate context + ordinal head giving calibrated P(Y≥1/2/3). Ties LightGBM on NDCG@10 (0.4154 vs 0.4159), beats it on cold slates (0.1172 vs 0.1138) and Purchase-HR@10 (0.6756 vs 0.6692) | `lsa64_*.npy` (embedding cache), `step5_results.csv` |
| `JDSearch_Data_Anlaysis.ipynb` | Original exploration from the interim report (slate lengths, position leakage, query terms, category hierarchy) | – |

## Running in Google Colab

1. Put the raw data in your Drive at `MyDrive/JDSearch/Raw Data/` (`user_behavior_data.txt`, `product_meta_data.txt`).
2. Open a notebook from GitHub in Colab (File → Open notebook → GitHub → `imel0n/BT4222-Group7`) and run all cells. The first cell mounts Drive.
3. Processed tables are written to `MyDrive/JDSearch/Raw Data/processed/`.

To run locally instead, set `JD_DATA_DIR` to the folder containing the raw files (default `~/Downloads/JDsearch`).

## Local purchase-sequence feature experiment (Step 4e)

Open `step4e_purchase_sequence_affinity.ipynb` and run its cells from top to bottom,
just like the other feature notebooks. All data loading, feature calculations,
correctness checks, saving and diagnostics are inside the notebook; no custom
Python modules, terminal runner or external test files are needed.

The notebook uses NumPy, pandas, PyArrow and IPython, plus the Python standard
library. Install missing packages in your notebook environment if needed:

```bash
python -m pip install numpy pandas pyarrow jupyterlab ipykernel
```

It starts directly from `user_behavior_data.txt` and `product_meta_data.txt`, with
headers intact. Its local default is this repository's `Raw Data/` directory. Set
`JD_DATA_DIR` or change `DATA_DIR` in the setup cell to use another location. Run
with the repository as the notebook working directory so the notebook's own code
can be fingerprinted for provenance.

The first cells explain and define the parameters, loaders, episode/association
calculations and scoring. The correctness-check cells run 14 synthetic tests.
The build cell then creates a reproducible 20,000-record sample; later cells show
coverage, within-slate variation, distributions and saved artifacts.

```text
Raw Data/
  user_behavior_data.txt
  product_meta_data.txt
  processed/
    purchase_sequence_affinity_sample/
      candidate_features.parquet
      sample_membership.parquet
      associations_all_train.parquet
      associations_fold_0.parquet
      associations_fold_1.parquet
      associations_fold_2.parquet
      source_windows_*.parquet
      diagnostics.json
      manifest.json
```

`REBUILD=True` rebuilds the sample. Set it to `False` to verify and inspect a
completed run. Change `OUT_DIR` for separate experiments; do not build concurrently
in the same directory. Individual outputs are verified before replacement, and
`manifest.json` is written last. Provenance includes raw-file fingerprints and a
hash of notebook code cells, independent of saved execution outputs.

Candidate outputs use `(user, candidate_pos)` as occurrence keys, preserving
repeated products. `user` is Step 1's original record index, not a verified persistent
customer ID. Identifiers, position, labels, splits and fold IDs are not predictors.
The four feature columns are `sequence_score`, `sequence_support_log`,
`sequence_source_age_days`, and `sequence_available`; candidate-category availability
is retained separately as a diagnostic.

Existing main-table splits are reused when present, with candidate alignment
checked. Otherwise the notebook reproduces Step 2's seed-42 split draw over all
original records before sampling. Training features are three-fold out-of-fold;
validation/test use an association table fitted only on training histories.
The main `processed/features.parquet` is not overwritten. Ranking evaluation and
main-table integration are deferred.

Measured results and limitations are in
[`reports/purchase_sequence_affinity_sample.md`](reports/purchase_sequence_affinity_sample.md).

## Key data facts that shape the modelling

- **History retention differs by type**: clicks are kept for the last ~7 days, carts ~90 days, orders ~365 days. Ratios across types are only valid inside a window both types cover. Within 7 days the funnel is normal (≈6.2 clicks per cart, ≈6.6 carts per order).
- **Candidate position leaks the label** (interacted items are listed first), so it is never used as a feature, and slates are shuffled before training and evaluation.
- **History is positive-only**: the candidate slates are the only labelled data with negatives, so splits are made by user over the slates.

## Dataset

Liu, J., Dou, Z., Tang, G., & Xu, S. (2023). *JDsearch: A Personalized Product Search Dataset with Real Queries and Full Interactions.* SIGIR '23. Samples in `JD dataset/`.
