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
| `step5_neural_ranker.ipynb` | Neural ranker (L4 GPU): feature MLP + fixed LSA embeddings + DIN history attention over the last 50 events + slate context + ordinal head giving calibrated P(Y≥1/2/3). Ties LightGBM on NDCG@10 (0.4154 vs 0.4159), beats it on cold slates (0.1172 vs 0.1138) and Purchase-HR@10 (0.6756 vs 0.6692) | `lsa64_*.npy` (embedding cache), `step5_results.csv` |
| `step6_tuning_ensemble.ipynb` | Neural tuning with early stopping and 3-seed averaging; purchase-weighted LightGBM; within-slate z-score ensemble; paired-bootstrap significance; session-conversion estimates. **Final model: ensemble, NDCG@10 0.4201** (vs LightGBM 0.4166, p < 0.001; vs neural 0.4185, p < 0.001). Purchased item in top 10: 67.6% (popularity 57%, random 14%) | `step6_results.csv`, `step6_test_predictions.parquet` |
| `JDSearch_Data_Anlaysis.ipynb` | Original exploration from the interim report (slate lengths, position leakage, query terms, category hierarchy) | – |

## Running in Google Colab

1. Put the raw data in your Drive at `MyDrive/JDSearch/Raw Data/` (`user_behavior_data.txt`, `product_meta_data.txt`).
2. Open a notebook from GitHub in Colab (File → Open notebook → GitHub → `imel0n/BT4222-Group7`) and run all cells. The first cell mounts Drive.
3. Processed tables are written to `MyDrive/JDSearch/Raw Data/processed/`.

To run locally instead, set `JD_DATA_DIR` to the folder containing the raw files (default `~/Downloads/JDsearch`).

## Key data facts that shape the modelling

- **History retention differs by type**: clicks are kept for the last ~7 days, carts ~90 days, orders ~365 days. Ratios across types are only valid inside a window both types cover. Within 7 days the funnel is normal (≈6.2 clicks per cart, ≈6.6 carts per order).
- **Candidate position leaks the label** (interacted items are listed first), so it is never used as a feature, and slates are shuffled before training and evaluation.
- **History is positive-only**: the candidate slates are the only labelled data with negatives, so splits are made by user over the slates.

## Dataset

Liu, J., Dou, Z., Tang, G., & Xu, S. (2023). *JDsearch: A Personalized Product Search Dataset with Real Queries and Full Interactions.* SIGIR '23. Samples in `JD dataset/`.
