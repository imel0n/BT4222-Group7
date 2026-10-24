# BT4222 Group 7 — Personalised Search Re-ranking on JDSearch

Ordinal ranking of the candidate slate shown for each user's test query (0 = none, 1 = click, 2 = cart, 3 = purchase), with the goal of raising session conversion.

## Pipeline

Run the notebooks in order. Each one reads the previous step's outputs from `processed/` next to the raw data.

| Notebook | What it does | Outputs |
|---|---|---|
| `step1_preprocess_and_data_understanding.ipynb` | Parses the raw file into flat tables; data findings (per-type history retention windows, in-window funnel, query-field checks, roles of each field) | `events`, `candidates`, `users`, `queries` `.parquet` |
| `step2_feature_engineering.ipynb` | Window-aware user / item / category / query features; user-level train/val/test split; leak-free LightGBM LambdaRank benchmark with ablation | `features.parquet` |
| `step3_text_relevance.ipynb` | Query→product TF-IDF text features over the anonymised term ids (name cosine, match fraction/IDF coverage, brand+category cosine); benchmark NDCG@10 0.408 → 0.414, cold-slate NDCG +23% relative | `features.parquet` (4 text columns added) |
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
