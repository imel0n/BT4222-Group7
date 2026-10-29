# Purchase-sequence affinity: local sample construction report

Run date: 2026-10-29 (Singapore). Feature construction and correctness validation only; no ranking evaluation or model selection.

## Implementation and repository integration

The new Step 4e notebook follows Step 4b: hypothesis and derivation, candidate-aligned `NEW` feature columns, coverage/distributions, verified temporary-file replacement, and takeaways. All calculations, TXT loaders, experiment orchestration and the 14 correctness checks are defined directly in notebook cells, matching the repository’s existing notebook design. There are no custom Python modules, terminal runner or external test files. Raw files and the main `processed/features.parquet` are unchanged.

Inputs are the original tab-separated `user_behavior_data.txt` (history product/action/time lists, candidate product IDs and labels for alignment only) and `product_meta_data.txt` (`wid`, `cate_id_2`). Their headers were checked by the loaders. Metadata is streamed in bounded batches; only needed product IDs are retained. Conflicting metadata for needed IDs is rejected. IDs remain integers, and unknown categories never match one another.

`user` is the original zero-based behavior-record index; `(user, candidate_pos)` is the candidate occurrence key. Repeated candidate products remain separate rows. Position is an alignment key, never a predictor or scoring tie-break. Candidate labels do not enter association fitting or transformation. No persistent customer identifier is available.

## Exact definition

- Level-2 category associations use historical ORD events only.
- Category-specific episodes have a fixed, inclusive 30-minute window anchored at the first purchase and represented by the last included purchase.
- Destinations must be in a different category, 1 hour–7 days later inclusive. Each destination is counted once per source window. Source age must be at least 8 days, preserving complete follow-up plus the prototype’s 1-day buffer. Empty eligible windows count toward denominators.
- `p_B = n_B/N`; `p_B_given_A = (n_AB + 20*p_B)/(n_A + 20)`; `lift = p_B_given_A/p_B`.
- Score only observed pairs with at least 50 source episodes, at least 10 distinct supporting training records, and lift > 1.
- Use the most recent actual order per source category within 30 days inclusive. Choose maximum `exp(-age_days/7) * log(lift)`. Exact float64 score ties choose the smallest integer source category.
- Output float32 score, selected log1p(record support), selected source age, plus int8 availability. Missing relationships produce 0, 0, missing, 0. Candidate-metadata availability is a separate diagnostic.

There are no intentional changes to the handoff formula/default parameters. Relative to the teammate’s feature, this deliberately changes category level, episode/window construction, smoothing, support rules, recency scoring and fitting isolation. Associations remain observational; real anonymized IDs are not assigned product meanings.

## Sampling, splits and fitting isolation

Uniform sample: 20,000 of 173,831 original records without replacement, seed 42. Complete histories and candidate slates are preserved. Split policy: Step 2 seed=42, 70/15/15 draw across all original record IDs. No main feature table existed for this run. The split draw was made across the full raw-file record count before subsetting; the 20,000-row sample did not get a fresh 70/15/15 split.

Training records use three balanced random folds, seed 43. Each training row is transformed with associations fitted on the other two folds. Validation/test use the full training-only table and may use their own pre-query purchase histories as session context. No validation/test record contributes to shared associations. The original prediction cutoff is represented by relative event ages and the unchanged one-day buffer; no absolute timestamp source was introduced.

| Association fit | Eligible source windows | Observed pairs | Supported positive pairs |
|---|---:|---:|---:|
| fold_0 | 240,332 | 47,381 | 4,660 |
| fold_1 | 246,434 | 48,545 | 4,725 |
| fold_2 | 243,320 | 48,459 | 4,618 |
| all_train | 365,043 | 55,830 | 6,115 |

## Measured diagnostics

- Selected historical orders: **785,868**.
- Candidate rows: **1,781,978**.
- Unique known level-2 categories among retained metadata: **574**.
- Needed product IDs: **1,766,452**; matched metadata IDs: **1,704,046**. These counts include IDs with unusable categories; the coverage table below measures known categories.
- Duplicate metadata rows among needed IDs: **0**.
- Full build elapsed time: **27.12 seconds**, including raw-file SHA-256 scans, loading, fitting, transformation, candidate saving and diagnostics. Final manifest hashing/write happens after this timer.
- Measured process peak RSS: **0.891 GiB**, including the notebook/IPython runtime. This is a process high-water mark, not a prediction for other machines.

| Split | Sessions | Candidate rows | Order category coverage | Candidate category coverage | Candidate score available | Session has a score | Session has varying scores |
|---|---:|---:|---:|---:|---:|---:|---:|
| train | 14,019 | 1,249,132 | 91.14% | 99.54% | 39.15% | 50.82% | 43.73% |
| val | 3,018 | 272,207 | 91.28% | 99.59% | 39.77% | 50.86% | 43.31% |
| test | 2,963 | 260,639 | 90.87% | 99.54% | 40.56% | 51.33% | 43.98% |

These denominators include every complete selected slate, even those with no positive candidate labels. “Varying scores” means at least two distinct score values within the slate. Availability alone does not establish ranking discrimination. The gap between historical-order and candidate metadata coverage is retained as a limitation.

Full per-split count/mean/std/min/quartiles/max and missing-value counts for all four outputs are saved in `diagnostics.json` and displayed in the executed notebook. Missing source ages represent unavailable relationships; they were not imputed to zero.

## Validation performed

**14 unittest tests pass.** They cover age reconstruction including the final query gap, invalid gaps, fixed-anchor and equal-time episodes, inclusive 30-minute/1-hour/7-day/8-day/30-day boundaries, directionality and self-category exclusion, empty window denominators, repeated destination deduplication, distinct-record support, smoothing and support thresholds, zero-background handling, score/selected support/age, deterministic source ties, no orders and unknown categories, candidate-order invariance, lossless large IDs, metadata conflicts, existing split reuse and candidate-order mismatch rejection.

A raw-TXT integration fixture exercises nonzero supported scores, candidate duplicates, split reproduction and fold exclusion counts. It changes all current candidate outcomes and verifies identical feature outputs. Runtime assertions also check occurrence-key uniqueness, candidate counts, finite nonnegative scores, unavailable-value semantics and selected-source ages in range.

The self-contained notebook completed a real 20,000-record run, including all 14 embedded correctness checks. All code cells executed successfully and their outputs were saved. Notebook execution used an in-process IPython session because this sandbox prevents Jupyter from binding local kernel sockets; a normal separate Jupyter kernel launch was not verified here. All 20 code cells passed execution; notebook schema validation and Python syntax checks passed. The saved candidate, membership, association and source-window Parquet files are byte-for-byte identical to the earlier modular implementation.

Validation environment: Python 3.13.2, NumPy 2.5.4, pandas 3.1.0, PyArrow 26.0.0. The existing `.venv-features` interpreter from the handoff project was reused without modifying it; the README documents notebook dependencies and setup.

## Saved artifacts and reproducibility

Outputs are under `Raw Data/processed/purchase_sequence_affinity_sample/` (ignored by Git):

- `candidate_features.parquet`: occurrence keys, original labels/splits, association-fold marker, category availability and the four new features.
- `sample_membership.parquet`: selected original IDs, splits and training fold membership.
- `associations_all_train.parquet` and `associations_fold_0/1/2.parquet`: observed pair probabilities, background probabilities, lift, counts, explicit support and positive-lift flags.
- `source_windows_*.parquet`: every source-category denominator, including categories without destination pairs.
- `diagnostics.json`: measured coverage, distributions, counts, runtime and process memory.
- `manifest.json`: completion marker with parameters, category definition, split/fold policy, predictor exclusions, environment versions and SHA-256 fingerprints of raw inputs, notebook code cells and artifacts. Notebook-code hashing excludes saved outputs.

Every build recomputes intermediates. Existing outputs are replaced only within the experiment directory; no stale cache is silently reused. The notebook’s `REBUILD=False` path checks input/notebook-code/artifact fingerprints and configuration/environment before inspection. Do not run concurrent builds in the same output directory.

## Interpretation and deferred work

This run establishes that the feature can be constructed locally with the specified semantics. It does **not** establish a ranking improvement. Category associations may reflect popularity, frequent shoppers, seasonality, overlapping windows or missing metadata. Within-category candidates cannot be distinguished by these features alone. Cross-fitted training tables see less data than the full training table, so coverage differs across splits.

No parameters were adjusted to reproduce the handoff’s earlier association or coverage counts. The prior 20,000-record experiment used different sampling/split provenance; its results are context, not acceptance targets.

Next milestone: compare the existing LightGBM baseline against baseline plus these four columns on identical candidates/splits. Keep `cat3_cross_lift_max` in both versions initially. Use validation data for choices, and only then report test results, uncertainty and covered-session/warm/cold breakdowns. Model training, feature ranking against random, importance/audit and main-table integration remain deferred.
