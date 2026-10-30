# Supplementary feature validation

This contribution adds `step8_feature_validation.ipynb` without modifying Steps 1–7.
It addresses the lecturer's question about whether the short click history adds predictive
value, and tests a small extension using the historical query marker.

## Running

1. Run Steps 1–4b to produce `processed/events.parquet` and `processed/features.parquet`.
2. Open this branch's notebook in Colab. Install the listed dependencies if necessary.
3. Run all cells. The default is three seeds, five feature configurations and 300 trees each.
   For a preliminary run, use `SEEDS = [42]`; label that result as a single-seed run.
4. Review train/validation metrics, paired intervals, click-coverage segments, and runtime.
5. Preserve notebook outputs and the timestamped output folder under `processed/step8_validation/`.
   Discuss the validation results with the group before incorporating features into the final model.

No test split is loaded and no existing feature table or prediction file is overwritten.
Outputs are diagnostic CSVs and manifests; the notebook does not save or change the final model.
The repository's prior test-based feature audit means the existing test set is not a fresh holdout.

## Scope

The five variants use the same data, seed and tree budget: baseline, without numeric click
features, without item popularity and its slate-rank derivative, without queryless share,
and baseline plus candidate click counts split by search-linked/queryless at 1 and 7 days.
The new counts are historical inputs only. Queryless means non-search, not verified homepage exposure.

Click information remains indirectly in semantic profiles and query/category priors. A negative
drop-group delta supports the selected numeric feature group; it does not measure all click information.
Popularity ablation similarly leaves category-level priors in place. Fixed-budget retraining controls
compute differences but is not a fully tuned optimum for each feature set.

A 30-day post-purchase click tendency is omitted because clicks are observed only for about 7 days.
Estimating it would require an eligible, fully observed follow-up period inside that window and
enough purchases with support. Treat the history windows as observed coverage, not proof of complete logging.

## Review and contribution record

Before merging, the contributor should explain the four new counts, denominators of the metrics,
the meaning of each ablation, and how the retention pattern affected the design. Record actual
findings, including null results. The contribution is the implementation and validation study;
do not claim a model improvement until the full-data run establishes it.

Add this AI-assisted interaction to the required appendix, including the proposal, the rejected
30-day click-window assumption, checks performed, and revisions made after reviewing real outputs.
This file documents assistance transparently and does not substitute for the group's evaluation.

Local automated checks use synthetic examples for window boundaries, cross-session isolation,
candidate-order invariance, hand-calculated NDCG, purchase denominator handling, and split contracts.
Run from the repository root:

```sh
python -m unittest discover -s tests -p 'test_step8*.py' -v
```

Full-data results are pending a Colab run; no model performance is fabricated here.
