# Leave-one-driver-out clustering stability

## Setup

- **data_dir**: `/home/zwx/driver_model/following/outputs/following_il_clean_gap04`
- **seed**: 42
- **cluster_dim_weights**: `2,1,1,1,1,1,1,1,1,1,1` (order: headway_median, headway_p25, time_headway_median, time_headway_p25, headway_mean, ego_v_var, acc_var, accel_abs_median, accel_abs_p75, decel_abs_median, decel_abs_p75)
- **Cohort**: T1, T2, T3, T4, T5, T6, T7, T8, T9, T10, T11, T12, T13, T14, T15, T16, T17, T18, T19, T20 drivers (`n=20`)

**Baseline**: run `assign_kmeans_styles` on **all** drivers. **LOO**: exclude one driver, re-cluster the rest; compare each remaining driver's label to the baseline.

## Global summary

- Pairwise comparisons: **380** (each is one remaining driver vs baseline for one excluded driver).
- Label mismatches vs baseline: **119** (31.3% of pairwise rows).

**How to read this**: z-scoring and k-means are fit on whoever is in the cohort. Removing one driver changes everyone else's normalized coordinates and cluster boundaries, so some label flips are expected. A **high** mismatch rate means the three-style assignment is **not stable** under small cohort changes; consider more drivers, softer labels (probabilities), or fixed thresholds instead of global k-means.

## Most influential exclusions (by number of changed labels)

| excluded_driver | n_remaining | n_labels_changed |
|-----------------|------------:|-----------------:|
| T8 | 19 | 15 |
| T7 | 19 | 12 |
| T12 | 19 | 12 |
| T17 | 19 | 12 |
| T18 | 19 | 12 |
| T20 | 19 | 12 |
| T1 | 19 | 10 |
| T3 | 19 | 7 |
| T4 | 19 | 7 |
| T2 | 19 | 6 |

## Per-driver LOO style frequencies

For each driver: across the **19** folds where they remain (each exclusion is some other driver once), tally how often `style_loo` is conservative / neutral / aggressive. The three integers always sum to 19. **style_baseline** is the full-cohort assignment.

| driver_id | style_baseline | n_loo_fold | conservative | neutral | aggressive |
|-----------|----------------|-----------:|---------------:|--------:|-----------:|
| T1 | neutral | 19 | 0 | 10 | 9 |
| T2 | neutral | 19 | 8 | 7 | 4 |
| T3 | aggressive | 19 | 0 | 1 | 18 |
| T4 | aggressive | 19 | 0 | 7 | 12 |
| T5 | aggressive | 19 | 0 | 1 | 18 |
| T6 | aggressive | 19 | 0 | 0 | 19 |
| T7 | neutral | 19 | 4 | 10 | 5 |
| T8 | aggressive | 19 | 0 | 0 | 19 |
| T9 | conservative | 19 | 19 | 0 | 0 |
| T10 | aggressive | 19 | 1 | 10 | 8 |
| T11 | aggressive | 19 | 0 | 7 | 12 |
| T12 | aggressive | 19 | 0 | 5 | 14 |
| T13 | aggressive | 19 | 0 | 10 | 9 |
| T14 | aggressive | 19 | 0 | 10 | 9 |
| T15 | neutral | 19 | 5 | 10 | 4 |
| T16 | conservative | 19 | 19 | 0 | 0 |
| T17 | neutral | 19 | 6 | 10 | 3 |
| T18 | neutral | 19 | 6 | 10 | 3 |
| T19 | aggressive | 19 | 0 | 1 | 18 |
| T20 | neutral | 19 | 5 | 10 | 4 |

## Per-driver sensitivity

Open `loo_stability_by_driver.csv`. **mismatch_rate** = among LOO folds where this driver is still present (excluding each *other* driver once), the fraction of folds where `style_loo != style_baseline`.

## Output files

| file | content |
|------|---------|
| `loo_pairwise.csv` | per excluded driver, each remaining driver's baseline vs LOO label |
| `loo_impact_by_excluded.csv` | how many drivers changed label when a given driver is removed |
| `loo_stability_by_driver.csv` | per driver mismatch count and rate |
| `loo_style_counts_by_driver.csv` | per driver: LOO histogram over the three styles (+ baseline + percentages) |
