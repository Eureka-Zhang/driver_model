# Leave-one-driver-out clustering stability

## Setup

- **data_dir**: `following/outputs/following_il_clean_gap04`
- **seed**: 1
- **cluster_dim_weights**: `1,1,1,1,1,1,1,1,1,1,1,1` (order: headway_median, headway_p25, inv_time_headway_p25, inv_time_headway_median, inv_ttc_p50, inv_ttc_p95, ego_v_var, accel_abs_median, accel_abs_p75, decel_abs_median, decel_abs_p75, jerk_abs_p75)
- **Cohort**: T1, T2, T3, T4, T5, T6, T7, T8, T9, T10, T11, T12, T13, T14, T15, T16, T17, T18, T19, T20 drivers (`n=20`)

**Baseline**: run `assign_kmeans_styles` on **all** drivers. **LOO**: exclude one driver, re-cluster the rest; compare each remaining driver's label to the baseline.

## Global summary

- Pairwise comparisons: **380** (each is one remaining driver vs baseline for one excluded driver).
- Label mismatches vs baseline: **83** (21.8% of pairwise rows).

**How to read this**: z-scoring and k-means are fit on whoever is in the cohort. Removing one driver changes everyone else's normalized coordinates and cluster boundaries, so some label flips are expected. A **high** mismatch rate means the three-style assignment is **not stable** under small cohort changes; consider more drivers, softer labels (probabilities), or fixed thresholds instead of global k-means.

## Most influential exclusions (by number of changed labels)

| excluded_driver | n_remaining | n_labels_changed |
|-----------------|------------:|-----------------:|
| T1 | 19 | 12 |
| T2 | 19 | 11 |
| T3 | 19 | 11 |
| T4 | 19 | 6 |
| T6 | 19 | 6 |
| T8 | 19 | 6 |
| T9 | 19 | 6 |
| T5 | 19 | 5 |
| T7 | 19 | 4 |
| T12 | 19 | 3 |

## Per-driver LOO style frequencies

For each driver: across the **19** folds where they remain (each exclusion is some other driver once), tally how often `style_loo` is conservative / neutral / aggressive. The three integers always sum to 19. **style_baseline** is the full-cohort assignment.

| driver_id | style_baseline | n_loo_fold | conservative | neutral | aggressive |
|-----------|----------------|-----------:|---------------:|--------:|-----------:|
| T1 | neutral | 19 | 0 | 16 | 3 |
| T2 | neutral | 19 | 8 | 11 | 0 |
| T3 | neutral | 19 | 2 | 17 | 0 |
| T4 | aggressive | 19 | 0 | 3 | 16 |
| T5 | neutral | 19 | 3 | 16 | 0 |
| T6 | aggressive | 19 | 0 | 0 | 19 |
| T7 | neutral | 19 | 5 | 11 | 3 |
| T8 | neutral | 19 | 3 | 16 | 0 |
| T9 | conservative | 19 | 19 | 0 | 0 |
| T10 | neutral | 19 | 4 | 15 | 0 |
| T11 | neutral | 19 | 0 | 9 | 10 |
| T12 | aggressive | 19 | 0 | 0 | 19 |
| T13 | aggressive | 19 | 0 | 0 | 19 |
| T14 | aggressive | 19 | 0 | 3 | 16 |
| T15 | neutral | 19 | 3 | 16 | 0 |
| T16 | neutral | 19 | 9 | 10 | 0 |
| T17 | aggressive | 19 | 3 | 3 | 13 |
| T18 | neutral | 19 | 9 | 10 | 0 |
| T19 | aggressive | 19 | 0 | 0 | 19 |
| T20 | neutral | 19 | 9 | 10 | 0 |

## Per-driver sensitivity

Open `loo_stability_by_driver.csv`. **mismatch_rate** = among LOO folds where this driver is still present (excluding each *other* driver once), the fraction of folds where `style_loo != style_baseline`.

## Output files

| file | content |
|------|---------|
| `loo_pairwise.csv` | per excluded driver, each remaining driver's baseline vs LOO label |
| `loo_impact_by_excluded.csv` | how many drivers changed label when a given driver is removed |
| `loo_stability_by_driver.csv` | per driver mismatch count and rate |
| `loo_style_counts_by_driver.csv` | per driver: LOO histogram over the three styles (+ baseline + percentages) |
