# Data dictionary

All ID lists preserve their specified order. P001…P010 identify PSW titles;
A001…A085 identify AI Hub titles; M001…M109 identify Manga109 volumes. No original
title-name mapping is included. IDs are stable across files for a dataset, but
query indices are local to the particular scoring run and candidate title.
AI Hub 41-title and 85-title records originate from distinct stored runs; identical
title IDs do not imply identical query-index-to-image mappings between those runs.

## Verification NPZ files

- `scores`: float64, observations × 4. Columns: absolute, CCN, AS-norm top-2,
  AS-norm all. CCN subtracts the top-two rival mean. AS-norm uses the corresponding
  rival standard deviation (`ddof=0`) plus 1e-6.
- `label`: 1 for a positive query, 0 for a negative query for this candidate.
- `group`: 1 positive, 2 hard negative, 3 easy negative.
- `title_index`: zero-based index into that file's `titles` list in
  `verification_catalogs.json`; identifies the candidate being verified.
- `query_index`: zero-based record index within the candidate's source run.
- `seed`: sampling seed; repeated PSW observations use 0–29.
- `frame_index`: zero-based frame index chosen within the original record.
  These are numerical indices only, not file names.

`verification_catalogs.json` lists record counts, positive counts and whether a
title enters per-title metrics. All 109 Manga109 volumes remain represented in
the catalog metadata; the zero-positive volume has no evaluated rows here.

## Open-set NPZ files

- `scores`: float64, observations × 8, ordered as:
  maximum title similarity; LSE at T=1, 0.1, 0.05, 0.02; contrast with one rival;
  contrast with two rivals (CCN); contrast with all rivals.
- `owner`: anonymous source-title ID of the positive source record.
- `prediction`: anonymous registered-title ID of the common argmax predictor.
- `role`: 0 registered evaluation, 1 calibration unknown, 2 evaluation unknown.
- `split`: 0 for PSW/AI Hub; 0–19 for Manga109 series partitions.
- `fold`: PSW leave-one-title-out registry identifier; 0 otherwise.
- `query_index`, `seed`, `frame_index`: the original record index and the selected
  frame sampling identifiers. PSW seeds 0–4 are pooled as separate single-frame
  observations. The RNG is reset for each record in this open-set protocol.

Correct attribution for registered rows is `prediction == owner`. Unknown rows
remain unknown even if a gate accepts them. Registered wrong-title acceptance is
not included in unknown FAR. Denominators: correct/wrong acceptance use **all
registered observations**; unknown FAR uses **evaluation unknown observations**.

## Figure 5 scores

`figure5_scores.npz` uses the same fields as the open-set NPZ files, except:

- `scores` has two columns: maximum title similarity, CCN.
- `split=-1` is the 41/44 registry split. Splits 0–29 are prefix-stratified 40/45
  partitions. There is no separate calibration group (`role=1` is absent).
- The original Figure 5 plotting data are in `figure5_archived.json`; detection
  acceptance and correct attribution are separate curves.

## Learned-head JSON files

`rows` has one object per split 0–29. Prefixes `frozen_abs`, `frozen_con`,
`arcface`, `arcface_con`, and `lr` identify the scorer/decision combination.
Suffixes: `auc` = detection AUROC, `oracle` = split-optimal detection F1,
`blind` = F1 at the base-registry threshold, `loss` = oracle minus blind.
`thresholds` contains base-registry thresholds; ArcFace values are seed means.
`axis1` reports base-registry closed-set accuracy and argmax agreement, not
30-split averages. `tau_main` is the logistic-head base-registry threshold.

## Growth NPZ and JSON files

- `growth_verification_orders.npz`: `k` contains 12 registry sizes; `theta` is
  30 × 12, rows corresponding to order IDs 0–29 in `splits.json`.
- `growth_far_orders.npz`: `k` contains sizes 41,45,…,81; `far_absolute` and
  `far_ccn` are 20 × 11; `theta_ccn` gives 20 initially matched gates. Absolute
  uses 0.363 throughout. Rows correspond to order IDs 0–19 in `splits.json`.
- `growth_*_archived.json`: original medians and quartiles, retained for checks.
- `growth_fit_inputs.json`: dataset-specific K and threshold input points used
  by the final joint fit. These rounded fit points have different provenance
  from the separately computed growth trajectory.

## Geometry and robustness

`geometry.json` is grouped by dataset. Each title has `margin_conmax` (mean over
positive records of the within-record frame mean of own minus strongest rival),
`margin_contop2`, `f1_abs`, `f1_con`, `th_abs`, `th_con`, positive count, `reg_self`,
`unreg`, and `flagged` (`reg_self <= unreg`). F1/threshold values are retained
per-title summaries. The archived rank-scan convention applies to oracle F1.

`robustness.json` preserves cell-level F1/FPR/AUROC and available title/seed
summaries. `*_std` describes across-title population dispersion where indicated;
`*_seedstd_ddof1` describes seed-level sample dispersion. The five-seed clean
control is distinct from the main 30-seed verification result.

## Split manifests

`splits.json` separately defines Table 4 registries, calibration/evaluation roles,
Manga109 series groups, Figure 5 partitions, learned-head partitions and the two
growth order sets. Manga109 volumes from the same series are assigned together.
Table 5 strata use original genre-label sorting and catalog order, whereas
Figure 5 strata use sorted ID prefixes. The resulting exact lists are provided
so readers do not need the original genre labels or title dictionary.
