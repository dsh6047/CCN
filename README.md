# CCN numerical evaluation data

Version 1, prepared 16 September 2026.

This is a minimal **post-scoring numerical reproducibility package** for the
Catalog contrast normalization study prepared for submission to Multimedia Systems. The study
is not yet published. Table and figure numbers follow manuscript and Online
Resource 1. 
The package contains anonymized scalar evaluation scores, labels, experimental
partitions, retained run summaries, and a small analysis script. It does not
contain images, image crops, embeddings, model weights, original image filenames,
episode names, local filesystem paths, or a title-name lookup dictionary.

## Quick start

Use Python 3.9 or later with NumPy, SciPy and scikit-learn installed:

```sh
python -m pip install -r requirements.txt
python reproduce.py
```

The script works offline after dependencies are installed. It writes
`computed_results.json` containing per-title metrics, calibrated gate metrics,
transfer results, per-split curve areas, learned-head means, geometry statistics,
the logarithmic fit, and numerical comparisons with archived results. Source data
in `data/` are not modified. No GPU, image data or pretrained encoder is needed.

The dependency versions in `requirements.txt` describe the environment used to
validate **this numerical package**, not a claim about the original training
environment. Original images and learned-head training are outside this package.

## What supports each result

- **Table 2:** `verification_psw_sscd.npz` and `verification_psw_dinov2.npz`.
  Query scores support F1, FPR and AUROC calculations for four scoring methods.
- **Table 3:** the three SSCD verification files, with `table3_archived.json`
  preserving the original numerical reference. The 109-volume Manga109 registry
  contributes 108 evaluated volumes because one volume has no positive queries.
- **Sect. 4.3 and Fig. 4:** `geometry.json` contains retained per-title thresholds,
  margins, oracle F1, the two quantities in Eq. (6), and the flag. The diagnostics
  are retrospective, not independent tests on new episodes.
- **Table 4 and Table S1:** `openset_psw.npz`, `openset_aihub.npz` and
  `openset_manga109.npz`, with `table4_archived.json`. All eight gates are retained,
  including the four log-sum-exp temperatures. Manga109 split 0 is the displayed
  table; splits 0–19 are included for the repeated series partitions.
- **Fig. 5:** `figure5_scores.npz` and `figure5_archived.json`. The latter retains
  plotted means, interquartile ranges and median areas. Areas are computed from
  exact curves before interpolation onto a common FAR grid.
- **Fig. 6:** the same PSW and AI Hub open-set scores; `figure6_archived.json`
  preserves within-catalog and cross-catalog operating points for target FAR
  0.10, 0.20 and 0.30. Full curves can be regenerated from the scores.
- **Fig. 7:** `growth_fit_inputs.json`, `growth_verification_orders.npz` and
  `growth_far_orders.npz`, plus the two archived growth summaries. Order-level
  values were reconstructed from stored scores during package preparation and
  checked against the retained medians and quartiles. No new inference was run.
- **Table 5:** `table5_sscd.json`, `table5_dinov3.json` and
  `table5_logistic.json`. These are original retained **per-split metrics**;
  arithmetic means can be recalculated without training.
- **Table S2:** `robustness.json` retains transformation-cell summaries,
  per-title F1 and available per-seed summaries. It does not contain transformed
  images or per-query transformed scores.
- **Partitions:** `splits.json` gives the exact ordered anonymous title IDs for
  the separate protocols. Figure 5 and Table 5 partitions must not be interchanged.

Figures 1–3 contain illustrations or source images and have no image assets in
this numerical release. Source images are not needed to run the included checks.

## Scope and precision

1. **Numerical reproduction starts from saved scores.** The package permits
   checking reported calculations; it does not reconstruct anchors, rerun image
   encoding, retrain learned heads, or prove the original image-level split.
2. Stored source precision is preserved without additional rounding. Some source
   similarity records were already rounded. The joint-fit input points have
   three decimal places because those are the points actually used by the final
   plotting script. They are not represented as full-precision raw measurements.
3. Per-title verification resets `default_rng(seed)` once per title and consumes
   it sequentially over records. PSW uses seeds 0–29; single-image records in
   AI Hub and Manga109 require only one deterministic evaluation. Repeated rows
   from the same query are dependent, not additional independent observations.
4. The original verification oracle scans sorted positions, including positions
   within tied scores. `archived_rank_oracle` preserves that implementation.
   It is not a tie-grouped threshold sweep. F1 at zero and FPR at an indicated
   threshold are computed from actual `score >= threshold` decisions. This
   distinction matters when interpreting oracle values or recomputing them with
   a different library implementation.
5. Table 2's displayed FPR agrees with **strict `score > oracle threshold`** at
   three decimals. The script reports both strict and inclusive FPR; these differ
   when negative scores equal the threshold. Table 3 fixed-zero F1 uses `>=`.
   Table 4 and Fig. 6 use **strict `score > threshold`**. Growth FAR uses
   **`score >= threshold`**, as in its original experiment. A nonnegative maximum
   CCN score is a mathematical property, not the open-set acceptance condition.
6. ArcFace split metrics are means over three training seeds. Individual
   seed-specific transfer thresholds and query predictions were not retained in
   the saved comparison results. Mean thresholds are descriptive only and must
   not be substituted for the unavailable individual thresholds. Table 5 is
   reproducible here at the split-summary aggregation level.
7. Verification and FAR growth use distinct orders, grids and query pools.
   `growth_verification_orders.npz` has 30 orders; `growth_far_orders.npz` has 20.
   The fixed FAR thresholds were initially matched at registry size 41. As
   unknown titles are registered, the unknown query pool changes. Size 85 has
   no remaining unknown queries and is absent from the FAR data.
8. The joint fit excludes K=2. Old fit coefficients embedded in historical files
   are deliberately not included. The included input points yield approximately
   `theta = 0.0698 - 0.0195 ln(K)`, R² = 0.919, zero crossing K ≈ 36.
9. Archived values are retained independently of any manuscript display rounding.
   Use the numerical files, not rounded table cells, for further calculations.

## Source data access

- **PSW:** copyrighted source images are not redistributed. This release does
  not offer image access on request and does not include download locations.
- **AI Hub:** obtain the Comic and Webtoon Generation dataset (71717) directly
  from [AI Hub](https://aihub.or.kr/aihubdata/data/view.do?dataSetSn=71717), subject
  to the provider's access and reuse conditions.
- **Manga109:** request access from the [Manga109 project](http://www.manga109.org/en/),
  subject to the provider's access and reuse conditions.

This package conveys no rights to the underlying images. The anonymous IDs are
internal numerical identifiers and do not provide a mapping to provider files.

## Reading the files

JSON files are UTF-8 text. NPZ files are compressed NumPy arrays, **not embeddings**;
they contain scalar scores and experiment identifiers. No pickle loading is needed.

```python
import numpy as np
with np.load('data/openset_aihub.npz', allow_pickle=False) as data:
    print(data.files)
    print(data['scores'].shape)
```

See `DATA_DICTIONARY.md` for array columns and identifier meanings.
`SHA256SUMS.txt` records file hashes for integrity checking.
