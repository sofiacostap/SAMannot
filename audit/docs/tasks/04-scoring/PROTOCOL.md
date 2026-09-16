# Task 4: consolidated evaluation of GT-free quality signals

Status: **COMPLETE — evaluation checked 15 September 2026.** The current scorer
is not ready for reliable deployment. See [findings](FINDINGS.md) and
[verification record](../../../TASK4_CLOSEOUT.json). The released
result is `results/task4-20260915T084226849510Z`.
No new SAM2 propagation or biological annotation is required.

## Frozen inputs and separation from GT

- Predictions: `results/propagation-20260911T180552787557Z`, independent binary
  masks under `masks_by_gt_id/{14,38,75,113}/<video_frame>.png`.
- B: that run's `foreground_confidence.csv`, already computed during GPU propagation.
- Reference: `results/task2-reference-20260915T080605348526Z`, released by
  `TASK2_CLOSEOUT.json`. Only `trusted_screened` bird/frame observations qualify.
- Fixed identity mapping throughout. No per-frame best-match reassignment.

The evaluator validates the release checksums and inventories. It first writes
`signals.csv` using only predictions, saved B and annotation-frame indices. GT
pixels and GT exclusion decisions do not affect these signal calculations.
Then it reads the released reference, checks normalized mask hashes and areas,
and joins predictions using the manifest's video and GT indices. Prediction
hashes are recorded and rechecked between the two phases. Scores also exist for
frames with excluded or absent GT; those frames receive no evaluation labels.

## Signals under test

| Signal | Exact meaning | Limitation |
|---|---|---|
| A | Existing area-ratio/centroid-jump continuity rule: area <50 or >2% of image ->0; otherwise minimum of clipped area-change and centroid-change scores | Stable mistakes and gradual drift can pass |
| B | Saved mean sigmoid over positive foreground logits; no further sigmoid/clipping | Not a calibrated probability of correct segmentation; absent for empty masks |
| C | Spatial IoU with the predicted mask at the annotation frame | Legitimate movement reduces overlap; it is not independently established as a drift detector |
| Combined | Existing `0.3*A+0.4*B+0.3*C`; missing B uses `0.5*A+0.5*C` | Preserves existing limitations, not a newly fitted model |
| Empty baseline | Quality 0 for exactly empty prediction, 1 otherwise | Cannot distinguish absence from tracking loss without other evidence |

A is calculated along the propagation direction, comparing each mask with the
neighbour closer to its block's prompt. Later branch uses t-1; earlier branch
uses t+1. Reset at each block and branch. The earlier branch is seeded by the
saved prompt prediction; its own prompt image was not separately exported.
No comparison crosses block boundaries. This order is explicit in
`continuity_previous_frame`; it is not chronological continuity across a whole
video. The prompt appears once only.

C is unavailable when current or prompt area is below 50 pixels. For faithfully
testing the existing combination, `C_legacy` is 1 in these cases. The combination
uses this sentinel; standalone C reports missingness instead of pretending it is
evidence of quality. In particular, empty predictions with missing B produce a
combined score of 0.5 and may escape the existing combined flag threshold.
This is a behavior to measure, not silently correct while evaluating.

## Evaluation labels and cohorts

Primary evaluation excludes annotation frames to avoid scoring the directly
prompted examples. GT-assisted annotation still limits generalization claims.

- GT present: compute fixed-identity pixel IoU. Error50 means IoU <0.5;
  Error75 repeats analysis with <0.75 as a sensitivity check. An empty prediction
  has IoU 0 and is labelled `missing_visible_bird`.
- GT absent: any positive prediction is a false positive. Both empty is
  `expected_empty`, with IoU left undefined, not rewarded as IoU=1. The count
  of absent-bird false positives at least 50 pixels is also recorded.
- Held GT and missing GT are never assumed correct or counted as absences.

Report `all_presence_aware`, `gt_present`, `gt_absent`, and `common_available`
cohorts. The last uses exactly the same observations for A/B/C/combined, excluding
their missing scores. This prevents B from appearing superior merely because
empty masks are missing from its dataset. Full-cohort coverage and empty-mask
outcomes are also necessary to judge practical usefulness.

ROC AUC and average precision measure ranking of **error risk = 1-quality**.
Average precision groups tied scores; undefined one-class metrics remain blank.
Spearman correlates quality with IoU on GT-present observations only. Report
precision, recall, confusion counts and alert rate at frozen quality thresholds:
A <0.35, B <0.95, C <0.1, combined <0.35, empty baseline <0.5. C and combined
thresholds come from the existing scorer. A and B thresholds are descriptive
starting points, not calibrated deployment choices. A fixed threshold sweep
is provided; no threshold is selected by optimizing this evaluation set.

## Outputs and interpretation

One entry point: `evaluate_task4.py`, CPU-only NumPy/Pillow. Matplotlib creates
an SVG comparison when available; missing plotting support does not invalidate
the numerical evaluation. No predictor/GPU packages are imported.

The new `results/task4-<timestamp>` folder contains:

- `signals.csv`: all GT-free scores, neighbour/annotation indices and masks' hashes.
- `evaluation.csv`: reference status, eligible labels, IoU and error outcomes.
- `signal_metrics.csv`: ranking, correlation, confusion and coverage by cohort.
- `threshold_sweep.csv`: fixed descriptive operating points, no tuned choice.
- `outcome_alerts.csv`: warnings and missing scores separately for expected
  absence, missing visible birds, segmentation mistakes and acceptable masks.
- `by_block_bird.csv`: error and presence summaries by bird/block.
- `prediction_hashes.csv`, `verification.json`: input provenance, coverage,
  failures, counts and headline results.
- `report.md`, and `signal_comparison.svg` when Matplotlib is installed.

This is one video, not independent deployment validation. Consecutive frames are
correlated. Do not interpret counts as independent trials or report spurious IID
confidence intervals. GT reference screening may bias retained examples toward
easier cases. No signal, including B, should be presented as a percentage
probability of segmentation accuracy without a later held-out calibration study.

Task 4 is complete after the full Lumen result is checked and signal strengths,
failures, coverage, and implications are summarized. These checks do not produce
a trained presence detector or implement identity-swap correction.

## Lumen commands

```bash
cd /media/hbvcai/HBVCSSD/samannot/SAMannot/SAMannot-audit
conda activate samannot
git pull --ff-only origin audit/lek4-validation &&
python -m unittest discover -s audit -p test_evaluate_task4.py &&
python audit/evaluate_task4.py
```

After a completed result:

```bash
git add audit/results/task4-*/
git commit -m "Add consolidated Task 4 quality-signal evaluation"
git push origin audit/lek4-validation
```

If the commit already exists after an authentication failure, retry only
`git push origin audit/lek4-validation`.
