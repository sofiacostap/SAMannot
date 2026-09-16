# Task 4 findings — 15 September 2026

**Task 4 evaluation is COMPLETE. The current quality scorer is not ready for
reliable deployment.** This result evaluates the frozen implementation; it does
not fit a replacement scorer or change thresholds after seeing the results.

Source: `results/task4-20260915T084226849510Z`. Inputs use the closed Task 2
categorical reference and the completed reviewed propagation. No SAM2 rerun.

## What was evaluated

14,199 released GT bird/frame observations minus 24 annotation observations
leaves **14,175 primary observations**. All 1,009 held GT observations and the
48 prediction observations lacking GT remain outside evaluation.

| Outcome | Observations |
|---|---:|
| GT present, IoU at least 0.5 | 11,485 |
| GT present, nonempty prediction with IoU below 0.5 | 1,139 |
| GT present, prediction empty | 444 |
| GT absent, prediction nonempty | 586 |
| GT absent, prediction empty | 521 |

There are **2,169 errors (15.3%)** under the declared presence-aware definition.
For the 13,068 GT-present observations, 1,583 (12.1%) have IoU below 0.5.
Mean GT-present IoU is **0.7415**, median **0.8361**. These are overlap measures,
not percentages of correctly identified birds. Both-empty cases are not assigned
IoU=1. All 586 absent-bird false positives also exceed the 50-pixel size threshold.

## Equal-coverage comparison

B and C are unavailable for 965 eligible empty predictions. Compare ranking
on the **same 13,210 nonempty observations** before comparing full-cohort results.
This cohort contains 1,725 errors (13.06%). AUC 0.5 means chance-level ranking;
no independent-frame statistical significance is claimed.

| Signal | ROC AUC | Average precision | IoU/quality Spearman |
|---|---:|---:|---:|
| A continuity | 0.5051 | 0.1437 | -0.1409 |
| B foreground confidence | 0.5898 | 0.2083 | 0.4410 |
| C prompt-position overlap | 0.5142 | 0.1330 | -0.0813 |
| Existing combination | 0.5357 | 0.1595 | -0.0759 |

B carries the strongest ranking information among these scores on this common
cohort, but discrimination at the IoU<0.5 failure criterion is weak. At the
stricter IoU<0.75 sensitivity criterion its AUC is 0.6711; that supports retaining
B as an input to future work, not calling it calibrated accuracy.

The combination's full-cohort AUC of 0.6076 is not directly comparable to B's
0.5898 because the former includes empty masks. On the same nonempty observations
the combination is weaker than B alone. Earlier claims that continuity is the
only useful signal must be revised in light of this corrected evaluation.

## Behavior at the frozen flag thresholds

| Signal | Errors flagged | Non-errors flagged | Missing primary scores |
|---|---:|---:|---:|
| A <0.35 | 463 | 557 | 0 |
| B <0.95 | 16 | 17 | 965 |
| C <0.1 | 1,183 | 7,985 | 965 |
| Combined <0.35 | 0 | 0 | 0 |
| Empty-mask baseline | 444 | 521 | 0 |

- **A mainly detects emptiness.** Of its 463 error detections, 444 are empty
  predictions for visible GT birds. It adds only 19 nonempty-error detections
  beyond the empty baseline and flags 36 acceptable nonempty masks. It also
  flags all 521 expected-empty cases. A does not establish presence.
- **B is frequently confidently wrong.** Among 13,177 observations with B>=0.95,
  1,709 (13.0%) are errors. Its score is the mean sigmoid over predicted foreground
  pixels, not the probability the segmentation is correct. Missing B on empty
  masks must not be interpreted as a correct prediction.
- **C generates many false alerts.** It flags 7,985 of 11,485 acceptable masks
  (69.5%). A fixed-position prompt overlap can fall when a correctly segmented
  bird moves. This result does not support using C as a general drift alarm.
- **The existing combination generates zero alerts.** The minimum evaluated
  combined score is 0.358345, above the strict flag threshold of 0.35.

The last behavior is explained by the formula: for nonempty predictions,
`combined = 0.3*A + 0.4*B + 0.3*C_legacy`. All components are nonnegative, so
when B>=0.875, `0.4*B>=0.35`; the strict `<0.35` flag cannot trigger even if
A=C=0. Empty masks have A=0, missing B, and legacy C=1, yielding fallback 0.5.
The evaluator correctly preserved this rule. The issue is the rule's operating
range and missing-score handling, not an absent result or evaluator failure.
Changing one threshold may change recall, but does not resolve weak ranking,
unavailable presence evidence or C's false alarms. No replacement threshold
has been selected using this evaluation data.

## Coverage across blocks

| Block | Evaluated bird/frames | Errors |
|---|---:|---:|
| 0 | 2,623 | 295 |
| 1 | 2,793 | 144 |
| 2 | 2,799 | 623 |
| 3 | 2,838 | 383 |
| 4 | 2,874 | 724 |
| 5 | 248 | 0 |

The last block is short. These totals do not prove temporal deterioration or
the within-block hypothesis. The completed temporal analysis is documented in Task 6. Bird-level breakdowns are in `by_block_bird.csv`.

## Verification and scope

All recorded input checksums match. Local checks verified all 15,256 prediction
keys, saved B values, combined formulas, continuity neighbours and block indices,
reference decisions, annotation holdout, IoU arithmetic and error definitions.
Recomputing from the uploaded observation table reproduced every metric,
threshold table, outcome alert and block/bird summary to 1e-12 floating-point
tolerance. Pixel processing and mask hashes were checked on Lumen; original PNGs
are not hosted in the GitHub reports. `TASK4_CLOSEOUT.json` pins this result.

Limitations remain: one video, GT-assisted annotation, temporally correlated
observations, screened GT that may select easier cases, half-resolution reference
boundaries, and the inherited correspondence/provenance qualifications recorded
in Task 2. These findings are descriptive, not independent deployment validation.

## Current follow-up status

Task 5 proposals and Task 6 baseline temporal analysis are complete. Identity
corrections remain pending visual approval. Baseline predictions are immutable.
Any approved correction needs a separate overlay-aware evaluation and updated
temporal scores. Scorer redesign and validation on held-out data remain future work.
