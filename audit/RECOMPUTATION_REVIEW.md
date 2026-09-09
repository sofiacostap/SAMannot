# Review of Lek4 recomputation — 9 September 2026

Evidence: run `recompute-20260909T145908Z`, uploaded as commit `3778e78`. Numerical review uses the uploaded full CSVs; visual review includes alignment images at 750, 2251, 3000 and 3750, plus representative annotation/final-frame overlays. Original production outputs were not changed.

## What can be said at the meeting

The historical IoU CSV reproduces the reported mean 0.46323 and error rate 44.97%. Those figures are numerically reproducible, but they use per-frame identity rematching and a selected population. They are not yet certified as individual-tracking accuracy.

The new fixed-identity recomputation completed and produced mean IoU 0.32999 and error rate 60.71% on 14,389 GT-present observations after excluding annotation/calibration frames. These are provisional calculations under the assumption that equal numeric video/FRAMES indices correspond, and under a proposed identity mapping. They are not evidence that SAM2 became worse: the masks were unchanged and the evaluation definition/population changed.

The next unresolved issue is source-image timing and identity at later annotation frames. Video and FRAMES share an exact image at index 0 (maximum pixel difference zero), but later same-index images visibly show different moving-bird poses/positions while the enclosure is broadly stationary. This makes temporal mismatch a concrete hypothesis to test. It does not establish an offset, prove that all differences are temporal, or rule out decoding/other image differences. The earlier claim that the discrepancy was definitively spatial and not temporal was not supported by the pixel-difference statistic alone.

## Numerical comparison

| Evaluation | Evaluated pairs | Mean IoU | Error rate at IoU <0.5 |
|---|---:|---:|---:|
| Historical CSV | 13,627 | 0.46323 | 44.97% |
| New fixed mapping, including calibration frames | 14,410 | 0.33018 | 60.68% |
| New fixed mapping, excluding calibration frames | 14,389 | 0.32999 | 60.71% |

The new table records 1,064 empty predictions for GT-present birds. It separately records GT absence and missing GT files. There are 3,802 GT images, 3,802 FRAMES images, 3,813 SAM PNGs and 3,814 reported video frames. SAM frames 3802–3812 have no GT counterpart by filename. No GT filename lacks a SAM PNG. Frame 3813 is absent from SAM export, consistent with the boundary bug.

In 4,770 historical rows, the selected GT identity differs from the new global proposal. This is a count of mapping disagreement, not a proven count of identity swaps. On the 13,209 frame/palette keys evaluated in both tables, mean IoU changes from 0.46401 to 0.36020. This still combines changed target mapping and GT extraction policy; it is not a controlled single-factor ablation.

All 15,300 historical quality rows have `logit_score=0.1`. Signal B is therefore constant in that run. Its failure to discriminate is confirmed, not merely inferred from samples. This does not establish that every possible confidence signal is useless.

## Annotation metadata and identity concern

Block size is confirmed from the session as 750. Six common point-prompt frames are 0, 750, 1500, 2251, 3000 and 3750. Five use local frame 0; block 3 uses local frame 1. Therefore the available backward segment is only one frame in block 3, not a substantial backward interval in each block.

Session palette names: 1=darkest, 2=brownish, 3=orange, 4=spotted. The mask-derived global proposal is 1→75, 2→113, 3→38, 4→14, which differs from the earlier written name-to-GT association. Do not silently rename the birds to force agreement.

The first three annotation frames support that proposal (mean IoUs 0.774, 0.561 and 0.757). Later annotation agreement is poor: approximately 0.115 at 2251, 0.239 at 3000 and zero at 3750 under the global mapping. The per-frame assignment at 2251/3000 is ambiguous; at 3750 an alternative assignment fits better but is weak. Visually, the proposed label1/GT75 pair at 3750 highlights different physical birds.

Possible explanations include changed label identity across manual blocks, prompt/mask errors, GT identity issues, source-frame mismatch or historical pipeline effects. The evidence does not yet isolate them. A GPU rerun alone will not resolve incorrect annotation identity or GT correspondence.

My first automatic mapping screen used aggregate margins, which passed despite these weak later frames. This is why outputs remain provisional. The global proposal must not be treated as confirmed merely because that screen passed. Further review must consider each annotation block.

## Quality-score interpretation

On the common A/C-available noncalibration population of 14,328 observations, equal weights with threshold 0.35 yield precision 0.9829, recall 0.1920 and F1 0.3213. These numbers are conditional on provisional error labels. They suggest a selective setting with low recall, not a validated recommendation.

The best exploratory F1 is approximately 0.7546 at A=0.1, C=0.9 and threshold 0.95, flagging 14,307/14,328 observations (99.85%). This is effectively an almost-always-flag strategy, not a useful quality-control improvement. The higher F1 must not be presented as a breakthrough. Search was performed on the same dataset, without held-out validation.

## Next diagnostic

Run `audit/check_alignment.py` on Lumen against this result folder. It searches video indices within ±24 of the seven sampled FRAMES indices using image differences, writes all candidate scores and comparison images, and reports exact pixel matches where they exist. It does not change masks, GT, prompts or evaluation results, and does not automatically apply a temporal shift. Its new output is a sibling folder inside `audit/results/`.

After reviewing those results, establish the frame correspondence before changing IoU joins or moving annotation indices. Then inspect identity consistency across annotation blocks. Full-video propagation should wait for these two checks.
