# Project overview

The baseline evaluation and temporal analysis are complete. Six proposed swaps
remain pending visual approval. They have not altered any reported baseline score.

## Findings overview

1. **Establish correspondence first.** All 3,802 FRAMES images were matched exactly
   to decoded video images. The video has 3,814 frames; frames 3–14 have no FRAMES
   counterpart. Explicit correspondence replaces assumptions about offsets.
2. **Screen the reference.** Task 2 released 14,199 bird/frame observations and
   withheld 1,009 (1,006 uncertain, 3 explicitly excluded). These are individual
   bird/frame decisions, not exclusion of the whole video. Categorical reference
   masks use nearest-neighbour upscaling. Screening does not certify every GT mask
   as biologically perfect, and the old conversion's origin is not fully established.
3. **Evaluate a frozen run.** Reviewed annotation and propagation produced the
   baseline. Task 4 excludes its 24 prompt observations, leaving 14,175. Mean
   GT-present IoU is 0.7415; 2,169 observations (15.3%) are errors under the declared
   presence-aware definition. Report this as descriptive single-video evidence.
4. **The current scorer needs redesign.** B has the strongest ranking among the
   tested scores on the same nonempty observations, but ROC AUC is only 0.5898.
   The existing combined threshold flags zero observations. A mostly flags
   emptiness, including legitimate absence; C flags many correctly moving birds.
   High confidence is not proof of a correct mask or identity.
5. **Separate identity errors from mask shape.** Six GT-assisted proposals concern
   identities 38/75 in block 4, spanning 150 frames. Review previews demonstrate
   a possible ownership exchange; approval and corrections remain pending.
6. **Show when failures happen.** Temporal plots show different patterns by bird,
   block and propagation direction. The middle is not consistently worst in this
   one-interior-annotation design. Error is higher far from annotation in four of
   six blocks, but distance alone is not a validated failure detector.

## What the numbers mean

- One bird/frame observation means one identity evaluated in one image. Four
  birds in a single image can contribute four observations.
- IoU measures overlap with the reference, from zero to one. It requires GT.
- The plotted segmentation error is 1 − IoU, not a count of failures.
- The 15.3% rate includes visible-bird IoU below 0.5 and absent-bird false masks.
  Among GT-present observations alone, 12.1% have IoU below 0.5.
- B averages model confidence over predicted foreground pixels. It is not the
  probability that a mask is accurate and is unavailable for an empty mask.
- GT-free scores are inputs available during deployment. GT is used here to
  test whether those inputs actually distinguish good and bad predictions.

## Status and limits

Task 2 reference partition: complete within its documented scope.
Task 4 frozen scorer evaluation: complete; scorer not deployment-ready.
Task 5 proposal prototype/analysis: complete; human approval and corrections pending.
Task 6 descriptive temporal analysis: complete.

The baseline is reportable with its limitations. Numerical/provenance checks
passed; this is not a claim of zero possible bugs or perfect GT. One video,
GT-assisted prompt selection, correlated frames and reference exclusions prevent
claims of independently validated performance on new recordings. Proposed
GT-assisted swaps must not be presented as improvements already achieved by the
GT-free scorer.

## Suggested reading order

Read the [scoring findings](tasks/04-scoring/FINDINGS.md) first to understand the
denominators, then the [temporal findings](tasks/06-temporal/FINDINGS.md) and
three figures. Finally, inspect the six proposed identity exchanges and their
limitations. No additional broad manual GT audit is required for this baseline.

Before correction, resolve approval or further review of the proposed swaps and agree
the next scoring experiment: distinguish absence from missed segmentation,
reconsider fixed-position C, and validate any new combination on held-out data.
Task 6 helps locate failure episodes and design that experiment; it does not
itself provide a deployed automatic flagger.

## Evidence to open

- [Frozen evaluation](tasks/04-scoring/FINDINGS.md)
- [Swap findings](tasks/05-identity-review/FINDINGS.md) and [visual review](../results/swap-review-20260915T154510599128Z/index.html)
- [Temporal findings](tasks/06-temporal/FINDINGS.md)
- [Per-block figure](../results/temporal-20260916T101412571082Z/per_block_error.png)
- [Within-block position](../results/temporal-20260916T101412571082Z/block_position.png)
- [Distance and direction](../results/temporal-20260916T101412571082Z/distance_from_annotation.png)
