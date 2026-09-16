# Task 6 temporal analysis — 16 September 2026

**COMPLETE as a descriptive analysis of the frozen baseline.** This reads Task 4's
existing observation table; no propagation, pixel evaluation, threshold fitting
or identity correction was repeated.

Outputs: [per-block error](results/temporal-20260916T101412571082Z/per_block_error.png),
[position within blocks](results/temporal-20260916T101412571082Z/block_position.png),
[distance from annotation](results/temporal-20260916T101412571082Z/distance_from_annotation.png).
CSV tables and verification.json are in the same directory; SVGs permit export.

## What the plots establish

There is one annotation frame per block: 449, 1124, 1874, 2624, 3324 and 3781.
Propagation proceeds toward earlier and later frames from that annotation.
With this arrangement, the most distant frames are generally near the block
edges, not in its middle. The hypothesis that the middle is consistently worst
is not supported: its mean error is not the highest of the three thirds in any
of the five full blocks. Block 5 is only 64 frames and should be treated separately.

Mean segmentation error (1 − IoU), for GT-present observations:

| Block | Nearest quarter of each branch | Farthest quarter of each branch |
|---|---:|---:|
| 0 | 0.225 | 0.275 |
| 1 | 0.146 | 0.241 |
| 2 | 0.146 | 0.414 |
| 3 | 0.197 | 0.173 |
| 4 | 0.235 | 0.322 |
| 5 | 0.181 | 0.143 |

The pooled mean is worse far from annotation in four of six blocks. This is not
a universal rule: the direction-specific plot shows distinct earlier/later
patterns. Block 2 deteriorates strongly in its later portion; block 4 has large
early identity-related errors, including the pending pair-swap proposals.
The proposals do not explain every error in block 4.

## Reading and checking the figures

In the per-block chart, colours identify birds; faint dots show individual
observations, lines show 25-frame bin means, dashed lines mark annotations, and
orange shading marks pending proposals. Each panel has its own frame range.
IoU curves use only GT-present observations: absent-bird false predictions are
included in the accompanying presence-aware tables, not these curves. Both-empty
observations are never presented as IoU=1.

The frozen Task 4 checksums match. Totals across blocks, thirds, position bins
and frame bins each reproduce 14,175 eligible observations. Block totals also
reproduce 2,169 presence-aware errors and 13,068 GT-present observations. All
three figures were visually checked for layout and labels.

These are descriptive findings on one video, with correlated frames and unequal
retained counts. Distance is associated with some failures but is not established
as their cause or validated as an alert rule. GT-assisted annotations and screened
reference selection limit generalisation. No corrections were applied.
