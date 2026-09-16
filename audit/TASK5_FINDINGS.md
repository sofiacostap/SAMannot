# Task 5 findings — 16 September 2026

The prototype produced six proposed intervals, all involving identities 38 and
75 in block 4. **All corrections remain pending for Friday by user decision.**
The baseline predictions and Task 4 measurements have not changed.

Source: [review page](results/swap-review-20260915T154510599128Z/index.html).

| Event | Video frames, inclusive | Frames |
|---|---|---:|
| SWAP0001 | 3161–3233 | 73 |
| SWAP0002 | 3099–3151 | 53 |
| SWAP0003 | 3087–3095 | 9 |
| SWAP0004 | 3154–3159 | 6 |
| SWAP0005 | 3002–3006 | 5 |
| SWAP0006 | 3235–3238 | 4 |

Together these cover 150 frames, or 300 bird/frame assignments. Across those
assignments the current-identity IoUs are zero; hypothetical exchange gives mean
IoU 0.8486. **This is a counterfactual diagnostic, not an achieved improvement.**
Mask shapes would remain unchanged; only ownership would change.

The assistant inspected all six midpoint previews. They support the exchange
relative to the reference, but do not independently establish biological identity
or approve interval boundaries. There are 30 preview images in total.

Checks confirmed that grouping the 157 candidate pair/frames reproduces the six
intervals and their 150 frames. Seven candidates fail the minimum three-frame
duration. Gaps may reflect screening or thresholds; they do not prove a swap
ended and restarted. Input/event provenance was checked against the frozen run.

This is a **GT-assisted diagnostic**, not a GT-free deployed detector. It cannot
repair missing masks, wall masks, boundary errors or arbitrary multi-bird swaps.
Approval, application, and a separate evaluation of any corrected output remain
pending. The technical proposal stage is finished; the full correction stage is
not complete. No further Lumen run is needed for Friday's discussion.
