# Task 2 GT reference status

Current review method: [playback and artifact separation](ARTIFACT_REVIEW.md).
Only irregular frames require a confirmed flag; correct frames need no individual
approval. The full viewing pass is still pending. The per-bird approval tool remains
available as an alternative, with a different acceptance policy.

**Screened reference released; exhaustive visual review pending.** The release
below completed the automatic-screening and localized-exclusion scope on
2026-09-15. The expanded requirement is to inspect every GT image and record
each bird/frame decision using the [visual review workflow](VISUAL_REVIEW.md).
The historical release and its evaluation remain unchanged. They are not a
fully visually validated reference. No SAM2 propagation is needed for review.

## Released reference and verified counts

Use `results/task2-reference-20260915T080605348526Z/evaluation_set.csv` and that
run's `masks/` on Lumen. Evaluate only rows with `status=trusted_screened`.
The mask names use GT image indices; prediction names use video indices. Join
through the manifest's `gt_frame` and `video_frame`, never by filename equality.
The previous grayscale masks and provisional partition are superseded for Task 4.

| Label | Retained | Withheld | Available |
|---|---:|---:|---:|
| 14 | 3,613 | 189 | 3,802 |
| 38 | 3,516 | 286 | 3,802 |
| 75 | 3,504 | 298 | 3,802 |
| 113 | 3,566 | 236 | 3,802 |
| Total | **14,199 (93.4%)** | **1,009 (6.6%)** | **15,208** |

The 1,009 held observations comprise 1,006 uncertain plus 3 explicit exclusions.
All 56 bird/frame decisions around the confirmed problems remain held; they are
included in these totals. At least one bird remains evaluable on 3,800 of 3,802
GT images; only two images have all four bird references withheld. The 12 video
frames without GT are separate missing references, not excluded GT images.

Every bird/block group retains at least 86.4% of available observations. Retained
data include 13,092 GT-present and 1,107 GT-absent observations. This is adequate
coverage to proceed to Task 4, with per-bird/block reporting and explicit missing
reference handling. Temporally adjacent observations are not independent samples.

Local closeout checks independently reconstructed all decisions from metrics,
candidates and the copied review ledger; verified uniqueness/completeness of all
15,208 keys and the frame mapping; checked all human holds and correspondence
failures are withheld; and checked normalized-mask paths and the complete
3,802-image hash inventory. Counts match the Lumen summary. The full image/palette,
photograph fingerprint and normalization roundtrip checks ran on Lumen; the PNGs
remain there. `TASK2_CLOSEOUT.json` records the release and report checksums.

This closes the reference-release scope: a defined usable reference with localized
uncertainty, not repair of every biological mask. Remaining limitations: the
historical conversion and original-transfer checksum are unknown; labels inherit
the verified FRAMES mapping with an interior-consistency screen rather than new
independent timing proof; half-resolution annotations limit boundary precision;
screening is not exhaustive biological certification; and GT-assisted prompts
limit claims about independent novice annotation. Exclusions can bias evaluation
toward easier GT, so Task 4 must report coverage and evaluate only the released
reference subset. No claim of a universally bug-free pipeline is made.


Earlier provisional counts and encoding investigations are retained in [the historical record](../../archive/TASK2_INVESTIGATION.md).
