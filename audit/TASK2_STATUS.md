# Task 2 GT reference status — closed 15 September 2026

**COMPLETE WITH DOCUMENTED EXCLUSIONS.** No further broad manual review or SAM2
propagation is required for this Task 2 closeout.

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

This closes the user's requested scope: a defined usable reference with localized
uncertainty, not repair of every biological mask. Remaining limitations: the
historical conversion and original-transfer checksum are unknown; labels inherit
the verified FRAMES mapping with an interior-consistency screen rather than new
independent timing proof; half-resolution annotations limit boundary precision;
screening is not exhaustive biological certification; and GT-assisted prompts
limit claims about independent novice annotation. Exclusions can bias evaluation
toward easier GT, so Task 4 must report coverage and evaluate only the released
reference subset. No claim of a universally bug-free pipeline is made.

## Historical investigation notes (superseded status and provisional counts)

The sections below preserve the investigation history. Their earlier open status,
provisional counts and proposed commands do not override the release above.

### Final reference preparation (15 September)

The half-resolution check `gt-encoding-20260915T074933632634Z` found an inspected
1344 x 760 palette PNG containing exactly IDs 0–4. No recipe reproduced all
13 current grayscale samples exactly, so no full reproduction check ran.
The supervisor's supplied message establishes a dataset delivery location on
the cluster and subsequently Lumen, but includes no checksums. It does not
establish corruption by the user or certify this particular mask derivative.

The authorized closeout route is now **preserve categorical source labels and
exclude uncertainties**, rather than indefinitely reconstruct the historical
grayscale recipe. `prepare_task2_reference.py` performs the full pass on Lumen:

- Require the inspected palette and label set on every usable source mask. Map
  source 1->38, 2->75, 3->113, 4->14 explicitly; repeat each source pixel into a
  2x2 square. Do not convert colours to grayscale or blend bird identities.
- Verify the photograph fingerprint and shape against the established image
  correspondence. Test source/reference interior agreement in both directions
  against unchanged historical GT, holding disagreement per bird/frame. This
  is a consistency screen, not independent proof of biological identity/timing.
- Recompute fragment, area, movement and possible exchange screening on the
  categorical derivatives. Preserve the user's ledger and all source pixels;
  questionable components are held out, never erased or relabelled.
- Write new masks, hashes, bird/frame decisions and coverage by bird/block into
  a new `audit/results/task2-reference-<timestamp>` directory. Missing/unusable
  source frames hold their four references locally; they do not remove a video.
- `trusted_screened` means passes the documented operational reference policy,
  not that every biological contour was manually certified. A result marked
  `failed` must never be consumed by evaluation.

Nine local tests passed for categorical normalization, label/palette/shape
rejection, identity/support disagreement, paired exchange exclusions, and the
previous review/provenance tests. Full OpenCV screening and dataset coverage
must be checked from the Lumen result. No SAM2 propagation is requested.

Task 2 may be closed with explicit exclusions once this run succeeds and its
coverage is assessed; reconstructing the historical conversion is no longer
required for this source-preserving route. No new broad manual-review exercise
is planned. Task 4 must use the new result's `evaluation_set.csv`, normalized
mask paths and hashes, not the old provisional partition or grayscale masks.
Report exclusions and per-bird coverage; half-resolution boundaries and
unverified transfer provenance remain limitations.

### 15 September source discovery update

The uploaded report `gt-encoding-20260915T074030534574Z/verification.json`
inspected 13 grayscale masks and found a sibling RGB `upscaled_masks_v2` set.
No tested conversion reproduced any sampled target exactly. The user believes
v2 was generated during project work; it is not an independent original reference.
No evaluation input or exclusion count has changed.

A subsequent directory listing found the sibling clip folder
`cut_Lek4_right_2021_604_00-00-00_00-02-32_50percent/masks`, outside the first
check's clip-level search. This is a candidate precursor, not yet certified.
Pass it to the existing `check_gt_encoding.py --original ".../masks"` option.
The checker tests native indices, palette/grayscale conversions and resizing;
any all-sample exact recipe is then checked against every target mask. Use
`masks`, not `overlayed_masks`: image overlays are not categorical references.

## Persistent decisions and counts

The versioned `task2_gt_review.json` records the user's confirmations. Frame numbers
below refer to the video; GT75/38/113 are bird label values, not image numbers.

- Video 663 / GT image 651: GT75 incorrectly covers two birds. Exclude that
  bird/frame. Hold GT75 and potentially missing GT38 over video 650–676 (the
  associated substantial-component interval). Only frame 663 is user-confirmed;
  the surrounding interval and GT38 hold are precautionary.
- Video 1751 / GT image 1739: GT113 contaminates GT38. Exclude both bird/frame
  references. Do not paint the purple region into a different bird.
- The combined review hold is **56 bird/frame observations across 28 images**:
  3 explicitly excluded observations and 53 uncertain observations. These are
  included, not additional to, the conservative totals below. Other birds on
  these images remain eligible for screening. The entire video is not excluded.

The reproducible provisional partition is in
`results/task2-reference-policy-20260912/evaluation_set.csv` and `summary.json`.
It contains exactly 3,802 GT images × four bird labels = **15,208 observations**.

| GT label | Screened, pending encoding | Uncertain or excluded | Total |
|---|---:|---:|---:|
| 14 | 223 | 3,579 | 3,802 |
| 38 | 3,675 | 127 | 3,802 |
| 75 | 3,593 | 209 | 3,802 |
| 113 | 3,719 | 83 | 3,802 |
| All | **11,210** | **3,998** | **15,208** |

Of the 3,998 held observations, 3 are explicitly excluded and 3,995 uncertain.
The 11,210 screened observations contain 10,439 present and 771 absent GT cases.
Video frames 3–14 have no matching GT: 48 bird/frame combinations lack a reference,
separately from the 15,208 available GT observations. They cannot receive GT-based
evaluation labels, but their predictions are not removed.

This is a **conservative provisional inventory, not a released trusted set**.
Every non-encoding candidate is held; both sides of an abrupt area/centroid change
are held because the heuristic cannot identify which side is wrong. Small
disconnected components are not necessarily biological errors. In particular,
3,579 of the 3,837 small-fragment findings concern GT14. Blindly retaining these
exclusions would bias evaluation severely against that bird. The available count
is substantial, but four-bird representativeness is not yet defensible. These
holds are automatic policy decisions, not requests for human review.

## Encoding investigation and safe interpretation

`gt-audit-20260912T121456077403Z` found 115,279 unexpected-value records across
**all 3,802 images**. Their widespread occurrence supports a systematic encoding
hypothesis; it does not prove interpolation. Current and historical captured
SAMannot source consumes `upscaled_masks`; the inspected source does not contain
the script that generated that folder. Original GT PNGs and precursor masks are
not present in this GitHub evidence checkout.

Do not round grayscale values to the nearest bird ID: these IDs are categories,
not numerical measurements. Do not interpret unknown pixels as background.
If using such an encoded mask, unknown pixels must be void for every bird and
excluded from both IoU intersection and union. **Voiding unknown values alone is
insufficient to certify GT**: interpolation between labels can generate another
valid label value, and palette-to-grayscale conversion can change IDs.

Preferred resolution is to recover the original categorical mask representation,
establish its identity mapping, and resize with nearest-neighbor if required.
Any normalized derivatives must be separate, hash-traceable outputs. Biological
ambiguities remain excluded; predictions must not be used to repair reference GT.

`check_gt_encoding.py` performs one bounded read-only Lumen check:

1. Reads native PNG mode, values and palette at 13 spread-out GT indices, including
   the two confirmed cases and annotation locations.
2. Inventories nearby precursor-mask folders within this GT clip (depth 3).
3. Searches workspace code (depth 4, excluding datasets/dependencies/results) for
   upscaling evidence and records numbered snippets and hashes.
4. Tests explicit native/palette-gray/RGB resizing and conversion recipes. If one
   reproduces every sample exactly, checks that recipe over all 3,802 target files
   and records source hashes. Full exact reproduction establishes that transform
   explains these files, not which historical command was actually executed.
5. Rejects changed sampled GT hashes and, during full verification, any changed
   target hash. If no recipe matches, reports uncertainty rather than claiming
   interpolation is proven. No GPU, SAM2, mask edits or manual-review queue.

## Verification completed locally

Five tests passed: real 15,208-row partition and localized decisions; duplicate
and correspondence rejection; unexpected labels do not become independent bird
exclusions; palette indices remain distinguishable from grayscale; synthetic
resize reproduction and changed-GT rejection. Tests exercised Pillow recipes;
OpenCV recipes require Lumen's installed OpenCV. The ledger, partition and input
hashes are persistent. No propagation was rerun.

## Next step and work plan

Run the command below from the designated Lumen checkout, then upload the new
report. This requires no new biological annotation or visual review.

```bash
cd /media/hbvcai/HBVCSSD/samannot/SAMannot/SAMannot-audit
conda activate samannot
git pull --ff-only origin audit/lek4-validation &&
python audit/check_gt_encoding.py
```

Upload the resulting report:

```bash
git add audit/results/gt-encoding-*/verification.json
git commit -m "Add GT encoding provenance check" &&
git push origin audit/lek4-validation
```

Task 1 image correspondence is complete for the verified source. Task 2 remains
at the encoding gate, with manual confirmations recorded and exclusions defined.
The reviewed reannotation and propagation already exist; do not rerun them.
Task 4 should use a released reference manifest after this technical gate is
resolved and screening is recomputed if normalization changes masks. Evaluate
GT-free signals against that reference; GT is never an input to deployed signals.
Report coverage by bird/block and presence, and retain exclusion/sensitivity
counts. Do not call unreviewed surviving masks biologically perfect or treat
GT-assisted annotation as an independent test of novice manual annotation.
