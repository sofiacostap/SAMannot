# Task 2 GT reference status — 12 September 2026

**NOT COMPLETE: one technical encoding gate remains. No further broad manual review is requested.**

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
