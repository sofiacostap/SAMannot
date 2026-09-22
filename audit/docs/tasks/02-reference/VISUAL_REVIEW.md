# Complete visual GT review

Inspect all 3,802 GT images (15,208 bird/frame observations), including previously
withheld observations and annotation frames. Every observation starts unreviewed.
The earlier automatic partition is context, not a visual decision. The 12 video
frames without GT do not acquire references through this review.

## Start on Lumen

```bash
cd /media/hbvcai/HBVCSSD/samannot/SAMannot/SAMannot-audit
conda activate samannot
git pull --ff-only origin audit/lek4-validation
python -m unittest discover -s audit -p test_review_all_gt.py
python audit/review_all_gt.py
```

Open the printed `http://127.0.0.1:8767/?token=...` address in a browser on Lumen.
Keep the terminal running. Existing NumPy and Pillow dependencies suffice; no GPU
inference is performed. Photos and normalized categorical masks stay on Lumen.
The tool checks the source inventory and verifies photograph fingerprints and
mask hashes when an image is loaded. Verification failure prevents decisions.

## Review procedure

1. Compare the photograph on the left with its GT overlay on the right. Check
   all birds: identity, visible body coverage, missing masks and stray regions.
2. Use left/right arrows to inspect adjacent frames, or Space to play/pause.
   Playback is for motion and identity context; it never accepts observations.
3. Use the GT display selector to isolate a bird. Enable full pixel size and
   scroll for small defects. The default preview is reduced for screen space.
4. If every remaining unreviewed bird is correct, press A or the acceptance
   button. This saves decisions for the current frame only. It does not advance
   or overwrite previous flags. An empty GT can be accepted only when absence
   of a visible mask is appropriate, including complete occlusion when justified.
5. For a questionable bird, select Uncertain or Confirmed GT problem, describe
   the issue and press Save this bird. Other birds remain independent. Partial
   visibility with no GT mask is a problem, not automatically accepted absence.
6. Continue to the next frame. Next unreviewed returns to remaining work. The
   video-frame jump uses the verified mapping; an index without GT is rejected.

Use neighbouring images to follow identities. If appearance or occlusion prevents
a defensible decision, retain uncertainty. Complete review coverage cannot prove
absolute biological correctness. Half-resolution source annotations also limit
boundary precision. No ambiguous masks are automatically repaired.

Start with 10–20 images to learn the controls, then review manageable batches.
At 3–5 seconds per image, 3,802 images alone take about 3.2–5.3 hours; detailed
inspection, pauses and identity checks add time. Fast playback alone is not a
documented inspection of every bird/frame.

## Save, stop and resume

Every explicit save updates `review.json` atomically and appends `events.jsonl`
inside a new `audit/results/full-gt-review-<timestamp>/` directory. Notes being
typed are not saved until Save this bird is pressed. Navigation warns about
unsaved edits. Stop with Ctrl+C in the terminal after saving.

Resume the exact directory printed by the previous run:

```bash
python audit/review_all_gt.py --resume audit/results/full-gt-review-<timestamp>
```

Replace `<timestamp>` with the existing folder suffix. Starting without `--resume`
creates a separate review. Resume verifies that the input manifests are unchanged.
Use one review session at a time. Optional `--reference` and `--source` select
another compatible inventory; repeat those arguments when resuming that review.

After stopping, publish the review records through GitHub:

```bash
git add audit/results/full-gt-review-*/
git commit -m "Record visual GT review progress"
git push origin audit/lek4-validation
```

If nothing changed, there is nothing new to commit; an already committed record
can still be uploaded with the push command alone.

## Interpretation and next release

After all observations have a decision, stop the review server and export the
accepted subset with `export_reviewed_gt.py --review <review-folder> --destination
<existing-destination-folder>`. Supply destination paths at runtime. The exporter
refuses any pending observations and exports only accepted_present/accepted_absent.
Uncertain and confirmed-problem pairs are omitted. Accepted absence has a valid
empty binary mask; an omitted identity must never be interpreted as background.

Each export creates a unique `accepted-gt-<timestamp>` folder with individual
0/255 bird masks, matching photographs, overlays, a browser-readable `index.html`,
an eligibility manifest and file hashes. Inspect index.html locally without a
server. Source masks and photographs are verified during export. A failed export
retains an `.incomplete` suffix and is not a released package. Existing exports
are never overwritten. Metadata uses relative file paths and hashes; it does not
copy free-text review notes or absolute machine paths. This package is for visual
inspection and a subsequent versioned evaluation; it does not update old scores.

`pending` means not yet decided. `accepted_present` and `accepted_absent` are
explicit visual acceptances. `uncertain` and `confirmed_problem` remain unresolved
for evaluation. All four non-pending categories count toward review coverage;
coverage completion is separate from the number of usable references.

The tool records bird/frame decisions, not whole-video exclusions. It does not
change the released Task 2 manifest, GT masks, predictions or scores. Once review
coverage is complete, reconcile recorded problems and any supported intervals,
publish a versioned reference subset, and rerun evaluation against that subset.
Do not rerun SAM2 merely because the evaluation reference changes. The conversion
of review decisions into that new release is a separate, pending step.
