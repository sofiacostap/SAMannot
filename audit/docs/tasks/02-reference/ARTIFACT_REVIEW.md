# Playback and artifact separation

## Flagging an interval

Expand Flag a frame range. Enter inclusive VIDEO start/end numbers, the affected
identity and a reason. Flag range opens a confirmation showing bounds and count.
Confirm range copies the affected whole frames to the artifact folder and adds them
to existing flags; it never replaces the flag list. A progress backup is saved first.
Ranges containing missing GT frames are refused. Applying the same range again does
not duplicate frame flags. Frame-by-frame Undo remains available. The interval note
is historical context; current flags remain authoritative after individual undo.

Range confirmation records a reported issue, not playback coverage. Unseen images
are not automatically counted as inspected. Keep the window open during copying.
If copying fails, successfully saved flags remain; retry the same range. Interval
notes, including the affected identity, are saved in progress and final exports.

## Original GT inspection

The player now reads original categorical palette PNGs directly from the source
directory recorded in the reference report. It verifies their recorded hashes.
It does not read the generated normalized mask pixels for display. Overlay display
uses an explicit identity lookup and exact 2x pixel repetition to match the photograph;
it does not smooth, repair or remove regions. Original palette view serves the source
PNG itself, unchanged, at its native image resolution. Binary view isolates one
identity at that same resolution. Photograph-only view removes every overlay.

Use the View selector while paused to distinguish a natural image colour from a
labelled speck. Original palette colours differ from overlay colours. Binary white
pixels are the selected identity; these can confirm whether a suspect patch exists
in the original annotation. A flag means needs inspection, not proof of its cause.

The same launch command still works. Existing flags are retained. Earlier display
coverage is archived separately and the original-GT viewing pass starts with zero
coverage. No flags, source masks or historical exports are deleted. New artifact
copies and final exports include the original palette PNG as well as the full-size
derived label mask. Previously flagged folders remain historical; final exports
include original files for every retained and flagged frame.

Use `review_gt_artifacts.py` for a complete visual pass without individual approvals.
This is an alternative to `review_all_gt.py`, not an automatic conversion of its
decisions. Existing scores and original GT files remain unchanged.

From the repository root, run:

```bash
python audit/review_gt_artifacts.py --destination "/path/to/existing/output/folder"
```

Open the printed local address in a browser on the same computer. Keep the terminal
running. Start only one server and one review tab for this output folder.

1. Press Play. Speeds are maximum targets; image verification and disk reads can
   reduce the actual rate. Frames are not skipped to meet the speed target.
2. Click the image to pause immediately. Pending loads cannot replace the frozen
   image. Use arrows to inspect adjacent images.
3. Click Flag this frame, check its displayed video/GT numbers, and Confirm artifact.
   The entire GT image (all bird labels) is separated, never the whole video.
4. Use Undo flag if needed. Press Play to continue. Correct frames need no approval.
5. After the full pass, press Finish and separate GT and confirm that the sequence
   was watched. Next unseen frame locates gaps if the slider skipped any images.

Progress saves automatically under `gt-review/progress.json` in the destination.
Stop with Ctrl+C after saving; run the same command to resume. Hiding the browser
tab pauses playback. Displayed-frame coverage records display, not proof of attention.

Confirmed artifacts immediately receive a photograph, original categorical GT
mask and overlay in `gt-review/gt artifacts/<video-frame>/`. Undo moves these copies
to `undo history`; it never deletes source data. The final versioned package under
`gt-review/exports/<timestamp>/` contains `retained GT` and `gt artifacts`, matching
photographs, a frame mapping manifest and counts. Filenames use video indices.
Masks preserve all original categorical IDs. A failed export is labelled `.incomplete`.

Retained means unflagged after the confirmed visual pass, not independently certified
perfect. This partition does not inherit the earlier automatic exclusions and must
not silently replace the frozen evaluation set. A new evaluation must explicitly use
its manifest and report the changed reference policy. No SAM2 inference is required.

The tool uses verified source images and the original categorical GT underlying the
existing reference release. Input-manifest changes prevent accidental resumption
against another dataset. Actual mask/photo hashes are checked on load and rechecked
at export. No personal names, machine paths or free-text review notes are embedded
in the exported manifest.
