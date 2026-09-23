# Playback and artifact separation

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

The tool uses the same verified source images and normalized categorical GT as the
existing reference release. Input-manifest changes prevent accidental resumption
against another dataset. Actual mask/photo hashes are checked on load and rechecked
at export. No personal names, machine paths or free-text review notes are embedded
in the exported manifest.
