# Local audit workspace

This isolated clone is for code review and CPU-side validation development in Visual Studio Code. It does not emulate Lumen's GPU, CUDA, video decoder, or installed dependencies. The original laptop checkout and synced project sources are unchanged.

Baseline laptop commit: `7dffa0794abfcd434501e4606adb8df4ce2b27dc`.
GitHub remote: https://github.com/sofiacostap/SAMannot.git
Working branch: `audit/lek4-validation`.

## Established locally; not yet verified on Lumen

- `cross_block_propagate.py:67-71` converts block-local prompts with `block_id * block_size + pt.idx`. The local implementation loads all prompts into a full-video predictor, calls propagation once without explicit direction (`:177`), and saves per-object Boolean NumPy arrays named `frame_{index:06d}_obj_{id}.npy` (`:185-186`). It does not implement the described per-block PNG pipeline. Do not substitute this version on Lumen.
- `sam_annotator.py:112-115` generates IDs from label name plus current block. Its GUI helper `propagate_to_all` calls both directions. This does not establish which calls Lumen's headless script makes.
- `mask_quality_scorer.py:296-310` builds online records; `:341-344` declares the schema; `:347-358` writes `{run_name}_quality_scores.csv`. Columns: `abs_frame, local_frame, block_id, label, logit_score, continuity_score, ref_iou_score, combined_score, flagged, area_px, area_ratio, centroid_jump, reason`. The offline scoring path also builds records separately. The headless caller supplying logit scores must be inspected on Lumen.
- `iou_validator.py` matches identities greedily on every frame, despite an unused variable/comment suggesting a fixed mapping. It excludes absent SAM labels, skips frames with no SAM/GT masks, and only records matched pairs. Its reported denominator therefore does not represent all intended frame-bird observations. An absent prediction for a present GT bird should count as a miss in a tracking evaluation.
- The same validator infers numeric SAM labels from alphabetical quality-score names. That ordering needs evidence from the mask exporter/session; otherwise scorer precision/recall may join different birds.
- The validator compares PNGs directly. This avoids dependence on the visualization background but does not prove source frame correspondence. It resizes unequal mask dimensions automatically and assumes equal numeric filenames correspond. Neither assumption establishes spatial or temporal alignment.

## Rerun decision pending evidence

Do not rerun SAM2 yet. First identify the exact Lumen code and inputs. If masks used incorrect block IDs or mismatched source images, regenerate affected masks and downstream results. If masks are sound but validation matching/denominator/scorer joins are wrong, rerun validation and downstream calibration/plots only. If only visual backgrounds were wrong, regenerate affected visual proofs. File timestamps alone cannot establish which code produced an output.

## First Lumen collection

Transfer `audit/collect_evidence.py` to Lumen (a separate audit checkout is preferable). Activate the existing `samannot` environment and run:

```bash
python audit/collect_evidence.py --root /media/hbvcai/HBVCSSD/samannot > /tmp/samannot-lumen-evidence.json
```

The script only reads the repository and outputs; shell redirection creates the report in `/tmp`. Attach the JSON here. It contains numbered source files, hashes, Git history/status, representative listings and PNG details, actual CSV headers/counts, and duplicate key checks. It does not load sessions or run propagation. Missing output folders are reported explicitly. Source paths, labels, and source code are included; no credentials or shell history are collected.

After comparing versions, the next checks will inspect trusted session prompt indices and label order, reconcile GT/FRAMES/video numbering and image geometry, and recompute representative IoUs across block boundaries. Those require Lumen data. Do not infer exact output coverage from nominal video length alone: the GUI has an extra boundary-frame mechanism.
