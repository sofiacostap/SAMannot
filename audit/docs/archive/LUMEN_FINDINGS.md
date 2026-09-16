# Historical report — superseded

Do not use these provisional metrics as the current baseline or follow these commands as the current workflow. See [current status](../README.md).

# Lumen audit — evidence received 9 September 2026

Evidence: `audit/results/lumen-evidence.json`, uploaded in commit `a1faca3`. All code line numbers below refer to the numbered Lumen source embedded in that report, not the different laptop source files. Lumen HEAD was `8235ecfa4cdda4d3c11cd4b740c5411efa8279de`, with staged and unstaged edits to core pipeline files. HEAD alone therefore cannot reproduce the working code or establish the code used by earlier runs.

## Conclusion

The reported 13,627 validation observations exist, but the report contains only sample rows and counts, so the claimed mean IoU 0.463 and error rate 45% have not been independently recomputed. They should not yet be presented as validated individual-tracking performance. Current validation rematches identities per frame and excludes missing predictions. Scorer joins and duplicate records introduce additional problems. The `current_block` fix is present in current Lumen code; the code that produced the August outputs is not established. No production files or outputs were modified, and no propagation or validation was rerun.

## 1. Indexing and block boundaries

Blocks and internal frame indices are zero-based. `cross_block_propagate.py:241` loops over `range(num_blocks)`; `:84`, `:166` and scorer `:259` implement absolute frame = block ID × block size + local frame. The session supplies block size (`cross_block_propagate.py:194`); the observed CSV counts and boundary behavior are consistent with 750.

For 750-frame blocks, nominal ownership is:

| Block | Inclusive absolute frames |
|---|---|
| 0 | 0–749 |
| 1 | 750–1499 |
| 2 | 1500–2249 |
| 3 | 2250–2999 |
| 4 | 3000–3749 |
| 5 | 3750–4499, clipped to actual length |
| 6 | 4500–5249 |
| 7 | 5250–5999 |
| 8 | 6000–6749 |
| 9 | 6750–7499 |
| 10 | 7500–8249 |
| 11 | 8250–8999 |
| 12 | 9000–9749, clipped to actual length |

This nominal mapping is not identical to processing/export behavior. `extract_block_frames`, lines 87–110, sets `end=min((b+1)*B,total-1)`, then reads inclusively through `min(end+1,total-1)`. Full blocks place 752 images in the SAM input folder, including local indices 750 and 751. `save_combined_masks:164-165` excludes only local index `len(frame_files)`, so local 751 can be exported into the next block's range and later overwritten. The final video frame is excluded from `frame_files` and treated as the extra frame; the export skip removes it. Quality scoring occurs before export filtering.

Observed output: v9 has 9,023 PNGs, with listed endpoints `000000.png` and `009022.png`; quality rows extend to 9023. Lek4 has 3,813 PNGs, endpoints `000000.png` and `003812.png`; quality rows extend to 3813. These are consistent with the final-frame omission. The v9 count contradicts the supplied 9,750-frame description. Exact source-video length, session block size, full filename gaps, and prompt frames still need direct collection; do not assume the missing v9 tail consists of failed tracking frames.

## 2. Mask files

`cross_block_propagate.py:152-176` writes one combined color-coded PNG per absolute frame, named `{abs_idx:06d}.png`, directly inside each output directory. Background is black; labels use Pascal VOC RGB colors: 1=(128,0,0), 2=(0,128,0), 3=(128,128,0), 4=(0,0,128). The image is uint8, three channels. Overlapping masks are assigned in iteration order, with later masks overwriting earlier pixels.

Representative listings and inspections:

| Directory | Example files | PNG count | Sample shape, H×W×C |
|---|---|---:|---|
| headless_output_v9 | 000000.png, 004511.png, 009022.png | 9,023 | 1080×1920×3 |
| headless_output_lek4_v2 | 000000.png, 001906.png, 003812.png | 3,813 | 1520×2688×3 |

The samples contain black plus the expected palette colors (one v9 sample lacks one label color). The collector reports OpenCV BGR ordering; the RGB colors above account for that conversion. The exporter assigns palette indices in saved session label order (`:228`), not alphabetical order. That session ordering should be verified explicitly before attaching bird names to colors.

## 3. Quality CSV schema and writers

Actual filename in both directories: `headless_window16_quality_scores.csv`.

Lek4 header, in exact order:

```text
abs_frame,local_frame,block_id,label,logit_score,continuity_score,ref_iou_score,combined_score,flagged,area_px,area_ratio,centroid_jump,reason
```

v9 has the same ordering except that `ref_iou_score` is absent (12 columns rather than 13). Its sample combined values are consistent with the older 0.6 B + 0.4 A formula; the exact historical writer has not been collected.

For current Lumen code, `mask_quality_scorer.py:296-310` constructs all online record fields and `:340-358` declares/writes the CSV. `cross_block_propagate.py:446` calls `save_scores(..., run_name="headless_window16")`.

| Fields | Computation/source |
|---|---|
| abs_frame, local_frame, block_id, label | Scorer lines 259, 297–300; caller passes block ID, size, local predictor index and label names |
| logit_score | `score_logits_for_frame:136-143`: mean over the entire object's spatial logit tensor, then sigmoid clipped to [0.1,0.9] (`:131-133`); pipeline captures this live (`:334`, `:372`) |
| continuity_score | `continuity_score:157-204`: minimum of normalized area-change and centroid-motion scores, with dropout/oversized cases |
| ref_iou_score | `reference_iou_score:108-124`; reference captured at best frame (`cross_block_propagate.py:335-340`) |
| combined_score | Scorer `:285-288`: 0.4 B + 0.3 A + 0.3 C if logits exist; otherwise 0.5 A + 0.5 C |
| flagged | Scorer `:290`: unrounded combined score <0.35, stored as integer |
| area_px, area_ratio, centroid_jump | Details returned by continuity calculation; areas in pixels, current/previous area ratio, Euclidean centroid displacement in pixels |
| reason | Continuity reason, with conditional reference-drift replacement (`:292-294`) |

Scores are rounded to four decimal places in CSV. Some unavailable details serialize as empty cells, and some use `N/A`. Thresholding CSV values can differ from thresholding unrounded values at the boundary.

Observed quality rows: v9 36,220 with 124 duplicate frame-label keys; Lek4 15,300 with 44 duplicates. The duplicates are real collector counts, not inferred. They need resolution before joining or aggregating data.

## 5. Propagation direction

Both directions. `find_best_frame:123-133` takes the maximum point-prompt local index, defaulting to zero. Forward call: `:323-327`. Predictor state is reset and prompts re-added (`:347-360`), then backward call uses `reverse=True` (`:362-367`). Predictor `sam2_video_predictor.py:553` defaults to forward; `:574-583` includes the start frame in both passes when best_frame>0, but skips reverse processing when best_frame=0.

The quality scorer is reused across passes without resetting previous-mask state. At a nonzero best frame, the backward pass therefore initially compares the prompt frame against the end of the forward pass. It also appends another prompt-frame record, while tracking_results overwrites the earlier mask. Subsequent backward continuity measures the next chronological frame, not the previous chronological frame. Combined with boundary frames, this explains mechanisms producing duplicate rows and inconsistent score/mask ownership.

## Validation and rerun assessment

1. **current_block:** assignment exists at `cross_block_propagate.py:256`, before ID lookup (`:290-293`) and prompt feeding. This establishes the present code fix, not the origin of stored masks. The output timestamps precede the evidence snapshot, and relevant code is uncommitted. Do not use timestamps alone to certify an earlier run.
2. **Identity and missing predictions:** `iou_validator.py:190-207` rematches every frame. Empty SAM masks are excluded during extraction and frames with no SAM or GT masks are skipped (`:194-195`). Unmatched GT birds do not get explicit miss records. The 13,627 rows therefore measure selected matched pairs, not a complete fixed-identity tracking population. The nominal 3,802×4=15,208 pairs is only an upper bound: genuine absence/GT validity also need explicit handling.
3. **Scorer alignment:** validator `:267-269` uses alphabetical names; the exporter uses session order. The quality samples begin darkest, brownish, orange, suggesting a different order. `hyperparam_search.py:60-62` instead uses first occurrence in the CSV. Those methods disagree, and neither verifies saved palette-to-name metadata. The frame-zero IoU examples (label 2→GT113, 3→GT38, 4→GT14) also differ from the supplied name-to-GT association. This is evidence to investigate, not permission to infer a new mapping from one frame.
4. **Duplicates:** hyperparameter search appends each matched quality row (`:53-84`) without deduplication. It reports 13,664 pairs versus 13,627 IoU records, consistent with repeated weighting of matched observations. Recompute after establishing one authoritative score per exported mask/frame/identity.
5. **Signal B:** this implementation averages foreground and background logits over the full image and clips the sigmoid. All supplied quality samples show 0.1. Background domination is a plausible explanation for poor discrimination, but the full distribution is not collected. It is not simply an unmodified SAM object-confidence score.
6. **FRAMES versus video:** current `visual_proof.py:77-79` reads FRAMES backgrounds, so that source-code correction is present. Actual generated images were not collected. The script still opens video unnecessarily (`:64-65`), writes to Desktop (`:11`), resizes SAM labels with default interpolation (`:84`), and filters GT components at display resolution (`:106-112`). The displayed mask can therefore differ from the mask used for the caption's IoU. The validator itself compares PNGs, but source-image correspondence remains unverified: headless propagation decodes video and saves JPEG inputs. Pixel differences alone do not prove geometric misalignment; matched landmarks and frame correspondence must be checked.

### What needs rerunning

- **Validation and downstream precision/recall, ablation, grid search, and temporal plots:** recompute with explicit fixed identity mapping, missing-prediction policy, verified frame correspondence, and deduplicated score joins before treating them as individual-tracking results. Preserve old outputs as historical results.
- **Quality scoring:** recompute or reconcile records against exported masks with clear frame ownership and direction handling. Signal A/C can be recalculated from masks once prompt references are known; corrected live-logit measurements require saved logits or a future propagation run.
- **Visual proof:** regenerate against the corrected validation data, use nearest-neighbor label resizing, and disclose any GT filtering. Write future outputs inside the designated workspace.
- **SAM2 masks:** no blanket GPU rerun is justified yet. Investigate historical block IDs, actual video length, missing terminal frames, and source-image alignment first. Rerun affected blocks if those checks demonstrate faulty inputs or masks. A correct boundary implementation is needed for future complete exports.

Still required from Lumen: session metadata/prompt frames and label order; pipeline summary text and run provenance; complete validation/quality CSVs; GT/FRAMES/video numbering, dimensions and representative aligned images. The current report does not contain enough data to recompute full metrics or certify historical run validity.
