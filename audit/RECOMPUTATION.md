# Corrected recomputation workflow

Run `bash audit/run_lumen.sh` inside the nested audit clone with the existing `samannot` environment active. It runs CPU regression tests, reads the original session/masks/GT/FRAMES, and creates a fresh timestamped folder under `audit/results/`. It does not launch SAM2 or modify the original scripts, masks, or metrics. Temporary test files stay under `audit/.tmp/` on Lumen.

## Corrections implemented

- One fixed palette-to-GT assignment proposed from annotation frames only, using an exhaustive four-bird assignment. It is never rematched at evaluated frames. Session order supplies palette names. Calibration observations are reported separately and excluded from the main evaluation. The proposal and montage need review; the previous text label-to-GT mapping is recorded for comparison rather than silently assumed.
- Ambiguous mappings (best/second mean-IoU margin <0.05, or any assigned per-label calibration mean <0.2) stop metric generation and return evidence for identity review. These are screening thresholds, not proof that an accepted mapping is correct.
- A present GT bird with an empty predicted mask contributes IoU=0 and an error. GT absence is a separate status, with prediction presence retained. Both-empty pairs are not treated as perfect tracking. Missing whole prediction files halt recomputation; absence outside GT coverage is listed separately.
- Exact mask dimensions and known palette colors are required. No automatic resize, temporal offset, or GT component removal occurs in IoU calculation. Unexpected GT IDs and tiny target regions are flagged in the table. Known GT identity splits remain a limitation requiring review; they are not automatically repaired.
- A and C are recalculated from the actual exported masks, once per frame-label pair. A uses the previous chronological frame within the same block and resets across gaps/boundaries. Missing adjacent reference or previous dropout is unavailable, not invented confidence. C compares against the exported annotation-frame mask; an empty current mask with a nonempty reference gives zero. An empty reference is unavailable.
- Existing CSV duplicate keys and B-score distribution are audited, not carried into recomputation. The old whole-image-mean logit score is not recoverable as a corrected confidence measure from PNGs. Recomputed AC uses equal weights; AC-only grid search uses the same score-available population for every weight and is explicitly exploratory/in-sample, not held-out performance. B/ABC comparisons await trustworthy raw logits.
- Temporal CSV/SVG show signed distance from prompt, with calibration frames excluded and per-bin counts included. They are descriptive provisional results, not a validated temporal error model.
- Visual proofs use FRAMES, raw target-ID GT pixels and full-resolution binary mask overlays before resizing the rendered photograph. They never interpolate a label image to extract a mask and do not silently filter GT. Separate video/FRAMES comparison sheets support image-alignment review; pixel differences alone are not a geometry test.

## Outputs and interpretation

`manifest.json` is authoritative about run status. `completed_provisional` means numerical recomputation finished under documented assumptions; it does not certify source-image correspondence, historical masks' generating code, or bird identities. `needs_identity_review` or `missing_mask_files` means metric generation stopped intentionally. `failed` includes the exception. Send back the complete run folder through GitHub for review.

Principal files: `session.json`, `coverage.json`, `mapping_proposal.json`, `calibration_pairs.csv`, `legacy_quality_audit.json`, `video_alignment.json`, montage JPGs, and (if checks allow) `recomputed.csv`, `summary.json`, `ac_grid_exploratory.csv`, `scorer_population.json`, `temporal_bins.csv`, `temporal.svg`. Every new output directory must be absent before the run; repeated runs never overwrite old results.

The script reads trusted project pickle sessions. It replaces data-only SAMannot label/mask classes during deserialization so the CPU audit does not need to import the GUI or initialize SAM2. Other pickle objects use ordinary deserialization; do not run it on untrusted sessions.

## Corrected future propagation

`audit/corrected_propagation.py` is a separate GPU entry point, not activated by this recomputation workflow. It creates a fresh per-block inference state, assigns stable IDs 1–4, uses exactly the owned half-open frame interval, exports the final frame, and refuses duplicate exports. It runs forward and backward with the prompt exported only once. FRAMES PNG pixels are passed losslessly; the installed SAM2 loader must support PNG. It validates that the requested memory window reaches the predictor. A new foreground-only mean sigmoid is recorded with the distinct column name `foreground_logit_score`; empty foreground is unavailable. No clipping or whole-image background averaging is applied. This measures model confidence, not correctness.

The GPU entry point requires explicit `--repo`, `--workspace`, `--session`, `--frames`, `--config`, `--checkpoint`, `--output` and optional `--window`. Paths to weights/config must be verified on Lumen before use. Its logic is covered by CPU contract tests, but actual SAM2/CUDA integration has not been run on the laptop. Do not run it on the full dataset until the CPU results, identity proposal, source-frame alignment and a short GPU smoke test are reviewed. Click coordinates were originally placed on video images; using FRAMES for propagation requires checking their correspondence first.

## Remaining work after Lumen returns results

Review identity/contact sheets and frame correspondence, inspect mismatched/uncertain GT cases, confirm the historical mask provenance where possible, and decide which blocks need GPU regeneration. No claim that all old masks are valid is made. Recompute B-inclusive evaluation only after corrected logits exist; perform independent held-out evaluation before treating tuned thresholds as generalizable.
