# Script catalogue

Run commands from the repository root. Classification concerns workflow suitability,
not a claim that an old script has stopped executing. Many historical tools also
export helpers imported by current tools, so deleting them would break dependencies.

## Current workflow, grouped by task

| Task | Scripts under `audit/` | Purpose |
|---|---|---|
| 1 | `verify_correspondence.py`, `prepare_verified_frames.py` | Exact image correspondence and sequential source |
| 2 | `prepare_task2_reference.py` | Released categorical reference and evaluation partition |
| 3 | `reannotate.py`, `gpu_prompt_check.py`, `propagate_reviewed.py` | Review points, check prompts, generate baseline |
| 4 | `evaluate_task4.py` | Evaluate frozen scores against released reference |
| 5 | `identity_swap_review.py` | GT-assisted proposals, previews and approval-gated overlay |
| 6 | `temporal_analysis.py` | Plot existing Task 4 measurements; no new inference |

The current scripts contain Lek4-specific defaults, label IDs and source paths.
They are research tools, not yet a generic multi-dataset package. Check each
command's `--help` and the task protocol before use. Do not rerun completed stages
simply because their commands are available. GPU/data checks require Lumen.

## Supporting diagnostics and shared dependencies

| Scripts | Role and limitation |
|---|---|
| `collect_evidence.py` | Read-only source/output inventory |
| `check_alignment.py`, `verify_extraction.py` | Diagnose historical image matching and extraction |
| `check_gt_encoding.py` | Inspect encoding and test explicit conversion recipes |
| `gt_audit.py` | Screening and shared component/identity helpers; old candidate counts are not the released partition |
| `build_task2_set.py` | Shared partition logic; standalone provisional partition is superseded |
| `preview_empty_intervals.py`, `classify_empty_visibility.py` | Empty-mask diagnostics; visibility classification uses GT and is not deployable as a GT-free detector |
| `corrected_propagation.py` | Earlier propagation entry point; shared export helper remains used by current prompt/propagation tools |
| `recompute.py` | Historical evaluator and shared I/O/image helpers used by current scripts |

## Superseded entry points

`run_lumen.sh` and standalone `recompute.py` reproduce the early provisional
recomputation, not the current released Task 4 evaluation. `review_results.py`
reviews that historical output. `corrected_propagation.py` is not the reviewed
baseline entry point; use the Task 3 workflow for any intentionally new run.

Root-level `iou_validator.py`, `visual_proof.py`, `calibrate_gt_frame_offset.py`,
`cross_block_propagate.py` and `patch_cross_block.py` belong to the earlier
application/investigation path. They are not certified replacements for the
audited workflow. The old validator's per-frame identity rematching and selected
denominator make its metrics unsuitable as the released individual-tracking baseline.
The main application files remain in place; this cleanup does not certify or
rewrite the upstream GUI.

## Verification

On 2026-09-16, all 65 local audit tests passed with
`python -m unittest discover -s audit -p "test_*.py"`.
This verifies the covered CPU logic and synthetic cases, including swap-overlay
application on test data. It is not a fresh GPU run or application of project swaps.
No script was identified as broken by these tests; this does not establish that
every command works with absent data, different dependencies or another dataset.
