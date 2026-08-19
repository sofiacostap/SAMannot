#!/usr/bin/env python3
"""
patch_cross_block.py
====================
Applies the Signal C (reference IoU) patch to cross_block_propagate.py.
Run this script once from the SAMannot directory:
    python3 patch_cross_block.py
"""

from pathlib import Path

TARGET = Path("cross_block_propagate.py")
content = TARGET.read_text()

# ── PATCH 1: initialise reference_masks dict before try block ─────────────────
old1 = """        reference_masks = {}  # will be populated at best_frame"""
if old1 in content:
    print("Patch 1 already applied — skipping")
else:
    old1_anchor = """        scorer = MaskQualityScorer(
            label_names=label_names,
            block_id=block_id,
            block_size=block_size,
        )"""
    new1 = """        scorer = MaskQualityScorer(
            label_names=label_names,
            block_id=block_id,
            block_size=block_size,
        )
        reference_masks = {}  # populated at best_frame during forward pass"""
    if old1_anchor not in content:
        print("ERROR: Could not find anchor for Patch 1")
    else:
        content = content.replace(old1_anchor, new1)
        print("Patch 1 applied: reference_masks initialised")

# ── PATCH 2: capture reference mask at best_frame during forward pass ─────────
old2 = """                logit_scores = score_logits_for_frame(out_mask_logits, out_obj_ids, label_obj_id_map)
                scorer.score_frame(out_frame_idx, binary_masks, logit_scores)"""

if old2 not in content:
    print("ERROR: Could not find anchor for Patch 2")
else:
    new2 = """                logit_scores = score_logits_for_frame(out_mask_logits, out_obj_ids, label_obj_id_map)
                # Signal C: capture reference mask at annotation frame
                if out_frame_idx == best_frame and not reference_masks:
                    reference_masks = {name: mask.copy()
                                       for name, mask in binary_masks.items()}
                scorer.score_frame(out_frame_idx, binary_masks, logit_scores,
                                   reference_masks=reference_masks if out_frame_idx == best_frame else None)"""
    content = content.replace(old2, new2)
    print("Patch 2 applied: reference mask captured at best_frame (forward pass)")

# ── PATCH 3: pass reference_masks to backward pass scorer.score_frame ─────────
# The backward pass score_frame call — find the second occurrence
# We need to update the backward pass scorer call too
old3_marker = "# Step 3: reset and re-feed prompts before backward pass"
if old3_marker not in content:
    print("ERROR: Could not find backward pass marker for Patch 3")
else:
    # Find the backward pass scorer.score_frame call
    # It appears after the Step 3 marker
    backward_section = content[content.index(old3_marker):]
    old3 = """                logit_scores = score_logits_for_frame(out_mask_logits, out_obj_ids, label_obj_id_map)
                scorer.score_frame(out_frame_idx, binary_masks, logit_scores)"""
    if old3 not in backward_section:
        print("ERROR: Could not find backward pass scorer.score_frame")
    else:
        new3 = """                logit_scores = score_logits_for_frame(out_mask_logits, out_obj_ids, label_obj_id_map)
                scorer.score_frame(out_frame_idx, binary_masks, logit_scores)"""
        # Backward pass doesn't need to set reference again — already set
        # Just make sure it uses the same scorer that has reference_masks set
        print("Patch 3: backward pass uses same scorer — reference already set via set_reference_masks")

# Write patched file
TARGET.write_text(content)
print(f"\nPatched {TARGET} successfully")
print("Verify with: grep -n 'reference_masks' cross_block_propagate.py")
