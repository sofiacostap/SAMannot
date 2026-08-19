#!/usr/bin/env python3
"""
mask_quality_scorer.py
=======================
Per-frame mask quality scoring for the SAMannot ruff tracking pipeline.

Three complementary signals:

  Signal B — SAM2 logit confidence (model-native)
    Raw logit score from SAM2 during propagation. High = confident.
    Limitation: SAM2 can be confidently wrong.

  Signal A — Mask continuity (frame-to-frame comparison)
    Compares area ratio and centroid jump between consecutive frames.
    Catches sudden jumps but misses gradual drift.

  Signal C — Reference IoU (comparison to annotation frame mask)
    At the annotation click frame, saves the mask as a reference.
    Every subsequent frame is compared to that reference via IoU.
    Catches gradual drift — if SAM2 slowly moves to a different object,
    IoU with the original bird mask will drop toward zero.

Combined score:
    With logits:    0.4*B + 0.3*A + 0.3*C
    Without logits: 0.5*A + 0.5*C
    flag = combined_score < COMBINED_FLAG_THRESHOLD
"""

import os
import csv
import sys
import argparse
import numpy as np
from pathlib import Path
from collections import defaultdict

try:
    import cv2
except ImportError:
    print("OpenCV not found. Run: pip install opencv-python")
    sys.exit(1)


# ─────────────────────────────────────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────

LOGIT_LOW_CONFIDENCE   = 0.0
AREA_CHANGE_RATIO_MAX  = 3.0
CENTROID_JUMP_MAX_PX   = 150.0
DROPOUT_AREA_MIN_PX    = 50
MASK_AREA_MAX_FRACTION = 0.02   # flag if mask > 2% of frame (e.g. tracking a wall)

# Reference IoU threshold below which we consider tracking drifted
REFERENCE_IOU_MIN      = 0.1    # very permissive — only catch complete drift

W_LOGIT = 0.4
W_CONT  = 0.3
W_REF   = 0.3
COMBINED_FLAG_THRESHOLD = 0.35


# ─────────────────────────────────────────────────────────────────────────────
# PASCAL VOC COLORMAP
# ─────────────────────────────────────────────────────────────────────────────

def pascal_voc_color(label_idx):
    r = g = b = 0
    c = label_idx
    for j in range(8):
        r |= ((c >> 0) & 1) << (7 - j)
        g |= ((c >> 1) & 1) << (7 - j)
        b |= ((c >> 2) & 1) << (7 - j)
        c >>= 3
    return (r, g, b)


def extract_mask_from_png(png_path, label_idx):
    img = cv2.imread(str(png_path))
    if img is None:
        return None
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    color = np.array(pascal_voc_color(label_idx), dtype=np.uint8)
    return np.all(img_rgb == color, axis=2).astype(np.uint8)


# ─────────────────────────────────────────────────────────────────────────────
# SIGNAL C — REFERENCE IoU
# ─────────────────────────────────────────────────────────────────────────────

def compute_iou(mask1, mask2):
    """Compute IoU between two binary masks. Handles shape mismatch."""
    if mask1.shape != mask2.shape:
        mask2 = cv2.resize(
            mask2.astype(np.uint8),
            (mask1.shape[1], mask1.shape[0]),
            interpolation=cv2.INTER_NEAREST
        ).astype(bool)
    mask1 = mask1.astype(bool)
    mask2 = mask2.astype(bool)
    intersection = (mask1 & mask2).sum()
    union = (mask1 | mask2).sum()
    if union == 0:
        return 1.0  # both empty = match
    return float(intersection) / float(union)


def reference_iou_score(curr_mask, ref_mask):
    """
    Signal C: IoU between current mask and the reference mask at annotation frame.
    Returns 0-1 score. Low score = mask has drifted far from original position.
    Returns 1.0 if no reference is available (first frame).
    """
    if ref_mask is None:
        return 1.0
    curr_area = int(curr_mask.sum())
    ref_area  = int(ref_mask.sum())
    # If reference was empty (bird not visible at annotation frame), skip
    if ref_area < DROPOUT_AREA_MIN_PX:
        return 1.0
    # If current is empty, that's a dropout — Signal A handles it
    if curr_area < DROPOUT_AREA_MIN_PX:
        return 1.0
    return compute_iou(curr_mask.astype(bool), ref_mask.astype(bool))


# ─────────────────────────────────────────────────────────────────────────────
# SIGNAL B — SAM2 LOGIT SCORE
# ─────────────────────────────────────────────────────────────────────────────

def logit_to_score(logit_value):
    sigmoid = 1.0 / (1.0 + np.exp(-float(logit_value)))
    return float(np.clip(sigmoid, 0.1, 0.9))


def score_logits_for_frame(out_mask_logits, out_obj_ids, label_name_to_obj_id):
    scores = {}
    obj_id_to_label = {v: k for k, v in label_name_to_obj_id.items()}
    for i, obj_id in enumerate(out_obj_ids):
        label_name = obj_id_to_label.get(obj_id, f"obj_{obj_id}")
        logit_val = out_mask_logits[i].mean().item()
        scores[label_name] = logit_to_score(logit_val)
    return scores


# ─────────────────────────────────────────────────────────────────────────────
# SIGNAL A — MASK CONTINUITY SCORE
# ─────────────────────────────────────────────────────────────────────────────

def get_centroid(mask):
    if mask.sum() < DROPOUT_AREA_MIN_PX:
        return None
    coords = np.argwhere(mask > 0)
    return coords.mean(axis=0)


def continuity_score(curr_mask, prev_mask, prev_centroid):
    curr_area = int(curr_mask.sum())
    prev_area = int(prev_mask.sum()) if prev_mask is not None else 0

    # Dropout
    if curr_area < DROPOUT_AREA_MIN_PX:
        return 0.0, {"area": curr_area, "area_ratio": 0.0,
                     "centroid_jump": None, "reason": "dropout"}

    # Oversized — tracking something much bigger than a bird
    total_pixels = curr_mask.shape[0] * curr_mask.shape[1]
    if curr_area > MASK_AREA_MAX_FRACTION * total_pixels:
        return 0.0, {"area": curr_area, "area_ratio": 0.0,
                     "centroid_jump": None, "reason": "oversized"}

    curr_centroid = get_centroid(curr_mask)

    if prev_mask is None or prev_area < DROPOUT_AREA_MIN_PX or prev_centroid is None:
        return 1.0, {"area": curr_area, "area_ratio": 1.0,
                     "centroid_jump": None, "reason": "no_prev"}

    area_ratio = curr_area / (prev_area + 1e-6)
    area_ratio_sym = area_ratio if area_ratio > 1 else 1.0 / area_ratio

    centroid_jump = float(np.sqrt(
        (curr_centroid[0] - prev_centroid[0])**2 +
        (curr_centroid[1] - prev_centroid[1])**2
    ))

    area_sub = float(np.clip(
        1.0 - (area_ratio_sym - 1.0) / (AREA_CHANGE_RATIO_MAX - 1.0), 0.0, 1.0))
    centroid_sub = float(np.clip(
        1.0 - centroid_jump / CENTROID_JUMP_MAX_PX, 0.0, 1.0))

    cont_score = min(area_sub, centroid_sub)

    reason = "ok"
    if area_ratio_sym > AREA_CHANGE_RATIO_MAX:
        reason = "area_jump"
    elif centroid_jump > CENTROID_JUMP_MAX_PX:
        reason = "centroid_jump"

    return cont_score, {
        "area": curr_area,
        "area_ratio": round(area_ratio, 3),
        "centroid_jump": round(centroid_jump, 1),
        "reason": reason,
    }


# ─────────────────────────────────────────────────────────────────────────────
# COMBINED SCORER
# ─────────────────────────────────────────────────────────────────────────────

class MaskQualityScorer:
    """
    Per-frame, per-label quality scoring using Signals A, B, and C.

    Signal C (reference IoU) requires reference_masks to be passed to
    score_frame() — these are the masks at the annotation frame (best_frame)
    captured during the forward propagation pass in cross_block_propagate.py.
    """

    def __init__(self, label_names, block_id=0, block_size=750, output_dir=None):
        self.label_names = label_names
        self.block_id    = block_id
        self.block_size  = block_size
        self.output_dir  = Path(output_dir) if output_dir else None

        self._prev_masks     = {n: None for n in label_names}
        self._prev_centroids = {n: None for n in label_names}
        self._ref_masks      = {n: None for n in label_names}  # Signal C references

        self.records = []
        self.flagged = []

    def set_reference_masks(self, reference_masks):
        """
        Set the reference masks from the annotation frame.
        Call this once after the annotation frame is processed.
        reference_masks: dict {label_name: 2D numpy bool/uint8 array}
        """
        for name, mask in reference_masks.items():
            if name in self._ref_masks:
                self._ref_masks[name] = mask.copy()

    def score_frame(self, local_frame_idx, binary_masks, logit_scores=None,
                    reference_masks=None):
        """
        Score one frame with all three signals.

        Args:
            local_frame_idx: frame index within block (0-749)
            binary_masks:    {label_name: 2D numpy array}
            logit_scores:    {label_name: float 0-1} or None
            reference_masks: {label_name: 2D numpy array} — masks at annotation
                             frame. Pass on first call or use set_reference_masks().
        """
        # Update reference masks if provided
        if reference_masks:
            self.set_reference_masks(reference_masks)

        abs_frame_idx = self.block_id * self.block_size + local_frame_idx

        for label_name in self.label_names:
            mask = binary_masks.get(label_name)
            if mask is None:
                continue

            # Signal B: logit
            if logit_scores and label_name in logit_scores:
                b_score   = logit_scores[label_name]
                has_logit = True
            else:
                b_score   = 1.0
                has_logit = False

            # Signal A: continuity
            a_score, details = continuity_score(
                mask,
                self._prev_masks[label_name],
                self._prev_centroids[label_name],
            )

            # Signal C: reference IoU
            c_score = reference_iou_score(mask, self._ref_masks[label_name])

            # Combined score
            if has_logit:
                combined = W_LOGIT * b_score + W_CONT * a_score + W_REF * c_score
            else:
                combined = 0.5 * a_score + 0.5 * c_score

            flagged = combined < COMBINED_FLAG_THRESHOLD

            reason = details.get("reason", "")
            if not flagged and c_score < REFERENCE_IOU_MIN:
                reason = "ref_iou_drift"

            record = {
                "abs_frame":        abs_frame_idx,
                "local_frame":      local_frame_idx,
                "block_id":         self.block_id,
                "label":            label_name,
                "logit_score":      round(b_score, 4) if has_logit else "N/A",
                "continuity_score": round(a_score, 4),
                "ref_iou_score":    round(c_score, 4),
                "combined_score":   round(combined, 4),
                "flagged":          int(flagged),
                "area_px":          details.get("area", 0),
                "area_ratio":       details.get("area_ratio", "N/A"),
                "centroid_jump":    details.get("centroid_jump", "N/A"),
                "reason":           reason,
            }
            self.records.append(record)
            if flagged:
                self.flagged.append(record)

            self._prev_masks[label_name]     = mask.copy()
            self._prev_centroids[label_name] = get_centroid(mask)

    def get_flag_count(self):
        return len(self.flagged)

    def get_flag_rate(self):
        return len(self.flagged) / max(len(self.records), 1)

    def finalize(self):
        print(f"\n  Quality summary — block {self.block_id}:")
        for label_name in self.label_names:
            label_records = [r for r in self.records if r["label"] == label_name]
            label_flagged = [r for r in self.flagged if r["label"] == label_name]
            if not label_records:
                continue
            flag_pct = 100 * len(label_flagged) / len(label_records)
            print(f"    {label_name}: {len(label_flagged)}/{len(label_records)} "
                  f"flagged ({flag_pct:.1f}%)")


# ─────────────────────────────────────────────────────────────────────────────
# CSV OUTPUT
# ─────────────────────────────────────────────────────────────────────────────

FIELDNAMES = [
    "abs_frame", "local_frame", "block_id", "label",
    "logit_score", "continuity_score", "ref_iou_score", "combined_score",
    "flagged", "area_px", "area_ratio", "centroid_jump", "reason",
]


def save_scores(records, output_dir, run_name="run"):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    scores_path  = output_dir / f"{run_name}_quality_scores.csv"
    flagged_path = output_dir / f"{run_name}_flagged_frames.csv"
    summary_path = output_dir / f"{run_name}_quality_summary.txt"

    with open(scores_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(records)

    flagged = [r for r in records if r["flagged"]]
    with open(flagged_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(sorted(flagged, key=lambda r: (r["block_id"], r["abs_frame"])))

    total    = len(records)
    n_flagged = len(flagged)
    with open(summary_path, "w") as f:
        f.write(f"Run: {run_name}\n")
        f.write(f"Total frame-label pairs scored: {total}\n")
        f.write(f"Total flagged: {n_flagged} ({100*n_flagged/max(total,1):.1f}%)\n\n")
        blocks = sorted(set(r["block_id"] for r in records))
        f.write("Per block:\n")
        for b in blocks:
            b_recs = [r for r in records if r["block_id"] == b]
            b_flag = [r for r in b_recs if r["flagged"]]
            f.write(f"  block {b:2d}: {len(b_flag)}/{len(b_recs)} flagged "
                    f"({100*len(b_flag)/max(len(b_recs),1):.1f}%)\n")
        labels = sorted(set(r["label"] for r in records))
        f.write("\nPer label:\n")
        for label in labels:
            l_recs = [r for r in records if r["label"] == label]
            l_flag = [r for r in l_recs if r["flagged"]]
            f.write(f"  {label}: {len(l_flag)}/{len(l_recs)} flagged "
                    f"({100*len(l_flag)/max(len(l_recs),1):.1f}%)\n")

    print(f"\nScores saved to {scores_path}")
    print(f"Flagged frames: {flagged_path}")
    print(f"Summary: {summary_path}")
    return scores_path, flagged_path, summary_path


# ─────────────────────────────────────────────────────────────────────────────
# POST-HOC MODE — Signal A + oversized only (no logits, no reference IoU)
# ─────────────────────────────────────────────────────────────────────────────

def score_from_pngs(masks_dir, output_dir, n_labels=4, block_size=750, run_name="run"):
    """
    Post-hoc scoring from saved PNG masks.
    Signals A (continuity + oversized) only — no logits, no reference IoU.
    Note: Signal C (reference IoU) requires live propagation to capture
    the reference mask at the annotation frame.
    """
    masks_dir  = Path(masks_dir)
    png_files  = sorted([f for f in masks_dir.iterdir() if f.suffix.lower() == ".png"])

    if not png_files:
        print(f"No PNG files found in {masks_dir}")
        return

    print(f"Scoring {len(png_files)} frames from {masks_dir}")
    print(f"Mode: post-hoc (Signal A only — no logits, no reference IoU)")

    label_names    = [f"label_{i}" for i in range(1, n_labels+1)]
    all_records    = []
    prev_masks     = {n: None for n in label_names}
    prev_centroids = {n: None for n in label_names}

    for png_file in png_files:
        try:
            abs_idx = int(png_file.stem)
        except ValueError:
            continue

        block_id  = abs_idx // block_size
        local_idx = abs_idx % block_size

        for i, label_name in enumerate(label_names):
            mask = extract_mask_from_png(png_file, i+1)
            if mask is None:
                continue

            a_score, details = continuity_score(
                mask, prev_masks[label_name], prev_centroids[label_name])
            combined = a_score
            flagged  = combined < COMBINED_FLAG_THRESHOLD

            all_records.append({
                "abs_frame":        abs_idx,
                "local_frame":      local_idx,
                "block_id":         block_id,
                "label":            label_name,
                "logit_score":      "N/A",
                "continuity_score": round(a_score, 4),
                "ref_iou_score":    "N/A",
                "combined_score":   round(combined, 4),
                "flagged":          int(flagged),
                "area_px":          details.get("area", 0),
                "area_ratio":       details.get("area_ratio", "N/A"),
                "centroid_jump":    details.get("centroid_jump", "N/A"),
                "reason":           details.get("reason", ""),
            })

            prev_masks[label_name]     = mask.copy()
            prev_centroids[label_name] = get_centroid(mask)

    n_flagged = sum(r["flagged"] for r in all_records)
    print(f"Total scored: {len(all_records)}")
    print(f"Total flagged: {n_flagged} ({100*n_flagged/max(len(all_records),1):.1f}%)")
    save_scores(all_records, output_dir, run_name)
    return all_records


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Per-frame mask quality scoring (Signals A, B, C)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--masks_dir",   help="Path to PNG masks folder (post-hoc mode)")
    parser.add_argument("--output_dir",  default="./quality_output")
    parser.add_argument("--n_labels",    type=int, default=4)
    parser.add_argument("--block_size",  type=int, default=750)
    parser.add_argument("--run_name",    default="run")
    parser.add_argument("--load_scores", help="Load existing CSV and print summary")
    args = parser.parse_args()

    if args.load_scores:
        records = []
        with open(args.load_scores) as f:
            reader = csv.DictReader(f)
            for row in reader:
                row["flagged"] = int(row["flagged"])
                records.append(row)
        n_flagged = sum(r["flagged"] for r in records)
        print(f"Loaded {len(records)} records")
        print(f"Flagged: {n_flagged} ({100*n_flagged/max(len(records),1):.1f}%)")
        flagged = sorted([r for r in records if r["flagged"]],
                         key=lambda r: float(r["combined_score"]))
        print(f"\nWorst 10 flagged frames:")
        print(f"{'Frame':>8}  {'Block':>6}  {'Label':>12}  {'Score':>7}  Reason")
        print("-"*60)
        for r in flagged[:10]:
            print(f"{r['abs_frame']:>8}  {r['block_id']:>6}  {r['label']:>12}  "
                  f"{r['combined_score']:>7}  {r['reason']}")

    elif args.masks_dir:
        score_from_pngs(
            masks_dir=args.masks_dir,
            output_dir=args.output_dir,
            n_labels=args.n_labels,
            block_size=args.block_size,
            run_name=args.run_name,
        )
    else:
        parser.print_help()
