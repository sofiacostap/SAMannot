#!/usr/bin/env python3
"""
iou_validator.py
================
Compares SAM2 headless output masks against GT_MIVOS ground truth masks
using Intersection over Union (IoU) per bird per frame.

This validates the mask quality scorer by providing ground truth labels:
  - IoU > threshold = tracking correct
  - IoU < threshold = tracking error

Then cross-references with quality_scores.csv to compute:
  - Precision: of flagged frames, how many are genuine errors?
  - Recall:    of genuine errors, how many were flagged?
  - F1 score:  balance of precision and recall

Usage:
    python3 iou_validator.py \
        --sam2_masks /media/hbvcai/HBVCSSD/samannot/headless_output_lek4 \
        --gt_masks "/media/hbvcai/HATTER_4TB/MPI_data/MaxPlanckData/ruffs_2022/males_annotation/test_data/GT_MIVOS/cut_Lek4_right_2021_604_00-00-00_00-02-32/upscaled_masks" \
        --quality_scores /media/hbvcai/HBVCSSD/samannot/headless_output_lek4/headless_window16_quality_scores.csv \
        --output_dir /media/hbvcai/HBVCSSD/samannot/iou_validation \
        --iou_threshold 0.5
"""

import os
import csv
import argparse
import numpy as np
from pathlib import Path
from collections import defaultdict

try:
    import cv2
except ImportError:
    print("OpenCV not found. Run: pip install opencv-python")
    import sys; sys.exit(1)


# ─────────────────────────────────────────────────────────────────────────────
# PASCAL VOC COLORMAP — matches SAMannot/headless output format
# ─────────────────────────────────────────────────────────────────────────────

def pascal_voc_color(label_idx):
    """Return (R, G, B) for a label index."""
    r = g = b = 0
    c = label_idx
    for j in range(8):
        r |= ((c >> 0) & 1) << (7 - j)
        g |= ((c >> 1) & 1) << (7 - j)
        b |= ((c >> 2) & 1) << (7 - j)
        c >>= 3
    return (r, g, b)


def extract_sam2_masks(png_path, n_labels=4):
    """
    Extract per-label binary masks from SAM2 output PNG.
    Returns dict {label_idx: binary_mask}
    """
    img = cv2.imread(str(png_path))
    if img is None:
        return {}
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    masks = {}
    for i in range(1, n_labels + 1):
        color = np.array(pascal_voc_color(i))
        mask = np.all(img_rgb == color, axis=2)
        if mask.sum() > 0:
            masks[i] = mask
    return masks


def extract_gt_masks(png_path, gt_bird_ids):
    """
    Extract per-bird binary masks from GT_MIVOS mask PNG.
    GT masks use grayscale label indices (R=G=B=bird_id).
    Returns dict {bird_id: binary_mask}
    """
    img = cv2.imread(str(png_path))
    if img is None:
        return {}
    gray = img[:, :, 0]  # R=G=B so any channel works
    masks = {}
    for bird_id in gt_bird_ids:
        mask = (gray == bird_id)
        if mask.sum() > 50:  # ignore tiny noise regions
            masks[bird_id] = mask
    return masks


def compute_iou(mask1, mask2):
    """Compute IoU between two binary masks."""
    # Resize if shapes don't match
    if mask1.shape != mask2.shape:
        mask2 = cv2.resize(
            mask2.astype(np.uint8),
            (mask1.shape[1], mask1.shape[0]),
            interpolation=cv2.INTER_NEAREST
        ).astype(bool)
    intersection = (mask1 & mask2).sum()
    union = (mask1 | mask2).sum()
    if union == 0:
        return 0.0
    return float(intersection) / float(union)


def match_sam2_to_gt(sam2_masks, gt_masks):
    """
    Match each SAM2 label to the GT bird ID with highest IoU.
    Returns dict {sam2_label_idx: (gt_bird_id, iou)}
    """
    matches = {}
    used_gt_ids = set()

    # Sort SAM2 labels by best available IoU (greedy matching)
    all_ious = {}
    for sam_idx, sam_mask in sam2_masks.items():
        for gt_id, gt_mask in gt_masks.items():
            all_ious[(sam_idx, gt_id)] = compute_iou(sam_mask, gt_mask)

    # Greedy: assign best IoU matches
    sorted_pairs = sorted(all_ious.items(), key=lambda x: x[1], reverse=True)
    assigned_sam = set()
    for (sam_idx, gt_id), iou in sorted_pairs:
        if sam_idx not in assigned_sam and gt_id not in used_gt_ids:
            matches[sam_idx] = (gt_id, iou)
            assigned_sam.add(sam_idx)
            used_gt_ids.add(gt_id)

    return matches


# ─────────────────────────────────────────────────────────────────────────────
# MAIN VALIDATION
# ─────────────────────────────────────────────────────────────────────────────

def run_validation(sam2_dir, gt_dir, quality_scores_csv, output_dir,
                   iou_threshold=0.5, n_labels=4, gt_frame_offset=0):

    sam2_dir  = Path(sam2_dir)
    gt_dir    = Path(gt_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Find GT bird IDs ──────────────────────────────────────────────────────
    # Sample a few GT frames to find the meaningful bird IDs
    print("Detecting GT bird IDs...")
    gt_bird_counts = defaultdict(int)
    sample_files = sorted(gt_dir.glob("*.png"))[:50]
    for f in sample_files:
        img = cv2.imread(str(f))
        if img is None:
            continue
        gray = img[:, :, 0]
        unique, counts = np.unique(gray, return_counts=True)
        for val, count in zip(unique, counts):
            if val > 0 and count > 500:  # meaningful area
                gt_bird_counts[int(val)] += 1

    # Keep IDs that appear in most sampled frames
    gt_bird_ids = [bid for bid, cnt in gt_bird_counts.items()
                   if cnt > len(sample_files) * 0.1]
    gt_bird_ids = sorted(gt_bird_ids)
    print(f"GT bird IDs found: {gt_bird_ids}")

    # ── Find matching frames ──────────────────────────────────────────────────
    # gt_frame_offset corrects for GT_MIVOS using a different frame numbering
    # convention than the SAM2 pipeline (e.g. 1-indexed export vs 0-indexed
    # cv2.VideoCapture reads). GT PNG "N" is treated as depicting the same
    # video instant as SAM2 frame "N - gt_frame_offset". Use
    # calibrate_gt_frame_offset.py to determine the right value before trusting
    # IoU numbers near frame boundaries or bird motion.
    sam2_files = {int(f.stem): f for f in sam2_dir.glob("*.png")
                  if f.stem.isdigit()}
    gt_files   = {int(f.stem) - gt_frame_offset: f for f in gt_dir.glob("*.png")
                  if f.stem.isdigit()}

    common_frames = sorted(set(sam2_files.keys()) & set(gt_files.keys()))
    print(f"SAM2 frames: {len(sam2_files)}")
    print(f"GT frames:   {len(gt_files)}")
    print(f"Common frames: {len(common_frames)}")

    if not common_frames:
        print("ERROR: No common frame indices between SAM2 and GT output.")
        print("This may be a frame numbering mismatch.")
        print(f"SAM2 frame range: {min(sam2_files)} – {max(sam2_files)}")
        print(f"GT frame range:   {min(gt_files)} – {max(gt_files)}")
        return

    # ── Compute IoU per frame ─────────────────────────────────────────────────
    print(f"\nComputing IoU for {len(common_frames)} frames...")
    iou_records = []
    label_mapping = {}  # sam2_idx -> gt_bird_id (established from first frame)

    for frame_idx in common_frames:
        sam2_masks = extract_sam2_masks(sam2_files[frame_idx], n_labels)
        gt_masks   = extract_gt_masks(gt_files[frame_idx], gt_bird_ids)

        if not sam2_masks or not gt_masks:
            continue

        matches = match_sam2_to_gt(sam2_masks, gt_masks)

        for sam_idx, (gt_id, iou) in matches.items():
            is_error = iou < iou_threshold
            iou_records.append({
                "frame_idx":    frame_idx,
                "sam2_label":   sam_idx,
                "gt_bird_id":   gt_id,
                "iou":          round(iou, 4),
                "is_error":     int(is_error),
                "iou_threshold": iou_threshold,
            })

        if frame_idx % 200 == 0:
            avg_iou = np.mean([r["iou"] for r in iou_records[-n_labels:]])
            print(f"  Frame {frame_idx}: mean IoU = {avg_iou:.3f}")

    # ── Save IoU results ───────────────────────────────────────────────────────
    iou_path = output_dir / "iou_per_frame.csv"
    with open(iou_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["frame_idx","sam2_label","gt_bird_id","iou","is_error","iou_threshold"])
        writer.writeheader()
        writer.writerows(iou_records)
    print(f"\nIoU results saved to {iou_path}")

    # ── Summary stats ──────────────────────────────────────────────────────────
    total = len(iou_records)
    errors = sum(r["is_error"] for r in iou_records)
    mean_iou = np.mean([r["iou"] for r in iou_records])
    print(f"\nIoU Summary (threshold={iou_threshold}):")
    print(f"  Total frame-label pairs: {total}")
    print(f"  Tracking errors (IoU < {iou_threshold}): {errors} ({100*errors/total:.1f}%)")
    print(f"  Mean IoU: {mean_iou:.3f}")

    # Per GT bird ID
    for gt_id in gt_bird_ids:
        bird_recs = [r for r in iou_records if r["gt_bird_id"] == gt_id]
        if not bird_recs:
            continue
        bird_errors = sum(r["is_error"] for r in bird_recs)
        bird_mean = np.mean([r["iou"] for r in bird_recs])
        print(f"  GT bird {gt_id}: mean IoU={bird_mean:.3f}, errors={bird_errors}/{len(bird_recs)} ({100*bird_errors/len(bird_recs):.1f}%)")

    # ── Cross-reference with quality scorer ───────────────────────────────────
    if not quality_scores_csv or not Path(quality_scores_csv).exists():
        print("\nNo quality scores CSV provided — skipping precision/recall/F1")
        return

    print("\nCross-referencing with quality scorer...")

    # Load quality scores
    scorer_flags = {}  # (frame_idx, sam2_label) -> flagged (0 or 1)
    with open(quality_scores_csv) as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                frame = int(row["abs_frame"])
                label_name = row["label"]
                flagged = int(row["flagged"])
                scorer_flags[(frame, label_name)] = flagged
            except (ValueError, KeyError):
                continue

    print(f"  Loaded {len(scorer_flags)} scorer records")

    # Build label name mapping from sam2_label index
    # We need to map sam2_label (1,2,3,4) to label names (darkest, brownish, etc.)
    # Use alphabetical order as default (matches SAMannot label creation order)
    label_idx_to_name = {}
    if scorer_flags:
        label_names = sorted(set(k[1] for k in scorer_flags.keys()))
        for i, name in enumerate(label_names):
            label_idx_to_name[i+1] = name
        print(f"  Label mapping: {label_idx_to_name}")

    # Compute precision, recall, F1 at multiple thresholds
    print("\nPrecision / Recall / F1 at different scorer thresholds:")
    print(f"{'Threshold':>12} {'Precision':>10} {'Recall':>8} {'F1':>8} {'Flagged':>8}")
    print("-" * 55)

    # Load all scores to test different thresholds
    all_scores = []
    with open(quality_scores_csv) as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                frame = int(row["abs_frame"])
                sam_label_name = row["label"]
                combined = float(row["combined_score"])

                # Find matching IoU record
                sam_idx = next((i for i, n in label_idx_to_name.items()
                                if n == sam_label_name), None)
                if sam_idx is None:
                    continue

                iou_rec = next((r for r in iou_records
                                if r["frame_idx"] == frame
                                and r["sam2_label"] == sam_idx), None)
                if iou_rec is None:
                    continue

                all_scores.append({
                    "frame": frame,
                    "label": sam_label_name,
                    "combined_score": combined,
                    "is_error": iou_rec["is_error"],
                })
            except (ValueError, KeyError):
                continue

    if not all_scores:
        print("Could not match quality scores to IoU records.")
        print("Check that frame indices and label names align between the two files.")
        return

    # Test thresholds from 0.1 to 0.9
    best_f1 = 0
    best_threshold = 0.35
    results = []
    for thresh in [round(t * 0.05, 2) for t in range(2, 18)]:
        flagged = [s for s in all_scores if s["combined_score"] < thresh]
        tp = sum(1 for s in flagged if s["is_error"] == 1)
        fp = sum(1 for s in flagged if s["is_error"] == 0)
        fn = sum(1 for s in all_scores
                 if s["is_error"] == 1 and s["combined_score"] >= thresh)

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall    = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1        = 2 * precision * recall / (precision + recall) \
                    if (precision + recall) > 0 else 0

        marker = " ← best" if f1 > best_f1 else ""
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = thresh

        results.append((thresh, precision, recall, f1, len(flagged)))
        print(f"{thresh:>12.2f} {precision:>10.3f} {recall:>8.3f} {f1:>8.3f} {len(flagged):>8}{marker}")

    print(f"\nBest threshold: {best_threshold} (F1={best_f1:.3f})")
    print(f"Current threshold (0.35): see row above")

    # Save full results
    pr_path = output_dir / "precision_recall_f1.csv"
    with open(pr_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["threshold","precision","recall","f1","n_flagged"])
        writer.writerows(results)

    summary_path = output_dir / "validation_summary.txt"
    with open(summary_path, "w") as f:
        f.write(f"IoU threshold for ground truth: {iou_threshold}\n")
        f.write(f"Total frame-label pairs compared: {total}\n")
        f.write(f"GT tracking errors: {errors} ({100*errors/total:.1f}%)\n")
        f.write(f"Mean IoU: {mean_iou:.3f}\n\n")
        f.write(f"Best scorer threshold: {best_threshold}\n")
        f.write(f"Best F1: {best_f1:.3f}\n")

    print(f"\nSaved: {iou_path}")
    print(f"Saved: {pr_path}")
    print(f"Saved: {summary_path}")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="IoU validation of SAM2 masks vs GT_MIVOS")
    parser.add_argument("--sam2_masks",      required=True, help="Path to headless_output_lek4/")
    parser.add_argument("--gt_masks",        required=True, help="Path to GT_MIVOS upscaled_masks/")
    parser.add_argument("--quality_scores",  help="Path to quality_scores.csv")
    parser.add_argument("--output_dir",      default="./iou_validation")
    parser.add_argument("--iou_threshold",   type=float, default=0.5)
    parser.add_argument("--n_labels",        type=int, default=4)
    parser.add_argument("--gt_frame_offset", type=int, default=0,
                         help="GT PNG 'N' depicts SAM2/video frame 'N - offset'. "
                              "Determine with calibrate_gt_frame_offset.py.")
    args = parser.parse_args()

    run_validation(
        sam2_dir=args.sam2_masks,
        gt_dir=args.gt_masks,
        quality_scores_csv=args.quality_scores,
        output_dir=args.output_dir,
        iou_threshold=args.iou_threshold,
        n_labels=args.n_labels,
        gt_frame_offset=args.gt_frame_offset,
    )
