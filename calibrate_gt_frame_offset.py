#!/usr/bin/env python3
"""
calibrate_gt_frame_offset.py
=============================
Finds the integer offset between SAM2 headless_output frame numbering and
GT_MIVOS frame numbering.

SAM2 headless_output/NNNNNN.png is frame-accurate by construction — it is
written from the same 0-indexed cv2.VideoCapture reads used to seek the raw
video. GT_MIVOS/NNNNNN.png comes from an external annotation tool and may
use a different starting frame (e.g. 1-indexed export), so GT frame NNNNNN
can actually depict video frame NNNNNN + offset in SAM2's numbering.

This compares combined foreground masks (any tracked bird vs background)
between SAM2 and GT across a sample of frames for a range of candidate
offsets, and reports the offset with the highest mean IoU. That offset is
the one to plug into GT_FRAME_OFFSET in visual_proof.py and
--gt_frame_offset in iou_validator.py.

Usage:
    python3 calibrate_gt_frame_offset.py \
        --sam2_masks /media/hbvcai/HBVCSSD/samannot/headless_output_lek4_v2 \
        --gt_masks   "/media/.../GT_MIVOS/.../upscaled_masks" \
        --gt_bird_ids 14 38 75 113 \
        --search -5 5
"""

import argparse
import numpy as np
from pathlib import Path

try:
    import cv2
except ImportError:
    print("OpenCV not found. Run: pip install opencv-python")
    raise SystemExit(1)


def combined_sam2_mask(png_path):
    img = cv2.imread(str(png_path))
    if img is None:
        return None
    return np.any(img > 10, axis=2)


def combined_gt_mask(png_path, gt_bird_ids):
    img = cv2.imread(str(png_path))
    if img is None:
        return None
    gray = img[:, :, 0]
    mask = np.zeros(gray.shape, dtype=bool)
    for bird_id in gt_bird_ids:
        mask |= (gray == bird_id)
    return mask


def iou(mask_a, mask_b):
    if mask_a.shape != mask_b.shape:
        mask_b = cv2.resize(
            mask_b.astype(np.uint8),
            (mask_a.shape[1], mask_a.shape[0]),
            interpolation=cv2.INTER_NEAREST,
        ).astype(bool)
    union = (mask_a | mask_b).sum()
    if union == 0:
        return None  # both empty at this frame — uninformative, skip
    return float((mask_a & mask_b).sum()) / float(union)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sam2_masks", required=True)
    ap.add_argument("--gt_masks", required=True)
    ap.add_argument("--gt_bird_ids", type=int, nargs="+", required=True)
    ap.add_argument("--search", type=int, nargs=2, default=[-5, 5], metavar=("MIN", "MAX"))
    ap.add_argument("--sample_every", type=int, default=20, help="Use every Nth SAM2 frame as a sample point")
    ap.add_argument("--max_samples", type=int, default=300)
    args = ap.parse_args()

    sam2_dir = Path(args.sam2_masks)
    gt_dir = Path(args.gt_masks)

    sam2_files = {int(f.stem): f for f in sam2_dir.glob("*.png") if f.stem.isdigit()}
    gt_files = {int(f.stem): f for f in gt_dir.glob("*.png") if f.stem.isdigit()}
    if not sam2_files or not gt_files:
        print("ERROR: no numbered PNGs found in one of the two directories.")
        raise SystemExit(1)

    sample_frames = sorted(sam2_files.keys())[::args.sample_every][: args.max_samples]
    print(f"SAM2 frames: {len(sam2_files)} (range {min(sam2_files)}-{max(sam2_files)})")
    print(f"GT frames:   {len(gt_files)} (range {min(gt_files)}-{max(gt_files)})")
    print(f"Sampling {len(sample_frames)} SAM2 frames, testing offsets {args.search[0]}..{args.search[1]}\n")

    results = {}
    for offset in range(args.search[0], args.search[1] + 1):
        ious = []
        for frame_idx in sample_frames:
            gt_idx = frame_idx + offset
            if gt_idx not in gt_files:
                continue
            sam2_mask = combined_sam2_mask(sam2_files[frame_idx])
            gt_mask = combined_gt_mask(gt_files[gt_idx], args.gt_bird_ids)
            if sam2_mask is None or gt_mask is None:
                continue
            val = iou(sam2_mask, gt_mask)
            if val is not None:
                ious.append(val)
        if ious:
            results[offset] = (float(np.mean(ious)), len(ious))

    if not results:
        print("No overlapping frames found for any offset — check the directories/IDs.")
        raise SystemExit(1)

    print(f"{'offset':>8} {'mean IoU':>10} {'n samples':>10}")
    print("-" * 32)
    best_offset = max(results, key=lambda k: results[k][0])
    for offset in sorted(results):
        mean_iou, n = results[offset]
        marker = "  <- best" if offset == best_offset else ""
        print(f"{offset:>8} {mean_iou:>10.3f} {n:>10}{marker}")

    print(f"\nBest offset: {best_offset} (mean IoU={results[best_offset][0]:.3f})")
    print("If this is not clearly better than offset 0, the mismatch may not be")
    print("a simple constant frame offset (check fps / trim / spatial alignment instead).")
    print(f"\nApply it as GT_FRAME_OFFSET = {best_offset} in visual_proof.py")
    print(f"and --gt_frame_offset {best_offset} in iou_validator.py.")


if __name__ == "__main__":
    main()
