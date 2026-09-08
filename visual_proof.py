#!/usr/bin/env python3
"""
visual_proof.py
===============
Generates side-by-side comparison images showing SAM2 tracking errors
vs professional GT_MIVOS ground truth masks.

Left panel:  raw frame + SAM2 mask for the error bird (colored) + other birds dimmed
Right panel: raw frame + GT mask for the error bird (bright green outline)

Output: ~/Desktop/visual_proof/frame_XXXXXX_bird_XX_iou_X.XXX.jpg
"""

import csv
import cv2
import numpy as np
from pathlib import Path

# ── Paths ─────────────────────────────────────────────────────────────────────
SAM2_DIR    = Path('/media/hbvcai/HBVCSSD/samannot/headless_output_lek4_v2')
GT_DIR      = Path('/media/hbvcai/HATTER_4TB/MPI_data/MaxPlanckData/ruffs_2022/males_annotation/test_data/GT_MIVOS/cut_Lek4_right_2021_604_00-00-00_00-02-32/upscaled_masks')
# GT masks were annotated on these pre-extracted PNG frames, not on the raw
# video — decoding the video directly (cv2.VideoCapture) yields a spatially
# different frame (codec differences, max pixel diff observed = 207), which
# broke overlay alignment. Read background frames from here instead.
FRAMES_DIR  = Path('/media/hbvcai/HATTER_4TB/MPI_data/MaxPlanckData/ruffs_2022/males_annotation/test_data/FRAMES/cut_Lek4_right_2021_604_00-00-00_00-02-32')
IOU_CSV     = '/media/hbvcai/HBVCSSD/samannot/iou_validation_v4/iou_per_frame.csv'
OUT_DIR     = Path('/home/hbvcai/Desktop/visual_proof')
OUT_DIR.mkdir(exist_ok=True)

# ── Pascal VOC colormap — matches SAMannot output ─────────────────────────────
def pascal_voc_color(label_idx):
    r = g = b = 0
    c = label_idx
    for j in range(8):
        r |= ((c >> 0) & 1) << (7 - j)
        g |= ((c >> 1) & 1) << (7 - j)
        b |= ((c >> 2) & 1) << (7 - j)
        c >>= 3
    return (b, g, r)  # OpenCV uses BGR

# SAM2 label index → color (BGR)
LABEL_COLORS = {i: pascal_voc_color(i) for i in range(1, 5)}

# GT bird IDs present in this video
GT_BIRD_IDS = [14, 38, 75, 113]

# ── Load worst IoU records ────────────────────────────────────────────────────
print("Loading IoU records...")
records = []
with open(IOU_CSV) as f:
    for row in csv.DictReader(f):
        iou = float(row['iou'])
        if 0.0 < iou < 0.15:   # clear errors but not dropout (IoU=0 = bird absent)
            records.append({
                'frame':   int(row['frame_idx']),
                'gt_id':   int(row['gt_bird_id']),
                'sam_label': int(row['sam2_label']),
                'iou':     iou,
            })

print(f"Found {len(records)} clear error records (0 < IoU < 0.15)")

# Sort by IoU ascending (worst first), then pick 10 spread across the video
records.sort(key=lambda r: r['iou'])

# Pick frames spread across the video (not all clustered in one spot)
selected = []
used_frames = set()
# First pass: worst errors, one per 300-frame window
for r in records:
    window = r['frame'] // 300
    key = (window, r['gt_id'])
    if key not in used_frames and len(selected) < 12:
        selected.append(r)
        used_frames.add(key)

# Fill to 10 if needed
for r in records:
    if r['frame'] not in {s['frame'] for s in selected} and len(selected) < 10:
        selected.append(r)

selected = sorted(selected[:10], key=lambda r: r['frame'])
print(f"Selected {len(selected)} frames:")
for s in selected:
    print(f"  frame={s['frame']:5d}  gt_bird={s['gt_id']:3d}  sam_label={s['sam_label']}  IoU={s['iou']:.3f}")

TARGET_W = 960
TARGET_H = 540

def resize(img, w=TARGET_W, h=TARGET_H):
    return cv2.resize(img, (w, h))

def add_label(img, text, y=40, color=(255,255,255)):
    cv2.putText(img, text, (14, y), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0,0,0), 4)
    cv2.putText(img, text, (14, y), cv2.FONT_HERSHEY_SIMPLEX, 1.1, color, 2)

# ── Generate comparison images ────────────────────────────────────────────────
saved = 0
for rec in selected:
    frame_idx = rec['frame']
    gt_id     = rec['gt_id']
    sam_label = rec['sam_label']
    iou_val   = rec['iou']

    # Read raw background frame — must come from the same PNG frames the GT
    # was annotated on (see FRAMES_DIR note above), not the video.
    frame_path = FRAMES_DIR / f'{frame_idx:06d}.png'
    raw = cv2.imread(str(frame_path))
    if raw is None:
        print(f"  Frame {frame_idx}: could not read from FRAMES dir — skipping")
        continue

    # Read SAM2 mask PNG
    sam2_path = SAM2_DIR / f'{frame_idx:06d}.png'
    sam2_img = cv2.imread(str(sam2_path))
    if sam2_img is None:
        print(f"  Frame {frame_idx}: SAM2 mask not found — skipping")
        continue

    # Read GT mask PNG — GT frame numbering matches FRAMES/SAM2 numbering
    # exactly, no offset needed.
    gt_path = GT_DIR / f'{frame_idx:06d}.png'
    gt_img = cv2.imread(str(gt_path))
    if gt_img is None:
        print(f"  Frame {frame_idx}: GT mask not found — skipping")
        continue

    # Resize everything. GT is a label-ID image (pixel value = bird ID), so it
    # must use nearest-neighbor interpolation — linear (cv2's default) blends
    # adjacent bird IDs at mask boundaries and corrupts the exact-value match
    # done below (gt_gray == gt_id).
    raw      = resize(raw)
    sam2_img = resize(sam2_img)
    gt_gray  = cv2.resize(gt_img[:, :, 0], (TARGET_W, TARGET_H), interpolation=cv2.INTER_NEAREST)

    H, W = raw.shape[:2]

    # ── LEFT PANEL: SAM2 overlay ──────────────────────────────────────────────
    left = raw.copy()

    # Dim all SAM2 masks first (other birds at 20% opacity)
    all_mask_px = np.any(sam2_img > 10, axis=2)
    left[all_mask_px] = (left[all_mask_px] * 0.6).astype(np.uint8)

    # Highlight the error bird's SAM2 mask at full opacity
    error_color = LABEL_COLORS.get(sam_label, (0, 255, 255))
    # Find pixels of this specific label color in SAM2 output
    ref_color = np.array(pascal_voc_color(sam_label))  # BGR
    error_px = np.all(sam2_img == ref_color, axis=2)
    if error_px.sum() > 0:
        overlay = left.copy()
        overlay[error_px] = error_color
        left = cv2.addWeighted(left, 0.2, overlay, 0.8, 0)
        # Draw outline
        contours, _ = cv2.findContours(error_px.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(left, contours, -1, error_color, 2)

    add_label(left, f"SAM2 tracking (bird {sam_label})", color=(255,200,50))

    # ── RIGHT PANEL: GT overlay ────────────────────────────────────────────────
    right = raw.copy()

    # Show where GT says the bird actually is (bright green)
    gt_mask = (gt_gray == gt_id)
    if gt_mask.sum() > 0:
        overlay = right.copy()
        overlay[gt_mask] = (0, 220, 80)   # bright green
        right = cv2.addWeighted(right, 0.3, overlay, 0.7, 0)
        # Draw outline
        contours, _ = cv2.findContours(gt_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(right, contours, -1, (0, 255, 60), 3)

    add_label(right, f"Ground truth (bird {gt_id})", color=(80, 255, 80))

    # ── Combine and add caption ───────────────────────────────────────────────
    # Divider line
    divider = np.zeros((H, 6, 3), dtype=np.uint8)
    divider[:] = (60, 160, 140)   # teal divider

    combined = np.hstack([left, divider, right])

    # Caption bar at bottom
    caption_h = 52
    caption = np.zeros((caption_h, combined.shape[1], 3), dtype=np.uint8)
    caption[:] = (30, 45, 70)
    caption_text = f"Frame {frame_idx:,}  |  IoU = {iou_val:.3f}  |  SAM2 label {sam_label} vs GT bird {gt_id}  |  Low IoU = tracking error"
    cv2.putText(caption, caption_text, (14, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 220, 255), 2)

    final = np.vstack([combined, caption])

    fname = OUT_DIR / f'frame_{frame_idx:06d}_bird{gt_id}_iou{iou_val:.3f}.jpg'
    cv2.imwrite(str(fname), final, [cv2.IMWRITE_JPEG_QUALITY, 92])
    print(f"  Saved: {fname.name}")
    saved += 1

print(f"\nDone — {saved} images saved to {OUT_DIR}")
