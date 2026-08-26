#!/usr/bin/env python3
"""
hyperparam_search.py
====================
Grid search over quality scorer weights and threshold to find the best
combination of (w_logit, w_continuity, w_ref_iou, threshold).

No GPU needed — works entirely on existing CSV files.

Inputs:
  quality_scores.csv  — per-frame scores for each signal (B, A, C)
  iou_per_frame.csv   — ground truth (IoU < 0.5 = genuine error)

Outputs (in output_dir):
  ablation.csv          — each signal alone and each pair
  grid_search.csv       — all weight + threshold combinations ranked by F1
  best_recall.csv       — top combinations optimising recall (>=0.7)
  best_precision.csv    — top combinations optimising precision (>=0.5)
  summary.txt           — human-readable summary of key findings
"""

import csv
import numpy as np
from pathlib import Path
from itertools import product

# ── Paths ─────────────────────────────────────────────────────────────────────
QUALITY_CSV = '/media/hbvcai/HBVCSSD/samannot/headless_output_lek4_v2/headless_window16_quality_scores.csv'
IOU_CSV     = '/media/hbvcai/HBVCSSD/samannot/iou_validation_v4/iou_per_frame.csv'
OUT_DIR     = Path('/media/hbvcai/HBVCSSD/samannot/hyperparam_search')
OUT_DIR.mkdir(exist_ok=True)

IOU_ERROR_THRESHOLD = 0.5   # IoU below this = genuine tracking error

# ── Load ground truth from IoU CSV ───────────────────────────────────────────
print("Loading ground truth from IoU CSV...")
gt = {}  # (frame, sam_label) -> is_error (1=error, 0=correct)
with open(IOU_CSV) as f:
    for row in csv.DictReader(f):
        frame     = int(row['frame_idx'])
        sam_label = int(row['sam2_label'])
        iou       = float(row['iou'])
        gt[(frame, sam_label)] = 1 if iou < IOU_ERROR_THRESHOLD else 0

print(f"  GT pairs loaded: {len(gt)}")
print(f"  Genuine errors:  {sum(gt.values())} ({100*sum(gt.values())/len(gt):.1f}%)")

# ── Load individual signal scores ─────────────────────────────────────────────
print("\nLoading quality scores...")
scores = []  # list of dicts with frame, label_idx, B, A, C, gt_label
label_name_to_idx = {}

with open(QUALITY_CSV) as f:
    reader = csv.DictReader(f)
    for row in reader:
        frame = int(row['abs_frame'])
        label_name = row['label']

        # Map label name to index
        if label_name not in label_name_to_idx:
            label_name_to_idx[label_name] = len(label_name_to_idx) + 1
        label_idx = label_name_to_idx[label_name]

        # Get individual signal scores
        try:
            b_score = float(row['logit_score']) if row['logit_score'] != 'N/A' else None
            a_score = float(row['continuity_score'])
            c_score = float(row['ref_iou_score']) if row.get('ref_iou_score', 'N/A') != 'N/A' else None
        except (ValueError, KeyError):
            continue

        # Look up ground truth
        key = (frame, label_idx)
        if key not in gt:
            continue

        scores.append({
            'frame':     frame,
            'label_idx': label_idx,
            'B':         b_score,   # logit confidence (None if unavailable)
            'A':         a_score,   # continuity
            'C':         c_score,   # reference IoU (None if unavailable)
            'gt':        gt[key],   # 1=error, 0=correct
        })

print(f"  Matched pairs:   {len(scores)}")
print(f"  Label mapping:   {label_name_to_idx}")

# Check signal availability
has_B = sum(1 for s in scores if s['B'] is not None)
has_C = sum(1 for s in scores if s['C'] is not None)
print(f"  Signal B available: {has_B}/{len(scores)}")
print(f"  Signal C available: {has_C}/{len(scores)}")

# Fill missing signals with neutral value (1.0 = no flag contribution)
for s in scores:
    if s['B'] is None: s['B'] = 1.0
    if s['C'] is None: s['C'] = 1.0

gt_labels = np.array([s['gt'] for s in scores])
B_scores  = np.array([s['B']  for s in scores])
A_scores  = np.array([s['A']  for s in scores])
C_scores  = np.array([s['C']  for s in scores])

total_errors  = gt_labels.sum()
total_correct = len(gt_labels) - total_errors
print(f"\n  Total pairs in analysis: {len(scores)}")
print(f"  Genuine errors:          {total_errors}")
print(f"  Clean frames:            {total_correct}")

# ── Precision / Recall / F1 calculator ───────────────────────────────────────
def compute_metrics(combined_scores, threshold, gt_labels):
    flagged = (combined_scores < threshold).astype(int)
    tp = int(((flagged == 1) & (gt_labels == 1)).sum())
    fp = int(((flagged == 1) & (gt_labels == 0)).sum())
    fn = int(((flagged == 0) & (gt_labels == 1)).sum())
    tn = int(((flagged == 0) & (gt_labels == 0)).sum())
    n_flagged = int(flagged.sum())
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1        = 2*precision*recall / (precision+recall) if (precision+recall) > 0 else 0.0
    return precision, recall, f1, n_flagged, tp, fp, fn, tn

# ── PART 1: Ablation study ────────────────────────────────────────────────────
print("\n" + "="*60)
print("PART 1: ABLATION STUDY")
print("="*60)

THRESHOLDS = [round(t*0.05, 2) for t in range(2, 20)]  # 0.10 to 0.95

ablation_configs = [
    ('B only',     1.0, 0.0, 0.0),
    ('A only',     0.0, 1.0, 0.0),
    ('C only',     0.0, 0.0, 1.0),
    ('A + B',      0.5, 0.5, 0.0),
    ('A + C',      0.0, 0.5, 0.5),
    ('B + C',      0.5, 0.0, 0.5),
    ('A + B + C',  0.33, 0.33, 0.34),  # equal weights
    ('Current (0.4/0.3/0.3)', 0.4, 0.3, 0.3),
]

ablation_results = []
print(f"\n{'Config':<25} {'Best thresh':>12} {'Precision':>10} {'Recall':>8} {'F1':>8} {'Flagged':>8}")
print("-"*75)

for name, wb, wa, wc in ablation_configs:
    combined = wb*B_scores + wa*A_scores + wc*C_scores
    best = {'f1': -1}
    for thresh in THRESHOLDS:
        p, r, f1, nf, tp, fp, fn, tn = compute_metrics(combined, thresh, gt_labels)
        if f1 > best['f1']:
            best = {'thresh':thresh, 'p':p, 'r':r, 'f1':f1, 'nf':nf,
                    'tp':tp, 'fp':fp, 'fn':fn, 'tn':tn, 'wb':wb, 'wa':wa, 'wc':wc, 'name':name}
    ablation_results.append(best)
    print(f"{name:<25} {best['thresh']:>12.2f} {best['p']:>10.3f} {best['r']:>8.3f} {best['f1']:>8.3f} {best['nf']:>8,}")

# Save ablation CSV
with open(OUT_DIR/'ablation.csv', 'w', newline='') as f:
    w = csv.writer(f)
    w.writerow(['config','w_logit','w_cont','w_ref','best_threshold','precision','recall','f1','n_flagged','tp','fp','fn','tn'])
    for r in ablation_results:
        w.writerow([r['name'],r['wb'],r['wa'],r['wc'],r['thresh'],
                    round(r['p'],4),round(r['r'],4),round(r['f1'],4),
                    r['nf'],r['tp'],r['fp'],r['fn'],r['tn']])
print(f"\nSaved: {OUT_DIR/'ablation.csv'}")

# ── PART 2: Full grid search ──────────────────────────────────────────────────
print("\n" + "="*60)
print("PART 2: GRID SEARCH — weights + threshold")
print("="*60)

# Generate all weight combinations where w_B + w_A + w_C = 1.0
# in steps of 0.1
weight_combos = []
for wb in range(0, 11):
    for wa in range(0, 11-wb):
        wc = 10 - wb - wa
        weight_combos.append((wb/10, wa/10, wc/10))

print(f"Weight combinations: {len(weight_combos)}")
print(f"Thresholds:          {len(THRESHOLDS)}")
print(f"Total evaluations:   {len(weight_combos)*len(THRESHOLDS)}")

all_results = []
for wb, wa, wc in weight_combos:
    combined = wb*B_scores + wa*A_scores + wc*C_scores
    for thresh in THRESHOLDS:
        p, r, f1, nf, tp, fp, fn, tn = compute_metrics(combined, thresh, gt_labels)
        all_results.append({
            'w_logit':   wb, 'w_cont': wa, 'w_ref': wc,
            'threshold': thresh,
            'precision': round(p, 4), 'recall': round(r, 4), 'f1': round(f1, 4),
            'n_flagged': nf, 'tp': tp, 'fp': fp, 'fn': fn, 'tn': tn,
        })

# Sort by F1 descending
all_results.sort(key=lambda r: r['f1'], reverse=True)

# Save full grid search
with open(OUT_DIR/'grid_search.csv', 'w', newline='') as f:
    w = csv.writer(f)
    w.writerow(['w_logit','w_cont','w_ref','threshold','precision','recall','f1','n_flagged','tp','fp','fn','tn'])
    for r in all_results:
        w.writerow([r['w_logit'],r['w_cont'],r['w_ref'],r['threshold'],
                    r['precision'],r['recall'],r['f1'],r['n_flagged'],
                    r['tp'],r['fp'],r['fn'],r['tn']])
print(f"Saved: {OUT_DIR/'grid_search.csv'}")

# Top 10 by F1
print(f"\nTop 10 by F1:")
print(f"{'w_B':>6} {'w_A':>6} {'w_C':>6} {'Thresh':>8} {'Precision':>10} {'Recall':>8} {'F1':>8} {'Flagged':>8}")
print("-"*65)
for r in all_results[:10]:
    print(f"{r['w_logit']:>6.1f} {r['w_cont']:>6.1f} {r['w_ref']:>6.1f} "
          f"{r['threshold']:>8.2f} {r['precision']:>10.3f} {r['recall']:>8.3f} "
          f"{r['f1']:>8.3f} {r['n_flagged']:>8,}")

# ── PART 3: Best for recall (priority) ───────────────────────────────────────
print("\n" + "="*60)
print("PART 3: BEST FOR RECALL (recall >= 0.70, then max F1)")
print("="*60)

recall_priority = [r for r in all_results if r['recall'] >= 0.70]
recall_priority.sort(key=lambda r: (r['f1'], r['precision']), reverse=True)

print(f"\nCombinations with recall >= 0.70: {len(recall_priority)}")
print(f"\nTop 10:")
print(f"{'w_B':>6} {'w_A':>6} {'w_C':>6} {'Thresh':>8} {'Precision':>10} {'Recall':>8} {'F1':>8} {'Flagged':>8}")
print("-"*65)
for r in recall_priority[:10]:
    print(f"{r['w_logit']:>6.1f} {r['w_cont']:>6.1f} {r['w_ref']:>6.1f} "
          f"{r['threshold']:>8.2f} {r['precision']:>10.3f} {r['recall']:>8.3f} "
          f"{r['f1']:>8.3f} {r['n_flagged']:>8,}")

with open(OUT_DIR/'best_recall.csv', 'w', newline='') as f:
    w = csv.writer(f)
    w.writerow(['w_logit','w_cont','w_ref','threshold','precision','recall','f1','n_flagged'])
    for r in recall_priority[:50]:
        w.writerow([r['w_logit'],r['w_cont'],r['w_ref'],r['threshold'],
                    r['precision'],r['recall'],r['f1'],r['n_flagged']])
print(f"Saved: {OUT_DIR/'best_recall.csv'}")

# ── PART 4: Best precision while keeping recall >= 0.5 ───────────────────────
print("\n" + "="*60)
print("PART 4: BEST PRECISION (recall >= 0.50, then max precision)")
print("="*60)

prec_priority = [r for r in all_results if r['recall'] >= 0.50]
prec_priority.sort(key=lambda r: (r['precision'], r['f1']), reverse=True)

print(f"\nTop 10:")
print(f"{'w_B':>6} {'w_A':>6} {'w_C':>6} {'Thresh':>8} {'Precision':>10} {'Recall':>8} {'F1':>8} {'Flagged':>8}")
print("-"*65)
for r in prec_priority[:10]:
    print(f"{r['w_logit']:>6.1f} {r['w_cont']:>6.1f} {r['w_ref']:>6.1f} "
          f"{r['threshold']:>8.2f} {r['precision']:>10.3f} {r['recall']:>8.3f} "
          f"{r['f1']:>8.3f} {r['n_flagged']:>8,}")

with open(OUT_DIR/'best_precision.csv', 'w', newline='') as f:
    w = csv.writer(f)
    w.writerow(['w_logit','w_cont','w_ref','threshold','precision','recall','f1','n_flagged'])
    for r in prec_priority[:50]:
        w.writerow([r['w_logit'],r['w_cont'],r['w_ref'],r['threshold'],
                    r['precision'],r['recall'],r['f1'],r['n_flagged']])
print(f"Saved: {OUT_DIR/'best_precision.csv'}")

# ── PART 5: Summary ───────────────────────────────────────────────────────────
best_f1       = all_results[0]
best_recall_r = recall_priority[0] if recall_priority else None
best_prec_r   = prec_priority[0]   if prec_priority  else None
current       = next((r for r in all_results
                      if r['w_logit']==0.4 and r['w_cont']==0.3
                      and r['w_ref']==0.3 and r['threshold']==0.35), None)

summary = f"""
HYPERPARAMETER SEARCH SUMMARY
==============================
Dataset:    Lek4 video — {len(scores):,} frame-label pairs
GT errors:  {int(total_errors):,} ({100*total_errors/len(scores):.1f}%)
IoU error threshold: {IOU_ERROR_THRESHOLD}

ABLATION RESULTS (best threshold per config)
---------------------------------------------
Each signal alone and each pair — see ablation.csv for full table.

GRID SEARCH
-----------
Weight combinations tested: {len(weight_combos)}
Thresholds tested:          {len(THRESHOLDS)}
Total evaluations:          {len(all_results):,}

CURRENT SETTINGS (w_B=0.4, w_A=0.3, w_C=0.3, threshold=0.35):
  Precision: {current['precision']:.3f}
  Recall:    {current['recall']:.3f}
  F1:        {current['f1']:.3f}
  Flagged:   {current['n_flagged']:,}

BEST BY F1:
  Weights:   w_B={best_f1['w_logit']}, w_A={best_f1['w_cont']}, w_C={best_f1['w_ref']}
  Threshold: {best_f1['threshold']}
  Precision: {best_f1['precision']:.3f}
  Recall:    {best_f1['recall']:.3f}
  F1:        {best_f1['f1']:.3f}
  Flagged:   {best_f1['n_flagged']:,}

BEST FOR RECALL (recall >= 0.70, highest F1):
  Weights:   w_B={best_recall_r['w_logit']}, w_A={best_recall_r['w_cont']}, w_C={best_recall_r['w_ref']}
  Threshold: {best_recall_r['threshold']}
  Precision: {best_recall_r['precision']:.3f}
  Recall:    {best_recall_r['recall']:.3f}
  F1:        {best_recall_r['f1']:.3f}
  Flagged:   {best_recall_r['n_flagged']:,}

BEST PRECISION (recall >= 0.50):
  Weights:   w_B={best_prec_r['w_logit']}, w_A={best_prec_r['w_cont']}, w_C={best_prec_r['w_ref']}
  Threshold: {best_prec_r['threshold']}
  Precision: {best_prec_r['precision']:.3f}
  Recall:    {best_prec_r['recall']:.3f}
  F1:        {best_prec_r['f1']:.3f}
  Flagged:   {best_prec_r['n_flagged']:,}

Output files:
  ablation.csv       — signal ablation study
  grid_search.csv    — all combinations ranked by F1
  best_recall.csv    — top combinations with recall >= 0.70
  best_precision.csv — top combinations with recall >= 0.50
"""

print(summary)
with open(OUT_DIR/'summary.txt', 'w') as f:
    f.write(summary)
print(f"Saved: {OUT_DIR/'summary.txt'}")
print(f"\nAll outputs in: {OUT_DIR}")
