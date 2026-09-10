#!/usr/bin/env bash
set -euo pipefail

# Run from any folder: all new files stay inside this audit checkout.
AUDIT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
CHECKOUT_DIR="$(dirname -- "$AUDIT_DIR")"
WORKSPACE_DIR="$(dirname -- "$CHECKOUT_DIR")"
DATA_ROOT="$(dirname -- "$WORKSPACE_DIR")"
mkdir -p "$AUDIT_DIR/.tmp"
export TMPDIR="$AUDIT_DIR/.tmp"
GT_ROOT="/media/hbvcai/HATTER_4TB/MPI_data/MaxPlanckData/ruffs_2022/males_annotation/test_data"
CLIP="cut_Lek4_right_2021_604_00-00-00_00-02-32"
SESSION_PATH="$DATA_ROOT/Session_Lek4_baseline.pkl"
if [[ ! -f "$SESSION_PATH" ]]; then
    SESSION_PATH="$GT_ROOT/Session_Lek4_baseline.pkl"
fi
if [[ ! -f "$SESSION_PATH" ]]; then
    echo "Session_Lek4_baseline.pkl not found in the two documented locations. Paste this message back."
    exit 1
fi
CORRESPONDENCE_DIR="$AUDIT_DIR/results/recompute-20260909T145908Z-full-correspondence"
RUN_DIR="$AUDIT_DIR/results/aligned-$(date -u +%Y%m%dT%H%M%SZ)"
python -m unittest discover -s "$AUDIT_DIR" -p 'test_*.py'
python "$AUDIT_DIR/recompute.py" \
    --workspace "$WORKSPACE_DIR" \
    --correspondence "$CORRESPONDENCE_DIR" \
    --session "$SESSION_PATH" \
    --masks "$DATA_ROOT/headless_output_lek4_v2" \
    --gt "$GT_ROOT/GT_MIVOS/$CLIP/upscaled_masks" \
    --frames "$GT_ROOT/FRAMES/$CLIP" \
    --quality "$DATA_ROOT/headless_output_lek4_v2/headless_window16_quality_scores.csv" \
    --legacy-iou "$DATA_ROOT/iou_validation_v4/iou_per_frame.csv" \
    --video "$GT_ROOT/vids/cut_Lek4_right_2021_604_a_lek_00_00_00_to_00_02_32.mp4" \
    --output "$RUN_DIR"
printf '\nResults saved inside the workspace: %s\n' "$RUN_DIR"
printf 'Run the git add/commit/push commands provided in the chat to send them back.\n'
