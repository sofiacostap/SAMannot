"""
cross_block_propagate.py
========================
Runs SAM2 propagation across ALL blocks using saved SAMannot prompt annotations.

Usage:
    python cross_block_propagate.py \
        --pkl export/Session.pkl \
        --frames_dir export/Session/frames \
        --output_dir ./output_masks \
        --ckpt ./checkpoints/sam2.1_hiera_base_plus.pt \
        --cfg ./sam2_configs/sam2.1_hiera_b+.yaml

Requirements:
    - conda activate samannot
    - SAM2 installed (cd sam2 && pip install -e .)
    - Model checkpoints downloaded
    - A GPU with enough VRAM
"""

import os
import pickle
import argparse
import numpy as np
import torch
import cv2
from pathlib import Path


# ─────────────────────────────────────────────
# STEP 1: Load prompts from SAMannot pkl file
# ─────────────────────────────────────────────

def load_all_prompts(pkl_path: str) -> tuple[list[dict], int]:
    """
    Load all saved annotation prompts from a SAMannot Session.pkl file.
    Converts block-local frame indices to absolute frame indices.

    Returns:
        all_prompts: list of prompt dicts ready for SAM2
        block_size:  the block size used in SAMannot
    """
    with open(pkl_path, 'rb') as f:
        data = pickle.load(f)

    block_size = data['block_size']
    labels     = data['sam_handler_labels']
    obj_id_map = data['sam_handler_object_id_to_label_name']

    print(f"Video:      {data['video_name']}")
    print(f"Block size: {block_size}")
    print(f"Num blocks: {data['num_blocks']}")
    print(f"Labels:     {[l.name for l in labels]}")
    print()

    # Build reverse map: label_name -> obj_id
    name_to_obj_id = {v: k for k, v in obj_id_map.items()}

    all_prompts = []

    for label in labels:
        # Get obj_id for this label — use name lookup, fall back to index
        obj_id = name_to_obj_id.get(label.name, labels.index(label) + 1)

        # Point prompts
        for block_id, pts in label.pts.items():
            block_start = block_id * block_size
            for pt in pts:
                absolute_frame = block_start + pt.idx
                all_prompts.append({
                    'frame_idx':    absolute_frame,
                    'obj_id':       obj_id,
                    'label_name':   label.name,
                    'coords':       np.array([[pt.x, pt.y]], dtype=np.float32),
                    'point_labels': np.array([pt.pt_type],   dtype=np.int32),
                    'box':          None,
                })
                print(f"  [{label.name}] block={block_id} local_frame={pt.idx} "
                      f"-> absolute_frame={absolute_frame}  ({pt.x}, {pt.y})")

        # Box prompts
        for block_id, boxes in label.boxes.items():
            block_start = block_id * block_size
            for box in boxes:
                absolute_frame = block_start + box.idx
                all_prompts.append({
                    'frame_idx':    absolute_frame,
                    'obj_id':       obj_id,
                    'label_name':   label.name,
                    'coords':       np.empty((0, 2), dtype=np.float32),
                    'point_labels': np.empty(0,       dtype=np.int32),
                    'box':          np.array([box.fx, box.fy, box.x, box.y], dtype=np.float32),
                })

    # Sort by frame index so SAM2 receives them in order
    all_prompts.sort(key=lambda p: p['frame_idx'])

    print(f"\nTotal prompts loaded: {len(all_prompts)}")
    print(f"Frames with prompts:  {sorted(set(p['frame_idx'] for p in all_prompts))}")
    return all_prompts, block_size


# ─────────────────────────────────────────────
# STEP 2: Run SAM2 propagation on full video
# ─────────────────────────────────────────────

def run_cross_block_propagation(
    pkl_path:   str,
    frames_dir: str,
    output_dir: str,
    ckpt_path:  str,
    cfg_path:   str,
    visualize:  bool = False,
):
    # --- Load prompts ---
    all_prompts, block_size = load_all_prompts(pkl_path)

    if not all_prompts:
        print("No prompts found in pkl file. Did you save annotations in SAMannot?")
        return

    # --- Set up device ---
    if torch.cuda.is_available():
        device = torch.device('cuda')
        torch.autocast(device_type='cuda', dtype=torch.float16).__enter__()
        if torch.cuda.get_device_properties(0).major >= 8:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
        print(f"\nGPU: {torch.cuda.get_device_name(0)}")
        print(f"VRAM available: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    else:
        device = torch.device('cpu')
        print("WARNING: No GPU found. Propagation will be very slow.")

    # --- Load SAM2 model ---
    from sam2.build_sam import build_sam2_video_predictor
    print("\nLoading SAM2 model...")
    predictor = build_sam2_video_predictor(cfg_path, ckpt_path).to(device)

    # --- Init inference state for ENTIRE video ---
    print(f"Initialising SAM2 on full video frames: {frames_dir}")
    inference_state = predictor.init_state(video_path=frames_dir)

    # --- Feed all prompts into SAM2 ---
    print(f"\nAdding {len(all_prompts)} prompts to SAM2...")
    device_str = 'cuda' if torch.cuda.is_available() else 'cpu'

    with torch.autocast(device_str, dtype=torch.float16):
        for prompt in all_prompts:
            kwargs = dict(
                inference_state=inference_state,
                frame_idx=prompt['frame_idx'],
                obj_id=prompt['obj_id'],
            )
            if prompt['coords'].shape[0] > 0:
                predictor.add_new_points_or_box(
                    **kwargs,
                    points=prompt['coords'],
                    labels=prompt['point_labels'],
                )
            if prompt['box'] is not None:
                predictor.add_new_points_or_box(
                    **kwargs,
                    box=prompt['box'],
                )

    # --- Propagate across entire video ---
    os.makedirs(output_dir, exist_ok=True)
    print(f"\nRunning propagation across full video...")
    print("(This may take a while depending on video length and GPU speed)")

    import time
    start_time = time.time()
    results = {}

    with torch.autocast(device_str, dtype=torch.float16):
        for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(inference_state):
            masks = (out_mask_logits > 0.0).cpu().numpy()
            frame_results = {}

            for j, obj_id in enumerate(out_obj_ids):
                mask = masks[j]
                frame_results[int(obj_id)] = mask
                # Save mask as numpy array
                mask_path = os.path.join(output_dir, f"frame_{out_frame_idx:06d}_obj_{obj_id}.npy")
                np.save(mask_path, mask)

            results[out_frame_idx] = frame_results

            if out_frame_idx % 100 == 0:
                elapsed = time.time() - start_time
                print(f"  Frame {out_frame_idx} ({elapsed:.1f}s elapsed)")

    elapsed = time.time() - start_time
    predictor.reset_state(inference_state)
    torch.cuda.empty_cache()

    print(f"\nDone! {len(results)} frames processed in {elapsed:.1f}s")
    print(f"Masks saved to: {output_dir}")

    # --- Optional visualisation ---
    if visualize:
        visualize_results(frames_dir, results, output_dir)

    return results


# ─────────────────────────────────────────────
# STEP 3: Visualisation
# ─────────────────────────────────────────────

COLORS = [
    (255,  80,  80),   # red
    ( 80, 255,  80),   # green
    ( 80, 130, 255),   # blue
    (255, 255,  80),   # yellow
    (255,  80, 255),   # magenta
    ( 80, 255, 255),   # cyan
]

def visualize_results(frames_dir: str, results: dict, output_dir: str):
    vis_dir = os.path.join(output_dir, 'visualization')
    os.makedirs(vis_dir, exist_ok=True)

    frame_files = sorted([
        f for f in os.listdir(frames_dir)
        if f.lower().endswith(('.jpg', '.jpeg', '.png'))
    ])

    print(f"\nGenerating visualisations...")
    for frame_idx, frame_masks in results.items():
        if frame_idx >= len(frame_files):
            continue

        frame_path = os.path.join(frames_dir, frame_files[frame_idx])
        frame = cv2.imread(frame_path)
        if frame is None:
            continue

        overlay = frame.copy()
        for obj_id, mask in frame_masks.items():
            color = COLORS[(obj_id - 1) % len(COLORS)]
            mask_2d = mask.squeeze()
            overlay[mask_2d > 0] = (
                overlay[mask_2d > 0] * 0.5 + np.array(color) * 0.5
            ).astype(np.uint8)

        out_path = os.path.join(vis_dir, f"frame_{frame_idx:06d}.jpg")
        cv2.imwrite(out_path, overlay)

    print(f"Visualisations saved to: {vis_dir}")


# ─────────────────────────────────────────────
# STEP 4: Quick inspection tool (no GPU needed)
# ─────────────────────────────────────────────

def inspect_pkl(pkl_path: str):
    """
    Just print what's inside the pkl — no GPU needed.
    Useful for checking your annotations on your laptop.
    Run with: python cross_block_propagate.py --inspect --pkl path/to/Session.pkl
    """
    with open(pkl_path, 'rb') as f:
        data = pickle.load(f)

    print(f"Video:      {data['video_name']}")
    print(f"Block size: {data['block_size']}")
    print(f"Num blocks: {data['num_blocks']}")
    print()

    labels = data['sam_handler_labels']
    for label in labels:
        print(f"Label: {label.name}")
        total_pts = sum(len(pts) for pts in label.pts.values())
        print(f"  Total clicks: {total_pts}")
        for block_id, pts in label.pts.items():
            block_start = block_id * data['block_size']
            for pt in pts:
                abs_frame = block_start + pt.idx
                print(f"  block={block_id} local={pt.idx} absolute={abs_frame} "
                      f"xy=({pt.x},{pt.y}) type={'positive' if pt.pt_type==1 else 'negative'}")
        print()


# ─────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Cross-block SAM2 propagation for SAMannot')
    parser.add_argument('--pkl',        required=True,  help='Path to SAMannot Session.pkl file')
    parser.add_argument('--frames_dir', help='Directory containing extracted video frames')
    parser.add_argument('--output_dir', default='./output_masks', help='Where to save output masks')
    parser.add_argument('--ckpt',       default='./checkpoints/sam2.1_hiera_base_plus.pt')
    parser.add_argument('--cfg',        default='./sam2_configs/sam2.1_hiera_b+.yaml')
    parser.add_argument('--visualize',  action='store_true', help='Generate colour overlay images')
    parser.add_argument('--inspect',    action='store_true', help='Just print pkl contents, no GPU needed')
    args = parser.parse_args()

    if args.inspect:
        inspect_pkl(args.pkl)
    else:
        if not args.frames_dir:
            print("Error: --frames_dir is required for propagation")
        else:
            run_cross_block_propagation(
                pkl_path=args.pkl,
                frames_dir=args.frames_dir,
                output_dir=args.output_dir,
                ckpt_path=args.ckpt,
                cfg_path=args.cfg,
                visualize=args.visualize,
            )
