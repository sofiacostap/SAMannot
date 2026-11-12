"""Run the 502m left-view pilot from its saved four-click prediction session."""
import csv
import os
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "audit"))
from corrected_propagation import foreground_confidence
from recompute import indexed, load_session, sha, write_json

PILOT = ROOT / "experiments/lek3/pilots/502m_left_20m25s_20m40s"
NAME = "502m_left_20m25s_20m40s_prediction_01"


def read_inputs():
    session = load_session(ROOT / "export" / (NAME + ".pkl"))
    frames = indexed(PILOT / "frames")
    if set(frames) != set(range(375)) or session["block_size"] != 375 or session["num_blocks"] != 1:
        raise ValueError("Expected 375 numbered pilot frames and one 375-frame session block.")
    points = session["prompts"]
    if (session["best"] != {0: 87} or len(points) != 4
            or {p["label"] for p in points} != {"bird1", "bird2", "bird3", "bird4"}):
        raise ValueError("Expected one click per bird on image 88 (index 87).")
    with Image.open(frames[87]) as image:
        width, height = image.size
    for p in points:
        if not p["positive"] or not (0 <= p["x"] < width and 0 <= p["y"] < height):
            raise ValueError(f"Invalid point: {p}")
    return frames, session, (height, width)


def run():
    import torch
    from sam2.build_sam import build_sam2_video_predictor

    frames, session, shape = read_inputs()
    checkpoint = ROOT / "checkpoints/sam2.1_hiera_large.pt"
    if not torch.cuda.is_available() or not checkpoint.is_file():
        raise RuntimeError("Lumen CUDA GPU and large SAM2.1 checkpoint are required.")
    out = PILOT / ("scored-" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ"))
    out.mkdir()
    print(f"Results: {out}", flush=True)
    points = session["prompts"]
    names = {p["sam2_label"]: p["label"] for p in points}
    config = "sam2.1_hiera_l.yaml"
    report = dict(status="running", session_sha256=session["session_sha256"], points=points,
                  checkpoint_sha256=sha(checkpoint), config=config, prompt_index=87,
                  direction_method="Independent states in chronological and reversed image order.",
                  score="Mean sigmoid over positive mask logits; blank for an empty mask.",
                  gpu=torch.cuda.get_device_name(), torch=str(torch.__version__))
    write_json(out / "run.json", report)
    seen = set()
    try:
        predictor = build_sam2_video_predictor(config, str(checkpoint), device="cuda")
        report["memory_window_size"] = predictor.memory_window_size
        for name in names.values():
            (out / "masks" / name).mkdir(parents=True)
        with (out / "scores.csv").open("w", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow(["frame_index", "bird", "direction", "is_prompt", "area_px", "foreground_confidence"])
            for direction, indices in [("forward", range(87, 375)), ("backward", range(87, -1, -1))]:
                print(f"Starting {direction} pass ({len(indices)} images)", flush=True)
                with tempfile.TemporaryDirectory(dir=out) as tmp:
                    for local, index in enumerate(indices):
                        os.symlink(frames[index], Path(tmp) / f"{local:06d}.jpg")
                    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
                        state = predictor.init_state(video_path=tmp, offload_video_to_cpu=True, offload_state_to_cpu=True)
                        for p in points:
                            predictor.add_new_points_or_box(
                                inference_state=state, frame_idx=0, obj_id=p["sam2_label"],
                                points=np.array([[p["x"], p["y"]]], dtype=np.float32),
                                labels=np.array([1], dtype=np.int32))
                        for local, ids, logits in predictor.propagate_in_video(state, start_frame_idx=0):
                            if not 0 <= local < len(indices):
                                raise ValueError(f"Unexpected frame {local} in {direction} pass")
                            index = indices[local]
                            if direction == "backward" and index == 87:
                                continue
                            if index in seen or sorted(map(int, ids)) != sorted(names):
                                raise ValueError(f"Duplicate frame or unexpected identities at {index}")
                            for j, obj in enumerate(ids):
                                bird = names[int(obj)]
                                values = logits[j].float().cpu().numpy().squeeze()
                                if values.shape != shape or not np.isfinite(values).all():
                                    raise ValueError(f"Invalid logits at {index} for {bird}")
                                mask = values > 0
                                score = foreground_confidence(values)
                                Image.fromarray(mask).save(out / "masks" / bird / f"{index:06d}.png")
                                writer.writerow([index, bird, direction, int(index == 87), int(mask.sum()), score])
                            seen.add(index)
                            if len(seen) % 25 == 0:
                                stream.flush()
                                print(f"Scored {len(seen)}/375 frames", flush=True)
                        predictor.reset_state(state)
                        del state
                torch.cuda.empty_cache()
        if seen != set(frames):
            raise RuntimeError("Incomplete frame coverage")
        report["status"] = "completed"
    except Exception as exc:
        report.update(status="failed", error=str(exc))
        raise
    finally:
        report["frames_written"] = len(seen)
        write_json(out / "run.json", report)
    print(f"Completed: 375 frames, 1500 frame-bird scores. Results: {out}", flush=True)


if __name__ == "__main__":
    run()
