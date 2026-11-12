"""Compare the 502m pilot predictions with the reviewed per-bird session masks."""
import argparse
import csv
import json
import math
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.stats import rankdata

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "audit"))
from recompute import SessionReader, sha, write_csv, write_json

BIRDS = ("bird1", "bird2", "bird3", "bird4")


def evaluate(run):
    pilot = run.parent
    manifest = json.loads((run / "run.json").read_text())
    if (manifest["status"], manifest["frames_written"], manifest["prompt_index"]) != ("completed", 375, 87):
        raise ValueError("Expected a complete 375-frame pilot prompted at index 87")
    keys = {(i, bird) for i in range(375) for bird in BIRDS}
    with (run / "scores.csv").open(newline="") as stream:
        scores = {}
        for row in csv.DictReader(stream):
            key = int(row["frame_index"]), row["bird"]
            if key in scores or int(row["is_prompt"]) != int(key[0] == 87):
                raise ValueError(f"Duplicate key or wrong prompt marker: {key}")
            scores[key] = row
    if set(scores) != keys:
        raise ValueError("Score table does not cover the expected 1500 keys")
    ref = pilot / "reviewed_reference/Session.pkl"
    print("Reading reviewed per-bird masks from Session.pkl...", flush=True)
    with ref.open("rb") as stream:
        session = SessionReader(stream).load()
    masks = session.pop("masks")
    del session
    truth = {}
    for filename, labels in masks.items():
        path = Path(filename)
        index = int(path.stem)
        if index in truth or path.resolve() != (pilot / "frames" / f"{index:06d}.jpg").resolve():
            raise ValueError(f"Unexpected reviewed source image: {filename}")
        truth[index] = labels
    if set(truth) != set(range(375)):
        raise ValueError("Reviewed session has incomplete frame coverage")
    rows = []
    for i, bird in sorted(keys):
        if i == 87:
            continue
        score = scores[i, bird]
        conf = float(score["foreground_confidence"]) if score["foreground_confidence"] else None
        with Image.open(run / "masks" / bird / f"{i:06d}.png") as image:
            if image.mode != "1":
                raise ValueError(f"Expected binary prediction at {i}, {bird}")
            pred = np.asarray(image)
        area = int(pred.sum())
        if (area != int(score["area_px"]) or (area == 0) != (conf is None)
                or (conf is not None and not 0.5 <= conf <= 1)):
            raise ValueError(f"Inconsistent score or predicted mask at {i}, {bird}")
        packed = truth[i].get(bird)
        gt_area = iou = error = None
        if packed is None:
            status = "missing_reference_entry"
        else:
            if (packed.h, packed.w) != pred.shape or packed.data.shape != (packed.h, (packed.w + 7) // 8):
                raise ValueError(f"Reference dimension mismatch at {i}, {bird}")
            gt = np.unpackbits(packed.data, axis=-1, count=packed.w).view(np.bool_)
            gt_area = int(gt.sum())
            if gt_area:
                intersection = int(np.count_nonzero(gt & pred))
                iou = intersection / (area + gt_area - intersection)
                error = int(iou < 0.5)
                status = "missing_prediction" if not area else ("low_iou" if error else "matched")
            else:
                error = int(area > 0)
                status = "false_presence" if area else "both_absent"
        rows.append(dict(frame_index=i, bird=bird, direction=score["direction"], predicted_pixels=area,
                         reference_pixels=gt_area, iou=iou, error=error, status=status, confidence=conf))
    eligible = [r for r in rows if r["error"] is not None]
    scored = [r for r in eligible if r["confidence"] is not None]
    ious = [r["iou"] for r in eligible if r["iou"] is not None]
    errors = sum(r["error"] for r in eligible)
    positives = sum(r["error"] for r in scored)
    negatives = len(scored) - positives
    auc = None
    if positives and negatives:
        ranks = rankdata([-r["confidence"] for r in scored])
        rank_sum = sum(rank for rank, r in zip(ranks, scored) if r["error"])
        auc = float((rank_sum - positives * (positives + 1) / 2) / (positives * negatives))
    summary = dict(prompt_frame_excluded=87, nonprompt_pairs=len(rows), eligible_pairs=len(eligible),
                   counts=dict(Counter(r["status"] for r in rows)), errors=errors,
                   error_rate=errors / len(eligible) if eligible else None,
                   mean_iou_gt_present=float(np.mean(ious)) if ious else None, gt_present_pairs=len(ious),
                   confidence_available=len(scored), errors_with_confidence=positives,
                   errors_without_confidence=errors-positives, low_confidence_error_auroc=auc,
                   iou_error_threshold=0.5, reference_sha256=sha(ref))
    out = run / ("evaluation-" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ"))
    out.mkdir()
    write_csv(out / "pairs.csv", rows)
    write_json(out / "summary.json", summary)
    print(json.dumps(summary, indent=2))
    print("Evaluation saved to", out)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    evaluate(parser.parse_args().run.resolve())
