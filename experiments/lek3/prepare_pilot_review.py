"""Build a portable visual review of bird2 from an evaluated pilot."""
import argparse
import csv
import html
import json
import shutil
from collections import defaultdict
from itertools import groupby
from pathlib import Path

import numpy as np
from PIL import Image

BIRDS = ("bird1", "bird2", "bird3", "bird4")
COLORS = [(220, 55, 55), (44, 195, 79), (218, 186, 24), (74, 114, 239)]


def review(evaluation):
    run = evaluation.parent
    pilot = run.parent
    out = evaluation / "meeting_review"
    out.mkdir(exist_ok=True)
    for name in ("pairs.csv", "summary.json"):
        shutil.copyfile(evaluation / name, out / name)
    with (out / "pairs.csv").open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    by_frame = defaultdict(list)
    for row in rows:
        by_frame[int(row["frame_index"])].append(row)
    bird = {int(r["frame_index"]): r for r in rows if r["bird"] == "bird2"}
    low = sorted(i for i, r in bird.items() if r["status"] == "low_iou")
    if not low:
        raise ValueError("No bird2 low-IoU frames in this evaluation")
    runs = []
    for _, group in groupby(enumerate(low), lambda p: p[1] - p[0]):
        indices = [i for _, i in group]
        runs.append((indices[0], indices[-1], len(indices)))
    first, last, length = max(runs, key=lambda interval: interval[2])
    choices = defaultdict(list)
    for title, i in [("Before", first-1), ("Start", first), ("Middle", (first+last)//2),
                     ("Last low-IoU frame", last), ("After", last+1)]:
        if i in bird:
            choices[i].append(title)
    high = max(low, key=lambda i: float(bird[i]["confidence"]))
    choices[high].append("Highest-confidence bird2 low-IoU example")
    def lowest(i):
        return min((float(r["confidence"]) for r in by_frame[i] if r["confidence"]), default=float("inf"))
    selected = sorted(by_frame, key=lambda i: (lowest(i), i))[:38]
    errors = {i: sum(r["error"] == "1" for r in group) for i, group in by_frame.items()}
    report = dict(frames_reviewed=len(selected), selected_error_frames=sum(errors[i]>0 for i in selected),
                  all_error_frames=sum(n>0 for n in errors.values()), selected_bird_frame_errors=sum(errors[i] for i in selected),
                  total_errors=sum(errors.values()), bird2_low_iou_pairs=len(low), bird2_low_iou_runs=len(runs),
                  longest_run_first_index=first, longest_run_last_index=last, longest_run_frames=length,
                  highest_confidence_error_index=high, highest_confidence_error_score=float(bird[high]["confidence"]),
                  highest_confidence_error_iou=float(bird[high]["iou"]))
    (out/"review_summary.json").write_text(json.dumps(report, indent=2))
    with (out/"bird2_intervals.csv").open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["first_index","last_index","length_frames"])
        writer.writerows(runs)
    sections = []
    for i, reasons in choices.items():
        filename = f"{i:06d}.jpg"
        with Image.open(pilot/"frames"/filename) as photo:
            photo = photo.convert("RGB")
        source = np.asarray(photo)
        overlay = source.copy()
        for name, color in zip(BIRDS, COLORS):
            with Image.open(run/"masks"/name/f"{i:06d}.png") as im:
                mask = np.asarray(im.convert("1"), dtype=bool)
            if mask.shape != source.shape[:2]:
                raise ValueError(f"Shape mismatch at {i}: {name}")
            overlay[mask] = (0.54*source[mask] + 0.46*np.array(color)).astype(np.uint8)
        with Image.open(pilot/"reviewed_reference/Session/overlays"/filename) as im:
            reference = im.convert("RGB")
        if reference.size != photo.size:
            raise ValueError(f"Reference dimension mismatch at {i}")
        panels = []
        for title, panel in [("Photograph", photo), ("Prediction", Image.fromarray(overlay)), ("Reviewed reference", reference)]:
            panel.thumbnail((1440, 900))
            file = f"{i:06d}_{title.split()[0].lower()}.jpg"
            panel.save(out/file, quality=90)
            panels.append(f'<figure><figcaption>{title}</figcaption><a href="{file}"><img src="{file}"></a></figure>')
        row = bird[i]
        sections.append(f'<section><h2>{html.escape(" / ".join(reasons))}</h2>'
                        f'<p>Image {i+1} in SAMannot · file index {i} · bird2 {row["status"]} · '
                        f'IoU {row["iou"] or "unavailable"} · confidence {row["confidence"] or "unavailable"}</p>'
                        f'<div>{"".join(panels)}</div></section>')
    (out/"index.html").write_text('<meta charset="utf-8"><title>Bird 2 pilot review</title>'
        '<style>body{font:17px system-ui;margin:2vw}div{display:flex;gap:8px}figure{flex:1;margin:0}img{width:100%}'
        'section{margin:2em 0}figcaption{margin-bottom:6px}</style><h1>Bird 2 pilot review</h1>'
        '<p>Bird 1 red · Bird 2 green · Bird 3 yellow · Bird 4 blue. Click an image to enlarge.</p>'
        + ''.join(sections), encoding="utf-8")
    print(json.dumps(report, indent=2))
    print("Open:", out/"index.html")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("evaluation", type=Path)
    review(parser.parse_args().evaluation.resolve())
