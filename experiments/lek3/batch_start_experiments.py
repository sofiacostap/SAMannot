"""Prepare reviewed point prompts and run independent Lek3 propagation experiments.

Commands: prepare (CPU), run (GPU). Review and edit plan.json before approving.
"""
import argparse
import csv
import json
import os
import sys
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
from scipy.ndimage import distance_transform_edt

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "audit")]
from recompute import SessionReader, sha, write_json
from corrected_propagation import foreground_confidence

BIRDS = ["bird1", "bird2", "bird3", "bird4"]


def unpack(mask):
    return np.unpackbits(mask.data, axis=-1, count=mask.w).astype(bool)


def prepare(args):
    sections = sorted(args.source.glob("part[0-9][0-9]_*"))[:5]
    if len(sections) != 5:
        raise ValueError("Expected five minute folders")
    images = []
    for section in sections:
        files = sorted((section / "frames").glob("*.jpg"))
        if [p.stem for p in files] != [f"{i:06d}" for i in range(1500)]:
            raise ValueError(f"Missing/extra images in {section}")
        images.extend(str(p.resolve()) for p in files)
    args.output.mkdir(parents=True, exist_ok=False)
    targets = np.linspace(0, len(images)-1, args.count).round().astype(int).tolist()
    plan = dict(approved=False, frames=images, window=args.window, experiments=[],
                method="Independent starting photographs; separate states for each time direction.",
                prompt_method="Positive point inside largest distance-to-boundary region of reviewed mask.",
                reference_use="Only starting-frame masks generate prompts. Other reference masks are withheld.",
                reference_sessions=[], notes="Resolve unclear starting identities and edit points before approval.")
    for part, section in enumerate(sections):
        session_path = section / "reference_all15blocks_reviewed.pkl"
        with session_path.open("rb") as f:
            data = SessionReader(f).load()
        plan["reference_sessions"].append(dict(path=str(session_path.resolve()), sha256=sha(session_path)))
        masks = {int(Path(str(k).replace("\\", "/")).stem): v for k, v in data["masks"].items()}
        for index in [i for i in targets if i//1500 == part]:
            local = index % 1500
            points = {}
            with Image.open(images[index]) as im:
                preview = im.convert("RGB")
            draw = ImageDraw.Draw(preview)
            for bird in BIRDS:
                packed = masks.get(local, {}).get(bird)
                if packed is None:
                    continue
                mask = unpack(packed)
                if not mask.any():
                    continue
                # Padding ensures a finite boundary even for a full-image mask.
                distance = distance_transform_edt(np.pad(mask, 1))[1:-1, 1:-1]
                y, x = np.unravel_index(distance.argmax(), mask.shape)
                points[bird] = [int(x), int(y)]
                draw.ellipse((x-10, y-10, x+10, y+10), fill="white", outline="black", width=3)
                draw.text((x+12, y), bird, fill="white", stroke_fill="black", stroke_width=2)
            preview.thumbnail((1440, 900))
            preview.save(args.output / f"prompt_{index:06d}.jpg")
            plan["experiments"].append(dict(start=index, points=points,
                unavailable_birds=[b for b in BIRDS if b not in points],
                review_status="pending", reference_section=section.name, local_index=local))
        del masks, data
    write_json(args.output / "plan.json", plan)
    (args.output / "index.html").write_text(
        '<meta charset="utf-8"><h1>Starting photographs and proposed points</h1>'
        '<p>Review all points and identities. Unavailable birds cannot be tracked in that run.</p>'
        + "".join(f'<h2>Global index {e["start"]}, {e["reference_section"]}, local {e["local_index"]}</h2>'
                  f'<p>Unavailable birds: {e["unavailable_birds"]}</p>'
                  f'<img style="width:95%" src="prompt_{e["start"]:06d}.jpg">'
                  for e in plan["experiments"]), encoding="utf-8")
    print("Prepared", args.output / "index.html")
    print("Review points and uncertainty notes. Set each review_status to accepted and approved to true in plan.json.")


def bounds(start, total, window):
    first = max(0, min(start-window//2, total-window))
    return first, min(total, first+window)


def run(args):
    plan_path = args.output / "plan.json"
    plan = json.loads(plan_path.read_text())
    if not plan["approved"] or any(e["review_status"] != "accepted" for e in plan["experiments"]):
        raise ValueError("Review the proposed starting points before approving plan.json")
    if args.experiments:
        chosen = set(args.experiments)
        experiments = [e for e in plan["experiments"] if e["start"] in chosen]
        if {e["start"] for e in experiments} != chosen:
            raise ValueError("Requested start index is absent from plan")
    else:
        experiments = plan["experiments"]
    import torch
    from sam2.build_sam import build_sam2_video_predictor
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU required")
    predictor = build_sam2_video_predictor("sam2.1_hiera_l.yaml", str(args.checkpoint), device="cuda")
    for experiment in experiments:
        start = experiment["start"]
        out = args.output / f"run_{start:06d}"
        # Completed runs are immutable; interrupted runs need a new output directory.
        if (out / "run.json").is_file() and json.loads((out/"run.json").read_text())["status"] == "completed":
            print("Already complete:", out, flush=True)
            continue
        out.mkdir(exist_ok=False)
        points = experiment["points"]
        if not points:
            raise ValueError(f"No available birds at {start}")
        first, stop = (experiment["first"], experiment["stop_exclusive"]) if "first" in experiment else bounds(start, len(plan["frames"]), plan["window"])
        if not 0 <= first <= start < stop <= len(plan["frames"]):
            raise ValueError("Invalid experiment interval")
        report = dict(status="running", start=start, first=first, stop_exclusive=stop,
                      points=points, unavailable_birds=experiment["unavailable_birds"],
                      plan_sha256=sha(plan_path), checkpoint_sha256=sha(args.checkpoint),
                      config="sam2.1_hiera_l.yaml", memory_window=predictor.memory_window_size,
                      torch=str(torch.__version__), gpu=torch.cuda.get_device_name(),
                      score="Mean sigmoid of positive logits. Empty masks have no score.")
        seen = set()
        write_json(out/"run.json", report)
        try:
            for bird in points:
                (out/"masks"/bird).mkdir(parents=True)
            with (out/"scores.csv").open("w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["frame_index", "bird", "direction", "is_prompt", "area_px", "foreground_confidence"])
                for direction, indices in [("later", range(start, stop)), ("earlier", range(start, first-1, -1))]:
                    if direction == "earlier" and len(indices) == 1:
                        continue
                    with tempfile.TemporaryDirectory(dir=out) as tmp:
                        for local, index in enumerate(indices):
                            os.symlink(plan["frames"][index], Path(tmp)/f"{local:06d}.jpg")
                        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
                            state = predictor.init_state(video_path=tmp, offload_video_to_cpu=True, offload_state_to_cpu=True)
                            names = list(points)
                            for identity, bird in enumerate(names, 1):
                                predictor.add_new_points_or_box(inference_state=state, frame_idx=0, obj_id=identity,
                                    points=np.array([points[bird]], np.float32), labels=np.array([1],np.int32))
                            for local, ids, logits in predictor.propagate_in_video(state, start_frame_idx=0):
                                if not 0 <= local < len(indices):
                                    raise ValueError("Unexpected local index")
                                index = indices[local]
                                if direction == "earlier" and index == start:
                                    continue
                                if index in seen or set(map(int,ids)) != set(range(1,len(names)+1)):
                                    raise ValueError("Duplicate frame or unexpected identities")
                                for j, identity in enumerate(ids):
                                    bird = names[int(identity)-1]
                                    values = logits[j].float().cpu().numpy().squeeze()
                                    if values.ndim != 2 or not np.isfinite(values).all():
                                        raise ValueError("Invalid mask logits")
                                    mask = values > 0
                                    Image.fromarray(mask).save(out/"masks"/bird/f"{index:06d}.png")
                                    writer.writerow([index,bird,direction,int(index==start),int(mask.sum()),foreground_confidence(values)])
                                seen.add(index)
                                if len(seen)%100 == 0:
                                    f.flush()
                                    print(out.name, len(seen), "/", stop-first, flush=True)
                            predictor.reset_state(state)
                            del state
                        torch.cuda.empty_cache()
            if seen != set(range(first,stop)):
                raise ValueError("Incomplete coverage")
            report["status"] = "completed"
        except Exception as exc:
            report.update(status="failed",error=str(exc))
            raise
        finally:
            report["frames_written"] = len(seen)
            write_json(out/"run.json",report)
        print("Completed",out,flush=True)



def review(args):
    plan = json.loads((args.output/"plan.json").read_text())
    page = ['<meta charset="utf-8"><title>Propagation comparisons</title>',
            '<style>body{font:18px system-ui}img{width:48%}section{margin-bottom:35px}</style>',
            '<h1>Reviewed reference and independent predictions</h1>',
            '<p>bird1 red, bird2 green, bird3 yellow, bird4 blue. Reference boundaries and identities in recorded uncertain intervals require verification.</p>']
    assets = args.output/"comparison_images"
    assets.mkdir(exist_ok=True)
    colors = dict(zip(BIRDS, [(220,55,55),(44,195,79),(218,186,24),(74,114,239)]))
    def overlay(photo, masks):
        image = np.array(photo.convert("RGB"))
        for bird, mask in masks.items():
            if mask.shape != image.shape[:2]:
                raise ValueError("Mask and photograph shapes differ")
            image[mask] = .55*image[mask] + .45*np.array(colors[bird])
        result = Image.fromarray(image)
        result.thumbnail((1000,700))
        return result
    for part, item in enumerate(plan["reference_sessions"]):
        needed = []
        for e in plan["experiments"]:
            out = args.output/f'run_{e["start"]:06d}'
            if not (out/"run.json").is_file():
                continue
            report = json.loads((out/"run.json").read_text())
            if report["status"] != "completed":
                continue
            for i in range(report["first"], report["stop_exclusive"], args.stride):
                if i//1500 == part:
                    needed.append((out,report,i))
        if not needed:
            continue
        with Path(item["path"]).open("rb") as f:
            session = SessionReader(f).load()
        truth = {int(Path(str(k).replace("\\","/")).stem):v for k,v in session["masks"].items()}
        for out,report,i in needed:
            with Image.open(plan["frames"][i]) as photo:
                reference = {b:unpack(m) for b,m in truth[i%1500].items() if b in BIRDS}
                predicted = {}
                for b in report["points"]:
                    with Image.open(out/"masks"/b/f"{i:06d}.png") as im:
                        predicted[b] = np.asarray(im.convert("1"),bool)
                a=f'ref_{i:06d}.jpg'
                b=f'{out.name}_{i:06d}.jpg'
                overlay(photo, reference).save(assets/a,quality=88)
                overlay(photo, predicted).save(assets/b,quality=88)
            page.append(f'<section><h2>Start {report["start"]} · global frame {i} · part {part+1} local {i%1500}</h2><p>Left reviewed reference · right prediction. Unprompted birds: {report["unavailable_birds"]}</p><img src="comparison_images/{a}"><img src="comparison_images/{b}"></section>')
        del session, truth
    (args.output/"comparison.html").write_text("\n".join(page),encoding="utf-8")
    print("Open",args.output/"comparison.html")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("command",choices=["prepare","run","review"])
    ap.add_argument("--source",type=Path,default=ROOT/"experiments/lek3/reference_20m_35m")
    ap.add_argument("--output",type=Path,required=True)
    ap.add_argument("--count",type=int,default=10)
    ap.add_argument("--window",type=int,default=750)
    ap.add_argument("--checkpoint",type=Path,default=ROOT/"checkpoints/sam2.1_hiera_large.pt")
    ap.add_argument("--stride",type=int,default=25)
    ap.add_argument("--experiments",type=int,nargs="+")
    args=ap.parse_args()
    if not 1 <= args.count <= 7500 or not 1 <= args.window <= 1500:
        ap.error("count must be 1–7500; window must be 1–1500 to bound memory use")
    if args.command=="prepare":
        prepare(args)
    elif args.command=="run":
        run(args)
    else:
        if args.stride < 1: ap.error("stride must be positive")
        review(args)
