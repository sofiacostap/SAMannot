"""Compare sampled visual observations with confidence flags and show separate bird masks."""
import argparse
import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[2]
BASE = Path("/mnt/hatter/MPI_data/Sofia/lek3_502m_reference")
BIRDS = ("bird1", "bird2", "bird3", "bird4")
ERRORS = {"missing_prediction", "partial_mask", "cross_identity_contamination",
          "missing_prediction_after_return"}
UNCERTAIN = {"reference_or_prediction_uncertain", "observation_unspecified", "reference_invalid"}


def read_csv(path):
    with path.open(newline="", encoding="utf-8-sig") as stream:
        return list(csv.DictReader(stream))


def audit(base, source, output, observations, overrides=None):
    runs = base / "baseline_regular30s_v1"
    scores = {}
    frames = defaultdict(list)
    unavailable = {}
    for start in range(0, 7500, 750):
        folder = runs / f"run_{start:06d}"
        manifest = json.loads((folder/"run.json").read_text())
        if manifest["status"] != "completed" or manifest["frames_written"] != 750:
            raise ValueError(f"Incomplete run {folder}")
        unavailable[start] = set(manifest["unavailable_birds"])
        records = read_csv(folder/"scores.csv")
        for row in records:
            i, bird = int(row["frame_index"]), row["bird"]
            if not start <= i < start+750 or (i,bird) in scores:
                raise ValueError("Duplicate or out-of-range score")
            scores[i,bird] = row
            if row["is_prompt"] == "0" and row["foreground_confidence"]:
                frames[i].append(float(row["foreground_confidence"]))
    ranked = sorted((min(values),i) for i,values in frames.items())
    selected = {i for _,i in ranked[:math.ceil(len(ranked)*.1)]}
    sections = sorted(source.glob("part[0-9][0-9]_*"))
    if len(sections) != 5:
        raise ValueError("Expected five reference sections")

    # The visual review covered stride-25 samples, rather than all video frames.
    decisions = {}
    for row in read_csv(observations):
        first, last = int(row["first_global_index"]), int(row["last_global_index"])
        for i in range(first,last+1):
            if i%25:
                continue
            birds = BIRDS if row["birds"]=="all" else row["birds"].split(";")
            for bird in birds:
                decisions[i,bird] = row["review_outcome"]

    if overrides:
        override_rows = read_csv(overrides)
        seen_overrides = set()
        for row in override_rows:
            key = int(row["global_index"]), row["bird"]
            if key in seen_overrides or key[0] % 25 or key[1] not in BIRDS:
                raise ValueError(f"Invalid or duplicate review override: {key}")
            if row["review_outcome"] not in ERRORS | UNCERTAIN | {"visually_correct"}:
                raise ValueError("Unknown review outcome")
            seen_overrides.add(key)
            decisions[key] = row["review_outcome"]

    rows = []
    frame_rows = []
    for i in range(0,7500,25):
        if i%750 == 0:
            continue
        frame_errors = frame_uncertain = 0
        for bird in BIRDS:
            outcome = decisions.get((i,bird),"visually_correct")
            if bird in unavailable[(i//750)*750]:
                outcome = "uninitialized_return" if outcome=="uninitialized_return" else "unprompted_not_evaluable"
            error = int(outcome in ERRORS)
            uncertain = outcome in UNCERTAIN
            frame_errors += error
            frame_uncertain += uncertain
            record = scores.get((i,bird))
            confidence = record["foreground_confidence"] if record else ""
            rows.append(dict(global_index=i,player_seconds=1200+i/25,bird=bird,
                visual_outcome=outcome,frame_flagged=int(i in selected),
                bird_confidence=confidence,
                scored_error=int(error and bool(confidence)),
                is_error=error,reference_uncertain=int(uncertain)))
        frame_rows.append(dict(index=i,flagged=i in selected,error=frame_errors>0,
                               uncertain=frame_uncertain>0))

    output.mkdir(parents=True,exist_ok=False)
    import shutil
    shutil.copyfile(observations, output/"baseline_visual_observations.csv")
    if overrides:
        shutil.copyfile(overrides, output/"review_overrides.csv")
    with (output/"review_vs_confidence.csv").open("w",newline="") as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]))
        writer.writeheader();writer.writerows(rows)
    evaluable_frames=[r for r in frame_rows if not r["uncertain"]]
    flagged=[r for r in evaluable_frames if r["flagged"]]
    errors=[r for r in evaluable_frames if r["error"]]
    summary=dict(
        scope="Reported one-second visual samples, excluding prompt frames. Unlisted samples assumed correct as reported.",
        flag_rule="Lowest 10% of dense nonprompt frames, ranked by minimum available bird confidence.",
        dense_flagged_frames=len(selected),observed_cutoff=ranked[math.ceil(len(ranked)*.1)-1][0],
        sampled_nonprompt_frames=len(frame_rows),sampled_frames_with_uncertainty=sum(r["uncertain"] for r in frame_rows),
        sampled_evaluable_frames=len(evaluable_frames),
        sampled_flagged_frames=len(flagged),
        sampled_flagged_error_frames=sum(r["error"] for r in flagged),
        sampled_error_frames=len(errors),
        sampled_error_frames_flagged=sum(r["flagged"] for r in errors),
        visual_bird_errors=sum(r["is_error"] for r in rows),
        visual_bird_errors_without_confidence=sum(r["is_error"] and not r["bird_confidence"] for r in rows),
        visual_bird_errors_confidence_at_least_095=sum(r["is_error"] and bool(r["bird_confidence"]) and float(r["bird_confidence"])>=.95 for r in rows),
        visual_outcome_counts=dict(Counter(r["visual_outcome"] for r in rows)),
        limitation="Unknown exact interval boundaries; unprompted identities separated. This is a sampled visual audit, not full-frame numerical evaluation.")
    (output/"summary.json").write_text(json.dumps(summary,indent=2))
    build_viewer(base,source,sections,runs,output,scores)
    print(json.dumps(summary,indent=2))
    print("OPEN:",output/"individual_masks.html")


def build_viewer(base,source,sections,runs,output,scores):
    assets=output/"images";assets.mkdir()
    page=['<meta charset="utf-8"><title>Individual bird mask review</title>',
          '<style>body{font:18px system-ui;margin:25px}.row{display:flex;gap:10px}figure{margin:0;flex:1;min-width:0}img{width:100%}section{margin:40px 0}a{color:#146b75}</style>',
          '<h1>Green and blue masks inspected separately</h1>',
          '<p>Each panel contains only the named bird. Reference uncertainty still applies. Click images for full resolution.</p>']
    for i in (550,5550,6775,7000,7475):
        part,local=divmod(i,1500); section=sections[part]
        with Image.open(section/"frames"/f"{local:06d}.jpg") as im:
            photo=np.array(im.convert("RGB"))
        page.append(f'<section><h2>Global index {i} · part {part+1} local {local}</h2>')
        for bird,color in [("bird2",(44,195,79)),("bird4",(74,114,239))]:
            captions=[]
            page.append(f'<h3>{bird}</h3><div class="row">')
            for title,path in [
                ("Reference",base/"png_reference"/section.name/"masks"/bird/f"{local:06d}.png"),
                ("Prediction",runs/f"run_{(i//750)*750:06d}"/"masks"/bird/f"{i:06d}.png")]:
                if not path.is_file():
                    page.append(f'<figure>{title}: missing mask entry</figure>')
                    continue
                with Image.open(path) as im:
                    mask=np.array(im.convert("1"),bool)
                if mask.shape != photo.shape[:2]:
                    raise ValueError("Mask dimensions differ")
                overlay=photo.copy()
                overlay[mask]=(.5*photo[mask]+.5*np.array(color)).astype(np.uint8)
                name=f"{i:06d}_{bird}_{title.lower()}.jpg"
                Image.fromarray(overlay).save(assets/name,quality=94)
                page.append(f'<figure><figcaption>{title} · {int(mask.sum())} pixels</figcaption><a href="images/{name}"><img src="images/{name}"></a></figure>')
            page.append('</div>')
            score=scores.get((i,bird),{}).get("foreground_confidence","")
            page.append(f'<p>Prediction confidence: {score or "unavailable"}</p>')
        page.append('</section>')
    (output/"individual_masks.html").write_text("\n".join(page),encoding="utf-8")


if __name__=="__main__":
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--base",type=Path,default=BASE)
    ap.add_argument("--source",type=Path,default=ROOT/"experiments/lek3/reference_20m_35m")
    ap.add_argument("--output",type=Path,required=True)
    ap.add_argument("--observations",type=Path,default=Path(__file__).with_name("baseline_visual_observations.csv"))
    ap.add_argument("--overrides",type=Path)
    args=ap.parse_args()
    audit(args.base,args.source,args.output,args.observations,args.overrides)
