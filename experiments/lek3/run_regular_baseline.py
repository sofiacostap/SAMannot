"""Run ten independent forward 750-frame blocks from approved regular prompts."""
import argparse
import json
from pathlib import Path
from types import SimpleNamespace

from batch_start_experiments import ROOT, BIRDS, run, review, sha, write_json
from PIL import Image
import numpy as np


def make_plan(prompt_file, source):
    prompts = json.loads(prompt_file.read_text())
    entries = prompts["prompts"]
    if [e["global_index"] for e in entries] != list(range(0, 7500, 750)):
        raise ValueError("Expected ten starting frames spaced 750 frames apart")
    sections = sorted(source.glob("part[0-9][0-9]_*"))
    if len(sections) != 5:
        raise ValueError("Expected five source sections")
    frames, references = [], []
    for section in sections:
        files = sorted((section/"frames").glob("*.jpg"))
        if [p.stem for p in files] != [f"{i:06d}" for i in range(1500)]:
            raise ValueError(f"Incomplete source images: {section}")
        frames.extend(str(p.resolve()) for p in files)
        session = section/"reference_all15blocks_reviewed.pkl"
        if not session.is_file():
            raise FileNotFoundError(session)
        references.append(dict(path=str(session.resolve()), sha256=sha(session)))
    experiments = []
    for e in entries:
        start = e["global_index"]
        unavailable = set(e["unavailable"])
        expected = {"bird1"} if start == 0 else ({"bird2"} if start == 750 else set())
        if unavailable != expected or set(e["points"]) != set(BIRDS)-expected:
            raise ValueError(f"Prompt identities differ from reviewed decisions at {start}")
        if Path(e["photograph"]).resolve() != Path(frames[start]).resolve():
            raise ValueError("Prompt photograph does not match source sequence")
        with Image.open(frames[start]) as im:
            width, height = im.size
        for point in e["points"].values():
            if len(point) != 2 or not np.isfinite(point).all() or not (0 <= point[0] < width and 0 <= point[1] < height):
                raise ValueError("Invalid point coordinates")
        experiments.append(dict(start=start, first=start, stop_exclusive=start+750,
                                points=e["points"], unavailable_birds=sorted(unavailable),
                                review_status="accepted"))
    return dict(approved=True, window=750, frames=frames, experiments=experiments,
                reference_sessions=references, prompt_file_sha256=sha(prompt_file),
                method="Fresh state per 30-second block; forward only; no prompts carried between blocks.",
                absence_decisions={"0": {"bird1": "visually_confirmed_absent"},
                                  "750": {"bird2": "visually_confirmed_absent"}},
                coverage_policy="Unprompted birds are unavailable for their entire block. No mask is fabricated.",
                evaluation_status="No IoU calculation. Reference tolerances and exclusions remain pending.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--prompts", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--source", type=Path, default=ROOT/"experiments/lek3/reference_20m_35m")
    ap.add_argument("--checkpoint", type=Path, default=ROOT/"checkpoints/sam2.1_hiera_large.pt")
    ap.add_argument("--preview-stride", type=int, default=25)
    args = ap.parse_args()
    if args.preview_stride < 1:
        ap.error("preview stride must be positive")
    # Existing outputs require the exact same plan before completed runs are reused.
    plan = make_plan(args.prompts, args.source)
    if args.output.exists():
        if json.loads((args.output/"plan.json").read_text()) != plan:
            raise ValueError("Existing output has a different plan; choose a new output folder")
    else:
        args.output.mkdir(parents=True)
        write_json(args.output/"plan.json",plan)
    run(SimpleNamespace(output=args.output, checkpoint=args.checkpoint, experiments=None))
    review(SimpleNamespace(output=args.output, stride=args.preview_stride))
    print("BASELINE COMPLETE:",args.output/"comparison.html",flush=True)
