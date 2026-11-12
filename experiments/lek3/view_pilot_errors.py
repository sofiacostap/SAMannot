"""Create a portable, frame-indexed reference/prediction viewer for the Lek3 pilot."""
import argparse
import csv
import json
import shutil
from itertools import groupby
from pathlib import Path

import numpy as np
from PIL import Image

from prepare_pilot_review import BIRDS, COLORS


def build(evaluation, out, previews=None):
    with (evaluation / "pairs.csv").open(newline="") as stream:
        records = list(csv.DictReader(stream))
    rows = {}
    for r in records:
        key = int(r["frame_index"]), r["bird"]
        if key in rows:
            raise ValueError(f"Duplicate pair {key}")
        rows[key] = r
    if set(rows) != {(i, b) for i in range(375) if i != 87 for b in BIRDS}:
        raise ValueError("Expected the complete nonprompt evaluation")
    run = evaluation.parent
    pilot = run.parent
    indices = (sorted(int(p.name.split("_")[0]) for p in previews.glob("*_reviewed.jpg"))
               if previews else list(range(375)))
    if not indices:
        raise ValueError("No images available")
    out.mkdir(parents=True, exist_ok=True)
    assets = out / "images"
    assets.mkdir(exist_ok=True)
    for i in indices:
        stem = f"{i:06d}"
        if previews:
            for tag in ["prediction", "reviewed"]:
                shutil.copyfile(previews/f"{stem}_{tag}.jpg", assets/f"{stem}_{tag}.jpg")
            continue
        with Image.open(pilot/"frames"/f"{stem}.jpg") as photo:
            source = np.array(photo.convert("RGB"))
        prediction = source.copy()
        for bird, color in zip(BIRDS, COLORS):
            with Image.open(run/"masks"/bird/f"{stem}.png") as im:
                mask = np.asarray(im.convert("1"), dtype=bool)
            if mask.shape != source.shape[:2]:
                raise ValueError(f"Image dimensions differ at {i} for {bird}")
            prediction[mask] = 0.54*source[mask] + 0.46*np.asarray(color)
        pred = Image.fromarray(prediction)
        pred.thumbnail((1440,900))
        pred.save(assets/f"{stem}_prediction.jpg", quality=90)
        with Image.open(pilot/"reviewed_reference/Session/overlays"/f"{stem}.jpg") as ref:
            if ref.size != (source.shape[1], source.shape[0]):
                raise ValueError(f"Reference dimensions differ at {i}")
            ref.thumbnail((1440,900))
            ref.save(assets/f"{stem}_reviewed.jpg", quality=90)
        if (i+1)%25 == 0:
            print(f"Prepared {i+1}/375 frames", flush=True)

    intervals = []
    for b in BIRDS:
        errors = sorted((i,r["status"]) for (i,bird),r in rows.items() if bird==b and r["error"]=="1")
        for (_,status), group in groupby(enumerate(errors), lambda t:(t[1][0]-t[0], t[1][1])):
            seq = [x[0] for _,x in group]
            intervals.append(dict(bird=b, status=status, first=seq[0], last=seq[-1], frames=len(seq)))
    with (out/"error_intervals.csv").open("w",newline="") as f:
        w=csv.DictWriter(f, fieldnames=["bird","status","first","last","frames"])
        w.writeheader()
        w.writerows(intervals)
    shutil.copyfile(evaluation/"pairs.csv",out/"pairs.csv")
    data = dict(indices=indices, intervals=intervals, rows={i:[rows[i,b] for b in BIRDS] for i in indices if i!=87})
    (out/"index.html").write_text(PAGE.replace("__DATA__",json.dumps(data)),encoding="utf-8")
    print("Open",out/"index.html")


PAGE="""<!doctype html><meta charset="utf-8"><title>Lek3 mask comparison</title>
<style>body{font:17px system-ui;margin:2vw;color:#152b37}h1{margin-bottom:4px}button,select{font:inherit;padding:4px}figure{margin:0;min-width:0}img{width:100%}.panels{display:grid;grid-template-columns:1fr 1fr;gap:12px}figcaption{font-weight:bold;margin:8px 0}td,th{text-align:left;padding:4px 22px 4px 0}tr.bad{color:#962f10}#jump{max-width:90vw}</style>
<h1>Reviewed ground truth and prediction</h1><p>Bird 1 red · Bird 2 green · Bird 3 yellow · Bird 4 blue</p>
<button id="prev">← Previous</button> <select id="frame" aria-label="Frame index"></select> <button id="next">Next →</button>
<p><label>Jump to an error interval <select id="jump"><option value="">Choose an interval</option></select></label></p>
<div class="panels"><figure><figcaption>Reviewed reference</figcaption><a id="ra"><img id="ref"></a></figure><figure><figcaption>Prediction</figcaption><a id="pa"><img id="pred"></a></figure></div>
<h3 id="label"></h3><table><thead><tr><th>Bird</th><th>Outcome</th><th>IoU</th><th>Confidence</th></tr></thead><tbody id="body"></tbody></table>
<p><a href="error_intervals.csv">All error intervals (inclusive file indices)</a> · <a href="pairs.csv">All per-bird measurements</a></p>
<script>const d=__DATA__, el=id=>document.getElementById(id), ix=d.indices;
for(const i of ix)el('frame').add(new Option('Index '+i,i));
for(const v of d.intervals){const i=ix.find(n=>n>=v.first&&n<=v.last);if(i!==undefined)el('jump').add(new Option(v.bird+' · '+v.status.replaceAll('_',' ')+' · '+v.first+'–'+v.last+' ('+v.frames+' frames)',i))}
function show(i){el('frame').value=i;const j=ix.indexOf(i);el('prev').disabled=j===0;el('next').disabled=j===ix.length-1;
const prefix='images/'+String(i).padStart(6,'0');el('ref').src=el('ra').href=prefix+'_reviewed.jpg';el('pred').src=el('pa').href=prefix+'_prediction.jpg';
el('label').textContent='File index '+i+' · SAMannot image '+(i+1)+' · '+(i/25).toFixed(2)+' s from clip start'+(i===87?' · prompt excluded':'');
el('body').replaceChildren();for(const r of d.rows[i]||[]){let tr=el('body').insertRow();if(r.error==='1')tr.className='bad';for(const val of [r.bird,r.status.replaceAll('_',' '),r.iou?Number(r.iou).toFixed(3):'—',r.confidence?Number(r.confidence).toFixed(4):'—'])tr.insertCell().textContent=val}}
function step(n){const j=ix.indexOf(Number(el('frame').value))+n;if(j>=0&&j<ix.length)show(ix[j])}
el('prev').onclick=()=>step(-1);el('next').onclick=()=>step(1);el('frame').onchange=e=>show(Number(e.target.value));
el('jump').onchange=e=>{if(e.target.value!=='')show(Number(e.target.value));e.target.value=''};
document.onkeydown=e=>{if(e.target.tagName==='SELECT')return;if(e.key==='ArrowLeft')step(-1);if(e.key==='ArrowRight')step(1)};show(ix[0]);</script>"""


if __name__=="__main__":
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("evaluation",type=Path)
    args=ap.parse_args()
    build(args.evaluation,args.evaluation/"mask_comparison")
