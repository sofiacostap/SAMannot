"""Audit binary masks and preview empty intervals; no inference or data edits."""
import argparse
import csv
import html
import json
from collections import defaultdict, Counter
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw
from recompute import indexed, gt_labels, sha, write_json, write_csv
from verify_correspondence import fingerprint


def intervals(indices):
    result=[]
    for frame in sorted(set(indices)):
        if result and frame==result[-1][1]+1:result[-1][1]=frame
        else:result.append([frame,frame])
    return result


def sample_frames(start,end,first,last,context):
    return sorted(set([max(first,start-context),start,(start+end)//2,end,min(last,end+context)]))


def binary_mask(path):
    with Image.open(path) as image:arr=np.asarray(image)
    if arr.ndim!=2 or not set(map(int,np.unique(arr))).issubset({0,255}):
        raise ValueError(f'Unexpected binary mask format: {path}')
    return arr>0


def run(args):
    run_dir=args.results.resolve()
    manifest=json.loads((run_dir/'verification.json').read_text())
    if manifest['status']!='completed':raise ValueError('Propagation must be completed')
    annotations=json.loads((run_dir/'annotations.json').read_text())
    source=Path(manifest['source_directory'])
    if sha(source/'verification.json')!=annotations['source_verification_sha256']:
        raise ValueError('Source verification changed')
    with (source/'index.csv').open(newline='') as f:mapping={int(r['video_index']):r for r in csv.DictReader(f)}
    total=annotations['total_frames'];size=annotations['block_size'];ids=annotations['gt_ids']
    with (run_dir/'foreground_confidence.csv').open(newline='') as f:rows=list(csv.DictReader(f))
    keys=[(int(r['abs_frame']),int(r['gt_id'])) for r in rows]
    if len(keys)!=len(set(keys)) or set(keys)!={(f,g) for f in range(total) for g in ids}:
        raise ValueError('Confidence rows have duplicate, missing or extra keys')
    prior=json.loads((args.prior/'manifest.json').read_text());gtfiles=indexed(prior['inputs']['gt'])
    output=Path(__file__).resolve().parent/'results'/('empty-review-'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ'))
    output.mkdir();report=dict(status='in_progress',propagation=str(run_dir),source_sha256=sha(__file__),
        propagation_report_sha256=sha(run_dir/'verification.json'),
        note='Empty means zero foreground pixels in a saved per-ID mask. A review flag, not a confirmed biological tracking error. No GT needed to detect emptiness.',
        issues=[],empty_records=0)
    flags=[];groups=defaultdict(list);areas={};interval_rows=[];preview_rows=[]
    try:
        for number,row in enumerate(rows,1):
            frame=int(row['abs_frame']);bird=int(row['gt_id']);block=frame//size
            prompt=annotations['blocks'][str(block)]['video_frame']
            direction='earlier' if frame<prompt else 'later'
            if int(row['block_id'])!=block or row['direction']!=direction:raise ValueError('CSV direction/block mismatch')
            path=run_dir/'masks_by_gt_id'/str(bird)/f'{frame:06d}.png'
            if not path.exists():
                report['issues'].append(dict(frame=frame,gt_id=bird,issue='mask_file_missing'));continue
            mask=binary_mask(path);area=int(mask.sum());areas[(frame,bird)]=area
            if (area==0)!=(row['foreground_logit_score']==''):
                report['issues'].append(dict(frame=frame,gt_id=bird,issue='confidence_and_mask_disagree'))
            if area==0:
                groups[(block,bird,direction)].append(frame)
                flags.append(dict(video_frame=frame,gt_id=bird,block=block,direction=direction,
                    reason='empty_mask',requires_review=True,distance_from_prompt=frame-prompt))
            if number%1000==0:print(f'Checked {number}/{len(rows)} binary masks',flush=True)
        for (block,bird,direction),frames in groups.items():
            prompt=annotations['blocks'][str(block)]['video_frame']
            for start,end in intervals(frames):
                interval_rows.append(dict(block=block,gt_id=bird,direction=direction,start=start,end=end,
                    length=end-start+1,prompt=prompt))
        interval_rows.sort(key=lambda r:(-r['length'],r['block'],r['gt_id'],r['start']))
        pages=['<!doctype html><meta charset="utf-8"><title>Empty-mask review</title>',
            '<style>body{font:16px system-ui;margin:24px}img{max-width:100%}section{margin:30px 0}a{margin-right:12px}</style>',
            '<h1>Empty-mask intervals</h1><p>Left: verified photograph. Middle: prediction (red). Right: selected GT identity (green). No clicks are overlaid. GT may contain errors. Frames without GT are labelled explicitly. Frame numbers are video indices.</p>',
            '<p>Earlier passes run from the prompt toward decreasing frame numbers. Previews below are displayed in chronological video order, not processing order.</p>']
        rendered={}
        for n,interval in enumerate(interval_rows,1):
            b=interval['block'];g=interval['gt_id'];lo=b*size;hi=min(lo+size,total)-1
            chosen=sample_frames(interval['start'],interval['end'],lo,hi,args.context)
            # Include the starting annotation as a reference for each interval.
            chosen=sorted(set(chosen+[interval['prompt']]))
            pages.append(f'<section><h2>Block {b}, GT {g}: {interval["start"]}–{interval["end"]} ({interval["length"]} empty masks)</h2>')
            for frame in chosen:
                key=(frame,g)
                if key not in rendered:
                    if key not in areas:continue
                    row=mapping[frame]
                    rawpath=source/'frames'/f'{frame:06d}.png'
                    with Image.open(rawpath) as image:rgb=np.asarray(image.convert('RGB')).copy()
                    if fingerprint(rgb)!=row['rgb_sha256']:raise ValueError(f'Photograph changed: {frame}')
                    predpath=run_dir/'masks_by_gt_id'/str(g)/f'{frame:06d}.png';pred=binary_mask(predpath)
                    if pred.shape!=rgb.shape[:2]:raise ValueError('Prediction/image dimensions differ')
                    gtindex=int(row['gt_frames_index']) if row['gt_frames_index'] else None
                    truth=None;gtsha=None
                    if gtindex is not None:
                        gtpath=gtfiles[gtindex];truth=gt_labels(gtpath)==g;gtsha=sha(gtpath)
                        if truth.shape!=pred.shape:raise ValueError('GT/image dimensions differ')
                    canvas=Image.new('RGB',(2400,530),'white');draw=ImageDraw.Draw(canvas)
                    draw.text((10,8),f'Video {frame}; block {b}; GT ID {g}; prompt {interval["prompt"]}; signed distance {frame-interval["prompt"]}',fill='black')
                    labels=['Verified photograph',f'Prediction: {areas[key]} foreground pixels',
                        f'GT/FRAMES {gtindex}: {int(truth.sum())} pixels' if truth is not None else 'NO MATCHING GT IMAGE']
                    for side,mask in enumerate([None,pred,truth]):
                        panel=rgb.copy()
                        if mask is not None:
                            tint=np.array([255,65,65] if side==1 else [40,230,80])
                            panel[mask]=(.5*panel[mask]+.5*tint).astype(np.uint8)
                        canvas.paste(Image.fromarray(panel).resize((800,452)),(side*800,70))
                        draw.text((side*800+10,38),labels[side],fill='black')
                    filename=f'video{frame:06d}_GT{g}.jpg';canvas.save(output/filename,quality=94)
                    rendered[key]=filename
                    preview_rows.append(dict(video_frame=frame,gt_id=g,gt_frame=gtindex,pred_area=areas[key],
                        gt_area=int(truth.sum()) if truth is not None else None,image=filename,
                        source_rgb_sha256=row['rgb_sha256'],prediction_sha256=sha(predpath),gt_sha256=gtsha))
                pages.append(f'<p><a href="{rendered[key]}">Video {frame}</a></p><img loading="lazy" src="{rendered[key]}">')
            pages.append('</section>')
            print(f'Previewed interval {n}/{len(interval_rows)}',flush=True)
        (output/'index.html').write_text('\n'.join(pages),encoding='utf-8')
        report.update(status='completed_with_issues' if report['issues'] else 'completed',
            mask_records_checked=len(areas),empty_records=len(flags),interval_count=len(interval_rows),
            previews=len(preview_rows),empty_by_block=dict(Counter(r['block'] for r in flags)))
    except Exception as exc:
        report.update(status='failed',error=str(exc));raise
    finally:
        write_csv(output/'flags.csv',flags,['video_frame','gt_id','block','direction','reason','requires_review','distance_from_prompt'])
        write_csv(output/'intervals.csv',interval_rows,['block','gt_id','direction','start','end','length','prompt'])
        write_csv(output/'previews.csv',preview_rows,['video_frame','gt_id','gt_frame','pred_area','gt_area','image','source_rgb_sha256','prediction_sha256','gt_sha256'])
        write_json(output/'verification.json',report)
        print(f"Status: {report['status']}; results: {output}",flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results',type=Path,required=True)
    parser.add_argument('--prior',type=Path,required=True)
    parser.add_argument('--context',type=int,default=5)
    args=parser.parse_args()
    if args.context<1:parser.error('context must be positive')
    run(args)
