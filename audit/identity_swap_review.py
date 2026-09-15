"""Task 5 research prototype: suggest, visually review, explicitly approve swaps.

Suggestions use trusted GT. This is not a GT-free deployed detector. Applying a
swap changes identity ownership only in a new overlay, never mask geometry or
the original prediction/GT files. No SAM2 imports or automatic approvals.
"""
import argparse
import html
import itertools
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from evaluate_task4 import (IDS, binary_mask, read_csv, sha, unique_rows,
                            validate_metadata, write_csv, write_json)
from verify_correspondence import fingerprint

COLORS = {14:(255,102,102),38:(54,217,239),75:(255,212,71),113:(197,138,255)}


def pair_candidates(matrix, eligible, cross_min=.5, gain_min=.2):
    found=[]
    for a,b in itertools.combinations(sorted(eligible),2):
        direct_a,direct_b=matrix[a,a],matrix[b,b]
        cross_a,cross_b=matrix[a,b],matrix[b,a]
        if min(cross_a,cross_b)>=cross_min and min(cross_a-direct_a,cross_b-direct_b)>=gain_min:
            found.append(dict(a=a,b=b,direct_a=direct_a,direct_b=direct_b,
                cross_a=cross_a,cross_b=cross_b,gain=(cross_a+cross_b-direct_a-direct_b)/2))
    # Ambiguous multi-label assignments are not reduced to an arbitrary pair.
    counts=Counter(g for r in found for g in (r['a'],r['b']))
    return [r for r in found if counts[r['a']]==counts[r['b']]==1]


def group_events(records, minimum_length=3):
    groups={}
    for r in records:groups.setdefault((r['block_id'],r['a'],r['b']),[]).append(r)
    events=[]
    for (block,a,b),items in sorted(groups.items()):
        items.sort(key=lambda r:r['video_frame'])
        chunks=[]
        for item in items:
            if not chunks or item['video_frame']!=chunks[-1][-1]['video_frame']+1:chunks.append([])
            chunks[-1].append(item)
        for chunk in chunks:
            if len(chunk)<minimum_length:continue
            events.append(dict(block_id=block,a=a,b=b,start=chunk[0]['video_frame'],end=chunk[-1]['video_frame'],
                length=len(chunk),minimum_gain=min(r['gain'] for r in chunk),decision='pending',
                boundary_status='Observed evidence interval; not certified biological start/end'))
    events.sort(key=lambda e:(-e['length'],e['start'],e['a'],e['b']))
    for i,e in enumerate(events,1):e['event_id']=f'SWAP{i:04d}'
    return events


def approved_mapping(events, approved, total, block_size, prompts):
    by_id={e['event_id']:e for e in events}
    if not approved or len(set(approved))!=len(approved) or not set(approved)<=set(by_id):
        raise ValueError('Explicit, unique valid event approvals required')
    result={}
    for name in approved:
        e=by_id[name];a,b=e['a'],e['b'];start,end=e['start'],e['end']
        if a==b or a not in IDS or b not in IDS or not 0<=start<=end<total:
            raise ValueError('Invalid identity pair or frame interval')
        if start//block_size!=end//block_size or any(start<=p<=end for p in prompts):
            raise ValueError('A swap cannot cross blocks or include an annotation frame')
        for frame in range(start,end+1):
            if (frame,a) in result or (frame,b) in result:
                raise ValueError('Conflicting approved intervals; do not silently compose swaps')
            result[frame,a]=b;result[frame,b]=a
    return result


def new_output(prefix):
    p=Path(__file__).resolve().parent/'results'/(prefix+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ'))
    p.mkdir()
    return p


def open_review(review):
    report=json.loads((review/'verification.json').read_text())
    if report['status']!='completed_suggestions':raise ValueError('Completed review required')
    prediction=Path(report['prediction_directory']);reference=Path(report['reference_directory'])
    for path,digest in report['input_hashes'].items():
        if sha(Path(path),True)!=digest:raise ValueError(f'Review input changed: {path}')
    if sha(review/'events.json',True)!=report['events_sha256']:raise ValueError('Proposal intervals changed')
    events=json.loads((review/'events.json').read_text())
    hashes={(int(r['video_frame']),int(r['gt_id'])):r['sha256'] for r in read_csv(review/'baseline_prediction_hashes.csv')}
    if sha(review/'baseline_prediction_hashes.csv',True)!=report['baseline_hashes_sha256']:
        raise ValueError('Baseline prediction hash inventory changed')
    return report,prediction,reference,events,hashes


def draw_preview(review, report, prediction, reference, hashes, event, frame, refs, images, gt_hashes):
    entry=images[frame]
    image_path=Path(report['source_directory'])/'frames'/entry['file']
    with Image.open(image_path) as im:rgb=np.array(im.convert('RGB'))
    if fingerprint(rgb)!=entry['rgb_sha256']:raise ValueError('Preview photograph changed')
    pair=(event['a'],event['b']);pred={}
    for g in pair:
        path=prediction/'masks_by_gt_id'/str(g)/f'{frame:06d}.png'
        if sha(path)!=hashes[frame,g]:raise ValueError('Preview prediction changed')
        pred[g]=binary_mask(path,rgb.shape[:2])
    truth={};held=[]
    for g in pair:
        row=refs.get((frame,g))
        if not row or row['status']!='trusted_screened':held.append(g);continue
        gf=int(row['gt_frame']);path=reference/row['normalized_mask']
        if sha(path)!=gt_hashes[str(gf)]['normalized']:raise ValueError('Preview GT changed')
        with Image.open(path) as im:gt=np.array(im)
        if gt.shape!=rgb.shape[:2]:raise ValueError('Preview dimensions differ')
        truth[g]=gt==g
    def overlay(masks):
        arr=rgb.copy()
        for g,mask in masks.items():arr[mask]=(.5*arr[mask]+.5*np.array(COLORS[g])).astype(np.uint8)
        return Image.fromarray(arr)
    width=1200;height=round(rgb.shape[0]*width/rgb.shape[1]);title=42
    inside=event['start']<=frame<=event['end']
    proposed={pair[0]:pred[pair[1]],pair[1]:pred[pair[0]]} if inside else pred
    panels=[('Photograph',Image.fromarray(rgb)),('Current prediction (selected pair only)',overlay(pred)),
            ('Trusted GT (held/unavailable: '+(','.join(map(str,held)) or 'none')+')',overlay(truth)),
            ('Proposed identities after swap - preview only' if inside else 'Outside proposed interval - unchanged',overlay(proposed))]
    canvas=Image.new('RGB',(width*2,(height+title)*2+55),'white');draw=ImageDraw.Draw(canvas)
    try:font=ImageFont.truetype('DejaVuSans.ttf',20)
    except OSError:font=ImageFont.load_default()
    draw.text((12,8),f"{event['event_id']} / video {frame} / GT {entry['gt_frames_index'] or 'unavailable'} / pair {pair}; 14 red, 38 cyan, 75 yellow, 113 purple",fill='black',font=font)
    for i,(label,panel) in enumerate(panels):
        x=(i%2)*width;y=55+(i//2)*(height+title)
        draw.text((x+12,y+8),label,fill='black',font=font)
        canvas.paste(panel.resize((width,height)),(x,y+title))
    dest=review/'previews'/f"{event['event_id']}_{frame:06d}.jpg"
    canvas.save(dest,quality=95)
    return dest.relative_to(review).as_posix()


def render(review, selected):
    report,prediction,reference,events,hashes=open_review(review)
    wanted=set(selected)
    if not wanted<={e['event_id'] for e in events}:raise ValueError('Unknown event ID')
    refs=unique_rows(read_csv(reference/'evaluation_set.csv'),('video_frame','gt_id'))
    images={int(r['video_index']):r for r in read_csv(Path(report['source_directory'])/'index.csv')}
    gt_hashes=json.loads((reference/'mask_hashes.json').read_text())
    (review/'previews').mkdir(exist_ok=True)
    preview_file=review/'preview_index.json'
    preview_index=json.loads(preview_file.read_text()) if preview_file.exists() else {}
    ann=json.loads((prediction/'annotations.json').read_text())
    for e in events:
        if e['event_id'] not in wanted:continue
        first=e['block_id']*ann['block_size'];last=min(ann['total_frames']-1,first+ann['block_size']-1)
        frames=sorted(set([max(first,e['start']-1),e['start'],(e['start']+e['end'])//2,e['end'],min(last,e['end']+1)]))
        preview_index[e['event_id']]=[dict(frame=f,file=draw_preview(review,report,prediction,reference,hashes,e,f,refs,images,gt_hashes)) for f in frames]
        print(f"Previewed {e['event_id']}: {e['start']}–{e['end']}",flush=True)
    write_json(preview_file,preview_index)
    lines=['<!doctype html><meta charset="utf-8"><title>Task 5 identity review</title>',
        '<style>body{font:17px system-ui;margin:24px}img{max-width:100%}section{border-top:2px solid #bbb;margin-top:32px}table{border-collapse:collapse}td,th{padding:8px;border:1px solid #bbb}</style>',
        '<h1>Possible identity swaps — human review required</h1>',
        '<p>Research suggestions use trusted GT. No correction has been applied. The proposed panel changes labels only, not mask shapes.</p>',
        '<p>Check the two physical birds, GT plausibility and interval boundaries. Samples do not certify every intervening frame. Leave uncertain cases unapproved. Original predictions and GT stay unchanged.</p>',
        '<table><tr><th>Event</th><th>Pair</th><th>Video interval</th><th>Frames</th><th>Preview</th></tr>']
    for e in events:
        name=e['event_id'];available=name in preview_index
        lines.append(f'<tr><td>{name}</td><td>{e["a"]} / {e["b"]}</td><td>{e["start"]}–{e["end"]}</td><td>{e["length"]}</td><td>'+ (f'<a href="#{name}">Review</a>' if available else 'Not yet rendered')+'</td></tr>')
    lines.append('</table>')
    for e in events:
        name=e['event_id']
        if name not in preview_index:continue
        lines.append(f'<section id="{name}"><h2>{name}: labels {e["a"]} / {e["b"]}, video {e["start"]}–{e["end"]}</h2>')
        for item in preview_index[name]:lines.append(f'<p>Video {item["frame"]}</p><a href="{html.escape(item["file"])}"><img loading="lazy" src="{html.escape(item["file"])}"></a>')
        lines.append('</section>')
    if not events:lines.append('<p>No intervals passed this conservative rule. This does not prove that no identity swaps exist.</p>')
    (review/'index.html').write_text('\n'.join(lines),encoding='utf-8')


def propose(args):
    prediction,reference=args.prediction.resolve(),args.reference.resolve()
    pv,rv,ann,confidence,refs=validate_metadata(prediction,reference,args.closeout)
    task4=args.task4.resolve()
    task4_release=json.loads(args.task4_closeout.read_text())
    if task4.name!=task4_release['result_run']:raise ValueError('Use the completed Task 4 baseline')
    for file,digest in task4_release['checksums'].items():
        if sha(task4/file,True)!=digest:raise ValueError('Task 4 result changed')
    ev=json.loads((task4/'verification.json').read_text())
    if Path(ev['prediction_run']).resolve()!=prediction or Path(ev['reference_run']).resolve()!=reference:
        raise ValueError('Task 4 input runs differ')
    baseline=unique_rows(read_csv(task4/'prediction_hashes.csv'),('video_frame','gt_id'))
    hashes= {k:r['sha256'] for k,r in baseline.items()}
    if set(baseline)!=set(confidence):raise ValueError('Baseline hash coverage differs')
    evaluated=unique_rows(read_csv(task4/'evaluation.csv'),('video_frame','gt_id'))
    if set(evaluated)!=set(baseline) or any(evaluated[k]['prediction_sha256']!=hashes[k] for k in hashes):
        raise ValueError('Prediction hashes differ from checked Task 4 evaluation')
    gt_hashes=json.loads((reference/'mask_hashes.json').read_text())
    output=new_output('swap-review-');records=[];eligible_frames=0
    report=dict(status='in_progress',source_directory=pv['source_directory'],prediction_directory=str(prediction),reference_directory=str(reference),
        method='Trusted GT-assisted pairwise identity comparison; no deployed GT-free detection claim.',
        parameters=dict(cross_iou_min=args.cross_min,individual_gain_min=args.gain_min,minimum_interval=args.min_length),
        input_hashes={str(p.resolve()):sha(p,True) for p in [prediction/'verification.json',prediction/'annotations.json',prediction/'foreground_confidence.csv',
            reference/'verification.json',reference/'evaluation_set.csv',reference/'mask_hashes.json',task4/'prediction_hashes.csv',args.closeout,args.task4_closeout]},
        source_sha256=sha(Path(__file__),True),
        limitations=['Suggestions use GT and are only research diagnostics.',
            'No bridges over excluded/missing GT or block boundaries; observed interval endpoints are not certified true endpoints.',
            'Three-way exchanges, short/weak swaps, empty masks and shape failures can be missed.',
            'Swapping identity cannot repair a wrong shape, wall mask or missing bird.'])
    try:
        for frame in range(ann['total_frames']):
            block=frame//ann['block_size']
            if frame==ann['blocks'][str(block)]['video_frame']:continue
            eligible=[g for g in IDS if (frame,g) in refs and refs[frame,g]['status']=='trusted_screened' and int(refs[frame,g]['gt_area'])>=50]
            if len(eligible)<2:continue
            row=refs[frame,eligible[0]];gf=int(row['gt_frame']);path=reference/row['normalized_mask']
            if sha(path)!=gt_hashes[str(gf)]['normalized']:raise ValueError('Released GT changed')
            with Image.open(path) as im:gt=np.array(im)
            if gt.ndim!=2 or not np.isin(gt,(0,)+IDS).all():raise ValueError('Unexpected GT labels')
            areas={g:int(np.count_nonzero(gt==g)) for g in eligible};matrix={}
            for a in eligible:
                path=prediction/'masks_by_gt_id'/str(a)/f'{frame:06d}.png'
                if sha(path)!=hashes[frame,a]:raise ValueError('Baseline prediction changed')
                mask=binary_mask(path,gt.shape);pa=int(np.count_nonzero(mask))
                counts=np.bincount(gt[mask],minlength=114)
                for b in eligible:
                    inter=int(counts[b]);union=pa+areas[b]-inter
                    matrix[a,b]=inter/union if union else 0.
            for candidate in pair_candidates(matrix,eligible,args.cross_min,args.gain_min):
                records.append(dict(video_frame=frame,gt_frame=gf,block_id=block,**candidate))
            eligible_frames+=1
            if frame%250==0:print(f'Swap evidence checked through video {frame}/{ann["total_frames"]}',flush=True)
        events=group_events(records,args.min_length)
        write_csv(output/'candidate_frames.csv',records,['video_frame','gt_frame','block_id','a','b','direct_a','direct_b','cross_a','cross_b','gain'])
        write_json(output/'events.json',events)
        write_csv(output/'baseline_prediction_hashes.csv',list(baseline.values()))
        report.update(status='completed_suggestions',eligible_frames=eligible_frames,candidate_pair_frames=len(records),
            proposed_intervals=len(events),proposed_pair_frames=sum(e['length'] for e in events),
            events_sha256=sha(output/'events.json',True),baseline_hashes_sha256=sha(output/'baseline_prediction_hashes.csv',True))
        write_json(output/'verification.json',report)
        render(output,[e['event_id'] for e in events[:args.preview_limit]])
    except Exception as exc:
        report.update(status='failed',error=str(exc));raise
    finally:
        write_json(output/'verification.json',report)
        print(f'Status: {report["status"]}; results: {output}',flush=True)
        if 'proposed_intervals' in report:print(f"Proposed intervals: {report['proposed_intervals']}; no corrections applied.")


def apply_review(args):
    review=args.review.resolve()
    report,prediction,reference,events,hashes=open_review(review)
    previews=json.loads((review/'preview_index.json').read_text())
    if not all(previews.get(e) for e in args.approve):raise ValueError('Render visual previews before approving these events')
    for name in args.approve:
        for item in previews[name]:
            path=(review/item['file']).resolve()
            if not path.is_relative_to(review) or not path.is_file():raise ValueError('Visual preview missing or outside review folder')
    if not args.reviewer.strip():raise ValueError('Reviewer name required')
    ann=json.loads((prediction/'annotations.json').read_text())
    mapping=approved_mapping(events,args.approve,ann['total_frames'],ann['block_size'],[b['video_frame'] for b in ann['blocks'].values()])
    output=new_output('swap-correction-');sources=[];changes=[]
    status=dict(status='in_progress',review=str(review),reviewer=args.reviewer,approved_events=args.approve,
        approval_method='Explicit command-line event approval after visual review',
        reference_assisted=True,baseline_prediction=str(prediction),
        limitations=['Identity ownership only; shapes unchanged.',
            'Post-hoc GT-assisted corrections must not be presented as independent scorer improvement.',
            'This is an overlay: resolve masks through mask_sources.csv. Unchanged files remain in the baseline.',
            'B is transferred with each mask. Recompute A/C before evaluating this corrected sequence.'])
    try:
        for frame in range(ann['total_frames']):
            for target in IDS:
                source=mapping.get((frame,target),target)
                path=prediction/'masks_by_gt_id'/str(source)/f'{frame:06d}.png'
                if sha(path)!=hashes[frame,source]:raise ValueError('Baseline changed before correction')
                if source!=target:
                    dest=output/'masks_by_gt_id'/str(target)/f'{frame:06d}.png';dest.parent.mkdir(parents=True,exist_ok=True)
                    dest.write_bytes(path.read_bytes())
                    if sha(dest)!=hashes[frame,source]:raise ValueError('Correction copy changed mask bytes')
                    changes.append(dict(video_frame=frame,target_gt_id=target,source_gt_id=source,sha256=sha(dest)))
                    resolved=dest
                else:resolved=path
                sources.append(dict(video_frame=frame,gt_id=target,source_gt_id=source,mask_path=str(resolved),sha256=hashes[frame,source]))
        confidence=unique_rows(read_csv(prediction/'foreground_confidence.csv'),('abs_frame','gt_id'))
        corrected=[]
        for frame in range(ann['total_frames']):
            for target in IDS:
                source=mapping.get((frame,target),target)
                corrected.append(dict(confidence[frame,source],gt_id=target,sam2_label=IDS.index(target)+1))
        write_csv(output/'mask_sources.csv',sources)
        write_csv(output/'corrections.csv',changes)
        write_csv(output/'foreground_confidence.csv',corrected)
        write_json(output/'approved_events.json',[e for e in events if e['event_id'] in args.approve])
        status.update(status='completed_identity_overlay',changed_bird_frames=len(changes),changed_video_frames=len({r['video_frame'] for r in changes}),
            input_hashes=report['input_hashes'],events_sha256=report['events_sha256'],mask_sources_sha256=sha(output/'mask_sources.csv',True))
    except Exception as exc:status.update(status='failed',error=str(exc));raise
    finally:
        write_json(output/'verification.json',status)
        print(f'Status: {status["status"]}; results: {output}',flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);sub=parser.add_subparsers(dest='command',required=True)
    p=sub.add_parser('propose')
    p.add_argument('--prediction',type=Path,default=Path('audit/results/propagation-20260911T180552787557Z'))
    p.add_argument('--reference',type=Path,default=Path('audit/results/task2-reference-20260915T080605348526Z'))
    p.add_argument('--closeout',type=Path,default=Path('audit/TASK2_CLOSEOUT.json'))
    p.add_argument('--task4',type=Path,default=Path('audit/results/task4-20260915T084226849510Z'))
    p.add_argument('--task4-closeout',type=Path,default=Path('audit/TASK4_CLOSEOUT.json'))
    p.add_argument('--cross-min',type=float,default=.5);p.add_argument('--gain-min',type=float,default=.2)
    p.add_argument('--min-length',type=int,default=3);p.add_argument('--preview-limit',type=int,default=12)
    p=sub.add_parser('preview');p.add_argument('--review',type=Path,required=True);p.add_argument('--events',nargs='+',required=True)
    p=sub.add_parser('apply');p.add_argument('--review',type=Path,required=True);p.add_argument('--approve',nargs='+',required=True);p.add_argument('--reviewer',required=True)
    args=parser.parse_args()
    if args.command=='propose':
        if not 0<args.cross_min<=1 or not 0<args.gain_min<=1 or args.min_length<1 or args.preview_limit<0:parser.error('Invalid thresholds or limits')
        propose(args)
    elif args.command=='preview':render(args.review.resolve(),args.events)
    else:apply_review(args)
