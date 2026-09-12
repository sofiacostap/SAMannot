"""Task 2: read-only GT screening with traceable uncertainty and review queue.

Heuristics identify candidates, never automatically remove pixels or swap IDs.
No prediction masks are used to decide whether GT is suspicious.
"""
import argparse
import csv
import json
from collections import Counter,defaultdict
from datetime import datetime,timezone
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw
from recompute import indexed,gt_labels,sha,write_json,write_csv
from reannotate import IDS,COLORS
from preview_empty_intervals import intervals
from verify_correspondence import fingerprint


def component_flags(area,sizes):
    flags=[]
    if 0<area<50:flags.append('tiny_target_region')
    if len(sizes)>1:
        significant=sum(s>=max(50,.1*area) for s in sizes)
        flags.append('multiple_substantial_regions' if significant>=2 else 'small_disconnected_regions')
    return flags


def pair_swap_candidates(previous,current):
    """Strong pairwise overlap reversal; a review candidate, not identity proof."""
    a=np.zeros(previous.shape,np.uint8);b=np.zeros(current.shape,np.uint8)
    for i,g in enumerate(IDS,1):a[previous==g]=i;b[current==g]=i
    table=np.bincount((a.astype(np.int16)*5+b).ravel(),minlength=25).reshape(5,5)
    areas_a=table.sum(axis=1);areas_b=table.sum(axis=0)
    def iou(i,j):
        union=areas_a[i]+areas_b[j]-table[i,j]
        return float(table[i,j]/union) if union else 0.
    result=[]
    for i in range(1,5):
        for j in range(i+1,5):
            if min(areas_a[i],areas_a[j],areas_b[i],areas_b[j])<50:continue
            cross=(iou(i,j)+iou(j,i))/2;direct=(iou(i,i)+iou(j,j))/2
            if min(iou(i,j),iou(j,i))>.2 and cross-direct>.25:
                result.append((IDS[i-1],IDS[j-1]))
    return result


def scan(args):
    import cv2
    source=args.source.resolve();prior=args.prior.resolve()
    verification=json.loads((source/'verification.json').read_text())
    if verification['status']!='verified':raise ValueError('Verified source required')
    gtroot=Path(json.loads((prior/'manifest.json').read_text())['inputs']['gt'])
    gtfiles=indexed(gtroot)
    with (source/'index.csv').open(newline='') as f:
        mapping={int(r['gt_frames_index']):r for r in csv.DictReader(f) if r['gt_frames_index']}
    if set(mapping)!=set(gtfiles):raise ValueError('GT inventory differs from verified correspondence')
    output=Path(__file__).resolve().parent/'results'/('gt-audit-'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ'))
    output.mkdir()
    report=dict(status='in_progress',source_directory=str(source),gt_directory=str(gtroot),
        source_sha256=sha(__file__),source_verification_sha256=sha(source/'verification.json'),
        limitations=['Heuristics do not certify GT identity or mask boundaries.',
                     'No GT pixels are modified. No small component is automatically discarded.',
                     'Only selected candidate frames are previewed; all candidate intervals remain in the queue.',
                     'Task 2 stays open until review decisions and uncertain/excluded cases are recorded.'])
    metrics=[];candidates=[];hashes={};previous=None;prev_video=None;previous_stats={}
    def flag(frame,gtframe,bird,reason,detail=''):
        candidates.append(dict(video_frame=frame,gt_frame=gtframe,gt_id=str(bird),reason=reason,detail=detail))
    try:
        for n,(gtframe,path) in enumerate(sorted(gtfiles.items()),1):
            entry=mapping[gtframe];frame=int(entry['video_index']);gt=gt_labels(path);hashes[str(gtframe)]=sha(path)
            with Image.open(source/'frames'/f'{frame:06d}.png') as im:
                if (im.height,im.width)!=gt.shape:raise ValueError(f'GT dimensions differ at {frame}')
            unknown=sorted(set(map(int,np.unique(gt)))-set(IDS)-{0})
            for value in unknown:flag(frame,gtframe,value,'unexpected_label',f'{int(np.count_nonzero(gt==value))} pixels')
            current_stats={}
            for bird in IDS:
                mask=(gt==bird).astype(np.uint8);area=int(mask.sum())
                count,_,components,centres=cv2.connectedComponentsWithStats(mask,connectivity=8)
                sizes=components[1:,cv2.CC_STAT_AREA].tolist()
                xy=np.average(centres[1:],axis=0,weights=sizes) if area else None
                edge=bool(mask[0].any() or mask[-1].any() or mask[:,0].any() or mask[:,-1].any())
                for reason in component_flags(area,sizes):flag(frame,gtframe,bird,reason,str(sorted(sizes,reverse=True)))
                stats=dict(area=area,x=float(xy[0]) if xy is not None else None,y=float(xy[1]) if xy is not None else None)
                current_stats[bird]=stats
                metrics.append(dict(video_frame=frame,gt_frame=gtframe,gt_id=bird,area=area,components=count-1,
                    largest_component=max(sizes) if sizes else 0,touches_edge=edge,centroid_x=stats['x'],centroid_y=stats['y']))
                if prev_video==frame-1:
                    old=previous_stats[bird]
                    if min(old['area'],area)>=50:
                        jump=float(np.hypot(stats['x']-old['x'],stats['y']-old['y']))
                        if jump>150:flag(frame,gtframe,bird,'centroid_jump',f'{jump:.1f} pixels from previous video frame')
                        ratio=max(old['area'],area)/min(old['area'],area)
                        if ratio>3:flag(frame,gtframe,bird,'area_change',f'{ratio:.2f} fold')
            if prev_video==frame-1:
                for left,right in pair_swap_candidates(previous,gt):flag(frame,gtframe,f'{left}+{right}','possible_identity_exchange')
            # Previously reported example: GT/FRAMES 651, NOT video frame 651.
            if gtframe==651:flag(frame,gtframe,75,'previously_reported_case','Review reported split ID; not assumed confirmed')
            previous=gt;prev_video=frame;previous_stats=current_stats
            if n%250==0:print(f'GT audited: {n}/{len(gtfiles)}',flush=True)
        grouped=defaultdict(list)
        for row in candidates:grouped[(row['gt_id'],row['reason'])].append(row['video_frame'])
        priorities={'possible_identity_exchange':0,'previously_reported_case':0,'multiple_substantial_regions':1,
                    'centroid_jump':2,'area_change':2,'unexpected_label':3,'tiny_target_region':3,'small_disconnected_regions':4}
        events=[]
        for (bird,reason),frames in grouped.items():
            for start,end in intervals(frames):events.append(dict(gt_id=bird,reason=reason,start=start,end=end,length=end-start+1,decision='unreviewed',notes=''))
        events.sort(key=lambda r:(priorities[r['reason']],-r['length'],r['start'],r['gt_id']))
        for i,event in enumerate(events):event['event_id']=f'GT{i+1:05d}'
        video_to_entry={int(r['video_index']):r for r in mapping.values()}
        pages=['<!doctype html><meta charset="utf-8"><title>Task 2 GT audit</title><style>body{font:16px system-ui;margin:24px}img{max-width:100%}</style>',
            '<h1>GT audit: candidates for human review</h1><p>Left: verified photograph. Right: all target GT masks. Colours: 14 red, 38 cyan, 75 yellow, 113 purple. Unexpected IDs white. No prediction masks, corrections or annotation clicks shown.</p>',
            '<p>This is a limited preview of the full queue in review_queue.csv. A suspicious component may be a valid bird part. A clean scan does not certify identity consistency.</p>']
        rendered=set()
        for event in events[:args.preview_limit]:
            pages.append(f'<h2>{event["event_id"]}: GT {event["gt_id"]}, {event["reason"]}, video {event["start"]}–{event["end"]}</h2>')
            for frame in sorted(set([event['start'],(event['start']+event['end'])//2,event['end']])):
                filename=f'video{frame:06d}.jpg'
                if frame not in rendered:
                    entry=video_to_entry[frame];gf=int(entry['gt_frames_index'])
                    with Image.open(source/'frames'/f'{frame:06d}.png') as im:rgb=np.asarray(im.convert('RGB')).copy()
                    if fingerprint(rgb)!=entry['rgb_sha256']:raise ValueError('Preview photograph changed')
                    gt=gt_labels(gtfiles[gf]);overlay=rgb.copy()
                    for bird,color in zip(IDS,COLORS):
                        tint=np.array([int(color[i:i+2],16) for i in [1,3,5]])
                        mask=gt==bird;overlay[mask]=(.5*overlay[mask]+.5*tint).astype(np.uint8)
                    mask=(gt!=0)&~np.isin(gt,IDS);overlay[mask]=(.5*overlay[mask]+127).astype(np.uint8)
                    canvas=Image.new('RGB',(1920,585),'white');draw=ImageDraw.Draw(canvas)
                    draw.text((10,8),f'Video {frame} / GT-FRAMES {gf}. Original GT, no filtering. 14 red / 38 cyan / 75 yellow / 113 purple.',fill='black')
                    canvas.paste(Image.fromarray(rgb).resize((960,543)),(0,40));canvas.paste(Image.fromarray(overlay).resize((960,543)),(960,40))
                    canvas.save(output/filename,quality=95);rendered.add(frame)
                pages.append(f'<p>Video {frame}</p><img loading="lazy" src="{filename}">')
        (output/'index.html').write_text('\n'.join(pages),encoding='utf-8')
        write_csv(output/'review_queue.csv',events,['event_id','gt_id','reason','start','end','length','decision','notes'])
        report.update(status='awaiting_human_review',gt_frames_scanned=len(gtfiles),candidate_records=len(candidates),
            candidate_counts=dict(Counter(r['reason'] for r in candidates)),review_events=len(events),
            previewed_events=min(len(events),args.preview_limit),preview_images=len(rendered),
            screening_parameters=dict(tiny_area=50,substantial_component_min_pixels=50,substantial_component_fraction=.1,
                centroid_jump_pixels=150,area_ratio=3,pairwise_cross_iou_min=.2,pairwise_advantage_min=.25))
    except Exception as exc:report.update(status='failed',error=str(exc));raise
    finally:
        write_csv(output/'metrics.csv',metrics,['video_frame','gt_frame','gt_id','area','components','largest_component','touches_edge','centroid_x','centroid_y'])
        write_csv(output/'candidates.csv',candidates,['video_frame','gt_frame','gt_id','reason','detail'])
        write_json(output/'gt_hashes.json',hashes);write_json(output/'verification.json',report)
        print(f"Status: {report['status']}; results: {output}",flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',required=True,type=Path)
    parser.add_argument('--prior',required=True,type=Path)
    parser.add_argument('--preview-limit',type=int,default=30)
    args=parser.parse_args()
    if args.preview_limit<0:parser.error('preview-limit must be nonnegative')
    scan(args)
