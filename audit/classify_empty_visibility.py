"""Classify audited empty predictions using aligned GT, without running SAM2.

GT absence is evidence, not proof of biological absence. This is a research
diagnostic, not a deployable GT-free visibility detector.
"""
import argparse
import csv
import json
from collections import Counter,defaultdict
from datetime import datetime,timezone
from pathlib import Path
import numpy as np
from PIL import Image
from recompute import indexed,gt_labels,sha,write_json,write_csv
from preview_empty_intervals import intervals


def visibility_category(area):
    if area is None:return 'gt_unavailable'
    if area==0:return 'gt_absent_candidate_expected_absence'
    return 'gt_present_candidate_tracking_loss'


def run(args):
    review=args.review.resolve()
    report=json.loads((review/'verification.json').read_text())
    if report['status']!='completed' or report['issues']:raise ValueError('Resolve mask audit issues first')
    propagation=Path(report['propagation'])
    manifest=json.loads((propagation/'verification.json').read_text())
    source=Path(manifest['source_directory'])
    with (source/'index.csv').open(newline='') as f:mapping={int(r['video_index']):r for r in csv.DictReader(f)}
    with (review/'flags.csv').open(newline='') as f:flags=list(csv.DictReader(f))
    keys=[(int(r['video_frame']),int(r['gt_id'])) for r in flags]
    if len(keys)!=len(set(keys)) or len(keys)!=report['empty_records']:raise ValueError('Flags changed or contain duplicates')
    gtroot=json.loads((args.prior/'manifest.json').read_text())['inputs']['gt'];gtfiles=indexed(gtroot)
    output=Path(__file__).resolve().parent/'results'/('visibility-review-'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ'))
    output.mkdir();rows=[];grouped=defaultdict(list);gt_hashes={}
    for row in flags:grouped[int(row['video_frame'])].append(row)
    for n,(frame,items) in enumerate(sorted(grouped.items()),1):
        mapped=mapping[frame];gtindex=int(mapped['gt_frames_index']) if mapped['gt_frames_index'] else None
        gt=None
        if gtindex is not None:
            path=gtfiles[gtindex];gt=gt_labels(path);gt_hashes[str(gtindex)]=sha(path)
            with Image.open(source/'frames'/f'{frame:06d}.png') as image:
                if (image.height,image.width)!=gt.shape:raise ValueError('GT/image dimensions differ')
        for item in items:
            bird=int(item['gt_id']);area=int(np.count_nonzero(gt==bird)) if gt is not None else None
            rows.append(dict(video_frame=frame,gt_id=bird,block=int(item['block']),direction=item['direction'],
                gt_frame=gtindex,gt_area=area,category=visibility_category(area),
                tiny_gt_region=area is not None and 0<area<50))
        if n%250==0:print(f'Checked GT for {n}/{len(grouped)} flagged frames',flush=True)
    groups=defaultdict(list)
    for r in rows:groups[(r['block'],r['gt_id'],r['direction'],r['category'])].append(r['video_frame'])
    events=[]
    for (b,g,d,c),frames in sorted(groups.items()):
        for start,end in intervals(frames):events.append(dict(block=b,gt_id=g,direction=d,category=c,start=start,end=end,length=end-start+1))
    write_csv(output/'classified_flags.csv',rows,['video_frame','gt_id','block','direction','gt_frame','gt_area','category','tiny_gt_region'])
    write_csv(output/'events.csv',events,['block','gt_id','direction','category','start','end','length'])
    write_json(output/'verification.json',dict(status='completed',empty_records=len(rows),
        category_counts=dict(Counter(r['category'] for r in rows)),
        by_block={str(b):dict(Counter(r['category'] for r in rows if r['block']==b)) for b in sorted(set(r['block'] for r in rows))},
        tiny_gt_records=sum(r['tiny_gt_region'] for r in rows),review=str(review),
        flags_sha256=sha(review/'flags.csv'),source_sha256=sha(__file__),gt_file_hashes=gt_hashes,
        note='GT absence does not prove the bird left view; missing/incorrect GT and occlusion remain possible. Present GT is a candidate tracking-loss flag, not human validation. No GT-free detector implemented.'))
    print(f'Completed: {output}',flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--review',type=Path,required=True)
    parser.add_argument('--prior',type=Path,required=True)
    run(parser.parse_args())
