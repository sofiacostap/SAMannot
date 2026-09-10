"""Verify every FRAMES image against all sequentially decoded video images.

Compares lossless RGB pixel fingerprints, including dimensions. No offsets are
assumed, and duplicate-image candidates remain ambiguous. Writes only a new
directory beside the supplied prior result directory.
"""
import argparse
import hashlib
import json
from collections import defaultdict, Counter
from pathlib import Path
import numpy as np
from PIL import Image
from recompute import indexed, write_json, write_csv, sha


def fingerprint(rgb):
    rgb=np.ascontiguousarray(rgb,dtype=np.uint8)
    digest=hashlib.sha256(str(rgb.shape).encode('ascii'))
    digest.update(memoryview(rgb))
    return digest.hexdigest()


def match_records(frame_hashes, video_hashes):
    lookup=defaultdict(list)
    for index,value in enumerate(video_hashes):
        lookup[value].append(index)
    rows=[]
    for frame,value in sorted(frame_hashes.items()):
        matches=lookup.get(value,[])
        unique=len(matches)==1
        rows.append(dict(frames_index=frame, video_index=matches[0] if unique else None,
            offset_video_minus_frames=matches[0]-frame if unique else None,
            status='unique_exact' if unique else 'ambiguous_exact' if matches else 'no_exact_match',
            candidate_video_indices=matches, rgb_sha256=value))
    return rows


def run(args):
    import cv2
    prior=args.results.resolve()
    source=json.loads((prior/'manifest.json').read_text())
    video=json.loads((prior/'video_alignment.json').read_text())['video']
    output=prior.parent/(prior.name+'-full-correspondence')
    output.mkdir(exist_ok=False)
    report=dict(status='in_progress', video=video, frame_source=source['inputs']['frames'],
                method='SHA-256 of RGB shape and full decoded pixel bytes; sequential decode, no seeking',
                source_sha256=sha(__file__),
                limitations=['Pixel identity establishes image correspondence, not why frames were omitted.',
                             'Duplicate images retain all candidates and are not assigned automatically.',
                             'Historical SAM PNGs still require verification that their filenames refer to these video indices.'])
    cap=None
    try:
        files=indexed(source['inputs']['frames'])
        if not files: raise ValueError('No FRAMES images')
        hashes={};sizes={}
        for number,(index,path) in enumerate(sorted(files.items()),1):
            with Image.open(path) as image:
                rgb=np.asarray(image.convert('RGB'))
                hashes[index]=fingerprint(rgb);sizes[index]=image.size
            if number%250==0: print(f'FRAMES fingerprints: {number}/{len(files)}',flush=True)
        cap=cv2.VideoCapture(video)
        if not cap.isOpened(): raise ValueError('Cannot open video')
        reported=int(cap.get(cv2.CAP_PROP_FRAME_COUNT));video_hashes=[]
        while True:
            ok,bgr=cap.read()
            if not ok: break
            video_hashes.append(fingerprint(cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB)))
            if len(video_hashes)%250==0: print(f'Video fingerprints: {len(video_hashes)} (header: {reported})',flush=True)
        rows=match_records(hashes,video_hashes)
        unique=[r for r in rows if r['status']=='unique_exact']
        monotonic=all(a['video_index']<b['video_index'] for a,b in zip(unique,unique[1:]))
        used={v for r in rows for v in r['candidate_video_indices']}
        not_represented=sorted(set(range(len(video_hashes)))-used)
        runs=[]
        for row in unique:
            if runs and row['frames_index']==runs[-1]['frames_last']+1 and row['offset_video_minus_frames']==runs[-1]['offset']:
                runs[-1]['frames_last']=row['frames_index'];runs[-1]['video_last']=row['video_index']
            else:
                runs.append(dict(frames_first=row['frames_index'],frames_last=row['frames_index'],
                                 video_first=row['video_index'],video_last=row['video_index'],offset=row['offset_video_minus_frames']))
        gt=indexed(source['inputs']['gt']);pred=indexed(source['inputs']['masks'])
        shape_mismatches=[]
        for index,path in sorted(gt.items()):
            if index in sizes:
                with Image.open(path) as image:
                    if image.size!=sizes[index]: shape_mismatches.append(index)
        unresolved=[r['frames_index'] for r in rows if r['status']!='unique_exact']
        report.update(status='verified_image_correspondence' if not unresolved and monotonic and len(video_hashes)==reported else 'requires_review',
            frames_count=len(files),video_reported_count=reported,video_decoded_count=len(video_hashes),
            match_status_counts=dict(Counter(r['status'] for r in rows)),
            unique_mapping_monotonic=monotonic,unresolved_frames=unresolved,
            constant_offset_intervals=runs,video_indices_without_any_identical_FRAMES_image=not_represented,
            gt_indices_without_frames=sorted(set(gt)-set(files)),frames_indices_without_gt=sorted(set(files)-set(gt)),
            gt_frames_dimension_mismatch_indices=shape_mismatches,
            proposed_video_index_prediction_missing=[dict(frames_index=r['frames_index'],video_index=r['video_index']) for r in unique if r['video_index'] not in pred],
            video_file_sha256=sha(video))
        write_json(output/'frame_mapping.json',rows)
        write_csv(output/'frame_mapping.csv',[dict(r,candidate_video_indices=';'.join(map(str,r['candidate_video_indices']))) for r in rows])
        lines=['# Full image correspondence check','',f"Status: {report['status']}",
               f'FRAMES images: {len(files)}; sequentially decoded video images: {len(video_hashes)}; header: {reported}.',
               f'Match counts: {report["match_status_counts"]}', '',
               'Intervals below contain contiguous uniquely matched FRAMES indices. Positive offset means video index = FRAMES index + offset.', '',
               '| FRAMES interval | Video interval | Offset |','|---|---|---:|']
        for segment in runs:
            lines.append(f"| {segment['frames_first']}–{segment['frames_last']} | {segment['video_first']}–{segment['video_last']} | {segment['offset']} |")
        lines.extend(['',f'Video indices without an identical FRAMES image: {not_represented}',
            f'Unresolved FRAMES indices: {unresolved}', '',
            'This verifies image correspondence only. It neither cleans GT identities nor certifies old SAM PNG filename provenance.'])
        (output/'correspondence.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
        print(f"Correspondence: {report['match_status_counts']}; status: {report['status']}",flush=True)
    except Exception as exc:
        report.update(status='failed',error=str(exc));raise
    finally:
        if cap is not None: cap.release()
        write_json(output/'verification.json',report)
        print(f'Results: {output}',flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results',type=Path,required=True)
    run(parser.parse_args())
