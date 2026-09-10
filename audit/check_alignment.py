"""Search nearby video frames against selected FRAMES images, without changing data.

This diagnoses timing correspondence. Non-exact matches require visual review;
it does not estimate spatial transforms or automatically change IoU alignment.
"""
import argparse
import json
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw
from recompute import indexed, write_json, sha


def coverage(indices):
    indices=sorted(indices)
    return dict(count=len(indices),first=indices[0] if indices else None,
                last=indices[-1] if indices else None,
                missing_numeric_indices=sorted(set(range(indices[0],indices[-1]+1))-set(indices)) if indices else [])


def run(args):
    import cv2
    prior=args.results.resolve()
    manifest=json.loads((prior/'manifest.json').read_text())
    session=json.loads((prior/'session.json').read_text())
    alignment=json.loads((prior/'video_alignment.json').read_text())
    output=prior.parent/(prior.name+'-alignment')
    output.mkdir(exist_ok=False)
    frames=indexed(manifest['inputs']['frames'])
    if not frames:
        raise ValueError('No numbered FRAMES images found')
    samples=sorted((set(int(v) for v in session['best'].values()) |
                    set(range(min(frames),max(frames)+1,getattr(args,'step',250))) |
                    {min(frames),max(frames)}) & set(frames))
    capture=cv2.VideoCapture(alignment['video'])
    report=dict(status='in_progress', video=alignment['video'], radius=args.radius,
                note='Only exact pixel matches establish sampled image equality. Nearby best matches are diagnostic, not an approved offset.',
                samples=[], source_sha256=sha(__file__), frames_coverage=coverage(frames),
                decoding='Single sequential video pass from frame zero; no random seeking',
                unsampled_correspondence='Unverified; sample offsets are not applied to intermediate frames')
    try:
        if not capture.isOpened():
            raise ValueError('Cannot open video')
        total=int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        report['video_reported_count']=total
        for name in ['gt','masks']:
            if name in manifest['inputs']:
                entries=indexed(manifest['inputs'][name])
                report[name+'_coverage']=coverage(entries)
                report[name+'_without_frames']=sorted(set(entries)-set(frames))
                report['frames_without_'+name]=sorted(set(frames)-set(entries))
        cache={}
        next_index=0
        eof=False
        for frame in samples:
            with Image.open(frames[frame]) as image:
                reference=np.asarray(image.convert('RGB'))
            lo,hi=max(0,frame-args.radius),frame+args.radius+1
            # Bound memory while retaining overlapping search windows. Decode skipped
            # intervals too, so indices mean sequential decoded position, not seeks.
            cache={i:image for i,image in cache.items() if i>=lo}
            while next_index<hi and not eof:
                ok,decoded=capture.read()
                if not ok:
                    eof=True
                    break
                if next_index>=lo:
                    cache[next_index]=cv2.cvtColor(decoded,cv2.COLOR_BGR2RGB)
                next_index+=1
            candidates=[]
            best_image=None
            best_score=float('inf')
            for index in range(lo,min(hi,next_index)):
                decoded=cache[index]
                if decoded.shape != reference.shape:
                    raise ValueError('Video/FRAMES dimension mismatch')
                difference=np.abs(decoded.astype(np.int16)-reference.astype(np.int16))
                score=float(difference.mean())
                candidates.append(dict(video_index=index, offset_video_minus_frames=index-frame,
                                       mean_absolute_difference=score,max_difference=int(difference.max())))
                if score < best_score:
                    best_score=score;best_image=decoded.copy()
            if not candidates:
                raise ValueError(f'No decoded candidates for FRAMES index {frame}')
            candidates.sort(key=lambda r:r['mean_absolute_difference'])
            best=candidates[0]
            exact=[c['video_index'] for c in candidates if c['max_difference']==0]
            report['samples'].append(dict(frames_index=frame, frames_sha256=sha(frames[frame]),
                best=best,exact_video_matches=[c['video_index'] for c in candidates if c['max_difference']==0],
                correspondence_status='unique_exact_sample_match' if len(exact)==1 else 'ambiguous_exact_matches' if exact else 'requires_visual_review',
                best_at_search_boundary=best['video_index'] in [lo,min(hi,next_index)-1], candidates=candidates))
            sheet=Image.new('RGB',(1344,415),'white')
            sheet.paste(Image.fromarray(best_image).resize((672,380)),(0,35))
            sheet.paste(Image.fromarray(reference).resize((672,380)),(672,35))
            ImageDraw.Draw(sheet).text((8,8),f"Best video {best['video_index']} (left), FRAMES {frame} (right); offset {best['offset_video_minus_frames']}; exact={best['max_difference']==0}",fill='black')
            sheet.save(output/f'match_{frame:06d}.jpg',quality=93)
            print(f"FRAMES {frame}: best video {best['video_index']}, offset {best['offset_video_minus_frames']}, mean difference {best_score:.6f}",flush=True)
        while not eof:
            ok,_=capture.read()
            if not ok:
                eof=True
                break
            next_index+=1
        report['video_sequentially_decoded_count']=next_index
        report['decoded_count_matches_reported']=next_index==total
        report['count_difference_video_minus_frames']=next_index-len(frames)
        report['count_caution']='A count difference does not identify which frames were omitted or why decoding stopped.'
        unique=[s for s in report['samples'] if s['correspondence_status']=='unique_exact_sample_match']
        report['unique_exact_sample_offsets']=sorted(set(s['best']['offset_video_minus_frames'] for s in unique))
        report['unique_exact_mapping_monotonic']=all(a['best']['video_index']<b['best']['video_index'] for a,b in zip(unique,unique[1:]))
        report['exact_match_caution']='Unique means unique only within the searched window. Repeated imagery can still make temporal identity uncertain.'
        lines=['# Sampled video / FRAMES correspondence', '',
               f'Video header reports {total} frames; sequential decoding returned {next_index}.',
               f'FRAMES contains {len(frames)} numbered images.', '',
               'Positive offset means the best matching video index is later than the FRAMES index.',
               'Exact matches establish pixel equality at sampled positions, not a verified full-sequence time map.',
               'Non-exact matches, repeated images, search-boundary matches and unsampled intervals require further review.', '',
               '| FRAMES index | Best video index | Offset | Mean pixel difference | Evidence |',
               '|---:|---:|---:|---:|---|']
        for sample in report['samples']:
            best=sample['best']
            lines.append(f"| {sample['frames_index']} | {best['video_index']} | {best['offset_video_minus_frames']} | {best['mean_absolute_difference']:.6f} | {sample['correspondence_status']} |")
        (output/'correspondence.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
        report['status']='completed_diagnostic'
    except Exception as exc:
        report.update(status='failed',error=str(exc));raise
    finally:
        capture.release()
        write_json(output/'alignment_search.json',report)
        print(f'Results: {output}',flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results',type=Path,required=True)
    parser.add_argument('--radius',type=int,default=24)
    parser.add_argument('--step',type=int,default=250,help='Sample every N FRAMES indices plus annotation frames and endpoints')
    args=parser.parse_args()
    if args.radius < 1:
        parser.error('Radius must be positive')
    if args.step < 1:
        parser.error('Step must be positive')
    run(args)
