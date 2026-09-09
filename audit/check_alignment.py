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


def run(args):
    import cv2
    prior=args.results.resolve()
    manifest=json.loads((prior/'manifest.json').read_text())
    session=json.loads((prior/'session.json').read_text())
    alignment=json.loads((prior/'video_alignment.json').read_text())
    output=prior.parent/(prior.name+'-alignment')
    output.mkdir(exist_ok=False)
    frames=indexed(manifest['inputs']['frames'])
    samples=sorted(set(int(v) for v in session['best'].values()) | {min(frames),max(frames)})
    capture=cv2.VideoCapture(alignment['video'])
    report=dict(status='in_progress', video=alignment['video'], radius=args.radius,
                note='Only exact pixel matches establish sampled image equality. Nearby best matches are diagnostic, not an approved offset.',
                samples=[], source_sha256=sha(__file__))
    try:
        if not capture.isOpened():
            raise ValueError('Cannot open video')
        total=int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        report['video_reported_count']=total
        for frame in samples:
            with Image.open(frames[frame]) as image:
                reference=np.asarray(image.convert('RGB'))
            lo,hi=max(0,frame-args.radius),min(total,frame+args.radius+1)
            capture.set(cv2.CAP_PROP_POS_FRAMES,lo)
            candidates=[]
            best_image=None
            best_score=float('inf')
            for index in range(lo,hi):
                ok, decoded=capture.read()
                if not ok:
                    break
                decoded=cv2.cvtColor(decoded,cv2.COLOR_BGR2RGB)
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
            report['samples'].append(dict(frames_index=frame, frames_sha256=sha(frames[frame]),
                best=best,exact_video_matches=[c['video_index'] for c in candidates if c['max_difference']==0],
                best_at_search_boundary=best['video_index'] in [lo,hi-1], candidates=candidates))
            sheet=Image.new('RGB',(1344,415),'white')
            sheet.paste(Image.fromarray(best_image).resize((672,380)),(0,35))
            sheet.paste(Image.fromarray(reference).resize((672,380)),(672,35))
            ImageDraw.Draw(sheet).text((8,8),f"Best video {best['video_index']} (left), FRAMES {frame} (right); offset {best['offset_video_minus_frames']}; exact={best['max_difference']==0}",fill='black')
            sheet.save(output/f'match_{frame:06d}.jpg',quality=93)
            print(f"FRAMES {frame}: best video {best['video_index']}, offset {best['offset_video_minus_frames']}, mean difference {best_score:.6f}",flush=True)
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
    args=parser.parse_args()
    if args.radius < 1:
        parser.error('Radius must be positive')
    run(args)
