"""Create a lossless sequential image source; preserve every video frame."""
import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw
from recompute import sha, write_json, write_csv
from verify_correspondence import fingerprint


def run(prior, correspondence):
    import cv2
    prior=prior.resolve()
    verification=json.loads((correspondence/'verification.json').read_text())
    mapping=json.loads((correspondence/'frame_mapping.json').read_text())
    if verification['status'] != 'verified_image_correspondence':
        raise ValueError('Correspondence must be verified first')
    expected={r['video_index']:r for r in mapping if r['status']=='unique_exact'}
    if len(expected)!=len(mapping): raise ValueError('Ambiguous or duplicate mapping')
    video=verification['video']
    if sha(video)!=verification['video_file_sha256']: raise ValueError('Source video changed')
    session=json.loads((prior/'session.json').read_text())
    best={v:int(k) for k,v in session['best'].items()}
    output=prior.parent/('verified-source-'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ'))
    frames=output/'frames';frames.mkdir(parents=True)
    report=dict(status='in_progress',video=video,video_sha256=verification['video_file_sha256'],
        source_sha256=sha(__file__),opencv=cv2.__version__,frames_directory=str(frames),
        note='Sequential video indices. Saved clicks are unreviewed session coordinates, not new prompts. No inference or GT edits.')
    rows=[];cap=cv2.VideoCapture(video)
    try:
        if not cap.isOpened(): raise ValueError('Cannot open video')
        i=0
        while True:
            ok,bgr=cap.read()
            if not ok: break
            rgb=cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB)
            digest=fingerprint(rgb)
            match=expected.get(i)
            if match and digest!=match['rgb_sha256']:
                raise ValueError(f'Previously verified image differs at video index {i}')
            path=frames/f'{i:06d}.png'
            Image.fromarray(rgb).save(path,compress_level=1)
            with Image.open(path) as saved:
                if fingerprint(np.asarray(saved.convert('RGB')))!=digest:
                    raise ValueError(f'Saved image failed round-trip verification: {i}')
            rows.append(dict(video_index=i,gt_frames_index=match['frames_index'] if match else None,
                             rgb_sha256=digest,file=path.name))
            if i in best:
                panel=Image.new('RGB',(1344,810),'white')
                panel.paste(Image.fromarray(rgb).resize((1344,760)),(0,50))
                draw=ImageDraw.Draw(panel)
                draw.text((8,8),f'Block {best[i]}, verified video frame {i}. Saved clicks: review before reuse.',fill='black')
                for point in session['prompts']:
                    if point['abs_frame']!=i: continue
                    x=point['x']*1344/rgb.shape[1]; y=50+point['y']*760/rgb.shape[0]
                    draw.ellipse((x-7,y-7,x+7,y+7),outline='yellow',width=2)
                    draw.text((x+10,y),point['label'],fill='yellow')
                panel.save(output/f'review_block{best[i]}.jpg',quality=95)
            i+=1
            if i%250==0: print(f'Saved and verified {i} frames',flush=True)
        if i!=verification['video_decoded_count'] or not set(expected).issubset(range(i)):
            raise ValueError('Incomplete decode')
        report.update(status='verified',count=i,gt_matched_count=len(expected),
                      without_gt=[r['video_index'] for r in rows if r['gt_frames_index'] is None])
    except Exception as exc:
        report.update(status='failed',error=str(exc));raise
    finally:
        cap.release()
        write_csv(output/'index.csv',rows,['video_index','gt_frames_index','rgb_sha256','file'])
        write_json(output/'verification.json',report)
        print(f"Status: {report['status']}; results: {output}",flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results',required=True,type=Path)
    parser.add_argument('--correspondence',required=True,type=Path)
    args=parser.parse_args()
    run(args.results,args.correspondence)
