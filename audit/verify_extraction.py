"""Read-only reproduction of historical block decoding, before JPEG encoding."""
import argparse
import json
from collections import defaultdict, Counter
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw
from recompute import sha, write_json, write_csv
from verify_correspondence import fingerprint


def run(prior):
    import cv2
    prior = prior.resolve()
    manifest = json.loads((prior/'manifest.json').read_text())
    session = json.loads((prior/'session.json').read_text())
    video = json.loads((prior/'video_alignment.json').read_text())['video']
    output = prior.parent/('extraction-'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ'))
    output.mkdir()
    report = dict(status='in_progress', video=video, opencv=cv2.__version__,
        source_sha256=sha(__file__), video_sha256=sha(video),
        limitations=['Reproduces captured extract_block_frames lines 84-98 before JPEG encoding; does not run SAM2.',
                     'Current decoder behaviour cannot certify historical run provenance.',
                     'Saved coordinates are displayed unchanged; original GUI coordinate transforms still need review.'])
    expected = manifest.get('correspondence', {}).get('video_sha256')
    if expected and expected != report['video_sha256']:
        raise ValueError('Video changed since correspondence verification')
    lookup = defaultdict(list)
    reference = {}
    best = {int(k): v for k,v in session['best'].items()}
    cap = cv2.VideoCapture(video)
    try:
        if not cap.isOpened(): raise ValueError('Cannot open video')
        report['backend'] = cap.getBackendName()
        i = 0
        while True:
            ok,bgr = cap.read()
            if not ok: break
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            lookup[fingerprint(rgb)].append(i)
            if i in best.values(): reference[i] = rgb.copy()
            i += 1
            if i%250 == 0: print(f'Sequential reference: {i}', flush=True)
    finally:
        cap.release()
    report['sequential_count'] = i
    rows = []
    try:
        for block, annotation in sorted(best.items()):
            cap = cv2.VideoCapture(video)
            try:
                if not cap.isOpened(): raise ValueError('Cannot reopen video')
                total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                start = block*session['block_size']
                end = min((block+1)*session['block_size'], total-1)
                extra_end = min(end+1, total-1)
                seek_ok = cap.set(cv2.CAP_PROP_POS_FRAMES, start)
                for intended in range(start, extra_end+1):
                    ok,bgr = cap.read()
                    if not ok:
                        rows.append(dict(block=block,intended_video_index=intended,status='decode_failed'))
                        break
                    rgb = cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB)
                    candidates = lookup.get(fingerprint(rgb), [])
                    actual = candidates[0] if len(candidates)==1 else None
                    rows.append(dict(block=block,intended_video_index=intended,
                        actual_video_index=actual, offset=actual-intended if actual is not None else None,
                        candidates=';'.join(map(str,candidates)), seek_return=bool(seek_ok),
                        status='unique_exact' if actual is not None else 'ambiguous' if candidates else 'no_exact_match'))
                    if intended == annotation:
                        canvas = Image.new('RGB',(1344,430),'white')
                        draw = ImageDraw.Draw(canvas)
                        for side,pixels in enumerate([rgb,reference[annotation]]):
                            canvas.paste(Image.fromarray(pixels).resize((672,380)),(side*672,50))
                            for point in session['prompts']:
                                if point['abs_frame'] != annotation: continue
                                x=side*672+point['x']*672/pixels.shape[1]
                                y=50+point['y']*380/pixels.shape[0]
                                draw.ellipse((x-5,y-5,x+5,y+5),outline='yellow',width=2)
                                draw.text((x+7,y),str(point['sam2_label']),fill='yellow')
                        draw.text((8,5),f'Block {block}, requested {annotation}: old extraction left / sequential reference right',fill='black')
                        draw.text((8,23),f'Left exact sequential candidates: {candidates}. Numbers are saved SAM labels, NOT GT clicks.',fill='black')
                        canvas.save(output/f'clicks_block{block}.jpg',quality=94)
                        Image.fromarray(rgb).save(output/f'extracted_annotation_block{block}.png')
                print(f'Block {block} checked',flush=True)
            finally:
                cap.release()
        report.update(status='completed', match_counts=dict(Counter(r['status'] for r in rows)),
            blocks=[dict(block=b,offset_counts=dict(Counter(str(r.get('offset')) for r in rows if r['block']==b)),
                         annotation=[r for r in rows if r['block']==b and r['intended_video_index']==best[b]]) for b in sorted(best)])
    except Exception as exc:
        report.update(status='failed',error=str(exc))
        raise
    finally:
        write_csv(output/'decoded_mapping.csv',rows)
        write_json(output/'verification.json',report)
        print(f'Results: {output}',flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results',required=True,type=Path)
    run(parser.parse_args().results)
