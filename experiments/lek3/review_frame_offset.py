"""Build a local fixed-GT viewer over a sequentially decoded video window.

Uses masks already exported by alignment_pilot.py. No source data is changed.
The selected offset is a visual candidate, never automatic alignment certification.
"""
import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from PIL import Image


def load_layers(results, gt_frame):
    report = json.loads((results / 'report.json').read_text(encoding='utf-8'))
    if gt_frame not in report['selected_session_indices']:
        raise ValueError('This GT frame was not exported in the selected pilot')
    layers = []
    shape = None
    for label in report['labels']:
        number = label['preview_label']
        check = next(x for x in report['mask_checks']
                     if x['session_index'] == gt_frame and x['label'] == label['name'])
        if check['status'] == 'missing_entry':
            layers.append({'label': number, 'name': label['name'], 'missing': True})
            continue
        with Image.open(results / f'mask_{gt_frame:06d}_label{number}.png') as image:
            mask = np.asarray(image)
        if mask.ndim != 2 or not np.isin(mask, [0, 255]).all():
            raise ValueError('Expected an unchanged binary exported mask')
        if list(mask.shape) != check['shape'] or int((mask > 0).sum()) != check['area']:
            raise ValueError('Exported mask does not agree with pilot metadata')
        if shape is not None and mask.shape != shape:
            raise ValueError('Mask dimensions differ')
        shape = mask.shape
        rgba = np.zeros((*shape, 4), dtype=np.uint8)
        rgba[..., :3] = label['preview_rgb']
        rgba[..., 3] = mask
        layers.append({'label': number, 'name': label['name'], 'missing': False,
                       'colour': label['preview_rgb'], 'rgba': rgba})
    if shape is None:
        raise ValueError('No masks are available for this frame')
    return shape, layers


def run(args):
    import cv2
    shape, layers = load_layers(args.results, args.gt_frame)
    start, requested_end = max(0, args.gt_frame-args.radius), args.gt_frame+args.radius
    folder = Path(__file__).resolve().parent / 'results' / (
        'offset-view-' + datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ'))
    folder.mkdir(parents=True, exist_ok=False)
    (folder / 'frames').mkdir()
    # The local review cache is reproducible and need not be uploaded.
    (folder / '.gitignore').write_text('frames/\nmask_*.png\n', encoding='utf-8')
    for layer in layers:
        if not layer['missing']:
            filename = f'mask_{layer["label"]}.png'
            Image.fromarray(layer.pop('rgba')).save(folder / filename)
            layer['file'] = filename
    cap = cv2.VideoCapture(str(args.video))
    last = None
    try:
        if not cap.isOpened():
            raise ValueError('Video could not be opened')
        for index in range(requested_end+1):
            ok, frame = cap.read()
            if not ok:
                raise ValueError(f'Decoding stopped at {index}; requested window is incomplete')
            if index % 1000 == 0:
                print('Decoded:', index, flush=True)
            if index < start:
                continue
            if frame.shape[:2] != shape:
                raise ValueError('Video and mask dimensions differ; no resizing applied')
            if not cv2.imwrite(str(folder / 'frames' / f'{index:06d}.jpg'), frame,
                               [cv2.IMWRITE_JPEG_QUALITY, 94]):
                raise OSError('Could not save review photograph')
            last = index
    finally:
        cap.release()
    config = {'gt': args.gt_frame, 'start': start, 'end': last,
              'width': shape[1], 'height': shape[0], 'layers': layers,
              'pilot': args.results.name, 'video_file': args.video.name,
              'status': 'visual_review_only',
              'limitations': ['JPEG previews are for visual review, not pixel equality checks.',
                              'A selected offset applies only to this reviewed GT frame until further verification.']}
    template = Path(__file__).with_suffix('.html').read_text(encoding='utf-8')
    (folder / 'index.html').write_text(template.replace('__CONFIG__',
        json.dumps(config).replace('<', '\\u003c')), encoding='utf-8')
    (folder / 'report.json').write_text(json.dumps(config, indent=2), encoding='utf-8')
    print('Viewer ready. Open:', folder / 'index.html')
    print('Use Left/Right to step one frame; Shift+Left/Right to step ten.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results', type=Path, required=True)
    parser.add_argument('--video', type=Path, required=True)
    parser.add_argument('--gt-frame', type=int, required=True)
    parser.add_argument('--radius', type=int, default=100)
    args = parser.parse_args()
    if args.gt_frame < 0 or not 1 <= args.radius <= 500:
        parser.error('gt-frame must be non-negative; radius must be 1 to 500')
    run(args)
