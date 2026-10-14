"""Rank nearby video images against a cached overlay; never certify alignment.

Sequential video decode. Original files remain unchanged. Overlay colours and
JPEG compression affect the score; candidates require visual inspection.
"""
import argparse
import heapq
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from PIL import Image


def appearance_score(reference, frame):
    if reference.shape != frame.shape:
        raise ValueError('Comparison images have different dimensions')
    return float(np.abs(reference.astype(np.float32) - frame.astype(np.float32)).mean())


def run(args):
    import cv2
    reference = cv2.imread(str(args.results / 'overlay_img.png'))
    if reference is None:
        raise ValueError('Cannot read saved overlay')
    report = json.loads((args.results / 'report.json').read_text(encoding='utf-8'))
    entry = report['current_media_entry'].replace('\\', '/')
    centre = int(Path(entry).stem)
    start, end = max(0, centre - args.radius), centre + args.radius
    small_ref = cv2.resize(reference, (672, 380), interpolation=cv2.INTER_AREA)
    cap = cv2.VideoCapture(str(args.video))
    candidates = []
    count = 0
    try:
        if not cap.isOpened():
            raise ValueError('Cannot open video')
        for index in range(end + 1):
            ok, frame = cap.read()
            if not ok:
                raise ValueError(f'Video reading stopped before search completed at {index}')
            if index % 2000 == 0:
                print('Decoded:', index, flush=True)
            if index < start:
                continue
            if frame.shape != reference.shape:
                raise ValueError('Video and saved overlay dimensions differ')
            score = appearance_score(small_ref, cv2.resize(frame, (672, 380),
                                                       interpolation=cv2.INTER_AREA))
            count += 1
            item = (-score, index, frame.copy())
            if len(candidates) < 5:
                heapq.heappush(candidates, item)
            elif score < -candidates[0][0]:
                heapq.heapreplace(candidates, item)
    finally:
        cap.release()
    folder = Path(__file__).resolve().parent / 'results' / (
        'overlay-search-' + datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ'))
    folder.mkdir(parents=True, exist_ok=False)
    ranked = []
    for rank, (negative, index, frame) in enumerate(sorted(candidates, key=lambda x: (-x[0], x[1])), 1):
        name = f'candidate_{rank}_{index:06d}.png'
        Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)).save(folder / name)
        ranked.append({'rank': rank, 'video_index': index, 'difference_from_saved_index': index-centre,
                       'mean_absolute_colour_difference': -negative, 'image': name})
    result = {'status': 'candidates_only_alignment_not_certified',
              'reference_results': args.results.name, 'saved_index': centre,
              'search_start': start, 'search_end': end, 'compared_frames': count,
              'method': 'Sequential decode; mean absolute BGR difference after area resize to 672x380',
              'limitations': ['Overlay colours can bias ranking.',
                              'A single cached image cannot certify mask indexing or a video-wide offset.',
                              'No match outside this search window has been tested.'],
              'candidates': ranked}
    (folder / 'report.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps(result, indent=2))
    print('Results:', folder)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results', type=Path, required=True)
    parser.add_argument('--video', type=Path, required=True)
    parser.add_argument('--radius', type=int, default=150)
    args = parser.parse_args()
    if args.radius < 1:
        parser.error('radius must be positive')
    run(args)
