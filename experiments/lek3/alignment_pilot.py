"""Inspect stored prompts and compare a small sample of raw masks with video.

Read-only inputs. No SAM2, automatic alignment certification, or GT correction.
"""
import argparse
import csv
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from inspect_session_structure import Reader, Payload


def state(obj):
    value = getattr(obj, 'state', None)
    if not isinstance(value, dict):
        raise ValueError('Unsupported object state: ' + str(getattr(obj, 'stored_type', type(obj))))
    return value


def unpack_mask(obj, stream):
    if getattr(obj, 'stored_type', '') != 'annotator.PackedMasks':
        raise ValueError('Expected PackedMasks; refusing an inferred conversion')
    fields = state(obj)
    h, w = fields['h'], fields['w']
    if type(h) is not int or type(w) is not int or min(h, w) < 1 or h*w > 30_000_000:
        raise ValueError('Unexpected mask dimensions')
    array = fields['data']
    if getattr(array, 'stored_type', '') not in (
            'numpy._core.numeric._frombuffer', 'numpy.core.numeric._frombuffer'):
        raise ValueError('Unsupported packed array representation')
    payload, dtype, shape, order = array.args
    if (not isinstance(payload, Payload) or getattr(dtype, 'stored_type', '') != 'numpy.dtype'
            or dtype.args[0] not in ('u1', 'uint8') or order != 'C'
            or tuple(shape) != (h, (w+7)//8) or payload.size != h*((w+7)//8)):
        raise ValueError('Packed mask schema or payload size mismatch')
    stream.seek(payload.offset)
    data = stream.read(payload.size)
    if len(data) != payload.size:
        raise ValueError('Truncated packed mask')
    packed = np.frombuffer(data, dtype=np.uint8).reshape(shape)
    # Matches PackedMasks._unpack in annotator.py; do not interpolate labels.
    return np.unpackbits(packed, axis=-1)[:, :w].astype(bool)


def prompt_rows(backend):
    block_size = backend['block_size']
    rows = []
    labels = []
    for label in backend['sam_handler_labels']:
        fields = state(label)
        labels.append({'name': fields['name'], 'colour': fields.get('col')})
        for kind in ('pts', 'boxes'):
            for block, entries in fields.get(kind, {}).items():
                for item in entries:
                    point = state(item)
                    index = point['idx']
                    if type(block) is not int or type(index) is not int:
                        raise ValueError('Non-integer prompt index')
                    valid = 0 <= index < block_size and block >= 0
                    rows.append({'label': fields['name'], 'kind': kind,
                        'block': block, 'local_index': index,
                        'session_index_candidate': block*block_size+index if valid else None,
                        'index_status': 'block_local_interpretation' if valid else 'outside_block_range',
                        'x': point.get('x'), 'y': point.get('y'),
                        'fx': point.get('fx'), 'fy': point.get('fy'), 'pt_type': point.get('pt_type')})
    return labels, rows


def run(args):
    import cv2
    before = args.session.stat()
    with args.session.open('rb') as stream:
        backend = Reader(stream).load()
    labels, prompts = prompt_rows(backend)
    names = [v['name'] for v in labels]
    if len(names) != len(set(names)):
        raise ValueError('Duplicate label names')
    colours = [(255, 70, 70), (70, 240, 100), (255, 220, 40), (90, 160, 255)]
    if len(names) > len(colours):
        raise ValueError('Pilot currently supports up to four labels')
    frames = {}
    for key, masks in backend['masks'].items():
        index = int(Path(str(key).replace('\\', '/')).stem)
        if index in frames:
            raise ValueError('Duplicate numeric frame key')
        frames[index] = masks
    selected = [0, 250, 500, 4250, 6750, 10000, 13497]
    if any(i not in frames for i in selected):
        raise ValueError('Expected Lek3 sample frame is missing; inspect coverage first')
    folder = Path(__file__).resolve().parent / 'results' / (
        'pilot-' + datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ'))
    folder.mkdir(parents=True, exist_ok=False)
    with (folder / 'prompts.csv').open('w', newline='', encoding='utf-8') as f:
        columns = ['label', 'kind', 'block', 'local_index', 'session_index_candidate',
                   'index_status', 'x', 'y', 'fx', 'fy', 'pt_type']
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        writer.writerows(prompts)
    sampled = {}
    mask_info = []
    with args.session.open('rb') as stream:
        for index in selected:
            sampled[index] = {}
            for number, name in enumerate(names):
                if name not in frames[index]:
                    mask_info.append({'session_index': index, 'label': name, 'status': 'missing_entry'})
                    continue
                mask = unpack_mask(frames[index][name], stream)
                sampled[index][name] = mask
                Image.fromarray(mask.astype(np.uint8)*255).save(folder / f'mask_{index:06d}_label{number+1}.png')
                mask_info.append({'session_index': index, 'label': name,
                                  'status': 'present_entry', 'area': int(mask.sum()), 'shape': list(mask.shape)})
    block_size = backend['block_size']
    # Exploratory alternatives, NOT an accepted mapping. block+2 tests possible
    # cumulative extraction drift; +28 comes from one saved-overlay candidate.
    requests = {}
    for index in selected:
        for offset in sorted({0, 1, index//block_size+2, 28}):
            requests.setdefault(index+offset, []).append((index, offset))
    cap = cv2.VideoCapture(str(args.video))
    comparisons = []
    try:
        if not cap.isOpened():
            raise ValueError('Cannot open video')
        for video_index in range(max(requests)+1):
            ok, bgr = cap.read()
            if not ok:
                raise ValueError(f'Video read stopped at {video_index}')
            if video_index % 2000 == 0:
                print('Decoded:', video_index, flush=True)
            if video_index not in requests:
                continue
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            for index, offset in requests[video_index]:
                overlay = rgb.copy()
                missing = []
                for number, name in enumerate(names):
                    mask = sampled[index].get(name)
                    if mask is None:
                        missing.append(number+1)
                        continue
                    if mask.shape != rgb.shape[:2]:
                        raise ValueError('Raw GT and video dimensions differ; no resizing applied')
                    overlay[mask] = (0.65*overlay[mask]+0.35*np.array(colours[number])).astype(np.uint8)
                    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    cv2.drawContours(overlay, contours, -1, colours[number], 2)
                height = round(rgb.shape[0]*960/rgb.shape[1])
                panel = Image.new('RGB', (1920, height+65), 'white')
                panel.paste(Image.fromarray(rgb).resize((960, height)), (0, 65))
                panel.paste(Image.fromarray(overlay).resize((960, height)), (960, 65))
                draw = ImageDraw.Draw(panel)
                draw.text((10, 8), f'Video {video_index} | Stored GT {index} | candidate offset +{offset} | NOT CERTIFIED', fill='black')
                draw.text((10, 28), f'Left: photograph. Right: raw stored masks. Missing label entries: {missing}', fill='black')
                filename = f'comparison_{index:06d}_offset{offset:03d}.jpg'
                panel.save(folder / filename, quality=94)
                comparisons.append({'session_index': index, 'video_index': video_index,
                                    'candidate_offset': offset, 'preview': filename})
    finally:
        cap.release()
    after = args.session.stat()
    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        raise ValueError('Session changed during inspection')
    unique = sorted({r['session_index_candidate'] for r in prompts if r['session_index_candidate'] is not None})
    result = {'status': 'pilot_complete_correspondence_unverified',
        'session_file': args.session.name, 'video_file': args.video.name,
        'labels': [dict(v, preview_label=n+1, preview_rgb=colours[n]) for n,v in enumerate(labels)],
        'prompt_records': len(prompts), 'unique_prompt_session_indices_candidate': unique,
        'consecutive_prompt_spacings_candidate': [b-a for a,b in zip(unique, unique[1:])],
        'selected_session_indices': selected, 'mask_checks': mask_info, 'comparisons': comparisons,
        'limitations': ['Stored prompts may include later corrections; original annotation provenance is unverified.',
            'Prompt indices use block*block_size+local_index, not certified video indices.',
            'Offsets are exploratory alternatives; no automatic selection is made.',
            'Absent dictionary entries are unknown, not confirmed bird absence.',
            'Sparse visual checks cannot certify all frame correspondences.']}
    (folder / 'report.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    links = ''.join(f'<p>GT {r["session_index"]}, video {r["video_index"]}</p><img width="100%" src="{r["preview"]}">' for r in comparisons)
    (folder / 'index.html').write_text('<!doctype html><meta charset="utf-8"><title>Alignment pilot</title><h1>Candidate comparisons - alignment unverified</h1>'+links, encoding='utf-8')
    print('Stored prompt records:', len(prompts))
    print('Distinct candidate prompt frames:', len(unique))
    print('Status:', result['status'])
    print('Results:', folder)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--session', type=Path, required=True)
    parser.add_argument('--video', type=Path, required=True)
    run(parser.parse_args())
