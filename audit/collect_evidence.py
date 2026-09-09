"""Read-only evidence collector. Standard library only; prints JSON to stdout.

Does not import project code, load pickle sessions, or change existing outputs.
Optional PNG inspection uses installed NumPy/OpenCV without requiring SAM2.
"""
import argparse
import csv
import hashlib
import json
import subprocess
from collections import Counter
from pathlib import Path


def git(repo, *args):
    result = subprocess.run(['git', '-C', str(repo), *args], capture_output=True,
                            text=True, timeout=30)
    return {'returncode': result.returncode, 'stdout': result.stdout.strip(),
            'stderr': result.stderr.strip()}


def metadata(path):
    return {'path': str(path), 'bytes': path.stat().st_size,
            'mtime_ns': path.stat().st_mtime_ns}


def csv_info(path):
    result = metadata(path)
    with path.open(newline='', encoding='utf-8-sig') as stream:
        reader = csv.DictReader(stream)
        result['columns'] = reader.fieldnames
        count, samples, blocks, labels = 0, [], Counter(), Counter()
        low, high, seen, duplicates = None, None, set(), 0
        for row in reader:
            count += 1
            if len(samples) < 3:
                samples.append(row)
            blocks[row.get('block_id', '')] += 1
            labels[row.get('label', row.get('sam2_label', ''))] += 1
            frame = row.get('abs_frame', row.get('frame_idx'))
            if frame is not None:
                try:
                    frame = int(frame)
                    low = frame if low is None else min(low, frame)
                    high = frame if high is None else max(high, frame)
                    key = (frame, row.get('label', row.get('sam2_label', '')))
                    duplicates += key in seen
                    seen.add(key)
                except ValueError:
                    pass
        result.update(rows=count, samples=samples, blocks=dict(blocks),
                      labels=dict(labels), frame_range=[low, high],
                      duplicate_frame_label_keys=duplicates)
    return result


def collect(root, repo):
    report = {'root': str(root), 'repo': str(repo),
              'git_head': git(repo, 'rev-parse', 'HEAD'),
              'git_status': git(repo, 'status', '--short'),
              'git_history': git(repo, 'log', '-12', '--format=%h %aI %s'),
              'sources': [], 'outputs': []}
    for name in ['cross_block_propagate.py', 'annotator.py', 'sam_annotator.py',
                 'mask_quality_scorer.py', 'iou_validator.py', 'visual_proof.py',
                 'hyperparam_search.py', 'patch_cross_block.py',
                 'sam2/sam2_video_predictor.py', 'sam2/sam2/sam2_video_predictor.py']:
        path = repo / name
        if path.is_file():
            data = path.read_bytes()
            report['sources'].append(dict(metadata(path),
                sha256=hashlib.sha256(data).hexdigest(),
                numbered_source=['%d: %s' % (i, line) for i, line in
                                 enumerate(data.decode('utf-8', errors='replace').splitlines(), 1)]))
    for name in ['headless_output_v9', 'headless_output_lek4_v2',
                 'iou_validation_v4', 'hyperparam_search']:
        directory = root / name
        item = {'path': str(directory), 'exists': directory.is_dir()}
        if directory.is_dir():
            files = sorted(p for p in directory.rglob('*') if p.is_file())
            item['extensions'] = dict(Counter(p.suffix for p in files))
            item['listing'] = [metadata(p) for p in files[:15] + files[-5:]]
            item['csv'] = []
            for path in files:
                if path.suffix.lower() == '.csv':
                    try:
                        item['csv'].append(csv_info(path))
                    except Exception as exc:
                        item['csv'].append({'path': str(path), 'error': str(exc)})
            pngs = [p for p in files if p.suffix.lower() == '.png']
            item['png_samples'] = []
            try:
                import cv2
                import numpy as np
                indices = sorted(set([0, len(pngs)//2, max(0, len(pngs)-1)]))
                for i in indices:
                    if not pngs:
                        break
                    p = pngs[i]
                    arr = cv2.imread(str(p), cv2.IMREAD_UNCHANGED)
                    sample = metadata(p)
                    if arr is None:
                        sample['error'] = 'Unreadable PNG'
                    else:
                        values = np.unique(arr.reshape(-1, arr.shape[-1]), axis=0) if arr.ndim == 3 else np.unique(arr)
                        sample.update(shape=list(arr.shape), dtype=str(arr.dtype),
                                      values=values[:32].tolist(), unique_count=len(values),
                                      channel_order='BGR/BGRA if multichannel')
                    item['png_samples'].append(sample)
            except ImportError:
                item['png_inspection'] = 'Skipped: OpenCV/NumPy unavailable'
        report['outputs'].append(item)
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--repo', type=Path)
    args = parser.parse_args()
    print(json.dumps(collect(args.root.resolve(), (args.repo or args.root / 'SAMannot').resolve()), indent=2))
