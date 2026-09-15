"""Build a conservative, provisional bird/frame partition from existing evidence.

No mask edits, no inference, no biological corrections. Encoding is a release gate.
"""
import argparse
import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path


def read_csv(path):
    with path.open(newline='', encoding='utf-8') as stream:
        return list(csv.DictReader(stream))


def partition(metrics, candidates, ledger):
    reasons = defaultdict(set)
    keys = {(int(r['video_frame']), int(r['gt_id'])) for r in metrics}
    if len(keys) != len(metrics):
        raise ValueError('Duplicate GT bird/frame keys')
    for r in candidates:
        if r['reason'] == 'unexpected_label':
            continue  # One encoding investigation, not 115,279 manual decisions.
        for bird in str(r['gt_id']).split('+'):
            key = (int(r['video_frame']), int(bird))
            if key not in keys:
                raise ValueError('Candidate not in GT inventory')
            reasons[key].add(r['reason'])
            # Both sides of an abrupt change or exchange are uncertain.
            if r['reason'] in ('centroid_jump', 'area_change', 'possible_identity_exchange'):
                previous = (key[0] - 1, key[1])
                if previous in keys:
                    reasons[previous].add('adjacent_to_' + r['reason'])
    explicit = defaultdict(set)
    held = set()
    for d in ledger['decisions']:
        for frame in range(d['video_first'], d['video_last'] + 1):
            for gt_id in d['gt_ids']:
                key = (frame, gt_id)
                if key not in keys:
                    raise ValueError('Review decision outside GT inventory')
                held.add(key)
                explicit[key].add(d['decision'])
                reasons[key].add('review_' + d['authority'])
    result = []
    for r in metrics:
        frame, gt_id, gt_frame = int(r['video_frame']), int(r['gt_id']), int(r['gt_frame'])
        if gt_frame != (frame if frame < 3 else frame - 12) or 3 <= frame <= 14:
            raise ValueError('Unexpected image correspondence')
        key = (frame, gt_id)
        status = ('exclude_from_evaluation' if 'exclude_from_evaluation' in explicit[key]
                  else 'uncertain' if reasons[key] else 'screened_pending_encoding')
        result.append(dict(video_frame=frame, gt_frame=gt_frame, gt_id=gt_id,
                           block_id=frame // 750, status=status,
                           reasons=';'.join(sorted(reasons[key])),
                           gt_present=int(r['area']) > 0, gt_area=int(r['area'])))
    return result, held


def run(args):
    ledger = json.loads(args.review.read_text())
    rows, held = partition(read_csv(args.audit / 'metrics.csv'),
                           read_csv(args.audit / 'candidates.csv'), ledger)
    args.output.mkdir(parents=True, exist_ok=False)
    with (args.output / 'evaluation_set.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    counts = Counter(r['status'] for r in rows)
    summary = dict(status='provisional_encoding_gate_not_passed', task2_complete=False,
                   gt_images=len({r['gt_frame'] for r in rows}), gt_bird_frames=len(rows),
                   review_held_bird_frames=len(held),
                   review_held_video_frames=len({f for f, g in held}),
                   counts=dict(counts),
                   by_gt_id={str(g): dict(Counter(r['status'] for r in rows if r['gt_id'] == g))
                             for g in (14, 38, 75, 113)},
                   by_block={str(b): dict(Counter(r['status'] for r in rows if r['block_id'] == b))
                             for b in range(6)},
                   screened_present_absent=dict(Counter('present' if r['gt_present'] else 'absent'
                       for r in rows if r['status'] == 'screened_pending_encoding')),
                   missing_gt_video_frames=list(range(3, 15)),
                   limitations=['Screening is not exhaustive biological certification.',
                     'Small fragments are held conservatively pending the single encoding investigation; not manual review tasks.',
                     'No reference row is released as trusted until encoding is resolved.',
                     'Coverage is highly biased against GT14; do not pool provisional counts as representative performance.'],
                   input_sha256={str(p): hashlib.sha256(p.read_bytes()).hexdigest()
                        for p in (args.review, args.audit / 'metrics.csv', args.audit / 'candidates.csv')})
    (args.output / 'summary.json').write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--audit', type=Path, default=Path('audit/results/gt-audit-20260912T121456077403Z'))
    parser.add_argument('--review', type=Path, default=Path('audit/task2_gt_review.json'))
    parser.add_argument('--output', type=Path, required=True)
    run(parser.parse_args())
