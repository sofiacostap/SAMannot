"""Export only explicitly accepted bird/frame references after complete review."""
import argparse
import csv
import html
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from PIL import Image
from recompute import sha
from review_all_gt import Review, counts
from verify_correspondence import fingerprint


def export(review_dir, reference, source, destination):
    # Reopen from disk: do not reuse cached images from an active review session.
    review = Review(reference, source, review_dir, resume=True)
    tally = counts(review.record['decisions'])
    if tally['pending']:
        raise ValueError('Review incomplete: unreviewed observations remain')
    accepted = tally['accepted_present'] + tally['accepted_absent']
    if not accepted:
        raise ValueError('No accepted observations to export')
    destination = destination.resolve()
    if not destination.is_dir():
        raise ValueError('Destination directory must already exist')
    review_hash = sha(review.output/'review.json')
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    staging = destination/('accepted-gt-'+stamp+'.incomplete')
    final = destination/('accepted-gt-'+stamp)
    staging.mkdir(exist_ok=False)
    for folder in ('photos', 'masks', 'previews'):
        (staging/folder).mkdir()
    rows, cards = [], []
    for index, frame in enumerate(review.frames):
        birds = [b for b in frame['birds'] if review.record['decisions'][
            f"{frame['video_frame']}:{b['id']}"]['status'] in ('accepted_present', 'accepted_absent')]
        if not birds:
            continue
        rgb, gt = review.pixels(index)
        vf = frame['video_frame']
        photo = f'photos/{vf:06d}.png'
        Image.fromarray(rgb).save(staging/photo)
        with Image.open(staging/photo) as im:
            if fingerprint(np.asarray(im.convert('RGB'))) != frame['rgb_sha256']:
                raise ValueError('Exported photograph failed pixel verification')
        for bird in birds:
            bid = bird['id']
            status = review.record['decisions'][f'{vf}:{bid}']['status']
            mask = f'masks/{vf:06d}_id{bid}.png'
            preview = f'previews/{vf:06d}_id{bid}.png'
            binary = (gt == bid).astype(np.uint8)*255
            Image.fromarray(binary).save(staging/mask)
            with Image.open(staging/mask) as im:
                if not np.array_equal(np.asarray(im), binary):
                    raise ValueError('Exported mask failed pixel roundtrip')
            (staging/preview).write_bytes(review.picture(index, 'overlay', bid, False))
            rows.append(dict(video_frame=vf, gt_frame=frame['gt_frame'], gt_id=bid,
                status=status, photo=photo, mask=mask, preview=preview,
                mask_sha256=sha(staging/mask), photo_sha256=sha(staging/photo),
                source_rgb_sha256=frame['rgb_sha256'],
                source_mask_sha256=frame['mask_sha256']))
            cards.append(f'<article><h2>Video {vf} · GT {frame["gt_frame"]} · ID {bid}</h2>'
                f'<p>{html.escape(status)}</p><div><a href="{photo}"><img loading="lazy" src="{photo}" alt="Photograph"></a>'
                f'<a href="{preview}"><img loading="lazy" src="{preview}" alt="Accepted bird overlay"></a></div>'
                f'<a href="{mask}">Full-size binary reference mask</a></article>')
        if (index+1) % 100 == 0:
            print(f'Checked {index+1}/{len(review.frames)} images', flush=True)
    if len(rows) != accepted or len({(r['video_frame'], r['gt_id']) for r in rows}) != accepted:
        raise ValueError('Export count or uniqueness mismatch')
    if sha(review.output/'review.json') != review_hash:
        raise ValueError('Review changed during export; stop the review server first')
    with (staging/'manifest.csv').open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    (staging/'index.html').write_text('<!doctype html><meta charset="utf-8"><title>Accepted GT review</title>'
        '<style>body{font:16px system-ui;margin:24px}article{margin:30px 0;border-top:1px solid #aaa}'
        'article div{display:flex;gap:10px}article div a{width:50%}img{width:100%}</style>'
        '<h1>Visually accepted GT references</h1><p>Only listed bird/frame pairs are eligible. '
        'An omitted bird is unknown here, not background or confirmed absence. '
        'Accepted empty masks explicitly represent accepted_absent. Click a photograph or overlay to enlarge.</p>'
        + ''.join(cards), encoding='utf-8')
    (staging/'README.txt').write_text(
        'This is a visually accepted reference subset, not a guarantee of biological truth.\n'
        'manifest.csv is the eligibility authority. Each mask contains one bird only, encoded as 0/255.\n'
        'Do not combine these masks into a complete scene reference: omitted birds are not validated background.\n'
        'Full photographs may show other birds; only the listed bird reference is accepted.\n'
        'No uncertain, confirmed-problem or unreviewed pair is included.\n'
        'Open index.html for photograph/overlay comparisons. No inference was rerun.\n', encoding='utf-8')
    summary = dict(status='completed_accepted_gt_export', accepted_observations=len(rows),
        accepted_present=tally['accepted_present'], accepted_absent=tally['accepted_absent'],
        omitted_uncertain=tally['uncertain'], omitted_confirmed_problem=tally['confirmed_problem'],
        pending=0, photographs=len({r['video_frame'] for r in rows}), review_sha256=review_hash,
        limitations=['Visual acceptance does not guarantee absolute biological correctness.',
            'Excluded identities must not be treated as background in evaluation.'])
    # Export metadata contains hashes and relative filenames, not machine paths or review notes.
    summary['file_sha256'] = {p.relative_to(staging).as_posix(): sha(p)
        for p in sorted(staging.rglob('*')) if p.is_file()}
    (staging/'verification.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    staging.rename(final)  # A failed export keeps its explicit .incomplete name.
    return final


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--review', type=Path, required=True)
    parser.add_argument('--destination', type=Path, required=True)
    parser.add_argument('--reference', type=Path, default=Path('audit/results/task2-reference-20260915T080605348526Z'))
    parser.add_argument('--source', type=Path, default=Path('audit/results/verified-source-20260911T164528649441Z'))
    args = parser.parse_args()
    print('Completed export:', export(args.review, args.reference, args.source, args.destination))
