"""Build separate categorical GT derivatives and a conservative evaluation set.

No prediction input, SAM2, biological edits, or source overwrites. Source/identity
uncertainty is excluded locally. Historical grayscale conversion is not repaired.
"""
import argparse
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from PIL import Image

from build_task2_set import partition, read_csv
from gt_audit import component_flags, pair_swap_candidates
from recompute import indexed, gt_labels, sha, write_csv, write_json
from verify_correspondence import fingerprint

IDS = (14, 38, 75, 113)
PALETTE = {0: (0, 0, 0), 1: (128, 0, 0), 2: (0, 128, 0),
           3: (128, 128, 0), 4: (0, 0, 128)}
# Explicit identity mapping from the inspected palette, never nearest-value rounding.
LABEL_MAP = np.array([0, 38, 75, 113, 14], dtype=np.uint8)


def normalize(path, target_shape):
    with Image.open(path) as im:
        if im.mode != 'P':
            raise ValueError(f'Expected categorical palette PNG, found {im.mode}')
        raw = np.array(im)
        values = set(map(int, np.unique(raw)))
        if not values <= set(PALETTE):
            raise ValueError(f'Unknown source IDs: {sorted(values)}')
        palette = im.getpalette()
        if palette is None or any(tuple(palette[3*i:3*i+3]) != PALETTE[i] for i in values):
            raise ValueError('Palette differs from inspected source mapping')
        if tuple(target_shape) != (raw.shape[0]*2, raw.shape[1]*2):
            raise ValueError('Expected exact 2x enlargement in both dimensions')
        return np.repeat(np.repeat(LABEL_MAP[raw], 2, axis=0), 2, axis=1)


def compare_core(core, old, bird):
    """Agreement at new-mask interior pixels with known historical labels.

This is a correspondence screen, not a proof of original-dataset provenance.
"""
    known = np.isin(old, (0,) + IDS)
    support = int(np.count_nonzero(core))
    comparable = int(np.count_nonzero(core & known))
    disagreed = int(np.count_nonzero(core & known & (old != bird)))
    enough = support == 0 or comparable >= .95 * support
    acceptable = disagreed <= max(4, .01 * comparable)
    return support, comparable, disagreed, enough and acceptable


def run(args):
    import cv2
    source = args.source.resolve()
    audit = args.audit.resolve()
    original = args.original.resolve()
    verified = json.loads((source / 'verification.json').read_text())
    if verified['status'] != 'verified':
        raise ValueError('Verified image source required')
    all_mapping = read_csv(source / 'index.csv')
    mapping = {int(r['gt_frames_index']): r for r in all_mapping if r['gt_frames_index']}
    if len(mapping) != sum(bool(r['gt_frames_index']) for r in all_mapping):
        raise ValueError('Duplicate GT correspondence')
    old_report = json.loads((audit / 'verification.json').read_text())
    old_hashes = json.loads((audit / 'gt_hashes.json').read_text())
    if set(mapping) != set(map(int, old_hashes)):
        raise ValueError('Image correspondence and audited GT inventory differ')
    if sha(source / 'verification.json') != old_report['source_verification_sha256']:
        raise ValueError('Verified source manifest changed since GT audit')
    old_files = indexed(Path(old_report['gt_directory']))
    inputs = indexed(original)
    ledger = json.loads(args.review.read_text())
    output = Path(__file__).resolve().parent / 'results' / ('task2-reference-' + datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ'))
    output.mkdir()
    masks = output / 'masks'
    masks.mkdir()
    write_json(output / 'review_decisions.json', ledger)
    report = dict(status='in_progress', original_directory=str(original), output_directory=str(output),
                  source_directory=str(source), normalization='Native palette IDs -> explicit label map -> exact 2x nearest-neighbor repeat',
                  label_map={str(i): int(v) for i, v in enumerate(LABEL_MAP)},
                  extra_source_indices=sorted(set(inputs)-set(mapping)),
                  input_sha256={str(p): sha(p) for p in (Path(__file__), args.review, source/'index.csv',
                      source/'verification.json', audit/'gt_hashes.json')},
                  screening=dict(core_erosion_radius_pixels=2, min_core_known_fraction=.95,
                      max_core_disagreement_fraction=.01, max_core_disagreement_floor_pixels=4,
                      component_policy='Hold all disconnected/tiny candidates; preserve their pixels',
                      continuity_policy='Hold both frames of abrupt changes or potential exchanges'),
                  limitations=[
                      'No original source checksum is available; original transfer integrity is not established.',
                      'Half-resolution mask file indices inherit the verified FRAMES mapping, with historical-mask interior agreement used as a consistency screen.',
                      'Interior agreement is not proof of frame timing; small movements and boundary differences can pass.',
                      'Trusted means passed the declared conservative screening, not exhaustive human certification.',
                      'Nearest-neighbor preserves half-resolution annotation geometry; it adds no new boundary detail.',
                      'Historical grayscale conversion remains unidentified; current grayscale masks are not used as final truth.',
                      'GT-assisted annotation limits claims about independent novice annotation performance.'])
    metrics, candidates, correspondence, hashes = [], [], [], {}
    previous, previous_stats, previous_frame = None, {}, None
    kernel = np.ones((5, 5), np.uint8)
    def flag(frame, gf, bird, reason, detail=''):
        candidates.append(dict(video_frame=frame, gt_frame=gf, gt_id=str(bird), reason=reason, detail=detail))
    try:
        for n, (gf, entry) in enumerate(sorted(mapping.items()), 1):
            frame = int(entry['video_index'])
            try:
                if gf not in inputs:
                    raise ValueError('Source categorical mask missing')
                with Image.open(source / 'frames' / entry['file']) as im:
                    shape = (im.height, im.width)
                    if fingerprint(np.asarray(im.convert('RGB'))) != entry['rgb_sha256']:
                        raise ValueError('Verified photograph pixels changed')
                src_hash = sha(inputs[gf])
                gt = normalize(inputs[gf], shape)
                if sha(inputs[gf]) != src_hash:
                    raise ValueError('Source changed during normalization')
            except (ValueError, OSError) as exc:
                for bird in IDS:
                    metrics.append(dict(video_frame=frame, gt_frame=gf, gt_id=bird, area=0, components=0,
                        largest_component=0, touches_edge=False, centroid_x=None, centroid_y=None))
                    flag(frame, gf, bird, 'source_unusable', str(exc))
                previous, previous_stats, previous_frame = None, {}, None
                continue
            dest = masks / f'{gf:06d}.png'
            Image.fromarray(gt).save(dest)
            if not np.array_equal(gt_labels(dest), gt):
                raise ValueError('Normalized mask roundtrip changed labels')
            hashes[str(gf)] = dict(original=src_hash, normalized=sha(dest))
            # A missing/changed historical reference holds this frame, not the video.
            old = None
            if gf in old_files and sha(old_files[gf]) == old_hashes[str(gf)]:
                old = gt_labels(old_files[gf])
                if old.shape != gt.shape:
                    old = None
            stats = {}
            for bird in IDS:
                binary = (gt == bird).astype(np.uint8)
                area = int(binary.sum())
                count, _, components, centres = cv2.connectedComponentsWithStats(binary, connectivity=8)
                sizes = components[1:, cv2.CC_STAT_AREA].tolist()
                xy = np.average(centres[1:], axis=0, weights=sizes) if area else None
                stats[bird] = dict(area=area, x=float(xy[0]) if area else None, y=float(xy[1]) if area else None)
                metrics.append(dict(video_frame=frame, gt_frame=gf, gt_id=bird, area=area,
                    components=count-1, largest_component=max(sizes, default=0),
                    touches_edge=bool(binary[0].any() or binary[-1].any() or binary[:,0].any() or binary[:,-1].any()),
                    centroid_x=stats[bird]['x'], centroid_y=stats[bird]['y']))
                for reason in component_flags(area, sizes):
                    flag(frame, gf, bird, reason)
                if old is None:
                    flag(frame, gf, bird, 'historical_reference_unavailable_or_changed')
                    continue
                core = cv2.erode(binary, kernel, borderType=cv2.BORDER_CONSTANT, borderValue=0).astype(bool)
                support, comparable, wrong, ok = compare_core(core, old, bird)
                # Also check the opposite direction to catch a disappeared historical bird.
                old_core = cv2.erode((old == bird).astype(np.uint8), kernel,
                                     borderType=cv2.BORDER_CONSTANT, borderValue=0).astype(bool)
                osupport, _, owrong, reverse_ok = compare_core(old_core, gt, bird)
                correspondence.append(dict(video_frame=frame, gt_frame=gf, gt_id=bird,
                    new_core_pixels=support, comparable_old_pixels=comparable, new_core_disagreements=wrong,
                    old_core_pixels=osupport, old_core_disagreements=owrong, passed=ok and reverse_ok))
                if not (ok and reverse_ok):
                    flag(frame, gf, bird, 'source_correspondence_uncertain')
                if previous_frame == frame-1:
                    last = previous_stats[bird]
                    if min(last['area'], area) >= 50:
                        if float(np.hypot(stats[bird]['x']-last['x'], stats[bird]['y']-last['y'])) > 150:
                            flag(frame, gf, bird, 'centroid_jump')
                        if max(area, last['area']) / min(area, last['area']) > 3:
                            flag(frame, gf, bird, 'area_change')
            if previous_frame == frame-1:
                for a, b in pair_swap_candidates(previous, gt):
                    flag(frame, gf, f'{a}+{b}', 'possible_identity_exchange')
            previous, previous_stats, previous_frame = gt, stats, frame
            if n % 250 == 0:
                print(f'Categorical GT checked {n}/{len(mapping)}', flush=True)
        rows, held = partition(metrics, candidates, ledger)
        for row in rows:
            if row['status'] == 'screened_pending_encoding':
                row['status'] = 'trusted_screened'
            if 'source_unusable' in row['reasons']:
                row['gt_present'], row['gt_area'] = None, None
            row['normalized_mask'] = f'masks/{row["gt_frame"]:06d}.png' if str(row['gt_frame']) in hashes else ''
        if len(rows) != len(mapping)*4:
            raise ValueError('Incomplete bird/frame partition')
        write_csv(output/'evaluation_set.csv', rows)
        report.update(status='completed_reference_partition', gt_images=len(mapping),
            normalized_images=len(hashes), gt_bird_frames=len(rows),
            counts=dict(Counter(r['status'] for r in rows)), review_held_bird_frames=len(held),
            by_gt_id={str(g): dict(Counter(r['status'] for r in rows if r['gt_id']==g)) for g in IDS},
            by_block_and_gt={f'{b}/{g}': dict(Counter(r['status'] for r in rows if r['block_id']==b and r['gt_id']==g))
                             for b in sorted({r['block_id'] for r in rows}) for g in IDS},
            trusted_present_absent=dict(Counter('present' if r['gt_present'] else 'absent'
                         for r in rows if r['status']=='trusted_screened')),
            candidate_counts=dict(Counter(r['reason'] for r in candidates)),
            task2_closeout='Assess coverage from this report; no new manual review queue. All uncertain rows are held out.',
            missing_gt_video_frames=[int(r['video_index']) for r in all_mapping if not r['gt_frames_index']])
    except Exception as exc:
        report.update(status='failed', error=str(exc))
        raise
    finally:
        write_csv(output/'metrics.csv', metrics, ['video_frame','gt_frame','gt_id','area','components','largest_component','touches_edge','centroid_x','centroid_y'])
        write_csv(output/'candidates.csv', candidates, ['video_frame','gt_frame','gt_id','reason','detail'])
        write_csv(output/'correspondence.csv', correspondence, ['video_frame','gt_frame','gt_id','new_core_pixels','comparable_old_pixels','new_core_disagreements','old_core_pixels','old_core_disagreements','passed'])
        write_json(output/'mask_hashes.json', hashes)
        write_json(output/'verification.json', report)
        print(f'Status: {report["status"]}; results: {output}', flush=True)
        if 'counts' in report:
            print(json.dumps(dict(counts=report['counts'], by_gt_id=report['by_gt_id']), indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--original', required=True, type=Path)
    parser.add_argument('--source', type=Path, default=Path('audit/results/verified-source-20260911T164528649441Z'))
    parser.add_argument('--audit', type=Path, default=Path('audit/results/gt-audit-20260912T121456077403Z'))
    parser.add_argument('--review', type=Path, default=Path('audit/task2_gt_review.json'))
    run(parser.parse_args())
