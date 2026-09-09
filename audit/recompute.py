"""CPU-only Lek4 recomputation from existing PNGs. Never overwrites a run.

Uses NumPy and Pillow. Existing SAM2 masks remain historical inputs; results
are provisional until identity and source-image correspondence are reviewed.
"""
import argparse
import csv
import hashlib
import itertools
import json
import math
import pickle
import shutil
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw


class SessionObject:
    """Data-only stand-in for SAMannot classes when reading trusted sessions."""
    def __new__(cls, *args, **kwargs):
        return super().__new__(cls)


class SessionReader(pickle.Unpickler):
    def find_class(self, module, name):
        if (module, name) in {('label', 'Label'), ('label', 'Label_Handler'),
                               ('label', 'PPoint'), ('label', 'PBox'),
                               ('annotator', 'PackedMasks')}:
            return SessionObject
        return super().find_class(module, name)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for data in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(data)
    return digest.hexdigest()


def write_json(path, data):
    Path(path).write_text(json.dumps(data, indent=2, allow_nan=False), encoding='utf-8')


def write_csv(path, rows, fields=None):
    rows = list(rows)
    with Path(path).open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields or list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def indexed(directory):
    result = {}
    for p in sorted(Path(directory).iterdir()):
        if p.suffix.lower() not in {'.png', '.jpg', '.jpeg'} or not p.stem.isdigit():
            continue
        key = int(p.stem)
        if key in result:
            raise ValueError(f'Duplicate numeric image index {key}: {p}')
        result[key] = p
    return result


def color(index):
    rgb = [0, 0, 0]
    for bit in range(8):
        for channel in range(3):
            rgb[channel] |= ((index >> channel) & 1) << (7-bit)
        index >>= 3
    return tuple(rgb)


def sam_labels(path, n=4):
    with Image.open(path) as image:
        rgb = np.asarray(image.convert('RGB'))
    result = np.zeros(rgb.shape[:2], np.uint8)
    known = np.all(rgb == 0, axis=2)
    for i in range(1, n+1):
        mask = np.all(rgb == color(i), axis=2)
        result[mask] = i
        known |= mask
    if not known.all():
        raise ValueError(f'Unexpected palette colors in {path}; no interpolation allowed')
    return result


def gt_labels(path):
    with Image.open(path) as image:
        arr = np.asarray(image)
        if arr.ndim == 2:
            # Indexed palette PNG: indices themselves are GT IDs.
            return arr.copy()
        if arr.ndim == 3 and np.array_equal(arr[..., 0], arr[..., 1]) and np.array_equal(arr[..., 0], arr[..., 2]):
            return arr[..., 0].copy()
    raise ValueError(f'GT is not a grayscale/index-label image: {path}')


def overlap(a, b):
    if a.shape != b.shape:
        raise ValueError(f'Mask shape mismatch: {a.shape} versus {b.shape}')
    union = int(np.count_nonzero(a | b))
    return float(np.count_nonzero(a & b) / union) if union else None


def observation(pred, truth, threshold=.5):
    """Absent GT is explicit, not an invented correct tracking observation."""
    pred_area, gt_area = int(pred.sum()), int(truth.sum())
    if gt_area == 0:
        return dict(iou=None, is_error=None, status='gt_absent_prediction_present' if pred_area else 'both_absent',
                    pred_area=pred_area, gt_area=gt_area)
    score = overlap(pred, truth)
    return dict(iou=score, is_error=int(score < threshold),
                status='missing_prediction' if pred_area == 0 else 'evaluated',
                pred_area=pred_area, gt_area=gt_area)


def geometry(mask):
    yy, xx = np.nonzero(mask)
    return len(yy), (float(yy.mean()), float(xx.mean())) if len(yy) else None


def continuity(current, previous):
    area, centroid = geometry(current)
    if area < 50:
        return 0., area, None, None, 'dropout'
    if area > .02 * current.size:
        return 0., area, None, None, 'oversized'
    if previous is None:
        return None, area, None, None, 'no_previous_adjacent_frame'
    old_area, old_center = geometry(previous)
    if old_area < 50:
        return None, area, None, None, 'previous_dropout'
    ratio = area / old_area
    jump = math.dist(centroid, old_center)
    symmetric = max(ratio, 1/ratio)
    score = min(max(0., 1-(symmetric-1)/2), max(0., 1-jump/150))
    reason = 'area_jump' if symmetric > 3 else 'centroid_jump' if jump > 150 else 'ok'
    return score, area, ratio, jump, reason


def load_session(path):
    # Only run on the researcher's own trusted session files.
    with Path(path).open('rb') as stream:
        data = SessionReader(stream).load()
    labels = data['sam_handler_labels']
    names = [label.name for label in labels]
    if len(names) != 4 or len(set(names)) != 4:
        raise ValueError(f'Expected four unique labels, got {names}')
    size, blocks = int(data['block_size']), int(data['num_blocks'])
    if size <= 0 or blocks <= 0:
        raise ValueError('Invalid session block size/count')
    prompts, best = [], {}
    for block in range(blocks):
        indices = set()
        for index, label in enumerate(labels, 1):
            for pt in label.pts.get(block, []):
                local = int(pt.idx)
                if not 0 <= local < size:
                    raise ValueError(f'Prompt outside block {block}: {local}')
                indices.add(local)
                prompts.append(dict(block_id=block, local_frame=local,
                                    abs_frame=block*size+local, label=label.name,
                                    sam2_label=index, x=float(pt.x), y=float(pt.y),
                                    positive=int(pt.pt_type)==1))
            if getattr(label, 'boxes', {}).get(block):
                raise ValueError('Box prompts present; this point-prompt workflow needs explicit adaptation')
        if len(indices) != 1:
            raise ValueError(f'Expected one shared prompt frame in block {block}, got {sorted(indices)}')
        best[block] = block*size + next(iter(indices))
    return dict(names=names, block_size=size, num_blocks=blocks, best=best,
                prompts=prompts, video_name=str(data.get('video_name', '')),
                session_sha256=sha(path))


def calibrate(sam_files, gt_files, session, bird_ids):
    """Propose ONE fixed palette-to-GT map using prompt frames only."""
    values = defaultdict(list)
    evidence = []
    for frame in session['best'].values():
        if frame not in sam_files or frame not in gt_files:
            continue
        sam, gt = sam_labels(sam_files[frame]), gt_labels(gt_files[frame])
        for index in range(1, 5):
            for bird in bird_ids:
                value = overlap(sam == index, gt == bird)
                if value is not None:
                    values[index, bird].append(value)
                    evidence.append(dict(frame=frame, sam2_label=index, gt_bird_id=bird, iou=value))
    if not evidence:
        raise ValueError('No common prompt frames for identity calibration')
    candidates = []
    for permutation in itertools.permutations(bird_ids):
        means = [float(np.mean(values.get((i, bird), [0.]))) for i, bird in enumerate(permutation, 1)]
        candidates.append(dict(mapping=dict(zip(range(1, 5), permutation)), mean_iou=float(np.mean(means)), per_label_iou=means))
    candidates.sort(key=lambda x: x['mean_iou'], reverse=True)
    margin = candidates[0]['mean_iou'] - candidates[1]['mean_iou']
    return candidates, evidence, margin


def precision_recall(rows, score_field, threshold):
    usable = [r for r in rows if r['is_error'] is not None and r[score_field] is not None]
    tp = sum(r[score_field] < threshold and r['is_error'] == 1 for r in usable)
    fp = sum(r[score_field] < threshold and r['is_error'] == 0 for r in usable)
    fn = sum(r[score_field] >= threshold and r['is_error'] == 1 for r in usable)
    tn = len(usable)-tp-fp-fn
    p, recall = tp/(tp+fp) if tp+fp else 0., tp/(tp+fn) if tp+fn else 0.
    return dict(n=len(usable), tp=tp, fp=fp, fn=fn, tn=tn, precision=p, recall=recall,
                f1=2*p*recall/(p+recall) if p+recall else 0.)


def summarize(rows):
    valid = [r for r in rows if r['is_error'] is not None]
    return dict(rows=len(rows), evaluated=len(valid), statuses=dict(Counter(r['status'] for r in rows)),
                mean_iou=float(np.mean([r['iou'] for r in valid])) if valid else None,
                error_rate=float(np.mean([r['is_error'] for r in valid])) if valid else None)


def proof(path, raw_path, sam, gt, index, bird, caption):
    with Image.open(raw_path) as image:
        raw = image.convert('RGB')
    if (raw.height, raw.width) != sam.shape or gt.shape != sam.shape:
        raise ValueError('FRAMES/SAM/GT dimension mismatch; refusing misleading overlay')
    rgb = np.asarray(raw).copy()
    panels = []
    for mask, tint in [(sam == index, [255, 80, 80]), (gt == bird, [40, 230, 80])]:
        panel = rgb.copy()
        panel[mask] = (panel[mask]*.5 + np.asarray(tint)*.5).astype(np.uint8)
        panels.append(Image.fromarray(panel).resize((672, 380)))
    result = Image.new('RGB', (1344, 415), 'white')
    result.paste(panels[0], (0, 35)); result.paste(panels[1], (672, 35))
    ImageDraw.Draw(result).text((8, 8), caption + ' | prediction (left), raw GT (right)', fill='black')
    result.save(path, quality=90)


def temporal(rows, output):
    groups = defaultdict(list)
    for row in rows:
        if row['is_error'] is not None and not row['is_calibration_frame']:
            groups[(row['label'], (row['distance_from_prompt']//50)*50)].append(row)
    bins = [dict(label=name, distance_bin_start=distance, **summarize(items)) for (name, distance), items in sorted(groups.items())]
    for row in bins:
        row.pop('statuses')
    write_csv(output/'temporal_bins.csv', bins, ['label','distance_bin_start','rows','evaluated','mean_iou','error_rate'])
    # Standalone SVG needs no plotting dependency; bins explicitly show sample counts in CSV.
    import html
    parts = ['<svg xmlns="http://www.w3.org/2000/svg" width="1000" height="430"><rect width="100%" height="100%" fill="white"/>',
             '<text x="40" y="25">Provisional fixed-identity IoU by signed distance from prompt (50-frame bins)</text>',
             '<path d="M60 50 V350 H960" fill="none" stroke="black"/>',
             '<text x="360" y="410">Frames from annotation (negative = before)</text>']
    low = min([r['distance_bin_start'] for r in bins] or [0])
    high = max([r['distance_bin_start'] for r in bins] or [1])
    for i, name in enumerate(sorted(set(r['label'] for r in bins))):
        palette = ['#c44','#087','#36c','#a70'][i]
        points = ' '.join(f"{60+900*(r['distance_bin_start']-low)/max(1,high-low):.1f},{350-300*r['mean_iou']:.1f}" for r in bins if r['label']==name)
        parts.append(f'<polyline points="{points}" fill="none" stroke="{palette}" stroke-width="2"/>')
        parts.append(f'<text x="{80+i*220}" y="380" fill="{palette}">{html.escape(name)}</text>')
    for x in sorted(set([low, 0, high])):
        if low <= x <= high:
            parts.append(f'<text x="{60+900*(x-low)/max(1,high-low):.1f}" y="366">{x}</text>')
    parts.extend(['<text x="25" y="55">1.0</text><text x="25" y="350">0.0</text>', '</svg>'])
    (output/'temporal.svg').write_text('\n'.join(parts), encoding='utf-8')


def check_video(video, raw_files, sample_frames, output):
    """Diagnostic evidence only: pixel differences do not establish alignment."""
    if video is None:
        return dict(status='not_checked', reason='No video path provided')
    try:
        import cv2
    except ImportError:
        return dict(status='not_checked', reason='OpenCV unavailable')
    capture = cv2.VideoCapture(str(video))
    if not capture.isOpened():
        raise ValueError(f'Cannot open source video {video}')
    result = dict(status='requires_visual_review', video=str(video),
                  reported_frame_count=int(capture.get(cv2.CAP_PROP_FRAME_COUNT)), samples=[])
    try:
        for frame in sorted(sample_frames):
            if frame not in raw_files:
                continue
            capture.set(cv2.CAP_PROP_POS_FRAMES, frame)
            ok, decoded = capture.read()
            if not ok:
                result['samples'].append(dict(frame=frame, error='video_decode_failed'))
                continue
            decoded = cv2.cvtColor(decoded, cv2.COLOR_BGR2RGB)
            with Image.open(raw_files[frame]) as image:
                raw = np.asarray(image.convert('RGB'))
            sample = dict(frame=frame, video_shape=list(decoded.shape), frames_shape=list(raw.shape))
            if decoded.shape == raw.shape:
                diff = np.abs(decoded.astype(np.int16)-raw.astype(np.int16))
                sample.update(mean_absolute_pixel_difference=float(diff.mean()), max_pixel_difference=int(diff.max()))
            canvas = Image.new('RGB',(1344,415),'white')
            canvas.paste(Image.fromarray(decoded).resize((672,380)),(0,35))
            canvas.paste(Image.fromarray(raw).resize((672,380)),(672,35))
            ImageDraw.Draw(canvas).text((8,8),f'Frame {frame}: video decode (left), FRAMES source (right); alignment NOT certified',fill='black')
            canvas.save(output/f'alignment_{frame:06d}.jpg',quality=92)
            result['samples'].append(sample)
    finally:
        capture.release()
    return result


def run(args):
    output = args.output.resolve()
    workspace = args.workspace.resolve()
    if not output.is_relative_to(workspace):
        raise ValueError('Output must be inside the designated SAMannot workspace')
    output.mkdir(parents=True, exist_ok=False)
    manifest = dict(status='in_progress', limitations=[
        'Historical masks: generating code/current_block provenance not certified.',
        'Equal numeric filenames assumed to identify corresponding FRAMES and video images; visual review required.',
        'Identity mapping calibrated on prompt masks; proposal requires review.',
        'No recoverable raw logits: B and historical three-signal combined scores are not recomputed.',
        'Raw target GT pixels retained; GT absence is not scored as successful tracking.'], inputs={})
    try:
        session = load_session(args.session)
        write_json(output/'session.json', session)
        sam_files, gt_files, raw_files = indexed(args.masks), indexed(args.gt), indexed(args.frames)
        coverage = dict(sam_count=len(sam_files), gt_count=len(gt_files), frames_count=len(raw_files),
                        gt_missing_sam=sorted(set(gt_files)-set(sam_files)),
                        sam_without_gt=sorted(set(sam_files)-set(gt_files)),
                        gt_missing_background=sorted(set(gt_files)-set(raw_files)))
        write_json(output/'coverage.json', coverage)
        for name, path in [('session', args.session), ('legacy_quality', args.quality)]:
            manifest['inputs'][name] = dict(path=str(path.resolve()), sha256=sha(path))
        manifest['inputs'].update(masks=str(args.masks.resolve()), gt=str(args.gt.resolve()), frames=str(args.frames.resolve()))
        with args.quality.open(newline='', encoding='utf-8-sig') as stream:
            legacy = list(csv.DictReader(stream))
        shutil.copyfile(args.quality, output/'legacy_quality.csv')
        legacy_iou_path = getattr(args, 'legacy_iou', None)
        if legacy_iou_path is not None:
            with legacy_iou_path.open(newline='', encoding='utf-8-sig') as stream:
                old_iou = list(csv.DictReader(stream))
            old_keys = Counter((int(r['frame_idx']), int(r['sam2_label'])) for r in old_iou)
            write_json(output/'legacy_iou_audit.json', dict(rows=len(old_iou), unique_keys=len(old_keys),
                mean_iou=float(np.mean([float(r['iou']) for r in old_iou])) if old_iou else None,
                error_rate=float(np.mean([int(r['is_error']) for r in old_iou])) if old_iou else None,
                label_gt_pairs=dict(Counter(f"{r['sam2_label']}->{r['gt_bird_id']}" for r in old_iou))))
            shutil.copyfile(legacy_iou_path, output/'legacy_iou.csv')
            manifest['inputs']['legacy_iou'] = dict(path=str(legacy_iou_path.resolve()), sha256=sha(legacy_iou_path))
        summaries = {}
        for name in ['pipeline_summary.txt','headless_window16_quality_summary.txt']:
            path = args.masks/name
            if path.is_file():
                summaries[name] = path.read_text(encoding='utf-8',errors='replace')
        write_json(output/'historical_summaries.json', summaries)
        keys = Counter((int(r['abs_frame']), r['label']) for r in legacy)
        write_json(output/'legacy_quality_audit.json', dict(rows=len(legacy), unique_keys=len(keys),
            duplicate_rows=sum(n-1 for n in keys.values()), duplicate_keys=[list(k)+[n] for k,n in keys.items() if n>1],
            logit_distribution=dict(Counter(r.get('logit_score','') for r in legacy))))
        candidates, calibration, margin = calibrate(sam_files, gt_files, session, args.gt_ids)
        write_json(output/'video_alignment.json', check_video(getattr(args,'video',None), raw_files,
                   set(session['best'].values()) | {min(gt_files),max(gt_files)}, output))
        write_json(output/'mapping_proposal.json', dict(candidates=candidates, margin=margin,
            session_palette_order=session['names'], supplied_name_to_gt={'darkest':14,'brownish':38,'orange':75,'spotted':113}))
        write_csv(output/'calibration_pairs.csv', calibration)
        mapping = candidates[0]['mapping']
        manifest['mapping'] = mapping
        manifest['mapping_margin'] = margin
        for frame in session['best'].values():
            if frame in sam_files and frame in gt_files and frame in raw_files:
                sam, gt = sam_labels(sam_files[frame]), gt_labels(gt_files[frame])
                for index, name in enumerate(session['names'], 1):
                    proof(output/f'calibration_{frame:06d}_label{index}.jpg', raw_files[frame],
                          sam, gt, index, mapping[index], f'IDENTITY PROPOSAL frame {frame}, {name}, GT {mapping[index]}')
        # Ambiguous mapping still yields evidence, but no misleading chosen-identity metrics.
        if margin < .05 or min(candidates[0]['per_label_iou']) < .2:
            manifest['status'] = 'needs_identity_review'
            return
        if coverage['gt_missing_sam']:
            manifest['status'] = 'missing_mask_files'
            return
        refs = {}
        for block, frame in session['best'].items():
            if frame not in sam_files:
                raise ValueError(f'Missing annotation reference mask {frame}')
            refs[block] = sam_labels(sam_files[frame])
        records, previous, previous_frame = [], None, None
        input_hash = hashlib.sha256()
        sample_frames = set(session['best'].values()) | {min(gt_files), max(gt_files)}
        proof_count = 0
        all_frames = sorted(set(sam_files) | set(gt_files))
        for position, frame in enumerate(all_frames):
            if frame not in sam_files:
                continue
            sam = sam_labels(sam_files[frame])
            gt = gt_labels(gt_files[frame]) if frame in gt_files else None
            input_hash.update(f'{frame}:sam:{sha(sam_files[frame])}\n'.encode())
            if gt is not None:
                if sam.shape != gt.shape:
                    raise ValueError(f'Frame {frame}: unequal mask dimensions; no automatic resizing')
                input_hash.update(f'{frame}:gt:{sha(gt_files[frame])}\n'.encode())
                unexpected = set(int(v) for v in np.unique(gt))-set(args.gt_ids)-{0}
            else:
                unexpected = set()
            block = frame//session['block_size']
            if block not in refs:
                raise ValueError(f'Frame {frame} outside session blocks')
            prompt = session['best'][block]
            adjacent = previous_frame == frame-1 and previous_frame//session['block_size']==block
            for index, name in enumerate(session['names'], 1):
                mask = sam == index
                a, area, ratio, jump, reason = continuity(mask, previous == index if adjacent else None)
                ref = refs[block] == index
                c = overlap(mask, ref) if ref.any() else None
                # Equal-weight AC is a newly defined score, explicitly not the historical ABC score.
                ac = .5*a+.5*c if a is not None and c is not None else None
                truth = observation(mask, gt == mapping[index]) if gt is not None else dict(iou=None,is_error=None,status='gt_file_missing',pred_area=area,gt_area=None)
                row = dict(frame_idx=frame, block_id=block, local_frame=frame%session['block_size'],
                    label=name, sam2_label=index, gt_bird_id=mapping[index], annotation_frame=prompt,
                    distance_from_prompt=frame-prompt, is_calibration_frame=int(frame in session['best'].values()),
                    **truth, continuity_score=a, ref_iou_score=c, ac_score=ac,
                    area_ratio=ratio, centroid_jump=jump, reason=reason,
                    gt_has_other_ids=int(bool(unexpected)), gt_tiny_target=int(0 < (truth['gt_area'] or 0) < 50))
                records.append(row)
                if frame in sample_frames and frame in raw_files and proof_count < 32:
                    proof(output/f'proof_{frame:06d}_label{index}.jpg',raw_files[frame],sam,gt if gt is not None else np.zeros_like(sam),index,mapping[index],
                          f'PROVISIONAL frame {frame}, {name}, GT {mapping[index]}, IoU {truth["iou"]}')
                    proof_count += 1
            previous, previous_frame = sam, frame
            if position % 250 == 0:
                print(f'Processed {position}/{len(all_frames)} image indices', flush=True)
        write_csv(output/'recomputed.csv', records)
        evaluation = [r for r in records if not r['is_calibration_frame']]
        summary = dict(all_frames=summarize(records), excluding_identity_calibration_frames=summarize(evaluation),
                       by_bird={name:summarize([r for r in evaluation if r['label']==name]) for name in session['names']})
        write_json(output/'summary.json', summary)
        # AC-only search. One common available-score population for all weights.
        population = [r for r in evaluation if r['is_error'] is not None and r['continuity_score'] is not None and r['ref_iou_score'] is not None]
        grid = []
        for wa in range(11):
            weighted = [dict(r, search_score=(wa/10)*r['continuity_score']+(1-wa/10)*r['ref_iou_score']) for r in population]
            for t in range(2, 20):
                grid.append(dict(w_cont=wa/10,w_ref=1-wa/10,threshold=t*.05,
                                 **precision_recall(weighted,'search_score',t*.05)))
        write_csv(output/'ac_grid_exploratory.csv', grid)
        write_json(output/'scorer_population.json', dict(eligible_noncalibration=len([r for r in evaluation if r['is_error'] is not None]),
            common_ac_population=len(population), note='Exploratory in-sample calibration, not held-out performance. B omitted because raw logits are unavailable.'))
        temporal(records, output)
        manifest.update(status='completed_provisional', mask_gt_content_digest=input_hash.hexdigest(),
                        score_definition='Chronological A reset at block/gaps; spatial reference C; AC equal weights; no B',
                        source_sha256=sha(__file__))
    except Exception as exc:
        manifest.update(status='failed', error=f'{type(exc).__name__}: {exc}')
        raise
    finally:
        write_json(output/'manifest.json', manifest)
        print(f'Result status: {manifest["status"]}; files: {output}', flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ['workspace','session','masks','gt','frames','quality','output']:
        p.add_argument('--'+name, type=Path, required=True)
    p.add_argument('--gt-ids', type=int, nargs=4, default=[14,38,75,113])
    p.add_argument('--video', type=Path, help='Optional source video for alignment diagnostics')
    p.add_argument('--legacy-iou', type=Path, help='Historical IoU CSV to audit and preserve for comparison')
    args = p.parse_args()
    if len(set(args.gt_ids)) != 4:
        p.error('GT IDs must be unique')
    run(args)


if __name__ == '__main__':
    main()
