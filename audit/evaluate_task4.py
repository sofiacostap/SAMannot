"""Task 4: frozen GT-free scores evaluated against the released Task 2 reference.

CPU-only; no predictor imports, new prompts, fitted weights, or GT-derived scores.
Two phases: prediction-only signals first, then reference-labelled evaluation.
"""
import argparse
import csv
import hashlib
import json
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from PIL import Image

IDS = (14, 38, 75, 113)
SIGNALS = ('A', 'B', 'C', 'combined', 'empty_baseline')
THRESHOLDS = dict(A=.35, B=.95, C=.1, combined=.35, empty_baseline=.5)


def sha(path, canonical=False):
    data = path.read_bytes()
    return hashlib.sha256(data.replace(b'\r\n', b'\n') if canonical else data).hexdigest()


def read_csv(path):
    with path.open(newline='', encoding='utf-8') as stream:
        return list(csv.DictReader(stream))


def write_csv(path, rows, fields=None):
    with path.open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields or list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False), encoding='utf-8')


def unique_rows(rows, names):
    result = {}
    for row in rows:
        key = tuple(int(row[n]) for n in names)
        if key in result:
            raise ValueError(f'Duplicate key {key}')
        result[key] = row
    return result


def mask_stats(mask):
    area = int(np.count_nonzero(mask))
    if area < 50:
        return area, None
    ys, xs = np.nonzero(mask)
    return area, (float(ys.mean()), float(xs.mean()))


def continuity(current, previous, pixels):
    """Same A area/jump rule as mask_quality_scorer, explicit pass-local state."""
    area, centroid = current
    if area < 50:
        return 0., 'dropout', None, None
    if area > .02*pixels:
        return 0., 'oversized', None, None
    if previous is None or previous[0] < 50 or previous[1] is None:
        return 1., 'no_prev', None, None
    ratio = area / (previous[0]+1e-6)
    symmetric = max(ratio, 1/ratio)
    jump = float(np.hypot(centroid[0]-previous[1][0], centroid[1]-previous[1][1]))
    score = min(float(np.clip(1-(symmetric-1)/2, 0, 1)), float(np.clip(1-jump/150, 0, 1)))
    reason = 'area_jump' if symmetric > 3 else 'centroid_jump' if jump > 150 else 'ok'
    return score, reason, ratio, jump


def reference_score(mask, anchor):
    if np.count_nonzero(mask) < 50 or np.count_nonzero(anchor) < 50:
        return None  # Legacy C substitutes 1.0 in this case; retain separately.
    return float(np.count_nonzero(mask & anchor) / np.count_nonzero(mask | anchor))


def combine(a, b, c):
    legacy_c = 1. if c is None else c
    return .5*a+.5*legacy_c if b is None else .3*a+.4*b+.3*legacy_c


def binary_mask(path, shape=None):
    with Image.open(path) as im:
        arr = np.array(im)
    if arr.ndim != 2 or not np.isin(arr, (0,255)).all():
        raise ValueError(f'Expected independent binary PNG: {path}')
    if shape is not None and arr.shape != shape:
        raise ValueError(f'Mask shape mismatch: {path}')
    return arr > 0


def passes(first, last, prompt):
    if not first <= prompt <= last:
        raise ValueError('Prompt outside block')
    return [('later', list(range(prompt,last+1))), ('earlier', list(range(prompt-1,first-1,-1)))]


def validate_metadata(prediction, reference, closeout_path):
    release = json.loads(closeout_path.read_text())
    if release['task2_status'] != 'COMPLETE_WITH_DOCUMENTED_EXCLUSIONS' or reference.name != release['reference_run']:
        raise ValueError('Use the closed Task 2 reference')
    for filename, key in [('verification.json','verification_sha256'), ('evaluation_set.csv','evaluation_set_sha256')]:
        if sha(reference/filename, canonical=True) != release[key]:
            raise ValueError(f'Released reference changed: {filename}')
    pv = json.loads((prediction/'verification.json').read_text())
    rv = json.loads((reference/'verification.json').read_text())
    annotations = json.loads((prediction/'annotations.json').read_text())
    if pv['status'] != 'completed' or rv['status'] != 'completed_reference_partition':
        raise ValueError('Completed input runs required')
    if pv['source_directory'] != rv['source_directory'] or annotations['source_directory'] != pv['source_directory']:
        raise ValueError('Prediction/reference image sources differ')
    if pv['palette_to_gt'] != {str(i+1): g for i,g in enumerate(IDS)}:
        raise ValueError('Prediction identity mapping differs')
    total, block_size = annotations['total_frames'], annotations['block_size']
    if pv['frames_written'] != total:
        raise ValueError('Incomplete propagation')
    if set(annotations['blocks']) != {str(b) for b in range((total+block_size-1)//block_size)}:
        raise ValueError('Incomplete annotation block inventory')
    confidence = unique_rows(read_csv(prediction/'foreground_confidence.csv'), ('abs_frame','gt_id'))
    expected = {(f,g) for f in range(total) for g in IDS}
    if set(confidence) != expected:
        raise ValueError('Confidence table incomplete or extra')
    reference_rows = unique_rows(read_csv(reference/'evaluation_set.csv'), ('video_frame','gt_id'))
    missing = set(rv['missing_gt_video_frames'])
    if set(reference_rows) != {(f,g) for f,g in expected if f not in missing}:
        raise ValueError('Reference coverage differs from released mapping')
    for (frame,bird), row in reference_rows.items():
        gf = int(row['gt_frame'])
        if gf != (frame if frame < 3 else frame-12) or row['normalized_mask'] != f'masks/{gf:06d}.png':
            raise ValueError('Invalid GT/video frame correspondence or mask path')
    return pv, rv, annotations, confidence, reference_rows


def score_predictions(prediction, annotations, confidence, output):
    """No GT masks or review decisions enter this function."""
    total, block_size = annotations['total_frames'], annotations['block_size']
    rows, hashes, seen = [], [], set()
    def load(frame, bird, shape=None):
        path = prediction/'masks_by_gt_id'/str(bird)/f'{frame:06d}.png'
        return binary_mask(path, shape), sha(path)
    for block in sorted(map(int, annotations['blocks'])):
        prompt = int(annotations['blocks'][str(block)]['video_frame'])
        first, last = block*block_size, min(total-1,(block+1)*block_size-1)
        anchors = {g: load(prompt,g)[0] for g in IDS}
        shape = anchors[IDS[0]].shape
        if any(a.shape != shape for a in anchors.values()):
            raise ValueError('Anchor shapes differ')
        for direction, frames in passes(first,last,prompt):
            previous = {g: mask_stats(anchors[g]) for g in IDS} if direction == 'earlier' else {g: None for g in IDS}
            previous_frame = prompt if direction == 'earlier' else None
            for frame in frames:
                for bird in IDS:
                    key = frame, bird
                    if key in seen:
                        raise ValueError('Scoring pass produced duplicate bird/frame')
                    seen.add(key)
                    mask, digest = load(frame,bird,shape)
                    record = confidence[key]
                    if int(record['block_id']) != block or record['direction'] != direction or int(record['sam2_label']) != IDS.index(bird)+1:
                        raise ValueError(f'Confidence direction/block/identity mismatch: {key}')
                    stats = mask_stats(mask)
                    b = float(record['foreground_logit_score']) if record['foreground_logit_score'] else None
                    if (stats[0] == 0) != (b is None) or (b is not None and not .5 <= b <= 1):
                        raise ValueError(f'Confidence/foreground mismatch: {key}')
                    a, reason, ratio, jump = continuity(stats,previous[bird],mask.size)
                    c = reference_score(mask,anchors[bird])
                    rows.append(dict(video_frame=frame,gt_id=bird,block_id=block,direction=direction,
                        prompt_frame=prompt,distance_from_prompt=abs(frame-prompt),is_prompt=frame==prompt,
                        continuity_previous_frame=previous_frame,pred_area=stats[0],A=a,B=b,C=c,
                        C_legacy=1. if c is None else c,combined=combine(a,b,c),
                        empty_baseline=0. if stats[0]==0 else 1.,A_reason=reason,area_ratio=ratio,centroid_jump=jump,
                        prediction_sha256=digest))
                    hashes.append(dict(video_frame=frame,gt_id=bird,sha256=digest))
                    previous[bird] = stats
                previous_frame = frame
                if len(seen) % 1000 == 0:
                    print(f'Prediction-only scores: {len(seen)}/{total*4}',flush=True)
    if seen != set(confidence):
        raise ValueError('Scoring coverage mismatch')
    rows.sort(key=lambda r:(r['video_frame'],r['gt_id']))
    write_csv(output/'signals.csv',rows)
    write_csv(output/'prediction_hashes.csv',hashes)
    return rows


def label_outcome(mask, truth):
    pa, ga = int(np.count_nonzero(mask)), int(np.count_nonzero(truth))
    inter = int(np.count_nonzero(mask & truth))
    union = pa+ga-inter
    iou = inter/union if union else None
    if ga:
        outcome = 'missing_visible_bird' if not pa else 'segmentation_error' if iou < .5 else 'acceptable_segmentation'
    else:
        outcome = 'false_positive_absent_bird' if pa else 'expected_empty'
    return dict(gt_area=ga,gt_present=ga>0,intersection=inter,union=union,iou=iou,
                error50=(iou < .5) if ga else pa>0,error75=(iou < .75) if ga else pa>0,
                outcome=outcome,absent_substantial_false_positive=ga==0 and pa>=50)


def join_reference(prediction, reference, signals, reference_rows, output):
    hashes = json.loads((reference/'mask_hashes.json').read_text())
    rows = []
    last_gf, gt = None, None
    for signal in signals:
        frame, bird = signal['video_frame'], signal['gt_id']
        decision = reference_rows.get((frame,bird))
        row = dict(signal,reference_status=decision['status'] if decision else 'no_gt',
            reference_reasons=decision['reasons'] if decision else 'no_matching_GT',
            gt_frame=int(decision['gt_frame']) if decision else None,primary_eligible=False,
            gt_area=None,gt_present=None,intersection=None,union=None,iou=None,error50=None,error75=None,
            outcome='not_evaluated',absent_substantial_false_positive=None)
        if decision and decision['status']=='trusted_screened':
            gf = int(decision['gt_frame'])
            if gf != last_gf:
                path = reference/decision['normalized_mask']
                if sha(path) != hashes[str(gf)]['normalized']:
                    raise ValueError(f'Released normalized GT changed: {gf}')
                with Image.open(path) as im:
                    gt = np.array(im)
                if gt.ndim != 2 or not np.isin(gt,(0,)+IDS).all():
                    raise ValueError('Unexpected released GT encoding')
                last_gf = gf
            path = prediction/'masks_by_gt_id'/str(bird)/f'{frame:06d}.png'
            if sha(path) != signal['prediction_sha256']:
                raise ValueError('Prediction changed between scoring and evaluation')
            mask = binary_mask(path, gt.shape)
            if int(np.count_nonzero(mask)) != signal['pred_area']:
                raise ValueError('Prediction area mismatch')
            truth = gt==bird
            if int(np.count_nonzero(truth)) != int(decision['gt_area']):
                raise ValueError('Reference decision area mismatch')
            row.update(label_outcome(mask,truth),primary_eligible=not signal['is_prompt'])
        rows.append(row)
        if len(rows) % 1000 == 0:
            print(f'Reference evaluation: {len(rows)}/{len(signals)}',flush=True)
    write_csv(output/'evaluation.csv',rows)
    return rows


def average_ranks(values):
    values = np.asarray(values)
    order = np.argsort(values,kind='stable')
    sorted_values = values[order]
    starts = np.r_[0,np.flatnonzero(sorted_values[1:] != sorted_values[:-1])+1]
    ends = np.r_[starts[1:],len(values)]
    ranks = np.empty(len(values),float)
    for start,end in zip(starts,ends):
        ranks[order[start:end]] = (start+1+end)/2
    return ranks


def ranking_metrics(labels, risk):
    y, score = np.asarray(labels,dtype=int), np.asarray(risk,dtype=float)
    positives, negatives = int(y.sum()), int(len(y)-y.sum())
    if not positives or not negatives:
        return dict(roc_auc=None,average_precision=None)
    ranks = average_ranks(score)
    auc = (float(ranks[y==1].sum())-positives*(positives+1)/2)/(positives*negatives)
    order = np.argsort(-score,kind='stable')
    ys, scores = y[order], score[order]
    ends = np.r_[np.flatnonzero(scores[1:] != scores[:-1])+1,len(scores)]
    tp = np.cumsum(ys)[ends-1]
    ap = float(np.sum(np.diff(np.r_[0,tp])/positives*(tp/ends)))
    return dict(roc_auc=auc,average_precision=ap)


def confusion(labels, flagged):
    y, f = np.asarray(labels,bool), np.asarray(flagged,bool)
    tp, fp = int(np.count_nonzero(y & f)), int(np.count_nonzero(~y & f))
    fn, tn = int(np.count_nonzero(y & ~f)), int(np.count_nonzero(~y & ~f))
    return dict(tp=tp,fp=fp,fn=fn,tn=tn,precision=tp/(tp+fp) if tp+fp else None,
                recall=tp/(tp+fn) if tp+fn else None,f1=2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else None,
                flag_rate=(tp+fp)/len(y) if len(y) else None)


def spearman_quality(rows, signal):
    pairs = [(r[signal],r['iou']) for r in rows if r['gt_present'] and r['iou'] is not None]
    if len(pairs)<2:
        return None
    x,y = map(average_ranks,zip(*pairs))
    if np.std(x)==0 or np.std(y)==0:
        return None
    return float(np.corrcoef(x,y)[0,1])


def summarize(rows,output):
    primary = [r for r in rows if r['primary_eligible']]
    cohorts = dict(all_presence_aware=primary,
        gt_present=[r for r in primary if r['gt_present']],
        gt_absent=[r for r in primary if not r['gt_present']],
        common_available=[r for r in primary if all(r[s] is not None for s in ('A','B','C','combined'))])
    metrics, sweeps = [], []
    for cohort, base in cohorts.items():
        for signal in SIGNALS:
            available = [r for r in base if r[signal] is not None]
            for error in ('error50','error75'):
                labels = [r[error] for r in available]
                scores = [r[signal] for r in available]
                m = dict(cohort=cohort,error_definition=error,signal=signal,cohort_n=len(base),n=len(available),
                    missing=len(base)-len(available),errors=sum(labels),prevalence=sum(labels)/len(labels) if labels else None,
                    quality_flag_threshold=THRESHOLDS[signal],
                    spearman_quality_iou=spearman_quality(available,signal))
                m.update(ranking_metrics(labels,[1-s for s in scores]),
                         **confusion(labels,[s<THRESHOLDS[signal] for s in scores]))
                metrics.append(m)
                if error=='error50' and cohort in ('all_presence_aware','common_available'):
                    for threshold in sorted(set([0.,.1,.25,.35,.5,.75,.9,.95,.975,.99,1.,THRESHOLDS[signal]])):
                        sweeps.append(dict(cohort=cohort,signal=signal,quality_threshold=threshold,n=len(scores),
                                           **confusion(labels,[s<threshold for s in scores])))
    groups=[]
    grouped=defaultdict(list)
    for r in primary:
        grouped[(r['block_id'],r['gt_id'])].append(r)
    for (block,bird), subset in sorted(grouped.items()):
        present=[r for r in subset if r['gt_present']]
        groups.append(dict(block_id=block,gt_id=bird,n=len(subset),gt_present=len(present),
            mean_iou_gt_present=float(np.mean([r['iou'] for r in present])) if present else None,
            error50=sum(r['error50'] for r in subset),error75=sum(r['error75'] for r in subset),
            expected_empty=sum(r['outcome']=='expected_empty' for r in subset),
            missing_visible_bird=sum(r['outcome']=='missing_visible_bird' for r in subset),
            false_positive_absent_bird=sum(r['outcome']=='false_positive_absent_bird' for r in subset)))
    write_csv(output/'signal_metrics.csv',metrics)
    write_csv(output/'threshold_sweep.csv',sweeps)
    write_csv(output/'by_block_bird.csv',groups)
    alerts=[]
    for outcome in sorted({r['outcome'] for r in primary}):
        subset=[r for r in primary if r['outcome']==outcome]
        for signal in SIGNALS:
            available=[r for r in subset if r[signal] is not None]
            alerts.append(dict(outcome=outcome,signal=signal,n=len(subset),available=len(available),
                missing=len(subset)-len(available),flags=sum(r[signal]<THRESHOLDS[signal] for r in available)))
    write_csv(output/'outcome_alerts.csv',alerts,['outcome','signal','n','available','missing','flags'])
    present=[r for r in primary if r['gt_present']]
    high_confidence=[r for r in primary if r['B'] is not None and r['B']>=.95]
    return dict(primary_observations=len(primary),outcomes=dict(Counter(r['outcome'] for r in primary)),
        mean_iou_gt_present=float(np.mean([r['iou'] for r in present])) if present else None,
        median_iou_gt_present=float(np.median([r['iou'] for r in present])) if present else None,
        primary_error50=sum(r['error50'] for r in primary),primary_error75=sum(r['error75'] for r in primary),
        high_B_observations=len(high_confidence),high_B_error50=sum(r['error50'] for r in high_confidence),
        primary_gt_present=sum(r['gt_present'] for r in primary),
        primary_gt_absent=sum(not r['gt_present'] for r in primary),
        annotation_observations_excluded=sum(r['is_prompt'] and r['reference_status']=='trusted_screened' for r in rows),
        reference_status_counts=dict(Counter(r['reference_status'] for r in rows)),
        saved_B_missing=sum(r['B'] is None for r in rows),
        substantial_absent_false_positives=sum(r['absent_substantial_false_positive'] for r in primary)), metrics


def render_report(rows,metrics,summary,output):
    figures=[]
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        plt=None
    if plt is not None:
        base=[r for r in rows if r['primary_eligible'] and all(r[s] is not None for s in ('A','B','C','combined'))]
        fig,axes=plt.subplots(1,2,figsize=(11,4))
        if base:
            for signal in ('A','B','C','combined'):
                risk=np.array([1-r[signal] for r in base]); y=np.array([r['error50'] for r in base],int)
                if not y.sum() or y.sum()==len(y):
                    continue
                order=np.argsort(-risk,kind='stable'); risk,y=risk[order],y[order]
                ends=np.r_[np.flatnonzero(risk[1:]!=risk[:-1])+1,len(risk)]
                tp=np.cumsum(y)[ends-1]; fp=ends-tp
                axes[0].plot(np.r_[0,fp/(len(y)-y.sum())],np.r_[0,tp/y.sum()],label=signal)
                axes[1].step(np.r_[0,tp/y.sum()],np.r_[1,tp/ends],where='pre',label=signal)
            axes[0].plot([0,1],[0,1],color='grey',linestyle='--')
            prevalence=np.mean([r['error50'] for r in base])
            axes[1].axhline(prevalence,color='grey',linestyle='--',label='Error prevalence')
        for ax in axes:
            ax.set_xlim(0,1);ax.set_ylim(0,1.02);ax.legend();ax.grid(alpha=.2)
        axes[0].set(xlabel='False-positive rate',ylabel='Recall',title='ROC: common available observations')
        axes[1].set(xlabel='Recall',ylabel='Precision',title='Precision–recall: common observations')
        fig.suptitle('Task 4: GT-free quality scores; annotation frames excluded')
        fig.tight_layout();fig.savefig(output/'signal_comparison.svg');plt.close(fig)
        figures.append('signal_comparison.svg')
    lines=['# Task 4 consolidated evaluation','',
        'Descriptive evaluation on one video and GT-assisted annotation experiment. No weights or thresholds were fitted.',
        f"Primary observations: {summary['primary_observations']}; GT present: {summary['primary_gt_present']}; GT absent: {summary['primary_gt_absent']}.",
        '', 'Outcomes: '+json.dumps(summary['outcomes']), '',
        '| Cohort | Signal | Available / cohort | ROC AUC | Average precision | Precision | Recall |',
        '|---|---|---:|---:|---:|---:|---:|']
    fmt=lambda x:'unavailable' if x is None else f'{x:.3f}'
    for m in metrics:
        if m['error_definition']=='error50' and m['cohort'] in ('all_presence_aware','common_available'):
            lines.append(f"| {m['cohort']} | {m['signal']} | {m['n']} / {m['cohort_n']} | {fmt(m['roc_auc'])} | {fmt(m['average_precision'])} | {fmt(m['precision'])} | {fmt(m['recall'])} |")
    lines += ['', 'Precision/recall use the frozen thresholds in the protocol; they are not deployment recommendations.',
        'Compare signals on common_available for equal coverage. Full-cohort results show missing B/C and practical coverage.',
        'IoU averages/correlations use GT-present cases. Both-empty is expected absence, not IoU=1.',
        'C is fixed-position overlap with the predicted annotation mask: legitimate bird movement can lower it.',
        'Combined preserves the existing weights and legacy C=1 sentinel for small/empty masks; empty B uses the existing A/C fallback.',
        'Empty-mask warnings are an explicit baseline. GT distinguishes expected absence from tracking failure during evaluation only.',
        'No independent deployment validation or confidence interval treating adjacent frames as independent is claimed.',
        '', 'See signal_metrics.csv, threshold_sweep.csv and by_block_bird.csv for all cohorts and IoU=0.75 sensitivity.']
    for figure in figures:
        lines += ['',f'![Signal comparison]({figure})']
    if plt is None:
        lines += ['', 'Plot omitted because matplotlib is unavailable; all numerical reports were generated.']
    (output/'report.md').write_text('\n'.join(lines),encoding='utf-8')
    return figures


def run(args):
    prediction,reference=args.prediction.resolve(),args.reference.resolve()
    pv,rv,annotations,confidence,decisions=validate_metadata(prediction,reference,args.closeout)
    if args.preflight:
        print(json.dumps(dict(status='metadata_preflight_passed',prediction_rows=len(confidence),reference_rows=len(decisions),
                              trusted=sum(r['status']=='trusted_screened' for r in decisions.values())),indent=2))
        return
    output=Path(__file__).resolve().parent/'results'/('task4-'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ'))
    output.mkdir()
    report=dict(status='in_progress',prediction_run=str(prediction),reference_run=str(reference),
        quality_thresholds=THRESHOLDS,signal_protocol='A: pass-local continuity; B: saved foreground sigmoid confidence; C: predicted prompt-mask spatial IoU; combined: existing weights and fallback.',
        primary_protocol='Trusted GT only, prompt frames excluded. Error50: GT present IoU<0.5; GT absent any positive prediction. Error75 sensitivity uses 0.75.',
        limitations=['One video; descriptive evaluation, not independent deployment validation.',
                    'GT-assisted prompts; GT only enters evaluation labels, never score formulas.',
                    'Reference screening may favor easier cases; GT exclusions are preserved.',
                    'Missing B/C cause unequal coverage; compare common_available as well as full-cohort coverage.',
                    'Adjacent frames are correlated; no IID significance claims.',
                    'Annotation anchor for both passes is the saved later-pass prompt prediction; earlier-pass prompt PNG was not separately exported.'],
        inputs={str(p):sha(p,canonical=True) for p in [prediction/'verification.json',prediction/'annotations.json',
            prediction/'foreground_confidence.csv',reference/'verification.json',reference/'evaluation_set.csv',
            reference/'mask_hashes.json',args.closeout,Path(__file__)]})
    try:
        signals=score_predictions(prediction,annotations,confidence,output)
        print('GT-free signals saved. Starting reference-labelled evaluation.',flush=True)
        rows=join_reference(prediction,reference,signals,decisions,output)
        summary,metrics=summarize(rows,output)
        figures=render_report(rows,metrics,summary,output)
        report.update(status='completed',**summary,figures=figures)
    except Exception as exc:
        report.update(status='failed',error=str(exc))
        raise
    finally:
        write_json(output/'verification.json',report)
        print(f'Status: {report["status"]}; results: {output}',flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prediction',type=Path,default=Path('audit/results/propagation-20260911T180552787557Z'))
    parser.add_argument('--reference',type=Path,default=Path('audit/results/task2-reference-20260915T080605348526Z'))
    parser.add_argument('--closeout',type=Path,default=Path('audit/TASK2_CLOSEOUT.json'))
    parser.add_argument('--preflight',action='store_true',help='Check metadata without opening image files')
    run(parser.parse_args())
