"""Task 6: descriptive temporal plots from the frozen Task 4 observation table.

No image inference, changed scores, fitted thresholds, or applied identity swaps.
"""
import argparse
import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
from evaluate_task4 import read_csv, sha, write_csv, write_json

COLORS={14:'#c74444',38:'#1599ae',75:'#b58b00',113:'#8755b5'}


def summarize_group(rows):
    present=[r for r in rows if r['gt_present']]
    return dict(n=len(rows),gt_present=len(present),errors=sum(r['error50'] for r in rows),
        error_rate=sum(r['error50'] for r in rows)/len(rows) if rows else None,
        present_errors=sum(r['error50'] for r in present),
        present_error_rate=sum(r['error50'] for r in present)/len(present) if present else None,
        mean_segmentation_error=float(np.mean([1-r['iou'] for r in present])) if present else None)


def run(args):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    release=json.loads(args.closeout.read_text())
    if args.task4.name!=release['result_run'] or release['task4_status']!='COMPLETE_EVALUATION':
        raise ValueError('Use frozen completed Task 4 run')
    for name,digest in release['checksums'].items():
        if sha(args.task4/name,True)!=digest:raise ValueError('Frozen Task 4 evidence changed')
    ann=json.loads(args.annotations.read_text())
    rows=[]
    for r in read_csv(args.task4/'evaluation.csv'):
        if r['primary_eligible']!='True':continue
        for key in ('video_frame','block_id','gt_id','distance_from_prompt','prompt_frame'):r[key]=int(r[key])
        for key in ('gt_present','error50'):r[key]=r[key]=='True'
        r['iou']=float(r['iou']) if r['iou'] else None
        first=r['block_id']*ann['block_size'];last=min(ann['total_frames']-1,first+ann['block_size']-1)
        r['position']=(r['video_frame']-first)/(last-first)
        extent=r['prompt_frame']-first if r['direction']=='earlier' else last-r['prompt_frame']
        r['relative_distance']=r['distance_from_prompt']/extent if extent else 0.
        rows.append(r)
    if len(rows)!=release['primary_observations']:raise ValueError('Observation total changed')
    out=Path(__file__).resolve().parent/'results'/('temporal-'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ'))
    out.mkdir()
    bands=[];distance=[];positions=[];frame_bins=[];blocks=[]
    for block in sorted({r['block_id'] for r in rows}):
        selected=[r for r in rows if r['block_id']==block]
        blocks.append(dict(block_id=block,**summarize_group(selected)))
        for region,predicate in [('first_third',lambda r:r['position']<1/3),
                                 ('middle_third',lambda r:1/3<=r['position']<2/3),
                                 ('last_third',lambda r:r['position']>=2/3)]:
            bands.append(dict(block_id=block,region=region,**summarize_group([r for r in selected if predicate(r)])))
        for direction in ('earlier','later'):
            branch=[r for r in selected if r['direction']==direction]
            for band,predicate in [('near',lambda r:r['relative_distance']<=.25),
                                   ('far',lambda r:r['relative_distance']>.75)]:
                distance.append(dict(block_id=block,direction=direction,band=band,
                                     **summarize_group([r for r in branch if predicate(r)])))
        for decile in range(10):
            subset=[r for r in selected if min(9,int(r['position']*10))==decile]
            positions.append(dict(block_id=block,position_bin=decile,**summarize_group(subset)))
        first=block*ann['block_size'];last=min(ann['total_frames']-1,first+ann['block_size']-1)
        for start in range(first,last+1,25):
            for bird in COLORS:
                subset=[r for r in selected if r['gt_id']==bird and start<=r['video_frame']<start+25]
                frame_bins.append(dict(block_id=block,gt_id=bird,start=start,end=min(last,start+24),**summarize_group(subset)))
    for filename,data in [('block_summary.csv',blocks),('block_thirds.csv',bands),('near_far.csv',distance),
                          ('position_bins.csv',positions),('frame_bins.csv',frame_bins)]:write_csv(out/filename,data)
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    events=json.loads((args.swap_review/'events.json').read_text()) if args.swap_review else []
    fig,axes=plt.subplots(3,2,figsize=(14,11),layout='constrained')
    for block,ax in enumerate(axes.flat):
        first=block*ann['block_size'];last=min(ann['total_frames']-1,first+ann['block_size']-1)
        prompt=ann['blocks'][str(block)]['video_frame']
        for bird,color in COLORS.items():
            subset=[r for r in rows if r['block_id']==block and r['gt_id']==bird and r['gt_present']]
            ax.scatter([r['video_frame'] for r in subset],[1-r['iou'] for r in subset],s=3,alpha=.16,color=color,rasterized=True)
            bins=[b for b in frame_bins if b['block_id']==block and b['gt_id']==bird]
            ax.plot([(b['start']+b['end'])/2 for b in bins],
                    [b['mean_segmentation_error'] if b['mean_segmentation_error'] is not None else np.nan for b in bins],
                    color=color,lw=1.5,label=f'ID {bird}')
        ax.axvline(prompt,color='black',ls='--',lw=1,label='Annotation')
        for e in events:
            if e['block_id']==block:ax.axvspan(e['start'],e['end'],color='#eab56c',alpha=.17)
        ax.set(xlim=(first,last),ylim=(0,1.02),title=f'Block {block}: annotation {prompt}',
               xlabel='Video frame (each panel has its own frame range)',ylabel='Segmentation error = 1 - IoU')
        ax.grid(alpha=.2)
    axes.flat[0].legend(fontsize=8,ncol=3)
    fig.suptitle('Frozen baseline: per-frame error and 25-frame bin means\nGT-present, screened observations; annotation frames excluded; orange = proposed swap intervals',fontsize=13)
    for suffix in ('svg','png'):fig.savefig(out/f'per_block_error.{suffix}',dpi=160)
    plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(12,4.7),layout='constrained')
    for block in range(6):
        data=[r for r in positions if r['block_id']==block]
        x=[(r['position_bin']+.5)/10 for r in data]
        for ax,key in zip(axes,['mean_segmentation_error','present_error_rate']):
            ax.plot(x,[r[key] if r[key] is not None else np.nan for r in data],marker='o',ms=3,label=f'Block {block}')
            ax.set(xlim=(0,1),ylim=(0,1),xlabel='Relative position within block')
            ax.grid(alpha=.2)
    axes[0].set_ylabel('Mean 1 - IoU (GT present)');axes[1].set_ylabel('Fraction with IoU < 0.5 (GT present)')
    axes[0].legend(ncol=2,fontsize=8)
    fig.suptitle('Within-block pattern: descriptive, unequal sample counts; block 5 is only 64 frames')
    for suffix in ('svg','png'):fig.savefig(out/f'block_position.{suffix}',dpi=160)
    plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(12,4.7),layout='constrained')
    for direction,ax in zip(('earlier','later'),axes):
        for block in range(6):
            vals=[next(r for r in distance if r['block_id']==block and r['direction']==direction and r['band']==band) for band in ('near','far')]
            ax.plot([0,1],[r['mean_segmentation_error'] if r['mean_segmentation_error'] is not None else np.nan for r in vals],marker='o',label=f'Block {block}')
        ax.set(xticks=[0,1],xticklabels=['Nearest 25% of branch','Farthest 25% of branch'],ylim=(0,1),
               title=f'{direction.capitalize()} propagation branch',ylabel='Mean 1 - IoU (GT present)')
        ax.grid(alpha=.2)
    axes[0].legend(ncol=2,fontsize=8)
    fig.suptitle('Distance from the annotation: direction-specific descriptive comparison')
    for suffix in ('svg','png'):fig.savefig(out/f'distance_from_annotation.{suffix}',dpi=160)
    plt.close(fig)
    comparisons=[]
    for block in range(6):
        near=[r for r in rows if r['block_id']==block and r['relative_distance']<=.25]
        far=[r for r in rows if r['block_id']==block and r['relative_distance']>.75]
        comparisons.append(dict(block_id=block,near=summarize_group(near),far=summarize_group(far)))
    summary=dict(status='completed_descriptive_analysis',baseline_unchanged=True,n=len(rows),
        primary_errors=sum(r['error50'] for r in rows),gt_present=sum(r['gt_present'] for r in rows),
        distance_comparison=comparisons,
        inputs={str(args.task4/'evaluation.csv'):sha(args.task4/'evaluation.csv',True),str(args.closeout):sha(args.closeout,True)},
        limitations=['No causal claim or fitted alert rule.', 'Neighbours are correlated; no IID significance test.',
                     'GT-present curves exclude absent-bird outcomes; tables also show presence-aware errors.',
                     '25-frame bins and block-position bins have unequal retained counts, recorded in CSV.',
                     'Reference exclusions and baseline predictions unchanged. Proposed swaps are not applied.',
                     'Each block has one interior annotation; geometric distance is generally largest toward the block ends.'])
    assert sum(b['n'] for b in blocks)==14175
    assert sum(b['errors'] for b in blocks)==2169
    assert sum(b['gt_present'] for b in blocks)==13068
    assert sum(b['n'] for b in bands)==14175
    assert sum(b['n'] for b in positions)==14175
    assert sum(b['n'] for b in frame_bins)==14175
    write_json(out/'verification.json',summary)
    print(json.dumps(summary,indent=2))
    print('Results:',out)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--task4',type=Path,default=Path('audit/results/task4-20260915T084226849510Z'))
    p.add_argument('--closeout',type=Path,default=Path('audit/TASK4_CLOSEOUT.json'))
    p.add_argument('--annotations',type=Path,default=Path('audit/results/propagation-20260911T180552787557Z/annotations.json'))
    p.add_argument('--swap-review',type=Path,default=Path('audit/results/swap-review-20260915T154510599128Z'))
    run(p.parse_args())
