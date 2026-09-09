"""Read-only numerical review of an uploaded recomputation directory."""
import csv
import itertools
import json
from collections import defaultdict
from pathlib import Path
import sys


def review(directory):
    directory = Path(directory)
    def read_csv(name):
        with (directory/name).open(newline='', encoding='utf-8-sig') as stream:
            return list(csv.DictReader(stream))
    rows = read_csv('recomputed.csv')
    cal = read_csv('calibration_pairs.csv')
    global_map = json.loads((directory/'manifest.json').read_text())['mapping']
    old = read_csv('legacy_iou.csv')
    frame_matrices = defaultdict(dict)
    for row in cal:
        frame_matrices[int(row['frame'])][int(row['sam2_label']),int(row['gt_bird_id'])] = float(row['iou'])
    checks = []
    for frame, matrix in sorted(frame_matrices.items()):
        candidates = sorted(((sum(matrix.get((i,b),0.) for i,b in enumerate(perm,1))/4,perm)
                             for perm in itertools.permutations([14,38,75,113])), reverse=True)
        checks.append(dict(frame=frame, best_mapping=dict(enumerate(candidates[0][1],1)),
                           best_mean_iou=candidates[0][0], margin=candidates[0][0]-candidates[1][0],
                           global_mapping_mean_iou=sum(matrix.get((i,int(global_map[str(i)])),0.) for i in range(1,5))/4))
    blocks = []
    for block in range(6):
        subset=[r for r in rows if int(r['block_id'])==block and r['iou']!='' and r['is_calibration_frame']=='0']
        blocks.append(dict(block=block, n=len(subset), mean_iou=sum(float(r['iou']) for r in subset)/len(subset),
                           error_rate=sum(int(r['is_error']) for r in subset)/len(subset)))
    grid=read_csv('ac_grid_exploratory.csv')
    best=sorted(grid,key=lambda r:float(r['f1']),reverse=True)[:3]
    baseline=next(r for r in grid if float(r['w_cont'])==.5 and abs(float(r['threshold'])-.35)<1e-8)
    old_disagree=sum(int(r['gt_bird_id'])!=int(global_map[r['sam2_label']]) for r in old)
    paired_new={(r['frame_idx'],r['sam2_label']):r for r in rows if r['iou']!=''}
    pairs=[(r,paired_new[(r['frame_idx'],r['sam2_label'])]) for r in old if (r['frame_idx'],r['sam2_label']) in paired_new]
    comparison=dict(n=len(pairs),old_mean_iou=sum(float(a['iou']) for a,b in pairs)/len(pairs),
                    new_mean_iou=sum(float(b['iou']) for a,b in pairs)/len(pairs))
    return dict(calibration_by_frame=checks, blocks_excluding_calibration=blocks,
                old_rows_differing_from_proposed_fixed_map=old_disagree,
                matched_key_comparison=comparison, best_ac_exploratory=best, ac_equal_weight_threshold_035=baseline)


if __name__=='__main__':
    print(json.dumps(review(sys.argv[1]),indent=2))
