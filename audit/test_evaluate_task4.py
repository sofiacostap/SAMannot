import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from evaluate_task4 import (IDS, average_ranks, ranking_metrics, confusion,
    continuity, mask_stats, reference_score, combine, label_outcome, passes,
    unique_rows, sha, write_csv, write_json, validate_metadata,
    score_predictions, join_reference, summarize)


class MetricTests(unittest.TestCase):
    def test_rank_ties_and_perfect_reversed_constant_rankings(self):
        np.testing.assert_equal(average_ranks([2,1,2]), [2.5,1,2.5])
        self.assertEqual(ranking_metrics([0,1],[0,1]),dict(roc_auc=1.,average_precision=1.))
        self.assertEqual(ranking_metrics([0,1],[1,0]),dict(roc_auc=0.,average_precision=.5))
        self.assertEqual(ranking_metrics([0,1],[1,1]),dict(roc_auc=.5,average_precision=.5))
        self.assertEqual(ranking_metrics([1,1],[0,1]),dict(roc_auc=None,average_precision=None))

    def test_missing_class_and_no_flags_not_reported_as_perfect(self):
        c=confusion([True,False],[False,False])
        self.assertIsNone(c['precision'])
        self.assertEqual(c['recall'],0.)
        self.assertEqual(c['fn'],1)
        self.assertIsNone(confusion([],[])['flag_rate'])

    def test_presence_aware_outcomes_and_no_empty_iou_reward(self):
        empty=np.zeros((20,20),bool); bird=empty.copy();bird[2:10,2:10]=True
        self.assertIsNone(label_outcome(empty,empty)['iou'])
        self.assertFalse(label_outcome(empty,empty)['error50'])
        self.assertEqual(label_outcome(empty,bird)['outcome'],'missing_visible_bird')
        self.assertEqual(label_outcome(bird,empty)['outcome'],'false_positive_absent_bird')
        self.assertEqual(label_outcome(bird,bird)['iou'],1.)

    def test_continuity_reset_jump_and_empty(self):
        current=(100,(10.,10.))
        self.assertEqual(continuity(current,None,10000)[0],1.)
        self.assertAlmostEqual(continuity(current,current,10000)[0],1.,places=6)
        self.assertEqual(continuity((100,(500.,500.)),current,10000)[0],0.)
        self.assertEqual(continuity((0,None),current,10000)[0],0.)

    def test_c_tracks_position_not_biological_correctness_and_legacy_fallback(self):
        a=np.zeros((100,100),bool);a[0:10,0:10]=True
        b=np.zeros_like(a);b[20:30,20:30]=True
        self.assertEqual(reference_score(a,a),1.)
        self.assertEqual(reference_score(b,a),0.)
        self.assertIsNone(reference_score(np.zeros_like(a),a))
        self.assertEqual(combine(0,None,None),.5)
        self.assertAlmostEqual(combine(.2,.9,.5),.57)

    def test_passes_do_not_repeat_prompt_or_cross_block(self):
        self.assertEqual(passes(750,1499,1124)[1][1][0],1123)
        covered=[f for _,fs in passes(750,1499,1124) for f in fs]
        self.assertEqual(sorted(covered),list(range(750,1500)))
        self.assertEqual(len(covered),len(set(covered)))
        with self.assertRaises(ValueError):passes(750,1499,0)

    def test_duplicate_keys_fail_closed(self):
        with self.assertRaises(ValueError):unique_rows([{'a':1},{'a':1}],('a',))


class EndToEndTests(unittest.TestCase):
    def test_prediction_only_scoring_release_join_exclusions_and_hash_guard(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);pred=root/'prediction';ref=root/'reference';out=root/'output'
            for p in (pred,ref,out,ref/'masks'):p.mkdir(parents=True,exist_ok=True)
            conf=[];decisions=[];hashes={}
            for g in IDS:(pred/'masks_by_gt_id'/str(g)).mkdir(parents=True)
            for frame in range(3):
                gt=np.zeros((100,100),np.uint8)
                for index,g in enumerate(IDS):
                    mask=np.zeros_like(gt,dtype=bool)
                    mask[10+index*20:18+index*20,10:18]=True
                    gt[mask]=g
                    if frame==0 and g==14:mask[:]=False
                    Image.fromarray(mask.astype(np.uint8)*255).save(pred/'masks_by_gt_id'/str(g)/f'{frame:06d}.png')
                    conf.append(dict(abs_frame=frame,gt_id=g,sam2_label=index+1,block_id=0,
                        direction='earlier' if frame==0 else 'later',foreground_logit_score='' if not mask.any() else .98))
                    decisions.append(dict(video_frame=frame,gt_frame=frame,gt_id=g,status='uncertain' if (frame,g)==(2,38) else 'trusted_screened',
                                          reasons='fixture_hold' if (frame,g)==(2,38) else '',gt_area=64,normalized_mask=f'masks/{frame:06d}.png'))
                Image.fromarray(gt).save(ref/'masks'/f'{frame:06d}.png')
                hashes[str(frame)]={'normalized':sha(ref/'masks'/f'{frame:06d}.png')}
            ann=dict(total_frames=3,block_size=3,source_directory='verified_source',blocks={'0':{'video_frame':1}})
            write_json(pred/'annotations.json',ann)
            write_json(pred/'verification.json',dict(status='completed',source_directory='verified_source',frames_written=3,
                        palette_to_gt={str(i+1):g for i,g in enumerate(IDS)}))
            write_json(ref/'verification.json',dict(status='completed_reference_partition',source_directory='verified_source',missing_gt_video_frames=[]))
            write_json(ref/'mask_hashes.json',hashes)
            write_csv(pred/'foreground_confidence.csv',conf)
            write_csv(ref/'evaluation_set.csv',decisions)
            release=root/'closeout.json'
            write_json(release,dict(task2_status='COMPLETE_WITH_DOCUMENTED_EXCLUSIONS',reference_run=ref.name,
                verification_sha256=sha(ref/'verification.json',True),evaluation_set_sha256=sha(ref/'evaluation_set.csv',True)))
            _,_,ann,confidence,refs=validate_metadata(pred,ref,release)
            signals=score_predictions(pred,ann,confidence,out)
            self.assertEqual(len(signals),12)
            s={(r['video_frame'],r['gt_id']):r for r in signals}
            self.assertIsNone(s[(0,14)]['B'])
            self.assertEqual(s[(0,14)]['continuity_previous_frame'],1)
            self.assertEqual(s[(2,14)]['continuity_previous_frame'],1)
            self.assertIsNone(s[(1,14)]['continuity_previous_frame'])
            rows=join_reference(pred,ref,signals,refs,out)
            summary,metrics=summarize(rows,out)
            self.assertEqual(summary['primary_observations'],7)
            self.assertEqual(summary['annotation_observations_excluded'],4)
            self.assertEqual(summary['outcomes']['missing_visible_bird'],1)
            held=next(r for r in rows if (r['video_frame'],r['gt_id'])==(2,38))
            self.assertIsNone(held['iou'])
            self.assertFalse(held['primary_eligible'])
            overall_b=next(m for m in metrics if m['cohort']=='all_presence_aware' and m['signal']=='B' and m['error_definition']=='error50')
            self.assertEqual(overall_b['missing'],1)
            # GT alteration cannot alter the already saved prediction-only signals.
            signal_hash=sha(out/'signals.csv')
            Image.new('L',(100,100),0).save(ref/'masks/000000.png')
            with self.assertRaises(ValueError):join_reference(pred,ref,signals,refs,out)
            self.assertEqual(sha(out/'signals.csv'),signal_hash)
            (ref/'evaluation_set.csv').write_text('changed')
            with self.assertRaises(ValueError):validate_metadata(pred,ref,release)


if __name__=='__main__':
    unittest.main()
