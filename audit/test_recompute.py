import argparse
import json
import pickle
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from PIL import Image
from recompute import (observation, continuity, sam_labels, color, precision_recall,
                       overlap, run)


class RecomputeTests(unittest.TestCase):
    def test_missing_prediction_is_error(self):
        empty = np.zeros((20, 20), bool)
        gt = empty.copy(); gt[3:10, 3:10] = True
        row = observation(empty, gt)
        self.assertEqual((row['iou'], row['is_error'], row['status']), (0., 1, 'missing_prediction'))

    def test_gt_absence_is_not_perfect_tracking(self):
        empty = np.zeros((5, 5), bool)
        self.assertIsNone(observation(empty, empty)['iou'])
        self.assertIsNone(observation(~empty, empty)['is_error'])

    def test_shape_mismatch_is_not_resized(self):
        with self.assertRaises(ValueError):
            overlap(np.zeros((2,2), bool), np.zeros((3,3), bool))

    def test_identity_swap_counts_as_failure_under_fixed_mapping(self):
        a = np.zeros((10,10), bool); a[:3,:3] = True
        b = np.zeros((10,10), bool); b[6:,6:] = True
        self.assertEqual(observation(a,b)['iou'], 0.)
        self.assertEqual(observation(b,a)['is_error'], 1)

    def test_continuity_missing_reference_is_unavailable(self):
        mask = np.zeros((100,100), bool); mask[:8,:8] = True
        self.assertIsNone(continuity(mask,None)[0])
        self.assertEqual(continuity(mask,mask)[0], 1.)

    def test_palette_is_not_interpolated(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/'0.png'
            Image.new('RGB',(4,4),(127,0,0)).save(path)
            with self.assertRaises(ValueError):
                sam_labels(path)

    def test_metrics_use_only_available_scores(self):
        rows=[dict(is_error=1, score=.1),dict(is_error=0,score=.9),dict(is_error=1,score=None)]
        value=precision_recall(rows,'score',.5)
        self.assertEqual((value['n'],value['tp'],value['tn']), (2,1,1))

    def test_end_to_end_unique_population_and_missing_mask(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            for name in ['sam','gt','frames']:
                (root/name).mkdir()
            ids=[14,38,75,113]
            names=['darkest','brownish','orange','spotted']
            gt=np.zeros((100,100),np.uint8)
            rgb=np.zeros((100,100,3),np.uint8)
            labels=[]
            for i,(bird,name) in enumerate(zip(ids,names),1):
                x=10*i
                gt[10:18,x:x+8]=bird
                rgb[10:18,x:x+8]=color(i)
                labels.append(SimpleNamespace(name=name,pts={0:[SimpleNamespace(idx=0,x=x+3,y=13,pt_type=1)]},boxes={}))
            for frame in range(3):
                sample=rgb.copy()
                if frame==1:
                    sample[10:18,10:18]=0
                Image.fromarray(sample).save(root/'sam'/f'{frame:06d}.png')
                Image.fromarray(gt).save(root/'gt'/f'{frame:06d}.png')
                Image.new('RGB',(100,100),'gray').save(root/'frames'/f'{frame:06d}.png')
            session=root/'session.pkl'
            session.write_bytes(pickle.dumps(dict(sam_handler_labels=labels,block_size=3,num_blocks=1)))
            quality=root/'quality.csv'
            quality.write_text('abs_frame,label,logit_score\n0,darkest,0.1\n0,darkest,0.1\n')
            args=argparse.Namespace(workspace=root,session=session,masks=root/'sam',gt=root/'gt',frames=root/'frames',quality=quality,output=root/'result',gt_ids=ids)
            run(args)
            summary=json.loads((root/'result'/'summary.json').read_text())
            self.assertEqual(summary['all_frames']['evaluated'],12)
            self.assertEqual(summary['all_frames']['statuses']['missing_prediction'],1)
            self.assertEqual(summary['excluding_identity_calibration_frames']['evaluated'],8)
            with self.assertRaises(FileExistsError):
                run(args)


if __name__=='__main__':
    unittest.main()
