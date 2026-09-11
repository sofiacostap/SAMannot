import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
from reannotate import Review, bounds, validate_points, IDS


class ReannotationTest(unittest.TestCase):
    def setUp(self):
        self.gt=np.array([[14,38],[75,113]],dtype=np.uint8)
        self.points=[dict(gt_id=bird,x=i%2+.2,y=i//2+.2) for i,bird in enumerate(IDS)]

    def test_block_limits_include_short_final_block(self):
        self.assertEqual(bounds(5,750,3814),(3750,3813))
        with self.assertRaises(ValueError):bounds(6,750,3814)

    def test_wrong_identity_and_nonfinite_click_rejected(self):
        self.assertEqual(len(validate_points(self.points,self.gt)),4)
        self.points[0]['x']=1.2
        with self.assertRaisesRegex(ValueError,'outside that GT mask'):validate_points(self.points,self.gt)
        self.points[0]['x']=float('nan')
        with self.assertRaises(ValueError):validate_points(self.points,self.gt)

    def test_duplicate_identity_rejected(self):
        self.points[0]['gt_id']=38
        with self.assertRaises(ValueError):validate_points(self.points,self.gt)

    def test_save_requires_review_and_persists_frame_provenance(self):
        with tempfile.TemporaryDirectory() as tmp:
            review=Review.__new__(Review);review.output=Path(tmp);review.size=750;review.total=3814
            review.num_blocks=6;review.data=dict(blocks={})
            meta=dict(gt_frame=362,gt_sha256='GT_HASH',rgb_sha256='IMAGE_HASH')
            payload=dict(block=0,frame=374,points=self.points,reviewed=False)
            with patch.object(review,'metadata',return_value=meta),patch.object(review,'frame',return_value=(None,self.gt,meta)):
                with self.assertRaisesRegex(ValueError,'Visually review'):review.save(payload)
                payload['reviewed']=True
                result=review.save(payload)
            self.assertEqual(result['saved_blocks'],[0])
            saved=json.loads((review.output/'annotations.json').read_text())['blocks']['0']
            self.assertEqual(saved['video_frame'],374)
            self.assertEqual(saved['gt_frame'],362)
            self.assertEqual(saved['gt_sha256'],'GT_HASH')
