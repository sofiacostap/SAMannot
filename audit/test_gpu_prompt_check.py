import copy
import unittest
from gpu_prompt_check import validate_structure


class PromptCheckTest(unittest.TestCase):
    def setUp(self):
        self.data=dict(status='ready_for_integration',gt_ids=[14,38,75,113],total_frames=4,block_size=3,
            blocks={str(b):dict(video_frame=f,local_frame=f-b*3,human_reviewed=True,
                points=[dict(gt_id=g,positive=True) for g in [14,38,75,113]]) for b,f in [(0,1),(1,3)]})
    def test_valid_middle_and_short_last_block(self):validate_structure(self.data)
    def test_reject_cross_block_annotation(self):
        self.data['blocks']['0']['video_frame']=3
        with self.assertRaises(ValueError):validate_structure(self.data)
    def test_reject_unreviewed_or_duplicate_ids(self):
        bad=copy.deepcopy(self.data);bad['blocks']['0']['human_reviewed']=False
        with self.assertRaises(ValueError):validate_structure(bad)
        self.data['blocks']['0']['points'][0]['gt_id']=38
        with self.assertRaises(ValueError):validate_structure(self.data)
