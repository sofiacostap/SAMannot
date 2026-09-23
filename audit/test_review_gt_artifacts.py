import json
import tempfile
import unittest
from pathlib import Path
from PIL import Image
import numpy as np
from review_gt_artifacts import ArtifactReview
from test_review_all_gt import fixture
from review_all_gt import read_csv
from recompute import sha


class ArtifactTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory(); self.root=Path(self.temp.name)
        self.ref,self.src=fixture(self.root)
        self.review=ArtifactReview(self.ref,self.src,self.root)
    def tearDown(self): self.temp.cleanup()
    def test_navigation_does_not_flag(self):
        self.review.visit(0)
        self.assertEqual(self.review.record['flagged'],[])
        self.assertEqual(self.review.record['displayed'],[0])
    def test_confirmation_and_display_required(self):
        with self.assertRaises(ValueError): self.review.flag(0,True)
        self.review.visit(0)
        with self.assertRaises(ValueError): self.review.flag(0,False)
    def test_flag_exact_frame_undo_and_sources_unchanged(self):
        original=sha(self.ref/'masks/000000.png')
        self.review.visit(0); self.review.flag(0,True)
        self.assertTrue((self.review.root/'gt artifacts/000000/gt_mask.png').exists())
        self.assertEqual(self.review.record['flagged'],[0])
        self.review.undo(0)
        self.assertFalse((self.review.root/'gt artifacts/000000').exists())
        self.assertEqual(self.review.record['flagged'],[])
        self.assertEqual(sha(self.ref/'masks/000000.png'),original)
    def test_resume_preserves_progress(self):
        self.review.visit(1);self.review.flag(1,True)
        resumed=ArtifactReview(self.ref,self.src,self.root)
        self.assertEqual(resumed.record['last'],1)
        self.assertEqual(resumed.record['flagged'],[1])
    def test_export_requires_full_pass(self):
        self.review.visit(0)
        with self.assertRaises(ValueError):self.review.finish(True)
    def test_partition_exact_counts_and_mask_pixels(self):
        self.review.visit(0);self.review.visit(1);self.review.flag(1,True)
        result=self.review.root/self.review.finish(True)
        rows=read_csv(result/'manifest.csv')
        self.assertEqual([r['category'] for r in rows],['retained GT','gt artifacts'])
        for row in rows:
            with Image.open(result/row['mask']) as im: actual=np.asarray(im)
            with Image.open(self.ref/'masks'/f"{int(row['gt_frame']):06d}.png") as im: expected=np.asarray(im)
            np.testing.assert_array_equal(actual,expected)
        self.assertEqual(json.loads((result/'verification.json').read_text())['retained_frames'],1)
    def test_export_rechecks_previously_viewed_mask(self):
        self.review.visit(0);self.review.visit(1)
        Image.fromarray(np.zeros((16,24),np.uint8)).save(self.ref/'masks/000000.png')
        with self.assertRaises(ValueError):self.review.finish(True)
        self.assertTrue(all(p.name.endswith('.incomplete') for p in (self.review.root/'exports').iterdir()))


if __name__=='__main__': unittest.main()
