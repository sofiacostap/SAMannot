import json
import io
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
    def test_photo_or_binary_view_can_be_flagged_without_full_gt_coverage(self):
        self.review.visit(0, gt_view=False)
        self.review.flag(0, True)
        self.assertEqual(self.review.record['flagged'], [0])
        self.assertEqual(self.review.record['displayed'], [])
        with self.assertRaises(ValueError): self.review.finish(True)
        self.review.visit(0, gt_view=True)
        self.assertEqual(self.review.record['displayed'], [0])
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


class OriginalTests(unittest.TestCase):
    def setUp(self):
        from prepare_task2_reference import PALETTE
        self.temp=tempfile.TemporaryDirectory();self.root=Path(self.temp.name)
        self.ref,self.src=fixture(self.root);self.raw=self.root/'original';self.raw.mkdir()
        hashes=json.loads((self.ref/'mask_hashes.json').read_text())
        for i in (0,1):
            data=np.zeros((8,12),np.uint8);data[2:5,3:6]=4;data[0,11]=2
            im=Image.fromarray(data).convert('P');palette=[0]*768
            for k,color in PALETTE.items():palette[k*3:k*3+3]=color
            im.putpalette(palette);im.save(self.raw/f'{i:06d}.png')
            hashes[str(i)]['original']=sha(self.raw/f'{i:06d}.png')
        (self.ref/'mask_hashes.json').write_text(json.dumps(hashes))
        # The fixture inventory has two identities; use a listed ID for the detached speck.
        self.review=ArtifactReview(self.ref,self.src,self.root,self.raw)
    def tearDown(self):self.temp.cleanup()
    def test_original_palette_bytes_are_unchanged(self):
        self.assertEqual(self.review.source.picture(0,'raw',0,False),(self.raw/'000000.png').read_bytes())
    def test_binary_reads_source_not_converted_mask(self):
        binary=self.review.source.picture(0,'binary',14,False)
        with Image.open(io.BytesIO(binary)) as im: actual=np.asarray(im)
        with Image.open(self.raw/'000000.png') as im: expected=np.asarray(im)==4
        np.testing.assert_array_equal(actual,expected.astype(np.uint8)*255)
        Image.fromarray(np.zeros((16,24),np.uint8)).save(self.ref/'masks/000000.png')
        _,gt=self.review.source.pixels(0)
        self.assertEqual(gt[0,22],75)
    def test_changed_original_rejected(self):
        (self.raw/'000000.png').write_bytes(b'changed')
        with self.assertRaises(ValueError):self.review.source.pixels(0)
    def test_migration_keeps_flags_and_separates_coverage(self):
        dest=self.root/'legacy'
        old=ArtifactReview(self.ref,self.src,dest);old.visit(0);old.flag(0,True)
        new=ArtifactReview(self.ref,self.src,dest,self.raw)
        self.assertEqual(new.record['flagged'],[0])
        self.assertEqual(new.record['displayed'],[])
        self.assertEqual(new.record['previous_derivative_displayed'],[0])
    def test_export_preserves_original_file(self):
        self.review.visit(0);self.review.visit(1)
        output=self.review.root/self.review.finish(True)
        rows=read_csv(output/'manifest.csv')
        self.assertEqual((output/rows[0]['original_gt']).read_bytes(),(self.raw/'000000.png').read_bytes())


if __name__=='__main__': unittest.main()
