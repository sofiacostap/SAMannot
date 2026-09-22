import csv
import json
import tempfile
import unittest
from pathlib import Path
import numpy as np
from PIL import Image
from review_all_gt import Review, counts, inventory, validate_decision
from recompute import sha
from verify_correspondence import fingerprint


def fixture(root):
    ref,src=root/'reference',root/'source'
    (ref/'masks').mkdir(parents=True);(src/'frames').mkdir(parents=True)
    rows=[];mapping=[];hashes={}
    for frame in (0,1):
        rgb=np.full((16,24,3),80+frame,np.uint8)
        gt=np.zeros((16,24),np.uint8);gt[2:8,3:10]=14
        Image.fromarray(rgb).save(src/'frames'/f'{frame:06d}.png')
        Image.fromarray(gt).save(ref/'masks'/f'{frame:06d}.png')
        mapping.append(dict(video_index=frame,gt_frames_index=frame,rgb_sha256=fingerprint(rgb),file=f'{frame:06d}.png'))
        hashes[str(frame)]={'normalized':sha(ref/'masks'/f'{frame:06d}.png')}
        for bird in (14,38):rows.append(dict(video_frame=frame,gt_frame=frame,gt_id=bird,
            gt_present=bird==14,status='uncertain' if bird==14 else 'exclude_from_evaluation',
            reasons='test',normalized_mask=f'masks/{frame:06d}.png'))
    for path,data in [(ref/'evaluation_set.csv',rows),(src/'index.csv',mapping)]:
        with path.open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=list(data[0]));w.writeheader();w.writerows(data)
    (ref/'mask_hashes.json').write_text(json.dumps(hashes));(src/'verification.json').write_text('{"status":"verified"}')
    return ref,src


class ReviewTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.root=Path(self.temp.name)
        self.ref,self.src=fixture(self.root);self.review=Review(self.ref,self.src,self.root/'review')
    def tearDown(self):self.temp.cleanup()
    def test_all_prior_categories_start_pending(self):
        self.assertEqual(self.review.record['counts']['pending'],4)
        self.assertEqual(len(self.review.frames),2)
    def test_view_does_not_approve_and_png_is_valid(self):
        self.assertTrue(self.review.picture(0,'overlay',14,True).startswith(b'\x89PNG'))
        self.assertEqual(self.review.record['counts']['pending'],4)
    def test_explicit_save_resume_and_other_birds_unchanged(self):
        self.review.update(0,[dict(id=14,status='confirmed_problem',note='Wrong identity')])
        resumed=Review(self.ref,self.src,self.root/'review',True)
        self.assertEqual(resumed.record['counts']['pending'],3)
        self.assertEqual(resumed.record['counts']['confirmed_problem'],1)
        self.assertEqual(resumed.record['decisions']['0:38']['status'],'pending')
    def test_absence_acceptance_and_flag_reason(self):
        with self.assertRaises(ValueError):validate_decision({'gt_present':True},'accepted_absent','')
        with self.assertRaises(ValueError):validate_decision({'gt_present':False},'accepted_present','')
        with self.assertRaises(ValueError):validate_decision({'gt_present':False},'uncertain','')
        self.review.update(0,[dict(id=38,status='accepted_absent',note='No visible bird')])
        self.assertEqual(self.review.record['counts']['accepted_absent'],1)
    def test_invalid_batch_is_not_partially_saved(self):
        with self.assertRaises(ValueError):self.review.update(0,[dict(id=14,status='accepted_present',note=''),dict(id=38,status='confirmed_problem',note='')])
        self.assertEqual(self.review.record['counts']['pending'],4)
    def test_changed_pixels_rejected(self):
        Image.fromarray(np.zeros((16,24),np.uint8)).save(self.ref/'masks/000000.png')
        with self.assertRaises(ValueError):self.review.pixels(0)
    def test_complete_review_can_still_be_unresolved(self):
        c=counts({'0':{'status':'uncertain'}})
        self.assertTrue(c['all_observations_reviewed']);self.assertEqual(c['unresolved'],1)
    def test_resume_rejects_changed_manifest(self):
        with (self.src/'index.csv').open('a') as f:f.write('\n')
        with self.assertRaises(ValueError):Review(self.ref,self.src,self.root/'review',True)


if __name__=='__main__':unittest.main()
