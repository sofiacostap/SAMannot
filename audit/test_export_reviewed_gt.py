import json
import tempfile
import unittest
from pathlib import Path
import numpy as np
from PIL import Image
from review_all_gt import Review, read_csv
from export_reviewed_gt import export
from test_review_all_gt import fixture


class ExportTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.ref, self.src = fixture(self.root)
        self.review = Review(self.ref, self.src, self.root/'review')
        self.dest = self.root/'exports'; self.dest.mkdir()

    def tearDown(self):
        self.temp.cleanup()

    def complete(self):
        self.review.update(0, [dict(id=14, status='accepted_present', note=''),
                               dict(id=38, status='uncertain', note='Unclear')])
        self.review.update(1, [dict(id=14, status='confirmed_problem', note='Wrong'),
                               dict(id=38, status='accepted_absent', note='')])

    def run_export(self):
        return export(self.root/'review', self.ref, self.src, self.dest)

    def test_incomplete_review_refused_without_output(self):
        with self.assertRaises(ValueError): self.run_export()
        self.assertEqual(list(self.dest.iterdir()), [])

    def test_only_accepted_pairs_and_exact_masks(self):
        self.complete(); output = self.run_export()
        rows = read_csv(output/'manifest.csv')
        self.assertEqual([(r['video_frame'], r['gt_id']) for r in rows], [('0','14'), ('1','38')])
        for row in rows:
            with Image.open(output/row['mask']) as im: mask = np.asarray(im)
            with Image.open(self.ref/'masks'/f"{int(row['gt_frame']):06d}.png") as im: gt = np.asarray(im)
            np.testing.assert_array_equal(mask, (gt==int(row['gt_id'])).astype(np.uint8)*255)
        report = json.loads((output/'verification.json').read_text())
        self.assertEqual(report['accepted_observations'], 2)
        self.assertEqual(report['omitted_uncertain'], 1)
        self.assertEqual(report['omitted_confirmed_problem'], 1)
        self.assertNotIn(str(self.root), (output/'verification.json').read_text())

    def test_changed_mask_blocks_completed_export(self):
        self.complete()
        Image.fromarray(np.zeros((16,24), np.uint8)).save(self.ref/'masks/000000.png')
        with self.assertRaises(ValueError): self.run_export()
        self.assertTrue(all(p.name.endswith('.incomplete') for p in self.dest.iterdir()))

    def test_repeated_exports_preserve_prior_output(self):
        self.complete(); first = self.run_export(); second = self.run_export()
        self.assertNotEqual(first, second)
        self.assertEqual((first/'manifest.csv').read_bytes(), (second/'manifest.csv').read_bytes())


if __name__ == '__main__': unittest.main()
