import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from prepare_task2_reference import normalize, compare_core, PALETTE
from build_task2_set import partition


def save_palette(path, data, palette=PALETTE):
    im = Image.fromarray(np.array(data, dtype=np.uint8)).convert('P')
    colors = [0] * 768
    for index, rgb in palette.items():
        colors[index*3:index*3+3] = rgb
    im.putpalette(colors)
    im.save(path)


class ReferenceTests(unittest.TestCase):
    def test_enlargement_preserves_labels_and_area(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp)/'0.png'
            save_palette(path, [[0,1,2],[3,4,0]])
            gt = normalize(path, (4,6))
            self.assertEqual(set(np.unique(gt)), {0,14,38,75,113})
            self.assertEqual(gt[:2,2:4].tolist(), [[38,38],[38,38]])
            for bird in (14,38,75,113):
                self.assertEqual(np.count_nonzero(gt==bird), 4)

    def test_unknown_label_palette_change_and_wrong_size_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp)/'0.png'
            save_palette(path, [[0,5]])
            with self.assertRaises(ValueError): normalize(path, (2,4))
            changed = dict(PALETTE, **{})
            changed[1] = (0,128,0)
            save_palette(path, [[0,1]], changed)
            with self.assertRaises(ValueError): normalize(path, (2,4))
            save_palette(path, [[0,1]])
            with self.assertRaises(ValueError): normalize(path, (4,8))
            Image.new('L', (2,1)).save(path)
            with self.assertRaises(ValueError): normalize(path, (2,4))

    def test_correspondence_detects_wrong_identity_and_missing_support(self):
        core = np.ones((20,20), bool)
        self.assertTrue(compare_core(core, np.full(core.shape,75),75)[-1])
        self.assertFalse(compare_core(core, np.full(core.shape,38),75)[-1])
        self.assertFalse(compare_core(core, np.full(core.shape,7),75)[-1])
        self.assertFalse(compare_core(core, np.zeros(core.shape),75)[-1])

    def test_exchange_holds_both_birds_on_both_sides_only(self):
        metrics = [dict(video_frame=f, gt_frame=f-12, gt_id=g, area=400)
                   for f in (99,100,101) for g in (14,38,75,113)]
        candidates = [dict(video_frame=100, gt_id='38+75', reason='possible_identity_exchange')]
        rows, _ = partition(metrics, candidates, {'decisions': []})
        held = {(r['video_frame'],r['gt_id']) for r in rows if r['status']=='uncertain'}
        self.assertEqual(held, {(99,38),(99,75),(100,38),(100,75)})


if __name__ == '__main__':
    unittest.main()
