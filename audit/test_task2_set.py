import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from PIL import Image

from build_task2_set import partition, read_csv
from check_gt_encoding import run, sha, load


class PolicyTests(unittest.TestCase):
    def test_real_partition_conserves_inventory_and_localizes_decisions(self):
        root = Path(__file__).resolve().parent
        audit = root / 'results/gt-audit-20260912T121456077403Z'
        rows, held = partition(read_csv(audit / 'metrics.csv'), read_csv(audit / 'candidates.csv'),
                               json.loads((root / 'task2_gt_review.json').read_text()))
        lookup = {(r['video_frame'], r['gt_id']): r for r in rows}
        self.assertEqual(len(lookup), 15208)
        self.assertEqual(len(held), 56)
        for key in ((663, 75), (1751, 38), (1751, 113)):
            self.assertEqual(lookup[key]['status'], 'exclude_from_evaluation')
        self.assertNotIn((663, 14), held)
        self.assertNotIn((1751, 75), held)
        self.assertEqual(lookup[(663, 75)]['gt_frame'], 651)
        self.assertEqual(lookup[(1751, 113)]['gt_frame'], 1739)
        self.assertFalse(any(r['status'] == 'trusted' for r in rows))

    def test_duplicate_and_invalid_mapping_rejected(self):
        row = dict(video_frame='15', gt_frame='3', gt_id='14', area='1')
        with self.assertRaises(ValueError):
            partition([row, row], [], {'decisions': []})
        with self.assertRaises(ValueError):
            partition([dict(row, gt_frame='15')], [], {'decisions': []})

    def test_unknown_values_do_not_become_bird_exclusions(self):
        row = dict(video_frame='15', gt_frame='3', gt_id='14', area='1')
        result, _ = partition([row], [dict(row, gt_id='7', reason='unexpected_label')], {'decisions': []})
        self.assertEqual(result[0]['status'], 'screened_pending_encoding')


class EncodingTests(unittest.TestCase):
    def test_palette_is_preserved_and_reported(self):
        with tempfile.TemporaryDirectory() as temp:
            p = Path(temp) / '0.png'
            im = Image.fromarray(np.array([[0, 1]], np.uint8)).convert('P')
            im.putpalette([0, 0, 0, 128, 0, 0] + [0] * 762)
            im.save(p)
            raw, gray, _, detail = load(p)
            self.assertEqual(raw.tolist(), [[0, 1]])
            self.assertEqual(gray.tolist(), [[0, 38]])
            self.assertEqual(detail['palette']['1'], [128, 0, 0])

    def test_exact_source_recipe_and_changed_gt_detection(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            gt = root / 'clip/upscaled_masks'
            orig = root / 'clip/original'
            audit = root / 'audit'
            for p in (gt, orig, audit):
                p.mkdir(parents=True)
            for index in (0, 1):
                im = Image.fromarray(np.array([[0, 75], [113, 38]], np.uint8))
                im.save(orig / f'{index}.png')
                im.resize((8, 8), Image.Resampling.BILINEAR).save(gt / f'{index}.png')
            (audit / 'verification.json').write_text(json.dumps({'gt_directory': str(gt)}))
            (audit / 'gt_hashes.json').write_text(json.dumps({str(i): sha(gt / f'{i}.png') for i in (0, 1)}))
            args = SimpleNamespace(audit=audit, workspace=root, output=root / 'result')
            result = run(args)
            self.assertEqual(result['status'], 'exact_reproduction_all_files')
            self.assertEqual(result['full_recipe_check']['counts']['exact'], 2)
            Image.new('L', (8, 8), 0).save(gt / '0.png')
            args.output = root / 'changed_result'
            with self.assertRaises(ValueError):
                run(args)


if __name__ == '__main__':
    unittest.main()
