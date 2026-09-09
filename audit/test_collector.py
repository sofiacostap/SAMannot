import tempfile
import unittest
from pathlib import Path
from collect_evidence import csv_info


class EvidenceTests(unittest.TestCase):
    def test_reports_duplicate_keys_and_actual_frame_bounds(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'scores.csv'
            path.write_text('abs_frame,label,block_id\n750,darkest,1\n'
                            '749,darkest,0\n750,darkest,1\n', encoding='utf-8')
            info = csv_info(path)
            self.assertEqual(info['frame_range'], [749, 750])
            self.assertEqual(info['duplicate_frame_label_keys'], 1)
            self.assertEqual(info['blocks'], {'1': 2, '0': 1})
            self.assertEqual(info['rows'], 3)

    def test_empty_csv_retains_schema_without_inventing_coverage(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'iou.csv'
            path.write_text('frame_idx,sam2_label,iou\n', encoding='utf-8')
            info = csv_info(path)
            self.assertEqual(info['columns'], ['frame_idx', 'sam2_label', 'iou'])
            self.assertEqual(info['rows'], 0)
            self.assertEqual(info['frame_range'], [None, None])


if __name__ == '__main__':
    unittest.main()
