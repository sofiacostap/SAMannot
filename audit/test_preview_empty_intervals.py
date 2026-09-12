import tempfile
import unittest
from pathlib import Path
import numpy as np
from PIL import Image
from preview_empty_intervals import intervals, sample_frames, binary_mask


class EmptyPreviewTest(unittest.TestCase):
    def test_intervals_keep_recovery_gaps(self):
        self.assertEqual(intervals([3,1,2,6,6,8]),[[1,3],[6,6],[8,8]])
        self.assertEqual(intervals([]),[])
    def test_samples_stay_in_block(self):
        self.assertEqual(sample_frames(750,760,750,1499,5),[750,755,760,765])
        self.assertEqual(sample_frames(3813,3813,3750,3813,5),[3808,3813])
    def test_empty_distinguished_from_missing_or_invalid_mask(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'mask.png'
            with self.assertRaises(FileNotFoundError):binary_mask(path)
            Image.fromarray(np.zeros((3,3),np.uint8)).save(path)
            self.assertFalse(binary_mask(path).any())
            Image.fromarray(np.ones((3,3),np.uint8)).save(path)
            with self.assertRaises(ValueError):binary_mask(path)
