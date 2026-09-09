import tempfile
import unittest
from pathlib import Path
import numpy as np
from corrected_propagation import owned_frames, canonical_direction_frames, foreground_confidence, export_result


class PropagationContractTests(unittest.TestCase):
    def test_every_frame_owned_once_including_final_frame(self):
        frames=[f for b in range(6) for f in owned_frames(3814,750,b)]
        self.assertEqual(frames,list(range(3814)))
        self.assertEqual(list(owned_frames(3814,750,5)),list(range(3750,3814)))

    def test_direction_union_has_no_duplicate_annotation(self):
        for prompt in [0,3,9]:
            forward,backward=canonical_direction_frames(10,prompt)
            self.assertEqual(sorted(forward+backward),list(range(10)))
            self.assertEqual(len(set(forward+backward)),10)

    def test_confidence_does_not_average_background(self):
        logits=np.full((100,100),-50.)
        logits[0,0]=2.
        self.assertAlmostEqual(foreground_confidence(logits),1/(1+np.exp(-2)))
        self.assertIsNone(foreground_confidence(np.full((5,5),-1.)))

    def test_export_rejects_stale_block_object_ids(self):
        with tempfile.TemporaryDirectory() as folder:
            with self.assertRaises(ValueError):
                export_result(Path(folder),0,[5,6,7,8],np.ones((4,1,10,10)))

    def test_duplicate_export_cannot_overwrite(self):
        with tempfile.TemporaryDirectory() as folder:
            output=Path(folder)
            export_result(output,3813,[1,2,3,4],np.ones((4,1,10,10)))
            with self.assertRaises(ValueError):
                export_result(output,3813,[1,2,3,4],np.ones((4,1,10,10)))


if __name__=='__main__':
    unittest.main()
