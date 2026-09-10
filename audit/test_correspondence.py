import unittest
import numpy as np
from verify_correspondence import fingerprint,match_records


class CorrespondenceTests(unittest.TestCase):
    def test_detects_offset_change_from_omitted_images(self):
        rows=match_records({0:'a',1:'d',2:'e'},['a','b','c','d','e'])
        self.assertEqual([r['video_index'] for r in rows],[0,3,4])
        self.assertEqual([r['offset_video_minus_frames'] for r in rows],[0,2,2])

    def test_duplicates_stay_ambiguous(self):
        row=match_records({0:'a'},['a','a'])[0]
        self.assertEqual(row['status'],'ambiguous_exact')
        self.assertIsNone(row['video_index'])
        self.assertEqual(row['candidate_video_indices'],[0,1])

    def test_unmatched_is_not_assigned_nearest_index(self):
        self.assertEqual(match_records({0:'z'},['a'])[0]['status'],'no_exact_match')

    def test_fingerprint_includes_dimensions_and_pixels(self):
        a=np.zeros((2,3,3),np.uint8)
        self.assertNotEqual(fingerprint(a),fingerprint(a.reshape(3,2,3)))
        b=a.copy();b[0,0,0]=1
        self.assertNotEqual(fingerprint(a),fingerprint(b))
        self.assertEqual(fingerprint(a),fingerprint(a.copy()))


if __name__=='__main__': unittest.main()
