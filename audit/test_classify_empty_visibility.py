import unittest
from classify_empty_visibility import visibility_category


class VisibilityTest(unittest.TestCase):
    def test_missing_gt_does_not_become_absence(self):
        self.assertEqual(visibility_category(None),'gt_unavailable')
    def test_absent_and_present_are_different_candidates(self):
        self.assertEqual(visibility_category(0),'gt_absent_candidate_expected_absence')
        self.assertEqual(visibility_category(1),'gt_present_candidate_tracking_loss')
        self.assertEqual(visibility_category(20000),'gt_present_candidate_tracking_loss')
