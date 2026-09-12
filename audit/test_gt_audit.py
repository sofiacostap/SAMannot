import unittest
import numpy as np
from gt_audit import component_flags,pair_swap_candidates


class GTAuditTest(unittest.TestCase):
    def test_absence_is_not_artifact_and_fragments_not_removed(self):
        self.assertEqual(component_flags(0,[]),[])
        self.assertIn('small_disconnected_regions',component_flags(1001,[1000,1]))
        self.assertIn('multiple_substantial_regions',component_flags(1500,[1000,500]))
    def test_pair_exchange_detected_but_static_masks_not_flagged(self):
        a=np.zeros((30,30),np.uint8);a[:10,:10]=14;a[20:,20:]=38
        self.assertEqual(pair_swap_candidates(a,a),[])
        b=a.copy();b[a==14]=38;b[a==38]=14
        self.assertEqual(pair_swap_candidates(a,b),[(14,38)])
