import unittest
from propagate_reviewed import direction_inputs


class DirectionTest(unittest.TestCase):
    def test_actual_prompts_cover_all_frames_once(self):
        result=[]
        for b,prompt in enumerate([449,1124,1874,2624,3324,3781]):
            later,earlier=direction_inputs(b*750,min((b+1)*750,3814)-1,prompt)
            self.assertEqual(later[1][0],earlier[1][0])
            self.assertTrue(all(a>b for a,b in zip(earlier[1],earlier[1][1:])))
            result.extend(later[1]+earlier[1][1:])
        self.assertEqual(sorted(result),list(range(3814)))
    def test_endpoint_and_single_frame(self):
        for first,last,prompt in [(0,4,0),(0,4,4),(3,3,3)]:
            passes=direction_inputs(first,last,prompt)
            self.assertEqual(sorted(passes[0][1]+passes[1][1][1:]),list(range(first,last+1)))
