import argparse
import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np
from PIL import Image
from check_alignment import run


class AlignmentTests(unittest.TestCase):
    def test_recovers_known_one_frame_offset_without_modifying_input(self):
        images=[np.full((20,30,3),value,np.uint8) for value in [10,40,90,150]]
        class Capture:
            def __init__(self,*args): self.position=0
            def isOpened(self): return True
            def get(self,prop): return len(images)
            def set(self,prop,value): self.position=value
            def read(self):
                if self.position>=len(images): return False,None
                value=images[self.position].copy(); self.position+=1
                return True,value
            def release(self): pass
        fake=SimpleNamespace(VideoCapture=Capture,CAP_PROP_FRAME_COUNT=1,CAP_PROP_POS_FRAMES=2,
                             COLOR_BGR2RGB=3,cvtColor=lambda arr,mode:arr)
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary); prior=root/'run'; prior.mkdir(); frames=root/'frames'; frames.mkdir()
            for i in range(2): Image.fromarray(images[i+1]).save(frames/f'{i:06d}.png')
            (prior/'manifest.json').write_text(json.dumps({'inputs':{'frames':str(frames)}}))
            (prior/'session.json').write_text(json.dumps({'best':{'0':0}}))
            (prior/'video_alignment.json').write_text(json.dumps({'video':'synthetic'}))
            before=(frames/'000000.png').read_bytes()
            with patch.dict(sys.modules,{'cv2':fake}):
                run(argparse.Namespace(results=prior,radius=2))
            report=json.loads((root/'run-alignment'/'alignment_search.json').read_text())
            self.assertEqual([r['best']['offset_video_minus_frames'] for r in report['samples']],[1,1])
            self.assertEqual([r['exact_video_matches'] for r in report['samples']],[[1],[2]])
            self.assertEqual((frames/'000000.png').read_bytes(),before)


if __name__=='__main__': unittest.main()
