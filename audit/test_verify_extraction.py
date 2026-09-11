import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np
from verify_extraction import run


class ExtractionTest(unittest.TestCase):
    def test_reports_shift_and_duplicate_without_guessing(self):
        images=[np.full((8,8,3),v,dtype=np.uint8) for v in [0,1,2,3,3,5]]
        class Capture:
            def __init__(self,*args): self.index=0
            def isOpened(self): return True
            def getBackendName(self): return 'synthetic'
            def get(self,*args): return len(images)
            def set(self,prop,index): self.index=index+1 if index else 0; return True
            def read(self):
                if self.index>=len(images): return False,None
                result=images[self.index]; self.index+=1; return True,result
            def release(self): pass
        cv=SimpleNamespace(VideoCapture=Capture,__version__='test',CAP_PROP_FRAME_COUNT=1,
                           CAP_PROP_POS_FRAMES=2,COLOR_BGR2RGB=3,cvtColor=lambda a,b:a)
        with tempfile.TemporaryDirectory() as tmp:
            prior=Path(tmp)/'prior'; prior.mkdir()
            video=prior/'video';video.write_bytes(b'test')
            for name,data in [('manifest',{}),('video_alignment',{'video':str(video)}),
                ('session',{'best':{'0':0,'1':2},'block_size':2,'prompts':[]})]:
                (prior/(name+'.json')).write_text(json.dumps(data))
            with patch.dict('sys.modules',{'cv2':cv}): run(prior)
            output=next(Path(tmp).glob('extraction-*'))
            report=json.loads((output/'verification.json').read_text())
            self.assertEqual(report['status'],'completed')
            self.assertEqual(report['blocks'][0]['annotation'][0]['actual_video_index'],0)
            self.assertEqual(report['blocks'][1]['annotation'][0]['status'],'ambiguous')
            self.assertEqual(report['blocks'][1]['offset_counts']['1'],1)
            self.assertTrue((output/'clicks_block1.jpg').exists())
