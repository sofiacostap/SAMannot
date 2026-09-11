import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np
from prepare_verified_frames import run
from recompute import sha
from verify_correspondence import fingerprint


class PrepareTest(unittest.TestCase):
    def test_preserves_frame_without_gt_and_rejects_changed_pixels(self):
        images=[np.full((8,8,3),v,dtype=np.uint8) for v in [0,1,2]]
        class Capture:
            def __init__(self,*args): self.index=0
            def isOpened(self): return True
            def read(self):
                if self.index==len(images): return False,None
                result=images[self.index];self.index+=1;return True,result
            def release(self): pass
        cv=SimpleNamespace(VideoCapture=Capture,__version__='test',COLOR_BGR2RGB=1,cvtColor=lambda a,b:a)
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);prior=root/'prior';prior.mkdir();corr=root/'corr';corr.mkdir()
            video=root/'video';video.write_bytes(b'test')
            (prior/'session.json').write_text(json.dumps({'best':{'0':0},'prompts':[]}))
            (corr/'verification.json').write_text(json.dumps(dict(status='verified_image_correspondence',
                video=str(video),video_file_sha256=sha(video),video_decoded_count=3)))
            mapping=[dict(video_index=i,frames_index=j,status='unique_exact',rgb_sha256=fingerprint(images[i])) for j,i in enumerate([0,2])]
            (corr/'frame_mapping.json').write_text(json.dumps(mapping))
            with patch.dict('sys.modules',{'cv2':cv}): run(prior,corr)
            output=next(root.glob('verified-source-*'))
            report=json.loads((output/'verification.json').read_text())
            self.assertEqual(report['status'],'verified')
            self.assertEqual(report['without_gt'],[1])
            self.assertEqual(len(list((output/'frames').glob('*.png'))),3)
            mapping[0]['rgb_sha256']='invalid'
            (corr/'frame_mapping.json').write_text(json.dumps(mapping))
            with patch.dict('sys.modules',{'cv2':cv}),self.assertRaisesRegex(ValueError,'differs'):
                run(prior,corr)
