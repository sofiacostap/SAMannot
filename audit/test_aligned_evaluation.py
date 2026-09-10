import argparse
import csv
import json
import pickle
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np
from PIL import Image
from recompute import run, color, sha, load_correspondence
from verify_correspondence import fingerprint


class AlignedEvaluationTests(unittest.TestCase):
    def test_evaluates_mapped_frames_and_reports_export_gap(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary)
            for name in ['sam','gt','frames','correspondence']:
                (root/name).mkdir()
            birds=[14,38,75,113];names=['darkest','brownish','orange','spotted']
            labels=[]
            for index,name in enumerate(names,1):
                labels.append(SimpleNamespace(name=name,pts={0:[SimpleNamespace(idx=0,x=index*15+3,y=13,pt_type=1)]},boxes={}))
            table=[]
            for gt_index,video_index in [(0,0),(1,2),(2,3)]:
                gt=np.zeros((120,120),np.uint8);pred=np.zeros((120,120,3),np.uint8)
                for index,bird in enumerate(birds,1):
                    y=10 if video_index==0 else 50
                    gt[y:y+8,index*15:index*15+8]=bird
                    pred[y:y+8,index*15:index*15+8]=color(index)
                raw=np.full((120,120,3),gt_index,np.uint8)
                Image.fromarray(raw).save(root/'frames'/f'{gt_index:06d}.png')
                Image.fromarray(gt).save(root/'gt'/f'{gt_index:06d}.png')
                if video_index!=3:
                    Image.fromarray(pred).save(root/'sam'/f'{video_index:06d}.png')
                table.append(dict(frames_index=gt_index,video_index=video_index,status='unique_exact',
                                  candidate_video_indices=[video_index],rgb_sha256=fingerprint(raw)))
            Image.new('RGB',(120,120),'black').save(root/'sam'/'000001.png')
            session=root/'session.pkl'
            session.write_bytes(pickle.dumps(dict(sam_handler_labels=labels,block_size=4,num_blocks=1)))
            quality=root/'quality.csv';quality.write_text('abs_frame,label,logit_score\n0,darkest,0.1\n')
            video=root/'video.mp4';video.write_bytes(b'video fixture is not decoded in this test')
            corr=root/'correspondence'
            (corr/'verification.json').write_text(json.dumps(dict(status='verified_image_correspondence',video_file_sha256=sha(video))))
            (corr/'frame_mapping.json').write_text(json.dumps(table))
            args=argparse.Namespace(workspace=root,session=session,masks=root/'sam',gt=root/'gt',frames=root/'frames',
                quality=quality,video=video,correspondence=corr,output=root/'output',gt_ids=birds)
            with patch('recompute.check_video',return_value={'status':'mocked_in_test'}):
                run(args)
            with (root/'output'/'recomputed.csv').open() as stream:
                rows=list(csv.DictReader(stream))
            aligned=[r for r in rows if r['video_frame_idx']=='2']
            self.assertEqual(len(rows),16)
            self.assertEqual({r['gt_frame_idx'] for r in aligned},{'1'})
            self.assertTrue(all(float(r['iou'])==1 for r in aligned))
            self.assertTrue(all(r['distance_from_prompt']=='2' for r in aligned))
            gaps=[r for r in rows if r['video_frame_idx']=='3']
            self.assertTrue(all(r['status']=='prediction_file_missing' and r['iou']=='' for r in gaps))
            self.assertEqual(len(gaps),4)
            unreferenced=[r for r in rows if r['video_frame_idx']=='1']
            self.assertTrue(all(r['status']=='gt_file_missing' and r['gt_frame_idx']=='' for r in unreferenced))
            summary=json.loads((root/'output'/'summary.json').read_text())
            self.assertEqual(summary['all_frames']['evaluated'],8)
            self.assertEqual(summary['all_frames']['mean_iou'],1.)

    def test_rejects_stale_image_fingerprint(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary);image=root/'0.png';Image.new('RGB',(2,2),'black').save(image)
            (root/'verification.json').write_text(json.dumps({'status':'verified_image_correspondence'}))
            (root/'frame_mapping.json').write_text(json.dumps([dict(frames_index=0,video_index=12,
                status='unique_exact',candidate_video_indices=[12],rgb_sha256='wrong')]))
            with self.assertRaisesRegex(ValueError,'image changed'):
                load_correspondence(root,{0:image})

    def test_rejects_many_frames_to_one_video_index(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary)
            (root/'verification.json').write_text(json.dumps({'status':'verified_image_correspondence'}))
            (root/'frame_mapping.json').write_text(json.dumps([dict(frames_index=f,video_index=12,
                status='unique_exact',candidate_video_indices=[12],rgb_sha256='unused') for f in [0,1]]))
            with self.assertRaisesRegex(ValueError,'Duplicate'):
                load_correspondence(root,{0:root/'0.png',1:root/'1.png'})


if __name__=='__main__': unittest.main()
