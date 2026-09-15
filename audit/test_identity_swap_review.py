import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
from PIL import Image

from identity_swap_review import pair_candidates,group_events,approved_mapping,apply_review,draw_preview,IDS
from evaluate_task4 import sha,write_csv,write_json,read_csv
from verify_correspondence import fingerprint


class CandidateTests(unittest.TestCase):
    def test_swap_requires_both_cross_matches(self):
        m={(a,b):(.85 if a!=b else .05) for a in (14,38) for b in (14,38)}
        self.assertEqual(len(pair_candidates(m,[14,38])),1)
        m[14,14]=.95
        self.assertEqual(pair_candidates(m,[14,38]),[])

    def test_correct_identity_and_ambiguous_multiple_pairs_not_suggested(self):
        ids=[14,38,75]
        m={(a,b):(.9 if a==b else .1) for a in ids for b in ids}
        self.assertEqual(pair_candidates(m,ids),[])
        m={(a,b):(.1 if a==b else .8) for a in ids for b in ids}
        self.assertEqual(pair_candidates(m,ids),[])

    def test_untrusted_identity_is_not_used(self):
        m={(a,b):(.8 if a!=b else 0) for a in (14,38) for b in (14,38)}
        self.assertEqual(pair_candidates(m,[14]),[])

    def test_events_do_not_bridge_gaps_or_blocks(self):
        r=[dict(video_frame=f,block_id=f//750,a=14,b=38,gain=.7) for f in [100,101,102,104,105,106,749,750]]
        events=group_events(r,3)
        self.assertEqual({(e['start'],e['end']) for e in events},{(100,102),(104,106)})
        events=group_events(r,1)
        self.assertIn((749,749),{(e['start'],e['end']) for e in events})
        self.assertIn((750,750),{(e['start'],e['end']) for e in events})

    def test_approval_bounds_conflicts_and_prompt_protection(self):
        e=dict(event_id='SWAP0001',a=14,b=38,start=100,end=102)
        mapping=approved_mapping([e],['SWAP0001'],1500,750,[374,1124])
        self.assertEqual(len(mapping),6)
        self.assertEqual(mapping[100,14],38)
        self.assertNotIn((99,14),mapping)
        for events,approval in [([e],[]),([e],['wrong']),([e,e],['SWAP0001','SWAP0001']),
                ([dict(e,start=749,end=751)],['SWAP0001']),([dict(e,start=374,end=375)],['SWAP0001']),
                ([e,dict(e,event_id='SWAP0002')],['SWAP0001','SWAP0002'])]:
            with self.assertRaises(ValueError):approved_mapping(events,approval,1500,750,[374,1124])


class ApplyTests(unittest.TestCase):
    def test_overlay_preserves_baseline_and_moves_confidence_with_mask(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);pred=root/'prediction';review=root/'review';out=root/'output'
            for p in (pred,review,out):p.mkdir()
            ann=dict(total_frames=3,block_size=3,blocks={'0':{'video_frame':0}})
            write_json(pred/'annotations.json',ann)
            records=[];confidence=[];original={}
            for f in range(3):
                for i,g in enumerate(IDS):
                    path=pred/'masks_by_gt_id'/str(g)/f'{f:06d}.png';path.parent.mkdir(parents=True,exist_ok=True)
                    arr=np.zeros((10,10),np.uint8);arr[i+f,i]=255;Image.fromarray(arr).save(path)
                    original[f,g]=path.read_bytes()
                    records.append(dict(video_frame=f,gt_id=g,sha256=sha(path)))
                    confidence.append(dict(abs_frame=f,gt_id=g,sam2_label=i+1,block_id=0,
                                           direction='later',foreground_logit_score=.8+.04*i))
            write_csv(pred/'foreground_confidence.csv',confidence)
            write_csv(review/'baseline_prediction_hashes.csv',records)
            events=[dict(event_id='SWAP0001',a=14,b=38,start=1,end=1)]
            write_json(review/'events.json',events)
            source=root/'source';(source/'frames').mkdir(parents=True)
            ref=root/'reference';(ref/'masks').mkdir(parents=True)
            photo=np.full((10,10,3),100,np.uint8)
            Image.fromarray(photo).save(source/'frames/000001.png')
            gt=np.zeros((10,10),np.uint8);gt[1,0]=14;gt[2,1]=38
            Image.fromarray(gt).save(ref/'masks/000001.png')
            refs={(1,g):dict(status='trusted_screened',gt_frame='1',normalized_mask='masks/000001.png') for g in (14,38)}
            images={1:dict(file='000001.png',rgb_sha256=fingerprint(photo),gt_frames_index='1')}
            gt_hashes={'1':dict(normalized=sha(ref/'masks/000001.png'))}
            lookup={(r['video_frame'],r['gt_id']):r['sha256'] for r in records}
            (review/'previews').mkdir()
            preview=draw_preview(review,dict(source_directory=str(source)),pred,ref,lookup,events[0],1,refs,images,gt_hashes)
            self.assertTrue((review/preview).is_file())
            with Image.open(review/preview) as im:self.assertEqual(im.width,2400)
            write_json(review/'preview_index.json',{'SWAP0001':[dict(frame=1,file=preview)]})
            write_json(review/'verification.json',dict(status='completed_suggestions',prediction_directory=str(pred),reference_directory=str(root/'reference'),
                input_hashes={str(pred/'annotations.json'):sha(pred/'annotations.json',True)},
                events_sha256=sha(review/'events.json',True),baseline_hashes_sha256=sha(review/'baseline_prediction_hashes.csv',True)))
            args=SimpleNamespace(review=review,approve=['SWAP0001'],reviewer='Synthetic test reviewer')
            with patch('identity_swap_review.new_output',return_value=out):apply_review(args)
            for (f,g),data in original.items():
                self.assertEqual((pred/'masks_by_gt_id'/str(g)/f'{f:06d}.png').read_bytes(),data)
            self.assertEqual((out/'masks_by_gt_id/14/000001.png').read_bytes(),original[1,38])
            self.assertEqual((out/'masks_by_gt_id/38/000001.png').read_bytes(),original[1,14])
            self.assertEqual(len(list((out/'masks_by_gt_id').rglob('*.png'))),2)
            fixed=read_csv(out/'foreground_confidence.csv')
            r=next(r for r in fixed if r['abs_frame']=='1' and r['gt_id']=='14')
            self.assertAlmostEqual(float(r['foreground_logit_score']),.84)
            self.assertEqual(r['sam2_label'],'1')
            self.assertEqual(len(read_csv(out/'mask_sources.csv')),12)
            self.assertEqual(json.loads((out/'verification.json').read_text())['status'],'completed_identity_overlay')


if __name__=='__main__':unittest.main()
