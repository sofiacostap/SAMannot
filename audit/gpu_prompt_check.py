"""GPU initial-mask check using reviewed points on verified source images.

No temporal propagation, GT mask prompts, GT cleanup or final scoring.
"""
import argparse
import csv
import inspect
import json
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
from PIL import Image
from recompute import sha, indexed, gt_labels, sam_labels, proof, write_json
from reannotate import IDS, bounds, validate_points
from verify_correspondence import fingerprint
from corrected_propagation import export_result


def validate_structure(data):
    if data.get('status')!='ready_for_integration': raise ValueError('Finish reviewing every block first')
    if data.get('gt_ids')!=IDS: raise ValueError('Unexpected GT identities')
    total=data['total_frames'];size=data['block_size']
    if total<=0 or size<=0: raise ValueError('Invalid frame bounds')
    count=(total+size-1)//size
    if set(data['blocks'])!={str(b) for b in range(count)}: raise ValueError('Missing or extra blocks')
    for b,record in data['blocks'].items():
        lo,hi=bounds(int(b),size,total);frame=record['video_frame']
        if not lo<=frame<=hi or record['local_frame']!=frame-lo:
            raise ValueError('Prompt frame outside its block or inconsistent local index')
        if record.get('human_reviewed') is not True: raise ValueError('Unreviewed block')
        if len(record['points'])!=4 or sorted(p['gt_id'] for p in record['points'])!=IDS:
            raise ValueError('Expected four distinct GT IDs per block')
        if any(p.get('positive') is not True for p in record['points']):
            raise ValueError('This experiment requires positive clicks')


def validate_inputs(args):
    data=json.loads(args.annotations.read_text());validate_structure(data)
    source=Path(data['source_directory'])
    if sha(source/'verification.json')!=data['source_verification_sha256']:
        raise ValueError('Verified source report changed')
    verified=json.loads((source/'verification.json').read_text())
    if verified['status']!='verified' or verified['count']!=data['total_frames']:
        raise ValueError('Verified source is incomplete')
    with (source/'index.csv').open(newline='') as f:
        rows={int(r['video_index']):r for r in csv.DictReader(f)}
    prior=json.loads((args.prior/'manifest.json').read_text())
    gtfiles=indexed(prior['inputs']['gt']);prepared=[]
    for block,record in sorted(data['blocks'].items(),key=lambda item:int(item[0])):
        frame=record['video_frame'];row=rows[frame]
        if int(row['gt_frames_index'])!=record['gt_frame'] or row['rgb_sha256']!=record['rgb_sha256']:
            raise ValueError('Annotation correspondence differs from verified source')
        path=source/'frames'/f'{frame:06d}.png'
        with Image.open(path) as image:rgb=np.asarray(image.convert('RGB'))
        if fingerprint(rgb)!=record['rgb_sha256']: raise ValueError(f'Image changed: {frame}')
        gtpath=gtfiles[record['gt_frame']]
        if sha(gtpath)!=record['gt_sha256']: raise ValueError(f'GT changed: {frame}')
        gt=gt_labels(gtpath)
        if gt.shape!=rgb.shape[:2]: raise ValueError('GT/image dimensions differ')
        points=validate_points(record['points'],gt)
        prepared.append((int(block),frame,path,gtpath,points))
    print('All 24 clicks and six source/GT pairs passed validation.',flush=True)
    return data,prepared


def run(args):
    data,prepared=validate_inputs(args)
    repo=Path(__file__).resolve().parent.parent
    sys.path.insert(0,str(repo))
    import torch
    import sam2
    from sam2.build_sam import build_sam2_video_predictor
    from sam2.utils.misc import load_video_frames_from_jpg_images
    if not Path(sam2.__file__).resolve().is_relative_to(repo): raise RuntimeError('Wrong SAM2 package imported')
    if not torch.cuda.is_available(): raise RuntimeError('CUDA unavailable')
    if '.png' not in inspect.getsource(load_video_frames_from_jpg_images):
        raise RuntimeError('SAM2 loader must support verified PNG input')
    if not args.checkpoint.is_file(): raise FileNotFoundError(args.checkpoint)
    output=repo/'audit'/'results'/('prompt-check-'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ'))
    output.mkdir();masks=output/'masks';masks.mkdir()
    report=dict(status='in_progress',experiment=data['experiment'],annotations_sha256=sha(args.annotations),
        annotations=str(args.annotations.resolve()),checkpoint_sha256=sha(args.checkpoint),
        config=args.config,torch=torch.__version__,gpu=torch.cuda.get_device_name(),
        predictor_sha256=sha(repo/'sam2'/'sam2_video_predictor.py'),
        source_sha256=sha(__file__),palette_to_gt={i+1:g for i,g in enumerate(IDS)},
        note='GT is used for validation and pictures only. Model receives RGB images and reviewed point prompts. No temporal propagation or final IoU evaluation.',blocks=[])
    try:
        predictor=build_sam2_video_predictor(args.config,str(args.checkpoint),device='cuda')
        for block,frame,path,gtpath,points in prepared:
            with tempfile.TemporaryDirectory(prefix='input_',dir=output) as tmp:
                with Image.open(path) as im:im.convert('RGB').save(Path(tmp)/'000000.png')
                with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):
                    state=predictor.init_state(video_path=tmp,offload_video_to_cpu=True)
                    for index,point in enumerate(points,1):
                        predictor.add_new_points_or_box(inference_state=state,frame_idx=0,obj_id=index,
                            points=np.array([[point['x'],point['y']]],dtype=np.float32),
                            labels=np.array([1],dtype=np.int32))
                    emitted=0
                    # Single-frame sequence: consolidate all prompts, with no temporal tracking.
                    for local,ids,logits in predictor.propagate_in_video(state,start_frame_idx=0):
                        if local!=0 or emitted:raise RuntimeError('Unexpected initial-mask output')
                        export_result(masks,frame,ids,logits.float().cpu().numpy());emitted+=1
                    if emitted!=1:raise RuntimeError('No initial masks produced')
                    predictor.reset_state(state);del state
                torch.cuda.empty_cache()
            sam=sam_labels(masks/f'{frame:06d}.png');gt=gt_labels(gtpath)
            for index,point in enumerate(points,1):
                proof(output/f'block{block}_GT{point["gt_id"]}.jpg',path,sam,gt,index,point['gt_id'],
                    f'Initial mask: block {block}, video {frame}, GT {point["gt_id"]}; reviewed click on both sides',
                    prompts=[point])
            report['blocks'].append(dict(block=block,video_frame=frame))
            print(f'Initial masks checked: block {block+1}/{len(prepared)}',flush=True)
        report['status']='completed_requires_visual_review'
    except Exception as exc:
        report.update(status='failed',error=str(exc));raise
    finally:
        write_json(output/'verification.json',report)
        print(f"Status: {report['status']}; results: {output}",flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--annotations',required=True,type=Path)
    parser.add_argument('--prior',required=True,type=Path)
    parser.add_argument('--checkpoint',required=True,type=Path)
    # This checkout initializes Hydra with pkg://sam2_configs (sam2/__init__.py).
    parser.add_argument('--config',default='sam2.1_hiera_l.yaml')
    run(parser.parse_args())
