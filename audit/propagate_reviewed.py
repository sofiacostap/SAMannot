"""Propagate reviewed GT-assisted points on verified images using the GPU.

Each direction has a separate predictor state and an explicit original-index
map. Earlier frames are supplied in reverse chronological order to the normal
forward predictor. This avoids the local predictor's forward-only memory
eviction rule when processing an earlier-frame pass. It is recorded as a new
propagation method, not assumed identical to native reverse=True execution.
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
from gpu_prompt_check import validate_inputs
from reannotate import IDS, bounds
from recompute import sha, write_json, sam_labels, gt_labels, proof
from verify_correspondence import fingerprint
from corrected_propagation import export_result


def direction_inputs(first,last,prompt):
    if not first<=prompt<=last: raise ValueError('Prompt outside block')
    return [('later',list(range(prompt,last+1))),('earlier',list(range(prompt,first-1,-1)))]


def run(args):
    data,prepared=validate_inputs(args)
    repo=Path(__file__).resolve().parent.parent
    source=Path(data['source_directory'])
    with (source/'index.csv').open(newline='') as f:
        hashes={int(r['video_index']):r['rgb_sha256'] for r in csv.DictReader(f)}
    if sorted(hashes)!=list(range(data['total_frames'])): raise ValueError('Incomplete image index')
    sys.path.insert(0,str(repo))
    import torch
    import sam2
    from sam2.build_sam import build_sam2_video_predictor
    from sam2.utils.misc import load_video_frames_from_jpg_images
    if not Path(sam2.__file__).resolve().is_relative_to(repo): raise RuntimeError('Wrong SAM2 import')
    if not torch.cuda.is_available(): raise RuntimeError('CUDA unavailable')
    if '.png' not in inspect.getsource(load_video_frames_from_jpg_images): raise RuntimeError('PNG input required')
    if not args.checkpoint.is_file(): raise FileNotFoundError(args.checkpoint)
    output=repo/'audit'/'results'/('propagation-'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ'))
    output.mkdir();masks=output/'masks';masks.mkdir()
    for bird in IDS:(output/'masks_by_gt_id'/str(bird)).mkdir(parents=True)
    report=dict(status='in_progress',experiment=data['experiment'],annotations_sha256=sha(args.annotations),
        source_directory=str(source),checkpoint_sha256=sha(args.checkpoint),config=args.config,
        source_sha256=sha(__file__),predictor_sha256=sha(repo/'sam2'/'sam2_video_predictor.py'),
        torch=torch.__version__,gpu=torch.cuda.get_device_name(),window=args.window,
        direction_method='Separate states. Later: ascending source indices. Earlier: descending source indices. Both call forward predictor starting at local 0.',
        palette_to_gt={i+1:g for i,g in enumerate(IDS)},
        mask_policy='Independent per-ID binary masks are authoritative; combined RGB PNG uses higher palette index for overlaps.',
        confidence_definition='Mean sigmoid of positive foreground logits. Empty foreground is unavailable. Not calibrated probability of correctness.',
        frames_written=0,completed_blocks=[],note='No GT supplied to predictor. No final IoU or error-rate calculation.')
    write_json(output/'annotations.json',data)
    write_json(output/'verification.json',report)
    seen=set()
    try:
        predictor=build_sam2_video_predictor(args.config,str(args.checkpoint),device='cuda',
            hydra_overrides_extra=[f'++model.memory_window_size={args.window}'])
        if predictor.memory_window_size!=args.window: raise RuntimeError('Memory window override failed')
        with (output/'foreground_confidence.csv').open('w',newline='') as stream:
            writer=csv.DictWriter(stream,fieldnames=['abs_frame','sam2_label','gt_id','block_id','direction','foreground_logit_score'])
            writer.writeheader()
            for block,prompt,rawpath,gtpath,points in prepared:
                first,last=bounds(block,data['block_size'],data['total_frames'])
                for direction,indices in direction_inputs(first,last,prompt):
                    if direction=='earlier' and len(indices)==1:continue
                    print(f'Block {block}: preparing {direction} pass ({len(indices)} images)',flush=True)
                    with tempfile.TemporaryDirectory(prefix=f'block{block}_{direction}_',dir=output) as tmp:
                        for local,absolute in enumerate(indices):
                            with Image.open(source/'frames'/f'{absolute:06d}.png') as image:
                                rgb=np.asarray(image.convert('RGB'))
                            if fingerprint(rgb)!=hashes[absolute]: raise ValueError(f'Image changed: {absolute}')
                            Image.fromarray(rgb).save(Path(tmp)/f'{local:06d}.png',compress_level=1)
                        with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):
                            state=predictor.init_state(video_path=tmp,offload_video_to_cpu=True,offload_state_to_cpu=True)
                            for index,point in enumerate(points,1):
                                predictor.add_new_points_or_box(inference_state=state,frame_idx=0,obj_id=index,
                                    points=np.array([[point['x'],point['y']]],dtype=np.float32),labels=np.array([1],dtype=np.int32))
                            local_seen=set()
                            for local,ids,logits in predictor.propagate_in_video(state,start_frame_idx=0,reverse=False):
                                if not 0<=local<len(indices) or local in local_seen:raise RuntimeError('Invalid/duplicate local index')
                                local_seen.add(local)
                                if direction=='earlier' and local==0:continue
                                absolute=indices[local]
                                if absolute in seen:raise RuntimeError('Duplicate absolute frame')
                                arrays=logits.float().cpu().numpy()
                                scores=export_result(masks,absolute,ids,arrays)
                                for j,index in enumerate(ids):
                                    binary=(np.asarray(arrays[j]).squeeze()>0).astype(np.uint8)*255
                                    Image.fromarray(binary).save(output/'masks_by_gt_id'/str(IDS[int(index)-1])/f'{absolute:06d}.png')
                                for score in scores:
                                    writer.writerow(dict(score,gt_id=IDS[score['sam2_label']-1],block_id=block,direction=direction))
                                seen.add(absolute)
                                if len(seen)%100==0:
                                    stream.flush();print(f'Exported {len(seen)}/{data["total_frames"]} frames',flush=True)
                            if len(local_seen)!=len(indices):raise RuntimeError('Incomplete directional output')
                            predictor.reset_state(state);del state
                        torch.cuda.empty_cache()
                sam=sam_labels(masks/f'{prompt:06d}.png');gt=gt_labels(gtpath)
                for index,point in enumerate(points,1):
                    proof(output/f'block{block}_GT{point["gt_id"]}.jpg',rawpath,sam,gt,index,point['gt_id'],
                        f'Propagation start: block {block}, video {prompt}, GT {point["gt_id"]}',prompts=[point])
                stream.flush();report['completed_blocks'].append(block)
                report['frames_written']=len(seen);write_json(output/'verification.json',report)
                print(f'Completed block {block+1}/{len(prepared)}',flush=True)
        if seen!=set(range(data['total_frames'])):raise RuntimeError('Incomplete total coverage')
        report['status']='completed'
    except Exception as exc:
        report.update(status='failed',error=str(exc));raise
    finally:
        report['frames_written']=len(seen)
        write_json(output/'verification.json',report)
        print(f"Status: {report['status']}; results: {output}",flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--annotations',required=True,type=Path)
    parser.add_argument('--prior',required=True,type=Path)
    parser.add_argument('--checkpoint',required=True,type=Path)
    parser.add_argument('--config',default='sam2.1_hiera_l.yaml')
    parser.add_argument('--window',type=int,default=16)
    run(parser.parse_args())
