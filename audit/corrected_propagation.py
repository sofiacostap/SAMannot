"""Future GPU run: clean per-block SAM2 states, exact FRAMES inputs, no overlap.

Requires explicit model config/checkpoint. This is not run by run_lumen.sh.
Numerical quality/IoU scoring is a separate chronological CPU pass.
"""
import argparse
import csv
import inspect
import json
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image
from recompute import indexed, load_session, color, write_json, sha


def owned_frames(total, size, block):
    if total < 0 or size <= 0 or block < 0:
        raise ValueError('Invalid frame bounds')
    return range(block*size, min((block+1)*size,total))


def canonical_direction_frames(frame_count, prompt):
    if not 0 <= prompt < frame_count:
        raise ValueError('Prompt outside owned block')
    return list(range(prompt,frame_count)), list(range(prompt-1,-1,-1))


def foreground_confidence(logits):
    """Mean sigmoid over positive-logit pixels; absent foreground -> unavailable."""
    foreground = logits[logits > 0]
    if foreground.size == 0:
        return None
    return float(np.mean(1/(1+np.exp(-foreground))))


def export_result(folder, absolute, obj_ids, logits):
    if len(set(map(int,obj_ids))) != len(obj_ids) or set(map(int,obj_ids)) != {1,2,3,4}:
        raise ValueError(f'Unexpected predictor IDs: {obj_ids}')
    arrays = {int(index):np.asarray(logits[j]).squeeze() for j,index in enumerate(obj_ids)}
    shape = arrays[1].shape
    if len(shape) != 2 or any(a.shape != shape for a in arrays.values()):
        raise ValueError('Unexpected prediction dimensions')
    combined = np.zeros((*shape,3),np.uint8)
    scores=[]
    for index in sorted(arrays):
        mask=arrays[index]>0
        combined[mask]=color(index)
        scores.append(dict(abs_frame=absolute,sam2_label=index,
                           foreground_logit_score=foreground_confidence(arrays[index])))
    path=folder/f'{absolute:06d}.png'
    if path.exists():
        raise ValueError(f'Duplicate output frame {absolute}')
    Image.fromarray(combined).save(path)
    return scores


def run(args):
    # Imports happen only for an explicitly requested GPU run.
    sys.path.insert(0,str(args.repo.resolve()))
    import torch
    from sam2.build_sam import build_sam2_video_predictor
    from sam2.utils.misc import load_video_frames_from_jpg_images
    if '.png' not in inspect.getsource(load_video_frames_from_jpg_images):
        raise RuntimeError('Installed SAM2 loader lacks PNG support; refusing lossy frame conversion')
    if not torch.cuda.is_available():
        raise RuntimeError('GPU required for this propagation command')
    session=load_session(args.session)
    frames=indexed(args.frames)
    if sorted(frames) != list(range(len(frames))):
        raise ValueError('FRAMES must be contiguous and zero-based; reconcile numbering first')
    if session['num_blocks'] != (len(frames)+session['block_size']-1)//session['block_size']:
        raise ValueError('Session block count and FRAMES length differ')
    output=args.output.resolve()
    if not output.is_relative_to(args.workspace.resolve()):
        raise ValueError('Output must remain inside SAMannot workspace')
    output.mkdir(parents=True,exist_ok=False)
    masks=output/'masks'; masks.mkdir()
    manifest=dict(status='in_progress',session_sha256=sha(args.session),checkpoint_sha256=sha(args.checkpoint),
                  config=args.config,window=args.window,source='FRAMES lossless PNG copy',
                  label_indices={i:name for i,name in enumerate(session['names'],1)},
                  overlap_policy='higher palette index wins in combined PNG',
                  confidence_definition='mean sigmoid over positive-logit pixels; empty foreground unavailable')
    try:
        predictor=build_sam2_video_predictor(args.config,str(args.checkpoint),device='cuda',
                    hydra_overrides_extra=[f'++model.memory_window_size={args.window}'])
        if getattr(predictor,'memory_window_size',None) != args.window:
            raise RuntimeError('Requested memory window did not reach predictor')
        scores=[]
        for block in range(session['num_blocks']):
            owned=list(owned_frames(len(frames),session['block_size'],block))
            prompt=session['best'][block]-owned[0]
            canonical_direction_frames(len(owned),prompt)
            with tempfile.TemporaryDirectory(prefix='block_',dir=output) as temporary:
                for local,absolute in enumerate(owned):
                    with Image.open(frames[absolute]) as image:
                        image.convert('RGB').save(Path(temporary)/f'{local:06d}.png')
                with torch.inference_mode(), torch.autocast('cuda',dtype=torch.bfloat16):
                    state=predictor.init_state(video_path=temporary)
                    for reverse in [False,True]:
                        if reverse and prompt==0:
                            continue
                        predictor.reset_state(state)
                        for index in range(1,5):
                            pts=[p for p in session['prompts'] if p['block_id']==block and p['sam2_label']==index]
                            if not pts:
                                raise ValueError(f'Missing prompts for block {block} label {index}')
                            predictor.add_new_points_or_box(inference_state=state,frame_idx=prompt,obj_id=index,
                                points=np.asarray([[p['x'],p['y']] for p in pts],np.float32),
                                labels=np.asarray([int(p['positive']) for p in pts],np.int32))
                        for local,ids,logits in predictor.propagate_in_video(state,start_frame_idx=prompt,reverse=reverse):
                            if reverse and local==prompt:
                                continue
                            if not 0 <= local < len(owned):
                                raise RuntimeError('Predictor yielded a frame outside its block')
                            scores.extend(export_result(masks,owned[local],ids,logits.float().cpu().numpy()))
                    predictor.reset_state(state)
                    del state
                torch.cuda.empty_cache()
            print(f'Completed block {block+1}/{session["num_blocks"]}',flush=True)
        if len(list(masks.glob('*.png'))) != len(frames):
            raise RuntimeError('Incomplete exported frame coverage')
        with (output/'foreground_confidence.csv').open('w',newline='') as stream:
            writer=csv.DictWriter(stream,fieldnames=['abs_frame','sam2_label','foreground_logit_score'])
            writer.writeheader(); writer.writerows(sorted(scores,key=lambda r:(r['abs_frame'],r['sam2_label'])))
        manifest.update(status='completed',frames=len(frames))
    except Exception as exc:
        manifest.update(status='failed',error=str(exc)); raise
    finally:
        write_json(output/'propagation_manifest.json',manifest)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ['repo','workspace','session','frames','checkpoint','output']:
        parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--config',required=True)
    parser.add_argument('--window',type=int,default=16)
    run(parser.parse_args())
