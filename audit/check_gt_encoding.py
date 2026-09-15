"""Read-only, bounded GT encoding provenance check. Never runs SAM2.

Inspect native PNG mode/palette, nearby original-mask folders and local generator
code. Compare explicit resize/conversion recipes against sampled upscaled files.
Exact sample matches are evidence, not proof of the historical command used.
"""
import argparse
import hashlib
import json
import os
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from PIL import Image


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def inventory(root, depth=3):
    """Bound traversal to one clip directory, never the full external drive."""
    result = []
    for base, dirs, files in os.walk(root):
        p = Path(base)
        dirs[:] = [d for d in dirs if d != '@eaDir' and not d.startswith('.')]
        if len(p.relative_to(root).parts) >= depth:
            dirs[:] = []
        images = [p / f for f in files if Path(f).suffix.lower() in ('.png', '.tif', '.tiff')]
        numeric = {}
        for f in images:
            if f.stem.isdigit():
                if int(f.stem) in numeric:
                    raise ValueError(f'Duplicate mask index in {p}: {f.stem}')
                numeric[int(f.stem)] = f
        if numeric:
            result.append((p, numeric))
    return result


def load(path):
    with Image.open(path) as im:
        arr = np.array(im)
        palette = im.getpalette()
        detail = dict(path=str(path), sha256=sha(path), mode=im.mode, shape=list(arr.shape))
        if arr.ndim == 2:
            vals, counts = np.unique(arr, return_counts=True)
            detail['values'] = {str(int(v)): int(c) for v, c in zip(vals, counts)}
            if palette:
                detail['palette'] = {str(int(v)): palette[3*int(v):3*int(v)+3] for v in vals}
        else:
            detail['channels_identical'] = bool(np.all(arr[..., :1] == arr[..., :3]))
        return arr, np.array(im.convert('L')), np.array(im.convert('RGB')), detail


def recipes(path, size):
    try:
        import cv2
    except ImportError:
        cv2 = None
    raw, gray, rgb, _ = load(path)
    sources = {'native': raw, 'pillow_gray': gray, 'rgb': rgb}
    if cv2:
        sources['cv2_rgb_gray'] = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        sources['cv2_bgr_gray'] = cv2.cvtColor(rgb, cv2.COLOR_BGR2GRAY)
    for name, src in sources.items():
        methods = [('nearest', cv2.INTER_NEAREST), ('linear', cv2.INTER_LINEAR),
                   ('cubic', cv2.INTER_CUBIC), ('area', cv2.INTER_AREA)] if cv2 else []
        for method, flag in methods:
            resized = cv2.resize(src, size, interpolation=flag)
            yield name + '/cv2_' + method, resized
            if resized.ndim == 3 and resized.shape[2] == 3:
                yield name + '/cv2_' + method + '/rgb_to_gray', cv2.cvtColor(resized, cv2.COLOR_RGB2GRAY)
                yield name + '/cv2_' + method + '/bgr_to_gray', cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
        if src.dtype == np.uint8:
            for method, flag in [('nearest', Image.Resampling.NEAREST), ('bilinear', Image.Resampling.BILINEAR)]:
                yield name + '/pillow_' + method, np.array(Image.fromarray(src).resize(size, flag))


def code_evidence(workspace):
    findings = []
    skipped = {'.git', 'audit', 'sam2', 'checkpoints', 'node_modules', '__pycache__',
               'frames', 'data', 'export', 'temp_dir', 'sam_temp_dir'}
    for base, dirs, files in os.walk(workspace):
        rel = Path(base).relative_to(workspace)
        dirs[:] = [d for d in dirs if d.lower() not in skipped and
                   not d.startswith(('headless_', 'propagation-', 'sam_temp', '.'))]
        if len(rel.parts) >= 4:
            dirs[:] = []
        for name in files:
            p = Path(base) / name
            if p.suffix not in ('.py', '.sh', '.ipynb') or p.stat().st_size > 2_000_000:
                continue
            content = p.read_text(errors='replace')
            if 'upscaled_masks' not in content and not ('upscal' in content.lower() and 'mask' in content.lower()):
                continue
            lines = content.splitlines()
            hits = set()
            for i, line in enumerate(lines):
                if any(s in line.lower() for s in ('upscal', 'resize', 'inter_', 'imwrite', 'convert(', 'cvtcolor')):
                    hits.update(range(max(0, i-2), min(len(lines), i+3)))
            findings.append(dict(path=str(p), sha256=sha(p),
                                 snippets=[f'{i+1}: {lines[i][:500]}' for i in sorted(hits)][:100]))
    return findings


def run(args):
    report = json.loads((args.audit / 'verification.json').read_text())
    gt = Path(report['gt_directory']).resolve()
    if not gt.is_dir():
        raise FileNotFoundError(gt)
    output = args.output or Path(__file__).resolve().parent / 'results' / ('gt-encoding-' + datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ'))
    output.mkdir()
    folders = inventory(gt.parent)
    original = getattr(args, 'original', None)
    if original is not None:
        original = original.resolve()
        if not original.is_dir():
            raise FileNotFoundError(original)
        extra = inventory(original, depth=0)
        if not extra:
            raise ValueError(f'No numerically indexed masks in {original}')
        folders.extend((p, files) for p, files in extra
                       if p.resolve() not in {q.resolve() for q, _ in folders})
    target = next((files for p, files in folders if p.resolve() == gt), None)
    if not target:
        raise ValueError('No numerically indexed target masks')
    indices = sorted(set([0, 1, 2, 3, 651, 1739, 437, 1112, 1862, 2612, 3312, 3769, 3801]) & set(target))
    if not indices:
        raise ValueError('Expected sample frame indices missing')
    if original is not None:
        original_files = next(files for p, files in folders if p.resolve() == original)
        missing = sorted(set(indices) - set(original_files))
        if missing:
            raise ValueError(f'Explicit source lacks sample indices: {missing}')
    expected_hashes = json.loads((args.audit / 'gt_hashes.json').read_text())
    if set(map(int, expected_hashes)) != set(target):
        raise ValueError('GT inventory changed since audit')
    try:
        import cv2
        cv_version = cv2.__version__
    except ImportError:
        cv_version = None
    result = dict(status='encoding_investigation', gt_directory=str(gt),
                  explicit_candidate_directory=str(original) if original else None,
                  source_sha256=sha(Path(__file__)), opencv_version=cv_version,
                  inventory=[dict(directory=str(p), count=len(files), examples=[str(f) for f in list(files.values())[:3]]) for p, files in folders],
                  target_samples=[], candidate_comparisons=[], code_evidence=code_evidence(args.workspace.resolve()),
                  limitations=['A sample match does not certify all masks or reveal the historical command.',
                               'No original masks are modified; no extra values are rounded or reassigned.',
                               'Search is bounded to the GT clip folder (depth 3), explicit candidate folder (depth 0), and workspace code (depth 4).'])
    candidates = [(p, files) for p, files in folders if p.resolve() != gt and all(i in files for i in indices)]
    comparisons = {}
    for index in indices:
        raw, gray, rgb, detail = load(target[index])
        if detail['sha256'] != expected_hashes[str(index)]:
            raise ValueError(f'Audited GT changed: {index}')
        result['target_samples'].append(dict(gt_frame=index, **detail))
        target_arr = raw[..., 0] if raw.ndim == 3 and detail.get('channels_identical') else raw
        size = (raw.shape[1], raw.shape[0])
        for folder, files in candidates:
            for recipe, regenerated in recipes(files[index], size):
                if regenerated.shape != target_arr.shape:
                    continue
                key = (str(folder), recipe)
                row = comparisons.setdefault(key, dict(directory=str(folder), recipe=recipe, samples=0,
                                                       exact_samples=0, mismatch_fraction_sum=0.0))
                fraction = float(np.mean(regenerated != target_arr))
                row['samples'] += 1
                row['exact_samples'] += int(fraction == 0)
                row['mismatch_fraction_sum'] += fraction
        print(f'Encoding sample GT {index} checked', flush=True)
    ranked = sorted(comparisons.values(), key=lambda r: (-r['exact_samples'], r['mismatch_fraction_sum']))
    result['candidate_comparisons'] = ranked[:30]
    result['original_samples'] = [load(files[indices[0]])[3] for _, files in candidates]
    result['status'] = 'sample_exact_recipe_found' if ranked and ranked[0]['exact_samples'] == len(indices) else 'cause_not_yet_established'
    if result['status'] == 'sample_exact_recipe_found':
        best = ranked[0]
        original = next(files for p, files in candidates if str(p) == best['directory'])
        verified = Counter()
        failures = []
        original_values = set()
        source_hashes = {}
        # Full coverage check for ONE matching recipe; no repeated manual review.
        for n, (index, path) in enumerate(sorted(target.items())):
            if index not in original:
                verified['missing_original'] += 1
                continue
            raw, _, _, detail = load(path)
            if detail['sha256'] != expected_hashes[str(index)]:
                raise ValueError(f'Audited GT changed: {index}')
            target_arr = raw[..., 0] if raw.ndim == 3 and detail.get('channels_identical') else raw
            src, _, _, source_detail = load(original[index])
            source_hashes[str(index)] = source_detail['sha256']
            if src.ndim == 2:
                original_values.update(int(v) for v in np.unique(src))
            generated = next(a for name, a in recipes(original[index], (raw.shape[1], raw.shape[0])) if name == best['recipe'])
            exact = np.array_equal(generated, target_arr)
            verified['exact' if exact else 'different'] += 1
            if not exact and len(failures) < 20:
                failures.append(index)
            if n % 250 == 0:
                print(f'Full recipe check {n}/{len(target)}', flush=True)
        result['full_recipe_check'] = dict(recipe=best, counts=dict(verified), first_mismatches=failures,
                                          original_native_values=sorted(original_values),
                                          original_sha256=source_hashes)
        result['status'] = 'exact_reproduction_all_files' if verified['exact'] == len(target) else 'partial_recipe_match'
    (output / 'verification.json').write_text(json.dumps(result, indent=2))
    print(f'Status: {result["status"]}; results: {output}', flush=True)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--audit', type=Path, default=Path('audit/results/gt-audit-20260912T121456077403Z'))
    parser.add_argument('--workspace', type=Path, default=Path('/media/hbvcai/HBVCSSD/samannot'))
    parser.add_argument('--output', type=Path, help='New result directory; default stays inside this checkout')
    parser.add_argument('--original', type=Path, help='Explicit candidate precursor mask folder; provenance is tested, not assumed')
    run(parser.parse_args())
