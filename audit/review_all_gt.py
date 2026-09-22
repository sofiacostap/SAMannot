"""Review every GT image in a local browser; save decisions without editing masks."""
import argparse
import csv
import io
import json
import secrets
import threading
from collections import Counter
from datetime import datetime, timezone
from functools import lru_cache
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import numpy as np
from PIL import Image
from recompute import sha
from verify_correspondence import fingerprint

STATES = ('pending', 'accepted_present', 'accepted_absent', 'uncertain', 'confirmed_problem')
COLORS = ('#ff6666', '#36d9ef', '#ffd447', '#c58aff', '#65d69a', '#f299cf')


def now():
    return datetime.now(timezone.utc).isoformat()


def read_csv(path):
    with path.open(encoding='utf-8', newline='') as stream:
        return list(csv.DictReader(stream))


def inventory(reference, source):
    if json.loads((source/'verification.json').read_text())['status'] != 'verified':
        raise ValueError('Verified sequential image source required')
    rows = read_csv(reference/'evaluation_set.csv')
    mapping = read_csv(source/'index.csv')
    mapped = {int(r['gt_frames_index']): r for r in mapping if r['gt_frames_index'] != ''}
    if len(mapped) != sum(r['gt_frames_index'] != '' for r in mapping):
        raise ValueError('Duplicate source correspondence')
    hashes = json.loads((reference/'mask_hashes.json').read_text())
    ids = sorted({int(r['gt_id']) for r in rows})
    groups, keys = {}, set()
    for r in rows:
        gf, bird, video = int(r['gt_frame']), int(r['gt_id']), int(r['video_frame'])
        if (gf, bird) in keys:
            raise ValueError('Duplicate bird/frame reference')
        keys.add((gf, bird))
        if gf not in mapped or int(mapped[gf]['video_index']) != video:
            raise ValueError('Reference/source correspondence differs')
        if not r['normalized_mask'] or str(gf) not in hashes:
            raise ValueError('A reference mask is missing; cannot claim complete visual coverage')
        mask = (reference/r['normalized_mask']).resolve()
        photo = (source/'frames'/mapped[gf]['file']).resolve()
        if reference.resolve() not in mask.parents or (source/'frames').resolve() not in photo.parents:
            raise ValueError('Input path outside expected image directory')
        frame = groups.setdefault(gf, dict(gt_frame=gf, video_frame=video, mask=str(mask),
            photo=str(photo), mask_sha256=hashes[str(gf)]['normalized'],
            rgb_sha256=mapped[gf]['rgb_sha256'], birds=[]))
        if frame['mask'] != str(mask):
            raise ValueError('Bird rows disagree on mask path')
        frame['birds'].append(dict(id=bird, gt_present=r['gt_present']=='True',
            prior_status=r['status'], prior_reasons=r['reasons']))
    if set(groups) != set(mapped) or any(sorted(b['id'] for b in f['birds']) != ids for f in groups.values()):
        raise ValueError('Incomplete frame/bird inventory')
    return ids, sorted(groups.values(), key=lambda f: f['video_frame'])


def initial_decisions(frames):
    return {f"{f['video_frame']}:{b['id']}": dict(status='pending', note='', updated_at=None)
            for f in frames for b in f['birds']}


def counts(decisions):
    result = {s: 0 for s in STATES}
    result.update(Counter(d['status'] for d in decisions.values()))
    result['total'] = len(decisions)
    result['reviewed'] = len(decisions)-result['pending']
    result['unresolved'] = result['pending']+result['uncertain']+result['confirmed_problem']
    result['all_observations_reviewed'] = result['pending']==0
    return result


def validate_decision(bird, status, note):
    if status not in STATES or not isinstance(note, str) or len(note)>4000:
        raise ValueError('Invalid decision or note')
    if status=='accepted_present' and not bird['gt_present']:
        raise ValueError('Reference mask is empty; use accepted_absent or flag the problem')
    if status=='accepted_absent' and bird['gt_present']:
        raise ValueError('Reference mask is nonempty; use accepted_present or flag the problem')
    if status in ('uncertain', 'confirmed_problem') and not note.strip():
        raise ValueError('Add a short reason for uncertain/problem decisions')


class Review:
    def __init__(self, reference, source, output, resume=False):
        self.reference, self.source, self.output = reference.resolve(), source.resolve(), output.resolve()
        self.ids, self.frames = inventory(self.reference, self.source)
        self.inputs = {str(p):sha(p) for p in [self.reference/'evaluation_set.csv',
            self.reference/'mask_hashes.json', self.source/'index.csv', self.source/'verification.json']}
        self.lock = threading.Lock()
        if resume:
            self.record = json.loads((self.output/'review.json').read_text())
            if self.record['inputs'] != self.inputs or set(self.record['decisions']) != set(initial_decisions(self.frames)):
                raise ValueError('Saved review belongs to different inputs')
            for f in self.frames:
                for b in f['birds']:
                    d=self.record['decisions'][f"{f['video_frame']}:{b['id']}"]
                    validate_decision(b,d['status'],d['note'])
        else:
            self.output.mkdir(parents=True, exist_ok=False)
            self.record = dict(schema_version=1, purpose='Exhaustive visual GT review',
                created_at=now(), reference=str(self.reference), source=str(self.source), inputs=self.inputs,
                decisions=initial_decisions(self.frames),
                limitations=['Accepted means visually checked, not guaranteed biological truth.',
                    'No masks edited and no evaluation release changed.',
                    'Playback/navigation do not approve observations.',
                    'Complete visual coverage does not resolve uncertain or confirmed-problem cases.'])
            self.save()

    def save(self):
        self.record['counts'] = counts(self.record['decisions'])
        self.record['saved_at'] = now()
        temp=self.output/'review.tmp'
        temp.write_text(json.dumps(self.record,indent=2),encoding='utf-8')
        temp.replace(self.output/'review.json')

    @lru_cache(maxsize=3)
    def pixels(self, index):
        f=self.frames[index]
        if sha(Path(f['mask'])) != f['mask_sha256']:
            raise ValueError('Reference mask changed since release')
        with Image.open(f['photo']) as im: rgb=np.asarray(im.convert('RGB'))
        if fingerprint(rgb) != f['rgb_sha256']:
            raise ValueError('Photograph differs from verified source')
        with Image.open(f['mask']) as im: gt=np.asarray(im)
        if gt.ndim!=2 or gt.shape != rgb.shape[:2] or not set(np.unique(gt)) <= {0,*self.ids}:
            raise ValueError('Unexpected reference shape or labels')
        for b in f['birds']:
            if bool(np.any(gt==b['id'])) != b['gt_present']:
                raise ValueError('Reference presence disagrees with manifest')
        return rgb,gt

    def picture(self,index,view,selected,native):
        rgb,gt=self.pixels(index)
        out=rgb.copy()
        if view=='overlay':
            for i,bird in enumerate(self.ids):
                if selected and bird!=selected: continue
                color=COLORS[i%len(COLORS)]
                tint=np.array([int(color[k:k+2],16) for k in (1,3,5)])
                mask=gt==bird
                out[mask]=(.5*out[mask]+.5*tint).astype(np.uint8)
        im=Image.fromarray(out)
        if not native and im.width>1344:
            im=im.resize((1344,round(im.height*1344/im.width)),Image.Resampling.NEAREST)
        buf=io.BytesIO(); im.save(buf,format='PNG'); return buf.getvalue()

    def update(self,index,changes):
        if not isinstance(changes,list) or not changes:
            raise ValueError('No decisions supplied')
        self.pixels(index)  # Refuse decisions if the actual evidence cannot be verified.
        f=self.frames[index]; birds={b['id']:b for b in f['birds']}; seen=set()
        for change in changes:
            bird=change['id']
            if bird not in birds or bird in seen: raise ValueError('Invalid/duplicate bird ID')
            seen.add(bird); validate_decision(birds[bird],change['status'],change['note'])
        with self.lock:
            previous={str(c['id']):dict(self.record['decisions'][f"{f['video_frame']}:{c['id']}"]) for c in changes}
            for c in changes:
                self.record['decisions'][f"{f['video_frame']}:{c['id']}"]=dict(status=c['status'],note=c['note'].strip(),updated_at=now())
            self.save()
            with (self.output/'events.jsonl').open('a',encoding='utf-8') as stream:
                stream.write(json.dumps(dict(at=now(),video_frame=f['video_frame'],previous=previous,changes=changes))+'\n')
        return self.record['counts']


def serve(review, port):
    token=secrets.token_urlsafe(24)
    class Handler(BaseHTTPRequestHandler):
        def log_message(self,*args): pass
        def respond(self,body,kind='application/json',status=200):
            if isinstance(body,(dict,list)):body=json.dumps(body).encode()
            elif isinstance(body,str):body=body.encode()
            self.send_response(status);self.send_header('Content-Type',kind)
            self.send_header('Cache-Control','no-store');self.end_headers();self.wfile.write(body)
        def route(self):
            parsed=urlparse(self.path);q=parse_qs(parsed.query)
            if q.get('token',[''])[0]!=token: raise ValueError('Invalid local session token')
            return parsed.path,q
        def index(self,q):
            i=int(q['index'][0])
            if not 0<=i<len(review.frames):raise ValueError('Frame outside inventory')
            return i
        def do_GET(self):
            try:
                route,q=self.route()
                if route=='/':self.respond(Path(__file__).with_name('review_all_gt.html').read_text(encoding='utf-8'),'text/html; charset=utf-8')
                elif route=='/state':
                    with review.lock:
                        self.respond(dict(ids=review.ids,colors=COLORS,frames=[{k:v for k,v in f.items() if k in ('gt_frame','video_frame','birds')} for f in review.frames],decisions=review.record['decisions'],counts=review.record['counts'],output=str(review.output)))
                elif route=='/image':
                    view=q.get('view',['photo'])[0];selected=int(q.get('bird',['0'])[0])
                    if view not in ('photo','overlay') or selected not in [0]+review.ids:raise ValueError('Invalid display option')
                    self.respond(review.picture(self.index(q),view,selected,q.get('native',['0'])[0]=='1'),'image/png')
                else:self.respond('Not found','text/plain',404)
            except (ValueError,KeyError,OSError,IndexError) as exc:self.respond(dict(error=str(exc)),status=400)
        def do_POST(self):
            try:
                route,q=self.route()
                if route!='/decision':raise ValueError('Unknown action')
                size=int(self.headers.get('Content-Length','0'))
                if not 0<size<50000:raise ValueError('Invalid request size')
                payload=json.loads(self.rfile.read(size))
                self.respond(dict(counts=review.update(self.index(q),payload['changes'])))
            except (ValueError,KeyError,OSError,IndexError,TypeError) as exc:self.respond(dict(error=str(exc)),status=400)
    server=ThreadingHTTPServer(('127.0.0.1',port),Handler)
    print(f'Open: http://127.0.0.1:{server.server_address[1]}/?token={token}',flush=True)
    print(f'Review saves automatically inside: {review.output}',flush=True)
    print('Ctrl+C stops the server. Resume with --resume and this review directory.',flush=True)
    try:server.serve_forever()
    except KeyboardInterrupt:pass
    finally:server.server_close()


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--reference',type=Path,default=Path('audit/results/task2-reference-20260915T080605348526Z'))
    p.add_argument('--source',type=Path,default=Path('audit/results/verified-source-20260911T164528649441Z'))
    p.add_argument('--resume',type=Path)
    p.add_argument('--port',type=int,default=8767)
    args=p.parse_args()
    output=args.resume or Path(__file__).resolve().parent/'results'/('full-gt-review-'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ'))
    serve(Review(args.reference,args.source,output,bool(args.resume)),args.port)
