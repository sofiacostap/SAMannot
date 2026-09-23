"""Play verified GT images and separate confirmed frame artifacts reversibly."""
import argparse
import csv
import json
import secrets
import threading
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from PIL import Image
from recompute import sha
from review_all_gt import Review, inventory


class Source(Review):
    def __init__(self, reference, source):
        self.ids, self.frames = inventory(reference, source)


class ArtifactReview:
    def __init__(self, reference, source, destination):
        self.source = Source(reference, source)
        self.root = destination/'gt-review'
        self.root.mkdir(parents=True, exist_ok=True)
        self.lock = threading.Lock()
        inputs = {name: sha(path) for name, path in [
            ('reference', reference/'evaluation_set.csv'), ('masks', reference/'mask_hashes.json'),
            ('source', source/'index.csv'), ('verification', source/'verification.json')]}
        self.path = self.root/'progress.json'
        if self.path.exists():
            self.record = json.loads(self.path.read_text(encoding='utf-8'))
            if self.record['inputs'] != inputs:
                raise ValueError('Existing review belongs to different source inputs')
        else:
            self.record = dict(inputs=inputs, displayed=[], flagged=[], last=0,
                completed=False, policy='Unflagged after visual playback; not individual mask certification')
            self.save()
        for folder in ('gt artifacts', 'undo history', 'exports'):
            (self.root/folder).mkdir(exist_ok=True)

    def save(self):
        temp = self.path.with_suffix('.tmp')
        temp.write_text(json.dumps(self.record, indent=2), encoding='utf-8')
        temp.replace(self.path)

    def check(self, index):
        if type(index) is not int or not 0 <= index < len(self.source.frames):
            raise ValueError('Invalid frame index')

    def visit(self, index):
        self.check(index)
        self.source.pixels(index)
        if index not in self.record['displayed']:
            self.record['displayed'].append(index)
        self.record['last'] = index
        self.save()

    def flag(self, index, confirmed):
        self.check(index)
        if confirmed is not True:
            raise ValueError('Explicit confirmation required')
        if index not in self.record['displayed']:
            raise ValueError('Display the frame before flagging')
        vf = self.source.frames[index]['video_frame']
        folder = self.root/'gt artifacts'/f'{vf:06d}'
        if index not in self.record['flagged']:
            rgb, gt = self.source.pixels(index)
            folder.mkdir(exist_ok=True)
            Image.fromarray(rgb).save(folder/'photograph.png')
            Image.fromarray(gt).save(folder/'gt_mask.png')
            (folder/'overlay.png').write_bytes(self.source.picture(index, 'overlay', 0, True))
            (folder/'frame.json').write_text(json.dumps(dict(video_frame=vf,
                gt_frame=self.source.frames[index]['gt_frame'], scope='whole_frame'), indent=2))
            self.record['flagged'].append(index)
            self.record['completed'] = False
            self.save()

    def undo(self, index):
        self.check(index)
        if index in self.record['flagged']:
            vf = self.source.frames[index]['video_frame']
            folder = self.root/'gt artifacts'/f'{vf:06d}'
            if folder.exists():
                folder.rename(self.root/'undo history'/f'{vf:06d}-{datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")}')
            self.record['flagged'].remove(index)
            self.record['completed'] = False
            self.save()

    def finish(self, confirmed):
        if confirmed is not True or len(set(self.record['displayed'])) != len(self.source.frames):
            raise ValueError('Display all frames and confirm the viewing pass before export')
        stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
        target = self.root/'exports'/stamp
        staging = target.with_name(stamp+'.incomplete')
        staging.mkdir()
        for category in ('gt artifacts', 'retained GT'):
            (staging/category).mkdir()
        rows = []
        for index, frame in enumerate(self.source.frames):
            category = 'gt artifacts' if index in self.record['flagged'] else 'retained GT'
            folder = staging/category
            folder.mkdir(exist_ok=True)
            # Recheck all images when exporting, including files previously cached during playback.
            self.source.pixels.cache_clear()
            rgb, gt = self.source.pixels(index)
            stem = f"{frame['video_frame']:06d}"
            photo, mask = folder/(stem+'_photo.png'), folder/(stem+'_gt.png')
            Image.fromarray(rgb).save(photo); Image.fromarray(gt).save(mask)
            rows.append(dict(video_frame=frame['video_frame'], gt_frame=frame['gt_frame'],
                category=category, photograph=photo.relative_to(staging).as_posix(),
                mask=mask.relative_to(staging).as_posix(), photo_sha256=sha(photo), mask_sha256=sha(mask)))
        with (staging/'manifest.csv').open('w', newline='', encoding='utf-8') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
        (staging/'README.txt').write_text('GT partition after visual playback. Retained means no artifact was flagged, not independently certified perfect.\n'
            'Flags exclude the entire image (all bird labels), never the whole video. Original source files are unchanged.\n'
            'Filenames use video indices; manifest.csv also records GT indices. Do not mix this with the earlier automatic partition.\n')
        (staging/'verification.json').write_text(json.dumps(dict(status='completed_visual_playback_partition',
            retained_frames=len(rows)-len(self.record['flagged']), artifact_frames=len(self.record['flagged']),
            total_frames=len(rows), source_inputs=self.record['inputs']), indent=2))
        staging.rename(target)
        self.record['completed'] = True
        self.record['latest_export'] = str(target.relative_to(self.root))
        self.save()
        return self.record['latest_export']


def serve(review, port):
    token = secrets.token_urlsafe(24)
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args): pass
        def reply(self, data, kind='application/json', status=200):
            if not isinstance(data, bytes): data = (data if isinstance(data, str) else json.dumps(data)).encode()
            self.send_response(status); self.send_header('Content-Type', kind)
            self.send_header('Cache-Control', 'no-store'); self.end_headers(); self.wfile.write(data)
        def route(self):
            p = urlparse(self.path); q = parse_qs(p.query)
            if q.get('token', [''])[0] != token: raise ValueError('Invalid local session')
            return p.path, q
        def do_GET(self):
            try:
                route, q = self.route()
                if route == '/': self.reply(Path(__file__).with_suffix('.html').read_bytes(), 'text/html; charset=utf-8')
                elif route == '/state':
                    with review.lock:
                        self.reply(dict(record=review.record, frames=[dict(video_frame=f['video_frame'],gt_frame=f['gt_frame']) for f in review.source.frames]))
                elif route == '/image':
                    i = int(q['index'][0]); review.check(i)
                    self.reply(review.source.picture(i, 'overlay', 0, False), 'image/png')
                else: self.reply('Not found', 'text/plain', 404)
            except (ValueError, KeyError, OSError) as exc: self.reply(dict(error=str(exc)), status=400)
        def do_POST(self):
            try:
                route, q = self.route()
                size = int(self.headers.get('Content-Length', 0))
                if not 0 < size < 4096: raise ValueError('Invalid request')
                data = json.loads(self.rfile.read(size))
                with review.lock:
                    if route == '/visit': review.visit(data['index'])
                    elif route == '/flag': review.flag(data['index'], data.get('confirmed'))
                    elif route == '/undo': review.undo(data['index'])
                    elif route == '/finish': review.finish(data.get('confirmed'))
                    else: raise ValueError('Unknown action')
                    self.reply(review.record)
            except (ValueError, KeyError, OSError, TypeError) as exc: self.reply(dict(error=str(exc)), status=400)
    server = ThreadingHTTPServer(('127.0.0.1', port), Handler)
    print(f'Open http://127.0.0.1:{port}/?token={token}', flush=True)
    print('Progress and separated artifacts:', review.root, flush=True)
    try: server.serve_forever()
    except KeyboardInterrupt: pass
    finally: server.server_close()


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--destination', type=Path, required=True)
    p.add_argument('--reference', type=Path, default=Path('audit/results/task2-reference-20260915T080605348526Z'))
    p.add_argument('--source', type=Path, default=Path('audit/results/verified-source-20260911T164528649441Z'))
    p.add_argument('--port', type=int, default=8768)
    a = p.parse_args()
    if not a.destination.is_dir(): p.error('Destination must already exist')
    serve(ArtifactReview(a.reference, a.source, a.destination), a.port)
