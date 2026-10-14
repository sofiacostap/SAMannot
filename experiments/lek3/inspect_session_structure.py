"""Recover session structure with inert objects and deferred binary payloads.

No classes or functions named in the pickle are imported or executed.
Uses CPython's pure-Python pickle reader; requires Python 3.10 or later.
"""
import argparse
import csv
import json
import pickle
import struct
from datetime import datetime, timezone
from pathlib import Path


class Payload:
    def __init__(self, offset, size):
        self.offset, self.size = offset, size


class Inert:
    def __new__(cls, *args, **kwargs):
        return object.__new__(cls)

    def __init__(self, *args, **kwargs):
        self.args = args

    def __setstate__(self, state):
        self.state = state


class Reader(pickle._Unpickler):
    dispatch = pickle._Unpickler.dispatch.copy()

    def __init__(self, stream):
        super().__init__(stream)
        self.stream = stream
        self.types = {}
        self.size = stream.seek(0, 2)
        stream.seek(0)

    def find_class(self, module, name):
        key = module + '.' + name
        if key not in self.types:
            self.types[key] = type('StoredObject', (Inert,), {'stored_type': key})
        return self.types[key]

    def persistent_load(self, pid):
        raise ValueError('External persistent references are unsupported')

    def extension(self, *args):
        raise ValueError('Pickle extension registry is unsupported')

    def frame(self):
        size = struct.unpack('<Q', self.read(8))[0]
        if size > 16 * 1024 * 1024:
            raise ValueError('Unexpected large framing buffer; inspection stopped')
        self._unframer.load_frame(size)

    def payload(self, width):
        size = int.from_bytes(self.read(width), 'little')
        frame = self._unframer.current_frame
        if frame is not None:
            pos = frame.tell()
            end = frame.seek(0, 2)
            frame.seek(pos)
            if end - pos >= size:
                offset = self.stream.tell() - end + pos
                frame.seek(size, 1)
                self.append(Payload(offset, size))
                return
            if end != pos:
                raise ValueError('Payload crosses a pickle frame boundary')
            self._unframer.current_frame = None
        offset = self.stream.tell()
        if size > self.size - offset:
            raise ValueError('Truncated binary payload')
        self.stream.seek(size, 1)
        self.append(Payload(offset, size))

    dispatch[pickle.FRAME[0]] = frame
    dispatch[pickle.BINBYTES[0]] = lambda self: self.payload(4)
    dispatch[pickle.SHORT_BINBYTES[0]] = lambda self: self.payload(1)
    dispatch[pickle.BINBYTES8[0]] = lambda self: self.payload(8)
    dispatch[pickle.BYTEARRAY8[0]] = lambda self: self.payload(8)
    dispatch[pickle.EXT1[0]] = extension
    dispatch[pickle.EXT2[0]] = extension
    dispatch[pickle.EXT4[0]] = extension


def summary(value):
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, (dict, list, tuple, set)):
        return {'type': type(value).__name__, 'count': len(value)}
    return {'type': getattr(value, 'stored_type', type(value).__name__)}


def run(session):
    before = session.stat()
    output = Path(__file__).resolve().parent / 'results' / (
        'structure-' + datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ'))
    output.mkdir(parents=True, exist_ok=False)
    with session.open('rb') as stream:
        reader = Reader(stream)
        backend = reader.load()
        # Each pickle.dump starts a fresh memo.
        reader.memo.clear()
        frontend = reader.load()
        if stream.read(1):
            raise ValueError('Unexpected trailing data')
    if not isinstance(backend, dict) or not isinstance(frontend, dict):
        raise ValueError('Expected two session dictionaries')
    report = {'status': 'structure_inspected_alignment_unverified',
              'backend': {str(k): summary(v) for k, v in backend.items()},
              'frontend': {str(k): summary(v) for k, v in frontend.items()},
              'limitations': ['Stored frame keys are not yet certified video indices.',
                             'Mask contents and annotation provenance are not validated.']}
    masks = backend.get('masks')
    if not isinstance(masks, dict):
        raise ValueError('Unsupported masks container')
    with (output / 'mask_keys.csv').open('w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['outer_key', 'outer_key_type', 'inner_key', 'value_type'])
        for key, value in masks.items():
            entries = value.items() if isinstance(value, dict) else [('', value)]
            for inner, item in entries:
                writer.writerow([str(key), type(key).__name__, str(inner),
                                 getattr(item, 'stored_type', type(item).__name__)])
    keys = list(masks)
    report['mask_outer_keys'] = {'count': len(keys), 'first': list(map(str, keys[:10])),
                                 'last': list(map(str, keys[-10:]))}
    if keys and all(type(k) is int for k in keys):
        ordered = sorted(keys)
        report['mask_outer_keys'].update(min=min(keys), max=max(keys),
            missing_between_extremes=max(keys)-min(keys)+1-len(keys))
    report['saved_images'] = {}
    from PIL import Image
    for name in ('original_img', 'overlay_img'):
        obj = backend.get(name)
        state = getattr(obj, 'state', None)
        # Pillow Image.__getstate__: info, mode, size, palette, pixel bytes.
        if (not isinstance(state, (list, tuple)) or len(state) != 5
                or state[1] != 'RGB' or not isinstance(state[4], Payload)):
            report['saved_images'][name] = 'Unsupported image state; not extracted'
            continue
        width, height = state[2]
        payload = state[4]
        if (type(width) is not int or type(height) is not int or min(width, height) <= 0
                or width*height*3 != payload.size or payload.size > 100_000_000):
            raise ValueError('Image dimensions do not agree with binary payload')
        with session.open('rb') as f:
            f.seek(payload.offset)
            data = f.read(payload.size)
        Image.frombytes('RGB', (width, height), data).save(output / (name + '.png'))
        report['saved_images'][name] = {'width': width, 'height': height}
    media = backend.get('media_files', [])
    index = backend.get('curr_img_idx')
    report['current_media_entry'] = (media[index] if isinstance(media, list)
        and type(index) is int and 0 <= index < len(media) else None)
    after = session.stat()
    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        raise ValueError('Source changed during inspection')
    (output / 'report.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps({'mask_outer_keys': report['mask_outer_keys'],
                      'current_media_entry': report['current_media_entry'],
                      'saved_images': report['saved_images']}, indent=2))
    print('Results:', output)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--session', type=Path, required=True)
    run(parser.parse_args().session)
