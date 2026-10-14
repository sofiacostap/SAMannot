import io
import pickle
import unittest
from PIL import Image
import numpy as np
from inspect_session_structure import Reader, Payload, Inert, image_state


class Tests(unittest.TestCase):
    def read(self, obj):
        data = pickle.dumps(obj, protocol=5)
        return Reader(io.BytesIO(data)).load(), data

    def test_large_payload_preserves_following_keys(self):
        obj, data = self.read({'data': b'x' * 2_000_000, 'masks': {4: {'bird': 1}}})
        payload = obj['data']
        self.assertEqual(data[payload.offset:payload.offset+payload.size], b'x'*2_000_000)
        self.assertEqual(obj['masks'], {4: {'bird': 1}})

    def test_small_framed_payload(self):
        obj, data = self.read({'data': b'abc', 'next': 3})
        self.assertEqual(data[obj['data'].offset:obj['data'].offset+3], b'abc')
        self.assertEqual(obj['next'], 3)

    def test_numpy_is_inert(self):
        obj, _ = self.read(np.arange(12, dtype=np.uint8))
        self.assertIsInstance(obj, Inert)
        self.assertIsInstance(obj.args[0], Payload)

    def test_pillow_pixels_and_dimensions(self):
        obj, data = self.read(Image.new('RGB', (8, 6), 'red'))
        self.assertEqual(obj.state[1:3], ['RGB', (8, 6)])
        p = obj.state[4]
        self.assertEqual(data[p.offset:p.offset+p.size], bytes([255, 0, 0])*48)

    def test_two_independent_pickles(self):
        f = io.BytesIO(pickle.dumps({'a': [1]}, protocol=5)
                       + pickle.dumps({'b': [2]}, protocol=5))
        reader = Reader(f)
        self.assertEqual(reader.load(), {'a': [1]})
        reader.memo.clear()
        self.assertEqual(reader.load(), {'b': [2]})

    def test_jpeg_extended_state(self):
        buffer = io.BytesIO()
        Image.new('RGB', (8, 6), 'red').save(buffer, format='JPEG')
        buffer.seek(0)
        with Image.open(buffer) as source:
            expected = source.tobytes()
            obj, data = self.read(source)
        state = image_state(obj)
        self.assertIsNotNone(state)
        payload = state[4]
        self.assertEqual(data[payload.offset:payload.offset+payload.size], expected)

    def test_global_never_imported(self):
        obj = Reader(io.BytesIO(b'cnot_a_real_module\nWhatever\n)R.')).load()
        self.assertIsInstance(obj, Inert)


if __name__ == '__main__':
    unittest.main()
