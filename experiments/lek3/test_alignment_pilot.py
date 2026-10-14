import io
import pickle
import unittest
from types import SimpleNamespace
import numpy as np
from inspect_session_structure import Reader
from alignment_pilot import unpack_mask, prompt_rows


class Tests(unittest.TestCase):
    def packed(self, mask):
        data = pickle.dumps(np.packbits(mask, axis=-1), protocol=5)
        array = Reader(io.BytesIO(data)).load()
        return SimpleNamespace(stored_type='annotator.PackedMasks',
            state={'h': mask.shape[0], 'w': mask.shape[1], 'data': array}), io.BytesIO(data)

    def test_lossless_non_byte_aligned_mask(self):
        mask = np.array([[1,0,1,1,0,0,1,0,1], [0,1,0,0,1,1,0,1,0]], dtype=bool)
        obj, stream = self.packed(mask)
        np.testing.assert_array_equal(unpack_mask(obj, stream), mask)

    def test_reject_inconsistent_dimensions(self):
        obj, stream = self.packed(np.zeros((2, 9), dtype=bool))
        obj.state['h'] = 3
        with self.assertRaises(ValueError):
            unpack_mask(obj, stream)

    def test_empty_mask_remains_empty(self):
        obj, stream = self.packed(np.zeros((3, 8), dtype=bool))
        self.assertEqual(int(unpack_mask(obj, stream).sum()), 0)

    def test_prompt_block_local_indices_and_invalid_index(self):
        label = SimpleNamespace(state={'name': 'bird', 'pts': {
            2: [SimpleNamespace(state={'idx': 12, 'x': 3, 'y': 4}),
                SimpleNamespace(state={'idx': 500})]}, 'boxes': {}})
        labels, rows = prompt_rows({'block_size': 500, 'sam_handler_labels': [label]})
        self.assertEqual(rows[0]['session_index_candidate'], 1012)
        self.assertIsNone(rows[1]['session_index_candidate'])
        self.assertEqual(labels[0]['name'], 'bird')


if __name__ == '__main__':
    unittest.main()
