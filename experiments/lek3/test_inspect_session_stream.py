import pickle
import tempfile
import unittest
from pathlib import Path
from inspect_session_stream import inspect


class ScanTests(unittest.TestCase):
    def scan(self,data):
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'session.pkl';path.write_bytes(data)
            result=inspect(path)
            self.assertEqual(path.read_bytes(),data)
            return result
    def test_skips_payload_and_finds_later_metadata(self):
        r=self.scan(pickle.dumps({'image':b'x'*3000000,'block_size':750},protocol=5))
        self.assertEqual(r['status'],'opcode_scan_completed')
        self.assertEqual(r['skipped_payload_bytes'],3000000)
        self.assertTrue(any(f['field']=='block_size' and f['value']==750 for f in r['field_observations']))
    def test_two_consecutive_objects(self):
        r=self.scan(pickle.dumps({'mode':'correction'})+pickle.dumps({'max_frame':16477}))
        self.assertEqual(len(r['stop_offsets']),2)
        self.assertEqual(r['status'],'opcode_scan_completed')
    def test_global_reference_is_not_imported(self):
        r=self.scan(b'cnot_a_real_module\nnot_a_real_function\n.')
        self.assertEqual(r['status'],'opcode_scan_completed')
    def test_truncated_payload(self):
        r=self.scan(b'\x80\x05B'+(1000000).to_bytes(4,'little')+b'x')
        self.assertEqual(r['status'],'scan_stopped')
    def test_missing_stop(self):
        r=self.scan(b'\x80\x05}')
        self.assertEqual(r['status'],'incomplete_pickle_stream')


if __name__=='__main__':unittest.main()
