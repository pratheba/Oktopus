"""0056 prediction spool: identical bytes, explicit scratch, no data leaks."""
import os
from pathlib import Path
import tempfile
import unittest
import numpy as np
from app.sdf_puffer_prediction_stream_0053 import PredictionStream0053


class Writer0056Tests(unittest.TestCase):
    def _make_writer(self, override, zero_copy):
        return PredictionStream0053(
            agent=None, accessory_key='a', avatar_key='b',
            arg={'use_accessory_support_clamp': True,
                 'sdf_prediction_scratch_dir_0056': override,
                 'sdf_prediction_zero_copy_write_0056': zero_copy},
            seam_corrections=None, scratch_dir='/this-path-does-not-exist',
            batch_size=10)

    def test_zero_copy_bytes_match_0053_even_for_strided_arrays(self):
        with tempfile.TemporaryDirectory() as root:
            rng = np.random.default_rng(56)
            arrays = [rng.normal(size=(97,3)),
                      np.arange(120,dtype=np.int64),
                      rng.normal(size=(15,5))[::2],
                      np.array([1.2,2.3],dtype=np.float32)]
            records = []
            for zero_copy in (False,True):
                writer = self._make_writer(root, zero_copy)
                for j, a in enumerate(arrays):
                    writer._write(str(j),a)
                for f in writer.files.values(): f.close()
                records.append({j:(Path(writer.temp.name) / f'{j}.bin').read_bytes()
                                for j in range(len(arrays))})
                writer.cleanup()
            self.assertEqual(records[0], records[1])

    def test_0056_scratch_expands_environment_and_rejects_bad_dir(self):
        with tempfile.TemporaryDirectory() as root:
            old = os.environ.get('OKTOPUS_0056_SCRATCH')
            os.environ['OKTOPUS_0056_SCRATCH'] = root
            try:
                writer = self._make_writer('$OKTOPUS_0056_SCRATCH', True)
                self.assertEqual(Path(writer.temp.name).parent, Path(root))
                tmpdir = writer.temp.name
                writer.cleanup()
                self.assertFalse(os.path.exists(tmpdir))
                with self.assertRaises(ValueError):
                    self._make_writer('$UNRESOLVED_OKTOPUS_0056_ABC', True)
            finally:
                if old is None: os.environ.pop('OKTOPUS_0056_SCRATCH',None)
                else: os.environ['OKTOPUS_0056_SCRATCH'] = old


if __name__ == '__main__':
    unittest.main()
