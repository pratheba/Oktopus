import importlib.util
import tempfile
import unittest
from pathlib import Path

import numpy as np

spec = importlib.util.spec_from_file_location(
    'mc_utils_test', Path(__file__).resolve().parents[1] / 'src/utils/mc_utils.py'
)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
MCGrid = module.MCGrid

class MemmapTests(unittest.TestCase):
    def test_storage_and_updates_equal(self):
        N=12
        with tempfile.TemporaryDirectory() as scratch:
            base=MCGrid({'reso':N,'level':0.,'size':1.2})
            mapped=MCGrid({'reso':N,'level':0.,'size':1.2,'memmap_dir':scratch})
            self.assertIsInstance(mapped.val_grid, np.memmap)
            self.assertIsInstance(mapped.empty_marks,np.memmap)
            rng=np.random.default_rng(171)
            ids=rng.integers(0,(N+1)**3,size=350)
            values=rng.uniform(-1,1,len(ids))
            for mode in ('overwrite','minimum'):
                base.update_grid(values,ids,mode=mode,mark=True)
                mapped.update_grid(values,ids,mode=mode,mark=True)
                np.testing.assert_array_equal(base.val_grid,mapped.val_grid)
                np.testing.assert_array_equal(base.empty_marks,mapped.empty_marks)
            self.assertTrue(Path(mapped._grid_tempdir.name,'values.f64').is_file())
            mapped.clear_grid()
            base.clear_grid()
            np.testing.assert_array_equal(base.val_grid,mapped.val_grid)
            np.testing.assert_array_equal(base.empty_marks,mapped.empty_marks)
            mapped._grid_tempdir.cleanup()

if __name__=='__main__': unittest.main()
