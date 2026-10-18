"""Small-resolution parity tests for the 2048 boot startup optimizations."""
import ast
import importlib.util
from pathlib import Path
import tempfile
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('runtime_mc_grid', ROOT / 'src/utils/mc_utils.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
MCGrid = module.MCGrid

# Extract the real function without importing the whole neural inference stack.
app_source = (ROOT / 'src/app/app_utils_3dvec.py').read_text()
app_tree = ast.parse(app_source)
create_node = next(n for n in app_tree.body if isinstance(n, ast.FunctionDef) and n.name == 'create_grid_like')


class StubOldGrid:
    def __init__(self, config):
        self.init_arg = config


context = {'MCGrid': StubOldGrid}
exec(compile(ast.Module(body=[create_node], type_ignores=[]), filename='app_utils_3dvec.py', mode='exec'), context)
create_grid_like = context['create_grid_like']


class StartupMemoryTests(unittest.TestCase):
    def test_geometry_only_skips_full_storage_and_can_generate_candidates(self):
        with tempfile.TemporaryDirectory() as root:
            grid = MCGrid({'reso': 8, 'level': 0., 'size': 1.2, 'memmap_dir': root, 'geometry_only': True})
            self.assertIsNone(grid.val_grid)
            self.assertIsNone(grid.empty_marks)
            self.assertIsNone(grid._grid_tempdir)
            self.assertTrue(grid._deferred_initialization)
            self.assertEqual(list(Path(root).iterdir()), [])
            marks = grid.create_marks()
            marks[[2, 8, 36]] = True
            xyz, ids = grid.get_marked(marks)
            np.testing.assert_array_equal(ids, [2, 8, 36])
            np.testing.assert_array_equal(xyz, grid.idx2pts(ids))
            with self.assertRaisesRegex(RuntimeError, 'geometry-only'):
                grid.clear_grid()

    def test_deferred_memmap_matches_legacy_filled_grid(self):
        with tempfile.TemporaryDirectory() as root:
            proto = MCGrid({'reso': 10, 'level': 0., 'size': 1.2, 'memmap_dir': root, 'geometry_only': True})
            normal = MCGrid({'reso': 10, 'level': 0., 'size': 1.2})
            for seed in (1, 2):
                piece = create_grid_like(proto, defer_initial_fill=True)
                self.assertIs(type(piece), MCGrid)
                self.assertIsInstance(piece.val_grid, np.memmap)
                self.assertIsInstance(piece.empty_marks, np.memmap)
                self.assertTrue(piece._deferred_initialization)
                piece.clear_grid()
                self.assertFalse(piece._deferred_initialization)
                normal.clear_grid()
                rng = np.random.default_rng(seed)
                ids = rng.choice(11**3, size=240, replace=False)
                values = rng.standard_normal(len(ids))
                piece.update_grid(values, ids, mark=True, mode='overwrite')
                normal.update_grid(values, ids, mark=True, mode='overwrite')
                np.testing.assert_array_equal(piece.val_grid, normal.val_grid)
                np.testing.assert_array_equal(piece.empty_marks, normal.empty_marks)
                piece._grid_tempdir.cleanup()

    def test_existing_memmap_copies_and_in_memory_legacy_preserved(self):
        with tempfile.TemporaryDirectory() as root:
            grid = MCGrid({'reso': 4, 'level': 0., 'size': 1.2, 'memmap_dir': root})
            twin = create_grid_like(grid)
            np.testing.assert_array_equal(grid.val_grid, twin.val_grid)
            self.assertIsInstance(twin.val_grid, np.memmap)
            self.assertFalse(twin._deferred_initialization)
            grid._grid_tempdir.cleanup()
            twin._grid_tempdir.cleanup()
        fake = StubOldGrid({'level': 0., 'reso': 4, 'size': 1.2})
        fake.grid_config = {'level': 0.}
        fake.reso = 4
        fake.size = 1.2
        twin = create_grid_like(fake)
        self.assertIsInstance(twin, StubOldGrid)
        self.assertEqual(twin.init_arg, {'level': 0., 'reso': 4, 'size': 1.2})


if __name__ == '__main__':
    unittest.main()
