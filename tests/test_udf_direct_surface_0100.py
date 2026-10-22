import sys
import unittest
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src/app'))
from udf_direct_surface_0100 import classify_consistency, classify_owner_scores


class TestDirectSurface0100(unittest.TestCase):
    def test_tristate_and_overlap(self):
        assigned = [1, 1, 1, 1, 0, 2, 2]
        valid = [1, 1, 0, 1, 0, 2, 2]
        supported = [1, 0, 0, 0, 0, 1, 0]
        absent = [0, 1, 0, 0, 0, 1, 2]
        keep, remove, unknown = classify_consistency(assigned, valid, supported, absent)
        np.testing.assert_array_equal(keep, [1,0,0,0,0,1,0])
        np.testing.assert_array_equal(remove, [0,1,0,0,0,0,1])
        np.testing.assert_array_equal(unknown, [0,0,1,1,1,0,0])

    def test_far_sentinel_is_never_absence(self):
        valid, support, absent = classify_owner_scores(
            [0.0, 0.015, 0.05, 0.1, float('nan')],
            [1, 1, 1, 1, 1], grow=.01, seed=.02, far=.1)
        np.testing.assert_array_equal(support, [1,0,0,0,0])
        np.testing.assert_array_equal(absent, [0,0,1,0,0])
        self.assertTrue(valid[3])

    def test_invalid_localizer_not_removed(self):
        valid, support, absent = classify_owner_scores([.1], [False], grow=.01, seed=.02, far=.1)
        self.assertFalse(valid[0]); self.assertFalse(support[0]); self.assertFalse(absent[0])

    def test_inconsistent_overlap_rejected(self):
        with self.assertRaises(ValueError):
            classify_consistency([1], [1], [1], [1])

    def test_bad_metric_threshold(self):
        with self.assertRaises(ValueError):
            classify_owner_scores([.1], [True], grow=.02, seed=.01, far=.1)

if __name__ == '__main__':
    unittest.main()
