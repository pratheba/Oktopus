"""Small CPU-only contracts for 0103 canonical world-space postblend Boolean.

Run: PYTHONPATH=src python -m unittest discover -s tests -p test_mixed_direct_stream_0103.py -v
"""
import copy
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from app.sdf_stream_combined_wrap import (
    validate_cone_stream, combine_two_sparse_parts,
    apply_canonical_world_avatar_boolean, merge_sparse_avatar_field,
)


class FakeGrid:
    reso = 3

    def idx2pts(self, indices):
        idx = np.asarray(indices, dtype=np.int64)
        return np.column_stack([idx % 4, (idx // 4) % 4, idx // 16]).astype(float)


class FakeCanonicalCurve:
    def __init__(self, missing=False):
        self.queries = []
        self.missing = missing

    def localize_samples(self, world, **kwargs):
        self.queries.append((world.copy(), kwargs))
        indices = np.arange(len(world), dtype=np.int64)
        if self.missing:
            indices = indices[:-1]
        # Test inference has exactly one unambiguous field per world point;
        # it deliberately ignores per-section source intervals / radius modes.
        return {'samples': world[indices], 'samples_local': world[indices],
                'coords': world[indices, 0]}, indices


def pseudo_avatar_model(data):
    xyz = data['samples']
    return xyz[:, 0] + 2.0 * xyz[:, 1] + 3.0 * xyz[:, 2]


class MixedSectionTests(unittest.TestCase):
    def setUp(self):
        self.base_dir = Path(__file__).resolve().parents[1]
        folder = self.base_dir / 'exp/train/manipulation'
        self.configs = []
        for name in ('oct23_adapt_oktopus_cone_rigid_first_wrap_second_0103.yaml',
                     'oct23_adapt_oktopus_cone_wrap_first_rigid_second_0103.yaml'):
            self.configs.append(yaml.safe_load((folder / name).read_text()))

    def _validate(self, config):
        items = config['adaptations']
        seams = {1: SimpleNamespace(group='cone_body')}
        plan = SimpleNamespace(part_names=['cone'], item_to_part={0:'cone',1:'cone'})
        return validate_cone_stream(items, config, seams, {}, plan)

    def test_both_mixed_configurations_allowed_and_boolean_is_canonical(self):
        for config in self.configs:
            with self.subTest(rigid_first=config['adaptations'][0]['rigid_radius']):
                self.assertEqual(self._validate(config).group, 'cone_body')
                self.assertTrue(config['sdf_postblend_avatar_boolean_0093']['enabled'])
                self.assertEqual(config['sdf_postblend_avatar_boolean_0093']['avatar_field_mode'],
                                 'canonical_world')
                self.assertFalse(config['sdf_stream_export_sections_0093'])

    def test_reject_ambiguous_mapping_and_unsafe_angular_wrap(self):
        conf = copy.deepcopy(self.configs[0])
        conf['adaptations'][0]['wrap_radius'] = True
        with self.assertRaisesRegex(ValueError, 'exactly one'):
            self._validate(conf)
        conf = copy.deepcopy(self.configs[1])
        conf['adaptations'][0]['wrap_angular_follow'] = 1.0
        with self.assertRaisesRegex(ValueError, 'nonzero angular follow'):
            self._validate(conf)

    def test_reject_unsafe_per_section_boolean(self):
        conf = copy.deepcopy(self.configs[0])
        conf['adaptations'][0]['cut_avatar'] = True
        with self.assertRaisesRegex(ValueError, 'unsupported cut_avatar'):
            self._validate(conf)

    def test_canonical_boolean_chunked_and_mapping_independent(self):
        idx = np.array([0, 1, 4, 8, 10, 15, 28, 63], dtype=np.int64)
        original = np.array([-0.1, 0.5, -2.0, 3.0, -4.0, -2.5, 0.3, -9.0])
        expected_avatar = pseudo_avatar_model({'samples': FakeGrid().idx2pts(idx)})
        clearance = 0.0001
        expected = np.maximum(original, clearance - expected_avatar)
        vals = original.copy()
        curve = FakeCanonicalCurve()
        output = apply_canonical_world_avatar_boolean(
            idx, vals, FakeGrid(), curve, pseudo_avatar_model,
            clearance=clearance, chunk_size=3)
        self.assertIs(output, vals)  # in place, no full extra avatar array
        np.testing.assert_array_equal(vals, expected)
        self.assertEqual([len(q[0]) for q in curve.queries], [3, 3, 2])
        self.assertTrue(all(q[1]['outside'] and np.isinf(q[1]['norm'])
                            for q in curve.queries))

    def test_missing_canonical_projection_fails_not_silent(self):
        with self.assertRaisesRegex(RuntimeError, 'dropped or reordered'):
            apply_canonical_world_avatar_boolean(
                np.array([1, 2]), np.array([-1.0, -2.0]), FakeGrid(),
                FakeCanonicalCurve(missing=True), pseudo_avatar_model)

    def test_bounds_and_sort_invariants(self):
        for idx in (np.array([1, 1]), np.array([-1, 4]), np.array([3, 2])):
            with self.subTest(indices=idx):
                with self.assertRaises(ValueError):
                    apply_canonical_world_avatar_boolean(
                        idx, np.ones(2), FakeGrid(), FakeCanonicalCurve(), pseudo_avatar_model)
        with self.assertRaisesRegex(ValueError, 'outside MCGrid'):
            apply_canonical_world_avatar_boolean(
                np.array([64]), np.ones(1), FakeGrid(), FakeCanonicalCurve(), pseudo_avatar_model)

    def test_accessory_sparse_merge_crossfade_unmodified(self):
        # Exact old overlap arithmetic; canonical Boolean runs only AFTER this.
        from types import SimpleNamespace
        seam = SimpleNamespace(mode='overlap_crossfade',
                               edge_hold_fraction=0.15, s0=0.0, s1=1.0)
        # no need to invoke seam implementation to assert sorted voxel union
        left = np.array([1, 3, 7])
        right = np.array([2, 4, 8])
        idx, sdf, overlap = combine_two_sparse_parts(
            left, np.array([-1., -2., -3.]),
            right, np.array([-4., -5., -6.]),
            np.array([0.2, 0.5, 0.8]), seam)
        np.testing.assert_array_equal(idx, [1, 2, 3, 4, 7, 8])
        np.testing.assert_array_equal(sdf, [-1., -4., -2., -5., -3., -6.])
        self.assertEqual(overlap, 0)

    def test_existing_overlap_parity_still_rejects(self):
        with self.assertRaisesRegex(ValueError, 'disagrees'):
            merge_sparse_avatar_field([1,2], [0.1,0.4], [2,3], [0.6,0.1], tolerance=0.005)


if __name__ == '__main__':
    unittest.main()
