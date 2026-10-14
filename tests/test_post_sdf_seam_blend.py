import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src' / 'app'))
from post_sdf_seam_blend import (
    build_post_sdf_seam_plan,
    overlap_crossfade_sdf,
    overlap_crossfade_weights,
)


def item(name, src0, src1, group='cone_body', target='avatar|body'):
    return {
        'name': name,
        'target_key': target,
        'accessory_key': 'cone|cone',
        'src_0': src0,
        'src_1': src1,
        'blend_group': group,
        'mode': 'direct',
    }


class PostSDFSeamBlendTests(unittest.TestCase):
    def setUp(self):
        self.items = [
            item('cone_a', 0.0, 0.3),
            item('cone_b', 0.2, 0.55),
        ]
        self.groups = {
            'cone_body': {
                'seam_blend': {
                    'enabled': True,
                    'mode': 'overlap_crossfade',
                    'edge_hold_fraction': 0.15,
                }
            }
        }

    def test_plan_uses_explicit_source_overlap(self):
        plan = build_post_sdf_seam_plan(self.items, self.groups)
        self.assertEqual(set(plan), {1})
        pair = plan[1]
        self.assertAlmostEqual(pair.source_lo, 0.2)
        self.assertAlmostEqual(pair.source_hi, 0.3)
        self.assertEqual(pair.left_index, 0)
        self.assertEqual(pair.right_index, 1)

    def test_crossfade_preserves_each_original_field_at_overlap_edges(self):
        pair = build_post_sdf_seam_plan(self.items, self.groups)[1]
        s = np.array([0.2, 0.215, 0.25, 0.285, 0.3])
        left = np.full(s.shape, -0.2)
        right = np.full(s.shape, 0.4)
        out = overlap_crossfade_sdf(left, right, s, pair)
        self.assertAlmostEqual(out[0], left[0])
        self.assertAlmostEqual(out[1], left[1])
        self.assertAlmostEqual(out[2], 0.1)
        self.assertAlmostEqual(out[3], right[3])
        self.assertAlmostEqual(out[4], right[4])

    def test_crossfade_weight_has_zero_slope_at_held_edges(self):
        pair = build_post_sdf_seam_plan(self.items, self.groups)[1]
        # edge_hold_fraction=.15 -> active transition [0.215, 0.285]
        eps = 1e-6
        samples = np.array([0.215 - eps, 0.215, 0.215 + eps,
                            0.285 - eps, 0.285, 0.285 + eps])
        w = overlap_crossfade_weights(samples, pair)
        self.assertAlmostEqual(w[0], 0.0)
        self.assertAlmostEqual(w[1], 0.0)
        self.assertLess(abs(w[2] - w[1]), 1e-8)
        self.assertLess(abs(w[4] - w[3]), 1e-8)
        self.assertAlmostEqual(w[4], 1.0)
        self.assertAlmostEqual(w[5], 1.0)

    def test_no_overlap_is_rejected(self):
        items = [item('a', 0.0, 0.2), item('b', 0.21, 0.5)]
        with self.assertRaisesRegex(ValueError, 'positive src overlap'):
            build_post_sdf_seam_plan(items, self.groups)

    def test_cross_target_group_is_rejected(self):
        items = [item('a', 0.0, 0.3), item('b', 0.2, 0.5, target='avatar|other')]
        with self.assertRaisesRegex(ValueError, 'mixes avatar targets'):
            build_post_sdf_seam_plan(items, self.groups)

    def test_streaming_yaml_order_is_validated(self):
        items = [item('right', 0.2, 0.55), item('left', 0.0, 0.3)]
        with self.assertRaisesRegex(ValueError, 'in increasing source-curve order'):
            build_post_sdf_seam_plan(items, self.groups)


if __name__ == '__main__':
    unittest.main()
