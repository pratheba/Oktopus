import tempfile
from pathlib import Path
import unittest

import numpy as np
import trimesh
import yaml

from app.root_leaf_combination import build_root_leaf_plan, export_root_leaf_scene


HERE = Path(__file__).resolve().parents[1]


class RootLeavesTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config = yaml.safe_load(
            (HERE / 'adapt_oktopus_puffer_root_leaves.yaml').read_text()
        )

    def test_all_fifteen_puffer_adaptations_have_one_part(self):
        plan = build_root_leaf_plan(self.config['adaptations'],
                                    self.config['root_leaves'])
        self.assertEqual(len(plan.item_to_part), 15)
        self.assertEqual(len(plan.part_names), 8)
        self.assertEqual(plan.item_to_part[0], 'body')
        self.assertEqual([plan.item_to_part[i] for i in (1, 2, 3)],
                         ['tentacle_t1'] * 3)
        self.assertEqual(plan.group_to_part['t4'], 'tentacle_t4')
        self.assertEqual(plan.item_to_part[14], 'tentacle_t8')
        self.assertFalse(plan.export_legacy_combined)

    def test_cross_leaf_blend_group_is_rejected(self):
        pieces = [dict(x) for x in self.config['adaptations']]
        pieces[4]['blend_group'] = 't1'
        with self.assertRaisesRegex(ValueError, 'spans'):
            build_root_leaf_plan(pieces, self.config['root_leaves'])

    def test_unknown_or_unassigned_piece_is_rejected(self):
        pieces = [dict(x) for x in self.config['adaptations']]
        pieces.append({'name': 'unknown', 'target_key': 'oktopus|unknown'})
        with self.assertRaisesRegex(ValueError, 'Unassigned adaptation'):
            build_root_leaf_plan(pieces, self.config['root_leaves'])

    def test_multi_object_glb_keeps_intersecting_tentacles_distinct(self):
        pieces = [
            {'name': 'body', 'target_key': 'shape|body'},
            {'name': 'a', 'target_key': 'shape|a'},
            {'name': 'b', 'target_key': 'shape|b'},
        ]
        plan = build_root_leaf_plan(pieces, {
            'root': 'body',
            'leaves': {'tentacle_a': 'shape|a', 'tentacle_b': 'shape|b'},
        })
        body = trimesh.creation.box(extents=(2, 2, 2))
        a = trimesh.creation.box(extents=(1.2, 0.7, 0.7))
        a.apply_translation((1.0, 0.35, 0.0))
        b = trimesh.creation.box(extents=(1.2, 0.7, 0.7))
        b.apply_translation((1.0, 0.0, 0.0))  # intersects a and body
        meshes = {'body': body, 'tentacle_a': a, 'tentacle_b': b}
        with tempfile.TemporaryDirectory() as out:
            path = export_root_leaf_scene(meshes, out, 'puffer', plan)
            self.assertTrue(path.exists())
            readback = trimesh.load(path, force='scene')
            self.assertEqual(set(readback.geometry), set(plan.part_names))
            for p in plan.part_names:
                self.assertEqual(len(readback.geometry[p].faces), len(meshes[p].faces))
                self.assertTrue((Path(out) / 'riggable_parts' / f'{p}.ply').exists())
            self.assertEqual(sum(len(x.faces) for x in readback.geometry.values()),
                             sum(len(x.faces) for x in meshes.values()))

    def test_export_refuses_to_omit_a_leaf(self):
        pieces = [
            {'name': 'body', 'target_key': 's|body'},
            {'name': 'a', 'target_key': 's|a'},
        ]
        plan = build_root_leaf_plan(pieces, {'root': 'body', 'leaves': {'a': 's|a'}})
        with tempfile.TemporaryDirectory() as out:
            with self.assertRaisesRegex(RuntimeError, 'missing parts'):
                export_root_leaf_scene({'body': trimesh.creation.box()}, out, 'x', plan)


if __name__ == '__main__':
    unittest.main()
