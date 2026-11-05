"""Read-only preclosed SDF preprocessing: never repair or recalculate caps."""
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import trimesh

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'SDF' / 'preprocess'))
from sdf_closure_utils_v2 import prepare_existing_closed_meshes


class Curve:
    idx = 0
    name = 'body'


class Handle:
    curves = [Curve()]


class TestExistingClosed(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.item = Path(self.temp.name) / 'dress_2_v2'
        self.item.mkdir()
        for directory in ('full_parts', 'base_parts', 'closed_sdf_meshes/parts_full',
                          'closed_sdf_meshes/parts_base'):
            (self.item / directory).mkdir(parents=True)
        closed = trimesh.creation.icosphere(subdivisions=1, radius=1)
        # Open source without modifying the previously closed target mesh.
        source = closed.copy()
        source.update_faces(np.arange(len(source.faces)) != 0)
        source.remove_unreferenced_vertices()
        for name in ('mesh.ply', 'mesh_base.ply'):
            closed.export(self.item / name)
        for variant in ('full', 'base'):
            source.export(self.item / f'{variant}_parts/0_body.ply')
            closed.export(self.item / f'closed_sdf_meshes/parts_{variant}/000_body_closed.ply')
        self.args = dict(
            item_path=self.item, item_name='dress_2_v2', handle=Handle(),
            global_full_mesh_spec='mesh.ply', global_base_mesh_spec='mesh_base.ply',
            full_parts_dir_spec='full_parts', base_parts_dir_spec='base_parts',
            full_part_pattern='{id}_{name}.ply', base_part_pattern='{id}_{name}.ply',
            closed_mesh_dir_spec='closed_sdf_meshes',
        )

    def tearDown(self):
        self.temp.cleanup()

    def test_uses_existing_full_and_base_without_closure(self):
        paths = [self.item / f'closed_sdf_meshes/parts_{k}/000_body_closed.ply'
                 for k in ('full', 'base')]
        before = [(p.stat().st_size, p.stat().st_mtime_ns) for p in paths]
        with patch('sdf_closure_utils_v2.close_all_boundaries',
                   side_effect=AssertionError('must never close mesh')):
            gfull, gbase, full, base = prepare_existing_closed_meshes(**self.args)
        self.assertEqual(full['body'].closed_path, str(paths[0].resolve()))
        self.assertEqual(base['body'].closed_path, str(paths[1].resolve()))
        self.assertTrue(gfull.is_volume)
        self.assertTrue(gbase.is_volume)
        self.assertEqual(full['body'].cap_face_count, 0)
        self.assertEqual(before, [(p.stat().st_size, p.stat().st_mtime_ns) for p in paths])

    def test_missing_closed_part_fails_instead_of_closing(self):
        (self.item / 'closed_sdf_meshes/parts_base/000_body_closed.ply').unlink()
        with self.assertRaisesRegex(FileNotFoundError, 'No preclosed part'):
            prepare_existing_closed_meshes(**self.args)

    def test_invalid_preclosed_mesh_rejected(self):
        source = trimesh.load(self.item / 'full_parts/0_body.ply', process=False)
        source.export(self.item / 'closed_sdf_meshes/parts_full/000_body_closed.ply')
        with self.assertRaisesRegex(ValueError, 'not a valid oriented volume'):
            prepare_existing_closed_meshes(**self.args)

    def test_cap_only_mesh_is_used_if_present(self):
        p = self.item / 'closed_sdf_meshes/parts_full/000_body_closed_caps_only.ply'
        trimesh.Trimesh(vertices=np.array([[0., 0., 0.], [1., 0., 0.], [0., 1., 0.]]),
                        faces=np.array([[0, 1, 2]]), process=False).export(p)
        _, _, full, base = prepare_existing_closed_meshes(**self.args)
        self.assertEqual(full['body'].cap_face_count, 1)
        self.assertEqual(full['body'].cap_path, str(p.resolve()))
        self.assertIsNone(base['body'].cap_path)


if __name__ == '__main__':
    unittest.main()
