"""0050: explicit blend-group ownership and SDF/UDF agreement."""
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import trimesh
import yaml

from app.root_leaf_combination import (
    build_root_leaf_plan, export_root_leaf_scene,
)
from app.udf_cut_root_leaf_transfer import transfer_udf_cut_root_leaves

ROOT = Path(__file__).resolve().parents[1] / 'exp/train/manipulation'


class RootLeafOwnership0050(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.sdf = yaml.safe_load((ROOT/'adapt_oktopus_puffer_root_leaves_0050.yaml').read_text())
        cls.udf = yaml.safe_load((ROOT/'adapt_oktopus_puffer_blend_udf_cut_root_leaves_0050.yaml').read_text())

    def test_fifteen_sections_eight_parts_matching_fingerprint(self):
        a = build_root_leaf_plan(self.sdf['adaptations'], self.sdf['root_leaves'])
        b = build_root_leaf_plan(self.udf['adaptations'], self.udf['root_leaves'])
        self.assertEqual(a, b)
        self.assertEqual(len(a.item_to_part), 15)
        self.assertEqual(len(a.part_names), 8)
        self.assertEqual([a.item_to_part[i] for i in (1,2,3)], ['tentacle_t1']*3)
        self.assertEqual(a.group_to_part['t4'], 'tentacle_t4')
        self.assertEqual([a.item_to_part[i] for i in (7,8,9)], ['tentacle_t4']*3)
        self.assertEqual([a.item_to_part[i] for i in (10,11,12)], ['tentacle_t5']*3)
        self.assertEqual(a.item_to_part[0], 'body')
        self.assertEqual(a.item_to_part[14], 'tentacle_t8')
        self.assertTrue(a.adaptation_fingerprint)
        self.assertFalse(a.export_legacy_combined)
        self.assertEqual(self.udf['udf_cut_combined_mode'], 'transfer')
        self.assertTrue(self.udf['udf_cut_require_owner_manifest'])

    def test_runtime_only_yaml_indices_do_not_change_fingerprint(self):
        from app.root_leaf_combination import adaptation_fingerprint
        runtime = [dict(item, _yaml_index=i, _target_index=0)
                   for i, item in enumerate(self.sdf['adaptations'])]
        self.assertEqual(adaptation_fingerprint(runtime),
                         adaptation_fingerprint(self.udf['adaptations']))

    def test_cross_leaf_owner_and_extra_segment_rejected(self):
        specs = self.sdf['root_leaves']
        items = [dict(a) for a in self.sdf['adaptations']]
        items[7]['blend_group'] = 't1'
        with self.assertRaisesRegex(ValueError, '(spans|multiple|shared)'):
            build_root_leaf_plan(items, specs)
        items = [dict(a) for a in self.sdf['adaptations']]
        items.append(dict(name='unmatched', target_key='shape|other'))
        with self.assertRaisesRegex(ValueError, 'Unassigned'):
            build_root_leaf_plan(items, specs)

    def test_no_fake_stitched_output(self):
        spec = dict(self.sdf['root_leaves'])
        spec['output_mode'] = 'single_connected'
        with self.assertRaisesRegex(ValueError, 'stitching'):
            build_root_leaf_plan(self.sdf['adaptations'], spec)

    def test_one_file_has_part_labels_but_is_unstitched(self):
        plan = build_root_leaf_plan(self.sdf['adaptations'], self.sdf['root_leaves'])
        root = trimesh.creation.box()
        meshes = {'body':root}
        for i, part in enumerate(plan.part_names[1:]):
            m = root.copy()
            m.apply_translation((0.2 + .02*i, 0, 0)) # deliberately intersects
            meshes[part] = m
        with tempfile.TemporaryDirectory() as tmp:
            path = export_root_leaf_scene(meshes, tmp, 'puffer', plan)
            self.assertTrue(path.is_file())
            manifest = json.loads((Path(tmp)/'root_leaves_ownership.json').read_text())
            self.assertFalse(manifest['connected_mesh'])
            self.assertFalse(manifest['stitched_root_leaf_interfaces'])
            self.assertEqual(manifest['adaptation_fingerprint'], plan.adaptation_fingerprint)
            one = trimesh.load(tmp+'/puffer_UNSTITCHED_parts.ply', process=False, force='mesh')
            self.assertEqual(len(one.faces), 8*len(root.faces))
            data = (Path(tmp)/'puffer_UNSTITCHED_parts.ply').read_bytes()
            header = data[:data.index(b'end_header\n')].decode('ascii')
            self.assertEqual(header.count('property int part_id'), 2)
            self.assertIn('comment unstitched_root_leaf_assembly true', header)
            self.assertEqual(manifest['parts']['tentacle_t8']['part_id'], 7)
            self.assertEqual(len(one.split(only_watertight=False)), 8)

    def test_udf_checks_manifest_before_cut(self):
        plan = build_root_leaf_plan(self.sdf['adaptations'], self.sdf['root_leaves'])
        m = trimesh.creation.box()
        meshes = {part: m.copy() for part in plan.part_names}
        empty = trimesh.Trimesh(vertices=m.vertices.copy(), faces=m.faces[:0], process=False)
        with tempfile.TemporaryDirectory() as tmp:
            export_root_leaf_scene(meshes, tmp, 'puffer', plan)
            removed = [empty]*15
            kept = [m]*15
            result, report = transfer_udf_cut_root_leaves(
                plan=plan, mc_parts_dir=str(Path(tmp)/'riggable_parts'), root_path=tmp,
                output_folder=str(Path(tmp)/'udf'), removed_references=removed,
                kept_references=kept, config={'udf_cut_require_owner_manifest': True})
            self.assertFalse(report['connected_mesh'])
            self.assertTrue((Path(tmp)/'udf/riggable_after_udf_cut_UNSTITCHED_parts.ply').exists())
            changed = dict(self.sdf['adaptations'][8]); changed['tloc']=[0,0,-0.05]
            altered = [dict(a) for a in self.sdf['adaptations']]
            altered[8] = changed
            altered_plan = build_root_leaf_plan(altered, self.sdf['root_leaves'])
            with self.assertRaisesRegex(ValueError, 'fingerprint mismatch'):
                transfer_udf_cut_root_leaves(
                    plan=altered_plan,
                    mc_parts_dir=str(Path(tmp)/'riggable_parts'), root_path=tmp,
                    output_folder=str(Path(tmp)/'nope'), removed_references=removed,
                    kept_references=kept, config={'udf_cut_require_owner_manifest': True})
            self.assertFalse((Path(tmp)/'nope').exists())


if __name__ == '__main__':
    unittest.main()
