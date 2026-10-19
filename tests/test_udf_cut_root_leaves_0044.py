import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import trimesh
import yaml

from app.root_leaf_combination import build_root_leaf_plan
from app.udf_cut_root_leaf_transfer import transfer_udf_cut_root_leaves

ROOT = Path(__file__).resolve().parents[1]


class RootLeafUDFCutTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.sdf_cfg = yaml.safe_load((ROOT / 'adapt_oktopus_puffer_root_leaves_udf.yaml').read_text())
        cls.udf_cfg = yaml.safe_load((ROOT / 'adapt_oktopus_puffer_blend_udf_cut_root_leaves.yaml').read_text())

    def test_configs_match_all_fifteen_adaptations_and_eight_parts(self):
        a = self.sdf_cfg
        b = self.udf_cfg
        self.assertTrue(a['export_individual_meshes'])
        self.assertEqual(a['root_leaves'], b['root_leaves'])
        self.assertEqual(
            [(x['target_key'], x['accessory_key'], x.get('name'), x.get('blend_group')) for x in a['adaptations']],
            [(x['target_key'], x['accessory_key'], x.get('name'), x.get('blend_group')) for x in b['adaptations']],
        )
        # Preserve the user's historical UDF-specific tloc corrections; these
        # predate 0044 and are not introduced or removed by the rigging patch.
        offsets = [i for i, (x, y) in enumerate(zip(a['adaptations'], b['adaptations']))
                   if x.get('tloc') != y.get('tloc')]
        self.assertEqual(offsets, [8, 11])
        self.assertEqual(a['blend_groups'], b['blend_groups'])
        self.assertEqual(b['udf_cut_combined_mode'], 'transfer')
        self.assertNotIn('udf_cut_combined_mc_mesh', b)
        self.assertEqual(len(a['adaptations']), 15)
        self.assertEqual(len(build_root_leaf_plan(b['adaptations'], b['root_leaves']).part_names), 8)

    def test_per_part_evidence_cannot_touch_neighbor_part(self):
        adaptations = [
            {'name': 'root', 'target_key': 's|body'},
            {'name': 'arm_a_1', 'target_key': 's|a', 'blend_group': 'a'},
            {'name': 'arm_a_2', 'target_key': 's|a', 'blend_group': 'a'},
            {'name': 'arm_b', 'target_key': 's|b'},
        ]
        spec = {'root': 'root', 'leaves': {'tentacle_a': 's|a', 'tentacle_b': 's|b'}}
        plan = build_root_leaf_plan(adaptations, spec)
        box = trimesh.creation.box(extents=(1,1,1))
        untouched = trimesh.Trimesh(vertices=box.vertices.copy(), faces=box.faces[:0], process=False)
        # Part b is spatially overlapping with a but has no removed UDF evidence.
        refs_removed = [untouched, box, untouched, untouched]
        refs_kept = [box, box, box, box]
        with tempfile.TemporaryDirectory() as temp:
            src = Path(temp) / 'source'; src.mkdir()
            for part in plan.part_names:
                box.export(src / (part+'.ply'))
            recorded = []
            def fake_transfer(mesh, *, removed_reference_meshes, kept_reference_meshes, **kwargs):
                recorded.append((len(removed_reference_meshes),
                                 [len(x.faces) for x in removed_reference_meshes]))
                return mesh.copy(), untouched, {'removed_faces':0}
            out = Path(temp)/'out'
            with patch('app.udf_cut_root_leaf_transfer.transfer_udf_cut_to_combined_mesh', fake_transfer):
                meshes, report = transfer_udf_cut_root_leaves(
                    plan=plan, mc_parts_dir=str(src), root_path=temp,
                    output_folder=str(out), removed_references=refs_removed,
                    kept_references=refs_kept, config={},
                )
            self.assertEqual(recorded, [(2, [len(box.faces), 0])])
            self.assertEqual(report['parts']['tentacle_b']['method'], 'no_cut_no_removed_evidence')
            self.assertEqual(report['parts']['body']['method'], 'no_cut_no_removed_evidence')
            self.assertEqual(set(meshes), set(plan.part_names))
            scene = trimesh.load(str(out / 'riggable_after_udf_cut.glb'), force='scene')
            self.assertEqual(set(scene.geometry), set(plan.part_names))
            self.assertTrue((out/'riggable_after_udf_cut_report.json').exists())
            self.assertTrue(all((out/'riggable_parts_after_udf_cut'/(p+'.ply')).exists()
                                for p in plan.part_names))

    def test_missing_source_part_fails_before_writing_output(self):
        plan = build_root_leaf_plan(
            [{'name':'root','target_key':'s|body'}, {'name':'a','target_key':'s|a'}],
            {'root':'root','leaves':{'a':'s|a'}}
        )
        box = trimesh.creation.box()
        with tempfile.TemporaryDirectory() as temp:
            directory = Path(temp)/'source'; directory.mkdir()
            box.export(directory/'body.ply')
            with self.assertRaisesRegex(FileNotFoundError, 'Missing root-leaves MC part'):
                transfer_udf_cut_root_leaves(
                    plan=plan, mc_parts_dir=str(directory), root_path=temp,
                    output_folder=str(Path(temp)/'out'),
                    removed_references=[box,box], kept_references=[box,box], config={}
                )
            self.assertFalse((Path(temp)/'out').exists())

    def test_misaligned_references_fail_loudly(self):
        plan = build_root_leaf_plan(
            [{'name':'root','target_key':'s|body'}, {'name':'a','target_key':'s|a'}],
            {'root':'root','leaves':{'a':'s|a'}}
        )
        with self.assertRaisesRegex(ValueError, 'Missing per-adaptation removed'):
            transfer_udf_cut_root_leaves(
                plan=plan, mc_parts_dir='unused', root_path='unused',
                output_folder='unused', removed_references=[], kept_references=[], config={}
            )

if __name__=='__main__':
    unittest.main()
