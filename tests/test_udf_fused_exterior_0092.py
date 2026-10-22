"""Fast synthetic tests of 0092 fused-SDF evidence transfer (no GPU)."""
import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
import trimesh

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src/app'))
from udf_cut_fused_exterior_0092 import (
    transfer_cut_to_fused, choose_zero_seed_endcaps, generate_fused_cut)


class FusedUdfEvidenceTests(unittest.TestCase):
    def test_identity_surface_preserves_face_selections(self):
        box = trimesh.creation.box()
        expected = np.arange(len(box.faces)) < 4
        cut, owner, dist, info = transfer_cut_to_fused(box, {'body': (box, expected)})
        np.testing.assert_array_equal(cut, expected)
        self.assertEqual(info['unmatched_faces'], 0)
        self.assertEqual(info['proposed_remove_faces'], 4)
        self.assertTrue(np.all(dist < 1e-12))

    def test_wrong_side_surface_nearby_does_not_cut(self):
        src = trimesh.creation.box()
        opposite = src.copy()
        opposite.invert()
        cut, _, _, info = transfer_cut_to_fused(opposite, {
            'source': (src, np.ones(len(src.faces), bool))},
            distance_edge_factor=.2, normal_dot=.9)
        self.assertEqual(int(cut.sum()), 0)
        self.assertEqual(info['unmatched_faces'], len(src.faces))

    def test_spatially_distant_source_does_not_cut(self):
        src = trimesh.creation.box()
        target = src.copy()
        target.apply_translation([10, 0, 0])
        cut, _, _, info = transfer_cut_to_fused(target, {
            'source': (src, np.ones(len(src.faces), bool))})
        self.assertEqual(int(cut.sum()), 0)
        self.assertEqual(info['unmatched_faces'], len(src.faces))

    def test_zero_seed_valid_terminal_cap(self):
        m = trimesh.creation.cylinder(radius=.2, height=1.5, sections=32)
        for _ in range(2):
            v, f = trimesh.remesh.subdivide(m.vertices, m.faces)
            m = trimesh.Trimesh(vertices=v, faces=f, process=False)
        z = np.asarray(m.triangles_center)[:, 2]
        s = (z + .75) / 1.5
        cap = (m.face_normals[:, 2] > .95) & (s > .95)
        found, audit = choose_zero_seed_endcaps(m, np.zeros(len(z),bool), cap, s)
        np.testing.assert_array_equal(found, cap)
        self.assertTrue(audit[0]['accepted'])

    def test_zero_seed_nonterminal_sidewall_stays(self):
        m = trimesh.creation.cylinder(radius=.2, height=1.5, sections=32)
        for _ in range(2):
            v, f = trimesh.remesh.subdivide(m.vertices, m.faces)
            m = trimesh.Trimesh(vertices=v, faces=f, process=False)
        s = (m.triangles_center[:, 2] + .75) / 1.5
        sides = (np.abs(m.face_normals[:,2]) < .1) & (s > .4) & (s < .6)
        found, audit = choose_zero_seed_endcaps(m, np.zeros(len(s),bool), sides, s)
        self.assertFalse(found.any())

    def test_zero_seed_inactive_if_seeds_present(self):
        m = trimesh.creation.box()
        n = len(m.faces)
        found, audit = choose_zero_seed_endcaps(m, np.eye(1,n,dtype=bool).ravel(),
                               np.ones(n,bool), np.linspace(0,1,n))
        self.assertFalse(found.any())
        self.assertEqual(audit, [])

    def test_input_manifest_and_evidence_alignment(self):
        with tempfile.TemporaryDirectory() as td:
            base = Path(td)
            sdf = base/'sdf'; udf = base/'udf'; out = base/'result'
            (sdf/'riggable_parts').mkdir(parents=True)
            (udf/'udf_cut_curve_local_0081'/'body').mkdir(parents=True)
            shape = trimesh.creation.box()
            shape.export(sdf/'riggable_fused_sdf_exterior.ply')
            shape.export(sdf/'riggable_parts'/'body.ply')
            (sdf/'root_leaves_ownership.json').write_text(json.dumps({'part_names':['body']}))
            (sdf/'riggable_fused_sdf_exterior_report.json').write_text('{}')
            n=len(shape.faces)
            np.savez_compressed(udf/'udf_cut_curve_local_0081'/'body'/'locality.npz',
                seed_faces=np.array([True]*3+[False]*(n-3)),
                selected_faces=np.array([True]*3+[False]*(n-3)),
                candidate_faces=np.array([True]*3+[False]*(n-3)),
                face_curve_arclength=np.linspace(0,1,n),
                face_centroids_xyz=shape.triangles_center.astype('float32'))
            result = generate_fused_cut(sdf_shape=sdf,udf_shape=udf,out=out,
                                        allow_zero_seed_caps=False)
            self.assertTrue((out/'riggable_fused_sdf_exterior_after_udf_cut.ply').exists())
            self.assertTrue(result['not_direct_fused_model_inference'])
            self.assertGreater(result['faces_removed'],0)
            with np.load(udf/'udf_cut_curve_local_0081'/'body'/'locality.npz') as z:
                self.assertEqual(int(np.count_nonzero(z['selected_faces'])),3)
            # Detection of stale UDF-run and wrong SDF source geometry.
            with np.load(udf/'udf_cut_curve_local_0081'/'body'/'locality.npz') as z:
                evidence={k:z[k] for k in z.files}
            evidence['face_centroids_xyz'] = evidence['face_centroids_xyz'] + 1
            np.savez_compressed(udf/'udf_cut_curve_local_0081'/'body'/'locality.npz', **evidence)
            with self.assertRaisesRegex(ValueError, 'do not match'):
                generate_fused_cut(sdf_shape=sdf, udf_shape=udf,
                                  out=base/'bad',allow_zero_seed_caps=False)

    def test_schema_and_wrapper_switch(self):
        text=(ROOT/'test_udf_sdf_cut_stitch.py').read_text()
        self.assertIn('udf_fused_exterior_0092', text)
        self.assertIn('if is_fused', text)
        self.assertIn('not is_fused and report', text)


if __name__=='__main__':
    unittest.main()

class IntegratedPipelineTests(unittest.TestCase):
    def test_0092_branch_skips_failed_loop_stitch(self):
        from unittest import mock
        import importlib.util
        import types
        spec=importlib.util.spec_from_file_location('pipeline_0092',str(ROOT/'test_udf_sdf_cut_stitch.py'))
        pipeline=importlib.util.module_from_spec(spec)
        # Test the pipeline handoff without importing the entire GPU model
        # stack from the production assembly/exporter modules.
        promoter=types.ModuleType('tools.promote_udf_0081_assembly')
        promoter.promote=lambda **kw: None
        stitcher=types.ModuleType('tools.udf_stitch_0084')
        stitcher.PARTS=('body',)
        stitcher._load_part=lambda *a: None
        stitcher.stitch_parts=lambda *a,**kw: None
        stitcher.preflight=lambda *a: {}
        with mock.patch.dict(sys.modules, {
             'tools.promote_udf_0081_assembly':promoter,
             'tools.udf_stitch_0084':stitcher}):
            spec.loader.exec_module(pipeline)
        with tempfile.TemporaryDirectory() as td:
            root=Path(td)
            final=root/'final'; final.mkdir()
            (final/'riggable_after_udf_cut_report.json').write_text('{}')
            sdf=root/'sdf'; udf=root/'udf'
            args=type('Args',(),{'yaml':'dummy', 'out':'dummy','shape':'shape'})()
            conf={'udf_full_pipeline_0084':{'enabled':True,'preview_vertex_tolerance':2e-5,
                  'stitch':{'require_success':True}},
                  'udf_fused_exterior_0092':{'enabled':True}}
            report={'status':'FUSED_UDF_EVIDENCE_TRANSFER_0092','output_ply':str(final/'fused.ply'),
                    'method':'test projection'}
            with mock.patch.object(pipeline,'_read_yaml',return_value=(root/'yaml', conf)), \
                 mock.patch.object(pipeline,'_locate_sdf_and_output',return_value=(sdf,udf,final)), \
                 mock.patch.object(pipeline,'_run_inference'), \
                 mock.patch.object(pipeline,'promote'), \
                 mock.patch.object(pipeline,'generate_fused_cut',return_value=report), \
                 mock.patch.object(pipeline,'_export_stitch') as stitch:
                summary=pipeline.execute(args,[])
                stitch.assert_not_called()
                self.assertEqual(summary['status'],'FUSED_UDF_EVIDENCE_TRANSFER_0092')
                self.assertIsNone(summary['stitched_ply'])
                self.assertEqual(summary['fused_sdf_cut_ply'],str(final/'fused.ply'))
                payload=json.loads((final/'riggable_after_udf_cut_report.json').read_text())
                self.assertIn('fused_sdf_cut_0092',payload)
