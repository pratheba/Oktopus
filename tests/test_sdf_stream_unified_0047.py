"""0047 shared rigid/wrap writer, seam coordinates and fail-closed routing."""
import tempfile
import unittest

import numpy as np

from app.sdf_stream_evaluator import SDFStreamPiece
from app.sdf_stream_modes import configure_unified_sdf_stream
from app.dcsdd_stream_one_pass import OnePassBootStreamWriter
from app.dcsdd_stream_replay import compose_replay_chunk


def dataset(n):
    coords = np.linspace(.1, .9, n)
    local = np.stack((coords, np.linspace(.02, .5, n),
                      np.linspace(.1, .2, n)), axis=1)
    acc = {'coords': coords, 'samples_local': local}
    from app.dcsdd_stream_boot import INFERENCE_KEYS
    for key in INFERENCE_KEYS:
        if key not in acc:
            acc[key] = np.full(n, .15, dtype=np.float64)
    return acc


def infer(batch):
    x = batch['coords'].reshape(-1)
    local = batch['samples_local']
    return {'dist': x*.1 + local[:, 1]*.2 - .5,
            'dist_base': x*.05 + local[:, 2]*.3 - .6,
            'dist_detail': x*.02}


class SharedStreamTests(unittest.TestCase):
    def test_identical_rigid_and_wrap_prediction_and_replay(self):
        acc = dataset(87)
        with tempfile.TemporaryDirectory() as root:
            import os
            os.mkdir(root + '/rigid')
            os.mkdir(root + '/wrap')
            a = SDFStreamPiece(batch_size=13, outdir=root+'/rigid', inference=infer)
            b = SDFStreamPiece(batch_size=13, outdir=root+'/wrap', inference=infer,
                               record_avatar_source=True)
            for start in range(0, 87, 11):
                end = min(start+11,87)
                part = {k: v[start:end] for k,v in acc.items()}
                idx = np.arange(start, end, dtype=np.int64)*3
                a.push(part, idx)
                b.push(part, idx, {'coords': np.arange(start,end)*.011})
            pa = a.finish(expected_accepted=87, fallback=1.2)
            pb = b.finish(expected_accepted=87, fallback=1.2)
            for k in pa:
                np.testing.assert_array_equal(pa[k],pb[k])
            self.assertEqual(a.writer.batch_sizes,b.writer.batch_sizes)
            self.assertEqual(a.writer.batch_sizes[:-1], [13]*6)
            for k in a.writer.raw:
                np.testing.assert_array_equal(a.writer.raw[k],b.writer.raw[k])
            np.testing.assert_array_equal(a.replay.records,b.replay.records)
            np.testing.assert_allclose(b.source_s, np.arange(87)*.011, rtol=0, atol=0)
            va = a.compose_values(sigma=.03, rho_limit=3.,positive_value=1.,chunk_size=17)
            vb = b.compose_values(sigma=.03, rho_limit=3.,positive_value=1.,chunk_size=17)
            np.testing.assert_array_equal(va,vb)
            # Preserve existing global-calibration + replay composition bitwise.
            for start,end,records in a.replay.iter_chunks(17):
                expected = compose_replay_chunk(
                    {k:v[start:end] for k,v in a.writer.raw.items()},
                    records['coord'],records['rho'],pa,
                    rho_limit=3.,positive_value=1.,sigma=.03)[0]
                np.testing.assert_array_equal(va[start:end], expected)

    def test_rejects_mismatched_source_or_duplicate_index(self):
        with tempfile.TemporaryDirectory() as root:
            s = SDFStreamPiece(batch_size=5,outdir=root,inference=infer,
                               record_avatar_source=True)
            with self.assertRaisesRegex(ValueError,'avatar source'):
                s.push(dataset(3),np.arange(3))
            with self.assertRaisesRegex(ValueError,'seam coordinate'):
                s.push(dataset(3),np.arange(3),{'coords':np.arange(2)})
            s.push(dataset(3),np.arange(3),{'coords':np.arange(3)})
            with self.assertRaisesRegex(ValueError,'unique and increasing'):
                s.push(dataset(3),np.arange(2,5),{'coords':np.arange(3)})
            s.finish(expected_accepted=3,fallback=1.2)
            with self.assertRaisesRegex(ValueError,'finalized'):
                s.push(dataset(3),np.arange(3))

    def test_rigid_routing(self):
        item={'mode':'direct','rigid_radius':True,'wrap_radius':False,
              'use_accessory_support_clamp':True}
        out=configure_unified_sdf_stream(
            {'sdf_stream_unified':True, 'combine_adaptations':False},
            [item], extractor='ours')
        self.assertTrue(out['dcsdd_stream_one_pass'])
        self.assertTrue(out['dcsdd_stream_replay_metadata'])
        self.assertTrue(out['dcsdd_crop_grid'])
        self.assertFalse(out.get('sdf_stream_combined_wrap',False))

    def test_combined_wrap_routing(self):
        item={'mode':'direct','wrap_radius':True,'rigid_radius':False}
        out=configure_unified_sdf_stream(
            {'sdf_stream_unified':True, 'combine_adaptations':True,
             'export_individual_meshes':False},
            [item,item], extractor='ours')
        self.assertTrue(out['sdf_stream_combined_wrap'])
        self.assertTrue(out['sdf_optimize_zero_angular_wrap'])
        self.assertFalse(out.get('dcsdd_stream_one_pass',False))

    def test_unsupported_paths_rejected(self):
        rigid={'rigid_radius':True,'mode':'direct','use_accessory_support_clamp':True}
        wrap={'wrap_radius':True,'mode':'direct'}
        base={'sdf_stream_unified':True,'combine_adaptations':True,
              'export_individual_meshes':False}
        for items,extra in (([rigid,wrap],{}),([wrap,wrap],{'root_leaves':{'root':'body'}})):
            with self.subTest(items=items,extra=extra):
                with self.assertRaises(ValueError):
                    configure_unified_sdf_stream({**base,**extra}, items, extractor='ours')
        with self.assertRaisesRegex(ValueError,'DCSDD'):
            configure_unified_sdf_stream({**base}, [wrap,wrap], extractor='marching_cubes')
        with self.assertRaisesRegex(ValueError,'accepted'):
            configure_unified_sdf_stream({'sdf_stream_unified':True,'dcsdd_stream_accepted_count':1},
                                         [rigid], extractor='ours')

if __name__=='__main__':
    unittest.main()
