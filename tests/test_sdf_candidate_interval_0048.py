"""0048 candidate-ROI validation: exact excluded-voxel audit and fail-closed guards."""
import unittest
import numpy as np
from ngc.handle_utils.interval_audit import audit_candidate_subset
from ngc.curve_handle import CurveHandle
from utils.mc_utils import MCGrid


class StraightCore:
    def build_adapt_control_field(self, ss, **kwargs):
        points = np.stack((1.6*np.asarray(ss)-.8, np.zeros_like(ss), np.zeros_like(ss)),axis=1)
        return {'points':points,
                'frame':np.tile(np.eye(3),(len(ss),1,1)),
                'radius':np.full((len(ss),2),.18)}

    def localize_samples_adapt(self, pts, arg):
        # Pointwise surrogate with accepted points safely within the interval
        # gate; the original coordinate mapping remains the localizer's job.
        idx = np.flatnonzero((pts[:,0] > -.26) & (pts[:,0] < -.18)
                             & (pts[:,1]**2+pts[:,2]**2 < .07**2))
        return {'coords':pts[idx,0]}, {'coords':pts[idx,0]}, idx


class StraightHandle(CurveHandle):
    def __init__(self):
        self.core=StraightCore()


class IntervalAuditTests(unittest.TestCase):
    def setUp(self):
        self.grid=MCGrid({'reso':24,'size':1.2,'level':0.0})
        self.curve=StraightHandle()
        self.args={'mode':'direct','src_0':.3,'src_1':.4,
                   'use_adapt_control_filter_grid':True,
                   'rigid_radius':True,'dcsdd_bounded_filtering':True,
                   'dcsdd_filter_batch_size':19,
                   'sdf_localize_after_gate':True,
                   'adapt_debug_counts':False,
                   'sdf_candidate_interval_prune':True,
                   'sdf_candidate_interval_pad':.05}

    def test_audit_passes_then_filtered_results_match_original(self):
        def gather(roi):
            found=[]
            def cb(_acc,idx): found.append(np.asarray(idx).copy())
            c, n, _ = self.curve.filter_grid_adapt(self.grid, {
                **self.args, **roi, '_dcsdd_stream_callback':cb})
            indices=np.concatenate(found) if found else np.array([],dtype=np.int64)
            self.assertEqual(n,len(indices))
            return c,indices
        full, ref=gather({'sdf_candidate_interval_prune':False})
        narrowed, opt=gather({'sdf_candidate_interval_audit':True})
        self.assertLess(narrowed,full)
        np.testing.assert_array_equal(ref,opt)

    def test_fail_closed_when_a_pruned_candidate_would_be_accepted(self):
        old=self.curve.core.localize_samples_adapt
        try:
            self.curve.core.localize_samples_adapt=lambda pts,arg: ({},{}, np.arange(len(pts)))
            with self.assertRaisesRegex(ValueError,'UNSAFE'):
                self.curve.filter_grid_adapt(self.grid, {
                    **self.args, 'sdf_candidate_interval_audit':True,
                    '_dcsdd_stream_callback':lambda *_: None})
        finally:
            self.curve.core.localize_samples_adapt=old

    def test_rejects_missing_stream_or_audit_without_prune(self):
        with self.assertRaisesRegex(ValueError,'bounded direct streaming'):
            self.curve.filter_grid_adapt(self.grid,self.args)
        with self.assertRaisesRegex(ValueError,'requires interval pruning'):
            self.curve.filter_grid_adapt(self.grid,{
                **self.args,'sdf_candidate_interval_prune':False,
                'sdf_candidate_interval_audit':True})

    def test_subset_audit_detects_non_subset(self):
        with self.assertRaisesRegex(ValueError,'not found'):
            audit_candidate_subset([np.array([1,3,5])],[np.array([1,2])],
                                   lambda a: np.stack([a,a,a],axis=-1),
                                   lambda a: np.array([],dtype=np.int64))


if __name__=='__main__': unittest.main()
