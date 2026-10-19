"""0048 candidate-ROI validation: exact excluded-voxel audit and fail-closed guards."""
import unittest
import numpy as np
from ngc.handle_utils.interval_audit import audit_candidate_subset
from ngc.curve_handle import CurveHandle
from utils.mc_utils import MCGrid


class StraightCore:
    def interpolate(self, ss, **kwargs):
        ss=np.asarray(ss, dtype=float)
        if kwargs.get('points', True) is False:
            if kwargs.get('radius', True):
                return {'radius':np.full((len(ss),2),.18)}
            return {'frame':np.tile(np.eye(3),(len(ss),1,1))}
        return {'points':np.stack((1.6*ss-.8,np.zeros_like(ss),np.zeros_like(ss)),axis=1),
                'frame':np.tile(np.eye(3),(len(ss),1,1)),
                'radius':np.full((len(ss),2),.18)}

    def calc_x_radius(self,ss):
        return np.full(len(ss),.21)

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

    def test_gate_envelope_includes_tangent_endpoint_acceptance(self):
        from ngc.handle_utils.interval_envelope import localizer_gate_envelope
        lower,upper=localizer_gate_envelope(self.curve.core,.3,.4,self.args,self.grid)
        # Source segment center starts at -.32, but x-radius accepts samples
        # farther left than the shortened cylinder's truncated endcap.
        self.assertLess(lower[0], -.5)
        self.assertGreater(upper[0], -.01)

    def test_real_x_radius_one_includes_far_endpoint_region(self):
        from ngc.handle_utils.interval_envelope import localizer_gate_envelope
        old=self.curve.core.calc_x_radius
        try:
            # PWLACurve.calc_x_radius currently returns 1.0 for every sample.
            self.curve.core.calc_x_radius=lambda s: np.ones(len(s))
            lower,upper=localizer_gate_envelope(self.curve.core,.3,.4,self.args,self.grid)
            # The interval's left endpoint is x=-.32. Valid ellipsoid support
            # reaches beyond x=-1, rather than stopping at the cylinder cap.
            self.assertLess(lower[0], -1.0)
            self.assertGreater(upper[0], .6)
        finally:
            self.curve.core.calc_x_radius=old

    def test_world_aabb_clips_original_union_without_reordering(self):
        from ngc.handle_utils.bounded_filter import iter_bbox_kidx
        c=self.curve.core
        samples=c.build_adapt_control_field(np.linspace(0,1,45))
        cylinder=self.curve._CurveHandle__gen_cyl_mesh({
            **samples,'thetas':2*np.pi*np.linspace(0,1,16,endpoint=False)})
        full=np.concatenate(list(iter_bbox_kidx(cylinder.cylinder_samples,self.grid,batch_size=31)))
        clip=np.concatenate(list(iter_bbox_kidx(
            cylinder.cylinder_samples,self.grid,batch_size=31,
            clip_world_aabb=(np.array([-.4,-.4,-.4]), np.array([.2,.4,.4])))))
        self.assertLess(len(clip),len(full))
        np.testing.assert_array_equal(clip,full[np.isin(full,clip)])

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
