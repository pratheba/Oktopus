import os, pickle
from time import perf_counter
import numpy as np
import os.path as op
import trimesh
from scipy.spatial import KDTree
from scipy.spatial.transform import Rotation, Slerp
from handle_utils import CylindersMesh
from scipy.ndimage import gaussian_filter1d
from PWLA_curve_handle import PWLACurve


n_sample_curve = 200
n_sample_circle = 120

class CurveHandle():
    """docstring for CurveHandle."""
    def __init__(self, arg=None):
        if arg is None:
            return
        
        self.set_curve(arg) 

    def update(self):
        self.core.update()
        #n_sample_curve = 100
        #n_sample_circle = 360
        self.cyl_mesh = self.gen_cylinder_mesh(n_sample_curve, n_sample_circle, radius_type='cylinder')

    def set_curve(self, arg):
        self.name = arg['name']
        self.idx = arg['idx']
        if not hasattr(self, 'core'):
            self.core = PWLACurve(arg)
        else:
            self.core.set_curve(arg)
        
        self.update()

    def set_points(self, points):
        # number of points remain same
        self.core.set_points(points)

    def set_resamples(self, points, z_axis0=None):
        self.core.set_resamples(points, z_axis0)

    def apply_action_arg(self, arg):
        if 'pose_file' in arg:
            pose_file = arg['pose_file']
            pose_data = pickle.load(open(pose_file, 'rb'))
            sk = pose_data[self.name]
            self.set_resamples(sk['vertices'])
            self.update()

        if 'rotation' in arg:
            rot_arg = arg['rotation']
            vec = rot_arg['vec']
            anchor_idx = rot_arg['anchor_idx']
            x_axis = self.core.key_frame[-1, 0]
            norm = np.linalg.norm(vec)
            vec = vec / norm
            quat = np.zeros(4)
            quat[:3] = np.cross(x_axis, vec)
            quat[3] = np.sqrt(np.sum(x_axis**2)* np.sum(vec**2)) + x_axis @ vec
            quat /= np.linalg.norm(quat)

            rot = Rotation.from_quat(quat)
            anchor = self.core.key_points[anchor_idx]
            self.apply_rotation(anchor, rot)
            self.update()

    def apply_rotation(self, anchor, rot):
        self.core.apply_rotation(anchor, rot)


    def radius_scaling(self, scales, coords):
        key_radius_scales = np.interp(self.core.key_ts, coords, scales)
        self.core.key_train_radius *= key_radius_scales[:, None]
        self.core.key_radius = self.core.key_train_radius
        if self.core.key_cylinder_radius is not None:
            self.core.key_cylinder_radius *= key_radius_scales[:, None]


    def rot_part(self, idx, axis, angle):
        anchor = self.core.key_points[idx]
        part_pts = self.core.key_points[idx:]
        part_zaxis = self.core.z_axis[idx:]
        rot = Rotation.from_euler(axis, angle, degrees=True)
        new_pts = rot.apply(part_pts - anchor) + anchor
        new_zaxis = rot.apply(part_zaxis)

        self.core.key_points[idx:] = new_pts
        self.core.z_axis[idx:] = new_zaxis
        self.core.flag_points = True
        return anchor, rot
    
    def stretch_part(self, front, back):
        idx1, offset1 = front
        idx2, offset2 = back

        self.core.key_points[idx1:] += offset1
        self.core.key_points[:(idx2+1)] += offset2
        # NOTE: do not update, since it will recalculate natural coords
        #n_sample_curve = 20
        #n_sample_circle = 12
        #n_sample_curve = 100
        #n_sample_circle = 360
        self.cyl_mesh = self.gen_cylinder_mesh(n_sample_curve, n_sample_circle, radius_type='cylinder')

    def rot_tilt(self, angles, coords):
        key_angles = np.interp(
            self.core.key_ts, coords, angles
        )
        a11 = np.sin(key_angles + np.pi/2)[:,None]
        a12 = np.cos(key_angles + np.pi/2)[:,None]
        a21 = np.sin(key_angles)[:,None]
        a22 = np.cos(key_angles)[:,None]
        
        kf = self.core.key_frame.copy()
        new_y = kf[:,1,:]*a11 + kf[:,2,:]*a12
        new_z = kf[:,1,:]*a21 + kf[:,2,:]*a22
        kf[:,1,:] = new_y
        kf[:,2,:] = new_z
        self.core.set_frame(kf)


    def apply_translation(self, offset):
        self.core.key_points += offset

    def clip_cylinder(self, t0=None, t1=None):
        if t0 is not None:
            idx0 = np.searchsorted(self.core.key_ts, t0)
            itp0 = self.core.interpolate([t0])
            p0,r0,f0 = itp0['points'], itp0['radius'], itp0['frame']
            key_ts = np.concatenate([[t0], self.core.key_ts[idx0+1:]])
            key_points = np.concatenate([p0, self.core.key_points[idx0+1:]], axis=0)
            key_radius = np.concatenate([r0, self.core.key_radius[idx0+1:]], axis=0)
            key_frame = np.concatenate([f0, self.core.key_frame[idx0+1:]], axis=0)
            self.core.key_ts = key_ts
            self.core.key_points = key_points
            self.core.key_radius = key_radius
            self.core.key_frame = key_frame
        
        if t1 is not None:
            idx1 = np.searchsorted(self.core.key_ts, t1)
            idx1 = min(idx1, self.core.key_ts.shape[0]-1)
            itp1 = self.core.interpolate([t1])
            p1,r1,f1 = itp1['points'], itp1['radius'], itp1['frame']
            key_ts = np.concatenate([self.core.key_ts[:idx1], [t1]])
            key_points = np.concatenate([self.core.key_points[:idx1], p1], axis=0)
            key_radius = np.concatenate([self.core.key_radius[:idx1], r1], axis=0)
            key_frame = np.concatenate([self.core.key_frame[:idx1], f1], axis=0)
            self.core.key_ts = key_ts
            self.core.key_points = key_points
            self.core.key_radius = key_radius
            self.core.key_frame = key_frame

        self.update()

    def get_end_data(self, side):
        return {
            'point': self.core.key_points[side],
            'radius': self.core.key_radius[side],
            'frame': self.core.key_frame[side],
        }

    def gen_cylinder_mesh(self, n_sample_curve, n_sample_circle, radius_type='cylinder'):
        # t0,t1 = [0., 1.]
        t0 = self.core.key_ts[0]
        t1 = self.core.key_ts[-1]
        ts = np.linspace(t0, t1, n_sample_curve)
        thetas = (2*np.pi)* np.linspace(0, 1, n_sample_circle, endpoint=False)

        intpl = self.core.interpolate(ts, radius_type=radius_type)
        intpl['thetas'] = thetas
        return self.__gen_cyl_mesh(intpl)

    def __gen_cyl_mesh(self, intpl):
        thetas = intpl['thetas']
        verts = intpl['points']
        yz_rs = intpl['radius']
        frames = intpl['frame']
        if 'level' in intpl:
            level = intpl['level']
        else:
            level = 0.

        theta_coo = np.asarray([np.cos(thetas), np.sin(thetas)])
        theta_coo = np.einsum('nj,ji->nji', yz_rs, theta_coo)
        cyl_pts = np.einsum('njk,nji->nik', frames[:,1:], theta_coo)
        
        # points of level set
        dists = np.linalg.norm(cyl_pts, axis=2)
        scales = (dists + level) / dists
        cyl_pts *= scales[..., None]

        cyl_pts += verts[:, None, :]

        v0, v1 = verts[0], verts[-1]

        cyl_mesh = CylindersMesh()
        vidx = cyl_mesh.add_cylinder(cyl_pts)

        cyl_mesh.add_cap(v0, vidx[0])
        cyl_mesh.add_cap(v1, vidx[-1], flip_face=True)
        #c_mesh = cyl_mesh.extract_mesh()
        #c_mesh.export('test_cylinder.ply')
        #exit()
        return cyl_mesh

    def generate_samples(self, num_samples):
        return self.core.generate_samples(num_samples)

    def localize_samples(self, samples, update_curve=False, update_radius=False, name=''):
        return self.core.localize_samples(samples, update_curve=update_curve, update_radius=update_radius, name=name)

    def localize_samples_test(self, name, samples):
        return self.core.localize_samples_test(name, samples)
    
    # def localize_samples_transition(self, samples, ts):
    #     return self.core.localize_samples_transition(samples, ts)

    def filter_grid(self, mc_grid):
        # find grid points around the cylinder
        self.core.update_coords()
        self.core.update_frame()
        samples, kidx = self.cyl_mesh.filter_grid(mc_grid)
        # calculate the neareast points and find inside points
        samples_data, inside = self.core.localize_samples(samples)
        kidx = kidx[inside]
        return samples_data, kidx
    
    def filter_grid_mix(self, mc_grid, mix_arg):
        # gen mixed cyl mesh 
        #ts = np.linspace(0., 1., 20)
        self.core.update_coords()
        self.core.update_frame()
        ts = np.linspace(0., 1., n_sample_curve)
        
        intpl = self.core.interpolate_mix(ts, mix_arg)
        #thetas = (2*np.pi)* np.linspace(0, 1, 12, endpoint=False)
        thetas = (2*np.pi)* np.linspace(0, 1, n_sample_circle, endpoint=False)
        intpl['thetas'] = thetas
        cyl_mesh = self.__gen_cyl_mesh(intpl)
        
        samples, kidx = cyl_mesh.filter_grid(mc_grid)
        samples_data, inside = self.core.localize_samples_mix(samples, mix_arg)
        kidx = kidx[inside]
        return samples_data, kidx
    
    def filter_grid_adapt_old(self, mc_grid, adapt_arg):
        samples, kidx = self.cyl_mesh.filter_grid(mc_grid)
        accessory_data, avatar_data, inside= self.core.localize_samples_adapt(samples, adapt_arg)
        kidx = kidx[inside]
        return accessory_data, avatar_data, kidx, inside

    def filter_grid_adapt_old2(self, mc_grid, adapt_arg):
        samples, kidx = self.cyl_mesh.filter_grid(mc_grid)
        #samples, kidx = mc_grid.generate_samples()

        mode = adapt_arg.get("mode", "direct")

        if mode == "direct":
            accessory_data, avatar_data, inside = self.core.localize_samples_adapt(samples, adapt_arg)
            #accessory_curve_handle = adapt_arg["accessory_curve_handle"]
            #accessory_data, kidx = accessory_curve_handle.filter_grid_on_runtime_support(
            #    mc_grid,
            #    runtime_support,
            #    norm=adapt_arg.get("infer_scale", 1.35),
            #)

            #inside = np.arange(kidx.shape[0])

        elif mode == "dependent":
            accessory_data, avatar_data, inside = self.core.localize_samples_dependent(samples, adapt_arg)

        elif mode == "dependent_split":
            accessory_data, avatar_data, inside = self.core.localize_samples_split_dependent(samples, adapt_arg)

        else:
            raise ValueError(f"Unknown adapt mode: {mode}")

        kidx = kidx[inside]
        return accessory_data, avatar_data, kidx, inside

    def filter_grid_adapt(self, mc_grid, adapt_arg):
        """
        Adaptation grid filtering.

        This is the FIRST support gate:
            MC grid -> candidate samples

        If this gate is too tight, bulky details/puffs are cut before
        localize_samples_adapt() and the neural SDF ever see them.

        Runtime knobs:
            avatar_filter_radius_scale/add/norm  : first gate
            avatar_cylinder_radius_scale/add     : fallback for first gate + second gate
            adapt_localize_norm                  : second gate in PWLA localize_samples_adapt

        High-resolution broad-phase optimization:
            adapt_filter_use_src_interval        : restrict the first cylinder gate to
                                                   a padded src_0/src_1 interval
            adapt_filter_src_interval_pad        : conservative parameter-space padding
                                                   on each side (default 0.10)

        The control field is still built on the full [0, 1] curve before it is
        sliced.  Therefore this optimization does not change the smoothing/control
        field itself; it only avoids sending obviously out-of-interval voxels to
        the expensive exact localizer.
        """

        filter_radius_scale = float(
            adapt_arg.get(
                "avatar_filter_radius_scale",
                adapt_arg.get("avatar_cylinder_radius_scale", 1.0),
            )
        )

        filter_radius_add = float(
            adapt_arg.get(
                "avatar_filter_radius_add",
                adapt_arg.get("avatar_cylinder_radius_add", 0.0),
            )
        )

        filter_norm = float(
            adapt_arg.get(
                "avatar_filter_norm",
                adapt_arg.get("infer_scale", 1.0),
            )
        )

        use_control_filter = bool(
            adapt_arg.get("use_adapt_control_filter_grid", False)
        )

        # localize_samples_adapt() already constrains direct projection to
        # src_0/src_1 by default.  At high resolution it is wasteful to build a
        # candidate cylinder over the entire avatar curve first.  Build the exact
        # same full control/interpolation field, then retain only a conservative
        # padded interval for the broad-phase cylinder.
        use_interval_projection = bool(
            adapt_arg.get("use_interval_projection", True)
        )
        # 0048: explicit experimental ROI optimization for shared streaming.
        # The old full-curve candidate set is available for exact localizer
        # audits before enabling pruning on a new adaptation/configuration.
        interval_prune = bool(adapt_arg.get('sdf_candidate_interval_prune', False))
        safe_prune_0094 = bool(adapt_arg.get('sdf_candidate_safe_prune_0094', False))
        safe_audit_0094 = bool(adapt_arg.get('sdf_candidate_safe_audit_0094', False))
        if safe_audit_0094 and not safe_prune_0094:
            raise ValueError('0094 exact audit requires certified ROI pruning')
        if safe_prune_0094 and interval_prune:
            raise ValueError('0094 certified ROI and 0048 sampled ROI are mutually exclusive')
        interval_audit = bool(adapt_arg.get('sdf_candidate_interval_audit', False))
        piecewise_0057 = bool(adapt_arg.get('sdf_candidate_piecewise_0057', False))
        piecewise_bins_0057 = int(adapt_arg.get('sdf_candidate_piecewise_bins_0057', 12))
        piecewise_verified_0057 = bool(adapt_arg.get('sdf_candidate_piecewise_verified_0057', False))
        if piecewise_0057 and (not interval_prune or
                              not (interval_audit or piecewise_verified_0057)):
            raise ValueError('0057 piecewise gate requires interval pruning and an exact audit or explicit previously verified configuration')
        if piecewise_0057 and not 2 <= piecewise_bins_0057 <= 32:
            raise ValueError('0057 interval bins must be from 2 to 32')
        if interval_prune and float(adapt_arg.get(
                'sdf_candidate_interval_pad',
                adapt_arg.get('adapt_filter_src_interval_pad', 0.10))) < 0.0:
            raise ValueError('0048 interval pad must be nonnegative')
        if interval_audit and not interval_prune:
            raise ValueError('0048 interval audit requires interval pruning')
        # 0052: the correction-rich puffer path can prune candidates WITHOUT
        # streaming inference.  Only its explicitly validated 0051 minimal
        # stage may opt in: accepted rows are still staged in original order,
        # and global support-quantile corrections are preserved downstream.
        # Other non-callback consumers retain the 0048 streaming-only guard.
        puffer_staged_roi = bool(adapt_arg.get('_sdf_puffer_minimal_staging', False)
                                 or adapt_arg.get('_sdf_puffer_direct_predictions_0053', False))
        puffer_staged_roi &= (adapt_arg.get('_dcsdd_stream_callback') is None
                              or bool(adapt_arg.get('_sdf_puffer_direct_predictions_0053', False)))
        puffer_staged_roi &= bool(adapt_arg.get('dcsdd_bounded_filtering', False))
        puffer_staged_roi &= (bool(adapt_arg.get('rigid_radius', False))
                              != bool(adapt_arg.get('wrap_radius', False)))
        if interval_prune and (
                not adapt_arg.get('dcsdd_bounded_filtering', False)
                or (adapt_arg.get('_dcsdd_stream_callback') is None
                    and not puffer_staged_roi)
                or not use_interval_projection
                or adapt_arg.get('mode', 'direct') != 'direct'):
            raise ValueError('0048 interval pruning requires bounded direct streaming '
                             'or validated puffer minimal staging with interval projection')
        use_src_interval_filter = (
            bool(adapt_arg.get("adapt_filter_use_src_interval", False))
            or interval_prune
        ) and use_interval_projection

        s_full = np.linspace(0.0, 1.0, n_sample_curve)
        interval_lo = 0.0
        interval_hi = 1.0
        interval_slice = slice(None)

        if use_src_interval_filter:
            src_0 = float(adapt_arg["src_0"])
            src_1 = float(adapt_arg["src_1"])
            src_lo = min(src_0, src_1)
            src_hi = max(src_0, src_1)
            interval_pad = max(
                0.0,
                float(adapt_arg.get(
                    "sdf_candidate_interval_pad",
                    adapt_arg.get("adapt_filter_src_interval_pad", 0.10))),
            )
            interval_lo = max(0.0, src_lo - interval_pad)
            interval_hi = min(1.0, src_hi + interval_pad)

            # Include one dense curve sample outside each requested bound when
            # possible.  This makes the broad phase conservative at the caps and
            # avoids numerical loss from sampling the interval boundary.
            i0 = max(
                0,
                int(np.searchsorted(s_full, interval_lo, side="left")) - 1,
            )
            i1 = min(
                len(s_full),
                int(np.searchsorted(s_full, interval_hi, side="right")) + 1,
            )
            if i1 - i0 >= 2:
                interval_slice = slice(i0, i1)
                interval_lo = float(s_full[i0])
                interval_hi = float(s_full[i1 - 1])
            else:
                # Defensive fallback: never create a degenerate cylinder.
                use_src_interval_filter = False
                interval_lo = 0.0
                interval_hi = 1.0
                interval_slice = slice(None)

        bounded = bool(adapt_arg.get("dcsdd_bounded_filtering", False))
        if safe_prune_0094 and (not bounded or adapt_arg.get('mode', 'direct') != 'direct'):
            raise ValueError('0094 certified ROI requires bounded direct localization')
        # 0046: the callback-based bounded iterator also supports cone's direct
        # wrap mapping, but ONLY via the explicitly guarded combined-wrap path.
        # Other bounded users retain the previous rigid-only restriction.
        cone_bounded = bool(adapt_arg.get('sdf_stream_combined_wrap', False))
        # 0050a: the root/leaf puffer pipeline needs the original *staged*
        # bounded localizer for its wrap body.  Unlike the cone callback path,
        # keep every avatar/accessory field: avatar cutting and local snug use
        # them after localization.  Only action_part_adapt sets this opt-in for
        # a plain direct wrap section in an explicitly sparse root/leaf run.
        puffer_wrap_staging = bool(adapt_arg.get('_sdf_puffer_bounded_wrap_staging', False))
        puffer_direct_0053 = bool(adapt_arg.get('_sdf_puffer_direct_predictions_0053', False))
        if puffer_direct_0053 and (
            not bounded or not puffer_staged_roi
            or adapt_arg.get('_dcsdd_stream_callback') is None
            or adapt_arg.get('mode', 'direct') != 'direct'
        ):
            raise ValueError('0053 prediction callback requires bounded direct puffer')
        if puffer_wrap_staging and (
            not bounded
            or adapt_arg.get('mode', 'direct') != 'direct'
            or not adapt_arg.get('wrap_radius', False)
            or adapt_arg.get('rigid_radius', False)
            or cone_bounded
            or (adapt_arg.get('_dcsdd_stream_callback') is not None
                and not puffer_direct_0053)
        ):
            raise ValueError('0050a puffer bounded wrap requires staged direct wrap (unless validated 0053 callback)')
        if bounded and (
            adapt_arg.get('mode', 'direct') != 'direct'
            or (cone_bounded and not adapt_arg.get('wrap_radius', False))
            or (not cone_bounded and not puffer_wrap_staging and (
                not adapt_arg.get('rigid_radius', False)
                or adapt_arg.get('wrap_radius', False)))
        ):
            raise ValueError('bounded filtering supports rigid direct, cone callback wrap, or guarded puffer wrap staging')
        # Some adaptation branches use whole-batch quantiles/medians; their
        # behavior must not silently change with chunk boundaries.
        if bounded and (
            adapt_arg.get("wrap_to_rigid", False)
            or adapt_arg.get("wrap_blend_rigid", False)
            or adapt_arg.get("unified_wrap", False)
            or adapt_arg.get("use_free_profile_scale", False)
            or adapt_arg.get("use_attach_to_free_blend", False)
            or adapt_arg.get("use_wrap_to_rigid_blend", False)
            or adapt_arg.get("free_anchor_to_release_scale", False)
        ):
            raise ValueError("bounded filtering does not support global-statistic wrap blending")
        fast_bbox = bool(adapt_arg.get("dcsdd_fast_bbox_marking", False)) or bounded
        broad_t0 = perf_counter()

        full_cylinder_for_audit = None
        if use_control_filter:
            # IMPORTANT: build the control field on full [0, 1] exactly as before.
            # Slicing only after this point preserves the previous smoothing and
            # frame reconstruction; only the broad-phase support extent changes.
            ctrl = self.core.build_adapt_control_field(
                s_full,
                n_control=int(adapt_arg.get("adapt_control_n_keypoints", 12)),
                smooth_points_sigma=float(
                    adapt_arg.get("avatar_control_smooth_points_sigma", 8.0)
                ),
                smooth_radius_sigma=float(
                    adapt_arg.get("avatar_control_smooth_radius_sigma", 0.0)
                ),
                rebuild_frames=bool(
                    adapt_arg.get("avatar_control_rebuild_frames", True)
                ),
                preserve_endpoints=bool(
                    adapt_arg.get("adapt_control_preserve_endpoints", True)
                ),
                radius_type="cylinder",
            )

            radius = np.asarray(ctrl["radius"], dtype=np.float64)
            radius = radius * filter_radius_scale + filter_radius_add
            radius = radius * filter_norm

            intpl = {
                "points": np.asarray(ctrl["points"])[interval_slice],
                "frame": np.asarray(ctrl["frame"])[interval_slice],
                "radius": radius[interval_slice],
                "thetas": (2.0 * np.pi)
                * np.linspace(0, 1, n_sample_circle, endpoint=False),
            }

            cyl_mesh = self.__gen_cyl_mesh(intpl)
            if interval_prune:
                full_cylinder_for_audit = self.__gen_cyl_mesh({
                    "points": np.asarray(ctrl["points"]),
                    "frame": np.asarray(ctrl["frame"]),
                    "radius": radius,
                    "thetas": intpl["thetas"],
                })
            samples, kidx = (None, None) if bounded else cyl_mesh.filter_grid(mc_grid, fast_bbox=fast_bbox)

        else:
            needs_runtime_filter = (
                abs(filter_radius_scale - 1.0) > 1e-12
                or abs(filter_radius_add) > 1e-12
                or abs(filter_norm - 1.0) > 1e-12
            )

            if needs_runtime_filter or use_src_interval_filter:
                intpl = self.core.interpolate(
                    s_full,
                    radius_type="cylinder",
                    radius=True,
                    frame=True,
                )

                radius = np.asarray(intpl["radius"], dtype=np.float64)
                radius = radius * filter_radius_scale + filter_radius_add
                radius = radius * filter_norm

                intpl_full = intpl
                intpl = {
                    "points": np.asarray(intpl["points"])[interval_slice],
                    "frame": np.asarray(intpl["frame"])[interval_slice],
                    "radius": radius[interval_slice],
                    "thetas": (2.0 * np.pi) * np.linspace(
                        0,
                        1,
                        n_sample_circle,
                        endpoint=False,
                    ),
                }

                cyl_mesh = self.__gen_cyl_mesh(intpl)
                if interval_prune:
                    full_cylinder_for_audit = self.__gen_cyl_mesh({
                        "points": np.asarray(intpl_full["points"]),
                        "frame": np.asarray(intpl_full["frame"]),
                        "radius": radius,
                        "thetas": intpl["thetas"],
                    })
                samples, kidx = (None, None) if bounded else cyl_mesh.filter_grid(mc_grid, fast_bbox=fast_bbox)

            else:
                cyl_mesh = self.cyl_mesh
                samples, kidx = (None, None) if bounded else cyl_mesh.filter_grid(mc_grid, fast_bbox=fast_bbox)

        if safe_prune_0094:
            from ngc.handle_utils.safe_candidate_roi_0094 import certified_source_boxes
            from ngc.handle_utils.interval_envelope import PiecewiseGateEnvelopeCandidates
            boxes_0094 = certified_source_boxes(
                self.core, adapt_arg['src_0'], adapt_arg['src_1'], adapt_arg, mc_grid)
            if boxes_0094 is None:
                print('[sdf_candidate_safe_0094] fallback=original_broad_phase '
                      'reason=uncertified_curve_or_gate', flush=True)
            else:
                # Clip ORIGINAL broad-phase boxes. Do not use a truncated
                # cylinder, which loses valid along-tangent endpoint samples.
                if use_src_interval_filter:
                    raise ValueError('0094 requires full-cylinder broad phase; '
                                     'disable adapt_filter_use_src_interval')
                original_cylinder_0094 = cyl_mesh
                cyl_mesh = PiecewiseGateEnvelopeCandidates(original_cylinder_0094, boxes_0094)
                if safe_audit_0094:
                    from ngc.handle_utils.interval_audit import audit_candidate_subset
                    audit_arg = dict(adapt_arg)
                    audit_arg.pop('_dcsdd_stream_callback', None)
                    audit_arg['adapt_debug_counts'] = False
                    audit_batch = int(adapt_arg.get('dcsdd_filter_batch_size', 262144))
                    proof_0094 = audit_candidate_subset(
                        original_cylinder_0094.iter_filter_grid_indices(
                            mc_grid, batch_size=audit_batch),
                        cyl_mesh.iter_filter_grid_indices(
                            mc_grid, batch_size=audit_batch),
                        mc_grid.idx2pts,
                        lambda points: self.core.localize_samples_adapt(
                            points, audit_arg)[2])
                    print('[sdf_candidate_safe_0094 audit] full=',
                          proof_0094['full_candidates'], 'narrow=',
                          proof_0094['narrow_candidates'], 'excluded=',
                          proof_0094['excluded_candidates'], 'missing_accepted=',
                          proof_0094['missing_accepted'], flush=True)
                print('[sdf_candidate_safe_0094] enabled=True boxes=',
                      len(boxes_0094), 'radius_modes=rigid_or_wrap', flush=True)
        if interval_prune:
            # 0048a: selecting only the truncated cylinder does not account
            # for the along-tangent x_radius in localize_samples(). Near the
            # interval endpoints valid points may lie OUTSIDE that tube.
            # Instead intersect the original full-cylinder bbox union with
            # the exact gate's sampled, tangent-aware world-space envelope.
            from ngc.handle_utils.interval_envelope import (
                localizer_gate_envelope, GateEnvelopeCandidates,
            )
            if full_cylinder_for_audit is None:
                raise RuntimeError('0048a requires the original full broad phase')
            gate_lo, gate_hi = localizer_gate_envelope(
                self.core, adapt_arg['src_0'], adapt_arg['src_1'],
                adapt_arg, mc_grid)
            if piecewise_0057:
                from ngc.handle_utils.interval_envelope import (
                    localizer_piecewise_gate_envelopes,
                    PiecewiseGateEnvelopeCandidates,
                )
                sub_envelopes = localizer_piecewise_gate_envelopes(
                    self.core, adapt_arg['src_0'], adapt_arg['src_1'],
                    adapt_arg, mc_grid, bins=piecewise_bins_0057)
                cyl_mesh = PiecewiseGateEnvelopeCandidates(
                    full_cylinder_for_audit, sub_envelopes)
                print('[sdf_candidate_piecewise_0057] enabled=True',
                      'bins=', piecewise_bins_0057,
                      'envelope_boxes=', len(sub_envelopes),
                      'audit=', interval_audit,
                      'verified_without_audit=', piecewise_verified_0057 and not interval_audit,
                      flush=True)
            else:
                cyl_mesh = GateEnvelopeCandidates(full_cylinder_for_audit,
                                                  gate_lo, gate_hi)
            print('[sdf_candidate_interval_0048a] world_gate_envelope=',
                  gate_lo.tolist(), gate_hi.tolist(), flush=True)
        if interval_prune and puffer_staged_roi:
            print('[sdf_puffer_0052] candidate_roi=staged_puffer',
                  'mode=', 'audit' if interval_audit else 'prune',
                  'global_support_calibration=preserved', flush=True)
        if interval_prune:
            print('[sdf_candidate_interval_0048] mode=',
                  'audit' if interval_audit else 'prune',
                  'src=', (float(adapt_arg['src_0']), float(adapt_arg['src_1'])),
                  'source_envelope=', (interval_lo, interval_hi),
                  'candidate_mode=localizer_xyz_ellipsoid',
                  'interval_pad_used_for_roi=False',
                  flush=True)
        if interval_audit:
            if full_cylinder_for_audit is None:
                raise RuntimeError('0048 audit requested without a full-curve cylinder')
            from ngc.handle_utils.interval_audit import audit_candidate_subset
            batch_audit = int(adapt_arg.get('dcsdd_filter_batch_size', 262144))
            quiet_arg = dict(adapt_arg)
            quiet_arg['adapt_debug_counts'] = False
            quiet_arg.pop('_dcsdd_stream_callback', None)

            def _localize_excluded(points):
                _, _, inside = self.core.localize_samples_adapt(points, quiet_arg)
                return inside

            proof = audit_candidate_subset(
                full_cylinder_for_audit.iter_filter_grid_indices(
                    mc_grid, batch_size=batch_audit),
                cyl_mesh.iter_filter_grid_indices(mc_grid, batch_size=batch_audit),
                mc_grid.idx2pts, _localize_excluded)
            if piecewise_0057:
                print('[sdf_candidate_piecewise_0057 audit] full=',
                      proof['full_candidates'],
                      'candidates=', proof['narrow_candidates'],
                      'pruned=', proof['excluded_candidates'],
                      'missing_accepted=', proof['missing_accepted'],
                      'audit_s=', f"{proof['elapsed_s']:.3f}",
                      'candidate_reduction_pct=',
                      f"{100.0 * proof['excluded_candidates'] / max(1, proof['full_candidates']):.2f}",
                      flush=True)
            print('[sdf_candidate_interval_0048 audit] full=', proof['full_candidates'],
                  'pruned=', proof['narrow_candidates'],
                  'excluded=', proof['excluded_candidates'],
                  'missing_accepted=', proof['missing_accepted'],
                  'audit_s=', f"{proof['elapsed_s']:.3f}",
                  'candidate_reduction_pct=',
                  f"{100.0 * proof['excluded_candidates'] / max(1, proof['full_candidates']):.2f}",
                  flush=True)
        sphere_0058 = bool(adapt_arg.get('sdf_localize_sphere_0058', False))
        sphere_audit_0058 = bool(adapt_arg.get('sdf_localize_sphere_audit_0058', False))
        if sphere_0058:
            if not (puffer_direct_0053 and puffer_wrap_staging and interval_prune
                    and use_interval_projection and
                    int(adapt_arg.get('loop_projection_candidates', 1)) == 1
                    and bool(adapt_arg.get('sdf_localize_after_gate', False))):
                raise ValueError('0058 sphere gate restricted to tested direct puffer single-projection streaming path')
            if not (sphere_audit_0058 or
                    adapt_arg.get('sdf_localize_sphere_verified_0058', False)):
                raise ValueError('0058 sphere gate requires audit or verified profile setting')
            print('[sdf_localize_sphere_0058] enabled=True audit=', sphere_audit_0058,
                  'verified_without_audit=', bool(adapt_arg.get('sdf_localize_sphere_verified_0058', False))
                  and not sphere_audit_0058, flush=True)
        reuse_0059 = bool(adapt_arg.get('sdf_localize_reuse_0059', False))
        reuse_audit_0059 = bool(adapt_arg.get('sdf_localize_reuse_audit_0059', False))
        if reuse_0059:
            if not (sphere_0058 and puffer_direct_0053 and puffer_wrap_staging
                    and interval_prune and use_interval_projection
                    and int(adapt_arg.get('loop_projection_candidates', 1)) == 1
                    and bool(adapt_arg.get('sdf_localize_after_gate', False))
                    and not adapt_arg.get('use_adapt_control_field', False)
                    and not adapt_arg.get('use_adapt_train_radius', False)):
                raise ValueError('0059 reuse only supports tested direct puffer '
                                 'with unchanged avatar frames and train radii')
            if not (reuse_audit_0059 or adapt_arg.get('sdf_localize_reuse_verified_0059', False)):
                raise ValueError('0059 reuse requires parity audit or verified profile')
            print('[sdf_localize_reuse_0059] enabled=True audit=', reuse_audit_0059,
                  'verified_without_audit=', bool(adapt_arg.get('sdf_localize_reuse_verified_0059', False))
                  and not reuse_audit_0059, flush=True)
        if bounded:
            # Optional compact staging is deliberately limited to the independent
            # rigid-radius SDF inference path. Its six fields are exactly those
            # consumed by model_withgrid._pack_common / support clamp.
            # 0036: no localized record staging. The callback consumes each
            # accepted batch and must not retain its arrays unless intentional.
            stream_callback = adapt_arg.get('_dcsdd_stream_callback')
            if stream_callback is not None:
                batch_size = int(adapt_arg.get('dcsdd_filter_batch_size', 262144))
                if batch_size <= 0:
                    raise ValueError('dcsdd_filter_batch_size must be positive')
                n_candidate = 0
                n_accepted = 0
                localize_compute_s = 0.0
                profile_sdf = bool(adapt_arg.get('sdf_localize_after_gate', False))
                stage_profile_0060 = bool(adapt_arg.get('sdf_localize_profile_0060', False))
                stage_before_0060 = (
                    {k: (v[0], v[1]) for k, v in
                     getattr(self.core, '_localize_stages_0060', {}).items()}
                    if stage_profile_0060 else None
                )
                iterator_s_0060 = 0.0
                audit_s_0060 = 0.0
                select_s_0060 = 0.0
                batch_total_0060 = []
                batch_count_0060 = 0
                previous_batch_end_0060 = perf_counter()
                stream_t0 = perf_counter()
                point_decode_s = 0.0
                callback_s = 0.0
                quiet_arg = dict(adapt_arg)
                quiet_arg['adapt_debug_counts'] = False
                sphere_previous_count_0058 = int(getattr(self.core, '_projection_sphere_rejected_0058', 0))
                core_previous_s_0059 = float(getattr(self.core, '_localize_core_s_0059', 0.0))
                avatar_previous_s_0059 = float(getattr(self.core, '_localize_avatar_s_0059', 0.0))
                for candidate_idx in cyl_mesh.iter_filter_grid_indices(mc_grid, batch_size=batch_size):
                    if stage_profile_0060:
                        batch_start_0060 = perf_counter()
                        iterator_s_0060 += batch_start_0060 - previous_batch_end_0060
                        batch_count_0060 += 1
                    n_candidate += len(candidate_idx)
                    point_t0 = perf_counter() if profile_sdf else None
                    sample_chunk = mc_grid.idx2pts(candidate_idx)
                    if profile_sdf:
                        point_decode_s += perf_counter() - point_t0
                    clock = perf_counter()
                    acc_chunk, av_chunk, accepted_local = self.core.localize_samples_adapt(
                        sample_chunk, quiet_arg)
                    localize_compute_s += perf_counter() - clock
                    if stage_profile_0060:
                        after_localize_0060 = perf_counter()
                    if sphere_audit_0058 or reuse_audit_0059:
                        reference_arg = dict(quiet_arg)
                        if sphere_audit_0058:
                            reference_arg['sdf_localize_sphere_0058'] = False
                        reference_arg['sdf_localize_reuse_0059'] = False
                        ref_acc, ref_av, ref_inside = self.core.localize_samples_adapt(
                            sample_chunk, reference_arg)
                        if not np.array_equal(accepted_local, ref_inside):
                            raise ValueError('0058/0059 UNSAFE: optimization changed accepted voxel indices')
                        for label, actual, expected in (('accessory', acc_chunk, ref_acc),
                                                        ('avatar', av_chunk, ref_av)):
                            if actual.keys() != expected.keys():
                                raise ValueError('0058/0059 UNSAFE: changed ' + label + ' field keys')
                            for k in actual:
                                if not np.array_equal(actual[k], expected[k], equal_nan=True):
                                    raise ValueError('0058/0059 UNSAFE: changed ' + label + '.' + k)
                    if stage_profile_0060:
                        after_audit_0060 = perf_counter()
                        audit_s_0060 += after_audit_0060 - after_localize_0060
                    selected = candidate_idx[accepted_local]
                    n_accepted += len(selected)
                    if stage_profile_0060:
                        select_s_0060 += perf_counter() - after_audit_0060
                    if len(selected):
                        callback_t0 = perf_counter() if profile_sdf else None
                        if cone_bounded or puffer_direct_0053:
                            stream_callback(acc_chunk, selected, av_chunk)
                        else:
                            stream_callback(acc_chunk, selected)
                        if profile_sdf:
                            callback_s += perf_counter() - callback_t0
                    if stage_profile_0060:
                        previous_batch_end_0060 = perf_counter()
                        batch_total_0060.append(previous_batch_end_0060 - batch_start_0060)
                if sphere_0058:
                    rejected_0058 = int(getattr(self.core, '_projection_sphere_rejected_0058', 0)) - sphere_previous_count_0058
                    print('[sdf_localize_sphere_0058] candidates=', n_candidate,
                          'early_rejected=', rejected_0058,
                          'early_reduction_pct=', f'{100.0 * rejected_0058 / max(1, n_candidate):.2f}',
                          'audit=', sphere_audit_0058, flush=True)
                if reuse_0059:
                    print('[sdf_localize_reuse_0059] accepted=', n_accepted,
                          'core_s=', f"{getattr(self.core, '_localize_core_s_0059', 0.0) - core_previous_s_0059:.3f}",
                          'avatar_s=', f"{getattr(self.core, '_localize_avatar_s_0059', 0.0) - avatar_previous_s_0059:.3f}",
                          'audit=', reuse_audit_0059, flush=True)
                if adapt_arg.get('sdf_vectorize_wrap_0061', False):
                    accessory = adapt_arg.get('accessory_curve_handle')
                    for label, core in (
                            ('source', self.core),
                            ('target', getattr(accessory, 'core', None))):
                        if core is None:
                            continue
                        stats = getattr(core, '_wrap_vector_stats_0061', None)
                        if stats:
                            print('[sdf_wrap_vector_0061]', 'curve=', label,
                                  'calls=', stats['calls'],
                                  'points=', stats['points'],
                                  'fallback_calls=', stats['fallback_calls'],
                                  'audited_points=', stats['audited_points'],
                                  'max_abs_error=', f"{stats['max_abs_error']:.3e}",
                                  flush=True)
                if stage_profile_0060:
                    # Exclusivity: `adapt.avatar_localize` includes the whole
                    # `core.*` scope. Do not add these scope totals together.
                    profile_now_0060 = getattr(self.core, '_localize_stages_0060', {})
                    scopes_0060 = {}
                    for stage_key, (cumulative_s, cumulative_calls) in profile_now_0060.items():
                        before_s, before_calls = stage_before_0060.get(stage_key, (0.0, 0))
                        elapsed_s = cumulative_s - before_s
                        calls = cumulative_calls - before_calls
                        if calls and elapsed_s >= 0.0:
                            scope, phase = stage_key.split('.', 1)
                            scopes_0060.setdefault(scope, []).append((phase, elapsed_s, calls))
                    stream_total_0060 = perf_counter() - stream_t0
                    batch_median_0060 = float(np.median(batch_total_0060)) if batch_total_0060 else 0.0
                    batch_p95_0060 = float(np.percentile(batch_total_0060, 95)) if batch_total_0060 else 0.0
                    print('[sdf_profile_0060] scope=stream',
                          'candidates=', n_candidate, 'accepted=', n_accepted,
                          'batches=', batch_count_0060,
                          'wall_s=', f'{stream_total_0060:.3f}',
                          'candidate_iterator_s=', f'{iterator_s_0060:.3f}',
                          'decode_s=', f'{point_decode_s:.3f}',
                          'localize_s=', f'{localize_compute_s:.3f}',
                          'audit_s=', f'{audit_s_0060:.3f}',
                          'selection_s=', f'{select_s_0060:.3f}',
                          'callback_s=', f'{callback_s:.3f}',
                          'batch_p50_s=', f'{batch_median_0060:.4f}',
                          'batch_p95_s=', f'{batch_p95_0060:.4f}', flush=True)
                    for scope, stages in sorted(scopes_0060.items()):
                        total_s = sum(value for _, value, _ in stages)
                        print('[sdf_profile_0060] scope=', scope,
                              'exclusive_sum_s=', f'{total_s:.3f}',
                              'stages=', len(stages), flush=True)
                        for phase, seconds, calls in sorted(stages, key=lambda x: (-x[1], x[0])):
                            print('[sdf_profile_0060_stage] scope=', scope,
                                  'stage=', phase,
                                  'seconds=', f'{seconds:.3f}',
                                  'pct_scope=', f'{100.0*seconds/max(1e-12, total_s):.1f}',
                                  'calls=', calls,
                                  'avg_ms=', f'{1000.0*seconds/max(1, calls):.3f}',
                                  flush=True)
                if profile_sdf:
                    print('[sdf_localize_0041] candidates=', n_candidate,
                          'accepted=', n_accepted,
                          'decode_s=', f'{point_decode_s:.3f}',
                          'localize_s=', f'{localize_compute_s:.3f}',
                          'inference_write_callback_s=', f'{callback_s:.3f}',
                          'traversal_total_s=', f'{perf_counter() - stream_t0:.3f}',
                          flush=True)
                if adapt_arg.get('adapt_debug_counts', True):
                    print('[filter_grid_adapt] bounded_stream=True',
                          'candidate_samples=', n_candidate,
                          'after_localize=', n_accepted,
                          'localize_compute_s=', f'{localize_compute_s:.3f}')
                return n_candidate, n_accepted, localize_compute_s

            compact_staging = bool(adapt_arg.get("dcsdd_compact_staging", False))
            # 0051: puffer's correction-rich path still needs the full SIX
            # model inputs for both curves when an avatar SDF is queried. It
            # does not consume frame/points/x_radius/world samples after
            # localization, nor the candidate-relative `inside` array.
            # Keep original dtypes and ordering; do not quantize or recompute.
            puffer_minimal = bool(adapt_arg.get('_sdf_puffer_minimal_staging', False))
            if puffer_minimal and (
                not bounded or compact_staging
                or adapt_arg.get('_dcsdd_stream_callback') is not None
                or adapt_arg.get('mode', 'direct') != 'direct'
                or not (adapt_arg.get('rigid_radius', False)
                        != adapt_arg.get('wrap_radius', False))
            ):
                raise ValueError('0051 puffer minimal staging requires bounded, non-callback direct rigid or wrap')
            # Use the actual consumer requirements. Post-SDF blend and query
            # seam fitting need avatar coords even without avatar inference.
            puffer_avatar_inference = (
                bool(adapt_arg.get('auto_avatar_snug_field', False))
                or bool(adapt_arg.get('cut_avatar', False))
                or bool(adapt_arg.get('detail_avatar_gate', False))
                or bool(adapt_arg.get('final_carve', False))
                or str(adapt_arg.get('snug_mode', 'multiplicative')).lower() == 'additive'
                or str(adapt_arg.get('accessory_offset_mode', 'none')).lower() in ('local', 'both')
            )
            puffer_model_keys = frozenset((
                'samples_local', 'coords', 'angles', 'radius', 'rho', 'rho_n'))
            puffer_avatar_keys = (
                puffer_model_keys if puffer_avatar_inference else frozenset(('coords',)))
            if compact_staging and not adapt_arg.get("rigid_radius", False):
                raise ValueError("compact staging requires direct rigid_radius")
            # First pass counts accepted broad-phase indices without storing
            # their coordinates. Second pass localizes in fixed-size chunks.
            import tempfile
            from pathlib import Path
            batch_size = int(adapt_arg.get("dcsdd_filter_batch_size", 262144))
            if batch_size <= 0:
                raise ValueError("dcsdd_filter_batch_size must be positive")
            n_candidate = sum(len(idx) for idx in cyl_mesh.iter_filter_grid_indices(
                mc_grid, batch_size=batch_size))
            broad_s = perf_counter() - broad_t0
            localize_t0 = perf_counter()
            # Staging arrays are backed by disk for the 2048 memmap path,
            # allowing final accepted samples to exceed available RAM.
            scratch_dir = getattr(mc_grid, 'memmap_dir', None)
            if scratch_dir:
                stage = tempfile.TemporaryDirectory(prefix="oktopus-bounded-", dir=scratch_dir)
                if not hasattr(mc_grid, '_bounded_scratch'):
                    mc_grid._bounded_scratch = []
                mc_grid._bounded_scratch.append(stage)
            else:
                stage = None
            output_buffers = {}
            shared_keys = set()
            # These fields are the actual input to model_withgrid._pack_common.
            # Never change the type or values of any retained field.
            inference_keys = frozenset((
                "samples_local", "coords", "angles", "radius", "rho", "rho_n",
            ))
            written = 0
            candidate_offset = 0
            core_localize_s = 0.0
            stage_write_s = 0.0
            stage_bytes = 0
            original_stage_bytes = 0

            def _alloc_and_write(kind, key, value, at, size):
                arr = np.asarray(value)
                if kind not in output_buffers:
                    output_buffers[kind] = {}
                if key not in output_buffers[kind]:
                    shape = (n_candidate,) + arr.shape[1:]
                    if stage is not None:
                        # Use memmap for all numeric localization arrays.
                        path = str(Path(stage.name) / f"{kind}_{key}.bin")
                        result = np.memmap(path, dtype=arr.dtype, mode="w+", shape=shape)
                    else:
                        result = np.empty(shape, dtype=arr.dtype)
                    output_buffers[kind][key] = result
                if size:
                    output_buffers[kind][key][at:at+size] = arr

            if stage is not None:
                idx_buffer = np.memmap(str(Path(stage.name)/"indices.i64"),
                                       dtype=np.int64, mode="w+", shape=(max(n_candidate,1),))
                if not compact_staging and not puffer_minimal:
                    inside_buffer = np.memmap(str(Path(stage.name)/"inside.i64"),
                                              dtype=np.int64, mode="w+", shape=(max(n_candidate,1),))
            else:
                idx_buffer = np.empty(n_candidate, dtype=np.int64)
                if not compact_staging and not puffer_minimal:
                    inside_buffer = np.empty(n_candidate, dtype=np.int64)
            # Avoid printing one 'input vs' block per chunk.
            quiet_arg = dict(adapt_arg)
            quiet_arg["adapt_debug_counts"] = False
            for candidate_idx in cyl_mesh.iter_filter_grid_indices(mc_grid, batch_size=batch_size):
                sample_chunk = mc_grid.idx2pts(candidate_idx)
                call_t0 = perf_counter()
                acc_chunk, av_chunk, accepted_local = self.core.localize_samples_adapt(
                    sample_chunk, quiet_arg)
                core_localize_s += perf_counter() - call_t0
                write_t0 = perf_counter()
                n_take = len(accepted_local)
                idx_buffer[written:written+n_take] = candidate_idx[accepted_local]
                stage_bytes += n_take * np.dtype(np.int64).itemsize
                if puffer_minimal:
                    # Count the equivalent legacy output payload for diagnostics;
                    # it was recorded on disk even where fields were unused.
                    original_stage_bytes += n_take * np.dtype(np.int64).itemsize * 2
                    original_stage_bytes += sum(np.asarray(v).nbytes for v in av_chunk.values())
                    original_stage_bytes += sum(np.asarray(v).nbytes for v in acc_chunk.values())
                    for kind, data, keep in (
                        ('avatar', av_chunk, puffer_avatar_keys),
                        ('accessory', acc_chunk, puffer_model_keys)):
                        missing = keep.difference(data)
                        if missing:
                            raise ValueError(f'0051 {kind} localizer missing keys: {sorted(missing)}')
                        for key in keep:
                            value = data[key]
                            _alloc_and_write(kind, key, value, written, n_take)
                            stage_bytes += np.asarray(value).nbytes
                else:
                    if not compact_staging:
                        inside_buffer[written:written+n_take] = candidate_offset + accepted_local
                        stage_bytes += n_take * np.dtype(np.int64).itemsize
                        if not output_buffers:
                            shared_keys = {k for k in av_chunk if k in acc_chunk and av_chunk[k] is acc_chunk[k]}
                        for key, value in av_chunk.items():
                            _alloc_and_write("avatar", key, value, written, n_take)
                            stage_bytes += np.asarray(value).nbytes
                    for key, value in acc_chunk.items():
                        if compact_staging:
                            if key not in inference_keys:
                                continue
                        elif key in shared_keys:
                            continue
                        _alloc_and_write("accessory", key, value, written, n_take)
                        stage_bytes += np.asarray(value).nbytes
                stage_write_s += perf_counter() - write_t0
                written += n_take
                candidate_offset += len(candidate_idx)
            assert candidate_offset == n_candidate
            accessory_data = {
                **{k: v[:written] for k, v in output_buffers.get("avatar", {}).items()},
                **{k: v[:written] for k, v in output_buffers.get("accessory", {}).items()},
            }
            avatar_data = (None if compact_staging else
                           {k: v[:written] for k, v in output_buffers.get("avatar", {}).items()})
            kidx = idx_buffer[:written]
            # The original candidate-relative `inside` indices are never read
            # after filter_grid_adapt by independent boots. Do not allocate the
            # ~3 GB array in this explicitly selected mode.
            inside = None if (compact_staging or puffer_minimal) else inside_buffer[:written]
            localize_s = perf_counter() - localize_t0
            if adapt_arg.get("adapt_debug_counts", True):
                print("input vs:", n_candidate)
                print("after src interval crop:", written)
                print("[filter_grid_adapt]", "bounded=True", "fast_bbox_marking=", fast_bbox,
                      "use_control_filter=", use_control_filter,
                      "src_interval_filter=", use_src_interval_filter,
                      "candidate_samples=", n_candidate, "after_localize=", written,
                      "batch_size=", batch_size,
                      "compact_staging=", compact_staging,
                      "broad_s=", f"{broad_s:.3f}", "localize_s=", f"{localize_s:.3f}",
                      "localize_compute_s=", f"{core_localize_s:.3f}",
                      "stage_write_s=", f"{stage_write_s:.3f}",
                      "staged_GB=", f"{stage_bytes/1e9:.3f}")
                if puffer_minimal:
                    print('[sdf_puffer_0051] minimal_staging=True',
                          'avatar_model_inputs=', puffer_avatar_inference,
                          'avatar_keys=', ','.join(sorted(puffer_avatar_keys)),
                          'accessory_keys=', ','.join(sorted(puffer_model_keys)),
                          'inside_staged=False',
                          'legacy_estimated_GB=', f'{original_stage_bytes/1e9:.3f}',
                          'payload_reduction_pct=',
                          f'{100*(original_stage_bytes-stage_bytes)/max(1,original_stage_bytes):.2f}',
                          flush=True)
            return accessory_data, avatar_data, kidx, inside
        broad_s = perf_counter() - broad_t0
        mode = adapt_arg.get("mode", "direct")
        localize_t0 = perf_counter()

        if mode == "direct":
            accessory_data, avatar_data, inside = self.core.localize_samples_adapt(
                samples,
                adapt_arg,
            )
        else:
            raise ValueError(f"Unknown adapt mode: {mode}")

        localize_s = perf_counter() - localize_t0
        kidx = kidx[inside]

        if adapt_arg.get("adapt_debug_counts", True):
            print(
                "[filter_grid_adapt]",
                "use_control_filter=", use_control_filter,
                "src_interval_filter=", use_src_interval_filter,
                "fast_bbox_marking=", fast_bbox,
                "filter_s=", f"{interval_lo:.4f}->{interval_hi:.4f}",
                "filter_radius_scale=", filter_radius_scale,
                "filter_radius_add=", filter_radius_add,
                "filter_norm=", filter_norm,
                "candidate_samples=", len(samples),
                "after_localize=", len(inside),
                "broad_s=", f"{broad_s:.3f}",
                "localize_s=", f"{localize_s:.3f}",
            )

        return accessory_data, avatar_data, kidx, inside

    def filter_grid_adapt_test(self, mc_grid, adapt_arg):
        self.core.update_coords()
        self.core.update_frame()

        accessory_curve_handle = adapt_arg['accessory_curve_handle']
        acc_coords = self.core.map_coords_to_by_arclen(self.core.key_ts, accessory_curve_handle.core)

        acc_intpl = accessory_curve_handle.core.interpolate(acc_coords)
        tangent_acc = accessory_curve_handle.core.calc_x_radius(acc_coords)
        thetas = (2*np.pi)* np.linspace(0, 1, n_sample_circle, endpoint=False)
        acc_intpl['thetas'] = thetas
        cyl_mesh = accessory_curve_handle.__gen_cyl_mesh(acc_intpl)
        samples, kidx = cyl_mesh.filter_grid(mc_grid)
        #samples_data, inside = self.core.localize_samples_stretch(samples, stretch_arg)
        #accessory_data, avatar_data, inside = self.core.localize_samples_adapt(samples, adapt_arg)
        accessory_data, inside = accessory_curve_handle.core.localize_samples(samples)
        kidx = kidx[inside]

        avatar_samples, avatar_kidx = self.cyl_mesh.filter_grid(mc_grid)
        # calculate the neareast points and find inside points
        avatar_data, avatar_inside = self.core.localize_samples(avatar_samples)
        
        return accessory_data, avatar_data, kidx

    def filter_grid_stretch(self, mc_grid, stretch_arg):
        print("stretch arg ", stretch_arg)

        use_runtime_support = bool(stretch_arg.get("use_runtime_support", False))

        if use_runtime_support:
            # Build explicit support for inference only
            runtime_support = self.core.build_stretch_runtime_support(stretch_arg, n_samples=n_sample_curve)

            # Build cylinder mesh from runtime support
            intpl = {
                "points": runtime_support["points"],
                "frame": runtime_support["frame"],
                "radius": runtime_support["radius"],
                "thetas": (2 * np.pi) * np.linspace(0, 1, n_sample_circle, endpoint=False),
            }
            cyl_mesh = self.__gen_cyl_mesh(intpl)

            samples, kidx = cyl_mesh.filter_grid(mc_grid)

            # Localize directly on runtime support
            samples_data, inside = self.core.localize_samples_on_runtime_support(
                samples, runtime_support, norm=1.0
            )

            # For now keep detail identical to base
            samples_data["samples_detail"] = samples_data["samples_local"].copy()
            samples_data["coords_detail"] = samples_data["coords"].copy()
            samples_data["w_seam"] = np.ones_like(samples_data["coords"], dtype=np.float64)

            kidx = kidx[inside]
            return samples_data, kidx

        else:
            # Old curve-mutate path
            self.core.update_coords()
            self.core.update_frame()
            self.core.localize_stretch(stretch_arg)

            ts = np.linspace(0., 1., n_sample_curve)
            intpl, _ = self.core.interpolate_stretch(ts, stretch_arg)
            thetas = (2*np.pi) * np.linspace(0, 1, n_sample_circle, endpoint=False)
            intpl['thetas'] = thetas

            cyl_mesh = self.__gen_cyl_mesh(intpl)
            samples, kidx = cyl_mesh.filter_grid(mc_grid)

            samples_data, inside = self.core.localize_samples_stretch(samples, stretch_arg)
            kidx = kidx[inside]

            # important: restore original curve after temporary stretch
            self.core.restore_stretch()

            return samples_data, kidx


    def filter_grid_stretch1(self, mc_grid, stretch_arg):
        # gen mixed cyl mesh 
        #points, ts_new = self.core.localize_stretch(stretch_arg)
        print("stretch arg ", stretch_arg)
        self.core.update_coords()
        self.core.update_frame()
        self.core.localize_stretch(stretch_arg)
        #try:
        ts = np.linspace(0., 1., n_sample_curve)
        intpl,_ = self.core.interpolate_stretch(ts, stretch_arg)
        thetas = (2*np.pi)* np.linspace(0, 1, n_sample_circle, endpoint=False)
        intpl['thetas'] = thetas
        cyl_mesh = self.__gen_cyl_mesh(intpl)
        
        samples, kidx = cyl_mesh.filter_grid(mc_grid)
        samples_data, inside = self.core.localize_samples_stretch(samples, stretch_arg)

        kidx = kidx[inside]
        return samples_data, kidx


    def calc_cylinder_SDF(self, mc_grid):
        samples, kidx = self.cyl_mesh.filter_grid(mc_grid, extend_bbox=True)
        # NOTE: value of samples_sdf is calculated in std space
        samples_sdf = self.core.calc_cylinder_SDF(samples)
        return samples_sdf, kidx
    
    def calc_global_implicit(self, mc_grid, sigma, return_coords=False):
        # gen blended cyl mesh 
        ts = np.linspace(0., 1., n_sample_curve)
        
        intpl = self.core.interpolate(ts)
        thetas = (2*np.pi)* np.linspace(0, 1, n_sample_circle, endpoint=False)
        intpl['thetas'] = thetas
        intpl['level'] = sigma
        cyl_mesh = self.__gen_cyl_mesh(intpl)
        
        samples, kidx = cyl_mesh.filter_grid(mc_grid)
        if not return_coords:
            sdfs = self.core.calc_global_implicit(samples)
            return sdfs, samples, kidx
        else:
            sdfs, coords, ts = self.core.calc_global_implicit(samples, True)
            return {
                'sdf': sdfs,
                'kidx': kidx,
                'coords': coords,
                'ts': ts,
            }
        # neg = sdfs <= 0.
        # return samples[neg]
   
    
    def get_bbox_scale(self):
        bmax, bmin = self.cyl_mesh.calc_bbox()
        scale = np.max(bmax-bmin)/ 2.
        return scale
    
    def find_inbbox(self, points):
        bmax, bmin = self.cyl_mesh.calc_bbox()
        side1 = np.all((points - bmin) >= 0, axis=1)
        side2 = np.all((bmax - points) >= 0, axis=1)
        inside = np.logical_and(side1, side2)
        return inside

    def find_inside(self, points):
        pidx = np.arange(points.shape[0])
        inbbox = self.find_inbbox(points)
        pidx_bbox = pidx[inbbox]

        _, inside = self.core.localize_samples(points[inbbox])
        pidx_inside = pidx_bbox[inside]
        return pidx_inside
    
    def print_info(self):
        # print('Points: ', self.core.key_points)
        # print('Radius: ', self.core.key_radius)
        print('Start radius: ', self.core.key_radius[0])
        print('Start x radius: ', self.core.start_ball_x)
        
    def load_data(self, arg):
        self.set_curve(arg)

    def export_data(self):
        data = self.core.export_data()
        data['name'] = self.name
        return data
    
    def export_vis(self):
        return self.core.export_vis()


    def _gen_cyl_mesh_runtime(self, intpl):
        thetas = intpl['thetas']
        verts = intpl['points']
        yz_rs = intpl['radius']
        frames = intpl['frame']

        if 'level' in intpl:
            level = intpl['level']
        else:
            level = 0.0

        theta_coo = np.asarray([np.cos(thetas), np.sin(thetas)])
        theta_coo = np.einsum('nj,ji->nji', yz_rs, theta_coo)
        cyl_pts = np.einsum('njk,nji->nik', frames[:,1:], theta_coo)

        dists = np.linalg.norm(cyl_pts, axis=2)
        scales = (dists + level) / (dists + 1e-12)
        cyl_pts *= scales[..., None]

        cyl_pts += verts[:, None, :]

        v0, v1 = verts[0], verts[-1]

        cyl_mesh = CylindersMesh()
        vidx = cyl_mesh.add_cylinder(cyl_pts)
        cyl_mesh.add_cap(v0, vidx[0])
        cyl_mesh.add_cap(v1, vidx[-1], flip_face=True)
        return cyl_mesh


    def build_cylmesh_from_runtime_support(self, runtime_support, n_sample_curve=n_sample_curve, n_sample_circle=n_sample_circle):
        s0 = float(runtime_support["coords"][0])
        s1 = float(runtime_support["coords"][-1])
        s_dense = np.linspace(s0, s1, n_sample_curve)

        intpl = self.core.interpolate_runtime_support(runtime_support, s_dense)
        thetas = (2.0 * np.pi) * np.linspace(0.0, 1.0, n_sample_circle, endpoint=False)
        intpl["thetas"] = thetas
        return self._gen_cyl_mesh_runtime(intpl)


    def filter_grid_on_runtime_support(self, mc_grid, runtime_support, norm=1.0):
        """
        Runtime-support version of filter_grid(...):
        build cylinder around already-placed support,
        collect candidate voxels,
        localize directly against that support.
        """
        cyl_mesh = self.build_cylmesh_from_runtime_support(runtime_support)
        samples, kidx = cyl_mesh.filter_grid(mc_grid)

        samples_data, inside = self.core.localize_samples_on_runtime_support(
            samples,
            runtime_support,
            norm=norm,
        )
        kidx = kidx[inside]
        return samples_data, kidx


    
