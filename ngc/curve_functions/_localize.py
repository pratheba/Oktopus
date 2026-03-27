import os, pickle
import numpy as np
import os.path as op
import trimesh
from scipy.spatial import KDTree
from scipy.spatial.transform import Rotation, Slerp
from handle_utils import CylindersMesh
from scipy.ndimage import gaussian_filter1d
from curve_utils import *


n_sample_curve = 200
n_sample_circle = 120


def localize_samples(self, pointcloudsamples, return_sdf=False, norm=1.0, update_curve=False, update_radius=False, name=''):
    sample_keypoint_map = self.curve_projection(pointcloudsamples)
    sample_keypoint_map_range = np.logical_and(sample_keypoint_map >= 0., sample_keypoint_map <= 1.)
    sample_index = np.arange(pointcloudsamples.shape[0])

    ### Keep only the points that fall within the rane of 0 and 1
    if update_curve:
        pointcloudsamples0 = pointcloudsamples.copy()
    sample_keypoint_map = sample_keypoint_map[sample_keypoint_map_range]
    pointcloudsamples = pointcloudsamples[sample_keypoint_map_range]
    sample_index = sample_index[sample_keypoint_map_range]

    # interpolate with the new additional non linear skeletal keypoints
    intpl = self.interpolate(sample_keypoint_map, radius_type='train')

    ## The new keypoiints in 3D world coord system based on the curve projection from the surface/space samples
    proj_vs = intpl['points']
    yz_radius = intpl['radius']
    frame_mat = intpl['frame']

    # frame: (N, 3,3), vs (N, 3)
    # The vector of the keypoint to the skeletam point is rotated using the rotation from
    samples_local0 = np.einsum('nij,nj->ni', frame_mat, (pointcloudsamples - proj_vs))
    # And all are bounding to radius
    w, u, v = samples_local0[:,0], samples_local0[:, 1], samples_local0[:, 2]
    if update_curve:
        stats = compute_local_centering_stats(samples_local0, sample_keypoint_map)
        #C_old, C_new, C_key_unsmooth = compute_centered_curve_world(self, stats)
        C_old, C_new = compute_centered_curve_world(self, stats)
        s_dense = 0.5 * (stats["edges"][:-1] + stats["edges"][1:])
        #C_key_smooth = resample_curve_to_key_ts(s_dense, C_new_smooth, self.key_ts)
        s_target = np.linspace(0.0, 1.0, n_sample_curve)
        C_key_old_smooth = resample_curve_to_key_ts(s_dense, C_old, s_target)
        old_frame = self.get_new_frame(C_key_old_smooth) #key_frame 
        #self.update_frame()
        #old_frame = self.key_frame 
        old_T = old_frame[:, 0, :]

        C_new_smooth = gaussian_filter1d(C_new, sigma=2.0, axis=0)
        C_key_smooth = resample_curve_to_key_ts(s_dense, C_new_smooth, s_target)
        z0 = self.z_axis[0] if getattr(self, "z_axis", None) is not None else None
        self.set_resamples(C_key_smooth, z_axis0=z0)
        self.update_coords()
        #C_new_smooth = gaussian_filter1d(C_new, sigma=2, axis=0)
        export_curve_points_as_ply(C_key_old_smooth, C_key_smooth, str(idx)+"_curve_compare_points.ply")
        export_shape_and_curves_as_ply(
            points=pointcloudsamples,
            C_old=C_old,
            C_new=C_new_smooth,
            out_path=str(idx)+"_shape_and_curves.ply"
        new_frame = self.get_new_frame(self.key_points) #key_frame 
        new_T = new_frame[:, 0, :]
        auto_N = new_frame[:, 1, :]
        xfer_N = self.key_frame[:, 1, :]

        dots = np.sum(auto_N * xfer_N, axis=1)
        print("normal alignment min/max:", dots.min(), dots.max())
        print("normal alignment mean:", dots.mean())

        self.key_frame = transfer_frame_orientation(
            old_frame=old_frame,
            old_tangent=old_T,
            new_tangent=new_T,
            enforce_continuity=True,
            orthonormalize=True,
        )
        self.z_axis = self.key_frame[:, 2, :].copy()
        self.y_axis = self.key_frame[:, 1, :].copy()
        #self.update_frame()
        #C_key_smooth = resample_curve_to_key_ts(s_dense, C_old, s_target)
        #self.update_frame()


        return self.localize_samples(pointcloudsamples0, return_sdf=return_sdf, norm=norm, update_curve=False, update_radius=True, name=name)

    if update_radius:
        update_wrap_profile_from_coords(self, sample_keypoint_map, w, u, v, n_curve_bins=24, n_theta_bins=24, quantile=0.98, gaussian_smooth_curve=2.0, gaussian_smooth_theta=2.0, min_count=25, radius_type='wrap')
        update_wrap_occupancy_from_coords(self, sample_keypoint_map, u, v, n_curve_bins=24, quantile=0.7, min_count=50)

        ################# Uncomment later for curve center and cylinder #################################
#            self.update_radius_from_coords(sample_keypoint_map, w, u, v)
#            #self.update_wrap_profile_from_coords(sample_keypoint_map, w, u, v, radius_type='wrap')
#            self.update_cylinder_radius_from_coords(sample_keypoint_map, w, u, v, radius_type='cylinder')
#            #self.update_cylinder_radius_from_wrap(eps=0.03, isotropic=False)
#            radius_yz = self.interpolate(sample_keypoint_map, points=False, frame=False, radius_type='train')['radius']
#            #print(yz_radius.shape)
#            yz_radius = radius_yz.copy()
#            self.update_coords()
#            # 3D trimesh overlay
#            visualize_keyframes_with_profiles_trimesh(
#                self,
#                pointcloudsamples,
#                sample_keypoint_map,
#                name=name,
#                show_train=False,
#                show_cylinder=True,
#                show_wrap=False,
#                export_glb=False,
#                export_ply=True
#            )


    # If the end ball is None then x_radius  = 1.0 
    x_radius = self.calc_x_radius(sample_keypoint_map)
    #print(yz_radius.shape)
    #print(x_radius[:,None].shape)
    radius = np.concatenate([x_radius[:,None], yz_radius], axis=1)

    samples_local = samples_local0.copy()
    samples_local /= (radius + 1e-12)
    rho = np.sqrt(v**2 + u**2)
    u_n = samples_local[:,1] #u / (radius[:,1] + 1e-12)
    v_n = samples_local[:,2] #v / (radius[:,2] + 1e-12)
    angle = np.arctan2(v_n, u_n)
    rho_n = np.sqrt(v_n**2 + u_n**2)
    
    # in std cylinder
    norms = np.linalg.norm(samples_local, axis=1)
    if return_sdf:
        return norms - 1, sidx
    
    inside_cyl = (norms <= norm)
    inside = sample_index[inside_cyl]

    # NOTE: vs -> (vx, *, *). (vert -> (vx, 0, 0))
    # [0,1] -> [-1,1]
    vx = 2*sample_keypoint_map - 1
    samples_local[:, 0] += vx
    return {
        'samples': pointcloudsamples[inside_cyl],
        'samples_local': samples_local[inside_cyl],
        'coords': sample_keypoint_map[inside_cyl],
        'rho': rho[inside_cyl],
        'rho_n': rho_n[inside_cyl],
        'angles': angle[inside_cyl],
        'radius': yz_radius[inside_cyl],
        'frame_mat': frame_mat[inside_cyl]
    }, inside


def localize_samples_global(self, vs):
    ts = self.curve_projection(vs, outside=True)

    intpl = self.interpolate(ts)
    proj_vs = intpl['points']
    yz_rs = intpl['radius']
    frame_mat = intpl['frame']

    x_rs = self.calc_x_radius(ts)
    radius = np.concatenate([x_rs[:,None], yz_rs], axis=1)

    # frame: (N, 3,3), vs (N, 3)
    samples_local = np.einsum('nij,nj->ni', frame_mat, (vs - proj_vs))
    samples_local /= radius
    vx = 2*ts - 1
    samples_local[:, 0] += vx
    return samples_local, ts


def localize_samples_mix(self, vs, mix_arg):
    ts = self.curve_projection(vs)
    ts_range = np.logical_and(ts >= 0., ts <= 1.)
    sidx = np.arange(vs.shape[0])
    ts = ts[ts_range]
    vs = vs[ts_range]
    sidx = sidx[ts_range]
    
    intpl = self.interpolate_mix(ts, mix_arg)
    proj_vs = intpl['points']
    yz_rs = intpl['radius']
    frame_mat = intpl['frame']

    x_rs = self.calc_x_radius(ts)
    radius = np.concatenate([x_rs[:,None], yz_rs], axis=1)

    # frame: (N, 3,3), vs (N, 3)
    samples_local0 = np.einsum('nij,nj->ni', frame_mat, (vs - proj_vs))
    w, u, v = samples_local0[:,0], samples_local0[:, 1], samples_local0[:, 2]
    samples_local = samples_local0.copy()
    samples_local /= (radius + 1e-12)
    rho = np.sqrt(v**2 + u**2)
    #angle = np.arctan2(v, u)
    u_n = samples_local[:,1] #u / (radius[:,1] + 1e-12)
    v_n = samples_local[:,2] #v / (radius[:,2] + 1e-12)
    angle = np.arctan2(v_n, u_n)
    rho_n = np.sqrt(v_n**2 + u_n**2)

    # in std cylinder
    norms = np.linalg.norm(samples_local, axis=1)
    inside_cyl = norms <= 1
    inside = sidx[inside_cyl]

    # NOTE: vs -> (vx, *, *). (vert -> (vx, 0, 0))
    # [0,1] -> [-1,1]
    vx = 2*ts - 1
    samples_local[:, 0] += vx
    return {
        'samples': vs[inside_cyl],
        'samples_local': samples_local[inside_cyl],
        'coords': ts[inside_cyl],
        'rho': rho[inside_cyl],
        'rho_n': rho_n[inside_cyl],
        'angles': angle[inside_cyl],
        'radius': yz_rs[inside_cyl],
    }, inside


def localize_samples_stretch(self, vs, stretch_arg, return_sdf=False):
    # for stretch or offset 
    # Project samples from surface onto the curve to get ts - they key points
    #return self.localize_samples(vs)
    ts = self.curve_projection(vs)
    valid_range = np.logical_and(ts >= 0., ts <= 1.)

    sidx = np.arange(vs.shape[0])
    ts = ts[valid_range]
    vs = vs[valid_range]
    sidx = sidx[valid_range]
    

    # Interpolate the keypoints (points on curve), radius and frame
    intpl, ts_new = self.interpolate_stretch(ts, stretch_arg)
    #intpl, ts_new = self.interpolate(ts) #, stretch_arg)
    #intpl = self.interpolate(ts) #, stretch_arg)
    #ts_new = ts.copy()
    proj_vs = intpl['points']
    frame_mat = intpl['frame']
    yz_radius = intpl['radius']


    # Build local coordinate system
    x_rs = self.calc_x_radius(ts_new)
    radius = np.concatenate([x_rs[:,None], yz_radius], axis=1)
    # frame: (N, 3,3), vs (N, 3)
    samples_local0 = np.einsum('nij,nj->ni', frame_mat, (vs - proj_vs))
    w, u, v = samples_local0[:,0], samples_local0[:, 1], samples_local0[:, 2]
    #samples_local = np.einsum('nij,nj->ni', frame_mat, (vs - proj_vs))
    #w, u, v = samples_local[:,0], samples_local[:, 1], samples_local[:, 2]
    rho = np.sqrt(v**2 + u**2)
    theta = np.arctan2(v, u)


    samples_local = samples_local0.copy()
    samples_local /= (radius + 1e-12)
    #samples_local /= radius
    norms = np.linalg.norm(samples_local, axis=1)
    inside_cyl = norms <= 1.0
    inside = sidx[inside_cyl]
    u_n = samples_local[:,1] # u / (radius[:,1] + 1e-12)
    v_n = samples_local[:,2] # v / (radius[:,2] + 1e-12)
    #u_n = u / (radius[:,1]**2 + 1e-12)
    #v_n = v / (radius[:,2]**2 + 1e-12)
    angle = np.arctan2(v_n, u_n)
    rho_n = np.sqrt(v_n**2 + u_n**2)


   # geometry coords mapped to [-1, 1]
    vx_base = 2.0*ts_new - 1.0
    #vx_base = 2.0*ts - 1.0
    samples_local[:, 0] += vx_base

    stretch_length = stretch_arg['length']
    t0 = stretch_arg['t0']
    t1 = stretch_arg['t1']
    eps_region = stretch_arg.get('eps_region', 0.03)
    eps_seam = stretch_arg.get('eps_seam', 0.05)
  
    w_region = make_detail_mask(ts_new, t0, t1, eps_region)
    eps = 1e-12
    tau = np.clip((ts_new - t0)/((t1 - t0)+eps), 0.0, 1.0)
    ts_tile_phase = np.mod((stretch_length * tau), 1.0)

    # seam fade: 0 near phase seam, 1 away from seam
    w_seam = seam_fade(ts_tile_phase, eps_seam)
    w_wrap = w_region * w_seam

    ts_wrapped = t0 + (t1-t0)*ts_tile_phase
    ts_used = (1.0 - w_wrap) * ts_new + w_wrap * ts_wrapped
    vx_used = 2.0 * ts_used - 1.0

    #ts_detail = np.mod(stretch_arg['length'] * ts_new, 1.0)
    # NOTE: vs -> (vx, *, *). (vert -> (vx, 0, 0))
    # [0,1] -> [-1,1]
    samples_detail  = samples_local0.copy()
    samples_detail /= (radius + 1e-12)
    #x_radius_detail = self.calc_x_radius(ts_detail)
    #r_detail = np.concatenate([x_rs[:,None], intpl['radius_detail']], axis=1)
    #samples_detail /= r_detail #intpl['radius_detail']
    samples_detail[:,0] += vx_used

    return {
        #'samples': vs[inside_cyl],
        'samples_local': samples_local[inside_cyl],
        'samples_detail': samples_detail[inside_cyl],
        'coords': ts_new[inside_cyl],
        #'coords': ts[inside_cyl],
        'coords_detail': ts_used[inside_cyl], 
        'w_seam': w_seam,
        'rho': rho[inside_cyl],
        'rho_n': rho_n[inside_cyl],
        'angles': angle[inside_cyl],
        'radius': yz_radius[inside_cyl],
    }, inside


def localize_samples_adapt(self, vs, adapt_arg):
    # calculate the neareast points and find inside points
    avatar_data, inside = self.localize_samples(vs)

    accessory_curve_handle = adapt_arg['accessory_curve_handle']
    accessory_curve_handle.core.update_coords()
    accessory_curve_handle.core.update_frame()

    avatar_coords = avatar_data["coords"]                         
    avatar_samples_local = avatar_data["samples_local"].copy()   

    vx_avatar = 2.0 * avatar_coords - 1.0
    w_n_avatar = avatar_samples_local[:, 0] - vx_avatar
    u_n_avatar = avatar_samples_local[:, 1]
    v_n_avatar = avatar_samples_local[:, 2]

    theta_avatar = np.arctan2(v_n_avatar, u_n_avatar)
    rho_n_avatar = np.sqrt(u_n_avatar**2 + v_n_avatar**2)

    avatar_radius_y = avatar_data["radius"][:,0]
    avatar_radius_z = avatar_data["radius"][:,1]

    tangent_avatar = self.calc_x_radius(avatar_coords)          
    w_avatar = w_n_avatar * (tangent_avatar + 1e-12)
    u_avatar = u_n_avatar * (avatar_radius_y)
    v_avatar = v_n_avatar * (avatar_radius_z)

    # map avatar coords -> accessory coords by arclen
    #avatar_coords = maybe_flip_coords(avatar_coords, True) #adapt_arg.get("flip_s", False))

    t = np.load('ngc/boots_on.npz', allow_pickle=True)['arr_0'].item()['boots_on_3']
    #print("loaded")
    #print(t)
    #accessory_curve_handle.core.key_points = t['key_points']
    #accessory_curve_handle.core.key_ts = t['key_ts']
    #self.core.key_radius = t['key_radius']
    #accessory_curve_handle.core.key_frame = t['key_frame']
    #accessory_curve_handle.update()
    #accessory_curve_handle.core.update_coords()
    #accessory_curve_handle.core.update_frame()

    acc_coords = self.map_coords_to_by_arclen(avatar_coords, accessory_curve_handle.core)
    #acc_coords = maybe_flip_coords(acc_coords, True) #adapt_arg.get("flip_s", False))

#        import matplotlib.pyplot as plt
#        plt.hist(acc_coords, bins=50)
#        plt.title("Accessory coords coverage")
#        plt.savefig("coords_coverage_acc.jpg")
#        plt.hist(avatar_coords, bins=50)
#        plt.title("Avatar coords coverage")
#        plt.savefig("coords_coverage_avatar.jpg")
    #exit()

    acc_intpl = accessory_curve_handle.core.interpolate(acc_coords)

    #accessory_curve_handle.core.update_coords()
    #accessory_curve_handle.core.update_frame()
    tangent_acc = accessory_curve_handle.core.calc_x_radius(acc_coords)

    acc_radius_y = acc_intpl["radius"][:,0]
    acc_radius_z = acc_intpl["radius"][:,1]

    source_npz  = np.load('ngc/armadillo_on.npz', allow_pickle=True)['arr_0'].item()['armadillo_on_8']
    r_src = self.interpolate_wrap_radius_test1(avatar_coords, theta_avatar, source_npz['wrap_theta_bins'], source_npz['wrap_s_bins'], source_npz['wrap_radius'])
    r_tgt = accessory_curve_handle.core.interpolate_wrap_radius_test1(acc_coords, theta_avatar, t['wrap_theta_bins'], t['wrap_s_bins'], t['wrap_radius'])

    rho_avatar = np.sqrt(u_avatar**2 + v_avatar**2)
    scale = r_tgt / (r_src + 1e-12)
    rho_acc = rho_avatar * scale 

    u_acc = 0.85*rho_acc * np.cos(theta_avatar)
    v_acc = 0.85*rho_acc * np.sin(theta_avatar)

    scale_y = acc_radius_y / (avatar_radius_y + 1e-12)
    scale_z = acc_radius_z / (avatar_radius_z + 1e-12)
    scale_w = tangent_acc / (tangent_avatar + 1e-12)
    w_acc = w_avatar * scale_w 

    


    #acc_frame = maybe_swap_nb(acc_intpl["frame"], True) #adapt_arg.get("swap_nb", False))
    acc_frame = acc_intpl["frame"] #adapt_arg.get("swap_nb", False))
    avatar_frame = avatar_data["frame_mat"]

#        w_acc, u_acc, v_acc = rotate_wuv_avatar_to_acc(
#            w_avatar, u_avatar, v_avatar,
#            avatar_frame[:,0], avatar_frame[:,1], avatar_frame[:,2],
#            acc_frame[:,0],    acc_frame[:,1],    acc_frame[:,2],
#            project_SO3=True
#        )
#        delta = np.median(estimate_delta(avatar_frame[:,1], avatar_frame[:,2], acc_frame[:,1]))
#        cd, sd = np.cos(delta), np.sin(delta)
#        u2 =  cd*u_acc - sd*v_acc
#        v2 =  sd*u_acc + cd*v_acc
#        u_acc, v_acc = u2, v2

    u_acc = 0.9*u_avatar * scale_y 
    v_acc = 0.9*v_avatar * scale_z

    w_n_acc = w_acc / (tangent_acc + 1e-12)
    u_n_acc = u_acc / (acc_radius_y + 1e-12)
    v_n_acc = v_acc / (acc_radius_z + 1e-12)

    #theta_offset = np.deg2rad(adapt_arg.get('theta_offset_deg', -35.0))
    #theta_offset = 0

    #angles_acc = theta_avatar + theta_offset
    #angles_acc = (angles_acc + np.pi) % (2*np.pi) - np.pi
    #rho_scale = 0.9
    #rho_n_acc = rho_scale * rho_n_avatar

    #u_n_acc = rho_n_acc * np.cos(angles_acc)
    #v_n_acc = rho_n_acc * np.sin(angles_acc)


    vx_acc = 2.0 * acc_coords - 1.0
    #samples_local_acc = np.stack([w_n_avatar + vx_acc, u_n_acc, v_n_acc], axis=1)
    samples_local_acc = np.stack([w_n_acc + vx_acc, u_n_acc, v_n_acc], axis=1)

    #u_acc = u_n_acc * acc_radius_y
    #v_acc = v_n_acc * acc_radius_z
    rho_acc = np.sqrt(u_acc**2 + v_acc**2)
    rho_n_acc = np.sqrt(u_n_acc**2 + v_n_acc**2)
    angles_acc = np.arctan2(v_n_acc, u_n_acc)

    accessory_data = dict(avatar_data)
    accessory_data["coords"] = acc_coords
    accessory_data["samples_local"] = samples_local_acc
    accessory_data["angles"] = angles_acc
    accessory_data["rho_n"] = rho_n_acc
    accessory_data["rho"] = rho_acc
    accessory_data["radius"] = acc_intpl["radius"]   # <-- keep canonicalV
    accessory_data["frame"] = acc_intpl["frame"]

    return accessory_data, avatar_data, inside


def localize_occ_samples(self, samples):
    ts = self.curve_projection(samples, outside=True)

    intpl = self.interpolate(ts)
    proj_vs = intpl['points']
    frame_mat = intpl['frame']

    # frame: (N, 3,3), vs (N, 3)
    samples_local = np.einsum('nij,nj->ni', frame_mat, (samples - proj_vs))

    vx = 2*ts - 1
    samples_local[:, 0] += vx
    return {
        'samples_local': samples_local,
        'coords': ts
    }


