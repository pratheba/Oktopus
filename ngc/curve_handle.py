import os, pickle
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

    def filter_grid_adapt(self, mc_grid, adapt_arg):
        samples, kidx = self.cyl_mesh.filter_grid(mc_grid)
        #samples, kidx = mc_grid.generate_samples()

        mode = adapt_arg.get("mode", "direct")

        if mode == "direct":
            accessory_data, avatar_data, inside, runtime_support = self.core.localize_samples_adapt(samples, adapt_arg)
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


    
