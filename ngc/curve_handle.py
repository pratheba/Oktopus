import os, pickle
import numpy as np
import os.path as op
import trimesh
from scipy.spatial import KDTree
from scipy.spatial.transform import Rotation, Slerp
from handle_utils import CylindersMesh
from curve_mask import *


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
        self.cyl_mesh = self.gen_cylinder_mesh(n_sample_curve, n_sample_circle)

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
        key_radius_scales = np.interp(
            self.core.key_ts, coords, scales
        )
        self.core.key_radius *= key_radius_scales[:,None]


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
        self.cyl_mesh = self.gen_cylinder_mesh(n_sample_curve, n_sample_circle)

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

    def gen_cylinder_mesh(self, n_sample_curve, n_sample_circle):
        # t0,t1 = [0., 1.]
        t0 = self.core.key_ts[0]
        t1 = self.core.key_ts[-1]
        ts = np.linspace(t0, t1, n_sample_curve)
        thetas = (2*np.pi)* np.linspace(0, 1, n_sample_circle, endpoint=False)

        intpl = self.core.interpolate(ts)
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
        if self.core.use_ball('start'):
            # for level set
            disp = self.core.ball_disp('start', level)

            start_pts = cyl_pts[0] + disp
            cyl_pts = np.concatenate([
                start_pts[None, ...], cyl_pts], axis=0)
            
            v0 += disp
        
        if self.core.use_ball('end'):
            # for level set
            disp = self.core.ball_disp('end', level)

            end_pts = cyl_pts[-1] + disp
            cyl_pts = np.concatenate([
                cyl_pts, end_pts[None, ...]], axis=0)
            v1 += disp

        cyl_mesh = CylindersMesh()
        vidx = cyl_mesh.add_cylinder(cyl_pts)

        cyl_mesh.add_cap(v0, vidx[0])
        cyl_mesh.add_cap(v1, vidx[-1], flip_face=True)
        return cyl_mesh

    def generate_samples(self, num_samples):
        return self.core.generate_samples(num_samples)

    def localize_samples(self, samples):
        return self.core.localize_samples(samples)
    
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

    def filter_grid_stretch(self, mc_grid, stretch_arg):
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
        #samples_data, inside = self.core.localize_samples_stretch(samples, stretch_arg)
        #sd1, inside1 = self.core.localize_samples(samples)
        sd0, inside0 = self.core.localize_samples_stretch(samples, stretch_arg)
        #print("inside counts:", inside0.shape[0], inside1.shape[0])
        #print("coords range:", sd0["coords"].min(), sd0["coords"].max(),
        #            sd1["coords"].min(), sd1["coords"].max())
        #print("angle range:",  sd0["angles"].min(), sd0["angles"].max(),
        #            sd1["angles"].min(), sd1["angles"].max())

        # Key: compare distributions of normalized coords
        #print("samples_local mean abs (w,u,v):",
        #   np.mean(np.abs(sd0["samples_local"]), axis=0),
        #   np.mean(np.abs(sd1["samples_local"]), axis=0))
        #exit()

        kidx = kidx[inside0]
        #kidx = kidx[inside1]
        #return samples_data, kidx
        #return sd1, kidx
        return sd0, kidx
        #finally:
        #    return
        #    self.core.restore_stretch()
        #    self.core.update_coords()
        #    self.core.update_frame()


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


class PWLACurve():
    """docstring for PWLACurve."""
    def __init__(self, arg=None):
        if arg is None:
            return 
        
        self.set_curve(arg)

    def update(self):
        if self.flag_points:
            # if some keypoints changed, update the curve
            self.update_coords()
            self.update_frame()

            self.flag_points = False
            return
        
    def need_update(self):
        return self.flag_points
    
    def use_ball(self, side):
        return getattr(self, f'{side}_ball_x') is not None
    
    def ball_disp(self, side, level=0.):
        if side == 'start':
            tan = self.key_points[1] - self.key_points[0]
            tan /= np.linalg.norm(tan)
            disp = self.start_ball_x + level
            return -disp* tan
        
        if side == 'end':
            tan = self.key_points[-1] - self.key_points[-2]
            tan /= np.linalg.norm(tan)
            disp = self.end_ball_x + level
            return disp* tan

    def set_curve(self, arg):
        # self.step = arg['resample_step']
        self.key_points = arg['key_points']
        # NOTE: radius: (N, 2), y-z radius
        self.key_radius = arg['key_radius']
        # self.end_ball_x = self.key_radius[-1].mean()
        # self.start_ball_x = self.key_radius[0].mean()
        # self.end_ball_x = None
        # self.start_ball_x = None
        if 'ball' in arg:
            ball_data = arg['ball']
        else:
            ball_data = None
        if ball_data is None:
            self.end_ball_x = None
            self.start_ball_x = None
        else:
            if ball_data['end_x'] is not None:
                self.end_ball_x = ball_data['end_x']
            else:
                self.end_ball_x = None

            if ball_data['start_x'] is not None:
                self.start_ball_x = ball_data['start_x']
            else:
                self.start_ball_x = None

        x_axis = self.estimate_tangent(self.key_points)
        z_axis = arg['z_axis']
        if len(z_axis.shape) == 1:
            z_axis = np.tile(z_axis, (x_axis.shape[0], 1))
        
        self.z_axis = self.project_z_axis(x_axis, z_axis)

        self.update_coords()
        # check if points or radius have to be updated
        self.flag_points = True
        # self.flag_radius = False

    def calc_curve_length(self):
        pts = self.key_points
        edge_vec = pts[1:] - pts[:-1]
        edge_lengths = np.linalg.norm(edge_vec, axis=1)
        curve_length = np.sum(edge_lengths)
        return curve_length

    def set_points(self, points):
        assert (points.shape == self.key_points.shape)
        self.key_points = points
        self.flag_points = True


    def set_resamples(self, points, z_axis0):
        if z_axis0 is None:
            z_axis0 = self.z_axis[0]
        # new points can be different in number
        ts = self.keypoints_segment_length(points)
        #edge_vec = points[1:] - points[:-1]
        #edge_lengths = np.linalg.norm(edge_vec, axis=1)
        #curve_length = np.sum(edge_lengths)
        #ts = np.cumsum(np.r_[0., edge_lengths]) / curve_length

        new_rs = self.interpolate(ts, radius=True)['radius']
        self.key_points = points
        self.key_ts = ts
        self.key_radius = new_rs
        x_axis = self.estimate_tangent(self.key_points)
        self.z_axis = self.propagate_z_axis(x_axis, z_axis0)
        y_axis = np.cross(self.z_axis, x_axis)
        self.key_frame = np.concatenate([
            x_axis.reshape(-1,1,3),
            y_axis.reshape(-1,1,3),
            self.z_axis.reshape(-1,1,3)
        ], axis=1)
        #self.rotation = Rotation.from_matrix(self.key_frame)
        #self.rot_slerp = Slerp(self.key_ts, self.rotation)
        self.flag_points = True


    def apply_rotation(self, anchor, rot):
        self.key_points = rot.apply(self.key_points - anchor) + anchor
        self.z_axis = rot.apply(self.z_axis)
        self.flag_points = True

    def estimate_tangent(self, points):
        edge_vec = points[1:] - points[:-1]
        edge_vec /= (np.linalg.norm(edge_vec, axis=1, keepdims=True) + 1e-12)

        if edge_vec.shape[0] > 1:
            tan_start = edge_vec[0]
            tan_end = edge_vec[-1]
            tans = (edge_vec[1:] + edge_vec[:-1]) / 2.
            tans /= np.linalg.norm(tans, axis=1, keepdims=True)

            vert_tan = np.concatenate([
                tan_start.reshape(1,3), 
                tans, 
                tan_end.reshape(1,3)
            ], axis=0)
        else:
            vert_tan = np.tile(edge_vec, (2,1))

        return vert_tan
    
    def project_z_axis(self, x_axis, z_axis):
        dots = np.sum(x_axis*z_axis, axis=1)
        if not np.allclose(dots, 0):
            # project z_axis to x_axis
            z_axis = z_axis - dots[:, None]* x_axis
            # NOTE: huh???? forgot this
            z_axis /= np.linalg.norm(z_axis, axis=1, keepdims=True)

        return z_axis
    
    def propagate_z_axis(self, x_axis, z_axis0):
        final_z = []
        # current z_axis
        c_zx = z_axis0 
        for i in range(x_axis.shape[0]):
            xx = x_axis[i]
            zx = c_zx - (xx @ c_zx)*xx
            zx /= np.linalg.norm(zx)
            final_z.append(zx)
            c_zx = zx

        return np.asarray(final_z)

    def rotation_from_vectors(self, a, b):
         v = np.cross(a, b)
         c = np.dot(a, b)

         if np.linalg.norm(v) < 1e-8:
             return np.eye(3)

         vx = np.array([
              [0, -v[2], v[1]],
              [v[2], 0, -v[0]],
              [-v[1], v[0], 0]
              ])
         R = np.eye(3) + vx + vx @ vx * (1.0 / (1.0 + c))
         return R

    # Parallel transport frame
    def update_frame(self):
        points = self.key_points
        n = points.shape[0]

        T = self.estimate_tangent(self.key_points)
        z0 = self.z_axis[0] if self.z_axis is not None else np.array([0,0,1], dtype=np.float64)

        z_axis = np.zeros_like(T)
        z_axis[0] = z0 - np.dot(z0, T[0]) * T[0]
        z_axis[0] /= np.linalg.norm(z_axis[0]) + 1e-12

        for i in range(1, len(T)):
            R = self.rotation_from_vectors(T[i-1], T[i])
            z_axis[i] = R @ z_axis[i-1]
            z_axis[i] /= np.linalg.norm(z_axis[i]) + 1e-12

        y_axis = np.cross(z_axis, T)
        y_axis /= np.linalg.norm(y_axis, axis=1, keepdims=True) + 1e-12
        z_axis = np.cross(T, y_axis)
        z_axis /= (np.linalg.norm(z_axis, axis=1, keepdims=True) + 1e-12)
        self.z_axis = z_axis
        self.key_frame = np.stack([T, y_axis, z_axis], axis=1)

#        world_up = np.array([0.0, 0.0, 1.0], dtype=np.float64)
#        if abs(np.dot(T[0], world_up)) > 0.95:
#            world_up = np.array([1.0, 0.0, 0.0], dtype=np.float64)
#
#        N = np.zeros_like(T)
#        B = np.zeros_like(T)
#
#        N0 = world_up - np.dot(world_up, T[0]) * T[0]
#        N0 = N0 / (np.linalg.norm(N0) + 1e-12)
#        N[0] = N0
#        B[0] = np.cross(T[0], N[0])
#
#        for i in range(1, n):
#            R = self.rotation_from_vectors(T[i-1], T[i])
#            N[i] = R @ N[i-1]
#            N[i] /= (np.linalg.norm(N[i]) + 1e-12)
#
#            B[i] = np.cross(T[i], N[i])
#            B[i] /= (np.linalg.norm(B[i]) + 1e-12)
         
        self.key_frame = np.stack([T, y_axis, z_axis], axis=1)
        self.rotation = None
        self.rot_slerp = None
 

    def update_frame_slerp(self):
        x_axis = self.estimate_tangent(self.key_points)

        z0 = self.z_axis[0] if self.z_axis is not None else np.array([0.0, 0.0, 1.0], dtype=np.float64)
        z_axis = self.propagate_z_axis(x_axis, z0)

        # Sign continuity
        for i in range(1, z_axis.shape[0]):
            if np.dot(z_axis[i], z_axis[i-1]) < 0:
                z_axis[i] *= -1.0
        y_axis = np.cross(z_axis, x_axis)
        y_axis /= (np.linalg.norm(y_axis, axis=1, keepdims=True) + 1e-12)

        z_axis = np.cross(x_axis, y_axis)
        z_axis /= (np.linalg.norm(z_axis, axis=1, keepdims=True) + 1e-12)
        self.z_axis = z_axis

        self.key_frame = np.concatenate([
            x_axis.reshape(-1,1,3),
            y_axis.reshape(-1,1,3),
            z_axis.reshape(-1,1,3)
        ], axis=1)
        self.rotation = Rotation.from_matrix(self.key_frame)
        self.rot_slerp = Slerp(self.key_ts, self.rotation)

    def update_frame_old(self):
        x_axis = self.estimate_tangent(self.key_points)
        self.z_axis = self.project_z_axis(x_axis, self.z_axis)
        y_axis = np.cross(self.z_axis, x_axis)
        self.key_frame = np.concatenate([
            x_axis.reshape(-1,1,3),
            y_axis.reshape(-1,1,3),
            self.z_axis.reshape(-1,1,3)
        ], axis=1)
        self.rotation = Rotation.from_matrix(self.key_frame)
        self.rot_slerp = Slerp(self.key_ts, self.rotation)

    def set_frame(self, new_frame):
        self.key_frame = new_frame
        
        #idx = np.searchsorted(self.key_ts, ts)
        #idx = np.clip(idx, 0, len(self.key_ts)-1)
        #frame = self.key_frame[idx]
        #self.rotation = Rotation.from_matrix(self.key_frame)
        #self.rot_slerp = Slerp(self.key_ts, self.rotation)

    def update_coords(self):
        edge_vec = self.key_points[1:] - self.key_points[:-1]
        edge_lengths = np.linalg.norm(edge_vec, axis=1)
        self.curve_length = np.sum(edge_lengths)
        self.key_ts = np.cumsum(np.r_[0., edge_lengths]) / self.curve_length


    def is_points_in_edge(self, points, vt0, vt1):
        # project points on line segment(edge), return if inside the edge 
        v0, t0 = vt0
        v1, t1 = vt1

        length = np.linalg.norm(v1 - v0)
        vec = (v1 - v0) / length
        proj_len = (points - v0) @ vec
        inside_flag = np.logical_and(proj_len >= 0., proj_len <= length)

        # calculate natural coords for projected points
        ts = t0 + ((t1 - t0)/length)* proj_len
        return inside_flag, ts
    

    def curve_projection(self, samples, N_discrete=n_sample_curve, outside=False):
        uniform_linear_points = np.linspace(0., 1., N_discrete, endpoint=False)
        ii = np.searchsorted(uniform_linear_points, self.key_ts)
        non_uniform_linear_points = np.insert(uniform_linear_points, ii, self.key_ts)
        non_uniform_linear_points = np.unique(non_uniform_linear_points)

        skeletal_verts = self.interpolate(non_uniform_linear_points, radius=False, frame=False)['points']
        tree = KDTree(skeletal_verts)
        # not accurate for radius-varying skeleton
        _, vidx = tree.query(samples)
        samples3D_to_skeleton = -1*np.ones(samples.shape[0])
        # basically project samples onto the piecewise linear curve
        num_vert = skeletal_verts.shape[0]
        for vid in range(num_vert):
            sample_index = np.argwhere(vidx == vid).flatten()
            if len(sample_index) == 0:
                continue

            samples_v = samples[sample_index]

            if 0 < vid < num_vert - 1:
                # middle part
                ## The samples which belong to the sample_index is mapped to the nearest keypoints through their index 
                samples3D_to_skeleton[sample_index] = non_uniform_linear_points[vid]

                in1, px1 = self.is_points_in_edge(
                    samples_v, 
                    (skeletal_verts[vid], non_uniform_linear_points[vid]), 
                    (skeletal_verts[vid+1], non_uniform_linear_points[vid+1])
                )
                in2, px2 = self.is_points_in_edge(
                    samples_v, 
                    (skeletal_verts[vid-1], non_uniform_linear_points[vid-1]), 
                    (skeletal_verts[vid], non_uniform_linear_points[vid])
                )
                in_p = np.logical_xor(in1, in2)
                px = (in1*px1 + in2*px2)[in_p]
                samples3D_to_skeleton[sample_index[in_p]] = px

            elif vid == 0:
                in1, px1 = self.is_points_in_edge(
                    samples_v, 
                    (skeletal_verts[vid], non_uniform_linear_points[vid]), 
                    (skeletal_verts[vid+1], non_uniform_linear_points[vid+1])
                )
                # consider halfball+ cylinder
                # left side of cylinder remain valid
                if self.start_ball_x is not None or outside:
                    samples3D_to_skeleton[sample_index] = 0.

                samples3D_to_skeleton[sample_index[in1]] = px1[in1]

            else:
                in2, px2 = self.is_points_in_edge(
                    samples_v, 
                    (skeletal_verts[vid-1], non_uniform_linear_points[vid-1]), 
                    (skeletal_verts[vid], non_uniform_linear_points[vid]), 
                )
                if self.end_ball_x is not None or outside:
                    samples3D_to_skeleton[sample_index] = 1.

                samples3D_to_skeleton[sample_index[in2]] = px2[in2]

        #import pdb; pdb.set_trace();
        return samples3D_to_skeleton

    def calc_x_radius(self, ts):
        xrs = np.ones(ts.shape[0])
        if self.end_ball_x is not None:
            xrs[ts == 1.] = self.end_ball_x
        
        if self.start_ball_x is not None:
            xrs[ts == 0.] = self.start_ball_x
        
        return xrs
    
    def calc_cylinder_SDF(self, vs):
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
        norms_cyl = np.linalg.norm(samples_local, axis=1)
        
        vx = 2*ts - 1
        samples_local[:, 0] += vx
        xpos = vx >= 0.
        xneg = np.logical_not(xpos)

        norms_max = np.linalg.norm(samples_local, axis=1, ord=np.inf)
        if self.start_ball_x is None:
            norms_cyl[xneg] = np.maximum(norms_cyl[xneg], norms_max[xneg])

        if self.end_ball_x is None:
            norms_cyl[xpos] = np.maximum(norms_cyl[xpos], norms_max[xpos])
            
        return norms_cyl - 1.
    
    def calc_std_cylinder_SDF(self, vs):
        # assume this is a standard cylinder
        # vs all inside the cylinder
        ts = self.curve_projection(vs)
        intpl = self.interpolate(ts)
        proj_vs = intpl['points']
        yz_rs = intpl['radius']
        
        dist_cyl = np.linalg.norm(vs - proj_vs, axis=1)
        dist_cyl = yz_rs[:,0] - dist_cyl

        tsd = np.minimum(ts, 1-ts)
        curve_length = self.calc_curve_length()
        dist_side = tsd*curve_length

        dist = np.minimum(dist_cyl, dist_side)
        return -dist
    
    def calc_global_implicit(self, vs, return_coords=False):
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
        norms_cyl = np.linalg.norm(samples_local, axis=1)
        # return norms_cyl - 1.
        signs = np.sign(norms_cyl - 1.)
        
        nearest_local = samples_local / norms_cyl[:, None]
        nearest_global = nearest_local* radius
        # NOTE: here it should be the inverse of frames
        # so it is the transpose(since they are unitary mats)
        # and we simply modify it in einsum: nij,nj->ni => nji,nj->ni
        nearest_global = np.einsum('nji,nj->ni', frame_mat, nearest_global)
        nearest_global += proj_vs
        ds = np.linalg.norm(nearest_global - vs, axis=1)
        if return_coords:
            vx = 2*ts - 1
            samples_local[:, 0] += vx
            return ds*signs, samples_local, ts
        else:
            return ds*signs

    def localize_samples(self, pointcloudsamples, return_sdf=False):
        sample_keypoint_map = self.curve_projection(pointcloudsamples)
        #print("sample_keypoint_map", sample_keypoint_map)

        sample_keypoint_map_range = np.logical_and(sample_keypoint_map >= 0., sample_keypoint_map <= 1.)
        sample_index = np.arange(pointcloudsamples.shape[0])

        ### Keep only the points that fall within the rane of 0 and 1
        sample_keypoint_map = sample_keypoint_map[sample_keypoint_map_range]
        pointcloudsamples = pointcloudsamples[sample_keypoint_map_range]
        sample_index = sample_index[sample_keypoint_map_range]


        # interpolate with the new additional non linear skeletal keypoints
        intpl = self.interpolate(sample_keypoint_map)

        ## The new keypoiints in 3D world coord system based on the curve projection from the surface/space samples
        proj_vs = intpl['points']
        #print("proj vs", proj_vs)
        #print(self.key_points)
        #exit()
        yz_radius = intpl['radius']
        frame_mat = intpl['frame']

        # If the end ball is None then x_radius  = 1.0 
        x_radius = self.calc_x_radius(sample_keypoint_map)
        radius = np.concatenate([x_radius[:,None], yz_radius], axis=1)

        # frame: (N, 3,3), vs (N, 3)
        # The vector of the keypoint to the skeletam point is rotated using the rotation from
        samples_local0 = np.einsum('nij,nj->ni', frame_mat, (pointcloudsamples - proj_vs))
        # And all are bounding to radius
        #print("samples_local", samples_local)
        #import pdb; pdb.set_trace()
        w, u, v = samples_local0[:,0], samples_local0[:, 1], samples_local0[:, 2]
        samples_local = samples_local0.copy()
        samples_local /= (radius + 1e-12)
#        samples_local_n[:, 1] /= radius[:, 1]
#        samples_local_n[:, 2] /= radius[:, 2]
        rho = np.sqrt(v**2 + u**2)
        u_n = samples_local[:,1] #u / (radius[:,1] + 1e-12)
        v_n = samples_local[:,2] #v / (radius[:,2] + 1e-12)
        angle = np.arctan2(v_n, u_n)
        rho_n = np.sqrt(v_n**2 + u_n**2)
        
#        print("samples_local normalized", samples_local[0:10])
#        print("samples_local normalized N", samples_local_n[0:10])
#        exit()

        # in std cylinder
        norms = np.linalg.norm(samples_local, axis=1)
        if return_sdf:
            return norms - 1, sidx
        
        inside_cyl = norms <= 1
        inside = sample_index[inside_cyl]

        # NOTE: vs -> (vx, *, *). (vert -> (vx, 0, 0))
        # [0,1] -> [-1,1]
        vx = 2*sample_keypoint_map - 1
        #import pdb; pdb.set_trace()
        #print(samples_local, flush=True)
        samples_local[:, 0] += vx
        #samples_local_n[:, 0] += vx
        #print(samples_local, flush=True)
        #import pdb; pdb.set_trace()
        return {
            'samples': pointcloudsamples[inside_cyl],
            'samples_local': samples_local[inside_cyl],
            #'samples_local_n': samples_local_n[inside_cyl],
            'coords': sample_keypoint_map[inside_cyl],
            'rho': rho[inside_cyl],
            'rho_n': rho_n[inside_cyl],
            'angles': angle[inside_cyl],
            'radius': yz_radius[inside_cyl]
            # 'radius': yz_rs[inside_cyl],
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
        samples_local = np.einsum('nij,nj->ni', frame_mat, (vs - proj_vs))
        w, u, v = samples_local[:,0], samples_local[:, 1], samples_local[:, 2]
        rho = np.sqrt(v**2 + u**2)
        #angle = np.arctan2(v, u)
        samples_local /= radius
        u_n = u / (radius[:,1] + 1e-12)
        v_n = v / (radius[:,2] + 1e-12)
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
            # 'radius': yz_rs[inside_cyl],
        }, inside

    def keypoints_segment_length(self, points):
        edge_vec = points[1:] - points[:-1]
        edge_lengths = np.linalg.norm(edge_vec, axis=1)
        curve_length = np.sum(edge_lengths)
        s = np.concatenate([[0.0], np.cumsum(edge_lengths)])
        ts = s / (s[-1] + 1e-12)
        return ts

    def calc_curve_length(self):
        pts = self.key_points
        edge_vec = pts[1:] - pts[:-1]
        seglength = np.linalg.norm(edge_vec, axis=1)
        cumulative_length = np.concatenate([[0], np.cumsum(seglength)])
        curve_length = cumulative_length[-1] #np.sum(edge_lengths)
        return curve_length, cumulative_length

    def stretch_from_end_smooth_nonuniform(self, stretch_arg):
        points = self.key_points.copy()
        anchor = stretch_arg['anchor']
        stretch_length = stretch_arg['length']

        curve_length, cumulative_length = self.calc_curve_length()


        flipped = False
        if anchor == 'end':
            points = points[::-1].copy()
            flipped = True

        delta_length = (stretch_length - 1.0) * curve_length

        ts = self.keypoints_segment_length(points)
        # smooth curve
        w = ts*ts*(3 - 2*ts)
        w = w[:, None]

        t_end = points[-1] - points[-2]
        t_end /= (np.linalg.norm(t_end) + 1e-12)

        points = points + w * (delta_length * t_end)
        if flipped:
            points = points[::-1].copy()
        return points

    def stretch_uniform(self, stretch_arg):
        anchor = stretch_arg['anchor']
        stretch_length = stretch_arg['length']
        points = self.key_points.copy()

        if anchor == "start":
            out = points.copy()
            out[0] = points[0]
            for k in range(len(points)-1):
                out[k+1] = out[k] + stretch_length * (points[k+1] - points[k])
            return out

        if anchor == "end":
            points_reverse = points[::-1].copy()
            out = points_reverse
            out[0] = points_reverse[0]
            for k in range(len(points_reverse)-1):
                out[k+1] = out[k] + stretch_length * (points_reverse[k+1] - points_reverse[k])
            return out[::-1].copy()

        if anchor == "center":
            # stretch about mid index (keeps center fixed, stretches both directions)
            m = len(points)//2
            out = points.copy()
            out[m] = points[m]
            # forward
            for k in range(m, len(points)-1):
                out[k+1] = out[k] + stretch_length * (points[k+1] - points[k])
            # backward
            for k in range(m, 0, -1):
                out[k-1] = out[k] - stretch_length * (points[k] - points[k-1])
            return out


    def localize_stretch(self, stretch_arg):
        self.key_ts0 = self.key_ts.copy()
        self.key_radius0 = self.key_radius.copy()
        self.key_points0 = self.key_points.copy()
        self.key_frame0 = self.key_frame.copy()
        #self.rotation0 = self.rotation.copy()
        #self.rot_slerp0 = self.rot_slerp.copy()

        #points = self.stretch_uniform(stretch_arg)
        points = self.stretch_from_end_smooth_nonuniform(stretch_arg)
        self.key_points = points
        self.update_coords()
        self.update_frame()

    def restore_stretch(self):
        self.key_ts = self.key_ts0.copy()
        self.key_radius = self.key_radius0.copy()
        self.key_points = self.key_points0.copy()
        self.key_frame = self.key_frame0.copy()
        #self.rotation = self.rotation0.copy()
        #self.rot_slerp = self.rot_slerp0.copy()


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

#        return {
#            #'samples': vs[inside_cyl],
#            'samples_local': samples_local[inside_cyl],
#            #'samples_detail': samples_detail[inside_cyl],
#            #'coords': ts_new[inside_cyl],
#            'coords': ts[inside_cyl],
#            #'coords_detail': ts_used[inside_cyl], 
#            #'w_seam': w_seam,
#            'rho': rho[inside_cyl],
#            'rho_n': rho_n[inside_cyl],
#            'angles': angle[inside_cyl],
#            'radius': yz_radius[inside_cyl],
#        }, inside


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


    def interpolate(self, non_uniform_linear_skeletal_points, points=True, radius=True, frame=True):
        res = {}
        # key_points: (N, 3); key_radius: (N, 2)
        ts = non_uniform_linear_skeletal_points
        if points:
            pts_ts = np.stack([
                np.interp(ts, self.key_ts, self.key_points[:, 0]),
                np.interp(ts, self.key_ts, self.key_points[:, 1]),
                np.interp(ts, self.key_ts, self.key_points[:, 2])
            ]).T
            res['points'] = pts_ts
        
        if radius:
            rs_ts = np.stack([
                np.interp(ts, self.key_ts, self.key_radius[:, 0]),
                np.interp(ts, self.key_ts, self.key_radius[:, 1])
            ]).T
            res['radius'] = rs_ts

        if frame:
            # requirement of Slerp from Scipy
            ts_rot = np.clip(ts, a_min=1e-10, a_max=(1 - 1e-10))
            idx = np.searchsorted(self.key_ts, ts_rot)
            idx = np.clip(idx, 0, len(self.key_ts)-1)
            frame_ts= self.key_frame[idx]
            #frame_ts = self.rot_slerp(ts_rot).as_matrix()
            res['frame'] = frame_ts

        return res

    def interpolate_mix(self, ts, mix_arg):
        new_curve = mix_arg['curve_handle']
        func1 = mix_arg['mix_func1']
        func2 = mix_arg['mix_func2']
        ts1, weights1 = func1(ts)
        ts2, weights2 = func2(ts)

        rs1 = np.stack([
            np.interp(ts1, self.key_ts, self.key_radius[:, 0]),
            np.interp(ts1, self.key_ts, self.key_radius[:, 1])
        ]).T
        rs1_mean = rs1.mean(axis=1)
        scales1 = rs1 / rs1_mean[:, None]

        intpl2 = new_curve.core.interpolate(ts2, points=False, frame=False)
        rs2 = intpl2['radius']
        rs2_mean = rs2.mean(axis=1)
        scales2 = rs2 / rs2_mean[:, None]

        scales = scales1*weights1[:,None] + scales2*weights2[:,None]
        radius = rs1_mean[:,None]*scales

        intpl = self.interpolate(ts, radius=False)
        intpl['radius'] = radius
        return intpl

    def periodic_interpolate(self, ts_detail, ts, radius):
        x = np.mod(ts_detail, 1.0)

        ts_periodic = np.concatenate([ts, ts[1:]])
        radius_periodic = np.concatenate([radius, radius[1:]])

        x_copy = x.copy()
        x_copy[x_copy < ts_periodic[0]] += 1.0

        return np.interp(x_copy, ts_periodic, radius_periodic)

    def interpolate_stretch(self, ts, stretch_arg):
        #func = stretch_arg['mix_func']
        #ts_new = func(ts)
        radius = np.stack([
            np.interp(ts, self.key_ts, self.key_radius[:, 0]),
            np.interp(ts, self.key_ts, self.key_radius[:, 1])
        ]).T
        intpl = self.interpolate(ts, radius=False)

        intpl['radius'] = radius 
        return intpl, ts


    def inverse_transform(self, samples_local, ts):
        x_proj = 2*ts - 1
        res = self.interpolate(ts)
        verts, yz_rs, frame = res['points'], res['radius'], res['frame']

        x_rs = self.calc_x_radius(ts)
        radius = np.concatenate([x_rs[:,None], yz_rs], axis=1)
        # F(v - Pv) = (1,ry,rz)*(v_ - (v_x,0,0))
        # i.e. samples_local[:,0] - samples_x
        samples_global = samples_local.copy()
        samples_global[:, 0] -= x_proj
        samples_global *= radius
        # NOTE: here it should be the inverse of frames
        # so it is the transpose(since they are unitary mats)
        # and we simply modify it in einsum: nij,nj->ni => nji,nj->ni
        samples_global = np.einsum('nji,nj->ni', frame, samples_global)
        samples_global += verts
        return samples_global

    def generate_samples(self, num_samples):
        # mapping: cylinder <-> y^2+z^2 = 1, -1 <= x <= 1
        samples = np.random.uniform(-1., 1., size=(num_samples, 3))
        yz_norms = np.linalg.norm(samples[:,1:], axis=1)
        inside = yz_norms <= 1
        samples = samples[inside]

        # coords: [-1,1] to [0,1]
        samples_x = samples[:, 0]
        ts = (samples_x + 1) / 2
        res = self.interpolate(ts)
        verts, yz_rs, frame = res['points'], res['radius'], res['frame']

        # F(v - Pv) = (1,ry,rz)*(v_ - (v_x,0,0))
        # i.e. samples[:,0] - samples_x
        samples_global = np.zeros((samples.shape[0], 3))
        samples_global[:, 1:] = samples[:, 1:]*yz_rs
        # NOTE: here it should be the inverse of frames
        # so it is the transpose(since they are unitary mats)
        # and we simply modify it in einsum: nij,nj->ni => nji,nj->ni
        samples_global = np.einsum('nji,nj->ni', frame, samples_global)
        samples_global += verts

        # sdf_scales = np.sqrt(np.product(yz_rs, axis=1))
        return {
            'samples': samples_global,
            'samples_local':samples,
            'radius': yz_rs,
            'coords': ts,
        }


    def export_data(self):
        return {
            'key_points': self.key_points,
            'key_radius': self.key_radius,
            'z_axis': self.z_axis,
            'ball': {
                'start_x': self.start_ball_x,
                'end_x': self.end_ball_x,
            }
        }
    
    def export_vis(self):
        vidx = np.arange(self.key_points.shape[0])
        return {
            'vertices': self.key_points,
            'edges': np.asarray([vidx[:-1], vidx[1:]]).T,
        }
    
