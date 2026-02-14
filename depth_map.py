import numpy as np
import open3d as o3d

# --- inputs ---
mesh_path = "Pack50Dataset/boots/boots_unitbb.ply"          # any format Open3D supports
W, H = 1024, 1024                 # output image size
fx, fy = 1200.0, 1200.0          # focal lengths (pixels)
cx, cy = W / 2.0, H / 2.0        # principal point (pixels)
# Camera-to-world pose (4x4). Replace with yours.
#cam_T_world = np.eye(4)
#cam_T_world[:3, 3] = [0, 0, 1.2]  # move camera 1.2m along +Z (example)

# --- load mesh ---
mesh = o3d.io.read_triangle_mesh(mesh_path)
mesh.compute_vertex_normals()
normals = np.asarray(mesh.vertex_normals)
verts = mesh.vertices
#mesh.vertices = o3d.utility.Vector3dVector(verts + 0.2 * normals)

# --- offscreen renderer ---
from open3d.visualization.rendering import OffscreenRenderer, MaterialRecord
renderer = OffscreenRenderer(W, H)
scene = renderer.scene
scene.set_background([0, 0, 0, 0])

# Add mesh
mat = MaterialRecord()
mat.shader = "defaultUnlit"
mat.base_color = [1, 1, 1, 1]
#mesh.vertex_colors = False
mat.base_metallic = 0.0
mat.base_roughness = 0.0
mesh.vertex_colors = o3d.utility.Vector3dVector(0.5 * (np.array(mesh.vertex_normals) + 1.0))
scene.add_geometry("mesh", mesh, mat)

# --- camera setup from intrinsics + extrinsics ---
cam = scene.camera
near, far = 0.01, 10.0
# Open3D expects column-major 4x4 view matrix = world_to_camera
#world_T_cam = np.linalg.inv(cam_T_world).astype(np.float32)
intrinsics = np.array([[fx, 0, cx],[0, fy, cy], [0, 0, 1]])
cam.set_projection(intrinsics, near, far, W, H)

#cam.set_projection(fx, fy, cx, cy, W, H, near, far)   # pinhole intrinsics
world_T_cam = np.eye(4)
world_T_cam[:3, 3] = [0, 0, 1]

R = world_T_cam[:3, :3]
t = world_T_cam[:3, 3]
cam_T_world = np.eye(4)
cam_T_world[:3, :3] = R.T
cam_T_world[:3, 3] = -R.T @ t

eye = [0, 0, 2.5] #cam_T_world[:3, 3]
forward = [0, 0, 0] #cam_T_world[:3, 2]
center = [0, 0, 0] #eye + forward
up = [0, 1, 0] #cam_T_world[:3, 1]

cam.look_at(center, eye, up)

#cam.set_view_matrix(world_T_cam)

# --- render depth ---
depth_o3d = renderer.render_to_depth_image(z_in_view_space=True)  # True → metric depth (view space Z)
depth = np.asarray(depth_o3d)                                     # shape (H, W), float32; meters; 0 == background

# Save if you want:
height = (depth - np.min(depth))
height /= np.max(height)

o3d.io.write_image("height.png", o3d.geometry.Image((height* 255 ).astype(np.uint8)))  # millimeters

#z_min, z_max = np.percentile(depth, [5, 95])
#depth_vis = (depth - z_min) / (z_max - z_min)
#depth_vis = np.clip(depth_vis, 0, 1)

#o3d.io.write_image("depth.png", o3d.geometry.Image((depth_vis * 100000).astype(np.uint8)))  # millimeters
o3d.io.write_image("depth.png", o3d.geometry.Image((depth * 10000 ).astype(np.uint16)))  # millimeters
#o3d.io.write_image("depth.png", o3d.geometry.Image((((depth - np.min(depth))/(np.max(depth) - np.min(depth)))* 255).astype(np.uint16)))  # millimeters
#o3d.io.write_image("depth.png", o3d.geometry.Image((depth * 1000000).astype(np.uint16)))  # millimeters
#o3d.io.write_image("depth.png", o3d.geometry.Image((depth / np.max(depth) * 255).astype(np.uint16)))  # millimeters
#normals = renderer.render_to_normal_image()
#o3d.io.write_image("normals.png", normals)
img = renderer.render_to_image()
o3d.io.write_image("normal.png", img) 

