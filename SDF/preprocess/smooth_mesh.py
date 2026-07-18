import trimesh
import os


in_path  = "oktopus_unitbb_.obj"          # or .ply/.stl
out_path = "oktopus_taubin.ply"

def taubin_smooth(input_mesh, output_path, filename):
    mesh = trimesh.load(input_mesh, process=False)
    if isinstance(mesh, trimesh.Scene):
        # if the file loads as a scene, merge geometry
        mesh = trimesh.util.concatenate(tuple(mesh.geometry.values()))

    # --- Taubin smoothing (good default) ---
    # iterations: more = smoother
    # lamb: smoothing step (positive)
    # nu: inflation step (negative); typical nu ~ -0.53 * lamb to reduce shrink
    mesh_s = trimesh.smoothing.filter_taubin(
        mesh,
        lamb=0.2,
        nu=-0.2,
        iterations=30
    )
    #out_path = os.path.join(output_path, filename+'_smooth_taubin.ply')
    out_path = os.path.join(output_path, 'mesh_base.ply')
    mesh_s.export(out_path)
    print("Wrote:", out_path)

if __name__ == '__main__':
    input_mesh = '/fast/pselvaraju/Oktopus_now/Pack10Dataset/oktopus_9_v1/mesh.ply'
    output_path = '/fast/pselvaraju/Oktopus_now/Pack10Dataset/oktopus_9_v1'
    filename = 'oktopus'
    taubin_smooth(input_mesh, output_path, filename)
