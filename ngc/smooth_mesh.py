import trimesh
import os


in_path  = "armadillo.obj"          # or .ply/.stl
out_path = "armadillo_taubin.ply"

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
    out_path = os.path.join(output_path, filename+'_smooth_taubin.ply')
    mesh_s.export(out_path)
    print("Wrote:", out_path)

if __name__ == '__main__':
    input_mesh = '/fast/pselvaraju/Oktopus_now/Pack50Dataset/boots/mesh.ply'
    output_path = '/fast/pselvaraju/Oktopus_now/Pack50Dataset/boots'
    filename = 'boots'
    taubin_smooth(input_mesh, output_path, filename)
