"""Run the NSDUDF diagnostics on an exact mesh-distance UDF.

Use this as the clean baseline against ``test_nsdudf_diag.py``. The supplied
mesh may be open; only closest-point distance and its gradient are required.

Example:

    python diagnose_nsdudf_reference_mesh.py \
        --mesh path/to/mc_or_gt_mesh.ply \
        --nsdudf-repo third_party/nsdudf \
        --grid 65 \
        --output exact_mesh_diag_64
"""

from __future__ import annotations

import argparse
import os
import os.path as op
import sys

import numpy as np
import torch
import trimesh


_ROOT = op.dirname(op.abspath(__file__))
for path in (op.join(_ROOT, "src"), op.join(_ROOT, "src", "app")):
    if path not in sys.path:
        sys.path.insert(0, path)

from nsdudf_diagnostics import DiagnosticOptions, run_nsdudf_diagnostics


def _load_nsdudf(repo: str, model_path: str | None):
    repo = op.abspath(op.expanduser(repo))
    for path in (
        op.join(repo, "DualMesh-UDF"),
        op.join(repo, "custom_mc"),
        repo,
    ):
        if path not in sys.path:
            sys.path.insert(0, path)

    import core.utils as nsd_utils
    import core.meshing as nsd_meshing

    if model_path is None:
        model_path = op.join(repo, "model.pt")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = nsd_utils.load_model(model_path, device)
    return nsd_meshing, model


def _face_normals(vertices: np.ndarray, faces: np.ndarray) -> np.ndarray:
    tri = vertices[faces]
    normals = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    norm = np.linalg.norm(normals, axis=1, keepdims=True)
    norm[norm < 1e-12] = 1.0
    return normals / norm


def _make_exact_mesh_oracle(
    vertices: np.ndarray,
    faces: np.ndarray,
    *,
    center: np.ndarray,
    half: float,
    query_chunk: int,
):
    try:
        import igl
    except Exception as exc:
        raise ImportError(
            "This baseline requires libigl's Python module (`import igl`)."
        ) from exc

    vertices = np.ascontiguousarray(vertices, dtype=np.float64)
    faces = np.ascontiguousarray(faces, dtype=np.int64)
    normals = _face_normals(vertices, faces)
    chunk = max(1, int(query_chunk))

    def oracle(query_points):
        if torch.is_tensor(query_points):
            cube = query_points.detach().cpu().numpy()
        else:
            cube = np.asarray(query_points)
        cube = np.asarray(cube, dtype=np.float64).reshape(-1, 3)
        world = center[None, :] + half * cube

        distances = np.empty((len(world),), dtype=np.float32)
        gradients = np.empty((len(world), 3), dtype=np.float32)

        for i0 in range(0, len(world), chunk):
            i1 = min(i0 + chunk, len(world))
            sqr_d, face_id, closest = igl.point_mesh_squared_distance(
                np.ascontiguousarray(world[i0:i1]), vertices, faces
            )
            sqr_d = np.maximum(np.asarray(sqr_d, dtype=np.float64), 0.0)
            face_id = np.asarray(face_id, dtype=np.int64).reshape(-1)
            closest = np.asarray(closest, dtype=np.float64).reshape(-1, 3)
            d_world = np.sqrt(sqr_d)

            delta = world[i0:i1] - closest
            g = np.zeros_like(delta)
            regular = d_world > 1e-10
            g[regular] = delta[regular] / d_world[regular, None]
            if np.any(~regular):
                safe_face = np.clip(face_id[~regular], 0, len(normals) - 1)
                g[~regular] = normals[safe_face]

            distances[i0:i1] = (d_world / half).astype(np.float32)
            gradients[i0:i1] = g.astype(np.float32)

        return torch.from_numpy(distances), torch.from_numpy(gradients)

    return oracle


def main() -> None:
    parser = argparse.ArgumentParser(
        description="NSDUDF diagnostics using exact UDF/gradients from a mesh"
    )
    parser.add_argument("--mesh", required=True)
    parser.add_argument("--nsdudf-repo", required=True)
    parser.add_argument("--nsdudf-model", default=None)
    parser.add_argument("--grid", type=int, default=65)
    parser.add_argument("--output", required=True)
    parser.add_argument("--padding", type=float, default=0.10)
    parser.add_argument("--query-chunk", type=int, default=200000)
    parser.add_argument("--query-slab", type=int, default=4)
    parser.add_argument("--cell-slab", type=int, default=2)
    parser.add_argument("--classifier-batch-size", type=int, default=32768)
    parser.add_argument("--max-avg-factor", type=float, default=1.2)
    parser.add_argument("--max-max-factor", type=float, default=2.0)
    parser.add_argument("--loose-min-factor", type=float, default=1.0)
    parser.add_argument("--max-points", type=int, default=200000)
    parser.add_argument("--no-mesh", action="store_true")
    args = parser.parse_args()

    loaded = trimesh.load(args.mesh, process=False)
    if isinstance(loaded, trimesh.Scene):
        meshes = [g for g in loaded.geometry.values() if isinstance(g, trimesh.Trimesh)]
        if not meshes:
            raise ValueError(f"No triangle mesh found in {args.mesh}")
        mesh = trimesh.util.concatenate(meshes)
    elif isinstance(loaded, trimesh.Trimesh):
        mesh = loaded
    else:
        raise TypeError(f"Unsupported mesh type: {type(loaded)!r}")

    vertices = np.asarray(mesh.vertices, dtype=np.float64)
    faces = np.asarray(mesh.faces, dtype=np.int64)
    bmin = vertices.min(axis=0)
    bmax = vertices.max(axis=0)
    center = 0.5 * (bmin + bmax)
    half = 0.5 * float(np.max(bmax - bmin)) * (1.0 + float(args.padding))
    if half <= 0.0:
        raise ValueError("Reference mesh has a degenerate bounding box")

    nsd_meshing, model = _load_nsdudf(args.nsdudf_repo, args.nsdudf_model)
    oracle = _make_exact_mesh_oracle(
        vertices,
        faces,
        center=center,
        half=half,
        query_chunk=args.query_chunk,
    )

    options = DiagnosticOptions(
        n_grid_samples=args.grid,
        classifier_batch_size=args.classifier_batch_size,
        query_slab=args.query_slab,
        cell_slab=args.cell_slab,
        normalize_udf=True,
        max_avg_factor=args.max_avg_factor,
        max_max_factor=args.max_max_factor,
        loose_min_factor=args.loose_min_factor,
        max_visualization_points=args.max_points,
        extract_mesh=not args.no_mesh,
    )

    run_nsdudf_diagnostics(
        model,
        oracle,
        options=options,
        output_dir=args.output,
        domain_center=center,
        domain_half_extent=half,
        meshing_module=nsd_meshing,
    )


if __name__ == "__main__":
    main()
