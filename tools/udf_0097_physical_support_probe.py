#!/usr/bin/env python3
"""0097: READ-ONLY physical-scale UDF evidence connectivity diagnostic.

Tests whether the *saved* 0074 face scores that were fragmented by 0079 at
high resolution can form accepted evidence islands when scored faces are
connected by short paths along the source mesh. Does NOT call the learned
UDF, change any YAML, edit source meshes, or write a production removal mask.

IMPORTANT: a positive result is NOT permission to delete those faces. This
probe bridges score support; it does not prove that scores themselves should
be smoothed or that the resulting cut boundary is safe.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import trimesh
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components, dijkstra


def find_shape(run: Path, shape: str) -> Path:
    for d in (run / shape, run / "_udf_inference_0084" / shape, run):
        if (d / "udf_cut_curve_local_0081/body/locality.npz").is_file():
            return d
    raise FileNotFoundError(f"No saved 0081 body under {run}")


def read_case(run: Path, shape: str, override_mesh: Path | None = None) -> dict:
    p = find_shape(run, shape)
    report = json.loads((p / "riggable_after_udf_cut_report.json").read_text())
    owner = report["parts"]["body"]["direct_udf"]["owners"]
    if len(owner) != 1 or int(owner[0]["item_index"]) != 0:
        raise ValueError("0097 handles the single-owner body only; refuses ambiguous owners")
    path = override_mesh or Path(report["parts"]["body"]["input"])
    if not path.is_file():
        raise FileNotFoundError(f"Saved SDF source mesh unavailable: {path}")
    mesh = trimesh.load(path, force="mesh", process=False)
    with np.load(p / "udf_cut_curve_local_0081/body/locality.npz") as z:
        arc = z["face_curve_arclength"].astype(np.float64)
        saved_0081_seed = z["seed_faces"].astype(bool)
        expected_centroids = z["face_centroids_xyz"]
    n = len(mesh.faces)
    if len(arc) != n:
        raise ValueError("UDF arc and source mesh have different face counts")
    probe = np.unique(np.linspace(0, n - 1, min(n, 256), dtype=np.int64))
    actual = np.asarray(mesh.vertices)[np.asarray(mesh.faces)[probe]].mean(axis=1)
    if np.max(np.linalg.norm(actual - expected_centroids[probe], axis=1)) > 5e-6:
        raise ValueError("Source mesh and saved 0081 evidence are not aligned")
    with np.load(p / "udf_cut_evidence_0074/body/face_evidence.npz") as z:
        ids = z["item_00_faces"].astype(np.int64)
        raw = z["item_00_raw_scores"].astype(np.float64)
        effective = z["item_00_effective_valid"].astype(bool)
    with np.load(p / "udf_cut_reconcile_0079/body/votes.npz") as z:
        saved_remove_votes = z["remove_votes"].astype(np.int64)
    scores = np.full(n, np.nan, dtype=np.float64)
    valid = np.zeros(n, dtype=bool)
    scores[ids] = raw
    valid[ids] = effective & np.isfinite(raw)
    if len(saved_remove_votes) != n:
        raise ValueError("Stored vote arrays do not align with source mesh")
    return dict(label=str(run), mesh=mesh, arc=arc, scores=scores,
                valid=valid, saved_votes=saved_remove_votes > 0,
                saved_0081_seed=saved_0081_seed,
                seed=float(owner[0]["seed_world"]),
                grow=float(owner[0]["grow_world"]))


def adjacency_graph_for_window(mesh: trimesh.Trimesh, eligible: np.ndarray,
                               arc: np.ndarray, *, max_arc_step: float,
                               min_normal_dot: float):
    """Surface-edge graph; no Euclidean links across disconnected/folded surfaces."""
    selected = np.flatnonzero(eligible)
    inv = np.full(len(mesh.faces), -1, dtype=np.int64)
    inv[selected] = np.arange(len(selected))
    edges = np.asarray(mesh.face_adjacency, dtype=np.int64)
    ok = eligible[edges[:, 0]] & eligible[edges[:, 1]]
    edges = edges[ok]
    if len(edges):
        ok = np.abs(arc[edges[:, 0]] - arc[edges[:, 1]]) <= max_arc_step
        edges = edges[ok]
    if len(edges):
        normals = np.asarray(mesh.face_normals)
        ok = np.einsum("ij,ij->i", normals[edges[:, 0]], normals[edges[:, 1]]) >= min_normal_dot
        edges = edges[ok]
    if len(edges):
        ctr = np.asarray(mesh.triangles_center)
        lengths = np.maximum(np.linalg.norm(ctr[edges[:, 0]] - ctr[edges[:, 1]], axis=1), 1e-12)
        a, b = inv[edges[:, 0]], inv[edges[:, 1]]
        weighted = coo_matrix((np.r_[lengths, lengths],
                               (np.r_[a, b], np.r_[b, a])),
                              shape=(len(selected), len(selected))).tocsr()
        binary = coo_matrix((np.ones(2 * len(edges), dtype=np.uint8),
                             (np.r_[a, b], np.r_[b, a])),
                            shape=(len(selected), len(selected))).tocsr()
    else:
        weighted = coo_matrix((len(selected), len(selected))).tocsr()
        binary = coo_matrix((len(selected), len(selected)), dtype=np.uint8).tocsr()
    return selected, weighted, binary


def accepted_components(binary, supported, seeds, areas, boundary_faces, *,
                        min_faces=8, min_seed_faces=2, min_seed_fraction=.10):
    """Replay 0079 component-count/seed-fraction/boundary gates in one arc window."""
    act = np.flatnonzero(supported)
    if not len(act):
        return np.zeros(len(supported), dtype=bool), dict(components=0, accepted_components=0,
                    accepted_faces=0, accepted_seed_faces=0, rejected_counts={})
    g = binary[act][:, act]
    num, labels = connected_components(g, directed=False)
    sizes = np.bincount(labels, minlength=num)
    scount = np.bincount(labels, weights=seeds[act].astype(np.int64), minlength=num)
    bcount = np.bincount(labels, weights=boundary_faces[act].astype(np.int64), minlength=num)
    good = ((sizes >= min_faces) & (scount >= min_seed_faces) &
            (scount / np.maximum(1, sizes) >= min_seed_fraction) & (bcount == 0))
    acc = np.zeros(len(supported), dtype=bool)
    acc[act] = good[labels]
    reasons = {
        "too_few_faces": int(np.count_nonzero(sizes < min_faces)),
        "too_few_seed_faces": int(np.count_nonzero((sizes >= min_faces) & (scount < min_seed_faces))),
        "low_seed_fraction": int(np.count_nonzero((sizes >= min_faces) & (scount >= min_seed_faces) &
                                                  (scount / np.maximum(1, sizes) < min_seed_fraction))),
        "touches_boundary": int(np.count_nonzero((sizes >= min_faces) & (scount >= min_seed_faces) &
                                                (scount / np.maximum(1, sizes) >= min_seed_fraction) &
                                                (bcount > 0))),
    }
    return acc, dict(components=int(num), accepted_components=int(np.count_nonzero(good)),
                     accepted_faces=int(np.count_nonzero(acc)),
                     accepted_seed_faces=int(np.count_nonzero(acc & seeds)), rejected_counts=reasons)


def boundary_face_mask(faces: np.ndarray) -> np.ndarray:
    """Original mesh edge incidence, not artificial boundaries of a cropped arc window."""
    f = np.asarray(faces, dtype=np.int64)
    n = len(f)
    edges = np.sort(np.concatenate((f[:, (0, 1)], f[:, (1, 2)], f[:, (2, 0)]), axis=0), axis=1)
    _, inverse, counts = np.unique(edges, axis=0, return_inverse=True, return_counts=True)
    fid = np.tile(np.arange(n, dtype=np.int64), 3)
    out = np.zeros(n, dtype=bool)
    out[fid[counts[inverse] == 1]] = True
    return out


def probe_window(case: dict, *, low: float, high: float, radii: tuple[float, ...],
                 max_arc_step: float, min_normal_dot: float,
                 use_boundary_mask: bool = True) -> dict:
    mesh, s = case["mesh"], case["arc"]
    scores, valid = case["scores"], case["valid"]
    region = (s >= low) & (s < high)
    eligible = region & valid
    selected, graph, binary = adjacency_graph_for_window(mesh, eligible, s,
                        max_arc_step=max_arc_step, min_normal_dot=min_normal_dot)
    strong = scores[selected] >= case["seed"]
    grow = scores[selected] >= case["grow"]
    boundary = case["boundary"][selected] if use_boundary_mask else np.zeros(len(selected), bool)
    areas = np.asarray(mesh.area_faces)[selected]
    existing_seed = int(np.count_nonzero(case["saved_votes"] & region))
    baseline, base_stat = accepted_components(binary, grow, strong, areas, boundary)
    results = []
    dist = None
    for radius in radii:
        if radius > 0 and np.any(strong):
            if dist is None:
                dist = dijkstra(graph, directed=False, indices=np.flatnonzero(strong),
                                limit=max(radii), min_only=True)
            support = grow | (dist <= radius)
        else:
            support = grow.copy()
        acc, stats = accepted_components(binary, support, strong, areas, boundary)
        results.append(dict(radius_world=float(radius),
                            supported_faces=int(np.count_nonzero(support)),
                            extra_supported_faces=int(np.count_nonzero(support & ~grow)),
                            accepted_from_strong_faces=int(np.count_nonzero(acc & strong)),
                            **stats))
    return dict(arc=[low, high], total_faces=int(region.sum()),
                valid_faces=int(eligible.sum()), strong_faces=int(strong.sum()),
                grow_faces=int(grow.sum()),
                stored_0079_removed_votes=int(existing_seed),
                stored_0081_seed_faces=int(np.count_nonzero(case["saved_0081_seed"] & region)),
                baseline_window_only=base_stat,
                baseline_vs_saved_difference_note=(
                    "Window replay truncates connected components at arc bounds; it is not full-mesh 0079 parity."),
                radii=results)


def probe_case(case: dict, radii: tuple[float, ...], *,
               max_arc_step: float, min_normal_dot: float) -> dict:
    # Cache full-mesh boundary test once for both windows (avoid duplicate memory work).
    # Surface construction is intentionally unmodified.
    mesh = case["mesh"]
    boundary = boundary_face_mask(np.asarray(mesh.faces))
    case["boundary"] = boundary
    cap = probe_window(case, low=.885, high=.900, radii=radii,
                       max_arc_step=max_arc_step, min_normal_dot=min_normal_dot)
    neck = probe_window(case, low=.400, high=.450, radii=radii,
                        max_arc_step=max_arc_step, min_normal_dot=min_normal_dot)
    return dict(run=case["label"], mesh_faces=int(len(mesh.faces)), seed=case["seed"],
                grow=case["grow"], cap_edge=cap, working_control=neck)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--new", type=Path, required=True, help="Failing 1024 UDF run")
    ap.add_argument("--old", type=Path, help="Optional working 512 UDF run")
    ap.add_argument("--new-mesh", type=Path, help="Override saved body source PLY")
    ap.add_argument("--old-mesh", type=Path)
    ap.add_argument("--shape", default="oktopus_9_v1")
    ap.add_argument("--radii", default="0,0.002,0.004,0.006",
                    help="Physical geodesic support radius in world units")
    ap.add_argument("--max-arc-step", type=float, default=.02)
    ap.add_argument("--min-normal-dot", type=float, default=.5)
    ap.add_argument("--out", type=Path, default=Path("/tmp/puffer_0097_physical_support.json"))
    a = ap.parse_args()
    radii = tuple(float(x) for x in a.radii.split(","))
    if not radii or any(not np.isfinite(r) or r < 0 for r in radii):
        ap.error("radii must be nonnegative finite numbers")
    if a.max_arc_step <= 0 or not -1 <= a.min_normal_dot <= 1:
        ap.error("invalid arc or normal guard")
    output = {}
    for key, run, meshfile in (("new", a.new, a.new_mesh), ("old", a.old, a.old_mesh)):
        if run is None:
            continue
        case = read_case(run, a.shape, meshfile)
        output[key] = probe_case(case, radii, max_arc_step=a.max_arc_step,
                                 min_normal_dot=a.min_normal_dot)
        print(f"\n{key.upper()}: {run}")
        for section in ("cap_edge", "working_control"):
            c = output[key][section]
            print(f"  {section} arc={c['arc']} strong={c['strong_faces']} "
                  f"stored_votes={c['stored_0079_removed_votes']} baseline_window={c['baseline_window_only']['accepted_seed_faces']}")
            for r in c["radii"]:
                print(f"    radius={r['radius_world']:.4g} added_support={r['extra_supported_faces']:>6} "
                      f"accepted_components={r['accepted_components']:>5} "
                      f"accepted_strong={r['accepted_from_strong_faces']:>6} "
                      f"accepted_faces={r['accepted_faces']:>6}")
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(output, indent=2))
    print(f"\nSaved read-only diagnostic: {a.out}")
    print("NO CUT OR FACE-REMOVAL MASK HAS BEEN WRITTEN. THIS IS NOT A PRODUCTION FIX.")


if __name__ == "__main__":
    main()
