import os
import re
import glob
import numpy as np
from copy import deepcopy
import postprocess
import argparse
import json

# reuse from your postprocess.py
from postprocess import (
     load_segments, reassign,
     compute_parallel_transport_frames,
     nearest_polyline_projection,
     radii_from_support_local_frame,
 )
from plyfile import PlyData



def save_segments(path, out_dir, segs):
    print(path)
    print(out_dir)
    np.savez_compressed(path, segments=np.array(segs, dtype=object))
    os.makedirs(out_dir, exist_ok=True)

    summary = []
    for s in segs:
        fp = os.path.join(out_dir, f"segment_{int(s['id'])}.npz")
        np.savez_compressed(fp, segment=np.array(s, dtype=object))
        summary.append(
            {
                "id": int(s["id"]),
                "name": s.get("name", ""),
                "n_keypoints": int(len(s["keypoints"])),
                "n_surface_all": int(len(s.get("surface_points_all", []))),
                "file": fp,
            }
        )

    json.dump({"segments": summary}, open(path.replace(".npz", "_summary.json"), "w"), indent=2)

def load_ply_xyz(path):
    ply = PlyData.read(path)
    v = ply["vertex"].data

    x = np.asarray(v["x"], dtype=np.float64)
    y = np.asarray(v["y"], dtype=np.float64)
    z = np.asarray(v["z"], dtype=np.float64)

    return np.stack([x, y, z], axis=1)


def recompute_segment_fields_preserve_owned(seg):
    key = np.asarray(seg["keypoints"], dtype=np.float64)
    pts_all = np.asarray(seg.get("surface_points_all", np.zeros((0, 3))), dtype=np.float64)
    pts_owned = np.asarray(seg.get("surface_points_owned", pts_all), dtype=np.float64)

    T, U, V, frames = compute_parallel_transport_frames(key)

    _, point_s, point_key_ids, _ = nearest_polyline_projection(key, pts_all)

    # radii/support are recomputed from surface_points_all, matching your current reassign behavior
    r_train, r_wrap, r_cyl = radii_from_support_local_frame(key, T, U, V, pts_all)

    seg["point_s"] = point_s
    seg["point_key_ids"] = point_key_ids
    seg["radius_train"] = r_train
    seg["radius_wrap"] = r_wrap
    seg["radius_cylinder"] = r_cyl
    seg["surface_points_owned"] = pts_owned
    seg["surface_points_shared"] = np.zeros((0, 3), dtype=np.float64)
    seg["frame_t"] = T
    seg["frame_u"] = U
    seg["frame_v"] = V
    seg["frames"] = frames

    meta = dict(seg.get("metadata", {}))
    meta["n_total_keypoints"] = int(len(key))
    meta["has_local_frames"] = True
    meta["manual_ply_patch"] = True
    seg["metadata"] = meta
    return seg


def patch_segments_from_ply_folder(in_npz, ply_dir, out_npz, out_dir, update_keypoints=True):
    segs = load_segments(in_npz)
    by_id = {int(s["id"]): deepcopy(s) for s in segs}

    # Match files like:
    # 0_keypoints.ply
    # 0_surface_points_all.ply
    # 0_surface_points_owned.ply
    pat = re.compile(r"^(\d+)_(keypoints|surface_points_all|surface_points_owned)\.ply$")

    grouped = {}
    for fp in glob.glob(os.path.join(ply_dir, "*.ply")):
        name = os.path.basename(fp)
        m = pat.match(name)
        if not m:
            continue
        sid = int(m.group(1))
        kind = m.group(2)
        grouped.setdefault(sid, {})[kind] = fp

    for sid, files in grouped.items():
        if sid not in by_id:
            print(f"[skip] segment id {sid} not found in old NPZ")
            continue

        seg = by_id[sid]

        if update_keypoints and "keypoints" in files:
            seg["keypoints"] = load_ply_xyz(files["keypoints"])

        if "surface_points_all" in files:
            seg["surface_points_all"] = load_ply_xyz(files["surface_points_all"])

        if "surface_points_owned" in files:
            seg["surface_points_owned"] = load_ply_xyz(files["surface_points_owned"])
        elif "surface_points_all" in files:
            # if only all is supplied, keep behavior simple: owned := all
            seg["surface_points_owned"] = np.asarray(seg["surface_points_all"], dtype=np.float64).copy()

        seg = recompute_segment_fields_preserve_owned(seg)
        by_id[sid] = seg
        print(f"[updated] segment {sid}: {list(files.keys())}")

    print(out_dir)
    save_segments(
        out_npz,
        out_dir,
        sorted(by_id.values(), key=lambda s: int(s["id"]))
    )


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_npz", required=True)
    ap.add_argument("--ply_dir", required=True)
    ap.add_argument("--out_npz", required=True)
    ap.add_argument("--out_dir", required=True)
    args = ap.parse_args()

    patch_segments_from_ply_folder(
        in_npz=args.in_npz,
        ply_dir=args.ply_dir,
        out_npz=args.out_npz,
        out_dir=args.out_dir,
        update_keypoints=True,   # set False if you only changed surfaces
    )

#    patch_segments_from_ply_folder(
#        in_npz="/fast/pselvaraju/Oktopus_now/preprocess/skeletal_extraction/postprocess/output/npz/oktopus_unitbb_300k_9_postprocess_segments_v1.npz",
#        ply_dir="/fast/pselvaraju/Oktopus_now/preprocess/skeletal_extraction/postprocess/output/oktopus_unitbb_300k/segments/oktopus_unitbb_300k_9_v1",
#        out_npz="patched_oktopus_unitbb_300k_9.npz",
#        out_dir="/fast/pselvaraju/Oktopus_now/preprocess/skeletal_extraction/postprocess/output/oktopus_unitbb_300k/segments/",
#        update_keypoints=True,   # set False if you only changed surfaces
#    )
