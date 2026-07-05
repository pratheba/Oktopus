import os, pickle
import numpy as np
import trimesh
import copy
import os.path as op
import pymeshlab as ml
from time import time
from tqdm.autonotebook import tqdm
#from ngc.handle import Handle
from handle_3dvec import Handle

from sdf_closure_utils_v2 import (
    PerCurveClosedSDFEvaluator,
    prepare_closed_meshes,
)

import argparse
import ast

def meshlab_on_surface_sampling(shape_path, num_samples):
    ms = ml.MeshSet()
    ms.load_new_mesh(shape_path)
    ms.generate_sampling_poisson_disk(samplenum=num_samples)
    mesh = ms.current_mesh()

    verts = mesh.vertex_matrix()
    print("num on surface vertices = ", len(verts))
    return verts

def adaptive_surface_perturbation(
    handle,
    surface_samples,
    alpha_scales,
    tag="tmp",
    sigma_min=1e-4,
    sigma_max=5e-3,
    max_total_perturbed=None
):
    """
    surface_samples: raw on-surface xyz points
    alpha_scales: list like [0.02, 0.04, 0.06]

    Returns
    -------
    perturbed_samples : (M * len(alpha_scales), 3)
        Perturbed world-space samples, where M is the number of samples
        that survive prepare_samples / inside filtering.
    sigma_used : (M * len(alpha_scales),)
    band_id : (M * len(alpha_scales),)
    """

    local_data,_ = handle.prepare_samples(tag, surface_samples)

    base_xyz = np.asarray(local_data['samples']).astype(np.float32)
    rho = np.asarray(local_data['rho']).astype(np.float32).squeeze()

    print("input surface_samples.shape =", surface_samples.shape)
    print("localized base_xyz.shape     =", base_xyz.shape)
    print("localized rho.shape          =", rho.shape)

    if rho.ndim != 1:
        raise ValueError(f"Expected rho to be 1D after squeeze, got shape {rho.shape}")

    if base_xyz.shape[0] != rho.shape[0]:
        raise ValueError(
            f"Mismatch: base_xyz has {base_xyz.shape[0]} samples but rho has {rho.shape[0]}"
        )
    if max_total_perturbed is not None:
        max_base_points = max(1, int(max_total_perturbed // len(alpha_scales)))

        if base_xyz.shape[0] > max_base_points:
            ids = np.random.choice(base_xyz.shape[0], max_base_points, replace=False)
            base_xyz = base_xyz[ids]
            rho = rho[ids]

    rho = np.maximum(rho, 1e-8)

    perturbed_all = []
    sigma_all = []
    band_all = []

    for band_idx, alpha in enumerate(alpha_scales):
        sigma = np.clip(alpha * rho, sigma_min, sigma_max).astype(np.float32)

        noise_dir = np.random.normal(0.0, 1.0, size=base_xyz.shape).astype(np.float32)
        noise_dir /= (np.linalg.norm(noise_dir, axis=1, keepdims=True) + 1e-12)

        signed_mag = np.random.normal(0.0, sigma, size=base_xyz.shape[0]).astype(np.float32)
        perturbed = base_xyz + signed_mag[:, None] * noise_dir

        perturbed_all.append(perturbed)
        sigma_all.append(sigma)
        band_all.append(np.full(base_xyz.shape[0], band_idx, dtype=np.int32))

    perturbed_all = np.concatenate(perturbed_all, axis=0)
    sigma_all = np.concatenate(sigma_all, axis=0)
    band_all = np.concatenate(band_all, axis=0)

    return perturbed_all, sigma_all, band_all


def meshlab_shape_sampling(shape_path,  num_samples, noise_scale):
    # add noise 
    verts_around = []
    ms = ml.MeshSet()
    ms.load_new_mesh(shape_path)
    on_surface_samples = int(0.8 * num_samples)
    ms.generate_sampling_poisson_disk(samplenum=on_surface_samples)
    mesh = ms.current_mesh()

    on_surface_verts = mesh.vertex_matrix()
    print("num on surface vertices = ", len(on_surface_verts))
    #verts_around = verts

    verts_around = []
    off_surface_samples = int((0.25 * num_samples)/len(noise_scale))

    for ns in noise_scale:
        ms = ml.MeshSet()
        ms.load_new_mesh(shape_path)
        ms.generate_sampling_poisson_disk(samplenum=off_surface_samples)
        mesh = ms.current_mesh()

        verts = mesh.vertex_matrix()
        #print("num vertices ns = ", len(verts))
        vn = mesh.vertex_normal_matrix()
        vn /= np.linalg.norm(vn, axis=1, keepdims=True)
        noise = np.random.normal(0, ns, size=verts.shape[0])
        #verts_around = np.concatenate((verts_around, verts + noise[:, None]* vn), axis=0)
        if len(verts_around):
            verts_around = np.concatenate((verts_around, verts + noise[:, None]* vn), axis=0)
        else:
            verts_around = verts + noise[:, None]*vn
        #print("verts shape = ", verts.shape)
    print("verts shape = ", verts_around.shape)
    return on_surface_verts, verts_around

def meshlab_volumetric_sampling(shape_path, num_samples):
    ms = ml.MeshSet()
    ms.load_new_mesh(shape_path)
    ms.generate_sampling_volumetric(
        samplesurfradius = ml.PercentageValue(0.),
        samplevolnum = num_samples,
    )
    mesh = ms.mesh(1)
    verts = mesh.vertex_matrix()
    return verts

def bbox_volumetric_sampling(shape_path, num_samples):
    mesh = trimesh.load(shape_path, process=False)
    V = np.asarray(mesh.vertices)
    bmin, bmax = V.min(axis=0), V.max(axis=0)
    scales = bmax - bmin

    # [0,1]^3
    samples = np.random.rand(num_samples, 3)
    samples *= scales[None,:]
    samples += bmin
    return samples


def get_surface_samples_by_curve(handle, source="full"):
    """
    Pull per-curve surface samples already saved in std_handle.npz.

    source:
        "full" -> curve.core.surface_points_all
        "base" -> curve.core.surface_points_base
    """
    if source not in ["full", "base"]:
        raise ValueError(f"Unknown source={source}. Use 'full' or 'base'.")

    out = {}
    total = 0

    for curve in handle.curves:
        if source == "full":
            pts = getattr(curve.core, "surface_points_all", None)
        else:
            pts = getattr(curve.core, "surface_points_base", None)

        if pts is None:
            raise RuntimeError(
                f"Missing {source} surface samples for curve '{curve.name}'. "
                f"Expected them to be saved in std_handle.npz."
            )

        pts = np.asarray(pts, dtype=np.float64)

        if pts.ndim != 2 or pts.shape[1] != 3:
            raise ValueError(
                f"{source} surface samples for curve '{curve.name}' "
                f"must have shape (N,3), got {pts.shape}"
            )

        out[curve.name] = pts
        total += len(pts)

    print(f"[npz surface samples] source={source} total={total}")
    for name, pts in out.items():
        print(f"  {name}: {len(pts)}")

    return out


def prepare_samples_by_curve(handle, name, samples_by_curve):
    """
    Localize each curve only on the samples that belong to that curve.

    This replaces:
        handle.prepare_samples(name, one_global_surface_cloud)

    for on-surface and perturbed samples.
    """
    samples_glob = []
    samples_local = []
    coords = []
    angles = []
    radius = []
    rho = []
    rho_n = []
    cids = []

    inside_by_curve = {}

    for cid, curve in enumerate(handle.curves):
        part_samples = np.asarray(
            samples_by_curve.get(curve.name, np.zeros((0, 3))),
            dtype=np.float64,
        )

        if len(part_samples) == 0:
            inside_by_curve[curve.name] = np.zeros((0,), dtype=np.int64)
            print(f"[prepare_by_curve] {name} curve={curve.name} input=0 inside=0")
            continue

        curve_data, inside = curve.localize_samples(
            part_samples,
            update_curve=False,
            update_radius=False,
            name=name + "_" + str(cid),
        )

        inside = np.asarray(inside, dtype=np.int64)
        inside_by_curve[curve.name] = inside

        num_inside = len(inside)

        print(
            f"[prepare_by_curve] {name} curve={curve.name} "
            f"input={len(part_samples)} inside={num_inside}"
        )

        if num_inside == 0:
            continue

        samples_glob.append(curve_data["samples"])
        samples_local.append(curve_data["samples_local"])
        coords.append(curve_data["coords"])
        angles.append(curve_data["angles"])
        radius.append(curve_data["radius"])
        rho.append(curve_data["rho"])
        rho_n.append(curve_data["rho_n"])
        cids.append(np.full(num_inside, cid, dtype=int))

    if len(samples_glob) == 0:
        raise RuntimeError(f"No samples survived prepare_samples_by_curve for {name}.")

    data = {
        "samples": np.concatenate(samples_glob, axis=0),
        "samples_local": np.concatenate(samples_local, axis=0),
        "coords": np.concatenate(coords, axis=0),
        "curve_idx": np.concatenate(cids, axis=0),
        "angles": np.concatenate(angles, axis=0),
        "rho": np.concatenate(rho, axis=0),
        "rho_n": np.concatenate(rho_n, axis=0),
        "radius": np.concatenate(radius, axis=0),
    }

    meta = {
        "inside_by_curve": inside_by_curve,
    }

    return data, meta


def concat_prepared_data(data_a, data_b):
    """
    Concatenate already-localized prepared data.
    Used for:
        full-surface + base-surface
        full-perturbed + base-perturbed
    """
    keys = [
        "samples",
        "samples_local",
        "coords",
        "curve_idx",
        "angles",
        "rho",
        "rho_n",
        "radius",
    ]

    return {
        k: np.concatenate([data_a[k], data_b[k]], axis=0)
        for k in keys
    }


def _allocate_keep_counts(counts, max_total):
    """
    Allocate a global cap across curves approximately proportional to available count.
    """
    counts = np.asarray(counts, dtype=np.int64)

    if max_total is None or counts.sum() <= max_total:
        return counts.copy()

    max_total = int(max_total)
    raw = counts.astype(np.float64) / max(float(counts.sum()), 1.0) * max_total
    keep = np.floor(raw).astype(np.int64)

    positive = counts > 0
    if max_total >= int(np.sum(positive)):
        keep[(positive) & (keep == 0)] = 1

    # If floor/minimum overshoots, reduce from largest allocations.
    while keep.sum() > max_total:
        candidates = np.where(keep > 1)[0]
        if len(candidates) == 0:
            break
        idx = candidates[np.argmax(keep[candidates])]
        keep[idx] -= 1

    # Distribute remainder by largest fractional part.
    remainder = max_total - int(keep.sum())
    if remainder > 0:
        frac = raw - np.floor(raw)
        order = np.argsort(-frac)
        for idx in order:
            if remainder <= 0:
                break
            if keep[idx] < counts[idx]:
                keep[idx] += 1
                remainder -= 1

    keep = np.minimum(keep, counts)
    return keep


def adaptive_surface_perturbation_by_curve(
    handle,
    surface_samples_by_curve,
    alpha_scales,
    tag="tmp",
    sigma_min=1e-4,
    sigma_max=5e-3,
    max_total_perturbed=None,
):
    """
    Per-curve version of adaptive_surface_perturbation.

    Critical difference:
        - each curve perturbs only its own NPZ surface samples
        - perturbed points remain assigned to the same curve
    """

    localized_by_curve = {}
    counts = []

    # ------------------------------------------------------------
    # 1. Localize each curve's own on-surface samples
    # ------------------------------------------------------------
    for cid, curve in enumerate(handle.curves):
        pts = np.asarray(
            surface_samples_by_curve.get(curve.name, np.zeros((0, 3))),
            dtype=np.float64,
        )

        if len(pts) == 0:
            localized_by_curve[curve.name] = (
                np.zeros((0, 3), dtype=np.float32),
                np.zeros((0,), dtype=np.float32),
            )
            counts.append(0)
            continue

        local_data, _ = curve.localize_samples(
            pts,
            update_curve=False,
            update_radius=False,
            name=f"{tag}_{cid}",
        )

        base_xyz = np.asarray(local_data["samples"], dtype=np.float32)
        rho = np.asarray(local_data["rho"], dtype=np.float32).squeeze()

        if rho.ndim != 1:
            raise ValueError(
                f"Expected rho to be 1D for curve={curve.name}, got {rho.shape}"
            )

        if base_xyz.shape[0] != rho.shape[0]:
            raise ValueError(
                f"Mismatch for curve={curve.name}: "
                f"base_xyz={base_xyz.shape[0]}, rho={rho.shape[0]}"
            )

        localized_by_curve[curve.name] = (base_xyz, rho)
        counts.append(len(base_xyz))

    # ------------------------------------------------------------
    # 2. Optional global cap on number of base points perturbed
    # ------------------------------------------------------------
    if max_total_perturbed is not None:
        max_base_points = max(
            1,
            int(max_total_perturbed // max(len(alpha_scales), 1)),
        )
    else:
        max_base_points = None

    keep_counts = _allocate_keep_counts(counts, max_base_points)

    # ------------------------------------------------------------
    # 3. Perturb each curve independently
    # ------------------------------------------------------------
    perturbed_by_curve = {}
    sigma_by_curve = {}
    band_by_curve = {}

    for curve_idx, curve in enumerate(handle.curves):
        curve_name = curve.name
        base_xyz, rho = localized_by_curve[curve_name]

        n_keep = int(keep_counts[curve_idx])

        if n_keep < len(base_xyz):
            ids = np.random.choice(len(base_xyz), n_keep, replace=False)
            base_xyz = base_xyz[ids]
            rho = rho[ids]

        rho = np.maximum(rho, 1e-8)

        if len(base_xyz) == 0:
            perturbed_by_curve[curve_name] = np.zeros((0, 3), dtype=np.float32)
            sigma_by_curve[curve_name] = np.zeros((0,), dtype=np.float32)
            band_by_curve[curve_name] = np.zeros((0,), dtype=np.int32)
            continue

        perturbed_all = []
        sigma_all = []
        band_all = []

        for band_idx, alpha in enumerate(alpha_scales):
            sigma = np.clip(alpha * rho, sigma_min, sigma_max).astype(np.float32)

            noise_dir = np.random.normal(
                0.0,
                1.0,
                size=base_xyz.shape,
            ).astype(np.float32)

            noise_dir /= (
                np.linalg.norm(noise_dir, axis=1, keepdims=True) + 1e-12
            )

            signed_mag = np.random.normal(
                0.0,
                sigma,
                size=base_xyz.shape[0],
            ).astype(np.float32)

            perturbed = base_xyz + signed_mag[:, None] * noise_dir

            perturbed_all.append(perturbed)
            sigma_all.append(sigma)
            band_all.append(
                np.full(base_xyz.shape[0], band_idx, dtype=np.int32)
            )

        perturbed_by_curve[curve_name] = np.concatenate(perturbed_all, axis=0)
        sigma_by_curve[curve_name] = np.concatenate(sigma_all, axis=0)
        band_by_curve[curve_name] = np.concatenate(band_all, axis=0)

        print(
            f"[perturb_by_curve] {tag} curve={curve_name} "
            f"base={len(base_xyz)} perturbed={len(perturbed_by_curve[curve_name])}"
        )

    return perturbed_by_curve, sigma_by_curve, band_by_curve


def filter_curve_field_after_prepare(handle, field_by_curve, prepare_meta):
    """
    Reorder/filter sigma or band arrays to match prepare_samples_by_curve output.
    """
    out = []
    inside_by_curve = prepare_meta["inside_by_curve"]

    for curve in handle.curves:
        vals = np.asarray(field_by_curve[curve.name])
        inside = np.asarray(
            inside_by_curve.get(curve.name, np.zeros((0,), dtype=np.int64)),
            dtype=np.int64,
        )

        if len(inside) > 0:
            out.append(vals[inside])

    if len(out) == 0:
        return np.zeros((0,), dtype=np.float32)

    return np.concatenate(out, axis=0)


def export_handle_data(handle, graph_path, handle_path):
    handle.export_skeleton_mesh(handle_path)
    graph_data = handle.export_neural_graph()
    with open(op.join(graph_path, 'graph.pkl'), 'wb') as f:
        pickle.dump(graph_data, f)


def split_train_test(num_surface, num_space, num_on_surface):

    split = 0.8

    surface_ids = np.arange(num_surface)
    on_surface_ids = np.arange(num_on_surface)
    space_ids = np.arange(num_space)

    np.random.shuffle(surface_ids)
    np.random.shuffle(space_ids)
    np.random.shuffle(on_surface_ids)

    num_train_surface = int(split * num_surface)
    num_train_on_surface = int(split * num_on_surface)
    num_train_space = int(split * num_space)

    surface_train_ids = surface_ids[0:num_train_surface]
    on_surface_train_ids = on_surface_ids[0:num_train_on_surface]
    space_train_ids = space_ids[0:num_train_space]

    surface_val_ids = surface_ids[num_train_surface:]
    on_surface_val_ids = on_surface_ids[num_train_on_surface:]
    space_val_ids = space_ids[num_train_space:]

    return surface_train_ids, surface_val_ids, on_surface_train_ids, on_surface_val_ids, space_train_ids, space_val_ids

def get_full_base_residual_samples(
    handle,
    handle_mesh_file,
    full_closed_assets_by_curve,
    base_closed_assets_by_curve,
    n_surface_samples,
    n_space_samples,
    noise_scales=[0.02, 0.04, 0.06],
    name_prefix="",
    sigma_min=1e-4,
    sigma_max=5e-3,
    sdf_chunk_size=200000,
):
    """
    Surface locations:
        - unchanged from std_handle.npz
        - full samples come from surface_points_all
        - base samples come from surface_points_base

    Perturbation:
        - unchanged
        - generated only from existing NPZ surface points
        - no artificial cap points are added

    Volumetric locations:
        - unchanged
        - sampled from handle/std_mesh.ply

    SDF targets:
        - evaluated ONLY against the corresponding CLOSED part meshes
        - curve_idx selects the full/base closed part oracle
    """

    curve_names = [curve.name for curve in handle.curves]

    full_sdf_evaluator = PerCurveClosedSDFEvaluator(
        full_closed_assets_by_curve
    )
    base_sdf_evaluator = PerCurveClosedSDFEvaluator(
        base_closed_assets_by_curve
    )

    # ------------------------------------------------------------
    # 1. Load per-curve surface samples from NPZ-loaded handle
    # ------------------------------------------------------------
    on_surface_full_by_curve = get_surface_samples_by_curve(
        handle,
        source="full",
    )

    on_surface_base_by_curve = get_surface_samples_by_curve(
        handle,
        source="base",
    )

    n_full_surface_total = sum(len(v) for v in on_surface_full_by_curve.values())
    n_base_surface_total = sum(len(v) for v in on_surface_base_by_curve.values())

    print(
        "[surface totals]",
        "full=", n_full_surface_total,
        "base=", n_base_surface_total,
    )

    # `n_surface_samples` no longer controls actual on-surface samples.
    # Those were already allocated/sampled in update_radius.py.
    # We retain the argument only for CLI/backward compatibility.
    print(
        "[surface sampling]",
        "using NPZ part samples; --n_surface_samples is not used "
        "to resample mesh surfaces in this path.",
    )

    # ------------------------------------------------------------
    # 2. Adaptive perturbation from per-curve surface samples
    # ------------------------------------------------------------
    target_perturbed_full = int(0.25 * n_full_surface_total)
    target_perturbed_base = int(0.25 * n_base_surface_total)

    (
        surface_full_by_curve,
        full_sigma_by_curve,
        full_band_by_curve,
    ) = adaptive_surface_perturbation_by_curve(
        handle,
        on_surface_full_by_curve,
        alpha_scales=noise_scales,
        tag=f"{name_prefix}_full_for_rho" if name_prefix else "full_for_rho",
        sigma_min=sigma_min,
        sigma_max=sigma_max,
        max_total_perturbed=target_perturbed_full,
    )

    (
        surface_base_by_curve,
        base_sigma_by_curve,
        base_band_by_curve,
    ) = adaptive_surface_perturbation_by_curve(
        handle,
        on_surface_base_by_curve,
        alpha_scales=noise_scales,
        tag=f"{name_prefix}_base_for_rho" if name_prefix else "base_for_rho",
        sigma_min=sigma_min,
        sigma_max=sigma_max,
        max_total_perturbed=target_perturbed_base,
    )

    # ------------------------------------------------------------
    # 3. Volumetric samples stay global
    # ------------------------------------------------------------
    space_samples = meshlab_volumetric_sampling(
        handle_mesh_file,
        n_space_samples,
    )

    # ------------------------------------------------------------
    # 4. ON-SURFACE samples
    #    Localize FULL and BASE surfaces independently per curve,
    #    then concatenate.
    # ------------------------------------------------------------
    on_full_tag = f"{name_prefix}_on_full" if name_prefix else "on_full"
    on_base_tag = f"{name_prefix}_on_base" if name_prefix else "on_base"

    on_full_data, on_full_meta = prepare_samples_by_curve(
        handle,
        on_full_tag,
        on_surface_full_by_curve,
    )

    on_base_data, on_base_meta = prepare_samples_by_curve(
        handle,
        on_base_tag,
        on_surface_base_by_curve,
    )

    on_surface_data = concat_prepared_data(
        on_full_data,
        on_base_data,
    )

    if "rho" in on_surface_data:
        on_surface_data["surface_rho"] = np.asarray(
            on_surface_data["rho"],
            dtype=np.float32,
        )

    # Query points remain unchanged. Only the SDF oracle changes.
    on_surface_full_sdf = full_sdf_evaluator.evaluate(
        on_surface_data,
        curve_names=curve_names,
        chunk_size=sdf_chunk_size,
    )
    on_surface_base_sdf = base_sdf_evaluator.evaluate(
        on_surface_data,
        curve_names=curve_names,
        chunk_size=sdf_chunk_size,
    )

    on_surface_data["sdf"] = on_surface_full_sdf.astype(np.float32)
    on_surface_data["sdf_base"] = on_surface_base_sdf.astype(np.float32)
    on_surface_data["sdf_res"] = (
        on_surface_data["sdf"] - on_surface_data["sdf_base"]
    ).astype(np.float32)

    on_surface_data["sample_origin"] = np.concatenate([
        np.zeros(len(on_full_data["samples"]), dtype=np.int32),   # 0 = full surface
        np.ones(len(on_base_data["samples"]), dtype=np.int32),    # 1 = base surface
    ], axis=0)

    on_surface_inferencedata = {
        "full": on_full_meta,
        "base": on_base_meta,
    }

    # ------------------------------------------------------------
    # 5. OFF-SURFACE / perturbed samples
    #    Again: per-curve localization first, then concatenate.
    # ------------------------------------------------------------
    off_full_tag = f"{name_prefix}_off_full" if name_prefix else "off_full"
    off_base_tag = f"{name_prefix}_off_base" if name_prefix else "off_base"

    pert_full_data, pert_full_meta = prepare_samples_by_curve(
        handle,
        off_full_tag,
        surface_full_by_curve,
    )

    pert_base_data, pert_base_meta = prepare_samples_by_curve(
        handle,
        off_base_tag,
        surface_base_by_curve,
    )

    pert_surface_data = concat_prepared_data(
        pert_full_data,
        pert_base_data,
    )

    full_sigma_used = filter_curve_field_after_prepare(
        handle,
        full_sigma_by_curve,
        pert_full_meta,
    )

    base_sigma_used = filter_curve_field_after_prepare(
        handle,
        base_sigma_by_curve,
        pert_base_meta,
    )

    full_band_used = filter_curve_field_after_prepare(
        handle,
        full_band_by_curve,
        pert_full_meta,
    ).astype(np.int32)

    base_band_used = filter_curve_field_after_prepare(
        handle,
        base_band_by_curve,
        pert_base_meta,
    ).astype(np.int32)

    pert_surface_sigma = np.concatenate(
        [full_sigma_used, base_sigma_used],
        axis=0,
    ).astype(np.float32)

    pert_surface_band = np.concatenate(
        [full_band_used, base_band_used],
        axis=0,
    ).astype(np.int32)

    pert_surface_data["perturb_sigma"] = pert_surface_sigma
    pert_surface_data["perturb_band"] = pert_surface_band

    pert_surface_full_sdf = full_sdf_evaluator.evaluate(
        pert_surface_data,
        curve_names=curve_names,
        chunk_size=sdf_chunk_size,
    )
    pert_surface_base_sdf = base_sdf_evaluator.evaluate(
        pert_surface_data,
        curve_names=curve_names,
        chunk_size=sdf_chunk_size,
    )

    pert_surface_data["sdf"] = pert_surface_full_sdf.astype(np.float32)
    pert_surface_data["sdf_base"] = pert_surface_base_sdf.astype(np.float32)
    pert_surface_data["sdf_res"] = (
        pert_surface_data["sdf"] - pert_surface_data["sdf_base"]
    ).astype(np.float32)

    pert_surface_data["sample_origin"] = np.concatenate([
        np.zeros(len(pert_full_data["samples"]), dtype=np.int32),  # 0 = full perturb
        np.ones(len(pert_base_data["samples"]), dtype=np.int32),   # 1 = base perturb
    ], axis=0)

    # ------------------------------------------------------------
    # 6. SPACE / volumetric samples stay as before
    # ------------------------------------------------------------
    space_tag = f"{name_prefix}_space" if name_prefix else "space"

    print("space_sample", space_samples.shape)
    space_data, _ = handle.prepare_samples(space_tag, space_samples)
    print("space data after prepare samples", space_data["samples_local"].shape)

    space_full_sdf = full_sdf_evaluator.evaluate(
        space_data,
        curve_names=curve_names,
        chunk_size=sdf_chunk_size,
    )
    space_base_sdf = base_sdf_evaluator.evaluate(
        space_data,
        curve_names=curve_names,
        chunk_size=sdf_chunk_size,
    )

    space_data["sdf"] = space_full_sdf.astype(np.float32)
    space_data["sdf_base"] = space_base_sdf.astype(np.float32)
    space_data["sdf_res"] = (
        space_data["sdf"] - space_data["sdf_base"]
    ).astype(np.float32)

    return (
        on_surface_data,
        pert_surface_data,
        space_data,
        on_surface_inferencedata,
    )


def ngc_dataset(arg):
    root_path = arg['root_path']
    data_path = arg['data_path']
    file_name = arg['file_name']
    n_surface_samples = arg['n_surface_samples']
    n_space_samples = arg['n_space_samples']
    noise_scales = arg['noise_scales']
    overwrite = arg['overwrite']
    n_keypoints = arg['n_keypoints']

    global_full_mesh_spec = arg['global_full_mesh']
    global_base_mesh_spec = arg['global_base_mesh']
    full_parts_dir_spec = arg['full_parts_dir']
    base_parts_dir_spec = arg['base_parts_dir']
    full_part_pattern = arg['full_part_pattern']
    base_part_pattern = arg['base_part_pattern']
    closed_mesh_dir_spec = arg['closed_mesh_dir']
    reclose_meshes = bool(arg.get('reclose_meshes', False))
    sdf_chunk_size = int(arg.get('sdf_chunk_size', 200000))

    # items = os.listdir(root_path)
    items = np.atleast_1d(np.loadtxt(op.join(root_path, data_path), dtype=str)).tolist()

    with tqdm(total=len(items)) as pbar:
        for name in items:
            name, shape_type = name.split('|')
            item_path = op.join(root_path, f'{name}')
            # Source mesh paths are supplied through CLI options.
            handle_path = op.join(item_path, 'handle')
            #print("handle_path = ", handle_path)
            handle_file = op.join(handle_path, 'std_handle.npz')
            #handle_file = op.join(handle_path, 'std_handle.pkl')
            handle_mesh_file = op.join(handle_path, 'std_mesh.ply')
            output_train_path = op.join(item_path, 'train_data') # str(n_keypoints))
            output_val_path = op.join(item_path, 'val_data') # str(n_keypoints))
            output_all_path = op.join(item_path, 'all_data')# str(n_keypoints))
            os.makedirs(output_train_path, exist_ok=True) 
            os.makedirs(output_val_path, exist_ok=True)
            os.makedirs(output_all_path, exist_ok=True)

            output_train_file = op.join(output_train_path, file_name)
            output_val_file = op.join(output_val_path, file_name)
            output_all_file = op.join(output_all_path, file_name)
            output_all_inferencefile = op.join(handle_path, 'inference.npz')
            #output_path = os.path.join(item_


            #if not overwrite: #op.exists(output_all_file):
            if (not overwrite) and op.exists(output_all_file):
                print('Exists: ', item_path, flush=True)
                pbar.update(1)
                continue

            handle = Handle()
            handle.load(handle_file, shape_type, n_keypoints)

            # --------------------------------------------------------
            # Mandatory closure stage.
            #
            # Close and cache:
            #   1. global full mesh
            #   2. global base mesh
            #   3. every full part mesh
            #   4. every base part mesh
            #
            # All SDF values are then evaluated only against the closed
            # per-part meshes. No cap points are added to the samples.
            # --------------------------------------------------------
            (
                global_full_closed,
                global_base_closed,
                full_closed_assets_by_curve,
                base_closed_assets_by_curve,
            ) = prepare_closed_meshes(
                item_path=item_path,
                item_name=name,
                handle=handle,
                global_full_mesh_spec=global_full_mesh_spec,
                global_base_mesh_spec=global_base_mesh_spec,
                full_parts_dir_spec=full_parts_dir_spec,
                base_parts_dir_spec=base_parts_dir_spec,
                full_part_pattern=full_part_pattern,
                base_part_pattern=base_part_pattern,
                closed_mesh_dir_spec=closed_mesh_dir_spec,
                reclose=reclose_meshes,
            )

            print(
                "[closed global full]", global_full_closed.closed_path
            )
            print(
                "[closed global base]", global_base_closed.closed_path
            )
            print("full_closed_assets_by_curve", full_closed_assets_by_curve)
            #exit()

            #if not op.exists(handle_mesh_file):
            #    export_handle_data(handle, output_path, handle_path)
            export_handle_data(handle, item_path, handle_path)

            on_surface_data, pert_surface_data, space_data, on_surface_inferencedata = (
                get_full_base_residual_samples(
                    handle=handle,
                    handle_mesh_file=handle_mesh_file,
                    full_closed_assets_by_curve=full_closed_assets_by_curve,
                    base_closed_assets_by_curve=base_closed_assets_by_curve,
                    n_surface_samples=n_surface_samples,
                    n_space_samples=n_space_samples,
                    noise_scales=noise_scales,
                    name_prefix=name,
                    sdf_chunk_size=sdf_chunk_size,
                )
            )

            all_data = {
                'on_surface': on_surface_data,
                'pert_surface': pert_surface_data,
                'space': space_data,

                'base_on_surface_sdf': on_surface_data['sdf_base'],
                'base_pert_surface_sdf': pert_surface_data['sdf_base'],
                'base_space_sdf': space_data['sdf_base'],

                'residual_on_surface_sdf': on_surface_data['sdf_res'],
                'residual_pert_surface_sdf': pert_surface_data['sdf_res'],
                'residual_space_sdf': space_data['sdf_res']
            }

            with open(output_all_file, 'wb') as f:
                pickle.dump(all_data, f)

            #np.savez(output_all_inferencefile, on_surface_inferencedata)

            pbar.update(1)

    print('Done')

if __name__ == "__main__":
    # np.random.seed(2024)

    #root_path = '/path/to/dataset'
    p = argparse.ArgumentParser(description='Input to preoprocessing')
    p.add_argument('-r', '--root_path', required=True)
    p.add_argument('-d', '--data_path', required=True)
    p.add_argument('-w', '--overwrite', action="store_true")
    p.add_argument('-k', '--n_keypoints', type=int, required=True)
    p.add_argument('-ns', '--noise_scales', type=ast.literal_eval, default=[0.02], required=True)
    p.add_argument('-ss', '--n_surface_samples', type=int, default=320000, required=True)
    p.add_argument('-sv', '--n_space_samples', type=int, default=40000, required=True)

    p.add_argument(
        '--global_full_mesh',
        default='mesh.ply',
        help=(
            'Global full source mesh. Absolute path, path relative to each '
            'item, or template using {item_path} and {name}.'
        ),
    )
    p.add_argument(
        '--global_base_mesh',
        default='mesh_base.ply',
        help=(
            'Global base source mesh. Absolute path, path relative to each '
            'item, or template using {item_path} and {name}.'
        ),
    )
    p.add_argument(
        '--full_parts_dir',
        default='full_parts',
        help=(
            'Directory containing open full part meshes. Absolute, relative '
            'to each item, or template using {item_path} and {name}.'
        ),
    )
    p.add_argument(
        '--base_parts_dir',
        default='base_parts',
        help=(
            'Directory containing open base part meshes. Absolute, relative '
            'to each item, or template using {item_path} and {name}.'
        ),
    )
    p.add_argument(
        '--full_part_pattern',
        default='{id}_{name}.ply',
        help=(
            'Full part filename pattern. Available fields: '
            '{id}, {name}, {raw_name}.'
        ),
    )
    p.add_argument(
        '--base_part_pattern',
        default='{id}_{name}.ply',
        help=(
            'Base part filename pattern. Available fields: '
            '{id}, {name}, {raw_name}.'
        ),
    )
    p.add_argument(
        '--closed_mesh_dir',
        default='closed_sdf_meshes',
        help=(
            'Output/cache directory for closed global and part meshes. '
            'Absolute, relative to each item, or template using '
            '{item_path} and {name}.'
        ),
    )
    p.add_argument(
        '--reclose_meshes',
        action='store_true',
        help='Ignore cached closed meshes and close all source meshes again.',
    )
    p.add_argument(
        '--sdf_chunk_size',
        type=int,
        default=200000,
        help='Maximum number of points in one closed-mesh SDF query.',
    )

    args = p.parse_args()
    arg = {
        'root_path': args.root_path,
        'data_path': args.data_path,
        'overwrite': args.overwrite,
        'n_keypoints': args.n_keypoints,
        'file_name': 'sdf_samples.pkl',
        'n_surface_samples' : args.n_surface_samples,
        'n_space_samples' : args.n_space_samples,
        'noise_scales': args.noise_scales,
        'global_full_mesh': args.global_full_mesh,
        'global_base_mesh': args.global_base_mesh,
        'full_parts_dir': args.full_parts_dir,
        'base_parts_dir': args.base_parts_dir,
        'full_part_pattern': args.full_part_pattern,
        'base_part_pattern': args.base_part_pattern,
        'closed_mesh_dir': args.closed_mesh_dir,
        'reclose_meshes': args.reclose_meshes,
        'sdf_chunk_size': args.sdf_chunk_size,
    }
    ngc_dataset(arg)
