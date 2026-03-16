import os
import numpy as np
import trimesh
import matplotlib.pyplot as plt


def _make_point_cloud(points, color=(140, 140, 140, 60)):
    points = np.asarray(points, dtype=np.float64)
    colors = np.tile(np.array(color, dtype=np.uint8), (len(points), 1))
    return trimesh.points.PointCloud(points, colors=colors)


def _make_spheres(points, radius=0.003, color=(255, 0, 255, 255)):
    meshes = []
    for p in np.asarray(points):
        s = trimesh.creation.icosphere(subdivisions=1, radius=radius)
        s.apply_translation(p)
        s.visual.vertex_colors = np.tile(np.array(color, dtype=np.uint8), (len(s.vertices), 1))
        meshes.append(s)
    return trimesh.util.concatenate(meshes) if meshes else None


def _make_segments(p0, p1, color=(255, 255, 0, 255), tube_radius=0.0012, sections=8):
    p0 = np.asarray(p0, dtype=np.float64)
    p1 = np.asarray(p1, dtype=np.float64)
    valid = np.isfinite(p0).all(axis=1) & np.isfinite(p1).all(axis=1)
    p0 = p0[valid]
    p1 = p1[valid]
    if len(p0) == 0:
        return None

    meshes = []
    for a, b in zip(p0, p1):
        seg = b - a
        seg_len = np.linalg.norm(seg)
        if seg_len < 1e-10:
            continue

        cyl = trimesh.creation.cylinder(radius=tube_radius, height=seg_len, sections=sections)

        z_axis = np.array([0.0, 0.0, 1.0])
        direction = seg / seg_len

        v = np.cross(z_axis, direction)
        c = np.dot(z_axis, direction)

        if np.linalg.norm(v) < 1e-12:
            if c > 0:
                R = np.eye(3)
            else:
                R = np.array([[1,0,0],[0,-1,0],[0,0,-1]], dtype=np.float64)
        else:
            vx = np.array([
                [0, -v[2], v[1]],
                [v[2], 0, -v[0]],
                [-v[1], v[0], 0]
            ], dtype=np.float64)
            R = np.eye(3) + vx + vx @ vx * (1.0 / (1.0 + c))

        T = np.eye(4)
        T[:3, :3] = R
        T[:3, 3] = 0.5 * (a + b)

        cyl.apply_transform(T)
        cyl.visual.vertex_colors = np.tile(np.array(color, dtype=np.uint8), (len(cyl.vertices), 1))
        meshes.append(cyl)

    if len(meshes) == 0:
        return None
    return trimesh.util.concatenate(meshes)

def _make_polyline(points, closed=False, color=(255, 255, 255, 255), tube_radius=0.0015, sections=10):
    points = np.asarray(points, dtype=np.float64)
    points = points[np.isfinite(points).all(axis=1)]
    if len(points) < 2:
        return None

    meshes = []

    if closed:
        pts0 = points
        pts1 = np.roll(points, -1, axis=0)
    else:
        pts0 = points[:-1]
        pts1 = points[1:]

    for p0, p1 in zip(pts0, pts1):
        seg = p1 - p0
        seg_len = np.linalg.norm(seg)
        if seg_len < 1e-10:
            continue

        cyl = trimesh.creation.cylinder(radius=tube_radius, height=seg_len, sections=sections)

        # cylinder is along +Z by default; rotate it to segment direction
        z_axis = np.array([0.0, 0.0, 1.0])
        direction = seg / seg_len

        v = np.cross(z_axis, direction)
        c = np.dot(z_axis, direction)

        if np.linalg.norm(v) < 1e-12:
            if c > 0:
                R = np.eye(3)
            else:
                # 180-degree flip around x
                R = np.array([[1,0,0],[0,-1,0],[0,0,-1]], dtype=np.float64)
        else:
            vx = np.array([
                [0, -v[2], v[1]],
                [v[2], 0, -v[0]],
                [-v[1], v[0], 0]
            ], dtype=np.float64)
            R = np.eye(3) + vx + vx @ vx * (1.0 / (1.0 + c))

        T = np.eye(4)
        T[:3, :3] = R
        T[:3, 3] = 0.5 * (p0 + p1)

        cyl.apply_transform(T)
        cyl.visual.vertex_colors = np.tile(np.array(color, dtype=np.uint8), (len(cyl.vertices), 1))
        meshes.append(cyl)

    if len(meshes) == 0:
        return None
    return trimesh.util.concatenate(meshes)

def _ellipse_points_world(C, N, B, ru, rv, n=64):
    theta = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
    pts = (
        C[None, :]
        + (ru * np.cos(theta))[:, None] * N[None, :]
        + (rv * np.sin(theta))[:, None] * B[None, :]
    )
    return pts

def visualize_all_keyframes_rho_spikes(
    curve_core,
    surface_points,
    slab_half_width=0.003,
    ellipse_samples=64,
    max_spikes_per_key=120,
    spike_samples=12,
    name="rho_all"
):
    key_ts = np.asarray(curve_core.key_ts, dtype=np.float64)
    pts = np.asarray(surface_points, dtype=np.float64)

    all_pts = []
    all_cols = []

    for key_index, s0 in enumerate(key_ts):
        s_arr = np.array([s0], dtype=np.float64)

        intpl = curve_core.interpolate(s_arr)
        C = intpl["points"][0]
        frame = intpl["frame"][0]      # world -> local
        yz_radius = intpl["radius"][0]

        if not np.isfinite(C).all() or not np.isfinite(frame).all() or not np.isfinite(yz_radius).all():
            continue

        axes = frame.T
        T = axes[:, 0]
        N = axes[:, 1]
        B = axes[:, 2]

        ru = yz_radius[0]
        rv = yz_radius[1]

        # project all points into THIS keyframe's local frame
        local = np.einsum('ij,nj->ni', frame, (pts - C[None, :]))
        w = local[:, 0]

        # thin slab for actual cross-section
        mask = np.abs(w) <= slab_half_width
        pts_slice = pts[mask]
        local_slice = local[mask]

        if len(pts_slice) == 0:
            continue

        # optional subsample of slice points so spikes stay visible
        if len(pts_slice) > max_spikes_per_key:
            idx = np.random.choice(len(pts_slice), max_spikes_per_key, replace=False)
            pts_slice = pts_slice[idx]
            local_slice = local_slice[idx]

        # ellipse points in world
        theta = np.linspace(0.0, 2.0 * np.pi, ellipse_samples, endpoint=False)
        epts = (
            C[None, :]
            + (ru * np.cos(theta))[:, None] * N[None, :]
            + (rv * np.sin(theta))[:, None] * B[None, :]
        )
        epts = epts[np.isfinite(epts).all(axis=1)]

        # 1. slice points
        pts_slice_valid = pts_slice[np.isfinite(pts_slice).all(axis=1)]
        if len(pts_slice_valid) > 0:
            all_pts.append(pts_slice_valid)
            all_cols.append(
                np.tile(np.array([[180, 180, 180, 120]], dtype=np.uint8), (len(pts_slice_valid), 1))
            )

        # 2. ellipse points
        if len(epts) > 0:
            all_pts.append(epts)
            all_cols.append(
                np.tile(np.array([[255, 0, 0, 255]], dtype=np.uint8), (len(epts), 1))
            )

        # 3. center point
        all_pts.append(C[None, :])
        all_cols.append(np.array([[255, 0, 255, 255]], dtype=np.uint8))

        # 4. rho spikes as sampled points
        p0 = np.repeat(C[None, :], len(pts_slice_valid), axis=0)
        p1 = pts_slice_valid
        spike_pts = sample_segments_as_points(p0, p1, n=spike_samples)

        if len(spike_pts) > 0:
            all_pts.append(spike_pts)
            all_cols.append(
                np.tile(np.array([[0, 255, 0, 255]], dtype=np.uint8), (len(spike_pts), 1))
            )

    if len(all_pts) == 0:
        print("No valid keyframes / slabs found.")
        return

    pts_export = np.vstack(all_pts)
    cols_export = np.vstack(all_cols)

    pc = trimesh.points.PointCloud(pts_export, colors=cols_export)
    pc.export(name + "_rho_all.ply")
    print("saved", name + "_rho_all.ply")


def visualize_section_radius_uv(
    curve_core,
    surface_points,
    key_index,
    slab_half_width=0.002,
    n_angle_bins=72,
    quantile=0.98,
    name='radius_uv'
):
    key_ts = np.asarray(curve_core.key_ts, dtype=np.float64)
    s0 = key_ts[key_index:key_index+1]

    intpl = curve_core.interpolate(s0)
    C = intpl["points"][0]
    frame = intpl["frame"][0]      # world -> local
    yz_radius = intpl["radius"][0]

    ru, rv = yz_radius[0], yz_radius[1]

    pts = np.asarray(surface_points, dtype=np.float64)

    # project into this keyframe's local frame
    local = np.einsum('ij,nj->ni', frame, (pts - C[None, :]))
    w = local[:, 0]
    u = local[:, 1]
    v = local[:, 2]

    # thin slab
    mask = np.abs(w) <= slab_half_width
    u = u[mask]
    v = v[mask]

    if len(u) == 0:
        print("No points in slab.")
        return

    rho = np.sqrt(u**2 + v**2)
    theta = np.arctan2(v, u)

    # stored ellipse in uv plane
    t = np.linspace(0.0, 2.0*np.pi, 400)
    u_ell = ru * np.cos(t)
    v_ell = rv * np.sin(t)

    # ellipse radial function
    def r_ellipse(th):
        return 1.0 / np.sqrt((np.cos(th)**2)/(ru**2 + 1e-12) +
                             (np.sin(th)**2)/(rv**2 + 1e-12))

    # directional boundary estimate from slice points
    edges = np.linspace(-np.pi, np.pi, n_angle_bins + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])
    r_surf = np.full(n_angle_bins, np.nan)

    bin_ids = np.clip(np.digitize(theta, edges) - 1, 0, n_angle_bins - 1)
    for b in range(n_angle_bins):
        m = bin_ids == b
        if np.sum(m) < 5:
            continue
        r_surf[b] = np.quantile(rho[m], quantile)

    valid = np.isfinite(r_surf)
    u_surf = r_surf[valid] * np.cos(centers[valid])
    v_surf = r_surf[valid] * np.sin(centers[valid])

    # plot 1: uv cross-section
    plt.figure(figsize=(7, 7))
    plt.scatter(u, v, s=4, alpha=0.25, label="slice points")
    plt.plot(u_ell, v_ell, 'r-', linewidth=2, label="stored ellipse")
    plt.plot(u_surf, v_surf, 'b.-', linewidth=2, markersize=4, label="directional boundary")
    plt.scatter([0], [0], c='m', s=80, label="center")
    plt.gca().set_aspect('equal', adjustable='box')
    plt.xlabel("u")
    plt.ylabel("v")
    plt.title(f"Keyframe {key_index}: local cross-section")
    plt.legend()
    plt.grid(True)
    plt.savefig(name+"_local_crosssec.jpg") #show()

    # plot 2: radius vs angle
    plt.figure(figsize=(10, 4))
    th_plot = np.linspace(-np.pi, np.pi, 400)
    plt.plot(th_plot, r_ellipse(th_plot), 'r-', linewidth=2, label="ellipse radius")
    plt.plot(centers[valid], r_surf[valid], 'b.-', linewidth=2, markersize=4, label="surface directional radius")
    plt.xlabel("theta")
    plt.ylabel("radius")
    plt.title(f"Keyframe {key_index}: radius as function of angle")
    plt.legend()
    plt.grid(True)
    plt.savefig(name+"_radius_fn_angle.jpg")

    # plot 3: residual eta
    eta = (u**2)/(ru**2 + 1e-12) + (v**2)/(rv**2 + 1e-12)
    plt.figure(figsize=(10, 4))
    plt.hist(eta, bins=50)
    plt.xlabel("eta = u^2/ru^2 + v^2/rv^2")
    plt.ylabel("count")
    plt.title(f"Keyframe {key_index}: ellipse residual")
    plt.grid(True)
    #plt.show()
    plt.savefig(name+"_ellipse_residual.jpg")

    print("eta mean:", eta.mean())
    print("eta q95:", np.quantile(eta, 0.95))
    print("eta max:", eta.max())


def _circle_points_world(C, N, B, r, n=64):
    theta = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
    pts = (
        C[None, :]
        + (r * np.cos(theta))[:, None] * N[None, :]
        + (r * np.sin(theta))[:, None] * B[None, :]
    )
    return pts


def _wrap_points_world(C, N, B, r_theta, theta, n=None):
    theta = np.asarray(theta, dtype=np.float64)
    r_theta = np.asarray(r_theta, dtype=np.float64)
    pts = (
        C[None, :]
        + (r_theta * np.cos(theta))[:, None] * N[None, :]
        + (r_theta * np.sin(theta))[:, None] * B[None, :]
    )
    return pts


def _safe_radius_fields(curve_core, key_ts):
    """
    Returns:
        train_radius:    (K,2)
        cylinder_radius: (K,2)
        wrap_radius:     (K,Ntheta) or None
        wrap_theta_bins: (Ntheta,) or None
    """
    key_ts = np.asarray(key_ts, dtype=np.float64)

    # training radius: read direct field, not default interpolate alias
    key_train_radius = getattr(curve_core, "key_train_radius", None)
    if key_train_radius is None:
        intpl = curve_core.interpolate(key_ts, radius_type='train')
        train_radius = intpl["radius"]
    else:
        train_radius = np.stack([
            np.interp(key_ts, curve_core.key_ts, key_train_radius[:, 0]),
            np.interp(key_ts, curve_core.key_ts, key_train_radius[:, 1]),
        ], axis=1)

    # cylinder radius
    key_cylinder_radius = getattr(curve_core, "key_cylinder_radius", None)
    if key_cylinder_radius is None:
        cylinder_radius = train_radius
    else:
        cylinder_radius = np.stack([
            np.interp(key_ts, curve_core.key_ts, key_cylinder_radius[:, 0]),
            np.interp(key_ts, curve_core.key_ts, key_cylinder_radius[:, 1]),
        ], axis=1)

    # wrap radius
    wrap_radius = getattr(curve_core, "key_wrap_radius", None)
    wrap_theta_bins = getattr(curve_core, "wrap_theta_bins", None)
    wrap_s_bins = getattr(curve_core, "wrap_s_bins", None)

    wrap_radius_interp = None
    if wrap_radius is not None and wrap_theta_bins is not None and wrap_s_bins is not None:
        wrap_radius_interp = np.stack([
            np.interp(key_ts, wrap_s_bins, wrap_radius[:, j])
            for j in range(wrap_radius.shape[1])
        ], axis=1)

    return train_radius, cylinder_radius, wrap_radius_interp, wrap_theta_bins

def _section_color(i, alpha=180):
    rng = np.random.default_rng(12345 + int(i))
    rgb = rng.integers(40, 256, size=3)
    return (int(rgb[0]), int(rgb[1]), int(rgb[2]), alpha)


def visualize_keyframes_with_profiles_trimesh(
    curve_core,
    surface_points,
    sample_keypoint_map,
    key_ts=None,
    slab_half_width=0.003,
    show_all_surface=False,
    key_sphere_radius=0.004,
    frame_scale=0.015,
    ellipse_samples=40,
    ellipse_stride=10,
    show_train=True,
    show_cylinder=True,
    show_wrap=True,
    show_section_points=True,
    export_glb=True,
    export_ply=False,
    name="keyframe_profiles"
):
    """
    3D visualization of:
      - training ellipse
      - cylinder radius contour
      - wrap contour
    """
    if key_ts is None:
        key_ts = curve_core.key_ts

    key_ts = np.asarray(key_ts, dtype=np.float64)
    surface_points = np.asarray(surface_points, dtype=np.float64)

    intpl = curve_core.interpolate(key_ts)
    C = intpl["points"]
    frame = intpl["frame"]

    train_radius, cylinder_radius, wrap_radius, wrap_theta_bins = _safe_radius_fields(curve_core, key_ts)

    #axes = np.transpose(frame, (0, 2, 1))
    T = frame[:,0,:] #axes[:, :, 0]
    N = frame[:,1,:] #axes[:, :, 1]
    B = frame[:,2,:] #axes[:, :, 2]

    scene = trimesh.Scene()

    # full surface
    if show_all_surface and len(surface_points) > 0:
        scene.add_geometry(_make_point_cloud(surface_points, color=(140, 140, 140, 40)))

    # local neighborhoods: use the actual keyframe-local w-slab, not projected s-neighborhood
    pts_valid = surface_points

#    for i in range(0, len(key_ts), max(1, ellipse_stride)):
#        Ci = C[i]
#        frame_i = frame[i]   # world -> local, row-wise [T; N; B]
#
#        local_i = np.einsum('ij,nj->ni', frame_i, (pts_valid - Ci[None, :]))
#        w_i = local_i[:, 0]
#
#        m = np.abs(w_i) <= slab_half_width
#        pts_local = pts_valid[m]
#        if len(pts_local) > 0:
#            scene.add_geometry(_make_point_cloud(pts_local, color=(255, 200, 0, 70)))

    # local neighborhoods
#    s_proj = curve_core.curve_projection(surface_points)
#    valid = (s_proj >= 0.0) & (s_proj <= 1.0)
#    pts_valid = surface_points[valid]
#    s_valid = s_proj[valid]
#
#    for i, s0 in enumerate(key_ts[::ellipse_stride]):
#        m = np.abs(s_valid - s0) <= neighborhood_half_width
#        pts_local = pts_valid[m]
#        if len(pts_local) > 0:
#            scene.add_geometry(_make_point_cloud(pts_local, color=(255, 200, 0, 70)))

    # key centers
    kp = _make_spheres(C, radius=key_sphere_radius, color=(255, 0, 255, 255))
    if kp is not None:
        scene.add_geometry(kp)

    # frame axes
    segT = _make_segments(C, C + frame_scale * T, color=(255, 120, 120, 255))
    segN = _make_segments(C, C + frame_scale * N, color=(120, 255, 120, 255))
    segB = _make_segments(C, C + frame_scale * B, color=(120, 120, 255, 255))
    if segT is not None: scene.add_geometry(segT)
    if segN is not None: scene.add_geometry(segN)
    if segB is not None: scene.add_geometry(segB)

    # centerline
    centerline = _make_polyline(C, closed=False, color=(255, 255, 255, 255))
    if centerline is not None:
        scene.add_geometry(centerline)

    #theta_dense = np.linspace(0.0, 2.0 * np.pi, ellipse_samples, endpoint=False)
    if show_all_surface:
        if len(pts_valid) > 0:
            all_pts = _make_spheres(pts_valid, radius=0.0015, color=(250, 127, 100, 255))
            if all_pts is not None:
                scene.add_geometry(all_pts)

    for i in range(0, len(key_ts), max(1, ellipse_stride)):
        Ci = C[i]
        Ni = N[i]
        Bi = B[i]

        section_color = _section_color(i, alpha=180)

        if show_section_points:
            if sample_keypoint_map is not None:
                sec_half = 0.5 / len(key_ts)
                m_sec = np.abs(sample_keypoint_map - key_ts[i]) <= sec_half
            else:
                # fallback only if sample_keypoint_map was not passed
                s_proj = curve_core.curve_projection(pts_valid)
                sec_half = 0.5 / len(key_ts)
                m_sec = np.abs(s_proj - key_ts[i]) <= sec_half

            pts_section = pts_valid[m_sec]

            if len(pts_section) > 0:
                sec_pts = _make_spheres(pts_section, radius=0.0015, color=section_color)
                if sec_pts is not None:
                    scene.add_geometry(sec_pts)

#        if show_section_points:
#            frame_i = frame[i]   # world -> local, row-wise [T; N; B]
#
#            # all points assigned to this section (by projected s / nearest section id)
#            if hasattr(curve_core, "curve_projection"):
#                s_proj = curve_core.curve_projection(pts_valid)
#                sec_half = 0.5 / len(key_ts)
#                m_sec = np.abs(s_proj - key_ts[i]) <= sec_half
#            else:
#                local_i = np.einsum('ij,nj->ni', frame_i, (pts_valid - Ci[None, :]))
#                w_i = local_i[:, 0]
#                m_sec = np.abs(w_i) <= slab_half_width
#
#            pts_section = pts_valid[m_sec]
#
#            if len(pts_section) > 0:
#                sec_pts = _make_spheres(pts_section, radius=0.0015, color=section_color)
#                if sec_pts is not None:
#                    scene.add_geometry(sec_pts)

#        if show_section_points:
#            frame_i = frame[i]   # world -> local, row-wise [T; N; B]
#            local_i = np.einsum('ij,nj->ni', frame_i, (pts_valid - Ci[None, :]))
#            w_i = local_i[:, 0]
#
#            m = np.abs(w_i) <= slab_half_width
#            pts_local = pts_valid[m]
#            if len(pts_local) > 0:
#                sec_pts = _make_spheres(pts_local, radius=0.0015, color=section_color)
#                if sec_pts is not None:
#                    scene.add_geometry(sec_pts)

        if show_train:
            ru_t, rv_t = train_radius[i]
            if np.isfinite(ru_t) and np.isfinite(rv_t) and ru_t > 1e-8 and rv_t > 1e-8:
                pts_train = _ellipse_points_world(Ci, Ni, Bi, ru_t, rv_t, n=ellipse_samples)
                #poly_train = _make_polyline(pts_train, closed=True, color=(0, 255, 0, 255))
                poly_train = _make_polyline(pts_train, closed=True, color=section_color)
                if poly_train is not None:
                    scene.add_geometry(poly_train)

        if show_cylinder:
            ru_c, rv_c = cylinder_radius[i]
            if np.isfinite(ru_c) and np.isfinite(rv_c) and ru_c > 1e-8 and rv_c > 1e-8:
                pts_cyl = _ellipse_points_world(Ci, Ni, Bi, ru_c, rv_c, n=ellipse_samples)
                #poly_cyl = _make_polyline(pts_cyl, closed=True, color=(0, 180, 255, 255))
                poly_cyl = _make_polyline(pts_cyl, closed=True, color=section_color)
                if poly_cyl is not None:
                    scene.add_geometry(poly_cyl)

        if show_wrap and wrap_radius is not None and wrap_theta_bins is not None:
            r = wrap_radius[i]
            if np.all(np.isfinite(r)) and np.max(r) > 1e-8:
                pts_wrap = _wrap_points_world(Ci, Ni, Bi, r, wrap_theta_bins)
                #poly_wrap = _make_polyline(pts_wrap, closed=True, color=(255, 60, 60, 255))
                poly_wrap = _make_polyline(pts_wrap, closed=True, color=section_color)
                if poly_wrap is not None:
                    scene.add_geometry(poly_wrap)


    print("key_ts:", key_ts.shape, key_ts[:3], key_ts[-3:])
    print("train_radius:", None if train_radius is None else train_radius.shape,
          None if train_radius is None else (np.nanmin(train_radius), np.nanmax(train_radius)))
    print("cylinder_radius:", None if cylinder_radius is None else cylinder_radius.shape,
          None if cylinder_radius is None else (np.nanmin(cylinder_radius), np.nanmax(cylinder_radius)))
    print("wrap_radius:", None if wrap_radius is None else wrap_radius.shape,
          None if wrap_radius is None else (np.nanmin(wrap_radius), np.nanmax(wrap_radius)))
    print("wrap_theta_bins:", None if wrap_theta_bins is None else wrap_theta_bins.shape)

    if export_glb:
        scene.export(f"{name}.glb")
    if export_ply:
        try:
            print("name = ", name)
            # flatten scene to mesh where possible
            geom = []
            for g in scene.geometry.values():
                if isinstance(g, trimesh.Trimesh):
                    geom.append(g)

            if len(geom) > 0:
                merged = trimesh.util.concatenate(geom)
                merged.export(f"{name}.ply")
                print(f"saved: {name}.ply")
            else:
                print("No mesh geometry found for PLY export")
        except Exception as e:
            print("PLY export skipped:", e)

    return scene


