import numpy as np
from curve_utils.curve_utils import get_bins, fill_invalid_theta, fill_invalid_bins
from scipy.ndimage import gaussian_filter1d
from ._interpolate import interpolate_wrap_radius1


def update_wrap_occupancy_from_coords(self, coord_points, u, v, n_curve_bins=24, quantile=0.75, gaussian_smooth=2.0, min_count=50):
    bin_data = get_bins(coord_points, n_curve_bins)
    rho = np.sqrt(u**2 + v**2)
    theta = np.arctan2(v, u)

    radius_theta_wrap = interpolate_wrap_radius1(self, coord_points, theta, self.key_wrap_radius, self.wrap_theta_bins, self.wrap_s_bins)
    q = rho / radius_theta_wrap

    occupancy = np.full(n_curve_bins, np.nan, dtype=np.float64)
    
    for b in range(n_curve_bins):
        m = bin_data.ids == b
        if np.sum(m) < min_count:
            continue

        occupancy[b] = np.quantile(q[m], quantile)

    valid = np.isfinite(occupancy)
    occupancy = fill_invalid_bins(occupancy, valid)

    if gaussian_smooth > 0:
        occupancy = gaussian_filter1d(occupancy, sigma=gaussian_smooth)
    self.key_occupancy_rho = occupancy
    return occupancy

     
def update_wrap_profile_from_coords(self, coord_points, w, u, v, n_curve_bins=24, n_theta_bins=36, quantile=0.97, gaussian_smooth_curve=2.0, gaussian_smooth_theta=0.75, min_count = 25, radius_type = 'wrap'):
    rho = np.sqrt(u*u + v*v)
    theta = np.arctan2(v, u)
    #bin_edges_curve = np.linspace(0.0, 1.0, n_curve_bins+1)
    #bin_center_curve = 0.5 * (bin_edges_curve[:-1] + bin_edges_curve[1:])
    #bin_ids_curve = np.clip(np.digitize(coord_points, bin_edges_curve) -1, 0, n_curve_bins-1)
    bin_curve = get_bins(coord_points, n_curve_bins)
    bin_theta = get_bins(theta, n_theta_bins, True)

#    bin_edges_theta = np.linspace(-np.pi, np.pi, n_theta_bins + 1)
#    bin_center_theta = 0.5 * (bin_edges_theta[:-1] + bin_edges_theta[1:])
#    bin_ids_theta = np.clip(np.digitize(theta, bin_edges_theta) -1, 0, n_theta_bins-1)

    radius_theta_wrap = np.full((n_curve_bins, n_theta_bins), np.nan, dtype=np.float64)
    counts = np.zeros((n_curve_bins, n_theta_bins), dtype=np.int32)

    slab_half_width = 2.0 / n_curve_bins
    slab_mask = np.abs(w) <= slab_half_width

    for s in range(n_curve_bins):
        s_mask = (bin_curve.ids == s) & slab_mask 
        if np.sum(s_mask) == 0:
            continue
        for t in range(n_theta_bins):
            m = s_mask & (bin_theta.ids == t) 
            counts[s,t] = np.sum(m)
            if counts[s,t] < min_count:
                continue
            radius_theta_wrap[s, t] = np.quantile(rho[m], quantile)

    for s in range(n_curve_bins):
        thetas = radius_theta_wrap[s]
        valid = np.isfinite(thetas)
        if np.any(valid):
            radius_theta_wrap[s] = fill_invalid_theta(thetas, valid)
   
    for t in range(n_theta_bins):
        radius = radius_theta_wrap[:,t]
        valid = np.isfinite(radius)
        if np.any(valid):
            radius_theta_wrap[:,t] = fill_invalid_bins(radius, valid) 

    # gaussian smooth of Theta
    if gaussian_smooth_theta > 0:
        radius_theta_wrap = gaussian_filter1d(np.concatenate([radius_theta_wrap, radius_theta_wrap, radius_theta_wrap], axis=1), sigma=gaussian_smooth_theta, axis=1)[:, n_theta_bins:2*n_theta_bins]
    if gaussian_smooth_curve > 0:
        radius_theta_wrap = gaussian_filter1d(radius_theta_wrap, sigma=gaussian_smooth_curve, axis=0)

    self.wrap_s_bins = bin_curve.center 
    self.wrap_theta_bins = bin_theta.center
    self.key_wrap_radius = radius_theta_wrap
    self.wrap_radius_max = np.max(radius_theta_wrap, axis=1)

    return {
        "curve_bins": bin_curve.center,
        "theta_bins": bin_theta.center,
        "key_wrap_radius": self.key_wrap_radius,
        "wrap_radius_max": self.wrap_radius_max,
        "counts": counts,
    }
