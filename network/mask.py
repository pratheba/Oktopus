import torch
import numpy as np

def smoothstep01(x):
    return x * x * (3 - 2*x)

def make_detail_region(t, t0, t1, eps=0.03):
    if t.ndim > 0 and t.shape[-1] == 1:
        t = t[..., 0]

    up = torch.clamp((t - (t0 - eps)) / (2*eps), 0.0, 1.0)
    dn = torch.clamp(((t1 + eps) - t) / (2*eps), 0.0, 1.0)
    return smoothstep01(up) * smoothstep01(dn)


def apply_random_tiling(
                model_input,
                p_tile_max=0.05,
                p_detail=1.00,
                k_range=(1.0, 3.0),
                seg_len_range=(0.01, 0.99),
                eps_region=0.03,
                eps_seam=0.06):

    device = model_input['coords'].device
    B, Ns = model_input['coords'].shape
    ts = model_input['coords']

    samples_detail = model_input['samples'].clone()

    detail_on = (torch.rand((B,), device=device) < p_detail).float()
    do_tile = ((torch.rand((B,), device=device) < p_tile).float() * detail_on)

    if do_tile.sum() == 0:
        model_input['samples_detail'] = samples_detail
        model_input['detail_on'] = detail_on.view(B, 1, 1)
        model_input['coords_detail'] = model_input['coords']
        return model_input


    seg_len = torch.empty((B,), device=device).uniform_(*seg_len_range)
    t0 = torch.empty((B,), device=device).uniform_(0.05, 0.95)
    t1 = torch.clamp(t0+ seg_len, max=0.95)
    t0 = torch.clamp(t1 - seg_len, min=0.05)


    k = torch.empty((B,), device=device).uniform_(*k_range)

    vx_base = 2.0 * ts - 1.0
    w_local = model_input['samples'][...,0] - vx_base

    w_region = make_detail_region(ts, t0[:,None], t1[:,None], eps=eps_region)
    tau = (ts - t0[:,None]) / (t1[:,None] - t0[:,None] + 1e-12)
    tau = torch.clamp(tau, 0.0, 1.0)

    ts_detail = (k[:,None] * tau)
    phi = torch.frac(k[:,None] * tau)
    dist = torch.minimum(phi, 1-phi)
    x = torch.clamp(dist / eps_seam, 0.0, 1.0)
    w_seam = smoothstep01(x)

    w_wrap = w_region * w_seam * do_tile[:,None]

    t_wrapped = t0[:,None] + (t1[:,None] - t0[:,None]) * phi    
    t_used = (1.0 - w_wrap) * ts + w_wrap * t_wrapped
    vx_used = 2.0 * t_used - 1.0

    x_new = w_local + vx_used
    samples_detail[...,0] = x_new

    model_input['samples_detail'] = samples_detail
    model_input['detail_on'] = detail_on.view(B, 1 , 1)
    model_input['tile_mask'] = do_tile.view(B, 1)
    model_input['coords_detail'] = ts_detail

    return model_input

@torch.no_grad()
def apply_random_tiling_twohead(
                model_input,
                epoch: int,
                p_tile_max=0.05,
                warmup_epochs = 1000,
                ramp_epochs = 1000,
                k_range=(0.5, 3.0),
                seg_len_range=(0.01, 0.99),
                eps_region=0.03,
                eps_seam=0.06):

    device = model_input['coords'].device
    B, N = model_input['coords'].shape
    ts = model_input['coords']

    samples_tile = model_input['samples'].clone()
    coords_tile = model_input['coords'].clone()
    w_tile = torch.zeros((B, N, 1), device=device, dtype=ts.dtype)
    tile_on = torch.zeros((B, 1, 1), device=device, dtype=ts.dtype)

    if epoch < warmup_epochs:
        model_input['samples_tile'] = samples_tile
        model_input['w_tile'] = w_tile
        model_input['coords_tile'] = coords_tile
        model_input['tile_on'] = tile_on
        model_input['tile_params'] = {'t0': None, 't1': None, 'k': None}
        return model_input

    if ramp_epochs > 0:
        r = min(1.0, (epoch - warmup_epochs) / float(ramp_epochs))
    else:
        r = 1.0

    p_tile = p_tile_max * r
    do_tile = ((torch.rand((B,), device=device) < p_tile).float())

    if do_tile.sum() == 0:
        model_input['samples_tile'] = samples_tile
        model_input['w_tile'] = w_tile
        model_input['coords_tile'] = coords_tile
        model_input['tile_on'] = tile_on
        model_input['tile_params'] = {'t0': None, 't1': None, 'k': None}
        return model_input

    #### Select the batch index where tiling is true
    idx = torch.where(do_tile)[0]

    seg_len = torch.empty((idx.numel(),), device=device).uniform_(*seg_len_range)
    t0 = torch.empty((idx.numel(),), device=device).uniform_(0.05, 0.95)
    #seg_len = torch.empty((B,), device=device).uniform_(*seg_len_range)
    #t0 = torch.empty((B,), device=device).uniform_(0.05, 0.95)
    t1 = torch.clamp(t0+ seg_len, max=0.95)
    t0 = torch.clamp(t1 - seg_len, min=0.05)


    k = torch.empty((idx.numel(),), device=device).uniform_(*k_range)
    #k = torch.empty((B,), device=device).uniform_(*k_range)

    vx_base = 2.0 * ts[idx] - 1.0
    w_local = model_input['samples'][idx,:,0] - vx_base

    w_region = make_detail_region(ts[idx], t0[:, None], t1[:, None], eps=eps_region)
    tau = (ts[idx] - t0[:, None]) / (t1[:, None] - t0[:, None] + 1e-12)
    tau = torch.clamp(tau, 0.0, 1.0)

    ts_tile = (k[:, None] * tau)
    phi = torch.frac(ts_tile)
    dist = torch.minimum(phi, 1-phi)
    x = torch.clamp(dist / eps_seam, 0.0, 1.0)
    w_seam = smoothstep01(x)

    w_wrap = w_region * w_seam 

    t_wrapped = t0[:,None] + (t1[:, None] - t0[:, None]) * phi    
    ts_used = (1.0 - w_wrap) * ts[idx] + w_wrap * t_wrapped
    vx_used = 2.0 * ts_used - 1.0

    x_new = w_local + vx_used
    samples_tile[idx,:,0] = x_new

    w_tile[idx, :, 0] = w_wrap
    coords_tile[idx] = ts_used
    tile_on[idx, 0, 0] = 1.0

    model_input['samples_tile'] = samples_tile
    model_input['coords_tile'] = coords_tile
    model_input['w_tile'] = w_tile
    model_input['tile_on'] = tile_on
    model_input['tile_params'] = {'t0': t0, 't1': t1, 'k': k}

    return model_input


def active_freqs_from_epoch(epoch, total_freq):
    if epoch < 500:  return 3   # 4..6
    if epoch < 1500: return 5   # 4..8
    if epoch < 3000: return 9   # 4..12
    if total_freq == 11:
        return 11
    if epoch < 3500: return 11   # 4..14
    if epoch < 4000: return 13   # 4..16
    if epoch < 4500: return 15   # 4..18
    if epoch < 5000: return 18   # 4..21
    if total_freq == 20:
        return 20
    if epoch < 6000: return 20   # 4..23
    if epoch < 6500: return 24   # 4..23
    if epoch < 7000: return 26   # 4..23
    if total_freq == 28:
        return 28

def pe_band_mask(num_freqs, d_in, include_input, active_freqs):
    """
    num_freqs: total F in PE module (e.g. 11 for 4..14)
    d_in: 3
    active_freqs: how many of the lowest freqs to keep ON (e.g. 3 means keep first 3 freqs)
    returns mask of shape (D_out,)
    """
    # sincos part length = 2 * F * d_in
    sincos_len = 2 * num_freqs * d_in
    if include_input:
        D = d_in + sincos_len
        mask = torch.zeros(D)
        mask[:d_in] = 1.0
        # enable first active_freqs frequencies
        keep = 2 * active_freqs * d_in
        mask[d_in:d_in+keep] = 1.0
    else:
        D = sincos_len
        mask = torch.zeros(D)
        keep = 2 * active_freqs * d_in
        mask[:keep] = 1.0
    return mask
