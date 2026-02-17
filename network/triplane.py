import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import ast

def wrap(x):
    return x - torch.floor(x)


def curvetheta_to_grid(ts, theta, ts_min, ts_max):
    theta_wrap = wrap((theta + math.pi) / (2.0 * math.pi))
    gx = 2.0 * theta_wrap - 1.0

    ts = (ts - ts_min)/ (ts_max - ts_min +1e-12)
    gy = torch.clamp(ts, -1.0, 1.0)
    grid = torch.stack([gx, gy], dim=-1)
    return grid.view(ts.shape[0], ts.shape[1], 1, 2)

def curverho_to_grid(ts, rho, ts_min, ts_max):
    gx = 2.0 * torch.clamp(rho, 0.0, 1.0) - 1.0

    ts = (ts - ts_min)/ (ts_max - ts_min +1e-12)
    gy = 2.0 * torch.clamp(ts, 0.0, 1.0) - 1.0
    grid = torch.stack([gx, gy], dim=-1)
    return grid.view(ts.shape[0], ts.shape[1], 1, 2)

def grid_sample(feat, grid, align_corners=True):
    out = F.grid_sample(
        feat, grid, mode="bilinear",
        padding_mode="border", align_corners=align_corners
    )
    return out.squeeze(-1).transpose(1,2)

class CurveThetaMultiResGrid(nn.Module):
    def __init__(self, base_hw=(64, 256), levels=4, dim=32):
        super().__init__()
        self.levels = levels
        self.dim = dim 
        H0, W0 = map(int, ast.literal_eval(base_hw))
        self.grids = nn.ParameterList()

        for l in range(self.levels):
            Hl = H0 * (2 ** l)
            Wl = W0 * (2 ** l)
            g = nn.Parameter(torch.randn(1, dim, Hl, Wl) * 1e-3)
            self.grids.append(g)

    def forwardbase(self, ts, theta, ts_min=0.0, ts_max=1.0):
        grid = curvetheta_to_grid(ts, theta, ts_min, ts_max)
        grid_feat = []
        for g in self.grids:
            feat = g.expand(ts.shape[0], -1, -1, -1)
            grid_feat.append(grid_sample(feat, grid))
        return torch.cat(grid_feat, dim=-1)

    def forward_levels(self, ts, theta, ts_min=0.0, ts_max=1.0, start=0, end=3):
        grid = curvetheta_to_grid(ts, theta, ts_min, ts_max)
        grid_feat = []
        for idx, g in enumerate(self.grids):
            if idx >= start and idx <= end:
                #print("{} {} {}", idx, start, end)
                feat = g.expand(ts.shape[0], -1, -1, -1)
                grid_feat.append(grid_sample(feat, grid))
        return torch.cat(grid_feat, dim=-1)

    def forward(self, ts, theta, ts_min=0.0, ts_max=1.0):
        return self.forwardbase(ts, theta, ts_min, ts_max)

    @torch.no_grad()
    def inference(self, ts, theta, ts_min=0.0, ts_max=1.0):
        return self.forwardbase(ts, theta, ts_min, ts_max)

    @torch.no_grad()
    def inference_stretch(self, ts, theta, ts_min=0.0, ts_max=1.0, start=0, end=3):
        return self.forward_levels(ts, theta, ts_min, ts_max, start, end)

class CurveRhoMultiResGrid(nn.Module):
    def __init__(self, base_hw=(64, 256), levels=4, dim=32):
        super().__init__()
        self.levels = levels
        self.dim = dim 
        H0, W0 = map(int, ast.literal_eval(base_hw))
        self.grids = nn.ParameterList()

        for l in range(self.levels):
            Hl = H0 * (2 ** l)
            Wl = W0 * (2 ** l)
            g = nn.Parameter(torch.randn(1, dim, Hl, Wl) * 1e-3)
            self.grids.append(g)

    def forwardbase(self, ts, rho, ts_min=0.0, ts_max=1.0):
        grid = curverho_to_grid(ts, rho, ts_min, ts_max)
        grid_feat = []
        for g in self.grids:
            feat = g.expand(ts.shape[0], -1, -1, -1)
            grid_feat.append(grid_sample(feat, grid))
        return torch.cat(grid_feat, dim=-1)

    def forward(self, ts, rho, ts_min=0.0, ts_max=1.0):
        return self.forwardbase(ts, rho, ts_min, ts_max)

    @torch.no_grad()
    def inference(self, ts, rho, ts_min=0.0, ts_max=1.0):
        return self.forwardbase(ts, rho, ts_min, ts_max)
