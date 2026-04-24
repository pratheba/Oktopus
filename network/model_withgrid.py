import math
import torch
import torch.nn as nn

from mlp import *
from pos_encoding_fromokto import PosEncoding
from periodic_encoding import PeriodicEncoding
from triplane import CurveThetaMultiResGrid, CurveRhoMultiResGrid
from mask import *


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------

def _grid_outdim(levels, dim, reduce, rho=False):
    if reduce:
        outdim = 0
        for i in range(levels):
            outdim += int(dim / (2 ** i))
    else:
        outdim = levels * dim
    if rho:
        outdim *= 2
    return outdim


def _parse_hw(hw):
    if isinstance(hw, (list, tuple)):
        return tuple(int(x) for x in hw)
    if isinstance(hw, str):
        s = hw.strip().strip('()').strip('[]')
        return tuple(int(x.strip()) for x in s.split(','))
    raise ValueError(f"Unexpected grid_hw value: {hw!r}")


def _run_film_chain(film_list, x, curve_feats, type_sample):
    x = film_list[0](x, curve_feats) + type_sample
    for block in film_list[1:]:
        res = x
        x = block(x, curve_feats)
        x = x + res
    return x


def _ramp(curr_epoch, phase_start, warmup):
    denom = max(1, int(warmup))
    return max(0.0, min(1.0, (curr_epoch - phase_start) / denom))


# ---------------------------------------------------------------------
# Shared sub-modules
# ---------------------------------------------------------------------

class FiLMEncoder(nn.Module):
    def __init__(self, dim, out_dim, cond_dim):
        super().__init__()
        self.fc = nn.Linear(dim, out_dim)
        self.film = FiLM(cond_dim, out_dim)
        self.act = nn.SiLU()

    def forward(self, x, cond):
        x = self.fc(x)
        x = self.film(x, cond)
        return self.act(x)

    @torch.no_grad()
    def inference(self, x, cond):
        return self.forward(x, cond)


class FiLM(nn.Module):
    def __init__(self, cond_dim, out_dim):
        super().__init__()
        self.gamma = nn.Linear(cond_dim, out_dim)
        self.beta = nn.Linear(cond_dim, out_dim)
        self.initialize()

    def initialize(self):
        self.gamma.weight.data.zero_()
        self.gamma.bias.data.fill_(1.0)
        self.beta.weight.data.zero_()
        self.beta.bias.data.fill_(0.0)

    def forward(self, sample_feat, curve_feat):
        gamma = self.gamma(curve_feat)
        beta = self.beta(curve_feat)
        return gamma * sample_feat + beta

    @torch.no_grad()
    def inference(self, sample_feat, curve_feat):
        return self.forward(sample_feat, curve_feat)


class FeatSampleEncoder(nn.Module):
    def __init__(self, dim, out_dim):
        super().__init__()
        self.fc = nn.Linear(dim, out_dim)
        self.act = nn.SiLU()

    def forward(self, x):
        return self.act(self.fc(x))

    @torch.no_grad()
    def inference(self, x):
        return self.forward(x)


class FeatCurveEncoder(nn.Module):
    def __init__(self, arg, tile=False):
        super().__init__()
        if tile:
            t_posenc = (arg['num_pos_encoding'] * 2 + 1) + (arg['num_pos_encoding'] * 2)
        else:
            t_posenc = arg['num_pos_encoding'] * 2 + 1
        radii_y_posenc = arg['num_y_pos_encoding'] * 2 + 1
        radii_z_posenc = arg['num_z_pos_encoding'] * 2 + 1
        diff = arg['dim_code'] + t_posenc + radii_y_posenc + radii_z_posenc
        enc_cfg = dict(arg['encoder_curve'])
        enc_size = list(enc_cfg['size'])
        if enc_size[0] != diff:
            enc_size.insert(0, diff)
        enc_cfg['size'] = enc_size
        self.mlp_t = MLP(**enc_cfg)

    def forward(self, code_coords):
        return self.mlp_t.forward_simple(code_coords)

    @torch.no_grad()
    def inference(self, code_coords):
        return self.forward(code_coords)


# ---------------------------------------------------------------------
# Base1 branch
# ---------------------------------------------------------------------

class NGCNetGridBase1(nn.Module):
    def __init__(self, arg, dim_cond, rho_grid):
        super().__init__()
        self.rho_grid = rho_grid
        self.reduce_channel = arg['base1_reduce_channel']
        self.grid_levels = arg['base1_grid_levels']
        self.grid_dim = arg['base1_grid_dim']
        self.grid_hw = _parse_hw(arg['base1_grid_hw'])

        self.grid_curvetheta = CurveThetaMultiResGrid(
            base_hw=self.grid_hw,
            levels=self.grid_levels,
            dim=self.grid_dim,
            reduce=self.reduce_channel,
        )
        if self.rho_grid:
            self.grid_curverho = CurveRhoMultiResGrid(
                base_hw=self.grid_hw,
                levels=self.grid_levels,
                dim=self.grid_dim,
                reduce=self.reduce_channel,
            )
        else:
            self.grid_curverho = None

        self.grid_outdim = _grid_outdim(
            self.grid_levels, self.grid_dim, self.reduce_channel, rho=self.rho_grid,
        )
        dim_in = 1 + self.grid_outdim
        dim_out = arg['dim_base1_feat']
        self.num_hidden_layers = arg['num_base1_hidden_layers']
        self.filmenc = nn.ModuleList(
            [FiLMEncoder(dim_in, dim_out, dim_cond)] +
            [FiLMEncoder(dim_out, dim_out, dim_cond) for _ in range(self.num_hidden_layers)]
        )
        self.decoder = MLP(**arg['decoder_base1'])

    def sample_features(self, coords, angles, rho, rho_n, istrain):
        if istrain:
            ct = self.grid_curvetheta(coords, angles)
            if self.rho_grid:
                cr = self.grid_curverho(coords, rho_n)
        else:
            ct = self.grid_curvetheta.inference(coords, angles)
            if self.rho_grid:
                cr = self.grid_curverho.inference(coords, rho_n)
        if self.rho_grid:
            return torch.cat([rho.unsqueeze(-1), ct, cr], dim=-1)
        return torch.cat([rho.unsqueeze(-1), ct], dim=-1)

    def forward(self, mi, curve_feats, type_sample, istrain=True):
        feat = self.sample_features(mi['coords'], mi['angles'], mi['rho'], mi['rho_n'], istrain)
        x = _run_film_chain(self.filmenc, feat, curve_feats, type_sample)
        sdf = self.decoder.forward_simple(x).squeeze(-1)
        return {
            'sdf_base1': sdf,
            'base1_grid_ct': self.grid_curvetheta.grids,
            'base1_grid_cr': self.grid_curverho.grids if self.grid_curverho is not None else None,
        }

    def branch_modules(self):
        mods = [self.grid_curvetheta]
        if self.grid_curverho is not None:
            mods.append(self.grid_curverho)
        mods += list(self.filmenc) + [self.decoder]
        return mods

    def parameters_iter(self):
        for m in self.branch_modules():
            for p in m.parameters():
                yield p

    def named_parameters_iter(self, prefix='base1_model'):
        mods = {f'{prefix}.grid_curvetheta': self.grid_curvetheta}
        if self.grid_curverho is not None:
            mods[f'{prefix}.grid_curverho'] = self.grid_curverho
        for i, m in enumerate(self.filmenc):
            mods[f'{prefix}.filmenc.{i}'] = m
        mods[f'{prefix}.decoder'] = self.decoder
        for mprefix, m in mods.items():
            for name, p in m.named_parameters(recurse=True):
                yield f'{mprefix}.{name}', p


# ---------------------------------------------------------------------
# Base2 branch
# ---------------------------------------------------------------------

class NGCNetGridBase2(nn.Module):
    def __init__(self, arg, dim_out_angle, dim_cond):
        super().__init__()
        self.reduce_channel = arg['base2_reduce_channel']
        self.grid_levels = arg['base2_grid_levels']
        self.grid_dim = arg['base2_grid_dim']
        self.grid_hw = _parse_hw(arg['base2_grid_hw'])
        self.grid_curvetheta = CurveThetaMultiResGrid(
            base_hw=self.grid_hw,
            levels=self.grid_levels,
            dim=self.grid_dim,
            reduce=self.reduce_channel,
        )
        self.grid_outdim = _grid_outdim(
            self.grid_levels, self.grid_dim, self.reduce_channel, rho=False,
        )
        dim_in = 1 + self.grid_outdim + dim_out_angle + 1
        dim_out = arg['dim_base2_feat']
        self.num_hidden_layers = arg['num_base2_hidden_layers']
        self.filmenc = nn.ModuleList(
            [FiLMEncoder(dim_in, dim_out, dim_cond)] +
            [FiLMEncoder(dim_out, dim_out, dim_cond) for _ in range(self.num_hidden_layers)]
        )
        self.decoder = MLP(**arg['decoder_base2'])

    def sample_features(self, coords, angles, rho_n, istrain):
        if istrain:
            ct = self.grid_curvetheta(coords, angles)
        else:
            ct = self.grid_curvetheta.inference(coords, angles)
        return torch.cat([rho_n.unsqueeze(-1), ct], dim=-1)

    def forward(self, mi, curve_feats, type_sample, sample_angle_periodenc, sdf_base1, istrain=True):
        feat = self.sample_features(mi['coords'], mi['angles'], mi['rho_n'], istrain)
        y = torch.cat([feat, sample_angle_periodenc, sdf_base1.detach().unsqueeze(-1)], dim=-1)
        x = _run_film_chain(self.filmenc, y, curve_feats, type_sample)
        sdf = self.decoder.forward_simple(x).squeeze(-1)
        return {
            'sdf_base2': sdf,
            'base2_grid_ct': self.grid_curvetheta.grids,
        }

    def branch_modules(self):
        return [self.grid_curvetheta] + list(self.filmenc) + [self.decoder]

    def parameters_iter(self):
        for m in self.branch_modules():
            for p in m.parameters():
                yield p

    def named_parameters_iter(self, prefix='base2_model'):
        mods = {f'{prefix}.grid_curvetheta': self.grid_curvetheta}
        for i, m in enumerate(self.filmenc):
            mods[f'{prefix}.filmenc.{i}'] = m
        mods[f'{prefix}.decoder'] = self.decoder
        for mprefix, m in mods.items():
            for name, p in m.named_parameters(recurse=True):
                yield f'{mprefix}.{name}', p


# ---------------------------------------------------------------------
# Base model orchestrator
# ---------------------------------------------------------------------

class NGCNetGridBase(nn.Module):
    def __init__(self, arg, dim_out_angle, dim_cond, rho_grid):
        super().__init__()
        self.base1_model = NGCNetGridBase1(arg, dim_cond, rho_grid)
        self.base2_model = NGCNetGridBase2(arg, dim_out_angle, dim_cond)
        combine = arg['combine']
        self.sigma = float(combine['sigma'])
        self.alpha_warmup = int(combine['alpha_warmup'])
        self.gate_warmup = int(combine['gate_warmup'])
        self.base1_epoch = 0
        self.base2_epoch = 0
        self.base_epoch = 0

    def set_phase_schedule(self, base1_epoch, base2_epoch, base_epoch):
        self.base1_epoch = int(base1_epoch)
        self.base2_epoch = int(base2_epoch)
        self.base_epoch = int(base_epoch)

    def _phase(self, curr_epoch, istrain):
        if not istrain:
            return 'base_joint'
        if curr_epoch < self.base1_epoch:
            return 'base1_only'
        if curr_epoch < self.base2_epoch:
            return 'base2_only'
        return 'base_joint'

    def combine_base(self, sdf_base1, sdf_base2, curr_epoch, phase):
        gate = torch.exp(-(sdf_base1.detach() ** 2) / (2.0 * self.sigma ** 2))
        if phase == 'base1_only':
            return sdf_base1, 0.0, torch.zeros_like(sdf_base1)
        if phase == 'base2_only':
            alpha = _ramp(curr_epoch, self.base1_epoch, self.alpha_warmup)
            return sdf_base1.detach() + alpha * gate * sdf_base2, alpha, gate
        return sdf_base1 + gate * sdf_base2, 1.0, gate

    def forward(self, mi, curve_feats, type_sample, sample_angle_periodenc, curr_epoch=0, istrain=True, phase=None):
        base_phase = self._phase(curr_epoch, istrain) if phase is None else (
            phase if phase in ('base1_only', 'base2_only', 'base_joint') else 'base_joint'
        )
        out1 = self.base1_model(mi, curve_feats, type_sample, istrain=istrain)
        sdf_base1 = out1['sdf_base1']

        if base_phase == 'base1_only':
            sdf_base2 = torch.zeros_like(sdf_base1)
            sdf_base, alpha_base2, gate_base2 = self.combine_base(sdf_base1, sdf_base2, curr_epoch, base_phase)
            out2_grid = None
        else:
            out2 = self.base2_model(
                mi, curve_feats, type_sample, sample_angle_periodenc,
                sdf_base1=sdf_base1, istrain=istrain,
            )
            sdf_base2 = out2['sdf_base2']
            out2_grid = out2['base2_grid_ct']
            sdf_base, alpha_base2, gate_base2 = self.combine_base(sdf_base1, sdf_base2, curr_epoch, base_phase)

        return {
            'sdf_base1': sdf_base1,
            'sdf_base2': sdf_base2,
            'sdf_base': sdf_base,
            'alpha_base2': alpha_base2,
            'gate_base2': gate_base2,
            'base1_grid_ct': out1['base1_grid_ct'],
            'base1_grid_cr': out1['base1_grid_cr'],
            'base2_grid_ct': out2_grid,
        }

    def base1_parameters(self):
        yield from self.base1_model.parameters_iter()

    def base2_parameters(self):
        yield from self.base2_model.parameters_iter()

    def named_base1_parameters(self):
        yield from self.base1_model.named_parameters_iter(prefix='base_model.base1_model')

    def named_base2_parameters(self):
        yield from self.base2_model.named_parameters_iter(prefix='base_model.base2_model')

    def shared_and_base1_parameters(self, shared_modules=()):
        for m in shared_modules:
            for p in m.parameters():
                yield p
        yield from self.base1_parameters()

    def shared_and_base2_parameters(self, shared_modules=()):
        for m in shared_modules:
            for p in m.parameters():
                yield p
        yield from self.base2_parameters()

    def named_shared_and_base1_parameters(self, shared_named_items=()):
        for prefix, m in shared_named_items:
            for name, p in m.named_parameters(recurse=True):
                yield f'{prefix}.{name}', p
        yield from self.named_base1_parameters()

    def named_shared_and_base2_parameters(self, shared_named_items=()):
        for prefix, m in shared_named_items:
            for name, p in m.named_parameters(recurse=True):
                yield f'{prefix}.{name}', p
        yield from self.named_base2_parameters()


# ---------------------------------------------------------------------
# Detail model
# ---------------------------------------------------------------------

class NGCNetGridDetail(nn.Module):
    def __init__(self, arg, dim_out_angle, dim_cond):
        super().__init__()
        self.reduce_channel = arg['detail_reduce_channel']
        self.grid_levels = arg['detail_grid_levels']
        self.grid_dim = arg['detail_grid_dim']
        self.grid_hw = _parse_hw(arg['detail_grid_hw'])
        self.grid_curvetheta = CurveThetaMultiResGrid(
            base_hw=self.grid_hw,
            levels=self.grid_levels,
            dim=self.grid_dim,
            reduce=self.reduce_channel,
        )
        self.grid_outdim = _grid_outdim(
            self.grid_levels, self.grid_dim, self.reduce_channel, rho=False,
        )
        dim_in = 1 + self.grid_outdim + dim_out_angle + 1
        dim_out = arg['dim_detail_feat']
        self.num_hidden_layers = arg['num_detail_hidden_layers']
        self.filmenc = nn.ModuleList(
            [FiLMEncoder(dim_in, dim_out, dim_cond)] +
            [FiLMEncoder(dim_out, dim_out, dim_cond) for _ in range(self.num_hidden_layers)]
        )
        self.decoder = MLP(**arg['decoder_detail'])
        combine = arg['combine']
        self.sigma = float(combine['sigma'])
        self.alpha_warmup = int(combine['alpha_warmup'])
        self.gate_warmup = int(combine['gate_warmup'])

    def sample_features(self, coords, angles, rho_n, istrain):
        if istrain:
            ct = self.grid_curvetheta(coords, angles)
        else:
            ct = self.grid_curvetheta.inference(coords, angles)
        return torch.cat([rho_n.unsqueeze(-1), ct], dim=-1)

    def forward(self, mi, curve_feats, type_sample, sample_angle_periodenc, sdf_base, istrain=True):
        feat = self.sample_features(mi['coords'], mi['angles'], mi['rho_n'], istrain)
        y = torch.cat([feat, sample_angle_periodenc, sdf_base.detach().unsqueeze(-1)], dim=-1)
        x = _run_film_chain(self.filmenc, y, curve_feats, type_sample)
        sdf_detail = self.decoder.forward_simple(x).squeeze(-1)
        gate_detail = torch.exp(-(sdf_base.detach() ** 2) / (2.0 * self.sigma ** 2))
        return {
            'sdf_detail': sdf_detail,
            'gate_detail': gate_detail,
            'detail_grid_ct': self.grid_curvetheta.grids,
        }

    def detail_parameters(self):
        for m in ([self.grid_curvetheta] + list(self.filmenc) + [self.decoder]):
            for p in m.parameters():
                yield p

    def named_detail_parameters(self):
        mods = {'detail_model.grid_curvetheta': self.grid_curvetheta}
        for i, m in enumerate(self.filmenc):
            mods[f'detail_model.filmenc.{i}'] = m
        mods['detail_model.decoder'] = self.decoder
        for mprefix, m in mods.items():
            for name, p in m.named_parameters(recurse=True):
                yield f'{mprefix}.{name}', p

    def shared_and_detail_parameters(self, shared_modules=()):
        for m in shared_modules:
            for p in m.parameters():
                yield p
        yield from self.detail_parameters()

    def named_shared_and_detail_parameters(self, shared_named_items=()):
        for prefix, m in shared_named_items:
            for name, p in m.named_parameters(recurse=True):
                yield f'{prefix}.{name}', p
        yield from self.named_detail_parameters()


# ---------------------------------------------------------------------
# Outer model
# ---------------------------------------------------------------------

class NGCNetGrid(nn.Module):
    def __init__(self, arg, arg_train):
        super().__init__()
        self.device = arg['device']
        self.dim_code = arg['dim_code']
        self.dim_type_base = arg['dim_type_base']
        self.details_level = arg['details_level']

        self.base1_epoch = int(arg_train['base1_epoch'])
        self.base2_epoch = int(arg_train['base2_epoch'])
        self.base_epoch = int(arg_train['base_epoch'])
        self.detail_epoch = int(arg_train['detail_epoch'])

        self.period_angle_enc = PeriodicEncoding(arg['num_period_encoding'], include_input=False)
        dim_out_angle = 2 * arg['num_period_encoding']
        dim_cond = arg['dim_curve_feat']
        dim_cond_detail = arg.get('dim_curve_feat_detail', dim_cond)

        self.pos_enc_curve = PosEncoding(arg['num_pos_encoding'], d_in=1)
        self.pos_enc_y = PosEncoding(arg['num_y_pos_encoding'], d_in=1)
        self.pos_enc_z = PosEncoding(arg['num_z_pos_encoding'], d_in=1)

        self.curve_embd = nn.Embedding(arg['n_curve'], self.dim_code)
        self.type_embd = nn.Embedding(2, self.dim_type_base)
        self.init_embedding(self.curve_embd, self.dim_code)
        self.init_embedding(self.type_embd, self.dim_type_base)
        self.curveencoder = FeatCurveEncoder(arg)

        self.rho_grid = arg['rho_grid']
        self.base_model = NGCNetGridBase(arg['base'], dim_out_angle, dim_cond, rho_grid=self.rho_grid)
        self.base_model.set_phase_schedule(self.base1_epoch, self.base2_epoch, self.base_epoch)
        self.detail_model = NGCNetGridDetail(arg['detail'], dim_out_angle, dim_cond_detail)

    def init_embedding(self, embedding, dim_code):
        nn.init.normal_(embedding.weight.data, 0., 1. / math.sqrt(dim_code))

    def set_post_mode(self):
        for pname, param in self.named_parameters():
            if 'embd' in pname:
                continue
            param.requires_grad = False

    def get_phase(self, curr_epoch):
        if curr_epoch < self.base1_epoch:
            return 'base1_only'
        if curr_epoch < self.base2_epoch:
            return 'base2_only'
        if curr_epoch < self.base_epoch:
            return 'base_joint'
        if curr_epoch < self.detail_epoch:
            return 'detail_only'
        return 'all_joint'

    def _shared_modules(self):
        return [
            self.curve_embd, self.type_embd, self.curveencoder,
            self.pos_enc_curve, self.pos_enc_y, self.pos_enc_z,
            self.period_angle_enc,
        ]

    def _shared_named(self):
        return {
            'curve_embd': self.curve_embd,
            'type_embd': self.type_embd,
            'curveencoder': self.curveencoder,
            'pos_enc_curve': self.pos_enc_curve,
            'pos_enc_y': self.pos_enc_y,
            'pos_enc_z': self.pos_enc_z,
            'period_angle_enc': self.period_angle_enc,
        }

    def base1_parameters(self):
        yield from self.base_model.shared_and_base1_parameters(self._shared_modules())

    def base2_parameters(self):
        yield from self.base_model.shared_and_base2_parameters(self._shared_modules())

    def detail_parameters(self):
        yield from self.detail_model.shared_and_detail_parameters(self._shared_modules())

    def base_parameters(self):
        yield from self.base_model.shared_and_base1_parameters(self._shared_modules())
        yield from self.base_model.base2_parameters()

    def named_base1_parameters(self):
        yield from self.base_model.named_shared_and_base1_parameters(self._shared_named().items())

    def named_base2_parameters(self):
        yield from self.base_model.named_shared_and_base2_parameters(self._shared_named().items())

    def named_detail_parameters(self):
        yield from self.detail_model.named_shared_and_detail_parameters(self._shared_named().items())

    def _encode_curve(self, mi, istrain):
        curve_code = self.curve_embd(mi['curve_idx'])
        B = curve_code.shape[0]
        type_curve = self.type_embd(torch.zeros(B, dtype=torch.long, device=self.device)).unsqueeze(1)
        type_sample = self.type_embd(torch.ones(B, dtype=torch.long, device=self.device)).unsqueeze(1)

        if istrain:
            curve_coords_posenc = self.pos_enc_curve(mi['coords'].unsqueeze(-1))
            radii_y_posenc = self.pos_enc_y(torch.log(mi['radius'][:, :, 0] + 1e-12).unsqueeze(-1))
            radii_z_posenc = self.pos_enc_z(torch.log(mi['radius'][:, :, 1] + 1e-12).unsqueeze(-1))
            sample_angle_periodenc = self.period_angle_enc(mi['angles'].unsqueeze(-1))
        else:
            curve_coords_posenc = self.pos_enc_curve.inference(mi['coords'].unsqueeze(-1))
            radii_y_posenc = self.pos_enc_y.inference(torch.log(mi['radius'][:, :, 0] + 1e-12).unsqueeze(-1))
            radii_z_posenc = self.pos_enc_z.inference(torch.log(mi['radius'][:, :, 1] + 1e-12).unsqueeze(-1))
            sample_angle_periodenc = self.period_angle_enc.inference(mi['angles'].unsqueeze(-1))

        curve_code_coords = torch.cat([curve_code, curve_coords_posenc, radii_y_posenc, radii_z_posenc], dim=-1)
        curve_feats = self.curveencoder(curve_code_coords) + type_curve
        return {
            'curve_code': curve_code,
            'type_curve': type_curve,
            'type_sample': type_sample,
            'curve_feats': curve_feats,
            'sample_angle_periodenc': sample_angle_periodenc,
        }

    def forwardsimple(self, model_input, curr_epoch=0, istrain=True):
        mi = model_input
        enc_curve = self._encode_curve(mi, istrain)
        phase = self.get_phase(curr_epoch) if istrain else 'all_joint'

        base_phase = phase if phase in ('base1_only', 'base2_only', 'base_joint') else 'base_joint'
        base_out = self.base_model(
            mi,
            curve_feats=enc_curve['curve_feats'],
            type_sample=enc_curve['type_sample'],
            sample_angle_periodenc=enc_curve['sample_angle_periodenc'],
            curr_epoch=curr_epoch,
            istrain=istrain,
            phase=base_phase,
        )
        sdf_base = base_out['sdf_base']

        detail_active = (phase in ('detail_only', 'all_joint')) or (not istrain)
        sdf_detail = torch.zeros_like(sdf_base)
        gate_detail = torch.zeros_like(sdf_base)
        alpha_detail = 0.0
        detail_grid_ct = None

        if detail_active:
            det_out = self.detail_model(
                mi,
                curve_feats=enc_curve['curve_feats'],
                type_sample=enc_curve['type_sample'],
                sample_angle_periodenc=enc_curve['sample_angle_periodenc'],
                sdf_base=sdf_base,
                istrain=istrain,
            )
            sdf_detail = det_out['sdf_detail']
            gate_detail = det_out['gate_detail']
            detail_grid_ct = det_out['detail_grid_ct']
            if phase == 'detail_only':
                alpha_detail = _ramp(curr_epoch, self.base_epoch, self.detail_model.alpha_warmup)
            else:
                alpha_detail = 1.0

        if not istrain:
            sdf = sdf_base + gate_detail * sdf_detail
        else:
            if phase in ('base1_only', 'base2_only', 'base_joint'):
                sdf = sdf_base
            elif phase == 'detail_only':
                sdf = sdf_base.detach() + alpha_detail * gate_detail * sdf_detail
            else:
                sdf = sdf_base + gate_detail * sdf_detail

        return {
            'sdf': sdf,
            'sdf_base': sdf_base,
            'sdf_base1': base_out['sdf_base1'],
            'sdf_base2': base_out['sdf_base2'],
            'sdf_detail': sdf_detail,
            'alpha_base2': base_out['alpha_base2'],
            'gate_base2': base_out['gate_base2'],
            'alpha_detail': alpha_detail,
            'gate_detail': gate_detail,
            'base1_grid_ct': base_out['base1_grid_ct'],
            'base1_grid_cr': base_out['base1_grid_cr'],
            'base2_grid_ct': base_out['base2_grid_ct'],
            'detail_grid_ct': detail_grid_ct,
            'base_grid_ct': base_out['base1_grid_ct'],
            'base_grid_cr': base_out['base1_grid_cr'],
            'code': enc_curve['curve_code'],
            'phase': phase,
        }

    def forwardstretch(self, model_input):
        mi = model_input
        enc = self._encode_curve(mi, istrain=False)
        base_out = self.base_model(
            mi,
            curve_feats=enc['curve_feats'],
            type_sample=enc['type_sample'],
            sample_angle_periodenc=enc['sample_angle_periodenc'],
            curr_epoch=self.base_epoch,
            istrain=False,
            phase='base_joint',
        )
        sdf_base = base_out['sdf_base']

        mi_detail = dict(mi)
        mi_detail['coords'] = mi['coords_detail']
        det_out = self.detail_model(
            mi_detail,
            curve_feats=enc['curve_feats'],
            type_sample=enc['type_sample'],
            sample_angle_periodenc=enc['sample_angle_periodenc'],
            sdf_base=sdf_base,
            istrain=False,
        )
        sdf_detail = det_out['sdf_detail']
        gate = det_out['gate_detail']
        
        if self.details_level == 0:
            sdf = sdf_base
        elif self.details_level == 1:
            sdf = sdf_detail
        else:
            sdf = sdf_base + gate * sdf_detail
        return {'sdf': sdf, 'sdf_base': sdf_base, 'code': enc['curve_code']}

    def forward(self, model_input, curr_epoch=0):
        return self.forwardsimple(model_input, curr_epoch)

    @torch.no_grad()
    def validation(self, model_input):
        return self.forwardsimple(model_input, istrain=False)

    @torch.no_grad()
    def inference(self, model_input, transform=None):
        if transform == 'stretch':
            curve_input = self.pack_data_stretch(model_input)
            out = self.forwardstretch(curve_input)
        else:
            curve_input = self.pack_data(model_input)
            out = self.forwardsimple(curve_input, istrain=False)
        return out['sdf'], out['sdf_base']

    @torch.no_grad()
    def mix_curve(self, model_input):
        mi = model_input
        curve_data = self.pack_data(mi)
        curve_idx = curve_data['curve_idx']
        curve_code = self.curve_embd(curve_idx)
        new_idx = mi['new_idx'] * torch.ones_like(curve_idx, dtype=int).to(curve_code.device)
        new_code = self.curve_embd(new_idx)

        B = curve_code.shape[0]
        type_curve = self.type_embd(torch.zeros(B, dtype=torch.long, device=self.device)).unsqueeze(1)
        type_sample = self.type_embd(torch.ones(B, dtype=torch.long, device=self.device)).unsqueeze(1)

        curve_coords_posenc = self.pos_enc_curve.inference(curve_data['coords'].unsqueeze(-1))
        radii_y_posenc = self.pos_enc_y.inference(torch.log(curve_data['radius'][:, :, 0] + 1e-12).unsqueeze(-1))
        radii_z_posenc = self.pos_enc_z.inference(torch.log(curve_data['radius'][:, :, 1] + 1e-12).unsqueeze(-1))
        sample_angle_periodenc = self.period_angle_enc.inference(curve_data['angles'].unsqueeze(-1))

        curve_code_coords = torch.cat([curve_code, curve_coords_posenc, radii_y_posenc, radii_z_posenc], dim=-1)
        new_code_coords = torch.cat([new_code, curve_coords_posenc, radii_y_posenc, radii_z_posenc], dim=-1)
        orig_curve_feats = self.curveencoder(curve_code_coords)
        new_curve_feats = self.curveencoder(new_code_coords)

        func1 = mi['mix_func1']
        func2 = mi['mix_func2']
        _ts1, weights1 = func1(curve_data['coords'])
        _ts2, weights2 = func2(curve_data['coords'])
        curve_feats = (orig_curve_feats * weights1[..., None] + new_curve_feats * weights2[..., None]) + type_curve

        base_out = self.base_model(
            curve_data,
            curve_feats=curve_feats,
            type_sample=type_sample,
            sample_angle_periodenc=sample_angle_periodenc,
            curr_epoch=self.base_epoch,
            istrain=False,
            phase='base_joint',
        )
        sdf_base = base_out['sdf_base']
        det_out = self.detail_model(
            curve_data,
            curve_feats=curve_feats,
            type_sample=type_sample,
            sample_angle_periodenc=sample_angle_periodenc,
            sdf_base=sdf_base,
            istrain=False,
        )
        sdf = sdf_base + det_out['gate_detail'] * det_out['sdf_detail']
        return sdf, sdf_base

    def _pack_common(self, mi, extra_keys=()):
        device = mi['device']
        n_sample = mi['coords'].shape[0]
        curve_idx = mi['curve_idx'] * torch.ones(n_sample, dtype=int)
        res = {
            'samples': torch.from_numpy(mi['samples_local']).float().to(device),
            'coords': torch.from_numpy(mi['coords']).float().to(device),
            'angles': torch.from_numpy(mi['angles']).float().to(device),
            'radius': torch.from_numpy(mi['radius']).float().to(device),
            'rho': torch.from_numpy(mi['rho']).float().to(device),
            'rho_n': torch.from_numpy(mi['rho_n']).float().to(device),
            'curve_idx': curve_idx.to(device),
        }
        for k in extra_keys:
            res[k] = torch.from_numpy(mi[k]).float().to(device)
        return {key: val.unsqueeze(0) for key, val in res.items()}

    def pack_data_base(self, model_input):
        return self._pack_common(model_input)

    def pack_data(self, model_input):
        return self._pack_common(model_input)

    def pack_data_stretch(self, model_input):
        return self._pack_common(model_input, extra_keys=('samples_detail', 'coords_detail'))


if __name__ == "__main__":
    pass
