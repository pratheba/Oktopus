import math
import torch
import torch.nn as nn
from torch.nn.utils import spectral_norm

from mlp import *
#from pos_encoding import PosEncoding
from pos_encoding_fromokto import PosEncoding
from periodic_encoding import PeriodicEncoding
from triplane import CurveThetaMultiResGrid, CurveRhoMultiResGrid
from mask import *
from model_utils import wrap01, wrap_theta


class NGCNetPEGrid(nn.Module):
    """docstring for NGCNetGrid ."""
    def __init__(self, arg, arg_train):
        super().__init__()
        self.device = arg['device']
        self.dim_code = arg['dim_code']
        self.dim_type_base = arg['dim_type_base']
        self.dim_type_detail = arg['dim_type_detail']

        self.details_level = arg['details_level']
        self.sigma = float(arg['sigma'])

        self.start_epoch = arg_train['base_only_end']
        self.linear_epoch = arg_train['detail_only_end']
        self.alpha_warmup = arg_train['alpha_warmup']
        self.gate_warmup = arg_train['gate_warmup']

        ########## BASE ###################      
        self.pos_enc_sample_base = PosEncoding(arg['num_pos_encoding_base'], d_in=3)
       
        self.rho_grid = arg['rho_grid']
        self.base_outdim = (arg['num_pos_encoding_base']* 2 +1)*3 #+ (arg['num_period_encoding'] * 2 + 1)*1

        ########## DETAIL ###################      
        self.gate = arg['gate']
        self.detail_grid_levels = arg['detail_grid_levels']
        self.detail_grid_dim = arg['detail_grid_dim']
        self.detail_grid_hw = arg['detail_grid_hw']
        self.detail_grid_curvetheta = CurveThetaMultiResGrid(base_hw = self.detail_grid_hw, levels=self.detail_grid_levels, dim=self.detail_grid_dim)
        if self.rho_grid:
            self.detail_grid_curverho = CurveRhoMultiResGrid(base_hw = self.detail_grid_hw, levels=self.detail_grid_levels, dim=self.detail_grid_dim)
            self.detail_grid_outdim = 2*self.detail_grid_levels * self.detail_grid_dim
        else:
            self.detail_grid_outdim = self.detail_grid_levels * self.detail_grid_dim

        self.period_angle_enc = PeriodicEncoding(arg['num_period_encoding'], include_input=False)
        dim_out_angle = 2*arg['num_period_encoding']


        ########## CURVE / SKELETON ###################      
        self.pos_enc_curve = PosEncoding(arg['num_pos_encoding'], d_in=1)
        self.pos_enc_y = PosEncoding(arg['num_y_pos_encoding'], d_in=1)
        self.pos_enc_z = PosEncoding(arg['num_z_pos_encoding'], d_in=1)

        self.curve_embd = nn.Embedding(arg['n_curve'], self.dim_code)
        self.type_embd = nn.Embedding(2, self.dim_type_base)
        self.init_embedding(self.curve_embd, self.dim_code)
        self.init_embedding(self.type_embd, self.dim_type_base)
        
        self.curveencoder = FeatCurveEncoder(arg)

        ######## DIMENSIONS for FILM ###########################
        dim_in_base = self.base_outdim
        dim_in_detail =  1 + dim_out_angle + self.detail_grid_outdim + 1 # rho, detail_features, periodenc_angle, sdf_base.detach

        dim_out_base = arg['dim_base_feat']
        dim_cond_base = arg['dim_curve_feat']

        ######## FILM BASE ###########################
        self.filmenc_base1 = FiLMEncoder(dim_in_base, dim_out_base, dim_cond_base) 
        self.filmenc_base2 = FiLMEncoder(dim_out_base, dim_out_base, dim_cond_base)
        self.decoder_base = MLP(**arg['decoder_base'])

        ############# FILM DETAIL ################

        dim_out_detail = arg['dim_detail_feat']
        dim_cond_detail = arg['dim_curve_feat_detail']
        self.filmenc_detail1 = FiLMEncoder(dim_in_detail, dim_out_detail, dim_cond_detail) # arg['dim_sample_feat'], arg[''])
        self.filmenc_detail2 = FiLMEncoder(dim_out_detail, dim_out_detail, dim_cond_detail)
        self.filmenc_detail3 = FiLMEncoder(dim_out_detail, dim_out_detail, dim_cond_detail)
    
        self.decoder_detail = MLP(**arg['decoder_detail'])

    def base_parameters(self):
        modules = [
            self.curve_embd,
            self.type_embd,
            self.curveencoder,
            self.filmenc_base1,
            self.filmenc_base2,
            self.decoder_base,
            ]

        for m in modules:
            for p in m.parameters():
                yield p


    def detail_parameters(self):
        modules = [
            self.curve_embd,
            self.type_embd,
            self.curveencoder,
            self.detail_grid_curvetheta,
            ]

        modules += [
            self.filmenc_detail1,
            self.filmenc_detail2,
            self.filmenc_detail3,
            self.decoder_detail,
        ]

        for m in modules:
            for p in m.parameters():
                yield p

    def init_embedding(self, embedding, dim_code):
        nn.init.normal_(embedding.weight.data, 0., 1./ math.sqrt(dim_code))
    
    def set_post_mode(self):
        for pname, param in self.core.named_parameters():
            if 'embd' in pname:
                continue
            param.requires_grad = False


    def forwardsimple(self, model_input, curr_epoch=0, istrain=True):
        mi = model_input
        # curve_idx:(Nb, Ns); coords:(Nb, Ns); samples(Nb,Ns,3)
        curve_code = self.curve_embd(mi['curve_idx'])
        B = curve_code.shape[0]

        type_curve  = self.type_embd(torch.zeros(B, dtype=torch.long, device=self.device)).unsqueeze(1)
        type_sample = self.type_embd(torch.ones(B, dtype=torch.long, device=self.device)).unsqueeze(1)

        if istrain:
            ############ BASE FEATURES ###################
            #samples_base_ctfeatures = self.base_grid_curvetheta(mi['coords'], mi['angles'])
            samples_base_features = self.pos_enc_sample_base(mi['samples'])
            #if curr_epoch >= self.start_epoch:
            samples_detail_ctfeatures = self.detail_grid_curvetheta(mi['coords'], mi['angles'])
            if self.rho_grid:
                samples_detail_crfeatures = self.detail_grid_curverho(mi['coords'], mi['rho_n'])
                samples_detail_features = torch.cat([mi['rho_n'].unsqueeze(-1), samples_detail_ctfeatures, samples_detail_crfeatures], dim=-1)
            else:
                samples_detail_features = torch.cat([mi['rho_n'].unsqueeze(-1), samples_detail_ctfeatures], dim=-1)

            ############ DETAIL FEATURES ###################
            #samples_detail_features = torch.cat([mi['rho_n'].unsqueeze(-1), samples_detail_ctfeatures], dim=-1)
            sample_angle_periodenc = self.period_angle_enc(mi['angles'].unsqueeze(-1))
            ############ CURVE FEATURES ###################
            curve_coords_posenc = self.pos_enc_curve(mi['coords'].unsqueeze(-1))
            radii_y_posenc = self.pos_enc_y(torch.log(mi['radius'][:,:,0]).unsqueeze(-1))
            radii_z_posenc = self.pos_enc_z(torch.log(mi['radius'][:,:,1]).unsqueeze(-1))
        else:
            samples_base_features = self.pos_enc_sample_base.inference(mi['samples'])
            samples_detail_ctfeatures = self.detail_grid_curvetheta.inference(mi['coords'], mi['angles'])
            if self.rho_grid:
                samples_detail_crfeatures = self.detail_grid_curverho.inference(mi['coords'], mi['rho_n'])
                samples_detail_features = torch.cat([mi['rho_n'].unsqueeze(-1), samples_detail_ctfeatures, samples_detail_crfeatures], dim=-1)
            else:
                samples_detail_features = torch.cat([mi['rho_n'].unsqueeze(-1), samples_detail_ctfeatures], dim=-1)

            sample_angle_periodenc = self.period_angle_enc.inference(mi['angles'].unsqueeze(-1))

            curve_coords_posenc = self.pos_enc_curve.inference(mi['coords'].unsqueeze(-1))
            radii_y_posenc = self.pos_enc_y.inference(torch.log(mi['radius'][:,:,0]).unsqueeze(-1))
            radii_z_posenc = self.pos_enc_z.inference(torch.log(mi['radius'][:,:,1]).unsqueeze(-1))

        
        curve_code_coords = torch.cat([curve_code, curve_coords_posenc, radii_y_posenc, radii_z_posenc], dim=-1)
        curve_feats = self.curveencoder(curve_code_coords) 

        ############ FILM BASE ################
        curve_feats = curve_feats + type_curve
        x = self.filmenc_base1(samples_base_features, curve_feats) + type_sample
        res = x
        x = self.filmenc_base2(x, curve_feats)
        x = x + res
        sdf_base = self.decoder_base.forward_simple(x).squeeze(-1)
        ############ FILM DETAILS ################
        #if curr_epoch >= self.start_epoch:
        y = torch.cat([samples_detail_features, sample_angle_periodenc, sdf_base.detach().unsqueeze(-1)], dim=-1)
        x = self.filmenc_detail1(y, curve_feats) + type_sample
        res = x
        x = self.filmenc_detail2(x, curve_feats)
        x = x + res
        res = x
        x = self.filmenc_detail3(x, curve_feats)
        x = x + res
        sdf_detail = self.decoder_detail.forward_simple(x).squeeze(-1)
        ############ Final ###########

        self.alpha_warmup += self.start_epoch
        self.gate_warmup += self.start_epoch
        if istrain:
            if curr_epoch < self.start_epoch:
                sdf = sdf_base
            elif curr_epoch < self.linear_epoch:
                #if curr_epoch < gate_warmup:
                #    alpha = 0.1
                #    gate = 1
                alpha = min(1.0, (curr_epoch - self.start_epoch) / (self.alpha_warmup - self.start_epoch))
                gate = 1.0 if curr_epoch < self.gate_warmup else torch.exp(-(sdf_base.detach()**2)/(2*(self.sigma**2)))
                #if curr_epoch < alpha_warmup:
                #    alpha = float(curr_epoch - start_epoch)/float(alpha_warmup - start_epoch) #  % 1000)/1000
                #gate = 1.0
                sdf = sdf_base.detach() + alpha * gate * sdf_detail
            else:
                gate = torch.exp(-(sdf_base.detach()**2)/(2*(self.sigma**2)))
                sdf = sdf_base + gate * sdf_detail
        else:
            return {
                'sdf': sdf_base, 
                'code': curve_code,
            }
            if self.details_level == 0: 
                sdf = sdf_base
            elif self.details_level == 1: 
                sdf = sdf_detail
            else:
                gate = torch.exp(-(sdf_base.detach()**2)/(2*(self.sigma**2)))
                sdf = sdf_base +  gate * sdf_detail

        return {
            'sdf': sdf, 
            'sdf_base': sdf_base,
            'code': curve_code,
        }

    def forwardstretch(self, model_input):
        mi = model_input
        # curve_idx:(Nb, Ns); coords:(Nb, Ns); samples(Nb,Ns,3)
        curve_code = self.curve_embd(mi['curve_idx'])
        B = curve_code.shape[0]

        type_curve  = self.type_embd(torch.zeros(B, dtype=torch.long, device=self.device)).unsqueeze(1)
        type_sample = self.type_embd(torch.ones(B, dtype=torch.long, device=self.device)).unsqueeze(1)

        samples_base_ctfeatures = self.base_grid_curvetheta.inference(mi['coords'], mi['angles'])
        if self.rho_grid:
            samples_base_crfeatures = self.base_grid_curverho.inference(mi['coords'], mi['rho_n'])
            samples_base_features = torch.cat([mi['rho'].unsqueeze(-1), samples_base_ctfeatures, samples_base_crfeatures], dim=-1)
        else:
            samples_base_features = torch.cat([mi['rho'].unsqueeze(-1), samples_base_ctfeatures], dim=-1)

        samples_detail_ctfeatures = self.detail_grid_curvetheta.inference(mi['coords_detail'], mi['angles'])
        samples_detail_features = torch.cat([mi['rho_n'].unsqueeze(-1), samples_detail_ctfeatures], dim=-1)
        sample_angle_periodenc = self.period_angle_enc.inference(mi['angles'].unsqueeze(-1))

        curve_coords_posenc = self.pos_enc_curve.inference(mi['coords'].unsqueeze(-1))
        radii_y_posenc = self.pos_enc_y.inference(torch.log(mi['radius'][:,:,0]).unsqueeze(-1))
        radii_z_posenc = self.pos_enc_z.inference(torch.log(mi['radius'][:,:,1]).unsqueeze(-1))

        curve_code_coords = torch.cat([curve_code, curve_coords_posenc, radii_y_posenc, radii_z_posenc], dim=-1)
        curve_feats = self.curveencoder(curve_code_coords) 
        curve_feats = curve_feats + type_curve

        ############ FILM BASE ################
        x = self.filmenc_base1(samples_base_features, curve_feats) + type_sample
        res = x
        x = self.filmenc_base2(x, curve_feats)
        x = x + res
        sdf_base = self.decoder_base.forward_simple(x).squeeze(-1)
        ############ FILM DETAILS ################
        y = torch.cat([samples_detail_features, sample_angle_periodenc, sdf_base.detach().unsqueeze(-1)], dim=-1)
        x = self.filmenc_detail1(y, curve_feats) + type_sample
        res = x
        x = self.filmenc_detail2(x, curve_feats)
        x = x + res
        res = x
        x = self.filmenc_detail3(x, curve_feats)
        x = x + res
        sdf_detail = self.decoder_detail.forward_simple(x).squeeze(-1)

        ############ Final ###########
        if self.details_level == 0: 
            sdf = sdf_base
        elif self.details_level == 1: 
            sdf = sdf_detail
        else:
            gate = torch.exp(-(sdf_base**2)/(2*(self.sigma**2)))
            sdf = sdf_base +  gate * sdf_detail

        return {
            'sdf': sdf,
            'code': curve_code,
        }

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
        return out['sdf']

    @torch.no_grad()
    def mix_curve(self, model_input):
        # assume only one curve
        mi = model_input
        curve_data = self.pack_data(mi)
        curve_idx = curve_data['curve_idx']
        curve_code = self.embd(curve_idx)
        new_idx = mi['new_idx']* torch.ones_like(curve_idx, dtype=int).to(curve_code.device)
        new_code = self.embd(new_idx)
        out = self.forwardsimple(curve_input, istrain=False)

        curve_feats = self.encoder.mix_feats(
            curve_code, new_code, curve_data['coords'], mi)
        
        if hasattr(self, 'pos_enc'):
            samples = self.pos_enc(curve_data['samples'])
        else:
            samples = curve_data['samples']
        curve_feats = torch.cat([curve_feats, samples], dim=-1)
        curve_sdf = self.decoder.forward_simple(curve_feats).squeeze(-1)
        return curve_sdf


    def pack_data(self, model_input):
        mi = model_input
        device = mi['device']

        n_sample = mi['coords'].shape[0]
        curve_idx = mi['curve_idx']*torch.ones(n_sample, dtype=int)
        
        res = {
            'samples': torch.from_numpy(mi['samples_local']).float().to(device),
            'coords': torch.from_numpy(mi['coords']).float().to(device),
            #'rho': torch.from_numpy(mi['rho']).float().to(device),
            'angles': torch.from_numpy(mi['angles']).float().to(device),
            'radius': torch.from_numpy(mi['radius']).float().to(device),
            'rho': torch.from_numpy(mi['rho']).float().to(device),
            'rho_n': torch.from_numpy(mi['rho_n']).float().to(device),
            'curve_idx': curve_idx.to(device)
        }

        res = {key:val.unsqueeze(0) for key,val in res.items()}
        return res

    def pack_data_stretch(self, model_input):
        mi = model_input
        device = mi['device']

        n_sample = mi['coords'].shape[0]
        curve_idx = mi['curve_idx']*torch.ones(n_sample, dtype=int)
        
        res = {
            'samples': torch.from_numpy(mi['samples_local']).float().to(device),
            #'seam': torch.from_numpy(mi['seam']).float().to(device),
            'samples_detail': torch.from_numpy(mi['samples_detail']).float().to(device),
            #'samples_detail_notile': torch.from_numpy(mi['samples_detail_notile']).float().to(device),
            'coords': torch.from_numpy(mi['coords']).float().to(device),
            'coords_detail': torch.from_numpy(mi['coords_detail']).float().to(device),
            #'rho': torch.from_numpy(mi['rho']).float().to(device),
            'angles': torch.from_numpy(mi['angles']).float().to(device),
            'radius': torch.from_numpy(mi['radius']).float().to(device),
            'rho': torch.from_numpy(mi['rho']).float().to(device),
            'rho_n': torch.from_numpy(mi['rho_n']).float().to(device),
            #'radius_detail': torch.from_numpy(mi['radius_detail']).float().to(device),
            'curve_idx': curve_idx.to(device)
        }

        res = {key:val.unsqueeze(0) for key,val in res.items()}
        return res


class NGCNetGrid(nn.Module):
    """docstring for NGCNetGrid ."""
    def __init__(self, arg, arg_train):
        super().__init__()
        self.device = arg['device']
        self.dim_code = arg['dim_code']
        self.dim_type_base = arg['dim_type_base']
        self.dim_type_detail = arg['dim_type_detail']

        self.details_level = arg['details_level']
        self.sigma = float(arg['sigma'])
        self.reduce_channel = arg['reduce_channel']

        self.start_epoch = arg_train['base_only_end']
        self.linear_epoch = arg_train['detail_only_end']
        self.alpha_warmup = arg_train['alpha_warmup']
        self.gate_warmup = arg_train['gate_warmup']

        ########## BASE ###################      
        self.base_grid_levels = arg['base_grid_levels']
        self.base_grid_dim = arg['base_grid_dim']
        self.base_grid_hw = arg['base_grid_hw']
        self.base_grid_curvetheta = CurveThetaMultiResGrid(base_hw = self.base_grid_hw, levels=self.base_grid_levels, dim=self.base_grid_dim, reduce=self.reduce_channel)
       
        self.rho_grid = arg['rho_grid']
        if self.rho_grid:
            self.base_grid_curverho = CurveRhoMultiResGrid(base_hw = self.base_grid_hw, levels=self.base_grid_levels, dim=self.base_grid_dim, reduce=self.reduce_channel)
            if self.reduce_channel:
                dim = 0
                for i in range(self.base_grid_levels):
                    dim +=  int(self.base_grid_dim /(2**i))
                self.base_grid_outdim = 2*dim
            else:
                self.base_grid_outdim = 2*self.base_grid_levels * self.base_grid_dim
        else:
            #self.base_grid_outdim = self.base_grid_levels * self.base_grid_dim
            if self.reduce_channel:
                dim = self.base_grid_dim
                for i in range(self.base_grid_levels):
                    dim +=  int(dim /(2**i))
                self.base_grid_outdim = dim
            else:
                self.base_grid_outdim = self.base_grid_levels * self.base_grid_dim

        ########## DETAIL ###################      
        self.gate = arg['gate']
        self.detail_grid_levels = arg['detail_grid_levels']
        self.detail_grid_dim = arg['detail_grid_dim']
        self.detail_grid_hw = arg['detail_grid_hw']
        self.detail_grid_curvetheta = CurveThetaMultiResGrid(base_hw = self.detail_grid_hw, levels=self.detail_grid_levels, dim=self.detail_grid_dim, reduce=self.reduce_channel)
        if self.reduce_channel:
            dim = 0 #self.detail_grid_dim
            for i in range(self.detail_grid_levels):
                dim +=  int(self.detail_grid_dim /(2**i))
            self.detail_grid_outdim = dim #self.detail_grid_levels * self.detail_grid_dim
        else:
            self.detail_grid_outdim = self.detail_grid_levels * self.detail_grid_dim

        self.period_angle_enc = PeriodicEncoding(arg['num_period_encoding'], include_input=False)
        dim_out_angle = 2*arg['num_period_encoding']


        ########## CURVE / SKELETON ###################      
        self.pos_enc_curve = PosEncoding(arg['num_pos_encoding'], d_in=1)
        self.pos_enc_y = PosEncoding(arg['num_y_pos_encoding'], d_in=1)
        self.pos_enc_z = PosEncoding(arg['num_z_pos_encoding'], d_in=1)

        self.curve_embd = nn.Embedding(arg['n_curve'], self.dim_code)
        self.type_embd = nn.Embedding(2, self.dim_type_base)
        self.init_embedding(self.curve_embd, self.dim_code)
        self.init_embedding(self.type_embd, self.dim_type_base)
        
        self.curveencoder = FeatCurveEncoder(arg)

        ######## DIMENSIONS for FILM ###########################
        dim_in_base = 1 + self.base_grid_outdim
        dim_in_detail =  1 + dim_out_angle + self.detail_grid_outdim + 1 # rho, detail_features, periodenc_angle, sdf_base.detach

        dim_out_base = arg['dim_base_feat']
        dim_cond_base = arg['dim_curve_feat']

        ######## FILM BASE ###########################
        self.filmenc_base1 = FiLMEncoder(dim_in_base, dim_out_base, dim_cond_base) 
        self.filmenc_base2 = FiLMEncoder(dim_out_base, dim_out_base, dim_cond_base)
        self.decoder_base = MLP(**arg['decoder_base'])

        ############# FILM DETAIL ################

        dim_out_detail = arg['dim_detail_feat']
        dim_cond_detail = arg['dim_curve_feat_detail']
        self.filmenc_detail1 = FiLMEncoder(dim_in_detail, dim_out_detail, dim_cond_detail) # arg['dim_sample_feat'], arg[''])
        self.filmenc_detail2 = FiLMEncoder(dim_out_detail, dim_out_detail, dim_cond_detail)
        self.filmenc_detail3 = FiLMEncoder(dim_out_detail, dim_out_detail, dim_cond_detail)
    
        self.decoder_detail = MLP(**arg['decoder_detail'])

    def named_base_parameters(self):
        modules = {
            "curve_embd": self.curve_embd,
            "type_embd": self.type_embd,
            "curveencoder": self.curveencoder,
            "base_grid_curvetheta": self.base_grid_curvetheta,
            "filmenc_base1": self.filmenc_base1,
            "filmenc_base2": self.filmenc_base2,
            "decoder_base": self.decoder_base,
        }
        if self.rho_grid:
            modules["base_grid_curverho"] = self.base_grid_curverho

        for prefix, m in modules.items():
            for name, p in m.named_parameters(recurse=True):
                yield f"{prefix}.{name}", p

    def base_parameters(self):
        modules = [
            self.curve_embd,
            self.type_embd,
            self.curveencoder,
            self.base_grid_curvetheta,
            ]

        if self.rho_grid:
            modules += [self.base_grid_curverho]
        modules += [
            self.filmenc_base1,
            self.filmenc_base2,
            self.decoder_base,
        ]

        for m in modules:
            for p in m.parameters():
                yield p


    def named_detail_parameters(self):
        modules = {
            "curve_embd": self.curve_embd,
            "type_embd": self.type_embd,
            "curveencoder": self.curveencoder,
            "detail_grid_curvetheta": self.detail_grid_curvetheta,
            "filmenc_detail1": self.filmenc_detail1,
            "filmenc_detail2": self.filmenc_detail2,
            "filmenc_detail3": self.filmenc_detail2,
            "decoder_detail": self.decoder_detail,
        }

        for prefix, m in modules.items():
            for name, p in m.named_parameters(recurse=True):
                yield f"{prefix}.{name}", p

    def detail_parameters(self):
        modules = [
            self.curve_embd,
            self.type_embd,
            self.curveencoder,
            self.detail_grid_curvetheta,
            ]

        modules += [
            self.filmenc_detail1,
            self.filmenc_detail2,
            self.filmenc_detail3,
            self.decoder_detail,
        ]

        for m in modules:
            for p in m.parameters():
                yield p

    def init_embedding(self, embedding, dim_code):
        nn.init.normal_(embedding.weight.data, 0., 1./ math.sqrt(dim_code))
    
    def set_post_mode(self):
        for pname, param in self.core.named_parameters():
            if 'embd' in pname:
                continue
            param.requires_grad = False


    def forwardsimple(self, model_input, curr_epoch=0, istrain=True):
        mi = model_input
        # curve_idx:(Nb, Ns); coords:(Nb, Ns); samples(Nb,Ns,3)
        curve_code = self.curve_embd(mi['curve_idx'])
        B = curve_code.shape[0]

        type_curve  = self.type_embd(torch.zeros(B, dtype=torch.long, device=self.device)).unsqueeze(1)
        type_sample = self.type_embd(torch.ones(B, dtype=torch.long, device=self.device)).unsqueeze(1)

        if istrain:
            ############ BASE FEATURES ###################
            samples_base_ctfeatures = self.base_grid_curvetheta(mi['coords'], mi['angles'])
            if self.rho_grid:
                samples_base_crfeatures = self.base_grid_curverho(mi['coords'], mi['rho_n'])
                samples_base_features = torch.cat([mi['rho'].unsqueeze(-1), samples_base_ctfeatures, samples_base_crfeatures], dim=-1)
            else:
                samples_base_features = torch.cat([mi['rho'].unsqueeze(-1), samples_base_ctfeatures], dim=-1)

            ############ DETAIL FEATURES ###################
            samples_detail_ctfeatures = self.detail_grid_curvetheta(mi['coords'], mi['angles'])
            samples_detail_features = torch.cat([mi['rho_n'].unsqueeze(-1), samples_detail_ctfeatures], dim=-1)
            sample_angle_periodenc = self.period_angle_enc(mi['angles'].unsqueeze(-1))
            ############ CURVE FEATURES ###################
            curve_coords_posenc = self.pos_enc_curve(mi['coords'].unsqueeze(-1))
            radii_y_posenc = self.pos_enc_y(torch.log(mi['radius'][:,:,0]+1e-12).unsqueeze(-1))
            radii_z_posenc = self.pos_enc_z(torch.log(mi['radius'][:,:,1]+1e-12).unsqueeze(-1))
            #radii_y_posenc = self.pos_enc_y(torch.log(mi['radius'][:,:,0]).unsqueeze(-1))
            #radii_z_posenc = self.pos_enc_z(torch.log(mi['radius'][:,:,1]).unsqueeze(-1))
        else:
            samples_base_ctfeatures = self.base_grid_curvetheta.inference(mi['coords'], mi['angles'])
            if self.rho_grid:
                samples_base_crfeatures = self.base_grid_curverho.inference(mi['coords'], mi['rho_n'])
                samples_base_features = torch.cat([mi['rho'].unsqueeze(-1), samples_base_ctfeatures, samples_base_crfeatures], dim=-1)
            else:
                samples_base_features = torch.cat([mi['rho'].unsqueeze(-1), samples_base_ctfeatures], dim=-1)

            samples_detail_ctfeatures = self.detail_grid_curvetheta.inference(mi['coords'], mi['angles'])
            samples_detail_features = torch.cat([mi['rho_n'].unsqueeze(-1), samples_detail_ctfeatures], dim=-1)
            sample_angle_periodenc = self.period_angle_enc.inference(mi['angles'].unsqueeze(-1))

            curve_coords_posenc = self.pos_enc_curve.inference(mi['coords'].unsqueeze(-1))
            radii_y_posenc = self.pos_enc_y.inference(torch.log(mi['radius'][:,:,0]+1e-12).unsqueeze(-1))
            radii_z_posenc = self.pos_enc_z.inference(torch.log(mi['radius'][:,:,1]+1e-12).unsqueeze(-1))
            #radii_y_posenc = self.pos_enc_y.inference(torch.log(mi['radius'][:,:,0]).unsqueeze(-1))
            #radii_z_posenc = self.pos_enc_z.inference(torch.log(mi['radius'][:,:,1]).unsqueeze(-1))

        
        curve_code_coords = torch.cat([curve_code, curve_coords_posenc, radii_y_posenc, radii_z_posenc], dim=-1)
        curve_feats = self.curveencoder(curve_code_coords) 

        ############ FILM BASE ################
        curve_feats = curve_feats + type_curve
        x = self.filmenc_base1(samples_base_features, curve_feats) + type_sample
        res = x
        x = self.filmenc_base2(x, curve_feats)
        x = x + res
        sdf_base = self.decoder_base.forward_simple(x).squeeze(-1)
        ############ FILM DETAILS ################
        y = torch.cat([samples_detail_features, sample_angle_periodenc, sdf_base.detach().unsqueeze(-1)], dim=-1)
        x = self.filmenc_detail1(y, curve_feats) + type_sample
        res = x
        x = self.filmenc_detail2(x, curve_feats)
        x = x + res
        res = x
        x = self.filmenc_detail3(x, curve_feats)
        x = x + res
        sdf_detail = self.decoder_detail.forward_simple(x).squeeze(-1)
        ############ Final ###########

        #start_epoch = 2000
        #linear_epoch = 4500
        #alpha_warmup = 3000
        #gate_warmup = 3500
        #detail_band = 0.05
        self.alpha_warmup += self.start_epoch
        self.gate_warmup += self.start_epoch
        if istrain:
            if curr_epoch < self.start_epoch:
                sdf = sdf_base
            elif curr_epoch < self.linear_epoch:
                #if curr_epoch < gate_warmup:
                #    alpha = 0.1
                #    gate = 1
                alpha = min(1.0, (curr_epoch - self.start_epoch) / (self.alpha_warmup - self.start_epoch))
                #gate = 1.0 if curr_epoch < self.gate_warmup else torch.exp(-(sdf_base.detach()**2)/(2*(self.sigma**2)))
                gate = torch.exp(-(sdf_base.detach()**2)/(2*(self.sigma**2)))
                #if curr_epoch < alpha_warmup:
                #    alpha = float(curr_epoch - start_epoch)/float(alpha_warmup - start_epoch) #  % 1000)/1000
                #gate = 1.0
                sdf = sdf_base.detach() + alpha * gate * sdf_detail
            else:
                gate = torch.exp(-(sdf_base.detach()**2)/(2*(self.sigma**2)))
                sdf = sdf_base + gate * sdf_detail
        else:
            #return {
            #    'sdf': sdf,
            #    'sdf_base': sdf_base,
            #    'code': curve_code
            #}
            if self.details_level == 0: 
                sdf = sdf_base
            elif self.details_level == 1: 
                sdf = sdf_detail
            else:
                gate = torch.exp(-(sdf_base.detach()**2)/(2*(self.sigma**2)))
                sdf = sdf_base +  gate * sdf_detail

        #print(self.base_grid_curvetheta.grids, flush=True)
        return {
            'sdf': sdf, 
            'sdf_base': sdf_base,
            #'sdf_detail': sdf_detail,
            'base_grid_ct': self.base_grid_curvetheta.grids,
            'base_grid_cr': self.base_grid_curverho.grids,
            'code': curve_code,
        }

    def forwardstretch(self, model_input):
        mi = model_input
        # curve_idx:(Nb, Ns); coords:(Nb, Ns); samples(Nb,Ns,3)
        curve_code = self.curve_embd(mi['curve_idx'])
        B = curve_code.shape[0]

        type_curve  = self.type_embd(torch.zeros(B, dtype=torch.long, device=self.device)).unsqueeze(1)
        type_sample = self.type_embd(torch.ones(B, dtype=torch.long, device=self.device)).unsqueeze(1)

        samples_base_ctfeatures = self.base_grid_curvetheta.inference(mi['coords'], mi['angles'])
        if self.rho_grid:
            samples_base_crfeatures = self.base_grid_curverho.inference(mi['coords'], mi['rho_n'])
            samples_base_features = torch.cat([mi['rho'].unsqueeze(-1), samples_base_ctfeatures, samples_base_crfeatures], dim=-1)
        else:
            samples_base_features = torch.cat([mi['rho'].unsqueeze(-1), samples_base_ctfeatures], dim=-1)

        samples_detail_ctfeatures = self.detail_grid_curvetheta.inference(mi['coords_detail'], mi['angles'])
        samples_detail_features = torch.cat([mi['rho_n'].unsqueeze(-1), samples_detail_ctfeatures], dim=-1)
        sample_angle_periodenc = self.period_angle_enc.inference(mi['angles'].unsqueeze(-1))

        curve_coords_posenc = self.pos_enc_curve.inference(mi['coords'].unsqueeze(-1))
        radii_y_posenc = self.pos_enc_y.inference(torch.log(mi['radius'][:,:,0]).unsqueeze(-1))
        radii_z_posenc = self.pos_enc_z.inference(torch.log(mi['radius'][:,:,1]).unsqueeze(-1))

        curve_code_coords = torch.cat([curve_code, curve_coords_posenc, radii_y_posenc, radii_z_posenc], dim=-1)
        curve_feats = self.curveencoder(curve_code_coords) 
        curve_feats = curve_feats + type_curve

        ############ FILM BASE ################
        x = self.filmenc_base1(samples_base_features, curve_feats) + type_sample
        res = x
        x = self.filmenc_base2(x, curve_feats)
        x = x + res
        sdf_base = self.decoder_base.forward_simple(x).squeeze(-1)
        ############ FILM DETAILS ################
        y = torch.cat([samples_detail_features, sample_angle_periodenc, sdf_base.detach().unsqueeze(-1)], dim=-1)
        x = self.filmenc_detail1(y, curve_feats) + type_sample
        res = x
        x = self.filmenc_detail2(x, curve_feats)
        x = x + res
        res = x
        x = self.filmenc_detail3(x, curve_feats)
        x = x + res
        sdf_detail = self.decoder_detail.forward_simple(x).squeeze(-1)

        ############ Final ###########
        if self.details_level == 0: 
            sdf = sdf_base
        elif self.details_level == 1: 
            sdf = sdf_detail
        else:
            gate = torch.exp(-(sdf_base**2)/(2*(self.sigma**2)))
            sdf = sdf_base +  gate * sdf_detail

        return {
            'sdf': sdf,
            'code': curve_code,
        }

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
        # assume only one curve
        mi = model_input
        curve_data = self.pack_data(mi)
        curve_idx = curve_data['curve_idx']
        curve_code = self.curve_embd(curve_idx)
        new_idx = mi['new_idx']* torch.ones_like(curve_idx, dtype=int).to(curve_code.device)
        new_code = self.curve_embd(new_idx)

        B = curve_code.shape[0]
        type_curve  = self.type_embd(torch.zeros(B, dtype=torch.long, device=self.device)).unsqueeze(1)
        type_sample = self.type_embd(torch.ones(B, dtype=torch.long, device=self.device)).unsqueeze(1)

        samples_base_ctfeatures = self.base_grid_curvetheta.inference(curve_data['coords'], curve_data['angles'])
        if self.rho_grid:
            samples_base_crfeatures = self.base_grid_curverho.inference(curve_data['coords'], curve_data['rho_n'])
            samples_base_features = torch.cat([curve_data['rho'].unsqueeze(-1), samples_base_ctfeatures, samples_base_crfeatures], dim=-1)
        else:
            samples_base_features = torch.cat([curve_data['rho'].unsqueeze(-1), samples_base_ctfeatures], dim=-1)

        samples_detail_ctfeatures = self.detail_grid_curvetheta.inference(curve_data['coords'], curve_data['angles'])
        samples_detail_features = torch.cat([curve_data['rho_n'].unsqueeze(-1), samples_detail_ctfeatures], dim=-1)
        sample_angle_periodenc = self.period_angle_enc.inference(curve_data['angles'].unsqueeze(-1))

        curve_coords_posenc = self.pos_enc_curve.inference(curve_data['coords'].unsqueeze(-1))
        radii_y_posenc = self.pos_enc_y.inference(torch.log(curve_data['radius'][:,:,0]+1e-12).unsqueeze(-1))
        radii_z_posenc = self.pos_enc_z.inference(torch.log(curve_data['radius'][:,:,1]+1e-12).unsqueeze(-1))

        curve_code_coords = torch.cat([curve_code, curve_coords_posenc, radii_y_posenc, radii_z_posenc], dim=-1)
        new_code_coords = torch.cat([new_code, curve_coords_posenc, radii_y_posenc, radii_z_posenc], dim=-1)
        orig_curve_feats = self.curveencoder(curve_code_coords) 
        new_curve_feats = self.curveencoder(new_code_coords) 

        #curve_feats, ts1, ts2, weights1, weights2 = self.encoder.mix_feats(curve_data['coords'], mi)

        func1 = mi['mix_func1']
        func2 = mi['mix_func2']
        ts1, weights1 = func1(curve_data['coords'])
        ts2, weights2 = func2(curve_data['coords'])

        curve_feats = orig_curve_feats*weights1[...,None] + new_curve_feats*weights2[...,None]
        ############ FILM BASE ################
        curve_feats = curve_feats + type_curve
        x = self.filmenc_base1(samples_base_features, curve_feats) + type_sample
        res = x
        x = self.filmenc_base2(x, curve_feats)
        x = x + res
        sdf_base = self.decoder_base.forward_simple(x).squeeze(-1)
        ############ FILM DETAILS ################
        y = torch.cat([samples_detail_features, sample_angle_periodenc, sdf_base.detach().unsqueeze(-1)], dim=-1)
        x = self.filmenc_detail1(y, curve_feats) + type_sample
        res = x
        x = self.filmenc_detail2(x, curve_feats)
        x = x + res
        res = x
        x = self.filmenc_detail3(x, curve_feats)
        x = x + res
        sdf_detail = self.decoder_detail.forward_simple(x).squeeze(-1)

        gate = torch.exp(-(sdf_base.detach()**2)/(2*(self.sigma**2)))
        sdf = sdf_base +  gate * sdf_detail

        return sdf, sdf_base


    def pack_data_base(self, model_input):
        mi = model_input
        device = mi['device']

        n_sample = mi['coords'].shape[0]
        curve_idx = mi['curve_idx']*torch.ones(n_sample, dtype=int)
        
        res = {
            'samples': torch.from_numpy(mi['samples_local']).float().to(device),
            'coords': torch.from_numpy(mi['coords']).float().to(device),
            'angles': torch.from_numpy(mi['angles']).float().to(device),
            'radius': torch.from_numpy(mi['radius']).float().to(device),
            'rho': torch.from_numpy(mi['rho']).float().to(device),
            'rho_n': torch.from_numpy(mi['rho_n']).float().to(device),
            'curve_idx': curve_idx.to(device)
        }

        res = {key:val.unsqueeze(0) for key,val in res.items()}
        return res

    def pack_data(self, model_input):
        mi = model_input
        device = mi['device']

        n_sample = mi['coords'].shape[0]
        curve_idx = mi['curve_idx']*torch.ones(n_sample, dtype=int)
        
        res = {
            'samples': torch.from_numpy(mi['samples_local']).float().to(device),
            'coords': torch.from_numpy(mi['coords']).float().to(device),
            #'rho': torch.from_numpy(mi['rho']).float().to(device),
            'angles': torch.from_numpy(mi['angles']).float().to(device),
            'radius': torch.from_numpy(mi['radius']).float().to(device),
            'rho': torch.from_numpy(mi['rho']).float().to(device),
            'rho_n': torch.from_numpy(mi['rho_n']).float().to(device),
            'curve_idx': curve_idx.to(device)
        }

        res = {key:val.unsqueeze(0) for key,val in res.items()}
        return res

    def pack_data_stretch(self, model_input):
        mi = model_input
        device = mi['device']

        n_sample = mi['coords'].shape[0]
        curve_idx = mi['curve_idx']*torch.ones(n_sample, dtype=int)
        
        res = {
            'samples': torch.from_numpy(mi['samples_local']).float().to(device),
            #'seam': torch.from_numpy(mi['seam']).float().to(device),
            'samples_detail': torch.from_numpy(mi['samples_detail']).float().to(device),
            #'samples_detail_notile': torch.from_numpy(mi['samples_detail_notile']).float().to(device),
            'coords': torch.from_numpy(mi['coords']).float().to(device),
            'coords_detail': torch.from_numpy(mi['coords_detail']).float().to(device),
            #'rho': torch.from_numpy(mi['rho']).float().to(device),
            'angles': torch.from_numpy(mi['angles']).float().to(device),
            'radius': torch.from_numpy(mi['radius']).float().to(device),
            'rho': torch.from_numpy(mi['rho']).float().to(device),
            'rho_n': torch.from_numpy(mi['rho_n']).float().to(device),
            #'radius_detail': torch.from_numpy(mi['radius_detail']).float().to(device),
            'curve_idx': curve_idx.to(device)
        }

        res = {key:val.unsqueeze(0) for key,val in res.items()}
        return res


class NGCNet(nn.Module):
    """docstring for NGCNet ."""
    def __init__(self, arg):
        super().__init__()
        self.device = arg['device']
        self.dim_code = arg['dim_code']
        self.dim_feat = arg['dim_feat']
        self.dim_type = arg['dim_type']


        self.details_level = arg['details_level']
        self.num_freq_detail = arg['num_pos_encoding_detail']
      
        self.gate = arg['gate']
        self.grid_levels = arg['grid_levels']
        self.grid_dim = arg['grid_dim']
        self.grid_hw = arg['grid_hw']
        self.grid_curvetheta = CurveThetaMultiResGrid(base_hw = self.grid_hw, levels=self.grid_levels, dim=self.grid_dim)
       
        self.rho_grid = arg['rho_grid']
        if self.rho_grid:
            self.grid_curverho = CurveRhoMultiResGrid(base_hw = self.grid_hw, levels=self.grid_levels, dim=self.grid_dim)
        #self.grid_curverho = CurveRhoMultiResGrid(base_hw = self.grid_hw, levels=self.grid_levels, dim=self.grid_dim)
        self.grid_outdim = self.grid_levels * self.grid_dim

        self.pos_enc_sample_base = PosEncoding(arg['num_pos_encoding_base'], d_in=3)
        self.pos_enc_sample_detail = PosEncoding(arg['num_pos_encoding_detail'], start=4, d_in=3)

        self.pos_enc_curve = PosEncoding(arg['num_pos_encoding'], d_in=1)
        self.pos_enc_y = PosEncoding(arg['num_y_pos_encoding'], d_in=1)
        self.pos_enc_z = PosEncoding(arg['num_z_pos_encoding'], d_in=1)
        self.pos_enc_curve_tile = PosEncoding(arg['num_pos_encoding'], d_in=1, include_input=False)


        self.period_enc = PeriodicEncoding(arg['num_period_encoding'], include_input=False)
        #self.period_enc_tile = PeriodicEncoding(arg['num_period_encoding'], include_input=False)

        self.embd = nn.Embedding(arg['n_curve'], self.dim_code)
        self.type_embd = nn.Embedding(3, self.dim_type)
        self.init_embedding(self.embd, self.dim_code)
        self.init_embedding(self.type_embd, self.dim_type)
        
        self.curveencoder = FeatCurveEncoder(arg)
        #self.curveencoder_detail = FeatCurveEncoder(arg)
        self.curveencoder_tile = FeatCurveEncoder(arg, tile=True)

        ######## BASE ###########################
        if self.rho_grid:
            dim_in_base = 1 + 2* self.grid_outdim
        else:
            dim_in_base = 1 + self.grid_outdim

        dim_out = arg['dim_sample_feat']
        dim_cond = arg['dim_curve_feat']

        self.filmenc_base1 = FiLMEncoder(dim_in_base, dim_out, dim_cond) # arg['dim_sample_feat'], arg[''])
        self.filmenc_base2 = FiLMEncoder(dim_out, dim_out, dim_cond)
    
        self.decoder_base = MLP(**arg['decoder_curve'])

        ############# DETAIL ################
        dim_in_detail = (arg['num_pos_encoding_detail']* 2 + 1 )*3 + (arg['num_period_encoding'] * 2 )*1 + self.grid_outdim
        dim_out = arg['dim_sample_feat']
        dim_cond = arg['dim_curve_feat']

        self.filmenc_detail1 = FiLMEncoder(dim_in_detail, dim_out, dim_cond) # arg['dim_sample_feat'], arg[''])
        self.filmenc_detail2 = FiLMEncoder(dim_out, dim_out, dim_cond)
        self.filmenc_detail3 = FiLMEncoder(dim_out, dim_out, dim_cond)
    
        self.decoder_detail = MLP(**arg['decoder_curve'])


        ############# TILE ################
        dim_in_tile = (arg['num_pos_encoding_tile']* 2 )*3 + (arg['num_period_encoding'] * 2 )*1
        dim_out = arg['dim_sample_feat']
        dim_cond = arg['dim_curve_feat']


    def init_embedding(self, embedding, dim_code):
        nn.init.normal_(embedding.weight.data, 0., 1./ math.sqrt(dim_code))

    
    def set_post_mode(self):
        for pname, param in self.core.named_parameters():
            if 'embd' in pname:
                continue
            param.requires_grad = False


    def forwardsimple(self, model_input, curr_epoch=0, istrain=True):
        mi = model_input
        # curve_idx:(Nb, Ns); coords:(Nb, Ns); samples(Nb,Ns,3)
        curve_code = self.embd(mi['curve_idx'])
        B = curve_code.shape[0]

        type_curve  = self.type_embd(torch.zeros(B, dtype=torch.long, device=self.device)).unsqueeze(1)
        type_curve_tile  = self.type_embd(2*torch.ones(B, dtype=torch.long, device=self.device)).unsqueeze(1)
        type_sample = self.type_embd(torch.ones(B, dtype=torch.long, device=self.device)).unsqueeze(1)

        #if hasattr(self, 'pos_enc_sample'):
        if istrain:
            #if self.issplit:
            #samples_posenc_base = self.pos_enc_sample_base(mi['samples'])
            samples_base_ctfeatures = self.grid_curvetheta(mi['coords'], mi['angles'])
            samples_base_features = torch.cat([mi['rho'].unsqueeze(-1), samples_base_ctfeatures], dim=-1)
            if self.rho_grid:
                samples_base_crfeatures = self.grid_curverho(mi['coords'], mi['rho_n'])
                samples_base_features = torch.cat([mi['rho'].unsqueeze(-1), samples_base_ctfeatures, samples_base_crfeatures], dim=-1)
            samples_posenc_detail = self.pos_enc_sample_detail(mi['samples'])

            sample_angle_periodenc = self.period_enc(mi['angles'].unsqueeze(-1))
            curve_coords_posenc = self.pos_enc_curve(mi['coords'].unsqueeze(-1))
            radii_y_posenc = self.pos_enc_y(torch.log(mi['radius'][:,:,0]).unsqueeze(-1))
            radii_z_posenc = self.pos_enc_z(torch.log(mi['radius'][:,:,1]).unsqueeze(-1))
        else:
            #if self.issplit:
            #samples_posenc_base = self.pos_enc_sample_base.inference(mi['samples'])
            samples_base_ctfeatures = self.grid_curvetheta.inference(mi['coords'], mi['angles'])
            if self.rho_grid:
                samples_base_crfeatures = self.grid_curverho.inference(mi['coords'], mi['rho_n'])
                #samples_base_ctfeatures_base = self.grid_curvetheta.inference_stretch(mi['coords'], mi['angles'], start=0, end=1)
                #samples_base_ctfeatures_detail = self.grid_curvetheta.inference_stretch(mi['coords'], mi['angles'], start=2, end=3)
                #samples_base_ctfeatures = torch.cat([samples_base_ctfeatures_base, samples_base_ctfeatures_detail], dim=-1)
                samples_base_features = torch.cat([mi['rho'].unsqueeze(-1), samples_base_ctfeatures, samples_base_crfeatures], dim=-1)
            else:
                samples_base_features = torch.cat([mi['rho'].unsqueeze(-1), samples_base_ctfeatures], dim=-1)

            samples_posenc_detail = self.pos_enc_sample_detail.inference(mi['samples'])
            sample_angle_periodenc = self.period_enc.inference(mi['angles'].unsqueeze(-1))
            curve_coords_posenc = self.pos_enc_curve.inference(mi['coords'].unsqueeze(-1))
            radii_y_posenc = self.pos_enc_y.inference(torch.log(mi['radius'][:,:,0]).unsqueeze(-1))
            radii_z_posenc = self.pos_enc_z.inference(torch.log(mi['radius'][:,:,1]).unsqueeze(-1))

        #print(curve_code.shape)
        #print(curve_coords_posenc.shape)
        #print(radii_y_posenc.shape)
        #print(radii_z_posenc.shape)
        
        curve_code_coords = torch.cat([curve_code, curve_coords_posenc, radii_y_posenc, radii_z_posenc], dim=-1)
        curve_feats = self.curveencoder(curve_code_coords) 
        curve_feats = curve_feats + type_curve

        ############ BASE ################
        #x = self.filmenc_base1(samples_posenc_base, curve_feats) + type_sample
        x = self.filmenc_base1(samples_base_features, curve_feats) + type_sample
        res = x
        x = self.filmenc_base2(x, curve_feats)
        x = x + res
        curve_sdf_base = self.decoder_base.forward_simple(x).squeeze(-1)
        #curve_sdf = self.decoder.forward_simple(curve_feats).squeeze(-1)
        ############ DETAILS ################
        y = torch.cat([samples_posenc_detail, sample_angle_periodenc, samples_base_ctfeatures.detach()], dim=-1)
        x = self.filmenc_detail1(y, curve_feats) + type_sample
        res = x
        x = self.filmenc_detail2(x, curve_feats)
        x = x + res
        res = x
        x = self.filmenc_detail3(x, curve_feats)
        x = x + res
        curve_sdf_detail = self.decoder_detail.forward_simple(x).squeeze(-1)
        ############ Final ###########

        start_epoch = 500
        linear_epoch = 2000
        if istrain:
            if curr_epoch < start_epoch:
                alpha = 0
            elif curr_epoch < linear_epoch:
                alpha = float(curr_epoch - start_epoch)/float(linear_epoch - start_epoch) #  % 1000)/1000
            else:
                alpha = 1

            if self.gate == "rho":
                gate = torch.exp(-((mi['rho'] - 0.5) ** 2) / (2 * (0.15 ** 2)))
            else:
                gate = torch.exp(-(curve_sdf_base**2)/(2*(0.05**2)))

            curve_sdf = curve_sdf_base + alpha * gate * curve_sdf_detail
        else:
                #curve_sdf = curve_sdf_base + curve_sdf_detail
                if self.details_level == 0: 
                    #print("details 0")
                    curve_sdf = curve_sdf_base
                elif self.details_level == 1: 
                    curve_sdf = curve_sdf_detail
                else:
                    if self.gate == "rho":
                        gate = torch.exp(-((mi['rho'] - 0.5) ** 2) / (2 * (0.15 ** 2)))
                    else:
                        gate = torch.exp(-(curve_sdf_base**2)/(2*(0.05**2)))
                    #gate = torch.exp(-((mi['rho'] - 0.5) ** 2) / (2 * (0.15 ** 2)))
                    curve_sdf = curve_sdf_base +  gate * curve_sdf_detail
                    #curve_sdf = curve_sdf_base + curve_sdf_detail


        return {
            'sdf': curve_sdf,
            'code': curve_code,
        }

    def forwardstretch(self, model_input):
        #print("in forwqrd stretch")
        #exit()
        mi = model_input
        # curve_idx:(Nb, Ns); coords:(Nb, Ns); samples(Nb,Ns,3)
        curve_code = self.embd(mi['curve_idx'])
        B = curve_code.shape[0]

        type_curve  = self.type_embd(torch.zeros(B, dtype=torch.long, device=self.device)).unsqueeze(1)
        type_curve_tile  = self.type_embd(2*torch.ones(B, dtype=torch.long, device=self.device)).unsqueeze(1)
        type_sample = self.type_embd(torch.ones(B, dtype=torch.long, device=self.device)).unsqueeze(1)

        samples_base_ctfeatures_base = self.grid_curvetheta.inference_stretch(mi['coords'], mi['angles'], start=0, end=1)
        #samples_base_ctfeatures_detail = self.grid_curvetheta.inference_stretch(mi['coords'], mi['angles'], start=0, end=0)
        samples_base_ctfeatures_detail = self.grid_curvetheta.inference_stretch(mi['coords_detail'], mi['angles'], start=2, end=3)
        #print(samples_base_ctfeatures_base.shape)
        #print(samples_base_ctfeatures_detail.shape)
        #samples_base_features = torch.cat([mi['rho'].unsqueeze(-1), samples_base_ctfeatures_base], dim=-1)
        if self.rho_grid:
            samples_base_crfeatures = self.grid_curverho.inference(mi['coords'], mi['rho_n'])
            #samples_base_features = torch.cat([mi['rho'].unsqueeze(-1), samples_base_ctfeatures, samples_base_crfeatures], dim=-1)
            samples_base_ctfeatures = torch.cat([samples_base_ctfeatures_base, samples_base_ctfeatures_detail], dim=-1)
            #samples_base_features = torch.cat([mi['rho'].unsqueeze(-1), samples_base_ctfeatures_base, samples_base_ctfeatures_base, samples_base_crfeatures], dim=-1)
            samples_base_features = torch.cat([mi['rho'].unsqueeze(-1), samples_base_ctfeatures, samples_base_crfeatures], dim=-1)

        #samples_posenc_detail = self.pos_enc_sample_detail.inference(mi['samples_detail'])
        sample_angle_periodenc = self.period_enc.inference(mi['angles'].unsqueeze(-1))
        curve_coords_posenc = self.pos_enc_curve.inference(mi['coords'].unsqueeze(-1))

        radii_y_posenc = self.pos_enc_y.inference(torch.log(mi['radius'][:,:,0]).unsqueeze(-1))
        radii_z_posenc = self.pos_enc_z.inference(torch.log(mi['radius'][:,:,1]).unsqueeze(-1))

        curve_code_coords = torch.cat([curve_code, curve_coords_posenc, radii_y_posenc, radii_z_posenc], dim=-1)
        curve_feats = self.curveencoder(curve_code_coords) 
        curve_feats = curve_feats + type_curve

        ############ BASE ################
        #x = self.filmenc_base1(samples_posenc_base, curve_feats) + type_sample
        x = self.filmenc_base1(samples_base_features, curve_feats) + type_sample
        res = x
        x = self.filmenc_base2(x, curve_feats)
        x = x + res
        curve_sdf_base = self.decoder_base.forward_simple(x).squeeze(-1)
        return {
            'sdf': curve_sdf_base,
            'code': curve_code,
        }
        #curve_sdf = self.decoder.forward_simple(curve_feats).squeeze(-1)
        ############ DETAILS ################
        y = torch.cat([samples_posenc_detail, sample_angle_periodenc, samples_base_ctfeatures.detach()], dim=-1)
        x = self.filmenc_detail1(y, curve_feats) + type_sample
        res = x
        x = self.filmenc_detail2(x, curve_feats)
        x = x + res
        res = x
        x = self.filmenc_detail3(x, curve_feats)
        x = x + res
        curve_sdf_detail = self.decoder_detail.forward_simple(x).squeeze(-1)
        ############ Final ###########
        if self.details_level == 0: 
            #print("details 0")
            curve_sdf = curve_sdf_base
        elif self.details_level == 1: 
            curve_sdf = curve_sdf_detail
        else:
            gate = torch.exp(-(curve_sdf_base**2)/(2*(0.03**2)))
            curve_sdf = curve_sdf_base +  gate * curve_sdf_detail

        return {
            'sdf': curve_sdf,
            'code': curve_code,
        }

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
        return out['sdf']

    @torch.no_grad()
    def mix_curve(self, model_input):
        # assume only one curve
        mi = model_input
        curve_data = self.pack_data(mi)



        curve_idx = curve_data['curve_idx']
        curve_code = self.embd(curve_idx)
        new_idx = mi['new_idx']* torch.ones_like(curve_idx, dtype=int).to(curve_code.device)
        new_code = self.embd(new_idx)

        curve_feats = self.encoder.mix_feats(
            curve_code, new_code, curve_data['coords'], mi)
        
        if hasattr(self, 'pos_enc'):
            samples = self.pos_enc(curve_data['samples'])
        else:
            samples = curve_data['samples']
        curve_feats = torch.cat([curve_feats, samples], dim=-1)
        curve_sdf = self.decoder.forward_simple(curve_feats).squeeze(-1)
        return curve_sdf


    def pack_data(self, model_input):
        mi = model_input
        device = mi['device']

        n_sample = mi['coords'].shape[0]
        curve_idx = mi['curve_idx']*torch.ones(n_sample, dtype=int)
        
        res = {
            'samples': torch.from_numpy(mi['samples_local']).float().to(device),
            'coords': torch.from_numpy(mi['coords']).float().to(device),
            #'rho': torch.from_numpy(mi['rho']).float().to(device),
            'angles': torch.from_numpy(mi['angles']).float().to(device),
            'radius': torch.from_numpy(mi['radius']).float().to(device),
            'rho': torch.from_numpy(mi['rho']).float().to(device),
            'rho_n': torch.from_numpy(mi['rho_n']).float().to(device),
            'curve_idx': curve_idx.to(device)
        }

        res = {key:val.unsqueeze(0) for key,val in res.items()}
        return res

    def pack_data_stretch(self, model_input):
        mi = model_input
        device = mi['device']

        n_sample = mi['coords'].shape[0]
        curve_idx = mi['curve_idx']*torch.ones(n_sample, dtype=int)
        
        res = {
            'samples': torch.from_numpy(mi['samples_local']).float().to(device),
            #'seam': torch.from_numpy(mi['seam']).float().to(device),
            'samples_detail': torch.from_numpy(mi['samples_detail']).float().to(device),
            #'samples_detail_notile': torch.from_numpy(mi['samples_detail_notile']).float().to(device),
            'coords': torch.from_numpy(mi['coords']).float().to(device),
            'coords_detail': torch.from_numpy(mi['coords_detail']).float().to(device),
            #'rho': torch.from_numpy(mi['rho']).float().to(device),
            'angles': torch.from_numpy(mi['angles']).float().to(device),
            'radius': torch.from_numpy(mi['radius']).float().to(device),
            'rho': torch.from_numpy(mi['rho']).float().to(device),
            'rho_n': torch.from_numpy(mi['rho_n']).float().to(device),
            #'radius_detail': torch.from_numpy(mi['radius_detail']).float().to(device),
            'curve_idx': curve_idx.to(device)
        }

        res = {key:val.unsqueeze(0) for key,val in res.items()}
        return res


class FiLMEncoder(nn.Module):
    """docstring for FiLMEncoder."""
    def __init__(self, dim, out_dim, cond_dim):
        super(FiLMEncoder, self).__init__()
        #self.fc = spectral_norm(nn.Linear(dim, out_dim))
        self.fc = nn.Linear(dim, out_dim)
        self.film = FiLM(cond_dim, out_dim)
        self.act = nn.SiLU()
    
    def forward(self, x, cond):
        # code(Nb, Ns, N_code); coords(Nb, Ns)
        x = self.fc(x)
        x = self.film(x, cond)
        return self.act(x)

    @torch.no_grad()
    def inference(self, x, cond):
        x = self.fc(x)
        x = self.film(x, cond)
        return self.act(x)

class FiLM(nn.Module):
    def __init__(self, cond_dim, out_dim):
        super(FiLM, self).__init__()
        #self.dim_curve_feat = arg['dim_curve_feat']
        #self.dim_sample_feat = arg['dim_sample_feat']
        #self.gamma = nn.Linear(self.dim_curve_feat, self.dim_sample_feat)
        #self.beta = nn.Linear(self.dim_curve_feat, self.dim_sample_feat)

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
        #return sample_feat + (gamma * sample_feat + beta)
        return gamma * sample_feat + beta
    
    @torch.no_grad()
    def inference(self, sample_feat, curve_feat):
        gamma = self.gamma(curve_feat)
        beta = self.beta(curve_feat)
        #return sample_feat + (gamma * sample_feat + beta)
        return gamma * sample_feat + beta

class FeatSampleEncoder(nn.Module):
    """docstring for FeatCurveEncoder."""
    def __init__(self, dim, out_dim):
        super(FeatSampleEncoder, self).__init__()
        self.fc = nn.Linear(dim, out_dim)
        self.act = nn.SiLU()
    
    def forward(self, x):
        # code(Nb, Ns, N_code); coords(Nb, Ns)
        return self.act(self.fc(x))

    @torch.no_grad()
    def inference(self, code_coords):
        return self.act(self.fc(x))

class FeatCurveEncoder(nn.Module):
    """docstring for FeatCurveEncoder."""
    def __init__(self, arg, tile=False):
        super(FeatCurveEncoder, self).__init__()
        if tile:
            t_posenc = (arg['num_pos_encoding'] * 2 + 1 ) + (arg['num_pos_encoding']*2)
        else:
            t_posenc = arg['num_pos_encoding'] * 2 + 1
        radii_y_posenc = arg['num_y_pos_encoding'] * 2 + 1
        radii_z_posenc = arg['num_z_pos_encoding'] * 2 + 1
        diff =  arg['dim_code'] + t_posenc + radii_y_posenc + radii_z_posenc
        arg['encoder_curve']['size'].insert(0, diff)
        self.mlp_t = MLP(**arg['encoder_curve'])
    
    def forward(self, code_coords):
        # code(Nb, Ns, N_code); coords(Nb, Ns)
        return self.mlp_t.forward_simple(code_coords)

    @torch.no_grad()
    def inference(self, code_coords):
        return self.mlp_t.forward_simple(code_coords)

    @torch.no_grad() 
    def mix_feats(self, code_coords, arg):
        func1 = arg['mix_func1']
        func2 = arg['mix_func2']
        ts1, weights1 = func1(code_coords)
        ts2, weights2 = func2(code_coords)

        #feat = self.mlp_t.forward_simple(code_coords)
        #feat2 = self.forward(code2, ts2)
        #feat = feat1*weights1[...,None] + feat2*weights2[...,None]
        return feat, ts1, ts2, weights1, weights2

class FeatCurveEncoder_old(nn.Module):
    """docstring for FeatCurveEncoder."""
    def __init__(self, arg):
        super(FeatCurveEncoder, self).__init__()
        self.dim_code = arg['dim_code']
        self.dim_feat = arg['dim_feat']

        self.linear1 = nn.Linear(self.dim_code, 2*self.dim_feat)
        self.mlp = MLP(**arg['encoder_curve'])
    
    def forward(self, code, coords):
        # code(Nb, Ns, N_code); coords(Nb, Ns)
        Nb, Ns = code.shape[0], code.shape[1]
        end_feats = self.linear1(code).view(Nb, Ns, self.dim_feat, 2)
        # weights (Nb, Ns, 2)
        weights = torch.stack([coords, 1-coords], dim=-1)
        feat_curve = torch.einsum('bnjk,bnk->bnj', end_feats, weights)
        return self.mlp.forward_simple(feat_curve)
   
    def mix_feats(self, code1, code2, coords, arg):
        func1 = arg['mix_func1']
        func2 = arg['mix_func2']

        ts1, weights1 = func1(coords)
        ts2, weights2 = func2(coords)
        feat1 = self.forward(code1, ts1)
        feat2 = self.forward(code2, ts2)
        feat = feat1*weights1[...,None] + feat2*weights2[...,None]
        return feat



if __name__ == "__main__":
    pass
