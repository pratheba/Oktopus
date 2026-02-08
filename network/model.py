import math
import torch
import torch.nn as nn

from mlp import *
#from pos_encoding import PosEncoding
from pos_encoding_fromokto import PosEncoding
from periodic_encoding import PeriodicEncoding
from mask import *


class NGCNet(nn.Module):
    """docstring for NGCNet ."""
    def __init__(self, arg):
        super().__init__()
        self.device = arg['device']
        self.dim_code = arg['dim_code']
        self.dim_feat = arg['dim_feat']
        self.dim_type = arg['dim_type']

        self.issplit = arg['split']
        if self.issplit:
            self.details_level = arg['details_level']
            self.pos_enc_sample_base = PosEncoding(arg['num_pos_encoding_base'], d_in=3)
            self.pos_enc_sample_detail = PosEncoding(arg['num_pos_encoding_detail'], start=4, d_in=3)
        else:
            self.num_film_layers = arg['num_film_layers']
            self.pos_enc_sample = PosEncoding(arg['num_pos_encoding'], d_in=3)

        self.pos_enc_curve = PosEncoding(arg['num_pos_encoding'], d_in=1)
        self.period_enc = PeriodicEncoding(arg['num_period_encoding'])

        self.embd = nn.Embedding(arg['n_curve'], self.dim_code)
        self.type_embd = nn.Embedding(2, self.dim_type)
        self.init_embedding(self.embd, self.dim_code)
        self.init_embedding(self.type_embd, self.dim_type)
        
        self.curveencoder = FeatCurveEncoder(arg)

        dim_sample_in = (arg['num_pos_encoding']* 2 +1)*3 + (arg['num_period_encoding'] * 2 + 1)*1
        #self.sampleencoder1 = FeatSampleEncoder(dim_sample_in , arg['dim_sample_feat'])
        #self.sampleencoder2 = FeatSampleEncoder(arg['dim_sample_feat'], arg['dim_sample_feat'])

        #self.film1 = FiLM(arg)
        #self.film2 = FiLM(arg)

        #self.dim_curve_feat = arg['dim_curve_feat']
        #self.dim_sample_feat = arg['dim_sample_feat']
        #self.gamma = nn.Linear(self.dim_curve_feat, self.dim_sample_feat)
        #self.beta = nn.Linear(self.dim_curve_feat, self.dim_sample_feat)
        #self.decoder = MLP(**arg['decoder_curve'])
        
        if self.issplit:
            ######## BASE ###########################
            dim_in_base = (arg['num_pos_encoding_base']* 2 +1)*3 #+ (arg['num_period_encoding'] * 2 + 1)*1
            dim_out = arg['dim_sample_feat']
            dim_cond = arg['dim_curve_feat']

            self.filmenc_base1 = FiLMEncoder(dim_in_base, dim_out, dim_cond) # arg['dim_sample_feat'], arg[''])
            self.filmenc_base2 = FiLMEncoder(dim_out, dim_out, dim_cond)
        
            self.decoder_base = MLP(**arg['decoder_curve'])

            ############# DETAIL ################

            dim_in_detail = (arg['num_pos_encoding_detail']* 2 +1)*3 + (arg['num_period_encoding'] * 2 + 1)*1
            dim_out = arg['dim_sample_feat']
            dim_cond = arg['dim_curve_feat']

            self.filmenc_detail1 = FiLMEncoder(dim_in_detail, dim_out, dim_cond) # arg['dim_sample_feat'], arg[''])
            self.filmenc_detail2 = FiLMEncoder(dim_out, dim_out, dim_cond)
            self.filmenc_detail3 = FiLMEncoder(dim_out, dim_out, dim_cond)
        
            self.decoder_detail = MLP(**arg['decoder_curve'])
        else:
            dim_in = (arg['num_pos_encoding']* 2 +1)*3 + (arg['num_period_encoding'] * 2 + 1)*1
            dim_out = arg['dim_sample_feat']
            dim_cond = arg['dim_curve_feat']

            self.filmenc1 = FiLMEncoder(dim_in, dim_out, dim_cond) # arg['dim_sample_feat'], arg[''])
            self.filmenc2 = FiLMEncoder(dim_out, dim_out, dim_cond)
            if self.num_film_layers == 3:
                self.filmenc3 = FiLMEncoder(dim_out, dim_out, dim_cond)
        
            self.decoder = MLP(**arg['decoder_curve'])


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
        type_sample = self.type_embd(torch.ones(B, dtype=torch.long, device=self.device)).unsqueeze(1)

        #if hasattr(self, 'pos_enc_sample'):
        if istrain:
            if self.issplit:
                samples_posenc_base = self.pos_enc_sample_base(mi['samples'])
                samples_posenc_detail = self.pos_enc_sample_detail(mi['samples'])
            else:
                samples_posenc = self.pos_enc_sample(mi['samples'])

            sample_angle_periodenc = self.period_enc(mi['angles'].unsqueeze(-1))
            curve_coords_posenc = self.pos_enc_curve(mi['coords'].unsqueeze(-1))
            radii_y_posenc = self.pos_enc_curve(torch.log(mi['radius'][:,:,0]).unsqueeze(-1))
            radii_z_posenc = self.pos_enc_curve(torch.log(mi['radius'][:,:,1]).unsqueeze(-1))
        else:
            if self.issplit:
                samples_posenc_base = self.pos_enc_sample_base.inference(mi['samples'])
                samples_posenc_detail = self.pos_enc_sample_detail.inference(mi['samples'])
            else:
                samples_posenc = self.pos_enc_sample.inference(mi['samples'])

            sample_angle_periodenc = self.period_enc.inference(mi['angles'].unsqueeze(-1))
            curve_coords_posenc = self.pos_enc_curve.inference(mi['coords'].unsqueeze(-1))
            radii_y_posenc = self.pos_enc_curve.inference(torch.log(mi['radius'][:,:,0]).unsqueeze(-1))
            radii_z_posenc = self.pos_enc_curve.inference(torch.log(mi['radius'][:,:,1]).unsqueeze(-1))

        curve_code_coords = torch.cat([curve_code, curve_coords_posenc, radii_y_posenc, radii_z_posenc], dim=-1)
        curve_feats = self.curveencoder(curve_code_coords) 
        curve_feats = curve_feats + type_curve


        ########## ALL #####################
        if not self.issplit:
            y = torch.cat([samples_posenc, sample_angle_periodenc], dim=-1)
            x = self.filmenc1(y, curve_feats) + type_sample
            res = x
            x = self.filmenc2(x, curve_feats)
            x = x + res
            if self.num_film_layers == 3:
                res = x
                x = self.filmenc3(x, curve_feats)
                x = x + res
            curve_sdf = self.decoder.forward_simple(x).squeeze(-1)
        else:
            ############ BASE ################
            x = self.filmenc_base1(samples_posenc_base, curve_feats) + type_sample
            res = x
            x = self.filmenc_base2(x, curve_feats)
            x = x + res
            curve_sdf_base = self.decoder_base.forward_simple(x).squeeze(-1)
            #curve_sdf = self.decoder.forward_simple(curve_feats).squeeze(-1)
            ############ DETAILS ################

            y = torch.cat([samples_posenc_detail, sample_angle_periodenc], dim=-1)
            x = self.filmenc_detail1(y, curve_feats) + type_sample
            res = x
            x = self.filmenc_detail2(x, curve_feats)
            x = x + res
            res = x
            x = self.filmenc_detail3(x, curve_feats)
            x = x + res
            curve_sdf_detail = self.decoder_detail.forward_simple(x).squeeze(-1)
            ############ Final ###########

            if istrain:
                if curr_epoch < 200:
                    curve_sdf = curve_sdf_base
                elif curr_epoch < 1000:
                    alpha = (curr_epoch % 1000)/1000
                    curve_sdf = curve_sdf_base + alpha * curve_sdf_detail
                else:
                    curve_sdf = curve_sdf_base + curve_sdf_detail
            else:
                    #curve_sdf = curve_sdf_base + curve_sdf_detail
                    if self.details_level == 0: 
                        curve_sdf = curve_sdf_base
                    elif self.details_level == 1: 
                        curve_sdf = curve_sdf_detail
                    else:
                        curve_sdf = curve_sdf_base + curve_sdf_detail


        return {
            'sdf': curve_sdf,
            'code': curve_code,
        }

    def forwardstretch(self, model_input):
        mi = model_input
        # curve_idx:(Nb, Ns); coords:(Nb, Ns); samples(Nb,Ns,3)
        curve_code = self.embd(mi['curve_idx'])
        B = curve_code.shape[0]

        type_curve  = self.type_embd(torch.zeros(B, dtype=torch.long, device=self.device)).unsqueeze(1)
        type_sample = self.type_embd(torch.ones(B, dtype=torch.long, device=self.device)).unsqueeze(1)
        m = make_detail_mask(mi['coords'], 0.0, 0.2, eps=0.03)
        #print(m)

        #if hasattr(self, 'pos_enc_sample'):
        samples_posenc_base = self.pos_enc_sample_base.inference(mi['samples'])
        #radii_z_posenc_detail = self.pos_enc_curve.inference(torch.log(mi['radius_detail'][:,:,1]).unsqueeze(-1))
        #x_period = torch.sin(2*torch.pi*k*coords.squeeze(-1) if coords.ndim==3 else 2*torch.pi*k*coords)

        #samples_detail = mi['samples'].clone()
        #vx_stretch = 2.0 * mi['coords'] - 1.0
        #vx_tile = 2.0 * (torch.frac(2 * mi['coords'])) - 1.0   
        #samples_detail[..., 0] = mi['samples'][..., 0] - vx_stretch #+ vx_tile
        #print(samples_detail.shape)
        #samples_detail[..., 0] = 0.0
        #print(samples_detail.shape)
        m1 = (m * mi['seam']).unsqueeze(-1)  
        #x_used    = (1.0 - m1) * mi['samples_detail_notile'][...,0:1] + m1 * mi['samples_detail'][...,0:1]
        #samples_detail = mi['samples'].clone()

        #samples_detail[...,0:1] = x_used

        samples_posenc_detail = self.pos_enc_sample_detail.inference(mi['samples_detail'])
        #samples_posenc_detail = self.pos_enc_sample_detail.inference(samples_detail)

        sample_angle_periodenc = self.period_enc.inference(mi['angles'].unsqueeze(-1))

        curve_coords_posenc = self.pos_enc_curve.inference(mi['coords'].unsqueeze(-1))
        curve_coords_posenc_detail = self.pos_enc_curve.inference(mi['coords_detail'].unsqueeze(-1))
        #curve_coords_posenc_detail = self.pos_enc_curve.inference((2*mi['coords']).unsqueeze(-1))
        curve_coords_used_posenc   = (1.0 - m).unsqueeze(-1) * curve_coords_posenc + m.unsqueeze(-1) * curve_coords_posenc_detail
        #curve_coords_used_posenc   = (1.0 - m1).unsqueeze(-1) * curve_coords_posenc + m1.unsqueeze(-1) * curve_coords_posenc_detail

        radii_y_posenc_base = self.pos_enc_curve.inference(torch.log(mi['radius'][:,:,0]).unsqueeze(-1))
        radii_z_posenc_base = self.pos_enc_curve.inference(torch.log(mi['radius'][:,:,1]).unsqueeze(-1))
        radii_y_posenc_detail = self.pos_enc_curve.inference(torch.log(mi['radius_detail'][:,:,0]).unsqueeze(-1))
        radii_z_posenc_detail = self.pos_enc_curve.inference(torch.log(mi['radius_detail'][:,:,1]).unsqueeze(-1))
        #radii_y_posenc_detail = self.pos_enc_curve.inference((torch.log(mi['radius_detail'][:,:,0]) - torch.log(mi['radius'][:,:,0])).unsqueeze(-1))
        #radii_z_posenc_detail = self.pos_enc_curve.inference((torch.log(mi['radius_detail'][:,:,1]) - torch.log(mi['radius'][:,:,1])).unsqueeze(-1))

        curve_code_coords = torch.cat([curve_code, curve_coords_posenc, radii_y_posenc_base, radii_z_posenc_base], dim=-1)
        curve_feats = self.curveencoder.inference(curve_code_coords) 
        curve_feats = curve_feats + type_curve

        #curve_code_coords_detail = torch.cat([curve_code, curve_coords_posenc_detail, radii_y_posenc_detail, radii_z_posenc_detail], dim=-1)
        curve_code_coords_detail = torch.cat([curve_code, curve_coords_used_posenc, radii_y_posenc_detail, radii_z_posenc_detail], dim=-1)
        #curve_code_coords_detail = torch.cat([curve_code, curve_coords_used_posenc, radii_y_posenc_detail, radii_z_posenc_detail], dim=-1)
        #curve_code_coords_detail = torch.cat([curve_code, curve_coords_posenc_detail, radii_y_posenc_base, radii_z_posenc_base], dim=-1)
        #curve_code_coords_detail = torch.cat([curve_code, curve_coords_posenc_detail, radii_y_posenc, radii_z_posenc_], dim=-1)
        curve_feats_detail = self.curveencoder.inference(curve_code_coords_detail) 
        curve_feats_detail = curve_feats_detail + type_curve


        ############ BASE ################
        x = self.filmenc_base1.inference(samples_posenc_base, curve_feats) + type_sample
        res = x
        x = self.filmenc_base2.inference(x, curve_feats)
        x = x + res
        curve_sdf_base = self.decoder_base.forward_simple(x).squeeze(-1)
        #curve_sdf = self.decoder.forward_simple(curve_feats).squeeze(-1)
        ############ DETAILS ################

        y = torch.cat([samples_posenc_detail, sample_angle_periodenc], dim=-1)
        x = self.filmenc_detail1.inference(y, curve_feats_detail) + type_sample
        #x = self.filmenc_detail1(y, curve_feats) + type_sample
        res = x
        x = self.filmenc_detail2.inference(x, curve_feats_detail)
        #x = self.filmenc_detail2(x, curve_feats)
        x = x + res
        res = x
        x = self.filmenc_detail3.inference(x, curve_feats_detail)
        #x = self.filmenc_detail3(x, curve_feats)
        x = x + res
        curve_sdf_detail = self.decoder_detail.forward_simple(x).squeeze(-1)
        ############ Final ###########
        
        curve_sdf = curve_sdf_base + curve_sdf_detail
        #curve_sdf = 0.8*curve_sdf_base + m * curve_sdf_detail
        #curve_sdf = curve_sdf_base


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
            'seam': torch.from_numpy(mi['seam']).float().to(device),
            'samples_detail': torch.from_numpy(mi['samples_detail']).float().to(device),
            'samples_detail_notile': torch.from_numpy(mi['samples_detail_notile']).float().to(device),
            'coords': torch.from_numpy(mi['coords']).float().to(device),
            'coords_detail': torch.from_numpy(mi['coords_detail']).float().to(device),
            #'rho': torch.from_numpy(mi['rho']).float().to(device),
            'angles': torch.from_numpy(mi['angles']).float().to(device),
            'radius': torch.from_numpy(mi['radius']).float().to(device),
            'radius_detail': torch.from_numpy(mi['radius_detail']).float().to(device),
            'curve_idx': curve_idx.to(device)
        }

        res = {key:val.unsqueeze(0) for key,val in res.items()}
        return res


class FiLMEncoder(nn.Module):
    """docstring for FiLMEncoder."""
    def __init__(self, dim, out_dim, cond_dim):
        super(FiLMEncoder, self).__init__()
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
    def __init__(self, arg):
        super(FeatCurveEncoder, self).__init__()

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

   
    def mix_feats(self, code1, code2, coords, arg):
        func1 = arg['mix_func1']
        func2 = arg['mix_func2']

        ts1, weights1 = func1(coords)
        ts2, weights2 = func2(coords)
        feat1 = self.forward(code1, ts1)
        feat2 = self.forward(code2, ts2)
        feat = feat1*weights1[...,None] + feat2*weights2[...,None]
        return feat

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


class DeepSDF(nn.Module):
    """docstring for DeepSDF."""
    def __init__(self, arg):
        super(DeepSDF, self).__init__()
        self.dim_code = arg['dim_code']
        self.dim_feat = arg['dim_feat']
        
        self.embd = nn.Embedding(arg['n_shape'], self.dim_code)
        self.init_embedding(self.embd, self.dim_code)
        if arg['num_pos_encoding'] > 0:
            self.pos_enc = PosEncoding(arg['num_pos_encoding'])
            diff = self.pos_enc.d_out - self.pos_enc.d_in
            arg['decoder']['size'][0] += diff
        self.decoder = MLP(**arg['decoder'])

    def init_embedding(self, embedding, dim_code):
        nn.init.normal_(embedding.weight.data, 0., 1./ math.sqrt(dim_code))


    def forward(self, model_input):
        mi = model_input
        idx = mi['idx']
        # codes: (Nb, 256)
        codes = self.embd(idx)

        if hasattr(self, 'pos_enc'):
            samples = self.pos_enc(mi['samples'])
        else:
            samples = mi['samples']

        # samples: (Nb, Ns, 3)
        Ns = samples.shape[1]
        codes = codes.repeat(1, Ns, 1)
        feats = torch.cat([codes, samples], dim=-1)
        sdfs = self.decoder.forward_simple(feats).squeeze(-1)
        return {
            'sdf': sdfs,
            'code': codes,
        }

    def inference(self, model_input):
        mi = model_input
        idx = mi['idx']
        codes = self.embd(idx)

        if hasattr(self, 'pos_enc'):
            samples = self.pos_enc(mi['samples'])
        else:
            samples = mi['samples']

        # samples: (Nb, Ns, 3)
        Ns = samples.shape[1]
        codes = codes[:,None,:].repeat(1, Ns, 1)
        feats = torch.cat([codes, samples], dim=-1)
        sdfs = self.decoder.forward_simple(feats).squeeze(-1)
        return sdfs


if __name__ == "__main__":
    pass
