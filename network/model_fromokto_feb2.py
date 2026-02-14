import math
import torch
import torch.nn as nn

from mlp import *
#from pos_encoding import PosEncoding
from pos_encoding_fromokto import PosEncoding
from periodic_encoding import PeriodicEncoding


class NGCNet(nn.Module):
    """docstring for NGCNet ."""
    def __init__(self, arg):
        super().__init__()
        self.device = arg['device']
        self.dim_code = arg['dim_code']
        self.dim_feat = arg['dim_feat']
        self.dim_type = arg['dim_type']

        self.pos_enc_sample = PosEncoding(arg['num_pos_encoding'], d_in=3)
        self.pos_enc_curve = PosEncoding(arg['num_pos_encoding'], d_in=1)
        self.period_enc = PeriodicEncoding(arg['num_period_encoding'])

        self.embd = nn.Embedding(arg['n_curve'], self.dim_code)
        self.type_embd = nn.Embedding(2, self.dim_type)
        self.init_embedding(self.embd, self.dim_code)
        self.init_embedding(self.type_embd, self.dim_type)
        
        self.curveencoder = FeatCurveEncoder(arg)

        dim_sample_in = (arg['num_pos_encoding']* 2 +1)*3 + (arg['num_period_encoding'] * 2 + 1)*1
        self.sampleencoder1 = FeatSampleEncoder(dim_sample_in , arg['dim_sample_feat'])
        self.sampleencoder2 = FeatSampleEncoder(arg['dim_sample_feat'], arg['dim_sample_feat'])

        self.film1 = FiLM(arg)
        self.film2 = FiLM(arg)

        #self.dim_curve_feat = arg['dim_curve_feat']
        #self.dim_sample_feat = arg['dim_sample_feat']
        #self.gamma = nn.Linear(self.dim_curve_feat, self.dim_sample_feat)
        #self.beta = nn.Linear(self.dim_curve_feat, self.dim_sample_feat)
        dim_in = dim_sample_in
        dim_out = arg['dim_sample_feat']
        dim_cond = arg['dim_curve_feat']

        self.filmenc1 = FilmEncoder(dim_in, dim_out, dim_cond) # arg['dim_sample_feat'], arg[''])
        self.filmenc2 = FilmEncoder(dim_out, dim_out, dim_cond)


        #if arg['num_pos_encoding'] > 0:
        #    self.pos_enc = PosEncoding(arg['num_pos_encoding'])
        #    diff = self.pos_enc.d_out - self.pos_enc.d_in
            #print("diff = ", diff)
        self.decoder = MLP(**arg['decoder_curve'])


    def init_embedding(self, embedding, dim_code):
        nn.init.normal_(embedding.weight.data, 0., 1./ math.sqrt(dim_code))

    
    def set_post_mode(self):
        for pname, param in self.core.named_parameters():
            if 'embd' in pname:
                continue
            param.requires_grad = False


    def forwardsimple(self, model_input, istrain=True):
        mi = model_input
        # curve_idx:(Nb, Ns); coords:(Nb, Ns); samples(Nb,Ns,3)
        curve_code = self.embd(mi['curve_idx'])
        B = curve_code.shape[0]

        type_curve  = self.type_embd(torch.zeros(B, dtype=torch.long, device=self.device)).unsqueeze(1)
        type_sample = self.type_embd(torch.ones(B, dtype=torch.long, device=self.device)).unsqueeze(1)

        if hasattr(self, 'pos_enc_sample'):
            if istrain:
                #print(mi['samples'].shape)
                #print(mi['angles'].shape)
                samples_posenc = self.pos_enc_sample(mi['samples'])
                sample_angle_periodenc = self.period_enc(mi['angles'].unsqueeze(-1))
                curve_coords_posenc = self.pos_enc_curve(mi['coords'].unsqueeze(-1))
                radii_y_posenc = self.pos_enc_curve(torch.log(mi['radius'][:,:,0]).unsqueeze(-1))
                radii_z_posenc = self.pos_enc_curve(torch.log(mi['radius'][:,:,1]).unsqueeze(-1))
            else:
                samples_posenc = self.pos_enc_sample.inference(mi['samples'])
                sample_angle_periodenc = self.period_enc.inference(mi['angles'].unsqueeze(-1))
                curve_coords_posenc = self.pos_enc_curve.inference(mi['coords'].unsqueeze(-1))
                radii_y_posenc = self.pos_enc_curve.inference(torch.log(mi['radius'][:,:,0]).unsqueeze(-1))
                radii_z_posenc = self.pos_enc_curve.inference(torch.log(mi['radius'][:,:,1]).unsqueeze(-1))
        else:
            samples_posenc = mi['samples']
            sample_angle_periodenc = mi['angles'].unsqueeze(-1)
            curve_coords_posenc = mi['coords'].unsqueeze(-1)
            radii_y_posenc = torch.log(mi['radius'][:,:,0]).unsqueeze(-1)
            radii_z_posenc = torch.log(mi['radius'][:,:,1]).unsqueeze(-1)

        #print(curve_code.shape)
        #print(curve_coords_posenc.shape)

        #print(samples_posenc.shape)

        #curve_code_coords = torch.cat([curve_code, curve_coords_posenc], dim=-1)
        curve_code_coords = torch.cat([curve_code, curve_coords_posenc, radii_y_posenc, radii_z_posenc], dim=-1)
        #curve_feats = torch.cat([self.curveencoder(curve_code_coords), self.type_curve], dim=-1)
        curve_feats = self.curveencoder(curve_code_coords) 
        curve_feats = curve_feats + type_curve


        #x = torch.cat([self.sampleencoder1(samples_posenc), self.type_sample], dim=-1)
        #print(samples_posenc.shape)
        #print(sample_angle_periodenc.shape)
        y = torch.cat([samples_posenc, sample_angle_periodenc], dim=-1)
        x = self.sampleencoder1(torch.cat([samples_posenc, sample_angle_periodenc], dim=-1)) 
        x = x + type_sample

        x = self.film1(x, curve_feats)
        res = x
        x = self.sampleencoder2(x)
        x = self.film2(x, curve_feats) 
        x = x + res

        #curve_feats = torch.cat([curve_feats, samples], dim=-1)


        curve_sdf = self.decoder.forward_simple(x).squeeze(-1)
        #curve_sdf = self.decoder.forward_simple(curve_feats).squeeze(-1)

        return {
            'sdf': curve_sdf,
            'code': curve_code,
        }

    def forward(self, model_input):
        return self.forwardsimple(model_input)

    @torch.no_grad()
    def validation(self, model_input):
        return self.forwardsimple(model_input, False)
    
    @torch.no_grad()
    def inference(self, model_input):
        curve_input = self.pack_data(model_input)
        out = self.forwardsimple(curve_input, False)
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
            'rho': torch.from_numpy(mi['rho']).float().to(device),
            'angles': torch.from_numpy(mi['angles']).float().to(device),
            'radius': torch.from_numpy(mi['radius']).float().to(device),
            'curve_idx': curve_idx.to(device)
        }

        res = {key:val.unsqueeze(0) for key,val in res.items()}
        return res


class FiLMEncoder(nn.Module):
    """docstring for FeatCurveEncoder."""
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
        

class FeatSampleEncoder(nn.Module):
    """docstring for FeatCurveEncoder."""
    def __init__(self, dim, out_dim):
        super(FeatSampleEncoder, self).__init__()
        self.fc = nn.Linear(dim, out_dim)
        self.act = nn.SiLU()
    
    def forward(self, x):
        # code(Nb, Ns, N_code); coords(Nb, Ns)
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
