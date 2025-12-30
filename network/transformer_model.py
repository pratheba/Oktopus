import math
import torch
import torch.nn as nn

from pos_encoding import PosEncoding
from surface_encoder import SurfaceEncoder
from curve_encoder import CurveEncoder

class NGCNet(nn.Module):
    """docstring for NGCNet ."""
    def __init__(self, arg):
        super().__init__()

        ###### Curve encoding
        curve_arg = arg['curve_encoder']
        self.dim_curvecode = curve_arg['dim_code']
        self.embd = nn.Embedding(curve_arg['n_curve'], self.dim_curvecode)
        self.init_embedding(self.embd, self.dim_curvecode)
        self.curveEncoder = CurveEncoder(curve_arg)

        ###### Surface encoding
        surface_arg = arg['surface']
        self.surfaceEncoder = SurfaceEncoder(surface_arg)
        self.output_dim = arg['output_dim']


        #if arg['num_pos_encoding'] > 0:
        #    self.pos_enc = PosEncoding(arg['num_pos_encoding'])
        #    diff = self.pos_enc.d_out - self.pos_enc.d_in
        #    arg['decoder_curve']['size'][0] += diff
        #self.decoder = MLP(**arg['decoder_curve'])


    def init_embedding(self, embedding, dim_code):
        nn.init.normal_(embedding.weight.data, 0., 1./ math.sqrt(dim_code))

    
    def set_post_mode(self):
        for pname, param in self.core.named_parameters():
            if 'embd' in pname:
                continue

            param.requires_grad = False


    def forward(self, x):
        curve_idx = x['curve_idx']
        curve_coords = x['coords']
        local_samples = x['samples']

        # curve_idx:(Nb, Ns); coords:(Nb, Ns); samples(Nb,Ns,3)
        curve_code = self.embd(curve_idx)
        curve_feats = self.curveEncoder(curve_code, coords)
        surface_feats = self.surfaceEncoder(local_samples)

        cylinder_feats = torch.cat([curve_feats, surface_feats], dim=-1)
        
        curve_sdf = self.decoder.forward_simple(curve_feats).squeeze(-1)
        return {
            'sdf': curve_sdf,
            'code': curve_code,
        }

    @torch.no_grad()
    def validation(self, model_input):
        curve_input = model_input
        ci = curve_input
        curve_code = self.embd(ci['curve_idx'])

        curve_feats = self.encoder(curve_code, ci['coords'])
        if hasattr(self, 'pos_enc'):
            samples = self.pos_enc(ci['samples'])
        else:
            samples = ci['samples']
        curve_feats = torch.cat([curve_feats, samples], dim=-1)

        curve_sdf = self.decoder.forward_simple(curve_feats).squeeze(-1)
        return {
            'sdf': curve_sdf,
            'code': curve_code,
        }
    
    @torch.no_grad()
    def inference(self, model_input):
        curve_input = self.pack_data(model_input)
        ci = curve_input
        curve_code = self.embd(ci['curve_idx'])

        curve_feats = self.encoder(curve_code, ci['coords'])
        if hasattr(self, 'pos_enc'):
            print("positional encoding", flush=True)
            samples = self.pos_enc.inference(ci['samples'])
        else:
            samples = ci['samples']
        curve_feats = torch.cat([curve_feats, samples], dim=-1)

        curve_sdf = self.decoder.forward_simple(curve_feats).squeeze(-1)
        return curve_sdf

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
            'curve_idx': curve_idx.to(device)
        }

        res = {key:val.unsqueeze(0) for key,val in res.items()}
        return res



if __name__ == "__main__":
    pass
