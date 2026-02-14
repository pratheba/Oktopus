import math
import torch
import torch.nn as nn

from mlp import *
#from pos_encoding import PosEncoding
from pos_encoding_fromokto import PosEncoding
import shape_encoder


class NGCNet(nn.Module):
    """docstring for NGCNet ."""
    def __init__(self, arg):
        super().__init__()
        self.device = torch.device(arg['device'])
        self.curvecode_dim = arg['curvecode_dim']
      
        ########## CURVE #######################
        ####### Learn curve embdddings ###########
        self.curve_embeddings = nn.Embedding(arg['n_curve'], self.curvecode_dim)
        self.init_embedding(self.curve_embeddings, self.curvecode_dim)
        
        ########## Curve Linear interpolant encoder #########
        self.curve_encoder = CurveEncoder(arg)

        ############## SHAPE #########################
        self.shapeEncoder = shape_encoder.__dict__[arg['shape_model']](N=arg['surface_point_size'], output_dim=1)
        #self.shapeEncoder = shape_encoder.__dict__[arg['shape_model']](N=arg['surface_point_size'], output_dim=1)
        #self.shapeEncoder.to(self.device)

        self.decoder = MLP(**arg['decoder_curve'])


    def init_embedding(self, embedding, dim):
        nn.init.normal_(embedding.weight.data, 0., 1./ math.sqrt(dim))
    
    def set_post_mode(self):
        for pname, param in self.core.named_parameters():
            if 'embd' in pname:
                continue
            param.requires_grad = False

    def forwardabs(self, inputs, isinference=False):
        curve_idx = inputs['curve_idx']
        #print(curve_idx)
        # The coordinates on the curve obtained by projecting samples from surface to obtain
        # lot of keypoints
        curve_coords = inputs['coords']
        #print("curve coords = ", curve_coords.shape, flush=True)

        # curve_idx:(Nb, Ns); coords:(Nb, Ns); samples(Nb,Ns,3)
        curve_code = self.curve_embeddings(curve_idx)
        curve_features = self.curve_encoder(curve_code, curve_coords)
        #print("curve_features =", curve_features.shape)
        #index_curve_coords = curve_idx + curve_coords


        #curve_context_idx = inputs['on_curve_idx']
        #curve_context_coords = inputs['on_coords']
        #curve_context_code = self.curve_embeddings(curve_context_idx)
        #curve_context_features = self.curve_encoder(curve_context_code, curve_context_coords)


        ######### Auto encoder for shapes ############

        #context_samples = inputs['on_surface_samples']
        query_samples = inputs['samples']
        #print(query_samples.shape, flush=True)

        ################ Combine to get sdf output or put the curve info as well to get the sdf output ? #######
        #out = self.shapeEncoder(context_samples, query_samples)
        #out = self.shapeEncoder(curve_coords.unsqueeze(-1), query_samples)
        out = self.shapeEncoder(curve_features, query_samples)
        #print(curve_coords.shape)
        query_features = out['query_features']
        #context_features = out['context_features']

        #kl = out['kl']
    
        #out_query_features = torch.cat([curve_features, query_features], dim=-1)
        #out_context_features = torch.cat([curve_context_features, context_features], dim=-1)
        #print("out_features.shape = ", out_features.shape, flush=True)
        
        
        #out_query_features = torch.cat([curve_features, query_samples], dim=-1)
        #print(out_query_features)
        #sdf_query = self.decoder.forward_simple(out_query_features).squeeze(-1)
        sdf_query = query_features
        #sdf_context = self.decoder.forward_simple(out_context_features).squeeze(-1)
        #print("sdf query shape = ", sdf_query.shape)
        #print("sdf context shape = ", sdf_context.shape)
        #sdf = torch.cat([sdf_query, sdf_context],dim=-1)
        #all_curve_code = torch.cat([curve_code, curve_context_code], dim=-1) 
        #print("sdf shape = ", sdf.shape)
        kl = None
        sdf = sdf_query
        #sdf = out_query_features['query_logits']
        return {
            'sdf': sdf,
            #'enc_features': encoder_features,
            #'dec_features': decoder_features,
            #'curve_code': all_curve_code,
            'curve_code': curve_code,
            'kl': kl
        }

    def forward(self, inputs):
        return self.forwardabs(inputs)

    @torch.no_grad()
    def validation(self, inputs):
        return self.forwardabs(inputs)
    
    @torch.no_grad()
    def inference(self, inputs):
        inputs = self.pack_data(inputs)
        out = self.forwardabs(inputs, isinference=True)
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
            'curve_idx': curve_idx.to(device),
            #'on_surface_samples': mi['on_surface_samples'].float().to(device),
            #'on_coords': mi['on_coords'].float().to(device),
            #'on_curve_idx': mi['on_curve_idx'].to(device)
        }

        res = {key:val.unsqueeze(0) for key,val in res.items()}
        return res


class CurveEncoder(nn.Module):
    """docstring for CurveEncoder."""
    def __init__(self, arg):
        super(CurveEncoder, self).__init__()
        self.curvecode_dim = arg['curvecode_dim']
        self.feature_dim = arg['curvefeature_dim']

        ####### map from curvecode dimension to 2* feature dimension for end points of the curve
        self.linear1 = nn.Linear(self.curvecode_dim, 2*self.feature_dim)
        self.mlp = MLP(**arg['encoder_curve'])
    
    def forward(self, curvecode, curvecoords):
        # code(Nb, Ns, N_code); coords(Nb, Ns)
        B, N, curvecode_dim = curvecode.shape

        curve_endpoint_features = self.linear1(curvecode).view(B, N, self.feature_dim, 2)
        # weights (Nb, Ns, 2)
        weights = torch.stack([curvecoords, 1-curvecoords], dim=-1)
        curve_features = torch.einsum('bnjk,bnk->bnj', curve_endpoint_features, weights)
        return self.mlp.forward_simple(curve_features)
   
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
