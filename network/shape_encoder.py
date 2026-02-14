import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
from torch_cluster import fps


#from pos_encoding import PosEncoding
from pos_encoding_fromokto import PosEncoding
from components import Attention, FeedForward, PreNorm, DiagonalGaussianDistribution
from embeddings import PointEmbed
from model_utils import cache_fn, exists


class KLSurfaceEncoder(nn.Module):
    def __init__(
            self,
            *,
            depth=24, # number of layers of self or cross attention
            dim=512, # this is on surface points during cross attention
            queries_dim=512, # this is on off surface points during cross_attention
            output_dim=1,
            num_inputs = 2048, # number of on surface points
            num_latents = 512, # number of latents of queries also from on surface
            latent_dim = 64,
            num_freq = 32,
            heads = 8,
            dim_head= 8,
            weight_tie_layers = False,
            decoder_ff = True
            ):

        super().__init__()

        self.depth = depth # number of layers
        self.num_inputs = num_inputs
        self.num_latents = num_latents


        self.point_embed = PointEmbed(num_freq=num_freq, dim=dim)
        #self.point_embed = PosEncoding(num_freqs=num_freq, dim=dim)

        ############ ENCODER modules ############################

        self.encoder_cross_attn_blocks = nn.ModuleList([
            PreNorm(dim, fn=Attention(queries_dim=dim, context_dim=dim, heads=1, head_dim=dim_head), context_dim= dim),
            PreNorm(dim, fn=FeedForward(dim))
            ])


        ############ DECODER modules ###########################

        get_latent_attn = lambda: PreNorm(dim, Attention(dim, heads= heads, head_dim = dim_head, drop_path_rate=0.1))
        get_latent_ff = lambda: PreNorm(dim, FeedForward(dim, drop_path_rate=0.1))
        get_latent_attn, get_latent_ff = map(cache_fn, (get_latent_attn, get_latent_ff))


        self.layers = nn.ModuleList([])
        cache_args = {'_cache': weight_tie_layers}

        for i in range(depth):
            self.layers.append(nn.ModuleList([
                get_latent_attn(**cache_args),
                get_latent_ff(**cache_args)
                ]))



        self.decoder_cross_attn = PreNorm(queries_dim, Attention(queries_dim, context_dim=dim, heads= heads, head_dim = dim_head), context_dim = dim)
        self.decoder_ff = PreNorm(queries_dim, FeedForward(queries_dim)) if decoder_ff else None

#        get_latent_attn = lambda: PreNorm(queries_dim, Attention(queries_dim, context_dim=dim, heads= heads, head_dim = dim_head, drop_path_rate=0.1), context_dim=dim)
#        get_latent_ff = lambda: PreNorm(queries_dim, FeedForward(queries_dim, drop_path_rate=0.1))
#        get_latent_attn, get_latent_ff = map(cache_fn, (get_latent_attn, get_latent_ff))
#        self.layers = nn.ModuleList([])
#        cache_args = {'_cache': weight_tie_layers}
#
#
#        for i in range(depth):
#            self.layers.append(nn.ModuleList([
#                get_latent_attn(**cache_args),
#                get_latent_ff(**cache_args)
#                ]))


        self.proj = nn.Linear(latent_dim, dim)
        #print("outut dim = ", output_dim)

        self.to_outputs = nn.Linear(queries_dim, output_dim) if exists(output_dim) else nn.Identity()

        ############ KL REG modules ############################

        self.mean_fc = nn.Linear(dim, latent_dim)
        self.logvar_fc = nn.Linear(dim, latent_dim)



    def encoderonsurface(self, onsurface_pts):
        B, N, D = onsurface_pts.shape
        #print(onsurface_pts.shape, flush=True)
        #print(self.num_inputs, flush=True)

        assert N == self.num_inputs

        flattened_pts = onsurface_pts.view(B*N, D)
        batch = torch.arange(B).to(onsurface_pts.device)
        batch = torch.repeat_interleave(batch, N)

        pos = flattened_pts
        ratio = 1.0 * self.num_latents / self.num_inputs

        idx = fps(pos, batch, ratio=ratio)
        #print("idx shape = ", len(idx))

        patch_center = pos[idx]
        patch_center = patch_center.view(B, -1, 3)


        ############## EMBEDDINGS ###################

        patch_center_embeddings = self.point_embed(patch_center)
        surface_embeddings = self.point_embed(onsurface_pts)
        #print("patch surface shape = ", patch_center_embeddings.shape, flush=True)

        

        ############ SELF ATTENTION ( PATCH CENTERS and SURFACE POINTS) ##########

        cross_attn, cross_ff = self.encoder_cross_attn_blocks

        # Query = patch center
        x = cross_attn(patch_center_embeddings, context=surface_embeddings, mask=None) + patch_center_embeddings
        #print(x.shape)
        #print(cross_ff)
        x = cross_ff(x) + x

        ########### KL REGULARIZATION ##############
        mean = self.mean_fc(x)
        logvar = self.logvar_fc(x)

        posterior = DiagonalGaussianDistribution(mean, logvar)
        x = posterior.sample()
        kl = posterior.kl()

        return kl, x

    def encoder_coords(self, curvecoords_pts):
        B, N, D = curvecoords_pts.shape
        #print(onsurface_pts.shape, flush=True)
        #print(self.num_inputs, flush=True)

        #assert N == self.num_inputs

        flattened_pts = onsurface_pts.view(B*N, D)
        batch = torch.arange(B).to(onsurface_pts.device)
        batch = torch.repeat_interleave(batch, N)

        pos = flattened_pts
        ratio = 1.0 * self.num_latents / self.num_inputs

        idx = fps(pos, batch, ratio=ratio)
        #print("idx shape = ", len(idx))

        patch_center = pos[idx]
        patch_center = patch_center.view(B, -1, 3)


        ############## EMBEDDINGS ###################

        patch_center_embeddings = self.point_embed(patch_center)
        surface_embeddings = self.point_embed(onsurface_pts)
        #print("patch surface shape = ", patch_center_embeddings.shape, flush=True)

        

        ############ SELF ATTENTION ( PATCH CENTERS and SURFACE POINTS) ##########

        cross_attn, cross_ff = self.encoder_cross_attn_blocks

        # Query = patch center
        x = cross_attn(patch_center_embeddings, context=surface_embeddings, mask=None) + patch_center_embeddings
        #print(x.shape)
        #print(cross_ff)
        x = cross_ff(x) + x

        ########### KL REGULARIZATION ##############
        mean = self.mean_fc(x)
        logvar = self.logvar_fc(x)

        posterior = DiagonalGaussianDistribution(mean, logvar)
        x = posterior.sample()
        kl = posterior.kl()

        return kl, x



    def decoder(self, x, offsurface_pts): # offsurface = queries
        queries = offsurface_pts

        # latent dim from encoder projected to be in alignment with dim
        #print("x shape = ", x.shape)
        x = self.proj(x)

        #for attn, ff in self.layers:
        #    x = attn(x) + x
        #    x = ff(x) + x


        #print("x shape = ", x.shape)


        ### cross attention with queries
        queries_embeddings = self.point_embed(queries)

        #print("q emb = ", queries_embeddings.shape)
        latents = self.decoder_cross_attn(queries_embeddings, context=x)
        #return latents

        if exists(self.decoder_ff):
            latents = latents + self.decoder_ff(latents)

        return self.to_outputs(latents)

    def forward(self, coords, queries):
        #kl, x = self.encoder_coords(coords)
        #encoder_logits = x
        #print("get latent for the encoded pat")

        #context_logits = self.decoder(x, onsurface_pts).squeeze(-1)
        #print("now decoder")
        #x = onsurface_pts
        x = coords
        #print("x shape = ", x.shape)
        kl = None
        
        query_logits = self.decoder(x, queries).squeeze(-1)

        #return {'encoder_logits':  encoder_logits, 'decoder_logits': decoder_logits, 'out_features': out, 'kl': kl}
        #return {'query_features': query_logits, 'context_features': context_logits, 'kl': kl}
        return {'query_features': query_logits, 'kl': kl}


def kl_d512_m512_f64_l512(N=2048, output_dim=1):
    return KLSurfaceEncoder(dim=512, num_latents=512, latent_dim=512, num_freq=64, num_inputs=N, output_dim=ouput_dim)

def kl_d512_m512_f32_l512(N=2048, output_dim=1):
    return KLSurfaceEncoder(dim=512, num_latents=512, latent_dim=512, num_freq=32, num_inputs=N, output_dim=output_dim)

def kl_d512_m512_f32_l128(N=2048, output_dim=1):
    return KLSurfaceEncoder(dim=512, num_latents=512, latent_dim=128, num_freq=32, num_inputs=N, output_dim=output_dim)

def kl_d512_m512_f24_l128(N=2048, output_dim=1):
    return KLSurfaceEncoder(dim=1024, queries_dim=1024, depth=24, num_latents=512, latent_dim=1024, num_freq=32, num_inputs=N, heads=24, dim_head=1024, output_dim=output_dim)

def kl_d512_m512_f64_l64(N=2048, output_dim=1):
    return KLSurfaceEncoder(dim=512, num_latents=512, latent_dim=64, num_freq=64, num_inputs=N, output_dim=output_dim)

def kl_d512_m512_f32_l64(N=2048, output_dim=1):
    return KLSurfaceEncoder(dim=512, num_latents=512, latent_dim=64, num_freq=32, num_inputs=N, output_dim=output_dim)

def kl_d512_m512_f32_l32(N=2048, output_dim=1):
    return KLSurfaceEncoder(dim=512, num_latents=512, latent_dim=32, num_freq=32, num_inputs=N, output_dim=output_dim)

def kl_d512_m512_f32_l16(N=2048, output_dim=1):
    return KLSurfaceEncoder(dim=512, num_latents=512, latent_dim=16, num_freq=32, num_inputs=N, output_dim=output_dim)

def kl_d512_m512_f32_l8(N=2048, freq=32, output_dim=1):
    return KLSurfaceEncoder(dim=512, num_latents=512, latent_dim=8, num_inputs=N, output_dim=output_dim)
