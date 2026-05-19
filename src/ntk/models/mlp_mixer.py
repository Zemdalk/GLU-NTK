# The code is adapted from the 'mlp-mixer-pytorch' repository under MIT license.
# Original repository link: https://github.com/lucidrains/mlp-mixer-pytorch

import torch
import torch.nn as nn
from functools import partial
from einops.layers.torch import Rearrange, Reduce
from .mlp import MLP

pair = lambda x: x if isinstance(x, tuple) else (x, x)

class PreNormResidual(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        return self.fn(self.norm(x)) + x

class TokenMixer(nn.Module):
    def __init__(self, num_patches, hidden_dim, hidden_type="gelu", init_type="xavier", dropout=0.):
        super().__init__()
        self.ff = MLP(in_dim=num_patches, hidden_dim=hidden_dim, num_classes=num_patches,
                      hidden_type=hidden_type, init_type=init_type, in_flatten=False)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        x_t = x.transpose(1, 2)
        B, C, N = x_t.shape
        y = self.ff(x_t.reshape(B * C, N))  
        y = y.view(B, C, N).transpose(1, 2) 
        return self.drop(y)

class ChannelMixer(nn.Module):
    def __init__(self, dim, hidden_dim, hidden_type="gelu", init_type="xavier", dropout=0.):
        super().__init__()
        self.ff = MLP(in_dim=dim, hidden_dim=hidden_dim, num_classes=dim,
                      hidden_type=hidden_type, init_type=init_type, in_flatten=False)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        B, N, C = x.shape
        y = self.ff(x.reshape(B * N, C))     
        y = y.view(B, N, C)
        return self.drop(y)

def MLPMixer(
    *, image_size, channels, patch_size, dim, depth, num_classes,
    expansion_factor = 4,               
    expansion_factor_token = 0.5,       
    dropout = 0.,
    token_hidden_type = "gelu", token_init_type = "xavier",
    channel_hidden_type = "gelu", channel_init_type = "xavier",
):
    image_h, image_w = pair(image_size)
    assert (image_h % patch_size) == 0 and (image_w % patch_size) == 0, 'image must be divisible by patch size'
    num_patches = (image_h // patch_size) * (image_w // patch_size)

    token_hidden   = int(expansion_factor * dim)
    channel_hidden = int(expansion_factor_token * dim)

    return nn.Sequential(
        Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = patch_size, p2 = patch_size),
        nn.Linear((patch_size ** 2) * channels, dim),

        *[nn.Sequential(
            PreNormResidual(dim, TokenMixer(num_patches, token_hidden, hidden_type=token_hidden_type, init_type=token_init_type, dropout=dropout)),
            PreNormResidual(dim, ChannelMixer(dim, channel_hidden, hidden_type=channel_hidden_type, init_type=channel_init_type, dropout=dropout))
        ) for _ in range(depth)],

        nn.LayerNorm(dim),
        Reduce('b n c -> b c', 'mean'),
        nn.Linear(dim, num_classes)
    )
