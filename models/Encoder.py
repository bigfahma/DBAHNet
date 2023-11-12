import torch.nn as nn
import torch
import numpy as np

import torch.nn.functional as F
from models.CACBlock import CACBlock
from models.TCFF import TCFF, Downsample
from models.utils import BasicLayerTr

class Enc_layer(nn.Module):
   
    def __init__(self,
                 dim,
                 in_dim,
                 depth,
                 num_heads,
                 window_size=7,
                 mlp_ratio=4.,
                 qkv_bias=True,
                 qk_scale=None,
                 drop=0.,
                 attn_drop=0.,
                 drop_path=0.,
                 norm_layer=nn.LayerNorm,
                 downsample=True
                 ):
        super().__init__()
        self.basic_layer_tr = BasicLayerTr(dim=dim, in_dim=in_dim, depth=depth, 
                                num_heads=num_heads, window_size= window_size)
        self.cac_block = CACBlock(dim)
        self.tcff = TCFF(dim, in_dim=in_dim,  downsample=downsample)

    def forward(self, x, D, H, W):
        B,L,C = x.shape
        assert L == H * W * D, "input feature has wrong size"
        x_cnn = x.view(-1, D, H, W, C).permute(0, 4, 1, 2, 3).contiguous()
        out_cnn = self.cac_block(x_cnn)
        out_tr = self.basic_layer_tr(x, D, H, W)
        out_tr = out_tr.view(-1, D, H, W, C).permute(0, 4, 1, 2, 3).contiguous()
        out = self.tcff(out_tr, out_cnn)
        Dd, Hd, Wd = out.shape[2:]
        out = out.flatten(2).transpose(1, 2).contiguous()
        return out, Dd, Hd, Wd


class Encoder(nn.Module):
    def __init__(self, dim, in_dim, depth, num_heads, window_size, downsample = Downsample):
        super(Encoder, self).__init__()
        D, H, W = in_dim
        self.mrha_enc1 = Enc_layer(dim=dim, in_dim= (D, H, W), depth=depth, num_heads=num_heads[0], window_size=window_size, downsample=downsample)
        self.mrha_enc2 = Enc_layer(dim=2*dim, in_dim= (D//2, H//2, W//2), depth=depth, num_heads=num_heads[1], window_size=window_size, downsample=downsample)
        self.mrha_enc3 = Enc_layer(dim=4*dim, in_dim =(D//4, H//4, W//4), depth=depth, num_heads=num_heads[2], window_size=window_size, downsample=downsample)
        
    def forward(self, x, D, H, W):
        x1, D1, H1, W1 = self.mrha_enc1(x, D, H, W)
        x2, D2, H2, W2 = self.mrha_enc2(x1, D1, H1, W1)
        xout, _, _, _ = self.mrha_enc3(x2, D2, H2, W2)
        skips = [x, x1, x2]
        return xout, skips
    
if __name__ == "__main__":


    dim = 96
    in_dim = (16, 32, 32)
    depth = 2
    num_heads = [6,12,24]
    window_size = 7
    batch_size = 1
    D, H, W = in_dim
    x = torch.rand((batch_size, H * W * D, dim))
    print("Input shape :",x.shape)
    x_view = x.view(-1, D, H, W, dim).permute(0, 4, 1, 2, 3).contiguous()
    print("MRHA Encoder")
    MRHA_Encoder = Encoder(dim = dim, in_dim= in_dim, depth = depth, num_heads = num_heads, window_size = window_size)
    xout, skips = MRHA_Encoder(x, D, H, W)
    skip1, skip2, skip3 = skips
    print("Skips :", skip1.shape, skip2.shape, skip3.shape)