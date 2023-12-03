from models.Decoder import Decoder
from models.Encoder import Encoder
from models.Bottleneck import Bottleneck
from models.PatchEmbedding import PatchEmbed
import torch.nn as nn
import torch
import time

class DBAHNet(nn.Module):
    def __init__(self, emb_dim, in_dim, num_classes, depth, num_heads, window_size):
        super().__init__()
        H, W, D = in_dim
        downD, downH, downW = 4,4,4
        self.De, self.He, self.We = D//downD, H//downH, W//downW
        self.Dd, self.Hd, self.Wd = self.De//8, self.He//8, self.We//8
        self.in_dim_encoder = [self.De, self.He, self.We]
        self.in_dim_decoder = [self.Dd, self.Hd, self.Wd]
        self.patch_embedding = PatchEmbed(patch_size=4, in_chans=1, embed_dim=emb_dim, 
                                 norm_layer=nn.LayerNorm, out_proj1 = (2,2,2), out_proj2 = (2,2,2))
        self.mrha_encoder = Encoder(dim = emb_dim, in_dim = self.in_dim_encoder, depth = depth, 
                                        num_heads = num_heads, window_size = window_size)
        self.mrha_decoder = Decoder(dim = 8*emb_dim, in_dim = self.in_dim_decoder, num_classes=num_classes,
                                        depth = depth, num_heads = num_heads, 
                                         window_size = window_size, stride =(downD, downH, downW))
        self.bottleneck = Bottleneck(dim = 8*emb_dim, depth = 2, num_heads = num_heads[2])
    def forward(self, x):
        x = x.permute(0, 1, 4, 3, 2)
        x = self.patch_embedding(x)
        x, skips = self.mrha_encoder(x, self.De, self.He, self.We)
        x = self.bottleneck(x)
        x = self.mrha_decoder(x, skips)
        x = x.permute(0, 1, 4, 3, 2)
        return x