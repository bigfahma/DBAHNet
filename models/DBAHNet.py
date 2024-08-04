from models.Decoder import Decoder
from models.Encoder import Encoder
from models.Bottleneck import Bottleneck
from models.PatchEmbedding import PatchEmbed
import torch.nn as nn
import torch
import time
import torchprofile

class DBAHNet(nn.Module):
    def __init__(self, emb_dim, in_dim, num_classes, depth, num_heads, window_size):
        super().__init__()
        D,H,W = in_dim
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
        self.bottleneck = Bottleneck(dim = 8*emb_dim, depth = 2, num_heads = num_heads[3])
    def forward(self, x):
        #print("IMAGE DIM : ",self.D, self.H, self.W )
        #print("ENCODER IN : ",self.De, self.He, self.We)
        #print("in",x.shape)
        x = self.patch_embedding(x)
        #print("Out Patch Embedding :", x.shape)
        x, skips = self.mrha_encoder(x, self.De, self.He, self.We)
        #print("Out Encoder :", x.shape)
        x = self.bottleneck(x)
        #print("Out Bottleneck :", x.shape)
        skip1, skip2, skip3 = skips
        #print("Skip1 skip2 skip3 shape :",skip1.shape, skip2.shape, skip3.shape)
        #print("IN DECODER :",self.Dd, self.Hd, self.Wd)
        x = self.mrha_decoder(x, skips)
        #print("Out Decoder :", x.shape)

        return x

if __name__ == "__main__":
    x = torch.randn(1, 1,320, 320, 32) 
    emb_dim = 96
    in_dim = x.shape[2:]
    num_classes = 3
    num_heads = [6,12,24,48]

    dbahnet = DBAHNet(emb_dim= emb_dim, in_dim=in_dim, num_classes= num_classes,
                      depth = 2, num_heads=num_heads, window_size=7)
    print("Input shape", x.shape)
    start_time = time.time()
    output = dbahnet(x)
    end_time = time.time()
    print("Time forward :", end_time - start_time)
    print("Output :",output.shape)
    flops = torchprofile.profile_macs(model, inputs)
    print(f"FLOPs: {flops}")