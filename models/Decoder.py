import torch.nn as nn
import torch
from models.utils import  SwinTransformerBlock_kv, SwinTransformerBlock, get_attention_mask
import torch.nn.functional as F
from models.SACBlock import SACBlock
from models.TCFF import TCFF, Upsample
from models.CACBlock import CACBlock
class AttentionGate(nn.Module):

    def __init__(self,
                 dim,
                 in_dim,
                 num_heads,
                 window_size=7,
                 mlp_ratio=4.,
                 qkv_bias=True,
                 qk_scale=None,
                 drop=0.,
                 attn_drop=0.,
                 drop_path=0.,
                 norm_layer=nn.LayerNorm,
                ):
        super().__init__()
        self.window_size = window_size
        self.shift_size = window_size // 2

        self.ag = SwinTransformerBlock_kv(
                    dim=dim,
                    in_dim = in_dim,
                    num_heads=num_heads,
                    window_size=window_size,
                    shift_size=0 ,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    qk_scale=qk_scale,
                    drop=drop,
                    attn_drop=attn_drop,
                    drop_path=drop_path[0] if isinstance(drop_path, list) else drop_path, norm_layer=norm_layer)
                   
    def forward(self, skip, D, H, W, x_up):
        #print("Attention gate")
        x = x_up + skip
        attn_mask = get_attention_mask(D, H, W, window_size=self.window_size, shift_size=self.shift_size, device=x.device)
        x = self.ag(x, attn_mask, skip=skip, x_up=x_up)
        return x, D, H, W

class BasicLayer_up(nn.Module):

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
                 upsample=True
                ):
        super().__init__()
        self.window_size = window_size
        self.shift_size = window_size // 2
        self.depth = depth
        
        self.blocks = nn.ModuleList()
        for i in range(depth-1):
            self.blocks.append(
                SwinTransformerBlock(
                        dim=dim,
                        in_dim=in_dim,
                        num_heads=num_heads,
                        window_size=window_size,
                        shift_size=self.shift_size ,
                        mlp_ratio=mlp_ratio,
                        qkv_bias=qkv_bias,
                        qk_scale=qk_scale,
                        drop=drop,
                        attn_drop=attn_drop,
                        drop_path=drop_path[i+1] if isinstance(drop_path, list) else drop_path, norm_layer=norm_layer)
                        )
    def forward(self,x, D, H, W):
        #print("Before ag :", x.shape)
        #print("Basic Layer UP")
        attn_mask = get_attention_mask(D, H, W, self.window_size, self.shift_size, x.device)
        #print("After ag :",x.shape)
        for i in range(self.depth - 1):
            x = self.blocks[i](x,attn_mask)
        
        return x
    
class RegularConvBlock(nn.Module):
    def __init__(self, dim):
        super(RegularConvBlock, self).__init__()
        self.conv = nn.Conv3d(in_channels=dim, out_channels=dim, kernel_size=3, padding=1)
        self.bn = nn.BatchNorm3d(dim)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x
class MRHA_dec_layer(nn.Module):
   
    def __init__(self,
                 dim,
                 in_dim,
                 depth,
                 num_heads,
                 window_size=7,
                 norm_layer=nn.LayerNorm,
                 upsample=True
                 ):
        super().__init__()
        self.window_size = window_size
        self.shift_size = window_size // 2
        self.in_dim_kv = (2*in_dim[0], 2*in_dim[1], 2*in_dim[2])
        self.dim_kv = dim // 2
        self.attention_gate = AttentionGate(dim = self.dim_kv, in_dim = self.in_dim_kv, num_heads = num_heads,
                                            window_size=self.window_size)
        self.basic_layer_up = BasicLayer_up(dim=dim, in_dim=in_dim, depth=depth, 
                             num_heads=num_heads, window_size=window_size, upsample = upsample, norm_layer=norm_layer)
        self.sac_block = SACBlock(dim)
        self.tcff = TCFF( dim,in_dim=in_dim, upsample=upsample)
        #self.cac_block = CACBlock(dim)
        #self.conv_block = RegularConvBlock(dim)
    def forward(self, x, skip, D, H, W):
        B,L,C = x.shape
        assert L == H * W * D, "input feature has wrong size"
        x_cnn = x.view(-1, D, H, W, C).permute(0, 4, 1, 2, 3).contiguous()
        #print("X CNN :", x_cnn.shape)
        out_cnn = self.sac_block(x_cnn)
        #out_cnn = self.cac_block(x_cnn)
        #out_cnn = self.conv_block(x_cnn)

        out_tr = self.basic_layer_up(x, D, H, W)
        out_tr = out_tr.view(-1, D, H, W, C).permute(0, 4, 1, 2, 3).contiguous()
        #print("Out tr :", out_tr.shape)
        out_tcff = self.tcff(out_tr, out_cnn)
        H, W, D = H*2, W*2, D*2
        #print("Out TCFF :", out_tcff.shape)
        #print("Skip :", skip.shape)
        out_tcff_flattened = out_tcff.flatten(2).transpose(1, 2).contiguous()
        #print("out TCFF flattened :", out_tcff_flattened.shape)
        out, D, H, W = self.attention_gate(skip, D, H, W, out_tcff_flattened)
        #print("Out :", out.shape)
        return out, D, H, W

class Decoder(nn.Module):
    def __init__(self, dim, in_dim, num_classes, depth, num_heads, window_size, upsample = Upsample, stride = [2,2,2]):
        super(Decoder, self).__init__()
        self.D_in, self.H_in, self.W_in = in_dim
        self.dim = dim
        self.mrha_dec1 = MRHA_dec_layer(dim=self.dim, in_dim =(self.D_in, self.H_in, self.W_in), depth=depth, num_heads=num_heads[2], window_size=window_size,  upsample=upsample)
        self.mrha_dec2 = MRHA_dec_layer(dim=self.dim//2, in_dim= (2*self.D_in, 2*self.H_in, 2*self.W_in), depth=depth, num_heads=num_heads[1], window_size=window_size,  upsample=upsample)
        self.mrha_dec3 = MRHA_dec_layer(dim=self.dim//4, in_dim= (4*self.D_in, 4*self.H_in, 4*self.W_in), depth=depth, num_heads=num_heads[0], window_size=window_size, upsample=upsample)
       
        self.up_final = nn.ConvTranspose3d(dim//8 , num_classes, kernel_size = stride, stride = stride)
    def forward(self, x, skips):
        skip1, skip2, skip3 = skips
        #print("D IN H IN :",self.D_in, self.H_in, self.W_in)
        x_view = x.view(-1, self.D_in, self.H_in, self.W_in, self.dim).permute(0, 4, 1, 2, 3).contiguous()
        #print("x view:",x_view.shape)
        # First layer
        x1, D1, H1, W1 = self.mrha_dec1(x, skip3, self.D_in, self.H_in, self.W_in)
        x1_view = x1.view(-1, D1, H1, W1, self.dim//2).permute(0, 4, 1, 2, 3).contiguous()
        #print("l1 output shape :", x1_view.shape)
        # Second layer
        x2, D2, H2, W2 = self.mrha_dec2(x1, skip2, D1, H1, W1)
        x2_view = x2.view(-1, D2, H2, W2, self.dim//4).permute(0, 4, 1, 2, 3).contiguous()
        #print("l2 output shape :", x2_view.shape)
        # Third layers
        x3, D3, H3, W3 = self.mrha_dec3(x2, skip1, D2, H2, W2)
        x3_view = x3.view(-1, D3, H3, W3, self.dim//8).permute(0, 4, 1, 2, 3).contiguous()
        #print("l3 output Decoder :", x3_view.shape)
        out = self.up_final(x3_view)
        #print("outshape :",out.shape)
        #out = out.permute(0,1,4,3,2)
        #print("Expanding output :", out.shape)

        return out

if __name__ == "__main__":

    print("MRHA Decoder ")
    in_dim = (4, 8, 8)
    depth = 2
    num_heads = [6,12,24]
    window_size = 7
    batch_size = 1
    num_classes = 3
    C = 96*8
    dim = C
    D, H, W = in_dim
    x = torch.rand((batch_size, H * W  * D , C))
    skip3 = torch.rand((batch_size, 2*H * 2*W * 2*D, dim//2))
    skip2 = torch.rand((batch_size, 4*H * 4*W * 4*D, dim//4))
    skip1 = torch.rand((batch_size, 8*H * 8*W * 8*D, dim//8))
    skips = [skip1, skip2, skip3]
    mrha_decoder = MRHADecoder(dim, in_dim,num_classes , depth, num_heads, window_size, upsample= Upsample)
    print("input shape :", x.shape)
    print("skip 3 shape :", skip3.shape)
    print("skip 2 shape :", skip2.shape)
    print("skip 1 shape :", skip1.shape)
    out_decoder = mrha_decoder(x, skips)
    print("Output shape:", out_decoder.shape)

