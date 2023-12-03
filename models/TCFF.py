import torch
import torch.nn as nn
import torch.nn.functional as F
from models.utils import BasicLayerTr
class ContiguousGrad(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        return x
    @staticmethod
    def backward(ctx, grad_out):
        return grad_out.contiguous()


class Downsample(nn.Module):
  

    def __init__(self, dim, norm_layer=nn.BatchNorm3d):
        super().__init__()
        self.dim = dim
        self.conv = nn.Conv3d(dim,dim*2,kernel_size=3,stride=2,padding=1)
        self.gelu = nn.GELU()
        self.norm = norm_layer(2*dim)

    def forward(self, x):
        x=self.gelu(self.conv(x)) 
        x = self.norm(x) 
        return x
    

class Upsample(nn.Module):
    def __init__(self, dim, norm_layer=nn.BatchNorm3d):
        super().__init__()
        self.dim = dim   
        self.norm = norm_layer(dim)
        self.transconv=nn.ConvTranspose3d(dim,dim//2,2,2)
        self.gelu = nn.GELU()
        self.norm = norm_layer(dim//2)
    def forward(self, x):
        x = self.transconv(x)
        x = self.norm(x)
        return x
    

class TCFF(nn.Module):
    def __init__(self, input_channels,in_dim , downsample = None, upsample = None):
        super(TCFF, self).__init__()
        self.in_dim = in_dim
        self.dim = input_channels
        self.avg_pool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.GELU()     
        self.conv1 = nn.Conv3d(input_channels * 2, input_channels, kernel_size=1)
        self.norm_layer1 = nn.BatchNorm3d(input_channels)
        self.downsample = downsample
        self.upsample = upsample
        if self.downsample is not None:
            self.downsample = downsample(dim=input_channels)
        elif self.upsample is not None:
            self.Upsample = upsample(dim= input_channels)

    def forward(self, x_tr, x_cnn):
        D, H, W = self.in_dim
        attmap_tr = self.sigmoid(self.avg_pool(x_tr))
        attmap_cnn = self.sigmoid(self.avg_pool(x_cnn))
        x_tr = x_tr * attmap_tr
        x_cnn = x_cnn * attmap_cnn
        x = torch.cat((x_tr, x_cnn), dim=1)
        x = self.relu(self.conv1(x))
        x = self.norm_layer1(x)
        if self.downsample:
            x = self.downsample(x)
            return x
        elif self.Upsample:
            x = self.Upsample(x)
            return x
