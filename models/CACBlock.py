import torch
import torch.nn as nn

class CACBlock(nn.Module):
    def __init__(self, in_channels, reduction_ratio=4, norm_layer = nn.BatchNorm3d):
        super(CACBlock, self).__init__()
        self.global_avg_pool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.global_max_pool = nn.AdaptiveMaxPool3d((1, 1, 1))
        self.fc1 = nn.Conv3d( 2*in_channels, in_channels // reduction_ratio, kernel_size=3, bias=True, padding = "same")
        self.gelu = nn.GELU()
        self.fc2 = nn.Conv3d(in_channels // reduction_ratio, in_channels , kernel_size=3, bias=True, padding= "same")
        self.sigmoid = nn.Sigmoid()
        self.fc3 = nn.Conv3d(in_channels, in_channels, kernel_size=3, bias=True, padding="same")
        self.norm_layer2 = norm_layer(in_channels)

    def forward(self, x):
        avg_pool = self.global_avg_pool(x)
        max_pool = self.global_max_pool(x)
        concatenated = torch.cat((avg_pool, max_pool), dim=1)
        fc1 = self.gelu(self.fc1(concatenated))
        attention_map = self.sigmoid(self.fc2(fc1))
        output = self.gelu(self.fc3(x * attention_map))
        output = self.norm_layer2(output)
        return output
