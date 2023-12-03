import torch
import torch.nn as nn

class SACBlock(nn.Module):
    def __init__(self, in_channels, norm_layer = nn.BatchNorm3d):
        super(SACBlock, self).__init__()
        self.conv3d = nn.Conv3d(in_channels=2, out_channels=1, kernel_size=1)
        self.sigmoid = nn.Sigmoid()
        self.conv3d2 = nn.Conv3d(in_channels= in_channels, out_channels= in_channels , kernel_size=3, padding="same")
        self.gelu = nn.GELU()
        self.norm_layer = norm_layer(in_channels)
    def forward(self, x):
        avg_pool = torch.mean(x, dim=1, keepdim=True)
        max_pool = torch.max(x, dim=1, keepdim=True)[0]
        concatenated = torch.cat((avg_pool, max_pool), dim=1)
        conv3d_out = self.conv3d(concatenated)
        attention_map = self.sigmoid(conv3d_out)
        output = self.gelu(self.conv3d2(attention_map * x))     
        output = self.norm_layer(output)   
        return output
