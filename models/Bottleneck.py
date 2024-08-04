import torch
from torch import nn

from einops import rearrange, repeat
from einops.layers.torch import Rearrange

# helpers

def pair(t):
    return t if isinstance(t, tuple) else (t, t)

# classes

class FeedForward(nn.Module):
    def __init__(self, dim, mlp_ratio, dropout = 0.):
        super().__init__()
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, mlp_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)

class Attention(nn.Module):
    def __init__(self, dim, heads = 8, dropout = 0.):
        super().__init__()
        dim_head = dim // heads
        inner_dim = dim_head *  heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.norm = nn.LayerNorm(dim)

        self.attend = nn.Softmax(dim = -1)
        self.dropout = nn.Dropout(dropout)

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        x = self.norm(x)

        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

class Bottleneck(nn.Module):
    def __init__(self, dim, depth, num_heads, mlp_ratio = 0.4, dropout = 0.):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Attention(dim, heads = num_heads, dropout = dropout),
                FeedForward(dim, mlp_ratio, dropout = dropout)
            ]))

    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return self.norm(x)
    

if __name__ == "__main__":
    depth = 4  # Number of layers in the transformer
    num_heads = 48  # Number of attention heads
    mlp_ratio = 0.4  # Dimension of the hidden layer in the feed-forward network
    dropout = 0.1  # Dropout rate

    in_dim = (2, 4, 4)
    B = 1
    C = 96*3
    D, H, W = in_dim
    
    model = Bottleneck(C, depth, num_heads, mlp_ratio, dropout)

    x = torch.rand((B,  D * H * W, C))
    print(x.shape)
    # Forward pass through the model
    output = model(x)

    # Print the shape of the output tensor
    print(output.shape)  # Should print torch.Size([16, 50, 512])