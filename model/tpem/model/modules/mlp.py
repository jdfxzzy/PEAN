import torch
import math
from torch import nn

class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)


class PositionalEncoding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, noise_level):
        count = self.dim // 2
        step = torch.arange(count, dtype=noise_level.dtype, device=noise_level.device) / count
        encoding = noise_level.unsqueeze(1) * torch.exp(-math.log(1e4) * step.unsqueeze(0))
        encoding = torch.cat([torch.sin(encoding), torch.cos(encoding)], dim=-1)
        return encoding             # (N, 1, 2*dim)


class FeatureWiseAffine(nn.Module):
    def __init__(self, in_channels, out_channels, use_affine_level=False):
        super(FeatureWiseAffine, self).__init__()
        self.use_affine_level = use_affine_level
        self.noise_func = nn.Linear(in_channels, out_channels*(1+self.use_affine_level))

    def forward(self, x, noise_embed):                                     
        batch = x.shape[0]
        if self.use_affine_level:
            gamma, beta = self.noise_func(noise_embed).view(batch, -1, 1).chunk(2, dim=1)
            x = (1 + gamma) * x + beta
        else:
            x = x + self.noise_func(noise_embed).view(batch, -1, 1)     # (N, 1, 26) -> (N, 1, out_channels)
        return x


class MlpBlock(nn.Module):
    def __init__(self, in_features, out_features, channels, act_layer=Swish, drop=0.):
        super().__init__()
        self.bn = nn.BatchNorm1d(2*channels)
        self.fc = nn.Linear(in_features, out_features)
        self.act = act_layer()
        self.drop = nn.Dropout(drop)
        self.noise_func = FeatureWiseAffine(2*channels, 2*channels)

    def forward(self, x, t):
        x = self.bn(x)                  
        x = self.fc(x)
        x = self.act(x)
        x = self.drop(x)                # (N, 26, out_features)
        x = self.noise_func(x, t)       # (N, 26, out_features)
        return x


class Mlp(nn.Module):
    def __init__(self, in_features, out_features, channels, act_layer=Swish, drop=0.):
        super().__init__()
        self.conv = nn.Conv1d(2*channels, channels, 1, 1, 0)
        self.block_1 = MlpBlock(in_features, 4*in_features, channels//2, act_layer, drop)
        self.block_2 = MlpBlock(4*in_features, 8*in_features, channels//2, act_layer, drop)
        self.block_3 = MlpBlock(8*in_features, 4*in_features, channels//2, act_layer, drop)
        self.block_4 = MlpBlock(4*in_features, out_features, channels//2, act_layer, drop)
        self.noise_level_mlp = nn.Sequential(
                PositionalEncoding(channels),
                nn.Linear(channels, channels * 4),
                Swish(),
                nn.Linear(channels * 4, channels)
            )

    def forward(self, x, t):
        t = self.noise_level_mlp(t)     # (N, 1) -> (N, 1, 26)
        x = self.conv(x)                # (N, 52, 95) -> (N, 26, 95)
        x = self.block_1(x, t)
        x = self.block_2(x, t)
        x = self.block_3(x, t)
        x = self.block_4(x, t)
        return x