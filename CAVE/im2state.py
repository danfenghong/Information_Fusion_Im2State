import torch.nn as nn
import torch
import torch.nn.functional as F
from einops.layers.torch import Rearrange
import math
import warnings
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[2]))
# from mamba_p import Spa_SSM1
# from mamba_channel111111 import Spa_SSM1
# from mamba_channelwoall_inf_ouf import Spa_SSM1
from mamba_channel import Spa_SSM1

def _no_grad_trunc_normal_(tensor, mean, std, a, b):
    def norm_cdf(x):
        return (1. + math.erf(x / math.sqrt(2.))) / 2.

    if (mean < a - 2 * std) or (mean > b + 2 * std):
        warnings.warn("mean is more than 2 std from [a, b] in nn.init.trunc_normal_. "
                      "The distribution of values may be incorrect.",
                      stacklevel=2)
    with torch.no_grad():
        l = norm_cdf((a - mean) / std)
        u = norm_cdf((b - mean) / std)
        tensor.uniform_(2 * l - 1, 2 * u - 1)
        tensor.erfinv_()
        tensor.mul_(std * math.sqrt(2.))
        tensor.add_(mean)
        tensor.clamp_(min=a, max=b)
        return tensor


def pair(t):
    return t if isinstance(t, tuple) else (t, t)


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm = nn.LayerNorm(dim)

    def forward(self, x, *args, **kwargs):
        x = self.norm(x)
        return self.fn(x, *args, **kwargs)


class GELU(nn.Module):
    def forward(self, x):
        return F.gelu(x)


def conv(in_channels, out_channels, kernel_size, bias=False, padding=1, stride=1):
    return nn.Conv2d(
        in_channels, out_channels, kernel_size,
        padding=(kernel_size // 2), bias=bias, stride=stride)


# ----------------------------------------
#       Spe_SSM Block
# ----------------------------------------
class ESSM(nn.Module):  # [bs, 28, 256, 256]
    def __init__(
            self,
            dim,
    ):
        super().__init__()
        self.Spa_SSM = Spa_SSM1(
            # This module uses roughly 3 * expand * d_model^2 parameters
            d_model=dim,  # Model dimension d_model  dimself.token_size * self.token_size * zipdim dim
            d_state=16,  # SSM state expansion factor # 64
            expand=2,  # Block expansion factor
            use_fast_path=False,
        )
        self.conv_v = nn.Conv2d(in_channels=2 * dim, out_channels=dim, kernel_size=3, padding=1)  # 3,1
        self.BN = nn.BatchNorm2d(dim)
        self.v_out_head = nn.Sequential(
            Rearrange('b h w c -> b c h w'),
            nn.Conv2d(dim, dim, 3, 1, 1, bias=False, groups=dim),
            nn.GELU(),  # 0531
            # nn.ReLU(),
            # nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Conv2d(dim, dim, 3, 1, 1, bias=False, groups=dim),
            Rearrange('b c h w -> b h w c'),
        )
        self.v_in_head = nn.Sequential(
            # nn.LayerNorm(dim),
            Rearrange('b h w c -> b c h w'),
            nn.Conv2d(dim, dim, 1, 1, bias=False, groups=1),
            nn.Conv2d(dim, dim, 3, 1, 1, bias=False, groups=dim),
            nn.GELU(),  # 0531
            # nn.ReLU(),
            # nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Conv2d(dim, dim, 1, 1, bias=False, groups=1),  # 0531
            Rearrange('b c h w -> b h w c'),
        )

    def forward(self, x, pan):
        """
        x_in:
        return out:
        """
        b, h, w, c = x.shape
        # 0522
        x = x.permute(0, 3, 1, 2).contiguous()
        # #普通的融合
        x = torch.concat((x, pan), dim=1)  # K
        x0 = self.BN(self.conv_v(x)).permute(0, 2, 3, 1)
        x = self.v_in_head(x0)
        x = self.v_out_head(x) + x
        x = self.Spa_SSM(x, pan.permute(0, 2, 3, 1)) + x

        return x  # x


# -------------------------------------------------------
#           FeedForward
# ------------------------------------------------------
class FeedForward(nn.Module):
    def __init__(self, dim, mult=4):
        super().__init__()
        self.FF = nn.Sequential(
            nn.Conv2d(dim, dim * mult, 1, 1, bias=False),
            GELU(),
            nn.Conv2d(dim * mult, dim * mult, 3, 1, 1, bias=False, groups=dim * mult),
            GELU(),
            nn.Conv2d(dim * mult, dim, 1, 1, bias=False),
        )

    def forward(self, x):
        """
        x: [b,h,w,c]
        return out: [b,h,w,c]
        """
        out = self.FF(x.permute(0, 3, 1, 2))
        return out.permute(0, 2, 3, 1)


# -------------------------------------------------------
#           Cascade  Transformer
# ------------------------------------------------------
class CambaBlock(nn.Module):
    def __init__(
            self, dim, num_blocks=1
    ):
        super().__init__()
        self.blocks = nn.ModuleList([])
        for _ in range(num_blocks):
            self.blocks.append(nn.ModuleList([
                ESSM(dim=dim),  # spectral
                PreNorm(dim, FeedForward(dim=dim))
            ]))

    def forward(self, x, x_pan):
        """
        x: [b,c,h,w]
        return out: [b,c,h,w]
        """
        x = x.permute(0, 2, 3, 1)
        # mask = mask.permute(0, 2, 3, 1)
        for (attn1, ff) in self.blocks:
            x = attn1(x, pan=x_pan) + x  # Xspe
            x = ff(x) + x  # X_all
        out = x.permute(0, 3, 1, 2)
        return out


class Upsample(nn.Sequential):
    def __init__(self, scale, num_feat):
        m = []
        if (scale & (scale - 1)) == 0:  # scale = 2^n
            for _ in range(int(math.log(scale, 2))):
                m.append(nn.Conv2d(num_feat, 4 * num_feat, 3, 1, 1))
                m.append(nn.PixelShuffle(2))
        elif scale == 3:
            m.append(nn.Conv2d(num_feat, 9 * num_feat, 3, 1, 1))
            m.append(nn.PixelShuffle(3))
        else:
            raise ValueError(f'scale {scale} is not supported. ' 'Supported scales: 2^n and 3.')
        super(Upsample, self).__init__(*m)


class Downsample(nn.Sequential):
    def __init__(self, scale, num_feat):
        m = []
        if (scale & (scale - 1)) == 0:  # scale = 2^n
            for _ in range(int(math.log(scale, 2))):
                m.append(nn.Conv2d(num_feat, num_feat // 4, 3, 1, 1))
                m.append(nn.PixelUnshuffle(2))
        elif scale == 3:
            m.append(nn.Conv2d(num_feat, num_feat // 9, 3, 1, 1))
            m.append(nn.PixelUnshuffle(3))
        else:
            raise ValueError(f'scale {scale} is not supported. ' 'Supported scales: 2^n and 3.')
        super(Downsample, self).__init__(*m)


class msi_y(nn.Module):
    def __init__(self, band=28,n_fts=28):
        super(msi_y, self).__init__()
        # Define number of input channels
        self.band = band
        self.n_fts = n_fts

        lv1_c = int(n_fts)
        lv2_c = int(n_fts)  # * 2
        lv4_c = int(n_fts)  # * 4
        # 3 256 256 ->28 256 256
        self.layer_1 = nn.Sequential(nn.Conv2d(in_channels=self.band, out_channels=lv1_c, kernel_size=3, padding=1),
                                     nn.LeakyReLU(negative_slope=0.2),
                                     )
        # 3 256 256 -> 56 128 128
        self.layer_2 = nn.Sequential(nn.Conv2d(in_channels=self.band, out_channels=lv2_c, kernel_size=3, padding=1),
                                     nn.LeakyReLU(negative_slope=0.2),
                                     Downsample(scale=2, num_feat=lv2_c),
                                     )
        # 3 256 256 -> 112 64 64
        self.layer_3 = nn.Sequential(nn.Conv2d(in_channels=self.band, out_channels=lv2_c, kernel_size=3, padding=1),
                                     nn.LeakyReLU(negative_slope=0.2),
                                     Downsample(scale=2, num_feat=lv2_c),
                                     nn.Conv2d(in_channels=lv2_c, out_channels=lv4_c, kernel_size=3, padding=1),
                                     nn.LeakyReLU(negative_slope=0.2),  # 0523
                                     Downsample(scale=2, num_feat=lv4_c),
                                     )

    def forward(self, x_rgb):
        B, C, H, W = x_rgb.shape
        x_rgb = x_rgb.view(B, C, -1)
        x_rgb = F.interpolate(x_rgb.permute(0,2,1),        # 变成 [1,1,3,H*W]
                      size=self.band,
                      mode='linear',
                      align_corners=True)      # 输出 [1,1,52,H*W]
        x_rgb = x_rgb.permute(0,2,1).view(B, self.band, H, W)     # 恢复形状
        x1 = self.layer_1(x_rgb)
        # x1 = x_rgb
        x2 = self.layer_2(x_rgb)
        x3 = self.layer_3(x_rgb)

        return [x1, x2, x3]

class hsi_y(nn.Module):
    def __init__(self, scale=4, band=28, n_fts=28):
        super(hsi_y, self).__init__()
        # Define number of input channels
        self.scale = scale
        self.band = band
        self.n_fts = n_fts

        lv1_c = int(n_fts)
        lv2_c = int(n_fts)  # * 2
        lv4_c = int(n_fts)  # * 4
        # 3 256 256 ->28 256 256
        self.layer_1 = nn.Sequential(nn.Conv2d(in_channels=self.band, out_channels=lv1_c, kernel_size=3, padding=1),
                                     nn.LeakyReLU(negative_slope=0.2),
                                     )
        # 3 256 256 -> 56 128 128
        self.layer_2 = nn.Sequential(nn.Conv2d(in_channels=self.band, out_channels=lv2_c, kernel_size=3, padding=1),
                                     nn.LeakyReLU(negative_slope=0.2),
                                     Downsample(scale=2, num_feat=lv2_c),
                                     )
        # 3 256 256 -> 112 64 64
        self.layer_3 = nn.Sequential(nn.Conv2d(in_channels=self.band, out_channels=lv2_c, kernel_size=3, padding=1),
                                     nn.LeakyReLU(negative_slope=0.2),
                                     Downsample(scale=2, num_feat=lv2_c),
                                     nn.Conv2d(in_channels=lv2_c, out_channels=lv4_c, kernel_size=3, padding=1),
                                     nn.LeakyReLU(negative_slope=0.2),  # 0523
                                     Downsample(scale=2, num_feat=lv4_c),
                                     )

    def forward(self, x):
        B, C, H, W = x.shape
        x = nn.functional.interpolate(input=x, scale_factor=self.scale, mode='bicubic', align_corners=False)
        
        x1 = self.layer_1(x)
        # x1 = x_rgb
        x2 = self.layer_2(x)
        x3 = self.layer_3(x)

        return [x1, x2, x3]

class im2state(nn.Module):
    def __init__(self, dim=64, band=8, scale=4, num_blocks=None):  #
        super(im2state, self).__init__()

        if num_blocks is None:
            num_blocks = [1, 1, 1]

        self.dim = dim
        self.stage = 2
        self.scale = scale
        # Input projection
        self.embedding = nn.Conv2d(band, self.dim, 3, 1, 1, bias=False)

        # Input projection
        self.embedding1 = nn.Conv2d(4, self.dim, 3, 1, 1, bias=False)

        dim_stage = dim

        # Encoder
        self.encoder_layers = nn.ModuleList([])
        for i in range(self.stage):
            self.encoder_layers.append(nn.ModuleList([
                CambaBlock(dim=dim_stage,num_blocks=num_blocks[i]),  # dim_stage // dim
                CambaBlock(dim=dim_stage,num_blocks=num_blocks[i]),  # dim_stage // dim
                Downsample(scale=2, num_feat=self.dim),
                nn.Conv2d(dim_stage * 2, dim_stage, 1, 1, bias=False),
            ]))
        # dim_stage

        # Bottleneck
        self.bottleneck = CambaBlock(dim=dim_stage,num_blocks=num_blocks[-1])
        self.bottleneck1 = CambaBlock(dim=dim_stage,num_blocks=num_blocks[-1])

        # Decoder
        self.decoder_layers = nn.ModuleList([])
        for i in range(self.stage):
            self.decoder_layers.append(nn.ModuleList([
                Upsample(scale=2, num_feat=dim_stage),
                nn.Conv2d(dim_stage * 2, dim_stage, 1, 1, bias=False),
                CambaBlock(dim=dim_stage,num_blocks=num_blocks[1-i]),
                CambaBlock(dim=dim_stage,num_blocks=num_blocks[1-i])]))

        # Output projection
        self.mapping = nn.Conv2d(self.dim, band, 3, 1, 1, bias=False)
        self.mapping1 = nn.Conv2d(self.dim, band, 3, 1, 1, bias=False)

        # activation function
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)  # LeakyReLU

        self.msi = msi_y(band=band, n_fts=self.dim)
        self.hsi = hsi_y(scale=self.scale, band=band, n_fts=self.dim)
        self.a = nn.Parameter(torch.tensor(0.5), requires_grad=True)

    def forward(self, x, x_pan):
        """
        x: [b,c,h,w] #have been masked from a pan
        return out:[b,c,h,w]
        """
        msi_list = self.msi(x_pan)  # rgb_list
        hsi_list = self.hsi(x)  # rgb_list
        # Embedding
        fea = nn.functional.interpolate(input=x, scale_factor=self.scale, mode='bicubic', align_corners=False)
        fea = self.lrelu(self.embedding(fea))

        fea1 = self.lrelu(self.embedding1(x_pan))
        
        x1 = fea
        x2 = fea1
        # Encoder
        fea_encoder = []
        fea1_encoder = []
        for i, (CambaBlock, CambaBlock1, FeaDownSample, Fution) in enumerate(self.encoder_layers):
            fea = CambaBlock(fea, msi_list[i])  # casformer don't change mask
            fea1 = CambaBlock1(fea1, hsi_list[i])  # casformer don't change mask
            fea_encoder.append(fea)
            fea1_encoder.append(fea1)
            fea = FeaDownSample(fea)  # 2DCNN,stride=2,become downsample
            fea1 = FeaDownSample(fea1)  # 2DCNN,stride=2,become downsample
            # if self.stage - 1 - 1 - i > -1:
            #     fea = Fution(torch.cat([fea, Ufea_encoder[self.stage - 1 - 1 - i]], dim=1))

        # Bottleneck
        fea = self.bottleneck(fea, msi_list[2])
        fea1 = self.bottleneck1(fea1, hsi_list[2])

        # Decoder  RGB_list[1-i]
        for i, (FeaUpSample, Fution, LeWinBlcok,LeWinBlcok1) in enumerate(self.decoder_layers):
            fea = FeaUpSample(fea)
            fea1 = FeaUpSample(fea1)
            fea = Fution(torch.cat([fea, fea_encoder[self.stage - 1 - i]], dim=1))
            fea1 = Fution(torch.cat([fea1, fea1_encoder[self.stage - 1 - i]], dim=1))
            fea = LeWinBlcok(fea, msi_list[1 - i])
            fea1 = LeWinBlcok1(fea1, hsi_list[1 - i])

        # Mapping
        out = self.mapping(fea + x1)  #
        out1 = self.mapping1(fea1 + x2)  #
        out_all = self.a*out+(1-self.a)*out1

        return out, out1, out_all
