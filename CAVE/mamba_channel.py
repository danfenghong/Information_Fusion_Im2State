import math
# from timm.models.layers import DropPath
from mamba_ssm.ops.selective_scan_interface import selective_scan_fn

# DropPath.__repr__ = lambda self: f"timm.DropPath({self.drop_prob})"

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
from einops.layers.torch import Rearrange

class Spa_SSM1(nn.Module):
    def __init__(
            self,
            d_model,
            d_state=16,
            expand=4,
            dt_rank="auto",
            dt_min=0.001,
            dt_max=0.1,
            dt_init="random",
            dt_scale=1.0,
            dt_init_floor=1e-4,
            dropout=0.,
            bias=False,
            device=None,
            dtype=None,
            **kwargs,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.fuse = nn.Sequential(nn.Linear(2*d_model,d_model),nn.GELU())#,
                                      #nn.Linear(d_model,d_model)
        self.d_model = d_model  #
        self.d_state = d_state
        self.expand = expand
        self.d_inner = int(self.expand * self.d_model)
        self.dt_rank = 8 #math.ceil(self.d_model / 16) if dt_rank == "auto" else dt_rank

        self.in_proj = nn.Linear(self.d_model, self.d_inner * 2, bias=bias, **factory_kwargs)

        self.pan_in_proj = nn.Linear(self.d_model, self.d_inner * 2, bias=bias, **factory_kwargs)

        self.dt_proj = (
            nn.Linear(self.d_inner, self.dt_rank, bias=False, **factory_kwargs),
        )
        self.dt_proj_weight = nn.Parameter(torch.stack([t.weight for t in self.dt_proj], dim=0))  # (K=4, N, inner)
        del self.dt_proj

        self.Cs_proj = (
            nn.Linear(self.d_inner, self.d_state, bias=False, **factory_kwargs),
        )
        self.Cs_proj_weight = nn.Parameter(torch.stack([t.weight for t in self.Cs_proj], dim=0))  # (K=4, N, inner)
        del self.Cs_proj

        self.Bs_proj = (
            nn.Linear(self.d_inner, self.d_state, bias=False, **factory_kwargs),
        )
        self.Bs_proj_weight = nn.Parameter(torch.stack([t.weight for t in self.Bs_proj], dim=0))  # (K=4, N, inner)
        del self.Bs_proj

        self.dt_projs = (
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor,
                         **factory_kwargs),
        )  # 偏置尺寸等于原始影像
        self.dt_projs_weight = nn.Parameter(torch.stack([t.weight for t in self.dt_projs], dim=0))  # (K=4, inner, rank)
        self.dt_projs_bias = nn.Parameter(torch.stack([t.bias for t in self.dt_projs], dim=0))  # (K=4, inner)
        del self.dt_projs
        #
        self.A_logs = self.A_log_init(self.d_state, self.d_inner, copies=1, merge=True)  # (K=4, D, N)
        self.Ds = self.D_init(self.d_inner, copies=1, merge=True)  # (K=4, D, N)

        self.selective_scan = selective_scan_fn

        self.out_norm = nn.LayerNorm(self.d_inner)
        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=bias, **factory_kwargs)
        self.dropout = nn.Dropout(dropout) if dropout > 0. else None

    @staticmethod
    def dt_init(dt_rank, d_inner, dt_scale=1.0, dt_init="random", dt_min=0.001, dt_max=0.1, dt_init_floor=1e-4,
                **factory_kwargs):
        dt_proj = nn.Linear(dt_rank, d_inner, bias=True, **factory_kwargs)

        # Initialize special dt projection to preserve variance at initialization
        dt_init_std = dt_rank ** -0.5 * dt_scale
        if dt_init == "constant":
            nn.init.constant_(dt_proj.weight, dt_init_std)
        elif dt_init == "random":
            nn.init.uniform_(dt_proj.weight, -dt_init_std, dt_init_std)
        else:
            raise NotImplementedError

        # Initialize dt bias so that F.softplus(dt_bias) is between dt_min and dt_max
        dt = torch.exp(
            torch.rand(d_inner, **factory_kwargs) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        # Inverse of softplus: https://github.com/pytorch/pytorch/issues/72759
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            dt_proj.bias.copy_(inv_dt)
        # Our initialization would set all Linear.bias to zero, need to mark this one as _no_reinit
        dt_proj.bias._no_reinit = True

        return dt_proj

    @staticmethod
    def A_log_init(d_state, d_inner, copies=1, device=None, merge=True):
        # S4D real initialization
        A = repeat(
            torch.arange(1, d_state + 1, dtype=torch.float32, device=device),
            "n -> d n",
            d=d_inner,
        ).contiguous()
        A_log = torch.log(A)  # Keep A_log in fp32
        if copies > 1:
            A_log = repeat(A_log, "d n -> r d n", r=copies)
            if merge:
                A_log = A_log.flatten(0, 1)
        A_log = nn.Parameter(A_log)
        A_log._no_weight_decay = True
        return A_log

    @staticmethod
    def D_init(d_inner, copies=1, device=None, merge=True):
        # D "skip" parameter
        D = torch.ones(d_inner, device=device)
        if copies > 1:
            D = repeat(D, "n1 -> r n1", r=copies)
            if merge:
                D = D.flatten(0, 1)
        D = nn.Parameter(D)  # Keep in fp32
        D._no_weight_decay = True
        return D

    def forward_core(self, x: torch.Tensor, pan: torch.Tensor):  #
        B, C, H, W = x.shape  # 输入影像尺寸
        L = H * W  # 将空间当作token，将不同空间排列视为token排序，光谱值作为编码值，但是光谱信息在第二维（transformer中token是才是第二维）
        K = 1  # 4

        x_hwwh = x.view(B, -1, L).view(B, 1, -1, L)  # 两个不同排列，左右上下，上下左右
        xs = x_hwwh  # (b, k, d, l) # 形成四个排列，左右上下，上下左右，右左下上，下上右左

        pan_hwwh = pan.view(B, -1, L).view(B, 1, -1, L)  # 两个不同排列，左右上下，上下左右
        pans = pan_hwwh  # (b, k, d, l) # 形成四个排列，左右上下，上下左右，右左下上，下上右左

        # K = 2  # 4

        # x_hwwh = torch.stack([x.view(B, -1, L), torch.transpose(x, dim0=2, dim1=3).contiguous().view(B, -1, L)], dim=1).view(B, 2, -1, L)  # 两个不同排列，左右上下，上下左右
        # xs = x_hwwh  # (b, k, d, l) # 形成四个排列，左右上下，上下左右，右左下上，下上右左

        # pan_hwwh = torch.stack([pan.view(B, -1, L), torch.transpose(pan, dim0=2, dim1=3).contiguous().view(B, -1, L)], dim=1).view(B, 2, -1, L)  # 两个不同排列，左右上下，上下左右
        # pans = pan_hwwh  # (b, k, d, l) # 形成四个排列，左右上下，上下左右，右左下上，下上右左


        Cs = pans
        Cs = torch.einsum("b k d l, k c d -> b k c l", Cs.view(B, K, -1, L),
                          self.Cs_proj_weight)  # 四个序列，利用线性变换进行四个不同的映射，注意此时光谱（第三个维度）变成dt+2*state
        dts = xs
        dts = torch.einsum("b k d l, k c d -> b k c l", dts.view(B, K, -1, L),
                           self.dt_proj_weight)  # 四个序列，利用线性变换进行四个不同的映射，注意此时光谱（第三个维度）变成dt+2*state

        dts = torch.einsum("b k r l, k d r -> b k d l", dts.view(B, K, -1, L), self.dt_projs_weight)  # dt视为偏置

        Bs = xs
        Bs = torch.einsum("b k d l, k c d -> b k c l", Bs.view(B, K, -1, L),
                          self.Bs_proj_weight)  # 四个序列，利用线性变换进行四个不同的映射，注意此时光谱（第三个维度）变成dt+2*state
        xs = xs.float().view(B, -1, L)  # (b, k * d, l)
        dts = dts.contiguous().float().view(B, -1, L)  # (b, k * d, l)
        Bs = Bs.float().view(B, K, -1, L)  # (b, k, d_state, l)
        Cs = Cs.float().view(B, K, -1, L)  # (b, k, d_state, l)
        Ds = self.Ds.float().view(-1)  # (k * d)
        As = -torch.exp(self.A_logs.float()).view(-1, self.d_state)  # (k * d, d_state)
        dt_projs_bias = self.dt_projs_bias.float().view(-1)  # (k * d)
        # ht=A*ht+B*xs:b k*d k d_state
        # y = C*ht+D*xt: b k*d l
        out_y = self.selective_scan(
            xs, dts,
            As, Bs, Cs, Ds, z=None,
            delta_bias=dt_projs_bias,
            delta_softplus=True,
            return_last_state=False,
        ).view(B, K, -1, L)  # 恢复复形状
        assert out_y.dtype == torch.float
        # wh_y = torch.transpose(out_y[:, 1].view(B, -1, W, H), dim0=2, dim1=3).contiguous().view(B, -1, L)
        return out_y[:, 0]#+wh_y  # 左右上下，右左下上，上下左右，下上右左

    def forward(self, x: torch.Tensor, pan: torch.Tensor, **kwargs):  #

        B, H, W, C = x.shape  # 输入影像尺寸

        x = self.fuse(torch.cat([x, pan],3))#.permute(0, 3, 1, 2)

        xz = self.in_proj(x)  # 首先利用线性变换将原始维度转换为mamba内部维度
        x, z = xz.chunk(2, dim=-1)  # (b, h, w, d) 将x分割成两块一块留下备用
        x = x.permute(0, 3, 1, 2).contiguous()

        panpanz = self.pan_in_proj(pan)
        pan, panz = panpanz.chunk(2, dim=-1)  # (b, h, w, d) 将x分割成两块一块留下备用
        pan = pan.permute(0, 3, 1, 2).contiguous()

        y = self.forward_core(x, pan)  # 进行扫描, rgb
        y = torch.transpose(y, dim0=1, dim1=2).contiguous().view(B, H, W, -1)  # 将h*w展开
        y = self.out_norm(y)  # 层归一化
        y = y * F.silu(z) + y * F.silu(panz)  # 留下的另一块形成权重进行加权

        out = self.out_proj(y) # 恢复成原来维度
        if self.dropout is not None:
            out = self.dropout(out)
        
        return out
