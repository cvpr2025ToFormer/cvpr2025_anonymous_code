import pdb
import time

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from timm.models.layers import DropPath
import math
import torch.cuda
from torchinfo import summary
from thop import profile
from torch.autograd import Variable

from .branch3d import Branch3d_stage0
from .layer_factory import convbnrelu, conv_bn_relu
from .resnet_cbam import BasicBlock

class PositionalEncodingFourier(nn.Module):

    def __init__(self, hidden_dim=32, dim=768, temperature=10000):
        super().__init__()
        self.token_projection = nn.Conv2d(hidden_dim * 2, dim, kernel_size=1)
        self.scale = 2 * math.pi
        self.temperature = temperature
        self.hidden_dim = hidden_dim
        self.dim = dim

    def forward(self, B, H, W):
        mask = torch.zeros(B, H, W).bool().to(self.token_projection.weight.device)
        not_mask = ~mask
        y_embed = not_mask.cumsum(1, dtype=torch.float32)
        x_embed = not_mask.cumsum(2, dtype=torch.float32)
        eps = 1e-6
        y_embed = y_embed / (y_embed[:, -1:, :] + eps) * self.scale
        x_embed = x_embed / (x_embed[:, :, -1:] + eps) * self.scale

        dim_t = torch.arange(self.hidden_dim, dtype=torch.float32, device=mask.device)
        dim_t = self.temperature ** (2 * (dim_t // 2) / self.hidden_dim)

        pos_x = x_embed[:, :, :, None] / dim_t
        pos_y = y_embed[:, :, :, None] / dim_t
        pos_x = torch.stack((pos_x[:, :, :, 0::2].sin(),
                             pos_x[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos_y = torch.stack((pos_y[:, :, :, 0::2].sin(),
                             pos_y[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos = torch.cat((pos_y, pos_x), dim=3).permute(0, 3, 1, 2)
        pos = self.token_projection(pos)
        return pos


class XCA(nn.Module):

    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        q = q.transpose(-2, -1)
        k = k.transpose(-2, -1)
        v = v.transpose(-2, -1)

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).permute(0, 3, 1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'temperature'}


class LayerNorm(nn.Module):
    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError
        self.normalized_shape = (normalized_shape,)

    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x


class BNGELU(nn.Module):
    def __init__(self, nIn):
        super().__init__()
        self.bn = nn.BatchNorm2d(nIn, eps=1e-5)
        self.act = nn.GELU()

    def forward(self, x):
        output = self.bn(x)
        output = self.act(output)

        return output


class Conv(nn.Module):
    def __init__(self, nIn, nOut, kSize, stride, padding=0, dilation=(1, 1), groups=1, bn_act=False, bias=False):
        super().__init__()

        self.bn_act = bn_act

        self.conv = nn.Conv2d(nIn, nOut, kernel_size=kSize,
                              stride=stride, padding=padding,
                              dilation=dilation, groups=groups, bias=bias)

        if self.bn_act:
            self.bn_gelu = BNGELU(nOut)

    def forward(self, x):
        output = self.conv(x)

        if self.bn_act:
            output = self.bn_gelu(output)

        return output


class CDilated(nn.Module):

    def __init__(self, nIn, nOut, kSize, stride=1, d=1, groups=1, bias=False):
        """
        :param nIn: number of input channels
        :param nOut: number of output channels
        :param kSize: kernel size
        :param stride: optional stride rate for down-sampling
        :param d: optional dilation rate
        """
        super().__init__()
        padding = int((kSize - 1) / 2) * d
        self.conv = nn.Conv2d(nIn, nOut, kSize, stride=stride, padding=padding, bias=bias,
                              dilation=d, groups=groups)

    def forward(self, input):
        """
        :param input: input feature map
        :return: transformed feature map
        """

        output = self.conv(input)
        return output


class DilatedConv(nn.Module):

    def __init__(self, dim, k, dilation=1, stride=1, drop_path=0.,
                 layer_scale_init_value=1e-6, expan_ratio=6):
        """
        :param dim: input dimension
        :param k: kernel size
        :param dilation: dilation rate
        :param drop_path: drop_path rate
        :param layer_scale_init_value:
        :param expan_ratio: inverted bottelneck residual
        """

        super().__init__()

        self.ddwconv = CDilated(dim, dim, kSize=k, stride=stride, groups=dim, d=dilation)
        self.bn1 = nn.BatchNorm2d(dim)

        self.norm = LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, expan_ratio * dim)
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(expan_ratio * dim, dim)
        self.gamma = nn.Parameter(layer_scale_init_value * torch.ones(dim),
                                  requires_grad=True) if layer_scale_init_value > 0 else None
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        input = x

        x = self.ddwconv(x)
        x = self.bn1(x)

        x = x.permute(0, 2, 3, 1)  # (N, C, H, W) -> (N, H, W, C)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.gamma is not None:
            x = self.gamma * x
        x = x.permute(0, 3, 1, 2)  # (N, H, W, C) -> (N, C, H, W)

        x = input + self.drop_path(x)

        return x


class LGFI(nn.Module):

    def __init__(self, dim, drop_path=0., layer_scale_init_value=1e-6, expan_ratio=6,
                 use_pos_emb=True, num_heads=6, qkv_bias=True, attn_drop=0., drop=0.):
        super().__init__()

        self.dim = dim
        self.pos_embd = None
        if use_pos_emb:
            self.pos_embd = PositionalEncodingFourier(dim=self.dim)

        self.norm_xca = LayerNorm(self.dim, eps=1e-6)

        self.gamma_xca = nn.Parameter(layer_scale_init_value * torch.ones(self.dim),
                                      requires_grad=True) if layer_scale_init_value > 0 else None
        self.xca = XCA(self.dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)

        self.norm = LayerNorm(self.dim, eps=1e-6)
        self.pwconv1 = nn.Linear(self.dim, expan_ratio * self.dim)
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(expan_ratio * self.dim, self.dim)
        self.gamma = nn.Parameter(layer_scale_init_value * torch.ones((self.dim)),
                                  requires_grad=True) if layer_scale_init_value > 0 else None
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        input_ = x

        # XCA
        B, C, H, W = x.shape
        x = x.reshape(B, C, H * W).permute(0, 2, 1)

        if self.pos_embd:
            pos_encoding = self.pos_embd(B, H, W).reshape(B, -1, x.shape[1]).permute(0, 2, 1)
            x = x + pos_encoding

        x = x + self.gamma_xca * self.xca(self.norm_xca(x))

        x = x.reshape(B, H, W, C)

        # Inverted Bottleneck
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.gamma is not None:
            x = self.gamma * x
        x = x.permute(0, 3, 1, 2)  # (N, H, W, C) -> (N, C, H, W)

        x = input_ + self.drop_path(x)

        return x


class MaxPool(nn.Module):
    def __init__(self, ratio):
        super().__init__()
        self.pool = nn.ModuleList()
        for i in range(0, ratio):
            padding_num = 0
            self.pool.append(nn.MaxPool2d(kernel_size=2, stride=2, padding=padding_num))  #

    def forward(self, x):
        for pool in self.pool:
            x = pool(x)

        return x


class DepthEncoder(nn.Module):
    """
    ToFormer Encoder
    """

    def __init__(self, in_chans=4, model='toformer-v2-s', height=480, width=640,
                 global_block=None, global_block_type=None,
                 drop_path_rate=0.2, layer_scale_init_value=1e-6, expan_ratio=6,
                 heads=None, use_pos_embd_xca=None, **kwargs):

        super().__init__()

        self.height = height
        self.width = width

        if use_pos_embd_xca is None:
            use_pos_embd_xca = [True, False, False, False]
        if heads is None:
            heads = [8, 8, 8, 8]
        if global_block_type is None:
            global_block_type = ['LGFI', 'LGFI', 'LGFI', 'LGFI']
        if global_block is None:
            global_block = [1, 1, 1, 1]
        if model == 'toformer-v2-s':
            self.num_ch_enc = np.array([64, 64, 96, 128, 160])
            self.depth = [4, 4, 6, 8]
            self.init_dims = 64
            self.dims = [64, 96, 128, 160]
            self.dilation = [[1, 2, 3], [1, 2, 3], [1, 2, 1, 2, 3], [1, 2, 3, 1, 2, 4, 6]]
            self.dgcnn_feat_dim = [32, 0, 0, 0]

        self.branch3d_stage0 = Branch3d_stage0(k=20, in_chans=self.init_dims, feat_dim=self.dgcnn_feat_dim[0],
                                               feat_h=self.height // 2, feat_w=self.width // 2)

        self.rgb_init = conv_bn_relu(3, 48, kernel=3, stride=1, padding=1,
                                      bn=False)
        self.dep_init = conv_bn_relu(1, 16, kernel=3, stride=1, padding=1,
                                      bn=False)
        self.rgbd_fusion1 = conv_bn_relu(self.init_dims, self.init_dims, kernel=3, stride=1, padding=1,
                                  bn=False)
        self.rgbd_fusion2 = nn.Sequential(BasicBlock(self.init_dims, self.init_dims, ratio=16),
                              BasicBlock(self.init_dims, self.init_dims, ratio=16))

        self.downsample_layer0 = nn.Sequential(
            BasicBlock(self.init_dims, self.dims[0], stride=2, ratio=16),
            BasicBlock(self.dims[0], self.dims[0], stride=1, ratio=16)
        )

        self.downsample_layer1 = nn.Sequential(
            BasicBlock(self.dims[0] * 2 + in_chans + self.dgcnn_feat_dim[0], self.dims[1], stride=2, ratio=16),
        )

        self.downsample_layer2 = nn.Sequential(
            BasicBlock(self.dims[1] * 2 + in_chans + self.dgcnn_feat_dim[1], self.dims[2], stride=2, ratio=16),
        )

        self.downsample_layer3 = nn.Sequential(
            BasicBlock(self.dims[2] * 2 + in_chans + self.dgcnn_feat_dim[2], self.dims[3], stride=2, ratio=16),
        )

        self.input_downsample = nn.ModuleList()
        for i in range(1, 4):
            self.input_downsample.append(MaxPool(i))

        self.stages = nn.ModuleList()
        dp_rates = [x.item() for x in torch.linspace(0, drop_path_rate, sum(self.depth))]
        cur = 0
        for i in range(4):
            stage_blocks = []
            for j in range(self.depth[i]):
                if j > self.depth[i] - global_block[i] - 1:
                    if global_block_type[i] == 'LGFI':
                        stage_blocks.append(LGFI(dim=self.dims[i] + self.dgcnn_feat_dim[i], drop_path=dp_rates[cur + j],
                                                 expan_ratio=expan_ratio,
                                                 use_pos_emb=use_pos_embd_xca[i], num_heads=heads[i],
                                                 layer_scale_init_value=layer_scale_init_value,
                                                 ))

                    else:
                        raise NotImplementedError
                else:
                    stage_blocks.append(
                        DilatedConv(dim=self.dims[i] + self.dgcnn_feat_dim[i], k=3, dilation=self.dilation[i][j], drop_path=dp_rates[cur + j],
                                    layer_scale_init_value=layer_scale_init_value,
                                    expan_ratio=expan_ratio))

            self.stages.append(nn.Sequential(*stage_blocks))
            cur += self.depth[i]

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

        elif isinstance(m, (LayerNorm, nn.LayerNorm)):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)

    def forward_features(self, rgb, dep, pc):
        features = []
        x_down = []
        raw_rgbd = torch.cat((rgb, dep), dim=1)

        for i in range(3):
            x_down.append(self.input_downsample[i](raw_rgbd))

        feat_rgb = self.rgb_init(rgb)
        feat_dep = self.dep_init(dep)
        feat_rgbd = torch.cat((feat_rgb, feat_dep), dim=1)
        feat_rgbd = self.rgbd_fusion1(feat_rgbd)
        x = self.rgbd_fusion2(feat_rgbd)
        features.append(x)  # 1/1

        feat_3d_s0, index1 = self.branch3d_stage0(pc, x)

        x = self.downsample_layer0(x)  # 1/2 dim0
        tmp_x = [x]  # 1/2 dim0

        x = torch.cat((x, feat_3d_s0), dim=1)
        for s in range(len(self.stages[0]) - 1):
            x = self.stages[0][s](x)  # 1/2 dim0
        x = self.stages[0][-1](x)  # 1/2 dim0
        features.append(x)  # 1/2

        tmp_x.append(x)
        tmp_x.append(x_down[0])
        x = torch.cat(tmp_x, dim=1)
        x = self.downsample_layer1(x)  # 1/4 dim1

        tmp_x = [x]

        for s in range(len(self.stages[1]) - 1):
            x = self.stages[1][s](x)
        x = self.stages[1][-1](x)
        features.append(x)

        tmp_x.append(x)
        tmp_x.append(x_down[1])
        x = torch.cat(tmp_x, dim=1)
        x = self.downsample_layer2(x)  # 1/8 dim2

        tmp_x = [x]

        for s in range(len(self.stages[2]) - 1):
            x = self.stages[2][s](x)
        x = self.stages[2][-1](x)
        features.append(x)

        tmp_x.append(x)
        tmp_x.append(x_down[2])
        x = torch.cat(tmp_x, dim=1)
        x = self.downsample_layer3(x)  # 1/16 dim3

        for s in range(len(self.stages[3]) - 1):
            x = self.stages[3][s](x)
        x = self.stages[3][-1](x)
        features.append(x)

        return features

    def forward(self, rgb, dep, pc):
        out = self.forward_features(rgb, dep, pc)

        return out
