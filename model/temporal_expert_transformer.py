#  Copyright (c) 2024. IPCRC, Lab. Jiangnig Wei
#  All rights reserved

import math
from einops import rearrange
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
from model.angular_feature import Angular_feature
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
import torch.utils.checkpoint as checkpoint


def import_class(name):
    components = name.split('.')
    mod = __import__(components[0])
    for comp in components[1:]:
        mod = getattr(mod, comp)
    return mod


def conv_branch_init(conv, branches):
    weight = conv.weight
    n = weight.size(0)
    k1 = weight.size(1)
    k2 = weight.size(2)
    nn.init.normal_(weight, 0, math.sqrt(2. / (n * k1 * k2 * branches)))
    nn.init.constant_(conv.bias, 0)


def conv_init(conv):
    nn.init.kaiming_normal_(conv.weight, mode='fan_out')
    nn.init.constant_(conv.bias, 0)


def bn_init(bn, scale):
    nn.init.constant_(bn.weight, scale)
    nn.init.constant_(bn.bias, 0)


class unit_tcn(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=9, stride=1):
        super(unit_tcn, self).__init__()

        pad = int((kernel_size - 1) / 2)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=(kernel_size, 1), padding=(pad, 0),
                              stride=(stride, 1))

        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()
        conv_init(self.conv)
        bn_init(self.bn, 1)

    def forward(self, x):
        x = self.bn(self.conv(x))
        return x


class unit_tcn_m(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, kernel_size=[1, 3, 7]):  # ks=9 initial
        super(unit_tcn_m, self).__init__()

        pad1 = int((kernel_size[0] - 1) / 2)
        pad2 = int((kernel_size[1] - 1) / 2)
        pad3 = int((kernel_size[2] - 1) / 2)

        mid_channels = out_channels // 3

        self.conv11 = nn.Conv2d(in_channels, in_channels, kernel_size=(1, 1))
        self.conv21 = nn.Conv2d(in_channels, in_channels, kernel_size=(1, 1))
        self.conv31 = nn.Conv2d(in_channels, in_channels, kernel_size=(1, 1))

        self.conv12 = nn.Conv2d(in_channels, mid_channels, kernel_size=(kernel_size[0], 1), padding=(pad1, 0),
                                stride=(stride, 1))
        self.conv22 = nn.Conv2d(in_channels, mid_channels, kernel_size=(kernel_size[1], 1), padding=(pad2, 0),
                                stride=(stride, 1))
        self.conv32 = nn.Conv2d(in_channels, mid_channels, kernel_size=(kernel_size[2], 1), padding=(pad3, 0),
                                stride=(stride, 1))

        self.bn = nn.BatchNorm2d(out_channels)
        conv_init(self.conv11)
        conv_init(self.conv21)
        conv_init(self.conv31)
        conv_init(self.conv12)
        conv_init(self.conv22)
        conv_init(self.conv32)
        bn_init(self.bn, 1)

    def forward(self, x):
        x1 = self.conv12(self.conv11(x))
        x2 = self.conv22(self.conv21(x))
        x3 = self.conv32(self.conv31(x))
        x = torch.cat([x1, x2, x3], dim=1)
        x = self.bn(x)
        return x


class my_simple_gcn_unit(nn.Module):
    def __init__(self, in_channels, out_channels, A, angular_channels=9, T=100, V=17):
        super().__init__()
        self.A = Variable(torch.from_numpy(A.astype(np.float32)), requires_grad=False)
        self.conv = nn.Conv2d(in_channels, out_channels, 1)
        self.relu = nn.ReLU()
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        # batchsize, channel, t, node_num
        N, C, T, V = x.size()

        ## feature parameter embedding
        # x = rearrange(x, 'n c t v  -> (n t v) c')
        # x += self.feature_embedding
        # x = rearrange(x, '(n t v) c -> n c t v', n=N, t=T, v=V)
        # # embedding = x[:, (self.angular_channels + 1) * (-1): -1, :, :] + self.angular_embedding
        # # x[:, (self.angular_channels + 1) * (-1): -1, :, :] = embedding

        A = self.A.cuda(x.get_device())  # A V*V
        support = torch.einsum('vu,nctu->nctv', A, x)  # N, C, T, V
        support = self.conv(x)
        support = self.relu(support)
        return self.bn(support)


class unit_gcn(nn.Module):
    def __init__(self, in_channels, out_channels, A, coff_embedding=4, num_subset=3):
        super(unit_gcn, self).__init__()
        inter_channels = out_channels // coff_embedding
        self.inter_c = inter_channels
        self.PA = nn.Parameter(torch.from_numpy(A.astype(np.float32)))
        nn.init.constant_(self.PA, 1e-6)
        self.A = Variable(torch.from_numpy(A.astype(np.float32)), requires_grad=False)
        self.num_subset = num_subset

        self.conv_a = nn.ModuleList()
        self.conv_b = nn.ModuleList()
        self.conv_d = nn.ModuleList()
        for i in range(self.num_subset):
            self.conv_a.append(nn.Conv2d(in_channels, inter_channels, 1))
            self.conv_b.append(nn.Conv2d(in_channels, inter_channels, 1))
            self.conv_d.append(nn.Conv2d(in_channels, out_channels, 1))

        if in_channels != out_channels:
            self.down = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.down = lambda x: x

        self.bn = nn.BatchNorm2d(out_channels)
        self.soft = nn.Softmax(-2)
        self.relu = nn.ReLU()

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                conv_init(m)
            elif isinstance(m, nn.BatchNorm2d):
                bn_init(m, 1)
        bn_init(self.bn, 1e-6)
        for i in range(self.num_subset):
            conv_branch_init(self.conv_d[i], self.num_subset)

    def forward(self, x):
        # batchsize, channel, t, node_num
        N, C, T, V = x.size()
        # print(N, C, T, V)
        A = self.A.cuda(x.get_device())
        A = A + self.PA

        y = None
        for i in range(self.num_subset):  # N V C T
            A1 = self.conv_a[i](x).permute(0, 3, 1, 2).contiguous().view(N, V, self.inter_c * T)  # N V ic*T
            A2 = self.conv_b[i](x).view(N, self.inter_c * T, V)  # N ic*T V
            A1 = self.soft(torch.matmul(A1, A2) / A1.size(-1))  # N V V
            A1 = A1 + A[i]
            A2 = x.view(N, C * T, V)
            z = self.conv_d[i](torch.matmul(A2, A1).view(N, C, T, V))
            y = z + y if y is not None else z

        y = self.bn(y)
        y += self.down(x)
        return self.relu(y)


class TCN_GCN_unit(nn.Module):

    def __init__(self, in_channels, out_channels, A, stride=1, residual=True):
        super(TCN_GCN_unit, self).__init__()
        self.gcn1 = unit_gcn(in_channels, out_channels, A)
        # self.gcn1 = my_simple_gcn_unit(in_channels, out_channels, A)
        self.tcn1 = unit_tcn_m(out_channels, out_channels, stride=stride)
        self.relu = nn.ReLU()
        if not residual:
            self.residual = lambda x: 0

        elif (in_channels == out_channels) and (stride == 1):
            self.residual = lambda x: x

        else:
            self.residual = unit_tcn(in_channels, out_channels, kernel_size=1, stride=stride)

    def forward(self, x):
        x = self.tcn1(self.gcn1(x)) + self.residual(x)
        return self.relu(x)


class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(x, **kwargs) + x


class LayerNormalize(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)


class MLP_Block(nn.Module):
    def __init__(self, dim, hid_dim, dropout=0.1):
        super().__init__()
        self.nn1 = nn.Linear(dim, hid_dim)
        torch.nn.init.xavier_uniform_(self.nn1.weight)
        torch.nn.init.normal_(self.nn1.bias, std=1e-6)
        self.af1 = nn.ReLU()
        self.do1 = nn.Dropout(dropout)
        self.nn2 = nn.Linear(hid_dim, dim)
        torch.nn.init.xavier_uniform_(self.nn2.weight)
        torch.nn.init.normal_(self.nn2.bias, std=1e-6)
        self.do2 = nn.Dropout(dropout)

    def forward(self, x):
        x = self.nn1(x)
        x = self.af1(x)
        x = self.do1(x)
        x = self.nn2(x)
        x = self.do2(x)

        return x


class Attention(nn.Module):
    def __init__(self, dim, out_dim, heads=3, dropout=0.1):
        super().__init__()
        self.heads = heads
        self.scale = dim ** -0.5  # 1/sqrt(dim)

        self.to_qkv = nn.Linear(dim, dim * 3, bias=True)
        torch.nn.init.xavier_uniform_(self.to_qkv.weight)
        torch.nn.init.zeros_(self.to_qkv.bias)

        self.nn1 = nn.Linear(dim, out_dim)
        torch.nn.init.xavier_uniform_(self.nn1.weight)
        torch.nn.init.zeros_(self.nn1.bias)
        self.do1 = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        b, n, _, h = *x.shape, self.heads
        qkv = self.to_qkv(x)  # gets q = Q = Wq matmul x1, k = Wk mm x2, v = Wv mm x3
        q, k, v = rearrange(qkv, 'b n (qkv h d) -> qkv b h n d', qkv=3, h=h)  # split into multi head attentions

        dots = torch.einsum('bhid,bhjd->bhij', q, k) * self.scale
        if mask is not None:
            assert mask.shape[-1] == dots.shape[-1], 'mask has incorrect dimensions'
            dots = (dots + mask) * 0.5

        attn = dots.softmax(dim=-1)  # follow the softmax,q,d,v equation in the paper

        out = torch.einsum('bhij,bhjd->bhid', attn, v)  # product of v times whatever inside softmax
        out = rearrange(out, 'b h n d -> b n (h d)')  # concat heads into one matrix, ready for next encoder block
        out = self.nn1(out)
        out = self.do1(out)

        return out


class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, mlp_dim, dropout):
        super().__init__()
        self.layers = nn.ModuleList([])
        if dim == mlp_dim:
            for _ in range(depth):
                self.layers.append(nn.ModuleList([
                    Residual(Attention(dim, mlp_dim, heads=heads, dropout=dropout)),
                    Residual(LayerNormalize(mlp_dim, MLP_Block(mlp_dim, mlp_dim * 2, dropout=dropout)))
                ]))
        else:
            for _ in range(depth):
                self.layers.append(nn.ModuleList([
                    Attention(dim, mlp_dim, heads=heads, dropout=dropout),
                    Residual(LayerNormalize(mlp_dim, MLP_Block(mlp_dim, mlp_dim * 2, dropout=dropout)))
                ]))

    def forward(self, x, mask=None):
        for attention, mlp in self.layers:
            ix = attention(x, mask=mask)  # go to attention
            mx = mlp(ix)  # go to MLP_Block
        return mx


class TCN_STRANSF_unit(nn.Module):
    def __init__(self, in_channels, out_channels, heads=3, stride=1, residual=True, dropout=0.1, spatial_mask=None,
                 mask_grad=True):
        super(TCN_STRANSF_unit, self).__init__()
        self.transf1 = Transformer(dim=in_channels, depth=1, heads=heads, mlp_dim=in_channels, dropout=dropout)
        self.tcn1 = unit_tcn_m(in_channels, out_channels, stride=stride)
        self.relu = nn.ReLU()
        self.out_channels = out_channels

        if not residual:
            self.residual = lambda x: 0

        elif (in_channels == out_channels) and (stride == 1):
            self.residual = lambda x: x

        else:
            self.residual = unit_tcn(in_channels, out_channels, kernel_size=1, stride=stride)

    def forward(self, x):
        B, C, T, V = x.size()
        tx = x.permute(0, 2, 3, 1).contiguous().view(B * T, V, C)
        tx = self.transf1(tx)
        tx = tx.view(B, T, V, C).permute(0, 3, 1, 2).contiguous()
        x = self.tcn1(tx) + self.residual(x)
        return self.relu(x)


# Transformer 原始位置编码
class SinPositionEncoding(nn.Module):
    def __init__(self, max_sequence_length, d_model, base=10000):
        super().__init__()
        self.max_sequence_length = max_sequence_length
        self.d_model = d_model
        self.base = base

    def forward(self):
        pe = torch.zeros(self.max_sequence_length, self.d_model,
                         dtype=torch.float)  # size(max_sequence_length, d_model)
        exp_1 = torch.arange(self.d_model // 2, dtype=torch.float)  # 初始化一半维度，sin位置编码的维度被分为了两部分
        exp_value = exp_1 / (self.d_model / 2)

        alpha = 1 / (self.base ** exp_value)  # size(dmodel/2)
        out = torch.arange(self.max_sequence_length, dtype=torch.float)[:, None] @ alpha[None,
                                                                                   :]  # size(max_sequence_length, d_model/2)
        embedding_sin = torch.sin(out)
        embedding_cos = torch.cos(out)

        pe[:, 0::2] = embedding_sin  # 奇数位置设置为sin
        pe[:, 1::2] = embedding_cos  # 偶数位置设置为cos
        pe = pe.unsqueeze(0)
        return torch.tensor(pe).cuda()


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


def temporal_window_partition(x, window_size):
    """
    Args:
        x: (B, N, C)
        window_size (int): window size

    Returns:
        windows: (num_windows*B, window_size, window_size, C)
    """
    B, N, C = x.shape
    x = x.view(B, N // window_size, window_size, C)  # B, nW, W, c
    windows = x.view(-1, window_size, C)  # B*nW, W, c
    return windows


def temporal_window_reverse(windows, window_size, N):
    """
    Args:
        windows: (num_windows*B, window_size, window_size, C)
        window_size (int): Window size
        H (int): Height of image
        W (int): Width of image

    Returns:
        x: (B, H, W, C)
    """
    B = int(windows.shape[0] / (N / window_size))
    x = windows.view(B, N // window_size, window_size, -1)
    x = x.view(B, N, -1)
    return x


class TemporalWindowAttention(nn.Module):  # attention block
    def __init__(self, dim, window_size, num_heads, qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.dim = dim
        self.window_size = window_size  # W
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        # define a parameter table of relative position bias
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros(((2 * window_size - 1), num_heads)))  # 2*W-1, nH

        coords = torch.arange(self.window_size)
        relative_distance = torch.transpose(torch.arange(self.window_size).unsqueeze(0), 0, 1)
        reversed_coords = torch.flip(coords, [0])
        _, relative_index = torch.meshgrid([reversed_coords, reversed_coords])
        relative_position_index = relative_index + relative_distance  # W, W

        self.register_buffer("relative_position_index", relative_position_index)
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        trunc_normal_(self.relative_position_bias_table, std=.02)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, mask=None):
        # print(x.shape)
        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))
        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.window_size, self.window_size, -1)  # W,W,nH
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
        attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)

        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class TemporalWindowTransformerBlock(nn.Module):  # transformer block
    def __init__(self, dim, num_frames, num_heads, window_size=10, shift_size=0, mlp_ratio=4., qkv_bias=True,
                 qk_scale=None, drop=0.,
                 attn_drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.num_frames = num_frames
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        self.norm1 = norm_layer(dim)
        self.attn = TemporalWindowAttention(
            dim, window_size=self.window_size, num_heads=num_heads,
            qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        if self.shift_size > 0:
            # calculate attention mask for SW-MSA
            N = self.num_frames
            img_mask = torch.zeros((1, N, 1))  # 1 H W 1
            slices = (slice(0, -self.window_size),
                      slice(-self.window_size, -self.shift_size),
                      slice(-self.shift_size, None))
            cnt = 0
            for s in slices:
                img_mask[:, s, :] = cnt
                cnt += 1

            mask_windows = temporal_window_partition(img_mask, self.window_size)  # nW, window_size, 1
            mask_windows = mask_windows.view(-1, self.window_size)  # nW, w_s
            attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)  # nW w_s w_s
            attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
        else:
            attn_mask = None
        self.register_buffer("attn_mask", attn_mask)

    def forward(self, x):
        B, N, C = x.shape
        shortcut = x
        x = self.norm1(x)

        # cyclic shift
        if self.shift_size > 0:
            shifted_x = torch.roll(x, shifts=-self.shift_size, dims=1)
            # partition windows
            x_windows = temporal_window_partition(shifted_x, self.window_size)  # nW*B, window_size, C
        else:
            shifted_x = x
            # partition windows
            x_windows = temporal_window_partition(shifted_x, self.window_size)  # nW*B,  window_size, C

        x_windows = x_windows.view(-1, self.window_size, C)  # nW*B, window_size, C

        # W-MSA/SW-MSA
        attn_windows = self.attn(x_windows, mask=self.attn_mask)  # nW*B, window_size*window_size, C

        # merge windows
        attn_windows = attn_windows.view(-1, self.window_size, C)

        # reverse cyclic shift
        if self.shift_size > 0:
            shifted_x = temporal_window_reverse(attn_windows, self.window_size, N)  # B N C
            x = torch.roll(shifted_x, shifts=self.shift_size, dims=1)
        else:
            shifted_x = temporal_window_reverse(attn_windows, self.window_size, N)  # B N C
            x = shifted_x
        x = shortcut + self.drop_path(x)

        # FFN
        x = x + self.drop_path(self.mlp(self.norm2(x)))

        return x


class TemporalWindowTransformerLayer(nn.Module):
    def __init__(self, dim, num_frames, depth, num_heads, window_size,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., norm_layer=nn.LayerNorm, downsample=None, use_checkpoint=False,
                 fused_window_process=False):
        super().__init__()
        self.dim = dim
        self.num_frames = num_frames
        self.depth = depth
        self.use_checkpoint = use_checkpoint
        # build blocks
        self.blocks = nn.ModuleList([
            TemporalWindowTransformerBlock(dim=dim, num_frames=num_frames,
                                           num_heads=num_heads, window_size=window_size,
                                           shift_size=0 if (i % 2 == 0) else window_size // 2,
                                           mlp_ratio=mlp_ratio,
                                           qkv_bias=qkv_bias, qk_scale=qk_scale,
                                           drop=drop, attn_drop=attn_drop,
                                           drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                                           norm_layer=norm_layer)
            for i in range(depth)])

        # patch merging layer
        if downsample is not None:
            self.downsample = downsample(dim=dim, norm_layer=norm_layer)
        else:
            self.downsample = None

    def forward(self, x):
        for blk in self.blocks:
            if self.use_checkpoint:
                x = checkpoint.checkpoint(blk, x)
            else:
                x = blk(x)
        if self.downsample is not None:
            x = self.downsample(x)
        return x


class Temporal_unit(nn.Module):
    def __init__(self, in_channels, out_channels, heads=3, residual=True, num_frames=100, depth=2, window_size=10,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1,
                 norm_layer=nn.LayerNorm, ape=False, patch_norm=True,
                 use_checkpoint=False, temporal_merge=False):
        super().__init__()
        assert num_frames % window_size == 0, "num_frames can not divide window size"
        self.mlp_ratio = mlp_ratio
        self.ape = ape
        self.patch_norm = patch_norm
        self.in_channels = in_channels
        self.window_size = window_size
        self.out_channels = out_channels

        # absolute position embedding
        if self.ape:
            self.absolute_pos_embed = nn.Parameter(torch.zeros(1, num_frames, in_channels))
            trunc_normal_(self.absolute_pos_embed, std=.02)

        self.trans = TemporalWindowTransformerLayer(
            dim=in_channels,
            num_frames=num_frames,
            depth=depth,
            num_heads=heads,
            window_size=window_size,
            qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate,
            # drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
            norm_layer=norm_layer,
            downsample=TemporalWindowPatchMerging if temporal_merge else None,
            use_checkpoint=use_checkpoint
        )

        self.norm = norm_layer(out_channels)
        self.apply(self._init_weights)
        # self.linear = nn.Linear(in_channels, out_channels) # 使用temporal_merge以后不需要使用线性层进行升维
        self.pos_drop = nn.Dropout(p=drop_rate)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward_features(self, x):
        assert x.size()[1] % self.window_size == 0, "current num_frames can not be divided by window size"
        if self.ape:
            x = x + self.absolute_pos_embed
        x = self.pos_drop(x)

        x = self.trans(x)

        x = self.norm(x)  # B L C
        return x

    def forward(self, x):
        x = self.forward_features(x)
        # x = self.linear(x)
        return x


class TemporalWindowPatchMerging(nn.Module):
    r""" Patch Merging Layer.

    Args:
        dim (int): Number of input channels.
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(self, dim, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.reduction = nn.Linear(2 * dim, 2 * dim, bias=False)
        self.norm = norm_layer(2 * dim)

    def forward(self, x):
        """
        x: B, L, C
        """
        B, L, C = x.shape
        assert L % 2 == 0, f"x length ({L}) are not even."

        x0 = x[:, 0::2, :]  # B L/2 C
        x1 = x[:, 1::2, :]  # B L/2 C
        x = torch.cat([x0, x1], -1)  # B L/2 2*C
        x = self.norm(x)
        x = self.reduction(x)
        return x


class MoE_Temporal_Spatial_Trans_unit(nn.Module):
    def __int__(self, in_channels, out_channels, spatial_heads=3, temporal_heads=3, residual=True, dropout=0.1,
                temporal_merge=False, t_expert_receptive_field=[4, 8, 16, 32], num_frames=100,
                t_expert_depth=[2, 2, 2, 2],
                spatial_depth=1, topk=1, temporal_part_size=32, use_zloss=1):
        assert len(t_expert_receptive_field) == len(
            t_expert_depth), "the num of expert fields and expert_depth is not equal"
        assert num_frames % temporal_part_size == 0, "the num_frames can not be divided by temporal_part_size"
        self.num_expert = len(t_expert_receptive_field)
        self.t_experts = nn.ModuleList()
        self.num_parts = num_frames // temporal_part_size
        self.w_gate = nn.Parameter(torch.zeros(in_channels, self.num_expert), requires_grad=True)
        self.k = topk
        self.temporal_part_size = temporal_part_size
        self.use_zloss = use_zloss
        for i in range(self.num_experts):
            self.t_experts.append(
                Temporal_unit(in_channels,
                              out_channels,
                              residual=residual,
                              heads=temporal_heads,
                              temporal_merge=temporal_merge, window_size=t_expert_receptive_field[i],
                              num_frames=temporal_part_size,
                              ape=False, depth=t_expert_depth[i]))
        self.S_trans = Transformer(dim=in_channels, depth=spatial_depth, heads=spatial_heads, mlp_dim=in_channels,
                                   dropout=dropout)
        self.temporal_merge = temporal_merge
        self.relu = nn.ReLU()
        self.drop = nn.Dropout(p=0.3)
        self.bn1 = nn.BatchNorm1d(in_channels)
        bn_init(self.bn1, 1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        bn_init(self.bn2, 1)
        if not residual:
            self.residual = lambda x: 0
        elif temporal_merge == True:
            self.residual = TemporalWindowPatchMerging(in_channels)
        elif temporal_merge == False:
            self.residual = lambda x: x

    def forward(self, x):
        B, C, T, V = x.size()
        rx = rearrange(x, "b c t v -> (b v) t c")
        sx = x.permute(0, 2, 3, 1).contiguous().view(B * T, V, C)
        sx = self.S_trans(sx)
        sx = rearrange(sx, "(b t) v c -> (b v) c t", t=T, v=V)
        sx = self.bn1(sx)
        tx = rearrange(sx, "(b v) c t -> (b v t) c", t=T, v=V)
        output = []
        logits = []
        for i in range(self.num_parts):
            ss = B * V * i * self.temporal_part_size
            tt = B * V * (i + 1) * self.temporal_part_size
            expert_input = tx[ss:tt, :]
            topk_experts, part_logits = self.topk_gate(expert_input)
            if self.temporal_merge:
                expert_output = torch.zeros((B * T * V // 2, C * 2))
            else:
                expert_output = torch.zeros((B * T * V, C))
            for e in topk_experts:
                expert_input = rearrange(expert_input, "(b v t) c -> (b v) t c", t=T, v=V)
                expert_output += self.t_experts[e](expert_input)
            output.append(expert_output)
            logits.append(part_logits)
        logits = torch.cat(logits, dim=0)
        zloss = self.use_zloss * self.compute_zloss(logits)
        atx = torch.cat(output, dim=0)
        stx = rearrange(atx, "(b v) t c -> b c t v", v=V)
        stx = self.bn2(stx)
        stx = self.drop(stx)
        rx = self.residual(rx)
        rx = rearrange(rx, "(b v) t c -> b c t v", v=V)
        stx = stx + rx
        return self.relu(stx), zloss

    def topk_gate(self, x):
        logits = x * self.w_gate
        logits = logits.mean(0)
        probs = torch.softmax(logits, dim=1)
        top_k_gates, top_k_indices = probs.topk(self.k, dim=1)
        top_k_gates = top_k_gates.flatten()
        top_k_experts = top_k_indices.flatten()
        nonzeros = top_k_gates.nonzero().squeeze(-1)
        top_k_experts_nonzero = top_k_experts[nonzeros]
        return top_k_experts_nonzero, logits

    def compute_zloss(self, logits):
        zloss = torch.mean(torch.log(torch.exp(logits).sum(dim=1)) ** 2)
        return zloss


class Temporal_Spatial_Trans_unit(nn.Module):
    def __init__(self, in_channels, out_channels, spatial_heads=3, temporal_heads=3, stride=1, residual=True,
                 dropout=0.1, temporal_merge=False, expert_windows_size=[8, 8], num_frames=100, temporal_depth=2,
                 spatial_depth=1, expert_weights=[0.5, 0.5], isLearnable=False, channelDivide=False,
                 spatial_mask=None, temporal_ape=False, use_zloss=0, spatial_mask_require_grad=True):
        super().__init__()
        if spatial_mask == None:
            self.spatial_mask = None
        else:
            self.spatial_mask = nn.Parameter(spatial_mask, requires_grad=spatial_mask_require_grad)
        self.S_trans = Transformer(dim=in_channels, depth=spatial_depth, heads=spatial_heads, mlp_dim=in_channels,
                                   dropout=dropout)
        self.bn1 = nn.BatchNorm1d(in_channels)
        bn_init(self.bn1, 1)
        self.channelDivide = channelDivide
        self.expert_weights_learnable = isLearnable

        if len(expert_windows_size) == 1:
            expert_weights = [1.0]
        assert len(expert_weights) == len(
            expert_windows_size), "the numbers of expert weights and their windows size are not equal"
        if not isLearnable:
            expert_weights = torch.tensor(expert_weights)
            self.register_buffer("expert_weights", expert_weights)
        self.num_experts = len(expert_windows_size)
        self.temporal_merge = temporal_merge
        self.expert_linear = nn.Linear(in_channels, self.num_experts)
        self.expert_softmax = nn.Softmax(dim=-1)
        self.experts = nn.ModuleList()
        for i in range(self.num_experts):
            self.experts.append(
                Temporal_unit(in_channels // self.num_experts if self.channelDivide else in_channels,
                              out_channels // self.num_experts if self.channelDivide else out_channels,
                              residual=residual,
                              heads=temporal_heads,
                              temporal_merge=temporal_merge, window_size=expert_windows_size[i],
                              num_frames=num_frames,
                              ape=temporal_ape, depth=temporal_depth))

        # TODO ape
        self.relu = nn.ReLU()
        self.drop = nn.Dropout(p=0.3)
        self.bn2 = nn.BatchNorm2d(out_channels)
        bn_init(self.bn2, 1)
        if not residual:
            self.residual = lambda x: 0
        elif temporal_merge == True:
            self.residual = TemporalWindowPatchMerging(in_channels)
        elif temporal_merge == False:
            self.residual = lambda x: x
        self.use_zloss = use_zloss

    def compute_zloss(self, logits):
        zloss = torch.mean(torch.log(torch.exp(logits).sum(dim=1)) ** 2)
        return zloss

    def forward(self, x):
        B, C, T, V = x.size()
        rx = rearrange(x, "b c t v -> (b v) t c")
        sx = x.permute(0, 2, 3, 1).contiguous().view(B * T, V, C)
        if self.spatial_mask is not None:
            sx = self.S_trans(sx, self.spatial_mask)
        else:
            sx = self.S_trans(sx)
        sx = rearrange(sx, "(b t) v c -> (b v) c t", t=T, v=V)
        sx = self.bn1(sx)
        tx = rearrange(sx, "(b v) c t -> (b v) t c", t=T, v=V)
        if self.expert_weights_learnable:
            logit = self.expert_linear(tx)
            expert_weights = self.expert_softmax(logit)
            # zloss = self.use_zloss * self.compute_zloss(logit)
        else:
            expert_weights = self.expert_weights
        if self.channelDivide:
            atx = self.T_multi_expert_channels(tx, B, C, T, V, expert_weights)
        elif self.expert_weights_learnable:
            atx = self.T_multi_expert_learn(tx, B, C, T, V, expert_weights)
        elif self.expert_weights_learnable == False:
            atx = self.T_multi_expert(tx, B, C, T, V, expert_weights)
        stx = rearrange(atx, "(b v) t c -> b c t v", v=V)
        stx = self.bn2(stx)
        stx = self.drop(stx)
        rx = self.residual(rx)
        rx = rearrange(rx, "(b v) t c -> b c t v", v=V)
        stx = stx + rx
        return self.relu(stx)

    def T_multi_expert_learn(self, x, B, C, T, V, expert_weights):
        if self.temporal_merge:
            atx = torch.zeros(B * V, T // 2, C * 2)
        else:
            atx = torch.zeros(B * V, T, C)  # get experts' embedding dimension
        atx = atx.cuda(x.device)
        for i in range(self.num_experts):
            expert_output = self.experts[i](x)
            if self.temporal_merge:
                expert_weights_i = ((expert_weights[:, 0::2, i] + expert_weights[0, 1::2, i]) / 2).unsqueeze(-1)
                atx += torch.mul(expert_output, expert_weights_i)
            else:
                atx += torch.mul(expert_output, expert_weights[:, :, i].unsqueeze(-1))
        return atx

    def T_multi_expert(self, x, B, C, T, V, expert_weights):
        # x : B T C
        if self.temporal_merge:
            atx = torch.zeros(B * V, T // 2, C * 2)
        else:
            atx = torch.zeros(B * V, T, C)  # get experts' embedding dimension
        atx = atx.cuda(x.device)
        for i in range(self.num_experts):
            atx = expert_weights[i] * self.experts[i](x)
        return atx

    def T_multi_expert_channels(self, x, B, C, T, V, expert_weights):
        # x : B T C
        if self.temporal_merge:
            atx = torch.zeros(B * V, T // 2, C * 2)
        else:
            atx = torch.zeros(B * V, T, C)  # get experts' embedding dimension
        logit = self.expert_linear(x.reshape(-1, C))
        expert_weights = self.expert_softmax(logit)
        atx = atx.cuda(x.device)
        for i in range(self.num_experts):
            atx[:, :, i::self.num_experts] = expert_weights[i] * self.experts[i](x[:, :, i::self.num_experts])
        return atx


class ZiT(nn.Module):
    def __init__(self, in_channels=3, num_person=5, num_point=18, num_head=6, graph=None, graph_args=dict(),
                 num_frame=100, embed_dim=48, depths=[2, 2, 2], expert_windows_size=[8, 8], expert_weights=[0.5, 0.5]):
        super(ZiT, self).__init__()
        self.data_bn = nn.BatchNorm1d(num_person * in_channels * num_point)
        bn_init(self.data_bn, 1)
        self.heads = num_head

        if graph is None:
            raise ValueError()
        else:
            Graph = import_class(graph)
            self.graph = Graph(**graph_args)

        self.A = torch.from_numpy(self.graph.A[0].astype(np.float32))
        self.l1 = TCN_GCN_unit(in_channels, 48, self.graph.A[0], residual=False)  # only contain A[0] (adjacency matrix)
        self.l2 = TCN_STRANSF_unit(48, 48, heads=num_head, mask=self.A, mask_grad=False)
        self.l3 = TCN_STRANSF_unit(48, 48, heads=num_head, mask=self.A, mask_grad=False)
        self.l4 = TCN_STRANSF_unit(48, 96, heads=num_head, stride=2, mask=self.A, mask_grad=True)
        self.l5 = TCN_STRANSF_unit(96, 96, heads=num_head, mask=self.A, mask_grad=True)
        self.l6 = TCN_STRANSF_unit(96, 192, heads=num_head, stride=2, mask=self.A, mask_grad=True)
        self.l7 = TCN_STRANSF_unit(192, 192, heads=num_head, mask=self.A, mask_grad=True)

    def forward(self, x):
        N, C, T, V, M = x.size()
        x = x.permute(0, 4, 3, 1, 2).contiguous().view(N, M * V * C, T)
        x = self.data_bn(x)
        x = x.view(N, M, V, C, T).permute(0, 1, 3, 4, 2).contiguous().view(N * M, C, T, V)  # 这样就达到了共享参数的目的

        x = self.l1(x)
        x = self.l2(x)
        x = self.l3(x)
        x = self.l4(x)
        x = self.l5(x)
        x = self.l6(x)
        x = self.l7(x)

        B, C_, T_, V_ = x.size()
        x = x.view(N, M, C_, T_, V_).mean(4)
        x = x.permute(0, 2, 3, 1).contiguous()

        return x


class myZiT(nn.Module):
    def __init__(self, in_channels=3, num_person=5, num_point=18, spatial_heads=6, temporal_heads=3, graph=None,
                 graph_args=dict(),
                 num_frame=100, embed_dim=48, expert_windows_size=[8, 8], expert_weights=[0.5, 0.5],
                 isLearnable=False, channelDivide=False, add_spatial_mask=False, temporal_ape=False,
                 layer_temporal_depths=[2, 2, 2, 2, 2],
                 block_structure=[[48, 48], [48, 96], [96, 96], [92, 192], [192, 192]], mergeSlow=False):
        super().__init__()
        self.data_bn = nn.BatchNorm1d(num_person * in_channels * num_point)
        bn_init(self.data_bn, 1)
        self.num_frames = num_frame
        self.num_layers = len(block_structure)
        assert len(layer_temporal_depths) == len(block_structure), "not equal"
        if graph is None:
            raise ValueError()
        else:
            Graph = import_class(graph)
            self.graph = Graph(**graph_args)

        self.A = torch.from_numpy(self.graph.A[0].astype(np.float32))
        self.tcn_gcn = TCN_GCN_unit(in_channels, embed_dim, self.graph.A[0],
                                    residual=False)  # only contain A[0] (adjacency matrix)
        self.layers = nn.ModuleList()
        assert embed_dim == block_structure[0][0], "the first embedding dimension is not equal"
        base_feature_dim = embed_dim
        spatial_mask_fix_layer = [0, 1] if self.num_layers >= 5 else [0]
        for index, hyper_paras in enumerate(block_structure):
            assert hyper_paras[1] % hyper_paras[0] == 0, "not follow hierarchical structure" + str(hyper_paras)
            temporal_merge = False if hyper_paras[0] == hyper_paras[1] else True
            num_frame_ratio = hyper_paras[0] // base_feature_dim
            self.layers.append(Temporal_Spatial_Trans_unit(hyper_paras[0], hyper_paras[1], spatial_heads=spatial_heads,
                                                           temporal_heads=temporal_heads * num_frame_ratio,
                                                           expert_windows_size=[w // 2 for w in
                                                                                expert_windows_size] if temporal_merge and mergeSlow else expert_windows_size,
                                                           num_frames=self.num_frames // num_frame_ratio,
                                                           expert_weights=expert_weights, isLearnable=isLearnable,
                                                           channelDivide=channelDivide,
                                                           spatial_mask=self.A if add_spatial_mask else None,
                                                           temporal_ape=temporal_ape,
                                                           temporal_depth=layer_temporal_depths[index],
                                                           temporal_merge=temporal_merge,
                                                           spatial_mask_require_grad=False if index in spatial_mask_fix_layer else True))

    def forward(self, x):
        N, C, T, V, M = x.size()
        x = x.permute(0, 4, 3, 1, 2).contiguous().view(N, M * V * C, T)
        x = self.data_bn(x)
        x = x.view(N, M, V, C, T).permute(0, 1, 3, 4, 2).contiguous().view(N * M, C, T, V)  # 这样就达到了共享参数的目的
        x = self.tcn_gcn(x)
        for i in range(self.num_layers):
            x = self.layers[i](x)
        B, C_, T_, V_ = x.size()
        x = x.view(N, M, C_, T_, V_).mean(4)
        x = x.permute(0, 2, 3, 1).contiguous()  # B C T M
        return x


class ZoT(nn.Module):
    def __init__(self, num_class=15, num_head=6):
        super(ZoT, self).__init__()

        self.heads = num_head

        self.conv1 = nn.Conv2d(192, num_head, kernel_size=(1, 1))
        self.conv2 = nn.Conv2d(192, num_head, kernel_size=(1, 1))
        conv_init(self.conv1)
        conv_init(self.conv2)

        self.l1 = TCN_STRANSF_unit(12, 276, heads=num_head)  # 192 276
        self.l2 = TCN_STRANSF_unit(276, 276, heads=num_head)

        self.fc = nn.Linear(276, num_class)
        nn.init.normal_(self.fc.weight, 0, math.sqrt(2. / num_class))

    def forward(self, x):
        # N,C,T,M
        x1 = self.conv1(x)
        x2 = self.conv2(x)
        x1 = x1.unsqueeze(3)  # N C T P M
        x2 = x2.unsqueeze(4)  # N C T M P
        mask = x1 - x2
        N, C, T, M, M2 = mask.shape
        mask = mask.permute(0, 2, 1, 3, 4).contiguous().view(N * T, C, M, M2).detach()
        mask = mask.softmax(dim=-1)

        x = self.l1(x, mask)
        x = self.l2(x, mask)
        x = x.mean(3).mean(2)

        return self.fc(x)


class simpleZoT(nn.Module):
    def __init__(self, num_class=15, num_head=6):
        super().__init__()
        self.heads = num_head
        self.l1 = TCN_STRANSF_unit(192, 276, heads=num_head)  # 192 276
        self.l2 = TCN_STRANSF_unit(276, 276, heads=num_head)
        self.fc = nn.Linear(276, num_class)
        nn.init.normal_(self.fc.weight, 0, math.sqrt(2. / num_class))

    def forward(self, x):
        # N,C,T,M
        x = self.l1(x)
        x = self.l2(x)
        x = x.mean(3).mean(2)
        return self.fc(x)


class ZoTTransformer(nn.Module):
    def __init__(self, num_class=15, spatial_heads=6, temporal_heads=3,
                 num_frame=100, expert_windows_size=[8, 8], expert_weights=[0.5, 0.5],
                 isLearnable=False, channelDivide=False, add_spatial_mask=False, temporal_ape=False,
                 layer_temporal_depths=[2], block_structure=[[192, 192]], base_feature_dim=48, spatial_depth=1):
        super().__init__()
        self.layers = nn.ModuleList()
        self.num_layers = len(block_structure)
        assert len(layer_temporal_depths) == len(block_structure), "not equal"
        for index, hyper_paras in enumerate(block_structure):
            assert hyper_paras[1] % hyper_paras[0] == 0, "not follow hierarchical structure" + str(hyper_paras)
            temporal_merge = False if hyper_paras[0] == hyper_paras[1] else True
            num_frame_ratio = hyper_paras[0] // base_feature_dim
            self.layers.append(
                Temporal_Spatial_Trans_unit(hyper_paras[0], hyper_paras[1], spatial_heads=spatial_heads,
                                            temporal_heads=temporal_heads * num_frame_ratio,
                                            expert_windows_size=expert_windows_size,
                                            num_frames=num_frame // num_frame_ratio,
                                            expert_weights=expert_weights, isLearnable=isLearnable,
                                            channelDivide=channelDivide,
                                            spatial_mask=self.A if add_spatial_mask else None,
                                            temporal_ape=temporal_ape,
                                            temporal_depth=layer_temporal_depths[index], temporal_merge=temporal_merge,
                                            spatial_depth=spatial_depth))
        self.fc = nn.Linear(block_structure[-1][1], num_class)
        nn.init.normal_(self.fc.weight, 0, math.sqrt(2. / num_class))

    def forward(self, x):
        # N,C,T,M
        for i in range(self.num_layers):
            x = self.layers[i](x)
        x = x.mean(3).mean(2)
        return self.fc(x)


class Model(nn.Module):
    def __init__(self, num_class=15, in_channels=3, num_person=5, num_point=18, num_frame=128, s_num_head=6, graph=None,
                 graph_args=dict(),
                 t_expert_windows_size=[8, 8], t_num_head=3,
                 expert_weights=[0.5, 0.5], expert_weights_learnable=False,
                 ZiTstrct=[[48, 96], [96, 192], [192, 192]], ZiT_layer_temporal_depths=[2, 2, 2, 2, 2], ZoTType="org",
                 ZoTTstrct=[[192, 192]], ZoT_layer_temporal_depths=[2], ZoT_spatial_depth=1,
                 addMotion=False, channelDivide=False, onlyXYZ=False, angularType="p2a",
                 addSpatialMask=False, temporalApe=False, mergeSlow=False,
                 ):
        super(Model, self).__init__()
        if in_channels == 3:
            onlyXYZ = True
        self.addMotion = addMotion
        self.onlyXYZ = onlyXYZ
        self.dataset = angularType
        self.body_transf = myZiT(in_channels=in_channels, num_person=num_person, num_point=num_point,
                                 graph=graph, graph_args=graph_args, num_frame=num_frame,
                                 spatial_heads=s_num_head,
                                 temporal_heads=t_num_head,
                                 expert_windows_size=t_expert_windows_size,
                                 expert_weights=expert_weights, isLearnable=expert_weights_learnable,
                                 channelDivide=channelDivide, add_spatial_mask=addSpatialMask,
                                 temporal_ape=temporalApe, layer_temporal_depths=ZiT_layer_temporal_depths,
                                 block_structure=ZiTstrct, embed_dim=48, mergeSlow=mergeSlow)
        if ZoTType == "direct":
            pass
        elif ZoTType == "org":
            self.group_transf = simpleZoT(num_class=num_class)
        elif ZoTType == "transformer":
            self.group_transf = ZoTTransformer(num_class=num_class, spatial_heads=s_num_head, temporal_heads=t_num_head,
                                               num_frame=num_frame, expert_windows_size=t_expert_windows_size,
                                               expert_weights=expert_weights, isLearnable=expert_weights_learnable,
                                               channelDivide=channelDivide, add_spatial_mask=False,
                                               temporal_ape=temporalApe,
                                               layer_temporal_depths=ZoT_layer_temporal_depths,
                                               block_structure=ZoTTstrct, base_feature_dim=48,
                                               spatial_depth=ZoT_spatial_depth)

        self.angular_feature = Angular_feature()

    def forward(self, x):
        if self.onlyXYZ:
            pass
        elif self.dataset == "p2a":
            x = self.angular_feature.preprocessing_pingpong_coco(
                x, self.addMotion)  # add 9 channels with original 3 channels, total 12 channels, all = 12
        elif self.dataset == "ntu-60" or self.dataset == "ntu-120":
            x = self.angular_feature.ntu_preprocessing(x)  # add 12 channels with original 3 channels, all = 15
        x = self.body_transf(x)
        x = self.group_transf(x)
        return x
