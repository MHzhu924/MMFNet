from functools import partial
import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import trunc_normal_, DropPath
from timm.models.registry import register_model
import os
import sys
import torch.fft
import math

import traceback
import torch.utils.checkpoint as checkpoint

class OmniAttention(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, groups=1, reduction=0.0625, kernel_num=4, min_channel=16):
        super(OmniAttention, self).__init__()
        attention_channel = max(int(in_planes * reduction), min_channel)
        self.kernel_size = kernel_size
        self.kernel_num = kernel_num
        self.temperature = 1.0

        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Conv2d(in_planes, attention_channel, 1, bias=False)
        self.bn = nn.BatchNorm2d(attention_channel)
        self.relu = nn.ReLU(inplace=True)

        self.channel_fc = nn.Conv2d(attention_channel, in_planes, 1, bias=True)
        self.func_channel = self.get_channel_attention

        if in_planes == groups and in_planes == out_planes:  # depth-wise convolution
            self.func_filter = self.skip
        else:
            self.filter_fc = nn.Conv2d(attention_channel, out_planes, 1, bias=True)
            self.func_filter = self.get_filter_attention

        if kernel_size == 1:  # point-wise convolution
            self.func_spatial = self.skip
        else:
            self.spatial_fc = nn.Conv2d(attention_channel, kernel_size * kernel_size, 1, bias=True)
            self.func_spatial = self.get_spatial_attention

        if kernel_num == 1:
            self.func_kernel = self.skip
        else:
            self.kernel_fc = nn.Conv2d(attention_channel, kernel_num, 1, bias=True)
            self.func_kernel = self.get_kernel_attention

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            if isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def update_temperature(self, temperature):
        self.temperature = temperature

    @staticmethod
    def skip(_):
        return 1.0

    def get_channel_attention(self, x):
        channel_attention = torch.sigmoid(self.channel_fc(x).view(x.size(0), -1, 1, 1) / self.temperature)
        return channel_attention

    def get_filter_attention(self, x):
        filter_attention = torch.sigmoid(self.filter_fc(x).view(x.size(0), -1, 1, 1) / self.temperature)
        return filter_attention

    def get_spatial_attention(self, x):
        spatial_attention = self.spatial_fc(x).view(x.size(0), 1, 1, 1, self.kernel_size, self.kernel_size)
        spatial_attention = torch.sigmoid(spatial_attention / self.temperature)
        return spatial_attention

    def get_kernel_attention(self, x):
        kernel_attention = self.kernel_fc(x).view(x.size(0), -1, 1, 1, 1, 1)
        kernel_attention = F.softmax(kernel_attention / self.temperature, dim=1)
        return kernel_attention

    def forward(self, x):
        x = self.avgpool(x)
        x = self.fc(x)
        x = self.bn(x)
        x = self.relu(x)
        return self.func_channel(x), self.func_filter(x), self.func_spatial(x), self.func_kernel(x)


import torch.nn.functional as F
def generate_laplacian_pyramid(input_tensor, num_levels, size_align=True, mode='bilinear'):
    pyramid = []
    current_tensor = input_tensor
    _, _, H, W = current_tensor.shape
    for _ in range(num_levels):
        b, _, h, w = current_tensor.shape
        downsampled_tensor = F.interpolate(current_tensor, (h//2 + h%2, w//2 + w%2), mode=mode, align_corners=(H%2) == 1) # antialias=True
        if size_align: 
            upsampled_tensor = F.interpolate(downsampled_tensor, (H, W), mode=mode, align_corners=(H%2) == 1)
            laplacian = F.interpolate(current_tensor, (H, W), mode=mode, align_corners=(H%2) == 1) - upsampled_tensor
        else:
            upsampled_tensor = F.interpolate(downsampled_tensor, (h, w), mode=mode, align_corners=(H%2) == 1)
            laplacian = current_tensor - upsampled_tensor
        pyramid.append(laplacian)
        current_tensor = downsampled_tensor
    if size_align: current_tensor = F.interpolate(current_tensor, (H, W), mode=mode, align_corners=(H%2) == 1)
    pyramid.append(current_tensor)
    return pyramid
                
class FrequencySelection(nn.Module):
    def __init__(self, 
                in_channels,
                k_list=[2],
                lowfreq_att=True,
                fs_feat='feat',
                lp_type='freq',
                act='sigmoid',
                spatial='conv',
                spatial_group=1,
                spatial_kernel=3,
                init='zero',
                global_selection=False,
                ):
        super().__init__()
        self.k_list = k_list
        self.lp_list = nn.ModuleList()
        self.freq_weight_conv_list = nn.ModuleList()
        self.fs_feat = fs_feat
        self.lp_type = lp_type
        self.in_channels = in_channels
        # self.residual = residual
        if spatial_group > 64: spatial_group=in_channels
        self.spatial_group = spatial_group
        self.lowfreq_att = lowfreq_att
        if spatial == 'conv':
            self.freq_weight_conv_list = nn.ModuleList()
            _n = len(k_list)
            if lowfreq_att:  _n += 1
            for i in range(_n):
                freq_weight_conv = nn.Conv2d(in_channels=in_channels, 
                                            out_channels=self.spatial_group, 
                                            stride=1,
                                            kernel_size=spatial_kernel, 
                                            groups=self.spatial_group,
                                            padding=spatial_kernel//2, 
                                            bias=True)
                if init == 'zero':
                    freq_weight_conv.weight.data.zero_()
                    freq_weight_conv.bias.data.zero_()   
                else:
                    # raise NotImplementedError
                    pass
                self.freq_weight_conv_list.append(freq_weight_conv)
        else:
            raise NotImplementedError
        
        if self.lp_type == 'laplacian':
            pass
        else:
            raise NotImplementedError
        
        self.act = act
        self.global_selection = global_selection
        if self.global_selection:
            self.global_selection_conv_real = nn.Conv2d(in_channels=in_channels, 
                                            out_channels=self.spatial_group, 
                                            stride=1,
                                            kernel_size=1, 
                                            groups=self.spatial_group,
                                            padding=0, 
                                            bias=True)
            self.global_selection_conv_imag = nn.Conv2d(in_channels=in_channels, 
                                            out_channels=self.spatial_group, 
                                            stride=1,
                                            kernel_size=1, 
                                            groups=self.spatial_group,
                                            padding=0, 
                                            bias=True)
            if init == 'zero':
                self.global_selection_conv_real.weight.data.zero_()
                self.global_selection_conv_real.bias.data.zero_()  
                self.global_selection_conv_imag.weight.data.zero_()
                self.global_selection_conv_imag.bias.data.zero_()  

    def sp_act(self, freq_weight):
        if self.act == 'sigmoid':
            freq_weight = freq_weight.sigmoid() * 2
        elif self.act == 'softmax':
            freq_weight = freq_weight.softmax(dim=1) * freq_weight.shape[1]
        else:
            raise NotImplementedError
        return freq_weight

    def forward(self, x, att_feat=None):
        if att_feat is None: att_feat = x
        x_list = []
        if self.lp_type == 'laplacian':
            b, _, h, w = x.shape
            pyramids = generate_laplacian_pyramid(x, len(self.k_list), size_align=True)
            for idx, avg in enumerate(self.k_list):
                high_part = pyramids[idx]
                freq_weight = self.freq_weight_conv_list[idx](att_feat)
                freq_weight = self.sp_act(freq_weight)
                tmp = freq_weight.reshape(b, self.spatial_group, -1, h, w) * high_part.reshape(b, self.spatial_group, -1, h, w)
                x_list.append(tmp.reshape(b, -1, h, w))
            if self.lowfreq_att:
                freq_weight = self.freq_weight_conv_list[len(x_list)](att_feat)
                tmp = freq_weight.reshape(b, self.spatial_group, -1, h, w) * pyramids[-1].reshape(b, self.spatial_group, -1, h, w)
                x_list.append(tmp.reshape(b, -1, h, w))
            else:
                x_list.append(pyramids[-1])
        x = sum(x_list)
        return x
    
from mmcv.ops.deform_conv import DeformConv2dPack
from mmcv.ops.modulated_deform_conv import ModulatedDeformConv2d, modulated_deform_conv2d, ModulatedDeformConv2dPack

class SAWB(ModulatedDeformConv2d):

    _version = 2
    def __init__(self, *args, 
                 offset_freq=None,
                 use_BFM=False,
                 kernel_decompose='both',
                 padding_mode='repeat',
                #  padding_mode='zero',
                 normal_conv_dim=0,
                 pre_fs=True, # False, use dilation
                 fs_cfg={
                    # 'k_list':[3,5,7,9],
                    'k_list':[2,4,8],
                    'fs_feat':'feat',
                    'lowfreq_att':False,
                    'lp_type':'laplacian',
                    'act':'sigmoid',
                    'spatial':'conv',
                    'spatial_group':1,
                },
                 **kwargs):
        super().__init__(*args, **kwargs)
        assert self.kernel_size[0] in (3, 7)
        assert self.groups == self.in_channels
        if kernel_decompose == 'both':
            self.OMNI_ATT1 = OmniAttention(in_planes=self.in_channels, out_planes=self.out_channels, kernel_size=1, groups=self.in_channels, reduction=0.0625, kernel_num=1, min_channel=16)
            self.OMNI_ATT2 = OmniAttention(in_planes=self.in_channels, out_planes=self.out_channels, kernel_size=1, groups=self.in_channels, reduction=0.0625, kernel_num=1, min_channel=16)
        elif kernel_decompose == 'high':
            self.OMNI_ATT = OmniAttention(in_planes=self.in_channels, out_planes=self.out_channels, kernel_size=1, groups=self.in_channels, reduction=0.0625, kernel_num=1, min_channel=16)
        elif kernel_decompose == 'low':
            self.OMNI_ATT = OmniAttention(in_planes=self.in_channels, out_planes=self.out_channels, kernel_size=1, groups=self.in_channels, reduction=0.0625, kernel_num=1, min_channel=16)
        self.kernel_decompose = kernel_decompose

        self.normal_conv_dim = normal_conv_dim

        if padding_mode == 'zero':
            self.PAD = nn.ZeroPad2d(self.kernel_size[0]//2)
        elif padding_mode == 'repeat':
            self.PAD = nn.ReplicationPad2d(self.kernel_size[0]//2)
        else:
            self.PAD = nn.Identity()
        print(self.in_channels, self.normal_conv_dim,)
        self.conv_offset = nn.Conv2d(
            self.in_channels - self.normal_conv_dim,
            self.deform_groups * 1,
            # self.groups * 1,
            kernel_size=self.kernel_size,
            stride=self.stride,
            padding=self.padding if isinstance(self.PAD, nn.Identity) else 0,
            dilation=1,
            bias=True)
        self.conv_mask = nn.Sequential(
            nn.Conv2d(
                self.in_channels - self.normal_conv_dim,
                self.in_channels - self.normal_conv_dim,
                kernel_size=self.kernel_size,
                stride=self.stride,
                padding=self.padding if isinstance(self.PAD, nn.Identity) else 0,
                groups=self.in_channels - self.normal_conv_dim,
                dilation=1,
                bias=False),
            nn.Conv2d(
                self.in_channels - self.normal_conv_dim,
                self.deform_groups * 1 * self.kernel_size[0] * self.kernel_size[1],
                kernel_size=1,
                stride=1,
                padding=0,
                groups=1,
                dilation=1,
                bias=True)
        )
        
        self.offset_freq = offset_freq

        if self.offset_freq in ('FLC_high', 'FLC_res'):
            self.LP = FLC_Pooling(freq_thres=min(0.5 * 1 / self.dilation[0], 0.25))
        elif self.offset_freq in ('SLP_high', 'SLP_res'):
            self.LP = StaticLP(self.in_channels, kernel_size=5, stride=1, padding=2, alpha=8)
        elif self.offset_freq is None:
            pass
        else:
            raise NotImplementedError

        # An offset is like [y0, x0, y1, x1, y2, x2, ⋯, y8, x8]
        if self.kernel_size[0] == 3:
            offset = [-1, -1,  -1, 0,   -1, 1,
                    0, -1,   0, 0,    0, 1,
                    1, -1,   1, 0,    1,1]
        elif self.kernel_size[0] == 7:
            offset = [
                -3, -3,  -3, -2,  -3, -1,  -3, 0,  -3, 1,  -3, 2,  -3, 3, 
                -2, -3,  -2, -2,  -2, -1,  -2, 0,  -2, 1,  -2, 2,  -2, 3, 
                -1, -3,  -1, -2,  -1, -1,  -1, 0,  -1, 1,  -1, 2,  -1, 3, 
                0, -3,   0, -2,   0, -1,   0, 0,   0, 1,   0, 2,   0, 3, 
                1, -3,   1, -2,   1, -1,   1, 0,   1, 1,   1, 2,   1, 3, 
                2, -3,   2, -2,   2, -1,   2, 0,   2, 1,   2, 2,   2, 3, 
                3, -3,   3, -2,   3, -1,   3, 0,   3, 1,   3, 2,   3, 3, 
            ]
        else: raise NotImplementedError

        offset = torch.Tensor(offset)
        # offset[0::2] *= self.dilation[0]
        # offset[1::2] *= self.dilation[1]
        # a tuple of two ints – in which case, the first int is used for the height dimension, and the second int for the width dimension
        self.register_buffer('dilated_offset', torch.Tensor(offset[None, None, ..., None, None])) # B, G, 49, 1, 1
        self.init_weights()

        self.use_BFM = use_BFM
        if use_BFM:
            alpha = 8
            BFM = np.zeros((self.in_channels, 1, self.kernel_size[0], self.kernel_size[0]))
            for i in range(self.kernel_size[0]):
                for j in range(self.kernel_size[0]):
                    point_1 = (i, j)
                    point_2 = (self.kernel_size[0]//2, self.kernel_size[0]//2)
                    dist = distance.euclidean(point_1, point_2)
                    BFM[:, :, i, j] = alpha / (dist + alpha)
            self.register_buffer('BFM', torch.Tensor(BFM))
            print(self.BFM)
        if fs_cfg is not None:
            if pre_fs:
                self.FS = FrequencySelection(self.in_channels - self.normal_conv_dim, **fs_cfg)
            else:
                self.FS = FrequencySelection(1, **fs_cfg) # use dilation
        self.pre_fs = pre_fs

    def freq_select(self, x):
        if self.offset_freq is None:
            pass
        elif self.offset_freq in ('FLC_high', 'SLP_high'):
            x - self.LP(x)
        elif self.offset_freq in ('FLC_res', 'SLP_res'):
            2 * x - self.LP(x)
        else:
            raise NotImplementedError
        return x

    def init_weights(self):
        super().init_weights()
        if hasattr(self, 'conv_offset'):
            self.conv_offset.weight.data.zero_()
            self.conv_offset.bias.data.fill_((self.dilation[0] - 1)/self.dilation[0] + 1e-4)
        if hasattr(self, 'conv_mask'):
            self.conv_mask[1].weight.data.zero_()
            self.conv_mask[1].bias.data.zero_()

    def forward(self, x):
        if hasattr(self, 'FS') and self.pre_fs: x = self.FS(x)
        c_att1, _, _, _, = self.OMNI_ATT1(x)
        c_att2, _, _, _, = self.OMNI_ATT2(x)
        x = self.PAD(x)
        offset = self.conv_offset(x)
        offset = F.relu(offset, inplace=True) * self.dilation[0] # ensure > 0
        b, _, h, w = offset.shape
        offset = offset.reshape(b, self.deform_groups, -1, h, w) * self.dilated_offset
        offset = offset.reshape(b, -1, h, w)
        mask = self.conv_mask(x)
        mask = torch.sigmoid(mask)
        offset = offset.reshape(1, -1, h, w)
        mask = mask.reshape(1, -1, h, w)
        x = x.reshape(1, -1, x.size(-2), x.size(-1))
        adaptive_weight = self.weight.unsqueeze(0).repeat(b, 1, 1, 1, 1) # b, out, in, k, k
        adaptive_weight_mean = adaptive_weight.mean(dim=(-1, -2), keepdim=True)
        adaptive_weight = adaptive_weight_mean * (2 * c_att1.unsqueeze(2)) + (adaptive_weight - adaptive_weight_mean) * (2 * c_att2.unsqueeze(2))
        adaptive_weight = adaptive_weight.reshape(-1, self.in_channels // self.groups, 3, 3)
        x = modulated_deform_conv2d(x, offset, mask, adaptive_weight, self.bias,
                                    self.stride, self.padding if isinstance(self.PAD, nn.Identity) else 0, #padding
                                    (1, 1), # dilation
                                    self.groups * b, self.deform_groups * b)
        return x.reshape(b, -1, h, w)

