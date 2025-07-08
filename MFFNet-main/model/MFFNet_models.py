import torch
import torch.nn as nn
from .pvtv2 import pvt_v2_b2
from model.AFMamba_models.AFMamba import AFMamba
from model.FSCM_models.FSCM import GlobalLocalFilter

class BasicConv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_planes, out_planes,
                              kernel_size=kernel_size, stride=stride,
                              padding=padding, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

class TransBasicConv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size=2, stride=2, padding=0, dilation=1, bias=False):
        super(TransBasicConv2d, self).__init__()
        self.Deconv = nn.ConvTranspose2d(in_planes, out_planes,
                                         kernel_size=kernel_size, stride=stride,
                                         padding=padding, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.Deconv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()

        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc1 = nn.Conv2d(in_planes, in_planes // 16, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_planes // 16, in_planes, 1, bias=False)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = max_out
        return self.sigmoid(out)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1

        self.conv1 = nn.Conv2d(1, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = max_out
        x = self.conv1(x)
        return self.sigmoid(x)


class MSFA_Decoder(nn.Module):
    def __init__(self, channel=32,dilation_1=3,dilation_2=2):
        super(MSFA_Decoder, self).__init__()

        self.conv1 = BasicConv2d(channel, channel, 3, padding=1)
        self.conv1_Dila = BasicConv2d(channel, channel, 3, padding=dilation_1, dilation=dilation_1)

        self.conv2 = BasicConv2d(channel, channel, 3, padding=1)
        self.conv2_Dila = BasicConv2d(channel, channel, 3, padding=dilation_2, dilation=dilation_2)
        self.conv3 = BasicConv2d(channel, channel, 3, padding=1)

        self.conv_fuse = BasicConv2d(channel * 3, channel, 3, padding=1)

    def forward(self, x):
        x1 = self.conv1(x)
        x1_dila = self.conv1_Dila(x1)
        
        x2 = self.conv2(x1)
        x2_dila = self.conv2_Dila(x2)
        
        x3 = self.conv3(x2)

        x_fuse = self.conv_fuse(torch.cat((x1_dila, x2_dila, x3), 1))

        return x_fuse


class decoder(nn.Module):
    def __init__(self, channel=32):
        super(decoder, self).__init__()
        self.relu = nn.ReLU(True)

        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        # self.decoder5 = nn.Sequential(
        #     MSFA_Decoder(channel),
        #     nn.Dropout(0.5),
        #     TransBasicConv2d(channel, channel, kernel_size=2, stride=2,
        #                      padding=0, dilation=1, bias=False)
        # )
        # self.S5 = nn.Conv2d(channel, 1, 3, stride=1, padding=1)

        self.decoder4 = nn.Sequential(
            MSFA_Decoder(channel ),
            nn.Dropout(0.5),
            TransBasicConv2d(channel , channel, kernel_size=2, stride=2,
                             padding=0, dilation=1, bias=False)
        )
        self.S4 = nn.Conv2d(channel, 1, 3, stride=1, padding=1)

        self.decoder3 = nn.Sequential(
            MSFA_Decoder(channel * 2),
            nn.Dropout(0.5),
            TransBasicConv2d(channel * 2, channel, kernel_size=2, stride=2,
                             padding=0, dilation=1, bias=False)
        )
        self.S3 = nn.Conv2d(channel, 1, 3, stride=1, padding=1)

        self.decoder2 = nn.Sequential(
            MSFA_Decoder(channel * 2),
            nn.Dropout(0.5),
            TransBasicConv2d(channel * 2, channel, kernel_size=2, stride=2,
                             padding=0, dilation=1, bias=False)
        )
        self.S2 = nn.Conv2d(channel, 1, 3, stride=1, padding=1)

        self.decoder1 = nn.Sequential(
            MSFA_Decoder(channel * 2),
            BasicConv2d(channel * 2, channel, 3, padding=1)
        )
        self.S1 = nn.Conv2d(channel, 1, 3, stride=1, padding=1)

    def forward(self, x4, x3, x2, x1):

        x4_up = self.decoder4(x4)

        s4 = self.S4(x4_up)

        x3_up = self.decoder3(torch.cat((x3, x4_up), 1))
        s3 = self.S3(x3_up)

        x2_up = self.decoder2(torch.cat((x2, x3_up), 1))
        s2 = self.S2(x2_up)

        x1_up = self.decoder1(torch.cat((x1, x2_up), 1))

        s1 = self.S1(x1_up)

        return s1, s2, s3, s4



class MFFNet(nn.Module):
    def __init__(self, channel=32):
        super(MFFNet, self).__init__()

        self.backbone = pvt_v2_b2()  # [64, 128, 320, 512]
        path = './model/pvt_v2_b2.pth'
        save_model = torch.load(path)
        model_dict = self.backbone.state_dict()
        state_dict = {k: v for k, v in save_model.items() if k in model_dict.keys()}
        model_dict.update(state_dict)
        self.backbone.load_state_dict(model_dict)
        
        
        self.FSCM_3 = GlobalLocalFilter(512)  
        self.FSCM_2 = GlobalLocalFilter(320)  
        self.FSCM_1 = GlobalLocalFilter(128) 
        self.FSCM_0 = GlobalLocalFilter(64)  

        
        self.channel_normalization0 = BasicConv2d(64, channel, 3, stride=1, padding=1)
        self.channel_normalization1 = BasicConv2d(128, channel, 3, stride=1, padding=1)
        self.channel_normalization2 = BasicConv2d(320, channel, 3, stride=1, padding=1)
        self.channel_normalization3 = BasicConv2d(512, channel, 3, stride=1, padding=1)

        
        self.AFMamba = AFMamba(dims=channel)
        self.decoder_rgb = decoder(channel)
       
        self.upsample16 = nn.Upsample(scale_factor=16, mode='bilinear', align_corners=True)
        self.upsample8 = nn.Upsample(scale_factor=8, mode='bilinear', align_corners=True)
        self.upsample4 = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x_rgb):

        pvt = self.backbone(x_rgb)
        x1 = pvt[0] # 64x128x256
        x2 = pvt[1] # 128x64x128
        x3 = pvt[2] # 320x32x64
        x4 = pvt[3] # 512x16x32

        fscm_3 = self.FSCM_3(x4)
        fscm_2 = self.FSCM_2(x3)
        fscm_1 = self.FSCM_1(x2)
        fscm_0 = self.FSCM_0(x1)
      
        x1_rgb = self.channel_normalization0(fscm_0)    # 32x128x256
        x2_rgb = self.channel_normalization1(fscm_1)    # 32x64x128
        x3_rgb = self.channel_normalization2(fscm_2)    # 32x32x64
        x4_rgb = self.channel_normalization3(fscm_3)    # 32x16x32
        
        x_decoder=[x4_rgb, x3_rgb, x2_rgb, x1_rgb]

        s4, s3, s2,s1 = self.AFMamba(x_decoder)

        s1, s2, s3, s4 = self.decoder_rgb(s4, s3, s2, s1)

        
        s1 = self.upsample4(s1)
        s2 = self.upsample4(s2)
        s3 = self.upsample8(s3)
        s4 = self.upsample16(s4)

        return s1, s2, s3, s4, self.sigmoid(s1), self.sigmoid(s2), self.sigmoid(s3), self.sigmoid(s4)


