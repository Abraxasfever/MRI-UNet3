#Model B: Efficient U-Net 3+
from model.unet_3_plus_parts import *
import torch
import torch.nn as nn
import torch.nn.functional as F
from efficientnet_pytorch import EfficientNet
import time
import torch.nn.init as init
from .unet_3_plus_parts import *

class SqueezeExcite(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super(SqueezeExcite, self).__init__()
        self.fc1 = nn.Conv2d(in_channels, in_channels // reduction, kernel_size=1)
        self.fc2 = nn.Conv2d(in_channels // reduction, in_channels, kernel_size=1)
        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        batch, channels, _, _ = x.size()
        y = F.adaptive_avg_pool2d(x, 1)
        y = self.relu(self.fc1(y))
        y = self.sigmoid(self.fc2(y))
        return x * y

class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DepthwiseSeparableConv, self).__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, groups=in_channels)
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.relu(self.depthwise(x))
        x = self.relu(self.pointwise(x))
        return x

class UNet3Plus(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=True):
        super(UNet3Plus, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        filters = [32, 64, 128, 256, 512]

        self.encoder = EfficientNet.from_pretrained('efficientnet-b0')

        self.encoder._conv_stem = nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1, bias=False)

        self.inc = DoubleConv(n_channels, filters[0])
        self.down1 = Down(filters[0], filters[1])
        self.down2 = Down(filters[1], filters[2])
        self.down3 = Down(filters[2], filters[3])
        self.down4 = Down(filters[3], filters[4])

        self.decoder4_enc1 = self._decoder_layer(filters[0], 8)
        self.decoder4_enc2 = self._decoder_layer(filters[1], 4)
        self.decoder4_enc3 = self._decoder_layer(filters[2], 2)
        self.decoder4_enc4 = self._decoder_layer(filters[3], 1)
        self.decoder4_dec5 = self._upsample_layer(filters[4], 2)

        self.decoder3_enc1 = self._decoder_layer(filters[0], 4)
        self.decoder3_enc2 = self._decoder_layer(filters[1], 2)
        self.decoder3_enc3 = self._decoder_layer(filters[2], 1)
        self.decoder3_dec4 = self._upsample_layer(filters[0] * 5, 2)
        self.decoder3_dec5 = self._upsample_layer(filters[4], 4)

        self.decoder2_enc1 = self._decoder_layer(filters[0], 2)
        self.decoder2_enc2 = self._decoder_layer(filters[1], 1)
        self.decoder2_dec3 = self._upsample_layer(filters[0] * 5, 2)
        self.decoder2_dec4 = self._upsample_layer(filters[0] * 5, 4)
        self.decoder2_dec5 = self._upsample_layer(filters[4], 8)

        self.decoder1_enc1 = self._decoder_layer(filters[0], 1)
        self.decoder1_dec2 = self._upsample_layer(filters[0] * 5, 2)
        self.decoder1_dec3 = self._upsample_layer(filters[0] * 5, 4)
        self.decoder1_dec4 = self._upsample_layer(filters[0] * 5, 8)
        self.decoder1_dec5 = self._upsample_layer(filters[4], 16)

        self.final_conv4 = DepthwiseSeparableConv(filters[0] * 5, filters[0] * 5)
        self.final_conv3 = DepthwiseSeparableConv(filters[0] * 5, filters[0] * 5)
        self.final_conv2 = DepthwiseSeparableConv(filters[0] * 5, filters[0] * 5)
        self.final_conv1 = DepthwiseSeparableConv(filters[0] * 5, filters[0] * 5)

        self.outc = nn.Sequential(
            SqueezeExcite(filters[0] * 5),
            nn.Conv2d(filters[0] * 5, n_classes, kernel_size=1)
        )

        self._initialize_weights()

    def _decoder_layer(self, in_channels, pool_size, dilation=1):
        layers = []
        if pool_size > 1:
            layers.append(nn.MaxPool2d(pool_size))
        layers.append(nn.Conv2d(in_channels, 32, kernel_size=3, padding=dilation, dilation=dilation))  # 引入空洞卷积
        layers.append(nn.ReLU(inplace=True))
        return nn.Sequential(*layers)

    def _upsample_layer(self, in_channels, scale_factor):
        return nn.Sequential(
            nn.Upsample(scale_factor=scale_factor, mode='bilinear', align_corners=True),
            DepthwiseSeparableConv(in_channels, 32),  # 使用深度可分离卷积
        )

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        # Decoder4
        dec4_enc1 = self.decoder4_enc1(x1)
        dec4_enc2 = self.decoder4_enc2(x2)
        dec4_enc3 = self.decoder4_enc3(x3)
        dec4_enc4 = self.decoder4_enc4(x4)
        dec4_dec5 = self.decoder4_dec5(x5)

        x4d = torch.cat([dec4_enc1, dec4_enc2, dec4_enc3, dec4_enc4, dec4_dec5], dim=1)
        x4d = self.final_conv4(x4d)

        # Decoder3
        dec3_enc1 = self.decoder3_enc1(x1)
        dec3_enc2 = self.decoder3_enc2(x2)
        dec3_enc3 = self.decoder3_enc3(x3)
        dec3_dec4 = self.decoder3_dec4(x4d)
        dec3_dec5 = self.decoder3_dec5(x5)

        x3d = torch.cat([dec3_enc1, dec3_enc2, dec3_enc3, dec3_dec4, dec3_dec5], dim=1)
        x3d = self.final_conv3(x3d)

        # Decoder2
        dec2_enc1 = self.decoder2_enc1(x1)
        dec2_enc2 = self.decoder2_enc2(x2)
        dec2_dec3 = self.decoder2_dec3(x3d)
        dec2_dec4 = self.decoder2_dec4(x4d)
        dec2_dec5 = self.decoder2_dec5(x5)

        x2d = torch.cat([dec2_enc1, dec2_enc2, dec2_dec3, dec2_dec4, dec2_dec5], dim=1)
        x2d = self.final_conv2(x2d)

        # Decoder1
        dec1_enc1 = self.decoder1_enc1(x1)
        dec1_dec2 = self.decoder1_dec2(x2d)
        dec1_dec3 = self.decoder1_dec3(x3d)
        dec1_dec4 = self.decoder1_dec4(x4d)
        dec1_dec5 = self.decoder1_dec5(x5)

        x1d = torch.cat([dec1_enc1, dec1_dec2, dec1_dec3, dec1_dec4, dec1_dec5], dim=1)
        x1d = self.final_conv1(x1d)

        logits = self.outc(x1d)
        return logits
