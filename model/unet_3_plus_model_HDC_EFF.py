# Model D: Full Attention U-Net 3+
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from .unet_3_plus_parts import *

class SEBlock(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super(SEBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels // reduction, in_channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class UNet3Plus(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=True):
        super(UNet3Plus, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        filters = [32, 64, 128, 256, 512]

        self.inc = DoubleConv(n_channels, filters[0])
        self.down1 = Down(filters[0], filters[1])
        self.down2 = Down(filters[1], filters[2])
        self.down3 = Down(filters[2], filters[3])
        self.down4 = Down(filters[3], filters[4])

        self.decoder4_enc1 = self._decoder_layer(filters[0], 8, attention=True)
        self.decoder4_enc2 = self._decoder_layer(filters[1], 4, attention=True)
        self.decoder4_enc3 = self._decoder_layer(filters[2], 2, attention=True)
        self.decoder4_enc4 = self._decoder_layer(filters[3], 1, attention=True)
        self.decoder4_dec5 = self._upsample_layer(filters[4], 2, attention=True)

        self.decoder3_enc1 = self._decoder_layer(filters[0], 4, attention=True)
        self.decoder3_enc2 = self._decoder_layer(filters[1], 2, attention=True)
        self.decoder3_enc3 = self._decoder_layer(filters[2], 1, attention=True)
        self.decoder3_dec4 = self._upsample_layer(filters[0] * 5, 2, attention=True)
        self.decoder3_dec5 = self._upsample_layer(filters[4], 4, attention=True)

        self.decoder2_enc1 = self._decoder_layer(filters[0], 2, attention=True)
        self.decoder2_enc2 = self._decoder_layer(filters[1], 1, attention=True)
        self.decoder2_dec3 = self._upsample_layer(filters[0] * 5, 2, attention=True)
        self.decoder2_dec4 = self._upsample_layer(filters[0] * 5, 4, attention=True)
        self.decoder2_dec5 = self._upsample_layer(filters[4], 8, attention=True)

        self.decoder1_enc1 = self._decoder_layer(filters[0], 1, attention=True)
        self.decoder1_dec2 = self._upsample_layer(filters[0] * 5, 2, attention=True)
        self.decoder1_dec3 = self._upsample_layer(filters[0] * 5, 4, attention=True)
        self.decoder1_dec4 = self._upsample_layer(filters[0] * 5, 8, attention=True)
        self.decoder1_dec5 = self._upsample_layer(filters[4], 16, attention=True)

        self.final_conv4 = nn.Conv2d(filters[0] * 5, filters[0] * 5, kernel_size=3, padding=1)
        self.final_conv3 = nn.Conv2d(filters[0] * 5, filters[0] * 5, kernel_size=3, padding=1)
        self.final_conv2 = nn.Conv2d(filters[0] * 5, filters[0] * 5, kernel_size=3, padding=1)
        self.final_conv1 = nn.Conv2d(filters[0] * 5, filters[0] * 5, kernel_size=3, padding=1)

        self.outc = nn.Conv2d(filters[0] * 5, n_classes, kernel_size=1)

        self._initialize_weights()

    def _decoder_layer(self, in_channels, pool_size, dilation=1, attention=False):
        layers = []
        if pool_size > 1:
            layers.append(nn.MaxPool2d(pool_size))
        layers.append(nn.Conv2d(in_channels, 32, kernel_size=3, padding=dilation, dilation=dilation))  # 引入空洞卷积
        layers.append(nn.ReLU(inplace=True))
        if attention:
            layers.append(SEBlock(32))
        return nn.Sequential(*layers)

    def _upsample_layer(self, in_channels, scale_factor, attention=False):
        layers = [
            nn.Upsample(scale_factor=scale_factor, mode='bilinear', align_corners=True),
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        ]
        if attention:
            layers.append(SEBlock(32))
        return nn.Sequential(*layers)

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

        x1 = self.inc(x)  # (B, 64, H, W)
        x2 = self.down1(x1)  # (B, 128, H/2, W/2)
        x3 = self.down2(x2)  # (B, 256, H/4, W/4)
        x4 = self.down3(x3)  # (B, 512, H/8, W/8)
        x5 = self.down4(x4)  # (B, 1024, H/16, W/16)

        # Decoder4
        dec4_enc1 = self.decoder4_enc1(x1)  # encoder1: maxpooling(8), 3*3conv, ReLU
        dec4_enc2 = self.decoder4_enc2(x2)  # encoder2: maxpooling(4), 3*3conv, ReLU
        dec4_enc3 = self.decoder4_enc3(x3)  # encoder3: maxpooling(2), 3*3conv, ReLU
        dec4_enc4 = self.decoder4_enc4(x4)  # encoder4: 3*3conv, ReLU
        dec4_dec5 = self.decoder4_dec5(x5)  # decoder5: bilinear upsample(2), 3*3conv, ReLU

        x4d = torch.cat([dec4_enc1, dec4_enc2, dec4_enc3, dec4_enc4, dec4_dec5], dim=1)

        x4d = self.final_conv4(x4d)  # (B, 320, H/8, W/8)

        # Decoder3
        dec3_enc1 = self.decoder3_enc1(x1)  # encoder1: maxpooling(4), 3*3conv, ReLU
        dec3_enc2 = self.decoder3_enc2(x2)  # encoder2: maxpooling(2), 3*3conv, ReLU
        dec3_enc3 = self.decoder3_enc3(x3)  # encoder2: maxpooling(2), 3*3conv, ReLU
        dec3_dec4 = self.decoder3_dec4(x4d)  # decoder4: bilinear upsample(2), 3*3conv, ReLU
        dec3_dec5 = self.decoder3_dec5(x5)  # decoder5: bilinear upsample(4), 3*3conv, ReLU

        x3d = torch.cat([dec3_enc1, dec3_enc2, dec3_enc3, dec3_dec4, dec3_dec5], dim=1)

        x3d = self.final_conv3(x3d)  # (B, 320, H/4, W/4)

        # Decoder2
        dec2_enc1 = self.decoder2_enc1(x1)  # encoder1: maxpooling(2), 3*3conv, ReLU
        dec2_enc2 = self.decoder2_enc2(x2)  # encoder2: 3*3conv, ReLU
        dec2_dec3 = self.decoder2_dec3(x3d)  # decoder3: bilinear upsample(2), 3*3conv, ReLU
        dec2_dec4 = self.decoder2_dec4(x4d)  # decoder4: bilinear upsample(4), 3*3conv, ReLU
        dec2_dec5 = self.decoder2_dec5(x5)  # decoder5: bilinear upsample(8), 3*3conv, ReLU

        x2d = torch.cat([dec2_enc1, dec2_enc2, dec2_dec3, dec2_dec4, dec2_dec5], dim=1)

        x2d = self.final_conv2(x2d)  # (B, 320, H/2, W/2)

        # Decoder1
        dec1_enc1 = self.decoder1_enc1(x1)  # encoder1: 3*3conv, ReLU
        dec1_dec2 = self.decoder1_dec2(x2d)  # decoder2: bilinear upsample(2), 3*3conv, ReLU
        dec1_dec3 = self.decoder1_dec3(x3d)  # decoder3: bilinear upsample(4), 3*3conv, ReLU
        dec1_dec4 = self.decoder1_dec4(x4d)  # decoder4: bilinear upsample(8), 3*3conv, ReLU
        dec1_dec5 = self.decoder1_dec5(x5)  # decoder5: bilinear upsample(16), 3*3conv, ReLU

        x1d = torch.cat([dec1_enc1, dec1_dec2, dec1_dec3, dec1_dec4, dec1_dec5], dim=1)

        x1d = self.final_conv1(x1d)  # (B, 320, H, W)

        logits = self.outc(x1d)
        return logits
