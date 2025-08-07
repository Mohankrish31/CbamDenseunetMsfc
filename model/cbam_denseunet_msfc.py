import torch
import torch.nn as nn
from models.cbam import cbam
from models.dense import denseblock
# === Multi-Scale Pooling Block ===
class MultiScalePool(nn.Module):
    def __init__(self, in_channels):
        super(MultiScalePool, self).__init__()
        self.pool1 = nn.AdaptiveAvgPool2d(1)
        self.pool2 = nn.AdaptiveAvgPool2d(2)
        self.pool3 = nn.AdaptiveAvgPool2d(4)
        self.conv1 = nn.Conv2d(in_channels, in_channels // 4, 1)
        self.conv2 = nn.Conv2d(in_channels, in_channels // 4, 1)
        self.conv3 = nn.Conv2d(in_channels, in_channels // 4, 1)
        self.out_conv = nn.Conv2d(in_channels + 3 * (in_channels // 4), in_channels, 1)
    def forward(self, x):
        size = x.size()[2:]
        p1 = nn.functional.interpolate(self.conv1(self.pool1(x)), size=size, mode='bilinear', align_corners=False)
        p2 = nn.functional.interpolate(self.conv2(self.pool2(x)), size=size, mode='bilinear', align_corners=False)
        p3 = nn.functional.interpolate(self.conv3(self.pool3(x)), size=size, mode='bilinear', align_corners=False)
        out = torch.cat([x, p1, p2, p3], dim=1)
        return self.out_conv(out)
# === Feature Compression (1x1 conv) ===
class FeatureCompressor(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(FeatureCompressor, self).__init__()
        self.compress = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1),
            nn.ReLU(inplace=True)
        )
    def forward(self, x):
        return self.compress(x)
# === Enhanced Decoder ===
class EnhancedDecoder(nn.Module):
    def __init__(self, in_channels, mid_channels, out_channels):
        super(EnhancedDecoder, self).__init__()
        self.decoder = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1),
            nn.Sigmoid()
        )
    def forward(self, x):
        return self.decoder(x)
# === Full Lightweight CBAM-DenseUNet with MSC, FC, and Enhanced Decoder ===
class cbam_denseunet(nn.Module):
    def __init__(self, in_channels=3, base_channels=32):
        super(cbam_denseunet, self).__init__()
        dense_out_channels = base_channels + 3 * 12  # denseblock + growth
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, base_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            denseblock(base_channels, growth_rate=12, num_layers=3),
            cbam(dense_out_channels),
            MultiScalePool(dense_out_channels)
        )
        self.feature_compression = FeatureCompressor(dense_out_channels, base_channels)
        self.decoder = EnhancedDecoder(
            in_channels=base_channels,
            mid_channels=base_channels // 2,
            out_channels=in_channels
        )
    def forward(self, x):
        enc = self.encoder(x)
        compressed = self.feature_compression(enc)
        dec = self.decoder(compressed)
        return dec + x  # Residual connection for stability
