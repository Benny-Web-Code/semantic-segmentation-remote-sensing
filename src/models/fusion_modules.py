import torch
import torch.nn as nn

class EarlyFusion(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)

class LateFusion(nn.Module):
    def __init__(self, num_streams, backbone='resnet50'):
        super().__init__()
        self.fusion_conv = nn.Sequential(
            nn.Conv2d(2048 * num_streams, 2048, 1),
            nn.BatchNorm2d(2048),
            nn.ReLU(inplace=True),
            nn.Conv2d(2048, 256, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )

    def forward(self, feat1, feat2, feat3):
        x = torch.cat([feat1, feat2, feat3], dim=1)
        return self.fusion_conv(x)

class DeepFusion(nn.Module):
    def __init__(self, num_streams, backbone='resnet50'):
        super().__init__()
        self.fusion_1x1 = nn.Sequential(
            nn.Conv2d(2048 * num_streams, 2048, 1),
            nn.BatchNorm2d(2048),
            nn.ReLU(inplace=True)
        )
        self.fusion_3x3 = nn.Sequential(
            nn.Conv2d(2048 * num_streams, 2048, 3, padding=1),
            nn.BatchNorm2d(2048),
            nn.ReLU(inplace=True)
        )
        self.fusion_combine = nn.Sequential(
            nn.Conv2d(2048 * 2, 512, 1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 256, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )

    def forward(self, feat1, feat2, feat3):
        x = torch.cat([feat1, feat2, feat3], dim=1)
        x_1x1 = self.fusion_1x1(x)
        x_3x3 = self.fusion_3x3(x)
        x = torch.cat([x_1x1, x_3x3], dim=1)
        return self.fusion_combine(x)
