import torch
import torch.nn as nn
from torchvision import models
from .fusion_modules import EarlyFusion, LateFusion, DeepFusion
from .attention import ChannelAttention, SpatialAttention

class MultiResolutionFusionNet(nn.Module):
    def __init__(self, num_classes=6, input_channels=[3, 8, 2], fusion_type='deep', backbone='resnet50', use_attention=True, dropout_rate=0.5):
        super().__init__()
        self.num_classes = num_classes
        self.fusion_type = fusion_type
        self.use_attention = use_attention
        
        self.encoder_rgb = self._build_encoder(input_channels[0], backbone)
        self.encoder_ms = self._build_encoder(input_channels[1], backbone)
        self.encoder_sar = self._build_encoder(input_channels[2], backbone)
        
        if fusion_type == 'early':
            self.fusion = EarlyFusion(sum(input_channels))
        elif fusion_type == 'late':
            self.fusion = LateFusion(3, backbone)
        else:
            self.fusion = DeepFusion(3, backbone)
        
        if use_attention:
            self.channel_attention = ChannelAttention(256)
            self.spatial_attention = SpatialAttention()
        
        self.decoder = self._build_decoder(256, num_classes, dropout_rate)

    def _build_encoder(self, in_channels, backbone_name):
        if backbone_name == 'resnet50':
            backbone = models.resnet50(pretrained=True)
        elif backbone_name == 'resnet34':
            backbone = models.resnet34(pretrained=True)
        else:
            backbone = models.resnet50(pretrained=True)
        
        if in_channels != 3:
            old_conv = backbone.conv1
            backbone.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        
        encoder = nn.Sequential(backbone.conv1, backbone.bn1, backbone.relu, backbone.maxpool,
                                backbone.layer1, backbone.layer2, backbone.layer3, backbone.layer4)
        return encoder

    def _build_decoder(self, encoder_channels, num_classes, dropout_rate):
        return nn.Sequential(
            nn.Conv2d(encoder_channels, 256, 3, padding=1),
            nn.BatchNorm2d(256), nn.ReLU(inplace=True), nn.Dropout2d(dropout_rate),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(256, 128, 3, padding=1),
            nn.BatchNorm2d(128), nn.ReLU(inplace=True), nn.Dropout2d(dropout_rate),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(128, 64, 3, padding=1),
            nn.BatchNorm2d(64), nn.ReLU(inplace=True), nn.Dropout2d(dropout_rate),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(64, 32, 3, padding=1),
            nn.BatchNorm2d(32), nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(32, num_classes, 1)
        )

    def forward(self, rgb, multispectral, sar):
        if self.fusion_type == 'early':
            x = torch.cat([rgb, multispectral, sar], dim=1)
            features = self.encoder_rgb(x)
        else:
            features_rgb = self.encoder_rgb(rgb)
            features_ms = self.encoder_ms(multispectral)
            features_sar = self.encoder_sar(sar)
            features = self.fusion(features_rgb, features_ms, features_sar)
        
        if self.use_attention:
            features = self.channel_attention(features) * features
            features = self.spatial_attention(features) * features
        
        output = self.decoder(features)
        return output

class UNetMultiFusion(nn.Module):
    def __init__(self, num_classes=6, input_channels=[3, 8, 2], use_attention=True):
        super().__init__()
        self.use_attention = use_attention
        
        self.enc1 = self._double_conv(sum(input_channels), 64)
        self.pool1 = nn.MaxPool2d(2)
        self.enc2 = self._double_conv(64, 128)
        self.pool2 = nn.MaxPool2d(2)
        self.enc3 = self._double_conv(128, 256)
        self.pool3 = nn.MaxPool2d(2)
        self.enc4 = self._double_conv(256, 512)
        self.pool4 = nn.MaxPool2d(2)
        
        self.bottleneck = self._double_conv(512, 1024)
        
        self.upconv4 = nn.ConvTranspose2d(1024, 512, 2, stride=2)
        self.dec4 = self._double_conv(1024, 512)
        self.upconv3 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.dec3 = self._double_conv(512, 256)
        self.upconv2 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.dec2 = self._double_conv(256, 128)
        self.upconv1 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.dec1 = self._double_conv(128, 64)
        
        self.final_conv = nn.Conv2d(64, num_classes, 1)
        
        if use_attention:
            from .attention import ChannelAttention
            self.attention4 = ChannelAttention(512)
            self.attention3 = ChannelAttention(256)
            self.attention2 = ChannelAttention(128)

    def _double_conv(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True)
        )

    def forward(self, rgb, multispectral, sar):
        x = torch.cat([rgb, multispectral, sar], dim=1)
        
        enc1 = self.enc1(x)
        x = self.pool1(enc1)
        enc2 = self.enc2(x)
        x = self.pool2(enc2)
        enc3 = self.enc3(x)
        x = self.pool3(enc3)
        enc4 = self.enc4(x)
        x = self.pool4(enc4)
        
        x = self.bottleneck(x)
        
        x = self.upconv4(x)
        if self.use_attention:
            enc4 = self.attention4(enc4) * enc4
        x = torch.cat([x, enc4], dim=1)
        x = self.dec4(x)
        
        x = self.upconv3(x)
        if self.use_attention:
            enc3 = self.attention3(enc3) * enc3
        x = torch.cat([x, enc3], dim=1)
        x = self.dec3(x)
        
        x = self.upconv2(x)
        if self.use_attention:
            enc2 = self.attention2(enc2) * enc2
        x = torch.cat([x, enc2], dim=1)
        x = self.dec2(x)
        
        x = self.upconv1(x)
        x = torch.cat([x, enc1], dim=1)
        x = self.dec1(x)
        
        return self.final_conv(x)
