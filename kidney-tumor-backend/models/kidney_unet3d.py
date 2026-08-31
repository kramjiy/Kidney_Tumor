"""
3D U-Net Architecture for Kidney Tumor Segmentation
FIXED: Corrected attention gate and decoder dimensions
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv3D(nn.Module):
    """Double 3D convolution block"""
    def __init__(self, in_channels, out_channels):
        super(DoubleConv3D, self).__init__()
        self.double_conv = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class AttentionGate3D(nn.Module):
    """Attention gate - FIXED dimensions"""
    def __init__(self, F_g, F_l, F_int):
        super(AttentionGate3D, self).__init__()
        
        self.W_g = nn.Sequential(
            nn.Conv3d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm3d(F_int)
        )
        
        self.W_x = nn.Sequential(
            nn.Conv3d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm3d(F_int)
        )
        
        self.psi = nn.Sequential(
            nn.Conv3d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm3d(1),
            nn.Sigmoid()
        )
        
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self, g, x):
        # g: gating signal (from decoder)
        # x: skip connection (from encoder)
        
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        
        # Upsample g1 to match x1 size if needed
        if g1.shape[2:] != x1.shape[2:]:
            g1 = F.interpolate(g1, size=x1.shape[2:], mode='trilinear', align_corners=True)
        
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        
        return x * psi


class UNet3D(nn.Module):
    """
    3D U-Net for volumetric segmentation - FIXED VERSION
    """
    def __init__(self, in_channels=1, out_channels=3, init_features=32, use_attention=False):
        super(UNet3D, self).__init__()
        
        self.use_attention = use_attention
        features = init_features
        
        # Encoder
        self.encoder1 = DoubleConv3D(in_channels, features)
        self.pool1 = nn.MaxPool3d(2)
        
        self.encoder2 = DoubleConv3D(features, features * 2)
        self.pool2 = nn.MaxPool3d(2)
        
        self.encoder3 = DoubleConv3D(features * 2, features * 4)
        self.pool3 = nn.MaxPool3d(2)
        
        self.encoder4 = DoubleConv3D(features * 4, features * 8)
        self.pool4 = nn.MaxPool3d(2)
        
        # Bottleneck
        self.bottleneck = DoubleConv3D(features * 8, features * 16)
        
        # Attention gates (if enabled)
        if use_attention:
            self.att4 = AttentionGate3D(F_g=features * 16, F_l=features * 8, F_int=features * 8)
            self.att3 = AttentionGate3D(F_g=features * 8, F_l=features * 4, F_int=features * 4)
            self.att2 = AttentionGate3D(F_g=features * 4, F_l=features * 2, F_int=features * 2)
            self.att1 = AttentionGate3D(F_g=features * 2, F_l=features, F_int=features)
        
        # Decoder
        self.upconv4 = nn.ConvTranspose3d(features * 16, features * 8, kernel_size=2, stride=2)
        self.decoder4 = DoubleConv3D(features * 16, features * 8)
        
        self.upconv3 = nn.ConvTranspose3d(features * 8, features * 4, kernel_size=2, stride=2)
        self.decoder3 = DoubleConv3D(features * 8, features * 4)
        
        self.upconv2 = nn.ConvTranspose3d(features * 4, features * 2, kernel_size=2, stride=2)
        self.decoder2 = DoubleConv3D(features * 4, features * 2)
        
        self.upconv1 = nn.ConvTranspose3d(features * 2, features, kernel_size=2, stride=2)
        self.decoder1 = DoubleConv3D(features * 2, features)
        
        # Output
        self.out = nn.Conv3d(features, out_channels, kernel_size=1)
        
    def forward(self, x):
        # Encoder path
        enc1 = self.encoder1(x)
        enc2 = self.encoder2(self.pool1(enc1))
        enc3 = self.encoder3(self.pool2(enc2))
        enc4 = self.encoder4(self.pool3(enc3))
        
        # Bottleneck
        bottleneck = self.bottleneck(self.pool4(enc4))
        
        # Decoder path with skip connections
        dec4 = self.upconv4(bottleneck)
        
        # Apply attention if enabled
        if self.use_attention:
            enc4 = self.att4(dec4, enc4)
        
        # Ensure spatial dimensions match before concatenation
        if dec4.shape[2:] != enc4.shape[2:]:
            dec4 = F.interpolate(dec4, size=enc4.shape[2:], mode='trilinear', align_corners=True)
        
        dec4 = torch.cat([dec4, enc4], dim=1)
        dec4 = self.decoder4(dec4)
        
        dec3 = self.upconv3(dec4)
        if self.use_attention:
            enc3 = self.att3(dec3, enc3)
        if dec3.shape[2:] != enc3.shape[2:]:
            dec3 = F.interpolate(dec3, size=enc3.shape[2:], mode='trilinear', align_corners=True)
        dec3 = torch.cat([dec3, enc3], dim=1)
        dec3 = self.decoder3(dec3)
        
        dec2 = self.upconv2(dec3)
        if self.use_attention:
            enc2 = self.att2(dec2, enc2)
        if dec2.shape[2:] != enc2.shape[2:]:
            dec2 = F.interpolate(dec2, size=enc2.shape[2:], mode='trilinear', align_corners=True)
        dec2 = torch.cat([dec2, enc2], dim=1)
        dec2 = self.decoder2(dec2)
        
        dec1 = self.upconv1(dec2)
        if self.use_attention:
            enc1 = self.att1(dec1, enc1)
        if dec1.shape[2:] != enc1.shape[2:]:
            dec1 = F.interpolate(dec1, size=enc1.shape[2:], mode='trilinear', align_corners=True)
        dec1 = torch.cat([dec1, enc1], dim=1)
        dec1 = self.decoder1(dec1)
        
        # Output
        out = self.out(dec1)
        
        return out


def test_unet():
    """Test the U-Net architecture"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Create model
    model = UNet3D(in_channels=1, out_channels=3, init_features=32, use_attention=True)
    model = model.to(device)
    
    # Test input
    x = torch.randn(1, 1, 64, 128, 128).to(device)
    
    # Forward pass
    output = model(x)
    
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    return model


if __name__ == "__main__":
    test_unet()