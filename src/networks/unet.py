import torch
import torch.nn as nn
import torch.nn.functional as F

class UNetExact(nn.Module):
    def __init__(self, in_channels=3, out_channels=2, norm="bn"):
        super().__init__()

        Norm = nn.BatchNorm2d if norm == "bn" else nn.GroupNorm

        def norm_layer(c):
            if norm == "bn":
                return Norm(c)
            # GroupNorm: 32 groups or fallback
            g = 32 if c >= 32 else 1
            return Norm(g, c)

        # Encoder
        self.enc1_conv1 = nn.Conv2d(in_channels, 64, 3, 1, 1)
        self.enc1_n1 = norm_layer(64)
        self.enc1_conv2 = nn.Conv2d(64, 64, 3, 1, 1)
        self.enc1_n2 = norm_layer(64)
        self.pool1 = nn.MaxPool2d(2, 2)

        self.enc2_conv1 = nn.Conv2d(64, 128, 3, 1, 1)
        self.enc2_n1 = norm_layer(128)
        self.enc2_conv2 = nn.Conv2d(128, 128, 3, 1, 1)
        self.enc2_n2 = norm_layer(128)
        self.pool2 = nn.MaxPool2d(2, 2)

        self.enc3_conv1 = nn.Conv2d(128, 256, 3, 1, 1)
        self.enc3_n1 = norm_layer(256)
        self.enc3_conv2 = nn.Conv2d(256, 256, 3, 1, 1)
        self.enc3_n2 = norm_layer(256)
        self.pool3 = nn.MaxPool2d(2, 2)

        self.enc4_conv1 = nn.Conv2d(256, 512, 3, 1, 1)
        self.enc4_n1 = norm_layer(512)
        self.enc4_conv2 = nn.Conv2d(512, 512, 3, 1, 1)
        self.enc4_n2 = norm_layer(512)
        self.pool4 = nn.MaxPool2d(2, 2)

        self.bottleneck_conv1 = nn.Conv2d(512, 1024, 3, 1, 1)
        self.bottleneck_n1 = norm_layer(1024)
        self.bottleneck_conv2 = nn.Conv2d(1024, 1024, 3, 1, 1)
        self.bottleneck_n2 = norm_layer(1024)

        # Decoder
        self.upconv1 = nn.ConvTranspose2d(1024, 512, 2, 2)
        self.dec1_conv1 = nn.Conv2d(1024, 512, 3, 1, 1)
        self.dec1_n1 = norm_layer(512)
        self.dec1_conv2 = nn.Conv2d(512, 512, 3, 1, 1)
        self.dec1_n2 = norm_layer(512)

        self.upconv2 = nn.ConvTranspose2d(512, 256, 2, 2)
        self.dec2_conv1 = nn.Conv2d(512, 256, 3, 1, 1)
        self.dec2_n1 = norm_layer(256)
        self.dec2_conv2 = nn.Conv2d(256, 256, 3, 1, 1)
        self.dec2_n2 = norm_layer(256)

        self.upconv3 = nn.ConvTranspose2d(256, 128, 2, 2)
        self.dec3_conv1 = nn.Conv2d(256, 128, 3, 1, 1)
        self.dec3_n1 = norm_layer(128)
        self.dec3_conv2 = nn.Conv2d(128, 128, 3, 1, 1)
        self.dec3_n2 = norm_layer(128)

        self.upconv4 = nn.ConvTranspose2d(128, 64, 2, 2)
        self.dec4_conv1 = nn.Conv2d(128, 64, 3, 1, 1)
        self.dec4_n1 = norm_layer(64)
        self.dec4_conv2 = nn.Conv2d(64, 64, 3, 1, 1)
        self.dec4_n2 = norm_layer(64)

        self.final_conv = nn.Conv2d(64, out_channels, 1, 1, 0)

    def forward(self, x):
        # Encoder
        enc1 = F.relu(self.enc1_n1(self.enc1_conv1(x)))
        enc1 = F.relu(self.enc1_n2(self.enc1_conv2(enc1)))
        pool1 = self.pool1(enc1)

        enc2 = F.relu(self.enc2_n1(self.enc2_conv1(pool1)))
        enc2 = F.relu(self.enc2_n2(self.enc2_conv2(enc2)))
        pool2 = self.pool2(enc2)

        enc3 = F.relu(self.enc3_n1(self.enc3_conv1(pool2)))
        enc3 = F.relu(self.enc3_n2(self.enc3_conv2(enc3)))
        pool3 = self.pool3(enc3)

        enc4 = F.relu(self.enc4_n1(self.enc4_conv1(pool3)))
        enc4 = F.relu(self.enc4_n2(self.enc4_conv2(enc4)))
        pool4 = self.pool4(enc4)

        bottleneck = F.relu(self.bottleneck_n1(self.bottleneck_conv1(pool4)))
        bottleneck = F.relu(self.bottleneck_n2(self.bottleneck_conv2(bottleneck)))

        # Decoder
        up1 = self.upconv1(bottleneck)
        up1 = torch.cat([up1, enc4], dim=1)
        dec1 = F.relu(self.dec1_n1(self.dec1_conv1(up1)))
        dec1 = F.relu(self.dec1_n2(self.dec1_conv2(dec1)))

        up2 = self.upconv2(dec1)
        up2 = torch.cat([up2, enc3], dim=1)
        dec2 = F.relu(self.dec2_n1(self.dec2_conv1(up2)))
        dec2 = F.relu(self.dec2_n2(self.dec2_conv2(dec2)))

        up3 = self.upconv3(dec2)
        up3 = torch.cat([up3, enc2], dim=1)
        dec3 = F.relu(self.dec3_n1(self.dec3_conv1(up3)))
        dec3 = F.relu(self.dec3_n2(self.dec3_conv2(dec3)))

        up4 = self.upconv4(dec3)
        up4 = torch.cat([up4, enc1], dim=1)
        dec4 = F.relu(self.dec4_n1(self.dec4_conv1(up4)))
        dec4 = F.relu(self.dec4_n2(self.dec4_conv2(dec4)))

        return self.final_conv(dec4)

    def get_bottleneck_features(self, x):
        """Return global pooled bottleneck features: [B, 1024]"""
        enc1 = F.relu(self.enc1_n1(self.enc1_conv1(x)))
        enc1 = F.relu(self.enc1_n2(self.enc1_conv2(enc1)))
        pool1 = self.pool1(enc1)

        enc2 = F.relu(self.enc2_n1(self.enc2_conv1(pool1)))
        enc2 = F.relu(self.enc2_n2(self.enc2_conv2(enc2)))
        pool2 = self.pool2(enc2)

        enc3 = F.relu(self.enc3_n1(self.enc3_conv1(pool2)))
        enc3 = F.relu(self.enc3_n2(self.enc3_conv2(enc3)))
        pool3 = self.pool3(enc3)

        enc4 = F.relu(self.enc4_n1(self.enc4_conv1(pool3)))
        enc4 = F.relu(self.enc4_n2(self.enc4_conv2(enc4)))
        pool4 = self.pool4(enc4)

        bottleneck = F.relu(self.bottleneck_n1(self.bottleneck_conv1(pool4)))
        bottleneck = F.relu(self.bottleneck_n2(self.bottleneck_conv2(bottleneck)))

        return torch.mean(bottleneck, dim=[2, 3])