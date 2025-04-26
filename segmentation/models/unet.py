import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(out_channels),
        )

    def forward(self, x):
        return self.conv(x)


class UNet(nn.Module):
    def __init__(self, in_channels, out_channels, features_dim):
        super().__init__()

        self.encoder1 = ConvBlock(in_channels, features_dim)
        self.pull1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.encoder2 = ConvBlock(features_dim, features_dim * 2)
        self.pull2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.encoder3 = ConvBlock(features_dim * 2, features_dim * 4)
        self.pull3 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.encoder4 = ConvBlock(features_dim * 4, features_dim * 8)
        self.pull4 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.bottleneck = ConvBlock(features_dim * 8, features_dim * 16)

        self.upconv4 = nn.ConvTranspose2d(
            features_dim * 16, features_dim * 8, kernel_size=2, stride=2
        )
        self.decoder4 = ConvBlock(features_dim * 16, features_dim * 8)

        self.upconv3 = nn.ConvTranspose2d(
            features_dim * 8, features_dim * 4, kernel_size=2, stride=2
        )
        self.decoder3 = ConvBlock(features_dim * 8, features_dim * 4)

        self.upconv2 = nn.ConvTranspose2d(
            features_dim * 4, features_dim * 2, kernel_size=2, stride=2
        )
        self.decoder2 = ConvBlock(features_dim * 4, features_dim * 2)

        self.upconv1 = nn.ConvTranspose2d(
            features_dim * 2, features_dim, kernel_size=2, stride=2
        )
        self.decoder1 = ConvBlock(features_dim * 2, features_dim)

        self.out = nn.Conv2d(features_dim, out_channels, kernel_size=1)

    def forward(self, x):
        encoder1 = self.encoder1(x)
        pull1 = self.pull1(encoder1)

        encoder2 = self.encoder2(pull1)
        pull2 = self.pull2(encoder2)

        encoder3 = self.encoder3(pull2)
        pull3 = self.pull3(encoder3)

        encoder4 = self.encoder4(pull3)
        pull4 = self.pull4(encoder4)

        bottleneck = self.bottleneck(pull4)

        upconv4 = self.upconv4(bottleneck)
        decoder4 = self.decoder4(torch.cat([encoder4, upconv4], dim=1))

        upconv3 = self.upconv3(decoder4)
        decoder3 = self.decoder3(torch.cat([encoder3, upconv3], dim=1))

        upconv2 = self.upconv2(decoder3)
        decoder2 = self.decoder2(torch.cat([encoder2, upconv2], dim=1))

        upconv1 = self.upconv1(decoder2)
        decoder1 = self.decoder1(torch.cat([encoder1, upconv1], dim=1))

        out = self.out(decoder1)
        return out
