import torch
import torch.nn as nn

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
    
    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.maxpool(x)
        return x

class BallLocalisation(nn.Module):
    """
    Ball localisation model
    input : tensor shape of (batch_size, 3, 320, 192)
    """
    def __init__(self, dropout_p=0.7):
        super(BallLocalisation, self).__init__()
        self.conv1 = ConvBlock(3, 64, kernel_size=1, stride=1, padding=0)
        self.norm = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()
        self.dropout2d = nn.Dropout2d(p=dropout_p)
        self.conv_block1 = ConvBlock(64, 64)
        self.conv_block2 = ConvBlock(64, 64)
        self.conv_block3 = ConvBlock(64, 128)
        self.conv_block4 = ConvBlock(128, 128)
        self.conv_block5 = ConvBlock(128, 256)
        self.conv_block6 = ConvBlock(256, 256)

        self.fc1 = nn.Linear(256*5*3, 2048)
        self.fc2 = nn.Linear(2048, 1024)
        self.fc3 = nn.Linear(1024, 512)
        self.dropout1d = nn.Dropout(p=dropout_p)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.norm(x)
        x = self.relu(x)

        x = self.conv_block1(x)
        x = self.conv_block2(x)
        x = self.dropout2d(x)

        x = self.conv_block3(x)
        x = self.conv_block4(x)
        x = self.dropout2d(x)

        x = self.conv_block5(x)
        x = self.conv_block6(x)
        x = self.dropout2d(x)

        x = x.view(x.size(0), -1)
        x = self.dropout1d(self.relu(self.fc1(x)))
        x = self.dropout1d(self.relu(self.fc2(x)))
        x = self.fc3(x)

        return self.sigmoid(x)

