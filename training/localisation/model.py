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
    input : tensor shape of (batch_size, n_last*3, 320, 192)
    """
    def __init__(self, n_last=5, dropout_p=0.7):
        self.width = 320
        self.height = 192
        super(BallLocalisation, self).__init__()
        self.conv1 = nn.Conv2d(n_last * 3, 64, kernel_size=1, stride=1, padding="same")
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

class TTNetWithProb(nn.Module):
    def __init__(self, local_model):
        super(TTNetWithProb, self).__init__()
        
        self.backbone = nn.Sequential(
            local_model.conv1,
            local_model.norm,
            local_model.relu,
            local_model.conv_block1,
            local_model.conv_block2,
            local_model.dropout2d,
            local_model.conv_block3,
            local_model.conv_block4,
            local_model.dropout2d,
            local_model.conv_block5,
            local_model.conv_block6,
            local_model.dropout2d
        )

        self.head = nn.Sequential(
            local_model.fc1,
            local_model.relu, 
            local_model.dropout1d,
            local_model.fc2,
            local_model.relu, 
            local_model.dropout1d,
            local_model.fc3,
            local_model.sigmoid
        )
        
        self.fc_cls = nn.Sequential(
            nn.Linear(256*5*3, 1024),
            nn.ReLU(),
            nn.Linear(1024, 128),
            nn.ReLU(),
            nn.Linear(128, 2)
        )
    
    def forward(self, x):

        features = self.backbone(x)
        features = features.view(x.size(0), -1)

        logit = self.head(features)
        cls = self.fc_cls(features)

        return logit, cls

