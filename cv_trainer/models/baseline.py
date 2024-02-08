import torch.nn as nn

class BaseLine(nn.Module):
    def __init__(self, kernel_size, channels, pool_stride, hidden_size):
        super(BaseLine, self).__init__()
        assert len(kernel_size) == len(channels) == len(pool_stride), "kernel_size, channels and pool_stride should have the same length"
        blocks = [nn.BatchNorm2d(3)]
        for i in range(len(kernel_size)):
            in_channels, out_channels = channels[i]
            blocks.append(nn.Conv2d(in_channels, out_channels, kernel_size[i], padding="same"))
            blocks.append(nn.ReLU())
            blocks.append(nn.MaxPool2d(2, stride=pool_stride[i]))
        
        self.feture_extractor = nn.Sequential(*blocks)
        self.fc1 = nn.Linear(hidden_size, 64)
        self.classifier_head = nn.Linear(64, 2)
        self.bbox_head = nn.Linear(64, 4)

    def forward(self, image, **kwargs):
        x = image / 255.0
        x = self.feture_extractor(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        class_output = self.classifier_head(x)
        bbox_output = self.bbox_head(x)
        return {
            "predicted_bbox": bbox_output,
            "predicted_mark": class_output
        }
