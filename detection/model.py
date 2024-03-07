import torchvision
import torch.nn as nn

from torchvision.models.detection.ssd import SSD, DefaultBoxGenerator, SSDHead
from torchvision.models import mobilenet_v3_small, MobileNet_V3_Small_Weights, mobilenet_v3_large, MobileNet_V3_Large_Weights



def get_model(n_classes=2, size=300, nms=0.45, backbone_name='resnet34'):
    if backbone_name == 'resnet34':
        backbone_model = torchvision.models.resnet34(pretrained=True)
        backbone = nn.Sequential(*list(backbone_model.children())[:-2])
        out_channels = 512
    elif backbone_name == 'mobilenet_v2':
        backbone_model = torchvision.models.mobilenet_v2(pretrained=True)
        backbone = backbone_model.features
        out_channels = 1280
    elif backbone_name == 'mobilenet_v3':
        backbone_model = mobilenet_v3_small(weights=MobileNet_V3_Small_Weights.IMAGENET1K_V1)
        backbone = backbone_model.features
        out_channels = 576
    elif backbone_name == 'mobilenet_v3_large':
        backbone_model = mobilenet_v3_large(weights=MobileNet_V3_Large_Weights.IMAGENET1K_V1)
        backbone = backbone_model.features
        out_channels = 960
    else:
        raise ValueError(f"Unknown backbone name: {backbone_name}")
    

    out_channels = [out_channels] * 6
    default_boxes = DefaultBoxGenerator(aspect_ratios=[[2], [2, 3], [2, 3], [2, 3], [2], [2]])
    num_anchors = default_boxes.num_anchors_per_location()
    head = SSDHead(out_channels, num_anchors, n_classes)
    model = SSD(backbone=backbone, 
                anchor_generator=default_boxes,
                size=(size, size),
                num_classes=n_classes,
                head=head,
                nms_thresh=nms)
    return model

if __name__ == '__main__':
    model = get_model(2, 300, backbone_name="mobilenet_v3_large")
    total_params = sum(p.numel() for p in model.parameters())
    print(f"{total_params:,} total parameters.")
    total_trainable_params = sum(
        p.numel() for p in model.parameters() if p.requires_grad)
    print(f"{total_trainable_params:,} training parameters.")
    print(model)

    
