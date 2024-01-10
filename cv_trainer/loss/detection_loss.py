import torch.nn as nn

class BoundingBoxMSELoss(nn.Module):
    def __init__(self, class_safe=True):
        super(BoundingBoxMSELoss, self).__init__()
        self.class_safe = class_safe

    def forward(self, predicted, target):
        target_bbox, target_class = target
        predicted_bbox, predicted_class = predicted
        if self.class_safe:
            return nn.MSELoos()(predicted_bbox[target_bbox.bool()], target_bbox[target_bbox.bool()])

class DetectionLoss():
    def __init__(self, bbox_loss, class_loss = nn.BCELoss()):
        self.bbox_loss = bbox_loss
        self.class_loss = class_loss
    
    def __call__(self, predicted, target):
        target_bbox, target_class = target
        predicted_bbox, predicted_class = predicted
        return self.bbox_loss(predicted, target) + self.class_loss(predicted_class, target_class)