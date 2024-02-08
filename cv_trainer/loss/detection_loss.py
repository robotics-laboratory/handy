import torch.nn as nn

class BoundingBoxMSELoss(nn.Module):
    def __init__(self, class_safe=True):
        super(BoundingBoxMSELoss, self).__init__()
        self.class_safe = class_safe

    def forward(self, target_bbox, target_class, predicted_bbox):
        if self.class_safe:
            return nn.MSELoss()(predicted_bbox[target_class.bool()], target_bbox[target_class.bool()])

class DetectionLoss(nn.Module):
    def __init__(self, bbox_loss = BoundingBoxMSELoss(), class_loss = nn.CrossEntropyLoss()):
        super(DetectionLoss, self).__init__()
        self.bbox_loss = bbox_loss
        self.class_loss = class_loss
    
    def forward(self, bbox, mark, predicted_bbox, predicted_mark, **kwargs):
        return self.bbox_loss(bbox, mark, predicted_bbox) + self.class_loss(predicted_mark, mark)