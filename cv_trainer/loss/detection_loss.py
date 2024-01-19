import torch.nn as nn

class BoundingBoxMSELoss(nn.Module):
    def __init__(self, class_safe=True):
        super(BoundingBoxMSELoss, self).__init__()
        self.class_safe = class_safe

    def forward(self, target_bbox, target_class, predicted_bbox):
        if self.class_safe:
            return nn.MSELoos()(predicted_bbox[target_class.bool()], target_bbox[target_class.bool()])

class DetectionLoss():
    def __init__(self, bbox_loss, class_loss = nn.BCELoss()):
        self.bbox_loss = bbox_loss
        self.class_loss = class_loss
    
    def __call__(self, bbox, mark, predicted_bbox, predicted_mark):
        return self.bbox_loss(bbox, mark, predicted_bbox) + self.class_loss(predicted_mark, mark)