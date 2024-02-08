from .base import BaseMetric

class IoU(BaseMetric):
    def __init__(self, name=None, *args, **kwargs):
        super().__init__(name, *args, **kwargs)
    
    def __call__(self, bbox, predicted_bbox, **kwargs):
        if bbox.dim() == 1:
            bbox = bbox.unsqueeze(0)
        if predicted_bbox.dim() == 1:
            predicted_bbox = predicted_bbox.unsqueeze(0)
        # Calculate IoU for a single pair of bounding boxes
        def calculate_iou(box1, box2):
            # Calculate intersection coordinates
            x1 = max(box1[0], box2[0])
            y1 = max(box1[1], box2[1])
            x2 = min(box1[2], box2[2])
            y2 = min(box1[3], box2[3])
            
            # Calculate intersection area
            intersection_area = max(0, x2 - x1 + 1) * max(0, y2 - y1 + 1)
            
            # Calculate union area
            box1_area = (box1[2] - box1[0] + 1) * (box1[3] - box1[1] + 1)
            box2_area = (box2[2] - box2[0] + 1) * (box2[3] - box2[1] + 1)
            union_area = box1_area + box2_area - intersection_area
            
            # Calculate IoU
            iou = intersection_area / (union_area + 1e6)
            return iou
        
        # Calculate IoU for each pair of bounding boxes in the batch
        iou_scores = []
        for i in range(len(bbox)):
            iou_scores.append(calculate_iou(bbox[i], predicted_bbox[i]))
        
        mean_iou = sum(iou_scores) / len(iou_scores)
        return mean_iou
        