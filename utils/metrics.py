import torch.nn.functional as F

class SegmentationMetrics:
    def __init__(self, num_classes):
        self.num_classes = num_classes
    
    def _one_hot_encode(self, tensor):
        return F.one_hot(tensor, self.num_classes).permute(0, 3, 1, 2)
    
    def calculate_metrics(self, pred, target):
        pred = pred.argmax(dim=1)
        metrics = {}
        
        # IoU for each class
        for cls in range(self.num_classes):
            pred_cls = (pred == cls)
            target_cls = (target == cls)
            intersection = (pred_cls & target_cls).sum().float()
            union = (pred_cls | target_cls).sum().float()
            iou = (intersection + 1e-7) / (union + 1e-7)
            metrics[f'iou_class_{cls}'] = iou.item()
        
        # Mean IoU
        metrics['mean_iou'] = sum(metrics[f'iou_class_{cls}'] 
                                for cls in range(self.num_classes)) / self.num_classes
        
        # Pixel Accuracy
        correct = (pred == target).sum().float()
        total = target.numel()
        metrics['pixel_accuracy'] = (correct / total).item()
        
        return metrics
