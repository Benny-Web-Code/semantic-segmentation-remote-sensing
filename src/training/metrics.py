import torch
import numpy as np
from sklearn.metrics import precision_recall_fscore_support

class SegmentationMetrics:
    def __init__(self, num_classes):
        self.num_classes = num_classes

    def compute_iou(self, predictions, targets):
        predictions = predictions.argmax(dim=1).cpu().numpy().flatten()
        targets = targets.cpu().numpy().flatten()
        
        iou_list = []
        for i in range(self.num_classes):
            intersection = np.sum((predictions == i) & (targets == i))
            union = np.sum((predictions == i) | (targets == i))
            iou = intersection / (union + 1e-8)
            iou_list.append(iou)
        return torch.tensor(iou_list)

    def compute_miou(self, predictions, targets):
        iou = self.compute_iou(predictions, targets)
        return float(iou.mean())

    def compute_pixel_accuracy(self, predictions, targets):
        predictions = predictions.argmax(dim=1).cpu().numpy().flatten()
        targets = targets.cpu().numpy().flatten()
        return float(np.mean(predictions == targets))

    def compute_metrics(self, predictions, targets):
        predictions_np = predictions.argmax(dim=1).cpu().numpy().flatten()
        targets_np = targets.cpu().numpy().flatten()
        
        metrics = {}
        metrics['accuracy'] = float(np.mean(predictions_np == targets_np))
        
        iou_list = []
        for i in range(self.num_classes):
            intersection = np.sum((predictions_np == i) & (targets_np == i))
            union = np.sum((predictions_np == i) | (targets_np == i))
            iou = intersection / (union + 1e-8)
            iou_list.append(iou)
        
        metrics['miou'] = float(np.mean(iou_list))
        
        precision, recall, f1, _ = precision_recall_fscore_support(
            targets_np, predictions_np, average='weighted', zero_division=0
        )
        metrics['precision'] = float(precision)
        metrics['recall'] = float(recall)
        metrics['f1'] = float(f1)
        
        return metrics
