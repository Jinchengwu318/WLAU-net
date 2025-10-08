import torch
import numpy as np
from scipy.spatial.distance import directed_hausdorff
from typing import Dict

def calculate_dice(predictions: torch.Tensor, targets: torch.Tensor) -> float:
    """计算Dice系数"""
    predictions = (torch.sigmoid(predictions) > 0.5).float()
    targets = targets.float()
    
    intersection = (predictions * targets).sum()
    union = predictions.sum() + targets.sum()
    
    dice = (2. * intersection) / (union + 1e-8)
    return dice.item()

def calculate_iou(predictions: torch.Tensor, targets: torch.Tensor) -> float:
    """计算IoU"""
    predictions = (torch.sigmoid(predictions) > 0.5).float()
    targets = targets.float()
    
    intersection = (predictions * targets).sum()
    union = (predictions + targets).sum() - intersection
    
    iou = intersection / (union + 1e-8)
    return iou.item()

def calculate_hd(predictions: torch.Tensor, targets: torch.Tensor) -> float:
    """计算Hausdorff距离"""
    predictions = (torch.sigmoid(predictions) > 0.5).float().cpu().numpy()
    targets = targets.float().cpu().numpy()
    
    hd_distances = []
    for i in range(predictions.shape[0]):
        pred_coords = np.argwhere(predictions[i, 0] > 0)
        target_coords = np.argwhere(targets[i, 0] > 0)
        
        if len(pred_coords) == 0 or len(target_coords) == 0:
            hd_distances.append(np.inf)
            continue
        
        hd1 = directed_hausdorff(pred_coords, target_coords)[0]
        hd2 = directed_hausdorff(target_coords, pred_coords)[0]
        hd_distances.append(max(hd1, hd2))
    
    return float(np.mean(hd_distances))

def calculate_acc(predictions: torch.Tensor, targets: torch.Tensor) -> float:
    """计算准确率"""
    predictions = (torch.sigmoid(predictions) > 0.5).float()
    targets = targets.float()
    
    correct = (predictions == targets).float()
    accuracy = correct.sum() / correct.numel()
    
    return accuracy.item()

def calculate_metrics(predictions: torch.Tensor, targets: torch.Tensor) -> Dict[str, float]:
    """计算所有指标"""
    return {
        'dice': calculate_dice(predictions, targets),
        'iou': calculate_iou(predictions, targets),
        'hd': calculate_hd(predictions, targets),
        'acc': calculate_acc(predictions, targets)
    }