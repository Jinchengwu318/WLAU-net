import matplotlib.pyplot as plt
import torch
import numpy as np
from typing import List

def visualize_results(original_image: torch.Tensor, prediction: torch.Tensor, 
                     ground_truth: torch.Tensor, save_path: str = None):
    """可视化结果"""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # 原始图像
    axes[0].imshow(original_image.cpu().numpy().squeeze(), cmap='gray')
    axes[0].set_title('Original Image')
    axes[0].axis('off')
    
    # 预测结果
    pred_mask = (torch.sigmoid(prediction) > 0.5).float().cpu().numpy().squeeze()
    axes[1].imshow(original_image.cpu().numpy().squeeze(), cmap='gray')
    axes[1].imshow(pred_mask, alpha=0.5, cmap='jet')
    axes[1].set_title('Prediction')
    axes[1].axis('off')
    
    # 真实标注
    axes[2].imshow(original_image.cpu().numpy().squeeze(), cmap='gray')
    axes[2].imshow(ground_truth.cpu().numpy().squeeze(), alpha=0.5, cmap='jet')
    axes[2].set_title('Ground Truth')
    axes[2].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    else:
        plt.show()
    
    plt.close()

def plot_training_curve(losses: List[float], metrics: Dict[str, List[float]], save_path: str = None):
    """绘制训练曲线"""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # 损失曲线
    axes[0].plot(losses)
    axes[0].set_title('Training Loss')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].grid(True)
    
    # 指标曲线
    for metric_name, metric_values in metrics.items():
        axes[1].plot(metric_values, label=metric_name)
    
    axes[1].set_title('Validation Metrics')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Score')
    axes[1].legend()
    axes[1].grid(True)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    else:
        plt.show()
    
    plt.close()