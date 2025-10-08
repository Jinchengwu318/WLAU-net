import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.optim as optim
from typing import Dict, List
import os
import time
from tqdm import tqdm

from utils.metrics import calculate_dice, calculate_iou, calculate_hd, calculate_acc

class ModelTrainer:
    """模型训练器"""
    
    def __init__(self, model, config: Dict):
        self.model = model
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # 移动到设备
        self.model.to(self.device)
        
        # 设置优化器
        self.optimizer = optim.Adam(
            model.parameters(),
            lr=config['training']['learning_rate'],
            weight_decay=config['training']['weight_decay']
        )
        
        # 损失函数
        self.dice_loss = DiceLoss()
        self.bce_loss = nn.BCEWithLogitsLoss()
        
        self.dice_weight = config['training']['dice_weight']
        self.bce_weight = config['training']['bce_weight']
        
        # 检查点目录
        self.checkpoint_dir = config['paths']['checkpoints']
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        
    def train(self):
        """训练模型"""
        print("开始训练...")
        
        # 获取数据加载器
        from data.data_loader import CTDataLoader
        data_loader = CTDataLoader(self.config)
        portal_loader, plain_loader = data_loader.get_data_loaders()
        
        best_dice = 0.0
        patience_counter = 0
        
        for epoch in range(self.config['training']['num_epochs']):
            # 训练阶段
            train_loss = self._train_epoch(plain_loader, epoch)
            
            # 验证阶段
            val_metrics = self._validate(portal_loader)  # 使用门脉期验证
            
            # 保存最佳模型
            if val_metrics['dice'] > best_dice:
                best_dice = val_metrics['dice']
                self._save_checkpoint(epoch, val_metrics, is_best=True)
                patience_counter = 0
            else:
                patience_counter += 1
            
            # 早停
            if patience_counter >= self.config['training']['patience']:
                print(f"早停于第 {epoch} 轮")
                break
            
            # 打印进度
            print(f"Epoch {epoch+1}/{self.config['training']['num_epochs']}")
            print(f"训练损失: {train_loss:.4f}")
            print(f"验证指标 - Dice: {val_metrics['dice']:.4f}, IoU: {val_metrics['iou']:.4f}")
    
    def _train_epoch(self, data_loader: DataLoader, epoch: int) -> float:
        """训练一个epoch"""
        self.model.train()
        total_loss = 0
        
        pbar = tqdm(data_loader, desc=f"Epoch {epoch+1}")
        for batch in pbar:
            images = batch['image'].to(self.device)
            masks = batch['mask'].to(self.device)
            
            self.optimizer.zero_grad()
            
            # 前向传播
            predictions, losses = self.model(images, Y_s=masks)
            
            # 计算分割损失
            dice_loss = self.dice_loss(predictions, masks)
            bce_loss = self.bce_loss(predictions, masks)
            segmentation_loss = self.dice_weight * dice_loss + self.bce_weight * bce_loss
            
            # 总损失
            total_batch_loss = segmentation_loss
            if 'attention' in losses:
                total_batch_loss += losses['attention']
            
            # 反向传播
            total_batch_loss.backward()
            self.optimizer.step()
            
            total_loss += total_batch_loss.item()
            
            pbar.set_postfix({'loss': total_batch_loss.item()})
        
        return total_loss / len(data_loader)
    
    def _validate(self, data_loader: DataLoader) -> Dict:
        """验证模型"""
        self.model.eval()
        all_predictions = []
        all_masks = []
        
        with torch.no_grad():
            for batch in data_loader:
                images = batch['image'].to(self.device)
                masks = batch['mask'].to(self.device)
                
                predictions, _ = self.model(images)
                all_predictions.append(predictions.cpu())
                all_masks.append(masks.cpu())
        
        # 计算指标
        predictions = torch.cat(all_predictions)
        masks = torch.cat(all_masks)
        
        metrics = {
            'dice': calculate_dice(predictions, masks),
            'iou': calculate_iou(predictions, masks),
            'hd': calculate_hd(predictions, masks),
            'acc': calculate_acc(predictions, masks)
        }
        
        return metrics
    
    def _save_checkpoint(self, epoch: int, metrics: Dict, is_best: bool = False):
        """保存检查点"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'metrics': metrics
        }
        
        filename = f"checkpoint_epoch_{epoch}.pth"
        if is_best:
            filename = "best_model.pth"
        
        torch.save(checkpoint, os.path.join(self.checkpoint_dir, filename))
        print(f"模型已保存: {filename}")

class DiceLoss(nn.Module):
    """Dice损失"""
    def __init__(self, smooth: float = 1e-5):
        super(DiceLoss, self).__init__()
        self.smooth = smooth
        
    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        predictions = torch.sigmoid(predictions)
        
        # 展平
        predictions = predictions.contiguous().view(-1)
        targets = targets.contiguous().view(-1)
        
        intersection = (predictions * targets).sum()
        dice = (2. * intersection + self.smooth) / (predictions.sum() + targets.sum() + self.smooth)
        
        return 1 - dice