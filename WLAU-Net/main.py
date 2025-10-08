"""
WLAU-net
Author: [ZBH]

"""

import torch
import torch.nn as nn
import argparse
import yaml
from pathlib import Path
import sys
import os

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from models.complete_pipeline import LiverTumorSegmentationPipeline
from data.data_loader import CTDataLoader
from training.trainer import ModelTrainer
from utils.metrics import calculate_metrics
from utils.visualization import visualize_results

def load_config(config_path: str) -> dict:
    """加载配置文件"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def main():
    parser = argparse.ArgumentParser(description='Liver Tumor Segmentation Training')
    parser.add_argument('--config', type=str, required=True, help='Path to config file')
    parser.add_argument('--mode', type=str, choices=['train', 'test', 'inference'], 
                       default='train', help='Running mode')
    parser.add_argument('--checkpoint', type=str, help='Model checkpoint path')
    parser.add_argument('--data_dir', type=str, help='Data directory')
    
    args = parser.parse_args()
    
    # 加载配置
    config = load_config(args.config)
    
    # 创建模型
    print("Initializing model...")
    model = LiverTumorSegmentationPipeline(
        hidden_dim=config['model']['hidden_dim'],
        encoder_channels=config['model']['encoder_channels'],
        num_transformer_layers=config['model']['num_transformer_layers'],
        lambda1=config['model']['wavelet_lambda1'],
        lambda2=config['model']['wavelet_lambda2']
    )
    
    # 根据模式运行
    if args.mode == 'train':
        trainer = ModelTrainer(model, config)
        trainer.train()
    elif args.mode == 'test':
        trainer = ModelTrainer(model, config)
        trainer.test(args.checkpoint)
    elif args.mode == 'inference':
        # 推理模式
        pass

if __name__ == "__main__":
    main()