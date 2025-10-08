import torch
from torch.utils.data import Dataset, DataLoader
import pydicom
import numpy as np
import os
from sklearn.model_selection import train_test_split
from typing import List, Tuple
import cv2

class LiverCTDataset(Dataset):
    """肝CT数据集"""
    
    def __init__(self, data_dir: str, phase: str, transform=None, is_training: bool = True):
        self.data_dir = data_dir
        self.phase = phase  # 'portal' or 'plain'
        self.transform = transform
        self.is_training = is_training
        
        self.image_files = []
        self.mask_files = []
        
        self._load_file_paths()
        
    def _load_file_paths(self):
        """加载文件路径"""
        image_dir = os.path.join(self.data_dir, 'images')
        mask_dir = os.path.join(self.data_dir, 'masks')
        
        if not os.path.exists(image_dir) or not os.path.exists(mask_dir):
            raise ValueError(f"数据目录不存在: {image_dir} 或 {mask_dir}")
        
        # 获取所有图像文件
        for f in sorted(os.listdir(image_dir)):
            if f.endswith('.dcm') or f.endswith('.npy'):
                image_path = os.path.join(image_dir, f)
                mask_path = os.path.join(mask_dir, f.replace('.dcm', '.npy').replace('.npy', '_mask.npy'))
                
                if os.path.exists(mask_path):
                    self.image_files.append(image_path)
                    self.mask_files.append(mask_path)
    
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        # 加载图像
        if self.image_files[idx].endswith('.dcm'):
            image = self._load_dicom(self.image_files[idx])
        else:
            image = np.load(self.image_files[idx])
        
        # 加载标注
        mask = np.load(self.mask_files[idx])
        
        # 预处理
        image = self._preprocess_image(image)
        mask = self._preprocess_mask(mask)
        
        # 数据增强
        if self.transform and self.is_training:
            image, mask = self.transform(image, mask)
        
        return {
            'image': torch.FloatTensor(image),
            'mask': torch.FloatTensor(mask),
            'image_path': self.image_files[idx]
        }
    
    def _load_dicom(self, dicom_path: str) -> np.ndarray:
        """加载DICOM文件"""
        dicom_data = pydicom.dcmread(dicom_path)
        image = dicom_data.pixel_array.astype(np.float32)
        return image
    
    def _preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """预处理图像"""
        # 标准化
        image = (image - image.min()) / (image.max() - image.min() + 1e-8)
        # 调整尺寸
        if image.shape != (512, 512):
            image = cv2.resize(image, (512, 512), interpolation=cv2.INTER_LINEAR)
        return image
    
    def _preprocess_mask(self, mask: np.ndarray) -> np.ndarray:
        """预处理标注"""
        # 二值化
        mask = (mask > 0.5).astype(np.float32)
        # 调整尺寸
        if mask.shape != (512, 512):
            mask = cv2.resize(mask, (512, 512), interpolation=cv2.INTER_NEAREST)
        return mask

class CTDataLoader:
    """CT数据加载器"""
    
    def __init__(self, config: dict):
        self.config = config
        self.batch_size = config['training']['batch_size']
        self.num_workers = config['data']['num_workers']
    
    def get_data_loaders(self) -> Tuple[DataLoader, DataLoader, DataLoader]:
        """获取数据加载器"""
        # 这里根据你的数据组织结构调整
        portal_dataset = LiverCTDataset(
            self.config['paths']['portal_data'],
            'portal',
            is_training=True
        )
        
        plain_dataset = LiverCTDataset(
            self.config['paths']['plain_data'], 
            'plain',
            is_training=True
        )
        
        # 分割数据集
        portal_train, portal_val = train_test_split(
            range(len(portal_dataset)), 
            test_size=0.2, 
            random_state=42
        )
        
        plain_train, plain_val = train_test_split(
            range(len(plain_dataset)),
            test_size=0.2,
            random_state=42
        )
        
        # 创建数据加载器
        portal_train_loader = DataLoader(
            portal_dataset, batch_size=self.batch_size, sampler=portal_train,
            num_workers=self.num_workers, pin_memory=True
        )
        
        plain_train_loader = DataLoader(
            plain_dataset, batch_size=self.batch_size, sampler=plain_train,
            num_workers=self.num_workers, pin_memory=True
        )
        
        return portal_train_loader, plain_train_loader