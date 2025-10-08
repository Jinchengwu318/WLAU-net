import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class CTEncoder(nn.Module):
    """
    专门的CT编码器：4次下采样提取多尺度特征
    输入: 2通道 [原始图像, 增强图像]
    输出: 4个不同尺度的特征图
    """
    
    def __init__(self, in_channels=2, base_channels=64):
        super(CTEncoder, self).__init__()
        
        self.in_channels = in_channels
        self.base_channels = base_channels
        
        # 第一次下采样: 512x512 -> 256x256
        self.down1 = nn.Sequential(
            nn.Conv2d(in_channels, base_channels, 3, padding=1),
            nn.BatchNorm2d(base_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_channels, base_channels, 3, padding=1),
            nn.BatchNorm2d(base_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2)  # 下采样2倍
        )
        
        # 第二次下采样: 256x256 -> 128x128
        self.down2 = nn.Sequential(
            nn.Conv2d(base_channels, base_channels*2, 3, padding=1),
            nn.BatchNorm2d(base_channels*2),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_channels*2, base_channels*2, 3, padding=1),
            nn.BatchNorm2d(base_channels*2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2)  # 下采样2倍
        )
        
        # 第三次下采样: 128x128 -> 64x64
        self.down3 = nn.Sequential(
            nn.Conv2d(base_channels*2, base_channels*4, 3, padding=1),
            nn.BatchNorm2d(base_channels*4),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_channels*4, base_channels*4, 3, padding=1),
            nn.BatchNorm2d(base_channels*4),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2)  # 下采样2倍
        )
        
        # 第四次下采样: 64x64 -> 32x32
        self.down4 = nn.Sequential(
            nn.Conv2d(base_channels*4, base_channels*8, 3, padding=1),
            nn.BatchNorm2d(base_channels*8),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_channels*8, base_channels*8, 3, padding=1),
            nn.BatchNorm2d(base_channels*8),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2)  # 下采样2倍
        )
        
    def forward(self, x):
        """
        前向传播，返回4个不同尺度的特征图
        
        Args:
            x: 输入图像 [batch_size, 2, 512, 512]
            
        Returns:
            features: 包含4个特征图的列表
                - f1: [batch_size, base_channels, 256, 256]
                - f2: [batch_size, base_channels*2, 128, 128] 
                - f3: [batch_size, base_channels*4, 64, 64]
                - f4: [batch_size, base_channels*8, 32, 32]
        """
        # 第一次下采样
        f1 = self.down1(x)  # [B, 64, 256, 256]
        
        # 第二次下采样
        f2 = self.down2(f1)  # [B, 128, 128, 128]
        
        # 第三次下采样
        f3 = self.down3(f2)  # [B, 256, 64, 64]
        
        # 第四次下采样
        f4 = self.down4(f3)  # [B, 512, 32, 32]
        
        # 返回所有尺度的特征图
        return [f1, f2, f3, f4]

class EncoderTransfer:
    """
    编码器迁移学习：将门脉期编码器参数迁移到平扫期并冻结
    """
    
    def __init__(self, base_channels=64):
        self.base_channels = base_channels
        
        # 创建门脉期编码器（源编码器）
        self.portal_encoder = CTEncoder(in_channels=2, base_channels=base_channels)
        
        # 创建平扫期编码器（目标编码器）
        self.plain_encoder = CTEncoder(in_channels=2, base_channels=base_channels)
        
    def load_portal_encoder_weights(self, checkpoint_path: str):
        """
        加载在门脉期上预训练的编码器权重
        """
        checkpoint = torch.load(checkpoint_path)
        
        # 如果checkpoint包含整个模型，提取编码器部分
        if 'encoder_state_dict' in checkpoint:
            self.portal_encoder.load_state_dict(checkpoint['encoder_state_dict'])
        elif 'model_state_dict' in checkpoint:
            # 假设checkpoint保存的是整个模型，我们需要提取编码器部分
            portal_state_dict = {}
            for key, value in checkpoint['model_state_dict'].items():
                if key.startswith('encoder.'):
                    new_key = key.replace('encoder.', '')
                    portal_state_dict[new_key] = value
            self.portal_encoder.load_state_dict(portal_state_dict)
        else:
            # 直接加载编码器权重
            self.portal_encoder.load_state_dict(checkpoint)
            
        print("成功加载门脉期编码器权重")
        
    def transfer_and_freeze(self):
        """
        将门脉期编码器参数复制到平扫期编码器并冻结
        """
        # 复制权重
        portal_state_dict = self.portal_encoder.state_dict()
        self.plain_encoder.load_state_dict(portal_state_dict)
        
        # 冻结平扫期编码器的所有参数
        for param in self.plain_encoder.parameters():
            param.requires_grad = False
            
        print("成功完成编码器参数迁移和冻结")
        print(f"平扫期编码器参数冻结状态: {all(not p.requires_grad for p in self.plain_encoder.parameters())}")
        
    def get_plain_encoder(self) -> CTEncoder:
        """
        获取冻结后的平扫期编码器
        """
        return self.plain_encoder
    
    def get_portal_encoder(self) -> CTEncoder:
        """
        获取门脉期编码器（用于验证等）
        """
        return self.portal_encoder

class MultiScaleFeatureProcessor:
    """
    多尺度特征处理器：演示如何使用提取的特征
    """
    
    def __init__(self, encoder: CTEncoder):
        self.encoder = encoder
        
    def extract_features(self, mixed_input: torch.Tensor):
        """
        提取多尺度特征
        
        Args:
            mixed_input: 混合输入 [batch_size, 2, 512, 512]
            
        Returns:
            features: 4个尺度的特征图列表
        """
        with torch.no_grad():  # 因为编码器被冻结
            features = self.encoder(mixed_input)
        return features
    
    def analyze_features(self, features: list):
        """
        分析提取的特征（这里只是一个示例，你可以根据实际需求修改）
        """
        print("多尺度特征分析:")
        for i, feat in enumerate(features):
            print(f"特征图 {i+1}: 尺寸 {feat.shape}")
            
        # 这里可以添加你的特定特征处理逻辑
        # 例如：特征融合、注意力机制、特征选择等
        
        return features

# 数据准备函数
def prepare_mixed_input(original_images: np.ndarray, enhanced_images: np.ndarray) -> torch.Tensor:
    """
    准备混合输入：将原始图像和增强图像堆叠为2通道
    
    Args:
        original_images: 原始图像 [batch_size, 512, 512]
        enhanced_images: 增强图像 [batch_size, 512, 512]
        
    Returns:
        mixed_input: 混合输入 [batch_size, 2, 512, 512]
    """
    # 转换为torch tensor
    original_tensor = torch.FloatTensor(original_images).unsqueeze(1)  # [B, 1, 512, 512]
    enhanced_tensor = torch.FloatTensor(enhanced_images).unsqueeze(1)  # [B, 1, 512, 512]
    
    # 堆叠为2通道
    mixed_input = torch.cat([original_tensor, enhanced_tensor], dim=1)  # [B, 2, 512, 512]
    
    return mixed_input

# 使用示例
if __name__ == "__main__":
    # 初始化编码器迁移
    transfer = EncoderTransfer(base_channels=64)
    
    # 加载预训练的门脉期编码器权重
    PORTAL_CHECKPOINT = "path/to/portal_encoder_weights.pth"
    transfer.load_portal_encoder_weights(PORTAL_CHECKPOINT)
    
    # 执行参数迁移和冻结
    transfer.transfer_and_freeze()
    
    # 获取冻结后的平扫期编码器
    plain_encoder = transfer.get_plain_encoder()
    
    # 初始化特征处理器
    feature_processor = MultiScaleFeatureProcessor(plain_encoder)
    
    # 示例：处理一批数据
    batch_size = 4
    # 模拟输入数据（在实际使用中替换为真实数据）
    original_batch = np.random.randn(batch_size, 512, 512).astype(np.float32)
    enhanced_batch = np.random.randn(batch_size, 512, 512).astype(np.float32)
    
    # 准备混合输入
    mixed_input = prepare_mixed_input(original_batch, enhanced_batch)
    
    # 提取多尺度特征
    print("提取多尺度特征...")
    multi_scale_features = feature_processor.extract_features(mixed_input)
    
    # 分析特征
    feature_processor.analyze_features(multi_scale_features)
    
    print("\n编码器迁移完成！现在你可以使用提取的多尺度特征进行后续操作。")
    print("特征尺寸总结:")
    print("- Level 1: 256x256 (高分辨率细节)")
    print("- Level 2: 128x128 (中等分辨率特征)") 
    print("- Level 3: 64x64 (语义特征)")
    print("- Level 4: 32x32 (高级抽象特征)")