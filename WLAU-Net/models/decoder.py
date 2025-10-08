import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple

class UNetDecoder(nn.Module):
    """
    U-net解码器：上采样、跳跃连接和分割输出
    """
    def __init__(self, decoder_channels: List[int] = [512, 256, 128, 64], 
                 output_channels: int = 1, use_skip_connections: bool = True):
        super(UNetDecoder, self).__init__()
        
        self.decoder_channels = decoder_channels
        self.output_channels = output_channels
        self.use_skip_connections = use_skip_connections
        
        # 上采样层
        self.upsample_layers = nn.ModuleList()
        self.conv_blocks = nn.ModuleList()
        
        # 构建解码器层
        for i in range(len(decoder_channels) - 1):
            in_ch = decoder_channels[i]
            out_ch = decoder_channels[i + 1]
            
            # 上采样 + 卷积块
            self.upsample_layers.append(
                UpsampleBlock(in_ch, out_ch)
            )
            
            self.conv_blocks.append(
                ConvBlock(out_ch * 2 if use_skip_connections else out_ch, out_ch)
            )
        
        # 最后一层上采样到原始尺寸
        self.final_upsample = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(decoder_channels[-1], decoder_channels[-1] // 2, 3, padding=1),
            nn.BatchNorm2d(decoder_channels[-1] // 2),
            nn.ReLU(inplace=True)
        )
        
        # 最终输出层
        self.output_conv = nn.Conv2d(decoder_channels[-1] // 2, output_channels, 1)
        
    def forward(self, decoder_features: List[torch.Tensor], 
                encoder_features: List[torch.Tensor]) -> torch.Tensor:
        """
        解码器前向传播
        
        Args:
            decoder_features: 解码器输入特征 [f4, f3, f2, f1] (从深到浅)
            encoder_features: 编码器特征 [f1, f2, f3, f4] (从浅到深)
            
        Returns:
            output: 分割输出 [B, output_channels, H, W]
        """
        # 反转编码器特征顺序以匹配解码器 [f4, f3, f2, f1]
        encoder_features_rev = encoder_features[::-1]
        
        x = decoder_features[0]  # 从最深层开始 f4
        
        # 逐层上采样
        for i, (upsample, conv_block) in enumerate(zip(self.upsample_layers, self.conv_blocks)):
            # 上采样
            x = upsample(x)  # 尺寸加倍，通道数减半
            
            # 跳跃连接（如果启用）
            if self.use_skip_connections and i < len(encoder_features_rev) - 1:
                skip_feature = encoder_features_rev[i + 1]  # 对应的编码器特征
                
                # 调整通道数（如果需要）
                if skip_feature.shape[1] != x.shape[1]:
                    skip_feature = self._adjust_channels(skip_feature, x.shape[1])
                
                # 拼接特征
                x = torch.cat([x, skip_feature], dim=1)
            
            # 卷积块
            x = conv_block(x)
        
        # 最终上采样到原始尺寸
        x = self.final_upsample(x)
        
        # 输出分割图
        output = self.output_conv(x)
        
        return output
    
    def _adjust_channels(self, feature: torch.Tensor, target_channels: int) -> torch.Tensor:
        """调整特征图通道数"""
        if feature.shape[1] != target_channels:
            adjust_conv = nn.Conv2d(feature.shape[1], target_channels, 1).to(feature.device)
            return adjust_conv(feature)
        return feature

class UpsampleBlock(nn.Module):
    """上采样块"""
    def __init__(self, in_channels: int, out_channels: int):
        super(UpsampleBlock, self).__init__()
        
        self.upsample = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.upsample(x)

class ConvBlock(nn.Module):
    """卷积块"""
    def __init__(self, in_channels: int, out_channels: int):
        super(ConvBlock, self).__init__()
        
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)

class SegmentationHead(nn.Module):
    """分割头：生成最终的分割掩码"""
    def __init__(self, in_channels: int, out_channels: int = 1):
        super(SegmentationHead, self).__init__()
        
        self.head = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // 2, 3, padding=1),
            nn.BatchNorm2d(in_channels // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // 2, out_channels, 1),
            nn.Sigmoid() if out_channels == 1 else nn.Softmax(dim=1)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(x)

class CompleteUNetDecoder(nn.Module):
    """
    完整的U-net解码器：集成所有组件
    """
    def __init__(self, decoder_channels: List[int] = [512, 256, 128, 64], 
                 output_channels: int = 1, use_skip_connections: bool = True):
        super(CompleteUNetDecoder, self).__init__()
        
        self.decoder = UNetDecoder(decoder_channels, output_channels, use_skip_connections)
        self.segmentation_head = SegmentationHead(output_channels, output_channels)
        
    def forward(self, decoder_features: List[torch.Tensor], 
                encoder_features: List[torch.Tensor]) -> torch.Tensor:
        """
        完整的前向传播
        
        Args:
            decoder_features: 解码器输入特征
            encoder_features: 编码器特征（用于跳跃连接）
            
        Returns:
            segmentation: 分割概率图 [B, output_channels, H, W]
        """
        # 通过解码器
        decoder_output = self.decoder(decoder_features, encoder_features)
        
        # 生成分割掩码
        segmentation = self.segmentation_head(decoder_output)
        
        return segmentation

class LiverTumorSegmentationModel(nn.Module):
    """
    完整的肝血管瘤分割模型：集成编码器、Transformer和解码器
    """
    def __init__(self, encoder_channels: List[int] = [64, 128, 256, 512],
                 hidden_dim: int = 512, output_channels: int = 1):
        super(LiverTumorSegmentationModel, self).__init__()
        
        self.encoder_channels = encoder_channels
        self.hidden_dim = hidden_dim
        
        # 编码器（使用之前实现的冻结编码器）
        self.encoder = None  # 将在外部设置
        
        # Transformer到解码器特征准备（使用之前实现的模块）
        self.transformer_to_decoder = None  # 将在外部设置
        
        # 解码器
        decoder_channels = [ch * 2 for ch in encoder_channels[::-1]]  # [1024, 512, 256, 128]
        self.decoder = CompleteUNetDecoder(
            decoder_channels=decoder_channels,
            output_channels=output_channels,
            use_skip_connections=True
        )
        
    def set_encoder(self, encoder):
        """设置编码器"""
        self.encoder = encoder
        
    def set_transformer_to_decoder(self, transformer_to_decoder):
        """设置Transformer到解码器转换模块"""
        self.transformer_to_decoder = transformer_to_decoder
        
    def forward(self, mixed_input: torch.Tensor, original_image: torch.Tensor) -> torch.Tensor:
        """
        完整模型的前向传播
        
        Args:
            mixed_input: 混合输入 [B, 2, H, W] (原始+增强)
            original_image: 原始图像 [B, 1, H, W] (用于高斯加权)
            
        Returns:
            segmentation: 分割输出 [B, output_channels, H, W]
        """
        # 1. 编码器特征提取
        encoder_features = self.encoder(mixed_input)  # [f1, f2, f3, f4]
        
        # 2. Transformer处理（这里简化表示，实际需要完整的Transformer管道）
        # transformer_output = self.transformer_pipeline(encoder_features, original_image)
        
        # 3. Transformer到解码器特征转换
        # decoder_features = self.transformer_to_decoder(transformer_output, encoder_features)
        
        # 为了演示，我们直接使用编码器特征作为解码器输入
        # 在实际使用中，这里应该是经过Transformer处理的特征
        decoder_features = [feat for feat in encoder_features[::-1]]  # 反转顺序 [f4, f3, f2, f1]
        
        # 4. 解码器生成分割结果
        segmentation = self.decoder(decoder_features, encoder_features)
        
        return segmentation

# 训练相关的辅助函数
class DiceLoss(nn.Module):
    """Dice损失"""
    def __init__(self, smooth: float = 1e-5):
        super(DiceLoss, self).__init__()
        self.smooth = smooth
        
    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        predictions = torch.sigmoid(predictions)
        
        # 展平
        predictions = predictions.view(-1)
        targets = targets.view(-1)
        
        intersection = (predictions * targets).sum()
        dice = (2. * intersection + self.smooth) / (predictions.sum() + targets.sum() + self.smooth)
        
        return 1 - dice

class CombinedLoss(nn.Module):
    """组合损失：Dice + BCE"""
    def __init__(self, dice_weight: float = 0.7, bce_weight: float = 0.3):
        super(CombinedLoss, self).__init__()
        self.dice_weight = dice_weight
        self.bce_weight = bce_weight
        self.dice_loss = DiceLoss()
        self.bce_loss = nn.BCEWithLogitsLoss()
        
    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        dice_loss = self.dice_loss(predictions, targets)
        bce_loss = self.bce_loss(predictions, targets)
        
        return self.dice_weight * dice_loss + self.bce_weight * bce_loss

# 使用示例
if __name__ == "__main__":
    # 参数设置
    batch_size = 2
    image_size = 512
    encoder_channels = [64, 128, 256, 512]
    output_channels = 1
    
    # 模拟输入
    mixed_input = torch.randn(batch_size, 2, image_size, image_size)
    original_image = torch.randn(batch_size, 1, image_size, image_size)
    
    # 模拟编码器特征
    encoder_features = [
        torch.randn(batch_size, 64, 256, 256),   # f1
        torch.randn(batch_size, 128, 128, 128),  # f2
        torch.randn(batch_size, 256, 64, 64),    # f3
        torch.randn(batch_size, 512, 32, 32)     # f4
    ]
    
    # 模拟解码器特征（经过Transformer处理后的）
    decoder_features = [
        torch.randn(batch_size, 1024, 32, 32),   # f4_ready
        torch.randn(batch_size, 512, 64, 64),    # f3_ready
        torch.randn(batch_size, 256, 128, 128),  # f2_ready
        torch.randn(batch_size, 128, 256, 256)   # f1_ready
    ]
    
    print("测试U-net解码器...")
    
    # 创建解码器
    decoder = CompleteUNetDecoder(
        decoder_channels=[1024, 512, 256, 128],
        output_channels=output_channels
    )
    
    # 前向传播
    segmentation_output = decoder(decoder_features, encoder_features)
    
    print(f"编码器特征尺寸: {[f.shape for f in encoder_features]}")
    print(f"解码器输入特征尺寸: {[f.shape for f in decoder_features]}")
    print(f"分割输出尺寸: {segmentation_output.shape}")
    print(f"分割输出范围: [{segmentation_output.min():.3f}, {segmentation_output.max():.3f}]")
    
    # 测试完整模型
    print("\n测试完整分割模型...")
    model = LiverTumorSegmentationModel(
        encoder_channels=encoder_channels,
        hidden_dim=512,
        output_channels=output_channels
    )
    
    # 模拟设置编码器（实际使用时需要真实的编码器）
    class MockEncoder(nn.Module):
        def forward(self, x):
            return [
                torch.randn(x.shape[0], 64, 256, 256),
                torch.randn(x.shape[0], 128, 128, 128),
                torch.randn(x.shape[0], 256, 64, 64),
                torch.randn(x.shape[0], 512, 32, 32)
            ]
    
    model.set_encoder(MockEncoder())
    
    # 完整模型前向传播
    final_output = model(mixed_input, original_image)
    
    print(f"最终分割输出尺寸: {final_output.shape}")
    print(f"最终输出范围: [{final_output.min():.3f}, {final_output.max():.3f}]")
    
    print("\nU-net解码器测试完成！")
    print("现在可以生成肝血管瘤的分割结果了。")