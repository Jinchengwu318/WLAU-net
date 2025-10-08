import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import List, Tuple, Optional
import numpy as np

class TransformerEncoder(nn.Module):
    """
    12层Transformer编码器
    """
    def __init__(self, hidden_dim: int, num_layers: int = 12, num_heads: int = 8, 
                 mlp_ratio: float = 4.0, dropout: float = 0.1):
        super(TransformerEncoder, self).__init__()
        
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # Transformer层
        self.layers = nn.ModuleList([
            TransformerLayer(hidden_dim, num_heads, mlp_ratio, dropout)
            for _ in range(num_layers)
        ])
        
        self.norm = nn.LayerNorm(hidden_dim)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        Args:
            x: 输入序列 [B, N, hidden_dim]
            
        Returns:
            transformed_x: Transformer输出 [B, N, hidden_dim]
        """
        for layer in self.layers:
            x = layer(x)
        
        return self.norm(x)

class TransformerLayer(nn.Module):
    """单个Transformer层"""
    def __init__(self, hidden_dim: int, num_heads: int, mlp_ratio: float, dropout: float):
        super(TransformerLayer, self).__init__()
        
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.attention = nn.MultiheadAttention(hidden_dim, num_heads, dropout=dropout, batch_first=True)
        
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.mlp = MLP(hidden_dim, int(hidden_dim * mlp_ratio), hidden_dim, dropout)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 自注意力
        attn_out, _ = self.attention(
            query=self.norm1(x),
            key=self.norm1(x),
            value=self.norm1(x),
            need_weights=False
        )
        x = x + self.dropout(attn_out)
        
        # MLP
        mlp_out = self.mlp(self.norm2(x))
        x = x + self.dropout(mlp_out)
        
        return x

class MLP(nn.Module):
    """多层感知机"""
    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int, dropout: float):
        super(MLP, self).__init__()
        
        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.act = nn.GELU()
        self.drop1 = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_dim, out_dim)
        self.drop2 = nn.Dropout(dropout)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x

class HiddenFeatureTransform(nn.Module):
    """
    隐藏特征变换：将Transformer输出转换为解码器输入
    """
    def __init__(self, hidden_dim: int, decoder_dims: List[int]):
        """
        Args:
            hidden_dim: Transformer隐藏维度
            decoder_dims: 解码器各层期望的通道数 [f4_dim, f3_dim, f2_dim, f1_dim]
        """
        super(HiddenFeatureTransform, self).__init__()
        
        self.hidden_dim = hidden_dim
        self.decoder_dims = decoder_dims
        
        # 隐藏特征变换
        self.hidden_transform = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim)
        )
        
        # 特征分割：将token序列分割回各尺度特征图
        self.feature_splits = [32*32, 64*64, 128*128, 256*256]  # 各尺度token数量
        
    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        """
        将Transformer输出转换为多尺度特征图
        
        Args:
            x: Transformer输出 [B, N, hidden_dim]
            
        Returns:
            features: 多尺度特征图列表 [f4, f3, f2, f1]
        """
        batch_size = x.shape[0]
        
        # 隐藏特征变换
        x_transformed = self.hidden_transform(x)  # [B, N, hidden_dim]
        
        # 分割token序列为各尺度特征
        features = []
        start_idx = 0
        
        for i, num_tokens in enumerate(self.feature_splits):
            # 提取当前尺度的token
            end_idx = start_idx + num_tokens
            scale_tokens = x_transformed[:, start_idx:end_idx, :]  # [B, num_tokens, hidden_dim]
            
            # 重塑为特征图
            if i == 0:  # f4: 32x32
                H, W = 32, 32
            elif i == 1:  # f3: 64x64
                H, W = 64, 64
            elif i == 2:  # f2: 128x128
                H, W = 128, 128
            else:  # f1: 256x256
                H, W = 256, 256
                
            feature_map = scale_tokens.transpose(1, 2).view(batch_size, self.hidden_dim, H, W)
            features.append(feature_map)
            
            start_idx = end_idx
        
        # 反转特征列表顺序以匹配解码器期望的顺序 [f4, f3, f2, f1]
        return features[::-1]

class FeatureReshapeForDecoder(nn.Module):
    """
    特征重整形：调整特征图通道数以匹配U-net解码器
    """
    def __init__(self, hidden_dim: int, decoder_channels: List[int]):
        """
        Args:
            hidden_dim: 输入特征图通道数
            decoder_channels: 解码器各层期望的通道数 [f4_out, f3_out, f2_out, f1_out]
        """
        super(FeatureReshapeForDecoder, self).__init__()
        
        self.hidden_dim = hidden_dim
        self.decoder_channels = decoder_channels
        
        # 通道调整卷积
        self.channel_adjustments = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(hidden_dim, out_channels, 3, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_channels, out_channels, 3, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)
            ) for out_channels in decoder_channels
        ])
        
    def forward(self, features: List[torch.Tensor]) -> List[torch.Tensor]:
        """
        调整特征图通道数以匹配解码器
        
        Args:
            features: 输入特征图列表 [f4, f3, f2, f1]
            
        Returns:
            adjusted_features: 调整后的特征图列表 [f4_adj, f3_adj, f2_adj, f1_adj]
        """
        adjusted_features = []
        
        for i, (feature, adjust_conv) in enumerate(zip(features, self.channel_adjustments)):
            adjusted_feature = adjust_conv(feature)
            adjusted_features.append(adjusted_feature)
            
        return adjusted_features

class UNetDecoderFeaturePreparer(nn.Module):
    """
    U-net解码器特征准备器：完整的Transformer到解码器特征转换
    """
    def __init__(self, hidden_dim: int, num_transformer_layers: int = 12, 
                 num_heads: int = 8, decoder_channels: List[int] = None):
        super(UNetDecoderFeaturePreparer, self).__init__()
        
        # 默认解码器通道数（与编码器对称）
        if decoder_channels is None:
            decoder_channels = [512, 256, 128, 64]  # [f4, f3, f2, f1]
        
        self.hidden_dim = hidden_dim
        self.decoder_channels = decoder_channels
        
        # 12层Transformer编码器
        self.transformer = TransformerEncoder(
            hidden_dim=hidden_dim,
            num_layers=num_transformer_layers,
            num_heads=num_heads
        )
        
        # 隐藏特征变换
        self.hidden_transform = HiddenFeatureTransform(hidden_dim, decoder_channels)
        
        # 特征重整形
        self.feature_reshape = FeatureReshapeForDecoder(hidden_dim, decoder_channels)
        
    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        """
        完整的Transformer到解码器特征转换
        
        Args:
            x: Transformer输入序列 [B, N, hidden_dim]
            
        Returns:
            decoder_features: 解码器可处理的特征图列表 [f4, f3, f2, f1]
        """
        # 通过12层Transformer
        transformer_out = self.transformer(x)  # [B, N, hidden_dim]
        
        # 隐藏特征变换和序列分割
        features = self.hidden_transform(transformer_out)  # [f4, f3, f2, f1]
        
        # 特征重整形以匹配解码器通道数
        decoder_features = self.feature_reshape(features)
        
        return decoder_features

class SkipConnectionFusion(nn.Module):
    """
    跳跃连接融合：将编码器特征与解码器准备的特征融合
    """
    def __init__(self, encoder_channels: List[int], decoder_channels: List[int]):
        super(SkipConnectionFusion, self).__init__()
        
        self.fusion_convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(enc_ch + dec_ch, dec_ch, 3, padding=1),
                nn.BatchNorm2d(dec_ch),
                nn.ReLU(inplace=True),
                nn.Conv2d(dec_ch, dec_ch, 3, padding=1),
                nn.BatchNorm2d(dec_ch),
                nn.ReLU(inplace=True)
            ) for enc_ch, dec_ch in zip(encoder_channels, decoder_channels)
        ])
        
    def forward(self, encoder_features: List[torch.Tensor], 
                decoder_features: List[torch.Tensor]) -> List[torch.Tensor]:
        """
        融合编码器和解码器特征
        
        Args:
            encoder_features: 编码器特征 [f1, f2, f3, f4] (注意顺序与解码器相反)
            decoder_features: 解码器准备的特征 [f4, f3, f2, f1]
            
        Returns:
            fused_features: 融合后的特征 [f4_fused, f3_fused, f2_fused, f1_fused]
        """
        fused_features = []
        
        # 反转编码器特征顺序以匹配解码器
        encoder_features_rev = encoder_features[::-1]  # [f4, f3, f2, f1]
        
        for i, (enc_feat, dec_feat, fusion_conv) in enumerate(
            zip(encoder_features_rev, decoder_features, self.fusion_convs)):
            
            # 拼接特征
            combined = torch.cat([enc_feat, dec_feat], dim=1)
            
            # 融合卷积
            fused = fusion_conv(combined)
            fused_features.append(fused)
            
        return fused_features

# 完整的Transformer到U-net解码器管道
class TransformerToUNetPipeline(nn.Module):
    """
    完整的Transformer到U-net解码器管道
    """
    def __init__(self, hidden_dim: int, encoder_channels: List[int], 
                 num_transformer_layers: int = 12, num_heads: int = 8):
        super(TransformerToUNetPipeline, self).__init__()
        
        self.hidden_dim = hidden_dim
        self.encoder_channels = encoder_channels
        
        # 解码器通道数（通常与编码器对称）
        decoder_channels = [ch * 2 for ch in encoder_channels[::-1]]  # [512, 256, 128, 64]
        
        # Transformer到解码器特征准备
        self.feature_preparer = UNetDecoderFeaturePreparer(
            hidden_dim=hidden_dim,
            num_transformer_layers=num_transformer_layers,
            num_heads=num_heads,
            decoder_channels=decoder_channels
        )
        
        # 跳跃连接融合
        self.skip_fusion = SkipConnectionFusion(encoder_channels, decoder_channels)
        
    def forward(self, transformer_input: torch.Tensor, 
                encoder_features: List[torch.Tensor]) -> List[torch.Tensor]:
        """
        完整的管道前向传播
        
        Args:
            transformer_input: Transformer输入序列 [B, N, hidden_dim]
            encoder_features: 编码器特征列表 [f1, f2, f3, f4]
            
        Returns:
            decoder_ready_features: 解码器可用的特征列表 [f4_ready, f3_ready, f2_ready, f1_ready]
        """
        # 准备解码器特征
        decoder_features = self.feature_preparer(transformer_input)
        
        # 融合跳跃连接
        fused_features = self.skip_fusion(encoder_features, decoder_features)
        
        return fused_features

# 使用示例
if __name__ == "__main__":
    # 参数设置
    hidden_dim = 512
    encoder_channels = [64, 128, 256, 512]  # 编码器各层通道数
    batch_size = 2
    
    # 模拟输入
    transformer_input = torch.randn(batch_size, 86080, hidden_dim)  # 86080 = 256*256 + 128*128 + 64*64 + 32*32
    
    # 模拟编码器特征
    encoder_features = [
        torch.randn(batch_size, 64, 256, 256),   # f1
        torch.randn(batch_size, 128, 128, 128),  # f2
        torch.randn(batch_size, 256, 64, 64),    # f3
        torch.randn(batch_size, 512, 32, 32)     # f4
    ]
    
    # 创建管道
    pipeline = TransformerToUNetPipeline(
        hidden_dim=hidden_dim,
        encoder_channels=encoder_channels,
        num_transformer_layers=12,
        num_heads=8
    )
    
    print("进行Transformer到U-net解码器的特征转换...")
    
    # 前向传播
    decoder_ready_features = pipeline(transformer_input, encoder_features)
    
    print(f"输入Transformer序列尺寸: {transformer_input.shape}")
    print(f"编码器特征尺寸: {[f.shape for f in encoder_features]}")
    print(f"解码器就绪特征尺寸: {[f.shape for f in decoder_ready_features]}")
    
    print("\n特征转换完成！现在可以输入到U-net解码器了。")
    print("特征顺序: [f4_ready, f3_ready, f2_ready, f1_ready]")
    print("这些特征可以直接用于U-net解码器的上采样和跳跃连接。")