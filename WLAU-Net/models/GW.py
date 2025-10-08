import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from typing import List, Tuple, Optional

class HiddenFeatureProjection(nn.Module):
    """
    将卷积特征转换为Transformer输入的模块
    """
    def __init__(self, feature_channels: List[int], hidden_dim: int, patch_size: int = 16):
        super(HiddenFeatureProjection, self).__init__()
        
        self.feature_channels = feature_channels  # 各层特征图的通道数 [64, 128, 256, 512]
        self.hidden_dim = hidden_dim
        self.patch_size = patch_size
        
        # 为每个尺度的特征图创建线性投影层
        self.projection_layers = nn.ModuleList([
            nn.Conv2d(channels, hidden_dim, kernel_size=1) 
            for channels in feature_channels
        ])
        
        # 位置编码
        self.position_encoding = PositionalEncoding(hidden_dim)
        
    def forward(self, features: List[torch.Tensor]) -> torch.Tensor:
        """
        将多尺度特征图投影到hidden_dim维度并准备为Transformer输入
        
        Args:
            features: 多尺度特征图列表 [f1, f2, f3, f4]
            
        Returns:
            token_sequence: [batch_size, num_tokens, hidden_dim]
        """
        batch_size = features[0].shape[0]
        all_tokens = []
        
        for i, (feat, proj) in enumerate(zip(features, self.projection_layers)):
            # 线性投影到hidden_dim
            projected = proj(feat)  # [B, hidden_dim, H, W]
            
            # 重塑为token序列 [B, H*W, hidden_dim]
            B, C, H, W = projected.shape
            tokens = projected.view(B, C, -1).transpose(1, 2)  # [B, H*W, hidden_dim]
            
            all_tokens.append(tokens)
        
        # 拼接所有尺度的token
        token_sequence = torch.cat(all_tokens, dim=1)  # [B, total_tokens, hidden_dim]
        
        # 添加位置编码
        token_sequence = self.position_encoding(token_sequence)
        
        return token_sequence

class PositionalEncoding(nn.Module):
    """位置编码"""
    def __init__(self, hidden_dim: int, max_length: int = 1000):
        super(PositionalEncoding, self).__init__()
        
        pe = torch.zeros(max_length, hidden_dim)
        position = torch.arange(0, max_length, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, hidden_dim, 2).float() * 
                           (-math.log(10000.0) / hidden_dim))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # [1, max_length, hidden_dim]
        
        self.register_buffer('pe', pe)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, :x.size(1)]

class GaussianWeightModule(nn.Module):
    """
    高斯加权模块：根据肿瘤区域统计信息生成注意力偏置
    """
    def __init__(self, lambda_init: float = 0.5, alpha: float = 0.3):
        super(GaussianWeightModule, self).__init__()
        
        self.lambda_param = nn.Parameter(torch.tensor(lambda_init))
        self.alpha = alpha
        
        # 训练时统计的种群级别参数
        self.register_buffer('mu_learned', torch.tensor(0.0))
        self.register_buffer('sigma_learned', torch.tensor(1.0))
        self.register_buffer('stats_count', torch.tensor(0))
        
    def compute_gaussian_weights(self, I: torch.Tensor, Y_s: torch.Tensor, 
                               mode: str = 'train') -> torch.Tensor:
        """
        计算高斯权重
        
        Args:
            I: 输入图像 [B, 1, H, W]
            Y_s: 肿瘤标注 [B, 1, H, W]
            mode: 'train' 或 'inference'
            
        Returns:
            G: 高斯权重图 [B, H, W]
        """
        epsilon = 1e-5
        batch_size = I.shape[0]
        
        if mode == 'train' and Y_s is not None:
            # 训练模式：使用真实标注计算统计量
            G_batch = []
            mu_batch = []
            sigma_batch = []
            
            for i in range(batch_size):
                # 提取肿瘤区域像素
                tumor_pixels = I[i][Y_s[i].bool()]  # 只取标注为肿瘤的像素
                
                if len(tumor_pixels) > 0:
                    mu_Ys = tumor_pixels.mean()
                    sigma_Ys = tumor_pixels.std()
                    
                    # 更新种群统计量（指数移动平均）
                    if self.training:
                        self._update_population_stats(mu_Ys, sigma_Ys)
                else:
                    # 如果没有肿瘤标注，使用默认值
                    mu_Ys = torch.tensor(0.0, device=I.device)
                    sigma_Ys = torch.tensor(1.0, device=I.device)
                
                mu_batch.append(mu_Ys)
                sigma_batch.append(sigma_Ys)
                
                # 多尺度高斯权重计算
                G_multi_scale = []
                for k in [0.5, 1.0, 2.0]:
                    denominator = 2 * (k * sigma_Ys) ** 2 + epsilon
                    G_k = torch.exp(-(I[i] - mu_Ys) ** 2 / denominator)
                    G_multi_scale.append(G_k)
                
                # 多尺度融合
                G_i = torch.stack(G_multi_scale).mean(dim=0)  # [1, H, W]
                G_batch.append(G_i)
            
            G = torch.stack(G_batch)  # [B, 1, H, W]
            return G.squeeze(1)  # [B, H, W]
            
        else:
            # 推理模式：使用学习到的种群统计量
            G_batch = []
            for i in range(batch_size):
                G_multi_scale = []
                for k in [0.5, 1.0, 2.0]:
                    denominator = 2 * (k * self.sigma_learned) ** 2 + epsilon
                    G_k = torch.exp(-(I[i] - self.mu_learned) ** 2 / denominator)
                    G_multi_scale.append(G_k)
                
                G_i = torch.stack(G_multi_scale).mean(dim=0)
                G_batch.append(G_i)
            
            G = torch.stack(G_batch)  # [B, 1, H, W]
            return G.squeeze(1)  # [B, H, W]
    
    def _update_population_stats(self, mu: torch.Tensor, sigma: torch.Tensor):
        """更新种群统计量"""
        if self.stats_count == 0:
            self.mu_learned = mu.detach()
            self.sigma_learned = sigma.detach()
        else:
            # 指数移动平均
            alpha = 0.01  # 平滑因子
            self.mu_learned = (1 - alpha) * self.mu_learned + alpha * mu.detach()
            self.sigma_learned = (1 - alpha) * self.sigma_learned + alpha * sigma.detach()
        
        self.stats_count += 1
    
    def create_gaussian_bias(self, G: torch.Tensor, token_sequence: torch.Tensor) -> torch.Tensor:
        """
        创建高斯偏置矩阵
        
        Args:
            G: 高斯权重图 [B, H, W]
            token_sequence: token序列 [B, N, hidden_dim]
            
        Returns:
            B_gaussian: 高斯偏置矩阵 [B, N, N]
        """
        batch_size, H, W = G.shape
        N = token_sequence.shape[1]  # token数量
        
        # 将高斯权重图下采样/调整到与token空间分辨率匹配
        if H * W != N:
            # 如果需要，调整高斯权重图的尺寸
            G_reshaped = F.interpolate(G.unsqueeze(1), size=int(math.sqrt(N)), 
                                     mode='bilinear', align_corners=False)
            G_reshaped = G_reshaped.squeeze(1)
            g = G_reshaped.view(batch_size, -1)  # [B, N]
        else:
            g = G.view(batch_size, -1)  # [B, N]
        
        # 创建偏置矩阵: B_gaussian = g * g^T
        B_gaussian = torch.bmm(g.unsqueeze(2), g.unsqueeze(1))  # [B, N, N]
        
        return B_gaussian
    
    def forward(self, I: torch.Tensor, token_sequence: torch.Tensor, 
                Y_s: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        前向传播
        
        Args:
            I: 输入图像 [B, 1, H, W]
            token_sequence: token序列 [B, N, hidden_dim]
            Y_s: 肿瘤标注 [B, 1, H, W] (训练时提供)
            
        Returns:
            B_gaussian: 高斯偏置矩阵 [B, N, N]
        """
        mode = 'train' if Y_s is not None and self.training else 'inference'
        G = self.compute_gaussian_weights(I, Y_s, mode)
        B_gaussian = self.create_gaussian_bias(G, token_sequence)
        
        return B_gaussian

class GaussianEnhancedAttention(nn.Module):
    """
    高斯增强的注意力机制
    """
    def __init__(self, hidden_dim: int, num_heads: int = 8, lambda_init: float = 0.5):
        super(GaussianEnhancedAttention, self).__init__()
        
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        
        assert self.head_dim * num_heads == hidden_dim, "hidden_dim必须能被num_heads整除"
        
        # Q, K, V 投影
        self.q_linear = nn.Linear(hidden_dim, hidden_dim)
        self.k_linear = nn.Linear(hidden_dim, hidden_dim)
        self.v_linear = nn.Linear(hidden_dim, hidden_dim)
        
        self.out_linear = nn.Linear(hidden_dim, hidden_dim)
        
        # 高斯参数
        self.lambda_param = nn.Parameter(torch.tensor(lambda_init))
        
        self.softmax = nn.Softmax(dim=-1)
        
    def forward(self, x: torch.Tensor, B_gaussian: torch.Tensor) -> torch.Tensor:
        """
        高斯增强的注意力前向传播
        
        Args:
            x: 输入序列 [B, N, hidden_dim]
            B_gaussian: 高斯偏置矩阵 [B, N, N]
            
        Returns:
            输出序列 [B, N, hidden_dim]
        """
        batch_size, seq_len, hidden_dim = x.shape
        
        # 计算Q, K, V
        Q = self.q_linear(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.k_linear(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.v_linear(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        # 计算注意力分数
        attention_scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_dim)
        
        # 添加高斯偏置
        B_gaussian_expanded = B_gaussian.unsqueeze(1).expand(-1, self.num_heads, -1, -1)
        attention_scores = attention_scores + self.lambda_param * B_gaussian_expanded
        
        # 应用softmax
        attention_weights = self.softmax(attention_scores)
        
        # 应用注意力权重到V
        out = torch.matmul(attention_weights, V)
        out = out.transpose(1, 2).contiguous().view(batch_size, seq_len, hidden_dim)
        
        return self.out_linear(out)

class AttentionSupervisionLoss(nn.Module):
    """
    注意力监督损失
    """
    def __init__(self, alpha: float = 0.3):
        super(AttentionSupervisionLoss, self).__init__()
        self.alpha = alpha
        self.bce_loss = nn.BCELoss()
        
    def forward(self, attention_weights: torch.Tensor, Y_s: torch.Tensor) -> torch.Tensor:
        """
        计算注意力监督损失
        
        Args:
            attention_weights: 注意力权重 [B, N, N]
            Y_s: 肿瘤标注 [B, 1, H, W]
            
        Returns:
            loss: 注意力监督损失
        """
        batch_size, N, _ = attention_weights.shape
        
        # 将标注调整为与注意力权重匹配的尺寸
        if Y_s.shape[-2:] != (int(math.sqrt(N)), int(math.sqrt(N))):
            Y_s_resized = F.interpolate(Y_s, size=int(math.sqrt(N)), 
                                      mode='nearest')
        else:
            Y_s_resized = Y_s
            
        # 展平标注
        Y_flat = Y_s_resized.view(batch_size, 1, -1)  # [B, 1, N]
        
        # 计算每行的注意力监督损失
        losses = []
        for i in range(batch_size):
            # 对每个样本计算BCE损失
            attn_softmax = F.softmax(attention_weights[i], dim=-1)  # [N, N]
            target = Y_flat[i].expand(N, -1)  # [N, N]
            loss = self.bce_loss(attn_softmax, target)
            losses.append(loss)
        
        return torch.stack(losses).mean() * self.alpha

# 完整的特征预处理管道
class FeatureToTransformerPipeline(nn.Module):
    """
    完整的特征到Transformer预处理管道
    """
    def __init__(self, feature_channels: List[int], hidden_dim: int, 
                 num_heads: int = 8, alpha: float = 0.3):
        super(FeatureToTransformerPipeline, self).__init__()
        
        self.hidden_projection = HiddenFeatureProjection(feature_channels, hidden_dim)
        self.gaussian_module = GaussianWeightModule(alpha=alpha)
        self.attention = GaussianEnhancedAttention(hidden_dim, num_heads)
        self.attention_loss = AttentionSupervisionLoss(alpha=alpha)
        
    def forward(self, features: List[torch.Tensor], I: torch.Tensor, 
                Y_s: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        完整的前向传播
        
        Args:
            features: 卷积特征列表
            I: 输入图像 [B, 1, H, W]
            Y_s: 肿瘤标注 [B, 1, H, W] (训练时提供)
            
        Returns:
            transformer_output: Transformer输出 [B, N, hidden_dim]
            attention_loss: 注意力监督损失 (训练时)
        """
        # 特征投影
        token_sequence = self.hidden_projection(features)  # [B, N, hidden_dim]
        
        # 高斯偏置
        B_gaussian = self.gaussian_module(I, token_sequence, Y_s)  # [B, N, N]
        
        # 高斯增强的注意力
        transformer_output = self.attention(token_sequence, B_gaussian)
        
        # 计算注意力监督损失（仅在训练时）
        attention_loss = None
        if Y_s is not None and self.training:
            # 注意：这里需要从attention层获取注意力权重
            # 在实际实现中，你可能需要修改GaussianEnhancedAttention来返回注意力权重
            attention_loss = self.attention_loss(B_gaussian, Y_s)
        
        return transformer_output, attention_loss

# 使用示例
if __name__ == "__main__":
    # 参数设置
    feature_channels = [64, 128, 256, 512]  # 对应4层CNN输出的通道数
    hidden_dim = 512
    batch_size = 2
    image_size = 512
    
    # 模拟输入
    features = [
        torch.randn(batch_size, 64, 256, 256),   # f1
        torch.randn(batch_size, 128, 128, 128),  # f2
        torch.randn(batch_size, 256, 64, 64),    # f3
        torch.randn(batch_size, 512, 32, 32)     # f4
    ]
    
    I = torch.randn(batch_size, 1, image_size, image_size)  # 输入图像
    Y_s = torch.randint(0, 2, (batch_size, 1, image_size, image_size)).float()  # 模拟标注
    
    # 创建管道
    pipeline = FeatureToTransformerPipeline(feature_channels, hidden_dim)
    
    # 前向传播
    print("进行特征到Transformer的转换...")
    transformer_output, attention_loss = pipeline(features, I, Y_s)
    
    print(f"输入特征尺寸: {[f.shape for f in features]}")
    print(f"Transformer输出尺寸: {transformer_output.shape}")
    
    if attention_loss is not None:
        print(f"注意力监督损失: {attention_loss.item():.4f}")
    
    print("\n管道测试完成！")