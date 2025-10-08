import torch
import torch.nn as nn
from typing import List, Tuple, Optional

class LiverTumorSegmentationPipeline(nn.Module):
    """
    完整的肝血管瘤分割管道：集成所有组件
    """
    def __init__(self, hidden_dim: int = 512, encoder_channels: List[int] = None,
                 num_transformer_layers: int = 12, lambda1: float = 1.2, 
                 lambda2: float = 1.1, output_channels: int = 1):
        super(LiverTumorSegmentationPipeline, self).__init__()
        
        if encoder_channels is None:
            encoder_channels = [64, 128, 256, 512]
            
        self.hidden_dim = hidden_dim
        self.encoder_channels = encoder_channels
        
        # 导入各个模块
        from models.DWT import WaveletEnhancement
        from models.encoder import CTEncoder, EncoderTransfer
        from models.GW import FeatureToTransformerPipeline
        from models.reship import TransformerToUNetPipeline
        from models.decoder import CompleteUNetDecoder
        
        # 1. 小波增强
        self.wavelet_enhancer = WaveletEnhancement(lambda1=lambda1, lambda2=lambda2)
        
        # 2. CNN编码器
        self.encoder = CTEncoder(in_channels=2, base_channels=64)
        
        # 3. Transformer特征准备
        self.transformer_preparer = FeatureToTransformerPipeline(
            feature_channels=encoder_channels,
            hidden_dim=hidden_dim,
            num_heads=8,
            alpha=0.3
        )
        
        # 4. Transformer到解码器转换
        self.transformer_to_decoder = TransformerToUNetPipeline(
            hidden_dim=hidden_dim,
            encoder_channels=encoder_channels,
            num_transformer_layers=num_transformer_layers,
            num_heads=8
        )
        
        # 5. U-net解码器
        decoder_channels = [ch * 2 for ch in encoder_channels[::-1]]
        self.decoder = CompleteUNetDecoder(
            decoder_channels=decoder_channels,
            output_channels=output_channels
        )
        
        # 编码器迁移工具
        self.encoder_transfer = EncoderTransfer(base_channels=64)
        
    def forward(self, original_image: torch.Tensor, enhanced_image: torch.Tensor = None,
                Y_s: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, dict]:
        """
        完整的前向传播
        
        Args:
            original_image: 原始图像 [B, 1, H, W]
            enhanced_image: 增强图像 [B, 1, H, W] (如果为None则内部生成)
            Y_s: 肿瘤标注 [B, 1, H, W] (训练时提供)
            
        Returns:
            segmentation: 分割输出 [B, 1, H, W]
            losses: 损失字典
        """
        losses = {}
        
        # 1. 小波增强（如果未提供增强图像）
        if enhanced_image is None:
            enhanced_image = self.wavelet_enhancer.enhance_image(original_image)
        
        # 2. 准备混合输入
        mixed_input = torch.cat([original_image, enhanced_image], dim=1)  # [B, 2, H, W]
        
        # 3. CNN编码器特征提取
        encoder_features = self.encoder(mixed_input)  # [f1, f2, f3, f4]
        
        # 4. Transformer特征准备
        transformer_output, attention_loss = self.transformer_preparer(
            encoder_features, original_image, Y_s
        )
        
        if attention_loss is not None:
            losses['attention'] = attention_loss
        
        # 5. Transformer到解码器转换
        decoder_ready_features = self.transformer_to_decoder(
            transformer_output, encoder_features
        )
        
        # 6. 解码器生成分割结果
        segmentation = self.decoder(decoder_ready_features, encoder_features)
        
        return segmentation, losses
    
    def setup_transfer_learning(self, portal_checkpoint_path: str):
        """
        设置迁移学习：从门脉期模型迁移参数
        """
        # 加载预训练的门脉期编码器权重
        self.encoder_transfer.load_portal_encoder_weights(portal_checkpoint_path)
        
        # 执行参数迁移和冻结
        self.encoder_transfer.transfer_weights(freeze_encoder=True)
        
        # 设置当前编码器为迁移后的编码器
        self.encoder = self.encoder_transfer.get_plain_encoder()
        
        print("迁移学习设置完成！编码器参数已冻结。")