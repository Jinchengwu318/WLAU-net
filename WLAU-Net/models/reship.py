import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import List, Tuple, Optional
import numpy as np


class TransformerEncoder(nn.Module):
    """
    12-layer Transformer encoder
    """

    def __init__(self, hidden_dim: int, num_layers: int = 12, num_heads: int = 8,
                 mlp_ratio: float = 4.0, dropout: float = 0.1):
        super(TransformerEncoder, self).__init__()

        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        # Transformer layers
        self.layers = nn.ModuleList([
            TransformerLayer(hidden_dim, num_heads, mlp_ratio, dropout)
            for _ in range(num_layers)
        ])

        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward propagation

        Args:
            x: Input sequence [B, N, hidden_dim]

        Returns:
            transformed_x: Transformer output [B, N, hidden_dim]
        """
        for layer in self.layers:
            x = layer(x)

        return self.norm(x)


class TransformerLayer(nn.Module):
    """Single Transformer layer"""

    def __init__(self, hidden_dim: int, num_heads: int, mlp_ratio: float, dropout: float):
        super(TransformerLayer, self).__init__()

        self.norm1 = nn.LayerNorm(hidden_dim)
        self.attention = nn.MultiheadAttention(hidden_dim, num_heads, dropout=dropout, batch_first=True)

        self.norm2 = nn.LayerNorm(hidden_dim)
        self.mlp = MLP(hidden_dim, int(hidden_dim * mlp_ratio), hidden_dim, dropout)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Self-attention
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
    """Multi-layer perceptron"""

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
    Hidden feature transform: Convert Transformer output to decoder input
    """

    def __init__(self, hidden_dim: int, decoder_dims: List[int]):
        """
        Args:
            hidden_dim: Transformer hidden dimension
            decoder_dims: Expected channel numbers for each decoder layer [f4_dim, f3_dim, f2_dim, f1_dim]
        """
        super(HiddenFeatureTransform, self).__init__()

        self.hidden_dim = hidden_dim
        self.decoder_dims = decoder_dims

        # Hidden feature transformation
        self.hidden_transform = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim)
        )

        # Feature splitting: Split token sequence back to multi-scale feature maps
        self.feature_splits = [32 * 32, 64 * 64, 128 * 128, 256 * 256]  # Number of tokens at each scale

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        """
        Convert Transformer output to multi-scale feature maps

        Args:
            x: Transformer output [B, N, hidden_dim]

        Returns:
            features: Multi-scale feature map list [f4, f3, f2, f1]
        """
        batch_size = x.shape[0]

        # Hidden feature transformation
        x_transformed = self.hidden_transform(x)  # [B, N, hidden_dim]

        # Split token sequence into scale features
        features = []
        start_idx = 0

        for i, num_tokens in enumerate(self.feature_splits):
            # Extract current scale tokens
            end_idx = start_idx + num_tokens
            scale_tokens = x_transformed[:, start_idx:end_idx, :]  # [B, num_tokens, hidden_dim]

            # Reshape to feature map
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

        # Reverse feature list order to match decoder expected order [f4, f3, f2, f1]
        return features[::-1]


class FeatureReshapeForDecoder(nn.Module):
    """
    Feature reshaping: Adjust feature map channel numbers to match U-net decoder
    """

    def __init__(self, hidden_dim: int, decoder_channels: List[int]):
        """
        Args:
            hidden_dim: Input feature map channel number
            decoder_channels: Expected channel numbers for each decoder layer [f4_out, f3_out, f2_out, f1_out]
        """
        super(FeatureReshapeForDecoder, self).__init__()

        self.hidden_dim = hidden_dim
        self.decoder_channels = decoder_channels

        # Channel adjustment convolutions
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
        Adjust feature map channel numbers to match decoder

        Args:
            features: Input feature map list [f4, f3, f2, f1]

        Returns:
            adjusted_features: Adjusted feature map list [f4_adj, f3_adj, f2_adj, f1_adj]
        """
        adjusted_features = []

        for i, (feature, adjust_conv) in enumerate(zip(features, self.channel_adjustments)):
            adjusted_feature = adjust_conv(feature)
            adjusted_features.append(adjusted_feature)

        return adjusted_features


class UNetDecoderFeaturePreparer(nn.Module):
    """
    U-net decoder feature preparer: Complete Transformer to decoder feature conversion
    """

    def __init__(self, hidden_dim: int, num_transformer_layers: int = 12,
                 num_heads: int = 8, decoder_channels: List[int] = None):
        super(UNetDecoderFeaturePreparer, self).__init__()

        # Default decoder channel numbers (symmetric with encoder)
        if decoder_channels is None:
            decoder_channels = [512, 256, 128, 64]  # [f4, f3, f2, f1]

        self.hidden_dim = hidden_dim
        self.decoder_channels = decoder_channels

        # 12-layer Transformer encoder
        self.transformer = TransformerEncoder(
            hidden_dim=hidden_dim,
            num_layers=num_transformer_layers,
            num_heads=num_heads
        )

        # Hidden feature transformation
        self.hidden_transform = HiddenFeatureTransform(hidden_dim, decoder_channels)

        # Feature reshaping
        self.feature_reshape = FeatureReshapeForDecoder(hidden_dim, decoder_channels)

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        """
        Complete Transformer to decoder feature conversion

        Args:
            x: Transformer input sequence [B, N, hidden_dim]

        Returns:
            decoder_features: Decoder-processable feature map list [f4, f3, f2, f1]
        """
        # Pass through 12-layer Transformer
        transformer_out = self.transformer(x)  # [B, N, hidden_dim]

        # Hidden feature transformation and sequence splitting
        features = self.hidden_transform(transformer_out)  # [f4, f3, f2, f1]

        # Feature reshaping to match decoder channel numbers
        decoder_features = self.feature_reshape(features)

        return decoder_features


class SkipConnectionFusion(nn.Module):
    """
    Skip connection fusion: Fuse encoder features with decoder-prepared features
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
        Fuse encoder and decoder features

        Args:
            encoder_features: Encoder features [f1, f2, f3, f4] (note: order is reversed from decoder)
            decoder_features: Decoder-prepared features [f4, f3, f2, f1]

        Returns:
            fused_features: Fused features [f4_fused, f3_fused, f2_fused, f1_fused]
        """
        fused_features = []

        # Reverse encoder feature order to match decoder
        encoder_features_rev = encoder_features[::-1]  # [f4, f3, f2, f1]

        for i, (enc_feat, dec_feat, fusion_conv) in enumerate(
                zip(encoder_features_rev, decoder_features, self.fusion_convs)):
            # Concatenate features
            combined = torch.cat([enc_feat, dec_feat], dim=1)

            # Fusion convolution
            fused = fusion_conv(combined)
            fused_features.append(fused)

        return fused_features


# Complete Transformer to U-net decoder pipeline
class TransformerToUNetPipeline(nn.Module):
    """
    Complete Transformer to U-net decoder pipeline
    """

    def __init__(self, hidden_dim: int, encoder_channels: List[int],
                 num_transformer_layers: int = 12, num_heads: int = 8):
        super(TransformerToUNetPipeline, self).__init__()

        self.hidden_dim = hidden_dim
        self.encoder_channels = encoder_channels

        # Decoder channel numbers (typically symmetric with encoder)
        decoder_channels = [ch * 2 for ch in encoder_channels[::-1]]  # [512, 256, 128, 64]

        # Transformer to decoder feature preparation
        self.feature_preparer = UNetDecoderFeaturePreparer(
            hidden_dim=hidden_dim,
            num_transformer_layers=num_transformer_layers,
            num_heads=num_heads,
            decoder_channels=decoder_channels
        )

        # Skip connection fusion
        self.skip_fusion = SkipConnectionFusion(encoder_channels, decoder_channels)

    def forward(self, transformer_input: torch.Tensor,
                encoder_features: List[torch.Tensor]) -> List[torch.Tensor]:
        """
        Complete pipeline forward propagation

        Args:
            transformer_input: Transformer input sequence [B, N, hidden_dim]
            encoder_features: Encoder feature list [f1, f2, f3, f4]

        Returns:
            decoder_ready_features: Decoder-ready feature list [f4_ready, f3_ready, f2_ready, f1_ready]
        """
        # Prepare decoder features
        decoder_features = self.feature_preparer(transformer_input)

        # Fuse skip connections
        fused_features = self.skip_fusion(encoder_features, decoder_features)

        return fused_features


# Usage example
if __name__ == "__main__":
    # Parameter settings
    hidden_dim = 512
    encoder_channels = [64, 128, 256, 512]  # Encoder layer channel numbers
    batch_size = 2

    # Simulate input
    transformer_input = torch.randn(batch_size, 86080, hidden_dim)  # 86080 = 256*256 + 128*128 + 64*64 + 32*32

    # Simulate encoder features
    encoder_features = [
        torch.randn(batch_size, 64, 256, 256),  # f1
        torch.randn(batch_size, 128, 128, 128),  # f2
        torch.randn(batch_size, 256, 64, 64),  # f3
        torch.randn(batch_size, 512, 32, 32)  # f4
    ]

    # Create pipeline
    pipeline = TransformerToUNetPipeline(
        hidden_dim=hidden_dim,
        encoder_channels=encoder_channels,
        num_transformer_layers=12,
        num_heads=8
    )

    print("Performing Transformer to U-net decoder feature conversion...")

    # Forward propagation
    decoder_ready_features = pipeline(transformer_input, encoder_features)

    print(f"Input Transformer sequence dimension: {transformer_input.shape}")
    print(f"Encoder feature dimensions: {[f.shape for f in encoder_features]}")
    print(f"Decoder-ready feature dimensions: {[f.shape for f in decoder_ready_features]}")

    print("\nFeature conversion completed! Now ready to input to U-net decoder.")
    print("Feature order: [f4_ready, f3_ready, f2_ready, f1_ready]")
    print("These features can be directly used for U-net decoder upsampling and skip connections.")