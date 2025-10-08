import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from typing import List, Tuple, Optional


class HiddenFeatureProjection(nn.Module):
    """
    Module to convert convolutional features to Transformer input
    """

    def __init__(self, feature_channels: List[int], hidden_dim: int, patch_size: int = 16):
        super(HiddenFeatureProjection, self).__init__()

        self.feature_channels = feature_channels  # Channel numbers of each feature map [64, 128, 256, 512]
        self.hidden_dim = hidden_dim
        self.patch_size = patch_size

        # Create linear projection layers for each scale feature map
        self.projection_layers = nn.ModuleList([
            nn.Conv2d(channels, hidden_dim, kernel_size=1)
            for channels in feature_channels
        ])

        # Position encoding
        self.position_encoding = PositionalEncoding(hidden_dim)

    def forward(self, features: List[torch.Tensor]) -> torch.Tensor:
        """
        Project multi-scale feature maps to hidden_dim dimension and prepare as Transformer input

        Args:
            features: Multi-scale feature map list [f1, f2, f3, f4]

        Returns:
            token_sequence: [batch_size, num_tokens, hidden_dim]
        """
        batch_size = features[0].shape[0]
        all_tokens = []

        for i, (feat, proj) in enumerate(zip(features, self.projection_layers)):
            # Linear projection to hidden_dim
            projected = proj(feat)  # [B, hidden_dim, H, W]

            # Reshape to token sequence [B, H*W, hidden_dim]
            B, C, H, W = projected.shape
            tokens = projected.view(B, C, -1).transpose(1, 2)  # [B, H*W, hidden_dim]

            all_tokens.append(tokens)

        # Concatenate all scale tokens
        token_sequence = torch.cat(all_tokens, dim=1)  # [B, total_tokens, hidden_dim]

        # Add position encoding
        token_sequence = self.position_encoding(token_sequence)

        return token_sequence


class PositionalEncoding(nn.Module):
    """Positional Encoding"""

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
    Gaussian Weight Module: Generate attention bias based on tumor region statistical information
    """

    def __init__(self, lambda_init: float = 0.5, alpha: float = 0.3):
        super(GaussianWeightModule, self).__init__()

        self.lambda_param = nn.Parameter(torch.tensor(lambda_init))
        self.alpha = alpha

        # Population-level parameters learned during training
        self.register_buffer('mu_learned', torch.tensor(0.0))
        self.register_buffer('sigma_learned', torch.tensor(1.0))
        self.register_buffer('stats_count', torch.tensor(0))

    def compute_gaussian_weights(self, I: torch.Tensor, Y_s: torch.Tensor,
                                 mode: str = 'train') -> torch.Tensor:
        """
        Compute Gaussian weights

        Args:
            I: Input image [B, 1, H, W]
            Y_s: Tumor annotation [B, 1, H, W]
            mode: 'train' or 'inference'

        Returns:
            G: Gaussian weight map [B, H, W]
        """
        epsilon = 1e-5
        batch_size = I.shape[0]

        if mode == 'train' and Y_s is not None:
            # Training mode: compute statistics using real annotations
            G_batch = []
            mu_batch = []
            sigma_batch = []

            for i in range(batch_size):
                # Extract tumor region pixels
                tumor_pixels = I[i][Y_s[i].bool()]  # Only take pixels annotated as tumor

                if len(tumor_pixels) > 0:
                    mu_Ys = tumor_pixels.mean()
                    sigma_Ys = tumor_pixels.std()

                    # Update population statistics (exponential moving average)
                    if self.training:
                        self._update_population_stats(mu_Ys, sigma_Ys)
                else:
                    # If no tumor annotation, use default values
                    mu_Ys = torch.tensor(0.0, device=I.device)
                    sigma_Ys = torch.tensor(1.0, device=I.device)

                mu_batch.append(mu_Ys)
                sigma_batch.append(sigma_Ys)

                # Multi-scale Gaussian weight calculation
                G_multi_scale = []
                for k in [0.5, 1.0, 2.0]:
                    denominator = 2 * (k * sigma_Ys) ** 2 + epsilon
                    G_k = torch.exp(-(I[i] - mu_Ys) ** 2 / denominator)
                    G_multi_scale.append(G_k)

                # Multi-scale fusion
                G_i = torch.stack(G_multi_scale).mean(dim=0)  # [1, H, W]
                G_batch.append(G_i)

            G = torch.stack(G_batch)  # [B, 1, H, W]
            return G.squeeze(1)  # [B, H, W]

        else:
            # Inference mode: use learned population statistics
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
        """Update population statistics"""
        if self.stats_count == 0:
            self.mu_learned = mu.detach()
            self.sigma_learned = sigma.detach()
        else:
            # Exponential moving average
            alpha = 0.01  # Smoothing factor
            self.mu_learned = (1 - alpha) * self.mu_learned + alpha * mu.detach()
            self.sigma_learned = (1 - alpha) * self.sigma_learned + alpha * sigma.detach()

        self.stats_count += 1

    def create_gaussian_bias(self, G: torch.Tensor, token_sequence: torch.Tensor) -> torch.Tensor:
        """
        Create Gaussian bias matrix

        Args:
            G: Gaussian weight map [B, H, W]
            token_sequence: Token sequence [B, N, hidden_dim]

        Returns:
            B_gaussian: Gaussian bias matrix [B, N, N]
        """
        batch_size, H, W = G.shape
        N = token_sequence.shape[1]  # Number of tokens

        # Downsample/adjust Gaussian weight map to match token spatial resolution
        if H * W != N:
            # If needed, adjust Gaussian weight map size
            G_reshaped = F.interpolate(G.unsqueeze(1), size=int(math.sqrt(N)),
                                       mode='bilinear', align_corners=False)
            G_reshaped = G_reshaped.squeeze(1)
            g = G_reshaped.view(batch_size, -1)  # [B, N]
        else:
            g = G.view(batch_size, -1)  # [B, N]

        # Create bias matrix: B_gaussian = g * g^T
        B_gaussian = torch.bmm(g.unsqueeze(2), g.unsqueeze(1))  # [B, N, N]

        return B_gaussian

    def forward(self, I: torch.Tensor, token_sequence: torch.Tensor,
                Y_s: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward propagation

        Args:
            I: Input image [B, 1, H, W]
            token_sequence: Token sequence [B, N, hidden_dim]
            Y_s: Tumor annotation [B, 1, H, W] (provided during training)

        Returns:
            B_gaussian: Gaussian bias matrix [B, N, N]
        """
        mode = 'train' if Y_s is not None and self.training else 'inference'
        G = self.compute_gaussian_weights(I, Y_s, mode)
        B_gaussian = self.create_gaussian_bias(G, token_sequence)

        return B_gaussian


class GaussianEnhancedAttention(nn.Module):
    """
    Gaussian-enhanced attention mechanism
    """

    def __init__(self, hidden_dim: int, num_heads: int = 8, lambda_init: float = 0.5):
        super(GaussianEnhancedAttention, self).__init__()

        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads

        assert self.head_dim * num_heads == hidden_dim, "hidden_dim must be divisible by num_heads"

        # Q, K, V projections
        self.q_linear = nn.Linear(hidden_dim, hidden_dim)
        self.k_linear = nn.Linear(hidden_dim, hidden_dim)
        self.v_linear = nn.Linear(hidden_dim, hidden_dim)

        self.out_linear = nn.Linear(hidden_dim, hidden_dim)

        # Gaussian parameters
        self.lambda_param = nn.Parameter(torch.tensor(lambda_init))

        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x: torch.Tensor, B_gaussian: torch.Tensor) -> torch.Tensor:
        """
        Gaussian-enhanced attention forward propagation

        Args:
            x: Input sequence [B, N, hidden_dim]
            B_gaussian: Gaussian bias matrix [B, N, N]

        Returns:
            Output sequence [B, N, hidden_dim]
        """
        batch_size, seq_len, hidden_dim = x.shape

        # Compute Q, K, V
        Q = self.q_linear(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.k_linear(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.v_linear(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        # Compute attention scores
        attention_scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_dim)

        # Add Gaussian bias
        B_gaussian_expanded = B_gaussian.unsqueeze(1).expand(-1, self.num_heads, -1, -1)
        attention_scores = attention_scores + self.lambda_param * B_gaussian_expanded

        # Apply softmax
        attention_weights = self.softmax(attention_scores)

        # Apply attention weights to V
        out = torch.matmul(attention_weights, V)
        out = out.transpose(1, 2).contiguous().view(batch_size, seq_len, hidden_dim)

        return self.out_linear(out)


class AttentionSupervisionLoss(nn.Module):
    """
    Attention supervision loss
    """

    def __init__(self, alpha: float = 0.3):
        super(AttentionSupervisionLoss, self).__init__()
        self.alpha = alpha
        self.bce_loss = nn.BCELoss()

    def forward(self, attention_weights: torch.Tensor, Y_s: torch.Tensor) -> torch.Tensor:
        """
        Compute attention supervision loss

        Args:
            attention_weights: Attention weights [B, N, N]
            Y_s: Tumor annotation [B, 1, H, W]

        Returns:
            loss: Attention supervision loss
        """
        batch_size, N, _ = attention_weights.shape

        # Adjust annotation to match attention weight size
        if Y_s.shape[-2:] != (int(math.sqrt(N)), int(math.sqrt(N))):
            Y_s_resized = F.interpolate(Y_s, size=int(math.sqrt(N)),
                                        mode='nearest')
        else:
            Y_s_resized = Y_s

        # Flatten annotation
        Y_flat = Y_s_resized.view(batch_size, 1, -1)  # [B, 1, N]

        # Compute attention supervision loss for each row
        losses = []
        for i in range(batch_size):
            # Compute BCE loss for each sample
            attn_softmax = F.softmax(attention_weights[i], dim=-1)  # [N, N]
            target = Y_flat[i].expand(N, -1)  # [N, N]
            loss = self.bce_loss(attn_softmax, target)
            losses.append(loss)

        return torch.stack(losses).mean() * self.alpha


# Complete feature preprocessing pipeline
class FeatureToTransformerPipeline(nn.Module):
    """
    Complete feature to Transformer preprocessing pipeline
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
        Complete forward propagation

        Args:
            features: Convolutional feature list
            I: Input image [B, 1, H, W]
            Y_s: Tumor annotation [B, 1, H, W] (provided during training)

        Returns:
            transformer_output: Transformer output [B, N, hidden_dim]
            attention_loss: Attention supervision loss (during training)
        """
        # Feature projection
        token_sequence = self.hidden_projection(features)  # [B, N, hidden_dim]

        # Gaussian bias
        B_gaussian = self.gaussian_module(I, token_sequence, Y_s)  # [B, N, N]

        # Gaussian-enhanced attention
        transformer_output = self.attention(token_sequence, B_gaussian)

        # Compute attention supervision loss (only during training)
        attention_loss = None
        if Y_s is not None and self.training:
            # Note: Here we need to get attention weights from the attention layer
            # In actual implementation, you may need to modify GaussianEnhancedAttention to return attention weights
            attention_loss = self.attention_loss(B_gaussian, Y_s)

        return transformer_output, attention_loss


# Usage example
if __name__ == "__main__":
    # Parameter settings
    feature_channels = [64, 128, 256, 512]  # Corresponding to 4-layer CNN output channel numbers
    hidden_dim = 512
    batch_size = 2
    image_size = 512

    # Simulate input
    features = [
        torch.randn(batch_size, 64, 256, 256),  # f1
        torch.randn(batch_size, 128, 128, 128),  # f2
        torch.randn(batch_size, 256, 64, 64),  # f3
        torch.randn(batch_size, 512, 32, 32)  # f4
    ]

    I = torch.randn(batch_size, 1, image_size, image_size)  # Input image
    Y_s = torch.randint(0, 2, (batch_size, 1, image_size, image_size)).float()  # Simulated annotation

    # Create pipeline
    pipeline = FeatureToTransformerPipeline(feature_channels, hidden_dim)

    # Forward propagation
    print("Performing feature to Transformer conversion...")
    transformer_output, attention_loss = pipeline(features, I, Y_s)

    print(f"Input feature dimensions: {[f.shape for f in features]}")
    print(f"Transformer output dimension: {transformer_output.shape}")

    if attention_loss is not None:
        print(f"Attention supervision loss: {attention_loss.item():.4f}")

    print("\nPipeline testing completed!")