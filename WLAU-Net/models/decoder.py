import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple


class UNetDecoder(nn.Module):
    """
    U-net Decoder: Upsampling, skip connections and segmentation output
    """

    def __init__(self, decoder_channels: List[int] = [512, 256, 128, 64],
                 output_channels: int = 1, use_skip_connections: bool = True):
        super(UNetDecoder, self).__init__()

        self.decoder_channels = decoder_channels
        self.output_channels = output_channels
        self.use_skip_connections = use_skip_connections

        # Upsampling layers
        self.upsample_layers = nn.ModuleList()
        self.conv_blocks = nn.ModuleList()

        # Build decoder layers
        for i in range(len(decoder_channels) - 1):
            in_ch = decoder_channels[i]
            out_ch = decoder_channels[i + 1]

            # Upsampling + convolution block
            self.upsample_layers.append(
                UpsampleBlock(in_ch, out_ch)
            )

            self.conv_blocks.append(
                ConvBlock(out_ch * 2 if use_skip_connections else out_ch, out_ch)
            )

        # Final upsampling to original size
        self.final_upsample = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(decoder_channels[-1], decoder_channels[-1] // 2, 3, padding=1),
            nn.BatchNorm2d(decoder_channels[-1] // 2),
            nn.ReLU(inplace=True)
        )

        # Final output layer
        self.output_conv = nn.Conv2d(decoder_channels[-1] // 2, output_channels, 1)

    def forward(self, decoder_features: List[torch.Tensor],
                encoder_features: List[torch.Tensor]) -> torch.Tensor:
        """
        Decoder forward propagation

        Args:
            decoder_features: Decoder input features [f4, f3, f2, f1] (deep to shallow)
            encoder_features: Encoder features [f1, f2, f3, f4] (shallow to deep)

        Returns:
            output: Segmentation output [B, output_channels, H, W]
        """
        # Reverse encoder feature order to match decoder [f4, f3, f2, f1]
        encoder_features_rev = encoder_features[::-1]

        x = decoder_features[0]  # Start from deepest layer f4

        # Upsample layer by layer
        for i, (upsample, conv_block) in enumerate(zip(self.upsample_layers, self.conv_blocks)):
            # Upsampling
            x = upsample(x)  # Size doubles, channels halve

            # Skip connection (if enabled)
            if self.use_skip_connections and i < len(encoder_features_rev) - 1:
                skip_feature = encoder_features_rev[i + 1]  # Corresponding encoder feature

                # Adjust channel number (if needed)
                if skip_feature.shape[1] != x.shape[1]:
                    skip_feature = self._adjust_channels(skip_feature, x.shape[1])

                # Concatenate features
                x = torch.cat([x, skip_feature], dim=1)

            # Convolution block
            x = conv_block(x)

        # Final upsampling to original size
        x = self.final_upsample(x)

        # Output segmentation map
        output = self.output_conv(x)

        return output

    def _adjust_channels(self, feature: torch.Tensor, target_channels: int) -> torch.Tensor:
        """Adjust feature map channel number"""
        if feature.shape[1] != target_channels:
            adjust_conv = nn.Conv2d(feature.shape[1], target_channels, 1).to(feature.device)
            return adjust_conv(feature)
        return feature


class UpsampleBlock(nn.Module):
    """Upsampling block"""

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
    """Convolution block"""

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
    """Segmentation head: Generate final segmentation mask"""

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
    Complete U-net Decoder: Integrate all components
    """

    def __init__(self, decoder_channels: List[int] = [512, 256, 128, 64],
                 output_channels: int = 1, use_skip_connections: bool = True):
        super(CompleteUNetDecoder, self).__init__()

        self.decoder = UNetDecoder(decoder_channels, output_channels, use_skip_connections)
        self.segmentation_head = SegmentationHead(output_channels, output_channels)

    def forward(self, decoder_features: List[torch.Tensor],
                encoder_features: List[torch.Tensor]) -> torch.Tensor:
        """
        Complete forward propagation

        Args:
            decoder_features: Decoder input features
            encoder_features: Encoder features (for skip connections)

        Returns:
            segmentation: Segmentation probability map [B, output_channels, H, W]
        """
        # Pass through decoder
        decoder_output = self.decoder(decoder_features, encoder_features)

        # Generate segmentation mask
        segmentation = self.segmentation_head(decoder_output)

        return segmentation


class LiverTumorSegmentationModel(nn.Module):
    """
    Complete liver hemangioma segmentation model: Integrate encoder, Transformer and decoder
    """

    def __init__(self, encoder_channels: List[int] = [64, 128, 256, 512],
                 hidden_dim: int = 512, output_channels: int = 1):
        super(LiverTumorSegmentationModel, self).__init__()

        self.encoder_channels = encoder_channels
        self.hidden_dim = hidden_dim

        # Encoder (using previously implemented frozen encoder)
        self.encoder = None  # Will be set externally

        # Transformer to decoder feature preparation (using previously implemented module)
        self.transformer_to_decoder = None  # Will be set externally

        # Decoder
        decoder_channels = [ch * 2 for ch in encoder_channels[::-1]]  # [1024, 512, 256, 128]
        self.decoder = CompleteUNetDecoder(
            decoder_channels=decoder_channels,
            output_channels=output_channels,
            use_skip_connections=True
        )

    def set_encoder(self, encoder):
        """Set encoder"""
        self.encoder = encoder

    def set_transformer_to_decoder(self, transformer_to_decoder):
        """Set Transformer to decoder conversion module"""
        self.transformer_to_decoder = transformer_to_decoder

    def forward(self, mixed_input: torch.Tensor, original_image: torch.Tensor) -> torch.Tensor:
        """
        Complete model forward propagation

        Args:
            mixed_input: Mixed input [B, 2, H, W] (original + enhanced)
            original_image: Original image [B, 1, H, W] (for Gaussian weighting)

        Returns:
            segmentation: Segmentation output [B, output_channels, H, W]
        """
        # 1. Encoder feature extraction
        encoder_features = self.encoder(mixed_input)  # [f1, f2, f3, f4]

        # 2. Transformer processing (simplified representation here, actual full Transformer pipeline needed)
        # transformer_output = self.transformer_pipeline(encoder_features, original_image)

        # 3. Transformer to decoder feature conversion
        # decoder_features = self.transformer_to_decoder(transformer_output, encoder_features)

        # For demonstration, we directly use encoder features as decoder input
        # In actual use, this should be features processed by Transformer
        decoder_features = [feat for feat in encoder_features[::-1]]  # Reverse order [f4, f3, f2, f1]

        # 4. Decoder generates segmentation result
        segmentation = self.decoder(decoder_features, encoder_features)

        return segmentation


# Training-related helper functions
class DiceLoss(nn.Module):
    """Dice loss"""

    def __init__(self, smooth: float = 1e-5):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        predictions = torch.sigmoid(predictions)

        # Flatten
        predictions = predictions.view(-1)
        targets = targets.view(-1)

        intersection = (predictions * targets).sum()
        dice = (2. * intersection + self.smooth) / (predictions.sum() + targets.sum() + self.smooth)

        return 1 - dice


class CombinedLoss(nn.Module):
    """Combined loss: Dice + BCE"""

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


# Usage example
if __name__ == "__main__":
    # Parameter settings
    batch_size = 2
    image_size = 512
    encoder_channels = [64, 128, 256, 512]
    output_channels = 1

    # Simulate input
    mixed_input = torch.randn(batch_size, 2, image_size, image_size)
    original_image = torch.randn(batch_size, 1, image_size, image_size)

    # Simulate encoder features
    encoder_features = [
        torch.randn(batch_size, 64, 256, 256),  # f1
        torch.randn(batch_size, 128, 128, 128),  # f2
        torch.randn(batch_size, 256, 64, 64),  # f3
        torch.randn(batch_size, 512, 32, 32)  # f4
    ]

    # Simulate decoder features (after Transformer processing)
    decoder_features = [
        torch.randn(batch_size, 1024, 32, 32),  # f4_ready
        torch.randn(batch_size, 512, 64, 64),  # f3_ready
        torch.randn(batch_size, 256, 128, 128),  # f2_ready
        torch.randn(batch_size, 128, 256, 256)  # f1_ready
    ]

    print("Testing U-net decoder...")

    # Create decoder
    decoder = CompleteUNetDecoder(
        decoder_channels=[1024, 512, 256, 128],
        output_channels=output_channels
    )

    # Forward propagation
    segmentation_output = decoder(decoder_features, encoder_features)

    print(f"Encoder feature dimensions: {[f.shape for f in encoder_features]}")
    print(f"Decoder input feature dimensions: {[f.shape for f in decoder_features]}")
    print(f"Segmentation output dimension: {segmentation_output.shape}")
    print(f"Segmentation output range: [{segmentation_output.min():.3f}, {segmentation_output.max():.3f}]")

    # Test complete model
    print("\nTesting complete segmentation model...")
    model = LiverTumorSegmentationModel(
        encoder_channels=encoder_channels,
        hidden_dim=512,
        output_channels=output_channels
    )


    # Simulate setting encoder (need real encoder in actual use)
    class MockEncoder(nn.Module):
        def forward(self, x):
            return [
                torch.randn(x.shape[0], 64, 256, 256),
                torch.randn(x.shape[0], 128, 128, 128),
                torch.randn(x.shape[0], 256, 64, 64),
                torch.randn(x.shape[0], 512, 32, 32)
            ]


    model.set_encoder(MockEncoder())

    # Complete model forward propagation
    final_output = model(mixed_input, original_image)

    print(f"Final segmentation output dimension: {final_output.shape}")
    print(f"Final output range: [{final_output.min():.3f}, {final_output.max():.3f}]")

    print("\nU-net decoder testing completed!")
    print("Now ready to generate liver hemangioma segmentation results.")
