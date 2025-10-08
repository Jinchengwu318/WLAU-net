import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class CTEncoder(nn.Module):
    """
    Specialized CT encoder: 4 downsampling steps to extract multi-scale features
    Input: 2 channels [original image, enhanced image]
    Output: 4 feature maps at different scales
    """

    def __init__(self, in_channels=2, base_channels=64):
        super(CTEncoder, self).__init__()

        self.in_channels = in_channels
        self.base_channels = base_channels

        # First downsampling: 512x512 -> 256x256
        self.down1 = nn.Sequential(
            nn.Conv2d(in_channels, base_channels, 3, padding=1),
            nn.BatchNorm2d(base_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_channels, base_channels, 3, padding=1),
            nn.BatchNorm2d(base_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2)  # 2x downsampling
        )

        # Second downsampling: 256x256 -> 128x128
        self.down2 = nn.Sequential(
            nn.Conv2d(base_channels, base_channels * 2, 3, padding=1),
            nn.BatchNorm2d(base_channels * 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_channels * 2, base_channels * 2, 3, padding=1),
            nn.BatchNorm2d(base_channels * 2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2)  # 2x downsampling
        )

        # Third downsampling: 128x128 -> 64x64
        self.down3 = nn.Sequential(
            nn.Conv2d(base_channels * 2, base_channels * 4, 3, padding=1),
            nn.BatchNorm2d(base_channels * 4),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_channels * 4, base_channels * 4, 3, padding=1),
            nn.BatchNorm2d(base_channels * 4),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2)  # 2x downsampling
        )

        # Fourth downsampling: 64x64 -> 32x32
        self.down4 = nn.Sequential(
            nn.Conv2d(base_channels * 4, base_channels * 8, 3, padding=1),
            nn.BatchNorm2d(base_channels * 8),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_channels * 8, base_channels * 8, 3, padding=1),
            nn.BatchNorm2d(base_channels * 8),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2)  # 2x downsampling
        )

    def forward(self, x):
        """
        Forward propagation, returns 4 feature maps at different scales

        Args:
            x: Input image [batch_size, 2, 512, 512]

        Returns:
            features: List containing 4 feature maps
                - f1: [batch_size, base_channels, 256, 256]
                - f2: [batch_size, base_channels*2, 128, 128]
                - f3: [batch_size, base_channels*4, 64, 64]
                - f4: [batch_size, base_channels*8, 32, 32]
        """
        # First downsampling
        f1 = self.down1(x)  # [B, 64, 256, 256]

        # Second downsampling
        f2 = self.down2(f1)  # [B, 128, 128, 128]

        # Third downsampling
        f3 = self.down3(f2)  # [B, 256, 64, 64]

        # Fourth downsampling
        f4 = self.down4(f3)  # [B, 512, 32, 32]

        # Return feature maps at all scales
        return [f1, f2, f3, f4]


class EncoderTransfer:
    """
    Encoder transfer learning: Transfer portal phase encoder parameters to plain phase and freeze
    """

    def __init__(self, base_channels=64):
        self.base_channels = base_channels

        # Create portal phase encoder (source encoder)
        self.portal_encoder = CTEncoder(in_channels=2, base_channels=base_channels)

        # Create plain phase encoder (target encoder)
        self.plain_encoder = CTEncoder(in_channels=2, base_channels=base_channels)

    def load_portal_encoder_weights(self, checkpoint_path: str):
        """
        Load pre-trained encoder weights from portal phase
        """
        checkpoint = torch.load(checkpoint_path)

        # If checkpoint contains entire model, extract encoder part
        if 'encoder_state_dict' in checkpoint:
            self.portal_encoder.load_state_dict(checkpoint['encoder_state_dict'])
        elif 'model_state_dict' in checkpoint:
            # Assume checkpoint saves entire model, we need to extract encoder part
            portal_state_dict = {}
            for key, value in checkpoint['model_state_dict'].items():
                if key.startswith('encoder.'):
                    new_key = key.replace('encoder.', '')
                    portal_state_dict[new_key] = value
            self.portal_encoder.load_state_dict(portal_state_dict)
        else:
            # Directly load encoder weights
            self.portal_encoder.load_state_dict(checkpoint)

        print("Successfully loaded portal phase encoder weights")

    def transfer_and_freeze(self):
        """
        Copy portal phase encoder parameters to plain phase encoder and freeze
        """
        # Copy weights
        portal_state_dict = self.portal_encoder.state_dict()
        self.plain_encoder.load_state_dict(portal_state_dict)

        # Freeze all parameters of plain phase encoder
        for param in self.plain_encoder.parameters():
            param.requires_grad = False

        print("Successfully completed encoder parameter transfer and freezing")
        print(
            f"Plain phase encoder parameter frozen status: {all(not p.requires_grad for p in self.plain_encoder.parameters())}")

    def get_plain_encoder(self) -> CTEncoder:
        """
        Get frozen plain phase encoder
        """
        return self.plain_encoder

    def get_portal_encoder(self) -> CTEncoder:
        """
        Get portal phase encoder (for validation, etc.)
        """
        return self.portal_encoder


class MultiScaleFeatureProcessor:
    """
    Multi-scale feature processor: Demonstrates how to use extracted features
    """

    def __init__(self, encoder: CTEncoder):
        self.encoder = encoder

    def extract_features(self, mixed_input: torch.Tensor):
        """
        Extract multi-scale features

        Args:
            mixed_input: Mixed input [batch_size, 2, 512, 512]

        Returns:
            features: List of 4 feature maps at different scales
        """
        with torch.no_grad():  # Because encoder is frozen
            features = self.encoder(mixed_input)
        return features

    def analyze_features(self, features: list):
        """
        Analyze extracted features (this is just an example, you can modify according to actual needs)
        """
        print("Multi-scale feature analysis:")
        for i, feat in enumerate(features):
            print(f"Feature map {i + 1}: Size {feat.shape}")

        # You can add your specific feature processing logic here
        # For example: feature fusion, attention mechanisms, feature selection, etc.

        return features


# Data preparation function
def prepare_mixed_input(original_images: np.ndarray, enhanced_images: np.ndarray) -> torch.Tensor:
    """
    Prepare mixed input: Stack original and enhanced images as 2 channels

    Args:
        original_images: Original images [batch_size, 512, 512]
        enhanced_images: Enhanced images [batch_size, 512, 512]

    Returns:
        mixed_input: Mixed input [batch_size, 2, 512, 512]
    """
    # Convert to torch tensor
    original_tensor = torch.FloatTensor(original_images).unsqueeze(1)  # [B, 1, 512, 512]
    enhanced_tensor = torch.FloatTensor(enhanced_images).unsqueeze(1)  # [B, 1, 512, 512]

    # Stack as 2 channels
    mixed_input = torch.cat([original_tensor, enhanced_tensor], dim=1)  # [B, 2, 512, 512]

    return mixed_input


# Usage example
if __name__ == "__main__":
    # Initialize encoder transfer
    transfer = EncoderTransfer(base_channels=64)

    # Load pre-trained portal phase encoder weights
    PORTAL_CHECKPOINT = "path/to/portal_encoder_weights.pth"
    transfer.load_portal_encoder_weights(PORTAL_CHECKPOINT)

    # Execute parameter transfer and freezing
    transfer.transfer_and_freeze()

    # Get frozen plain phase encoder
    plain_encoder = transfer.get_plain_encoder()

    # Initialize feature processor
    feature_processor = MultiScaleFeatureProcessor(plain_encoder)

    # Example: Process a batch of data
    batch_size = 4
    # Simulate input data (replace with real data in actual use)
    original_batch = np.random.randn(batch_size, 512, 512).astype(np.float32)
    enhanced_batch = np.random.randn(batch_size, 512, 512).astype(np.float32)

    # Prepare mixed input
    mixed_input = prepare_mixed_input(original_batch, enhanced_batch)

    # Extract multi-scale features
    print("Extracting multi-scale features...")
    multi_scale_features = feature_processor.extract_features(mixed_input)

    # Analyze features
    feature_processor.analyze_features(multi_scale_features)

    print("\nEncoder transfer completed! Now you can use the extracted multi-scale features for subsequent operations.")
    print("Feature size summary:")
    print("- Level 1: 256x256 (high-resolution details)")
    print("- Level 2: 128x128 (medium-resolution features)")
    print("- Level 3: 64x64 (semantic features)")
    print("- Level 4: 32x32 (high-level abstract features)")