import numpy as np
import pydicom
import pywt
import matplotlib.pyplot as plt
from typing import Tuple, List
import os
import cv2


class WaveletEnhancement:
    """
    CT image enhancement class based on sym4 wavelet
    """

    def __init__(self, lambda1: float = 1.2, lambda2: float = 1.1, wavelet: str = 'sym4', level: int = 4):
        """
        Initialize enhancement parameters

        Args:
            lambda1: Enhancement coefficient for HL frequency band
            lambda2: Enhancement coefficient for LH frequency band
            wavelet: Wavelet type, default is sym4
            level: Wavelet decomposition levels, default is 4
        """
        self.lambda1 = lambda1
        self.lambda2 = lambda2
        self.wavelet = wavelet
        self.level = level

    def load_dicom(self, dicom_path: str) -> np.ndarray:
        """
        Load DICOM file and convert to numpy array

        Args:
            dicom_path: DICOM file path

        Returns:
            CT image data
        """
        dicom_data = pydicom.dcmread(dicom_path)
        ct_image = dicom_data.pixel_array.astype(np.float32)

        # Apply window width/level (if needed)
        if hasattr(dicom_data, 'WindowCenter') and hasattr(dicom_data, 'WindowWidth'):
            window_center = dicom_data.WindowCenter
            window_width = dicom_data.WindowWidth
            if isinstance(window_center, pydicom.multival.MultiValue):
                window_center = window_center[0]
            if isinstance(window_width, pydicom.multival.MultiValue):
                window_width = window_width[0]

            ct_image = self.apply_window_level(ct_image, window_center, window_width)

        return ct_image

    def apply_window_level(self, image: np.ndarray, center: float, width: float) -> np.ndarray:
        """
        Apply window width/level processing
        """
        min_val = center - width / 2
        max_val = center + width / 2
        image = np.clip(image, min_val, max_val)
        image = (image - min_val) / (max_val - min_val)
        return image

    def wavelet_decomposition(self, image: np.ndarray) -> Tuple:
        """
        Perform 4-level decomposition using sym4 wavelet

        Args:
            image: Input image

        Returns:
            Wavelet decomposition coefficients
        """
        # Ensure image size is suitable (wavelet transform requires size divisible by 2^level)
        original_shape = image.shape
        target_shape = self._get_optimal_shape(original_shape)

        if original_shape != target_shape:
            image = cv2.resize(image, target_shape, interpolation=cv2.INTER_LINEAR)

        # Perform wavelet decomposition
        coeffs = pywt.wavedec2(image, self.wavelet, level=self.level)
        return coeffs, original_shape, target_shape

    def _get_optimal_shape(self, shape: Tuple[int, int]) -> Tuple[int, int]:
        """
        Calculate optimal size suitable for wavelet decomposition
        """
        base = 2 ** self.level
        h = (shape[0] // base) * base
        w = (shape[1] // base) * base
        return (w, h)  # OpenCV uses (width, height)

    def enhance_frequency_bands(self, coeffs: List) -> List:
        """
        Enhance specific frequency bands: HL multiplied by λ1, LH multiplied by λ2

        Args:
            coeffs: Wavelet decomposition coefficients

        Returns:
            Enhanced wavelet coefficients
        """
        enhanced_coeffs = [coeffs[0]]  # Keep LL low-frequency component

        # Process high-frequency components at each level
        for i in range(1, len(coeffs)):
            cH, cV, cD = coeffs[i]  # Horizontal detail (cH), Vertical detail (cV), Diagonal detail (cD)

            # Enhance HL and LH frequency bands
            enhanced_cH = cH * self.lambda1  # HL band × λ1
            enhanced_cV = cV * self.lambda2  # LH band × λ2
            enhanced_cD = cD  # HH band remains unchanged

            enhanced_coeffs.append((enhanced_cH, enhanced_cV, enhanced_cD))

        return enhanced_coeffs

    def inverse_wavelet_transform(self, coeffs: List, target_shape: Tuple[int, int],
                                  original_shape: Tuple[int, int]) -> np.ndarray:
        """
        Inverse wavelet transform to reconstruct image

        Args:
            coeffs: Wavelet coefficients
            target_shape: Target size for wavelet transform
            original_shape: Original image size

        Returns:
            Reconstructed enhanced image
        """
        # Inverse wavelet transform
        enhanced_image = pywt.waverec2(coeffs, self.wavelet)

        # Resize back to original dimensions
        if target_shape != original_shape:
            enhanced_image = cv2.resize(enhanced_image, original_shape, interpolation=cv2.INTER_LINEAR)

        return enhanced_image

    def enhance_image(self, image: np.ndarray) -> np.ndarray:
        """
        Complete image enhancement pipeline

        Args:
            image: Input CT image

        Returns:
            Enhanced image
        """
        # Wavelet decomposition
        coeffs, original_shape, target_shape = self.wavelet_decomposition(image)

        # Frequency band enhancement
        enhanced_coeffs = self.enhance_frequency_bands(coeffs)

        # Inverse transform reconstruction
        enhanced_image = self.inverse_wavelet_transform(enhanced_coeffs, target_shape, original_shape)

        return enhanced_image

    def process_dicom_series(self, dicom_folder: str, output_folder: str):
        """
        Process entire DICOM series

        Args:
            dicom_folder: Input DICOM folder path
            output_folder: Output folder path
        """
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)

        dicom_files = [f for f in os.listdir(dicom_folder) if f.endswith('.dcm')]

        for i, filename in enumerate(dicom_files):
            print(f"Processing file {i + 1}/{len(dicom_files)}: {filename}")

            try:
                # Load DICOM
                dicom_path = os.path.join(dicom_folder, filename)
                ct_image = self.load_dicom(dicom_path)

                # Enhancement processing
                enhanced_image = self.enhance_image(ct_image)

                # Save result (saved as numpy format here, you can modify as needed)
                output_path = os.path.join(output_folder, f"enhanced_{filename.replace('.dcm', '.npy')}")
                np.save(output_path, enhanced_image)

            except Exception as e:
                print(f"Error processing file {filename}: {e}")
                continue


def visualize_comparison(original: np.ndarray, enhanced: np.ndarray,
                         lambda1: float, lambda2: float):
    """
    Visualize comparison between original and enhanced images
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

    ax1.imshow(original, cmap='gray')
    ax1.set_title('Original Image')
    ax1.axis('off')

    ax2.imshow(enhanced, cmap='gray')
    ax2.set_title(f'Enhanced Image (λ1={lambda1}, λ2={lambda2})')
    ax2.axis('off')

    plt.tight_layout()
    plt.show()


# Usage example
if __name__ == "__main__":
    # Initialize enhancer
    enhancer = WaveletEnhancement(lambda1=1.2, lambda2=1.1)

    # Single image processing example
    # dicom_path = "path/to/your/dicom_file.dcm"
    # ct_image = enhancer.load_dicom(dicom_path)
    # enhanced_image = enhancer.enhance_image(ct_image)
    # visualize_comparison(ct_image, enhanced_image, enhancer.lambda1, enhancer.lambda2)

    # Batch processing example
    # input_folder = "path/to/your/dicom/folder"
    # output_folder = "path/to/output/folder"
    # enhancer.process_dicom_series(input_folder, output_folder)

    print("Wavelet enhancement code is ready!")
    print("Please modify the file paths in the example and uncomment to run")