import numpy as np
import pydicom
import pywt
import matplotlib.pyplot as plt
from typing import Tuple, List
import os
import cv2

class WaveletEnhancement:
    """
    基于sym4小波的CT图像增强类
    """
    
    def __init__(self, lambda1: float = 1.2, lambda2: float = 1.1, wavelet: str = 'sym4', level: int = 4):
        """
        初始化增强参数
        
        Args:
            lambda1: HL频段的增强系数
            lambda2: LH频段的增强系数  
            wavelet: 小波类型，默认为sym4
            level: 小波分解层数，默认为4
        """
        self.lambda1 = lambda1
        self.lambda2 = lambda2
        self.wavelet = wavelet
        self.level = level
    
    def load_dicom(self, dicom_path: str) -> np.ndarray:
        """
        加载DICOM文件并转换为numpy数组
        
        Args:
            dicom_path: DICOM文件路径
            
        Returns:
            CT图像数据
        """
        dicom_data = pydicom.dcmread(dicom_path)
        ct_image = dicom_data.pixel_array.astype(np.float32)
        
        # 应用窗宽窗位（如果需要）
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
        应用窗宽窗位处理
        """
        min_val = center - width / 2
        max_val = center + width / 2
        image = np.clip(image, min_val, max_val)
        image = (image - min_val) / (max_val - min_val)
        return image
    
    def wavelet_decomposition(self, image: np.ndarray) -> Tuple:
        """
        使用sym4小波进行4层分解
        
        Args:
            image: 输入图像
            
        Returns:
            小波分解系数
        """
        # 确保图像尺寸合适（小波变换要求尺寸可被2^level整除）
        original_shape = image.shape
        target_shape = self._get_optimal_shape(original_shape)
        
        if original_shape != target_shape:
            image = cv2.resize(image, target_shape, interpolation=cv2.INTER_LINEAR)
        
        # 进行小波分解
        coeffs = pywt.wavedec2(image, self.wavelet, level=self.level)
        return coeffs, original_shape, target_shape
    
    def _get_optimal_shape(self, shape: Tuple[int, int]) -> Tuple[int, int]:
        """
        计算适合小波分解的最佳尺寸
        """
        base = 2 ** self.level
        h = (shape[0] // base) * base
        w = (shape[1] // base) * base
        return (w, h)  # OpenCV使用(width, height)
    
    def enhance_frequency_bands(self, coeffs: List) -> List:
        """
        增强特定频段：HL乘以λ1，LH乘以λ2
        
        Args:
            coeffs: 小波分解系数
            
        Returns:
            增强后的小波系数
        """
        enhanced_coeffs = [coeffs[0]]  # 保留LL低频分量
        
        # 处理各层高频分量
        for i in range(1, len(coeffs)):
            cH, cV, cD = coeffs[i]  # 水平细节(cH), 垂直细节(cV), 对角细节(cD)
            
            # 增强HL和LH频段
            enhanced_cH = cH * self.lambda1  # HL频段 × λ1
            enhanced_cV = cV * self.lambda2  # LH频段 × λ2
            enhanced_cD = cD  # HH频段保持不变
            
            enhanced_coeffs.append((enhanced_cH, enhanced_cV, enhanced_cD))
        
        return enhanced_coeffs
    
    def inverse_wavelet_transform(self, coeffs: List, target_shape: Tuple[int, int], 
                                 original_shape: Tuple[int, int]) -> np.ndarray:
        """
        逆小波变换重构图像
        
        Args:
            coeffs: 小波系数
            target_shape: 小波变换时的目标尺寸
            original_shape: 原始图像尺寸
            
        Returns:
            重构后的增强图像
        """
        # 逆小波变换
        enhanced_image = pywt.waverec2(coeffs, self.wavelet)
        
        # 调整回原始尺寸
        if target_shape != original_shape:
            enhanced_image = cv2.resize(enhanced_image, original_shape, interpolation=cv2.INTER_LINEAR)
        
        return enhanced_image
    
    def enhance_image(self, image: np.ndarray) -> np.ndarray:
        """
        完整的图像增强流程
        
        Args:
            image: 输入CT图像
            
        Returns:
            增强后的图像
        """
        # 小波分解
        coeffs, original_shape, target_shape = self.wavelet_decomposition(image)
        
        # 频段增强
        enhanced_coeffs = self.enhance_frequency_bands(coeffs)
        
        # 逆变换重构
        enhanced_image = self.inverse_wavelet_transform(enhanced_coeffs, target_shape, original_shape)
        
        return enhanced_image
    
    def process_dicom_series(self, dicom_folder: str, output_folder: str):
        """
        处理整个DICOM序列
        
        Args:
            dicom_folder: 输入DICOM文件夹路径
            output_folder: 输出文件夹路径
        """
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)
        
        dicom_files = [f for f in os.listdir(dicom_folder) if f.endswith('.dcm')]
        
        for i, filename in enumerate(dicom_files):
            print(f"处理文件 {i+1}/{len(dicom_files)}: {filename}")
            
            try:
                # 加载DICOM
                dicom_path = os.path.join(dicom_folder, filename)
                ct_image = self.load_dicom(dicom_path)
                
                # 增强处理
                enhanced_image = self.enhance_image(ct_image)
                
                # 保存结果（这里保存为numpy格式，你可以根据需要修改）
                output_path = os.path.join(output_folder, f"enhanced_{filename.replace('.dcm', '.npy')}")
                np.save(output_path, enhanced_image)
                
            except Exception as e:
                print(f"处理文件 {filename} 时出错: {e}")
                continue

def visualize_comparison(original: np.ndarray, enhanced: np.ndarray, 
                        lambda1: float, lambda2: float):
    """
    可视化原始图像和增强图像的对比
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
    
    ax1.imshow(original, cmap='gray')
    ax1.set_title('原始图像')
    ax1.axis('off')
    
    ax2.imshow(enhanced, cmap='gray')
    ax2.set_title(f'增强图像 (λ1={lambda1}, λ2={lambda2})')
    ax2.axis('off')
    
    plt.tight_layout()
    plt.show()

# 使用示例
if __name__ == "__main__":
    # 初始化增强器
    enhancer = WaveletEnhancement(lambda1=1.2, lambda2=1.1)
    
    # 单张图像处理示例
    # dicom_path = "path/to/your/dicom_file.dcm"
    # ct_image = enhancer.load_dicom(dicom_path)
    # enhanced_image = enhancer.enhance_image(ct_image)
    # visualize_comparison(ct_image, enhanced_image, enhancer.lambda1, enhancer.lambda2)
    
    # 批量处理示例
    # input_folder = "path/to/your/dicom/folder"
    # output_folder = "path/to/output/folder"
    # enhancer.process_dicom_series(input_folder, output_folder)
    
    print("小波增强代码已准备就绪！")
    print("请修改示例中的文件路径并取消注释来运行")