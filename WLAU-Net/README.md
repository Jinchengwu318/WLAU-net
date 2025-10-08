# Liver Tumor Segmentation with Improved TransUnet

This repository contains the implementation of our paper "Liver Tumor Segmentation in Non-Contrast CT using Improved TransUnet and Transfer Learning".

## Features

- **Wavelet-based Image Enhancement**: Enhances non-contrast CT images using sym4 wavelet
- **Transfer Learning**: Pre-trained on portal phase CT and transferred to non-contrast CT  
- **Gaussian-weighted Attention**: Incorporates tumor prior in Transformer attention
- **Multi-scale Feature Fusion**: Combines features from multiple scales

## Installation

```bash
git clone https://github.com/your_username/liver-tumor-segmentation.git
cd liver-tumor-segmentation
pip install -r requirements.txt