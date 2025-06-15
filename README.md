# Visual-Language Model with CLIP-ViT and Qwen-1.5B

This repository implements a vision-language model where the visual part is based on CLIP-ViT and the language model is based on Qwen-1.5B. Below you will find instructions on how to set up the environment and download the necessary models.

## Model Overview

- **Visual Model**: openai/clip-vit-base-patch32
- **Language Model**: Qwen/Qwen2.5-1.5B
- **Dataset**: jxie/coco_captions

The visual model extracts image features using the Vision Transformer (ViT) architecture and also the Resnet50/101, while the language model processes and generates text based on those features.

## How to download these models and datasets

<href>https://hf-mirror.com/

## Requirements

Make sure to have the following dependencies installed:

- Python 
- PyTorch 
- Transformers 
- Other libraries as required by your specific setup (e.g., NumPy, OpenCV)

You can install the required libraries using pip:

```bash
pip install torch transformers datasets numpy opencv-python
