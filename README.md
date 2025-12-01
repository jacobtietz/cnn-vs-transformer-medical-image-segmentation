# Comparative Analysis of CNN and Transformer-Based Models for 2D Medical Image Segmentation (ISIC 2018 Lesions)

This repository contains implementations and experiments comparing **UNet** and **TransUNet** architectures for medical image segmentation on the ISIC 2018 skin lesion dataset.

---

## Dataset

The dataset can be downloaded from the official ISIC archive:  
[ISIC 2018 Dataset](https://challenge.isic-archive.com/data/#2018)  

Place the downloaded dataset in the `data` folder of this repository.

---

## Software Requirements

Python packages required for running the models are listed in [`requirements.txt`](requirements.txt).

- **Python:** 3.13.7  
- **CUDA Version:** 12.9  
- **NVIDIA Driver Version:** 576.83  

---

## Hardware

The models were trained on the following hardware:

- **GPU:** NVIDIA RTX 4090 (24GB)  
- **RAM:** 64GB  
- **CPU:** Intel(R) Core(TM) i9-14900KF  

---

## Pre-trained Models

For **TransUNet**, you need to download Google pre-trained Vision Transformer (ViT) models:

1. Visit the storage link: [Google ViT Models](https://console.cloud.google.com/storage/browser/vit_models)  
   Available models include: `R50-ViT-B_16`, `ViT-B_16`, `ViT-L_16`, etc.

2. Download the model using `wget`, create the checkpoint folder, and move the file:

```bash
wget https://storage.googleapis.com/vit_models/imagenet21k/{MODEL_NAME}.npz
mkdir -p ../model/vit_checkpoint/imagenet21k
mv {MODEL_NAME}.npz ../model/vit_checkpoint/imagenet21k/{MODEL_NAME}.npz
```

---

## UNet

<img width="1063" height="519" alt="UNet Architecture" src="https://github.com/user-attachments/assets/3d622cca-6ef9-4e79-a2c6-c4c22c7386af" />

---

## [TransUNet(https://github.com/Beckschen/TransUNet?tab=readme-ov-file)](https://github.com/Beckschen/TransUNet?tab=readme-ov-file)

<img width="1063" height="519" alt="TransUNet Architecture" src="https://github.com/user-attachments/assets/d19f9a84-c774-4390-8b71-c83ca12b3815" />
