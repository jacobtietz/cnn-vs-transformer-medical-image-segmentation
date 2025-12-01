import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from tqdm import tqdm

class SegmentationDataset(Dataset):
    def __init__(self, image_dir, mask_dir, image_size=(256, 256), transform=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.image_size = image_size
        self.transform = transform

        self.image_files = sorted(os.listdir(image_dir))
        self.mask_files = sorted(os.listdir(mask_dir))

        self.pairs = []
        for img_file, mask_file in zip(self.image_files, self.mask_files):
            if img_file.split('.')[0] == mask_file.split('_segmentation')[0]:
                self.pairs.append((img_file, mask_file))
            else:
                print(f"Skipping unmatched pair: {img_file} and {mask_file}")

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        img_file, mask_file = self.pairs[idx]
        img_path = os.path.join(self.image_dir, img_file)
        mask_path = os.path.join(self.mask_dir, mask_file)

        img = np.array(Image.open(img_path).convert("RGB").resize(self.image_size))
        mask = np.array(Image.open(mask_path).convert("L").resize(self.image_size))

        if self.transform:
            augmented = self.transform(image=img, mask=mask)
            img = augmented['image']
            mask = augmented['mask'].unsqueeze(0).float()
        else:
            img = transforms.ToTensor()(img)
            mask = torch.tensor(mask / 255., dtype=torch.float32).unsqueeze(0)

        mask = (mask > 0.5).float()
        return img, mask

def dice_coefficient(y_true, y_pred, smooth=1e-6):
    y_true = y_true.view(-1)
    y_pred = y_pred.view(-1)
    intersection = (y_true * y_pred).sum()
    return (2. * intersection + smooth) / (y_true.sum() + y_pred.sum() + smooth)

def mean_iou(y_true, y_pred, threshold=0.5):
    y_pred = (y_pred > threshold).float()
    intersection = (y_true * y_pred).sum(dim=[1,2,3])
    union = y_true.sum(dim=[1,2,3]) + y_pred.sum(dim=[1,2,3]) - intersection
    iou = (intersection + 1e-6) / (union + 1e-6)
    return iou.mean()

def plot_metrics(history, output_dir="plots"):
    os.makedirs(output_dir, exist_ok=True)

    if 'dice' in history and 'val_dice' in history:
        plt.figure(figsize=(6,4))
        plt.plot(history['dice'], label='Train Dice')
        plt.plot(history['val_dice'], label='Val Dice')
        plt.title('Dice Coefficient')
        plt.xlabel('Epochs')
        plt.ylabel('Dice')
        plt.legend()
        plt.savefig(os.path.join(output_dir, 'dice_coefficient.png'))
        plt.close()

    if 'iou' in history and 'val_iou' in history:
        plt.figure(figsize=(6,4))
        plt.plot(history['iou'], label='Train IoU')
        plt.plot(history['val_iou'], label='Val IoU')
        plt.title('IoU')
        plt.xlabel('Epochs')
        plt.ylabel('IoU')
        plt.legend()
        plt.savefig(os.path.join(output_dir, 'iou.png'))
        plt.close()

    if 'loss' in history and 'val_loss' in history:
        plt.figure(figsize=(6,4))
        plt.plot(history['loss'], label='Train Loss')
        plt.plot(history['val_loss'], label='Val Loss')
        plt.title('Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.savefig(os.path.join(output_dir, 'loss.png'))
        plt.close()

def predict_and_overlay(model, device, image_dir, save_dir, image_size=(128, 128)):
    os.makedirs(save_dir, exist_ok=True)
    model.eval()

    transform = transforms.Compose([
        transforms.Resize(image_size),
        transforms.ToTensor()
    ])

    image_files = sorted(os.listdir(image_dir))
    print("Saving overlayed images...")

    with torch.no_grad():
        for img_file in tqdm(image_files, desc="Overlay Progress"):
            img_path = os.path.join(image_dir, img_file)
            img = Image.open(img_path).convert("RGB")
            input_tensor = transform(img).unsqueeze(0).to(device)

            pred_mask = model(input_tensor).squeeze(0).squeeze(0).cpu()
            binary_mask = (pred_mask > 0.5).float()

            img_np = np.array(img.resize(image_size)) / 255.0
            mask_np = binary_mask.numpy()
            overlay = img_np * np.expand_dims(mask_np, axis=2)

            overlay_img = Image.fromarray((overlay * 255).astype(np.uint8))
            overlay_img.save(os.path.join(save_dir, img_file))

    print("All overlayed images saved!")
