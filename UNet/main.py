import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from unet import UNet
from utils import SegmentationDataset, dice_coefficient, mean_iou
import json
from tqdm import tqdm
import albumentations as A
from albumentations.pytorch import ToTensorV2

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    data_folder = 'data'
    train_images_dir = os.path.join(data_folder, 'train_images')
    train_masks_dir = os.path.join(data_folder, 'train_masks')
    val_images_dir = os.path.join(data_folder, 'val_images')
    val_masks_dir = os.path.join(data_folder, 'val_masks')

    image_size = (256, 256)
    batch_size = 64
    epochs = 70
    learning_rate = 1e-3

    train_transform = A.Compose([
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=0.5),
        A.RandomResizedCrop(
            size=image_size,
            scale=(0.8, 1.0),
            ratio=(0.75, 1.333),
            interpolation=1,
            p=0.5
        ),
        A.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
        ToTensorV2()
    ])

    val_transform = A.Compose([
        A.Resize(height=image_size[0], width=image_size[1]),
        A.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
        ToTensorV2()
    ])

    train_dataset = SegmentationDataset(
        train_images_dir, train_masks_dir, image_size=image_size, transform=train_transform
    )
    val_dataset = SegmentationDataset(
        val_images_dir, val_masks_dir, image_size=image_size, transform=val_transform
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2)

    model = UNet(input_channels=3, output_channels=1).to(device)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    history = {'loss': [], 'dice': [], 'iou': [], 'val_loss': [], 'val_dice': [], 'val_iou': []}

    for epoch in range(epochs):
        model.train()
        train_loss = train_dice = train_iou = 0

        train_pbar = tqdm(train_loader, desc=f"Epoch {epoch+1} Training", ncols=100)
        for imgs, masks in train_pbar:
            imgs, masks = imgs.to(device), masks.to(device)
            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * imgs.size(0)
            train_dice += dice_coefficient(masks, outputs).item() * imgs.size(0)
            train_iou += mean_iou(masks, outputs).item() * imgs.size(0)
            train_pbar.set_postfix({"loss": loss.item()})

        train_loss /= len(train_loader.dataset)
        train_dice /= len(train_loader.dataset)
        train_iou /= len(train_loader.dataset)

        model.eval()
        val_loss = val_dice = val_iou = 0

        val_pbar = tqdm(val_loader, desc=f"Epoch {epoch+1} Validation", ncols=100)
        with torch.no_grad():
            for imgs, masks in val_pbar:
                imgs, masks = imgs.to(device), masks.to(device)
                outputs = model(imgs)
                loss = criterion(outputs, masks)
                val_loss += loss.item() * imgs.size(0)
                val_dice += dice_coefficient(masks, outputs).item() * imgs.size(0)
                val_iou += mean_iou(masks, outputs).item() * imgs.size(0)
                val_pbar.set_postfix({"val_loss": loss.item()})

        val_loss /= len(val_loader.dataset)
        val_dice /= len(val_loader.dataset)
        val_iou /= len(val_loader.dataset)

        print(f"\nEpoch [{epoch+1}/{epochs}] "
              f"Train Loss: {train_loss:.4f}, Dice: {train_dice:.4f}, IoU: {train_iou:.4f} | "
              f"Val Loss: {val_loss:.4f}, Dice: {val_dice:.4f}, IoU: {val_iou:.4f}\n")

        history['loss'].append(train_loss)
        history['dice'].append(train_dice)
        history['iou'].append(train_iou)
        history['val_loss'].append(val_loss)
        history['val_dice'].append(val_dice)
        history['val_iou'].append(val_iou)

    os.makedirs("metrics", exist_ok=True)
    with open("metrics/training_metrics.txt", "w") as f:
        json.dump(history, f, indent=4)

    os.makedirs("models", exist_ok=True)
    torch.save(model.state_dict(), "models/unet_weights.pth")
    torch.save(model, "models/unet_model.pth")

    print("Training complete.")

if __name__ == "__main__":
    main()
