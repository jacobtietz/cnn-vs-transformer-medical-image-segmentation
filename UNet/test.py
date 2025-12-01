import os
import torch
from torch.utils.data import DataLoader
from torch import nn
import matplotlib.pyplot as plt
from unet import UNet
from utils import SegmentationDataset, dice_coefficient, mean_iou, predict_and_overlay
from tqdm import tqdm
import json

def plot_combined_metrics(history, test_metrics, output_dir="plots"):
    os.makedirs(output_dir, exist_ok=True)
    epochs = range(1, len(history['loss'])+1)
    plt.figure(figsize=(8,6))

    plt.plot(epochs, history['loss'], label='Train Loss', color='red', marker='o')
    plt.plot(epochs, history['val_loss'], label='Val Loss', color='red', marker='x')
    plt.hlines(test_metrics['loss'], 1, len(epochs), colors='red', linestyles='dashed', label='Validation Loss')

    plt.plot(epochs, history['dice'], label='Train Dice', color='green', marker='o')
    plt.plot(epochs, history['val_dice'], label='Val Dice', color='green', marker='x')
    plt.hlines(test_metrics['dice'], 1, len(epochs), colors='green', linestyles='dashed', label='Validation Dice')

    plt.plot(epochs, history['iou'], label='Train IoU', color='blue', marker='o')
    plt.plot(epochs, history['val_iou'], label='Val IoU', color='blue', marker='x')
    plt.hlines(test_metrics['iou'], 1, len(epochs), colors='blue', linestyles='dashed', label='Validation IoU')

    plt.xlabel('Epochs')
    plt.ylabel('Metric')
    plt.title('Segmentation Metrics: Train / Val / Validation')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'combined_metrics.png'))
    plt.close()
    print(f"Saved combined metrics plot to {output_dir}/combined_metrics.png")

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    data_folder = "data"
    val_images_dir = os.path.join(data_folder, "val_images")
    val_masks_dir = os.path.join(data_folder, "val_masks")
    save_overlay_dir = "new_data/val"
    metrics_file = "metrics/training_metrics.txt"

    image_size = (256, 256)
    batch_size = 32

    val_dataset = SegmentationDataset(val_images_dir, val_masks_dir, image_size=image_size)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2)

    model = UNet(input_channels=3, output_channels=1)
    model.load_state_dict(torch.load("models/unet_weights.pth", map_location=device))
    model.to(device)
    model.eval()

    criterion = nn.BCELoss()
    val_loss = val_dice = val_iou = 0

    val_pbar = tqdm(val_loader, desc="Validating", ncols=100)
    with torch.no_grad():
        for imgs, masks in val_pbar:
            imgs, masks = imgs.to(device), masks.to(device)
            outputs = model(imgs)
            loss = criterion(outputs, masks)
            val_loss += loss.item() * imgs.size(0)
            val_dice += dice_coefficient(masks, outputs).item() * imgs.size(0)
            val_iou += mean_iou(masks, outputs).item() * imgs.size(0)
            val_pbar.set_postfix({"loss": loss.item()})

    val_loss /= len(val_loader.dataset)
    val_dice /= len(val_loader.dataset)
    val_iou /= len(val_loader.dataset)
    print(f"\nValidation Loss: {val_loss:.4f}, Dice: {val_dice:.4f}, IoU: {val_iou:.4f}")

    predict_and_overlay(model, device, val_images_dir, save_overlay_dir, image_size=image_size)

    os.makedirs("metrics", exist_ok=True)
    val_metrics = {"loss": val_loss, "dice": val_dice, "iou": val_iou}
    with open("metrics/val_metrics.txt", "w") as f:
        json.dump(val_metrics, f, indent=4)

    if os.path.exists(metrics_file):
        with open(metrics_file, "r") as f:
            history = json.load(f)
        plot_combined_metrics(history, val_metrics)
    else:
        print("Training metrics file not found. Skipping combined plot.")

if __name__ == "__main__":
    main()
