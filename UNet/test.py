# test.py
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
    """
    history: dict loaded from training_metrics.txt
    test_metrics: dict with 'loss', 'dice', 'iou' keys
    """
    os.makedirs(output_dir, exist_ok=True)

    # Create lists for training/val/test
    epochs = range(1, len(history['loss'])+1)

    plt.figure(figsize=(8,6))

    # Loss
    plt.plot(epochs, history['loss'], label='Train Loss', color='red', marker='o')
    plt.plot(epochs, history['val_loss'], label='Val Loss', color='red', marker='x')
    plt.hlines(test_metrics['loss'], 1, len(epochs), colors='red', linestyles='dashed', label='Test Loss')

    # Dice
    plt.plot(epochs, history['dice'], label='Train Dice', color='green', marker='o')
    plt.plot(epochs, history['val_dice'], label='Val Dice', color='green', marker='x')
    plt.hlines(test_metrics['dice'], 1, len(epochs), colors='green', linestyles='dashed', label='Test Dice')

    # IoU
    plt.plot(epochs, history['iou'], label='Train IoU', color='blue', marker='o')
    plt.plot(epochs, history['val_iou'], label='Val IoU', color='blue', marker='x')
    plt.hlines(test_metrics['iou'], 1, len(epochs), colors='blue', linestyles='dashed', label='Test IoU')

    plt.xlabel('Epochs')
    plt.ylabel('Metric')
    plt.title('Segmentation Metrics: Train / Val / Test')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'combined_metrics.png'))
    plt.close()
    print(f"Saved combined metrics plot to {output_dir}/combined_metrics.png")

def main():
    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Paths
    data_folder = "data"
    test_images_dir = os.path.join(data_folder, "test_images")
    test_masks_dir = os.path.join(data_folder, "test_masks")  # if you have masks
    save_overlay_dir = "new_data/test"
    metrics_file = "metrics/training_metrics.txt"

    # Hyperparameters
    image_size = (256, 256)
    batch_size = 32

    # Dataset and DataLoader
    test_dataset = SegmentationDataset(test_images_dir, test_masks_dir, image_size=image_size)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2)

    # Load model
    model = UNet(input_channels=3, output_channels=1)
    model.load_state_dict(torch.load("models/unet_weights.pth", map_location=device))
    model.to(device)
    model.eval()

    criterion = nn.BCELoss()

    # Metrics
    test_loss = 0
    test_dice = 0
    test_iou = 0

    # TEST LOOP WITH TQDM
    test_pbar = tqdm(test_loader, desc="Testing", ncols=100)
    with torch.no_grad():
        for imgs, masks in test_pbar:
            imgs, masks = imgs.to(device), masks.to(device)
            outputs = model(imgs)
            loss = criterion(outputs, masks)

            test_loss += loss.item() * imgs.size(0)
            test_dice += dice_coefficient(masks, outputs).item() * imgs.size(0)
            test_iou += mean_iou(masks, outputs).item() * imgs.size(0)

            test_pbar.set_postfix({"loss": loss.item()})

    # Average metrics
    test_loss /= len(test_loader.dataset)
    test_dice /= len(test_loader.dataset)
    test_iou /= len(test_loader.dataset)

    print(f"\nTest Loss: {test_loss:.4f}, Dice: {test_dice:.4f}, IoU: {test_iou:.4f}")

    # Save overlayed predictions
    predict_and_overlay(model, device, test_images_dir, save_overlay_dir, image_size=image_size)

    # Save metrics to txt
    os.makedirs("metrics", exist_ok=True)
    test_metrics = {
        "loss": test_loss,
        "dice": test_dice,
        "iou": test_iou
    }
    with open("metrics/test_metrics.txt", "w") as f:
        json.dump(test_metrics, f, indent=4)

    # Plot combined metrics if training history exists
    if os.path.exists(metrics_file):
        with open(metrics_file, "r") as f:
            history = json.load(f)
        plot_combined_metrics(history, test_metrics)
    else:
        print("Training metrics file not found. Skipping combined plot.")

if __name__ == "__main__":
    main()
