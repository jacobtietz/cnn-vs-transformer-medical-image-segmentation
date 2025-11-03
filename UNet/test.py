import os
import torch
from torch.utils.data import DataLoader
from torch import nn
import matplotlib.pyplot as plt
from unet import UNet
from utils import SegmentationDataset, dice_coefficient, mean_iou, predict_and_overlay

def plot_test_metrics(loss, dice, iou, output_dir="plots"):
    os.makedirs(output_dir, exist_ok=True)
    metrics = [loss, dice, iou]
    names = ['Loss', 'Dice', 'IoU']

    plt.figure(figsize=(6,4))
    plt.bar(names, metrics, color=['red', 'green', 'blue'])
    plt.title('Test Set Metrics')
    plt.ylabel('Value')
    for i, v in enumerate(metrics):
        plt.text(i, v + 0.01, f"{v:.4f}", ha='center', va='bottom')
    plt.savefig(os.path.join(output_dir, 'test_metrics.png'))
    plt.close()
    print(f"Saved test metrics plot to {output_dir}/test_metrics.png")

def main():
    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Paths
    data_folder = "data"
    test_images_dir = os.path.join(data_folder, "test_images")
    test_masks_dir = os.path.join(data_folder, "test_masks")  # if you have masks
    save_overlay_dir = "new_data/test"

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

    with torch.no_grad():
        for imgs, masks in test_loader:
            imgs, masks = imgs.to(device), masks.to(device)
            outputs = model(imgs)
            loss = criterion(outputs, masks)

            test_loss += loss.item() * imgs.size(0)
            test_dice += dice_coefficient(masks, outputs).item() * imgs.size(0)
            test_iou += mean_iou(masks, outputs).item() * imgs.size(0)

    # Average metrics
    test_loss /= len(test_loader.dataset)
    test_dice /= len(test_loader.dataset)
    test_iou /= len(test_loader.dataset)

    print(f"Test Loss: {test_loss:.4f}, Dice: {test_dice:.4f}, IoU: {test_iou:.4f}")

    # Save overlayed predictions
    predict_and_overlay(model, device, test_images_dir, save_overlay_dir, image_size=image_size)

    # Save test metrics plot
    plot_test_metrics(test_loss, test_dice, test_iou, output_dir="plots")

if __name__ == "__main__":
    main()
