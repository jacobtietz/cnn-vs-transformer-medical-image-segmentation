import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from unet import UNet
from utils import SegmentationDataset, dice_coefficient, mean_iou, plot_metrics, predict_and_overlay

def main():
    # Check device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Paths
    data_folder = 'data'
    train_images_dir = os.path.join(data_folder, 'train_images')
    train_masks_dir = os.path.join(data_folder, 'train_masks')
    val_images_dir = os.path.join(data_folder, 'val_images')
    val_masks_dir = os.path.join(data_folder, 'val_masks')

    # Hyperparameters
    image_size = (256, 256)
    batch_size = 32
    epochs = 1
    learning_rate = 1e-3

    # Dataset and DataLoader
    train_dataset = SegmentationDataset(train_images_dir, train_masks_dir, image_size=image_size)
    val_dataset = SegmentationDataset(val_images_dir, val_masks_dir, image_size=image_size)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2)

    # Model, optimizer, loss
    model = UNet(input_channels=3, output_channels=1).to(device)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Track metrics
    history = {
        'loss': [],
        'dice': [],
        'iou': [],
        'val_loss': [],
        'val_dice': [],
        'val_iou': []
    }

    for epoch in range(epochs):
        model.train()
        train_loss = 0
        train_dice = 0
        train_iou = 0

        for imgs, masks in train_loader:
            imgs, masks = imgs.to(device), masks.to(device)
            optimizer.zero_grad()

            outputs = model(imgs)
            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * imgs.size(0)
            train_dice += dice_coefficient(masks, outputs).item() * imgs.size(0)
            train_iou += mean_iou(masks, outputs).item() * imgs.size(0)

        train_loss /= len(train_loader.dataset)
        train_dice /= len(train_loader.dataset)
        train_iou /= len(train_loader.dataset)

        # Validation
        model.eval()
        val_loss = 0
        val_dice = 0
        val_iou = 0
        with torch.no_grad():
            for imgs, masks in val_loader:
                imgs, masks = imgs.to(device), masks.to(device)
                outputs = model(imgs)
                loss = criterion(outputs, masks)

                val_loss += loss.item() * imgs.size(0)
                val_dice += dice_coefficient(masks, outputs).item() * imgs.size(0)
                val_iou += mean_iou(masks, outputs).item() * imgs.size(0)

        val_loss /= len(val_loader.dataset)
        val_dice /= len(val_loader.dataset)
        val_iou /= len(val_loader.dataset)

        print(f"Epoch [{epoch+1}/{epochs}] "
              f"Train Loss: {train_loss:.4f}, Dice: {train_dice:.4f}, IoU: {train_iou:.4f} | "
              f"Val Loss: {val_loss:.4f}, Dice: {val_dice:.4f}, IoU: {val_iou:.4f}")

        history['loss'].append(train_loss)
        history['dice'].append(train_dice)
        history['iou'].append(train_iou)
        history['val_loss'].append(val_loss)
        history['val_dice'].append(val_dice)
        history['val_iou'].append(val_iou)

    # Save model and weights
    os.makedirs("models", exist_ok=True)
    torch.save(model.state_dict(), "models/unet_weights.pth")
    torch.save(model, "models/unet_model.pth")  # Entire model (optional)

    # Plot metrics
    plot_metrics(history)

    # Predict and save overlayed images
    predict_and_overlay(model, device, "data/train_images", "new_data/train", image_size=image_size)


if __name__ == "__main__":
    main()
