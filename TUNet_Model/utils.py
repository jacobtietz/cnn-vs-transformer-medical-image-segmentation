import numpy as np
import torch
from medpy import metric
from scipy.ndimage import zoom
import torch.nn as nn
import SimpleITK as sitk
import os
from PIL import Image

class DiceLoss(nn.Module):
    def __init__(self, n_classes):
        super(DiceLoss, self).__init__()
        self.n_classes = n_classes

    def _one_hot_encoder(self, input_tensor):
        return torch.cat([(input_tensor == i).unsqueeze(1) for i in range(self.n_classes)], dim=1).float()

    def _dice_loss(self, score, target):
        smooth = 1e-5
        intersect = torch.sum(score * target)
        return 1 - (2 * intersect + smooth) / (torch.sum(score*score) + torch.sum(target*target) + smooth)

    def forward(self, inputs, target, weight=None, softmax=False):
        if softmax:
            inputs = torch.softmax(inputs, dim=1)
        target = self._one_hot_encoder(target)
        if weight is None:
            weight = [1] * self.n_classes
        assert inputs.size() == target.size(), f'predict {inputs.size()} & target {target.size()} do not match'
        loss = sum(self._dice_loss(inputs[:, i], target[:, i]) * weight[i] for i in range(self.n_classes))
        return loss / self.n_classes

def compute_iou(preds, labels, num_classes):
    ious = []
    preds = preds.view(-1)
    labels = labels.view(-1)
    for cls in range(num_classes):
        pred_inds = preds == cls
        target_inds = labels == cls
        intersection = (pred_inds & target_inds).sum().item()
        union = (pred_inds | target_inds).sum().item()
        if union != 0:
            ious.append(intersection / union)
    return np.mean(ious) if ious else 0.0

def calculate_metric_percase(pred, gt):
    pred[pred > 0] = 1
    gt[gt > 0] = 1
    if pred.sum() > 0 and gt.sum() > 0:
        dice = metric.binary.dc(pred, gt)
        hd95 = metric.binary.hd95(pred, gt)
        return dice, hd95
    elif pred.sum() > 0 and gt.sum() == 0:
        return 1, 0
    else:
        return 0, 0

def save_overlay(image_slice, prediction_slice, save_dir, case_name, slice_idx=None):
    os.makedirs(save_dir, exist_ok=True)
    img = ((image_slice - image_slice.min()) / (image_slice.max() - image_slice.min()) * 255).astype(np.uint8)
    if img.ndim == 2:
        img = np.stack([img]*3, axis=2)
    overlay = img.copy()
    overlay[prediction_slice > 0, 0] = 255
    overlay[prediction_slice > 0, 1] = 0
    overlay[prediction_slice > 0, 2] = 0
    overlay_img = Image.fromarray(overlay)
    if slice_idx is not None:
        overlay_img.save(os.path.join(save_dir, f"{case_name}_slice{slice_idx}.png"))
    else:
        overlay_img.save(os.path.join(save_dir, f"{case_name}.png"))

def test_single_volume(image, label, net, classes, patch_size=[256,256],
                       test_save_path=None, case=None, z_spacing=1):
    """
    Handles 2D slices (H,W) or batched slices (C,H,W) from your CustomDataset.
    Returns list of (dice, hd95) for each class.
    """
    image = np.squeeze(image.cpu().detach().numpy())
    label = label.squeeze(0).cpu().detach().numpy()
    prediction = np.zeros_like(label, dtype=np.uint8)
    input_tensor = torch.from_numpy(image).unsqueeze(0).unsqueeze(0).float().cuda()
    net.eval()
    with torch.no_grad():
        out = torch.softmax(net(input_tensor), dim=1)
        out = torch.argmax(out, dim=1).squeeze(0).cpu().numpy()
        prediction = out.astype(np.uint8)

    metric_list = [calculate_metric_percase(prediction==i, label==i) for i in range(1, classes)]

    if test_save_path is not None and case is not None:
        img_itk = sitk.GetImageFromArray(image.astype(np.float32))
        prd_itk = sitk.GetImageFromArray(prediction.astype(np.float32))
        lab_itk = sitk.GetImageFromArray(label.astype(np.float32))
        img_itk.SetSpacing((1,1,z_spacing))
        prd_itk.SetSpacing((1,1,z_spacing))
        lab_itk.SetSpacing((1,1,z_spacing))
        sitk.WriteImage(prd_itk, os.path.join(test_save_path, f"{case}_pred.nii.gz"))
        sitk.WriteImage(img_itk, os.path.join(test_save_path, f"{case}_img.nii.gz"))
        sitk.WriteImage(lab_itk, os.path.join(test_save_path, f"{case}_gt.nii.gz"))
        save_overlay(image, prediction, test_save_path, case)

    return metric_list
