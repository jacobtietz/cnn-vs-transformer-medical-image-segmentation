import os
import glob
import random
import numpy as np
import torch
from scipy import ndimage
from scipy.ndimage import zoom
from torch.utils.data import Dataset
import cv2


def random_rot_flip(image, label):
    k = np.random.randint(0, 4)
    image = np.rot90(image, k)
    label = np.rot90(label, k)

    axis = np.random.randint(0, 2)
    image = np.flip(image, axis=axis).copy()
    label = np.flip(label, axis=axis).copy()
    return image, label


def random_rotate(image, label):
    angle = np.random.randint(-20, 20)
    image = ndimage.rotate(image, angle, order=0, reshape=False)
    label = ndimage.rotate(label, angle, order=0, reshape=False)
    return image, label


class RandomGenerator(object):
    def __init__(self, output_size):
        self.output_size = output_size

    def __call__(self, sample):
        image, label = sample['image'], sample['label']

        if random.random() > 0.5:
            image, label = random_rot_flip(image, label)
        elif random.random() > 0.5:
            image, label = random_rotate(image, label)

        # Resize
        x, y = image.shape[:2]
        if x != self.output_size[0] or y != self.output_size[1]:
            if image.ndim == 2:
                image = zoom(
                    image,
                    (self.output_size[0] / x, self.output_size[1] / y),
                    order=3
                )
            else:
                image = zoom(
                    image,
                    (self.output_size[0] / x, self.output_size[1] / y, 1),
                    order=3
                )
            label = zoom(
                label,
                (self.output_size[0] / x, self.output_size[1] / y),
                order=0
            )

        # To tensor
        if image.ndim == 2:
            image = torch.from_numpy(image.astype(np.float32)).unsqueeze(0)
        else:
            image = torch.from_numpy(image.astype(np.float32)).permute(2, 0, 1)

        # Ensure labels are 0 and 1
        label = torch.from_numpy(label.astype(np.int64))
        label[label > 0] = 1

        return {'image': image, 'label': label}


class CustomDataset(Dataset):
    def __init__(self, base_dir, split='train', transform=None):
        """
        Automatically load images and masks from folders.
        Expected folder structure:
            base_dir/
                train_images/
                train_masks/
                val_images/
                val_masks/
                test_images/
                test_masks/
        """
        self.transform = transform
        self.split = split

        if split == 'train':
            image_dir = os.path.join(base_dir, "train_images")
            mask_dir = os.path.join(base_dir, "train_masks")
        elif split == 'val':
            image_dir = os.path.join(base_dir, "val_images")
            mask_dir = os.path.join(base_dir, "val_masks")
        else:  # test
            image_dir = os.path.join(base_dir, "test_images")
            mask_dir = os.path.join(base_dir, "test_masks")

        self.image_list = sorted(glob.glob(os.path.join(image_dir, "*")))
        self.mask_list = sorted(glob.glob(os.path.join(mask_dir, "*")))

        assert len(self.image_list) == len(self.mask_list), \
            "Images and masks count must match!"

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        img_path = self.image_list[idx]
        mask_path = self.mask_list[idx]

        # Load image
        image = cv2.imread(img_path, cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # single channel

        # Load mask
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        mask = mask.astype(np.int64)
        mask[mask > 0] = 1

        sample = {'image': image, 'label': mask}

        if self.transform:
            sample = self.transform(sample)

        sample['case_name'] = os.path.basename(img_path).split('.')[0]
        return sample
