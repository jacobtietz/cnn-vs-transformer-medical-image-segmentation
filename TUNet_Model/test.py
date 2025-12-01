import argparse
import logging
import os
import random
import sys
import numpy as np
import torch
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from tqdm import tqdm

from datasets.dataset_synapse import CustomDataset, RandomGenerator
from utils import test_single_volume, compute_iou
from networks.vit_seg_modeling import VisionTransformer as ViT_seg
from networks.vit_seg_modeling import CONFIGS as CONFIGS_ViT_seg

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='CustomDataset')
parser.add_argument('--vit_name', type=str, default='R50+ViT-B_16')
parser.add_argument('--img_size', type=int, default=128)
parser.add_argument('--n_skip', type=int, default=3)
parser.add_argument('--vit_patches_size', type=int, default=16)
parser.add_argument('--num_classes', type=int, default=2)
parser.add_argument('--volume_path', type=str, default='./data')
parser.add_argument('--is_savenii', action='store_true')
parser.add_argument('--test_save_dir', type=str, default='./new_data')
parser.add_argument('--deterministic', type=int, default=1)
parser.add_argument('--seed', type=int, default=1234)
args = parser.parse_args()

def inference(args, model, test_save_path=None):
    transform = RandomGenerator(output_size=(args.img_size, args.img_size))
    db_test = args.Dataset(base_dir=args.volume_path, split='val', transform=transform)
    testloader = DataLoader(db_test, batch_size=1, shuffle=False, num_workers=1)

    logging.info(f"{len(testloader)} test iterations")
    model.eval()

    all_metrics = []
    all_ious = []

    for i_batch, sampled_batch in enumerate(tqdm(testloader, desc='Testing')):
        image, label, case_name = sampled_batch["image"], sampled_batch["label"], sampled_batch['case_name'][0]

        metric_i = test_single_volume(
            image, label, model, classes=args.num_classes,
            patch_size=[args.img_size, args.img_size],
            test_save_path=test_save_path, case=case_name, z_spacing=1
        )
        all_metrics.append(metric_i)

        with torch.no_grad():
            image_tensor = image.unsqueeze(0).float().cuda() if image.ndim == 3 else image.float().cuda()
            label_tensor = label.cuda()
            outputs = model(image_tensor)
            preds = torch.argmax(torch.softmax(outputs, dim=1), dim=1)
            iou = compute_iou(preds, label_tensor, args.num_classes)
            all_ious.append(iou)

        logging.info(
            f'Case {case_name} mean_dice {np.mean(metric_i, axis=0)[0]:.4f} '
            f'mean_hd95 {np.mean(metric_i, axis=0)[1]:.4f} IoU {iou:.4f}'
        )

    all_metrics = np.array(all_metrics)
    for cls in range(1, args.num_classes):
        cls_dice = np.mean(all_metrics[:, cls-1, 0])
        cls_hd95 = np.mean(all_metrics[:, cls-1, 1])
        logging.info(f'Mean class {cls} dice {cls_dice:.4f} hd95 {cls_hd95:.4f}')

    mean_dice = np.mean(all_metrics[:, :, 0])
    mean_hd95 = np.mean(all_metrics[:, :, 1])
    mean_iou = np.mean(all_ious)
    logging.info(f'Testing performance: mean_dice {mean_dice:.4f} mean_hd95 {mean_hd95:.4f} mean_iou {mean_iou:.4f}')
    return "Testing Finished!"

if __name__ == "__main__":
    cudnn.benchmark = not args.deterministic
    cudnn.deterministic = bool(args.deterministic)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    dataset_config = {'CustomDataset': {'Dataset': CustomDataset,
                                        'volume_path': args.volume_path,
                                        'num_classes': args.num_classes}}
    args.Dataset = dataset_config[args.dataset]['Dataset']
    args.volume_path = dataset_config[args.dataset]['volume_path']
    args.num_classes = dataset_config[args.dataset]['num_classes']

    snapshot_path = f"./model/TU_{args.dataset}{args.img_size}/TU_pretrain_{args.vit_name}_skip{args.n_skip}_10k_epo50_bs8_{args.img_size}"
    print("Resolved snapshot_path:", os.path.abspath(snapshot_path))

    checkpoint_file = os.path.join(snapshot_path, 'epoch_49.pth')
    if not os.path.exists(checkpoint_file):
        raise FileNotFoundError(f"No checkpoint found in {snapshot_path}")

    config_vit = CONFIGS_ViT_seg[args.vit_name]
    config_vit.n_classes = args.num_classes
    config_vit.n_skip = args.n_skip
    config_vit.patches.size = (args.vit_patches_size, args.vit_patches_size)
    if 'R50' in args.vit_name:
        config_vit.patches.grid = (args.img_size // args.vit_patches_size,
                                   args.img_size // args.vit_patches_size)

    net = ViT_seg(config_vit, img_size=args.img_size, num_classes=config_vit.n_classes).cuda()
    net.load_state_dict(torch.load(checkpoint_file))

    log_folder = f'./test_log/test_log_{args.dataset}{args.img_size}'
    os.makedirs(log_folder, exist_ok=True)
    logging.basicConfig(filename=os.path.join(log_folder, "log.txt"),
                        level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s',
                        datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))

    test_save_path = args.test_save_dir if args.is_savenii else None
    if test_save_path:
        os.makedirs(test_save_path, exist_ok=True)

    inference(args, net, test_save_path)
