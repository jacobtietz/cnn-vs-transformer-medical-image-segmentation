import argparse
import logging
import os
import random
import numpy as np
import torch
import torch.backends.cudnn as cudnn
from networks.vit_seg_modeling import VisionTransformer as ViT_seg
from networks.vit_seg_modeling import CONFIGS as CONFIGS_ViT_seg
from trainer import trainer_synapse

parser = argparse.ArgumentParser()
parser.add_argument('--root_path', type=str, default='./data')
parser.add_argument('--dataset', type=str, default='CustomDataset')
parser.add_argument('--list_dir', type=str, default=None)
parser.add_argument('--num_classes', type=int, default=2)
parser.add_argument('--max_iterations', type=int, default=1000)
parser.add_argument('--max_epochs', type=int, default=50)
parser.add_argument('--batch_size', type=int, default=8)
parser.add_argument('--n_gpu', type=int, default=1)
parser.add_argument('--deterministic', type=int, default=1)
parser.add_argument('--base_lr', type=float, default=0.01)
parser.add_argument('--img_size', type=int, default=128)
parser.add_argument('--seed', type=int, default=1234)
parser.add_argument('--n_skip', type=int, default=3)
parser.add_argument('--vit_name', type=str, default='R50+ViT-B_16')
parser.add_argument('--vit_patches_size', type=int, default=16)
args = parser.parse_args()

if __name__ == "__main__":
    cudnn.benchmark = not args.deterministic
    cudnn.deterministic = bool(args.deterministic)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    dataset_name = args.dataset
    dataset_config = {
        'CustomDataset': {
            'root_path': './data',
            'list_dir': None,
            'num_classes': args.num_classes,
        },
    }

    args.num_classes = dataset_config[dataset_name]['num_classes']
    args.root_path = dataset_config[dataset_name]['root_path']
    args.list_dir = dataset_config[dataset_name]['list_dir']
    args.is_pretrain = True
    args.exp = f'TU_{dataset_name}{args.img_size}'

    snapshot_path = f"./model/{args.exp}/TU"
    if args.is_pretrain:
        snapshot_path += "_pretrain"
    snapshot_path += f"_{args.vit_name}_skip{args.n_skip}"
    if args.vit_patches_size != 16:
        snapshot_path += f"_vitpatch{args.vit_patches_size}"
    if args.max_iterations != 30000:
        snapshot_path += f"_{str(args.max_iterations)[:2]}k"
    if args.max_epochs != 30:
        snapshot_path += f"_epo{args.max_epochs}"
    snapshot_path += f"_bs{args.batch_size}"
    if args.base_lr != 0.01:
        snapshot_path += f"_lr{args.base_lr}"
    snapshot_path += f"_{args.img_size}"
    if args.seed != 1234:
        snapshot_path += f"_s{args.seed}"

    os.makedirs(snapshot_path, exist_ok=True)

    config_vit = CONFIGS_ViT_seg[args.vit_name]
    config_vit.n_classes = args.num_classes
    config_vit.n_skip = args.n_skip
    if 'R50' in args.vit_name:
        config_vit.patches.grid = (args.img_size // args.vit_patches_size,
                                   args.img_size // args.vit_patches_size)

    net = ViT_seg(config_vit, img_size=args.img_size, num_classes=config_vit.n_classes).cuda()
    net.load_from(weights=np.load(config_vit.pretrained_path))

    trainer = {'CustomDataset': trainer_synapse}
    trainer[dataset_name](args, net, snapshot_path)
