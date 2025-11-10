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

# Import your custom dataset and transforms
from datasets.dataset_synapse import CustomDataset, RandomGenerator
from utils import test_single_volume
from networks.vit_seg_modeling import VisionTransformer as ViT_seg
from networks.vit_seg_modeling import CONFIGS as CONFIGS_ViT_seg

# ---------------------------
# Parser
# ---------------------------
parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='CustomDataset', help='dataset name')
parser.add_argument('--vit_name', type=str, default='R50+ViT-B_16', help='vit model name')
parser.add_argument('--img_size', type=int, default=128, help='input image size')
parser.add_argument('--n_skip', type=int, default=3, help='number of skip connections')
parser.add_argument('--vit_patches_size', type=int, default=16, help='patch size for ViT')
parser.add_argument('--num_classes', type=int, default=2, help='number of output classes')
parser.add_argument('--volume_path', type=str, default='./data', help='root dir for validation data')
parser.add_argument('--is_savenii', action='store_true', help='save predicted masks')
parser.add_argument('--test_save_dir', type=str, default='./new_data', help='dir to save predictions')
parser.add_argument('--deterministic', type=int, default=1, help='deterministic test')
parser.add_argument('--seed', type=int, default=1234, help='random seed')
args = parser.parse_args()

# ---------------------------
# Inference function
# ---------------------------
def inference(args, model, test_save_path=None):
    transform = RandomGenerator(output_size=(args.img_size, args.img_size))
    db_test = args.Dataset(base_dir=args.volume_path, split='val', transform=transform)
    testloader = DataLoader(db_test, batch_size=1, shuffle=False, num_workers=1)

    logging.info("{} test iterations".format(len(testloader)))
    model.eval()
    metric_list = 0.0

    for i_batch, sampled_batch in tqdm(enumerate(testloader)):
        image, label, case_name = sampled_batch["image"], sampled_batch["label"], sampled_batch['case_name'][0]

        # Ensure image shape is (C, H, W) for 2D slices
        image = image.squeeze(0)
        if image.ndim == 2:
            image = image[np.newaxis, :, :]

        metric_i = test_single_volume(
            image, label, model, classes=args.num_classes,
            patch_size=[args.img_size, args.img_size],
            test_save_path=test_save_path, case=case_name, z_spacing=1
        )
        metric_list += np.array(metric_i)
        logging.info('idx %d case %s mean_dice %f mean_hd95 %f' %
                     (i_batch, case_name, np.mean(metric_i, axis=0)[0], np.mean(metric_i, axis=0)[1]))

    metric_list = metric_list / len(db_test)
    for i in range(1, args.num_classes):
        logging.info('Mean class %d mean_dice %f mean_hd95 %f' %
                     (i, metric_list[i-1][0], metric_list[i-1][1]))
    performance = np.mean(metric_list, axis=0)[0]
    mean_hd95 = np.mean(metric_list, axis=0)[1]
    logging.info('Testing performance: mean_dice: %f mean_hd95: %f' % (performance, mean_hd95))
    return "Testing Finished!"

# ---------------------------
# Main
# ---------------------------
if __name__ == "__main__":
    if not args.deterministic:
        cudnn.benchmark = True
        cudnn.deterministic = False
    else:
        cudnn.benchmark = False
        cudnn.deterministic = True

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    dataset_config = {
        'CustomDataset': {
            'Dataset': CustomDataset,
            'volume_path': args.volume_path,
            'num_classes': args.num_classes
        }
    }
    dataset_name = args.dataset
    args.Dataset = dataset_config[dataset_name]['Dataset']
    args.volume_path = dataset_config[dataset_name]['volume_path']
    args.num_classes = dataset_config[dataset_name]['num_classes']
    args.is_pretrain = True

    # Snapshot path for your trained model
    args.exp = 'TU_' + dataset_name + str(args.img_size)
    snapshot_path = f"./model/{args.exp}/TU_pretrain_{args.vit_name}_skip{args.n_skip}_10k_epo1_bs8_{args.img_size}"
    checkpoint_file = os.path.join(snapshot_path, 'epoch_0.pth')
    if not os.path.exists(checkpoint_file):
        raise FileNotFoundError(f"No checkpoint found in {snapshot_path}")

    # Load model
    config_vit = CONFIGS_ViT_seg[args.vit_name]
    config_vit.n_classes = args.num_classes
    config_vit.n_skip = args.n_skip
    config_vit.patches.size = (args.vit_patches_size, args.vit_patches_size)
    if args.vit_name.find('R50') != -1:
        config_vit.patches.grid = (int(args.img_size / args.vit_patches_size),
                                   int(args.img_size / args.vit_patches_size))
    net = ViT_seg(config_vit, img_size=args.img_size, num_classes=config_vit.n_classes).cuda()
    net.load_state_dict(torch.load(checkpoint_file))

    snapshot_name = os.path.basename(snapshot_path)
    log_folder = './test_log/test_log_' + args.exp
    os.makedirs(log_folder, exist_ok=True)
    logging.basicConfig(filename=log_folder + '/'+snapshot_name+".txt",
                        level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s',
                        datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))
    logging.info(snapshot_name)

    # Save predictions to relative new_data folder
    if args.is_savenii:
        test_save_path = "./new_data"
        os.makedirs(test_save_path, exist_ok=True)
    else:
        test_save_path = None

    inference(args, net, test_save_path)
