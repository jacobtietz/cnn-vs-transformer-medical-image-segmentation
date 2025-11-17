import argparse
import logging
import os
import random
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.modules.loss import CrossEntropyLoss
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
from tqdm import tqdm
from utils import DiceLoss, compute_iou
from torchvision import transforms
from datasets.dataset_synapse import CustomDataset, RandomGenerator


def trainer_synapse(args, model, snapshot_path):
    logging.basicConfig(
        filename=os.path.join(snapshot_path, "log.txt"),
        level=logging.INFO,
        format='[%(asctime)s.%(msecs)03d] %(message)s',
        datefmt='%H:%M:%S'
    )
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))

    base_lr = args.base_lr
    num_classes = args.num_classes
    batch_size = args.batch_size * args.n_gpu

    # Datasets
    db_train = CustomDataset(base_dir=args.root_path, split="train",
                             transform=transforms.Compose([RandomGenerator([args.img_size, args.img_size])]))
    db_val = CustomDataset(base_dir=args.root_path, split="val",
                           transform=transforms.Compose([RandomGenerator([args.img_size, args.img_size])]))
    print(f"Train set: {len(db_train)}, Val set: {len(db_val)}")

    def worker_init_fn(worker_id):
        random.seed(args.seed + worker_id)
        np.random.seed(args.seed + worker_id)

    trainloader = DataLoader(db_train, batch_size=batch_size, shuffle=True, num_workers=0,
                             pin_memory=True, worker_init_fn=worker_init_fn)
    valloader = DataLoader(db_val, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)

    if args.n_gpu > 1:
        model = nn.DataParallel(model)
    model.train()

    ce_loss = CrossEntropyLoss()
    dice_loss = DiceLoss(num_classes)
    optimizer = optim.SGD(model.parameters(), lr=base_lr, momentum=0.9, weight_decay=0.0001)
    writer = SummaryWriter(os.path.join(snapshot_path, 'log'))

    iter_num = 0
    max_epoch = args.max_epochs
    max_iterations = max_epoch * len(trainloader)
    logging.info(f"{len(trainloader)} iterations per epoch. {max_iterations} max iterations")

    epoch_iterator = tqdm(range(max_epoch), ncols=70, desc='Epochs')
    for epoch_num in epoch_iterator:
        # Training metrics
        train_loss_epoch = 0.0
        train_dice_epoch = 0.0
        train_iou_epoch = 0.0

        batch_iterator = tqdm(trainloader, ncols=70, desc='Train', leave=False)
        for i_batch, sampled_batch in enumerate(batch_iterator):
            image_batch, label_batch = sampled_batch['image'], sampled_batch['label']
            image_batch, label_batch = image_batch.cuda(), label_batch.cuda()

            outputs = model(image_batch)
            loss_ce = ce_loss(outputs, label_batch.long())
            loss_dice_val = dice_loss(outputs, label_batch, softmax=True)
            loss = 0.5 * loss_ce + 0.5 * loss_dice_val

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            lr_ = base_lr * (1 - iter_num / max_iterations) ** 0.9
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr_

            iter_num += 1
            train_loss_epoch += loss.item()
            train_dice_epoch += loss_dice_val.item()
            preds = torch.argmax(torch.softmax(outputs, dim=1), dim=1)
            train_iou_epoch += compute_iou(preds, label_batch, num_classes)

            # TensorBoard
            writer.add_scalar('train/lr', lr_, iter_num)
            writer.add_scalar('train/loss', loss.item(), iter_num)
            writer.add_scalar('train/loss_ce', loss_ce.item(), iter_num)

            batch_iterator.set_postfix({'loss': loss.item(), 'lr': lr_})

        # Average metrics
        train_loss_epoch /= len(trainloader)
        train_dice_epoch /= len(trainloader)
        train_iou_epoch /= len(trainloader)
        logging.info(f"Epoch {epoch_num} Train Loss: {train_loss_epoch:.4f}, Dice: {train_dice_epoch:.4f}, IoU: {train_iou_epoch:.4f}")

        # Validation metrics
        model.eval()
        val_loss_epoch, val_dice_epoch, val_iou_epoch = 0.0, 0.0, 0.0
        with torch.no_grad():
            for sampled_batch in tqdm(valloader, ncols=70, desc='Val', leave=False):
                image_batch, label_batch = sampled_batch['image'], sampled_batch['label']
                image_batch, label_batch = image_batch.cuda(), label_batch.cuda()

                outputs = model(image_batch)
                loss_ce = ce_loss(outputs, label_batch.long())
                loss_dice_val = dice_loss(outputs, label_batch, softmax=True)
                loss = 0.5 * loss_ce + 0.5 * loss_dice_val

                preds = torch.argmax(torch.softmax(outputs, dim=1), dim=1)
                val_loss_epoch += loss.item()
                val_dice_epoch += loss_dice_val.item()
                val_iou_epoch += compute_iou(preds, label_batch, num_classes)

        val_loss_epoch /= len(valloader)
        val_dice_epoch /= len(valloader)
        val_iou_epoch /= len(valloader)
        logging.info(f"Epoch {epoch_num} Val Loss: {val_loss_epoch:.4f}, Dice: {val_dice_epoch:.4f}, IoU: {val_iou_epoch:.4f}")

        # TensorBoard logging
        writer.add_scalar('val/loss', val_loss_epoch, epoch_num)
        writer.add_scalar('val/dice', val_dice_epoch, epoch_num)
        writer.add_scalar('val/iou', val_iou_epoch, epoch_num)

        # Update epoch tqdm
        epoch_iterator.set_postfix({
            'Train Loss': f"{train_loss_epoch:.4f}",
            'Train Dice': f"{train_dice_epoch:.4f}",
            'Train IoU': f"{train_iou_epoch:.4f}",
            'Val Loss': f"{val_loss_epoch:.4f}",
            'Val Dice': f"{val_dice_epoch:.4f}",
            'Val IoU': f"{val_iou_epoch:.4f}"
        })

        # Save model
        save_interval = 50
        if epoch_num > max_epoch // 2 and (epoch_num + 1) % save_interval == 0:
            save_path = os.path.join(snapshot_path, f'epoch_{epoch_num}.pth')
            torch.save(model.state_dict(), save_path)
            logging.info(f"Saved model to {save_path}")

        if epoch_num >= max_epoch - 1:
            save_path = os.path.join(snapshot_path, f'epoch_{epoch_num}.pth')
            torch.save(model.state_dict(), save_path)
            logging.info(f"Saved model to {save_path}")
            epoch_iterator.close()
            break

        model.train()

    writer.close()
    return "Training Finished!"
