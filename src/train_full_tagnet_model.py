import random
from tqdm import tqdm
import numpy as np
from torch.utils.data import DataLoader
from config.config_args import *
from config.config_setup import get_net, get_dataset, init_seeds, get_optimizer_and_scheduler
from util.utils import *
import os
import sys
import logging
import torch
import torch.nn as nn

from monai.losses import DiceCELoss, DiceLoss
from util.train import save_feature_map



sys.path = list(dict.fromkeys(sys.path))
if os.getcwd() not in sys.path:
    sys.path.append(os.getcwd())
project_root = os.path.dirname(os.getcwd())
if project_root in sys.path:
    sys.path.remove(project_root)


def worker_init_fn(worker_id):
    random.seed(42 + worker_id)


def train_net_sup(args, net, trainset, valset, save_cp=True):
    n_val, n_train = len(valset), len(trainset)
    logging.info("total frames is: {}".format(n_train))

    train_loader = DataLoader(
        trainset,
        batch_size=args.batch_size,
        pin_memory=False,
        num_workers=args.num_workers,
        shuffle=True,
        drop_last=True,
        worker_init_fn=worker_init_fn
    )

    val_loader = DataLoader(
        valset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        drop_last=False
    )

    logging.info(f'''Starting training:
        Epochs:          {args.total_epoch}
        Batch size:      {args.batch_size}
        Learning rate:   {args.lr}
        Training size:   {n_train}
        Validation size: {n_val}
        Checkpoints:     {save_cp}
        Device:          {torch.device(f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu').type}
    ''')

    optimizer, scheduler = get_optimizer_and_scheduler(args, net)
    seg_criterion = DiceCELoss(
        include_background=True,
        to_onehot_y=True,
        softmax=True,
        lambda_ce=1.0,
        lambda_dice=1.0
    )
    skel_bce = nn.BCEWithLogitsLoss()
    skel_dice = DiceLoss(sigmoid=True)

    best_dice = 0
    for epoch in range(args.total_epoch):
        saved_feature_this_epoch = (epoch % 5 == 0)
        train_sup(args, train_loader, net, seg_criterion, skel_bce, skel_dice, optimizer, epoch, scheduler, saved_feature_this_epoch)
        mean_dice, std_dice = validate_sup(args, net, val_loader, args.device)

        logging.info('')
        logging.info('Model, batch-wise validation Dice coeff: {}, std: {}'.format(mean_dice, std_dice))
        logging.info('===================================================================================')

        if save_cp and mean_dice > best_dice:
            save_checkpoint(net, args.save_dir, epoch, best=True)
            best_dice = mean_dice

        torch.cuda.empty_cache()


def validate_sup(args, net, loader, device):
    dice_list = []
    net.eval()

    with torch.no_grad():
        with tqdm(total=len(loader), desc='Validation', unit='batch', leave=False) as pbar:
            for batch in loader:
                images = batch['image'].to(device=device, dtype=torch.float32)
                targets = batch['label'].to(device=device).long()

                output = net(images)
                seg_logits = output["seg_fine"]

                preds = torch.argmax(seg_logits, dim=1)
                dice = dice_coefficient_multiclass_batch(preds, targets, args.num_classes, epsilon=1e-6)
                mean_dice = dice.mean()
                dice_list.append(mean_dice.item())
                pbar.update(1)

    mean_dice = np.mean(dice_list)
    std_dice = np.std(dice_list)
    return mean_dice, std_dice


def train_sup(args, train_loader, model, seg_criterion, skel_bce, skel_dice, optimizer, epoch, scheduler, saved_feature_this_epoch):
    model.train()
    total_loss_list = []
    seg_coarse_loss_list = []
    seg_fine_loss_list = []
    skel_loss_list = []

    pbar = tqdm(total=len(train_loader), desc=f"Epoch {epoch + 1}/{args.total_epoch}")

    has_saved = False

    for batch_labeled in train_loader:
        input_labeled = batch_labeled['image'].to(device=args.device, dtype=torch.float32)
        target_labeled = batch_labeled['label'].to(device=args.device).long().unsqueeze(1)
        gt_skel = batch_labeled['centerline_mask'].to(device=args.device).float()

        if gt_skel.ndim == 3:
            gt_skel = gt_skel.unsqueeze(1)
        elif gt_skel.ndim == 4:
            pass
        else:
            raise ValueError("GT_skel mask has invalid dimension")

        output = model(input_labeled)

        seg_logits_coarse = output["seg_coarse"]
        seg_logits_fine = output["seg_fine"]
        skel_logits = output["skeleton"]
        fused_features = output["features"]
        refined_features = output["refined_features"]

        # save features map
        if saved_feature_this_epoch and not has_saved:
            save_feature_map(fused_features, args.save_dir, epoch, "coarse_features")
            save_feature_map(refined_features, args.save_dir, epoch, "refine_features")
            has_saved = True

        # compute the segmentation loss of the fine and coarse decoder features
        loss_seg_coarse = seg_criterion(seg_logits_coarse, target_labeled)
        loss_seg_fine = seg_criterion(seg_logits_fine, target_labeled)

        # Skeleton loss Original
        loss_skel = 0.5 * skel_bce(skel_logits, gt_skel) + 0.5 * skel_dice(skel_logits, gt_skel)
        loss = 0.3 * loss_seg_coarse + 1.0 * loss_seg_fine  + args.lambda_skel * loss_skel
    
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss_list.append(loss.item())
        seg_coarse_loss_list.append(loss_seg_coarse.item())
        seg_fine_loss_list.append(loss_seg_fine.item())
        skel_loss_list.append(loss_skel.item())

        pbar.update(1)

    logging.info('===================================================================================')
    logging.info(
        f"Epoch: {epoch}, total loss: {np.mean(total_loss_list):.6f}, "
        f"seg coarse loss: {np.mean(seg_coarse_loss_list):.6f}, "
        f"seg fine loss: {np.mean(seg_fine_loss_list):.6f}, "
        f"skel loss: {np.mean(skel_loss_list):.6f}"
    )

    pbar.close()

    checkpoint_dict = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
    }

    save_checkpoint(model, net_dict=checkpoint_dict, save_dir=args.save_dir, epoch=epoch, best=False)
    logging.info(f"Epoch {epoch + 1}, learning rate: {scheduler.get_last_lr()}")
    scheduler.step()


if __name__ == '__main__':
    init_seeds(42)
    args = parser.parse_args()
    setup_logging(args, mode='train')

    logging.info(os.path.dirname(os.path.abspath(__file__)))
    logging.info(args)

    assert args.json_path is not None, 'input your split file'

    train_set = get_dataset(args, mode='train', json=True)
    val_set = get_dataset(args, mode='val', json=True)

    net = get_net(args, net=args.net)

    logging.info('Models and datasets are loaded')
    logging.info('Training CAO full supervision...')
    train_net_sup(args, net=net, trainset=train_set, valset=val_set)