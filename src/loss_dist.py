import random
from tqdm import tqdm
import numpy as np
from torch.utils.data import DataLoader
from config.config_args import *
from config.config_setup import get_net, get_dataset, init_seeds, get_optimizer_and_scheduler
from util.utils import *
import os
import sys
from monai.losses import *
from scipy.ndimage import distance_transform_edt


sys.path = list(dict.fromkeys(sys.path))
if os.getcwd() not in sys.path:
    sys.path.append(os.getcwd())
project_root = os.path.dirname(os.getcwd())
if project_root in sys.path:
    sys.path.remove(project_root)


def worker_init_fn(worker_id):
        random.seed(42 + worker_id)


def topology_aware_loss(gt, probs, centerline_mask, sigma=0.3, eps=1e-6):

    # sigma is been applied on normalize distances so the sigma sigma ∈ [0.1, 0.5]
    # where sigma in sigma ∈ [0, 1]. The value of sigma also depends on the size of the image

    # Ensure float
    gt = (gt > 0).float()
    prob = probs.float()
    centerline_mask = (centerline_mask > 0).float()

    with torch.no_grad():
        dist_list = []
        for b in range(centerline_mask.shape[0]):
            dist = distance_transform_edt(1 - centerline_mask[b].cpu().numpy())
            dist_list.append(torch.from_numpy(dist))
        dist_cent = torch.stack(dist_list).to(prob.device).float()

    # dist_cent = dist_cent / (dist_cent.max() + 1e-6)
    dist_cent = dist_cent / (dist_cent.view(dist_cent.shape[0], -1).max(dim=1)[0].view(-1,1,1) + 1e-6)
    weight = torch.exp(-dist_cent / sigma)
    weight = 0.5 + 0.5 * weight
    # weighted regression
    loss = ((prob - gt) ** 2 * weight).mean()
    return loss


def prototype_loss(gt_mask, feats, centerline_mask, tau=0.2):
    B, C, H, W = feats.shape
    feats = F.normalize(feats, dim=1)

    total_loss = 0.0

    for b in range(B):
        feat = feats[b].view(C, -1).T       # [N, C]
        label = gt_mask[b].view(-1)
        center_label = centerline_mask[b].view(-1)

        loss = 0.0
        count = 0

        classes = torch.unique(label)

        for cls in classes:
            if cls == 0:
                continue

            class_mask = (label == cls)
            if class_mask.sum() < 10:
                continue

            center_mask = (center_label > 0) & class_mask

            # fallback if centerline too small
            if center_mask.sum() < 5:
                proto = feat[class_mask].mean(dim=0)
            else:
                proto_center = feat[center_mask & class_mask].mean(dim=0)
                proto_region = feat[class_mask].mean(dim=0)

                proto = 0.7 * proto_center + 0.3 * proto_region

            proto = F.normalize(proto, dim=0)

            # similarity with all pixels
            sim = torch.matmul(feat, proto)   # [N]

            pos = (1 - sim[class_mask]).mean()
            neg_mask = (label != cls) & (label != 0)
            if neg_mask.sum() > 0:
                neg = F.relu(sim[neg_mask] - tau).mean()
            else:
                neg = 0.0

            loss += pos + neg
            count += 1

        if count > 0:
            total_loss += loss / count

    return total_loss / B



def train_net_sup(args, net, trainset, valset, save_cp=True):
    n_val, n_train = len(valset), len(trainset)
    logging.info("total frames is: {}".format(n_train))

    train_loader = DataLoader(trainset, batch_size=args.batch_size, pin_memory=False, 
                                      num_workers=args.num_workers, shuffle=True, drop_last=True, worker_init_fn=worker_init_fn)

    val_loader = DataLoader(valset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, drop_last=False)

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
    criterion = DiceCELoss(include_background=True, to_onehot_y=True, softmax=True, lambda_ce=1.0, lambda_dice=1.0)

    best_dice1 = 0
    for epoch in range(args.total_epoch):
        train_sup(args, train_loader, net, criterion, optimizer, epoch, scheduler)
        mean_dice, std_dice = validate_sup(args, net, val_loader, args.device)

        logging.info('')
        logging.info('Model, batch-wise validation Dice coeff: {}, std: {}'.format(mean_dice, std_dice))
        logging.info('===================================================================================')


        if save_cp and mean_dice > best_dice1:
            save_checkpoint(net, args.save_dir, epoch, best=True)
            best_dice1 = mean_dice

        torch.cuda.empty_cache()


def validate_sup(args, net, loader, device):
    dice_list = []
    net.eval()
    with torch.no_grad():
        with tqdm(total=len(loader), desc='Validation', unit='batch', leave=False) as pbar:
            for batch in loader:
                images = batch['image'].to(device=device, dtype=torch.float32)
                targets = batch['label'].to(device=device).long() 
                output, _ = net(images)            
                preds = torch.argmax(output, dim=1) 
                dice = dice_coefficient_multiclass_batch(preds, targets, args.num_classes, epsilon=1e-6)
                mean_dice = dice.mean()
                dice_list.append(mean_dice.item())
            pbar.close()
    mean_dice = np.mean(dice_list)
    std_dice = np.std(dice_list)
    return mean_dice, std_dice


def train_sup(args, train_loader, model, criterion, optimizer, epoch, scheduler):
    model.train()
    loss1_list_sup = []

    pbar = tqdm(total=len(train_loader))

    for batch in train_loader:
        input_labeled = batch['image'].to(device=args.device, dtype=torch.float32)
        target_labeled = batch['label'].to(device=args.device).long().unsqueeze(1) 
        centerline_mask = batch['centerline_mask'].to(device=args.device)

        output, feats = model(input_labeled)
        probs = torch.softmax(output, dim=1)
        vessel_prob = probs[:, 1:, :, :].sum(dim=1)
        
        if epoch < 5:
            lambda_cl = 0.0
        else:
            lambda_cl = min(0.05, (epoch - 5) / 20 * 0.05)

        loss = criterion(output, target_labeled)
        cl_loss = topology_aware_loss(gt=target_labeled.squeeze(1), probs=vessel_prob, centerline_mask=centerline_mask)
        proto_loss = prototype_loss(gt_mask=target_labeled.squeeze(1), feats=feats, centerline_mask=centerline_mask)
        loss = loss + lambda_cl * cl_loss + 0.05 * proto_loss
        # loss = loss + lambda_cl * cl_loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        loss1_list_sup.append(loss.item())
        pbar.update(1)

    logging.info('===================================================================================')
    logging.info('Epoch: {}, model1 supervised loss: {}'.format(epoch, np.mean(loss1_list_sup)))


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
