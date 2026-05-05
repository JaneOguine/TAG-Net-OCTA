import torch
import segmentation_models_pytorch as smp
import albumentations as A
from albumentations.core.composition import Compose
from dataset.OCTA_dataset import OCTA_dataset
import random, os
import numpy as np
import logging
from torch import optim
from models.TAGNet import *
import cv2

def get_net(args, pretrain=False, model=None, net=None):
    if torch.cuda.is_available():
        device = torch.device(f"cuda:{args.device}")  # NVIDIA GPU
        print(device)
    elif torch.backends.mps.is_available():
        device = torch.device("mps")  # Apple MPS (Mac M1/M2)
    else:
        device = torch.device("cpu")

    net_name = args.net
    logging.info(f'Using device {device}')
    logging.info(f'Building:  {net_name}')

    if net_name == 'unet':
        inference_mode = True if args.mode == 'test' else False
        net = smp.Unet(encoder_name='efficientnet-b3', 
                       encoder_weights='imagenet',
                       in_channels=args.in_channels, 
                       classes=args.out_channels)
        net.encoder.inference_mode=inference_mode 

    elif net_name == "unetpp":
        net = smp.UnetPlusPlus(
            encoder_name="efficientnet-b3",
            encoder_weights="imagenet",
            in_channels=3,
            classes=args.num_classes
        )

    elif net_name == "deeplabv3+":
        net = smp.DeepLabV3Plus(
            encoder_name="efficientnet-b3",
            encoder_weights="imagenet",
            in_channels=3,
            classes=args.num_classes
        )

    elif net_name == "segformer":
        net = smp.Segformer(
            encoder_name="efficientnet-b3",
            encoder_weights="imagenet",
            in_channels=3,
            classes=args.num_classes
        )

    
    elif net_name == "full_model":
        net = TAGNet(
            architecture=args.backbone_name,
            encoder_name="efficientnet-b3",
            in_channels=args.in_channels,
            classes=args.num_classes,
            encoder_weights="imagenet",
        )
    
    elif net_name == "topo":
        net = TAGNet_TopologyOnly(
            architecture=args.backbone_name,
            encoder_name="efficientnet-b3",
            in_channels=args.in_channels,
            classes=args.num_classes,
            encoder_weights="imagenet",
        )

    elif net_name == "fusenet":
        net = TAGNet_TopologyAndFusion(
            architecture=args.backbone_name,
            encoder_name="efficientnet-b3",
            in_channels=args.in_channels,
            classes=args.num_classes,
            encoder_weights="imagenet",
        )

    else:
        raise ValueError(f"Unknown network: {net_name}")
    
    
    total_params = sum(p.numel() for p in net.parameters() if p.requires_grad)
    logging.info(f"Total trainable parameters: {total_params}")

    if pretrain:
        pretrain_path = os.path.join(args.save_dir, 'cp', 'best_net.pth')
        net.load_state_dict(torch.load(pretrain_path, map_location=device))
        logging.info(f'Model{model}  loaded from {pretrain_path}')

    net.to(device=device)
    return net


def get_optimizer_and_scheduler(args, net):
    params = filter(lambda p: p.requires_grad, net.parameters())  # added from
    optimizer = optim.AdamW(params, lr=args.lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.total_epoch, eta_min=args.min_lr)
    return optimizer, scheduler


def get_dataset(args, mode=None, json=True):

    if mode is None:
        raise ValueError("mode must be specified")

    # Transform for OCTA500_3mm (PAD only, no resize distortion)
    if mode == "train":
        transform_pad = Compose([
            A.PadIfNeeded(min_height=args.height, min_width=args.width,
                          border_mode=cv2.BORDER_CONSTANT,
                          value=0, mask_value=0),
            A.RandomRotate90(),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.Affine(
                scale=(0.97, 1.03),
                translate_percent=(0.0, 0.03),
                rotate=(-7, 7),
                interpolation=cv2.INTER_LINEAR,
                mask_interpolation=cv2.INTER_NEAREST,
                p=0.2
            ),
            # VERY mild intensity
            A.RandomGamma(gamma_limit=(90, 110), p=0.15),
            A.Normalize(mean=(0.5,), std=(0.5,))
        ], seed=42)

        # Transform for OCTA500_6mm and ROSSA (resize to args size)
        transform_resize = Compose([
            A.Resize(args.height, args.width,
                     interpolation=cv2.INTER_LINEAR,
                     mask_interpolation=cv2.INTER_NEAREST),
            A.RandomRotate90(),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.Affine(scale=(0.97, 1.03), translate_percent=(0.0, 0.03),
                rotate=(-7, 7), interpolation=cv2.INTER_LINEAR,
                mask_interpolation=cv2.INTER_NEAREST,
                p=0.2),
            # VERY mild intensity
            A.RandomGamma(gamma_limit=(90, 110), p=0.15),
            A.Normalize(mean=(0.5,), std=(0.5,))
        ], seed=42)

    else:
        transform_pad = Compose([
            A.PadIfNeeded(min_height=args.height, min_width=args.width, border_mode=cv2.BORDER_CONSTANT,value=0, mask_value=0),
            A.Normalize(mean=(0.5,), std=(0.5,))], seed=42)

        transform_resize = Compose([
            A.Resize(args.height, args.width, interpolation=cv2.INTER_LINEAR, mask_interpolation=cv2.INTER_NEAREST),
            A.Normalize(mean=(0.5,), std=(0.5,))], seed=42)

    logging.info(f"Transform PAD: {transform_pad}")
    logging.info(f"Transform RESIZE: {transform_resize}")
    dataset = OCTA_dataset(args, mode=mode, transform_pad=transform_pad, transform_resize=transform_resize)
    return dataset


def init_seeds(seed=42, cuda_deterministic=True):
    from torch.backends import cudnn
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # Speed-reproducibility tradeoff https://pytorch.org/docs/stable/notes/randomness.html
    if cuda_deterministic:  # slower, more reproducible
        cudnn.deterministic = True
        cudnn.benchmark = False
    else:  # faster, less reproducible
        cudnn.deterministic = False
        cudnn.benchmark = True
