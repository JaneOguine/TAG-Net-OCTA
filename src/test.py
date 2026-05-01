import os.path
import os
import sys
import logging

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
import cv2
from skimage.morphology import skeletonize

from util.utils import *
from util.metrics import *
from config.config_args import *
from config.config_setup import get_net, init_seeds, get_dataset
import csv


sys.path = list(dict.fromkeys(sys.path))
if os.getcwd() not in sys.path:
    sys.path.append(os.getcwd())
project_root = os.path.dirname(os.getcwd())
if project_root in sys.path:
    sys.path.remove(project_root)


def validate_baseline(args, net, loader, save_results_dir=None):

    net.eval()

    overall_dice = []
    vessel_overall_dice = []
    binary_dice = []
    thin_dice = []
    large_dice = []

    precision_list = []
    recall_list = []
    accuracy_list = []
    specificity_list = []

    cldice_binary = []
    cldice_thin = []
    cldice_large = []
    cldice_overall = []

    grouped = {}

    with torch.no_grad():
        for batch in loader:

            images = batch["image"].to(args.device, dtype=torch.float32)
            target = batch["label"].to(args.device, dtype=torch.long)

            # seg_logits, feat = net(images)
            seg_logits = net(images)
            # seg_logits = output["seg"] 
            probs = F.softmax(seg_logits, dim=1)
            preds = torch.argmax(probs, dim=1)

            pred = preds[0].cpu()
            gt = target[0].cpu()

            dt = batch["data_type"][0]
            ot = batch["octa_type"][0]
            key = f"{dt}_{ot}"

            
            if key not in grouped:
                grouped[key] = {
                    "overall": [],
                    "vessel_overall": [],
                    "binary": [],
                    "thin": [],
                    "large": [],
                    "prec": [],
                    "rec": [],
                    "acc": [],
                    "spec": [],
                    "cl_bin": [],
                    "cl_thin": [],
                    "cl_large": [],
                    "cl_overall": [],
                    "names": []
                }

         
            bin_pred = (pred > 0).float()
            bin_gt = (gt > 0).float()

            bin_d = dice_binary(bin_pred, bin_gt).item()
            prec = precision_score(bin_pred, bin_gt).item()
            rec = recall_score(bin_pred, bin_gt).item()
            acc = accuracy_score(bin_pred, bin_gt).item()
            spec = specificity_score(bin_pred, bin_gt).item()
            cl_bin = cldice_score(bin_pred.numpy().astype(np.uint8), bin_gt.numpy().astype(np.uint8))

            thin_gt = (gt == 1).float()
            thin_pred = (pred == 1).float()

            if thin_gt.sum() == 0:
                thin_d = np.nan
                cl_thin = np.nan
            else:
                thin_d = dice_binary(thin_pred, thin_gt).item()
                cl_thin = cldice_score(thin_pred.numpy().astype(np.uint8), thin_gt.numpy().astype(np.uint8))

            large_gt = (gt == 2).float()
            large_pred = (pred == 2).float()

            if large_gt.sum() == 0:
                large_d = np.nan
                cl_large = np.nan
            else:
                large_d = dice_binary(large_pred, large_gt).item()
                cl_large = cldice_score(large_pred.numpy().astype(np.uint8), large_gt.numpy().astype(np.uint8))

            per_class = dice_coefficient_multiclass_batch(pred.unsqueeze(0),gt.unsqueeze(0),
                                                          args.num_classes, epsilon=1e-6)

            per_class = np.array(per_class).reshape(-1)

            valid_classes = [per_class[0]]
            if thin_gt.sum() > 0:
                valid_classes.append(per_class[1])
            if large_gt.sum() > 0:
                valid_classes.append(per_class[2])

            overall_mean_dice = float(np.mean(valid_classes))

            valid_fg = []
            if thin_gt.sum() > 0:
                valid_fg.append(per_class[1])
            if large_gt.sum() > 0:
                valid_fg.append(per_class[2])

            mean_dice = float(np.mean(valid_fg)) if len(valid_fg) > 0 else np.nan

            cl_overall_list = []
            if not np.isnan(cl_thin):
                cl_overall_list.append(cl_thin)
            if not np.isnan(cl_large):
                cl_overall_list.append(cl_large)

            cl_overall = float(np.mean(cl_overall_list)) if len(cl_overall_list) > 0 else np.nan

            overall_dice.append(overall_mean_dice)
            vessel_overall_dice.append(mean_dice)
            binary_dice.append(bin_d)
            thin_dice.append(thin_d)
            large_dice.append(large_d)

            precision_list.append(prec)
            recall_list.append(rec)
            accuracy_list.append(acc)
            specificity_list.append(spec)

            cldice_binary.append(cl_bin)
            cldice_thin.append(cl_thin)
            cldice_large.append(cl_large)
            cldice_overall.append(cl_overall)

            # ---------------- Store grouped ----------------
            grouped[key]["overall"].append(overall_mean_dice)
            grouped[key]["vessel_overall"].append(mean_dice)
            grouped[key]["binary"].append(bin_d)
            grouped[key]["thin"].append(thin_d)
            grouped[key]["large"].append(large_d)
            grouped[key]["prec"].append(prec)
            grouped[key]["rec"].append(rec)
            grouped[key]["acc"].append(acc)
            grouped[key]["spec"].append(spec)
            grouped[key]["cl_bin"].append(cl_bin)
            grouped[key]["cl_thin"].append(cl_thin)
            grouped[key]["cl_large"].append(cl_large)
            grouped[key]["cl_overall"].append(cl_overall)

            # ---------------- SAVE RESULTS ----------------
            if args.save_results:               
                path = os.path.split(args.save_results_dir.rstrip("/"))[-1]
                overlay_dir = os.path.join("./all_results/overlay_results/", path)
                fp_pt = os.path.join("./all_results/fp", path)
                fn_pt = os.path.join("./all_results/fn", path)

                os.makedirs(overlay_dir, exist_ok=True)
                os.makedirs(fp_pt, exist_ok=True)
                os.makedirs(fn_pt, exist_ok=True)
                os.makedirs(args.save_results_dir, exist_ok=True)

                frame_name = os.path.basename(batch["name"][0])
                base = os.path.splitext(frame_name)[0]

                h0, w0 = batch["label"].shape[-2:]

                # ---- Initialize output image ----
                pred_classes = np.zeros((h0, w0), dtype=np.uint8)

                # ---- Convert tensors to numpy ----
                thin_np = thin_pred.cpu().numpy()
                large_np = large_pred.cpu().numpy()

                # ---- Assign class intensities ----
                pred_classes[thin_np == 1] = 127      # thin vessels
                pred_classes[large_np == 1] = 255     # large vessels (FIXED)

                # ---- Binary masks for evaluation ----
                pred_np = (bin_pred.cpu().numpy() * 255).astype(np.uint8)
                gt_np   = (bin_gt.cpu().numpy() * 255).astype(np.uint8)

                # ---- Resize if needed ----
                if pred_np.shape != (h0, w0):
                    pred_np = cv2.resize(pred_np, (w0, h0), interpolation=cv2.INTER_NEAREST)
                    gt_np   = cv2.resize(gt_np, (w0, h0), interpolation=cv2.INTER_NEAREST)

                # ---- TP / FP / FN visualization ----
                vis, fp, fn = tpfpfn_mask_only(gt_np, pred_np)

                # ---- Save outputs ----
                cv2.imwrite(os.path.join(overlay_dir, f"{base}.png"), vis)
                cv2.imwrite(os.path.join(fp_pt, f"{base}.png"), fp)
                cv2.imwrite(os.path.join(fn_pt, f"{base}.png"), fn)
                cv2.imwrite(os.path.join(args.save_results_dir, f"{base}.png"), pred_classes)

            frame_name = os.path.basename(batch["name"][0])
            grouped[key]["names"].append(frame_name)

    save_per_sample_dice_csv(args, grouped)

    return (
        overall_dice,
        vessel_overall_dice,
        binary_dice,
        thin_dice,
        large_dice,
        precision_list,
        recall_list,
        accuracy_list,
        specificity_list,
        cldice_binary,
        cldice_thin,
        cldice_large,
        cldice_overall,
        grouped
    )


def test_net_baseline(args, net1, dataset, batch_size=1):
    test_loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    logging.info(
        f"""Starting testing:
            Num test:        {len(dataset)}
            Batch size:      {batch_size}
            Device:          {torch.device(f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu')}
        """)

    net1.eval()

    (overall_dice,
    vessel_overall_dice,
    binary_dice,
    thin_dice,
    large_dice,
    precision_list,
    recall_list,
    accuracy_list,
    specificity_list,
    cl_bin, cl_thin, cl_large, cldice_overall, grouped) = validate_baseline(args, net1, test_loader, save_results_dir=args.save_results_dir)


    # ---- OVERALL ----
    logging.info("========================================")
    logging.info("Model Evaluation Results (Sample-wise) - OVERALL:")
    logging.info("Dice (Overall-all_classes):    {:.4f} ± {:.4f}".format(safe_mean(overall_dice), safe_std(overall_dice)))
    logging.info("Dice (Vessel_only):    {:.4f} ± {:.4f}".format(safe_mean(vessel_overall_dice), safe_std(vessel_overall_dice)))
    logging.info("Dice (Binary):     {:.4f} ± {:.4f}".format(safe_mean(binary_dice), safe_std(binary_dice)))
    logging.info("Dice (Thin):       {:.4f} ± {:.4f}".format(safe_mean(thin_dice), safe_std(thin_dice)))
    logging.info("Dice (Large):      {:.4f} ± {:.4f}".format(safe_mean(large_dice), safe_std(large_dice)))
    logging.info("clDice (Overall): {:.4f} ± {:.4f}".format(safe_mean(cldice_overall), safe_std(cldice_overall)))
    logging.info("clDice (Binary):   {:.4f} ± {:.4f}".format(safe_mean(cl_bin), safe_std(cl_bin)))
    logging.info("clDice (Thin):     {:.4f} ± {:.4f}".format(safe_mean(cl_thin), safe_std(cl_thin)))
    logging.info("clDice (Large):    {:.4f} ± {:.4f}".format(safe_mean(cl_large), safe_std(cl_large)))
    logging.info("Precision:         {:.4f} ± {:.4f}".format(safe_mean(precision_list), safe_std(precision_list)))
    logging.info("Specificity:      {:.4f} ± {:.4f}".format(safe_mean(specificity_list), safe_std(specificity_list)))
    logging.info("Recall:            {:.4f} ± {:.4f}".format(safe_mean(recall_list), safe_std(recall_list)))
    logging.info("Accuracy:          {:.4f} ± {:.4f}".format(safe_mean(accuracy_list), safe_std(accuracy_list)))
    logging.info("========================================")


    # ---- PER DATASET ----
    for group_key in grouped.keys():

        if len(grouped[group_key]["overall"]) == 0:
            continue

        logging.info("========================================")
        logging.info(f"Model Evaluation Results (Sample-wise) - {group_key}:")
        logging.info("Dice (Overall):    {:.4f} ± {:.4f}".format(safe_mean(grouped[group_key]["overall"]), safe_std(grouped[group_key]["overall"])))
        logging.info("Dice (Vessel_only):    {:.4f} ± {:.4f}".format(safe_mean(grouped[group_key]["vessel_overall"]), safe_std(grouped[group_key]["vessel_overall"])))
        logging.info("Dice (Binary):     {:.4f} ± {:.4f}".format(safe_mean(grouped[group_key]["binary"]), safe_std(grouped[group_key]["binary"])))
        logging.info("Dice (Thin):       {:.4f} ± {:.4f}".format(safe_mean(grouped[group_key]["thin"]), safe_std(grouped[group_key]["thin"])))
        logging.info("Dice (Large):      {:.4f} ± {:.4f}".format(safe_mean(grouped[group_key]["large"]), safe_std(grouped[group_key]["large"])))
        logging.info("clDice (Overall):   {:.4f} ± {:.4f}".format(safe_mean(grouped[group_key]["cl_overall"]), safe_std(grouped[group_key]["cl_overall"])))
        logging.info("clDice (Binary):   {:.4f} ± {:.4f}".format(safe_mean(grouped[group_key]["cl_bin"]), safe_std(grouped[group_key]["cl_bin"])))
        logging.info("clDice (Thin):     {:.4f} ± {:.4f}".format(safe_mean(grouped[group_key]["cl_thin"]), safe_std(grouped[group_key]["cl_thin"])))
        logging.info("clDice (Large):    {:.4f} ± {:.4f}".format(safe_mean(grouped[group_key]["cl_large"]), safe_std(grouped[group_key]["cl_large"])))
        logging.info("Precision:         {:.4f} ± {:.4f}".format(safe_mean(grouped[group_key]["prec"]), safe_std(grouped[group_key]["prec"])))
        logging.info("Specificity:         {:.4f} ± {:.4f}".format(safe_mean(grouped[group_key]["spec"]), safe_std(grouped[group_key]["spec"])))
        logging.info("Recall:            {:.4f} ± {:.4f}".format(safe_mean(grouped[group_key]["rec"]), safe_std(grouped[group_key]["rec"])))
        logging.info("Accuracy:          {:.4f} ± {:.4f}".format(safe_mean(grouped[group_key]["acc"]), safe_std(grouped[group_key]["acc"])))
        logging.info("========================================")


if __name__ == "__main__":
    init_seeds(42)
    args = parser.parse_args()
    setup_logging(args, mode="test")
    logging.info(os.path.abspath(__file__))
    logging.info(args)

    dataset = get_dataset(args, mode="test")
    net = get_net(args, pretrain=True)

    test_net_baseline(
        args,
        net1=net,
        dataset=dataset,
    )
