import os
import csv
import numpy as np
from skimage.morphology import skeletonize



def safe_mean(x):
    return float(np.mean(x)) if len(x) > 0 else float("nan")

def safe_std(x):
    return float(np.std(x)) if len(x) > 0 else float("nan")

def dice_binary(pred, gt, eps=1e-6):
    pred = pred.float()
    gt = gt.float()
    inter = (pred * gt).sum()
    union = pred.sum() + gt.sum()
    return (2 * inter + eps) / (union + eps)

def precision_score(pred, gt, eps=1e-6):
    tp = (pred * gt).sum()
    fp = (pred * (1 - gt)).sum()
    return tp / (tp + fp + eps)

def recall_score(pred, gt, eps=1e-6):
    tp = (pred * gt).sum()
    fn = ((1 - pred) * gt).sum()
    return tp / (tp + fn + eps)

def accuracy_score(pred, gt):
    return (pred == gt).float().mean()


def specificity_score(pred, gt, eps=1e-6):
    tn = ((1 - pred) * (1 - gt)).sum()
    fp = (pred * (1 - gt)).sum()
    return tn / (tn + fp + eps)


def cldice_score(pred_mask, gt_mask, eps=1e-6):
    pred_mask = pred_mask.astype(np.uint8)
    gt_mask   = gt_mask.astype(np.uint8)
    if pred_mask.sum() == 0 and gt_mask.sum() == 0:
        return 1.0
    if pred_mask.sum() == 0 or gt_mask.sum() == 0:
        return 0.0
    
    skel_pred = skeletonize(pred_mask > 0)
    skel_gt   = skeletonize(gt_mask > 0)
    tprec = (skel_pred & gt_mask).sum() / (skel_pred.sum() + eps)
    tsens = (skel_gt & pred_mask).sum() / (skel_gt.sum() + eps)
    return float(2 * tprec * tsens / (tprec + tsens + eps))


def dice_binary(pred, gt, eps=1e-6):
    pred = pred.float()
    gt = gt.float()
    intersection = (pred * gt).sum()
    union = pred.sum() + gt.sum()
    return (2.0 * intersection + eps) / (union + eps)


def tpfpfn_mask_only(gt_mask, pred_mask):

    gt = gt_mask > 0
    pred = pred_mask > 0

    tp = pred & gt
    fp = pred & (~gt)
    fn = (~pred) & gt

    out = np.zeros((*gt.shape, 3), dtype=np.uint8)

    out[tp] = (255, 255, 255)  # white
    out[fp] = (0, 255, 0)      # green
    out[fn] = (0, 0, 255)      # red

    fp_img = np.zeros_like(out)
    fp_img[fp] = (0, 255, 0)

    fn_img = np.zeros_like(out)
    fn_img[fn] = (0, 0, 255)

    return out, fp_img, fn_img


def save_per_sample_dice_csv(args, grouped, save_dir="./dice_csv_results"):
    os.makedirs(save_dir, exist_ok=True)
    for group_key in grouped.keys():

        vessel_only_dice = grouped[group_key]["vessel_overall"]
        overall_dice     = grouped[group_key]["overall"]  
        vessel_only_cl   = grouped[group_key]["cl_overall"]
        binary_cl        = grouped[group_key]["cl_bin"]

        names = grouped[group_key]["names"]

        if len(vessel_only_dice) == 0:
            continue
        
        dir_name = os.path.basename(args.save_results_dir.rstrip("/"))
        save_path = os.path.join(save_dir, dir_name)
        os.makedirs(save_path, exist_ok=True)
        csv_path = os.path.join(save_path, f"{group_key}.csv")

        with open(csv_path, mode="w", newline="") as f:
            writer = csv.writer(f)

            writer.writerow([
                "sample_id",
                "image_name",
                "vessel_only_dice",
                "dice_overall",
                "cldice_vessel_only",
                "cldice_binary"
            ])

            for i in range(len(vessel_only_dice)):
                writer.writerow([
                    i,
                    names[i],
                    float(vessel_only_dice[i]) if not np.isnan(vessel_only_dice[i]) else "",
                    float(overall_dice[i])     if not np.isnan(overall_dice[i])     else "",
                    float(vessel_only_cl[i])   if not np.isnan(vessel_only_cl[i])   else "",
                    float(binary_cl[i])        if not np.isnan(binary_cl[i])        else ""
                ])

        print(f"Saved CSV: {csv_path}")