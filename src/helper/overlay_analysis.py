import os
import json
import numpy as np
from PIL import Image
from natsort import natsorted


def apply_tpfpfn_overlay(image, gt_mask, pred_mask, alpha=0.6):
    overlay = image.copy()

    gt = gt_mask > 0
    pred = pred_mask > 0

    tp = pred & gt
    fp = pred & (~gt)
    fn = (~pred) & gt
    tn = (~pred) & (~gt)

    overlay[tn] = (np.array([0, 0, 0]) * alpha + overlay[tn] * (1 - alpha)).astype(np.uint8)        # TN (Black)
    overlay[tp] = (np.array([0, 255, 0]) * alpha + overlay[tp] * (1 - alpha)).astype(np.uint8)      # TP (Green)
    overlay[fp] = (np.array([255, 0, 0]) * alpha + overlay[fp] * (1 - alpha)).astype(np.uint8)      # FP (Red)
    overlay[fn] = (np.array([255, 255, 0]) * alpha + overlay[fn] * (1 - alpha)).astype(np.uint8)    # FN (Yellow)

    return overlay


import os
import json
import numpy as np
from PIL import Image
from natsort import natsorted


# =========================================================
# TP/FP/FN MASK-ONLY VISUALIZATION
# =========================================================
def tpfpfn_mask_only(gt_mask, pred_mask):
    """
    Creates mask-only visualization:
        TP = White
        FP = Green
        FN = Red
        Background = Black
    """

    gt = gt_mask > 0
    pred = pred_mask > 0

    tp = pred & gt
    fp = pred & (~gt)
    fn = (~pred) & gt

    # Create blank RGB image
    out = np.zeros((*gt.shape, 3), dtype=np.uint8)

    out[tp] = (255, 255, 255)  # TP = white
    out[fp] = (0, 255, 0)      # FP = green
    out[fn] = (255, 0, 0)      # FN = red

    return out


# =========================================================
# GET FILES FROM JSON SPLIT
# =========================================================
def get_images(json_file, image_path, split):
    with open(json_file, "r") as file:
        data = json.load(file)

    data_split = data[split]
    images = [os.path.basename(item["image"]) for item in data_split]
    masks = [os.path.basename(item["bin_mask"]) for item in data_split]

    images_required = []
    for img in os.listdir(image_path):
        if img in images:
            images_required.append(img)

    return natsorted(images_required), natsorted(masks)


def main():

    gt_mask_path = "/home/oguinekj/Documents/Scripts/Retina_OCTA_2/results/Baseline_Unet"
    pred_mask_path = "/home/oguinekj/Documents/Scripts/Retina_OCTA_2/results/Thickness_conditioned_enc"
    image_path = "/home/oguinekj/Documents/Data/OCTA500/OCTA_6mm_part8/OCTA_6mm/Projection Maps/OCTA(FULL)"
    json_file = "/home/oguinekj/Documents/Scripts/Retina_OCTA_2/datasplit/split.json"
    save_path = "/home/oguinekj/Documents/Scripts/Retina_OCTA_2/overlay/MaskOnly_TPFPFN"

    os.makedirs(save_path, exist_ok=True)

    overlay_source = "json"   # options: "json" | "image_path"

    if overlay_source == "json":
        images, gt_masks = get_images(json_file, image_path, split="test")
        pred_masks = gt_masks  # assumes same filenames

    elif overlay_source == "image_path":
        images = natsorted(os.listdir(image_path))
        gt_masks = natsorted(os.listdir(gt_mask_path))
        pred_masks = natsorted(os.listdir(pred_mask_path))

    else:
        raise ValueError("Invalid overlay_source")

    print(f"Total images: {len(images)}")

    for img, gt_msk, pred_msk in zip(images, gt_masks, pred_masks):

        gt_path = os.path.join(gt_mask_path, gt_msk)
        pred_path = os.path.join(pred_mask_path, pred_msk)

        gt_mask = np.array(Image.open(gt_path).convert("L"))
        pred_mask = np.array(Image.open(pred_path).convert("L"))

        # Ensure same size
        if pred_mask.shape != gt_mask.shape:
            pred_mask = np.array(
                Image.fromarray(pred_mask).resize(
                    gt_mask.shape[::-1],
                    resample=Image.NEAREST
                )
            )

        colored = tpfpfn_mask_only(gt_mask, pred_mask)

        save_file = os.path.join(save_path, img)
        Image.fromarray(colored).save(save_file)

        print(f"Saved: {save_file}")


if __name__ == "__main__":
    main()