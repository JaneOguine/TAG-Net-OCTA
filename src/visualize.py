import matplotlib.pyplot as plt
import torch
import os
import numpy as np


def visualize_thickness_map(mask, tmap, vessel_type="Large"):
    """
    mask is the binary mask
    tmap is the thickness map
    """
    plt.figure(figsize=(12, 4))

    plt.subplot(1, 3, 1)
    plt.title(f"{vessel_type} vessel mask")
    plt.imshow(mask, cmap="gray")
    plt.axis("off")

    plt.subplot(1, 3, 2)
    plt.title("Thickness map (raw)")
    plt.imshow(tmap, cmap="gray")
    plt.colorbar()
    plt.axis("off")

    plt.subplot(1, 3, 3)
    plt.title("Overlay")
    plt.imshow(mask, cmap="gray")
    plt.imshow(tmap, cmap="jet", alpha=0.6)
    plt.axis("off")

    plt.show()


def visualize_normailize_thickness_map(Tn, vessel_type="large vessels"):
    plt.figure(figsize=(6, 6))
    plt.title(f"Normalized thickness ({vessel_type})")
    plt.imshow(Tn, cmap="jet")
    plt.colorbar(label="Normalized thickness")
    plt.axis("off")
    plt.show()



def visualize_sample(image, mask, thickness, idx=None):

    if torch.is_tensor(image):
        image = image.cpu().numpy()
    if torch.is_tensor(mask):
        mask = mask.cpu().numpy()
    if torch.is_tensor(thickness):
        thickness = thickness.cpu().numpy()

    image_gray = image[..., 0]
    thickness  = thickness[..., 0]

    fig, axs = plt.subplots(1, 3, figsize=(16, 4))

    axs[0].imshow(image_gray, cmap='gray')
    axs[0].set_title("OCTA image")

    axs[1].imshow(mask, cmap='gray')
    axs[1].set_title("Vessel mask")

    im = axs[2].imshow(thickness, cmap='inferno')
    axs[2].set_title("Log thickness")
    plt.colorbar(im, ax=axs[2])

    for ax in axs:
        ax.axis("off")

    if idx is not None:
        fig.suptitle(f"Sample {idx}", fontsize=14)

    plt.tight_layout()
    plt.show()
    plt.savefig(f"/home/oguinekj/Documents/Scripts/Retina_OCTA_2/debug/debug_sample_{idx}.png", dpi=150) 
    plt.close()




def visualize_full_sample(sample, idx=0, save_dir=None):
    """
    sample: output dictionary from your dataset __getitem__
    idx: sample index
    save_dir: if not None, saves the image
    """

    image = sample["image"]
    bin_mask = sample["bin_mask"]
    lv_mask = sample["mask_binary_large"]
    cap_mask = sample["mask_binary_cap"]
    thick_lv = sample["thickness_large"]
    thick_cap = sample["thickness_cap"]

    if torch.is_tensor(image):
        image = image.cpu().numpy()
    if torch.is_tensor(bin_mask):
        bin_mask = bin_mask.cpu().numpy()
    if torch.is_tensor(lv_mask):
        lv_mask = lv_mask.cpu().numpy()
    if torch.is_tensor(cap_mask):
        cap_mask = cap_mask.cpu().numpy()
    if torch.is_tensor(thick_lv):
        thick_lv = thick_lv.cpu().numpy()
    if torch.is_tensor(thick_cap):
        thick_cap = thick_cap.cpu().numpy()


    image_gray = image[0]   # first channel
    bin_mask = bin_mask.squeeze()
    lv_mask = lv_mask.squeeze()
    cap_mask = cap_mask.squeeze()
    thick_lv = thick_lv.squeeze()
    thick_cap = thick_cap.squeeze()


    fig, axs = plt.subplots(2, 4, figsize=(22, 10))

    axs[0, 0].imshow(image_gray, cmap="gray")
    axs[0, 0].set_title("OCTA Image")

    axs[0, 1].imshow(bin_mask, cmap="gray")
    axs[0, 1].set_title("Binary Vessel Mask")

    axs[0, 2].imshow(lv_mask, cmap="gray")
    axs[0, 2].set_title("Large Vessel Mask")

    axs[0, 3].imshow(cap_mask, cmap="gray")
    axs[0, 3].set_title("Capillary Mask")

    im1 = axs[1, 0].imshow(thick_lv, cmap="inferno")
    axs[1, 0].set_title("Thickness Map (Large)")
    plt.colorbar(im1, ax=axs[1, 0], fraction=0.046)

    im2 = axs[1, 1].imshow(thick_cap, cmap="inferno")
    axs[1, 1].set_title("Thickness Map (Capillary)")
    plt.colorbar(im2, ax=axs[1, 1], fraction=0.046)

    axs[1, 2].imshow(image_gray, cmap="gray")
    axs[1, 2].imshow(thick_lv, cmap="jet", alpha=0.6)
    axs[1, 2].set_title("Overlay Large Thickness")

    axs[1, 3].imshow(image_gray, cmap="gray")
    axs[1, 3].imshow(thick_cap, cmap="jet", alpha=0.6)
    axs[1, 3].set_title("Overlay Cap Thickness")

    for ax in axs.flatten():
        ax.axis("off")

    fig.suptitle(f"Sample {idx}", fontsize=18)
    plt.tight_layout()

  
    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, f"debug_sample_{idx}.png")
        plt.savefig(save_path, dpi=200)
        print(f"Saved visualization -> {save_path}")

    plt.show()
    plt.close()


def debug_visualize_dataset(
        image,
        bin_mask,
        lv_mask,
        cap_mask,
        thick_map,
        idx=0,
        save_path=None):

    fig, axs = plt.subplots(2, 3, figsize=(18, 10))

    #  Row 1 
    axs[0, 0].imshow(image, cmap="gray")
    axs[0, 0].set_title("Image")

    axs[0, 1].imshow(bin_mask, cmap="gray")
    axs[0, 1].set_title("Binary Mask")

    axs[0, 2].imshow(lv_mask, cmap="gray")
    axs[0, 2].set_title("Large Vessel Mask")

    #  Row 2 
    axs[1, 0].imshow(cap_mask, cmap="gray")
    axs[1, 0].set_title("Capillary Mask")

    im = axs[1, 1].imshow(thick_map, cmap="inferno")
    axs[1, 1].set_title("Thickness Map")
    plt.colorbar(im, ax=axs[1, 1], fraction=0.046)

    axs[1, 2].imshow(image, cmap="gray")
    axs[1, 2].imshow(thick_map, cmap="jet", alpha=0.6)
    axs[1, 2].set_title("Thickness Overlay")

    for ax in axs.flatten():
        ax.axis("off")

    fig.suptitle(f"Debug Sample {idx}", fontsize=16)
    plt.tight_layout()

    if save_path is not None:
        os.makedirs(save_path, exist_ok=True)
        save_file = os.path.join(save_path, f"debug_sample_{idx}.png")
        plt.savefig(save_file, dpi=200)
        print("Saved debug plot:", save_file)

    plt.show()
    plt.close()



def _save_visualization(batch, image, gt, pred, mask, save_dir):

    image_np = image[0].detach().cpu().permute(1, 2, 0).numpy()
    image_np = (image_np - image_np.min()) / (image_np.max() - image_np.min() + 1e-8)

    mask_np = mask[0].detach().cpu().numpy()

    gt_np = gt[0, 0].detach().cpu().numpy()
    pred_np = pred[0, 0].detach().cpu().numpy()

    # mask background for error visualization
    vessel_mask = gt_np > 0
    err_np = np.abs(pred_np - gt_np)
    err_np[~vessel_mask] = np.nan

    # shared color scale for GT & prediction
    vmin = min(gt_np.min(), pred_np.min())
    vmax = max(gt_np.max(), pred_np.max())
    name = os.path.splitext(os.path.basename(batch["name"][0]))[0]

    fig, axs = plt.subplots(1, 5, figsize=(18, 4))

    #  Input 
    axs[0].imshow(image_np, cmap="gray")
    axs[0].set_title("Input OCTA", fontsize=12)
    axs[0].axis("off")

    axs[1].imshow(mask_np, cmap="gray")
    axs[0].set_title("Input OCTA", fontsize=12)
    axs[1].axis("off")

    #  GT 
    im2 = axs[2].imshow(gt_np, cmap="magma", vmin=vmin, vmax=vmax)
    axs[1].set_title("Ground Truth Thickness", fontsize=12)
    axs[2].axis("off")
    plt.colorbar(im2, ax=axs[2], fraction=0.046, pad=0.02)

    #  Prediction 
    im3 = axs[3].imshow(pred_np, cmap="magma", vmin=vmin, vmax=vmax)
    axs[2].set_title("Predicted Thickness", fontsize=12)
    axs[3].axis("off")
    plt.colorbar(im3, ax=axs[3], fraction=0.046, pad=0.02)

    # absolute error 
    im4 = axs[4].imshow(err_np, cmap="magma", vmin=0, vmax=np.nanpercentile(err_np, 95))
    axs[4].axis("off")
    plt.colorbar(im4, ax=axs[4], fraction=0.046, pad=0.02)

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"{name}_thickness.png"), dpi=300)
    plt.close()