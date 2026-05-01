import torch
import numpy as np
from torch.utils.data import Dataset
import json
import PIL.Image as Image
from skimage.morphology import  medial_axis
from visualize import *
from config.config_args import *


filename_types = ['png', 'jpg', 'jpeg', 'tif', 'tiff', 'PNG', 'JPG', 'JPEG', 'TIFF', 'TIFF', 'npz', 'npy']


def thickness_centerline(mask):   
    M = mask.astype(bool)
    skel = medial_axis(M).astype(bool)
    return skel


class OCTA_dataset(Dataset):
    def __init__(self, args, mode='train', transform_pad=None, transform_resize=None):
        self.args = args
        self.mode = mode
        self.transform_pad = transform_pad
        self.transform_resize = transform_resize
        

        if mode != 'test':
            json_file_path = args.json_path
            with open(json_file_path, 'r') as file:
                split = json.load(file)
            self.data = split[mode]
        else:
            if args.evaluate:  
                json_file_path = args.json_path
                with open(json_file_path, 'r') as file:
                    split = json.load(file)
                self.data = split[mode]
        
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if self.mode != 'test':
            image_path = self.data[idx]['image']
            large_label_path = self.data[idx]['large_label']
            thin_label_path = self.data[idx].get('cap_label', None)
            data_type = self.data[idx]['data_type']
            octa_type = self.data[idx]['octa_type']

            img = Image.open(image_path)
            image = np.array(img.convert("L"))
            h, w = image.shape[:2]

            # read thick vessel mask
            large_mask = np.array(Image.open(large_label_path).convert("L"))
            large_mask = (large_mask > 0).astype(np.uint8)

            thin_mask = np.zeros((h, w), dtype=np.uint8)
            if thin_label_path is not None:
                thin_mask = np.array(Image.open(thin_label_path).convert("L"))
                thin_mask = (thin_mask > 0).astype(np.uint8)

            # multiclass mask
            mask = np.zeros((h, w), dtype=np.uint8)
            mask[(thin_mask == 1) & (large_mask == 0)] = 1    # thin vessel first
            mask[large_mask == 1] = 2    # thick vessel second


        else:
            image_path = self.data[idx]['image']
            large_label_path = self.data[idx]['large_label']
            thin_label_path = self.data[idx].get('cap_label', None)
            data_type = self.data[idx]['data_type']
            octa_type = self.data[idx]['octa_type']

            img = Image.open(image_path)
            image = np.array(img.convert("L"))
            h, w = image.shape[:2]

            # read thick vessel mask
            large_mask = np.array(Image.open(large_label_path).convert("L"))
            large_mask = (large_mask > 0).astype(np.uint8)

            thin_mask = np.zeros((h, w), dtype=np.uint8)
            if thin_label_path is not None:
                thin_mask = np.array(Image.open(thin_label_path).convert("L"))
                thin_mask = (thin_mask > 0).astype(np.uint8)
        
            mask = np.zeros((h, w), dtype=np.uint8)
            mask[(thin_mask == 1) & (large_mask == 0)] = 1    # thin vessel first
            mask[large_mask == 1] = 2    # thick vessel second

          # Choose transform
        if data_type == "OCTA_500" and octa_type == "3mm":
            transform = self.transform_pad
        else:
            transform = self.transform_resize

        image = image[..., None].astype(np.float32)     
        mask  = mask[..., None].astype(np.uint8)
        
        if transform is not None:
            augmented = transform(image=image, mask=mask)
            image = augmented['image']
            mask  = augmented['mask']
 
        if image.ndim == 3 and image.shape[0] in [1, 3]:
            image = np.moveaxis(image, 0, -1)  

        if mask.ndim == 3 and mask.shape[0] == 1: 
            mask = np.moveaxis(mask, 0, -1)    

        mask2d_aug = mask[..., 0].astype(np.uint8)
        bin_mask = (mask2d_aug > 0).astype(np.uint8)

        #  Topology (Skeleton) after augmentation 
        mask2d = mask[..., 0].astype(np.uint8)
        mask_binary_thin  = (mask2d == 1).astype(np.uint8)
        mask_binary_large = (mask2d == 2).astype(np.uint8)
    
        skel_thin  = medial_axis(mask_binary_thin)
        skel_large = medial_axis(mask_binary_large)
        skel = (skel_thin | skel_large).astype(np.uint8)

        centerline_mask = skel[None, ...]   
        image = np.repeat(image, 3, axis=-1)  
        image = np.moveaxis(image, -1, 0) 

        mask_binary_thin = mask_binary_thin[None, ...]
        mask_binary_large = mask_binary_large[None, ...]

        data = {
            'image': torch.from_numpy(image).float(),
            'label': torch.from_numpy(mask2d).long(),
            "bin_mask": torch.from_numpy(bin_mask).bool(),
            'mask_binary_thin': torch.from_numpy(mask_binary_thin).bool(),
            'mask_binary_large': torch.from_numpy(mask_binary_large).bool(),
            "centerline_mask": torch.from_numpy(centerline_mask).bool(),
            'name': image_path,
            'data_type': data_type,
            'octa_type': octa_type,
            'original_shape': (h, w)
            }
        return data


# if __name__== "__main__":
#     args = parser.parse_args()
#     dataset = OCTA_dataset(args, mode="train", transform_pad=None, transform_resize=None)
#     idx = 1115
#     sample = dataset[idx]  # pick any index

#     img = sample['image'].numpy()          # [3,H,W]
#     mask = sample['label'].numpy()         # [H,W]
#     skel = sample['centerline_mask'].numpy()[0]  # [H,W]

#     # convert image to H,W,C
#     img = img.transpose(1, 2, 0)
#     img_vis = (img - img.min()) / (img.max() - img.min() + 1e-8)

#     plt.figure(figsize=(12, 3))  # reduce height → tighter visually

# thin = sample['mask_binary_thin'].numpy()[0]
# large = sample['mask_binary_large'].numpy()[0]

# from skimage.morphology import medial_axis

# skel_thin = medial_axis(thin)
# skel_large = medial_axis(large)

# fig, axes = plt.subplots(1, 5, figsize=(12, 3))

# images = [img_vis, mask, skel_thin, skel_large, skel]
# cmaps  = ['gray'] * 5

# for ax, im, cmap in zip(axes, images, cmaps):
#     ax.imshow(im, cmap=cmap)
#     ax.axis('off')

# # KEY PART — remove gaps
# plt.subplots_adjust(wspace=0.02, hspace=0)

# # save with NO padding
# plt.savefig(f"visualize/image_{idx}.png",
#             bbox_inches='tight',
#             pad_inches=0)

# plt.show()