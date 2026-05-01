import os
import json
import glob


def stem(x):
    return os.path.splitext(os.path.basename(x))[0]

def collect_files(folder):
    return sorted(glob.glob(os.path.join(folder, "**", "*.bmp"), recursive=True) +
                  glob.glob(os.path.join(folder, "**", "*.png"), recursive=True) +
                  glob.glob(os.path.join(folder, "**", "*.jpg"), recursive=True) +
                  glob.glob(os.path.join(folder, "**", "*.jpeg"), recursive=True) +
                  glob.glob(os.path.join(folder, "**", "*.tif"), recursive=True) +
                  glob.glob(os.path.join(folder, "**", "*.tiff"), recursive=True))



def load_or_create_json(save_json):
    if os.path.exists(save_json):
        with open(save_json, "r") as f:
            split = json.load(f)
        print(f"Loaded existing JSON: {save_json}")
    else:
        split = {"train": [], "val": [], "test": []}
        print(f"Creating new JSON: {save_json}")
    return split


def get_existing_images(split):
    existing = set()
    for k in ["train", "val", "test"]:
        for item in split[k]:
            existing.add(item["image"])
    return existing


def add_octa_dataset_fixed_split(
        image_dir,
        large_gt_dir,
        cap_gt_dir,
        save_json,
        data_type="OCTA_500",
        octa_type="6mm",
        train_range=(10001, 10240),
        val_range=(10241, 10250),
        test_range=(10251, 10300)
    ):

    split = load_or_create_json(save_json)
    existing_images = get_existing_images(split)

    # Load GT masks
    large_gt_paths = collect_files(large_gt_dir)
    cap_gt_paths = collect_files(cap_gt_dir)

    large_gt_dict = {stem(p): p for p in large_gt_paths}
    cap_gt_dict = {stem(p): p for p in cap_gt_paths}

    print("\n=====================================")
    print(f"ADDING OCTA DATASET ({octa_type}) FROM: {image_dir}")
    print("Train range:", train_range)
    print("Val range:", val_range)
    print("Test range:", test_range)
    print("Found Large GT:", len(large_gt_paths))
    print("Found Capillary GT:", len(cap_gt_paths))

    image_paths = collect_files(image_dir)
    print("Found images:", len(image_paths))

    new_counts = {"train": 0, "val": 0, "test": 0}

    for img_path in image_paths:

        if img_path in existing_images:
            continue

        key = stem(img_path)

        # skip if filename not numeric
        if not key.isdigit():
            continue

        idx = int(key)

        # Assign split
        if train_range[0] <= idx <= train_range[1]:
            split_name = "train"
        elif val_range[0] <= idx <= val_range[1]:
            split_name = "val"
        elif test_range[0] <= idx <= test_range[1]:
            split_name = "test"
        else:
            continue  # not part of split

        large_label_path = large_gt_dict.get(key)
        cap_label_path = cap_gt_dict.get(key)

        if large_label_path is None:
            raise FileNotFoundError(f"Missing LARGE vessel GT for {key} in {large_gt_dir}")

        if cap_label_path is None:
            raise FileNotFoundError(f"Missing CAPILLARY GT for {key} in {cap_gt_dir}")

        paired_data = {
            "image": img_path,
            "large_label": large_label_path,
            "cap_label": cap_label_path,
            "data_type": data_type,
            "octa_type": octa_type
        }

        split[split_name].append(paired_data)
        new_counts[split_name] += 1

    with open(save_json, "w") as f:
        json.dump(split, f, indent=4)

    print("\n✅ OCTA update complete.")
    print("Newly added -> Train:", new_counts["train"],
          "Val:", new_counts["val"],
          "Test:", new_counts["test"])
    print("Total -> Train:", len(split["train"]),
          "Val:", len(split["val"]),
          "Test:", len(split["test"]))




def add_rossa_dataset(rossa_root,
                      save_json,
                      data_type="ROSSA"):

    split = load_or_create_json(save_json)
    existing_images = get_existing_images(split)

    split_folders = {
        "train": [os.path.join(rossa_root, "train_manual"),
                  os.path.join(rossa_root, "train_sam")],
        "val": [os.path.join(rossa_root, "val")],
        "test": [os.path.join(rossa_root, "test")]
    }

    print("\n=====================================")
    print(f"ADDING ROSSA DATASET: {rossa_root}")

    new_counts = {"train": 0, "val": 0, "test": 0}

    for split_name, folders in split_folders.items():
        for folder in folders:

            image_dir = os.path.join(folder, "image")
            label_dir = os.path.join(folder, "label")

            image_paths = collect_files(image_dir)
            label_paths = collect_files(label_dir)

            label_dict = {stem(p): p for p in label_paths}

            print(f"[{split_name.upper()}] {folder}")
            print("   Images:", len(image_paths))
            print("   Labels:", len(label_paths))

            for img_path in image_paths:

                if img_path in existing_images:
                    continue

                key = stem(img_path)

                large_label_path = label_dict.get(key)

                if large_label_path is None:
                    raise FileNotFoundError(f"Missing label for {key} in {label_dir}")

                paired_data = {
                    "image": img_path,
                    "large_label": large_label_path,
                    "cap_label": None,
                    "data_type": data_type,
                    "octa_type": "N/A",
                    "source_folder": os.path.basename(folder)
                }

                split[split_name].append(paired_data)
                new_counts[split_name] += 1

    with open(save_json, "w") as f:
        json.dump(split, f, indent=4)

    print("\n✅ ROSSA update complete.")
    print("Newly added -> Train:", new_counts["train"],
          "Val:", new_counts["val"],
          "Test:", new_counts["test"])
    print("Total -> Train:", len(split["train"]),
          "Val:", len(split["val"]),
          "Test:", len(split["test"]))




def add_rose_dataset(
        rose_root,
        save_json,
        data_type="ROSE",
        octa_type="N/A"
    ):

    split = load_or_create_json(save_json)
    existing_images = get_existing_images(split)

    print("\n=====================================")
    print(f"ADDING ROSE DATASET FROM: {rose_root}")

    new_counts = {"train": 0, "val": 0, "test": 0}

    # Define all possible train/test locations
    rose_splits = []

    # ROSE-1 SVC
    rose_splits.append(("train", os.path.join(rose_root, "ROSE-1", "SVC", "train")))
    rose_splits.append(("test",  os.path.join(rose_root, "ROSE-1", "SVC", "test")))

    # # ROSE-1 DVC (if you want it)
    # rose_splits.append(("train", os.path.join(rose_root, "ROSE-1", "DVC", "train")))
    # rose_splits.append(("test",  os.path.join(rose_root, "ROSE-1", "DVC", "test")))

    # # ROSE-2
    # rose_splits.append(("train", os.path.join(rose_root, "ROSE-2", "train")))
    # rose_splits.append(("test",  os.path.join(rose_root, "ROSE-2", "test")))

    for split_name, base_folder in rose_splits:

        img_dir = os.path.join(base_folder, "img")
        thick_dir = os.path.join(base_folder, "thick_gt")
        thin_dir = os.path.join(base_folder, "thin_gt")

        if not os.path.exists(img_dir):
            continue

        image_paths = collect_files(img_dir)
        thick_paths = collect_files(thick_dir)
        # thin_paths = collect_files(thin_dir)

        thick_dict = {stem(p): p for p in thick_paths}
        # thin_dict  = {stem(p): p for p in thin_paths}

        print(f"\n[{split_name.upper()}] {base_folder}")
        print("Images:", len(image_paths))
        print("Thick GT:", len(thick_paths))
        # print("Thin GT:", len(thin_paths))

        for img_path in image_paths:

            if img_path in existing_images:
                continue

            key = stem(img_path)

            large_label_path = thick_dict.get(key)
            # cap_label_path   = thin_dict.get(key)

            if large_label_path is None:
                raise FileNotFoundError(f"Missing thick_gt for {key} in {thick_dir}")

            # if cap_label_path is None:
            #     raise FileNotFoundError(f"Missing thin_gt for {key} in {thin_dir}")

            paired_data = {
                "image": img_path,
                "large_label": large_label_path,
                "cap_label": None,
                "data_type": data_type,
                "octa_type": octa_type,
                "source_folder": "ROSE"
            }

            split[split_name].append(paired_data)
            new_counts[split_name] += 1

    with open(save_json, "w") as f:
        json.dump(split, f, indent=4)

    print("\n✅ ROSE update complete.")
    print("Newly added -> Train:", new_counts["train"],
          "Val:", new_counts["val"],
          "Test:", new_counts["test"])
    print("Total -> Train:", len(split["train"]),
          "Val:", len(split["val"]),
          "Test:", len(split["test"]))
    

if __name__ == "__main__":

    dataset_root = "/home/oguinekj/Documents/Scripts/Retina_OCTA_3/dataset"
    save_json = "all_vessel_data_split.json"

    # Shared GT directories (one for all IDs)
    large_gt_dir = "/home/oguinekj/Documents/Data/OCTA500/Label/GT_LargeVessel/"
    cap_gt_dir   = "/home/oguinekj/Documents/Data/OCTA500/Label/GT_Capillary/"

    # OCTA500 6mm image directory
    octa6_image_dir = "/home/oguinekj/Documents/Data/OCTA500/OCTA_6mm_part8/OCTA_6mm/Projection Maps/OCTA(ILM_OPL)/"
    add_octa_dataset_fixed_split(
        image_dir=octa6_image_dir,
        large_gt_dir=large_gt_dir,
        cap_gt_dir=cap_gt_dir,
        save_json=save_json,
        data_type="OCTA_500",
        octa_type="6mm",
        train_range=(10001, 10240),
        val_range=(10241, 10250),
        test_range=(10251, 10300)
    )

    # OCTA500 3mm image directory
    octa3_image_dir = "/home/oguinekj/Documents/Data/OCTA500/OCTA_3mm_part3/OCTA_3mm/Projection Maps/OCTA(ILM_OPL)/"
    add_octa_dataset_fixed_split(
        image_dir=octa3_image_dir,
        large_gt_dir=large_gt_dir,
        cap_gt_dir=cap_gt_dir,
        save_json=save_json,
        data_type="OCTA_500",
        octa_type="3mm",
        train_range=(10301, 10440),
        val_range=(10441, 10450),
        test_range=(10451, 10500)
    )

    # ROSSA
    rossa_root = os.path.join(dataset_root, "ROSSA")
    add_rossa_dataset(
        rossa_root=rossa_root,
        save_json=save_json
    )

    rose_root = "/home/oguinekj/Documents/Data/ROSE"
    add_rose_dataset(
        rose_root=rose_root,
        save_json=save_json,
        octa_type="3mm"
    )


    print("\n🔥 ALL DATASETS ADDED SUCCESSFULLY!")

