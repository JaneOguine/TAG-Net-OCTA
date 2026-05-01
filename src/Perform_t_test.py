import pandas as pd
import numpy as np
from scipy.stats import ttest_rel

def run_paired_tests(baseline_csv, thickness_csv):

    base_df = pd.read_csv(baseline_csv)
    thick_df = pd.read_csv(thickness_csv)

    # Merge by image name 
    merged = pd.merge(
        base_df,
        thick_df,
        on="image_name",
        suffixes=("_base", "_thick")
    )

    if len(merged) == 0:
        raise ValueError("No matching image names found.")

    metrics = {
        "Dice Overall": "dice_overall",
        "Dice Vessel Only": "vessel_only_dice",
        "clDice Overall": "cldice_binary",
        "clDice Vessel Only": "cldice_vessel_only"
    }

    print("="*60)
    print(f"Matched samples: {len(merged)}")
    print("="*60)

    for label, column in metrics.items():

        base = merged[f"{column}_base"].astype(float)
        thick = merged[f"{column}_thick"].astype(float)

        # remove NaNs
        valid = (~base.isna()) & (~thick.isna())
        base = base[valid]
        thick = thick[valid]

        if len(base) < 2:
            print(f"{label}: Not enough samples.")
            continue

        # paired t-test
        t_stat, p_value = ttest_rel(thick, base)

        diff = thick - base
        cohens_d = diff.mean() / diff.std(ddof=1)

        print(f"\n{label}")
        print(f"Baseline mean : {base.mean():.4f}")
        print(f"Thickness mean: {thick.mean():.4f}")
        print(f"Mean gain     : {diff.mean():.4f}")
        print(f"T-stat        : {t_stat:.4f}")
        print(f"P-value       : {p_value:.6f}")
        print(f"Cohen's d     : {cohens_d:.4f}")

        if p_value < 0.001:
            print("Significance  : *** (p < 0.001)")
        elif p_value < 0.01:
            print("Significance  : **  (p < 0.01)")
        elif p_value < 0.05:
            print("Significance  : *   (p < 0.05)")
        else:
            print("Significance  : Not significant")

    print("="*60)



if __name__=="__main__":
    baseline_path = "/home/user/Documents/OCTA/dice_csv_results/Baseline_unet_model/OCTA_500_6mm.csv"    #path to baseline model(unet, segformer, etc) dice results
    thickness_path = "/home/user/Documents/OCTA/dice_csv_results/Thickness_cond_unet_model/OCTA_500_6mm.csv"  # path to thickness conditioned model (same model as basline)
    run_paired_tests(baseline_path, thickness_path)