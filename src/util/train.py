from sklearn.decomposition import PCA
import os
import matplotlib.pyplot as plt

def save_feature_map(feature_tensor, save_dir, epoch, feature_name="features"):
    feature_dir = os.path.join(save_dir, feature_name)
    os.makedirs(feature_dir, exist_ok=True)

    feat = feature_tensor[0].detach().cpu()
    C, H, W = feat.shape

    feat_flat = feat.permute(1, 2, 0).reshape(-1, C).numpy()
    pca = PCA(n_components=1)
    reduced = pca.fit_transform(feat_flat)  
    feat_map = reduced.reshape(H, W)

    plt.figure(figsize=(6, 6))
    plt.imshow(feat_map, cmap="viridis")
    plt.axis("off")
    plt.savefig(os.path.join(feature_dir, f"epoch_{epoch + 1}.png"), bbox_inches="tight", pad_inches=0)
    plt.close()

