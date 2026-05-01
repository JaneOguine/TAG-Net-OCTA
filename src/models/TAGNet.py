import torch
import torch.nn as nn
import torch.nn.functional as F
import segmentation_models_pytorch as smp


class TopologyFusionBlock(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Conv2d(in_channels * 2, in_channels, kernel_size=1, bias=True),
            nn.Sigmoid()
        )

    def forward(self, seg_feat, topo_feat):
        fusion = torch.cat([seg_feat, topo_feat], dim=1)
        gate = self.gate(fusion)
        return seg_feat + gate * topo_feat


class ProjectionHead(nn.Module):
    def __init__(self, in_channels, emb_dim=128):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, emb_dim, kernel_size=1, bias=False),
        )

    def forward(self, x):
        z = self.proj(x)
        z = F.normalize(z, dim=1)
        return z


class TAGNet(nn.Module):
    def __init__(
        self,
        architecture="unetpp",
        encoder_name="resnet50",
        in_channels=3,
        classes=3,
        encoder_weights="imagenet",
    ):
        super().__init__()

        architecture = architecture.lower()

        if architecture == "unet":
            self.model = smp.Unet(
                encoder_name=encoder_name,
                encoder_weights=encoder_weights,
                in_channels=in_channels,
                classes=classes,
                activation=None,
            )
        elif architecture in ["unetpp", "unetplusplus"]:
            self.model = smp.UnetPlusPlus(
                encoder_name=encoder_name,
                encoder_weights=encoder_weights,
                in_channels=in_channels,
                classes=classes,
                activation=None,
            )
        elif architecture in ["deeplabv3+", "deeplabv3plus"]:
            self.model = smp.DeepLabV3Plus(
                encoder_name=encoder_name,
                encoder_weights=encoder_weights,
                in_channels=in_channels,
                classes=classes,
                activation=None,
            )
        elif architecture == "segformer":
            self.model = smp.Segformer(
                encoder_name=encoder_name,
                encoder_weights=encoder_weights,
                in_channels=in_channels,
                classes=classes,
                activation=None,
            )
        else:
            raise ValueError(f"Unsupported architecture: {architecture}")

        # safer
        if hasattr(self.model.segmentation_head, "0") and hasattr(self.model.segmentation_head[0], "in_channels"):
            decoder_channels = self.model.segmentation_head[0].in_channels
        elif hasattr(self.model.segmentation_head, "in_channels"):
            decoder_channels = self.model.segmentation_head.in_channels
        else:
            raise ValueError("Could not infer decoder_channels from segmentation head.")

        self.topology_head = nn.Sequential(
            nn.Conv2d(decoder_channels, decoder_channels // 2, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(decoder_channels // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(decoder_channels // 2, 1, kernel_size=1)
        )

        self.topo_feat = nn.Sequential(
            nn.Conv2d(decoder_channels, decoder_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(decoder_channels),
            nn.ReLU(inplace=True),
        )

        self.fusion = TopologyFusionBlock(decoder_channels)
        self.projection_head = ProjectionHead(decoder_channels, emb_dim=128)

    def forward(self, x):
        features = self.model.encoder(x)

        try:
            decoder_output = self.model.decoder(*features)
        except TypeError:
            decoder_output = self.model.decoder(features)

        # skeleton logits
        skel_logits = self.topology_head(decoder_output)

        # bring skeleton logits to image size for supervision/visualization
        if skel_logits.shape[-2:] != x.shape[-2:]:
            skel_logits_up = F.interpolate(
                skel_logits, size=x.shape[-2:], mode="bilinear", align_corners=False
            )
        else:
            skel_logits_up = skel_logits

        topo_mask = torch.sigmoid(skel_logits_up)

        topo_features = self.topo_feat(decoder_output)

        # resize topo mask back to decoder size for feature gating
        topo_mask_small = topo_mask
        if topo_mask_small.shape[-2:] != topo_features.shape[-2:]:
            topo_mask_small = F.interpolate(
                topo_mask_small, size=topo_features.shape[-2:], mode="bilinear", align_corners=False
            )

        topo_features = topo_features * topo_mask_small
        fused_features = self.fusion(decoder_output, topo_features)

        seg_logits1 = self.model.segmentation_head(fused_features)

        probs = torch.softmax(seg_logits1, dim=1)
        vessel_prob = probs[:, 1:, ...].sum(dim=1, keepdim=True).detach().clamp(0, 1)

        if vessel_prob.shape[-2:] != fused_features.shape[-2:]:
            vessel_prob = F.interpolate(
                vessel_prob, size=fused_features.shape[-2:], mode="bilinear", align_corners=False
            )

        refined_features = fused_features * (1 + 0.5 * vessel_prob)
        seg_logits2 = self.model.segmentation_head(refined_features)

        return {
            "seg_coarse": seg_logits1,
            "seg_fine": seg_logits2,
            "skeleton": skel_logits_up,
            "topo_mask": topo_mask,
            "features": fused_features,
            "refined_features": refined_features,
            "proj_features": self.projection_head(fused_features),
        }
    



class TAGNet_TopologyOnly(nn.Module):
    """
    Ablation: topology branch only, no fusion.

    What remains:
    - segmentation branch
    - topology/skeleton prediction branch

    What is removed:
    - topology feature modulation
    - topology fusion block
    """

    def __init__(
        self,
        architecture="unetpp",
        encoder_name="resnet50",
        in_channels=3,
        classes=3,
        encoder_weights="imagenet",
    ):
        super().__init__()

        architecture = architecture.lower()

        if architecture == "unet":
            self.model = smp.Unet(
                encoder_name=encoder_name,
                encoder_weights=encoder_weights,
                in_channels=in_channels,
                classes=classes,
                activation=None,
            )
        elif architecture in ["unetpp", "unetplusplus"]:
            self.model = smp.UnetPlusPlus(
                encoder_name=encoder_name,
                encoder_weights=encoder_weights,
                in_channels=in_channels,
                classes=classes,
                activation=None,
            )
        elif architecture in ["deeplabv3+", "deeplabv3plus"]:
            self.model = smp.DeepLabV3Plus(
                encoder_name=encoder_name,
                encoder_weights=encoder_weights,
                in_channels=in_channels,
                classes=classes,
                activation=None,
            )
        elif architecture == "segformer":
            self.model = smp.Segformer(
                encoder_name=encoder_name,
                encoder_weights=encoder_weights,
                in_channels=in_channels,
                classes=classes,
                activation=None,
            )
        else:
            raise ValueError(f"Unsupported architecture: {architecture}")

        # infer decoder channels safely
        if hasattr(self.model.segmentation_head, "0") and hasattr(self.model.segmentation_head[0], "in_channels"):
            decoder_channels = self.model.segmentation_head[0].in_channels
        elif hasattr(self.model.segmentation_head, "in_channels"):
            decoder_channels = self.model.segmentation_head.in_channels
        else:
            raise ValueError("Could not infer decoder_channels from segmentation head.")

        # topology branch stays
        self.topology_head = nn.Sequential(
            nn.Conv2d(decoder_channels, decoder_channels // 2, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(decoder_channels // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(decoder_channels // 2, 1, kernel_size=1)
        )

        # optional only if you still want embeddings for contrastive experiments
        self.projection_head = ProjectionHead(decoder_channels, emb_dim=128)

    def forward(self, x):
        features = self.model.encoder(x)

        try:
            decoder_output = self.model.decoder(*features)
        except TypeError:
            decoder_output = self.model.decoder(features)

        # segmentation WITHOUT topology fusion
        seg_logits = self.model.segmentation_head(decoder_output)

        # auxiliary topology prediction
        skel_logits = self.topology_head(decoder_output)

        if skel_logits.shape[-2:] != x.shape[-2:]:
            skel_logits_up = F.interpolate(
                skel_logits, size=x.shape[-2:], mode="bilinear", align_corners=False
            )
        else:
            skel_logits_up = skel_logits

        topo_mask = torch.sigmoid(skel_logits_up)

        return {
            "seg_coarse": seg_logits,
            "skeleton": skel_logits_up,
            "topo_mask": topo_mask,
            "features": decoder_output,
                }
    


class TAGNet_TopologyAndFusion(nn.Module):
    def __init__(
        self,
        architecture="unetpp",
        encoder_name="resnet50",
        in_channels=3,
        classes=3,
        encoder_weights="imagenet",
    ):
        super().__init__()

        architecture = architecture.lower()

        if architecture == "unet":
            self.model = smp.Unet(
                encoder_name=encoder_name,
                encoder_weights=encoder_weights,
                in_channels=in_channels,
                classes=classes,
                activation=None,
            )
        elif architecture in ["unetpp", "unetplusplus"]:
            self.model = smp.UnetPlusPlus(
                encoder_name=encoder_name,
                encoder_weights=encoder_weights,
                in_channels=in_channels,
                classes=classes,
                activation=None,
            )
        elif architecture in ["deeplabv3+", "deeplabv3plus"]:
            self.model = smp.DeepLabV3Plus(
                encoder_name=encoder_name,
                encoder_weights=encoder_weights,
                in_channels=in_channels,
                classes=classes,
                activation=None,
            )
        elif architecture == "segformer":
            self.model = smp.Segformer(
                encoder_name=encoder_name,
                encoder_weights=encoder_weights,
                in_channels=in_channels,
                classes=classes,
                activation=None,
            )
        else:
            raise ValueError(f"Unsupported architecture: {architecture}")

        # safer
        if hasattr(self.model.segmentation_head, "0") and hasattr(self.model.segmentation_head[0], "in_channels"):
            decoder_channels = self.model.segmentation_head[0].in_channels
        elif hasattr(self.model.segmentation_head, "in_channels"):
            decoder_channels = self.model.segmentation_head.in_channels
        else:
            raise ValueError("Could not infer decoder_channels from segmentation head.")

        self.topology_head = nn.Sequential(
            nn.Conv2d(decoder_channels, decoder_channels // 2, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(decoder_channels // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(decoder_channels // 2, 1, kernel_size=1)
        )

        self.topo_feat = nn.Sequential(
            nn.Conv2d(decoder_channels, decoder_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(decoder_channels),
            nn.ReLU(inplace=True),
        )

        self.fusion = TopologyFusionBlock(decoder_channels)
        self.projection_head = ProjectionHead(decoder_channels, emb_dim=128)

    
    def forward(self, x):
        features = self.model.encoder(x)

        try:
            decoder_output = self.model.decoder(*features)
        except TypeError:
            decoder_output = self.model.decoder(features)

        # ---- Topology branch ----
        skel_logits = self.topology_head(decoder_output)

        if skel_logits.shape[-2:] != x.shape[-2:]:
            skel_logits_up = F.interpolate(
                skel_logits, size=x.shape[-2:], mode="bilinear", align_corners=False
            )
        else:
            skel_logits_up = skel_logits

        topo_mask = torch.sigmoid(skel_logits_up)

        # ---- Topology feature extraction ----
        topo_features = self.topo_feat(decoder_output)

        topo_mask_small = topo_mask
        if topo_mask_small.shape[-2:] != topo_features.shape[-2:]:
            topo_mask_small = F.interpolate(
                topo_mask_small,
                size=topo_features.shape[-2:],
                mode="bilinear",
                align_corners=False
            )

        topo_features = topo_features * topo_mask_small

        # ---- Fusion ----
        fused_features = self.fusion(decoder_output, topo_features)

        # ---- Single segmentation head ONLY ----
        seg_logits = self.model.segmentation_head(fused_features)

        return {
            "seg_coarse": seg_logits,
            "skeleton": skel_logits_up,
            "topo_mask": topo_mask,
            "features": fused_features,
        }