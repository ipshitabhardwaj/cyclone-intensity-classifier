"""
Multi-task cyclone model: one EfficientNet-B0 backbone (adapted to accept an
arbitrary number of single-channel satellite bands instead of 3-channel RGB),
with two heads:
  - classification head -> IMD intensity category (8 classes)
  - regression head     -> [normalized wind speed, normalized pressure]

Explainability (Grad-CAM) hooks onto `self.backbone.features[-1]`, the last
conv block, which `gradcam.py` targets by name.
"""
import torch
import torch.nn as nn
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights


class CycloneNet(nn.Module):
    def __init__(self, in_channels: int = 4, num_classes: int = 8, pretrained: bool = True,
                 num_reg_outputs: int = 2):
        super().__init__()

        weights = None
        if pretrained:
            try:
                weights = EfficientNet_B0_Weights.DEFAULT
            except Exception as e:
                print(f"[model] Could not resolve pretrained weights enum ({e}); using random init.")

        try:
            backbone = efficientnet_b0(weights=weights)
        except Exception as e:
            # Common in sandboxed/offline environments: weight download blocked.
            print(f"[model] Pretrained weight download failed ({e}); falling back to random init.")
            backbone = efficientnet_b0(weights=None)

        # --- adapt the stem conv to accept `in_channels` bands instead of 3 ---
        old_conv = backbone.features[0][0]  # Conv2d(3, 32, k=3, s=2, p=1, bias=False)
        new_conv = nn.Conv2d(
            in_channels, old_conv.out_channels,
            kernel_size=old_conv.kernel_size, stride=old_conv.stride,
            padding=old_conv.padding, bias=(old_conv.bias is not None),
        )
        with torch.no_grad():
            # average the pretrained 3-channel filters and replicate across
            # the new channel count, so pretrained low-level edge/texture
            # filters are preserved rather than thrown away
            mean_w = old_conv.weight.mean(dim=1, keepdim=True)  # (32,1,3,3)
            new_conv.weight.copy_(mean_w.repeat(1, in_channels, 1, 1))
        backbone.features[0][0] = new_conv

        self.backbone_features = backbone.features  # conv stack, up to last block
        self.avgpool = backbone.avgpool
        in_feats = backbone.classifier[1].in_features  # 1280 for b0

        self.dropout = nn.Dropout(p=0.2)
        self.cls_head = nn.Linear(in_feats, num_classes)
        self.reg_head = nn.Sequential(
            nn.Linear(in_feats, 64), nn.ReLU(inplace=True), nn.Linear(64, num_reg_outputs)
        )

        # exposed so Grad-CAM can register hooks by name
        self.target_layer = self.backbone_features[-1]

    def forward(self, x):
        feats = self.backbone_features(x)          # (B, 1280, h, w)
        pooled = self.avgpool(feats).flatten(1)     # (B, 1280)
        pooled = self.dropout(pooled)
        cls_out = self.cls_head(pooled)             # (B, num_classes)
        reg_out = torch.sigmoid(self.reg_head(pooled))  # (B, 2), matches normalized [0,1] targets
        return cls_out, reg_out
