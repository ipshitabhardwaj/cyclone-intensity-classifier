"""
Temporal forecasting model: given the last SEQ_LEN observed frames of a
storm, predict wind speed at +6h / +12h / +24h ahead.

Reuses the ALREADY-TRAINED CycloneNet classifier's conv backbone as a
per-frame feature extractor (loaded from checkpoints_combined/best.pt)
rather than learning image features from scratch a second time -- with only
~700 training sequences (11 storms) there isn't nearly enough data to train
a second CNN, but there IS enough to train a small GRU on top of features
the classifier already learned from all 1,032 images.

Architecture:
  per-frame image (2,H,W) --[frozen CycloneNet backbone]--> 1280-d embedding
  [embedding ++ normalized elapsed-hours] for each of SEQ_LEN frames
    --[GRU]--> final hidden state
    --[Linear]--> 3 outputs (sigmoid), one per horizon in HORIZONS_HOURS
"""
import os
import sys

import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(__file__))
from model import CycloneNet
from temporal_dataset import HORIZONS_HOURS


class TemporalCycloneNet(nn.Module):
    def __init__(self, backbone_ckpt: str, freeze_backbone: bool = True,
                 gru_hidden: int = 128, num_horizons: int = len(HORIZONS_HOURS)):
        super().__init__()

        base = CycloneNet(in_channels=2, num_classes=8, pretrained=False, num_reg_outputs=1)
        if backbone_ckpt and os.path.exists(backbone_ckpt):
            ckpt = torch.load(backbone_ckpt, map_location="cpu", weights_only=False)
            base.load_state_dict(ckpt["model_state"])
            print(f"[temporal_model] loaded backbone weights from {backbone_ckpt} "
                  f"(classifier val_acc={ckpt.get('val_acc'):.3f} @ epoch {ckpt.get('epoch')})")
        else:
            print(f"[temporal_model] WARNING: no backbone checkpoint found at "
                  f"{backbone_ckpt!r} -- using randomly initialized features")

        self.backbone_features = base.backbone_features
        self.avgpool = base.avgpool
        self.feat_dim = 1280  # EfficientNet-B0

        self.freeze_backbone = freeze_backbone
        if freeze_backbone:
            for p in self.backbone_features.parameters():
                p.requires_grad = False

        self.gru = nn.GRU(
            input_size=self.feat_dim + 1,  # + delta-hours feature
            hidden_size=gru_hidden,
            num_layers=1,
            batch_first=True,
        )
        # predicts a DELTA (km/h change from the current, last-observed wind
        # speed) rather than an absolute value. The head sees the current
        # wind speed explicitly (concatenated in) so it can learn "how much
        # is this storm likely to intensify/weaken", which is a much better-
        # posed regression target than absolute wind speed -- an image-only
        # model with no explicit "current state" input was, in practice,
        # regressing towards the dataset's average wind speed and losing to
        # a trivial persistence baseline at +6h/+12h. tanh bounds the
        # predicted change to a physically sane range (+-150 km/h) so early
        # untrained outputs can't blow up the loss.
        self.head = nn.Sequential(
            nn.Linear(gru_hidden + 1, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, num_horizons),
        )
        self.max_delta_kmph = 150.0

    def encode_frame(self, x_frame):
        """x_frame: (B,2,H,W) -> (B, feat_dim)"""
        feats = self.backbone_features(x_frame)
        return self.avgpool(feats).flatten(1)

    def forward(self, x_seq, dt_feat, anchor_kmph_norm):
        """x_seq: (B, T, 2, H, W)   dt_feat: (B, T)   anchor_kmph_norm: (B,)
        current wind speed (normalized 0-1, same range as kaggle_dataset's
        WIND_MIN/WIND_MAX), the reference point the delta is predicted from.

        Returns predicted DELTA in km/h, shape (B, num_horizons). Add this to
        the real anchor_kmph to get the forecast; see train_temporal.py.
        """
        b, t = x_seq.shape[0], x_seq.shape[1]
        x_flat = x_seq.view(b * t, *x_seq.shape[2:])

        if self.freeze_backbone:
            with torch.no_grad():
                feats = self.encode_frame(x_flat)
        else:
            feats = self.encode_frame(x_flat)

        feats = feats.view(b, t, self.feat_dim)
        seq_in = torch.cat([feats, dt_feat.unsqueeze(-1)], dim=-1)  # (B,T,feat_dim+1)

        _, h_n = self.gru(seq_in)          # h_n: (1, B, gru_hidden)
        h_last = h_n[-1]                    # (B, gru_hidden)
        h_with_anchor = torch.cat([h_last, anchor_kmph_norm.unsqueeze(-1)], dim=-1)
        delta_kmph = torch.tanh(self.head(h_with_anchor)) * self.max_delta_kmph
        return delta_kmph
