"""
CNN-LSTM model for driver fatigue detection.

Architecture
------------
  CNNEncoder       : ResNet-34 backbone (pretrained ImageNet) → 512-dim per frame
  FatigueLSTMHead  : 2-layer LSTM over a sequence of CNN features → binary logit

Training is two-stage for efficiency:
  1. extract_cnn_features.py runs CNNEncoder once over all crops → .npy files
  2. train.py trains only FatigueLSTMHead on the pre-computed sequences

Inference (real-time in main.py):
  Each incoming frame is cropped and passed through CNNEncoder → 512-dim vector
  pushed into a rolling buffer.  When the buffer is full, FatigueLSTMHead
  classifies the sequence.
"""

import torch
import torch.nn as nn
from torchvision import models


class CNNEncoder(nn.Module):
    """
    ResNet-34 backbone up to the global average pool.
    Outputs a 512-dim feature vector for each input frame.
    All parameters are frozen — used purely as a fixed feature extractor.
    """
    FEATURE_DIM = 512

    def __init__(self, pretrained=True):
        super().__init__()
        weights = models.ResNet34_Weights.DEFAULT if pretrained else None
        resnet  = models.resnet34(weights=weights)
        for param in resnet.parameters():
            param.requires_grad = False
        # Drop the final FC layer; keep everything up to avgpool
        self.backbone = nn.Sequential(*list(resnet.children())[:-1])

    def forward(self, x):
        # x: (B, 3, 224, 224)
        return self.backbone(x).flatten(1)   # (B, 512)


class FatigueLSTMHead(nn.Module):
    """
    2-layer bidirectional LSTM that classifies a sequence of 512-dim CNN features.

    Input  : (batch, seq_len, 512)
    Output : (batch, 1)  — raw logit (sigmoid → probability of sleepy)
    """

    def __init__(self, feature_dim=512, hidden_size=256,
                 num_layers=2, dropout=0.4, bidirectional=False):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=feature_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=bidirectional,
        )
        lstm_out_dim = hidden_size * (2 if bidirectional else 1)
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(lstm_out_dim, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 1),
        )

    def forward(self, x):
        # x: (B, T, 512)
        out, _ = self.lstm(x)
        return self.head(out[:, -1, :])   # last time-step hidden state


def load_cnn_lstm(checkpoint_path, device="cpu"):
    """
    Load a trained FatigueLSTMHead for inference.

    Returns
    -------
    encoder   : CNNEncoder  (frozen ResNet-34, pretrained)
    lstm_head : FatigueLSTMHead  (trained weights from checkpoint)
    """
    state     = torch.load(checkpoint_path, map_location=device, weights_only=False)
    lstm_head = FatigueLSTMHead()
    lstm_head.load_state_dict(state["model_state_dict"])

    encoder   = CNNEncoder(pretrained=True).to(device).eval()
    lstm_head = lstm_head.to(device).eval()
    return encoder, lstm_head
