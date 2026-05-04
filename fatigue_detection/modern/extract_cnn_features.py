"""
Stage 2 of the CNN-LSTM pipeline: extract 512-dim CNN features from face crops.

Runs the frozen ResNet-34 backbone over every crop saved by extract_face_crops.py
and writes one .npy file per video containing an (N_frames, 512) feature matrix.

Must be run AFTER extract_face_crops.py.

Usage (from project root, with venv active):
    python fatigue_detection/modern/extract_cnn_features.py

Output:
    data/crops/features/<video_stem>.npy   — (N_frames, 512) float32 array
    data/crops/features_manifest.csv       — maps each video to its .npy path
"""

import os
import sys
import numpy as np
import pandas as pd
import torch
from PIL import Image
from torchvision import transforms
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from fatigue_detection.modern.model import CNNEncoder

MANIFEST    = "data/crops/manifest.csv"
FEATURE_DIR = "data/crops/features"
BATCH_SIZE  = 64
DEVICE      = "mps" if torch.backends.mps.is_available() else \
              "cuda" if torch.cuda.is_available() else "cpu"

TRANSFORM = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225]),
])


def extract_video_features(crop_paths, encoder, device):
    """Return (N, 512) float32 array for an ordered list of crop paths."""
    features = []
    for start in range(0, len(crop_paths), BATCH_SIZE):
        batch_paths = crop_paths[start : start + BATCH_SIZE]
        imgs = torch.stack([
            TRANSFORM(Image.open(p).convert("RGB")) for p in batch_paths
        ]).to(device)
        with torch.no_grad():
            feats = encoder(imgs).cpu().numpy()   # (B, 512)
        features.append(feats)
    return np.concatenate(features, axis=0).astype(np.float32)


def main():
    if not os.path.exists(MANIFEST):
        print(f"ERROR: {MANIFEST} not found.")
        print("Run  python fatigue_detection/modern/extract_face_crops.py  first.")
        return

    manifest = pd.read_csv(MANIFEST)
    os.makedirs(FEATURE_DIR, exist_ok=True)

    print(f"Device  : {DEVICE}")
    print(f"Crops   : {len(manifest)}")
    print(f"Videos  : {manifest['video'].nunique()}\n")

    encoder = CNNEncoder(pretrained=True).to(DEVICE).eval()

    records = []
    videos = manifest.groupby("video")

    for i, (video, group) in enumerate(videos):
        group   = group.sort_values("frame")
        paths   = group["crop_path"].tolist()
        label   = group["label"].iloc[0]
        person  = group["person"].iloc[0]

        features = extract_video_features(paths, encoder, DEVICE)

        npy_path = os.path.join(FEATURE_DIR, f"{Path(video).stem}.npy")
        np.save(npy_path, features)

        records.append({
            "video":    video,
            "person":   person,
            "label":    label,
            "npy_path": npy_path,
            "n_frames": len(features),
        })
        print(f"  [{i+1:>3}/{len(videos)}]  {video:<40}  {len(features)} frames")

    feat_manifest = pd.DataFrame(records)
    feat_manifest.to_csv("data/crops/features_manifest.csv", index=False)

    print(f"\nDone!")
    print(f"  Feature arrays : {FEATURE_DIR}/")
    print(f"  Manifest       : data/crops/features_manifest.csv")
    print(f"  Awake videos   : {len(feat_manifest[feat_manifest['label']=='awake'])}")
    print(f"  Sleepy videos  : {len(feat_manifest[feat_manifest['label']=='sleepy'])}")


if __name__ == "__main__":
    main()
