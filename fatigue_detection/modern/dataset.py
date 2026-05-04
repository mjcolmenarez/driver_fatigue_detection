"""
Sequence dataset for CNN-LSTM fatigue classification.

Loads pre-computed 512-dim CNN feature vectors (one .npy per video, produced by
extract_cnn_features.py) and builds sliding-window sequences.
Split is by video — same anti-leakage principle as the classical pipeline.
"""

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

SEQ_LEN  = 16   # frames per sequence (~2.7 seconds at 6 fps)
STRIDE   = 4    # step between consecutive sequences
FEAT_DIM = 512
MANIFEST = "data/crops/features_manifest.csv"


class SequenceDataset(Dataset):
    def __init__(self, sequences, labels):
        self.X = torch.from_numpy(sequences.astype(np.float32))
        self.y = torch.from_numpy(labels.astype(np.float32))

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def _extract_sequences(feat_manifest, video_set, seq_len, stride):
    seqs, labels, video_ids = [], [], []
    subset = feat_manifest[feat_manifest["video"].isin(video_set)]
    for _, row in subset.iterrows():
        features = np.load(row["npy_path"]).astype(np.float32)  # (N, 512)
        y = 1 if row["label"] == "sleepy" else 0
        for start in range(0, len(features) - seq_len + 1, stride):
            seqs.append(features[start : start + seq_len])
            labels.append(y)
            video_ids.append(row["video"])
    if not seqs:
        return (np.empty((0, seq_len, FEAT_DIM), dtype=np.float32),
                np.array([]), [])
    return np.stack(seqs), np.array(labels, dtype=np.float32), video_ids


def make_splits(manifest_path=MANIFEST, seq_len=SEQ_LEN, stride=STRIDE, seed=42):
    """
    Build train / val / test splits from the CNN feature manifest.

    Returns
    -------
    train_ds, val_ds, test_ds : SequenceDataset
    test_video_ids            : list[str]
    """
    if not __import__("os").path.exists(manifest_path):
        raise FileNotFoundError(
            f"{manifest_path} not found.\n"
            "Run extract_face_crops.py then extract_cnn_features.py first."
        )

    feat_manifest = pd.read_csv(manifest_path)
    rng = np.random.default_rng(seed)

    train_vids, val_vids, test_vids = set(), set(), set()

    for person in feat_manifest["person"].unique():
        for label in ["awake", "sleepy"]:
            mask   = (feat_manifest["person"] == person) & \
                     (feat_manifest["label"]  == label)
            videos = rng.permutation(feat_manifest[mask]["video"].unique())
            n = len(videos)
            n_train = max(1, int(0.70 * n))
            n_val   = max(1, int(0.15 * n))
            if n_train + n_val >= n:
                n_val = max(0, n - n_train - 1)
            train_vids.update(videos[:n_train])
            val_vids.update(videos[n_train : n_train + n_val])
            test_vids.update(videos[n_train + n_val :])

    train_X, train_y, _            = _extract_sequences(feat_manifest, train_vids, seq_len, stride)
    val_X,   val_y,   _            = _extract_sequences(feat_manifest, val_vids,   seq_len, stride)
    test_X,  test_y,  test_vid_ids = _extract_sequences(feat_manifest, test_vids,  seq_len, stride)

    def _fmt(y):
        return f"{int((y==0).sum())} awake / {int((y==1).sum())} sleepy"

    print(f"Train : {len(train_y):>5} sequences  ({_fmt(train_y)})")
    print(f"Val   : {len(val_y):>5} sequences  ({_fmt(val_y)})")
    print(f"Test  : {len(test_y):>5} sequences  ({_fmt(test_y)})")

    return (SequenceDataset(train_X, train_y),
            SequenceDataset(val_X,   val_y),
            SequenceDataset(test_X,  test_y),
            test_vid_ids)
