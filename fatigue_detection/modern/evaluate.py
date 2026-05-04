"""
Evaluate the CNN-LSTM and compare with the classical SVM.

Usage (from project root, with venv active):
    python fatigue_detection/modern/evaluate.py
"""

import sys
import time
from pathlib import Path
from collections import defaultdict

import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from fatigue_detection.modern.dataset import make_splits
from fatigue_detection.modern.model import load_cnn_lstm

DEVICE     = "mps" if torch.backends.mps.is_available() else \
             "cuda" if torch.cuda.is_available() else "cpu"
CHECKPOINT = "fatigue_detection/modern/cnn_lstm_model.pth"

# Classical SVM results from train_classifier_all.py
SVM_FRAME_ACC = 0.843
SVM_VIDEO_ACC = 0.917
SVM_MS_FRAME  = 2.0


def predict(lstm_head, loader, device):
    lstm_head.eval()
    probs, labels = [], []
    with torch.no_grad():
        for X, y in loader:
            p = torch.sigmoid(lstm_head(X.to(device))).squeeze(1).cpu().numpy()
            probs.extend(p)
            labels.extend(y.numpy())
    return np.array(probs), np.array(labels)


def per_video_accuracy(probs, labels, video_ids):
    vid_probs  = defaultdict(list)
    vid_labels = {}
    for p, l, v in zip(probs, labels, video_ids):
        vid_probs[v].append(p)
        vid_labels[v] = int(l)
    correct = sum(
        int((np.mean(ps) > 0.5) == vid_labels[v])
        for v, ps in vid_probs.items()
    )
    return correct / len(vid_probs)


def main():
    if not Path(CHECKPOINT).exists():
        print(f"Checkpoint not found: {CHECKPOINT}")
        print("Run  python fatigue_detection/modern/train.py  first.")
        return

    _, _, test_ds, test_vid_ids = make_splits()
    test_loader = DataLoader(test_ds, batch_size=128, shuffle=False, num_workers=0)

    _, lstm_head = load_cnn_lstm(CHECKPOINT, device=DEVICE)

    # Warm-up pass
    dummy = next(iter(test_loader))[0][:1].to(DEVICE)
    lstm_head(dummy)

    t0 = time.perf_counter()
    probs, labels = predict(lstm_head, test_loader, DEVICE)
    ms_per_seq = (time.perf_counter() - t0) / len(test_ds) * 1000

    preds     = (probs > 0.5).astype(int)
    frame_acc = (preds == labels).mean()
    video_acc = per_video_accuracy(probs, labels, test_vid_ids)
    auc       = roc_auc_score(labels, probs)

    W = 62
    print("=" * W)
    print("RESULTS COMPARISON")
    print("=" * W)
    print(f"\n{'Metric':<28} {'SVM (Classical)':>16} {'CNN-LSTM (Modern)':>15}")
    print("-" * W)
    print(f"{'Per-sequence accuracy':<28} {SVM_FRAME_ACC*100:>15.1f}% {frame_acc*100:>14.1f}%")
    print(f"{'Per-video accuracy':<28} {SVM_VIDEO_ACC*100:>15.1f}% {video_acc*100:>14.1f}%")
    print(f"{'AUC-ROC':<28} {'N/A':>16} {auc:>15.3f}")
    print(f"{'Inference / sequence':<28} {SVM_MS_FRAME:>14.1f}ms {ms_per_seq:>14.1f}ms")
    print("=" * W)

    print("\nCNN-LSTM Classification Report:")
    print(classification_report(labels, preds, target_names=["awake", "sleepy"]))

    cm = confusion_matrix(labels, preds)
    print("Confusion Matrix (CNN-LSTM):")
    print(f"                 Pred Awake  Pred Sleepy")
    print(f"  True Awake       {cm[0,0]:>6}       {cm[0,1]:>6}")
    print(f"  True Sleepy      {cm[1,0]:>6}       {cm[1,1]:>6}")


if __name__ == "__main__":
    main()
