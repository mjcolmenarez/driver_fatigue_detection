"""
Train the CNN-LSTM fatigue detector (LSTM head on pre-computed CNN features).

Pre-requisites:
    python fatigue_detection/modern/extract_face_crops.py
    python fatigue_detection/modern/extract_cnn_features.py

Usage (from project root, with venv active):
    python fatigue_detection/modern/train.py

Output:
    fatigue_detection/modern/cnn_lstm_model.pth
"""

import sys
from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from fatigue_detection.modern.dataset import make_splits
from fatigue_detection.modern.model import FatigueLSTMHead

# ── Hyperparameters ──────────────────────────────────────────────────────────
DEVICE     = "mps" if torch.backends.mps.is_available() else \
             "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 64
EPOCHS     = 40
PATIENCE   = 8
LR         = 5e-4
CHECKPOINT = "fatigue_detection/modern/cnn_lstm_model.pth"
NUM_WORKERS = 0   # set to 4 if on Linux/CUDA
# ─────────────────────────────────────────────────────────────────────────────


def run_epoch(model, loader, optimizer, criterion, device, train=True):
    model.train(train)
    total_loss = correct = total = 0
    ctx = torch.enable_grad() if train else torch.no_grad()
    with ctx:
        for X, y in loader:
            X, y = X.to(device), y.to(device).unsqueeze(1)
            if train:
                optimizer.zero_grad()
            logits = model(X)
            loss   = criterion(logits, y)
            if train:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
            total_loss += loss.item() * X.size(0)
            preds       = (torch.sigmoid(logits) > 0.5).float()
            correct    += (preds == y).sum().item()
            total      += X.size(0)
    return total_loss / total, correct / total


def main():
    print(f"Device : {DEVICE}\n")

    train_ds, val_ds, _, _ = make_splits()

    # Class weights to handle imbalance (more sleepy than awake in the dataset)
    n_awake  = int((train_ds.y == 0).sum())
    n_sleepy = int((train_ds.y == 1).sum())
    pos_weight = torch.tensor([n_awake / n_sleepy], dtype=torch.float32).to(DEVICE)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=NUM_WORKERS)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False,
                              num_workers=NUM_WORKERS)

    model     = FatigueLSTMHead().to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", patience=4, factor=0.5
    )
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    best_val_acc     = 0.0
    patience_counter = 0

    print(f"Training CNN-LSTM head for up to {EPOCHS} epochs "
          f"(early stop patience={PATIENCE})\n")
    header = f"{'Epoch':>5}  {'TrLoss':>7}  {'TrAcc':>6}  {'VaLoss':>7}  {'VaAcc':>6}"
    print(header)
    print("-" * len(header))

    for epoch in range(1, EPOCHS + 1):
        tr_loss, tr_acc = run_epoch(model, train_loader, optimizer,
                                    criterion, DEVICE, train=True)
        va_loss, va_acc = run_epoch(model, val_loader,   optimizer,
                                    criterion, DEVICE, train=False)
        scheduler.step(va_loss)

        marker = ""
        if va_acc > best_val_acc:
            best_val_acc     = va_acc
            patience_counter = 0
            torch.save({
                "epoch":            epoch,
                "model_state_dict": model.state_dict(),
                "val_acc":          va_acc,
                "val_loss":         va_loss,
            }, CHECKPOINT)
            marker = "  ← best"
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                print(f"\nEarly stopping at epoch {epoch}.")
                break

        print(f"{epoch:>5}  {tr_loss:>7.4f}  {tr_acc:>6.3f}  "
              f"{va_loss:>7.4f}  {va_acc:>6.3f}{marker}")

    print(f"\nBest val accuracy : {best_val_acc:.3f}")
    print(f"Checkpoint saved  : {CHECKPOINT}")


if __name__ == "__main__":
    main()
