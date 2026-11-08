import argparse
import os
import sys
import time

import pandas as pd
import torch
import torch.nn as nn

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "data"))
sys.path.insert(0, HERE)
from split_data import get_dataloaders
from vgg16_model import get_model

SAVE_DIR = os.path.join(HERE, "saved_models")
RESULTS = os.path.join(HERE, "..", "experiments", "results")


def run_epoch(model, loader, crit, device, opt=None):
    train = opt is not None
    model.train(train)
    loss_sum, correct, n = 0.0, 0, 0
    with torch.set_grad_enabled(train):
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            out = model(x)
            loss = crit(out, y)
            if train:
                opt.zero_grad()
                loss.backward()
                opt.step()
            loss_sum += loss.item() * len(y)
            correct += (out.argmax(1) == y).sum().item()
            n += len(y)
    return loss_sum / n, correct / n


def main(epochs=10, batch_size=32, lr=1e-4):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device:", device)
    os.makedirs(SAVE_DIR, exist_ok=True)
    os.makedirs(RESULTS, exist_ok=True)

    train_loader, val_loader, _ = get_dataloaders(batch_size=batch_size, num_workers=2)
    model = get_model(freeze_features=True).to(device)
    opt = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=lr)
    crit = nn.CrossEntropyLoss()

    log, best_val = [], 0.0
    for ep in range(1, epochs + 1):
        t0 = time.time()
        tr_loss, tr_acc = run_epoch(model, train_loader, crit, device, opt)
        va_loss, va_acc = run_epoch(model, val_loader, crit, device)
        log.append(dict(epoch=ep, train_loss=tr_loss, train_acc=tr_acc,
                        val_loss=va_loss, val_acc=va_acc, time_s=time.time() - t0))
        print(f"Epoch {ep}/{epochs}: train_acc={tr_acc:.3f} val_acc={va_acc:.3f} val_loss={va_loss:.3f}")
        if va_acc > best_val:
            best_val = va_acc
            torch.save(model.state_dict(), os.path.join(SAVE_DIR, "vgg16_centralized.pth"))

    pd.DataFrame(log).to_csv(os.path.join(RESULTS, "centralized_results.csv"), index=False)
    print(f"Meilleure accuracy validation: {best_val:.4f}. Poids sauvegardes dans {SAVE_DIR}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-4)
    a = ap.parse_args()
    main(a.epochs, a.batch_size, a.lr)