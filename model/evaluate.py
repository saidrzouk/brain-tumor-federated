"""
Script d'évaluation centralisée (Brain Tumor MRI) :

Rôle : Charge un modèle entraîné (par défaut le modèle centralisé) et calcule 
toutes les métriques de performance sur l'ensemble de test :
- Accuracy et Loss globale.
- Précision, Recall, et F1-score (macro et par classe).
- Matrice de confusion détaillée pour l'analyse des 4 classes de tumeurs.
"""

import argparse
import os
import sys

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (accuracy_score, confusion_matrix,
                             precision_recall_fscore_support)

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "data"))
sys.path.insert(0, HERE)
from data.split_data import get_dataloaders, CLASSES
from vgg16_model import get_model


def evaluate_model(model, loader, device="cpu"):
    """Retourne accuracy, loss, precision/recall/F1 (macro et par classe) et matrice de confusion."""
    model.to(device).eval()
    crit = nn.CrossEntropyLoss()
    loss_sum, n = 0.0, 0
    preds, labels = [], []
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            out = model(x)
            loss_sum += crit(out, y).item() * len(y)
            n += len(y)
            preds += out.argmax(1).cpu().tolist()
            labels += y.cpu().tolist()
    ids = list(range(len(CLASSES)))
    p, r, f1, _ = precision_recall_fscore_support(labels, preds, labels=ids, zero_division=0)
    return {
        "accuracy": accuracy_score(labels, preds),
        "loss": loss_sum / n,
        "precision": float(np.mean(p)),
        "recall": float(np.mean(r)),
        "f1": float(np.mean(f1)),
        "per_class": {c: {"precision": float(p[i]), "recall": float(r[i]), "f1": float(f1[i])}
                      for i, c in enumerate(CLASSES)},
        "confusion_matrix": confusion_matrix(labels, preds, labels=ids),
    }


def print_metrics(m):
    print(f"Accuracy : {m['accuracy']:.4f} | Loss : {m['loss']:.4f}")
    print(f"Precision: {m['precision']:.4f} | Recall: {m['recall']:.4f} | F1: {m['f1']:.4f}  (macro)")
    for c, v in m["per_class"].items():
        print(f"  {c:12s} P={v['precision']:.2f} R={v['recall']:.2f} F1={v['f1']:.2f}")
    print("Matrice de confusion :")
    print(m["confusion_matrix"])


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default=os.path.join(HERE, "saved_models", "vgg16_centralized.pth"))
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = get_model(pretrained=False)
    model.load_state_dict(torch.load(args.weights, map_location=device))
    _, _, test_loader = get_dataloaders(batch_size=32)
    print_metrics(evaluate_model(model, test_loader, device))