"""
Simulation fédérée manuelle et séquentielle — Brain Tumor MRI.

Rôle : Exécute NUM_ROUNDS rounds d'apprentissage fédéré sur une seule
       machine (Colab ou locale) sans serveur Flower ni moteur Ray,
       ce qui évite les crashs OOM typiques de Colab.

Fonctionnement :
  1. Initialisation   — Charge BrainTumorVGG16 sur GPU/CPU.
  2. Fit séquentiel   — Entraîne chaque client l'un après l'autre et
                        libère la VRAM/RAM entre chaque client.
  3. FedAvg           — Moyenne pondérée des poids par nombre d'exemples.
  4. Évaluation       — Teste le modèle agrégé sur chaque jeu de test
                        local et affiche les métriques globales.

─── Setup Colab (exécuter dans des cellules séparées) ──────────────────────
  !unzip brain-tumor-federated.zip -d /content/project
  %cd /content/project/brain-tumor-federated
  !pip install -r requirements.txt
  !pip install -U "flwr[simulation]"
────────────────────────────────────────────────────────────────────────────
"""

import gc
import os
import sys

import numpy as np
import torch

# ── Résolution du chemin projet ───────────────────────────────────────────────
# Fonctionne que le script soit lancé depuis Colab, en local, ou importé.
PROJECT_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_PATH not in sys.path:
    sys.path.insert(0, PROJECT_PATH)

from data.split_data import get_client_data, get_dataloaders
from federated.client import BrainTumorClient
from federated.flower_config import BATCH_SIZE, NUM_CLIENTS, NUM_ROUNDS
from model.vgg16_model import BrainTumorVGG16


# ── Utilitaires ───────────────────────────────────────────────────────────────

def get_model_params(model):
    """Extrait les paramètres d'un modèle sous forme de liste de tableaux NumPy (copies sûres)."""
    return [val.cpu().numpy().copy() for val in model.state_dict().values()]


def set_model_params(model, params, device):
    """Charge une liste de tableaux NumPy dans le state_dict d'un modèle."""
    state_dict = {
        key: torch.tensor(param, device=device)
        for key, param in zip(model.state_dict().keys(), params)
    }
    model.load_state_dict(state_dict)


def fedavg(weights_list, num_samples_list):
    """
    Agrégation FedAvg : moyenne pondérée des paramètres clients.

    Args:
        weights_list     : liste de [params_client_0, params_client_1, ...]
                           chaque élément est une liste de tableaux NumPy.
        num_samples_list : nombre d'exemples d'entraînement par client.

    Returns:
        Liste de tableaux NumPy représentant les poids agrégés.
    """
    total = sum(num_samples_list)
    num_layers = len(weights_list[0])

    return [
        np.sum(
            [weights_list[c][i] * (num_samples_list[c] / total)
             for c in range(len(weights_list))],
            axis=0,
        )
        for i in range(num_layers)
    ]


def cleanup(device, *objects):
    """Supprime les objets passés, vide la VRAM et force le GC."""
    for obj in objects:
        del obj
    if device.type == "cuda":
        torch.cuda.empty_cache()
    gc.collect()


# ── Simulation principale ─────────────────────────────────────────────────────

def run_simulation():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Dispositif utilisé : {device}")
    print(f"Démarrage : {NUM_ROUNDS} rounds, {NUM_CLIENTS} clients.\n")

    global_model = BrainTumorVGG16().to(device)

    for round_idx in range(NUM_ROUNDS):
        print(f"\n{'=' * 55}")
        print(f"  ROUND {round_idx + 1} / {NUM_ROUNDS}")
        print(f"{'=' * 55}")

        global_params = get_model_params(global_model)
        client_weights = []
        client_lens = []

        # ── Phase d'entraînement ─────────────────────────────────────────────
        for client_id in range(1, NUM_CLIENTS + 1):
            print(f"  [Train] Client {client_id}...")

            client_samples = get_client_data(client_id)
            train_loader, val_loader, test_loader = get_dataloaders(
                batch_size=BATCH_SIZE,
                train_samples=client_samples,
            )

            client_model = BrainTumorVGG16()
            client = BrainTumorClient(
                model=client_model,
                train_loader=train_loader,
                val_loader=val_loader,
                test_loader=test_loader,
            )

            updated_params, num_samples, metrics = client.fit(global_params, {})
            train_loss = metrics.get("train_loss", float("nan"))
            print(f"           exemples={num_samples}  |  loss={train_loss:.4f}")

            client_weights.append(updated_params)
            client_lens.append(num_samples)

            cleanup(device, client, client_model, train_loader, val_loader, test_loader)

        # ── Agrégation FedAvg ────────────────────────────────────────────────
        print("  [Agrégation] FedAvg...")
        new_global_params = fedavg(client_weights, client_lens)
        set_model_params(global_model, new_global_params, device)

        # ── Phase d'évaluation ───────────────────────────────────────────────
        print("  [Évaluation] Modèle global sur tous les jeux de test...")
        total_test_samples = 0
        weighted_loss = 0.0
        weighted_accuracy = 0.0

        for client_id in range(1, NUM_CLIENTS + 1):
            # On réutilise le test_loader déjà créé pendant le fit serait idéal,
            # mais comme cleanup() l'a supprimé, on le recrée ici.
            # Le seul loader nécessaire pour evaluate() est test_loader.
            client_samples = get_client_data(client_id)
            _, _, test_loader = get_dataloaders(
                batch_size=BATCH_SIZE,
                train_samples=client_samples,
            )

            client_model = BrainTumorVGG16()
            client = BrainTumorClient(
                model=client_model,
                train_loader=None,
                val_loader=None,
                test_loader=test_loader,
            )

            loss, n, eval_metrics = client.evaluate(new_global_params, {})
            accuracy = eval_metrics["accuracy"]
            print(
                f"           Client {client_id}: "
                f"loss={loss:.4f}  |  acc={accuracy * 100:.2f}%"
            )

            weighted_loss += loss * n
            weighted_accuracy += accuracy * n
            total_test_samples += n

            cleanup(device, client, client_model, test_loader)

        # Métriques globales (moyenne pondérée sur tous les clients)
        global_loss = weighted_loss / total_test_samples
        global_accuracy = weighted_accuracy / total_test_samples

        print(
            f"\n  >>> ROUND {round_idx + 1} RÉSULTAT : "
            f"Loss={global_loss:.4f}  |  Accuracy={global_accuracy * 100:.2f}%"
        )

    print("\nSimulation fédérée terminée avec succès !")


if __name__ == "__main__":
    run_simulation()
