
"""
Script de simulation fédérée manuelle et séquentielle pour Google Colab (Brain Tumor MRI) :

Rôle : Permet d'exécuter et de valider les 10 rounds d'apprentissage fédéré sur Colab sans déclencher les erreurs de mémoire (OOM) causées par le moteur Ray par défaut.

Fonctionnement de la simulation :
1. Initialisation : Charge le modèle global `BrainTumorVGG16` sur GPU (ou CPU).
2. Boucle des 10 rounds :
   - Entraîne séquentiellement les 3 clients IID (`CLIENT_ID` 1, 2 et 3) via `client.fit()`.
   - Libère explicitement la mémoire GPU et RAM (`torch.cuda.empty_cache()`, `gc.collect()`) après chaque client.
3. Agrégation manuelle (`FedAvg`) : Calcule la moyenne pondérée des poids des 3 clients pour mettre à jour le modèle global.
4. Évaluation globale : Teste le modèle agrégé sur les jeux de test des 3 clients et affiche la Loss et l'Accuracy globales à chaque round.
"""

"""
# 1. Extraction dyal le fichier ZIP li fih le projet vers le dossier /content/project
!unzip brain-tumor-federated.zip -d /content/project

# 2. Déplacement vers le dossier principal du projet pour exécuter le code
%cd /content/project/brain-tumor-federated

# 3. Installation de toutes les dépendances requises (PyTorch, Torchvision, NumPy, etc.)
!pip install -r requirements.txt

# 4. Installation / Mise à jour du framework Flower avec le support de simulation
!pip install -U "flwr[simulation]"

# 5. Confirmation du répertoire de travail avant de lancer la simulation
# %cd /content/project/brain-tumor-federated

# Commented out IPython magic to ensure Python compatibility.
%cd /content/project/brain-tumor-federated
"""

import os
import sys
import torch
import gc
import copy

project_path = "/content/project/brain-tumor-federated"
if project_path not in sys.path:
    sys.path.append(project_path)

from data.split_data import get_client_data, get_dataloaders
from federated.client import BrainTumorClient
from model.vgg16_model import BrainTumorVGG16
from federated.flower_config import NUM_ROUNDS, NUM_CLIENTS, BATCH_SIZE

# 1. Initialisation du Modèle Global
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
global_model = BrainTumorVGG16().to(device)

print("Starting Manual Federated Learning Loop with Evaluation & 10 Rounds...\n")

for round_idx in range(NUM_ROUNDS):
    print(f"\n--- [ROUND {round_idx + 1}/{NUM_ROUNDS}] ---")

    client_weights = []
    client_lens = []

    # Récupérer les poids globaux actuels
    global_state = copy.deepcopy(global_model.state_dict())
    global_params = [val.cpu().numpy() for val in global_state.values()]

    # 2. Entraînement local (Fit) sur chaque client
    for client_id in range(1, NUM_CLIENTS + 1):
        print(f"  -> Training Client {client_id}...")
        client_samples = get_client_data(client_id)
        train_loader, val_loader, test_loader = get_dataloaders(batch_size=BATCH_SIZE, train_samples=client_samples)

        client_model = BrainTumorVGG16()
        client = BrainTumorClient(client_model, train_loader, val_loader, test_loader)

        updated_params, num_samples, metrics = client.fit(global_params, {})

        client_weights.append(updated_params)
        client_lens.append(num_samples)

        # Nettoyage mémoire après chaque client
        del client, client_model, train_loader, val_loader, test_loader
        torch.cuda.empty_cache()
        gc.collect()

    # 3. Agrégation FedAvg (Moyenne pondérée des poids)
    print("  -> Aggregating weights (FedAvg)...")
    total_samples = sum(client_lens)
    new_global_params = []

    for layer_idx in range(len(client_weights[0])):
        layer_avg = sum(client_weights[c][layer_idx] * client_lens[c] for c in range(NUM_CLIENTS)) / total_samples
        new_global_params.append(layer_avg)

    # Mettre à jour le modèle global temporairement pour l'évaluation
    new_state_dict = {}
    for (k, v), param in zip(global_model.state_dict().items(), new_global_params):
        new_state_dict[k] = torch.tensor(param).to(device)
    global_model.load_state_dict(new_state_dict)

    # 4. Évaluation du modèle global agrégé sur les jeux de test de tous les clients
    print("  -> Evaluating Global Model...")
    total_test_samples = 0
    weighted_loss = 0.0
    weighted_accuracy = 0.0

    for client_id in range(1, NUM_CLIENTS + 1):
        client_samples = get_client_data(client_id)
        _, _, test_loader = get_dataloaders(batch_size=BATCH_SIZE, train_samples=client_samples)

        client_model = BrainTumorVGG16()
        # On passe test_loader au client pour l'évaluation
        client = BrainTumorClient(client_model, None, None, test_loader)

        loss, num_test_samples, eval_metrics = client.evaluate(new_global_params, {})

        weighted_loss += loss * num_test_samples
        weighted_accuracy += eval_metrics["accuracy"] * num_test_samples
        total_test_samples += num_test_samples

        del client, client_model, test_loader
        torch.cuda.empty_cache()
        gc.collect()

    # Calcul des métriques globales (Moyenne pondérée)
    global_loss = weighted_loss / total_test_samples
    global_accuracy = weighted_accuracy / total_test_samples

    print(f">>> [ROUND {round_idx + 1} RESULT] Global Loss: {global_loss:.4f} | Global Accuracy: {global_accuracy * 100:.2f}%")

print("\nFederated Learning Simulation Finished Successfully with Full Evaluation!")



