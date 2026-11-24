"""
Client Flower pour l'apprentissage fédéré (Brain Tumor MRI) :

Rôle : Représente un hôpital local qui entraîne et évalue le modèle sans partager ses images IRM.

Fonctionnement du client (`BrainTumorClient`) :
1. Réception (`set_parameters`) : Charge les poids du modèle global envoyés par le serveur.
2. Entraînement local (`fit`) :
   - Gèle les couches convolutives (`features`) de VGG16 pour réduire le coût de communication.
   - Entraîne uniquement le classifieur sur les données locales de l'hôpital (optimiseur Adam, lr=0.0001).
   - Nettoie la mémoire (VRAM GPU et RAM CPU) après l'entraînement pour éviter les crashs OOM.
3. Transmission (`get_parameters`) : Renvoie uniquement les poids mis à jour au serveur.
4. Évaluation (`evaluate`) : Calcule la loss et l'accuracy du modèle global sur le jeu de test local.
5. Connexion réseau (`main`) : Lit la variable d'environnement `CLIENT_ID` et se connecte au serveur Flower via gRPC.
"""

import gc
import os
from collections import OrderedDict

import flwr as fl
import torch
import torch.nn as nn
import torch.optim as optim

from data.split_data import get_client_data, get_dataloaders
from federated.flower_config import (
    BATCH_SIZE,
    LEARNING_RATE,
    LOCAL_EPOCHS,
    SERVER_ADDRESS,
)
from model.vgg16_model import BrainTumorVGG16


class BrainTumorClient(fl.client.NumPyClient):
    def __init__(
        self,
        model,
        train_loader,
        val_loader=None,
        test_loader=None,
    ):
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.model = model.to(self.device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader

        # Geler les couches convolutives — seuls les poids du classifieur
        # sont échangés avec le serveur, ce qui réduit le coût de communication.
        for param in self.model.features.parameters():
            param.requires_grad = False

        # AMP : accélère l'entraînement sur GPU et réduit l'utilisation VRAM.
        # Utilise l'API non dépréciée (PyTorch >= 2.0).
        self.use_amp = self.device.type == "cuda"
        self.scaler = torch.amp.GradScaler("cuda", enabled=self.use_amp)

    # ── Paramètres ────────────────────────────────────────────────────────────

    def get_parameters(self, config):
        """Retourne les paramètres du modèle en tant que liste de tableaux NumPy."""
        return [
            val.detach().cpu().numpy().copy()   # .copy() évite le partage mémoire
            for val in self.model.state_dict().values()
        ]

    def set_parameters(self, parameters):
        """Charge les paramètres envoyés par le serveur dans le modèle local."""
        state_dict = OrderedDict(
            (key, torch.as_tensor(value, device=self.device))
            for key, value in zip(self.model.state_dict().keys(), parameters)
        )
        self.model.load_state_dict(state_dict, strict=True)

    # ── Entraînement ──────────────────────────────────────────────────────────

    def fit(self, parameters, config):
        """Entraînement local sur les données de l'hôpital."""
        self.set_parameters(parameters)
        self.model.train()

        trainable_params = [
            p for p in self.model.parameters() if p.requires_grad
        ]
        if not trainable_params:
            raise RuntimeError("Aucun paramètre entraînable dans le modèle.")

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(trainable_params, lr=LEARNING_RATE)

        total_loss = 0.0
        num_batches = 0

        try:
            for _ in range(LOCAL_EPOCHS):
                for images, labels in self.train_loader:
                    images = images.to(self.device, non_blocking=True)
                    labels = labels.to(self.device, non_blocking=True)

                    optimizer.zero_grad(set_to_none=True)

                    # torch.amp.autocast remplace l'API dépréciée torch.cuda.amp.autocast
                    with torch.amp.autocast("cuda", enabled=self.use_amp):
                        outputs = self.model(images)
                        loss = criterion(outputs, labels)

                    if self.use_amp:
                        self.scaler.scale(loss).backward()
                        self.scaler.step(optimizer)
                        self.scaler.update()
                    else:
                        loss.backward()
                        optimizer.step()

                    total_loss += loss.item()
                    num_batches += 1

            avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
            num_examples = len(self.train_loader.dataset)

            return (
                self.get_parameters({}),
                num_examples,
                {"train_loss": float(avg_loss)},   # métriques visibles côté serveur
            )

        finally:
            # Nettoyage explicite pour éviter les OOM entre les rounds
            del optimizer, criterion
            if self.device.type == "cuda":
                torch.cuda.empty_cache()
            gc.collect()

    # ── Évaluation ────────────────────────────────────────────────────────────

    def evaluate(self, parameters, config):
        """Évalue le modèle global sur le jeu de test local."""
        if self.test_loader is None:
            raise ValueError("test_loader est requis pour l'évaluation.")

        self.set_parameters(parameters)
        self.model.eval()

        # reduction="sum" permet d'accumuler la loss sur tous les exemples
        # puis de diviser une seule fois — plus précis que la moyenne de moyennes.
        criterion = nn.CrossEntropyLoss(reduction="sum")
        total_loss = 0.0
        total_correct = 0
        total_examples = 0

        # inference_mode est plus rapide que no_grad (pas de version graph)
        with torch.inference_mode():
            for images, labels in self.test_loader:
                images = images.to(self.device, non_blocking=True)
                labels = labels.to(self.device, non_blocking=True)

                outputs = self.model(images)
                total_loss += criterion(outputs, labels).item()
                total_correct += (outputs.argmax(dim=1) == labels).sum().item()
                total_examples += labels.size(0)

        if total_examples == 0:
            raise ValueError("Le jeu de test est vide.")

        avg_loss = total_loss / total_examples
        accuracy = total_correct / total_examples

        return float(avg_loss), total_examples, {"accuracy": float(accuracy)}


# ── Point d'entrée ────────────────────────────────────────────────────────────

def main():
    client_id = int(os.getenv("CLIENT_ID", "1"))
    print(f"Démarrage du client Flower {client_id}...")

    model = BrainTumorVGG16()

    print(f"Chargement des données du client {client_id}...")
    client_samples = get_client_data(client_id)
    train_loader, val_loader, test_loader = get_dataloaders(
        batch_size=BATCH_SIZE,
        train_samples=client_samples,
    )

    client = BrainTumorClient(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
    )

    # Remplacer 0.0.0.0 par 127.0.0.1 pour la connexion client locale
    server_address = SERVER_ADDRESS
    if server_address.startswith("0.0.0.0"):
        server_address = server_address.replace("0.0.0.0", "127.0.0.1", 1)

    print(f"Connexion au serveur Flower : {server_address}")

    # start_numpy_client est déprécié depuis Flower 1.0 — utiliser start_client + to_client()
    fl.client.start_client(
        server_address=server_address,
        client=client.to_client(),
    )


if __name__ == "__main__":
    main()
