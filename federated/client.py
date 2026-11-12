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

# %%writefile federated/client.py

import os
import gc  # Ajouté pour vider la RAM Python
import torch
import torch.nn as nn
import torch.optim as optim
from collections import OrderedDict
import flwr as fl

# Import d l-modèle
model_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
from model.vgg16_model import BrainTumorVGG16

# Import s-sahiḥ mn data/split_data.py (kansta3mlo get_client_data w get_dataloaders)
from data.split_data import get_client_data, get_dataloaders
from federated.flower_config import BATCH_SIZE, SERVER_ADDRESS,LEARNING_RATE

class BrainTumorClient(fl.client.NumPyClient):
    def __init__(self, model, train_loader, val_loader, test_loader, num_classes=4):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        
        # Training ghir 3la l-classifier layer bach n-hebtou communication cost
        for param in self.model.features.parameters():
            param.requires_grad = False

    def get_parameters(self, config):
        # Kan-t-akdo bli kol weight t-convertar l numpy array b njaḥ
        return [val.detach().cpu().numpy() for _, val in self.model.state_dict().items()]

    def set_parameters(self, parameters):
        params_dict = zip(self.model.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        self.model.load_state_dict(state_dict, strict=True)

    def fit(self, parameters, config):
        try:
            self.set_parameters(parameters)
            self.model.train()
            criterion = nn.CrossEntropyLoss()
            optimizer = optim.Adam(filter(lambda p: p.requires_grad, self.model.parameters()), lr=LEARNING_RATE)

            # Local training epochs
            for epoch in range(1):
                for images, labels in self.train_loader:
                    images, labels = images.to(self.device), labels.to(self.device)
                    optimizer.zero_grad()
                    outputs = self.model(images)
                    loss = criterion(outputs, labels)
                    loss.backward()
                    optimizer.step()

            # ==========================================================
            # VIDAGE DE LA MÉMOIRE (GPU VRAM & RAM CPU) POUR ÉVITER OOM
            # ==========================================================
            torch.cuda.empty_cache()
            gc.collect()

            return self.get_parameters(config={}), len(self.train_loader.dataset), {}
            
        except Exception as e:
            print(f"\n[CRITICAL ERROR IN FIT]: {e}")
            import traceback
            traceback.print_exc()
            raise e

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        self.model.eval()
        criterion = nn.CrossEntropyLoss()
        loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for images, labels in self.test_loader:
                images, labels = images.to(self.device), labels.to(self.device)
                outputs = self.model(images)
                # Ligne corrigée ici :
                loss += criterion(outputs, labels).item() * images.size(0)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        loss = loss / total
        accuracy = correct / total
        return float(loss), total, {"accuracy": float(accuracy)}

def main():
    # Récupérer l'ID d'client mn l-environment variable (default 1)
    client_id = int(os.getenv("CLIENT_ID", "1"))
    print(f"Starting Flower client ID: {client_id}...")

    # Chargement d l-modèle
    model = BrainTumorVGG16()

    # Jbdna les samples dyal had l-client b get_client_data(client_id)
    print(f"Loading data samples for client {client_id}...")
    client_samples = get_client_data(client_id)

    # 3eytna l get_dataloaders bach t-creer lina train_loader, val_loader, w test_loader dyal had l-client
    train_loader, val_loader, test_loader = get_dataloaders(batch_size=BATCH_SIZE, train_samples=client_samples)

    # Initialisation d l-client dyalna
    client = BrainTumorClient(model, train_loader, val_loader, test_loader)

    try:
        print(f"Connecting client {client_id} to server at 127.0.0.1:8080...")
        fl.client.start_numpy_client(
            server_address=SERVER_ADDRESS,
            client=client,
        )
    except Exception as e:
        print(f"[ERROR] Client failed: {e}")

if __name__ == "__main__":
    main()