"""
Serveur central Flower pour l'orchestration de l'apprentissage fédéré (Brain Tumor MRI) :

Rôle : Coordonne les cycles d'apprentissage (rounds) entre les différents hôpitaux (clients).

Fonctionnement du serveur :
1. Configuration : Charge les paramètres réseau et fédérés depuis `flower_config.py`.
2. Stratégie d'agrégation (`FedAvg`) :
   - Combine les poids reçus des clients en effectuant une moyenne pondérée selon le nombre d'images.
   - Calcule l'accuracy globale pondérée à chaque cycle via `weighted_average`.
3. Tolérance aux pannes : Configure le nombre minimal de clients requis (`MIN_FIT_CLIENTS`, `MIN_AVAILABLE_CLIENTS`) pour démarrer ou poursuivre un round.
4. Lancement (`main`) : Démarre le serveur gRPC sur `SERVER_ADDRESS` pour exécuter les 10 rounds prévus.
"""

import flwr as fl
from federated.flower_config import NUM_ROUNDS, MIN_FIT_CLIENTS, SERVER_ADDRESS

# Fonction pour calculer la moyenne pondérée de l'accuracy des clients
def weighted_average(metrics):
    accuracies = [num_examples * m["accuracy"] for num_examples, m in metrics]
    examples = [num_examples for num_examples, _ in metrics]
    return {"accuracy": sum(accuracies) / sum(examples)}

def main():
    # Stratégie FedAvg avec agrégation de l'accuracy et tolérance aux pannes
    strategy = fl.server.strategy.FedAvg(
        min_fit_clients=MIN_FIT_CLIENTS,
        min_available_clients=MIN_FIT_CLIENTS,
        evaluate_metrics_aggregation_fn=weighted_average,
    )
    
    print(f"Starting Flower server on {SERVER_ADDRESS} with FedAvg strategy...")
    
    # Lancement du serveur Flower
    fl.server.start_server(
        server_address=SERVER_ADDRESS,
        config=fl.server.ServerConfig(num_rounds=NUM_ROUNDS),
        strategy=strategy,
    )

if __name__ == "__main__":
    main()