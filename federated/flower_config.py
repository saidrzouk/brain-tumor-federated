"""
Configuration centralisée pour l'apprentissage fédéré (Brain Tumor MRI) :

Fonctionnement du cycle (10 rounds au total) :
1. Le serveur envoie le modèle global aux hôpitaux (clients).
2. Chaque hôpital entraîne le modèle localement sur ses propres données :
   - 3 époques d'entraînement par round.
   - Les données sont lues par lots (batches) de 16 images.
   - Le taux d'apprentissage (learning rate) est fixé à 0.0001.
3. Chaque hôpital renvoie les poids mis à jour au serveur.
4. Le serveur agrège les résultats via l'algorithme FedAvg.
5. Tolérance aux pannes : Définit le nombre minimal de clients disponibles pour lancer un round.
6. Le problème comporte 4 classes de tumeurs cérébrales.
"""

# Paramètres Réseau
SERVER_ADDRESS = "127.0.0.1:8080"

# Paramètres des Cycles Fédérés
NUM_ROUNDS = 10
NUM_CLIENTS = 3

# Tolérance aux pannes
MIN_FIT_CLIENTS = 3
MIN_EVALUATE_CLIENTS = 2
MIN_AVAILABLE_CLIENTS = 2

# Hyperparamètres d'entraînement local (Client)
LOCAL_EPOCHS = 3
BATCH_SIZE = 16
LEARNING_RATE = 0.0001

# Nombre de classes du Brain Tumor MRI
NUM_CLASSES = 4