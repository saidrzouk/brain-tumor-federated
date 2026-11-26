import os
import sys

import numpy as np
import pytest
import torch.nn as nn
from PIL import Image

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from data import split_data as sd
from federated import flower_config
from federated.client import BrainTumorClient
from federated.run_simulation import fedavg, get_model_params, set_model_params
from federated.server import weighted_average


class TinyBrainTumorNet(nn.Module):
    """Modèle léger reproduisant l'interface features/classifier de BrainTumorVGG16."""

    def __init__(self, num_classes=4):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 8, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((2, 2)),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(8 * 2 * 2, num_classes),
        )

    def forward(self, x):
        return self.classifier(self.features(x))


@pytest.fixture(scope="module")
def federated_dataset_dir(tmp_path_factory):
    """Dataset factice partagé pour les tests d'intégration (30 images par classe)."""
    root = tmp_path_factory.mktemp("brain_tumor_fl")
    for cls_name in sd.CLASSES:
        folder = root / cls_name
        folder.mkdir()
        for i in range(30):
            arr = (np.random.rand(32, 32) * 255).astype("uint8")
            Image.fromarray(arr).save(folder / f"{i}.jpg")
    return str(root)


def test_fedavg_weighted_aggregation():
    """Vérifie le calcul mathématique de l'agrégation pondérée FedAvg."""
    w_client_1 = [np.ones((2, 2), dtype=np.float32) * 1.0]
    w_client_2 = [np.ones((2, 2), dtype=np.float32) * 4.0]

    # Poids 1/3 pour client 1 et 2/3 pour client 2 => (1*1 + 2*4)/3 = 3.0
    aggregated = fedavg([w_client_1, w_client_2], [100, 200])
    assert len(aggregated) == 1
    assert aggregated[0].shape == (2, 2)
    assert np.allclose(aggregated[0], 3.0)


@pytest.mark.parametrize("split_type", ["iid", "noniid"])
def test_end_to_end_federated_round_three_clients(
    monkeypatch, federated_dataset_dir, split_type
):
    """
    Test d'intégration complet d'un cycle fédéré (3 clients -> FedAvg -> évaluation globale)
    en mode IID et non-IID.
    """
    monkeypatch.setattr("federated.client.LOCAL_EPOCHS", 1)

    global_model = TinyBrainTumorNet()
    initial_global_params = get_model_params(global_model)

    client_weights = []
    client_sizes = []

    # 1. Entraînement local sur les 3 clients hospitaliers
    for client_id in range(1, 4):
        client_samples = sd.get_client_data(
            client_id=client_id,
            root=federated_dataset_dir,
            split_type=split_type,
        )
        train_loader, val_loader, test_loader = sd.get_dataloaders(
            batch_size=8,
            train_samples=client_samples,
            root=federated_dataset_dir,
            num_workers=0,
        )

        client = BrainTumorClient(
            model=TinyBrainTumorNet(),
            train_loader=train_loader,
            val_loader=val_loader,
            test_loader=test_loader,
        )

        updated_params, num_examples, fit_metrics = client.fit(initial_global_params, {})
        assert num_examples > 0
        assert np.isfinite(fit_metrics["train_loss"])

        client_weights.append(updated_params)
        client_sizes.append(num_examples)

    # 2. Agrégation FedAvg côté serveur
    aggregated_params = fedavg(client_weights, client_sizes)
    set_model_params(global_model, aggregated_params, device="cpu")

    assert any(
        not np.allclose(w0, w1)
        for w0, w1 in zip(initial_global_params, aggregated_params)
    )

    # 3. Évaluation du modèle agrégé sur les clients et agrégation des métriques
    eval_metrics_list = []
    for client_id in range(1, 4):
        client_samples = sd.get_client_data(
            client_id=client_id,
            root=federated_dataset_dir,
            split_type=split_type,
        )
        _, _, test_loader = sd.get_dataloaders(
            batch_size=8,
            train_samples=client_samples,
            root=federated_dataset_dir,
            num_workers=0,
        )
        client = BrainTumorClient(
            model=TinyBrainTumorNet(),
            train_loader=None,
            val_loader=None,
            test_loader=test_loader,
        )
        loss, n_test, metrics = client.evaluate(aggregated_params, {})
        assert np.isfinite(loss)
        assert 0.0 <= metrics["accuracy"] <= 1.0
        eval_metrics_list.append((n_test, metrics))

    global_metrics = weighted_average(eval_metrics_list)
    assert 0.0 <= global_metrics["accuracy"] <= 1.0


def test_fault_tolerance_round_continues_with_two_clients(monkeypatch, federated_dataset_dir):
    """
    Vérifie le scénario d'indisponibilité d'un client (hospital_2 arrêté) :
    le round fédéré aboutit avec les 2 clients restants (>= MIN_FIT_CLIENTS).
    """
    monkeypatch.setattr("federated.client.LOCAL_EPOCHS", 1)

    active_clients = [1, 3]  # hospital_2 indisponible
    assert len(active_clients) >= flower_config.MIN_FIT_CLIENTS

    global_model = TinyBrainTumorNet()
    global_params = get_model_params(global_model)

    client_weights = []
    client_sizes = []

    for client_id in active_clients:
        client_samples = sd.get_client_data(
            client_id=client_id,
            root=federated_dataset_dir,
            split_type="iid",
        )
        train_loader, val_loader, test_loader = sd.get_dataloaders(
            batch_size=8,
            train_samples=client_samples,
            root=federated_dataset_dir,
            num_workers=0,
        )
        client = BrainTumorClient(
            model=TinyBrainTumorNet(),
            train_loader=train_loader,
            val_loader=val_loader,
            test_loader=test_loader,
        )
        updated_params, num_examples, _ = client.fit(global_params, {})
        client_weights.append(updated_params)
        client_sizes.append(num_examples)

    aggregated_params = fedavg(client_weights, client_sizes)
    assert len(aggregated_params) == len(global_params)