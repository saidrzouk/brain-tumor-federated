import importlib
import os
import sys

import flwr as fl
import pytest

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from federated import flower_config, server


def test_flower_config_defaults():
    """Vérifie les paramètres fédérés exigés par le cahier des charges."""
    assert flower_config.NUM_ROUNDS == 10
    assert flower_config.NUM_CLIENTS == 3
    assert flower_config.LOCAL_EPOCHS == 3
    assert flower_config.BATCH_SIZE == 16
    assert flower_config.LEARNING_RATE == pytest.approx(1e-4)
    assert flower_config.NUM_CLASSES == 4


def test_fault_tolerance_thresholds():
    """Vérifie la tolérance à la panne d'un client (minimum 2 clients sur 3)."""
    assert flower_config.MIN_FIT_CLIENTS == 2
    assert flower_config.MIN_EVALUATE_CLIENTS == 2
    assert flower_config.MIN_AVAILABLE_CLIENTS == 2
    assert flower_config.MIN_FIT_CLIENTS < flower_config.NUM_CLIENTS


def test_server_address_env_override(monkeypatch):
    """Vérifie la surcharge de SERVER_ADDRESS pour Docker (server:8080 / 0.0.0.0:8080)."""
    monkeypatch.setenv("SERVER_ADDRESS", "server:8080")
    reloaded = importlib.reload(flower_config)
    assert reloaded.SERVER_ADDRESS == "server:8080"

    monkeypatch.delenv("SERVER_ADDRESS", raising=False)
    reloaded = importlib.reload(flower_config)
    assert reloaded.SERVER_ADDRESS == "127.0.0.1:8080"


def test_weighted_average_accuracy():
    """Vérifie que l'agrégation des métriques est bien pondérée par le nombre d'exemples."""
    metrics = [
        (100, {"accuracy": 0.80}),
        (300, {"accuracy": 1.00}),
    ]
    result = server.weighted_average(metrics)
    assert "accuracy" in result
    # (100 * 0.80 + 300 * 1.00) / 400 = 380 / 400 = 0.95
    assert result["accuracy"] == pytest.approx(0.95)


def test_server_main_configures_fedavg_and_starts_server(monkeypatch):
    """Vérifie que server.main() instancie FedAvg avec les bons seuils et démarre gRPC."""
    captured = {}

    def fake_start_server(server_address, config, strategy):
        captured["server_address"] = server_address
        captured["config"] = config
        captured["strategy"] = strategy

    monkeypatch.setattr(fl.server, "start_server", fake_start_server)
    server.main()

    assert captured["server_address"] == server.SERVER_ADDRESS
    assert captured["config"].num_rounds == flower_config.NUM_ROUNDS

    strategy = captured["strategy"]
    assert isinstance(strategy, fl.server.strategy.FedAvg)
    assert strategy.min_fit_clients == flower_config.MIN_FIT_CLIENTS
    assert strategy.min_evaluate_clients == flower_config.MIN_EVALUATE_CLIENTS
    assert strategy.min_available_clients == flower_config.MIN_AVAILABLE_CLIENTS
    assert strategy.evaluate_metrics_aggregation_fn is server.weighted_average