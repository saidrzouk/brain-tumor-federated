import gc
import os
import sys

import numpy as np
import pytest
import torch
import torch.nn as nn
from PIL import Image

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from data import split_data as sd
from federated.client import BrainTumorClient
from model.vgg16_model import get_model


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


@pytest.fixture
def tiny_loaders(tmp_path):
    """Crée un mini-dataset factice (8 images par classe) et retourne les DataLoaders."""
    for cls_name in sd.CLASSES:
        cls_dir = tmp_path / cls_name
        cls_dir.mkdir()
        for i in range(8):
            arr = (np.random.rand(32, 32) * 255).astype("uint8")
            Image.fromarray(arr).save(cls_dir / f"{i}.jpg")

    train_loader, val_loader, test_loader = sd.get_dataloaders(
        batch_size=4,
        root=str(tmp_path),
        num_workers=0,
    )
    return train_loader, val_loader, test_loader


def test_client_freezes_features_and_keeps_classifier_trainable(tiny_loaders):
    train_loader, val_loader, test_loader = tiny_loaders
    client = BrainTumorClient(
        model=TinyBrainTumorNet(),
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
    )
    assert all(not p.requires_grad for p in client.model.features.parameters())
    assert any(p.requires_grad for p in client.model.classifier.parameters())


def test_get_and_set_parameters_roundtrip(tiny_loaders):
    train_loader, val_loader, test_loader = tiny_loaders
    client = BrainTumorClient(
        model=TinyBrainTumorNet(),
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
    )

    initial_params = client.get_parameters({})
    assert isinstance(initial_params, list)
    assert len(initial_params) == len(client.model.state_dict())
    assert all(isinstance(arr, np.ndarray) for arr in initial_params)

    modified_params = [np.zeros_like(arr) for arr in initial_params]
    client.set_parameters(modified_params)

    updated_params = client.get_parameters({})
    for arr in updated_params:
        assert np.all(arr == 0.0)


def test_client_fit_updates_weights_and_returns_metrics(tiny_loaders):
    train_loader, val_loader, test_loader = tiny_loaders
    client = BrainTumorClient(
        model=TinyBrainTumorNet(),
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
    )

    params_before = client.get_parameters({})
    updated_params, num_examples, metrics = client.fit(params_before, {})

    assert num_examples == len(train_loader.dataset)
    assert "train_loss" in metrics
    assert isinstance(metrics["train_loss"], float)
    assert np.isfinite(metrics["train_loss"])

    changed = any(
        not np.allclose(before, after)
        for before, after in zip(params_before, updated_params)
    )
    assert changed


def test_client_evaluate_returns_valid_loss_and_accuracy(tiny_loaders):
    train_loader, val_loader, test_loader = tiny_loaders
    client = BrainTumorClient(
        model=TinyBrainTumorNet(),
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
    )

    params = client.get_parameters({})
    loss, num_examples, metrics = client.evaluate(params, {})

    assert isinstance(loss, float)
    assert loss >= 0.0
    assert num_examples == len(test_loader.dataset)
    assert "accuracy" in metrics
    assert 0.0 <= metrics["accuracy"] <= 1.0


def test_client_evaluate_raises_without_test_loader(tiny_loaders):
    train_loader, val_loader, _ = tiny_loaders
    client = BrainTumorClient(
        model=TinyBrainTumorNet(),
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=None,
    )
    params = client.get_parameters({})
    with pytest.raises(ValueError):
        client.evaluate(params, {})


def test_client_accepts_real_vgg16_architecture(tiny_loaders):
    """Vérifie la compatibilité directe avec BrainTumorVGG16 sans dupliquer les poids."""
    train_loader, val_loader, test_loader = tiny_loaders
    vgg = get_model(freeze_features=True, pretrained=False)
    client = BrainTumorClient(
        model=vgg,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
    )
    assert all(not p.requires_grad for p in client.model.features.parameters())
    assert any(p.requires_grad for p in client.model.classifier.parameters())
    del client, vgg
    gc.collect()