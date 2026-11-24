import os
import sys

import numpy as np
import pytest
from PIL import Image

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "data"))
import split_data as sd


@pytest.fixture
def fake_root(tmp_path):
    """Mini dataset factice: 30 images par classe, dossiers de classes."""
    for c in sd.CLASSES:
        folder = tmp_path / c
        folder.mkdir()
        for i in range(30):
            arr = (np.random.rand(40, 40) * 255).astype("uint8")
            Image.fromarray(arr).save(folder / f"{i}.jpg")
    return str(tmp_path)


@pytest.fixture
def samples():
    # 4 classes x 100 echantillons factices
    return [(f"img_{c}_{i}.jpg", c) for c in range(4) for i in range(100)]


def _check_exact_cover(parts, n):
    flat = [i for p in parts for i in p]
    assert len(flat) == len(set(flat))      # pas d'overlap
    assert sorted(flat) == list(range(n))   # union = tout


def test_four_classes():
    assert sd.CLASSES == ["glioma", "meningioma", "no_tumor", "pituitary"]


def test_list_samples(fake_root):
    s = sd.list_samples(fake_root)
    assert len(s) == 120
    assert {l for _, l in s} == {0, 1, 2, 3}


def test_item_shape_and_rgb(fake_root):
    s = sd.list_samples(fake_root)
    x, y = sd.BrainTumorDataset(s)[0]   # image source en niveaux de gris
    assert x.shape == (3, 224, 224)
    assert y in (0, 1, 2, 3)


def test_normalisation(fake_root):
    s = sd.list_samples(fake_root)
    x, _ = sd.BrainTumorDataset(s)[0]
    assert x.min() < 0 or x.max() > 1   # normalisation ImageNet: hors de [0, 1]


def test_dataloaders(fake_root):
    tr, va, te = sd.get_dataloaders(batch_size=4, root=fake_root)
    x, y = next(iter(tr))
    assert x.shape == (4, 3, 224, 224)
    assert len(te.dataset) == 24        # 20 % de 120


def test_iid_sizes_and_no_overlap(samples):
    parts = sd.partition_iid(samples, 3)
    assert len(parts) == 3
    _check_exact_cover(parts, len(samples))
    for p in parts:
        counts = sd.class_counts(samples, p)
        assert max(counts) - min(counts) <= 1


def test_noniid_sizes_and_no_overlap(samples):
    parts = sd.partition_noniid(samples, 3, alpha=0.3, min_size=20)
    assert len(parts) == 3
    _check_exact_cover(parts, len(samples))
    counts = np.array([sd.class_counts(samples, p) for p in parts])
    assert counts.std() > 5             # distributions differentes entre clients


def test_clients_disjoint_and_exclude_test(fake_root):
    pool, test = sd.split_train_test(sd.list_samples(fake_root))
    for st in sd.SPLIT_TYPES:
        sets = [set(sd.get_client_data(k, root=fake_root, split_type=st)) for k in (1, 2, 3)]
        assert not (sets[0] & sets[1] or sets[0] & sets[2] or sets[1] & sets[2])
        assert not (set().union(*sets) & set(test))


def test_split_type_env(monkeypatch, fake_root):
    monkeypatch.setenv("CLIENT_ID", "2")
    monkeypatch.setenv("SPLIT_TYPE", "noniid")
    assert len(sd.get_client_data(root=fake_root)) > 0


def test_client_id_missing(monkeypatch, fake_root):
    monkeypatch.delenv("CLIENT_ID", raising=False)
    with pytest.raises(ValueError):
        sd.get_client_data(root=fake_root)


def test_client_id_invalid(fake_root):
    with pytest.raises(ValueError):
        sd.get_client_data(7, root=fake_root)


def test_split_type_invalid(fake_root):
    with pytest.raises(ValueError):
        sd.get_client_data(1, root=fake_root, split_type="foo")