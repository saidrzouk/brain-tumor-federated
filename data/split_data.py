import os
import random

import numpy as np
import torch
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "raw")
CLASSES = ["glioma", "meningioma", "no_tumor", "pituitary"]
CLASS_TO_IDX = {c: i for i, c in enumerate(CLASSES)}
IMG_SIZE = 224
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]
SEED = 42
N_CLIENTS = 3
TEST_RATIO = 0.20
VAL_RATIO = 0.15
SPLIT_TYPES = ("iid", "noniid")


def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def get_transforms(train):
    if train:
        return transforms.Compose([
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ToTensor(),
            transforms.Normalize(MEAN, STD)])
    return transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD)])


def list_samples(root=ROOT):
    """Toutes les images: liste de (chemin, label), triee pour etre reproductible."""
    samples = []
    for c in CLASSES:
        folder = os.path.join(root, c)
        for f in sorted(os.listdir(folder)):
            if f.lower().endswith((".jpg", ".jpeg", ".png")):
                samples.append((os.path.join(folder, f), CLASS_TO_IDX[c]))
    return samples


class BrainTumorDataset(Dataset):
    def __init__(self, samples, train=False):
        self.samples = samples
        self.transform = get_transforms(train)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = Image.open(path).convert("RGB")
        return self.transform(img), label


def split_train_test(samples, test_ratio=TEST_RATIO, seed=SEED):
    """Test set commun a tous, stratifie, graine fixe."""
    labels = [l for _, l in samples]
    return train_test_split(samples, test_size=test_ratio,
                            stratify=labels, random_state=seed)


def split_train_val(samples, val_ratio=VAL_RATIO, seed=SEED):
    labels = [l for _, l in samples]
    return train_test_split(samples, test_size=val_ratio,
                            stratify=labels, random_state=seed)


def partition_iid(samples, n_clients=N_CLIENTS, seed=SEED):
    """IID: chaque classe est melangee puis repartie egalement entre les clients."""
    rng = np.random.RandomState(seed)
    labels = np.array([l for _, l in samples])
    parts = [[] for _ in range(n_clients)]
    for c in range(len(CLASSES)):
        idx = np.where(labels == c)[0]
        rng.shuffle(idx)
        for k, chunk in enumerate(np.array_split(idx, n_clients)):
            parts[k].extend(chunk.tolist())
    return [sorted(p) for p in parts]


def partition_noniid(samples, n_clients=N_CLIENTS, alpha=0.5, seed=SEED,
                     min_size=None, max_tries=1000):
    """Non-IID (Dirichlet): chaque classe est distribuee selon des proportions
    aleatoires differentes. alpha petit = tres heterogene."""
    if min_size is None:
        min_size = max(1, len(samples) // (n_clients * 10))
    rng = np.random.RandomState(seed)
    labels = np.array([l for _, l in samples])
    for _ in range(max_tries):
        parts = [[] for _ in range(n_clients)]
        for c in range(len(CLASSES)):
            idx = np.where(labels == c)[0]
            rng.shuffle(idx)
            props = rng.dirichlet([alpha] * n_clients)
            cuts = (np.cumsum(props) * len(idx)).astype(int)[:-1]
            for k, chunk in enumerate(np.split(idx, cuts)):
                parts[k].extend(chunk.tolist())
        if min(len(p) for p in parts) >= min_size:
            return [sorted(p) for p in parts]
    raise RuntimeError("Impossible de generer une partition non-IID valide (min_size trop grand).")


def class_counts(samples, indices):
    counts = [0] * len(CLASSES)
    for i in indices:
        counts[samples[i][1]] += 1
    return counts


def get_client_data(client_id=None, n_clients=N_CLIENTS, root=ROOT, split_type=None):
    """
    Partition du client (1..n_clients), prise dans l'ensemble d'entrainement
    (test set exclu). Sans client_id: variable d'environnement CLIENT_ID.
    Sans split_type: variable d'environnement SPLIT_TYPE ('iid' ou 'noniid', defaut 'iid').
    """
    if client_id is None:
        if "CLIENT_ID" not in os.environ:
            raise ValueError("CLIENT_ID non defini (variable d'environnement).")
        client_id = int(os.environ["CLIENT_ID"])
    if not 1 <= client_id <= n_clients:
        raise ValueError(f"CLIENT_ID={client_id} invalide (1..{n_clients}).")
    if split_type is None:
        split_type = os.environ.get("SPLIT_TYPE", "iid").lower()
    if split_type not in SPLIT_TYPES:
        raise ValueError(f"SPLIT_TYPE={split_type!r} invalide (attendu: {SPLIT_TYPES}).")

    train_pool, _ = split_train_test(list_samples(root))
    if split_type == "iid":
        parts = partition_iid(train_pool, n_clients)
    else:
        parts = partition_noniid(train_pool, n_clients)
    return [train_pool[i] for i in parts[client_id - 1]]


def get_dataloaders(batch_size=32, num_workers=0, train_samples=None, root=ROOT):
    """Retourne (train_loader, val_loader, test_loader). Le test set est le meme pour tous."""
    set_seed()
    train_pool, test_s = split_train_test(list_samples(root))
    if train_samples is None:
        train_samples = train_pool
    train_s, val_s = split_train_val(train_samples)
    kw = dict(batch_size=batch_size, num_workers=num_workers)
    return (DataLoader(BrainTumorDataset(train_s, train=True), shuffle=True, **kw),
            DataLoader(BrainTumorDataset(val_s), shuffle=False, **kw),
            DataLoader(BrainTumorDataset(test_s), shuffle=False, **kw))


if __name__ == "__main__":
    samples = list_samples()
    train_pool, test_s = split_train_test(samples)
    print(f"total={len(samples)} | train={len(train_pool)} | test={len(test_s)}")
    for st in SPLIT_TYPES:
        print(st)
        for k in range(1, N_CLIENTS + 1):
            data = get_client_data(k, split_type=st)
            counts = [sum(1 for _, l in data if l == c) for c in range(len(CLASSES))]
            print(f"  client {k}: {len(data)} images, classes = {counts}")