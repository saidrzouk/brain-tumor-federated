import os
import random
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from sklearn.model_selection import train_test_split

ROOT = "data/raw"
CLASSES = ["glioma", "meningioma", "notumor", "pituitary"]
CLASS_TO_IDX = {c: i for i, c in enumerate(CLASSES)}
IMG_SIZE = 224
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]
SEED = 42


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
            transforms.Normalize(MEAN, STD),
        ])
    return transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])


def list_samples(split, root=ROOT):
    """Retourne une liste de (chemin_image, label) pour Training ou Testing."""
    samples = []
    for c in CLASSES:
        folder = os.path.join(root, split, c)
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


def split_train_val(samples, val_ratio=0.15, seed=SEED):
    labels = [l for _, l in samples]
    train_s, val_s = train_test_split(
        samples, test_size=val_ratio, stratify=labels, random_state=seed
    )
    return train_s, val_s


def get_dataloaders(batch_size=32, val_ratio=0.15, num_workers=0,
                    train_samples=None, root=ROOT):
    """
    Retourne (train_loader, val_loader, test_loader).
    train_samples: liste (chemin, label) optionnelle, pour un client federe.
    """
    set_seed()
    if train_samples is None:
        train_samples = list_samples("Training", root)
    train_s, val_s = split_train_val(train_samples, val_ratio)
    test_s = list_samples("Testing", root)

    train_loader = DataLoader(BrainTumorDataset(train_s, train=True),
                              batch_size=batch_size, shuffle=True,
                              num_workers=num_workers)
    val_loader = DataLoader(BrainTumorDataset(val_s, train=False),
                            batch_size=batch_size, shuffle=False,
                            num_workers=num_workers)
    test_loader = DataLoader(BrainTumorDataset(test_s, train=False),
                             batch_size=batch_size, shuffle=False,
                             num_workers=num_workers)
    return train_loader, val_loader, test_loader


if __name__ == "__main__":
    tr, va, te = get_dataloaders(batch_size=8)
    x, y = next(iter(tr))
    print("Batch:", x.shape, y.tolist())
    print("Train/Val/Test:", len(tr.dataset), len(va.dataset), len(te.dataset))