import torch
import torch.nn as nn
from torchvision import models

NUM_CLASSES = 4


class BrainTumorVGG16(nn.Module):
    """VGG16 pre-entraine (ImageNet), couches convolutives gelees,
    derniere couche remplacee par une sortie a 4 classes."""

    def __init__(self, freeze_features=True, unfreeze_last_block=False, pretrained=True):
        super().__init__()
        weights = models.VGG16_Weights.IMAGENET1K_V1 if pretrained else None
        base = models.vgg16(weights=weights)
        base.classifier[6] = nn.Linear(4096, NUM_CLASSES)
        self.features = base.features
        self.avgpool = base.avgpool
        self.classifier = base.classifier

        if freeze_features:
            for p in self.features.parameters():
                p.requires_grad = False
            if unfreeze_last_block:
                # les couches 24 a 30 forment le bloc conv5
                for layer in list(self.features.children())[24:]:
                    for p in layer.parameters():
                        p.requires_grad = True

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return self.classifier(x)


def get_model(freeze_features=True, unfreeze_last_block=False, pretrained=True):
    return BrainTumorVGG16(freeze_features, unfreeze_last_block, pretrained)


def count_params(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


if __name__ == "__main__":
    m = get_model(pretrained=False)
    out = m(torch.randn(2, 3, 224, 224))
    total, trainable = count_params(m)
    print("Sortie:", out.shape)
    print(f"Parametres total: {total:,} | entrainables: {trainable:,}")