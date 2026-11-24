import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "model"))
from vgg16_model import BrainTumorVGG16, get_model, count_params


@pytest.fixture(scope="module")
def model():
    return get_model(pretrained=False)


def test_model_loads(model):
    assert isinstance(model, BrainTumorVGG16)


def test_output_shape(model):
    model.eval()
    with torch.no_grad():
        out = model(torch.randn(2, 3, 224, 224))
    assert out.shape == (2, 4)


def test_forward_no_nan(model):
    model.eval()
    with torch.no_grad():
        out = model(torch.randn(2, 3, 224, 224))
    assert not torch.isnan(out).any()


def test_features_frozen_classifier_trainable(model):
    assert all(not p.requires_grad for p in model.features.parameters())
    assert all(p.requires_grad for p in model.classifier.parameters())


def test_unfreeze_last_block():
    m = get_model(pretrained=False, freeze_features=True, unfreeze_last_block=True)
    convs = [l for l in m.features.children() if hasattr(l, "weight")]
    assert not any(p.requires_grad for p in convs[0].parameters())
    assert all(p.requires_grad for p in convs[-1].parameters())


def test_param_counts(model):
    total, trainable = count_params(model)
    assert 0 < trainable < total