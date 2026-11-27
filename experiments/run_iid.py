"""Run the sequential federated experiment with an IID partition."""

from __future__ import annotations

import sys
import os
import gc

import numpy as np
import torch
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
MODEL_DIR = ROOT / "model" / "saved_models"

from data.split_data import get_client_data, get_dataloaders  # noqa: E402
from federated.client import BrainTumorClient  # noqa: E402
from federated.flower_config import BATCH_SIZE, NUM_CLIENTS, NUM_ROUNDS  # noqa: E402
from model.vgg16_model import BrainTumorVGG16  # noqa: E402
from experiments.result_utils import save_results  # noqa: E402


def get_model_params(model):
    return [value.detach().cpu().numpy().copy() for value in model.state_dict().values()]


def set_model_params(model, params, device):
    state_dict = {
        key: torch.as_tensor(param, device=device)
        for key, param in zip(model.state_dict().keys(), params)
    }
    model.load_state_dict(state_dict)


def fedavg(weights_list, num_samples_list):
    total = sum(num_samples_list)
    return [
        np.sum(
            [weights_list[c][i] * (num_samples_list[c] / total)
             for c in range(len(weights_list))],
            axis=0,
        )
        for i in range(len(weights_list[0]))
    ]


def cleanup(device, *objects):
    for obj in objects:
        del obj
    if device.type == "cuda":
        torch.cuda.empty_cache()
    gc.collect()


def run_experiment():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    global_model = BrainTumorVGG16().to(device)
    result_rows = []

    for round_idx in range(NUM_ROUNDS):
        print(f"\nROUND {round_idx + 1} / {NUM_ROUNDS}")
        global_params = get_model_params(global_model)
        client_weights, client_lens = [], []
        train_metrics = {}

        for client_id in range(1, NUM_CLIENTS + 1):
            client_samples = get_client_data(client_id, split_type="iid")
            train_loader, val_loader, test_loader = get_dataloaders(
                batch_size=BATCH_SIZE, train_samples=client_samples
            )
            client_model = BrainTumorVGG16()
            client = BrainTumorClient(client_model, train_loader, val_loader, test_loader)
            updated_params, num_samples, metrics = client.fit(global_params, {})
            print(
                f"Client {client_id}: dataset_size={len(client_samples)} | "
                f"loss={metrics['train_loss']:.4f}"
            )
            client_weights.append(updated_params)
            client_lens.append(num_samples)
            train_metrics[client_id] = (len(client_samples), metrics["train_loss"])
            cleanup(device, client, client_model, train_loader, val_loader, test_loader)

        new_global_params = fedavg(client_weights, client_lens)
        set_model_params(global_model, new_global_params, device)

        total, loss_sum, accuracy_sum = 0, 0.0, 0.0
        evaluation_rows = []
        for client_id in range(1, NUM_CLIENTS + 1):
            client_samples = get_client_data(client_id, split_type="iid")
            _, _, test_loader = get_dataloaders(
                batch_size=BATCH_SIZE, train_samples=client_samples
            )
            client_model = BrainTumorVGG16()
            client = BrainTumorClient(client_model, None, None, test_loader)
            loss, count, metrics = client.evaluate(new_global_params, {})
            print(
                f"Client {client_id}: test_size={count} | "
                f"loss={loss:.4f} | accuracy={metrics['accuracy']:.2%}"
            )
            loss_sum += loss * count
            accuracy_sum += metrics["accuracy"] * count
            total += count
            evaluation_rows.append((client_id, count, loss, metrics["accuracy"]))
            cleanup(device, client, client_model, test_loader)

        global_loss = loss_sum / total
        global_accuracy = accuracy_sum / total
        print(f"Loss={global_loss:.4f} | Accuracy={global_accuracy:.2%}")
        for client_id, test_size, test_loss, accuracy in evaluation_rows:
            dataset_size, train_loss = train_metrics[client_id]
            result_rows.append({
                "round": round_idx + 1,
                "client_id": client_id,
                "dataset_size": dataset_size,
                "train_loss": train_loss,
                "test_size": test_size,
                "test_loss": test_loss,
                "accuracy": accuracy,
                "global_loss": global_loss,
                "global_accuracy": global_accuracy,
            })

    save_results("iid", result_rows)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    model_path = MODEL_DIR / "federated_iid.pth"
    torch.save(
        {
            "model_state_dict": global_model.state_dict(),
            "split_type": "iid",
            "rounds": NUM_ROUNDS,
        },
        model_path,
    )
    print(f"Saved IID model to {model_path}")


if __name__ == "__main__":
    run_experiment()
