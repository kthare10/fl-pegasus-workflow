#!/usr/bin/env python3

"""Shared PyTorch and Flower-compatible helpers for FL workflow jobs."""

from collections import OrderedDict
import json
import os
import random

import numpy as np

try:
    import torch
    from torch import nn
except ImportError as exc:  # pragma: no cover
    raise RuntimeError("PyTorch is required for the FL training stack") from exc

try:
    from flwr.server.strategy.aggregate import aggregate as flwr_aggregate
except ImportError:  # pragma: no cover
    flwr_aggregate = None

try:
    import yaml
except ImportError:  # pragma: no cover
    yaml = None


def load_config(path):
    with open(path, "r", encoding="utf-8") as handle:
        text = handle.read()
    return yaml.safe_load(text) if yaml else json.loads(text)


def ensure_parent(path):
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def infer_output_dim(config):
    if "num_classes" in config:
        return int(config["num_classes"])
    task = str(config.get("task", "")).lower()
    dataset = str(config.get("dataset_name", "")).lower()
    if "multi_label" in task:
        labels = config.get("labels", [])
        return len(labels) if labels else 1
    if "binary" in task or dataset.startswith("tcia") or "lidc" in dataset:
        return 1
    labels = config.get("labels", [])
    return len(labels) if labels else 1


def get_model_spec(config):
    hidden_dims = config.get("hidden_dims", [128, 64])
    if isinstance(hidden_dims, str):
        hidden_dims = [int(part) for part in hidden_dims.split(",") if part.strip()]
    return {
        "model_name": config.get("model_name", "torch-mlp"),
        "feature_dim": int(config.get("feature_dim", 16)),
        "output_dim": infer_output_dim(config),
        "hidden_dims": [int(value) for value in hidden_dims],
        "dropout": float(config.get("dropout", 0.1)),
        "learning_rate": float(config.get("learning_rate", 0.001)),
        "local_epochs": int(config.get("local_epochs", 1)),
        "batch_size": int(config.get("batch_size", 8)),
        "aggregation": str(config.get("aggregation", "fedavg")).lower(),
        "fedprox_mu": float(config.get("fedprox_mu", config.get("mu", 0.0))),
    }


class FeatureMLP(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dims, dropout):
        super().__init__()
        layers = []
        last_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(last_dim, hidden_dim))
            layers.append(nn.ReLU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            last_dim = hidden_dim
        layers.append(nn.Linear(last_dim, output_dim))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


def build_model_from_spec(model_spec):
    name = str(model_spec["model_name"]).lower()
    if name in {"torch-mlp", "mlp", "feature-mlp", "linear-smoke"}:
        hidden_dims = [] if name == "linear-smoke" else model_spec["hidden_dims"]
        dropout = 0.0 if name == "linear-smoke" else model_spec["dropout"]
        return FeatureMLP(
            model_spec["feature_dim"],
            model_spec["output_dim"],
            hidden_dims,
            dropout,
        )
    raise ValueError(f"Unsupported model_name: {model_spec['model_name']}")


def state_dict_to_jsonable(state_dict):
    return {name: tensor.detach().cpu().tolist() for name, tensor in state_dict.items()}


def jsonable_to_state_dict(payload):
    return OrderedDict(
        (name, torch.tensor(values, dtype=torch.float32))
        for name, values in payload.items()
    )


def model_payload(round_idx, model_spec, state_dict):
    return {
        "round": round_idx,
        "framework": "flwr+pytorch",
        "model_name": model_spec["model_name"],
        "feature_dim": model_spec["feature_dim"],
        "output_dim": model_spec["output_dim"],
        "hidden_dims": model_spec["hidden_dims"],
        "dropout": model_spec["dropout"],
        "state_dict": state_dict_to_jsonable(state_dict),
    }


def load_model_payload(path):
    with open(path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)
    state_dict = jsonable_to_state_dict(payload["state_dict"])
    model_spec = {
        "model_name": payload["model_name"],
        "feature_dim": int(payload["feature_dim"]),
        "output_dim": int(payload["output_dim"]),
        "hidden_dims": [int(value) for value in payload.get("hidden_dims", [])],
        "dropout": float(payload.get("dropout", 0.0)),
    }
    return payload, model_spec, state_dict


def select_device(config):
    requested = int(config.get("request_gpus", 0))
    if requested > 0 and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def numpy_state_list(state_dict):
    keys = list(state_dict.keys())
    arrays = [state_dict[key].detach().cpu().numpy() for key in keys]
    return keys, arrays


def aggregate_state_dicts(weighted_state_dicts):
    keys, first_arrays = numpy_state_list(weighted_state_dicts[0][0])
    if flwr_aggregate is not None:
        aggregated = flwr_aggregate(
            [(numpy_state_list(state_dict)[1], sample_count) for state_dict, sample_count in weighted_state_dicts]
        )
    else:
        total = sum(weight for _, weight in weighted_state_dicts)
        if total <= 0:
            total = len(weighted_state_dicts)
            weighted_state_dicts = [(state, 1) for state, _ in weighted_state_dicts]
        aggregated = [np.zeros_like(array, dtype=np.float32) for array in first_arrays]
        for state_dict, sample_count in weighted_state_dicts:
            _, arrays = numpy_state_list(state_dict)
            factor = float(sample_count) / float(total)
            aggregated = [acc + factor * arr for acc, arr in zip(aggregated, arrays)]
    return OrderedDict(
        (key, torch.tensor(array, dtype=torch.float32))
        for key, array in zip(keys, aggregated)
    )


def make_optimizer(model, config):
    optimizer_name = str(config.get("optimizer", "adam")).lower()
    lr = float(config.get("learning_rate", 0.001))
    weight_decay = float(config.get("weight_decay", 0.0))
    if optimizer_name == "sgd":
        momentum = float(config.get("momentum", 0.9))
        return torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
    return torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
