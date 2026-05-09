#!/usr/bin/env python3

"""Train one federated client for one round using PyTorch and FedProx."""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))

import torch  # noqa: E402
from torch import nn  # noqa: E402
from torch.utils.data import DataLoader, TensorDataset  # noqa: E402

from flwr_torch_utils import (  # noqa: E402
    build_model_from_spec,
    ensure_parent,
    get_model_spec,
    load_config,
    load_model_payload,
    make_optimizer,
    model_payload,
    select_device,
    set_seed,
)


def load_records(path, split):
    features = []
    labels = []
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            record = json.loads(line)
            if record["split"] != split:
                continue
            features.append(record["x"])
            labels.append(record["y"])
    return features, labels


def proximal_penalty(model, global_state, device):
    penalty = torch.zeros(1, device=device)
    for name, parameter in model.named_parameters():
        penalty = penalty + torch.sum((parameter - global_state[name].to(device)) ** 2)
    return penalty


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--client-id", required=True, type=int)
    parser.add_argument("--round", required=True, type=int)
    parser.add_argument("--global-model", required=True)
    parser.add_argument("--client-data", required=True)
    parser.add_argument("--output-model", required=True)
    parser.add_argument("--metrics", required=True)
    parser.add_argument("--count-output", required=True)
    args = parser.parse_args()

    config = load_config(args.config)
    set_seed(int(config.get("split_seed", 13)) + args.client_id + args.round)
    _, model_spec, global_state = load_model_payload(args.global_model)
    model_spec.update(get_model_spec(config))
    device = select_device(config)

    model = build_model_from_spec(model_spec).to(device)
    model.load_state_dict(global_state)
    optimizer = make_optimizer(model, config)
    criterion = nn.BCEWithLogitsLoss()
    local_epochs = int(config.get("local_epochs", 1))
    features, labels = load_records(args.client_data, "train")

    x_tensor = torch.tensor(features, dtype=torch.float32)
    y_tensor = torch.tensor(labels, dtype=torch.float32)
    if y_tensor.ndim == 1:
        y_tensor = y_tensor.unsqueeze(1)
    dataset = TensorDataset(x_tensor, y_tensor)
    batch_size = min(int(config.get("batch_size", 8)), max(len(dataset), 1))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    use_fedprox = str(config.get("aggregation", "fedavg")).lower() == "fedprox" or str(
        config.get("algorithm", "")
    ).lower() == "fedprox"
    fedprox_mu = float(config.get("fedprox_mu", config.get("mu", 0.0)))

    losses = []
    model.train()
    for _ in range(local_epochs):
        for batch_x, batch_y in loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            optimizer.zero_grad()
            logits = model(batch_x)
            loss = criterion(logits, batch_y)
            if use_fedprox and fedprox_mu > 0:
                loss = loss + 0.5 * fedprox_mu * proximal_penalty(model, global_state, device)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu().item()))

    state_dict = {name: tensor.detach().cpu() for name, tensor in model.state_dict().items()}
    update = model_payload(args.round, model_spec, state_dict)
    update["client_id"] = args.client_id
    update["sample_count"] = len(dataset)

    metrics = {
        "round": args.round,
        "client_id": args.client_id,
        "device": str(device),
        "framework": "flwr+pytorch",
        "algorithm": "fedprox" if use_fedprox and fedprox_mu > 0 else "fedavg",
        "fedprox_mu": fedprox_mu if use_fedprox else 0.0,
        "train_samples": len(dataset),
        "mean_loss": sum(losses) / len(losses) if losses else None,
    }
    count = {"client_id": args.client_id, "sample_count": len(dataset)}

    for path in (args.output_model, args.metrics, args.count_output):
        ensure_parent(path)
    with open(args.output_model, "w", encoding="utf-8") as handle:
        json.dump(update, handle, indent=2, sort_keys=True)
    with open(args.metrics, "w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2, sort_keys=True)
    with open(args.count_output, "w", encoding="utf-8") as handle:
        json.dump(count, handle, indent=2, sort_keys=True)

    print(
        f"Trained client {args.client_id} round {args.round} on {len(dataset)} samples using {metrics['algorithm']}"
    )


if __name__ == "__main__":
    main()
