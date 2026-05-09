#!/usr/bin/env python3

"""Evaluate the final global PyTorch model on held-out client records."""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))

import torch  # noqa: E402

from flwr_torch_utils import (  # noqa: E402
    build_model_from_spec,
    ensure_parent,
    load_config,
    load_model_payload,
    select_device,
)


def evaluate_file(path, model, device):
    total = 0
    correct = 0
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            record = json.loads(line)
            if record["split"] != "test":
                continue
            features = torch.tensor(record["x"], dtype=torch.float32, device=device).unsqueeze(0)
            labels = torch.tensor(record["y"], dtype=torch.float32, device=device).unsqueeze(0)
            logits = model(features)
            probs = torch.sigmoid(logits)
            preds = (probs >= 0.5).float()
            total += 1
            correct += int(torch.equal(preds.cpu(), labels.cpu()))
    return {
        "test_samples": total,
        "accuracy": correct / total if total else None,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--client-data", action="append", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    config = load_config(args.config)
    payload, model_spec, state_dict = load_model_payload(args.model)
    device = select_device(config)
    model = build_model_from_spec(model_spec).to(device)
    model.load_state_dict(state_dict)
    model.eval()

    client_metrics = {}
    total = 0
    weighted_accuracy = 0.0
    with torch.no_grad():
        for index, client_path in enumerate(args.client_data):
            metrics = evaluate_file(client_path, model, device)
            client_metrics[f"client_{index:03d}"] = metrics
            if metrics["test_samples"]:
                total += metrics["test_samples"]
                weighted_accuracy += metrics["accuracy"] * metrics["test_samples"]

    output = {
        "dataset_name": config["dataset_name"],
        "framework": "flwr+pytorch",
        "model_round": payload["round"],
        "model_name": payload["model_name"],
        "test_samples": total,
        "accuracy": weighted_accuracy / total if total else None,
        "client_metrics": client_metrics,
    }
    ensure_parent(args.output)
    with open(args.output, "w", encoding="utf-8") as handle:
        json.dump(output, handle, indent=2, sort_keys=True)
    print(f"Evaluated {total} held-out samples")


if __name__ == "__main__":
    main()
