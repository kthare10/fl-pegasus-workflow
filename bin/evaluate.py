#!/usr/bin/env python3

"""Evaluate a PyTorch image model on held-out client records."""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))

from flwr_torch_utils import (  # noqa: E402
    build_model_from_spec,
    ensure_parent,
    evaluate_records,
    load_config,
    load_model_payload,
    load_records,
    select_device,
)


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
    for index, client_path in enumerate(args.client_data):
        records = load_records([client_path], split="test")
        metrics = evaluate_records(model, records, config, device)
        client_metrics[f"client_{index:03d}"] = metrics
        if metrics["test_samples"] and metrics["accuracy"] is not None:
            total += metrics["test_samples"]
            weighted_accuracy += metrics["accuracy"] * metrics["test_samples"]

    output = {
        "dataset_name": config["dataset_name"],
        "framework": "flwr+pytorch",
        "model_round": payload["round"],
        "model_name": model_spec["model_name"],
        "test_samples": total,
        "accuracy": weighted_accuracy / total if total else None,
        "client_metrics": client_metrics,
        "checkpoint_path": args.model,
    }
    ensure_parent(args.output)
    with open(args.output, "w", encoding="utf-8") as handle:
        json.dump(output, handle, indent=2, sort_keys=True)
    print(f"Evaluated {total} held-out samples")


if __name__ == "__main__":
    main()
