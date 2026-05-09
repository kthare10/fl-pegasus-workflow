#!/usr/bin/env python3

"""Aggregate client checkpoints with Flower-compatible FedAvg/FedProx weighting."""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))

from flwr_torch_utils import aggregate_state_dicts, checkpoint_payload, ensure_parent, load_config, load_model_payload, save_model_payload  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--round", required=True, type=int)
    parser.add_argument("--client-update", action="append", required=True)
    parser.add_argument("--client-count", action="append", required=True)
    parser.add_argument("--output-model", required=True)
    parser.add_argument("--metrics", required=True)
    args = parser.parse_args()

    config = load_config(args.config)
    updates = []
    sample_counts = []
    model_spec = None
    for update_path in args.client_update:
        payload, update_spec, state_dict = load_model_payload(update_path)
        model_spec = update_spec
        updates.append((state_dict, int(payload.get("sample_count", 0)), int(payload.get("client_id", -1))))
        sample_counts.append(int(payload.get("sample_count", 0)))

    total = sum(max(0, count) for count in sample_counts)
    if total == 0:
        total = len(updates)
        updates = [(state_dict, 1, client_id) for state_dict, _, client_id in updates]

    aggregated_state = aggregate_state_dicts([(state_dict, count) for state_dict, count, _ in updates])
    model = checkpoint_payload(
        args.round,
        model_spec,
        aggregated_state,
        extra={
            "aggregation": str(config.get("aggregation", "fedavg")).lower(),
            "fedprox_mu": float(config.get("fedprox_mu", config.get("mu", 0.0))),
        },
    )

    metrics = {
        "round": args.round,
        "framework": "flwr+pytorch",
        "aggregation": str(config.get("aggregation", "fedavg")).lower(),
        "num_clients": len(updates),
        "total_train_samples": total,
        "client_samples": {str(client_id): int(count) for _, count, client_id in updates},
    }

    ensure_parent(args.output_model)
    ensure_parent(args.metrics)
    save_model_payload(args.output_model, model)
    with open(args.metrics, "w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2, sort_keys=True)
    print(f"Aggregated round {args.round} from {len(updates)} clients using {metrics['aggregation']}")


if __name__ == "__main__":
    main()
