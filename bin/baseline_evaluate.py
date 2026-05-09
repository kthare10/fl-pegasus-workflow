#!/usr/bin/env python3

"""Evaluate a simple prevalence baseline on held-out client records."""

import argparse
import json
import os

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


def scalar_label(record):
    return 1 if any(int(v) == 1 for v in record["y"]) else 0


def load_test_labels(path):
    labels = []
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            record = json.loads(line)
            if record["split"] == "test":
                labels.append(scalar_label(record))
    return labels


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--client-data", action="append", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    config = load_config(args.config)
    client_metrics = {}
    total = 0
    positive = 0
    for index, path in enumerate(args.client_data):
        labels = load_test_labels(path)
        positives = sum(labels)
        samples = len(labels)
        total += samples
        positive += positives
        majority_label = 1 if positives >= max(samples - positives, 0) else 0
        correct = positives if majority_label else samples - positives
        client_metrics[f"client_{index:03d}"] = {
            "test_samples": samples,
            "positive_samples": positives,
            "predicted_label": majority_label,
            "accuracy": correct / samples if samples else None,
        }

    prevalence = positive / total if total else 0.0
    predicted_label = 1 if prevalence >= 0.5 else 0
    correct = positive if predicted_label else total - positive
    output = {
        "dataset_name": config["dataset_name"],
        "method": "majority_prevalence_baseline",
        "test_samples": total,
        "positive_rate": prevalence,
        "predicted_label": predicted_label,
        "accuracy": correct / total if total else None,
        "client_metrics": client_metrics,
    }
    ensure_parent(args.output)
    with open(args.output, "w", encoding="utf-8") as handle:
        json.dump(output, handle, indent=2, sort_keys=True)
    print(f"Computed baseline for {output['dataset_name']}")


if __name__ == "__main__":
    main()
