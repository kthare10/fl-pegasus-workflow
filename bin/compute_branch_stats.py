#!/usr/bin/env python3

"""Summarize per-branch dataset and evaluation statistics."""

import argparse
import csv
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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--client-manifest", required=True)
    parser.add_argument("--evaluation", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    config = load_config(args.config)
    with open(args.evaluation, "r", encoding="utf-8") as handle:
        evaluation = json.load(handle)

    rows = []
    with open(args.client_manifest, newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            rows.append(row)

    clients = sorted({row["client_id"] for row in rows})
    patients = sorted({row["patient_id"] for row in rows})
    split_counts = {}
    label_counts = {}
    for row in rows:
        split_counts[row["split"]] = split_counts.get(row["split"], 0) + 1
        label_counts[row["labels"]] = label_counts.get(row["labels"], 0) + 1

    output = {
        "dataset_name": config["dataset_name"],
        "num_clients": len(clients),
        "num_samples": len(rows),
        "num_patients": len(patients),
        "split_counts": split_counts,
        "top_labels": sorted(label_counts.items(), key=lambda item: (-item[1], item[0]))[:10],
        "evaluation_accuracy": evaluation.get("accuracy"),
        "evaluation_test_samples": evaluation.get("test_samples"),
    }
    ensure_parent(args.output)
    with open(args.output, "w", encoding="utf-8") as handle:
        json.dump(output, handle, indent=2, sort_keys=True)
    print(f"Wrote branch stats for {output['dataset_name']}")


if __name__ == "__main__":
    main()
