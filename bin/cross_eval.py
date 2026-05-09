#!/usr/bin/env python3

"""Compare branches and evaluate cross-dataset model generalization."""

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


def evaluate_branch_model(model_path, config_path, target_client_data):
    config = load_config(config_path)
    _, model_spec, state_dict = load_model_payload(model_path)
    device = select_device(config)
    model = build_model_from_spec(model_spec).to(device)
    model.load_state_dict(state_dict)
    records = load_records(target_client_data, split="test")
    return evaluate_records(model, records, config, device)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--branch", action="append", required=True)
    parser.add_argument("--evaluation", action="append", required=True)
    parser.add_argument("--baseline", action="append", required=True)
    parser.add_argument("--stats", action="append", required=True)
    parser.add_argument("--matrix-spec", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    comparisons = {}
    for branch, evaluation_path, baseline_path, stats_path in zip(
        args.branch, args.evaluation, args.baseline, args.stats
    ):
        with open(evaluation_path, "r", encoding="utf-8") as handle:
            evaluation = json.load(handle)
        with open(baseline_path, "r", encoding="utf-8") as handle:
            baseline = json.load(handle)
        with open(stats_path, "r", encoding="utf-8") as handle:
            stats = json.load(handle)
        comparisons[branch] = {
            "dataset_name": evaluation.get("dataset_name"),
            "accuracy": evaluation.get("accuracy"),
            "baseline_accuracy": baseline.get("accuracy"),
            "lift_over_baseline": None
            if evaluation.get("accuracy") is None or baseline.get("accuracy") is None
            else evaluation["accuracy"] - baseline["accuracy"],
            "test_samples": evaluation.get("test_samples"),
            "num_clients": stats.get("num_clients"),
            "num_samples": stats.get("num_samples"),
        }

    with open(args.matrix_spec, "r", encoding="utf-8") as handle:
        matrix_spec = json.load(handle)

    cross_dataset = {}
    best_branch = None
    best_accuracy = None
    for branch, record in comparisons.items():
        accuracy = record.get("accuracy")
        if accuracy is not None and (best_accuracy is None or accuracy > best_accuracy):
            best_accuracy = accuracy
            best_branch = branch

    for source_branch, source_spec in matrix_spec["branches"].items():
        cross_dataset[source_branch] = {}
        for target_branch, target_spec in matrix_spec["branches"].items():
            metrics = evaluate_branch_model(
                source_spec["model"],
                target_spec["config"],
                target_spec["client_data"],
            )
            cross_dataset[source_branch][target_branch] = metrics

    output = {
        "branches": comparisons,
        "best_branch": best_branch,
        "best_accuracy": best_accuracy,
        "cross_dataset": cross_dataset,
    }
    ensure_parent(args.output)
    with open(args.output, "w", encoding="utf-8") as handle:
        json.dump(output, handle, indent=2, sort_keys=True)
    print(f"Compared {len(comparisons)} branches with cross-dataset evaluation")


if __name__ == "__main__":
    main()
