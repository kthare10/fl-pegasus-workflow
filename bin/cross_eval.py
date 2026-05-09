#!/usr/bin/env python3

"""Compare final metrics across two or more FL branches."""

import argparse
import json
import os


def ensure_parent(path):
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--branch", action="append", required=True)
    parser.add_argument("--evaluation", action="append", required=True)
    parser.add_argument("--baseline", action="append", required=True)
    parser.add_argument("--stats", action="append", required=True)
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

    best_branch = None
    best_accuracy = None
    for branch, record in comparisons.items():
        accuracy = record.get("accuracy")
        if accuracy is not None and (best_accuracy is None or accuracy > best_accuracy):
            best_accuracy = accuracy
            best_branch = branch

    output = {
        "branches": comparisons,
        "best_branch": best_branch,
        "best_accuracy": best_accuracy,
    }
    ensure_parent(args.output)
    with open(args.output, "w", encoding="utf-8") as handle:
        json.dump(output, handle, indent=2, sort_keys=True)
    print(f"Compared {len(comparisons)} branches")


if __name__ == "__main__":
    main()
