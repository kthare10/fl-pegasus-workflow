#!/usr/bin/env python3

"""Assemble a plotting-ready summary artifact for branch comparison results."""

import argparse
import json
import os


def ensure_parent(path):
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cross-eval", required=True)
    parser.add_argument("--branch-stats", action="append", default=[])
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    with open(args.cross_eval, "r", encoding="utf-8") as handle:
        cross_eval = json.load(handle)

    stats = []
    for path in args.branch_stats:
        with open(path, "r", encoding="utf-8") as handle:
            stats.append(json.load(handle))

    output = {
        "cross_eval": cross_eval,
        "branches": stats,
        "plot_series": [
            {
                "dataset_name": branch["dataset_name"],
                "accuracy": branch.get("evaluation_accuracy"),
                "num_clients": branch.get("num_clients"),
                "num_samples": branch.get("num_samples"),
            }
            for branch in stats
        ],
    }
    ensure_parent(args.output)
    with open(args.output, "w", encoding="utf-8") as handle:
        json.dump(output, handle, indent=2, sort_keys=True)
    print(f"Wrote plotting summary for {len(stats)} branches")


if __name__ == "__main__":
    main()
