#!/usr/bin/env python3

"""Package final evaluation outputs for paper tables and provenance."""

import argparse
import csv
import json
import os
import tarfile
from datetime import datetime, timezone

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
    parser.add_argument("--evaluation", required=True)
    parser.add_argument("--round-metric", action="append", default=[])
    parser.add_argument("--results", required=True)
    parser.add_argument("--paper-tables", required=True)
    parser.add_argument("--provenance", required=True)
    args = parser.parse_args()

    config = load_config(args.config)
    with open(args.evaluation, "r", encoding="utf-8") as handle:
        evaluation = json.load(handle)

    round_metrics = []
    for path in args.round_metric:
        with open(path, "r", encoding="utf-8") as handle:
            round_metrics.append(json.load(handle))

    for path in (args.results, args.paper_tables, args.provenance):
        ensure_parent(path)

    with open(args.paper_tables, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=["dataset", "round", "test_samples", "accuracy"]
        )
        writer.writeheader()
        writer.writerow(
            {
                "dataset": evaluation["dataset_name"],
                "round": evaluation["model_round"],
                "test_samples": evaluation["test_samples"],
                "accuracy": evaluation["accuracy"],
            }
        )

    provenance = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "dataset_name": config["dataset_name"],
        "num_clients": config["num_clients"],
        "rounds": config["rounds"],
        "evaluation_file": args.evaluation,
        "round_metric_files": args.round_metric,
    }
    with open(args.provenance, "w", encoding="utf-8") as handle:
        json.dump(provenance, handle, indent=2, sort_keys=True)

    with tarfile.open(args.results, "w:gz") as archive:
        archive.add(args.evaluation, arcname=os.path.basename(args.evaluation))
        archive.add(args.paper_tables, arcname=os.path.basename(args.paper_tables))
        archive.add(args.provenance, arcname=os.path.basename(args.provenance))
        for path in args.round_metric:
            round_dir = os.path.basename(os.path.dirname(path))
            archive.add(
                path,
                arcname=os.path.join(
                    "round_metrics", round_dir, os.path.basename(path)
                ),
            )

    print(f"Packaged results in {args.results}")


if __name__ == "__main__":
    main()
