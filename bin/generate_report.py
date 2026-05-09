#!/usr/bin/env python3

"""Generate final comparison artifacts for a dual-branch FL workflow."""

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
    parser.add_argument("--branch", action="append", default=[])
    parser.add_argument("--evaluation", action="append", default=[])
    parser.add_argument("--baseline", action="append", default=[])
    parser.add_argument("--stats", action="append", default=[])
    parser.add_argument("--validation", action="append", default=[])
    parser.add_argument("--round-metric", action="append", default=[])
    parser.add_argument("--cross-eval", required=True)
    parser.add_argument("--plot-summary", required=True)
    parser.add_argument("--report", required=True)
    parser.add_argument("--results", required=True)
    parser.add_argument("--paper-tables", required=True)
    parser.add_argument("--provenance", required=True)
    parser.add_argument("--final-evaluation", required=True)
    args = parser.parse_args()

    config = load_config(args.config)
    with open(args.cross_eval, "r", encoding="utf-8") as handle:
        cross_eval = json.load(handle)
    with open(args.plot_summary, "r", encoding="utf-8") as handle:
        plot_summary = json.load(handle)

    branches = []
    for branch, evaluation_path, baseline_path, stats_path, validation_path in zip(
        args.branch, args.evaluation, args.baseline, args.stats, args.validation
    ):
        with open(evaluation_path, "r", encoding="utf-8") as handle:
            evaluation = json.load(handle)
        with open(baseline_path, "r", encoding="utf-8") as handle:
            baseline = json.load(handle)
        with open(stats_path, "r", encoding="utf-8") as handle:
            stats = json.load(handle)
        with open(validation_path, "r", encoding="utf-8") as handle:
            validation = json.load(handle)
        branches.append(
            {
                "branch": branch,
                "evaluation": evaluation,
                "baseline": baseline,
                "stats": stats,
                "validation": validation,
            }
        )

    final_evaluation = {
        "experiment": config.get("dataset_name"),
        "mode": "dual_branch",
        "best_branch": cross_eval.get("best_branch"),
        "best_accuracy": cross_eval.get("best_accuracy"),
        "branches": {
            item["branch"]: {
                "dataset_name": item["evaluation"].get("dataset_name"),
                "accuracy": item["evaluation"].get("accuracy"),
                "baseline_accuracy": item["baseline"].get("accuracy"),
                "test_samples": item["evaluation"].get("test_samples"),
            }
            for item in branches
        },
    }

    for path in (
        args.report,
        args.results,
        args.paper_tables,
        args.provenance,
        args.final_evaluation,
    ):
        ensure_parent(path)

    with open(args.paper_tables, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["branch", "dataset", "accuracy", "baseline_accuracy", "test_samples"],
        )
        writer.writeheader()
        for item in branches:
            writer.writerow(
                {
                    "branch": item["branch"],
                    "dataset": item["evaluation"].get("dataset_name"),
                    "accuracy": item["evaluation"].get("accuracy"),
                    "baseline_accuracy": item["baseline"].get("accuracy"),
                    "test_samples": item["evaluation"].get("test_samples"),
                }
            )

    with open(args.report, "w", encoding="utf-8") as handle:
        handle.write(f"# {config.get('dataset_name', 'dual-branch experiment')}\n\n")
        handle.write(f"Best branch: `{cross_eval.get('best_branch')}`\n\n")
        for item in branches:
            handle.write(f"## {item['branch']}\n")
            handle.write(f"- Dataset: {item['evaluation'].get('dataset_name')}\n")
            handle.write(f"- Accuracy: {item['evaluation'].get('accuracy')}\n")
            handle.write(f"- Baseline accuracy: {item['baseline'].get('accuracy')}\n")
            handle.write(f"- Test samples: {item['evaluation'].get('test_samples')}\n")
            handle.write(f"- Clients: {item['stats'].get('num_clients')}\n\n")

    provenance = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "experiment": config.get("dataset_name"),
        "rounds": config.get("rounds"),
        "branch_ids": args.branch,
        "cross_eval": args.cross_eval,
        "plot_summary": args.plot_summary,
        "round_metrics": args.round_metric,
    }
    with open(args.provenance, "w", encoding="utf-8") as handle:
        json.dump(provenance, handle, indent=2, sort_keys=True)

    with open(args.final_evaluation, "w", encoding="utf-8") as handle:
        json.dump(final_evaluation, handle, indent=2, sort_keys=True)

    with tarfile.open(args.results, "w:gz") as archive:
        archive.add(args.report, arcname=os.path.basename(args.report))
        archive.add(args.paper_tables, arcname=os.path.basename(args.paper_tables))
        archive.add(args.provenance, arcname=os.path.basename(args.provenance))
        archive.add(args.final_evaluation, arcname=os.path.basename(args.final_evaluation))
        archive.add(args.cross_eval, arcname=os.path.basename(args.cross_eval))
        archive.add(args.plot_summary, arcname=os.path.basename(args.plot_summary))
        for path in args.evaluation + args.baseline + args.stats + args.validation + args.round_metric:
            archive.add(path, arcname=os.path.join("artifacts", os.path.basename(path)))

    print(f"Generated report for {len(branches)} branches")


if __name__ == "__main__":
    main()
