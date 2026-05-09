#!/usr/bin/env python3

"""Prepare a dataset branch by validating paths and optionally extracting an archive."""

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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--archive")
    parser.add_argument("--metadata", action="append", default=[])
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    config = load_config(args.config)
    metadata_files = args.metadata or config.get("metadata_files", [])
    data_root = config.get("data_root")
    archive_path = args.archive or config.get("dataset_archive")
    data_root_exists = bool(data_root and os.path.isdir(data_root))
    archive_exists = bool(archive_path and os.path.exists(archive_path))
    if not data_root and not archive_path:
        raise ValueError("either data_root or dataset_archive must be set in the branch config")
    if not data_root_exists and not archive_exists:
        raise FileNotFoundError(data_root or archive_path)

    output = {
        "dataset_name": config.get("dataset_name"),
        "data_root": data_root,
        "data_root_exists": data_root_exists,
        "dataset_archive": archive_path,
        "archive_exists": archive_exists,
        "metadata_files": metadata_files,
        "metadata_exists": {path: os.path.exists(path) for path in metadata_files},
        "auto_discover_images": bool(config.get("auto_discover_images", False)),
        "stage_input_data": bool(config.get("stage_input_data", False)),
        "num_clients": int(config.get("num_clients", 0)),
        "rounds": int(config.get("rounds", 0)),
    }

    ensure_parent(args.output)
    with open(args.output, "w", encoding="utf-8") as handle:
        json.dump(output, handle, indent=2, sort_keys=True)
    print(f"Prepared dataset {output['dataset_name']}")


if __name__ == "__main__":
    main()
