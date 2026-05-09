#!/usr/bin/env python3

"""Create dataset, client, and split manifests for an FL experiment."""

import argparse
import csv
import hashlib
import json
import os
import random
import sys
import tarfile
from glob import glob
from datetime import datetime, timezone
from pathlib import Path

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


def synthetic_rows(config):
    rng = random.Random(int(config.get("split_seed", 13)))
    labels = config.get("labels", ["positive"])
    rows = []
    samples_per_client = int(config.get("samples_per_client", 16))
    for client_id in range(int(config["num_clients"])):
        for idx in range(samples_per_client):
            patient_id = f"synthetic-c{client_id:03d}-p{idx:04d}"
            label = labels[(idx + client_id) % len(labels)]
            if rng.random() < float(config.get("no_finding_rate", 0.25)):
                label = "No Finding"
            rows.append(
                {
                    "sample_id": f"{patient_id}-study",
                    "patient_id": patient_id,
                    "client_id": str(client_id),
                    "split": "unassigned",
                    "image_path": f"synthetic/{patient_id}.dat",
                    "labels": label,
                    "dataset": config["dataset_name"],
                }
            )
    return rows


def merge_pipeline_config(base_config, pipeline):
    merged = dict(base_config)
    merged.update(pipeline)
    if "dataset_name" not in merged:
        merged["dataset_name"] = pipeline.get("name", base_config.get("dataset_name", "dataset"))
    return merged


SUPPORTED_IMAGE_SUFFIXES = (
    ".dcm",
    ".nii.gz",
    ".nii",
    ".mha",
    ".mhd",
    ".png",
    ".jpg",
    ".jpeg",
    ".tif",
    ".tiff",
    ".bmp",
)

IMAGE_STAGE_PREFIX = "inputs"


def build_path_index(data_root):
    index = {}
    for path in glob(os.path.join(data_root, "**", "*"), recursive=True):
        if os.path.isfile(path):
            index.setdefault(os.path.basename(path), path)
    return index


def build_archive_index(archive_path):
    index = {}
    with tarfile.open(archive_path, "r:*") as handle:
        for member in handle.getmembers():
            if member.isfile():
                index.setdefault(os.path.basename(member.name), member.name)
    return index


def is_supported_image_path(path):
    lower = path.lower()
    return any(lower.endswith(suffix) for suffix in SUPPORTED_IMAGE_SUFFIXES)


def staged_image_lfn(sample_id, image_path):
    safe_sample_id = str(sample_id).replace(os.sep, "__")
    lower = image_path.lower()
    for suffix in SUPPORTED_IMAGE_SUFFIXES:
        if lower.endswith(suffix):
            return f"{IMAGE_STAGE_PREFIX}/{safe_sample_id}{suffix}"
    return f"{IMAGE_STAGE_PREFIX}/{safe_sample_id}{Path(image_path).suffix or '.dat'}"


def collection_name_for_path(path, data_root):
    if not data_root or not os.path.isabs(path) or not os.path.exists(path):
        rel_path = Path(path)
        return rel_path.parts[0] if len(rel_path.parts) > 1 else "default"
    rel_path = Path(os.path.relpath(path, data_root))
    return rel_path.parts[0] if len(rel_path.parts) > 1 else "default"


def label_for_discovered_path(path, data_root, config):
    collection = collection_name_for_path(path, data_root)
    label_map = config.get("directory_label_map", {})
    if collection in label_map:
        return label_map[collection]
    if config.get("label_from_top_level_dir", False):
        return collection
    return config.get("default_label", "unknown")


def patient_id_for_discovered_path(path, data_root, config):
    if not data_root or not os.path.isabs(path) or not os.path.exists(path):
        rel_path = Path(path)
    else:
        rel_path = Path(os.path.relpath(path, data_root))
    strategy = config.get("patient_id_from", "filename_stem")
    if strategy == "parent_dir" and rel_path.parent.name not in ("", "."):
        return rel_path.parent.name
    if strategy == "top_level_dir":
        return collection_name_for_path(path, data_root)
    if strategy == "full_relative_path":
        return str(rel_path.with_suffix("")).replace(os.sep, "__")
    return rel_path.stem


def discover_rows_from_directory(config):
    data_root = config.get("data_root")
    archive_path = config.get("_runtime_archive") or config.get("dataset_archive")
    if not data_root:
        raise ValueError("data_root must be set when auto_discover_images is enabled")
    if not os.path.isdir(data_root) and not archive_path:
        raise FileNotFoundError(data_root)

    rows = []
    if os.path.isdir(data_root):
        for path in sorted(glob(os.path.join(data_root, "**", "*"), recursive=True)):
            if not os.path.isfile(path) or not is_supported_image_path(path):
                continue
            rel_path = os.path.relpath(path, data_root)
            patient_id = patient_id_for_discovered_path(path, data_root, config)
            sample_id = str(Path(rel_path).with_suffix("")).replace(os.sep, "__")
            rows.append(
                {
                    "sample_id": sample_id,
                    "patient_id": patient_id,
                    "client_id": "",
                    "split": "unassigned",
                    "image_path": path,
                    "source_path": path,
                    "labels": label_for_discovered_path(path, data_root, config),
                    "dataset": config["dataset_name"],
                    "collection": collection_name_for_path(path, data_root),
                }
            )
        return rows

    with tarfile.open(archive_path, "r:*") as handle:
        for member in sorted(handle.getmembers(), key=lambda item: item.name):
            if not member.isfile() or not is_supported_image_path(member.name):
                continue
            rel_path = member.name
            patient_id = patient_id_for_discovered_path(rel_path, data_root, config)
            sample_id = str(Path(rel_path).with_suffix("")).replace(os.sep, "__")
            rows.append(
                {
                    "sample_id": sample_id,
                    "patient_id": patient_id,
                    "client_id": "",
                    "split": "unassigned",
                    "image_path": member.name,
                    "source_path": member.name,
                    "labels": label_for_discovered_path(member.name, data_root, config),
                    "dataset": config["dataset_name"],
                    "collection": collection_name_for_path(member.name, data_root),
                }
            )
    return rows


def assign_client_ids(rows, config):
    num_clients = int(config["num_clients"])
    strategy = config.get("client_partition_strategy")
    client_col = config.get("client_id_column")

    if strategy == "top_level_directory":
        collections = sorted({row.get("collection", "default") for row in rows})
        collection_to_client = {
            collection: index % max(num_clients, 1)
            for index, collection in enumerate(collections)
        }
        for row in rows:
            row["client_id"] = str(collection_to_client[row.get("collection", "default")])
        return

    for row in rows:
        if client_col and row.get(client_col) not in (None, ""):
            row["client_id"] = str(int(row[client_col]) % num_clients)
            continue
        digest = hashlib.sha256(str(row["patient_id"]).encode("utf-8")).hexdigest()
        row["client_id"] = str(int(digest[:8], 16) % num_clients)


def resolve_image_path(row, config, path_index=None):
    archive_path = config.get("_runtime_archive") or config.get("dataset_archive")
    archive_index = config.get("_archive_index")
    direct_keys = [
        config.get("image_path_column"),
        "image_path",
        "Image Path",
        "filepath",
        "path",
    ]
    for key in direct_keys:
        if key and row.get(key):
            value = row[key]
            if archive_path and archive_index is not None and not os.path.isabs(value):
                return archive_index.get(os.path.basename(value), value)
            return value if os.path.isabs(value) else os.path.join(config.get("data_root", "."), value)

    sample_id = row.get(config.get("sample_id_column", "Image Index")) or row.get("Image Index") or row.get("sample_id")
    if sample_id and path_index is not None:
        found = path_index.get(sample_id) or path_index.get(os.path.basename(sample_id))
        if found:
            return found
    if sample_id and archive_path and archive_index is not None:
        found = archive_index.get(sample_id) or archive_index.get(os.path.basename(sample_id))
        if found:
            return found
    return os.path.join(config.get("data_root", "."), sample_id or "unknown")


def load_metadata_rows_single(config):
    metadata_files = config.get("_runtime_metadata_files") or config.get("metadata_files", [])
    archive_path = config.get("_runtime_archive") or config.get("dataset_archive")
    if not metadata_files:
        if config.get("auto_discover_images", False):
            rows = discover_rows_from_directory(config)
            assign_client_ids(rows, config)
            return rows
        if config.get("allow_synthetic_fallback", False):
            return synthetic_rows(config)
        raise ValueError("metadata_files must be set when synthetic fallback is disabled")

    path = metadata_files[0]
    if not os.path.exists(path):
        basename_candidate = os.path.basename(path)
        if os.path.exists(basename_candidate):
            path = basename_candidate
        else:
            print(f"Metadata file not found: {path}", file=sys.stderr)
            if config.get("allow_synthetic_fallback", False):
                return synthetic_rows(config)
            raise FileNotFoundError(path)

    rows = []
    data_root = config.get("data_root", ".")
    path_index = build_path_index(data_root) if os.path.isdir(data_root) else None
    archive_index = build_archive_index(archive_path) if archive_path else None
    config["_archive_index"] = archive_index
    patient_col = config.get("patient_id_column")
    sample_col = config.get("sample_id_column")
    label_col = config.get("label_column")
    with open(path, newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for idx, row in enumerate(reader):
            patient_id = (
                row.get(patient_col)
                or row.get("Patient ID")
                or row.get("patient_id")
                or row.get("Subject ID")
                or row.get("subject_id")
                or f"patient-{idx}"
            )
            sample_id = row.get(sample_col) or row.get("Image Index") or row.get("sample_id") or f"sample-{idx}"
            labels = row.get(label_col) or row.get("Finding Labels") or row.get("labels") or "unknown"
            image_path = resolve_image_path(row, config, path_index)
            rows.append(
                {
                    "sample_id": sample_id,
                    "patient_id": str(patient_id),
                    "client_id": row.get(config.get("client_id_column", ""), ""),
                    "split": "unassigned",
                    "image_path": image_path,
                    "source_path": image_path,
                    "labels": labels,
                    "dataset": config["dataset_name"],
                    "collection": collection_name_for_path(image_path, data_root)
                    if os.path.commonpath([os.path.abspath(image_path), os.path.abspath(data_root)])
                    == os.path.abspath(data_root)
                    else "default",
                }
            )
    assign_client_ids(rows, config)
    return rows


def load_metadata_rows(config):
    pipelines = config.get("dataset_pipelines", [])
    if not pipelines:
        return load_metadata_rows_single(config)

    rows = []
    client_offset = 0
    for index, pipeline in enumerate(pipelines):
        pipeline_config = merge_pipeline_config(config, pipeline)
        if "num_clients" not in pipeline_config:
            raise ValueError(f"dataset_pipelines[{index}] must define num_clients")
        pipeline_rows = load_metadata_rows_single(pipeline_config)
        pipeline_id = pipeline.get("pipeline_id", pipeline.get("name", f"pipeline_{index:02d}"))
        for row in pipeline_rows:
            row["client_id"] = str(int(row["client_id"]) + client_offset)
            row["pipeline_id"] = pipeline_id
            row["dataset"] = pipeline_config["dataset_name"]
        rows.extend(pipeline_rows)
        client_offset += int(pipeline_config["num_clients"])
    return rows


def assign_splits(rows, seed, train_fraction, val_fraction):
    by_patient = {}
    for row in rows:
        by_patient.setdefault(row["patient_id"], []).append(row)

    patients = sorted(by_patient)
    random.Random(seed).shuffle(patients)
    train_cut = int(len(patients) * train_fraction)
    val_cut = train_cut + int(len(patients) * val_fraction)
    assignments = {}
    for index, patient_id in enumerate(patients):
        if index < train_cut:
            split = "train"
        elif index < val_cut:
            split = "val"
        else:
            split = "test"
        assignments[patient_id] = split

    for row in rows:
        row["split"] = assignments[row["patient_id"]]
    return assignments


def output_rows(rows):
    keys = {
        "sample_id",
        "patient_id",
        "client_id",
        "split",
        "image_path",
        "labels",
        "dataset",
        "pipeline_id",
    }
    return [{key: row.get(key, "") for key in keys} for row in rows]


def apply_staged_image_paths(rows, config):
    if not config.get("stage_input_data", False) or config.get("_runtime_archive") or config.get("dataset_archive"):
        return

    for row in rows:
        source_path = row.get("source_path", row["image_path"])
        row["image_path"] = staged_image_lfn(row["sample_id"], source_path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--archive")
    parser.add_argument("--metadata", action="append", default=[])
    parser.add_argument("--dataset-manifest", required=True)
    parser.add_argument("--client-manifest", required=True)
    parser.add_argument("--splits", required=True)
    args = parser.parse_args()

    config = load_config(args.config)
    if args.archive:
        config["_runtime_archive"] = args.archive
    if args.metadata:
        config["_runtime_metadata_files"] = args.metadata
    rows = load_metadata_rows(config)
    apply_staged_image_paths(rows, config)
    assignments = assign_splits(
        rows,
        int(config.get("split_seed", 13)),
        float(config.get("train_fraction", 0.7)),
        float(config.get("val_fraction", 0.1)),
    )

    for path in (args.dataset_manifest, args.client_manifest, args.splits):
        ensure_parent(path)

    fieldnames = [
        "sample_id",
        "patient_id",
        "client_id",
        "split",
        "image_path",
        "labels",
        "dataset",
        "pipeline_id",
    ]
    with open(args.client_manifest, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(output_rows(rows))

    with open(args.dataset_manifest, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["key", "value"])
        writer.writeheader()
        writer.writerow({"key": "dataset_name", "value": config["dataset_name"]})
        writer.writerow({"key": "generated_at", "value": datetime.now(timezone.utc).isoformat()})
        writer.writerow({"key": "num_samples", "value": len(rows)})
        writer.writerow({"key": "num_patients", "value": len(assignments)})
        if config.get("dataset_pipelines"):
            num_clients = sum(int(pipeline["num_clients"]) for pipeline in config["dataset_pipelines"])
        else:
            num_clients = config["num_clients"]
        writer.writerow({"key": "num_clients", "value": num_clients})

    with open(args.splits, "w", encoding="utf-8") as handle:
        json.dump({"patient_splits": assignments}, handle, indent=2, sort_keys=True)

    print(f"Wrote {len(rows)} rows to {args.client_manifest}")


if __name__ == "__main__":
    main()
