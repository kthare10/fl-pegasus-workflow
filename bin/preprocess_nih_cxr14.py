#!/usr/bin/env python3

"""Preprocess one NIH ChestX-ray14 federated client into JSONL image records."""

import argparse
import csv
import json
import os
import tarfile
from io import BytesIO

import numpy as np

try:
    from PIL import Image
except ImportError:  # pragma: no cover
    Image = None

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


def load_image_array(path, image_size, archive_path=None):
    if Image is None:
        raise RuntimeError("Pillow is required to preprocess NIH image files")
    if archive_path:
        with tarfile.open(archive_path, "r:*") as archive:
            member = archive.extractfile(path)
            if member is None:
                raise FileNotFoundError(path)
            with Image.open(BytesIO(member.read())) as img:
                img = img.convert("L")
                img = img.resize((image_size, image_size))
                return np.asarray(img, dtype=np.float32) / 255.0
    with Image.open(path) as img:
        img = img.convert("L")
        img = img.resize((image_size, image_size))
        return np.asarray(img, dtype=np.float32) / 255.0


def image_record_from_image(path, image_size, archive_path=None):
    image = load_image_array(path, image_size, archive_path=archive_path)
    return image.tolist()


def label_vector(raw_labels, labels):
    parts = {part.strip() for part in raw_labels.replace("|", ";").split(";")}
    return [1 if label in parts else 0 for label in labels]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--client-manifest", required=True)
    parser.add_argument("--client-id", required=True, type=int)
    parser.add_argument("--archive")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    config = load_config(args.config)
    labels = config.get("labels", ["Atelectasis", "Cardiomegaly", "Effusion", "No Finding"])
    image_size = int(config.get("image_size", 224))
    ensure_parent(args.output)

    count = 0
    with open(args.client_manifest, newline="", encoding="utf-8") as manifest, open(
        args.output, "w", encoding="utf-8"
    ) as out:
        for row in csv.DictReader(manifest):
            if int(row["client_id"]) != args.client_id:
                continue
            record = {
                "sample_id": row["sample_id"],
                "patient_id": row["patient_id"],
                "split": row["split"],
                "image": image_record_from_image(row["image_path"], image_size, archive_path=args.archive),
                "y": label_vector(row["labels"], labels),
            }
            out.write(json.dumps(record, sort_keys=True) + "\n")
            count += 1

    print(f"Wrote {count} NIH records for client {args.client_id} to {args.output}")


if __name__ == "__main__":
    main()
